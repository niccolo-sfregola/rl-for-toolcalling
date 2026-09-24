"""messages = [system, user(question)]
for turn in range(MAX_TURNS):
    prompt = apply_chat_template(messages, tools=[...], add_generation_prompt=True)
    output = model.generate(prompt)           # stops at <|im_end|>
    text   = decode(only the new tokens)
    call   = parse_tool_call(text)
    if call is None:                          # no <tool_call>: final answer
        break
    result = lookup(call["entity"], call["relation"], facts_dict)
    messages += [assistant(tool_calls=[call]), tool(result)]
reward = 1 if final answer == gold, else 0
"""

import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from dataset_generator import relations


def load_model():
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda")
    model.eval()

    return tok, model


lookup_dict = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description":"Look up the entity linked to the given entity by the given relation in the knowledge graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "entity": {"type":"string"},
                "relation": {"type":"string", "enum": relations}
            },
            "required": ["entity","relation"],
        },
    },
}



def parse_tool_call(text: str) -> dict | None:
    """Extract {"name": ..., "arguments": {...}} from the first <tool_call> in the text,
    or None if there is none or it is malformed."""

    #look for the <tool_call> pattern in the text
    match = re.search(r"<tool_call>(.*?)</tool_call>",text,re.DOTALL)
    if match:
        tool = match.group(1).strip()
    else:
        return None

    try:
        tool_call_dict = json.loads(tool)
    except json.JSONDecodeError:
        return None  

    #the JSON structure must be correct:
    #  it must be a dict, it must have name == lookup,
    #  it must have arguments as a dict,
    #  with entity and relation inside

    if not isinstance(tool_call_dict,dict):
        return None

    if tool_call_dict.get("name") != "lookup":
        return None

    args = tool_call_dict.get("arguments")
    if not isinstance(args, dict):
        return None

    if not isinstance(args.get("entity"), str) or not isinstance(args.get("relation"), str):
        return None

    
    return tool_call_dict


def parse_answer(text: str) -> str | None:
    """Extract the content of the first <answer>...</answer>, normalized
    (outer whitespace stripped, lowercased), or None if missing or empty."""

    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match is None:
        return None

    answer = match.group(1).strip().lower()
    if not answer:
        return None

    return answer


