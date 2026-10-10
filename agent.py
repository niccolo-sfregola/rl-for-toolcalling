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

from dataset_generator import entities, relations, lookup



SYSTEM_PROMPT = (
    "You answer questions about a knowledge graph of fictional states. "
    "You do not know any fact of this graph: you must use the lookup tool, "
    "one call at a time, and use the result of each call to decide the next one.\n"
    "Questions are nested and must be resolved from the inside out: "
    "for 'Who is the A of the B of X?', first look up the B of X, "
    "then look up the A of that result.\n"
    "If a tool call returns an error, fix the arguments and call the tool again. Never ask the user for help: nobody will answer.\n"
    "When the last lookup gives you the final answer, do not write a sentence: "
    "reply only with the entity name inside answer tags, exactly like this:\n"
    "<answer>ENTITY_NAME</answer>\n"
    "A final reply without <answer></answer> tags counts as wrong."
)

#appended to every tool response: in the rendered prompt the system prompt is followed
#  by the long # Tools block, and the tool response arrives as a user turn, so the model
#  answers it in prose unless the rule is right there. Putting it in the user turn with the
#  question made the model answer at once, without any lookup. It never says that the
#  result is the final answer, so in two-hop questions it does not push to stop at hop 1
RESULT_REMINDER = ("(Call the tool again if you need another lookup, "
                   "otherwise reply with the final entity inside <answer></answer> tags.)")
ERROR_REMINDER = "(Fix the arguments and call the tool again.)"

#reward for a correct final reply written as a sentence, without <answer> tags
PARTIAL_CREDIT = 0.5

#sampling temperature for DO_SAMPLE=True: 1.0 as in training, the distribution is not sharpened
TEMPERATURE = 1.0



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


def execute_tool_call(call, facts_dict) -> str:

    args = call.get("arguments")

    try:
       return lookup(entity= args.get("entity"), relation= args.get("relation"), facts_dict = facts_dict)   
    except (ValueError, LookupError) as e:
        return str(e)

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


def tool_turn_tokens(tok, messages):
    """Token ids that the chat template puts between the end of the last assistant
    generation and the start of the next one: <|im_end|> closing the assistant turn,
    the tool response wrapped as a user turn, and the new assistant header.
    messages must end with the assistant tool call and its tool response."""

    #render up to the assistant header of the tool call turn, then the whole history:
    #  the difference is the re-serialized tool call (skipped, the model generated its own
    #  version) followed by everything up to the next generation prompt
    before = tok.apply_chat_template(messages[:-2], tools=[lookup_dict],
                                     add_generation_prompt=True, tokenize=False)
    full = tok.apply_chat_template(messages, tools=[lookup_dict],
                                   add_generation_prompt=True, tokenize=False)
    rest = full[len(before):]
    suffix = rest[rest.index("<|im_end|>"):]

    return tok(suffix, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


def run_episode(tok, model, facts_dict, question, max_turns, DO_SAMPLE):
    """Run one multi-turn episode on a question.
    Returns (answer, turns_used, messages, token_ids, completion_mask):
    answer is the normalized content of <answer>, or None if the model never gave one;
    token_ids is the whole episode exactly as the model saw and generated it (1D, on cpu),
    completion_mask has 1 on the generated tokens and 0 on prompt and tool tokens."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    answer = None
    turns_used = 0

    #the chat template is rendered only once, for the first prompt. After that the
    #  sequence grows with the generated tokens and the tool tokens: re-rendering messages
    #  would re-serialize the tool calls, and the model would be trained on tokens
    #  different from the ones it sampled
    ids = tok.apply_chat_template(
        messages,
        tools=[lookup_dict],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )["input_ids"].to("cuda")
    mask = [0] * ids.shape[1]

    for turn in range(max_turns):
        turns_used = turn + 1

        with torch.no_grad():
            out = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=256,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE if DO_SAMPLE else None,
                pad_token_id=tok.eos_token_id,
                stop_strings=["</tool_call>"],
                tokenizer=tok,
            )

        new_tokens = out[0, ids.shape[1]:]
        ids = out
        mask += [1] * len(new_tokens)
        text = tok.decode(new_tokens, skip_special_tokens=True)

        call_dict = parse_tool_call(text)

        #no valid tool call (including a malformed one): this turn is the final answer
        if call_dict is None:
            answer = parse_answer(text)
            messages.append({"role": "assistant", "content": text})
            break

        result = execute_tool_call(call_dict, facts_dict)
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": call_dict}],
        })
        #lookup errors start with "Error:" (episode_stats relies on it, the reminder goes after)
        reminder = ERROR_REMINDER if result.startswith("Error") else RESULT_REMINDER
        messages.append({"role": "tool", "content": f"{result}\n{reminder}"})

        tool_ids = tool_turn_tokens(tok, messages).to("cuda")
        ids = torch.cat([ids, tool_ids.unsqueeze(0)], dim=1)
        mask += [0] * len(tool_ids)

    return answer, turns_used, messages, ids[0].cpu(), torch.tensor(mask)


def compute_reward(messages, answer, gold, h0) -> float:
    """1 for <answer>gold</answer>, PARTIAL_CREDIT for a final sentence without tags
    that names the gold and no other entity except h0 (the one in the question), 0 otherwise.
    A wrong answer always gets 0."""

    if answer is not None:
        return 1.0 if answer == gold.lower() else 0.0

    #the episode ended on max_turns: the last message is a tool response, no final reply
    last = messages[-1]
    if last["role"] != "assistant" or last.get("tool_calls"):
        return 0.0

    #whole words, case insensitive: "Galtron" matches "galtron." but not "Galtronia"
    text = last["content"].lower()
    mentioned = {e for e in entities if re.search(rf"\b{e.lower()}\b", text)}
    others = mentioned - {gold, h0}

    #"The ally of Uldria is Galtron." -> credit; listing several candidates -> 0
    if gold in mentioned and not others:
        return PARTIAL_CREDIT
    return 0.0
