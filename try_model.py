"""
the goal is to see what Qwen2.5-0.5B-Instruct actually produces
when it is given the lookup tool and a two-hop question, and whether our
parsers can read it.
"""

import torch

from agent import load_model, lookup_dict, parse_tool_call, parse_answer
from dataset_generator import (
    entities, relations, relation_phrases,
    generate_random_graph, sample_chain, build_questions,
)


SYSTEM_PROMPT = (
    "You answer questions about a knowledge graph of fictional states. "
    "You do not know any fact of this graph: you must use the lookup tool, "
    "one call at a time, and use the result of each call to decide the next one. "
    "When you know the final answer, reply only with the entity name inside "
    "<answer></answer> tags, for example <answer>Aldoria</answer>."
)

#greedy decoding: same prompt -> same output, easier to debug

DO_SAMPLE = False


if __name__ == "__main__":
    tok, model = load_model()

    _, facts_dict = generate_random_graph(40, entities, relations)
    h0, r1, r2, gold = sample_chain(relations, facts_dict)
    question = build_questions(h0, r1, r2, relation_phrases)

    print("QUESTION:", question)
    print("CHAIN:   ", (h0, r1), "->", facts_dict[(h0, r1)], "|", r2, "->", gold)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    #return_dict=True gives both input_ids and attention_mask
    inputs = tok.apply_chat_template(
        messages,
        tools=[lookup_dict],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=DO_SAMPLE,
            temperature=0.7 if DO_SAMPLE else None,
            pad_token_id=tok.eos_token_id,
        )

    #out contains prompt + new tokens: keep only the new ones
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    text = tok.decode(new_tokens, skip_special_tokens=True)

    print("\nRAW OUTPUT:")
    print(repr(text))
    print("\nparse_tool_call:", parse_tool_call(text))
    print("parse_answer:   ", parse_answer(text))
