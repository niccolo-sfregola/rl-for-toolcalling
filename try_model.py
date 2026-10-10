"""
the goal is to see what Qwen2.5-0.5B-Instruct actually produces
when it is given the lookup tool and an n-hop question: a full multi-turn
episode, from the question to the final answer, with the whole trajectory printed.
"""
import random as rnd

from agent import load_model, run_episode, compute_reward
from dataset_generator import (
    entities, relations, relation_phrases,
    generate_random_graph, sample_chain, build_questions,
)


#greedy decoding: same prompt -> same output, easier to debug
DO_SAMPLE = True
MAX_TURNS = 4
N_HOPS = 2


if __name__ == "__main__":
    SEED = 77
    rnd.seed(SEED)

    tok, model = load_model()

    _, facts_dict = generate_random_graph(40, entities, relations)
    path, rels = sample_chain(relations, facts_dict, N_HOPS)
    question = build_questions(path[0], rels, relation_phrases)
    gold = path[-1]

    print("QUESTION:", question)
    print("CHAIN:   ", " -> ".join(f"{e} --{r}" for e, r in zip(path, rels)), "->", gold)

    answer, turns_used, messages, token_ids, completion_mask = run_episode(
        tok, model, facts_dict, question, max_turns=MAX_TURNS, DO_SAMPLE=DO_SAMPLE
    )

    #skip system and user: they are already printed above
    print("\nTRAJECTORY:")
    for msg in messages[2:]:
        if msg.get("tool_calls"):
            print(f"  [{msg['role']}] tool_call: {msg['tool_calls'][0]['function']['arguments']}")
        else:
            print(f"  [{msg['role']}] {msg['content']!r}")

    reward = compute_reward(messages, answer, gold, path[0])
    print("\nANSWER:", answer, "| GOLD:", gold.lower(), "| TURNS:", turns_used, "| REWARD:", reward)

    #token check: the sequence the model saw and generated, from its first generated token
    #  (the system prompt with the # Tools block is long and comes from the template as is)
    first_gen = completion_mask.nonzero()[0].item()
    print("\nTOKENS FROM THE FIRST GENERATION:")
    print(tok.decode(token_ids[first_gen:]))

    #only the tokens that get gradient: the model's own text, no prompt and no tool response
    print("\nMASKED TOKENS (completion_mask == 1):")
    print(tok.decode(token_ids[completion_mask == 1]))
