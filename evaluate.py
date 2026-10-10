from dataset_generator import (build_questions, generate_random_graph,
                               entities, relations,relation_phrases, sample_chain)
import random as rnd

import torch

from agent import load_model, run_episode, compute_reward, TEMPERATURE


N_QUESTIONS = 30
EVAL_SEED = 999   #different from any training seed: the model never trains on these questions
MAX_TURNS = 6
N_HOPS = 1        #1 for one-hop questions, 2 for two-hop ones
G = 8             #episodes per question in the sampling loop (the GRPO group size)
RUN_GREEDY = True #set to False to skip the greedy loop and only run the sampling one





def build_eval_set(n, seed, n_hops) -> list:

    rnd.seed(seed)

    list_dictionaries = []

    for i in range(n):

        
        _, facts_dict = generate_random_graph(40,entities,relations)
        path, rels = sample_chain(relations, facts_dict, n_hops)
        question = build_questions(path[0], rels, relation_phrases)

        list_dictionaries.append({
            "facts_dict": facts_dict,
            "question": question,
            "gold": path[-1],
            "path": path,   #[h0, h1, ..., gold]: the intermediate entities too
            "rels": rels,
            "n_hops": n_hops,
        })

    return list_dictionaries


def episode_stats(messages, answer, gold, h0, n_hops) -> dict:

    saw_error = False
    has_answer = answer is not None #if there's the answer
    correct = answer == gold.lower() # if the answer is correct
    reward = compute_reward(messages, answer, gold, h0) # correct also without tags: partial credit
    n_calls = 0 # of total tool calls
    n_ok_lookups = 0 # of successful lookups
    retried = False # after mistake, agent called tool

    for msg in messages:
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            n_calls += 1 # of total tool calls
        elif msg["role"] == "tool":
            if saw_error:
                retried = True
            if msg["content"].startswith("Error"):
                saw_error = True
            else:
                n_ok_lookups += 1
    
    all_hops = n_ok_lookups >= n_hops #at least one successful lookup per hop

    return {"has_answer":has_answer,
            "correct": correct,
            "reward": reward,
            "n_calls": n_calls,
            "n_ok_lookups": n_ok_lookups,
            "all_hops": all_hops,
            "retried": retried
            }


def summarize(all_stats, title):
    """Print the mean of every metric over a list of episode_stats dicts."""

    #True counts as 1: the mean of a boolean is the fraction of True,
    #  the mean of a count (n_calls, n_ok_lookups) is the average per episode
    print(f"\n{title} ({len(all_stats)} episodes)")
    for key in all_stats[0]:
        mean = sum(s[key] for s in all_stats) / len(all_stats)
        print(f"  {key:<14} {mean:.2f}")


if __name__ == "__main__":
    tok, model = load_model()
    eval_set = build_eval_set(N_QUESTIONS, EVAL_SEED, N_HOPS)

    #3. greedy loop: one episode per question
    if RUN_GREEDY:
        all_stats = []
        failures = []   #trajectories of wrong episodes

        for i, item in enumerate(eval_set):
            answer, turns_used, messages, _, _ = run_episode(
                tok, model, item["facts_dict"], item["question"],
                max_turns=MAX_TURNS, DO_SAMPLE=False,
            )
            stats = episode_stats(messages, answer, item["gold"], item["path"][0], item["n_hops"])
            all_stats.append(stats)
            if not stats["correct"]:
                failures.append((item, messages))

            print(f"{i + 1}/{len(eval_set)} correct={stats['correct']} "
                  f"n_calls={stats['n_calls']} n_ok_lookups={stats['n_ok_lookups']} "
                  f"retried={stats['retried']}")

        summarize(all_stats, f"GREEDY BASELINE, {N_HOPS} hop(s)")

    #4. sampling loop: G episodes per question, like one GRPO group
    #rnd.seed only covers the graphs: token sampling uses the torch generator
    torch.manual_seed(EVAL_SEED)

    sampled_stats = []    #all N_QUESTIONS * G episodes, for the averages
    n_correct_per_q = []  #how many of the G episodes were correct, per question
    rewards_per_q = []    #the G rewards of each question, as GRPO would see them

    for i, item in enumerate(eval_set):
        n_correct = 0
        rewards = []
        for g in range(G):
            answer, turns_used, messages, _, _ = run_episode(
                tok, model, item["facts_dict"], item["question"],
                max_turns=MAX_TURNS, DO_SAMPLE=True,
            )
            stats = episode_stats(messages, answer, item["gold"], item["path"][0], item["n_hops"])
            sampled_stats.append(stats)
            n_correct += stats["correct"]
            rewards.append(stats["reward"])

        n_correct_per_q.append(n_correct)
        rewards_per_q.append(rewards)
        print(f"{i + 1}/{len(eval_set)} correct {n_correct}/{G} rewards {rewards}")

    summarize(sampled_stats, f"SAMPLING, T={TEMPERATURE}, {N_HOPS} hop(s)")

    #pass@G: a question is solved if at least one of its G episodes is correct (strict, with tags)
    pass_at_g = sum(k > 0 for k in n_correct_per_q) / len(n_correct_per_q)
    #a GRPO group gives gradient only if rewards differ inside it:
    #  all equal (all 0, all PARTIAL_CREDIT, all 1) -> every advantage is 0
    with_signal = sum(len(set(r)) > 1 for r in rewards_per_q) / len(rewards_per_q)

    print(f"  pass@{G:<9} {pass_at_g:.2f}")
    print(f"  groups with signal {with_signal:.2f}")
