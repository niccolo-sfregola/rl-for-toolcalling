import os
import random as rnd

import torch

from dataset_generator import (build_questions, generate_random_graph,
                               entities, relations, relation_phrases, sample_chain)
from agent import load_model, run_episode, compute_reward, lookup_dict
from evaluate import build_eval_set, episode_stats, summarize, EVAL_SEED


TRAIN_SEED = 0            #different from EVAL_SEED
N_HOPS = 1                #curriculum
MAX_TURNS = 6

#DAPO, only two of its four changes to GRPO:
#  - dynamic sampling: keep sampling questions until the batch holds N_GROUPS_PER_STEP
#    groups with signal, so that every update has G * N_GROUPS_PER_STEP useful episodes
#  - token-level loss: one mean over all the model tokens of the batch (grpo_loss)
#  clip-higher is left out: one optimizer step per batch of rollouts, the ratio is always 1
#  and the clip never triggers. Overlong shaping too: episodes are short, and the ones
#  that reach MAX_TURNS already get 0
G = 8                     #episodes per question (group size)
N_GROUPS_PER_STEP = 4     #groups with signal per optimizer step
MAX_QUESTIONS_PER_STEP = 32   #cap on the questions sampled in one step, against endless loops
N_STEPS = 200
LR = 1e-6
EPS = 1e-4                #in the advantage denominator, against std close to 0

EVAL_EVERY = 25           #steps between two evaluations
N_EVAL_QUESTIONS = 30
CHECKPOINT_DIR = "checkpoints/fp32_run"
SAVE_EVERY = 10           #steps between two saves of the full training state
RESUME_FROM = None        #path of a train_state.pt to resume from, None to start from scratch



def sample_train_question(n_hops) -> dict:
    """One training question with its own graph.
    Uses the global rnd state: the seed is fixed once in main."""
                
    _, facts_dict = generate_random_graph(40,entities,relations)
    path, rels = sample_chain(relations, facts_dict, n_hops)
    question = build_questions(path[0], rels, relation_phrases)

    return {
        "facts_dict": facts_dict,
        "question": question,
        "gold": path[-1],
        "path": path,
        "rels": rels,
        "n_hops": n_hops,
    }


def rollout_group(tok, model, item, g) -> list:
    """Play g sampled episodes on the same question.
    Returns a list of g dicts with at least messages, answer and reward."""

    episodes = []
    for i in range(g):
        answer, _, messages, token_ids, completion_mask = run_episode(tok, model, facts_dict=item["facts_dict"] ,question=item["question"] ,max_turns=MAX_TURNS, DO_SAMPLE=True)
        reward = compute_reward(messages=messages, answer=answer, gold=item["gold"],h0= item["path"][0])
        episodes.append({
            "messages": messages,
            "answer": answer,
            "reward": reward,
            "token_ids": token_ids,
            "completion_mask": completion_mask,
        })

    return episodes


def group_advantages(rewards) -> torch.Tensor:
    """GRPO advantages of one group: (r - mean) / (std + EPS).
    The baseline is the mean of the group itself"""
    rewards_tensor = torch.tensor(rewards)

    return (rewards_tensor -rewards_tensor.mean() ) /(rewards_tensor.std() + EPS)

def build_training_tensors(episodes):
    """Token ids of a whole episode and a mask with 1 only on the tokens
    the model generated (assistant turns), 0 on system, user and tool tokens.
    Returns (input_ids, completion_mask), both of shape (1, seq_len)."""

    input_ids = episodes["token_ids"].unsqueeze(0).to("cuda")
    completion_mask = episodes["completion_mask"].unsqueeze(0).to("cuda")

    return input_ids, completion_mask



def token_logprobs(model, input_ids):
    """Log-prob of every token given the previous ones, shape (1, seq_len - 1):
    the logits at position t predict the token at t + 1."""

    logits = model(input_ids).logits #(1, seq_length, vocab)

    #the last logit predicts a token that does not exist, the first token is predicted by nobody
    target_tok = input_ids[:, 1:]    #(1, seq_len - 1)

    logprobs_all = torch.log_softmax(logits[:, :-1], dim=-1)   #(1, seq_len - 1, vocab)

    #keep only the log-prob of the token that was actually there:
    return torch.gather(logprobs_all, dim=-1, index=target_tok.unsqueeze(-1)).squeeze(-1)



def grpo_loss(logprobs, completion_mask, advantage):
    """Policy gradient loss of one episode: - advantage * SUM of the log-probs
    of the model tokens. The advantage is a constant, it gets no gradient.
    Token-level loss (DAPO): the caller divides by the model tokens of the whole batch,
    so every token weighs the same, whatever the length of its episode."""

    #same shift as in token_logprobs: logprobs[:, i] is the log-prob of token i + 1
    #float32, not bfloat16: bfloat16 cannot count exactly beyond 256 tokens
    mask = completion_mask[:, 1:].float()   #(1, seq_len - 1)

    #model tokens only: prompt and tool tokens are multiplied by 0
    sum_logprob = (logprobs.float() * mask).sum()

    #minus: the optimizer minimizes, we want to raise the log-probs when advantage > 0
    return -advantage * sum_logprob


def n_model_tokens(episode) -> int:
    """Tokens of the episode that enter the loss, with the same shift as grpo_loss."""
    return int(episode["completion_mask"][1:].sum())


def save_train_state(path, step, model, optimizer, eval_history):
    """Everything needed to resume exactly from this step: weights, AdamW state,
    the random states (questions come from rnd, token sampling from torch) and the evals.
    Written to a temporary file and then renamed: a crash while saving
    does not destroy the previous state."""

    tmp = path + ".tmp"
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rnd_state": rnd.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
        "eval_history": eval_history,
    }, tmp)
    os.replace(tmp, path)


def evaluate(tok, model, eval_set):
    """Greedy episodes on the fixed eval set, prints the summary of evaluate.py.
    Returns the mean of every metric, to follow it across training."""

    all_stats = []
    for item in eval_set:
        answer, _, messages, _, _ = run_episode(
            tok, model, item["facts_dict"], item["question"],
            max_turns=MAX_TURNS, DO_SAMPLE=False,
        )
        all_stats.append(episode_stats(messages, answer, item["gold"],
                                       item["path"][0], item["n_hops"]))

    summarize(all_stats, f"EVAL, greedy, {N_HOPS} hop(s)")

    return {key: sum(s[key] for s in all_stats) / len(all_stats) for key in all_stats[0]}



if __name__ == "__main__":
    tok, model = load_model()

    #float32 weights: in bfloat16 an update of ~LR is smaller than the gap between two
    #  representable values of a typical weight (~1e-4 around 0.02), and gets rounded away.
    #  The computations still run in bfloat16 through autocast, so speed barely changes
    model.float()

    eval_set = build_eval_set(N_EVAL_QUESTIONS, EVAL_SEED, N_HOPS)

    #build_eval_set reseeds rnd: the training seed is set after it
    rnd.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    state_path = f"{CHECKPOINT_DIR}/train_state.pt"

    if RESUME_FROM is not None:
        state = torch.load(RESUME_FROM, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rnd.setstate(state["rnd_state"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        eval_history = state["eval_history"]
        start_step = state["step"] + 1
        del state
        print(f"resumed from {RESUME_FROM}, step {start_step}")
    else:
        #baseline of this run, before any update: the reference for every later eval
        with torch.autocast("cuda", dtype=torch.bfloat16):
            eval_history = [(0, evaluate(tok, model, eval_set))]
        start_step = 1

    for step in range(start_step, N_STEPS + 1):

        #rollouts with dynamic sampling: one question at a time, until the batch holds
        #  N_GROUPS_PER_STEP groups with signal (all rewards equal = every advantage is 0)
        all_episodes = []   #every sampled episode, kept or not, for the log
        kept = []           #(episode, advantage) pairs that give gradient
        n_questions = 0
        n_kept_groups = 0
        while n_kept_groups < N_GROUPS_PER_STEP and n_questions < MAX_QUESTIONS_PER_STEP:
            item = sample_train_question(N_HOPS)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                episodes = rollout_group(tok, model, item, G)
            n_questions += 1
            all_episodes += episodes

            rewards = [ep["reward"] for ep in episodes]
            if len(set(rewards)) == 1:
                continue
            advantages = group_advantages(rewards)
            kept += list(zip(episodes, advantages))
            n_kept_groups += 1

        #loss over the kept episodes, backward, optimizer step
        #token-level mean: every backward is divided by the model tokens of the WHOLE batch,
        #  so the sum of the backwards is the gradient of the mean over all those tokens.
        #  One backward per episode: only one computation graph at a time stays in memory
        n_tokens = sum(n_model_tokens(episode) for episode, _ in kept)
        optimizer.zero_grad()
        total_loss = 0.0
        for episode, advantage in kept:
            input_ids, completion_mask = build_training_tensors(episode)
            #only the forward pass under autocast: backward follows the dtypes it chose
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logprobs = token_logprobs(model, input_ids)
            loss = grpo_loss(logprobs, completion_mask, advantage) / n_tokens
            loss.backward()
            total_loss += loss.item()

        if kept:
            #a single bad batch must not move the weights too far
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        #log: mean reward and has_answer over every sampled episode,
        #  groups with signal over the questions sampled to fill the batch
        mean_reward = sum(ep["reward"] for ep in all_episodes) / len(all_episodes)
        has_answer = sum(ep["answer"] is not None for ep in all_episodes) / len(all_episodes)
        with_signal = n_kept_groups / n_questions
        print(f"step {step:>4} | reward {mean_reward:.2f} | has_answer {has_answer:.2f} | "
              f"groups with signal {n_kept_groups}/{n_questions} ({with_signal:.2f}) | "
              f"tokens {n_tokens} | loss {total_loss:+.4f}")

        #every EVAL_EVERY steps: evaluate and save the weights in bfloat16 (half the disk),
        #  enough to load them for inference with from_pretrained
        if step % EVAL_EVERY == 0:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                eval_history.append((step, evaluate(tok, model, eval_set)))
            bf16_weights = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
            model.save_pretrained(f"{CHECKPOINT_DIR}/step_{step}", state_dict=bf16_weights)
            tok.save_pretrained(f"{CHECKPOINT_DIR}/step_{step}")
            del bf16_weights

        #every SAVE_EVERY steps: full training state, overwritten, to resume after a crash
        if step % SAVE_EVERY == 0:
            save_train_state(state_path, step, model, optimizer, eval_history)

    print("\nEVAL HISTORY (step, correct, reward, has_answer)")
    for step, means in eval_history:
        print(f"  {step:>4}  {means['correct']:.2f}  {means['reward']:.2f}  {means['has_answer']:.2f}")
