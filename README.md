# RL for tool calling

> **Work in progress.** The environment, agent loop and GRPO training loop are in place, and training runs are in progress. The design is still changing.

Post-training a small language model (**Qwen2.5-0.5B-Instruct**) with reinforcement learning to answer **multi-hop questions by calling a tool**, one lookup at a time, over a synthetic knowledge graph.

The model is never told the intermediate entities. To answer *"Who is the ruler of the neighbor of Fervia?"* it has to look up the neighbor of Fervia first, read the result, and then look up the ruler of that entity. The reward is fully verifiable: the graph is known, so the gold answer is too.

## Why a synthetic environment

- **Verifiable reward.** Every question comes with its graph and its gold answer, so the reward is exact and does not depend on a judge model (RLVR, RL from verifiable rewards).
- **No memorization.** The states and their relations are fictional and every question has its own random graph, so the model cannot answer from pretraining knowledge. It has to use the tool.
- **Controlled difficulty.** The number of hops is a parameter, which makes a curriculum easy: one hop first, then two.

## How it works

### Environment ([dataset_generator.py](dataset_generator.py))
- 25 fictional states and 4 relations: `ally_of`, `enemy_of`, `governed_by`, `borders`.
- For each question, a random graph of 40 facts and an n-hop chain sampled from it. The question is built from the first entity and the relations, read from right to left: `Who is the ally of the ruler of Aldoria?`
- A `lookup(entity, relation)` tool. Invalid calls never crash the episode: they return informative error messages (unknown entity, unknown relation with the list of valid ones, or no fact found).

### Agent loop ([agent.py](agent.py))
- Uses Qwen2.5's native tool-calling format: the tool schema goes through `apply_chat_template`, the model emits `<tool_call>{...}</tool_call>` and the tool result comes back inside `<tool_response>`.
- Generation stops at `</tool_call>`, so the model makes one call per turn and must wait for the result.
- A turn without a valid tool call is the final answer, expected as `<answer>ENTITY</answer>`.
- **Token-exact trajectories.** The chat template is applied only to the first prompt. After that the sequence grows by concatenating the sampled tokens and the tool-response tokens, and a `completion_mask` marks which tokens the model generated. Re-rendering the history at every turn would re-serialize the tool calls, and the model would be trained on tokens it never sampled.

### Reward
| final turn | correct entity | wrong entity |
|---|---|---|
| `<answer>...</answer>` | 1.0 | 0 |
| plain sentence | 0.5 | 0 |

The partial credit for a correct answer without tags gives the policy some signal while the base model almost never uses the answer tags.

### Training ([train_grpo.py](train_grpo.py))
GRPO with two of the changes introduced by DAPO:
- **Group-relative advantages.** G = 8 sampled episodes per question, advantage `(r - mean) / (std + eps)` within the group, no value network.
- **Dynamic sampling.** Groups where all rewards are equal have zero advantage and give no gradient. Questions are sampled until the batch holds a fixed number of groups *with* signal.
- **Token-level loss.** One mean over all model tokens in the batch, so long episodes (retries, multiple turns) weigh in proportion to the tokens they generated.
- Clip-higher and overlong reward shaping are left out: with one optimizer step per batch the importance ratio is always 1, and episodes are short.

Implementation details:
- float32 master weights with bfloat16 autocast. Updates of the size of the learning rate are smaller than the bfloat16 resolution of a typical weight and would otherwise be rounded away.
- Gradients are accumulated one episode at a time, so only one computation graph is in memory.
- Resumable checkpoints (weights, optimizer state and RNG states).

### Evaluation ([evaluate.py](evaluate.py))
A fixed eval set (separate seed from training), with greedy and sampled episodes. Metrics: answer format, correctness, reward, number of calls, successful lookups, whether all hops were completed, retries after an error, `pass@8`, and the fraction of groups that would give a GRPO gradient.

## Results

Training runs on one-hop questions are in progress. Results will be added here once they are available.

## Roadmap
- [x] Synthetic graph environment and `lookup` tool
- [x] Multi-turn agent loop with token-exact trajectories
- [x] Baseline evaluation
- [x] GRPO training loop with dynamic sampling and token-level loss
- [ ] Training runs on one-hop questions (in progress)
- [ ] Format bonus for tagged answers grounded in a lookup result, if the answer tags still do not appear
- [ ] Two-hop questions (curriculum)
- [ ] Per-error-type and per-relation metrics

## Usage

Requires a CUDA GPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python try_model.py    # play and print one episode
python evaluate.py     # baseline on the fixed eval set
python train_grpo.py   # GRPO training
```

Hyperparameters are constants at the top of each script (`N_HOPS`, `G`, `LR`, `N_STEPS`, `RESUME_FROM`, ...).

## Repository layout
| file | content |
|---|---|
| [dataset_generator.py](dataset_generator.py) | graph, n-hop questions, `lookup` tool |
| [agent.py](agent.py) | system prompt, tool schema, parsers, episode loop, reward |
| [evaluate.py](evaluate.py) | fixed eval set and metrics |
| [train_grpo.py](train_grpo.py) | GRPO / DAPO training loop |
| [try_model.py](try_model.py) | one episode, printed turn by turn, with a token-level check |
| [verify_tokenizer.py](verify_tokenizer.py) | how Qwen2.5's chat template renders tools and tool calls |
