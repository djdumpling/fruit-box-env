# End-to-end Environments-to-evals

In progress! We develop a complete RL pipeline by developing an environment, for a grid-based reasoning game, using Prime Intellect's verifiers library, benchmarking LLMs, and (currently) training RL agents to play.

This blog is roughly structured chronologically. First, writing scripted policies to generate gameplay data and validate the core environment mechanics. Second, developing the environment and integrating with Prime Intellect's verifiers library to create an LLM-facing eval framework. Third, benchmarking said LLMs. Finally, training RL agents (SFT and GRPO) to learn optimal play strategies.

## Fruit Box and intuition

Fruit Box is a grid-based puzzle game played on a 10×17 board where each cell contains a digit from 1-9. The objective is to clear as many cells as possible by selecting rectangular regions that sum to exactly 10. When a valid rectangle is chosen, those cells are cleared and the player earns points equal to the number of non-zero cells cleared in that move.

![Fruit box](/public/fruit_box_updated.png)

The game is quite fun. A few friends had introduced me to the game maybe a month ago, and I quickly got hooked. I didn't think higher scores were possible without faster scanning, but simple games often have 'cool tech'. One example that comes to mind is the 2048 game, where speedrun strategys always put the biggest tile in a corner (e.g. bottom-right) then button-mash arrows (e.g. down and right) to quickly accumulate large tables. After a few rounds, there were a few intuitions and motivate the intuition behind the scripted policies:

- 1's must go with 9's. Therefore, 1's should almost always be saved for a neighboring 9. Similarly, 2's *usually* go with 8's; in some occasions, it makes sense to select 1—1—8 over 2—8.
- A corrollary of #1 is that the greedy algorithm is bad. It will try to find a rectangle using as many digits as possible, which increases the likelihood of using 1's or 2's. In fact, we find that the opposite of greedy (a minimal approach) is *very* effective.
- There's more "low-level tech" like patterns that are clearable only a certain way. For example, clearing the middle 9—1 instead of the first-column 1—9 because in the later case, the leftover 1 and 9 can't be cleared.
<div style="text-align: center;">
<img src="/public/fruit_box_tech.png" alt="Fruit box" style="max-width: 100px; height: auto;">
</div>

- As a human (?), I find it helpful to search for the next rectangle while clearing the current one. This probably saves ~0.2 seconds per rectangle, but across the ~50 rectangles cleared, this saves a (non-trivial) 10 seconds.
- This is person-specific, but systematic search (like starting in the top-left and working row-by-row while checking columns) is algorithmically faster due to less redundancy with overlapping cells, but it helped just to let my eyes naturally wander, clearing nearby rectangles.

## Scripting Policies

With this in mind, there are 4 policies that are of interest

* Random: choose a legal rectangle randomly. This serves as a sort of "baseline."
* Greedy: choose the legal rectangle that maximizes cells cleared.
* 2-step lookahead: choose the legal rectangle that maximizes a $Q$ value with discount factor.
* Minimal: choose the legal rectangle that minimizes cells cleared.

We assume ties are broken arbitrarily. Also, there's a nuance to grid generation: each cell is distributed multinomially across the 9 digits, but the total sum across all 170 cells must be a multiple of 10. This is to allow a perfect clearing [ADD LINK], many grid configurations may be impossible to perfect, e.g. if the number of 9's is greater than the number of 1's. I'm not sure if there's a way to sequentially generate such a constrained sequence of 170 digits, so we continuously sample sequences (10 expected) until the sum is a multiple of 10. 

We generate 1000 seeds and bench the four policies:

![Policy Comparisons](/public/policy_comparisons.png)

<div style="text-align: center; margin-bottom: 20px;">

<table style="margin: 0 auto; width: auto; max-width: 350px;">
<thead>
<tr>
<th style="width: 150px; text-align: left;">Policy</th>
<th style="width: 80px; text-align: center;">Mean</th>
<th style="width: 80px; text-align: center;">Std</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">Greedy</td>
<td style="text-align: center;">97.61</td>
<td style="text-align: center;">10.53</td>
</tr>
<tr>
<td style="text-align: left;">Random</td>
<td style="text-align: center;">102.89</td>
<td style="text-align: center;">12.04</td>
</tr>
<tr>
<td style="text-align: left;">2-step Lookahead</td>
<td style="text-align: center;">96.22</td>
<td style="text-align: center;">10.05</td>
</tr>
<tr>
<td style="text-align: left;">Minimal</td>
<td style="text-align: center;">113.72</td>
<td style="text-align: center;">14.89</td>
</tr>
</tbody>
</table>

</div>

All four policies had roughly the same minimum score, but the minimal area policy had 5 runs which scores over 150, achieving a maximum of 157. These results roughly match the inuition that minimal is better than greedy and that 2-step lookahead doesn't really affect much. In the process of benching the four policies, we note all trajectories, (roughly) summarized below:

```Python
{
  episode_id: string,
  step: int,
  grid: Arr[10][17],
  action: {"c1": int, "c2": int, "r1": int, "r2": int},
  num_legal_actions: int,
  reward: int,
  done: bool,
  agent_tag: string,
}
```

## Creating the Environment

With validated policies and trajectory data in hand, we're now ready to build the environment using Prime Intellect's verifiers library. The verifiers framework structures environment creation into six key components: data, interaction style, environment logic, rewards function, parser, and packaging. The actual implementation is 550 lines long, so we provide an abbreviated version, keeping the core functionality of each component.

### 1. Data

We load episodes from HuggingFace datasets and format them into the verifiers' expected structure such that each example contains an initial prompt with the game rules and grid state, along with ground truth trajectory information.

```python
GAME_RULES = textwrap.dedent("""...""")

def build_dataset() -> Dataset:
    hf_dataset = load_dataset(dataset_name, split = dataset_split)
    
    # Group trajectories by episode_id and agent_tag
    episodes = {}
    for row in hf_dataset:
        ep_id = row["episode_id"]
        agent_tag = row.get("agent_tag", "unknown")
        key = f"{ep_id}_{agent_tag}"
        if key not in episodes:
            episodes[key] = []
        episodes[key].append(row)
    
    # Build examples with initial prompt and ground truth
    data = []
    for key, trajectory in episodes.items():
        initial_state = trajectory[0]
        initial_grid = initial_state["grid"]
        total_reward = sum(step.get("reward", 0) for step in trajectory)
        
        grid_json = json.dumps({"grid": initial_grid})
        initial_prompt = f"{GAME_RULES}\n## Initial Grid State\n{grid_json}\n What move do you make?"
        
        data.append({
            "prompt": [{"role": "user", "content": initial_prompt}],
            "answer": json.dumps({"trajectory": ground_truth_actions, ...}),
            "info": {"initial_grid": initial_grid, "total_reward": total_reward, ...}
        })
    
    return Dataset.from_list(data)
```

The dataset serves two key purposes during evaluation. First, it provides test cases across different initial grid states. Second, it supplies ground truth for normalization: the `"info"` field stores the expert's `total_reward` on that initial grid, which the rubric (covered later) uses to normalize LLM performance scores. As an analogy, take the AIME25 environment: the initial grid states is to the problem statement as the ground truth action is to the answer.

### 2. Interaction Style

The verifiers library provides a lovely multi-turn interaction between the environment and the LLM to exchange messages. We inherit from `MultiTurnEnv` and implement the `env_response` method, which processes the LLM's JSON-formatted move and returns the updated game state.

```python
class FruitBoxEnv(MultiTurnEnv):
    """Multi-turn environment for the Fruit Box puzzle game."""
    
    async def env_response(self, messages: Messages, state: State, **kwargs) -> Tuple[Messages, State]:
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        turn_num = len(assistant_messages)
        
        # Parse last assistant message to extract action
        last_content = assistant_messages[-1]["content"]
        parsed = json.loads(last_content)
        action = parsed.get("action", {})
        r1, c1, r2, c2 = action.get("r1"), action.get("c1"), action.get("r2"), action.get("c2")
        
        # Execute move on game environment
        current_grid = state.get("current_grid", state["info"]["initial_grid"])
        env = Sum10Env()
        env.reset(grid=np.array(current_grid))
        step_info = env.step(r1, c1, r2, c2)
        
        # Update state and return response
        state["current_grid"] = env.grid.tolist()
        response = {
            "valid": step_info.valid,
            "reward": step_info.reward,
            "done": step_info.done,
            "grid": env.grid.tolist()
        }
        return [{"role": "user", "content": json.dumps(response)}], state
```

### 3. Environment Logic

The core game logic lives in `Sum10Env`, which manages grid state, validates moves, and computes rewards. It uses prefix sums for efficient rectangle sum queries (allows O(1) rectangle queries across all $\binom{10+1}{2}\cdot\binom{17+1}{2}=8415$ combinations). The step function validates coordinates, checks that the sum equals 10, clears cells, and determines game termination.

```python
class Sum10Env:
    """Game environment for managing the grid state and move validation."""
    
    def __init__(self):
        self.grid = np.zeros((10, 17), dtype=np.uint8)
        self.turn = 0
        self.sum = None  # Prefix sum for fast rectangle queries
        self.count = None  # Prefix sum for non-zero cell counts
    
    def rebuild_prefix_sums(self):
        self.sum = self.grid.astype(np.int32).cumsum(axis=0).cumsum(axis=1)
        non_zero = (self.grid > 0).astype(np.int32)
        self.count = non_zero.cumsum(axis=0).cumsum(axis=1)
    
    def step(self, r1, c1, r2, c2) -> StepInfo:
        # Normalize coordinates
        if r1 > r2: r1, r2 = r2, r1
        if c1 > c2: c1, c2 = c2, c1
        
        # Validate bounds and sum
        if not (0 <= r1 <= r2 < 10 and 0 <= c1 <= c2 < 17):
            return StepInfo(valid=False, sum=0, reward=0, done=True)
        
        s = self.box_sum(r1, c1, r2, c2)
        reward = self.box_nonzero_count(r1, c1, r2, c2)
        
        if s != 10 or reward == 0:
            return StepInfo(valid=False, sum=s, reward=0, done=False)
        
        # Clear cells and update state
        self.grid[r1:r2+1, c1:c2+1] = 0
        self.rebuild_prefix_sums()
        self.turn += 1
        done = not self.has_any_legal()
        
        return StepInfo(valid=True, sum=10, reward=reward, done=done)
```

### 4. Rewards Function (Rubric)

The rubric defines how we evaluate LLM performance. Our `reward_total_score` function replays the LLM's trajectory and computes the total score, normalized by expert (minimal-area policy) performance. This provides a score between 0.0 and 1.0, where 1.0 indicates matching or exceeding the expert's performance.

```python
def reward_total_score(completion: List[dict], state: dict, **kwargs) -> float:
    """Reward function that measures total score normalized by expert performance."""
    initial_grid = state["info"]["initial_grid"]
    env = Sum10Env()
    env.reset(grid=np.array(initial_grid))
    
    total_reward = 0
    assistant_messages = [m for m in completion if m["role"] == "assistant"]
    
    for msg in assistant_messages:
        action = parse_action(msg["content"])
        if action is None:
            continue
        
        step_info = env.step(action.get("r1"), action.get("c1"), 
                            action.get("r2"), action.get("c2"))
        
        if step_info.valid:
            total_reward += step_info.reward
        else:
            break  # Invalid move ends trajectory
        
        if step_info.done:
            break
    
    # Normalize by expert performance
    expert_reward = state["info"]["total_reward"]
    return min(1.0, total_reward / expert_reward) if expert_reward > 0 else 0.0
```

Our rubric is solely dependent on the normalized trajectory, but we could expand the rubric by adding other factors like penalizing excessive turns, invalid moves, etc. 

### 5. Parser

The parser extracts structured actions from LLM responses. A lightweight regex to find JSON objects within the response suffices.

```python
def parse_action(content: str) -> Optional[Dict]:
    """Parse action from model response JSON."""
    try:
        # Try direct JSON parse first
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Extract JSON from text using regex
        import re
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            return None
    
    action = parsed.get("action", {})
    if all(k in action for k in ["r1", "c1", "r2", "c2"]):
        # Check for "no valid moves" signal
        if action.get("r1") == -1 and action.get("c1") == -1:
            return None
        return action
    return None
```

### 6. Package Environment

The `load_environment` function ties all components together, creating a complete verifiers environment ready for evaluation or training. It combines the dataset, environment class, and rubric into a single `vf.Environment` object.

```python
def load_environment(
    dataset_name: str = "djdumpling/fruit-box-minimal-area",
    dataset_split: str = "train",
    max_turns: int = 85,
    seed: Optional[int] = None,
) -> vf.Environment:
    """Load the Fruit Box environment with dataset and rubric."""
    
    # Build dataset from HuggingFace
    dataset = build_dataset()
    
    # Create rubric with reward function
    rubric = vf.Rubric(funcs=[reward_total_score], weights=[1.0])
    
    # Instantiate environment
    env_instance = FruitBoxEnv(
        max_turns=max_turns,
        dataset=dataset,
        rubric=rubric,
    )
    
    return env_instance
```

## LLM Benchmarking

We're now ready to benchmark different LLMs. We first tested on a suite of non-reasoning models (primarily gpt-4.1), which had a mean of ~20 and a maximum of 44. There are two main reasons for the performance:

1. The non-reasoning models sometimes fail to output the required JSON format, leading to parsing issues that ended runs earlier. This is possibly a symptom of a non-rigorous prompt, and adding more explicit instruction-following helped sufficiently
2. They roughly follow a systematic search, sweeping top-left to bottom-right and ending and always choosing the first legal rectangle it sees. Many moves are only possible after clearing neighboring cells, and non-reasoning models had neglected this.

To remedy this problem, we ask the LLM to also generate other candidate rectangles and append it to the prompt in the next turn. This improved both the mean and maximum, which reached ~70.

Reasoning models do much better at this puzzle game. Primarily benched on o3 and gpt5, they consistently achieved over 110, reaching a peak of 135. CoT was hidden, but through the logs, we can get a small glimpse into their decision-making. 2 late-stage turns show that the LLM is very careful, checking multiple grid values and verifying the sum is indeed 10.

```
=== Grid Visualization ===
 1  7  4  9  7  2  2  4  7  7  7  0  0  0  0  8  8
 0  0  9  7  8  0  0  0  0  6  0  0  0  9  7  7  1
 2  5  6  8  8  9  0  0  7  6  1  0  0  7  1  0  3
 9  3  0  6  6  9  0  0  2  9  0  0  0  0  0  0  4
 7  5  0  0  0  4  2  9  2  0  0  0  0  0  0  0  4
 0  0  0  0  0  8  0  0  0  0  0  0  0  0  0  0  1
 0  0  1  0  0  0  0  0  3  0  0  0  0  6  7  0  8
 0  0  0  0  0  0  0  8  0  0  0  0  0  0  0  0  5
 0  0  8  0  0  5  0  3  3  0  0  0  0  2  5  7  0
 0  0  3  5  0  0  0  0  0  6  0  9  0  6  0  0  0


=== Turn 43 ===
Action: (2,10) -> (9,11)
Reasoning: Identified a vertical rectangle from rows 2 to 9 and columns 10 to 11. Values: [(2,10)=1, (2,11)=0, (3,10)=0, (3,11)=0, (4,10)=0, (4,11)=0, (5,10)=0, (5,11)=0, (6,10)=0, (6,11)=0, (7,10)=0, (7,11)=0, (8,10)=0, (8,11)=0, (9,10)=0, (9,11)=9]. Sum = 1 + 9 = 10.
Candidate moves found: 1
  1. (2,10) -> (9,11)
Actual sum: 10
✓ Valid! Cleared 2 cells. Total: 103

=== Turn 44 ===
Action: (3,8) -> (8,8)
Reasoning: Examined column 8 for a zero-assisted vertical rectangle. From rows 3 to 8, the values are [2,2,0,3,0,3], which sum to exactly 10.
Candidate moves found: 2
  1. (3,8) -> (8,8)
  2. (6,2) -> (6,13)
Actual sum: 10
✓ Valid! Cleared 4 cells. Total: 107

=== Grid Visualization ===
 1  7  4  9  7  2  2  4  7  7  7  0  0  0  0  8  8
 0  0  9  7  8  0  0  0  0  6  0  0  0  9  7  7  1
 2  5  6  8  8  9  0  0  7  6  0  0  0  7  1  0  3
 9  3  0  6  6  9  0  0  0  9  0  0  0  0  0  0  4
 7  5  0  0  0  4  2  9  0  0  0  0  0  0  0  0  4
 0  0  0  0  0  8  0  0  0  0  0  0  0  0  0  0  1
 0  0  1  0  0  0  0  0  0  0  0  0  0  6  7  0  8
 0  0  0  0  0  0  0  8  0  0  0  0  0  0  0  0  5
 0  0  8  0  0  5  0  3  0  0  0  0  0  2  5  7  0
 0  0  3  5  0  0  0  0  0  6  0  0  0  6  0  0  0
```

Just one run took ~1.2 hours, which is far beyond the 2-minute time limit, but it's unfair to impose time constraints on LLMs since the objective is purely to benchmark LLM reasoning on games. Perhaps as inference time accelerates, it makes more sense to begin considering time constraints, or adding a criteria to the rubric that penalizes long thinking times.

Thankfully, Prime Intellect has some great parallelization tools, 5 examples (initial grids) and 3 rollouts per took a little over an hour instead of what would be 18 hours sequentially. The 5 initial grids seemed to be above the median, because even with consistent 110+ scores, the average score was still ony 0.835. For example, seed 393 was included, which the minimal area policy achieved a score of 153 on. Still, consistent 110+ scores is roughly (maybe a little lower than) what expert human would achieve.

## SFT and GRPO

In progress, coming soon!