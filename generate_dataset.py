from __future__ import annotations
import argparse, json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np

# ---------- Utilities ----------

def write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, "w", encoding = "utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators = (",", ":"), ensure_ascii = False) + "\n")

def write_parquet(path: Path, rows: List[dict]):
    path.parent.mkdir(parents = True, exist_ok = True)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index = False)

# ---------- Environment ----------

# bounding box (r1, c1, r2, c2)
Box = Tuple[int, int, int, int]

# all information after a step
@dataclass
class StepInfo:
    valid: bool
    sum: int
    reward: int
    done: bool

class Sum10Env:
    def __init__(self, H = 10, W = 17, seed: Optional[int] = None):
        self.H = H
        self.W = W
        self.rng = np.random.default_rng(seed = seed)
        self.boxes = self.precompute_boxes(H = H, W = W)
        self.grid = np.zeros((H, W), dtype = np.uint8)
        self.turn = 0

        # prefix sum of the grid
        self.sum = None  # summed values
        self.count = None  # summed nonzero mask

    # ---------- Setup ----------

    def reset(self, grid: Optional[np.ndarray] = None):
        if grid is None:
            self.grid = self.sample_initial_grid()
        else:
            self.grid = grid.astype(np.uint8).copy()

        self.turn = 0
        self.rebuild_prefix_sums()
        return self.obs()

    def sample_initial_grid(self) -> np.ndarray:
        while True:
            g = self.rng.integers(1, 10, size = (self.H, self.W), dtype = np.uint8)
            if (int(g.sum()) % 10) == 0:
                return g

    # (11 choose 2) * (18 choose 2) = 8415 combinations
    @staticmethod
    def precompute_boxes(H, W) -> List[Box]:
        boxes = []
        for r1 in range(H):
            for r2 in range(r1, H):
                for c1 in range(W):
                    for c2 in range(c1, W):
                        boxes.append((r1, c1, r2, c2))
        return boxes

    # ---------- Observations ----------

    def obs(self) -> Dict: 
        return {
            "grid": self.grid.tolist(),
            "turn": self.turn
        }

    # ---------- Prefix sums & queries ----------

    def rebuild_prefix_sums(self):
        # prefix sum of values
        # use int32 to avoid overflow
        self.sum = self.grid.astype(np.int32).cumsum(axis = 0).cumsum(axis = 1)
        # count: prefix sum of non-zero mask
        non_zero = (self.grid > 0).astype(np.int32)
        self.count = non_zero.cumsum(axis = 0).cumsum(axis = 1)

    @staticmethod
    # PIE to find sum
    def box_query(grid, r1, c1, r2, c2):
        s = grid[r2, c2]
        if r1 > 0:
            s -= grid[r1 - 1, c2]
        if c1 > 0:
            s -= grid[r2, c1 - 1]
        if r1 > 0 and c1 > 0:
            s += grid[r1 - 1, c1 - 1]
        return int(s)

    def box_sum(self, r1, c1, r2, c2):
        return self.box_query(self.sum, r1, c1, r2, c2)

    def box_nonzero_count(self, r1, c1, r2, c2):
        return self.box_query(self.count, r1, c1, r2, c2)

    # ---------- Legality & enumeration ----------

    # return list of ((r1,c1,r2,c2), reward_nonzero_count) for all legal rectangles
    def enumerate_legal(self):
        out = []
        for r1, c1, r2, c2 in self.boxes:
            if self.box_sum(r1, c1, r2, c2) == 10:
                reward = self.box_nonzero_count(r1, c1, r2, c2)
                if reward > 0:
                    out.append(((r1, c1, r2, c2), reward))
        return out

    def has_any_legal(self):
        for r1, c1, r2, c2 in self.boxes:
            if self.box_sum(r1, c1, r2, c2) == 10 and self.box_nonzero_count(r1, c1, r2, c2) > 0:
                return True
        return False

    # ---------- Step ----------

    def step(self, r1, c1, r2, c2) -> StepInfo:
        # normalize coordinates
        if r1 > r2: r1, r2 = r2, r1
        if c1 > c2: c1, c2 = c2, c1

        # valid bounds check
        if not (0 <= r1 <= r2 < self.H and 0 <= c1 <= c2 < self.W):
            return StepInfo(valid = False, sum = -1, reward = 0, done = False)

        # sum = 10 check
        s = self.box_sum(r1, c1, r2, c2)
        if s != 10:
            return StepInfo(valid = False, sum = s, reward = 0, done = False)

        # positive reward check
        reward = self.box_nonzero_count(r1, c1, r2, c2)
        if reward == 0:
            return StepInfo(valid = False, sum = s, reward = 0, done = False)

        # otherwise, valid so zero the masked grid and rebuild
        self.grid[r1:r2 + 1, c1:c2 + 1] = 0
        self.rebuild_prefix_sums()
        self.turn += 1
        done = not self.has_any_legal()
        
        return StepInfo(valid = True, sum = 10, reward = reward, done = done)

# ---------- Policies ----------

def policy_random_legal(env: Sum10Env) -> Optional[Box]:
    choices = env.enumerate_legal()
    if not choices: return None
    idx = env.rng.integers(0, len(choices))
    return choices[idx][0]

# maximize the non-zero count (reward)
def policy_greedy_area(env: Sum10Env) -> Optional[Box]:
    choices = env.enumerate_legal()
    if not choices: return None
    box, _ = max(choices, key = lambda x: x[1])
    return box

# simulate one step and look at max immediate reward next state.
def policy_look_ahead(env: Sum10Env, lam: float = 0.2) -> Optional[Box]:
    choices = env.enumerate_legal()
    if not choices: return None
    best_score = -1
    best_box = None
    original_grid = env.grid.copy()
    original_sum = env.sum.copy()
    original_count = env.count.copy()
    original_turn = env.turn

    for box, reward_now in choices:
        (r1, c1, r2, c2) = box

        # simulate
        env.grid[r1:r2 + 1, c1:c2 + 1] = 0
        env.rebuild_prefix_sums()
        next = env.enumerate_legal()
        reward_next = max((reward for _, reward in next), default = 0)
        score = reward_now + lam * reward_next
        if score > best_score:
            best_score = score
            best_box = box
        
        # rollback
        env.grid[:] = original_grid
        env.sum[:] = original_sum
        env.count[:] = original_count
        env.turn = original_turn

    return best_box

# ---------- Generation Loop ----------

# returns (trajectory_rows, episode_header)
def generate_episode(seed, policy = "greedy_area", H = 10, W = 17) -> Tuple[List[dict], dict]:
    env = Sum10Env(H = H, W = W, seed = seed)
    env.reset()
    rows = []
    step = 0

    def select_action() -> Optional[Box]:
        if policy == "random_legal":
            return policy_random_legal(env)
        elif policy == "greedy_area":
            return policy_greedy_area(env)
        elif policy.startswith("look_ahead"):
            # default lambda is 0.2, otherwise parse command input
            lam = 0.2
            if ":" in policy:
                try:
                    lam = float(policy.split(":")[1])
                except: pass
            return policy_look_ahead(env, lam = lam)
        else:
            raise ValueError(f"unknown policy: {policy}")

    while env.has_any_legal():
        grid_before = env.grid.copy()
        box = select_action()
        if box is None:
            break
        r1, c1, r2, c2 = box

        # do step
        info = env.step(r1, c1, r2, c2)
        step += 1
        row = {
            "episode_id": f"seed{seed}",
            "step": step,
            "grid": grid_before.tolist(),     # pre-action state
            "action": {"r1": r1, "c1": c1, "r2": r2, "c2": c2},
            "legal": bool(info.valid),
            "sum": int(info.sum),
            "reward": int(info.reward),
            "done": bool(info.done),
            "agent_tag": policy,
            "rng_seed": int(seed)
        }
        rows.append(row)
        if info.done:
            break

    header = {
        "episode_id": f"seed{seed}",
        "seed": int(seed),
        "agent_tag": policy,
        "total_reward": int(sum(r["reward"] for r in rows)),
        "total_steps": int(len(rows))
    }
    return rows, header

# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser(description = "Generate Sum-10 trajectories.")
    p.add_argument("--episodes", type = int, default = 100, help = "number of episodes to generate")
    p.add_argument("--seed_start", type = int, default = 1, help = "first RNG seed")
    p.add_argument("--policy", type = str, default = "greedy_area",
                   choices = ["random_legal", "greedy_area", "look_ahead", "look_ahead:0.3", "look_ahead:0.0"],
                   help = "scripted policy")
    p.add_argument("--out_dir", type = str, default = "out_data", help = "output directory")
    p.add_argument("--format", type = str, default = "jsonl", choices = ["jsonl", "parquet"], help = "output format for trajectories")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    traj_rows: List[dict] = []
    episode_rows: List[dict] = []

    for i in range(args.episodes):
        seed = args.seed_start + i
        rows, header = generate_episode(seed = seed, policy = args.policy, H = 10, W = 17)
        traj_rows.extend(rows)
        episode_rows.append(header)
        if (i + 1) % 10 == 0:
            print(f"Generated {i+1}/{args.episodes} episodes")

    # trajectories format
    if args.format == "jsonl":
        write_jsonl(out_dir / "trajectories.jsonl", traj_rows)
        print(f"Wrote jsonl: {out_dir/'trajectories.jsonl'}")
    else:
        write_parquet(out_dir / "trajectories.parquet", traj_rows)

    # episodes header (summary of trajectories)
    write_jsonl(out_dir / "episodes.jsonl", episode_rows)
    print(f"Wrote episodes: {out_dir/'episodes.jsonl'}")

if __name__ == "__main__":
    main()