from __future__ import annotations
import argparse, json, math, os, sys, gzip
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pandas as pd

import numpy as np

# -----------------------------
# Utilities
# -----------------------------

def write_jsonl(path: Path, rows: List[dict], compress: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress or str(path).endswith(".gz"):
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n")

def maybe_write_parquet(path: Path, rows: List[dict]):
    """
    Optional Parquet writer (requires pandas + pyarrow). If unavailable, skip.
    """
    try:
        df = pd.DataFrame(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"[info] wrote parquet: {path}")
    except Exception as e:
        print(f"[warn] parquet not written ({e}); install pandas+pyarrow to enable.")


# -----------------------------
# Environment
# -----------------------------

# (r1, c1, r2, c2)
Rect = Tuple[int, int, int, int]

@dataclass
class StepInfo:
    valid: bool
    sum: int
    reward: int

class Sum10Env:
    def __init__(self, H: int = 10, W: int = 17, seed: Optional[int] = None):
        self.H, self.W = H, W
        self.rng = np.random.default_rng(seed)
        self.rects: List[Rect] = self._precompute_rects(H, W)
        self.grid: np.ndarray = np.zeros((H, W), dtype=np.uint8)
        # Integral images
        self.S: Optional[np.ndarray] = None  # summed values
        self.C: Optional[np.ndarray] = None  # summed nonzero mask
        self.turn: int = 0

    # ---------- Setup ----------

    def reset(self, grid: Optional[np.ndarray] = None):
        if grid is None:
            self.grid = self._sample_initial_grid()
        else:
            assert grid.shape == (self.H, self.W)
            self.grid = grid.astype(np.uint8).copy()
        self.turn = 0
        self._rebuild_integrals()
        return self.obs()

    def _sample_initial_grid(self) -> np.ndarray:
        """
        Sample 10x17 digits uniformly from 1...9 such that total sum % 10 == 0
        """
        while True:
            g = self.rng.integers(1, 10, size=(self.H, self.W), dtype=np.uint8)
            if (int(g.sum()) % 10) == 0:
                return g

    @staticmethod
    def _precompute_rects(H: int, W: int) -> List[Rect]:
        """
        (11 choose 2) * (18 choose 2) = 8415 combinations
        """
        rects: List[Rect] = []
        for r1 in range(H):
            for r2 in range(r1, H):
                for c1 in range(W):
                    for c2 in range(c1, W):
                        rects.append((r1, c1, r2, c2))
        return rects

    # ---------- Observations ----------

    def obs(self) -> Dict:
        return {
            "board_size": {"H": self.H, "W": self.W},
            "grid": self.grid.tolist(),
            "turn": self.turn
        }

    # ---------- Integral images & queries ----------

    def _rebuild_integrals(self):
        # S: integral of values (use int32 to avoid overflow)
        self.S = self.grid.astype(np.int32).cumsum(axis=0).cumsum(axis=1)
        # C: integral of non-zero mask
        nz = (self.grid > 0).astype(np.int32)
        self.C = nz.cumsum(axis=0).cumsum(axis=1)

    @staticmethod
    def _rect_query(I: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> int:
        s = I[r2, c2]
        if r1 > 0:
            s -= I[r1 - 1, c2]
        if c1 > 0:
            s -= I[r2, c1 - 1]
        if r1 > 0 and c1 > 0:
            s += I[r1 - 1, c1 - 1]
        return int(s)

    def rect_sum(self, r1: int, c1: int, r2: int, c2: int) -> int:
        return self._rect_query(self.S, r1, c1, r2, c2)

    def rect_nonzero_count(self, r1: int, c1: int, r2: int, c2: int) -> int:
        return self._rect_query(self.C, r1, c1, r2, c2)

    # ---------- Legality & enumeration ----------

    def enumerate_legal(self) -> List[Tuple[Rect, int]]:
        """
        Returns list of ((r1,c1,r2,c2), reward_nonzero_count) for all legal rectangles
        """
        out: List[Tuple[Rect, int]] = []
        for r1, c1, r2, c2 in self.rects:
            if self.rect_sum(r1, c1, r2, c2) == 10:
                reward = self.rect_nonzero_count(r1, c1, r2, c2)
                if reward > 0:
                    out.append(((r1, c1, r2, c2), reward))
        return out

    def has_any_legal(self) -> bool:
        for r1, c1, r2, c2 in self.rects:
            if self.rect_sum(r1, c1, r2, c2) == 10 and self.rect_nonzero_count(r1, c1, r2, c2) > 0:
                return True
        return False

    # ---------- Step ----------

    def step(self, r1: int, c1: int, r2: int, c2: int) -> Tuple[Dict, int, bool, StepInfo]:
        # Normalize coordinates
        if r1 > r2: r1, r2 = r2, r1
        if c1 > c2: c1, c2 = c2, c1
        # Bounds check
        if not (0 <= r1 <= r2 < self.H and 0 <= c1 <= c2 < self.W):
            info = StepInfo(valid=False, sum=-1, reward=0)
            return self.obs(), 0, False, info

        s = self.rect_sum(r1, c1, r2, c2)
        if s != 10:
            info = StepInfo(valid=False, sum=s, reward=0)
            return self.obs(), 0, False, info

        reward = self.rect_nonzero_count(r1, c1, r2, c2)
        if reward == 0:
            # Clearing zeros only is pointless but still legal if sum==10 (here means all zeros, impossible unless 0=10).
            info = StepInfo(valid=False, sum=s, reward=0)
            return self.obs(), 0, False, info

        # Apply: set to zero
        self.grid[r1:r2 + 1, c1:c2 + 1] = 0
        self._rebuild_integrals()
        self.turn += 1
        done = not self.has_any_legal()
        info = StepInfo(valid=True, sum=10, reward=reward)
        return self.obs(), reward, done, info


# -----------------------------
# Policies
# -----------------------------

def policy_random_legal(env: Sum10Env) -> Optional[Rect]:
    cand = env.enumerate_legal()
    if not cand: return None
    idx = env.rng.integers(0, len(cand))
    return cand[idx][0]

def policy_greedy_area(env: Sum10Env) -> Optional[Rect]:
    cand = env.enumerate_legal()
    if not cand: return None
    # maximize the non-zero count (reward)
    rect, _ = max(cand, key=lambda x: x[1])
    return rect

def policy_lookahead1(env: Sum10Env, lam: float = 0.2) -> Optional[Rect]:
    cand = env.enumerate_legal()
    if not cand: return None
    best_score = -1e9
    best_rect = None
    # For each action, simulate one step and look at max immediate reward next state.
    original_grid = env.grid.copy()
    original_S = env.S.copy()
    original_C = env.C.copy()
    original_turn = env.turn
    for rect, r_now in cand:
        (r1, c1, r2, c2) = rect
        # simulate
        env.grid[r1:r2 + 1, c1:c2 + 1] = 0
        env._rebuild_integrals()
        nxt = env.enumerate_legal()
        r_next = max((rew for _, rew in nxt), default=0)
        score = r_now + lam * r_next
        if score > best_score:
            best_score = score
            best_rect = rect
        # rollback
        env.grid[:] = original_grid
        env.S[:] = original_S
        env.C[:] = original_C
        env.turn = original_turn
    return best_rect


# -----------------------------
# Generation Loop
# -----------------------------

def generate_episode(
    seed: int,
    policy: str = "greedy_area",
    H: int = 10,
    W: int = 17,
) -> Tuple[List[dict], dict]:
    """
    Returns (trajectory_rows, episode_header).
    Each trajectory row is a dict suitable for JSONL/Parquet.
    """
    env = Sum10Env(H=H, W=W, seed=seed)
    obs = env.reset()
    rows: List[dict] = []

    def select_action() -> Optional[Rect]:
        if policy == "random_legal":
            return policy_random_legal(env)
        elif policy == "greedy_area":
            return policy_greedy_area(env)
        elif policy.startswith("lookahead1"):
            # parse lambda if provided as lookahead1:0.3
            lam = 0.2
            if ":" in policy:
                try:
                    lam = float(policy.split(":")[1])
                except: pass
            return policy_lookahead1(env, lam=lam)
        else:
            raise ValueError(f"unknown policy: {policy}")

    step = 0
    while env.has_any_legal():
        grid_before = env.grid.copy()
        rect = select_action()
        if rect is None:
            break
        r1, c1, r2, c2 = rect
        # do step
        next_obs, reward, done, info = env.step(r1, c1, r2, c2)
        step += 1
        row = {
            "episode_id": f"seed{seed}_H{H}W{W}",
            "step": step,
            "grid": grid_before.tolist(),     # pre-action state
            "action": {"r1": r1, "c1": c1, "r2": r2, "c2": c2},
            "legal": bool(info.valid),
            "sum": int(info.sum),
            "reward": int(info.reward),
            "done": bool(done),
            "agent_tag": policy,
            "rng_seed": int(seed)
        }
        rows.append(row)
        if done:
            break

    header = {
        "episode_id": f"seed{seed}_H{H}W{W}",
        "seed": int(seed),
        "agent_tag": policy,
        "total_reward": int(sum(r["reward"] for r in rows)),
        "total_steps": int(len(rows)),
        "terminated_reason": "no_legal_moves"
    }
    return rows, header


# -----------------------------
# CLI
# -----------------------------

def main():
    p = argparse.ArgumentParser(description="Generate Sum-10 trajectories.")
    p.add_argument("--episodes", type=int, default=100, help="number of episodes to generate")
    p.add_argument("--seed_start", type=int, default=1, help="first RNG seed")
    p.add_argument("--policy", type=str, default="greedy_area",
                   choices=["random_legal", "greedy_area", "lookahead1", "lookahead1:0.3", "lookahead1:0.0"],
                   help="scripted policy")
    p.add_argument("--out_dir", type=str, default="out_data", help="output directory")
    p.add_argument("--format", type=str, default="jsonl", choices=["jsonl", "parquet"], help="output format for trajectories")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    traj_rows: List[dict] = []
    episode_rows: List[dict] = []

    for i in range(args.episodes):
        seed = args.seed_start + i
        rows, header = generate_episode(
            seed=seed,
            policy=args.policy,
            H=10, W=17
        )
        traj_rows.extend(rows)
        episode_rows.append(header)
        if (i + 1) % 10 == 0:
            print(f"[info] generated {i+1}/{args.episodes} episodes")

    # trajectories format
    if args.format == "jsonl":
        write_jsonl(out_dir / "trajectories.jsonl", traj_rows, compress=False)
        print(f"[info] wrote jsonl: {out_dir/'trajectories.jsonl'}")
    else:
        maybe_write_parquet(out_dir / "trajectories.parquet", traj_rows)

    # episodes header (summary of trajectories)
    write_jsonl(out_dir / "episodes.jsonl", episode_rows, compress=False)
    print(f"[info] wrote episodes: {out_dir/'episodes.jsonl'}")

if __name__ == "__main__":
    main()