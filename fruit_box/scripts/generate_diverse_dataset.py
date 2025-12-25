#!/usr/bin/env python3
"""
Generate a diversified SFT dataset with borderline negative examples.

Each episode is generated with the minimal-area policy (best-performing among the
built-in heuristics). For every state we log the legal move the policy executes
plus a handful of "nearby" rectangles that are illegal but close to the legal
box (e.g., expand the bounds by one cell). This should give train_sft.py more
informative negatives without requiring on-the-fly generation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Set

from tqdm import tqdm

# Reuse existing environment/policy utilities from policies.generate_dataset
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policies.generate_dataset import (  # type: ignore
    Sum10Env,
    policy_minimal_area,
    write_jsonl,
)


Box = Tuple[int, int, int, int]


def generate_borderline_negatives(
    env: Sum10Env,
    legal_box: Box,
    max_candidates: int,
) -> List[Box]:
    """Create rectangles near the legal_box that are likely illegal."""
    r1, c1, r2, c2 = legal_box
    candidates: List[Box] = []
    seen: Set[Box] = set()
    adjustments = [
        (r1, c1, min(r2 + 1, env.H - 1), c2),
        (r1, c1, r2, min(c2 + 1, env.W - 1)),
        (max(r1 - 1, 0), c1, r2, c2),
        (r1, max(c1 - 1, 0), r2, c2),
        (r1, c1, min(r2 + 2, env.H - 1), c2),
        (r1, c1, r2, min(c2 + 2, env.W - 1)),
    ]

    for cand in adjustments:
        if cand == legal_box:
            continue
        cr1, cc1, cr2, cc2 = cand
        normalized = (cr1, cc1, cr2, cc2)
        if normalized in seen:
            continue
        seen.add(normalized)
        if cr1 > cr2 or cc1 > cc2:
            continue
        # Skip if rectangle identical to legal box or out of bounds
        if not (0 <= cr1 <= cr2 < env.H and 0 <= cc1 <= cc2 < env.W):
            continue
        total = env.box_sum(cr1, cc1, cr2, cc2)
        reward = env.box_nonzero_count(cr1, cc1, cr2, cc2)
        if total == 10 or reward == 0:
            continue  # not a good negative candidate
        candidates.append(normalized)
        if len(candidates) >= max_candidates:
            break
    return candidates


@dataclass
class Config:
    episodes: int = 1000
    seed_start: int = 1
    out_dir: Path = Path("out_data/diverse_1k")
    negatives_per_step: int = 3


def generate_episode(
    seed: int,
    config: Config,
) -> Tuple[List[Dict], Dict]:
    env = Sum10Env(H=10, W=17, seed=seed)
    env.reset()
    rows: List[Dict] = []
    step = 0

    while env.has_any_legal():
        legal_choices = env.enumerate_legal()
        if not legal_choices:
            break

        action_box = policy_minimal_area(env)
        if action_box is None:
            break

        r1, c1, r2, c2 = action_box
        grid_before = env.grid.copy()
        num_legal_actions = len(legal_choices)

        # positive row
        rows.append(
            {
                "episode_id": f"seed{seed}",
                "step": step + 1,
                "grid": grid_before.tolist(),
                "action": {"r1": r1, "c1": c1, "r2": r2, "c2": c2},
                "num_legal_actions": num_legal_actions,
                "legal": True,
                "reward": int(env.box_nonzero_count(r1, c1, r2, c2)),
                "done": False,
                "agent_tag": "minimal_area",
                "rng_seed": int(seed),
            }
        )

        # borderline negatives
        negative_boxes = generate_borderline_negatives(
            env,
            action_box,
            config.negatives_per_step,
        )
        for neg_box in negative_boxes:
            nr1, nc1, nr2, nc2 = neg_box
            rows.append(
                {
                    "episode_id": f"seed{seed}",
                    "step": step + 1,
                    "grid": grid_before.tolist(),
                    "action": {"r1": nr1, "c1": nc1, "r2": nr2, "c2": nc2},
                    "num_legal_actions": num_legal_actions,
                    "legal": False,
                    "reward": 0,
                    "done": False,
                    "agent_tag": "minimal_area_negative",
                    "rng_seed": int(seed),
                }
            )

        # execute the legal move
        info = env.step(r1, c1, r2, c2)
        step += 1
        if info.done:
            break

    header = {
        "episode_id": f"seed{seed}",
        "seed": int(seed),
        "agent_tag": "minimal_area",
        "total_reward": int(sum(r["reward"] for r in rows if r["legal"])),
        "total_steps": int(step),
    }
    return rows, header


def main():
    parser = argparse.ArgumentParser(description="Generate diverse dataset with negatives.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--negatives_per_step", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default="out_data/diverse_1k")
    args = parser.parse_args()

    config = Config(
        episodes=args.episodes,
        seed_start=args.seed_start,
        out_dir=Path(args.out_dir),
        negatives_per_step=args.negatives_per_step,
    )

    trajectories: List[Dict] = []
    episode_headers: List[Dict] = []

    for i in tqdm(range(config.episodes), desc="Generating episodes"):
        seed = config.seed_start + i
        rows, header = generate_episode(seed, config)
        trajectories.extend(rows)
        episode_headers.append(header)

    config.out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = config.out_dir / "trajectories.jsonl"
    episodes_path = config.out_dir / "episodes.jsonl"

    write_jsonl(traj_path, trajectories)
    write_jsonl(episodes_path, episode_headers)

    print(f"Wrote trajectories: {traj_path}")
    print(f"Wrote episode summaries: {episodes_path}")
    print(f"Total rows written: {len(trajectories)}")


if __name__ == "__main__":
    main()

