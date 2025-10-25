import json
import random
import asyncio
import os
from run_single_trajectory import run_single_trajectory

async def run_o3_on_seed(seed: int) -> dict:
    print(f"Running o3 on seed {seed}...")
    
    total_reward, total_steps = await run_single_trajectory(seed=seed)
    
    return {
        "seed": seed,
        "total_reward": total_reward,
        "total_steps": total_steps
    }

def load_policy_results():
    policies = ["minimal_area_1k", "greedy_area_1k", "look_ahead_1k_2_70_0.95", "random_legal_1k"]
    results = {}
    
    for policy in policies:
        results[policy] = {}
        file_path = f"out_data/{policy}/episodes.jsonl"
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    results[policy][data["seed"]] = {
                        "total_reward": data["total_reward"],
                        "total_steps": data["total_steps"]
                    }
        except FileNotFoundError:
            print(f"Warning: {file_path} not found")
            results[policy] = {}
    
    return results

async def main():
    random.seed(42)
    test_seeds = random.sample(range(1, 1001), 5)
    print(f"Testing seeds: {test_seeds}")
    
    policy_results = load_policy_results()
    
    o3_results = []
    for seed in test_seeds:
        try:
            result = await run_o3_on_seed(seed)
            o3_results.append(result)
            print(f"o3 seed {seed}: {result['total_reward']} reward, {result['total_steps']} steps")
        except Exception as e:
            print(f"Error running o3 on seed {seed}: {e}")
            o3_results.append({
                "seed": seed,
                "total_reward": 0,
                "total_steps": 0,
                "error": str(e)
            })
    
    benchmark_results = []
    
    for i, o3_result in enumerate(o3_results):
        seed = o3_result["seed"]
        
        result_entry = {
            "seed": seed,
            "o3": {
                "total_reward": o3_result["total_reward"],
                "total_steps": o3_result["total_steps"]
            }
        }
        
        for policy in ["minimal_area_1k", "greedy_area_1k", "look_ahead_1k_2_70_0.95", "random_legal_1k"]:
            if seed in policy_results[policy]:
                result_entry[policy] = policy_results[policy][seed]
            else:
                result_entry[policy] = {"total_reward": 0, "total_steps": 0, "error": "seed not found"}
        
        benchmark_results.append(result_entry)
    
    with open("results.jsonl", "w") as f:
        for result in benchmark_results:
            f.write(json.dumps(result) + "\n")
    
    print(f"\nBenchmark complete! Results saved to results.jsonl")
    print(f"Tested {len(test_seeds)} seeds: {test_seeds}")
    
    print("\n=== SUMMARY ===")
    for result in benchmark_results:
        seed = result["seed"]
        o3_reward = result["o3"]["total_reward"]
        print(f"\nSeed {seed}:")
        print(f"  o3: {o3_reward} reward")
        for policy in ["minimal_area_1k", "greedy_area_1k", "look_ahead_1k_2_70_0.95", "random_legal_1k"]:
            if policy in result:
                policy_reward = result[policy]["total_reward"]
                diff = o3_reward - policy_reward
                print(f"  {policy}: {policy_reward} reward (o3 +{diff})")

if __name__ == "__main__":
    asyncio.run(main())