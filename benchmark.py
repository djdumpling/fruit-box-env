import json
import random
import os
import argparse
from run_single_trajectory import run_single_trajectory

def run_model_on_seed(seed: int, model: str) -> dict:
    print(f"Running {model} on seed {seed}...")
    
    total_reward, total_steps = run_single_trajectory(seed=seed, model=model)
    
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

def main(model: str, num_rollouts: int):
    random.seed()
    test_seeds = random.sample(range(1, 1001), num_rollouts)
    print(f"Testing seeds: {test_seeds}")
    print(f"Using model: {model}")
    
    policy_results = load_policy_results()
    
    model_results = []
    for seed in test_seeds:
        try:
            result = run_model_on_seed(seed, model)
            model_results.append(result)
            print(f"{model} seed {seed}: {result['total_reward']} reward, {result['total_steps']} steps")
        except Exception as e:
            print(f"Error running {model} on seed {seed}: {e}")
            model_results.append({
                "seed": seed,
                "total_reward": 0,
                "total_steps": 0,
                "error": str(e)
            })
    
    benchmark_results = []
    
    for i, model_result in enumerate(model_results):
        seed = model_result["seed"]
        
        result_entry = {
            "seed": seed,
            model: {
                "total_reward": model_result["total_reward"],
                "total_steps": model_result["total_steps"]
            }
        }
        
        for policy in ["minimal_area_1k", "greedy_area_1k", "look_ahead_1k_2_70_0.95", "random_legal_1k"]:
            if seed in policy_results[policy]:
                result_entry[policy] = policy_results[policy][seed]
            else:
                result_entry[policy] = {"total_reward": 0, "total_steps": 0, "error": "seed not found"}
        
        benchmark_results.append(result_entry)
    
    # filename
    model_name = model.replace("/", "_").replace("-", "_")
    output_filename = f"results_{model_name}_{num_rollouts}.jsonl"
    
    with open(output_filename, "w") as f:
        for result in benchmark_results:
            f.write(json.dumps(result) + "\n")
    
    print(f"\nBenchmark complete! Results saved to {output_filename}")
    print(f"Tested {len(test_seeds)} seeds: {test_seeds}")
    
    print("\n=== SUMMARY ===")
    for result in benchmark_results:
        seed = result["seed"]
        model_reward = result[model]["total_reward"]
        print(f"\nSeed {seed}:")
        print(f"  {model}: {model_reward} reward")
        for policy in ["minimal_area_1k", "greedy_area_1k", "look_ahead_1k_2_70_0.95", "random_legal_1k"]:
            if policy in result:
                policy_reward = result[policy]["total_reward"]
                diff = model_reward - policy_reward
                print(f"  {policy}: {policy_reward} reward ({model} +{diff})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Benchmark a model against existing policies")
    parser.add_argument("-m", "--model", default = "openai/gpt-5", help = "Model to benchmark (default: openai/gpt-5)")
    parser.add_argument("-n", "--num-rollouts", type = int, default = 5, help = "Number of rollouts to run (default: 5)")
    args = parser.parse_args()
    
    main(model = args.model, num_rollouts = args.num_rollouts)
