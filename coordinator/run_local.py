"""
run_local.py

Phase 1 of the build (per the project timeline): validate the evolution
algorithm end-to-end on a single machine, with no networking or
distribution yet. This proves the coordinator + worker logic is correct
before we split it across cloud VMs.

Run this directly: python coordinator/run_local.py
"""

import sys
import os
import time
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolution_engine.evolution import init_population, next_generation, population_stats
from evolution_engine.model import TinyMLP
from worker.fitness import evaluate_batch

# --- Config ---
POP_SIZE = 40
GENERATIONS = 15
INPUT_SIZE = 4      # CartPole observation size
HIDDEN_SIZE = 8
OUTPUT_SIZE = 2      # CartPole action space size
ELITE_COUNT = 4
MUTATION_RATE = 0.05
MUTATION_STRENGTH = 0.4

dummy_model = TinyMLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
TOTAL_PARAMS = dummy_model.total_params


def run():
    print(f"Starting EvoCloud local run: pop={POP_SIZE}, generations={GENERATIONS}, params/model={TOTAL_PARAMS}")

    population = init_population(POP_SIZE, TOTAL_PARAMS, seed=42)
    history = []

    for gen in range(1, GENERATIONS + 1):
        start = time.time()

        # In the distributed version, this batch gets split across worker VMs.
        # For now, one "worker" evaluates the whole population locally.
        fitness_scores = evaluate_batch(population, input_size=INPUT_SIZE,
                                         hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE)

        stats = population_stats(fitness_scores)
        elapsed = time.time() - start
        print(f"Gen {gen:3d} | best={stats['best']:.1f} avg={stats['average']:.1f} "
              f"worst={stats['worst']:.1f} std={stats['std']:.1f} | {elapsed:.1f}s")

        history.append({"generation": gen, **stats})

        population = next_generation(
            population, fitness_scores, POP_SIZE,
            elite_count=ELITE_COUNT,
            mutation_rate=MUTATION_RATE,
            mutation_strength=MUTATION_STRENGTH,
        )

    # Save the champion (best individual from the final generation)
    final_scores = evaluate_batch(population, input_size=INPUT_SIZE,
                                   hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE)
    best_idx = final_scores.index(max(final_scores))
    champion_weights = population[best_idx]

    os.makedirs("results", exist_ok=True)
    import numpy as np
    np.save("results/champion_weights.npy", champion_weights)
    with open("results/history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Champion fitness: {max(final_scores):.1f}")
    print("Saved: results/champion_weights.npy, results/history.json")


if __name__ == "__main__":
    run()
