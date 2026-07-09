"""
evolution.py

Core evolutionary operators: population initialization, selection,
crossover, and mutation. This module is deliberately independent of
where fitness evaluation happens (local process or remote worker) --
it just needs a list of (weights, fitness) pairs to produce the next
generation.
"""

import numpy as np
from typing import List, Tuple


def init_population(pop_size: int, total_params: int, seed: int = None) -> List[np.ndarray]:
    """Create a random initial population of weight vectors."""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(total_params).astype(np.float32) * 0.5 for _ in range(pop_size)]


def select_top_k(population: List[np.ndarray], fitness_scores: List[float], k: int) -> List[np.ndarray]:
    """Select the top-k performing individuals (elitism)."""
    ranked = sorted(zip(population, fitness_scores), key=lambda pair: pair[1], reverse=True)
    return [genome for genome, _ in ranked[:k]]


def tournament_select(population: List[np.ndarray], fitness_scores: List[float], tournament_size: int = 3) -> np.ndarray:
    """Pick one parent via tournament selection."""
    idxs = np.random.choice(len(population), size=tournament_size, replace=False)
    best_idx = max(idxs, key=lambda i: fitness_scores[i])
    return population[best_idx]


def crossover(parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
    """Uniform crossover: each gene comes from a or b with 50% probability."""
    mask = np.random.rand(len(parent_a)) < 0.5
    child = np.where(mask, parent_a, parent_b)
    return child.astype(np.float32)


def mutate(genome: np.ndarray, mutation_rate: float = 0.02, mutation_strength: float = 0.3) -> np.ndarray:
    """
    Add Gaussian noise to a fraction of genes.
    mutation_rate: probability that any given weight gets perturbed.
    mutation_strength: standard deviation of the noise added.
    """
    mask = np.random.rand(len(genome)) < mutation_rate
    noise = np.random.randn(len(genome)).astype(np.float32) * mutation_strength
    return genome + mask * noise


def next_generation(
    population: List[np.ndarray],
    fitness_scores: List[float],
    pop_size: int,
    elite_count: int = 5,
    mutation_rate: float = 0.02,
    mutation_strength: float = 0.3,
) -> List[np.ndarray]:
    """
    Produce the next generation:
    1. Carry over the top `elite_count` individuals unchanged (elitism).
    2. Fill the rest via tournament selection + crossover + mutation.
    """
    new_population = select_top_k(population, fitness_scores, elite_count)

    while len(new_population) < pop_size:
        parent_a = tournament_select(population, fitness_scores)
        parent_b = tournament_select(population, fitness_scores)
        child = crossover(parent_a, parent_b)
        child = mutate(child, mutation_rate, mutation_strength)
        new_population.append(child)

    return new_population[:pop_size]


def population_stats(fitness_scores: List[float]) -> dict:
    """Basic stats for dashboard/logging: best, average, worst, diversity proxy."""
    arr = np.array(fitness_scores)
    return {
        "best": float(arr.max()),
        "average": float(arr.mean()),
        "worst": float(arr.min()),
        "std": float(arr.std()),
    }
