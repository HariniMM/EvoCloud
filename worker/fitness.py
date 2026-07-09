"""
fitness.py

Worker-side logic: given a model's weights, run it in the CartPole
environment and return a fitness score. This is the function that,
later, gets wrapped in a FastAPI endpoint so a coordinator can call it
remotely on a separate cloud VM. For now it runs as a plain local
function so we can validate the evolution loop end-to-end.
"""

import numpy as np
import gymnasium as gym
from evolution_engine.model import TinyMLP


def evaluate_genome(
    weights: np.ndarray,
    input_size: int = 4,
    hidden_size: int = 8,
    output_size: int = 2,
    episodes: int = 3,
    max_steps: int = 500,
) -> float:
    """
    Run a model in CartPole-v1 for a few episodes and return average reward.
    Higher reward = pole balanced for longer = better fitness.
    """
    model = TinyMLP(input_size, hidden_size, output_size, weights)
    env = gym.make("CartPole-v1")

    total_reward = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        for _ in range(max_steps):
            action = model.act(np.array(obs, dtype=np.float32))
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        total_reward += episode_reward

    env.close()
    return total_reward / episodes


def evaluate_batch(genomes: list, **kwargs) -> list:
    """Evaluate a batch of genomes sequentially. This is what a single
    worker node runs -- the coordinator splits the full population into
    batches and sends one batch per worker."""
    return [evaluate_genome(g, **kwargs) for g in genomes]
