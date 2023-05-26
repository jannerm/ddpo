import numpy as np
from collections import deque


# ringbuffer version
class PerPromptStatTracker:
    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {}

    def update(self, prompts, rewards):
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = deque(maxlen=self.buffer_size)
            self.stats[prompt].extend(prompt_rewards)

            if len(self.stats[prompt]) < self.min_count:
                mean = np.mean(rewards)
                std = np.std(rewards) + 1e-6
            else:
                mean = np.mean(self.stats[prompt])
                std = np.std(self.stats[prompt]) + 1e-6
            advantages[prompts == prompt] = (prompt_rewards - mean) / std

        return advantages

    def get_stats(self):
        return {
            k: {"mean": np.mean(v), "std": np.std(v), "count": len(v)}
            for k, v in self.stats.items()
        }


# ema decay version
"""
class PerPromptStatTracker:
    def __init__(self, ema, min_count):
        self.ema = ema
        self.min_count = min_count
        self.stats = {}

    def update(self, prompts, rewards):
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            prompt_mean = np.mean(prompt_rewards)
            prompt_squared_mean = np.mean(prompt_rewards**2)
            if prompt not in self.stats:
                self.stats[prompt] = {
                    "mean": prompt_mean,
                    "squared_mean": prompt_squared_mean,
                    "count": len(prompt_rewards),
                }
            else:
                self.stats[prompt]["mean"] = (
                    1 - self.ema
                ) * prompt_mean + self.ema * self.stats[prompt]["mean"]
                self.stats[prompt]["squared_mean"] = (
                    1 - self.ema
                ) * prompt_squared_mean + self.ema * self.stats[prompt][
                    "squared_mean"
                ]
                self.stats[prompt]["count"] += len(prompt_rewards)

            if self.stats[prompt]["count"] < self.min_count:
                advantages[prompts == prompt] = (
                    prompt_rewards - np.mean(rewards)
                ) / np.std(rewards)
            else:
                advantages[prompts == prompt] = (
                    prompt_rewards - self.stats[prompt]["mean"]
                ) / np.sqrt(
                    self.stats[prompt]["squared_mean"]
                    - self.stats[prompt]["mean"] ** 2
                )

        return advantages

    def get_stats(self):
        return self.stats
"""
