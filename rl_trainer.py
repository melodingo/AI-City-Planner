"""
rl_trainer.py
-------------
Headless reinforcement-learning trainer for AI City Planner.

Algorithm
---------
Linear Double Q-learning with epsilon-greedy exploration.

We approximate Q(s, a) with a linear model:
  Q(s, a) = w . phi(s, a)

Update (Double Q):
  With 50% chance update Q1 using greedy action from Q1 and value from Q2,
  otherwise swap the roles.

This is intentionally lightweight (numpy-only) so it can run for long periods
without browser overhead or GPU dependencies.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from environment import TrafficEnv


@dataclass
class TrainConfig:
    episodes: int = 300
    episode_length: int = 200
    width: int = 20
    height: int = 20
    max_cars: int = 30
    spawn_rate: float = 0.3
    gamma: float = 0.98
    alpha: float = 0.02
    epsilon_start: float = 0.25
    epsilon_end: float = 0.02
    epsilon_decay_episodes: int = 250
    seed: int = 42
    eval_every: int = 10
    eval_episodes: int = 30
    td_clip: float = 2.0
    l2_weight_decay: float = 1e-5
    save_path: str = "models/linear_double_q_weights.npz"


class LinearDoubleQAgent:
    """Linear function approximator with Double Q-learning updates."""

    def __init__(
        self,
        env: TrafficEnv,
        alpha: float,
        gamma: float,
        td_clip: float = 0.0,
        l2_weight_decay: float = 0.0,
        seed: int = 42,
    ):
        self.env = env
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.td_clip = max(0.0, float(td_clip))
        self.l2_weight_decay = max(0.0, float(l2_weight_decay))
        self.rng = np.random.default_rng(seed)

        # phi(s,a) has: 1 bias + 7 state + 7 action + (7x7) cross = 64 features.
        self.feature_dim = 64
        self.w1 = np.zeros(self.feature_dim, dtype=np.float64)
        self.w2 = np.zeros(self.feature_dim, dtype=np.float64)

        self.width = env.width
        self.height = env.height
        self.actions_per_type = self.width * self.height

    def _decode_action(self, action: int) -> Tuple[int, int, int]:
        action_type = action // self.actions_per_type
        tile_index = action % self.actions_per_type
        y = tile_index // self.width
        x = tile_index % self.width
        return action_type, x, y

    def _state_features(self, obs: np.ndarray, info: Dict[str, float], step: int) -> np.ndarray:
        """Global state features shared across all actions."""
        avg_travel = float(info.get("avg_travel_time", 0.0)) / 60.0
        stopped = float(info.get("stopped_cars", 0.0)) / 30.0
        active = float(info.get("active_cars", 0.0)) / 30.0
        completed = float(info.get("completed_cars", 0.0)) / 30.0
        congestion = float(np.mean(obs[1]))
        intersection_ratio = float(np.mean(obs[2]))
        progress = step / max(1.0, float(self.env.episode_length))

        feats = np.array(
            [
                np.clip(avg_travel, 0.0, 1.0),
                np.clip(stopped, 0.0, 1.0),
                np.clip(active, 0.0, 1.0),
                np.clip(completed, 0.0, 1.0),
                np.clip(congestion, 0.0, 1.0),
                np.clip(intersection_ratio, 0.0, 1.0),
                np.clip(progress, 0.0, 1.0),
            ],
            dtype=np.float64,
        )
        return feats

    def _action_features(self, action_type: int, x: int, y: int) -> np.ndarray:
        """Local action descriptor."""
        tx = x / max(1.0, float(self.width - 1))
        ty = y / max(1.0, float(self.height - 1))
        cx = abs(tx - 0.5) * 2.0
        cy = abs(ty - 0.5) * 2.0

        a_type = np.zeros(3, dtype=np.float64)
        a_type[action_type] = 1.0

        return np.concatenate([a_type, np.array([tx, ty, cx, cy], dtype=np.float64)])

    def phi(self, obs: np.ndarray, info: Dict[str, float], step: int, action: int) -> np.ndarray:
        s = self._state_features(obs, info, step)
        action_type, x, y = self._decode_action(action)
        a = self._action_features(action_type, x, y)

        # [bias] + state + action + flattened outer-product(state, action)
        cross = np.outer(s, a).reshape(-1)
        feats = np.concatenate([
            np.array([1.0], dtype=np.float64),
            s,
            a,
            cross,
        ])
        return feats

    def q1(self, obs: np.ndarray, info: Dict[str, float], step: int, action: int) -> float:
        return float(np.dot(self.w1, self.phi(obs, info, step, action)))

    def q2(self, obs: np.ndarray, info: Dict[str, float], step: int, action: int) -> float:
        return float(np.dot(self.w2, self.phi(obs, info, step, action)))

    def q_sum(self, obs: np.ndarray, info: Dict[str, float], step: int, action: int) -> float:
        feats = self.phi(obs, info, step, action)
        return float(np.dot(self.w1 + self.w2, feats))

    def valid_actions(self, obs: np.ndarray) -> np.ndarray:
        """Return encoded actions that can change the grid under current state."""
        # obs[0] stores normalized tile type: EMPTY=0, ROAD=1/3, HIGHWAY=2/3, INTERSECTION=1.
        cells = np.rint(np.clip(obs[0], 0.0, 1.0) * 3.0).astype(np.int8).reshape(-1)

        add_road = np.where(cells == 0)[0]
        upgrade_highway = np.where(cells == 1)[0] + self.actions_per_type
        add_intersection = np.where((cells == 1) | (cells == 2))[0] + 2 * self.actions_per_type

        valid = np.concatenate([add_road, upgrade_highway, add_intersection])
        if valid.size == 0:
            return np.arange(self.env.action_space_n, dtype=np.int64)
        return valid.astype(np.int64)

    def best_action(
        self,
        obs: np.ndarray,
        info: Dict[str, float],
        step: int,
        use_q1: bool = True,
        actions: np.ndarray | None = None,
    ) -> int:
        action_set = actions if actions is not None else self.valid_actions(obs)
        best_a = int(action_set[0])
        best_val = -float("inf")
        for a in action_set:
            aa = int(a)
            v = self.q1(obs, info, step, aa) if use_q1 else self.q2(obs, info, step, aa)
            if v > best_val:
                best_val = v
                best_a = aa
        return best_a

    def act(self, obs: np.ndarray, info: Dict[str, float], step: int, epsilon: float) -> int:
        actions = self.valid_actions(obs)
        if self.rng.random() < epsilon:
            idx = int(self.rng.integers(0, len(actions)))
            return int(actions[idx])

        best_a = int(actions[0])
        best_val = -float("inf")
        for a in actions:
            aa = int(a)
            v = self.q_sum(obs, info, step, aa)
            if v > best_val:
                best_val = v
                best_a = aa
        return best_a

    def update(
        self,
        obs: np.ndarray,
        info: Dict[str, float],
        step: int,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        next_info: Dict[str, float],
        done: bool,
    ) -> float:
        feats = self.phi(obs, info, step, action)

        if self.rng.random() < 0.5:
            current = float(np.dot(self.w1, feats))
            if done:
                target = reward
            else:
                next_actions = self.valid_actions(next_obs)
                a_star = self.best_action(next_obs, next_info, step + 1, use_q1=True, actions=next_actions)
                target = reward + self.gamma * self.q2(next_obs, next_info, step + 1, a_star)
            td_error = target - current
            td_used = float(np.clip(td_error, -self.td_clip, self.td_clip)) if self.td_clip > 0.0 else td_error
            self.w1 += self.alpha * (td_used * feats - self.l2_weight_decay * self.w1)
        else:
            current = float(np.dot(self.w2, feats))
            if done:
                target = reward
            else:
                next_actions = self.valid_actions(next_obs)
                a_star = self.best_action(next_obs, next_info, step + 1, use_q1=False, actions=next_actions)
                target = reward + self.gamma * self.q1(next_obs, next_info, step + 1, a_star)
            td_error = target - current
            td_used = float(np.clip(td_error, -self.td_clip, self.td_clip)) if self.td_clip > 0.0 else td_error
            self.w2 += self.alpha * (td_used * feats - self.l2_weight_decay * self.w2)

        return float(td_error)

    def save(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, w1=self.w1, w2=self.w2)

    def load(self, path: str) -> None:
        data = np.load(path)
        self.w1 = data["w1"].astype(np.float64)
        self.w2 = data["w2"].astype(np.float64)


def epsilon_for_episode(cfg: TrainConfig, episode: int) -> float:
    if episode >= cfg.epsilon_decay_episodes:
        return cfg.epsilon_end
    span = max(1, cfg.epsilon_decay_episodes)
    frac = episode / span
    return float(cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * frac)


def make_env(cfg: TrainConfig) -> TrafficEnv:
    return TrafficEnv(
        width=cfg.width,
        height=cfg.height,
        max_cars=cfg.max_cars,
        spawn_rate=cfg.spawn_rate,
        episode_length=cfg.episode_length,
    )


def evaluate_policy(agent: LinearDoubleQAgent, cfg: TrainConfig, episodes: int) -> float:
    env = make_env(cfg)
    total = 0.0

    for _ in range(episodes):
        obs = env.reset()
        info = dict(env._last_metrics)
        done = False
        step = 0
        ep_reward = 0.0

        while not done:
            action = agent.act(obs, info, step, epsilon=0.0)
            next_obs, reward, done, next_info = env.step(action)
            ep_reward += float(reward)
            obs, info = next_obs, next_info
            step += 1

        total += ep_reward

    return total / max(1, episodes)


def train(cfg: TrainConfig) -> Dict[str, float]:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = make_env(cfg)
    agent = LinearDoubleQAgent(
        env,
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        td_clip=cfg.td_clip,
        l2_weight_decay=cfg.l2_weight_decay,
        seed=cfg.seed,
    )

    best_eval_reward = -float("inf")
    best_episode = -1

    print("=== RL Training (Linear Double Q) ===")
    print(
        f"episodes={cfg.episodes} | episode_length={cfg.episode_length} | "
        f"gamma={cfg.gamma:.3f} | alpha={cfg.alpha:.4f}"
    )
    checkpoint_steps = max(1, cfg.episode_length // 5)

    for ep in range(1, cfg.episodes + 1):
        epsilon = epsilon_for_episode(cfg, ep)
        print(f"Episode {ep}/{cfg.episodes} started", flush=True)

        obs = env.reset()
        info = dict(env._last_metrics)

        done = False
        step = 0
        episode_reward = 0.0
        td_abs_sum = 0.0

        while not done:
            action = agent.act(obs, info, step, epsilon)
            next_obs, reward, done, next_info = env.step(action)
            td = agent.update(
                obs=obs,
                info=info,
                step=step,
                action=action,
                reward=float(reward),
                next_obs=next_obs,
                next_info=next_info,
                done=done,
            )

            td_abs_sum += abs(td)
            episode_reward += float(reward)
            obs, info = next_obs, next_info
            step += 1

            if step % checkpoint_steps == 0 or done:
                print(
                    f"Episode {ep}/{cfg.episodes} | step {step}/{cfg.episode_length} | "
                    f"reward {episode_reward:>9.3f} | epsilon {epsilon:.3f} | "
                    f"mean|TD| {(td_abs_sum / max(1, step)):.4f}",
                    flush=True,
                )

        mean_abs_td = td_abs_sum / max(1, step)
        print(
            f"Episode {ep:>4}/{cfg.episodes} | "
            f"reward {episode_reward:>9.3f} | "
            f"epsilon {epsilon:.3f} | "
            f"mean|TD| {mean_abs_td:.4f}"
        )

        if ep % cfg.eval_every == 0 or ep == cfg.episodes:
            eval_reward = evaluate_policy(agent, cfg, cfg.eval_episodes)
            print(f"  Eval over {cfg.eval_episodes} episodes: {eval_reward:.3f}")
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_episode = ep
                agent.save(cfg.save_path)
                print(f"  New best model saved to {cfg.save_path}")

    metrics = {
        "best_eval_reward": float(best_eval_reward),
        "best_episode": float(best_episode),
    }

    meta_path = Path(cfg.save_path).with_suffix(".json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "algorithm": "linear_double_q",
        "config": cfg.__dict__,
        "result": metrics,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Best eval reward: {best_eval_reward:.3f} at episode {best_episode}")
    print(f"Weights file: {cfg.save_path}")
    print(f"Metadata file: {meta_path}")

    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Headless RL training for AI City Planner")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--max-cars", type=int, default=30)
    parser.add_argument("--spawn-rate", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--epsilon-start", type=float, default=0.25)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--td-clip", type=float, default=2.0)
    parser.add_argument("--l2-weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="models/linear_double_q_weights.npz")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        episodes=args.episodes,
        episode_length=args.episode_length,
        width=args.width,
        height=args.height,
        max_cars=args.max_cars,
        spawn_rate=args.spawn_rate,
        gamma=args.gamma,
        alpha=args.alpha,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        seed=args.seed,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        td_clip=args.td_clip,
        l2_weight_decay=args.l2_weight_decay,
        save_path=args.save_path,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)
    train(cfg)


if __name__ == "__main__":
    main()
