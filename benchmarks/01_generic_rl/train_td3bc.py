"""Train TD3+BC benchmark on existing replay buffers.

This script is benchmark-only and intentionally keeps:
- no hard intent constraints
- no intent auxiliary penalties

It still uses the repository's ActorNetwork checkpoint format so that
`scripts/evaluate_run.py` can evaluate outputs directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation import analyzer
from src.training.data import ReplayDataset, load_buffer
from src.training.models import ActorNetwork, CriticNetwork
from src.utils.logger import get_logger
from train import TrainingConfig as EvalTrainingConfig

LOGGER = get_logger(__name__)


@dataclass
class TD3BCConfig:
    kol: str = "youtube/Adam_Khoo"
    replay_dir: str = "data/multisource_ready_22-25/08_replay_buffer"
    checkpoints_dir: str = "models/checkpoints"
    output_dir: str = "outputs/benchmarks/generic_rl/td3bc"

    batch_size: int = 256
    steps: int = 200_000
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    alpha: float = 2.5
    clamp_delta: float = 1.8

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    seed: int = 42
    log_interval: int = 200
    write_csv: bool = True
    progress_bar: bool = True
    action_threshold: float = 0.02

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> TD3BCConfig:
    p = argparse.ArgumentParser(description="Train TD3+BC benchmark.")
    p.add_argument("--kol", required=True)
    p.add_argument("--replay-dir", default="data/multisource_ready_22-25/08_replay_buffer")
    p.add_argument("--checkpoints-dir", default="models/checkpoints")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--policy-noise", type=float, default=0.2)
    p.add_argument("--noise-clip", type=float, default=0.5)
    p.add_argument("--policy-freq", type=int, default=2)
    p.add_argument("--alpha", type=float, default=2.5)
    p.add_argument("--clamp-delta", type=float, default=1.8)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--write-csv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--progress-bar", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--action-threshold", type=float, default=0.02)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    return TD3BCConfig(
        kol=args.kol,
        replay_dir=args.replay_dir,
        checkpoints_dir=args.checkpoints_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        steps=args.steps,
        gamma=args.gamma,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        alpha=args.alpha,
        clamp_delta=args.clamp_delta,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        seed=args.seed,
        log_interval=args.log_interval,
        write_csv=args.write_csv,
        progress_bar=args.progress_bar,
        action_threshold=args.action_threshold,
        device=args.device,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_delta(actor_out, baseline_action: torch.Tensor) -> torch.Tensor:
    if isinstance(actor_out, torch.Tensor):
        return actor_out
    if isinstance(actor_out, dict):
        has_signal = baseline_action.abs() > 1e-6
        delta_signal = actor_out.get("delta_signal")
        delta_decay = actor_out.get("delta_decay")
        if delta_signal is None or delta_decay is None:
            raise KeyError("ActorNetwork dict output missing delta_signal/delta_decay.")
        return torch.where(has_signal, delta_signal, delta_decay)
    raise TypeError(f"Unsupported actor output type: {type(actor_out)}")


def build_action(actor: ActorNetwork, state: torch.Tensor, baseline_action: torch.Tensor, clamp_delta: float) -> torch.Tensor:
    out = actor(state)
    delta = _extract_delta(out, baseline_action)
    delta = torch.clamp(delta, -clamp_delta, clamp_delta)
    return baseline_action + delta


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.mul_(1.0 - tau).add_(tau * s_param.data)


def evaluate_on_split(
    actor: ActorNetwork,
    split_path: Path,
    cfg: TD3BCConfig,
    device: torch.device,
) -> Dict[str, float]:
    if not split_path.exists():
        return {}
    buffer = load_buffer(split_path)
    eval_cfg = EvalTrainingConfig(
        hard_intent_constraints=False,
        clamp_delta=cfg.clamp_delta,
        entry_threshold=5e-4,
    )
    metrics, _ = analyzer.run_policy(
        actor,
        buffer,
        device,
        action_threshold=cfg.action_threshold,
        cfg=eval_cfg,
    )
    return metrics


def save_checkpoints(
    actor: ActorNetwork,
    critic1: CriticNetwork,
    critic2: CriticNetwork,
    checkpoint_dir: Path,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), checkpoint_dir / "actor.pt")
    torch.save(critic1.state_dict(), checkpoint_dir / "critic1.pt")
    torch.save(critic2.state_dict(), checkpoint_dir / "critic2.pt")
    torch.save({"actor_state_dict": actor.state_dict()}, checkpoint_dir / "policy.pt")
    LOGGER.info("Saved checkpoints to %s", checkpoint_dir)


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_path = Path(cfg.replay_dir) / cfg.kol / "train.pt"
    val_path = Path(cfg.replay_dir) / cfg.kol / "val.pt"
    if not train_path.exists():
        raise FileNotFoundError(f"Replay buffer not found: {train_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.kol}_{timestamp}"
    run_dir = Path(cfg.output_dir) / run_name
    log_dir = run_dir / "logs"
    checkpoint_dir = run_dir / Path(cfg.checkpoints_dir).name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "training.log"
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(fh)

    LOGGER.info("Starting TD3+BC benchmark run %s", run_name)
    LOGGER.info("Logging to %s", log_path)
    LOGGER.info("Checkpoints will be saved under %s", checkpoint_dir)
    LOGGER.info("Config: %s", json.dumps(asdict(cfg), ensure_ascii=False, sort_keys=True))
    if torch.cuda.is_available() and device.type == "cuda":
        LOGGER.info("CUDA device: %s", torch.cuda.get_device_name(device))
    LOGGER.info("Device: %s", device)

    train_dataset = ReplayDataset(train_path)
    state_dim = int(train_dataset.states.shape[1])
    if len(train_dataset) == 0:
        raise RuntimeError(f"Empty train dataset for {cfg.kol}")
    drop_last = len(train_dataset) >= cfg.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=drop_last,
        pin_memory=True,
    )
    if len(train_loader) == 0:
        raise RuntimeError(
            f"No train batches for {cfg.kol}. dataset={len(train_dataset)}, batch_size={cfg.batch_size}. "
            "Try smaller batch size."
        )
    LOGGER.info(
        "Loaded replay buffer for %s with %d samples, state_dim=%d, drop_last=%s",
        cfg.kol,
        len(train_dataset),
        state_dim,
        drop_last,
    )

    actor = ActorNetwork(state_dim).to(device)
    actor_target = ActorNetwork(state_dim).to(device)
    actor_target.load_state_dict(actor.state_dict())

    critic_state_dim = state_dim + 1  # append baseline action to state
    critic1 = CriticNetwork(critic_state_dim).to(device)
    critic2 = CriticNetwork(critic_state_dim).to(device)
    critic1_target = CriticNetwork(critic_state_dim).to(device)
    critic2_target = CriticNetwork(critic_state_dim).to(device)
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(
        list(critic1.parameters()) + list(critic2.parameters()),
        lr=cfg.critic_lr,
    )

    csv_fp: Optional[object] = None
    csv_writer = None
    if cfg.write_csv:
        csv_path = log_dir / "td3bc_metrics.csv"
        csv_fp = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fp)
        csv_writer.writerow(
            [
                "step",
                "critic_loss",
                "actor_loss",
                "bc_loss",
                "q_loss_term",
                "lambda",
                "q1_mean",
                "target_q_mean",
                "reward_mean",
                "done_ratio",
                "policy_abs_mean",
                "behavior_abs_mean",
            ]
        )
        csv_fp.flush()
        LOGGER.info("TD3+BC metrics CSV: %s", csv_path)

    iterator = cycle(train_loader)
    rolling = {
        "critic_loss": 0.0,
        "actor_loss": 0.0,
        "bc_loss": 0.0,
        "q_loss_term": 0.0,
        "lambda": 0.0,
        "q1_mean": 0.0,
        "target_q_mean": 0.0,
        "reward_mean": 0.0,
        "done_ratio": 0.0,
        "policy_abs_mean": 0.0,
        "behavior_abs_mean": 0.0,
    }
    rolling_n = 0

    steps_iter = range(1, cfg.steps + 1)
    if cfg.progress_bar:
        steps_iter = tqdm(steps_iter, desc="TD3+BC Training")

    for step in steps_iter:
        batch = next(iterator)
        state = batch["state"].to(device)
        next_state = batch["next_state"].to(device)
        reward = batch["reward"].to(device).squeeze(-1)
        done = batch["done"].to(device).float().squeeze(-1)
        behavior_action = batch["action"].to(device)
        baseline_action = batch["baseline_action"].to(device)
        next_baseline_action = batch["next_baseline_action"].to(device)

        ext_state = torch.cat([state, baseline_action], dim=-1)
        ext_next_state = torch.cat([next_state, next_baseline_action], dim=-1)

        with torch.no_grad():
            next_action = build_action(actor_target, next_state, next_baseline_action, cfg.clamp_delta)
            noise = torch.randn_like(next_action) * cfg.policy_noise
            noise = torch.clamp(noise, -cfg.noise_clip, cfg.noise_clip)
            next_action = next_action + noise
            target_q1 = critic1_target(ext_next_state, next_action).squeeze(-1)
            target_q2 = critic2_target(ext_next_state, next_action).squeeze(-1)
            target_q = reward + cfg.gamma * (1.0 - done) * torch.minimum(target_q1, target_q2)

        current_q1 = critic1(ext_state, behavior_action).squeeze(-1)
        current_q2 = critic2(ext_state, behavior_action).squeeze(-1)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        actor_loss_value = 0.0
        bc_loss_value = 0.0
        q_loss_term_value = 0.0
        lambda_value = 0.0
        policy_abs_mean = 0.0
        if step % cfg.policy_freq == 0:
            policy_action = build_action(actor, state, baseline_action, cfg.clamp_delta)
            q_for_policy = critic1(ext_state, policy_action)
            q_mean_abs = q_for_policy.abs().mean().detach().clamp(min=1e-6)
            lambda_coef = cfg.alpha / q_mean_abs
            bc_loss = F.mse_loss(policy_action, behavior_action)
            q_loss_term = -lambda_coef * q_for_policy.mean()
            actor_loss = q_loss_term + bc_loss

            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            soft_update(actor_target, actor, cfg.tau)
            soft_update(critic1_target, critic1, cfg.tau)
            soft_update(critic2_target, critic2, cfg.tau)

            actor_loss_value = float(actor_loss.item())
            bc_loss_value = float(bc_loss.item())
            q_loss_term_value = float(q_loss_term.item())
            lambda_value = float(lambda_coef.item())
            policy_abs_mean = float(policy_action.abs().mean().item())

        rolling["critic_loss"] += float(critic_loss.item())
        rolling["actor_loss"] += actor_loss_value
        rolling["bc_loss"] += bc_loss_value
        rolling["q_loss_term"] += q_loss_term_value
        rolling["lambda"] += lambda_value
        rolling["q1_mean"] += float(current_q1.mean().item())
        rolling["target_q_mean"] += float(target_q.mean().item())
        rolling["reward_mean"] += float(reward.mean().item())
        rolling["done_ratio"] += float(done.mean().item())
        rolling["policy_abs_mean"] += policy_abs_mean
        rolling["behavior_abs_mean"] += float(behavior_action.abs().mean().item())
        rolling_n += 1

        if step % max(cfg.log_interval, 1) == 0 or step == cfg.steps:
            avg = {k: v / max(rolling_n, 1) for k, v in rolling.items()}
            LOGGER.info(
                "TD3+BC step %d/%d | critic=%.6f actor=%.6f bc=%.6f q_term=%.6f lambda=%.6f q1=%.6f target_q=%.6f reward=%.6f done=%.4f | |pi|=%.6f |beh|=%.6f",
                step,
                cfg.steps,
                avg["critic_loss"],
                avg["actor_loss"],
                avg["bc_loss"],
                avg["q_loss_term"],
                avg["lambda"],
                avg["q1_mean"],
                avg["target_q_mean"],
                avg["reward_mean"],
                avg["done_ratio"],
                avg["policy_abs_mean"],
                avg["behavior_abs_mean"],
            )
            if csv_writer is not None:
                csv_writer.writerow(
                    [
                        step,
                        avg["critic_loss"],
                        avg["actor_loss"],
                        avg["bc_loss"],
                        avg["q_loss_term"],
                        avg["lambda"],
                        avg["q1_mean"],
                        avg["target_q_mean"],
                        avg["reward_mean"],
                        avg["done_ratio"],
                        avg["policy_abs_mean"],
                        avg["behavior_abs_mean"],
                    ]
                )
                csv_fp.flush()
            for k in rolling:
                rolling[k] = 0.0
            rolling_n = 0

    if csv_fp is not None:
        csv_fp.close()

    metrics = evaluate_on_split(actor, val_path, cfg, device)
    if metrics:
        LOGGER.info(
            "Validation metrics for %s: cumulative_return=%.4f, sharpe=%.4f, max_drawdown=%.4f",
            cfg.kol,
            metrics.get("cumulative_return", 0.0),
            metrics.get("sharpe", 0.0),
            metrics.get("max_drawdown", 0.0),
        )
    else:
        LOGGER.warning("Validation buffer %s not found; skipping evaluation.", val_path)

    save_checkpoints(actor, critic1, critic2, checkpoint_dir)

    summary = {
        "run_name": run_name,
        "timestamp": timestamp,
        "kol": cfg.kol,
        "train_samples": len(train_dataset),
        "metrics": metrics,
        "config": asdict(cfg),
    }
    summary_path = run_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    LOGGER.info("Saved run summary to %s", summary_path)


if __name__ == "__main__":
    main()
