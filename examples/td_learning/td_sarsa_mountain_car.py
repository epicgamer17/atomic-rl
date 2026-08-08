"""
Reproduction of Sutton (2014): "True Online TD(lambda)"
Experiment: Mountain Car Control (Section 5).

Recreates Figure 4 comparing True Online Sarsa(lambda) with standard Sarsa(lambda) variants.

NOTE: We are unable to exactly recreate the results of the original papers/scores (around -200 for true online), but we mostly match the results from Richard Suttons textbook on True Online Sarsa for the official python version of the textbook (Our shapes are the same). The cause of the difference is unclear, but it is probably something related to the tiling or env more than the math as all are results seem shifted compared to the paper, and compared to the textbook our results are a little less stable. Gemini believes the difference is because they use a Hash Table for their tiling and we use a full table, and that the Hash Table can acts as a regulerizer. Im not sure I believe this at all, but to be honest dont have many other explanations.

NOTE: the textbook uses 8x8 with 5000 steps and in the recreation code only TD with Dutch Traces (not True Online TD lambda) and does not show a divergence for TD with dutch traces at alpha = 2.0. The paper uses 10x10 with (assumed) 1000 steps and True Online TD and shows a drop but not divergence at alpha = 2.0. The paper also shows a more clear improvement of True Online TD Lamdba over the other methods. The textbook less so. The textbook code used float64 which helps to alleviate divergence issues at alpha = 2.0. It should also be noted there is a slight difference in the math between the textbook and the paper. Regardless of all this I can only for the most part seem to recreate the textbook results (i have not tested with only dutch traces and not True Online TD learning) but get a consistent divergence at alpha = 2.0 for true online TD Lambda.
"""

# TODO: clean up and make more consistent.
# TODO: try and make all examples more consistent.
# TODO: should we use the existing functions for computing td error for value and stuff?
# Removed inline TODO about updating to use existing functions
import torch
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import concurrent.futures
from typing import Tuple
from pathlib import Path

from atomic_rl.td import true_online_td_update_, semi_gradient_td_update_
from atomic_rl.traces import (
    update_true_online_traces,
    update_replacing_traces,
    update_accumulating_traces,
)
from atomic_rl.utils import compute_tile_coding_features
from atomic_rl.action_selection import with_epsilon_greedy, argmax_selector

epsilon_greedy_selector = with_epsilon_greedy(argmax_selector)

# --- CONSTANTS ---
NUM_TILINGS = 10
TILES_PER_TILING = 10
NUM_ACTIONS = 3
TOTAL_FEATURES = NUM_TILINGS * ((TILES_PER_TILING + 1) ** 2) * NUM_ACTIONS
MAX_STEPS = 1000  # Matches paper bound to allow returns down to -550+


def epsilon_greedy_action(q_values: torch.Tensor, epsilon: float) -> int:
    """Standard epsilon-greedy action selection."""
    action, _ = epsilon_greedy_selector(q_values.unsqueeze(0), epsilon, NUM_ACTIONS)
    return action.item()


def get_q_values(state: np.ndarray, weights: torch.Tensor) -> torch.Tensor:
    """Computes Q(s, a) for all actions to allow for action selection."""
    q_vals = []
    for a in range(NUM_ACTIONS):
        phi_sa = compute_tile_coding_features(
            state, a, NUM_ACTIONS, NUM_TILINGS, TILES_PER_TILING
        )
        q_vals.append(torch.dot(weights, phi_sa))
    return torch.stack(q_vals)


def true_online_sarsa_episode(
    env: gym.Env,
    weights: torch.Tensor,
    alpha: float,
    lam: float,
    gamma: float,
    epsilon: float,
) -> Tuple[float, torch.Tensor]:
    state, _ = env.reset()
    q_values = get_q_values(state, weights)
    action = epsilon_greedy_action(q_values, epsilon)
    phi_t = compute_tile_coding_features(
        state, action, NUM_ACTIONS, NUM_TILINGS, TILES_PER_TILING
    )

    traces = torch.zeros_like(weights)
    q_old = torch.tensor(0.0, dtype=torch.float64)
    episode_return = 0.0
    step_count = 0

    with torch.inference_mode():
        while True:
            next_state, reward, terminated, _, _ = env.step(action)
            episode_return += reward
            step_count += 1
            done = terminated or step_count >= MAX_STEPS

            next_q_values = get_q_values(next_state, weights)
            next_action = epsilon_greedy_action(next_q_values, epsilon)
            phi_next = compute_tile_coding_features(
                next_state, next_action, NUM_ACTIONS, NUM_TILINGS, TILES_PER_TILING
            )

            traces = update_true_online_traces(
                traces=traces.unsqueeze(0),
                features=phi_t.unsqueeze(0),
                alpha=alpha,
                gamma=gamma,
                lam=lam,
                terminated=torch.tensor([False]),
            ).squeeze(0)

            v_t = torch.dot(weights, phi_t)
            v_next = torch.dot(weights, phi_next) * (1.0 - float(terminated))
            td_error = torch.tensor(reward, dtype=torch.float64) + gamma * v_next - v_t

            true_online_td_update_(
                error=td_error,
                v_current=v_t,
                v_old=q_old,
                features=phi_t,
                weights=weights,
                alpha=alpha,
                trace=traces,
            )
            q_old = v_next

            if torch.isnan(weights).any() or torch.isinf(weights).any():
                # torch.set_printoptions(profile="full")
                print(f"Diverged at step {step_count} for True Online Sarsa")
                # print(f"Weights: {weights}")
                # print(f"Traces: {traces}")
                # print(f"Q_old: {q_old}")
                # print(f"Reward: {reward}")
                # print(f"Next Q_values: {next_q_values}")
                # print(f"Next Action: {next_action}")
                # print(f"Phi Next: {phi_next}")
                # print(f"DEBUG: phi_next sum (Active Bits): {phi_next.sum().item()}")
                # print(f"DEBUG: phi_next max value: {phi_next.max().item()}")
                # print(f"DEBUG: Weights norm: {torch.norm(weights).item()}")

                # print(f"DEBUG: prev_weights: {prev_weights}")
                # print(f"DEBUG: prev_traces: {prev_traces}")
                # print(f"DEBUG: prev_q_old: {prev_q_old}")
                # print(f"DEBUG: prev_reward: {prev_reward}")
                # print(f"DEBUG: prev_action: {prev_action}")
                # print(f"DEBUG: prev_next_action: {prev_next_action}")
                # print(f"DEBUG: prev_q_values: {prev_q_values}")
                # print(f"DEBUG: prev_phi_next: {prev_phi_next}")
                # print(
                #     f"DEBUG: prev_phi_next sum (Active Bits): {prev_phi_next.sum().item()}"
                # )
                # print(f"DEBUG: prev_phi_next max value: {prev_phi_next.max().item()}")
                # print(f"DEBUG: prev_Weights norm: {torch.norm(prev_weights).item()}")

                return -float(MAX_STEPS), weights  # Diverged, return maximum penalty

            # DEBUG
            # prev_weights = weights.clone()
            # prev_traces = traces.clone()
            # prev_q_old = q_old.clone()
            # prev_reward = reward.clone()
            # prev_action = action
            # prev_next_action = next_action
            # prev_q_values = next_q_values.clone()
            # prev_phi_next = phi_next.clone()

            if done:
                break
            state, action, phi_t = next_state, next_action, phi_next

    return episode_return, weights


def replacing_sarsa_episode(
    env: gym.Env,
    weights: torch.Tensor,
    alpha: float,
    lam: float,
    gamma: float,
    epsilon: float,
    clear_traces: bool,
) -> Tuple[float, torch.Tensor]:
    state, _ = env.reset()
    q_values = get_q_values(state, weights)
    action = epsilon_greedy_action(q_values, epsilon)
    phi_t = compute_tile_coding_features(
        state, action, NUM_ACTIONS, NUM_TILINGS, TILES_PER_TILING
    )

    traces = torch.zeros_like(weights)
    episode_return = 0.0
    step_count = 0

    with torch.inference_mode():
        while True:
            next_state, reward, terminated, _, _ = env.step(action)
            episode_return += reward
            step_count += 1
            done = terminated or step_count >= MAX_STEPS

            next_q_values = get_q_values(next_state, weights)
            next_action = epsilon_greedy_action(next_q_values, epsilon)
            phi_next = compute_tile_coding_features(
                next_state, next_action, NUM_ACTIONS, NUM_TILINGS, TILES_PER_TILING
            )

            # Explicitly clear traces for non-selected actions in the current state
            if clear_traces:
                for a in range(NUM_ACTIONS):
                    if a != action:
                        phi_other = compute_tile_coding_features(
                            state, a, NUM_ACTIONS, NUM_TILINGS, TILES_PER_TILING
                        )
                        traces[phi_other == 1.0] = 0.0

            traces = update_replacing_traces(
                traces=traces.unsqueeze(0),
                features=phi_t.unsqueeze(0),
                gamma=gamma,
                lam=lam,
                terminated=torch.tensor([False]),
            ).squeeze(0)

            v_t = torch.dot(weights, phi_t)
            v_next = torch.dot(weights, phi_next) * (1.0 - float(terminated))
            td_error = reward + gamma * v_next - v_t

            semi_gradient_td_update_(
                error=td_error,
                weights=weights,
                alpha=alpha,
                update_vector=traces,
            )

            if torch.isnan(weights).any() or torch.isinf(weights).any():
                print(
                    f"Diverged at step {step_count} for Replacing Sarsa with Clearing {clear_traces}"
                )
                return -float(MAX_STEPS), weights  # Diverged, return maximum penalty

            if done:
                break
            state, action, phi_t = next_state, next_action, phi_next

    return episode_return, weights


def accumulating_sarsa_episode(
    env: gym.Env,
    weights: torch.Tensor,
    alpha: float,
    lam: float,
    gamma: float,
    epsilon: float,
) -> Tuple[float, torch.Tensor]:
    state, _ = env.reset()
    q_values = get_q_values(state, weights)
    action = epsilon_greedy_action(q_values, epsilon)
    phi_t = compute_tile_coding_features(
        state, action, NUM_ACTIONS, NUM_TILINGS, TILES_PER_TILING
    )

    traces = torch.zeros_like(weights)
    episode_return = 0.0
    step_count = 0

    with torch.inference_mode():
        while True:
            next_state, reward, terminated, _, _ = env.step(action)
            episode_return += reward
            step_count += 1
            done = terminated or step_count >= MAX_STEPS

            next_q_values = get_q_values(next_state, weights)
            next_action = epsilon_greedy_action(next_q_values, epsilon)
            phi_next = compute_tile_coding_features(
                next_state, next_action, NUM_ACTIONS, NUM_TILINGS, TILES_PER_TILING
            )

            traces = update_accumulating_traces(
                traces=traces.unsqueeze(0),
                gradients=phi_t.unsqueeze(0),
                gamma=gamma,
                lam=lam,
                terminated=torch.tensor([False]),
            ).squeeze(0)

            v_t = torch.dot(weights, phi_t)
            v_next = torch.dot(weights, phi_next) * (1.0 - float(terminated))
            td_error = reward + gamma * v_next - v_t

            semi_gradient_td_update_(
                error=td_error,
                weights=weights,
                alpha=alpha,
                update_vector=traces,
            )

            if torch.isnan(weights).any() or torch.isinf(weights).any():
                print(f"Diverged at step {step_count} for Accumulating Sarsa")
                return -float(MAX_STEPS), weights  # Diverged, return maximum penalty

            if done:
                break
            state, action, phi_t = next_state, next_action, phi_next

    return episode_return, weights


def run_single_experiment(args):
    """Worker function for multiprocessing."""
    algo_name, alpha_0, lam, gamma, epsilon, episodes, seed = args
    alpha = alpha_0 / NUM_TILINGS

    # Unwrap environment to bypass the 200 step TimeLimit truncation
    env = gym.make("MountainCar-v0").unwrapped
    env.reset(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    weights = torch.zeros(TOTAL_FEATURES, dtype=torch.float64)
    returns = []

    for _ in range(episodes):
        if algo_name == "true_online":
            ret, weights = true_online_sarsa_episode(
                env, weights, alpha, lam, gamma, epsilon
            )
        elif algo_name == "replacing_clearing":
            ret, weights = replacing_sarsa_episode(
                env, weights, alpha, lam, gamma, epsilon, clear_traces=True
            )
        elif algo_name == "replacing_no_clearing":
            ret, weights = replacing_sarsa_episode(
                env, weights, alpha, lam, gamma, epsilon, clear_traces=False
            )
        elif algo_name == "accumulating":
            ret, weights = accumulating_sarsa_episode(
                env, weights, alpha, lam, gamma, epsilon
            )

        returns.append(ret)

        # If the run diverged, stop training and penalize remaining episodes
        if torch.isnan(weights).any() or torch.isinf(weights).any():
            remaining_episodes = episodes - len(returns)
            returns.extend([-float(MAX_STEPS)] * remaining_episodes)
            break

    env.close()
    return algo_name, alpha_0, np.mean(returns)


if __name__ == "__main__":
    # Paper Hyperparameters
    lam = 0.9
    gamma = 1.0
    epsilon = 0.0  # Optimistic zero-initialization requires no epsilon exploration
    num_runs = 10
    episodes = 20

    alphas = np.round(np.arange(0.2, 2.1, 0.2), 1)
    algos = [
        "true_online",
        "replacing_clearing",
        "replacing_no_clearing",
        "accumulating",
    ]

    # Setup job arguments
    experiments = []
    for algo in algos:
        for alpha_0 in alphas:
            for seed in range(num_runs):
                experiments.append((algo, alpha_0, lam, gamma, epsilon, episodes, seed))

    results = {algo: {a: [] for a in alphas} for algo in algos}

    print(f"Running {len(experiments)} experiments across CPU cores...")

    # Parallel execution
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(run_single_experiment, exp): exp for exp in experiments
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(experiments)
        ):
            algo_name, alpha_0, avg_return = future.result()
            results[algo_name][alpha_0].append(avg_return)

    # Aggregate results for plotting
    plot_data = {algo: [] for algo in algos}
    for algo in algos:
        for alpha_0 in alphas:
            # Average over all 100 seeds
            plot_data[algo].append(np.mean(results[algo][alpha_0]))

    print(plot_data)

    # Plotting to match Figure 4
    plt.figure(figsize=(10, 6))

    plt.plot(
        alphas,
        plot_data["true_online"],
        marker="s",
        color="#2ca02c",
        label="true online Sarsa(λ)",
    )
    plt.plot(
        alphas,
        plot_data["replacing_clearing"],
        marker="v",
        color="#d62728",
        label="Sarsa(λ), replacing, clearing",
    )
    plt.plot(
        alphas,
        plot_data["replacing_no_clearing"],
        marker="o",
        color="#1f77b4",
        label="Sarsa(λ), replacing, no clearing",
    )
    plt.plot(
        alphas,
        plot_data["accumulating"],
        marker="s",
        color="black",
        label="Sarsa(λ), accumulating",
    )

    plt.ylim(-550, -150)
    plt.xlim(0.2, 2.0)
    plt.xlabel(r"$\alpha_0$")
    plt.ylabel("return")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.title("Recreation of Sutton (2014) Figure 4 (Mountain Car Control)")
    plot_path = (
        Path(__file__).resolve().parents[2]
        / "figures"
        / "true_online_sarsa_mountain_car.png"
    )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300)

    print(f"Experiment complete! Saved plot to {plot_path}")
