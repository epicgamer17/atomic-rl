"""
Reproduction of Sutton (1988): "Learning to Predict by the Methods of Temporal Differences"
Experiment: The Random Walk (Section 3).

Essentially introduces TD learning as an alternative to more traditional batch based backprop learning. Instead of learning on batches, we learn online 1 step at a time. It saves compute and saves space, among other benefits.

TD methods assign credit based on the difference between temporally successive predictions, rather than waiting for the final outcome. This allows spreading out the computation across time steps, it also allows adjusting predictions on the fly before a reward is even received instead of waiting for the entire episode to finish.

It also has the benefit that it doesn't suffer from monte carlo methods high variance, and less accurate credit assignment. The example given is you enter a state that loses 90% of the time and wins 10% of the time. Lets say for example we reach this state and end up winning. With standard "supervised" learning we would move our prediction towards 1, which is incorrect if it in fact usually loses. With TD learning we have move the target of the novel state leading to that state toward the value of the next state (or the bad 90% lose state) instead of the value of the result of the game. This is arguably more correct. It is "uncorrupted by random factors that affect the result of the game".

"This game-playing example can also be used to show how TD methods can fail. Suppose the bad state is usually followed by defeats except when it is preceded by the novel state, in which case it always leads to a victory. In this odd case, TD methods could not perform better and might perform worse than supervised-learning methods. Although there are several techniques for eliminating or minimizing this sort of problem, it remains a greater difficulty for TD methods than it does for supervised-learning methods. TD methods try to take advantage of the information provided by the temporal sequence of states, whereas supervised-learning methods ignore it."

"Finally, note that although this example involved learning an evaluation function, nothing about it was specific to evaluation functions. The methods can equally well be used to predict outcomes unrelated to the player's goals, such as the number of pieces left at the end of the game. If TD methods are more efficient than supervised-learning methods in learning evaluation functions, then they should also be more efficient in general prediction-learning problems." (TODO an example of TD-learning to predict stuff unrelated to the agent's goals, not just outcome)

TD(lambda) connects 1-step TD(0) to Monte Carlo TD(1) via an eligibility trace. The update rule is: delta_w = alpha * (P_{t+1} - P_t) * sum_{k=1}^t lambda^{t-k} nabla_w P_k
Implemented via Eligibility Traces (Backward View).

In this form TD learning can only be used with Linear function approximation or the tabular case. Additionally, it must be online. Furthermore, in this example and in the paper TD learning was only used for state values. It was later expanded to learn Q values, again in a strictly online and linear function approximation way. And finally DQN expanded it to the offline non-linear case.

Other TD papers expand TD learning to be more suitable for deep learning/approximate settings as well.

TODO: is this the foundation for stream RL or can stream RL practices work well without TD learning? eg stream RL with gradient descent or something?
TODO: possible example or experiment of this on mining with Navarra? ie mine optimization or mine planning or simulation/prediction might work a little bit in a stream fashion. Is TD learning good for this? Is stream RL good for this? What are comparisons of Stream RL vs TD learning vs Traditional RL in general and on the mine simulation/optimization problem? What exactly is TD learning. I still need to wrap my head around it and where it falls in the scheme of things. Could you effectively say the TD learning works for any problem by essentially instead of predicting V(s) you try to predict the value of any feature? So could you use TD Learning for predicting mine throughput given current state and inputs as other features? Is the mine optimization or planning problem a nonstationairy continual learning problem or can it be modeled as a stationary episodic problem? What are the pros and cons of this?

NOTE: This is a simple almost tabular example of TD(lambda).
"""

# TODO: dont inline the env/ or state-space logic into the functions/scripts, use a proper env or put it elsewhere so it can be reused.
# TODO: should we use the existing functions for computing td error for value and stuff?

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple

from envs.mdp.random_walk import RandomWalkEnv
from functional.td import semi_gradient_td_update
from functional.traces import update_accumulating_traces

# --- CONSTANTS ---
NUM_NON_TERMINAL_STATES = 5
START_STATE = 2  # State D (0:B, 1:C, 2:D, 3:E, 4:F)
INITIAL_WEIGHT = 0.5
TRUE_VALUES = torch.tensor([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])

# Experiment parameters for Figure 3 (Online)
LAMBDAS_ONLINE = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
ALPHAS_ONLINE = np.linspace(0, 0.6, 13)

# Experiment parameters for Figure 4 (Batch/Repeated)
LAMBDAS_BATCH = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
ALPHA_BATCH = 0.01  # Small alpha for convergence
CONVERGENCE_THRESHOLD = 1e-6
MAX_BATCH_ITERATIONS = 1000

NUM_TRAINING_SETS = 100
EPISODES_PER_SET = 10
SEED = 42

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def generate_training_sets(
    num_sets: int, episodes_per_set: int
) -> List[List[List[Tuple[torch.Tensor, float, torch.Tensor, bool]]]]:
    """
    Generates training sets for the random walk.
    Each set contains a list of episodes.
    Each episode is a list of (phi_t, reward, phi_next, terminated) transitions.
    """
    env = RandomWalkEnv(num_states=NUM_NON_TERMINAL_STATES, start_state=START_STATE)
    training_sets = []
    for _ in range(num_sets):
        episodes = []
        for _ in range(episodes_per_set):
            episode = []
            phi_t = env.reset()
            while True:
                phi_next, reward, terminated = env.step()
                episode.append((phi_t, reward, phi_next, terminated))
                if terminated:
                    break
                phi_t = phi_next
            episodes.append(episode)
        training_sets.append(episodes)
    return training_sets


def random_walk_episode(
    weights: torch.Tensor, alpha: float, lam: float
) -> torch.Tensor:
    """
    Runs a single random walk episode and updates weights online using TD(lambda).

    Args:
        weights: Current weight vector [num_states].
        alpha: Learning rate.
        lam: Trace decay rate (lambda).

    Returns:
        Updated weights after one episode.
    """
    env = RandomWalkEnv(num_states=NUM_NON_TERMINAL_STATES, start_state=START_STATE)
    phi_t = env.reset()
    traces = torch.zeros_like(weights)

    with torch.inference_mode():
        while True:
            # 1. Take a step in the environment
            phi_next, reward, terminated = env.step()

            # 2. Update Eligibility Trace: e_t = gamma * lambda * e_{t-1} + phi_t
            # Note: Gamma is 1.0 for this task.
            # update_accumulating_traces expects [batch, features]
            traces = update_accumulating_traces(
                traces.unsqueeze(0),
                phi_t.unsqueeze(0),
                gamma=1.0,
                lam=lam,
                terminated=torch.tensor([False]),  # Only clear trace after episode
            ).squeeze(0)

            # 3. Compute TD error and Apply semi-gradient Update
            v_t = torch.dot(weights, phi_t)
            v_next = torch.dot(weights, phi_next) * (1.0 - float(terminated))
            td_error = reward + 1.0 * v_next - v_t

            # weights = weights + alpha * delta * traces
            weights = semi_gradient_td_update(
                error=td_error,
                weights=weights,
                alpha=alpha,
                update_vector=traces,
            )

            if terminated:
                break

            phi_t = phi_next

    return weights


def run_online_experiment():
    """
    Runs the online experiment (Experiment 2) to replicate Figure 3
    and Figure 5 from Sutton (1988).
    """
    results = {lam: [] for lam in LAMBDAS_ONLINE}

    print(f"Running Experiment 2 (Online Updates - Figure 3/5)...")

    for lam in LAMBDAS_ONLINE:
        print(f"Testing lambda = {lam}")
        for alpha in ALPHAS_ONLINE:
            total_rms_error = 0.0

            for _ in range(NUM_TRAINING_SETS):
                weights = torch.full((NUM_NON_TERMINAL_STATES,), INITIAL_WEIGHT)
                for _ in range(EPISODES_PER_SET):
                    weights = random_walk_episode(weights, alpha, lam)

                rms_error = torch.sqrt(torch.mean((weights - TRUE_VALUES) ** 2)).item()
                total_rms_error += rms_error

            avg_rms_error = total_rms_error / NUM_TRAINING_SETS
            results[lam].append(avg_rms_error)

    plot_results(
        results,
        ALPHAS_ONLINE,
        "Replication of Sutton (1988) Figure 3",
        "random_walk_fig3.png",
        xlabel=r"$\alpha$",
    )


def run_batch_experiment():
    """
    Runs the batch experiment (Experiment 1: Repeated Presentation)
    to replicate Figure 4 from Sutton (1988).

    Instead of iterating with a small alpha, we solve for the convergence point
    of the weight vector directly using the linear system Aw = b.
    """
    training_sets = generate_training_sets(NUM_TRAINING_SETS, EPISODES_PER_SET)
    results = []

    print(f"Running Experiment 1 (Batch/Repeated Presentation - Figure 4)...")

    for lam in LAMBDAS_BATCH:
        print(f"Testing lambda = {lam}")
        total_rms_error = 0.0

        for episodes in training_sets:
            # TODO: have we inlined something here that should be in td.py in the functional folder?
            # We want to find w such that sum(delta_t * e_t) = 0
            # sum( (r_t + gamma * w.T * phi_{t+1} - w.T * phi_t) * e_t ) = 0
            # sum( r_t * e_t ) + sum( e_t * (gamma * phi_{t+1} - phi_t).T * w ) = 0
            # sum( e_t * (phi_t - gamma * phi_{t+1}).T ) * w = sum( r_t * e_t )
            # A * w = b

            A = torch.zeros((NUM_NON_TERMINAL_STATES, NUM_NON_TERMINAL_STATES))
            b = torch.zeros(NUM_NON_TERMINAL_STATES)

            for episode in episodes:
                traces = torch.zeros(NUM_NON_TERMINAL_STATES)
                for phi_t, reward, phi_next, terminated in episode:
                    # Update traces: e_t = gamma * lam * e_{t-1} + phi_t
                    traces = update_accumulating_traces(
                        traces.unsqueeze(0),
                        phi_t.unsqueeze(0),
                        gamma=1.0,
                        lam=lam,
                        terminated=torch.tensor([False]),
                    ).squeeze(0)

                    # Accumulate A and b
                    # Note: gamma is 1.0 for this task
                    phi_diff = phi_t - (
                        phi_next if not terminated else torch.zeros_like(phi_next)
                    )

                    # A += e_t * (phi_t - gamma * phi_{next})^T
                    A += torch.outer(traces, phi_diff)
                    # b += r_t * e_t
                    b += reward * traces

            # Solve Aw = b for the converged weight vector
            # We use pseudo-inverse in case A is singular (rare for this task)
            weights = torch.linalg.pinv(A) @ b

            rms_error = torch.sqrt(torch.mean((weights - TRUE_VALUES) ** 2)).item()
            total_rms_error += rms_error

        avg_rms_error = total_rms_error / NUM_TRAINING_SETS
        results.append(avg_rms_error)

    plot_batch_results(results)


def plot_results(
    results: dict, x_values: np.ndarray, title: str, filename: str, xlabel: str
):
    """
    General plotting function for Figure 3.
    """
    plt.figure(figsize=(10, 6))

    for lam, errors in results.items():
        valid_indices = [i for i, error in enumerate(errors) if error < 0.7]
        plt.plot(
            [x_values[i] for i in valid_indices],
            [errors[i] for i in valid_indices],
            marker="o",
            label=r"$\lambda$ = " + str(lam),
        )

    plt.xlabel(xlabel)
    plt.ylabel("ERROR (RMS)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plot_path = f"/Users/jonathanlamontange-kratz/Documents/GitHub/rl-stuff/{filename}"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.show()


def plot_batch_results(results: List[float]):
    """
    Plots the results for Figure 4.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(LAMBDAS_BATCH, results, marker="o", linestyle="-", color="black")

    plt.xlabel(r"$\lambda$")
    plt.ylabel("ERROR (RMS)")
    plt.title("Replication of Sutton (1988) Figure 4\n(Repeated Presentation)")
    plt.grid(True, linestyle="--", alpha=0.7)

    plot_path = (
        "/Users/jonathanlamontange-kratz/Documents/GitHub/rl-stuff/random_walk_fig4.png"
    )
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    run_online_experiment()
    run_batch_experiment()
