# TODO compare to a standard deep learning task too.
# TODO compare to standard weight decay/shrink and perturb
# TODO why do both fail with adversarial noise. what actual RL problem would have that?
# TODO improvement is so little find a better task.
# TODO/NOTE: i think the math of True Online TD(lambda) doesnt work with the feature wise alphas (like for IDBD) because of the eligibility trace update rule for it. It should work for TD(0) though. if we want to use True Online TD(lambda) for this, we need to find a way to make it work, something like AdaGain or TIDBD or like MetaTraces or something.
# NOTE: we may be using a semi gradient method at the moment for IDBD with TD(0) the more correct version would be TIDBD.
"""
Idea of this example is to attempt to combine the Alberta Plan Related papers i have created so far.

Step 1:
Online Normalization: [atomic_rl/utils.py]
Meta-Learned Step-Sizes: IDBD or AutoStep [atomic_rl/optimizer/metaoptimization/]
Feature Relevance Tracking: This is handled by both Meta Optimization Methods (IDBD and AutoStep) and Generate and Test methods (CBP and SWR).
    In CBP/SWR, it's the utilities tensor (tracking how much a feature contributes to the output).
    In IDBD/AutoStep, it's the h trace (tracking the correlation of recent gradients).
Generate-and-Test Mechanics: [atomic_rl/plasticity.py] (e.g. SWR, CBP). In combination with Meta Optimization Methods high learning rate features are not pruned, low learning rate features are pruned, and a new feature is generated.
Resource Budgeting: Controlled by: k (SWR) or replacement_rate (CBP).

"""

# TODO: should this use ema?
from atomic_rl.initialization import make_gnt_init
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from atomic_rl.td import true_online_td_update_, semi_gradient_td_update_
from atomic_rl.traces import update_true_online_traces
from atomic_rl.optimizer.metaoptimization import (
    update_autostep_rates_,
    update_idbd_rates_,
)
from atomic_rl.plasticity import apply_continual_backprop, init_cbp_state

# Configuration
NUM_STATES = 19
START_STATE = 9
NUM_STEPS = 500000
PERMUTATION_INTERVAL = 2000
TRUE_VALUES = torch.tensor([(i + 1) / (NUM_STATES + 1) for i in range(NUM_STATES)])


class PermutedDriftingRandomWalk:
    def __init__(self, num_states=19, start_state=9, observation_size=100):
        self.num_states = num_states
        self.start_state = start_state
        self.state = self.start_state
        self.observation_size = observation_size

        # Initialize active_channels as the first num_states channels
        self.active_channels = torch.arange(self.num_states)
        # Remaining are noise channels
        self.noise_channels = torch.arange(self.num_states, self.observation_size)

    def reset(self):
        self.state = self.start_state
        return self._get_features(self.state)

    def step(self):
        # Random transition
        self.state += 1 if torch.rand(1).item() > 0.5 else -1

        terminated = False
        reward = 0.0

        if self.state == self.num_states:  # Right terminal
            reward = 1.0
            terminated = True
        elif self.state == -1:  # Left terminal
            reward = 0.0
            terminated = True

        next_features = (
            self._get_features(self.state)
            if not terminated
            else torch.zeros(self.observation_size)
        )

        return next_features, reward, terminated

    def _get_features(self, state):
        features = torch.zeros(self.observation_size)
        # Set the active channel corresponding to the current state to 1.0
        if 0 <= state < self.num_states:
            features[self.active_channels[state]] = 1.0

        # Add random normal noise to the noise channels
        features[self.noise_channels] = torch.randn(len(self.noise_channels)) * 0.1
        return features

    def permute_features(self):
        # Remap 10% of the active channels (e.g. 2 channels) with random noise channels to simulate drift in usefulness
        num_to_swap = max(1, int(self.num_states * 0.10))

        active_swap_idx = torch.randperm(self.num_states)[:num_to_swap]
        noise_swap_idx = torch.randperm(len(self.noise_channels))[:num_to_swap]

        # Swap their physical channels
        temp = self.active_channels[active_swap_idx].clone()
        self.active_channels[active_swap_idx] = self.noise_channels[noise_swap_idx]
        self.noise_channels[noise_swap_idx] = temp


class RepresentationNetwork(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        a1, a2 = self.get_activations(x)
        return a2

    def get_activations(self, x):
        a1 = F.relu(self.fc1(x))
        a2 = F.relu(self.fc2(a1))
        return a1, a2


def get_rms_error(backbone, theta, env):
    # Calculate RMS Error over all possible states
    with torch.inference_mode():
        preds = []
        for s in range(NUM_STATES):
            # We evaluate clean state features (no noise) to get accurate prediction value
            features = torch.zeros(env.observation_size)
            features[env.active_channels[s]] = 1.0

            rep = backbone(features)
            v = torch.dot(rep, theta)
            preds.append(v)
        preds = torch.stack(preds)
        return torch.sqrt(torch.mean((preds - TRUE_VALUES) ** 2)).item()


def run_experiment(config: str):
    torch.manual_seed(42)
    env = PermutedDriftingRandomWalk(NUM_STATES, START_STATE)

    meta_lr = 1e-2
    init_lr = 0.05
    gamma = 1.0
    lam = 0.0 if config in ["B", "C", "D", "E", "F"] else 0.0

    # Model: Capacity Bottleneck (hidden_size=128, out_features=128)
    backbone = RepresentationNetwork(100, 8, 8)
    theta = torch.empty(8, requires_grad=False)
    nn.init.uniform_(theta, -0.01, 0.01)

    opt = torch.optim.Adam(backbone.parameters(), lr=1e-3)

    use_autostep = config in ["C", "D"]
    use_idbd = config in ["E", "F"]
    use_meta = use_autostep or use_idbd

    if use_meta:
        betas_head = torch.full_like(theta, math.log(init_lr))
        h_head = torch.zeros_like(theta)
        v_head = torch.zeros_like(theta) if use_autostep else None
        alphas_head = torch.full_like(theta, init_lr)
    else:
        alphas_head = torch.full_like(theta, init_lr)

    traces = torch.zeros_like(theta)

    # Plasticity setup (Config B, D, F use CBP)
    use_plasticity = config in ["B", "D", "F"]
    if use_plasticity:
        cbp_states = {
            backbone.fc1.weight: init_cbp_state(backbone.fc1.weight),
            backbone.fc2.weight: init_cbp_state(backbone.fc2.weight),
        }
        layer_pairs = [
            (
                backbone.fc1.weight,
                backbone.fc1.bias,
                backbone.fc2.weight,
                backbone.fc2.bias,
            ),
            (backbone.fc2.weight, backbone.fc2.bias, theta.unsqueeze(0), None),
        ]
        init_fn = make_gnt_init(lambda t: nn.init.kaiming_uniform_(t, a=math.sqrt(5)))
    else:
        cbp_states = None
        layer_pairs = None
        init_fn = None

    rms_errors = []

    v_old = 0.0
    features_t = env.reset()

    for step in tqdm(range(1, NUM_STEPS + 1), desc=f"Running Config {config}"):
        if step % PERMUTATION_INTERVAL == 0:
            env.permute_features()
            if use_autostep:
                print(
                    f"[{config} Step {step}] Post-Permutation Alphas (Mean): {alphas_head.mean().item():.5f}"
                )

        with torch.inference_mode():
            a1_t, rep_t = backbone.get_activations(features_t)
            v_current = torch.dot(theta, rep_t)

        if step == 1 or features_t.sum() == 0:
            v_old = v_current

        a1_t_grad, rep_t_grad = backbone.get_activations(features_t)
        v_current_grad = torch.dot(theta, rep_t_grad)

        next_features, reward, terminated = env.step()

        with torch.inference_mode():
            rep_next = (
                backbone(next_features) if not terminated else torch.zeros_like(theta)
            )
            v_next_target = (
                torch.dot(theta, rep_next) if not terminated else torch.tensor(0.0)
            )

        # --- Head Update ---
        error_td = reward + gamma * v_next_target - v_current

        if use_autostep:
            # TODO: Pass raw features, not traces, to AutoStep. The meta-optimizer needs to track correlations of the actual state representation so the per-feature alpha multipliers are well-defined. OR When using TD(lambda), the weight update is driven by the eligibility trace, not the raw features. We must pass traces to the meta-optimizer to ensure the meta-gradients are calculated w.r.t the actual update vector.
            # TODO: TIDBD? or does Autostep work?
            alphas_head = update_autostep_rates_(
                betas=betas_head,
                h=h_head,
                v=v_head,
                inputs=rep_t.detach(),
                error=error_td.detach(),
                meta_lr=meta_lr,
            )
        elif use_idbd:
            # NOTE: Pass raw features, not traces, to IDBD. Same rationale as AutoStep.
            alphas_head = update_idbd_rates_(
                betas=betas_head,
                h=h_head,
                inputs=rep_t.detach(),
                error=error_td.detach(),
                meta_lr=meta_lr,
            )

        if lam > 0.0:
            traces = update_true_online_traces(
                traces=traces.unsqueeze(0),
                features=rep_t.detach().unsqueeze(0),
                alpha=alphas_head,
                gamma=gamma,
                lam=lam,
                terminated=torch.tensor([terminated]),
            ).squeeze(0)

            # TODO: does this work with AutoStep alphas?
            true_online_td_update_(
                error=error_td.detach(),
                v_current=v_current.detach(),
                v_old=v_old,
                features=rep_t.detach(),
                weights=theta,
                alpha=alphas_head,
                trace=traces,
            )
            v_next_for_td = v_next_target.detach()
        else:
            # TD(0) update
            semi_gradient_td_update_(
                error=error_td.detach(),
                weights=theta,
                alpha=alphas_head,
                update_vector=rep_t.detach(),
            )
            v_next_for_td = v_next_target

        # --- Backbone Update ---
        loss = F.mse_loss(v_current_grad, (reward + gamma * v_next_target).detach())
        opt.zero_grad()
        loss.backward()

        # Check gradient norm to ensure plasticity injection doesn't kill the computation graph
        if step % 10000 == 0:
            if backbone.fc2.weight.grad is not None:
                grad_norm = backbone.fc2.weight.grad.norm().item()
                # Print if grad is zero or NaN
                if grad_norm == 0.0 or math.isnan(grad_norm):
                    print(
                        f"WARNING: [{config} Step {step}] backbone.fc2.weight.grad norm is abnormal: {grad_norm:.6f}!"
                    )
                else:
                    print(
                        f"[{config} Step {step}] backbone.fc2.weight.grad norm: {grad_norm:.6f}"
                    )
            else:
                print(
                    f"WARNING: [{config} Step {step}] backbone.fc2.weight.grad is None!"
                )

        if use_plasticity:
            # TODO: is theta is updated out-of-place? we assume it is so we dynamically reconstruct
            # the layer pairs to avoid a "stale reference" bug where CBP refers to the step 0 theta.
            current_layer_pairs = [
                (
                    backbone.fc1.weight,
                    backbone.fc1.bias,
                    backbone.fc2.weight,
                    backbone.fc2.bias,
                ),
                (backbone.fc2.weight, backbone.fc2.bias, theta.unsqueeze(0), None),
            ]
            replacement_masks = apply_continual_backprop(
                layer_pairs=current_layer_pairs,
                activations=[a1_t_grad.unsqueeze(0), rep_t_grad.unsqueeze(0)],
                cbp_states=cbp_states,
                optimizer=opt,
                init_fn=init_fn,
                maturity_threshold=500,
                replacement_rate=1e-4,
            )

            replacement_mask = replacement_masks.get(backbone.fc2.weight, None)

            if replacement_mask is not None and replacement_mask.any():
                num_replaced = replacement_mask.sum().item()
                # print(
                #     f"[{config} Step {step}] CBP replaced {num_replaced} features in fc2 (Maturity threshold: 500)"
                # )
                with torch.no_grad():
                    # Reinitialize replaced theta weights to break the zero-gradient/zero-utility loop
                    temp_theta = torch.empty_like(theta)
                    nn.init.uniform_(temp_theta, -0.01, 0.01)
                    theta[replacement_mask] = temp_theta[replacement_mask]

                    traces[replacement_mask] = 0.0
                    if use_meta:
                        h_head[replacement_mask] = 0.0
                        if use_autostep:
                            v_head[replacement_mask] = 0.0
                        betas_head[replacement_mask] = math.log(init_lr)
                        alphas_head[replacement_mask] = init_lr

        opt.step()

        if terminated:
            features_t = env.reset()
        else:
            features_t = next_features
            v_old = v_next_for_td

        if step % 500 == 0:
            rms_errors.append(get_rms_error(backbone, theta, env))

        if step % 10000 == 0 and use_meta:
            print(
                f"[{config} Step {step}] alphas_head - mean: {alphas_head.mean().item():.5f}, max: {alphas_head.max().item():.5f}, min: {alphas_head.min().item():.5f}"
            )
            print(
                f"[{config} Step {step}] h_head      - max: {h_head.max().item():.5f}, min: {h_head.min().item():.5f}"
            )

            # Magnitude checks
            meta_grad = error_td.detach() * traces.detach()
            meta_grad_mag = torch.mean(torch.abs(meta_grad)).item()
            print(
                f"[{config} Step {step}] IDBD/AutoStep meta-gradient magnitude (abs mean): {meta_grad_mag:.5f}"
            )
            if use_plasticity and cbp_states is not None:
                cbp_util_mag = torch.mean(
                    torch.abs(cbp_states[backbone.fc2.weight]["utilities"])
                ).item()
                print(
                    f"[{config} Step {step}] CBP utilities magnitude (abs mean):  {cbp_util_mag:.5f}"
                )

    return rms_errors


def main():
    # TODO: B and C perform the EXACT same. that should not be the case right? IDBD should change something? also none seems to be particularly better.
    configs = {
        "A": "Fixed NN + TD(0)",
        # "B": "CBP NN + TD(lambda)",
        # "C": "Fixed NN + TD(lambda) + AutoStep",
        "D": "CBP NN + TD(lambda) + AutoStep",
        # "E": "Fixed NN + TD(lambda) + IDBD",
        # "F": "CBP NN + TD(lambda) + IDBD",
    }

    results = {}
    for code, desc in configs.items():
        results[desc] = run_experiment(code)

    # Plotting Dashboard
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    x_axis = np.arange(1, len(results[configs["A"]]) + 1) * 500

    # 1. Value Function RMS Error Over Time
    for desc, errors in results.items():
        ax1.plot(x_axis, errors, label=desc, alpha=0.8, linewidth=2)

    for i in range(PERMUTATION_INTERVAL, NUM_STEPS, PERMUTATION_INTERVAL):
        ax1.axvline(x=i, color="r", linestyle="--", alpha=0.15)

    ax1.set_xlabel("Environment Steps", fontsize=12)
    ax1.set_ylabel("Value Function RMS Error", fontsize=12)
    ax1.set_title(
        "Ablation Study: Value Function RMS Error", fontsize=14, fontweight="bold"
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Cumulative Regret (Integral of RMS Error Over Time)
    for desc, errors in results.items():
        # Cumulative regret is the integral of the error over time
        cumulative_regret = np.cumsum(errors) * 500
        ax2.plot(x_axis, cumulative_regret, label=desc, alpha=0.8, linewidth=2)

    for i in range(PERMUTATION_INTERVAL, NUM_STEPS, PERMUTATION_INTERVAL):
        ax2.axvline(x=i, color="r", linestyle="--", alpha=0.15)

    ax2.set_xlabel("Environment Steps", fontsize=12)
    ax2.set_ylabel("Cumulative Regret (Integral of RMS Error)", fontsize=12)
    ax2.set_title("Ablation Study: Cumulative Regret", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(__file__).resolve().parents[2] / "figures" / "ablation_results.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Results saved to {save_path.absolute()}")


if __name__ == "__main__":
    main()
