[![Interactive Research Labs](https://img.shields.io/badge/Live_Demos-Interactive_Research_Labs-blue?style=for-the-badge&logo=flask)](https://kratzj.vercel.app/labs)
[![Website](https://img.shields.io/badge/Website-kratzj.vercel.app-10b981?style=for-the-badge&logo=vercel)](https://kratzj.vercel.app)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/epicgamer17)
[![Patreon](https://img.shields.io/badge/Patreon-F1465A?style=for-the-badge&logo=patreon&logoColor=white)](https://www.patreon.com/epicgamer17)

# Modular RL (Functional)

A high-performance, researcher-centric Reinforcement Learning library for PyTorch built on the **Functional Core, Imperative Shell** design pattern. Think of it as **RLax for PyTorch**—engineered to eliminate the rigidity of deep OOP frameworks and the chaos of monolithic single-file copy-pasting.

---

## 📦 Installation

```bash
pip install atomic-rl
```

Or, to run the examples and experiments directly from a clone of this repository:

```bash
git clone https://github.com/epicgamer17/modular-rl.git
cd modular-rl
pip install -e ".[envs,examples,plot,test]"
```

## 🚀 Quickstart

The library ships as three flat, top-level import packages: `atomic_rl` (pure mathematical primitives), `networks` (network building blocks), and `envs` (environments, streams, and wrappers).

```python
import torch
from atomic_rl.initialization import layer_init_, set_seed
from atomic_rl.action_selection import argmax_selector, with_epsilon_greedy
from atomic_rl.td import compute_v_td_target
from networks.resnet import ResNetBlock

set_seed(42)

# One-step TD value targets: r + gamma * V(s') * (1 - done)
next_values = torch.tensor([3.0, 4.0])
rewards = torch.tensor([0.5, 1.0])
terminated = torch.tensor([0.0, 1.0])
gamma = torch.tensor([0.9, 0.9])
targets = compute_v_td_target(next_values, rewards, terminated, gamma)
# -> tensor([3.2, 1.0])

# Greedy action selection with epsilon-greedy wrapper over Q-values [B, A]
select = with_epsilon_greedy(argmax_selector)
q_values = torch.randn(1, 4)
actions, _ = select(predictions=q_values, epsilon=0.1, num_actions=4,
                    generator=torch.Generator().manual_seed(0))
```

Complete, runnable algorithms (DQN, PPO, A2C, AlphaZero, MuZero, TD-learning, and more) live in the [`examples/`](examples/) directory.

---

## 💡 The Philosophy: Resolving the RL Paradigm War

Developing Reinforcement Learning algorithms typically forces researchers to choose between three flawed paradigms:

1. **The Monolith (Single-File Implementations):** 
   * *Pros:* Everything is in one local scope; logging, file-diffing, and fast prototyping are frictionless.
   * *Cons:* Combinatorial explosion of parameters and massive `if/else` configuration trees (e.g., handling frame stacking, continuous actions, recurrent states). Fixing a bug in one DQN implementation does not propagate to others, leading to destructive **feature drift**. Moving to multi-GPU (DDP) or TPU injections pollutes pure mathematical logic with system-level engineering blocks. Unit testing a loop monolith is practically impossible, forcing reliance on slow, flaky integration tests.
2. **Standard OOP & Strategy Patterns:**
   * *Pros:* Promotes abstraction and modular reuse.
   * *Cons:* Hides design details and heavily relies on internal state mutations (`self._step_count`, `self._hidden_state`), creating subtle off-by-one errors and tracking nightmares. To customize a single feature, you must master deep parent interfaces, inheriting a mountain of hidden knowledge debt. Mixing orthogonal features causes a combinatorial explosion of classes (e.g., `RecurrentContinuousPPO`), or God Classes. Strict interfaces strip away critical paper-specific mathematical optimizations to stay general.
3. **Execution Graphs & DAGs:**
   * *Pros:* Maximizes component reuse and layout validation.
   * *Cons:* Deep data-flow graphs aggressively reject standard Python dynamic control flows (like nested dynamic `while` loops inside an MCTS tree search), forcing clunky graph operators. Building a graph system creates immense engineering overhead, resulting in 10–15 node classes for a baseline algorithm, which damages execution speed in PyTorch due to structural dictionary-passing overhead.

### 🛠️ The Solution: Functional Core, Imperative Shell

This repository maps out a clean compromise. We isolate mathematical and algorithmic actions into a **Functional Core** composed of stateless, pure, side-effect-free functions. We then assemble these primitives inside an easy-to-read, linear, monolithic loop—the **Imperative Shell**. This allows our implementations to benefit from the monolithic paradigm, while allowing code reuse, modularity, and testing. Since each function is pure and has a simple interface, and (for the most part) functions don't use other functions internally, it is trivial to read and understand the codebase without getting lost in abstractions or having to dive deep into the codebase. 

```python
# --- 1. Initialization (Defining the State) ---
params = init_network()
optimizer_state = init_optimizer()
buffer_state = init_buffer(capacity=10000)
env_state, obs = env.reset()
hidden_state = init_rnn_state()

# --- 2. The Monolithic Loop (The Imperative Shell) ---
for step in range(MAX_STEPS):
    # 1. Act (Pure function)
    action, next_hidden_state = select_action(params, obs, hidden_state)
    
    # 2. Step Env (Pure-ish function adapter)
    next_env_state, next_obs, reward, done = env.step(env_state, action)
    
    # 3. Add to Buffer (Pure state mutation)
    transition = (obs, action, reward, hidden_state)
    buffer_state = add_to_buffer(buffer_state, transition)
    
    # Update loop states
    obs, env_state, hidden_state = next_obs, next_env_state, next_hidden_state
    
    # --- 3. The Functional Update Core ---
    if step % UPDATE_FREQ == 0:
        batch, rng_key = sample_buffer(buffer_state, rng_key, BATCH_SIZE)
        
        # Calculate Loss & Gradients via Pure Math
        loss, grads = calculate_loss(params, batch)
        params, optimizer_state = apply_gradients_(params, optimizer_state, grads)
        
        # Monolithic layout makes logging and tracking effortless
        wandb.log({"loss": loss, "step": step})
```

## ⚡ Core Engineering Practices (TODO: improve this section, add more of our actual rules in CONTRIBUTING.md)
1. Explicit Over Implicit
We avoid magic configuration objects, automated parameter routing, and hidden global variables. High-level orchestration functions are stateless and clear. You pass tensors explicitly, enabling seamless tracking and debugging.

2. Fail Fast & Guardrails
We utilize strong type hints, strict validation checks, and inline assertions at the boundaries of the functional layers. Shape mismatches, device conflicts, and invalid boundaries throw exceptions at their origin point—not deep inside a backend compiled gradient execution.

3. Documentation by Signature
Functions are structured so that a researcher can easily interpret the underlying math just by inspecting the name, inputs, and type annotations, removing the need to trace internal source operations across 15 separate tracking scripts.

4. PyTorch Native Performance & Conventions
torch.compile Friendly: Pure functions minimize internal state and dictionary unpacks, enabling the compiler to run graph optimizations across the mathematical core.

No CUDA Synchronizations: Absolutely zero internal calls to .item(), .tolist(), or .numpy() inside the computational blocks. This keeps the CPU and GPU timelines decoupled, eliminating latency bubbles.

Device & Dtype Agnostic: We never hardcode strings like device='cuda'. When initializing tensors, we use factory methods like .new_zeros() or torch.zeros_like() relative to the incoming tensor states to protect against distributed data-parallel breaks.

Explicit Dimensions: PyTorch broadcasting can mask bugs. We favor explicit .unsqueeze() calls, allowing code readers to visually audit array matching.