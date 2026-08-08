# Modular RL Testing Suite

Reinforcement Learning bugs are rarely loud crashes; they are usually silent math errors—a detached gradient, a broadcast error, or a faulty advantage estimation—that manifest simply as "the agent isn't learning." 

This testing suite is designed to rigorously verify the underlying logic and mathematics of the codebase without requiring full training loops for every commit.

## 📁 Organization Philosophy

This directory uses a **hybrid organization strategy**:
1. **Unit Tests** strictly mirror the source code directory structure.
2. **Integration, Smoke, and Regression Tests** are organized by algorithm or feature.

```text
tests/
├── conftest.py                             # Global PyTorch fixtures (dummy tensors, seeded envs)
│
├── unit/                                   # Tier 1: strictly mirrors the source tree
│   ├── agents/                             #   ↳ e.g., tests/unit/agents/learner/atomic_rl/
│   ├── modules/                            #   ↳ e.g., tests/unit/modules/heads/
│   ├── replay_buffers/                     #   ↳ e.g., tests/unit/replay_buffers/
│   └── search/                             #   ↳ e.g., tests/unit/search/aos_search/
│
├── integration/                            # Tier 2: Organized by system interaction
│   ├── test_compilation.py                 #   ↳ Ensures torch.compile compatibility
│   └── test_loss_pipeline.py               #   ↳ Modules passing gradients correctly
│
├── smoke/                                  # Tier 3: Organized by execution target
│   ├── test_local_executor.py              #   ↳ Fast 2-epoch local loop
│   └── test_torch_mp_executor.py           #   ↳ Fast 2-epoch multiprocessing loop
│
├── regression/                             # Tier 4: Organized by algorithm/paper
│   ├── ppo/                                #   ↳ Verifies PPO's 37 implementation details
│   └── muzero/                             #   ↳ Verifies MuZero can solve tic-tac-toe
│
├── performance/                            # Tier 5: Throughput and Efficiency
│   ├── micro/                              # Isolated component speed (e.g., Replay Buffer inserts)
│   ├── sub_systems/                        # Interaction speed (e.g., MCTS + Neural Net)
│   └── scaling/                            # Concurrency (e.g., Ray workers, multi-threading)
│
└── parity/                                 # Tier 6: Output Equivalence
    ├── backends/                           # e.g., C++ MCTS vs Python MCTS
    └── external_libs/                      # e.g., Custom PPO loss vs StableBaselines3

🚥 The 6-Tier Testing Pipeline
To maintain high developer velocity while ensuring mathematical rigor, tests are divided into five tiers based on execution time and determinism:

Tier 1: Unit (Core Math & Logic)
Scope: Isolated functions (Segment trees, GAE, MuZero MinMax stats, tensor shapes).

Execution: Runs in milliseconds.

CI Target: Runs on every single push.

Tier 2: Integration (Component Wiring)
Scope: Ensures modules wire together correctly without actual training (e.g., passing a dummy batch through the LossAggregator, testing Pufferlib environment adapters).

Execution: Runs in seconds.

CI Target: Runs on every Pull Request.

Tier 3: Smoke Tests (Execution Loops)
Scope: 1-to-2 epoch training loops on trivial environments (e.g., slippery_grid_world). Asserts that the loop runs, logs, and checkpoints without crashing, not that the agent learns.

Execution: Runs in minutes.

CI Target: Runs Nightly.

Tier 4: Algorithmic Regression (Feature Verification)
Scope: Full training jobs on baseline environments (e.g., CartPole) to ensure sample efficiency hasn't degraded and paper details (PPO, EfficientZero) remain intact.

Execution: Runs in tens of minutes to hours.

CI Target: Runs on Releases or Weekly.

## 🤝 Contract Testing: Protecting Architectural Assumptions

While pure math unit tests ensure the agent learns efficiently, **Contract Tests** ensure the agent doesn't crash 18 hours into a training run due to a shape mismatch or a broken dictionary payload.

In RL, modules act like microservices. Contract tests verify the strict formatting agreements between these modules (tensors, padding conventions, structural invariants) without validating the underlying math. These live inside `tests/unit/` (often named `test_[module]_contracts.py`) and execute in Tier 1.

The 3 Tiers of Efficiency Tests
To get the insights you are looking for, break your benchmarks down into these three categories:

Micro-Benchmarks (The Isolated Baseline):

Goal: How fast is pure MCTS without the neural network? How fast is the replay buffer?

Implementation: Mock the neural network inference step to return a random tensor instantly. Run 10,000 MCTS simulations and measure pure CPU traversal/expansion speed.

Sub-system Benchmarks (The Interaction):

Goal: How does batched neural network inference affect MCTS throughput?

Implementation: Combine the actual MCTS logic with the actual PyTorch model. Benchmark unbatched inference vs. batched inference.

Scaling Benchmarks (The Distributed Reality):

Goal: Does adding Ray workers actually increase Frames Per Second (FPS), or does IPC (Inter-Process Communication) overhead ruin it?

Implementation: Spin up a dummy environment. Measure the FPS with 1 worker. Spin up 4 workers. Assert that the 4-worker FPS is at least > 2.5x the 1-worker FPS (accounting for overhead).

1. The 3 Types of Parity Tests (And How to Handle Them)
Internal Backend Parity (Must Have): This verifies that two different implementations of the exact same logic yield the exact same tensors. For your repo, this means asserting that search_py.search() outputs the exact same policy logits and values as search_cpp.search() given the identical starting state and seed.

External Library Parity (Highly Recommended): This verifies your custom code against an established open-source truth. If you write a custom SegmentTree for your prioritized replay buffer, you write a test that runs identical operations on yours and OpenAI/Ray's SegmentTree and asserts the outputs match.

Legacy Parity (Temporary - Exclude from Main): Testing your new refactored code against your old unrefactored code is a great way to ensure you didn't break anything during the transition. However, do not keep these in your permanent test suite. Once the refactor merges, the old code is deleted. If you keep the tests, you have to maintain the old code just to run the tests, defeating the purpose of the refactor. Use them as temporary scripts, then throw them away.

### The 4 Core RL Contracts

When adding or modifying components, ensure the following contracts are explicitly tested:

**1. The Padding & Alignment Contract**
* **The Rule:** If a trajectory ends prematurely, the unrolled sequence must still equal length $K$. Padded steps must be zeroed, and the generated loss mask must explicitly ignore these indices during `.backward()`.
* **The Test:** Pass a sequence of length 2 into a module expecting length 5. Assert `shape == 5`, assert indices `[2:] == 0.0`, and assert the loss mask correctly identifies the invalid steps.

**2. The Terminal State Invariant**
* **The Rule:** The value of a terminal state is strictly `0.0`, regardless of neural network predictions.
* **The Test:** Pass a batch where `done = True`. Assert the bootstrap target values for those specific indices equal exactly `0.0`.

**3. The Shape and DType Contract**
* **The Rule:** PyTorch silently broadcasts mismatched tensors and promotes data types, causing silent memory bloat or massive logical errors.
* **The Test:** Isolate module boundaries (e.g., the transition between the Representation and Dynamics network). Explicitly `assert output.dtype == torch.float32` and `assert output.shape == (B, hidden_dim)`.

**4. The Dictionary/Payload Contract**
* **The Rule:** Distributed workers (Ray Actors) and Replay Buffers must agree on the exact keys of the transition payloads.
* **The Test:** Mock an environment step and assert `set(payload.keys()) == {"obs", "action", "reward", "done", "policy_logits"}`. Trigger extreme edge cases (like an empty dictionary) to ensure the buffer handles them gracefully.

⚖️ The 5 Rules of RL Test Driven Development
When adding tests to this repository, strictly adhere to the following rules:

The "Analytical Oracle" Rule: For math functions (GAE, n-step returns, UCB calculations), calculate the expected output by hand for a small sequence (e.g., length 3). Hardcode those exact float values into the test and use torch.testing.assert_close().

Strict Determinism: RL tests must be 100% reproducible. Rely on fixtures in conftest.py that strictly seed torch, numpy, and random before every test. Flaky tests are failing tests.

Explicit Tensor Shapes: A massive portion of PyTorch bugs stem from accidental broadcasting. Explicitly assert tensor.shape == expected_shape before asserting values.

The "Gradient Flow" Check: When testing complex pipelines, compute a dummy loss, call .backward(), and explicitly assert that .grad is not None for active networks, and is None for frozen/target networks.

Mock the Environment: Unit tests should never instantiate an actual Gym or Pufferlib environment. Pass synthetic PyTorch tensors (torch.randn(...)) to represent observations. Isolate agent logic from environment logic.

⚡ The 4 Rules of Performance Testing
Performance tests live in tests/performance/. Unlike unit tests, they do not test if the math is right; they test how fast it executes. To prevent flaky CI pipelines, adhere to these rules:

The Warm-Up Rule: Code execution speed changes drastically after the first run due to PyTorch memory caching, CPU cache warming, and JIT compilation (e.g., torch.compile). Always run a "dummy" loop 10 times before starting the benchmark timer.

Relative over Absolute Assertions: CI runners have highly variable CPU speeds. Never assert throughput > 1000 iter/sec. Instead, assert relative algorithmic improvements: assert batched_throughput > unbatched_throughput * 1.5 or assert worker_pool_4x.fps > worker_pool_1x.fps * 2.0.

Strict Isolation for Compute: When testing MCTS tree-search speed, completely mock the PyTorch network inference. When testing PyTorch inference throughput, mock the MCTS tree. You must know exactly which component is bottlenecking.

Track, Don't Fail: Use tools like pytest-benchmark. Configure the pipeline to generate a JSON report of the speeds rather than failing the build. Track these over time to easily spot commits that introduce silent regressions in throughput.

🪞 The 3 Rules of Parity Testing
Parity tests live in tests/parity/ and ensure that optimized or custom implementations are mathematically identical to reference implementations.

Strict Seed Synchronization: Parity tests will fail instantly if environments or network initializations differ. You must explicitly seed the torch, numpy, and random generators at the exact moment before calling each implementation.

Assert Close, Not Equal: Due to floating-point drift between Python/C++ or CPU/GPU, never use assert a == b. Always use torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4).

No Legacy Code: Parity tests are for comparing active backends (e.g., Python vs C++) or active code against external libraries (e.g., StableBaselines3). Do not commit parity tests that compare new refactored code against old deprecated code; those are temporary development scripts only.

🚀 Getting Started
Run the fast unit tests locally before pushing:

Bash
pytest tests/unit/
Run tests with print statements for debugging:

Bash
pytest tests/unit/ -s

Bash
pytest tests/performance/ --benchmark-json=new_benchmarks.json