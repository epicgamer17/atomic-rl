"""
LayerNorm needed for nonstationarity stability, by normalizing the activations.

1. sparse init
2. eligibility traces
3. ObGD optimizer
4. LayerNorm (pre activation) without learning any scaling or bias parameters. Specifically, the LayerNorm normalization ϕ we use is given by:
ϕ(a) = (a − µ) / sqrt(σ^2 + ϵ), where μ = 1/n * sum(a_i) and σ^2 = 1/n * sum((a_i - μ)^2)
where n is the dimensionality of a and ϵ is a small number used for numerical stability
5. Online Normalization of Reward and Observation using Welford 1962

Algorithm 4 SampleMeanVar (Welford 1962)
Require: Input x, mean µ, statistic p, and counter n.
    n ← n + 1
    ¯µ ← µ + 1
    n (x − µ)
    p ← p + (x − µ)(x − ¯µ)
    σ2 ← p
    n−1 if n ≥ 2, otherwise σ2 ← 1
return ¯µ, p, σ2, n

Algorithm 5 ScaleReward
Initialize: u ← 0
Require: r, γ, p, T, n
u ← γ(1 − T )u + r
, p, σ2, n ← SampleMeanVar(u, 0, p, n)
Return: r√σ2+ϵ , p

Algorithm 6 NormalizeObservation
Require: S, µ, p, n
µ, σ2, p, n ← SampleMeanVar(S, µ, p, n)
Return: S−µ√σ2+ϵ , µ, p

TODO: possibly create a "LayerNorm Network" class in networks.py
TODO: possibly extract other networks? like Q learning network policy network PPO network (tanh activation) etc.

NOTE: The use of LayerNorm Networks (and it seems to be important) means that SWR needs to be used instead of CBP. Aditionally its unclear if ObGD replaces IDBD methods or is additional to those methods. The paper seems to indicate that they build off of AutoStep and Kearny 2023 (which i think might be AdaGain? idk) which makes me think this replaces the IDBD methods.

TODO/NOTE: for using more complicated TD methods (ones limited to the linear case) follow this paper: Javed et al. (2024) introduced the SwiftTD algorithm for prediction problems by applying SwiftTD to the last linear layer of a deep network and using TD(λ) to all other layers.

TODO: compatible with Double Q Learning, Dueling Q Networks, and Noisy Nets. Apply to these as suggested in the original paper.
TODO: adding incremental model based methods (dyna?)
TODO: adding recurrent neural networks based on  "A learning algorithm for continually running fully recurrent neural networks" with ith the recent scalable approaches (e.g., Irie et al. 2024, Zucchet et al. 2024, Elelimy et al. 2024, Javed et al. 2023 from Stream RL Paper)
TODO: extending ObGD with metatrace methods from Young et al 2018 Metatrace actor-critic: Online step-size tuning by meta-gradient descent for reinforcement learning control.
"""
