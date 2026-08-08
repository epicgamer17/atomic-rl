"""
Notes on AdaGain:

"Consider a learning system whose objective is to learn a large collection of predictions about an agent’s future interactions with the world. The predictions specify the value of some signal many steps in the future, given that the agent follows some specific course of action. There are many examples of such prediction learning systems including:
- Predictive State Representations (Littman, Sutton, and Singh 2001)
- Observable Operator Models (Jaeger 2000)
- Temporal-difference Networks (Sutton and Tanner 2004)
- General Value Functions (Sutton et al. 2011)

In our setting, the agent continually interacts with the world, making new predictions about the future, and revising its previous predictions as new outcomes are revealed. Occasionally, partially due to changes in the world and partially due to changes in the agent’s own behaviour, the targets may change and the agent must refine its predictions"

AdaGain is a new meta-descent algorithm that attempts to optimize the stability of the base learner, rather than convergence to a fixed point (TODO what does that mean). It is built in a way that allows it to be easily combined with a variety of base-learners including SGD, TD learning, and even AMSGrad!

The problem they use in the paper is given an imperfect observation, agent must make a prediction using that observation. Observations come in a continual online stream. Agent wants to learn a predictor, this could be discounted returns, or predict next observation from current observation, etc. In this setting the agent updates its weights after each new observation (not in batches like is done traditionally).

Extensions from "Meta-descent for Online, Continual Prediction" (Jacobsen et al., 2019):

1. Optimization for Stability vs. Convergence:
Traditional meta-descent (IDBD/Autostep) minimizes the squared prediction error. AdaGain takes a dynamical systems approach and instead minimizes the squared *norm of the update vector* (||Delta||^2). By keeping the updates small, it explicitly optimizes for system stability in non-stationary environments, rather than convergence to a fixed point.

2. Algorithm Agnostic (Composability):
Because IDBD is derived specifically for the LMS/squared-error objective, it cannot easily wrap other optimizers. AdaGain's derivation allows it to be layered on top of *any* base update vector, including quasi-second-order methods like RMSProp or semi-gradient updates like Temporal Difference (TD) learning.

3. The Finite Difference Approximation:
To wrap any generic algorithm, AdaGain needs the Jacobian of the update function. The authors propose a linear-complexity, finite-difference approximation. It evaluates the base update function three times per step (at current weights, w + r*update, and w - r*update) to approximate the curvature and adapt the step sizes.

4. Linear TD AdaGain:
For linear Reinforcement Learning, the analytic Jacobian can be computed directly without finite differences, resulting in a highly efficient meta-descent algorithm specifically tailored for TD(lambda) (TODO: an experiment showing this and its benefits)

TODO: learn the gradient math of AdaGain better.

"""
