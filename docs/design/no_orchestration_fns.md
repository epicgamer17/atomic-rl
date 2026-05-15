we decided to remove the compute_td_error function, compute_q_td_loss and compute_v_td_loss as they were getting hacky and complicated to use because they were trying to do too much. 

they were trying to handle LSTMs and simple networks, and they were trying to handle Distribution Bootstrapping (having to make it an expected value or transform to a distribution etc) and also handle double Q-learning or not double q learning. it ended up being that you passed in a bunch of partial fns that were quite messy and hard to read. they had unintuitive signatures, as for example standard dqns bellman loss needed a partial for action selection that took the argmax of the pred but was lambda that accepts preds and obs (in case it wanted to do double dqn). it was only going to get more complex when we tried to add SARSA or Soft Q learning etc. 

this allows for slightly more explicit readable code in the examples that abstracts less implementation details, however it does mean there is a lot of code reuse between algorithms that is not in the form of a function which is unfortunate. 

This may be a larger problem with our algorithm agnosticism and avoidance of combinatorial explosions which stops us making functions like double_dqn_loss and categorical_dqn_loss and rainbow_loss. 

it could be possible to add something back later. a sort of orchestration function, something like: 
# functional/losses.py
def compute_q_td_loss(
    q_predict_fn: Callable[[], torch.Tensor],      # <--- Changed from nn.Module
    q_target_fn: Callable[[], torch.Tensor],       # <--- Changed from nn.Module
    batch: TensorDict,
    ...
):
    predictions = q_predict_fn()
    with torch.no_grad():
        next_preds = q_target_fn()
    # ... rest of the math remains the same

How it looks in standard DQN:

Python
loss, info = compute_q_td_loss(
    q_predict_fn=lambda: model(batch["obs"]),
    q_target_fn=lambda: target_model(batch["next_obs"]),
    ...
)
How it easily adapts to DRQN:

Python
# The user handles the hidden state extraction, the functional layer stays pure
loss, info = compute_q_td_loss(
    q_predict_fn=lambda: model(batch["obs"], batch["hidden"])[0], # Extract just Q-values
    q_target_fn=lambda: target_model(batch["next_obs"], batch["next_hidden"])[0],
    ...
)
Pros: Keeps the orchestration.
Cons: lambda functions can't be easily compiled by torch.compile(), which might violate your efficiency rule down the line.

this would allow the code reuse. but suffers from some of the problems above. one thing though is it is much more testable and prevents orchestration errors. perhaps its worth doing?

the solution we went with for now is not much more complicated and doesnt make the code harder to read. if it does end up being more complicated or there is a TON of reuse, we will go with the alternative.

FOR HISTORY the removed code was: 
def compute_q_td_loss(
    model: torch.nn.Module,
    batch: TensorDict,
    target_model: torch.nn.Module,
    next_action_selector_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    target_calculator_fn: Callable,
    loss_fn: Callable = mse_loss,
) -> Tuple[torch.Tensor, dict]:
    """
    Calculate the Bellman error for a batch of transitions.
    The 'Imperative Shell' for DQN-style updates.

    This function orchestrates the evaluation of next-states and target calculation,
    ensuring that the pure math functions receive correctly formatted tensors.

    Args:
        model: The online Q-network.
        batch: A dictionary containing the batch of transitions.
        target_model: Model used to evaluate the selected action's value.
        next_action_selector_fn: Function to select the best action for bootstrapping.
            Takes (next_obs, next_preds) and returns next_actions.
        target_calculator_fn: Function to calculate targets.
        loss_fn: Function to calculate loss (e.g. mse_loss).

    Returns:
        torch.Tensor: The loss for the batch.
        dict: Information for logging and debugging.
    """

    # 1. Current Q-values (Prediction)
    predictions = model(batch["obs"])

    batch_size = predictions.shape[0]
    batch_idx = torch.arange(batch_size, device=predictions.device)
    # Handle both [B] and [B, 1] action shapes gracefully
    actions = batch["action"].long()
    if actions.dim() == 2:
        actions = actions.squeeze(-1)

    pred_sa = predictions[batch_idx, actions]

    # 2. Next State Evaluation
    with torch.no_grad():
        # NOTE: Noisy DQN with Double DQN/Dueling samples a 3rd epsilon here but we do not,
        # and neither do most implementations online.
        next_obs = batch["next_obs"]
        next_preds = target_model(next_obs)

        # Select the best next action using the provided selector logic
        next_actions = next_action_selector_fn(next_obs, next_preds)

        # 3. Target Calculation
        # Calculate TD target (standard, n-step, or categorical)
        td_target = target_calculator_fn(
            next_preds,
            next_actions.squeeze(
                -1
            ),  # TODO: need to be careful with this reshaping stuff in functions
            batch["reward"],
            batch["terminated"],
            batch["gamma"],
        )

    # FAIL FAST: Ensure shapes match exactly for standard DQN
    if pred_sa.dim() == 1:
        assert (
            pred_sa.shape == td_target.shape
        ), f"Shape mismatch: pred {pred_sa.shape} vs target {td_target.shape}"

    loss, info = loss_fn(pred_sa, td_target)

    # 5. Augment info with orchestration-level metrics for W&B
    info.update(
        {
            "q_values/mean": pred_sa.mean().detach(),
            "q_values/min": pred_sa.min().detach(),
            "q_values/max": pred_sa.max().detach(),
            "td_targets/mean": td_target.mean().detach(),
            "rewards/mean": batch["reward"].mean().detach(),
        }
    )

    return loss, info

Along with a simpler compute_v_td_loss as well (simpler because we had not encorporated Distributional RL, lstms etc yet)