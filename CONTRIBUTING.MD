Explicit over Implicit (with Orchestration): It is better to avoid "magic" objects that hide hidden state mutations. However, using stateless, high-level orchestration functions to group operations and keep the main algorithmic loop clean and readable is highly encouraged.

Fail Fast: Use assertions and type hints in the functional layer to ensure that shape mismatches or device errors are caught at the point of origin rather than deep in a gradient calculation.

Documentation by Signature: A well-named function with a clean signature and type hints should tell you exactly what the math is doing without needing to read the implementation.

Spin straw into gold: Try to make the code as readable and understandable as possible, fully implement features to the best of your ability (following this philosophy) before moving on to the next feature. Maximum utilty per feature/commit.

*atomic_rl/*
This layer contains "pure" mathematical functions. 
These functions should be stateless, have no side effects, and strictly avoid guessing the shapes of incoming tensors wherever possible. Exceptions are permitted for standard ML conventions that inherently require state (e.g., in-place network updates or buffer history closures), provided they are clearly documented. 
If a function expects a batch dimension, it must receive a batch dimension. On top of this, shapes and expectations should aim to make it so reshaping is rarely needed and if it is, it should be explicit. 
Ideally, functions are "pure", where the same inputs yield the same outputs, but this is not strictly enforced.
Minimize footguns
Functions should try to be compilable with torch.compile()
Functions should try to be composable and build upon each other.
Functions should try to be reusable. 
Functions should be general purpose and not specific to a particular algorithm or environment.
Functions should be well documented with type hints and docstrings.
Functions should try to handle edge cases, and they should be tested.
Functions should be efficient and performant.
Functions should be reusable across algorithms and papers. For example, gaussian action selection should be usable in ppo, a2c, td3, sac, etc. It shouldn't be hardcoded for a particular algorithm.
Functions should attempt to have consistent signatures/interfaces so that "similar"/related functions can be swapped in and out easily. For example, mc returns should be easily swappable with n step returns for VPG. Ie they should return the same output format and stuff, and they should take the same inputs (as much as possible).
Avoid combinatorial explosion of options. If multiple options/architectures exist for a function try to solve via composability or handle it in a case by case basis in the orchestration code.
Solutions/decisions should be expandable. When designing a function don't design it only for the algorithm it is currently being created for. Instead, design it such that it could be easily extended/adapted to other algorithms.
Functions should be designed in a way that tries to minimize the amount of code a person has to write to get the desired functionality, while maintaining readability and semantic understandability.
Where possible avoid greek letters, unless it is the absolute standard convention. 
Functions should be designed such that a person can understand the function by just reading its signature and docstring. 
Functions should try their best to avoid using `*args` and `**kwargs`, they should instead use proper type hints and named arguments.
Functions should be designed in a way that is easy to test. 
Functions should be designed in a way that is easy to debug. 
Functions should be designed in a way that minimizes the need for boiler plate functions in the orchestration code (not including functools.partial etc.). For example, if for categorical DQN we had to make a categorical_greedy_selector which used the expected value and argmax to meet the signature required by a generic functional loss function (that could be used for any q function), we should instead try to design the functions such that this boilerplate is avoided. This may require more creative solutions but it is worth it to avoid this boilerplate.
Functions have semantic meaning, although GAE can represent MC returns and N step returns, we have seperate functions for those to improve readability. Try to keep functions to a single responsibility, which improves readability and reusability.
Minimize code that does multiple things. If a function is doing more than one thing, try to split it into multiple functions.
Minimze dead code and useless functions. Without good reason (semantic or efficiency) we should not have two functions that perform the exact same mathematical operation.
Avoid OOP unless absolutely necessary, aim for a functional style of programming.
Break large functions into smaller, more manageable functions and compose them together to achieve the desired functionality. This is preferred over writing a single monolithic function and prevents combinatorial explosion. 
In general follow PyTorch best practices and try and make sure things integrate well with PyTorch
Minimize use of isinstance, hasattr, getattr, try-except blocks and other checks or retrievals like this when possible. It makes the code messy and harder to read. It can lead to hidden defaults, hidden value, and hidden behaviour. And it probably indicates bad design. It may be necessary sometimes, but should be limited when possible. 
Try to avoid flags like is_zero_sum that alter behaviour drastically based on a flag. It may be necessary however to reduce boilerplate and increase compatability as done with the rnn_unrolling. But in general should be avoided whenever possible.
Generally but not always, functional core, stateful wrapper (like PyTorch)
Minimize dependencies where possible. Prefer using torch utilities over third party libraries, unless there is a clear advantage to the third party library. For example, this is the reason we don't want to use einops. It makes it more torch native, and decreases the learning curve, reduces boilerplate, and the einops benefit is small if not none. 
Where possible we should not try and recreate the wheel and use existing torch utilities. Try to avoid making simple wrappers for things that are already implemented in torch, unless the wrapper provides clear semantic meaning or establishes a consistent interface for swappability (e.g., hard_update vs. soft_update).
Attempt as much as possible to follow the Liskov Substitution Principle. If a user expects a classthey should be able to drop in any subclass and use it the exact same way
Make the code as easy as possible to use for developers. Make it easy to read and easy to understand, but make it correct at all costs.
In the vein of Fail Fast, add gaurd rails, assertions, and errors.

Common solution methods: 
Dependency injection
Composability
Higher order functions
Partial execution of functions
Functional core, OOP wrapper
(More to be added as I learn more) 


Torch-like conventions
Usability over Performance. The primary goal is useability, the secondary goal is reasonable performance. (From PyTorch)
Explicit is better than implicit (From PyTorch and Zen of Python)
Simple is better than complex (From PyTorch)
Python First with Best In Class Language Interoperability (From PyTorch)


Follow PyTorch naming conventions. For example, if using inplace operations the function name should end with an underscore (_)
Functions should never hardcode device='cuda' or device='cpu', nor should they default to float32 unless absolutely mathematically necessary.
When creating new tensors inside a functional block, use .new_zeros(), .new_empty(), or torch.zeros_like(tensor) on an incoming tensor to inherently match its device and dtype. This prevents silent device-mismatch crashes during distributed training or when swapping from CPU to GPU.
Functions that compute targets (e.g., TD-targets, GAE) should explicitly note in their docstrings whether they return detached tensors.
If a functional operation mathematically does not require gradients (like computing evaluation metrics or updating meta-learning traces like IDBD's $h$ vector), wrap it in @torch.no_grad() or expect it to be called within that context to save memory.
Never use .item(), .tolist(), or .numpy() inside the functional core or the main orchestration loop. These operations trigger a CUDA Synchronization, forcing the GPU to wait for the CPU, destroying performance.
Avoid Python for loops over batch or feature dimensions at all costs. This is an absolute prerequisite if you want your functions to be friendly to torch.compile(). (If an operation is notoriously hard to vectorize, explore torch.vmap).
PyTorch allows implicit broadcasting (e.g., adding a [Batch, 1] tensor to a [Batch, Features] tensor). While efficient, this is a massive source of silent bugs. Require explicit .unsqueeze() calls in the code so the reader can visually track how the dimensions are aligning before an operation.
If a function returns a view of a tensor rather than an allocated copy, it must be documented.
If a function fundamentally requires a contiguous block of memory (common when reshaping after a .transpose()), use .contiguous() explicitly and document why it is necessary.

TorchRL-like conventions
Never blindly zero-out values using a generic done mask. If an episode is Truncated (hit a time limit), the state still has a bootstrap value, and masking it to 0 artificially corrupts the value function. Your functional TD-learning and return functions must explicitly expect terminated masks, not generic done masks

Relative imports (from .utils import ...) internally is the production best practice. PyTorch does exactly this — torch/nn/modules/linear.py uses from .module import Module, from ..parameter import Parameter. 

*examples/*
This layer contains examples of how to use the functions in the functional layer.
These examples are meant to be illustrative and should be used as a reference for how to use the functions in the functional layer.
These examples should be as CLEAN as possible, and should be well documented. They should be written in a way that is easy to understand, and they should be easy to modify.
Each algorithm example should include notes on the paper its based off of. The notes should roughly cover the papers goal, what it is trying to solve, what it achieves, what the key ideas are, the reason those ideas/contributions help solve the problem/achieve the goal, and the math behind the algorithm (roughly). it should also include notes on what was done differently than the paper (if anything) and why. ie why its needed, why its wanted, and why it works, intuitively and mathematically (briefly).
Should follow best practices of this library and ideally libraries it uses. Should include notes on on design decisions, different options/architectures, and design trade-offs. Things like "we use mc returns for this example but any XYZ could be used for the algorithm" or "we compute n step at ingestion instead of on rollout since..." 
If running a single discrete action space actions should be cast to ints (otherwise they will have a batch dimension). If running a single continuous action space actions should be flattened (again to remove the batch dim). This is because our functions require a batch dimension, but gym internal logic can silently break with a single batch dim of 1.

*networks/*
This layer contains the neural network architectures used in the examples.
Should contain common network components (like resnets, transformers, etc.) but networks "orchestration" should be left in the example files.
All components should make their behaviour clear. It should not be possible for a architectural decision to be hidden behind a network component, for example pre or post activations etc, unless clearly documented in the docstring.

*envs/* 
This layer contains the environments used in the examples.
These environments are not only gym or RL environments, they can also be any function that returns data/examples that can be used to train or learn a model.
Some examples of what should exist here: 
- all gym environments
- all openAI Gym environments
- all deepMind Environments
- custom synthetic environments (for meta optimization, function learning, etc)
- Streaming Envs and Data (for continual learning and online learning examples)
- Toy examples (like tabular cases, etc)
Essentially the goal here is to provide reusable "environments" that can be used to train or learn a model. Many of these will be used by the examples, but they can also be used by other examples, or by other projects. 
These tend to be based of experiments from papers, but they don't have to be. For example the Permuted MNIST. 
Wrappers for environments should be included here as well. Things like some of the wrappers for gym environments used in PPO for Atari. 
By default if possible existing gym envs etc should be used instead of creating new ones. But if an env doesn't exist or needs to be modified for a specific purpose (e.g. for an algorithm), then it should be included here.
The envs/functions folder should include helper functions for envs. Things like hand-coded dynamics functions, etc. 

--- 

Rough file structure: 
1. 1 file per feature
2. 1 folder per concept/field 
3. semantic structure of folder (parent folders represent broader concepts, child folders represent more specific concepts, files represent the "atomic" implementations of a concept)

If done well, folders should almost exist as their own packages. Users can import the env package, or the optimizers package, or the optimizer.metaoptimization package, depending on their field of research or interest. They should be able to look at the file structure and immediately understand what is in the library. 

It should be easy conceptually to find where the code for a specific feature lives, without needing to understand the algorithms that use it. In other words, the user doesn't need to know that algorithm X uses feature Q, and that paper introduced the feature, so they should look there. They can say I'm looking for this feature, which belongs to this concept and find the feature. 