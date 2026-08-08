Pain points. 
Monoliths are hard to do ablations with for things like MuZero and its improvements or Rainbow and its improvements. 
Monoliths either lead to branching and god classes or many files with repeated code. 
Monoliths are hard to debug as its harder to check intermediate values. Global changes are not propagated. If i fix a bug in one DQN implementation I must now apply the same fix to all others. Leads to feature drift across algorithms
Although OOP prevents god classes you now have many interfaces, adding to an interface requires learning about that interfaces internals and contracts and API. For example adding a replay buffer I need to look into the replay buffer API see what methods it provides etc so my custom buffer will work. Must do the same for search, networks, etc 
Expanding/making custom stuff with OOP can require creating a whole new file or class despite changing only a small feature. Inheritance requires knowledge of the parent class which is a bit of knowledge debt. 
A modular approach (OOP with strategy pattern) allows for modularity, so you can just add one feature at a time but still need to know about the modular class, what its strategies must provide etc. In some ways harder as adding a feature now requires diving into modular class and finding right strategy to create. 
Modular class requires sometimes fitting many approaches into one class, adding a “replay buffer” to OOP, simulating Alpha Beta search as MCTS with a certain selection strategy. This is unintuitive and harder that just the single file or OOP approach where you simply create what you want (adhering to interface) 
Modularity sometimes led to performance reductions because abstraction prevented certain optimizations
OOP/Modularity sometimes led to rigidity, having to use existing replay buffer system, couldnt easily use a torch dataset or use train method A into method B. Forced certain loops. Didnt allow for freedom and quick development. Didnt fail fast. Or you have the issue that you essentially have an multiple modular classes sharing very similar purposes and it makes design harder, as now you not only have to pick components, but also the modular class to use for each of the base modular classes. 
Modular approach didnt allow reuse of components across modules, sometimes forced to reimplement the same operation for the different modular classes and interfaces. 
OOP and Modular Classes hide design details making it hard to verify all desired features exist. 
Monoliths dont allow for switching of execution method (synchronous, ray etc) without full refactor. 
Execution graphs take a lot of work to make the system/represent it. its harder to make a simple DQN because you now need to think about data flow, how components link together etc. You are almost remaking the monolith file but with a harder to use system of graphs and nodes. However nodes can be reused unlike in monolith files. 
Execution graphs share a some of the issues with the different interfaces etc that the modular classes and OOP have but to a lesser degree. 
Execution graphs are not always as efficient as singular files, but are often able to allow for optimization more that OOP or Modular Classes (though some libraries use OOP and are much more efficient) 
Some things in execution graphs (similar to Modular Classes) become much more complicated to design and reason about (like MCTS) 
Abstracting the concepts its hard to cover RL, or make them general enough to allow all algorithms or possibilities, in some ways Monoliths dont have this issue as they are not rigidly defined, you make your file and use it. no restrictions. 
OOP, Modular Classes, and Execution graphs all suffer in differing degrees, the problems that operations or strategies or classes sometimes get messy trying to serve multiple purposes or features (ie type checking, reshaping, etc). Monoliths simply have one flow, seen clearly in the file. 
Monoliths can get VERY large for complicated algorithms like MuZero or Dreamer. And as those are more abstract it can sometimes be harder to read. 
The more abstract we get, the more we lose the concept of an algorithm, the harder sometimes it can be to verify features on an algorithm. In some ways on a monolith you can read the the file and check off features. OOP and modular classes hide features in their objects and strategies and makes verifying the existence of features for a specific algorithm or paper harder. The freedom of execution graphs can lead to the same problem, though if operations are flat you can verify at a high level the correct operations, and semantics, using the graph, and then if needed look at the individual operations used and verify their correctness. Still for execution graphs this tends to be more work than monoliths which allow for a more linear read. 
Difficulty and complexity can arise when adding custom stuff. Adding a custom operation in some ways requires knowledge of compiler validation passes and the graph interfaces and API and what your operator needs to provide (there are possibly ways to handle this). Similar issues for Modular Classes. 
Modular classes are surprisingly rigid. if you want functionality not provided in a modular class, you must make your own modular class (too much work). For example, if a modular class for search were not provided, and correctly incorporated into the modular class for acting this would be lots of work to add. 
Tight coupling between modular classes (like mentioned actor modular class must properly use search and replay buffer modular class) makes it hard to add custom code that doesnt use existing modular classes. Less so for OOP if done right. 
For monoliths you are constantly redefining things like actor loops, and it can be hard to allow functionality for many env types. 
For monoliths moving a single-file PyTorch implementation to a multi-GPU setup (Distributed Data Parallel) or TPU requires injecting system-level code directly into your mathematical logic, making the file significantly uglier.
As monoliths grow to accommodate minor variations (e.g., "should I use frame stacking or not?"), the argparse configuration block becomes massive, leading to giant if/else trees inside the single file.
You cannot easily unit test a monolith. Because everything is coupled in one for loop, you are forced to do Integration Testing (running the whole algorithm for a few steps to see if it crashes or learns). You can't easily mock out the environment or test just the advantage calculation without rewriting the code.
OOP relies heavily on internal state mutation (self._step_count, self._hidden_state, self._buffer_pointer). In RL, where algorithms are highly sensitive to off-by-one errors or stale data, state mutation makes bugs incredibly hard to track down.
For OOP, if PPO inherits from ActorCritic, which inherits from BaseAlgorithm, figuring out exactly which update() method is being called requires mental gymnastics. If an error occurs, the stack trace spans 15 files.
To make modular components plug-and-play, their interfaces must be generalized. This often means stripping away unique optimizations. For instance, if an API expects step(action) -> next_obs, it becomes very hard to pass recurrent RNN states or MuZero's hidden tree states without hacking the API via kwargs or dictionary unpacking, which defeats the purpose of the strict contract.
for execution graphs, graphs hate dynamic control flow. Writing a simple if statement or a dynamic while loop (like expanding an MCTS tree until a time limit is reached) requires using specialized, clunky graph operators (like tf.cond or jax.lax.cond) instead of standard Python.
for execution graphs, there is a lot of work developing either the graph system to support RL on top of existing systems, for the pros mentioned, a tradeoff must also be decided between the granularity of nodes, and operations, and users must adhere to those. For example, is an Actor Node acceptable? What about a simlar Argmax Node? When should users use simple torch or jax vs the execution graph? When should we as developers define a node for something vs using a torch operation directly vs wrapping that operation, and how do we handle varying granularities. How do we as developers what defines a "feature" to turn into an operation or several features. 
Execution graphs lead to more files and methods and classes for each feature, while each feature is defined mostly in one place which is nice, we now for a simple algorithm like DQN have possibly 10 to 15 operations. We must now as developers group and organize operations so that users can easily find them, and we must also document constraints of the execution graph so that the user understands what it is doing, why, what it requires, etc. 
RL requires mixing orthogonal features. What if you want an agent that is both Recurrent (RNN) AND handles Continuous Actions? In OOP, you either write a new class RecurrentContinuousPPO (combinatorial explosion of classes) or you use Python Mixins. Mixins lead to Method Resolution Order (MRO) nightmares where super().init() calls are impossible to trace.

Pros of each: 
Monoliths allow for a file diff to see the difference between to algorithms and when its very similar (ie just switching out the network or using mc returns instead of gae) its easy to see what led to changes in performance 
Monoliths allow you to see the whole implementation in one place, all the features, details etc. Although they are all there sometimes its difficult to read but you know where to look. 
Monoliths have minimal overhead, there is not as much work chaining things together etc. in some ways you can focus just on your algorithm/implementation in one file. For fast development this can be nice as you don't care about the longterm reuse and just want results quickly. 
As we get deeper to more levels like OOP and modular classes they get harder to parse and read. 
OOP and modular classes can allow for code reuse. Modular classes can have more code reuse than others. A execution graph based system allows for the most reuse 
OOP and modular classes can allow for 
 "Custom Changes" to monoliths just require making a copy of the file you are making custom changes of and making your changes to it. 
Execution graphs allow full code reuse. 
The more levels of abstraction the more validation we can do. So for modular classes we can have contracts types, but even more for execution graphs if we go deep enough we can verify semantics of operations and RL math. 
Execution graphs allow debugging and verification of each individual operation, and reuse of those verified operations. 
Execution graphs allow for mixing of any RL operations with each other given they are semantically compatible. To a lesser extent you get this per modular class with modular classes, and an even lesser extent with OOP. 
For OOP to add custom stuff you find something that does must of what you want and add a child class. People are used to this. 
For Modular Classes you swap out the functionality you want on the modular class, if it doesnt exist you must simply make your own. 
For execution graphs you only need to change the operation you want in the graph. 
Execution graphs allow you to compose operations in (mostly) any execution order meaning you can "make" a new functionality not provided by the library by composing nodes etc into a graph (unlike the case for Modular classes and search, although it would take work it would be possible), comes with the downside that you must design the "loops" unlike OOP and Modular Classes, you must each time define the Actor Loop and semantics (using the graph though) 
OOP, Modular classes, and execution graphs can easily incorporate many environment libraries through adapters. 
Can do DSL for execution graphs to ease the problem of defining graphs, but gives you essentially an OOP (or modular classes) approach where you need to wire together the objects, and therefore need to know the inputs and outputs of operators, which vary.
For monoliths because everything is in one local scope, adding a wandb.log() or saving a model checkpoint midway through a loop is completely frictionless. You don't need to write custom callbacks or hooks
Monoliths rarely require deep dependency trees. They are largely future-proof against library updates breaking internal framework APIs, because there is no internal API.
Standard OOP works flawlessly with IDEs (VSCode, PyCharm). Autocomplete, type checking (mypy), and "Go to Definition" make navigating the codebase much easier than dynamic graphs or massive monoliths.
For modular classes, the strategy pattern maps 1:1 with config files. You can define an entire experiment by combining YAML blocks (e.g., buffer: prioritize, network: resnet, optimizer: adam), meaning you can launch thousands of ablations without changing a line of Python.
For modular classes, because components are injected (Dependency Injection), testing is incredibly easy. You can pass a MockReplayBuffer or MockNetwork to test your actor loop in isolation.
For execution graphs, if your RL math is a graph/pure functions, you can compile the entire loop (actor, environment, and learner) directly to the GPU/TPU (e.g., via JAX's jax.jit or jax.lax.scan). This yields massive speedups that OOP and Monoliths literally cannot match in standard Python.
For execution graphs, if operations are graph nodes, you can backpropagate through the entire graph. This is highly relevant for model-based RL (MuZero, Dreamer) or meta-learning, where you might need gradients to flow through the environment or the search tree itself.

--- 
Thoughts on existing libraries. (Don't have much knowledge though) 
ACME to hard to learn. no search? 
Torch RL no search functionality
Ray RL Lib, must use ray, does it work great sequentially? no search functionality
Clean RL (all the monolith downsides)
Puffer RL (all the monolith downsides), must use c mostly, and not many existing algorithms
Light Zero hard to edit existing algo beyond hyperparams without coding yourself or deep knowledge of how things are implemented. (this is the case for many). 
RL Lax, pure RL in jax etc. steep shape learning curve but functional approach. 

I think I will go for a functional approach. the goal being to define algorithms as the following: 
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
    
    # 2. Step Env (Pure-ish function)
    next_env_state, next_obs, reward, done = env.step(env_state, action)
    
    # 3. Add to Buffer (Pure function returning new state)
    transition = (obs, action, reward, hidden_state)
    buffer_state = add_to_buffer(buffer_state, transition)
    
    # Update state for next tick
    obs = next_obs
    env_state = next_env_state
    hidden_state = next_hidden_state
    
    # --- 3. The Update Loop ---
    if step % UPDATE_FREQ == 0:
        # Sample (Pure function)
        batch, rng_key = sample_buffer(buffer_state, rng_key, BATCH_SIZE)
        
        # Calculate Loss & Gradients (Pure math from your Functional Core)
        loss, grads = calculate_loss(params, batch)
        
        # Apply Updates (Pure function)
        params, optimizer_state = apply_gradients(params, optimizer_state, grads)
        
        # Because we are in a monolith, logging is effortless:
        wandb.log({"loss": loss, "step": step})

I tried OOP but moved to modularity because of similarities between all my versions of DQN and MuZero. From modularity I moved to a blackboard system with some functional some modularity, but in an effort to make everything a part of my DAG (something I added with the blackboard) including replay buffers and stateful operators, I removed my Blackboard system and moved to an Execution Graph system. It was honestly quite similar to my Blackboard system but cleaner, and I just had a registration pattern and would register stateful or persistent objects like the replay buffer and network. However what I found in the end was my end goal was essentially to define my Execution Graph the exact same way I would do it if I was using some sort of a functional approach (which is also lets be honest somewhat similar to OOP and Modular Approach). I realized I had done all this work to make graphs, to optimize those graphs with asynchronous running and scheduling, but in the end it was slower than any performant libraries, not good for torch.compile because of dicts and passing everything etc, harder to initialize than functional (because of ports). Essentially I had done all this work and all I got out of it was 2 very basic algorithms (which I will say ran fast without torch.compile and well) for the same result I could have got in like 1 or 2 hours with functional approach. My system had all these layers and unintuitive style for quick research and testing for 2 benefits that I am now dropping. The first benefit was semantic and shape (and more) validation (before running any training) using my graph. The second was the scheduling optimizations. Overall, I think development could be faster, easier to keep track of, and the library could be easier to use if it was something like RLax but for PyTorch. I think theres a hope of later adding my validation (maybe using a blackboard or with modular_rl.validation():) to basically read some tags from my modular functions requires and provides, which will be overall generic. This will be the best balance I think. 

Before finalizing my decision. At the moment replay buffers still follow OOP (could do modularity). Is there a way to add this to the functional approach (efficiently)?
At the moment no semantic checking, checks of mathematical correctness, or checks of correct use etc (and no plan for it but i think its nice) 
Networks defined inline. Do I love this? What about reuse of backbones, output heads etc? 
Will I have combinatorial explosion using the functional approach? Will I have it for stateful objects (like buffers)? In other words will there be a QValueLoss, CategoricalQvalueLoss, DoubleQValueLoss DoubleCategoricalQValueLoss etc? What about the same for support types?
How will I handle granular control for search? As in able to do MCTS, Gumbel Sequential Halving, PUCT, Stochastic MuZero, Gumbel Stochastic MuZero, Sampled MuZero etc without a combinatorial explosion? 
How will I handle multi agent? 
How will I handle execution backend (ie sequential, threading, Ray, etc) and making that easy for the user but also efficient?
A brief example for plotting, logging, etc.
How will I handle Reanalyze for MuZero. 
More on Replay Buffers in Functional Approach - How will I handle things like DQN buffer returning transitions, vs MuZero sequences etc? 
Execution loops are often reused, am I sad that the monolithic shell + functional approach requires you to redefine it every time? (evaluation and learner loops too)
    

What will my examples be? 
    DQN on Cartpole
    Double DQN on Cartpole
    Dueling DQN on Cartpole
    Noisy DQN on Cartpole
    Categorical DQN on Cartpole
    Prioritized Experience Replay on Cartpole
    Rainbow DQN on Cartpole
    Ape-X DQN on Cartpole?
    NFSP on LeDuc
    NFSP on Tic Tac Toe
    PPO on Cartpole
    PPO on Pendulum
    PPO on Mujoco
    DPO on Cartpole
    DPO on Stack
    DPO on Webtext
    SAC on Cartpole
    SAC on Mujoco
    SAC on Bullet
    SAC on DeepSea
    TD3 on Cartpole
    TD3 on Mujoco
    TD3 on Bullet
    TD3 on DeepSea
    A3C on Cartpole
    A2C on Cartpole
    IMPALA on Cartpole
    REINFORCE on Cartpole
    Policy Gradient on Cartpole
    DAgger 
    MuZero on Cartpole
    Sampled MuZero on Continuous
    Gumbel MuZero
    MuZero Unplugged
    EfficientZero 
    EfficientZeroV2
    Stochastic MuZero
    Gumbel Stochastic MuZero
    MuZero with AlphaBeta Pruning
    AlphaZero? 
    World Models
    World Models V2
    World Models V3
    World Models V4
    Dreamer V1
    Dreamer V2
    Dreamer V3
    Dreamer V4
    Sutton stuff?
    Option Zero? 

As you can see there is a lot of algorithms I want to try and so I need to be able to do it quickly, with easy implementation and reuse of code, and ideally correctly (semantic checking or unit testing). 


IN A FEW SENTENCE: Monolitic approach but with functions for stuff that is reused often. No routing code, that is handled by the monolithic shell. Then I can possibly add back validation systems later if I want to. More work on buffers, but in comparison with the Graph Approach I get the persistent layer simply in my files global variables, rather than havin to do graph work. User controls plotting, user makes evaluation loops. 

ROUGH PLAN ON LAYERS?: 
    Functional Core (rl.atomic_rl/)
    Replay Buffers (rl.buffers/)
    Search Algorithms (rl.search/)?? (possibly included in functional, and would just call other functional code), possibly define in the monolithic shell?
    Validation Systems (rl.validation/)??
    Execution Loops (rl.exec/)?? (is this where I include backend stuff like ray, threading, sequential, etc?? or another layer??)

--- 

FURTHER NOTE: 
Unlike many algorithms, and after using many libraries, I no longer believe that plug and play on any environment is desireable for a library. At least not as far as learning to understand that algorithm, and learning to use that algorithm properly. I have found that there are subtle tricks involved for each env that are often not mentioned explicitly in the paper. When they are implicilty hidden by the "plug and play" parts of a library it hides those details and forces the user to dive deep into the library code to understand the tricks occuring for that algorithm on that environment. These may not be major things, perhaps detecting that an environment has a continuous action space and automatically switching the action selection and network to handle it. But there are details here that are now performed by the defaults of the library. Additionally, I find that many object oriented libraries like torch rl become hard to use, they become unpythonic. You suddenly begin to nest objects and start managing object hiearchies. Each of the objects have objects inside (some that are not apparantly obvious) this can make them hard to alter, or managing these objects adds additional boilerplate and relationships that users must learn, instead of simply understanding the concept and math behind the RL, they must now also understand how to use it for that library. This boilerplate also makes it difficult to use these RL libraries with other libraries and Python packages. What if they do not integrate nicely with the boilerplate? Your code becomes a jumbled mess.

A major benefit of the approach we eneded up going with is that it works with essentially any python library stylistically, and if you disagree with our approach chosen, it is easy to simply use the functions we have created to make an OOP Algorithm class that is plug and play. 