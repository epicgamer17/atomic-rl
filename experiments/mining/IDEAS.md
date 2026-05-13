Compare continual online learning algos (e.g. TD learning) with offline learning algos. 
Compare episodic learning algos (e.g. full RL) with continual learning algos (sutton stuff).
Use TD Learning for feature predictions and forecasting. (Kind of online sequence prediction?)

If the features from the mine or the data from the mine are good enough can we use the IDBD ideas to do continual meta-learning (the linear case) to learn stuff? TODO: what do we learn? 

I think (though not certain) the current best applications of what ive seen are not necessarily RL but instead continual/online learning (e.g. TD learning) from a stream of data (non stationairy in the case of a mine). (are there more methods than what sutton covers in his book?). 

Is the best case to try and predict things about the mine, ie throughput etc, or to try and predict the best actions to take, or something else? Like what info is my agent getting and what is it putting out?

I think this is really a good use case for online continual learning (not necessarily RL) from a stream of data as seen in step 1 and 2 of the alberta plan (and maybe step 3). I also think there are possible applications of meta learning here (e.g. from the IDBD ideas in chapter 19 of Sutton?) and of Stream RL (and a comparison of Stream RL vs Episodic RL vs Online RL for mining). Theres also questions about how to model the environment? is it episodic? is it hierarchical? is it a stream? is it stationary? can we make it stationary with the right inputs? for example if we give it the number of active machines it can know/learn about the change in the state of the mine and thus there isnt non stationarity right? however perhaps if we dont have that information it would be able to adapt to a shift (say a workers strike or wage increase or other major event that leads to a shift in the distribution).

i really think its great because i think mine data is probably good enough we can 1 use pure linear layers and 2 use the features from the environment, and so we can use what i have done so far. no need for non linearities, deep networks, or all that (ie MetaOptimize or AdaGain), though perhaps those methods are applicable and useful if we decide to use them.