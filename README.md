Code is based on [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html).
___
Data collection, i.e. main running files, are the `data_collection_dqn.py` / `data_collection_sac.py` files.  
In `algs_expanded.py` there are the stable-baselines3 implementations of SAC and DQN, with slight variations such as adding a method to change the environment.  
Results are plotted with `plot_results.py`.  
`Changes to stable_baselines3.txt` includes changes made locally to the stable-baselines3 package to fix some issues, and are not included in this repository.  
In the `environments` folder there is the ContinuousCartPole environment.  
The other files include the algorithms used in this project - DQN+MER, SAC+MER, variations and building blocks.

