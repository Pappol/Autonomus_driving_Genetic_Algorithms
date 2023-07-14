# Bio_Inspired_Ai

This repository contains the work done by Jacopo Donà, Giovanni Ambrosi, Riccardo Parola, and Dung Van on the Highway environment for the Bio-Inspired AI course.

It is subdivided into 4 different folders, each of which contains the script for training and solving the environment with different algorithms.

The environment configuration was modified due to the "light" punishment of crashes in the original parametrization and a different Observation Type. All the algorithms use the following configuration:

```
config = {
        "observation": {
            "type": "TimeToCollision",
            "horizon": 10
        }
        , "action": {
            "type": "DiscreteMetaAction",
        },
        "duration": 40,  # [s]
        "lanes_count": 4,
        "collision_reward": -5,
        "high_speed_reward": 1,
        "reward_speed_range": [23, 30],
        "normalize_reward": False
    }
```

In addition, the original TimeToCollision observation was post-processed due to data sparsity, details on the modification are available in the convert_observation() method available in all the scripts.

### DQN
The DQN folder contains the main.py file where the script is executed and the model.py file contains the Agent and DQN class definition.

The algorithm is pretty straightforward, in learnDQNetwork() the agent is initialized and it contains the main parameters to train a DQN model. 

The default configuration runs for 4000 iterations, once complete it saves the agent in a pkl file and the DQN network in a pt file (loading the pickle gives an error due to the lack of a serialization method, this issue is known and was not addressed yet as the training takes approximately 90 minutes)-

Once the agent has been trained, the best-performing agent is taken and evaluated on 100 different scenarios. The training environments are not initialized by seed and are thus random.

Note: to correctly save the results, there must be a manually made directory called run.

### GA
The GA folder contains the script for running a Genetic Algorithm. At the beginning of highway.py, a dictionary lists the customizable parameters that were tried when testing the performance:

- "num_individuals": The number of individuals generated at each generation
- "num_generations": Number of generations.
- "num_parents_mating": Number of solutions to be selected as parents in the mating pool.
- "parent_selection_type": Parent selection method, can be rank, rws (roulette wheel), or tournament.
- "crossover_type": The type of the crossover operator, can be single_point, two_points, or uniform
- "mutation_type": The type of the mutation operator, can be random (gaussian noise), swap, inversion, scramble, or adaptive
- "mutation_probability": Probability of modifying a gene, if adaptive is selected, need to use a tuple of 2 values with a probability of mutation of bad solution and good solution  
- "parents_percentage": Percentage of parents to keep in the next population, goes from 0 to 1. Not used in the default configuration
- "simulation_type": Choose from [evolution_seed,population_seed,individual_seed] Evolution means a single seed is used for the whole process, population seed means all individuals in the same population share the same environment, individual means every environment is different,
- "evaluation_scenarios": How many runs is the individual evaluated on when computing the fitness
}

Once complete the algorithm saves the best-performing model learned during training in a pt file (which can be loaded) and outputs visual and text results of 100 environments.

Note: to correctly save the results, there must be a manually made directory called run.

### CMA-ES
The CMA-ES folder contains the script for running a Covariance Matrix Adaptation Evolution Strategy. At the beginning of highway.py, a dictionary lists the customizable parameters that were tried when testing the performance:

- "num_evaluations": How many runs is the individual evaluated on when computing the fitness.
- "lambda": Population size
- "mu": Number of top-performing individuals selected  for covariance matrix adaptation
- "initialization_method": Initialization of the mean vector, can be random or zeros. Better results were obtained with zeros
- "seed_mode": How the seed is defined through generations, can be "random" or "fixed" or "generation". Fixed means every individual in the evolution is evaluated in the same environment, Generation means a unique seed for all individuals is used at each generation.
- "hidden_layers_net": The number of hidden layers in the net, can be 1 or 2. Better results were found with 2
- "num_generations": Number of generations 

Note: to correctly save the results, there must be a manually made directory called run.

### NEAT

In order to run the NEAT solution to the environment you can simply run the python file called neat_solver by running:
    
    cd NEAT
    python neat_solver.py

The script will run the NEAT algorithm for 100 generations and then it will evaluate the best genome on 100 different scenarios. 

Different parameters can be set in order to better understand the performance of the algorithm. The parameters are the following:

- --config_path (type: str, default: 'config_ex.txt'): Specifies the path of the config file. This file contains the configuration settings for the NEAT algorithm, such as population size, mutation rates, and other genetic parameters. You can provide your own config file or use the default one provided.

- --generations (type: int, default: 100): Sets the number of generations for which the NEAT algorithm will evolve the population. Each generation represents one iteration of the genetic algorithm, where individuals are evaluated, selected, and evolved based on their fitness.

- --save_path (type: str, default: 'gif/'): Specifies the path of the folder where the generated GIF files will be saved. During training, the script creates GIF files to visualize the agent's progress and performance over time. The GIF files will be saved in the specified folder.

- --test (type: int, default: 100): Determines the number of test episodes to perform after the training is complete. After training, the trained agent will be tested in the environment for the specified number of episodes. The agent's performance during these test episodes will be evaluated and analyzed.

- --n_training_env (type: int, default: 10): Sets the number of training environments to use during the training process. The highway environment can have multiple variations or scenarios, and using multiple training environments helps the agent generalize its learned behavior across different scenarios. The agent will be trained in each of the specified number of training environments.

By adjusting these command-line arguments, you can customize the NEAT algorithm's behavior and training process according to your specific requirements.

It will then show the results of the evaluation and save the best genome. The best performing tests are saved as gifs in the gif folder.

If you want to replicate the evaluation process in order to choose the parameters run the run_tests.sh witch will run the neat solver multiple times with different parameters and save the outupts. This will plot into wandb all the results.

To build environment for dependecies refer to NEAT repo [here](https://github.com/CodeReclaimers/neat-python)

