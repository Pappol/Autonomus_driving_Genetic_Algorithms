import gymnasium as gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import math
import pygad.kerasga
import pygad
import statistics
import wandb

wandb.init(project='GA_highway')

#PARAMETERS
NUM_INDIVIDUALS=10
NUM_GENERATIONS = 100  # Number of generations.
NUM_PARENTS_MATING = 2  # Number of solutions to be selected as parents in the mating pool.
PARENT_SELECTION_TYPE = "tournament"  # Type of parent selection.
CROSSOVER_TYPE = "single_point"  # Type of the crossover operator.
MUTATION_TYPE = "random"  # Type of the mutation operator.
MUTATION_PERCENT_GENES = 10  # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
PARENTS_PERCENTAGE=0 #Percentage of parents to keep in the next population, goes from 0 to 1

def list_envs():
    all_envs = gym.envs.registry

    print(sorted(all_envs))

def fitness_func(solution, sol_idx):
    global keras_ga, model, observation_space_size, env

    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
    model.set_weights(weights=model_weights_matrix)

    # play game
    observation = env.reset()[0].flatten()
    sum_reward = 0
    done = False
    truncated=False
    while (not done) and (not truncated):
        state = np.reshape(observation, [1, observation_space_size])
        final_layer = model.predict(state,verbose=0)
        action = np.argmax(final_layer[0])
        observation_next, reward, done,truncated, info = env.step(action)
        observation = observation_next.flatten()
        sum_reward += reward

    return sum_reward


def callback_generation(ga_instance):
    generation_index=ga_instance.generations_completed
    print("Generation = {generation}".format(generation=generation_index))
    solutions_fitness = ga_instance.last_generation_fitness
    best_solution=max(solutions_fitness)
    median_solution=statistics.median(solutions_fitness)
    worst_solution=min(solutions_fitness)
    print("Best Fitness = {best}".format(best=best_solution))
    print("Median Fitness = {average}".format(average=median_solution))
    print("Worst Fitness = {worst}".format(worst=worst_solution))
    wandb.log({"Generation":generation_index,"Best Fitness": best_solution, "Median Fitness": median_solution,"Worst Fitness":worst_solution})
    print("="*30)

env = gym.make("highway-fast-v0", render_mode='rgb_array')
config = {
        "observation": {
            "type": "TimeToCollision",
            "horizon": 10
        }
        , "action": {
            "type": "DiscreteMetaAction",
        },
        "duration": 40,  # [s]
        "lanes_count": 3,
        "normalize_reward": True
    }
env.configure(config)
observation_space_size=len(env.reset()[0].flatten())
action_space_size = env.action_space.n

model = Sequential()
model.add(Dense(64, input_shape=(observation_space_size,), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(action_space_size, activation='linear'))
model.summary()

keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=NUM_INDIVIDUALS)

if(PARENTS_PERCENTAGE==1):
    keep_parents=-1
else:
    keep_parents = int(NUM_INDIVIDUALS*PARENTS_PERCENTAGE)  # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

initial_population = keras_ga.population_weights  # Initial population of network weights
ga_instance = pygad.GA(num_generations=NUM_GENERATIONS,
                       num_parents_mating=NUM_PARENTS_MATING,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       parent_selection_type=PARENT_SELECTION_TYPE,
                       crossover_type=CROSSOVER_TYPE,
                       mutation_type=MUTATION_TYPE,
                       mutation_percent_genes=MUTATION_PERCENT_GENES,
                       keep_parents=keep_parents,
                       keep_elitism=0,
                       on_generation=callback_generation,
                       save_solutions=False)

ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_fitness(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

wandb.finish()
# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
model.set_weights(weights=model_weights_matrix)
model.save("highway_weights")