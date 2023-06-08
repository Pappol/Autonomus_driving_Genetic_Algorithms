import gymnasium as gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from matplotlib import animation
import pygad.kerasga
import pygad
import statistics
import wandb
import random

#NOTE: used pygad version 2.19.2

"""EXPECTED RESULT
Based on the results we’ve observed above, we can come to the conclusion that
genetic algorithms such as NeuroEvolution can speed up the initial phase of
generalizing features several fold compared to traditional techniques such as
back-propagation which place the prerequisite of procuring a massive dataset
for training so as to not over-fit the solution. But once the network attains the
basic cognitive abilities for driving, it can be further improved upon through
reinforcement learning techniques such as Deep Q-learning, since it has already
gathered a plethora of experiences over several generations, which is one of the
key barriers slowing down reinforcement, since now we can quickly jump to the
phase where the focus is more on obtaining as many rewards as possible rather
than the initial phase of gathering experience where the agent primarily tries to
just not get punished for its actions"""

wandb.init(project='GA_highway')

#PARAMETERS
params={
    "num_individuals" : 10,
    "num_generations" : 100,  # Number of generations.
    "num_parents_mating" : 2,  # Number of solutions to be selected as parents in the mating pool.
    "parent_selection_type" : "rank",  # Type of parent selection.
    "crossover_type" : "single_point",  # Type of the crossover operator.
    "mutation_type" : "adaptive",  # Type of the mutation operator.
    "mutation_probability" : (0.5,0.1),  #Probability of modifying a gene, if adaptive is selected, its a tuple of 2 values with probability of mutation of bad solution and good solution
    "parents_percentage":0, #Percentage of parents to keep in the next population, goes from 0 to 1
}

#Load parameters onto memory
NUM_INDIVIDUALS=params['num_individuals']
NUM_GENERATIONS = params['num_generations']  # Number of generations.
NUM_PARENTS_MATING = params['num_parents_mating'] # Number of solutions to be selected as parents in the mating pool.
PARENT_SELECTION_TYPE =params['parent_selection_type']  # Type of parent selection.
CROSSOVER_TYPE = params['crossover_type']  # Type of the crossover operator.
MUTATION_TYPE = params['mutation_type']  # Type of the mutation operator.
MUTATION_PROBABILITY = params['mutation_probability']  # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
PARENTS_PERCENTAGE= params['parents_percentage'] #Percentage of parents to keep in the next population, goes from 0 to 1

gen_counter=0

def list_envs():
    all_envs = gym.envs.registry

    print(sorted(all_envs))

def convert_observation(observation_matrix):
    """
    Converts observation matrix from VxLxH to L array, where the array contains the depth of the closest collision
    :param observation_matrix:
    :return:
    """
    collision_vector=[]
    for row in observation_matrix[1]:
        index = np.where(row > 0)[0]
        if len(index) > 0:
            collision_vector.append(index[0])
        else:
            collision_vector.append(-1)
    return collision_vector


def fitness_func(solution, sol_idx):
    global keras_ga, model, observation_space_size, env,gen_counter

    model_weights_matrix = pygad.kerasga.model_weights_as_matrix(model=model, weights_vector=solution)
    model.set_weights(weights=model_weights_matrix)

    # play game
    observation = env.reset(seed=gen_counter)[0]
    observation = convert_observation(observation)
    sum_reward = 0
    done = False
    truncated=False
    while (not done) and (not truncated):
        state = np.reshape(observation, [1, observation_space_size])
        final_layer = model.predict(state,verbose=0)
        action = np.argmax(final_layer[0])
        observation_next, reward, done,truncated, info = env.step(action)
        observation = convert_observation(observation_next)
        sum_reward += reward
    return sum_reward

def save_frames_as_gif(frames, path='./run/', filename='gym_animation.gif'):
    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)
def testDemo(model, env):
    for i in range(10):
        frames=[]
        score=0
        observation = convert_observation(env.reset()[0])
        done = False
        truncated = False
        while (not done) and (not truncated):
            state = np.reshape(observation, [1, observation_space_size])
            final_layer = model.predict(state, verbose=0)
            action = np.argmax(final_layer[0])
            observation_, reward, done, truncated, info = env.step(action)
            observation_ = convert_observation(observation_)
            frames.append(env.render())
        save_frames_as_gif(frames=frames, filename=str(i)+"_best_agent_visualized_.gif")

def mutation_func(offspring, ga_instance):
    mean=0
    std=5
    for new_gen_individual in range(offspring.shape[0]):
        for gene_index in range(offspring.shape[1]):
            if(random.uniform(0, 100)<=ga_instance.mutation_percent_genes):
                offspring[new_gen_individual,gene_index]=offspring[new_gen_individual,gene_index] + np.random.normal(mean, std)
    return offspring

def callback_generation(ga_instance):
    global gen_counter
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
    gen_counter+=1
    print("="*35)

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
        "lanes_count": 4,
        "collision_reward":-5,
        "reward_speed_range": [23, 30],
        "normalize_reward": False
    }
env.configure(config)
observation_space_size=len(convert_observation(env.reset()[0]))
action_space_size = env.action_space.n

model = Sequential()
model.add(Dense(16, input_shape=(observation_space_size,), activation='tanh'))
model.add(Dense(8, activation='tanh'))
model.add(Dense(action_space_size, activation='linear'))
model.summary()

keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=NUM_INDIVIDUALS)

if(PARENTS_PERCENTAGE==1):
    keep_parents=-1
else:
    keep_parents = int(NUM_INDIVIDUALS*PARENTS_PERCENTAGE)  # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

initial_population = keras_ga.population_weights  # Initial population of network weights
print("GA settings:")
print(params)
print("Environment settings")
print(config)
print("Input type:",observation_space_size)
ga_instance = pygad.GA(num_generations=NUM_GENERATIONS,
                       num_parents_mating=NUM_PARENTS_MATING,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       parent_selection_type=PARENT_SELECTION_TYPE,
                       crossover_type=CROSSOVER_TYPE,
                       mutation_type=MUTATION_TYPE,
                       mutation_probability=MUTATION_PROBABILITY,
                       keep_parents=keep_parents,
                       keep_elitism=0,
                       on_generation=callback_generation,
                       save_solutions=False)

"""ga_instance = pygad.GA(num_generations=NUM_GENERATIONS,
                       num_parents_mating=NUM_PARENTS_MATING,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       parent_selection_type=PARENT_SELECTION_TYPE,
                       crossover_type=CROSSOVER_TYPE,
                       mutation_type=mutation_func,
                       mutation_percent_genes=MUTATION_PERCENT_GENES,
                       keep_parents=keep_parents,
                       keep_elitism=0,
                       on_generation=callback_generation,
                       save_solutions=False)"""

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