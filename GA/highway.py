import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
import pygad.torchga
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
device=0

wandb.init(project='GA_highway')

#PARAMETERS
params={
    "num_individuals" : 50,
    "num_generations" : 100,  # Number of generations.
    "num_parents_mating" : 5,  # Number of solutions to be selected as parents in the mating pool.
    "parent_selection_type" : "rank",  # Type of parent selection.
    "crossover_type" : "single_point",  # Type of the crossover operator.
    "mutation_type" : "random",  # Type of the mutation operator.
    "mutation_probability" : 0.1,
    #"mutation_probability" : (0.35,0.05),  #Probability of modifying a gene, if adaptive is selected, its a tuple of 2 values with probability of mutation of bad solution and good solution
    #"parents_percentage":0.1, #Percentage of parents to keep in the next population, goes from 0 to 1
    "simulation_type": "population_seed", #Choose from [evolution_seed,population_seed,individual_seed] Evolution means a single seed is used for the whole process, population seed means all individuals in the same population share the same environment, individual means every environoment is different,
    "evaluation_scenarios":5 #How many runs is the individual evaluated on when computing the fitness. Has no effect if simulation_type is evolution_seed
}

#Load parameters onto memory
NUM_INDIVIDUALS=params['num_individuals']
NUM_GENERATIONS = params['num_generations']  # Number of generations.
NUM_PARENTS_MATING = params['num_parents_mating'] # Number of solutions to be selected as parents in the mating pool.
PARENT_SELECTION_TYPE =params['parent_selection_type']  # Type of parent selection.
CROSSOVER_TYPE = params['crossover_type']  # Type of the crossover operator.
MUTATION_TYPE = params['mutation_type']  # Type of the mutation operator.
MUTATION_PROBABILITY = params['mutation_probability']  # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
#PARENTS_PERCENTAGE= params['parents_percentage'] #Percentage of parents to keep in the next population, goes from 0 to 1
SIMULATION_TYPE= params['simulation_type']
if(SIMULATION_TYPE!="evolution_seed"):
    EVALUATION_SCENARIOS= params['evaluation_scenarios']
else:
    EVALUATION_SCENARIOS=1

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
    global torch_ga, model, observation_space_size, env,gen_counter

    weights=pygad.torchga.model_weights_as_dict(model=model, weights_vector=solution)
    model.load_state_dict(weights)
    fitness=[]
    for evaluation_idx in range(1,EVALUATION_SCENARIOS+1):
        # play a series of games, return individual average reward
        if(SIMULATION_TYPE=="evolution_seed"):
            observation = env.reset(seed=100)[0]
        elif(SIMULATION_TYPE=="population_seed"):
            observation = env.reset(seed=gen_counter * evaluation_idx)[0]
        elif(SIMULATION_TYPE=="individual_seed"):
            observation = env.reset()[0]
        observation = convert_observation(observation)
        sum_reward = 0
        done = False
        truncated=False
        while (not done) and (not truncated):
            state = torch.FloatTensor(observation)
            state=state.to(device)
            final_layer = model(state)
            output=final_layer.cpu().detach().numpy()
            action = np.argmax(output)
            observation_next, reward, done,truncated, info = env.step(action)
            observation = convert_observation(observation_next)
            sum_reward += reward
        """if truncated:
            sum_reward+=20
        if done:
            sum_reward-=10"""
        fitness.append(sum_reward)
    return sum(fitness)/len(fitness)

def save_frames_as_gif(frames, path='./run/', filename='gym_animation.gif'):
    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

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


def evaluateModel(model, env):
    """
    Evaluate the model on 10 different scenarios, saves gifs and stores score results in a txt file
    :param model:
    :param env:
    :return:
    """
    results=[]
    for i in range(10):
        observation = env.reset()[0]
        observation = convert_observation(observation)
        sum_reward = 0
        done = False
        truncated = False
        frames=[]
        while (not done) and (not truncated):
            state = torch.FloatTensor(observation)
            state = state.to(device)
            final_layer = model(state)
            output = final_layer.cpu().detach().numpy()
            action = np.argmax(output)
            observation_next, reward, done, truncated, info = env.step(action)
            observation = convert_observation(observation_next)
            sum_reward += reward
            frames.append(env.render())
        save_frames_as_gif(frames=frames, filename=str(i) + "_best_agent_visualized_.gif")
        results.append(sum_reward)
    with open('score_results.txt', 'w') as f:
        for result_idx in range(len(results)):
            string="Scenario "+str(result_idx)+" score: "+str(results[result_idx])+"\n"
            f.write(string)

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
        "collision_reward":-10,
        "high_speed_reward":1,
        "reward_speed_range": [23, 30],
        "normalize_reward": False
    }
env.configure(config)
observation_space_size=len(convert_observation(env.reset()[0]))
action_space_size = env.action_space.n

layer1 = torch.nn.Linear(observation_space_size, 16)
relu = torch.nn.ReLU()
layer2 = torch.nn.Linear(16, 5)
#layer3 = torch.nn.Linear(8, 5)

model = torch.nn.Sequential(layer1,
                            relu,
                            layer2)

model=model.to(device)
torch_ga=pygad.torchga.TorchGA(model=model,num_solutions=NUM_INDIVIDUALS)


"""if(PARENTS_PERCENTAGE==1):
    keep_parents=-1
else:
    keep_parents = int(NUM_INDIVIDUALS*PARENTS_PERCENTAGE)"""  # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

initial_population = torch_ga.population_weights  # Initial population of network weights
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
                       #keep_parents=keep_parents,
                       keep_elitism=4,
                       on_generation=callback_generation,
                       allow_duplicate_genes=True,
                       save_solutions=True)

ga_instance.run()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
best_weights = pygad.torchga.model_weights_as_dict(model=model, weights_vector=solution)
model.load_state_dict(best_weights)
torch.save(model.state_dict(), "best_solution.pt")
evaluateModel(model,env)

wandb.finish()
# Returning the details of the best solution.
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

