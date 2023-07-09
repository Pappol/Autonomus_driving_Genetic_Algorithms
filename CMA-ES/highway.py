import gymnasium as gym
import cma
import numpy as np
import wandb
import torch
import statistics
import matplotlib.pyplot as plt
from matplotlib import animation

params={
    "num_evaluations": 3,
    "lambda":100,
    "mu":20,
    "initialization_method":"zeros", #Choose between "random" or "zeros"
    "seed_mode":"fixed", #Choose between "random" or "fixed" or "generation"
    "hidden_layers_net": 1, #Choose between 1 or 2
    "num_generations":100
}
evaluation_scenarios=params["num_evaluations"]
lambda_=params["lambda"]
mu=params["mu"]
mutation_magnitude=2
initialization_method=params["initialization_method"]
seed_mode=params["seed_mode"]
hidden_layers_net=params["hidden_layers_net"]
gen_number=params["num_generations"]

device=0
wandb.init(project='highway_CMA')
model=None
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
        "high_speed_reward":1,
        "reward_speed_range": [23, 30],
        "normalize_reward": False
    }
env.configure(config)

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

# Define the fitness function
def loadNetWeights(params):
    offset = 0
    net=model
    for param in net.parameters():
        param_shape = param.size()
        param.data = torch.tensor(params[offset:offset + param.numel()]).view(param_shape).float()
        offset += param.numel()
    return net


def evaluate_policy(params):
    net=loadNetWeights(params)
    net=net.to(device)
    fitnesses=[]
    for evaluation_idx in range(1, evaluation_scenarios + 1):
        sum_reward = 0
        done = False
        truncated = False
        if seed_mode=="fixed":
            observation = env.reset(seed=evaluation_idx)[0]
        elif seed_mode=="random":
            observation = env.reset()[0]
        elif seed_mode=="generation":
            observation = env.reset(seed=gen_number)[0]
        observation = convert_observation(observation)
        while (not done) and (not truncated):
            state = torch.FloatTensor(observation)
            state = state.to(device)
            final_layer = net(state)
            output = final_layer.cpu().detach().numpy()
            action = np.argmax(output) #Get the action with highest confidence
            observation_next, reward, done, truncated, info = env.step(action)
            observation = convert_observation(observation_next)
            sum_reward += reward
        fitnesses.append(sum_reward)
    #return -sum_reward  # CMA-ES minimizes the fitness function, so we negate the reward
    return -(sum(fitnesses) / len(fitnesses))# CMA-ES minimizes the fitness function, so we negate the reward

def save_frames_as_gif(frames, path='./run/', filename='gym_animation.gif'):
    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

def evaluateModel(best_solution, env):
    model=loadNetWeights(best_solution)
    torch.save(model.state_dict(),"model.pt")
    model=model.to(device)
    results=[]
    for i in range(100):
        observation = env.reset(seed=i)[0]
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
    print("Mean:", statistics.mean(results))
    print("Std:", statistics.stdev(results))
    sorted_scores = sorted(results, reverse=True)
    top_10_scores = sorted_scores[:10]
    print("Top 10 mean", statistics.mean(top_10_scores))


def testSavedModel(model):
    model.load_state_dict(torch.load("runn_100-40 zero initialization, random seed/model.pt"))
    model = model.to(device)
    results = []
    for i in range(100):
        observation = env.reset(seed=i)[0]
        observation = convert_observation(observation)
        sum_reward = 0
        done = False
        truncated = False
        frames = []
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
            string = "Scenario " + str(result_idx) + " score: " + str(results[result_idx]) + "\n"
            f.write(string)
    print("Mean:", statistics.mean(results))
    print("Std:", statistics.stdev(results))
    sorted_scores = sorted(results, reverse=True)
    top_10_scores = sorted_scores[:10]
    print("Top 10 mean", statistics.mean(top_10_scores))

observation_space_size=len(convert_observation(env.reset()[0]))
layer1 = torch.nn.Linear(observation_space_size, 16)
layer2a = torch.nn.Linear(16, 5)
layer2b = torch.nn.Linear(16, 16)
layer3 = torch.nn.Linear(16, 5)
relu = torch.nn.ReLU()
softmax=torch.nn.Softmax(dim=0)
if hidden_layers_net==1:
    model = torch.nn.Sequential(layer1,
                            relu,
                            layer2a,
                            softmax
                            )
elif hidden_layers_net==2:
    model = torch.nn.Sequential(layer1,
                                relu,
                                layer2b,
                                relu,
                                layer3,
                                softmax
                                )
"""model=torch.nn.Sequential(torch.nn.Linear(3,8),
                            relu,
                            torch.nn.Linear(8,5),
                            )"""
num_params = sum(p.numel() for p in model.parameters())# Get the total number of weights in the network


if initialization_method=="random":
    initial_mean = np.random.randn(num_params)  #Initialize weights randomly
elif initialization_method=="zeros":
    initial_mean=np.zeros(num_params) #Initialize weights to zero

#Define CMA-ES istance
es = cma.CMAEvolutionStrategy(initial_mean, mutation_magnitude,{'popsize': lambda_, 'CMA_mu': mu})  # Initialize CMA-ES with zero-mean and 0.5 initial step size

# Run CMA-ES
best_fitness = np.inf
gen_idx=0
print(params)
while gen_idx!=gen_number:
    solutions = es.ask()
    fitness_list = [evaluate_policy(params) for params in solutions]
    es.tell(solutions, fitness_list)
    #es.logger.add()  # Optional: log the progress
    best_fitness=min(fitness_list)
    median_fitness = statistics.median(fitness_list)
    worst_fitness=max(fitness_list)
    print("Generation",gen_idx)
    print("Best fitness:",-best_fitness)
    print("Median fitness:",-median_fitness)
    print('='*40)
    wandb.log({"Generation": gen_idx, "Best Fitness": -best_fitness, "Median Fitness": -median_fitness,
               "Worst Fitness": -worst_fitness})
    gen_idx+=1

# Print the best solution and its fitness
best_solution,best_fitness,index = es.best.get()
evaluateModel(best_solution,env)
print('Best solution:', best_solution)
print('Best fitness:', best_fitness)
wandb.finish()



#testSavedModel(model)