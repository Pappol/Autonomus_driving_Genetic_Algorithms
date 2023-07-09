import os
import gym
import json
import string
import datetime
import argparse
import subprocess
import numpy as np
from time import time, sleep
from numpy import random
import sys
@sys.path.append("ge_q_dts/")
from grammatical_evolution_custom import GrammaticalEvolutionTranslator, grammatical_evolution, differential_evolution
# from ge_q_dts.dt import EpsGreedyLeaf, PythonDT, RandomlyInitializedEpsGreedyLeaf
# import gymnasium
from sklearn.metrics import f1_score

from gymnasium.envs.registration import register
from tqdm import tqdm
import glob
#from grammar_attn import grammar, env_tracker
from grammar_flow import grammar, env_tracker

translator = GrammaticalEvolutionTranslator(grammar, "<init>")

register(
    id='highway-custom-v0',
    entry_point='HighwayEnv.highway_env.envs.highway_env:HighwayEnvCustom',
)

def string_to_dict(x):
    """
    This function splits a string into a dict.
    The string must be in the format: key0-value0#key1-value1#...#keyn-valuen
    """
    result = {}
    items = x.split("#")

    for i in items:
        key, value = i.split("-")
        try:
            result[key] = int(value)
        except:
            try:
                result[key] = float(value)
            except:
                result[key] = value

    return result


parser = argparse.ArgumentParser()
parser.add_argument("--jobs", default=1, type=int, help="The number of jobs to use for the evolution")
parser.add_argument("--eval_step", default=5000, type=int)

parser.add_argument("--lambda_", default=30, type=int, help="Population size")
parser.add_argument("--generations", default=1000, type=int, help="Number of generations")
parser.add_argument("--cxp", default=0.5, type=float, help="Crossover probability")
parser.add_argument("--mp", default=0.5, type=float, help="Mutation probability")
parser.add_argument("--mutation", default="function-tools.mutUniformInt#low-0#up-40000#indpb-0.1", type=string_to_dict, help="Mutation operator. String in the format function-value#function_param_-value_1... The operators from the DEAP library can be used by setting the function to 'function-tools.<operator_name>'. Default: Uniform Int Mutation")
parser.add_argument("--crossover", default="function-tools.cxOnePoint", type=string_to_dict, help="Crossover operator, see Mutation operator. Default: One point")
parser.add_argument("--selection", default="function-tools.selTournament#tournsize-2", type=string_to_dict, help="Selection operator, see Mutation operator. Default: tournament of size 2")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--genotype_len", default=100, type=int, help="Length of the fixed-length genotype")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

# Setup of the logging
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logdir = "logs/{}_{}".format(date, "".join(np.random.choice(list(string.ascii_lowercase), size=8)))
logfile = os.path.join(logdir, "log.txt")
os.makedirs(logdir)

# Log all the parameters
with open(logfile, "a") as f:
    vars_ = locals().copy()
    for k, v in vars_.items():
        f.write("{}: {}\n".format(k, v))

class PythonDTCustomize():
    def __init__(self, phenotype, _=None, verbose=False):
        self.ACTIONS_ALL = {
            0: 'Left',
            1: 'Idle',
            2: 'Right',
            3: 'Faster',
            4: 'Slower'
            }
        self.ACTIONS = {}
        for key, val in self.ACTIONS_ALL.items():
            self.ACTIONS[val] = key
            self.ACTIONS[key] = val

        self.program = phenotype
        if verbose:
            print(" ------------------- program --------------------------")
            print(phenotype)

        self.exec_ = compile(self.program, "<string>", "exec", optimize=2)
        if verbose:
            print("-----------------------------------------------------")

    def new_episode(self):
        #reset
        pass

    def get_action(self, obs):
        variables = {
            "obs":obs,
            "env_tracker":env_tracker,
            "action": None,
            'verbose':False,
            }

        try:
            exec(self.exec_, globals(),variables)
        except:
            print("[ERROR] EXEC FAILED")
            print("--------------------------------------------")
            print(self.program)
            print("=============================================")
            print(variables)
            variables["verbose"] = True
            exec(self.exec_, globals(),variables)
        try:
            assert variables["action"] is not None
        except:
            print("[ERROR] ACTION IS NONE")
            print("--------------------------------------------")
            print(self.program)
            print("=============================================")
            print(variables)
            breakpoint()

        return self.ACTIONS[variables["action"]], variables

    def cvt_action(self, action_no):
        return self.ACTIONS[action_no]


def evaluate_fitness(fitness_function, data, genotype):
    phenotype, _ = translator.genotype_to_str(genotype)
    if len(phenotype) == 0:
        return -1000, None, None
    dt = PythonDTCustomize(phenotype)
    return fitness_function(dt, data)


def fitness(decision_tree, data):
    decision_tree.new_episode()
    reward_norm = np.zeros((5,5))
    cm = {"y_pred":[], "y_true":[]}


    for step_idx in range(len(data)):
        env_step = data[step_idx] # keys: ['obs', 'grad', 'reward', 'action', 'action_prob', 'crashed']; # env_step['grad'].keys() = [str(num)]
        if env_step["action"] < 0:
            continue # TODO:fix this later, add condition crash

        # pred_lob {"select":obs_like_map, "params":obs_like_map} (num_vehicles, 4)
        action_pred, pred_log = decision_tree.get_action(np.array(env_step["obs"]))
        if action_pred is None:
            continue

        cm["y_pred"].append(decision_tree.cvt_action(action_pred))
        cm["y_true"].append(decision_tree.cvt_action(env_step["action"]))

        # RL-AGENT INFORMATION
        # Q_VALUES -> ACTION_PROB
        action_prob = np.array(env_step["action_prob"]).flatten()
        action_prob[env_step["action"]] = np.max(action_prob)

        # STRUCTURE INFORMATION
        dt_p_attn = pred_log["env"].log["params"]
        dt_p_attn = abs(dt_p_attn)/np.sum(abs(dt_p_attn))

        # TRACKING INFO
        def get_cls_grad(cls_idx):
            rl_grad_prob = np.round(np.array(env_step["grad"][str(cls_idx)]),3)[:, 1:]
            # print(rl_grad)
            # rl_grad = np.array([np.e**x for x in rl_grad]).reshape(5,5)[:,1:]
            # rl_grad_prob = np.round(rl_grad/np.sum(rl_grad),3)
            rl_p_attn = rl_grad_prob
            rl_v_attn = 0 #rl_grad_prob.sum(axis=-1).copy()
            return rl_p_attn, rl_v_attn

        def sim(vect1, vect2):
            up_ = np.sum(vect1*vect2)
            down = np.sqrt(np.sum(vect1**2))*np.sqrt(np.sum(vect2**2))
            return up_/down

        def get_norm_prob(cls_idx):
           w_prob = action_prob[cls_idx]/np.max(action_prob)
           assert (w_prob<=1), w_prob
           return w_prob

        # REWARD CALCULATION
        for cls_idx in [action_pred]:
            
            cls_prob = get_norm_prob(cls_idx)


            # STRUCTURE SIMILARITY
            cls_p_attn, cls_v_attn = get_cls_grad(cls_idx)
            structure_sim =  sim(cls_p_attn,dt_p_attn)

            if action_pred == env_step["action"]:
                reward = cls_prob * structure_sim
            else:
                reward = 1-cls_prob

            reward_norm[env_step["action"], action_pred] += reward
            # reward_count[env_step["action"], action_pred] += 1

    cls_reward = {}
    for action_idx in range(5):
        tp = reward_norm[action_idx,action_idx]
        tp_fp = np.sum(reward_norm[action_idx,:])
        tp_tn = np.sum(reward_norm[:,action_idx])
        if tp == 0:
            p, r, f = 0, 0, 0
        else:
            p,r = tp/tp_fp, tp/tp_tn
            f = 2/(1/p+1/r)
        cls_reward[action_idx] = {"fscore":f, "precision":p, "recall":r}    

    fitness = np.mean([reward["fscore"] for reward in cls_reward.values()])
    return fitness, cm, [cls_reward, reward_norm]

if __name__ == '__main__':
    import collections
    from joblib import parallel_backend

    class DataPreprocess:
        def __init__(self, path="data_new/*.json", num_sample=200, is_balance=True, num_cls=5):
            # breakpoint()
            self.dataset = sorted(glob.glob(path))
            self.num_sample = num_sample
            self.is_balance = is_balance
            self.cls_nsample = int(num_sample//num_cls)
            self.num_cls = num_cls
        
        def statistics(self, dataset):
            static = {}
            for data in dataset:
                if data["action"] not in static:
                    static[data["action"]]=1
                else:
                    static[data["action"]]+=1
            print(static)

        
        def select_ratio(self, list_, ratio):
            num_sample = int(len(list_)*ratio)
            selected_idx = np.random.choice(range(len(list_)), num_sample)
            return [list_[idx] for idx in selected_idx]


        def prep_data(self, random_seed=0, prev_data=None, mix_prob=0.5):
            # Preprocess with previous dataset
            prev_storage = {}
            if prev_data is not None:
                for tmp_data in prev_data:
                    if tmp_data["action"] not in prev_storage:
                        prev_storage[tmp_data["action"]] = [tmp_data]
                    else:
                        prev_storage[tmp_data["action"]].append(tmp_data)
                mix_prob = np.round(max(min(mix_prob, 1),0),3)
                print("Exploration ratio:",mix_prob)

            # Grab current dataset
            np.random.seed(random_seed)
            flag_continue, count_sample, storage, dataset = True, 0, {}, []
            while (count_sample < self.num_sample) and flag_continue:
                data_path = np.random.choice(self.dataset)
                data = json.load(open(data_path))
                for step in data:
                    if step["reward"] < 0: continue
                    if not self.is_balance:
                        dataset.append(step)
                        count_sample +=1
                    else:
                        if step["action"] not in storage:
                            storage[step["action"]] = []
                        storage[step["action"]].append(step)

                if self.is_balance:
                    cls_nsamples = [len(samples) for samples in storage.values()]
                    flag_continue = False
                    if np.sum(cls_nsamples) < self.num_sample:
                        flag_continue=True
                    elif len(storage.keys()) < self.num_cls:
                        flag_continue = True
                    else:
                        for sample_len in cls_nsamples:
                            if sample_len < self.cls_nsample*1.5:
                                flag_continue = True

            # Post process
            if not self.is_balance:
                dataset = list(np.random.choice(dataset, self.num_sample))
                dataset = self.select_ratio(dataset, mix_prob) + self.select_ratio(prev_data, 1-mix_prob)
            else:
                assert len(dataset) == 0
                for action_idx, samples in storage.items():
                    if len(samples) != self.cls_nsample:
                        samples = list(np.random.choice(samples, self.cls_nsample))
                    if prev_data is not None:
                        prev_samples = prev_storage[action_idx]
                        if len(prev_samples) != self.cls_nsample:
                            prev_samples = list(np.random.choice(prev_samples, self.cls_nsample))
                        dataset+= self.select_ratio(samples, mix_prob) + self.select_ratio(prev_samples, 1-mix_prob)
                    else:
                        dataset+= samples
            return dataset

    data_ins = DataPreprocess(num_sample=args.eval_step)



    def fit_fcn(x, data):
        return evaluate_fitness(fitness, data, x)

    with parallel_backend("multiprocessing"):
        pop, log, hof, best_leaves = grammatical_evolution(fit_fcn, None, None, individuals=args.lambda_,
            generations=args.generations, jobs=args.jobs, cx_prob=args.cxp, m_prob=args.mp,
            logfile=logfile, seed=args.seed, mutation=args.mutation, crossover=args.crossover,
            initial_len=args.genotype_len, selection=args.selection, translator=translator,data_ins=data_ins)


    # Log best individual
    with open(logfile, "a") as log_:
        phenotype, _ = translator.genotype_to_str(hof[0])
        log_.write(str(log) + "\n")
        log_.write(str(hof[0]) + "\n")
        log_.write(phenotype + "\n")
        log_.write("best_fitness: {}".format(hof[0].fitness.values[0]))
    with open(os.path.join(logdir, "fitness.tsv"), "w") as f:
        f.write(str(log))
    # breakpoint()

