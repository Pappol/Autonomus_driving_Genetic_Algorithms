#!/usr/bin/python3
"""
Implementation of the grammatical evolution

Author: Leonardo Lucio Custode
Creation Date: 04-04-2020
Last modified: mer 6 mag 2020, 16:30:41
"""
import re
import os
import sys
import deap
import random
from deap.algorithms import varAnd
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import datetime
import pathos

from functools import partial
from deap import base, creator, tools, algorithms
from deap.tools import mutShuffleIndexes, mutUniformInt

from dt import EpsGreedyLeaf, PythonDT
import gym
import numpy as np
import multiprocess
from matplotlib import pyplot as plt
import argparse
from scoop import futures
from joblib import Parallel, delayed


TAB = " " * 4

class ListWithParents(list):
    def __init__(self, *iterable):
        super(ListWithParents, self).__init__(*iterable)
        self.parents = []


def get_map(jobs, timeout=None):
    def map_(f, args):
        return Parallel(jobs, timeout=timeout)(delayed(f)(i[0], i[1]) for i in args)
    return map_


def varAnd(population, toolbox, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.  A
    first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
    is returned.

    This variation is named *And* because of its propensity to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematically, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]
    for i, o in enumerate(offspring):
        o.parents = [i] 

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            offspring[i-1].parents.append(i)
            offspring[i].parents.append(i - 1)
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring



def eaSimple(population, toolbox, cxpb, mutpb, ngen, data_ins, stats=None,
             halloffame=None, verbose=__debug__, logfile=None, var=varAnd, translator=None):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'f_score'] + (stats.fields if stats else [])
    re_eval = True
    # dataset = data_ins.prep_data(random_seed=0, prev_data=None)
    dataset = np.random.randint(1,1000)
    print("[LOG] Initialized dataset")
    nstep_explore = 1000
    max_boundary = 0.70
    min_boundary = 0.05
    step_exploration_prob = (max_boundary-min_boundary)/nstep_explore
    mix_prob = 0.5
    def reset_patient(): return 20
    patient = reset_patient()
    def rand_prob(prob_true): return np.random.choice([True, False], p=(prob_true, 1-prob_true))

    # Begin the generational process
    for gen in range(1, ngen + 1):
        patient -=1
        exploration_prob = max(max_boundary-step_exploration_prob*gen, min_boundary)
        if patient > 0:
            re_eval_step = False
        else:
            re_eval_step = re_eval and rand_prob(exploration_prob)
        if re_eval_step:
            patient = reset_patient()
            dataset = np.random.randint(1,1000)
            # dataset = data_ins.prep_data(random_seed=gen, prev_data=dataset, mix_prob=exploration_prob)

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = var(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        if re_eval_step or gen==1:
        #if gen==1:
            invalid_ind = offspring + population
        else:
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = [*toolbox.map(toolbox.evaluate, [(ind, dataset) for ind in invalid_ind])]
        cfs_mtx = [f[1] for f in fitnesses]
        norm_fit = [f[2] for f in fitnesses]
        fitnesses = [f[0] for f in fitnesses]

        for i, (ind, fit, mtx, normed) in enumerate(zip(invalid_ind, fitnesses, cfs_mtx, norm_fit)):
            fit = tuple(np.array(fit).flatten())
            ind.fitness.values = fit
            ind.mtx = mtx
            ind.normed = normed


        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.clear()
            halloffame.update(offspring)
            halloffame.update(population)

        # Replace the current population by the offspring
        for o in offspring:
            argmin = np.argmin(map(lambda x: population[x].fitness.values[0], o.parents))
            # breakpoint()

            if o.fitness.values[0] > population[o.parents[argmin]].fitness.values[0]:
                population[o.parents[argmin]] = o

        from sklearn.metrics import confusion_matrix, f1_score
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        cm = halloffame[0].mtx
        score = np.mean(f1_score(cm["y_true"], cm["y_pred"], average=None))
        logbook.record(gen=gen, nevals=len(invalid_ind), f_score=score, **record)
        if verbose:
            print(logbook.stream)
            if (gen%10 == 0) and (translator is not None):
                phenotype, _ = translator.genotype_to_str(halloffame[0])
                f = open("best.py", "w")
                f.write("from grammar_ import *\ndef dt(obs):\n")
                f.write("    "+phenotype.replace("\n", "\n    ").replace(" action =", " return "))
                f.write("\n #----------------- grammar ---------------------------------\n"+str(translator.operators))
                f.close()
                print(" ---------------- best tree ---------------- ")
                print(phenotype)
                print("------------------ confusion matrix ---------------------")
                from pcmtx import print_cm
                print_cm(halloffame[0].mtx["y_true"], halloffame[0].mtx["y_pred"], labels=["Left", "Idle", "Right", "Faster", "Slower"])
                print("Fitness:", halloffame[0].fitness.values)
                import pandas as pd
                if isinstance(halloffame[0].normed, list):
                    dict_data = halloffame[0].normed[0]
                    mtx = halloffame[0].normed[1]
                else:
                    dict_data = halloffame[0].normed
                    mtx = None
                print(pd.DataFrame(dict_data))
                if mtx is not None:
                    mtx = np.round(mtx, 3)
                    mtx[mtx==0] =-1
                    print(mtx)
                print("---------------------------------------------------------")
                # breakpoint()

    return population, logbook


class GrammaticalEvolutionTranslator:
    def __init__(self, grammar, init_string="<bt>"):
        """
        Initializes a new instance of the Grammatical Evolution
        :param grammar: A dictionary containing the rules of the grammar and their production
        """
        self.operators = grammar
        self.init_string=init_string

    def _find_candidates(self, string):
        return re.findall("<[^> ]+>", string)

    def _find_replacement(self, candidate, gene):
        key = candidate.replace("<", "").replace(">", "")
        value = self.operators[key][gene % len(self.operators[key])]
        return value

    def genotype_to_str(self, genotype):
        """ This method translates a genotype into an executable program (python) """
        string = self.init_string[:]
        candidates = [None]
        ctr = 0
        _max_trials = 1
        genes_used = 0

        # Generate phenotype starting from the genotype
        #   If the individual runs out of genes, it restarts from the beginning, as suggested by Ryan et al 1998
        while len(candidates) > 0 and ctr <= _max_trials:
            if ctr == _max_trials:
                return "", len(genotype)
            for gene in genotype:
                candidates = self._find_candidates(string)
                # tmp = string[:]
                if len(candidates) > 0:
                    value = self._find_replacement(candidates[0], gene)
                    string = string.replace(candidates[0], value, 1)
                    genes_used += 1
                    # breakpoint()
                else:
                    break
            ctr += 1
        string = self._fix_indentation(string)

        return string, genes_used

    def _fix_indentation(self, string):
        # If parenthesis are present in the outermost block, remove them
        if string[0] == "{":
            string = string[1:-1]

        # Split in lines
        string = string.replace(";", "\n")
        string = string.replace("{", "{\n")
        string = string.replace("}", "}\n")
        lines = string.split("\n")

        fixed_lines = []
        n_tabs = 0

        # Fix lines
        for line in lines:
            if len(line) > 0:
                fixed_lines.append(TAB * n_tabs + line.replace("{", "").replace("}", ""))

                if line[-1] == "{":
                    n_tabs += 1
                while len(line) > 0 and line[-1] == "}":
                    n_tabs -= 1
                    line = line[:-1]
                if n_tabs >= 100:
                    return "None"

        return "\n".join(fixed_lines)


def mutate(individual, attribute, mut_rate, max_value):
    random_number = np.random.uniform()
    """
    if random_number < 0.33:
        return ge_mutate(ind, attribute)
    elif random_number < 0.67:
        return mutUniformInt(individual, 0, _max_value, 0.1)
    else:
        return mutShuffleIndexes(individual, 0.1)
    """
    if random_number < 0.5:
        return mutUniformInt(individual, 0, max_value, mut_rate)
    else:
        return mutShuffleIndexes(individual, mut_rate)


def ge_mate(ind1, ind2, individual):
    offspring = tools.cxOnePoint(ind1, ind2)

    if random.random() < 0.5:
        new_offspring = []
        for idx, ind in enumerate([ind1, ind2]):
            _, used = GrammaticalEvolutionTranslator(1, [object], [0]).genotype_to_str(ind)
            if used > len(ind):
                used = len(ind)
            new_offspring.append(individual(offspring[idx][:used]))

        offspring = (new_offspring[0], new_offspring[1])

    return offspring


def ge_mutate(ind, attribute):
    random_int = random.randint(0, len(ind) - 1)
    assert random_int >= 0

    if random.random() < 0.5:
        ind[random_int] = attribute()
    else:
        ind.extend(np.random.choice(ind, size=random_int))
    return ind,


def grammatical_evolution(fitness_function, inputs, leaf, individuals, generations,
    cx_prob, m_prob, data_ins, initial_len=100,
    selection={'function': "tools.selBest"},
    mutation={'function': "ge_mutate", 'attribute': None},
    crossover={'function': "ge_mate", 'individual': None},
    seed=0, jobs=1, logfile=None, timeout=10*60, translator=None, ):

    random.seed(seed)
    np.random.seed(seed)

    _max_value = 40000

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", ListWithParents, typecode='d', fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_bool", random.randint, 0, _max_value)

    # Structure initializers
    if jobs > 1:
        toolbox.register("map", get_map(jobs, timeout))
        # toolbox.register("map", multiprocess.Pool(jobs).map)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, initial_len)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitness_function)

    for d in [mutation, crossover]:
        if "attribute" in d:
            d['attribute'] = toolbox.attr_bool
        if "individual" in d:
            d['individual'] = creator.Individual

    toolbox.register("mate", eval(crossover['function']), **{k: v for k, v in crossover.items() if k != "function"})
    toolbox.register("mutate", eval(mutation['function']), **{k: v for k, v in mutation.items() if k != "function"})
    toolbox.register("select", eval(selection['function']), **{k: v for k, v in selection.items() if k != "function"})

    pop = toolbox.population(n=individuals)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    # breakpoint()
    pop, log, best_leaves = eaSimple(pop, toolbox, cxpb=cx_prob, mutpb=m_prob, ngen=generations, 
                                   stats=stats, halloffame=hof, verbose=True, logfile=logfile, translator=translator, data_ins=data_ins)
    
    return pop, log, hof, best_leaves


def varDE(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]

    for i, o in enumerate(offspring):
        o.parents = [i] 

    b = sorted(range(len(offspring)), key=lambda x: offspring[x].fitness, reverse=True)[0]

    # Apply crossover and mutation on the offspring
    for i in range(len(offspring)):
        j, k, l = np.random.choice([a for a in range(len(offspring)) if a != i], 3, False)
        if random.uniform(0, 1) > 1:
            j = b
        # j, k, l = np.random.choice([a for a in range(len(offspring)) if a != i], 3, False)
        """
        if offspring[k].fitness > offspring[j].fitness:
            j, k = k, j
        elif offspring[l].fitness > offspring[j].fitness:
            j, l = l, j
        """
        offspring[i] = toolbox.mate(offspring[j],
                                    offspring[k],
                                    offspring[l],
                                    offspring[i])
        offspring[i].parents.append(i)
        del offspring[i].fitness.values

    return offspring


def de_mate(a, b, c, x, F, CR, individual):
    a, b, c, x = (np.array(k) for k in [a, b, c, x])
    d = a + F * (b - c)
    d[d < 0] = 0
    positions = np.random.uniform(0, 1, len(x)) < CR
    x[positions] = d[positions]
    return individual(x)


def differential_evolution(fitness_function, inputs, leaf, individuals, generations, cx_prob, m_prob, initial_len=100, selection=None, mutation=None, crossover=None, seed=0, jobs=1, logfile=None, F=1.2, CR=0.9, timeout=10*60):
    """
    NOTE: cx_prob and m_prob are not used here. They are present just to keep the compatibility with the interface
    """
    random.seed(seed)
    np.random.seed(seed)

    _max_value = 40000

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", ListWithParents, typecode='d', fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_bool", random.randint, 0, _max_value)

    # Structure initializers
    if jobs > 1:
        toolbox.register("map", get_map(jobs, timeout))
        # toolbox.register("map", multiprocess.Pool(jobs).map)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, initial_len)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitness_function)

    for d in [mutation, crossover]:
        if "attribute" in d:
            d['attribute'] = toolbox.attr_bool
        if "individual" in d:
            d['individual'] = creator.Individual

    toolbox.register("mate", de_mate, CR=CR, F=F, individual=creator.Individual)
    toolbox.register("select", tools.selRandom)

    pop = toolbox.population(n=individuals)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    pop, log, best_leaves = eaSimple(pop, toolbox, cxpb=cx_prob, mutpb=m_prob, ngen=generations, 
                                   stats=stats, halloffame=hof, verbose=True, logfile=logfile, var=varDE)
    
    return pop, log, hof, best_leaves



if __name__ == '__main__':
    import sys

    n_inputs = int(sys.argv[1])
    max_constant = int(sys.argv[2])
    step_size = int(sys.argv[3])
    divisor = int(sys.argv[4])
    list_ = eval("".join(sys.argv[5:]))

    phenotype = GrammaticalEvolutionTranslator(n_inputs, [EpsGreedyLeaf], [i/divisor for i in range(0, max_constant, step_size)]).genotype_to_str(list_)
    # print(phenotype)
