import random
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms


class GeneticAlgorithm:
    def __init__(self, d, s, sw, sr, hw, hr):
        self.d = d.copy()
        self.n_items = d.shape[0]
        self.times = d.shape[1]
        
        self.s = s
        self.sw = sw
        self.sr = sr
        
        self.hw = hw
        self.hr = hr

    def _decoding(self, d, i, individual):
        individual = np.array(individual)
        dummies = np.zeros(self.times)
        dummies[-1] = d[i][-1]
        
        for j in range(len(individual)-2, -1, -1):
            if individual[j+1] == 1:
                value = d[i][j]
            else:
                value = d[i][j] + dummies[j+1]
            dummies[j] = value
        
        result = individual * dummies
        
        return result.round(0)

    def _inventory(self, d, i, q):
        result = np.zeros(self.times)
        for j in range(len(q)):
            if j == 0:
                result[j] = 0 + q[j] - d[i][j]
            else:
                result[j] = result[j-1] + q[j] - d[i][j]
        return result.round(0)

    def _fitness(self, individual):
        individual = np.array(individual).reshape((self.n_items * 2),-1)
        fr = individual[:int((self.n_items * 2) / 2)]
        fw = individual[int((self.n_items * 2) / 2):]
        
        fr = np.insert(fr, 0, 1, axis=1)
        fw = np.insert(fw, 0, 1, axis=1)
        
        fr_res = np.zeros((fr.shape[0], fr.shape[1]))
        fw_res = np.zeros((fw.shape[0], fw.shape[1]))
            
        # Decision Quantity
        for i, ind in enumerate(fr):
            value = self._decoding(self.d, i, ind)
            fr_res[i] = value
            
        for i, ind in enumerate(fw):
            value = self._decoding(fr_res, i, ind)
            fw_res[i] = value
        
        # Minor ordering costs
        fr_minor = np.where(fr_res > 0, 1, 0)
        fw_minor = np.where(fw_res > 0, 1, 0)
        fr_minor_cost = fr_minor.sum(1) * self.sr
        fw_minor_cost = fw_minor.sum(1) * self.sw
        
        # Major ordering cost
        fw_major = (fw_minor == 1).any(axis=0)
        
        # Calculate ending inventory
        fr_inv = np.zeros((fr.shape[0], fr.shape[1]))
        fw_inv = np.zeros((fw.shape[0], fw.shape[1]))
            
        for i, ind in enumerate(fr_res):
            value = self._inventory(self.d, i, ind)
            fr_inv[i] = value
        
        for i, ind in enumerate(fw_res):
            value = self._inventory(fr_res, i, ind)
            fw_inv[i] = value
        
        fr_inv_cost = fr_inv.sum(1) * self.hr
        fw_inv_cost = fw_inv.sum(1) * self.hw
        
        tc = (fw_major.sum() * self.s) + fr_minor_cost.sum() + fw_minor_cost.sum() + fr_inv_cost.sum() + fw_inv_cost.sum()

        return tc,

    def optimize(self, n=100, ngen=10, verbose=True):
        creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register('choromosome', random.randint, 0, 1)
        toolbox.register('individual', tools.initRepeat, creator.Individual, 
                         toolbox.choromosome, n=(self.n_items * self.times * 2) - (self.n_items * 2))
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)

        toolbox.register('evaluate', self._fitness)
        toolbox.register('mate', tools.cxTwoPoint) # tools.cxOrdered
        toolbox.register('mutate', tools.mutFlipBit, indpb=0.5) # tools.mutShuffleIndexes, indpb = 0.6) # tools.mutFlipBit
        toolbox.register('select', tools.selTournament, tournsize=3) # tools.selTournament, tournsize=3) # tools.selRoulette

        pop = toolbox.population(n=n)
        # hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('avg', np.mean)
        stats.register('std', np.std)
        stats.register('min', np.min)
        stats.register('max', np.max)
        
        pop, self.log = algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.4, ngen=ngen, 
                                            stats=stats, verbose=verbose)
        

class GeneticAlgorithmQD:
    def __init__(self, d, s, sw, sr, hw, hr, q, p):
        self.d = d.copy()
        self.n_items = d.shape[0]
        self.times = d.shape[1]
        
        self.s = s
        self.sw = sw
        self.sr = sr
        
        self.hw = hw
        self.hr = hr

        self.q = q
        self.p = p

    def _decoding(self, d, i, individual):
        individual = np.array(individual)
        dummies = np.zeros(self.times)
        dummies[-1] = d[i][-1]
        
        for j in range(len(individual)-2, -1, -1):
            if individual[j+1] == 1:
                value = d[i][j]
            else:
                value = d[i][j] + dummies[j+1]
            dummies[j] = value
        
        result = individual * dummies
        
        return result.rount(0)

    def _inventory(self, d, i, q):
        result = np.zeros(self.times)
        for j in range(len(q)):
            if j == 0:
                result[j] = 0 + q[j] - d[i][j]
            else:
                result[j] = result[j-1] + q[j] - d[i][j]
        return result.round(0)

    def _fitness(self, individual):
        individual = np.array(individual).reshape((self.n_items * 2),-1)
        fr = individual[:int((self.n_items * 2) / 2)]
        fw = individual[int((self.n_items * 2) / 2):]
        
        fr = np.insert(fr, 0, 1, axis=1)
        fw = np.insert(fw, 0, 1, axis=1)
        
        fr_res = np.zeros((fr.shape[0], fr.shape[1]))
        fw_res = np.zeros((fw.shape[0], fw.shape[1]))
            
        # Decision Quantity
        for i, ind in enumerate(fr):
            value = self._decoding(self.d, i, ind)
            fr_res[i] = value
            
        for i, ind in enumerate(fw):
            value = self._decoding(fr_res, i, ind)
            fw_res[i] = value
        
        # Minor ordering costs
        fr_minor = np.where(fr_res > 0, 1, 0)
        fw_minor = np.where(fw_res > 0, 1, 0)
        fr_minor_cost = fr_minor.sum(1) * self.sr
        fw_minor_cost = fw_minor.sum(1) * self.sw
        
        # Major ordering cost
        fw_major = (fw_minor == 1).any(axis=0)
        
        # Compute quantity discounts
        pcost = 0
        for i, (v1, v2) in enumerate(zip(fw_res, self.q.values())):
            for j, v3 in enumerate(v2):
                if (v1[(v1 >= v3[0]) & (v1 <= v3[1])]).sum():
                    pcost += (v1[(v1 >= v3[0]) & (v1 <= v3[1])]).sum() * self.p[i][j]            

        # Calculate ending inventory
        fr_inv = np.zeros((fr.shape[0], fr.shape[1]))
        fw_inv = np.zeros((fw.shape[0], fw.shape[1]))
            
        for i, ind in enumerate(fr_res):
            value = self._inventory(self.d, i, ind)
            fr_inv[i] = value
        
        for i, ind in enumerate(fw_res):
            value = self._inventory(fr_res, i, ind)
            fw_inv[i] = value
        
        fr_inv_cost = fr_inv.sum(1) * self.hr
        fw_inv_cost = fw_inv.sum(1) * self.hw
        
        tc = (fw_major.sum() * self.s) + fr_minor_cost.sum() + fw_minor_cost.sum() \
                + fr_inv_cost.sum() + fw_inv_cost.sum() + pcost

        return tc,

    def optimize(self, n=100, ngen=10, verbose=True):
        creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register('choromosome', random.randint, 0, 1)
        toolbox.register('individual', tools.initRepeat, creator.Individual, 
                         toolbox.choromosome, n=(self.n_items * self.times * 2) - (self.n_items * 2))
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)

        toolbox.register('evaluate', self._fitness)
        toolbox.register('mate', tools.cxTwoPoint) # tools.cxOrdered, cxTwoPoint
        toolbox.register('mutate', tools.mutFlipBit, indpb=0.5) # tools.mutShuffleIndexes, indpb = 0.6) # tools.mutFlipBit 0.05
        toolbox.register('select', tools.selTournament, tournsize=3) # tools.selTournament, tournsize=3) # tools.selRoulette

        pop = toolbox.population(n=n)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('avg', np.mean)
        stats.register('std', np.std)
        stats.register('min', np.min)
        stats.register('max', np.max)
        
        pop, self.log = algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.4, ngen=ngen, 
                                            stats=stats, halloffame=hof, verbose=verbose)
        

    