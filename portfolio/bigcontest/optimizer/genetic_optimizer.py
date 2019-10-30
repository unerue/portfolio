import random
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, log_loss
from .utils import timeit

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import warnings
warnings.filterwarnings('ignore')


class GeneticOptimizer:
    def __init__(self, num_class=None, num_boost_round=50, n_splits=3, pop=10, ngen=10, n_jobs=-1):
        self.num_class = int(num_class)
        self.num_boost_round = int(num_boost_round)
        self.n_splits = int(n_splits)
        self.pop = int(pop)
        self.ngen = int(ngen)
        self.n_jobs = int(n_jobs)
    
    def fit(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_train = y_train.copy()
        self.y_test = y_test.copy()
        
        self._genetic()
    
    def _genetic(self):
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        toolbox.register('eta', random.uniform, 0.01, 0.1)
        toolbox.register('n_estimators', random.randint, 100, 1000)
        toolbox.register('max_depth', random.randint, 5, 15)

        toolbox.register('individual', tools.initCycle, creator.Individual,
                         (toolbox.eta, toolbox.n_estimators, toolbox.max_depth), n=1)
        
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)

        def fitness(individual):
            score = self._training(individual)
            return score, 

        toolbox.register('evaluate', fitness)
        toolbox.register('mate', tools.cxTwoPoint)
        toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
        toolbox.register('select', tools.selTournament, tournsize=3)
        
        pop = toolbox.population(n=self.pop)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('avg', np.mean)
        stats.register('std', np.std)
        stats.register('min', np.min)
        stats.register('max', np.max)
    
        pop, self.log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, 
                                                 ngen=self.ngen, stats=stats, halloffame=hof, verbose=True)
    
        best_ind = tools.selBest(pop, 1)[0]
        print('Best individual is {}\nBest fitness value {}'.format(best_ind, best_ind.fitness.values))
        
    def _training(self, individual):
        params = {
            'objective': 'multi:softprob',
            'num_class': self.num_class,
            'n_jobs': self.n_jobs,
            'seed': 42,
            'silent': True,
            'eta': individual[0],
            'n_estimators': individual[1],
            'max_depth': individual[2],
        }

        train_preds = np.zeros((len(self.X_train), len(np.unique(self.y_train))))
        test_preds = np.zeros((len(self.X_test), len(np.unique(self.y_test))))

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True)
        for i, (train_index, valid_index) in enumerate(skf.split(self.X_train, self.y_train)):
            X_train_data, X_valid_data = self.X_train.values[train_index], self.X_train.values[valid_index]
            y_train_data, y_valid_data = self.y_train.values[train_index], self.y_train.values[valid_index]

            d_train = xgb.DMatrix(X_train_data, y_train_data)
            d_valid = xgb.DMatrix(X_valid_data, y_valid_data)
            d_test = xgb.DMatrix(self.X_test.values)
            watchlist = [(d_train, 'train'), (d_valid, 'valid')]

            clf = xgb.train(params, d_train, num_boost_round=self.num_boost_round, evals=watchlist, verbose_eval=False)
            
            train_preds[valid_index] = clf.predict(d_valid)
            test_preds += clf.predict(d_test) / skf.n_splits
            
        score = f1_score(self.y_test, np.argmax(test_preds, axis=1), average='macro')
        del clf, X_train_data, X_valid_data, y_train_data, y_valid_data, train_preds, test_preds
        return score