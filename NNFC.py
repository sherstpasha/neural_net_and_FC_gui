import numpy as np
import itertools
from deap import gp
import pandas as pd
from scipy.stats import rankdata

from functools import reduce
import copy
from copy import deepcopy
from inspect import isclass
import random
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import os

import tkinter as tk
from tkinter import filedialog
import pandas as pd
from pandastable import Table, TableModel
import numpy as np
from sklearn.preprocessing import scale
from threading import Thread
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import networkx as nx


def sigma(x):
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    x_ = x.copy()
    x_[x_ < 0] = 0.01 * x_[x_ < 0]
    return x_


def linear(x):
    return x


def th(x):
    left = np.exp(x)
    right = np.exp(-x)
    down = left + right
    down[down == 0] = 1.0
    return (left - right) / down


def gauss(x):
    return np.exp(-(x**2))


def softmax(x):
    exps = np.exp(
        x
        - x.max(axis=0)[
            np.newaxis :,
            :,
        ]
    )
    sum_ = np.sum(exps, axis=1)[:, np.newaxis]
    sum_[sum_ == 0] = 1
    return np.nan_to_num(exps / sum_)


def cat_crossentropy(target, output):
    output = np.clip(output, 1e-7, 1 - 1e-7)
    return np.mean(
        np.sum(target[:, : np.newaxis] * -np.log(output), axis=2, keepdims=False),
        axis=1,
    )


class SelfCGA:

    def funk_tour(self, x):
        return np.random.choice(self.arr_pop_size, size=self.tour_size, replace=False)

    def __init__(
        self, function, iters, pop_size, len_, tour_size=3, K=2, threshold=0.1
    ):

        self.function = function
        self.iters = iters
        self.pop_size = pop_size
        self.len_ = len_
        self.tour_size = tour_size

        self.thefittest = {"individ": None, "fitness": None}

        self.arr_pop_size = np.arange(pop_size, dtype=int)
        self.row_cut = np.arange(1, len_ - 1, dtype=int)
        self.row = np.arange(len_, dtype=int)

        self.m_sets = {
            "average": 1 / (self.len_),
            "strong": min(1, 3 / self.len_),
            "weak": 1 / (3 * self.len_),
        }
        self.c_sets = {
            "one_point": self.one_point_crossing,
            "two_point": self.two_point_crossing,
            "uniform": self.uniform_crossing,
        }
        self.s_sets = {
            "proportional": self.proportional_selection,
            "rank": self.rank_selection,
            "tournament": self.tournament_selection,
        }

        self.operators_list = [
            self.m_sets.keys(),
            self.c_sets.keys(),
            self.s_sets.keys(),
        ]

        self.K = K
        self.threshold = threshold

        self.stats_fitness = np.array([])
        self.stats_proba_m = np.zeros(shape=(0, len(self.m_sets.keys())))
        self.stats_proba_c = np.zeros(shape=(0, len(self.c_sets.keys())))
        self.stats_proba_s = np.zeros(shape=(0, len(self.s_sets.keys())))

    def mutation(self, population, probability):
        population = population.copy()
        roll = np.random.random(size=population.shape) < probability
        population[roll] = 1 - population[roll]
        return population

    def one_point_crossing(self, individ_1, individ_2):
        cross_point = np.random.choice(self.row, size=1)[0]
        if np.random.random() > 0.5:
            offspring = individ_1.copy()
            offspring[:cross_point] = individ_2[:cross_point]
            return offspring
        else:
            offspring = individ_2.copy()
            offspring[:cross_point] = individ_1[:cross_point]
            return offspring

    def two_point_crossing(self, individ_1, individ_2):
        c_point_1, c_point_2 = np.sort(
            np.random.choice(self.row, size=2, replace=False)
        )
        if np.random.random() > 0.5:
            offspring = individ_1.copy()
            offspring[c_point_1:c_point_2] = individ_2[c_point_1:c_point_2]
        else:
            offspring = individ_2.copy()
            offspring[c_point_1:c_point_2] = individ_1[c_point_1:c_point_2]
        return offspring

    def uniform_crossing(self, individ_1, individ_2):
        roll = np.random.random(size=individ_1.shape[0]) > 0.5
        if np.random.random() > 0.5:
            offspring = individ_1.copy()
            offspring[roll] = individ_2[roll]
        else:
            offspring = individ_2.copy()
            offspring[roll] = individ_1[roll]
        return offspring

    def proportional_selection(self, population, fitness):
        max_ = fitness.max()
        min_ = fitness.min()
        if max_ == min_:
            fitness_n = np.ones(fitness.shape)
        else:
            fitness_n = (fitness - min_) / (max_ - min_)

        probability = fitness_n / fitness_n.sum()
        offspring = population[
            np.random.choice(self.arr_pop_size, size=1, p=probability)
        ][0].copy()
        return offspring

    def tournament_selection(self, population, fitness):
        tournament = np.random.choice(
            self.arr_pop_size, size=self.tour_size, replace=False
        )
        max_fit_id = np.argmax(fitness[tournament])
        return population[tournament[max_fit_id]]

    def rank_selection(self, population, fitness):
        ranks = rankdata(fitness)
        probability = ranks / np.sum(ranks)
        offspring = population[
            np.random.choice(self.arr_pop_size, size=1, p=probability)
        ][0]
        return offspring

    def __update_thefittest(self, population, fitness):
        temp_best_id = np.argmax(fitness)
        temp_best_fitness = fitness[temp_best_id]
        if temp_best_fitness > self.thefittest["fitness"]:
            self.thefittest["fitness"] = temp_best_fitness
            self.thefittest["individ"] = population[temp_best_id].copy()

    def __update_statistic(self, fitness, m_proba, c_proba, s_proba):
        self.stats_fitness = np.append(self.stats_fitness, np.max(fitness))
        self.stats_proba_m = np.vstack([self.stats_proba_m, c_proba])
        self.stats_proba_c = np.vstack([self.stats_proba_c, c_proba])
        self.stats_proba_s = np.vstack([self.stats_proba_s, s_proba])
        for proba, proba_list in zip(
            [m_proba, c_proba, s_proba],
            [self.stats_proba_m, self.stats_proba_c, self.stats_proba_s],
        ):
            proba_list = np.append(proba_list, proba)

    def __update_proba(self, proba, z, operators_fitness, fitness):
        new_proba = proba.copy()

        operators_fitness = np.vstack([operators_fitness, fitness]).T
        operators_fitness = operators_fitness[operators_fitness[:, 0].argsort()]
        cut_index = np.unique(operators_fitness[:, 0], return_index=True)[1]
        groups = np.split(operators_fitness[:, 1].astype(float), cut_index)[1:]

        mean_fit = np.array(list(map(np.mean, groups)))

        new_proba[mean_fit.argmax()] = (
            new_proba[mean_fit.argmax()] + self.K / self.iters
        )
        new_proba = new_proba - self.K / (z * self.iters)
        new_proba = new_proba.clip(self.threshold, 1)
        new_proba = new_proba / new_proba.sum()

        return new_proba

    def choice_operators(self, operators, proba):
        return np.random.choice(list(operators), self.pop_size, p=proba)

    def create_offspring(self, operators, popuation, fitness):
        mutation, crossover, selection = operators

        parent_1 = self.s_sets[selection](popuation, fitness)
        parent_2 = self.s_sets[selection](popuation, fitness)
        offspring_no_mutated = self.c_sets[crossover](parent_1, parent_2)
        offspring_mutated = self.mutation(offspring_no_mutated, self.m_sets[mutation])
        return offspring_mutated

    def fit(self, in_population=None):

        z_list = [len(self.m_sets), len(self.c_sets), len(self.s_sets)]

        m_proba = np.full(z_list[0], 1 / z_list[0])
        c_proba = np.full(z_list[1], 1 / z_list[1])
        s_proba = np.full(z_list[2], 1 / z_list[2])

        population = np.random.randint(
            low=2, size=(self.pop_size, self.len_), dtype=np.byte
        )
        if in_population is not None:
            population[-1] = in_population
        fitness = self.function(population)

        self.thefittest["individ"] = population[np.argmax(fitness)].copy()
        self.thefittest["fitness"] = fitness[np.argmax(fitness)].copy()
        self.__update_statistic(fitness, m_proba, c_proba, s_proba)

        for i in range(1, self.iters):

            chosen_operators = list(
                map(
                    self.choice_operators,
                    self.operators_list,
                    [m_proba, c_proba, s_proba],
                )
            )
            chosen_operators = np.array(chosen_operators).T

            def func(x):
                return self.create_offspring(x, population, fitness)

            population = np.array(list(map(func, chosen_operators)))
            fitness = self.function(population)

            def func(x, y, z):
                return self.__update_proba(x, y, z, fitness)

            m_proba, c_proba, s_proba = list(
                map(func, [m_proba, c_proba, s_proba], z_list, chosen_operators.T)
            )
            fitness[-1] = self.thefittest["fitness"].copy()
            population[-1] = self.thefittest["individ"].copy()

            self.__update_statistic(fitness, m_proba, c_proba, s_proba)
            self.__update_thefittest(population, fitness)

        return self


class SamplingGrid:

    def __init__(self, borders, parts):
        self.borders = borders
        self.parts = parts
        self.h = np.abs(borders["right"] - borders["left"]) / (2.0**parts - 1)

    @staticmethod
    def __decoder(population_parts, left_i, h_i):
        ipp = population_parts.astype(int)
        int_convert = np.sum(ipp * (2 ** np.arange(ipp.shape[1], dtype=int)), axis=1)
        return left_i + h_i * int_convert

    def transform(self, population):
        splits = np.add.accumulate(self.parts)
        p_parts = np.split(population, splits[:-1], axis=1)
        fpp = [
            self.__decoder(p_parts_i, left_i, h_i)
            for p_parts_i, left_i, h_i in zip(p_parts, self.borders["left"], self.h)
        ]
        return np.vstack(fpp).T


class Net:

    def __init__(
        self,
        i=set([]),
        h=[],
        o=set([]),
        c=np.zeros((0, 2), dtype=int),
        w=np.array([], dtype=float),
        a=dict(),
    ):
        self.inputs = i
        self.hiddens = h
        self.outputs = o
        self.connects = c
        self.weights = w
        self.activs = a

        self.act_dict = {"rl": relu, "sg": sigma, "th": th, "gs": gauss}

    @staticmethod
    def __merge_layers(layers):
        return layers[0].union(layers[1])

    def __add__(self, other):
        len_i_1, len_i_2 = len(self.inputs), len(other.inputs)
        len_h_1, len_h_2 = len(self.hiddens), len(other.hiddens)

        if (len_i_1 > 0 and len_i_2 == 0) and (len_h_1 == 0 and len_h_2 > 0):
            return self > other
        elif (len_i_1 == 0 and len_i_2 > 0) and (len_h_1 > 0 and len_h_2 == 0):
            return other > self

        map_res = map(self.__merge_layers, zip(self.hiddens, other.hiddens))
        if len_h_1 < len_h_2:
            excess = other.hiddens[len_h_1:]
        elif len_h_1 > len_h_2:
            excess = self.hiddens[len_h_2:]
        else:
            excess = []

        hidden = list(map_res) + excess
        return Net(
            i=self.inputs.union(other.inputs),
            h=hidden,
            o=self.outputs.union(other.outputs),
            c=np.vstack([self.connects, other.connects]),
            w=np.hstack([self.weights, other.weights]),
            a={**self.activs, **other.activs},
        )

    def assemble_hiddens(self):
        if len(self.hiddens) > 0:
            return set.union(*self.hiddens)
        else:
            return set([])

    @staticmethod
    def connect(left, right):
        if len(left) and len(right):
            connects = np.array(list(itertools.product(left, right)))
            weights = np.random.normal(0, 1, len(connects))
            return connects, weights
        else:
            return (np.zeros((0, 2), dtype=int), np.zeros((0), dtype=float))

    def __gt__(self, other):
        len_i_1, len_i_2 = len(self.inputs), len(other.inputs)
        len_h_1, len_h_2 = len(self.hiddens), len(other.hiddens)

        if (len_i_1 > 0 and len_h_1 == 0) and (len_i_2 > 0 and len_h_2 == 0):
            return self + other
        elif (len_i_1 == 0 and len_h_1 > 0) and (len_i_2 > 0 and len_h_2 == 0):
            return other > self

        inputs_hidden = self.inputs.union(self.assemble_hiddens())
        from_ = inputs_hidden.difference(self.connects[:, 0])

        cond = other.connects[:, 0][:, np.newaxis] == np.array(list(other.inputs))
        cond = np.any(cond, axis=1)

        connects_no_i = other.connects[:, 1][~cond]
        hidden_outputs = other.assemble_hiddens().union(other.outputs)
        to_ = hidden_outputs.difference(connects_no_i)

        connects, weights = self.connect(from_, to_)
        return Net(
            i=self.inputs.union(other.inputs),
            h=self.hiddens + other.hiddens,
            o=self.outputs.union(other.outputs),
            c=np.vstack([self.connects, other.connects, connects]),
            w=np.hstack([self.weights, other.weights, weights]),
            a={**self.activs, **other.activs},
        )

    def fix(self, inputs):
        hidden_outputs = self.assemble_hiddens().union(self.outputs)
        to_ = hidden_outputs.difference(self.connects[:, 1])
        if len(to_) > 0:
            if not len(self.inputs):
                self.inputs = inputs

            connects, weights = self.connect(self.inputs, to_)
            self.connects = np.vstack([self.connects, connects])
            self.weights = np.hstack([self.weights, weights])

        biass = np.max(list(inputs)) + 1
        connects, weights = self.connect({biass}, hidden_outputs)

        self.inputs.add(biass)
        self.connects = np.vstack([self.connects, connects])
        self.weights = np.hstack([self.weights, weights])

        self.connects = np.unique(self.connects, axis=0)
        self.weights = self.weights[: len(self.connects)]
        return self

    @staticmethod
    def map_dot(left, right):
        def fdot(x, y):
            return x.T @ y

        return np.array(list(map(fdot, left, right)))

    def forward(self, X, w=None):
        X_shape_1 = X.shape[1]
        if w is None:
            weight = self.weights.reshape(1, -1)
        else:
            weight = w

        hidden = self.assemble_hiddens()

        nodes = np.zeros((X_shape_1 + len(hidden) + len(self.outputs), X.shape[0]))

        from_ = self.connects[:, 0]
        to_ = self.connects[:, 1]
        list_inputs = list(self.inputs)

        nodes[list_inputs] = X.T[list_inputs]
        nodes = np.array([nodes.copy() for _ in weight])

        calculated = self.inputs.copy()

        bool_hidden = to_[:, np.newaxis] == np.array(list(hidden))
        order = {j: set(from_[bool_hidden[:, i]]) for i, j in enumerate(hidden)}

        current = self.inputs.union(hidden)
        while calculated != current:
            for i in hidden:
                if order[i].issubset(calculated) and i not in calculated:
                    cond = to_ == i
                    from_i = from_[cond]
                    weight_i = weight[:, cond]

                    i_dot_w_sum = self.map_dot(nodes[:, from_i], weight_i)
                    i_dot_w_sum = np.clip(i_dot_w_sum, -700, 700)

                    f = self.act_dict[self.activs[i]]
                    nodes[:, i] = f(i_dot_w_sum)
                    calculated.add(i)

        for i in self.outputs:
            cond = to_ == i
            from_i = from_[cond]
            weight_i = weight[:, cond]
            i_dot_w_sum = self.map_dot(nodes[:, from_i], weight_i)
            i_dot_w_sum = np.clip(i_dot_w_sum, -700, 700)
            nodes[:, i] = i_dot_w_sum

        out = softmax(nodes[:, list(self.outputs)].T)
        return np.transpose(out, axes=(2, 0, 1))

    def predict(self, X):
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        proba = self.forward(X)[0]
        y_predict = np.argmax(proba, axis=1)
        return y_predict

    def float_int_grid(self, X_float, n_bins, left, right):
        left = min([X_float.min(), left])
        right = max([X_float.max(), right])
        borders = {"left": np.full(1, left), "right": np.full(1, right)}

        parts = np.full(1, n_bins)

        sg = SamplingGrid(borders, parts)
        bins = np.array(list(itertools.product([0, 1], repeat=n_bins)), dtype="byte")
        bins_float = sg.transform(bins)

        def f(arg):
            return np.argmin(np.abs(arg - bins_float[:, 0]))

        argmin = np.array(list(map(f, X_float)))

        return bins[argmin], left, right

    def get_fitness(self, population, x_true, y_true):
        error = cat_crossentropy(y_true, self.forward(x_true, population))
        return -error

    def fit_SelfCGA(self, X, y, n_bit, pop_size, iters):
        vars_ = len(self.weights)
        eye = np.eye(len(set(y)))

        weights_bin, left, right = self.float_int_grid(self.weights, n_bit, -3, 3)

        borders = {"left": np.full(vars_, left), "right": np.full(vars_, right)}

        parts = np.full(vars_, n_bit)

        grid_model = SamplingGrid(borders, parts)

        def fitness(attr):
            return self.get_fitness(grid_model.transform(attr), X, eye[y])

        opt_model = SelfCGA(
            fitness,
            pop_size=pop_size,
            iters=iters,
            len_=np.sum(parts),
            tour_size=3,
            K=2,
            threshold=0.1,
        )

        opt_model.fit(weights_bin.flatten())

        thefittest = opt_model.thefittest
        self.weights = grid_model.transform(thefittest["individ"].reshape(1, -1))[0]
        # print(thefittest["fitness"])


def add(x, y):
    return x + y


def more(x, y):
    return x > y


class GPNN:

    def um_low(self, x):
        return self.uniform_mutation(x, 0.25)

    def um_mean(self, x):
        return self.uniform_mutation(x, 1)

    def um_strong(self, x):
        return self.uniform_mutation(x, 4)

    def pm_low(self, x):
        return self.one_point_mutation(x, 0.25)

    def pm_mean(self, x):
        return self.one_point_mutation(x, 1)

    def pm_strong(self, x):
        return self.one_point_mutation(x, 4)

    def __init__(self, iters, pop_size, tour_size=5, max_height=5, ngram=3):

        self.iters = iters
        self.pop_size = pop_size
        self.tour_size = tour_size
        self.max_height = max_height
        self.ngram = ngram

        self.thefittest = {"individ": None, "fitness": None, "net": None}
        self.stats = pd.DataFrame(columns=["max", "median", "min", "std"])
        self.pset = None

        self.in_dict = None

        self.sl_dict = {
            "1rl": (1, "rl"),
            "2rl": (2, "rl"),
            "1sg": (1, "sg"),
            "2sg": (2, "sg"),
            "1th": (1, "th"),
            "2th": (2, "th"),
            "1gs": (1, "gs"),
            "2gs": (2, "gs"),
        }

        self.m_sets = {
            "uniform_low": self.um_low,
            "uniform_mean": self.um_mean,
            "uniform_strong": self.um_strong,
            "point_low": self.pm_low,
            "point_mean": self.pm_mean,
            "point_strong": self.pm_strong,
        }

        self.s_sets = {
            "tournament": self.tournament_selection,
            "rank": self.rank_selection,
            "proportional": self.proportional_selection,
        }

        self.c_sets = {
            "standart": self.standart_crossing,
            "one_point": self.one_point_crossing,
            "empty": self.empty_crossing,
        }

        self.r_sets = {"iters": 0.5, "both": 0, "pop_size": -0.3}

        self.arr_pop_size = np.arange(pop_size, dtype=int)

        self.runs = 0

        self.fittest_history = []

    def defining_variables(self, n_vars, ngram=2, shuffle=False):
        n_vars = n_vars - 1
        arrange = np.arange(n_vars, dtype=int)
        if shuffle:
            np.random.shuffle(arrange)
        temp = n_vars // ngram
        n = temp * ngram
        lost = arrange[n:]

        combs = arrange[:n].reshape(-1, ngram)

        self.in_dict = {"in" + str(i): set(comb) for i, comb in enumerate(combs)}
        if len(lost) > 0:

            self.in_dict["in" + str(len(combs))] = set(lost)

        f_terms = len(self.sl_dict)
        all_terms = len(self.in_dict) + len(self.sl_dict)
        self.pset = gp.PrimitiveSet("MAIN", all_terms)
        self.pset.addPrimitive(add, 2)
        self.pset.addPrimitive(more, 2)
        for i, key in enumerate(self.sl_dict.keys()):
            eval("self.pset.renameArguments(ARG" + str(i) + "='" + str(key) + "')")
        for i, key in enumerate(self.in_dict.keys()):
            eval(
                "self.pset.renameArguments(ARG"
                + str(f_terms + i)
                + "='"
                + str(key)
                + "')"
            )

    def generate_tree(self):
        return gp.PrimitiveTree(gp.genHalfAndHalf(self.pset, 2, 5))

    def empty_crossing(self, ind1, ind2):
        if np.random.random() > 0.5:
            return ind1
        else:
            return ind2

    def compile_(self, tree, outs):
        from operator import or_

        string = ""
        stack = []
        origin = max(reduce(or_, self.in_dict.values())) + 2
        activ = {}
        for i, node in enumerate(tree):
            stack.append((node, []))

            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args = stack.pop()
                temp = prim.format(*args)
                if len(args) == 0:
                    if temp in self.sl_dict:
                        hiddens = set(range(origin, origin + self.sl_dict[temp][0]))
                        act_funk = self.sl_dict[temp][1]
                        temp_act = {i: act_funk for i in hiddens}
                        activ = {**activ, **temp_act}
                        origin = origin + self.sl_dict[temp][0]
                        string = "Net(h=[" + str(hiddens) + "])"

                    else:
                        string = "Net(i=" + str(self.in_dict[temp]) + ")"
                else:
                    string = prim.format(*args)
                if len(stack) == 0:
                    break
                stack[-1][1].append(string)
        outs = set(range(origin, origin + outs))

        to_return = eval(string + "> Net(o=" + str(outs) + ")")

        to_return.activs = activ
        to_return = to_return.fix(set.union(*self.in_dict.values()))

        return to_return

    @staticmethod
    def mark_tree(tree):
        stack = []
        current = ""
        n_arg = "0"
        markers = np.array([])
        for k, node in enumerate(tree):
            current += n_arg
            markers = np.append(markers, current)
            if node.arity == 0:
                if len(stack) > 0:
                    n_arg = "1"
                    current = stack.pop()
            elif node.arity == 1:
                n_arg = "0"
            else:
                stack.append(current)
                n_arg = "0"
        return markers

    def expr_mut(self, pset, type_, len_):
        return gp.genGrow(pset, 0, len_, type_)

    @staticmethod
    def replace_node(node, pset):
        def filter_(x):
            return x != node

        if node.arity == 0:  # Terminal
            pool = list(filter(filter_, pset.terminals[node.ret]))
            term = random.choice(pool)
            if isclass(term):
                term = term()
            return term
        else:  # Primitive
            pool = list(filter(filter_, pset.primitives[node.ret]))
            prims = [p for p in pool if p.args == node.args]
            return random.choice(prims)

    def one_point_mutation(self, some_net, proba):
        some_net = copy.deepcopy(some_net)
        proba = proba / len(some_net)
        for i, node in enumerate(some_net):
            if np.random.random() < proba:

                some_net[i] = self.replace_node(node, self.pset)

        return some_net

    def uniform_mutation(self, some_net, proba):
        some_net = copy.deepcopy(some_net)
        proba = proba / len(some_net)
        for i, node in enumerate(some_net[1:]):
            i = i + 1
            if np.random.random() < proba:
                slice_ = some_net.searchSubtree(i)
                type_ = node.ret
                some_net[slice_] = self.expr_mut(
                    pset=self.pset, type_=type_, len_=len(some_net[slice_])
                )
                break

        return some_net

    def rank_selection(self, population, fitness):
        ranks = rankdata(fitness)
        probability = ranks / np.sum(ranks)
        ind = np.random.choice(self.arr_pop_size, size=1, p=probability)
        offspring = population[ind][0]
        return copy.deepcopy(offspring), fitness[ind][0]

    def tournament_selection(self, population, fitness):
        tournament = np.random.choice(
            self.arr_pop_size, size=self.tour_size, replace=False
        )
        max_fit_id = np.argmax(fitness[tournament])
        return (
            copy.deepcopy(population[tournament[max_fit_id]]),
            fitness[tournament[max_fit_id]],
        )

    def proportional_selection(self, population, fitness):
        max_ = fitness.max()
        min_ = fitness.min()
        if max_ == min_:
            fitness_n = np.ones(fitness.shape)
        else:
            fitness_n = (fitness - min_) / (max_ - min_)

        probability = fitness_n / fitness_n.sum()
        ind = np.random.choice(self.arr_pop_size, size=1, p=probability)
        offspring = population[ind][0]

        return copy.deepcopy(offspring), fitness[ind][0]

    @staticmethod
    def standart_crossing(ind_1, ind_2):
        offs_1, offs_2 = gp.cxOnePoint(copy.deepcopy(ind_1), copy.deepcopy(ind_2))
        if np.random.random() > 0.5:
            return offs_1
        else:
            return offs_2

    def one_point_crossing(self, ind1, ind2):
        ind1 = copy.deepcopy(ind1)
        ind2 = copy.deepcopy(ind2)
        if len(ind1) < 2 or len(ind2) < 2:
            if np.random.random() > 0.5:
                return ind1
            else:
                return ind2
        mark_1 = self.mark_tree(ind1)
        mark_2 = self.mark_tree(ind2)
        common, c_1, c_2 = np.intersect1d(mark_1, mark_2, return_indices=True)

        index = random.choice(range(1, len(c_1)))
        index1 = c_1[index]
        index2 = c_2[index]

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

        if np.random.random() > 0.5:
            return ind1
        else:
            return ind2

    def __update_statistic(self, fitness):
        self.stats = self.stats.append(
            {
                "max": fitness.max(),
                "min": fitness.min(),
                "median": np.median(fitness),
                "std": fitness.std(),
            },
            ignore_index=True,
        )

    def __update_thefittest(self, population, population_net, fitness):
        temp_best_id = np.argmax(fitness)
        temp_best_fitness = fitness[temp_best_id]
        if temp_best_fitness > self.thefittest["fitness"]:
            self.fittest_history.append(
                [
                    str(copy.deepcopy(population[temp_best_id])),
                    copy.copy(temp_best_fitness),
                ]
            )
            self.thefittest["fitness"] = temp_best_fitness
            self.thefittest["individ"] = copy.deepcopy(population[temp_best_id])
            self.thefittest["net"] = copy.deepcopy(population_net[temp_best_id])

    @staticmethod
    def find_vars(tree):
        res = np.unique([node.name for node in tree if type(node) == gp.Terminal])
        return res

    def train_and_test(
        self, ind, net, X_train, y_train, X_test, y_test, n_iters, n_size
    ):

        height = ind.height
        if height > self.max_height:
            fine_h = height
        else:
            fine_h = 0.0

        complexity = len(net.connects)

        net.fit_SelfCGA(X_train, y_train, 12, n_size, n_iters)
        proba = net.forward(X_test)

        fitnes_value = 1 / (
            1 + cat_crossentropy(y_test, proba) + 0.00001 * complexity + 0.1 * fine_h
        )

        return fitnes_value, net

    def predict(self, X):
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        proba = self.thefittest["net"].forward(X)[0]
        y_predict = np.argmax(proba, axis=1)
        return y_predict


class selfCGPNN(GPNN):

    def __init__(
        self,
        iters,
        pop_size,
        tour_size=3,
        max_height=10,
        K=0.5,
        threshold=0.1,
        ngram=3,
        min_res=1000,
        max_res=2000,
    ):

        super().__init__(iters, pop_size, tour_size, max_height, ngram)

        self.operators_list = ["mutation", "crossing", "selection", "resource"]
        self.min_res = min_res
        self.max_res = max_res
        self.K = K
        self.threshold = threshold
        self.stats = {
            "fitness": pd.DataFrame(columns=["max", "median", "min", "std"]),
            "proba": {
                "mutation": pd.DataFrame(columns=self.m_sets.keys()),
                "crossing": pd.DataFrame(columns=self.c_sets.keys()),
                "selection": pd.DataFrame(columns=self.s_sets.keys()),
                "resource": pd.DataFrame(columns=self.r_sets.keys()),
            },
        }

    def __update_statistic(self, fitness, m_proba, c_proba, s_proba, r_proba):
        self.stats["fitness"] = self.stats["fitness"].append(
            {
                "max": fitness.max(),
                "min": fitness.min(),
                "median": np.median(fitness),
                "std": fitness.std(),
            },
            ignore_index=True,
        )

        for proba, oper in zip(
            [m_proba, c_proba, s_proba, r_proba], self.operators_list
        ):
            self.stats["proba"][oper] = self.stats["proba"][oper].append(
                proba.copy(), ignore_index=True
            )

    def __update_proba(self, some_proba, operator, some_history, z):
        mutate_avg = some_history.groupby(operator).mean()["fitness"]
        argmax_mutate = mutate_avg.idxmax()
        some_proba[argmax_mutate] = some_proba[argmax_mutate] + self.K / self.iters
        new_proba = some_proba - self.K / (z * self.iters)
        new_proba = new_proba.clip(self.threshold, 1)
        return new_proba / new_proba.sum()

    def __update_thefittest(self, population, population_net, fitness):
        temp_best_id = np.argmax(fitness)
        temp_best_fitness = fitness[temp_best_id]
        if temp_best_fitness > self.thefittest["fitness"]:
            self.fittest_history.append(
                [
                    str(copy.deepcopy(population[temp_best_id])),
                    copy.copy(temp_best_fitness),
                ]
            )
            self.thefittest["fitness"] = temp_best_fitness
            self.thefittest["individ"] = copy.deepcopy(population[temp_best_id])
            self.thefittest["net"] = copy.deepcopy(population_net[temp_best_id])

    def mean_parents(self, parents_fit):
        return np.mean(parents_fit)

    def max_parents(self, parents_fit):
        return np.max(parents_fit)

    def min_parents(self, parents_fit):
        return np.min(parents_fit)

    def rand_parents(self, parents_fit):
        if np.random.random() < 0.5:
            return parents_fit[0]
        else:
            return parents_fit[1]

    def resource_culc(self, some_impact, mean, std):
        delta = mean - 5
        standart = (some_impact - np.mean(some_impact)) / np.std(some_impact)
        return np.clip(((standart * std)).astype(int) + mean, 5, mean + delta)

    def resource_culc_g(self, some_impact, group):
        result = np.zeros(len(some_impact))
        new_group = some_impact.copy()

        h_range = np.ceil((len(some_impact)) / len(group))
        for i, g in enumerate(group):
            result[int(i * h_range) : int((i + 1) * h_range)] = g

        argsort = np.argsort(some_impact)
        new_group[argsort] = result

        return new_group.astype(int)

    @staticmethod
    def get_m_n(all_, a):
        return int(all_ * (1 + a)), int(all_ * (1 / (1 + a)))

    def init_plot(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training Progress")
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=train_main_tab)
        self.plot_canvas_widget = self.plot_canvas.get_tk_widget()
        self.plot_canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def fit(self, X, y):
        nn_fit_progress_var.set(0)
        self._iteration = 0

        runs_must = 0
        resource_min = self.min_res
        resource_max = self.max_res
        resource_h = (resource_max - resource_min) / (self.iters - 1)
        resource = np.full(self.pop_size, resource_min)

        classes = len(set(y))
        eye = np.eye(len(set(y)))
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        self.defining_variables(X.shape[1], self.ngram, False)

        proba_history = pd.DataFrame(
            np.empty((self.pop_size, 5)),
            columns=["mutation", "crossing", "selection", "resource", "fitness"],
            dtype=object,
        )

        z_m = len(self.m_sets)
        z_c = len(self.c_sets)
        z_s = len(self.s_sets)
        z_r = len(self.r_sets)

        m_proba = pd.Series(np.full(z_m, 1 / z_m), index=self.m_sets.keys())
        c_proba = pd.Series(np.full(z_c, 1 / z_c), index=self.c_sets.keys())
        s_proba = pd.Series(np.full(z_s, 1 / z_s), index=self.s_sets.keys())
        r_proba = pd.Series(np.full(z_r, 1 / z_r), index=self.r_sets.keys())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.5
        )

        population = np.array(
            [self.generate_tree() for _ in range(self.pop_size)], dtype=object
        )
        population_temp = copy.deepcopy(population)
        population_nets = np.array([self.compile_(ind, classes) for ind in population])

        fitness = np.zeros(len(population))
        for i, ind, net in zip(range(self.pop_size), population[:], population_nets[:]):

            all_ = np.sqrt(resource[i])
            m, n = self.get_m_n(all_, 0)

            fitness[i], population_nets[i] = self.train_and_test(
                ind, net, X_train, y_train, X_test, eye[y_test], n_iters=m, n_size=n
            )
            self.runs += m * n
            runs_must += round(resource[i])

            progress_value = int(
                ((self._iteration) / ((self.iters - 1) * self.pop_size)) * 100
            )

            self._iteration = self._iteration + 1

            nn_fit_progress_var.set(progress_value)
            root.update_idletasks()  # Обновление интерфейса

        self.thefittest["individ"] = copy.deepcopy(population[np.argmax(fitness)])
        self.thefittest["fitness"] = fitness[np.argmax(fitness)].copy()
        self.thefittest["net"] = copy.deepcopy(population_nets[np.argmax(fitness)])
        self.fittest_history.append(
            [
                str(copy.deepcopy(population[np.argmax(fitness)])),
                fitness[np.argmax(fitness)].copy(),
            ]
        )
        self.__update_statistic(fitness, m_proba, c_proba, s_proba, r_proba)
        for i in range(1, self.iters):
            for type_, list_, proba in zip(
                self.operators_list,
                [m_proba.index, c_proba.index, s_proba.index, r_proba.index],
                [m_proba, c_proba, s_proba, r_proba],
            ):
                proba_history[type_] = np.random.choice(
                    list_, self.pop_size, p=proba.values
                )

            for j, m_o, c_o, s_o, r_o in zip(
                range(self.pop_size),
                proba_history["mutation"],
                proba_history["crossing"],
                proba_history["selection"],
                proba_history["resource"],
            ):

                parent_1, fitness_1 = self.s_sets[s_o](population, fitness)
                parent_2, fitness_2 = self.s_sets[s_o](population, fitness)

                offspring = self.c_sets[c_o](parent_1, parent_2)
                offspring = self.m_sets[m_o](offspring)

                population_temp[j] = copy.deepcopy(offspring)

            resource = resource + resource_h
            population_temp[-1] = copy.deepcopy(self.thefittest["individ"])

            population = copy.deepcopy(population_temp)
            population_nets = np.array(
                [self.compile_(ind, classes) for ind in population]
            )

            for j, ind, net in zip(
                range(self.pop_size - 1), population[:-1], population_nets[:-1]
            ):
                all_ = np.sqrt(resource[j])
                m, n = self.get_m_n(all_, 0)
                fitness[j], population_nets[j] = self.train_and_test(
                    ind, net, X_train, y_train, X_test, eye[y_test], n_iters=m, n_size=n
                )
                self.runs += m * n
                runs_must += round(resource[j])

                progress_value = int(
                    ((self._iteration) / ((self.iters - 1) * self.pop_size)) * 100
                )
                self._iteration = self._iteration + 1

                nn_fit_progress_var.set(progress_value)
                root.update_idletasks()  # Обновление интерфейса

            fitness[-1] = self.thefittest["fitness"].copy()
            population_nets[-1] = copy.deepcopy(self.thefittest["net"])
            population[-1] = copy.deepcopy(self.thefittest["individ"])

            proba_history["fitness"] = fitness

            m_proba = self.__update_proba(m_proba, "mutation", proba_history, z_m)
            c_proba = self.__update_proba(c_proba, "crossing", proba_history, z_c)
            s_proba = self.__update_proba(s_proba, "selection", proba_history, z_s)
            r_proba = self.__update_proba(r_proba, "resource", proba_history, z_r)

            self.__update_thefittest(population, population_nets, fitness)
            self.__update_statistic(fitness, m_proba, c_proba, s_proba, r_proba)

        return self

    def plot_graph(self, x, y):
        self.ax.clear()
        self.ax.plot(x, y)
        self.ax.set_title("График")

        # Обновление холста
        self.canvas.draw()


class SplitFunction:
    def __init__(self, var_index):
        self.var_index = var_index
        self.__name__ = "%" + str(var_index)


class Rulebase:

    def __init__(self, rules, num_terms, markers):
        self.rules = np.array(list(rules.values()), dtype=object)
        self.num_terms = np.array(num_terms)
        self.markers = markers
        self.test_borders = np.zeros(3 * np.sum(num_terms) - 3 * len(num_terms))
        self.terms_borders = None
        self.mask = None

        self.X_min = None
        self.X_max = None
        self.n_t = None
        self.center = None
        self.right = None
        self.right_temp = None
        self.centers = None

    def test(self, x_min, x_max, n_t, n):
        h = (x_max - x_min) / (n + 1)

        points = np.linspace(x_min + h, x_min + h * n_t, n_t)
        if len(points):
            points[0] = x_min
            return points
        else:
            return []

    def init_borders(self, some_X):
        r_min = []
        for i, rule in enumerate(self.rules):
            prereq, concl = self.rules[i]
            min_ = np.array(list(map(len, prereq)))
            r_min.append(min_)
        r_min = np.array(r_min).min(axis=0)

        self.mask = r_min != self.num_terms
        self.num_terms[~self.mask] = 0

        self.X_min = some_X.min(axis=0)
        self.X_max = some_X.max(axis=0)
        self.n_t = np.ceil(self.num_terms / 2).astype(int)

        self.left = [np.full(n_t_i, self.X_min[i]) for i, n_t_i in enumerate(self.n_t)]

        self.center = self.X_min + (self.X_max - self.X_min) / 2

        right_temp = [
            self.test(self.X_min[i], self.X_max[i], self.n_t[i], self.num_terms[i])
            for i in range(len(self.n_t))
        ]

        self.right = copy.deepcopy(right_temp)
        self.right_temp = np.hstack(right_temp)

    @staticmethod
    def trimf(x, abc):
        x_res = np.zeros((len(x), len(abc)))
        b_a = abc[:, 1] - abc[:, 0]
        c_b = abc[:, 2] - abc[:, 1]
        b_a[b_a == 0] = 1
        c_b[c_b == 0] = 1
        left_cond = (
            (x >= abc[:, 0][:, np.newaxis]) & (x <= abc[:, 1][:, np.newaxis])
        ).T
        right_cond = (
            (x >= abc[:, 1][:, np.newaxis]) & (x <= abc[:, 2][:, np.newaxis])
        ).T
        x_res[left_cond] = ((x[:, np.newaxis] - abc[:, 0]) / b_a)[left_cond]
        x_res[right_cond] = ((abc[:, 2] - x[:, np.newaxis]) / c_b)[right_cond]

        return x_res

    def culc_rule(self, some_X, index, terms_borders):
        prereq, concl = self.rules[index]
        n = len(terms_borders[0])
        acc = []
        k = 0
        for i, or_chain in enumerate(prereq):
            if self.mask[i]:
                slice_terms = np.hstack(terms_borders[k])[or_chain].reshape(-1, 3)

                k += 1
                mfx = self.trimf(some_X[:, i], slice_terms)
                mfx = np.array(np.split(mfx, int(mfx.shape[1] / n), axis=1))
                acc.append(np.max(mfx, axis=0))

        return np.min(acc, axis=0)

    def predict(self, some_X, terms_borders):
        mf = []
        for i, rule in enumerate(self.rules):
            mf.append(self.culc_rule(some_X, i, terms_borders))

        argmax = np.argmax(mf, axis=0).T

        def func(x):
            return self.rules[x][:, 1].astype(int)

        return np.array(list(map(func, argmax)))

    def culc_terms2(self, some_X, ind_float, centers, n, n_t, center):
        termss = []
        f_vars = n_t
        cut_index = np.add.accumulate(f_vars)

        cutted = ind_float
        term_inf = []
        for i in range(some_X.shape[1]):
            cutted = np.split(cutted, cut_index, axis=1)
            term_inf.append(cutted[:-1])
            cutted = cutted[-1]

        term_inf = np.split(ind_float, cut_index, axis=1)

        centers = np.split(centers, cut_index)

        for i, X_i, term_inf, centers_i in zip(
            range(some_X.shape[1]), some_X.T, term_inf, centers
        ):

            if n_t[i] > 0:
                terms = np.zeros((len(ind_float), n[i], 3))

                # print(center[i])
                centers = np.hstack(
                    [centers_i, (2 * center[i] - centers_i[: n[i] - n_t[i]])[::-1]]
                )

                terms[:, :, 1] = centers
                terms[:, :, 0][:, 1 : 1 + term_inf[:, 1:].shape[1]] = term_inf[:, 1:]

                terms[:, :, 0][:, 0] = terms[:, :, 1][:, 0]
                terms[:, :, -1][:, -1] = terms[:, :, -2][:, -1]

                terms[:, :, 2][:, 0] = term_inf[:, 0]
                terms[:, :, 0][:, -1] = self.X_max[i] - (term_inf[:, 0] - self.X_min[i])

                len_ = np.ceil((terms.shape[1] - 2) / 2).astype(int)

                terms[:, :, 2][:, -len_ - 1 : -1] = (
                    self.X_max[i]
                    - (terms[:, :, 0][:, 1 : len_ + 1] - self.X_min[i])[:, ::-1]
                )

                terms[:, :, 0][:, -len_ - 1 : -1] = (
                    2 * terms[:, :, 1][:, -len_ - 1 : -1]
                    - terms[:, :, 2][:, -len_ - 1 : -1]
                )

                terms[:, :, 2][:, 1 : len_ + 1] = (
                    2 * terms[:, :, 1][:, 1 : len_ + 1]
                    - terms[:, :, 0][:, 1 : len_ + 1]
                )

                termss.append(terms)
        return termss

    def culc_terms(self, some_X, ind_float):
        f_vars = self.num_terms[self.mask] - 1
        cut_index = np.add.accumulate(f_vars)
        cutted = ind_float
        term_inf = []
        for i in range(3):
            cutted = np.split(cutted, cut_index, axis=1)
            term_inf.append(cutted[:-1])
            cutted = cutted[-1]
        termss = []
        X_range = np.arange(some_X.shape[1], dtype=int)[self.mask]
        for i, h_points, l_points, r_points in zip(X_range, *term_inf):
            terms = np.zeros((h_points.shape[0], h_points.shape[1] + 1, 3))
            X_i_min = some_X[:, i].min()
            X_i_max = some_X[:, i].max()
            dX = X_i_max - X_i_min

            norm = h_points / np.sum(h_points, axis=1)[:, np.newaxis]
            norm_scaled = norm * dX

            hights = np.hstack(
                [
                    np.full((len(norm), 1), X_i_min),
                    X_i_min + np.add.accumulate(norm, axis=1) * dX,
                ]
            )

            # вычисления повторяются norm_scaled*l_points
            left = hights[:, :-1] + norm_scaled * l_points

            right_raw_1 = hights[:, :-1] + norm_scaled * l_points  # это тот же left
            # тот же left
            right_raw_2 = hights[:, 1:] - (hights[:, :-1] + norm_scaled * l_points)
            right = right_raw_1 + right_raw_2 * r_points

            terms[:, :, 1] = hights
            terms[:, :, 0][:, 1:] = left
            terms[:, :, 2][:, :-1] = right

            terms[:, :, 0][:, 0] = terms[:, :, 1][:, 0]
            terms[:, :, -1][:, -1] = terms[:, :, -2][:, -1]

            termss.append(terms)

        return termss

    def get_fitness(self, pop_terms_borders, x_true, y_true):

        pop_terms_borders = self.culc_terms2(
            x_true,
            pop_terms_borders,
            self.right_temp,
            self.num_terms,
            self.n_t,
            self.center,
        )

        predicts = self.predict(x_true, pop_terms_borders)

        def func(x):
            return f1_score(y_true, x, average="macro")

        f1 = np.array(list(map(func, predicts)))

        return f1

    def SelfCGAfit(self, some_X, some_y, iters, pop_size, tour_size, n_bit):
        right = copy.deepcopy(self.right)

        for i in range(len(right)):
            if len(right[i]):
                right[i][0] = self.X_max[i]

        left = np.hstack(self.left)
        right = np.hstack(right)

        vars_ = len(left)
        parts = n_bit

        # print(left, right)
        # return 1
        borders = {"left": left, "right": right}
        parts = np.full(vars_, parts)

        grid_model = SamplingGrid(borders, parts)

        def function(x):
            return self.get_fitness(grid_model.transform(x), some_X, some_y)

        model_opt = SelfCGA(
            function, iters, pop_size, np.sum(parts), tour_size=tour_size
        )

        model_opt.fit()

        thefittest = model_opt.thefittest
        float_borders = grid_model.transform(thefittest["individ"].reshape(1, -1))

        self.terms_borders = self.culc_terms2(
            some_X,
            float_borders,
            self.right_temp,
            self.num_terms,
            self.n_t,
            self.center,
        )

        return model_opt

    def show_rules(self, var_names=[], term_names=[], class_names=[], keep_any=False):
        text_rulebase = ""
        len_rule = np.zeros(len(self.rules))
        vars_used = []

        for id_, rule in enumerate(self.rules):
            k = 0
            text_rule = "if "
            prereq, concl = rule
            if len(class_names):
                concl = class_names[concl]

            for i, or_chain in enumerate(prereq):
                if self.mask[i]:
                    last_any = False
                    var = i
                    if len(var_names):
                        var = var_names[var]
                    or_chain_text = "(" + str(var) + " is ("
                    if len(or_chain) == 1:
                        bracket = ""
                        or_chain_text = or_chain_text[:-1]
                    if len(or_chain) == len(self.terms_borders[k][0]):
                        k += 1
                        if keep_any:
                            or_chain_text = or_chain_text[:-1]
                            bracket = ""
                            or_chain_text += " Any----"
                        else:
                            last_any = True
                            or_chain_text = ""

                    else:
                        bracket = ")"
                        for cond in or_chain:
                            cond_i = cond
                            if len(term_names):
                                cond_i = term_names[i][cond_i]
                            or_chain_text += str(cond_i) + " or "

                    if not last_any:
                        vars_used.append(var)
                        len_rule[id_] += 1
                        or_chain_text = or_chain_text[:-4] + ")" + bracket + " and "
                else:
                    or_chain_text = ""
                text_rule += or_chain_text
            text_rule = text_rule[:-4] + "then " + str(concl)

            text_rulebase += text_rule + "\n"

        return text_rulebase, len_rule, vars_used


class SelfGPFLClassifier:

    def um_low(self, x):
        return self.uniform_mutation(x, 0.25)

    def um_mean(self, x):
        return self.uniform_mutation(x, 1)

    def um_strong(self, x):
        return self.uniform_mutation(x, 4)

    def pm_low(self, x):
        return self.one_point_mutation(x, 0.25)

    def pm_mean(self, x):
        return self.one_point_mutation(x, 1)

    def pm_strong(self, x):
        return self.one_point_mutation(x, 4)

    def __init__(
        self,
        iters,
        pop_size,
        tour_size=5,
        max_height=5,
        K=0.5,
        threshold=0.1,
        min_res=1000,
        max_res=1000,
    ):

        self.K = K
        self.threshold = threshold

        self.iters = iters
        self.pop_size = pop_size
        self.tour_size = tour_size
        self.max_height = max_height
        self.parts = None
        self.final_rulebase = None
        self.min_res = min_res
        self.max_res = max_res

        self.thefittest = {"individ": None, "fitness": None, "net": None}
        self.pset = None

        self.operators_list = ["mutation", "crossing", "selection"]

        self.m_sets = {
            "uniform_low": self.um_low,
            "uniform_mean": self.um_mean,
            "uniform_strong": self.um_strong,
            "point_low": self.pm_low,
            "point_mean": self.pm_mean,
            "point_strong": self.pm_strong,
        }

        self.s_sets = {
            "tournament": self.tournament_selection,
            "rank": self.rank_selection,
            "proportional": self.proportional_selection,
        }

        self.c_sets = {
            "standart": self.standart_crossing,
            "one_point": self.one_point_crossing,
            "empty": self.empty_crossing,
        }

        self.stats = {
            "fitness": pd.DataFrame(columns=["max", "median", "min", "std"]),
            "proba": {
                "mutation": pd.DataFrame(columns=self.m_sets.keys()),
                "crossing": pd.DataFrame(columns=self.c_sets.keys()),
                "selection": pd.DataFrame(columns=self.s_sets.keys()),
            },
        }

        self.arr_pop_size = np.arange(pop_size, dtype=int)

        self.runs = 0

        self.fittest_history = []

    def init_sets(self, num_vars, num_outs, parts):
        if num_vars != len(parts):
            raise Exception(
                "the size of num_vars is not"
                + "equal to the length of parts. {} != {}".format(num_vars, len(parts))
            )
        self.pset = gp.PrimitiveSet("MAIN", num_outs + 1)
        for i in range(num_vars):
            self.pset.addPrimitive(SplitFunction(i), 2)
        for i in range(num_outs):
            eval("self.pset.renameArguments(ARG" + str(i) + "='u" + str(i) + "')")

        eval("self.pset.renameArguments(ARG" + str(num_outs) + "='u" + str(-1) + "')")
        self.parts = parts

    def generate_tree(self):
        return gp.PrimitiveTree(gp.genHalfAndHalf(self.pset, 2, 5))

    @staticmethod
    def in_blacklist(s_marker, s_blacklist):
        for blacked in s_blacklist:
            if len(blacked) > len(s_marker):
                continue
            else:
                if s_marker[: len(blacked)] == blacked:
                    return True
        return False

    def compile_base(self, some_tree):
        stack = []
        current = ""
        n_arg = "0"
        markers = np.array([])

        rule_temp = [[i for i in range(self.parts[j])] for j in range(len(self.parts))]

        rules_stack = []
        rules_current = deepcopy(rule_temp)
        rules_markers = {}
        rules = {}
        j = 0
        blacklist = []

        for k, node in enumerate(some_tree):
            current += n_arg
            markers = np.append(markers, current)
            if node.arity == 0:
                if len(stack) > 0:
                    n_arg = "1"
                    current = stack.pop()
            else:
                stack.append(current)
                n_arg = "0"

                index = int(node.name[1:])
                len_ = len(rules_current[index])

                if len_ == 1:
                    blacklist.append(current + "1")
                    rules_markers[markers[-1]] = deepcopy(rules_current)
                    continue

            if not self.in_blacklist(markers[-1], blacklist):
                rules_markers[markers[-1]] = deepcopy(rules_current)
                if node.arity == 0:
                    if int(node.value[1:]) != -1:
                        rules[markers[-1]] = (
                            deepcopy(rules_current),
                            int(node.value[1:]),
                        )
                    j += 1
                    if len(rules_stack) > 0:
                        rules_current = rules_stack.pop()
                else:
                    if len_ % 2 == 0:
                        left_slice = slice(0, int(len_ / 2))
                        right_slice = slice(int(len_ / 2), len_)
                    else:
                        left_slice = slice(0, int(len_ / 2))
                        right_slice = slice(int(len_ / 2), len_)

                    rules_stack.append(deepcopy(rules_current))

                    rules_stack[-1][index] = deepcopy(
                        rules_stack[-1][index][right_slice]
                    )

                    rules_current[index] = deepcopy(rules_current[index][left_slice])
            else:
                rules_markers[markers[-1]] = []

        return Rulebase(rules, self.parts, rules_markers)

    @staticmethod
    def mark_tree(tree):
        stack = []
        current = ""
        n_arg = "0"
        markers = np.array([])
        for k, node in enumerate(tree):
            current += n_arg
            markers = np.append(markers, current)
            if node.arity == 0:
                if len(stack) > 0:
                    n_arg = "1"
                    current = stack.pop()
            elif node.arity == 1:
                n_arg = "0"
            else:
                stack.append(current)
                n_arg = "0"
        return markers

    def expr_mut(self, pset, type_, len_):
        return gp.genGrow(pset, 0, len_, type_)

    @staticmethod
    def replace_node(node, pset):
        def filter_(x):
            return x != node

        if node.arity == 0:  # Terminal
            pool = list(filter(filter_, pset.terminals[node.ret]))
            term = random.choice(pool)
            if isclass(term):
                term = term()
            return term
        else:  # Primitive
            pool = list(filter(filter_, pset.primitives[node.ret]))
            prims = [p for p in pool if p.args == node.args]
            return random.choice(prims)

    def one_point_mutation(self, some_net, proba):
        some_net = copy.deepcopy(some_net)
        proba = proba / len(some_net)
        for i, node in enumerate(some_net):
            if np.random.random() < proba:

                some_net[i] = self.replace_node(node, self.pset)

        return some_net

    def uniform_mutation(self, some_net, proba):
        some_net = copy.deepcopy(some_net)
        proba = proba / len(some_net)
        for i, node in enumerate(some_net[1:]):
            i = i + 1
            if np.random.random() < proba:
                slice_ = some_net.searchSubtree(i)
                type_ = node.ret
                some_net[slice_] = self.expr_mut(
                    pset=self.pset, type_=type_, len_=len(some_net[slice_])
                )
                break

        return some_net

    def rank_selection(self, population, fitness):
        ranks = rankdata(fitness)
        probability = ranks / np.sum(ranks)
        ind = np.random.choice(self.arr_pop_size, size=1, p=probability)
        offspring = population[ind][0]
        return copy.deepcopy(offspring), fitness[ind][0]

    def tournament_selection(self, population, fitness):
        tournament = np.random.choice(
            self.arr_pop_size, size=self.tour_size, replace=False
        )
        max_fit_id = np.argmax(fitness[tournament])
        return (
            copy.deepcopy(population[tournament[max_fit_id]]),
            fitness[tournament[max_fit_id]],
        )

    def proportional_selection(self, population, fitness):
        max_ = fitness.max()
        min_ = fitness.min()
        if max_ == min_:
            fitness_n = np.ones(fitness.shape)
        else:
            fitness_n = (fitness - min_) / (max_ - min_)

        probability = fitness_n / fitness_n.sum()
        ind = np.random.choice(self.arr_pop_size, size=1, p=probability)
        offspring = population[ind][0]

        return copy.deepcopy(offspring), fitness[ind][0]

    @staticmethod
    def standart_crossing(ind_1, ind_2):
        offs_1, offs_2 = gp.cxOnePoint(copy.deepcopy(ind_1), copy.deepcopy(ind_2))
        if np.random.random() > 0.5:
            return offs_1
        else:
            return offs_2

    def one_point_crossing(self, ind1, ind2):
        ind1 = copy.deepcopy(ind1)
        ind2 = copy.deepcopy(ind2)
        if len(ind1) < 2 or len(ind2) < 2:
            if np.random.random() > 0.5:
                return ind1
            else:
                return ind2
        mark_1 = self.mark_tree(ind1)
        mark_2 = self.mark_tree(ind2)
        common, c_1, c_2 = np.intersect1d(mark_1, mark_2, return_indices=True)

        index = random.choice(range(1, len(c_1)))
        index1 = c_1[index]
        index2 = c_2[index]

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

        if np.random.random() > 0.5:
            return ind1
        else:
            return ind2

    def empty_crossing(self, ind1, ind2):
        if np.random.random() > 0.5:
            return ind1
        else:
            return ind2

    def train_and_test(
        self, ind, rbase, X_train, y_train, X_test, y_test, n_iters, n_size
    ):
        if len(rbase.rules) < 2:
            return 0, rbase
        if not np.all(rbase.terms_borders):
            rbase.init_borders(X_train)
        height = ind.height
        if height > self.max_height:
            fine_h = height
        else:
            fine_h = 0.0

        res = rbase.SelfCGAfit(X_train, y_train, n_iters, n_size, 5, 12)
        len_ = len(rbase.rules)

        predict = rbase.predict(X_test, rbase.terms_borders)[0]
        fitnes_value = (
            f1_score(y_test, predict, average="macro") - 0.01 * fine_h - 0.01 * len_
        )
        (n_iters, n_size, fitnes_value, res.thefittest["fitness"], height)

        return fitnes_value, rbase

    def __update_statistic(self, fitness, m_proba, c_proba, s_proba):
        self.stats["fitness"] = self.stats["fitness"].append(
            {
                "max": fitness.max(),
                "min": fitness.min(),
                "median": np.median(fitness),
                "std": fitness.std(),
            },
            ignore_index=True,
        )

        for proba, oper in zip([m_proba, c_proba, s_proba], self.operators_list):
            self.stats["proba"][oper] = self.stats["proba"][oper].append(
                proba.copy(), ignore_index=True
            )

    def __update_proba(self, some_proba, operator, some_history, z):
        mutate_avg = some_history.groupby(operator).mean()["fitness"]
        argmax_mutate = mutate_avg.idxmax()
        some_proba[argmax_mutate] = some_proba[argmax_mutate] + self.K / self.iters
        new_proba = some_proba - self.K / (z * self.iters)
        new_proba = new_proba.clip(self.threshold, 1)
        return new_proba / new_proba.sum()

    def __update_thefittest(self, population, population_net, fitness):
        temp_best_id = np.argmax(fitness)
        temp_best_fitness = fitness[temp_best_id]
        if temp_best_fitness > self.thefittest["fitness"]:
            self.fittest_history.append(
                [
                    str(copy.deepcopy(population[temp_best_id])),
                    copy.copy(temp_best_fitness),
                ]
            )
            self.thefittest["fitness"] = temp_best_fitness
            self.thefittest["individ"] = copy.deepcopy(population[temp_best_id])
            self.thefittest["net"] = copy.deepcopy(population_net[temp_best_id])

    @staticmethod
    def get_m_n(all_, a):
        return int(all_ * (1 + a)), int(all_ * (1 / (1 + a)))

    def fit(self, some_X, some_y, some_X_test, some_y_test):
        fc_fit_progress_var.set(0)
        self._iteration = 0

        runs_must = 0
        resource_min = self.min_res
        resource_max = self.max_res
        resource_h = (resource_max - resource_min) / (self.iters - 1)
        resource = np.full(self.pop_size, resource_min)

        proba_history = pd.DataFrame(
            np.empty((self.pop_size, 4)),
            columns=["mutation", "crossing", "selection", "fitness"],
            dtype=object,
        )

        z_m = len(self.m_sets)
        z_c = len(self.c_sets)
        z_s = len(self.s_sets)

        m_proba = pd.Series(np.full(z_m, 1 / z_m), index=self.m_sets.keys())
        c_proba = pd.Series(np.full(z_c, 0.9 / (z_c - 1)), index=self.c_sets.keys())
        c_proba["empty"] = 0.1
        s_proba = pd.Series(np.full(z_s, 1 / z_s), index=self.s_sets.keys())

        X_train, X_test, y_train, y_test = train_test_split(
            some_X, some_y, stratify=some_y, test_size=0.6
        )

        population = np.array(
            [self.generate_tree() for _ in range(self.pop_size)], dtype=object
        )
        population_temp = copy.deepcopy(population)
        population_rules = np.array([self.compile_base(ind) for ind in population])

        fitness = np.zeros(len(population))
        for i, ind, rbase in zip(
            range(self.pop_size), population[:], population_rules[:]
        ):

            all_ = np.sqrt(resource[i])
            m, n = self.get_m_n(all_, 0)
            fitness[i], population_rules[i] = self.train_and_test(
                ind, rbase, X_train, y_train, X_test, y_test, int(n), int(m)
            )
            self.runs += m * n
            runs_must += round(resource[i])

            progress_value = int(
                ((self._iteration) / ((self.iters - 1) * self.pop_size)) * 100
            )
            self._iteration = self._iteration + 1

            fc_fit_progress_var.set(progress_value)
            root.update_idletasks()  # Обновление интерфейса

        self.thefittest["individ"] = copy.deepcopy(population[np.argmax(fitness)])
        self.thefittest["fitness"] = fitness[np.argmax(fitness)].copy()
        self.thefittest["net"] = copy.deepcopy(population_rules[np.argmax(fitness)])
        self.fittest_history.append(
            [
                str(copy.deepcopy(population[np.argmax(fitness)])),
                fitness[np.argmax(fitness)].copy(),
            ]
        )
        self.__update_statistic(fitness, m_proba, c_proba, s_proba)
        for i in range(1, self.iters):

            predict = self.thefittest["net"].predict(
                some_X_test, self.thefittest["net"].terms_borders
            )[0]
            fitnes_value = f1_score(some_y_test, predict, average="macro")
            for type_, list_, proba in zip(
                self.operators_list,
                [m_proba.index, c_proba.index, s_proba.index],
                [m_proba, c_proba, s_proba],
            ):
                proba_history[type_] = np.random.choice(
                    list_, self.pop_size, p=proba.values
                )
            for (
                j,
                m_o,
                c_o,
                s_o,
            ) in zip(
                range(self.pop_size),
                proba_history["mutation"],
                proba_history["crossing"],
                proba_history["selection"],
            ):

                parent_1, fitness_1 = self.s_sets[s_o](population, fitness)
                parent_2, fitness_2 = self.s_sets[s_o](population, fitness)

                offspring = self.c_sets[c_o](parent_1, parent_2)
                offspring = self.m_sets[m_o](offspring)

                population_temp[j] = copy.deepcopy(offspring)

            resource = resource + resource_h
            # print(resource.mean(), "resource.mean()")
            population_temp[-1] = copy.deepcopy(self.thefittest["individ"])

            population = copy.deepcopy(population_temp)
            population_rules = np.array([self.compile_base(ind) for ind in population])
            for j, ind, rbase in zip(
                range(self.pop_size), population[:], population_rules[:]
            ):
                all_ = np.sqrt(resource[j])
                m, n = self.get_m_n(all_, 0)
                fitness[j], population_rules[j] = self.train_and_test(
                    ind, rbase, X_train, y_train, X_test, y_test, int(n), int(m)
                )
                self.runs += m * n
                runs_must += round(resource[j])

                progress_value = int(
                    ((self._iteration) / ((self.iters - 1) * self.pop_size)) * 100
                )
                self._iteration = self._iteration + 1

                fc_fit_progress_var.set(progress_value)
                root.update_idletasks()  # Обновление интерфейса

            proba_history["fitness"] = fitness

            m_proba = self.__update_proba(m_proba, "mutation", proba_history, z_m)
            c_proba = self.__update_proba(c_proba, "crossing", proba_history, z_c)
            s_proba = self.__update_proba(s_proba, "selection", proba_history, z_s)

            self.__update_thefittest(population, population_rules, fitness)
            self.__update_statistic(fitness, m_proba, c_proba, s_proba)

            fitness[-1] = self.thefittest["fitness"]
            population_rules[-1] = self.thefittest["net"]

        return self

    def predict(self, some_X):
        rbase = self.thefittest["net"]
        pred = rbase.predict(some_X, rbase.terms_borders)[0]
        return pred


def print_net(net, show_edge=False, ax=None, in_dict=None):
    net = copy.deepcopy(net)

    if type(in_dict) != type(None):
        print("работает")
        bias = list(net.inputs)[-1]
        for i, value in enumerate(in_dict.values()):

            cond = net.connects[:, 0] == np.array(list(value))[:, np.newaxis]
            cond = np.any(cond, axis=0)
            net.connects[:, 0][cond] = i
            # print(i, value, len(cond), np.arange(len(cond)))

            # cond[np.min(np.arange(len(cond))[cond])] = False
            # ind = np.arange(len(cond))[cond]
            # net.weights = np.delete(net.weights, ind)

        dup = pd.DataFrame(net.connects).duplicated().values
        dup_all = pd.DataFrame(net.connects).duplicated(keep=False).values
        net.weights[dup_all] = 0
        net.weights = np.delete(net.weights, dup)
        net.connects = np.delete(net.connects, dup, axis=0)
        net.inputs = set(range(len(in_dict)))
        net.inputs.add(bias)
    # return net
    G = nx.DiGraph()

    weights = net.weights.copy()
    if len(weights) > 0:
        weights = (weights - min(weights)) / (max(weights) - min(weights))

    len_i = len(net.inputs)
    len_h = len(net.assemble_hiddens())
    len_o = len(net.outputs)

    sum_ = len_i + len_h + len_o
    positions = np.zeros((sum_, 2), dtype=float)
    colors = np.zeros((sum_, 4))
    w_colors = np.zeros((len(weights), 4))

    G.add_nodes_from(net.inputs)  # входы
    positions[: len_i - 1][:, 1] = np.arange(len_i - 1) - (len_i - 1) / 2
    colors[:len_i] = np.array([0.11, 0.67, 0.47, 1])
    colors[len_i - 1] = np.array([0.11, 0.67, 0.47, 0.1])
    positions[len_i - 1][1] = np.max(positions[: len_i - 1][:, 1]) + 1

    n = len_i
    for i, layer in enumerate(net.hiddens):  # скрытые

        G.add_nodes_from(np.sort(list(layer)))

        positions[n : n + len(layer)][:, 0] = i + 1
        positions[n : n + len(layer)][:, 1] = np.arange(len(layer)) - len(layer) / 2
        colors[n : n + len(layer)] = np.array([0.0, 0.74, 0.99, 1])
        n += len(layer)

    G.add_nodes_from(net.outputs)  # выходы
    positions[n : n + len_o][:, 0] = len(net.hiddens) + 1
    positions[n : n + len_o][:, 1] = np.arange(len_o) - len_o / 2
    colors[n : n + len_o] = np.array([0.94, 0.50, 0.50, 1])

    G.add_edges_from(net.connects)  # связи
    w_colors[:, 0] = 1 - weights
    w_colors[:, 2] = weights
    w_colors[:, 3] = 0.8

    cond = np.array(G.edges)[:, 0] == list(net.inputs)[-1]
    w_colors[:, 3][cond] = 0.1

    positions = dict(zip(G.nodes, positions))

    if type(in_dict) != type(None):
        labels = {
            **dict(zip(net.inputs, list(in_dict.keys()))),
            **net.activs,
            **dict(zip(net.outputs, range(len_o))),
        }
    else:
        labels = {
            **dict(zip(net.inputs, net.inputs)),
            **net.activs,
            **dict(zip(net.outputs, range(len_o))),
        }

    edge_labels = dict(zip(list(G.edges), list(G.edges)))

    nx.draw_networkx_nodes(G, pos=positions, node_color=colors, edgecolors="black")

    nx.draw_networkx_edges(G, pos=positions, edge_color=w_colors, ax=ax)

    nx.draw_networkx_labels(G, positions, labels, ax=ax)
    if show_edge:
        nx.draw_networkx_edge_labels(
            G,
            positions,
            horizontalalignment="center",
            edge_labels=edge_labels,
            label_pos=0.8,
            bbox={"alpha": 0},
            ax=ax,
        )


def predict_net(net, x):
    X = np.hstack([x, np.ones((x.shape[0], 1))])
    proba = net.forward(X)[0]
    predict = np.argmax(proba, axis=1)
    return predict, proba


def cat_crossentropy2(target, output):
    output /= output.sum(axis=-1, keepdims=True)
    output = np.clip(output, 1e-7, 1 - 1e-7)
    return np.mean(np.sum(target * -np.log(output), axis=-1, keepdims=False)) / (
        -np.log(1e-7)
    )


def show_table(data_frame):
    top = tk.Toplevel(root)
    top.title("Таблица данных")

    table = Table(top, dataframe=data_frame, showtoolbar=True, showstatusbar=True)
    table.show()


def data_open_file():
    global data
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        data = pd.read_csv(file_path)
        show_table(data)
        nn_fit_start_button.config(state=tk.NORMAL)
        fc_fit_start_button.config(state=tk.NORMAL)


def nn_save_model():
    if "nn_model" in globals():
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl")],
            initialfile="nn_model.pkl",
        )
        if file_path:
            with open(file_path, "wb") as file:
                pickle.dump(nn_model, file)


def fc_save_model():

    if "fc_model" in globals():
        if trained_on_nn_data:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".pkl",
                filetypes=[("Pickle files", "*.pkl")],
                initialfile="fc_model(nn).pkl",
            )
        else:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".pkl",
                filetypes=[("Pickle files", "*.pkl")],
                initialfile="fc_model.pkl",
            )

        if file_path:
            with open(file_path, "wb") as file:
                pickle.dump(fc_model, file)


def nn_save_all_stats():
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        initialfile="gpnn_fit_stats.csv",
    )

    if file_path:
        # Сохраняем DataFrame в выбранный файл
        nn_all_stats.to_csv(file_path, index=False)
        print(f"DataFrame saved to {file_path}")


def fc_save_all_stats():
    if trained_on_nn_data:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="gpfc_fit_stats(nn).csv",
        )
    else:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="gpfc_fit_stats.csv",
        )

    if file_path:
        # Сохраняем DataFrame в выбранный файл
        fc_all_stats.to_csv(file_path, index=False)
        print(f"DataFrame saved to {file_path}")


def nn_start_fit():
    if "data" in globals():
        # Получение значений из окон ввода
        iters = int(nn_iters_entry.get())
        pop_size = int(nn_pop_size_entry.get())
        ngram = 1
        tour_size = int(nn_tour_size_entry.get())
        min_res = int(nn_resources_entry.get())
        max_res = int(nn_resources_entry.get())
        global nn_model

        nn_model = selfCGPNN(
            iters=iters,
            pop_size=pop_size,
            ngram=ngram,
            tour_size=tour_size,
            min_res=min_res,
            max_res=max_res,
        )

        def run_algorithm():
            global X_scaled_NN
            global columns_NN
            global y_NN
            global nn_all_stats

            nn_fit_start_button.config(state=tk.DISABLED)
            fc_fit_start_button.config(state=tk.DISABLED)

            X_ = data.loc[:, data.columns != "class"].values
            labels = data["class"].values

            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(labels)

            X_scaled = scale(X_)
            columns = list(data.columns)

            eye = np.eye(len(set(y)))
            nn_model.fit(X_scaled, y)

            net = nn_model.thefittest["net"]
            mask = list(net.inputs)
            mask.sort()
            X_scaled_NN = X_scaled[:, mask[:-1]]
            columns_NN = np.array(columns, dtype=object)[mask[:-1]]
            y_NN = net.predict(X_scaled)

            fc_use_nn_inputs_checkbox.config(state=tk.NORMAL)

            nn_fit_start_button.config(state=tk.NORMAL)
            fc_fit_start_button.config(state=tk.NORMAL)
            nn_save_button.config(state=tk.NORMAL)
            # Ваш код для дополнительных действий после выполнения кода

            stats = nn_model.stats
            nn_all_stats = pd.concat(
                [
                    stats["proba"]["mutation"],
                    stats["proba"]["crossing"],
                    stats["proba"]["selection"],
                    stats["fitness"],
                ],
                axis=1,
            )
            nn_save_stat_button.config(state=tk.NORMAL)

        # Запуск алгоритма в отдельном потоке
        algorithm_thread = Thread(target=run_algorithm)
        algorithm_thread.start()


def fc_start_fit():
    if "data" in globals():
        # Получение значений из окон ввода
        iters = int(fc_iters_entry.get())
        pop_size = int(fc_pop_size_entry.get())
        tour_size = int(fc_tour_size_entry.get())
        num_fuzzy_vars = int(fc_num_fuzzy_vars_entry.get())
        min_res = int(fc_resources_entry.get())
        max_res = int(fc_resources_entry.get())
        global fc_model

        # Обновление значения атрибута iters в объекте SelfGPFLClassifier
        fc_model = SelfGPFLClassifier(
            iters,
            pop_size,
            max_height=10,
            tour_size=tour_size,
            min_res=min_res,
            max_res=max_res,
        )

        def run_algorithm_fuzzy():
            global fc_all_stats
            global trained_on_nn_data

            if fc_use_nn_inputs_var.get() == 1:  # 1 - чекбокс включен, 0 - выключен
                nn_fit_start_button.config(state=tk.DISABLED)
                fc_fit_start_button.config(state=tk.DISABLED)

                labels = data["class"].values

                label_encoder = LabelEncoder()
                label_encoder.fit(labels)

                fc_model.init_sets(
                    X_scaled_NN.shape[1],
                    len(set(y_NN)),
                    X_scaled_NN.shape[1] * [num_fuzzy_vars],
                )
                fc_model.fit(X_scaled_NN, y_NN, X_scaled_NN, y_NN)

                rules = fc_model.thefittest["net"].show_rules(
                    var_names=columns_NN, class_names=label_encoder.classes_
                )[0]

                nn_fit_start_button.config(state=tk.NORMAL)
                fc_fit_start_button.config(state=tk.NORMAL)
                fc_save_button.config(state=tk.NORMAL)
                trained_on_nn_data = True
            else:
                nn_fit_start_button.config(state=tk.DISABLED)
                fc_fit_start_button.config(state=tk.DISABLED)

                X_ = data.loc[:, data.columns != "class"].values
                labels = data["class"].values

                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(labels)

                X_scaled = scale(X_)
                columns = list(data.columns)

                fc_model.init_sets(
                    X_scaled.shape[1], len(set(y)), X_scaled.shape[1] * [3]
                )
                fc_model.fit(X_scaled, y, X_scaled, y)

                rules = fc_model.thefittest["net"].show_rules(
                    var_names=columns, class_names=label_encoder.classes_
                )[0]

                nn_fit_start_button.config(state=tk.NORMAL)
                fc_fit_start_button.config(state=tk.NORMAL)
                fc_save_button.config(state=tk.NORMAL)
                trained_on_nn_data = False

            stats = fc_model.stats
            fc_all_stats = pd.concat(
                [
                    stats["proba"]["mutation"],
                    stats["proba"]["crossing"],
                    stats["proba"]["selection"],
                    stats["fitness"],
                ],
                axis=1,
            )
            fc_save_stat_button.config(state=tk.NORMAL)

            fc_result_rulebase_widget.config(state=tk.NORMAL)
            fc_result_rulebase_widget.delete(1.0, tk.END)
            fc_result_rulebase_widget.insert(tk.END, rules)
            fc_result_rulebase_widget.config(state=tk.DISABLED)

        # Запуск алгоритма в отдельном потоке
        algorithm_thread = Thread(target=run_algorithm_fuzzy)
        algorithm_thread.start()


# Создание основного окна
root = tk.Tk()
root.title("Программа для отображения данных")

# Создание вкладок
notebook = ttk.Notebook(root)

############################################ Вкладка "Обучение"
train_main_tab = ttk.Frame(notebook)
notebook.add(train_main_tab, text="Обучение")

# Кнопка выбора файла
data_button = tk.Button(
    train_main_tab, text="Выбрать файл", command=data_open_file, anchor=tk.W
)
data_button.pack(pady=10, anchor=tk.W)

# Вложенные вкладки
nested_notebook_train = ttk.Notebook(train_main_tab)

############################################ Вкладка "Нейронная сеть" внутри "Обучение"
nn_tab = ttk.Frame(nested_notebook_train)
nested_notebook_train.add(nn_tab, text="Нейронная сеть")

# Создаем фрейм для управления
frame = ttk.Frame(nn_tab)
frame.pack(pady=10, anchor=tk.W)

# Создаем фрейм для кнопок и окошек
controls_frame = ttk.Frame(frame)
controls_frame.pack(side="left")

# Окна ввода с значениями по умолчанию
nn_iters_label = tk.Label(controls_frame, text="Количество итераций:", anchor=tk.W)
nn_iters_entry = tk.Entry(controls_frame)
nn_iters_entry.insert(0, "10")  # Значение по умолчанию
nn_iters_label.pack(anchor=tk.W)
nn_iters_entry.pack(anchor=tk.W)

nn_pop_size_label = tk.Label(controls_frame, text="Размер популяции:", anchor=tk.W)
nn_pop_size_entry = tk.Entry(controls_frame)
nn_pop_size_entry.insert(0, "10")  # Значение по умолчанию
nn_pop_size_label.pack(anchor=tk.W)
nn_pop_size_entry.pack(anchor=tk.W)

nn_tour_size_label = tk.Label(controls_frame, text="Размер турнира:", anchor=tk.W)
nn_tour_size_entry = tk.Entry(controls_frame)
nn_tour_size_entry.insert(0, "3")  # Значение по умолчанию
nn_tour_size_label.pack(anchor=tk.W)
nn_tour_size_entry.pack(anchor=tk.W)

nn_resources_label = tk.Label(
    controls_frame,
    text="Ресурсы для обучения весов сетей (iters*pop_size):",
    anchor=tk.W,
)
nn_resources_entry = tk.Entry(controls_frame)
nn_resources_entry.insert(0, "2500")  # Значение по умолчанию
nn_resources_label.pack(anchor=tk.W)
nn_resources_entry.pack(anchor=tk.W)

# Кнопка "Старт"
nn_fit_start_button = tk.Button(
    controls_frame, text="Старт", command=nn_start_fit, state=tk.DISABLED, anchor=tk.W
)
nn_fit_start_button.pack(pady=10, anchor=tk.W)

# Фрейм для прогресс-бара и кнопок
nn_progress_and_safe_frame = tk.Frame(controls_frame)
nn_progress_and_safe_frame.pack(pady=10, anchor=tk.W)

# Подпись "Построение сети"
nn_fit_progress_label = tk.Label(nn_progress_and_safe_frame, text="Построение сети:")
nn_fit_progress_label.pack(side=tk.TOP, anchor=tk.W)

# Прогресс бар
nn_fit_progress_var = tk.DoubleVar()
nn_fit_progress_bar = ttk.Progressbar(
    nn_progress_and_safe_frame,
    variable=nn_fit_progress_var,
    length=200,
    mode="determinate",
)
nn_fit_progress_bar.pack(side=tk.LEFT, anchor=tk.W)

# Кнопка "Сохранить сеть"
nn_save_button = tk.Button(
    nn_progress_and_safe_frame,
    text="Сохранить сеть",
    command=nn_save_model,
    state=tk.DISABLED,
)
nn_save_button.pack(side=tk.RIGHT, anchor=tk.W)

# Кнопка для сохранения статистики
nn_save_stat_button = ttk.Button(
    controls_frame,
    text="Сохранить статистику обучения",
    command=nn_save_all_stats,
    state=tk.DISABLED,
)
nn_save_stat_button.pack(pady=10, anchor=tk.W)


fig, ax = Figure(), plt.axes()
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill="both", expand=True, side=tk.RIGHT)


############################################ Вкладка "Нечеткая система" внутри "Обучение"
fc_tab = ttk.Frame(nested_notebook_train)
nested_notebook_train.add(fc_tab, text="Нечеткая система")

# Окна ввода параметров
fc_iters_label = tk.Label(fc_tab, text="Количество итераций:", anchor=tk.W)
fc_iters_entry = tk.Entry(fc_tab)
fc_iters_entry.insert(0, "10")  # Значение по умолчанию
fc_iters_label.pack(anchor=tk.W)
fc_iters_entry.pack(anchor=tk.W)

fc_pop_size_label = tk.Label(fc_tab, text="Размер популяции:", anchor=tk.W)
fc_pop_size_entry = tk.Entry(fc_tab)
fc_pop_size_entry.insert(0, "10")  # Значение по умолчанию
fc_pop_size_label.pack(anchor=tk.W)
fc_pop_size_entry.pack(anchor=tk.W)

fc_tour_size_label = tk.Label(fc_tab, text="Размер турнира:", anchor=tk.W)
fc_tour_size_entry = tk.Entry(fc_tab)
fc_tour_size_entry.insert(0, "3")  # Значение по умолчанию
fc_tour_size_label.pack(anchor=tk.W)
fc_tour_size_entry.pack(anchor=tk.W)

fc_num_fuzzy_vars_label = tk.Label(
    fc_tab, text="Количество нечетких переменных:", anchor=tk.W
)
fc_num_fuzzy_vars_entry = tk.Entry(fc_tab)
fc_num_fuzzy_vars_entry.insert(0, "3")  # Значение по умолчанию
fc_num_fuzzy_vars_label.pack(anchor=tk.W)
fc_num_fuzzy_vars_entry.pack(anchor=tk.W)

fc_resources_label = tk.Label(
    fc_tab,
    text="Ресурсы для обучения линг. переменных (iters*pop_size):",
)
fc_resources_entry = tk.Entry(fc_tab)
fc_resources_entry.insert(0, "2500")  # Значение по умолчанию
fc_resources_label.pack(anchor=tk.W)
fc_resources_entry.pack(anchor=tk.W)

# Фрейм для прогресс-бара и кнопок
fc_progress_and_safe_frame = tk.Frame(fc_tab)
fc_progress_and_safe_frame.pack(pady=10, anchor=tk.W)

# Чекбокс для выбора использования входов и выходов нейронной сети
fc_use_nn_inputs_var = tk.IntVar()
use_nn_outputs_var = tk.IntVar()

fc_use_nn_inputs_checkbox = tk.Checkbutton(
    fc_progress_and_safe_frame,
    text="Использовать входы и выходы нейронной сети",
    variable=fc_use_nn_inputs_var,
    anchor=tk.W,
    state=tk.DISABLED,
)
fc_use_nn_inputs_checkbox.pack(side=tk.RIGHT, anchor=tk.W)

# Кнопка "Старт" обучения
fc_fit_start_button = tk.Button(
    fc_progress_and_safe_frame,
    text="Старт",
    command=fc_start_fit,
    state=tk.DISABLED,
    anchor=tk.W,
)
fc_fit_start_button.pack(side=tk.LEFT, anchor=tk.W)

# Фрейм для прогресс-бара и кнопок
fc_progress_and_safe_frame = tk.Frame(fc_tab)
fc_progress_and_safe_frame.pack(pady=10, anchor=tk.W)

# Подпись "Построение нечеткой системы"
build_label_fuzzy = tk.Label(
    fc_progress_and_safe_frame, text="Построение нечеткой системы:"
)
build_label_fuzzy.pack(side=tk.TOP, anchor=tk.W)

# Прогресс бар
fc_fit_progress_var = tk.DoubleVar()
fc_fit_progress_bar = ttk.Progressbar(
    fc_progress_and_safe_frame,
    variable=fc_fit_progress_var,
    length=200,
    mode="determinate",
)
fc_fit_progress_bar.pack(side=tk.LEFT, anchor=tk.W)

# Кнопка "Сохранить нечеткую систему"
fc_save_button = tk.Button(
    fc_progress_and_safe_frame,
    text="Сохранить нечеткую систему",
    command=fc_save_model,
    state=tk.DISABLED,
)
fc_save_button.pack(side=tk.RIGHT, anchor=tk.W)

# Добавление окна с текстом
fc_result_rulebase_widget = tk.Text(
    fc_tab, wrap=tk.WORD, height=10, width=40, state=tk.DISABLED
)
fc_result_rulebase_widget.pack(pady=10, fill=tk.BOTH, expand=True)

# Кнопка для сохранения статистики
fc_save_stat_button = ttk.Button(
    fc_tab,
    text="Сохранить статистику обучения",
    command=fc_save_all_stats,
    state=tk.DISABLED,
)
fc_save_stat_button.pack(pady=10, anchor=tk.W)

# Пакет вложенных вкладок
nested_notebook_train.pack(expand=1, fill="both")


# Вкладка "Предсказание"
predict_main_tab = ttk.Frame(notebook)
notebook.add(predict_main_tab, text="Предсказание")


# # Вложенные вкладки внутри "Предсказание"
# nested_notebook_predict = ttk.Notebook(predict_main_tab)

# # Вкладка "Нейронная сеть"
# nn_tab_predict = ttk.Frame(nested_notebook_predict)
# nested_notebook_predict.add(nn_tab_predict, text="Нейронная сеть")

# # Вкладка "Нечеткая система"
# fuzzy_ta_predict = ttk.Frame(nested_notebook_predict)
# nested_notebook_train.add(fuzzy_ta_predict, text="Нечеткая система")

# # Пакет вложенных вкладок
# nested_notebook_predict.pack(expand=1, fill="both")

# # Вложенные вкладки внутри "Обучение"
# nested_notebook_train = ttk.Notebook(train_main_tab)


# Пакет всех вкладок
notebook.pack(expand=1, fill="both")


# Запуск основного цикла
root.mainloop()

# dir_path = os.path.dirname(os.path.realpath(__file__))

# data_filename = input("data file path:")
# data = pd.read_csv(data_filename)

# X_ = data.iloc[:, :-1].values
# y = data.iloc[:, -1].values.astype(int)

# X_scaled = scale(X_)
# columns = list(data.columns)

# print(X_scaled.shape)
# print(y.shape)
# print(columns)

# ngram = int(input("ngram NN:")) # убрать параметр
# iters = int(input("iters NN:")) #
# pop_size = int(input("pop_size NN:")) #
# min_res = int(input("min_res NN:")) # в пределах от 2500
# max_res = int(input("max_res NN:")) # до бесконечности

# iters_FC = int(input("iters FC:"))
# pop_size_FC = int(input("pop_size FC:"))
# min_res_FC = int(input("min_res FC:")) # в пределах от 2500
# max_res_FC = int(input("max_res FC:")) # до бесконечности

# nn_model = selfCGPNN(
#     iters=iters,
#     pop_size=pop_size,
#     ngram=ngram,
#     tour_size=3,
#     min_res=min_res,
#     max_res=max_res,
# )

# eye = np.eye(len(set(y)))
# new_net = nn_model.fit(X_scaled, y)
# stats = nn_model.stats
# in_dict = nn_model.in_dict

# predict_train, proba_train = predict_net(nn_model.thefittest["net"], X_scaled)

# f1_score_train = f1_score(y, predict_train, average="macro")
# acc_train = accuracy_score(y, predict_train)
# confusion_matrix_train = confusion_matrix(predict_train, y)
# cat_train = cat_crossentropy2(eye[y], proba_train)

# net = nn_model.thefittest["net"]
# tree = nn_model.thefittest["individ"]

# fittest_history = np.array(nn_model.fittest_history, dtype=object)
# fittest_history = pd.DataFrame(
#     {"tree": fittest_history[:, 0], "fitness": fittest_history[:, 1]}
# )

# stats = nn_model.stats
# all_stats = pd.concat(
#     [
#         stats["proba"]["mutation"],
#         stats["proba"]["crossing"],
#         stats["proba"]["selection"],
#         stats["fitness"],
#     ],
#     axis=1,
# )

# save_net(
#     some_net=net,
#     some_tree=tree,
#     path="result_net.csv",
#     train_acc=acc_train,
#     fitness_hist=fittest_history,
#     all_stats=all_stats,
#     ngram=ngram,
# )


# mask = list(net.inputs)
# mask.sort()
# X_scaled_NN = X_scaled[:, mask[:-1]]
# columns_NN = np.array(columns, dtype=object)[mask[:-1]]

# y_NN = net.predict(X_scaled)


# nn_model = SelfGPFLClassifier(
#     iters_FC,
#     pop_size_FC,
#     max_height=10,
#     tour_size=3,
#     min_res=min_res_FC,
#     max_res=max_res_FC,
# )
# nn_model.init_sets(X_scaled_NN.shape[1], len(set(y_NN)), X_scaled_NN.shape[1] * [3])
# nn_model.fit(X_scaled_NN, y_NN, X_scaled_NN, y_NN)

# rules = nn_model.thefittest["net"].show_rules()[0]

# final_tree = nn_model.thefittest["individ"]
# final_dtree = nn_model.thefittest["net"]

# fittest_history = np.array(nn_model.fittest_history, dtype=object)
# fittest_history = pd.DataFrame(
#     {"tree": fittest_history[:, 0], "fitness": fittest_history[:, 1]}
# )

# save_rulebase(
#     some_rulebase=final_dtree,
#     some_tree=final_tree,
#     path="result_base.csv",
#     fitness_hist=fittest_history["fitness"].values,
#     tree_hist=fittest_history["tree"].values,
#     all_stats=all_stats,
#     columns=columns_NN,
# )


# print(rules)
# os.system("pause")
