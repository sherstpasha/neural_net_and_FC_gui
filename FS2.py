import numpy as np
from sklearn.datasets import load_iris, load_digits, load_diabetes
from thefittest.tools.transformations import SamplingGrid
from thefittest.optimizers import SelfCGA
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

from typing import Optional


def remove_duplicate_rows(arr):
    # Преобразование каждой строки в упорядоченный кортеж и использование множества для удаления дубликатов
    unique_rows = np.unique([tuple(row) for row in arr], axis=0)
    # Воссоздание массива NumPy из уникальных строк
    return np.array(unique_rows, dtype = np.int64)

class FuzzyRegressor:
    def __init__(self, iters, pop_size, n_features_fuzzy_sets, n_target_fuzzy_sets, max_rules_in_base, target_grid_volume = None):
        self.iters = iters
        self.pop_size = pop_size
        self.n_features_fuzzy_sets = n_features_fuzzy_sets
        self.n_target_fuzzy_sets = n_target_fuzzy_sets
        self.max_rules_in_base = max_rules_in_base
        self.target_grid_volume = target_grid_volume

    def find_number_bits(self, value):
        number_bits = np.ceil(np.log2(value))
        return number_bits

    def create_features_terms(self, X):
        self.features_terms_point = []
        self.ignore_terms_id = []

        for i, x_i in enumerate(X.T):
            n_fsets_i = self.n_features_fuzzy_sets[i]
            term_dict_temp = np.full((n_fsets_i + 1, 3),
                                 np.nan)
            
            self.ignore_terms_id.append(n_fsets_i)

            i = 0
            x_i_min = x_i.min()
            x_i_max = x_i.max()
            cuts, h = np.linspace(x_i_min, x_i_max, n_fsets_i,
                                  retstep = True)
            l = n_fsets_i*(i)
            r = n_fsets_i*(i+1)

            term_dict_temp[:,1][l+i:r+i] = cuts
            term_dict_temp[:,0][l+i:r+i] = term_dict_temp[:,1][l+i:r+i] - h
            term_dict_temp[:,2][l+i:r+i] = term_dict_temp[:,1][l+i:r+i] + h
            term_dict_temp[:,0][l+i] = x_i_min
            term_dict_temp[:,2][r+i-1] = x_i_max

            self.features_terms_point.append(term_dict_temp)
        
        self.features_terms_point = tuple(self.features_terms_point)

    def create_target_terms(self, y):

        target_terms_point = np.zeros(shape = (self.n_target_fuzzy_sets, 3))
        
        self.y_i_min = y.min()
        self.y_i_max = y.max()
        cuts, h = np.linspace(self.y_i_min, self.y_i_max, self.n_target_fuzzy_sets,
                              retstep = True)
        
        i = 0
        l = self.n_target_fuzzy_sets*(i)
        r = self.n_target_fuzzy_sets*(i+1)

        target_terms_point[:,1][l+i:r+i] = cuts
        target_terms_point[:,0][l+i:r+i] = target_terms_point[:,1][l+i:r+i] - h
        target_terms_point[:,2][l+i:r+i] = target_terms_point[:,1][l+i:r+i] + h
        target_terms_point[:,0][l+i] = self.y_i_min
        target_terms_point[:,2][r+i-1] = self.y_i_max

        self.target_terms_point = target_terms_point
        self.target_grid = np.linspace(self.y_i_min, self.y_i_max, self.target_grid_volume)


    def fuzzification(self, X, rule):
        fuzzy_features = self.get_triangular_membership(X, rule)
        fuzzy_features = np.nan_to_num(fuzzy_features, nan=0)
        return fuzzy_features   
    
    def inference(self, fuzzy_features):
        activation_degree = np.min(fuzzy_features, axis = 1)
        return activation_degree

    def aggregate_activations(self, cut_membership):
        return np.max(cut_membership, axis = 0)

    def get_triangular_membership(self, X, rule):
        membership_triangular = np.zeros(shape = X.shape)

        for i, term_i in enumerate(rule):

            isnanall = np.any(np.isnan(self.features_terms_point[i][term_i]))

            if isnanall:
                continue

            left = self.features_terms_point[i][term_i][0]
            center = self.features_terms_point[i][term_i][1]
            right = self.features_terms_point[i][term_i][2]

            l_mask = np.all([left <= X[:,i], X[:,i] <= center], axis = 0)
            r_mask = np.all([center < X[:,i], X[:,i] <= right], axis = 0)

            l_down = center - left
            if l_down == 0:
                l_down = 1

            r_down = right - center
            if r_down == 0:
                r_down = 1

            membership_triangular[:,i][l_mask] = (1 - (center - X[:,i])/l_down)[l_mask]
            membership_triangular[:,i][r_mask] = (1 - (X[:,i] - center)/r_down)[r_mask]
        
        return membership_triangular
    
    def get_triangular_membership_out(self, y, term):
        membership_triangular = np.zeros(shape = y.shape)

        left = term[0]
        center = term[1]
        right = term[2]

        l_mask = np.all([left <= y, y <= center], axis = 0)
        r_mask = np.all([center < y, y <= right], axis = 0)

        l_down = center - left
        if l_down == 0:
            l_down = 1

        r_down = right - center
        if r_down == 0:
            r_down = 1

        membership_triangular[l_mask] = (1 - (center - y)/l_down)[l_mask]
        membership_triangular[r_mask] = (1 - (y - center)/r_down)[r_mask]

        return membership_triangular

    def get_cut(self, interpolate_membership, activation_degrees, consequents):

        interpolate_membership_consequents = interpolate_membership[consequents]
        cuted_memberships = np.fmin(interpolate_membership_consequents[:,:,np.newaxis], activation_degrees[:,np.newaxis])
        
        return cuted_memberships

    def define_sets(self, X, y, set_names: Optional[dict[str, list[str]]] = None, feature_names: Optional[list[str]] = None, target_names: Optional[list[str]] = None):

        self.n_features = X.shape[1]

        self.n_features_target_fuzzy_sets = self.n_features_fuzzy_sets + [self.n_target_fuzzy_sets]

        self.fuzzy_sets_max_value = np.array(self.n_features_target_fuzzy_sets)
        self.fuzzy_sets_max_value[-1] = self.fuzzy_sets_max_value[-1] - 1      

        assert self.n_features == len(self.n_features_fuzzy_sets)


        if feature_names is None:
            self.feature_names = [f"X_{i}" for i in range(self.n_features)]
        else:
            self.feature_names = feature_names

        if set_names is None:
            self.set_names = {feature_name: [f"set_{i}" for i in range(n_sets)]
                         for feature_name, n_sets in zip(self.feature_names, self.n_features_fuzzy_sets)}
        else:
            self.set_names = set_names

        if target_names is None:
            self.target_names = [f"Y_{i}" for i in range(self.n_target_fuzzy_sets)]
        else:
            self.target_names = target_names

        if self.target_grid_volume is None:
            self.target_grid_volume = len(y)

        self.create_features_terms(X)
        self.create_target_terms(y)

    def fit(self, X, y):
        y = y.astype(np.float64)

        self.interpolate_membership = np.array([self.get_triangular_membership_out(self.target_grid, term) for term in self.target_terms_point])

        number_bits = np.array([self.find_number_bits(n_sets + 1)
                                for n_sets in self.n_features_target_fuzzy_sets + [1]]*self.max_rules_in_base, dtype = np.int64)

        number_bits = np.array(number_bits, dtype = np.int64)

        left = np.full(shape = len(number_bits), fill_value=0, dtype = np.float64)
        right = np.array(2**number_bits - 1, dtype = np.float64)

        grid = SamplingGrid(fit_by="parts").fit(left = left,
                                                right= right,
                                                arg = number_bits)
        
        def genotype_to_phenotype(population_g):

            population_ph = []
            population_g_int = grid.transform(population_g).astype(np.int64)

            for population_g_int_i in population_g_int:
                rulebase_switch = population_g_int_i.reshape(self.max_rules_in_base, -1)
                rulebase_switch_u = remove_duplicate_rows(rulebase_switch)

                rulebase, switch = rulebase_switch_u[:,:-1], rulebase_switch_u[:,-1]
                rulebase = rulebase[switch == 1]
                
                overborder_cond = rulebase > self.fuzzy_sets_max_value
                rulebase[overborder_cond] = (rulebase - self.fuzzy_sets_max_value)[overborder_cond]
                
                population_ph.append(rulebase)

            population_ph = np.array(population_ph, dtype = object)

            return population_ph

        def fitness_function(population_ph):

            fitness = np.zeros(shape=len(population_ph), dtype = np.float64)

            for i, rulebase_i in enumerate(population_ph):
                antecedents, consequents = rulebase_i[:,:-1], rulebase_i[:,-1]
                n_rules = len(rulebase_i)

                if n_rules < 1:
                    fitness[i] = -np.inf
                    continue
                else:

                    activation_degrees = np.array([self.inference(self.fuzzification(X, antecedent))
                                                for antecedent in antecedents])

                    cuted_memberships = self.get_cut(self.interpolate_membership, activation_degrees, consequents)


                    aggregated_activations = self.aggregate_activations(cuted_memberships)

                    up = np.sum(self.target_grid[:,np.newaxis]*aggregated_activations, axis = 0)

                    down = np.sum(aggregated_activations, axis = 0)

                    down[down == 0] = 1

                    y_predict = up/down

                    fitness[i] = r2_score(y, y_predict)

            return fitness

        optimizer = SelfCGA(fitness_function = fitness_function,
                            genotype_to_phenotype=genotype_to_phenotype,
                            iters=self.iters,
                            pop_size=self.pop_size,
                            str_len=sum(grid.parts),
                            show_progress_each=1)
        
        optimizer.fit()

        self.base = optimizer.get_fittest()["phenotype"]

        return self
        
    def predict(self, X):
        rulebase = self.base
        antecedents, consequents = rulebase[:,:-1], rulebase[:,-1]

        activation_degrees = np.array([self.inference(self.fuzzification(X, antecedent))
                                       for antecedent in antecedents])
        
        cuted_memberships = self.get_cut(self.interpolate_membership, activation_degrees, consequents)


        aggregated_activations = self.aggregate_activations(cuted_memberships)

        up = np.sum(self.target_grid[:,np.newaxis]*aggregated_activations, axis = 0)

        down = np.sum(aggregated_activations, axis = 0)

        down[down == 0] = 1

        y_predict = up/down

        return y_predict
        
    def get_text_rules(self, print_y_intervals = False):

        print(self.base)
        print(np.unique(self.base, axis = 1))
        print(remove_duplicate_rows(self.base))

        text_rules = []
        for rule_i in self.base:
            rule_text = "if "
            for j, rule_i_j in enumerate(rule_i[:-1]):
                if rule_i_j == self.ignore_terms_id[j]:
                    continue
                else:
                    rule_text += f"({self.feature_names[j]} is {self.set_names[self.feature_names[j]][rule_i_j]}) and "
                
            rule_text = rule_text[:-4]

            if print_y_intervals:
                rule_text += f"then Y is {self.target_names[rule_i[-1]]} ({self.target_terms_point[rule_i[-1]][0]}, {self.target_terms_point[rule_i[-1]][-1]})"
            else:
                rule_text += f"then (Y is {self.target_names[rule_i[-1]]})"
            text_rules.append(rule_text)
        
        return text_rules


def problem(x):
    return np.sin(x[:,0])


function = problem
left_border = -4.5
right_border = 4.5
sample_size = 200
n_dimension = 1

X = np.array([np.linspace(left_border, right_border, sample_size)
              for _ in range(n_dimension)]).T
y = function(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

features = ["x_1"]
set_names = {str(features[0]): ["Очень маленькое", "Маленькое", "Среднее", "Большое", "Очень большое"]}
targets = ["y1", "y2", "y3", "y3", "y4"]

model = FuzzyRegressor(iters=30,
                       pop_size=200,
                       n_features_fuzzy_sets = [5],
                       n_target_fuzzy_sets = 5,
                       max_rules_in_base = 30,
                       )


model.define_sets(X, y, set_names=set_names, feature_names=features, target_names=targets)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("r2_score: ", r2_score(y_test, y_pred))

text_rules = model.get_text_rules()

print(*text_rules, sep="\n")


plt.plot(X[:,0], y, label = "True y")
plt.scatter(X_test[:,0], y_pred, label = "Predict y")
plt.legend()

plt.tight_layout()
plt.savefig("test.png")