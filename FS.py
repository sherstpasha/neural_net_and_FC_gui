import numpy as np
from sklearn.datasets import load_iris
from thefittest.tools.transformations import SamplingGrid
from thefittest.optimizers import SelfCGA


class FuzzyClassifier:
    def __init__(self, iters, pop_size, n_features_fuzzy_sets, max_rules_in_base):
        self.iters = iters
        self.pop_size = pop_size
        self.n_features_fuzzy_sets = n_features_fuzzy_sets
        self.max_rules_in_base = max_rules_in_base
        # self.n_features_fuzzy_sets_in_base = n_features_fuzzy_sets*max_rules_in_base

    def create_terms(self, X, y):
        self.terms = []

        for i, x_i in enumerate(X.T):
            n_fsets_i = self.n_features_fuzzy_sets[i]
            term_dict_temp = np.full((n_fsets_i + 1, 3),
                                 np.nan)

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

            self.terms.append(term_dict_temp)
        
        self.terms = tuple(self.terms)

    def find_number_bit(self, value):
        number_bit = np.ceil(np.log2(value))
        return number_bit

    def rule_membership(self, x, rule):
        result = self.culc_triangular_r(x, rule)
        if np.isnan(result).all():
            return np.full(x.shape[0], 0)
        else:
            return np.nanmean(result, axis = 1)     
    
    def culc_triangular_r(self, x, rule_id):
        result = np.full(x.shape, np.nan)
        
        print(rule_id)
        left = self.terms[rule_id][:,0]
        center = self.terms[rule_id][:,1]
        right = self.terms[rule_id][:,2]
        
        l_mask = np.all([left <= x, x <= center], axis = 0)
        r_mask = np.all([center < x, x <= right], axis = 0)
        else_mask = np.invert(l_mask) & np.invert(r_mask)
        isnanall = np.any(np.isnan(self.term_dict[rule_id]),
                          axis = 1)
        else_mask[:,isnanall] = False
        
        l_down = center - left
        l_down[l_down == 0] = 1
        r_down = right - center
        r_down[r_down == 0] = 1

        result[l_mask] = (1 - (center - x)/l_down)[l_mask]
        result[r_mask] = (1 - (x - center)/r_down)[r_mask]
        result[else_mask] = 0

        if len(rule_id) == 1:
            return result[:,0]
        
        return result  

    def fit(self, X, y):
        self.n_features = X.shape[1]

        assert self.n_features == len(self.n_features_fuzzy_sets)

        self.create_terms(X, y)

        number_bits = np.array([self.find_number_bit(n_sets + 1)
                                for n_sets in self.n_features_fuzzy_sets + [1]]*self.max_rules_in_base, dtype = np.int64)

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

                rulebase, switch = rulebase_switch[:,:-1], rulebase_switch[:,-1]

                rulebase = rulebase[switch == 1]

                overborder_cond = rulebase > self.n_features_fuzzy_sets
                rulebase[overborder_cond] = (rulebase - self.n_features_fuzzy_sets)[overborder_cond]
                
                population_ph.append(rulebase)

            population_ph = np.array(population_ph, dtype = object)

            return population_ph


        def fitness_function(population_ph):

            for rulebase in population_ph:

                print(rulebase)

                query = np.array([self.rule_membership(X, rule[:-1])
                              for rule in rulebase])
                
                print(query)
                # print(population_ph_i)
                # print(switch, rulebase)
                # # print('rulebase', rulebase)

                # rules = rulebase.reshape()
                return

            return np.ones(len(population_ph))
        
        optimizer = SelfCGA(fitness_function = fitness_function,
                            genotype_to_phenotype=genotype_to_phenotype,
                            iters=self.iters,
                            pop_size=self.pop_size,
                            str_len=sum(grid.parts),
                            show_progress_each=1)
        
        optimizer.fit()

        



        # self.n_fsets = n_fsets
        # self.n_rules = n_rules
        # self.rl = rl
        # self.bl = bl
        # self.tour_size = tour_size
        # self.keep_best = keep_best
        # self.K = K
        # self.threshold = threshold
        
        # self.n_bin = np.ceil(np.log2(self.n_fsets+1)).astype(int)
        
        # self.n_bin_c = None
        # self.n_vars = None
        # self.n_classes = None
        # self.len_ = None
        # self.term_dict = None
        # self.ignore_dict = None
        # self.list_ = np.array([], dtype = int)
        
        # self.borders = {'left':np.array([0]),
        #                 'right':np.array([1])}
        # self.parts = np.array([1], dtype = int)
        # self.grid_model = None
        
        # self.base = None
        # self.opt_model = None

    # def create_terms(self, X, y):
    #     self.n_vars = X.shape[1]
    #     self.n_classes = len(np.unique(y))
    #     self.n_bin_c = np.ceil(np.log2(self.n_classes)).astype(int)
    #     self.len_ = (1 + self.n_bin_c + X.shape[1]*self.n_bin)*self.n_rules
        
    #     self.term_dict = np.full((self.n_fsets*X.shape[1] + X.shape[1], 4),
    #                              np.nan)
        
    #     for i, x_i in enumerate(X.T):
    #         x_i_min = x_i.min()
    #         x_i_max = x_i.max()
    #         cuts, h = np.linspace(x_i_min, x_i_max, self.n_fsets,
    #                               retstep = True)
    #         l = self.n_fsets*(i)
    #         r = self.n_fsets*(i+1)
    #         self.borders['left'] = np.append(self.borders['left'],
    #                                          l + i)
    #         self.borders['right'] = np.append(self.borders['right'],
    #                                           l + i + 2**self.n_bin - 1)
    #         self.term_dict[:,1][l+i:r+i] = cuts
    #         self.term_dict[:,0][l+i:r+i] = self.term_dict[:,1][l+i:r+i] - h
    #         self.term_dict[:,2][l+i:r+i] = self.term_dict[:,1][l+i:r+i] + h
    #         self.term_dict[:,3][l+i:r+i] = i
    #         self.term_dict[:,0][l+i] = x_i_min
    #         self.term_dict[:,2][r+i-1] = x_i_max
    #         self.term_dict[r+i][3] = i
    #         self.parts = np.append(self.parts, self.n_bin)
    #     self.list_ = np.arange((self.n_fsets+1)*X.shape[1]).reshape(X.shape[1], -1)
    #     self.ignore_dict = self.list_[:,-1]
    #
        

data = load_iris()
X = data.data
y = data.target

targets = data.target_names
features = data.feature_names

print(targets)
print(features)

n_features_fuzzy_sets = [3, 4, 5, 6]

model = FuzzyClassifier(iters=5,
                        pop_size=10,
                        n_features_fuzzy_sets = n_features_fuzzy_sets,
                        max_rules_in_base=3)


model.fit(X, y)