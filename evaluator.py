import numpy as np
from math import *
from network import Network

class Evaluator():
    """
    Provides a set of tools for deriving utility for a 2x2 game.
    """

    def __init__(self, mean_type, var_type, rho, alpha, network):
        """
        mean_type: list of per type means for value distribution
        var_type: list of per type variances for value distribution
        rho: risk parameter of meeting someone. Constant across each edge.
        alpha: the disutility weight of disagreeing on i,j meeting (alpha>0).
        network: the nx graph of the network
        """
        self.mean_type = mean_type
        self.var_type = var_type
        self.rho = rho
        self.alpha = alpha
        self.g = network
        self.disutility_of_death = -1000
        self.disutility_of_hospitalization = -100
        self.disutility_of_illness = -10
        self.family_scaling = .33
        # Need to fill in real values below
        self.p = {'death': .01, 'hosp': .05, 'ill': .2}
        self.p_ic = {'death': .1, 'hosp': .5, 'ill': .8}
        self.pf = {'death': .1, 'hosp': .3, 'ill': .5}
        self.pf_icf = {'death': .5, 'hosp': .7, 'ill': .9}


    def V(self, i, j):
        """
        Given two nodes and their friendship level,
        determine the value of the relationship for node i.
        """
        if self.g.get_friendship_level(i, j):
            level = 'close'
        else:
            level = 'friend'
        return np.random.normal(self.mean_type[level], self.var_type[level])

    def calc_disutility(self, p):
        return p['death'] * self.disutility_of_death +\
                        p['hosp'] * self.disutility_of_hospitalization +\
                        p['ill'] * self.disutility_of_illness

    def virus_disutility(self, i):
        """
        Return the disutility of catching covid depending on their unique status.
        c:  Subject is immunocompromised.
        f:  Number of immunocompromised family member (Or indicator var?)
        """
        node_attrs = self.g.get_node_attrs(i)
        if node_attrs['ic']:
            disutility_for_self = self.calc_disutility(self.p_ic)
        else:
            disutility_for_self = self.calc_disutility(self.p)

        if node_attrs['icf']:
            disutility_for_family = self.calc_disutility(self.pf_icf)
        else:
            disutility_for_family = self.calc_disutility(self.pf)
        disutility_for_family *= self.family_scaling

        return disutility_for_self + disutility_for_family

    def decision_utility(self, i, j, s_i, s_j):
        """
        Given two nodes and their decisions to meet, return the
        utility for node i.
        """
        return min(s_i, s_j)*(self.V(i,j) - self.virus_disutility(i)*self.rho) \
            - self.alpha * abs(s_i-s_j)

    def generate_payoff_matrix(self, i, j):
        """
        Given two nodes, return the unique 2x2 payoff matrix.
        [[both distance, I distance],
        [you distance, neither distance]]
        """
        both_dist_i = self.decision_utility(i, j, 0, 0)
        both_dist_j = self.decision_utility(j, i, 0, 0)

        i_dist_i = self.decision_utility(i, j, 0, 1)
        i_dist_j = self.decision_utility(j, i, 1, 0)

        j_dist_i = self.decision_utility(i, j, 1, 0)
        j_dist_j = self.decision_utility(j, i, 0, 1)

        none_dist_i = self.decision_utility(i, j, 1, 1)
        none_dist_j = self.decision_utility(j, i, 1, 1)

        payoff_matrix = [[(both_dist_i, both_dist_j), (i_dist_i, i_dist_j)],
                         [(j_dist_i, j_dist_j), (none_dist_i, none_dist_j)]]

        return payoff_matrix

    def transpose_2x2_matrix(self, mat):
        return [row for row in zip(*mat)]

    def get_2x2_matrix(self, l: list):
        result = \
        [[(l[0],l[4]), (l[1],l[5])],
         [(l[2],l[6]), (l[3],l[7])]]
        return result

    # Returns the indices of pure nash equilibrium for row player, given matrix form.
    def get_pure_nash_idx(self, mat, dim, debug=False):

        # Check input dimensions.
        if len(mat) == dim**3:
            mat = self.get_2x2_matrix(mat)
        if len(mat) != dim:
            raise Exception("matrix wrong dim.")
        if dim != 2:
            raise Exception("wrong dimensions.")

        # Get set of row dominant strategies.
        T = self.transpose_2x2_matrix(mat)
        row_nash_idx = set()
        for i in range(len(T)):
            lst = T[i]
            for j in range(len(lst)):
                if lst[j][0] == max(lst, key=lambda x: x[0])[0]:
                    row_nash_idx.add(i+dim*j)

        # Get set of col dominant strategies.
        col_nash_idx = set()
        for i in range(len(mat)):
            lst = mat[i]
            for j in range(len(lst)):
                if lst[j][1] == max(lst, key=lambda x: x[1])[1]:
                    col_nash_idx.add(dim*i+j)

        # Debug prints.
        if debug:
            print("row dominant strategy", row_nash_idx)
            print("col dominant strategy", col_nash_idx)

        # Find the intersection of row/col dominant strategies.
        pure_eq = row_nash_idx.intersection(col_nash_idx)

        # This game has no pure nash eq.
        if len(pure_eq) == 0:
            return None

        if debug: print("pure eq", pure_eq)

        # Return most profitable pure eq for row player.
        row_profit_by_idx = []
        for idx in list(pure_eq):
            i = floor(idx/len(mat))
            j = idx % len(mat)
            profit = mat[i][j][0]
            row_profit_by_idx.append((profit, (i,j)))
        if debug:
            print("Profit by Idx:", row_profit_by_idx)
        row_action = max(row_profit_by_idx)
        action_val = row_action[1][0]+1
        return action_val
