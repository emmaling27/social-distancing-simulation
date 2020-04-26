import numpy as np
from math import *
from network import Network

class Evaluator():
    """
    Provides a set of tools for deriving utility for a 2x2 game.
    """

    def __init__(self, mean_type, var_type, rho):
        """
        mean_type: list of per type means for value distribution
        var_type: list of per type variances for value distribution
        rho: risk parameter of meeting someone. Constant across each edge.
        alpha: the disutility weight of disagreeing on i,j meeting (alpha>0).
        """
        self.mean_type = mean_type
        self.var_type = var_type
        self.rho = rho
        self.alpha = alpha

    def extract_node_data(self, node):
        pass
        
    def V(self, i, j):
        """
        Given two nodes and their friendship level, 
        determine the value of the relationship for node i.
        """
        level = # TODO:  Extract level info from graph.
        v = np.random.normal(self.mean_type[level], self.var_type[level])
        pass
    
    def virus_disutility(self, c, f):
        """
        Return the disutility of catching covid depending on their unique status.
        c:  Subject is immunocompromised.
        f:  Number of immunocompromised family member (Or indicator var?)
        """
        pass
        
    def decision_utility(self, i, j, s_i, s_j):
        """
        Given two nodes and their decisions to meet, return the
        utility for node i.
        """
        
        c_i, f_i = self.extract_node_data(i)
        d_i = self.virus_disutility(c_i, f_i)
        
        u_i = min(s_i, s_j)*(self.V(i,j) - d_i*self.rho) - self.alpha * abs(s_i-s_j)
        
        return u_i

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








