from evaluator import Evaluator
from network import Network

class Analyzer():

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.network = evaluator.g
    
    def get_edge_attrs(self, edge):
        attrs = ['all']
        u_attrs = self.network.get_node_attrs(edge[0])
        v_attrs = self.network.get_node_attrs(edge[1])
        if u_attrs['ic'] or v_attrs['ic']:
            attrs.append('ic')
        if u_attrs['icf'] or v_attrs['icf']:
            attrs.append('icf')
        if u_attrs['senior'] and v_attrs['senior']:
            attrs.append('senior-senior')
        elif u_attrs['senior'] != v_attrs['senior']:
            attrs.append('senior-junior')
        else:
            attrs.append('junior-junior')
        if self.network.get_friendship_level(edge[0], edge[1]):
            attrs.append('close')
        return attrs

    def _get_start_distribution(self):
        return {'both_distance': 0, 'conflict': 0, 'meet_up': 0}

    def get_outcome_distribution(self, reference_dependent=False, debug=False):
        outcomes = {
            'all': self._get_start_distribution(),
            'ic': self._get_start_distribution(),
            'icf': self._get_start_distribution(),
            'senior-senior': self._get_start_distribution(),
            'junior-junior': self._get_start_distribution(),
            'senior-junior': self._get_start_distribution(),
            'close': self._get_start_distribution()
        }

        for edge in self.network.g.edges():
            payoff_matrix = self.evaluator.generate_payoff_matrix(edge[0], edge[1], reference_dependent)
            (row_action, col_action) = self.evaluator.get_pure_nash_idx(payoff_matrix)
            attrs = self.get_edge_attrs(edge)
            if debug:
                print(payoff_matrix)
                print(row_action, col_action)
                print(self.evaluator.g.get_node_attrs(edge[0]))
                print(self.evaluator.g.get_node_attrs(edge[1]))
                print(attrs)
            if row_action != col_action:
                for attr in attrs:
                    outcomes[attr]['conflict'] += 1
            else:
                if row_action == 0:
                    for attr in attrs:
                        outcomes[attr]['both_distance'] += 1
                else:
                    for attr in attrs:
                        outcomes[attr]['meet_up'] += 1
        return outcomes