import numpy as np
from scipy.special import factorial

class match_stats:

    CALC_LIMIT = 100.0 # Used in approximation of infinite sums

    def __init__(self, our_rate, their_rate, current_lead, remaining_time):
        '''
        our_rate - Our goal rate per match
        their_rate - Their goal rate per match
        current_lead - Our current goal lead (negative if losing)
        remaining_time - Fraction of match remaining
        '''

        self.our_rate = our_rate
        self.their_rate = their_rate
        self.current_lead = current_lead
        self.remaining_time = remaining_time

    def our_goals_exp(self):
        '''Our expected number of goals to come'''

        return self.our_rate * self.remaining_time

    def their_goals_exp(self):
        '''Their expected number of goals to come'''

        return self.their_rate * self.remaining_time

    def prob_draw(self):
        '''Probability of draw'''

        n = np.arange(self.CALC_LIMIT)
        if self.current_lead > 0:
            to_sum = ((self.our_goals_exp()**n) * (self.their_goals_exp()**(n+self.current_lead))) / (factorial(n) * factorial(n+self.current_lead))
        else:
            to_sum = ((self.our_goals_exp()**(n-self.current_lead)) * (self.their_goals_exp()**(n))) / (factorial(n) * factorial(n-self.current_lead))

        return np.exp(-1*(self.our_goals_exp() + self.their_goals_exp())) * np.sum(to_sum)


    def prob_win(self):
        '''Probability of win'''

        n = np.arange(self.CALC_LIMIT)
        our_terms = self.our_goals_exp()**n / factorial(n)
        their_terms = self.their_goals_exp()**n / factorial(n)
        product = np.outer(our_terms, their_terms)
        win_test = np.outer(n + 1 + self.current_lead, (n+1)**(-1))
        win_prob = np.sum(product[win_test > 1]) * np.exp(-1*(self.our_goals_exp() + self.their_goals_exp()))
        
        return win_prob

    def prob_lose(self):
        '''Probability of loss'''

        return 1 - (self.prob_win() + self.prob_draw())