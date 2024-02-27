# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 02:50:50 2024

@author: david.tapiap

This file adapts the SimpleMountainCar Preference environment in order to model the 
irrational teachers of the B-Pref testbench.

SimpleMountainCar Preference environment is as described in "Dueling Posterior Sampling
for Preference-Based Reinforcement Learning" by Ellen R. Novoseller, Yibing Wei,
Yanan Sui, Yisong Yue, Joel W. Burdick. https://arxiv.org/abs/1908.01289

The B-Pref implementation here is based upon "B-Pref: Benchmarking 
Preference-Based Reinforcement Learning", by Kimin Lee, Laura Smith, Anca Dragan
and Pieter Abbeel.  https://openreview.net/forum?id=ps95-mkHF_
"""

import numpy as np
import warnings

import sys
if "../" not in sys.path:
  sys.path.append("../")
from SimpleMountainCar import SimpleMountainCarPreferenceEnv

class SimpleMountainCarPreferenceEnvBPref(SimpleMountainCarPreferenceEnv):
    """
    This class is a wrapper for the SimpleMountainCar Preference Environment which 
    implements B-Pref's SimTeacher algorithm.
    The following extensions are made to the RiverSwimEnv class:
        1) get_trajectory_preference now takes into consideration the different
            parameters of the Simteacher algorithm and can return 0, 1, 0.5 or NaN.
            
            *   Changes might have to be made to the learning algorithms in order
                to support NaN and 0.5 as an answer.
           
           **   It doesn't really make sense to change the get_trajectory_return
                method in this scenario. Because here the reward is simply the 
                (negative) length of the episode
    """

    def __init__(self, user_noise_type, beta=np.inf, gamma=1,
                 eps=0, d_skip=0,d_equal=0):
        """
        Arguments:
            1) user_noise_type: specifies the type of noisiness in the 
                   generated preferences. In order to better align to B-Pref's
                   SimTeacher, now the degree of noisiness is specified by beta.
            3) beta: Rationality constant. The simmulated teacher is perfectly 
                    rational and deterministic as beta tends to infinity, while
                    beta = 0 will produce uniformly random choices.
                    Works the same as RiverSwimPreferenceEnv's user_noise_param.
            4) gamma: Myopic factor. If set to one, all the steps of the demonstration
                    are equally important to determine the preference. As gamma
                    tends to zero, the last steps will weigh more in the decision.
            5) eps: Error probability. The teacher flips the preference with
                    probability eps.
            6) d_skip: Skipping Threshold. If the true reward of the shown trajectories
                    does not surpass this threshold, the query will be skipped.
            7) d_equal: Equality Threshold. For the teacher to be able to decide
                    for one option over the other, the true rewards must differ
                    in more than the d_equal threshold.
                    
        *   The default teacher is the Oracle.
       **   Gamma has no meaning in this scenario.
        """
        
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.d_skip = d_skip
        self.d_equal = d_equal
        
        if gamma != 1:
            warnings.warn('gamma has no use in this environment', UserWarning)
        
        user_noise_model=beta, user_noise_type
        if beta == np.inf:
            user_noise_model[1]=0   # Just in case somebody tries to model a perfectly
                                    # rational teacher but inputs the wrong kind of
                                    # noise type.
        super().__init__(user_noise_model)
        
    '''
    # This doesn't really make much sense, because gamma is supposed to model 
    # the teacher remembering more the last steps, but in this scenario each 
    # step does not have a specific reward. 
    # In Simple Mountain Car the total reward is the negative length of the episode
    # Gamma is not applicable.
    def get_trajectory_return(self, tr,gamma=1):
        """
        Return the total reward accrued in a particular trajectory.
        Format of inputted trajectory: [[s1, s2, ..., sH], [a1, a2, ..., aH]]
        
        This modification takes into account the "myopic factor" gamma.
        With 0 < gamma < 1 the last steps of the trayectory weighs more than the
        ones at the beginning of it.
        """   
        if gamma != 1:
            reward = -(-1+gamma**len(tr[1]))/(-1+gamma) 
        else:
            reward = -len(tr[1])
        return reward
    ''' 
    
    
    def get_trajectory_preference(self, tr1, tr2):
        """
        Returns a preference between two given trajectories of states and 
        actions, tr1 and tr2. 
        
        This implementation is based on the SimTeacher algorithm as described 
        in "B-Pref: Benchmarking Preference-Based Reinforcement Learning"
        
        Format of inputted trajectories: [[s1, s2, ..., sH], [a1, a2, ..., aH]]
        
        
        Preference information: 0 = trajectory 1 preferred; 1 = trajectory 2 
        preferred; 0.5 = trajectories preferred equally (i.e., a tie);
        nan = Skipped query, none of trajectories had a high enough cumulative 
        reward for a desicion to be made.
        
        Preferences are determined by comparing the rewards accrued in the 2 
        trajectories.
        
        Here, wether noise_type is zero or not, the tie is still an option.
        First it will be checked if either of the trajectories has a cumulative
        reward higher than d_skip. If not, then the query is skipped.
        
        Then it will be checked if the difference between the rewards is significant
        enough to not declare a tie.
        
        Only after this will the noise type be relevant.
        Although B-Pref's SimTeacher always uses a logistic model to obtain the
        preferences, this implementarion does take the 3 user noise models of 
        Dueling Posterior Sampling into consideration.
        
        The 3 models are modified in order to take B-Pref's rationality constant.
        There is no myopic factor.
        """

        noise_param, noise_type = self.user_noise_model

        assert (noise_type in [0,1,2]), "noise_type %i invalid" % noise_type
        
        if noise_param != self.beta: # Just in case, although it should never be true.
            print(f'Using beta = {self.beta} as noise_param')
        
        trajectories = [tr1, tr2]
        num_traj = len(trajectories) # Siempre será 2.
        skip_or_equal = False
        
        
        # For both trajectories, determine cumulative reward / total return:
        returns = np.empty(num_traj)
        
        # Get cumulative reward for each trajectory
        for i in range(num_traj):
            
            returns[i] = self.get_trajectory_return(trajectories[i])
        
            
        if max(returns) < self.d_skip:
            preference = np.nan
            skip_or_equal = True
        elif (returns[0] - returns[1]) < self.d_equal:
            preference = 0.5
            skip_or_equal = True
        
        if not skip_or_equal:
            
            if noise_type == 0 or self.beta == np.inf:  # Deterministic preference:
                
                if returns[0] > returns[1]:
                    preference = 0
                else:
                    preference = 1
                    
                    
            elif noise_type == 1:   # Logistic noise model
                
                # Probability of preferring the 2nd trajectory:
                prob = 1 / (1 + np.exp(-self.beta * (returns[1] - returns[0])))
                
                preference = np.random.choice([0, 1], p = [1 - prob, prob])
    
            elif noise_type == 2:   # Linear noise model
                
                # Probability of preferring the 2nd trajectory:
                prob = self.beta *(returns[1] - returns[0]) + 0.5
                
                # Clip to ensure it's a valid probability:
                prob = np.clip(prob, 0, 1) # I If prob was smaller than 0 now it is zero and if it was greater than 1 now it's one.
    
                preference = np.random.choice([0, 1], p = [1 - prob, prob])                
            
            
            
            
            preference = np.random.choice([preference, 1 - preference], p=[1 - self.eps, self.eps])
        return preference
