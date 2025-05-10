# -*- coding: utf-8 -*-
"""
This file defines the RiverSwim environment and RiverSwim preference environment.

The RiverSwim environment is as described in "(More) Efficient Reinforcement 
Learning via Posterior Sampling" by Ian Osband, Benjamin Van Roy, and Daniel
Russo (2013).
"""

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

class RiverSwimEnv(gym.Env): #RiverSwimEnv es una clase hija de gym.Env
    """
    This class defines the RiverSwim environment.
    """
    
    def __init__(self, num_states = 6):
        """
        Constructor for the RiverSwim environment class.        
        Input: num_states = number of states in the MDP.
        """
        
        # Initialize state and action spaces.
        self.nA = 2     # Two actions: left and right # Por definición, en RiverSwim solo se puede ir a la izquierda o a la derecha, no haymás acciones posibles.
        self.action_space = spaces.Discrete(self.nA) # A = {0,1} Espacio de acciones (izq, der)
        self.observation_space = spaces.Discrete(num_states) # S = {0,1,2,3,4,5} Espacio de estados
        self.nS = num_states
        
        self.states_per_dim = [num_states]  # State space has only one dimension
        self.store_episode_reward = False   # Track rewards at each step, not
                                            # over whole episode
        self.done = False                   # This stays false: in this 
        # environment, an episode can only finish at the episode time horizon.
        
        self._seed()

        # Construct transition probability matrix and rewards. Format:
        # self.P[s][a] is a list of transition tuples (prob, next_state, reward).
        self.P = {} #Se inicializa como un diccionario vacío.
        
        for s in range(self.nS): # Para cada estado
            
            self.P[s] = {a : [] for a in range(self.nA)} # Inicializa una lista vacía para ambas acciones en cada estado
            
            for a in range(self.nA):
                
                if a == 0:  # Left action
                    
                    next_state = np.max([s - 1, 0]) # Si voy hacia la izquierda, el siguiente estado será s-1 ó 0, el que sea mayor.
                    
                    reward = 5/1000 if (s == 0 and next_state == 0) else 0 # Escogiendo ir a a la izquierda, si me quedo en el estado inicial, la recompensa será 5/1000. Si no estoy en el estado inicial o mi estado siguiente no es el inicio, si retrocedo, la recompensa es 0
                    
                    self.P[s][a] = [(1, next_state, reward)] # Si quiero ir a la izquierda, es un sistema determinístico.
                    
                elif s == 0:  # Leftmost state, and right action # Trato de ir a la derecha desde el estado inicial.
                    
                    self.P[s][a] = [(0.4, s, 0), (0.6, s + 1, 0)]   # Hay un 60% de probabilidad de que logre avanzar. Ya sea que lo logre o no, la recompensa será cero.
                    
                elif s == self.nS - 1:   # Rightmost state, and right action
                    
                    self.P[s][a] = [(0.4, s - 1, 0), (0.6, s, 1)] # Hay un 60% de probabilidad de que logre quedarme en el estado final. Si logro esto, la recompensa es 1. Si retrocedo, la recompensa es 0.
                    
                else:   # Intermediate state, and right action
                    
                    self.P[s][a] = [(0.05, s - 1, 0), (0.6, s, 0), 
                          (0.35, s + 1, 0)] # Si estoy en un estado intermedio y trato de avanzar a la izquierda, hay un 5% de probabilidad de retroceder, un 60% de quedarme en el lugar y un 35% de avanzar. Mientras no parta en el estado final, no importa lo que pase, mi recompensa será 0.

        # Reset the starting state:
        self.reset()
        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        """
        Reset initial state, so that we can start a new episode. Always start
        in the leftmost state.
        """
        
        self.state = 0
        return self.state
    
    def step(self, action): # Define step, sobreescribiendo la función step de gym.Envs. 
        """
        Take a step using the transition probability matrix specified in the 
        constructor.
        
        Input: action = 0 (left action) or 1 (right action).
        """        
        transition_probs = self.P[self.state][action]
        
        num_next_states = len(transition_probs) # Número de siguientes estados posibles, según la distribución de probabilidad para el estado actual y la acción escogida.

        next_state_probs = [transition_probs[i][0] for i in range(num_next_states)] # La probabilidad de ir a parar a cada uno de esos siguientes estados posibles
            
        outcome = np.random.choice(np.arange(num_next_states), p = next_state_probs) # Se escoge aleatoriamente un resultado según las probabilidades definidas. Avanzar, Quedarse, Retroceder (si estamos en un estado que soporta 3 estados siguientes posibles)
        
        self.state = transition_probs[outcome][1]    # Update state # El estado al que se lle
        reward = transition_probs[outcome][2]
        
        return self.state, reward, self.done    # done = False always
        # Esta definición de step retorna el estado al que se llegó, la recompensa que se obtuvo e informa que aún no se termina.
        # Por definición, done siempre será falso, pues la única forma de terminar el proceso es cuando se acabe el tiempo.
        # El nadador quiere quedarse al final del río, contra la corriente, la mayor cantidad de tiempo posible, pero no puede simplemente llegar a un punto y decidir salir.
    
    def get_step_reward(self, state, action, next_state):
        """
        Return the reward corresponding to the given state, action and 
        subsequent state.
        """
    
        transition_probs = self.P[state][action]
        
        reward = 0
        
        for i in range(len(transition_probs)):
            
            if transition_probs[i][1] == next_state:
                reward = transition_probs[i][2]
                break
            
        return reward
    
        
    def get_trajectory_return(self, tr):
        """
        Return the total reward accrued in a particular trajectory.
        Format of inputted trajectory: [[s1, s2, ..., sH], [a1, a2, ..., aH]]
        """    
        
        states = tr[0]
        actions = tr[1]
        
        # Sanity check:        
        if not len(states) == len(actions) + 1:
            print('Invalid input given to get_trajectory_return.')
            print('State sequence expected to be one element longer than corresponding action sequence.')      
        
        total_return = 0
        
        for i in range(len(actions)):
            
            total_return += self.get_step_reward(states[i], actions[i], \
                                         states[i + 1])
            
        return total_return

         
      
class RiverSwimPreferenceEnv(RiverSwimEnv): # No es una clase hija, la relación entre clases es más del tipo "tiene un" que "es un". Contiene un RiverSwimEnv y puede modificar o extender su comportamiento sin necesidad de heredar de ella.
    """
    This class is a wrapper for the RiverSwim environment, which gives
    preferences over trajectories instead of numerical rewards at each step.
    
    The following extensions are made to the RiverSwimEnv class defined above:
        1) The step function no longer returns reward feedback.
        2) We add a function that calculates a preference between 2 inputted
            trajectories.
    """
    # Modifica RiverSwimEnv para poder utilizarlo en algoritmos basados en preferencias.

    def __init__(self, user_noise_model, num_states = 6):
        """
        Arguments:
            1) user_noise_model: specifies the degree of noisiness in the 
                   generated preferences. See description of the function 
                   get_trajectory_preference for details.
            2) num_states: number of states in the MDP.
        """
        
        self.user_noise_model = user_noise_model
        
        super().__init__(num_states) # Esto lo está haciendo como si fuese una clase hija, está iniciando un RiverSwimEnv con los estados que le entregues al constructor de RiverSwimPreferenceEnv

    def step(self, action):
        """
        Take a step using the transition probability matrix specified in the 
        constructor. This is identical to the RiverSwim class, except that now 
        we no longer return the reward.
        """
        state, _, done = super().step(action)
        return state, done
        # Dada la naturaleza del aprendizaje basado en preferencias, el método 
        # step() de RiverSwimPreferenceEnv no entrega el valor de recompensa 
        # asociado al paso actual, simplemente retorna el estado al que se 
        # llega e informa si se ha terminado el episodio (En RiverSwimEnv este 
        # valor es siempre False).
       
   
    def get_trajectory_preference(self, tr1, tr2):
        """
        Return a preference between two given trajectories of states and 
        actions, tr1 and tr2.
        
        Format of inputted trajectories: [[s1, s2, ..., sH], [a1, a2, ..., aH]]
        
        Preference information: 0 = trajectory 1 preferred; 1 = trajectory 2 
        preferred; 0.5 = trajectories preferred equally (i.e., a tie).
        
        Preferences are determined by comparing the rewards accrued in the 2 
        trajectories.
        
        self.user_noise_model takes the form [noise_type, noise_param].
        
        noise_type should be equal to 0, 1, or 2.
        noise_type = 0: deterministic preference; return 0.5 if tie.
        noise_type = 1: logistic noise model; user_noise parameter determines 
        degree of noisiness.
        noise_type = 2: linear noise model; user_noise parameter determines 
        degree of noisiness
        
        noise_param is not used if noise_type = 0. Otherwise, smaller values
        correspond to noisier preferences.
        """          
        
        # Unpack self.user_noise_model:
        noise_param, noise_type = self.user_noise_model

        assert (noise_type in [0,1,2]), "noise_type %i invalid" % noise_type
        
        trajectories = [tr1, tr2]
        num_traj = len(trajectories) # Siempre será 2.
        
        # For both trajectories, determine cumulative reward / total return:
        returns = np.empty(num_traj)
        
        # Get cumulative reward for each trajectory
        for i in range(num_traj):
            
            returns[i] = self.get_trajectory_return(trajectories[i])
            
        if noise_type == 0:  # Deterministic preference:
            
            if returns[0] == returns[1]:  # Compare returns to determine preference
                preference = 0.5
            elif returns[0] > returns[1]:
                preference = 0
            else:
                preference = 1
                
        elif noise_type == 1:   # Logistic noise model
            
            # Probability of preferring the 2nd trajectory:
            prob = 1 / (1 + np.exp(-noise_param * (returns[1] - returns[0])))
            
            preference = np.random.choice([0, 1], p = [1 - prob, prob])

        elif noise_type == 2:   # Linear noise model
            
            # Probability of preferring the 2nd trajectory:
            prob = noise_param * (returns[1] - returns[0]) + 0.5
            
            # Clip to ensure it's a valid probability:
            prob = np.clip(prob, 0, 1)

            preference = np.random.choice([0, 1], p = [1 - prob, prob])                
        
        return preference


