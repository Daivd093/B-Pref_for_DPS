# -*- coding: utf-8 -*-
"""
This script implements infinite-horizon and finite-horizon value iteration.

The infinite-horizon value iteration code is adapted from Denny Britz. See:
    https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb
    http://www.wildml.com/2016/10/learning-reinforcement-learning/
    
"""

import numpy as np

def value_iteration(P, R, num_states, num_actions, theta=0.0001, 
                    discount_factor=1.0, epsilon = 0, H = np.infty):

    """
    Value Iteration Algorithm: infinite and finite-horizon.
    
    Inputs:
        1) P[s][a] is an array of length num_states, with the probability of 
        transitioning to each state after taking action a in state s.
        2) R is a num_states x num_actions matrix (or num_states-length vector), 
        where R[s, a] (or R[s]) is the expected reward for taking action a in 
        state s (or for landing in state s).
        3) num_states is a number of states in the environment. 
        4) num_actions is a number of actions in the environment.
        5) theta: For infinite-horizon case only. We stop evaluation once our 
            value function change is less than theta for all states. Ignored if
            a finite H is specified.
        6) discount_factor: Gamma discount factor.
        7) epsilon: make policy epsilon-greedy with parameter epsilon.
        8) H: episode horizon. Set to np.infty for infinite horizon case (this is
        the default), or to a positive integer for the finite-horizon case.
        
    Returns:
        1) A tuple (policy, V) of the optimal policy and corresponding value 
           function. The policy is a policy is num_states x num_actions matrix 
           for the infinite horizon case, and an H x num_states x num_actions) 
           matrix for the finite horizon case. The value funtion V is a 
           num_states-length vector for infinite horizon case, and an
           H x num_states matrix for the finite horizon case.
    """
    
    if H == np.infty:    # Infinite-horizon value iteration
        
        policy, V = value_iteration_inf_horizon(P, R, num_states, num_actions, 
                        theta, discount_factor, epsilon)
        
    else:    # Finite-horizon value iteration
        
        policy, V = value_iteration_finite_horizon(P, R, num_states, 
                    num_actions, H, discount_factor, epsilon)
        
    return policy, V


def value_iteration_inf_horizon(P, R, num_states, num_actions, theta=0.0001, 
                    discount_factor=1.0, epsilon = 0):
    """
    Value Iteration Algorithm: infinite horizon.
    
    Inputs: same as for value_iteration function above, except without the 
        episode horizon argument H.
        
    Returns:
        1) A tuple (policy, V) of the optimal policy and corresponding value 
           function. See value_iteration function above for details.
    """
    
    # If given only state rewards, as opposed to state-action rewards, then 
    # set the following flag to True.
    states_only = (len(R.shape) == 1 or R.shape[1] == 1) # Si R es 1D, entonces
                                                         # es R(s), no R(s,a).
    
    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value of all actions in a given state.
        
        Inputs:
            state: The state to consider (int)
            V: The value to use as an estimator: vector of length num_states
        
        Returns:
            A vector of length num_actions containing the expected value of each action.
        """
        A = np.zeros(num_actions)
        # Busca encontrar la recompensa esperada para cada acción desde un estado dado,
        # utilizando una aproximación del valor esperado de las recompensas dada. 
        for a in range(num_actions):
            for next_state, prob in enumerate(P[state][a]):
                if states_only:
                    reward = R[next_state] # Si depende del estado no más, R = R(s')
                else:
                    reward = R[state, a] # Si no, R = R(s,a)
                
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A
    
    V = np.zeros(num_states)
    # Itera para encontrar el valor real de V[s] para cada estado.
    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(num_states):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V) # Busca encontrar la recompensa esperada por acción para mejorar la aproximación de V[s]
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10. 
            V[s] = best_action_value        
        # Check if we can stop 
        if delta < theta: # Con horizonte infinito se detiene el cálculo de V[s] cuando este valor converja. Se usa theta para determinar cuándo ocurre esto.
            break
    
    # Create an epsilon-greedy policy using the optimal value function:
    policy = (epsilon / num_actions) * np.ones([num_states, num_actions])
        
    for s in range(num_states):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V) # Busca eencontrar la recompensa esperada por acción una vez que ya se conocen los valores de V[s]. Para poder costruir la política.
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] += 1 - epsilon
        # epsilon = 0 entregaría una política determinística, donde en cada estado
        # la política toma el valor 1 para la mejor acción posible y 0 para todos los otros.
        # Siempre se escogerá explotación por sobre exploración.
        
        # Mientras que con epsilon > 0, la mejor opción toma el valor epsilon/num_actions + 1-epsilon = 1+e(-1+1/n) 
        # y las otras opciones toman el valor epsilon/num_actions. En total estos valores suman 1.
        
        # De esta forma, entre mayor sea epsilon, menor será el valor de la política para la acción óptima
        # y mayor será su valor para las otras opciones, incentivando la exploración.
    return policy, V

def value_iteration_finite_horizon(P, R, num_states, num_actions, H = 50, 
                    discount_factor=1.0, epsilon = 0):
    """
    Value Iteration Algorithm: finite horizon.
    
    Inputs: same as for value_iteration function above, except without the 
        stopping condition parameter theta.
        
    Returns:
        1) A tuple (policy, V) of the optimal policy and corresponding value 
           function. See value_iteration function above for details.
    """
    
    R = R.flatten() # Ya sea que R = R(s,a) pasa a ser R(i), donde i es un índice que representará cada una de las combinaciones de estado x acción que podrían llegar a existir. R se convierte en un vector 1D
    # Al parecer esta versión no concibe la opción de que la recompensa dependa solamente del estado actual.
    
    
    # Initialize value function V and action-value function Q:
    V = np.zeros((H + 1, num_states)) # Recompensa acumulada esperada desde el estado actual para cualquier acción. V_t(s)
    Q = np.zeros((H + 1, num_states * num_actions)) # Recompensa acumulada esperada desde el estado actual luego de la acción actual. Q_t(s,a)
    
    # Create probability transition matrix:
    prob_matrix = np.empty((num_states * num_actions, num_states)) # En el eje 0 tiene espacio para todas las combinaciones posibles de estado x acción y en el eje 1 tiene espacio para num_states valores de probabilidad.
    # Por cada combinación posible de estado y acción (representado por count) hay asignado un vector de las probabilidades de ir a parar a cada uno de los estados existentes.
    
    count = 0
    
    for state in range(num_states):
        for action in range(num_actions):
            
            prob_matrix[count, :] = P[state][action] # El valor de la matriz de probabilidad en el índice count-ésimo será igual a un vector de longitud num_states con la probabilidad de llegar a cada uno de los estados.
            count += 1 # Count es el índice que caracteriza a la combinación state x action
        
    # En la versión finita de Value iteration no es necesaria una función como one_step_lookahead(state, V) porque esta era utilizada para estimar el valor cuando V es infinito. En este caso podemos llegar al final del episodio y ver qué pasa.
    
    # Create an epsilon-greedy policy using the optimal action-value function:
    policy = (epsilon / num_actions) * np.ones([H, num_states, num_actions])
    # Se inicializa de la misma forma que en el caso infinito, con la diferencia de que ahora la política depende también del paso actual, por lo que es 3D.
    # policy_t(s,a) en vez de simplemente p(s,a)
    
    # Iterate through each time step:
    for t in np.arange(H - 1, -1, -1):
        
        Q[t, :] = R + discount_factor * prob_matrix @ V[t + 1, :] # Q_t = R_vector1D + factor_de_descuento * matriz_de_probabilidad_de_pasar_de_combinación_sxa_a_cualquier_estado . V_{t+1}
        # El valor esperado de tomar cualquier combinación de estado x acción en el tiempo t es igual a las recompensas asociadas a cada combinación sxa más la multiplicación matricial entre la matriz de probabilidad y la recompensa esperada desde cualquier estado en t+1, multiplicado por un factor de descuento..
        # Es decir, la recompensa por la combinación R[count] más (descuento por) la probabilidad de pasar a cualquier estado debido a la combinación count multiplicado por el valor esperado de la recompensa asociada a llegar a cada uno de esos estados en el tiempo t+1  
        
        # Simplemente está calculando el valor de la recompensa esperada para cada par [estado,acción] posible en el tiempo t.
        
        # Update value function for this time step:
        Q_matrix = Q[t, :].reshape((num_states, num_actions)) # Luego, para actualizar V_t(s), se crea Q_m = Q_t(state,action) en vez de utilizar Q_t(count)
        V[t, :] = np.max(Q_matrix, axis = 1) # Un vector con el valor máximo de cada fila de Q_matrix_{num_states x num_actions}
        # V_t(s) va a ser igual a la recompensa esperada para el estado s. Se consideran las recompensas esperadas en t para cada par [s, acción] y se asigna a V_t(s) aquel cuya acción entregue una mayor recompensa esperada, según los datos de Q_t(s,acción).
        
        # Best actions at this time step for each state:
        best_actions = np.argmax(Q_matrix, axis = 1)
        # Las mejores acciones para cada s en el tiempo t serán aquellas acciones que llevan la recompensa que se utilizó parae actualizar V_t anteriormente.
        
        # Update policy for this time step:
        for state, best_action in enumerate(best_actions):
            policy[t, state, best_action] += 1 - epsilon
            
    return policy, V # Se retorna la política y el valor esperado de recompensa por estado, como en el caso finito, no el valor esperado por estado y acción (Q).
    

"""
En el caso infinito es más eficiente calcular V directamente, pues no es necesario
obtener las combinaciones estado x acción, sólo importa el estado en que se esté.

En el caso finito, se suele calcular Q para luego obtener V, pues por la naturaleza
finita del proceso, importa llegar a un estado dado en el menor tiempo posible,
para que no se acabe el tiempo antes de esto.
 
"""

