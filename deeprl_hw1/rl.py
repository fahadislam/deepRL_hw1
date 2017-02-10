# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np


def evaluate_policy(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    
    V = np.zeros(env.nS)
    # V_last = np.ones(env.nS)*100
    for i in range(max_iterations):
        for s in range(env.nS):
            V[s] = sum([(p[0] * (p[2] + gamma * V[p[1]])) for p in env.P[s][policy[s]]])
        # if np.linalg.norm(V[s]-V_last[s]) < tol:
        #   break
        # V_last = V
    """Evaluate the value of a policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.


    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray
      The value for the given policy
    """
    return i, V


def value_function_to_policy(env, gamma, value_function):
    
    policy = np.zeros(env.nS, dtype='int')
    for s in range(env.nS):
        X = []
        for a in range(env.nA):
            v = sum((p[0] * (p[2] + gamma * value_function[p[1]])) for p in env.P[s][a])    #all possible transitions
            X.append(v)
        policy[s] = X.index(max(X))
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """    
    return policy


def improve_policy(env, gamma, value_func, policy):
    
    policy_new = value_function_to_policy(env, gamma, value_func)
    """Given a policy and value function improve the policy.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

        Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    return ~np.array_equal(policy, policy_new), policy_new 


def policy_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):

    # actions = [lake_env.LEFT, lake_env.DOWN, lake_env.RIGHT, lake_env.UP]
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)

    for loop in range(max_iterations):
        # evaluation
        iterations, value_func = evaluate_policy(env, gamma, policy)

        # improvement
        is_improved, policy = improve_policy(env, gamma, value_func, policy)
            
        if ~is_improved:
            break         
    ####
    """Runs policy iteration.

    See page 87 (pg 105 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    You should use the improve_policy and evaluate_policy methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    return policy, value_func, loop, iterations


def value_iteration(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
    book.

    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    return np.zeros(env.nS), 0


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)
