#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import input

import deeprl_hw1.lake_envs as lake_env
import gym
import time
import deeprl_hw1.rl as rl

def run_policy(env, policy):
    """Run a policy for the given environment.

    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    initial_state = env.reset()
    env.render()
    # time.sleep(1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 0

    state = initial_state
    while True:
        nextstate, reward, is_terminal, debug_info = env.step(
            policy[state])
        env.render()

        total_reward += reward
        num_steps += 1

        if is_terminal:
            break

        time.sleep(1)
        state = nextstate

    return total_reward, num_steps

def run_random_policy(env):
    """Run a random policy for the given environment.

    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    initial_state = env.reset()
    env.render()
    time.sleep(1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 0
    while True:
        nextstate, reward, is_terminal, debug_info = env.step(
            env.action_space.sample())
        env.render()

        total_reward += reward
        num_steps += 1

        if is_terminal:
            break

        time.sleep(1)

    return total_reward, num_steps


def print_env_info(env):
    print('Environment has %d states and %d actions.' % (env.nS, env.nA))


def print_model_info(env, state, action):
    transition_table_row = env.P[state][action]
    print(
        ('According to transition function, '
         'taking action %s(%d) in state %d leads to'
         ' %d possible outcomes') % (lake_env.action_names[action],
                                     action, state, len(transition_table_row)))
    for prob, nextstate, reward, is_terminal in transition_table_row:
        state_type = 'terminal' if is_terminal else 'non-terminal'
        print(
            '\tTransitioning to %s state %d with probability %f and reward %f'
            % (state_type, nextstate, prob, reward))

def main():
    # create the environment
    # env = gym.make('FrozenLake-v0')
    # uncomment next line to try the deterministic version

    # env = gym.make('Deterministic-4x4-FrozenLake-v0')
    # env = gym.make('Deterministic-8x8-FrozenLake-v0')
    # env = gym.make('Stochastic-4x4-FrozenLake-v0')
    # env = gym.make('Stochastic-8x8-FrozenLake-v0')
    env = gym.make('Deterministic-4x4-neg-reward-FrozenLake-v0')

    print_env_info(env)
    print_model_info(env, 0, lake_env.DOWN)
    print_model_info(env, 1, lake_env.DOWN)
    print_model_info(env, 14, lake_env.RIGHT)

    input('Hit enter to run a random policy...')
    # method = 0  # policy iteration
    method = 1  # value iteration
    gamma = 0.9
    ### random policy
    # total_reward, num_steps = run_random_policy(env)

    start_time = time.time()

    if method == 0:
        policy, value_func, iter_policy, iter_value = rl.policy_iteration(env, gamma)
        print(value_func)
    else:
        value_func, iter_value = rl.value_iteration(env, gamma)
        print(value_func)
        policy = rl.value_function_to_policy(env, gamma, value_func)

    rl.print_policy(policy, lake_env.action_names)
    print("--- %s seconds ---" % (time.time() - start_time))

    total_reward, num_steps = run_policy(env, policy)

    print('Agent received total reward of: %f' % total_reward)
    print('Agent took %d steps' % num_steps)


if __name__ == '__main__':
    main()
