import gym
import time
from gym.envs.registration import register
import argparse
import numpy as np

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-e', '--env', default='covering', type=str)

args = parser.parse_args()

def main():

    if args.env == 'soccer':
        register(
            id='multigrid-soccer-v0',
            entry_point='gym_multigrid.envs:SoccerGame4HEnv10x15N2',
        )
        env = gym.make('multigrid-soccer-v0')

    else:
        register(
            id='multigrid-covering-v0',
            entry_point='gym_multigrid.envs:CoveringGame4HEnv10x10N3',
        )
        env = gym.make('multigrid-covering-v0')

    pos_lst = [np.array([1, 1]), np.array([7, 7]),
               np.array([1, 8]), np.array([3, 1]),
               np.array([1, 1]), np.array([3, 1]), np.array([5, 5])]
    _ = env.reset(pos_lst)

    nb_agents = len(env.agents)
    #print(env.agent_view_size)
    while True:
        env.render(mode='human', highlight=True)
        time.sleep(0.1)

        #ac = [env.action_space.sample() for _ in range(nb_agents)]
        ac = [0 for _ in range(nb_agents)]

        obs, rewards, done, _ = env.step(ac)
        print(rewards)

        if done:
            break

if __name__ == "__main__":
    main()