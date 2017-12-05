import sys

if '/home/jeremy/progressive_transfer' not in sys.path:
    sys.path.append('/home/jeremy/progressive_transfer')

import gym
import deepq
from lib.env.mountain_car import MountainCarEnv
from lib.env.threedmountain_car import ThreeDMountainCarEnv

def main():
    env = MountainCarEnv()
    env_transfer = ThreeDMountainCarEnv()
    # Enabling layer_norm here is import for parameter space noise!
    # model = deepq.models.mlp([64], layer_norm=True)
    model = deepq.models.prog_nn([64], layer_norm=False)

    act = deepq.learn(
        env,
        env_transfer,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        print_freq=1,
        param_noise=False
    )
    print("Saving model to mountaincar_model.pkl")
    act.save("mountaincar_model.pkl")


if __name__ == '__main__':
    main()
