import time
import numpy as np
import gym
import minerl
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import VisionManager
import gc
import multiprocessing

action_space = {
    "forward": 2,
    "back": 2,
    "left": 2,
    "right": 2,
    "jump": 2,
    "sneak": 2,
    "sprint": 2,
    "attack": 2,
    "camera_0": 11,
    "camera_1": 11,
}

def worker(env_name, queue):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    vision = VisionManager(
        image_size=64,
        patch_size=8,
        hidden_dim=384,
        num_heads=6,
        num_layers_encode=6,
        num_layers_decode=6,
        device=device,
        max_lr=1e-4,
        batch_size=32,
        warmup_steps=10000,
        total_steps=100000,
        ema_decay=0.999
    )
    vision.load('vision.pth')

    env = gym.make(env_name)
    obs = env.reset()
    done = False
    total_reward = 0
    step_reward = 0
    i = 0
    while not done:
        i += 1
        if i % 5 != 1:
            obs, reward, done, info = env.step(actions)
            step_reward += reward
            continue
        # get the image from the observation
        image = obs['pov']

        vision.input_image(image)

        # random action
        actions = {}
        for key in action_space:
            actions[key] = np.random.randint(action_space[key])
            if key == 'camera_0' or key == 'camera_1':
                actions[key] = np.random.randint(-5, 6)

        # combine camera_0 and camera_1 into a single action
        actions['camera'] = [actions['camera_0'], actions['camera_1']]

        # take a step in the environment
        obs, reward, done, info = env.step(actions)
        total_reward += step_reward
        step_reward = reward

    # Send rewards or other data back to the main process if needed
    queue.put(total_reward)

    env.close()
    
    vision.save('vision.pth')


def main():
    n_games = 0
    all_rewards = []
    while True:
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=worker, args=("MineRLTreechop-v0", queue))
        p.start()
        p.join()  # Wait for the process to complete

        # Retrieve data from the queue if needed
        total_reward = queue.get()
        print(n_games, total_reward)

        all_rewards.append(total_reward)
        print(all_rewards)

        n_games += 1
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    # vision = VisionManager(
    #     image_size=64,
    #     patch_size=8,
    #     hidden_dim=384,
    #     num_heads=6,
    #     num_layers_encode=6,
    #     num_layers_decode=6,
    #     device='cuda:0',
    #     max_lr=1e-4,
    #     batch_size=32,
    #     warmup_steps=10000,
    #     total_steps=100000,
    #     ema_decay=0.999
    # )

    # vision.save('vision.pth')

    main()
