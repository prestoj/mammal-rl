import time
import numpy as np
import gym
import minerl
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import WorldManager
import gc
import multiprocessing

TEST = 0

action_space = {
    "action_forward": 2,
    "action_back": 2,
    "action_left": 2,
    "action_right": 2,
    "action_jump": 2,
    "action_sneak": 2,
    "action_sprint": 2,
    "action_attack": 2,
    "action_camera_0": 11,
    "action_camera_1": 11,
}

def worker(env_name, queue):
    if TEST:
        device = 'cuda:1'
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = WorldManager(
        image_size=64,
        patch_size=8,
        hidden_dim=768,
        num_heads=12,
        num_layers=12,
        num_vision_encode_layers=12,
        num_vision_decode_layers=4,
        actions_dict=action_space,
        device=device,
    )
    model.load('world.pth', 'vision.pth')
    if TEST:
        model.SARS_dataset = []

    env = gym.make(env_name)
    obs = env.reset()
    done = False
    total_reward = 0
    step_reward = 0
    i = 0
    while not done:
        if TEST:
            env.render()
        i += 1
        if i % 5 != 1:
            obs, reward, done, info = env.step(actions)
            step_reward += reward
            continue
        # get the image from the observation
        image = obs['pov']

        actions = model.get_action_from_environment(image, step_reward)
        
        # remove 'action_' from the keys
        actions = {k.replace('action_', ''): v for k, v in actions.items()}

        # combine camera_0 anSARS_datasetd camera_1 into a single action
        actions['camera'] = [actions['camera_0'] - 5, actions['camera_1'] - 5]

        # take a step in the environment
        obs, reward, done, info = env.step(actions)
        total_reward += step_reward
        step_reward = reward

    # Send rewards or other data back to the main process if needed
    queue.put(total_reward)

    env.close()
    
    if not TEST:
        model.save('world.pth', 'vision.pth')


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
    # model = WorldManager(
    #     image_size=64,
    #     patch_size=8,
    #     hidden_dim=768,
    #     num_heads=12,
    #     num_layers=12,
    #     num_vision_encode_layers=12,
    #     num_vision_decode_layers=4,
    #     actions_dict=action_space,
    #     device='cuda:0' if torch.cuda.is_available() else 'cpu',
    # )

    # model.save('world.pth', 'vision.pth')

    # print('Number of parameters in world model: %d' % sum(p.numel() for p in model.model.parameters() if p.requires_grad))
    # print('Number of parameters in vision model: %d' % sum(p.numel() for p in model.vision_manager.model.parameters() if p.requires_grad))

    main()
