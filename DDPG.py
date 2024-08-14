import gym
import gym_carla
import carla
from stable_baselines3 import DDPG
from stable_baselines3.ddpg import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os

def main():
    # Parameters for the CARLA environment
    params = {
        'number_of_vehicles': 1,
        'number_of_walkers': 0,
        'display_size': 256,
        'max_past_step': 1,
        'dt': 0.1,
        'discrete': False,
        'discrete_acc': [-3.0, 0.0, 3.0],
        'discrete_steer': [-0.2, 0.0, 0.2],
        'continuous_accel_range': [-3.0, 3.0],
        'continuous_steer_range': [-0.3, 0.3],
        'ego_vehicle_filter': 'vehicle.lincoln*',
        'port': 4000,
        'town': 'Town03',
        'max_time_episode': 1000,
        'max_waypt': 12,
        'obs_range': 32,
        'lidar_bin': 0.125,
        'd_behind': 12,
        'out_lane_thres': 2.0,
        'desired_speed': 8,
        'max_ego_spawn_times': 200,
        'display_route': True,
    }

    # The CARLA environment
    env = gym.make('carla-v0', params=params)

    log_dir = "./ddpg_carla/"
    os.makedirs(log_dir, exist_ok=True)
    #checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix='ddpg_carla_model')
    eval_callback = EvalCallback(env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=500, deterministic=True, render=False)

    model = DDPG(MlpPolicy, env, verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=10000, callback=[checkpoint_callback, eval_callback])

    model.save(os.path.join(log_dir, "ddpg_carla_model"))

    obs = env.reset()
    i = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        print(i, rewards)
        i += 1
        if dones:
            break

if __name__ == '__main__':
    main()
