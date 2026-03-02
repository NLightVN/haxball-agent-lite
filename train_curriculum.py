import os
import random
import numpy as np
import pprint
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from training.env import HaxballCurriculumEnv

class CurriculumManagerCallback(BaseCallback):
    def __init__(self, envs, save_path, verbose=1):
        super().__init__(verbose)
        self.envs = envs  # The DummyVecEnv holding our HaxballCurriculumEnv instances
        self.save_path = save_path
        self.best_a0_model_path = os.path.join(save_path, "A0_baseline.zip")
        self.phase = 'A0'
        
        # A1 Opponent Pools
        self.previously_trained_opponent = None # Path to latest frozen opponent
        self.better_op_pool = []
        self.worse_op_pool = []
        
        # A1 tracking
        self.a1_episodes = 0
        self.a1_eval_games = 0
        self.a1_eval_wins = 0

    def _on_step(self) -> bool:
        # Sync timesteps to environment for logging (if needed)
        for env in self.envs.envs:
            env.total_timesteps_elapsed = self.num_timesteps
            
        if self.phase == 'A0':
            # Check if we should transition to A1
            if "rollout/ep_rew_mean" in self.logger.name_to_value:
                ep_rew_mean = self.logger.name_to_value.get("rollout/ep_rew_mean", 0)
                if ep_rew_mean >= 2.5:
                    print(f"\\n--- A0 Target Reached (rew_mean={ep_rew_mean:.2f} >= 2.5) at step {self.num_timesteps} ---")
                    self.model.save(self.best_a0_model_path)
                    self.previously_trained_opponent = self.best_a0_model_path
                    self.transition_to_a1()
        else:
            # Phase A1
            # Every 20 episodes, we evaluate against previously_trained_opponent.
            # To do this cleanly within SB3, we count 'dones' (episodes finished)
            dones = self.locals.get("dones", [False])
            # For each done, we check if it was against the 'evaluate' opponent
            for i, d in enumerate(dones):
                if d:
                    self.a1_episodes += 1
                    env = self.envs.envs[i]
                    if getattr(env, 'is_eval_game', False):
                        self.a1_eval_games += 1
                        # If agent (team_id) won (score >= 3)
                        agent_idx = env.team_id - 1
                        if env.scores[agent_idx] >= 3:
                            self.a1_eval_wins += 1
                            
            # If 20 eval games hit, process pool update
            if self.a1_eval_games >= 20:
                win_rate = self.a1_eval_wins / self.a1_eval_games
                print(f"\\n--- Evaluated against previous_trained: Win Rate = {win_rate:.2f} ---")
                self.logger.record("curriculum/a1_win_rate", win_rate)
                
                # Freeze current model to become a new trained opponent
                latest_path = os.path.join(self.save_path, f"trained_op_{self.num_timesteps}.zip")
                self.model.save(latest_path)
                
                # Update pools and determine sampling pool for the NEXT 20 games
                if win_rate >= 0.6:
                    self.better_op_pool.append(latest_path)
                    self.current_sampling_pool = [self.previously_trained_opponent] + self.better_op_pool
                elif win_rate < 0.4:
                    self.worse_op_pool.append(latest_path)
                    if len(self.worse_op_pool) == 0:
                        self.current_sampling_pool = [] # Empty means use scripted later
                    else:
                        self.current_sampling_pool = [self.previously_trained_opponent] + self.worse_op_pool
                else: # 0.4 <= win_rate < 0.6
                    self.better_op_pool.append(latest_path) # Neutral can go to better as baseline
                    self.current_sampling_pool = [self.previously_trained_opponent] + self.better_op_pool + self.worse_op_pool
                    
                self.previously_trained_opponent = latest_path
                self.a1_eval_games = 0
                self.a1_eval_wins = 0

            # Assign new random opponents to any recently `done` envs
            for i, d in enumerate(dones):
                if d:
                    env = self.envs.envs[i]
                    self._assign_a1_opponent(env)

        return True

    def transition_to_a1(self):
        print("Transitioning environments to A1 phase...")
        self.phase = 'A1'
        self.a1_episodes = 0
        self.a1_start_timestep = self.num_timesteps
        self.current_sampling_pool = [self.previously_trained_opponent]
        
        for env in self.envs.envs:
            env.phase = 'A1'
            self._assign_a1_opponent(env)
            env.reset()

    def _assign_a1_opponent(self, env):
        # Time threshold sampling ratios: [Follower, Defender, Trained]
        # We define steps relative to transition point
        t = self.num_timesteps - self.a1_start_timestep
        if t < 500_000:
            probs = [0.25, 0.25, 0.50]
        elif t < 1_000_000:
            probs = [0.15, 0.15, 0.70]
        else:
            probs = [0.10, 0.10, 0.80]
            
        # Is this an eval game? Designate 1 in every 5 games as an eval game tracking previous_trained_opponent
        if random.random() < 0.2:
            env.is_eval_game = True
            env.opponent_type = 'Trained'
            env.opponent_policy = PPO.load(self.previously_trained_opponent, device='cpu')
            return
            
        env.is_eval_game = False
        choice = random.choices(['Follower', 'Defender', 'Trained'], weights=probs, k=1)[0]
        
        # Override to Scripted if we triggered the < 0.4 win_rate AND empty worse_op_pool
        if choice == 'Trained' and len(self.current_sampling_pool) == 0:
            choice = random.choices(['Follower', 'Defender'], weights=[0.5, 0.5], k=1)[0]
            
        env.opponent_type = choice
        
        if choice == 'Trained':
            # Sample from the actively designated pool from the last 20 evaluations
            chosen_model = random.choice(self.current_sampling_pool)
            env.opponent_policy = PPO.load(chosen_model, device='cpu')


if __name__ == "__main__":
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
    
    # Create 4 parallel environments
    def make_env():
        return Monitor(HaxballCurriculumEnv(phase='A0'))
        
    vec_env = DummyVecEnv([make_env for _ in range(4)])
    
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./ppo_curriculum_tensorboard/")
    
    os.makedirs("./models", exist_ok=True)
    
    # Save a checkpoint every 100,000 total steps (100k / 4 envs = 25,000 per env)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // 4, 1),
        save_path='./models/checkpoints/',
        name_prefix='rl_model'
    )
    
    curriculum_callback = CurriculumManagerCallback(vec_env, save_path="./models")
    
    # Combine the custom curriculum callback with the checkpoint callback
    callback = CallbackList([checkpoint_callback, curriculum_callback])
    
    print("Starting Training...")
    model.learn(total_timesteps=3_000_000, callback=callback)
    model.save("models/final_curriculum_model")
