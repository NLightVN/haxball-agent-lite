import os, torch
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from training.env import HaxballCurriculumEnv
from training.train_a1 import OpponentManager, SelfPlayEnv
import numpy as np

def run():
    model_path = "models/experiment/a1_checkpoints/snapshot_1000000.zip"
    
    opponent_manager = OpponentManager(None)
    opponent_manager.add_opponent("snapshot_1000000", model_path)
    opponent_manager.agent_idx = 0
    opponent_manager.agent_pool = ["snapshot_1000000"]
    def patched_sample_opponent():
        return opponent_manager.opponents["snapshot_1000000"], "snapshot_1000000"
    opponent_manager.sample_opponent = patched_sample_opponent

    def make_env():
        return SelfPlayEnv(opponent_manager=opponent_manager, phase='A1')

    vec_env = DummyVecEnv([make_env for _ in range(8)])
    model = PPO.load(model_path, env=vec_env, device='cpu')
    
    obs = vec_env.reset()
    wins = losses = draws = 0
    print("Simulating matches using EXACT PPO sampling...")
    
    while wins + losses + draws < 200:
        # PPO sampling EXACTLY as in model.learn()
        obs_tensor = torch.as_tensor(obs).float()
        with torch.no_grad():
            distribution = model.policy.get_distribution(obs_tensor)
            actions = distribution.sample()
        action = actions.cpu().numpy()
        
        obs, rewards, dones, infos = vec_env.step(action)
        
        for i, done in enumerate(dones):
            if done:
                env_i = vec_env.envs[i]
                red_score = env_i.scores[0]
                blue_score = env_i.scores[1]
                team_id = env_i.team_id
                
                agent_score = red_score if team_id == 1 else blue_score
                opp_score = blue_score if team_id == 1 else red_score
                
                if agent_score > opp_score: wins += 1
                elif opp_score > agent_score: losses += 1
                else: draws += 1
                    
                if (wins + losses + draws) % 20 == 0:
                    print(f"Matches: {wins+losses+draws}/200 | A: {wins} | B: {losses} | D: {draws}")

    total = wins + losses + draws
    print(f"\nFinal: A={wins} ({wins/total*100:.1f}%) | B={losses} ({losses/total*100:.1f}%) | D={draws}")

if __name__ == "__main__":
    run()
