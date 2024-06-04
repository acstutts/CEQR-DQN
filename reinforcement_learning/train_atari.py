from models import CNN_UADQN, CNN_CEQRDQN
from ceqrdqn import CEQRDQN
from uadqn import UADQN
import gymnasium as gym
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env_ids = ['MinAtar/Asterix-v1','MinAtar/Breakout-v1','MinAtar/Freeway-v1','MinAtar/Seaquest-v1','MinAtar/SpaceInvaders-v1']
folders = ["MinAtar-Asterix-CEQRDQN","MinAtar-Breakout-CEQRDQN","MinAtar-Freeway-CEQRDQN","MinAtar-Seaquest-CEQRDQN","MinAtar-SpaceInvaders-CEQRDQN"]
obs_space_chs = [4,4,7,10,6]
uncertainty_lamdas_el = [0.001,0.01,0.0002,0.005,0.0005]
steps = [2500000,2500000,2500000,2500000,2500000]
evi_coeffs = [0.5,0.5,0.5,0.5,0.5]

for i, env_id in enumerate(env_ids):
    env = gym.make(env_id)
    env.game.sticky_action_prob = 0.0
    notes = "CEQRDQN"
    currentfolder = folders[i]
    obs_space_ch = obs_space_chs[i]
    unc_lamda_el = uncertainty_lamdas_el[i]
    unc_lamda_al = 0
    num_steps = steps[i]
    evi_coeff = evi_coeffs[i]
    agent = CEQRDQN(env,
                   CNN_CEQRDQN,
                   n_quantiles=50,
                   obs_space_channel = obs_space_ch,
                   kappa=1,
                   replay_start_size=5000,
                   replay_buffer_size=100000,
                   gamma=0.99,
                   update_target_frequency=1000,
                   minibatch_size=32,
                   learning_rate=1e-4,
                   adam_epsilon=1e-8,
                   update_frequency=1,
                   evi_coeff=evi_coeff,
                   unc_lamda_el = unc_lamda_el,
                   unc_lamda_al = unc_lamda_al,
                   log_folder_details=currentfolder,
                   seed=None,
                   logging=True,
                   notes=notes)

    agent.learn(timesteps=num_steps, verbose=True)
    agent.save()

folders = ["MinAtar-Asterix-UADQN","MinAtar-Breakout-UADQN","MinAtar-Freeway-UADQN","MinAtar-Seaquest-UADQN","MinAtar-SpaceInvaders-UADQN"]
for i, env_id in enumerate(env_ids):
    env = gym.make(env_id)
    env.game.sticky_action_prob = 0.0
    notes = "UADQN"
    currentfolder = folders[i]
    obs_space_ch = obs_space_chs[i]
    num_steps = steps[i]
    agent = UADQN(
            env,
            CNN_UADQN,
            n_quantiles=50,
            weight_scale=3,
            noise_scale=1,
            epistemic_factor=0.2,
            aleatoric_factor=0,
            kappa=1,
            replay_start_size=5000,
            replay_buffer_size=100000,
            gamma=0.99,
            update_target_frequency=1000,
            minibatch_size=32,
            learning_rate=1e-4,
            adam_epsilon=1e-8,
            log_folder_details=currentfolder,
            update_frequency=1,
            obs_space_channel= obs_space_ch,
            logging=True,
            notes=notes,
            seed=None
            )

    agent.learn(timesteps=num_steps, verbose=True)
    agent.save()
