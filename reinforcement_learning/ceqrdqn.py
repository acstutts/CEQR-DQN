import random
import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import pprint as pprint

from replay_buffer import ReplayBuffer
from logger import Logger
from utils import set_global_seed
from utils import loss_fn, gamma_cal_loss
from quantilelosses import loss_evi

class textcolors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    DEFAULT = "\033[37m"

class CEQRDQN():
    def __init__(
        self,
        env,
        network,
        n_quantiles=100,
        obs_space_channel=4,
        kappa=1,
        replay_start_size=50000,
        replay_buffer_size=1000000,
        gamma=0.99,
        update_target_frequency=10000,
        minibatch_size=32,
        epsilon_12=0.00001,
        learning_rate=1e-4,
        update_frequency=1,
        adam_epsilon=1e-8,
        evi_coeff=1.0, 
        unc_lamda_el=0.1,
        unc_lamda_al=0, 
        logging=False,
        log_folder_details=None,
        seed=None,
        notes=None
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.replay_start_size = replay_start_size
        self.replay_buffer_size = replay_buffer_size
        self.gamma = gamma
        self.epsilon_12 = epsilon_12
        self.update_target_frequency = update_target_frequency
        self.minibatch_size = minibatch_size
        self.learning_rate = learning_rate
        self.update_frequency = update_frequency
        self.adam_epsilon = adam_epsilon
        self.logging = logging
        self.evi_coeff = evi_coeff
        self.unc_lamda_el = unc_lamda_el
        self.unc_lamda_al = unc_lamda_al
        self.obs_space_channel = obs_space_channel
        self.logger = []
        self.timestep=0
        self.log_folder_details = log_folder_details

        self.env = env
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.seed = random.randint(0, 2e8) if seed is None else seed
        self.logger=None

        set_global_seed(self.seed, self.env)

        self.n_quantiles = n_quantiles

        self.network = network(self.device,self.env.observation_space, self.n_quantiles, self.env.action_space.n*self.n_quantiles,self.obs_space_channel).to(self.device)
        self.target_network = network(self.device,self.env.observation_space, self.n_quantiles, self.env.action_space.n*self.n_quantiles,self.obs_space_channel).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, eps=self.adam_epsilon)

        self.loss = loss_fn
        self.kappa = kappa

        self.train_parameters = {'Notes':notes,
                'env':env.unwrapped.spec.id,
                'network':str(self.network),
                'replay_start_size':replay_start_size,
                'replay_buffer_size':replay_buffer_size,
                'gamma':gamma,
                'update_target_frequency':update_target_frequency,
                'minibatch_size':minibatch_size,
                'learning_rate':learning_rate,
                'update_frequency':update_frequency,
                'kappa':kappa,
                'n_quantiles':n_quantiles,
                'weight_scale':self.network.weight_scale,
                'evidence coeff':evi_coeff,
                'unc_lamda_el':unc_lamda_el,
                'unc_lamda_al':unc_lamda_al,
                'obs_space_channel':obs_space_channel,
                'adam_epsilon':adam_epsilon,
                'seed':self.seed}

        self.n_greedy_actions = 0

    def learn(self, timesteps, verbose=False):

        self.train_parameters['train_steps'] = timesteps
        pprint.pprint(self.train_parameters)

        if self.logging:
            self.logger = Logger(self.log_folder_details,self.train_parameters)

        # Initialize the state
        obs, info = self.env.reset(seed=self.seed)
        state = torch.as_tensor(np.array(obs))
        this_episode_time = 0
        score = 0
        t1 = time.time()

        actions_uncertain = 0
        actions_greedy = 0
        max_score = 0

        with open(self.logger.log_folder + '/output_log.txt',"w") as f:

            for timestep in range(timesteps):

                is_training_ready = timestep >= self.replay_start_size

                # Select action
                action, least_uncertain, greedy = self.act(state.to(self.device).float())
                actions_uncertain += least_uncertain
                actions_greedy += greedy

                # Perform action in environments
                state_next, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Store transition in replay buffer
                action = torch.as_tensor([action], dtype=torch.long)
                reward = torch.as_tensor([reward], dtype=torch.float)
                done = torch.as_tensor([done], dtype=torch.float)
                state_next = torch.as_tensor(np.array(state_next))
                self.replay_buffer.add(state, action, reward, state_next, done)

                score += reward.item()
                this_episode_time += 1

                if done:
        
                    if score > max_score:
                        textcolor = textcolors.BLUE
                        save_path = self.logger.log_folder + '/max_score.pth'
                        torch.save(self.network.state_dict(), save_path)
                        max_score = score
                    else:
                        textcolor = textcolors.DEFAULT

                    if verbose:
                        log_statement = textcolor + "Timestep: {}, score: {}, Time: {} s, actions: {}, ALU: {} ({}%), AG: {} ({}%)".format(timestep, score, round(time.time() - t1, 3),
                                                                                                                                    this_episode_time,
                                                                                                                                    actions_uncertain,
                                                                                                                                    round((actions_uncertain/this_episode_time)*100,2),
                                                                                                                                    actions_greedy,
                                                                                                                                    round((actions_greedy/this_episode_time)*100,2),
                                                                                                                                    )
                        print(log_statement)
                        f.write(log_statement.removeprefix(textcolor) + "\n")
                    if self.logging:
                        self.logger.add_scalar('Episode_score', score, timestep)
                        self.logger.add_scalar('Non_greedy_fraction', 1-self.n_greedy_actions/this_episode_time, timestep)
                    obs, info = self.env.reset(seed=self.seed)
                    state = torch.as_tensor(np.array(obs))
                    score = 0
                    if self.logging:
                        self.logger.add_scalar('Q_at_start', self.get_max_q(state.to(self.device).float()), timestep)
                    t1 = time.time()
                    self.n_greedy_actions = 0
                    actions_uncertain = 0
                    actions_greedy = 0
                    this_episode_time = 0
                else:
                    state = state_next

                if is_training_ready:

                    # Update main network
                    if timestep % self.update_frequency == 0:

                        # Sample a batch of transitions
                        transitions = self.replay_buffer.sample(self.minibatch_size, self.device)

                        # Train on selected batch
                        loss = self.train_step(transitions)
                        if done:
                            pass
                        if self.logging and timesteps < 1000000:
                            self.logger.add_scalar('Loss', loss, timestep)

                    # Update target Q
                    if timestep % self.update_target_frequency == 0:
                        self.target_network.load_state_dict(self.network.state_dict())

                if (timestep+1) % 250000 == 0:
                    self.save(timestep=timestep+1)
                    self.logger.save()

                self.timestep=timestep

            if self.logging:
                self.logger.save()
                self.save()
        
    def train_step(self, transitions):
        states, actions, rewards, states_next, dones = transitions

        with torch.no_grad():
            target1,_ = self.target_network(states_next.float())
            target1 = target1.view(self.minibatch_size,self.env.action_space.n,self.n_quantiles)

        best_action_idx = torch.mean(target1,dim=2).max(1, True)[1].unsqueeze(2)
        q_value_target = target1.gather(1, best_action_idx.repeat(1,1,self.n_quantiles))

        # Calculate TD target
        td_target = (rewards.unsqueeze(2).repeat(1,1,self.n_quantiles) + (1 - dones.unsqueeze(2).repeat(1,1,self.n_quantiles)) * self.gamma * q_value_target).squeeze()

        out, evi = self.network(states.float())
        out = out.view(self.minibatch_size,self.env.action_space.n,self.n_quantiles) 

        gamma, v, alpha, beta = torch.split(evi, int(self.env.action_space.n*2*self.n_quantiles), dim=-1)
        gamma = gamma.view(self.minibatch_size, self.env.action_space.n, 2, self.n_quantiles).gather(1, actions.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2, self.n_quantiles)).squeeze()
        v = v.view(self.minibatch_size, self.env.action_space.n, 2, self.n_quantiles).gather(1, actions.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2, self.n_quantiles)).squeeze()
        alpha = alpha.view(self.minibatch_size, self.env.action_space.n, 2, self.n_quantiles).gather(1, actions.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2, self.n_quantiles)).squeeze()
        beta = beta.view(self.minibatch_size, self.env.action_space.n, 2, self.n_quantiles).gather(1, actions.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2, self.n_quantiles)).squeeze()

        q_value = out.gather(1, actions.unsqueeze(2).repeat(1,1,self.n_quantiles)).squeeze()

        quantile_losses = self.loss(q_value, td_target, 0.5, self.kappa, self.device)
        loss_evidence = loss_evi(td_target, gamma, v, alpha, beta, self.evi_coeff)
        loss_gamma_cal = gamma_cal_loss(gamma,td_target,0.5,self.device)

        loss = quantile_losses + loss_evidence + loss_gamma_cal

        # Update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def act(self, state):

        action, least_uncertain, greedy = self.predict(state)
        
        return action, least_uncertain, greedy

    @torch.no_grad()
    def predict(self, state):
            
        out,evi = self.network(state)
        out = out.view(self.env.action_space.n,self.n_quantiles)
        action_means = torch.mean(out,dim=1)

        gamma, v, alpha, beta = torch.split(evi, int(self.env.action_space.n*2*self.n_quantiles), dim=-1)
        gamma = gamma.view(self.env.action_space.n,2,self.n_quantiles)
        v = v.view(self.env.action_space.n,2,self.n_quantiles)
        alpha = alpha.view(self.env.action_space.n,2,self.n_quantiles)
        beta = beta.view(self.env.action_space.n,2,self.n_quantiles)

        var = torch.sqrt((beta /(v*(alpha - 1))))

        global_aleatoric = torch.mean(torch.abs(gamma[:,1,:] - gamma[:,0,:]),dim=1)
        global_epistemic = torch.mean(0.5*(torch.abs((gamma[:,0,:]+var[:,0,:])-(gamma[:,0,:]-var[:,0,:])) + torch.abs((gamma[:,1,:]+var[:,1,:])-(gamma[:,1,:]-var[:,1,:]))),dim=1) + 1e-8

        #action_means -= self.unc_lamda_al*(global_aleatoric)
        action_uncertainties_cov = self.unc_lamda_el*torch.diagflat(global_epistemic)
        
        samples = torch.distributions.multivariate_normal.MultivariateNormal(action_means,covariance_matrix=action_uncertainties_cov).sample()
        action_uncertain = (action_means - self.unc_lamda_el*global_epistemic).argmax().item()
        action = samples.argmax().item()

        if action == action_means.argmax().item():
            greedy = 1
            self.n_greedy_actions += 1
        else:
            greedy = 0

        if action_uncertain == action:
            least_uncertain = 1
        else:
            least_uncertain = 0

        return action, least_uncertain, greedy

    @torch.no_grad()
    def get_max_q(self,state):
        net1,_ = self.network(state)
        net1 = net1.view(self.env.action_space.n,self.n_quantiles)
        action_means = torch.mean(net1,dim=1)
        max_q = action_means.max().item()
        return max_q


    def save(self,timestep=None):
        if not self.logging:
            raise NotImplementedError('Cannot save without log folder.')

        if timestep is not None:
            filename = 'network_' + str(timestep) + '.pth'
        else:
            filename = 'network.pth'

        save_path = self.logger.log_folder + '/' + filename

        torch.save(self.network.state_dict(), save_path)

    def load(self,path):
        self.network.load_state_dict(torch.load(path,map_location='cpu'))
        self.target_network.load_state_dict(torch.load(path,map_location='cpu'))