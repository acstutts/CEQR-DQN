

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
from utils import quantile_huber_loss

class textcolors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    DEFAULT = "\033[37m"

class UADQN():
    def __init__(
        self,
        env,
        network,
        gamma=0.99,
        replay_start_size=50000,
        replay_buffer_size=1000000,
        n_quantiles=50,
        kappa=1,
        weight_scale=3,
        noise_scale=0.1,
        epistemic_factor=1,
        aleatoric_factor=1,
        update_target_frequency=10000,
        minibatch_size=32,
        update_frequency=1,
        learning_rate=1e-3,
        seed=None,
        adam_epsilon=1e-8,
        biased_aleatoric=False,
        logging=False,
        log_folder_details=None,
        save_period=250000,
        obs_space_channel =4,
        notes=None,
        render=False,
    ):

        # Agent parameters
        self.env = env
        self.gamma = gamma
        self.replay_start_size = replay_start_size
        self.replay_buffer_size = replay_buffer_size
        self.n_quantiles = n_quantiles
        self.kappa = kappa
        self.weight_scale = weight_scale
        self.noise_scale = noise_scale
        self.epistemic_factor = epistemic_factor,
        self.aleatoric_factor = aleatoric_factor,
        self.update_target_frequency = update_target_frequency
        self.minibatch_size = minibatch_size
        self.update_frequency = update_frequency
        self.learning_rate = learning_rate
        self.seed = random.randint(0, 1e6) if seed is None else seed
        self.adam_epsilon = adam_epsilon
        self.biased_aleatoric = biased_aleatoric
        self.logging = logging
        self.log_folder_details = log_folder_details
        self.save_period = save_period
        self.render = render
        self.notes = notes
        self.obs_space_ch = obs_space_channel

        # Set global seed before creating network
        set_global_seed(self.seed, self.env)

        # Initialize agent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = None
        self.loss = quantile_huber_loss
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        # Initialize main Q learning network
        n_outputs = self.env.action_space.n*self.n_quantiles
        self.network = network(self.env.observation_space, n_outputs,self.obs_space_ch).to(self.device)
        self.target_network = network(self.env.observation_space, n_outputs,self.obs_space_ch).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        # Initialize anchored networks
        self.posterior1 = network(self.env.observation_space, n_outputs, self.obs_space_ch,weight_scale=weight_scale).to(self.device)
        self.posterior2 = network(self.env.observation_space, n_outputs, self.obs_space_ch,weight_scale=weight_scale).to(self.device)
        self.anchor1 = [p.data.clone() for p in list(self.posterior1.parameters())]
        self.anchor2 = [p.data.clone() for p in list(self.posterior2.parameters())]

        # Initialize optimizer
        params = list(self.network.parameters()) + list(self.posterior1.parameters()) + list(self.posterior2.parameters())
        self.optimizer = optim.Adam(params, lr=self.learning_rate, eps=self.adam_epsilon)

        # Figure out what the scale of the prior is from empirical std of network weights
        with torch.no_grad():
            std_list = []
            for i, p in enumerate(self.posterior1.parameters()):
                std_list.append(torch.std(p))
        self.prior_scale = torch.stack(std_list).mean().item()
        
        # Parameters to save to log file
        self.train_parameters = {
                    'Notes': notes,
                    'env': env.unwrapped.spec.id,
                    'network': str(self.network),
                    'n_quantiles': n_quantiles,
                    'replay_start_size': replay_start_size,
                    'replay_buffer_size': replay_buffer_size,
                    'gamma': gamma,
                    'update_target_frequency': update_target_frequency,
                    'minibatch_size': minibatch_size,
                    'learning_rate': learning_rate,
                    'update_frequency': update_frequency,
                    'weight_scale': weight_scale,
                    'noise_scale': noise_scale,
                    'epistemic_factor': epistemic_factor,
                    'aleatoric_factor': aleatoric_factor,
                    'biased_aleatoric': biased_aleatoric,
                    'adam_epsilon': adam_epsilon,
                    'seed': self.seed
                    }


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
                action, ep, al, least_uncertain, greedy = self.act(state.to(self.device).float(), thompson_sampling=True)
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

        # Calculate target Q
        with torch.no_grad():
            target = self.target_network(states_next.float())
            target = target.view(self.minibatch_size, self.env.action_space.n, self.n_quantiles)

        # Calculate max of target Q values
        best_action_idx = torch.mean(target, dim=2).max(1, True)[1].unsqueeze(2)
        q_value_target = target.gather(1, best_action_idx.repeat(1, 1, self.n_quantiles))

        # Calculate TD target
        rewards = rewards.unsqueeze(2).repeat(1, 1, self.n_quantiles)
        dones = dones.unsqueeze(2).repeat(1, 1, self.n_quantiles)
        td_target = rewards + (1 - dones) * self.gamma * q_value_target

        # Calculate Q value of actions played
        outputs = self.network(states.float())
        outputs = outputs.view(self.minibatch_size, self.env.action_space.n, self.n_quantiles)
        actions = actions.unsqueeze(2).repeat(1, 1, self.n_quantiles)
        q_value = outputs.gather(1, actions)

        # TD loss for main network
        loss = self.loss(q_value.squeeze(), td_target.squeeze(), self.device, kappa=self.kappa)

        # Calculate predictions of posterior networks
        posterior1 = self.posterior1(states.float())
        posterior1 = posterior1.view(self.minibatch_size, self.env.action_space.n, self.n_quantiles)
        posterior1 = posterior1.gather(1, actions)

        posterior2 = self.posterior2(states.float())
        posterior2 = posterior2.view(self.minibatch_size, self.env.action_space.n, self.n_quantiles)
        posterior2 = posterior2.gather(1, actions)

        # Regression loss for the posterior networks
        loss_posterior1 = self.loss(posterior1.squeeze(), td_target.squeeze(), self.device, kappa=self.kappa)
        loss_posterior2 = self.loss(posterior2.squeeze(), td_target.squeeze(), self.device, kappa=self.kappa)
        loss += loss_posterior1 + loss_posterior2

        diff1=[]
        for i, p in enumerate(self.posterior1.parameters()):
            diff1.append(torch.sum((p - self.anchor1[i])**2))

        diff2=[]
        for i, p in enumerate(self.posterior2.parameters()):
            diff2.append(torch.sum((p - self.anchor2[i])**2))

        diff1 = torch.stack(diff1).sum()
        diff2 = torch.stack(diff2).sum()
        diff = diff1 + diff2

        num_data = np.min([self.timestep, self.replay_buffer_size])
        anchor_loss = self.noise_scale**2*diff/(self.prior_scale**2*num_data)

        loss += anchor_loss

        # Update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def act(self, state, thompson_sampling=False):

        action, least_uncertain, greedy = self.predict(state,thompson_sampling=thompson_sampling)
        
        return action, least_uncertain, greedy

    @torch.no_grad()
    def predict(self, state):
        """
        Returns action to be performed using Thompson sampling
        with estimates provided by the two posterior networks
        """

        net = self.network(state).view(self.env.action_space.n, self.n_quantiles)

        posterior1 = self.posterior1(state).view(self.env.action_space.n, self.n_quantiles)
        posterior2 = self.posterior2(state).view(self.env.action_space.n, self.n_quantiles)

        mean_action_values = torch.mean(net, dim=1)

        # Calculate aleatoric uncertainty
        if self.biased_aleatoric:
            uncertainties_aleatoric = torch.std(net, dim=1)
        else:
            covariance = torch.mean((posterior1-torch.mean(posterior1))*(posterior2-torch.mean(posterior2)), dim=1)
            uncertainties_aleatoric = torch.sqrt(F.relu(covariance))

        # Aleatoric-adjusted Q values
        aleatoric_factor = torch.FloatTensor(self.aleatoric_factor).to(self.device)
        adjusted_action_values = mean_action_values - aleatoric_factor*uncertainties_aleatoric

        # Calculate epistemic uncertainty
        uncertainties_epistemic = torch.mean((posterior1-posterior2)**2, dim=1)/2 + 1e-8
        epistemic_factor = torch.FloatTensor(self.epistemic_factor).to(self.device)**2
        uncertainties_cov = epistemic_factor*torch.diagflat(uncertainties_epistemic)

        # Draw samples using Thompson sampling
        epistemic_distrib = torch.distributions.multivariate_normal.MultivariateNormal
        samples = epistemic_distrib(adjusted_action_values, covariance_matrix=uncertainties_cov).sample()
        action = samples.argmax().item()
        action_uncertain = (mean_action_values - uncertainties_epistemic*epistemic_factor).argmax().item()
        
        
        if action == mean_action_values.argmax().item():
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
        net = self.network(state).view(self.env.action_space.n, self.n_quantiles)
        action_means = torch.mean(net, dim=1)
        q = action_means.max().item()
        return q

    def save(self,timestep=None):
        """
        Saves network weights
        """
        if timestep is not None:
            filename = 'network_' + str(timestep) + '.pth'
            filename_posterior1 = 'network_posterior1_' + str(timestep) + '.pth'
            filename_posterior2 = 'network_posterior2_' + str(timestep) + '.pth'
        else:
            filename = 'network.pth'
            filename_posterior1 = 'network_posterior1.pth'
            filename_posterior2 = 'network_posterior2.pth'

        save_path = self.logger.log_folder + '/' + filename
        save_path_posterior1 = self.logger.log_folder + '/' + filename_posterior1
        save_path_posterior2 = self.logger.log_folder + '/' + filename_posterior2

        torch.save(self.network.state_dict(), save_path)
        torch.save(self.posterior1.state_dict(), save_path_posterior1)
        torch.save(self.posterior2.state_dict(), save_path_posterior2)

    def load(self,path):
        """
        Loads network weights
        """
        self.network.load_state_dict(torch.load(path + 'network.pth', map_location='cpu'))
        self.posterior1.load_state_dict(torch.load(path + 'network_posterior1.pth', map_location='cpu'))
        self.posterior2.load_state_dict(torch.load(path + 'network_posterior2.pth', map_location='cpu'))