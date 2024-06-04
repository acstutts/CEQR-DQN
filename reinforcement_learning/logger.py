

import pickle
import pprint
import matplotlib.pyplot as plt
import numpy as np

import torch
from datetime import datetime
import os
import pandas as pd

class Logger:
    def __init__(self,log_folder_details=None,train_details=None):
        self.memory = {}
        self.log_folder_details = log_folder_details
        self.train_details = train_details

        today = datetime.now()
        if self.log_folder_details is None:
            directory = 'results/'+today.strftime('%Y-%m-%d-%H%M%S')
        else:
            directory = 'results/'+today.strftime('%Y-%m-%d-%H%M%S') + '-' + self.log_folder_details
        
        os.mkdir(directory)
        self.log_folder = directory

        with open(self.log_folder + '/' + 'experimental-setup', 'w') as handle:
                pprint.pprint(self.train_details, handle)

    def add_scalar(self, name, data, timestep):
        """
        Saves a scalar
        """
        if isinstance(data, torch.Tensor):
            data = data.item()

        self.memory.setdefault(name, []).append([data, timestep])

    def save(self):
        filename = self.log_folder + '/log_data.pkl'
        
        with open(filename, 'wb') as output:
            pickle.dump(self.memory, output, pickle.HIGHEST_PROTOCOL)

        self.save_graphs()

    def save_graphs(self):
        for key in self.memory.keys():
            fig, ax = plt.subplots()
            series = pd.Series(np.array(self.memory[key])[:,0])
            moving_avg = series.rolling(window=100).mean()
            ax.scatter(np.array(self.memory[key])[:,1],np.array(self.memory[key])[:,0])
            ax2 = ax.twinx()
            ax2.plot(np.array(self.memory[key])[:,1],moving_avg.to_numpy(),color='r')
            plt.grid
            if self.log_folder is None:
                plt.savefig(key+'.png')
                plt.close()
            else:
                plt.savefig(self.log_folder + '/' + key+'.png')
                plt.close()

