import datetime

import pandas as pd
import plotly.express as px
import os


class Plotting:
    def __init__(self, experiment_name):
        self.results_dir = 'results'
        self.experiment_name = experiment_name
        self.file_suffix = datetime.datetime.now().strftime('%m-%d-%Y-%H%M%S')

        self.save_path = f'{self.results_dir}/{self.experiment_name}'
        os.makedirs(self.save_path, exist_ok=True)

    def plot_train_loss(self, train_loss):
        data = pd.DataFrame({'train loss': train_loss, 'epoch': range(len(train_loss))})
        fig = px.line(data, x='epoch', y='train loss', title='Train loss')
        fig.write_image(f'{self.save_path}/train_loss_{self.file_suffix}.png')

    def plot_train_val_F1(self, train_f1, val_f1):
        train = pd.DataFrame({'f1': train_f1, 'train or val': 'train', 'epoch': range(len(train_f1))})
        val = pd.DataFrame({'f1': val_f1, 'train or val': 'val', 'epoch': range(len(val_f1))})
        data = pd.concat([train, val])
        fig = px.line(data, x='epoch', y='f1', color='train or val', title='Train vs Val F1')
        fig.write_image(f'{self.save_path}/f1_{self.file_suffix}.png')
