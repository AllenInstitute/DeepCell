import pandas as pd
import plotly.express as px


def plot_loss(train_loss, val_loss):
    train = pd.DataFrame({'loss': train_loss, 'train or val': 'train', 'epoch': range(len(train_loss))})
    val = pd.DataFrame({'loss': val_loss, 'train or val': 'val', 'epoch': range(len(val_loss))})
    data = pd.concat([train, val])
    fig = px.line(data, x='epoch', y='loss', color='train or val', title='Loss')
    return fig
