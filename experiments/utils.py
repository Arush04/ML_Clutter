import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import math
import matplotlib.pyplot as plt

def l2_penalty(w):
    return (w ** 2).sum() / 2

class Data(Dataset):
    """
    Custom Dataset class for our sythetic data
    Generator formula:
    y = 0.05 + ∑(i->n)(0.01xi) + e
    where e is gaussian noise
    """
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        
        # Initialize the hyper-parameters
        self.num_train = num_train
        self.num_val = num_val
        self.num_inputs = num_inputs
        self.batch_size = batch_size
        
        n = num_train + num_val
        self.X = torch.randn(n, num_inputs)
        noise = torch.randn(n ,1) * 0.01
        w, b = torch.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = torch.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        # convert slice to DataLoader
        tensors = tuple(a[i] for a in [self.X, self.y])
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, self.batch_size, shuffle=train)

    def get_length(self, train):
        if train:
            return math.ceil(self.num_train/self.batch_size)
        else:
            return math.ceil(self.num_val/self.batch_size)

class ProgressBoard:
    """The board that plots data points in animation."""
    def __init__(self, xlabel=None, ylabel=None, figsize=(5, 3)):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.fig = None
        self.ax = None
        self._data = {}
        self.figsize = figsize

    def _init_plot(self):
        """Create the figure only when first needed."""
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=self.figsize)

    def draw(self, x, y, label, every_n=1):
        self._init_plot()

        if label not in self._data:
            self._data[label] = {'x': [], 'y': []}

        self._data[label]['x'].append(x)
        self._data[label]['y'].append(float(y))

        if len(self._data[label]['x']) % every_n == 0:
            self.ax.clear()

            for lbl, items in self._data.items():
                self.ax.plot(items['x'], items['y'], label=lbl)

            self.ax.set_xlabel(self.xlabel)
            self.ax.legend()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

class Trainer():
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.max_epochs = max_epochs
        self.num_gpus = num_gpus
        self.gradient_clip_val = gradient_clip_val
        self.board = ProgressBoard()
        assert num_gpus == 0

    def prepare_data(self, data):
        self.train_dataloader = data.get_dataloader(train=True)
        self.val_dataloader = data.get_dataloader(train=False)
        self.num_train_batches = data.get_length(train=True)
        self.num_val_batches = data.get_length(train=False) if data.get_length(train=True) > 0 else 0

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.config_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.training_step(batch)
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
            self.train_batch_idx += 1
        if self.val_dataloader is None:
            return
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(batch)
            self.val_batch_idx += 1