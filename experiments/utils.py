import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, TensorDataset, DataLoader
import math
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline

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

class FashionMNIST(Dataset):
    """Data Loader for FashionMNIST Dataset"""
    def __init__(self, batch_size=32, resize=(28, 28)):
        super().__init__()
        self.batch_size = batch_size
        self.resize = resize

        transformed = transforms.Compose(
                [transforms.Resize(resize), transforms.ToTensor()]                
            )
        self.train = torchvision.datasets.FashionMNIST(
                root="/home/arush/Arush/ML_Clutter/test/data", train=True, transform=transformed, download=False
            )
        self.val = torchvision.datasets.FashionMNIST(
                root="/home/arush/Arush/ML_Clutter/test/data", train=False, transform=transformed, download=False
            )
    
    def text_labels(self, indices):
        """Convert labels from int to text labels"""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        # Convert tensor → list
        if torch.is_tensor(indices):
            indices = indices.tolist()

        # If single integer
        if isinstance(indices, int):
            return labels[int(indices)]

        return [labels[int(i)] for i in indices]
    
    def get_dataloader(self, train):
        data = self.train if train else self.val
        return DataLoader(data, self.batch_size, shuffle=train)

    def get_length(self, train):
        if train:
            return len(self.get_dataloader(train=True))
        else:
            return len(self.get_dataloader(train=False))
    
    def visualize_data(self, batch, nrows=1, ncols=8, labels=[]):
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        plt.figure(figsize=(ncols * 2, nrows * 2))

        for i in range(nrows * ncols):
            if i >= len(X):
                break
            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(X[i].squeeze(), cmap="gray")
            plt.title(labels[i], fontsize=8)
            plt.axis("off")

        plt.tight_layout()
        plt.show()

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

    def draw(self, x, y, label, every_n=1, xscale=None, yscale=None):
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
            self.ax.set_ylabel(self.ylabel)
    
            if xscale:
                self.ax.set_xscale(xscale)
            if yscale:
                self.ax.set_yscale(yscale)
    
            self.ax.legend()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()


class Trainer():
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.max_epochs = max_epochs
        self.num_gpus = num_gpus
        self.gradient_clip_val = gradient_clip_val
        self.board = ProgressBoard()
        self.device = 'cpu'
        if self.num_gpus > 0:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # assert num_gpus == 0

    def prepare_data(self, data):
        self.train_dataloader = data.get_dataloader(train=True)
        self.val_dataloader = data.get_dataloader(train=False)
        self.num_train_batches = data.get_length(train=True)
        self.num_val_batches = data.get_length(train=False) if data.get_length(train=True) > 0 else 0

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model.to(self.device)

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.config_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self.total_train_loss = 0.0
        self.total_train_batches = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
        train_loss = self.total_train_loss / self.total_train_batches
        print(f"Final Train Loss: {train_loss:.4f}")

    
    def to_device(self, batch):
        # helper function to move all samples to device
        return [b.to(self.device) for b in batch]
    
    def fit_epoch(self):
        self.model.train()
        for batch in self.train_dataloader:
            batch = self.to_device(batch)
            loss = self.model.training_step(batch)
            self.total_train_loss += loss.item()
            self.total_train_batches += 1
            self.optim.zero_grad()
            if self.num_gpus > 0:
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
            else:
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
            batch = self.to_device(batch)
            with torch.no_grad():
                self.model.validation_step(batch)
            self.val_batch_idx += 1

    def clip_gradients(self, grad_clip_val, model):
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points."""

    def has_one_axis(X):  # True if X (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    backend_inline.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize
    if axes is None:
        axes = plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)