import torch
from torch import nn
import torch.nn.functional as F
from utils import ProgressBoard

def init_cnn(module):
    """
    Uniform Xavier parameter initialization
    x = sqrt(6 / n_in + n_out)
    """
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)

def vgg_block(num_convs, out_channels):
    """
    ((Conv → ReLU) X num_convs) + Pool
    """
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, arch, lr=0.1, num_classes=10):
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes
        self.lr = lr
        # self.w = nn.Parameter(torch.normal(0, self.sigma, (self.num_inputs, self.num_outputs)))
        # self.b = nn.Parameter(torch.zeros(self.num_outputs))
        self.board = ProgressBoard()
        self.plot_train_per_epoch=1
        self.plot_valid_per_epoch=1
        
        conv_blks = []
        for (num_convs, out_channels) in arch:
            conv_blks.append(vgg_block(num_convs, out_channels))
        self.net = nn.Sequential(
            *conv_blks, nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(num_classes))
        self.net.apply(init_cnn)

    def forward(self, X):
        return self.net(X)

    def layer_summary(self, X_shape):
        """Defined in :numref:`sec_lenet`"""
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)

    def apply_init(self, inputs, init=None):
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)
    
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        y_pred = self(*batch[:-1])
        self.plot('loss', self.loss(y_pred, batch[-1]), train=False)
        self.plot('acc', self.accuracy(y_pred, batch[-1]), train=False)
    
    def loss(self, y_pred, y, averaged=True):
        y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
        y = y.reshape((-1,))
        return F.cross_entropy(
            y_pred, y, reduction='mean' if averaged else 'none'
        )

    def accuracy(self, y_pred, y, averaged=True):
        y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
        preds = y_pred.argmax(axis=1).type(y.dtype)
        compare = (preds == y.reshape(-1)).type(torch.float32)
        return compare.mean() if averaged else compare

    def config_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)

    def dropout_layer(X, dropout):
        assert 0 <= dropout <= 1
        if dropout == 1: return torch.zeros_like(X)
        mask = (torch.rand(X.shape) > dropout).float()
        return mask * X / (1.0 - dropout)
        
    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        device = torch.device('cpu')
        self.board.draw(x, value.to(device).detach().numpy(),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))