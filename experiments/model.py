import torch
from torch import nn
import torch.nn.functional as F
from utils import ProgressBoard, l2_penalty

class WeightDecayPenalty(nn.Module):
    def __init__(self, num_inputs, lbd, lr, sigma=0.01):
        super().__init__()
        self.net = nn.LazyLinear(1)
        self.num_inputs = num_inputs
        self.lbd = lbd
        self.lr = lr
        self.sigma = sigma
        self.w = torch.normal(0, self.sigma, (self.num_inputs, 1), requires_grad=True)
        self.board = ProgressBoard()
        self.plot_train_per_epoch=1
        self.plot_valid_per_epoch=1

    def forward(self, X):
        # y` = Xw + b 
        return self.net(X)

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)
    
    def loss(self, y_pred, y):
        fn = nn.MSELoss()
        w_net = self.net.weight
        return (fn(y_pred, y) + self.lbd * l2_penalty(w_net))

    def config_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)
        
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

class Classification(nn.Module):
    """Base class for classification models"""
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        # self.lbd = lbd
        self.lr = lr
        self.sigma = sigma
        self.w = nn.Parameter(torch.normal(0, self.sigma, (self.num_inputs, self.num_outputs)))
        self.b = nn.Parameter(torch.zeros(self.num_outputs))
        self.board = ProgressBoard()
        self.plot_train_per_epoch=1
        self.plot_valid_per_epoch=1

    def forward(self, X):
        X = X.reshape((X.shape[0], -1))
        # y_pred = softmax(Xw + b)
        return F.softmax(torch.matmul(X, self.w) + self.b, dim=1)

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
        y_pred = y_pred.reshape((-1, y_pred.shape[-1])) # reshape the prediction logit into the same shape as y label
        preds = y_pred.argmax(axis=1).type(y.dtype) # get the index value of the largest value (this gives us the class number)
        compare = (preds == y.reshape(-1)).type(torch.float32) # compare y_pred with the label y
        return compare.mean() if averaged else compare

    def config_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)
        
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