import torch
from torch import nn
import torch.nn.functional as F
from utils import ProgressBoard, l2_penalty

class WeightDecayPenalty(nn.Module):
    def __init__(self, num_inputs, lbd, lr, sigma=0.01):
        super().__init__()
        # Model hyperparameters
        self.net = nn.LazyLinear(1)
        self.num_inputs = num_inputs
        self.lbd = lbd
        self.lr = lr
        self.sigma = sigma
        self.w = torch.normal(0, self.sigma, (self.num_inputs, 1), requires_grad=True)
        # utils for plotting grapgh
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

class MLPClassificationModel(nn.Module):
    """MLP architecture for classification model"""
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.lr = lr
        # initialzing the model architecture
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_hiddens),
            nn.ReLU(),
            nn.LazyLinear(num_outputs)
        )
        self.board = ProgressBoard()
        self.plot_train_per_epoch=1
        self.plot_valid_per_epoch=1

    def forward(self, X):
        # X = X.reshape((X.shape[0], -1))
        return self.net(X)
        # X = X.reshape((X.shape[0], -1))
        # H = nn.ReLU(torch.matmul(X, self.W1) + self.b1)
        # return torch.matmul(H, self.W2) + self.b2

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

class MLPClassificationDropout(nn.Module):
    """MLP architecture for classification model with dropout regularization"""
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2, dropout_1, dropout_2, lr):
        super().__init__()
        self.num_outputs = num_outputs
        self.num_hiddens_1 = num_hiddens_1
        self.num_hiddens_2 = num_hiddens_2
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2
        self.lr = lr
        # initialzing the model architecture
        self.net = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(num_hiddens_1), nn.ReLU(),
            nn.Dropout(dropout_1), nn.LazyLinear(num_hiddens_2), nn.ReLU(),
            nn.Dropout(dropout_2), nn.LazyLinear(num_outputs)
        )
        self.board = ProgressBoard()
        self.plot_train_per_epoch=1
        self.plot_valid_per_epoch=1

    def forward(self, X):
        return self.net(X)

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
        # creat a mask tensor with same shape as input X and if check each element with dropout
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

# class Conv2D(nn.Module):
#     # Convulation block
#     def __init__(self, kernel_size):
#         super().__init__()
#         self.weight = nn.Parameter((torch.rand(kernel_size)))
#         self.bais = nn.Parameter(torch.zeros(1))

#     def forward(self, X):
#         # perform cross-correlation (convulation)
#         h, w = self.weight.shape
#         # output tensor size is given by the formula:
#         # (h1 - h2 + 1) X (w1 - w2 + 1), where
#         # (h1, w1) is the size of image and (h2, w2) is the size of the kernel
#         Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
#         for i in range(Y.shape[0]):
#             for j in range(Y.shape[1]):
#                 Y[i, j] = (X[i:i + h, j:j + w] * self.weight).sum()
#         return Y + self.bias

class LeNet(nn.Module):
    """The LeNet-5 model."""
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.board = ProgressBoard()
        self.plot_train_per_epoch=1
        self.plot_valid_per_epoch=1
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.Sigmoid(),
            nn.LazyLinear(84), nn.Sigmoid(),
            nn.LazyLinear(num_classes))

    def forward(self, X):
        return self.net(X)

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