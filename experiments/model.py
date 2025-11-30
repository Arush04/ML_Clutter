import torch
from torch import nn
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