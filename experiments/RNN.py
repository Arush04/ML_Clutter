import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import ProgressBoard, l2_penalty

class RNN(nn.Module):
    """The RNN model class """
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.sigma = sigma
        self.W_xh = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.W_hh = nn.Parameter(torch.randn(num_hiddens, num_hiddens) * sigma)
        self.b_h = nn.Parameter(torch.zeros(num_hiddens))

    def forward(self, inputs, state=None):
        if state is None:
            # Initial state with shape: (batch_size, num_hiddens)
            state = torch.zeros((inputs.shape[1], self.num_hiddens),
                              device=inputs.device)
        else:
            state, = state
        outputs = []
        for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs)
            state = torch.tanh(torch.matmul(X, self.W_xh) +
                             torch.matmul(state, self.W_hh) + self.b_h)
            outputs.append(state)
        return outputs, state

class RNNLM(nn.Module):
    """The RNN-based language model."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.rnn = rnn
        self.vocab_size = vocab_size
        self.lr = lr
        self.init_params()
        # utils for plotting grapgh
        self.board = ProgressBoard()
        self.plot_train_per_epoch=1
        self.plot_valid_per_epoch=1

    def init_params(self):
        self.W_hq = nn.Parameter(
            torch.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
        self.b_q = nn.Parameter(torch.zeros(self.vocab_size))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', torch.exp(l), train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', torch.exp(l), train=False)

    def one_hot(self, X):
        # Output shape: (num_steps, batch_size, vocab_size)
        return F.one_hot(X.T, self.vocab_size).type(torch.float32)
    
    def output_layer(self, rnn_outputs):
        outputs = [torch.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
        return torch.stack(outputs, 1)
    
    def forward(self, X, state=None):
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state)
        return self.output_layer(rnn_outputs)

    def config_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)
    
    def clip_gradients(self, grad_clip_val, model):
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm

    def loss(self, y_pred, y):
        y_pred = y_pred.reshape(-1, self.vocab_size)
        y = y.reshape(-1)
        return F.cross_entropy(y_pred, y)

    def predict(self, prefix, num_preds, vocab, device=None):
        """Generative function"""
        state, outputs = None, [vocab[prefix[0]]]
        for i in range(len(prefix) + num_preds - 1):
            X = torch.tensor([[outputs[-1]]], device=device)
            embs = self.one_hot(X)
            rnn_outputs, state = self.rnn(embs, state)
            if i < len(prefix) - 1:  # Warm-up period
                outputs.append(vocab[prefix[i + 1]])
            else:  # Predict num_preds steps
                Y = self.output_layer(rnn_outputs)
                outputs.append(int(Y.argmax(axis=2).reshape(1)))
        return ''.join([vocab.idx_to_token[i] for i in outputs])
    
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

class LSTM(RNN):
    "LSTM based LM"
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        nn.Module.__init__(self)
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.sigma = sigma
        self.rnn = nn.LSTM(num_inputs, num_hiddens)

    def forward(self, inputs, H_C=None):
        return self.rnn(inputs, H_C)

class StackedGRU(RNN):
    "Deep GRU implementation"
    def __init__(self, num_inputs, num_hiddens, num_layers, dropout=0, sigma=0.01):
        nn.Module.__init__(self)
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_layers = int(num_layers)
        self.sigma = sigma
        self.dropout = dropout
        self.rnn = nn.GRU(num_inputs, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, inputs, Hs=None):
        outputs, Hs = self.rnn(inputs, Hs)
        return outputs, Hs

class BiStackedGRU(RNN):
    """Bi directional deep GRU"""
    def __init__(self, num_inputs, num_hiddens, num_layers, sigma=0.01):
        nn.Module.__init__(self)
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.sigma = sigma
        self.rnn = nn.GRU(num_inputs, num_hiddens, num_layers=self.num_layers, bidirectional=True)
        self.num_hiddens *= 2

    def forward(self, inputs, Hs=None):
        outputs, Hs = self.rnn(inputs, Hs)
        return outputs, Hs