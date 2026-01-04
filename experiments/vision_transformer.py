import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_model import MultiHeadAttention
from utils import ProgressBoard, l2_penalty

class PatchEmbedding(nn.Module):
    """
    Creating patch embeddings for the images
    num_patches = hw/p^2, where h, w and p are height, width and patch size respectively
    """
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_hiddens = num_hiddens
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)

class ViTMLP(nn.Module):
    """
    MLP -> GeLU -> dropout -> MLP -> dropout
    """
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.mlp_num_hiddens = mlp_num_hiddens
        self.mlp_num_outputs = mlp_num_outputs
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(
            self.dense1(x)))))

class ViTBlock(nn.Module):
    """
    Pre-normalization ViT block
    LayerNorm -> Multi-Head Self-Attention -> Residual
    """
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens,
                 num_heads, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.norm_shape = norm_shape
        self.mlp_num_hiddens = mlp_num_hiddens
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_bias = use_bias
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, X, valid_lens=None):
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))

class ViT(nn.Module):
    """
    Vision Transformer
    1. m(num_patches) = hw/p^2 (see patch embeddings), so total vectors = m+1 (extra cls)
    2. X = [cls; patch_embeddings] + pos_embeddings
    3. dropout(X)
    4. (ViTBlock(X)) x num_blks
    5. LayerNorm(X -> DenseLayer -> output
    """
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, emb_dropout, blk_dropout, lr=0.1,
                 use_bias=False, num_classes=10):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_hiddens = num_hiddens
        self.mlp_num_hiddens = mlp_num_hiddens
        self.num_heads = num_heads
        self.num_blks = num_blks
        self.emb_dropout = emb_dropout
        self.blk_dropout = blk_dropout
        self.lr = lr
        self.use_bias = use_bias
        self.num_classes = num_classes
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout, use_bias))
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens),
                                  nn.Linear(num_hiddens, num_classes))
        self.board = ProgressBoard()
        self.plot_train_per_epoch=1
        self.plot_valid_per_epoch=1

    def forward(self, X):
        X = self.patch_embedding(X)
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])

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
        assert hasattr(self, 'trainer'), 'Trainer is not iniated'
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