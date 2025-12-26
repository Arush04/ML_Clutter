import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """Base encoder interface for encoder-decoder architecture"""
    def __init__(self):
        super().__init__()

    def forward(self, X, *args):
        pass

class Decoder(nn.Module):
    """Base decoder interface for encoder-decoder architecture"""
    def __init__(self):
        super().__init__()

    # enc_all_outputs is the output from the encoder
    def init_state(self, enc_all_outputs, *args):
        pass

    def forward(self, X, state):
        pass

class EncoderDecoder(nn.Module):
    """Base class for the encoder-decoder architecture"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state)[0]

    def loss(self, y_pred, y, averaged):
        y_pred = y_pred.reshape(-1, self.vocab_size)
        y = y.reshape(-1)
        loss = F.cross_entropy(y_pred, y, reduction='none')
        if averaged:
            return loss.mean()
        return loss