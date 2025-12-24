import os
import re
import requests
import collections
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

URL = "https://www.gutenberg.org/cache/epub/35/pg35.txt"
FILENAME = "timemachine.txt"

class TimeMachineData(Dataset):
    """Class for the time machine dataset"""
    def __init__(self, batch_size, num_steps, num_train=10000, num_val=5000):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_train = num_train
        self.num_val = num_val
        corpus, self.vocab = self.build(self._download_and_read(url=URL, filename=FILENAME))
        array = torch.tensor([corpus[i:i+num_steps+1]
                            for i in range(len(corpus)-num_steps)])
        self.X, self.Y = array[:,:-1], array[:,1:]
    
    def _download_and_read(self, url, filename, folder="data"):
        self.url = url
        self.filename = filename
        self.directory = folder

        # create directory if it does not exist
        os.makedirs(folder, exist_ok=True)

        # make fname an instance variable
        fname = os.path.join(folder, filename)

        req = requests.get(url, stream=True, verify=True)

        # write data to file
        with open(fname, 'wb') as f:
            f.write(req.content)
        # read
        with open(fname) as f:
            return f.read()

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, self.batch_size, shuffle=train)
    
    def get_dataloader(self, train):
        idx = slice(0, self.num_train) if train else slice(
            self.num_train, self.num_train + self.num_val)
        return self.get_tensorloader([self.X, self.Y], train, idx)
    
    def get_length(self, train):
        if train:
            return len(self.get_dataloader(train))
        else:
            return len(self.get_dataloader(train))
    
    def _preprocess(self, text):
        """Ignore punctuation and make all text to lower case"""
        return re.sub('[^A-Za-z]+', ' ', text).lower()

    def _tokenize(self, text):
        return list(text)

    def build(self, raw_text, vocab=None):
        """
        Pre-process data -> tokenize -> create vocab
        """
        tokens = self._tokenize(self._preprocess(raw_text))
        if vocab is None: vocab = Vocab(tokens)
        corpus = [vocab[token] for token in tokens]
        return corpus, vocab

class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']