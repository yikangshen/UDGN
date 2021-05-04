import datasets
import nltk
import re
import pickle
import numpy as np
import torch
import random
import argparse

from torch import nn

def tokenise(x):
    x = x.lower()
    x = re.sub('[0-9]+', 'N', x)
    x = [w for w in nltk.word_tokenize(x) if re.match('.*[a-zA-Z]+.*', w)]
    return x

def batchify(items, bsz, device, pad=0, shuffle=True):
    """Batchify the training data."""
    items = sorted(items, key=lambda x: max(x[0], x[1]))

    def get_batches():
        buffer_x1x2 = []
        buffer_y = []
        total_length = 0
        for sentence1, sentence2, score in items:
            item_length = len(sentence1) + len(sentence2)
            if total_length + item_length > bsz:
                yield buffer_x1x2, buffer_y
                buffer_x1x2 = []
                buffer_y = []
                total_length = 0

            buffer_x1x2.append(sentence1)
            buffer_x1x2.append(sentence2)
            buffer_y.append(score)
            total_length += item_length

        if len(buffer_x1x2) > 0:
            yield buffer_x1x2, buffer_y

    def create_padded_tensor(buffer_x1x2, buffer_y):
        t_x1x2 = torch.full((len(buffer_x1x2), max(len(x) for x in buffer_x1x2)),  pad, dtype=torch.long,
                            device=device)
        for i, x in enumerate(buffer_x1x2):
            t_x1x2[i, :len(x)] = torch.tensor(x)
        return t_x1x2, torch.tensor(buffer_y, device=device)

    data_batched = []
    for buffer_x1x2, buffer_y in get_batches():
        x1x2, y = create_padded_tensor(buffer_x1x2, buffer_y)
        data_batched.append((x1x2, y))

    if shuffle:
        random.shuffle(data_batched)

    return data_batched

def load_dataset(dataset, dictionary, device):
    data = []
    for x in dataset:
        sentence1 = x['sentence1']
        sentence2 = x['sentence2']
        score = x['label']
        idxs1 = [dictionary[w] for w in tokenise(sentence1)]
        idxs2 = [dictionary[w] for w in tokenise(sentence2)]
        data.append((idxs1, idxs2, score))
    data = batchify(data, 4096, pad=dictionary['<pad>'], device=device)
    
    return  data


class Classifier(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nout, dropouti, dropouto, encoder, padding_idx):
        super(Classifier, self).__init__()

        self.padding_idx = padding_idx
        self._encoder = encoder

        self.mlp = nn.Sequential(
            nn.Dropout(dropouto),
            nn.Linear(4 * nhid, nhid),
            nn.ELU(),
            nn.Dropout(dropouto),
            nn.Linear(nhid, nout),
        )

        self.drop = nn.Dropout(dropouti)

        self.cost = nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def encode(self, x, mask):
        pos = torch.arange(x.size(1), device=x.device)[None, :]
        _, p_dict = self._encoder(x, x, pos)
        raw_output = p_dict['raw_output']
        head = p_dict['head']
        root_ = (1 - torch.sum(head, dim=-1)).masked_fill(~mask, 0.)
        root_p = root_ / root_.sum(dim=-1, keepdim=True)
        root_emb = torch.einsum('bih,bi->bh', raw_output, root_p)
        return root_emb

    def forward(self, input):
        batch_size = input.size(1)
        mask = (input != self.padding_idx)
        output = self.encoder(input, mask)

        clause_1 = output[::2]
        clause_2 = output[1::2]
        output = self.mlp(torch.cat([
            clause_1, clause_2,
            clause_1 * clause_2,
            torch.abs(clause_1 - clause_2)
        ], dim=1))
        return output



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetune on STS-B')
    parser.add_argument('--dictionary', type=str, help='Dictionary location')
    parser.add_argument('--model', type=str, help='Model location')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()

    # Load data
    dataset = datasets.load_dataset('glue', 'stsb')
    dictionary = pickle.load(open(args.dictionary, 'rb'))

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        if not args.cuda:
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed(args.seed)

    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    train_data = load_dataset(dataset['train'], dictionary, device=device)
    valid_data = load_dataset(dataset['validation'], dictionary, device=device)
    test_data = load_dataset(dataset['validation'], dictionary, device=device)

    for x,y in train_data:
        x1 = x[::2]
        x2 = x[1::2]
        print(' '.join(dictionary.idx2word[i] for i in x1[0] if i != dictionary['<pad>']))
        print(' '.join(dictionary.idx2word[i] for i in x2[0] if i != dictionary['<pad>']))
        print(y[0])
        print()
        # print(tokenise(sentence1))


