import datasets
import nltk
import re
import pickle
import numpy as np
import torch
import random
import argparse
from pprint import pprint
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler

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


def pearson_correlation(x, y):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    x_norm = x - x_mean
    y_norm = y - y_mean
    denom = (torch.sqrt(torch.sum(x_norm**2)) *
             torch.sqrt(torch.sum(y_norm**2)))
    numer = torch.sum(x_norm * y_norm)
    return numer / denom

def evaluate(cls, dataset):
    cls.eval()
    with torch.no_grad():
        pred_y = []
        true_y = []
        for x, y in dataset:
            pred_y.append(torch.argmax(cls(x), dim=-1))
            true_y.append(y)
        pred_y = torch.cat(pred_y)
        true_y = torch.cat(true_y)
        score = pearson_correlation(pred_y.float(), true_y.float())
    return score




class Classifier(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, nhid, nout, dropouti, dropouto, encoder, padding_idx):
        super(Classifier, self).__init__()

        self.padding_idx = padding_idx
        self._encoder = encoder

        self.mlp = nn.Sequential(
            nn.Dropout(dropouti),
            nn.Linear(4 * nhid, nhid),
            nn.ELU(),
            nn.Dropout(dropouto),
            nn.Linear(nhid, nout),
        )


        self.cost = nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def encode(self, x, mask):
        pos = torch.arange(x.size(1), device=x.device)[None, :]
        _, p_dict = self._encoder(x, x, pos)
        raw_output = p_dict['raw_output']
        root_emb = torch.mean(raw_output, dim=1)
#        head = p_dict['head']
#        root_ = (1 - torch.sum(head, dim=-1)).masked_fill(~mask, 0.)
#        root_p = root_ / root_.sum(dim=-1, keepdim=True)
#        root_emb = torch.einsum('bih,bi->bh', raw_output, root_p)
        return root_emb

    def forward(self, input):
        batch_size = input.size(1)
        mask = (input != self.padding_idx)
        output = self.encode(input, mask)

        clause_1_ = output[::2]
        clause_2_ = output[1::2]

        if self.training:
            mask = torch.rand_like(clause_1_[:, 0]) > 0.5
            clause_1 = torch.empty_like(clause_1_)
            clause_2 = torch.empty_like(clause_2_)
            clause_1[mask] = clause_1_[mask]
            clause_2[mask] = clause_2_[mask]
            clause_1[~mask] = clause_2_[~mask]
            clause_2[~mask] = clause_1_[~mask]
        else:
            clause_1 = clause_1_
            clause_2 = clause_2_

        output = self.mlp(torch.cat([
            clause_1, clause_2,
            clause_1 * clause_2,
            torch.abs(clause_1 - clause_2)
        ], dim=1))
        return output

    def loss(self, output, y):
        score_mu = torch.arange(output.size(1),
                                device=y.device,
                                dtype=torch.float)
        dists = -0.5 * (score_mu[None, :] - y[:, None])**2
        log_p = torch.log_softmax(output, dim=-1)

        return -torch.logsumexp(log_p + dists, dim=-1).mean()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetune on STS-B')
    parser.add_argument('--dictionary', type=str, help='Dictionary location')
    parser.add_argument('--model', type=str, help='Model location')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument(
        '--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument(
        '--epochs', type=int, default=50, help='Number of epochs')
    args = parser.parse_args()

    # Load data
    print("Loading data...")
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
    test_data = load_dataset(dataset['test'], dictionary, device=device)

    print('Loading model...')
    with open(args.model, 'rb') as f:
        model, _, _, _ = torch.load(f, map_location=device)
        torch.cuda.manual_seed(args.seed)
        if args.cuda:
            model.cuda()

    print('Initialising classfier...')
    cls = Classifier(
        nhid=512, nout=6,
        dropouti=0.1, dropouto=0.1,
        encoder=model,
        padding_idx=dictionary['<pad>']
    ).to(device)
    pprint([n for n, _ in cls.named_parameters()]) 
    mlm_params = [p for n, p in cls.named_parameters()
                  if n.startswith('_encoder.layers') ] + \
                 [cls._encoder.emb.weight]
    cls_params = [p for n, p in cls.named_parameters()
                  if not n.startswith('_encoder') ]

    params = mlm_params + cls_params
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', 0.5, patience=2, threshold=0)


    for epoch in range(args.epochs):
        cls.train()
        for x, y in train_data:
            loss = cls.loss(cls(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        val_score = evaluate(cls, valid_data)
        print("Epoch", epoch, "score:", val_score)
        scheduler.step(val_score)
    test_score = evaluate(cls, test_data)
    print("Test score:", test_score)
