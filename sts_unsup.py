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
import torch.nn.functional as F

from scipy import stats

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

def evaluate(cls, dataset, whitened=False):
    cls.eval()
    with torch.no_grad():
        pred_y = []
        true_y = []
        for x, y in dataset:
            pred_y.append(cls(x, whitened=whitened))
            true_y.append(y)
        pred_y = torch.cat(pred_y)
        true_y = torch.cat(true_y)
        score, _ = stats.spearmanr(pred_y.cpu().numpy(), true_y.cpu().numpy())
    return score

class Classifier(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, encoder, padding_idx):
        super(Classifier, self).__init__()
        self._encoder = encoder
        self.padding_idx = padding_idx

    def encode(self, x, mask):
        pos = torch.arange(x.size(1), device=x.device)[None, :]
        _, p_dict = self._encoder(x, x, pos)
        all_layers = p_dict['all_layers']
        all_layers = [all_layers[1], all_layers[-1]]
        x = sum(all_layers) / len(all_layers)
        # x = all_layers[-1]
        root_emb = torch.sum(
            x.masked_fill(~mask[:, :, None], 0.),
            dim=1
        ) / torch.sum(mask, dim=1)[:, None]

        """
        head = p_dict['head']
        root_ = (1 - torch.sum(head, dim=-1)).masked_fill(~mask, 0.)
        root_p = root_ / root_.sum(dim=-1, keepdim=True)
        root_emb = torch.einsum('bih,bi->bh', raw_output, root_p)
        """
        return root_emb

    def whiten(self, dataset):
        self.eval()
        with torch.no_grad():
            sum_emb = 0.
            sum_emb_outer = 0.
            sum_counts = 0
            for x, _ in dataset:
                mask = (x!= self.padding_idx)
                output = self.encode(x, mask)
                sum_counts += output.size(0)
                sum_emb += torch.sum(output, dim=0)
                sum_emb_outer += torch.einsum('bi,bj->ij', output, output)
            mean = sum_emb / sum_counts
            cov = sum_emb_outer / sum_counts - mean[:, None] * mean[None, :]
        u, s, v = torch.linalg.svd(cov)
        self.mean = mean
        self.W = u / torch.sqrt(s[None, :])



    def forward(self, input, whitened=False):
        batch_size = input.size(1)
        mask = (input != self.padding_idx)
        output = self.encode(input, mask)
        if whitened:
            output = torch.einsum(
                'bj,ji->bi',
                output - self.mean,
                self.W
            )


        clause_1 = output[::2]
        clause_2 = output[1::2]
        # output = torch.sum(-(clause_1 - clause_2)**2, dim=1)
        # output = -torch.sqrt(torch.sum((clause_1 - clause_2)**2, dim=1))

        output = F.cosine_similarity(clause_1, clause_2, dim=1)
        # output = F.cosine_similarity(clause_1, clause_2, dim=1)
        return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetune on STS-B')
    parser.add_argument('--dictionary', type=str, help='Dictionary location')
    parser.add_argument('--model', type=str, help='Model location')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument( '--lr', type=float, default=0.001, help='initial learning rate')
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
    test_data = load_dataset(dataset['validation'], dictionary, device=device)
    print('Loading model...')
    with open(args.model, 'rb') as f:
        model, _, _, _ = torch.load(f, map_location=device)
        torch.cuda.manual_seed(args.seed)
        if args.cuda:
            model.cuda()
    cls = Classifier(encoder=model,
                     padding_idx=dictionary['<pad>']).to(device)

    cls.whiten(train_data)


    test_score = evaluate(cls, test_data)
    print("Test score:", test_score)
    test_score = evaluate(cls, test_data, whitened=True)
    print("Test score (Whitened):", test_score)

