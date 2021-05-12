import argparse
import pickle
import random
import re
import sts

import datasets
import nltk
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn

from structformer import StructFormer


def tokenise(x):
    x = x.lower()
    x = re.sub('[0-9]+', 'N', x)
    x = [w for w in nltk.word_tokenize(x) if re.match('.*[a-zA-Z]+.*', w)]
    return x

def batchify(items, bsz, device, pad=0, shuffle=True):
    """Batchify the training data."""
    if shuffle:
        random.shuffle(items)

    def get_batches():
        buffer_x1x2 = []
        buffer_y = []
        for sentence1, sentence2, score in items:
            if len(buffer_y) == bsz:
                yield buffer_x1x2, buffer_y
                buffer_x1x2 = []
                buffer_y = []

            buffer_x1x2.append(sentence1)
            buffer_x1x2.append(sentence2)
            buffer_y.append(score)

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

    return data_batched

def load_dataset(dataset, dictionary, device, bsz):
    data = []
    for x in dataset:
        sentence1 = x['sentence1']
        sentence2 = x['sentence2']
        score = x['label']
        idxs1 = [dictionary[w] for w in tokenise(sentence1)]
        idxs2 = [dictionary[w] for w in tokenise(sentence2)]
        data.append((idxs1, idxs2, score))
    data = batchify(data, bsz, pad=dictionary['<pad>'], device=device)
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
            pred_y.append(cls(x))
            true_y.append(y)
        pred_y = torch.cat(pred_y)
        true_y = torch.cat(true_y)
        score = pearson_correlation(pred_y.float(), true_y.float())
    return score




class Classifier(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, nhid, dropout, encoder: StructFormer, padding_idx):
        super(Classifier, self).__init__()

        self.padding_idx = padding_idx
        self._encoder = encoder

        self.mlp = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(2 * nhid, nhid),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(nhid, 1),
            nn.Sigmoid(),
        )

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_uniform_(self.mlp[3].weight)
        nn.init.zeros_(self.mlp[3].bias)

    def encode(self, x, mask):
        pos = torch.arange(x.size(1), device=x.device)[None, :]
        _, p_dict = self._encoder(x, x, pos)
        raw_output = p_dict['raw_output']
        root_emb = torch.sum(
            raw_output.masked_fill(~mask[:, :, None], 0.),
            dim=1
        ) / torch.sum(mask, dim=1)[:, None]
        return root_emb

    def forward(self, input):
        mask = (input != self.padding_idx)
        output = self.encode(input, mask)

        clause_1 = output[::2]
        clause_2 = output[1::2]

        # if self.training:
        #     mask = torch.rand_like(clause_1_[:, 0]) > 0.5
        #     clause_1 = torch.empty_like(clause_1_)
        #     clause_2 = torch.empty_like(clause_2_)
        #     clause_1[mask] = clause_1_[mask]
        #     clause_2[mask] = clause_2_[mask]
        #     clause_1[~mask] = clause_2_[~mask]
        #     clause_2[~mask] = clause_1_[~mask]
        # else:
        #     clause_1 = clause_1_
        #     clause_2 = clause_2_

        output = self.mlp(torch.cat([
            # clause_1, clause_2,
            clause_1 * clause_2,
            torch.abs(clause_1 - clause_2)
        ], dim=1))
        return output.squeeze(-1) * 5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetune on STS-B')
    parser.add_argument('--dictionary', type=str, help='Dictionary location')
    parser.add_argument('--model', type=str, help='Model location')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument(
        '--lr', type=float, default=0.0003, help='initial learning rate')
    parser.add_argument(
        '--dropout', type=float, default=0.2)
    parser.add_argument(
        '--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument(
        '--bsz', type=int, default=128, help='Number of epochs')
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

    valid_data = load_dataset(dataset['validation'], dictionary, device=device, bsz=args.bsz)
    test_data = load_dataset(dataset['test'], dictionary, device=device, bsz=args.bsz)

    print('Loading model...')
    with open(args.model, 'rb') as f:
        model, _, _, _ = torch.load(f, map_location=torch.device('cpu'))
        torch.cuda.manual_seed(args.seed)

    print('Initialising classfier...')
    cls = Classifier(
        nhid=model.emb_size,
        dropout=args.dropout, 
        encoder=model,
        padding_idx=dictionary['<pad>']
    ).to(device)
    print('done')
    mlm_params = cls._encoder.lm_parameters()
    cls_params = list(cls.mlp.parameters())

    params = mlm_params + cls_params
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-6)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', 0.5, patience=2, threshold=0)
    criterion = nn.MSELoss()
    best_score = -1.
    try:
        for epoch in range(args.epochs):
            cls.train()
            train_data = load_dataset(dataset['train'], dictionary, device=device, bsz=args.bsz)
            for x, y in train_data:
                output = cls(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            val_score = evaluate(cls, valid_data)
            if val_score > best_score:
                best_score = val_score
                torch.save(cls, 'sts_ft.pt')
            print("Epoch %3d, Score: %.3f" % (epoch, val_score))
            scheduler.step(val_score)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    taskpath ="data/STS/"
    taskpath_year = taskpath + "STS%d-en-test"
    taskpath_stsb = taskpath + "STSBenchmark"
    taskpath_sick = taskpath + "SICK"

    cls = torch.load('sts_ft.pt')
    stsb_eval = sts.STSBenchmarkEval(taskpath_stsb)
    sick_eval = sts.SICKRelatednessEval(taskpath_sick)
    evals = [eval("sts.STS%dEval" % year)(taskpath_year % year)
             for year in [12, 13, 14, 15, 16]] + [stsb_eval, sick_eval]
    for se in evals:
        test_data = load_dataset(se.data, dictionary, device=device,
                                 bsz=args.bsz)
        test_score = evaluate(cls, test_data)
        print(type(se).__name__, test_score)

