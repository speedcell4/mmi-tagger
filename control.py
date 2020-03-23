import math
import time
from datetime import timedelta

import torch
import torch.nn as nn

from evaluate import compute_many2one_acc, compute_v_measure


class Control(nn.Module):

    def __init__(self, model, model_path, batch_size, device, logger):
        super(Control, self).__init__()
        self.model = model
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device
        self.logger = logger

    def train(self, data, lr, epochs):
        self.log_data(data)
        self.logger.log('[TRAINING]')
        self.logger.log(f'   num_labels:    {self.model.num_labels:d}')
        self.logger.log(f'   dim:           {self.model.wemb.embedding_dim:d}')
        self.logger.log(f'   batch_size:    {self.batch_size:d}')
        self.logger.log(f'   lr:            {lr:g}')
        self.logger.log(f'   epochs:        {epochs:d}')
        self.logger.log('')

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        best_acc = float('-inf')
        start_time = time.time()

        try:
            for epoch in range(1, epochs + 1):
                avg_loss, acc, vm, epoch_time = self.do_epoch(data, optimizer)
                bits = (- avg_loss) * math.log2(math.e)
                self.logger.log('| epoch {:3d} | loss {:6.2f} | {:6.2f} bits | '
                                'acc {:6.2f} | vm {:6.2f} | time {:10s}'.format(
                    epoch, avg_loss, bits, acc, vm,
                    str(timedelta(seconds=int(epoch_time)))))
                if best_acc < acc:
                    best_acc = acc
                    with open(self.model_path, 'wb') as f:
                        torch.save(self.model, f)

        except KeyboardInterrupt:
            self.logger.log('-' * 89)
            self.logger.log('Exiting from training early')

        self.logger.log(f'\nTraining time {str(timedelta(seconds=int(time.time() - start_time))):10s}')

        self.load_model()
        acc, vm, zseqs, clustering = self.evaluate(data)
        self.logger.log('=' * 89)
        self.logger.log(f'| Best | acc {acc:5.2f} | vm {vm:5.2f}')
        self.logger.log('=' * 89)

        return acc, vm, zseqs, clustering

    def do_epoch(self, data, optimizer):
        self.model.train()
        avg_loss = 0
        epoch_start_time = time.time()
        batches = data.get_batches(self.batch_size)
        for batch in batches:
            self.model.zero_grad()
            X, Y1, Y2, lengths = data.tensorize_batch(batch, self.device,
                                                      self.model.width)
            loss = self.model(X, Y1, Y2, lengths, is_training=True)
            avg_loss += loss.item() / len(batches)
            loss.backward()
            optimizer.step()

        acc, vm, _, _ = self.evaluate(data)
        epoch_time = time.time() - epoch_start_time

        return avg_loss, acc, vm, epoch_time

    def evaluate(self, data):
        self.model.eval()
        batches = data.get_batches(self.batch_size)
        zseqs = [[False for w in sent] for sent in data.sents]
        clustering = [{} for z in range(self.model.num_labels)]
        with torch.no_grad():
            for batch in batches:
                X, Y1, Y2, lengths = data.tensorize_batch(batch, self.device,
                                                          self.model.width)
                future_probs, future_indices = self.model(X, Y1, Y2, lengths,
                                                          is_training=False)
                for k, (i, j) in enumerate(batch):
                    z = future_indices[k].item()
                    zseqs[i][j] = z
                    clustering[z][data.sents[i][j]] = True

        acc = compute_many2one_acc(data.golds, zseqs)
        vm = compute_v_measure(data.golds, zseqs)
        return acc, vm, zseqs, clustering

    def load_model(self):
        with open(self.model_path, 'rb') as f:
            self.model = torch.load(f)
        self.model.future.lstm.flatten_parameters()

    def log_data(self, data):
        self.logger.log('-' * 89)
        self.logger.log('[DATA]')
        self.logger.log(f'   data:          {data.data_path}')
        self.logger.log(f'   # word types:  {len(data.w2i):d}')
        self.logger.log(f'   # char types:  {len(data.c2i):d}')
        self.logger.log('   # words:       %d' % sum(len(sent) for sent in
                                                     data.sents))
        self.logger.log(f'   # tag types:   {len(data.label_counter):d}')
        self.logger.log('-' * 89)
