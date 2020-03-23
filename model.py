import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence


class MMIModel(nn.Module):
    def __init__(self, num_word_types, num_char_types, num_labels, word_dim,
                 char_dim, width, num_lstm_layers):
        super(MMIModel, self).__init__()
        self.word_embedding = nn.Embedding(num_word_types, word_dim, padding_idx=0)
        self.char_embedding = nn.Embedding(num_char_types, char_dim, padding_idx=0)
        self.num_labels = num_labels
        self.width = width

        self.loss = Loss()

        self.past = PastEncoder(
            word_embedding=self.word_embedding,
            width=width,
            num_labels=num_labels,
        )
        self.future = FutureEncoder(
            word_embedding=self.word_embedding,
            char_embedding=self.char_embedding,
            num_layers=num_lstm_layers,
            num_labels=num_labels,
        )

    def forward(self, past_words, future_words, padded_chars, char_lengths,
                is_training=True):
        past_rep = self.past(past_words)
        future_rep = self.future(future_words, padded_chars, char_lengths)

        if is_training:
            return self.loss(past_rep, future_rep)

        else:
            future_probs, future_indices = future_rep.max(1)
            return future_probs, future_indices


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.entropy = Entropy()

    def forward(self, past_rep, future_rep):
        pZ_Y = F.softmax(future_rep, dim=1)
        pZ = pZ_Y.mean(0)
        hZ = self.entropy(pZ)

        x = pZ_Y * F.log_softmax(past_rep, dim=1)  # B x m
        return -1.0 * x.sum(dim=1).mean() - hZ


class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, probs):
        x = probs * torch.log(probs)
        entropy = -1.0 * x.sum()
        return entropy


class PastEncoder(nn.Module):
    def __init__(self, word_embedding, width, num_labels):
        super(PastEncoder, self).__init__()
        self.word_embedding = word_embedding
        self.fc = nn.Linear(2 * width * word_embedding.embedding_dim, num_labels)

    def forward(self, words):
        word = self.word_embedding(words)  # B x 2width x d_w
        # B x m
        return self.fc(word.view(words.size(0), -1))


class FutureEncoder(nn.Module):
    def __init__(self, word_embedding, char_embedding, num_layers, num_labels):
        super(FutureEncoder, self).__init__()
        self.word_embedding = word_embedding
        self.char_embedding = char_embedding
        self.char_rnn = nn.LSTM(
            char_embedding.embedding_dim, char_embedding.embedding_dim,
            num_layers, bidirectional=True,
        )
        self.fc = nn.Linear(
            word_embedding.embedding_dim + 2 * char_embedding.embedding_dim,
            num_labels,
        )

    def forward(self, words, padded_chars, char_lengths):
        B = len(char_lengths)
        word = self.word_embedding(words)  # B x d_w

        pack = pack_padded_sequence(self.char_embedding(padded_chars), char_lengths)
        _, (ctx, _) = self.char_rnn(pack)

        ctx = ctx.view(self.char_rnn.num_layers, 2, B, self.char_rnn.hidden_size)[-1]  # 2 x B x d_c
        char = ctx.transpose(0, 1).contiguous().view(B, -1)  # B x 2d_c

        # B x m
        return self.fc(torch.cat([word, char], 1))
