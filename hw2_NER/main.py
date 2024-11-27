import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import eojeol_etri_tokenizer.file_utils
from eojeol_etri_tokenizer.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from eojeol_etri_tokenizer.eojeol_tokenization import eojeol_BertTokenizer

import numpy as np
import os
import json

class NER_model_RNN(nn.Module):
    def __init__(self, token_vocab_size, dim_embedding, num_hidden_layers, hidden_state_size):
        super(NER_model_RNN, self).__init__()
        self.embedding = nn.Embedding(token_vocab_size, dim_embedding, padding_idx=0)

        self.lstm = nn.RNN(input_size=dim_embedding, hidden_size=hidden_state_size, 
                           num_layers=num_hidden_layers, batch_first=True, bidirectional=True)

        self.linear1 = nn.Linear(2*hidden_state_size, 512, bias=True)
        self.relu = nn.ReLU()

        self.linear2 = nn.Linear(512, 256, bias=True)
        self.relu = nn.ReLU()

        self.linear3 = nn.Linear(256, 13)

    def forward(self, x):
        # x : shape (batch, msl) where each example is a list of token ids of length msl
        x1 = self.embedding(x)  # output shape is (batch, msl, dim_embedding)
        x2 = self.lstm(x1)      # output shape is (batch, msl, 2*hidden_state_size)

        x2 = x2[0]  # the seq of the outputs of all timesteps.

        x3 = self.linear1(x2)   # output shape is (batch, msl, 512).
        x4 = self.relu(x3)
        x5 = self.linear2(x4)   # output shape is (batch, msl, 256).
        x6 = self.relu(x5)
        x7 = self.linear3(x6)   # output shape is (batch, msl, num_class).
        return x7

class NER_model_LSTM(nn.Module):
    def __init__(self, token_vocab_size, dim_embedding, num_hidden_layers, hidden_state_size):
        super(NER_model_LSTM, self).__init__()
        self.embedding = nn.Embedding(token_vocab_size, dim_embedding, padding_idx=0)

        self.lstm = nn.LSTM(input_size=dim_embedding, hidden_size=hidden_state_size, 
                            num_layers=num_hidden_layers, batch_first=True, bidirectional=True)

        self.linear1 = nn.Linear(2*hidden_state_size, 512, bias=True)
        self.relu = nn.ReLU()

        self.linear2 = nn.Linear(512, 256, bias=True)
        self.relu = nn.ReLU()

        self.linear3 = nn.Linear(256, 13)

    def forward(self, x):
        # x : shape (batch, msl) where each example is a list of token ids of length msl
        x1 = self.embedding(x)  # output shape is (batch, msl, dim_embedding)
        x2, (h_n, c_n) = self.lstm(x1)      # output shape is (batch, msl, 2*hidden_state_size)

        x3 = self.linear1(x2)   # output shape is (batch, msl, 512).
        x4 = self.relu(x3)
        x5 = self.linear2(x4)   # output shape is (batch, msl, 256).
        x6 = self.relu(x5)
        x7 = self.linear3(x6)   # output shape is (batch, msl, num_class).
        return x7

def load_data(file_root, tokenizer):
    fp = open(file_root, 'r', encoding='utf-8')

    label_list = {
        '[PAD]': 0,
        'B-DT': 1,
        'I-DT': 2,
        'O': 3,
        'B-LC': 4,
        'I-LC': 5,
        'B-OG': 6,
        'I-OG': 7,
        'B-PS': 8,
        'I-PS': 9,
        'B-TI': 10,
        'I-TI': 11,
        'X': 12,
        '[CLS]': 13,
        '[SEP]': 14     
    }

    tokens = []
    ids = []
    labels = []

    max_len = 0
    while True:
        sentence = fp.readline()

        if not sentence:
            break

        sentence_split = sentence.split()

        if len(sentence_split) < 2:
            continue

        eoj = sentence_split[0] 
        tag = sentence_split[1]

        eoj_tk = tokenizer.tokenize(eoj)
        len_tk = len(eoj_tk)
        eoj_tkid = tokenizer.convert_tokens_to_ids(eoj_tk)
        
        label = [label_list['X']]*len_tk
        label[0] = label_list[tag]

        labels.append(label)
        tokens.append(eoj_tk)
        ids.append(eoj_tkid)

        if max_len < len_tk:
            max_len = len_tk

    nb_data = len(labels)
    for i in range(nb_data):
        len_data = len(labels[i])
        if len_data < max_len:
            diff = max_len-len_data
            labels[i] = labels[i] + ([0]*diff)
            tokens[i] = tokens[i] + (['[PAD]']*diff)    
            ids[i] = ids[i] + ([0]*diff)   

    return tokens, ids, labels, nb_data

def get_dataloader(x, y, train_rate=0.8, batch_size=32):
    train_size = int(nb_data*train_rate)

    x, y = torch.LongTensor(ids), torch.LongTensor(labels)
    train_x, train_y = x[:train_size], y[:train_size]
    test_x, test_y = x[train_size:], y[train_size:]

    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train(device, model, train_loader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0

    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        model.zero_grad()

        x = x.to(device)
        y = y.to(device)

        preds = model(x)
        preds_tr = torch.transpose(preds, 1, 2)

        loss = loss_fn(preds_tr, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().cpu().item()
    
    total_loss /= len(train_loader)

    return total_loss

def test(device, model, test_loader, loss_fn):
    softmax_fn = nn.Softmax(dim=2)
    all_preds = []
    all_labels = []

    total_loss = 0.0
    with torch.no_grad():
        model.eval()
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            preds_tr = torch.transpose(preds, 1, 2)
            preds = softmax_fn(preds)
            pred_label = torch.argmax(preds, dim=2)

            loss = loss_fn(preds_tr, y)

            mask = (y != 0)
            all_preds.append(pred_label[mask].cpu().numpy())
            all_labels.append(y[mask].cpu().numpy())
            total_loss += loss.detach().cpu().item()
    total_loss /= len(train_loader)

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    accuracy = accuracy_score(all_labels.flatten(), all_preds.flatten())
    precision = precision_score(all_labels.flatten(), all_preds.flatten(), average='macro', zero_division=0)
    recall = recall_score(all_labels.flatten(), all_preds.flatten(), average='macro', zero_division=0)
    f1 = f1_score(all_labels.flatten(), all_preds.flatten(), average='macro', zero_division=0)
    cm = confusion_matrix(all_labels.flatten(), all_preds.flatten())

    return total_loss, accuracy, precision, recall, f1, cm

if __name__ == '__main__':
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load tokenizer
    tokenizer = eojeol_BertTokenizer('./vocab.korean.rawtext.list', do_lower_case=False)
    vocab_size = len(tokenizer.vocab)

    # load data
    file_root = './ner_eojeol_label_per_line.txt'
    tokens, ids, labels, nb_data = load_data(file_root, tokenizer)    

    # load model
    nb_hidden_layer = 3
    embed_size = 768
    hidden_size = 512
    model_name = 'LSTM'

    if model_name == 'RNN':
        model = NER_model_RNN(vocab_size, embed_size, nb_hidden_layer, hidden_size)
    elif model_name == 'LSTM':
        model = NER_model_LSTM(vocab_size, embed_size, nb_hidden_layer, hidden_size)
    else:
        print('[Error] Please check your selected model name.')

    model = model.to(device)

    # prepare data
    train_loader, test_loader = get_dataloader(ids, labels)

    # training
    epochs = 50

    lr = 1e-4
    betas = (0.9, .999)
    eps = 1e-08
    weight_decay=0.01

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
       
    train_log = {
        'TRAIN_LOSS': [],
        'TEST_LOSS': [],
        'ACCURACY': [],
        'PRECISION': [],
        'RECALL': [],
        'F1': [],
        'CONFUSION_MATRIX': []
    }

    for epoch in range(epochs):
        train_loss = train(device, model, train_loader, loss_fn, optimizer)
        test_loss, accuracy, precision, recall, f1, cm = test(device, model, test_loader, loss_fn)

        print(f'[{epoch+1} epochs] Train Loss: {train_loss:.3f} / Test Loss: {test_loss:.3f} / Test Accuracy: {accuracy*100:.3f}%')

        train_log['TRAIN_LOSS'].append(train_loss)
        train_log['TEST_LOSS'].append(test_loss)
        train_log['ACCURACY'].append(accuracy)
        train_log['PRECISION'].append(precision)
        train_log['RECALL'].append(recall)
        train_log['F1'].append(f1)
        train_log['CONFUSION_MATRIX'].append(cm.tolist())

    torch.save(model.state_dict(), f'./{model_name}.pt')
    with open(f'./{model_name}_train_log.json', 'w') as json_file:
        json.dump(train_log, json_file, indent=4)