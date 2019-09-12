'''
Created on Jul 18, 2019

@author: mitom
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchtext import data
from torchtext import datasets
import time
import random
from unicodedata import bidirectional

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
RANDOM_SEED = 123
train_data, valid_data = train_data.split(random_state=random.seed(RANDOM_SEED))

print("Num Train: ", len(train_data))
print("Num Valid: ", len(valid_data))
print("Num Test: ", len(test_data))

#构建单词表
TEXT.build_vocab(train_data, max_size=25000, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

TEXT.vocab.freqs.most_common(20)

print(TEXT.vocab.itos[:10])
print(LABEL.vocab.stoi)

#创建iterators
train_loader, valid_loader, test_loader = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size=64,
    device=torch.device('cpu'))

print('Train')
for batch in train_loader:
    print("Text matrix size: ", batch.text.size())
    print("Target vector size: ", batch.label.size())
    break
    
print('\nValid:')
for batch in valid_loader:
    print("Text matrix size: ", batch.text.size())
    print("Target vector size: ", batch.label.size())
    print(batch.text)
    break
    
print('\nTest:')
for batch in test_loader:
    print("Text matrix size: ", batch.text.size())
    print("Target vector size: ", batch.label.size())
    break


class RNN_MOVIE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden.squeeze(0))

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0)).view(-1) 


    
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
OUTPUT_DIM = 1  
NUM_EPOCHS = 10

torch.manual_seed(RANDOM_SEED)
model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)  

def compute_binary_accuracy(model, data_loader):
    model.eval()
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            out = model(batch_data.text)
            # binary classify 
            predicted_labels = (torch.sigmoid(out) > 0.5).long()
            num_examples += batch_data.label.size(0)
            correct_pred += (predicted_labels == batch_data.label.long()).sum()
        return correct_pred.float()/num_examples * 100
    
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, batch_data in enumerate(train_loader):
        
        ### FORWARD AND BACK PROP
        logits = model(batch_data.text)
        cost = F.binary_cross_entropy_with_logits(logits, batch_data.label)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 10:
            print("Epoch:", (epoch+1),'/',NUM_EPOCHS,' Batch:', batch_idx, '/', len(train_loader), ' Cost: %.4f',cost.item())

    with torch.set_grad_enabled(False):
        print('training accuracy: %.4f' % compute_binary_accuracy(model, train_loader), "%")
        print('valid accuracy: %.4f' % compute_binary_accuracy(model, valid_loader), "%")
        print('test accuracy: %.4f' % compute_binary_accuracy(model, test_loader),"%")
        
    print('Time elapsed: %.2f' % (time.time() - start_time), '(s)')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



