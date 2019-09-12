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

# spacy分割英文句子
TEXT = data.Field(tokenize='spacy', include_lengths=True)
LABEL = data.LabelField(dtype=torch.float)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
RANDOM_SEED = 123
train_data, valid_data = train_data.split(random_state=random.seed(RANDOM_SEED))

print("Num Train: ", len(train_data))
print("Num Valid: ", len(valid_data))
print("Num Test: ", len(test_data))

#构建单词表
TEXT.build_vocab(train_data, max_size=20000, vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

TEXT.vocab.freqs.most_common(20)

#查看单词表， itos：int-》string itoi：string-》int
print(TEXT.vocab.itos[:10])
print(LABEL.vocab.stoi)

#创建iterators
train_loader, valid_loader, test_loader = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size=64,
    sort_within_batch=True,
    device=torch.device('cpu'))


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim) #for LSTM
        #self.rnn = nn.RNN(embedding_dim, hidden_dim) #for RNN
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text, text_length):
        embedded = self.embedding(text)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, text_length)
        output, (hidden, cell) = self.rnn(packed) #for LSTM
        #output, hidden = self.rnn(packed)  #for RNN 
        return self.fc(hidden.squeeze(0)).view(-1) 


    
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 1  
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

torch.manual_seed(RANDOM_SEED)
model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(model)  
params = model.state_dict()
for k, v in params.items():
    print(k)

# valid
def compute_binary_accuracy(model, data_loader):
    model.eval()
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            text, text_length = batch_data.text
            out = model(text, text_length)
            # binary classify 
            predicted_labels = (torch.sigmoid(out) > 0.5).long()
            num_examples += batch_data.label.size(0)
            correct_pred += (predicted_labels == batch_data.label.long()).sum()
        return correct_pred.float()/num_examples * 100
    
start_time = time.time()

#train
for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, batch_data in enumerate(train_loader):
        text, text_length = batch_data.text
        ### FORWARD AND BACK PROP
        out = model(text, text_length)
        loss = F.binary_cross_entropy_with_logits(out, batch_data.label)
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        if not batch_idx % 10:
            print("Epoch:", (epoch+1),'/',NUM_EPOCHS,' Batch:', batch_idx, '/', len(train_loader), ' Loss: %.4f' % loss.item())

    with torch.set_grad_enabled(False):
        print('training accuracy: %.2f' % compute_binary_accuracy(model, train_loader), "%")
        print('valid accuracy:    %.2f' % compute_binary_accuracy(model, valid_loader), "%")
        print('test accuracy:     %.2f' % compute_binary_accuracy(model, test_loader),"%")
        
    print('Time elapsed: %.2f' % (time.time() - start_time), '(s)')
    print('---------------------------------------------------')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



