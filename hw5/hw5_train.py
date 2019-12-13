import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd
import numpy as np
import spacy
import re
from gensim.models import Word2Vec
from torch.nn.utils.rnn import pad_sequence
import sys
import os
import pickle
import multiprocessing

def load_data(train_x, train_y, test_x):
    train_clean = clean_data(train_x['comment'])
    test_clean = clean_data(test_x['comment'])
    train_clean_drop = [s for s in train_clean if s != '']
    test_clean_fill = [s if s != '' else "good" for s in test_clean]
    train_label = [train_y['label'][i] for i, s in enumerate(train_clean) if s != '']
    train_df_c = pd.DataFrame({'id': range(len(train_clean_drop)), 'comment': train_clean_drop})
    test_df_c = pd.DataFrame({'id': range(len(test_clean_fill)), 'comment': test_clean_fill}) 
    corp = train_clean.copy()
    corp.extend(test_clean)
    return train_df_c, test_df_c, train_label, corp
    
def cleaning(doc):
    txt = [token.lemma_ for token in doc if not token.is_stop]
    txt = [w for w in txt if w != ' ']
    return ' '.join(txt)

def clean_data(data):
    nlp = spacy.load("en_core_web_sm") 
    rm_user = (re.sub("@user", '', str(row)).lower() for row in data)
    rm_url = (re.sub("url", '', str(row)).lower() for row in rm_user)
    brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in rm_url)
    txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]
    return txt

def word2vec(corp):
    cores = multiprocessing.cpu_count()
    dictionary = [w.split() for w in corp]
    w2v_model = Word2Vec(size=256, window=5, min_count=1, workers=cores)
    w2v_model.build_vocab(dictionary) 
    w2v_model.train(dictionary, total_examples=w2v_model.corpus_count, epochs=1000)
    w2v_model.save("./model/train_word2vec_256.model")
    return w2v_model
    
def prepare_sequence(seq, word_to_ix):
    seq = seq.split(' ')
    idxs = [word_to_ix[w] for w in seq if w != ' ']
    return torch.tensor(idxs, dtype=torch.long)

def make_dict(train_df_clean, test_df_clean):
    word_to_ix = dict()
    corpus = pd.concat([train_df_clean['comment'], test_df_clean['comment']], ignore_index=True)
    for row in corpus:
        sen = str(row).split(' ')
        for word in sen:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix

def build_pretrained_dict(words_dict, model):
    dict_size = len(words_dict)
    vec_dim = 256
    tensor = torch.zeros([dict_size, vec_dim])
    for word in words_dict:
        idx = words_dict[word]
        vec = model.wv[word]
        tensor[idx,:] = torch.FloatTensor(vec)
    return tensor

class dataset(Dataset):
    def __init__(self, train_x, train_y, word_to_ix):
        self.train_x = train_x['comment']
        self.label = train_y
        self.w2v_model = w2v_model
        
    def __getitem__(self, index):
        sentence = self.train_x[index]
        sentence = prepare_sequence(sentence, word_to_ix)
        return torch.LongTensor(sentence), self.label[index]

    def __len__(self):
        return self.train_x.shape[0]
    
class dataset_test(Dataset):
    def __init__(self, test_x, word_to_ix):
        self.test_x = test_x['comment']
    def __getitem__(self, index):
        sentence = self.test_x[index]
        sentence = prepare_sequence(sentence, word_to_ix)
        return torch.LongTensor(sentence)
    def __len__(self):
        return self.test_x.shape[0]

def add_padding(data):
    sents = [s[0] for s in data]
    labels = [s[1] for s in data]
    sort_sents = sorted(sents, key=lambda x: len(x), reverse=True)
    pad_sent = pad_sequence(sents, batch_first=True, padding_value=0)
    return pad_sent, torch.LongTensor(labels)

def add_padding_test(data):
    sents = data
    sort_sents = sorted(sents, key=lambda x: len(x), reverse=True)
    pad_sent = pad_sequence(sents, batch_first=True, padding_value=0)
    return pad_sent
    
class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, pretrained_vec):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)  
        self.word_embeddings.weight.data.copy_(pretrained_vec)
        self.word_embeddings.weight.requires_grad = False
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        self.hidden2class = nn.Sequential( 
            nn.Linear(4*hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, tagset_size),
            nn.Sigmoid(),
        )        

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        norm_lstm_out = lstm_out[:, -1, :].view(lstm_out.size(0), -1)
        reverse_lstm_out = lstm_out[:, 0, :].view(lstm_out.size(0), -1)
        lstm_out = torch.cat((norm_lstm_out, reverse_lstm_out), 1)
        tag_space = self.hidden2class(lstm_out)
        return tag_space
    
def train(model, train_dataloader, EPOCH):
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)

    if not os.path.exists(f'./hist_model'):
        os.mkdir(f'./hist_model')

    for epoch in range(1, 1+EPOCH):
        model.train()
        correct = 0
        total_loss = 0
        for batch_idx, (sentence, label) in enumerate(train_dataloader):
            data, label = sentence.to(device), label.to(device)
            optimizer.zero_grad() 
            output = model(data)
            loss = loss_function(output, label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(output.data, 1)
            predicted = torch.squeeze(predicted)
            correct += (predicted == label).sum().item()    
        
        if epoch % 10 == 0:
            print("epoch :", epoch)
            accu = correct/len(train_loader.dataset)
            print("TRAIN", "accuracy:", accu, "loss:", total_loss)
            torch.save(model.state_dict(), f'./hist_model/{epoch}')

def predict(PATH, test_loader, pretrained_vec, voc_size):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    pred_model = LSTMClassifier(256, 128, voc_size, 2, pretrained_vec)
    pred_model = pred_model.to(device)
    pred_model.load_state_dict(torch.load(PATH))
    pred_model.eval()
    
    prediction = []
    with torch.no_grad():
        for batch_idx, sentence in enumerate(test_loader):
            sentence = sentence.to(device)
            out = pred_model(sentence)
            _,pred_label = torch.max(out,1)
            prediction.extend(pred_label.cpu().numpy())
    return prediction

def out(result, out_file):
    df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
    df.to_csv(out_file, index=False)
        
if __name__ == '__main__':
    train_x = pd.read_csv(sys.argv[1])
    train_y = pd.read_csv(sys.argv[2])
    test_x = pd.read_csv(sys.argv[3])
    
    train_df_c, test_df_c, train_label, corp = load_data(train_x, train_y, test_x)
    word_to_ix = make_dict(train_df_c, test_df_c)
    
    with open('./model/new_word_to_ix.pickle', 'wb') as file:
        pickle.dump(word_to_ix, file)

    w2v_model = word2vec(corp)
    pretrained_vec = build_pretrained_dict(word_to_ix, w2v_model)

    train_dataset = dataset(train_df_c, train_label, w2v_model)
    test_dataset = dataset_test(test_df_c, word_to_ix)
    train_loader = DataLoader(train_dataset, batch_size = 256, shuffle=False, collate_fn = add_padding)
    test_loader = DataLoader(test_dataset, batch_size = 256, shuffle=False, collate_fn = add_padding_test)
    
    model = LSTMClassifier(256, 128, len(word_to_ix), 2, pretrained_vec)
    EPOCH = 100
    train(model, train_loader, EPOCH)
    PATH = f'./hist_model/{EPOCH}'
    if not os.path.exists('./result'):
        os.mkdir('./result')
    prediction = predict(PATH, test_loader, pretrained_vec, len(word_to_ix))
    output = out(prediction, './result/train_prediction.csv')