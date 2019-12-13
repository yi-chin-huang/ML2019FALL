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
import pickle

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

def prepare_sequence(seq, word_to_ix):
    seq = seq.split(' ')
    idxs = [word_to_ix[w] for w in seq if w != ' ']
    return torch.tensor(idxs, dtype=torch.long)

class dataset_test(Dataset):
    def __init__(self, test_x, word_to_ix):
        self.test_x = test_x['comment']
    def __getitem__(self, index):
        sentence = self.test_x[index]
        sentence = prepare_sequence(sentence, word_to_ix)
        return torch.LongTensor(sentence)
    def __len__(self):
        return self.test_x.shape[0]

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

def predict(test_loader, pretrained_vec):
    FILE = './model/lstm.model'
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    pred_model = LSTMClassifier(256, 128, 16255, 2, pretrained_vec)
    pred_model = pred_model.to(device)
    pred_model.load_state_dict(torch.load(FILE))
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
    
    test_x = pd.read_csv(sys.argv[1])
    test_clean = clean_data(test_x['comment'])
    test_clean_fill = [s if s != '' else "good" for s in test_clean]   
    test_df_c = pd.DataFrame({'id': range(len(test_clean_fill)), 'comment': test_clean_fill})
    with open('./model/word_to_ix.pickle', 'rb') as file:
        word_to_ix = pickle.load(file)
    pretrained_vec = torch.load('model/pretrained_vec')
    test_dataset = dataset_test(test_df_c, word_to_ix)
    test_loader = DataLoader(test_dataset, batch_size = 256, shuffle=False, collate_fn = add_padding_test)

    prediction = predict(test_loader, pretrained_vec)
    output = out(prediction, sys.argv[2])