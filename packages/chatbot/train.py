import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from packages.chatbot.nltk_utils import bag_of_words, tokenize, stem
from packages.chatbot.model import NeuralNet
import streamlit as st

import io
import pandas as pd


class ChatDataset(Dataset):

    def __init__(self, X_train, y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

class train:
    def __init__(self, intents):
        intents = pd.read_json(intents)['intents']
        intents = pd.json_normalize(intents)
        #show intents
        st.dataframe(intents)
        #convert to dict
        
        self.all_words, self.tags, self.xy = self.tokenise(intents)

        #clean up words
        self.all_words, self.tags = self.clean_up(self.all_words, self.tags)

        #create training data
        self.create_train_data()

        #hyperparameters
        self.hyperparameters()

        #create dataset
        self.dataset = ChatDataset(self.X_train, self.y_train)

    def tokenise(self, intents: pd.DataFrame) -> list:
        all_words = []
        tags = []
        xy = []

        # loop through each sentence in our intents patterns
        for index in range(len(intents)):
            tag = intents.iloc[index]['tag']
            # add to tag list
            tags.append(tag)
            for pattern in intents.iloc[index]['patterns']:
                # tokenize each word in the sentence
                w = tokenize(pattern)
                # add to our words list
                all_words.extend(w)
                # add to xy pair
                xy.append((w, tag))

        return all_words, tags, xy
    
    def read_intents_file(self) -> dict:
        with open(self.intents_file_path, 'r') as f:
            intents = json.load(f)
        return intents
    
    def clean_up(self, all_words: list, tags: list):
        # stem and lower each word
        ignore_words = ['?', '.', '!','@','#','$','%','^','&','*','(',')','-','_','+','=','{','}','[',']','|','\\',':',';','"','\'','<','>','/','~','`']
        all_words = [stem(w) for w in all_words if w not in ignore_words]
        # remove duplicates and sort
        all_words = sorted(set(all_words))
        tags = sorted(set(tags))

        return all_words, tags

    def create_train_data(self):
        # create training data
        X_train = []
        y_train = []
        for (pattern_sentence, tag) in self.xy:
            # X: bag of words for each pattern_sentence
            bag = bag_of_words(pattern_sentence, self.all_words)
            X_train.append(bag)
            # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
            label = self.tags.index(tag)
            y_train.append(label)

        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def hyperparameters(self):
        # Hyper-parameters 
        self.num_epochs = 1000
        self.batch_size = 8
        self.learning_rate = 0.001
        self.input_size = len(self.X_train[0])
        self.hidden_size = 8
        self.output_size = len(self.tags)

    def train(self):

        train_loader = DataLoader(dataset=self.dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=0)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = NeuralNet(self.input_size, self.hidden_size, self.output_size).to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Train the model
        for epoch in range(self.num_epochs):
            for (words, labels) in train_loader:
                words = words.to(self.device)
                labels = labels.to(dtype=torch.long).to(self.device)
        
                # Forward pass
                self.outputs = self.model(words)
               # if y would be one-hot, we must apply
                # labels = torch.max(labels, 1)[1]
                self.loss = self.criterion(self.outputs, labels)
        
                # Backward and optimize
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
        
            if (epoch+1) % 100 == 0:
                st.write(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {self.loss.item():.4f}')


        st.success(f'final loss: {self.loss.item():.4f}')

    @property
    def model_file(self):
        self.data = {
                "model_state": self.model.state_dict(),
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "output_size": self.output_size,
                "all_words": self.all_words,
                "tags": self.tags
            }
        buffer = io.BytesIO()
        torch.save(self.data, buffer)

        return buffer
    
    @property
    def save_model(self, id):
        FILE = "packages/chatbot/temp/data.pth"
        torch.save(self.data, FILE)
