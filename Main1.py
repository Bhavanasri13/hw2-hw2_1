import torch.optim as optim
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
import os
import sys
import json
import re
from torch.utils.data import DataLoader, Dataset
import pickle
import models

torch.backends.cudnn.enabled = False

# Define calculate_loss function with NaN checks
def calculate_loss(x, y, lengths, loss_fn):
    batch_size = len(x)
    predict_cat = None
    groundT_cat = None
    flag = True

    for batch in range(batch_size):
        predict = x[batch]
        ground_truth = y[batch]
        seq_len = lengths[batch] - 1

        predict = predict[:seq_len]
        ground_truth = ground_truth[:seq_len]

        if flag:
            predict_cat = predict
            groundT_cat = ground_truth
            flag = False
        else:
            predict_cat = torch.cat((predict_cat, predict), dim=0)
            groundT_cat = torch.cat((groundT_cat, ground_truth), dim=0)

    loss = loss_fn(predict_cat, groundT_cat)

    # Check for NaN in the loss
    if torch.isnan(loss):
        print("NaN detected in loss calculation.")
        loss = torch.tensor(0.0, requires_grad=True).cuda()  # Set the loss to zero and ensure it requires grad

    return loss

def train(model, epoch, train_loader, loss_func, optimizer, epochs_n):
    model.train()
    print(f"Epoch {epoch+1}/{epochs_n} started...")
    total_loss = 0

    for batch_idx, (avi_feats, ground_truths, lengths) in enumerate(train_loader):
        avi_feats, ground_truths = avi_feats.cuda(non_blocking=True), ground_truths.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        seq_logProb, seq_predictions = model(avi_feats, target_sentences=ground_truths, mode='train', tr_steps=epoch)
        
        ground_truths = ground_truths[:, 1:]
        loss = calculate_loss(seq_logProb, ground_truths, lengths, loss_func)

        if torch.isnan(loss):
            print("NaN loss detected, skipping this batch.")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch: {epoch+1}, Loss: {avg_loss:.4f}')
    return avg_loss

def dictonaryFunc(word_min): 
    #Json Loader#
    with open('training_label.json', 'r') as f:
        file = json.load(f)

    word_count = {}
    for d in file:
        for s in d['caption']:
            word_sentence = re.sub('[.!,;?]]', ' ', s).split()
            for word in word_sentence:
                word = word.replace('.', '') if '.' in word else word
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1
    #Dictonary#    
    dictonary = {}
    for word in word_count:
        if word_count[word] > word_min:
            dictonary[word] = word_count[word]
    useful_tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    i2w = {i + len(useful_tokens): w for i, w in enumerate(dictonary)}
    w2i = {w: i + len(useful_tokens) for i, w in enumerate(dictonary)}
    for token, index in useful_tokens:
        i2w[index] = token
        w2i[token] = index
    return i2w, w2i, dictonary
# Minibatch function
def minibatch(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data)
    avi_data = torch.stack(avi_data, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths
def evaluate(test_loader, model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (avi_feats, ground_truths, lengths) in enumerate(test_loader):
            avi_feats, ground_truths = avi_feats.cuda(non_blocking=True), ground_truths.cuda(non_blocking=True)
            seq_logProb, seq_predictions = model(avi_feats, mode='inference')

            ground_truths = ground_truths[:, 1:]
            min_len = min(seq_predictions.size(1), ground_truths.size(1))
            seq_predictions = seq_predictions[:, :min_len]
            ground_truths = ground_truths[:, :min_len]

            predicted = seq_predictions
            mask = (ground_truths != 0)
            correct += ((predicted == ground_truths) & mask).sum().item()
            total += mask.sum().item()

    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Evaluation Accuracy: {accuracy:.2f}%")
# Define the s_split function before it's used in annotate
def s_split(sentence, dictonary, w2i):  
    sentence = re.sub(r'[.!,;?]', ' ', sentence).split()
    for i in range(len(sentence)):
        if sentence[i] not in dictonary:
            sentence[i] = 3  # Assuming 3 is the index for <UNK>
        else:
            sentence[i] = w2i[sentence[i]]
    sentence.insert(0, 1)  # Insert <SOS> token at the beginning
    sentence.append(2)  # Append <EOS> token at the end
    return sentence
# annotate function (Ensure this is defined before it's called)
def annotate(label_file, dictonary, w2i):
    label_json = label_file
    annotated_caption = []
    with open(label_json, 'r') as f:
        label = json.load(f)
    for d in label:
        for s in d['caption']:
            s = s_split(s, dictonary, w2i)
            annotated_caption.append((d['id'], s))
    return annotated_caption
# avi function to load avi files (Ensure this is defined first)
def avi(files_dir):
    avi_data = {}
    training_feats = files_dir
    files = os.listdir(training_feats)
    for file in files:
        value = np.load(os.path.join(training_feats, file))
        avi_data[file.split('.npy')[0]] = value
    return avi_data

    
class Dataprocessor(Dataset):
    def __init__(self, label_file, files_dir, dictonary, w2i):
        self.label_file = label_file
        self.files_dir = files_dir
        self.avi = avi(label_file)
        self.w2i = w2i
        self.dictonary = dictonary
        self.data_pair = annotate(files_dir, dictonary, w2i)

    def __len__(self):
        return len(self.data_pair)

    def __getitem__(self, idx):
        avi_file_name, sentence = self.data_pair[idx]
        data = torch.Tensor(self.avi[avi_file_name])
        data += torch.Tensor(data.size()).random_(0, 2000) / 10000
        return data, torch.Tensor(sentence)
class test_dataloader(Dataset):
    def __init__(self, test_data_path):
        self.avi = []
        files = os.listdir(test_data_path)
        for file in files:
            key = file.split('.npy')[0]
            value = np.load(os.path.join(test_data_path, file))
            self.avi.append([key, value])
    def __len__(self):
        return len(self.avi)
    def __getitem__(self, idx):
        return self.avi[idx]
def test(test_loader,model,i2w):
        
        # set model to evaluation(testing) mode
        model.eval()
        ss = []
        for batch_idx, batch in enumerate(test_loader):
            # prepare data
            id, avi_feats = batch
            if torch.cuda.is_available():
                avi_feats = avi_feats.cuda()
            else:
                avi_feats=avi_feats
                
            id, avi_feats = id, Variable(avi_feats).float()

            # start inferencing process
            seq_logProb, seq_predictions = model(avi_feats, mode='inference')
            test_predictions = seq_predictions
#             result = [[x if x != '<UNK>' else 'something' for x in s] for s in test_predictions]
#             result = [' '.join(s).split('<EOS>')[0] for s in result]

            result = [[i2w[x.item()] if i2w[x.item()] != '<UNK>' else 'something' for x in s] for s in test_predictions]
            result = [' '.join(s).split('<EOS>')[0] for s in result]
        
            rr = zip(id, result)
            for r in rr:
                ss.append(r)
        return ss

def main():
    label_file = 'training_data/feat'
    files_dir = 'training_label.json'
    i2w, w2i, dictonary = dictonaryFunc(4)
    train_dataset = Dataprocessor(label_file, files_dir, dictonary, w2i)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=5, shuffle=True, num_workers=8, collate_fn=minibatch)

    test_dataset = Dataprocessor(label_file, files_dir, dictonary, w2i)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=8, collate_fn=minibatch)

    epochs_n = 100
    ModelSaveLoc = 'SavedModel'
    with open('i2wData.pickle', 'wb') as f:
        pickle.dump(i2w, f)

    x = len(i2w) + 4
    if not os.path.exists(ModelSaveLoc):
        os.mkdir(ModelSaveLoc)

    loss_fn = nn.CrossEntropyLoss()
    encode = models.EncoderNet()
    decode = models.DecoderNet(512, x, x, 1024, 0.3)
    model = models.ModelMain(encoder=encode, decoder=decode).cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    start = time.time()
    for epoch in range(epochs_n):
        avg_loss = train(model, epoch, train_loader=train_dataloader, loss_func=loss_fn, optimizer=optimizer, epochs_n=epochs_n)
        evaluate(test_dataloader, model)
        scheduler.step()

    end = time.time()
    torch.save(model, f"{ModelSaveLoc}/model0.h5")
    print(f"Training finished. Elapsed time: {end - start:.3f} seconds.")

if __name__ == "__main__":
    main()
