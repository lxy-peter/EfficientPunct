from helper import list_non_hidden
import numpy as np
import pickle
import random
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset



if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print('Using', device)

f = open('conf/off_center.txt', 'r')
OFF_CENTER = int(f.read())
f.close()


class PRTrainDataset(Dataset):
    
    def __init__(self):
        
        f = open('embed_custom_train/1792/egs_txt/egs.txt', 'r')
        self.egs = f.read().split('\n')
        f.close()
        try:
            while True:
                self.egs.remove('')
        except ValueError:
            pass
        for i in range(len(self.egs)):
            eg = self.egs[i].split()
            eg[1] = int(eg[1])
            eg[2] = int(eg[2])
            self.egs[i] = eg
            
        self.feat_files = [file for file in list_non_hidden('embed_custom_train/1792/egs') if '.feat' in file]
        
        egs_by_class = [[] for _ in range(4)]
        for eg in self.egs:
            if eg[2] == 0:
                egs_by_class[0].append(eg)
            elif eg[2] == 1:
                egs_by_class[1].append(eg)
            elif eg[2] == 2:
                egs_by_class[2].append(eg)
            elif eg[2] == 3:
                egs_by_class[3].append(eg)
        
        most_freq = np.argmax([len(l) for l in egs_by_class])
        max_count = len(egs_by_class[most_freq])
        
        for i in set(range(4)) - {most_freq}:
            additional = self.sample(egs_by_class[i], max_count - len(egs_by_class[i]))
            egs_by_class[i].extend(additional)
        
        self.egs = [eg for egs_class in egs_by_class for eg in egs_class]
        
    def __len__(self):
        return len(self.egs)
    
    def __getitem__(self, idx):
        eg = self.egs[idx]
        
        try:
            f = open('embed_custom_train/1792/egs/' + eg[0] + '.feat', 'rb')
        except FileNotFoundError:
            return self.__getitem__(random.randint(0, len(self.egs) - 1))
        full = np.transpose(pickle.load(f))
        f.close()
        start = eg[1]
        end = start + 2*OFF_CENTER + 1 # non-inclusive
        label = eg[2]
    
        if start < 0:
            left_pad = np.zeros((1792, abs(start)))
            start = 0
        else:
            left_pad = np.zeros((1792, 0))
        
        if end > full.shape[1]:
            right_pad = np.zeros((1792, end - full.shape[1]))
            end = full.shape[1]
        else:
            right_pad = np.zeros((1792, 0))
        
        eg = torch.from_numpy(np.hstack((left_pad, full[:, start:end], right_pad))).type(torch.float32)
        assert eg.shape == (1792, 2*OFF_CENTER + 1)
        
        return {'Input' : eg, 'Label': label}

        
    def sample(self, seq, n):
        if n <= len(seq):
            return random.sample(seq, n)
        else:
            additional = seq * (n // len(seq))
            additional.extend(random.sample(seq, n % len(seq)))
            return additional
    

class PR(nn.Module):
    
    def __init__(self):
        super(PR, self).__init__()
        
        self.linear0 = nn.Linear(1792, 512)
        
        self.tdnn1 = nn.Conv1d(512, 256, kernel_size=9)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.tdnn2 = nn.Conv1d(256, 256, kernel_size=9, dilation=2)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.tdnn3 = nn.Conv1d(256, 128, kernel_size=5)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.tdnn4 = nn.Conv1d(128, 128, kernel_size=5, dilation=2)
        self.batchnorm4 = nn.BatchNorm1d(128)
        self.tdnn5 = nn.Conv1d(128, 64, kernel_size=7)
        self.batchnorm5 = nn.BatchNorm1d(64)
        self.tdnn6 = nn.Conv1d(64, 64, kernel_size=7, dilation=2)
        self.batchnorm6 = nn.BatchNorm1d(64)
        self.tdnn7 = nn.Conv1d(64, 4, kernel_size=5)
        self.batchnorm7 = nn.BatchNorm1d(4)
        
        self.linear1 = nn.Linear(243, 70)
        self.batchnorm8 = nn.BatchNorm1d(70)
        
        self.linear2 = nn.Linear(70, 1)
        
    
    def forward(self, x):
        
        x = torch.transpose(x, 1, 2)
        x = self.linear0(x)
        x = F.relu(x)
        x = torch.transpose(x, 1, 2)
        
        x = self.tdnn1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)
        x = self.tdnn2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.tdnn3(x)
        x = F.relu(x)
        x = self.batchnorm3(x)
        x = self.tdnn4(x)
        x = F.relu(x)
        x = self.batchnorm4(x)
        x = self.tdnn5(x)
        x = F.relu(x)
        x = self.batchnorm5(x)
        x = self.tdnn6(x)
        x = F.relu(x)
        x = self.batchnorm6(x)
        x = self.tdnn7(x)
        x = F.relu(x)
        x = self.batchnorm7(x)
        
        x0 = self.linear1(x[:, 0, :])
        x0 = F.relu(x0)
        x0 = self.batchnorm8(x0)
    
        x1 = self.linear1(x[:, 1, :])
        x1 = F.relu(x1)
        x1 = self.batchnorm8(x1)
        
        x2 = self.linear1(x[:, 2, :])
        x2 = F.relu(x2)
        x2 = self.batchnorm8(x2)
        
        x3 = self.linear1(x[:, 3, :])
        x3 = F.relu(x3)
        x3 = self.batchnorm8(x3)
        
        x0 = self.linear2(x0)
        x1 = self.linear2(x1)
        x2 = self.linear2(x2)
        x3 = self.linear2(x3)
        
        x = torch.hstack((x0, x1, x2, x3))
        
        return x
    

def train(load_model=None):
    batch_size = 128
    
    model = PR()
    print(model)
    
    if load_model and device == 'cpu':
        model.load_state_dict(torch.load(load_model, map_location=torch.device('cpu')))
    elif load_model and 'cuda' in device:
        model.load_state_dict(torch.load(load_model))
    elif load_model and 'mps' in device:
        model.load_state_dict(torch.load(load_model, map_location=torch.device('mps')))
    print('Beginning training from', load_model)
        
    model.eval()
    
    print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
    
    train_data = PRTrainDataset()
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=2, shuffle=True)
    
    epochs = 25
    for epoch in range(0, epochs):
        total_loss = 0
        data_count = 0
    
        for i, data in enumerate(train_loader, 0):
            
            inputs = data['Input'].to(device)
            labels = data['Label'].to(device)
            
            data_count += batch_size
            
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            
            loss = criterion(outputs, labels)
            total_loss += float(loss)
            
            if i % 10 == 0:
                print("Epoch", epoch+1,
                      "| Processing data", data_count, "/", len(train_data),
                      "| Loss/sample:", float(loss) / batch_size)
                if i % 100000 == 0:
                    torch.save(model.state_dict(), "tdnn/model.pt")
            
            loss.backward()
            optimizer.step()
            
        torch.save(model.state_dict(), 'tdnn/tdnn-epoch' + str(epoch+1) + '.pt')
        print('')
        print('**************************************************************')
        print('Epoch ' + str(epoch+1) + ' loss:', total_loss)
        print('**************************************************************')
        print('')


if __name__ == '__main__':
    train()


