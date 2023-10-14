from helper import list_non_hidden
import numpy as np
import pickle
from tdnn_train import PR
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel



device = "cuda:0" if torch.cuda.is_available() else "cpu"



f = open('conf/off_center.txt', 'r')
OFF_CENTER = int(f.read())
f.close()



n_features = 1792



class BERTFineTuneForPunct(nn.Module):
    
    def __init__(self):
        super(BERTFineTuneForPunct, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.lin1 = nn.Linear(768, 1024)
        self.lin2 = nn.Linear(1024, 4)
    
    def forward(self, x):
        
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        
        x = torch.squeeze(x)
        
        return x



class PRPredictDataset(Dataset):
    
    def __init__(self):
        f = open('embed_custom_predict/' + str(n_features) + '/egs_txt/egs.txt', 'r')
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
        
    def __len__(self):
        return len(self.egs)
    
    def __getitem__(self, idx):
        eg = self.egs[idx]
        utt = eg[0]
        
        f = open('embed_custom_predict/1792/egs/' + utt + '.feat', 'rb')
        full = np.transpose(pickle.load(f))
        f.close()
        
        start = eg[1]
        end = start + 2*OFF_CENTER + 1 # non-inclusive
        label = eg[2]
    
        if start < 0:
            left_pad = np.zeros((n_features, abs(start)))
            start = 0
        else:
            left_pad = np.zeros((n_features, 0))
        
        if end > full.shape[1]:
            right_pad = np.zeros((n_features, end - full.shape[1]))
            end = full.shape[1]
        else:
            right_pad = np.zeros((n_features, 0))
        
        eg = torch.from_numpy(np.hstack((left_pad, full[:, start:end], right_pad))).type(torch.float32)
        assert eg.shape == (n_features, 2*OFF_CENTER + 1)
        
        return {'Input': eg, 'Label': label, 'Uttword': (utt, start)}
    


class PREvaluation():
    
    def __init__(self, tdnn_path, bert_path):
        # self.batch_size = 16
        self.batch_size = 3
        
        self.tdnn = PR()
        self.bert = BERTFineTuneForPunct()
        
        self.load_models(tdnn_path, bert_path)
        
        self.data = PRPredictDataset()
        self.loader = DataLoader(self.data, batch_size=self.batch_size, num_workers=16, shuffle=False)


    def load_models(self, tdnn_path, bert_path):
        self.tdnn_path = tdnn_path
        self.bert_path = bert_path
        
        
        if torch.cuda.is_available() and 'cuda' in device:
            self.tdnn.load_state_dict(torch.load(self.tdnn_path))
        else:
            self.tdnn.load_state_dict(torch.load(self.tdnn_path, map_location=torch.device('cpu')))
        
        self.tdnn = self.tdnn.to(device)
        self.tdnn.eval()
        
        
        if torch.cuda.is_available() and 'cuda' in device:
            self.bert.load_state_dict(torch.load(self.bert_path))
        else:
            self.bert.load_state_dict(torch.load(self.bert_path, map_location=torch.device('cpu')))
        
        self.bert = self.bert.to(device)
        self.bert.eval()


    def predict(self):
        
        comma_TP = 0
        comma_FP = 0
        comma_TN = 0
        comma_FN = 0
        
        fs_TP = 0
        fs_FP = 0
        fs_TN = 0
        fs_FN = 0
        
        qm_TP = 0
        qm_FP = 0
        qm_TN = 0
        qm_FN = 0
        
        count = np.zeros((4,4), dtype=int)
        
        data_count = 0
        
        for i, data in enumerate(self.loader, 0):
            if i % 100 == 0:
                print('Processing sample', data_count, '/', len(self.data))
            
            inputs = data['Input'].to(device)
            labels = data['Label']
            uttword = data['Uttword']
            
            data_count += self.batch_size
            
            
            
            
            bert_embed = torch.unsqueeze(inputs[:, :768, 150], dim=1)
            
            # for b in range(int(bert_embed.shape[0])):
            #     same = []
            #     prev = bert_embed[b, :, 0]
            #     start = 0
            #     end = 0
            #     for idx in range(1, int(bert_embed.shape[2])):
            #         current = bert_embed[b, :, idx]
            #         if not torch.equal(prev, current):
            #             prev = current
            #             same.append((start, end))
            #             start = idx
            #         else:
            #             end = idx
            #     print(same)
            #     print('')
            
            
            
            tdnn_outputs = self.tdnn.forward(inputs)
            bert_outputs = self.bert.forward(bert_embed)
            
            tdnn_probs = F.softmax(tdnn_outputs, dim=1)
            bert_probs = F.softmax(bert_outputs, dim=1)
            assert tdnn_probs.shape == bert_probs.shape
            
            w_tdnn = 0.7
            w_bert = 0.3
            final_probs = w_tdnn*tdnn_probs + w_bert*bert_probs
            _, pred = torch.max(final_probs, dim=1)
            
            for i in range(len(pred)):
                
                # Comma
                if pred[i] == 2 and labels[i] == 2:
                    comma_TP += 1
                elif pred[i] == 2 and labels[i] != 2:
                    comma_FP += 1
                elif pred[i] != 2 and labels[i] != 2:
                    comma_TN += 1
                elif pred[i] != 2 and labels[i] == 2:
                    comma_FN += 1
                        
                # Full stops
                if pred[i] == 1 and labels[i] == 1:
                    fs_TP += 1
                elif pred[i] == 1 and labels[i] != 1:
                    fs_FP += 1
                elif pred[i] != 1 and labels[i] != 1:
                    fs_TN += 1
                elif pred[i] != 1 and labels[i] == 1:
                    fs_FN += 1
                    
                # Question marks
                if pred[i] == 3 and labels[i] == 3:
                    qm_TP += 1
                elif pred[i] == 3 and labels[i] != 3:
                    qm_FP += 1
                elif pred[i] != 3 and labels[i] != 3:
                    qm_TN += 1
                elif pred[i] != 3 and labels[i] == 3:
                    qm_FN += 1
                
                pred_int = int(pred[i])
                label_int = int(labels[i])
                count[label_int, pred_int] += 1
        
        overall_TP = comma_TP + fs_TP + qm_TP
        overall_FP = comma_FP + fs_FP + qm_FP
        overall_FN = comma_FN + fs_FN + qm_FN       
        
        try:
            comma_precision = comma_TP / (comma_TP + comma_FP) * 100
        except ZeroDivisionError:
            comma_precision = None
        try:
            comma_recall = comma_TP / (comma_TP + comma_FN) * 100
        except ZeroDivisionError:
            comma_recall = None
        if comma_precision is None or comma_recall is None:
            comma_f1 = None
        else:
            comma_f1 = 2 * comma_precision * comma_recall / (comma_precision + comma_recall)
        
        try:
            fs_precision = fs_TP / (fs_TP + fs_FP) * 100
        except ZeroDivisionError:
            fs_precision = None
        try:
            fs_recall = fs_TP / (fs_TP + fs_FN) * 100
        except ZeroDivisionError:
            fs_recall = None
        if fs_precision is None or fs_recall is None:
            fs_f1 = None
        else:
            fs_f1 = 2 * fs_precision * fs_recall / (fs_precision + fs_recall)
        
        try:
            qm_precision = qm_TP / (qm_TP + qm_FP) * 100
        except ZeroDivisionError:
            qm_precision = None
        try:
            qm_recall = qm_TP / (qm_TP + qm_FN) * 100
        except ZeroDivisionError:
            qm_recall = None
        if qm_precision is None or qm_recall is None:
            qm_f1 = None
        else:
            qm_f1 = 2 * qm_precision * qm_recall / (qm_precision + qm_recall)
        
        try:
            overall_precision = overall_TP / (overall_TP + overall_FP) * 100
        except ZeroDivisionError:
            overall_precision = None
        try:
            overall_recall = overall_TP / (overall_TP + overall_FN) * 100
        except ZeroDivisionError:
            overall_recall = None
        if overall_precision is None or overall_recall is None:
            overall_f1 = None
        else:
            overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        
        prf1 = {'comma': (comma_precision, comma_recall, comma_f1),
                'fs': (fs_precision, fs_recall, fs_f1),
                'qm': (qm_precision, qm_recall, qm_f1),
                'overall': (overall_precision, overall_recall, overall_f1)}
        
        N = np.sum(count)
        count = count / N * 100
        
        
        print('Results for', self.tdnn_path, ' ensembled with BERT:')
        print('')

        print('Precision, Recall, F1:')    
        print(prf1)
        print('')
        
        print('Confusion Matrix:')
        print(count)
        print('')
        
        log_filename = self.tdnn_path.replace('.pt', '.log')
        f = open(log_filename, 'w')
        f.write('Results for ' + self.tdnn_path + ' :\n')
        f.write('\n')
        f.write('Precision, Recall, F1:\n')
        f.write(str(prf1) + '\n')
        f.write('\n')
        f.write('Confusion matrix:\n')
        f.write(str(count) + '\n')
        f.write('\n')
        f.close()
        
        # print('Running Time:')
        # print('Total:', self.total_time)
        # print('Time per sample:', self.total_time / (n+1))
        
        return prf1, count
            


if __name__ == '__main__':
    evaluation = PREvaluation('tdnn/model.pt', 'bert/bert.pt')
    prf1, count = evaluation.predict()
        