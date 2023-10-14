from helper import list_non_hidden
import numpy as np
import pickle
from tdnn_train import PR
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset



device = "cuda:0" if torch.cuda.is_available() else "cpu"


f = open('conf/off_center.txt', 'r')
OFF_CENTER = int(f.read())
f.close()


n_features = 1792



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
    
    def __init__(self, model_path):
        self.batch_size = 16
        
        self.model = PR()
        # self.model = nn.DataParallel(self.model)
        self.load_model(model_path)
        
        self.data = PRPredictDataset()
        self.loader = DataLoader(self.data, batch_size=self.batch_size, num_workers=2, shuffle=False)


    def load_model(self, model_path):
        self.model_path = model_path
        
        if torch.cuda.is_available() and 'cuda' in device:
            self.model.load_state_dict(torch.load(self.model_path))
        else:
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        
        self.model = self.model.to(device)
        self.model.eval()


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
            outputs = self.model.forward(inputs)
            _, pred = torch.max(outputs, dim=1)
            
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
        
        
        print('Results for', self.model_path, ':')
        print('')

        print('Precision, Recall, F1:')    
        print(prf1)
        print('')
        
        print('Confusion Matrix:')
        print(count)
        print('')
        
        log_filename = self.model_path.replace('.pt', '.log')
        f = open(log_filename, 'w')
        f.write('Results for ' + self.model_path + ' :\n')
        f.write('\n')
        f.write('Precision, Recall, F1:\n')
        f.write(str(prf1) + '\n')
        f.write('\n')
        f.write('Confusion Matrix:\n')
        f.write(str(count) + '\n')
        f.write('\n')
        f.close()
        
        # print('Running Time:')
        # print('Total:', self.total_time)
        # print('Time per sample:', self.total_time / (n+1))
        
        return prf1, count
            


if __name__ == '__main__':
    evaluation = PREvaluation('tdnn/model.pt')
    prf1, count = evaluation.predict()
        
        