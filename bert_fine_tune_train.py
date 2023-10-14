import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
    


class BERTFineTuneForPunct(nn.Module):
    
    def __init__(self):
        super(BERTFineTuneForPunct, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.lin1 = nn.Linear(768, 1024)
        self.lin2 = nn.Linear(1024, 4)
    
    
    def bert_last_hidden(self, x):
        return self.bert(**x).last_hidden_state[:, 1:-1, :]
    
    def forward(self, x):
        
        x = self.bert_last_hidden(x)
        print(x.shape)
        raise RuntimeError
        
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        
        return x

