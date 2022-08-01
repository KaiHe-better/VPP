import sys
sys.path.append('..')
# from .base_model import FewShotREModel
import torch
from torch import nn

class FewShotREModel(nn.Module):
    def __init__(self, sentence_encoder):
        '''
        sentence_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.sentence_encoder = nn.DataParallel(sentence_encoder)
        self.cost = nn.CrossEntropyLoss()
    
    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))
    
class Pair(FewShotREModel):
    
    def __init__(self, sentence_encoder, hidden_size=768):
        FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, 2)
        self.drop = nn.Dropout()

    def forward(self, batch, N, K, total_Q):
        '''
        batch: (batch_size, total_Q, N, K, 2)
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        _, logits = self.sentence_encoder(batch)
        logits = self.fc(logits[:,0,:])
        logits = logits.view(-1, total_Q, N, K, 2)
        logits = logits.mean(3) # (-1, total_Q, N, 2)
        # logits_na, _ = logits[:, :, :, 0].min(2, keepdim=True) # (-1, totalQ, 1)
        logits = logits[:, :, :, 1] # (-1, total_Q, N)
        # logits = torch.cat([logits, logits_na], 2) # (B, total_Q, N + 1 (+1=None of above class ) )
        # _, pred = torch.max(logits.view(-1, N + 1), 1)
        _, pred = torch.max(logits.view(-1, N), 1)
        return logits, pred


