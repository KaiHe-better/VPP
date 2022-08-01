import sys
sys.path.append('..')
from .base_model import FewShotREModel
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F



class FewShotREModel_raw(nn.Module):
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



class Proto(FewShotREModel_raw):
    
    def __init__(self, sentence_encoder, dot=False):
        FewShotREModel_raw.__init__(self, sentence_encoder)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.dot = dot

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support, query, N, K, total_Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        _, support_emb = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
        _, query_emb = self.sentence_encoder(query) # (B * total_Q, D)
        hidden_size = support_emb.size(-1)
        support = self.drop(support_emb[:,0,:])
        query = self.drop(query_emb[:,0,:])
        support = support.view(-1, N, K, hidden_size) # (B, N, K, D)
        query = query.view(-1, total_Q, hidden_size) # (B, total_Q, D)

        B = support.size(0) # Batch size
         
        # Prototypical Networks 
        # Ignore NA policy
        support = torch.mean(support, 2) # Calculate prototype for each class # (B, N, D)
        logits = self.__batch_dist__(support, query) # (B, total_Q, N)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2) # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N+1), 1)
        return logits, pred

    
    
    
