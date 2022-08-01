import sys
sys.path.append('..')
from .base_model import FewShotREModel
import fewshot_re_kit
import torch
from torch import nn


class No_Question(FewShotREModel):

    def __init__(self, sentence_encoder):
        FewShotREModel.__init__(self, sentence_encoder)
        self.drop = nn.Dropout()

    def forward(self, input, placehold, N, K, total_Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        logits = self.sentence_encoder(input)
        logits = logits.view(-1, N, K, N)
        logits = logits.mean(2) # (-1, N, N)
        _, pred = torch.max(logits, 2)
        return logits, pred
