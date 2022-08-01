import torch
import torch.nn as nn
import numpy as np
from .. import network

from transformers import RobertaModel, RobertaTokenizer, RobertaForSequenceClassification, RobertaForMaskedLM


class RobertaSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, Ptuing=False, entity_pseudo_token=None, cat_entity_rep=False):
        nn.Module.__init__(self)
        self.model = RobertaForMaskedLM.from_pretrained(pretrain_path, output_hidden_states=True)
        self.max_length = max_length
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.cat_entity_rep = cat_entity_rep
        if Ptuing:
            self.tokenizer.add_special_tokens({'additional_special_tokens': [entity_pseudo_token]})
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            
    def forward(self, inputs, Ptuing=False):
        if not self.cat_entity_rep:
            logits = self.model(inputs['word'], attention_mask=inputs['mask'])
            return logits["logits"], logits["hidden_states"][-1]  # (125,128,50265), (125,128,768)
        
        else:
            outputs = self.model(inputs['word'], attention_mask=inputs['mask'])
            tensor_range = torch.arange(inputs['word'].size()[0])
            h_state = outputs[0][tensor_range, inputs["pos1"]]
            t_state = outputs[0][tensor_range, inputs["pos2"]]
            state = torch.cat((h_state, t_state), -1)
            return state
                
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        
        def getIns(bped, bpeTokens, tokens, L):
            resL = 0
            tkL = " ".join(tokens[:L])
            bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
            if bped.find(bped_tkL) == 0:
                resL = len(bped_tkL.split())
            else:
                tkL += " "
                bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
                if bped.find(bped_tkL) == 0:
                    resL = len(bped_tkL.split())
                else:
                    raise Exception("Cannot locate the position")
            return resL

        s = " ".join(raw_tokens)
        sst = self.tokenizer.tokenize(s)
        headL = pos_head[0]
        headR = pos_head[-1] + 1
        hiL = getIns(" ".join(sst), sst, raw_tokens, headL)
        hiR = getIns(" ".join(sst), sst, raw_tokens, headR)

        tailL = pos_tail[0]
        tailR = pos_tail[-1] + 1
        tiL = getIns(" ".join(sst), sst, raw_tokens, tailL)
        tiR = getIns(" ".join(sst), sst, raw_tokens, tailR)

        E1b = 'madeupword0000'
        E1e = 'madeupword0001'
        E2b = 'madeupword0002'
        E2e = 'madeupword0003'
        ins = [(hiL, E1b), (hiR, E1e), (tiL, E2b), (tiR, E2e)]
        ins = sorted(ins)
        
        pE1 = 0
        pE2 = 0
        pE1_ = 0
        pE2_ = 0
        for i in range(0, 4):
            sst.insert(ins[i][0] + i, ins[i][1])
            if ins[i][1] == E1b:
                pE1 = ins[i][0] + i
            elif ins[i][1] == E2b:
                pE2 = ins[i][0] + i
            elif ins[i][1] == E1e:
                pE1_ = ins[i][0] + i
            else:
                pE2_ = ins[i][0] + i
                
        pos1_in_index = pE1 + 1
        pos2_in_index = pE2 + 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(sst)

        # padding
        # while len(indexed_tokens) < self.max_length:
        #     indexed_tokens.append(1)
        # indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # # mask
        # mask = np.zeros((self.max_length), dtype=np.int32)
        # mask[:len(sst)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        return indexed_tokens, pos1_in_index, pos2_in_index

    def tokenize_question(self, raw_tokens, entity):
        raw_tokens= raw_tokens.replace("{ENTITY1}", entity[0]).replace("{ENTITY2}", entity[1])
        tokens = self.tokenizer.tokenize(raw_tokens)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        return indexed_tokens


class RobertaPAIRSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, num_labels=2):
        nn.Module.__init__(self)
        self.roberta = RobertaForSequenceClassification.from_pretrained(pretrain_path, num_labels=num_labels)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.max_length = max_length

    def forward(self, inputs):
        x = self.roberta(inputs['word'], attention_mask=inputs['mask'])[0]
        return x

    def tokenize(self, raw_tokens, pos_head, pos_tail):

        def getIns(bped, bpeTokens, tokens, L):
            resL = 0
            tkL = " ".join(tokens[:L])
            bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
            if bped.find(bped_tkL) == 0:
                resL = len(bped_tkL.split())
            else:
                tkL += " "
                bped_tkL = " ".join(self.tokenizer.tokenize(tkL))
                if bped.find(bped_tkL) == 0:
                    resL = len(bped_tkL.split())
                else:
                    raise Exception("Cannot locate the position")
            return resL

        s = " ".join(raw_tokens)
        sst = self.tokenizer.tokenize(s)
        headL = pos_head[0]
        headR = pos_head[-1] + 1
        hiL = getIns(" ".join(sst), sst, raw_tokens, headL)
        hiR = getIns(" ".join(sst), sst, raw_tokens, headR)

        tailL = pos_tail[0]
        tailR = pos_tail[-1] + 1
        tiL = getIns(" ".join(sst), sst, raw_tokens, tailL)
        tiR = getIns(" ".join(sst), sst, raw_tokens, tailR)

        E1b = 'madeupword0000'
        E1e = 'madeupword0001'
        E2b = 'madeupword0002'
        E2e = 'madeupword0003'
        ins = [(hiL, E1b), (hiR, E1e), (tiL, E2b), (tiR, E2e)]
        ins = sorted(ins)
        for i in range(0, 4):
            sst.insert(ins[i][0] + i, ins[i][1])
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(sst)
        return indexed_tokens

    def tokenize_question(self, raw_tokens, entity):
        raw_tokens= raw_tokens.replace("{ENTITY1}", entity[0]).replace("{ENTITY2}", entity[1])
        tokens = self.tokenizer.tokenize(raw_tokens)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        return indexed_tokens
