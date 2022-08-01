import torch
import torch.nn as nn
import numpy as np
from .. import network

from transformers import BertTokenizer, BertModel, BertForSequenceClassification, BertForMaskedLM


class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=False, mask_entity=False):
        nn.Module.__init__(self)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.bos_token = "[CLS]"
        self.tokenizer.eos_token = "[SEP]"
        self.cat_entity_rep = cat_entity_rep
        self.mask_entity = mask_entity
        
        if pretrain_path =='./pretrain/CP/pytorch_model.bin':
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.cp_MASK_head = nn.Linear(768, len(self.tokenizer))
            self.CP_flag = True
        else:
            self.bert = BertForMaskedLM.from_pretrained(pretrain_path)
            self.CP_flag = False
            
        if pretrain_path =='./pretrain/CP/pytorch_model.bin':
            self.bert.load_state_dict(torch.load(pretrain_path)["bert-base"])
            print("We load "+ pretrain_path+" to train!")
        else:
            print("Path is None, We use Bert-base!")
            

    def forward(self, inputs, Ptuing=False):
        if Ptuing:
            logits = self.bert(inputs_embeds=inputs['word_embed'], 
                                attention_mask=inputs['mask'], output_hidden_states=True)
        else:
            logits = self.bert(inputs['word'], attention_mask=inputs['mask'], output_hidden_states=True)
        
        
        
        if self.CP_flag:
            mlm_logit = self.cp_MASK_head(logits[0])   # (N*N*K, 128, 768)
            last_hidden_state = logits[0]  # (N*N*K, 128, 768)
        else:
            mlm_logit = logits["logits"]
            last_hidden_state = logits["hidden_states"][-1]
        
        return mlm_logit, last_hidden_state    

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = []
        cur_pos = 0
        pos1_in_index = 1
        pos2_in_index = 1
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                tokens += ['[unused4]']
            else:
                tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        # while len(indexed_tokens) < self.max_length:
        #     indexed_tokens.append(0)
        # indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        # mask = np.zeros((self.max_length), dtype=np.int32)
        # mask[:len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1

    def tokenize_question(self, raw_tokens, entity):
        raw_tokens= raw_tokens.replace("{ENTITY1}", entity[0]).replace("{ENTITY2}", entity[1])
        tokens = self.tokenizer.tokenize(raw_tokens)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        return indexed_tokens

        
        
class BERTPAIRSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length):
        nn.Module.__init__(self)
        # self.bert = BertForSequenceClassification.from_pretrained(pretrain_path, num_labels=2)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # if pretrain_path =='./pretrain/CP/pytorch_model.bin':
        #     # CP need transformers paskage published from CP paper to install 
        #     self.bert = BertModel.from_pretrained('bert-base-uncased')
        #     self.cp_MASK_head = nn.Linear(768, len(self.tokenizer))
        #     self.CP_flag = True
        # else:
        self.bert = BertForMaskedLM.from_pretrained(pretrain_path)
        self.CP_flag = False
        
        # if pretrain_path is not None and pretrain_path != "None":
        #     self.bert.load_state_dict(torch.load(pretrain_path)["bert-base"])
        #     print("We load "+ pretrain_path+" to train!")
        # else:
        #     print("Path is None, We use Bert-base!")
            
        

    def forward(self, inputs):
        x = self.bert(inputs['word'], token_type_ids=inputs['seg'], attention_mask=inputs['mask'])[0]
        return "", x

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        # tokens = ['[CLS]']
        tokens = []
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)

            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        # mask = np.zeros((self.max_length), dtype=np.int32)
        # mask[:len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)
        
        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1
