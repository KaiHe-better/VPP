import torch
import torch.nn as nn
import numpy as np

from transformers import BartTokenizer, BartModel, BartTokenizerFast, BartForConditionalGeneration, BartForSequenceClassification


def shift_tensors_right(input_tensors: torch.Tensor, replace_tensor: torch.Tensor):
    """
    Shift input ids one token to the right.
    """
    shifted_tensors = input_tensors.new_zeros(input_tensors.shape)
    shifted_tensors[:, 1:] = input_tensors[:, :-1].clone()
    shifted_tensors[:, 0] = replace_tensor

    return shifted_tensors


class Classifier(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, num_classes, input_dim=768, inner_dim=200, pooler_dropout=0.2):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = nn.ReLU()
        self.dropout = nn.Dropout(pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    def init_param(self):
        nn.init.xavier_normal_(self)



class BART_SentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, Ptuing, entity_pseudo_token, dropout=0):
        nn.Module.__init__(self)
        self.model = BartForConditionalGeneration.from_pretrained(pretrain_path, dropout=dropout)
        self.max_length = max_length
        self.tokenizer = BartTokenizerFast.from_pretrained(pretrain_path)
        if Ptuing:
            self.tokenizer.add_special_tokens({'additional_special_tokens': [entity_pseudo_token]})

    def forward(self, inputs, Ptuing=False):
        if Ptuing:
            decoder_word_embed = shift_tensors_right(inputs['word_embed'], 
                                                     self.model.model.shared(torch.LongTensor([self.tokenizer.bos_token_id]).to(inputs['word_embed'].device) ))
            decoder_attention_mask = shift_tensors_right(inputs['mask'], torch.ones((1)).long().to(inputs['mask'].device))
            logits = self.model(inputs_embeds=inputs['word_embed'], 
                                attention_mask=inputs['mask'], 
                                decoder_inputs_embeds = decoder_word_embed,
                                decoder_attention_mask = decoder_attention_mask,
                                output_hidden_states=True)
        else:
            logits = self.model(inputs['word'], attention_mask=inputs['mask'], output_hidden_states=True)
        
        mlm_logit = logits["logits"]
        last_hidden_state = logits["decoder_hidden_states"][-1]
            
        return mlm_logit, last_hidden_state    # (B*N*B, 128, 50365), (B*N*B, 128, 768)

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

        # # padding
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


class BART_PairSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, Ptuing, entity_pseudo_token, dropout=0):
        nn.Module.__init__(self)
        self.model = BartForSequenceClassification.from_pretrained(pretrain_path, dropout=dropout)
        self.max_length = max_length
        self.tokenizer = BartTokenizerFast.from_pretrained(pretrain_path)
        if Ptuing:
            self.tokenizer.add_special_tokens({'additional_special_tokens': [entity_pseudo_token]})

    def forward(self, inputs, Ptuing=False):
        if Ptuing:
            decoder_word_embed = shift_tensors_right(inputs['word_embed'], 
                                                     self.model.model.shared(torch.LongTensor([self.tokenizer.bos_token_id]).to(inputs['word_embed'].device) ))
            decoder_attention_mask = shift_tensors_right(inputs['mask'], torch.ones((1)).long().to(inputs['mask'].device))
            logits = self.model(inputs_embeds=inputs['word_embed'], 
                                attention_mask=inputs['mask'], 
                                decoder_inputs_embeds = decoder_word_embed,
                                decoder_attention_mask = decoder_attention_mask,
                                output_hidden_states=True)
        else:
            logits = self.model(inputs['word'], attention_mask=inputs['mask'], output_hidden_states=True)
        
        mlm_logit = logits["logits"]
        last_hidden_state = logits["decoder_hidden_states"][-1]
            
        return mlm_logit, last_hidden_state    # (B*N*B, 128, 50365), (B*N*B, 128, 768)

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

        # # padding
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

