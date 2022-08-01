
import sys
sys.path.append('..')
import os
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from .prompt_encoder import PromptEncoder
from fewshot_re_kit.utils import BalancedDataParallel



WEIGHTS_NAME = "pytorch_model.bin"

def load_state_dict(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} #filter out unnecessary keys 
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


class FewShotREModel(nn.Module):
    def __init__(self, sentence_encoder, gpu1_data_num=10):
        nn.Module.__init__(self)
        if torch.cuda.device_count()>1 :
            self.sentence_encoder = BalancedDataParallel(gpu1_data_num, sentence_encoder, dim=0)
        else:
            self.sentence_encoder = nn.DataParallel(sentence_encoder)

        self.vocab_len = len(self.sentence_encoder.module.tokenizer.get_vocab())
        
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

    def loss(self, model_res_dic, label, promot_label, N, K, Q, proto_weight=1, prompt_weight=0):
        total_loss = 0
        proto_loss=0
        if proto_weight!=0:
            proto_logits = model_res_dic["proto_logits"]    
            N = proto_logits.size(-1)
            proto_loss = F.cross_entropy(proto_logits.view(-1, N), label.view(-1))
            total_loss += proto_weight+proto_loss
            
        prompt_loss=0
        if prompt_weight!=0:
            prompt_logits = model_res_dic["prompt_logits"]  # (B, N, N, num_labels)
            N = prompt_logits.size(-1)
            prompt_loss = F.cross_entropy(prompt_logits.view(-1, N), label.view(-1))
            
            total_loss += prompt_weight*prompt_loss 
        return total_loss, proto_loss, prompt_loss

    def accuracy(self, model_res_dic, label):
        final_pred = model_res_dic["final_pred"]
        return torch.mean((final_pred.view(-1) == label.view(-1)).type(torch.FloatTensor))
    

class Proto_Quesion(FewShotREModel):
    
    def __init__(self, sentence_encoder, dot=False, cat_entity_rep=False,  gpu1_data_num=10, 
                 Ptuing=False, template=(3,3,3,3), pseudo_token='[PROMPT]', prompt_encoder_ckpt=None):
        FewShotREModel.__init__(self, sentence_encoder, gpu1_data_num)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.dot = dot
        self.cat_entity_rep = cat_entity_rep
        self.Ptuing = Ptuing
        self.tokenizer= self.sentence_encoder.module.tokenizer
        
        if self.Ptuing:
            if "tokenization_roberta.RobertaTokenizer" in str(type(self.tokenizer)):
                self.embeddings = self.sentence_encoder.module.model.roberta.embeddings
                embeddings_size = self.embeddings.word_embeddings.embedding_dim
            else:
                if "tokenization_bert.BertTokenizer" in str(type(self.tokenizer)):
                     # For bert-based p-tuning
                    # self.embeddings = self.sentence_encoder.module.bert.bert.embeddings
                    self.embeddings = self.sentence_encoder.module.bert.embeddings
                    embeddings_size = 768
                else:
                    self.embeddings = self.sentence_encoder.module.model.model.shared
                    embeddings_size = self.embeddings.embedding_dim
                
            self.template = template
            self.spell_length = sum(self.template)
            self.prompt_encoder = PromptEncoder(self.spell_length, embeddings_size)
            
            if (prompt_encoder_ckpt is not None) and ("CQARE" in prompt_encoder_ckpt):
                pretrained_dict = torch.load(os.path.join(prompt_encoder_ckpt, WEIGHTS_NAME))
                # load_state_dict(self.prompt_encoder, pretrained_dict)
                # print("loading prompt encoder success !")
                print("using random initialized prompt encoder !")
                
            if pseudo_token not in self.tokenizer.get_vocab():
                self.tokenizer.add_special_tokens({'additional_special_tokens': [pseudo_token]})
            self.pseudo_token_id = self.tokenizer.get_vocab()[pseudo_token]
        
    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3) # S:(B, 1, N, D)，Q:(B, N*total_Q, 1, D)
    
    def get_embed_inputs(self, sentences, params=None):
        sentences_for_embedding = sentences.clone()
        if self.Ptuing:
            sentences_for_embedding[(sentences == self.pseudo_token_id)] = self.sentence_encoder.module.tokenizer.unk_token_id
            raw_embeds = self.embeddings(sentences_for_embedding)
            bz = sentences.shape[0]
            blocked_indices = (sentences == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
            replace_embeds = self.prompt_encoder(torch.LongTensor(list(range(self.spell_length))).to(sentences.device))

            for bidx in range(bz):
                for i in range(self.spell_length):
                    raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        else:
            raw_embeds = self.embeddings(sentences_for_embedding)
        return raw_embeds
    
    def get_supportProto_and_query(self, support_emb, query_emb, N, K, total_Q):
        hidden_size = support_emb.size(-1)
        
        support = self.drop(support_emb) 
        query = self.drop(query_emb)
        
        support = support.view(-1, N, N, K, hidden_size) # (B, N, N, K, D)
        support_proto = torch.mean(support, 3) # Calculate prototype for each class  # (B, N, temp_N, D)
        support_proto =torch.mean(support_proto, 1) # Calculate prototype for each class  # (B, temp_N, D)
        
        query = query.view(-1, N*total_Q, hidden_size) # (B, N*total_Q, D)
        
        return support_proto, query
         
    def proto_task(self, support_emb, query_emb, N, K, total_Q):
        support_proto, query = self.get_supportProto_and_query(support_emb, query_emb, N, K, total_Q) # support_proto (B, N, D)   query (B，total_Q, D)

        proto_logits = self.__batch_dist__(support_proto, query).view(-1, N, total_Q, N)  
        proto_logits = torch.mean(proto_logits, dim=1).view(-1, N) # logits  (B* total_Q, N)
        proto_pred = torch.argmax(proto_logits, -1)
        
        # new_data = sorted(proto_logits.squeeze())
        # max_gap = abs(new_data[-1] - new_data[0])
        # min_gap = abs(new_data[1] - new_data[0])
        # with open("./10-5-proto_gap.txt", "a") as f:
        #     f.write(str((float(max_gap.data), float(min_gap.data))))
        #     f.write("\n")
            
        return proto_pred, proto_logits
     
    def prompt_mask_task(self, support_mask_logit, query_mask_logit, N, K, Q, temp_N, total_Q): # support_mask_logit, query_mask_logit :(N*N*K,50265)
        
        support_mask_logit, query_mask_logit = self.get_supportProto_and_query(support_mask_logit, query_mask_logit, N, K, total_Q) # support_prompt (B, temp_N, D)   query (B，total_Q, D)
        prompt_logits = self.__batch_dist__(support_mask_logit, query_mask_logit).view(-1, N, total_Q, N)  
        prompt_logits = torch.mean(prompt_logits, dim=1).view(-1, N) # logits  (B* total_Q, N)
        prompt_pred = torch.argmax(prompt_logits, -1)
        
        # new_data = sorted(prompt_logits.squeeze())
        # max_gap = abs(new_data[-1] - new_data[0])
        # min_gap = abs(new_data[1] - new_data[0])
        # with open("./prompt_gap_bart.txt", "a") as f:
        #     f.write(str((float(max_gap.data), float(min_gap.data))))
        #     f.write("\n")
        
        return prompt_logits, prompt_pred # [N, N], [N]
    
    def record_res_(self, support_mask_logit, query_mask_logit):
        print()
        no_list = ["No", "NO", "no", "wrong", "negative", "Ġdisagree", "ĠWrong", "Ġwrong", "Ġdisapproval", 
                        "Ġindefinite", "Ġincorrect", "Ġincorrectly", "Ġuntrue", "Ġimproper","False", "false", 
                        "Ġinaccurate", "Error", "error", "Ġerroneous"]

        yes_list =  ["Yes", "YES", "yes", "Right", "right", "True", "true", "Great", "great", "Good", "good", 
                        "Sure", "sure", "agree", "positive", "Certain", "certain", "Exactly", "Correct", "correct"]

        no_index_list = self.tokenizer.convert_tokens_to_ids(no_list)
        yes_index_list = self.tokenizer.convert_tokens_to_ids(yes_list)
        
        support_data = nn.functional.softmax(support_mask_logit, dim=-1).squeeze(0)
        query_data = nn.functional.softmax(query_mask_logit, dim=-1).squeeze(0)
        
        for support_item in support_data:
            yes_prob = support_item[yes_index_list].sum()
            no_prob = support_item[no_index_list].sum()
            
            if yes_prob>no_prob:
                print("yes")
            else:
                print("no")
            # print("support yes_prob", yes_prob)
            # print("support no_prob", no_prob)

            # topk_prob, topk_pred = support_item.topk(100)
            # print("support topk word", self.tokenizer.decode(topk_pred).split())
            # print()
        print("================================================================")
        
        for query_item in query_data:
            yes_prob = query_item[yes_index_list].sum()
            no_prob = query_item[no_index_list].sum()
            
            if yes_prob>no_prob:
                print("yes")
            else:
                print("no")
                
            # print("query yes_prob", yes_prob)
            # print("query no_prob", no_prob)

            # topk_prob, topk_pred = query_item.topk(100)
            # print("query topk word", self.tokenizer.decode(topk_pred).split())
            # print()
        print("================================================================")
            
            
        # # with open("./others/prompt_support_5w_10.txt", "a") as f:
        # with open("./prompt_support_5w_10.txt", "a") as f:
        #     f.write(str(support_data.tolist()))
        #     f.write("\n")
        
        # query_data = nn.functional.softmax(query_mask_logit, dim=-1)
        # # with open("./others/prompt_query_5w_10.txt", "a") as f:
        # with open("./prompt_query_5w_10.txt", "a") as f:
        #     f.write(str(query_data.tolist()))
        #     f.write("\n")
    
    def view_data(self, inputs, N, temp_N, K):
        """batch size=1 for data DataParallel"""
        size = inputs["word"].size()
        gpu_num = torch.cuda.device_count()
        if N*temp_N*K==size[0] and gpu_num>1:
            assert size[0]%gpu_num==0  
            for i, v in inputs.items():
                if i=="sent_len":
                    inputs[i]=v.view(gpu_num, int(size[0]/gpu_num)).view(size[0])
                elif i=="word_embed" :
                    inputs[i]=v.view((gpu_num, int(size[0]/gpu_num), size[-1], 768)).view((size[0], size[-1], 768))
                else:
                    inputs[i]=v.view((gpu_num, int(size[0]/gpu_num), size[-1])).view((size[0], size[-1]))
        
        return inputs
    
    def get_mask_eos_logits(self, inputs, N, temp_N, K):
        if self.Ptuing:
            input_id = inputs["word"]
            input_embeding = self.get_embed_inputs(input_id)
            inputs["word_embed"] = input_embeding
        
        mlm_logits, last_hs = self.sentence_encoder(inputs, Ptuing=self.Ptuing)  # (B*N*N, 128, 50365/768)
                    
        masked_index = inputs['word'].eq(self.tokenizer.mask_token_id)  # [B*N*N, 128] 
        mask_logit = mlm_logits[masked_index, :]
        
        two_eos_index = inputs['word'].eq(self.tokenizer.eos_token_id) # [B*N*N, 128] , each have 2 eos(True)
        eos_logit = last_hs[two_eos_index,:].view(last_hs.size(0), -1, last_hs.size(-1))[:,0,:]  # [B*N*N, 768]
        
        if self.cat_entity_rep:
            tensor_range = torch.arange(inputs['word'].size()[0])
            h_state = mlm_logits[tensor_range, inputs["pos1"]]
            t_state = mlm_logits[tensor_range, inputs["pos2"]]
            mask_logit = mask_logit+ h_state + t_state
                
        return mask_logit, eos_logit # [B*N, 768]
    
    def forward(self, support, query, N, K, Q, temp_N=None, na_rate=0, prompt_weight=0.5, proto_weight=0.5):
        if proto_weight==0 and prompt_weight==0:
            raise Exception("at least choose one task !")
        
        if temp_N is None:
            temp_N = N
        total_Q = (Q*temp_N+na_rate*Q)

        support_mask_logit, S_eos_logit = self.get_mask_eos_logits(support, N, N, K)   # [B*N, 768]
        query_mask_logit, Q_eos_logit = self.get_mask_eos_logits(query, N, temp_N, K)   # [B*N+NA*(B*N), 768]
        
        
        
        if proto_weight!=0:
            proto_pred, proto_logits = self.proto_task(S_eos_logit, Q_eos_logit, N, K, total_Q) # proto_logits:(B*total_Q, N)   proto_pred:(temp_N)
        else:
            proto_logits, proto_pred = 0, 0
        
        if prompt_weight!=0:    
            prompt_logits, prompt_pred = self.prompt_mask_task(support_mask_logit, query_mask_logit, N, K, Q, temp_N, total_Q) # [N, N], [N]
        else:
            prompt_logits, prompt_pred=0, 0
            
            
        if prompt_weight!=0 and proto_weight==0:
            final_pred = prompt_pred
        elif proto_weight!=0 and prompt_weight==0:
            final_pred = proto_pred
        else:
            combined_logits = prompt_logits.view(-1, N) + proto_logits.view(-1, N)
            final_pred = torch.argmax(combined_logits, 1) # (batch_size, temp_N)
            
        model_res_dic = {"proto_logits":proto_logits, 
                         "prompt_logits":prompt_logits, 
                         "final_pred":final_pred,
                         }
        
        return model_res_dic