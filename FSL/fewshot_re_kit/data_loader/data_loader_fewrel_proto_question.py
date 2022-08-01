import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json



class FewRelDataset_Proto_Question(data.Dataset):
    def __init__(self, name, encoder, N, K, Q, na_rate, root, encoder_name, 
                 Ptuing, pattern, P_template_format, Yes_No_token_dic, pseudo_token='[PROMPT]'):
        self.root = root
        path = os.path.join(root, name + ".json")
        self.json_data = json.load(open(path))
        if type(self.json_data)==dict:
            self.classes = list(self.json_data.keys())
        
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.max_length = encoder.max_length
        self.tokenizer = encoder.tokenizer
        self.encoder_name = encoder_name
        self.CLS = self.tokenizer.bos_token_id
        self.SEP = self.tokenizer.eos_token_id
        self.MASK = self.tokenizer.mask_token_id
        self.question_marker_id = self.tokenizer.convert_tokens_to_ids("?")
        self.vocab_len = len(self.tokenizer)
        
        self.prompt_label = torch.zeros(self.N, self.N).long()  # (N, temp_N）
        for i in range(self.N):
            for j in range(self.N):
                if i==j:
                    self.prompt_label[i][j]= list(Yes_No_token_dic.keys()).index("yes")
                else:
                    self.prompt_label[i][j]= list(Yes_No_token_dic.keys()).index("no")
                    
        self.name = name
        pid2name_path = os.path.join(root, "pid2name.json")
        self.pid2name = json.load(open(pid2name_path, 'r'))

        name2id_path = os.path.join(root, "name2id_all.json")
        self.name2id = json.load(open(name2id_path, 'r'))

        id2question_path = os.path.join(root, "id2question_all.json")
        self.id2question = json.load(open(id2question_path, 'r'))

        self.Ptuing = Ptuing
        self.pattern = pattern
        if self.Ptuing:
            self.pseudo_token = pseudo_token
            if pseudo_token not in self.tokenizer.get_vocab():
                self.tokenizer.add_special_tokens({'additional_special_tokens': [pseudo_token]})
            self.pseudo_token_id = self.tokenizer.get_vocab()[pseudo_token]
            self.template = P_template_format

    def __len__(self):
        return 1000000000

    def __relation_name2ID__(self, relation_name):
        if "wiki" in self.name:
            relation_id = self.name2id[self.pid2name[relation_name][0]]
        elif "pubmed" in self.name:
            relation_id = self.name2id[relation_name]
        else:
            raise NotImplementedError
        return relation_id

    def __additem__(self, d, data_tuple):
        word_list, mask_list, seg_list, sent_len_list, pos1_list, pos2_list = data_tuple
        for word, mask, seg, pos1, pos2 in zip(word_list, mask_list, seg_list, pos1_list, pos2_list):
            d['word'].append(word)
            d['mask'].append(mask)
            d['seg'].append(seg)
            d['pos1'].append(pos1)
            d['pos2'].append(pos2)
        d['sent_len'].append(sent_len_list)
            
    def __pid2name__(self, pid):
        if "wiki" in self.name:
            relation = self.pid2name[pid][0]
        elif "pubmed" in self.name:
            relation = pid
        else:
            raise NotImplementedError
        return relation

    def __get_question(self, question_name, entity):
        if self.Ptuing:
            if self.pattern =="p1":
                return_question = self.pseudo_token * self.template[0] + entity[0] \
                                    + self.pseudo_token * self.template[1] + entity[1]  \
                                    + self.pseudo_token * self.template[2] + ' '.join(self.__pid2name__(question_name).split('_')) \
                                    + self.pseudo_token * self.template[3] 
               
            elif self.pattern =="p2":
                return_question =     self.pseudo_token * self.template[0] + ' '.join(self.__pid2name__(question_name).split('_'))  \
                                    + self.pseudo_token * self.template[1] + entity[0] \
                                    + self.pseudo_token * self.template[2] + entity[1] \
                                    + self.pseudo_token * self.template[3] 
            elif self.pattern =="p3":
                return_question = self.pseudo_token * self.template[0] + entity[0] \
                                    + self.pseudo_token * self.template[1] + ' '.join(self.__pid2name__(question_name).split('_')) \
                                    + self.pseudo_token * self.template[2] + entity[1] \
                                    + self.pseudo_token * self.template[3] 
            elif self.pattern =="p4":
                return_question = self.pseudo_token * self.template[0]  \
                                    + self.pseudo_token * self.template[1]  \
                                    + self.pseudo_token * self.template[2]  \
                                    + self.pseudo_token * self.template[3]  \
                                    + entity[0]  + entity[1] +' '.join(self.__pid2name__(question_name).split('_')) 
            elif self.pattern =="p5":
                return_question = self.pseudo_token * self.template[0]  \
                                    + entity[0] \
                                    + self.pseudo_token * self.template[1]  \
                                    + entity[1] \
                                    + self.pseudo_token * self.template[2]  \
                                    +' '.join(self.__pid2name__(question_name).split('_'))      
            else:
                raise Exception("error !")                           
                                
                                
            return self.tokenizer.encode(return_question, add_special_tokens=False)
        else:
            relation_id = self.__relation_name2ID__(question_name)
            return_question = self.id2question[str(relation_id)] 
            return_question = self.encoder.tokenize_question(return_question, entity)
            return return_question
    
    def __getraw__(self, item):
        word, pos1, pos2 = self.encoder.tokenize(item['tokens'], item['h'][2][0], item['t'][2][0])
        # word = torch.tensor(word).long()
        pos1 = torch.tensor(pos1).long()
        pos2 = torch.tensor(pos2).long()
        return word, pos1, pos2 

    def contact_question(self, data_set, entity_list, target_classes, pos1, pos2):
        word_tensor_list=[]
        mask_tensor_list=[]
        seg_tensor_list=[]
        sent_len_list=[]
        pos1_list=[]
        pos2_list=[]
        
        for class_name in target_classes:
            for N_index, N_way_data_list in enumerate(data_set):
                for k_index, word in enumerate(N_way_data_list):
                    question_tokens = self.__get_question(class_name, entity_list[N_index][k_index])
                    add_q = [self.SEP] + question_tokens +[self.question_marker_id]+ [self.MASK]+ [self.SEP]
                    # add_q = question_tokens +[self.question_marker_id]+[self.MASK]+ [self.SEP]
                    
                    if self.encoder_name in ['bert', 'CP', 'roberta', "KEPLER"]:
                        new_word = [self.CLS] + word + add_q
                    elif self.encoder_name in ['bart', "CQARE_1", "CQARE_2"]:
                        new_word = word + add_q
                    else:
                        raise Exception("LM error") 
                       
                    # padding
                    while len(new_word) > self.max_length:
                        new_word.pop(-(len(add_q)+1) )
                    word_tensor = torch.tensor([self.tokenizer.pad_token_id] *self.max_length).long()  
                    sentence_len = min(self.max_length, len(new_word))
                    for i in range(sentence_len):
                        word_tensor[i] = new_word[i]

                    mask_tensor = torch.zeros((self.max_length)).long()
                    mask_tensor[:min(self.max_length, len(new_word))] = 1
                    seg_tensor = torch.ones((self.max_length)).long()
                    seg_tensor[:min(self.max_length, len(word) + 1)] = 0

                    word_tensor_list.append(word_tensor)
                    mask_tensor_list.append(mask_tensor)
                    seg_tensor_list.append(seg_tensor)
                    sent_len_list.append(len(word))
            pos1_list.extend(pos1)
            pos2_list.extend(pos2)
        return word_tensor_list, mask_tensor_list, seg_tensor_list, sent_len_list, pos1_list, pos2_list
    
    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        # target_classes = self.classes[:self.N]
        support_set = {'word': [], 'mask': [], 'seg': [], "sent_len":[], 'pos1': [], 'pos2': []}
        query_set = {'word': [], 'mask': [], 'seg': [], "sent_len":[], 'pos1': [], 'pos2': []}
        label = []
        support_entity_list = []
        support_word_list =[]
        query_entity_list = []
        query_word_list =[]
        support_pos1_list=[]
        support_pos2_list=[]
        query_pos1_list=[]
        query_pos2_list=[]
        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(list(range(len(self.json_data[class_name]))), self.K + self.Q, False)
            count = 0
            temp_support_entity_list =[]
            temp_query_entity_list =[]
            temp_support_word_list =[]
            temp_query_word_list =[]
            for j in indices:
                word, pos1, pos2 = self.__getraw__(self.json_data[class_name][j])
                if count < self.K:
                    temp_support_entity_list.append([self.json_data[class_name][j]['h'][0], self.json_data[class_name][j]['t'][0]])
                    temp_support_word_list.append(word)
                    support_pos1_list.append(pos1)
                    support_pos2_list.append(pos2)
                else:
                    temp_query_entity_list.append([self.json_data[class_name][j]['h'][0], self.json_data[class_name][j]['t'][0]])
                    temp_query_word_list.append(word)
                    query_pos1_list.append(pos1)
                    query_pos2_list.append(pos2)
                count += 1
            support_entity_list.append(temp_support_entity_list)    
            query_entity_list.append(temp_query_entity_list)    
            support_word_list.append(temp_support_word_list)
            query_word_list.append(temp_query_word_list)
            label += [i]
        self.__additem__(support_set, self.contact_question(support_word_list, support_entity_list, target_classes, support_pos1_list, support_pos2_list))
        self.__additem__(query_set, self.contact_question(query_word_list, query_entity_list, target_classes, query_pos1_list, query_pos2_list))
        
        # # NA
        # Q_na = int(self.na_rate * self.Q)
        # na_classes = list(filter(lambda x: x not in target_classes, self.classes))
        # NA_query_entity_list = []
        # NA_query_word_list =[]
        # for j in range(Q_na):
        #     cur_class = np.random.choice(na_classes, 1, False)[0]
        #     index = np.random.choice(list(range(len(self.json_data[cur_class]))), 1, False)[0]
        #     word, pos1, pos2 = self.__getraw__(self.json_data[cur_class][index])
        #     NA_query_word_list.append([word])
        #     NA_query_entity_list.append([[self.json_data[cur_class][j]['h'][0], self.json_data[cur_class][j]['t'][0]]])
        # if Q_na>0:
        #     self.__additem__(query_set, self.contact_question(NA_query_word_list, NA_query_entity_list, target_classes))
        #     label += [self.N] * Q_na
        
        return support_set, query_set, label, self.prompt_label


class FewRelDataset_Proto_Question_Test(FewRelDataset_Proto_Question):

    def __getitem__(self, index):
        index_data = self.json_data[index]
        target_classes = index_data["relation"]
        meta_train = index_data["meta_train"]
        meta_test_dic = index_data["meta_test"]
        support_set = {'word': [], 'mask': [], 'seg': [], "sent_len":[], 'pos1': [], 'pos2': []}
        query_set = {'word': [], 'mask': [], 'seg': [], "sent_len":[], 'pos1': [], 'pos2': []}
        support_word_list =[]
        support_entity_list = []
        support_pos1_list=[]
        support_pos2_list=[]
        query_pos1_list=[]
        query_pos2_list=[]
        for i, N_way_data in enumerate(meta_train):
            temp_support_entity_list =[]
            temp_support_word_list =[]
            for meta_train_dic in N_way_data:
                entity=[meta_train_dic['h'][0], meta_train_dic['t'][0]]
                word, pos1, pos2 = self.__getraw__(meta_train_dic)
                temp_support_word_list.append(word)
                temp_support_entity_list.append(entity)
                support_pos1_list.append(pos1)
                support_pos2_list.append(pos2)
                # word_tensor_list, mask_tensor_list, seg_tensor_list = self.contact_question(word, entity, target_classes)
            support_word_list.append(temp_support_word_list)
            support_entity_list.append(temp_support_entity_list)
            
        
        word, pos1, pos2  = self.__getraw__(meta_test_dic)
        query_word_list = [[word]]
        query_pos1_list.append(pos1)
        query_pos2_list.append(pos2)
        query_entity_list = [[[meta_test_dic['h'][0], meta_test_dic['t'][0]]]]
        
        self.__additem__(support_set, self.contact_question(support_word_list, support_entity_list, target_classes, support_pos1_list, support_pos2_list))
        self.__additem__(query_set, self.contact_question(query_word_list, query_entity_list, target_classes, query_pos1_list, query_pos2_list))
        if "query_relation" in index_data.keys():
            label = [index_data["query_relation"]]
        else:
            label = [0]
        return support_set, query_set, label, self.prompt_label


def collate_fn_proto_question(data):
    batch_support ={'word': [], 'mask': [], 'seg': [], "sent_len":[], 'pos1': [], 'pos2': []}
    batch_query ={'word': [], 'mask': [], 'seg': [], "sent_len":[], 'pos1': [], 'pos2': []}
    batch_label_list = []

    support_sets, query_sets, labels, loss_label = zip(*data)
    for i in range(len(support_sets)):  # len(support_sets) = batch_size
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]

        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
            
        batch_label_list += labels[i]

    for k in batch_support:
        if k!="sent_len":
            batch_support[k] = torch.stack(batch_support[k], 0)
        else:
            batch_support[k] = torch.tensor(batch_support[k]).squeeze()
    for k in batch_query:
        if k!="sent_len":
            batch_query[k] = torch.stack(batch_query[k], 0)
        else:
            batch_query[k] = torch.tensor(batch_query[k]).squeeze()
    
        
    batch_label = torch.tensor(batch_label_list)
    return batch_support, batch_query, batch_label, torch.stack(loss_label, 0)

def get_loader_proto_question(name, encoder, N, K, Q, batch_size,
               num_workers=8, collate_fn=collate_fn_proto_question, na_rate=0, root='./data', encoder_name='bert', 
               Ptuing=False, pattern="p1", P_template_format=[], Yes_No_token_dic=[], prompt_token='[PROMPT]'):
    if 'test' in name:
        # name = "my_"+name
        dataset = FewRelDataset_Proto_Question_Test(name, encoder, N, K, Q, na_rate, root, encoder_name, 
                                             Ptuing, pattern, P_template_format, Yes_No_token_dic, pseudo_token=prompt_token)
    else:
        dataset = FewRelDataset_Proto_Question(name, encoder, N, K, Q, na_rate, root, encoder_name, 
                                        Ptuing, pattern, P_template_format, Yes_No_token_dic, pseudo_token=prompt_token)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)
