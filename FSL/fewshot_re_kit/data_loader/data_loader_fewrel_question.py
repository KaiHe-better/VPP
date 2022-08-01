import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json




class FewRelDatasetQuesiton(data.Dataset):
    def __init__(self, name, encoder, N, K, Q, na_rate, root, encoder_name, 
                 Ptuing, P_template_format, Yes_No_token_list, pseudo_token='[PROMPT]'):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert(0)
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
        self.Yes_No_dic = {"Yes": self.tokenizer.convert_tokens_to_ids("Yes"),
                           "No":  self.tokenizer.convert_tokens_to_ids("No")}

        self.vocab_len = len(self.tokenizer)

        self.name = name
        pid2name_path = os.path.join(root, "pid2name.json")
        self.pid2name = json.load(open(pid2name_path, 'r'))

        name2id_path = os.path.join(root, "name2id_all.json")
        self.name2id = json.load(open(name2id_path, 'r'))

        id2question_path = os.path.join(root, "id2question_all.json")
        self.id2question = json.load(open(id2question_path, 'r'))

        self.Ptuing = Ptuing
        if self.Ptuing:
            self.pseudo_token = pseudo_token
            if pseudo_token not in self.tokenizer.get_vocab():
                self.tokenizer.add_special_tokens({'additional_special_tokens': [pseudo_token]})
            self.pseudo_token_id = self.tokenizer.get_vocab()[pseudo_token]
            self.template = P_template_format

    def __len__(self):
        # data_length = 0
        # if type(self.json_data) == dict:
        #     for value in self.json_data.values():
        #         data_length += len(value)
        # else:
        #     data_length = len(self.json_data)
        # return data_length
        return 1000000000

    def __additem__(self, d, word_list, mask_list, seg_list):
        for word, mask, seg in zip(word_list, mask_list, seg_list):
            d['word'].append(word)
            d['mask'].append(mask)
            d['seg'].append(seg)

    def __relation_name2ID__(self, relation_name):
        if "wiki" in self.name:
            relation_id = self.name2id[self.pid2name[relation_name][0]]
        elif "pubmed" in self.name:
            relation_id = self.name2id[relation_name]
        else:
            raise NotImplementedError
        return relation_id

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
            return_question = self.pseudo_token * self.template[0] + entity[0] \
                                + self.pseudo_token * self.template[1] + entity[1] \
                                + self.pseudo_token * self.template[2] + ' '.join(self.__pid2name__(question_name).split('_')) \
                                + self.pseudo_token * self.template[3] 
                                
            return self.tokenizer.encode(return_question, add_special_tokens=False)
        else:
            relation_id = self.__relation_name2ID__(question_name)
            return_question = self.id2question[str(relation_id)] 
            return_question = self.encoder.tokenize_question(return_question, entity)
            return return_question

    def __getraw__(self, item):
        word = self.encoder.tokenize(item['tokens'], item['h'][2][0], item['t'][2][0])
        return word

    def contact_question(self, word, target_classes, entity):
        word_tensor_list=[]
        mask_tensor_list=[]
        seg_tensor_list=[]

        for class_name in target_classes:
            question_tokens = self.__get_question(class_name, entity)
            
            add_q = [self.SEP] + question_tokens +[self.question_marker_id]+[self.MASK]+ [self.SEP]
            if self.encoder_name in ['bert', 'roberta']:
                new_word = [self.CLS] + word + add_q
            elif self.encoder_name in ['bart']:
                new_word = word + add_q
            
            while len(new_word) > self.max_length:
                new_word.pop(-(len(add_q)+1) )
            word_tensor = torch.tensor([self.tokenizer.pad_token_id] *self.max_length).long()  
            sentence_len = min(self.max_length, len(new_word))
            for i in range(sentence_len):
                word_tensor[i] = new_word[i]

            mask_tensor = torch.zeros((self.max_length)).long()
            mask_tensor[:min(self.max_length, len(new_word))] = 1
            seg_tensor = torch.ones((self.max_length)).long()
            seg_tensor[:min(self.max_length, len(question_tokens) + 1)] = 0

            word_tensor_list.append(word_tensor)
            mask_tensor_list.append(mask_tensor)
            seg_tensor_list.append(seg_tensor)
        return word_tensor_list, mask_tensor_list, seg_tensor_list

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        support_set = {'word': [], 'mask': [], 'seg': []}
        query_set = {'word': [], 'mask': [], 'seg': []}
        label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes, self.classes))

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(list(range(len(self.json_data[class_name]))), self.K + self.Q, False)
            count = 0
            for j in indices:
                entity=[self.json_data[class_name][j]['h'][0], self.json_data[class_name][j]['t'][0]]
                word  = self.__getraw__(self.json_data[class_name][j])

                word_tensor_list, mask_tensor_list, seg_tensor_list = self.contact_question(word, target_classes, entity)
                if count < self.K:
                    self.__additem__(support_set, word_tensor_list, mask_tensor_list, seg_tensor_list)
                else:
                    self.__additem__(query_set, word_tensor_list, mask_tensor_list, seg_tensor_list)
                count += 1
                
            label += [i]
            
        loss_label = torch.zeros(self.N, self.N).long()
        for i in range(self.N):
            for j in range(self.N):
                if i==j:
                    loss_label[i][j]=self.Yes_No_dic["Yes"]
                else:
                    loss_label[i][j]=self.Yes_No_dic["No"]
        
        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(list(range(len(self.json_data[cur_class]))), 1, False)[0]
            word = self.__getraw__(self.json_data[cur_class][index])
            label.append(word)
            label += [self.N] * Q_na
        return support_set, query_set, label, loss_label


class FewRelDatasetQuesiton_Test(FewRelDatasetQuesiton):

    def __getitem__(self, index):
        index_data = self.json_data[index]
        target_classes = index_data["relation"]
        meta_train = index_data["meta_train"]
        meta_test_dic = index_data["meta_test"]
        support_set = {'word': [], 'mask': [], 'seg': []}
        query_set = {'word': [], 'mask': [], 'seg': []}
        for i, N_way_data in enumerate(meta_train):
            for meta_train_dic in N_way_data:
                entity=[meta_train_dic['h'][0], meta_train_dic['t'][0]]
                word  = self.__getraw__(meta_train_dic)
                word_tensor_list, mask_tensor_list, seg_tensor_list = self.contact_question(word, target_classes, entity)
                self.__additem__(support_set, word_tensor_list, mask_tensor_list, seg_tensor_list)
            
        loss_label = torch.zeros(self.N, self.N).long()
        for i in range(self.N):
            for j in range(self.N):
                if i==j:
                    loss_label[i][j]=self.Yes_No_dic["Yes"]
                else:
                    loss_label[i][j]=self.Yes_No_dic["No"]    
                    
        word  = self.__getraw__(meta_test_dic)
        word_tensor_list, mask_tensor_list, seg_tensor_list = self.contact_question(word, target_classes, entity)
        self.__additem__(query_set, word_tensor_list, mask_tensor_list, seg_tensor_list)
        
        if "query_relation" in index_data.keys():
            label = [index_data["query_relation"]]
        else:
            label = [0]
        return support_set, query_set, label, loss_label


def collate_fn_question(data):
    batch_support ={'word': [], 'seg': [], 'mask': []}
    batch_query = {'word': [], 'seg': [], 'mask': []}
    batch_label_list = []

    support_sets, query_sets, labels, loss_label = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]

        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
            
        batch_label_list += labels[i]

    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    
        
    batch_label = torch.tensor(batch_label_list)
    return batch_support, batch_query, batch_label, torch.stack(loss_label, 0)

def get_loader_question(name, encoder, N, K, Q, batch_size,
               num_workers=8, collate_fn=collate_fn_question, na_rate=0, root='./data', encoder_name='bert', 
               Ptuing=False, P_template_format=[], Yes_No_token_list=[], prompt_token='[PROMPT]'):
    if 'test' in name:
        # name = "my_"+name
        dataset = FewRelDatasetQuesiton_Test(name, encoder, N, K, Q, na_rate, root, encoder_name, 
                                             Ptuing, P_template_format, Yes_No_token_list, pseudo_token=prompt_token)
    else:
        dataset = FewRelDatasetQuesiton(name, encoder, N, K, Q, na_rate, root, encoder_name, 
                                        Ptuing, P_template_format, Yes_No_token_list, pseudo_token=prompt_token)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)
