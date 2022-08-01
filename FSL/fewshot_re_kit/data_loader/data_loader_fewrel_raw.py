import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json

class FewRelDataset(data.Dataset):
    def __init__(self, name, encoder, N, K, Q, na_rate, root, encoder_name):
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
        self.tokenizer = encoder.tokenizer
        self.CLS = self.encoder.tokenizer.bos_token_id
        self.SEP = self.encoder.tokenizer.eos_token_id
        self.max_length = encoder.max_length
        self.encoder_name = encoder_name
        
    def __getraw__(self, item):
        word, pos1, pos2 = self.encoder.tokenize(item['tokens'], item['h'][2][0], item['t'][2][0])
        pos1 = torch.tensor(pos1).long()
        pos2 = torch.tensor(pos2).long()
        return word, pos1, pos2

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes, self.classes))

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(list(range(len(self.json_data[class_name]))), self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2 = self.__getraw__( self.json_data[class_name][j])
                
                add_q = [self.SEP]
                if self.encoder_name in ['bert', 'CP', 'roberta', 'KEPLER']:
                    new_word = [self.CLS] + word + add_q
                elif self.encoder_name in ['bart']:
                    new_word = word + add_q
                else:
                    raise Exception("LM error")
                
                # padding
                while len(new_word) > self.max_length:
                    new_word.pop(-(len(add_q)+1) )
                word_tensor = torch.tensor([self.tokenizer.pad_token_id] *self.max_length).long()  
                sentence_len = min(self.max_length, len(new_word))
                for k in range(sentence_len):
                    word_tensor[k] = new_word[k]
                
                # attention mask
                mask_tensor = torch.zeros((self.max_length)).long()
                mask_tensor[:min(self.max_length, len(new_word))] = 1
                
                if count < self.K:
                    self.__additem__(support_set, word_tensor, pos1, pos2, mask_tensor)
                else:
                    self.__additem__(query_set, word_tensor, pos1, pos2, mask_tensor)
                count += 1
                
            query_label += [i] * self.Q

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice(
                    list(range(len(self.json_data[cur_class]))),
                    1, False)[0]
            word, pos1, pos2 = self.__getraw__(
                    self.json_data[cur_class][index])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            self.__additem__(query_set, word, pos1, pos2, mask)
        query_label += [self.N] * Q_na

        return support_set, query_set, query_label
    
    def __len__(self):
        return 1000000000


class FewRelDataset_Test(FewRelDataset):

    def __getitem__(self, index):
        index_data = self.json_data[index]
        target_classes = index_data["relation"]
        meta_train = index_data["meta_train"]
        meta_test_dic = index_data["meta_test"]
        label = []
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [] }
        for i, N_way_data in enumerate(meta_train):
            for meta_train_dic in N_way_data:
                entity=[meta_train_dic['h'][0], meta_train_dic['t'][0]]
                word, pos1, pos2  = self.__getraw__(meta_train_dic)
                
                add_q = [self.SEP]
                if self.encoder_name in ['bert', 'roberta']:
                    new_word = [self.CLS] + word + add_q
                elif self.encoder_name in ['bart']:
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
                
                # attention mask
                mask_tensor = torch.zeros((self.max_length)).long()
                mask_tensor[:min(self.max_length, len(new_word))] = 1
                
                self.__additem__(support_set, word_tensor, pos1, pos2, mask_tensor)
            label += [i]
                    
        word, pos1, pos2  = self.__getraw__(meta_test_dic)
        add_q = [self.SEP]
        if self.encoder_name in ['bert', 'roberta']:
            new_word = [self.CLS] + word + add_q
        elif self.encoder_name in ['bart']:
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
        
        # attention mask
        mask_tensor = torch.zeros((self.max_length)).long()
        mask_tensor[:min(self.max_length, len(new_word))] = 1
        self.__additem__(query_set, word_tensor, pos1, pos2, mask_tensor)

        return support_set, query_set, label


def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_label = []
    support_sets, query_sets, query_labels = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label, ""

def get_loader(name, encoder, N, K, Q, batch_size, encoder_name='bert', 
        num_workers=8, collate_fn=collate_fn, na_rate=0, root='./data'):
    if 'test' in name:
        dataset = FewRelDataset_Test(name, encoder, N, K, Q, na_rate, root, encoder_name)
    else:
        dataset = FewRelDataset(name, encoder, N, K, Q, na_rate, root, encoder_name)
        
    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)
    return iter(data_loader)




