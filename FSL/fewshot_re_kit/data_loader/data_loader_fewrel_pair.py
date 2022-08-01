import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json




class FewRelDatasetPair(data.Dataset):
    """
    FewRel Pair Dataset
    """
    def __init__(self, name, encoder, N, K, Q, na_rate, root, encoder_name):
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
        self.encoder_name = encoder_name
        self.max_length = encoder.max_length

    def __getraw__(self, item):
        word, pos_1, pos_2 = self.encoder.tokenize(item['tokens'], item['h'][2][0], item['t'][2][0])
        return word

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)

        support = []
        query = []
        query_label = []
        fusion_set = {'word': [], 'mask': [], 'seg': []}
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes, self.classes))

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(list(range(len(self.json_data[class_name]))), self.K + self.Q, False)
            count = 0
            for j in indices:
                if count < self.K:
                    word  = self.__getraw__(self.json_data[class_name][j])
                    support.append(word)
                else:
                    word  = self.__getraw__(self.json_data[class_name][j])
                    query.append(word)
                count += 1
            query_label += [i] * self.Q

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]
            index = np.random.choice( list(range(len(self.json_data[cur_class]))), 1, False)[0]
            word = self.__getraw__(self.json_data[cur_class][index])
            query.append(word)
        query_label += [self.N] * Q_na

        for index, word_query in enumerate(query):
            for word_support in support:
                if self.encoder_name == 'bert' or self.encoder_name == 'CP':
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['[SEP]'])
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])
                else:
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['</s>'])     
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['<s>'])

                """ pading , pad index must be 0"""
                word_tensor = torch.ones((self.max_length)).long()
                new_word = CLS + word_support + SEP + word_query + SEP
                for i in range(min(self.max_length, len(new_word))):
                    word_tensor[i] = new_word[i]
                
                if len(CLS + word_support + SEP)>=self.max_length:
                    word_tensor[-1] = SEP[0]
                    word_tensor[-2] = SEP[0]
                elif len(new_word)>=self.max_length:
                    word_tensor[-1] = SEP[0]
                
                mask_tensor = torch.zeros((self.max_length)).long()
                mask_tensor[:min(self.max_length, len(new_word))] = 1
                seg_tensor = torch.ones((self.max_length)).long()
                seg_tensor[:min(self.max_length, len(word_support) + 1)] = 0
                fusion_set['word'].append(word_tensor)
                fusion_set['mask'].append(mask_tensor)
                fusion_set['seg'].append(seg_tensor)

        return fusion_set, query_label
    
    def __len__(self):
        return 1000000000


class FewRelDatasetPair_Test(FewRelDatasetPair):

    def __getitem__(self, index):
        index_data = self.json_data[index]
        target_classes = index_data["relation"]
        meta_train = index_data["meta_train"]
        meta_test_dic = index_data["meta_test"]


        support = []
        query = []
        query_label = []
        fusion_set = {'word': [], 'mask': [], 'seg': []}
        Q_na = int(self.na_rate * self.Q)
        # na_classes = list(filter(lambda x: x not in target_classes, self.classes))

        for i, N_way_data in enumerate(meta_train):
            for meta_train_dic in N_way_data:
                word  = self.__getraw__(meta_train_dic)
                support.append(word)
            query_label += [i] * self.Q
        word  = self.__getraw__(meta_test_dic)
        query.append(word)
        # NA
        # for j in range(Q_na):
        #     cur_class = np.random.choice(na_classes, 1, False)[0]
        #     index = np.random.choice( list(range(len(self.json_data[cur_class]))), 1, False)[0]
        #     word = self.__getraw__(self.json_data[cur_class][index])
        #     query.append(word)
        # query_label += [self.N] * Q_na

        for index, word_query in enumerate(query):
            for word_support in support:
                if self.encoder_name == 'bert':
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['[SEP]'])
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])
                else:
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['</s>'])
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['<s>'])

                """ pading , pad index must be 0"""
                word_tensor = torch.ones((self.max_length)).long()
                new_word = CLS + word_support + SEP + word_query + SEP
                for i in range(min(self.max_length, len(new_word))):
                    word_tensor[i] = new_word[i]
                    
                if len(CLS + word_support + SEP)>=self.max_length:
                    word_tensor[-1] = SEP[0]
                    word_tensor[-2] = SEP[0]
                elif len(new_word)>=self.max_length:
                    word_tensor[-1] = SEP[0]
                
                mask_tensor = torch.zeros((self.max_length)).long()
                mask_tensor[:min(self.max_length, len(new_word))] = 1
                seg_tensor = torch.ones((self.max_length)).long()
                seg_tensor[:min(self.max_length, len(word_support) + 1)] = 0
                fusion_set['word'].append(word_tensor)
                fusion_set['mask'].append(mask_tensor)
                fusion_set['seg'].append(seg_tensor)

        return fusion_set, query_label


def collate_fn_pair(data):
    batch_set = {'word': [], 'seg': [], 'mask': []}
    batch_label = []
    fusion_sets, query_labels = zip(*data)

    for i in range(len(fusion_sets)):
        for k in fusion_sets[i]:
            batch_set[k] += fusion_sets[i][k]
        batch_label += query_labels[i]

    for k in batch_set:
        batch_set[k] = torch.stack(batch_set[k], 0)
    batch_label = torch.tensor(batch_label)

    return batch_set, batch_label

def get_loader_pair(name, encoder, N, K, Q, batch_size, num_workers=8,
                    collate_fn=collate_fn_pair, na_rate=0, root='./data', encoder_name='bert'):
    # dataset = FewRelDatasetPair(name, encoder, N, K, Q, na_rate, root, encoder_name)
    if 'test' in name:
        dataset = FewRelDatasetPair_Test(name, encoder, N, K, Q, na_rate, root, encoder_name)
    else:
        dataset = FewRelDatasetPair(name, encoder, N, K, Q, na_rate, root, encoder_name)

    data_loader = data.DataLoader(dataset=dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_fn)
    return iter(data_loader)

