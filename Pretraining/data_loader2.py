import os
import logging
import json

import gzip
import time
import random
import numpy as np
import torch
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence

from transformers import BartTokenizer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
# from run_pretrain import args

logger = logging.getLogger(__name__)

MAX_SENT_LEN = 256

Tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

def Traverse_dir(rootDir, total_files=[]):
    for root,dirs,files in os.walk(rootDir):
        for file in files:
            if file.endswith('.jsonl.gz'):
                total_files.append(os.path.join(root,file))
        for dir in dirs:
            Traverse_dir(dir)

def batch_convert_ids_to_tensors(batch_token_ids: List[List]) -> torch.Tensor:

    bz = len(batch_token_ids)
    batch_tensors = [torch.LongTensor(batch_token_ids[i]).squeeze(0) for i in range(bz)]
    batch_tensors = pad_sequence(batch_tensors, True, padding_value=Tokenizer.pad_token_id).long()
    return batch_tensors

def get_re_sample_rate(step, total_steps, start=0.5, end=0.5):
    return start + (end - start) / total_steps * step

step = 0
max_steps = 1000000
def choice_prompt_task():
    global step
    global total_steps
    rate = get_re_sample_rate(step, max_steps)
    step += 1

    if random.random() < rate:
        return 're'
    else:
        return 'ner'

def data_collator(features:List[dict])->List[torch.Tensor]:
    '''
    defining collator functioon for preparing batches on the fly ..

    params {
        features:
    }
    return {
        batch: 
    }
    '''
    batch = dict()
    batch['mask_encoder_inputs'] = batch_convert_ids_to_tensors([f["mask_encoder_input"] for f in features])
    batch['origin_decoder_inputs'] = batch_convert_ids_to_tensors([f["origin_decoder_input"] for f in features])
    batch['lm_labels'] = batch_convert_ids_to_tensors([f["lm_label"] for f in features])
    
    batch['prompt_task'] = choice_prompt_task()
    if batch['prompt_task'] == 're':
        batch['prompt_encoder_inputs'] = batch_convert_ids_to_tensors([f["re_encoder_input"] for f in features])
        batch['prompt_decoder_inputs'] = batch_convert_ids_to_tensors([f["re_decoder_input"] for f in features])
        batch['prompt_labels'] = torch.LongTensor([f["r_label"] for f in features])
        batch['anchor_encoder_inputs'] = batch_convert_ids_to_tensors([f["anchor_re_encoder_input"] for f in features])
        batch['anchor_decoder_inputs'] = batch_convert_ids_to_tensors([f["anchor_re_encoder_input"] for f in features])
        batch['pos_encoder_inputs'] = batch_convert_ids_to_tensors([f["pos_re_encoder_input"] for f in features])
        batch['pos_decoder_inputs'] = batch_convert_ids_to_tensors([f["pos_re_encoder_input"] for f in features])
        batch['neg_encoder_inputs'] = batch_convert_ids_to_tensors([f["neg_re_encoder_input"] for f in features])
        batch['neg_decoder_inputs'] = batch_convert_ids_to_tensors([f["neg_re_encoder_input"] for f in features])
    elif batch['prompt_task'] == 'ner':
        batch['prompt_encoder_inputs'] = batch_convert_ids_to_tensors([f["ner_encoder_input"] for f in features])
        batch['prompt_decoder_inputs'] = batch_convert_ids_to_tensors([f["ner_decoder_input"] for f in features])
        batch['prompt_labels'] = torch.LongTensor([f["e_label"] for f in features])
        batch['anchor_encoder_inputs'] = batch_convert_ids_to_tensors([f["anchor_ner_encoder_input"] for f in features])
        batch['anchor_decoder_inputs'] = batch_convert_ids_to_tensors([f["anchor_ner_encoder_input"] for f in features])
        batch['pos_encoder_inputs'] = batch_convert_ids_to_tensors([f["pos_ner_encoder_input"] for f in features])
        batch['pos_decoder_inputs'] = batch_convert_ids_to_tensors([f["pos_ner_encoder_input"] for f in features])
        batch['neg_encoder_inputs'] = batch_convert_ids_to_tensors([f["neg_ner_encoder_input"] for f in features])
        batch['neg_decoder_inputs'] = batch_convert_ids_to_tensors([f["neg_ner_encoder_input"] for f in features])
    return batch

def load_data(args):
    logger.info('Begin to load data ...')
    data_files = []
    Traverse_dir(args.data_dir, data_files)
    logger.info('total data files: [%d]' % len(data_files))

    # splitting dataset into train, validation
    split = 0.95
    train_files = data_files[:int(len(data_files)*split)]
    eval_files = data_files[int(len(data_files)*split):]
    logger.info('train data files: [%d]' % len(train_files))
    logger.info('train eval files: [%d]' % len(eval_files))

    # train_dataset = WikiDataset(train_files, 5, 100000, args=args)
    # eval_dataset = WikiDataset(eval_files, len(eval_files), 100000, args=args)
    train_dataset = WikiDataset(train_files[:3], 2, 1000, args=args)
    eval_dataset = WikiDataset(eval_files[-1:], 1, 5000, args=args)

    return train_dataset, eval_dataset

class TripletData():
    def __init__(self, triplet_file, entity_file, relation_file):
        self.triplet_file = triplet_file
        self.entity_file = entity_file
        self.relation_file = relation_file
        self._load_entity_data()
        self._load_relation_data()
        self._load_triplet_data()

    def _load_relation_data(self):
        self.relation_dict = dict()
        with open(self.relation_file, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.lower()
                data = line.strip().split('\t')
                assert len(data) >= 2
                self.relation_dict[data[0]] = data[1]
    
    def _load_entity_data(self):
        self.entity_dict = dict()
        with open(self.entity_file, 'r', encoding='utf-8') as fe:
            lines = fe.readlines()
            for line in lines:
                line = line.lower()
                data = line.strip().split('\t')
                assert len(data) >= 2
                for e in data[1:]:
                    self.entity_dict[e] = data[0]
    
    def _load_triplet_data(self):
        self.triplet_dict = dict()
        with open(self.triplet_file, 'r', encoding='utf-8') as ft:
            lines = ft.readlines()
            for line in lines:
                line = line.lower()
                data = line.strip().split('\t')
                assert len(data) == 3

                if data[1] not in self.relation_dict:
                    continue
                self.triplet_dict[(data[0],data[2])] = self.relation_dict[data[1]]
    
    def __len__(self):
        return len(self.triplet_dict)
    
    def find_relation(self, ohead, otail):
        head = ohead.lower()
        tail = otail.lower()
        if head not in self.entity_dict or tail not in self.entity_dict:
            return None
        head_id = self.entity_dict[head]
        tail_id = self.entity_dict[tail]
        if (head_id, tail_id) in self.triplet_dict:
            return (ohead, otail, self.triplet_dict[(head_id, tail_id)], 1)
        elif (tail_id, head_id) in self.triplet_dict:
            return (otail, ohead, self.triplet_dict[(tail_id, head_id)], 1)
        else:
            return None

class WikiDataset(torch.utils.data.Dataset):

    def __init__(self, 
                filepaths: list, 
                windows_size: int=10, 
                data_length_per_file: int=10000, 
                triplet_data: TripletData=None, 
                eType: dict=None,
                negative_rate: int=5,
                args: dict=None):

        self.filepaths = filepaths
        self.windows_size = windows_size
        self.current_index = 0
        self.current_data_length = 0
        self.idata_length = data_length_per_file
        self.texts = []
        self.triplet_data = triplet_data
        self.eType = eType
        self.negative_rate = negative_rate

        # for p-tuning
        if args:
            self.tokenizer = Tokenizer
            self.template = args.template
            self.ner_template = args.ner_template
            self.tokenizer.add_special_tokens({'additional_special_tokens': [args.entity_pseudo_token]})
            self.tokenizer.add_special_tokens({'additional_special_tokens': [args.relation_pseudo_token]})
            self.entity_pseudo_token_id = self.tokenizer.get_vocab()[args.entity_pseudo_token]
            self.relation_pseudo_token_id = self.tokenizer.get_vocab()[args.relation_pseudo_token]
            self.MASK = self.tokenizer.mask_token_id
            self.question_marker_id = self.tokenizer.convert_tokens_to_ids("?")

        self._load_data(self.current_index)

    def __len__(self):
        if self.windows_size == len(self.filepaths):
            return len(self.texts)
        return len(self.filepaths) * self.idata_length
    
    def _load_data(self, index):
        logger.info(f'current index is [{index}]')
        self.texts = []
        if (index + self.windows_size) > len(self.filepaths):
            files = self.filepaths[index:]
        else:
            files = self.filepaths[index:index+self.windows_size]
        for i, file in enumerate(files):
            logger.info('loading file [%d] ...' % i)
            with gzip.GzipFile(file, 'r') as fin:
                json_bytes = fin.read().splitlines()   

            for i, bytes in enumerate(json_bytes):
                if i >= self.idata_length:
                    break
                json_line = json.loads(str(bytes, 'utf-8'))
                self.texts.append({'text': json_line['text'], 
                                    'entities': json_line['entities'], 
                                    'triplets': json_line['triplets'] if 'triplets' in json_line else [],
                                    'pos_entities': json_line['pos_entities'],
                                    'neg_entities': json_line['neg_entities'],
                                    'pos_triplets': json_line['pos_triplets'],
                                    'neg_triplets': json_line['neg_triplets'],})
                # print({'text': json_line['text'], 
                #                     'entities': json_line['entities'], 
                #                     'triplets': json_line['triplets'] if 'triplets' in json_line else []})
        
        random.shuffle(self.texts)
        self.current_data_length = len(self.texts)
        logger.info(f'current size of data is [{len(self.texts)}]')

    def process_data(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_file_num = 0
        save_data_num = 0
        processed_texts = []

        for file in self.filepaths:
            print('processing file [%s] ...' % file)
            with gzip.GzipFile(file, 'r') as fin:
                json_bytes = fin.read().splitlines()   

            for bytes in json_bytes:

                json_line = json.loads(str(bytes, 'utf-8'))
                
                json_line['entities'] = [entity 
                                        for entity in json_line['entities'] 
                                        if entity[1] != 'None_type']

                sampled_entities = json_line['entities']
                pos_entities = []
                neg_entities = []
                if self.eType is not None:
                    # sample entities
                    sampled_entities = []
                    for entity in json_line['entities']:
                        name = entity[0]
                        type = self.eType[entity[1]]
                        sampled_entities.append([name, type, 1])
                        pos_entities.append([name, type])

                        # negtive sample
                        for _ in range(self.negative_rate):
                            negative_type = random.sample(list(self.eType.values()), 1)[0]
                            if negative_type != type:
                                sampled_entities.append([name, type, 0])
                                neg_entities.append([name, type])
                
                pos_triplets = []
                neg_triplets = []
                if self.triplet_data is not None:
                    # sample (h, t, o)
                    entities = list(set([e[0] for e in json_line['entities']]))
                    for i in range(len(entities)):
                        for j in range(i, len(entities)):
                            if entities[i] == entities[j]:
                                continue

                            triplet = self.triplet_data.find_relation(entities[i], entities[j])
                            if not triplet:
                                continue

                            head, tail, relation, r_label = triplet
                            pos_triplets.append((head, tail, relation, r_label))

                            # negtive sample
                            for _ in range(self.negative_rate):
                                false_relation = random.sample(list(self.triplet_data.relation_dict.values()), 1)[0]
                                if false_relation != relation:
                                    neg_triplets.append((head, tail, false_relation, 0))

                sampled_triplets = pos_triplets + neg_triplets
                random.shuffle(sampled_triplets)

                if len(pos_triplets) >= 2 and len(pos_entities) >= 2:
                    processed_texts.append({'text': json_line['text'], 
                                            'entities': sampled_entities,
                                            'pos_entities': pos_entities,
                                            'neg_entities': neg_entities,
                                            'triplets': sampled_triplets,
                                            'pos_triplets': [pos[:3] for pos in pos_triplets],
                                            'neg_triplets': [neg[:3] for neg in neg_triplets]
                                            })

                if len(processed_texts) >= 100000:
                    print('saving file [%s] ...' % os.path.join(save_dir, "%s.jsonl.gz" % str(save_file_num)))
                    with gzip.open(os.path.join(save_dir, "%s.jsonl.gz" % str(save_file_num)), "wb") as fout:
                        for d in processed_texts:
                            fout.write((json.dumps(d)+'\n').encode())
                    save_data_num += len(processed_texts)
                    save_file_num += 1
                    processed_texts = []

        if len(processed_texts) > 0:
            print('saving file [%s] ...' % os.path.join(save_dir, "%s.jsonl.gz" % str(save_file_num)))
            with gzip.open(os.path.join(save_dir, "%s.jsonl.gz" % str(save_file_num)), "wb") as fout:
                for d in processed_texts:
                    fout.write((json.dumps(d)+'\n').encode())
            save_data_num += len(processed_texts)
            save_file_num += 1
        
        print(f'Total got [{save_data_num}] data.')
        print(f'Total got [{save_file_num}] data files.')
    
    def _entity_aware_text_masking(self, text, entity):
        entity_tokens = entity.split(' ')
        text_tokens = text.split(' ')
        replace_text_tokens = []

        # random mask some entities
        i = 0
        while i < len(text_tokens):
            if ' '.join(entity_tokens).lower() == ' '.join(text_tokens[i:i+(len(entity_tokens))]).lower():
                if random.random() < 0.6:
                    replace_text_tokens.append('<mask>')
                    i += len(entity_tokens)
                else:
                    replace_text_tokens.append(text_tokens[i])
                    i += 1
            else:
                replace_text_tokens.append(text_tokens[i])
                i += 1

        return ' '.join(replace_text_tokens)

    def _token_aware_text_masking(self, text):
        replace_text_tokens = text.split(' ')
        # random mask some tokens
        for i, _token in enumerate(replace_text_tokens):
            if random.random() < 0.1:
                replace_text_tokens[i] = '<mask>'

        return ' '.join(replace_text_tokens)

    def _convert_data_to_bart_format(self, data:dict) -> dict:
        # logger.info('converting data to bart pretrain format ...')

        input_item = {}
        
        # random mask entities and tokens
        text = data['text']
        for entity in data['entities']:
            text = self._entity_aware_text_masking(text, entity[0])
        text = self._token_aware_text_masking(text)

        input_item['mask_encoder_input'] = self.tokenizer.encode(text, add_special_tokens=False, max_length=MAX_SENT_LEN) + [self.tokenizer.eos_token_id]
        input_item['origin_encoder_input'] = self.tokenizer.encode(data['text'], add_special_tokens=False, max_length=MAX_SENT_LEN) + [self.tokenizer.eos_token_id]
        input_item['origin_decoder_input'] = [self.tokenizer.bos_token_id] + self.tokenizer.encode(data['text'], add_special_tokens=False, max_length=MAX_SENT_LEN) + [self.tokenizer.eos_token_id]
        input_item['lm_label'] = self.tokenizer.encode(data['text'], add_special_tokens=False, max_length=MAX_SENT_LEN) + [self.tokenizer.eos_token_id]
        
        triplet = data['triplets'][random.randint(0, len(data['triplets'])-1)]
        entity = data['entities'][random.randint(0, len(data['entities'])-1)]

        triplet_input = triplet[:3]
        entity_input = entity[:2]

        input_item['triplet'] = [self.tokenizer.encode(item, add_special_tokens=False) for item in triplet_input]
        input_item['entity'] = [self.tokenizer.encode(item, add_special_tokens=False) for item in entity_input]

        input_item["r_label"] = triplet[3]
        input_item["e_label"] = entity[2]
        
        anchor_triplet = data['pos_triplets'][random.randint(0, len(data['pos_triplets'])-1)]
        pos_triplet = data['pos_triplets'][random.randint(0, len(data['pos_triplets'])-1)]
        neg_triplet = data['neg_triplets'][random.randint(0, len(data['neg_triplets'])-1)]

        input_item['anchor_triplet'] = [self.tokenizer.encode(item, add_special_tokens=False) for item in anchor_triplet]
        input_item['pos_triplet'] = [self.tokenizer.encode(item, add_special_tokens=False) for item in pos_triplet]
        input_item['neg_triplet'] = [self.tokenizer.encode(item, add_special_tokens=False) for item in neg_triplet]

        anchor_entity = data['pos_entities'][random.randint(0, len(data['pos_entities'])-1)]
        pos_entity = data['pos_entities'][random.randint(0, len(data['pos_entities'])-1)]
        neg_entity = data['neg_entities'][random.randint(0, len(data['neg_entities'])-1)]

        input_item['anchor_entity'] = [self.tokenizer.encode(item, add_special_tokens=False) for item in anchor_entity]
        input_item['pos_entity'] = [self.tokenizer.encode(item, add_special_tokens=False) for item in pos_entity]
        input_item['neg_entity'] = [self.tokenizer.encode(item, add_special_tokens=False) for item in neg_entity]

        return input_item

    def _get_p_tuning_input(self, x_h, prompt_token, x_t=None, type=None, task=0):
        if task == 'ner':
            prompt = [[prompt_token] * self.ner_template[0]
                        + x_h    # entity
                        + [prompt_token] * self.ner_template[1]
                        + type   # entity type
                        + [prompt_token] * self.ner_template[2]
                        + [self.question_marker_id]
                        ]
        elif task == 're':
            prompt = [[prompt_token] * self.template[0]
                        + x_h   # head entity
                        + [prompt_token] * self.template[1]
                        + x_t   # relation type
                        + [prompt_token] * self.template[3]
                        + type  # tail entity
                        + [prompt_token] * self.template[2]
                        + [self.question_marker_id]
                        ]
        else:
            prompt = [[]]
        return prompt[0]

    def __getitem__(self, _id):
        index = _id // self.idata_length
        if index >= (self.current_index + self.windows_size) or index < self.current_index:
            self._load_data(index)
            self.current_index = index
        text = self.texts[_id % self.current_data_length]

        data = self._convert_data_to_bart_format(text)

        sample = {}
        sample['mask_encoder_input'] = data['mask_encoder_input']
        sample['origin_decoder_input'] = data['origin_decoder_input']

        sample['ner_encoder_input'] = data['mask_encoder_input'] \
                                    + self._get_p_tuning_input(data['entity'][0], 
                                                                self.entity_pseudo_token_id, 
                                                                type = data['entity'][1],
                                                                task='ner') \
                                    + [self.MASK] \
                                    + [self.tokenizer.eos_token_id]
        sample['ner_decoder_input'] = data['origin_decoder_input'] \
                                    + self._get_p_tuning_input(data['entity'][0], 
                                                                self.entity_pseudo_token_id, 
                                                                type = data['entity'][1],
                                                                task='ner') \
                                    + [self.MASK] \
                                    + [self.tokenizer.eos_token_id]
        sample['re_encoder_input'] = data['mask_encoder_input'] \
                                    + self._get_p_tuning_input(data['triplet'][0], 
                                                                self.relation_pseudo_token_id, 
                                                                data['triplet'][1],
                                                                type = data['triplet'][2],
                                                                task='re') \
                                    + [self.MASK] \
                                    + [self.tokenizer.eos_token_id]
        sample['re_decoder_input'] = data['origin_decoder_input'] \
                                    + self._get_p_tuning_input(data['triplet'][0], 
                                                                self.relation_pseudo_token_id, 
                                                                data['triplet'][1],
                                                                type = data['triplet'][2],
                                                                task='re') \
                                    + [self.MASK] \
                                    + [self.tokenizer.eos_token_id]
        sample['lm_label'] = data['lm_label']
        sample['r_label'] = data['r_label']
        sample['e_label'] = data['e_label']

        # Contrastive Sample
        # anchor sample
        sample['anchor_ner_encoder_input'] = data['origin_encoder_input'] \
                                        + self._get_p_tuning_input(data['anchor_entity'][0], 
                                                                    self.entity_pseudo_token_id, 
                                                                    type = data['anchor_entity'][1],
                                                                    task='ner') \
                                        + [self.tokenizer.eos_token_id]
        sample['anchor_ner_decoder_input'] = data['origin_decoder_input'] \
                                        + self._get_p_tuning_input(data['anchor_entity'][0], 
                                                                    self.entity_pseudo_token_id, 
                                                                    type = data['anchor_entity'][1],
                                                                    task='ner') \
                                        + [self.tokenizer.eos_token_id]
        sample['anchor_re_encoder_input'] = data['origin_encoder_input'] \
                                        + self._get_p_tuning_input(data['anchor_triplet'][0], 
                                                                    self.relation_pseudo_token_id, 
                                                                    data['anchor_triplet'][1],
                                                                    type = data['anchor_triplet'][2],
                                                                    task='re') \
                                        + [self.tokenizer.eos_token_id]
        sample['anchor_re_decoder_input'] = data['origin_decoder_input'] \
                                        + self._get_p_tuning_input(data['anchor_triplet'][0], 
                                                                    self.relation_pseudo_token_id, 
                                                                    data['anchor_triplet'][1],
                                                                    type = data['anchor_triplet'][2],
                                                                    task='re') \
                                        + [self.tokenizer.eos_token_id]

        # pos_samples
        sample['pos_ner_encoder_input'] = data['origin_encoder_input'] \
                                        + self._get_p_tuning_input(data['pos_entity'][0], 
                                                                    self.entity_pseudo_token_id, 
                                                                    type = data['pos_entity'][1],
                                                                    task='ner') \
                                        + [self.tokenizer.eos_token_id]
        sample['pos_ner_decoder_input'] = data['origin_decoder_input'] \
                                        + self._get_p_tuning_input(data['pos_entity'][0], 
                                                                    self.entity_pseudo_token_id, 
                                                                    type = data['pos_entity'][1],
                                                                    task='ner') \
                                        + [self.tokenizer.eos_token_id]
        sample['pos_re_encoder_input'] = data['origin_encoder_input'] \
                                        + self._get_p_tuning_input(data['pos_triplet'][0], 
                                                                    self.relation_pseudo_token_id, 
                                                                    data['pos_triplet'][1],
                                                                    type = data['pos_triplet'][2],
                                                                    task='re') \
                                        + [self.tokenizer.eos_token_id]
        sample['pos_re_decoder_input'] = data['origin_decoder_input'] \
                                        + self._get_p_tuning_input(data['pos_triplet'][0], 
                                                                    self.relation_pseudo_token_id, 
                                                                    data['pos_triplet'][1],
                                                                    type = data['pos_triplet'][2],
                                                                    task='re') \
                                        + [self.tokenizer.eos_token_id]

        # neg_samples
        sample['neg_ner_encoder_input'] = data['origin_encoder_input'] \
                                        + self._get_p_tuning_input(data['neg_entity'][0], 
                                                                    self.entity_pseudo_token_id, 
                                                                    type = data['neg_entity'][1],
                                                                    task='ner') \
                                        + [self.tokenizer.eos_token_id]
        sample['neg_ner_decoder_input'] = data['origin_decoder_input'] \
                                        + self._get_p_tuning_input(data['neg_entity'][0], 
                                                                    self.entity_pseudo_token_id, 
                                                                    type = data['neg_entity'][1],
                                                                    task='ner') \
                                        + [self.tokenizer.eos_token_id]
        sample['neg_re_encoder_input'] = data['origin_encoder_input'] \
                                        + self._get_p_tuning_input(data['neg_triplet'][0], 
                                                                    self.relation_pseudo_token_id, 
                                                                    data['neg_triplet'][1],
                                                                    type = data['neg_triplet'][2],
                                                                    task='re') \
                                        + [self.tokenizer.eos_token_id]
        sample['neg_re_decoder_input'] = data['origin_decoder_input'] \
                                        + self._get_p_tuning_input(data['neg_triplet'][0], 
                                                                    self.relation_pseudo_token_id, 
                                                                    data['neg_triplet'][1],
                                                                    type = data['neg_triplet'][2],
                                                                    task='re') \
                                        + [self.tokenizer.eos_token_id]
        return sample

def test():
    data = []
    with open('dataset/sample.txt') as f1:
        for i, src in enumerate(f1):
            data.append(src.strip())
            if i >= 1:
                break

    print(f'total size of data is {len(data)}')

    batch = Tokenizer.prepare_seq2seq_batch(src_texts=data, max_length=MAX_SENT_LEN, padding='max_length')
    batch["labels"] = batch["input_ids"].copy()

    vocab = {}
    with open('pretrained_model/bart_base/vocab.json') as fp:
        vocab = json.load(fp)
    vocab = {v:k for k, v in vocab.items()}
    print(vocab)
    sent = []
    for sentence in batch["input_ids"]:
        line = []
        for token in sentence:
            line.append(vocab[token])
        sent.append(line)

    label = []
    for sentence in batch["labels"]:
        line = []
        for token in sentence:
            line.append(vocab[token])
        label.append(line)

    print(batch)
    print(sent)
    print(label)

def get_entity_type(data_dir):
    print('Begin to process data ...')
    data_files = []
    Traverse_dir(data_dir, data_files)
    print('total data files: [%d]' % len(data_files))
    
    type_set = set()
    for file in data_files[:]:
        print('processing file [%s] ...' % file)
        with gzip.GzipFile(file, 'r') as fin:
            json_bytes = fin.read().splitlines()   

        for i, bytes in enumerate(json_bytes):
            json_line = json.loads(str(bytes, 'utf-8'))
            
            for entity in json_line['entities']:
                if entity[1] != 'None_type':
                    type_set.add(str(entity[1])+'\n')

    print('Totally got [%d] entity types.' % len(type_set))
    with open('entity_type.txt', 'w', encoding='utf-8') as fp:
        type_list = list(type_set)
        fp.writelines(type_list)


if __name__ == '__main__':

    wikidata = TripletData('dataset/Wikidata5m/wikidata5m_all_triplet.txt', 'dataset/Wikidata5m/wikidata5m_entity.txt', 'dataset/Wikidata5m/wikidata5m_relation.txt')

    data_dir = 'dataset/wiki_NER'
    eType_dict = {}
    with open(os.path.join(data_dir, 'entity_type.txt'), 'r', encoding='utf-8') as fp:
        etypes = fp.readlines()
        for t in etypes:
            type, tname = t.strip("\n").split('\t')
            eType_dict[type] = tname

    data_files = []
    Traverse_dir(data_dir, data_files)

    dataset = WikiDataset(data_files[:], 0, 0, wikidata, eType_dict, negative_rate=5)
    dataset.process_data(os.path.join(data_dir, 'processed'))

    # WikiDataset([os.path.join(data_dir, 'processed', '0.jsonl.gz')], 1, 100, wikidata, eType_list)
