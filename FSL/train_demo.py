import argparse
from tempfile import template
from collections import OrderedDict
parser = argparse.ArgumentParser()
parser.add_argument('--ID', default=0, type=int)
parser.add_argument('--GPU', default="0", type=str)
parser.add_argument('--trainN', default=10, type=int, help='N in train, chould different from N-way')
parser.add_argument('--N', default=5, type=int, help='N way')
parser.add_argument('--K', default=1, type=int, help='K shot')
parser.add_argument('--Q', default=1, type=int, help='Num of query per class')
parser.add_argument('--grad_iter', default=1, type=int, help='accumulate gradient every x iterations, if use r_drop, min=2')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--gpu1_data_num', default=1, type=int, help='batch size')

parser.add_argument('--train_iter', default=20000, type=int, help='num of iters in training:30000')
parser.add_argument('--val_step', default=200, type=int, help='val after training how many iters:100')
parser.add_argument('--val_iter', default=200, type=int, help='num of iters in validation:100')
parser.add_argument('--test_iter', default=10000, type=int, help='num of iters in testing:10000')

parser.add_argument('--model', default='proto_question', help='model name',
                       choices=['proto', 'proto_question',  'gnn', 'snail', 'metanet', 'siamese', 'pair', 'mtb'])
parser.add_argument('--train_frame', default='my_proto_frame', choices=['raw', 'maml', "my_proto_frame"])
parser.add_argument('--encoder', default='CQARE_2', choices=["cnn", "bert", "KEPLER", "CP","bart", "CQARE_2", "CQARE_1"], help='CQARE_2 for 2.0， CQARE_1 for 1.0')
parser.add_argument('--only_test', action='store_true', default=False,  help='only test')
parser.add_argument('--load_ckpt', default=None, help="load ckpt : pth.tar")

parser.add_argument('--Ptuing', action='store_true', default=True, help='use P tuing')
parser.add_argument('--P_template_format', default="[1,3,3,1]", type=str, help='P_template_format')
parser.add_argument('--pattern', default='p3', help='prompt pattern', choices=['p1', 'p2',  "p3", 'p4', 'p5'])
parser.add_argument('--prompt_token', default='[PROMPT]', type=str, help='use P tuing')
parser.add_argument('--prompt_weight', default=1, type=float, help='prompt task loss weight')
parser.add_argument('--proto_weight', default=0, type=float, help='proto task loss weight')

parser.add_argument('--train', default='train_wiki', help='train file: train_wiki')
parser.add_argument('--val', default='val_pubmed', help='valfile: val_pubmed / val_wiki') 
parser.add_argument('--test', default='test_pubmed', help='test file: test_pubmed / test_wiki')

parser.add_argument('--cat_entity_rep', action='store_true', default=False, help='concatenate entity representation as sentence rep')
parser.add_argument('--dropout', default=0, type=float, help='dropout rate of classifer head, default=0.0,')
parser.add_argument('--lr', default=2e-5, type=float, help='learning rate: -1')
parser.add_argument('--weight_decay', default=1e-2, type=float, help='weight decay 1e-2')
parser.add_argument('--warmup_step', default=300, type=int, help='learning rate: 300')
parser.add_argument('--loss_scale', default=None, type=float, help='loss_scale, None:no fixed scale')
parser.add_argument('--early_stop', default=100, type=int, help='early_stop')
parser.add_argument('--max_length', default=128, type=int, help='max length: 128')
parser.add_argument('--optim', default='adamw', help='sgd / adam / adamw')
parser.add_argument('--hidden_size', default=768, type=int, help='hidden size')
parser.add_argument('--save_ckpt', default=None, help='save ckpt')
parser.add_argument('--na_rate', default=0, type=int, help='NA rate (NA = Q * na_rate), 50%=5 15%=1')
parser.add_argument('--fp16', action='store_true', default=True, help='use nvidia apex fp16')
parser.add_argument('--if_tensorboard', action='store_true', default=False)
parser.add_argument('--ckpt_name', type=str, default='[]', help='checkpoint name.')
parser.add_argument('--adv', default=None, help='adv file')
parser.add_argument('--num_query_steps', default=None, type=int, help='maml num_query_steps open=2')
parser.add_argument('--num_adaptation_steps', default=1, type=int, help='maml num_adaptation_steps')
parser.add_argument('--pair', action='store_true', default=False, help='use pair model')

# only for prototypical networks
parser.add_argument('--dot', action='store_true', help='use dot instead of L2 distance for proto')
# only for mtb
parser.add_argument('--no_dropout', action='store_true', help='do not use dropout after BERT (still has dropout in BERT).')

# experiment
parser.add_argument('--mask_entity', action='store_true', help='mask entity names')
parser.add_argument('--use_sgd_for_bert', action='store_true', help='use SGD instead of AdamW for BERT.')
opt = parser.parse_args()

opt.P_template_format = eval(opt.P_template_format)
opt.Yes_No_token_dic = OrderedDict()
opt.Yes_No_token_dic["no"]="No"
opt.Yes_No_token_dic["yes"]= "Yes"



if opt.test=="test_pubmed":
    opt.test="test_pubmed_input-{}-{}".format(opt.N, opt.K)
    opt.test_valid_flag = "test"
elif opt.test=="test_wiki":
    opt.test="test_wiki_input-{}-{}".format(opt.N, opt.K)
    opt.test_valid_flag = "test"
else:
    opt.test_valid_flag = "val"

if opt.encoder=="KEPLER":
    opt.pretrain_ckpt="./pretrain/KEPLER"
elif opt.encoder=="CQARE_1":
    opt.pretrain_ckpt="./pretrain/CQARE_1"
elif opt.encoder=="CQARE_2":
    opt.pretrain_ckpt="./pretrain/CQARE_2"
elif opt.encoder=="bert":
    opt.pretrain_ckpt="./pretrain/bert-base"
elif opt.encoder=="CP":
    opt.pretrain_ckpt="./pretrain/CP/pytorch_model.bin" 
elif opt.encoder=="bart":
    opt.pretrain_ckpt="./pretrain/bart_base"
elif opt.encoder=="cnn":
    opt.pretrain_ckpt=None
else:
    raise NotImplementedError

if opt.load_ckpt:
    opt.load_ckpt = "./checkpoint/" + opt.load_ckpt
if opt.num_query_steps is None:
    opt.num_query_steps =1


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU
import sys
from torch import nn
from fewshot_re_kit.data_loader.data_loader_fewrel_raw  import get_loader
from fewshot_re_kit.data_loader.data_loader_fewrel_pair import get_loader_pair
from fewshot_re_kit.data_loader.data_loader_fewrel_unsupervised import get_loader_unsupervised
from fewshot_re_kit.data_loader.data_loader_fewrel_question import get_loader_question
from fewshot_re_kit.data_loader.data_loader_fewrel_proto_question import get_loader_proto_question

from fewshot_re_kit.utils import print_execute_time
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.my_proto_frame import My_Proto_Framework
from fewshot_re_kit.sentence_encoder.cnn_encoder import CNNSentenceEncoder
from fewshot_re_kit.sentence_encoder.bert_encoder import BERTSentenceEncoder,  BERTPAIRSentenceEncoder
from fewshot_re_kit.sentence_encoder.bart_encoder import BART_SentenceEncoder, BART_PairSentenceEncoder
from fewshot_re_kit.sentence_encoder.roberta_encoder import RobertaSentenceEncoder, RobertaPAIRSentenceEncoder

from models.proto import Proto
from models.proto_question import Proto_Quesion
from models.gnn import GNN
from models.snail import SNAIL
from models.metanet import MetaNet
from models.siamese import Siamese
from models.pair import Pair
from models.question import Question
from models.no_question import No_Question
from models.d import Discriminator
from models.mtb import Mtb
import warnings
import torch
from torch import optim
import numpy as np
import json
warnings.filterwarnings("ignore")

      
        
def get_sentene_encoder(encoder_name, model_name, max_length):
    if encoder_name == 'cnn':
        try:
            glove_mat = np.load('./pretrain/glove/glove_mat.npy')
            glove_word2id = json.load(open('./pretrain/glove/glove_word2id.json'))
        except:
            raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")
        sentence_encoder = CNNSentenceEncoder(glove_mat,glove_word2id,max_length)
    elif encoder_name == 'bert' or encoder_name == 'CP':
        pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
        if model_name=="pair" :
            sentence_encoder = BERTPAIRSentenceEncoder(pretrain_ckpt, max_length)
        else:
            sentence_encoder = BERTSentenceEncoder(pretrain_ckpt, max_length, cat_entity_rep=opt.cat_entity_rep, mask_entity=opt.mask_entity)
    elif encoder_name == 'bart' or encoder_name == 'CQARE_1' or encoder_name == 'CQARE_2':
        pretrain_ckpt = opt.pretrain_ckpt
        if model_name=="pair" :
            sentence_encoder = BART_PairSentenceEncoder(pretrain_ckpt, max_length, opt.Ptuing, opt.prompt_token, dropout=opt.dropout)
        else: 
            sentence_encoder = BART_SentenceEncoder(pretrain_ckpt, max_length, opt.Ptuing, opt.prompt_token, dropout=opt.dropout)
    elif encoder_name == 'roberta' or encoder_name == 'KEPLER':
        pretrain_ckpt = opt.pretrain_ckpt 
        if model_name=="pair" :
            sentence_encoder = RobertaPAIRSentenceEncoder(pretrain_ckpt, max_length)
        else:
            sentence_encoder = RobertaSentenceEncoder(pretrain_ckpt, max_length, cat_entity_rep=opt.cat_entity_rep, Ptuing=opt.Ptuing, entity_pseudo_token=opt.prompt_token)
    else:
        raise NotImplementedError
    return sentence_encoder

def get_model(model_name, sentence_encoder, N, K, max_length):
    if model_name == 'proto':
        model = Proto(sentence_encoder, dot=opt.dot)
    elif model_name == 'proto_question':
        model = Proto_Quesion(sentence_encoder, dot=opt.dot, Ptuing=opt.Ptuing, cat_entity_rep=opt.cat_entity_rep, gpu1_data_num=opt.gpu1_data_num, 
                              template=opt.P_template_format, pseudo_token=opt.prompt_token, prompt_encoder_ckpt = opt.pretrain_ckpt)
        
    elif model_name == 'gnn':
        model = GNN(sentence_encoder, N, hidden_size=opt.hidden_size)
    elif model_name == 'snail':
        model = SNAIL(sentence_encoder, N, K, hidden_size=opt.hidden_size)
    elif model_name == 'metanet':
        model = MetaNet(N, K, sentence_encoder.embedding, max_length)
    elif model_name == 'siamese':
        model = Siamese(sentence_encoder, hidden_size=opt.hidden_size, dropout=opt.dropout)
    elif model_name == 'mtb':
        model = Mtb(sentence_encoder, use_dropout=not opt.no_dropout)
    elif model_name == 'pair':
        model = Pair(sentence_encoder, hidden_size=opt.hidden_size)
        opt.pair = True
    elif model_name == 'question':
        model = Question(sentence_encoder, Ptuing=opt.Ptuing, template=opt.P_template_format, pseudo_token=opt.prompt_token, prompt_encoder_ckpt = opt.pretrain_ckpt)
    elif model_name == 'no_question':
        model = No_Question(sentence_encoder)
    else:
        raise NotImplementedError

    return model

def get_data_loader(model_name, sentence_encoder, batch_size, trainN, N, K, Q, encoder_name):
    adv_data_loader = None
    if model_name=="pair":
        train_data_loader = get_loader_pair(opt.train, sentence_encoder,
                                            N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, encoder_name=encoder_name)
        val_data_loader = get_loader_pair(opt.val, sentence_encoder,
                                          N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size, encoder_name=encoder_name)
        test_data_loader = get_loader_pair(opt.test, sentence_encoder,
                                           N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=1, encoder_name=encoder_name)
    elif model_name=="question" :
        train_data_loader = get_loader_question(opt.train, sentence_encoder, N=trainN, K=K, Q=Q, na_rate=opt.na_rate,
                                                batch_size=batch_size, encoder_name=encoder_name, Ptuing=opt.Ptuing, 
                                                P_template_format=opt.P_template_format, 
                                                prompt_token=opt.prompt_token)
        val_data_loader = get_loader_question(opt.val, sentence_encoder, N=N, K=K, Q=Q, na_rate=opt.na_rate,
                                              batch_size=batch_size, encoder_name=encoder_name, Ptuing=opt.Ptuing, 
                                              P_template_format=opt.P_template_format, 
                                              prompt_token=opt.prompt_token)
        test_data_loader = get_loader_question(opt.test, sentence_encoder, N=N, K=K, Q=Q, na_rate=opt.na_rate,
                                               batch_size=1, encoder_name=encoder_name, Ptuing=opt.Ptuing, 
                                               P_template_format=opt.P_template_format, 
                                               prompt_token=opt.prompt_token)
    elif model_name=="proto_question":
        train_data_loader = get_loader_proto_question(opt.train, sentence_encoder, N=trainN, K=K, Q=Q, na_rate=opt.na_rate,
                                                batch_size=batch_size, encoder_name=encoder_name, Ptuing=opt.Ptuing, pattern=opt.pattern, 
                                                P_template_format=opt.P_template_format, Yes_No_token_dic=opt.Yes_No_token_dic, 
                                                prompt_token=opt.prompt_token)
        val_data_loader = get_loader_proto_question(opt.val, sentence_encoder, N=N, K=K, Q=Q, na_rate=opt.na_rate,
                                              batch_size=batch_size, encoder_name=encoder_name, Ptuing=opt.Ptuing, pattern=opt.pattern, 
                                              P_template_format=opt.P_template_format, Yes_No_token_dic=opt.Yes_No_token_dic, 
                                              prompt_token=opt.prompt_token)
        test_data_loader = get_loader_proto_question(opt.test, sentence_encoder, N=N, K=K, Q=Q, na_rate=opt.na_rate,
                                               batch_size=1, encoder_name=encoder_name, Ptuing=opt.Ptuing, pattern=opt.pattern, 
                                               P_template_format=opt.P_template_format, Yes_No_token_dic=opt.Yes_No_token_dic, 
                                               prompt_token=opt.prompt_token)
    else:
        train_data_loader = get_loader(opt.train, sentence_encoder,  encoder_name=encoder_name,
                                       N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
        val_data_loader = get_loader(opt.val, sentence_encoder,  encoder_name=encoder_name,
                                     N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
        test_data_loader = get_loader(opt.test, sentence_encoder,  encoder_name=encoder_name,
                                      N=N, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
        if opt.adv:
            adv_data_loader = get_loader_unsupervised(opt.adv, sentence_encoder,
                                                      N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
    return train_data_loader, val_data_loader, test_data_loader, adv_data_loader

def get_prefix(model_name, encoder_name, N, K):
    prefix = '-'.join([model_name, encoder_name, opt.train, opt.val, str(N), str(K)])
    if opt.adv is not None:
        prefix += '-adv_' + opt.adv
    if opt.na_rate != 0:
        prefix += '-na{}'.format(opt.na_rate)
    if opt.dot:
        prefix += '-dot'
    if opt.cat_entity_rep:
        prefix += '-catentity'
    if len(opt.ckpt_name) > 0:
        prefix += '-' + opt.ckpt_name
    return prefix

@print_execute_time
def main():

    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    max_length = opt.max_length
    
    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("GPU: {}".format(opt.GPU))
    print("model: {}".format(model_name))
    print("prompt_weight: {}".format(opt.prompt_weight))
    print("proto_weight: {}".format(opt.proto_weight))
    print("train_frame: {}".format(opt.train_frame))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))

    sentence_encoder = get_sentene_encoder(encoder_name, model_name, max_length)
    train_data_loader, val_data_loader, test_data_loader, adv_data_loader = get_data_loader(model_name, sentence_encoder, batch_size, trainN, N, K, Q, encoder_name)
    model = get_model(model_name, sentence_encoder, N, K, max_length)

    prefix = str(sys.argv[1:])
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        model.cuda()
    
    if opt.optim == 'sgd':
        pytorch_optim = optim.SGD
    elif opt.optim == 'adam':
        pytorch_optim = optim.Adam
    elif opt.optim == 'adamw':
        from transformers import AdamW
        pytorch_optim = AdamW
    else:
        raise NotImplementedError

    if opt.train_frame=="raw":
        if opt.adv:
            d = Discriminator(opt.hidden_size)
            framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader, adv_data_loader, adv=opt.adv, d=d)
        else:
            framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)
    elif opt.train_frame=="my_proto_frame":
        framework = My_Proto_Framework(train_data_loader, val_data_loader, test_data_loader, opt=opt)
    else:
        raise NotImplementedError

    if not opt.only_test:
        if opt.train_frame=="raw":
            framework.train(model, prefix, batch_size, trainN, N, K, Q,
                pytorch_optim=pytorch_optim, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                na_rate=opt.na_rate, val_step=opt.val_step, fp16=opt.fp16, pair=opt.pair,
                learning_rate=opt.lr, train_iter=opt.train_iter, val_iter=opt.val_iter)
            
        elif opt.train_frame=="my_proto_frame":
            framework.train(model, prefix, batch_size, trainN, N, K, Q,
                pytorch_optim=pytorch_optim, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                na_rate=opt.na_rate, val_step=opt.val_step, fp16=opt.fp16, 
                prompt_weight=opt.prompt_weight, proto_weight=opt.proto_weight,
                learning_rate=opt.lr, train_iter=opt.train_iter, val_iter=opt.val_iter)
            
        elif opt.train_frame=="maml":
            framework.train(model, prefix, batch_size, trainN, N, K, Q,
                    pytorch_optim=pytorch_optim, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                    val_step=opt.val_step, fp16=opt.fp16, warmup_step=opt.warmup_step, 
                    num_query_steps=opt.num_query_steps,  num_adaptation_steps=opt.num_adaptation_steps,
                    train_iter=opt.train_iter, val_iter=opt.val_iter, if_tensorboard=opt.if_tensorboard,
                    learning_rate=opt.lr, use_sgd_for_bert=opt.use_sgd_for_bert, grad_iter=opt.grad_iter, loss_scale=opt.loss_scale, early_stop=opt.early_stop)
        
    else:
        ckpt = opt.load_ckpt
        if ckpt is None:
            print("Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint.")
            ckpt = 'none'

    sentence_encoder = get_sentene_encoder(encoder_name, model_name, max_length)
    model = get_model(model_name, sentence_encoder, N, K, max_length).cuda()
    if opt.train_frame=="raw":
        _= framework.eval(model, batch_size, N, K, 1, opt.test_iter, na_rate=opt.na_rate, ckpt=ckpt, pair=opt.pair)
    elif opt.train_frame=="my_proto_frame":
        _= framework.eval(model, batch_size, N, K, 1, opt.test_iter, na_rate=opt.na_rate, ckpt=ckpt,
                          prompt_weight=opt.prompt_weight, proto_weight=opt.proto_weight)
    elif opt.train_frame=="maml":
        _= framework.eval(model, N, K, 1, opt.test_iter, learning_rate=opt.lr, use_sgd_for_bert=opt.use_sgd_for_bert, 
                         fp16=opt.fp16, ckpt=ckpt, model_name=model_name, 
                         test_valid_flag=opt.test_valid_flag)
        
if __name__ == "__main__":
    main()
