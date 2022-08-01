import os
import numpy as np
import sys
import json
from tqdm import tqdm
import time
import shutil
from . import sentence_encoder
from . import data_loader
import torch
from torch import optim, nn
from torch.nn import functional as F
# from pytorch_pretrained_bert import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup

def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0


class My_Proto_Framework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, adv_data_loader=None, adv=False, d=None, 
                 opt=None):
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.adv_data_loader = adv_data_loader
        self.adv = adv
        self.res_path =os.path.join("./results/output/", str(sys.argv[1:]))
        if os.path.exists(self.res_path):
            shutil.rmtree(self.res_path)
        os.mkdir(self.res_path)
        with open(os.path.join(self.res_path, "config.json"), "w") as f:
            json.dump(vars(opt), f, indent=2)
    
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    
    def item(self, x):
        return x.item()

    def train(self, model, model_name, B, N_for_train, N_for_eval, K, Q,
              na_rate=0,  learning_rate=1e-1, lr_step_size=20000, weight_decay=1e-2,
              train_iter=30000, val_iter=1000, val_step=2000, test_iter=3000,
              load_ckpt=None,  save_ckpt=None,
              pytorch_optim=optim.SGD, proto_weight=1, prompt_weight=0, 
              warmup=True, warmup_step=300,
              grad_iter=1, fp16=False,
              adv_dis_lr=1e-1, adv_enc_lr=1e-1):
        print("Start training...")
    
        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize 
                if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in parameters_to_optimize
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        
        optimizer = pytorch_optim(parameters_to_optimize, lr=learning_rate, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 


        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
            start_iter = 0
        else:
            start_iter = 0

        if fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        model.train()
        if self.adv:
            self.d.train()

        # Training
        best_acc = 0
        iter_loss = 0.0
        iter_proto_loss = 0.0
        iter_prompt_loss = 0.0
        iter_right = 0.0
        iter_sample = 0.0
        best_it = 0
        with tqdm(total=train_iter, desc="Train", ncols=135) as pbar:    
            for it in range(start_iter, start_iter + train_iter):
                support, query, label, prompt_label = next(self.train_data_loader)
                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].cuda()
                    for k in query:
                        query[k] = query[k].cuda()
                    label = label.cuda()
                    prompt_label = prompt_label.cuda()

                model_res_dic = model(support, query, N_for_train, K, Q,  na_rate=na_rate, 
                                      prompt_weight=prompt_weight, proto_weight=proto_weight)

                loss, proto_loss, prompt_loss = model.loss(model_res_dic, label, prompt_label, N_for_train, K, Q, 
                                                            prompt_weight=prompt_weight, proto_weight=proto_weight) 
                loss = loss / float(grad_iter)
                proto_loss = proto_loss / float(grad_iter)
                prompt_loss = prompt_loss / float(grad_iter)
                right = model.accuracy(model_res_dic, label)
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                
                if it % grad_iter == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                iter_loss += loss
                iter_proto_loss += proto_loss
                iter_prompt_loss += prompt_loss
                iter_right += right
                
                iter_sample += 1
                
                print_train_acc = 100 * iter_right / iter_sample
                print_train_loss = iter_loss / iter_sample
                print_proto_loss = iter_proto_loss / iter_sample
                print_prompt_loss = iter_prompt_loss / iter_sample
                
                postfix= {}
                postfix['total']= '{0:.4f}'.format(print_train_loss)
                postfix['proto']= '{0:.4f}'.format(print_proto_loss)
                postfix['prompt']= '{0:.4f}'.format(print_prompt_loss)
                postfix['acc']= '{0:.4f}'.format(print_train_acc)

                pbar.set_postfix(postfix)
                pbar.update(1)
                
                if (it + 1) % val_step == 0:
                # print_train_acc =0
                # if True:
                    val_acc = self.eval(model, B, N_for_eval, K, Q, val_iter, na_rate=na_rate, 
                                        prompt_weight=prompt_weight, proto_weight=proto_weight)
                    model.train()
                    # break
                    if val_acc > best_acc:
                        torch.save({'state_dict': model.state_dict()}, save_ckpt)
                        # torch.save({'state_dict': model.state_dict()}, save_ckpt+"_"+str(best_it))
                        best_acc = val_acc
                        best_it = it
                        
                    with open(self.res_path+"/performanc.txt", "a") as f:
                        f.write("train_step:  "+str(it)+"\r")
                        f.write("train_acc:  "+str(float(print_train_acc))+"\r")
                        f.write("valid_acc:  "+str(float(val_acc))+"\r\r")
                        f.write("best it {0}, acc {1:.4f} ".format(best_it, best_acc))

                    iter_loss = 0.
                    iter_proto_loss = 0.0
                    iter_prompt_loss = 0.0
                    iter_right = 0.
                    iter_sample = 0.
                    print("best it {0}, acc {1:.4f} ".format(best_it, best_acc))
                    print()
                    print()
                    
                    
        print("\n####################\n")
        print("Finish training " + model_name)

    def eval(self, model, B, N, K, Q, eval_iter, na_rate=0, ckpt=None, 
             proto_weight=1, prompt_weight=0):
        print()
        model.eval()
        if ckpt is None:
            test_valid_flag="valid"
            temp_N = N
            eval_dataset = self.val_data_loader
            save_file = self.res_path+"/val.json"
        else:
            test_valid_flag="test"
            state_dict = self.__load_model__(ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
            
            temp_N = 1
            save_file = self.res_path+"/pred-{}-{}.json".format(N, K)
            eval_dataset = self.test_data_loader

        iter_right = 0.0
        iter_sample = 0.0
        iter_loss = 0.0
        iter_proto_loss = 0.0
        iter_prompt_loss = 0.0
        pred_list = []
        with torch.no_grad():
            with tqdm(total=eval_iter, desc=test_valid_flag, ncols=135) as pbar:
                for it in range(eval_iter):
                    iter_sample += 1
                    support, query, label, prompt_label = next(eval_dataset)
                    if torch.cuda.is_available():
                        for k in support:
                            support[k] = support[k].cuda()
                        for k in query:
                            query[k] = query[k].cuda()
                        label = label.cuda()
                        prompt_label = prompt_label.cuda()
                    
                    model_res_dic = model(support, query, N, K, Q, temp_N=temp_N, na_rate=na_rate, 
                                          prompt_weight=prompt_weight, proto_weight=proto_weight)
                    pred_list.extend(model_res_dic["final_pred"].tolist())   
                       
                    if test_valid_flag=="valid":
                        loss, proto_loss, prompt_loss = model.loss(model_res_dic, label, prompt_label, N, K, Q,
                                                                   prompt_weight=prompt_weight, proto_weight=proto_weight)
                        iter_loss += loss
                        iter_proto_loss += proto_loss
                        iter_prompt_loss += prompt_loss
                        
                        print_loss = iter_loss / iter_sample
                        print_proto_loss = iter_proto_loss / iter_sample
                        print_prompt_loss = iter_prompt_loss / iter_sample
                    else:
                        print_loss, print_proto_loss, print_prompt_loss = 0,0,0
            
                    right = model.accuracy(model_res_dic, label)
                    iter_right += right
                    print_acc = 100 * iter_right / iter_sample

                    postfix= {}
                    postfix['total']= '{0:.4f}'.format(print_loss)
                    postfix['proto']= '{0:.4f}'.format(print_proto_loss)
                    postfix['prompt']= '{0:.4f}'.format(print_prompt_loss)
                    postfix['acc']= '{0:.4f}'.format(print_acc)
                    
                    pbar.set_postfix(postfix)
                    pbar.update(1)
                    
            with open(save_file, "w") as f:
                f.write(str(pred_list))
        return print_acc
