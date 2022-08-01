import os
import sys
import torch
import shutil
import json
from torch import optim, nn
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from fewshot_re_kit.utils import data_to_device, get_print_loss, \
    log_gradient_and_parameter_updates, log_support_gradient_and_parameter_updates, conmine_sent, get_dic_added
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F



class Maml_Model(MetaModule):
    def __init__(self, my_sentence_encoder):
        '''
        sentence_encoder: Sentence encoder
        You need to set self.loss as your own loss function.
        '''
        MetaModule.__init__(self)
        self.sentence_encoder = DP(my_sentence_encoder)
        self.ce_loss = nn.CrossEntropyLoss()
        # self.contrasive_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))
        self.contrasive_loss = F.triplet_margin_loss

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

    def r_drop_loss_function(self, query_logits_list, r_drop_weight):
        loss_list = []
        logits_1 = query_logits_list[0]
        for logits_2 in query_logits_list[1:]:
            KL_1_loss = F.kl_div(-F.log_softmax(logits_1, dim=-1), -F.log_softmax(logits_2, dim=-1), reduction='mean')
            KL_2_loss = F.kl_div(-F.log_softmax(logits_2, dim=-1), -F.log_softmax(logits_1, dim=-1), reduction='mean')
            KL_1_loss=KL_1_loss.sum()
            KL_2_loss=KL_2_loss.sum()

            loss = -(KL_1_loss+KL_2_loss) /2
            loss_list.append(loss)
        return r_drop_weight*sum(loss_list)/len(loss_list)

    def contrasive_loss_function(self, last_hs_list, N, Q):
        """ last_hs size = (N*N*K, sen_len, 768)"""
        last_hs = sum(last_hs_list) / len(last_hs_list)
        last_hs = last_hs.view(N, N, Q, -1, 768)
        contrasive_loss_list = []
        prompt_len = 15
        for index_way in range(N):
            pos_item = last_hs[index_way][index_way]
            anchor = torch.sum(pos_item[:, :prompt_len, :], 1)
            pos = torch.sum(pos_item[:, prompt_len:, :], 1)
            for Question_index in range(N):
                if index_way!=Question_index:
                    neg = torch.sum(last_hs[index_way][Question_index][:, prompt_len:, :], 1)
                    contrasive_loss_litem = self.contrasive_loss(anchor, pos, neg)
                    contrasive_loss_list.append(contrasive_loss_litem)
        return sum(contrasive_loss_list) / len(contrasive_loss_list)

    def loss(self, logits, label, N, Q, ce_weight=1, contrasive_weight=None, last_hs_list=None, r_drop_weight=None, query_logits_list=None):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size.
        return: [Loss] (A single value)
        '''
        class_num = logits.size()[-1]
        total_loss = 0
        assert (ce_weight or r_drop_weight or contrasive_weight)
        ce_loss=0
        if ce_weight:
            ce_loss = ce_weight*self.ce_loss(logits.view(-1, class_num), label.view(-1))  #[(batch_size, N, k, class_num) / (batch_size, N, k)]
            total_loss += ce_loss

        contrasive_loss=0
        if contrasive_weight:
            contrasive_loss = contrasive_weight*self.contrasive_loss_function(last_hs_list, N, Q)
            total_loss +=contrasive_loss

        r_drop_loss=0
        if r_drop_weight:
            r_drop_loss = self.r_drop_loss_function(query_logits_list, r_drop_weight)
            total_loss +=r_drop_loss

        return total_loss, ce_loss, contrasive_loss, r_drop_loss

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))


class Maml_Framework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, adv_data_loader=None, adv=False, d=None, opt=None):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.adv_data_loader = adv_data_loader
        self.adv = adv
        
        self.res_path ="./results/output/"+str(sys.argv[1:])
        if os.path.exists(self.res_path):
            shutil.rmtree(self.res_path)
        os.mkdir(self.res_path)
        
        with open(os.path.join(self.res_path, "config.json"), "w") as f:
            json.dump(vars(opt), f, indent=2)

        if adv:
            self.adv_cost = nn.CrossEntropyLoss()
            self.d = d
            self.d.cuda()

    def get_optimizer(self, model, use_sgd_for_bert, learning_rate):
        # Init
        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if use_sgd_for_bert:
            optimizer = torch.optim.SGD(parameters_to_optimize, lr=learning_rate)
        else:
            optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)
        return optimizer

    def __load_ckpt_fn__(self, load_ckpt, model, fp16, optimizer):
        if load_ckpt:
            if os.path.isfile(load_ckpt):
                checkpoint = torch.load(load_ckpt)
                print("Successfully loaded checkpoint '%s'" % load_ckpt)
            else:
                raise Exception("No checkpoint found at '%s'" % load_ckpt)

            # if fp16:
            #     amp.load_state_dict(checkpoint['amp'])

            if "optimizer" in checkpoint.keys():
                optimizer.load_state_dict(checkpoint["optimizer"])

            model.load_state_dict(checkpoint['state_dict'])
        start_iter = 0
        return start_iter

    def adapt(self, model, support, label, N, K,optimizer, learning_rate, num_adaptation_steps=1,
              ce_weight=1, contrasive_weight=None,  train_flag="train", fp16=True, if_tensorboard=False, writer=None, it=0):
        params=None
        for step in range(num_adaptation_steps):
            support_logits, last_hs = model(support, N, K, params=params, train_flag=train_flag)
            support_loss, _, _, _ = model.loss(support_logits, label, N, K,  ce_weight=ce_weight, contrasive_weight=contrasive_weight, last_hs_list=[last_hs])
            support_scaled_loss = torch.tensor(0.)
            if fp16:
                with amp.scale_loss(support_loss, optimizer) as support_scaled_loss:
                    params = gradient_update_parameters(model, support_scaled_loss, step_size=learning_rate,
                                                        if_tensorboard=if_tensorboard, writer=writer, it=it)
            else:
                params = gradient_update_parameters(model, support_loss, step_size=learning_rate,
                                                    if_tensorboard=if_tensorboard, writer=writer, it=it)
        return params, support_loss, support_scaled_loss

    def r_drop_forward(self, num_query_steps, query, model, N, Q, params, train_flag):
        last_hs_list=[]
        query_logits_list=[]
        if train_flag=="test":
            test_query_N=1
        else:
            test_query_N=None
        
        for i in range(num_query_steps):
            query_logits, last_hs = model(query, N, Q, params=params, train_flag=train_flag, test_query_N=test_query_N)
            query_logits_list.append(query_logits)
            last_hs_list.append(last_hs)

        query_logits= sum(query_logits_list)/len(query_logits_list)  # (batch_size, N, N, self.vocab_len)
        
        # pred = torch.argmax(query_logits.view(-1, N), -1)
        pred_logits = query_logits[:, :, :, self.Yes_No_dic["Yes"]] # (batch_size, N, N)
        pred = torch.argmax(pred_logits.view(-1, N), -1)
        return query_logits, last_hs_list, pred, query_logits_list

    def maml_iter(self, model, N, K, Q, data_set, optimizer,
                  fp16, grad_iter=1, num_adaptation_steps=1, num_query_steps=1, scheduler=None, train_flag="train",
                  ce_weight=1, r_drop_weight=None, contrasive_weight=None,
                  if_tensorboard=False, writer=None, it=0 ):

        dic_grad_accu_res = {"mean_query_acc_list":[], "mean_support_loss_list":[], "mean_query_loss_list":[],
                             "mean_query_ce_loss_list":[], "mean_query_contrasive_loss_list" :[], "query_r_drop_loss":[],
                             "mean_scaled_support_loss_list":[], "mean_scaled_query_loss_list":[],
                             "res_train":[], "res_val":[], "res_test":[], 
                             }
        for task in range(grad_iter):
            """ support: (batch_size, N, K, N)  /  query:(batch_size, N, Q, N)  / label:(batch_size, N, Q)
            if train set, N=train_N, else N=N"""

            support, query, label, batch_loss_label = data_to_device(next(data_set))
            params, support_loss, support_scaled_loss = self.adapt(model, support, batch_loss_label, N, K, optimizer, optimizer.param_groups[0]['lr'],
                                                                   ce_weight=ce_weight, contrasive_weight=contrasive_weight, train_flag=train_flag,
                                                                    num_adaptation_steps=num_adaptation_steps, fp16=fp16, if_tensorboard=if_tensorboard, writer=writer, it=it)

            query_logits, last_hs_list, pred, query_logits_list= self.r_drop_forward(num_query_steps, query, model, N, Q, params, train_flag)

            if train_flag=="test":
                with open(self.res_path+"/test_with_sentence", "a") as f:
                    for index, tokens in enumerate(query['word']):
                        f.write(conmine_sent(model.sentence_encoder.module.tokenizer.convert_ids_to_tokens(tokens))+"\r")
                    f.write(str(pred.tolist())+"\r")
            
            if train_flag=="train":
                dic_grad_accu_res["res_train"].append(str(pred.tolist()))
            if train_flag=="val":
                dic_grad_accu_res["res_val"].append(str(pred.tolist()))    
            if train_flag=="test":
                dic_grad_accu_res["res_test"].append(str(pred.tolist()))
            
            query_loss, query_ce_loss, query_contrasive_loss, r_drop_loss, query_scaled_loss = 0,0,0,0,0
            if train_flag !="test":
                query_loss, query_ce_loss, query_contrasive_loss, r_drop_loss = model.loss(query_logits, batch_loss_label, N, Q,
                                                 ce_weight=ce_weight, contrasive_weight=contrasive_weight, last_hs_list=last_hs_list,
                                                 r_drop_weight=r_drop_weight , query_logits_list=query_logits_list)
                if fp16:
                    
                    with amp.scale_loss(query_loss, optimizer) as query_scaled_loss:
                        query_scaled_loss= query_scaled_loss / float(grad_iter)
                        query_scaled_loss.backward()
                else:
                    query_loss = query_loss/ float(grad_iter)
                    query_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                    
            query_acc = model.accuracy(pred, label)
            value_list = [query_acc, support_loss, query_loss,
                            query_ce_loss, query_contrasive_loss, r_drop_loss,
                            support_scaled_loss, query_scaled_loss]
            dic_grad_accu_res = get_dic_added(dic_grad_accu_res, dic_grad_accu_res.keys(), value_list)


        if train_flag=="train":
            # if if_tensorboard:
            #     log_gradient_and_parameter_updates(model, writer, it)
            optimizer.step()
            scheduler.step()

        optimizer.zero_grad(set_to_none=True)
        return dic_grad_accu_res

    def train(self, model, model_name, B, N_for_train, N_for_eval, K, Q,
              ce_weight=1, r_drop_weight=None, contrasive_weight=None, Yes_No_token_list=[],
              learning_rate=1e-1, train_iter=30000, val_iter=1000, val_step=2000,  load_ckpt=None,  save_ckpt=None, num_adaptation_steps=1, num_query_steps=1,
              if_tensorboard=True, warmup_step=300,  use_sgd_for_bert=False, grad_iter=1, fp16=True, loss_scale=5.0, early_stop=20,
              lr_step_size=20000,  weight_decay=1e-5, test_iter=3000, pytorch_optim=optim.SGD, bert_optim=False, warmup=True, adv_dis_lr=1e-1, adv_enc_lr=1e-1):
        self.Yes_No_dic = {"Yes": model.sentence_encoder.module.tokenizer.convert_tokens_to_ids("Yes"),
                           "No" : model.sentence_encoder.module.tokenizer.convert_tokens_to_ids("No") }
        
        model.train()
        if self.adv:
            self.d.train()
        print("Start training...")

        optimizer = self.get_optimizer(model, use_sgd_for_bert, learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter)

        if if_tensorboard:
            tensor_board_path ="./results/runs/"+str(sys.argv[1:])
            writer = SummaryWriter(tensor_board_path)
            try: shutil.rmtree(tensor_board_path)
            except: pass
        else:
            writer=None

        if fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, loss_scale=loss_scale, opt_level='O1')

        start_iter = self.__load_ckpt_fn__(load_ckpt, model, fp16, optimizer)

        # Training
        iter_train_support_loss = []
        iter_train_query_loss = []
        iter_train_scaled_support_loss = []
        iter_train_scaled_query_loss = []
        iter_ce_loss = []
        iter_contrasive_loss = []
        iter_r_drop_loss = []
        iter_acc = []
        best_acc = 0
        best_it = 0
        eval_times = 0
        left_early_stop=early_stop
        
        with tqdm(total=train_iter, desc="Train", ncols=150) as pbar:
            for it in range(start_iter, start_iter + train_iter):
                dic_grad_accu_res = self.maml_iter(model, N_for_train, K, Q, self.train_data_loader, optimizer, fp16,
                                                   grad_iter=grad_iter, scheduler=scheduler, train_flag="train",
                                                   num_adaptation_steps=num_adaptation_steps, num_query_steps=num_query_steps,
                                                   ce_weight=ce_weight, r_drop_weight=r_drop_weight, contrasive_weight=contrasive_weight,
                                                   if_tensorboard=if_tensorboard, writer=writer, it=it)

                postfix= {}
                print_train_acc = get_print_loss(iter_acc, dic_grad_accu_res["mean_query_acc_list"])*100
                print_train_support_loss = get_print_loss(iter_train_support_loss, dic_grad_accu_res["mean_support_loss_list"])
                print_train_query_loss = get_print_loss(iter_train_query_loss, dic_grad_accu_res["mean_query_loss_list"])
                print_train_scaled_support_loss = get_print_loss(iter_train_scaled_support_loss, dic_grad_accu_res["mean_scaled_support_loss_list"])
                print_train_scaled_query_loss = get_print_loss(iter_train_scaled_query_loss, dic_grad_accu_res["mean_scaled_query_loss_list"])
                print_ce_loss = get_print_loss(iter_ce_loss, dic_grad_accu_res["mean_query_ce_loss_list"])
                print_contrasive_loss = get_print_loss(iter_contrasive_loss, dic_grad_accu_res["mean_query_contrasive_loss_list"])
                print_r_drop_loss = get_print_loss(iter_r_drop_loss, dic_grad_accu_res["query_r_drop_loss"])
                
                postfix['acc']= '{0:.4f}'.format(print_train_acc)
                postfix['support']= '{0:.4f}'.format(print_train_support_loss)
                postfix['query']= '{0:.4f}'.format(print_train_query_loss)
                postfix['ce']= '{0:.4f}'.format(print_ce_loss)
                postfix['contrasive']= '{0:.4f}'.format(print_contrasive_loss)
                postfix['r_drop']= '{0:.4f}'.format(print_r_drop_loss)
                # postfix['scaled_s']= '{0:.4f}'.format(print_train_scaled_support_loss)
                # postfix['scaled_q']= '{0:.4f}'.format(print_train_scaled_query_loss)

                pbar.set_postfix(postfix)
                pbar.update(1)
                
                if if_tensorboard:
                    writer.add_scalars("Loss/Train_Total_loss", {"Train_support_loss": print_train_support_loss}, it)
                    writer.add_scalars("Loss/Train_Total_loss", {"Train_query_loss": print_train_query_loss}, it)
                    writer.add_scalars("Loss/Train_Sub_loss", {"Train_CE_loss": print_ce_loss}, it)
                    writer.add_scalars("Loss/Train_Sub_loss", {"Train_contrasive_loss": print_contrasive_loss}, it)
                    writer.add_scalars("Loss/Train_Sub_loss", {"Train_R_drop_loss": print_r_drop_loss}, it)
                    writer.add_scalars("Loss/Train_scaled_loss", {"Train_scaled_support_loss": print_train_scaled_support_loss}, it)
                    writer.add_scalars("Loss/Train_scaled_loss", {"Train_scaled_query_loss": print_train_scaled_query_loss}, it)
                    writer.add_scalars("Acc/Train_acc", {"Train_Acc": print_train_acc }, it)
                    writer.close()

                if (it + 1) % val_step == 0:
                    val_acc, total_pred = self.eval(model, N_for_eval, K, Q, val_iter, learning_rate,
                                    optimizer=optimizer, use_sgd_for_bert=use_sgd_for_bert, fp16=fp16, model_name=model_name,
                                    if_tensorboard=if_tensorboard, writer=writer, eval_times=eval_times,
                                    num_adaptation_steps=num_adaptation_steps, num_query_steps=num_query_steps,
                                    ce_weight=ce_weight, r_drop_weight=r_drop_weight, contrasive_weight=contrasive_weight)
                    model.train()
                    if val_acc > best_acc:
                        if fp16:
                            from apex import amp
                            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'amp': amp.state_dict()}
                        else:
                            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}

                        with open(self.res_path+"/val", "w") as f:
                            for i in total_pred:
                                f.write(str(i)+"\r")
                        
                        left_early_stop=early_stop
                        best_acc = val_acc
                        best_it = it
                        # torch.save(checkpoint, save_ckpt+"_"+str(best_it))
                        torch.save(checkpoint, save_ckpt)
                        tqdm.write('\r Best checkpoint! \r')
                    else:
                        left_early_stop-=1

                    with open(self.res_path+"/performanc.txt", "a") as f:
                        f.write("train_step:  "+str(it)+"\r")
                        f.write("train_acc:  "+str(float(print_train_acc))+"\r")
                        f.write("valid_acc:  "+str(float(val_acc))+"\r\r")

                    iter_train_support_loss = []
                    iter_train_query_loss = []
                    iter_train_scaled_support_loss = []
                    iter_train_scaled_query_loss = []
                    iter_ce_loss = []
                    iter_contrasive_loss = []
                    iter_r_drop_loss = []
                    iter_acc = []
                    eval_times+=1
                    tqdm.write("best it {}, acc {} ".format(best_it, best_acc))
                    tqdm.write("\r\r")
                    print()

                if left_early_stop==0:
                    break

        print("\n####################\n")
        print("Finish training " + model_name)

    def eval(self, model, N, K, Q, eval_iter, learning_rate=2e-5, optimizer=None, use_sgd_for_bert=False,
             fp16=True, ckpt=None, model_name=None, if_tensorboard=False, writer=None, eval_times=0,
             num_adaptation_steps=1, num_query_steps=1, Yes_No_token_list=[],
             ce_weight=1, r_drop_weight=None, contrasive_weight=None, test_valid_flag="val"):
        
        self.Yes_No_dic = {"Yes": model.sentence_encoder.module.tokenizer.convert_tokens_to_ids("Yes"),
                           "No" : model.sentence_encoder.module.tokenizer.convert_tokens_to_ids("No") }
        
        model.eval()
        if ckpt is None:
            print(" \r\r Use val dataset: {}".format(str(eval_times)))
            eval_dataset = self.val_data_loader
        else:
            tqdm.write("Use test dataset")
            eval_dataset = self.test_data_loader
            optimizer = self.get_optimizer(model, use_sgd_for_bert, learning_rate)
            if fp16:
                model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

            if ckpt != 'none':
                _ = self.__load_ckpt_fn__(ckpt, model, fp16, optimizer)

        iter_support_loss = []
        iter_query_loss = []
        iter_support_scaled_loss = []
        iter_query_scaled_loss = []
        iter_query_CE_loss = []
        iter_query_contrasive_loss = []
        iter_query_r_drop_loss = []
        iter_right = []
        valid_query_acc=0
        total_pred=[]
        with tqdm(total=eval_iter, desc=test_valid_flag, ncols=150) as pbar:
            for it in range(eval_iter):
                if "question" or "pair" in model_name:
                    dic_grad_accu_res = self.maml_iter(model, N, K, Q, eval_dataset, optimizer, fp16, train_flag=test_valid_flag,
                                                       num_adaptation_steps=num_adaptation_steps, num_query_steps=num_query_steps,
                                                       ce_weight=ce_weight, r_drop_weight=r_drop_weight, contrasive_weight=contrasive_weight)

                else:
                    raise NotImplementedError

                if test_valid_flag=="val":
                    iter_right.extend(dic_grad_accu_res["mean_query_acc_list"])
                    iter_support_loss.extend(dic_grad_accu_res["mean_support_loss_list"])
                    iter_query_loss.extend(dic_grad_accu_res["mean_query_loss_list"])
                    iter_query_CE_loss.extend(dic_grad_accu_res["mean_query_ce_loss_list"])
                    iter_query_contrasive_loss.extend(dic_grad_accu_res["mean_query_contrasive_loss_list"])
                    iter_query_r_drop_loss.extend(dic_grad_accu_res["query_r_drop_loss"])
                    iter_support_scaled_loss.extend(dic_grad_accu_res["mean_scaled_support_loss_list"])
                    iter_query_scaled_loss.extend(dic_grad_accu_res["mean_scaled_query_loss_list"])
                    total_pred.extend(dic_grad_accu_res["res_val"])
                    
                    valid_support_loss = sum(iter_support_loss) / len(iter_support_loss)
                    valid_query_loss = sum(iter_query_loss) / len(iter_query_loss)
                    valid_query_CE_loss = sum(iter_query_CE_loss) / len(iter_query_CE_loss)
                    valid_query_contrasive_loss = sum(iter_query_contrasive_loss) / len(iter_query_contrasive_loss)
                    valid_query_r_drop_loss_loss = sum(iter_query_r_drop_loss) / len(iter_query_r_drop_loss)
                    valid_scaled_support_loss = sum(iter_support_scaled_loss) / len(iter_support_scaled_loss)
                    valid_scaled_query_loss = sum(iter_query_scaled_loss) / len(iter_query_scaled_loss)
                    valid_query_acc = 100 *sum(iter_right) / len(iter_right)

                    postfix= {}
                    postfix['acc']= '{0:.4f}'.format(valid_query_acc)
                    postfix['support']= '{0:.4f}'.format(valid_support_loss)
                    postfix['query']= '{0:.4f}'.format(valid_query_loss)
                    postfix['ce']= '{0:.4f}'.format(valid_query_CE_loss)
                    postfix['contrasive']= '{0:.4f}'.format(valid_query_contrasive_loss)
                    postfix['r_drop']= '{0:.4f}'.format(valid_query_r_drop_loss_loss)
                    # postfix['scaled_s_loss']= '{0:.4f}'.format(valid_scaled_support_loss)
                    # postfix['scaled_q_loss']= '{0:.4f}'.format(valid_scaled_query_loss)
                    
                    pbar.set_postfix(postfix)
                    pbar.update(1)

                    if if_tensorboard:
                        writer.add_scalars("Loss/Valid_loss", {"Valid_support_loss": valid_support_loss}, it+eval_times*eval_iter)
                        writer.add_scalars("Loss/Valid_loss", {"Valid_query_loss": valid_query_loss}, it+eval_times*eval_iter)
                        writer.add_scalars("Loss/Valid_scaled_loss", {"Valid_scaled_support_loss": valid_scaled_support_loss}, it+eval_times*eval_iter)
                        writer.add_scalars("Loss/Valid_scaled_loss", {"Valid_scaled_query_loss": valid_scaled_query_loss}, it+eval_times*eval_iter)
                        writer.add_scalars("Acc/Valid_all_acc", {"Valid_Acc": valid_query_acc }, it+eval_times*eval_iter)

                        writer.add_scalars("Loss/Valid_Sub_loss", {"Valid_CE_loss": valid_query_CE_loss}, it+eval_times*eval_iter)
                        writer.add_scalars("Loss/Valid_Sub_loss", {"Valid_contrasive_loss": valid_query_contrasive_loss}, it+eval_times*eval_iter)
                        writer.add_scalars("Loss/Valid_Sub_loss", {"Valid_R_drop_loss": valid_query_r_drop_loss_loss}, it+eval_times*eval_iter)

                else:
                    iter_right.extend(dic_grad_accu_res["mean_query_acc_list"])
                    valid_query_acc = 100 *sum(iter_right) / len(iter_right)
                    
                    postfix= {}
                    postfix['accuracy']= '{0:.4f}'.format(valid_query_acc)
                    pbar.set_postfix(postfix)
                    
                    total_pred.extend(dic_grad_accu_res["res_test"])
                    pbar.update(1)

                
        if test_valid_flag=="test":
            with open(self.res_path+"/pred-{}-{}.json".format(N, K), "w") as f:
                # new_list = [eval(i[:-1])[0] for i in  data[N:len(data):N+1]]
                new_list = [eval(i)[0] for i in total_pred]
                f.write(str(new_list))
                
        return valid_query_acc, total_pred