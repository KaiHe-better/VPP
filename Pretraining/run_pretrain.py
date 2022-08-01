'''
Author: Andrew
Date: 2021-06-29 14:26:53
LastEditTime: 2021-11-30 16:41:59
LastEditors: Andrew
Description: In User Settings Edit
FilePath: /BART-pretrain/run_pretrain.py
'''
#encoding=utf-8
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

from transformers.trainer_utils import SchedulerType

import logging
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ## Initiating model and trainer for training
from transformers import BartModel, BartConfig, BartForCausalLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

# from models import BartForMaskedLM as BartPretrainModel
from models import BartForFewShotLearning as BartPretrainModel

from utils import setup_logger
from data_loader import load_data, data_collator

logger = logging.getLogger()

setup_logger(logger)

def main(args):
    
    train_dataset, eval_dataset = load_data(args)

    # defining training related arguments
    train_args = Seq2SeqTrainingArguments(output_dir=args.output_dir,
                            do_train=True,
                            do_eval=True,
                            # resume_from_checkpoint='outputs/p_tuning/checkpoint-70000',
                            evaluation_strategy="steps",
                            per_device_train_batch_size=args.bsz_per_device,
                            per_device_eval_batch_size=args.bsz_per_device,
                            learning_rate=args.lr,
                            lr_scheduler_type='polynomial',
                            max_steps=args.max_steps,
                            warmup_ratio=0.0,
                            local_rank=args.local_rank,
                            log_level='info',
                            log_level_replica='warning',
                            log_on_each_node=True,
                            logging_steps=5000,
                            eval_steps=5000,
                            save_steps=10000,
                            save_total_limit=100,
                            p_tuning = args.p_tuning,
                            contrasive = args.contrasive,
                            entity_pseudo_token = args.entity_pseudo_token,
                            relation_pseudo_token = args.relation_pseudo_token,
                            template = args.template,
                            ner_template = args.ner_template,
                            label_names = ['lm_labels', 'prompt_labels']
                            )

    config = BartConfig.from_pretrained(args.model_name_or_path)
    
    model = BartPretrainModel.from_pretrained(args.model_name_or_path, config=config, args = train_args)
    
    # defining trainer using 🤗
    trainer = Seq2SeqTrainer(model=model, 
                    args=train_args,
                    data_collator=data_collator, 
                    train_dataset=train_dataset, 
                    eval_dataset=eval_dataset,
                    optimizers=(None, None))
    

    # ## Training time
    trainer.train()
    # trainer.train(model_path='outputs/p_tuning/checkpoint-420000')
    # It will take hours to train this model on this dataset

    # lets save model
    trainer.evaluate(eval_dataset=eval_dataset)
    trainer.save_model(args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    # data
    # parser.add_argument("--model_name_or_path", default="facebook/bart-base", type=str, 
    #                     help="Path to pre-trained model checkpoints or shortcut name selected in the list: " )
    parser.add_argument("--model_name_or_path", default="outputs/p_tuning/checkpoint-old-420000", type=str, 
                        help="Path to pre-trained model checkpoints or shortcut name selected in the list: " )
    parser.add_argument("--output_dir", default='outputs/p_tuning/', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.", )
    parser.add_argument("--data_dir", default='dataset/wiki_NER/processed', type=str, 
                        help="The input data dir.", )
    # prompt learning
    parser.add_argument("--p_tuning", type=bool, default=True)
    parser.add_argument("--relation_pseudo_token", type=str, default='[PROMPT]')
    parser.add_argument("--entity_pseudo_token", type=str, default='[PROMPT]')

    parser.add_argument("--template", type=str, default="(1, 3, 3, 1)")
    parser.add_argument("--ner_template", type=str, default="(2, 4, 2)")

    # contractive learning
    parser.add_argument("--contrasive", type=bool, default=True)

    # train/dev settting
    parser.add_argument("--bsz_per_device", default=9, type=int, 
                        help="train/dev batch size per device", )

    parser.add_argument("--max_steps", default=1000000, type=int, 
                        help="the number of training steps. \
                            If set to a positive number, \
                            the total number of training steps to perform. \
                            and it will override any value given in num_train_epochs", )
    parser.add_argument("--lr", default=1e-5, type=float, 
                        help="learning rate", )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()

    args.template = eval(args.template) if type(args.template) is not tuple else args.template
    args.ner_template = eval(args.ner_template) if type(args.ner_template) is not tuple else args.ner_template

    assert type(args.template) is tuple
    assert type(args.ner_template) is tuple

    main(args)