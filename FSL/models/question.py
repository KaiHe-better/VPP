import sys
import os
sys.path.append('..')
import fewshot_re_kit
from .base_model import FewShotREModel
import torch
from torch import nn
from .prompt_encoder import PromptEncoder

WEIGHTS_NAME = "pytorch_model.bin"

def load_state_dict(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} #filter out unnecessary keys 
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


class Question(FewShotREModel):

    def __init__(self, sentence_encoder, Ptuing=False, template=(3,3,3,3), pseudo_token='[PROMPT]', prompt_encoder_ckpt=None):
        FewShotREModel.__init__(self, sentence_encoder)
        self.drop = nn.Dropout()
        self.Ptuing = Ptuing
        self.embeddings = self.sentence_encoder.module.model.shared
        
        if self.Ptuing:
            self.template = template
            self.spell_length = sum(self.template)
            self.prompt_encoder = PromptEncoder(self.spell_length, self.embeddings.embedding_dim)
            if prompt_encoder_ckpt is not None:
                pretrained_dict = torch.load(os.path.join(prompt_encoder_ckpt, WEIGHTS_NAME))
                load_state_dict(self.prompt_encoder, pretrained_dict)

            if pseudo_token not in self.sentence_encoder.module.tokenizer.get_vocab():
                self.sentence_encoder.module.tokenizer.add_special_tokens({'additional_special_tokens': [pseudo_token]})
            self.pseudo_token_id = self.sentence_encoder.module.tokenizer.get_vocab()[pseudo_token]

    def get_embed_inputs(self, sentences):
        sentences_for_embedding = sentences.clone()
        sentences_for_embedding[(sentences == self.pseudo_token_id)] = self.sentence_encoder.module.tokenizer.unk_token_id

        raw_embeds = self.embeddings(sentences_for_embedding)

        if self.Ptuing:
            bz = sentences.shape[0]
            # print(self.pseudo_token_id)
            # print(sentences.shape)
            # print((sentences == self.pseudo_token_id).nonzero().shape)
            blocked_indices = (sentences == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
            
            replace_embeds = self.prompt_encoder(torch.LongTensor(list(range(self.spell_length))).to(sentences.device))

            for bidx in range(bz):
                for i in range(self.spell_length):
                    raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        return raw_embeds

    def forward(self, input, N, K, train_flag=None):
        input_id = input["word"]
        input_embeding = self.get_embed_inputs(input_id)
        input["word_embed"] = input_embeding

        logits, last_hs = self.sentence_encoder(input)  # logits (N*N*K ,2)  last_hs (N*N*K, sen_len, 768)

        logits = logits.view(-1, N, K, N, 2) # (-1, N, K, N, 2)

        logits = logits.mean(2) # (-1, N, N, 2)
        # logits_na, _ = logits[:, :, :, 0].min(2, keepdim=True) # (-1, N, 1)
        # logits = logits[:, :, :, 1] # (-1, N, N)
        # logits = torch.cat([logits, logits_na], 2) # (B, N, N + 1)
        # _, pred = torch.max(logits.view(-1, N + 1), 1)

        logits = logits[:, :, :, 1] # (B, N, N)

        return logits, last_hs
