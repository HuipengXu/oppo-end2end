import os
import json
import warnings
import argparse
import random
import pickle
from zipfile import ZipFile
from tqdm import tqdm, trange
from collections import defaultdict
from typing import Dict, Tuple, Iterable, List

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
# from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import AdamW, get_linear_schedule_with_warmup

from transformers import (
    BertPreTrainedModel,
    BertTokenizer,
    BertModel
)
from transformers.modeling_outputs import SequenceClassifierOutput

from .utils import WarmupLinearSchedule, Lookahead

warnings.filterwarnings('ignore')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def load_data(config, tokenizer):
    with open(config['data_cache_path'], 'rb') as f:
        data = pickle.load(f)
    train_dev_data = data['train']
    collate_fn = Collator(config['max_seq_len'], tokenizer)
    return collate_fn, train_dev_data


def load_cv_data(collate_fn, config, dev_idxs, train_dev_data, train_idxs, all_data=False):
    train_data = defaultdict(list)
    dev_data = defaultdict(list)
    for key, values in train_dev_data.items():
        train_data[key] = [values[idx] for idx in train_idxs][:100]
        dev_data[key] = [values[idx] for idx in dev_idxs]
    if all_data:
        train_data = train_dev_data
    train_dataset = OppoDataset(train_data)
    dev_dataset = OppoDataset(dev_data)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'],
                                  shuffle=True, num_workers=4, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=config['batch_size'],
                                shuffle=False, num_workers=4, collate_fn=collate_fn)
    return dev_dataloader, train_dataloader


class OppoDataset(Dataset):

    def __init__(self, data_dict: dict):
        super(OppoDataset, self).__init__()
        self.data_dict = data_dict

    def __getitem__(self, index: int) -> tuple:
        data = (self.data_dict['input_ids'][index], self.data_dict['token_type_ids'][index],
                self.data_dict['attention_mask'][index], self.data_dict['labels'][index])
        return data

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])


class Collator:
    def __init__(self, max_seq_len: int, tokenizer: BertTokenizer):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def pad_and_truncate(self, input_ids_list, token_type_ids_list,
                         attention_mask_list, labels_list, max_seq_len):
        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)
        for i in range(len(input_ids_list)):
            seq_len = len(input_ids_list[i])
            if seq_len <= max_seq_len:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i], dtype=torch.long)
                token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i], dtype=torch.long)
                attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i], dtype=torch.long)
            else:
                input_ids[i] = torch.tensor(input_ids_list[i][:max_seq_len - 1] + [self.tokenizer.sep_token_id],
                                            dtype=torch.long)
                token_type_ids[i] = torch.tensor(token_type_ids_list[i][:max_seq_len], dtype=torch.long)
                attention_mask[i] = torch.tensor(attention_mask_list[i][:max_seq_len], dtype=torch.long)

        labels = torch.tensor(labels_list, dtype=torch.long)
        return input_ids, token_type_ids, attention_mask, labels

    def __call__(self, examples: list) -> dict:
        input_ids_list, token_type_ids_list, attention_mask_list, labels_list = list(zip(*examples))
        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)

        input_ids, token_type_ids, attention_mask, labels = self.pad_and_truncate(input_ids_list, token_type_ids_list,
                                                                                  attention_mask_list, labels_list,
                                                                                  max_seq_len)

        data_dict = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

        return data_dict


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.1, emb_name='embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD:
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=0.1, alpha=0.3, emb_name='embeddings', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)])
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        logits_list = []

        for dropout in self.dropouts:
            out = dropout(pooled_output)
            logits = self.classifier(out)
            logits_list.append(logits)

        logits = torch.stack(logits_list, dim=0).mean(dim=0)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def train(config, train_dataloader):
    model = BertForSequenceClassification.from_pretrained(config['model_path'])

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": config['weight_decay']},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config['learning_rate'],
                      correct_bias=False, eps=1e-8)
    optimizer = Lookahead(optimizer, 5, 1)
    total_steps = config['num_epochs'] * len(train_dataloader)
    lr_scheduler = WarmupLinearSchedule(optimizer,
                                        warmup_steps=int(config['warmup_ratio'] * total_steps),
                                        t_total=total_steps)
    model.to(config['device'])
    if config['adv'] == 'fgm':
        fgm = FGM(model)
    else:
        pgd = PGD(model)
        K = 3
    epoch_iterator = trange(config['num_epochs'])
    global_steps = 0
    train_loss = 0.

    if config['n_gpus'] > 1:
        model = nn.DataParallel(model)

    for _ in epoch_iterator:

        train_iterator = tqdm(train_dataloader, desc='Training', total=len(train_dataloader))
        model.train()
        for batch in train_iterator:
            batch_cuda = {item: value.to(config['device']) for item, value in list(batch.items())}
            loss = model(**batch_cuda)[0]
            if config['n_gpus'] > 1:
                loss = loss.mean()
            loss.backward()

            if config['adv'] == 'fgm':
                fgm.attack()
                loss_adv = model(**batch_cuda)[0]
                if config['n_gpus'] > 1:
                    loss_adv = loss_adv.mean()
                loss_adv.backward()
                fgm.restore()
            else:
                pgd.backup_grad()
                for t in range(K):
                    pgd.attack(is_first_attack=(t == 0))
                    if t != K - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    loss_adv = model(**batch_cuda)[0]
                    if config['n_gpus'] > 1:
                        loss_adv = loss_adv.mean()
                    loss_adv.backward()
                pgd.restore()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if config['ema_start']:
                ema.update()

            train_loss += loss.item()
            global_steps += 1

            train_iterator.set_postfix_str(f'running training loss: {loss.item():.4f}')

            if global_steps >= config['ema_start_step'] and not config['ema_start']:
                print('\n>>> EMA starting ...')
                config['ema_start'] = True
                ema = EMA(model.module if hasattr(model, 'module') else model, decay=0.999)

    ema.apply_shadow()
    model_save_path = os.path.join(config['output_path'], f'checkpoint-{config["model_path"][-1]}')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(model_save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='model_ckpt-1', type=str)
    parser.add_argument('--adv', default='pgd', type=str)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    if args.model_dir == 'model_ckpt-1':
        data_cache_path = './user_data/data.pkl'
    else:
        data_cache_path = './user_data/reversed_data.pkl'

    config = {
        'data_cache_path': data_cache_path,
        'output_path': './user_data/bert-r2-results',
        'vocab_path': './user_data/r2_vocab_total.txt',
        'model_path': f'./user_data/self-pretrained-bert-base-r2/{args.model_dir}',
        'all': True,
        'batch_size': 64,
        'num_epochs': 3,
        # 'num_epochs': 1,
        'num_folds': 20,
        'cv': '',
        'adv': args.adv,
        'max_seq_len': 30,
        'learning_rate': 2e-5,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'device': 'cuda',
        'logging_step': 500,
        # 'logging_step': 1,
        'ema_start_step': 1500,
        # 'ema_start_step': 1,
        'ema_start': False,
        'seed': args.seed
    }

    if not torch.cuda.is_available():
        config['device'] = 'cpu'
    else:
        config['n_gpus'] = torch.cuda.device_count()
        config['batch_size'] *= config['n_gpus']

    if not os.path.exists(config['output_path']):
        os.makedirs((config['output_path']))

    tokenizer = BertTokenizer.from_pretrained(config['vocab_path'])
    collate_fn, train_dev_data = load_data(config, tokenizer)

    skf = StratifiedKFold(shuffle=True, n_splits=config['num_folds'], random_state=config['seed'])
    for train_idxs, dev_idxs in skf.split(X=train_dev_data['input_ids'], y=train_dev_data['labels']):
        dev_dataloader, train_dataloader = load_cv_data(collate_fn, config, dev_idxs,
                                                        train_dev_data, train_idxs, all_data=config['all'])

        seed_everything(config['seed'])

        train(config, train_dataloader)

        if not config['cv']:
            break


if __name__ == '__main__':
    main()
