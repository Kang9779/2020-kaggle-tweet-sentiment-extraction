#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import pandas as pd
import tokenizers
import transformers
from transformers import RobertaModel,RobertaConfig
from sklearn.model_selection import StratifiedKFold
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import torchcontrib
import tqdm
import utils


class config:
    SEED = 25
    N_FOLDS = 5
    MAX_LEN = 86
    EPOCHS = 3
    HIDDEN_SIZE = 768
    LEARNING_RATE = 0.2 * 3e-5
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 8
    TRAINING_FILE = "./data/train_folds.csv"
    ROBERTA_PATH = "./torch-roberta"
    MODEL_SAVE_PATH = './roberta-model'
    TOKENIZER = tokenizers.ByteLevelBPETokenizer(
        vocab=f"{ROBERTA_PATH}/vocab.json", 
        merges=f"{ROBERTA_PATH}/merges.txt", 
        lowercase=True,
        add_prefix_space=True
    )
    USE_SWA = False
    WEIGHT_DECAY = 0.001
    N_LAST_HIDDEN = 12
    HIGH_DROPOUT = 0.5
    SOFT_ALPHA = 0.6
    WARMUP_RATIO = 0.25
    WEIGHT_DECAY = 0.001
    USE_SWA = False
    SWA_RATIO = 0.9
    SWA_FREQ = 30

def process_data(tweet, selected_text, sentiment, tokenizer, max_len):
    tweet = " " + " ".join(str(tweet).split())
    selected_text = " " + " ".join(str(selected_text).split())

    len_st = len(selected_text) - 1
    idx0 = None
    idx1 = None
    
    # 找到select_text在tweet中的起始和终点位置
    for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
        if " " + tweet[ind: ind+len_st] == selected_text:
            idx0 = ind
            idx1 = ind + len_st - 1
            break

    char_targets = [0] * len(tweet)
    if idx0 != None and idx1 != None:
        for ct in range(idx0, idx1 + 1):
            char_targets[ct] = 1
    
    tok_tweet = tokenizer.encode(tweet)
    input_ids_orig = tok_tweet.ids
    tweet_offsets = tok_tweet.offsets
    
    target_idx = []
    for j, (offset1, offset2) in enumerate(tweet_offsets):
        if sum(char_targets[offset1: offset2]) > 0:
            target_idx.append(j)
    
    targets_start = target_idx[0]
    targets_end = target_idx[-1]

    # tokenizer.encode("positive") == 1313
    # tokenizer.encode("negative") == 2430
    # tokenizer.encode("neutral") == 7974
    sentiment_id = {
        'positive': 1313,
        'negative': 2430,
        'neutral': 7974
    }
    
    input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
    token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
    targets_start += 4
    targets_end += 4

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([1] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
    
    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'targets_start': targets_start,
        'targets_end': targets_end,
        'orig_tweet': tweet,
        'orig_selected': selected_text,
        'sentiment': sentiment,
        'offsets': tweet_offsets
    }

class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
    
    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        data = process_data(
            self.tweet[item], 
            self.selected_text[item], 
            self.sentiment[item],
            self.tokenizer,
            self.max_len
        )

        return {
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
            'orig_tweet': data["orig_tweet"],
            'orig_selected': data["orig_selected"],
            'sentiment': data["sentiment"],
            'offsets': torch.tensor(data["offsets"], dtype=torch.long)
        }


class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = RobertaModel.from_pretrained(
            config.ROBERTA_PATH,
            config=conf)
        self.high_dropout = torch.nn.Dropout(config.HIGH_DROPOUT)
        self.classifier = torch.nn.Linear(config.HIDDEN_SIZE * 2, 2)

        torch.nn.init.normal_(self.classifier.weight, std=0.02)

    def forward(self, ids, mask, token_type_ids):
        # sequence_output of N_LAST_HIDDEN + Embedding states
        # (N_LAST_HIDDEN + 1, batch_size, num_tokens, 768)
        _, _, out = self.roberta(ids, attention_mask=mask,
                                 token_type_ids=token_type_ids)

        out = torch.stack(
            tuple(out[-i - 1] for i in range(config.N_LAST_HIDDEN)), dim=0)
        out_mean = torch.mean(out, dim=0)
        out_max, _ = torch.max(out, dim=0)
        out = torch.cat((out_mean, out_max), dim=-1)

        # Multisample Dropout: https://arxiv.org/abs/1905.09788
        logits = torch.mean(torch.stack([
            self.classifier(self.high_dropout(out))
            for _ in range(5)
        ], dim=0), dim=0)

        start_logits, end_logits = logits.split(1, dim=-1)

        # (batch_size, num_tokens)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits,end_logits

def loss_fn(start_logits, end_logits, start_positions, end_positions):
    """
    Return the sum of the cross entropy losses for both the start and end logits
    """
    loss_fct = torch.nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss)
    return total_loss
    
def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    losses = utils.AverageMeter()

    tk0 = tqdm.tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk0):
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        start_labels = d['targets_start']
        end_labels = d['targets_end']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        start_labels = start_labels.to(device, dtype=torch.long)
        end_labels = end_labels.to(device, dtype=torch.long)

        model.zero_grad()
        outputs_start, outputs_end = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(outputs_start, outputs_end,
                       start_labels, end_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)


def calculate_jaccard_score(
    original_tweet, 
    target_string, 
    sentiment_val, 
    idx_start, 
    idx_end, 
    offsets,
    verbose=False):
    
    if idx_end < idx_start:
        idx_end = idx_start
    
    filtered_output  = ""
    for ix in range(idx_start, idx_end + 1):
        filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
            filtered_output += " "

    if len(original_tweet.split()) < 2:
        filtered_output = original_tweet

    jac = utils.jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output


def eval_fn(data_loader, model, device):
    model.eval()
    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()
    
    with torch.no_grad():
        for bi, d in enumerate(data_loader):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            offsets = d["offsets"].cpu().numpy()

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)

            outputs_start, outputs_end = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            jaccard_scores = []
            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                jaccard_score, _ = calculate_jaccard_score(
                    original_tweet=tweet,
                    target_string=selected_tweet,
                    sentiment_val=tweet_sentiment,
                    idx_start=np.argmax(outputs_start[px, :]),
                    idx_end=np.argmax(outputs_end[px, :]),
                    offsets=offsets[px]
                )
                jaccard_scores.append(jaccard_score)

            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))

    return jaccards.avg

def run(fold):
    dfx = pd.read_csv(config.TRAINING_FILE)

    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

    train_dataset = TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4,
        shuffle=True)

    valid_dataset = TweetDataset(
        tweet=df_valid.text.values,
        sentiment=df_valid.sentiment.values,
        selected_text=df_valid.selected_text.values)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=4,
        shuffle=False)

    device = torch.device('cpu')
    model_config = RobertaConfig.from_pretrained(config.ROBERTA_PATH)
    model_config.output_hidden_states = True
    model = TweetModel(conf=model_config)
    model = model.to(device)

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': config.WEIGHT_DECAY},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]
    
    base_opt = AdamW(optimizer_parameters,lr=config.LEARNING_RATE)
    optimizer = torchcontrib.optim.SWA(
        base_opt,
        swa_start=int(num_train_steps * config.SWA_RATIO),
        swa_freq=config.SWA_FREQ,
        swa_lr=None)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(num_train_steps * config.WARMUP_RATIO),
        num_training_steps=num_train_steps)

    print(f'Training is starting for fold = {fold}')

    for epoch in range(config.EPOCHS):
        train_fn(train_data_loader, model, optimizer,device,scheduler=scheduler)
        jaccard = eval_fn(valid_data_loader, model, device)

    if config.USE_SWA:
        optimizer.swap_swa_sgd()

    torch.save(model.state_dict(),f'{config.MODEL_SAVE_PATH}/model_{fold}.bin')

    return jaccard


if __name__ == '__main__':
    utils.seed_everything(seed=config.SEED)

    fold_scores = []
    for i in range(config.N_FOLDS):
        fold_score = run(i)
        fold_scores.append(fold_score)

    print('\nScores without SWA:')
    for i in range(config.N_FOLDS):
        print(f'Fold={i}, Jaccard = {fold_scores[i]}')
    print(f'Mean = {np.mean(fold_scores)}')
    print(f'Std = {np.std(fold_scores)}')
