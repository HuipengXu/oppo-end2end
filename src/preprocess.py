import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
from transformers import BertTokenizer


def reverse_data(df):
    data = df.apply(lambda x: x[0].strip() + ' ' + x[1].strip(), axis=1).values
    reverse_data = df.apply(lambda x: x[1].strip() + ' ' + x[0].strip(), axis=1).values
    return data, reverse_data


def generate_corpus(data, corpus_file_path):
    with open(corpus_file_path, 'w', encoding='utf8') as f:
        for row in tqdm(data, total=len(data)):
            f.write(row + '\n')


def generate_vocab(total_data, vocab_file_path):
    total_tokens = [token for sent in total_data for token in sent.split()]
    counter = Counter(total_tokens)
    vocab = [token for token, freq in counter.items()]
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + vocab
    with open(vocab_file_path, 'w', encoding='utf8') as f:
        f.write('\n'.join(vocab))


def read_data(train1_file_path, train2_file_path, data_cache_path, tokenizer):
    train1_df = pd.read_csv(train1_file_path, header=None, sep='\t')
    # train1_df = pd.read_csv(train1_file_path, header=None, sep='\t').head(200)
    train2_df = pd.read_csv(train2_file_path, header=None, sep='\t')
    # train2_df = pd.read_csv(train2_file_path, header=None, sep='\t').head(200)
    train_df = pd.concat([train1_df, train2_df], axis=0)

    data_df = {'train': train_df}
    processed_data = {}
    processed_reversed_data = {}

    for data_type, df in data_df.items():
        inputs = defaultdict(list)
        reversed_inputs = defaultdict(list)
        for i, row in tqdm(df.iterrows(), desc=f'Preprocessing {data_type} data', total=len(df)):
            label = row[2]
            sentence_a, sentence_b = row[0], row[1]
            build_bert_inputs(inputs, reversed_inputs, label, sentence_a, sentence_b, tokenizer)

        processed_data[data_type] = inputs
        processed_reversed_data[data_type] = reversed_inputs

    if not os.path.exists(data_cache_path):
        os.makedirs(data_cache_path)
    inputs_cache = os.path.join(data_cache_path, 'data.pkl')
    reversed_inputs_cache = os.path.join(data_cache_path, 'reversed_data.pkl')
    with open(inputs_cache, 'wb') as f:
        pickle.dump(processed_data, f)
    with open(reversed_inputs_cache, 'wb') as f:
        pickle.dump(processed_reversed_data, f)


def build_bert_inputs(inputs, reversed_inputs, label, sentence_a, sentence_b, tokenizer):
    inputs_dict = tokenizer.encode_plus(sentence_a, sentence_b, add_special_tokens=True,
                                        return_token_type_ids=True, return_attention_mask=True)
    reversed_inputs_dict = tokenizer.encode_plus(sentence_b, sentence_a, add_special_tokens=True,
                                                 return_token_type_ids=True, return_attention_mask=True)
    inputs['input_ids'].append(inputs_dict['input_ids'])
    inputs['token_type_ids'].append(inputs_dict['token_type_ids'])
    inputs['attention_mask'].append(inputs_dict['attention_mask'])
    inputs['labels'].append(label)

    reversed_inputs['input_ids'].append(reversed_inputs_dict['input_ids'])
    reversed_inputs['token_type_ids'].append(reversed_inputs_dict['token_type_ids'])
    reversed_inputs['attention_mask'].append(reversed_inputs_dict['attention_mask'])
    reversed_inputs['labels'].append(label)


def main():
    train_a_path = './tcdata/gaiic_track3_round1_train_20210228.tsv'
    test_a_path = './tcdata/gaiic_track3_round1_testA_20210228.tsv'
    test_b_path = './tcdata/gaiic_track3_round1_testB_20210317.tsv'
    train_r2_path = './tcdata/gaiic_track3_round2_train_20210407.tsv'
    vocab_file_path = './user_data/r2_vocab_total.txt'
    corpus_file_path = './user_data/r2_corpus.txt'
    reversed_corpus_file_path = './user_data/r2_reversed_corpus.txt'
    data_cache_path = './user_data'
    train_a_df = pd.read_csv(train_a_path, sep='\t', header=None)
    test_a_df = pd.read_csv(test_a_path, sep='\t', header=None)
    test_b_df = pd.read_csv(test_b_path, sep='\t', header=None)
    train_r2_df = pd.read_csv(train_r2_path, sep='\t', header=None)

    train_a_data, train_a_data_reversed = reverse_data(train_a_df)
    test_a_data, test_a_data_reversed = reverse_data(test_a_df)
    test_b_data, test_b_data_reversed = reverse_data(test_b_df)
    train_r2_data, train_r2_data_reversed = reverse_data(train_r2_df)

    total_data = np.concatenate([train_a_data, test_a_data, test_b_data, train_r2_data], axis=0)
    total_data_reversed = np.concatenate([train_a_data_reversed, test_a_data_reversed,
                                          test_b_data_reversed, train_r2_data_reversed], axis=0)

    assert len(total_data) == 450000, 'total examples should equal 450000'

    generate_corpus(total_data, corpus_file_path)
    generate_corpus(total_data_reversed, reversed_corpus_file_path)

    generate_vocab(total_data, vocab_file_path)

    tokenizer = BertTokenizer.from_pretrained(vocab_file_path)
    read_data(train_a_path, train_r2_path, data_cache_path, tokenizer)


if __name__ == '__main__':
    main()
