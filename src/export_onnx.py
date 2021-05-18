import torch

from .nezha.modeling_nezha import NeZhaForSequenceClassification
from transformers import BertForSequenceClassification, BertTokenizer

import onnx

import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def export_onnx():
    model = model_class.from_pretrained(model_path, torchscript=True)
    model.eval().to(device)
    inputs = tokenizer("444 29 19", '444 2893 12', padding='max_length', truncation='longest_first',
                       max_length=32, return_tensors="pt")
    dummy_inputs = (
        inputs["input_ids"].to(device),
        inputs["attention_mask"].to(device),
        inputs["token_type_ids"].to(device),
    )
    input_names = ['input_ids', 'attention_mask', 'token_type_ids']
    output_names = ['logits']

    opset_version = 11
    with torch.no_grad():
        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        torch.onnx.export(model,
                          args=dummy_inputs,
                          f=export_model_path,
                          opset_version=opset_version,
                          do_constant_folding=True,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes={'input_ids': symbolic_names,
                                        'attention_mask': symbolic_names,
                                        'token_type_ids': symbolic_names},
                          verbose=True)
    onnx_model = onnx.load(export_model_path)
    onnx.checker.check_model(onnx_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_class', default='bert', choices=['bert', 'nezha'],
                        type=str)
    parser.add_argument('--id', default=1, type=int)

    args = parser.parse_args()

    model_path = f'./user_data/{args.model_class}-r2-results/checkpoint-{args.id}'
    vocab_path = './user_data/r2_vocab_total.txt'
    export_model_path = f'./user_data/{args.model_class}-{args.id}.onnx'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    models = {
        'bert': BertForSequenceClassification,
        'nezha': NeZhaForSequenceClassification,
    }
    model_class = models[args.model_class]
    export_onnx()
