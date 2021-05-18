import torch

from transformers import BertTokenizer

import psutil
import numpy as np
import onnxruntime

from tqdm import trange

from flask import Flask, request

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

app = Flask(__name__)


def softmax(array):
    array -= array.max(axis=-1, keepdims=True)
    exp_array = np.exp(array)
    probs = exp_array / exp_array.sum(axis=1, keepdims=True)
    return np.mean(probs, axis=0)[1]


def init_model(onnx_model_path):
    sess_options = onnxruntime.SessionOptions()

    # Optional: store the optimized graph and view it using Netron to verify that model is fully optimized.
    # Note that this will increase session creation time so enable it for debugging only.
    # sess_options.optimized_model_filepath = './user_data/optimized_bert_gpu.onnx'

    # Please change the value according to best setting in Performance Test Tool result.
    sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)

    session = onnxruntime.InferenceSession(onnx_model_path, sess_options)
    return session


def infer(query_a, query_b):
    inputs = tokenizer([query_a, query_b], [query_b, query_a], return_tensors='pt',
                       add_special_tokens=True, truncation='longest_first', max_length=32)
    inputs = {key: value.numpy() for key, value in inputs.items()}
    # TODO: use IO Binding (see https://github.com/microsoft/onnxruntime/pull/4206) to improve performance.
    nezha1_logits = nezha1.run(None, inputs)[0]
    nezha2_logits = nezha2.run(None, inputs)[0]
    nezha3_logits = nezha3.run(None, inputs)[0]
    # bert1_logits = bert1.run(None, inputs)[0]
    logits = np.concatenate([nezha1_logits, nezha2_logits,
                             nezha3_logits], axis=0)
    prob = softmax(logits)
    return prob


@app.route("/tccapi", methods=['GET', 'POST'])
def tccapi():
    input_list = request.form.getlist('input')
    index_list = request.form.getlist('index')

    response_batch = {}
    response_batch["results"] = []
    for i in range(len(index_list)):
        index_str = index_list[i]

        response = {}
        try:
            input_sample = input_list[i].strip()
            elems = input_sample.strip().split("\t")
            query_a = elems[0].strip()
            query_b = elems[1].strip()
            predict = infer(query_a, query_b)
            response["predict"] = float(predict)
            response["index"] = index_str
            response["ok"] = True
        except Exception as e:
            response["predict"] = 0
            response["index"] = index_str
            response["ok"] = False
        response_batch["results"].append(response)

    return response_batch


if __name__ == '__main__':
    nezha1_path = './user_data/nezha-1.onnx'
    nezha2_path = './user_data/nezha-2.onnx'
    nezha3_path = './user_data/nezha-3.onnx'
    # bert1_path = './user_data/bert-1.onnx'
    vocab_path = './user_data/r2_vocab_total.txt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    nezha1 = init_model(nezha1_path)
    nezha2 = init_model(nezha2_path)
    nezha3 = init_model(nezha3_path)
    # bert1 = init_model(bert1_path)
    # for _ in trange(50000):
    #     infer('12 5 239 243 29 1001 126 1405 11', '29 485 12 251 1405 11')
    app.run(host="0.0.0.0", port=8080)
