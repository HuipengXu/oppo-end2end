import os
import argparse


def start_finetune():
    is_start = (os.path.exists(nezha_model_file1) and os.path.exists(nezha_model_file2)
                and os.path.exists(nezha_model_file3))
    return is_start


def while_wait():
    while True:
        is_start = start_finetune()
        if is_start:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='pretrain', type=str)
    args = parser.parse_args()
    if args.stage == 'pretrain':
        nezha_model_dir = './user_data/self-pretrained-nezha-base-r2'
        nezha_model_file1 = os.path.join(nezha_model_dir, 'model_ckpt-1/pytorch_model.bin')
        nezha_model_file2 = os.path.join(nezha_model_dir, 'model_ckpt-2/pytorch_model.bin')
        nezha_model_file3 = os.path.join(nezha_model_dir, 'model_ckpt-3/pytorch_model.bin')
    else:
        nezha_model_dir = './user_data/nezha-r2-results'
        nezha_model_file1 = os.path.join(nezha_model_dir, 'checkpoint-1/pytorch_model.bin')
        nezha_model_file2 = os.path.join(nezha_model_dir, 'checkpoint-2/pytorch_model.bin')
        nezha_model_file3 = os.path.join(nezha_model_dir, 'checkpoint-3/pytorch_model.bin')
    while_wait()
