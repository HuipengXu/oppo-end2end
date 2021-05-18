python -m src.preprocess

#echo "Starting pretraining ..."
CUDA_VISIBLE_DEVICES=0 nohup python -m src.nezha_pretrain_r2 &> ./user_data/nezha-pretrain-log1.txt &
CUDA_VISIBLE_DEVICES=1 nohup python -m src.nezha_pretrain_r2_2 &> ./user_data/nezha-pretrain-log2.txt &
CUDA_VISIBLE_DEVICES=2 nohup python -m src.nezha_pretrain_r2_3 &> ./user_data/nezha-pretrain-log3.txt &
#CUDA_VISIBLE_DEVICES=3 nohup python -m src.bert_pretrain_r2 &> ./user_data/bert-pretrain-log1.txt &

#echo "Starting while waiting ..."
python -m src.while_waiting --stage pretrain

#echo "Starting finetuning ..."
CUDA_VISIBLE_DEVICES=0 nohup python -m src.train_nezha_r2 --model_dir model_ckpt-1 --adv pgd --seed 4321 &> ./user_data/nezha-finetune-log1.txt &
CUDA_VISIBLE_DEVICES=1 nohup python -m src.train_nezha_r2 --model_dir model_ckpt-2 --adv pgd --seed 780 &> ./user_data/nezha-finetune-log2.txt &
CUDA_VISIBLE_DEVICES=2 nohup python -m src.train_nezha_r2 --model_dir model_ckpt-3 --adv pgd --seed 12340 &> ./user_data/nezha-finetune-log3.txt &
#CUDA_VISIBLE_DEVICES=3 nohup python -m src.train_bert_r2 --model_dir model_ckpt-1 --adv pgd --seed 273 &> ./user_data/bert-finetune-log1.txt &

#echo "Starting while waiting ..."
python -m src.while_waiting --stage finetune

#echo "Starting export onnx model ..."
#python -m src.export_onnx --model_class bert --id 1
python -m src.export_onnx --model_class nezha --id 1
python -m src.export_onnx --model_class nezha --id 2
python -m src.export_onnx --model_class nezha --id 3

#echo "Starting inferring ..."
python -m src.ensemble_onnx

#echo "Finished ..."
