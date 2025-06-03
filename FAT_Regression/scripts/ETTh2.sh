export CUDA_VISIBLE_DEVICES=0

MODEL=FAT
EXP_NAME=ETTh2

python -u run.py \
    --task_type reg \
    --task_name pretrain \
    --root_path ./dataset/ETT-small_old/ \
    --data_path ETTh2.csv \
    --model $MODEL \
    --data ETTh2 \
    --features M \
    --pretrain_mode 1 \
    --d_model 8 \
    --n_heads 8 \
    --e_layers 2 \
    --d_ff 32 \
    --learning_rate 0.001 \
    --seq_len 96 \
    --batch_size 16 \
    --pretrain_epochs 50 \
    --mask_rate 0.5 \
    --lm 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --positive_nums 3 \
    --exp_name $EXP_NAME \
    --n_knlg 32

for PRED_LEN in 96 192 336 720; do
  python -u run.py \
      --task_type reg \
      --task_name finetune \
      --root_path ./dataset/ETT-small_old/ \
      --data_path ETTh2.csv \
      --model $MODEL \
      --data ETTh2 \
      --features M \
      --freeze 1 \
      --d_model 8 \
      --n_heads 8 \
      --e_layers 2 \
      --d_ff 32 \
      --learning_rate 0.0002 \
      --batch_size 16 \
      --train_epochs 20 \
      --dropout 0.1 \
      --head_dropout 0.1 \
      --seq_len 96 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --positive_nums 3 \
      --exp_name $EXP_NAME \
      --pred_len $PRED_LEN \
      --patience 5 \
      --n_knlg 32
done