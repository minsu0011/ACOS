export CUDA_VISIBLE_DEVICES=-1

BERT_BASE_DIR=./bert-base-uncased
DATA_DIR=.
TASK_NAME=quad
MODEL=quad
DOMAIN=rest16

echo 'DOMAIN is chosen from [rest16, laptop]'
python run_step1.py \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --domain_type $DOMAIN \
  --model_type $MODEL\
  --do_lower_case \
  --data_dir $DATA_DIR \
  --bert_model $BERT_BASE_DIR\
  --max_seq_length 128 \
  --train_batch_size 24 \
  --learning_rate 2e-5 \
  --num_train_epochs 30 \
  --output_dir ./output/Extract-Classify-QUAD/${DOMAIN}_1st/


python tokenized_data/get_1st_pairs.py . $DOMAIN

TASK_NAME=categorysenti
MODEL=categorysenti

python run_step2.py \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --domain_type $DOMAIN \
  --model_type $MODEL\
  --do_lower_case \
  --data_dir $DATA_DIR \
  --bert_model $BERT_BASE_DIR\
  --max_seq_length 128 \
  --train_batch_size 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 30 \
  --output_dir ./output/Extract-Classify-QUAD/${DOMAIN}_2nd/