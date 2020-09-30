export GLUE_DIR="Mention Google bucket directory having GLUE Data eg: gs://my_bucket/glue_data"
export OUTPUT_DIR="Mention Google bucket directory for dumping Output files"
export BERT_BASE_DIR="Mention Google bucket directory containing the BERT checkpoint"
export APPLY_RP=True

# All ones i.e. no pruning. Replace the pruning mask here.
export PRUNING_MASK="[[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1]]"

python run_classifier.py \
  --use_tpu=true \
  --tpu_name="Mention TPU name" \
  --task_name= "Mention task name"\
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$GLUE_DIR/"Mention task name" \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --apply_rp=$APPLY_RP \
  --pruning_mask=$PRUNING_MASK &> ./log