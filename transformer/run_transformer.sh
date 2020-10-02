export PATH="/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
export PATH=/tools/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=/tools/cuda-9.0/lib64:/tools/cuda-9.0/extras/CUPTI/lib64:/scratch/scratch4/preksha/cuda:$LD_LIBRARY_PATH
export CUDA_HOME=/tools/cuda-9.0

source activate tensorflow_p36


cd /vol/transformer/models/official/transformer # Set this to the path ...../models/official/transformer
export PYTHONPATH=$PYTHONPATH:/vol/transformer/models # Set this to the path ...../models
DATA_DIR="/vol/transformer/models/official/transformer/data_ende" # Put the data here in the form of tfrecords
PARAM_SET=base
VOCAB_FILE=$DATA_DIR/vocab.ende.32768
MODEL_DIR="/vol/exps/model_dir" # Mention the path of the output directory here
concrete_heads="[]" # Don't change this when running code for random pruning! This was added for testing some other functionality. 
train_epochs=25
checkpoint_step=1220851 # Mention the checkpoint number from where you need to start fine-tuning.

# All ones configuration i.e. no pruning. Change the pruning masks below to apply pruning.
enc_self="[[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]]"
enc_dec="[[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]]"
dec_self="[[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]]"

CUDA_VISIBLE_DEVICES=0 python transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --vocab_file=$VOCAB_FILE --param_set=$PARAM_SET --bleu_source=$DATA_DIR/newstest2014.en --bleu_ref=$DATA_DIR/newstest2014.de --alive_heads_enc_self=$enc_self --alive_heads_dec_self=$dec_self --alive_heads_enc_dec=$enc_dec --concrete_heads=$concrete_heads --train_epochs=$train_epochs --checkpoint_step=$checkpoint_step &> $MODEL_DIR/log

#python translate.py --model_dir=$MODEL_DIR --vocab_file=$VOCAB_FILE --param_set=$PARAM_SET --file=$DATA_DIR/newstest2014.en --file_out=$MODEL_DIR/translations.de --alive_heads_enc_self=$enc_self --alive_heads_dec_self=$dec_self --alive_heads_enc_dec=$enc_dec --concrete_heads=$concrete_heads &> $MODEL_DIR/translations_log

#python compute_bleu.py --translation=$MODEL_DIR/translations.de  --reference=$DATA_DIR/newstest2014.de &> $MODEL_DIR/bleu_log

