# Requirements
tensorflow >= 1.11.0   # CPU Version of TensorFlow.<br>
#tensorflow-gpu  >= 1.11.0  # GPU version of TensorFlow.<br>

(Tensorflow version 1.11 on Google Cloud TPU was used) <br>

# Generate random heads
Example: Switch on random 100 heads out of 144 heads <br>
python3 bert_generateheads.py 100 <br>

# To run train/eval/prediction run_bert.sh
bash run_bert.sh <br>

The checkpoint used for BERT is: uncased_L-12_H-768_A-12 [Ref: <a href="https://github.com/google-research/bert">https://github.com/google-research/bert</a>]
