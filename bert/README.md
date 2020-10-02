# Requirements
tensorflow >= 1.11.0   # CPU Version of TensorFlow.<br>
#tensorflow-gpu  >= 1.11.0  # GPU version of TensorFlow.<br>

(Tensorflow version 1.11 on Google Cloud TPU was used) <br>

# Generate random heads
Example: Switch on random 100 heads out of 144 heads <br>
python3 bert_generateheads.py 100 <br>

# For training,evaluation and prediction:
bash run_bert.sh <br>

The checkpoint used for BERT is: uncased_L-12_H-768_A-12 [Ref: <a href="https://github.com/google-research/bert">https://github.com/google-research/bert</a>]

# Hyperparameters details
This was run on four GLUE tasks: SST-2, QQP, MNLI-m and QNLI. <br>
Batch size used = 128 for all the tasks. <br>
Learning rates used for the four GLUE tasks are: <br>
(1) SST-2: 2e-5 <br>
(2) QQP: 5e-5 <br>
(3) MNLI-m: 3e-5 <br>
(4) QNLI: 5e-5 <br>
