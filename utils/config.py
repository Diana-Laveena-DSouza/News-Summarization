#1. Import Libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import T5ForConditionalGeneration, T5TokenizerFast

#Class Configuration
class config:
  text_max_length = 300
  sum_max_length = 80
  EPOCHS =1
  TRAIN_BATCH_SIZE = 16
  VAL_BATCH_SIZE = 4
  tokenizer = T5TokenizerFast.from_pretrained('t5-base')


