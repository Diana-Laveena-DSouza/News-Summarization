#1. Import Libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import gc
import evaluate
from google.colab import drive
from utils.config import config
from utils.extractive_summ_model import ExtractiveSummaryModel

#2. Load File
drive.mount('/content/drive')
data = pd.read_csv("/content/drive/MyDrive/News_Summarize_Data/extractive_data.csv")
print(data.head())

#shape of the data
print(data.shape)

#3. EDA Analysis
data['article_length'] = data['text'].apply(lambda x : len(x.split()))
data['summary_length'] = data['summary'].apply(lambda x : len(x.split()))

#Statistics
print(data.describe())

#4. Data Preprocessing
data['text'] = data['text'].str.encode('ascii', 'ignore').str.decode('ascii')
data['summary'] = data['summary'].str.encode('ascii', 'ignore').str.decode('ascii')

#Train Test Split
train_df, test_df = train_test_split(data.loc[:, ['text', 'summary']], test_size=0.001, random_state=10)
print(train_df.shape, test_df.shape)

#Reset Index
train_df.reset_index(inplace = True,drop = True)
test_df.reset_index(inplace = True,drop = True)

#5. Create DataSet
#Class summaryDataset
class summaryDataset:
  
  def __init__(self, texts, summary):
    self.texts = texts
    self.summary = summary
  
  def __len__(self):
    return len(self.texts)

  def __getitem__(self, index):
    texts = self.texts[index]
    summary = self.summary[index]
  
    text_encoding = config.tokenizer(
        texts,
        max_length = config.text_max_length,
        padding = 'max_length',
        truncation = True,
        return_attention_mask = True,
        add_special_tokens=True,
        return_tensors = 'pt'
    )
  
    summary_encoding = config.tokenizer(
        summary,
        max_length = config.sum_max_length,
        padding = 'max_length',
        truncation = True,
        return_attention_mask = True,
        add_special_tokens = True,
        return_tensors = 'pt'
    )
    
    labels = summary_encoding['input_ids']

    return dict(text_input_ids = text_encoding['input_ids'].flatten(), 
                text_attention_mask = text_encoding['attention_mask'].flatten(),
                labels = labels.flatten(),
                labels_attention_mask = summary_encoding['attention_mask'].flatten())

#6. Create Data Loaders
train_dataset = summaryDataset(train_df['text'], train_df['summary'])
test_dataset = summaryDataset(test_df['text'], test_df['summary'])
train_dataloaders = DataLoader(train_dataset, batch_size = config.TRAIN_BATCH_SIZE, num_workers=2)
test_dataloaders = DataLoader(test_dataset, batch_size = config.VAL_BATCH_SIZE, num_workers=2)

#Print the First Data
for data in train_dataloaders:
  print(data)
  break

#Model Architecture
model = ExtractiveSummaryModel()
print(model)

#7. Model Training

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir ./tf_logs

checkpoint_callback = ModelCheckpoint(
    dirpath = 'checkpoints',
    filename = 'best-checkpoint',
    save_top_k = 1,
    verbose = True,
    monitor = 'val_loss',
    mode = 'min'
)

logger = TensorBoardLogger('tf_logs', name = 'news-extractive-summary')
earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='min')
trainer = pl.Trainer(
    logger = logger,
    callbacks=[earlystopper,checkpoint_callback],
    max_epochs = config.EPOCHS,
    accelerator = 'gpu',
    devices=1,
)

#Clear the Cache
torch.cuda.empty_cache()
del data
del train_df
del train_dataset
del test_dataset
gc.collect()

#8. Train the Model
trainer.fit(model, train_dataloaders=train_dataloaders, val_dataloaders=test_dataloaders)

#Load the Model
trained_model = ExtractiveSummaryModel.load_from_checkpoint(checkpoint_path='/content/checkpoints/best-checkpoint.ckpt', hparams_file = "/content/tf_logs/news-extractive-summary/version_0/hparams.yaml")
trained_model.freeze()

def summerizeText(text):
  text_encoding = config.tokenizer(
        text,
        max_length = config.text_max_length,
        padding = 'max_length',
        truncation = True,
        return_attention_mask = True,
        add_special_tokens=True,
        return_tensors = 'pt'
    )
    
  generated_ids = trained_model.model.generate(
      input_ids = text_encoding['input_ids'],
      attention_mask = text_encoding['attention_mask'],
      max_length = config.sum_max_length,
      num_beams = 4,
      repetition_penalty = 2.5,
      length_penalty = 3.0,
      early_stopping=True
  )
  preds = [
      config.tokenizer.decode(gen_id, skip_special_tokens = True, clean_up_tokenization_spaces = True)
      for gen_id in generated_ids
  ]
  return "".join(preds)

#9. Model Evaluation
summary_texts = [summerizeText(text) for text in test_df.loc[:, "text"].values]
rouge_score = evaluate.load("rouge")
scores = rouge_score.compute(predictions=summary_texts, references=test_df.loc[:, "summary"].values)
print(scores)

#10. Inference
text = test_df.loc[60, "text"]
print(summerizeText(text))
print(test_df.loc[60, "summary"])
