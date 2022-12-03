#1. Import Libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import pytorch_lightning as pl
from torch.optim import AdamW

#Class ExtractiveSummaryModel
class ExtractiveSummaryModel(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)
    
  def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
    output = self.model(
          input_ids=input_ids,
          attention_mask=attention_mask,
          decoder_attention_mask = decoder_attention_mask,
         labels = labels
  )
    return output.loss, output.logits

  def training_step(self, batch, batch_size):
    input_ids = batch['text_input_ids']
    attention_mask = batch['text_attention_mask']
    labels = batch['labels']
    labels_attention_mask = batch['labels_attention_mask']

    loss, outputs = self(
        input_ids = input_ids,
        attention_mask = attention_mask,
        decoder_attention_mask = labels_attention_mask,
        labels = labels
    )

    self.log('train_loss', loss, prog_bar=True, logger=True)
    return loss

  def validation_step(self, batch, batch_size):
    input_ids = batch['text_input_ids']
    attention_mask = batch['text_attention_mask']
    labels = batch['labels']
    labels_attention_mask = batch['labels_attention_mask']

    loss, outputs = self(
        input_ids = input_ids,
        attention_mask = attention_mask,
        decoder_attention_mask = labels_attention_mask,
        labels = labels
    )

    self.log('val_loss', loss, prog_bar=True, logger=True)
    return loss

  def configure_optimizers(self):
      return AdamW(self.parameters(), lr = 3e-5)