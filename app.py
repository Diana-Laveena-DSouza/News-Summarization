from flask import Flask, request, render_template
from utils.extractive_summ_model import ExtractiveSummaryModel
from utils.abstractive_summ_model import AbstractiveSummaryModel
from utils.config import config

app = Flask(__name__)

# Load the Models
extractive_model = ExtractiveSummaryModel.load_from_checkpoint(checkpoint_path='extractive model/best-checkpoint.ckpt', hparams_file = "extractive model/hparams.yaml")
abstractive_model = AbstractiveSummaryModel.load_from_checkpoint(checkpoint_path='abstractive_model/best-checkpoint.ckpt', hparams_file = "abstractive_model/hparams.yaml")
extractive_model.freeze()
abstractive_model.freeze()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
# Function for Prediction
def predict():
    input_data = [x for x in request.form.values()]
    print(input_data)

    text_encoding = config.tokenizer(
        input_data,
        max_length=config.text_max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
        )

    generated_ids1 = extractive_model.model.generate(
        input_ids=text_encoding['input_ids'],
        attention_mask=text_encoding['attention_mask'],
        max_length=config.sum_max_length,
        num_beams=4,
        repetition_penalty=2.5,
        length_penalty=3.0,
        early_stopping=True
        )
    preds1 = [
        config.tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for gen_id in generated_ids1
        ]

    generated_ids2 = abstractive_model.model.generate(
        input_ids=text_encoding['input_ids'],
        attention_mask=text_encoding['attention_mask'],
        max_length=config.sum_max_length,
        num_beams=4,
        repetition_penalty=2.5,
        length_penalty=3.0,
        early_stopping=True
        )
    preds2 = [
        config.tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for gen_id in generated_ids2
        ]
    prediction1 = "".join(preds1)
    prediction2 = "".join(preds2)

    return render_template('index.html', post_text=input_data[0], extractive_text = prediction1, abstractive_text = prediction2)


if __name__ == "__main__":
    app.run(debug = True)