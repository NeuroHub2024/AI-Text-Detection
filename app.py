# import flast module
from flask import Flask, request, jsonify
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoConfig, AutoTokenizer
import numpy as np

# instance of flask application
app = Flask(__name__)


#############################################
# Loading Model
#############################################


db_config = AutoConfig.from_pretrained("config.json")
reloaded_model = TFAutoModelForSequenceClassification.from_pretrained(
    "tf_model.h5", config=db_config
)

model_name = (
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"  # Example model name
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# home route that returns below text when root url is accessed
@app.route("/")
def hello_world():
    return "Hello World"


@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        data = request.json
        text = data.get("text")

        encoded_input = tokenizer(
            text, truncation=True, padding=True, max_length=128, return_tensors="tf"
        )
        tf_input = tf.data.Dataset.from_tensor_slices((dict(encoded_input)))

        predictions = reloaded_model.predict(tf_input)
        predicted_probabilities = tf.nn.softmax(predictions.logits, axis=1)
        predicted_labels = tf.argmax(predicted_probabilities, axis=1)

        result = {
            "label": predicted_labels.numpy()[0],
            "human-genrated": np.format_float_positional(
                predicted_probabilities.numpy()[0][0] * 100
            ),
            "ai-generated": np.format_float_positional(
                predicted_probabilities.numpy()[0][1] * 100
            ),
        }
        # print("Predicted label:", predicted_labels.numpy()[0])
        # print("Predicted probability:", predicted_probabilities.numpy()[0])

        return jsonify(str(result))
    else:
        return jsonify({"error": "Invalid JSON data in request body"}), 400
