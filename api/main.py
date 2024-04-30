# import flast module
from fastapi import FastAPI,Request
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoConfig, AutoTokenizer
import numpy as np
import uvicorn
import json

# instance of flask application
app = FastAPI()


#############################################
# Loading Model
#############################################
def get_model():
    model_name = ("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained("Yuhhi/ai-text-detector")
    return tokenizer,model


tokenizer,model = get_model()


# home route that returns below text when root url is accessed
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
async def read_root(request: Request):
    data = await request.json()
    # print(data['text'])
    if 'text' in data:

        text = data['text']

        encoded_input = tokenizer( text, truncation=True, padding=True, max_length=128, return_tensors="tf")

        tf_input = tf.data.Dataset.from_tensor_slices((dict(encoded_input)))

        predictions = model.predict(tf_input)
        predicted_probabilities = tf.nn.softmax(predictions.logits, axis=1)
        predicted_labels = tf.argmax(predicted_probabilities, axis=1)

        result = {
            "label": str(predicted_labels.numpy()[0]),
            "human-genrated": str(np.format_float_positional(predicted_probabilities.numpy()[0][0] * 100)),
            "ai-generated": str(np.format_float_positional(predicted_probabilities.numpy()[0][1] * 100))
        }
        
        return result
    else:
        return {"Recieved Text": "No Text Found"}


if __name__ == "__main__":
    uvicorn.run("main:app", port=8080, reload=True)
