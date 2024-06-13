from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request

from bin.best_classification_model import classification_model

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return render_template('classification.html')


@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.get_json()
    text = data['text']
    is_new_model = data['isNewModel']

    model = classification_model(is_new_model=is_new_model)
    prediction = model.predict_new_data(text)
    accuracy = model.accuracy

    return jsonify({'result': prediction, 'accuracy': accuracy})


if __name__ == '__main__':
    app.run()
