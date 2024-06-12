from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return render_template('classification.html')


@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.get_json()
    text = data['text']
    # 简单的文本处理：转换为大写
    result = text.upper()
    return jsonify({'result': result})


if __name__ == '__main__':
    app.run()
