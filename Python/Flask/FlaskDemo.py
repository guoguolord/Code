from flask import Flask, jsonify, request
import os
import argparse
import time
from PIL import Image

app = Flask(__name__)
def data_handler(img_path, file_name):
    file_path = os.path.join(img_path, file_name)
    if not os.path.isfile(file_path):
        print('file not exist')
        return None, f'{file_path} not exist'
    try:
        file = Image.open(file_path)
        # 读取图片长宽
        width, height = file.size

        return (width, height), None
    except Exception as e:
        print(e)
        return None, f'{file_path} open failed'

@app.route('/task', methods=['POST'])
def task():
    request_data = request.get_json()
    img_path = request_data['img_path']
    file_name = request_data['file_name']
    if not img_path or not file_name:
        return jsonify({"error": "请求数据缺失"}), 400

    dimensions, error = data_handler(img_path, file_name)
    if error:
        return jsonify({"error": error}), 400

    return jsonify({file_name: {"width": dimensions[0], "height": dimensions[1]}})

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--debug', type=bool, default=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args()
    app.run(debug=args.debug,port=args.port,host=args.host)