import requests
import json
import os
import time

url = "http://127.0.0.1:5000/cfg/"
cfg_path = 'E:\\Code\\Python\\config_kd.ini'

data = {"cfg_path":cfg_path}
response = requests.post(url, data=data)
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
exit()

data = {"img_path": input_dir, "file_name": file}
# 检查输入目录是否存在
if not os.path.exists(input_dir):
    print(f"输入路径 {input_dir} 不存在！")
    exit()

for file in os.listdir(input_dir):
    # 检查是否为文件
    file_path = os.path.join(input_dir, file)
    data = {"img_path":input_dir, "file_name": file}
    if not os.path.isfile(file_path):
        continue

    try:
        # response = requests.post(url, json={"img_path": input_dir, "file_name": file})
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print(f"文件 {file} 响应: {response.json()}")
        else:
            print(f"文件 {file} 请求失败，状态码: {response.status_code}, 响应: {response.text}")
    except Exception as e:
        print(f"请求文件 {file} 时发生错误: {e}")

    time.sleep(0.5)  # 动态调整为合理的值
