import requests


def result_counts(result_data):
    # 初始化计数器
    success_count = 0
    failed_count = 0

    # 遍历结果
    for result in result_data.get("results", []):
        if result.get("status") == "success":
            success_count += 1
        elif result.get("status") == "failed":
            failed_count += 1

    return success_count,failed_count


# 服务端的URL
url = "http://127.0.0.1:6666/detect/"

# 待检测图片文件夹路径
folder_path = "./InputImg"

# 设置请求的其他参数
data = {
    'folder_path': folder_path,  # 传递图片文件夹路径
    'draw': True,  # 是否画出结果图
    'export_json': True  # 是否将结果导出为JSON
}

print("...start to send requests...")
# 发送请求到服务端
response = requests.post(url, data=data)

# 打印请求的响应状态
if response.status_code == 200:
    # 解析JSON响应数据
    result_data = response.json()

    print(f"* det result response:\n{result_data}")

    success_count,failed_count = result_counts(result_data)

    # 打印详细结果
    print('\n----------- result_counts ------------')
    print(f"Successful detections: {success_count}")
    print(f"Failed detections: {failed_count}")

else:
    print(f"Error: Received unexpected status code {response.status_code}")
