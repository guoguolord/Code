from fastapi import FastAPI, Form
import os
from tool.v8Detection import v8Detection
from JoTools.txkjRes.deteRes import DeteRes
from JoTools.utils.FileOperationUtil import FileOperationUtil
import copy
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm
import uvicorn

app = FastAPI()

# 初始化模型
model_anjian = None
save_xml_dir = './result_xmlDir'
save_res_pic_dir = './result_picDir'
json_folder = './result_jsonDir'


# 用于将 XML 文件转换为 JSON 的工具函数
def parse_xml_to_json(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    json_data = []
    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)

    # 遍历所有对象
    for obj in root.findall('object'):
        label = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        prob = float(obj.find('prob').text)

        width = xmax - xmin
        height = ymax - ymin

        json_obj = {
            "Box": {
                "Angle": 0,
                "Height": height,
                "Width": width,
                "X": xmin,
                "Y": ymin
            },
            "Score": prob,
            "label": label
        }
        json_data.append(json_obj)

    return json_data


def delTmpFiles(data_path, name):
    """
    删除临时文件的工具函数
    """
    for i in os.listdir(data_path):
        file_path = os.path.join(data_path, i)
        if os.path.isfile(file_path) and name in i:
            os.remove(file_path)
        elif os.path.isdir(file_path):
            delTmpFiles(file_path, name)


@app.on_event("startup")
async def startup_event():
    """FastAPI启动时，初始化模型，并创建或清理目录"""
    global model_anjian
    modelDir = './models'
    config_path = './models/config_sdtd-3cls.ini'
    model_anjian = v8Detection(modelDir=modelDir, modelId='det', cfgPath=config_path)
    print("* Model Initialized!")

    # 创建并清理保存结果的文件夹
    os.makedirs(save_xml_dir, exist_ok=True)
    os.makedirs(save_res_pic_dir, exist_ok=True)
    os.makedirs(json_folder, exist_ok=True)

    # 清理旧的文件
    delTmpFiles(save_xml_dir, '.')
    delTmpFiles(save_res_pic_dir, '.')
    delTmpFiles(json_folder, '.')


@app.post("/detect/")
async def detect_image(
        folder_path: str = Form(...),  # 接收图片文件夹路径
        draw: bool = Form(False),
        export_json: bool = Form(False)
):
    """
    传递图片文件夹路径进行目标检测，逐个返回图片的检测结果。
    - `folder_path`: 输入的图片文件夹路径
    - `draw`: 是否画出结果图
    - `export_json`: 是否将XML结果转换为JSON返回
    """
    response = {"results": []}

    print(f"* start to detect_image in dir:{folder_path}")
    print(f"* draw = {draw}, export_json={export_json}")

    # 遍历文件夹中的所有图片
    pic_list = [os.path.join(folder_path, p) for p in os.listdir(folder_path) if
                os.path.splitext(p)[-1].lower() in ['.jpg', '.jpeg', '.png']]

    for each_img_path in tqdm(pic_list,colour='blue'):
        img_name = os.path.basename(each_img_path)
        try:
            # 加载图片并推理
            a = DeteRes()
            a.img_path = each_img_path
            img = a.get_img_array(RGB=True)
            res = model_anjian.detectSOUT(path=a.img_path, image=copy.deepcopy(img))

            # 保存推理结果为 XML 文件
            each_xml_path = os.path.join(save_xml_dir, FileOperationUtil.bang_path(each_img_path)[1] + ".xml")
            res.save_to_xml(each_xml_path)

            # 可选：画出结果图
            if draw:
                save_path = os.path.join(save_res_pic_dir, img_name)
                res.draw_dete_res(save_path=save_path, assign_img=img)

            # 可选：将 XML 转换为 JSON 并返回
            if export_json:
                json_data = parse_xml_to_json(each_xml_path)
                json_file = FileOperationUtil.bang_path(each_img_path)[1] + '.json'
                json_path = os.path.join(json_folder, json_file)
                # 保存为JSON文件
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=4)


            response["results"].append({
                    "image": img_name,
                    "status": "success",
                    "xml_result": each_xml_path
                })

        except Exception as e:
            # 如果检测失败，记录失败信息
            response["results"].append({
                "image": img_name,
                "status": "failed",
                "error": str(e)  # 你可以选择是否返回错误信息给客户端
            })

    return response


if __name__ == "__main__":
    # 启动FastAPI服务
    uvicorn.run(app, host="127.0.0.1", port=6666)
