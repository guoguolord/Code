#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: Gr_yolo.py
# Date: 2025/2/20
# Author: guoguolord
import os

import gradio as gr
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

# model_seg = YOLO('E:\Code\Python\segmentation\yolo11n-seg.pt')
# model_det = YOLO('E:\Code\Python\detection\yolo11n.pt')


def load_model(model_path):
    det_dict = {}
    for file in os.listdir(model_path):
        filename = os.path.join(model_path, file)
        if filename.endswith(".pt"):
            det_dict[file] = filename
    return det_dict

def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)  # 加载 YOLO 模型
        return model
    except Exception as e:
        print(f"Error loading model at {model_path}: {e}")
        raise

def detect(image, selected_model):
    path = r"E:\Code\Python\detection"
    print(f"Selected model: {selected_model}")  # 打印选中的模型
    print(f"Path: {path}")  # 打印路径
    # 从 det_dict 中获取模型路径
    det_dict = load_model(path)
    model_path = det_dict[selected_model]  # 获取选中的模型路径
    model = load_yolo_model(model_path)  # 加载模型

    # 进行检测（假设 model_det 进行推理）
    results = model([image])  # 这里假设模型是可以直接进行推理的

    for r in results:
        im_array = r.plot()  # 绘制结果
        img = Image.fromarray(im_array[..., ::-1])  # 转换为 PIL 图像
        boxes = r.boxes  # 获取检测框

    return img, boxes

def segment(image, selected_model):
    path = r"E:\Code\Python\segmentation"
    # 从 det_dict 中获取模型路径
    det_dict = load_model(path)
    model_path = det_dict[selected_model]  # 获取选中的模型路径
    model = load_yolo_model(model_path)  # 加载模型
    results = model([image])
    for r in results:
        im_array = r.plot()
        img = Image.fromarray(im_array[..., ::-1])

    return img

det_path = r"E:\Code\Python\detection"
seg_path = r"E:\Code\Python\segmentation"
det_dict = load_model(det_path)
seg_dict = load_model(seg_path)


def gr_yolo():
    with gr.Blocks() as yolo_demo:
        with gr.Tab("模型训练"):
            gr.Markdown("### 模型训练")
            with gr.Row():
                model_style = gr.Dropdown(choices=["模型训练", "模型评测"], label="选择类型", multiselect=False)
                model_choose = gr.Dropdown(choices=["yolov8", "yolo11"], label="选择模型", multiselect=False)
                dataset_file = gr.Dropdown(choices=["voc", "coco", "custom"], label="选择数据集", multiselect=False)


        with gr.Tab("目标检测"):
            gr.Markdown("### 目标检测")
            model_dropdown = gr.Dropdown(choices=list(det_dict.keys()), label="选择模型", multiselect=False)
            with gr.Column():
                with gr.Row():
                    input_img1 = gr.Image(sources=["upload"], label="输入图像", type="pil")
                    output_img1 = gr.Image(label="输出图像", type="pil")
                    txt_output = gr.Textbox(label="检测结果")
                gr.Examples(["../1.jpg"], inputs=[input_img1])
                with gr.Row():
                    input_video = gr.Video(sources=["upload"], label="输入视频")
                    output_video = gr.Video(label="输出视频")
                button1 = gr.Button("目标检测", variant="primary")
                button1.click(fn=detect, inputs=[input_img1, model_dropdown], outputs=[output_img1, txt_output])


        with gr.Tab("语义分割"):
            gr.Markdown("### 语义分割")
            model_dropdown2 = gr.Dropdown(choices=list(seg_dict.keys()), label="选择模型", multiselect=False)
            with gr.Row():
                input_img2 = gr.Image(sources=["upload"], label="输入图像", type="pil")
                output_img2 = gr.Image(label="输出图像", type="pil")
            gr.Examples(["../1.jpg"], inputs=[input_img2])
            button2 = gr.Button("语义分割", variant="primary")
            button2.click(fn=segment, inputs=[input_img2, model_dropdown2], outputs=output_img2)


    return yolo_demo

if __name__ == "__main__":
    yolo_demo = gr_yolo()
    yolo_demo.launch(share=True)