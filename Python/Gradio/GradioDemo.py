#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: GradioDemo.py
# Date: 2025/1/3
# Author: guoguolord

import gradio as gr
import cv2 as cv

def image_transform(image):
    # image = cv.resize(image, (800,600))
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    return gray

def greet(name):
    return "Hello " + name + "!"
    
input_list = [
    # 音频
    gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio"),
    # 复核框,布尔值,true and false
    gr.Checkbox(label="Checkbox"),
    # 颜色
    gr.ColorPicker(label="Color Picker"),
    # 图表
    gr.Dataframe(label="Dataframe"),
    # 文件
    gr.File(label="File", type="filepath"),
    # 图片
    gr.Image(sources=["upload"], type="pil", label="Image"),
    # 数字
    gr.Number(label="Number"),
    # 下拉选择框
    gr.Dropdown(["option 1", "option 2", "option 3"], label="Dropdown"),
    # 单选框
    gr.Radio(["option 1", "option 2", "option 3"], label="Radio"),
    # 滑块
    gr.Slider(0, 10, label="Slider", step=1),
    # 文本框
    gr.Textbox(label="Textbox"),
    gr.TextArea(label="Text Area"),
    # 视频
    gr.Video(sources=["upload"], label="Video"),
]

output_list = [
    gr.Audio(label="Audio"),
]

def input_output_pairs(*input_data):
    return input_data


demo = gr.Interface(
    fn=input_output_pairs,  # 要调用的函数
    inputs=input_list,  # 输入类型（文本输入）
    outputs=output_list,  # 输出类型（文本输出）
    title="Greeting App",  # 界面标题
    description="Enter your name and get a greeting!",
    live=True
)

if __name__ == "__main__":
    # image_path = "./1.jpg"
    demo.launch(server_port=8888, share=True)
    # demo.launch()