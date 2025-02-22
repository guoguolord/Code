#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: Gr_Blocks.py
# Date: 2025/2/19
# Author: guoguolord
from random import choices

import gradio as gr
import os

"""
gr.Blocks():Blocks模块
gr.Tab():标签
gr.Row():水平布局
gr.Column():垂直布局
gr.Accordion():伸缩组件
"""

with gr.Blocks() as demo:
    with gr.Tab(label="txt2img"):
        with gr.Row():
            with gr.Column(scale=15):
                txt1 = gr.Textbox(lines=2, label="")
                txt2 = gr.Textbox(lines=2, label="")

            with gr.Column(scale=1,min_width=1):
                button1 = gr.Button(value="1")
                button2 = gr.Button(value="2")
                button3 = gr.Button(value="3")
                button4 = gr.Button(value="4")

            with gr.Column(scale=6):
                generate_button = gr.Button(value="Generate", variant="primary", scale=2)
                with gr.Row():
                    dropdown1 = gr.Dropdown(choices=["1", "2", "3", "4"], label="Style1", multiselect=True)
                    dropdown2 = gr.Dropdown(choices=["1", "2", "3", "4"], label="Style2", multiselect=True)

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    dropdown3 = gr.Dropdown(choices=["1", "2", "3"], label="Sampling Method", multiselect=True)
                    slider1 = gr.Slider(minimum=0, maximum=100, step=5, label="Sampling Steps")
                checkboxgroup = gr.CheckboxGroup(["Restore faces", "Tilling", "hires.fix"], label="")
                with gr.Row():
                    slider2 = gr.Slider(minimum=0, maximum=1000, step=128, label="Width")
                    slider3 = gr.Slider(minimum=0, maximum=20, step=1, label="Batch count")
                with gr.Row():
                    slider4 = gr.Slider(minimum=0, maximum=1000, step=128, label="Height")
                    slider5 = gr.Slider(minimum=0, maximum=20, step=1, label="Batch size")
                slider6 = gr.Slider(minimum=0, maximum=50, step=1, label="CFG Scale")
                with gr.Row():
                    number1 = gr.Number(label="Seed", scale=10)
                    button5 = gr.Button(value="Randomize", scale=1, min_width=2)
                    button6 = gr.Button(value="Reset", scale=1, min_width=2)
                    checkbox1 = gr.Checkbox(label="Extra")
                dropdown4 = gr.Dropdown(choices=["1", "2", "3"], label="Script")

            with gr.Column():
                gallery = gr.Gallery([],columns=3)
                with gr.Row():
                    botton7 = gr.Button(value="Save", min_width=1)
                    botton8 = gr.Button(value="Save", min_width=1)
                    botton9 = gr.Button(value="Zip", min_width=1)
                    botton10 = gr.Button(value="Send to img2img", min_width=1)
                    botton11 = gr.Button(value="Send to inpaint", min_width=1)
                    botton12 = gr.Button(value="Send to extras", min_width=1)

                txt3 = gr.Textbox(lines=4, label="")




demo.launch(share=True, debug=True)