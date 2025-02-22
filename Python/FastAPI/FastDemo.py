#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: FastDemo.py
# Date: 2025/1/3
# Author: guoguolord
import configparser
from fastapi import FastAPI, Form
import argparse
import uvicorn

app = FastAPI()
def read_cfg(conf_file):
    cfg = configparser.ConfigParser()
    cfg.read(conf_file)
    model_name = cfg.get('det', 'modelname')
    classes =[class1 for class1 in cfg.get('det', 'classes').split(',')]
    img_size = cfg.get('det', 'img_size')

    return model_name, classes, img_size


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default='127.0.0.1')

    return parser.parse_args()
@app.post('/cfg')
async def cfg(
        cfg_path: str= Form(...)
):
    model_name, classes, img_size = read_cfg(cfg_path)

    response = {"model name": model_name,
                "classes": classes,
                "img_size": img_size,
                "status": "success"}

    return response

if __name__ == '__main__':
    default_args = args()
    uvicorn.run(app, host=default_args.host, port=default_args.port)