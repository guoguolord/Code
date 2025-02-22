#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: test.py
# Date: 2024/12/30
# Author: guoguolord
from FileOpt import FileOperation, XmlOperation

path = r"F:\电力信息驻场\负-变压器-表面不清洁_out1"
fileopt = XmlOperation()
filename = fileopt.ReadXml(path)
print(filename)