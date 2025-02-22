#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: FileOpt.py
# Date: 2024/12/30
# Author: guoguolord
import shutil
import os
from xml.etree import ElementTree as ET
from collections import defaultdict

class FileOperation():
    def __init__(self, path, new_path):
        self.path = path
        self.new_path = new_path

    def FileCopy(self, path, new_path):
        for file in os.listdir(path):
            ori_file = os.path.join(path, file)
            new_file = os.path.join(new_path, file)
            shutil.copy(ori_file, new_file)

    def FileMove(self, path, new_path):
        for file in os.listdir(path):
            ori_file = os.path.join(path, file)
            new_file = os.path.join(new_path, file)
            shutil.move(ori_file, new_file)

    def FileDelete(self, path):
        for file in os.listdir(path):
            os.remove(os.path.join(self.path, file))

    def FileName(self, path):
        file = os.listdir(path)
        return file

    def CountFileNumber(self, path):
        """
        :param path: 路径地址
        :return: 文件数量，字典类型 {'jpg': 13346}
        """
        file_counts = defaultdict(int)
        for root, dirs, files in os.walk(path):
            for file in files:
                file_ext = os.path.splitext(file)[-1].lower().strip(".")
                file_counts[file_ext] += 1

        return file_counts

class XmlOperation():
    def ReadXml(self,filename):
        results = []
        root = ET.parse(filename)
        obj = root.findall("object")
        width = int(float(root.find("size").find("width").text))
        height = int(float(root.find("size").find("height").text))
        objects = []
        for i in obj:
            name = i.find("name").text
            bndbox = i.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
            objects.append({
                "name": name,
                "bndbox": {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                }
            })
        results.append({
            "filename": filename,
            "width": width,
            "height": height,
            "objects": objects
        })

        return results

    def CountXml(self, filename):
        results = self.ReadXml(filename)
        name_list = {}
        for result in results:
            for obj in result["objects"]:
                name = obj["name"]
                if name in name_list:
                    name_list[name] += 1
                else:
                    name_list[name] = 1

        return name_list


if __name__ == '__main__':
    path = r"F:\电力信息驻场\负-变压器-表面不清洁_out1"
    xml_operation = XmlOperation()
    result_list = {}
    for file in os.listdir(path):
        if file.endswith(".xml"):
            filename = os.path.join(path, file)
            xml_results = xml_operation.CountXml(filename)
            for k, v in xml_results.items():
                if k in result_list:
                    result_list[k] += v
                else:
                    result_list[k] = v
    print(result_list)
            # for result in xml_results:
            #     print(f"Filename: {result['filename']}")
            #     print(f"Width: {result['width']}, Height: {result['height']}")
            #     print("Objects:")
            #     for obj in result["objects"]:
            #         print(f"  name: {obj['name']}")
            #         print(f"  BndBox: {obj['bndbox']}")
            #     print("-" * 40)  # 分隔符


    # path = r"E:\TomatoLeaf"
    # for name, count in result.items():
    #     print(f"{name}: {count}")
    # type, count = fileOpt.CountFileNumber(path)
    # print(f"文件类型: {type if type else '无扩展名'}, 数量: {count}")
    exit()
    xml_results = fileOpt.ReadXml(path)
    for result in xml_results:
        print(f"Filename: {result['filename']}")
        print(f"Width: {result['width']}, Height: {result['height']}")
        print("Objects:")
        for obj in result["objects"]:
            print(f"  name: {obj['name']}")
            print(f"  BndBox: {obj['bndbox']}")
        print("-" * 40)  # 分隔符
