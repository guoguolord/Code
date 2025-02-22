# -*- coding: utf-8  -*-
# -*- author: jokker -*-

import os
import copy
import cv2
import numpy as np
from JoTools.txkjRes.resTools import ResTools
from JoTools.txkjRes.deteRes import DeteRes
from JoTools.txkjRes.deteObj import DeteObj
from JoTools.utils.FileOperationUtil import FileOperationUtil
import math


def find_part(a, b):
    """寻找匹配的检测对象"""

    part_info = []
    for obj in a:
        each_dete = DeteRes()
        each_dete.add_obj_2(obj)
        res = b.filter_by_dete_res_mask(each_dete, cover_index_th=0.01, update=False)
        if len(res) > 0:
            # sort by iou
            assign_obj = None
            max_iou = 0
            for each_obj in res:
                if ResTools.cal_iou(obj, each_obj) > max_iou:
                    assign_obj = each_obj
                    max_iou = ResTools.cal_iou(obj, each_obj)
            part_info.append((obj, assign_obj))
    return part_info

def find_gt_lr_range(a):
    """返回最大的塔的左边和右边的范围，去除下边"""
    jgt = a.filter_by_tags(need_tag=["gt"], update=False)
    gt_max = -1

    for each_gt in jgt:
        if gt_max == -1:
            gt_max = each_gt
        elif each_gt.get_area() > gt_max.get_area():
            gt_max = each_gt

    if gt_max != -1:
        return [(0,0,(gt_max.x1 + gt_max.x2)/2, gt_max.y2), ((gt_max.x1 + gt_max.x2)/2, 0, a.width, gt_max.y2)]
    else:
        return []

def filter_by_range(a, assign_range, assign_tag="jgb"):
    """找到中心点在指定范围内的对象"""
    dete_res_new = DeteRes()
    for each_obj in a:
        if each_obj.tag == assign_tag:
            cx, cy = each_obj.get_center_point()
            if assign_range[0] < cx < assign_range[2] and assign_range[1] < cy < assign_range[3]:
                dete_res_new.add_obj_2(each_obj)
    return dete_res_new

def cal_offset(part_info):
    #补充晃动点坐标
    off_x, off_y = [], []
    loc_sub      = []
    for each in part_info:
        a, b = each
        cx_a, cy_a = a.get_center_point()
        cx_b, cy_b = b.get_center_point()
        off_x.append( cx_a - cx_b )
        off_y.append( cy_a - cy_b )
        loc_sub.append([cx_a, cy_a, cx_b, cy_b])
    return [[off_x, off_y], loc_sub]

def get_th_size(a):
    """根据间隔棒平均尺寸，获取偏移大小
    基于中心点横纵坐标偏移/2, 均值
    """
    area_list = []
    for each in a:
        if each.tag == "jgb":
            area_list.append( (each.x2 - each.x1 + each.y2 - each.y1)/2.0 )

    if len(area_list) == 0:
        return 5
    else:
        area_list = sorted(area_list, reverse=True)
        return area_list[int(len(area_list)/2)] / 2

def jgb_move_info(a, b, th=10):
    try:
        res_info = []
        res      = find_gt_lr_range(a)
        
        #晃动点坐标定位
        if len(res) > 0:
            loc_list = []
            for each in res:
                #左右分块
                ii = filter_by_range(a, each)
                jj = filter_by_range(b, each)
                
                #准备匹配项
                part_info = find_part(ii, jj)
                offset_info = cal_offset(part_info)

                each_src ,loc_list_s = offset_info
                for i,each_off in enumerate(each_src):
                    if len(each_off) == 0:
                        continue
                    off = sum(each_off)/len(each_off)
                    if abs(off) > th:
                        res_info.append(True)
                        for x1_t, y1_t, x2_t, y2_t in loc_list_s:
                            loc_list.append([x1_t, y1_t])
                            loc_list.append([x2_t, y2_t])
                    else:
                        res_info.append(False)
            return res_info, loc_list

        else:
            return [],[]
    except Exception as e:
        print(e.__traceback__.tb_frame.f_globals["__file__"])  # 发生异常所在的文件
        print(e.__traceback__.tb_lineno)  # 发生异常所在的行数
        return [],[]
    
def gt_move_info(a, b, th=0.93):
    """杆塔的移动信息"""

    gt_a = a.filter_by_tags(need_tag=["gt"], update=False)
    gt_b = b.filter_by_tags(need_tag=["gt"], update=False)

    if len(gt_a) ==0 or len(gt_b) == 0:
        # 没找到对应的杆塔
        return False
    else:
        alarms_a = gt_a.alarms
        alarms_b = gt_b.alarms
        alarms_a = sorted(alarms_a, key=lambda x:x.get_area(), reverse=True)
        alarms_b = sorted(alarms_b, key=lambda x:x.get_area(), reverse=True)
        gt_iou = ResTools.cal_iou(alarms_a[0], alarms_b[0])

        if gt_iou > th:
            return True
        else:
            return False

def get_dxhd_info(xml_path_list, hyparam_dct):

    gt_thre             = hyparam_dct.get("jz_thre", 0.93)
    shake_thre          = hyparam_dct.get("shake_thre", 0.5)
    augment_thre        = hyparam_dct.get("augment", 0)

    res_info     = {"pianyi":0 , "weipianyi":0, "weipipei":0}
    loc_list_all = []
    
    if len(xml_path_list)<2:
        return res_info,loc_list_all
    for i in range(0, len(xml_path_list) - 1):
        a = DeteRes(xml_path_list[i])
        b = DeteRes(xml_path_list[i+1])
#FIXME:no need
        # is_gt_move = gt_move_info(a, b, th=float(gt_thre))
        is_gt_move = True
        # do_augment
        a = a.do_augment([augment_thre, augment_thre, augment_thre, augment_thre], is_relative=True, update=True)
        b = b.do_augment([augment_thre, augment_thre, augment_thre, augment_thre], is_relative=True, update=True)

        if is_gt_move:
            th_size             = get_th_size(a) * float(shake_thre)
            move_info, loc_list = jgb_move_info(a, b, th_size)
            loc_list_all.extend(loc_list)
            for each_info in move_info:
                if each_info is True:
                    res_info["pianyi"] += 1
                else:
                    res_info["weipianyi"] += 1
        else:
            res_info["weipipei"] += 1
    
    return res_info, loc_list_all

def get_outer_rect(loc_list):
    x1,y1 = loc_list[0]
    x2,y2 = loc_list[0]
    for i in range(1, len(loc_list)-1):
        x,y = loc_list[i]
        x1 = x1 if x1<x else x
        y1 = y1 if y1<y else y
        x2 = x2 if x2>x else x
        y2 = y2 if y2>y else y
    return [x1,y1,x2,y2]

def get_rect(mapped_result_py):
    res = []
    for key, value in mapped_result_py.items():
        x1 = int(key[0][0])
        y1 = int(key[0][1])
        x2 = int(key[1][0])
        y2 = int(key[1][1])
        conf = value
        res_obj = [x1, y1, x2, y2, conf]
        res.append(res_obj)
    return res

def if_shake_by_4_param(move, not_move, camera_shake, hyparam_dct):
    camera_shake_thre   = hyparam_dct.get("camera_shake_thre", 1)
    pianyi_thre         = hyparam_dct.get("pianyi_thre", 1)
    if camera_shake * camera_shake_thre > (move + not_move):
        return False
    elif (move * pianyi_thre > not_move) and (move > 1):
        return True
#FIXME:部分线晃动，但整体不晃，设立晃动数量阈值
    elif (move * pianyi_thre <= not_move) and (move > 9):
        return True
    else:
        return False

def parse_result(dxhd_info, loc_list, hyparam_dct):
    det_res_dict = dict()
    move, not_move, camera_shake = dxhd_info["pianyi"], dxhd_info["weipianyi"], dxhd_info["weipipei"]
    if if_shake_by_4_param(move, not_move, camera_shake, hyparam_dct):
        det_res_dict['if_shake'] = True
        det_res_dict['shake_info'] = get_outer_rect(loc_list)
        det_res_dict['logic_info'] = dxhd_info
    else:
        det_res_dict['if_shake'] = False
        det_res_dict['shake_info'] = {}
        det_res_dict['logic_info'] = dxhd_info

    return det_res_dict

def parse_result_2(dxhd_info, loc_list, hyparam_dct, mapped_result_py):
    det_res_dict = dict()
    move, not_move, camera_shake = dxhd_info["pianyi"], dxhd_info["weipianyi"], dxhd_info["weipipei"]
    if any(value > 0.5 for value in mapped_result_py.values()):
        det_res_dict['if_shake'] = True
        det_res_dict['shake_info'] = get_rect(mapped_result_py)
        det_res_dict['logic_info'] = dxhd_info

    else:
        det_res_dict['if_shake'] = False
        det_res_dict['shake_info'] = {}
        det_res_dict['logic_info'] = dxhd_info

    return det_res_dict

def get_str_length(assign_str):
    try:
        res = float(assign_str)
        return res
    except:
        pass
    sum_num = 0
    for each in assign_str:
        sum_num += ord(each)
    return sum_num

def rm_uselessImgs(src_list, filter_list):
    for a in src_list:
        if a not in filter_list:
            os.remove(a)

def correct_and_filter_imgs(img_path_list, model_point):
    '''return 
    useful pic
    '''
    img_path_list_n = []
    
    #1、获取图片及对应的pointLst,不足2直接结束
    imgpath2poinLst = []
    for imgpath in img_path_list:
        sg_point_lst = model_point.detect_point_lst(imgpath)
        if len(sg_point_lst):
            imgpath2poinLst.append([imgpath, sg_point_lst])
    if len(imgpath2poinLst)<2:
        return img_path_list_n
    # print(imgpath2poinLst)
    #2、更新更新可用图片及对应的pointLst;纠片
    base_imgpath, base_point = imgpath2poinLst[0]
    base_point = np.array(base_point, dtype=np.float32)
    base_img   = cv2.imread(base_imgpath)
    
    img_path_list_n.append(base_imgpath)
    #2.1、以第一张为基准，纠正剩余图片
    for other_imgpath, other_point in imgpath2poinLst[1:]:
        other_img = cv2.imread(other_imgpath)
#FIXME:visual_point
        for point_vis in other_point:
            cv2.circle(other_img, point_vis, 5, (0, 0, 255), -1)
        pts_img2  = np.array(other_point, dtype=np.float32)
        
        M = cv2.estimateAffine2D(pts_img2, base_point)[0]
        other_img_correct = cv2.warpAffine(other_img, M, (base_img.shape[1], base_img.shape[0]))
        # print(other_imgpath.split('.')[0]+'_bak.jpg')
        # cv2.imwrite(os.path.join(os.path.dirname(other_imgpath), os.path.basename(other_imgpath).split('.')[0]+'_bak.jpg'), other_img)
        cv2.imwrite(other_imgpath, other_img_correct)
        img_path_list_n.append(other_imgpath)
    rm_uselessImgs(img_path_list, img_path_list_n)
    return img_path_list_n

def line_intersection(p1, p2, p3, p4):
    """
    检查线段 p1p2 是否与 p3p4 相交，并返回交点。
    p1, p2, p3, p4 是表示点 (x, y) 的元组。
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # 计算行列式
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return None  # 线段平行或共线

    # 计算交点
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    # 检查交点是否在线段内
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return (x, y)
    return None


def get_rectangle_edges(rect):
    """
    给定一个由 (x_min, y_min, x_max, y_max) 定义的矩形，
    返回其边界线段。
    """
    x_min, y_min, x_max, y_max = rect
    return [((x_min, y_min), (x_min, y_max)),  # 左边
            ((x_min, y_max), (x_max, y_max)),  # 上边
            ((x_max, y_max), (x_max, y_min)),  # 右边
            ((x_max, y_min), (x_min, y_min))]  # 下边


def find_intersections(rect1, rect2):
    edges1 = get_rectangle_edges(rect1)
    edges2 = get_rectangle_edges(rect2)

    intersections = []
    for edge1 in edges1:
        for edge2 in edges2:
            point = line_intersection(*edge1, *edge2)
            if point:
                intersections.append(point)
    return intersections

def matrix_center(matrix):
    xmin, ymin, xmax, ymax = matrix
    return ((xmin + xmax) / 2, (ymin + ymax) / 2)

# 定义计算点到线段距离的函数
def point_to_segment_distance(point, segment):
    p = np.array(point)
    a = np.array(segment[0])
    b = np.array(segment[1])
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    if t < 0.0:
        closest_point = a
    elif t > 1.0:
        closest_point = b
    else:
        closest_point = a + t * ab
    return np.linalg.norm(p - closest_point)


def filter_and_match_dicts(dict_list):
    result = {}
    if not dict_list:
        return result

    base_dict = dict_list[0]
    base_keys = list(base_dict.keys())

    filtered_dicts = [d for d in dict_list[1:] if len(d) == len(base_dict)]
    filtered_dicts_update = []

    for i in enumerate(dict_list):
        for idx, other_dict in enumerate(filtered_dicts, start=1):
            if i[1] == other_dict:
                filtered_dicts_update.append(i)

    for idx, other_dict in enumerate(filtered_dicts, start=1):
        for idx_2, other_dict_2 in enumerate(filtered_dicts_update):
            if other_dict_2[1] == other_dict:
                other_pos = other_dict_2[0]
            for key_idx, key in enumerate(base_keys):
                base_value = base_dict[key]
                for j in enumerate(other_dict):
                    if j[0] == key_idx:
                        other_value = other_dict[j[1]]

                if isinstance(base_value, list) and isinstance(other_value, list):
                    if len(base_value) == len(other_value):
                        result[(0, key_idx)] = base_value
                        result[(other_pos, key_idx)] = other_value
    return result


def get_bounding_box_and_avg_size(rectangles):
    """
    给定一组矩形，返回它们的最小外接矩形以及平均的宽和高。
    每个矩形以 (x1, y1, x2, y2) 形式表示。
    """

    min_x = min(rect[0] for rect in rectangles)
    min_y = min(rect[1] for rect in rectangles)
    max_x = max(rect[2] for rect in rectangles)
    max_y = max(rect[3] for rect in rectangles)

    total_width = sum(rect[2] - rect[0] for rect in rectangles)
    total_height = sum(rect[3] - rect[1] for rect in rectangles)
    avg_width = round(total_width / len(rectangles), 2)
    avg_height = round(total_height / len(rectangles), 2)

    return (min_x, min_y, max_x, max_y), (avg_width, avg_height)


def combine_rectangles_by_second_key(result):
    from collections import defaultdict

    grouped = defaultdict(list)
    for (x, y), value_list in result.items():
        grouped[y].append(value_list)

    combined_results = {}
    for y, value_lists in grouped.items():
        transposed = list(zip(*value_lists))
        combined_bboxes_and_avgs = [get_bounding_box_and_avg_size(group) for group in transposed]
        combined_results[(0, y)] = combined_bboxes_and_avgs

    return combined_results

def calculate_ratios(result):
    ratio_results = {}

    for key, value_list in result.items():
        ratio_list = []
        for bounding_box, avg_size in value_list:
            bbox_width = bounding_box[2] - bounding_box[0]
            bbox_height = bounding_box[3] - bounding_box[1]

            width_ratio = round( bbox_width / avg_size[0], 2)
            height_ratio = round( bbox_height / avg_size[1], 2)

            ratio_list.append((width_ratio, height_ratio))

        ratio_results[key] = ratio_list

    return ratio_results

def filter_values(result):
    filtered_by_greater_than_0_5 = {}
    filtered_by_greater_than_0_5_1 = {}
    filtered_by_greater_than_1 = {}

    for key, value_list in result.items():
        greater_than_0_5_list = []
        greater_than_0_5_1_list = []
        greater_than_1_list = []

        for value in value_list:
            if 0 < value[1] <= 0.5:
                greater_than_0_5_list.append(value)
            if 0.5 < value[1] <= 1:
                greater_than_0_5_1_list.append(value)
            if value[1] > 1:
                greater_than_1_list.append(value)

        filtered_by_greater_than_0_5[key] = greater_than_0_5_list
        filtered_by_greater_than_0_5_1[key] = greater_than_0_5_1_list
        filtered_by_greater_than_1[key] = greater_than_1_list

    return filtered_by_greater_than_0_5, filtered_by_greater_than_0_5_1, filtered_by_greater_than_1

# 检查两个矩阵是否有交集
def is_intersecting(r1, r2):
    r1_x_min, r1_y_min, r1_x_max, r1_y_max = r1
    r2_x_min, r2_y_min, r2_x_max, r2_y_max = r2

    # 检查 x 轴上的重叠
    x_overlap = not (r1_x_max < r2_x_min or r1_x_min > r2_x_max)
    # 检查 y 轴上的重叠
    y_overlap = not (r1_y_max < r2_y_min or r1_y_min > r2_y_max)

    # 如果在 x 轴和 y 轴上都有重叠，则两个矩形有交集
    return x_overlap and y_overlap

def calculate_area(box):
    # Unpack the coordinates
    x1, y1, x2, y2 = box
    # Calculate the width and height of the bounding box
    width = x2 - x1
    height = y2 - y1
    # Return the area
    return width * height

# 计算与基准字典中相同键值对的距离
def compare_distances(base, others):
    result = []
    for other in others:
        sublist_result = {}
        for key in base:
            if key in other:
                base_x, base_y = base[key]
                other_x, other_y = other[key]
                distance_x = other_x - base_x
                distance_y = other_y - base_y
                sublist_result[key] = (distance_x, distance_y)
        result.append(sublist_result)
    return result

# 计算距离的函数
def calculate_distances(lists):
    result = []
    for sublist in lists:
        sublist_result = {}
        for i in range(len(sublist)):
            for j in range(i + 1, len(sublist)):
                x1, y1, x2, y2 = sublist[i]
                x1_next, y1_next, x2_next, y2_next = sublist[j]
                distance_x = abs(x1_next - x1)
                distance_y = abs(y1_next - y1)
                sublist_result[(i, j)] = (distance_x, distance_y)
        result.append(sublist_result)
    return result

# 计算最大和最小距离的函数
def calculate_max_min_distances(comparison_result):
    keys = comparison_result[0].keys()
    result = {}

    for key in keys:
        x_distances = [subdict[key][0] for subdict in comparison_result]
        y_distances = [subdict[key][1] for subdict in comparison_result]

        max_distance_x = max(x_distances) - min(x_distances)
        max_distance_y = max(y_distances) - min(y_distances)
        if max_distance_x == 0:
            max_distance_x = max(x_distances)
        if max_distance_y == 0:
            max_distance_y = max(y_distances)

        result[key] = (max_distance_x, max_distance_y)

    return result

def calculate_xddx_ratios(max_min_distances, base_dict):
    ratios = {}

    for key in max_min_distances.keys():
        if key in base_dict:
            x1, y1 = max_min_distances[key]
            x2, y2 = base_dict[key]
            # 计算 x 和 y 的比值，避免除以0的情况
            x_ratio = round(x1 / x2 if x2 != 0 else 0, 2)
            y_ratio = round(y1 / y2 if y2 != 0 else 0, 2)
            ratios[key] = (x_ratio, y_ratio)

    return ratios

def extract_values(ratios):
    all_values = set()
    # Extract all x and y values
    for key, (x, y) in ratios.items():
        all_values.add(key[0])
        all_values.add(key[1])

    # Sort values
    sorted_values = sorted(all_values)
    #print("sorted_values:", sorted_values)

    # Create a new dictionary
    new_dict = {}
    # Iterate through sorted values
    for value in sorted_values:
        value_pairs = []
        for key, (x, y) in ratios.items():
            if key[0] == value or key[1] == value:
                value_pairs.append((x, y))
        new_dict[value] = value_pairs

    return new_dict

def filter_values_above_threshold(input_dict, threshold):
    filtered_dict = {}
    for key, value_pairs in input_dict.items():
        filtered_pairs = [(x, y) for (x, y) in value_pairs if x > threshold]
        filtered_dict[key] = filtered_pairs

    return filtered_dict

def calculate_metrics(filtered_jgb_area_dict):
    min_x1 = min(rect[0] for rect in filtered_jgb_area_dict.values())
    min_y1 = min(rect[1] for rect in filtered_jgb_area_dict.values())
    max_x2 = max(rect[2] for rect in filtered_jgb_area_dict.values())
    max_y2 = max(rect[3] for rect in filtered_jgb_area_dict.values())

    total_width = sum(rect[2] - rect[0] for rect in filtered_jgb_area_dict.values())
    total_height = sum(rect[3] - rect[1] for rect in filtered_jgb_area_dict.values())
    num_rectangles = len(filtered_jgb_area_dict)

    enclosing_width = max_x2 - min_x1
    enclosing_height = max_y2 - min_y1

    avg_width = total_width / num_rectangles
    avg_height = total_height / num_rectangles

    width_ratio = round(enclosing_width / avg_width, 2)
    height_ratio = round(enclosing_height / avg_height, 2)

    return {
        "enclosing_rectangle": (min_x1, min_y1, max_x2, max_y2),
        "average_width": avg_width,
        "average_height": avg_height,
        "width_ratio": width_ratio,
        "height_ratio": height_ratio
    }

def line_intersection_with_rect(point, center, box):
    x1, y1, x2, y2 = box
    px, py = point
    cx, cy = center

    intersections = []

    # Check intersection with left side (x1, y1) to (x1, y2)
    if cx != px:  # Avoid division by zero
        t1 = (x1 - px) / (cx - px)
        if 0 <= t1 <= 1:
            iy1 = py + t1 * (cy - py)
            if y1 <= iy1 <= y2:
                intersections.append((x1, iy1))

    # Check intersection with right side (x2, y1) to (x2, y2)
    if cx != px:
        t2 = (x2 - px) / (cx - px)
        if 0 <= t2 <= 1:
            iy2 = py + t2 * (cy - py)
            if y1 <= iy2 <= y2:
                intersections.append((x2, iy2))

    # Check intersection with bottom side (x1, y1) to (x2, y1)
    if cy != py:  # Avoid division by zero
        t3 = (y1 - py) / (cy - py)
        if 0 <= t3 <= 1:
            ix1 = px + t3 * (cx - px)
            if x1 <= ix1 <= x2:
                intersections.append((ix1, y1))

    # Check intersection with top side (x1, y2) to (x2, y2)
    if cy != py:
        t4 = (y2 - py) / (cy - py)
        if 0 <= t4 <= 1:
            ix2 = px + t4 * (cx - px)
            if x1 <= ix2 <= x2:
                intersections.append((ix2, y2))

    # Assuming there will always be one valid intersection point
    return intersections[0] if intersections else None

def pad_image_with_ratio(img, length_ratio=1.25):
    # 获取原始图像的长和宽
    original_length, original_width, channels = img.shape
    #print("111111111:",original_length, original_width, channels)

    # 计算填充后的图像长度和宽度
    target_length = int(original_length * length_ratio)
    new_width = int(target_length * original_width / original_length)
    #print("2222:", target_length,new_width)

    # 创建一个新的黑色图像矩阵
    new_img = np.zeros((target_length, new_width, channels), dtype=img.dtype)

    # 计算将原始图像粘贴到新图像中心的位置
    start_row = (target_length - original_length) // 2
    start_col = (new_width - original_width) // 2

    # 将原始图像粘贴到新图像中心位置
    new_img[start_row:start_row + original_length, start_col:start_col + original_width, :] = img

    return new_img, start_row, start_col

def map_coords_back(x1, y1, x2, y2, start_row, start_col):
    # 将填充后的图片中的矩形框的坐标映射回原图
    x1_orig = x1 - start_col
    y1_orig = y1 - start_row
    x2_orig = x2 - start_col
    y2_orig = y2 - start_row

    return x1_orig, y1_orig, x2_orig, y2_orig



def infer_main(img_cache_dir, save_xml_dir, save_xml_final_dir, hyparam_dct, model_anjian, model_point, model_dx, model_xddx, data_format):

    try:
        img_path_list = list(FileOperationUtil.re_all_file(img_cache_dir, endswitch=['.jpg', '.png', '.JPG'])) 
#FIXME:插入函数，返回新的img_path_list，img,并更新图片
        #print("56789:", len(img_path_list))
        if data_format == "video":
            img_path_list = img_path_list
        else:
            #img_path_list = correct_and_filter_imgs(img_path_list, model_point)
            img_path_list = img_path_list
        jgb_match_dx_all = [] #保存每张图片中导线坐标点以及对应jgb的坐标框
        result_py = {}   #保存每根导线上jgb最大偏移值
        mapped_result_py = {}  #保存每根导线坐标点和对应jgb的最大偏移值
        xddx_sort_all = [] #保存每张图片中xddx的位置
        jgb_sort_all = [] #保存每张图片中jgb的位置
        gt_max_area_all = [] #保存图片中没有jgb时候最大面积的gt
        dete_res_all = DeteRes()

        #阈值设置
        threshold_xddx_py = 0.5  #小段导线两两框之间的偏移值比例
        jgb_area_conf = 1600    #没有gt只有jgb的时候，判断图片中最大面积的jgb是否达到要求

        #
        # print(len(img_path_list))
        for each_img_path in img_path_list:
            dx_list = []
            results = {}
            jbg_match_dx = {}
            a = DeteRes()
            res_xddx_new = DeteRes()
            gt_rect_only = None  # 初始化 gt_rect_only
            a.img_path = each_img_path
            img = a.get_img_array(RGB=True)
            res_gt_jgb = model_anjian.detectSOUT(path=a.img_path, image=copy.deepcopy(img))

            res_dx = model_dx.detectSOUT(path=a.img_path, image=copy.deepcopy(img))

            # 填充黑条，将长变成原长的1.25倍，保持变换后的长宽比不变
            new_img, start_row, start_col = pad_image_with_ratio(img, length_ratio=1.25)
            res_xddx = model_xddx.detectSOUT(path=a.img_path, image=copy.deepcopy(new_img))
            
            for each_dete_obj_xddx in res_xddx:
                xmin, ymin, xmax, ymax = each_dete_obj_xddx.x1, each_dete_obj_xddx.y1, each_dete_obj_xddx.x2, each_dete_obj_xddx.y2
                #print("3333333333:", xmin, ymin, xmax, ymax)
                each_dete_obj_xddx.x1, each_dete_obj_xddx.y1, each_dete_obj_xddx.x2, each_dete_obj_xddx.y2 = map_coords_back(int(xmin), int(ymin), int(xmax), int(ymax), start_row, start_col)
                #print("4444444:",each_dete_obj_xddx.x1, each_dete_obj_xddx.y1, each_dete_obj_xddx.x2, each_dete_obj_xddx.y2)
                #res_xddx_new += each_dete_obj_xddx
            res_xddx.do_nms(0.5)

            res = res_gt_jgb + res_dx + res_xddx

            each_xml_path = os.path.join(save_xml_dir, FileOperationUtil.bang_path(each_img_path)[1] + ".xml")
            #print("each_xml_path:", each_xml_path)
            res.save_to_xml(each_xml_path)

            dx = res.filter_by_tags(need_tag=['dx'],update=False)
            gt = res.filter_by_tags(need_tag=['gt'],update=False)
            jgb = res.filter_by_tags(need_tag=['jgb'],update=False)
            xddx = res.filter_by_tags(need_tag=['xd_dx'],update=False)
            print("检测到的导线的个数和杆塔的个数,jgb和xddx分别为：", len(dx), len(gt), len(jgb), len(xddx))

            if len(dx) > 0 and len(gt) > 0 and len(jgb) > 1:
                # 如果图片中有多个塔，导线需要与杆塔相交，此外选取距离导线最小框中心点最近的
                # 杆塔坐标 (gt_obj.x1,gt_obj.y1,gt_obj.x2,gt_obj.y2)
                #print("检测到的导线的个数和杆塔的个数分别为：", len(dx), len(gt))
                # # 求取面积最大的杆塔
                # max_area = 0
                # max_area_gt_rect = DeteObj()
                # for each_dete_obj_gt in gt:
                #     area = ((each_dete_obj_gt.x2 - each_dete_obj_gt.x1)*(each_dete_obj_gt.y2 - each_dete_obj_gt.y1))
                #     if area > max_area:
                #         max_area = area
                #         max_area_gt_rect = each_dete_obj_gt.x1, each_dete_obj_gt.y1, each_dete_obj_gt.x2, each_dete_obj_gt.y2

                if len(gt) > 1:
                    dx_rect_dict = {}
                    # 存储与所有 dx 相交的 gt 元素
                    filtered_gt = DeteRes()
                    # 初始化最小距离和对应的中心点
                    min_distance = float('inf')
                    closest_center_point = None

                    # 遍历每个 gt 元素
                    for each_dete_obj_gt in gt:
                        gt_rect = (each_dete_obj_gt.x1, each_dete_obj_gt.y1, each_dete_obj_gt.x2, each_dete_obj_gt.y2)
                        all_intersect = True
                        # 检查是否与所有 dx 元素相交
                        for each_dete_obj_dx in dx:
                            dx_rect = (each_dete_obj_dx.x1, each_dete_obj_dx.y1, each_dete_obj_dx.x2, each_dete_obj_dx.y2)
                            if not is_intersecting(gt_rect, dx_rect):
                                all_intersect = False
                                break
                        # 如果与所有 dx 元素都相交，则保留该 gt 元素
                        if all_intersect:
                            filtered_gt.add_obj_2(each_dete_obj_gt)
                            #print("与导线都相交的杆塔：",filtered_gt)
                            for each_dete_obj in dx:
                                x1, y1, x2, y2 = each_dete_obj.x1, each_dete_obj.y1, each_dete_obj.x2, each_dete_obj.y2
                                dx_rect_area = (x2 - x1) * (y2 - y1)
                                center_point = ((x2 - x1) / 2, (y2 - y1) / 2)
                                dx_rect_dict[dx_rect_area] = center_point
                            dx_min_area = min(dx_rect_dict.keys())
                            dx_center_point = dx_rect_dict[dx_min_area]

                            for each_dete_obj_gt in filtered_gt:
                                center_point_x, center_point_y = ((each_dete_obj_gt.x2 - each_dete_obj_gt.x1) / 2,
                                                                  (each_dete_obj_gt.y2 - each_dete_obj_gt.y1) / 2)
                                # 计算中心点到dx_center_point的距离
                                distance = math.sqrt((center_point_x - dx_center_point[0]) ** 2 + (
                                            center_point_y - dx_center_point[1]) ** 2)

                                # 如果找到更近的中心点，更新最小距离和对应的中心点
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_center_point = (center_point_x, center_point_y)
                                    gt_rect_only = each_dete_obj_gt.x1, each_dete_obj_gt.y1, each_dete_obj_gt.x2, each_dete_obj_gt.y2
                        else:
                            print("没有与导线都相交的杆塔")
                            # #选取面积最大的塔
                            # gt_rect_only = max_area_gt_rect

                else:
                    for each_dete_obj_gt in gt:
                        gt_rect_only = each_dete_obj_gt.x1, each_dete_obj_gt.y1, each_dete_obj_gt.x2, each_dete_obj_gt.y2

                for each_dete_obj in dx:
                    dx_rect = each_dete_obj.x1, each_dete_obj.y1, each_dete_obj.x2, each_dete_obj.y2
                    if gt_rect_only:
                        intersection = find_intersections(dx_rect, gt_rect_only)
                    else:
                        intersection = []
                    if intersection:
                        #print("交点矩形的坐标为：", intersection)
                        min_point = min(intersection, key=lambda point: (point[0], point[1]))
                        max_point = max(intersection, key=lambda point: (point[0], -point[1]))
                        #print(f"第一个点 (x 最小值，如果 x 相等就 y 最小值): {min_point}")
                        #print(f"第二个点 (x 最大值，如果 x 相等就 y 最小值): {max_point}")

                        line_left = [(each_dete_obj.x1, each_dete_obj.y1), min_point]
                        #print("左导线的端点坐标为：", line_left)
                        line_right = [(each_dete_obj.x2, each_dete_obj.y1), max_point]
                        #print("右导线的端点坐标为：", line_right)

                        dx_list.append(line_left)
                        dx_list.append(line_right)
                    else:
                        print("两个矩形没有交点")

                # dx_list = [[(each_dete_obj.x1, each_dete_obj.y1), (intersect_x1, intersect_y1)], [(each_dete_obj.x2, each_dete_obj.y2), (intersect_x2, intersect_y2)], [(each_dete_obj.x3, each_dete_obj.y3), (intersect_x3, intersect_y3)]]
                dx_list_sorted = sorted(dx_list, key=lambda item: (item[1][1], item[1][0]))
                #print("排序后的dx_list：", dx_list_sorted)

                #jgb匹配导线
                for each_dete_obj in jgb:
                    jgb_box = (each_dete_obj.x1, each_dete_obj.y1, each_dete_obj.x2, each_dete_obj.y2)
                    jgb_center_point = matrix_center(jgb_box)
                    #print("jgb_center_point:", jgb_center_point)
                    if dx_list_sorted:
                        distances = [point_to_segment_distance(jgb_center_point, segment) for segment in dx_list_sorted]
                        min_distance = min(distances)
                        min_distance_index = distances.index(min_distance)
                        closest_segment = dx_list[min_distance_index]
                        results[tuple(jgb_box)] = {'distance': min_distance, 'segment': closest_segment}

                for segment in dx_list_sorted:
                    # 找到所有匹配的jgb
                    matching_keys = [key for key, value in results.items() if value['segment'] == segment]
                    # # 如果有匹配的jgb，就将它们作为值，否则将值置为None
                    # jbg_match_dx[tuple(segment)] = matching_keys if matching_keys else None

                    if matching_keys:
                        centers = [(np.mean([key[0], key[2]]), np.mean([key[1], key[3]])) for key in matching_keys]
                        # 将框和中心点绑定在一起
                        keys_with_centers = list(zip(matching_keys, centers))
                        # 按照中心点排序，先按照x从小到大，再按照y从小到大排序
                        keys_with_centers.sort(key=lambda item: (item[1][0], item[1][1]))
                        sorted_jgbs = [item[0] for item in keys_with_centers]
                        jbg_match_dx[tuple(segment)] = sorted_jgbs
                    else:
                        jbg_match_dx[tuple(segment)] = None
                # jbg_match_dx_all=[{((1, 1), (2, 1)): [(0, 0, 1, 1), (0, 0, 0.5, 0.5)], ((1, 2), (2, 2)): [(1, 1, 2, 2), (2, 2, 3, 3)], ((1, 3), (2, 3)): None}, {((1, 1), (5, 1)): [(0, 0, 2, 2), (3, 3, 5, 5)], ((1, 2), (3, 2)): [(6, 6, 9, 9), (4, 4, 8, 8)], ((1, 8), (2, 4)): [(2, 2, 9, 9), (5, 5, 6, 6)]}]
                jgb_match_dx_all.append(jbg_match_dx)
            #图中只有jgb没有gt的时候，找到最大的jgb，如果最大的jgb大于一定的尺寸，求取偏移量
            elif len(jgb) > 0 and len(gt) == 0:
                print("没有杆塔只有jgb")
                jgb_list = []
                for each_dete_obj in jgb:
                    jgb_box = (each_dete_obj.x1, each_dete_obj.y1, each_dete_obj.x2, each_dete_obj.y2)
                    # xddx_center_point = matrix_center(xddx_box)
                    jgb_list.append(jgb_box)
                # 计算中心点并排序
                sorted_jgb = sorted(jgb_list, key=lambda rect: ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2))
                jgb_sort_all.append(sorted_jgb)

            #图中没有jgb，但有gt和dx，求取图片上边缘的小段导线，计算偏移量
            elif len(jgb) == 0 or len(jgb) == 1 and len(gt) > 0 and len(dx) > 0 and len(xddx) > 0:
                xddx_list = []
                #对小段导线按照先x轴再y轴排序保存起来
                for each_dete_obj in xddx:
                    xddx_box = (each_dete_obj.x1, each_dete_obj.y1, each_dete_obj.x2, each_dete_obj.y2)
                    #xddx_center_point = matrix_center(xddx_box)
                    xddx_list.append(xddx_box)
                # 计算中心点并排序
                sorted_xddx = sorted(xddx_list, key=lambda rect: ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2))
                xddx_sort_all.append(sorted_xddx)

                gt_max_area = 0
                gt_largest_box = None
                for each_dete_obj in gt:
                    gt_box = (each_dete_obj.x1, each_dete_obj.y1, each_dete_obj.x2, each_dete_obj.y2)
                    gt_area = calculate_area(gt_box)
                    if gt_area > gt_max_area:
                        gt_max_area = gt_area
                        gt_largest_box = gt_box
                gt_max_area_all.append(gt_largest_box)
            else:
                print("不在考虑范围内")

        if jgb_match_dx_all:
            # 首先如果线段的个数与基准线段个数不一致，就放弃这张图片;其次对比每条线段上面间隔棒的个数，如果间隔棒的个数不对，就放弃这张图片中这条导线上间隔棒的计算
            filtered_result = filter_and_match_dicts(jgb_match_dx_all)
            #print(filtered_result)
            # 找出对应线段中每个jgb的最外接矩形，以及jgb宽和高的平均值
            combined_result = combine_rectangles_by_second_key(filtered_result)
            #print(combined_result)
            # 计算每个线段中每个jgb的偏移值
            ratios = calculate_ratios(combined_result)
            #print(ratios)

            # filtered_by_greater_than_0_5, filtered_by_greater_than_0_5_1, filtered_by_greater_than_1 = filter_values(ratios)
            # print("Filtered by greater than 0.5:")
            # print(filtered_by_greater_than_0_5)
            # print("Filtered by greater than 0.5 and 1:")
            # print(filtered_by_greater_than_0_5_1)
            # print("\nFiltered by greater than 1:")
            # print(filtered_by_greater_than_1)

            #获得每条导线中jgb偏移量最大的值
            for key, tuple_list in ratios.items():
                max_value = max(tup[1] for tup in tuple_list)
                result_py[key] = max_value

            #将导线的坐标和对应的jgb进行匹配
            for key, value in result_py.items():
                # 根据键的第一个元素找到对应位置的 jgb_match_dx_all 中的字典
                idx = key[0]
                corresponding_dict = jgb_match_dx_all[idx]

                # 根据键的第二个元素找到该字典中的对应键，并构建新的键值对
                corresponding_key = list(corresponding_dict.keys())[key[1]]
                mapped_result_py[corresponding_key] = value
            if data_format == "video":
                mapped_result_py = {key: round(value / 20, 2) for key, value in mapped_result_py.items() if value >= 10}
                print("video detect result one:", mapped_result_py)
            else:
                mapped_result_py = {key: round(value / 20, 2) for key, value in mapped_result_py.items() if value >= 10}
                print("images detect result one:", mapped_result_py)

            # #print("mapped_result_py:", mapped_result_py)
            # for key, value in mapped_result_py.items():
            #     dete_obj = DeteObj(x1=int(key[0][0]), y1=int(key[0][1]), x2=int(key[1][0]), y2=int(key[1][1]), tag='dx_wd_1', conf=value)
            #     if 5 <= dete_obj.conf < 10:
            #         dete_obj.tag = "qd_dxwd_1"
            #     elif 10 < dete_obj.conf:
            #         dete_obj.tag = "zb_dxwd_1"
            #     else:
            #         dete_obj.tag = "bwd_1"
            #     dete_res_all.add_obj_2(dete_obj)
            # each_xml_final_path = os.path.join(save_xml_final_dir, FileOperationUtil.bang_path(img_path_list[0])[1] + ".xml")
            # #print("each_xml_path:", each_xml_final_path)
            # dete_res_all.save_to_xml(each_xml_final_path)

        elif xddx_sort_all:
            #print("xddx_sort_all:", xddx_sort_all)
            first_list_length = len(xddx_sort_all[0])
            filtered_xddx_sort_all = [sublist for sublist in xddx_sort_all if len(sublist) == first_list_length]
            #print("filtered_xddx_sort_all:", filtered_xddx_sort_all)
            #求取子列表中对应位置的元素的最外接矩形
            # 获取每个位置的所有矩形
            num_positions = len(filtered_xddx_sort_all[0])
            num_sublists = len(filtered_xddx_sort_all)

            # 初始化最外接矩形列表
            bounding_rectangles = []
            for i in range(num_positions):
                # 初始化最外接矩形的坐标
                min_x1, min_y1 = float('inf'), float('inf')
                max_x2, max_y2 = float('-inf'), float('-inf')
                for sublist in filtered_xddx_sort_all:
                    rect = sublist[i]
                    x1, y1, x2, y2 = rect
                    # 更新最小和最大坐标
                    min_x1 = min(min_x1, x1)
                    min_y1 = min(min_y1, y1)
                    max_x2 = max(max_x2, x2)
                    max_y2 = max(max_y2, y2)
                # 添加最外接矩形到结果列表
                bounding_rectangles.append((min_x1, min_y1, max_x2, max_y2))
            #print("bounding_rectangles最外接矩形：", bounding_rectangles)
            #求取第一张图片的每个xddx与最外接矩形在横向和纵向上的比值
            # 计算比值并保存结果
            xddx_results_ratio = []
            first_sublist = filtered_xddx_sort_all[0]

            for i, rect in enumerate(first_sublist):
                x1, y1, x2, y2 = rect
                bx1, by1, bx2, by2 = bounding_rectangles[i]

                # 计算矩形的宽度和高度
                rect_width = x2 - x1
                rect_height = y2 - y1

                # 计算bounding_rectangles的宽度和高度
                brect_width = bx2 - bx1
                brect_height = by2 - by1

                # 计算横向和纵向比值
                width_ratio =  brect_width / rect_width
                height_ratio =  brect_height / rect_height

                # 保存框的坐标和对应的比值
                xddx_results_ratio.append({
                    "rect": rect,
                    "width_ratio": width_ratio,
                    "height_ratio": height_ratio
                })

            # 输出结果
            #print("xddx_results_ratio:", xddx_results_ratio)
            #gt_max_area_all=[(1,2,3,4),(5,6,7,8)],将xddx_results_ratio中rect的前两位作为一个点，gt_max_area_all中第一个矩阵的中心点作为第二个点，这两个点作为键，将jgb_results_ratio中Height Ratio作为值
            #最后得到mapped_result： {((1, 2), ((3-1)/2, (4-2)/2)): 0.2, ((5, 6), ((3-1)/2, (4-2)/2)): 0.2, ((9, 10), ((3-1)/2, (4-2)/2)): 0.2,}
            gt_rect = gt_max_area_all[0]
            gt_center = ((gt_rect[2] + gt_rect[0]) / 2, (gt_rect[3] + gt_rect[1]) / 2)

            for item in xddx_results_ratio:
                rect = item['rect']
                first_point = (rect[0], rect[1])
                # 计算xddx左上角的坐标与gt中心点的交点坐标
                intersection_point = line_intersection_with_rect(first_point, gt_center, gt_rect)
                key = (first_point, intersection_point)
                width_ratio = item['width_ratio']
                mapped_result_py[key] = width_ratio
            print("mapped_result_py:", mapped_result_py)

            if data_format == "video":
                mapped_result_py = {key: round(value / 20, 2) for key, value in mapped_result_py.items() if value >= 10}
                print("video detect result two:", mapped_result_py)

            else:
                mapped_result_py = {key: round(value / 20, 2) for key, value in mapped_result_py.items() if value >= 10}
                print("images detect result two:", mapped_result_py)

            #保底逻辑，当某一根导线与其他导线之间的距离变化都超过一定的阈值，认为这跟导线产生了晃动
            # 计算每张图片中导线两两之间的距离
            distances = calculate_distances(filtered_xddx_sort_all)
            #print("filtered_xddx_sort_all:", filtered_xddx_sort_all)
            #print("每张图片中导线两两之间的距离:", distances)
            # 获取基准字典
            base_dict = distances[0]
            if len(distances) > 1:
                # 以第一张图片为基准，比较图片之间相同导线与其他导线距离的偏移值
                comparison_result = compare_distances(base_dict, distances[1:])
                #print("以第一张图片为基准，比较图片之间相同导线与其他导线距离的偏移值:", comparison_result)
                # 计算图片之间相同导线偏移值最大和最小的差值
                max_min_distances = calculate_max_min_distances(comparison_result)
                #print("相同导线偏移值最大和最小的差值:", max_min_distances)
                # 计算max_min_distances与base_dict的比值
                xddx_ratios = calculate_xddx_ratios(max_min_distances, base_dict)
                #print("xddx_ratios:", xddx_ratios)
                #将每根线与其他线之间的偏移值放到一起
                new_xddx_ratios = extract_values(xddx_ratios)
                #print("new_dict:", new_xddx_ratios)
                #保留大于阈值的偏移
                filtered_dict = filter_values_above_threshold(new_xddx_ratios, threshold_xddx_py)
                #print("filtered_dict:", filtered_dict)
                xddx_length = len(filtered_xddx_sort_all[0])
                xddx_results_ratio_2 = {}
                for key, value_pairs in filtered_dict.items():
                    if value_pairs:
                        max_x_value = max(value_pairs, key=lambda pair: pair[0])
                        #print("max_x_value:", max_x_value[0])
                        hd_dx_length = len(value_pairs)
                        #print("hd_dx_length:", hd_dx_length)
                        if hd_dx_length > 1 / 2 * (xddx_length):
                            # dict[(key, max_x_value[0])]=filtered_jgq_sort_all[0][key][0],filtered_jgq_sort_all[0][key][1]
                            xddx_results_ratio_2[(filtered_xddx_sort_all[0][key][0], filtered_xddx_sort_all[0][key][1])] = max_x_value[0]
                            #print("xddx_results_ratio_2:", xddx_results_ratio_2)
                        else:
                            xddx_results_ratio_2 = {}
                    else:
                        xddx_results_ratio_2 = {}

                for key, value in xddx_results_ratio_2.items():
                    first_point = key
                    # 计算xddx左上角的坐标与gt中心点的交点坐标
                    intersection_point = line_intersection_with_rect(first_point, gt_center, gt_rect)
                    new_key = (first_point, intersection_point)
                    mapped_result_py[new_key] = value
                print("mapped_result_py two:", mapped_result_py)

            # #xml保存
            # for key, value in mapped_result_py.items():
            #     dete_obj = DeteObj(x1=int(key[0][0]), y1=int(key[0][1]), x2=int(key[1][0]), y2=int(key[1][1]), tag='dx_wd_2', conf=value)
            #     if 5 <= dete_obj.conf < 10:
            #         dete_obj.tag = "qd_dxwd_2"
            #     elif 10 < dete_obj.conf:
            #         dete_obj.tag = "zb_dxwd_2"
            #     else:
            #         dete_obj.tag = "bwd_2"
            #     dete_res_all.add_obj_2(dete_obj)
            # each_xml_final_path = os.path.join(save_xml_final_dir, FileOperationUtil.bang_path(img_path_list[0])[1] + ".xml")
            # #print("each_xml_path:", each_xml_final_path)
            # dete_res_all.save_to_xml(each_xml_final_path)

        elif jgb_sort_all:
            #print("jgb_sort_all:", jgb_sort_all)
            max_area_dict = {}
            for sublist in jgb_sort_all:
                max_area = 0
                max_rect = None
                for rect in sublist:
                    x1, y1, x2, y2 = rect
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        max_rect = rect
                # 将最大面积和对应的矩形存入字典
                if max_rect:
                    max_area_dict[max_area] = max_rect
            #以第一张图片为基准，比较字典中的矩形面积，如果面积的浮动范围不在[1/2*area,2*area],就忽略这张图片，计算符合条件矩形的最外接矩形，用所有符合条件的矩形宽和高的平均值与最外接矩形做比较
            if max_area_dict:
                first_key, first_value = next(iter(max_area_dict.items()))
                if first_key > jgb_area_conf:
                    area = first_key
                    min_area, max_area = 0.5 * area, 2 * area
                    filtered_jgb_area_dict = {k: v for k, v in max_area_dict.items() if min_area <= k <= max_area}
                    if filtered_jgb_area_dict:
                        jgb_results_ratio_2 = calculate_metrics(filtered_jgb_area_dict)
                        #print("jgb_results_ratio_2:", jgb_results_ratio_2)

                        mapped_result_py = {}
                        rect = jgb_results_ratio_2['enclosing_rectangle']
                        key = ((rect[0], rect[1]), (rect[2], rect[3]))
                        width_ratio = jgb_results_ratio_2['width_ratio']
                        mapped_result_py[key] = width_ratio

                        #print("mapped_result_py:", mapped_result_py)
                        mapped_result_py = {key: value for key, value in mapped_result_py.items() if 1.2 <= value}
                        print("mapped_result_py three:", mapped_result_py)
                else:
                    mapped_result_py = {}


            # # xml保存
            # for key, value in mapped_result_py.items():
            #     dete_obj = DeteObj(x1=int(key[0][0]), y1=int(key[0][1]), x2=int(key[1][0]), y2=int(key[1][1]),tag='dx_wd_3', conf=value)
            #     if 5 <= dete_obj.conf < 10:
            #         dete_obj.tag = "qd_dxwd_3"
            #     elif 10 < dete_obj.conf:
            #         dete_obj.tag = "zb_dxwd_3"
            #     else:
            #         dete_obj.tag = "bwd_3"
            #     dete_res_all.add_obj_2(dete_obj)
            # each_xml_final_path = os.path.join(save_xml_final_dir,
            #                                    FileOperationUtil.bang_path(img_path_list[0])[1] + ".xml")
            # #print("each_xml_path:", each_xml_final_path)
            # dete_res_all.save_to_xml(each_xml_final_path)

        else:
            mapped_result_py = {}

        xml_path_list       = list(FileOperationUtil.re_all_file(save_xml_dir, endswitch=[".xml"]))
        xml_path_list       = sorted(xml_path_list, key=lambda x: get_str_length(FileOperationUtil.bang_path(x)[1]))

        move_info, loc_list = get_dxhd_info(xml_path_list, hyparam_dct)
        # det_res             = parse_result(move_info, loc_list, hyparam_dct)
        det_res = parse_result_2(move_info, loc_list, hyparam_dct, mapped_result_py)
        print("det_res1234556:", det_res)
        return det_res

    except Exception as e:
        print('infer====>',e)
        print(e.__traceback__.tb_frame.f_globals["__file__"])  # 发生异常所在的文件
        print(e.__traceback__.tb_lineno)  # 发生异常所在的行数

        det_res ={
            "error_info": f"{e}, e.__traceback__.tb_frame.f_globals['__file__'], e.__traceback__.tb_lineno"
        }
        return det_res
