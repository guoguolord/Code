import torch
import cv2 as cv
import numpy as np
import copy
import os
import cv2
import pdb
import time
import os.path as osp
import configparser
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.engine.results import Results

def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
   
    cpu = device.lower() == 'cpu'
    cuda = not cpu and torch.cuda.is_available()
    # logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return torch.device('cuda:0' if cuda else 'cpu')


class pointDetector:
    # def __init__(self, model_path='./best.pt', device='cuda:0', conf=0.25, iou=0.7):
    def __init__(self, modelId,cfgPath):
        # self.workDir  = osp.dirname(osp.abspath(__file__))
        # self.modelDir = osp.join(self.workDir, 'models')
        self.modelDir = './models'
        
        self.modelId  = modelId #[model]
        self.cfgPath  = cfgPath
        self.readCfg()
        self.device = select_device(str(self.gpuID))
        # self.conf = conf
        # self.iou = iou
        modelPath=osp.join(self.modelDir, self.modelname)
        self.model = self.load_model(modelPath)
        self.warm_up()
        
        
    def readCfg(self):
        self.cf = configparser.ConfigParser()
        self.cf.read(self.cfgPath)
        self.gpuID     = self.cf.get(self.modelId, "device")
        self.modelname = self.cf.get(self.modelId, 'modelname')
        self.imgsz     = self.cf.getint(self.modelId, 'img_size')
        self.conf     = self.cf.getfloat(self.modelId, 'conf_threshold')
        self.iou       = self.cf.getfloat(self.modelId, 'iou_threshold')
        self.augment   = self.cf.getboolean(self.modelId, 'augment')
        self.classes   = [c.strip() for c in self.cf.get(self.modelId, "classes").split(",")]
        self.id2label  = dict(zip(range(len(self.classes)), self.classes))
        self.visible_classes = [c.strip() for c in self.cf.get(self.modelId, "visible_classes").split(",")]
        
    def load_model(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        model = ckpt['model'].to(self.device).float()
        model.eval()
        return model

    def warm_up(self):
        self.model(torch.zeros(1, 3, 1280, 1280).to(self.device).type_as(next(self.model.parameters())))
        print('pointModel start success!')

    def process_detection_results(self, pred_boxes, pred_kpts, image_width, image_height):
        # 你的后处理函数代码
        num_targets = pred_boxes.size(0)  # 目标框的数量

        # 创建一个空字典来存储结果
        detected_objects = {}

        # 遍历每个目标框
        for i in range(num_targets):
            target_data = {}  # 创建一个空字典来存储每个目标框的信息

            # 提取目标框的坐标信息并转换为整数
            target_box = pred_boxes[i].tolist()
            target_x1, target_y1, target_x2, target_y2 = map(int, target_box)
            target_x1 = max(0, min(target_x1, image_width - 1))
            target_y1 = max(0, min(target_y1, image_height - 1))
            target_x2 = max(0, min(target_x2, image_width - 1))
            target_y2 = max(0, min(target_y2, image_height - 1))
            target_data['bbox'] = [target_x1, target_y1, target_x2, target_y2]  # 存储目标框的坐标

            # 提取目标框关键点的信息并转换为整数
            target_kpts = pred_kpts[i].tolist()
            keypoint_data = []
            for kpt in target_kpts:
                (kpt_x, kpt_y), confidence = map(int, kpt[:2]), kpt[2]
                kpt_x = max(0, min(kpt_x, image_width - 1))
                kpt_y = max(0, min(kpt_y, image_height - 1))
                keypoint_data.append({'x': kpt_x, 'y': kpt_y, 'confidence': confidence})  # 存储关键点的信息
            target_data['keypoints'] = keypoint_data  # 存储关键点列表

            # 将目标框信息存储到主字典中
            detected_objects[f'Target_{i+1}'] = target_data

        return detected_objects

    def detect_objects(self, img_path):
        im = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)#只能用该方法读取含有中文路径的图片
        # im = cv.imread(img_path)
        orig_imgs = copy.deepcopy(im)
        im = [im]
        im = [LetterBox([1280, 1280], auto=True, stride=32)(image=x) for x in im]
        im = im[0][None]
        im = im[..., ::-1].transpose((0, 3, 1, 2))
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device).float() / 255

        preds = self.model(im)
        prediction = ops.non_max_suppression(preds, self.conf, self.iou, agnostic=False, max_det=300, classes=None, nc=len(self.model.names))

        results = []
        for i, pred in enumerate(prediction):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            src_h, src_w, _ = orig_img.shape
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(im.shape[2:], pred[:, :4], shape).round()
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(im.shape[2:], pred_kpts, shape)
            detect_objects = self.process_detection_results(pred[:, :4], pred_kpts, src_w, src_h)
            results.append(detect_objects)

        return results

    def sort_boxes_by_area(self,obj2dct):
        point_lst = []
        max_area = 0
        max_area_keypoints = []

        for _, info in obj2dct.items():
            bbox = info['bbox']
            area = bbox[2]*bbox[3]  # 计算面积
            if area > max_area:
                max_area = area
                max_area_keypoints = info['keypoints']
        for dct in max_area_keypoints:
            point_lst.append([dct['x'], dct['y']])
        return point_lst

    def sort_zp(self, points):
        sorted_points = sorted(points, key=lambda x:x[1])
        # 分组排序
        grouped_points = [sorted_points[i:min(i+2,len(sorted_points))] for i in range(0, len(sorted_points), 2)]

        sorted_grouped_points = []
        for group in grouped_points:
            sorted_grouped_points.extend(sorted(group, key=lambda x: x[0]))
        return sorted_grouped_points
    
    def detect_point_lst(self,img_path):
        det_lst    = self.detect_objects(img_path)
        if not len(det_lst):
            return []
        point_lst = self.sort_boxes_by_area(det_lst[0])
        if len(point_lst)<=5:
            return []
        point_lst  = self.sort_zp(point_lst)[:6]
        return point_lst
        
# # 使用示例
# img_dir = '/home/suanfa-2/jzp/new_start/0002_datasets/002_linemovetmp/datasets/completeTowerv1/imgs'
# detector = ObjectDetector()
# for imgname in os.listdir(img_dir):
#     if imgname.endswith('jpg'):
#         print(imgname)
#         s_t = time.time()
#         # results = detector.detect_objects(osp.join(img_dir, imgname))
#         results = detector.detect_point_lst(osp.join(img_dir, imgname))
#         print(time.time() - s_t)
#         print(results)