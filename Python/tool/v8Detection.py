import torch
import numpy as np
import copy
import os
import cv2
import pdb
import time
import configparser
import os.path as osp
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.engine.results import Results
from JoTools.txkjRes.resTools import ResTools
from JoTools.txkjRes.deteRes import DeteRes
from JoTools.txkjRes.deteObj import DeteObj

from JoTools.utils.FileOperationUtil import FileOperationUtil

def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
   
    cpu = device.lower() == 'cpu'
    cuda = not cpu and torch.cuda.is_available()
    # logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return torch.device('cuda:0' if cuda else 'cpu')

class v8Detection:
    # def __init__(self, model_path='./gt_jgqv8.pt', device='cuda:0', conf=0.5, iou=0.7):
    def __init__(self, modelDir,modelId,cfgPath):
        # self.workDir  = osp.dirname(osp.abspath(__file__))
        # self.modelDir = osp.join(self.workDir, 'models')
        self.modelDir = modelDir
        self.modelId  = modelId #[model]
        self.cfgPath  = cfgPath
        self.readCfg()
        self.device = select_device(str(self.gpuID))
        # self.conf = conf
        # self.iou = iou
        modelPath=osp.join(self.modelDir, self.modelname)
        #print(f'· modelPath = {modelPath},{os.path.exists(modelPath)}')
        self.model = self.load_model(modelPath)
        self.warm_up()
        
    def readCfg(self):
        self.cf = configparser.ConfigParser()
        self.cf.read(self.cfgPath)
        self.gpuID     = self.cf.get(self.modelId, "device")
        self.modelname = self.cf.get(self.modelId, 'modelname')
        print(f'· modelname = {self.modelname}')
        self.imgsz     = self.cf.getint(self.modelId, 'img_size')
        print(f'· inference imgsz = {self.imgsz}')
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
        self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))
        print('detModel start success!')


    def detect_objects(self, im, imgname="default.jpg"):
        # im = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)#只能用该方法读取含有中文路径的图片
        # im = cv.imread(img_path)
        orig_imgs = copy.deepcopy(im)
        im = [im]
        im = [LetterBox([self.imgsz, self.imgsz], auto=True, stride=32)(image=x) for x in im]
        im = im[0][None]
        im = im[..., ::-1].transpose((0, 3, 1, 2))
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device).float() / 255

        preds = self.model(im)
        prediction = ops.non_max_suppression(preds, self.conf, self.iou, agnostic=False, max_det=300, classes=None, nc=len(self.model.names))

        if len(prediction):
            pred = prediction[0]
            orig_img = orig_imgs[0] if isinstance(orig_imgs, list) else orig_imgs
            shape = orig_img.shape
            #boxes = ops.scale_boxes(im.shape[2:], pred[:, :4], shape).round().cpu().numpy()
            boxes = ops.scale_boxes(im.shape[2:], pred[:, :4], shape).round().cpu().detach().numpy()
            #scores= pred[:, -2].cpu().numpy()
            scores = pred[:, -2].cpu().detach().numpy()
            #classes = (pred[:, -1].cpu().numpy().astype(np.int8))
            classes = (pred[:, -1].cpu().detach().numpy().astype(np.int8))

            
            return boxes, classes, scores
        else:
            # print('white')
            return [],[],[]

    def detectSOUT(self, path=None,image=None, image_name="default.jpg",output_type='txkj'):
        if path==None and image is None:
            raise ValueError("path and image cannot be both None")
        dete_res = DeteRes()
        dete_res.img_path = path
        dete_res.file_name = image_name
        if image is None:
            image = dete_res.get_img_array()
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        boxes, classes, scores = self.detect_objects(bgr,image_name)
        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[i]
            label = self.id2label[classes[i]]
            # label = str(classes[i]) #self.id2label[classes[i]]
            prob = float(scores[i])
            dete_obj = DeteObj(x1=int(xmin), y1=int(ymin), x2=int(xmax), y2=int(ymax), tag=label, conf=prob, assign_id=i)
            dete_res.add_obj_2(dete_obj)
        if output_type == 'txkj':
            return dete_res
        elif output_type == 'json':
            pass
        return dete_res
        
# # 使用示例
# img_dir = '/home/suanfa-2/jzp/new_start/0002_datasets/002_linemovetmp/datasets/completeTowerv1/imgs'
# detector = v8Detection('det', './point.ini')
# for imgname in os.listdir(img_dir):
#     if imgname.endswith('jpg'):
#         print(imgname)
#         s_t = time.time()
#         # results = detector.detect_objects(osp.join(img_dir, imgname))
#         results = detector.detectSOUT(osp.join(img_dir, imgname))
#         results.print_as_fzc_format()
#         # pdb.set_trace()
#         print(time.time() - s_t)
#         # print(results)
