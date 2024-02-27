import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN
import argparse
import yaml
import torch
import torchvision

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)
def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        # if (time.time() - t) > time_limit:
        #     LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
        #     break  # time limit exceeded

    return output

class onnx2rknn(object):
    def __init__(self,opt):
        self.opt = opt 
        self.input_w,self.input_h = opt.input_size 
        self.QUANTIZE_ON = True
        self.get_class_dict()
        
    def get_class_dict(self):
        if 'txt' in  self.opt.class_file:
            class_list = open(self.opt.class_file,'r').readlines() 
        else:
            with open(self.opt.class_file, errors='ignore') as f:
                d = yaml.safe_load(f)
            class_list = d['names']
        self.id_class_dict = {} 
        for index,symbol_line in enumerate(class_list):
            symbol = symbol_line.strip()
            self.id_class_dict[str(index)] = symbol 
        
    def image_process(self,img):
        # # 获取训练图像的shape 高，宽
        # img_h,img_w,_ = img.shape
        # # 计算将高度缩放到input_height后的缩放比例
        # rate = self.input_h*1.0/img_h
        # # 将宽度进行等比例缩小
        # input_width = int(img_w*rate)
        # resize 
        img = cv2.resize(img,(self.input_h,self.input_h))
        # 设置一个空的img
        # if input_width>self.input_w:
        #     new_img = img[:,:self.input_w,:]
        # else:
        #     new_img = np.zeros((self.input_h,self.input_w,img.shape[2])).astype(np.uint8)
        #     new_img[:,0:input_width,:] = img
        # new_img = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
        return img
    
    def decode(self,preds):
        ''' 预测结果解码 '''

        pred = ''
        for i,w in enumerate(preds):
            if w != 0 and (i==0 or (i!=0 and w !=preds[i-1])):
                pred += self.id_class_dict[str(w)]
        return pred 


    def transformer(self):
        # Create RKNN object
        rknn = RKNN(verbose=True)

        # pre-process config
        print('--> Config model')
        rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]], target_platform="rk3568")
        print('done')

        # Load ONNX model
        print('--> Loading model')
        ret = rknn.load_onnx(model=self.opt.onnx_model)
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        print('done')

        # # 构建RKNN模型，这里设置do_quantization为true开启量化，dataset是指定用于量化校正的数据集
        # print('--> Building model')
        # ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
        # if ret != 0:
        #     print('Build model failed!')
        #     exit(ret)
        # print('done')

        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            print('Build model failed!')
            exit(ret)
        print('done')

        # Export RKNN model
        print('--> Export rknn model')
        ret = rknn.export_rknn(self.opt.rnkk_model)
        if ret != 0:
            print('Export rknn model failed!')
            exit(ret)
        print('done')

        # Init runtime environment
        print('--> Init runtime environment')
        ret = rknn.init_runtime()
        # ret = rknn.init_runtime(target='rk3568',device_id='DKUZ8B0PAB')
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)
        print('done')
        
        
        # 测试效果如下
        img = cv2.imread(self.opt.test_img)
        # img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
        img = self.image_process(img)
        # img = np.expand_dims(img,axis=0)
        # img = np.expand_dims(img,axis=3)
        # Inference
        print('--> Running model')
        # output = rknn.inference(inputs=[img])[0]
        # output = non_max_suppression(torch.Tensor(output))
        # output = output[0]
        # # pred_label = np.argmax(output,axis=1)
        # # pred = self.decode(pred_label)
        # for i in range(output.shape[0]):
        #     xmin, ymin, xamx, ymax = int(output[i, 0]), int(output[i, 1]), int(output[i, 2]), int(output[i, 3])
        #     cv2.rectangle(img, (xmin, ymin), (xamx, ymax), (0, 0, 255), thickness=3)
        #     # for j in range(5):
        #     #     cv2.circle(img, (int(kpss[i, j, 0]), int(kpss[i, j, 1])), 5, (0,242,255), thickness=-1)
        #     cv2.putText(img, str(round(float(output[i,4]), 3)), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), thickness=3)
        # cv2.imwrite('./test.png',img)
        # # if len(pred)==0:
        # #     pred = ' ---This is empty!--- '
        # # print('Result == > {}'.format(pred))


        # test code 2
        output = rknn.inference(inputs=[img])[0]
        bboxes = output[0][:,:4] #xywh
        scores = output[0][:,4]
        # bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]
        ratioh, ratiow = 1, 1
        bboxes[:, 0] = (bboxes[:, 0] - 0) * ratiow
        bboxes[:, 1] = (bboxes[:, 1] - 0) * ratioh
        bboxes[:, 2] = bboxes[:, 2] * ratiow
        bboxes[:, 3] = bboxes[:, 3] * ratioh
        # kpss[:, :, 0] = (kpss[:, :, 0] - padw) * ratiow
        # kpss[:, :, 1] = (kpss[:, :, 1] - padh) * ratioh
        scores_index = scores>0.5
        bboxes = bboxes[scores_index]
        scores = scores[scores_index]

        indices = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), 0.5, 0.45)
        for i in indices:
            xmin, ymin, xamx, ymax = int(bboxes[i, 0]), int(bboxes[i, 1]), int(bboxes[i, 0] + bboxes[i, 2]), int(bboxes[i, 1] + bboxes[i, 3])
            cv2.rectangle(img, (xmin, ymin), (xamx, ymax), (0, 0, 255), thickness=3)
            cv2.putText(img, str(round(scores[i], 3)), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), thickness=3)
        cv2.imwrite('./test2.png',img)

        rknn.release()


def set_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size',type=list,default=[640,640])
    parser.add_argument('--onnx_model',type=str,default='runs/train/exp7/weights/best.onnx')
    parser.add_argument('--rnkk_model',type=str,default='runs/train/exp7/weights/best_fp16.rknn')
    parser.add_argument('--class_file',type=str,default='data/pcb.yaml')
    parser.add_argument('--test_img',type=str,default='dataset/pcb/PCB_DATASET/images/Missing_hole/05_missing_hole_02.jpg')
    opt = parser.parse_args()
    return opt 

if __name__ == '__main__':
    opt = set_option()
    onnx2rknn_object = onnx2rknn(opt)
    onnx2rknn_object.transformer()



#reference url:https://zhuanlan.zhihu.com/p/671722037