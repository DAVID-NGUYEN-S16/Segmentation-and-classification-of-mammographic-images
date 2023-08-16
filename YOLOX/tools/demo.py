#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger
import glob
import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default=["./assets/dog.jpg"], help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            # print(outputs)
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        # print(bboxes)
     
        box_max =0 ;
        x_0= y_0= x_1= y_1 = 0
        for i in range(len(bboxes)):
            box = bboxes[i]

            score = scores[i]
            if score < cls_conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            if x0 < 0: x0=0
            if x1 < 0: x1=0
            if y1 < 0: y1=0
            if y0 < 0: y0=0
            
            if abs(y1-y0)*abs(x1-x0) >= box_max:
                x_0, y_0, x_1, y_1 = x0, y0, x1, y1
        # print(x_0, y_0, x_1, y_1)
        # return img[y_0:y_1, x_0: x_1]
        h, w, c = img.shape
        if abs(x_0 - x_1) < w/4: return 0, h, 0, w 
        else : return y_0,y_1, x_0, x_1

def resize_image(image):
    new_width=224
    new_height=224
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image
def image_demo(predictor, vis_folder, path, path_mask, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    # if os.path.exists('E:/WORKBASE/Project-rsna-breast-cancer-detection/DATA_STANDARD/Croped_DDSM') == False:
    #     os.makedirs('E:/WORKBASE/Project-rsna-breast-cancer-detection/DATA_STANDARD/Croped_DDSM')
    file_name = os.path.basename(path).replace('.png', '')
    # if status == 0: status = 'Benign'
    # else: status = 'Cancer'
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        # print(img_info)
        img = img_info["raw_img"]
        try:
            y_0,y_1, x_0, x_1 = predictor.visual(outputs[0], img_info, predictor.confthre)
        except:
            h, w, c = img.shape
            y_0,y_1, x_0, x_1 = 0, h, 0, w 
        
        
        
        if save_result:
            
            # save_file_name = os.path.join('E:/WORKBASE/Project-rsna-breast-cancer-detection/DATA_STANDARD/Croped_DDSM', os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(file_name))
            # print(result_image.shape)
            # result_image = cv2.resize(result_image, (512, 512))
            # try:
            #     result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2RGB)
            # except:
            #     result_image = result_image 
            
            # if os.path.exists(f'E:/Data_ddsm/{status}/{file_name}')  == False: os.makedirs(f'E:/Meta_data_test/{file_name}')
            cv2.imwrite(path, resize_image(img[y_0: y_1, x_0: x_1 ]))
            for paths in path_mask: 
                mask = cv2.imread(paths)
                cv2.imwrite(paths, resize_image(mask[y_0: y_1, x_0: x_1 ]))
        # ch = cv2.waitKey(0)
        # if ch == 27 or ch == ord("q") or ch == ord("Q"):
        #     break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
import pandas as pd

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    if args.demo == "image":
        
        # print(f'args.path = {args.path}')
        # list_path = glob.glob(f'{args.path}*.png')
        # print(list_path)
        data = pd.read_csv(f'E:/Processing Data/{args.path}')
        total_time = 0
        for id in data.index:
            t0 = time.time()
            
            path = '/'.join([data.Image[id].split('/')[0], data.Image[id].split('/')[-1]])
            path = 'E:/Processing Data/' + path
            # img = cv2.imread(path)
            # h, w, c = img.shape
            # if h == 1024 and w == 512: continue
            # path_mask = [f'E:/MINI-DDSM-Complete-PNG-16/{data.Tumour_Contour[id]}', f'E:/MINI-DDSM-Complete-PNG-16/{data.Tumour_Contour2[id]}']
            
            path_mask = []
            # if os.path.exists(f"E:/WORKBASE/Project-rsna-breast-cancer-detection/META_DATA/Data_image/{par}/") == False:
            #     os.makedirs(f"E:/WORKBASE/Project-rsna-breast-cancer-detection/META_DATA/Data_image/{par}/")
            # if os.path.exists(f"E:/WORKBASE/Project-rsna-breast-cancer-detection/META_DATA/Data_image/{par}/{view}/") == False:
            #     os.makedirs(f"E:/WORKBASE/Project-rsna-breast-cancer-detection/META_DATA/Data_image/{par}/{view}/")
            # if os.path.exists(f"E:/WORKBASE/Project-rsna-breast-cancer-detection/META_DATA/Data_image/{par}/{view}/{lat}/") == False:
            #     os.makedirs(f"E:/WORKBASE/Project-rsna-breast-cancer-detection/META_DATA/Data_image/{par}/{view}/{lat}/")
            # if os.path.exists(f"E:/WORKBASE/Project-rsna-breast-cancer-detection/META_DATA/Data_image/{par}/{lat}/{view}/") == False:
            #     os.mkdir(f"E:/WORKBASE/Project-rsna-breast-cancer-detection/META_DATA/Data_image/{par}/{lat}/{view}/")
            # if data.Tumour_Contour2[id] == '-': path_mask = [f'E:/MINI-DDSM-Complete-PNG-16/{data.Tumour_Contour[id]}']
            # if data.Status[id] == 'Normal': path_mask = []
            image_demo(predictor, vis_folder, path, path_mask, current_time, args.save_result)
            t = time.time() - t0
            print(f'Total time in {path} = {t}')
            total_time +=t
            logger.info("Total time in {} = {:.4f}s".format(path, time.time() - t0))
        print(f"Mean time in a image = {total_time/len(data.index)}")
        # for path in list_path:
        #     name = os.path.basename(path)
        #     file_name = f"E:/Download/Croped_Image/{name}"
        #     t0 = time.time()
        #     image_demo(predictor, vis_folder, path, current_time, args.save_result, file_name)
        #     t = time.time() - t0
        #     print(f'Total time in {path} = {t}')
        #     total_time +=t
        #     logger.info("Total time in {} = {:.4f}s".format(path, time.time() - t0))
        # print(f"Mean time in a image = {total_time/len(list_path)}")
            
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    # print(f'args.exp_file = {args.exp_file}')
    # print(f'exp = {exp}')
    # print(f'args = {args}')
    # print(args.path)
    main(exp, args)
