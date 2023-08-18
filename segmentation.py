import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import torch
import time
import logging
import argparse

 
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)  # 关闭图像自动缩放功能
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
 
        
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def show_boxes(boxes, ax):
    rects = [Rectangle((x-w/2, y-h/2), w, h) for x, y, w, h in boxes]   # 因为用到Rectangle，该函数的输入形式是左下角点+宽高
    # 创建一个 PatchCollection 对象，并将矩形列表添加到其中
    rect_collection = PatchCollection(rects, edgecolor='green', facecolor=(0,0,0,0), lw=2)
    # 将 PatchCollection 对象添加到轴对象上
    ax.add_collection(rect_collection)

def generate_points(label_path, img_size):
    input_points = None
    with open(label_path,'r') as f:
        for line in f.readlines():
            x, y = float(line.split()[1]), float(line.split()[2])
            x, y = x*img_size[0], y*img_size[1]
            if input_points is None:
                input_points = np.array([x, y])
            else:
                input_points = np.vstack((input_points, np.array([x, y])))
    return input_points        


def generate_box(label_path, img_size):
    input_box = None
    with open(label_path,'r') as f:
        for line in f.readlines():
            x, y, w, h = float(line.split()[1]), float(line.split()[2]), float(line.split()[3]), float(line.split()[4])
            x, y, w, h = x*img_size[0], y*img_size[1], w*img_size[0], h*img_size[1]
            if input_box is None:
                input_box = np.array([x, y, w, h])
            else:
                input_box = np.vstack((input_box, np.array([x, y, w, h]))) 
    return input_box        


sys.path.append("..")    # 让解释器在上一级目录搜索模块

# 设置日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建日志处理器，将日志输出到文件中
file_handler = logging.FileHandler('/hsiam02/dw/workspace/log/SAM.log')
file_handler.setLevel(logging.INFO)

# 创建日志格式化器
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 添加日志处理器到记录器中
logger.addHandler(file_handler)


def mask2origin(input_path, output_path, mask_generator):
    if os.path.isdir(input_path):
        img_list = os.listdir(input_path)
        logger.info('---------------------------------')
        logger.info('mask2origin---推理开始！')
        total_time = 0
        for i in img_list:
            if i.endswith(".bmp"):       # 如果还想指定他扩展名文件，输入一个元组即可
                img_path = os.path.join(input_path, i)
                image = cv2.imread(img_path)     # 读取的图像通常以BGR格式存储
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    # 将其转换为RGB格式
                
                
                start_time = time.time()
                masks = mask_generator.generate(image)
                end_time = time.time()
                run_time = end_time - start_time
                total_time += run_time
                logger.info('{} 推理用时: {}秒'.format(i, run_time))
                # print(len(masks))
                # plt.figure(figsize=(20,20))
                
                # plt.imshow(image)     # 将image变量中的图像显示在图像窗口中
                # show_anns(masks)
                # plt.axis('off')
                # filename = i.split('.')[0]
                # plt.savefig("/hsiam02/dw/segment-anything/output/mask_to_origin/{}.png".format(filename))
                # plt.close()
        logger.info('推理总用时: {}秒'.format(total_time))
        
    else:
        image = cv2.imread(input_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)   
        plt.figure(figsize=(20,20))
        show_anns(masks)
        plt.axis('off') 
        filename = i.split('/')[-1].split['.'][0]
        plt.savefig("/hsiam02/dw/segment-anything/mask_to_origin/{}.png".format(filename))
        plt.savefig(output_path + '/mask_to_origin/{}.png'.format(filename))
        plt.close()


def prompt(path, mode, sam):
    predictor = SamPredictor(sam)
    if os.path.isdir(path):
        img_list = os.listdir(path)
        logger.info('---------------------------------')
        logger.info('prompt_point2mask---推理开始！')
        total_time = 0
        for i in img_list:
            if i.endswith(".bmp"):   
                img_path = os.path.join(path, i)
                
                load_start_time = time.time()
                image = cv2.imread(img_path)
                image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                load_end_time = time.time()
                load_run_time = load_end_time - load_start_time
                logger.info('{} 读取用时: {}秒'.format(i, load_run_time))
                
                start_time = time.time()
                predictor.set_image(image_array)
                end_time = time.time()
                run_time_1 = end_time - start_time
                
                # input_point = np.array([[120, 135]])
                label_file = img_path.split('/')[-1].split('.')[0] + '.txt'
                label_path = os.path.join(path, label_file)
                
                if mode == 'points':
                    input_point = generate_points(label_path, (2448, 2048))
                    input_label = np.zeros(len(input_point)) + 1
                    filename = i.split('.')[0]
                    
                    # plt.imshow(image_array)
                    # show_points(input_point, input_label, plt.gca())
                    # plt.axis('off')
                    # plt.savefig("/hsiam02/dw/segment-anything/output/prompt/prompt_box_xy/prompt_points/{}.png".format(filename))
                    # plt.close()
                    
                    
                    start_time = time.time()
                    masks, scores, logits = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True,
                    )
                    end_time = time.time()
                    run_time_2 = end_time - start_time + run_time_1
                    logger.info('{} 推理用时: {}秒'.format(i, run_time_2))
                    total_time += run_time_2
                    
                    
                    for num, (mask, score) in enumerate(zip(masks, scores)):
                        # plt.figure(figsize=(20,20))
                        plt.imshow(image_array)
                        show_mask(mask, plt.gca())
                        show_points(input_point, input_label, plt.gca())
                        plt.axis('off')
                        plt.title(f"Mask {num+1}, Score: {score:.3f}", fontsize=18)
                        plt.savefig("/hsiam02/dw/segment-anything/output/prompt/prompt_box_xy/{}/mask/{}_mask{}.png".format(model_type, filename, num+1))
                        plt.close()  
                        
        logger.info('推理总用时: {}秒'.format(total_time))        
                # if mode == 'box':
                #     input_boxes = generate_box(label_path, (2448, 2048))
                #     plt.imshow(image_array)
                #     show_boxes(input_boxes, plt.gca())
                #     plt.axis('off')
                #     filename = i.split('.')[0]
                #     plt.savefig("/hsiam02/dw/segment-anything/output/prompt/prompt_box/prompt_boxes/{}.png".format(filename))
                #     plt.close()
                    
                #     transformed_boxes = predictor.transform.apply_boxes_torch(torch.from_numpy(input_boxes).to(device), image.shape[:2])
                #     masks, _, _ = predictor.predict_torch(
                #         point_coords=None,
                #         point_labels=None,
                #         boxes=transformed_boxes,
                #         multimask_output=False,
                #     )
                #     plt.imshow(image_array)
                #     for mask in masks:
                #         show_mask(mask, plt.gca(), random_color=True)
                #     for box in input_boxes:
                #         show_box(input_boxes.cpu().numpy(), plt.gca())
                #     plt.axis('off')
                #     plt.savefig("/hsiam02/dw/segment-anything/output/prompt/prompt_box/mask/{}.png".format(filename))
                #     plt.close()


def prompt2mask(input_path, output_path, sam, crop_mask_type):
    predictor = SamPredictor(sam)
    if os.path.isdir(input_path):
        img_list = os.listdir(input_path)
        img_folder = input_path.split('/')[-1]
        for i in img_list:
            if i.endswith(".bmp"):   
                img_path = os.path.join(input_path, i)
                image = cv2.imread(img_path)
                image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                predictor.set_image(image_array)
                label_file = img_path.split('/')[-1].split('.')[0] + '.txt'
                label_path = os.path.join(input_path, label_file)
                input_point = generate_points(label_path, (2448, 2048))
                input_label = np.zeros(len(input_point)) + 1
                filename = i.split('.')[0]
                                
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )
                
                if crop_mask_type == 'entire_img':
                    for num, mask_data in enumerate(masks):
                        if num == 2:      # vit_h对应的mask3分割效果最好
                            mask_data
                            cv2.imwrite(output_path + img_folder + '/whole_mask/{}.png'.format(filename), mask_data * 255)
                
                if crop_mask_type == 'crop_obj':
                    for num, mask_data in enumerate(masks):
                        if num == 2:   
                            # 将二进制掩码转为灰度掩码
                            
                            mask_data = mask_data.astype(np.uint8) * 255    # mask_data是布尔型

                            # 使用cv2.threshold函数进行阈值处理
                            _, binary_mask = cv2.threshold(mask_data, 127, 255, cv2.THRESH_BINARY)
                            
                            # 找到二进制掩码的轮廓
                            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            # 遍历每个轮廓
                            for n, contour in enumerate(contours):
                                # 获取边界框
                                x, y, w, h = cv2.boundingRect(contour)
                                
                                # 剪裁mask图像
                                cropped_mask = mask_data[y:y+h, x:x+w]
                                
                                # 剪裁原始图像
                                cropped_img = image[y:y+h, x:x+w]
                                # 在图像中把非分割部分全变黑
                                # cropped_img = cv2.bitwise_and(image, image, mask=mask_data)

                                cv2.imwrite(output_path + img_folder + '/crop/mask/{}_{}.png'.format(filename, n+1), cropped_mask)

                                cv2.imwrite(output_path + img_folder + '/crop/img/{}_{}.png'.format(filename, n+1), cropped_img)

def main():
    parser = argparse.ArgumentParser(description='segmentation based on SAM')
    
    parser.add_argument('--input_path', type=str, default='/hsiam02/dw/images_origin', help='path of the image to be segmented')
    
    parser.add_argument('--output_path', type=str, default='/hsiam02/dw/segment-anything/output', help='path to save the segmented image')
    
    parser.add_argument('--model_type', type=str, required=True, help='the type of pretrained model: vit_h, vit_l, vit_b')
    
    parser.add_argument('--mask_on_origin_img', type=bool, help='whether to draw the mask onto the original image')
    
    parser.add_argument('--prompt_crop_mask', type=bool, help='whether to crop masks based on prompt')
    
    parser.add_argument('--crop_mask_type', type=str, help='choose to output the entire image or crop by object: entire_img, crop_obj')
    
    args = parser.parse_args()
    
    pretrain_model_path = '/hsiam02/dw/segment-anything/pretrain_model'
    sam_checkpoint = None
    model_list = os.listdir(pretrain_model_path)
    for i in model_list:
        if args.model_type in i:
            sam_checkpoint = os.path.join(pretrain_model_path, i)  # SAM预训练模型路径
              
    model_type = args.model_type
    logger.info('---------------------------------')
    logger.info('加载的预训练模型为 SAM-{}'.format(model_type)) 
        
    device = "cuda"   #如果想用cpu,改成cpu即可	
    print(sam_checkpoint)
    start_time = time.time()
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)   # 加载模型
    end_time = time.time()
    run_time = end_time - start_time
    logger.info('模型加载时长: {}秒'.format(run_time))

    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam) 
      
    if args.mask_on_origin_img:
        mask2origin(args.input_path, args.oupt_path, mask_generator)
        
    if args.prompt_crop_mask and args.crop_mask_type is None:
        parser.error("--crop_mask_type must be specified if --prompt_crop_mask is True")   
        
    if args.prompt_crop_mask and args.crop_mask_type is not None:    
        prompt2mask(args.input_path, args.output_path, sam, args.crop_mask_type)
    
                                    
    # mask2origin('/hsiam02/dw/images_origin')
    # prompt('/hsiam02/dw/images_origin', 'points')
    
if __name__ == '__main__':
    main()
