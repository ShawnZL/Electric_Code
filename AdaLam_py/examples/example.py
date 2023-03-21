import cv2
import numpy as np
import argparse
import sys
import random
from adalam import AdalamFilter
import matplotlib.pyplot as plt


def extract_keypoints(impath):
    im = cv2.imread(impath, cv2.IMREAD_COLOR)
    # 读取照片
    d = cv2.xfeatures2d.SIFT_create(nfeatures=8000, contrastThreshold=1e-5)
    # d 是提取的SITF特征点 使用SIFT查找关键点key points和描述符descriptors
    kp1, desc1 = d.detectAndCompute(im, mask=np.ones(shape=im.shape[:-1] + (1,),dtype=np.uint8))

    pts = np.array([k.pt for k in kp1], dtype=np.float32)
    ors = np.array([k.angle for k in kp1], dtype=np.float32)
    scs = np.array([k.size for k in kp1], dtype=np.float32)
    # pts:坐标，ors方向，scs领域直径
    return pts, ors, scs, desc1, im


def show_matches(img1, img2, k1, k2, target_dim=800.):
    h1, w1 = img1.shape[:2] #获取图片长宽 [:3] 获取长宽和通道
    h2, w2 = img2.shape[:2]
    """
        img.shape[0]：图像的垂直尺寸（高度）
        img.shape[1]：图像的水平尺寸（宽度）
        img.shape[2]：图像的通道数
    """
    def resize_horizontal(h1, w1, h2, w2, target_height):
        scale_to_align = float(h1) / h2 #校准
        current_width = w1 + w2 * scale_to_align
        scale_to_fit = target_height / h1 # 高度差值
        target_w1 = int(w1 * scale_to_fit)
        target_w2 = int(w2 * scale_to_align * scale_to_fit)
        target_h = int(target_height)
        return (target_w1, target_h), (target_w2, target_h), scale_to_fit, scale_to_fit * scale_to_align, [target_w1, 0]

    target_1, target_2, scale1, scale2, offset = resize_horizontal(h1, w1, h2, w2, target_dim)

    im1 = cv2.resize(img1, target_1, interpolation=cv2.INTER_AREA)
    im2 = cv2.resize(img2, target_2, interpolation=cv2.INTER_AREA)

    h1, w1 = target_1[::-1]
    h2, w2 = target_2[::-1]

    vis = np.ones((max(h1, h2), w1 + w2, 3), np.uint8) * 255
    vis[:h1, :w1] = im1
    vis[:h2, w1:w1 + w2] = im2

    p1 = [np.int32(k * scale1) for k in k1]
    p2 = [np.int32(k * scale2 + offset) for k in k2]


    for (x1, y1), (x2, y2) in zip(p1, p2):
        # 画直线
        # cv2.line(vis, (x1, y1), (x2, y2), [0, 255, 0], 1)

        # 画圆形 半径为2
        # cv2.circle(vis, (x1, y1),2,[0,255,0],5)
        # cv2.circle(vis, (x2, y2),2,[0,255,0],5)
        # 使用不同颜色将圈标记出来
        
        colors = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        cv2.circle(vis, (x1, y1), 2, colors, 2)
        cv2.circle(vis, (x2, y2), 2, colors, 2)


    cv2.imshow("AdaLAM example", vis)
    cv2.imwrite('AdaLAM_circle_3242_difcol.jpg', vis)
    # 显示10S
    # cv2.waitKey(10)


if __name__ == '__main__':
    """
    p = argparse.ArgumentParser()
    p.add_argument("--im1", required=True)
    p.add_argument("--im2", required=True)
    opt = p.parse_args()
    """
    k1, o1, s1, d1, im1 = extract_keypoints('32.jpg')
    k2, o2, s2, d2, im2 = extract_keypoints('42.jpg')

    matcher = AdalamFilter()
    matches = matcher.match_and_filter(k1=k1, k2=k2,
                                       o1=o1, o2=o2,
                                       d1=d1, d2=d2,
                                       s1=s1, s2=s2,
                                       im1shape=im1.shape[:2], im2shape=im2.shape[:2]).cpu().numpy()

    show_matches(im1, im2, k1=k1[matches[:, 0]], k2=k2[matches[:, 1]])







