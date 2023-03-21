"""
首先顺时针点击需要矫正部分的左上，右上，右下，左下四个角点来对其进行图像矫正；
然后随机点击键盘任意按键得到矫正后的图像；
最后随机点击键盘任意按键退出。
"""
import numpy as np
import cv2 as cv
from PIL import Image
from PIL import ImageChops
import matplotlib.pyplot as plt

def order_points(pts):
    # 初始化坐标点
    rect = np.zeros((4, 2), dtype='float32')

    # 获取左上角和右下角坐标
    s = pts.sum(axis=1) # 每行像素值进行相加；若axis=0，每列像素值相加
    rect[0] = pts[np.argmin(s)] # top_left,返回s首个最小值索引，eg.[1,0,2,0],返回值为1
    rect[2] = pts[np.argmax(s)]  # bottom_left,返回s首个最大值索引，eg.[1,0,2,0],返回值为2

    # 分别计算左上角和右下角的离散差值
    diff = np.diff(pts, axis=1)  # 第i+1列减第i列
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    # 获取坐标点，并将它们分离开来
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算新图片的宽度值，选取水平差值的最大值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 计算新图片的高度值，选取垂直差值的最大值
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 构建新图片的4个坐标点,左上角为原点
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 获取透视变换矩阵并应用它
    M = cv.getPerspectiveTransform(rect, dst)
    # 进行透视变换
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后的结果
    return warped


def on_mouse(event, x, y, flags, param):
    """鼠标在原图像点击，获取矩形的四个边角点坐标"""

    global timg, points
    img2 = timg.copy()
    p0 = (0, 0)  # 初始化
    if event == cv.EVENT_LBUTTONDOWN:
        p1 = (x, y)
        points.append([x, y])
        print(p1)

        # 在点击图像处绘制圆
        # cv.circle(image, center_coordinates, radius, color, thickness)
        cv.circle(img2, p1, 4, (0, 255, 0), 4)
        cv.imshow('origin', img2)
    return p0

def Resize():
    """
    将图像的大小设置为一样的
    """
    im1 = cv.imread('32.jpg')
    im2 = cv.imread('42.jpg')
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    def resize_horizontal(h1, w1, h2, w2, target_height):
        scale_to_align = float(h1) / h2 #校准
        current_width = w1 + w2 * scale_to_align
        scale_to_fit = target_height / h1 # 高度差值
        target_w1 = int(w1 * scale_to_fit)
        target_w2 = int(w2 * scale_to_align * scale_to_fit)
        target_h = int(target_height)
        return (target_w1, target_h), (target_w2, target_h), scale_to_fit, scale_to_fit * scale_to_align, [target_w1, 0]

    target_1, target_2, scale1, scale2, offset = resize_horizontal(h1, w1, h2, w2, 800.)
    im1 = cv.resize(im1, target_1, interpolation=cv.INTER_AREA)
    im2 = cv.resize(im2, target_2, interpolation=cv.INTER_AREA)
    cv.imwrite('33.jpg', im1)
    cv.imwrite('43.jpg', im2)

def Grey_Pic():
    """
    将大小一致的图像，比较其中差异
    :return:
    """
    im1 = Image.open('34.jpg')
    im2 = Image.open('43.jpg')
    im3 = ImageChops.invert(im2) # 将图像反色
    Image.blend(im1, im3, 0.5).save('blend_pic.jpg')

def Merge_pic():
    # 读取图片
    src1 = cv.imread('34.jpg')
    src2 = cv.imread('43.jpg')

    # 图像融合
    # 第一个图像，第一个权重，第二个图像，第二个权重，[gamma图1与图2作和后添加的数值。不要太大，不然图片一片白。总和等于255以上就是纯白色了]
    # 参数6：dst，输出图片
    result = cv.addWeighted(src1, 0.5, src2, 1, 1)
    # 显示图像
    cv.imwrite('temp.jpg', result)


if __name__ == '__main__':
    # 对于透视转换

    global points, timg
    xscale, yscale = 0.5, 0.5  # 通过放大图像使点击位置更加精确
    points = []
    img = cv.imread('4.jpg')
    shape = img.shape
    timg = cv.resize(img, (int(shape[1] / xscale), int(shape[0] / yscale)))  # 放大图像
    print(timg.shape)
    cv.imshow('origin', timg)

    cv.setMouseCallback('origin', on_mouse)  # 此处设置显示的图片名称一定要和上一句以及on_mouse函数中设置的一样
    cv.waitKey(0)  # 四个角点点击完后，随机按键盘结束操作
    cv.destroyAllWindows()

    # 还原像素位置
    points = np.array(points, dtype=np.float32)
    points[:, 0] *= shape[1] / int(shape[1] / xscale)
    points[:, 1] *= shape[0] / int(shape[0] / yscale)
    warped = four_point_transform(img, points)

    cv.imshow('results', warped)
    cv.imwrite('42.jpg', warped)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Merge_pic()
