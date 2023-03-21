import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.image as mping

#image1 = mping.imread('1.jpg')
#image2 = mping.imread('2.jpg')
image1 = mping.imread('3.jpg')
image2 = mping.imread('4.jpg')

"""
plt.figure()
plt.imshow(image1)
plt.savefig('image1.png', dpi=300)

plt.figure()
plt.imshow(image2)
plt.savefig('image2.png', dpi=300)
"""

#  计算特征点提取&生成描述时间
start = time.time()
sift = cv2.xfeatures2d.SIFT_create()
#  使用SIFT查找关键点key points和描述符descriptors
kp1, des1 = sift.detectAndCompute(image1, None)
kp2, des2 = sift.detectAndCompute(image2, None)

"""
end = time.time()
#print("特征点提取&生成描述运行时间:%.2f秒" % (end - start))

kp_image3 = cv2.drawKeypoints(image1, kp1, None)
kp_image4 = cv2.drawKeypoints(image2, kp2, None)

plt.figure()
plt.imshow(kp_image3)
plt.savefig('kp_image3.jpg', dpi=300)

plt.figure()
plt.imshow(kp_image4)
plt.savefig('kp_image4.jpg', dpi=300)
"""

#查看关键点
print("关键点数目:", len(kp1))
for i in range(2):
    print("关键点", i)
    print("数据类型", type(kp1[i]))
    print("关键点坐标", kp1[i].pt)
    print("领域直径:", kp1[i].size)
    print("方向:", kp1[i].angle)
    print("所在的图像金字塔的组:", kp1[i].octave)
    print("================")
#  查看描述
print("描述的shape:", des1.shape)
for i in range(2):
    print("描述", i)
    print(des1[i])

"""
    DMatch
    queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。 对应kp1
    trainIdx：样本图像的特征点描述符下标,同时也是描述符对应特征点的下标。 对应kp2
    distance：代表这怡翠匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
"""

"""
ratio = 0.85
# 计算匹配点匹配时间
#  K近邻算法求取在空间中距离最近的K个数据点，并将这些数据点归为一类
matcher = cv2.BFMatcher()
raw_matches = matcher.knnMatch(des1, des2, k = 2) # k 代表欧式距离 返回两个人DMatch
# tuple
good_matches = []
bad_matches = []
for m1, m2 in raw_matches:
    #  如果最接近和次接近的比值大于一个既定的值，那么我们保留这个最接近的值，认为它和其匹配的点为good_match
    if m1.distance < ratio * m2.distance:
        good_matches.append([m1])
    else: #将没有匹配的节点显示出来，并且将节点的index统计形成新的kp3
        #bad_matches.append(kp2[m1.trainIdx])
        bad_matches.append([m1])

kp3 = tuple(bad_matches) #将list转换为tuple
print(type(kp3))
"""

"""
kp_image3 = cv2.drawKeypoints(image2, kp3, None) #在测试图像上查找没有匹配到的节点
plt.figure()
plt.imshow(kp_image3)
plt.savefig('kp_image4_1.jpg', dpi=300)
"""

"""
#matches = cv2.drawMatchesKnn(image1, kp1, image2, kp2, good_matches, None, flags = 2)
mismatches = cv2.drawMatchesKnn(image1, kp1, image2, kp2, bad_matches, None, flags = 2)
"""
"""
for i in range(0, len(bad_matches)):
    kp3.
"""

"""
plt.figure()
plt.imshow(mismatches)
plt.savefig('mismatches_small.jpg', dpi = 300)
"""

"""
print("匹配对的数目:", len(good_matches))
for i in range(2):
    print("匹配", i)
    print("数据类型:", type(good_matches[i][0]))
    print("描述符之间的距离:", good_matches[i][0].distance)
    print("查询图像中描述符的索引:", good_matches[i][0].queryIdx)
    print("目标图像中描述符的索引:", good_matches[i][0].trainIdx)
    print("================")
"""

