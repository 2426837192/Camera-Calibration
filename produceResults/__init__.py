#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


def create3DSquare(cameraMatrix, distCoeffs, rvec, tvec, size=[10, 7], sideLength=32):
    # 根据calibrateCamera获得的相机参数，生成一幅3D棋盘格
    # size参数为棋盘格的尺寸，sideLength参数为棋盘格中小正方形的物理尺寸
    image = np.ones([3024, 4032], np.uint8)
    white = [255, 255, 255]
    black = [0, 0, 0]

    # 由于opencv默认的原点不是真正棋盘格的端点，故真实棋盘格的坐标要从-1开始
    objp = np.zeros(((size[0] + 1) * (size[1] + 1), 3), np.float32)
    objp[:, :2] = (np.mgrid[-1:size[0], -1:size[1]] * sideLength).T.reshape(-1, 2)

    # 棋盘格边界的世界坐标
    platePoints3D = np.array([[-2, -2, 0], [size[0], -2, 0], [size[0], size[1], 0],
                              [-2, size[1], 0]], np.float32) * sideLength

    # 将3D点反投影为2D点
    squarePoints = cv2.projectPoints(objp, rvec, tvec, cameraMatrix, distCoeffs=None)[0]
    platePoints2D = cv2.projectPoints(platePoints3D, rvec, tvec, cameraMatrix, distCoeffs=None)[0]

    platePoints = np.array(platePoints2D, np.int32)  # 棋盘格轮廓的四个顶点
    cv2.fillConvexPoly(image, platePoints, color=white)  # 画棋盘格的轮廓

    # 开始画棋盘格
    flag = -1
    for i in range(size[1]):
        flag = 0 - flag
        for j in range(size[0]):
            if flag > 0:
                color = black
            else:
                color = white

            polyTopLeft = squarePoints[i * (size[0] + 1) + j][0]  # 左上角坐标
            polyTopRight = squarePoints[i * (size[0] + 1) + j + 1][0]  # 右上角坐标
            polyDownRight = squarePoints[(i + 1) * (size[0] + 1) + j + 1][0]  # 右下角坐标
            polyDownLeft = squarePoints[(i + 1) * (size[0] + 1) + j][0]  # 左下角坐标

            points = np.array([polyTopLeft, polyTopRight, polyDownRight, polyDownLeft], np.int32)

            cv2.fillConvexPoly(image, points, color)  # 画实心四边形
            flag = 0 - flag

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def draw3DBox(image, cameraMatrix, distCoeffs, rvec, tvec, sideLength=32, boxSide=1):
    # 在图像上画3DBox
    # BoxSide参数为所绘制的立方体边长

    # 立方体的世界坐标，每四个点构成一个面，共有六个面
    boxPoints3D = [[[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
                   [[0, 0, 0], [0, 1, 0], [0, 1, -1], [0, 0, -1]],
                   [[0, 0, 0], [1, 0, 0], [1, 0, -1], [0, 0, -1]],
                   [[1, 0, 0], [1, 1, 0], [1, 1, -1], [1, 0, -1]],
                   [[0, 1, 0], [1, 1, 0], [1, 1, -1], [0, 1, -1]],
                   [[0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]]]
    boxPoints3D = np.array(boxPoints3D, np.float32) * boxSide * sideLength

    # 逐个面画Box
    for i in range(6):
        points2D_eachFace = cv2.projectPoints(boxPoints3D[i], rvec, tvec, cameraMatrix, distCoeffs=None)[0]
        points2D_eachFace = np.array(points2D_eachFace, np.int32)

        cv2.polylines(image, [points2D_eachFace], True, color=[255, 255, 0], thickness=4)


def drawAxes(image, cameraMatrix, distCoeffs, rvec, tvec, sideLength=32, axesLength=3):
    # 在图像上画3D坐标轴
    # axesLength为坐标轴长度参数

    # 各坐标系端点的坐标
    endPoints = [[[0, 0, 0], [0, 1, 0]], [[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 0, -1]]]
    endPoints = np.array(endPoints, np.float32) * axesLength * sideLength

    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    texts = ['X', 'Y', 'Z']

    # 画三个坐标系
    for i in range(3):
        endPoint = cv2.projectPoints(endPoints[i], rvec, tvec, cameraMatrix, distCoeffs=None)[0]
        endPoint = np.array(endPoint, np.int32)

        cv2.polylines(image, [endPoint], False, colors[i], thickness=10)
        cv2.putText(image, texts[i], tuple(endPoint[1][0]), cv2.FONT_HERSHEY_COMPLEX, 4, colors[i], 4)


def writeParamsToTXT(ret, cameraMatrix, distCoeffs, rvecs, tvecs, imageNumbers=0):
    # ret, cameraMatrix, distCoeffs, rvec, tvec为相机参数，imageNumbers是用作标定的照片数量
    if imageNumbers == 0:
        print('imageNumbers参数不能为0！')
        return

    params_file = r'result\params\params.txt'  # 带有模型参数的txt文件
    with open(params_file, 'w') as result_params:
        # 写入内参数矩阵
        result_params.writelines('Camera matrix A:\n' + str(cameraMatrix) + '\n\n')
        # 写入畸变参数
        result_params.writelines('Distortion coefficients (k1,k2,p1,p2,k3):\n' + str(distCoeffs) + '\n\n')
        for i in range(imageNumbers):
            rotation_matrix = cv2.Rodrigues(src=rvecs[i])[0]  # 将旋转向量转化为旋转矩阵
            # 写入旋转矩阵
            result_params.writelines('Rotating matrix of image#' + str(i + 1).zfill(2) + ':\n' + str(
                rotation_matrix) + '\n')
            # 写入平移向量
            result_params.writelines(
                'Translation vector of image#' + str(i + 1).zfill(2) + ':\n' + str(tvecs[i]) + '\n\n')


def analyseAccuracy(corners3D, corners2D, cameraMatrix, distCoeffs, rvec, tvec):
    # 传入一张图片的2D角点和3D角点，返回重投影角点和2D角点之间的平均欧氏距离
    if (len(corners2D) != len(corners3D)):
        # 判断两种维度的点个数是否相同
        return None
    else:
        pointsNumber = len(corners2D)
        reprojectionPoints = cv2.projectPoints(corners3D, rvec, tvec, cameraMatrix, distCoeffs)[0]

        aveEuclideanDistance = 0
        for i in range(pointsNumber):
            distance = math.sqrt(
                (reprojectionPoints[i][0][0] - corners2D[i][0][0]) ** 2 + (
                            reprojectionPoints[i][0][1] - corners2D[i][0][1]) ** 2)

            aveEuclideanDistance += distance / pointsNumber

        return aveEuclideanDistance


def plotAccracy(obj_points, img_points, cameraMatrix, distCoeffs, rvecs, tvecs):
    photoNumber = len(obj_points)

    deviation=[]
    for i in range(photoNumber):
        aveEuclideanDistance = analyseAccuracy(obj_points[i], img_points[i], cameraMatrix, distCoeffs, rvecs[i],
                                               tvecs[i])

        if aveEuclideanDistance:
            deviation.append(aveEuclideanDistance)
        else:
            print('error\n')

    photos = []
    for i in range(photoNumber):
        photos.append('P' + str(i + 1))

    plt.style.use('ggplot')
    fig=plt.figure()
    ax1=fig.add_subplot(1,1,1)

    ax1.bar(range(photoNumber),deviation,align='center',color='peru')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')

    plt.xticks(range(photoNumber),photos,rotation=0,fontsize='small')
    plt.xlabel('Photos')
    plt.ylabel('Average Devation')
    plt.savefig('result/analyse/devations.png',dpi=800,bbox_inches='tight')