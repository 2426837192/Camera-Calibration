#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np


def detectCorners(image_files, nRow, nCol, sideLength):
    # 角点检测，并返回3D角点坐标、2D角点坐标和size参数

    # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 获取标定板角点的位置
    objp = np.zeros(((nCol - 1) * (nRow - 1), 3), np.float32)
    # 将世界坐标系建在标定板上
    objp[:, : 2] = (np.mgrid[0:(nCol - 1), 0:(nRow - 1)] * sideLength).T.reshape(-1, 2)

    obj_points = []  # 储存真实世界中的3D角点
    img_points = []  # 储存图像平面中的2D角点

    # 对每个图片执行角点检测，画出角点并保存至chessboard_corners文件夹中
    successfulImages = 0
    for image_file in image_files:
        image = cv2.imread(image_file)  # 读入图片
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将RGB图像转为灰度图像
        size = image_gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(image_gray, (9, 6), None)

        if ret:
            obj_points.append(objp)

            # 执行亚像素级角点检测
            corners_subpix = cv2.cornerSubPix(image_gray, corners, (5, 5), (-1, -1), criteria=criteria)
            img_points.append(corners_subpix)

            cv2.drawChessboardCorners(image, (nCol - 1, nRow - 1), corners, ret)
            successfulImages += 1
            cv2.imwrite(r'result\chessboard_corners\imagesWithCorners' + str(successfulImages).zfill(2) + '.jpg', image)

    return obj_points, img_points, size, successfulImages
