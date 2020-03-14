#!/usr/bin/env python
# -*- coding: utf-8 -*-


import glob
# 以下为自己定义的Package
from calibrate import *
from produceResults import *


def produceResults(ret, cameraMatrix, distCoeffs, rvecs, tvecs, imageNumbers=0, sideLength=32):
    # 产生结果方法
    # 绘制出3D棋盘格，并在其上绘制坐标系和3DBox
    # 将参数写入txt文件中

    for i in range(imageNumbers):
        image = create3DSquare(cameraMatrix, distCoeffs, rvecs[i], tvecs[i], sideLength=sideLength)  # 根据每幅图的参数生成3D棋盘格
        draw3DBox(image, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], sideLength)  # 在3D棋盘格上画Box
        drawAxes(image, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], sideLength)

        cv2.imwrite(r'result\images\yuan_zhong_' + str(i + 1).zfill(2) + '.jpg', image)  # 写入文件

    writeParamsToTXT(ret, cameraMatrix, distCoeffs, rvecs, tvecs, imageNumbers)  # 将参数写入txt文件


def calibrate(image_files):
    # 对输入的一组图片完成角点检测和标定工作
    obj_points, img_points, size, successfulImages = detectCorners(image_files, nRow, nCol, sideLength)  # 角点检测
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)  # 标定

    return ret, mtx, dist, rvecs, tvecs, successfulImages,obj_points,img_points


if __name__ == '__main__':
    # 所用相机：iphone XR
    # 所用的棋盘格规格为10x7
    # 棋盘格来源：http://cvrs.whu.edu.cn/courses/ComputerVision2015/camera-calibration-checker-board_9x7.pdf
    nRow = 7
    nCol = 10
    sideLength = 32  # 棋盘格方块在真实世界的尺寸，单位为mm

    # 载入所有图片路径
    image_files = glob.glob(r'picturesForCalibrate\*.jpg')
    imageNumbers = len(image_files)

    print('Start to process input images...')
    ret, mtx, dist, rvecs, tvecs, successfulImages,objPoints,imgPoints = calibrate(image_files)  # 图片处理
    print('{} images are processeed successfully, {} fail.'.format(successfulImages, imageNumbers - successfulImages))
    print('-------------------------------------')
    plotAccracy(objPoints,imgPoints,mtx,dist,rvecs,tvecs)

    print('Start to generate 3D results...')
    produceResults(ret, mtx, dist, rvecs, tvecs, successfulImages, sideLength)  # 结果输出
    print('Results are created successfully!')
