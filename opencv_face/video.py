#! usr/bin/python
#coding=utf-8

# 读取视频，并识别人脸

import cv2


# faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
# faceCascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

vc = cv2.VideoCapture('data/欢乐颂.mp4') #读入视频文件

if vc.isOpened(): #判断是否正常打开
    rval, frame = vc.read()
else:
    rval = False
num = 0
while rval:   #循环读取视频帧
    num = num + 1
    rval, frame = vc.read()
    if rval:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        # 横、纵坐标、宽、高度
        for x, y, w, h in faces:
            # 在人脸的位置上画方框、函数的参数分别为：图像，左上角坐标，右下角坐标，颜色，宽度
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # 展示画方框的人脸图片
        cv2.imshow("c", frame)
        if num % 100 == 0 :
            cv2.imwrite('data/image/'+str(num)+'.jpg', frame)
    cv2.waitKey(1)
vc.release()