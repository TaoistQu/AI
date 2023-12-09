"""
人脸考勤系统（可以运行在CPU）
1、人脸检测：
2、注册人脸，将人脸特征存储进feature.csv
3、识别人脸，将考勤数据存进attendance.csv
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import dlib
import csv 
import time
from argparse import ArgumentParser
from PIL import Image,ImageDraw,ImageFont

hog_face_detector = dlib.get_frontal_face_detector()
cnn_detector = dlib.cnn_face_detection_model_v1('./weights/mmod_human_face_detector.dat')
haar_face_detector = cv2.CascadeClassifier('./weights/haarcascade_frontalface_default.xml')

#加载关键点检测器
points_detector = dlib.shape_predictor('./weights/shape_predictor_68_face_landmarks.dat')
#加载resnet模型
face_descriptor_extractor = dlib.face_recognition_model_v1('./weights/dlib_face_recognition_resnet_model_v1.dat')

#绘制中文
def cv2AddChineseText(img,text,position,textColor=(0,255,0),textSize=30):
    if (isinstance(img,np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype(
        '../fonts/Songti.ttc',textSize,encoding='utf-8')
    draw.text(position,text,textColor,font=fontStyle)
    return cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

#绘制信息
def drawLeftInfo(frame,fpsText,mode='Reg',detector='haar',person=1, count=1):
    cv2.putText(frame,"FPS "+ str(round(fpsText,2)),(30,50),cv2.FONT_ITALIC,0.8,(0,255,0),2)
    #模式:注册、识别
    cv2.putText(frame,"Mode: "+str(mode),(30,80),cv2.FONT_ITALIC,0.8,(0,255,0),2)

    if mode == "Recog":
        #检测器
        cv2.putText(frame,"Detector: "+str(detector),(30,110),cv2.FONT_ITALIC,0.8,(0,255,0),2)
        #人数
        cv2.putText(frame,"Person: "+str(person),(30,140),cv2.FONT_ITALIC,0.8,(0,255,0),2)
        #总人数
        cv2.putText(frame,"Count: "+str(count),(30,170),cv2.FONT_ITALIC,0.8,(0,255,0),2)


#注册人脸
def faceRegister(faceId=1,userName='瞿雷',interval=3,faceCount=3,resize_w=700,resize_h=400):
    count = 0
    #开始注册时间
    startTime = time.time()

    #视频时间
    frameTime = startTime

    #控制显示打卡成功时间长
    show_time = (startTime -10)

    #打开文件
    f = open('data/feature.csv','a',newline='')
    csv_writer = csv.writer(f)

    cap = cv2.VideoCapture(0)
    while True:
        ret,frame = cap.read()
        frame = cv2.resize(frame,(resize_w,resize_h))
        frame = cv2.flip(frame,1)

        #检测
        face_detection = hog_face_detector(frame,1)
        for face in face_detection:
            points  = points_detector(frame,face)
            #绘制人脸关键点
            for point in points.parts():
                cv2.circle(frame,(point.x,point.y),2,(255,255,0),1)
            #绘制框
            l,t,r,b = face.left(),face.top(),face.right(),face.bottom()
            cv2.rectangle(frame,(l,t),(r,b),(0,255,0),2)

            now = time.time()

            if (now - show_time) < 0.5:
                frame = cv2AddChineseText(frame, "注册成功{count} / {faceCount}".format(count=(count+1),faceCount=faceCount),  (l, b+30), textColor=(0, 255, 0), textSize=40)    
            
            #检查次数
            if count < faceCount:
                if now - startTime > interval:
                    face_descriptor = face_descriptor_extractor.compute_face_descriptor(frame, points)
                    face_descriptor = [f for f in face_descriptor]

                    line = [faceId,userName,face_descriptor]
                    #写入csv
                    csv_writer.writerow(line)
                    #保存照片样本
                    print('人脸注册成功 {count}/{faceCount}, faceId:{faceId},userName={userName}'.format(count=(count+1), faceCount=faceCount,faceId=faceId,userName=userName))
                    frame = cv2AddChineseText(frame, "注册成功{count}/{faceCount}".format(count=(count+1),faceCount=faceCount),(l, b+30), textColor=(0, 255, 0), textSize=4)
                    show_time = time.time()

                    startTime = now
                    count += 1
            else:
                print('人脸注册完毕')
                return
            
            break
        now = time.time()
        fpsText = 1/ (now - frameTime)
        frameTime = now
        #绘制信息
        drawLeftInfo(frame, fpsText,'Register')

        cv2.imshow('Face Attendance Demo: Register', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


    f.close()
    cap.release()
    cv2.destroyAllWindows()


#刷新右边考勤信息

def updateRightInfo(frame,face_info_list,face_img_list):
    pass

def getDlibRect(detector='hog',face=None):
    l,t,r,b = None,None,None,None
    if detector == 'hog':
        l,t,r,b = face.left(),face.top(),face.right(),face.bottom()
    
    if detector == 'cnn':
        l = face.rect.left()
        t = face.rect.top()
        r = face.rect.right()
        b  = face.rect.bottom()
    if detector == 'haar':
        l = face[0]
        t = face[1]
        r = face[0] + face[2]
        b = face[1] + face[3]
    nonnegative = lambda x : x if x >= 0 else 0
    return map(nonnegative,(l,t,r,b))

#获取CSV中信息
def getFeatList():
    print('加载注册的人脸特征')

    feature_list = None
    label_list = []
    name_list = []

    with open('./data/feature.csv','r') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            #重新加载数据
            label_list.append(line[0])
            name_list.append(line[1])
            face_descriptor = eval(line[2])

            face_descriptor = np.asarray(face_descriptor,dtype=np.float64)

            face_descriptor = np.reshape(face_descriptor,(1,-1))

            if feature_list is None:
                feature_list = face_descriptor
            else:
                feature_list = np.concatenate(feature_list, face_descriptor)
    print('特征加载完成')
    return feature_list,label_list,name_list

#人脸识别
def faceRecognize(detector='haar',threshold=0.5,write_video=False,resize_w=700,resize_h=400):
    frameTime = time.time()
    feature_list,label_list,nane_list = getFeatList()

    face_time_dict = {}
    face_info_list = []
    face_img_list = []

    #侦测人数
    person_detect = 0
    #统计人脸数
    face_count = 0
    #控制显示打卡成功的时长
    show_time = (frameTime - 10)

    #考勤记录
    f = open('./data/attendance.csv', 'a')
    csv_writer = csv.writer(f)

    cap = cv2.VideoCapture(0)
    resize_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//2
    resize_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) //2

    videoWriter = cv2.VideoWriter('./record_video/out'+str(time.time())+'.mp4',
                                  cv2.VideoWriter_fourcc(*'MP4V'),15,(resize_w,resize_h))
    while True:
        ret,frame = cap.read()
        frame = cv2.resize(frame,(resize_w,resize_h))
        frame = cv2.flip(frame,1)

        #切换人脸检测器
        if detector == 'hog':
            face_detection = hog_face_detector(frame, 1)
        if detector == 'cnn':
            face_detection = cnn_detector(frame, 1)
        if detector == 'haar':
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            face_detection = haar_face_detector.detectMultiScale(frame,minNeighbors=7,minSize=(100,100))
        
        person_detect = len(face_detection)

        for face in face_detection:
            l,t,r,b = getDlibRect(detector,face)

            face = dlib.rectangle(l,t,r,b)

            points = points_detector(frame,face)

            cv2.rectangle(frame,(l,t),(r,b),(0,255,0),2)
            #人脸区域
            face_crop = frame[t:b,l:r]

            #特征
            face_descriptor = face_descriptor_extractor.compute_face_descriptor(frame, points)
            face_descriptor = [f for f in face_descriptor]
            face_descriptor = np.asarray(face_descriptor,dtype=np.float64)
            face_descriptor = np.reshape(face_descriptor,(1,-1))

            #判断人脸特征与人脸特征的距离
            dist = np.linalg.norm(face_descriptor - feature_list)

            min_index = np.argmin(dist)
            min_dist = dist[min_index]

            predict_name = 'Not recog'

            if min_dist < threshold:
                predict_id = label_list[min_index]
                predict_name = nane_list[min_index]

                #判断是否新增记录：如果一个人距上次检测时间大于3秒，或换人将这条记录新增
                need_insert = False
                now = time.time()

                if predict_name in face_time_dict:
                    if (now - face_time_dict[predict_name] > 3):
                        face_time_dict[predict_name] = now
                        need_insert = True
                    else :
                        need_insert = False
                else :
                    face_time_dict[predict_name] = now
                    need_insert = True
                if (now - show_time) < 1:
                    frame = cv2AddChineseText(frame, "打卡成功", (l, b+30), textColor=(0, 255, 0), textSize=40)
                
                if need_insert:
                    #连续显示打卡成功1s
                    frame = cv2AddChineseText(frame, "打卡成功", (l, b+30), textColor=(0, 255, 0), textSize=40)
                    show_time = time.time()

                    time_local = time.localtime(face_time_dict[predict_name])
                    face_time = time.strftime("%H:%M:%S",time_local)
                    face_time_full = time.strftime("%Y:-%m-%d %H:%M:%S",time_local)

                    #开始位置增加
                    face_info_list.insert(0,[predict_name,face_time])
                    face_img_list.insert(0,face_crop)

                    #写入考勤表
                    line = [predict_id,predict_name,min_dist,face_time_full]
                    csv_writer.writerow(line)

                    face_count += 1
            
            cv2.putText(frame,predict_name,+" "+str(round(min_dist,2)),(l,b+30),
                        cv2.FONT_ITALIC,0.8,(0,255,0),2)
        now = time.time()
        fpsText = 1/ (now - frameTime)
        frameTime = now

        
            








    
