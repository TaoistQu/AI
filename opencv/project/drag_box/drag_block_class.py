import cv2
import mediapipe as mp
import time
import math

class SquareManager:
    def __init__(self,rect_width,rect_height):
        self.rect_width = rect_width
        self.rect_height = rect_height
        
        self.square_count = 0
        self.rect_left_x_list = []
        self.rect_left_y_list = []
        self.alpha_list = []

        #中指与矩形左上角的距离
        self.L1 = 0
        self.L2 = 0

        self.drag_active = False

        self.active_index = -1

    def create(self,rect_left_x,rect_left_y,alpha=0.4):
        self.rect_left_x_list.append(rect_left_x)
        self.rect_left_y_list.append(rect_left_y)
        self.alpha_list.append(alpha)
        self.square_count += 1

    def display(self, class_obj):
        for i in range(self.square_count):
            x = self.rect_left_x_list[i]
            y = self.rect_left_y_list[i]
            alpha = self.alpha_list[i]
            overlay = class_obj.image.copy()
            
            if (i == self.active_index):
                cv2.rectangle(overlay, (x,y), (x+self.rect_width,y+self.rect_height), (255,0,255), -1)
            else:
                cv2.rectangle(overlay, (x,y), (x+self.rect_width,y+self.rect_height), (255,0,0), -1)
            
            class_obj.image = cv2.addWeighted(overlay,alpha,class_obj.image, 1-alpha,0)

    
    def checkOverlay(self,check_x, check_y):
        for i in range(0,self.square_count):
            x = self.rect_left_x_list[i]
            y = self.rect_left_y_list[i]
            
            if ((check_x > x) and (check_x < x+self.rect_width)) and ((check_y > y )and (check_y < y+self.rect_height)):
                self.active_index = i
                return i
            
        return -1
    
    def setLen(self, check_x,check_y):

        self.L1 = check_x - self.rect_left_x_list[self.active_index]
        self.L2 = check_y - self.rect_left_y_list[self.active_index]

    def updateSquare(self,new_x,new_y):
        self.rect_left_x_list[self.active_index] = new_x - self.L1
        self.rect_left_y_list[self.active_index] = new_y - self.L2
    
class HandCoutrolVolume:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.L1 = 0
        self.L2 = 0

        self.image = None

    def recognize(self):

        fpsTime = time.time()

        cap = cv2.VideoCapture(0)

        resize_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        resize_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        rect_percent_text = 0

        squareManager = SquareManager(80,80)

        for i in range(0,5):
            squareManager.create(100*i+50,200,0.6)
        
        with self.mp_hands.Hands(min_detection_confidence = 0.7,
                                 min_tracking_confidence = 0.5,
                                 max_num_hands = 2) as hands:
            while cap.isOpened():
                success, self.image = cap.read()
                self.image = cv2.resize(self.image, (resize_w, resize_h))
                if not success:
                    print('空帧')
                    continue

                # 提高性能
                self.image.flags.writeable = False
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                self.image = cv2.flip(self.image,1)
                
                results = hands.process(self.image)

                self.image.flags.writeable = True
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        #在画面标注手
                        self.mp_drawing.draw_landmarks(
                            self.image, 
                            hand_landmarks, 
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())
                        
                        landmark_list = []

                        paw_x_list = []
                        paw_y_list = []

                        for landmark_id, finger_axis in enumerate(
                            hand_landmarks.landmark):
                            landmark_list.append([
                                landmark_id,finger_axis.x,finger_axis.y,
                                finger_axis.z])
                            paw_x_list.append(finger_axis.x)
                            paw_y_list.append(finger_axis.y)

                        if landmark_list:
                            ratio_x_to_pixel = lambda x: math.ceil(x*resize_w)
                            ratio_y_to_pixel = lambda y: math.ceil(y*resize_h)

                            paw_left_top_x , paw_right_bottom_x  = map(ratio_x_to_pixel,
                                                                       [min(paw_x_list),max(paw_x_list)])
                            paw_left_top_y, paw_right_bottom_y  = map(ratio_y_to_pixel,
                                                                       [min(paw_y_list),max(paw_y_list)])
                            
                            cv2.rectangle(self.image,(paw_left_top_x,paw_left_top_y),
                                          (paw_right_bottom_x,paw_right_bottom_y),(0,255,0),2)
                            
                            middle_finger_tip = landmark_list[12]
                            middle_finger_tip_x = ratio_x_to_pixel(middle_finger_tip[1])
                            middle_finger_tip_y = ratio_y_to_pixel(middle_finger_tip[2])

                            index_finger_tip = landmark_list[8]
                            index_finger_tip_x = ratio_x_to_pixel(index_finger_tip[1])
                            index_finger_tip_y = ratio_y_to_pixel(index_finger_tip[2])

                            between_finger_tip = (middle_finger_tip_x+ index_finger_tip_x) // 2 , (
                                middle_finger_tip_y+ index_finger_tip_y) // 2
                            

                            thumb_finger_point = (middle_finger_tip_x , middle_finger_tip_y)
                            index_finger_point = (index_finger_tip_x , index_finger_tip_y)

                            circle_func = lambda point: cv2.circle(self.image,point,10,(255,0,255),-1)

                            # self.image = circle_func(thumb_finger_point)
                            # self.image = circle_func(index_finger_point)
                            # self.image = circle_func(between_finger_tip)

                            self.image = cv2.line(self.image,thumb_finger_point,index_finger_point,(255,0,255),5)

                            line_len = math.hypot((index_finger_tip_x - middle_finger_tip_x),
                                                  (index_finger_tip_y - middle_finger_tip_y))
                            
                            rect_percent_text = math.ceil(line_len)

                            if squareManager.drag_active:
                                squareManager.updateSquare(between_finger_tip[0],between_finger_tip[1])
                                if (line_len > 50):
                                    squareManager.drag_active = False
                                    squareManager.active_index = -1
                            elif (line_len < 50) and (squareManager.checkOverlay(between_finger_tip[0],
                                                                                     between_finger_tip[1]) != -1) and (
                                                                                         squareManager.drag_active == False):
                                    
                                squareManager.drag_active = True

                                squareManager.setLen(between_finger_tip[0], between_finger_tip[1])
                    squareManager.display(self)

                    cv2.putText(self.image,"Distance:"+str(rect_percent_text),(10,120),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
                    cv2.putText(self.image,"Active:"+ 
                                ( "None" if squareManager.active_index == -1 else 
                                 str(squareManager.active_index)) , (10,170),
                                 cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
                    
                    cTime = time.time()
                    fps_text = 1/(cTime - fpsTime)
                    fpsTime = cTime

                    cv2.putText(self.image, "FPS:" + str(int(fps_text)), (10,70),
                                 cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
                    
                    cv2.imshow('virtual drag and drop',self.image)

                    if cv2.waitKey(5) & 0xff == 27:
                        break
            cap.release()

control = HandCoutrolVolume()
control.recognize()
                                 
                                 
                          
                                
                            





                                
                                
                                
                        
           
            
            