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
