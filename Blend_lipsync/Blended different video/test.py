from landmarks_detection import *
import cv2
import numpy as np

cap = cv2.VideoCapture("/home/kientran/Code/Work/Overlayed video/Blend_lipsync/g4Nglchvdy_584a6855e9d74b8cb36a82873ef72974 (3).mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Frame", frame)
        


        
        

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release() 
cv2.destroyAllWindows() 
