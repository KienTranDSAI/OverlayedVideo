import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
# Define color range for green
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])



processedTime = []

cap = cv2.VideoCapture('/home/kientran/Code/Work/Overlayed video/hvRvskYqSe_ee2167cfd7714afb8fc3373329a28816.mp4')


if not cap.isOpened():
    print("Error opening video file")


cap2 = cv2.VideoCapture('/home/kientran/Code/Work/Overlayed video/baseVideo_15.mp4')
output = []
#Create output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('warpperspective_add.mp4', fourcc, 30.0, (int(cap2.get(3)), int(cap2.get(4))))

base_height = cap2.get(3)
base_width = cap2.get(4)


ratio1 = (0.68923821039903, 0.44731182795699)
ratio2 = (0.99879081015719, 0.44731182795699)
ratio3 = (0.99879081015719, 0.99784946236559)
ratio4 = (0.68923821039903,0.99784946236559)

corner1 = [int(base_height * ratio1[0]), int(base_width*ratio1[1])]
corner2 = [int(base_height * ratio2[0]), int(base_width*ratio2[1])]
corner3 = [int(base_height * ratio3[0]), int(base_width*ratio3[1])]
corner4 = [int(base_height * ratio4[0]), int(base_width*ratio4[1])]

pts1 = np.float32([[0,0], [cap.get(3), 0], [cap.get(3), cap.get(4)], [0, cap.get(4)]])
pts2 = np.float32([[corner1], [corner2], [corner3], [corner4]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)

while cap.isOpened() and cap2.isOpened():
    
    ret, frame = cap.read()
    ret1, frame1 = cap2.read()
    
    # start = time.time()
    
    if ret and ret1:

        portrait = cv2.warpPerspective(frame, matrix, (1920, 1080))
        

    #     # Create mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        # baseMask = mask
        mask = cv2.bitwise_not(mask)

        resized_mask = cv2.warpPerspective(mask, matrix, (1920, 1080))
        # resizedBaseMask = cv2.warpPerspective(baseMask, matrix, (1920, 1080))
        resizedBaseMask = cv2.bitwise_not(resized_mask)

        masked_frame = cv2.bitwise_and(frame, frame, mask = mask)
        resizedFrame = cv2.warpPerspective(masked_frame, matrix, (1920, 1080))
        resizedBase = cv2.bitwise_and(frame1, frame1, mask = resizedBaseMask)

        center = [int((corner1[0] + corner3[0])/2),int((corner1[1] + corner3[1])/2)]
        # result = cv2.seamlessClone(portrait, frame1, resized_mask, center, cv2.NORMAL_CLONE)
        result = cv2.add(resizedFrame, resizedBase)
        cv2.imshow("mask", result)

    #     # Write the frame to the output video
        out.write(result)

    #     # cv2.imshow("Frame", frame1)

        output.append(result)

    #     end = time.time()
        
    #     processedTime.append(end - start)


        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cap2.release()

out.release()

cv2.destroyAllWindows()

# print(processedTime)
# print(len(processedTime))

# plt.plot(processedTime)
# plt.savefig("ProcessedTime")
# plt.show()



