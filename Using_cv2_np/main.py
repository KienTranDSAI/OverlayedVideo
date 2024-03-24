import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
# Define color range for green
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])


def resize_frame(frame, width, height):
    size = (width, height)
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


processedTime = []

cap = cv2.VideoCapture('/home/kientran/Code/Work/Overlayed video/hvRvskYqSe_ee2167cfd7714afb8fc3373329a28816.mp4')


if not cap.isOpened():
    print("Error opening video file")


cap2 = cv2.VideoCapture('/home/kientran/Code/Work/Overlayed video/baseVideo_15.mp4')
output = []
#Create output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/home/kientran/Code/Work/Overlayed video/output.mp4', fourcc, 30.0, (int(cap2.get(3)), int(cap2.get(4))))

while cap.isOpened() and cap2.isOpened():
    
    ret, frame = cap.read()
    ret1, frame1 = cap2.read()

    base_width = frame1.shape[1]
    base_height = frame1.shape[0]
    x1 = 0.68923821039903
    x2 = 0.99879081015719
    y1 = 0.44731182795699
    y2 = 0.99784946236559

    start_x = int(x1 * base_width)
    start_y = int(y1 * base_height)
    width = int((x2 - x1) * base_width)
    height = int((y2 - y1) * base_height)
    
    start = time.time()
    
    if ret and ret1:
        # Display the resulting frame
        resized_frame = resize_frame(frame, width, height)

        # Create mask
        hsv = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.bitwise_not(mask)

        # Apply mask
        masked_frame = cv2.bitwise_and(resized_frame, resized_frame, mask=mask)
        apply_frame = frame1[start_y:start_y + height, start_x:start_x + width, :]
        apply_frame = cv2.bitwise_and(apply_frame, apply_frame, mask = 255 - mask)
        # Update the frame1 with the masked region
        frame1[start_y:start_y + height, start_x:start_x + width, :] = masked_frame + apply_frame

        # Write the frame to the output video
        out.write(frame1)

        # cv2.imshow("Frame", frame1)

        output.append(frame1)

        end = time.time()
        
        processedTime.append(end - start)


        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cap2.release()

out.release()

cv2.destroyAllWindows()

print(processedTime)
print(len(processedTime))

plt.plot(processedTime)
plt.savefig("ProcessedTime")
plt.show()



