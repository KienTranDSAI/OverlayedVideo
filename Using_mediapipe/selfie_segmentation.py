import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
import time
import matplotlib.pyplot as plt
import math



BG_COLOR = (0, 0, 0) # gray
MASK_COLOR = (255, 255, 255) # white

def resize_frame(frame, width, height):
    size = (width, height)
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


processedTime = []

cap = cv2.VideoCapture('/home/kientran/Code/Work/Overlayed video/hvRvskYqSe_ee2167cfd7714afb8fc3373329a28816.mp4')
if not cap.isOpened():
    print("Error opening video file")
cap2 = cv2.VideoCapture('/home/kientran/Code/Work/Overlayed video/baseVideo_15.mp4')

output = []
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('newOutput2.mp4', fourcc, 30.0, (int(cap2.get(3)), int(cap2.get(4))))

while cap.isOpened() and cap2.isOpened():
  start = time.time()
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

  if ret and ret1:

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:


        kernel = np.ones((9,9), np.uint8)
        blur_size = (13,13)

        #
        frame = resize_frame(frame, width, height)

        image_height, image_width, _ = frame.shape
        results = selfie_segmentation.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mask = results.segmentation_mask
        mask = cv2.erode(mask, kernel)
        mask = cv2.blur(mask, blur_size, cv2.BORDER_DEFAULT)
        
        condition = np.stack((mask,) * 3, axis=-1) > 0.1

        fg_image = np.zeros(frame.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(frame.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR


        portrait_image = condition * frame
        apply_frame = frame1[start_y:start_y + height, start_x:start_x + width, :]
        apply_frame = apply_frame * (1-condition) 
        frame1[start_y:start_y + height, start_x:start_x + width, :] = portrait_image + apply_frame

        # cv2.imshow("Frame", frame1)
        # output_image = condition * frame
        # output_image = np.where(condition, frame, bg_image)
        
        # output_image = cv2.blur(output_image, blur_size, cv2.BORDER_DEFAULT)
      

      # print(f'Segmentation mask of a:')
      # resize_and_show(output_image)
    ##Write out
    end = time.time()
    processedTime.append(end - start)
    # out.write(output_image)
    # output.append(output_image)
    out.write(frame1)
    output.append(frame1)

    if cv2.waitKey(25) & 0xFF == ord('q'):
            break
  else:
    break


cap.release()
out.release()

cv2.destroyAllWindows()
print(processedTime)
print(len(processedTime))

plt.plot(processedTime)
plt.savefig("ProcessedTime")
plt.show()