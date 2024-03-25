import mediapipe as mp
import cv2
# from google.colab.patches import cv2_imshow
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import matplotlib.pyplot as plt

import math
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2_imshow(img)


BG_COLOR = (0, 0, 0) # gray
MASK_COLOR = (255, 255, 255) # white


base_options = python.BaseOptions(model_asset_path='deeplabv3.tflite')
options = vision.ImageSegmenterOptions(base_options=base_options,
                                       output_category_mask=True)

processedTime = []

cap = cv2.VideoCapture('/home/kientran/Code/Work/Overlayed video/hvRvskYqSe_ee2167cfd7714afb8fc3373329a28816.mp4')
if not cap.isOpened():
    print("Error opening video file")
output = []
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('t.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
  start = time.time()
  ret, frame = cap.read()

  if ret:

    with vision.ImageSegmenter.create_from_options(options) as segmenter:



      image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)


      segmentation_result = segmenter.segment(image)
      category_mask = segmentation_result.category_mask

      image_data = image.numpy_view()
      my_data = image.numpy_view()
      fg_image = np.zeros(image_data.shape, dtype=np.uint8)
      fg_image[:] = MASK_COLOR
      bg_image = np.zeros(image_data.shape, dtype=np.uint8)
      bg_image[:] = BG_COLOR

      condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1
      output_image = np.where(condition, my_data, bg_image)

      # print(f'Segmentation mask of a:')
      # resize_and_show(output_image)
    ##Write out
    end = time.time()
    processedTime.append(end - start)
    out.write(output_image)
    output.append(output_image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
            break
  else:
    break

# if not cap.isOpened():
#   print("Error with opening video")

# while cap.isOpened():
#   ret, frame = cap.read()

#   if ret:
#     cv2.imshow("Frame", frame)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#       break
#   else:
#     break

cap.release()
out.release()

cv2.destroyAllWindows()
print(processedTime)
print(len(processedTime))

plt.plot(processedTime)
plt.savefig("ProcessedTime")
plt.show()