import cv2
import numpy as np

# Create two images (blue and green rectangles on black background)
blue_rect = np.zeros((300, 300, 3), dtype=np.uint8)
green_rect = np.zeros((300, 300, 3), dtype=np.uint8)

blue_rect[:, :150] = (255, 0, 0)  # Blue rectangle
green_rect[:, 150:] = (0, 255, 0)  # Green rectangle

# Use addWeighted to blend the two images
alpha = 0.5
beta = 0
gamma = 0
blended = cv2.addWeighted(blue_rect, alpha, green_rect, beta, gamma)

# Display the images and the blended result
cv2.imshow("Blue Rectangle", blue_rect)
cv2.imshow("Green Rectangle", green_rect)
cv2.imshow("Blended", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()
