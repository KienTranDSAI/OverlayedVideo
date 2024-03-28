import cv2
import numpy as np

# Load the image
image = cv2.imread("image.png")

# Define the source polygon vertices
src_points = np.array([[100, 50], [200, 50], [250, 150], [150, 200]], dtype=np.float32)

# Define the destination polygon vertices
dst_points = np.array([[100, 50], [250, 50], [250, 200], [100, 200]], dtype=np.float32)

# Calculate the perspective transformation matrix
M = cv2.getPerspectiveTransform(src_points, dst_points)
print(M.shape)
# Apply the perspective transformation to the entire image
result = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

# Display the result
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
