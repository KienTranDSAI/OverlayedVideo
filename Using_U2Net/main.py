import cv2
from rembg import remove
import matplotlib.pyplot as plt
import time

processedTime = []

cap = cv2.VideoCapture('/home/kientran/Code/Work/Overlayed video/hvRvskYqSe_ee2167cfd7714afb8fc3373329a28816.mp4')
if not cap.isOpened():
    print("Error opening video file")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('myOutput.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    
    start = time.time()
    
    ret, frame = cap.read()

    if ret:
        output_image = remove(frame)

        if output_image.shape[2] == 4:
            output_image = output_image[:, :, :3]

        # cv2.imshow("A",output_image)
        
        end = time.time()
        processedTime.append(end - start)

        out.write(output_image)

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
