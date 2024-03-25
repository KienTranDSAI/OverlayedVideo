import cv2
import numpy as np


capSilence = cv2.VideoCapture("simon_silence.mp4")
capTalk = cv2.VideoCapture("simon_talk.mp4")

height = int(capSilence.get(3))
width = int(capSilence.get(4))

if not capSilence.isOpened():
    print("Error with opening silence video!")
if not capTalk.isOpened():
    print("Error with opening talk video!")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (int(capSilence.get(3)), int(capSilence.get(4))))

while capTalk.isOpened():

    retTalk, frameTalk = capTalk.read()
    retSilence, frameSilence = capSilence.read()

    
    if retTalk:

        talkMask = np.zeros(frameTalk.shape[:2], dtype='uint8')
        points = np.array([[130,285], [370,285], [370,420], [130,420]])
        cv2.fillPoly(talkMask, pts=[points], color=(255,255,255))
        silenceMask = cv2.bitwise_not(talkMask)

        masked_silence = cv2.bitwise_and(frameSilence, frameSilence, mask = silenceMask)
        masked_talk = cv2.bitwise_and(frameTalk, frameTalk, mask = talkMask)
        result = cv2.add(masked_silence, masked_talk)
        # cv2.imshow("Talk",frameTalk)
        cv2.imshow("Silence", result)

        out.write(result)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break



capSilence.release()
capTalk.release()
out.release()


# pts1 = np.float32([[130,285], [370,285], [370,420], [130,420]])
# pts1 = np.float32([[0,0], [0,10], [20,10], [20,0]])
# pts2 = np.float32([[130,285], [370,285], [370,420], [130,420]])
# matrix = cv2.getPerspectiveTransform(pts1, pts2)

# resizedFrame = cv2.warpPerspective(frameTalk, matrix, (512, 512))

cv2.imshow("Silence",mask)
cv2.waitKey(0) 
  
cv2.destroyAllWindows() 