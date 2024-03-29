from landmarks_detection import *
import cv2
import numpy as np

cap = cv2.VideoCapture("cutVideo.mp4")

baseCap = cv2.VideoCapture("../simon_silence.mp4")
widthBase = int(baseCap.get(3))
heightBase = int(baseCap.get(4))
widthInsert = int(cap.get(3))
heightInsert = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/home/kientran/Code/Work/Overlayed video/Blend_lipsync/Blended different video/Large2Small.mp4', fourcc, 30.0, (int(baseCap.get(3)), int(baseCap.get(4))))


landmarks_detector = create_landmarks_dectection_model()


while cap.isOpened() and baseCap.isOpened():

    insertRet, insertFrame = cap.read()
    baseRet, baseFrame = baseCap.read()

    # myFrame = insertFrame[:,:,:]
    
    if insertRet and baseRet:

        basePolyPoint = []
        insertPolyPoint = []

        # mouth_points = [212,186,92,165,167,164,393,391,322,410,432,273,335,406,313,18,83,182,106,43]
        mouth_points = [192, 207,206,165,167,164,393,391,426,427,416,364,394,395,369,396,175,171,140,170,169,135]
        
        insertLandmarks = get_landmarks(landmarks_detector, insertFrame)
        baseLandmarks = get_landmarks(landmarks_detector, baseFrame)

        insertMask = np.zeros(insertFrame.shape[:2], dtype='uint8')
        baseMask = np.zeros(baseFrame.shape[:2], dtype='uint8')
        # mask = np.zeros((height, width), dtype="uint8")

        for point in mouth_points:
            inser_x,insert_y = insertLandmarks[0][point].x, insertLandmarks[0][point].y
            inser_x = int(inser_x*widthInsert)
            insert_y = int(insert_y*heightInsert)
            insertPolyPoint.append([inser_x,insert_y])

            # myFrame[insert_y, inser_x] = 0
            # if len(insertPolyPoint) > 1:
            #     pre = insertPolyPoint[-2]
            #     cv2.line(myFrame,(pre[0], pre[1]), (inser_x, insert_y), (0,0,0), 2)
            

            base_x, base_y = baseLandmarks[0][point].x,baseLandmarks[0][point].y
            base_x = int(base_x*widthBase)
            base_y = int(base_y * heightBase)
            basePolyPoint.append([base_x, base_y])

        insertPoint = np.array(insertPolyPoint,dtype=np.int32)
        cv2.fillPoly(insertMask, pts=[insertPoint], color=(255,255,255))
        basePoint = np.array(basePolyPoint, dtype=np.int32)
        cv2.fillPoly(baseMask, pts = [basePoint], color = (255,255,255))

        baseMask = cv2.bitwise_not(baseMask)
        matrix, mask = cv2.findHomography(np.array([insertPolyPoint]), np.array([basePolyPoint]), cv2.RANSAC, 5.0)

        warppedFrame = cv2.warpPerspective(insertFrame, matrix, (widthBase, heightBase))
        warppedMask = cv2.warpPerspective(insertMask, matrix, (widthBase, heightBase))
        finalMask = cv2.bitwise_and(warppedMask, cv2.bitwise_not(baseMask))
        finalMask[finalMask > 1] = 255
        finalMask[finalMask <=1] = 0
        # cv2.imshow("warp",finalMask)
        baseMask = cv2.bitwise_not(finalMask)
        maskedBase = cv2.bitwise_and(baseFrame, baseFrame, mask = baseMask)
        maskedInsert = cv2.bitwise_and(warppedFrame, warppedFrame, mask = finalMask)

        result = cv2.add(maskedBase, maskedInsert)
        out.write(result)
        cv2.imshow("a", result)



        
        

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release() 
out.release()
cv2.destroyAllWindows() 
