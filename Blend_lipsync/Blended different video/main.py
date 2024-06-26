from landmarks_detection import *
import cv2
import numpy as np

cap = cv2.VideoCapture("cutVideo.mp4")

baseCap = cv2.VideoCapture("../simon_silence.mp4")
width = int(baseCap.get(3))
height = int(baseCap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/home/kientran/Code/Work/Overlayed video/Blend_lipsync/Blended different video/output.mp4', fourcc, 30.0, (int(baseCap.get(3)), int(baseCap.get(4))))


landmarks_detector = create_landmarks_dectection_model()


while cap.isOpened() and baseCap.isOpened():

    insertRet, insertFrame = cap.read()
    insertFrame = cv2.resize(insertFrame, (width, height), interpolation=cv2.INTER_AREA)
    # cv2.imshow("asd", insertFrame)
    baseRet, baseFrame = baseCap.read()


    
    if insertRet and baseRet:

        basePolyPoint = []
        insertPolyPoint = []

        mouth_points = [212,186,92,165,167,164,393,391,322,410,432,273,335,406,313,18,83,182,106,43]

        
        insertLandmarks = get_landmarks(landmarks_detector, insertFrame)
        baseLandmarks = get_landmarks(landmarks_detector, baseFrame)

        insertMask = np.zeros(baseFrame.shape[:2], dtype='uint8')
        baseMask = np.zeros(baseFrame.shape[:2], dtype='uint8')
        # mask = np.zeros((height, width), dtype="uint8")

        for point in mouth_points:
            inser_x,insert_y = insertLandmarks[0][point].x, insertLandmarks[0][point].y
            inser_x = int(inser_x*width)
            insert_y = int(insert_y*height)
            insertPolyPoint.append([inser_x,insert_y])
            
            

            base_x, base_y = baseLandmarks[0][point].x,baseLandmarks[0][point].y
            base_x = int(base_x*width)
            base_y = int(base_y * height)
            basePolyPoint.append([base_x, base_y])

            

        insertPoint = np.array(insertPolyPoint,dtype=np.int32)
        cv2.fillPoly(insertMask, pts=[insertPoint], color=(255,255,255))
        basePoint = np.array(basePolyPoint, dtype=np.int32)
        cv2.fillPoly(baseMask, pts = [basePoint], color = (255,255,255))

        pts1 = np.float32([insertPolyPoint[0],insertPolyPoint[5],insertPolyPoint[10],insertPolyPoint[15] ])
        pts2 = np.float32([basePolyPoint[0],basePolyPoint[5],basePolyPoint[10],basePolyPoint[15]])
        
        # print(pts1)
        # print(pts2)
        
        # break
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        iMask = cv2.warpPerspective(insertMask, matrix, (width, height))
        
        iMask[iMask>1] = 255
        iMask[iMask <=10] = 0
        # test = cv2.warpPerspective(insertFrame, insertFrame, (width, height))
        for i in range(insertFrame.shape[2]):
            insertFrame[:,:,i] = cv2.warpPerspective(insertFrame[:,:,i], matrix, (width, height))
        
        
        insertMask = cv2.bitwise_and(iMask, baseMask)
        # insertMask = iMask * baseMask
        # print(np.unique(insertMask))
        # break
        # insertMask = iMask * baseMask
        # baseMask = cv2.bitwise_not(baseMask)
        
        baseMask = cv2.bitwise_not(insertMask)

        maskedBase = cv2.bitwise_and(baseFrame, baseFrame, mask = baseMask)
        maskInsert = cv2.bitwise_and(insertFrame,insertFrame, mask = insertMask)

        a = cv2.add(maskedBase, maskInsert)
        # print(np.unique(maskInsert))
        # print(maskInsert)
        # break
        b = cv2.bitwise_and(a, maskedBase)
        # a = cv2.addWeighted(maskedBase,0.5, insertFrame,0.5,0)
        # cv2.imshow("mask", insertMask)
        # cv2.imshow("basemask", baseMask)
        # cv2.imshow("fa", insertFrame)
        
        out.write(a)
        cv2.imshow("a", a)
        # cv2.imshow("real", c)
        # cv2.imshow("base", insertFrame)



        
        

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release() 
out.release()
cv2.destroyAllWindows() 
