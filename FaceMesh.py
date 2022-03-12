import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import random as rd

cap=cv.VideoCapture(0)

cTime=pTime=0

#this will help us draw on our faces
mpDraw=mp.solutions.drawing_utils
mpFaceMesh=mp.solutions.face_mesh
faceMesh=mpFaceMesh.FaceMesh(max_num_faces=2)


#while loop to start the streaming
while(True):
    success,img=cap.read()
    img=cv.flip(img,1)

    ih,iw,ic=img.shape

    imgRgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)

    #creating a blank image to draw the projection of the face
    blank=np.zeros((img.shape[0],img.shape[1],3),dtype='uint8')
    results=faceMesh.process(imgRgb)
    

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            #drawing the facial landmarks on the blank image
            mpDraw.draw_landmarks(blank,faceLms,mpFaceMesh.FACE_CONNECTIONS,
            mpDraw.DrawingSpec(thickness=1,circle_radius=2,color=(255,255,255)),
            mpDraw.DrawingSpec(thickness=1,circle_radius=2,color=(0,255,255))
            )               

            for id,lm in enumerate(faceLms.landmark):
                
                #determining the coordinates of the landmarks and multiplying them with the size of the image
                x,y=int(lm.x*iw),int(lm.y*ih)
                
                cv.putText(img,f'{id}',(x,y),cv.FONT_ITALIC,0.3,(0,0,0),1)

    #calculating and writing the FPS of the video stream on the blank image
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv.putText(blank,f'FPS:{int(fps)}',(20,70),cv.FONT_ITALIC,1,(234,243,324),3)

    #concatenating the video stream and the blank image side by side
    hori=np.concatenate((img,blank),axis=1)

    #display the complete concatenated video
    cv.imshow("Stream",hori)

    #press lowercase d to exit the stream/while loop
    if cv.waitKey(1) & 0xFF==ord("d") :
        break

cap.release()
cv.destroyAllWindows()
print(type(id))