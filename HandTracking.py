import cv2 as cv
import mediapipe as mp
import time
import numpy as np
import random as rd


#to capture the video, 0 for the webcam 0
cap=cv.VideoCapture(0)

mpHands=mp.solutions.hands

'''
Hands takes parameters which determine detecting and tracking.
The parameters also decide the minimum tracking confidence and minimum detecting confidence.
To use the default values, we don't give any parameters to Hands().


We can have multiple hands too.
'''
hands=mpHands.Hands()

#this method is provided by mediapipe, helps us to draw the landmarks of each hand.
mpDraw=mp.solutions.drawing_utils

pTime=0
cTime=0

b=rd.randrange(0,255,1)
g=rd.randrange(0,255,1)
r=rd.randrange(0,255,1)


while True:
    success,img=cap.read()
    img=cv.flip(img,1)

    ih,iw,ic=img.shape

    blank=np.zeros((img.shape[0],img.shape[1],3),dtype='uint8')
    
    imgRgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    
    #process will process the results, ie, the image, for us.
    results=hands.process(imgRgb)

    #detect if there are multiple hands or any hands at all
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            

            # now we will get the information from each hand.
            # lmList=[]
            # lmList2=[]
            # lmList3=[]
            # lmList4=[]
            # tips=[]
            for id,lm in enumerate(handLms.landmark):
                #get the height, width and channels of our image
                h,w,c=img.shape

                #cx and cy are the center of the positions
                cx,cy=int(lm.x*w),int(lm.y*h)

                # if id%4==0 and id!=0:
                #     lmList.append((cx,cy))
                # if id%4==3:
                #     lmList2.append((cx,cy))
                # if id%4==2:
                #     lmList3.append((cx,cy))
                # if id%4==1:
                #     lmList4.append((cx,cy))

            for id,lm in enumerate(handLms.landmark):
                

                #determining the coordinates of the landmarks and multiplying them with the size of the image
                x,y=int(lm.x*iw),int(lm.y*ih)
                
                cv.putText(img,f'{id}',(x,y),cv.FONT_ITALIC,0.4,(b,g,r),2)



            # #to draw lines between the fingertips
            # for i in range(0,4):
            #     cv.line(blank,lmList[i],lmList[i+1],(4,63,99),5)
            #     cv.line(blank,lmList2[i],lmList2[i+1],(4,63,99),5)
            #     cv.line(blank,lmList3[i],lmList3[i+1],(4,63,99),5)
            #     cv.line(blank,lmList4[i],lmList4[i+1],(4,63,99),5)

                       
            mpDraw.draw_landmarks(blank,handLms,mpHands.HAND_CONNECTIONS,
                                  mpDraw.DrawingSpec(color=(255,255,255),thickness=2,circle_radius=5),
                                  mpDraw.DrawingSpec(color=(0,42,255),thickness=5,circle_radius=5)
                                  )

            # cv.line(img,tips[0],tips[1],(255,255,255),10)
            #calculating the distance between the tips of the thumb and the forefinger


    #calculating and writing the FPS of the video stream on the blank image
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv.putText(blank,'FPS:'+str(int(fps)),(10,70),cv.FONT_HERSHEY_SCRIPT_COMPLEX,3,(255,255,255),3)

    #concatenating the video stream and the blank image side by side
    hori=np.concatenate((img,blank),axis=1)

    cv.imshow("Combo",hori)

    if cv.waitKey(1) & 0xFF==ord('d'):
        break

cap.release()
cv.destroyAllWindows()