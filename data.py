import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

data=[]
labels=[]
alphabet="Z"


mHands=mp.solutions.hands
hands=mHands.Hands(static_image_mode=False,max_num_hands=1)
mDraw=mp.solutions.drawing_utils

num_samples=200
count=0

while count<num_samples:
    s,img=cap.read()
    imgRgb=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    results=hands.process(imgRgb)
    
    if results.multi_hand_landmarks:
        for lmHands in results.multi_hand_landmarks:
            mDraw.draw_landmarks(img,lmHands,mHands.HAND_CONNECTIONS)
            
            landmarks=[]
            for lm in lmHands.landmark:
                landmarks.extend([lm.x,lm.y])
                
            data.append(landmarks)
            labels.append(alphabet)
            count+=1
            
    cv2.imshow('WOW',img)
    if cv2.waitKey(1) and 0xFF==ord('q'):break
    
df=pd.DataFrame(data)
df['labels']=labels
df.to_csv(f"sign_{alphabet}.csv",index=False)
    
cap.release()
cv2.destroyAllWindows()