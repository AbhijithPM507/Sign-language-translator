import cv2
import pickle
import mediapipe as mp

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

with open('model_1.pkl','rb') as f:
    model=pickle.load(f)

mHands=mp.solutions.hands
hands=mHands.Hands(max_num_hands=1)
mDraw=mp.solutions.drawing_utils

while True:
    ret,frame=cap.read()
    imgRgb=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    results=hands.process(imgRgb)
    
    if results.multi_hand_landmarks:
        for lmHands in results.multi_hand_landmarks:
            mDraw.draw_landmarks(frame,lmHands,mHands.HAND_CONNECTIONS)
            
            landmarks=[]
            for lm in lmHands.landmark:
                landmarks.extend([lm.x,lm.y])
            
            pred=model.predict([landmarks])[0]
            cv2.putText(frame, f"Pred: {pred}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                        2, (0, 255, 0), 3)
        
    cv2.imshow('Sign language translator',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()            
        