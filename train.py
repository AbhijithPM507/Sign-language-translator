import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import train_test_split


files=['sign_A.csv','sign_B.csv','sign_C.csv','sign_D.csv','sign_E.csv','sign_F.csv','sign_G.csv','sign_H.csv','sign_I.csv','sign_J.csv','sign_K.csv','sign_L.csv','sign_M.csv','sign_N.csv','sign_O.csv','sign_P.csv','sign_Q.csv','sign_R.csv','sign_S.csv','sign_T.csv','sign_U.csv','sign_V.csv','sign_W.csv','sign_X.csv','sign_Y.csv','sign_Z.csv',]
df=pd.concat([pd.read_csv(f) for f in files])

# print(df.head())
X=df.drop('labels',axis=1).values
y=df['labels'].values

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=RandomForestClassifier()
model.fit(x_train,y_train)

print('Accuracy: ',model.score(x_test,y_test))

with open('model_1.pkl','wb') as model_file:
    pickle.dump(model,model_file)