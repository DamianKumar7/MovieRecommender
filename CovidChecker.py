import pandas as pd
df = pd.read_csv('Covid_Dataset.csv')
print("The Shape Of The Data is :",df.shape)
#print(df.head(10))
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
df=df.apply(l.fit_transform).astype(int)
cor=df.corr()
x=df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
y=df.iloc[:,[20]]
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state= 0)
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()

classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu',input_dim=20))
classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu'))
classifier.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid'))