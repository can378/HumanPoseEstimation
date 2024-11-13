from typing import Sequence
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split

neutral_df = pd.read_csv("neutral.txt")
walking_df=pd.read_csv("walking.txt")
running_df=pd.read_csv("running.txt")
crouch_df=pd.read_csv("crouch.txt")
crouchWalk_df=pd.read_csv("crouchWalk.txt")
jumping_df=pd.read_csv("jumping.txt")

X = []
y = []
no_of_timesteps = 20

#NEUTRAL####################################
datasets = neutral_df.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(0)

#WALKING####################################
datasets = walking_df.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(1)

#RUNNING####################################
datasets = running_df.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(2) 

#CROUCH#####################################
datasets = crouch_df.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(3)  

#CROUCH+WALK#################################
datasets = crouchWalk_df.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(4)  

#JUMPING#####################################
datasets = jumping_df.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(5)  




X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=4, activation="softmax"))  

model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

model.save("lstm-hand-grasping.h5")
