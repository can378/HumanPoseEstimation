import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. Data Loading & Preprocessing
# (기존 데이터 로딩 코드는 유지)
neutral_df = pd.read_csv("neutral.txt")
walking_df = pd.read_csv("walking.txt")
running_df = pd.read_csv("running.txt")
crouch_df = pd.read_csv("crouch.txt")
crouchWalk_df = pd.read_csv("crouchWalk.txt")

# 데이터 시퀀스 생성
def create_sequences(df, label, no_of_timesteps=20):
    sequences = []
    labels = []
    dataset = df.iloc[:, 1:].values  # 첫 번째 열 제외
    
    for i in range(no_of_timesteps, len(dataset)):
        sequences.append(dataset[i-no_of_timesteps:i, :])
        labels.append(label)
    
    return sequences, labels

# 각 동작에 대한 시퀀스 생성
X = []
y = []
no_of_timesteps = 20

sequence_data = [
    (neutral_df, 0),
    (walking_df, 1),
    (running_df, 2),
    (crouch_df, 3),
    (crouchWalk_df, 4)
]

for df, label in sequence_data:
    seq, lab = create_sequences(df, label, no_of_timesteps)
    X.extend(seq)
    y.extend(lab)

X = np.array(X)
y = np.array(y)

# 데이터 스케일링
scaler = StandardScaler()
X_reshaped = X.reshape(-1, X.shape[-1])
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(X.shape)

# 데이터 증강
def augment_data(X, y):
    augmented_X = []
    augmented_y = []
    
    for i in range(len(X)):
        # 원본 데이터
        augmented_X.append(X[i])
        augmented_y.append(y[i])
        
        # 노이즈 추가 (작은 노이즈)
        noise = np.random.normal(0, 0.01, X[i].shape)
        augmented_X.append(X[i] + noise)
        augmented_y.append(y[i])
    
    return np.array(augmented_X), np.array(augmented_y)

X, y = augment_data(X, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Model Definition
def create_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        Bidirectional(LSTM(64)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(5, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 3. Training Setup
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.0001
    )
]

# Calculate class weights
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# Create and train model
model = create_model((X.shape[1], X.shape[2]))

history = model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# 4. Evaluation and Visualization
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Evaluate model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Plot results
plot_training_history(history)
plot_confusion_matrix(y_test, y_pred_classes)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# Save model
model.save('improved_lstm_hpe.h5')
