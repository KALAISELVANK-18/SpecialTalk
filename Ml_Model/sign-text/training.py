import numpy as np
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Flatten
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2

x = np.load('/content/drive/MyDrive/Sign_Language_Translation/cropped_dataset1.npy')
y = np.load('/content/drive/MyDrive/Sign_Language_Translation/cropped_labels1.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize label encoder
label_encoder = LabelEncoder()

# Fit label encoder on the labels and transform them
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


#model architechture with bidirectional features
model = Sequential([
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Define the ModelCheckpoint callback
checkpoint = ModelCheckpoint(
    filepath='/content/drive/MyDrive/Sign_Language_Translation/cropped_50_checkpoint.keras',
    monitor='val_accuracy',                  
    save_best_only=False,                          
    save_weights_only=False,                       
    mode='auto',                                   
    verbose=0                                       
)



model.fit(x_train, y_train_encoded, epochs=800, batch_size=32, validation_data=(x_test, y_test_encoded),callbacks=[checkpoint])


loss, accuracy = model.evaluate(x_test, y_test_encoded)
print('\nTest Loss:', loss)
print('\nTest Accuracy:', accuracy)

model.save('/content/drive/MyDrive/Sign_Language_Translation/cropped_50.keras')