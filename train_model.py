import os
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed

# Paths to your train and validation datasets
train_path = r"C:\Users\KARUNA\Desktop\KP\Projects\FacialExpression\archive\train"
validation_path = r"C:\Users\KARUNA\Desktop\KP\Projects\FacialExpression\archive\test"

# ImageDataGenerator for data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Flow training images in batches using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    color_mode='grayscale'
)

# Flow validation images in batches using train_datagen generator
validation_generator = train_datagen.flow_from_directory(
    validation_path,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    color_mode='grayscale'
)

# Model Definition
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model Training
history = model.fit(
    train_generator,
    epochs=150,
    validation_data=validation_generator
)

# Save the model
model.save("my_emotion_model_updated.h5")

# Image Processing Functions
def process_image(input_path):
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    equalized_image = cv2.equalizeHist(image)
    cv2.imwrite(input_path, equalized_image)

def override_noise_reduction(input_folder, kernel_size=(5, 5)):
    for emotion_folder in os.listdir(input_folder):
        emotion_path = os.path.join(input_folder, emotion_folder)

        for filename in os.listdir(emotion_path):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                input_path = os.path.join(emotion_path, filename)

                image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                blurred_image = cv2.GaussianBlur(image, kernel_size, 0)

                cv2.imwrite(input_path, blurred_image)

def override_histogram_equalization_parallel(input_folder, num_workers=4):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for emotion_folder in os.listdir(input_folder):
            emotion_path = os.path.join(input_folder, emotion_folder)

            image_files = [filename for filename in os.listdir(emotion_path) if filename.endswith('.jpg')]

            for filename in image_files:
                input_path = os.path.join(emotion_path, filename)

                futures.append(executor.submit(process_image, input_path))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"error processing image: {e}")
