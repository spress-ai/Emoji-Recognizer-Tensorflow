import os
import time
import numpy as np
from imgaug import augmenters as iaa
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

IMG_SIZE = 256
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001

NUM_CLASSES = 7

TRAIN_SET_DIRECTORY = "dataset/train/"
TEST_SET_DIRECTORY = "dataset/test/"

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

print(tf.__version__)

def load_model():
  print("Loading model...")
  base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')

  global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
  prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")

  model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
  ])

  model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=LEARNING_RATE),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model


def generator(gen):
    aug_pipeline = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.GaussianBlur((0, 3.0))), # Gaussian blur with sigma 0 and 3 to 50% of images

        iaa.OneOf([
            iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
            iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
        ]),
        
        iaa.SomeOf((0, 3),[
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
            iaa.Fliplr(1.0), # horizontally flip
            # iaa.Sometimes(0.6, iaa.Multiply((0.5, 1.5), per_channel=0.5)), # Change 
            iaa.Sometimes(0.5, iaa.CropAndPad(percent=(-0.25, 0.25))), # crop and pad 50% of the images
            iaa.Sometimes(0.5, iaa.Affine(rotate=5)) # rotate 50% of the images
        ])
    ],
    random_order=True # apply the augmentations in random order
    )

    while True:
        images, labels = gen.next()
        labels = np.repeat(labels, 8, 0)
        all_images = np.array([])
        for image in images:
            images_aug = np.array([aug_pipeline.augment_image(image) for _ in range(8)])
            if(len(all_images) == 0):
                all_images = images_aug
            else:
                all_images = np.append(all_images, images_aug, axis = 0)

        yield all_images, labels


def train():
    model = load_model()
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_generator = datagen.flow_from_directory(TRAIN_SET_DIRECTORY, target_size=(IMG_SIZE, IMG_SIZE), batch_size=4, shuffle=True, class_mode='categorical')
    test_generator = datagen.flow_from_directory(TEST_SET_DIRECTORY, target_size=(IMG_SIZE, IMG_SIZE), batch_size=4, class_mode='categorical')

    model.fit_generator(generator(train_generator), 
                        epochs=EPOCHS, 
                        steps_per_epoch=264,
                        validation_data=test_generator,
                        callbacks=[TensorBoard(log_dir='./logs/{}'.format(time.time()))])
    model.save("models/emoji_{}.h5".format(time.time()))

train()