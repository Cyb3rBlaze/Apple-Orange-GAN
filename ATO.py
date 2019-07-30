import tensorflow as tf
from PIL import Image
from os import listdir
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


#=============================

def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

def preprocessData(directory):
    appleList = listdir(directory + "/apple")
    orangeList = listdir(directory + "/orange")

    appleImages = []
    orangeImages = []

    for image in appleList:
        appleImages += [normalize(np.array(Image.open(directory + "/apple/" + image)).reshape((1, 512, 512, 3)))]
        orangeImages += [normalize(np.array(Image.open(directory + "/orange/" + image)).reshape((1, 512, 512, 3)))]

    return appleImages, orangeImages

#=============================

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.LeakyReLU())

    return result

#=============================

def generator():
    down_list = [
        downsample(64, 4, apply_batchnorm=False), #(256, 256, 64)
        downsample(128, 4), #(128, 128, 128)
        downsample(256, 4), #(64, 64, 256)
        downsample(512, 4), #(32, 32, 512)
        downsample(512, 4), #(16, 16, 512)
        downsample(512, 4), #(8, 8, 512)
        downsample(512, 4), #(4, 4, 512)
        downsample(512, 4), #(2, 2, 512)
    ]
    up_list = [
        upsample(512, 4, apply_dropout=True), #(4, 4, 512)
        upsample(512, 4, apply_dropout=True), #(8, 8, 512)
        upsample(512, 4, apply_dropout=True), #(16, 16, 512)
        upsample(512, 4, apply_dropout=True), #(32, 32, 512)
        upsample(256, 4), #(64, 64, 256)
        upsample(128, 4), #(128, 128, 128)
        upsample(64, 4), #(256, 256, 64)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[None,None,3])
    x = inputs

    skips = []
    for down in down_list:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_list, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

#=============================

def discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar]) #(512, 512, 6)

    down1 = downsample(64, 4, False)(x) #(256, 256, 64)
    down2 = downsample(128, 4)(down1) #(128, 128, 128)
    down3 = downsample(256, 4)(down2) #(64, 64, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) #(66, 66, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1) #(63, 63, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) #(65, 65, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2) #(62, 62, 512)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

#=============================

generator = generator()
discriminator = discriminator()

#=============================

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

LAMBDA = 100

def discriminator_loss(real_output, generated_output):
    real = loss(tf.ones_like(real_output), real_output)
    generated = loss(tf.zeros_like(generated_output), generated_output)

    total_loss = real + generated

    return total_loss

def generator_loss(generated_output, gen_output, target):
    gan_loss = loss(tf.ones_like(generated_output), generated_output)

    # MAE
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

#=============================

def saveEpochImage(model, test_input, epoch):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig("./output/" + str(epoch) + ".jpg")


EPOCHS = 150

def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

def train(appleData, orangeData):
    for epoch in range(EPOCHS):
        for apple, orange in tqdm(zip(appleData, orangeData)):
            train_step(apple, orange)

        if epoch % 5 == 0:
            saveEpochImage(generator, appleData[0], epoch)

#=============================

appleData, orangeData = preprocessData("./data")

train(appleData, orangeData)
