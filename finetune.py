from keras.models import Model
from keras.layers import Dense
from keras.utils.np_utils import *
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121,DenseNet169,DenseNet201
from keras.applications.resnet import ResNet101, ResNet152
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.optimizers import SGD
import numpy as np
import os
from keras.callbacks import TensorBoard, ReduceLROnPlateau


def finetune(nn_model):
    if nn_model == 'resnet50':
        base_model = ResNet50(include_top=False, weights='imagenet', pooling='avg')
    elif nn_model == 'resnet101':
        base_model = ResNet101(include_top=False, weights='imagenet', pooling='avg')
    elif nn_model == 'resnet152':
        base_model = ResNet152(include_top=False, weights='imagenet', pooling='avg')
    elif nn_model == 'densenet121':
        base_model = DenseNet121(include_top=False, weights='imagenet', pooling='avg')
    elif nn_model == 'densenet169':
        base_model = DenseNet169(include_top=False, weights='imagenet', pooling='avg')
    elif nn_model == 'densenet201':
        base_model = DenseNet201(include_top=False, weights='imagenet', pooling='avg')
    else:
        raise NotImplementedError("The NN model is not implemented!")

    save_dir = './finetune/' + nn_model

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    epochs = 50
    batch_size = 8
    learning_rate = 0.01
    decay_rate = 0
    momentum = 0.9
    train_dir = './dataset/img/train/'
    test_dir = './dataset/img/test/'


    train_imgs = []
    train_labels = []
    test_imgs = []
    test_labels= []
    train_img_names = os.listdir(train_dir)
    test_img_names = os.listdir(test_dir)
    for i in range(len(train_img_names)):
        train_img = load_img(os.path.join(train_dir, train_img_names[i]))
        train_img = img_to_array(train_img)
        train_imgs.append(train_img)
        train_class = train_img_names[i].split('-')[0]
        train_labels.append(train_class)
    for i in range(len(test_img_names)):
        test_img = load_img(os.path.join(test_dir, test_img_names[i]))
        test_img = img_to_array(test_img)
        test_imgs.append(test_img)
        test_class = test_img_names[i].split('-')[0]
        test_labels.append(test_class)
    train_imgs = np.array(train_imgs)
    test_imgs = np.array(test_imgs)

    # encode the string label to integer
    organs = ['bladder','bowel','gallbladder','kidney','liver','spleen']
    mapping = {}
    for i in range(len(organs)):
        mapping[organs[i]] = i
    for i in range(len(train_labels)):
        train_labels[i] = mapping[train_labels[i]]
    for i in range(len(test_labels)):
        test_labels[i] = mapping[test_labels[i]]

    # convert to one-hot vector
    train_labels = to_categorical(train_labels, 6)
    test_labels = to_categorical(test_labels, 6)

    # SGD optimizer
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    # Learning Rate Scheduler
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=3, min_lr=0)

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=30,
        data_format='channels_last',
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        data_format='channels_last')

    train_generator = train_datagen.flow(train_imgs, train_labels, batch_size=batch_size)
    image_numbers = len(train_imgs)

    validation_generator = test_datagen.flow(test_imgs, test_labels, batch_size=batch_size)

    predictions = Dense(6, activation='softmax')(base_model.output)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])
    tensorboard = TensorBoard(log_dir=os.path.join(save_dir, 'logs_50_epoch'),
                                  histogram_freq=0,
                                  write_graph=True,
                                  write_images=True)
    history = model.fit_generator(train_generator, steps_per_epoch=image_numbers // batch_size,
                                      epochs=epochs,
                                      validation_data=validation_generator,
                                      validation_steps=batch_size,
                                      callbacks=[tensorboard, reduce_lr])

    model.save_weights(os.path.join(save_dir, 'finetune_weights_50_epoch.h5'))

    history_dict = history.history

    loss = history.history['loss']
    np_loss = np.array(loss)
    np.savetxt(os.path.join(save_dir,'train_loss_50_epoch.txt'), np_loss, fmt='%.3f')

    val_loss = history.history['val_loss']
    np_val_loss = np.array(val_loss)
    np.savetxt(os.path.join(save_dir,'val_loss_50_epoch.txt'), np_val_loss, fmt='%.3f')

    accy = history.history['acc']
    np_accy = np.array(accy)
    np.savetxt(os.path.join(save_dir, 'train_acc_50_epoch.txt'), np_accy, fmt='%.3f')

    val_accy = history.history['val_acc']
    np_val_accy = np.array(accy)
    np.savetxt(os.path.join(save_dir, 'val_acc_50_epoch.txt'), np_val_accy, fmt='%.3f')

