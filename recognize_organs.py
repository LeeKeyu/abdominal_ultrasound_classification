import numpy as np
import os
from scipy.spatial.distance import cdist
import time
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Model
from keras.layers import Dense
from keras.utils.np_utils import *
from keras.applications.resnet import ResNet50
from keras.applications.densenet import DenseNet121,DenseNet169,DenseNet201
from keras.applications.resnet import ResNet101, ResNet152
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.resnet import preprocess_input
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from tensorflow.keras.optimizers import SGD
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.metrics import accuracy_score, precision_score,classification_report, confusion_matrix
import seaborn as sn
import cv2


def plot_pred_prob(result, result_dir, error_id):
    fig = plt.figure()
    fig.set_size_inches(5, 4)
    fig_6 = fig.add_subplot('111')
    fig.set_tight_layout(True)
    for n in range(len(result['logits'])):
        if n in error_id:
            print(n)
            fig_6.cla()
            fig.set_tight_layout(True)
            logits = result['logits'][n]
            fig_6.set_xlim(0, 1.1)
            plane_names =  ['bladder', 'bowel', 'gallbladder', 'kidney', 'liver', 'spleen']
            colors = ['red','orange', 'gold', 'green', 'blue', 'purple']
            fig_6.set_xlabel("Predicted Probability", fontsize=20)
            fig_6.set_yticks(range(len(plane_names)))
            fig_6.tick_params(axis='y', labelsize=18)
            fig_6.tick_params(axis='x', labelsize=16)
            fig_6.set_yticklabels(plane_names)
            fig_6.set_xticks(np.arange(0, 1.1, 0.2))
            fig_6.invert_yaxis()
            fig_6.barh(range(len(plane_names)), logits, color=colors)

            step = n
            img_name = "err_" + str(step) + ".jpg"
            result_dir_new = os.path.join(result_dir, "recognition_prob")
            if not os.path.exists(result_dir_new):
                os.makedirs(result_dir_new)
            img = os.path.join(result_dir_new, img_name)
            fig.savefig(img, quality=100, bbox_inches='tight')
    fig1 = plt.figure()
    fig1.set_size_inches(5, 4)
    fig_0 = fig.add_subplot('111')
    fig1.set_tight_layout(True)
    for n in range(len(result['logits'])):
        if n in error_id:
            print(n)
            fig_0.cla()
            fig1.set_tight_layout(True)
            # img = (result['img'][n]).astype('uint8')
            img = result['img'][n]
            img_name = "err_img_" + str(n) + os.path.basename(img)
            img = cv2.imread(img)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            result_dir_new = os.path.join(result_dir, "recognition_prob")
            if not os.path.exists(result_dir_new):
                os.makedirs(result_dir_new)
            img = os.path.join(result_dir_new, img_name)
            fig1.savefig(img, quality=100, bbox_inches='tight')


def recognize_organs_fc(nn_model):
    error_id = []
    if nn_model == 'resnet50':
        base_model = ResNet50(include_top=False, weights=None, pooling='avg')
    elif nn_model == 'resnet101':
        base_model = ResNet101(include_top=False, weights=None, pooling='avg')
    elif nn_model == 'resnet152':
        base_model = ResNet152(include_top=False, weights=None, pooling='avg')
    elif nn_model == 'densenet121':
        base_model = DenseNet121(include_top=False, weights=None, pooling='avg')
    elif nn_model == 'densenet169':
        base_model = DenseNet169(include_top=False, weights=None, pooling='avg')
    elif nn_model == 'densenet201':
        base_model = DenseNet201(include_top=False, weights=None, pooling='avg')
    else:
        raise NotImplementedError("The NN model is not implemented!")
    predictions = Dense(6, activation='softmax')(base_model.output)

    model = Model(inputs=base_model.input, outputs=predictions)
    weight_file = './finetune/' + nn_model + '/finetune_weights_50_epoch.h5'
    assert os.path.exists(weight_file) is True, "Weight path is empty"
    model.load_weights(weight_file, by_name=False)

    # preprocess test data
    test_dir = './dataset/img/test/'
    test_imgs = []
    test_imgs_original = []
    test_labels = []
    test_labels_int = []
    test_img_names = os.listdir(test_dir)
    start = time.clock()
    for i in range(len(test_img_names)):
        img_path = os.path.join(test_dir, test_img_names[i])
        test_imgs_original.append(img_path)
        test_img = load_img(img_path)
        test_img = img_to_array(test_img)
        test_img = preprocess_input(test_img)
        test_imgs.append(test_img)
        test_class = test_img_names[i].split('-')[0]
        test_labels.append(test_class)
    test_imgs = np.array(test_imgs)
    # encode the string label to integer
    organs = ['bladder', 'bowel', 'gallbladder', 'kidney', 'liver', 'spleen']
    mapping = {}
    for i in range(len(organs)):
        mapping[organs[i]] = i
    for i in range(len(test_labels)):
        test_labels_int.append(mapping[test_labels[i]])

    # compile model
    learning_rate = 0.01
    decay_rate = 0
    momentum = 0.9
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

    # predict with model
    test_logits = model.predict(test_imgs)
    test_predictions = np.argmax(test_logits, axis=1)
    num_acc = 0
    end = time.clock()
    total_time = end - start
    print("Average inference time for one image: {}".format(total_time/len(test_imgs)))
    for i in range(len(test_imgs)):
        # print("true: {} predict: {}".format(test_labels_int[i], test_predictions[i]))
        if test_predictions[i] == test_labels_int[i]:
            num_acc += 1
        else:
            error_id.append(i)
    result_dict = {"img": test_imgs_original,
                   "logits": test_logits}
    plot_pred_prob(result_dict, "fc_errors", error_id)

    acc = num_acc / len(test_imgs)
    print("Model: {}, acc: {:.4f}, {}/{} correct.".format(nn_model, acc, num_acc, len(test_imgs)))
    # scores = model.evaluate(test_imgs, test_labels, verbose=0)
    # print("Model: {} Test acc: {:.4f}".format(nn_model, scores[1]))
    # print(classification_report(test_labels_int, test_predictions, target_names=organs, digits=6))
    confusion = confusion_matrix(test_labels_int, test_predictions)
    df_cm = pd.DataFrame(confusion, index=['bladder', 'bowel', 'gallbladder', 'kidney', 'liver', 'spleen'],
                         columns=['bladder', 'bowel', 'gallbladder', 'kidney', 'liver', 'spleen'])
    plt.figure(figsize=(5, 4))
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    cmap = sn.cm.rocket_r
    # cmap = plt.cm.Blues
    ax = sn.heatmap(df_cm, annot=True, fmt='.20g', cmap=cmap)
    ax.set_xlabel("Predicted class", fontsize=12)
    ax.set_ylabel("True class", fontsize=12)
    result_dir = "confusion/{}".format(nn_model)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plt.savefig(
        os.path.join(result_dir, "confusion_matrix_{}_fc.jpg".format(nn_model)),
        quality=100,
        bbox_inches='tight')

def recognize_organs(nn_model, pca=False):
    '''
    Use the k-NN method to classify the abdominal organ in the image
    by comparing distances between features of the test images
    and features of images in the training set
    :return:
    '''
    if pca:
        nn_model = nn_model + '_pca'
    testlist = []
    databaselist = []
    test_class_list = []
    data_class_list = []

    test_dir = './dataset/feature_' + nn_model + '/test'
    train_dir = './dataset/feature_' + nn_model + '/train'
    result_dir = './result/'
    test_imgs = os.listdir(test_dir)
    train_imgs = os.listdir(train_dir)
    # read image feature vectors and the labels
    for i in range(len(test_imgs)):
        test_img = os.path.join(test_dir, test_imgs[i])
        testlist.append(test_img)
        test_class = test_imgs[i].split('-')[0]
        test_class_list.append(test_class)
    for i in range(len(train_imgs)):
        train_img = os.path.join(train_dir,train_imgs[i])
        databaselist.append(train_img)
        train_class = train_imgs[i].split('-')[0]
        data_class_list.append(train_class)


    # k values
    k_list = [1,3,5,7,9]
    distance_list = ['euclidean', 'cityblock', 'canberra', 'cosine']

    # k_list = [3]
    # distance_list = ['cityblock']

    correct_rate_list = dict()
    for dist_category in distance_list:
        correct_rate_list[dist_category] = []
        for k in k_list:
            num_test = len(testlist)
            num_database = len(databaselist)
            testlist_new = []
            test_class_list_new = []
            pred = []
            pred_img = []
            pred_num = []
            dists = np.zeros((num_test, num_database))
            dist_pred = []
            start = time.clock()

            for i in range(num_test):
                # print('image %d: %s' % (i, testlist[i]))
                kclose_list = []
                kclose_img_list = []
                kclose_dist_list = []  # k closest distances
                for j in range(num_database):
                    test_vec = np.load(testlist[i])
                    database_vec = np.load(databaselist[j])
                    dists[i][j] = cdist(test_vec, database_vec, dist_category)

                # find the k nearest neighbors
                dist_k_min = np.argsort(dists[i])[:k]
                pred_num.append(i)
                testlist_new.append(testlist[i])
                test_class_list_new.append(test_class_list[i])

                # k-NN majority vote
                for m in range(k):
                    kclose_list.append(data_class_list[dist_k_min[m]])
                    kclose_img_list.append(databaselist[dist_k_min[m]])
                    kclose_dist_list.append(dists[i][dist_k_min[m]])
                    # print('For %d ,the %d th closest img is %s' % (i, m, kclose_img_list[-1]))
                    # print('with the %d th smallest distance: %f' % (m, kclose_dist_list[-1]))

                # k-NN majority vote
                for n in range(len(kclose_list)):
                    old = kclose_list.count(kclose_list[n])
                    num = max(kclose_list.count(m) for m in kclose_list)
                    if (num == old):
                        pred.append(kclose_list[n])
                        pred_img.append(kclose_img_list[n])
                        dist_pred.append(kclose_dist_list[n])
                        # print('%d---true: %s ,pred: %s' % (i, test_class_list[i], pred[-1]))
                        break


            # calculate the accuracy
            correct = 0
            num_test_new = len(testlist_new)
            error_test_list = []
            error_pred_img = []
            error_num = []
            error_dist = []
            for i in range(num_test_new):

                if (pred[i] == test_class_list_new[i]):
                    correct += 1
                else:
                    # bad case
                    error_num.append(pred_num[i])
                    error_test_list.append(testlist_new[i])
                    error_pred_img.append(pred_img[i])
                    error_dist.append(dist_pred[i])
            for i in range(len(error_test_list)):
                print('%d is incorrect, true img: %s , pred img: %s, distance: %f' % (
                    error_num[i], error_test_list[i], error_pred_img[i], error_dist[i]))

            # copy wrong images to /result
            for file in error_test_list:
                file_name = os.path.split(file)[1]
                file_without_ext = os.path.splitext(file_name)[0]
                file_without_ext = os.path.splitext(file_without_ext)[0]
                # print(file_without_ext)
                raw_img_name = file_without_ext + '.png'
                raw_img = os.path.join('./dataset/img/test', raw_img_name)
                error_savepath = os.path.join('./result/k_{}/error/'.format(k))
                if not os.path.exists(error_savepath):
                    os.makedirs(error_savepath)
                shutil.copy(raw_img, error_savepath)

            correct_rate = correct / num_test_new
            correct_rate_list[dist_category].append(correct_rate)
            print("Current dist:", dist_category)
            print("Current feature extractor:", nn_model)
            print('k = %d, The correct rate is %.2f%%' % (k, correct_rate * 100))
            end = time.clock()
            total_time = end - start
            print("Average inference time for one image: {:.2f}".format(total_time / num_test_new))

    # find the best configuration of k, distance metric for current feature extractor
    max_acc = 0
    best_k = 0
    best_dist = None
    for dist in correct_rate_list:
        for k in range(len(correct_rate_list[dist])):
            if correct_rate_list[dist][k] > max_acc:
                max_acc = correct_rate_list[dist][k]
                best_k = k_list[k]
                best_dist = dist

    print("best acc: ",max_acc,"best distance metric: ",best_dist,"best k: ", best_k)
    #######################################

    # use best k and dist to calculate the confusion matrix
    num_test = len(testlist)
    num_database = len(databaselist)
    testlist_new = []
    test_class_list_new = []
    pred = []
    pred_img = []
    pred_num = []
    dists = np.zeros((num_test, num_database))
    dist_pred = []

    for i in range(num_test):
        # print('image %d: %s' % (i, testlist[i]))
        kclose_list = []
        kclose_img_list = []
        kclose_dist_list = []  # k closest distances
        for j in range(num_database):
            test_vec = np.load(testlist[i])
            database_vec = np.load(databaselist[j])
            dists[i][j] = cdist(test_vec, database_vec, best_dist)

        # find the k nearest neighbors
        dist_k_min = np.argsort(dists[i])[:best_k]
        pred_num.append(i)
        testlist_new.append(testlist[i])
        test_class_list_new.append(test_class_list[i])

        # k-NN majority vote
        for m in range(best_k):
            kclose_list.append(data_class_list[dist_k_min[m]])
            kclose_img_list.append(databaselist[dist_k_min[m]])
            kclose_dist_list.append(dists[i][dist_k_min[m]])
            print('For %d test img: %s, the %d th closest img is %s' % (i, testlist_new[i],m, kclose_img_list[-1]))
            # print('with the %d th smallest distance: %f' % (m, kclose_dist_list[-1]))

        # k-NN majority vote
        for n in range(len(kclose_list)):
            old = kclose_list.count(kclose_list[n])
            num = max(kclose_list.count(m) for m in kclose_list)
            if (num == old):
                pred.append(kclose_list[n])
                pred_img.append(kclose_img_list[n])
                dist_pred.append(kclose_dist_list[n])
                # print('%d---true: %s ,pred: %s' % (i, test_class_list[i], pred[-1]))
                break

    confusion = confusion_matrix(test_class_list_new, pred)
    df_cm = pd.DataFrame(confusion, index=['bladder', 'bowel', 'gallbladder', 'kidney', 'liver', 'spleen'],
                         columns=['bladder', 'bowel', 'gallbladder', 'kidney', 'liver', 'spleen'])
    plt.figure(figsize=(5, 4))
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    cmap = sn.cm.rocket_r
    # cmap = plt.cm.Blues
    ax = sn.heatmap(df_cm, annot=True, fmt='.20g', cmap=cmap)
    ax.set_xlabel("Predicted class", fontsize=12)
    ax.set_ylabel("True class", fontsize=12)
    result_dir = "confusion/{}".format(nn_model)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    whether_pca = 'ft_pca' if pca else 'ft'
    plt.savefig(
        os.path.join(result_dir, "confusion_matrix_{}_{}_k_{}_{}.jpg".format(nn_model, whether_pca, best_k, best_dist)),
        quality=100,
        bbox_inches='tight')
    ############################################

    acc_csv_file = os.path.join(result_dir, nn_model+"_kNN_accuracy.csv")

    save = pd.DataFrame(correct_rate_list)
    save.to_csv(acc_csv_file)

    # plot the classification accuracy with k
    fig, ax = plt.subplots()
    ax.set_xlabel("k", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    fig.suptitle("Accuracy of k-NN classifier with {}".format(nn_model), fontsize=14)
    ax.set_xticks(np.arange(1,10,2))
    ax.tick_params(labelsize=12)
    for dist in distance_list:
        ax.plot(k_list,correct_rate_list[dist],label=dist)
    ax.legend(fontsize=12)
    fig.savefig(os.path.join(result_dir, 'Accuracy_of_kNN_{}.png'.format(nn_model)))


if __name__ == '__main__':
    nn_model = 'resnet50'
    recognize_organs(nn_model)