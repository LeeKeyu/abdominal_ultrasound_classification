import numpy as np
import os
from scipy.spatial.distance import cdist
import time
import shutil
import matplotlib.pyplot as plt
import pandas as pd


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

    correct_rate_list = dict()
    for dist_category in distance_list:
        correct_rate_list[dist_category] = []
        for k in k_list:
            start = time.clock()
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


            for file in error_test_list:
                file_name = os.path.split(file)[1]
                print(file_name)
                file_without_ext = os.path.splitext(file_name)[0]
                file_without_ext = os.path.splitext(file_without_ext)[0]
                print(file_without_ext)
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
            print("Total time:" + str(total_time) + "s")

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