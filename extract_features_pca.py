import numpy as np
import os
from sklearn.decomposition import PCA
import glob


def extract_features_pca(nn_model):
   '''
   Do PCA dimension reduction on the features extracted by NN models
   and save as npz file
   e.g., dimension: (1, 2048) --> (1, 300) for ResNet-50
   :param nn_model:
   :return:
   '''
   raw_feature_path = './dataset/feature_' + nn_model
   ProcessedPath = './dataset/feature_' + nn_model +'_pca'
   train_list = []
   test_list = []
   train_files = glob.glob(raw_feature_path + '/train/*')
   test_files = glob.glob(raw_feature_path + '/test/*')
   for f in train_files:
      x = np.load(f)
      x = np.squeeze(x, axis=0)
      train_list.append(x)
   for f in test_files:
      x = np.load(f)
      x = np.squeeze(x, axis=0)
      test_list.append(x)

   train_list = np.array(train_list)
   test_list = np.array(test_list)
   print(train_list.shape, test_list.shape)  # (300, 2048) (60, 2048)

   # normalize the train and test data using train_mean
   train_mean = np.mean(train_list, axis=0)
   train_list -= train_mean
   test_list -= train_mean

   # do pca
   # pca = PCA(n_components=None)
   pca = PCA(n_components=0.99)
   train_pca = pca.fit_transform(train_list)
   test_pca = pca.transform(test_list)
   print(train_pca.shape, test_pca.shape)

   # expand dimension
   train_pca_list = train_pca.tolist()
   for i in range(len(train_pca_list)):
      train_pca_list[i] = np.expand_dims(train_pca_list[i], axis=0)

   test_pca_list = test_pca.tolist()
   for i in range(len(test_pca_list)):
      test_pca_list[i] = np.expand_dims(test_pca_list[i], axis=0)

   # save the features as npz files
   for i in range(len(train_pca_list)):
      train_feature = train_pca_list[i]
      train_file = train_files[i]
      train_file = os.path.split(train_file)[1]
      image_pre, ext = os.path.splitext(os.path.splitext(train_file)[0])
      savepath = os.path.join(ProcessedPath, 'train')
      if (not os.path.exists(savepath)):
         os.makedirs(savepath)
      np.save(os.path.join(savepath,image_pre+'.npz'), train_feature)

   for i in range(len(test_pca_list)):
      test_feature = test_pca_list[i]
      test_file = test_files[i]
      test_file = os.path.split(test_file)[1]
      image_pre, ext = os.path.splitext(os.path.splitext(test_file)[0])
      savepath = os.path.join(ProcessedPath, 'test')
      if (not os.path.exists(savepath)):
         os.makedirs(savepath)
      np.save(os.path.join(savepath,image_pre+'.npz'), test_feature)


if __name__ == '__main__':
   nn_model = 'resnet50'
   extract_features_pca(nn_model)