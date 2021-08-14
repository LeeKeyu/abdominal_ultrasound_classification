from keras.applications.resnet import ResNet50
from keras.applications.densenet import DenseNet121,DenseNet169,DenseNet201
from keras.applications.resnet import ResNet101, ResNet152
from keras.preprocessing import image
from keras.applications.resnet import preprocess_input
import numpy as np
import os


def extract_features(nn_model, fine_tune=False):
   if fine_tune is True:
      if nn_model == 'resnet50':
         model = ResNet50(include_top=False, weights=None, pooling='avg')
      elif nn_model == 'resnet101':
         model = ResNet101(include_top=False, weights=None, pooling='avg')
      elif nn_model == 'resnet152':
         model = ResNet152(include_top=False, weights=None, pooling='avg')
      elif nn_model == 'densenet121':
         model = DenseNet121(include_top=False, weights=None, pooling='avg')
      elif nn_model == 'densenet169':
         model = DenseNet169(include_top=False, weights=None, pooling='avg')
      elif nn_model == 'densenet201':
         model = DenseNet201(include_top=False, weights=None, pooling='avg')
      else:
         raise NotImplementedError("The NN model is not implemented!")
      model.load_weights('./finetune/' + nn_model + '/finetune_weights_50_epoch.h5', by_name=True)

   else:
      if nn_model == 'resnet50':
         model = ResNet50(include_top=False, weights='imagenet', pooling='avg')
      elif nn_model == 'resnet101':
         model = ResNet101(include_top=False, weights='imagenet', pooling='avg')
      elif nn_model == 'resnet152':
         model = ResNet152(include_top=False, weights='imagenet', pooling='avg')
      elif nn_model == 'densenet121':
         model = DenseNet121(include_top=False, weights='imagenet', pooling='avg')
      elif nn_model == 'densenet169':
         model = DenseNet169(include_top=False, weights='imagenet', pooling='avg')
      elif nn_model == 'densenet201':
         model = DenseNet201(include_top=False, weights='imagenet', pooling='avg')
      else:
         raise NotImplementedError("The NN model is not implemented!")

   ImgPath = './dataset/img'
   ProcessedPath = './dataset/feature_'+nn_model
   Lastlist = os.listdir(ImgPath)
   sum = 0
   for lastfolder in Lastlist:
       LastPath = os.path.join(ImgPath,lastfolder)
       savepath =os.path.join(ProcessedPath,lastfolder)
       imagelist = os.listdir(LastPath)
       for image1 in imagelist:
          sum += 1
          print(image1)
          print('sum is ', sum)
          image_pre, ext = os.path.splitext(image1)
          imgfile = LastPath + '/'+ image1
          img = image.load_img(imgfile, target_size=(64, 64))
          x = image.img_to_array(img)      # shape:(64,64,3)
          x = np.expand_dims(x, axis=0)
          x = preprocess_input(x)   # shape:(1,64,64,3)
          print(x.shape)
          want = model.predict(x)
          print(np.shape(want))   # shape:(1,2048)

          # normalize
          a = np.min(want)
          b = np.max(want)
          want = (want-a)/(b-a)
          if(not os.path.exists(savepath)):
             os.makedirs(savepath)
          np.save(os.path.join(savepath,image_pre+'.npz'),want)


if __name__ == '__main__':

   nn_model = 'densenet169'
   extract_features(nn_model)