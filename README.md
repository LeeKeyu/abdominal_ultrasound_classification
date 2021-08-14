# abdominal_ultrasound_classification

Different from pure deep learning-based methods, this work is committed to combining deep convolutional neural networks with PCA and k-Nearest Neighbor classification for real-time recognition of abdominal organs in ultrasound images, to reduce the requirement of large datasets and improve the classification performance with minimal training effort.

### Requirements

The packages used in this project include Tensorflow (2.6), Keras (2.6), numpy, pandas, scipy, scikit-learn, seaborn, and matplotlib.

### Method Overview

<img src="https://github.com/LeeKeyu/abdominal_ultrasound_classification/blob/master/result/workflow.jpg" width="60%" height="60%">

We use fine-tuned deep convolutional neural networks (e.g., ResNets, DenseNets) combined with PCA dimension reduction to extract features of the ultrasound images, and implement the k-Nearest-Neighbor approach with Euclidean distance, City block distance, Canberra distance and Cosine distance to achieve automatic recognition of abdominal organs in the ultrasound images in real time.

### Dataset

The [dataset](https://github.com/ftsvd/USAnotAI) we use contains 360 ultrasound images of six abdominal organs. 300 images are used as training set (database) and 60 images are used for testing.

### Code Structure & Use
 - **finetune.py**: fine-tune the deep neural networks (pretrained on the ImageNet dataset) using our training data
 - **extract_features.py**: extract features using pretrained or fine-tuned deep neural networks from abdominal ultrasound images
 - **extract_features_pca.py**: conduct PCA dimension reduction on the extracted features
 - **recognize_organs.py**: use k-NN classification to recognize the abdominal organ in the image by comparing distances between features of the database and test images

To simply conduct the fine-tuning, feature extraction, PCA dimension reduction and classification successively, run

```
python main.py
```


### Results

Learning curves during fine-tuning:

<img src="https://github.com/LeeKeyu/abdominal_ultrasound_classification/blob/master/result/learning_curves.jpg" width="50%" height="50%">

Classification accuracy using different feature extractors and classifiers:

<img src="https://github.com/LeeKeyu/abdominal_ultrasound_classification/blob/master/result/comparison.jpg" width="50%" height="50%">

### Questions

If you have any questions, please feel free to contact "kyli@link.cuhk.edu.hk".
