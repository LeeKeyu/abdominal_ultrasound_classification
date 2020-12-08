# abdominal_ultrasound_classification
Using deep neural networks with k-NN classification for abdominal organ recognition in ultrasound images.

### Requirements
This project uses Keras, numpy, pandas, scipy, scikit-learn and matplotlib packages.

### Method Overview
![](/result/workflow.jpg){:height="50%" width="50%"}
We use deep neural networks (e.g., ResNet and DenseNet) combined with PCA dimension reduction to extract features of the ultrasound images, and then use the k-Nearest-Neighbor approach with Euclidean distance, City block distance, Canberra distance and Cosine distance for automatic classification of abdominal organs in the ultrasound images.

### Dataset
The [dataset](https://github.com/ftsvd/USAnotAI) we use contains ultrasound images of six abdominal organs. 300 images are used for training and 60 images for testing.

### Code Structure & Use
 - **finetune.py**: fine-tune the deep neural networks which are pre-trained on the ImageNet dataset using our training data
 - **extract_features.py**: extract features using pre-trained or fine-tuned deep neural networks from original images
 - **extract_features_pca.py**: conduct PCA dimension reduction on the extracted features
 - **recognize_organs.py**: use k-NN to recognize the abdominal organ in the image by comparing distances between features of the train and
   test images

Simply run the fine-tuning, feature extraction, dimension reduction and classification successively:
```
python main.py
```

### Results
Learning curves during fine-tuning:
![](https://github.com/LeeKeyu/abdominal_ultrasound_classification/blob/master/result/learning_curve.png)
Comparison of classification accuracy using different feature extractors and classifiers:
![](https://github.com/LeeKeyu/abdominal_ultrasound_classification/blob/master/result/comparison.png)

