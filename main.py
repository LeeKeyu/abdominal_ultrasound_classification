from extract_features import extract_features
from extract_features_pca import extract_features_pca
from recognize_organs import recognize_organs, recognize_organs_fc
from finetune import finetune


if __name__ == '__main__':
    nn_model_list = ['resnet50', 'resnet101', 'resnet152', 'densenet121', 'densenet169', 'densenet201']

    for nn_model in nn_model_list:
        finetune(nn_model)
        extract_features(nn_model, fine_tune=True)
        extract_features_pca(nn_model)
        recognize_organs(nn_model, pca=False)
        recognize_organs(nn_model, pca=True)
        recognize_organs_fc(nn_model)
