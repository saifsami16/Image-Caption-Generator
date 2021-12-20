import pickle
from os import listdir
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


def extract_features():
    # We are done with the text, and now we have to work with the picture data
    # i.e. we have to extract features vector from the image data
    # also called transfer learning as we will use an already trained model from keras

    # Currently, I am using Xception, but I can also use 'MobileNet' by Google
    # for better understanding
    # https://towardsdatascience.com/how-to-choose-the-best-keras-pre-trained-model-for-image-classification-b850ca4428d4

    # When loading a given model, the “include_top” argument can be set to False, in which case the fully-connected
    # output layers of the model used to make predictions is not loaded, allowing a new output layer to be added and
    # trained. Supplying weights="imagenet" indicates that we want to use the pre-trained ImageNet weights for the
    # respective model

    # This is temporarily commented because we have generated the features file as 'features.p', and it takes a lot
    # of time so we will only use the generated file without making it from scratch again and again

    model = Xception(weights='imagenet', include_top=False, pooling='avg')
    features = dict()
    print(model.summary())
    path = "Data/Flickr8k_Dataset/"

    for file_name in listdir(path):
        new_file_name = path + file_name
        # Xception only accepts images of 299x299
        print(new_file_name)
        image = load_img(new_file_name, target_size=(299, 299))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        feature = model.predict(image, verbose=0)
        features[file_name] = feature

    pickle.dump(features, open("features.p", "wb"))

    print("Features successfully extracted")
