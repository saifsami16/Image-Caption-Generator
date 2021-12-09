import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input


def generator_helper(tokenizer_file, model_file, max_length_caption, picture):
    tokenizer = pickle.load(open(tokenizer_file, 'rb'))
    # create a dictionary to store all the values of a tokenizer to avoid reiteration
    tokenizer_dict = {}
    for text, integer in tokenizer.word_index.items():
        tokenizer_dict[integer] = text

    # extract features from photo
    model = Xception(weights='imagenet', include_top=False, pooling='avg')
    new_file_name = picture
    # Xception only accepts images of '299x299' dimensions
    image = load_img(new_file_name, target_size=(299, 299))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)

    # Now we will create description for our images
    # Here we are going to use Greedy Search algorithm to generate our image captions
    # We have stored probability against each word in our model, greedy will pick the word having the most probability
    model = load_model(model_file)
    generated_caption = 'startseq'
    curr_word = "something"
    found_word = True
    i = 0
    while i < max_length_caption and curr_word and curr_word != 'endseq':
        # converting word to number
        # making length of each seuqence equal by padding it with zeros
        converted_sentence = tokenizer.texts_to_sequences([generated_caption])[0]
        curr_sequence = pad_sequences([converted_sentence], maxlen=max_length_caption)
        prediction = model.predict([feature, curr_sequence], verbose=0)
        prediction = np.argmax(prediction)
        # now we will check if we have the prediction in our tokenizer or not
        if prediction in tokenizer_dict:
            curr_word = tokenizer_dict[prediction]
            generated_caption += ' ' + curr_word
        else:
            curr_word = None
    temp = generated_caption.split(" ")
    temp = temp[1:len(temp) - 1]
    generated_caption = ' '.join(temp)
    return generated_caption


def generator(picture):
    # picture = "example.jpg"
    tokenizer_file = "tokenizer.p"
    max_length_caption = 38
    model_file = "models/9.h5"
    caption = generator_helper(tokenizer_file, model_file, max_length_caption, picture)
    return caption
