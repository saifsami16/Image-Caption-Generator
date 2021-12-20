from keras.preprocessing.text import Tokenizer
import pickle


def generate_tokenizer(train_captions):

    # Computer don't understand words, so we will convert the words to numbers, so they are more relatable for computer
    # Keras has a tokenizer class which will help us achieve the mapping of these words to integers
    # converting train_captions from dictionary to strings
    train_captions_str = list()
    for key in train_captions.keys():
        [train_captions_str.append(c) for c in train_captions[key]]

    # print(len(train_captions_str))
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_captions_str)
    tokenizer_size = len(tokenizer.word_index) + 1
    pickle.dump(tokenizer, open("tokenizer.p", "wb"))
    # Print vocabulary size
    print(tokenizer_size)

    # Now we will calculate the maximum length of a caption in train_captions as we have to decide
    # model parameters in the future

    max_length_captions = max(len(c.split()) for c in train_captions_str)
    print(max_length_captions)

    return max_length_captions


def load_tokenizer():
    pickle_file = "tokenizer.p"
    tokenizer = pickle.load(open(pickle_file, 'rb'))
    return tokenizer
