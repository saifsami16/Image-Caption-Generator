from vocabulary_generator import create_vocabulary
from feature_extractor import extract_features
from train import load_train_features
from test import load_test_features
from create_tokenizer import generate_tokenizer
from create_tokenizer import load_tokenizer
from model import generate_model


# main script for managing all the other modules and passing data between them


# storing clean data in "descriptions.txt" file
create_vocabulary()

# extract features from images in "features.p" file
extract_features()

# extracting training data
train_captions, train_features = load_train_features()

# extracting testing data
test_captions, test_features = load_test_features()

# generating a tokenizer file
len_longest_caption = generate_tokenizer(train_captions)

# load "tokenizer" file into a variable
tokenizer = load_tokenizer()

# train model on generated features
# function will generate a "model.h5" file
generate_model(train_captions, train_features, tokenizer, len_longest_caption)
