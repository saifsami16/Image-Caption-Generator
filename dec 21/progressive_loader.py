import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# we have to train our model on thousands of images and this data cannot be store in our RAM
# so we are going to use a technique commonly practiced among machine learning engineers
# also known as progressive loading
# https://wiki.python.org/moin/Generators
# we will try to split our models and write them to files


def generator_func(train_captions, train_features, tokenizer, max_length_captions, tokenizer_size):
    x1 = []
    x2 = []
    y = []
    while True:
        for key, desc_list in train_captions.items():
            photo = train_features[key][0]
            for desc in desc_list:
                sequence = tokenizer.texts_to_sequences([desc])[0]

                for i in range(1, len(sequence)):

                    sequence_1, sequence_2 = sequence[:i], sequence[i]

                    sequence_1 = pad_sequences([sequence_1], maxlen=max_length_captions)[0]
                    sequence_2 = to_categorical([sequence_2], num_classes=tokenizer_size)[0]
                    x1.append(photo)
                    x2.append(sequence_1)
                    y.append(sequence_2)
            yield [np.array(x1), np.array(x2)], np.array(y)

            x1 = []
            x2 = []
            y = []
