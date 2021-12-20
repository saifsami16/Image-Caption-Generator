import os
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from tensorflow.keras.utils import plot_model
from progressive_loader import generator_func


def generate_model(train_captions, train_features, tokenizer, max_length_captions):

    # storing tokenizer size
    tokenizer_size = len(tokenizer.word_index) + 1

    # Now we will define our model
    # The features that we previously extracted has a large size
    # So we will try to reduce them to 256
    input_1 = Input(shape=(2048,))
    feature_1 = Dropout(0.5)(input_1)
    feature_2 = Dense(256, activation='relu')(feature_1)

    # Now we will create an embedding layer which will manage our vocabulary using LSTM
    input_2 = Input(shape=(max_length_captions,))
    sequence_1 = Embedding(tokenizer_size, 256, mask_zero=True)(input_2)
    sequence_2 = Dropout(0.5)(sequence_1)
    sequence_3 = LSTM(256)(sequence_2)

    # And now, we will combine both the above models generated
    decoder_1 = add([feature_2, sequence_3])
    decoder_2 = Dense(256, activation='relu')(decoder_1)
    output = Dense(tokenizer_size, activation='softmax')(decoder_2)
    model = Model(inputs=[input_1, input_2], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # also, writing the structure of our output model to a png file
    plot_model(model, to_file='model.png', show_shapes=True)

    # We can create a folder to hold our multiple model files
    os.mkdir("models_new")
    # we can set the number of epochs as 25 as it is enough for our model, as long as log loss is decreasing
    epochs = 25
    steps = len(train_captions)
    for i in range(0, epochs):
        generator = generator_func(train_captions, train_features, tokenizer, max_length_captions, tokenizer_size)

        model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        # check validity for path
        model.save(f'models_new/{str(i)}.h5')
