import string
import pandas as pd
from itertools import islice


def create_vocabulary():

    # loading data from file
    file_name = 'Data/Flickr8k_text/Flickr8k.token.txt'
    df = pd.read_csv(file_name, sep="\t", names=['id', 'caption'])

    # # printing sample data
    print(df.head())

    # as each image has 5 captions so storing each image's caption with each other
    # creating dictionary
    # separating on the basis of '#' in first column of data
    captions = {}
    # the name of the image will be the key and all its captions will be stored as values in dictionary
    # iterating rows of the dataframe
    for index, df_row in df.iterrows():
        name_array = df_row['id'].split('.')
        name = name_array[0]

        if name not in captions:
            captions[name] = []
        if df_row['caption'] not in captions[name]:
            captions[name].append(df_row['caption'])

    # # print sample data
    dict(islice(captions.items(), 0, 2))

    # Now that we have a dictionary of name and captions of images, we have to clean the captions
    # As we see from above sample that our data has uncosistent alphabetical case and also contains a lot of punctuation
    # We can remove punctuation and also make our alphabets of same case
    for key in captions:
        for i in range(len(captions[key])):
            # converting string to lowercase
            captions[key][i] = captions[key][i].lower()
            # removing punctuation
            captions[key][i] = captions[key][i].translate(str.maketrans('', '', string.punctuation))

    # print sample data
    dict(islice(captions.items(), 0, 2))

    # As we now have refined data, we can store it in the same format in a different file to keep our model generic
    file_name = "descriptions.txt"
    string_to_file = ""
    for key in captions:
        for c in captions[key]:
            string_to_file += key + "\t" + c + "\n"

    file = open(file_name, "w")
    file.write(string_to_file)
    file.close()

    print("Successfully printed the file")

    # Now we will create a set to store all the unique words
    # just to get an estimate for our vocabulary size
    vocab = set()
    for key in captions:
        split_list = []
        for c in captions[key]:
            split_list = c.split()
            # vocab.update adds the list "split_list" to the set "vocab"
            vocab.update(split_list)

    print(len(vocab))
