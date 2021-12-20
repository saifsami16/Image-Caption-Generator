import pickle
import pandas as pd
from itertools import islice


def load_train_features():

    # Now we are going to use the train_captions with the pickle file we generated from feature extraction of images
    pickle_file = "features.p"
    all_features = pickle.load(open(pickle_file, 'rb'))

    # Load train images that are pre-defined in the dataset and store them in a set

    file_name = 'Data/Flickr8k_text/Flickr_8k.trainImages.txt'
    df = pd.read_csv(file_name, sep="\t", names=['id'])
    train_image_ids = df["id"].tolist()
    for i in range(0, len(train_image_ids)):
        #     removing extension as well from the image name/id
        train_image_ids[i] = train_image_ids[i].split('.')[0]
    train_image_ids = set(train_image_ids)

    # Now we will load the captions that we cleaned previously
    # And we will check which captions are there in the train_image_ids and only store those that are present
    train_captions = {}
    file_name = 'descriptions.txt'
    df = pd.read_csv(file_name, sep="\t", names=['id', 'caption'])

    for index, df_row in df.iterrows():

        name = df_row['id']
        if name in train_image_ids:
            if name not in train_captions:
                train_captions[name] = []
            #             so as not to repeat any caption
            if df_row['caption'] not in train_captions[name]:
                train_captions[name].append('startseq ' + df_row['caption'] + ' endseq')

    # print sample data
    dict(islice(train_captions.items(), 0, 2))

    # we only have to use train_captions features so extracting only those training features

    # in pickle, we have the file names with extensions. But in our train_image_ids, we just
    # have id's without extensions. So adding '.jpg' to make it more consistent.
    # In the future, remove '.jpg' from pickle as well
    train_features = {k: all_features[k+'.jpg'] for k in train_image_ids}

    # print length of train_features
    print(len(train_features))
    return train_captions, train_features
