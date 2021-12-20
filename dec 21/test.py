import pickle
import pandas as pd
from itertools import islice


def load_test_features():

    # Now we are going to use the train_captions with the pickle file we generated from feature extraction of images
    pickle_file = "features.p"
    all_features = pickle.load(open(pickle_file, 'rb'))

    file_name = 'Data/Flickr8k_text/Flickr_8k.devImages.txt'
    df = pd.read_csv(file_name, sep="\t", names=['id'])
    test_image_ids = df["id"].tolist()
    for i in range(0, len(test_image_ids)):
        #     removing extension as well from the image name/id
        test_image_ids[i] = test_image_ids[i].split('.')[0]
    test_image_ids = set(test_image_ids)

    # Now we will load the captions that we cleaned previously
    # And we will check which captions are there in the test_image_ids and only store those that are present
    test_captions = {}
    file_name = 'descriptions.txt'
    df = pd.read_csv(file_name, sep="\t", names=['id', 'caption'])

    for index, df_row in df.iterrows():

        name = df_row['id']
        if name in test_image_ids:
            if name not in test_captions:
                test_captions[name] = []
            #             so as not to repeat any caption
            if df_row['caption'] not in test_captions[name]:
                test_captions[name].append('startseq ' + df_row['caption'] + ' endseq')

    # print sample data
    dict(islice(test_captions.items(), 0, 2))

    # Now as all of our features are stored in all_features variable, so we will extract only the test features
    test_features = {k: all_features[k+'.jpg'] for k in test_image_ids}

    # print length of train_features
    print(len(test_features))

    return test_captions, test_features
