train_test_ratio = 0.10
train_valid_ratio = 0.80

first_n_words = 510
#raw_data_path = 'data/dataset.csv'
destination_folder = 'data'

import pandas as pd
from sklearn.model_selection import train_test_split
from train import main

def trim_string(x):

    x = str(x).split(maxsplit=first_n_words)
    x = ' '.join(x[:first_n_words])

    return x


def prepare_data(raw_data_path):
    # Read raw data
    df_raw = pd.read_csv(raw_data_path)

    # Prepare columns
    df_raw['label'] = (df_raw['label'] == 'FAKE').astype('int')
    df_raw['titletext'] = df_raw['title'] + ". " + df_raw['text']
    df_raw = df_raw.reindex(columns=['label', 'title', 'text', 'titletext'])

    print('=======================')
    print(df_raw.text)
    print('=======================')
    print(df_raw['text'].values)
    # Drop rows with empty text
    # df_raw.drop( df_raw[df_raw.text.str.len() < 5].index, inplace=True)
    df_raw = df_raw.dropna()
    # Trim text and titletext to first_n_words
    df_raw['text'] = df_raw['text'].apply(trim_string)
    df_raw['titletext'] = df_raw['titletext'].apply(trim_string) 

    # Split according to label
    df_real = df_raw[df_raw['label'] == 0]
    df_fake = df_raw[df_raw['label'] == 1]

    # Train-test split
    df_real_full_train, df_real_test = train_test_split(df_real, train_size = train_test_ratio, random_state = 1)
    df_fake_full_train, df_fake_test = train_test_split(df_fake, train_size = train_test_ratio, random_state = 1)

    # Train-valid split
    df_real_train, df_real_valid = train_test_split(df_real_full_train, train_size = train_valid_ratio, random_state = 1)
    df_fake_train, df_fake_valid = train_test_split(df_fake_full_train, train_size = train_valid_ratio, random_state = 1)

    # Concatenate splits of different labels
    df_train = pd.concat([df_real_train, df_fake_train], ignore_index=True, sort=False)
    df_valid = pd.concat([df_real_valid, df_fake_valid], ignore_index=True, sort=False)
    df_test = pd.concat([df_real_test, df_fake_test], ignore_index=True, sort=False)

    # Write preprocessed data
    df_train.to_csv(destination_folder + '/train.csv', index=False)
    df_valid.to_csv(destination_folder + '/valid.csv', index=False)
    df_test.to_csv(destination_folder + '/test.csv', index=False)

    main()
