import json
import re

import env


def load_dictionary():

    dataset = None
    with open(env.QIPEDC_DATASET, 'rb') as file:
        dataset = json.load(file)

    encoded_word = {}
    for word_metadata in dataset['data']:
        id = str(word_metadata['_id']).upper()

        word_text = str(word_metadata['word']).lower()
        word_text = re.sub("[\(\[].*?[\)\]]", "", word_text)
        words = [word.strip() for word in word_text.split('/ ')]

        for word in words:
            if word not in encoded_word or not str(encoded_word[word]).endswith('B'):
                encoded_word[word] = id

    return encoded_word