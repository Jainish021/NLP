#Importing functions from other file.
from AlphaNumeric_vocabulary_builder import vocab

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# Importing the vocabulary built in Vocabulary.py file.
vocabulary = vocab()

# Reading all training data file and preparing a corresponding features*example matrix for bag_of_words model using CountVectorizer() method.
vec = CountVectorizer()

filePaths = #Enter the path to text files.

# Preproccesing the data and converting it into bernoulli model data.
def bernoulli():
    data = []

    for filePath in filePaths:
        with open(filePath, errors="ignore") as f:
            contents = f.read()
            data.append(contents)
            f.close()
    vec.fit(vocabulary)
    mat = vec.transform(data)

    cols = vec.get_feature_names()
    rows = range(len(mat.toarray()))
    df = pd.DataFrame(mat.toarray(), columns=cols, index=rows)

    test_data = df.mask(df > 0, 1)

    return test_data.to_numpy()


if __name__ == "__main__":
    ber = bernoulli()
    print(ber)
