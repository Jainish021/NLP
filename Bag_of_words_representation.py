#Importing function from other file.
from Vocabulary import vocab

import pandas as pd
from imutils import paths
from sklearn.feature_extraction.text import CountVectorizer


# Importing the vocabulary built in Vocabulary.py file
vocabulary = vocab()

# Defining the list of path where all the training data files are stored.
filePaths = #Enter the path to files

# Reading all training data file and preparing a corresponding features*example matrix for bag_of_words model using CountVectorizer() method.
vec = CountVectorizer()


def BOW():
    # Initializing the data list
    data = []
    # Reading all training data file and preparing a corresponding features*example matrix for bag_of_words model using CountVectorizer() method.
    vec = CountVectorizer()
    for filePath in filePaths:
        with open(filePath, errors="ignore") as f:
            contents = f.read()
            data.append(contents)
            f.close()
    vec.fit(vocabulary)
    mat = vec.transform(data)
    # Creatin the dataframe of the data matrix.
    cols = vec.get_feature_names()
    rows = range(len(mat.toarray()))
    df = pd.DataFrame(mat.toarray(), columns=cols, index=rows)
    return mat.toarray(), df


if __name__ == "__main__":
    bow, _ = BOW()
    print(bow)
