import pandas as pd
from datetime import datetime
import spacy

from spacy.tokens import DocBin

df = pd.read_csv("archive/all-data.csv", encoding="latin-1")
train = df.sample(frac=0.8, random_state=25)
test = df.drop(train.index)

nlp = spacy.load("en_core_web_trf")

train["tuples"] = train.apply(lambda row: (row["Text"], row["Sentiment"]), axis=1)
train = train.tuples.tolist()

test["tuples"] = test.apply(lambda row: (row["Text"], row["Sentiment"]), axis=1)
test = test.tuples.tolist()

print(train[0])


def document(data):
    text = []
    for doc, label in nlp.pipe(data, as_tuples=True):
        if label == "positive":
            doc.cats["positive"] = 1
            doc.cats["negative"] = 0
            doc.cats["neutral"] = 0
        elif label == "negative":
            doc.cats["positive"] = 0
            doc.cats["negative"] = 1
            doc.cats["neutral"] = 0
        else:
            doc.cats["positive"] = 0
            doc.cats["negative"] = 0
            doc.cats["neutral"] = 1
            # Adding the doc into the list 'text'
            text.append(doc)
    return text


train_docs = document(train)
doc_bin = DocBin(docs=train_docs)

doc_bin.to_disk("train.spacy")

test_docs = document(test)
doc_bin = DocBin(docs=test_docs)

doc_bin.to_disk("test.spacy")
