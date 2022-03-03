import spacy
from spacy import displacy

nlp = spacy.load("model-best")

with open("sentiment_test.txt", "r") as f:
    text = f.read()

TEST_DATA = text.splitlines()


def evaluate(text):
    doc = nlp(text)

    print(text)
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)


for line in TEST_DATA:
    evaluate(line)
