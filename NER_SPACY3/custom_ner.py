import json

import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

nlp = spacy.blank("en")
db = DocBin()

f = open("annotations.json", "r")

TRAIN_DATA = json.load(f)

for text, annot in tqdm(TRAIN_DATA["annotations"]):
    doc = nlp.make_doc(text)
    entities = []

    for start, end, label in annot["entities"]:
        # what is char_span?
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            entities.append(span)
    doc.ents = entities
    db.add(doc)

db.to_disk("./training_data.spacy")
