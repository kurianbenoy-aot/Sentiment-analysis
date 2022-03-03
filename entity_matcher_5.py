import spacy

nlp = spacy.load("en_core_web_trf")

# Sample text
text = (
    "The village of Treblinka is in Poland. Treblinka was also an extermination camp."
)

ruler = nlp.add_pipe("entity_ruler", after="ner")
patterns = [{"label": "GPE", "pattern": "Treblinka"}]

ruler.add_patterns(patterns)

# Create the Doc object
doc = nlp(text)

# extract entities
for ent in doc.ents:
    print(ent.text, ent.label_)
