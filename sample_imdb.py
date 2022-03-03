import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_trf")

with open("wiki_us.txt", "r") as f:
    text = f.read()
    doc = nlp(text)
    print(len(doc))

    # for token in doc[:10]:
    #     print(token)

    # Sentence boundary detection

    for sent in doc.sents:
        print(sent)

    sentence1 = list(doc.sents)[0]
    # print(sentence1)
    for token in sentence1:
        print(token.text, token.pos_, token.dep_)

    displacy.render(sentence1, style="dep")
