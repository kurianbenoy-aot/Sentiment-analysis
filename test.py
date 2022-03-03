import spacy


text = "Apple earnings: Huge iPhone 12 sales beat analyst expectations"
nlp = spacy.load("meraModel/model-best")
demo = nlp(text)
print(demo.cats)
