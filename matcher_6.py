import imp
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_trf")

matcher = Matcher(nlp.vocab)
pattern = [{"LIKE_EMAIL": True}]
matcher.add("EMAIL_ADDRESS", [pattern])

doc = nlp("This is an email address: kurian.bkk@gmail.com")
matches = matcher(doc)

print(matches)

with open("wiki_us.txt", "r") as f:
    text = f.read()

matcher = Matcher(nlp.vocab)
pattern = [{"POS": "PROPN"}]
matcher.add("PROPER_NOUNS", [pattern])

# doc = nlp(text)
# matches = matcher(doc)
# print(len(matches))
# for match in matches[:10000]:
#     print(match, doc[match[1] : match[2]])

matcher = Matcher(nlp.vocab)
pattern = [{"POS": "PROPN", "OP": "+"}]
matcher.add("PROPER_NOUNS", [pattern], greedy="LONGEST")
doc = nlp(text)
matches = matcher(doc)
print(len(matches))
for match in matches[:10]:
    print(match, doc[match[1] : match[2]])
