import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("I want a dog")

print(f"{'TEXT':<10} | {'POS':<6} | {'DEP':<10} | {'EXPLANATION'}")
print("-" * 50)

for token in doc:
    print(f"{token.text:<10} | {token.pos_:<6} | {token.dep_:<10} | {spacy.explain(token.dep_)}")