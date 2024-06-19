import spacy
nlp = spacy.load("fr_core_news_sm")


def process_clause_version_one(clause):
    """
    time + subject + obj + verb + others
    Ex:  je vais à l'ecole ce matin et je rentrerai le soir.
       Reponse:
        je Number=Sing|Person=1
        vais Mood=Ind|Number=Sing|Person=1|Tense=Pres|VerbForm=Fin
        ecole Gender=Masc|Number=Sing
        matin Gender=Masc|Number=Sing
        je Number=Sing|Person=1
        rentrerai Mood=Ind|Number=Sing|Person=1|Tense=Fut|VerbForm=Fin
        soir Gender=Masc|Number=Sing
        -> Glosses: pres je aller ecole matin et fut je soir rentrer

    """
    doc = nlp(clause)

    filtered_tokens = []
    for token in doc:
        if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV", "NUM", "PRON"}:
            filtered_tokens.append(token)

    subject = []
    verb = []
    obj = []
    others = []
    time = []

    for token in filtered_tokens:
        #print(token, token.morph)
        if "subj" in token.dep_:
            subject.append(token.lemma_)
        elif "obj" in token.dep_:
            obj.append(token.lemma_)
        elif token.pos_ == "VERB":
            morph = token.morph.get("Tense")
            if morph:
                time.append(morph[0].lower())
            verb.append(token.lemma_)
        else:
            others.append(token.lemma_)

    gloss_sequence = time + subject + obj + verb + others

    return " ".join(gloss_sequence)

def process_clause_version_two(clause):
    """
    time + location + subject + verb + obj + others
    """
    doc = nlp(clause)

    filtered_tokens = []
    for token in doc:
        if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV", "NUM", "PRON"} and token.pos_ != "DET":
            filtered_tokens.append(token)

    subject = []
    verb = []
    obj = []
    time = []
    location = []
    others = []

    for token in filtered_tokens:
        if "subj" in token.dep_:
            subject.append(token.lemma_)
        elif "obj" in token.dep_:
            obj.append(token.lemma_)
        elif token.pos_ == "VERB":
            morph = token.morph.get("Tense")
            if morph:
                time.append(morph[0].lower())
            verb.append(token.lemma_)
        elif token.dep_ in {"advmod", "npadvmod"} and token.pos_ == "ADV":
            time.append(token.lemma_)
        elif token.dep_ == "obl" and token.ent_type_ == "LOC":
            location.append(token.lemma_)
        else:
            others.append(token.lemma_)

    gloss_sequence = time + location + subject + verb + obj + others

    return " ".join(gloss_sequence)

def text_to_gloss(text):
    clauses = text.split(" et ")
    processed_clauses = [process_clause_version_two(clause) for clause in clauses]


    return " et ".join(processed_clauses)

def main():
    text = input("Entrez le texte à convertir en glosses: ")
    print("Texte: ", text)

    gloss = text_to_gloss(text)

    print("Glosses:", gloss)

if __name__ == "__main__":
    main()