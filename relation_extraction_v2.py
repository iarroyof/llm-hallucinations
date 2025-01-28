import spacy
from spacy.matcher import Matcher

class RelationExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        self._add_patterns()

    def _add_patterns(self):
        # Define patterns for verb-based relations
        patterns = [
            [{"POS": "VERB"}, {"POS": "PART", "OP": "*"}, {"POS": "ADV", "OP": "*"}],
            [{"POS": "VERB"}, {"POS": "ADP", "OP": "*"}, {"POS": "DET", "OP": "*"},
             {"POS": "AUX", "OP": "*"}, {"POS": "ADJ", "OP": "*"}, {"POS": "ADV", "OP": "*"}],
            [{"POS": "VERB"}, {"POS": "ADP", "OP": "*"}],  # Verb + Preposition
            [{"POS": "VERB"}, {"POS": "DET", "OP": "*"}, {"POS": "NOUN", "OP": "*"}],  # Verb + Det + Noun
            [{"POS": "VERB"}, {"POS": "NOUN", "OP": "*"}]  # Verb + Noun
        ]
        for pattern in patterns:
            self.matcher.add("SYNTACTIC_CONSTRAINT", [pattern])

    def _extract_relations_from_sentence(self, sentence):
        doc = self.nlp(sentence)
        relations = []

        # Extract verb-based relations using dependency parsing
        for token in doc:
            if token.pos_ == "VERB":
                subject = None
                object_ = None
                # Find the subject of the verb
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject = child.text
                    # Find the object of the verb
                    elif child.dep_ in ["dobj", "attr", "prep", "pobj"]:
                        object_ = child.text
                if subject and object_:
                    relations.append((subject, token.text, object_))

        # Extract additional relations for dates and locations
        for ent in doc.ents:
            if ent.label_ in ["DATE", "GPE", "LOC"]:
                for token in doc:
                    if token.pos_ == "VERB" and token.text in ["born", "died", "located"]:
                        relations.append((token.head.text, token.text, ent.text))

        return relations

    def extract_relations(self, texts):
        results = {}
        for text in texts:
            doc = self.nlp(text)
            relations = []
            for sentence in doc.sents:
                relations.extend(self._extract_relations_from_sentence(sentence.text))
            results[text] = relations
        return results

def extract_relations(answers, n_qs_semantic_search_results, extractor):
    """
    results = extractor.extract_relations(texts)
    for text, relations in results.items():
        print(f"Text: {text}")
        for relation in relations:
            print(f"  Relation: {relation}")
        print()
    """
    wiki_docs_fquestion_relations = []
    fanswer_relations = []
    for wiki_docs_from_question, answer in zip(n_qs_semantic_search_results, answers):
        relations = [rs for _, rs in extractor.extract_relations(wiki_docs_from_question).items()]
        wiki_docs_fquestion_relations.append(relations)
        relations = [rs for _, rs in extractor.extract_relations([answer]).items()]
        fanswer_relations.append(relations)

    return wiki_docs_fquestion_relations, fanswer_relations

# Example usage
"""
if __name__ == "__main__":
    texts = [
        "In July 2012, Ancestry.com found a strong likelihood that Dunham was descended from John Punch.",
        "Elizabeth was glad to be taken to her immediately. She was shown into the breakfast-parlour.",
        "Sean is going to the mall. Rochelle enjoys candy.",
        "Her appearance created a great deal of surprise. She was received, however, very politely by them.",
        "When the clock struck three, Elizabeth felt that she must go, and very unwillingly said so.",
        "Obama was born on August 4, 1961, at Kapiolani Medical Center for Women and Children in Honolulu, Hawaii.",
        "He was born to an American mother of European descent and an African father."
    ]
    extractor = RelationExtractor()
    results = extractor.extract_relations(texts)
    for text, relations in results.items():
        print(f"Text: {text}")
        for relation in relations:
            print(f"  Relation: {relation}")
        print()
"""
