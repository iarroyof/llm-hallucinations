import stanza

class OpenIERelationExtractor:
    def __init__(self):
        # Initialize the Stanza pipeline with OpenIE
        self.nlp = stanza.Pipeline('en', processors='tokenize,openie')

    def extract_relations(self, texts):
        results = {}
        for text in texts:
            doc = self.nlp(text)
            relations = []
            for sentence in doc.sentences:
                for triple in sentence.openie:
                    relations.append((triple.subject, triple.predicate, triple.object))
            results[text] = relations
        return results

# Example usage
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
    extractor = OpenIERelationExtractor()
    results = extractor.extract_relations(texts)
    for text, relations in results.items():
        print(f"Text: {text}")
        for relation in relations:
            print(f"  Relation: {relation}")
        print()
