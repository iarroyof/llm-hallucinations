import spacy
from spacy.matcher import Matcher

class RelationExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        self._add_patterns()

    def _add_patterns(self):
        # Define the syntactical constraint patterns
        patterns = [
            [{"POS": "VERB"}, {"POS": "PART", "OP": "*"}, {"POS": "ADV", "OP": "*"}],
            [{"POS": "VERB"}, {"POS": "ADP", "OP": "*"}, {"POS": "DET", "OP": "*"},
             {"POS": "AUX", "OP": "*"}, {"POS": "ADJ", "OP": "*"}, {"POS": "ADV", "OP": "*"}]
        ]
        for pattern in patterns:
            self.matcher.add("SYNTACTIC_CONSTRAINT", [pattern])

    def _merge_overlapping_consecutive_spans(self, spans):
        if not spans:
            return []
        spans = sorted(spans, key=lambda x: x.start)
        merged_spans = []
        current_span = spans[0]
        for span in spans[1:]:
            if span.start <= current_span.end:
                current_span = spacy.tokens.Span(
                    current_span.doc, current_span.start, span.end, label=current_span.label
                )
            else:
                merged_spans.append(current_span)
                current_span = span
        merged_spans.append(current_span)
        return merged_spans

    def _find_longest_span(self, spans):
        if not spans:
            return None
        return max(spans, key=lambda x: len(x))

    def _extract_relations_from_sentence(self, sentence):
        doc = self.nlp(sentence)
        matches = self.matcher(doc)
        spans = [doc[start:end] for _, start, end in matches]
        merged_spans = self._merge_overlapping_consecutive_spans(spans)
        longest_span = self._find_longest_span(merged_spans)
        if not longest_span:
            return []
        # Find the closest noun before and after the longest span
        left_noun = None
        right_noun = None
        for token in doc[:longest_span.start]:
            if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                left_noun = token
        for token in doc[longest_span.end:]:
            if token.pos_ == "NOUN" or token.pos_ == "PROPN":
                right_noun = token
                break
        if left_noun and right_noun:
            return [(left_noun.text, longest_span.text, right_noun.text)]
        return []

    def extract_relations(self, texts):
        relations = []
        for text in texts:
            doc = self.nlp(text)
            for sentence in doc.sents:
                relations.extend(self._extract_relations_from_sentence(sentence.text))
        return relations

# Example usage
if __name__ == "__main__":
    texts = [
        "In July 2012, Ancestry.com found a strong likelihood that Dunham was descended from John Punch.",
        "Elizabeth was glad to be taken to her immediately."
    ]
    extractor = RelationExtractor()
    relations = extractor.extract_relations(texts)
    for relation in relations:
        print(f"Relation: {relation}")
