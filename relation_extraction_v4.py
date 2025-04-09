import stanza
from openai import OpenAI

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
    
class SolarRelationExtractor:
    def __init__(self, api_key, temperature=0.1, top_p=0.9, max_tokens=1024, stream=True):
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stream = stream

        self.promt = "Extract all the semantic relationships in the following text. Detect all subject and object free text phrases, inlcuding those not related to the main topics of the text, that contain generic and domain specific entities to build these semantic relationships. Your output should be in jsonl format. these relationships and format sould be tailored to be read by you in next steps of a sematic fact verification against other set of semantic relationships extracted from other text: This is the text to process: '{}'"

        self.client = OpenAI(
            base_url = "https://integrate.api.nvidia.com/v1",
            api_key = api_key
        )
    
    def extract_relations(self, texts):
        results = {}
        for text in texts:
            response = self.client.chat.completions.create(
                model="upstage/solar-10.7b-instruct",
                messages=[{'role': 'user', 'content': self.promt.format(text.replace('"',"'"))}],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=self.stream
            )
            tokens = []

            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    tokens.append(chunk.choices[0].delta.content)
            
            results[text] = ''.join(tokens)
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
