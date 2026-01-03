import json
import stanza
from stanza.server import CoreNLPClient
from openai import OpenAI
import os

class OpenIERelationExtractor:
    def __init__(self):
        pass

    def extract_relations(self, texts):
        results = {}
        try:
            with CoreNLPClient(
                annotators=['tokenize','ssplit','pos','lemma','depparse','natlog','openie'],
                memory='4G', 
                endpoint='http://localhost:9001',
                be_quiet=True
            ) as client:
                
                for text in texts:
                    ann = client.annotate(text)
                    relations = []
                    
                    for sentence in ann.sentence:
                        for triple in sentence.openieTriple:
                            relations.append((triple.subject, triple.relation, triple.object))
                    
                    results[text] = relations
        except Exception as e:
            print(f"\nError crítico conectando con Java/CoreNLP: {e}")
            
        return results
    
class SolarRelationExtractor:
    def __init__(self, api_key, temperature=0.0, top_p=0.9, max_tokens=1024, stream=False):
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stream = stream 

        self.prompt = """Extract semantic relationships. Output strictly valid JSON list of objects with keys: "subject", "relation", "object". Text: '{}'"""

        self.client = OpenAI(
            base_url = "https://integrate.api.nvidia.com/v1",
            api_key = api_key
        )
    
    def extract_relations(self, texts):
        results = {}
        for text in texts:
            try:
                response = self.client.chat.completions.create(
                    model="upstage/solar-10.7b-instruct",
                    messages=[{'role': 'user', 'content': self.prompt.format(text.replace('"',"'"))}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    stream=False
                )
                
                content = response.choices[0].message.content
                clean_content = content.replace("```json", "").replace("```", "").strip()
                
                try:
                    results[text] = json.loads(clean_content)
                except json.JSONDecodeError:
                    results[text] = []

            except Exception as e:
                print(f"Error with Solar: {e}")
                results[text] = []
                
        return results

if __name__ == "__main__":
    api_key = os.environ.get("NVIDIA_API_KEY")

    texts = [
        "In July 2012, Ancestry.com found a strong likelihood that Dunham was descended from John Punch.",
        "Obama was born on August 4, 1961, at Kapiolani Medical Center for Women and Children in Honolulu, Hawaii."
    ]
    
    print("--- Probando Stanza (OpenIE) ---")
    extractor = OpenIERelationExtractor()
    results = extractor.extract_relations(texts)
    
    for text, relations in results.items():
        print(f"Text: {text}")
        for relation in relations:
            print(f"  Relation: {relation}")
        print("-" * 30)

    print("\n--- Probando Solar (LLM) ---")
    solar_extractor = SolarRelationExtractor(api_key=api_key)
    solar_results = solar_extractor.extract_relations(texts)
    print(solar_results)