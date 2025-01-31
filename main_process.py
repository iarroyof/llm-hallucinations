from relation_extraction_v2 import RelationExtractor, extract_relations
from semantic_verification_v04 import SemanticVerifier
from json_utils import JSONLIterator
from wiki_sample import WikipediaBatchGenerator
from wikipedia_str_search import WikipediaSearch
from pdb import set_trace as st
import torch
import json
import re
import os
import time

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

def parse_gemini_response(response):
    """Parses the 'text' field from a Gemini API response.

    Args:
        response: The Gemini API response object (as shown in your example).

    Returns:
        A dictionary containing the extracted data ('hard_labels', 'soft_labels',
        'explanation', and 'marked_text'), or None if there's an error or
        the expected fields are not found.  Returns an empty dict if the json string is empty.
            
    # Example usage (assuming you have the 'response' object directly):
    extracted_data = parse_gemini_response(response)  # Pass the response object
    
    if extracted_data:
        print(extracted_data)
        # ... (rest of the example code remains the same)
    else:
        print("Failed to extract data from response.")

    """
    try:
        candidates = response.candidates  # Access candidates directly
        if candidates:
            content_parts = candidates[0].content.parts
            if content_parts:
                text_field = content_parts[0].text
                if text_field:
                     # Extract JSON string from the code block
                    json_string = text_field.strip().removeprefix("```json\n").removesuffix("\n```") #strip to remove whitespace and remove code block markers
                    if json_string:
                        try:
                            data = json.loads(json_string)
                            extracted_data = {
                                "hard_labels": data.get("hard_labels"),
                                "soft_labels": data.get("soft_labels"),
                                "explanation": data.get("explanation"),
                                "marked_text": data.get("marked_text"),
                            }
                            return extracted_data
                        except json.JSONDecodeError:
                            print("Error: Could not decode JSON string.")
                            return None
                    else:
                        return {} #return empty dict if json string is empty
                else:
                    print("Error: 'text' field not found in response.")
                    return None
            else:
                print("Error: 'parts' field not found in response.")
                return None
        else:
            print("Error: 'candidates' list is empty in response.")
            return None
    except AttributeError as e:
        print(f"Error: Invalid response structure. {e}")
        return None

file_path = 'train/mushroom.en-train_nolabel.v1.jsonl'
keys = None # ['model_input', 'model_output_text']
# Simulate a list of documents from n questions (n_qs),
# here, each question qi has associated a batch of m documents
# n_qs_semantic_search_results = [m_documents_q0, m_documents_q1,..., m_documents_qn]
# m_documents_qi = [doc0_from_wiki, doc1_from_wiki,..., docm_from_wiki]
#generator = WikipediaBatchGenerator()
#n_qs_semantic_search_results = generator.get_batches()
questions_answers = JSONLIterator(file_path=file_path, keys=keys, n_samples=7)
extractor = RelationExtractor()
# Iterate over the file and process each item
# Search and get background knowledge based on titles:
searcher = WikipediaSearch(k=3)
if keys is None:
    answers = []
    n_qs_semantic_search_results = []
    full_data = []
    for q in questions_answers:
        n_qs_semantic_search_results.append(
            searcher.get_background_knowledge(q['model_input']))
        answers.append(q['model_output_text'])
        full_data.append(q)
else:        
    questions, answers = zip(*questions_answers)
    n_qs_semantic_search_results = [searcher.get_background_knowledge(q) for q in questions]

wiki_docs_fquestion_relations, fanswer_relations = extract_relations(
        answers,
        n_qs_semantic_search_results,
        extractor)
# Take wiki_docs_fquestion_relations and fanswer_relations and give them to the semantic verifier.
    # Initialize verifier
if GEMINI_API_KEY in [None, '']:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = 'deepseek-ai/deepseek-r1-distill-qwen-1.5b'
else:
    device = None
    model_name = "gemini-1.5-flash"
print(f"Working with {model_name} semantic verifier...")
verifier = SemanticVerifier(model_name=model_name, device=device, api_key=GEMINI_API_KEY)
    # Run verification
results = []
# Adaptar este loop para que solo haga 150 requests por minuto a lo mucho, esperar a que inicie el siguiente minuto
# y seguir haciendo requests.
filename = file_path + ".results"
i = 0
rpm = 15
for wiki_relations, answer_relations, answer, result_data in zip(wiki_docs_fquestion_relations,
                                                        fanswer_relations,
                                                        answers, full_data):
    result = verifier.verify_text(wiki_relations, answer_relations, answer)
    result = parse_gemini_response(result)
    if result:
        results.append(result.update(result_data))
    i += 1
    st()
    if i % rpm == 0 and GEMINI_API_KEY not in [None, '']:
        try:
            with open(filename, "w" if i < rpm else 'a', encoding="utf-8") as f:  # Use UTF-8 encoding
                json.dump(results, f, indent=4, ensure_ascii=False) #indent for readability, ensure_ascii handles special characters
                print(f"Data successfully written to {filename}")
        except Exception as e:
            print(f"Error writing to file: {e}")
        results = []
        print("Pausing for one minute...")
        time.sleep(60)  # Sleep for 60 seconds (1 minute)
