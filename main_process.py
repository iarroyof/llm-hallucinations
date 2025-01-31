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

def extract_specific_json(text):
    """
    Extract specific JSON structure containing text_to_verify, inconsistency_identification,
    and explanation from a larger text.

    Args:
        text (str): Input text containing the JSON structure

    Returns:
        dict: Parsed JSON with the specific structure or None if not found
    """
    try:
        # Pattern to match the specific JSON structure we want
        pattern = r'\{\s*"text_to_verify":\s*"[^"]+",\s*"inconsistency_identification":\s*\{[^}]+\},\s*"explanation":\s*"[^"]+"\s*\}'

        # Find the match
        match = re.search(pattern, text)
        if not match:
            return None

        # Parse the JSON
        json_str = match.group(0)
        result = json.loads(json_str)

        # Validate the structure
        required_keys = {'text_to_verify', 'inconsistency_identification', 'explanation'}
        if not all(key in result for key in required_keys):
            return None

        return result

    except (json.JSONDecodeError, AttributeError) as e:
        return None

file_path = 'train/mushroom.en-train_nolabel.v1.jsonl'
keys = ['model_input', 'model_output_text']
# Simulate a list of documents from n questions (n_qs),
# here, each question qi has associated a batch of m documents
# n_qs_semantic_search_results = [m_documents_q0, m_documents_q1,..., m_documents_qn]
# m_documents_qi = [doc0_from_wiki, doc1_from_wiki,..., docm_from_wiki]
#generator = WikipediaBatchGenerator()
#n_qs_semantic_search_results = generator.get_batches()
searcher = WikipediaSearch(k=3)

n_qs = 15
questions_answers = JSONLIterator(file_path, keys, n_qs)
extractor = RelationExtractor()
# Iterate over the file and process each item
# Search and get background knowledge based on titles:
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
    model_name = None
    
verifier = SemanticVerifier(model_name=model_name, device=device, api_key=GEMINI_API_KEY)
    # Run verification
results = []
# Adaptar este loop para que solo haga 150 requests por minuto a lo mucho, esperar a que inicie el siguiente minuto
# y seguir haciendo requests.
i = 0
for wiki_relations, answer_relations, answer in zip(wiki_docs_fquestion_relations,
                                                        fanswer_relations,
                                                        answers):
    result = verifier.verify_text(wiki_relations, answer_relations, answer)
    #print("Dictionary:")
    results.append(result)
    i += 1
    if i % 15 == 0 and GEMINI_API_KEY not in [None, '']:
        with open(file_path + '.results') as f:
            for r in results:
                f.write(r)
                f.write('\n')
        results = []
        print("Pausing for one minute...")
        time.sleep(60)  # Sleep for 60 seconds (1 minute)
