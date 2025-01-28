from relation_extraction_v2 import RelationExtractor, extract_relations
from semantic_verification_v04 import SemanticVerifier
from json_utils import JSONLIterator
from wiki_sample import WikipediaBatchGenerator
from pdb import set_trace as st
import torch
        

file_path = 'train/mushroom.en-train_nolabel.v1.jsonl'
keys = ['model_input', 'model_output_text']
device = "cuda" if torch.cuda.is_available() else "cpu"
# Simulate a list of documents from n questions (n_qs),
# here, each question qi has associated a batch of m documents
# n_qs_semantic_search_results = [m_documents_q0, m_documents_q1,..., m_documents_qn]
# m_documents_qi = [doc0_from_wiki, doc1_from_wiki,..., docm_from_wiki]
generator = WikipediaBatchGenerator()
n_qs_semantic_search_results = generator.get_batches()
n_qs = generator.len
questions_answers = JSONLIterator(file_path, keys, n_qs)
extractor = RelationExtractor()
# Iterate over the file and process each item
answers = [ans for _, ans in questions_answers] 
wiki_docs_fquestion_relations, fanswer_relations = extract_relations(
        answers,
        n_qs_semantic_search_results,
        extractor)
# Take wiki_docs_fquestion_relations and fanswer_relations and give them to the semantic verifier.
    # Initialize verifier
verifier = SemanticVerifier(model_name="meta-llama/Llama-3.2-1B-Instruct", device=device)
    
    # Run verification
results = []
for wiki_relations, answer_relations, answer in zip(wiki_docs_fquestion_relations,
                                                        fanswer_relations,
                                                        answers):
        result, explanation = verifier.verify_text(wiki_relations, answer_relations, answer)    
    # Print results
        print("Dictionary:")
        print(result)
        print("\nExplanation:")
        print(explanation)
        results.append(result)
        st()
