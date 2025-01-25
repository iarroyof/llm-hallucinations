from relation_extraction_v2 import RelationExtractor, extract_relations
from semantic_verifier_v04 import SemanticVerifier
from json_utils import JSONLIterator
from wiki_sample import WikipediaBatchGenerator
from pdb import set_trace as st
        

file_path = 'train/mushroom.en-train_nolabel.v1.jsonl'
keys = ['model_input', 'model_output_text']
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
wiki_docs_fquestion_relations, fanswer_relations = extract_relations(answers, n_qs_semantic_search_results, extractor)
# Take wiki_docs_fquestion_relations and fanswer_relations and give them to the semantic verifier.
    # Initialize verifier
verifier = SemanticVerifier(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Run verification
result = verifier.verify_text(relations_from_q, relations_a, answer)
    
    # Print results
print("Marked text:")
print(result.marked_text)
print("\nInconsistencies found:")
for inc in result.inconsistencies:
    print(f"- {inc['text']}: {inc['explanation']}")
    print(f"\nConfidence score: {result.confidence_score:.2f}")
st()
