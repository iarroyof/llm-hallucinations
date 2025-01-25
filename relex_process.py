from relation_extraction_v2 import RelationExtractor, extract_relations
from json_utils import JSONLIterator
from wiki_sample import WikipediaBatchGenerator
from pdb import set_trace as st
        

file_path = 'train/mushroom.en-train_nolabel.v1.jsonl'
keys = ['model_input', 'model_output_text']
# Simulate a list of documents from n questions (n_qs),
# here, each question has associated a batch of documents
# n_qs_semantic_search_results = [m_documents_q0, m_documents_q1,..., m_documents_qn]
# m_documents_q0 = [doc0_from_wiki, doc1_from_wiki,..., docm_from_wiki]
generator = WikipediaBatchGenerator()
n_qs_semantic_search_results = generator.get_batches()
n_qs = generator.len
questions_answers = JSONLIterator(file_path, keys, n_qs)
extractor = RelationExtractor()
# Iterate over the file and process each item
answers = [ans for _, ans in questions_answers] 
wiki_docs_fquestion_relations, fanswer_relations = extract_relations(answers, n_qs_semantic_search_results, extractor)
# Take wiki_docs_fquestion_relations and fanswer_relations and give them to the semantic verifier.
st()
