from relation_extraction_v2 import RelationExtractor
from json_utils import JSONLIterator
from wiki_sample import WikipediaBatchGenerator
from pdb import set_trace as st

def extract_relations(answers, n_qs_semantic_search_results, extractor):
    wiki_docs_fquestion_relations = []
    fanswer_relations = []
    for wiki_docs_from_question, answer in zip(n_qs_semantic_search_results, answers):
        wiki_docs_fquestion_relations.append(extractor.extraer_relaciones(wiki_docs_from_question))
        fanswer_relations.append(extractor.extraer_relaciones([answer]))

    return wiki_docs_fquestion_relations, fanswer_relations
        

file_path = 'train/mushroom.en-train_nolabel.v1.jsonl'
keys = ['model_input', 'model_output_text']
# Simulate a list of documents from n questions (n_qs)
generator = WikipediaBatchGenerator()
n_qs_semantic_search_results = generator.get_batches()
n_qs = generator.len
questions_answers = JSONLIterator(file_path, keys, n_qs)
extractor = RelationExtractor()
# Iterate over the file and process each item
answers = [ans for _, ans in questions_answers] 
wiki_docs_fquestion_relations, fanswer_relations = extract_relations(answers, n_qs_semantic_search_results, extractor)
st()
