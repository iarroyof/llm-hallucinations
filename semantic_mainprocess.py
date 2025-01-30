import torch
import json
import re
import os
import time

from relation_extraction_v2 import RelationExtractor, extract_relations
from semantic_verification_v04 import SemanticVerifier
from json_utils import JSONLIterator
from wiki_sample import WikipediaBatchGenerator
from pdb import set_trace as st

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

def extract_specific_json(text):
    """
    Extrae una estructura JSON específica con las llaves:
    'text_to_verify', 'inconsistency_identification' y 'explanation'
    de un texto más grande.

    Args:
        text (str): Texto de entrada que contiene la estructura JSON.

    Returns:
        dict: JSON parseado con la estructura específica o None si no se encuentra.
    """
    try:
        # Patrón que intenta capturar la estructura JSON con las claves requeridas
        pattern = (r'\{\s*"text_to_verify":\s*"[^"]+",'
                   r'\s*"inconsistency_identification":\s*\{[^}]+\},'
                   r'\s*"explanation":\s*"[^"]+"\s*\}')

        match = re.search(pattern, text)
        if not match:
            return None

        json_str = match.group(0)
        result = json.loads(json_str)

        required_keys = {'text_to_verify', 'inconsistency_identification', 'explanation'}
        if not all(key in result for key in required_keys):
            return None

        return result

    except (json.JSONDecodeError, AttributeError):
        return None

file_path = 'train/mushroom.en-train_nolabel.v1.jsonl'
keys = ['model_input', 'model_output_text']
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'deepseek-ai/deepseek-r1-distill-qwen-1.5b'

# Obtenemos lotes de Wikipedia
generator = WikipediaBatchGenerator()
n_qs_semantic_search_results = generator.get_batches()  # Lista de lotes
n_qs = generator.len  # Número de preguntas

# Extraemos iterador de JSONL
questions_answers = JSONLIterator(file_path, keys, n_qs)
answers = [ans for _, ans in questions_answers]

# Instanciamos el extractor de relaciones
extractor = RelationExtractor()

# Obtenemos relaciones de Wikipedia y del texto de respuesta
wiki_docs_fquestion_relations, fanswer_relations = extract_relations(
    answers,
    n_qs_semantic_search_results,
    extractor
)

# Inicializamos el verificador semántico
verifier = SemanticVerifier(
    model_name=model_name,
    device=device,
    api_key=GEMINI_API_KEY
)

# Preparamos control de tasa para no exceder 150 requests por minuto
MAX_REQUESTS_PER_MINUTE = 150
request_count = 0
start_time = time.time()

results = []

for wiki_relations, answer_relations, answer in zip(
        wiki_docs_fquestion_relations,
        fanswer_relations,
        answers
    ):
    # 1) Verificar si superamos el límite de peticiones
    if request_count >= MAX_REQUESTS_PER_MINUTE:
        elapsed = time.time() - start_time
        if elapsed < 60:
            # Esperamos hasta completar el minuto
            time.sleep(60 - elapsed)
        # Reiniciamos el conteo y el temporizador
        request_count = 0
        start_time = time.time()

    # 2) Llamada al modelo (1 request)
    result = verifier.verify_text(wiki_relations, answer_relations, answer)

    # 3) Aumentamos el contador de requests
    request_count += 1

    print("Dictionary:")
    print(result)

    # Opcionalmente, puedes extraer el JSON específico si así lo deseas
    # parsed_json = extract_specific_json(result)
    # if parsed_json:
    #     print("JSON extraído:", parsed_json)

    results.append(result)

# Al final, 'results' contendrá todas las respuestas del verificador