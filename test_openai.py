from relation_extraction_v2 import RelationExtractor, extract_relations_from_texts
from semantic_verification_v04 import SemanticVerifier  # Usado para Deepseek R1
from json_utils import JSONLIterator
from wiki_sample import WikipediaBatchGenerator
from wikipedia_str_search import WikipediaSearch
from pdb import set_trace as st
import torch
import json
import re
import os
import time

# Usaremos SOLAR_API_KEY para la integración con NVIDIA Solar.
SOLAR_API_KEY = os.environ.get('SOLAR_API_KEY')
RPM = 14

###############################################################################
# Clase para la verificación semántica usando NVIDIA Solar (modelo upstage/solar-10.7b-instruct)
###############################################################################
from openai import OpenAI

class SemanticVerifierSolar:
    def __init__(self, api_key):
        self.api_key = api_key
        print("Initializing NVIDIA Solar semantic verifier...")
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key
        )
        self.model_name = "upstage/solar-10.7b-instruct"
        self.temperature = 0.1
        self.top_p = 0.9
        self.max_tokens = 1024
        self.stream = True  # Si prefieres streaming, déjalo en True

    def verify_text(self, wiki_relations, answer_relations, answer):
        prompt = (
            f"Wiki info: {wiki_relations}\n"
            f"Answer relations: {answer_relations}\n"
            f"Answer: {answer}\n"
            "Verify hallucinations and provide a coherent explanation:"
        )
        messages = [{"role": "user", "content": prompt}]
        # Realiza la llamada al API de NVIDIA usando el cliente OpenAI
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stream=self.stream
        )
        # Acumula la respuesta (en modo streaming)
        response_text = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                response_text += chunk.choices[0].delta.content
        # Para mantener compatibilidad con la función de parseo,
        # creamos un objeto de respuesta "falso" que tenga la misma estructura esperada.
        FakeResponse = type("FakeResponse", (), {})  # Crea una clase dinámica
        fake_response = FakeResponse()
        FakeCandidate = type("FakeCandidate", (), {}) 
        FakeContent = type("FakeContent", (), {}) 
        FakePart = type("FakePart", (), {}) 
        fake_response.candidates = [FakeCandidate()]
        fake_response.candidates[0].content = FakeContent()
        fake_response.candidates[0].content.parts = [FakePart()]
        fake_response.candidates[0].content.parts[0].text = response_text
        return fake_response

###############################################################################
# Función de parseo de la respuesta (renombrada para claridad)
###############################################################################
def parse_solar_response(response):
    """
    Parsea el campo 'text' de la respuesta del API NVIDIA Solar.
    Extrae la cadena JSON contenida en un bloque de código y retorna un diccionario con:
    'hard_labels', 'soft_labels', 'explanation' y 'marked_text'.
    
    Retorna {} si la cadena JSON está vacía, o None si no se encuentra lo esperado.
    """
    try:
        candidates = response.candidates  # Acceso a candidates
        if candidates:
            content_parts = candidates[0].content.parts
            if content_parts:
                text_field = content_parts[0].text
                if text_field:
                    # Remueve los marcadores de bloque de código (asumiendo que se envuelve en ```json ... ```)
                    json_string = text_field.strip().removeprefix("```json\n").removesuffix("\n```")
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
                        return {}  # Retorna dict vacío si la cadena JSON está vacía
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

###############################################################################
# Función principal para verificar hallucinations y guardar el resultado
###############################################################################
def verify_hallucinations_and_save(index, q_search_results, answer, original_dict):
    relation_extractor = RelationExtractor()
    # Si SOLAR_API_KEY no está definida (o es vacía), se usa Deepseek R1.
    if SOLAR_API_KEY in [None, '']:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        deepseek_model_name = 'deepseek-ai/deepseek-r1-distill-qwen-1.5b'
        verifier = SemanticVerifier(model_name=deepseek_model_name, device=device, api_key=SOLAR_API_KEY)
        print(f"Working with Deepseek R1 semantic verifier: {deepseek_model_name}")
    else:
        verifier = SemanticVerifierSolar(api_key=SOLAR_API_KEY)
        print("Working with NVIDIA Solar semantic verifier: upstage/solar-10.7b-instruct")
    
    # Asegurarse de que q_search_results sea una lista.
    if not isinstance(q_search_results, list):
        q_search_results = [q_search_results]
    
    wiki_relations = relation_extractor.extract_relations(q_search_results)
    answer_relations = relation_extractor.extract_relations([answer])
    
    result = verifier.verify_text(wiki_relations, answer_relations, answer)
    parsed_result = parse_solar_response(result)
    if parsed_result:
        parsed_result.update(original_dict)
        try:
            with open(output_filename, "a" if index > 0 else "w", encoding="utf-8") as f:
                json.dump(parsed_result, f, ensure_ascii=False)
                f.write("\n")
            print(f"Data successfully written to {output_filename}")
        except Exception as e:
            print(f"Error writing to file: {e}")
    if (index + 1) % RPM == 0 and SOLAR_API_KEY not in [None, '']:
        st()
        print("Pausing for one minute...")
        time.sleep(60)  # Pausa de 60 segundos

###############################################################################
# PROCESAMIENTO PRINCIPAL
###############################################################################
file_path = "v1/mushroom.en-tst.v1.jsonl"  # Opción: 'train/mushroom.en-train_nolabel.v1.jsonl'
keys = None  # Si es None, se espera que cada entrada tenga 'model_input' y 'model_output_text'
questions_answers = JSONLIterator(file_path=file_path, keys=keys, n_samples=None)
relation_extractor = RelationExtractor()
searcher = WikipediaSearch(k=3)
results = []
output_filename = file_path + ".results"
i = 0

if keys is None:
    for i, q in enumerate(questions_answers):
        q_search_results = searcher.get_background_knowledge(q['model_input'])
        answer = q['model_output_text']
        verify_hallucinations_and_save(i, q_search_results, answer, q)
else:
    questions, answers = zip(*questions_answers)
    n_search_results = [searcher.get_background_knowledge(q) for q in questions]