import torch
import json
import re
import os
import time

# IMPORTACIONES NECESARIAS PARA DEEPSEEK (usando Hugging Face Transformers)
from transformers import AutoTokenizer, AutoModelForCausalLM

# IMPORTACIONES DE MÓDULOS CUSTOM (asegúrate de que estos archivos estén en el contenedor)
from relation_extraction_v2 import RelationExtractor, extract_relations
from json_utils import JSONLIterator
from wiki_sample import WikipediaBatchGenerator
from pdb import set_trace as st

# Variable para el modelo Deepseek (puedes configurar su valor vía variable de entorno)
DEEPSEEK_MODEL_KEY = os.environ.get('DEEPSEEK_MODEL_KEY', '')

def extract_specific_json(text):
    """
    Extrae una estructura JSON específica con las llaves:
    'text_to_verify', 'inconsistency_identification' y 'explanation'
    de un texto más grande.
    """
    try:
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

# Parámetros y configuración
file_path = 'train/mushroom.en-train_nolabel.v1.jsonl'
keys = ['model_input', 'model_output_text']
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 'deepseek-ai/deepseek-r1-distill-qwen-1.5b'

# 1) OBTENCIÓN DE DATOS
# Se utiliza WikipediaBatchGenerator para obtener información complementaria
generator = WikipediaBatchGenerator()
n_qs_semantic_search_results = generator.get_batches()  # Lista de lotes de información
n_qs = generator.len  # Número de preguntas

# Extraemos preguntas y respuestas desde el archivo JSONL
questions_answers = JSONLIterator(file_path, keys, n_qs)
answers = [ans for _, ans in questions_answers]

# 2) EXTRAER RELACIONES
# Se instancia el extractor de relaciones
extractor = RelationExtractor()
wiki_docs_fquestion_relations, fanswer_relations = extract_relations(
    answers,
    n_qs_semantic_search_results,
    extractor
)

# 3) VERIFICACIÓN SEMÁNTICA CON DEEPSEEK
# Definición de la clase verificador semántico utilizando el modelo Deepseek R1
class SemanticVerifierDeepseek:
    def __init__(self, model_name, device, model_key):
        self.device = device
        self.model_key = model_key  # Actualmente no se utiliza, pero se deja para futura configuración
        print(f"Cargando tokenizador y modelo: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        print("Modelo Deepseek R1 cargado correctamente.")

    def verify_text(self, wiki_relations, answer_relations, answer):
        """
        Combina la información de las relaciones extraídas y genera una respuesta utilizando Deepseek R1.
        """
        prompt = (
            f"Wiki info: {wiki_relations}\n"
            f"Answer relations: {answer_relations}\n"
            f"Answer: {answer}\n"
            "Verifica la coherencia y explica posibles inconsistencias:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs = self.model.generate(**inputs, max_length=256, num_return_sequences=1)
        result_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"result": result_text, "prompt": prompt}

# Instanciar el verificador semántico usando Deepseek R1
verifier = SemanticVerifierDeepseek(
    model_name=model_name,
    device=device,
    model_key=DEEPSEEK_MODEL_KEY
)

# 4) PROCESAMIENTO DE CONSULTAS CON CONTROL DE TASA
MAX_REQUESTS_PER_MINUTE = 150
request_count = 0
start_time = time.time()

results = []

for wiki_relations, answer_relations, answer in zip(
        wiki_docs_fquestion_relations,
        fanswer_relations,
        answers
    ):
    # Control de tasa para no exceder el límite
    if request_count >= MAX_REQUESTS_PER_MINUTE:
        elapsed = time.time() - start_time
        if elapsed < 60:
            time.sleep(60 - elapsed)
        request_count = 0
        start_time = time.time()

    # Llamada al modelo Deepseek R1
    result = verifier.verify_text(wiki_relations, answer_relations, answer)
    request_count += 1

    print("Dictionary:")
    print(result)

    # (Opcional) Si deseas extraer un JSON específico de la respuesta, descomenta lo siguiente:
    # parsed_json = extract_specific_json(result["result"])
    # if parsed_json:
    #     print("JSON extraído:", parsed_json)

    results.append(result)

# Al finalizar, 'results' contendrá todas las respuestas del verificador.