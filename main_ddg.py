import json
import os
import time
from ddgs import DDGS
from relation_extraction_v5 import SolarRelationExtractor
from semantic_verification_v05 import SemanticVerifier

# Configuración
INPUT_FILE = "val/mushroom.en-val.v2.jsonl"
MAX_EXAMPLES = 5

class SimplePipeline:
    def __init__(self, nvidia_key, google_key):
        self.extractor = SolarRelationExtractor(api_key=nvidia_key)
        self.verifier = SemanticVerifier(model_name="gemini-2.5-flash", api_key=google_key)
        self.ddgs = DDGS()

    def get_evidence(self, query):
        """Busca evidencia en DuckDuckGo sin reintentos complejos."""
        try:
            results = self.ddgs.text(query, max_results=3)
            if results:
                return "\n".join([r['body'] for r in results])
        except Exception:
            pass
        return ""

    def analyze_example(self, data):
        """Analiza un solo ejemplo del dataset."""
        print(f"ID: {data.get('id')}")
        print(f"Pregunta: {data.get('model_input')}")
        text_to_check = data.get('model_output_text')
        print(f"Texto a verificar: {text_to_check}")

        # 1. Extraer relaciones (Claims)
        extracted = self.extractor.extract_relations([text_to_check])
        triples = list(extracted.values())[0]

        if not triples:
            print("  -> No se detectaron afirmaciones para verificar.")
            return

        # 2. Verificar cada afirmación
        hallucinations_found = 0
        
        for triple in triples:
            claim = f"{triple.get('subject')} {triple.get('relation')} {triple.get('object')}"
            
            # Buscar evidencia
            evidence = self.get_evidence(claim)
            
            if not evidence:
                print(f"  [SIN EVIDENCIA] {claim}")
                continue

            # Verificar con Gemini
            # Se pasa la evidencia web como 'wiki_relations' y el claim como 'relations_ans'
            raw_result = self.verifier.verify_text(
                wiki_relations=f"Evidence: {evidence}",
                relations_ans=f"Claim: {claim}",
                ans=text_to_check
            )
            
            parsed = self.verifier.parse_model_output(raw_result, original_text=claim)

            if parsed.inconsistencies and parsed.inconsistencies != "[]":
                print(f"  [ALUCINACION] {claim}")
                hallucinations_found += 1
            else:
                print(f"  [CORRECTO] {claim}")

        # Mostrar etiquetas reales para comparar
        print(f"Etiquetas reales (hard_labels): {data.get('hard_labels')}")
        print("-" * 50)

    def run(self):
        if not os.path.exists(INPUT_FILE):
            print(f"Error: No se encuentra el archivo {INPUT_FILE}")
            return

        print(f"Procesando {MAX_EXAMPLES} ejemplos de {INPUT_FILE}...\n")
        
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                if count >= MAX_EXAMPLES:
                    break
                try:
                    data = json.loads(line.strip())
                    self.analyze_example(data)
                    count += 1
                    time.sleep(1) # Pausa breve para evitar límites de API
                except json.JSONDecodeError:
                    continue

if __name__ == "__main__":
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")

    if not nvidia_key or not google_key:
        print("Error: Faltan las API KEYS.")
        exit(1)

    pipeline = SimplePipeline(nvidia_key, google_key)
    pipeline.run()