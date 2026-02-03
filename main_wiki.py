import os
import json
from wikipedia_retriever import WikipediaSearch
from relation_extraction_v5 import SolarRelationExtractor
from semantic_verification_v05 import SemanticVerifier

def comparar_hechos(query_text, nvidia_key, google_key):
    print(f"\nINICIANDO VERIFICACIÓN PARA: '{query_text}'")
    
    wiki = WikipediaSearch()
    
    # ESTRATEGIA DE BÚSQUEDA:
    # En lugar de buscar la frase completa ("Obama was born in Kenya"), 
    # se extrae la entidad principal ("Barack Obama") para obtener la biografía limpia.
    # Una heurística simple es tomar las primeras 2 palabras si es un nombre propio.
    search_topic = "Barack Obama" 
    
    print(f"Buscando Ground Truth en Wikipedia para: '{search_topic}'...")
    ground_truth_text = wiki.get_background_knowledge(search_topic, max_chars=1500)
    
    if not ground_truth_text:
        print("No se encontró información en Wikipedia.")
        return


    print("Extrayendo Relaciones (Solar)...")
    extractor = SolarRelationExtractor(api_key=nvidia_key)


    gt_rels = list(extractor.extract_relations([ground_truth_text]).values())[0]
    
    hyp_rels = list(extractor.extract_relations([query_text]).values())[0]

    gt_str = json.dumps(gt_rels)
    hyp_str = json.dumps(hyp_rels)
    
    print(f"   Hechos en Wikipedia: {len(gt_rels)}")
    print(f"   Hechos en Hipótesis: {len(hyp_rels)}")


    print("Juez de Alucinaciones (Verifier)...")
    verifier = SemanticVerifier(model_name="gemini-2.5-flash", api_key=google_key)
    
    raw_result = verifier.verify_text(
        wiki_relations=gt_str,  # La verdad estructurada
        relations_ans=hyp_str,  # La mentira estructurada
        ans=query_text          # El texto original
    )
    
    result = verifier.parse_model_output(raw_result)
    

    print("\n" + "="*40)
    print("RESULTADO DE LA COMPARACIÓN")
    print("="*40)
    print(f"Hipótesis: {query_text}")
    print(f"Veredicto del Modelo: {result.marked_text}")
    print(f"Inconsistencias: {result.inconsistencies}")
    print(f"Confianza: {result.confidence_score}")

if __name__ == "__main__":
    NVIDIA_KEY = os.environ.get("NVIDIA_API_KEY")
    GOOGLE_KEY = os.environ.get("GOOGLE_API_KEY")
    
    if not NVIDIA_KEY or not GOOGLE_KEY:
        print("NO API KEY")
    else:
        texto_alucinado = "Barack Obama was born in Kenya in 1961."
        comparar_hechos(texto_alucinado, NVIDIA_KEY, GOOGLE_KEY)