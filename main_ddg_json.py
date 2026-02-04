import json
import os
import time
from relation_extraction_v5 import SolarRelationExtractor
from semantic_verification_v05 import SemanticVerifier
from ddg_retriever import find_evidence

INPUT_FILE = "val/mushroom.en-val.v2.jsonl"
OUTPUT_FILE = "resultados_simplificados.jsonl"
MAX_EXAMPLES = 5

class HallucinationDetector:
    def __init__(self, nvidia_key, google_key):
        self.extractor = SolarRelationExtractor(api_key=nvidia_key)
        self.verifier = SemanticVerifier(model_name="gemini-2.5-flash", api_key=google_key)

    def find_span_in_text(self, text, triple):
        subj = triple.get('subject', '')
        obj = triple.get('object', '')
        
        start_idx = text.find(subj)
        end_idx = text.find(obj)
        
        if start_idx != -1 and end_idx != -1:
            real_start = min(start_idx, end_idx)
            end_pos_subj = start_idx + len(subj)
            end_pos_obj = end_idx + len(obj)
            real_end = max(end_pos_subj, end_pos_obj)
            return [real_start, real_end]
        
        return None

    def run(self):
        if not os.path.exists(INPUT_FILE):
            print(f"File not found: {INPUT_FILE}")
            return

        results_to_save = []

        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                if count >= MAX_EXAMPLES:
                    break

                try:
                    data = json.loads(line.strip())
                    
                    q = data.get('model_input')
                    ans = data.get('model_output_text')
                    
                    print(f"ID: {data.get('id')}")

                    evidence_list = find_evidence(q, max_results=3, max_chars=1500)
                    evidence_text = "\n".join(evidence_list)
                    
                    extracted = self.extractor.extract_relations([ans])
                    triples = list(extracted.values())[0]

                    predicted_spans = []
                    is_hallucination = False

                    if triples and evidence_text:
                        for triple in triples:
                            claim = f"{triple.get('subject')} {triple.get('relation')} {triple.get('object')}"
                            
                            raw_result = self.verifier.verify_text(
                                wiki_relations=f"Evidence:\n{evidence_text}",
                                relations_ans=f"Claim: {claim}",
                                ans=ans
                            )
                            
                            parsed = self.verifier.parse_model_output(raw_result, original_text=claim)

                            if parsed.inconsistencies and parsed.inconsistencies != "[]":
                                print(f"  HALLUCINATION: {claim}")
                                is_hallucination = True
                                span = self.find_span_in_text(ans, triple)
                                if span:
                                    predicted_spans.append(span)
                            else:
                                print(f"  CORRECT: {claim}")
                    elif not evidence_text:
                        print("  NO EVIDENCE FOUND")

                    output_obj = {
                        "id": data.get('id'),
                        "model_input": q,
                        "model_output_text": ans,
                        "soft_labels": data.get('soft_labels', []),
                        "hard_labels": data.get('hard_labels', []),
                        "predicted_hard_labels": predicted_spans,
                        "predicted_is_hallucination": is_hallucination
                    }
                    
                    results_to_save.append(output_obj)
                    count += 1
                    time.sleep(1)

                except json.JSONDecodeError:
                    continue

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            for item in results_to_save:
                f_out.write(json.dumps(item) + "\n")
        
        print(f"Finished. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")

    if not nvidia_key or not google_key:
        print("Missing API KEYS")
        exit(1)

    app = HallucinationDetector(nvidia_key, google_key)
    app.run()