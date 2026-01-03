import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any
from dataclasses import dataclass
from huggingface_hub import login
import logging
import re
import os
import json
import google.generativeai as genai


@dataclass
class SemanticRelation:
    subject: str
    predicate: str
    object: str
    confidence: float
    source_doc: str

@dataclass
class VerificationResult:
    original_text: str
    marked_text: str
    inconsistencies: List[Dict[str, str]]
    confidence_score: float

class SemanticVerifier:
    def __init__(self, model_name: str, device: str="cuda", authenticate:bool=False, api_key:str=None):
        self.api_key = api_key
        
        if not api_key is None:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.device = "cloud" 
            return

        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        logging.info(f"Loading model {model_name} on {self.device}")
        
        if authenticate:
            with open('hf_token.txt') as f:
                token = f.readline().strip()
            login(token=token)
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _format_relations(self, relations: List[SemanticRelation]) -> str:
        formatted = "Semantic Relations:\n"
        if isinstance(relations, str):
            return relations
        for i, rel in enumerate(relations, 1):
            formatted += f"{i}. {rel.subject} {rel.predicate} {rel.object} (confidence: {rel.confidence:.2f})\n"
        return formatted

    def _create_verification_prompt(self, relations: Any, text: str, text_rels:Any=None, text_form_relations=True) -> str:
        if not text_form_relations:
            relations_text = self._format_relations(relations)
            ans_relations_text = self._format_relations(text_rels)
        else:
            relations_text = relations
            ans_relations_text = text_rels

        try:
            with open('prompt_template.txt', 'r') as f:
                prompt_base = f.read()
        except FileNotFoundError:
            prompt_base = "Analyze inconsistencies between the Text and Ground Truth relations."

        prompt = prompt_base + f"""\nInput:
            1. Text to check: {text}
            2. Relations extracted from the text: {ans_relations_text}
            3. Ground truth relations: {relations_text}
            Response:
            """
        return prompt

    def parse_model_output(self, output: str, original_text: str=None) -> VerificationResult:
        """
        Versión robusta para leer JSON directo o texto.
        """
        try:
            clean_output = output.replace("```json", "").replace("```", "").strip()
            
            try:
                data = json.loads(clean_output)
                return VerificationResult(
                    original_text=original_text,
                    marked_text=data.get("marked_text", original_text),
                    inconsistencies=data.get("hard_labels", []),
                    confidence_score=1.0
                )
            except json.JSONDecodeError:
                pass 

            inconsistency_pattern = r'inconsistency_identification:\s*(\{.*?\})'
            explanation_pattern = r'explanation:\s*(.*?)(?=\n\S+:|$)'
            
            inconsistency_match = re.search(inconsistency_pattern, output, re.DOTALL)
            explanation_match = re.search(explanation_pattern, output, re.DOTALL)
            
            inconsistencies = inconsistency_match.group(1).strip() if inconsistency_match else []
            explanation = explanation_match.group(1).strip() if explanation_match else "No explanation found"

            return VerificationResult(
                original_text=original_text,
                marked_text=original_text,
                inconsistencies=inconsistencies,
                confidence_score=0.5
            )

        except Exception as e:
            logging.error(f"Error parsing model output: {e}")
            return VerificationResult(
                original_text=original_text,
                marked_text=original_text,
                inconsistencies=[],
                confidence_score=0.0
            )

    def verify_text(self, wiki_relations: Any, relations_ans:Any, ans: str, beam:bool=False) -> str:
        prompt = self._create_verification_prompt(wiki_relations, ans, relations_ans)
        
        if self.api_key is not None:
           try:
               response = self.model.generate_content(prompt)
               return response.text
           except Exception as e:
               return f"API Error: {str(e)}"

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True).to(self.device)

        with torch.no_grad():
            if beam:
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=4096,
                    num_beams=5,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            else:
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=4096,
                    temperature=0.5,
                    top_p=0.95,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

if __name__ == "__main__":
    ground_truth_rels = """
    1. Titanic sank on April 15, 1912 (confidence: 1.0)
    2. Titanic collided with an iceberg (confidence: 1.0)
    3. Titanic was a British passenger liner (confidence: 1.0)
    """
    

    hypothesis_rels = """
    1. Titanic sank in 1999 (confidence: 0.95)
    2. Titanic hit a Kraken (confidence: 0.8)
    3. Titanic was an American warship (confidence: 0.9)
    """

    texto_a_verificar = "The Titanic was a huge American warship that sank in 1999 after hitting a giant Kraken."

    print("\n--- INICIANDO PRUEBA ---")
    
    google_api_key = os.environ.get("GOOGLE_API_KEY")

    if not google_api_key:
        print("\nERROR: No se encontró la variable de entorno 'GOOGLE_API_KEY'.")
        exit(1)
    
    print("Usando gemini-2.5-flash...")
    
    verifier = SemanticVerifier(
        model_name="gemini-2.5-flash", 
        api_key=google_api_key
    )

    print("\nVerificando texto...")
    try:
        resultado = verifier.verify_text(
            wiki_relations=ground_truth_rels,  # El texto verdadero
            relations_ans=hypothesis_rels,     # Lo que se extrajo del texto falso
            ans=texto_a_verificar              # El texto falso
        )
        
        parsed = verifier.parse_model_output(resultado)
        print("\n=== PARSEO FINAL ===")
        print(f"Texto Marcado: {parsed.marked_text}")
        print(f"Inconsistencias: {parsed.inconsistencies}")

    except Exception as e:
        print(f"Error durante la ejecución: {e}")