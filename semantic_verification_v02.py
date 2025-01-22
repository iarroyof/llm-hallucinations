import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict
from dataclasses import dataclass
import logging

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
    def __init__(self, model_name: str = "mosaicml/mpt-7b-instruct", device: str = "cuda"):
        """
        Initialize the semantic verifier with the MPT-7B-Instruct model.
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        logging.info(f"Loading model {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
    
    def _format_relations(self, relations: List[SemanticRelation]) -> str:
        """Format semantic relations into a string for the prompt."""
        formatted = "Semantic Relations:\n"
        for i, rel in enumerate(relations, 1):
            formatted += f"{i}. {rel.subject} {rel.predicate} {rel.object} (confidence: {rel.confidence:.2f})\n"
        
        return formatted
    
    def _create_verification_prompt(self, relations, text) -> str:
        """Create the verification prompt for the model."""
        #relations_text = self._format_relations(relations)
        print("entró a _create_verification_prompt")
        relations_text = "1.".join(relations)
        prompt = f"""
          Task: Analyze the following text for semantic inconsistencies using the provided semantic relations.

          {relations_text}

          Text to verify:
          {text}

          Instructions:
          1. Compare the text against the semantic relations
          2. Identify any inconsistencies or contradictions
          3. Mark inconsistent segments with XML tags
          4. Provide a brief explanation for each inconsistency
          
          Output format:
          1. First provide the text with <inconsistency> tags around problematic segments
          2. Then list each inconsistency with its explanation
          
          Response:"""
        return prompt
    
    def _parse_model_output(self, output: str, original_text: str) -> VerificationResult:
        """Parse the model's output into a structured format."""
        try:
            parts = output.split("\n\nInconsistencies found:")
            marked_text = parts[0].strip()
            
            inconsistencies = []
            if len(parts) > 1:
                explanations = parts[1].strip().split("\n")
                for exp in explanations:
                    if exp.strip():
                        inconsistencies.append({
                            "text": exp.split(": ")[0],
                            "explanation": exp.split(": ")[1]
                        })
            
            confidence_score = 1.0 - (len(inconsistencies) * 0.1)
            confidence_score = max(0.0, min(1.0, confidence_score))
            
            return VerificationResult(
                original_text=original_text,
                marked_text=marked_text,
                inconsistencies=inconsistencies,
                confidence_score=confidence_score
            )
        except Exception as e:
            logging.error(f"Error parsing model output: {e}")
            return VerificationResult(
                original_text=original_text,
                marked_text=original_text,
                inconsistencies=[],
                confidence_score=0.0
            )
    
    def verify_texts(self, relations, texts) -> List[VerificationResult]:
        """
        Verify a list of texts against semantic relations and identify inconsistencies.
        
        Args:
            relations: List of semantic relations extracted from reference documents
            texts: List of texts to verify
            
        Returns:
            List of VerificationResult for each input text
        """
        results = []
        for text in texts:
            print("entró al for y va con _create_verification_prompt")
            prompt = self._create_verification_prompt(relations, text)
            print("regresó prompt desde _create_verification_prompt")
            print("prompt:",prompt)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            print("hizo el tokenizer")
            print("inputs:",inputs)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=4096,
                    temperature=0.1,
                    top_p=0.95,
                    do_sample=True,
                    num_return_sequences=1
                )
            print("aplica metodo decode de tokenizer")
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("aplica el modelo")
            result = self._parse_model_output(response, text)
            print("sale del modelo")
            results.append(result)
        
        return results
    
if __name__ == "__main__":
    # Relaciones semánticas de ejemplo
    """""
    relations = [
        SemanticRelation(
            subject="Earth",
            predicate="has diameter",
            object="12,742 kilometers",
            confidence=0.95,
            source_doc="doc1"
        ),
        SemanticRelation(
            subject="Earth",
            predicate="orbits",
            object="Sun",
            confidence=0.98,
            source_doc="doc1"
        )
    ]"""
    ext_rel_resp = (['I write letter', 'They spend money', 'Juan eats apple', 'We saved money', 'Pedro sent email'], ['I write a romantic letter.', 'They spend much money', 'Juan eats a delicious apple.', 'We saved much money in the bank', 'Pedro sent an email to Ana.'])
    relations = (ext_rel_resp[0][0])
    text_to_verify = (ext_rel_resp[1][0])
    # Textos a verificar
    #text_to_verify = [
    #    """The Earth, which has a diameter of about 10,000 kilometers, 
    #    completes one orbit around the Sun every 365.25 days.""",
    #    """Mars, which is smaller than Earth, orbits the Sun every 687 days."""
    #]
    
    # Inicializar el verificador
    verifier = SemanticVerifier(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Verificar textos
    print("verifica textos")
    results = verifier.verify_texts(relations, text_to_verify)
    print("imprime resultados del verificador")
    
    # Mostrar resultados
    for result in results:
        print("\nOriginal Text:", result.original_text)
        print("Marked Text:", result.marked_text)
        print("Inconsistencies:")
        for inc in result.inconsistencies:
            print(f"  - {inc['text']}: {inc['explanation']}")
        print(f"Confidence Score: {result.confidence_score:.2f}")