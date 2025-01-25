import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import xml.etree.ElementTree as ET
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
        Initialize the semantic verifier with MPT-7B-Instruct model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        logging.info(f"Loading model {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
    def _format_relations(self, relations: ist[SemanticRelation]) -> str:
        """Format semantic relations into a string for the prompt."""
        formatted = "Semantic Relations:\n"
        for i, rel in enumerate(relations, 1):
            formatted += f"{i}. {rel.subject} {rel.predicate} {rel.object} (confidence: {rel.confidence:.2f})\n"
        return formatted
    
    def _create_verification_prompt(self, relations: Any, text: str, text_form_relations=True) -> str:
        """Create the verification prompt for the model."""
        if not text_form_relations:
            relations_text = self._format_relations(relations)
        else:
            relations_text = text_form_relations
          
        print("relations:", relations_text)
        print("tipo de relations:", type(relations_text))
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
            # Split output into marked text and explanations
            parts = output.split("\n\nInconsistencies found:")
            marked_text = parts[0].strip()
            
            # Extract inconsistencies
            inconsistencies = []
            if len(parts) > 1:
                explanations = parts[1].strip().split("\n")
                for exp in explanations:
                    if exp.strip():
                        inconsistencies.append({
                            "text": exp.split(": ")[0],
                            "explanation": exp.split(": ")[1]
                        })
            
            # Calculate confidence score based on number and severity of inconsistencies
            confidence_score = 1.0 - (len(inconsistencies) * 0.1)  # Simple scoring mechanism
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
    
    def verify_text(self, relations: List[SemanticRelation], text: str) -> VerificationResult:
        """
        Verify text against semantic relations and identify inconsistencies.
        
        Args:
            relations: List of semantic relations extracted from reference documents
            text: Text to verify (A_s)
            
        Returns:
            VerificationResult containing marked text and identified inconsistencies
        """
        # Create the prompt
        prompt = self._create_verification_prompt(relations, text)
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=4096,
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=1
            )
        
        # Decode the output
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse and return results
        return self._parse_model_output(response, text)
