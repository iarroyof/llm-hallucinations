import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple, Any
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from huggingface_hub import login
import logging
import re
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
        """
        Initialize the semantic verifier with *-Instruct model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.api_key = api_key
        if not api_key is None:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            return None

        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        logging.info(f"Loading model {model_name} on {self.device}")
        # Replace with your Hugging Face token
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
        """Format semantic relations into a string for the prompt."""
        formatted = "Semantic Relations:\n"
        for i, rel in enumerate(relations, 1):
            formatted += f"{i}. {rel.subject} {rel.predicate} {rel.object} (confidence: {rel.confidence:.2f})\n"
        return formatted

    def _create_verification_prompt(self, relations: Any, text: str, text_rels:Any=None, text_form_relations=True) -> str:
        """Create the verification prompt for the model."""
        if not text_form_relations:
            relations_text = self._format_relations(relations)
            ans_relations_text = self._format_relations(text_rels)
        else:
            relations_text = relations
            ans_relations_text = text_rels

        prompt = open('prompt_template.txt', 'r').read()

        prompt = prompt + f"""\nInput:
            1. Text to check: {text}
            2. Relations extracted from the text: {ans_relations_text}
            3. Ground truth relations: {relations_text}
            Response:
            """
        return prompt

    def _parse_model_output(self, output: str, original_text: str=None) -> str:
        return output, original_text

    def parse_model_output(self, output: str, original_text: str=None) -> VerificationResult:
        # Define regex patterns to match the required sections
        try:
            inconsistency_pattern = r'inconsistency_identification:\s*(\{.*?\})'
            explanation_pattern = r'explanation:\s*(.*?)(?=\n\S+:|$)'
            # Search for the patterns in the text
            inconsistency_match = re.search(inconsistency_pattern, output, re.DOTALL)
            explanation_match = re.search(explanation_pattern, output, re.DOTALL)
            # Extract the matched groups
            inconsistency_text = inconsistency_match.group(1).strip() if inconsistency_match else None
            explanation_text = explanation_match.group(1).strip() if explanation_match else None

            return inconsistency_text, explanation_text

        except Exception as e:
            logging.error(f"Error parsing model output: {e}")
            return VerificationResult(
                original_text=original_text,
                marked_text=original_text,
                inconsistencies=[],
                confidence_score=0.0
            )

    def verify_text(self, wiki_relations: Any, relations_ans:Any, ans: str, beam:bool=False) -> VerificationResult:
        """
        Verify text against semantic relations and identify inconsistencies.

        Args:
            relations: List of semantic relations extracted from reference documents
            text: Text to verify (A_s)

        Returns:
            VerificationResult containing marked text and identified inconsistencies
        """
        # Create the prompt__ relations: relations: Any, text: str, text_rels:Any=None, text_form_relations=True
        prompt = self._create_verification_prompt(wiki_relations, ans, relations_ans)
        if not self.api_key is None:
           return self.model.generate_content(prompt)

        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True).to(self.device)

        with torch.no_grad():
            if beam:
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=4096,
                    num_beams=5,  # Set num_beams > 1 for beam search
                    early_stopping=True,  # Now this is valid
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
                    eos_token_id=self.tokenizer.eos_token_id  # Keep this to stop generation at EOS
                )

        # Decode the output
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse and return results
        #return self._parse_model_output(response, ans)
        return response
