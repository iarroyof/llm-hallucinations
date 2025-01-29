import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple, Any
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from huggingface_hub import login
import logging
import re

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
    def __init__(self, model_name: str, device: str="cuda", authenticate:bool=False):
        """
        Initialize the semantic verifier with *-Instruct model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run the model on ('cuda' or 'cpu')
        """
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

        prompt = f"""
            Task:
              Identify inconsistencies in the "text to verify" using the provided semantic relations from ground truth sources.
            
            Instructions:
              1. Compare the extracted semantic relations from the "text to verify" with those from the ground truth.
              2. Identify any factual, logical, semantic, ortographic, or mathematical inconsistencies.
              3. Estimate the probability of inconsistency for each word (soft_labels).
              4. Mark the exact position (start index inclusive, end index exclusive) of inconsistent segments (hard_labels).
              5. Wrap inconsistent segments in XML tags.
              6. Briefly explain why each inconsistency exists.
            
            Format:
            ```json
            {{
              "text_to_verify": "<text>",
              "inconsistency_identification": {{
                "soft_labels": [
                  {{"start":<int>, "prob":<float>, "end":<int>}}
                ],
                "hard_labels": [[<int>, <int>]]
              }},
              "explanation": "<reasoning>"
            }}

            Inputs:
              1. Text to check: {text}
              2. Known facts (as relations): {relations_text}
              3. Relations found in given text: {ans_relations_text}
        """
        return prompt
    def __create_verification_prompt(self, relations: Any, text: str, text_rels:Any=None, text_form_relations=True) -> str:
        """Create the verification prompt for the model."""
        if not text_form_relations:
            relations_text = self._format_relations(relations)
            ans_relations_text = self._format_relations(text_rels)
        else:
            relations_text = relations
            ans_relations_text = text_rels
    
        prompt = f"""
            Task: Check if the given text has any mistakes or incorrect information by comparing it with known facts.
            Instructions:
            1. Comprehend and compare the text against the known facts (shown bellow, put again in the 'given_text' key of the output format)
            2. Find any parts that are:
               - Factually incorrect
               - Logically contradicting
               - Semantically incorrect
            3. For each sequence of characters found to be inconsistent in the given text, assign a probability (0-1) of it being part of a mistake ('soft_labels')
            4. Mark the start and end character positions of incorrect parts ('hard_labels')
            5. Add XML tags around the incorrect parts ('marked_text')
            6. Explain why these parts are incorrect ('explanation')
            
            Expected Output Format (jsonl):
            {{
                "given_text": "Text being given to verify for inconsistencies"
                "soft_labels": [
                    {{"start": number, "end": number, "prob": number}},
                    ...
                ],
                "hard_labels": [[start, end], ...],
                "marked_text": "Text being given <inc>to verify</inc> for inconsistencies",
                "explanation": "Here goes a brief explanation for each inconsistency identified and hard labeled"
            }}
            Please analyze the following inputs according to instrucionts and output format shown above.
            Only generate the jsonl output, whithout additional text.
            Input:
            1. Text to check: {text}
            2. Known facts (as relations): {relations_text}
            3. Relations found in given text: {ans_relations_text}
            """
        return prompt

    def create_verification_prompt(self, relations: Any, text: str, text_rels:Any=None, text_form_relations=True) -> str:
        """Create the verification prompt for the model."""
        if not text_form_relations:
            relations_text = self._format_relations(relations)
            ans_relations_text = self._format_relations(text_rels)
        else:
            relations_text = relations
            ans_relations_text = text_rels

        prompt = f"""
            Task:
              Analyze the following text ("text to verify") for semantic, factual, ortographic, logical and mathematical inconsistencies using the provided semantic
              relations extracted from documental (ground truth) textual sources and identify where specifically such inconsistences are in the text to verify. 
            
            Sequential Instructions to follow in this task:
              1. Project the semantic, factual, ortographic, logical and mathematical knowledge in the "text to verify" against the semantic relations extracted from the ground truth text and memorize the result;
              2. Project your own semantic, factual, ortographic, logical and mathematical knowledge onto the "text to verify" and memorize the result;
              3. Use the above results to identify any inconsistencies, contradictions, or errors in the "text to verify";
              4. Estimate a probabilty of that each word in the "text to verify" is part of any inconsistency ("soft_labels");
              5. Identify the position (indices) of words ("start" inclusive, "end" exclusive) in the "text to verify" that are part of any inconsistency ("hard_labels");
              6. Mark inconsistent segments in the "text to verify" with XML tags
              7. Prepare a brief explanation for each inconsistency
            
              NOTE: Be aware always that inconsistent segements maybe zero or more than one and they may overlap, so it may result in an empty list of indices or more than a list of indices identifying inconsistencies.
            
            Output format:
              The following is an example of the output fromat of this task (Note that only probabilities more than 0.5 will be included in the "hard_labels" list):
              text_to_verify: "Yes, Scotland made their debut in the UEFA Euro 1996 qualifying phase. This was their first appearance in a European Championship qualifying campaign since the inception of the UEFA European Football Championship in 1960. Scotland finished third in their group behind England and Switzerland, missing out on qualification for the tournament."
              inconsistency_identification:
                  "{{"soft_labels":[{{"start":1,"prob":0.6666666667,"end":4}},
                                  {{"start":6,"prob":0.3333333333,"end":31}},
                                  {{"start":39,"prob":0.3333333333,"end":49}},
                                  {{"start":49,"prob":0.6666666667,"end":53}},
                                  {{"start":53,"prob":0.3333333333,"end":70}},
                                  {{"start":72,"prob":0.3333333333,"end":87}},
                                  {{"start":87,"prob":1.0,"end":92}},
                                  {{"start":92,"prob":0.6666666667,"end":103}},
                                  {{"start":103,"prob":0.3333333333,"end":221}},
                                  {{"start":223,"prob":0.3333333333,"end":232}},
                                  {{"start":232,"prob":0.6666666667,"end":246}},
                                  {{"start":246,"prob":0.3333333333,"end":262}},
                                  {{"start":262,"prob":0.6666666667,"end":269}},
                                  {{"start":269,"prob":1.0,"end":276}},
                                  {{"start":276,"prob":0.6666666667,"end":281}},
                                  {{"start":281,"prob":1.0,"end":292}},
                                  {{"start":292,"prob":0.3333333333,"end":294}},
                                  {{"start":294,"prob":0.6666666667,"end":322}},
                                  {{"start":322,"prob":0.3333333333,"end":341}}],
                   "hard_labels":[[1,4],[49,53],[87,103],[232,246],[262,292],[294,322]]}}"
              
              explanation: Here goes the explanation on the reasoning leaving you to conclude that each pair of hard_labels is identified
            
            Let's do it:
            Text to verify:
            {text}
            
            Relations extracted from ground truth sources:
            {relations_text}
            
            Relations extracted from the text to verify:
            {ans_relations_text}
            
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
    
    def verify_text(self, wiki_relations: Any, relations_ans:Any, ans: str, beam:bool=True) -> VerificationResult:
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
        return self._parse_model_output(response, ans)
