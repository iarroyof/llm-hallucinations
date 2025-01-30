import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict
from dataclasses import dataclass
import logging
import cohere
from datasets import load_dataset

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

class SemanticProcessing:
    def __init__(self, model_name: str = "mosaicml/mpt-7b-instruct", device: str = "cuda", cohere_api_key: str = "lWkdWMdYZdueoxBnzwPdEshuyWWEN1hYspCNyirG"):
        """
        Initialize the semantic processing class.

        Args:
            model_name: HuggingFace model identifier for verification.
            device: Device to run the model on ('cuda' or 'cpu').
            cohere_api_key: API key for Cohere client (required for semantic search).
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        # Optimización para uso eficiente de memoria en GPU
        bnb_config = BitsAndBytesConfig(load_in_8bit=True) if self.device == "cuda" else None
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=bnb_config
        ).to(self.device)

        self.cohere_client = cohere.Client(cohere_api_key) if cohere_api_key else None
        
        # Liberar memoria de GPU antes de cargar el modelo de verificación
        torch.cuda.empty_cache()

    def _format_relations(self, relations: List[SemanticRelation]) -> str:
        formatted = "Semantic Relations:\n"
        for i, rel in enumerate(relations, 1):
            formatted += f"{i}. {rel.subject} {rel.predicate} {rel.object} (confidence: {rel.confidence:.2f})\n"
        return formatted

    def _create_verification_prompt(self, relations: List[SemanticRelation], text: str) -> str:
        relations_text = self._format_relations(relations)
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

            confidence_score = max(0.0, min(1.0, 1.0 - (len(inconsistencies) * 0.1)))

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
        prompt = self._create_verification_prompt(relations, text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=1024,  # Reduce memory usage
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=1
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_model_output(response, text)

    def semantic_search(self, query: str, vector_database, k: int = 3):
        if not self.cohere_client:
            raise ValueError("Cohere client is not initialized. Provide a valid API key.")

        query_emb = self.cohere_client.embed(texts=[query], model='multilingual-22-12').embeddings[0]
        results = vector_database.search(query_emb, k)
        return results

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from typing import List, Dict
# from dataclasses import dataclass
# import logging
# import cohere
# from datasets import load_dataset

# @dataclass
# class SemanticRelation:
#     subject: str
#     predicate: str
#     object: str
#     confidence: float
#     source_doc: str

# @dataclass
# class VerificationResult:
#     original_text: str
#     marked_text: str
#     inconsistencies: List[Dict[str, str]]
#     confidence_score: float

# class SemanticProcessing:
#     def __init__(self, model_name: str = "mosaicml/mpt-7b-instruct", device: str = "cuda", cohere_api_key: str = None):
#         """
#         Initialize the semantic processing class.

#         Args:
#             model_name: HuggingFace model identifier for verification.
#             device: Device to run the model on ('cuda' or 'cpu').
#             cohere_api_key: API key for Cohere client (required for semantic search).
#         """
#         self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             trust_remote_code=True,
#             torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
#         ).to(self.device)

#         self.cohere_client = cohere.Client(cohere_api_key) if cohere_api_key else None

#     def _format_relations(self, relations: List[SemanticRelation]) -> str:
#         formatted = "Semantic Relations:\n"
#         for i, rel in enumerate(relations, 1):
#             formatted += f"{i}. {rel.subject} {rel.predicate} {rel.object} (confidence: {rel.confidence:.2f})\n"
#         return formatted

#     def _create_verification_prompt(self, relations: List[SemanticRelation], text: str) -> str:
#         relations_text = self._format_relations(relations)
#         prompt = f"""
#           Task: Analyze the following text for semantic inconsistencies using the provided semantic relations.

#           {relations_text}

#           Text to verify:
#           {text}

#           Instructions:
#           1. Compare the text against the semantic relations
#           2. Identify any inconsistencies or contradictions
#           3. Mark inconsistent segments with XML tags
#           4. Provide a brief explanation for each inconsistency

#           Output format:
#           1. First provide the text with <inconsistency> tags around problematic segments
#           2. Then list each inconsistency with its explanation

#           Response:"""
#         return prompt

#     def _parse_model_output(self, output: str, original_text: str) -> VerificationResult:
#         try:
#             parts = output.split("\n\nInconsistencies found:")
#             marked_text = parts[0].strip()

#             inconsistencies = []
#             if len(parts) > 1:
#                 explanations = parts[1].strip().split("\n")
#                 for exp in explanations:
#                     if exp.strip():
#                         inconsistencies.append({
#                             "text": exp.split(": ")[0],
#                             "explanation": exp.split(": ")[1]
#                         })

#             confidence_score = max(0.0, min(1.0, 1.0 - (len(inconsistencies) * 0.1)))

#             return VerificationResult(
#                 original_text=original_text,
#                 marked_text=marked_text,
#                 inconsistencies=inconsistencies,
#                 confidence_score=confidence_score
#             )
#         except Exception as e:
#             logging.error(f"Error parsing model output: {e}")
#             return VerificationResult(
#                 original_text=original_text,
#                 marked_text=original_text,
#                 inconsistencies=[],
#                 confidence_score=0.0
#             )

#     def verify_text(self, relations: List[SemanticRelation], text: str) -> VerificationResult:
#         prompt = self._create_verification_prompt(relations, text)
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

#         with torch.no_grad():
#             outputs = self.model.generate(
#                 inputs.input_ids,
#                 max_length=4096,
#                 temperature=0.1,
#                 top_p=0.95,
#                 do_sample=True,
#                 num_return_sequences=1
#             )

#         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return self._parse_model_output(response, text)

#     def semantic_search(self, query: str, doc_embeddings: torch.Tensor, doc_texts: List[str], k: int = 3):
#         if not self.cohere_client:
#             raise ValueError("Cohere client is not initialized. Provide a valid API key.")

#         query_emb = self.cohere_client.embed(texts=[query], model='multilingual-22-12').embeddings
#         query_emb = torch.tensor(query_emb, dtype=torch.float32)

#         dot_scores = torch.matmul(query_emb, doc_embeddings.T)
#         scores, indices = torch.topk(dot_scores, k=k, dim=1)

#         results = []
#         for i in range(k):
#             idx = indices[0, i].item()
#             score = scores[0, i].item()
#             results.append((score, doc_texts[idx]))

#         return results

# # Ejemplo de uso
# if __name__ == "__main__":
#     relations = [
#         SemanticRelation(
#             subject="Earth",
#             predicate="has diameter",
#             object="12,742 kilometers",
#             confidence=0.95,
#             source_doc="doc1"
#         ),
#         SemanticRelation(
#             subject="Earth",
#             predicate="orbits",
#             object="Sun",
#             confidence=0.98,
#             source_doc="doc1"
#         )
#     ]

#     text_to_verify = """The Earth, which has a diameter of about 10,000 kilometers, 
#     completes one orbit around the Sun every 365.25 days."""

#     processor = SemanticProcessing(device="cuda" if torch.cuda.is_available() else "cpu", cohere_api_key="lWkdWMdYZdueoxBnzwPdEshuyWWEN1hYspCNyirG")

#     # Verificación semántica
#     result = processor.verify_text(relations, text_to_verify)
#     print("Marked text:")
#     print(result.marked_text)
#     print("\nInconsistencies found:")
#     for inc in result.inconsistencies:
#         print(f"- {inc['text']}: {inc['explanation']}")
#     print(f"\nConfidence score: {result.confidence_score:.2f}")

#     # Búsqueda semántica
#     docs = load_dataset("Cohere/wikipedia-22-12-simple-embeddings", split="train")
#     embedding_list = [torch.tensor(e, dtype=torch.float32) for e in docs["emb"]]
#     doc_embeddings = torch.stack(embedding_list)
#     doc_texts = docs["text"]

#     query = "Who founded YouTube?"
#     results = processor.semantic_search(query, doc_embeddings, doc_texts, k=3)

#     print(f"\nTop resultados para la query: '{query}'\n")
#     for rank, (score, txt) in enumerate(results, start=1):
#         print(f"Rank {rank} | Score: {score:.4f}")
#         print(f"Text snippet: {txt[:300]}...")
#         print("----")

##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from typing import List, Dict
# import xml.etree.ElementTree as ET
# from dataclasses import dataclass
# import logging

# import cohere
# from datasets import load_dataset

# @dataclass
# class SemanticRelation:
#     subject: str
#     predicate: str
#     object: str
#     confidence: float
#     source_doc: str

# @dataclass
# class VerificationResult:
#     original_text: str
#     marked_text: str
#     inconsistencies: List[Dict[str, str]]
#     confidence_score: float

# class SemanticVerifier:
#     def __init__(self, model_name: str = "mosaicml/mpt-7b-instruct", device: str = "cuda"):
#         """
#         Initialize the semantic verifier with MPT-7B-Instruct model.
        
#         Args:
#             model_name: HuggingFace model identifier
#             device: Device to run the model on ('cuda' or 'cpu')
#         """
#         self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
#         logging.info(f"Loading model {model_name} on {self.device}")
        
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             trust_remote_code=True,
#             torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
#         ).to(self.device)
        
#     def _format_relations(self, relations: List[SemanticRelation]) -> str:
#         """Format semantic relations into a string for the prompt."""
#         formatted = "Semantic Relations:\n"
#         for i, rel in enumerate(relations, 1):
#             formatted += f"{i}. {rel.subject} {rel.predicate} {rel.object} (confidence: {rel.confidence:.2f})\n"
#         return formatted
    
#     def _create_verification_prompt(self, relations: List[SemanticRelation], text: str) -> str:
#         """Create the verification prompt for the model."""
#         relations_text = self._format_relations(relations)
#         prompt = f"""
#           Task: Analyze the following text for semantic inconsistencies using the provided semantic relations.

#           {relations_text}

#           Text to verify:
#           {text}

#           Instructions:
#           1. Compare the text against the semantic relations
#           2. Identify any inconsistencies or contradictions
#           3. Mark inconsistent segments with XML tags
#           4. Provide a brief explanation for each inconsistency
          
#           Output format:
#           1. First provide the text with <inconsistency> tags around problematic segments
#           2. Then list each inconsistency with its explanation
          
#           Response:"""
#         return prompt
    
#     def _parse_model_output(self, output: str, original_text: str) -> VerificationResult:
#         """Parse the model's output into a structured format."""
#         try:
#             # Split output into marked text and explanations
#             parts = output.split("\n\nInconsistencies found:")
#             marked_text = parts[0].strip()
            
#             # Extract inconsistencies
#             inconsistencies = []
#             if len(parts) > 1:
#                 explanations = parts[1].strip().split("\n")
#                 for exp in explanations:
#                     if exp.strip():
#                         inconsistencies.append({
#                             "text": exp.split(": ")[0],
#                             "explanation": exp.split(": ")[1]
#                         })
            
#             # Calculate confidence score based on number and severity of inconsistencies
#             confidence_score = 1.0 - (len(inconsistencies) * 0.1)  # Simple scoring mechanism
#             confidence_score = max(0.0, min(1.0, confidence_score))
            
#             return VerificationResult(
#                 original_text=original_text,
#                 marked_text=marked_text,
#                 inconsistencies=inconsistencies,
#                 confidence_score=confidence_score
#             )
#         except Exception as e:
#             logging.error(f"Error parsing model output: {e}")
#             return VerificationResult(
#                 original_text=original_text,
#                 marked_text=original_text,
#                 inconsistencies=[],
#                 confidence_score=0.0
#             )
    
#     def verify_text(self, relations: List[SemanticRelation], text: str) -> VerificationResult:
#         """
#         Verify text against semantic relations and identify inconsistencies.
        
#         Args:
#             relations: List of semantic relations extracted from reference documents
#             text: Text to verify (A_s)
            
#         Returns:
#             VerificationResult containing marked text and identified inconsistencies
#         """
#         # Create the prompt
#         prompt = self._create_verification_prompt(relations, text)
        
#         # Tokenize and generate
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
#         with torch.no_grad():
#             outputs = self.model.generate(
#                 inputs.input_ids,
#                 max_length=4096,
#                 temperature=0.1,
#                 top_p=0.95,
#                 do_sample=True,
#                 num_return_sequences=1
#             )
        
#         # Decode the output
#         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         # Parse and return results
#         return self._parse_model_output(response, text)


# def semantic_search(query: str, co_client, doc_embeddings: torch.Tensor, doc_texts: List[str], k: int = 3):
#     """
#     Dada una query y una lista de embeddings (doc_embeddings),
#     calcula el dot product entre el embedding de la query y los docs,
#     y devuelve los k textos más similares.
    
#     Args:
#         query: La consulta a buscar
#         co_client: Instancia de cohere.Client
#         doc_embeddings: Tensor [N, 768]
#         doc_texts: Lista de strings (texto de cada documento)
#         k: Top k resultados
    
#     Returns:
#         Lista de (score, text) con los k más relevantes
#     """
#     # 1) Generar embedding de la query
#     query_emb = co_client.embed(texts=[query], model='multilingual-22-12').embeddings
#     query_emb = torch.tensor(query_emb, dtype=torch.float32)  # [1, 768]
    
#     # 2) Producto punto (dot product)
#     dot_scores = torch.matmul(query_emb, doc_embeddings.T)  # shape [1, N]
    
#     # 3) Obtener topk
#     scores, indices = torch.topk(dot_scores, k=k, dim=1)
    
#     # 4) Prepara resultado legible
#     results = []
#     for i in range(k):
#         idx = indices[0, i].item()
#         score = scores[0, i].item()
#         results.append((score, doc_texts[idx]))
    
#     return results


# if __name__ == "__main__":
#     # -------------------------------------------------------------------------
#     # 1) EJEMPLO BÁSICO: Verificador semántico con MPT-7B
#     # -------------------------------------------------------------------------
#     relations = [
#         SemanticRelation(
#             subject="Earth",
#             predicate="has diameter",
#             object="12,742 kilometers",
#             confidence=0.95,
#             source_doc="doc1"
#         ),
#         SemanticRelation(
#             subject="Earth",
#             predicate="orbits",
#             object="Sun",
#             confidence=0.98,
#             source_doc="doc1"
#         )
#     ]
    
#     text_to_verify = """The Earth, which has a diameter of about 10,000 kilometers, 
#     completes one orbit around the Sun every 365.25 days."""
    
#     # Inicializa el verificador
#     verifier = SemanticVerifier(device="cuda" if torch.cuda.is_available() else "cpu")
    
#     # Ejecuta verificación
#     result = verifier.verify_text(relations, text_to_verify)
    
#     print("Marked text:")
#     print(result.marked_text)
#     print("\nInconsistencies found:")
#     for inc in result.inconsistencies:
#         print(f"- {inc['text']}: {inc['explanation']}")
#     print(f"\nConfidence score: {result.confidence_score:.2f}")
    
    
#     # -------------------------------------------------------------------------
#     # 2) EJEMPLO: Búsqueda semántica con embeddings de Wikipedia
#     # -------------------------------------------------------------------------
#     # A) Carga tu API key y crea el cliente de Cohere
#     co = cohere.Client("lWkdWMdYZdueoxBnzwPdEshuyWWEN1hYspCNyirG")  # <--- Reemplaza con tu API Key real
    
#     # B) Descarga el dataset "wikipedia-22-12-simple-embeddings" completo
#     print("\nCargando dataset de Wikipedia (puede tardar bastante si no está en caché)...")
#     docs = load_dataset("Cohere/wikipedia-22-12-simple-embeddings", split="train")
    
#     # Imprimir la cantidad total de registros
#     print(f"El dataset completo tiene {len(docs)} registros.")
    
#     # C) Convertimos la columna 'emb' a tensores (TODOS los registros)
#     print("\nConvirtiendo TODOS los embeddings a PyTorch (podría requerir varios GB de RAM)...")
#     embedding_list = [torch.tensor(e, dtype=torch.float32) for e in docs["emb"]]
#     doc_embeddings = torch.stack(embedding_list)  # [N, 768]
    
#     # Guardamos los textos para luego desplegarlos
#     doc_texts = docs["text"]
    
#     # D) Hacemos la búsqueda semántica
#     query = "Who founded YouTube?"
#     print(f"\nRealizando semantic search para la query: '{query}' (top 3)...")
#     results = semantic_search(query, co, doc_embeddings, doc_texts, k=3)
    
#     # E) Imprimir resultados
#     print(f"\nTop resultados para la query: '{query}'\n")
#     for rank, (score, txt) in enumerate(results, start=1):
#         print(f"Rank {rank} | Score: {score:.4f}")
#         print(f"Text snippet: {txt[:300]}...")
#         print("----")
    