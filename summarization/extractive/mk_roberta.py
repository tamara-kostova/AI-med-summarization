import logging
import numpy as np
import torch
import re
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from summarization.extractive.extractive_summarizer import ExtractiveSummarizer
from summarization.utils import split_text_into_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MkRobertaSummarizer(ExtractiveSummarizer):
    """
    Specialized extractive summarizer for Macedonian medical texts
    using macedonizer/mk-roberta-base model
    """
    
    def __init__(
        self, 
        model_name="macedonizer/mk-roberta-base",
        use_first=True,
        use_medical_weighting=True,
        clustering_algorithm="kmeans",
        random_state=12345,
        num_clusters_ratio=0.15
    ):
        """
        Initialize the summarizer with macedonizer/mk-roberta-base.
        
        Args:
            model_name: The RoBERTa model to use
            use_first: Whether to always include the first sentence
            use_medical_weighting: Whether to boost medical terms
            clustering_algorithm: Algorithm for clustering sentences
            random_state: Random seed for clustering
            num_clusters_ratio: Ratio of sentences to select (0.15 = 15%)
        """
        logger.info(f"Initializing MkMedicalSummarizer with {model_name}")
        self.model_name = model_name
        self.use_first = use_first
        self.use_medical_weighting = use_medical_weighting
        self.clustering_algorithm = clustering_algorithm
        self.random_state = random_state
        self.num_clusters_ratio = num_clusters_ratio
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        self.model.eval()
        
        self.medical_terms = [
            "дијабетес", "пациент", "третман", "лекување", "болест", 
            "симптом", "здравје", "исхрана", "лек", "доктор", "болница",
            "терапија", "медицина", "дијагноза", "анализа", "крв",
            "притисок", "болка", "хронична", "акутна", "инфекција", 
            "вирус", "бактерија", "хирургија", "операција", "рак",
            "тумор", "кардио", "срце", "мозок", "бубрег", "црн дроб",
            "алергија", "антибиотик", "вакцина", "имунитет", "хормон",
            "метаболизам", "протеин", "гени", "клетки", "ткиво"
        ]

    def _get_sentence_embeddings(self, sentences):
        """Get embeddings for a list of sentences using mk-roberta-base"""
        embeddings = []
        
        for sentence in sentences:
            # Skip empty sentences
            if not sentence.strip():
                continue
                
            # Tokenize sentence
            inputs = self.tokenizer(
                sentence, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            # Get model output without calculating gradients
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Use mean pooling to get sentence embedding
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embedding)
            
        return np.array(embeddings)
    
    def _contains_medical_terms(self, sentence):
        """Check if a sentence contains medical terms"""
        sentence_lower = sentence.lower()
        return any(term in sentence_lower for term in self.medical_terms)
    
    def _medical_importance_score(self, sentence):
        """Calculate medical importance score for weighting"""
        sentence_lower = sentence.lower()
        term_count = sum(1 for term in self.medical_terms if term in sentence_lower)
        has_numbers = bool(re.search(r'\d+([.,]\d+)?', sentence))
        has_percentages = bool(re.search(r'\d+([.,]\d+)?%', sentence))
        has_results = bool(re.search(r'резултат|покажува|анализа|ниво', sentence_lower))
        
        score = 1.0
        if term_count > 0:
            score += min(1.0, term_count * 0.2)
        if has_numbers and has_results:
            score += 0.5
        if has_percentages:
            score += 0.3
            
        return score
    
    def _cluster_sentences(self, embeddings, sentences, num_sentences_to_return):
        """Cluster sentence embeddings with medical weighting"""
        if embeddings.shape[0] <= num_sentences_to_return:
            return list(range(embeddings.shape[0]))
        
        # Apply medical weighting if enabled
        if self.use_medical_weighting:
            weights = np.array([self._medical_importance_score(s) for s in sentences])
            weighted_embeddings = embeddings * weights[:, np.newaxis]
        else:
            weighted_embeddings = embeddings
        
        # Choose clustering algorithm
        if self.clustering_algorithm == "kmeans":
            clustering_model = KMeans(
                n_clusters=num_sentences_to_return,
                random_state=self.random_state
            )
            clustering_model.fit(weighted_embeddings)
            centers = clustering_model.cluster_centers_
            
            # Find closest sentences to each cluster center
            closest_indices = []
            for center in centers:
                distances = np.sqrt(((weighted_embeddings - center) ** 2).sum(axis=1))
                closest_index = np.argmin(distances)
                closest_indices.append(closest_index)
                
            return sorted(closest_indices)
        else:
            sim_matrix = cosine_similarity(weighted_embeddings)
            
            centrality_scores = np.sum(sim_matrix, axis=1)
            if self.use_medical_weighting:
                med_boost = np.array([1.0 + 0.5 * self._contains_medical_terms(s) for s in sentences])
                centrality_scores = centrality_scores * med_boost
            
            top_indices = centrality_scores.argsort()[-num_sentences_to_return:]
            return sorted(top_indices)
    
    def generate_extractive_summary_chunk(self, text):
        """Generate extractive summary for a chunk of text"""
        try:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) <= 3:
                return text
                
            # Get sentence embeddings
            embeddings = self._get_sentence_embeddings(sentences)
            if len(embeddings) == 0:
                logger.warning("No valid sentences to embed")
                return text
                
            num_sentences = max(3, int(len(sentences) * self.num_clusters_ratio))
            
            selected_indices = self._cluster_sentences(embeddings, sentences, num_sentences)
            
            if self.use_first and 0 not in selected_indices:
                selected_indices = [0] + selected_indices[:-1] if len(selected_indices) > 0 else [0]
                
            summary_sentences = [sentences[i] for i in sorted(selected_indices)]
            summary = '. '.join(summary_sentences)
            
            if summary and not summary.endswith('.'):
                summary += '.'
                
            return summary
            
        except Exception as e:
            logger.error(f"Error in extractive summarization: {str(e)}")
            return text[:500] + "..." if len(text) > 500 else text
    
    def generate_extractive_summary(self, text, num_sentences_per_chunk=None):
        """Generate extractive summary for a longer text by processing chunks"""
        if not num_sentences_per_chunk:
            num_sentences_per_chunk = max(3, int(10 * self.num_clusters_ratio))
            
        logger.info(f"Generating medical extractive summary with {num_sentences_per_chunk} sentences per chunk")
        
        try:
            if len(text.split()) < 500:
                return self.generate_extractive_summary_chunk(text)
                
            summaries = []
            for chunk in split_text_into_chunks(text, max_words=1000):
                if chunk.strip():
                    summary = self.generate_extractive_summary_chunk(chunk)
                    summaries.append(summary)
            
            if len(summaries) > 1:
                logger.info(f"Applying second-level medical summarization on {len(summaries)} chunks")
                combined_summary = " ".join(summaries)
                final_summary = self.generate_extractive_summary_chunk(combined_summary)
                return final_summary
            elif len(summaries) == 1:
                return summaries[0]
            else:
                logger.warning("No valid chunks found for summarization")
                return text[:500] + "..." if len(text) > 500 else text
                
        except Exception as e:
            logger.error(f"Error in extractive summarization pipeline: {str(e)}")
            return text[:500] + "..." if len(text) > 500 else text