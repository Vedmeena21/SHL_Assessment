"""
LLM-Based Recommendation Engine

Multi-stage pipeline for intelligent assessment recommendations:
1. Query Understanding (LLM)
2. Initial Retrieval (Vector Search)
3. Re-ranking (LLM)
4. Balancing (Test Type Distribution)
5. Final Selection
"""

import json
import logging
import os
from typing import List, Dict, Optional
import google.generativeai as genai
from embedding.vectorstore import AssessmentVectorStore

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    LLM-powered recommendation engine for SHL assessments
    """
    
    def __init__(self, vector_store: AssessmentVectorStore, gemini_api_key: Optional[str] = None):
        """
        Initialize recommendation engine
        
        Args:
            vector_store: Initialized vector store
            gemini_api_key: Gemini API key
        """
        self.vector_store = vector_store
        
        # Configure Gemini
        api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        logger.info("Initialized recommendation engine")
    
    def analyze_query(self, query: str) -> Dict:
        """
        Use LLM to understand query intent and extract requirements
        
        Args:
            query: Job description or search query
            
        Returns:
            Dictionary with extracted information
        """
        prompt = f"""Analyze this job description or query and extract key information.

Query: {query}

Extract the following and respond ONLY with valid JSON (no markdown, no code blocks):
{{
    "technical_skills": ["list of technical skills mentioned"],
    "soft_skills": ["list of soft skills mentioned"],
    "level": "entry/mid/senior",
    "domains": ["list of domains like programming, sales, admin"],
    "balance": {{"technical": 70, "behavioral": 30}}
}}

The balance should indicate what percentage should be technical (K) vs behavioral (P) tests.
For technical roles, use 70-80% technical. For leadership/sales roles, use 40-50% technical.

Respond with ONLY the JSON object, nothing else."""

        try:
            response = self.model.generate_content(prompt)
            analysis = self._parse_json_response(response.text)
            logger.debug(f"Query analysis: {analysis}")
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            # Return default analysis
            return {
                "technical_skills": [],
                "soft_skills": ["communication"],
                "level": "mid",
                "domains": ["general"],
                "balance": {"technical": 60, "behavioral": 40}
            }
    
    def _parse_json_response(self, text: str) -> Dict:
        """Parse JSON from LLM response"""
        # Remove markdown code blocks if present
        text = text.strip()
        if text.startswith('```'):
            # Extract content between code blocks
            lines = text.split('\n')
            text = '\n'.join(lines[1:-1]) if len(lines) > 2 else text
            text = text.replace('```json', '').replace('```', '')
        
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {text}")
            raise
    
    def retrieve_candidates(self, query: str, analysis: Dict, k: int = 50) -> List[Dict]:
        """
        Retrieve initial candidates using vector search
        
        Args:
            query: Original query
            analysis: Query analysis from analyze_query()
            k: Number of candidates to retrieve
            
        Returns:
            List of candidate assessments
        """
        # Main query search
        results = self.vector_store.search(query, k=k)
        
        # Create skill-specific queries for better coverage
        skill_results = []
        for skill in (analysis.get('technical_skills', []) + analysis.get('soft_skills', []))[:5]:
            skill_results.extend(self.vector_store.search(skill, k=10))
        
        # Combine and deduplicate
        all_results = results + skill_results
        seen_urls = set()
        unique_results = []
        
        for result in all_results:
            if result['url'] not in seen_urls:
                seen_urls.add(result['url'])
                unique_results.append(result)
        
        logger.debug(f"Retrieved {len(unique_results)} unique candidates")
        return unique_results[:k]
    
    def _keyword_overlap(self, query: str, text: str) -> float:
        """
        Calculate lexical keyword overlap between query and text
        
        Args:
            query: Search query
            text: Text to compare against
            
        Returns:
            Overlap score between 0.0 and 1.0
        """
        # Tokenize and normalize
        q_tokens = set(query.lower().split())
        t_tokens = set(text.lower().split())
        
        # Calculate overlap
        overlap = len(q_tokens & t_tokens)
        total = max(len(q_tokens), 1)
        
        return overlap / total
    
    def _test_type_alignment(self, assessment_type: str, analysis: Dict) -> float:
        """
        Calculate test type alignment score
        
        Args:
            assessment_type: Test type (K/P/O)
            analysis: Query analysis with balance preferences
            
        Returns:
            Alignment score between 0.0 and 1.0
        """
        balance = analysis.get('balance', {'technical': 60, 'behavioral': 40})
        technical_pct = balance.get('technical', 60) / 100
        
        # If query wants technical and this is technical (K), high score
        if assessment_type == 'K':
            return 0.5 + (technical_pct * 0.5)  # 0.5 to 1.0
        # If query wants behavioral and this is behavioral (P), high score
        elif assessment_type == 'P':
            behavioral_pct = 1.0 - technical_pct
            return 0.5 + (behavioral_pct * 0.5)  # 0.5 to 1.0
        # Other types get neutral score
        else:
            return 0.5
    
    def _compute_per_assessment_score(
        self,
        assessment: Dict,
        query: str,
        analysis: Dict
    ) -> float:
        """
        Compute deterministic relevance score for a single assessment
        
        Scoring formula:
        final_score = 0.6 × semantic_similarity 
                    + 0.3 × keyword_overlap 
                    + 0.1 × test_type_alignment
        
        Args:
            assessment: Assessment dictionary with metadata
            query: Original search query
            analysis: Query analysis from LLM
            
        Returns:
            Final relevance score (0.0 to 1.0)
        """
        # 1. Semantic similarity (from vector search)
        # ChromaDB returns distance, convert to similarity
        semantic_similarity = assessment.get('score', 0.5)  # Already converted in vectorstore
        
        # 2. Keyword overlap (lexical matching)
        # Compare query against assessment name + description
        assessment_text = f"{assessment.get('name', '')} {assessment.get('description', '')}"
        keyword_overlap = self._keyword_overlap(query, assessment_text)
        
        # 3. Test type alignment
        test_type = assessment.get('test_type', 'O')
        test_type_score = self._test_type_alignment(test_type, analysis)
        
        # Combine using weighted formula
        final_score = (
            0.6 * semantic_similarity +
            0.3 * keyword_overlap +
            0.1 * test_type_score
        )
        
        return final_score
    
    def retrieve_candidates(self, query: str, analysis: Dict, k: int = 50) -> List[Dict]:
        """
        Retrieve initial candidates using vector search
        
        Args:
            query: Original query
            analysis: Query analysis from analyze_query()
            k: Number of candidates to retrieve
            
        Returns:
            List of candidate assessments
        """
        # Main query search
        results = self.vector_store.search(query, k=k)
        
        # Create skill-specific queries for better coverage
        skill_results = []
        for skill in (analysis.get('technical_skills', []) + analysis.get('soft_skills', []))[:5]:
            skill_results.extend(self.vector_store.search(skill, k=10))
        
        # Combine and deduplicate
        all_results = results + skill_results
        seen_urls = set()
        unique_results = []
        
        for result in all_results:
            if result['url'] not in seen_urls:
                seen_urls.add(result['url'])
                unique_results.append(result)
        
        logger.debug(f"Retrieved {len(unique_results)} unique candidates")
        return unique_results[:k]
    
    def score_and_rank_candidates(
        self,
        query: str,
        candidates: List[Dict],
        analysis: Dict
    ) -> List[Dict]:
        """
        Score each candidate deterministically and rank by relevance
        
        This replaces LLM-based re-ranking with a transparent, 
        reproducible scoring function.
        
        Args:
            query: Original query
            candidates: List of candidate assessments
            analysis: Query analysis
            
        Returns:
            Ranked list of assessments with relevance scores
        """
        scored_candidates = []
        
        for candidate in candidates:
            # Compute per-assessment score using deterministic formula
            final_score = self._compute_per_assessment_score(
                candidate,
                query,
                analysis
            )
            
            # Create a copy and add the final score
            scored_candidate = candidate.copy()
            scored_candidate['final_score'] = final_score
            scored_candidate['relevance_percentage'] = round(final_score * 100, 2)
            
            scored_candidates.append(scored_candidate)
        
        # Sort by final score (highest first)
        scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        logger.debug(f"Scored and ranked {len(scored_candidates)} candidates")
        return scored_candidates
    
    def balance_recommendations(
        self,
        ranked_candidates: List[Dict],
        analysis: Dict,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Ensure balanced mix of technical vs behavioral tests
        
        IMPORTANT: This does not change scores, only selection distribution
        
        Args:
            ranked_candidates: Ranked list of candidates
            analysis: Query analysis with balance requirements
            top_k: Number of final recommendations
            
        Returns:
            Balanced list of recommendations
        """
        balance = analysis.get('balance', {'technical': 60, 'behavioral': 40})
        target_technical_pct = balance.get('technical', 60) / 100
        target_behavioral_pct = balance.get('behavioral', 40) / 100
        
        # Separate by test type (maintaining their scores)
        technical_tests = [c for c in ranked_candidates if c.get('test_type') == 'K']
        behavioral_tests = [c for c in ranked_candidates if c.get('test_type') == 'P']
        other_tests = [c for c in ranked_candidates if c.get('test_type') not in ['K', 'P']]
        
        # Calculate target counts
        num_technical = max(1, int(top_k * target_technical_pct))
        num_behavioral = max(1, int(top_k * target_behavioral_pct))
        num_other = top_k - num_technical - num_behavioral
        
        # Select top from each category (already sorted by score)
        final_recommendations = []
        final_recommendations.extend(technical_tests[:num_technical])
        final_recommendations.extend(behavioral_tests[:num_behavioral])
        if num_other > 0:
            final_recommendations.extend(other_tests[:num_other])
        
        # Fill remaining spots with highest scored
        while len(final_recommendations) < top_k and ranked_candidates:
            for candidate in ranked_candidates:
                if candidate not in final_recommendations:
                    final_recommendations.append(candidate)
                    break
            else:
                break
        
        # Re-sort final list by score to maintain ranking integrity
        final_recommendations.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        logger.debug(f"Balanced to {len(final_recommendations)} recommendations")
        return final_recommendations[:top_k]
    
    def recommend(
        self,
        query: str,
        min_results: int = 5,
        max_results: int = 10
    ) -> List[Dict]:
        """
        Main recommendation pipeline with deterministic scoring
        
        Pipeline:
        1. Query Understanding (LLM) - extracts intent only
        2. Retrieve Candidates (Vector Search) - gets similar assessments
        3. Score & Rank (Deterministic) - computes relevance per-assessment
        4. Balance (Optional) - ensures test type distribution
        5. Return Results - with distinct relevance scores
        
        Args:
            query: Job description or search query
            min_results: Minimum number of recommendations
            max_results: Maximum number of recommendations
            
        Returns:
            List of recommended assessments with relevance_percentage
        """
        logger.info(f"Generating recommendations for query: {query[:100]}...")
        
        # Step 1: Analyze query intent using LLM (ONLY for understanding)
        analysis = self.analyze_query(query)
        
        # Step 2: Retrieve candidates using vector search
        candidates = self.retrieve_candidates(query, analysis, k=50)
        
        if not candidates:
            logger.warning("No candidates found")
            return []
        
        # Step 3: Score and rank candidates DETERMINISTICALLY
        # This is the key fix - no LLM dependency for final ranking
        ranked = self.score_and_rank_candidates(query, candidates, analysis)
        
        # Step 4: Balance test types while preserving scores
        final = self.balance_recommendations(ranked, analysis, top_k=max_results)
        
        # Ensure minimum
        if len(final) < min_results and len(ranked) >= min_results:
            final = ranked[:min_results]
        
        # Log score distribution for debugging
        if final:
            scores = [r['relevance_percentage'] for r in final]
            logger.info(f"Score range: {min(scores):.1f}% to {max(scores):.1f}%")
        
        logger.info(f"Generated {len(final)} recommendations")
        return final[:max_results]


def main():
    """Test recommendation engine"""
    from dotenv import load_dotenv
    from embedding.vectorstore import AssessmentVectorStore, load_assessments_from_file
    
    load_dotenv()
    
    # Initialize components
    vector_store = AssessmentVectorStore()
    
    # Load and index assessments if needed
    if vector_store.get_count() == 0:
        assessments = load_assessments_from_file('data/assessments.json')
        vector_store.index_assessments(assessments)
    
    # Initialize engine
    engine = RecommendationEngine(vector_store)
    
    # Test query
    test_query = """I am hiring for Java developers who can also collaborate 
    effectively with my business teams. Looking for an assessment(s) that can 
    be completed in 40 minutes."""
    
    # Get recommendations
    recommendations = engine.recommend(test_query, min_results=5, max_results=10)
    
    # Display results
    print(f"\nRecommendations for: '{test_query[:80]}...'")
    print("="*80)
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['name']}")
        print(f"   URL: {rec['url']}")
        print(f"   Type: {rec['test_type']}")
        print(f"   Score: {rec.get('final_score', 0):.3f}")
        if 'reasoning' in rec:
            print(f"   Reasoning: {rec['reasoning']}")


if __name__ == "__main__":
    main()
