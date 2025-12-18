"""
Evaluation Metrics

Calculate Mean Recall@K and other metrics for recommendation quality.
"""

import json
import logging
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator for recommendation system performance
    """
    
    def __init__(self, engine):
        """
        Initialize evaluator
        
        Args:
            engine: RecommendationEngine instance
        """
        self.engine = engine
    
    def calculate_recall_at_k(
        self,
        predicted_urls: List[str],
        ground_truth_urls: List[str],
        k: int = 10
    ) -> float:
        """
        Calculate Recall@K metric
        
        Recall@K = (Number of relevant items in top K) / (Total relevant items)
        
        Args:
            predicted_urls: List of predicted assessment URLs
            ground_truth_urls: List of ground truth assessment URLs
            k: Number of top predictions to consider
            
        Returns:
            Recall@K score (0.0 to 1.0)
        """
        if not ground_truth_urls:
            return 0.0
        
        predicted_set = set(predicted_urls[:k])
        ground_truth_set = set(ground_truth_urls)
        
        relevant_retrieved = len(predicted_set.intersection(ground_truth_set))
        recall = relevant_retrieved / len(ground_truth_set)
        
        return recall
    
    def evaluate_on_dataset(
        self,
        dataset_path: str,
        k: int = 10
    ) -> Tuple[float, List[Dict]]:
        """
        Evaluate recommendation engine on labeled dataset
        
        Args:
            dataset_path: Path to CSV file with Query and Assessment_url columns
            k: K value for Recall@K
            
        Returns:
            Tuple of (mean_recall, detailed_results)
        """
        logger.info(f"Evaluating on dataset: {dataset_path}")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Group by query to get ground truth assessments
        query_groups = df.groupby('Query')['Assessment_url'].apply(list).to_dict()
        
        recalls = []
        results = []
        
        print(f"\n{'='*80}")
        print(f"EVALUATION RESULTS (Recall@{k})")
        print(f"{'='*80}\n")
        
        for i, (query, ground_truth_urls) in enumerate(query_groups.items(), 1):
            # Get predictions
            try:
                predictions = self.engine.recommend(query, max_results=k)
                predicted_urls = [p['url'] for p in predictions]
            except Exception as e:
                logger.error(f"Error getting predictions for query {i}: {str(e)}")
                predicted_urls = []
            
            # Calculate recall
            recall = self.calculate_recall_at_k(predicted_urls, ground_truth_urls, k=k)
            recalls.append(recall)
            
            # Store results
            result = {
                'query': query,
                'recall@10': recall,
                'predicted': predicted_urls,
                'ground_truth': ground_truth_urls,
                'num_predicted': len(predicted_urls),
                'num_ground_truth': len(ground_truth_urls),
                'num_correct': len(set(predicted_urls[:k]).intersection(set(ground_truth_urls)))
            }
            results.append(result)
            
            # Print progress
            query_preview = query[:70] + "..." if len(query) > 70 else query
            print(f"Query {i}: {query_preview}")
            print(f"  Recall@{k}: {recall:.3f} ({result['num_correct']}/{result['num_ground_truth']} relevant found)")
            print()
        
        # Calculate mean recall
        mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
        
        print(f"{'='*80}")
        print(f"MEAN RECALL@{k}: {mean_recall:.4f}")
        print(f"{'='*80}\n")
        
        return mean_recall, results
    
    def analyze_failures(self, results: List[Dict], threshold: float = 0.5):
        """
        Analyze queries with low recall scores
        
        Args:
            results: Results from evaluate_on_dataset()
            threshold: Recall threshold below which to consider a failure
        """
        failures = [r for r in results if r['recall@10'] < threshold]
        
        print(f"\n{'='*80}")
        print(f"LOW RECALL QUERIES (Recall < {threshold}): {len(failures)}")
        print(f"{'='*80}\n")
        
        for i, failure in enumerate(failures, 1):
            print(f"{i}. Query: {failure['query'][:100]}...")
            print(f"   Recall@10: {failure['recall@10']:.3f}")
            print(f"   Predicted: {failure['num_predicted']} assessments")
            print(f"   Ground Truth: {failure['num_ground_truth']} assessments")
            print(f"   Correct: {failure['num_correct']} assessments")
            
            # Show missing assessments
            missing = set(failure['ground_truth']) - set(failure['predicted'])
            if missing:
                print(f"   Missing URLs:")
                for url in list(missing)[:3]:
                    name = url.split('/')[-1].replace('-', ' ').title()
                    print(f"     - {name}")
            print()
    
    def save_results(self, results: List[Dict], output_path: str):
        """
        Save evaluation results to JSON file
        
        Args:
            results: Results from evaluate_on_dataset()
            output_path: Path to save results
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved evaluation results to {output_path}")


def generate_test_predictions(
    engine,
    test_queries_path: str,
    output_path: str,
    min_results: int = 5,
    max_results: int = 10
):
    """
    Generate predictions for test set and save to CSV
    
    Args:
        engine: RecommendationEngine instance
        test_queries_path: Path to CSV with Query column
        output_path: Path to save predictions CSV
        min_results: Minimum recommendations per query
        max_results: Maximum recommendations per query
    """
    logger.info(f"Generating predictions for test set: {test_queries_path}")
    
    # Load test queries
    test_df = pd.read_csv(test_queries_path)
    
    if 'Query' not in test_df.columns:
        raise ValueError("Test file must have 'Query' column")
    
    results = []
    
    print(f"\nGenerating predictions for {len(test_df)} queries...")
    
    for i, query in enumerate(test_df['Query'], 1):
        print(f"Processing query {i}/{len(test_df)}...")
        
        try:
            # Get recommendations
            predictions = engine.recommend(
                query,
                min_results=min_results,
                max_results=max_results
            )
            
            # Add to results
            for pred in predictions:
                results.append({
                    'Query': query,
                    'Assessment_url': pred['url']
                })
        
        except Exception as e:
            logger.error(f"Error processing query {i}: {str(e)}")
            # Add at least one result to avoid empty queries
            results.append({
                'Query': query,
                'Assessment_url': 'https://www.shl.com/solutions/products/product-catalog/'
            })
    
    # Save to CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Predictions saved to {output_path}")
    print(f"Total rows: {len(output_df)}")
    print(f"Unique queries: {output_df['Query'].nunique()}")
    print(f"Avg recommendations per query: {len(output_df) / output_df['Query'].nunique():.1f}")


def main():
    """Test evaluation functionality"""
    from dotenv import load_dotenv
    from embedding.vectorstore import AssessmentVectorStore, load_assessments_from_file
    from recommender.engine import RecommendationEngine
    
    load_dotenv()
    
    # Initialize components
    vector_store = AssessmentVectorStore()
    
    # Load and index assessments if needed
    if vector_store.get_count() == 0:
        assessments = load_assessments_from_file('data/assessments.json')
        vector_store.index_assessments(assessments)
    
    # Initialize engine
    engine = RecommendationEngine(vector_store)
    
    # Initialize evaluator
    evaluator = Evaluator(engine)
    
    # Evaluate on training data
    mean_recall, results = evaluator.evaluate_on_dataset('Gen_AI Dataset.csv', k=10)
    
    # Analyze failures
    evaluator.analyze_failures(results, threshold=0.5)
    
    # Save results
    evaluator.save_results(results, 'evaluation_results.json')


if __name__ == "__main__":
    main()
