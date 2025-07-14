#!/usr/bin/env python3
"""
Simple test script to try out the dataset loader.
Loads datasets and prints basic information.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from config import Config
    from dataset_loader import DatasetLoader
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


def test_natural_questions():
    """Test NaturalQuestions dataset loading."""
    print("=" * 60)
    print("TESTING NATURAL QUESTIONS DATASET")
    print("=" * 60)
    
    try:
        # Load config and create loader
        config = Config("src/config.yaml")
        loader = DatasetLoader(config)
        
        print(f"✓ Dataset name: {loader.dataset_name}")
        print(f"✓ Raw dataset size: {len(loader.raw_ds)} examples")
        
        # Load corpus
        print("\nLoading corpus...")
        passages = loader.load_corpus()
        print(f"✓ Loaded {len(passages)} passages")
        
        # Show some sample passages
        print("\nSample passages:")
        for i in range(min(3, len(passages))):
            print(f"  {i+1}. {passages[i][:100]}...")
        
        # Split queries
        print("\nSplitting queries...")
        Q_ben, Q_trg, gt_map = loader.split_queries()
        print(f"✓ Benign queries: {len(Q_ben)}")
        print(f"✓ Trigger queries: {len(Q_trg)}")
        print(f"✓ Ground truth map entries: {len(gt_map)}")
        
        # Show sample queries
        print("\nSample benign queries:")
        for i in range(min(3, len(Q_ben))):
            print(f"  {i+1}. {Q_ben[i]}")
            
        print("\nSample trigger queries:")
        for i in range(min(3, len(Q_trg))):
            print(f"  {i+1}. {Q_trg[i]}")
            
        # Show ground truth mapping
        print("\nSample ground truth mappings:")
        sample_gt = list(gt_map.items())[:3]
        for q_idx, passage_indices in sample_gt:
            print(f"  Query {q_idx} -> Passages {passage_indices}")
            
        return True
        
    except Exception as e:
        print(f"✗ NaturalQuestions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hotpot_qa():
    """Test HotpotQA dataset loading."""
    print("\n" + "=" * 60)
    print("TESTING HOTPOT_QA DATASET")
    print("=" * 60)
    
    try:
        # Create config with HotpotQA
        config = Config("src/config.yaml")
        config._cfg["dataset"]["name"] = "hotpot_qa"
        
        loader = DatasetLoader(config)
        
        print(f"✓ Dataset name: {loader.dataset_name}")
        print(f"✓ Raw dataset size: {len(loader.raw_ds)} examples")
        
        # Load corpus
        print("\nLoading corpus...")
        passages = loader.load_corpus()
        print(f"✓ Loaded {len(passages)} passages")
        
        # Show some sample passages
        print("\nSample passages:")
        for i in range(min(3, len(passages))):
            print(f"  {i+1}. {passages[i][:100]}...")
        
        # Split queries
        print("\nSplitting queries...")
        Q_ben, Q_trg, gt_map = loader.split_queries()
        print(f"✓ Benign queries: {len(Q_ben)}")
        print(f"✓ Trigger queries: {len(Q_trg)}")
        print(f"✓ Ground truth map entries: {len(gt_map)}")
        
        # Show sample queries
        print("\nSample benign queries:")
        for i in range(min(3, len(Q_ben))):
            print(f"  {i+1}. {Q_ben[i]}")
            
        print("\nSample trigger queries:")
        for i in range(min(3, len(Q_trg))):
            print(f"  {i+1}. {Q_trg[i]}")
            
        # Show ground truth mapping
        print("\nSample ground truth mappings:")
        sample_gt = list(gt_map.items())[:3]
        for q_idx, passage_indices in sample_gt:
            print(f"  Query {q_idx} -> Passages {passage_indices}")
            
        return True
        
    except Exception as e:
        print(f"✗ HotpotQA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Dataset Loader Test Script")
    print("This script will test loading both NaturalQuestions and "
          "HotpotQA datasets")
    print()
    
    # Test NaturalQuestions
    nq_success = test_natural_questions()
    
    # Test HotpotQA
    hp_success = test_hotpot_qa()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"NaturalQuestions: {'✓ PASSED' if nq_success else '✗ FAILED'}")
    print(f"HotpotQA: {'✓ PASSED' if hp_success else '✗ FAILED'}")
    
    if nq_success and hp_success:
        print("\n🎉 All tests passed! Dataset loader is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")


if __name__ == "__main__":
    main() 