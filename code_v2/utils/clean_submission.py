#!/usr/bin/env python3
# code_v2/utils/clean_submission.py
# Author: Lê Đăng Hoàng Tuấn
# Day 7: Clean duplicate review_ids in submission.csv
#
# Problem: submission.csv has 1,592,698 duplicates (91.8%)
# Solution: Deduplicate using 3 strategies (keep first/last/average)

import pandas as pd
import argparse
import json
import os
from datetime import datetime

def load_submission(input_path):
    """Load submission CSV and validate format"""
    print(f"\n[1/5] Loading submission from: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    
    df = pd.read_csv(input_path)
    
    print(f"  ✓ Loaded {len(df):,} rows")
    print(f"  ✓ Columns: {list(df.columns)}")
    
    # Validate columns
    required_cols = ['review_id', 'probability_helpful']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Expected: {required_cols}, Got: {list(df.columns)}")
    
    return df

def analyze_duplicates(df):
    """Analyze duplicate patterns"""
    print(f"\n[2/5] Analyzing duplicates...")
    
    total_rows = len(df)
    unique_ids = df['review_id'].nunique()
    duplicate_rows = total_rows - unique_ids
    duplicate_pct = (duplicate_rows / total_rows * 100) if total_rows > 0 else 0
    
    print(f"  Total rows: {total_rows:,}")
    print(f"  Unique review_ids: {unique_ids:,}")
    print(f"  Duplicate rows: {duplicate_rows:,} ({duplicate_pct:.2f}%)")
    
    if duplicate_rows == 0:
        print("  ✓ No duplicates found!")
        return None
    
    # Analyze duplicate patterns
    dup_counts = df['review_id'].value_counts()
    max_dups = dup_counts.max()
    
    print(f"\n  Most duplicated review_id: {dup_counts.index[0]} (appears {max_dups} times)")
    
    # Show distribution of duplicate counts
    print(f"\n  Duplicate count distribution:")
    dup_dist = dup_counts.value_counts().sort_index()
    for count, freq in dup_dist.head(10).items():
        if count > 1:
            print(f"    {count} duplicates: {freq:,} review_ids")
    
    # Show examples of duplicates
    print(f"\n  Example duplicates:")
    example_id = dup_counts.index[0]
    examples = df[df['review_id'] == example_id].head(10)
    print(examples.to_string(index=False))
    
    return {
        'total_rows': total_rows,
        'unique_ids': unique_ids,
        'duplicate_rows': duplicate_rows,
        'duplicate_pct': duplicate_pct,
        'max_duplicates': int(max_dups)
    }

def clean_duplicates(df, strategy='first'):
    """
    Clean duplicates using specified strategy
    
    Strategies:
    - 'first': Keep first occurrence (RECOMMENDED - preserves order)
    - 'last': Keep last occurrence
    - 'average': Average probabilities (WARNING: changes order due to groupby)
    
    IMPORTANT: To preserve original order, use 'first' strategy!
    """
    print(f"\n[3/5] Cleaning duplicates (strategy: {strategy})...")
    
    before_count = len(df)
    
    if strategy == 'first':
        print(f"  Strategy: Keep FIRST occurrence of each review_id")
        print(f"  ✓ Preserves original order")
        df_clean = df.drop_duplicates(subset=['review_id'], keep='first')
    
    elif strategy == 'last':
        print(f"  Strategy: Keep LAST occurrence of each review_id")
        print(f"  ⚠️ WARNING: May change order")
        df_clean = df.drop_duplicates(subset=['review_id'], keep='last')
    
    elif strategy == 'average':
        print(f"  Strategy: AVERAGE probabilities for duplicate review_ids")
        print(f"  ⚠️ WARNING: groupby() changes order - use 'first' to preserve!")
        
        # Aggregate probabilities but preserve first occurrence order
        # Step 1: Calculate average per review_id
        avg_probs = df.groupby('review_id', as_index=False).agg({
            'probability_helpful': 'mean'
        })
        
        # Step 2: Get first occurrence index to preserve order
        first_idx = df.drop_duplicates(subset=['review_id'], keep='first').index
        
        # Step 3: Merge to restore original order
        df_first = df.loc[first_idx, ['review_id']].reset_index(drop=True)
        df_clean = df_first.merge(avg_probs, on='review_id', how='left')
        
        print(f"  ✓ Order preserved using first occurrence index")
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from: first, last, average")
    
    after_count = len(df_clean)
    removed_count = before_count - after_count
    
    print(f"  Before: {before_count:,} rows")
    print(f"  After: {after_count:,} rows")
    print(f"  Removed: {removed_count:,} duplicates ({removed_count/before_count*100:.2f}%)")
    
    return df_clean

def validate_cleaned_submission(df):
    """Validate cleaned submission meets requirements"""
    print(f"\n[4/5] Validating cleaned submission...")
    
    issues = []
    
    # Check 1: No duplicates
    unique_count = df['review_id'].nunique()
    total_count = len(df)
    
    if unique_count != total_count:
        issues.append(f"⚠️ Still has duplicates: {total_count - unique_count} duplicate rows")
    else:
        print(f"  ✓ No duplicates: {unique_count:,} unique review_ids")
    
    # Check 2: All probabilities in [0, 1]
    prob_col = df['probability_helpful']
    
    if prob_col.min() < 0 or prob_col.max() > 1:
        issues.append(f"⚠️ Invalid probabilities: min={prob_col.min():.4f}, max={prob_col.max():.4f}")
    else:
        print(f"  ✓ All probabilities in [0, 1]: min={prob_col.min():.4f}, max={prob_col.max():.4f}")
    
    # Check 3: No NULLs
    null_count = df.isnull().sum().sum()
    
    if null_count > 0:
        issues.append(f"⚠️ Contains NULL values: {null_count} nulls")
    else:
        print(f"  ✓ No NULL values")
    
    # Check 4: Correct columns
    expected_cols = ['review_id', 'probability_helpful']
    if list(df.columns) != expected_cols:
        issues.append(f"⚠️ Unexpected columns: {list(df.columns)}")
    else:
        print(f"  ✓ Correct columns: {list(df.columns)}")
    
    # Statistics
    print(f"\n  Statistics:")
    print(f"    Mean probability: {prob_col.mean():.4f}")
    print(f"    Median probability: {prob_col.median():.4f}")
    print(f"    Std deviation: {prob_col.std():.4f}")
    
    if issues:
        print(f"\n  ❌ VALIDATION FAILED:")
        for issue in issues:
            print(f"    {issue}")
        return False
    else:
        print(f"\n  ✅ VALIDATION PASSED - Ready for submission!")
        return True

def save_cleaned_submission(df, output_path, metadata=None):
    """Save cleaned submission CSV"""
    print(f"\n[5/5] Saving cleaned submission...")
    
    # Save CSV
    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved to: {output_path}")
    print(f"  ✓ File size: {os.path.getsize(output_path):,} bytes ({os.path.getsize(output_path)/1024/1024:.2f} MB)")
    
    # Save metadata
    if metadata:
        metadata_path = output_path.replace('.csv', '_metadata.json')
        
        metadata['cleaned_at'] = datetime.now().isoformat()
        metadata['output_file'] = output_path
        metadata['final_count'] = len(df)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, indent=2, fp=f)
        
        print(f"  ✓ Metadata saved to: {metadata_path}")

def main():
    parser = argparse.ArgumentParser(description='Clean duplicate review_ids in submission.csv')
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input submission CSV path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output cleaned submission CSV path')
    parser.add_argument('--strategy', type=str, default='first',
                       choices=['first', 'last', 'average'],
                       help='Deduplication strategy (default: first - preserves order)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SUBMISSION CLEANING PIPELINE")
    print("="*80)
    print(f"\nInput: {args.input}")
    print(f"Output: {args.output}")
    print(f"Strategy: {args.strategy}")
    
    try:
        # Load
        df = load_submission(args.input)
        
        # Analyze
        dup_stats = analyze_duplicates(df)
        
        if dup_stats is None:
            print("\n✓ No cleaning needed - submission is already clean!")
            return
        
        # Clean
        df_clean = clean_duplicates(df, strategy=args.strategy)
        
        # Validate
        is_valid = validate_cleaned_submission(df_clean)
        
        # Save
        metadata = {
            'input_file': args.input,
            'strategy': args.strategy,
            'before_cleaning': dup_stats,
            'validation_passed': is_valid
        }
        
        save_cleaned_submission(df_clean, args.output, metadata=metadata)
        
        print("\n" + "="*80)
        print("✅ CLEANING COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        if not is_valid:
            print("\n⚠️ WARNING: Validation failed - please review the cleaned file before submission")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    # Example usage:
    # python clean_submission.py --input output_v2/submission.csv --output output_v2/submission_clean.csv --strategy average
    
    import sys
    sys.exit(main())
