#!/usr/bin/env python3
"""
Data cleaning and validation script for RCL prediction CSV files.

This script checks the integrity of CSV files and reports:
- Missing or invalid RCL annotations
- Sequence format issues
- Out of range indices
- Inconsistent RCL sequences

Usage:
    python data/scripts/data_cleaning.py --input data/raw/Alphafold_RCL_annotations.csv
    python data/scripts/data_cleaning.py --input data/raw/*.csv --output data/cleaned/
"""

import argparse
import pandas as pd
import sys
from pathlib import Path
from collections import defaultdict


class ValidationError:
    """Store information about a validation error."""
    
    def __init__(self, row_num, field, reason, value=None):
        self.row_num = row_num
        self.field = field
        self.reason = reason
        self.value = value
    
    def __str__(self):
        if self.value is not None:
            return f"Row {self.row_num:5d} | Field: {self.field:15s} | {self.reason} | Value: {self.value}"
        return f"Row {self.row_num:5d} | Field: {self.field:15s} | {self.reason}"


class DataValidator:
    """Validate RCL annotation CSV files."""
    
    def __init__(self, max_length=1024):
        self.max_length = max_length
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(int)
        
    def validate_file(self, csv_path):
        """Validate a CSV file and collect errors."""
        print(f"\nValidating: {csv_path}")
        print("=" * 80)
        
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"ERROR: Failed to read CSV file: {e}")
            return False
        
        print(f"Total rows: {len(df)}")
        
        # Check required columns
        required_cols = ['id', 'Sequence']
        optional_cols = ['rcl_start', 'rcl_end', 'rcl_seq']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"ERROR: Missing required columns: {missing_cols}")
            return False
        
        has_annotation_cols = all(col in df.columns for col in optional_cols)
        if not has_annotation_cols:
            print(f"WARNING: Missing annotation columns. Only checking sequence integrity.")
        
        # Validate each row
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(int)
        
        for idx, row in df.iterrows():
            row_num = idx + 2  # +2 because: 1-indexed + header row
            self._validate_row(row_num, row, has_annotation_cols)
        
        # Print results
        self._print_results()
        
        return len(self.errors) == 0
    
    def _validate_row(self, row_num, row, has_annotation_cols):
        """Validate a single row."""
        
        # Validate ID
        if pd.isna(row.get('id')):
            self.errors.append(ValidationError(row_num, 'id', 'Missing ID'))
            self.stats['missing_id'] += 1
            return
        
        # Validate Sequence
        sequence = row.get('Sequence')
        if pd.isna(sequence):
            self.errors.append(ValidationError(row_num, 'Sequence', 'Missing sequence'))
            self.stats['missing_sequence'] += 1
            return
        
        sequence = str(sequence).strip()
        
        # Check sequence length
        if len(sequence) == 0:
            self.errors.append(ValidationError(row_num, 'Sequence', 'Empty sequence'))
            self.stats['empty_sequence'] += 1
            return
        
        if len(sequence) > self.max_length:
            self.warnings.append(ValidationError(
                row_num, 'Sequence', 
                f'Sequence too long (>{self.max_length})', 
                len(sequence)
            ))
            self.stats['too_long'] += 1
        
        # Check for invalid amino acids
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        invalid_chars = set(sequence.upper()) - valid_aa
        if invalid_chars:
            self.errors.append(ValidationError(
                row_num, 'Sequence', 
                f'Invalid amino acids', 
                ''.join(sorted(invalid_chars))
            ))
            self.stats['invalid_amino_acids'] += 1
        
        # Validate RCL annotations if present
        if has_annotation_cols:
            self._validate_rcl_annotation(row_num, row, sequence)
    
    def _validate_rcl_annotation(self, row_num, row, sequence):
        """Validate RCL annotation fields."""
        
        rcl_start = row.get('rcl_start')
        rcl_end = row.get('rcl_end')
        rcl_seq = row.get('rcl_seq')
        
        # Check for missing or invalid values
        invalid_values = ['no result', 'no start', 'no end', '#value!', 'nan', '']
        
        rcl_start_str = str(rcl_start).lower().strip()
        rcl_end_str = str(rcl_end).lower().strip()
        
        # Check if annotation is missing
        if (pd.isna(rcl_start) or pd.isna(rcl_end) or 
            rcl_start_str in invalid_values or rcl_end_str in invalid_values):
            
            self.warnings.append(ValidationError(
                row_num, 'rcl_start/end', 
                'Missing or invalid RCL annotation',
                f'start={rcl_start}, end={rcl_end}'
            ))
            self.stats['missing_annotation'] += 1
            return
        
        # Validate numeric values
        try:
            start = int(rcl_start)
            end = int(rcl_end)
        except (ValueError, TypeError):
            self.errors.append(ValidationError(
                row_num, 'rcl_start/end', 
                'Non-numeric RCL indices',
                f'start={rcl_start}, end={rcl_end}'
            ))
            self.stats['non_numeric_indices'] += 1
            return
        
        # Validate index ranges
        if start < 1:
            self.errors.append(ValidationError(
                row_num, 'rcl_start', 
                'Start index < 1 (should be 1-indexed)',
                start
            ))
            self.stats['invalid_start_index'] += 1
        
        if end < start:
            self.errors.append(ValidationError(
                row_num, 'rcl_end', 
                'End index less than start index',
                f'start={start}, end={end}'
            ))
            self.stats['end_before_start'] += 1
        
        if start > len(sequence):
            self.errors.append(ValidationError(
                row_num, 'rcl_start', 
                'Start index beyond sequence length',
                f'start={start}, seq_len={len(sequence)}'
            ))
            self.stats['start_beyond_sequence'] += 1
        
        if end > len(sequence):
            self.errors.append(ValidationError(
                row_num, 'rcl_end', 
                'End index beyond sequence length',
                f'end={end}, seq_len={len(sequence)}'
            ))
            self.stats['end_beyond_sequence'] += 1
        
        # Validate RCL sequence if present
        if pd.notna(rcl_seq) and str(rcl_seq).strip():
            rcl_seq_str = str(rcl_seq).strip()
            
            # Extract expected RCL sequence from coordinates
            if 1 <= start <= end <= len(sequence):
                expected_rcl = sequence[start-1:end]  # Convert to 0-indexed
                
                if rcl_seq_str != expected_rcl:
                    self.errors.append(ValidationError(
                        row_num, 'rcl_seq', 
                        'RCL sequence does not match coordinates',
                        f'expected={expected_rcl}, got={rcl_seq_str}'
                    ))
                    self.stats['mismatched_rcl_seq'] += 1
        
        # Check RCL length
        if 1 <= start <= end <= len(sequence):
            rcl_length = end - start + 1
            if rcl_length < 10:
                self.warnings.append(ValidationError(
                    row_num, 'rcl_length', 
                    f'RCL very short ({rcl_length} residues)',
                    f'start={start}, end={end}'
                ))
                self.stats['short_rcl'] += 1
            elif rcl_length > 50:
                self.warnings.append(ValidationError(
                    row_num, 'rcl_length', 
                    f'RCL very long ({rcl_length} residues)',
                    f'start={start}, end={end}'
                ))
                self.stats['long_rcl'] += 1
    
    def _print_results(self):
        """Print validation results."""
        
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        
        # Print statistics
        print("\nStatistics:")
        print(f"  Missing ID:                    {self.stats['missing_id']}")
        print(f"  Missing sequence:              {self.stats['missing_sequence']}")
        print(f"  Empty sequence:                {self.stats['empty_sequence']}")
        print(f"  Sequence too long (>{self.max_length}):   {self.stats['too_long']}")
        print(f"  Invalid amino acids:           {self.stats['invalid_amino_acids']}")
        print(f"  Missing RCL annotation:        {self.stats['missing_annotation']}")
        print(f"  Non-numeric indices:           {self.stats['non_numeric_indices']}")
        print(f"  Invalid start index:           {self.stats['invalid_start_index']}")
        print(f"  End before start:              {self.stats['end_before_start']}")
        print(f"  Start beyond sequence:         {self.stats['start_beyond_sequence']}")
        print(f"  End beyond sequence:           {self.stats['end_beyond_sequence']}")
        print(f"  Mismatched RCL sequence:       {self.stats['mismatched_rcl_seq']}")
        print(f"  Short RCL (<10 residues):      {self.stats['short_rcl']}")
        print(f"  Long RCL (>50 residues):       {self.stats['long_rcl']}")
        
        # Print errors
        if self.errors:
            print(f"\n{'=' * 80}")
            print(f"ERRORS: {len(self.errors)}")
            print("=" * 80)
            for error in self.errors[:50]:  # Show first 50 errors
                print(error)
            if len(self.errors) > 50:
                print(f"\n... and {len(self.errors) - 50} more errors")
        else:
            print(f"\n✓ No errors found!")
        
        # Print warnings
        if self.warnings:
            print(f"\n{'=' * 80}")
            print(f"WARNINGS: {len(self.warnings)}")
            print("=" * 80)
            for warning in self.warnings[:50]:  # Show first 50 warnings
                print(warning)
            if len(self.warnings) > 50:
                print(f"\n... and {len(self.warnings) - 50} more warnings")
        else:
            print(f"\n✓ No warnings!")
        
        print("\n" + "=" * 80)
        if self.errors:
            print(f"❌ VALIDATION FAILED: {len(self.errors)} errors found")
        else:
            print(f"✓ VALIDATION PASSED")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Validate RCL annotation CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check single file
  python data/scripts/data_cleaning.py --input data/raw/Alphafold_RCL_annotations.csv
  
  # Check multiple files
  python data/scripts/data_cleaning.py --input data/raw/*.csv
  
  # Check with custom max length
  python data/scripts/data_cleaning.py --input data/raw/file.csv --max-length 2048
        """
    )
    
    parser.add_argument('--input', '-i', required=True, nargs='+',
                       help='Input CSV file(s) to validate')
    parser.add_argument('--max-length', type=int, default=1024,
                       help='Maximum sequence length (default: 1024)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output directory for validation reports (optional)')
    
    args = parser.parse_args()
    
    # Process each input file
    all_valid = True
    
    for input_file in args.input:
        input_path = Path(input_file)
        
        if not input_path.exists():
            print(f"ERROR: File not found: {input_file}")
            all_valid = False
            continue
        
        validator = DataValidator(max_length=args.max_length)
        is_valid = validator.validate_file(input_path)
        
        if not is_valid:
            all_valid = False
        
        # Save report if output directory specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = output_dir / f"{input_path.stem}_validation_report.txt"
            
            with open(report_file, 'w') as f:
                f.write(f"Validation Report for: {input_file}\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("ERRORS:\n")
                for error in validator.errors:
                    f.write(str(error) + "\n")
                
                f.write("\nWARNINGS:\n")
                for warning in validator.warnings:
                    f.write(str(warning) + "\n")
                
                f.write("\nSTATISTICS:\n")
                for key, value in validator.stats.items():
                    f.write(f"  {key}: {value}\n")
            
            print(f"\n✓ Report saved to: {report_file}")
    
    print("\n" + "=" * 80)
    if all_valid:
        print("✓ All files passed validation")
        sys.exit(0)
    else:
        print("❌ Some files failed validation")
        sys.exit(1)


if __name__ == '__main__':
    main()
