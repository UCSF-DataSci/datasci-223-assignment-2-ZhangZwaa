#!/usr/bin/env python3
"""
Test script for cohort analysis functionality.

This script tests that:
1. The cohort analysis script runs without errors
2. The script handles the generated data correctly
3. The output format matches the expected structure
"""

import os
import pytest
import subprocess
import polars as pl
from pathlib import Path
import hashlib

def get_git_root():
    """Get the git repository root directory."""
    try:
        result = subprocess.run(['git', 'rev-parse', '--show-toplevel'], 
                              capture_output=True, text=True, check=True)
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        # Fallback to current directory if not in git repo
        return Path.cwd()

def calculate_file_hash(filepath):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def test_script_runs():
    """Test that the script runs without errors."""
    # Get the git repository root directory
    project_root = get_git_root()
    
    # First check if we need to generate data
    data_file = project_root / "patients_large.csv"
    if not data_file.exists():
        # Generate data if file doesn't exist
        generate_script = project_root / "generate_large_health_data.py"
        subprocess.run(["python", str(generate_script)], check=True, cwd=str(project_root))
    else:
        # Verify hash of existing file
        expected_hash = "6f0a4bfef15ad875bd90a160f7cfdacff2def36c36a4af8ee06eade4d9611478"
        current_hash = calculate_file_hash(data_file)
        if current_hash != expected_hash:
            # Regenerate if hash doesn't match
            generate_script = project_root / "generate_large_health_data.py"
            subprocess.run(["python", str(generate_script)], check=True, cwd=str(project_root))
    
    # Run the analysis
    script_path = project_root / "3_cohort_analysis.py"
    result = subprocess.run(["python", str(script_path)], capture_output=True, text=True, cwd=str(project_root))
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

def test_output_format():
    """Test that the script produces correct output format."""
    # Get the git repository root directory
    project_root = get_git_root()
    
    # Run the script and capture output
    script_path = project_root / "3_cohort_analysis.py"
    result = subprocess.run(["python", str(script_path)], capture_output=True, text=True, cwd=str(project_root))
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"
    
    # Parse the output
    output = result.stdout
    assert "Cohort Analysis Results:" in output, "Missing results header"
    
    # Extract the table data - look for either type of table border
    table_lines = [line for line in output.split('\n') if '│' in line or '┆' in line]
    assert len(table_lines) >= 2, "Table not found in output"
    
    # Verify header row contains all expected columns
    header_row = table_lines[0]
    expected_columns = ["bmi_range", "avg_glucose", "patient_count", "avg_age"]
    for col in expected_columns:
        assert col in header_row.lower(), f"Missing column in header: {col}"
    
    # Verify all required BMI ranges are present (allow additional ranges)
    bmi_ranges = set()
    for line in table_lines[2:]:  # Skip header and separator lines
        # Split on either type of table border
        cells = [cell.strip() for cell in line.replace('┆', '│').split('│')[1:-1]]
        if len(cells) == 4:  # bmi_range, avg_glucose, patient_count, avg_age
            bmi_ranges.add(cells[0])
    
    required_ranges = {"Underweight", "Normal", "Overweight", "Obese"}
    assert required_ranges.issubset(bmi_ranges), f"Missing required BMI ranges. Required: {required_ranges}, Found: {bmi_ranges}"

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 