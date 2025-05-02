#!/usr/bin/env python3
"""
Basic test for patient_data_cleaner.py

This script tests that:
1. The patient data cleaner runs without errors
2. The original buggy lines have been fixed
"""
import sys
sys.path.append('/workspaces/datasci-223-assignment-2-ZhangZwaa')
import os
import pytest
import subprocess
from pathlib import Path

def test_buggy_lines_fixed():
    """Test that the original buggy lines have been fixed."""
    script_path = Path(__file__).parent.parent.parent / "1_patient_data_cleaner.py"
    with open(script_path) as f:
        content = f.read()
    
    # Check that buggy lines have been fixed
    # Original buggy key name
    assert "patient['nage'] = patient['name'].title()" not in content
    
    # Original buggy method names
    assert "patient['age'] = patient['age'].fill_na(0)" not in content
    assert "patient = patient.drop_duplcates()" not in content
    
    # Original buggy comparison and logic
    assert "if patient['age'] = 18:" not in content
    
    # Original buggy return for empty list
    assert "if not cleaned_patients:\n        return None" not in content
    
    # Original buggy key usage after change
    assert "print(f\"Name: {patient['name']}, Age: {patient['age']}, Diagnosis: {patient['diagnosis']}\")" not in content

def test_script_runs():
    """Test that the script runs without errors."""
    script_path = Path(__file__).parent.parent.parent / "1_patient_data_cleaner.py"
    result = subprocess.run(["python", str(script_path)], capture_output=True, text=True)
    assert result.returncode == 0

if __name__ == "__main__":
    pytest.main(["-v", __file__])