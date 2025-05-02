#!/usr/bin/env python3
"""
Basic test for med_dosage_calculator.py

This script tests that:
1. The medication dosage calculator runs without errors
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
    script_path = Path(__file__).parent.parent.parent / "2_med_dosage_calculator.py"
    with open(script_path) as f:
        content = f.read()
    
    # Check that buggy lines have been fixed
    # Original buggy LOADING_DOSE_MEDICATIONS without commas
    assert 'LOADING_DOSE_MEDICATIONS = [\n    "amiodarone"\n    "lorazepam"\n    "fentynal"\n]' not in content
    
    # Original buggy medication name handling
    assert 'factor = DOSAGE_FACTORS.get(medication + \'s\', 0)' not in content
    assert "factor = DOSAGE_FACTORS.get(medication + 's', 0)" not in content
    
    # Original buggy dosage calculations
    assert 'base_dosage = weight + factor' not in content
    assert 'final_dosage = base_dosage + base_dosage' not in content
    
    # Original buggy medication names in warnings
    assert 'if medication == "epinephrin":' not in content
    assert 'elif medication == "fentynal":' not in content

def test_script_runs():
    """Test that the script runs without errors."""
    script_path = Path(__file__).parent.parent.parent / "2_med_dosage_calculator.py"
    result = subprocess.run(["python", str(script_path)], capture_output=True, text=True)
    assert result.returncode == 0

if __name__ == "__main__":
    pytest.main(["-v", __file__])