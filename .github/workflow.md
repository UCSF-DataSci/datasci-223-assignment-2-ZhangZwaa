# Assignment 2 Workflow Log

## Overview
This log tracks our progress in completing Assignment 2, which involves debugging Python scripts and analyzing health data.

## Tasks
1. Fix bugs in `1_patient_data_cleaner.py`
2. Fix bugs in `2_med_dosage_calculator.py`
3. Complete `3_cohort_analysis.py`
4. Run tests to verify fixes

## Progress Log

### 2024-03-21
- Created workflow log
- Started debugging `1_patient_data_cleaner.py`
- Identified initial bugs:
  - Syntax error in comparison operator
  - Logic error in age filtering
  - Missing error handling
  - Incorrect key names
  - Missing return statement

### Fixes for 1_patient_data_cleaner.py
1. Added error handling for file not found in `load_patient_data()`
2. Fixed name capitalization using correct key 'name' instead of 'nage'
3. Added proper age conversion with error handling
4. Implemented duplicate removal using a set of unique patient identifiers
5. Fixed age filtering logic to only include patients 18 or older
6. Added proper error handling in main function
7. Fixed return value handling for empty cases

### Fixes for 2_med_dosage_calculator.py
1. Added commas between list items in LOADING_DOSE_MEDICATIONS
2. Corrected typo in medication name (fentynal -> fentanyl)
3. Added error handling for file not found in `load_patient_data()`
4. Added checks for required keys (weight, medication)
5. Removed 's' from medication name when looking up dosage factors
6. Fixed dosage calculation to use multiplication instead of addition
7. Added check for 'is_first_dose' key with default value
8. Corrected condition for loading dose application
9. Fixed loading dose calculation to use multiplication
10. Corrected medication name typos in warnings
11. Added error handling for invalid patient data
12. Added checks for required keys in output printing
13. Improved error handling in main function

### Implementation of 3_cohort_analysis.py
1. Added proper docstring and documentation
2. Implemented CSV to Parquet conversion for efficient processing
3. Added BMI outlier filtering (values < 10 or > 60)
4. Implemented BMI range categorization:
   - Underweight: 10 ≤ BMI < 18.5
   - Normal: 18.5 ≤ BMI < 25
   - Overweight: 25 ≤ BMI < 30
   - Obese: 30 ≤ BMI ≤ 60
5. Added statistical calculations:
   - Mean glucose level by BMI range
   - Patient count by BMI range
   - Mean age by BMI range
6. Implemented proper error handling for file operations
7. Added cleanup of temporary parquet file
8. Used polars' lazy evaluation for efficient processing
9. Added sorting of results by BMI range 