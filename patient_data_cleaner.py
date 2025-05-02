#!/usr/bin/env python3
"""
Patient Data Cleaner

This script standardizes and filters patient records according to specific rules:

Data Cleaning Rules:
1. Names: Capitalize each word (e.g., "john smith" -> "John Smith")
2. Ages: Convert to integers, set invalid ages to 0
3. Filter: Remove patients under 18 years old
4. Remove any duplicate records

Input JSON format:
    [
        {
            "name": "john smith",
            "age": "32",
            "gender": "male",
            "diagnosis": "hypertension"
        },
        ...
    ]

Output:
- Cleaned list of patient dictionaries
- Each patient should have:
  * Properly capitalized name
  * Integer age (≥ 18)
  * Original gender and diagnosis preserved
- No duplicate records
- Prints cleaned records to console

Example:
    Input: {"name": "john smith", "age": "32", "gender": "male", "diagnosis": "flu"}
    Output: {"name": "John Smith", "age": 32, "gender": "male", "diagnosis": "flu"}

Usage:
    python patient_data_cleaner.py
"""

import json
import os
import pdb
import pandas as pd
import sys

def load_patient_data(filepath):
    """
    Load patient data from a JSON file.
    
    Args:
        filepath (str): Path to the JSON file
        
    Returns:
        list: List of patient dictionaries
    """
    # BUG: No error handling for file not found
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("File not found!")
        sys.exit(1)

def clean_patient_data(patients):
    """
    Clean patient data by:
    - Capitalizing names
    - Converting ages to integers
    - Filtering out patients under 18
    - Removing duplicates
    
    Args:
        patients (list): List of patient dictionaries
        
    Returns:
        list: Cleaned list of patient dictionaries
    """
    cleaned_patients = []
    seen = set()

    for patient in patients:
        # BUG: Typo in key 'nage' instead of 'name'
        # FIX: Correct the name
        patient['name'] = patient['name'].title()
        
        # BUG: Wrong method name (fill_na vs fillna)
        # FIX: Using int to convert ages to integers
        patient['age'] = int(patient.get('age', 0))
        
        # BUG: Wrong method name (drop_duplcates vs drop_duplicates)
        # FIX: Drop duplicates at the end of this function
        # patient = patient.drop_duplicates()
        
        # BUG: Wrong comparison operator (= vs ==)
        # FIX: Change to >= to correctly fliter patients under 18 out
        if patient['age'] >= 18:
            # BUG: Logic error - keeps patients under 18 instead of filtering them out
            patient_tuple = tuple(sorted(patient.items()))
            # FIX: Duplicates should be removed here
            if patient_tuple not in seen:
                seen.add(patient_tuple)
                cleaned_patients.append(patient)
    
    # BUG: Missing return statement for empty list
    if not cleaned_patients:
        print("The list is empty!")
        return None
    
    return cleaned_patients

def main():
    """Main function to run the script."""
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the data file
    data_path = os.path.join(script_dir, 'data', 'raw', 'patients.json')
    
    # BUG: No error handling for load_patient_data failure
    patients = load_patient_data(data_path)
    
    # Clean the patient data
    cleaned_patients = clean_patient_data(patients)
    print(f"Type: {type(cleaned_patients)}")
    # BUG: No check if cleaned_patients is None
    # Print the cleaned patient data
    print("Cleaned Patient Data:")
    for patient in cleaned_patients:
        # BUG: Using 'name' key but we changed it to 'nage'
        print(f"Name: {patient['name']}, Age: {patient['age']}, Diagnosis: {patient['diagnosis']}")
    
    # Return the cleaned data (useful for testing)
    return cleaned_patients

if __name__ == "__main__":
    main()