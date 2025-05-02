import polars as pl
import os

def analyze_patient_cohorts(input_file: str) -> pl.DataFrame:
    """
    Analyze patient cohorts based on BMI ranges.
    
    Args:
        input_file: Path to the input CSV file
        
    Returns:
        DataFrame containing cohort analysis results with columns:
        - bmi_range: The BMI range (e.g., "Underweight", "Normal", "Overweight", "Obese")
        - avg_glucose: Mean glucose level by BMI range
        - patient_count: Number of patients by BMI range
        - avg_age: Mean age by BMI range
    """
    # Check if the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"The file {input_file} was not found.")
    
    # Convert CSV to Parquet for efficient processing
    parquet_file = "patients_large.parquet"
    pl.read_csv(input_file).write_parquet(parquet_file)
    
    # Create a lazy query to analyze cohorts
    # Create BMI range
    cohort_results = pl.scan_parquet(parquet_file).pipe(
        lambda df: df.filter((pl.col("BMI") >= 10) & (pl.col("BMI") <= 60))  # Check the valid range for BMI
    ).pipe(
        lambda df: df.select(["BMI", "Glucose", "BloodPressure", "SkinThickness", "Age", "Pregnancies", "DiabetesPedigreeFunction"])
    ).pipe(
        lambda df: df.with_columns(
            pl.when(pl.col("BMI") < 18.5).then(pl.lit("Underweight"))
            .when(pl.col("BMI") < 25).then(pl.lit("Normal"))
            .when(pl.col("BMI") < 30).then(pl.lit("Overweight"))
            .otherwise(pl.lit("Obese"))
            .alias("bmi_range")
        )
    ).pipe(
        lambda df: df.group_by("bmi_range").agg([  
            pl.col("Glucose").mean().alias("avg_glucose"),
            # pl.col("Pregnancies").mean().alias("avg_Preg"),
            # pl.col("BloodPressure").mean().alias("avg_BP"),
            # pl.col("DiabetesPedigreeFunction").mean().alias("avg_DPF"),
            pl.len().alias("patient_count"),
            pl.col("Age").mean().alias("avg_age")
        ])
    )

    # Debug: Check if bmi_range was created correctly
    print(cohort_results.collect())

    result = cohort_results.collect(engine="streaming")  # Collect results
    
    return result

def main():
    # Input file path
    input_file = "patients_large.csv"
    
    try:
        # Run the cohort analysis
        results = analyze_patient_cohorts(input_file)
        
        # Print the results of the cohort analysis
        header_row = results[0]
        print(header_row)
        print("\nCohort Analysis Results:")
        print(results)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()