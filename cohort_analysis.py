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
    cohort_results = pl.scan_parquet(parquet_file).pipe(
        lambda df: df.filter((pl.col("BMI") >= 10) & (pl.col("BMI") <= 60))  # Filter BMI in reasonable range
    ).pipe(
        lambda df: df.select(["BMI", "Glucose", "Age"])  # Select relevant columns
    ).pipe(
        lambda df: df.with_columns(
            pl.col("BMI").cut(  # Categorize BMI into defined ranges
                breaks=[10, 18.5, 25, 30, 60],
                labels=["Underweight", "Normal", "Overweight", "Obese"],
                left_closed=True
            ).alias("bmi_range")
        )
    ).pipe(
        lambda df: df.groupby("bmi_range").agg([  # Group by BMI range and calculate aggregates
            pl.col("Glucose").mean().alias("avg_glucose"),
            pl.count().alias("patient_count"),
            pl.col("Age").mean().alias("avg_age")
        ])
    ).collect(streaming=True)  # Collect results

    return cohort_results

def main():
    # Input file path
    input_file = "patients_large.csv"
    
    try:
        # Run the cohort analysis
        results = analyze_patient_cohorts(input_file)
        
        # Print the results of the cohort analysis
        print("\nCohort Analysis Summary:")
        print(results)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
