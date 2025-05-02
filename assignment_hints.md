# Assignment Hints

This document provides comprehensive hints and guidance for completing the assignment.

## General Debugging Tips

1. **Start Small**: Begin with a small dataset or test case
2. **Incremental Testing**: Test each function separately
3. **Print Statements**: Add print statements to inspect variables
4. **Debugger**: Use a debugger to step through code
5. **Documentation**: Read the function docstrings carefully

## Part 1: Debugging

### Common Debugging Techniques

1. **Print Statements**: Add print statements to inspect variables
   ```python
   print(f"Variable value: {variable}")
   ```

2. **pdb**: Use Python's debugger
   ```python
   import pdb; pdb.set_trace()
   ```

3. **VS Code Debugger**: Set breakpoints and inspect variables

4. **Logging**: Use Python's logging module
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   logging.debug(f"Variable value: {variable}")
   ```

### Bug Documentation Format

When documenting bugs, use these comment formats:

```python
# BUG: Description of the bug
# FIX: Description of how you fixed it
```

### Patient Data Cleaner Hints

#### General Approach
1. Start by running the script with the sample data
2. Compare the output with the expected output in the documentation
3. Use a debugger to inspect values at key processing points
4. Pay attention to error messages - they often point to the general area of the bug

#### Progressive Hints

##### Hint 1: Data Loading
- Check if the data is being loaded correctly
- Are all fields present in the expected format?
- Is the data path correct?

##### Hint 2: Name Capitalization
- How are names being processed?
- Check for typos in key names
- Are you modifying the original data or creating a new copy?

##### Hint 3: Age Processing
- How are ages being converted from strings to integers?
- Are invalid ages being handled correctly?
- Is the age filtering logic correct?

##### Hint 4: Duplicate Removal
- How are duplicates being identified?
- Are you using the correct method to remove duplicates?
- Are you preserving all necessary fields?

##### Hint 5: Output Format
- Does the output match the expected format?
- Are all required fields present?
- Is the data being printed correctly?

### Medication Dosage Calculator Hints

#### General Approach
1. Start by running the script with the sample data
2. Compare the output with the expected output in the documentation
3. Use a debugger to inspect values at key calculation points
4. Pay attention to error messages - they often point to the general area of the bug

#### Progressive Hints

##### Hint 1: Data Loading
- Check if the data is being loaded correctly
- Are all fields present in the expected format?
- Is the data path correct?

##### Hint 2: Medication Names
- How are medication names being handled?
- Check for case sensitivity
- Look for typos in medication names
- Are the medications in LOADING_DOSE_MEDICATIONS matching DOSAGE_FACTORS?

##### Hint 3: Calculations
- Review the dosage calculation formula in the documentation
- What mathematical operations are being used?
- Are loading doses being calculated correctly?
- Is the total medication sum working?

##### Hint 4: Data Structures
- Are lists and dictionaries properly formatted?
- Check for syntax errors in data structures
- Are we modifying the right dictionaries?

##### Hint 5: Output
- Compare each field in the output with the documentation
- Are warnings being generated correctly?
- Is loading dose information accurate?

## Part 2: Big Data Analysis

### Polars Basics

1. **Reading Data**:
   ```python
   # Eager evaluation
   df = pl.read_csv("file.csv")
   
   # Lazy evaluation
   lf = pl.scan_csv("file.csv")
   ```

2. **Filtering**:
   ```python
   # Eager
   df_filtered = df.filter(pl.col("column") > value)
   
   # Lazy
   lf_filtered = lf.filter(pl.col("column") > value)
   ```

3. **Grouping and Aggregation**:
   ```python
   # Eager
   df_grouped = df.groupby("column").agg([
       pl.col("value").mean().alias("avg_value"),
       pl.count().alias("count")
   ])
   
   # Lazy
   lf_grouped = lf.groupby("column").agg([
       pl.col("value").mean().alias("avg_value"),
       pl.count().alias("count")
   ])
   ```

4. **Executing Lazy Queries**:
   ```python
   # Without streaming
   df_result = lf.collect()
   
   # With streaming
   df_result = lf.collect(streaming=True)
   ```

### Performance Tips

1. **Early Filtering**: Filter data as early as possible in the pipeline
2. **Column Selection**: Select only needed columns
3. **Streaming**: Use streaming for large datasets
4. **Parquet Format**: Convert CSV to Parquet for better performance

### Cohort Analysis Hints

#### General Approach
1. Understand the data structure from the sample data
2. Plan the analysis steps before coding
3. Test with a small subset of data first
4. Verify the output format matches requirements

#### Progressive Hints

##### Hint 1: Data Loading and Conversion
- How do you convert CSV to Parquet format?
- What are the advantages of Parquet for this analysis?
- How do you use polars' lazy evaluation?
- What's the difference between `pl.read_csv()` and `pl.scan_csv()`?

##### Hint 2: BMI Filtering
- How do you filter out BMI outliers (< 10 or > 60)?
- Where should the filtering happen in the query pipeline?
- Why is early filtering important for performance?
- How do you use polars' filter expressions?

##### Hint 3: BMI Binning
- How do you create BMI ranges using `cut()`?
- What are the bin boundaries and labels?
- How do you handle the interval boundaries?
- What does `left_closed=True` mean?

##### Hint 4: Aggregation
- How do you group by BMI range?
- What aggregations are needed for each column?
- How do you alias columns to match the expected output?
- How do you ensure the columns are in the correct order?

##### Hint 5: Streaming Execution
- How do you execute the query with streaming?
- What are the benefits of streaming for large datasets?
- How do you collect the results?
- How do you handle memory constraints?

## Submission Checklist

Before submitting, ensure:

1. All tests pass
2. Bug documentation is complete
3. Code is clean and well-commented
4. Output format matches requirements exactly
5. Files are named correctly 