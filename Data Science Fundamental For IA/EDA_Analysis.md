# Titanic Dataset - Exploratory Data Analysis (EDA)

## Overview
This document provides an exploratory data analysis of the famous Titanic dataset, which contains information about passengers aboard the RMS Titanic that sank in 1912.

## Dataset Information
- **Source**: [Titanic Dataset from Kaggle](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- **Total Records**: 891 passengers
- **Features**: 12 columns
- **Target Variable**: Survived (0 = No, 1 = Yes)

## Dataset Structure

### Column Details
| Column | Non-Null Count | Data Type | Description |
|--------|----------------|-----------|-------------|
| PassengerId | 891 | int64 | Unique identifier for each passenger |
| Survived | 891 | int64 | Survival status (0 = No, 1 = Yes) |
| Pclass | 891 | int64 | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| Name | 891 | str | Passenger name |
| Sex | 891 | str | Gender |
| Age | 714 | float64 | Age in years |
| SibSp | 891 | int64 | Number of siblings/spouses aboard |
| Parch | 891 | int64 | Number of parents/children aboard |
| Ticket | 891 | str | Ticket number |
| Fare | 891 | float64 | Passenger fare |
| Cabin | 204 | str | Cabin number |
| Embarked | 889 | str | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

## Key Findings

### Sample Data Preview
The first 5 records show a mix of passengers from different classes and backgrounds:

```
   PassengerId  Survived  Pclass     Fare Cabin  Embarked
0            1         0       3   7.2500   NaN         S
1            2         1       1  71.2833   C85         C
2            3         1       3   7.9250   NaN         S
3            4         1       1  53.1000  C123         S
4            5         0       3   8.0500   NaN         S
```

### Statistical Summary

#### Survival Statistics
- **Overall Survival Rate**: 38.38%
- **Total Survivors**: 342 out of 891 passengers
- **Total Deaths**: 549 passengers

#### Demographic Insights
- **Average Age**: 29.7 years (from 714 non-null records)
- **Age Range**: 0.42 to 80 years
- **Median Age**: 28 years

#### Economic Indicators
- **Average Fare**: $32.20
- **Fare Range**: $0 to $512.33
- **Median Fare**: $14.45

#### Family Structure
- **Average Siblings/Spouses**: 0.52 per passenger
- **Average Parents/Children**: 0.38 per passenger
- **Maximum Family Size**: 8 siblings/spouses, 6 parents/children

### Data Quality Issues

#### Missing Values
1. **Age**: 177 missing values (19.87% missing)
   - Could impact age-based survival analysis
   
2. **Cabin**: 687 missing values (77.11% missing)
   - Significant missing data, may limit cabin-based analysis
   
3. **Embarked**: 2 missing values (0.22% missing)
   - Minimal impact on analysis

## Next Steps for Analysis

### Recommended Visualizations
1. **Survival Rate by Gender**: Compare male vs female survival rates
2. **Survival Rate by Class**: Analyze survival across passenger classes
3. **Age Distribution**: Histogram of passenger ages with survival overlay
4. **Fare Distribution**: Box plot showing fare distribution by class and survival
5. **Family Size Impact**: Correlation between family size and survival

### Advanced Analysis Opportunities
1. **Feature Engineering**: 
   - Create family size feature (SibSp + Parch + 1)
   - Extract titles from names (Mr., Mrs., Miss, etc.)
   - Age group categorization

2. **Correlation Analysis**: 
   - Examine relationships between features
   - Identify key survival predictors

3. **Machine Learning Preparation**:
   - Handle missing values
   - Encode categorical variables
   - Feature scaling for numerical variables

## Technical Implementation

### Code Used
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the titanic dataset
url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)
df = pd.read_csv(url)

# Inspect Data
print(df.head())
print(df.info())
print(df.describe())
```

### Environment
- **Python Version**: 3.14.2
- **Libraries Used**: pandas (3.0.0), matplotlib (3.10.8)
- **Data Source**: Remote CSV file

## Conclusion
The Titanic dataset provides a rich source for exploratory data analysis with clear patterns emerging around survival rates. The 38.4% overall survival rate varies significantly across different passenger characteristics, making it an excellent dataset for both statistical analysis and machine learning applications.

---
*Analysis performed on: February 5, 2026*
*File: EDA.py*