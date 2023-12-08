import streamlit as st
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Create a list of pages
pages = ['Introduction to dataset','Cleaning of dataset','Descriptive Analysis','Prediction of Salary']

# Add a selectbox to the sidebar
page = st.sidebar.selectbox('Choose a page:', pages)

# Display the selected page
if page == 'Introduction to dataset':
    # title
    st.title('Census Income')
    # sub title
    st.markdown('### **Introduction :**')
    # paragraph or a sentence
    st.markdown('Extraction was done by Barry Becker from the 1994 Census database. The prediction task is to determine whether a person makes over 50K a year.')
    # putting hyperlinks
    st.markdown('### **Reference :**')
    st.write("Reference: Kaggle Dataset: [link](https://www.kaggle.com/datasets/ayessa/salary-prediction-classification)")
    st.write("Source: UCI Machine Learning Repository: [link](https://archive.ics.uci.edu/ml/datasets/Census+Income)")
    st.write("Thumbnail: Deskera Blog: [link](https://www.deskera.com/blog/net-salary/)")
    # making introduction lists
    st.markdown('### **About Columns :**')
    st.markdown('Columns are:')
    st.markdown('- AGE: Age of the individual')
    st.markdown('- WORKCLASS: Work class of the individual')
    st.markdown('- FNLWGT: Annual Salary of an Individual')
    st.markdown('- EDUCATION: Level of education')
    st.markdown('- EDUCATION-NUM: Years of Study')
    st.markdown('- MARITAL-STATUS: Marital Status of the individual.')
    st.markdown('- OCCUPATION: Occupation of the individual')
    st.markdown('- OCCUPATION-NUM: Years of Work Experience')
    st.markdown('- RELATIONSHIP: Relationship to the individual')
    st.markdown('- RACE: Race of the individual')
    st.markdown('- SEX: Sex/Gender of the individual')
    st.markdown('- HOURS-PER-WEEK: Working hours per week')
    st.markdown('- NATIVE COUNTRY: The country of origin of the individual')
    st.markdown('- SALARY: If it is less than or greater than 50,000 each year')
elif page == 'Cleaning of dataset':
    # title
    st.title('Cleaning Methods')
    # Using pandas to read the Excel file
    df = pd.read_excel("Salary_Prediction_Classification.xlsx")
    # initial dataset
    st.markdown('### **Loading Uncleaned Dataset :**')
    # Displaying the DataFrame
    st.write(df)
    st.code("""
# Initial dataset
df = pd.read_excel("Salary_Prediction_Classification.xlsx")
""", language='python')
    # Getting dimension
    st.markdown('### **Dataset Dimensions :**')
    # Get the number of rows and columns
    rows, cols = df.shape
    # Display the number of rows and columns in Streamlit
    st.write(f'The dataset has {rows} rows and {cols} columns.')
    # Display the code
    st.code("""
# Getting dimension
rows, cols = df.shape
""", language='python')
    # getting data types
    st.markdown('### **Column type :**')
    # Get the data type of each column
    data_types = df.dtypes
    # Display the data types in Streamlit
    st.write(data_types)
    st.code("""
# Getting data types
data_types = df.dtypes
st.write(data_types)
""", language='python')
    # Checking for NaN Values
    st.markdown('### **Number of NaN Values  :**')
    # Get the number of missing values in each column
    missing_values = df.isnull().sum()
    # Display the missing values in Streamlit
    st.write(missing_values)
    st.code("""
# Checking for NaN Values
missing_values = df.isnull().sum()
st.write(missing_values)
""", language='python')
    # removing strip blank spaces in front or last of a sentence
    st.markdown('### **Removing strip blank spaces in front or last of a sentence :**')
    # Specify the columns because these are alphabetical columns
    columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    # Remove spaces before and after each word in the specified columns
    for col in columns:
        df[col] = df[col].str.strip()
    # Display the DataFrame in Streamlit
    st.write(df)
    st.code("""
# Removing strip blank spaces in front or last of a sentence
columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
for col in columns:
    df[col] = df[col].str.strip()
st.write(df)
""", language='python')
    # Uppercasing all the entries
    st.markdown('### **Uppercasing Dataset :**')
    # Convert all string entries to uppercase
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.upper() if type(x) == str else x)
    # Display the DataFrame in Streamlit
    st.write(df)
    # Adding a note
    st.markdown('**Note:** This operation does not convert the column names to uppercase.')
    st.code("""
# Uppercasing all the entries
for col in df.columns:
    df[col] = df[col].apply(lambda x: x.upper() if type(x) == str else x)
st.write(df)
""", language='python')
    # getting unique categories
    st.markdown('### **Getting the unique categories for each categorical variable :**')
    # Specify the columns
    columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country','salary']
    # Get the unique categories for each categorical variable
    for column in columns:
        unique_categories = df[column].unique()
        st.write(f"{column}: {unique_categories}")
    st.code("""
# Getting unique categories
columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country','salary']
for column in columns:
    unique_categories = df[column].unique()
    st.write(f"{column}: {unique_categories}")
""", language='python')
    # Adding a note
    st.markdown('**Note:** In the ‘native-country’ column, there is an error term represented by ‘?’. This usually indicates missing or unknown data.')
    # Removing the rows that has this "?" term
    st.markdown('### **Removing the rows that has this "?" term in the native-country column :**')
    # Remove rows that contain '?' in the 'NATIVE-COUNTRY' column
    df = df.loc[df['native-country'] != '?']
    # Display the DataFrame in Streamlit
    st.write(df)
    st.code("""
# Removing the rows that has this "?" term
df = df.loc[df['native-country'] != '?']
st.write(df)
""", language='python')
    # removing column named "capital-gain" and "capital-loss"
    st.markdown('### **Removing column named "capital-gain" and "capital-loss" :**')
    # Drop the columns
    df = df.drop(columns=['capital-gain', 'capital-loss'])
    # Display the DataFrame
    st.write(df)
    st.code("""
# Removing column named "capital-gain" and "capital-loss"
df = df.drop(columns=['capital-gain', 'capital-loss'])
st.write(df)
""", language='python')
    # upper casing the column name too
    st.markdown('### **UpperCasing the names of columns**')
    # making columns to be upper case too
    df.columns = df.columns.str.upper()
    #printing the cleaned data
    st.write(df)
    st.code("""
# Uppercasing the column name too
df.columns = df.columns.str.upper()
st.write(df)
""", language='python')
    # Convert DataFrame to CSV
    csv = df.to_csv(index=False)
    csv_bytes = csv.encode()
    # Create a download button
    st.download_button(
        label="Download data as CSV",
        data=csv_bytes,
        file_name='Cleaned_data.csv',
        mime='text/csv',
    )
    #final cleaned data markdown
    st.markdown('## **########Data is Cleaned#########**')
elif page == 'Descriptive Analysis':
    # The title 
    st.title('Descriptive Analysis: Unveiling the Secrets of our Data')
    # Read the CSV data file
    df = pd.read_csv('Cleaned_data.csv')
    # using the final df we have cleaned
    st.markdown('### Basic Statistics')
    st.write(df.describe())
    # plotting bargraph
    st.markdown('### Simple Bar Graph Between Numerical and Categorical Columns')
    avg_columns = ['FNLWGT', 'EDUCATION-NUM', 'OCCUPATION-NUM', 'HOURS-PER-WEEK']
    group_columns = ['WORKCLASS', 'EDUCATION', 'MARITAL-STATUS', 'OCCUPATION', 'RELATIONSHIP', 'RACE', 'SEX', 'NATIVE-COUNTRY', 'SALARY']
    selected_group_column = st.selectbox('Select a categorical column to group by:', group_columns)
    selected_avg_column = st.selectbox('Select a numerical column to calculate average:', avg_columns)
    # Calculate average of the columns
    avg_df = df[[selected_avg_column, selected_group_column]].groupby(selected_group_column).mean()
    # Plotting
    fig, ax = plt.subplots()
    ax.bar(avg_df.index, avg_df[selected_avg_column], label=selected_avg_column)
    ax.set_xlabel(selected_group_column)
    ax.set_ylabel('Average')
    ax.set_title('Average of ' + selected_avg_column + ' by ' + selected_group_column)
    ax.legend()
    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45)
    st.pyplot(fig)
    # Plotting a tree map to check presence of categories in our data
    st.markdown('### Plotting a tree map to check presence of categories in our data')
    selected_group_column = st.selectbox('Select a categorical column:', group_columns)
    # Count the occurrences of each unique value in the selected column
    counts = df[selected_group_column].value_counts().reset_index()
    counts.columns = [selected_group_column, 'Count']
    # Create a treemap using plotly express
    fig = px.treemap(counts, 
                     path=[selected_group_column], 
                     values='Count',
                     color='Count',
                     color_continuous_scale='viridis',
                     title='Distribution of ' + selected_group_column)
    # Show the treemap
    st.plotly_chart(fig)
    #plotting asctterplot between numerical variables
    st.markdown('### Plotting Scatterplot between Numerical variables')
    num_columns = ['FNLWGT', 'EDUCATION-NUM', 'OCCUPATION-NUM', 'HOURS-PER-WEEK']
    # Select x and y variables for scatterplot
    #plot
    x_var = st.selectbox('Select x variable:', num_columns)
    y_var = st.selectbox('Select y variable:', [col for col in num_columns if col != x_var])
    # Create scatterplot
    fig = px.scatter(df, x=x_var, y=y_var, trendline="ols")
    # Display scatterplot
    st.plotly_chart(fig)
elif page == 'Prediction of Salary':
    # Your code for the 'Prediction of Salary' page goes here
    st.title('Prediction of Salary')
    st.markdown('This section will use basic stats model to predict salary per annum based on the cleaned and analyzed data.')   
    # Reading the CSV data file
    df = pd.read_csv('Cleaned_data.csv')
    # Function to create dropdown widgets for each column
    def create_dropdown_widget(column_name):
        values = ['All'] + df[column_name].unique().tolist()
        dropdown = st.selectbox(f'{column_name}:', values)
        return dropdown
    # Create interactive dropdown widgets for each column
    age_dropdown = create_dropdown_widget('AGE')
    education_dropdown = create_dropdown_widget('EDUCATION')
    workclass_dropdown = create_dropdown_widget('WORKCLASS')
    # User input for name
    name_input = st.text_input("Enter your Name: ")
    if name_input:
       st.write(f"Welcome {name_input}")
       # Filter DataFrame based on dropdown selections
       filtered_df = df.copy()
       if age_dropdown != 'All':
           filtered_df = filtered_df[filtered_df['AGE'] == age_dropdown]
       if education_dropdown != 'All':
           filtered_df = filtered_df[filtered_df['EDUCATION'] == education_dropdown]
       if workclass_dropdown != 'All':
           filtered_df = filtered_df[filtered_df['WORKCLASS'] == workclass_dropdown]
       # Check if filtered_df is empty
       if filtered_df.empty:
           st.write('No data available for the selected filters.')
       else:
           # Calculate and display average FNLWGT
           average_fnlwgt = filtered_df['FNLWGT'].mean()
           if pd.isnull(average_fnlwgt):
               st.write('The average salary could not be calculated due to missing or non-numeric data.')
           else:
               st.write(f'Your Average Salary : {average_fnlwgt:,.0f}')
          
