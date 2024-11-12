import os
import pandas as pd
import sys
from utils import embedding_similarity, f1_similarity, llm_similarity

# Ensure the correct number of arguments are provided
similarity = sys.argv[1]

## Create dataframe with all answers
# List all CSV files in the directory
directory = 'LLMs/'
all_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

# Read each file and add it as a column to the dataframe
dataframes = []
for file in all_files:
    df = pd.read_csv(os.path.join(directory, file), header=None)
    df.columns = [file.split('.')[0]]  # Rename the column to the file name without extension
    dataframes.append(df)

# Concatenate all dataframes column-wise
result_df = pd.concat(dataframes, axis=1)

# Get correct answers
questions = pd.read_csv('questions.csv')[['answer', 'questions', 'text']]

# Concatenate datasets
df = pd.concat([questions, result_df], axis=1)

## Calculate metrics
# Ensure all columns are strings
for col in df.columns:
    df[col] = df[col].astype(str)

# if similarity == 'f1':
#     df.drop(columns=['questions', 'text'], inplace=True)
#     df = f1_similarity(df)
# elif similarity == 'cosine':
#     df.drop(columns=['questions', 'text'], inplace=True)
#     model = 'paraphrase-multilingual-mpnet-base-v2'
#     df = embedding_similarity(df, model)
# elif similarity == 'llm':
#     df = llm_similarity(df)

df = pd.read_csv('llm_scores.csv')

# Reduce column names
df.columns = df.columns.str.replace('_similarity_score', '', regex=False)
print(df)
## Exhibit the results
print('Metrics:', similarity)
print('Full dataset')

# Calculate mean cosine similarity scores for each model
model_scores = {col: df[col].mean() * 100 for col in df}

# Print the mean scores for each model
for model, score in model_scores.items():
    print(f'{model}: {score}')

print('-----------')
print('Partial dataset')
# Filter out rows with guaranteed correct chunks
# Read each file and add it as a column to the dataframe
mpnet_files = ['mpnet_3.csv', 'mpnet_5.csv', 'mpnet_8.csv']
dataframes = []

for file in mpnet_files:
    df_mpnet = pd.read_csv(file, header=None)
    df_mpnet.columns = [file.split('.')[0]]  # Rename the column to the file name without extension
    dataframes.append(df_mpnet)
    
# Concatenate all dataframes column-wise
result_df = pd.concat([df] + dataframes, axis=1)

# List of columns to be ignored
ignore_columns = ['mpnet_3', 'mpnet_5', 'mpnet_8']

# Get correct answers
for col in result_df.columns:

    # Verifies if the column is in the ignore_columns list
    if col in ignore_columns:
        continue  # Skip to the next iteration
        
    else:
        chunk_size = col.split('_')[1]
        if chunk_size == '3':
            df_filtered = result_df[result_df['mpnet_3'] == 1]
        elif chunk_size == '5':
            df_filtered = result_df[result_df['mpnet_5'] == 1]
        elif chunk_size == '8':
            df_filtered =result_df[result_df['mpnet_8'] == 1]
        else: # No_context datasets
            df_filtered = result_df
                
        # Calculate mean cosine similarity scores for the model
        model_score = df_filtered[col].mean() * 100

        # Print the mean scores for the model
        print(f'{col}: {model_score}')
