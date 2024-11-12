import string
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

## Calculate f1-score
def f1_similarity(df):
    for col in df:
        df[col + '_similarity_score'] = df.apply(lambda row: f1_score(row[col], row['answer']) if isinstance(row[col], str) and row[col].strip() != '' else 0, axis=1)

    # Remove f1-score for the correct answer
    df.drop(columns=['answer_similarity_score'], inplace=True)
    
    # Keep only columns with _similarity_score in their name
    df = df[[col for col in df.columns if '_similarity_score' in col]]

    return df

# Define the normalize_answer function
def normalize_answer(s):
    """Lower text and remove punctuation and extra whitespace."""
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

# Define the f1_score function
def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

## Calculate embedding similarity
def embedding_similarity(df, model):
    model = SentenceTransformer(model)

    # Create embeddings for the correct answers
    df['correct_embedding'] = df['answer'].apply(lambda x: model.encode(x))

    # Drop the answer column as it's no longer needed
    df.drop(columns=['answer'], inplace=True)

    # Calculate similarity score for each model and create a new column for each model
    for col in df.columns:
        df[col + '_similarity_score'] = df.apply(lambda row: calculate_similarity_score(model, row['correct_embedding'], row[col]), axis=1)

    # Drop the embeddings column as it's no longer needed
    df.drop(columns=['correct_embedding', 'correct_embedding_similarity_score'], inplace=True)

    # Keep only columns with _similarity_score in their name
    df = df[[col for col in df.columns if '_similarity_score' in col]]

    return df

# Function to calculate similarity score
def calculate_similarity_score(model, correct_embedding, predicted_answer):
    if isinstance(predicted_answer, str) and predicted_answer.strip() != '':
        predicted_embedding = model.encode(predicted_answer)
        cosine_sim = cosine_similarity([correct_embedding], [predicted_embedding])[0][0]
        return (cosine_sim + 1) / 2
    else:
        return 0

## Calculate GPT4 evaluation score
# Create a client
client = OpenAI(api_key='')
model = "gpt-4-0125-preview"
temperature = 0

def evaluate_with_gpt4(question, supporting_text, answer, predicted):
    prompt = f'''Estamos realizando uma avaliação de um sistema de resposta a perguntas. Sua tarefa é avaliar a qualidade das respostas geradas por um modelo de linguagem. Por favor, reveja a pergunta, o texto de apoio, a resposta correta e a resposta prevista fornecida abaixo.

Pergunta: {question}
Texto de apoio: {supporting_text}
Resposta correta: {answer}
Resposta prevista: {predicted}

Com base nas informações fornecidas, avalie a resposta prevista de acordo com a seguinte escala:

1. Totalmente correta: A resposta prevista corresponde totalmente à resposta correta e aborda corretamente a pergunta com base no texto de apoio.
2. Maioritariamente correta: A resposta prevista é majoritariamente precisa, mas pode ter pequenos erros ou omissões, ainda assim aborda em grande parte a pergunta com base no texto de apoio.
3. Incorreta: A resposta prevista está errada ou não aborda adequadamente a pergunta com base no texto de apoio.

Por favor, retorne o número correspondente à sua avaliação (1, 2 ou 3).
'''
    try:
        similarity_score = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": prompt}]
        ).choices[0].message.content

        # Extract the similarity score from the response
        print('Resposta:', similarity_score)

        # Convert the score to a float and normalize it to the range (0, 1)
        if similarity_score == '1':
            return 1.0
        elif similarity_score == '2':
            return 0.5
        elif similarity_score == '3':
            return 0.0
        else:
            return 0.0
    except Exception as e:
        print(f"Error: {e}")
        return 0.0
    
# Evaluate each model's answer with GPT-4
def llm_similarity(df):
    # List of model columns, excluding specific models
    exclude_models = ['questions', 'answer', 'text']
    model_columns = [col for col in df.columns if col not in exclude_models]

    for col in model_columns:
        score_column = col + '_similarity_score'
        if score_column not in df.columns:
            df[score_column] = 0.0
        for i in range(len(df)):
            if df.at[i, score_column] == 0.0:
                print(f"Evaluating: Column '{col}', Row {i}")
                df.at[i, score_column] = evaluate_with_gpt4(df.at[i, 'questions'], df.at[i, 'text'], df.at[i, 'answer'], df.at[i, col])
                # Save the intermediate results after each evaluation
                df.to_csv('llm_scores.csv', index=False)

    df.drop(columns=['answer', 'questions', 'text'], inplace=True)

    return df