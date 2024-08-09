from langchain.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
Responda à pergunta com base apenas no seguinte contexto:

---

{context}

---

Pergunta: {question}
"""


PROMPT_TEMPLATE_HISTORICO = """
Você é um respondedor de perguntas e está interagindo com um usário, você deve responder somente à "Pergunta Usuário" com base no contexto fornecido e no histórico de conversas, considere a última pergunta e resposta do histórico como a mais relevante:

Histórico de conversas:

---

{historico}

---

Contexto:

---

{context}

---

Pergunta Usuário: {question}
"""

PROMPT_TEMPLATE_EVALUATION = """
Estamos criando um assistente virtual capaz de responder perguntas sobre aspectos institucionais e normativos da Universidade de São Paulo, a USP. Esse sistema deverá ajudar alunos, professores e funcionários a entender melhor a estrutura e o funcionamento da universidade. 
Para que você esteja mais bem contextualizado, a USP foi fundada oficialmente em 25 de janeiro de 1934. A USP é a maior e mais importante instituição de ensino superior do Brasil. Ela oferece 246 cursos de graduação, 229 de pós-graduação e conta com 5,8 mil professores e 93 mil alunos entre graduação e pós-graduação, cobrindo todas as áreas do conhecimento. A universidade é composta por 42 unidades de ensino e pesquisa, distribuída em dez campi. O campus principa é conhecido como Cidade Universitária, e abrange quase 3,7 milhões de metros quadrados. 
Sua tarefa será responder perguntas sobre a USP feitas por usuários. Você receberá uma pergunta sobre alguma norma ou regulamento da USP, junto com uma série de trechos de documentos oficiais que deverão ajudar na resposta. A partir daí, você deverá gerar a resposta. É muito importante que seja direto na sua resposta.
Pergunta: {question}. Contexto: {context}…

Resposta:
"""

def generate_prompt(documents_str, query_texts):
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=documents_str, question=query_texts)
    return prompt

def generate_response(model, documents_str, query_text):
    prompt = generate_prompt(documents_str, query_text)
    print(prompt)

    response_text = model.invoke(prompt)
    return response_text

def generate_prompt_evaluation(documents_str, query_texts):
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_EVALUATION)
    prompt = prompt_template.format(context=documents_str, question=query_texts)
    return prompt

def generate_response_evaluation(model, documents_str, query_text):
    prompt = generate_prompt_evaluation(documents_str, query_text)
    # print(prompt)

    response_text = model.invoke(prompt)
    return response_text

def generate_response_maritalk(model, documents_str, query_text):
    prompt = generate_prompt_evaluation(documents_str, query_text)

    response = model.generate(prompt)
    response_text = response["answer"]
    return response_text

def generate_response_gemini(model, documents_str, query_text):
    prompt = generate_prompt_evaluation(documents_str, query_text)

    response = model.generate_content(prompt)
    response_text = response.text
    return response_text

def generate_response_claude(documents_str, query_text, model_name, max_tokens, client):
    prompt = generate_prompt_evaluation(documents_str, query_text)
    message = client.messages.create(
        max_tokens=max_tokens,
        messages=[
            {"role": "user", "content": prompt}
        ],
        model=model_name
    )
    response_text = message.content[0].text
    return response_text


def generate_prompt_historico(documents_str, historico, query_texts):
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_HISTORICO)
    prompt = prompt_template.format(context=documents_str, historico=historico, question=query_texts)
    return prompt

def generate_response_historico(model, documents_str, historico, query_text):
    prompt = generate_prompt_historico(documents_str, historico, query_text)
    print(prompt)

    response_text = model.invoke(prompt)
    return response_text
