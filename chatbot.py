import nltk
import io
import numpy as np
import random
import string
import warnings
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore') # libera avisos

f = open('', 'r', errors = 'ignore') # abre o arquivo
raw = f.read() # lê o arquivo
raw = raw.lower() # converte tudo para minúsculo
nltk.download('punkt') # baixa o pacote de pontuação
nltk.download('wordnet') # baixa o pacote de palavras
nltk.download('popular', quiet=True)
sent_tokens = nltk.sent_tokenize(raw) # converte em frases
word_tokens = nltk.word_tokenize(raw) # converte em palavras

sent_tokens[:2] # mostra as duas primeiras frases
word_tokens[:2] # mostra as duas primeiras palavras

lemmer = nltk.stem.WordNetLemmatizer() # seleciona o lemmatizer
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens] # lemmatiza as palavras
remove_pontuacao = dict((ord(punct), None) for punct in string.punctuation) # remove pontuação
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_pontuacao))) # lemmatiza e remove pontuação

Saudacao_input = ("e aí", "oi", "saudações", "eai", "como vai", "olá",) # entradas de saudação
Respostas_input = ["Oi", "Olá", "Oi, como vai?", "Oi, tudo bem?", "Oi, como você está?"] # respostas de saudação
def saudacao(sentence): # função de saudação
    for palavra in sentence.split(): # para cada palavra na frase
        if palavra.lower() in Saudacao_input: # se a palavra for uma saudação
            return random.choice(Respostas_input) # retorna uma resposta aleatória
        
        
def respostas(user_respostas): # função de resposta
    chatbot_respostas = '' # resposta do chatbot
    sent_tokens.append(user_respostas) # adiciona a resposta do usuário na lista de frases
    Vetorizar_palavras = TfidfVectorizer(tokenizer=LemNormalize, stop_words='portuguese') # vetoriza as palavras
    Vetor_palavars = Vetorizar_palavras.fit_transform(sent_tokens) # transforma as palavras em vetores
    similar = cosine_similarity(Vetor_palavars[-1], Vetor_palavars) # calcula a similaridade entre as frases
    indice = similar.argsort()[0][-2] # pega o índice da frase mais similar
    flat = similar.flatten() # transforma em uma lista
    flat.sort() # ordena a lista
    aprox_similar = flat[-2] # pega o valor da frase mais similar
    if(aprox_similar == 0):
        chatbot_respostas = chatbot_respostas + "Desculpe, não entendi." # se não entendeu, retorna essa frase
        return chatbot_respostas
    else:
        chatbot_respostas = chatbot_respostas + sent_tokens[indice] # se entendeu, retorna a frase mais similar
        return chatbot_respostas
    
flag = True # flag para continuar o chat
print("Chatbot: Meu nome é Chatbot. Eu responderei suas perguntas. Se você quiser sair, digite 'tchau'.") # saudação inicial
while(flag==True): # enquanto a flag for verdadeira
    user_respostas = input() # pega a resposta do usuário
    user_respostas=user_respostas.lower() # converte para minúsculo
    if(user_respostas!='tchau'): # se a resposta não for tchau
        if(user_respostas=='obrigado' or user_respostas=='obrigada'): # se a resposta for obrigado ou obrigada
            flag=False # a flag fica falso
            print("Chatbot: Você é bem-vindo.") # retorna essa frase
        else:
            if(saudacao(user_respostas)!=None): # se a resposta for uma saudação
                print("Chatbot: "+saudacao(user_respostas)) # retorna a saudação
            else:
                print("Chatbot: ",end="") # se não for uma saudação, retorna a resposta
                print(respostas(user_respostas)) # retorna a resposta
                sent_tokens.remove(user_respostas) # remove a resposta do usuário da lista de frases
    else:
        flag=False # se a resposta for tchau, a flag fica falso
        print("Chatbot: Tchau! Até mais.") # retorna essa frase