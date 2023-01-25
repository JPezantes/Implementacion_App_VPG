import tweepy as tw
import streamlit as st
import pandas as pd
import torch
import numpy as np
import re
import datetime

from pysentimiento.preprocessing import preprocess_tweet

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AdamW

tokenizer = AutoTokenizer.from_pretrained('JosePezantes/finetuned-robertuito-base-cased-V-P-G')
model = AutoModelForSequenceClassification.from_pretrained("JosePezantes/finetuned-robertuito-base-cased-V-P-G")

import torch
if torch.cuda.is_available():  
    device = torch.device("cuda")
    print('I will use the GPU:', torch.cuda.get_device_name(0))
    
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

consumer_key = st.secrets["consumer_key"]
consumer_secret = st.secrets["consumer_secret"]
access_token = st.secrets["access_token"]
access_token_secret = st.secrets["access_token_secret"]
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

def preprocess(text):
    text=text.lower()
    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    text = re.sub(r'http?:\/\/.*[\r\n]*', '', text)
    #Replace &amp, &lt, &gt with &,<,> respectively
    text=text.replace(r'&amp;?',r'and')
    text=text.replace(r'&lt;',r'<')
    text=text.replace(r'&gt;',r'>')
    #remove hashtag sign
    text=re.sub(r"#","",text)   
    #remove mentions
    text = re.sub(r"(?:\@)\w+", '', text)
    #remove non ascii chars
    text=text.encode("ascii",errors="ignore").decode()
    #remove some puncts (except . ! ?)
    text=re.sub(r'[:"#$%&\*+,-/:;<=>@\\^_`{|}~]+','',text)
    text=re.sub(r'[!]+','!',text)
    text=re.sub(r'[?]+','?',text)
    text=re.sub(r'[.]+','.',text)
    text=re.sub(r"'","",text)
    text=re.sub(r"\(","",text)
    text=re.sub(r"\)","",text)
    text=" ".join(text.split())
    return text
    
def highlight_survived(s):
    return ['background-color: red']*len(s) if (s.violencia_política_de_género == 1) else ['background-color: green']*len(s)

def color_survived(val):
    color = 'red' if val=='violencia política de género' else 'white'
    return f'background-color: {color}'                

st.set_page_config(layout="wide")
st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)

#background-color: Blue;

colT1,colT2 = st.columns([2,8])
with colT2:
    #st.title('Analisis de contenido de violencia política de género en Twitter') 
    st.markdown(""" <style> .font {
    font-size:40px ; font-family: 'Cooper Black'; color: #F15A28;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Violencia política de género en Twitter</p>', unsafe_allow_html=True)
    
    st.markdown(""" <style> .font1 {
    font-size:28px ; font-family: 'Times New Roman'; color: #07B6F5;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font1">Modelo de lenguaje utilizando RoBERTuito, para identificar tweets con contenido de violencia política de género </p>', unsafe_allow_html=True)
    
with colT1:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSP09HkQ52tAuccb8iFEWs9E4ag0xRVjDSYXHNHSdSIuzERFPxPZ6NQZYnd_WXB2j-kkoQ&usqp=CAU",width=200)

st.markdown(""" <style> .font2 {
    font-size:16px ; font-family: 'Times New Roman'; color: #181618;} 
    </style> """, unsafe_allow_html=True)
st.markdown('<p class="font2">La presente herramienta utiliza tweepy para descargar tweets de twitter en base a la información de entrada y procesa los tweets usando el modelo de lenguaje entrenado para identificar tweets que representan violencia política de género. Los tweets recolectados y su correspondiente clasificación se almacenan en un dataframe que se muestra como resultado final.</p>',unsafe_allow_html=True)

with open("style.css") as f: 
    st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
    
def run():
    df =  pd.DataFrame()
    showTable = False    
    col,col1,col2 = st.columns([2,3,2])
      
    with col1:
        myform = st.form(key='Introduzca Texto')
        search_words = myform.text_input("Introduzca el término o usuario para analizar y pulse el check correspondiente")
        number_of_tweets = myform.number_input('Introduzca número de tweets a analizar. Máximo 50', 0,50,10)
        filtro=myform.radio("Seleccione la opcion para filtrar por término o usuario",('Término', 'Usuario'))

                               
        submit_button = myform.form_submit_button(label='Analizar')
        
        if submit_button:
             
            if (filtro=='Término'):
                new_search = search_words + " -filter:retweets"
                tweets =tw.Cursor(api.search_tweets,q=new_search,lang="es",tweet_mode="extended").items(number_of_tweets)

            elif (filtro=='Usuario'):
                try:
                    if not search_words.startswith('@'):
                        st.error("Por favor, ingrese un usuario válido, iniciando con @")
                        return
                    tweets = api.user_timeline(screen_name = search_words,tweet_mode="extended",count=number_of_tweets)
                except tw.errors.NotFound:
                    st.error('"El usuario ingresado no existe. Por favor, ingrese un usuario existente" ⚠️', icon="⚠️")
                    return
            
            tweet_list = [i.full_text for i in tweets]
            
            text= pd.DataFrame(tweet_list)
            #text[0] = text[0].apply(preprocess)
            text[0] = text[0].apply(preprocess_tweet)
            text1=text[0].values
            indices1=tokenizer.batch_encode_plus(text1.tolist(),
                                     max_length=128,
                                     add_special_tokens=True, 
                                     return_attention_mask=True,
                                     pad_to_max_length=True,
                                     truncation=True)
            input_ids1=indices1["input_ids"]
            attention_masks1=indices1["attention_mask"]
            prediction_inputs1= torch.tensor(input_ids1)
            prediction_masks1 = torch.tensor(attention_masks1)
            # Set the batch size.  
            batch_size = 25
            # Create the DataLoader.
            prediction_data1 = TensorDataset(prediction_inputs1, prediction_masks1)
            prediction_sampler1 = SequentialSampler(prediction_data1)
            prediction_dataloader1 = DataLoader(prediction_data1, sampler=prediction_sampler1, batch_size=batch_size)
            print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs1)))
            # Put model in evaluation mode
            model.eval()
            # Tracking variables 
            predictions = []
            # Predict 
            for batch in prediction_dataloader1:
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids1, b_input_mask1 = batch
                # Telling the model not to compute or store gradients, saving memory and   # speeding up prediction
                with torch.no_grad():
                    # Forward pass, calculate logit predictions
                    outputs1 = model(b_input_ids1, token_type_ids=None,attention_mask=b_input_mask1)
                logits1 = outputs1[0]
                # Move logits and labels to CPU
                logits1 = logits1.detach().cpu().numpy()
                # Store predictions and true labels
                predictions.append(logits1)
            flat_predictions = [item for sublist in predictions for item in sublist]
            flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
            df = pd.DataFrame(list(zip(tweet_list, flat_predictions)),columns =['Últimos '+ str(number_of_tweets)+' Tweets'+' de '+search_words, 'violencia política de género'])
            df['violencia política de género']= np.where(df['violencia política de género']== 0, 'no violencia política de género', 'violencia política de género')
            showTable = True
                
    if (showTable):            
        df.index+=1
        print(df.index)
        st.table(df.head(50).style.set_properties(subset=['violencia política de género'], **{'width': '250px'}).applymap(color_survived, subset=['violencia política de género']))
                           
try:
    run()
except KeyError:
    cole,cole1,cole2 = st.columns([3,3,2])
      
    with cole1:
        st.error('Término no encontrado. Por favor, ingrese un término existente ⚠️', icon="⚠️")