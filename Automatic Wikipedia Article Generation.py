# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 16:38:19 2022

@author: Vansh
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize 
from bs4 import BeautifulSoup
from bs4.element import Comment
import requests
import urllib.request
from googlesearch import search
from bs4 import BeautifulSoup
from bs4.element import Comment
import requests
import urllib.request
import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import wiki_Sum
from wiki_Sum import wiki_Sum
import web_scraper
from web_scraper import extract_text_from_single_web_page
#from lsa_summarizer import LsaSummarizer
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()


stopWords = set(stopwords.words("english"))
#https://www.britannica.com/technology/
#https://en.wikipedia.org/wiki/


def getStandard(query):
    
    '''
    returns ground-truth text from article to compare web-scraped data to
    param str query: topic of interest
    '''
    
    page = requests.get("https://en.wikipedia.org/wiki/" + query)
    txt = text_from_html(page.content)
    return txt

def getEmbeddingScores(one, two):
    
    '''
    returns cosine similarity between two parameters
    param str one: first string 
    param str two: second string    
    '''
    
    embeddings1 = model.encode(one, convert_to_tensor=True)
    embeddings2 = model.encode(two, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)    
    return cosine_scores


def getWebsites(query):
    '''
    returns set of websites after google searching query
    param str query: what is searched on google 
    '''
    websites = set()
    for j in search(query, tld="co.in", num=20, stop=20, pause=2):
        if "https://en.wikipedia.org/" not in j:    
            websites.add(j)
    return websites

def getWebsiteInfo(website):
    '''
    returns str txt: visible text from website
    param str website: url of website 
    '''
    page  = requests.get(website)
    txt = text_from_html(page.content)
    return txt

def getWebsiteTitle(website):
    '''
    returns str txt: title of website
    param str website: url of website
    '''
    page = requests.get(website)
    txt = title_from_html(page.content)
    return txt

def title_from_html(tags):
    '''
    returns str text: returns text from HTML title tag
    param str tags: collection of HMTL tags 
    '''
    soup = BeautifulSoup(tags, 'html.parser')
    text = ""
    for title in soup.find_all('title'):
        text += title.get_text()
    return text

def tag_visible(element):
    '''
    returns bool if HTML tag represents readable text
    param str element: HTML tag from website 
    '''
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', 
                               '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    '''
    returns formatted visible text given HMTL body metadata 
    param str body: HTML metadata representing website's main information
    '''
    soup = BeautifulSoup(body, 'html.parser')
    page_body = soup.body
    texts = page_body.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)

#returns dictionary holding frequency of nouns 
def buildFreqTable(txt):
    '''
    returns dictionary holding frequency of nouns present in string
    param str txt: arbitrary string
    '''
    words = word_tokenize(txt)
    freqTable = dict()
    for word in words: 
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    return freqTable

def mostCommonNouns(txt):
    '''
    returns str mostCommon: the most common nouns found in a string using freq 
    tables
    param str txt: arbitrary string 
    '''
    mostCommon = ""
    words = word_tokenize(txt)
    freqTable = buildFreqTable(txt)
    fillers = ['.', '&', '#', ',', ';', '[', ']', '?', ')', '(', '\'', '\"', 
               ':']
    max_count = 0
    for word in words:
        if word in stopWords:
            continue
        
        if word in freqTable:
            if freqTable[word] > max_count and word not in fillers:
                mostCommon = word
                max_count = freqTable[word]
    return mostCommon

def extractiveSummary(txt):
    '''
    returns summary of given text built by evaluating frequency of words and 
    then ranking sentences based on largest frequency score 
    param str txt: arbitrary passage
    '''
    sentences = sent_tokenize(txt)
    sentenceValue = dict()
    freqTable = buildFreqTable(txt)
    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq         
    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]
        average= int(sumValues / len(sentenceValue))
    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > 
                                            (1.2 * average)):
            summary += " " + sentence
    return summary


def abstractiveSummary(text):
    '''
    returns summary of given text built by using Google's transformer based
    model (T5)
    param str text: arbitrary passage
    '''
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    device = torch.device('cpu')
    preprocess_text = text.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text
    # print ("original text preprocessed: \n", preprocess_text)
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
    # summmarize 
    summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=100,
                                        max_length=500,
                                        early_stopping=True)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output
    # print ("\n\nSummarized text: \n",output)


def getEntities(txt):
    '''
    returns array of pairs of entites using spacy library named entity extraction
    param str txt: arbitrary passage
    '''
    doc = nlp(txt)
    return([(X.text, X.label_) for X in doc.ents])


def getMostCommonEntity(entities):
    '''
    returns most frequent entity in array of entities provided by getEntities method
    using frequency table 
    param array entities: array of pairs gotten from named entity extraction 
    (getEntities method)
    '''
    freq = dict()
    maxCount = 0
    global maxVal
    for entity in entities:
        if entity[1] == "CARDINAL" or entity[1] == "DATE" or entity[1] == "ORDINAL" or entity[1] == "LOC":
            continue
        if entity in freq:
            freq[entity] += 1
        else:
            freq[entity] = 1
    for x in freq:
        if freq[x] > maxCount:
            maxCount = freq[x]
            maxVal = x
    return maxVal


'''
Sample use case for program below: takes top K most relevant passages from webscraped
data and applies extractive summarization followed by entity extraction to 
produce Wikipedia-like article

Can be modified to get larger articles but be aware of repititon. To solve this, 
consider combining passages with the same or similar entity headers and apply 
abtractive summarization to avoid redundancy. Many abstractive models are capable of 
doing this. 

Consider expanding on this porject by first extracting relvant entities, making sure there
are no repeats, and then webscraping information for each entity to produce a 
comprehensive article
'''

def createArticle(query):
    output = ""
    websites = getWebsites(query)
    print(websites)
    data = {}
    sim = dict()
    sim_list = []
    for url in websites:
        resp = requests.get(url)
        
        # 2. If the response content is 200 - Status Ok, Save The HTML Content:
        if resp.status_code == 200:
            data[url] = resp.text
        text_content = extract_text_from_single_web_page(url)
        toRet = wiki_Sum(text_content)
        standard = getStandard(query)
        similarity = getEmbeddingScores(toRet, standard)
        #if (similarity > 0.3 ):
        #print(similarity)
        sim[similarity] = toRet
        sim_list.append(similarity)
        #mostCommonEnt = getMostCommonEntity(getEntities(toRet))
        #print(mostCommonEnt)
        #print(toRet)
        #print('\n')
        
    sim_list.sort(reverse=True)
    
    sim_list = sim_list[:3]
    
    print(sim_list)
    output += '\n'
    
    #print('\n')
    #print(getStandard(query))
    #print(getEntities(getStandard(query)))
    
    for entry in sim_list:
        output += '\n'
        #print('\n')
        text = sim[entry]
        title = getMostCommonEntity(getEntities(text))
        output += title[0]
        output += '\n'
        output += text
        output += '\n'
        #print(title[0])
        #print('\n')
        #print(text)
        #print('\n')
    return output

query = "Query Language"
print(createArticle(query))
