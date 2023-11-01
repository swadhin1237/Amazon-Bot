from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from .dataset import laptop_products,phone_products
from rasa_sdk.executor import CollectingDispatcher
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity



# Load the English language model with lemmatization support
nlp = spacy.load("en_core_web_sm")

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def similarity(word1,word2):
  # Tokenize and encode the words
  word1_tokens = tokenizer(word1, return_tensors="pt")
  word2_tokens = tokenizer(word2, return_tensors="pt")

  # Get word embeddings from BERT
  with torch.no_grad():
      word1_output = model(**word1_tokens).last_hidden_state.mean(dim=1).numpy()
      word2_output = model(**word2_tokens).last_hidden_state.mean(dim=1).numpy()

  # Calculate cosine similarity between word embeddings
  return cosine_similarity(word1_output, word2_output)[0][0]

def lemma(sentence):
  # Process the sentence with spaCy
  doc = nlp(sentence)

  # Lemmatize and filter out stop words, punctuation, and non-alphabetic tokens
  lemmatized_words = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in STOP_WORDS]

  # Join the lemmatized words to create a clean sentence
  clean_sentence = " ".join(lemmatized_words)
  return clean_sentence

# Function to check if a word or its synonyms are present in a text
def word_or_synonyms_in_text1(word, text):
    word = word
    for token in nlp(text):
        if(similarity(word,token.text)>=0.95):
            return True
    return False

def word_or_synonyms_in_text2(word, text):
    word = word
    for token in nlp(text):
        if(similarity(word,token.text)>=0.85):
            return True
    return False

# Function to find relevant products based on brand and subcategory
def find_relevant_products(brand,product_data):
    relevant_products = []
    for product in product_data:
        desc = lemma(product["product_name"])
        # Check if the brand or subcategory (or their synonyms) are present in the description
        if word_or_synonyms_in_text1(brand, desc):
            relevant_products.append(product)
    return relevant_products

class ActionLaptop(Action):
    def name(self) -> Text:
        return "action_laptop"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        brand=next(tracker.get_latest_entity_values("brand"),None)
        storage=next(tracker.get_latest_entity_values("storage"),None)
        price=next(tracker.get_latest_entity_values("price"),None)
        RAM=next(tracker.get_latest_entity_values("RAM"),None)
        GPU=next(tracker.get_latest_entity_values("GPU"),None)
        processor=next(tracker.get_latest_entity_values("processor"),None)
        products=laptop_products
        if(brand==None):
            relevant_products=products
        else:
            relevant_products = find_relevant_products(brand, products)
        response = "No relevant product found"
        if(relevant_products):
            response = ""
            i=1
            for product in relevant_products:
                response += f"<br> Product {i}: <br> Brand: {brand}, <br> Price: {product['price']}, <br> Description: {product['description']} <br> "
                i+=1
        dispatcher.utter_message(response)
        return []

class ActionPhone(Action):
    def name(self) -> Text:
        return "action_phone"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        brand=next(tracker.get_latest_entity_values("brand"),None)
        storage=next(tracker.get_latest_entity_values("storage"),None)
        Price=next(tracker.get_latest_entity_values("Price"),None)
        products=phone_products
        if(brand==None):
            relevant_products=products
        else:
            relevant_products = find_relevant_products(brand, products)
        response = "No relevant product found"
        if(relevant_products):
            response = ""
            i=1
            for product in relevant_products:
                response += f"<br> Product {i}: <br> Brand: {brand}, <br> Price: {product['price']}, <br> Description: {product['description']} <br> "
                i+=1
        # str1=f'Brand:{brand},Storage:{storage},Price:{Price}'
        dispatcher.utter_message(response)
        return []
