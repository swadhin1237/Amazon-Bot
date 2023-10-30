from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity


products=[
  {
    "description": "A flagship smartphone with a powerful processor and stunning display.",
    "price": 799.99
  },
  {
    "description": "4K Ultra HD Smart TV with HDR support for cinematic viewing.",
    "price": 699.99
  },
  {
    "description": "Wireless noise-canceling headphones for immersive music experience.",
    "price": 249.99
  },
  {
    "description": "High-performance gaming laptop with RGB keyboard and powerful GPU.",
    "price": 1499.99
  },
  {
    "description": "Compact digital camera with 20MP sensor and 5x optical zoom.",
    "price": 299.99
  },
  {
    "description": "Smart Wi-Fi thermostat for remote climate control and energy savings.",
    "price": 149.99
  },
  {
    "description": "Tablet with a vibrant display, ideal for work and entertainment.",
    "price": 449.99
  },
  {
    "description": "Wireless charging pad compatible with various devices.",
    "price": 29.99
  },
  {
    "description": "High-quality over-ear headphones for audiophiles.",
    "price": 199.99
  },
  {
    "description": "Ultra-thin and lightweight laptop for on-the-go productivity.",
    "price": 899.99
  },
  {
    "description": "Professional-grade DSLR camera with multiple lenses and accessories.",
    "price": 1999.99
  },
  {
    "description": "Smart home security camera with motion detection and two-way audio.",
    "price": 129.99
  },
  {
    "description": "Portable Bluetooth speaker with powerful sound and long battery life.",
    "price": 79.99
  },
  {
    "description": "High-capacity external hard drive for data storage and backup.",
    "price": 129.99
  },
  {
    "description": "Compact drone with a 4K camera for aerial photography and videography.",
    "price": 499.99
  },
  {
    "description": "Smartwatch with fitness tracking and notification features.",
    "price": 199.99
  },
  {
    "description": "Smart home hub for controlling lights, appliances, and security devices.",
    "price": 129.99
  },
  {
    "description": "Advanced graphics card for gaming and content creation.",
    "price": 399.99
  },
  {
    "description": "Multi-function laser printer for home and office use.",
    "price": 249.99
  },
  {
    "description": "Portable power bank with fast charging capabilities.",
    "price": 39.99
  },
  {
    "description": "In-ear wireless earbuds with excellent sound quality and noise isolation.",
    "price": 149.99
  },
  {
    "description": "Ultra-wide curved gaming monitor with high refresh rate.",
    "price": 499.99
  },
  {
    "description": "Robotic vacuum cleaner with smart navigation and app control.",
    "price": 349.99
  },
  {
    "description": "Home theater soundbar with immersive audio for movies and music.",
    "price": 299.99
  },
  {
    "description": "Compact digital camcorder for capturing high-quality videos.",
    "price": 349.99
  },
  {
    "description": "Dual-band Wi-Fi router for fast and reliable internet connectivity.",
    "price": 79.99
  },
    {
    "description": "Apple iPhone 13 - A13 Bionic chip, Super Retina XDR display, 128GB storage.",
    "price": 799.99
  },
  {
    "description": "Apple iPhone 13 Pro - ProMotion display, 256GB storage, powerful A15 chip.",
    "price": 999.99
  },
  {
    "description": "Apple iPhone 13 Mini - Compact design, A15 Bionic chip, 64GB storage.",
    "price": 699.99
  },
  {
    "description": "Apple MacBook Air - M2 chip, 13.3-inch Retina display, 256GB SSD.",
    "price": 999.99
  },
  {
    "description": "Apple MacBook Pro 14 - ProMotion display, M2 Pro chip, 512GB SSD.",
    "price": 1799.99
  },
  {
    "description": "Apple MacBook Pro 16 - M2 Max chip, 1TB SSD, 16-inch Retina display.",
    "price": 2399.99
  },
  {
    "description": "Samsung Galaxy S22 - A high-end smartphone with a superb camera and powerful performance.",
    "price": 899.99
  },
  {
    "description": "Samsung Galaxy Book Pro - A lightweight laptop with a vivid AMOLED display and long battery life.",
    "price": 1199.99
  },
  {
    "description": "Samsung Galaxy Z Fold 3 - A foldable smartphone that doubles as a tablet for enhanced productivity.",
    "price": 1699.99
  },
  {
    "description": "Samsung Notebook 9 Pro - An ultra-slim 2-in-1 laptop with a responsive touch screen and S Pen support.",
    "price": 1299.99
  },
  {
    "description": "Samsung Galaxy A52 - A mid-range smartphone with a versatile camera system and large display.",
    "price": 349.99
  },
  {
    "description": "Samsung Galaxy Book Flex - A 2-in-1 laptop with a QLED display and impressive battery life.",
    "price": 1099.99
  }
]


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
def find_relevant_products(brand, subcategory, product_data):
    relevant_products = []

    for product in product_data:
        desc = lemma(product["description"])
        # Check if the brand or subcategory (or their synonyms) are present in the description
        if word_or_synonyms_in_text1(brand, desc) and word_or_synonyms_in_text2(subcategory, desc):
            relevant_products.append(product)

    return relevant_products



class ActionAskCategories(Action):
    def name(self) -> Text:
        return "action_ask_categories"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message("Please mention the categories")
        return []

class ActionAskSubcategories(Action):
    def name(self) -> Text:
        return "action_ask_subcategories"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message("Also provide the subcategories")
        return []

subcategory = ""

class ActionAskProductDetails(Action):
    def name(self) -> Text:
        return "action_ask_product_details"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print("here1")
        global subcategory
        subcategory = next(tracker.get_latest_entity_values("Subcategories"))
        print(f"here2 {subcategory}")
        dispatcher.utter_message("Now mention the details like brand and price")
        return []

class ActionShowProduct(Action):
    def name(self) -> Text:
        return "action_show_product"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        brand = next(tracker.get_latest_entity_values("Brand"))
        price = next(tracker.get_latest_entity_values("Price"))
        global subcategory,products
        
        relevant_products = find_relevant_products(brand, subcategory, products)

        response = "No relevant product found"

        if(relevant_products):
            response = ""
            i=1
            for product in relevant_products:
                response += f"<br> Product {i}: <br> Brand: {brand}, <br> Price: {product['price']}, <br> Description: {product['description']} <br>"
                i+=1
        # print(response)
        dispatcher.utter_message(response)
        return []
    

class ActionFeedbackProvided(Action):
    def name(self) -> Text:
        return "action_feedbackProvided"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message("Thank you for your feedback. Bye Bye!")
        return []