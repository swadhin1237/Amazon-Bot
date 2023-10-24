# Amazon Bot Chatbot

![Amazon Bot](/amazon%20bot/images/Screenshot%20from%202023-10-25%2002-49-25.png)

This is a chatbot project for a fictitious Amazon-like e-commerce platform created using Rasa. The chatbot allows users to interact with the e-commerce platform, search for products, and get product recommendations based on brand, subcategory, and price. The chatbot also allows users to add products to their cart and checkout. The chatbot is trained using Rasa's open source natural language understanding (NLU) and dialogue management tools.

## Demo

[![Alt text](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)

## Prerequisites

Before running the chatbot, make sure you have the following installed:

- Rasa (Follow [Rasa installation instructions](https://rasa.com/docs/rasa/installation) if you haven't already)

## Getting Started

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/swadhin1237/Amazon-Bot.git
   cd Amazon-Bot
   pip install -r requirements.txt
   cd amazon_bot
   ```

## Running the Chatbot

To interact with the chatbot, follow these steps:

1. Open two separate terminal windows:

2. In one terminal, start the Rasa shell using the following command:
   rasa shell

3. In the other terminal, start the Rasa action server using the following command:
   rasa run actions

4. Start chatting with the bot in the Rasa shell terminal!

## Training the Chatbot

After making changes to your Rasa project, such as modifying NLU data, stories, or actions, remember to train your model with the following command:
rasa train

## Testing the Chatbot

Check the description in running the chatbot for testing the chatbot and open three separate terminal windows:
   rasa run
   rasa run actions
   flask run




## Understanding the Code

The code for the Amazon Bot Chatbot project is organized into several Python scripts and custom actions for Rasa. Below is a comprehensive overview of the core components and functionalities:

### Custom Actions

The code includes custom actions implemented as Python classes to extend the chatbot's capabilities. These custom actions are designed to perform specific tasks, such as soliciting information related to categories, subcategories, product details, and presenting product information to the user.

### NLP and Semantic Analysis

The chatbot harnesses the power of Natural Language Processing (NLP) techniques for effective communication and comprehension of user queries. Key NLP methods used include:

- **Tokenization**: Breaking down user input into individual tokens or words, allowing for easier analysis.
- **Lemmatization**: The process of reducing words to their base or dictionary forms. This helps in understanding the core meaning of words in different forms.
- **Similarity Analysis**: Employing pre-trained models like BERT and spaCy for semantic similarity analysis, which is crucial for understanding the context and relevance of user queries. These techniques aid in searching for relevant products based on user input.

### Product Data

The chatbot operates based on a predefined list of products, each with a description and price. These product entries serve as the foundation for the chatbot's ability to provide users with information about products available on the e-commerce platform. The product data includes essential details that are crucial for search and recommendations.

### User Interaction

The chatbot is designed to engage in dynamic and intuitive conversations with users. It actively interacts with users by requesting input related to categories, subcategories, brand preferences, and price ranges. These user interactions facilitate the chatbot's ability to narrow down product searches and present relevant product information to the user in a user-friendly and informative manner.

### Customizing the Chatbot

Your chatbot project is fully customizable to cater to your specific requirements. You can tailor the chatbot by making the following adjustments:

- **Product Data**: You can modify the predefined product data to reflect the inventory of your specific e-commerce platform. Adding, removing, or updating products will ensure that the chatbot provides accurate and up-to-date information to users.

- **Custom Actions**: Extend the functionality of the chatbot by adding new custom actions to address unique use cases or functionalities specific to your e-commerce platform. Custom actions provide flexibility and adaptability.

- **NLP Enhancements**: Enhance the chatbot's NLP capabilities by fine-tuning the language models, integrating additional language resources, or adjusting similarity thresholds. This enables the chatbot to better understand user queries and provide more precise responses.

## Future Additions

[Download User Guide (DOCX)](https://docs.google.com/document/d/1jwPwr4RaICmUSm4qnAsGU8ng5kesE7PksEyw6AxrJ9s/edit?usp=sharing)

We have exciting plans to enhance this README and the Amazon Bot Chatbot project in the near future. Here are some of the upcoming additions:

1. **User Guide**: We are working on a comprehensive user guide in DOCX format that will provide detailed instructions on setting up, customizing, and using the Amazon Bot Chatbot.

2. **Additional Features**: We will be adding new features to the chatbot, such as improved product search capabilities, multilingual support, and user account management.

3. **Performance Enhancements**: Our development team is continuously optimizing the chatbot's performance to make it faster and more responsive.

Stay tuned for these exciting updates! We are committed to making the Amazon Bot Chatbot even more powerful and user-friendly in the coming months.
