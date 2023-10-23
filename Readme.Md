# Amazon Bot Chatbot

This is a simple chatbot project for an Amazon-like e-commerce platform created using Rasa.

## Prerequisites

Before running the chatbot, make sure you have the following installed:

- Rasa (Follow [Rasa installation instructions](https://rasa.com/docs/rasa/installation) if you haven't already)

## Getting Started

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/amazon_bot.git
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

Check the description in running the chatbot for testing the chatbot
