import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Setup SSL and NLTK data path
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
intents_file_path = os.path.abspath("intents.json")
with open(intents_file_path, "r") as file:
    intents_data = json.load(file)

# Initialize vectorizer and classifier
tfidf_vectorizer = TfidfVectorizer()
logistic_regression_clf = LogisticRegression(random_state=0, max_iter=10000)

# Prepare training data
intent_tags = []
training_patterns = []
for intent in intents_data:
    for pattern in intent['patterns']:
        intent_tags.append(intent['tag'])
        training_patterns.append(pattern)

# Train the model
X_train = tfidf_vectorizer.fit_transform(training_patterns)
y_train = intent_tags
logistic_regression_clf.fit(X_train, y_train)

def get_chatbot_response(user_input):
    transformed_input = tfidf_vectorizer.transform([user_input])
    predicted_tag = logistic_regression_clf.predict(transformed_input)[0]
    for intent in intents_data:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])
    return "Sorry, I don't understand that."

def main():
    st.title("Chatbot with NLP")

    # Sidebar menu
    menu_options = ["Home", "Conversation History", "About"]
    selected_option = st.sidebar.selectbox("Menu", menu_options)

    if selected_option == "Home":
        st.write("Welcome! Type a message and press Enter to chat with the bot.")

        # Ensure chat log file exists
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        user_message = st.text_input("You:")

        if user_message:
            bot_response = get_chatbot_response(user_message)
            st.text_area("Chatbot:", value=bot_response, height=120, max_chars=None)

            # Log the conversation
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_message, bot_response, timestamp])

            if bot_response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting! Have a great day!")
                st.stop()

    elif selected_option == "Conversation History":
        st.header("Conversation History")
        try:
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        except FileNotFoundError:
            st.write("No conversation history found.")

    elif selected_option == "About":
        st.write("This chatbot uses NLP and Logistic Regression to understand and respond to user inputs. Built with Streamlit for the interface.")

        st.subheader("Project Overview:")
        st.write("""
        This project involves:
        1. Training a chatbot using NLP techniques and Logistic Regression.
        2. Building an interactive web interface using Streamlit.
        """)

        st.subheader("Dataset:")
        st.write("""
        The dataset consists of labeled intents and entities:
        - Intents: The purpose of the user input (e.g., "greeting", "budget").
        - Entities: Extracted from user input (e.g., "Hi", "How do I create a budget?").
        """)

        st.subheader("Streamlit Interface:")
        st.write("The interface allows users to input text and receive responses from the chatbot.")

        st.subheader("Conclusion:")
        st.write("This project demonstrates a chatbot that understands and responds to user inputs using NLP and Logistic Regression. Future improvements could include more data and advanced NLP techniques.")

if __name__ == '__main__':
    main()
