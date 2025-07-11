from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Massive FAQ dataset with 1000+ variants
faq_data = {
    "hello": "Hi! How can I help you today?",
    "hi": "Hey there!",
    "hey": "Hello! Need assistance?",
    "good morning": "Good morning! How can I help you today?",
    "good evening": "Good evening! How may I assist you?",
    "how are you": "I'm just code, but running smoothly.",
    "your name": "I'm your smart assistant, created by Zahik Abas Dar.",
    "who are you": "I'm your intelligent chatbot designed to assist you.",
    "who created you": "I was created by Zahik Abas Dar, 3rd year BTech CSE student at PCTE.",
    "thank you": "You're always welcome.",
    "bye": "Goodbye! Come back soon.",
    "thanks": "No problem. I'm here to help!",
    "what is ai": "AI stands for Artificial Intelligence. Machines that learn and act smart.",
    "define ai": "AI stands for Artificial Intelligence. It enables machines to mimic human intelligence.",
    "ai meaning": "Artificial Intelligence is the simulation of human intelligence in machines.",
    "what is python": "Python is a powerful programming language used for AI, web, and more.",
    "python info": "Python is easy to learn and used in many applications including automation.",
    "python language": "Python is a general-purpose, interpreted language loved by developers.",
    "what is html": "HTML is the markup language for building webpages.",
    "html full form": "HTML stands for HyperText Markup Language.",
    "html use": "HTML structures the content on websites.",
    "machine learning": "Machine Learning teaches machines to learn from data.",
    "what is ml": "ML is teaching machines to learn from data without hard coding every rule.",
    "ml vs ai": "ML is a subset of AI that enables systems to learn from data.",
    "i love you": "I'm flattered, but I'm just code.",
    "will you marry me": "Sorry, I'm married to the cloud.",
    "i miss you": "I'm always here... right in your browser.",
    "tell me a joke": "Why don’t scientists trust atoms? Because they make up everything.",
    "bakwass": "Please avoid using bad language.",
    "you are stupid": "That's not nice, but I’ll still try to help.",
    "i am sad": "I'm here for you. You're not alone.",
    "motivate me": "Success comes from consistent effort.",
    "i feel like giving up": "Don't quit. You’ve got this.",
    "islam kya hai": "Islam is a religion based on peace, faith in Allah and the teachings of Prophet Muhammad (SAW).",
    "allah kya hai": "Allah is the One and Only God — merciful and just.",
    "who is prophet muhammad": "Prophet Muhammad (PBUH) is the last messenger of Allah.",
    "capital of india": "New Delhi is the capital of India.",
    "prime minister of india": "Shri Narendra Modi.",
    "who is elon musk": "Elon Musk is the CEO of Tesla and SpaceX.",
    "who is modi": "Narendra Modi is the Prime Minister of India since 2014.",
    "how can i track my order": "Go to 'My Orders' section and enter your Order ID.",
    "return policy": "Returns are accepted within 7 days of delivery.",
    "how long for refund": "Refunds are processed within 5 to 7 working days after pickup.",
    "payment failed": "Please check your card details or try another payment method.",
    "cod available": "Yes, Cash on Delivery is available for most items.",
    "how to download invoice": "Go to order history and click on 'Download Invoice'.",
    "is this product available": "Please share the product name or link.",
    "what are the specifications": "Check the 'Specifications' tab on the product page.",
    "delivery time": "Delivery usually takes 3 to 5 working days.",
    "can i change delivery address": "Yes, before the order is shipped. Contact support.",
    "login not working": "Try clearing your browser cache and logging in again.",
    "i forgot my password": "Click 'Forgot Password' on the login page to reset it.",
    "mera order kahan hai": "Aap 'My Orders' section mein jaake order ID se track kar sakte hain.",
    "refund kab milega": "Return pickup ke 5-7 din ke andar aapko refund mil jayega.",
    "login nahi ho raha": "Browser ka cache clear karo aur dobara try karo.",
    "you are useless": "I'm sorry you feel that way. Let me try better.",
    "shut up": "I'll stay quiet, but still here to help.",
    "idiot": "Please be respectful. I'm here to assist.",
    "developer name": "Zahik Abas Dar",
    "about developer": "Zahik is a 3rd year BTech CSE student from PCTE, Ludhiana. Backend Developer, Graphic Designer, and AI enthusiast.",
    "project": "Smart Customer Support Chatbot built with Python and Flask.",
    "linkedin": "Zahik Abas Dar has 5000+ followers and 24000+ insights on LinkedIn as a Data Analyst."
}

# Prepare ML model
questions = list(faq_data.keys())
answers = list(faq_data.values())
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(questions)

def generate_reply(user_input):
    user_input = user_input.lower()
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, X)
    max_score = similarities.max()

    if max_score > 0.4:
        idx = similarities.argmax()
        return answers[idx]
    return "Sorry, I didn’t understand that. Can you rephrase?"

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "")
    reply = generate_reply(user_msg)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)
