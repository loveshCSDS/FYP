import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the model and tokenizer
model_path = "distilbert_fakereview_model.pt"
tokenizer_path = "tokenizer"

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)


def predict_review(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction

# Streamlit app
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Review", "Seller"])

if page == "Review":
    st.title("Enter a Product Review")
    user_input = st.text_area("Enter review text:")
    if st.button("Predict"):
        if user_input:
            prediction = predict_review(user_input)
            label = "Genuine" if prediction == 0 else "Fake"
            st.write(f"The review is **{label}**.")
        else:
            st.write("Please enter a review text.")

elif page == "Seller":
    st.title("Seller Authenticity Checker")

    if "seller_name" not in st.session_state:
        st.session_state.seller_name = ""
    if "reviews" not in st.session_state:
        st.session_state.reviews = [""]
    
    st.write("Enter Seller Name")
    seller_name = st.text_input("Seller Name", st.session_state.seller_name)
    st.session_state.seller_name = seller_name
    
    st.write("Enter Product Reviews")
    
    for i in range(len(st.session_state.reviews)):
        st.session_state.reviews[i] = st.text_area(f"Review {i + 1}", st.session_state.reviews[i], key=f"review_{i}")
    
    add_review_clicked = st.button("Add another review")
    if add_review_clicked:
        st.session_state.reviews.append("")
        st.experimental_rerun()

    if st.button("Predict Reviews"):
        if all(st.session_state.reviews):
            predictions = []
            fake_count = 0
            genuine_count = 0
            
            for review in st.session_state.reviews:
                prediction = predict_review(review)
                label = "Genuine" if prediction == 0 else "Fake"
                predictions.append((review, label))
                if label == "Fake":
                    fake_count += 1
                else:
                    genuine_count += 1
            
            for i, (review, label) in enumerate(predictions):
                st.write(f"Review {i + 1}: **{label}**.")
            
            if fake_count >= genuine_count:
                st.write(f"The seller **{seller_name}** is potentially a **Fraud** seller.")
            else:
                st.write(f"The seller **{seller_name}** is potentially a **Genuine** seller.")
            
            # Explanation of how the prediction works
            st.write("""
                ### How the Prediction Works
                The model analyzes each review to determine if it is genuine or fake. This is done using a pre-trained DistilBERT model fine-tuned for classification tasks. 
                Each review is tokenized and passed through the model to generate a prediction score. If the majority of the reviews are predicted as fake, the seller is flagged as potentially fraudulent. Otherwise, the seller is considered genuine.
            """)
        else:
            st.write("Please enter all reviews.")
    
    if st.button("Clear All"):
        st.session_state.seller_name = ""
        st.session_state.reviews = [""]
        st.experimental_rerun()
