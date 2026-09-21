import streamlit as st
import pickle

model = pickle.load(open('spam.pkl','rb'))
cv = pickle.load(open('vectorizer.pkl','rb'))


st.title("SMS spam Detection Model")
st.write("This is a Machine Learning application to detect SMS as spam or not spam.")
user_input = st.text_area("Enter an SMS to predict it's spam or not spam", height=150)

if user_input:
    st.caption(f"{len(user_input)} characters, {len(user_input.split())} words")

if st.button("Predict"):
    if user_input:
        data = [user_input]
        vectorized_data = cv.transform(data).toarray()
        result = model.predict(vectorized_data)
        if result[0]==0:
            st.write("The SMS is not spam")
        else:
            st.write("The SMS is spam")  
    else:
        st.write("Please type SMS to predict it's spam or not spam")          
