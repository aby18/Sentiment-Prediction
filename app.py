import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd

st.set_page_config(page_title='Sentiment Prediction Model', layout='wide')

@st.cache_resource
def load_components():
    model = joblib.load('logistic_regression_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    embedder = SentenceTransformer('all-mpnet-base-v2')
    raw_df = pd.read_csv('clean_nus_sms.csv')
    english_df = pd.read_csv('english_messages.csv')
    initial_labeled_df = pd.read_csv('labeled.csv')
    full_labeled_df = pd.read_csv('fully_labeled.csv')

    return model, label_encoder, embedder, raw_df, english_df, initial_labeled_df, full_labeled_df

model, label_encoder, embedder, raw_df, english_df, initial_labeled_df, full_labeled_df = load_components()

def predict(input):
    embedding = embedder.encode([input], convert_to_tensor=False)
    sentiment = model.predict(embedding)
    label = label_encoder.inverse_transform(sentiment)[0]

    return label[0].upper() + label[1:]

def run():
    st.title('Sentiment Prediction')
    st.write('Dataset: The National University of Singapore SMS Corpus')

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        '1. Initial Data',
        '2. Cleaned English Messages',
        '3. Gemini-Labeled Data',
        '4. Semi-Supervised Labeling',
        '5. Final Classifier',
        '6. Test Final Model'
    ])

    with tab1:
        st.header('Initial Raw Dataset')
        st.write('The original dataset containing all messages before cleaning.')
        st.dataframe(raw_df[['Message', 'length', 'country', 'Date']]) 
        st.write(f'Total messages: {len(raw_df)}')

    with tab2:
        st.header('Filtered English Messages')
        st.write('Removed all non-English and empty messages.')
        st.dataframe(english_df[['Message']])
        st.write(f'English messages retained: {len(english_df)}')
    
    with tab3:
        st.header('30% Gemini-Labeled Data')
        st.write('Used Gemini API to label about 30% of the English messages.')
        st.dataframe(initial_labeled_df[['Message', 'Sentiment Label']])
        st.bar_chart(initial_labeled_df['Sentiment Label'].value_counts())
    
    with tab4:
        st.header('Semi-Supervised Labeling')
        st.markdown('Trained a Logistic Regression model using *all-mpnet-base-v2* embeddings on the Gemini-labeled data.')
        st.write('Labeled the remaining 70% data using this model.')
        st.bar_chart(full_labeled_df['Sentiment Label'].value_counts().rename(lambda x: x[0].upper() + x[1:]))
    
    with tab5:
        st.header('Final Classifier (Trained on Full Data)')
        st.metric('Accuracy', '87%')
        st.metric('F1 (Macro Avg)', '85%')
        st.write('Model trained on all labeled messages using Logistic Regression and sentence embeddings.')
    
    with tab6:
        st.header('Test the Final Model')
        text = st.text_area('Type something...', height=150)
        button = st.button('Predict Sentiment')
        if button:
            if text.strip() == '':
                st.warning('Please enter a message.')
            else:
                prediction = predict(text)
                if prediction == 'Positive':
                    st.success(prediction)
                elif prediction == 'Negative':
                    st.error(prediction)
                else:
                    st.info(prediction)

run()
