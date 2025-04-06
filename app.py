import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import requests
from twilio.rest import Client  # type: ignore
from nltk.sentiment import SentimentIntensityAnalyzer
import speech_recognition as sr

# Load the trained mental health model
pipe_lr = joblib.load(open(r'/Users/saisrivatsat/Downloads/MindMate!/saved_models/LogisticRegression_model.pkl', 'rb'))

# Initialize SentimentIntensityAnalyzer for sentiment analysis
sia = SentimentIntensityAnalyzer()

# Initialize Speech Recognizer
recognizer = sr.Recognizer()

# Function for mental health predictiona
def predict_mhealth(docs):
    results = pipe_lr.predict([docs])
    return results[0]

# Get prediction probabilities
def get_predictions_proba(docs):
    results = pipe_lr.predict_proba([docs])
    return results

# Function to send WhatsApp message when high suicide risk is detected
def send_alert_message(emergency_contact):
    to_number = f"whatsapp:{emergency_contact}"  
    from_number = "whatsapp:+14155238886" 
    account_sid = 'AC494751754ca953489010cb3151d36bf1'
    auth_token = '26aef7e58a4119e45bc0757254fbb09a'
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        from_=from_number,
        body='🚨Urgent: A high suicide risk score has been detected. Please reach out to your loved one and provide support during this time.',
        to=to_number
    )
    print("WhatsApp message sent. SID:", message.sid)

# Get risk level and suggestion based on prediction and confidence
def get_risk_level(prediction, confidence, emergency_contact=None):
    pred = prediction.lower()
    if pred == "suicidal" and confidence > 0.50 and emergency_contact:
        send_alert_message(emergency_contact)
    if pred == "depression":
        if confidence > 0.85:
            return "High Risk – Please consider reaching out to a professional.", "Call a helpline or talk to someone you trust."
        elif confidence > 0.6:
            return "Moderate Risk – Signs of distress detected.", "Try journaling or a calming activity."
        else:
            return "Mild Indicators – Monitor your mood.", "Consider mindfulness or taking a short walk."
    elif pred == "anxiety":
        if confidence > 0.85:
            return "High Anxiety Detected – Take action.", "Try deep breathing or grounding exercises."
        elif confidence > 0.6:
            return "Moderate Anxiety – Be kind to yourself.", "Take a short break, or listen to calming music."
        else:
            return "Slight Signs of Stress – Stay aware.", "Stay hydrated and take things slow."
    elif pred == "bipolar":
        if confidence > 0.85:
            return "High Risk – Please reach out to a mental health professional.", "Consider seeing a therapist or counselor."
        elif confidence > 0.6:
            return "Moderate Risk – Potential mood swings detected.", "Keep a mood journal to track your emotions."
        else:
            return "Mild Indicators – Stay aware of mood changes.", "Try relaxation techniques to calm your mood."
    elif pred == "personality disorder":
        if confidence > 0.85:
            return "High Risk – Seek professional help immediately.", "A mental health professional can offer support."
        elif confidence > 0.6:
            return "Moderate Risk – Monitor your emotional well-being.", "Engage in therapy or social support activities."
        else:
            return "Mild Indicators – Be mindful of your behavior and emotional states.", "Consider practicing mindfulness or self-reflection."
    elif pred == "normal":
        return "All Good – No concerns detected.", "Keep doing what works for your wellness!"
    elif pred == "stress":
        if confidence > 0.85:
            return "High Stress Detected – Take immediate action.", "Consider taking a break, deep breathing, or exercise."
        elif confidence > 0.6:
            return "Moderate Stress – Find ways to relax and calm your mind.", "Take a short walk or practice yoga."
        else:
            return "Mild Stress – Stay aware of your stress levels.", "Take a moment to breathe and relax."
    elif pred == "suicidal":
        if confidence > 0.85:
            return "High Risk – Urgent action needed.", "Call a crisis hotline or seek immediate professional help."
        elif confidence > 0.6:
            return "Moderate Risk – Signs of severe distress.", "Please reach out to someone you trust or a mental health professional."
        else:
            return "Mild Signs – Monitor your thoughts and feelings.", "Consider reaching out for emotional support."
    else:
        return "All Good – No concerns detected.", "Keep doing what works for your wellness!"

# Function to get a YouTube video link with error handling
def get_youtube_video(query):
    api_key = 'AIzaSyCWCXD50sfmr14bjPY1r7V4lkz-kqhnSww'
    url = f'https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=1&q={query}&key={api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'items' in data and len(data['items']) > 0:
            video_id = data['items'][0]['id']['videoId']
            return f'https://www.youtube.com/watch?v={video_id}'
        else:
            return "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Fallback video
    except Exception as e:
        st.error(f"Error fetching YouTube video: {e}")
        return "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Fallback video

# Sentiment Analysis: Classify text into Positive, Negative, or Neutral
def get_sentiment(docs):
    sentiment = sia.polarity_scores(docs)
    if sentiment['compound'] >= 0.05:
        return 'positive'
    elif sentiment['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Function for generating a static chatbot response based on prediction
def generate_response(raw_text, prediction, confidence):
    pred = prediction.lower()
    if pred == "depression":
        response = (
            "I’m really sorry you’re feeling this way. Depression can be heavy, but you’re not alone. "
            "Try talking to a trusted friend or writing down your thoughts to process them. "
            "Taking small steps, like a short walk, can also help lift your mood."
        )
    elif pred == "anxiety":
        response = (
            "I can sense your anxiety, and I’m here for you. It’s okay to feel overwhelmed sometimes. "
            "Take a moment to breathe deeply—inhale for 4 seconds, hold for 4, and exhale for 4. "
            "Listening to calming music or stepping away for a brief break might also ease your mind."
        )
    elif pred == "bipolar":
        response = (
            "Mood swings can be challenging, and I’m here to support you through this. "
            "Consider tracking your emotions in a journal to identify patterns over time. "
            "Reaching out to a therapist can provide strategies to manage these fluctuations."
        )
    elif pred == "personality disorder":
        response = (
            "I’m here for you as you navigate these feelings. Emotional intensity can be tough. "
            "Try grounding yourself with a mindfulness exercise, like focusing on your surroundings. "
            "A professional can offer tailored guidance, so consider reaching out for support."
        )
    elif pred == "normal":
        response = (
            "It looks like you’re doing well, and that’s wonderful to see! "
            "Keep nurturing your well-being with activities that bring you joy. "
            "Maybe try a new hobby or spend time with loved ones to stay connected."
        )
    elif pred == "stress":
        response = (
            "Stress can feel overwhelming, but you’ve got this! I’m here to help. "
            "Take a short break to stretch or practice deep breathing to calm your mind. "
            "A warm cup of tea or a quick walk outside might also help you unwind."
        )
    elif pred == "suicidal":
        response = (
            "I’m so sorry you’re feeling this way, and I’m here for you. You’re enough, and you matter. "
            "Please reach out to a crisis hotline or a loved one right away—they want to help. "
            "You don’t have to face this alone; support is just a call away."
        )
    elif pred == "no concern":
        response = (
            "It seems like you’re in a good place right now, which is great to hear! "
            "Keep up the positive habits that are working for you. "
            "Maybe take some time to relax with a favorite activity or connect with a friend."
        )
    else:
        response = (
            "I’m here to support you, no matter how you’re feeling. "
            "Try taking a few deep breaths or engaging in a calming activity like listening to music. "
            "If you need to talk, reach out to someone you trust—they’d be glad to listen."
        )

    youtube_link = get_youtube_video(f'guided meditation for {pred}')
    response += f" Here’s a [guided meditation video]({youtube_link}) that might help."
    return response

# Confidence Text Label for Easy Understanding
def get_confidence_label(confidence):
    if confidence >= 0.85:
        return "High Confidence (Model is very sure about the prediction)"
    elif confidence >= 0.6:
        return "Moderate Confidence (Model has some confidence but may be uncertain)"
    else:
        return "Low Confidence (Model is unsure about the prediction)"

# Function to record audio until silence is detected and transcribe it
def record_and_transcribe():
    try:
        with sr.Microphone() as source:
            # Adjust for ambient noise to improve silence detection
            st.info("Adjusting for background noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=1)

            # Configure silence detection
            recognizer.pause_threshold = 3.0  # Stop after 3 seconds of silence
            recognizer.energy_threshold = 600  # Adjust sensitivity to sound (default is 300)

            st.info("Recording... Please speak now. (Recording will stop after 3 seconds of silence)")
            audio = recognizer.listen(source, timeout=3)  # Timeout after 3 seconds if no speech is detected
            st.success("Recording complete! Transcribing...")
            text = recognizer.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        return "Sorry, I couldn’t understand the audio. Please try again."
    except sr.RequestError as e:
        return f"Could not request results from speech recognition service; {e}"
    except sr.WaitTimeoutError:
        return "No speech detected within 5 seconds. Please try again."
    except Exception as e:
        return f"An error occurred: {e}"

# Main Streamlit app
def main():
    st.set_page_config(page_title="MindMate: Mental Health Detection App", layout="wide", initial_sidebar_state="expanded")
    st.title("MindMate - Mental Health Detection Assistant")

    # Add "MindMate" title to the sidebar
    st.sidebar.title("MindMate")

    # Disclaimer and Consent Prompt
    consent = st.sidebar.selectbox(
        "Disclaimer and Consent",
        ["Please read and accept the terms to proceed"]
    )

    if consent == "Please read and accept the terms to proceed":
        st.warning(""" 
            **Disclaimer:**
            This app is not a substitute for professional mental health advice. 
            Please do not take any results as professional suggestions. If you are experiencing a crisis, 
            please contact a mental health professional or a helpline immediately.
        """)

        accept = st.sidebar.checkbox("I accept the terms and conditions.")
        if accept:
            st.sidebar.success("You have accepted the terms. You can now proceed with using the app.")
        else:
            st.sidebar.warning("You must accept the terms to proceed.")

    # Opt-In for Notifications and Emergency Contact Input
    with st.sidebar.form(key='notification_form'):
        opt_in = st.checkbox("I would like to receive emergency alerts via WhatsApp")
        emergency_contact = st.text_input("Enter your emergency contact number (with country code) for alerts", "")
        submit_notification = st.form_submit_button(label='Save Settings')

    if submit_notification:
        if opt_in and emergency_contact:
            st.success(f"Emergency contact saved: {emergency_contact}")
        elif not opt_in:
            st.info("You have opted out of emergency alerts.")
        else:
            st.warning("Please enter a valid emergency contact number.")

    if consent == "Please read and accept the terms to proceed" and accept:
        # Use buttons for navigation instead of a dropdown
        st.sidebar.markdown("### Navigation")
        home_button = st.sidebar.button("Home")
        monitor_button = st.sidebar.button("Monitor")
        about_button = st.sidebar.button("About")

        # Default to Home if no button is clicked
        if 'page' not in st.session_state:
            st.session_state['page'] = 'Home'

        # Update page based on button clicks
        if home_button:
            st.session_state['page'] = 'Home'
        if monitor_button:
            st.session_state['page'] = 'Monitor'
        if about_button:
            st.session_state['page'] = 'About'

        # Display content based on the selected page
        if st.session_state['page'] == 'Home':
            st.subheader('Welcome to MindMate')
            # Add an image to the Home page
            st.image(
                '/Users/saisrivatsat/Downloads/MindMate!/Panda.jpg',
                caption="Your mental health companion – MindMate",
                use_column_width=True
            )
            st.write("""
                MindMate is your companion for mental health awareness and support. Use the **Monitor** section to analyze your emotions 
                and receive personalized suggestions, or explore the **About** section to learn more about how this app works and its impact.
            """)

        elif st.session_state['page'] == 'Monitor':
            st.subheader('Monitor - Mental Health Detection')

            # Tabs for Text and Audio Input
            tab1, tab2 = st.tabs(["Text Input", "Audio Input"])

            # Tab 1: Text Input
            with tab1:
                with st.form(key='Mental_clf'):
                    raw_text = st.text_area('Type how you feel today', height=150)
                    submit_text = st.form_submit_button(label='Submit')
                    clear_text = st.form_submit_button(label='Clear')

                if submit_text and raw_text:
                    prediction = predict_mhealth(raw_text)
                    probability = get_predictions_proba(raw_text)
                    confidence = np.max(probability)
                    sentiment = get_sentiment(raw_text)

                    if confidence < 0.65:
                        prediction = "No Concern"
                        risk_msg = "Don't worry, the score isn't high enough to indicate any serious concern. Stay calm and take a deep breath."
                        suggestion = "Stay positive and enjoy the moment."
                    else:
                        risk_msg, suggestion = get_risk_level(prediction, confidence, emergency_contact if opt_in else None)

                    confidence_label = get_confidence_label(confidence)
                    chatbot_response = generate_response(raw_text, prediction, confidence)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.success('Original Text')
                        st.write(raw_text)
                        st.success("Prediction")
                        st.write(f"**{prediction}**")
                        st.write(f'Confidence: **{confidence:.2f}**')
                        st.write(f"Confidence Level: {confidence_label}")
                        st.markdown("### Risk Assessment")
                        st.info(risk_msg)
                        st.markdown("### Suggestion")
                        st.write(suggestion)
                        st.markdown("### Chatbot Response & Suggestion")
                        st.markdown(chatbot_response, unsafe_allow_html=True)

                    with col2:
                        st.success('Prediction Probability')
                        proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                        proba_df_clean = proba_df.T.reset_index()
                        proba_df_clean.columns = ['Mental_Health', 'Probability']
                        fig = alt.Chart(proba_df_clean).mark_bar().encode(
                            x='Mental_Health',
                            y='Probability',
                            color='Mental_Health'
                        )
                        st.altair_chart(fig, use_container_width=True)

                elif submit_text and not raw_text:
                    st.warning("Please enter some text to analyze.")

                if clear_text:
                    st.session_state['Mental_clf'] = ""  # Clear the text area

            # Tab 2: Audio Input
            with tab2:
                st.markdown("Speak about how you feel today (Recording will stop after 3 seconds of silence)")

                # Button to start recording
                if st.button("Start Recording"):
                    transcribed_text = record_and_transcribe()
                    st.session_state['transcribed_text'] = transcribed_text
                    st.session_state['audio_processed'] = True

                # Display transcribed text and analyze it
                if 'audio_processed' in st.session_state and st.session_state['audio_processed']:
                    raw_text = st.session_state['transcribed_text']
                    st.success("Transcribed Text")
                    st.write(raw_text)

                    # Analyze the transcribed text
                    prediction = predict_mhealth(raw_text)
                    probability = get_predictions_proba(raw_text)
                    confidence = np.max(probability)
                    sentiment = get_sentiment(raw_text)

                    if confidence < 0.65:
                        prediction = "No Concern"
                        risk_msg = "Don't worry, the score isn't high enough to indicate any serious concern. Stay calm and take a deep breath."
                        suggestion = "Stay positive and enjoy the moment."
                    else:
                        risk_msg, suggestion = get_risk_level(prediction, confidence, emergency_contact if opt_in else None)

                    confidence_label = get_confidence_label(confidence)
                    chatbot_response = generate_response(raw_text, prediction, confidence)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.success("Prediction")
                        st.write(f"**{prediction}**")
                        st.write(f'Confidence: **{confidence:.2f}**')
                        st.write(f"Confidence Level: {confidence_label}")
                        st.markdown("### Risk Assessment")
                        st.info(risk_msg)
                        st.markdown("### Suggestion")
                        st.write(suggestion)
                        st.markdown("### Chatbot Response & Suggestion")
                        st.markdown(chatbot_response, unsafe_allow_html=True)

                    with col2:
                        st.success('Prediction Probability')
                        proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                        proba_df_clean = proba_df.T.reset_index()
                        proba_df_clean.columns = ['Mental_Health', 'Probability']
                        fig = alt.Chart(proba_df_clean).mark_bar().encode(
                            x='Mental_Health',
                            y='Probability',
                            color='Mental_Health'
                        )
                        st.altair_chart(fig, use_container_width=True)

        elif st.session_state['page'] == 'About':
            st.subheader('About MindMate')
            st.markdown("""
            ### What is MindMate?

            **MindMate** is an AI-powered mental health detection and support system built to help individuals understand, track, and respond to their emotional well-being. With the power of **machine learning**, **natural language processing**, and **sentiment analysis**, it can identify early signs of:

            - Depression  
            - Anxiety  
            - Stress  
            - Bipolar Disorder  
            - Personality Disorders  
            - Suicidal Ideation  

            The app supports both text and audio inputs, allowing users to express their emotions in the way that feels most comfortable for them. It not only detects potential mental health concerns but also provides practical suggestions, empathetic responses, and emergency safeguards—offering timely help when it matters the most.

            ### Problem Statement

            Mental health issues are often invisible but deeply impactful:

            - Over 280 million people suffer from depression globally.
            - Suicide is among the leading causes of death for youth.
            - Many suffer in silence due to stigma, lack of access, or fear.
            - Therapy and psychiatric care are expensive or out of reach for many.

            Traditional systems are reactive. MindMate is proactive—offering immediate, private, judgment-free mental health insights 24/7 using either text or audio input from the user.

            ### Features that Make a Difference

            #### Real-Time Mental Health Prediction
            - Based on a trained Logistic Regression model
            - Supports both text and audio inputs, analyzed in seconds
            - High accuracy for key mental health conditions

            #### Confidence-Driven Insights
            - Prediction comes with a confidence score (0-1)
            - Clearly labeled as:
                - Low Confidence (< 0.6) - No alert, avoid false positives  
                - Moderate Confidence (0.6–0.85) - Suggest monitoring  
                - High Confidence (≥ 0.85) - Serious attention needed  

            #### Crisis Alert System (Life-Saving Feature)
            - If the detected state is "Suicidal" and confidence is ≥ 0.50
            - AND the user has opted-in + added a guardian contact
            - Then:  
            An automated WhatsApp emergency alert is sent to the contact  
            Message reads: "A high suicide risk score has been detected. Please reach out to your loved one and provide support during this time."

            #### Personalized Chatbot with Sentiment Understanding
            - Uses NLTK’s SentimentIntensityAnalyzer
            - Offers context-aware supportive responses
            - Suggests guided meditation YouTube videos specific to the detected state

            #### Visual Prediction Probability
            - Interactive bar chart showing prediction probability across all classes
            - Enhances transparency in how decisions are made

            #### Tailored Suggestions
            - Practical advice based on prediction:
                - Journaling
                - Breathing exercises
                - Therapy suggestions
                - Calming activities
                - Professional help encouragement

            #### User-Friendly Interface
            - Optimized layout with a compact sidebar to minimize scrolling
            - Adjusted image sizes for better visibility without excessive scrolling
            - Option to clear results in both text and audio input sections for a seamless user experience
            - Professional tone adopted throughout the app to ensure credibility and trust

            ### Who Will Benefit?

            MindMate is designed for:

            - Students and young adults struggling silently
            - Remote workers facing isolation or burnout
            - Elderly users needing emotional monitoring
            - Anyone battling emotional lows, anxiety, or suicidal thoughts
            - Families and guardians, alerted when a crisis may be emerging

            ### Our Vision

            MindMate is not just an app—it’s a lifeline.  
            We envision a future where mental health care is:

            - Accessible to all
            - Built into daily life
            - Driven by empathy, powered by AI

            Our goal is to remove the stigma, normalize conversations, and save lives through early detection and supportive action.

            ---
            **Disclaimer**: MindMate is a supportive assistant, not a licensed healthcare provider. It is meant to complement—not replace—professional mental health services. If you're experiencing a crisis, please seek immediate help from a licensed therapist or call a crisis hotline.
            """)

if __name__ == '__main__':
    main()
    st.success("Thank you for using MindMate!")