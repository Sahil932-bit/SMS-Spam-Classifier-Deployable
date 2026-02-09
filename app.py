import streamlit as st
import pickle
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import time
import io
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))

if 'history' not in st.session_state:
    st.session_state.history = []
    
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

def generate_wordcloud(text, style="Default"):
    if not text.strip():
        return None
        
    if style == "Dark":
        background_color = "black"
        colormap = "viridis"
    elif style == "Colorful":
        background_color = "white"
        colormap = "rainbow"
    else:
        background_color = "white"
        colormap = "viridis"
    
    words = text.lower().split()
    word_freq = Counter(words)
    
    if not word_freq:
        return None
        
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color=background_color,
        colormap=colormap,
        max_words=100,
        min_font_size=10,
        max_font_size=100,
        random_state=42,
        contour_width=1,
        contour_color='steelblue'
    ).generate_from_frequencies(word_freq)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def analyze_text(text):
    words = text.lower().split()
    word_freq = Counter(words)
    return word_freq

def apply_chart_theme(fig, theme="Default"):
    if theme == "Dark":
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
    elif theme == "Light":
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(color='black')
        )
    return fig

def analyze_message_insights(text):
    
    words = text.lower().split()
    word_freq = Counter(words)
    top_words = dict(word_freq.most_common(10))
    
    chars = [c.lower() for c in text if c.isalpha()]
    char_freq = Counter(chars)
    top_chars = dict(char_freq.most_common(10))
    

    stats = {
        'total_words': len(words),
        'unique_words': len(set(words)),
        'total_chars': len(text),
        'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0
    }
    
    return {
        'top_words': top_words,
        'top_chars': top_chars,
        'stats': stats
    }

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        body, .stApp {
            background: linear-gradient(-45deg, #1a237e, #0d47a1, #1565c0, #1976d2);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            font-family: 'Poppins', sans-serif;
            color: white;
        }
        .main-container {
            max-width: 1000px;
            margin: 1rem auto;
            padding: 1.5rem;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            animation: fadeIn 0.5s ease-out;
        }
        .header {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(120deg, #ff7f00, #ffff00, #00ff00, #00ffff, #4b0082, #8f00ff);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: rainbow 5s linear infinite;
            margin-bottom: 0.8rem;
            padding: 0.5rem;
        }
        .subtitle {
            text-align: center;
            font-size: 1.1rem;
            color: #e0f7fa;
            margin-bottom: 1.2rem;
            line-height: 1.4;
            font-weight: 500;
        }
        .stTextArea > div > div > textarea {
            background: rgba(255, 255, 255, 0.95) !important;
            border: 2px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 12px !important;
            color: #2d3436 !important;
            font-size: 1rem !important;
            padding: 0.8rem !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
        }
        .stButton > button {
            background: linear-gradient(45deg, #1a237e, #0d47a1) !important;
            color: white !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            padding: 0.6rem 2rem !important;
            border-radius: 50px !important;
            border: none !important;
            width: auto !important;
            margin: 0.8rem auto !important;
            display: block !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            transition: all 0.3s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3) !important;
        }
        .stButton > button[data-testid="baseButton-secondary"] {
            background: linear-gradient(45deg, #ff416c, #ff4b2b) !important;
        }
        .spam-message, .ham-message {
            padding: 1rem;
            border-radius: 15px;
            font-weight: 600;
            text-align: center;
            margin-top: 1rem;
            font-size: 1.1rem;
            animation: fadeIn 0.5s ease-out;
        }
        .spam-message {
            background: linear-gradient(45deg, #ff416c, #ff4b2b);
            color: white;
        }
        .ham-message {
            background: linear-gradient(45deg, #00b09b, #96c93d);
            color: white;
        }
        .stats-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            backdrop-filter: blur(5px);
        }
        .stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #fff;
        }
        .stat-label {
            font-size: 0.9rem;
            color: #e0f7fa;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

vectorizer_path = 'vectorizer.pkl'
model_path = 'model.pkl'

try:
    tfidf = pickle.load(open('vectorizer.pkl','rb'))
    random_forest_model = pickle.load(open('random_forest_model.pkl','rb'))
    naive_bayes_model = pickle.load(open('naive_bayes_model.pkl','rb'))
    logistic_regression_model = pickle.load(open('logistic_regression_model.pkl','rb'))

    models = {
        'Random Forest': random_forest_model,
        'Naive Bayes': naive_bayes_model,
        'Logistic Regression': logistic_regression_model
    }
except Exception as e:
    st.error(f"Error loading model/vectorizer: {e}")

st.markdown("""
    <div class="main-container">
        <div class="header">✨ SMS/Email Spam Shield 🛡️</div>
        <div class="subtitle">
            Advanced AI-powered spam detection system with real-time analysis and detailed insights.
        </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem; background: linear-gradient(45deg, #1a237e, #0d47a1); padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);'>
            <h2 style='color: #ffffff; font-size: 1.5rem; margin-bottom: 0.5rem;'>📊 Analysis Dashboard</h2>
            <p style='color: #e0f7fa; font-size: 0.9rem;'>Customize your analysis experience</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
            <h3 style='color: #e0f7fa; font-size: 1.2rem; margin-bottom: 0.5rem;'>🎯 Analysis Options</h3>
        </div>
    """, unsafe_allow_html=True)
    
    show_wordcloud = st.checkbox("Show Word Cloud", value=True, help="Visualize the most common words in your message")
    show_stats = st.checkbox("Show Statistics", value=True, help="Display detailed message statistics")
    show_history = st.checkbox("Show History", value=True, help="View your analysis history")
    
    if show_history:
        if st.button("Clear History", help="Clear all analysis history"):
            st.session_state.history = []
            st.success("History cleared successfully!")
    
    st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
            <h3 style='color: #e0f7fa; font-size: 1.2rem; margin-bottom: 0.5rem;'>📈 Visualization Settings</h3>
        </div>
    """, unsafe_allow_html=True)
    
    wordcloud_style = st.selectbox(
        "Word Cloud Style",
        ["Default", "Dark", "Colorful"],
        help="Choose the style for your word cloud visualization"
    )
    
    chart_theme = st.selectbox(
        "Chart Theme",
        ["Default", "Dark", "Light"],
        help="Select the theme for your charts"
    )
    
    st.markdown("""
        <div style='background: linear-gradient(45deg, #00b09b, #96c93d); padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
            <h3 style='color: #ffffff; font-size: 1.2rem; margin-bottom: 0.5rem;'>ℹ️ About</h3>
            <p style='color: #ffffff; font-size: 0.9rem;'>
                This dashboard provides advanced analysis tools for spam detection:
                <ul style='color: #ffffff; font-size: 0.9rem;'>
                    <li>Word Cloud visualization</li>
                    <li>Detailed message statistics</li>
                    <li>Analysis history tracking</li>
                    <li>Interactive charts</li>
                </ul>
            </p>
        </div>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload a File (Optional)", type=["txt", "csv"], label_visibility="collapsed")
df_result = pd.DataFrame()

if uploaded_file:
    try:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.error("The uploaded CSV file is empty.")
            elif df.shape[1] < 1:
                st.error("The CSV file must contain at least one column with messages.")
            else:
                input_sms = df.iloc[:, 0].tolist()
                second_column = df.iloc[:, 1].tolist() if df.shape[1] > 1 else [""] * len(input_sms)

                df_result['Message'] = input_sms
                predictions = {}
                
                for model_name, model in models.items():
                    transformed_messages = [transform_text(msg) for msg in input_sms]
                    vector_input = tfidf.transform(transformed_messages)
                    pred = model.predict(vector_input)
                    probs = model.predict_proba(vector_input)
                    confidences = [f"{max(p) * 100:.2f}%" for p in probs]
                    df_result[f'{model_name} Prediction'] = ["Spam" if p == 1 else "Ham" for p in pred]
                    df_result[f'{model_name} Confidence'] = confidences
                df_result['Second Column'] = second_column
                
                
                st.success(f"Successfully processed {len(input_sms)} messages from the CSV file.")
                
               
                csv = df_result.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="spam_detection_results.csv",
                    mime="text/csv",
                    help="Download the analysis results as a CSV file"
                )

        elif uploaded_file.type == "text/plain":
            input_sms = uploaded_file.read().decode("utf-8")
            if not input_sms.strip():
                st.error("The uploaded text file is empty.")
            else:
                st.text_area("📜 File Content Preview:", value=input_sms, height=150, disabled=True)
        else:
            st.error("Please upload either a CSV or TXT file.")
            
    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty.")
    except pd.errors.ParserError:
        st.error("Error parsing the CSV file. Please ensure it's properly formatted.")
    except UnicodeDecodeError:
        st.error("Error reading the file. Please ensure it's encoded in UTF-8 format.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

input_sms = st.text_area("💬 Enter your message:", height=100, placeholder="Type or paste your message here...")

if st.button('🔍 Analyze Message'):
    if input_sms:
        with st.spinner('Analyzing...'):
            time.sleep(0.5)
            try:
                transformed_sms = transform_text(input_sms)
                vector_input = tfidf.transform([transformed_sms])
                
                
                st.markdown("""
                    <div style='background: linear-gradient(45deg, #1a237e, #0d47a1); 
                             padding: 1.5rem; 
                             border-radius: 15px; 
                             margin: 1rem 0; 
                             box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);'>
                        <h2 style='color: #ffffff; text-align: center; margin-bottom: 1rem;'>
                            🤖 Model Predictions
                        </h2>
                    </div>
                """, unsafe_allow_html=True)
                
                
                col1, col2, col3 = st.columns(3)
                
                
                model_styles = {
                    'Random Forest': 'linear-gradient(45deg, #00b09b, #96c93d)',
                    'Naive Bayes': 'linear-gradient(45deg, #ff416c, #ff4b2b)',
                    'Logistic Regression': 'linear-gradient(45deg, #4776E6, #8E54E9)'
                }
                
                predictions = {}
                for (model_name, model), col in zip(models.items(), [col1, col2, col3]):
                    pred = model.predict(vector_input)[0]
                    confidence = model.predict_proba(vector_input)[0][pred] * 100
                    predictions[model_name] = {'result': pred, 'confidence': confidence}
                    
                    with col:
                        st.markdown(f"""
                            <div style='background: {model_styles[model_name]};
                                     padding: 1.2rem;
                                     border-radius: 12px;
                                     box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                                     text-align: center;'>
                                <h3 style='color: white; margin-bottom: 0.8rem; font-size: 1.2rem;'>
                                    {model_name}
                                </h3>
                                <div style='font-size: 1.5rem; color: white; margin-bottom: 0.8rem;'>
                                    {("⚠️ SPAM" if pred == 1 else "✅ HAM")}
                                </div>
                                <div style='font-size: 1.1rem; color: white; opacity: 0.9;'>
                                    Confidence: {confidence:.2f}%
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        
                        st.markdown(f"""
                            <div style='background: rgba(255, 255, 255, 0.2);
                                     border-radius: 10px;
                                     padding: 3px;
                                     margin-top: 0.8rem;'>
                                <div style='background: white;
                                          width: {confidence}%;
                                          height: 6px;
                                          border-radius: 10px;'></div>
                            </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"### {model_name}")
                        if pred == 1:
                            st.error("⚠️ SPAM", icon="🚫")
                        else:
                            st.success("✅ HAM", icon="✨")
                        
                        
                        st.progress(confidence/100)
                        st.markdown(f"**Confidence:** {confidence:.2f}%")

               
                st.markdown("---")
                st.markdown("## 🎯 Overall Consensus")
                
                spam_votes = sum(1 for pred in predictions.values() if pred['result'] == 1)
                consensus_col1, consensus_col2 = st.columns(2)
                
                with consensus_col1:
                    if spam_votes >= 2:
                        st.error("⚠️ Majority Vote: SPAM", icon="⚠️")
                    else:
                        st.success("✅ Majority Vote: HAM", icon="✨")
                
                with consensus_col2:
                    
                    avg_confidence = sum(pred['confidence'] for pred in predictions.values()) / len(predictions)
                    st.metric(
                        label="Average Confidence",
                        value=f"{avg_confidence:.2f}%",
                        delta=f"{avg_confidence - 50:.1f}% above baseline"
                    )

                
                st.markdown("---")
                st.markdown("## 🏆 Best Performing Algorithm")
                
                
                best_algo = max(predictions.items(), key=lambda x: x[1]['confidence'])
                best_name = best_algo[0]
                best_conf = best_algo[1]['confidence']
                best_result = "SPAM" if best_algo[1]['result'] == 1 else "HAM"
                
                # Create an attractive display for the best algorithm
                st.markdown(f"""
                    <div style='background: linear-gradient(45deg, #FFD700, #FFA500);
                             padding: 1.5rem;
                             border-radius: 15px;
                             box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                             text-align: center;
                             margin: 1rem 0;'>
                        <h3 style='color: #1a237e; margin-bottom: 1rem;'>
                            🎯 {best_name}
                        </h3>
                        <div style='font-size: 1.2rem; color: #1a237e; margin-bottom: 0.5rem;'>
                            Prediction: <strong>{best_result}</strong>
                        </div>
                        <div style='font-size: 1.1rem; color: #1a237e;'>
                            Confidence: <strong>{best_conf:.2f}%</strong>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
            

                # Continue with history section
                # Add to history
                st.session_state.history.append({
                    'message': input_sms,
                    'prediction': 'Spam' if spam_votes >= 2 else 'Ham',
                    'confidence': avg_confidence,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

                # Continue with insights section
                if show_wordcloud:
                    st.markdown("## 📊 Message Insights")
                    
                    try:
                        insights = analyze_message_insights(input_sms)
                        
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            
                            fig_words = go.Figure(data=[
                                go.Bar(
                                    x=list(insights['top_words'].keys()),
                                    y=list(insights['top_words'].values()),
                                    marker_color='#1a237e'
                                )
                            ])
                            fig_words.update_layout(
                                title="Top 10 Most Common Words",
                                xaxis_title="Words",
                                yaxis_title="Frequency",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig_words, use_container_width=True)
                            
                            
                            st.markdown("""
                                <div style='background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
                                    <h4 style='color: #ffffff; text-align: center; margin-bottom: 0.5rem;'>📝 Message Statistics</h4>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            stats_col1, stats_col2 = st.columns(2)
                            with stats_col1:
                                st.metric("Total Words", insights['stats']['total_words'])
                                st.metric("Unique Words", insights['stats']['unique_words'])
                            with stats_col2:
                                st.metric("Total Characters", insights['stats']['total_chars'])
                                st.metric("Avg Word Length", f"{insights['stats']['avg_word_length']:.1f}")
                        
                        with col2:
                            
                            fig_chars = go.Figure(data=[
                                go.Bar(
                                    x=list(insights['top_chars'].keys()),
                                    y=list(insights['top_chars'].values()),
                                    marker_color='#0d47a1'
                                )
                            ])
                            fig_chars.update_layout(
                                title="Top 10 Most Common Characters",
                                xaxis_title="Characters",
                                yaxis_title="Frequency",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig_chars, use_container_width=True)
                            
                            
                            word_lengths = [len(word) for word in input_sms.split()]
                            fig_lengths = go.Figure(data=[
                                go.Histogram(
                                    x=word_lengths,
                                    nbinsx=10,
                                    marker_color='#1565c0'
                                )
                            ])
                            fig_lengths.update_layout(
                                title="Word Length Distribution",
                                xaxis_title="Word Length",
                                yaxis_title="Count",
                                template="plotly_white"
                            )
                            st.plotly_chart(fig_lengths, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error generating message insights: {e}")

                if show_stats:
                    st.markdown("""
                        <div style='background: linear-gradient(45deg, #1a237e, #0d47a1); padding: 1.5rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);'>
                            <h3 style='color: #ffffff; text-align: center; margin-bottom: 1rem;'>📈 Message Statistics</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    word_freq = analyze_text(input_sms)
                    top_words = dict(word_freq.most_common(5))
                    
                    fig = go.Figure(data=[
                        go.Bar(x=list(top_words.keys()), y=list(top_words.values()))
                    ])
                    fig.update_layout(title="Top 5 Most Common Words")
                    fig = apply_chart_theme(fig, chart_theme)
                    st.plotly_chart(fig)

                if show_history and st.session_state.history:
                    st.markdown("""
                        <div style='background: linear-gradient(45deg, #1a237e, #0d47a1); padding: 1.5rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);'>
                            <h3 style='color: #ffffff; text-align: center; margin-bottom: 1rem;'>📜 Analysis History</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    history_df = pd.DataFrame(st.session_state.history)
                    
                   
                    with st.container():
                        
                        st.dataframe(
                            history_df.style.set_properties(**{
                                'background-color': 'rgba(255, 255, 255, 0.1)',
                                'color': '#e0f7fa',
                                'border': '1px solid rgba(255, 255, 255, 0.2)'
                            }),
                            use_container_width=True,
                            height=400  
                        )
                        
                       
                        if not history_df.empty:
                            csv = history_df.to_csv(index=False)
                            st.download_button(
                                "Download History",
                                csv,
                                "analysis_history.csv",
                                "text/csv",
                                help="Download your analysis history as a CSV file"
                            )

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("⚡ Please enter a message!")

st.markdown("""
    <div style='text-align: center; margin-top: 2rem; color: #e0f7fa;'>
        <p> Powered by Machine Learning </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
