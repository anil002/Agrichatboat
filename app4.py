import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import json
import PyPDF2
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import sqlite3
import hashlib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
import re
import feedparser
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your-google-api-key-here")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "your-openweather-api-key-here")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "your-news-api-key-here")
GOV_CROP_API_KEY = os.getenv("GOV_CROP_API_KEY", "")
GOV_CROP_API_URL = os.getenv("GOV_CROP_API_URL", "")

# Validate API keys
if GOOGLE_API_KEY == "your-google-api-key-here":
    st.error("Google API key is missing. Please set GOOGLE_API_KEY in the .env file.")
if WEATHER_API_KEY == "your-openweather-api-key-here":
    st.warning("Weather API key is missing. Weather features may not work.")
if NEWS_API_KEY == "your-news-api-key-here":
    st.warning("News API key is missing. News features may not work.")

# Configure APIs
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Failed to configure Google Gemini API: {str(e)}")
    model = None

# Helper Functions
def extract_pdf_text(file):
    """Extract text from uploaded PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        if not text.strip():
            return "No text could be extracted from the PDF. It may be scanned or image-based."
        return text[:5000]
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"

def scrape_weblink(url):
    """Scrape and summarize content from a weblink"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        article = soup.find('article') or soup.find('div', class_=re.compile('content|post|article', re.I))
        if article:
            paragraphs = article.find_all('p')
            text = ' '.join([p.get_text().strip() for p in paragraphs])
        else:
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text().strip() for p in paragraphs[:15]])
        
        if not text.strip():
            text = soup.get_text(separator=' ', strip=True)[:3000]
        
        prompt = f"""
        You are AgriBot, an expert agricultural assistant. Analyze the following web content related to agriculture and provide a detailed response tailored to the user's needs.

        Content: {text[:3000]}

        Please provide a comprehensive response including:
        1. Key agricultural insights from the content
        2. Practical recommendations for Indian farmers based on the content
        3. Important facts or data mentioned
        4. Relevance to Indian farming practices and challenges
        5. Potential limitations or gaps in the content
        6. Additional tips for applying the insights effectively

        Ensure the response is actionable, detailed, and at least 300 words, prioritizing Indian agricultural context.
        """
        if model:
            return get_model_response(prompt)
        else:
            return "Cannot summarize content: Google Gemini API is not configured."
    except Exception as e:
        return f"Error scraping weblink: {str(e)}. Please provide the text manually or try another URL."

def get_model_response(prompt, image=None, context=""):
    """Generate enhanced response using Gemini-1.5-Flash with context"""
    if not model:
        return "Error: Google Gemini API is not configured."
    try:
        # Add language instruction at the top
        language_instruction = f"Respond in {st.session_state.language} language.\n"
        enhanced_prompt = f"""
        {language_instruction}
        {context}
        
        You are AgriBot, an expert agricultural assistant for Indian farmers with deep knowledge of:
        - Crop cultivation and management
        - Pest and disease identification
        - Soil management and fertilization
        - Weather-based farming advice
        - Market trends and pricing
        - Sustainable farming practices
        - Government schemes and subsidies
        
        Provide a detailed, practical, and actionable response tailored to the user's query. Ensure the response is:
        - Comprehensive (at least 300 words unless specified otherwise)
        - Structured with clear sections
        - Specific to the query, incorporating relevant details from the context
        - Focused on Indian agricultural practices and challenges
        - Includes practical examples, step-by-step guidance, and potential outcomes
        
        Query: {prompt}
        """

        if image:
            response = model.generate_content(
                [enhanced_prompt, image],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.5,
                    max_output_tokens=4000
                )
            )
        else:
            response = model.generate_content(
                enhanced_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.5,
                    max_output_tokens=4000
                )
            )
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

@st.cache_data
def get_weather_data(city):
    """Fetch current weather and 5-day forecast"""
    try:
        # Current weather
        current_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        current_response = requests.get(current_url, timeout=10)
        current_response.raise_for_status()
        current_data = current_response.json()

        # 5-day forecast
        forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={WEATHER_API_KEY}&units=metric"
        forecast_response = requests.get(forecast_url, timeout=10)
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()

        # Validate forecast data
        if not forecast_data.get("list"):
            return {"error": "No forecast data available for this city."}

        return {
            "current": {
                "temperature": current_data["main"]["temp"],
                "humidity": current_data["main"]["humidity"],
                "description": current_data["weather"][0]["description"],
                "wind_speed": current_data["wind"]["speed"],
                "pressure": current_data["main"]["pressure"]
            },
            "forecast": forecast_data["list"]
        }
    except requests.exceptions.HTTPError as http_err:
        return {"error": f"HTTP error fetching weather data: {str(http_err)}"}
    except requests.exceptions.RequestException as req_err:
        return {"error": f"Error fetching weather data: {str(req_err)}"}
    except KeyError as key_err:
        return {"error": f"Invalid data structure from weather API: {str(key_err)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def get_agricultural_news():
    """Fetch latest agricultural news"""
    try:
        url = f"https://newsapi.org/v2/everything?q=agriculture+farming+crops&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data["articles"][:5]
    except Exception as e:
        return [{"title": f"Error fetching news: {str(e)}", "description": "", "url": ""}]

def get_enhanced_agricultural_news():
    """Get news from multiple sources"""
    news_sources = []
    
    # NewsAPI (current implementation)
    news_sources.extend(get_agricultural_news())
    
    # Add RSS feeds from agricultural websites
    rss_feeds = [
        "https://www.agriculture.com/rss",
        "https://www.agri-pulse.com/rss",
        "https://indianexpress.com/section/india/rss/"
    ]
    
    for feed_url in rss_feeds:
        try:
            feed_data = feedparser.parse(feed_url)
            for entry in feed_data.entries[:3]:
                news_sources.append({
                    "title": entry.title,
                    "description": entry.summary,
                    "url": entry.link,
                    "published": entry.published
                })
        except:
            continue
    
    return news_sources[:10]

def recommend_crops(soil_type, climate, rainfall, temperature):
    """AI-powered crop recommendation based on conditions"""
    recommendations = []
    
    for crop, info in knowledge_base.items():
        score = 0
        
        if soil_type.lower() in info["soil"].lower():
            score += 30
        if climate.lower() in info["climate"].lower():
            score += 25
        temp_range = info["climate"]
        if "°C" in temp_range:
            try:
                temp_nums = [int(x) for x in temp_range.split() if x.replace('°C', '').replace('-', '').isdigit()]
                if len(temp_nums) >= 2 and temp_nums[0] <= temperature <= temp_nums[1]:
                    score += 25
            except:
                pass
        if info["market_demand"] == "Very High":
            score += 15
        elif info["market_demand"] == "High":
            score += 10
        score += info["suitability_score"] * 5
        
        recommendations.append({
            "crop": crop,
            "score": score,
            "info": info
        })
    
    return sorted(recommendations, key=lambda x: x["score"], reverse=True)

# Enhanced agricultural knowledge base
knowledge_base = {
    "wheat": {
        "soil": "Loamy, well-drained, pH 6.0-7.5", 
        "climate": "Temperate, 15-25°C, 450-650mm rainfall",
        "pests": ["Aphids", "Hessian fly", "Rust", "Armyworm"],
        "diseases": ["Leaf rust", "Stripe rust", "Powdery mildew"],
        "recommendations": "Rotate with legumes, ensure proper irrigation, use resistant varieties, apply nitrogen in split doses",
        "sowing_season": "October-December (Rabi)",
        "harvest_season": "March-May",
        "yield_per_hectare": "3-4 tons",
        "suitability_score": 0.85,
        "market_demand": "High",
        "storage_tips": "Moisture content below 12%, proper ventilation"
    },
    "rice": {
        "soil": "Clayey, water-retentive, pH 5.5-6.5",
        "climate": "Tropical/Subtropical, 20-35°C, 1000-2000mm rainfall",
        "pests": ["Brown plant hopper", "Stem borer", "Leaf folder", "Rice bug"],
        "diseases": ["Blast", "Bacterial blight", "Sheath blight"],
        "recommendations": "Maintain flooded fields, use resistant varieties, monitor water levels, integrated pest management",
        "sowing_season": "June-July (Kharif), November-December (Rabi)",
        "harvest_season": "October-November (Kharif), April-May (Rabi)",
        "yield_per_hectare": "4-6 tons",
        "suitability_score": 0.90,
        "market_demand": "Very High",
        "storage_tips": "Proper drying, moisture below 14%, pest-free storage"
    },
    "maize": {
        "soil": "Well-drained, fertile loam, pH 6.0-7.5",
        "climate": "Warm temperate, 20-30°C, 600-1200mm rainfall",
        "pests": ["Corn borer", "Fall armyworm", "Cutworm"],
        "diseases": ["Northern corn leaf blight", "Gray leaf spot"],
        "recommendations": "Use Bt varieties, crop rotation, balanced fertilization",
        "sowing_season": "June-July (Kharif), October-November (Rabi)",
        "harvest_season": "September-October (Kharif), March-April (Rabi)",
        "yield_per_hectare": "5-8 tons",
        "suitability_score": 0.80,
        "market_demand": "High",
        "storage_tips": "Dry to 14% moisture, protect from rodents"
    },
    "sugarcane": {
        "soil": "Deep, well-drained loamy soil, pH 6.0-7.5",
        "climate": "Tropical/Subtropical, 20-30°C, 1500-2500mm rainfall",
        "pests": ["Shoot borer", "Root borer", "Scale insects"],
        "diseases": ["Red rot", "Wilt", "Smut"],
        "recommendations": "Proper drainage, disease-free seeds, balanced nutrition",
        "sowing_season": "February-April, October-November",
        "harvest_season": "12-18 months after planting",
        "yield_per_hectare": "60-80 tons",
        "suitability_score": 0.75,
        "market_demand": "High",
        "storage_tips": "Process immediately or store in cool, dry place"
    },
    "cotton": {
        "soil": "Black cotton soil, well-drained, pH 6.0-8.0",
        "climate": "Semi-arid, 21-30°C, 500-1000mm rainfall",
        "pests": ["Bollworm", "Aphids", "Thrips", "Whitefly"],
        "diseases": ["Fusarium wilt", "Bacterial blight"],
        "recommendations": "Bt cotton varieties, integrated pest management, proper spacing",
        "sowing_season": "April-June",
        "harvest_season": "September-January",
        "yield_per_hectare": "1.5-2.5 tons",
        "suitability_score": 0.70,
        "market_demand": "High",
        "storage_tips": "Dry storage, protect from moisture and pests"
    },
    "soybean": {
        "soil": "Well-drained loamy soil, pH 6.0-7.0",
        "climate": "Warm temperate, 20-30°C, 450-700mm rainfall",
        "pests": ["Soybean aphid", "Pod borer", "Stem fly"],
        "diseases": ["Rust", "Bacterial pustule", "Frogeye leaf spot"],
        "recommendations": "Crop rotation, rhizobium inoculation, balanced fertilization",
        "sowing_season": "June-July",
        "harvest_season": "September-October",
        "yield_per_hectare": "1-2 tons",
        "suitability_score": 0.75,
        "market_demand": "High",
        "storage_tips": "Dry to 11% moisture, store in cool, dry conditions"
    }
}

region_crop_boost = {
    "North India": ["wheat", "rice", "sugarcane", "maize", "cotton"],
    "South India": ["rice", "cotton", "soybean", "maize", "sugarcane"],
    "East India": ["rice", "maize", "sugarcane", "wheat"],
    "West India": ["cotton", "wheat", "soybean", "maize"]
}

def generate_crop_prices():
    base_prices = {
        "wheat": 25.50, "rice": 18.75, "maize": 20.30, 
        "soybean": 35.10, "sugarcane": 2.80, "cotton": 55.20
    }
    current_prices = {}
    price_trends = {}
    
    for crop, base_price in base_prices.items():
        variation = np.random.uniform(-0.1, 0.1)
        current_prices[crop] = round(base_price * (1 + variation), 2)
        days = list(range(30))
        trend = [base_price * (1 + np.random.uniform(-0.15, 0.15)) for _ in days]
        price_trends[crop] = trend
    
    return current_prices, price_trends

def init_database():
    conn = sqlite3.connect('agribot.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            session_id TEXT,
            query TEXT,
            response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_conversation(user_name, session_id, query, response):
    conn = sqlite3.connect('agribot.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_sessions (user_name, session_id, query, response)
        VALUES (?, ?, ?, ?)
    ''', (user_name, session_id, query, response))
    conn.commit()
    conn.close()

def get_real_crop_prices():
    """Fetch real crop prices from Indian Government API"""
    try:
        api_key = GOV_CROP_API_KEY
        api_url = GOV_CROP_API_URL
        params = {
            'api-key': api_key,
            'format': 'json',
            'limit': 100
        }
        response = requests.get(api_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Parse the records to get average modal price for each crop
            crop_map = {
                'wheat': ['wheat', 'gehun'],
                'rice': ['rice', 'chawal', 'paddy'],
                'maize': ['maize', 'makka', 'corn'],
                'sugarcane': ['sugarcane', 'ganna'],
                'cotton': ['cotton', 'kapas'],
                'soybean': ['soybean', 'soya']
            }
            crop_prices = {k: [] for k in crop_map}
            for rec in data.get('records', []):
                commodity = rec.get('commodity', '').lower()
                modal_price = rec.get('modal_price')
                try:
                    price = float(modal_price)
                except:
                    continue
                for crop, names in crop_map.items():
                    if any(name in commodity for name in names):
                        crop_prices[crop].append(price)
            # Calculate average price for each crop
            avg_prices = {}
            for crop, prices in crop_prices.items():
                if prices:
                    avg_prices[crop] = round(sum(prices)/len(prices), 2)
            # Fallback to simulated prices if any crop missing
            fallback = {
                "wheat": 25.50, "rice": 18.75, "maize": 20.30, 
                "soybean": 35.10, "sugarcane": 2.80, "cotton": 55.20
            }
            for crop, val in fallback.items():
                if crop not in avg_prices:
                    avg_prices[crop] = val
            return {'current_prices': avg_prices}
        else:
            st.warning("Could not fetch real price data, using fallback.")
            return {'current_prices': generate_crop_prices()[0]}
    except Exception as e:
        st.warning(f"Error fetching real price data: {e}")
        return {'current_prices': generate_crop_prices()[0]}

def calculate_heat_stress(temp, humidity):
    """Simple heat stress index calculation (0-10 scale)"""
    # Example: higher temp and humidity increases stress
    if temp is None or humidity is None:
        return None
    index = (0.7 * temp + 0.3 * humidity) / 6
    return round(min(max(index, 0), 10), 1)

def calculate_disease_risk(humidity, temp):
    """Estimate disease risk based on humidity and temperature (0-10 scale)"""
    # High humidity and moderate temperature increase risk
    if humidity is None or temp is None:
        return None
    risk = 0
    if humidity > 80 and 20 <= temp <= 35:
        risk = 8 + (humidity - 80) * 0.05
    elif humidity > 60:
        risk = 5 + (humidity - 60) * 0.05
    else:
        risk = 2
    return round(min(max(risk, 0), 10), 1)

def calculate_irrigation_need(temp, humidity, rainfall):
    """Estimate irrigation need based on temperature, humidity, and rainfall (Low/Medium/High)"""
    if temp is None or humidity is None or rainfall is None:
        return None
    if temp > 30 and humidity < 50 and rainfall < 20:
        return "High"
    elif temp > 25 and humidity < 60 and rainfall < 50:
        return "Medium"
    else:
        return "Low"

def get_enhanced_weather_data(city):
    """Enhanced weather data with farming-specific metrics"""
    try:
        # Current implementation + additional farming metrics
        base_data = get_weather_data(city)
        
        # Add UV Index, Soil Temperature, etc.
        # uv_url = f"http://api.openweathermap.org/data/2.5/uvi?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}"
        
        # Dummy values for demonstration
        temp = base_data.get('current', {}).get('temperature', 25)
        humidity = base_data.get('current', {}).get('humidity', 60)
        rainfall = 0  # You may extract from forecast if available
        
        # Add agricultural indices
        base_data['farming_metrics'] = {
            'heat_stress_index': calculate_heat_stress(temp, humidity),
            'disease_risk': calculate_disease_risk(humidity, temp),
            'irrigation_need': calculate_irrigation_need(temp, humidity, rainfall)
        }
        
        return base_data
    except Exception as e:
        return {"error": str(e)}

def calculate_weather_score(weather_data):
    """Calculate a weather score (0-10) based on temperature and humidity for farming."""
    if not weather_data or "error" in weather_data:
        return 5.0  # Default score if data is missing

    try:
        current = weather_data.get('current', {})
        temp = current.get('temperature', 25)
        humidity = current.get('humidity', 60)

        # Score is highest when temp is 20-30°C and humidity is 50-70%
        temp_score = 10 if 20 <= temp <= 30 else max(0, 10 - abs(temp - 25) * 0.5)
        humidity_score = 10 if 50 <= humidity <= 70 else max(0, 10 - abs(humidity - 60) * 0.2)

        # Average the two scores
        score = (temp_score + humidity_score) / 2
        return round(max(0, min(10, score)), 1)
    except Exception:
        return 5.0  # Fallback score

def calculate_market_trend(price_data):
    """Calculate market trend percentage based on price variation."""
    try:
        current_prices = price_data.get('current_prices', {})
        if not current_prices:
            return 0.0
        prices = list(current_prices.values())
        avg_price = sum(prices) / len(prices)
        # Simulate trend as random fluctuation for now
        price_std = np.std(prices) if len(prices) > 1 else 0
        trend = (price_std / avg_price) * 100 * (1 if np.random.random() > 0.5 else -1)
        return round(trend, 1)
    except Exception:
        return 0.0

# Streamlit UI Configuration
st.set_page_config(
    page_title="AgriBot - Ultimate Agriculture Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E8B57, #228B22);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f8f0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #228B22;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'session_id' not in st.session_state:
    st.session_state.session_id = hashlib.sha256(str(datetime.now()).encode()).hexdigest()
if 'language' not in st.session_state:
    st.session_state.language = "English"  # Default language


# Initialize database
init_database()

# Header
st.markdown("""
<div class="main-header">
    <h1>🌾 AgriBot: Ultimate Agriculture Dashboard</h1>
    <p>Your comprehensive solution for smart farming, crop management, and agricultural insights</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=300&h=200&fit=crop", 
             caption="Smart Farming Solutions")
    
    if not st.session_state.user_name:
        st.markdown("### 👤 Welcome!")
        user_name = st.text_input("Enter your name:")
        if st.button("Start Farming Journey") and user_name:
            st.session_state.user_name = user_name
            st.success(f"Welcome, {user_name}!")
            st.rerun()
    else:
        st.markdown(f"### Hello, {st.session_state.user_name}! 👋")
        # --- Add region selection here ---
        region_options = ["North India", "South India", "East India", "West India"]
        st.session_state.region = st.selectbox(
            "Select your region:",
            region_options,
            index=region_options.index(st.session_state.get("region", "North India"))
        )
        # --- End region selection ---
        st.markdown("### 🌐 Language Support")
        st.session_state.language = st.selectbox(
            "Choose your language:",
            [
                "English", "Hindi", "Bengali", "Marathi", "Telugu", "Tamil", "Gujarati", "Urdu",
                "Kannada", "Odia", "Punjabi", "Malayalam", "Assamese", "Maithili", "Santali"
            ],
            index=0
        )
        st.markdown("### 🚀 Quick Actions")
        if st.button("🌡️ Weather Update"):
            st.info("Go to the '🌾 Smart Weather' tab for weather updates.")
        if st.button("📰 Agri News"):
            st.info("Go to the '📈 Market Intelligence' tab for news.")
        if st.button("💰 Market Prices"):
            st.info("Go to the '📈 Market Intelligence' tab for prices.")
        if st.button("🌱 Crop Recommender"):
            st.info("Go to the '🔍 Analysis Tools' tab for crop recommendations.")

# --- About App and Developer Section ---
    st.markdown("""---""")
    st.markdown(
        """
        <div style="background: #e8f5e9; border-radius: 10px; padding: 18px 16px 14px 16px; margin-top: 18px; border-left: 6px solid #388e3c;">
            <h4 style="color: #228B22; margin-bottom: 8px;">ℹ️ About this App</h4>
            <p style="color: #333; font-size: 15px;">
                <b>AgriBot</b> is a comprehensive dashboard for smart farming, crop management, and agricultural insights.<br>
                <span style="color:#388e3c;">Empowering Indian farmers with AI-driven recommendations and real-time data.</span>
            </p>
            <hr style="margin: 10px 0;">
            <div style="display: flex; align-items: center;">
                <img src="https://media.licdn.com/dms/image/C5603AQGkQn5w6Qb1xw/profile-displayphoto-shrink_200_200/0/1516878938992?e=1721865600&v=beta&t=5k9w4k6v7Qw8k8k8k8k8k8k8k8k8k8k8k8k8k8k" width="60" height="60" style="border-radius: 50%; margin-right: 14px; border: 2px solid #388e3c;">
                <div>
                    <span style="font-size: 16px; color: #1a237e;"><b>Developed by</b></span><br>
                    <a href="https://www.linkedin.com/in/anil-kumar-singh-phd-b192554a/" target="_blank" style="color: #1565c0; font-weight: bold; text-decoration: none;">
                        Dr. Anil Kumar Singh
                    </a><br>
                    <span style="color: #333;">📱 Mob: <b>7011502494</b></span><br>
                    <span style="color: #333;">✉️ Email: <b>singhanil854@gmail.com</b></span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Main content
if st.session_state.user_name:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💬 Chat Assistant", 
        "📊 Dashboard", 
        "🔍 Analysis Tools", 
        "📈 Market Intelligence", 
        "📚 Knowledge Base",
        "🌾 Smart Weather"
    ])

    # Fetch real data (move this block above all tab code)
    real_prices = get_real_crop_prices()
    current_prices = real_prices.get('current_prices', {})  # <-- Add this line
    enhanced_weather = get_enhanced_weather_data("Delhi")  # Default city
    _, price_trends = generate_crop_prices()
    
    # --- Add YouTube Agriculture Videos section in the main tab (tab1) ---
    with tab1:
        st.info("""
        **How to use:**  
        - Ask any question about crops, pests, weather, or farming techniques in the text box.
        - You can also upload images or documents for analysis, or provide a web link.
        - The assistant uses Google Gemini AI for responses.

        **Data sources:**  
        - AI answers are generated using Google Gemini API.
        - Weather context is fetched from OpenWeatherMap if you provide a city.
        - Crop knowledge is based on the built-in knowledge base.
        """)

        # --- YouTube Agriculture Videos Section ---
        st.markdown("#### 📺 Explore Agriculture Videos on YouTube")
        st.markdown(
            '[Click here to view Agriculture & Farming videos on YouTube](https://www.youtube.com/results?search_query=agriculture+farming+india)',
            unsafe_allow_html=True
        )

        st.markdown(
            '[Watch more playlists from Indian Farmer on YouTube](https://www.youtube.com/@IndianFarmer/playlists)',
            unsafe_allow_html=True
        )

# ...rest of tab1 code...
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 💬 Chat with AgriBot")
            
            input_method = st.radio("Choose input method:", 
                                  ["💭 Text Query", "📸 Image Analysis", "📄 Document Analysis", "🔗 Web Analysis"])
            
            user_input = st.text_area("Your agricultural query:", 
                                    placeholder="Ask about crops, pests, weather, farming techniques...", 
                                    height=100)
            
            col_city, col_submit = st.columns([3, 1])
            with col_city:
                city = st.text_input("Your location (for weather context):", placeholder="e.g., Delhi, Mumbai")
            
            uploaded_file = None
            weblink = None
            
            if input_method == "📸 Image Analysis":
                uploaded_file = st.file_uploader("Upload crop/pest image:", 
                                               type=["jpg", "png", "jpeg"])
                st.info("💡 Upload images of crops, pests, diseases, or soil for AI analysis")
                
            elif input_method == "📄 Document Analysis":
                uploaded_file = st.file_uploader("Upload agricultural document:", 
                                               type=["pdf", "txt", "docx"])
                st.info("💡 Upload soil reports, farming guides, or research papers")
                
            elif input_method == "🔗 Web Analysis":
                weblink = st.text_input("Enter URL:", 
                                      placeholder="https://example.com/agricultural-article")
                st.info("💡 Analyze agricultural articles, research papers, or news")
            
            with col_submit:
                st.write("")
                st.write("")
                submit_button = st.button("🚀 Analyze", type="primary")
            
            if submit_button and (user_input or uploaded_file or weblink):
                with st.spinner("🤖 AgriBot is thinking..."):
                    context = f"User: {st.session_state.user_name}, Region: {st.session_state.region}, Language: {st.session_state.language}"
                    
                    if city:
                        weather_data = get_weather_data(city)
                        if "error" not in weather_data:
                            context += f"\nWeather in {city}: {weather_data['current']['temperature']}°C, {weather_data['current']['description']}"
                    
                    if input_method == "📸 Image Analysis" and uploaded_file:
                        try:
                            if uploaded_file.size > 10 * 1024 * 1024:
                                st.error("File size exceeds 10MB limit.")
                            else:
                                image = Image.open(uploaded_file)
                                prompt = f"""
                                Analyze this agricultural image for crops, pests, diseases, or soil issues. Context: {user_input}
                                
                                Provide a detailed analysis including:
                                1. Identification of crops, pests, or diseases
                                2. Severity assessment (mild, moderate, severe)
                                3. Step-by-step treatment recommendations
                                4. Preventive measures for Indian farmers
                                5. Expected recovery time and potential yield impact
                                6. Practical examples relevant to Indian farming
                                
                                Ensure the response is at least 300 words and tailored to Indian agricultural practices.
                                """
                                response = get_model_response(prompt, image, context)
                                st.markdown("**Analysis Results:**")
                                st.write(response)
                        except Exception as e:
                            response = f"Error processing image: {str(e)}"
                            st.error(response)
                        
                    elif input_method == "📄 Document Analysis" and uploaded_file:
                        try:
                            if uploaded_file.size > 10 * 1024 * 1024:
                                st.error("File size exceeds 10MB limit.")
                            else:
                                if uploaded_file.type == "application/pdf":
                                    pdf_text = extract_pdf_text(uploaded_file)
                                    if "Error" in pdf_text or "No text" in pdf_text:
                                        st.error(pdf_text)
                                        response = pdf_text
                                    else:
                                        prompt = f"""
                                        Analyze this agricultural document: {pdf_text[:5000]}
                                        Context: {user_input}
                                        
                                        Provide a detailed analysis including:
                                        1. Key findings from the document
                                        2. Practical recommendations for Indian farmers
                                        3. Relevant data or metrics
                                        4. Relevance to Indian farming challenges
                                        5. Limitations of the document
                                        6. Additional tips for applying the findings
                                        
                                        Ensure the response is at least 300 words and tailored to Indian agricultural practices.
                                        """
                                        response = get_model_response(prompt, None, context)
                                        st.markdown("**Document Analysis Results:**")
                                        st.write(response)
                                else:
                                    doc_text = uploaded_file.read().decode('utf-8', errors='ignore')[:5000]
                                    prompt = f"""
                                    Analyze this agricultural document: {doc_text}
                                    Context: {user_input}
                                    
                                    Provide a detailed analysis including:
                                    1. Key findings from the document
                                    2. Practical recommendations for Indian farmers
                                    3. Relevant data or metrics
                                    4. Relevance to Indian farming challenges
                                    5. Limitations of the document
                                    6. Additional tips for applying the findings
                                    
                                    Ensure the response is at least 300 words and tailored to Indian agricultural practices.
                                    """
                                    response = get_model_response(prompt, None, context)
                                    st.markdown("**Document Analysis Results:**")
                                    st.write(response)
                        except Exception as e:
                            response = f"Error processing document: {str(e)}"
                            st.error(response)
                        
                    elif input_method == "🔗 Web Analysis" and weblink:
                        try:
                            scraped_content = scrape_weblink(weblink)
                            if "Error" in scraped_content:
                                st.error(scraped_content)
                                response = scraped_content
                            else:
                                response = scraped_content
                                st.markdown("**Web Analysis Results:**")
                                st.write(response)
                        except Exception as e:
                            response = f"Error processing web content: {str(e)}"
                            st.error(response)
                        
                    else:
                        relevant_crops = [crop for crop in knowledge_base.keys() 
                                        if crop.lower() in user_input.lower()]
                        crop_context = ""
                        for crop in relevant_crops:
                            info = knowledge_base[crop]
                            crop_context += f"\n{crop.title()}: {info['recommendations']}"
                        
                        enhanced_context = context + crop_context
                        prompt = f"""
                        {user_input}

                        Respond in {st.session_state.language} language.

                        Provide a detailed response tailored to the query, including:
                        1. Direct answer to the query
                        2. Step-by-step recommendations for Indian farmers
                        3. Potential challenges and solutions
                        4. Practical examples relevant to Indian farming
                        5. Additional tips for success
                        
                        Ensure the response is at least 300 words, actionable, and specific to Indian agricultural practices.
                        """
                        response = get_model_response(prompt, None, enhanced_context)
                        st.markdown("**Query Response:**")
                        st.write(response)
                    
                    save_conversation(st.session_state.user_name, st.session_state.session_id, 
                                    user_input, response)
                    
                    st.session_state.history.append({
                        "role": "user", 
                        "content": user_input,
                        "method": input_method,
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                    st.session_state.history.append({
                        "role": "bot", 
                        "content": response,
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
        
        with col2:
            st.markdown("### 💬 Conversation History")
            
            for msg in st.session_state.history[-10:]:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>👤 You ({msg['timestamp']}):</strong><br>
                        {msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}
                        {f"<br><small>📝 {msg.get('method', '')}</small>" if msg.get('method') else ''}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>🤖 AgriBot ({msg['timestamp']}):</strong><br>
                        {msg['content'][:300]}{'...' if len(msg['content']) > 300 else ''}
                    </div>
                    """, unsafe_allow_html=True)
            
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()
    
    with tab2:
        st.info("""
        **How to use:**  
        - View real-time crop prices, weather index, and market trends.
        - Check crop price trends and suitability analysis for your region.

        **Data sources:**  
        - Crop prices: Indian Government Open Data API (data.gov.in)
        - Weather: OpenWeatherMap API
        - Crop knowledge: Built-in knowledge base
        """)
        
        st.markdown("### 📊 Agricultural Dashboard")
        
        # Fetch real data
        real_prices = get_real_crop_prices()
        enhanced_weather = get_enhanced_weather_data("Delhi")  # Default city

        # Add this line to get price_trends (simulate if not available from API)
        _, price_trends = generate_crop_prices()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Real crop count from your knowledge base + market data
            active_crops = len(knowledge_base) + len(real_prices.get('additional_crops', []))
            st.markdown(f"""
            <div class="metric-card">
                <h4>🌾 Monitored Crops</h4>
                <h2>{active_crops}</h2>
                <p>Real market data</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Real average price
            real_avg_price = np.mean(list(real_prices.get('current_prices', {}).values()))
            st.markdown(f"""
            <div class="metric-card">
                <h4>💰 Avg Market Price</h4>
                <h2>₹{real_avg_price:.1f}</h2>
                <p>Live market data</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Real weather score based on multiple factors
            weather_score = calculate_weather_score(enhanced_weather)
            st.markdown(f"""
            <div class="metric-card">
                <h4>🌡️ Weather Index</h4>
                <h2>{weather_score}/10</h2>
                <p>Live weather data</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Real market trend from price changes
            market_trend = calculate_market_trend(real_prices)
            st.markdown(f"""
            <div class="metric-card">
                <h4>📈 Market Trend</h4>
                <h2>{market_trend:+.1f}%</h2>
                <p>24h change</p>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Crop Price Trends (30 Days)")
            fig = go.Figure()
            
            for crop, prices in price_trends.items():
                fig.add_trace(go.Scatter(
                    x=list(range(30)),
                    y=prices,
                    mode='lines+markers',
                    name=crop.title(),
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                xaxis_title="Days",
                yaxis_title="Price (₹/unit)",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🌱 Crop Suitability Analysis")
            st.info("""
            **What is Crop Suitability Analysis?**

            Crop Suitability Analysis helps you identify which crops are best suited for your local soil, climate, rainfall, and temperature conditions, and also considers your selected region.
            By using this tool, you can:
            - **Maximize yield** by choosing crops that thrive in your area
            - **Reduce risk** of crop failure
            - **Use resources efficiently** (water, fertilizer, etc.)
            - **Respond to market demand** with the right crops

            The chart below shows the suitability scores for different crops based on scientific and practical knowledge, and your selected region, helping you make smart, data-driven planting decisions.
            """)

            region = st.session_state.get("region", "North India")
            region_crops = region_crop_boost.get(region, [])

            crops = list(knowledge_base.keys())
            scores = []
            for crop in crops:
                base_score = knowledge_base[crop]["suitability_score"]
                # Boost score if crop is preferred in the selected region
                if crop in region_crops:
                    score = min(base_score + 0.10, 1.0)  # Boost by 0.10, max 1.0
                else:
                    score = base_score
                scores.append(score)

            fig = px.bar(
                x=crops,
                y=scores,
                title=f"Suitability Scores for {region}",
                color=scores,
                color_continuous_scale="Greens"
            )
            fig.update_layout(height=400, xaxis_title="Crop", yaxis_title="Suitability Score (Region Adjusted)")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.info("""
        **How to use:**  
        - Select your soil type, climate, rainfall, and temperature to get crop recommendations.
        - Upload plant images for disease detection.

        **Data sources:**  
        - Crop recommendations: Built-in knowledge base and AI scoring
        - Disease detection: Google Gemini AI (image analysis)
        """)
        
        st.markdown("### 🔍 Agricultural Analysis Tools")
        
        st.markdown("#### 🌱 Smart Crop Recommender")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            soil_type = st.selectbox("Soil Type:", 
        [
            "Loamy", "Clayey", "Sandy", "Black Cotton", "Red Soil", "Alluvial", "Laterite", "Peaty", "Saline", "Marshy", "Desert", "Mountain", "Chalky", "Gravelly"
        ]
    )
            climate = st.selectbox("Climate:", 
                                 ["Tropical", "Subtropical", "Temperate", "Arid", "Semi-arid"])
        
        with col2:
            rainfall = st.slider("Annual Rainfall (mm):", 200, 3000, 800)
            temperature = st.slider("Average Temperature (°C):", 10, 45, 25)
        
        with col3:
            st.write("")
            if st.button("🎯 Get Recommendations", type="primary"):
                recommendations = recommend_crops(soil_type, climate, rainfall, temperature)
                
                st.markdown("#### 🏆 Top Crop Recommendations")
                for i, rec in enumerate(recommendations[:3]):
                    crop = rec["crop"]
                    score = rec["score"]
                    info = rec["info"]
                    
                    with st.expander(f"#{i+1} {crop.title()} (Score: {score:.1f})"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write(f"**Soil:** {info['soil']}")
                            st.write(f"**Climate:** {info['climate']}")
                            st.write(f"**Yield:** {info['yield_per_hectare']}")
                        with col_b:
                            st.write(f"**Market Demand:** {info['market_demand']}")
                            st.write(f"**Season:** {info['sowing_season']}")
                            st.write(f"**Price:** ₹{current_prices.get(crop, 'N/A')}")
        
        st.markdown("#### 🦠 Plant Disease Detection")
        disease_image = st.file_uploader("Upload plant image for disease detection:", 
                                       type=["jpg", "png", "jpeg"], 
                                       key="disease_detection")
        
        if disease_image:
            try:
                if disease_image.size > 10 * 1024 * 1024:
                    st.error("File size exceeds 10MB limit.")
                else:
                    image = Image.open(disease_image)
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    with col2:
                        with st.spinner("🔍 Analyzing image for diseases..."):
                            prompt = """
                            Analyze this plant image for diseases, pests, or health issues. Provide:
                            1. Disease/pest identification
                            2. Severity assessment
                            3. Treatment recommendations
                            4. Prevention measures
                            5. Expected recovery time
                            """
                            analysis = get_model_response(prompt, image)
                            st.markdown("**Analysis Results:**")
                            st.write(analysis)
            except Exception as e:
                st.error(f"Error analyzing image: {str(e)}")
    
    with tab4:
        st.info("""
        **How to use:**  
        - View current market prices and set price alerts.
        - Read the latest agricultural news.

        **Data sources:**  
        - Market prices: Indian Government Open Data API (data.gov.in)
        - News: NewsAPI and agricultural RSS feeds
        """)
        
        st.markdown("### 📈 Market Intelligence")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 💰 Current Market Prices")
            price_df = pd.DataFrame([
                {"Crop": crop.title(), 
                 "Current Price (₹/unit)": price, 
                 "Market Demand": knowledge_base[crop]["market_demand"],
                 "Trend": "📈" if np.random.random() > 0.5 else "📉"}
                for crop, price in current_prices.items()
            ])
            st.dataframe(price_df, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Price Alerts")
            alert_crop = st.selectbox("Set price alert for:", list(current_prices.keys()))
            alert_price = st.number_input("Alert when price reaches:", value=current_prices[alert_crop])
            if st.button("📢 Set Alert"):
                st.success(f"Alert set for {alert_crop} at ₹{alert_price}")
        
        st.markdown("#### 📰 Latest Agricultural News")
        news_articles = get_agricultural_news()
        
        for article in news_articles[:3]:
            with st.expander(article.get("title", "News Article")):
                st.write(article.get("description", "No description available"))
                if article.get("url"):
                    st.markdown(f"[Read more]({article['url']})")
    
    with tab5:
        st.info("""
        **How to use:**  
        - Search for crops, pests, or farming topics.
        - Browse the local knowledge base or use Google Gemini for external info.
        - View the seasonal farming calendar.

        **Data sources:**  
        - Knowledge base: Built-in crop database
        - External info: Google Gemini API
        """)
        
        st.markdown("### 📚 Agricultural Knowledge Base")
        
        search_term = st.text_input("🔍 Search crops, pests, or farming topics:", placeholder="e.g., wheat pests, soil management")
        
        if search_term:
            filtered_crops = {
                crop: info for crop, info in knowledge_base.items()
                if search_term.lower() in crop.lower() or 
                   search_term.lower() in str(info).lower()
            }
            
            if filtered_crops:
                st.markdown("#### Local Knowledge Base Results")
                cols = st.columns(2)
                for i, (crop, info) in enumerate(filtered_crops.items()):
                    with cols[i % 2]:
                        with st.expander(f"🌾 {crop.title()}", expanded=False):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown("**Growing Conditions:**")
                                st.write(f"🌱 **Soil:** {info['soil']}")
                                st.write(f"🌡️ **Climate:** {info['climate']}")
                                st.write(f"📅 **Sowing:** {info['sowing_season']}")
                                st.write(f"🌾 **Harvest:** {info['harvest_season']}")
                            with col_b:
                                st.markdown("**Production Info:**")
                                st.write(f"📊 **Yield:** {info['yield_per_hectare']}")
                                st.write(f"💰 **Price:** ₹{current_prices.get(crop, 'N/A')}")
                                st.write(f"📈 **Demand:** {info['market_demand']}")
                                st.write(f"⭐ **Score:** {info['suitability_score']}")
                            st.markdown("**Common Issues:**")
                            st.write(f"🐛 **Pests:** {', '.join(info['pests'])}")
                            st.write(f"🦠 **Diseases:** {', '.join(info['diseases'])}")
                            st.markdown("**Recommendations:**")
                            st.write(info['recommendations'])
                            st.markdown("**Storage Tips:**")
                            st.write(info['storage_tips'])
            else:
                st.info("No matches found in local knowledge base. Searching with Google Gemini API...")
                
                try:
                    search_prompt = f"""
                    Search for agricultural information related to: {search_term}
                    
                    Provide a detailed response including:
                    1. Overview of the topic (crop, pest, or farming practice)
                    2. Practical recommendations for Indian farmers
                    3. Common challenges and solutions
                    4. Relevant data or examples
                    5. Tips for sustainable practices
                    
                    Ensure the response is concise (200-300 words), actionable, and tailored to Indian agriculture.
                    """
                    api_response = get_model_response(search_prompt)
                    st.markdown("#### Gemini API Search Results")
                    st.write(api_response)
                except Exception as e:
                    st.error(f"Error fetching API results: {str(e)}")
        
        else:
            cols = st.columns(2)
            for i, (crop, info) in enumerate(knowledge_base.items()):
                with cols[i % 2]:
                    with st.expander(f"🌾 {crop.title()}", expanded=False):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("**Growing Conditions:**")
                            st.write(f"🌱 **Soil:** {info['soil']}")
                            st.write(f"🌡️ **Climate:** {info['climate']}")
                            st.write(f"📅 **Sowing:** {info['sowing_season']}")
                            st.write(f"🌾 **Harvest:** {info['harvest_season']}")
                        with col_b:
                            st.markdown("**Production Info:**")
                            st.write(f"📊 **Yield:** {info['yield_per_hectare']}")
                            st.write(f"💰 **Price:** ₹{current_prices.get(crop, 'N/A')}")
                            st.write(f"📈 **Demand:** {info['market_demand']}")
                            st.write(f"⭐ **Score:** {info['suitability_score']}")
                        st.markdown("**Common Issues:**")
                        st.write(f"🐛 **Pests:** {', '.join(info['pests'])}")
                        st.write(f"🦠 **Diseases:** {', '.join(info['diseases'])}")
                        st.markdown("**Recommendations:**")
                        st.write(info['recommendations'])
                        st.markdown("**Storage Tips:**")
                        st.write(info['storage_tips'])
        
        st.markdown("#### 📅 Seasonal Farming Calendar")
        
        region = st.session_state.get("region", "North India")
        region_crops = region_crop_boost.get(region, [])

        calendar_data = []
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        for crop in region_crops:
            info = knowledge_base[crop]
            sowing_months = []
            harvest_months = []
            
            if "October-December" in info['sowing_season'] or "Rabi" in info['sowing_season']:
                sowing_months = [10, 11, 12]
            if "June-July" in info['sowing_season'] or "Kharif" in info['sowing_season']:
                sowing_months += [6, 7]
            if "February-April" in info['sowing_season']:
                sowing_months += [2, 3, 4]
            if "April-June" in info['sowing_season']:
                sowing_months += [4, 5, 6]
            
            if "March-May" in info['harvest_season']:
                harvest_months = [3, 4, 5]
            if "October-November" in info['harvest_season']:
                harvest_months += [10, 11]
            if "September-October" in info['harvest_season']:
                harvest_months += [9, 10]
            if "September-January" in info['harvest_season']:
                harvest_months += [9, 10, 11, 12, 1]
            
            for month in range(1, 13):
                activity = ""
                if month in sowing_months:
                    activity = "🌱 Sowing"
                elif month in harvest_months:
                    activity = "🌾 Harvest"
                
                if activity:
                    calendar_data.append({
                        'Crop': crop.title(),
                        'Month': months[month-1],
                        'Activity': activity,
                        'Month_Num': month
                    })

        if calendar_data:
            calendar_df = pd.DataFrame(calendar_data)
            pivot_df = calendar_df.pivot_table(
                index='Crop', 
                columns='Month', 
                values='Month_Num', 
                aggfunc='count', 
                fill_value=0
            )
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            pivot_df = pivot_df.reindex(columns=month_order, fill_value=0)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(pivot_df, annot=True, cmap='Greens', ax=ax, 
                       cbar_kws={'label': 'Activity Level'})
            plt.title(f'Seasonal Farming Calendar for {region} (Major Crops)')
            plt.xlabel('Months')
            plt.ylabel('Crops')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No major crops found for this region.")
    
    with tab6:
        st.info("""
        **How to use:**  
        - Enter your city, crop, and soil type to get weather-based farming advice.
        - View current weather, 5-day forecast, and smart irrigation tips.

        **Data sources:**  
        - Weather: OpenWeatherMap API
        - Crop info: Built-in knowledge base
        """)
        
        st.markdown("### 🌾 Smart Weather Intelligence")
        st.markdown("Advanced weather-based farming recommendations and alerts")
        
        # Input section
        col1, col2, col3 = st.columns(3)
        
        with col1:
            weather_city = st.text_input("🏙️ City:", placeholder="e.g., Delhi, Mumbai", key="smart_weather_city")
            weather_crop = st.selectbox("🌱 Select Crop:", list(knowledge_base.keys()), key="smart_weather_crop")
        
        with col2:
            soil_type = st.selectbox("🌍 Soil Type:", 
        [
            "Loamy", "Sandy", "Clayey", "Black Cotton", "Red Soil", "Alluvial", "Laterite", "Peaty", "Saline", "Marshy", "Desert", "Mountain", "Chalky", "Gravelly"
        ], 
        key="smart_soil_type"
    )
        
        with col3:
            st.write("")
            st.write("")
            get_weather_analysis = st.button("🔍 Analyze Weather", type="primary", key="smart_weather_analyze")
        
        if get_weather_analysis and weather_city and weather_crop:
            with st.spinner("🌤️ Analyzing weather conditions for farming..."):
                
                # Get basic weather data
                weather_data = get_weather_data(weather_city)
                
                if "error" not in weather_data:
                    current = weather_data['current']
                    
                    # Display current weather
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🌡️ Temperature", f"{current['temperature']}°C")
                    with col2:
                        st.metric("💧 Humidity", f"{current['humidity']}%")
                    with col3:
                        st.metric("💨 Wind Speed", f"{current['wind_speed']} m/s")
                    with col4:
                        st.metric("📊 Pressure", f"{current['pressure']} hPa")
                    
                    # Weather-based farming recommendations
                    st.markdown("#### 🌾 Farming Recommendations")
                    
                    temp = current['temperature']
                    humidity = current['humidity']
                    wind_speed = current['wind_speed']
                    
                    recommendations = []
                    alerts = []
                    
                    # Temperature-based recommendations
                    if temp > 35:
                        alerts.append("🔴 **High Temperature Alert**: Extreme heat may stress crops")
                        recommendations.append("• Increase irrigation frequency")
                        recommendations.append("• Provide shade to sensitive crops")
                        recommendations.append("• Harvest early morning or evening")
                    elif temp < 10:
                        alerts.append("🔵 **Cold Temperature Alert**: Risk of frost damage")
                        recommendations.append("• Cover young plants with protective material")
                        recommendations.append("• Use row covers or greenhouse protection")
                        recommendations.append("• Delay planting of sensitive crops")
                    else:
                        recommendations.append("✅ Temperature is optimal for most farming activities")
                    
                    # Humidity-based recommendations
                    if humidity > 80:
                        alerts.append("🟡 **High Humidity Alert**: Increased disease risk")
                        recommendations.append("• Monitor for fungal diseases")
                        recommendations.append("• Improve air circulation around plants")
                        recommendations.append("• Reduce watering frequency")
                    elif humidity < 30:
                        alerts.append("🟠 **Low Humidity Alert**: Plants may experience stress")
                        recommendations.append("• Increase irrigation")
                        recommendations.append("• Use mulching to retain moisture")
                        recommendations.append("• Consider misting for sensitive crops")
                    
                    # Wind-based recommendations
                    if wind_speed > 10:
                        alerts.append("💨 **Strong Wind Alert**: Risk of physical damage")
                        recommendations.append("• Secure plant supports and stakes")
                        recommendations.append("• Harvest ripe crops if possible")
                        recommendations.append("• Check for damaged branches or stems")
                    
                    # Display alerts
                    if alerts:
                        st.markdown("#### 🚨 Weather Alerts")
                        for alert in alerts:
                            st.warning(alert)
                    else:
                        st.success("✅ Weather conditions are favorable for farming")
                    
                    # Display recommendations
                    st.markdown("#### 💡 Smart Recommendations")
                    for rec in recommendations:
                        st.write(rec)
                    
                    # Crop-specific advice
                    crop_info = knowledge_base[weather_crop]
                    st.markdown(f"#### 🌱 {weather_crop.title()}-Specific Advice")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Current Conditions vs Crop Requirements:**")
                        st.write(f"• **Ideal Climate:** {crop_info['climate']}")
                        st.write(f"• **Current Weather:** {weather_data['current']['description']}")
                        st.write(f"• **Soil Preference:** {crop_info['soil']}")
                        st.write(f"• **Selected Soil:** {soil_type}")
                    
                    with col2:
                        st.markdown("**Pest & Disease Watch:**")
                        for pest in crop_info['pests'][:3]:
                            st.write(f"🐛 {pest}")
                        for disease in crop_info['diseases'][:3]:
                            st.write(f"🦠 {disease}")
                    
                    # 5-day forecast
                    st.markdown("#### 📅 5-Day Weather Forecast")
                    
                    if weather_data.get('forecast'):
                        forecast_list = weather_data['forecast'][:40:8]  # Every 8th entry (24 hours apart)
                        
                        forecast_data = []
                        for entry in forecast_list:
                            date = entry['dt_txt'][:10]
                            temp = entry['main']['temp']
                            humidity = entry['main']['humidity']
                            description = entry['weather'][0]['description']
                            
                            forecast_data.append({
                                'Date': date,
                                'Temperature (°C)': temp,
                                'Humidity (%)': humidity,
                                'Conditions': description
                            })
                        
                        forecast_df = pd.DataFrame(forecast_data)
                        st.dataframe(forecast_df, use_container_width=True)
                        
                        # Forecast chart
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_df['Date'],
                            y=forecast_df['Temperature (°C)'],
                            mode='lines+markers',
                            name='Temperature (°C)',
                            line=dict(color='orange', width=3)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_df['Date'],
                            y=forecast_df['Humidity (%)'],
                            mode='lines+markers',
                            name='Humidity (%)',
                            yaxis='y2',
                            line=dict(color='blue', width=3)
                        ))
                        
                        fig.update_layout(
                            title=f"5-Day Weather Forecast for {weather_city}",
                            xaxis_title="Date",
                            yaxis_title="Temperature (°C)",
                            yaxis2=dict(title="Humidity (%)", overlaying='y', side='right'),
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Irrigation recommendations
                    st.markdown("#### 💧 Smart Irrigation Schedule")
                    
                    # Simple irrigation logic based on weather
                    if temp > 30 and humidity < 50:
                        irrigation_advice = "🔴 **High Priority**: Irrigate daily, preferably early morning"
                        water_need = "High (8-10 L/m²)"
                    elif temp > 25 and humidity < 60:
                        irrigation_advice = "🟡 **Medium Priority**: Irrigate every 2 days"
                        water_need = "Medium (5-7 L/m²)"
                    else:
                        irrigation_advice = "🟢 **Low Priority**: Monitor soil moisture, irrigate as needed"
                        water_need = "Low (3-5 L/m²)"
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Irrigation Priority:** {irrigation_advice}")
                        st.write(f"**Estimated Water Need:** {water_need}")
                    
                    with col2:
                        st.markdown("**Best Irrigation Times:**")
                        st.write("• Early morning (6:00-8:00 AM)")
                        st.write("• Evening (6:00-8:00 PM)")
                        st.write("• Avoid midday watering")
                
                else:
                    st.error(f"Unable to fetch weather data: {weather_data['error']}")
        
        # Quick weather tips
        st.markdown("#### ❓ Weather-Based Farming Tips")
        
        with st.expander("🌡️ Temperature Management"):
            st.write("""
            **Hot Weather (>30°C):**
            • Increase watering frequency
            • Provide shade cloth for sensitive crops
            • Harvest during cooler hours
            • Apply mulch to reduce soil temperature
            
            **Cold Weather (<15°C):**
            • Use row covers or plastic tunnels
            • Plant cold-resistant varieties
            • Protect from frost with water spraying
            • Delay planting of warm-season crops
            """)
        
        with st.expander("💧 Humidity Management"):
            st.write("""
            **High Humidity (>80%):**
            • Improve air circulation
            • Monitor for fungal diseases
            • Reduce irrigation frequency
            • Apply preventive fungicides
            
            **Low Humidity (<40%):**
            • Increase watering
            • Use drip irrigation systems
            • Apply organic mulch
            • Consider shade netting
            """)
        
        with st.expander("💨 Wind Protection"):
            st.write("""
            **Strong Winds (>15 km/h):**
            • Install windbreaks
            • Stake tall plants securely
            • Harvest ripe fruits early
            • Check and repair plant supports
            
            **Gentle Breeze (5-15 km/h):**
            • Good for pollination
            • Helps prevent fungal diseases
            • Natural air circulation
            • Monitor for beneficial effects
            """)

else:
    # Show welcome message for users not logged in
    st.markdown("""
    ## Welcome to AgriBot! 🌾
    
    Please enter your name in the sidebar to start your smart farming journey.
    
    ### What you'll get access to:
    - **🤖 AI-Powered Agricultural Assistant**
    - **🌤️ Smart Weather Intelligence**
    - **📊 Real-time Market Data**
    - **🔍 Advanced Analysis Tools**
    - **📚 Comprehensive Knowledge Base**
    - **And much more!**
    """)