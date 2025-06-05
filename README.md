# README.md
# AgriBot - Ultimate Agriculture Dashboard

AgriBot is a comprehensive Streamlit-based dashboard for smart farming, crop management, and agricultural insights. It empowers Indian farmers with AI-driven recommendations, real-time weather and market data, and advanced analysis tools.

## Features

- 🤖 **AI-Powered Chat Assistant**: Get expert agricultural advice using Google Gemini AI.
- 📸 **Image Analysis**: Identify pests and diseases from crop images.
- 📄 **Document Analysis**: Analyze soil reports and agricultural documents.
- 🌐 **Web Analysis**: Extract insights from agricultural websites and articles.
- 🌡️ **Weather Integration**: Real-time weather data and forecasts.
- 📊 **Market Intelligence**: Live crop prices and market trends.
- 🌱 **Crop Recommendations**: AI-powered crop selection based on your conditions.
- 📚 **Knowledge Base**: Comprehensive agricultural information.
- 📱 **Export Options**: Download your data in multiple formats.
- 🌍 **Multi-language Support**: Available in multiple Indian languages.

## Setup Instructions

1. **Clone the repository** and navigate to the project directory.

2. **Create a `.env` file** in the root directory with your API keys:
    ```
    GOOGLE_API_KEY=your_google_api_key_here
    WEATHER_API_KEY=your_openweather_api_key_here
    NEWS_API_KEY=your_news_api_key_here
    GOV_CROP_API_KEY=your_gov_crop_api_key_here
    GOV_CROP_API_URL=https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the app**:
    ```sh
    streamlit run app4.py
    ```

## Usage

- Enter your name in the sidebar to start.
- Use the tabs to chat with AgriBot, view dashboards, analyze images/documents, check market trends, and get weather-based advice.
- Upload images or documents for AI analysis.
- Search the knowledge base or use the web analysis tool for external content.

## Data Sources

- **Weather**: OpenWeatherMap API
- **Market Prices**: Indian Government Open Data API
- **News**: NewsAPI and agricultural RSS feeds
- **AI Responses**: Google Gemini API

## Support

For support or feature requests, contact the developer:

- **Dr. Anil Kumar Singh**  
  [LinkedIn](https://www.linkedin.com/in/anil-kumar-singh-phd-b192554a/)  
  📱 Mob: 7011502494  
  ✉️ Email: singhanil854@gmail.com

---

*Made with ❤️ for Indian Farmers | AgriBot v2.1*
