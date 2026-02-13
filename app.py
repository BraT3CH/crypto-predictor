"""
Crypto Predictor Web Dashboard
Beautiful UI for cryptocurrency price predictions
Deploy to Streamlit Cloud for free hosting!
"""

import streamlit as st
import requests
import json
from datetime import datetime
import time

# ============================================================
# CONFIGURATION - API Keys
# ============================================================
# For local testing, you can put keys here temporarily
# For deployment, use Streamlit Secrets instead!

# Try to get from Streamlit secrets first (when deployed)
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
    ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")
except:
    # If not in Streamlit Cloud, use these (for local testing only!)
    GROQ_API_KEY = ""  # ← Add your key here for LOCAL testing only
    ANTHROPIC_API_KEY = ""

# Which API to use? "anthropic" or "groq"
USE_API = "groq"  # ← Set to "groq" for free API!

# ============================================================
# DON'T EDIT BELOW THIS LINE
# ============================================================

# Initialize LLM client
llm_client = None
LLM_AVAILABLE = False
API_NAME = "None"

# Try Groq first (FREE!)
if USE_API == "groq" and GROQ_API_KEY and len(GROQ_API_KEY) > 10:
    try:
        from groq import Groq
        llm_client = Groq(api_key=GROQ_API_KEY)
        LLM_AVAILABLE = True
        API_NAME = "Groq (Free)"
        print(f"✅ Groq API loaded! (Free AI)")
    except ImportError:
        print("⚠️ Groq not installed. Run: pip install groq")
    except Exception as e:
        print(f"❌ Groq error: {e}")

# Try Anthropic
elif USE_API == "anthropic" and ANTHROPIC_API_KEY and len(ANTHROPIC_API_KEY) > 10:
    try:
        import anthropic
        llm_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        LLM_AVAILABLE = True
        API_NAME = "Anthropic Claude"
        print(f"✅ Anthropic API loaded!")
    except ImportError:
        print("⚠️ Anthropic not installed. Run: pip install anthropic")
    except Exception as e:
        print(f"❌ Anthropic error: {e}")
else:
    print("⚠️ No API configured. Technical analysis only.")


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Crypto Predictor AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .bullish {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .bearish {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .neutral {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    .signal-item {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 5px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA FETCHING FUNCTIONS
# ============================================================
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_crypto_data(crypto_symbol, days=30):
    """Fetch real crypto data from CoinGecko"""
    crypto_map = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "ADA": "cardano",
        "DOT": "polkadot",
        "DOGE": "dogecoin",
        "XRP": "ripple",
        "MATIC": "matic-network"
    }
    
    coin_id = crypto_map.get(crypto_symbol, "bitcoin")
    
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": days, "interval": "daily"}
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        prices = [p[1] for p in data['prices']]
        volumes = [v[1] for v in data['total_volumes']]
        timestamps = [p[0] for p in data['prices']]
        
        return {
            "prices": prices,
            "volumes": volumes,
            "timestamps": timestamps,
            "current_price": prices[-1],
            "symbol": crypto_symbol,
            "success": True
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def calculate_indicators(data):
    """Calculate technical indicators"""
    prices = data['prices']
    volumes = data['volumes']
    
    # Moving averages
    ma_7 = sum(prices[-7:]) / 7 if len(prices) >= 7 else prices[-1]
    ma_14 = sum(prices[-14:]) / 14 if len(prices) >= 14 else prices[-1]
    ma_30 = sum(prices[-30:]) / 30 if len(prices) >= 30 else prices[-1]
    
    current_price = prices[-1]
    
    # Price changes
    change_1d = ((prices[-1] - prices[-2]) / prices[-2] * 100) if len(prices) > 1 else 0
    change_7d = ((prices[-1] - prices[-8]) / prices[-8] * 100) if len(prices) > 7 else 0
    change_30d = ((prices[-1] - prices[-31]) / prices[-31] * 100) if len(prices) > 30 else 0
    
    # Volatility
    recent_prices = prices[-7:]
    avg_price = sum(recent_prices) / len(recent_prices)
    volatility = sum(abs(p - avg_price) for p in recent_prices) / len(recent_prices)
    volatility_pct = (volatility / avg_price) * 100
    
    # Volume trend
    avg_volume_recent = sum(volumes[-7:]) / 7 if len(volumes) >= 7 else volumes[-1]
    avg_volume_old = sum(volumes[-14:-7]) / 7 if len(volumes) >= 14 else avg_volume_recent
    volume_trend = "INCREASING" if avg_volume_recent > avg_volume_old else "DECREASING"
    volume_change = ((avg_volume_recent - avg_volume_old) / avg_volume_old * 100) if avg_volume_old > 0 else 0
    
    # RSI calculation
    gains = [prices[i] - prices[i-1] for i in range(1, len(prices)) if prices[i] > prices[i-1]]
    losses = [prices[i-1] - prices[i] for i in range(1, len(prices)) if prices[i] < prices[i-1]]
    
    avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else (sum(gains) / len(gains) if gains else 0)
    avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else (sum(losses) / len(losses) if losses else 1)
    
    rs = avg_gain / avg_loss if avg_loss > 0 else 100
    rsi = 100 - (100 / (1 + rs))
    
    return {
        "current_price": current_price,
        "ma_7": ma_7,
        "ma_14": ma_14,
        "ma_30": ma_30,
        "change_1d": change_1d,
        "change_7d": change_7d,
        "change_30d": change_30d,
        "volatility": volatility_pct,
        "volume_trend": volume_trend,
        "volume_change": volume_change,
        "rsi": rsi,
        "avg_volume_recent": avg_volume_recent
    }


def predict_direction(indicators):
    """Predict using technical analysis"""
    bullish_score = 0
    bearish_score = 0
    signals = []
    
    # Moving Average Analysis
    if indicators['current_price'] > indicators['ma_7']:
        bullish_score += 1
        signals.append(("✅", "Price above 7-day MA", "bullish"))
    else:
        bearish_score += 1
        signals.append(("❌", "Price below 7-day MA", "bearish"))
    
    if indicators['ma_7'] > indicators['ma_14']:
        bullish_score += 1
        signals.append(("✅", "7-day MA above 14-day MA", "bullish"))
    else:
        bearish_score += 1
        signals.append(("❌", "7-day MA below 14-day MA", "bearish"))
    
    # Recent Momentum
    if indicators['change_1d'] > 0:
        bullish_score += 1
        signals.append(("✅", f"Positive 24h: {indicators['change_1d']:.2f}%", "bullish"))
    else:
        bearish_score += 1
        signals.append(("❌", f"Negative 24h: {indicators['change_1d']:.2f}%", "bearish"))
    
    if indicators['change_7d'] > 0:
        bullish_score += 1
        signals.append(("✅", f"Positive 7-day: {indicators['change_7d']:.2f}%", "bullish"))
    else:
        bearish_score += 1
        signals.append(("❌", f"Negative 7-day: {indicators['change_7d']:.2f}%", "bearish"))
    
    # RSI Analysis
    if indicators['rsi'] < 30:
        bullish_score += 2
        signals.append(("✅✅", f"RSI {indicators['rsi']:.1f} - Oversold", "bullish"))
    elif indicators['rsi'] > 70:
        bearish_score += 2
        signals.append(("❌❌", f"RSI {indicators['rsi']:.1f} - Overbought", "bearish"))
    else:
        signals.append(("➖", f"RSI {indicators['rsi']:.1f} - Neutral", "neutral"))
    
    # Volume Analysis
    if indicators['volume_trend'] == "INCREASING":
        bullish_score += 1
        signals.append(("✅", f"Volume up {indicators['volume_change']:.1f}%", "bullish"))
    else:
        bearish_score += 1
        signals.append(("❌", f"Volume down {abs(indicators['volume_change']):.1f}%", "bearish"))
    
    # Determine prediction
    if bullish_score > bearish_score:
        direction = "RISE"
        confidence = "High" if bullish_score - bearish_score >= 3 else "Medium"
    elif bearish_score > bullish_score:
        direction = "FALL"
        confidence = "High" if bearish_score - bullish_score >= 3 else "Medium"
    else:
        direction = "NEUTRAL"
        confidence = "Low"
    
    return {
        "direction": direction,
        "confidence": confidence,
        "bullish_score": bullish_score,
        "bearish_score": bearish_score,
        "signals": signals
    }


def get_llm_sentiment(crypto_symbol, indicators, question):
    """Get LLM sentiment analysis"""
    if not llm_client:
        return None
    
    prompt = f"""Analyze this crypto and answer the question:

Cryptocurrency: {crypto_symbol}
Price: ${indicators['current_price']:,.2f}
24h: {indicators['change_1d']:+.2f}%
7d: {indicators['change_7d']:+.2f}%
RSI: {indicators['rsi']:.1f}

Question: {question}

Format:
ANSWER: [YES/NO]
REASONING: [One sentence]"""

    try:
        # Use Groq API (FREE)
        if USE_API == "groq":
            response = llm_client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Free model
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            response_text = response.choices[0].message.content
        
        # Use Anthropic API
        else:
            message = llm_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = message.content[0].text
        
        # Parse response
        answer = "Unknown"
        reasoning = "Analysis unavailable"
        
        for line in response_text.split('\n'):
            if "ANSWER:" in line:
                answer = line.split(":", 1)[1].strip()
            elif "REASONING:" in line:
                reasoning = line.split(":", 1)[1].strip()
        
        return {"answer": answer, "reasoning": reasoning}
        
    except:
        return None


# ============================================================
# MAIN APP
# ============================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Crypto Predictor AI</h1>', unsafe_allow_html=True)
    st.markdown("### Real-time cryptocurrency price predictions using technical analysis")
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    crypto_options = ["BTC", "ETH", "SOL", "ADA", "DOT", "DOGE", "XRP", "MATIC"]
    selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", crypto_options, index=0)
    
    question = st.sidebar.text_input("Ask a question (optional)", 
                                      placeholder="Will BTC rise tomorrow?")
    
    days = st.sidebar.slider("Historical data (days)", 7, 90, 30)
    
    use_llm = st.sidebar.checkbox("Use AI Sentiment Analysis", value=LLM_AVAILABLE)
    
    predict_button = st.sidebar.button("🔮 Predict Now", type="primary", use_container_width=True)
    
    # Info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        "This AI uses technical indicators like RSI, Moving Averages, "
        "and Volume analysis to predict crypto price movements."
    )
    
    # API Status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔑 API Status")
    if LLM_AVAILABLE and llm_client:
        st.sidebar.success(f"✅ AI Active: {API_NAME}")
    else:
        st.sidebar.warning("⚠️ AI Sentiment: Disabled")
        st.sidebar.caption("Technical analysis still works!")
    
    # Main content
    if predict_button:
        with st.spinner(f"Fetching {selected_crypto} data..."):
            data = fetch_crypto_data(selected_crypto, days)
        
        if not data['success']:
            st.error(f"❌ Error fetching data: {data.get('error', 'Unknown error')}")
            return
        
        # Calculate indicators
        indicators = calculate_indicators(data)
        
        # Display current price
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "💰 Current Price",
                f"${indicators['current_price']:,.2f}",
                f"{indicators['change_1d']:+.2f}%"
            )
        
        with col2:
            st.metric(
                "📊 7-Day Change",
                f"{indicators['change_7d']:+.2f}%",
                f"MA: ${indicators['ma_7']:,.2f}"
            )
        
        with col3:
            rsi_delta = "Oversold" if indicators['rsi'] < 30 else ("Overbought" if indicators['rsi'] > 70 else "Neutral")
            st.metric(
                "📈 RSI",
                f"{indicators['rsi']:.1f}",
                rsi_delta
            )
        
        with col4:
            st.metric(
                "📦 Volume Trend",
                indicators['volume_trend'],
                f"{indicators['volume_change']:+.1f}%"
            )
        
        # Make prediction
        prediction = predict_direction(indicators)
        
        # Display prediction
        st.markdown("---")
        st.markdown("### 🎯 Prediction")
        
        # Determine color
        if prediction['direction'] == "RISE":
            box_class = "bullish"
            emoji = "🚀"
        elif prediction['direction'] == "FALL":
            box_class = "bearish"
            emoji = "📉"
        else:
            box_class = "neutral"
            emoji = "➖"
        
        st.markdown(
            f'<div class="prediction-box {box_class}">'
            f'{emoji} {prediction["direction"]} - {prediction["confidence"]} Confidence'
            f'</div>',
            unsafe_allow_html=True
        )
        
        # Score
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🟢 Bullish Signals", prediction['bullish_score'])
        with col2:
            st.metric("🔴 Bearish Signals", prediction['bearish_score'])
        with col3:
            score_diff = prediction['bullish_score'] - prediction['bearish_score']
            st.metric("⚖️ Net Score", f"{score_diff:+d}")
        
        # Technical signals
        st.markdown("---")
        st.markdown("### 📊 Technical Signals")
        
        for emoji, signal, sentiment in prediction['signals']:
            if sentiment == "bullish":
                st.success(f"{emoji} {signal}")
            elif sentiment == "bearish":
                st.error(f"{emoji} {signal}")
            else:
                st.info(f"{emoji} {signal}")
        
        # LLM Analysis
        if use_llm and question:
            st.markdown("---")
            st.markdown("### 🤖 AI Sentiment Analysis")
            
            with st.spinner("Analyzing with AI..."):
                llm_result = get_llm_sentiment(selected_crypto, indicators, question)
            
            if llm_result:
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**AI Answer:** {llm_result['answer']}")
                st.info(f"💡 {llm_result['reasoning']}")
            else:
                st.warning("AI analysis unavailable. Check API key configuration.")
        
        # Price chart
        st.markdown("---")
        st.markdown("### 📈 Price History")
        
        import pandas as pd
        from datetime import datetime as dt
        
        chart_data = pd.DataFrame({
            'Date': [dt.fromtimestamp(ts/1000) for ts in data['timestamps']],
            'Price': data['prices']
        })
        
        st.line_chart(chart_data.set_index('Date'))
        
        # Disclaimer
        st.markdown("---")
        st.warning("⚠️ **Disclaimer:** This is for educational purposes only. Not financial advice. Cryptocurrency trading carries significant risk.")
    
    else:
        # Landing page
        st.markdown("### 👋 Welcome!")
        st.markdown("""
        This tool analyzes cryptocurrency prices using:
        - **RSI** (Relative Strength Index)
        - **Moving Averages** (7, 14, 30 day)
        - **Volume Trends**
        - **Price Momentum**
        - **AI Sentiment** (optional)
        
        👈 **Select a cryptocurrency and click "Predict Now" to get started!**
        """)
        
        # Sample predictions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**🚀 Bullish Signal**\n\nPrice above MA, RSI < 50, Volume increasing")
        
        with col2:
            st.warning("**📉 Bearish Signal**\n\nPrice below MA, RSI > 70, Volume decreasing")
        
        with col3:
            st.success("**➖ Neutral Signal**\n\nMixed indicators, Wait for clearer trend")


if __name__ == "__main__":
    main()
