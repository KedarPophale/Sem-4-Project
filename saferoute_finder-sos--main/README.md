================================================================================
🛡️ SAFEROUTE CHICAGO — AI-POWERED NAVIGATION WITH SOS ALERTING
================================================================================

SafeRoute AI is a specialized navigation engine for Chicago that uses 
Machine Learning to calculate the safest driving path, rather than just the 
shortest. It features an integrated Twilio SOS Emergency System to provide 
real-time location sharing during high-risk travel.

---
📋 TABLE OF CONTENTS
---
1. What This App Does
2. New Feature: SOS Emergency Alerting
3. Project Structure
4. Step-by-Step Setup
5. Twilio Configuration (Required)
6. Tech Stack
7. Troubleshooting

---
🔍 1. WHAT THIS APP DOES
---
- Geocoding: Converts neighborhood names (e.g., "Pilsen", "Loop") into 
  precise GPS coordinates using OpenStreetMap.
- Multi-Route Analysis: Fetches 3 distinct real-world driving routes via 
  the OpenRouteService API.
- Risk Scoring: A trained RandomForest ML Model analyzes 30+ sample points 
  along every route based on time, day, month, and crime density.
- Interactive Results:
    * Green Path: The AI-recommended safest route.
    * Crime Heatmap: Visual overlay of high-risk zones in Chicago.
    * Risk Legend: Real-time safety percentage breakdown for every path.

---
🚨 2. NEW FEATURE: SOS EMERGENCY ALERTING
---
The system now includes a critical safety layer for drivers:
- Real-time GPS Tracking: Captures your exact coordinates via the browser.
- One-Touch SMS: Sends an automated emergency message via Twilio API.
- Smart Context: SMS includes your current location, start/end points, 
  and a direct Google Maps link for emergency responders.
- Feedback: High-visibility pulsing SOS button and audio alert confirmation.

---
📂 3. PROJECT STRUCTURE
---
edilproject/
│
├── main.py                <-- FastAPI server (Logic + SOS System)
├── train_model.py         <-- ML training script (Run this first)
├── crime_model.pkl        <-- Saved ML Model (Auto-generated)
├── crime_data.pkl         <-- Heatmap points (Auto-generated)
├── requirements.txt       <-- List of Python libraries
│
├── templates/
│   ├── index.html         <-- Home page (User Input)
│   └── map.html           <-- Results page (Map + SOS Dashboard)
│
└── static/                <-- Folder for custom CSS/JS assets

---
💻 4. STEP-BY-STEP SETUP (WINDOWS)
---
1. Open Command Prompt and navigate to the project folder:
   cd path/to/edilproject

2. Create a Virtual Environment:
   python -m venv venv

3. Activate the Environment:
   venv\Scripts\activate

4. Install Dependencies:
   pip install -r requirements.txt

5. Train the ML Model (Required before running):
   python train_model.py

6. Start the Web Server:
   uvicorn main:app --reload

7. Open your browser to: http://localhost:8000

---
📲 5. TWILIO CONFIGURATION (REQUIRED FOR SOS)
---
Open 'main.py' and fill in your Twilio credentials to enable the SOS system:

TWILIO_SID    = "AC..."           # Your Account SID
TWILIO_TOKEN  = "your_token"       # Your Auth Token
TWILIO_FROM   = "+1234567890"      # Your Twilio Number
SOS_CONTACT   = "+91XXXXXXXXXX"    # The Emergency Contact Number

NOTE: If using a Twilio Trial Account, you MUST verify the 'SOS_CONTACT' 
number in your Twilio Dashboard before it can receive messages.

---
🛠 6. TECH STACK
---
- Framework: FastAPI (Python 3.10+)
- Emergency API: Twilio REST API
- Mapping: Folium & Leaflet.js
- Routing: OpenRouteService API
- ML Engine: Scikit-Learn (Random Forest)
- Geocoding: Nominatim (OpenStreetMap)

---
⚠️ 7. TROUBLESHOOTING
---
- "jinja2.exceptions.UndefinedError: hour": Ensure you are using the 
  latest main.py that passes 'hour', 'day', and 'month' to the template.
- "Fetching Location..." hangs: Grant Location/GPS permissions in your 
  browser settings.
- SMS Not Received: Check Twilio logs; ensure the recipient number is a 
  "Verified Caller ID" on trial accounts.
- ORS API error 401/403: Verify your ORS_API_KEY is valid and has not 
  exceeded its daily quota.

================================================================================
SafeRoute AI · Demo System · Chicago Crime Dataset · OpenRouteService
================================================================================
