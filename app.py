import os
import logging
from flask import Flask, request, jsonify
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from datetime import datetime
 current_date = datetime.today().strftime('%Y-%m-%d')
# Load environment variables
load_dotenv()

app = Flask(__name__)

# Environment Variables
API_SECRET_KEY = os.getenv("API_SECRET_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = "llama3-70b-8192"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 1024
CURRENT_DATE = datetime.today().strftime('%Y-%m-%d')

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize LLM
def initialize_llm():
    """Initialize the LLM system with Groq API."""
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY is missing. Cannot initialize LLM.")
        return None
    try:
        logger.info("Initializing LLM system...")
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model=DEFAULT_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS
        )
    except Exception as e:
        logger.exception("Failed to initialize LLM system: %s", str(e))
        return None

# Process Transaction Message
def process_transaction_message(message, llm):
    """Process transaction messages and extract structured details."""
    if llm is None:
        logger.error("LLM system is not initialized.")
        return {"error": "LLM system is not initialized."}

     system_prompt = (
        "Your input is a transaction message extracted from voice. Extract structured details like Amount, "
        "Transaction Type, Bank Name, Card Type, Paid To, Merchant, Transaction Mode, Transaction Date, Reference Number, and Tag."
        "Transaction Type should be consistant either debit or credit"
        "transaction date formate in dd/mm/yy"
        "Tag meaning which category of spending: if Amazon, then Shopping; if Zomato, then Eating, etc."
        "Just return the JSON output only. Don't say anything else. If no output, return null."
        "If mode of payment is not mentioned, assume cash by default."
        "If any field is missing, set it as null."
        "Return only a JSON or a list of JSON objects."
        "Handle unstructured, grammatically incorrect, and short human input."
        "Example: 'today I spent 500 at Domino's' should be extracted correctly."
        "If the user mentions multiple items with multiple prices, generate a list of JSON objects."
        f"""
        Today's date is {current_date}. You must use this date when interpreting time-related queries.
        For example:
        - If a user says "this month," assume it is the current month.
        """
        )

    input_prompt = f"{system_prompt}\nMessage: {message}"
    
    try:
        logger.info("Sending request to LLM...")
        response = llm.invoke(input_prompt)
        logger.info("Received response from LLM.")
        return response.content if hasattr(response, "content") else response
    except Exception as e:
        logger.exception("Error processing transaction message: %s", str(e))
        return {"error": str(e)}

# API Endpoint with Authentication
@app.route("/process", methods=["POST"])
def process_text():
    """API endpoint to process transaction messages with authentication."""
    try:
        # Check Authorization Header
        auth_header = request.headers.get("Authorization")
        if not auth_header or auth_header != f"Bearer {API_SECRET_KEY}":
            logger.warning("Unauthorized access attempt.")
            return jsonify({"error": "Unauthorized"}), 401

        # Validate Input
        data = request.get_json()
        if not data or "text" not in data:
            logger.warning("Invalid request: Missing 'text' parameter.")
            return jsonify({"error": "Missing 'text' parameter"}), 400

        # Process Message
        logger.info("Processing transaction message: %s", data["text"])
        llm = initialize_llm()
        if llm is None:
            return jsonify({"error": "LLM initialization failed"}), 500

        json_output = process_transaction_message(data["text"], llm)
        return jsonify(json_output)

    except Exception as e:
        logger.exception("Unexpected error in /process endpoint: %s", str(e))
        return jsonify({"error": str(e)}), 500

# Start Flask App
if __name__ == "__main__":
    logger.info("Starting Flask API server...")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=False)
