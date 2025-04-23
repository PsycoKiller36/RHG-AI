import os
import uuid
from datetime import datetime, timezone
import json
import io
import base64
import requests
import logging # Import logging
import logging.handlers # For rotating file handler

import google.generativeai as genai
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import SQLAlchemyError # Specific DB errors
from dotenv import load_dotenv

load_dotenv() # Load variable from .env file
print(f"Checking key after load_dotenv: {os.environ.get('GOOGLE_API_KEY')}") # Debug print

# --- Logging Configuration ---
# Recommended: Configure logging level via environment variable too
log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
hugging_face_available = False
try:
    # Gemini API Key (Required)
    GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("Gemini API Key loaded.")

    # Hugging Face Config (Optional - for image generation)
    HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")
    # Default model if not set in env var
    HF_IMAGE_MODEL_ID = os.environ.get("HF_IMAGE_MODEL_ID", "runwayml/stable-diffusion-v1-5")
    HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL_ID}"

    if HUGGING_FACE_API_KEY:
        hugging_face_available = True
        logger.info(f"Hugging Face API Key loaded. Using model: {HF_IMAGE_MODEL_ID}")
    else:
        logger.warning("HUGGING_FACE_API_KEY environment variable not set. Image generation will be disabled.")

    # ** Database URL (Required for deployed envs like Render/Neon/Supabase) **
    # Render/Heroku typically set DATABASE_URL. Use this standard.
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
        # SQLAlchemy requires 'postgresql://' instead of 'postgres://'
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        logger.info("DATABASE_URL found and configured for PostgreSQL.")

    # ** Frontend URL for CORS (Required for deployed env) **
    # Allow configuring multiple origins if needed, split by comma?
    FRONTEND_URL = os.environ.get('FRONTEND_URL') # e.g., https://your-site-name.netlify.app
    if not FRONTEND_URL:
        logger.warning("FRONTEND_URL environment variable not set. CORS might block frontend.")
        # Default to allowing localhost for local dev if not set? Or require it.
        # FRONTEND_URL = "http://localhost:5173" # Vite default port for Vue

except KeyError as e:
    logger.critical(f"FATAL: Required environment variable {e} not set. Application cannot start.")
    exit() # Exit if critical keys like Gemini are missing

# --- Initialize Flask App ---
app = Flask(__name__)


# --- NEW: Logging Configuration ---
# Configuration should happen right after app object is created
# and before the logger is used by other parts like CORS setup.

# Remove Flask's default handlers ONLY if you want complete control
# Might be needed if Flask in debug mode adds its own later causing duplicates
# Or configure the root logger instead of app.logger if preferred
# for handler in list(app.logger.handlers):
#    app.logger.removeHandler(handler)

log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)

log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_filename = 'app.log' # Name of the log file

# ** File Handler (Rotating) **
# Rotates logs so they don't grow infinitely large.
# maxBytes: Size in bytes before rotating (e.g., 5MB)
# backupCount: How many old log files to keep
max_log_size = 5 * 1024 * 1024 # 5 MB
backup_count = 3
file_handler = logging.handlers.RotatingFileHandler(
    log_filename, maxBytes=max_log_size, backupCount=backup_count
)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(log_level) # Set level for file logging

# ** Console Handler **
# Ensure logs still go to the console/terminal
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(log_level) # Set level for console logging

# Add handlers to Flask's logger
app.logger.addHandler(file_handler)
app.logger.addHandler(stream_handler)
app.logger.setLevel(log_level) # Set the logger's level
# app.logger.propagate = False # Prevent sending messages to root logger if needed

# Use the configured logger for initial messages
app.logger.info('Flask app logger configured with file and stream handlers.')
# --- END NEW Logging Configuration ---


# --- Configure CORS ---
# This block can now safely use app.logger because it was configured above
# Assumes FRONTEND_URL variable was loaded from os.environ earlier in the script
if FRONTEND_URL:
    # Allow requests only from the specified frontend URL
    CORS(app, origins=[FRONTEND_URL], supports_credentials=True)
    app.logger.info(f"CORS configured for origin: {FRONTEND_URL}") # Use app.logger
else:
    # More permissive for local development IF FRONTEND_URL is not set
    # WARNING: Do NOT use this permissive setting in production!
    CORS(app) # Allows all origins - ONLY for local testing without FRONTEND_URL set
    app.logger.warning("CORS allows all origins (FRONTEND_URL not set - OK for local dev ONLY)") # Use app.logger

# --- Configure Database ---
if not DATABASE_URL:
    # Fallback to SQLite ONLY if DATABASE_URL is not provided (e.g., for local dev)
    # Production deployments SHOULD provide DATABASE_URL.
    logger.warning("DATABASE_URL not set, falling back to local SQLite database (instance/chats.db).")
    basedir = os.path.abspath(os.path.dirname(__file__))
    instance_path = os.path.join(basedir, 'instance')
    os.makedirs(instance_path, exist_ok=True) # Ensure instance folder exists
    sqlite_path = os.path.join(instance_path, "chats.db")
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{sqlite_path}'
else:
    # Use the external database URL (e.g., PostgreSQL from Render/Neon/Supabase)
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # Disable tracking modifications

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# --- Database Models (ChatSession, ChatMessage - Unchanged) ---
class ChatSession(db.Model):
    id = db.Column(db.Integer, primary_key=True); session_uuid = db.Column(db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4())); created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)); messages = db.relationship('ChatMessage', backref='chat_session', lazy=True, cascade="all, delete-orphan") # type: ignore
class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True); chat_session_id = db.Column(db.Integer, db.ForeignKey('chat_session.id'), nullable=False); role = db.Column(db.String(10), nullable=False); content = db.Column(db.Text, nullable=False); timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)) # type: ignore

# --- Create Database Tables ---
# Needs application context. Best practice is often to use Flask-Migrate
# or run this once manually or at startup carefully.
with app.app_context():
    try:
        logger.info("Attempting to create database tables if they don't exist...")
        db.create_all()
        logger.info("Database tables checked/created.")
    except Exception as e:
        # Log error but don't necessarily crash if tables already exist or DB isn't reachable yet
        logger.error(f"Database table creation/check failed (maybe DB not ready or tables exist): {e}", exc_info=False)


# --- Initialize API Models ---
try:
    # Ensure Gemini model initializes correctly after config
    text_model = genai.GenerativeModel('gemini-pro')
    logger.info("Gemini text model initialized.")
except Exception as e:
    logger.critical(f"FATAL: Failed to initialize Gemini model: {e}", exc_info=True)
    # Consider exiting if the core model fails to load
    exit()


# --- Helper Functions (format_history_for_gemini, generate_image_huggingface, format_sse - Unchanged) ---
def format_history_for_gemini(messages): # (Keep as before)
    history = []
    for msg in messages:
        role = 'model' if msg.role == 'assistant' else msg.role
        history.append({"role": role, "parts": [{"text": msg.content}]})
    return history

def generate_image_huggingface(prompt: str) -> bytes: # (Keep as before)
    if not hugging_face_available: raise RuntimeError("Hugging Face API key not configured.")
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
    payload = {"inputs": prompt}
    logger.info(f"Requesting image from HF ({HF_IMAGE_MODEL_ID}) for prompt: {prompt[:50]}...")
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120) # Consider making timeout configurable
        response.raise_for_status()
        if 'image' in response.headers.get('content-type', ''):
            logger.info("Image received from Hugging Face.")
            return response.content
        else:
            try: error_detail = response.json().get("error", "Unknown non-image response")
            except json.JSONDecodeError: error_detail = response.text[:200]
            logger.error(f"HF API returned non-image data: {error_detail}")
            raise ValueError(f"Hugging Face API returned non-image data: {error_detail}")
    except requests.exceptions.Timeout: logger.error("HF API request timed out."); raise TimeoutError("Image generation timed out.")
    except requests.exceptions.HTTPError as http_err:
        status_code = http_err.response.status_code; error_text = http_err.response.text[:200]
        logger.error(f"HF API HTTP Error: {status_code} - {error_text}")
        if status_code == 429: raise ConnectionAbortedError("Rate limit exceeded.")
        if status_code == 503: raise ConnectionRefusedError(f"Model loading or unavailable ({error_text}).")
        raise ConnectionError(f"HF API Error ({status_code}): {error_text}")
    except requests.exceptions.RequestException as req_err: logger.error(f"HF API Request Error: {req_err}", exc_info=True); raise ConnectionError(f"Network error contacting HF: {req_err}")
    except Exception as e: logger.error(f"Unexpected error in image generation: {e}", exc_info=True); raise e

def format_sse(data: dict, event: str = None) -> str: # (Keep as before)
    payload = json.dumps(data); return f"event: {event}\ndata: {payload}\n\n" if event else f"data: {payload}\n\n"

# --- Centralized Error Handling (Unchanged) ---
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    status_code = 500; response = {'error': 'An internal server error occurred.', 'details': str(e)}
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException): status_code = e.code or 500; response['error'] = e.description or response['error']
    if isinstance(e, SQLAlchemyError): response['error'] = "Database operation failed."
    # Only show details in debug mode
    is_debug = os.environ.get('FLASK_ENV') == 'development' or os.environ.get('FLASK_DEBUG') == '1'
    if not is_debug: response.pop('details', None)
    return jsonify(response), status_code

# --- API Routes (Unchanged Logic, ensure logging and error handling are robust) ---

@app.route('/') # Health check / Index route
def index():
    # Optionally add a database check here
    # try:
    #     db.session.execute(db.text('SELECT 1'))
    #     db_status = "OK"
    # except Exception as e:
    #     logger.error(f"Health check DB connection error: {e}")
    #     db_status = "Error"
    return jsonify({
        "status": "OK",
        "message": "RHG AI Backend is running.",
        # "db_status": db_status # Optional
        })

@app.route('/api/chats', methods=['GET']) # List Chats
def list_chats():
    # (Keep logic as before, wrapped in try/except SQLAlchemyError)
    try:
        sessions = ChatSession.query.order_by(ChatSession.created_at.desc()).all()
        session_data = [{"session_id": s.session_uuid, "created_at": s.created_at.isoformat()} for s in sessions]
        return jsonify(session_data)
    except SQLAlchemyError as e: logger.error(f"DB Error listing chats: {e}", exc_info=True); return jsonify({"error": "Database error retrieving chat list"}), 500
    except Exception as e: logger.error(f"Unexpected error listing chats: {e}", exc_info=True); raise e
    
@app.route('/api/chat', methods=['POST'])
def initiate_chat():
    # Ensure 'request', 'jsonify', 'uuid', 'logger' are imported at the top
    logger.info(f"Received POST request for /api/chat from IP: {request.remote_addr} (Simplified Test)")
    try:
        # Bypassing all DB/API logic for this test
        temp_session_id = str(uuid.uuid4()) # Generate dummy session ID
        logger.info("Simplified route acknowledging request and returning dummy data.")
        # Return a success-like response immediately
        # Ensure frontend can handle this response structure (it expects stream_url/session_id)
        return jsonify({
            "stream_url": f'/api/stream/{temp_session_id}', # Keep structure
            "session_id": temp_session_id,
            "message": "Simplified POST OK" # Extra field for clarity
            })
    except Exception as e:
        # Log any unexpected error even in this simplified version
        logger.error(f"Error in simplified initiate_chat: {e}", exc_info=True)
        return jsonify({"error": "Simplified route failed"}), 500

@app.route('/api/chats/<session_uuid>', methods=['GET']) # Get Chat History
def get_chat_history(session_uuid):
     # (Keep logic as before, wrapped in try/except SQLAlchemyError)
     try:
        session = ChatSession.query.filter_by(session_uuid=session_uuid).first_or_404("Chat session not found")
        messages = ChatMessage.query.filter_by(chat_session_id=session.id).order_by(ChatMessage.timestamp.asc()).all()
        message_data = [{"role": msg.role, "content": msg.content, "timestamp": msg.timestamp.isoformat()} for msg in messages]
        return jsonify({"session_id": session.session_uuid, "messages": message_data})
     except SQLAlchemyError as e: logger.error(f"DB Error retrieving history for {session_uuid}: {e}", exc_info=True); return jsonify({"error": "Database error retrieving chat history"}), 500
     except Exception as e: logger.error(f"Unexpected error getting history for {session_uuid}: {e}", exc_info=True); raise e

@app.route('/api/stream/<session_uuid>', methods=['GET']) # SSE Stream for AI Response
def stream_chat(session_uuid):
    # (Keep logic as before, including error handling within the generator)
    logger.info(f"SSE connection opened for session: {session_uuid}")
    def generate_response():
        session = None; full_ai_response_content = ""; task_type = "text"
        try:
            try: # DB lookup block
                 session = ChatSession.query.filter_by(session_uuid=session_uuid).first();
                 if not session: logger.warning(f"Stream requested for non-existent session: {session_uuid}"); yield format_sse({'error': 'Chat session not found'}, event='error'); return
                 last_user_message = ChatMessage.query.filter_by(chat_session_id=session.id, role='user').order_by(ChatMessage.timestamp.desc()).first()
                 if not last_user_message: logger.warning(f"Stream req but no user prompt found for {session_uuid}"); yield format_sse({'error': 'User prompt not found'}, event='error'); return
            except SQLAlchemyError as e: logger.error(f"DB error loading session/prompt for {session_uuid}: {e}", exc_info=True); yield format_sse({'error': 'Database error loading chat state'}, event='error'); return

            user_prompt_text = last_user_message.content.strip(); image_prompt = None
            if hugging_face_available: # Check image trigger
                trigger_phrases = ["generate image:", "draw:", "create image:"]
                for phrase in trigger_phrases:
                    if user_prompt_text.lower().startswith(phrase):
                        image_prompt = user_prompt_text[len(phrase):].strip()
                        if not image_prompt: yield format_sse({'error': 'Image prompt cannot be empty'}, event='error'); return
                        task_type = "image"; break

            if task_type == "image": # Image Generation Task
                yield format_sse({'message': f'Requesting image ({HF_IMAGE_MODEL_ID})...'}, event='status')
                try:
                    image_bytes = generate_image_huggingface(image_prompt)
                    base64_image = base64.b64encode(image_bytes).decode('utf-8'); mime_type = "image/png"
                    image_data_url = f"data:{mime_type};base64,{base64_image}"
                    yield format_sse({'base64': image_data_url, 'prompt': image_prompt}, event='image')
                    full_ai_response_content = f"[Image generated for prompt: {image_prompt}]"
                except (TimeoutError, ConnectionError, ConnectionAbortedError, ConnectionRefusedError, ValueError, RuntimeError) as img_e:
                    logger.error(f"Image generation failed for {session_uuid}: {img_e}")
                    yield format_sse({'error': f'Image generation failed: {img_e}'}, event='error')
                except Exception as img_e: logger.error(f"Unexpected image gen error for {session_uuid}: {img_e}", exc_info=True); yield format_sse({'error': 'Unexpected error during image generation.'}, event='error')

            else: # Text Generation Task
                try:
                    messages = ChatMessage.query.filter_by(chat_session_id=session.id).order_by(ChatMessage.timestamp.asc()).all()
                    system_instruction = "You are RHG AI..." # System prompt
                    gemini_history = format_history_for_gemini(messages)
                    prompt_parts = [system_instruction] + gemini_history
                    logger.info(f"Calling Gemini for session {session_uuid}")
                    response_stream = text_model.generate_content(prompt_parts, stream=True)
                    for chunk in response_stream:
                        try: # Process chunk
                            if hasattr(chunk, 'text') and chunk.text: full_ai_response_content += chunk.text; yield format_sse({"text": chunk.text})
                        except ValueError: logger.warning(f"Gemini stream encountered empty/non-text part for {session_uuid}", exc_info=False); continue
                except Exception as gemini_e: logger.error(f"Gemini API error for {session_uuid}: {gemini_e}", exc_info=True); error_detail = str(gemini_e); yield format_sse({'error': f'AI text generation failed: {error_detail}'}, event='error')

        except Exception as e: # Catch-all for generator
            logger.error(f"Unexpected error during stream gen for {session_uuid}: {e}", exc_info=True)
            yield format_sse({'error': 'Unexpected server error during generation.'}, event='error')
        finally: # Save assistant response
            if session and full_ai_response_content:
                try:
                    ai_msg = ChatMessage(chat_session_id=session.id, role='assistant', content=full_ai_response_content)
                    db.session.add(ai_msg); db.session.commit()
                    logger.info(f"Saved assistant response ({task_type}) for session: {session_uuid}")
                except SQLAlchemyError as db_e: logger.error(f"DB Error saving assistant msg for {session_uuid}: {db_e}", exc_info=True); db.session.rollback()
            logger.info(f"SSE connection closing for session: {session_uuid}")
    return Response(stream_with_context(generate_response()), mimetype='text/event-stream')


# --- Run the App (for Development) ---
# Production should use Gunicorn as specified in Dockerfile or Procfile
if __name__ == '__main__':
    # Use environment variables for host, port, debug is better practice
    host = os.environ.get('FLASK_RUN_HOST', '127.0.0.1') # Default to localhost
    try:
       port = int(os.environ.get('FLASK_RUN_PORT', '5000')) # Default to 5000
    except ValueError:
       port = 5000
    # FLASK_DEBUG=1 activates debug mode (auto-reload, debugger)
    # Use FLASK_ENV=development for development mode without debugger if preferred
    is_debug = os.environ.get('FLASK_DEBUG') == '1' or os.environ.get('FLASK_ENV') == 'development'
    logger.info(f"Starting Flask dev server on {host}:{port} (Debug: {is_debug})")
    app.run(debug=is_debug, port=port, host='0.0.0.0')
