import asyncio, difflib #multiprocessser good for sockets
import websockets , json # the json is bc we're sending a json object from the website
import wave , base64
import os #library for anything that has to do with your hard-drive
import torch, sqlite3
import librosa #library to analyse and process audio . soundfile is similar 
import os #library for anything that has to do with your hard-drive
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import ssl  # Add this import at the top
import secrets  # Add secrets to generate tokens
from datetime import datetime, timedelta
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import mimetypes
from PIL import Image, ImageDraw, ImageFont


LANG_ID = "fr"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
AUDIO_DIR = "frenchtrial.wav"

# Add these global variables at the top level after the imports
loaded_processors = {}
loaded_models = {}

# Add these database constants after other constants
DB_PATH = "users.db"

# Add these constants after other constants
EPUB_DIR = Path("epub")
COVERS_DIR = Path("covers")
DEFAULT_COVER = "default-cover.png"

db = None

language_dict = {
    #"en": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    #"es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    #"it": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    #"de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
}

def get_or_load_model(lang):
    """Helper function to get or load model and processor"""
    MODEL_ID = language_dict[lang]
    
    if lang not in loaded_processors:
        loaded_processors[lang] = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    if lang not in loaded_models:
        loaded_models[lang] = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
        
    return loaded_processors[lang], loaded_models[lang]

def stt(AUDIO_DIR, lang):
    # Get or load the model and processor
    processor, model = get_or_load_model(lang)
    
    # Pre-processing the data
    audio = librosa.load(AUDIO_DIR, sr=16_000)

    # Tokenizing the data
    inputs = processor(audio[0], sampling_rate=audio[1], return_tensors="pt")

    # Inference process
    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentences = processor.batch_decode(predicted_ids)
    
    print("Prediction:", predicted_sentences)
    return predicted_sentences[0]

class TextComparator:
    @staticmethod
    def generate_html_report(text1, text2, output_file='text_comparison_report.html'):
        words1 = text1.lower().split()
        words2 = text2.lower().split()
        matcher = difflib.SequenceMatcher(None, words1, words2)       
        marked_words2 = words2.copy()        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            print((tag,i1, i2, j1, j2))
            if tag != 'equal':
                for j in range(j1, j2):
                    # Set id to the corresponding word in words1 if it exists, otherwise use empty string
                    corresponding_word = words1[i1] if i1 < len(words1) else ""
                    marked_words2[j] = f'<span id="{corresponding_word}" class="wrong" style="color:red;">{marked_words2[j]}</span>'
        
        marked_text2 = ' '.join(marked_words2)
        
        similarity_ratio = matcher.ratio()
        
       
        return marked_text2, similarity_ratio

def stt_task(data_object):
    print(f"language: {data_object['language']}") 

    fpath="received_audio.wav" #name of the audio in the server (our computer in this case)
    with open(fpath, "wb") as audio_file: #received_audio is f.open, initialising this file. audio_file is the variable
            base64_string = data_object["blob"]
            actual_base64 = base64_string.split(',')[1]  # Get the Base64 part
            binary_data = base64.b64decode(actual_base64)
            audio_file.write(binary_data)
    the_words = stt(fpath,data_object["language"])
    predicted_sentence, similarity_ratio = TextComparator.generate_html_report(data_object["sentence"], the_words)
    message_returned = {"pred_sentence":predicted_sentence}

    print("Received audio file and saved as 'received_audio.wav'")
    return message_returned

def extract_epub_cover(epub_path: Path, cover_dir: Path) -> str:
    """Extract cover image from EPUB file using the same approach as the PHP code"""
    try:
        with zipfile.ZipFile(epub_path) as zip_file:
            # First try META-INF/container.xml
            try:
                container = ET.fromstring(zip_file.read('META-INF/container.xml'))
                rootfile_path = container.find('.//{urn:oasis:names:tc:opendocol:xmlns:container}rootfile').get('full-path')
                
                # Read content.opf
                opf = ET.fromstring(zip_file.read(rootfile_path))
                
                # Look for cover image in manifest
                manifest = opf.find('.//{*}manifest')
                if manifest is not None:
                    # First try items with id containing 'cover'
                    for item in manifest.findall('.//{*}item'):
                        if 'cover' in item.get('id', '').lower():
                            href = item.get('href')
                            if href:
                                # Handle relative paths
                                if not href.startswith('/'):
                                    opf_dir = Path(rootfile_path).parent
                                    href = str(opf_dir / href)
                                
                                # Extract and save the cover
                                try:
                                    image_data = zip_file.read(href.lstrip('/'))
                                    cover_dir.mkdir(parents=True, exist_ok=True)
                                    output_path = cover_dir / f"{epub_path.stem}.jpg"
                                    output_path.write_bytes(image_data)
                                    return str(output_path.relative_to(COVERS_DIR.parent))
                                except Exception as e:
                                    print(f"Error saving cover image: {e}")
                                    continue
                    
                    # If no cover found, try first image in manifest
                    for item in manifest.findall('.//{*}item'):
                        media_type = item.get('media-type', '')
                        if media_type.startswith('image/'):
                            href = item.get('href')
                            if href:
                                # Handle relative paths
                                if not href.startswith('/'):
                                    opf_dir = Path(rootfile_path).parent
                                    href = str(opf_dir / href)
                                
                                # Extract and save the cover
                                try:
                                    image_data = zip_file.read(href.lstrip('/'))
                                    cover_dir.mkdir(parents=True, exist_ok=True)
                                    output_path = cover_dir / f"{epub_path.stem}.jpg"
                                    output_path.write_bytes(image_data)
                                    return str(output_path.relative_to(COVERS_DIR.parent))
                                except Exception as e:
                                    print(f"Error saving first image as cover: {e}")
                                    continue
            
            except Exception as e:
                print(f"Error reading EPUB structure: {e}")
                # Try direct image search as fallback
                for filename in zip_file.namelist():
                    if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                        try:
                            image_data = zip_file.read(filename)
                            cover_dir.mkdir(parents=True, exist_ok=True)
                            output_path = cover_dir / f"{epub_path.stem}.jpg"
                            output_path.write_bytes(image_data)
                            return str(output_path.relative_to(COVERS_DIR.parent))
                        except Exception as e:
                            print(f"Error saving fallback image: {e}")
                            continue
    
    except Exception as e:
        print(f"Error extracting cover from {epub_path}: {e}")
    
    print(f"No cover found for {epub_path.name}, using default")
    return DEFAULT_COVER

def get_file_as_base64(file_path: Path) -> str:
    """Convert a file to base64 string with proper mime type prefix"""
    try:
        mime_type = mimetypes.guess_type(str(file_path))[0]
        with open(file_path, 'rb') as file:
            b64_data = base64.b64encode(file.read()).decode('utf-8')
            return f"data:{mime_type};base64,{b64_data}"
    except Exception as e:
        print(f"Error converting file to base64: {e}")
        return ""

async def get_available_books():
    """Get list of available EPUB books and their covers"""
    books = []
    
    try:
        # Make sure default cover exists
        default_cover_path = Path("images/default-cover.png")
        if not default_cover_path.exists():
            print(f"Creating default cover at {default_cover_path}")
            img = Image.new('RGB', (120, 180), color='lightgray')
            d = ImageDraw.Draw(img)
            d.text((10,10), "No\nCover", fill='black')
            default_cover_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(default_cover_path)
            print("Default cover created successfully")
        
        # Process books
        for lang_dir in EPUB_DIR.glob("*"):
            if lang_dir.is_dir():
                language = lang_dir.name
                print(f"\nProcessing language directory: {language}")
                
                for epub_path in lang_dir.glob("*.epub"):
                    print(f"\nProcessing book: {epub_path.name}")
                    cover_dir = COVERS_DIR / language
                    cover_path = extract_epub_cover(epub_path, cover_dir)
                    
                    try:
                        if cover_path == DEFAULT_COVER:
                            cover_file = default_cover_path
                        else:
                            cover_file = COVERS_DIR.parent / cover_path
                        
                        print(f"Converting files to base64 for {epub_path.name}")
                        cover_base64 = get_file_as_base64(cover_file)
                        epub_base64 = get_file_as_base64(epub_path)
                        
                        books.append({
                            "filename": epub_path.name,
                            "language": language,
                            "path": str(epub_path.relative_to(EPUB_DIR.parent)),
                            "cover": cover_base64,
                            "epub": epub_base64
                        })
                        print(f"Successfully processed book: {epub_path.name}")
                    except Exception as e:
                        print(f"Error processing book {epub_path.name}: {e}")
                        continue
        
        return books
    except Exception as e:
        print(f"Error getting available books: {e}")
        return []

async def handle_connection(websocket):
    print(f"Client connected, {websocket.remote_address}")
    
    try:
        data = await websocket.recv()
        data_object = json.loads(data)
        
        message_returned = {"error": "Invalid task"}  # Default message
        
        if data_object["task"] == "get_books":
            # Handle book listing request
            books = await get_available_books()
            message_returned = {"books": books}
        elif data_object["task"] == "stt":
            message_returned = stt_task(data_object)
        elif data_object["task"] == "login":
            message_returned = db.login_task(data_object)
        elif data_object["task"] == "signup":
            message_returned = db.signup_task(data_object)
        elif data_object["task"] == "change_settings":
            message_returned = db.change_settings_task(data_object)
            
        await websocket.send(json.dumps(message_returned))
    
    except Exception as e:
        print(f"Error: {e}")
        await websocket.send(json.dumps({"error": str(e)}))
    finally:
        print("Client disconnected")

# Add a function to preload all models
def preload_all_models():
    """Preload all language models at startup"""
    print("Preloading all language models...")
    for lang in language_dict:
        print(f"Loading {lang} model...")
        get_or_load_model(lang)
    print("All models loaded!")
class DatabaseManager:
    def __init__(self, db_path):
        self.DB_PATH = db_path #a constructor defines what parameters are needed to create an object

    def init_database(self):
        """Initialize the SQLite database with necessary tables"""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users ( 
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            current_book TEXT DEFAULT '',
            book_position INTEGER DEFAULT 0,
            preferred_language TEXT DEFAULT 'en',
            login_token TEXT,
            token_expiry TIMESTAMP
        )
        ''')
        #the table is named users
        conn.commit()
        conn.close()

    async def login_task(self, data_object):
        """Handle user login"""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        
        username = data_object.get("username")
        password = data_object.get("password")
        
        cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", 
                    (username, password))
        user = cursor.fetchone()
        
        if user:
            # Generate new token and set expiry to 24 hours from now
            token = secrets.token_urlsafe(32)
            expiry = datetime.now() + timedelta(days=1)
            
            cursor.execute("""
                UPDATE users 
                SET login_token = ?, token_expiry = ? 
                WHERE username = ?""", 
                (token, expiry, username))
            conn.commit()
            
            message = {
                "status": "success",
                "token": token,
                "username": username,
                "current_book": user[2],
                "book_position": user[3],
                "preferred_language": user[4],
                "type": "login"
            }
        else:
            message = {
                "status": "error",
                "message": "Invalid credentials"
            }
        
        conn.close()
        return message
        

    def signup_task(self, data_object):
        """Handle user registration"""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        
        username = data_object.get("username")
        password = data_object.get("password")
        
        # Check if username already exists
        cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            conn.close()
            return {"status": "error", "message": "Username already exists"}
        
        # Create new user
        cursor.execute("""
            INSERT INTO users (username, password) 
            VALUES (?, ?)""", 
            (username, password))
        conn.commit()
        conn.close()
        
        return {"status": "success", "message": "User created successfully"}

    def change_settings_task(self, data_object):
        """Handle user settings updates"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        token = data_object.get("token")
        updates = {}
        
        if "current_book" in data_object:
            updates["current_book"] = data_object["current_book"]
        if "book_position" in data_object:
            updates["book_position"] = data_object["book_position"]
        if "preferred_language" in data_object:
            updates["preferred_language"] = data_object["preferred_language"]
        
        if not updates:
            return {"status": "error", "message": "No settings to update"}
        
        # Verify token and update settings
        cursor.execute("SELECT username FROM users WHERE login_token = ? AND token_expiry > ?", 
                    (token, datetime.now()))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return {"status": "error", "message": "Invalid or expired token"}
        
        # Build update query
        update_query = "UPDATE users SET " + ", ".join(f"{k} = ?" for k in updates.keys())
        update_query += " WHERE login_token = ?"
        
        # Execute update
        cursor.execute(update_query, list(updates.values()) + [token])
        conn.commit()
        conn.close()
        
        return {"status": "success", "message": "Settings updated successfully"}


async def main():

    global db 
    db = DatabaseManager("usersHablas.db")
    db.init_database();
    preload_all_models()
    print("Preloaded all models")

    async with websockets.serve(handle_connection, "localhost", 8765) as server:
        print("WebSocket server started on local NOT wss://carriertech.uk:8765")
        await asyncio.Future()  # Run forever 

if __name__ == "__main__": #when you use a multi processer load, you use this so it doesnt crash. with asyncio you always have to do it
    print("we're initializing where tf is the error??")
    asyncio.run(main()) 



