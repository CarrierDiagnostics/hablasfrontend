import asyncio, difflib, time #multiprocessser good for sockets

import websockets , json # the json is bc we're sending a json object from the website
import wave , base64
import os #library for anything that has to do with your hard-drive
import torch, sqlite3
from gtts import gTTS  # Add this import for Google Text-to-Speech

import librosa #library to analyse and process audio . soundfile is similar
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
from googletrans import Translator  # Add this import for translation
import Levenshtein  # Add this import for Levenshtein distance
from googletrans import Translator  # Add this import for translation
import uuid  # Add this import for UUID generation
import numpy as np
import soundfile as sf
import torch.cuda
from torch.cuda.amp import autocast
from torch.nn.parallel import DataParallel
import queue
from concurrent.futures import ThreadPoolExecutor
import threading
import ipaddress


LANG_ID = "fr"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
AUDIO_DIR = "frenchtrial.wav"

loaded_processors = {}
loaded_models = {}
translator = Translator()

EPUB_DIR = Path("epub")
COVERS_DIR = Path("covers")
DEFAULT_COVER = "default-cover.png"
RECORDINGS_DIR = Path("recordings")


db = None

language_dict = {
    "en": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "it": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
}

# Add this language mapping dictionary at the top level with other dictionaries
language_name_map = {
    "francais": "fr",
    "english": "en",
    "español": "es",
    "espagnol": "es",
    "spanish": "es",
    "deutsch": "de",
    "italiano": "it"
}

class ModelManager:
    def __init__(self):
        self.loaded_processors = {}
        self.loaded_models = {}
        self.batch_queue = queue.Queue()
        self.batch_size = 8  # Adjust based on your GPU memory
        self.processing_thread = threading.Thread(target=self._process_batch, daemon=True)
        self.processing_thread.start()
        self.lock = threading.Lock()

    def get_or_load_model(self, lang):
        """Thread-safe model loading with GPU optimization"""
        with self.lock:
            if lang not in self.loaded_processors:
                MODEL_ID = language_dict[lang]
                self.loaded_processors[lang] = Wav2Vec2Processor.from_pretrained(MODEL_ID)
                model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

                # Enable GPU parallel processing if multiple GPUs are available
                if torch.cuda.device_count() > 1:
                    model = DataParallel(model)
                model = model.to('cuda')
                model.eval()  # Set to evaluation mode
                self.loaded_models[lang] = model

            return self.loaded_processors[lang], self.loaded_models[lang]

    def _process_batch(self):
        """Process batches of audio in background"""
        while True:
            batch = []
            try:
                # Collect batch_size items or wait for timeout
                while len(batch) < self.batch_size:
                    try:
                        item = self.batch_queue.get(timeout=0.1)
                        batch.append(item)
                    except queue.Empty:
                        if batch:  # Process partial batch
                            break
                        continue

                if batch:
                    self._process_items(batch)

            except Exception as e:
                print(f"Batch processing error: {e}")

    def _process_items(self, batch):
        """Process a batch of audio files"""
        try:
            # Group by language
            by_language = {}
            for item in batch:
                lang = item['lang']
                if lang not in by_language:
                    by_language[lang] = []
                by_language[lang].append(item)

            # Process each language batch
            for lang, items in by_language.items():
                processor, model = self.get_or_load_model(lang)

                # Prepare batch inputs
                audio_inputs = []
                attention_masks = []

                for item in items:
                    audio = librosa.load(item['audio_path'], sr=16_000)[0]
                    inputs = processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
                    audio_inputs.append(inputs.input_values)
                    attention_masks.append(inputs.attention_mask)

                # Stack batch inputs
                batched_inputs = torch.cat(audio_inputs).to('cuda')
                batched_attention_mask = torch.cat(attention_masks).to('cuda')

                # Run inference with automatic mixed precision
                with autocast():
                    with torch.no_grad():
                        logits = model(batched_inputs, attention_mask=batched_attention_mask).logits

                # Process results
                predicted_ids = torch.argmax(logits, dim=-1)
                predicted_sentences = processor.batch_decode(predicted_ids)

                # Update results
                for item, sentence in zip(items, predicted_sentences):
                    item['future'].set_result(sentence)

        except Exception as e:
            print(f"Error processing batch: {e}")
            # Set error result for all items in batch
            for item in batch:
                item['future'].set_exception(e)

# Initialize the model manager
#model_manager = ModelManager()



def stt_task(data_object):
    try:
        # Map language name to code if needed
        lang = data_object['language'].lower()
        if lang in language_name_map:
            lang = language_name_map[lang]

        print(f"language code: {lang}")

        # Save audio file
        user_id = data_object.get('username', None)
        if not user_id:
            # Get IP from websocket connection info stored in data_object
            user_id = f"ip_{data_object.get('ip', 'unknown')}"
        
        book_name = data_object.get('book', '').replace('.epub', '')  # Remove .epub extension
        
        # Create user and book specific directory
        user_dir = RECORDINGS_DIR / user_id
        book_dir = user_dir / book_name if book_name else user_dir / "general"
        book_dir.mkdir(parents=True, exist_ok=True)

        # Save audio file with timestamp in the appropriate directory
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        fpath = book_dir / f"recording_{timestamp}.wav"
        with open(fpath, "wb") as audio_file:
            base64_string = data_object["blob"]
            actual_base64 = base64_string.split(',')[1]
            binary_data = base64.b64decode(actual_base64)
            audio_file.write(binary_data)

        # Process audio synchronously instead of using asyncio
        processor, model = model_manager.get_or_load_model(lang)

        # Load and process audio
        audio = librosa.load(fpath, sr=16_000)[0]
        inputs = processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)

        # Move inputs to GPU
        input_values = inputs.input_values.to('cuda')
        attention_mask = inputs.attention_mask.to('cuda')

        # Run inference with automatic mixed precision
        with autocast():
            with torch.no_grad():
                logits = model(input_values, attention_mask=attention_mask).logits

        # Process results
        predicted_ids = torch.argmax(logits, dim=-1)
        the_words = processor.batch_decode(predicted_ids)[0]  # Take first result since we're processing single audio

        # Rest of the processing remains the same
        predicted_sentence, similarity_ratio, original_words, spoken_words = TextComparator.generate_html_report(
            data_object["sentence"], the_words)

        total_words = len(spoken_words)
        wrong_words = predicted_sentence.count('class="wrong"')
        points = total_words - wrong_words

        message_returned = {
            "pred_sentence": predicted_sentence,
            "points": points,
            "total_words": total_words,
            "wrong_words": wrong_words,
        }

        db.set_current_book_task(data_object, fpath, predicted_sentence)

        return message_returned

    except Exception as e:
        print(f"Error in stt_task: {e}")
        return {"error": str(e)}

def extract_epub_cover(epub_path: Path, cover_dir: Path) -> str:
    """Extract cover image from EPUB file using the same approach as the PHP code"""
    try:
        with zipfile.ZipFile(epub_path) as zip_file:
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

                        print(f"Converting cover to base64 for {epub_path.name}")
                        cover_base64 = get_file_as_base64(cover_file)

                        books.append({
                            "filename": epub_path.name,
                            "language": language,
                            "path": str(epub_path.relative_to(EPUB_DIR.parent)),
                            "cover": cover_base64
                            # Removed the epub base64 data to make the response lighter
                        })
                        print(f"Successfully processed book: {epub_path.name}")
                    except Exception as e:
                        print(f"Error processing book {epub_path.name}: {e}")
                        continue

        return books
    except Exception as e:
        print(f"Error getting available books: {e}")
        return []

async def get_book_data(data_object):
    """Get specific book data by filename and language"""
    try:
        filename = data_object.get("filename")

        if not filename :
            return {"status": "error", "message": "Missing filename or language"}

        epubs = os.walk(EPUB_DIR)
        for root, dirs, files in epubs:
            for file in files:
                if file == filename:
                    epub_path = Path(root) / file
                    break
        print(epub_path)
        language = epub_path.parent.name
        print(f"language: {language}")
        if not epub_path.exists():
            return {"status": "error", "message": f"Book not found: {filename}"}

        # Convert the EPUB file to base64
        epub_base64 = get_file_as_base64(epub_path)

        return {
            "status": "success",
            "filename": filename,
            "language": language,
            "epub": epub_base64
        }
    except Exception as e:
        print(f"Error getting book data: {e}")
        return {"status": "error", "message": str(e)}

async def translate_task(data_object):
    source_text = data_object.get("text", "")
    source_lang = data_object.get("source_lang", "en")
    target_lang = data_object.get("target_lang", "fr")
    current_book = data_object.get("current_book", "")
    cfi = data_object.get("cfi", "")  # Get CFI instead of page
    username = data_object.get("username", "")

    if not source_text:
        return {"status": "error", "message": "No text provided for translation"}

    if source_lang == target_lang:
        return {"status": "success", "translated_text": source_text}

    # Map language codes to standard codes if needed
    source_lang_map = {
        "francais": "fr",
        "english": "en",
        "español": "es",
        "espagnol": "es",
        "spanish": "es",
        "deutsch": "de",
        "italiano": "it"
    }

    if source_lang in source_lang_map:
        source_lang = source_lang_map[source_lang]

    try:
        print(f"Attempting to translate from {source_lang} to {target_lang}")
        translator_instance = Translator()
        result = await translator_instance.translate(source_text, src=source_lang, dest=target_lang)

        if hasattr(result, 'text'):
            translated_text = result.text
            print(f"Translation successful: {translated_text}")
            print(f"current_book: {current_book}")
            print(f"username: {username}")
            if current_book and username:
                print(f"Updating database for user {username} with book {current_book} and CFI {cfi}")
                db.set_current_book_task({
                    "username": username,
                    "book": current_book,
                    "cfi": cfi,  # Update to use CFI instead of page
                }, "", "")

            return {
                "status": "success",
                "translated_text": translated_text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "current_book": current_book,
                "cfi": cfi  # Return CFI instead of page
            }
        else:
            print(f"Translation result has no 'text' attribute: {result}")
            return {
                "status": "error",
                "message": "Translation result format unexpected",
                "source_lang": source_lang,
                "target_lang": target_lang
            }
    except Exception as e:
        print(f"Translation error: {e}")
        return {
            "status": "error",
            "message": str(e),
            "source_lang": source_lang,
            "target_lang": target_lang
        }

async def login_task(self, data_object):
    conn = sqlite3.connect(self.DB_PATH)
    cursor = conn.cursor()

    username = data_object.get("username")
    password = data_object.get("password")

    # Get column names
    cursor.execute("PRAGMA table_info(users)")
    columns = [column[1] for column in cursor.fetchall()]

    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?",
                (username, password))
    user_row = cursor.fetchone()

    if user_row:
        # Convert row to dictionary
        user = dict(zip(columns, user_row))
        print(user)

        # Generate new token using UUID and set expiry to 24 hours from now
        token = str(uuid.uuid4())
        expiry = datetime.now() + timedelta(days=10000)

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
            "current_book": user.get("current_book", ""),
            "cfi": user.get("cfi", ""),  # Use CFI instead of page
            "preferred_language": user.get("preferred_language", "en"),
            "type": "login"
        }

        # Only add epub and language if there's a current book
        if user.get("current_book"):
            print("getting books")
            epub = None
            language = None
            epubs = os.walk(EPUB_DIR)
            for root, dirs, files in epubs:
                for file in files:
                    if file == user["current_book"]:
                        book_path = Path(root) / file
                        break

            if epub and language:
                message["epub"] = epub
                message["language"] = language

    else:
        message = {
            "status": "error",
            "message": "Invalid credentials"
        }

    conn.close()
    return message

async def verify_token_task(data_object):
    """Verify a login token and return user data if valid"""
    conn = sqlite3.connect(db.DB_PATH)
    cursor = conn.cursor()

    token = data_object.get("token")
    if not token:
        return {"status": "error", "message": "No token provided"}

    # Get column names
    cursor.execute("PRAGMA table_info(users)")
    columns = [column[1] for column in cursor.fetchall()]
    print(columns)

    # Check if token exists and is not expired
    cursor.execute("""
        SELECT * FROM users
        WHERE login_token = ? AND token_expiry > ?
    """, (token, datetime.now()))

    user_row = cursor.fetchall()
    conn.close()

    if not user_row:
        return {"status": "error", "message": "Invalid or expired token"}

    user_row = user_row[0]
    # Convert row to dictionary
    user = dict(zip(columns, user_row))
    print("User data:", user)

    epub = None
    language = None

    print(f"user.get('current_book'): {user.get('current_book')}")
    if user.get("current_book"):
        print("getting books")
        epubs = os.walk(EPUB_DIR)
        for root, dirs, files in epubs:
            for file in files:
                if file == user["current_book"]:
                    book_path = Path(root) / file
                    break
        print(f"book_path: {book_path}")
        if book_path.exists():
            language = book_path.parent.name
            print(f"language: {language}")
            with open(book_path, "rb") as f:
                book_data = f.read()
                book_base64 = base64.b64encode(book_data).decode('utf-8')
                epub = f"data:application/epub+zip;base64,{book_base64}"
                print(f"Including book data for {user['current_book']}")

    response = {
        "status": "success",
        "username": user.get("username", ""),
        "current_book": user.get("current_book", ""),
        "cfi": user.get("cfi", ""),
        "preferred_language": user.get("preferred_language", "en"),
        "token": token,
        "epub": epub,
        "language": language,
        "pagele": user.get("pagele", {})
    }
    return response

def pagele_task(data_object):
    """Get sentences from EPUB book for the Pagele game"""
    try:
        # Get user data
        username = data_object.get("username")
        book = data_object.get("book")
        cfi = data_object.get("cfi")

        if not all([username, book]):
            return {"status": "error", "message": "Missing required data"}

        # Find the book file
        book_path = None
        for root, _, files in os.walk(EPUB_DIR):
            if book in files:
                book_path = Path(root) / book
                break

        if not book_path or not book_path.exists():
            return {"status": "error", "message": "Book not found"}

        # Extract text from EPUB
        with zipfile.ZipFile(book_path) as zip_file:
            # Find the content files (usually XHTML/HTML files)
            content_files = [f for f in zip_file.namelist() if f.endswith(('.xhtml', '.html'))]
            
            all_text = []
            for content_file in content_files:
                content = zip_file.read(content_file).decode('utf-8')
                # Parse HTML and extract text
                root = ET.fromstring(content)
                # Remove script and style elements
                for elem in root.findall(".//script"):
                    elem.clear()
                for elem in root.findall(".//style"):
                    elem.clear()
                
                # Get text content
                text = "".join(root.itertext())
                # Split into sentences (basic implementation)
                sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
                all_text.extend(sentences)

            # Find current position using CFI if provided
            start_index = 0
            if cfi:
                # Simple implementation - just use it as an index
                try:
                    start_index = int(cfi)
                except ValueError:
                    start_index = 0

            # Get next 10 sentences
            selected_sentences = all_text[start_index:start_index + 10]
            next_cfi = str(start_index + len(selected_sentences))

            return {
                "status": "success",
                "sentences": selected_sentences,
                "next_cfi": next_cfi,
                "total_sentences": len(all_text)
            }

    except Exception as e:
        print(f"Error in pagele_task: {e}")
        return {"status": "error", "message": str(e)}

async def tts_task(data_object):
    """Convert text to speech using Google Text-to-Speech"""
    try:
        text = data_object.get("text", "")
        lang = data_object.get("language", "en")
        
        # Map language name to code if needed
        if lang.lower() in language_name_map:
            lang = language_name_map[lang.lower()]
            
        if not text:
            return {"status": "error", "message": "No text provided for TTS conversion"}
            
        # Generate a unique filename
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        audio_filename = f"tts_{timestamp}.mp3"
        audio_path = Path("tts_output") / audio_filename
        audio_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Generate speech using gTTS
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(str(audio_path))
        
        # Convert the audio file to base64
        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
        # Return the audio as base64 with appropriate MIME type
        return {
            "status": "success",
            "audio": f"data:audio/mp3;base64,{audio_base64}",
            "language": lang
        }
        
    except Exception as e:
        print(f"Error in tts_task: {e}")
        return {"status": "error", "message": str(e)}

async def init_pagele_task(data_object):
    print(f"init_pagele_task: {data_object}")
    token = data_object["token"]
    print(f"token: {token}")
    pagele_filename = data_object["pagele_filename"]
    print(f"pagele_filename: {pagele_filename}")
    language = data_object["language"]
    print(f"language: {language}")
    user_data = db.get_user_data(token)
    print(f"user_data: {user_data}")
    
    """if user_data["pagele"] == '{}':
        pagele_data = {}
    else:"""
    pagele_data = json.loads(user_data["pagele"])
        
    # Update user_data with the parsed dictionary
    user_data["pagele"] = pagele_data
    
    if not user_data["pagele"]:
        user_data["pagele"] = {"current_pagele": pagele_filename, "current_index": 0,
                                "books": {pagele_filename: {"completed_indices": {}, "total_points": 0, "last_played": ""}}}
    elif pagele_filename in user_data["pagele"]["books"].keys():
        user_data["pagele"]["current_pagele"] = pagele_filename
        user_data["pagele"]["current_index"] = user_data["pagele"]["books"][pagele_filename]["completed_indices"][-1] + 1
    else:
        user_data["pagele"]["current_pagele"] = pagele_filename
        user_data["pagele"]["current_index"] = 0
        user_data["pagele"]["books"][pagele_filename] = {"completed_indices": {}, "total_points": 0, "last_played": ""}


    pagele_data = open(f"{EPUB_DIR}/{language}/{pagele_filename}", "r")
    pagele_data = json.load(pagele_data)

    return {"status": "success", "pagele_data": pagele_data, "index": user_data["pagele"]["current_index"], "type": "get_pagele", "user_pagele": user_data["pagele"]}

async def get_available_pagele():
    pagele_books = []  # Initialize the list to store pagele data
    
    try:
        # Look for pagele JSON files in each language directory
        for lang_dir in EPUB_DIR.glob("*"):
            if lang_dir.is_dir():
                language = lang_dir.name
                print(f"\nProcessing pagele files for language: {language}")
                
                # Look for pagele JSON files
                for pagele_path in lang_dir.glob("*.json"):
                    print(f"\nProcessing pagele file: {pagele_path.name}")
                    
                    try:
                        # Load the pagele JSON file
                        
                        book_name = pagele_path.stem.replace('.json', '')
                        book_path = lang_dir / f"{book_name}.epub"
                        
                        # Get cover image if the book exists
                        cover_base64 = ""
                        if book_path.exists():
                            cover_dir = COVERS_DIR / language
                            cover_path = extract_epub_cover(book_path, cover_dir)
                            
                            if cover_path == DEFAULT_COVER:
                                cover_file = Path("images/default-cover.png")
                            else:
                                cover_file = COVERS_DIR.parent / cover_path
                                
                            cover_base64 = get_file_as_base64(cover_file)
                        
                        # Add pagele information to the list
                        pagele_books.append({
                            "filename": pagele_path.name,
                            "book_name": book_name,
                            "language": language,
                            "path": str(pagele_path.relative_to(EPUB_DIR.parent)),
                            "cover": cover_base64,
                        })
                        print(f"Successfully processed pagele file: {pagele_path.name}")
                    except Exception as e:
                        print(f"Error processing pagele file {pagele_path.name}: {e}")
                        continue
        print(f"pagele_books: {pagele_books}")
        return pagele_books
    except Exception as e:
        print(f"Error getting available pagele files: {e}")
        return []

async def handle_connection(websocket):
    print(f"Client connected, {websocket.remote_address}")
    
    try:
        data = await websocket.recv()
        data_object = json.loads(data)
        client_ip = websocket.remote_address[0]
        data_object['ip'] = client_ip
        print(f"Received data: {data_object}")
        message_returned = {"error": "Invalid task"}  # Default message

        if data_object.get("task") == "get_books":
            books = await get_available_books()
            message_returned = {"books": books}
        elif data_object.get("task") == "get_book_data":
            message_returned = await get_book_data(data_object)
        elif data_object.get("task") == "stt":
            message_returned = stt_task(data_object)
            logger.log_event("stt", {**data_object, **message_returned}, websocket)
        elif data_object.get("task") == "tts":
            message_returned = await tts_task(data_object)
            logger.log_event("tts", {**data_object, **message_returned}, websocket)
        elif data_object.get("task") == "login":
            message_returned = await db.login_task(data_object)
            logger.log_event("login", {**data_object, **message_returned}, websocket)
        elif data_object.get("task") == "signup":
            message_returned = db.signup_task(data_object)
            logger.log_event("signup", {**data_object, **message_returned}, websocket)
        elif data_object.get("task") == "change_settings":
            message_returned = db.change_settings_task(data_object)
            logger.log_event("settings_change", {**data_object, **message_returned}, websocket)
        elif data_object.get("task") == "translate":
            message_returned = await translate_task(data_object)
            logger.log_event("translation", {**data_object, **message_returned}, websocket)
        elif data_object.get("task") == "verify_token":
            message_returned = await verify_token_task(data_object)
            logger.log_event("token_verification", {**data_object, **message_returned}, websocket)
        elif data_object.get("task") == "pagele":
            message_returned = await pagele_task(data_object)
        elif data_object.get("task") == "get_pagele_list":
            message_returned = await get_available_pagele()
        elif data_object.get("task") == "init_pagele":
            message_returned = await init_pagele_task(data_object)

        await websocket.send(json.dumps(message_returned))

    except Exception as e:
        print(f"Error: {e}")
        logger.log_event("error", {"error": str(e)}, websocket)
        await websocket.send(json.dumps({"error": str(e)}))
    finally:
        print("Client disconnected")

# Add a function to preload all models
def preload_all_models():
    """Preload all language models at startup"""
    print("Preloading all language models...")
    for lang in language_dict:
        print(f"Loading {lang} model...")
        model_manager.get_or_load_model(lang)
    print("All models loaded!")

class DatabaseManager:
    def __init__(self, db_path):
        self.DB_PATH = db_path #a constructor defines what parameters are needed to create an object

    def init_database(self):
        """Initialize the SQLite database with necessary tables"""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        # Create users table with CFI instead of page
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            current_book TEXT DEFAULT '',
            cfi TEXT DEFAULT '',  -- Changed from page to cfi
            utterrance_fname TEXT DEFAULT '',
            predicted_sentence TEXT DEFAULT '',
            preferred_language TEXT DEFAULT 'en',
            login_token TEXT,
            token_expiry TIMESTAMP,
            pagele TEXT DEFAULT '{}'  -- New column for storing pagele data as JSON
        )
        ''')

        # Check if pagele column exists, add it if it doesn't
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'pagele' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN pagele TEXT DEFAULT '{}'")
            conn.commit()
            print("Added pagele column to users table")

        conn.commit()
        conn.close()

    async def login_task(self, data_object):
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        username = data_object.get("username")
        password = data_object.get("password")

        # Get column names
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]

        cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?",
                    (username, password))
        user_row = cursor.fetchone()

        if user_row:
            # Convert row to dictionary
            user = dict(zip(columns, user_row))
            print(user)

            # Generate new token using UUID and set expiry to 24 hours from now
            token = str(uuid.uuid4())
            expiry = datetime.now() + timedelta(days=10000)

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
                "current_book": user.get("current_book", ""),
                "cfi": user.get("cfi", ""),  # Use CFI instead of page
                "preferred_language": user.get("preferred_language", "en"),
                "pagele": json.loads(user.get("pagele", "{}")),
                "type": "login"
            }

            # Only add epub and language if there's a current book
            if user.get("current_book"):
                print("getting books")
                epub = None
                language = None
                epubs = os.walk(EPUB_DIR)
                for root, dirs, files in epubs:
                    for file in files:
                        if file == user["current_book"]:
                            book_path = Path(root) / file
                            break
                print(f"book_path: {book_path}")
                if book_path.exists():
                    language = book_path.parent.name
                    print(f"language: {language}")
                    with open(book_path, "rb") as f:
                        book_data = f.read()
                        book_base64 = base64.b64encode(book_data).decode('utf-8')
                        epub = f"data:application/epub+zip;base64,{book_base64}"
                        print(f"Including book data for {user['current_book']}")

            if epub and language:
                message["epub"] = epub
                message["language"] = language

        else:
            message = {
                "status": "error",
                "message": "Invalid credentials"
            }

        conn.close()
        return message

    def signup_task(self, data_object):
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        username = data_object.get("username")
        password = data_object.get("password")

        # Check if username already exists
        cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            conn.close()
            return {"status": "error", "message": "Username already exists"}

        # Generate token for new user
        token = str(uuid.uuid4())
        expiry = datetime.now() + timedelta(days=10000)

        # Create new user with token
        cursor.execute("""
            INSERT INTO users (username, password, login_token, token_expiry)
            VALUES (?, ?, ?, ?)""",
            (username, password, token, expiry))
        conn.commit()
        conn.close()

        return {
            "status": "success",
            "message": "User created successfully",
            "token": token,
            "username": username
        }

    def set_current_book_task(self, data_object, utterrance_fname, predicted_sentence):
        try:
            conn = sqlite3.connect(self.DB_PATH)
            cursor = conn.cursor()

            username = data_object.get("username")
            current_book = data_object.get("book", data_object.get("current_book", ""))  # Try both book and current_book
            cfi = data_object.get("cfi", "")  # Use CFI instead of page

            print(f"Updating database for {username} with book: {current_book}, cfi: {cfi}")

            # First verify the user exists
            cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
            if not cursor.fetchone():
                print(f"User {username} not found in database")
                return False

            # Update the user's current book and cfi
            cursor.execute("""
                UPDATE users
                SET current_book = ?,
                    cfi = ?,
                    utterrance_fname = ?,
                    predicted_sentence = ?
                WHERE username = ?
            """, (current_book, cfi, utterrance_fname, predicted_sentence, username))

            if cursor.rowcount == 0:
                print(f"No rows were updated for user {username}")
            else:
                print(f"Successfully updated book info for user {username}")

            conn.commit()

            # Verify the update
            cursor.execute("SELECT current_book, cfi FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()
            print(f"After update - Book: {result[0]}, CFI: {result[1]}")

            return True
        except Exception as e:
            print(f"Error in set_current_book_task: {e}")
            return False
        finally:
            conn.close()

    def change_settings_task(self, data_object):
        """Handle user settings updates"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        token = data_object.get("token")
        updates = {}

        if "current_book" in data_object:
            updates["current_book"] = data_object["current_book"]
        if "cfi" in data_object:
            updates["cfi"] = data_object["cfi"]
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

    def update_pagele_data(self, data_object):
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()            
        cursor.execute("SELECT pagele FROM users WHERE username = ?", (data_object.get("username"),))
        result = cursor.fetchone()
        pagele_data = data_object.get("pagele")
        cursor.execute(
            "UPDATE users SET pagele = ? WHERE username = ?",
            (json.dumps(pagele_data), data_object.get("username"))
        )
        
        conn.commit()
        conn.close()
        
    def get_pagele_data(self, username, book_name=None):
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT pagele FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        
        if not result:
            return {} if book_name is None else 0
        
        try:
            pagele_data = json.loads(result[0])
        except (json.JSONDecodeError, TypeError):
            pagele_data = {}
        conn.close()

        return pagele_data
    
    def get_user_data(self, token):
        print(f"get_user_data: {token}")
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        cursor.execute("SELECT * FROM users WHERE login_token = ?", (token,))
        result = cursor.fetchone()
        conn.close()
        result = dict(zip(columns, result))
        print(f"result: {result}")
        return result

class TextComparator:
    @staticmethod
    def generate_html_report(text1, text2, output_file='text_comparison_report.html'):
        def normalize_text(text):
            text = text.lower()
            for char in ',.!?;:«»""()[]{}':
                text = text.replace(char, ' ')
            text = text.replace("'", "'")
            return [word for word in text.split() if word]

        original_words = normalize_text(text1)
        spoken_words = normalize_text(text2)

        print(f"Original words: {original_words}")
        print(f"Spoken words: {spoken_words}")
        m, n = len(original_words), len(spoken_words)
        score = [[0 for _ in range(n+1)] for _ in range(m+1)]
        traceback = [[None for _ in range(n+1)] for _ in range(m+1)]
        for i in range(m+1):
            score[i][0] = -i
            if i > 0:
                traceback[i][0] = "up"

        for j in range(n+1):
            score[0][j] = -j
            if j > 0:
                traceback[0][j] = "left"
        for i in range(1, m+1):
            for j in range(1, n+1):
                word_similarity = 1 - Levenshtein.distance(original_words[i-1], spoken_words[j-1]) / max(len(original_words[i-1]), len(spoken_words[j-1]))

                match_score = score[i-1][j-1] + (2 * word_similarity - 1)  # Reward for similar words, penalty for different
                delete_score = score[i-1][j] - 0.5  # Gap penalty
                insert_score = score[i][j-1] - 0.5  # Gap penalty

                best_score = max(match_score, delete_score, insert_score)
                score[i][j] = best_score

                if best_score == match_score:
                    traceback[i][j] = "diag"
                elif best_score == delete_score:
                    traceback[i][j] = "up"
                else:
                    traceback[i][j] = "left"

        # Traceback to find the alignment
        aligned_original = []
        aligned_spoken = []
        i, j = m, n

        while i > 0 or j > 0:
            if i > 0 and j > 0 and traceback[i][j] == "diag":
                aligned_original.append(original_words[i-1])
                aligned_spoken.append(spoken_words[j-1])
                i -= 1
                j -= 1
            elif i > 0 and traceback[i][j] == "up":
                aligned_original.append(original_words[i-1])
                aligned_spoken.append(None)  # Gap in spoken
                i -= 1
            else:  # traceback[i][j] == "left"
                aligned_original.append(None)  # Gap in original
                aligned_spoken.append(spoken_words[j-1])
                j -= 1

        # Reverse the alignments
        aligned_original.reverse()
        aligned_spoken.reverse()

        # Generate HTML output
        marked_output = []

        for orig, spoken in zip(aligned_original, aligned_spoken):
            if orig is None:
                # Extra word in spoken text
                marked_output.append(f'<span id="" class="wrong">{spoken}</span>')
            elif spoken is None:
                # Missing word in spoken text (optional to include)
                continue
            else:
                # Both words exist - check similarity
                distance = Levenshtein.distance(orig, spoken)
                max_len = max(len(orig), len(spoken))
                ratio = distance / max_len if max_len > 0 else 0

                if distance > 1 and ratio > 0.2:
                    marked_output.append(f'<span id="{orig}" class="wrong">{spoken}</span>')
                else:
                    marked_output.append(spoken)

        # Join the marked words back into text
        marked_text = ' '.join(marked_output)
        print(f"marked_text: {marked_text}")
        # Calculate overall similarity
        similarity_ratio = difflib.SequenceMatcher(None, original_words, spoken_words).ratio()

        return marked_text, similarity_ratio, original_words, spoken_words

class Logger:
    def __init__(self, log_file="logs/activity_log.json"):
        self.log_file = Path(log_file)
        # Create logs directory if it doesn't exist
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        # Create the file if it doesn't exist
        if not self.log_file.exists():
            self.log_file.write_text("[]")

    def log_event(self, event_type, data, websocket=None):
        """Log an event with all relevant details"""
        try:
            timestamp = datetime.now().isoformat()
            ip_address = websocket.remote_address[0] if websocket else "unknown"

            log_entry = {
                "timestamp": timestamp,
                "event_type": event_type,
                "ip_address": ip_address,
                "username": data.get("username", "anonymous"),
                "data": {
                    key: value for key, value in data.items()
                    if key not in ["blob", "epub"] # Exclude large binary data
                }
            }

            # Add specific fields based on event type
            if event_type == "stt":
                log_entry["stt_details"] = {
                    "language": data.get("language"),
                    "sentence": data.get("sentence"),
                    "predicted_sentence": data.get("pred_sentence", ""),
                    "points": data.get("points"),
                    "total_words": data.get("total_words"),
                    "wrong_words": data.get("wrong_words")
                }
            elif event_type == "translation":
                log_entry["translation_details"] = {
                    "source_lang": data.get("source_lang"),
                    "target_lang": data.get("target_lang"),
                    "source_text": data.get("text"),
                    "translated_text": data.get("translated_text")
                }

            # Read existing logs
            with open(self.log_file, 'r') as f:
                logs = json.load(f)

            # Append new log
            logs.append(log_entry)

            # Write updated logs
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)

        except Exception as e:
            print(f"Logging error: {e}")

# Initialize the logger
logger = Logger()

async def main():

    global db
    db = DatabaseManager("usersHablas.db")
    db.init_database();
    #preload_all_models()

    # Create SSL context
    """ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(
        '/media/nas/SSLCerts/carriertech.uk/fullchain.pem',
        '/media/nas/SSLCerts/carriertech.uk/privkey.pem'
    )"""

    # Start the WebSocket server with SSL
    async with websockets.serve(
        handle_connection,
        "0.0.0.0",  # Changed from localhost to accept external connections
        8675,
        #ssl=ssl_context
    ):
        print("WebSocket server started on wss://carriertech.uk:8675")
        await asyncio.Future()  # Run forever


if __name__ == "__main__": #when you use a multi processer load, you use this so it doesnt crash. with asyncio you always have to do it
    asyncio.run(main())

