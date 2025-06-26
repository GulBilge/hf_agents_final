from langchain_community.tools import tool
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain_community.tools.tavily_search import TavilySearchResults
import openai
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.
    Args:
        a: first int
        b: second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a - b

@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers.
    
    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    """Get the modulus of two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a % b

@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 2 results.
    
    Args:
        query: The search query."""
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"wiki_results": formatted_search_docs}

@tool
def web_search(query: str) -> str:
    """Search Tavily for a query and return maximum 3 results.
    
    Args:
        query: The search query."""
    search_docs = TavilySearchResults(max_results=3).invoke(query=query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"web_results": formatted_search_docs}

@tool
def arvix_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.
    
    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ])
    return {"arvix_results": formatted_search_docs}

def _get_video_id_from_url(url: str) -> str | None:
    """Verilen YouTube URL'sinden video ID'sini çıkarır."""
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    
    query = urlparse(url).query
    params = parse_qs(query)
    
    if 'v' in params:
        return params['v'][0]
    
    return None

def _get_transcript(video_id: str) -> str | None:
    """Belirtilen video ID'si için altyazıyı çeker ve metin olarak birleştirir."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['tr', 'en'])
        return " ".join([item['text'] for item in transcript_list])
    except Exception as e:
        print(f"Altyazı alınırken hata: {e}")
        return None

@tool
def get_youtube_transcript(video_url: str) -> str:
    """
    Fetches the full text transcript of a YouTube video from its URL.
    Use this tool when you need to understand, summarize, or analyze the content of a specific YouTube video.
    The input must be a valid YouTube video URL.
    Returns the raw transcript text or an error message if it fails.
    """
    print(f"--- YouTube Transcript Aracı Çağrıldı: {video_url} ---")
    video_id = _get_video_id_from_url(video_url)
    if not video_id:
        return "HATA: Geçersiz YouTube URL'si."

    transcript = _get_transcript(video_id)
    
    # Çok uzun metinleri LLM'in context limitini aşmamak için kırpalım
    max_chars = 15000  # Yaklaşık 3500-4000 token
    if len(transcript) > max_chars:
        trimmed_transcript = transcript[:max_chars]
        return trimmed_transcript + "\n\n[...NOT: Metin çok uzun olduğu için kırpılmıştır...]"
        
    return transcript

@tool
def transcribe_audio(audio_file_path: str) -> str:
    """
    Transcribes an audio file to text using OpenAI's Whisper model.
    Use this tool whenever you need to understand the content of an audio file, such as a voice memo or recording.
    The input must be a valid path to an audio file (e.g., 'path/to/myaudio.mp3').
    Returns the transcribed text as a string.
    """
    print(f"--- Audio Transcription Tool Called: {audio_file_path} ---")
    try:
        with open(audio_file_path, "rb") as audio_file:
            # Whisper API'sini çağır
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        
        print("--- Transcription Successful ---")
        return transcript['text']
    except FileNotFoundError:
        return f"HATA: Belirtilen yolda dosya bulunamadı: {audio_file_path}"
    except Exception as e:
        return f"HATA: Ses dosyası işlenirken bir sorun oluştu: {e}"