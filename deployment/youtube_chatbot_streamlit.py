import streamlit as st
import os
from getpass import getpass
import openai
import pinecone
import yt_dlp
import whisper
from urllib.parse import urlparse, parse_qs
from langchain_pinecone import Pinecone, PineconeVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from transformers import pipeline
import json
import re
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
import asyncio


from project.deployments.multimodal_youtube_bot import (
    extract_video_id,
    recognize_google_cloud,
    transcribe_audio_with_timestamps,
    download_youtube_audio,
    summarizer_tool,
    answer_question_tool,
    store_transcript_in_pinecone,
    clean_json_like_string
)

def load_api_keys():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    if not openai_api_key or not pinecone_api_key:
        st.error("API keys not found. Please enter them in the settings.")
        return False
    
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    return True
def initialize_database():
    """Initialize SQLite database for storing video information"""
    conn = sqlite3.connect('youtube_videos_chats.db')
    c = conn.cursor()
    
    # Create table for videos if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS videos (
        video_id TEXT PRIMARY KEY,
        title TEXT,
        url TEXT,
        processed_date TEXT,
        transcript_chunks INTEGER
    )
    ''')
    
    # Create table for chat history
    c.execute('''
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT,
        chat_name TEXT,
        created_date TEXT,
        FOREIGN KEY (video_id) REFERENCES videos(video_id)
    )
    ''')
    
    # Create table for chat messages with tool column
    c.execute('''
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER,
        role TEXT,
        content TEXT,
        timestamp TEXT,
        chunks TEXT,
        tool TEXT,
        FOREIGN KEY (chat_id) REFERENCES chats(id)
    )
    ''')
    
    conn.commit()
    conn.close()
    return True

def check_video_in_database(video_id):
    """Check if video is already processed and stored in database"""
    conn = sqlite3.connect('youtube_videos_chats.db')  # Fixed database name
    c = conn.cursor()
    c.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,))
    result = c.fetchone()
    conn.close()
    
    if result:
        return True, {
            'video_id': result[0],
            'title': result[1],
            'url': result[2],
            'processed_date': result[3],
            'transcript_chunks': result[4]
        }
    return False, None

def save_video_to_database(video_id, title, url, chunks_count):
    """Save processed video information to database"""
    conn = sqlite3.connect('youtube_videos_chats.db')  # Fixed database name
    c = conn.cursor()
    processed_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute("""
    INSERT OR REPLACE INTO videos (video_id, title, url, processed_date, transcript_chunks)
    VALUES (?, ?, ?, ?, ?)
    """, (video_id, title, url, processed_date, chunks_count))
    
    conn.commit()
    conn.close()
    return True

def create_new_chat(video_id, chat_name=None):
    """Create a new chat session for a video"""
    conn = sqlite3.connect('youtube_videos_chats.db')  # Fixed database name
    c = conn.cursor()
    created_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if not chat_name:
        chat_name = f"Chat {created_date}"
    
    c.execute("""
    INSERT INTO chats (video_id, chat_name, created_date)
    VALUES (?, ?, ?)
    """, (video_id, chat_name, created_date))
    
    chat_id = c.lastrowid
    conn.commit()
    conn.close()
    return chat_id

def get_chats_for_video(video_id):
    """Get all chats for a specific video"""
    conn = sqlite3.connect('youtube_videos_chats.db')  # Fixed database name
    c = conn.cursor()
    c.execute("SELECT id, chat_name, created_date FROM chats WHERE video_id = ? ORDER BY created_date DESC", (video_id,))
    chats = c.fetchall()
    conn.close()
    return chats

def save_chat_message(chat_id, role, content, chunks=None, tool=None):
    """Save a chat message to the database"""
    conn = sqlite3.connect('youtube_videos_chats.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if chunks is not None and not isinstance(chunks, str):
        chunks = json.dumps(chunks)
    if tool is not None and not isinstance(tool, str):
        tool = json.dumps(tool)
    c.execute("""
        INSERT INTO chat_messages (chat_id, role, content, timestamp, chunks, tool)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (chat_id, role, content, timestamp, chunks, tool))
    conn.commit()
    conn.close()
    return True

def get_chat_messages(chat_id):
    """Get all messages for a specific chat"""
    conn = sqlite3.connect('youtube_videos_chats.db')  # Fixed database name
    c = conn.cursor()
    c.execute("SELECT role, content, chunks, tool FROM chat_messages WHERE chat_id = ? ORDER BY timestamp", (chat_id,))
    messages = c.fetchall()
    conn.close()
    
    formatted_messages = []
    for role, content, chunks_str, tool in messages:
        message = {"role": role, "content": content}
        
        # Parse chunks if available
        if chunks_str:
            try:
                chunks = json.loads(chunks_str)
                message["chunks"] = chunks
            except:
                pass  # If chunks can't be parsed, just skip it
        
        # Add tool if available
        if tool:
            message["tool"] = tool
            
        formatted_messages.append(message)
    
    return formatted_messages

def get_processed_videos():
    """Get all processed videos from database"""
    conn = sqlite3.connect('youtube_videos_chats.db')  # Fixed database name
    c = conn.cursor()
    c.execute("SELECT video_id, title, url, processed_date FROM videos ORDER BY processed_date DESC")
    videos = c.fetchall()
    conn.close()
    return videos

def delete_video(video_id):
    """Delete a video and all associated chats and messages from database"""
    conn = sqlite3.connect('youtube_videos_chats.db')  # Fixed database name
    c = conn.cursor()
    
    # Get all chat IDs for this video first
    c.execute("SELECT id FROM chats WHERE video_id = ?", (video_id,))
    chat_ids = [row[0] for row in c.fetchall()]
    
    # Delete all chat messages for these chats
    for chat_id in chat_ids:
        c.execute("DELETE FROM chat_messages WHERE chat_id = ?", (chat_id,))
    
    # Delete all chats for this video
    c.execute("DELETE FROM chats WHERE video_id = ?", (video_id,))
    
    # Finally delete the video
    c.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
    
    # Also delete from Pinecone if needed (can be implemented separately)
    
    conn.commit()
    conn.close()
    return True

def delete_chat(chat_id):
    """Delete a chat and all its messages from database"""
    conn = sqlite3.connect('youtube_videos_chats.db')  # Fixed database name
    c = conn.cursor()
    
    # Delete all messages in this chat
    c.execute("DELETE FROM chat_messages WHERE chat_id = ?", (chat_id,))
    
    # Delete the chat itself
    c.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    print("results: " , c.rowcount)
    conn.commit()
    conn.close()
    return True
def initialize_pinecone():
    pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = 'youtube-video-index00'
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=pinecone.ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    
    index = pc.Index(index_name)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectordb = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
    return index, vectordb, embeddings

def store_transcript_in_pinecone(transcript, video_title, video_id, index, embeddings):
    all_chunks = []
    full_text = " ".join([segment['text'] for segment in transcript])
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4", chunk_size=200, chunk_overlap=50
    )
    token_chunks = splitter.split_text(full_text)
    for chunk in token_chunks:
        all_chunks.append({ "text": chunk })
    
    texts = [chunk['text'] for chunk in all_chunks]
    vectors = embeddings.embed_documents(texts)
    metadata = [
        {
            "source": video_id,
            "title": video_title,
            "text": chunk['text']
        } for chunk in all_chunks
    ]
    
    index.upsert([
        (f"{video_id}-{i}", vector, meta)
        for i, (vector, meta) in enumerate(zip(vectors, metadata))
    ])
    
    vectordb = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
    return vectordb, len(all_chunks)

def process_youtube_video(youtube_url, index, embeddings):
    with st.spinner("Processing the video..."):
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("Invalid YouTube URL format")
            return None, None, None
        
        # Check if video already exists in database
        video_exists, video_data = check_video_in_database(video_id)
        if video_exists:
            st.info(f"Video '{video_data['title']}' already processed. Loading from database...")
            vectordb = PineconeVectorStore(
                index=index, 
                embedding=embeddings, 
                text_key="text",
                namespace=video_id
            )
            return vectordb, video_data['title'], video_id
        
        audio_file, video_title = download_youtube_audio(youtube_url)
        if not audio_file:
            st.error("Failed to download audio.")
            return None, None, None
        
        transcript_chunks = transcribe_audio_with_timestamps(audio_file)
        if not transcript_chunks:
            st.error("Failed to transcribe audio.")
            return None, None, None
        
        vectordb, chunks_count = store_transcript_in_pinecone(transcript_chunks, video_title, video_id, index, embeddings)
        if vectordb is None:
            st.error("Failed to store transcript in Pinecone.")
            return None, None, None
        
        # Save video information to database
        save_video_to_database(video_id, video_title, youtube_url, chunks_count)
        
        return vectordb, video_title, video_id
    
import os
import edge_tts

async def speak_text(text, lang="english"):
    filename = "output.mp3"

    # Delete the file if it already exists to avoid writing errors
    if os.path.exists(filename):
        try:
            os.remove(filename)
        except PermissionError:
            print("⚠️ Cannot delete 'output.mp3'. Make sure it's not open in another audio player.")
            return None

    # Voice map for different languages
    voice_map = {
        "english": "en-US-JennyNeural",
        "arabic": "ar-SA-ZariyahNeural"
    }

    # Choose appropriate voice based on the selected language
    voice = voice_map.get(lang.lower(), "en-US-JennyNeural")

    # Generate speech using edge-tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(filename)
    print("🔊 Response saved as 'output.mp3' ✅")

    return filename  # Return the path to the generated audio file


def generate_quiz_from_text(text, llm):
    prompt = f"""
    Create a 10-question multiple choice quiz based on the following content:

    {text}

    Each question should have four options (A, B, C, D) and specify the correct answer letter.
    Return it as a Python list of dictionaries like this:
    [
        {{
            "question": "...",
            "options": ["...", "...", "...", "..."],
            "answer": "A"
        }}
    ]
    """
    
    response = llm.invoke(prompt)
    response_text = getattr(response, "content", None) or (response.text() if callable(response.text) else response.text)
    if not isinstance(response_text, str):
        raise ValueError("The LLM response does not contain valid text content.")
    
    try:
        cleaned = clean_json_like_string(response_text)
        quiz = json.loads(cleaned)
        return quiz
    except json.JSONDecodeError as e:
        raise ValueError(f"The response from LLM is not valid JSON: {e}")

def run_quiz_from_vectordb(vectordb, llm, max_k=100):
    results = vectordb.similarity_search("summarize all content", k=max_k)
    retrieved_chunks = [doc.page_content for doc in results]
    full_text = "\n\n".join(retrieved_chunks)
    return generate_quiz_from_text(full_text, llm)


def initialize_langchain(vectordb):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
    
    def summarizer_tool_wrapper(query):
        result = summarizer_tool(query, vectordb)
        return {"tool_name": "SummarizerTool", "result": result}
    
    def answer_question_tool_wrapper(query):
        result = answer_question_tool(query, vectordb, llm)
        return {"tool_name": "AnswerQuestionTool", "result": result}
    
    def quiz_tool_wrapper(_):
        result = run_quiz_from_vectordb(vectordb, llm)
        return {"tool_name": "QuizTool", "result": result}
    
    tools = [
        Tool(
            name="SummarizerTool",
            func=summarizer_tool_wrapper,
            description="Use this tool ONLY if the input is asking to summarize a video."
        ),
        Tool(
            name="AnswerQuestionTool",
            func=answer_question_tool_wrapper,
            description="Use this tool ONLY if the user is asking a specific question that requires retrieval."
        ),
        Tool(
            name="QuizTool",
            func=quiz_tool_wrapper,
            description="Use this tool if the user wants to take a quiz based on the video content."
        )
    ]
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )
    
    return agent, llm


def main():
    st.set_page_config(page_title="YouTube Video Assistant", page_icon="🎬", layout="wide")
    
    # Initialize session states
    if 'api_keys_loaded' not in st.session_state:
        st.session_state.api_keys_loaded = load_api_keys()
    if 'db_initialized' not in st.session_state:
        st.session_state.db_initialized = initialize_database()

    if 'processed_video' not in st.session_state:
        st.session_state.processed_video = False
    if 'vectordb' not in st.session_state:
        st.session_state.vectordb = None
    if 'agent' not in st.session_state:
        st.session_state.agent = None
        
    if 'video_title' not in st.session_state:
        st.session_state.video_title = None
    if 'video_id' not in st.session_state:
        st.session_state.video_id = None
    if 'quiz_questions' not in st.session_state:
        st.session_state.quiz_questions = None
    if 'quiz_started' not in st.session_state:
        st.session_state.quiz_started = False
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = {}
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'show_input_box' not in st.session_state:
        st.session_state.show_input_box = True
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = None
    if 'last_response_chunks' not in st.session_state:
        st.session_state.last_response_chunks = []
    if 'last_used_tool' not in st.session_state:
        st.session_state.last_used_tool = None
    if 'input_mode' not in st.session_state:
        st.session_state.input_mode = "Text"
    if 'output_mode' not in st.session_state:
        st.session_state.output_mode = "Text"

    st.title("🎬 YouTube Video Assistant")
    st.markdown("*This assistant analyzes YouTube videos. You can summarize, ask questions, or take quizzes based on video content.*")

    if st.session_state.api_keys_loaded and st.session_state.vectordb is None:
        try:
            index, vectordb, embeddings = initialize_pinecone()
            st.session_state.index = index
            st.session_state.embeddings = embeddings
        except Exception as e:
            st.error(f"Error initializing Pinecone: {e}")
            return
    
    # Sidebar for video selection and management
    with st.sidebar:
        st.markdown("<h1 style='color: #8E7DBE ;'>Video Management</h1>", unsafe_allow_html=True)
        
        # Section to process new videos
        st.subheader("Process New Video")
        youtube_url = st.text_input("Enter YouTube URL:")
        
        if youtube_url and st.button("Process Video"):
            if not extract_video_id(youtube_url):
                st.error("Invalid YouTube URL format")
            else:
                vectordb, video_title, video_id = process_youtube_video(
                    youtube_url, 
                    st.session_state.index,
                    st.session_state.embeddings
                )
                if vectordb and video_title and video_id:
                    st.session_state.vectordb = vectordb
                    st.session_state.video_title = video_title
                    st.session_state.video_id = video_id
                    st.session_state.processed_video = True
                    agent, llm = initialize_langchain(vectordb)
                    st.session_state.agent = agent
                    st.session_state.llm = llm
                    
                    # Create a new chat for this video if one doesn't exist
                    if not st.session_state.current_chat_id:
                        chat_id = create_new_chat(video_id, f"Chat for {video_title}")
                        st.session_state.current_chat_id = chat_id
                        st.session_state.chat_history = []
                    
                    st.success(f"Video '{video_title}' processed successfully!")
                    st.rerun()
        
        # Section to select previously processed videos
        st.subheader("Previously Processed Videos")
        processed_videos = get_processed_videos()
        if processed_videos:
            video_options = {f"{title} ({id})": (id, title, url) for id, title, url, _ in processed_videos}
            selected_video = st.selectbox(
                "Select a video:",
                options=list(video_options.keys()),
                format_func=lambda x: x.split(" (")[0]
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if selected_video and st.button("Load Selected Video"):
                    video_id, video_title, url = video_options[selected_video]
                    
                    # Load video data from Pinecone
                    vectordb = PineconeVectorStore(
                        index=st.session_state.index, 
                        embedding=st.session_state.embeddings, 
                        text_key="text"
                    )
                    
                    st.session_state.vectordb = vectordb
                    st.session_state.video_title = video_title
                    st.session_state.video_id = video_id
                    st.session_state.processed_video = True
                    agent, llm = initialize_langchain(vectordb)
                    st.session_state.agent = agent
                    st.session_state.llm = llm
                    
                    st.success(f"Video '{video_title}' loaded successfully!")
                    st.rerun()
            
            with col2:
                if selected_video and st.button("🗑️ Delete Video", type="primary", use_container_width=True):
                    video_id, video_title, _ = video_options[selected_video]
                    
                    # Add a confirmation step
                    st.warning(f"Are you sure you want to delete '{video_title}' and all its chats?")
                    if st.button("✓ Confirm Delete", key="confirm_delete_video"):
                        try:
                            delete_video(video_id)
                            st.success(f"Video '{video_title}' and all associated chats deleted successfully!")
                            # Reset session state if the current video was deleted
                            if st.session_state.video_id == video_id:
                                st.session_state.processed_video = False
                                st.session_state.vectordb = None
                                st.session_state.video_title = None
                                st.session_state.video_id = None
                                st.session_state.current_chat_id = None
                                st.session_state.chat_history = []
                            
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting video: {e}")
        else:
            st.info("No processed videos found. Process a video first.")
        
        # Simplified Chat Management section
        if st.session_state.processed_video and st.session_state.video_id:
            st.markdown("<h1 style='color: #8E7DBE;'>Chat Sessions</h1>", unsafe_allow_html=True)
            
            # Create new chat button
            if st.button("🆕 New Chat Session", use_container_width=True):
                chat_id = create_new_chat(
                    st.session_state.video_id, 
                    f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )
                st.session_state.current_chat_id = chat_id
                st.session_state.chat_history = []
                st.success("New chat session started!")
                st.rerun()
            
            # Clear current chat button
            if st.session_state.chat_history and st.button("🧹 Clear Current Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.success("Chat cleared!")
                st.rerun()
            
            # Option to switch between previous chats
            chats = get_chats_for_video(st.session_state.video_id)
            if len(chats) > 1:  # Only show if there are multiple chats
                st.subheader("Previous Sessions")
                for chat_id, chat_name, created_date in chats:
                    # Don't list the current chat
                    if st.session_state.current_chat_id != chat_id:
                        if st.button(f"📝 {chat_name} ({created_date})", key=f"load_chat_{chat_id}", use_container_width=True):
                            st.session_state.current_chat_id = chat_id
                            st.session_state.chat_history = get_chat_messages(chat_id)
                            st.rerun()
            
            # Display current chat ID
            if st.session_state.current_chat_id:
                st.info(f"Current Session ID: {st.session_state.current_chat_id}")

    # Main content area
    if st.session_state.processed_video:
        video_id = st.session_state.video_id
        video_title = st.session_state.video_title

        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

        st.image(thumbnail_url, width=300)
        st.subheader(video_title)

        st.markdown("---")
        
        tab1, tab2 = st.tabs(["Chat", "Quiz"])

        with tab1:
            # Display chat history with chunks
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

                    # Display tool information if available - making it more prominent
                    if message["role"] == "assistant" and "tool" in message and message["tool"]:
                        st.success(f"🛠️ Tool Used: {message['tool']}")

                    # Always show retrieved chunks for assistant messages
                    if message["role"] == "assistant" and "chunks" in message:
                        with st.expander("📄 Source Chunks and Relevance Scores", expanded=False):
                            for i, (chunk, score) in enumerate(message["chunks"]):
                                st.markdown(f"**Chunk {i+1} (Score: {score:.4f})**")
                                st.markdown(f"> {chunk}")
                                st.markdown("---")

            if st.session_state.agent and st.session_state.vectordb and st.session_state.show_input_box:
                st.markdown("### 💬 Ask your question")
                
                # Let user choose between text or voice input
                st.session_state.input_mode = st.radio("Choose input mode:", ["Text", "Voice"])
                user_question = ""

                if st.session_state.input_mode == "Text":
                    user_question = st.chat_input("Ask your question about the video...")
                elif st.session_state.input_mode == "Voice":
                    if st.button("🎙️ Record Voice"):
                        with st.spinner("🎧 Listening..."):
                            user_question = recognize_google_cloud()
                        if user_question:
                            st.info(f"🗣️ Recognized: {user_question}")
                        else:
                            st.warning("⚠️ Could not recognize voice.")

            # Output mode selection
            output_mode = st.radio(
                "Choose output mode:", 
                ["Text", "Voice", "Text & Voice"],
                key="output_mode"
            )

            if user_question:
                # Add user message
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                if st.session_state.current_chat_id:

                    save_chat_message(st.session_state.current_chat_id, "user", user_question, [], None)

                with st.chat_message("user"):
                    st.write(user_question)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            # Get chunks with scores
                            retrieved_docs = st.session_state.vectordb.similarity_search_with_score(user_question, k=5)
                            retrieved_chunks = [(doc.page_content, score) for doc, score in retrieved_docs]

                            # Run the agent
                            response = st.session_state.agent.run(user_question)

                            # Extract tool name if available
                            tool_name = None
                            if isinstance(response, dict) and "tool_name" in response:
                                tool_name = response["tool_name"]
                                response = response["result"]

                            # Display text response
                            output_mode = st.session_state.get("output_mode", "Text")
                            if output_mode in ["Text", "Text & Voice"]:
                                st.write(response)
                            
                            # Generate and display voice response
                            if output_mode in ["Voice", "Text & Voice"]:
                                audio_file = asyncio.run(speak_text(response))
                                if audio_file:
                                    st.audio(audio_file, format="audio/mp3")
                                else:
                                    st.warning("⚠️ Sound generation failure.")

                            # Make tool information more prominent - displayed before chunks
                            if tool_name:
                                st.success(f"🛠️ Tool Used: {tool_name}")

                            # Show chunks
                            with st.expander("📄 Source Chunks and Relevance Scores", expanded=False):
                                for i, (chunk, score) in enumerate(retrieved_chunks):
                                    st.markdown(f"**Chunk {i+1} (Score: {score:.4f})**")
                                    st.markdown(f"> {chunk}")
                                    st.markdown("---")

                            # Save to history
                            assistant_message = {
                                "role": "assistant",
                                "content": response,
                                "chunks": retrieved_chunks
                            }
                            if tool_name:
                                assistant_message["tool"] = tool_name

                            st.session_state.chat_history.append(assistant_message)

                            if st.session_state.current_chat_id:
                                save_chat_message(
                                    st.session_state.current_chat_id,
                                    "assistant",
                                    response,
                                    json.dumps(retrieved_chunks),
                                    tool_name
                                )

                        except Exception as e:
                            error_msg = f"Error generating response: {str(e)}"
                            st.error(error_msg)
                            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                            if st.session_state.current_chat_id:
                                save_chat_message(st.session_state.current_chat_id, "assistant", error_msg, [], None)

        with tab2:
            st.subheader("Test Your Knowledge")

            if not st.session_state.get("quiz_started", False):
                if st.button("Generate Quiz"):
                    with st.spinner("Generating quiz questions..."):
                        try:
                            st.session_state.quiz_questions = run_quiz_from_vectordb(
                                st.session_state.vectordb,
                                st.session_state.llm
                            )
                            st.session_state.quiz_started = True
                            st.session_state.quiz_submitted = False
                            st.session_state.quiz_answers = {}
                        except Exception as e:
                            st.error(f"Error generating quiz: {e}")

            if st.session_state.get("quiz_started", False) and st.session_state.get("quiz_questions"):
                with st.form("quiz_form"):
                    for i, question in enumerate(st.session_state.quiz_questions):
                        st.write(f"**Question {i+1}**: {question['question']}")
                        options = {
                            "A": question['options'][0],
                            "B": question['options'][1],
                            "C": question['options'][2],
                            "D": question['options'][3]
                        }

                        if not st.session_state.get("quiz_submitted", False):
                            answer = st.radio(
                                f"Select your answer for question {i+1}:",
                                options=["A", "B", "C", "D"],
                                format_func=lambda x: f"{x}: {options[x]}",
                                key=f"q{i}"
                            )
                            st.session_state.quiz_answers[i] = answer
                        else:
                            user_answer = st.session_state.quiz_answers.get(i, "Not answered")
                            answer_text = options.get(user_answer, "No answer selected")
                            st.info(f"Your answer: {user_answer} - {answer_text}")

                        st.markdown("---")

                    submitted = st.form_submit_button(
                        "Submit Quiz",
                        disabled=st.session_state.get("quiz_submitted", False)
                    )
                    if submitted and not st.session_state.get("quiz_submitted", False):
                        st.session_state.quiz_submitted = True
                        st.rerun()

                if st.session_state.get("quiz_submitted", False):
                    st.subheader("Quiz Results")
                    correct_count = 0
                    for i, question in enumerate(st.session_state.quiz_questions):
                        user_answer = st.session_state.quiz_answers[i]
                        correct_answer = question['answer']
                        is_correct = user_answer == correct_answer
                        if is_correct:
                            st.success(f"Question {i+1}: Correct! Your answer: {user_answer}")
                            correct_count += 1
                        else:
                            st.error(f"Question {i+1}: Incorrect. Your answer: {user_answer}, Correct answer: {correct_answer}")
                    score_percentage = (correct_count / len(st.session_state.quiz_questions)) * 100
                    st.subheader(f"Your Score: {correct_count}/{len(st.session_state.quiz_questions)} ({score_percentage:.1f}%)")

                    if st.button("Take Another Quiz"):
                        st.session_state.quiz_started = False
                        st.session_state.quiz_questions = None
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_answers = {}
                        st.rerun()


if __name__ == "__main__":
    main()

