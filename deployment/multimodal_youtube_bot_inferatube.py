# %% [markdown]
# # Project : Multimodal AI ChatBot for YouTube Videos ✨🤖

# %% [markdown]
# ### Pakages Installization

# %%
# ! pip install pytube youtube-transcript-api langchain openai whisper chromadb langchain-community
# ! pip install -U openai-whisper
# ! pip install -U langchain-openai
# ! pip install yt-dlp
# ! pip install --upgrade langchain
# ! pip install pinecone

# ! pip uninstall openai
# ! pip install openai==0.28

# ! pip install -U langchain-pinecone
# ! pip install google-cloud-speech pyaudio
# ! pip install python-dotenv
# ! pip install whisper
# ! pip install transformers

# ! pip install google-cloud-speech
# ! pip install edge-tts
# ! pip install SpeechRecognition edge-tts
# ! pip install playsound==1.2.2

# ! pip install langchain langsmith openai
# ! pip install --upgrade langchain
# ! pip install langchain-core

# %%
import os
from getpass import getpass
import openai
import pinecone
import yt_dlp
import whisper
from urllib.parse import urlparse, parse_qs
from langchain_pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_pinecone import Pinecone
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain.agents import AgentType
from langchain.vectorstores import FAISS
from transformers import pipeline
import pyaudio
from google.cloud import speech
from dotenv import load_dotenv

load_dotenv()


# LangSmith configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")  
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"  

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


# %% [markdown]
# # Build the Vector Database (Pinecone) 

# %% [markdown]
# We initially used Chroma for its simplicity and ease of local development, which helped us iterate quickly. However, to support larger-scale use cases, faster similarity search, and seamless integration with cloud services, we transitioned to Pinecone, which offers a production-ready, scalable vector database solution.
# 
# 

# %%
import os
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone instance
pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index_name = 'youtube-video-index000'

# Check if the index already exists, otherwise create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536 ,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )


# %%
index = pc.Index(index_name)
# index.describe_index_stats()

# %%
index = pc.Index(index_name)

# Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Pinecone database
vectordb = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")


# %% [markdown]
# ## 1️⃣ Extract and Process Video Content 🎬
# 
# 1.   Extract Video ID from YouTube URL 
# 2.   Download YouTube Audio
# 3.   Transcribe Audio using Whisper API
# 4.   Store Transcript in Pinecon
# 
# 
# 

# %%
#Helper function to save chunks to a text file and save the transcript to a text file

def save_chunks_to_txt(chunks, filename="split_chunks.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"Chunk {i+1}:\n{chunk}\n\n")

def save_tra_to_txt(chunks, filename="split_chunks.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"Chunk {i+1}:\n{chunk}\n\n")

# %%
############################## STEP 1 : Extract Video ID from YouTube URL #################################
def extract_video_id(youtube_url):
    parsed_url = urlparse(youtube_url)
    if 'youtube.com' in parsed_url.netloc:
        # For long-form URLs like https://www.youtube.com/watch?v=VIDEO_ID
        # Extract the 'v' query parameter which contains the video ID

        query = parse_qs(parsed_url.query)
        return query['v'][0]
    
    elif 'youtu.be' in parsed_url.netloc:
        # For short-form URLs like https://youtu.be/VIDEO_ID
        # Extract the video ID directly from the path

        video_id = parsed_url.path.lstrip('/')
        # Remove any query parameters if they exist
        if '?' in video_id:
            video_id = video_id.split('?')[0]
        return video_id
    else:
        raise ValueError("Invalid YouTube URL format")

################################### STEP 2 : Download YouTube Audio as MP3  ###################################
def download_youtube_audio(youtube_url):
    try:
        # Create a directory to store downloaded audio files
        os.makedirs("./youtube_audio", exist_ok=True)

        # yt-dlp options for downloading best quality audio and converting to mp3
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': './youtube_audio/%(id)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
        }
        
        # Use yt-dlp to download the audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            filename = f"./youtube_audio/{info['id']}.mp3"
            return filename, info.get('title', None)
        
    except Exception as e:
        print(f"Error downloading YouTube audio: {e}")
        return None, None


############################# STEP 3 : Transcribe Audio with Timestamps Using Whisper ###################################
def transcribe_audio_with_timestamps(audio_file_path):
    # Load the Whisper speech-to-text model
    model = whisper.load_model("base")

    # Transcribe the audio file with word-level timestamps
    result = model.transcribe(audio_file_path, word_timestamps=True)

    segments = result['segments']  # Each segment contains text + start time + end time
    transcript_chunks = []

    # Loop through each segment and format it into a dictionary
    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text'].strip()

        transcript_chunks.append({
            'text': text,
            'start_time': start_time,
            'end_time': end_time
        })

    return transcript_chunks

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter

################################# STEP 4 : Store transcript in pinecone ###################################

def store_transcript_in_pinecone(transcript, video_title, video_id):
    all_chunks = []

    # Combine all segments into a single string
    full_text = " ".join([segment['text'] for segment in transcript])

    # Initialize the text splitter with a defined chunk size and overlap.
    # We experimented with different splitters like CharacterTextSplitter, TokenTextSplitter and RecursiveCharacterTextSplitter.
    # Based on our testing, RecursiveCharacterTextSplitter provided the best balance between preserving semantic coherence
    # and maintaining chunk sizes suitable for embedding, especially when using token-based models like GPT.
    # It also handles nested structures (like paragraphs and sentences) more intelligently, which improves retrieval quality.

    # splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=32)
    # splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=200, chunk_overlap=50 # found 200 and 50 to be a good balance for our use case
    )
    
    # Split the full transcript into smaller chunks of text
    token_chunks = splitter.split_text(full_text)

    # Store each chunk in a list of dictionaries with key "text"
    for chunk in token_chunks:
        all_chunks.append({
            "text": chunk
        })

    # Extract just the text content for embedding
    texts = [chunk['text'] for chunk in all_chunks]
    # Generate vector embeddings for each text chunk using your embedding model
    vectors = embeddings.embed_documents(texts)

    save_chunks_to_txt(texts, filename=f"split_chunks.txt")

    # Prepare metadata for each chunk to be stored in Pinecone
    metadata = [
        {
            "source": video_id,
            "title": video_title,
            "text": chunk['text']
        } for chunk in all_chunks
    ]

    # Upload vectors with metadata to the Pinecone index
    index.upsert([
        (f"{video_id}-{i}", vector, meta)
        for i, (vector, meta) in enumerate(zip(vectors, metadata))
    ])
    return vectordb

# %%
def process_youtube_video(youtube_url):
    video_id = extract_video_id(youtube_url)
    print(f"Processing video with ID: {video_id}")

    # Ensure audio and video are downloaded correctly
    audio_file, video_title = download_youtube_audio(youtube_url)
    if not audio_file:
        print("Failed to download audio.")
        return None

    # Ensure transcription is generated correctly
    transcript_chunks = transcribe_audio_with_timestamps(audio_file)
    if not transcript_chunks:
        print("Failed to transcribe audio.")
        return None

    # Store transcript in Pinecone
    vectordb = store_transcript_in_pinecone(transcript_chunks, video_title, video_id)
    if vectordb is None:
        print("Failed to store transcript in Pinecone.")
        return None

    print(f"Successfully processed video: {video_title}")
    return vectordb


# %% [markdown]
# ### Speech-To-Text & Text-To-Speech 🪄
# 
# * **STT (Speech-To-Text)** → Allows the user to ask their question verbally by converting their voice input into text for the agent to understand. (Using Google Cloud)
# 
# * **TTS (Text-To-Speech)** → Allows the agent to respond verbally by converting its text-based answer into speech. (Using Edge TTS )
# 

# %%
# Speech to Text using Google Cloud
def recognize_google_cloud():
    # Set up Google Cloud credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
    
    # Create a client to interact with the Google Cloud Speech-to-Text API
    client = speech.SpeechClient()

    # Define the audio sampling rate (16kHz is common for speech recognition)
    # Define the chunk size (how many samples to read at a time) — here it's 100ms

    RATE = 16000
    CHUNK = int(RATE / 10)

    # Initialize PyAudio to capture audio input from the microphone
    audio_interface = pyaudio.PyAudio()

    # Open an audio stream with specified parameters: mono channel, 16-bit format, 16kHz sample rate
    stream = audio_interface.open(format=pyaudio.paInt16,
                                   channels=1,
                                   rate=RATE,
                                   input=True,
                                   frames_per_buffer=CHUNK)

    print("Speak now...")

    # Collect audio frames for 5 seconds
    audio_frames = []
    for _ in range(0, int(RATE / CHUNK * 5)):  # total of 5 seconds
        data = stream.read(CHUNK)  # Read a chunk of audio data
        audio_frames.append(data)  # Append it to the list

    # Stop and close the audio stream after recording
    stream.stop_stream()
    stream.close()
    audio_interface.terminate()

    # Combine all recorded audio frames into a single byte string
    audio_data = b''.join(audio_frames)

    # Create a RecognitionAudio object using the recorded audio data
    audio = speech.RecognitionAudio(content=audio_data)
    # Configure recognition settings (audio format, sample rate, language, etc.)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )
    
    # Call the Google Speech-to-Text API to transcribe the audio
    response = client.recognize(config=config, audio=audio)

    # Handle the case where no speech was detected
    if not response.results:
        print("No speech detected.")
        return None

    # Extract the most likely transcript from the response
    transcript = response.results[0].alternatives[0].transcript
    print("Transcript:", transcript)
    return transcript



# %%
import os
import asyncio
import edge_tts
import edge_tts
import asyncio
from playsound import playsound 
from IPython.display import Audio, display

# Text to Speech using Edge TTS 

# Asynchronous function to convert text to speech and play the result
async def speak_text(text):

    filename = "output.mp3" # Output file where the speech will be saved

    if os.path.exists(filename):
        try:
            os.remove(filename)
        except PermissionError:
            print("The file 'output.mp3' cannot be deleted. Make sure it is not open in any audio player.")

            return

    # Speech generation
    # # Create a Communicate object from edge-tts using a natural-sounding English voice
    communicate = edge_tts.Communicate(text, "en-US-JennyNeural")

    # Generate speech from the text and save it to the output file
    await communicate.save(filename)
    print("Response saved as 'output.mp3' ")

   # automatically play the sound file
   # playsound("output.mp3")

    # manually play the sound file
    display(Audio(filename))   

# %% [markdown]
# # LangChain Agent 🤖✨✨
# with three Tools :
# 1. Question Answer Tool
# 2. Summarizer Tool
# 3. Quiz Tool

# %%
# Set up the LLM (Language Model)
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)

# %% [markdown]
# ### Question Answer Tool :
# 1. Question About youtube video stored in vectordb
# 2. Convert the Question into a Vector (Embedding)
# 3. Find the Most Relevant Chunks from VectorDB (Top-K Retrieval)
# 4. Send retrieved texts to the LLM
# 5. Generate a full answer
# 6. Display answer + sources

# %%
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

def answer_question_tool(query, vectordb, llm, top_k=4):
    # 1. Embed the query manually
    embeddings = OpenAIEmbeddings()
    query_embedding = embeddings.embed_query(query)

    # 2. Use similarity search with scores to get Top-K chunks + scores
    docs_and_scores = vectordb.similarity_search_with_score(query, k=top_k)

    # Separate documents and scores
    retrieved_chunks = [doc for doc, score in docs_and_scores]
    scores = [score for doc, score in docs_and_scores]

    # 3. Create a retriever from the vectordb
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})

    # 4. Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    # 5. Run the QA chain to get the final answer
    response = qa_chain.run(query)

    # 6. Print chunks and scores
    print("\nTop-K Chunks and Confidence Scores:")
    for i, (chunk, score) in enumerate(zip(retrieved_chunks, scores), 1):
        print(f"\nChunk {i} (Score: {score:.4f}):\n{chunk.page_content}")

    return response, list(zip(retrieved_chunks, scores))


# We use RetrievalQA from LangChain to simplify the Question Answering pipeline.
# This tool connects a retriever (our vector DB) with an LLM, making it easier to:
# 1. Automatically fetch the most relevant chunks from the vector DB.
# 2. Pass those chunks to the LLM as context for generating an accurate answer.
# 3. Ensure the answer is grounded in retrieved information (RAG pattern), reducing hallucinations.
# 4. Avoid writing custom logic to format, combine, and feed context manually.
# It's also customizable (e.g., chain type, number of chunks), making it flexible for our use case.


# %% [markdown]
# ### Summarizer Tool :
# 1. Embed the user query
# 2. Search for Top-K relevant chunks with scores
# 3. Combine the retrieved chunks
# 4. Summarize using BART model
# 5. Print Top-K chunks and their similarity scores
# 6. Return the summary and source chunks

# %%
# tool to summarize texts using HuggingFace's BART model
def summarizer_tool(query, vectordb, top_k=4):
    # 1. Embed the query manually
    embeddings = OpenAIEmbeddings()
    query_embedding = embeddings.embed_query(query)

    # 2. Use similarity search with scores to get Top-K chunks + scores
    docs_and_scores = vectordb.similarity_search_with_score(query, k=top_k)

    # Separate documents and scores
    relevant_chunks = [doc for doc, score in docs_and_scores]
    scores = [score for doc, score in docs_and_scores]

    # 3. Combine all the relevant chunks into a single text block
    combined_text = "\n".join(chunk.page_content for chunk in relevant_chunks)

    # 4. Use the BART model for summarization
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    num_words = len(combined_text.split())

    if num_words < 20:
        summary = combined_text  # Skip summarization if too short
    else:
        max_len = max(int(num_words * 0.4), 30)
        min_len = max(int(num_words * 0.2), 10)

        if max_len <= min_len:
            max_len = min_len + 10

        max_len = min(max_len, num_words - 1)

        if min_len >= max_len:
            min_len = max(5, max_len // 2)

        result = summarizer(
            combined_text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )
        summary = result[0]["summary_text"]

    # 5. Print chunks and scores for transparency
    print("\nTop-K Chunks and Confidence Scores:")
    for i, (chunk, score) in enumerate(zip(relevant_chunks, scores), 1):
        print(f"\nChunk {i} (Score: {score:.4f}):\n{chunk.page_content}")

    # 6. Return the summary and the chunks with their scores
    return summary, list(zip(relevant_chunks, scores))


# %% [markdown]
# ### Quiz Tool :
# 1. Fetch the most relevant content from a vector database to generate the quiz.
# 2. Combine the retrieved chunks into a single text block
# 3. Send a prompt to the LLM to create a 10-question multiple choice quiz from the input text
# 4. Clean up the raw LLM response string (usually JSON-like) so it can be parsed safely
# 5. Display each question to the user, collect answers, and compare them with the correct answers
# 

# %%
import json
import re

# This function cleans a JSON-like string produced by the LLM.
# Why? LLMs sometimes return extra characters like ```json, trailing commas, etc.,
# which make the response invalid for direct JSON parsing.

def clean_json_like_string(s):
    s = s.strip()
    s = re.sub(r"```json|```", "", s)
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    return s

# This function generates a quiz using the LLM.
# a list of 10 multiple-choice questions based on input text.

def generate_quiz_from_text(text):

    #Prompt the LLM to generate a 10-question multiple-choice quiz based on the input text, and parse the result into a usable Python list.
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

    # Send the prompt to the LLM
    response = llm.invoke(prompt)

   # Extract text from response in a secure way
    response_text = getattr(response, "content", None) or \
                    (response.text() if callable(response.text) else response.text)

    if not isinstance(response_text, str):
        raise ValueError("The LLM response does not contain valid text content.")

  # Trying to convert the text to JSON after cleaning
    try:
        cleaned = clean_json_like_string(response_text)
        quiz = json.loads(cleaned)  # convert to Python object (list of dicts)
        return quiz
    except json.JSONDecodeError as e:
        raise ValueError(f"The response from LLM is not valid JSON: {e}")


# This function runs the quiz in the terminal
# Why? It loops through each question, takes user input, compares answers,
# tracks performance, and builds a feedback summary.

def run_quiz(text):
    quiz_questions = generate_quiz_from_text(text)
    user_answers = []
    correct_answers = []

    print("Welcome to the Quiz!")
    print("======================\n")

    for i, question in enumerate(quiz_questions, 1):
        print(f"{i}. {question['question']}")
        print(f"A. {question['options'][0]}")
        print(f"B. {question['options'][1]}")
        print(f"C. {question['options'][2]}")
        print(f"D. {question['options'][3]}")

        answer = ""
        while answer.upper() not in ["A", "B", "C", "D"]:
            answer = input("Your answer (A/B/C/D): ")

        user_answers.append(answer.upper())
        correct_answers.append(question['answer'].upper())
        print()

    print("Quiz Completed!")
    print("======================\n")

    correct_count = 0
    results = []
    spoken_answers = []  # This could be used with a TTS (Text-to-Speech) engine

    for i, (user_ans, correct_ans) in enumerate(zip(user_answers, correct_answers), 1):
        is_correct = user_ans == correct_ans
        results.append({
            "question_num": i,
            "user_answer": user_ans,
            "correct_answer": correct_ans,
            "is_correct": is_correct
        })
        if is_correct:
            correct_count += 1

        # Build natural language feedback for TTS or reporting
        spoken_answers.append(
            f"Question {i}: {quiz_questions[i-1]['question']}. "
            f"Your answer was: {user_ans}. "
            f"The correct answer is: {correct_ans}. "
            f"{'Well done!' if is_correct else 'That was incorrect.'}"
        )

        # Print feedback for each question
        print(f"{i}. {'Correct' if is_correct else 'Incorrect'} "
            f"(Your answer: {user_ans}, Correct answer: {correct_ans})")


    # Combine everything to return as Answer
    full_spoken_text = "\n\n".join(spoken_answers)

    return {
        "total": len(quiz_questions),
        "correct": correct_count,
        "results": results,
        "Answer": full_spoken_text  
    }


#  Function to run the quiz using all chunks from the vectordb
def run_quiz_from_vectordb(vectordb, query="summarize all content", max_k=1000):
    # We manually set max_k=1000 to control how many chunks to retrieve,
    # because the similarity_search method doesn't automatically fetch "all" chunks.
    # There's no built-in option like `k=None` or `k='all'`, so we choose a large value (e.g., 1000)
    # assuming it covers most practical use cases without overloading memory.
    
    # Use similarity search to retrieve results
    results = vectordb.similarity_search(query, k=max_k)

    # Extract retrieved paragraphs
    retrieved_chunks = [doc.page_content for doc in results]

    # Print the retrieved chunks 
    print("\n Retrieved Chunks:")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"\n Chunk {i}:\n{chunk}")

    # Merge all chunks into one long text
    full_text = "\n\n".join(retrieved_chunks)

    # Run the quiz using the full text
    return run_quiz(full_text)


# %%
tools = [
    Tool(
        name="SummarizerTool",
        func=lambda query: summarizer_tool(query, vectordb),
        description=(
            "Use this tool ONLY if the input is asking to summarize a video. "
            "Examples: 'Summarize the video', etc."
        )
    ),
    Tool(
        name="AnswerQuestionTool",
        func=lambda query: answer_question_tool(query, vectordb, llm),
        description=(
            "Use this tool ONLY if the user is asking a specific question that requires factual retrieval "
            "or detailed answer based on the documents. "
            "Examples: 'What are the main points?', 'What did the speaker say about...', etc."
        )
    ),
    Tool(
        name="QuizTool",
        func=lambda _: json.dumps(run_quiz_from_vectordb(vectordb)),
        description=(
            "Use this tool if the user wants to take a quiz based on the video content. "
            "Trigger if the user says: 'I want to take a quiz', 'Start quiz', etc."
        )
    )

]


#=================================================================================#

# Initialize the agent
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True)

# %%
import nest_asyncio
nest_asyncio.apply()

async def chatbot(agent, vectordb):
    while True:
        video_url = input("Please enter a YouTube URL (or type 'exit' to quit): ")
        if video_url.lower() == "exit":
            print("Exiting the chatbot interface...")
            break

        print("Processing the video...")
        vectordb = process_youtube_video(video_url)

        while True:
            user_input = input("Would you like to ask a Text/Voice question, enter a New URL, or Exit? (Text/Voice/New/Exit): ")

            if user_input.lower() == "text":
                query = input("Enter your question: ")
            elif user_input.lower() == "voice":
                query = recognize_google_cloud()
                if query is None:
                    print("❗ Could not recognize your voice. Please try again.")
                    continue
                print(f"❓ Your question: {query}")
            elif user_input.lower() == "new":
                break
            elif user_input.lower() == "exit":
                print("Exiting the chatbot interface...")
                return
            else:
                print("Invalid input. Please try again.")
                continue

            answer = agent.run(query)

            response_mode = input("How would you like the response? (Text/Voice/Both): ").lower()

            if response_mode == "text":
                print(f"💬 Answer: {answer}")
            elif response_mode == "voice":
                await speak_text(answer)
            elif response_mode == "both":
                print(f"💬 Answer: {answer}")
                asyncio.run(speak_text(answer))
            else:
                print("Invalid response type. Showing text by default.")
                print(f"💬 Answer: {answer}")

            print("="*80)


# %%
# await chatbot(agent, vectordb)

# # %%
# await chatbot(agent, vectordb)

# %% [markdown]
# ---

# %% [markdown]
# # Hallucinations Evaluation (Langsmith)

# %% [markdown]
# How Hallucination is Calculated?
# 
# * The hallucination score is calculated based on a comparison of the agent's answer with the relevant content from the vector database.
# * If the agent's response contains information that is not in the source data, it is considered a hallucination.
# * The hallucination score is a value between 0 and 1, where 0 means the answer is accurate and 1 means the answer contains hallucinated information.

# %% [markdown]
# ### Temperature = 0.7

# %% [markdown]
# **Questation : What is the “technological gaze,” and how does Elise Hu define it?**
# 
# The output from the agent ()
# 
# Hallucination Score: 1.00
# (0 = factual, 1 = hallucinated)
# Warning: High potential for hallucination in this response!
# Reasoning: 1. The criterion in question is: Does this contain information not present in the source?
# 2. To assess this, we need to compare the information provided in the submission to the original input.
# 3. The...
# 
# Answer: Elise Hu defines the "technological gaze" as an algorithmically driven perspective that people learn to internalize, perform for, and optimize for. It is a process where machines, by taking in all our data, learn to perform us in an endless feedback loop. This technological gaze is often experienced through the use of filters or editing tools that alter appearances online, and these altered images can then impact real-world beauty standards. Hu suggests that the technological gaze is creating a gap between how individuals see themselves in reality and how they appear in the digital world, which can lead to a constant need to alter or enhance one's physical appearance in order to keep up with these digitally-influenced standards of beauty.
# 

# %% [markdown]
# ### temperature = 0.2

# %%
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.evaluation import load_evaluator
from langsmith import Client

# initialize the LangSmith client using an API Key

# client = Client(
#     api_key=os.getenv("LANGCHAIN_API_KEY"),
# )

# #  Initialize LLM Model with 0.2 temperature for more deterministic responses, 
# #  before we use 0.7 and the responses were not deterministic after we make the evaluation have to be changed to 0.2

# llm2 = ChatOpenAI(model_name="gpt-4", temperature=0.2)

# # Initialize the Agent

# memory = ConversationBufferMemory(memory_key="chat_history")
# agent_with_steps = initialize_agent(
#     tools=tools,
#     llm=llm2,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     memory=memory,
#     verbose=True,
#     return_intermediate_steps=True,
#     # The LangChainTracer is added to track the agent's steps in LangSmith for monitoring.
#     callbacks=[LangChainTracer(project_name="youtube-qa-hallucination-detection")]
# )

# Create hallucination evaluator
# Load the evaluator from LangChain's evaluation module
# This evaluator is designed to check whether the answer contains any information not present in the source data.

# hallucination_evaluator = load_evaluator(
#     "criteria",
#     llm=llm,
#     criteria={"hallucination": "Does this contain information not present in the source?"}
# )


# Evaluate Hallucinations in the Answer
# The evaluation function first retrieves the relevant context from the vector database 
# by performing a similarity search.
# It then evaluates the agent's answer against this context using the hallucination 
# evaluator to check if the answer contains information not present in the reference context.
# The hallucination score (0 to 1) is extracted from the evaluation result.

# async def evaluate_hallucinations(agent, vectordb, user_query, answer):
#     """
#     Evaluate the agent's answer for hallucinations compared to source content.
    
#     Args:
#         agent: The LangChain agent
#         vectordb: Vector database with source content
#         user_query: User's question
#         answer: Agent's answer
    
#     Returns:
#         Evaluation results dictionary with a numerical hallucination score (0-1)
#     """
#     # Get relevant context from vectordb for the query
#     docs = vectordb.similarity_search(user_query, k= 4)
#     reference_context = "\n\n".join([doc.page_content for doc in docs])
    
#     # Use LangSmith's factuality evaluator
#     eval_result = hallucination_evaluator.evaluate_strings(
#         prediction=answer,
#         reference=reference_context,
#         input=user_query
#     )
    
#     # Extract hallucination score (0-1 scale)
#     hallucination_score = eval_result.get("score", 0)
    
#     # Log to LangSmith
#     # Create a run in LangSmith to track the evaluation
#     run = client.create_run(
#         name=f"Hallucination Check: {user_query[:30]}...",
#         project_name="youtube-qa-hallucination-detection",
#         run_type="llm",  
#         inputs={
#             "query": user_query,
#             "reference_context": reference_context
#         },
#         outputs={
#             "answer": answer,
#             "hallucination_score": hallucination_score,
#             "evaluation": eval_result
#         }
#     )

  
#     return {
#         "hallucination_score": hallucination_score,
#         "reasoning": eval_result.get("reasoning", ""),
#     }

# # %%

# async def chatbot_with_steps(agent, vectordb):
#     while True:
#         video_url = input("Please enter a YouTube URL (or type 'exit' to quit): ")
#         if video_url.lower() == "exit":
#             print("Exiting the chatbot interface...")
#             break
        
#         # Process video and update vectordb
#         vectordb = process_youtube_video(video_url)
        
#         while True:
#             user_input = input("Would you like to ask a Text/Voice question, enter a New URL, or Exit? (Text/Voice/New/Exit): ")
            
#             if user_input.lower() == "text":
#                 query = input("Enter your question: ")
#             elif user_input.lower() == "voice":
#                 query = recognize_google_cloud()
#                 if query is None:
#                     print("Could not recognize your voice. Please try again.")
#                     continue
#                 print(f"Your question: {query}")
#             elif user_input.lower() == "new":
#                 break
#             elif user_input.lower() == "exit":
#                 print("Exiting the chatbot interface...")
#                 return
#             else:
#                 print("Invalid input. Please try again.")
#                 continue
            
#             result = agent_with_steps({"input": query})
#             answer = result["output"]
#             steps = result["intermediate_steps"]
            
#             # Evaluate answer for hallucinations
#             hallucination_results = await evaluate_hallucinations(agent, vectordb, query, answer)
            
#             # Display hallucination score as a number between 0-1
#             hallucination_score = hallucination_results["hallucination_score"]
#             print(f"\nHallucination Score: {hallucination_score:.2f}")
#             print(f"(0 = factual, 1 = hallucinated)")
            
#             # Warning for high hallucination scores
#             if hallucination_score > 0.7:
#                 print("Warning: High potential for hallucination in this response!")
#                 print(f"Reasoning: {hallucination_results['reasoning'][:200]}...\n")
            
#             response_mode = input("How would you like the response? (Text/Voice/Both): ").lower()
#             if response_mode == "text":
#                 print(f"Answer: {answer}")
#             elif response_mode == "voice":
#                 await speak_text(answer)
#             elif response_mode == "both":
#                 print(f"Answer: {answer}")
#                 await speak_text(answer)
#             else:
#                 print("Invalid response type. Showing text by default.")
#                 print(f"Answer: {answer}")
                
#             print("\nSteps taken by the agent:")
#             for i, (action, observation) in enumerate(steps, 1):
#                 print(f"Step {i}:")
#                 print(f"  Tool: {action.tool}")
#                 print(f"  Input: {action.tool_input}")
#                 print(f"  Observation: {observation}")
#             print("="*80)
            

# # Run the chatbot
# await chatbot_with_steps(agent_with_steps, vectordb)

# # %% [markdown]
# # To test for hallucinations, I asked a question that was completely unrelated to the video: **“What does Elise Hu think about the use of AI in education?”** Since this topic was not discussed in the video at all, the correct behavior would be for the agent to acknowledge that the information is not available. The agent responded appropriately, stating that it could not find any mention of Elise Hu’s opinion on AI in education. This shows that the answer contains no hallucination, which is also reflected by the low semantic similarity score (Score = 0.00). Therefore, the system behaved correctly in this case.

# # %%
# # Run the chatbot
# await chatbot_with_steps(agent_with_steps, vectordb)

# %% [markdown]
# Compare with output of llm 0.7:
# 
# The answer returned by the agent provides an accurate explanation of the term "technological gaze", but some parts of the analysis are slightly expanded or rephrased in a more interpretive way than what was explicitly stated in the video.
# While the explanation is logically consistent and aligns with the overall message, certain details — like the “constant need to enhance one’s appearance” — were not directly mentioned in the source. That’s why the system may flag them as “hallucination,” even though the content is contextually reasonable.

# %% [markdown]
# ---

# %% [markdown]
# # Retrieval quality (Recall and precision) 

# # %%
# await chatbot(agent, vectordb)

# # %%
# # the question which is asked to the model to evaluate the retrieval performance 
# # "What is the “technological gaze,” and how does Elise Hu define it?"
# # the ground truth chunks which are relevant to the question asked

# def evaluate_retrieval(retrieved_chunks_ids, ground_truth_ids, k=None):
#   # The value of k refers to the number of chunks you want to evaluate or use in calculating Precision and Recall.
#     if k is not None:
#         retrieved_chunks_ids = retrieved_chunks_ids[:k]

#     relevant_retrieved = [chunk_id for chunk_id in retrieved_chunks_ids if chunk_id in ground_truth_ids]

#     precision = len(relevant_retrieved) / len(retrieved_chunks_ids) if retrieved_chunks_ids else 0
#     recall = len(relevant_retrieved) / len(ground_truth_ids) if ground_truth_ids else 0

#     return precision, recall

# # Example usage:
# retrieved = [1, 2, 5 ,6]      # Model's returned chunk indices
# ground_truth = [1, 2]      # Ground truth chunk indices

# precision, recall = evaluate_retrieval(retrieved, ground_truth, k=4)
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")


# # %% [markdown]
# # * Precision@4 (0.50): Out of the 4 chunks the model retrieved, 50% were relevant (i.e., they matched the ground truth). This means that half of the chunks the model retrieved were useful for answering the question.
# # 
# # * Recall@4 (1.00): The model retrieved 100% of the relevant chunks that should have been included, meaning it retrieved all the relevant information from the ground truth.

# # %%
# index = pc.Index(index_name)
# index.describe_index_stats()

# # %%



