# %%
# ! pip install pytube youtube-transcript-api langchain openai whisper chromadb langchain-community
# ! pip install -U openai-whisper
# ! pip install -U langchain-openai
# ! pip install yt-dlp
# ! pip install --upgrade langchain
# ! pip install pinecone

# %%
# ! pip uninstall openai
# ! pip install openai==0.28

# %%
# ! pip install -U langchain-pinecone
# ! pip install google-cloud-speech pyaudio
# ! pip install python-dotenv
# ! pip install whisper
# ! pip install transformers

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
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# from google.colab import userdata
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import Pinecone
from langchain.agents import initialize_agent, Tool
from langchain_community.llms import OpenAI
from langchain.agents import AgentType
from langchain_community.vectorstores import FAISS
from transformers import pipeline
import os
import pyaudio
from google.cloud import speech

# Step 1: User inputs API keys

# OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# PINECONE_API_KEY = userdata.get('PINECONE_API_KEY')
# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# %% [markdown]
# # Build the Vector Database (Pinecone)

# %%
import os
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone instance
pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index_name = 'youtube-video-index00'

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
index.describe_index_stats()

# %%
# pc.delete_index(index_name)

# %%
index = pc.Index(index_name)

# Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Pinecone database
vectordb = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")


# %% [markdown]
# # Extract and Process Video Content
# 
# 
# 1.   Extract Video ID from YouTube URL
# 2.   Download YouTube Audio
# 3.    Transcribe Audio using Whisper API
# 4. Store Transcript in Pinecon
# 
# 
# 

# %%
def save_chunks_to_txt(chunks, filename="split_chunks.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"Chunk {i+1}:\n{chunk}\n\n")

def save_tra_to_txt(chunks, filename="split_chunks.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"Chunk {i+1}:\n{chunk}\n\n")

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter

def store_transcript_in_pinecone(transcript, video_title, video_id):
    all_chunks = []

    full_text = " ".join([segment['text'] for segment in transcript])


    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=200, chunk_overlap=50
    )

    # splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=32)

    token_chunks = splitter.split_text(full_text)

    for chunk in token_chunks:
        all_chunks.append({
            "text": chunk
        })

    texts = [chunk['text'] for chunk in all_chunks]
    vectors = embeddings.embed_documents(texts)
    save_chunks_to_txt(texts, filename=f"split_chunks.txt")

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
    return vectordb


# %%
# Extract Video ID from YouTube URL
def extract_video_id(youtube_url):
    parsed_url = urlparse(youtube_url)
    if 'youtube.com' in parsed_url.netloc:
        # Long link
        query = parse_qs(parsed_url.query)
        return query['v'][0]
    elif 'youtu.be' in parsed_url.netloc:
        # Short link
        video_id = parsed_url.path.lstrip('/')
        # Remove any query parameters if they exist
        if '?' in video_id:
            video_id = video_id.split('?')[0]
        return video_id
    else:
        raise ValueError("Invalid YouTube URL format")

# Download YouTube Audio
def download_youtube_audio(youtube_url):
    try:
        os.makedirs("./youtube_audio", exist_ok=True)
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

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            filename = f"./youtube_audio/{info['id']}.mp3"
            return filename, info.get('title', None)
    except Exception as e:
        print(f"Error downloading YouTube audio: {e}")
        return None, None


def transcribe_audio_with_timestamps(audio_file_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file_path, word_timestamps=True)

    segments = result['segments']  # Each segment contains text + start time + end time
    transcript_chunks = []

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
# def process_youtube_video(youtube_url):
#     video_id = extract_video_id(youtube_url)
#     print(f"Processing video with ID: {video_id}")

#     audio_file, video_title = download_youtube_audio(youtube_url)
#     if audio_file:
#         transcript_chunks = transcribe_audio_with_timestamps(audio_file)

#     if transcript_chunks:
#         vectordb = store_transcript_in_pinecone(transcript_chunks, video_title, video_id)

#         print(f"Successfully processed video: {video_title}")
#         return vectordb
#     else:
#         print("Failed to process video")
#         return None

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


# %%
def recognize_google_cloud():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"

    client = speech.SpeechClient()

    RATE = 16000
    CHUNK = int(RATE / 10)

    audio_interface = pyaudio.PyAudio()
    stream = audio_interface.open(format=pyaudio.paInt16,
                                   channels=1,
                                   rate=RATE,
                                   input=True,
                                   frames_per_buffer=CHUNK)

    print("Speak now...")

    audio_frames = []
    for _ in range(0, int(RATE / CHUNK * 5)):  # 5 seconds
        data = stream.read(CHUNK)
        audio_frames.append(data)

    stream.stop_stream()
    stream.close()
    audio_interface.terminate()

    audio_data = b''.join(audio_frames)

    audio = speech.RecognitionAudio(content=audio_data)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    if not response.results:
        print("No speech detected.")
        return None

    transcript = response.results[0].alternatives[0].transcript
    print("Transcript:", transcript)
    return transcript

import os
import asyncio
import edge_tts
import edge_tts
import asyncio
from playsound import playsound 
from IPython.display import Audio, display

# Text to Speech using Edge TTS 
async def speak_text(text):
    filename = "output.mp3"
    if os.path.exists(filename):
        try:
            os.remove(filename)
        except PermissionError:
            print("The file 'output.mp3' cannot be deleted. Make sure it is not open in any audio player.")

            return

    # Speech generation
    communicate = edge_tts.Communicate(text, "en-US-JennyNeural")
    await communicate.save(filename)
    print("Response saved as 'output.mp3' ")

   # automatically play the sound file
   # playsound("output.mp3")

    # manually play the sound file
    display(Audio(filename))   




# %% [markdown]
# # Question Answering using LangChain

# %% [markdown]
# 1. Question About youtube video stored in vectordb
# 2. Embed the Question
# 3. Query Top-K matches from vectordb
# 4. Send retrieved texts to the LLM
# 5. Generate a full answer
# 6. Display answer + sources
# 

# %%
# Set up the LLM (Language Model)
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

#=================================================================================#

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

#=================================================================================#

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
#=================================================================================#
import json
import re

def clean_json_like_string(s):
    s = s.strip()
    s = re.sub(r"```json|```", "", s)
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    return s

def generate_quiz_from_text(text):
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

   # Extract text from response in a secure way
    response_text = getattr(response, "content", None) or \
                    (response.text() if callable(response.text) else response.text)

    if not isinstance(response_text, str):
        raise ValueError("The LLM response does not contain valid text content.")

  # Trying to convert the text to JSON after cleaning
    try:
        cleaned = clean_json_like_string(response_text)
        quiz = json.loads(cleaned)
        return quiz
    except json.JSONDecodeError as e:
        raise ValueError(f"The response from LLM is not valid JSON: {e}")

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
        print(f"{i}. {'Correct' if is_correct else 'Incorrect'} "
              f"(Your answer: {user_ans}, Correct answer: {correct_ans})")

    print(f"\nYou got {correct_count} out of {len(quiz_questions)} correct.")

    return {
        "total": len(quiz_questions),
        "correct": correct_count,
        "results": results
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

    # Print the retrieved chunks (optional)
    print("\n Retrieved Chunks:")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"\n Chunk {i}:\n{chunk}")

    # Merge all chunks into one long text
    full_text = "\n\n".join(retrieved_chunks)

    # Run the quiz using the full text
    return run_quiz(full_text)


#=================================================================================#

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
    verbose=True
)

# %%
def chatbot(agent , vectordb):
    while True:
        # Ask the user to enter a YouTube URL or exit
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
                    print("Could not recognize your voice. Please try again.")
                    continue
            elif user_input.lower() == "new":
                # Break inner loop to allow new video URL
                break
            elif user_input.lower() == "exit":
                print("Exiting the chatbot interface...")
                return
            else:
                print("Invalid input. Please try again.")
                continue

            # Use the agent to get the answer
            answer = agent.run(query) 
            print(f"Answer: {answer}")
            print("=================================================================================")


# # %%
# chatbot(agent, vectordb)

# # %%
# chatbot(agent, vectordb)

# # %%
# chatbot(agent, vectordb)

# # %%
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



