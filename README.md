# Multimodal AI Chatbot for YouTube Videos 🎬🪄  
> "Ask questions. Get answers. Learn from any video — instantly and interactively."
...

## 📌 Overview

This project implements a **multimodal AI chatbot** that can answer questions about YouTube videos using both **text and voice inputs**.  
Users can:
- 🎯 Ask questions via voice or text
- 📑 Get text or audio responses
- 🧠 Take quizzes or generate summaries
---

## 🔄 System Flow
1. The user enters a YouTube video URL  
2. Chooses **input mode**: text or voice  
3. Chooses **output mode**: text or voice  
4. Backend:
   - Preprocessing the video
   - Stores the video URL in a database
   - Passes the question to the **agent**
   - Agent selects the appropriate tool (search, summary, quiz, etc.)
   - LLM generates a response
5. The response is returned to the user as text or speech

![Flowchart Diagram](assets/FlowChart.png)

---
---

## Evaluation
* **Hallucination Detection:**
   - Compares the LLM's response with source data.
   - Calculates a hallucination score (0 = factual, 1 = fully hallucinated).

* **Retrieval Evaluation:**
   - Precision@K and Recall@K are computed to evaluate the relevance of retrieved content.

---

## Setup Instructions

### Requirements
Make sure you have the following installed:

```bash
pip install -r requirements.txt
```

### Required API Keys:
- OpenAI API Key
- Pinecone API Key
- Google Cloud credentials (for ASR)
- LangChain (LangSmith) API Key

### Environment Variables
Configure the following in your `.env` file or export them:
```
OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key
LANGCHAIN_API_KEY=your_key
GOOGLE_APPLICATION_CREDENTIALS=your_credentials.json
```
---

## Usage Guide

1. **Run the chatbot**
```bash
python Multimodal_YouTube_Bot_ InferaTube.py
```

Example Output:
```
Question: What is the “technological gaze,” and how does Elise Hu define it?

Answer: The "technological gaze" is defined by Elise Hu as an algorithmically driven perspective that people learn to internalize, perform for, and optimize for. It is a process where machines take in our data and learn to perform us in an endless feedback loop. This concept manifests itself in how we present ourselves online, often through the use of filters or editing tools to alter our appearance, driven in part by AI-generated beauty standards. Hu mentions how these digital alterations can impact real-world beauty standards, creating a gap between our real appearances and the filtered images we present online. The technological gaze can thus contribute to issues of self-image and perpetuates a narrow, potentially harmful standard of beauty.

📄 Source Chunks and Relevance Scores
```
---

## How to Run the Code

### 1. Clone the Repository  
```bash
Start by cloning the repository to your local machine or environment:
git clone <repository-url>
cd <repository-folder>
```
### 2. Install Required Libraries  
```bash
Install all necessary libraries using pip:
pip install -r requirements.txt
```
### 3. Set Up the API Key for OpenAI 
```bash
Make sure to set up your OpenAI API key in the environment. You can do this by adding it to the script or configuring it as an environment variable.
export OPENAI_API_KEY="your_api_key"
```
Alternatively, you can pass the API key directly in the script, as shown in the provided code.

### 4. Run the Code  
After setting up your environment, you can run the code to classify reviews, cluster products, and generate summaries. Execute the following:

```bash
python main.py
```


## Demo

🔗(  ) This is our project for Multimodal AI Chatbot for YouTube Videos. Feel free to try it out and share it with your friends!
  

<br>


