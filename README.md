Project Summary
* GenAIRAG app runs end to end locally 100% using RAG & ollama
* Compatible with CPU/GPU
* Prompt answered in two ways:
* 1.if answer found in the local vector database, llm will use it for answer
* 2.if answer not found in local vector database, llm will use its trained trained material for answer.

Setup & Codes Running
* Install anaconda
* Open anaconda prompt
* Create a virtual env
* Activate virtual env
* Run pip install requirement file in virtual env
* Checkout this code to a project directory 
* 1. run create_RAG_db.py to create a RAG_Vector database
  2. launch the app in command prompt: python query_RAG_db.py'
