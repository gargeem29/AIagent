# AIagent
## (Assignment Task)

AIagent is a Python-based project that uses an AI agent to parse a dataset (CSV or Google Sheets) and perform web searches to retrieve specific information for each entity in a chosen column. The project leverages a Large Language Model (LLM) to analyze web search results and extract relevant data, which is then formatted into structured output. A simple dashboard is built to allow users to upload a file, define search queries, and view or download the results.

## Features

- **Dataset Upload**: Users can upload CSV or Google Sheets files to be processed by the AI agent. (`movies.csv`).
- **Web Search**: The AI agent performs web searches for each entity in a selected column..
- **Query Definition**: Users can define search queries to guide the AI in retrieving relevant information.
- **LLM Parsing**: A Large Language Model (LLM) analyzes and parses the search results.
- **Structured Output**: Extracted data is formatted into a structured output.
- **Dashboard**: A simple web dashboard built with Streamlit, where users can upload files, define search queries, and view or download results. 

## Dependencies

- `streamlit`
- `pandas`
- `numpy`
- `faiss`
- `pickle`
- `sentence-transformers`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gargeem29/AIagent.git

2. Install dependencies:
   ```bash
   pip install streamlit pandas numpy faiss-cpu sentence-transformers
3. Run the main script
   ```bash
   python -m streamlit run main.py




   ![image](https://github.com/user-attachments/assets/f1ff697e-6fba-444f-aaf3-3538dd5be1bc)

   ![image](https://github.com/user-attachments/assets/013b2d7e-24d4-4a5e-8cea-9adbda60c901)

   ![image](https://github.com/user-attachments/assets/91d746bb-7587-4728-a726-eb1a6a9a434d)

   ![image](https://github.com/user-attachments/assets/9957ffe6-8213-4ad9-a103-e0b58f84ea8a)


## License
This project is licensed under MIT 




