# AI Data Processor

A smart data analysis tool that automatically cleans CSV datasets, generates statistical summaries, and uses **Google Gemini 2.5 Flash** to provide actionable business insights.

## Live Demo
**[Try the App Here](https://zyndrex-data-analyzer.streamlit.app)**

<img width="1435" height="810" alt="Screenshot 2025-12-11 at 9 19 21 AM" src="https://github.com/user-attachments/assets/961b0bd7-b75c-439e-a94c-2d8d6ebaa088" />

## Features

* **Automated Cleaning:** Instantly fills missing values and standardizes numerical data.
* **Statistical Analysis:** Generates descriptive statistics (mean, median, std dev) automatically.
* **AI Analyst Agent:** Uses the **Gemini Flash** model to interpret data trends and provide 3 key business insights.
* **Database Persistence:** Automatically saves analysis results to a local SQLite database for persistent storage across sessions.
* **Privacy Focused:** Your data is processed locally; only the statistical summary is sent to the LLM.

## Tech Stack

* **Python 3.9+**
* **Streamlit** (Frontend)
* **Pandas** (Data Processing)
* **Google Gemini API** (LLM Engine)
* **SQLite** (Relational Database / Persistence)
  
## Run Locally (For Developers)

Follow these steps to run the project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/Dercypt/ai-data-processor.git
cd ai-data-processor
```

### 2. Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Secrets & Database Configuration
This project uses **Google Gemini** for AI insights and **SQLite** for relational database persistence.

1. **Google Gemini API Key (Required for AI insights):**
   - Create a folder named `.streamlit` in the root directory and create a `secrets.toml` file:
     ```toml
     GOOGLE_API_KEY = "Your-Gemini-Key-Here"
     ```
   - Alternatively, export the API key as an environment variable or define it in `.env`:
     ```bash
     export GOOGLE_API_KEY="Your-Gemini-Key-Here"
     ```

2. **SQLite Database Path (Optional):**
   - By default, the application persists analysis history locally to `history.db` in the root directory.
   - You can optionally customize the SQLite database path by setting the `SQLITE_DB_PATH` environment variable or defining it in `.streamlit/secrets.toml`:
     ```bash
     export SQLITE_DB_PATH="path/to/custom_history.db"
     ```
     or in `.streamlit/secrets.toml`:
     ```toml
     SQLITE_DB_PATH = "path/to/custom_history.db"
     ```
     
### 5. Run the Application
```bash
python -m streamlit run src/main.py
```

## Project Structure

```bash
ai-data-processor/
├── assets/             # Images and logos
├── src/
│   ├── main.py         # Streamlit UI Entry point
│   ├── analyzer.py     # Pandas data processing logic
│   ├── database.py     # SQLite persistence and history manager
│   └── llm_service.py  # Google Gemini AI integration
├── .gitignore          # Protected files (API keys, etc.)
├── LICENSE             
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

## Contributing 

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the [MIT License](LICENSE).
