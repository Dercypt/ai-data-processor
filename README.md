# AI Data Processor

A smart data analysis tool that automatically cleans CSV datasets, generates statistical summaries, and uses **Google Gemini 2.0 Flash** to provide actionable business insights.

## Demo

https://github.com/user-attachments/assets/36cce009-9066-41b9-bd03-573f93de7de3

## Features

* **Automated Cleaning:** Instantly fills missing values and standardizes numerical data.
* **Statistical Analysis:** Generates descriptive statistics (mean, median, std dev) automatically.
* **AI Analyst Agent:** Uses the **Gemini Flash** model to interpret data trends and provide 3 key business insights.
* **Privacy Focused:** Your data is processed locally; only the statistical summary is sent to the LLM.

## Tech Stack

* **Python 3.9+**
* **Streamlit** (Frontend)
* **Pandas** (Data Processing)
* **Google Gemini API** (LLM Engine)

## Installation & Setup

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

### 4. Set Up API Keys
This project uses Google's Gemini API (Free Tier).
  1. Get your free API key here: [Google AI Studio](https://aistudio.google.com/api-keys).
  2. Create a `.env` file in the root directory:
     ```bash
     touch .env  # Mac/Linux
     # Or manually create a file named .env
     ```
  3. Add your key to the file:
     ```bash
     GOOGLE_API_KEY=AIzaSyYourKeyHere...
     ```
     
### 5. Run the Application
```bash
python -m streamlit run src/main.py
# OR
streamlit run src/main.py
```

## Project Structure

```bash
ai-data-processor/
├── assets/             # Images and logos
├── src/
│   ├── main.py         # Streamlit UI Entry point
│   ├── analyzer.py     # Pandas data processing logic
│   └── llm_service.py  # Google Gemini AI integration
├── .gitignore          # Protected files (API keys, etc.)
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

## Contributing 

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the [MIT License](LICENSE).
