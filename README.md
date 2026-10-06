# 🍳 Smart Recipe & Pantry Planner

A simple Streamlit app that helps you track your pantry, avoid food waste, and decide what to cook.

## Features

- 🥫 Pantry tracking (edit, add, and delete items)
- ⚠️ Alerts for expired, expiring-soon, and low-stock items
- 📖 Recipe suggestions based on what you already have
- 🤖 AI recipes and weekly meal plans using a local Ollama model
- 🛒 Automatic shopping list (downloadable as CSV)
- 📡 Simulated smart-shelf sensor updates

## Requirements

- Python 3.9 or newer
- [Ollama](https://ollama.com) (only needed for the AI features)

Python packages:

- streamlit
- langchain
- langchain-community
- langchain-ollama
- pandas

## Setup

1. **Put `app.py` and `requirements.txt` in one folder**, then open a terminal there.

2. **(Optional) Create a virtual environment**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac / Linux
   source venv/bin/activate
   ```

3. **Install the packages**

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Set up Ollama for the AI features**

   Install Ollama, then download a model:

   ```bash
   ollama pull llama3.2
   ```

   Keep Ollama running in the background.

## Run the app

```bash
streamlit run app.py
```

The app opens in your browser at http://localhost:8501.

## How to use

| Tab | What it does |
|---|---|
| 🥫 Pantry | View and edit items, add new ones, see alerts |
| 📖 Recipe Matches | Recipes ranked by how many ingredients you already have |
| 🤖 AI Chef | Generate a custom recipe with Ollama |
| 📅 Meal Planner | Create a 1–7 day meal plan (rule-based or AI) |
| 🛒 Shopping List | Auto-built list of low/expired items, plus ingredients for chosen recipes |

Use the **sidebar** to set diet preference, max cooking time, servings, and the Ollama model name.

## Troubleshooting

- **"LangChain / Ollama packages are not installed"** → run `pip install -r requirements.txt` again.
- **"Could not reach Ollama"** → make sure Ollama is running and the model is pulled (`ollama pull llama3.2`). The model name in the sidebar must match.
- **App works without AI?** Yes. Pantry, recipe matching, rule-based meal plans, and the shopping list all work without Ollama.

## Notes

The smart-shelf sensor data is **simulated**. No physical IoT device is needed.
