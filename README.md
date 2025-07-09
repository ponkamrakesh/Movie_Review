# 🎬 Movie Review Sentiment Analyzer

A smart web app that analyzes movie reviews and predicts whether the sentiment is **Positive** or **Negative**.  
Built using **FastAPI**, **GRU-based Deep Learning**, and deployed with **Hugging Face Spaces**, all wrapped in an automated **CI/CD pipeline** using **GitHub Actions**.

---

## 🚀 What It Does

- ✅ Takes user input (movie review text)
- 🧠 Uses a trained GRU-RNN model to predict the sentiment
- 🔁 Automatically retrains the model if `dataset/` or `Model_Training.py` changes
- 🚀 Seamlessly redeploys to Hugging Face when any file updates

---

## 🧠 Tech Stack

| Layer       | Tools/Tech Used                        |
|-------------|-----------------------------------------|
| 🧠 Model     | GRU-based RNN (Keras + TensorFlow)      |
| 🧪 Training  | Python, IMDB-style Dataset, Pickle      |
| ⚙️ Backend   | FastAPI + Uvicorn                       |
| 🖥️ Frontend | Static HTML/CSS (served by FastAPI)     |
| ☁️ Hosting   | Hugging Face Spaces                    |
| 🔄 CI/CD     | GitHub Actions                          |

---

## 🗂️ Project Structure

Movie_Review_Analyzer/
├── dataset/ # CSV dataset for training
├── model/ # Exported .h5 model & tokenizer.pkl
├── static/ # Frontend HTML + CSS
│ └── index.html
├── main.py # FastAPI backend
├── Model_Training.py # Model training pipeline
├── requirements.txt # Python dependencies
├── .gitignore
├── .github/
│ └── workflows/
│ └── AutoTrain-and-Deploy.yml
└── README.md # You're reading this!



---

## 🔁 CI/CD Pipeline Summary

Your GitHub Actions workflow (`AutoTrain-and-Deploy.yml`) does this:

| 🔍 Condition                                  | 🛠️ Action                                            |
|----------------------------------------------|------------------------------------------------------|
| `dataset/` or `Model_Training.py` changed     | ✅ Retrain model, export `.h5` & `.pkl`, push to HF   |
| Only UI/backend files changed (`main.py`, `static/`) | 🚀 Rebuild and redeploy to Hugging Face Space         |

### ✅ Automatically pushes all updated files (model + code + HTML) to Hugging Face.

---

## 🧪 Local Setup Instructions

```bash
# Step 1: Clone the Repo
git clone https://github.com/ponkamrakesh/Movie_Review
cd Movie_Review

# Step 2: Create Virtual Environment
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Step 3: Install Dependencies
pip install -r requirements.txt

# Step 4: Train Model (optional if already trained)
python Model_Training.py

# Step 5: Run the Backend
uvicorn main:app --reload


🚀 Hugging Face Deployment
Live App:
🔗 https://huggingface.co/spaces/RakeshPonkam07/Movie_Review_Sentiment

Once you push to GitHub, the GitHub Actions workflow:
