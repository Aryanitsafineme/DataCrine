# 🧹 Datacrine: A Modular, Interpretable AI Framework for Smart Data Cleaning

Datacrine is a research prototype designed for my Bachelor's thesis (built to Master's/PhD quality).  
It integrates **traditional preprocessing techniques** (imputation, outlier handling) with **LLM-powered semantic anomaly detection** to evaluate the impact of cleaning on downstream machine learning performance.  

The framework is modular, interpretable, and export-ready for both academic and applied use.  

---

## ✨ Features
- ✅ **Modular pipeline** for tabular and text datasets  
- ✅ **Traditional cleaning**: mean imputation, IQR-based outlier capping  
- ✅ **Semantic rules** for domain-specific validation (e.g., unrealistic ages or credit limits)  
- ✅ **LLM-based anomaly detection** using HuggingFace Transformers  
- ✅ **Audit logging** for interpretability (all cleaning decisions recorded)  
- ✅ **Machine learning evaluation** (Random Forest + XGBoost)  
- ✅ **Export**: save cleaned datasets and full audit logs  

---

## 📊 Datasets Used
Datacrine is tested on two benchmark datasets:  
- **UCI Credit Card Default Dataset** (tabular)  
- **IMDb 50k Movie Reviews** (text sentiment)  

Both datasets are automatically fetched via APIs (`ucimlrepo`, `kagglehub`).  

---

## 🏗️ Installation

Clone the repository:

```bash
git clone https://github.com/Aryanitsafineme/datacrine.git
cd datacrine

pip install -r requirements.txt
from datacrine import Datacrine
import pandas as pd

# Initialize
dc = Datacrine()

# --- UCI Example ---
from ucimlrepo import fetch_ucirepo
uci_data = fetch_ucirepo(id=350)
uci_credit = pd.concat([uci_data.data.features, uci_data.data.targets], axis=1)

uci_cleaned = dc.impute_numeric_mean(uci_credit.copy(), ["X5"])
uci_cleaned = dc.cap_outliers_iqr(uci_cleaned, ["X1"])
uci_cleaned = dc.uci_semantic_rules(uci_cleaned)
uci_cleaned["Y"] = uci_credit["Y"].values

uci_results = dc.train_and_evaluate(uci_cleaned, target_col="Y", dataset="uci")
print("UCI Results:", uci_results)

# --- IMDb Example ---
import kagglehub
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
imdb = pd.read_csv(f"{path}/IMDB Dataset.csv")

imdb_subset = imdb.sample(2000, random_state=42).copy()
imdb_cleaned = dc.imdb_label_check(imdb_subset)
imdb_cleaned["Y"] = imdb_subset["sentiment"].map({"positive": 1, "negative": 0})

imdb_results = dc.train_and_evaluate(
    imdb_cleaned.rename(columns={"llm_sentiment":"Y"}), 
    target_col="Y", 
    dataset="imdb"
)
print("IMDb Results:", imdb_results)

# --- Export ---
dc.export_logs("datacrine_audit.txt")
dc.save_dataset(uci_cleaned, "uci_credit_cleaned.csv")

👤 Author

Aryan Singh
B.Sc. Data Science & AI @ GISMA Business School Potsdam
Thesis: "Datacrine: A Modular, Interpretable AI Framework for Smart Data Cleaning"
