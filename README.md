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
