# Sentiment Analysis Research Project

**Explainable Fine-Grained Aspect-Based Sentiment Analysis for Government Mobile App Reviews**

---

## **Project Overview**

This repository contains the complete codebase, data artifacts, and analysis notebooks for a large-scale Aspect-Based Sentiment Analysis (ABSA) research project applied to government mobile application reviews. The project integrates modern transformer-based sentiment modeling with explainable AI (XAI) techniques to ensure transparency, interpretability, and reproducibility.

The pipeline spans data scraping → preprocessing → model inference → evaluation → explanation quality analysis, and is organized to reflect a clean research workflow rather than a single monolithic script.

---

## **Research Goals**

* Perform fine-grained sentiment classification at the *review–aspect* level
* Apply transformer-based models for scalable inference
* Quantitatively evaluate both model performance and explanation quality
* Use XAI methods to understand why predictions are made
* Produce a reproducible, and well-documented research artifact

---

## **Repository Structure**

```
Sentiment-Analysis-Research-Project/
│
├── app_scraper/
│   ├── App_IDs_List.xlsx          # Reference list of app IDs and metadata
│   ├── batch_scrape.py            # Batched multi-app review scraping
│   ├── scrape_reviews.py          # Individual app review scraping logic
│   ├── merge_batch_results.py     # Merge per-app scrape outputs
│   ├── scraper_requirements.txt   # Dependencies for scraping environment
│   └── terminal_cmds_guide.md     # CLI usage and scraping instructions
│
├── data/
│   └── fabsa/
│       ├── fabsa_dataset.csv      # Full normalized FABSA dataset
│       ├── train.csv              # Review-level training split
│       ├── dev.csv                # Review-level validation split
│       ├── test.csv               # Review-level test split
│       ├── train_pairs.csv        # Expanded review–aspect training pairs
│       └── dev_pairs.csv          # Expanded review–aspect validation pairs
│
├── notebooks/
│   ├── Sentiment_Analysis_Project_FABSA_Section.ipynb
│   │   # Core FABSA modeling, inference, and evaluation
│   │
│   ├── Sentiment_Analysis_Project_XAI_Section.ipynb
│   │   # Explainability analysis using Integrated Gradients and XAI metrics
│   │
│   └── report_statistics.ipynb
│       # Aggregate statistics, plots, and reporting tables
│
├── Full Version Prediction XAI and Gemini Sample.xlsx
│   # Sample predictions with XAI explanations and LLM-based justifications
│
├── requirements.txt               # Main project dependencies
├── README.md                      # Project documentation
├── .gitignore
└── .gitattributes
```

---

## **Methodology Summary**

### **1. Data Collection**

* Reviews scraped from Google Play Store and Apple App Store
* Batched scraping across 68 government applications
* Unified schema across platforms
* Stored as structured CSV files for reproducibility

### **2. Dataset Construction**

* Reviews expanded into review–aspect pairs
* Train / validation / test splits created at both review and pair levels

### **3. Sentiment Modeling**

* Transformer-based sentiment classifier (fine-tuned)
* Multi-class sentiment prediction per aspect
* Batched inference for scalability

### **4. Evaluation**

* Accuracy
* Micro-F1 and Macro-F1
* Cohen’s κ (agreement beyond chance)
* Precision and Recall

### **5. Explainable AI (XAI)**

* Token-level attributions using Integrated Gradients (Captum)
* Quantitative explanation quality metrics:
  * Comprehensiveness
  * Sufficiency
  * Monotonicity
  * Sparsity
  * Local Sensitivity
  * Polarity Alignment
* Gemini Natural Langauge Justifications
* Explanation outputs stored alongside predictions for auditability

---

## **Key Outputs**

* FABSA predictions at scale
* Explanation quality metrics validating faithfulness and stability
* Example outputs combining model predictions, XAI, and LLM-based reasoning

---

## **Reproducibility Notes**

1. Install dependencies:

```
pip install -r requirements.txt
```

1. 
2. Scraping (optional, if reproducing raw data):
   * See app_scraper/terminal_cmds_guide.md
3. Run notebooks in order:
   * FABSA section
   * XAI section

Some large artifacts (models and intermediate outputs) may be generated dynamically or referenced externally.

---

## **Ethical Considerations**

* All data is publicly available user-generated content
* No personal or identifying information is collected
* Explainability is explicitly included to support responsible AI use, particularly in public-sector applications

---

## **Author**

**Taksh Girdhar**

BSc Data Science — University of British Columbia (Okanagan)

DATA 448 — Research Project

---
