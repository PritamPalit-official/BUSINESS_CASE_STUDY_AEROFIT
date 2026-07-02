<div align="center">

# 🏋️ Aerofit Business Case Study

### Customer Profiling & Purchase Behavior Analysis for Aerofit Treadmills

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-444876?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

*A comprehensive data analytics project analyzing customer purchasing behavior across Aerofit's treadmill product line to drive targeted marketing strategies and boost conversion rates.*

</div>

---

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [❓ Business Problem](#-business-problem)
- [📊 Dataset Description](#-dataset-description)
- [🔬 Analysis Performed](#-analysis-performed)
- [🛠️ Tools & Technologies](#️-tools--technologies)
- [💡 Key Business Insights](#-key-business-insights)
- [📈 Strategic Recommendations](#-strategic-recommendations)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [👤 Author](#-author)
- [📄 License](#-license)

---

## 🎯 Project Overview

This project analyzes **customer purchasing behavior** for Aerofit treadmill products using data analytics techniques. The goal is to understand how **demographic and fitness-related attributes** influence product selection and to provide **actionable insights** for marketing, pricing, and product strategy.

> **Objective:** Identify distinct customer profiles for each treadmill model to enable data-driven marketing decisions and improve overall business performance.

---

## ❓ Business Problem

Aerofit aims to improve its **marketing efficiency** and **product recommendations** by identifying distinct customer profiles for its three treadmill models:

| Model | Segment | Target Audience |
|:-----:|:-------:|:----------------|
| **KP281** | Entry-Level | Budget-conscious, younger customers |
| **KP481** | Mid-Range | Moderate fitness enthusiasts |
| **KP781** | Premium | High-income, fitness-oriented users |

The analysis focuses on understanding customer segments based on **age, income, gender, fitness level, and usage behavior** to craft precise, targeted marketing strategies.

---

## 📊 Dataset Description

The dataset contains **180 customer records** with **9 features**. The dataset is complete with **no missing values** and includes both numerical and categorical variables.

| # | Feature | Type | Description |
|:-:|:--------|:----:|:------------|
| 1 | **Product** | Categorical | Treadmill model purchased (KP281 / KP481 / KP781) |
| 2 | **Age** | Numerical | Age of the customer (in years) |
| 3 | **Gender** | Categorical | Male / Female |
| 4 | **Education** | Numerical | Number of years of education |
| 5 | **Marital Status** | Categorical | Single / Partnered |
| 6 | **Usage** | Numerical | Planned treadmill usage per week (times) |
| 7 | **Fitness** | Numerical | Self-rated fitness level (1–5 scale) |
| 8 | **Income** | Numerical | Annual household income (USD) |
| 9 | **Miles** | Numerical | Expected miles to walk/run per week |

---

## 🔬 Analysis Performed

The project follows a structured analytical approach covering multiple dimensions:

| Analysis Type | Description |
|:--------------|:------------|
| 📌 **Exploratory Data Analysis** | Data structure, distributions, and summary statistics |
| 📌 **Non-Graphical Analysis** | Grouped statistics and contingency tables |
| 📌 **Univariate Analysis** | Age and income distributions across customer segments |
| 📌 **Bivariate Analysis** | Product vs. income, usage frequency, and fitness level |
| 📌 **Probability Analysis** | Marginal and conditional probabilities for product purchase behavior |
| 📌 **Correlation Analysis** | Relationships between income, usage, fitness, and product selection |

---

## 🛠️ Tools & Technologies

| Category | Tools |
|:---------|:------|
| **Language** | Python |
| **Data Manipulation** | Pandas, NumPy |
| **Data Visualization** | Matplotlib, Seaborn |
| **Environment** | Jupyter Notebook |

---

## 💡 Key Business Insights

The analysis reveals **clear customer segmentation** across product lines:

- 🔹 **Entry-level models** attract younger, budget-conscious customers with moderate fitness levels.
- 🔹 **Premium models** are strongly preferred by high-income and fitness-oriented users.
- 🔹 **Income, usage frequency, and fitness level** show the strongest influence on treadmill choice.
- 🔹 Distinct demographic patterns enable **precise targeted marketing strategies**.

---

## 📈 Strategic Recommendations

| # | Recommendation | Impact |
|:-:|:---------------|:-------|
| 1 | **Target high-income, fitness-focused customers** for premium models via professional platforms (LinkedIn, fitness apps) | Higher conversion for KP781 |
| 2 | **Promote entry-level treadmills** through affordability-focused campaigns on social media | Wider market reach for KP281 |
| 3 | **Introduce upgrade & trade-in offers** to move customers up the product ladder | Increased customer lifetime value |
| 4 | **Bundle products & offer EMI options** to reduce purchase friction | Higher overall conversion rates |

---

## 📁 Project Structure

```
BUSINESS_CASE_STUDY_AEROFIT/
│
├── 📁 data/
│   └── 📊 aerofit_treadmill.csv          # Treadmill customer dataset
├── 📁 images/                            # Exported analysis visualizations
├── 📄 app.py                             # Interactive Streamlit dashboard
├── 📄 requirements.txt                   # Dependency definitions
├── 📓 Aerofit_Descriptive_Stats.ipynb    # Descriptive stats Jupyter notebook
├── 🐍 Aerofit_Descriptive_Stats.py       # Python counterpart of the notebook
├── 📑 BUSINESS_CASE_STUDY.pdf            # Detailed analysis report
└── 📜 LICENSE                            # MIT License
```

---

## 🚀 Getting Started

### 🖥️ Running the Interactive Dashboard (Streamlit)

To view the live interactive web dashboard:

1. **Clone the repository and navigate inside**:
   ```bash
   git clone https://github.com/PritamPalit-official/BUSINESS_CASE_STUDY_AEROFIT.git
   cd BUSINESS_CASE_STUDY_AEROFIT
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

This will automatically launch the dashboard in your default browser at `http://localhost:8501`.

### 📓 Running the Notebook Analysis

If you'd like to explore the step-by-step Jupyter Notebook analysis:

1. **Install Jupyter and base requirements**:
   ```bash
   pip install pandas numpy matplotlib seaborn jupyter
   ```

2. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```
   Open `Aerofit_Descriptive_Stats.ipynb` to see the complete exploration workflow.

---

## 🛠️ Development & Testing

To maintain production-ready code quality, this repository includes dev dependencies, unit testing configurations, and automated CI pipelines:

### 📦 Setup Developer Dependencies
Install the required development and testing packages:
```bash
pip install -r requirements-dev.txt
```

### 🧪 Run Unit Tests Locally
Run the test suite using Python's built-in `unittest` runner:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### ⚙️ Continuous Integration (CI)
A GitHub Actions workflow is configured in `.github/workflows/ci.yml`. On every `push` and `pull_request` to the repository, it automatically:
1. Provisions an Ubuntu runner with Python 3.10.
2. Installs dependencies from both `requirements.txt` and `requirements-dev.txt`.
3. Runs the test suite to verify code integrity and prevent regressions.

---

## 👤 Author

<div align="center">

**Pritam Palit**

Electronics & Communication Engineering Graduate

*Data Analytics • Statistics • Business Intelligence*

[![GitHub](https://img.shields.io/badge/GitHub-PritamPalit--official-181717?style=for-the-badge&logo=github)](https://github.com/PritamPalit-official)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Pritam%20Palit-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/pritam-palit-77b2071b4/)

</div>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

⭐ *If you found this project useful, consider giving it a star!* ⭐

</div>
