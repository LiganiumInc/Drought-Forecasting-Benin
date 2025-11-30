# Prediction and Uncertainty Quantification of Drought in North Benin

## 📌 Objective

This work aims to develop an **uncertainty-aware drought forecasting framework** for six key localities in the Alibori department of North Benin — **Banikoara, Gogounou, Kandi, Karimama, Malanville, and Segbana**.
Our goal is twofold:

1. **Predict drought conditions** using state-of-the-art machine learning and deep learning models based on the Standardized Precipitation Index at a 6-month scale (SPI-6).
2. **Quantify prediction uncertainty** using the Ensemble Batch Prediction Interval (EnbPI) method, enabling more informed and trustworthy decision-making.

---

## 📥 Installation & Setup

Follow these steps to set up the environment and run the experiments.

### 1️⃣ Create and activate a virtual environment

Using **Python 3.10+**:

```bash
# Create a virtual environment
python -m venv env

# Activate it
# On Linux/MacOS:
source env/bin/activate
# On Windows (PowerShell):
env\Scripts\activate
```

---

### 2️⃣ Clone the repository

```bash
git clone https://github.com/LiganiumInc/Drought-Forecasting-Benin.git
cd Drought-Forecasting-Benin/models
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running Experiments

### ▶ Run for a specific city (example: Banikoara)

```bash
python banikoara_generic_model_building.py
```

---

### ▶ Run for all six cities at once

```bash
chmod +x run_all.sh
./run_all.sh
```

---

## 📊 Results Overview

Our comparative study involved:

* **6 Machine Learning models**: Linear Regression, Ridge Regression, Random Forest, XGBoost, LightGBM, SVR
* **4 Deep Learning models**: Conv1D, LSTM, GRU, Conv1D-LSTM
* **Evaluation metrics**: R², RMSE, MAE, Carbon Footprint
* **Uncertainty metrics**: Empirical Coverage, Prediction Interval Width

The **Conv1D-LSTM** model emerged as the top performer, offering an optimal balance between predictive accuracy and uncertainty coverage.

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{Ligan_Tchokponhoue_Iliho-Manzi_Simperegui_Vodounon_2025,
  title={Prediction and uncertainty quantification of drought in North Benin},
  author={Ligan, Bernardin Marie Augustin Sèdjro and Tchokponhoue, Gbègninougbo Aurel Davy and Iliho Manzi, Alain Thierry and Simperegui, Kora B. Désiré and Vodounon, Rodrigue Bignon Wilfried},
  journal={Environmental Data Science},
  volume={4},
  pages={e50},
  year={2025},
  publisher={Cambridge University Press},
  doi={10.1017/eds.2025.10025}
}
```

---

## 📄 License

This project is released under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

