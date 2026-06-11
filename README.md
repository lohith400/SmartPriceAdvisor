# 🏷️ SmartPriceAdvisor

> **AI-powered dynamic pricing engine for Indian e-commerce** — uses causal inference (EconML LinearDML) to estimate heterogeneous price elasticity across regions and recommend optimal discount strategies.

---

## 📌 What This Project Does

SmartPriceAdvisor answers the question:

> *"If I give a 30% discount on this product — how much will demand increase, and does that effect differ between a West Urban shopper vs an East Rural household?"*

It uses **Double Machine Learning (EconML LinearDML)** — a causal inference model — to estimate **Conditional Average Treatment Effects (CATE)** of discounts on demand, segmented by regional economic conditions from India's HCES survey.

---

## 🗂️ Project Structure

```
SmartPriceAdvisor/
│
├── data/
│   ├── raw/                          # Source datasets (excluded from git — see .gitignore)
│   │   ├── flipkart_fashion_products_dataset.json   # Flipkart fashion product catalog
│   │   ├── amazon_sale_report.csv                   # Amazon India transaction records
│   │   └── hces_level01.csv                         # HCES 2022-23 regional spending survey
│   │
│   └── final/
│       └── model_ready.csv           # Processed dataset ready for EconML (100,000 rows)
│
├── build_pipeline.py                 # Full ETL pipeline: raw → model_ready.csv
├── requirements.txt                  # Python dependencies
├── .gitignore
└── README.md
```

---

## 📦 Datasets Used

### 1. Flipkart Fashion Products (`flipkart_fashion_products_dataset.json`)
- **What:** Product catalog with MRP (`actual_price`) and sale price (`selling_price`)
- **Why:** Provides realistic Indian fashion pricing with large discount ranges (30–80%)
- **Contributes:** `original_price`, `selling_price`, `discount_pct` → the **Treatment (T)**

### 2. Amazon India Sale Report (`amazon_sale_report.csv`)
- **What:** Real Amazon India transaction records with quantity sold
- **Why:** Provides actual `Qty` sold — the only real demand signal in the pipeline
- **Contributes:** `selling_price`, `demand_proxy` (units sold) → the **Outcome (Y)**

### 3. HCES 2022-23 — Household Consumer Expenditure Survey (`hces_level01.csv`)
- **What:** Government survey of monthly household spending by zone + rural/urban
- **Why:** Provides regional economic context — how much households in each zone spend on clothing/electronics relative to total budget
- **Contributes:** `avg_mpce_clothing`, `avg_mpce_electronics`, `avg_mpce_total`, `elasticity_weight` → the **Effect Modifiers (X)**

---

## 🧮 Model Architecture

```
EconML LinearDML
─────────────────────────────────────────────
T  (Treatment)      = discount_pct
Y  (Outcome)        = log_demand
X  (Effect modifers)= zone, sector, elasticity_weight
W  (Controls)       = brand, seller_rating, category_std
─────────────────────────────────────────────
Output: τ(x) — price elasticity per region/segment
```

The model learns that the **same discount has a different impact** in different regions:
- 🏙️ **West Urban** (higher budget share on clothing) → less price-sensitive
- 🌾 **East Rural** (tighter household budget) → more discount-responsive

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/lohith400/SmartPriceAdvisor.git
cd SmartPriceAdvisor
```

### 2. Set up virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add raw datasets
Place the following files inside `data/raw/`:
- `flipkart_fashion_products_dataset.json`
- `amazon_sale_report.csv`
- `hces_level01.csv`

> ⚠️ Raw data files are excluded from git due to size. Download them from [Kaggle](https://www.kaggle.com) and the [MoSPI HCES portal](https://mospi.gov.in).

### 5. Run the pipeline
```bash
python build_pipeline.py
```

This generates `data/final/model_ready.csv` — 100,000 rows ready for EconML training.

---

## 📋 Output Schema (`model_ready.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `original_price` | float | MRP / listed price |
| `selling_price` | float | Actual sale price |
| `discount_pct` | float | Discount percentage (0–95) |
| `demand_proxy` | float | Units sold (or 1.0 if unavailable) |
| `log_demand` | float | log(demand_proxy + 1) |
| `brand` | str | Brand / seller identifier |
| `seller_rating` | float | Product/seller rating (1–5) |
| `category_std` | str | `clothing` or `other` |
| `zone` | str | North / South / East / West / Central |
| `sector` | str | Rural or Urban |
| `avg_mpce_clothing` | int | Avg monthly clothing spend by zone+sector (HCES) |
| `avg_mpce_electronics` | int | Avg monthly electronics spend by zone+sector (HCES) |
| `avg_mpce_total` | int | Total avg monthly household spend by zone+sector (HCES) |
| `elasticity_weight` | float | `avg_mpce_clothing / avg_mpce_total` — price sensitivity proxy |
| `source` | str | `flipkart` or `amazon` |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| Data Processing | pandas, numpy |
| Causal Inference | EconML (LinearDML) |
| ML Backend | scikit-learn |
| Data Source | Kaggle, MoSPI HCES |

---

## 📄 License

This project is for academic and research purposes.
