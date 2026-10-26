# Customer Activity Prediction and Segmentation for an Online Store

A Yandex Practicum project on classification, model interpretation and customer segmentation. The notebook is written in Russian; this README summarizes it in English.

## Business task

The online store «В один клик» sells children's goods, home products, small appliances, cosmetics and groceries. Customer activity has started to fall, and attracting new customers is no longer efficient.
Goal: predict which regular customers will reduce their purchasing activity in the next three months, and use this with profitability data to design personalized offers.

## Data

Four tables: customer behavior and communications, revenue by period, time on site by period, and profit per customer.

## What I did

1. **Data loading and preprocessing**, including different separators and decimal marks across files.
2. **EDA:** kept customers who bought something in each of the last three months.
3. **Table merging** with separate columns per period, and **correlation analysis** with φK.
4. **Pipelines** with `ColumnTransformer` (several encoders and scalers) and hyperparameter search over KNN, Decision Tree, Logistic Regression and SVC. Main metric: ROC-AUC, because classes are imbalanced.
5. **Feature importance** with SHAP.
6. **Segmentation** by predicted risk and profitability, with a deep dive into one segment.

## Results

| Metric | Value |
|---|---|
| Best model | SVC |
| ROC-AUC (cross-validation) | 0.922 |
| **ROC-AUC (test)** | **0.894** |
| **F1 (test)** | **0.818** |

- Activity drops for customers with low marketing engagement, few categories and pages viewed per visit, many unpaid items in the cart and more service errors.
- **Recommendations:** expand the assortment, reduce service errors and build personalized offers for customers who buy children's goods and small appliances, the categories with the most loyal and profitable customers.

## Tech stack

Python · pandas · NumPy · Matplotlib · Seaborn · SciPy · phik · scikit-learn · SHAP

## How to run

```bash
pip install -r requirements.txt
jupyter notebook supervised_learning_quality.ipynb
```

---

**Кратко по-русски.** Прогноз снижения покупательской активности и сегментация клиентов интернет-магазина «В один клик» (учебный проект Яндекс Практикума). Лучшая модель SVC: ROC-AUC 0,894 и F1 0,818 на тесте. Важность признаков объяснена через SHAP, предложены меры для сегмента покупателей товаров для детей и мелкой бытовой техники.
