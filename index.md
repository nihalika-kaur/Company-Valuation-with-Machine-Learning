---
layout: default
title: Company Valuation with Machine Learning
subtitle: A Machine Learning Approach to Financial Valuation
authors: Vaishnavi, Nihalika, Joey, Ethan, Kevin
affiliation: Georgia Institute of Technology
conference: ML Project Proposal 2025
---

<div class="hero-section">
  <h1>{{ page.title }}</h1>
  <div class="hero-subtitle">{{ page.subtitle }}</div>
  <div class="hero-authors">{{ page.authors }}</div>
  <div class="hero-affiliation">{{ page.affiliation }}</div>
  <div class="hero-conference">{{ page.conference }}</div>
  
  <div class="hero-links">
      <a href="https://docs.google.com/spreadsheets/d/1Ujni_pAhbjUFA962Q-7-PO1h_qXdSkk5PwlB_SbiMZw/edit?usp=drive_link" target="_blank" class="gantt-chart-link">
      <i data-lucide="bar-chart-3"></i>
      Gantt Chart: Click to access chart
    </a>
  </div>
    <div class="hero-links">
    <a href="https://docs.google.com/spreadsheets/d/1LAgX2PcVrMXNXcEczlVuXmvG5UGg3BKb_QjkRI2ZjAs/edit?usp=drive_link" target="_blank" class="gantt-chart-link">
      <i data-lucide="bar-chart-3"></i>
      Contribution Table: Click to access table
    </a>
  </div>
  
  <div class="table-of-contents">
    <h3>Table of Contents</h3>
    <ol>
      <li><a href="#introduction">Introduction</a></li>
      <li><a href="#literature-review">Literature Review</a></li>
      <li><a href="#dataset-description">Dataset Description</a></li>
      <li><a href="#problem-definition">Problem Definition and Motivation</a></li>
      <li><a href="#methodology">Methodology</a></li>
      <li><a href="#models">Models</a></li>
      <li><a href="#results">Results and Discussion</a></li>
      <li><a href="#references">References</a></li>
    </ol>
  </div>
</div>

## Introduction {#introduction}

Company valuation is one of the most important problems in finance. Investors, analysts, and policymakers all need to know what a company is truly worth in order to make decisions about investing, mergers and acquisitions, or even risk management. Traditional valuation methods such as Discounted Cash Flow (DCF), Comparable Company Analysis, and P/E multiples, rely on economic assumptions and linear relationships. While predictable, "it remains hard to forecast the stock price movement mainly because the financial market is a complex, evolutionary, and non-linear dynamical system [2]."
We want to explore how supervised and unsupervised machine learning methods can be applied to company valuation. Specifically:

- Use supervised learning to predict valuation metrics such as P/E or analyst target prices from financial fundamentals.

- Try to use unsupervised learning to cluster firms into valuation-based groups (for example, value, growth, speculative).

- Compare ML-based valuation results against traditional finance benchmarks.


## Literature Review {#literature-review}

We have included some studies we came across during our research that show that ML can capture complex financial relationships better than traditional methods and is becoming a critical tool in modern valuation:

- [Empirical Asset Pricing via Machine Learning](https://academic.oup.com/rfs/article/33/5/2223/5758276?login=false):This paper talks about several classifiers like random forests and SVM to predict company valuation.

- [Evaluating multiple classifiers for stock price direction prediction](https://www.sciencedirect.com/science/article/abs/pii/S0957417415003334): This paper talks about several classifiers like random forests and SVM to predict company valuation and stock price direction.


- [Creating Value from Big Data in the Investment Management Process](https://rpc.cfainstitute.org/research/reports/2025/creating-value-from-big-data-in-the-investment-management-process) : This CFA Institute report shows that industry research also supports the trend that AI and big data are becoming integral to investment management.




## Dataset Description {#dataset-description}

We use publicly available datasets containing financial fundamentals and market valuations for U.S. companies. The datasets include:

- **Fundamentals**: revenue, earnings, assets, liabilities, free cash flow.

- **Valuation ratios**: P/E, P/B, Price-to-Sales, Dividend Yield.

- **Market data**: historical stock prices, volatility measures.


This structure will allow us to train supervised models (for example predicting valuation ratios from fundamentals) and apply clustering methods (for example grouping firms by financial similarity).

<span style="color:#A95FD0; font-weight:bold; font-size:25px;">Midterm- Dataset Report</span>

Midterm Check- Datasets

The dataset used in this project was obtained from Compustat Global - WRDS Basics, accessible through the Wharton Research Data Services (WRDS). Compustat Global is a database containing standardized financial statement data for publicly traded companies worldwide, maintained by S&P Global Market Intelligence.

Why we chose Compustat Global?

Academic Standard: Compustat is the standard database in finance research and is widely used in both academic studies and industry applications. This makes sure our findings are comparable to existing literature and professional practices
Data Quality and Reliability: Maintained by S&P Global Market Intelligence, Compustat provides professionally validated financial data with standardized accounting metrics across different countries and reporting standards. This eliminates the need for extensive manual data cleaning and normalization
Comprehensive Coverage: The dataset includes both active and delisted companies, which reduces survivorship bias which is a critical concern in financial modeling where failed companies are often excluded. This leads to overly optimistic results
Temporal Depth: With data spanning from 2010 to 2025, we have access to multiple economic cycles allowing our model to learn patterns across diverse market conditions.
The raw dataset contains financial metrics and company identifiers from multiple years of corporate financial data. We carefully selected four fundamental features (ROE, ROA, debt-to-equity, sales growth) to predict the E/P ratio based on established financial theory and practical considerations:

gvkey: Global Company Key
tic: Ticker symbol - stock exchange trading symbol
datadate: Fiscal year-end date - temporal identifier used with gvkey to uniquely identify each observation (company-date pair)
roe: ROE measures how efficiently a company generates profits from shareholder investments and directly relates to valuation as investors pay premium prices for companies with high returns. It is widely used by investors and analysts in fundamental analysis, and we expected it to have a positive relationship with the E/P ratio.
roa: ROA indicates how efficiently a company uses its total asset base to generate earnings and captures operational efficiency independent of capital structure. It is less influenced by financial leverage compared to ROE, providing complementary information, and we expected it to have a positive relationship with the E/P ratio.
debt_to_eq: This ratio reflects financial risk and capital structure decisions, where high leverage can amplify returns but also increases bankruptcy risk. It affects the cost of capital and investor required returns, and its relationship with E/P may be non-linear since moderate debt can be beneficial while excessive debt is harmful.
sales_growth: Sales growth is a forward-looking indicator of company momentum and market position, as growth companies often command valuation premiums. It captures revenue trajectory independent of current profitability, though we expected its relationship with E/P to be weaker for mature companies.
eq- Earnings to Price ratio- a valuation matrix
Note: The combination of gvkey and datadate serves as the composite primary key for our dataset, uniquely identifying each company's financial snapshot at a specific point in time. This is essential for preventing duplicate records and maintaining data integrity throughout our analysis.

Why E/P as the Target Variable?

We chose E/P ratio (inverse of the popular P/E ratio) as our target for several reasons. From a mathematical perspective, E/P avoids division-by-zero issues that occur when earnings are negative, making it more strong for modeling purposes. Additionally, E/P tends to have better statistical properties for regression modeling compared to P/E, as it follows a more normal distribution. From a financial interpretation standpoint, higher E/P indicates higher earnings yield, directly measuring the return on investment that shareholders receive. Finally, E/P has practical relevance as it is commonly used by value investors to identify undervalued stocks, making our model's predictions directly applicable to real-world investment decisions.



<span style="color:#A0AF85; font-weight:bold; font-size:25px;">Final Dataset Report</span>


The dataset used in this project was obtained from Compustat Global WRDS Basics, accessible through the Wharton Research Data Services (WRDS). Compustat Global is a database containing standardized financial statement data for publicly traded companies worldwide, maintained by S&P Global Market Intelligence.

Why we chose Compustat Global?

- Academic Standard: Compustat is the standard database in finance research and is widely used in both academic studies and industry applications. This makes sure our findings are comparable to existing literature and professional practices
- Data Quality and Reliability: Maintained by S&P Global Market Intelligence, Compustat provides professionally validated financial data with standardized accounting metrics across different countries and reporting standards. This eliminates the need for extensive manual data cleaning and normalization
- Comprehensive Coverage: The dataset includes both active and delisted companies, which reduces survivorship bias which is a critical concern in financial modeling where failed companies are often excluded. This leads to overly optimistic results
- Temporal Depth: With data spanning from 2009 to 2021, we have access to multiple economic cycles allowing our model to learn patterns across diverse market conditions.

### Dataset Construction

Our final dataset implements rigorous temporal splitting to prevent data leakage, a critical concern in time-series financial prediction. The dataset comprises **61,61,597 observations** over 13 years (2009-2021):

**Training Set (82.4%)**: 50,731 observations spanning 2009-2019
- Covers 11 years of financial data
- Includes 8,506 unique companies
- Year distribution ranges from 538 observations (2009, partial year) to 5,190 observations (2014)

**Test Set (17.6%)**: 10,866 observations spanning 2020-2021
- Covers 2 years of out-of-sample data (2020: 5,204 obs; 2021: 5,662 obs)
- Includes 6,136 unique companies
- 77.8% of test companies also appear in training data, allowing for genuine temporal prediction while maintaining company continuity

**Temporal Split Rationale**: The 2009-2019 (train) vs 2020-2021 (test) split ensures zero temporal overlap, preventing future information from leaking into model training. This mimics real-world deployment where models trained on historical data must predict future outcomes. The test period includes the COVID-19 pandemic year (2020) and recovery year (2021), providing a robust evaluation of model performance under extreme market conditions.

### Feature Engineering and Data Quality

The dataset contains **14 engineered features** across two categories plus one target variable:

**Financial Fundamentals (9 features)**:
- **Profitability Metrics**: Return on Assets (ROA), Return on Equity (ROE), Gross Profitability, Operating Margin
- **Growth Metrics**: Sales Growth, Asset Growth
- **Leverage & Efficiency**: Debt-to-Assets Ratio, Asset Turnover
- **Size**: Log Assets (natural log transformation to handle scale)

**Corporate Culture (Sentient) Metrics (5 features)**:
- **Innovation**: Measures company's investment in R&D and innovation initiatives
- **Integrity**: Captures ethical practices and governance quality
- **Quality**: Reflects product/service quality standards
- **Respect**: Gauges employee treatment and workplace culture
- **Teamwork**: Assesses collaborative organizational structure

**Target Variable**:
- **EP (Earnings-to-Price Ratio)**: Our target variable representing company valuation efficiency. Train mean: 0.104 (σ=0.240); Test mean: 0.058 (σ=0.247). The lower test mean reflects the 2020-2021 market conditions where valuations increased relative to earnings.

**Data Quality Verification**:
- **Zero missing values** across all 61,597 observations
- **Zero duplicate rows** ensuring data integrity
- **Standardized features**: All financial and culture metrics are z-score normalized (mean≈0, std≈1 in training set)
- **No data leakage**: Confirmed through temporal non-overlap and proper preprocessing sequence

### Corporate Culture (Sentient) Data Integration

A unique aspect of our dataset is the integration of **corporate culture metrics**, often referred to as "sentient data" in organizational research. These metrics capture qualitative organizational characteristics that traditional financial models ignore:

**Rationale for Culture Metrics**: Research in organizational psychology and management science demonstrates that corporate culture significantly impacts long-term firm performance. Companies with strong innovation cultures command valuation premiums, while those with integrity issues face discounts. However, culture metrics are notoriously difficult to quantify.

**Data Source & Methodology**: Culture metrics were derived from:
1. **Textual Analysis**: Natural language processing of annual reports, earnings calls, and corporate communications
2. **Employee Reviews**: Sentiment analysis of employee feedback on platforms like Glassdoor
3. **ESG Ratings**: Environmental, Social, and Governance scores from institutional rating agencies
4. **Patent Filings**: Innovation metrics derived from R&D expenditures and patent activity

**Standardization**: All culture metrics are z-score normalized relative to the training period (2009-2019), with train mean=0.00 and std=1.00. Test period metrics show slight positive drift (innovation: +0.26σ, respect: +0.22σ), reflecting post-pandemic shifts toward remote work, employee well-being, and digital transformation.

**Empirical Relevance**: While feature importance analysis reveals that culture metrics contribute modestly to predictive power (innovation: 2.32%, others <1% in Gradient Boosting models), their inclusion provides a more holistic view of organizational health. Future research may explore interaction effects between culture and financial metrics, particularly during periods of market stress.

### Dataset Limitations and Considerations

**Temporal Bias**: The test period (2020-2021) includes extraordinary market conditions due to COVID-19. Models trained on pre-pandemic data may underperform during unprecedented events, highlighting the importance of periodic retraining.

**Survivorship Bias**: While Compustat includes delisted companies, our filtering criteria (requiring complete financial data) may inadvertently exclude the weakest firms that fail to report consistently, potentially inflating performance estimates.

**Company Overlap**: The 77.8% overlap between train and test companies means our models primarily predict temporal changes for known firms rather than generalizing to entirely new companies. This is intentional and mirrors real-world portfolio management where analysts track existing positions over time.

**Culture Metric Reliability**: Corporate culture metrics are derived from secondary sources and proxies. While validated against academic research, they inherently contain measurement error and may not perfectly capture organizational dynamics.

**Data Freshness**: As of this analysis, our dataset extends through 2021. Real-world deployment would require continuous updates from WRDS to maintain predictive accuracy as new financial statements are released.

The dataset contains financial metrics and company identifiers from multiple years of corporate financial data. We selected eleven fundamental features to predict the E/P ratio based on established financial theory and practical considerations. This selection balances model complexity with interpretability while capturing the key dimensions of corporate financial performance: profitability, growth, financial structure, and operational efficiency.

Feature Descriptions:

  - **gvkey**: Global Company Key - unique identifier for each company in the Compustat database
  - **conm**: Company Name - full legal name of the company as registered. Provides human readable identification and can be used for industry classification or text analysis.
  - **tic**: Ticker symbol - stock exchange trading symbol
  - **fyear**: Fiscal Year - the year-end date for the financial reporting period. Combined with gvkey, this forms the composite primary key ensuring each observation represents a unique company-year combination.

Profitability Metrics:

  - **gross_profitability**:Measures the efficiency of production and pricing strategies by capturing the proportion of revenue retained after direct costs of goods sold. Higher values indicate stronger pricing power or more efficient production processes. This metric is particularly important for comparing companies within capital-intensive industries.
  - **operating_margin**: Indicates operational efficiency by showing what percentage of revenue remains after operating expenses, revealing management effectiveness in controlling costs while maintaining revenue. Unlike gross profitability, this captures the full operational picture including R&D, marketing, and administrative expenses.
  - **roe**: ROE measures how efficiently a company generates profits from shareholder investments and directly relates to valuation as investors pay premium prices for companies with high returns. It is widely used by investors and analysts in fundamental analysis, and we expected it to have a positive relationship with the E/P ratio.
  - **roa**: ROA indicates how efficiently a company uses its total asset base to generate earnings and captures operational efficiency independent of capital structure. It is less influenced by financial leverage compared to ROE, providing complementary information, and we expected it to have a positive relationship with the E/P ratio.

Growth Indicators:
  - **sales_growth**: Sales growth is a forward-looking indicator of company momentum and market position, as growth companies often command valuation premiums. It captures revenue trajectory independent of current profitability, though we expected its relationship with E/P to be weaker for mature companies.
  - **asset_growth**:  Reflects the rate of expansion in a company's asset base, indicating investment in future capacity and potential for scaling operations. High asset growth can signal expansion but may also indicate inefficient capital deployment if not accompanied by corresponding revenue and earnings growth.

Financial Structure and Efficiency: 
  - **debt_to_assets**: This ratio reflects financial risk and capital structure decisions. It affects the cost of capital and investor required returns. We expected its relationship with E/P to be non-linear: moderate debt can be beneficial by providing tax shields and disciplining management, while excessive debt increases financial risk and may depress valuations.
  - **log_assets**: Natural logarithm of total assets, used to normalize asset size and reduce the impact of extreme values while capturing company scale. This transformation is standard in financial econometrics as it addresses the right-skewed distribution of firm sizes and allows for better model performance. Larger companies may benefit from economies of scale but can also face diseconomies of scale and bureaucratic inefficiencies.
  - **asset_turnover**:  Measures how efficiently a company uses its assets to generate sales, with higher values indicating better asset utilization. This ratio varies significantly across industries: retail and service companies typically have high asset turnover, while capital-intensive industries like utilities and manufacturing have lower ratios. It provides insight into management's efficiency in deploying capital.

Target Variable
  - **eq**- Earnings to Price ratio- a valuation matrix

Note: The combination of gvkey and datadate serves as the composite primary key for our dataset, uniquely identifying each company's financial snapshot at a specific point in time. This is essential for preventing duplicate records and maintaining data integrity throughout our analysis.

Why E/P as the Target Variable?

We chose E/P ratio (inverse of the popular P/E ratio) as our target for several reasons. From a mathematical perspective, E/P avoids division-by-zero issues that occur when earnings are negative, making it more strong for modeling purposes. Additionally, E/P tends to have better statistical properties for regression modeling compared to P/E, as it follows a more normal distribution. From a financial interpretation standpoint, higher E/P indicates higher earnings yield, directly measuring the return on investment that shareholders receive. Finally, E/P has practical relevance as it is commonly used by value investors to identify undervalued stocks, making our model's predictions directly applicable to real-world investment decisions.

**Total Observations: 72,180**

| Feature | Count | Mean | Std | Min | 25% | Median | 75% | Max |
|:--------|------:|-----:|----:|----:|----:|-------:|----:|----:|
| **fyear** | 72,180 | 2015.39 | 3.51 | 2009 | 2012 | 2015 | 2018 | 2021 |
| **gross_profitability** | 72,180 | 0.00 | 1.00 | -7.84 | 0.16 | 0.20 | 0.24 | 0.38 |
| **operating_margin** | 72,180 | 0.00 | 1.00 | -7.84 | 0.16 | 0.20 | 0.24 | 0.38 |
| **sales_growth** | 72,180 | 0.00 | 1.00 | -7.84 | 0.16 | 0.20 | 0.24 | 0.38 |
| **asset_growth** | 72,180 | 0.00 | 1.00 | -7.84 | 0.16 | 0.20 | 0.24 | 0.38 |
| **debt_to_assets** | 72,180 | 0.00 | 1.00 | -7.84 | 0.16 | 0.20 | 0.24 | 0.38 |
| **log_assets** | 72,180 | 0.00 | 1.00 | -7.84 | 0.16 | 0.20 | 0.24 | 0.38 |
| **asset_turnover** | 72,180 | 0.00 | 1.00 | -7.84 | 0.16 | 0.20 | 0.24 | 0.38 |
| **roe** | 72,180 | 0.00 | 1.00 | -7.84 | 0.16 | 0.20 | 0.24 | 0.38 |
| **roa** | 72,180 | 0.00 | 1.00 | -7.84 | 0.16 | 0.20 | 0.24 | 0.38 |
| **ep** | 72,180 | 6.76 | 6.45e3 | -7.92e5 | -0.028 | 0.085 | 0.166 | 1.51e6 |


**Key Observations**:
- **Standardization**: Most features show mean ≈ 0 and standard deviation ≈ 1, indicating z-score normalization has been applied. This ensures all features contribute equally to distance-based algorithms and gradient descent optimization.
- **Temporal Balance**: The median fiscal year of 2015 indicates balanced coverage across the 2009-2021 training period.
- **Target Variable Distribution**: The E/P ratio shows significant skewness with extreme outliers (min: -791,667, max: 1,513,351), though the median of 0.085 suggests most companies have reasonable valuations. The negative minimum indicates companies with negative earnings.


The table below presents summary statistics for the testing dataset (11,955 observations):

| Feature | Count | Mean | Std | Min | 25% | 50% (Median) | 75% | Max |
|---------|-------|------|-----|-----|-----|--------------|-----|-----|
| **fyear** | 11,955 | 2022.48 | 0.50 | 2022 | 2022 | 2022 | 2023 | 2023 |
| **gross_profitability** | 11,955 | 0.01 | 0.82 | -7.84 | 0.09 | 0.19 | 0.23 | 0.38 |
| **operating_margin** | 11,955 | -0.01 | 0.90 | -7.84 | 0.06 | 0.17 | 0.23 | 0.38 |
| **sales_growth** | 11,955 | 0.05 | 1.00 | -7.84 | 0.08 | 0.18 | 0.23 | 0.38 |
| **asset_growth** | 11,955 | 0.04 | 0.95 | -7.84 | 0.11 | 0.20 | 0.24 | 0.38 |
| **debt_to_assets** | 11,955 | -0.00 | 1.00 | -7.84 | 0.13 | 0.21 | 0.24 | 0.38 |
| **log_assets** | 11,955 | 0.02 | 0.98 | -7.84 | 0.14 | 0.21 | 0.24 | 0.38 |
| **asset_turnover** | 11,955 | 0.02 | 0.96 | -7.84 | 0.13 | 0.21 | 0.24 | 0.38 |
| **roe** | 11,955 | 0.06 | 0.76 | -7.84 | 0.09 | 0.19 | 0.23 | 0.38 |
| **roa** | 11,955 | 0.03 | 0.77 | -7.84 | 0.09 | 0.19 | 0.23 | 0.38 |
| **ep** | 11,955 | -39.31 | 4,280.56 | -467,961.20 | -0.138 | 0.059 | 0.177 | 6,227.87 |

**Key Observations**:
- **Recent Time Period**: The test set focuses exclusively on fiscal years 2022-2023, representing the most recent market conditions and providing a true out-of-sample evaluation.
- **Distribution Shift**: Some features show slightly different distributions compared to the training set, which is expected given the different economic environment (post-COVID recovery, inflation, rising interest rates).
- **Target Variable Characteristics**: The test set E/P shows a lower median (0.059 vs 0.085 in training), suggesting potentially higher market valuations in recent years. The mean is notably negative (-39.31) due to extreme outliers.


**Data Quality Consideration**: 

The Compustat Global dataset is well recognized for its reliability and standardization, making it suitable for financial machine learning applications. It provides standardized metrics where financial ratios are calculated consistently across companies and countries, ensuring comparability. Also, it offers high coverage by including thousands of companies across multiple industries and geographies, giving our model diverse training examples. The dataset benefits from professional validation through S&P Global's rigorous data validation processes that ensure accuracy and reliability. Additionally, its temporal depth with historical data spanning multiple years allows for trend analysis and assessment of different economic cycles. The dataset mitigates survivorship bias by including delisted companies, which provides a more realistic view of historical performance rather than only showcasing successful firms that remain active today.

**Future Data Considerations**

**Higher Frequency Data**: 
- Incorporate quarterly reports (10-Q filings) instead of annual data
- Benefits: 4× more training samples, captures seasonal patterns, faster-moving market trends
- Challenges: More missing data, restatements, increased noise

**Expanded Feature Set**:
- **Cash Flow Metrics**: Operating cash flow, free cash flow, cash conversion rates
- **Profit Margins**: Gross, operating, and net margins for deeper profitability analysis
- **Industry Classification**: SIC/NAICS codes for sector-specific modeling and industry adjustments
- **Market Metrics**: Market capitalization, trading volume, volatility measures, liquidity ratios
- **Additional Multiples**: P/B, EV/EBITDA, EV/Sales, PEG ratios for comprehensive valuation
- **Quality Factors**: Accruals, earnings quality scores, Piotroski F-Score



### Potential Dataset Links we Looked at for The Project

- **Kaggle datasets**:
  - [S&P](https://www.kaggle.com/datasets/camnugent/sandp500)
  - [STOCK MARKET DATASET](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)

- [Yahoo Finance API](https://pypi.org/project/yfinance/)

- [WRDS](https://wrds-www.wharton.upenn.edu/): Great resources for academic finance research, GT students can request institutional access

- **Datasets focused on sentiment analysis:**:
  - [Daily News for Stock Market Prediction](https://www.kaggle.com/datasets/aaron7sun/stocknews)
  - [Financial News Headlines Data](https://www.kaggle.com/datasets/notlucasp/financial-news-headlines)
  - [Sentiment Analysis for Financial News](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)



## Problem Definition and Motivation {#problem-definition} 

Traditional company valuation methods such as DCF, P/E ratios etc. rely on human assumptions and linear relationships. While they provide a baseline estimate of a company's value, they fail to capture nonlinear interactions in financial data. Accurate valuation is crucial for investors, analysts, and policymakers, as it guides decisions regarding investment, risk management, and capital allocation. 

Hence, the problem we aim to address is: how can we predict a company's valuation more accurately by leveraging machine learning to combine both financial fundamentals and external signals?

## Methodology {#methodology}

### Data Preprocessing Methods

We will begin by addressing missing data through imputation, filling in gaps with representative statistics such as the median. Next, we will manage outliers in variables like P/E ratios or leverage, which can distort results if left uncorrected. Techniques such as winsorizing (limiting extreme values to a certain percentile) or clipping the top and bottom 1% will help control for these issues. Finally, we will apply scaling and normalization so that variables measured in very different units are placed on a comparable scale.

<span style="color:#A95FD0; font-weight:bold; font-size:25px;">Midterm- Preprocessing Method</span>

The preprocessing stage, implemented in preprocessing.py, transforms the raw Compustat export from WRDS into a clean, structured dataset suitable for machine learning. The goal of this pipeline is to ensure data consistency across companies and time, engineer meaningful financial ratios, and prepare features on comparable scales for model training.
  - **Input**: 
      - data/raw/compustat_data.csv (raw Compustat export from WRDS)
  - **Output**:
      - data/processed/train.parquet (49,960 samples, 2010-2022)
      - data/processed/test.parquet (17,113 samples, 2023-2025)

This pipeline follows a chronological structure that mimics how models are used in real-world finance:  trained on past data and tested on future firms

1.	Data Loading and Date Standardization
   - a.	The script begins by loading the Compustat CSV and converting the datadate field from string to pandas datetime format. The dataset is then sorted by gvkey (the firm identifier) and datadate to maintain a correct temporal sequence.
   - b. The goal of this was to order the Financial data into proper time-series ordering to prevent data leakage

2. Feature Engineering
  - a. We derived key financial ratios used for valuation analysis: 
      - ROE = Net Income / Equity 
      - ROA = Net Income / Assets 
      - Debt-to-Equity = Liabilities / Equity Sales 
      - Growth = Year-over-year % change in revenue 
      - E/P Ratio = Earnings per share / Price per share (target variable)
  - b. By establishing these key ratios, we are able to standardize across firm sizes and capture profitability, leverage, and growth which are key valuation indicators and drivers

3. Handling Outliers and Missing Values
  - a. Applied winsorization (1st–99th percentile) to reduce the influence of extreme financial values. 
  - b. Filled missing values using median imputation based on the training set.
  - c. Outliers would skew our results; thus handling outliers and missing values improves model stability while preserving the distribution of real-world financial data.
  - 
4. Scaling and Standardization
  - a. We ensure that all features were standardized using z-score scaling based on the training set mean and standard deviation.
  - b. This ensures that all variables are on comparable scales, preventing large-magnitude features from dominating the model.

5. Train/Test Split and Storage
  - a. Data was split chronologically: 
      - Train: ≤ Dec 2022 
      - Test: > Dec 2022
        
Our preprocessing pipeline cleans and standardizes raw Compustat data, engineers interpretable financial ratios, handles outliers and missing values, and ensures proper temporal separation—producing a consistent dataset ready for ML modeling.


<span style="color:#A0AF85; font-weight:bold; font-size:25px;">Final- Preprocessing Method</span>

Our preprocessing pipeline implemented a multi-stage approach encompassing data cleaning, feature engineering, integration of heterogeneous data sources, and standardization procedures designed to ensure model stability and prevent information leakage.

**Data Cleaning and Outlier Management**

We applied rigorous filtering criteria to eliminate noise and handle extreme values that could distort model learning. First, we implemented a penny stock filter, removing all companies with share prices below $1.00, as these securities typically exhibit high volatility and questionable data quality that introduce substantial noise into valuation models. Second, for the target variable (E/P ratio), we addressed missing Net Income values through median imputation within industry groups to preserve sector-specific financial characteristics. Subsequently, we applied Winsorization to the E/P ratio, clipping extreme outliers at the 1st and 99th percentiles to mitigate the influence of anomalous values while retaining meaningful variation in the distribution. This approach proved superior to complete outlier removal, as it preserved sample size while stabilizing model training dynamics.

**Feature Engineering**

We constructed a comprehensive feature set spanning three conceptual domains: profitability, operational efficiency, and financial structure. Profitability metrics included Return on Assets (ROA), Return on Equity (ROE), and Gross Profitability, computed directly from income statement and balance sheet fundamentals. Operational efficiency indicators encompassed Asset Turnover and Operating Margin, capturing the firm's ability to generate revenue relative to its asset base. Financial structure variables included Debt-to-Assets ratio and log-transformed Total Assets, the latter employed to normalize the heavy-tailed size distribution characteristic of publicly traded companies. Additionally, we engineered growth indicators by calculating year-over-year percentage changes in Sales and Total Assets, providing temporal dynamics that static cross-sectional snapshots omit. These engineered features transformed raw accounting data into economically interpretable ratios aligned with established valuation theory.

**Data Integration**

Our analysis required merging three heterogeneous data sources: Compustat Global (financial fundamentals), proprietary Corporate Culture Scores (organizational attributes), and IBES (analyst consensus recommendations). We implemented a multi-key join strategy, matching records on Global Company Key (GVKEY) for Compustat-Culture linkage and Ticker Symbol for IBES integration, with Fiscal Year serving as the temporal alignment dimension. This integration yielded a unified dataset containing 61,597 company-year observations spanning 2009-2021, with each record containing financial ratios, culture metrics (innovation, integrity, quality, respect, teamwork), and analyst sentiment (MEANREC). To address missing values introduced through the merge process, we applied median imputation separately within the training partition, ensuring that imputation statistics were derived exclusively from training data to prevent information leakage.
 
**Addressing Data Leakage: Temporal Split-First Methodology**

Initial preprocessing efforts inadvertently introduced data leakage through global standardization applied before temporal partitioning. Specifically, computing Z-scores across the entire dataset (2009-2021) and then splitting into train (2009-2019) and test (2020-2021) allowed training data to "see" test set statistics through the global mean and standard deviation. This violates the fundamental principle of temporal validation: training procedures must not access information from future time periods.

**Resolution Strategy:**

We adopted a split-first, fit-on-train, transform-consistently workflow to eliminate leakage:

Temporal Partitioning: Raw, unprocessed data was first divided into training (2009-2019, 82.4%) and test (2020-2021, 17.6%) partitions based strictly on fiscal year, ensuring no information overlap.
Fit Only on Training: All preprocessing transformations—including standardization parameters (mean, standard deviation), imputation values (medians), and Winsorization thresholds (1st/99th percentiles)—were computed exclusively on the training partition.
Transform Consistently: The preprocessing pipeline trained on training data was then applied identically to both training and test partitions, ensuring test data remained strictly held-out during all fitting procedures.

This rigorous temporal isolation ensures that reported performance metrics (MAE, RMSE, R²) reflect genuine out-of-sample generalization rather than artificially inflated accuracy from leaked future information. By maintaining strict temporal separation throughout the preprocessing pipeline, our validation strategy simulates real-world deployment conditions where models must predict future observations without access to contemporaneous or forward-looking data.

**Normalization and Final Scaling**

Following temporal partitioning and leakage-prevention measures, we applied Z-score standardization to all continuous features, transforming each variable to zero mean and unit variance based on training set statistics. This normalization serves three purposes: (1) equalizes feature scales, preventing high-magnitude variables (e.g., Total Assets) from dominating gradient-based optimization; (2) improves numerical stability in matrix operations during model fitting; and (3) facilitates interpretable feature importance comparisons by placing all predictors on a common scale. For tree-based ensemble methods (Random Forest, Gradient Boosting), standardization is less critical due to their invariance to monotonic transformations, but we maintained consistent preprocessing across all model classes to ensure fair comparative evaluation.



## ML Models Identified {#models}


We will focus on supervised learning methods that have been widely applied in finance.


1. **Linear Regression**: A regularized linear model that combines L1 and L2  penalties to handle multicollinearity among financial variables while performing feature selection. It provides interpretability through transparent coefficient estimates. 

2. **Random Forest**: Decision trees that capture nonlinear interactions between financial features. Patel et al. (2015) [10] applied Random Forests to stock price direction prediction and found they outperformed traditional linear models.

3. **Gradient Boosting Machines**:A sequential tree-based method that corrects errors iteratively and is highly effective on tabular financial data. Brockman and Turtle (2020) [11] used gradient boosting to improve relative valuation estimates compared to traditional comparable-company multiples

<span style="color:#A95FD0; font-weight:bold; font-size:25px;">Midterm- Linear Regression Model</span>

As an initial supervised learning baseline, we implemented an Ordinary Least Squares (OLS) Linear Regression model to predict the Earnings-to-Price (E/P) ratio from fundamental financial ratios.

Using the preprocessed datasets generated by preprocessing.py:
  -  Train: train.parquet (2010–2022) 
  -  Test: test.parquet (2023–2025)
    
All features in these files are already cleaned, winsorized, imputed, and standardized.

Model Setup:

  - We define:
    
      - Input features (X): 
          - roe – profitability relative to equity 
          - roa – profitability relative to total assets 
          - debt_to_eq – capital structure and leverage risk 
          - sales_growth – top-line growth dynamics
            
      - Output features (y):
          - ep – Earnings-to-Price ratio
            
Using scikit-learn's LinearRegression, we: 

  1. Load the processed train/test parquet files. 
  2. Extract X_train, y_train, X_test, y_test using the four features and ep. 
  3. Fit the model on the training set only: LRModel.fit(X_train, Y_train) 
  4. Generate predictions on both train and test sets for evaluation.

Linear Regression was selected as our baseline model because it offers a clear, interpretable framework for understanding how each financial fundamental contributes to a company's valuation. In the context of predicting the Earnings-to-Price (E/P) ratio, interpretability is crucial—each model coefficient directly quantifies how changes in profitability (ROE, ROA), leverage (Debt-to-Equity), or growth (Sales Growth) influence valuation outcomes. This transparency allows us to validate that the learned relationships align with established financial theory (for example, higher profitability or lower leverage typically corresponds to higher valuations). Beyond interpretability, Linear Regression is computationally efficient and provides a stable foundation for comparison. As a low-complexity model, it minimizes the risk of overfitting and establishes a performance benchmark to evaluate whether more advanced models—such as Random Forests, Gradient Boosting Machines, or Elastic Net Regression—meaningfully improve predictive accuracy through capturing nonlinear relationships.

Evaluation Metrics

We evaluate model performance on both the training and test sets using: 

  1. MAE (Mean Absolute Error): Average absolute deviation of predicted vs. actual E/P. Robust, easy to interpret. 
  2. RMSE (Root Mean Squared Error): Penalizes larger errors more heavily; highlights extreme misevaluations. 
  3. R² (Coefficient of Determination): Measures the proportion of variance in E/P explained by the four features.
     
 This Linear Regression model serves as our primary benchmark against which we will compare more flexible machine learning methods in later stages of the project.




<span style="color:#7F9161; font-weight:bold; font-size:25px;">Final - Linear Regression Model (Extended Feature Set)</span>

### Model 1: Linear Regression (Baseline)

Linear Regression models the target variable (Earnings-to-Price ratio, EP) as a weighted sum of input features plus an intercept:

### Dataset

Using the preprocessed datasets:
  - **Input**: `data/finalParquet/train.parquet` (2009-2019, 50,731 observations)
  - **Output**: `data/finalParquet/test.parquet` (2020-2021, 10,866 observations)

All features in these files are already cleaned, winsorized, imputed, and standardized using z-score normalization.

### Model Setup

**Input Features (X)** - 14 engineered features:

**Financial Fundamentals (9 features)**:
  - `gross_profitability` – Efficiency of production and pricing strategies
  - `operating_margin` – Operational efficiency after all operating expenses
  - `roe` – Return on Equity: profitability relative to shareholder equity
  - `roa` – Return on Assets: profitability relative to total assets
  - `sales_growth` – Year-over-year revenue growth rate
  - `asset_growth` – Year-over-year asset expansion
  - `debt_to_assets` – Financial leverage and default risk
  - `log_assets` – Company size (log-transformed for normalization)
  - `asset_turnover` – Revenue generation efficiency per dollar of assets

**Corporate Culture Metrics (5 features)**:
  - `innovation` – Corporate culture score for innovation emphasis
  - `integrity` – Corporate culture score for ethical practices
  - `quality` – Corporate culture score for quality focus
  - `respect` – Corporate culture score for workplace culture
  - `teamwork` – Corporate culture score for collaboration

**Output Feature (y)**:
  - `ep` – Earnings-to-Price ratio (target variable for valuation prediction)

### Implementation

Using scikit-learn's LinearRegression:

  1. Load the processed train/test parquet files using `pd.read_parquet()`
  2. Extract feature matrix `X_train`, target vector `y_train` from training data
  3. Extract feature matrix `X_test`, target vector `y_test` from testing data  
  4. Fit the model on the training set only: `lr_model.fit(X_train, y_train)`
  5. Generate predictions on both sets for evaluation

```python
from sklearn.linear_model import LinearRegression

# Load preprocessed data
X_train, y_train = load_data('train.parquet')
X_test, y_test = load_data('test.parquet')

# Train model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Generate predictions
y_pred_train = lr_model.predict(X_train)
y_pred_test = lr_model.predict(X_test)
```

### Why Linear Regression?

Linear Regression serves as our **baseline model** for three critical reasons:

1. **Interpretability**: Each coefficient βⱼ directly quantifies how a one-standard-deviation change in feature j affects EP, holding all other features constant. For example, if β_ROA = 0.15, a 1σ increase in ROA increases EP by 0.15 units. This transparency allows validation against financial theory (e.g., higher profitability should increase valuation).

2. **Computational Efficiency**: The closed-form OLS solution trains in milliseconds on 50,000+ observations, enabling rapid prototyping and hyperparameter experimentation without expensive grid searches.

3. **Performance Benchmark**: As a low-complexity model with minimal inductive bias, Linear Regression establishes a performance floor. If more complex models (Random Forest, Gradient Boosting) significantly outperform it, we confirm that nonlinear relationships and feature interactions are present and worth the added complexity.

**Evaluation Metrics:**

We evaluate model performance on both the training and test sets using:

  1. MAE (Mean Absolute Error): Average absolute deviation of predicted vs. actual E/P. Robust to outliers and easy to interpret in original units.
  2. RMSE (Root Mean Squared Error): Penalizes larger errors more heavily than MAE; highlights extreme misevaluations and is sensitive to outlier predictions.
  3. R² (Coefficient of Determination): Measures the proportion of variance in E/P explained by the 14 features. Values range from 0 (no explanatory power) to 1 (perfect prediction).


<span style="color:#7F9161; font-weight:bold; font-size:25px;">Final- Gradient Boosting Model</span>


Gradient Boosting is an ensemble method that builds a strong predictor by sequentially combining weak learners (shallow decision trees). The model constructs an additive function.


### Dataset

Using the same preprocessed datasets:
  - **Input**: `data/finalParquet/train.parquet` (2009-2019, 50,731 observations)
  - **Output**: `data/finalParquet/test.parquet` (2020-2021, 10,866 observations)

Features remain identical to the Linear Regression model (14 engineered features, all standardized).

### Model Setup

**Input Features (X)** - Same 14-feature set as Linear Regression:

**Financial Fundamentals (9 features)**:
  - Financial ratios: `gross_profitability`, `operating_margin`, `roe`, `roa`, `sales_growth`, `asset_growth`, `debt_to_assets`, `log_assets`, `asset_turnover`

**Corporate Culture Metrics (5 features)**:
  - Cultural metrics: `innovation`, `integrity`, `quality`, `respect`, `teamwork`

**Output Feature (y)**:
  - `ep` – Earnings-to-Price ratio

### Implementation

Using scikit-learn's GradientBoostingRegressor with hyperparameter tuning via GridSearchCV:

  1. Define parameter grid covering 243 configurations across 7 hyperparameters
  2. Perform 5-fold cross-validation on training data to select optimal hyperparameters
  3. Train final model with best configuration: `best_gb.fit(X_train, y_train)`
  4. Generate predictions on both train and test sets for evaluation

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [10, 20, 50],
    'min_samples_leaf': [5, 10, 20],
    'subsample': [0.5, 0.7, 0.9],
    'max_features': [0.5, 0.7, 1.0]
}

# Grid search with 5-fold CV
gb_model = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(gb_model, param_grid, cv=5, 
                          scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Train final model with best hyperparameters
best_gb = grid_search.best_estimator_
y_pred_test = best_gb.predict(X_test)
```

### Why Gradient Boosting?

Gradient Boosting addresses all limitations of Linear Regression:

1. **Nonlinear Relationship Capture**: Trees automatically learn threshold effects (e.g., debt is benign below 50% but harmful above) without manual feature engineering. Sequential learning directly targets prediction errors, progressively building a model that fits increasingly subtle patterns.

2. **Feature Interaction Detection**: Trees naturally model multiplicative interactions (e.g., growth × margin) through hierarchical splits. The algorithm captures higher-order interactions without explicit specification.

3. **Robustness to Outliers**: Tree-based splits are immune to extreme feature values, unlike linear models where outliers distort coefficients. This makes GBM ideal for financial data with heavy-tailed distributions.

4. **Heteroscedasticity Handling**: Prediction uncertainty can vary by region of feature space, which trees accommodate naturally. The bias reduction mechanism, combined with regularization through learning rate shrinkage and tree depth limits, achieves a superior bias-variance trade-off.

5. **Empirical Success**: Research by Gu, Kelly & Xiu (2020) in "Empirical Asset Pricing via Machine Learning" demonstrated that gradient boosting outperforms linear models in capturing complex asset pricing relationships, motivating its application to our company valuation problem.


**Evaluation Metrics:**

We evaluate model performance using the same metrics as Linear Regression for direct comparison:

  1. MAE (Mean Absolute Error): Average absolute prediction error
  2. RMSE (Root Mean Squared Error): Root mean squared prediction error (penalizes large errors)
  3. R² (Coefficient of Determination): Proportion of variance explained

Gradient Boosting serves as our primary production model due to its superior ability to capture the complex, nonlinear structure of company valuations.

<span style="color:#7F9161; font-weight:bold; font-size:25px;">Final- Random Forest Model</span>

Random Forest is a **bagging ensemble** that builds multiple independent decision trees using bootstrap sampling and random feature selection. The final prediction averages individual tree predictions. 

### Dataset

Using the same preprocessed datasets:
  - **Input**: `data/finalParquet/train.parquet` (2009-2019, 50,731 observations)
  - **Output**: `data/finalParquet/test.parquet` (2020-2021, 10,866 observations)

Features identical to previous models (14 engineered features, standardized).

### Model Setup

**Input Features (X)** - Same 14-feature set as Linear Regression and Gradient Boosting:

**Financial Fundamentals (9 features)**:
  - Financial ratios: `gross_profitability`, `operating_margin`, `roe`, `roa`, `sales_growth`, `asset_growth`, `debt_to_assets`, `log_assets`, `asset_turnover`

**Corporate Culture Metrics (5 features)**:
  - Cultural metrics: `innovation`, `integrity`, `quality`, `respect`, `teamwork`

**Output Feature (y)**:
  - `ep` – Earnings-to-Price ratio

### Implementation

Using scikit-learn's RandomForestRegressor with hyperparameter tuning via GridSearchCV:

  1. Define parameter grid covering 384 configurations across 6 hyperparameters
  2. Perform 5-fold cross-validation on training data to optimize hyperparameters
  3. Train final model with best configuration: `best_rf.fit(X_train, y_train)`
  4. Generate predictions and extract out-of-bag score for validation

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'max_features': ['sqrt', 'log2', 0.5, 1.0],
    'bootstrap': [True, False]
}

# Grid search with 5-fold CV
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1, oob_score=True)
grid_search = GridSearchCV(rf_model, param_grid, cv=5,
                          scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Train final model with best hyperparameters
best_rf = grid_search.best_estimator_
y_pred_test = best_rf.predict(X_test)
```

### Why Random Forest?

Random Forest was selected to provide a complementary nonlinear approach to Gradient Boosting. While both are tree-based ensembles, they differ in construction:

1. **Complementary Architecture**: GBM builds trees sequentially to correct errors, while Random Forest builds trees independently in parallel. This architectural difference offers several advantages:
   - **Overfitting resistance** through bootstrap sampling and feature randomness
   - **Parallel training** enabling 25% faster training time compared to GBM
   - **Built-in validation** via out-of-bag error estimates
   - **Robustness** with stable performance across different hyperparameter settings

2. **Diagnostic Tool**: Comparing RF performance to GBM reveals whether sequential error correction (boosting) provides significant advantages over parallel averaging (bagging) for our valuation problem.

3. **Feature Importance Validation**: RF provides an alternative feature importance perspective through impurity-based metrics, complementing GBM's analysis. If both ensemble methods identify the same features as important, we gain confidence in those findings.

4. **Production Efficiency**: RF's fast parallel training makes it suitable for frequent model retraining in production environments, while its lower sensitivity to hyperparameter choices reduces tuning costs.

### Key Mechanisms

1. **Bootstrap Aggregating (Bagging)**: Training each tree on a different random subsample reduces variance by decorrelating tree predictions. Averaging uncorrelated predictions yields lower variance than any individual tree.

2. **Feature Randomness**: Restricting feature choices at each split forces trees to explore different feature subspaces, further decorrelating predictions.

3. **Out-of-Bag (OOB) Validation**: Each tree is trained on ~63.2% of data (bootstrap sample). The remaining ~36.8% (OOB sample) provides a free validation set for unbiased error estimation.

4. **Parallel Training**: Unlike Gradient Boosting's sequential construction, Random Forest trees are independent and can be trained in parallel, reducing wall-clock time.


**Evaluation Metrics:**

We evaluate using the same metrics for consistent comparison:

  1. MAE (Mean Absolute Error): Average absolute prediction error
  2. RMSE (Root Mean Squared Error): Root mean squared prediction error
  3. R² (Coefficient of Determination): Proportion of variance explained
  4. OOB R²: Out-of-bag validation R² (built-in honest estimate)

Random Forest serves as a valuable robustness check, confirming that nonlinear methods substantially outperform Linear Regression while revealing how different ensemble architectures perform on our valuation task.


## Results and Discussion {#results}

This section presents the results of our machine learning approaches to predicting the Earnings-to-Price (E/P) ratio using company financial fundamentals. We evaluated three distinct modeling approaches: Linear Regression (baseline), Random Forest, and Gradient Boosting. Each model was trained on identical preprocessed datasets containing financial metrics such as profitability ratios, growth indicators, leverage metrics, and corporate culture scores.


<span style="color:#A95FD0; font-weight:bold; font-size:25px;">Midterm- Results from Linear Regression </span>

### Visualizations Midterm

<div style="margin: 2rem 0;">
  
  <!-- Figure 1 -->
  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/TestSet.png' | relative_url  }}" alt="Model Performance Comparison" style="max-width: 70%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 3: Scatterplot of predicted vs actual E/P values</p>
    <p style="font-size: 0.9rem; text-align: left; margin-top: 0.5rem; max-width: 800px; margin-left: auto; margin-right: auto;">
      <strong>Why is this important:</strong><br>
      The following plot compares actual E/P values (x-axis) with model-predicted E/P values (y-axis). If the model were perfectly accurate, all points would lie exactly on the red dashed diagonal line (the line of perfect prediction where predicted value == actual value).<br><br>
      
      <strong>Observations:</strong><br>
      Both axes range from approximately –5 to 0. Majority of datapoints are clustered near the horizontal center of the graph, around predicted values close to zero. This indicates that the model tends to output predictions near the mean E/P value (near 0 for most observations) regardless of the true E/P, reflecting a bias toward average outcomes.<br><br>
      
      <strong>Possible causes:</strong><br>
      • Relationship between fundamentals and valuation (E/P) might be nonlinear<br>
      • Small feature set might omit many factors that influence valuation<br>
      • Noise and reporting variability in accounting data<br><br>
      
      <strong>Interpretation/implications:</strong><br>
      • Model underfitting and weak predictive spread suggest need for expanded feature engineering and potentially nonlinear methods. <br>
      • Weak predictive spread: since the model's predictions remain close to zero, it effectively learns an average E/P level rather than differentiating across firms. This might be because the chosen features have weak or noisy relationships with E/P or that true underlying relationship is nonlinear. Outliers or skewed distributions in E/P can also distort the fitted coefficients.<br>
      • Asymmetric performance: The model performs particularly poorly for firms with large negative E/P values, suggesting it struggles with asymmetric or heavy-tailed financial data.<br>
      • Need for expanded feature engineering: The results indicate that additional variables such as profitability margins, valuation multiples, or size controls might be necessary to capture the differences in E/P. We might also need to explore nonlinear methods.<br><br>
      
    </p>
  </div>
  
  <!-- Figure 2 -->
  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/Residual.png' | relative_url }}" alt="Residual Plot" style="max-width: 70%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 4: Residuals vs Predicted E/P</p>
    <p style="font-size: 0.9rem; text-align: left; margin-top: 0.5rem; max-width: 800px; margin-left: auto; margin-right: auto;">
      <strong>Why is this important:</strong><br>
      The residual plot visualizes the relationship between the model's predicted E/P values and the corresponding residuals (actual minus predicted). Ideally, residuals should be randomly scattered around zero with no discernible pattern, indicating that the model's errors are evenly distributed and that the linear assumptions hold.<br><br>

      <strong>Observations:</strong><br>
      • Triangular spread: the residuals form a triangular pattern, being widely dispersed for more negative predicted E/P values and tightly clustered near zero as predictions approach zero. This indicates heteroscedasticity, which means that the model's error variance increases as the predicted value becomes more extreme. In other words, the model performs relatively consistently for firms with typical or near-zero E/P ratios but becomes increasingly unreliable for firms with large negative earnings yields.<br>
      • Systematic underestimation of negative E/P values: the visible diagonal cluster of points in the lower-left region (extending from approximately (–2.75, –2) to (0, –4.5)) suggests a systematic bias. For firms with strongly negative actual E/P values, the model tends to underpredict the magnitude of the losses, yielding large negative residuals. This pattern reinforces the results from the previous plot: the linear model is unable to capture the nonlinear dynamics that drive extremely low valuations.<br>
      • Dense concentration around (0, 0): near the center of the plot, residuals are tightly clustered around zero, implying that the model performs moderately well for the bulk of the data, companies with stable or near-zero E/P ratios.<br>
      • Lack of symmetry and randomness: the residuals are not symmetrically distributed around the zero line, nor do they appear random. Instead, clear structural patterns are present. This violates a key assumption of linear regression - that residuals are independent, identically distributed, and centered around zero - which further suggests that the linear model is inadequate.<br><br>
      
      <strong>Possible causes:</strong><br>
      • Relationship between fundamentals and valuation (E/P) might be nonlinear<br>
      • Small feature set might omit many factors that influence valuation<br>
      • Noise and reporting variability in accounting data<br><br>
      
      <strong>Implications/Next steps:</strong><br>
      The linear model systematically underestimates firms with extreme negative valuations. The presence of heteroscedasticity means that the variance of errors depends on the scale of the predicted E/P, which linear regression does not handle well. The model likely misses nonlinear relationships or interaction effects among financial ratios. To improve, we can take the following steps:
      • Introduce nonlinear or composite features to help capture more complex relationships. <br>
      • Experiment with tree-based models such as Random Forests, Gradient Boosting, or nonlinear regressors (SVR) that can model curvature and variable interactions automatically. <br>
      • We could attempt to model residuals separately or apply techniques like weighted least squares to address heteroscedasticity that is present. <br>
      • We could also try to evaluate the model separately for profitable vs unprofitable firms, as the E/P relationship may differ fundamentally between these groups. <br>


    </p>
  </div>
  
  <!-- Figure 3 -->
  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/Linear Regression Coefficients.png' | relative_url }}" alt="Coefficient Plot" style="max-width: 70%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 5: Feature coefficient chart</p>
    <p style="font-size: 0.9rem; text-align: left; margin-top: 0.5rem; max-width: 800px; margin-left: auto; margin-right: auto;">
      <strong>Why is this important:</strong><br>
      The bar chart of linear regression coefficients gives insight into the direction and relative importance of each predictor variable in explaining variation in the E/P ratio. In this model, the coefficients represent the expected change in E/P for a one-unit standardized increase (one std deviation) in each feature, holding all other features constant.<br><br>
      
      <strong>Observations:</strong><br>
      • ROA (≈ 0.22): ROA has the largest positive coefficient among all features, indicating that it has the strongest influence on the model's predicted E/P values. This suggests that firms generating higher earnings relative to total assets tend to have higher E/P ratios, which makes sense intuitively: more efficient asset utilization generally signals profitability and stability.<br>
      • ROE (≈ 0.187) ROE also contributes positively to predicted E/P, reflecting that companies earning higher returns on shareholders' equity are typically valued more favorably. The slightly smaller coefficient compared to ROA could imply that ROA captures a broader and more consistent signal than ROE, which can be more volatile.<br>
      • Debt-to-Equity Ratio (≈ 0.037) The positive but relatively small coefficient indicates a weak direct relationship between leverage and E/P ratio in this dataset. A modest positive value might suggest that, within certain bounds, higher Debt-to-Equity is associated with higher valuation yield. However, the low magnitude might suggest that the effect of this feature is complex and not strongly linear.<br>
      • Sales Growth (≈ 0.02) Sales growth has the smallest coefficient, implying that recent growth in revenue has limited explanatory power for current E/P ratios in the sample.<br><br>
      
      <strong>Possible causes:</strong><br>
      • Relationship between fundamentals and valuation (E/P) might be nonlinear<br>
      • Small feature set might omit many factors that influence valuation<br>
      • Noise and reporting variability in accounting data<br><br>
      
      <strong>Interpretation/ Next Steps:</strong><br>
      The model's predictions rely primarily on profitability indicators (ROA and ROE), while sales growth and debt-to-equity contribute marginally. This makes intuitive sense but also indicates that the model's feature space is too narrow, it captures only a small slice of what drives company valuations. The low overall predictive power implies that even though these coefficients are interpretable, the linear relationships are too simple to explain complex valuation dynamics.
      • Expand the feature set by incorporating additional accounting ratios<br>
      • Explore feature interactions: nonlinear interactions between profitability, leverage, and growth might be critical.<br>
      • Try nonlinear models such as Random Forests, Gradient Boosting or Neural Networks can automatically learn nonlinear feature interactions that linear regression misses.br>
      • When using tree-based methods, we can compute feature importance or SHAP values to understand which variables truly drive the predictions across different firms and conditions.<br><br>      
    </p>
  </div>
  
  <!-- Figure 4 -->
  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/Dist.png' | relative_url }}" alt="Distribution Plot" style="max-width: 70%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 6: Residual Distribution Histogram</p>
    <p style="font-size: 0.9rem; text-align: left; margin-top: 0.5rem; max-width: 800px; margin-left: auto; margin-right: auto;">
      <strong>Why is this important:</strong><br>
      The histogram of residuals (prediction errors) shows how the model's deviations from actual E/P values are distributed across the test set. Ideally, in a well-performing linear regression model, residuals should be centered around zero and roughly symmetrical, following a bell-shaped (normal) distribution. This would indicate that the model's errors are unbiased and consistent across the dataset. In our case, it seems to show a non-normal and highly skewed pattern.<br><br>
      
      <strong>Observations:</strong><br>
      • Sharp peak near zero, exceeding 10,000 observations. This indicates that for a large number of companies, the model's predictions are almost exactly equal to the actual E/P or only slightly off. This heavy central concentration means the model is effectively "collapsing" predictions toward a narrow range (around zero), producing many small residuals simply because it predicts near the overall mean for most firms.<br>
      • Long left tail or negative skew: as the histogram extends leftward, the residual frequency decreases gradually, meaning there are many cases where the actual E/P is much lower than the predicted value. This confirms that the model systematically underestimates companies with strongly negative E/P ratios. In other words, it fails to capture extreme losses.<br>
      • The right tail, in contrast, drops off sharply, showing that the model rarely overpredicts by large margins. This asymmetry indicates a directional bias: predictions are more likely to be too high (relative to the true value) than too low.<br>
      • Non-gaussian shape: distribution is not bell-shaped but rather spiked and skewed, suggesting that the residuals are not normally distributed. This violates one of the key assumptions of linear regression (normality of errors) and implies that the linear model's functional form may not be appropriate for the underlying data.<br><br>
      
      <strong>Possible causes:</strong><br>
      • Relationship between fundamentals and valuation (E/P) might be nonlinear<br>
      • Small feature set might omit many factors that influence valuation<br>
      • Noise and reporting variability in accounting data<br><br>
      
      <strong>Implications and Next Steps:</strong><br>
      The LR model is biased toward the mean, producing predictions centered around zero. The residuals show negative skewness, confirming that the model systematically underestimates extreme negative valuations. The assumption of normally distributed residuals does not hold which is a sign that the relationship between financial ratios and valuation is nonlinear. Next Steps:
      • Feature expansion.<br>
      • Apply nonlinear models.<br><br>
    </p>
  </div>

  <!-- Figure 5 -->
  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/colorcheck.png' | relative_url }}" alt="Correlation Matrix" style="max-width: 70%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 7: Feature Correlation Matrix</p>
    <p style="font-size: 0.9rem; text-align: left; margin-top: 0.5rem; max-width: 800px; margin-left: auto; margin-right: auto;">
      <strong>Why is this important: Feature Correlation Matrix</strong><br>
      The correlation heatmap visualizes the pairwise linear relationships among the five variables used in the model: ROE, ROA, Debt-to-Equity Ratio, Sales Growth, and the E/P ratio. Each cell in the heatmap represents the Pearson correlation coefficient between two variables, ranging from –1 (perfect negative correlation) to +1 (perfect positive correlation).<br><br>
      
      <strong>Observations:</strong><br>
      The heatmap reveals that while profitability variables (ROE, ROA) are meaningfully related to valuation (E/P), leverage and growth measures are weakly linked. Furthermore, the modest correlations suggest that the true relationship between financial fundamentals and valuation may be nonlinear or interaction-based, which a simple linear model cannot fully capture.<br><br>
    </p>
  </div>

  <!-- Figure 6 -->
  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/corel.png'| relative_url  }}" alt="Feature-Target Correlation" style="max-width: 70%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 8: Feature-Target Correlation</p>
    <p style="font-size: 0.9rem; text-align: left; margin-top: 0.5rem; max-width: 800px; margin-left: auto; margin-right: auto;">
      <strong>Why is this important: Feature-Target Correlation</strong><br>
      This output is the numerical correlation of each feature with the target variable (E/P). To quantify the linear relationship between each explanatory variable and the target variable (E/P ratio), a Pearson correlation analysis was performed using all numerical features in the dataset (ignore gvkey, non-related identifier).<br><br>
      
      Same observation as the previous plots: E/P is only moderately correlated with traditional profitability measures and largely uncorrelated with leverage or growth, at least in the linear sense.<br><br>
      
      <strong>Implications:</strong><br>
      • The relationship between firm fundamentals and valuation may be nonlinear or influenced by interactions not captured by simple regression<br>
      • Or the feature set is incomplete.<br><br>
    </p>
  </div>

  <!-- Figure 7 -->
  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/4data.png'| relative_url }}" alt="Feature vs Target" style="max-width: 70%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 9: Feature vs Target Relationship Analysis</p>
    <p style="font-size: 0.9rem; text-align: left; margin-top: 0.5rem; max-width: 800px; margin-left: auto; margin-right: auto;">
      <strong>Why is this important:</strong><br>
      This visualization displays the relationship between each individual financial feature and the target variable (E/P ratio) using a 2x2 grid of scatter plots. Each subplot shows how one feature (ROE, ROA, debt-to-equity, or sales growth) correlates with the E/P ratio on the test dataset.<br><br>

      <strong>Observations:</strong><br>
      • ROE (Return on Equity): Shows a positive linear relationship with E/P ratio, Clear upward trend: higher ROE values correspond to higher E/P ratios, Dense clustering around ROE = 0, with significant spread throughout, Notable horizontal band at E/P ≈ -5 and vertical band at ROE ≈ 0<br>
      • ROA: Demonstrates a strong positive correlation with E/P, similar to ROE, Clear linear pattern visible, particularly at higher values, Heavy concentration of points near ROA = 0, Horizontal band at E/P ≈ -5 indicates potential outliers or data artifacts<br>
      • Debt-to-Equity Ratio: Shows no clear linear relationship with E/P ratio, points heavily concentrated near debt-to-eq = 0, wide vertical spread across all E/P values at low debt levels, horizontal bands at E/P ≈ 0 and E/P ≈ -5 suggest categorical groupings<br>
      • Sales Growth: Exhibits minimal to no relationship with E/P, extremely concentrated around sales_growth = 0, almost entirely vertical scatter with no discernible pattern<br><br>

      <strong>Implications:</strong><br>
      • Strong Predictors: ROE and ROA show clear positive linear relationships with E/P, making them the most valuable features for the linear regression model.<br>
      • Weak Predictors: Debt-to-equity and sales growth show minimal linear relationships, suggesting they contribute less to prediction accuracy as standalone features.<br>
      • Feature Concentration: Heavy clustering near zero for all features indicates many companies have near-zero or small values, which could affect model training and may benefit from outlier handling or feature transformation.<br><br>
    </p>
  </div>
  

      
</div>

- ### Quantitative Metrics – Observations and Discussion

  <div style="text-align: center;">
    <img src="{{ '/assets/images/quantdata.png' | relative_url  }}" alt="Model Performance Comparison" style="max-width: 50%; width: 100%;">
    <p style="font-style: italic; color: #666;">Figure 10: Testing set Data </p>
  </div>


To evaluate the performance of the linear regression model predicting E/P ratios, three standard metrics were computed: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination (R^2).

**Mean Absolute Error (MAE):**

MAE measures the average magnitude of prediction errors, ignoring their direction.

- Ideal value: Close to 0, indicating predictions are very close to actual values.
- Observation: MAE is moderate on both train and test sets. Indicates that, on average, predictions deviate from actual E/P by a meaningful amount, consistent with earlier visualizations showing the model "collapsing" predictions near zero.

**Root Mean Squared Error (RMSE):**

RMSE measures the square root of the average squared prediction errors, penalizing larger errors more heavily.

- Ideal value: Close to 0. Lower RMSE indicates fewer large deviations.
- Observation: RMSE is higher than MAE, reflecting the presence of large residuals, particularly for firms with extreme negative E/P values. This confirms visual findings from the residual and histogram plots: the model struggles with outliers and extreme cases.

**Coefficient of Determination (R^2):**

R^2 quantifies the proportion of variance in the target variable explained by the model.
- Ideal value: 1.0 (perfect fit), 0 indicates the model performs no better than predicting the mean, negative values indicate worse-than-mean predictions.
- Observation: R^2 is low on both train and test sets (26% of the cases explained by current features), indicating that the linear model explains only a small fraction of the variability in E/P ratios. Consistent with earlier observations, the model captures the average trend but fails to account for firm-specific or extreme values.


  <!-- Figure 5 -->
  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/performance_metrics_visual.png' | relative_url }}" alt="Correlation Matrix" style="max-width: 70%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 11: Regression Preformance</p>
    <p style="font-size: 0.9rem; text-align: left; margin-top: 0.5rem; max-width: 800px; margin-left: auto; margin-right: auto;">
      <strong>Why is this important: Regression Preformance</strong><br>
 This plot shows MAE and RMSE are both higher on the test set, meaning the model makes larger prediction errors (but not by too much) on unseen data. The difference in RMSE between the training and test sets is larger than the difference in MAE, indicating that the model's errors increase disproportionately for certain observations. Since RMSE penalizes larger errors more heavily, this suggests the model struggles with outliers or extreme E/P values, producing occasional large deviations even though average errors (MAE) remain moderate. R² is similar (0.21 train vs 0.26 test), indicating the model explains only about 20–26% of the variance, suggesting weak predictive power overall. Because the train and test errors are not drastically different, the model is not overfitting, but it's also not capturing strong patterns ie it's a relatively simple or underfit model. <br><br>
      </p>
  </div>

  

- ### Analysis of Linear Regression: 

**Analysis of the Linear Regression Model**

This model serves as a baseline to predict the E/P ratio using four financial features: ROE, ROA, debt-to-equity ratio, and sales growth. From our results and visualizations, several patterns emerge that help us understand the model's performance:

**Predictive behavior:**

The scatterplot of actual vs. predicted E/P values shows that most predictions are clustered near zero, regardless of the actual E/P. This indicates the model tends to output the mean value for most firms, underestimating extreme negative E/P values. This pattern demonstrates underfitting ie the model cannot capture the full variability in the target variable.

**Residual patterns:**

The residual plot reveals a triangular shape with wider spread for extreme negative predicted values. There is a clear diagonal cluster of residuals for very negative actual E/P, suggesting systematic underestimation. Near zero, residuals are tighter, indicating better performance for typical or average firms. These patterns reveal heteroscedasticity and confirm that linear assumptions are violated for extreme values.

**Feature importance:**

The coefficient chart shows that ROA and ROE are the primary drivers of predictions, while debt-to-equity and sales growth contribute minimally. This aligns with the correlation analysis, where ROA and ROE have moderate correlations with E/P, and the other features show weak relationships. This explains why the model captures only general trends and fails for extreme cases.

**Residual distribution:**

The residual histogram confirms that the model predicts near-average E/P values and underestimates losses. The distribution is skewed and non-Gaussian, violating assumptions of linear regression and indicating that the relationship between features and E/P is likely nonlinear.
Quantitative metrics:

MAE and RMSE are moderate, while R² is low (~0.26), confirming that the model explains only about 26% of the variance in E/P. Together with the visualizations, this suggests that linear regression captures general trends but fails to account for firm-specific or extreme valuations.

**Final conclusion/TLDR:**

The linear regression model provides an interpretable baseline, showing that profitability metrics moderately drive E/P predictions. However, it underfits the data, struggles with extreme negative values, and cannot capture nonlinear or interaction effects between features. This analysis motivates exploring nonlinear models, additional features, and handling heteroscedasticity to improve predictive performance.


- ### Next Steps: 


**1. Feature Engineering:**
- Include additional financial ratios and metrics such as profit margins (gross, operating, net), Return on Invested Capital (ROIC), Earnings volatility (standard deviation of earnings over time), Market-related metrics (beta, market cap, volatility), Valuation multiples (P/B, P/S, EV/EBITDA).
- Explore feature interactions and transformations (log, standardization) to capture nonlinear relationships.
- Handle outliers or extreme negative E/P values via capping, winsorizing, or weighted models.

**2. Model Selection/Improvement:**
- Try nonlinear or ensemble models like Random Forests, Gradient Boosting, SVR, or neural nets.
- Test Ridge, Lasso, or ElasticNet LR models to reduce overfitting and improve feature selection.
- Address heteroscedasticity using weighted least squares or residual modeling.

**3 Evaluation & Interpretability:**
- Use additional metrics like Mean Absolute Percentage Error (MAPE) to understand relative prediction errors
- Visualize residuals after nonlinear modeling to ensure model captures extreme values better.
- For tree-based or nonlinear models, compute feature importance or SHAP values to identify which variables truly drive E/P predictions.
- Evaluate model performance across firm subgroups (profitable vs unprofitable, sectors).

**4. Data Segmentation & Other approaches:**
- Clustering/unsupervised models: we could cluster firms based on financial features to identify groups where E/P relationships differ. This could help us figure out whether separate models per cluster improve predictive accuracy.
- Incorporate time-related dynamics (previous quarter earnings, rolling averages) to improve predictions.


- ### Evaluation Metrics

To evaluate the performance of our models, we will rely on several quantitative metrics:

1. **Mean Absolute Error (MAE)**
2. **Root Mean Squared Error (RMSE)**  
3. **R² (Coefficient of Determination)**

### Project Goals

Our primary goal is to build a model that achieves lower error rates (MAE, RMSE) than simple baselines such as average sector multiples or linear regressions.
Beyond predictive accuracy, we plan to emphasize sustainability and ethical considerations where models should not overweight short-term sentiment signals at the expense of long-term fundamentals, and transparency in model explanations


### Expected Results

We expect that nonlinear methods like Gradient Boosting Machines will outperform simpler linear models. As a concrete goal, we hope our best-performing model will reduce prediction error by 15–20% relative to traditional methods.

<span style="color:#7F9161; font-weight:bold; font-size:25px;">Final- Visualizations For 3 Models</span>

### Visualizations

### Gradient Boosting Visualizations:

  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/ActualvPredGB.png'| relative_url }}" alt="Feature vs Target" style="max-width: 100%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 12: Predicted vs Actual GBM </p>
  </div>

**Interpretation:**
- Most points cluster around the 45° line, showing the model captures the general E/P relationship reasonably well across typical values.
- The model exhibits clear **regression-to-the-mean** behavior:
  - High E/P cases (actual > 0.5) are systematically underpredicted—predictions get compressed toward the center.
  - Low E/P cases (actual < -0.5) are overpredicted, showing the same conservative pattern.
- This happens because even though Gradient Boosting learns sequentially, the shallow trees (max_depth=3) limit how far any single iteration can push predictions. After 300 rounds of small adjustments, the model still struggles to reach truly extreme values that appear rarely in the training data.
- **Key observations:**
  - Vertical banding appears at specific E/P values (around 0.0, 0.25), indicating the model frequently predicts these values when feature combinations are similar.
  - Prediction scatter increases noticeably at the extremes, showing higher uncertainty for firms with unusual financial profiles.
  - The dense concentration between -0.25 and 0.5 represents where most firms fall and where the model performs best.
  - There's a visible "cloud" of predictions anchored around 0.15-0.25 across varying actual values, suggesting the model defaults toward this range when features don't provide strong signals in either direction.
- **Implication:** The model works well for typical firms in the central E/P range, but becomes conservative and uncertain at the extremes. For companies with very high or very low E/P ratios—whether distressed or exceptionally profitable—the predictions should be viewed as rough directional estimates rather than precise forecasts.



  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/ResidualvPredGB.png'| relative_url }}" alt="Feature vs Target" style="max-width: 70%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 13: Residual v Predicted </p>
  </div>

**Interpretation:**
- The residuals show a clear **V-shaped pattern** centered around predicted values of 0.15-0.20, which is a major concern for model reliability.
- **Systematic bias patterns:**
  - For low predicted E/P (< 0), residuals are predominantly negative, meaning the model overpredicts—it thinks these firms are better than they actually are.
  - For high predicted E/P (> 0.3), residuals are predominantly positive, meaning the model underpredicts—it's too conservative about high-performing firms.
  - The central region (0.1 to 0.25) shows the most balanced residuals, though still with noticeable spread.
- **Heteroscedasticity is evident:** The residual variance is not constant. The V-shape shows errors are smallest in the middle range and grow larger as predictions move toward the extremes in either direction.
- The strong diagonal bands (upper left and lower right) directly reflect the regression-to-the-mean problem seen in the actual vs predicted plot. When the model predicts extreme values, it's systematically wrong in predictable ways.
- **Key issue:** This pattern suggests the model hasn't fully learned the relationships at the extremes. The sequential boosting process keeps making small adjustments toward the center, but 300 iterations with shallow trees can't overcome the fundamental difficulty of predicting rare extreme cases.
- **Implication:** The model's predictions are most trustworthy in the 0.1-0.25 range where residuals center near zero. Outside this range, predictions carry systematic directional bias that users need to account for—particularly when screening for undervalued (high E/P) or overvalued (low E/P) stocks.



  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/AbsErrorVPred.png'| relative_url }}" alt="Feature vs Target" style="max-width: 70%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 14: Absolute Error v Predicted </p>
  </div>

**Interpretation:**
- **Clear V-shaped pattern** with the vertex around predicted E/P of 0.15-0.20, showing lowest absolute errors in this central range.
- The densest concentration of points sits between predicted values of 0.0 and 0.3, with most absolute errors under 0.20 in this zone.
- **Strong diagonal bands** emerge at the extremes:
  - Left side (predicted < -0.2): A tight diagonal band of high errors stretching from 0.25 up to 0.70, showing systematic underprediction of low E/P firms.
  - Right side (predicted > 0.3): Another diagonal band reaching absolute errors of 0.75-1.0, indicating systematic overprediction of high E/P firms.
- These diagonal bands are the absolute value version of the diagonal patterns we saw in the residual plot—they represent the regression-to-the-mean compression manifesting as large errors at extremes.
- **Minimum error floor**: Very few points have absolute error below 0.05, suggesting there's irreducible noise in predicting E/P even for the "easiest" cases.
- The V-shape is asymmetric—the right side (high predictions) shows slightly wider spread and reaches higher maximum errors (~1.0) compared to the left side (~0.70).

**This reveals:**
This visualization makes the model's confidence profile unmistakable. The model is genuinely reliable only for predictions between 0.0 and 0.3, where errors typically stay under 0.15. Outside this comfort zone, uncertainty explodes in a predictable way:
- Low predictions (< -0.2): Errors scale linearly with distance from center, reaching 0.5-0.7
- High predictions (> 0.4): Even worse, with errors reaching 0.8-1.0

The tight diagonal bands at extremes are particularly telling—these aren't random errors, but systematic failures where the model consistently misses by predictable amounts. The model basically says "I think this firm has extreme E/P" but is wrong about the magnitude by a huge margin.

The asymmetry (worse errors on high predictions) aligns with our earlier finding that the model underpredicts high E/P firms more severely than it overpredicts low E/P firms.

**Implications:**
- **Trust zone: 0.0 to 0.3**: Predictions here have typical errors of 0.05-0.15, acceptable for most applications. Use these directly.
- **Caution zone: -0.3 to 0.0 and 0.3 to 0.5**: Errors jump to 0.15-0.30. Predictions are directionally useful but need wider confidence intervals.
- **Danger zone: < -0.3 or > 0.5**: Errors exceed 0.30 and can reach 1.0. These predictions should be treated as rough signals only—manual verification essential before acting on them.
- **For portfolio screens**: If building a long strategy targeting high E/P firms, recognize that predictions above 0.4 are highly unreliable. You'd be better off using the model to filter out clearly bad firms (predicted < 0.1) rather than trying to pick the best ones.
- **Model limitation**: The V-shape is inherent to the shallow-tree boosting approach. To reduce errors at extremes, you'd need either deeper trees (risking overfitting), more extreme training examples, or a fundamentally different model architecture.



  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/MAE BY PREDICTED DECILE.png'| relative_url }}" alt="Feature vs Target" style="max-width: 70%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 15: MAE by Predicted Decile</p>
  </div>

**Interpretation:**
- The error pattern forms a clear **U-shape**, with highest errors at the extremes (deciles 0 and 9) and lowest errors in the middle deciles (3-6).
- **Decile 3** has the lowest MAE (~0.017), showing the model is most accurate when predicting E/P values in this range—roughly the 30th-40th percentile of predictions.
- **Deciles 0 and 9** show dramatically higher errors (~0.173 and ~0.175), meaning predictions for the lowest and highest E/P firms are off by nearly 10x compared to the middle range.
- The middle deciles (3-6) maintain consistently low error between 0.017-0.069, confirming this is the model's "comfort zone" where it has seen enough training examples to make reliable predictions.
- Errors start climbing noticeably in deciles 7-8 before spiking at decile 9, showing the model becomes progressively less confident as it tries to predict higher E/P values.
- **What this means:** The model is heavily optimized for predicting typical firms in the central distribution. When it encounters firms at the tails—either financially distressed (low E/P) or highly profitable (high E/P)—the sequential boosting hasn't seen enough examples to learn these patterns well, even after 300 iterations.
- **Practical implication:** If you're building a stock screen or portfolio strategy, be skeptical of predictions in the extreme deciles. The model is essentially guessing for these edge cases. Focus on firms where predictions fall in deciles 2-7 for more reliable signals.

  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/FEATURE IMPORTANCE.png'| relative_url }}" alt="Feature vs Target" style="max-width: 70%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 16: Feature Importance </p>
  </div>


**Interpretation:**
- **ROA (Return on Assets) dominates** with an importance of ~0.53, accounting for more than half of the model's predictive power. This makes sense—ROA directly measures profitability relative to assets, which is fundamentally what E/P captures.
- **ROE (Return on Equity)** comes in second at ~0.14, reinforcing that profitability metrics are the primary drivers of E/P predictions.
- **Debt-to-assets ratio** ranks third (~0.09), showing that leverage matters for earnings predictions—highly leveraged firms have different risk profiles that affect their E/P ratios.
- The next tier includes **gross profitability** and **operating margin** (~0.04 each), which provide additional nuance about operational efficiency beyond the headline ROA/ROE numbers.
- **Firm size (log_assets)** has modest importance (~0.03), suggesting size matters but isn't a dominant factor once you control for profitability.
- **Culture scores** (innovation, integrity, quality, respect, teamwork) are barely used—all sit near zero importance. The model essentially ignores these features, finding them uninformative for predicting E/P after accounting for financial metrics.
- **MEANREC (analyst consensus)** also has minimal importance (~0.01), which is surprising given that analyst recommendations theoretically incorporate forward-looking information. The model doesn't find this useful beyond what the financial ratios already capture.
- **Growth metrics** (sales_growth, asset_growth) have very low importance, suggesting the model focuses on current profitability rather than growth trajectories.
- **Bottom line:** This is fundamentally a profitability-driven model. The sequential boosting process repeatedly splits on ROA and ROE because they provide the most reliable signal for E/P. Everything else is mostly noise once you know those core metrics. This suggests either the culture and analyst features don't actually predict E/P well, or their signal is already embedded in the profitability ratios that companies with better culture achieve.

  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/PARTIAL DEPENDENCE PLOTS (Top 3 Features).png'| relative_url }}" alt="Feature vs Target" style="max-width: 70%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 17: Partial Dependence Plots Top 3 Features </p>
  </div>

### Partial Dependence — ROA

**Observations:**
- Strong **S-shaped (sigmoid) relationship** between ROA and predicted E/P.
- Three distinct zones:
  - Below ROA = 0: Flat and low (around -0.27), representing unprofitable firms.
  - ROA = 0 to 0.010: Steep upward climb, with predictions jumping from -0.20 to nearly +0.05.
  - Above ROA = 0.010: Plateaus around +0.04, representing highly profitable firms.
- The steepest part of the curve sits right at the profitability threshold (ROA ≈ 0 to 0.005).
- The tick marks show most firms cluster around the 0 to 0.010 range.

**Interpretation:**
The S-curve captures a fundamental business reality: crossing from unprofitable to profitable represents a regime change. Firms with negative ROA are destroying value—their assets generate losses rather than earnings, leading to low E/P predictions. Once ROA turns positive, the sequential boosting rapidly increases predictions, recognizing that profitable operations signal potential value. Beyond ROA = 0.010, the curve flattens—additional profitability helps, but the marginal impact diminishes as the market likely already prices in quality.

The critical zone around ROA = 0 is where small operational improvements create the biggest valuation swings. This makes economic sense: the difference between losing money and making money matters more than incremental gains among already-profitable firms.

**Implications:**
- Companies sitting near breakeven (ROA ≈ 0) are high-leverage opportunities—small efficiency improvements could shift valuations significantly.
- For portfolio construction, filtering out ROA < 0 eliminates the consistently weak prediction zone, focusing capital on profitable businesses.
- The plateau at high ROA suggests the model has learned that beyond a certain point, exceptional profitability is already priced in by the market.


### Partial Dependence — ROE

**Observations:**
- **U-shaped relationship** with a pronounced dip at ROE ≈ 0.
- Negative ROE zone (< -0.005) shows relatively flat predictions around -0.03.
- Sharp drop as ROE approaches zero, bottoming around -0.08 at ROE ≈ 0.
- Rapid recovery as ROE increases beyond zero, climbing to +0.03 at high positive ROE (> 0.010).
- The recovery slope on the positive side is steeper than the negative side is flat.

**Interpretation:**
This U-shape reveals a nuanced pattern: firms near zero ROE get the worst predictions. These are "zombie" companies—barely breaking even, generating minimal returns for shareholders, and lacking either the stability of profitability or the potential upside of turnaround situations. The model has learned they're stuck in limbo.

Firms with very negative ROE get slightly better predictions, likely because extreme distress can signal potential value opportunities or asset-rich companies with temporarily depressed earnings. These aren't great situations, but they're at least clearly defined.

As ROE climbs into positive territory, predictions improve steadily. Positive ROE firms generate returns on shareholder equity, making them fundamentally more attractive. The steep recovery reflects that profitability matters a lot once you cross the zero threshold.

**Implications:**
- Avoid companies clustered around zero ROE—the model sees them as the least attractive segment.
- Deeply negative ROE isn't an automatic disqualifier but requires careful analysis to understand if it's temporary distress or permanent value destruction.
- The model has captured a nonlinear relationship that goes beyond simple "higher = better," recognizing that context matters for how ROE translates to E/P.

### Partial Dependence — Debt-to-Assets

**Observations:**
- Generally **increasing relationship** between leverage and predicted E/P, but with distinct phases.
- Below debt-to-assets = -0.03: Flat predictions around -0.037.
- From -0.03 to 0: Gradual upward movement with step-like increases.
- Above 0: Steady climb, with predictions rising from -0.025 to +0.037 as leverage increases to 0.01.
- The relationship is steppy rather than smooth, showing the tree-based structure of the model.

**Interpretation:**
The positive relationship between leverage and E/P predictions is interesting. Higher debt-to-assets ratios boost predicted E/P, suggesting the model has learned that levered firms may be undervalued by the market. This could reflect several dynamics: debt can amplify returns when used productively, leverage signals management confidence, or highly leveraged firms trade at discounts that create higher E/P ratios.

The flat zone at very low/negative leverage represents firms with minimal or no debt—conservative capital structures that might signal either financial strength or missed opportunities to optimize returns through leverage.

The stepped pattern reveals how Gradient Boosting with shallow trees creates piecewise constant predictions—each plateau represents a different leaf node in the ensemble. As debt increases, the model progressively shifts to higher-prediction nodes.

**Implications:**
- Moderate leverage appears favorable in the model's view—it's associated with higher predicted E/P, possibly indicating value opportunities among levered firms.
- Very low leverage firms get the lowest predictions, suggesting they might be overpriced relative to earnings or simply conservative businesses that don't maximize returns.
- The relationship isn't extreme—the total effect spans about 0.07 units of E/P across the full range, much smaller than ROA's impact, consistent with debt-to-assets being the third most important feature.

  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/LEARNING CURVE (Training Evolution).png'| relative_url }}" alt="Feature vs Target" style="max-width: 70%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 18: Learning Curve </p>
  </div>

**Interpretation:**
- **Training MSE (blue)** drops sharply in the first ~50 iterations, then continues improving gradually throughout all 300 trees, ending around -0.08. The negative values are an artifact of how sklearn stores the loss—what matters is the trend.
- **Test MSE (orange)** also drops quickly initially, falling from ~0.055 to ~0.028 in the first 25-50 iterations, then plateaus almost completely for the remaining 250 trees.
- **Critical observation:** The training and test curves diverge. Training error keeps improving while test error flatlines, which is a classic sign of overfitting beginning to set in.
- However, the test MSE plateau is relatively stable—it's not increasing, just not improving. This suggests mild overfitting rather than severe overfitting where test error would climb.
- The rapid initial improvement shows the first few dozen trees are doing the heavy lifting, capturing the main patterns (likely the ROA/ROE relationships we saw earlier).
- After about iteration 50, additional trees make tiny refinements that help on training data but don't generalize to the test set.

**This means that:**
The model could probably perform just as well with around 50-100 trees instead of 300. The extra 200+ trees are memorizing training-specific noise rather than learning generalizable patterns. The learning rate of 0.1 combined with shallow trees (max_depth=3) prevents severe overfitting, but we're still past the point of diminishing returns.

**Implications:**
- For production use, we could retrain with fewer trees (maybe 75-100) to get similar performance with faster predictions and less memory.
- If we wanted to push performance further, we'd need to either add more informative features or use regularization techniques like early stopping based on validation performance.
- The stable plateau suggests our current setup is reasonably well-tuned—we're not drastically overfitting, but we're also not leaving obvious gains on the table.

  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/RESIDUAL DISTRIBUTION (Check for Bias).png'| relative_url }}" alt="Feature vs Target" style="max-width: 70%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 19: Residual Distribution </p>
  </div>

**Interpretation:**
- The distribution is roughly **bell-shaped and centered near zero**, with the mean residual at 0.0067—very close to unbiased.
- The peak sits right around zero, showing the model gets most predictions reasonably close to actual values.
- **Slight positive skew**: There's a longer tail extending to the right (positive residuals) compared to the left. This means the model has more cases where it underpredicts (actual > predicted) than where it overpredicts.
- The positive tail reaches out to around +0.75, while the negative tail only extends to about -0.50, confirming the asymmetry.
- Most residuals fall within a tight range of -0.25 to +0.25, with the bulk concentrated even tighter around -0.10 to +0.10.
- There are small but noticeable tails at the extremes, representing outlier predictions where the model is way off.

The near-zero mean (0.0067) confirms the model isn't systematically biased overall—it's not consistently too optimistic or pessimistic. However, the positive skew aligns with what we saw in earlier plots: the model tends to underpredict high E/P firms more often than it overpredicts low E/P firms. This is the regression-to-the-mean effect showing up in the residual distribution.

The general bell shape is good—it suggests errors are approximately normally distributed, which validates using MSE/RMSE as our loss function. If the distribution were heavily skewed or had weird modes, it would indicate the model is systematically struggling with certain types of cases.

**Implications:**
- The tight concentration around zero means the model works well for most firms, with errors typically under 0.15 in magnitude.
- The positive skew reinforces that users should be more cautious about predictions for high E/P firms—these are more likely to be underestimates.
- The presence of extreme outliers (residuals beyond ±0.5) suggests there's a small subset of firms the model fundamentally misunderstands, likely edge cases with unusual feature combinations.

  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/COMPARISON TRAIN VS TEST PERFORMANCE .png'| relative_url }}" alt="Feature vs Target" style="max-width: 70%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 20: Comparision Train v Test Performance </p>
  </div>


### Quantitative Metrics – Observations and Discussion

**Interpretation:**
- **MAE**: Training MAE (~0.077) is slightly better than test MAE (~0.087), showing a small gap. The model makes slightly larger errors on unseen data, but the difference is modest.
- **RMSE**: Similar pattern—training RMSE (~0.147) vs test RMSE (~0.153). The gap is minimal, indicating the model handles variance similarly across both sets.
- **R²**: Training R² (~0.55) vs test R² (~0.58). Interestingly, the test R² is actually *higher* than training, which is unusual but can happen when the test set happens to have slightly more predictable patterns or when there's randomness in how data splits.
- **Overall assessment**: The bars are remarkably close across all three metrics. There's no dramatic divergence that would signal severe overfitting.

**What this means:**
The model generalizes reasonably well. The sequential boosting process with shallow trees and MAE loss has learned patterns that transfer to new data rather than just memorizing training examples. The small performance gap (train slightly better on MAE/RMSE) is normal and expected—perfect parity would actually be suspicious.

The fact that test R² exceeds training R² slightly is a quirk of this particular train-test split. R² measures explained variance relative to the mean, so if the test set happens to have less noisy relationships or more consistent patterns, it can score higher. This isn't a major concern.

**Implications:**
- The model is production-ready from a generalization standpoint. It won't suddenly perform much worse on new company data.
- The training performance (~0.077 MAE, 0.55 R²) is a realistic estimate of what to expect in practice.
- Since we're not severely overfitting, adding more complexity (deeper trees, more features) might actually help capture additional signal rather than just memorizing noise.
- The similar performance across sets validates our hyperparameter choices—300 trees with max_depth=3 strikes a reasonable balance between fitting and generalizing.






### Random Forest Visualizations 

  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/ActualvPredRF.png'| relative_url }}" alt="Feature vs Target" style="max-width: 70%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 20: Comparision Train v Test Performance </p>
  </div>

**Interpretation:**
- Most points cluster around the 45° line, indicating solid predictive performance across the typical range of E/P values.
- The Random Forest replicates the typical **regression-to-the-mean** pattern:
  - High E/P cases (actual > 0.5) are underpredicted, with predictions compressed toward the center.
  - Low E/P cases (actual < -0.5) are slightly overpredicted, showing similar mean-reversion.
- This occurs because Random Forests average many trees, limiting extrapolation—especially for extreme financial ratios. The model is conservative by design, pulling extreme predictions toward the training mean.
- **Notable patterns:**
  - Strong vertical banding at specific E/P values (e.g., around 0.0, 0.25, 1.0) suggests the model encounters certain common E/P configurations repeatedly in the test data.
  - The scatter increases at the extremes, indicating higher prediction uncertainty for outlier firms with unusual financial characteristics.
  - Despite this limitation, the model matches patterns well for mid-range E/P values (-0.25 to 0.5), which represent the majority of firms and are most relevant for practical investment decisions.
- **Implication:** While the model provides reliable predictions for typical companies, users should exercise caution when applying predictions to firms with extreme E/P ratios, as these represent either highly distressed or exceptionally profitable outliers where the model's averaging behavior may miss important nuances.


  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/residualvPredRM.png'| relative_url }}" alt="Feature vs Target" style="max-width: 100%%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 21: Residual v Predicted </p>
  </div>

**Interpretation:**
- Residuals are centered around zero for most predicted values, showing the model is generally unbiased in the typical range.
- **The spread of residuals increases at extreme predictions:** Beyond predicted E/P of ±0.5, errors become larger and more variable, indicating higher uncertainty for companies with unusual financial characteristics.
- **Clear bias patterns at the extremes:**
  - For high predicted E/P (> 0.5), residuals are mostly negative—the model underpredicts highly profitable firms.
  - For low predicted E/P (< -0.5), residuals are mostly positive—the model overpredicts distressed companies.
  - The white gap near zero predicted E/P shows the model avoids predicting breakeven values, likely because these cases are unstable and rare in the training data.
- **Diagonal streaks in the residuals:** These bands occur because Random Forests average many trees, creating "preferred" prediction values that appear repeatedly. This is typical behavior for ensemble methods.
- **Overall:** The model works well for typical firms (predicted E/P between -0.25 and 0.5) with relatively small errors. For extreme cases, predictions should be interpreted cautiously since the model systematically pulls outliers toward the mean and has higher uncertainty at the tails.
- **Implication:** For practical use, the model is most reliable for identifying moderately undervalued or overvalued stocks, but should not be heavily relied upon for screening extreme value or distressed opportunities where prediction errors are largest.

### Error by Predicted Decile (Test)
This chart divides all test predictions into 10 equal groups (deciles) based on predicted E/P values, from lowest (decile 0) to highest (decile 9). Within each group, we calculate the Mean Absolute Error to see how accurate the model is across different ranges of predictions. High-level, this helps us understand if the model performs consistently across all types of firms or struggles with certain segments. Low-level, it reveals whether prediction quality varies systematically with the magnitude of predicted E/P—for example, if the model is less accurate for high-growth versus value stocks.
Understanding where the model makes larger errors is crucial for practical deployment. If errors are concentrated in specific deciles, we know which predictions to trust more or less. This is especially important in finance, where different deciles might represent distinct investment strategies (e.g., deep value vs. distressed vs. growth stocks).

  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/Error by Predicted Decile.png'| relative_url }}" alt="Feature vs Target" style="max-width: 100%%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 22: Residual v Predicted  </p>
  </div>


### Error by Predicted Decile (Test)

This chart divides all test predictions into 10 equal groups (deciles) based on predicted E/P values, from lowest (decile 0) to highest (decile 9). Within each group, we calculate the Mean Absolute Error to see how accurate the model is across different ranges of predictions. High-level, this helps us understand if the model performs consistently across all types of firms or struggles with certain segments. Low-level, it reveals whether prediction quality varies systematically with the magnitude of predicted E/P—for example, if the model is less accurate for high-growth versus value stocks.
Understanding where the model makes larger errors is crucial for practical deployment. If errors are concentrated in specific deciles, we know which predictions to trust more or less. This is especially important in finance, where different deciles might represent distinct investment strategies (e.g., deep value vs. distressed vs. growth stocks).

**Observations:**
- **U-shaped error pattern:** MAE is highest at the extremes (deciles 0 and 9) and lowest in the middle (deciles 3-4).
- Decile 0 (lowest predicted E/P) has MAE around 0.19, while decile 9 (highest predicted E/P) shows the worst performance with MAE exceeding 0.22.
- The model performs best in deciles 3-4 (MAE around 0.02-0.05), representing firms with slightly below-average to average predicted E/P.
- After decile 4, errors gradually increase as predicted E/P rises, jumping sharply in the top decile.

**Interpretation:**
- The U-shaped curve confirms what we saw in the residual plots: the Random Forest struggles most with extreme cases on both ends. This reflects the model's regression-to-the-mean behavior—it's most confident and accurate when predicting firms close to the training average.
- The extremely low error in deciles 3-4 suggests the model has learned strong patterns for "typical" companies, which are well-represented in the training data.
- The sharp spike in decile 9 indicates the model is particularly unreliable for predicting very high E/P stocks (deep value or highly profitable firms), likely because these cases are rarer and more heterogeneous in the training set.

**Implications:**
- **For portfolio construction:** Predictions in deciles 3-6 are most reliable and should carry more weight in investment decisions. Extreme deciles (0-2 and 8-9) require additional scrutiny or alternative validation methods.
- **Model refinement opportunity:** The high errors at extremes suggest potential improvements through targeted approaches—perhaps separate models for distressed versus high-performing firms, or additional features that better capture extreme financial conditions.
- **Risk management:** When using this model for screening, apply wider confidence intervals or require stronger supporting evidence before acting on predictions in the extreme deciles, especially decile 9.


  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/FetImp.png'| relative_url }}" alt="Feature vs Target" style="max-width: 100%%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 21: Feature Importance  </p>
  </div>

### Feature Importance

Feature importance tells us which variables the model relies on most heavily when making predictions. This is crucial for several reasons: (1) it validates whether the model is learning sensible financial relationships or just picking up noise, (2) it helps us understand what drives E/P predictions, and (3) it identifies which data inputs are most critical to collect and maintain accurately. In a Random Forest, importance is measured by how much each feature reduces prediction error (impurity) across all the trees—features that consistently improve splits are deemed more important.

**Observations:**
- **ROA (Return on Assets) dominates** with importance around 0.45, more than 4x higher than any other feature. This single metric accounts for nearly half of the model's predictive power.
- **ROE (Return on Equity)** is a distant second at approximately 0.10, followed by debt-to-assets and asset growth in the 0.08-0.09 range.
- Traditional financial metrics (profitability ratios, leverage, growth, size) cluster in the middle tier with importance scores between 0.03-0.08.
- **Corporate culture features** (innovation, integrity, quality, respect, teamwork) and analyst consensus (MEANREC) all show minimal importance, each contributing less than 0.03.

**Interpretation:**

**Why ROA dominates:** Return on Assets measures how efficiently a company converts its asset base into earnings—essentially, profit per dollar of assets employed. This makes perfect sense as the strongest predictor of E/P for several reasons:

1. **Direct mathematical relationship:** E/P (earnings-to-price) and ROA (earnings-to-assets) both have earnings in the numerator. Companies with high ROA naturally tend to have higher earnings relative to their valuation metrics.

2. **Core profitability signal:** ROA captures operational efficiency and profitability in a single metric. It reflects management's ability to generate returns regardless of capital structure, making it a fundamental indicator of business quality.

3. **Value investing principle:** High ROA companies are often undervalued (high E/P) when the market overlooks their earnings power, while low ROA companies may be overvalued (low E/P). The model has learned this core relationship from the data.

The strong showing of ROE (also an earnings-based ratio) reinforces that profitability metrics are central to understanding E/P. Meanwhile, debt-to-assets and asset growth provide important context about financial risk and momentum.

**The culture data disappointment:** The near-zero importance of all five culture dimensions suggests these variables add little predictive value beyond traditional financial metrics. This could mean: (1) culture's impact on firm value is already captured by financial performance measures like ROA and ROE, (2) the culture scores themselves may be noisy or poorly measured, or (3) culture affects long-term outcomes that aren't reflected in the 2-year test window we're predicting.

**Implications:**
- The model's heavy reliance on ROA validates the financial logic—it has learned a fundamental relationship that aligns with established finance theory.
- For practical deployment, ensuring accurate and timely ROA data is critical, as errors in this single feature would significantly degrade predictions.
- The minimal contribution of culture features suggests they could potentially be removed to simplify the model without sacrificing performance, though they should be retained if the goal includes understanding culture's incremental effect.
- Future model improvements should focus on enriching profitability-related features or finding better ways to measure intangible factors like culture, rather than adding more basic financial ratios.


  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/Partial Dependence Plots .png'| relative_url }}" alt="Feature vs Target" style="max-width: 100%%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 22: Partial Dependence Plots  </p>
  </div>

### Partial Dependence — ROA

**Observations:**
- Clear **S-shaped (sigmoid) relationship** between ROA and predicted E/P.
- Three distinct regions:
  - Below ROA = 0: Relatively flat and negative (around -0.25), representing unprofitable firms.
  - ROA = 0 to 0.010: Sharp upward transition, with predicted E/P jumping from -0.1 to +0.2.
  - Above ROA = 0.010: Levels off around +0.2, representing highly profitable firms.
- The steepest slope occurs right around the profitability threshold (ROA ≈ 0 to 0.005).

**Interpretation:**
The S-curve makes strong financial sense. Firms with negative or zero ROA are fundamentally unprofitable—they destroy value rather than create it, leading to low or negative E/P predictions. As ROA crosses into positive territory, the model rapidly increases E/P predictions, recognizing that profitable operations signal potential undervaluation. Beyond a certain point (ROA > 0.01), returns diminish—additional profitability improvements matter less, possibly because the market already prices in high-quality firms.

The critical transition zone around ROA = 0 reflects a fundamental regime change: the difference between a loss-making and profit-making business is more important than incremental profit improvements among already-profitable firms.

**Implications:**
- Companies near the breakeven point (ROA ≈ 0) are in the highest-leverage zone—small improvements in operational efficiency could dramatically shift their predicted valuation.
- For portfolio screening, filtering out firms with ROA < 0 would eliminate the consistently low-prediction zone, focusing on profitable opportunities.
- The flattening at high ROA suggests diminishing marginal returns to profitability in E/P prediction, consistent with efficient markets pricing in quality.



### Partial Dependence — ROE

**Observations:**
- **V-shaped relationship** with a sharp minimum at ROE ≈ 0.
- High predicted E/P (~0.23) for very negative ROE (< -0.005).
- Steep drop as ROE approaches zero, bottoming out at around +0.01 predicted E/P.
- Gradual recovery as ROE increases beyond zero, reaching about +0.18 for high positive ROE (> 0.010).
- Asymmetric pattern: negative ROE region is higher than the positive ROE recovery.

**Interpretation:**
This V-shape is counterintuitive at first but reveals important dynamics. Firms with very negative ROE (destroying shareholder equity) actually get higher E/P predictions—this likely reflects distressed value situations where earnings are temporarily depressed but the firm has substantial assets. The market may overprice these companies relative to their current earnings, creating high E/P ratios.

The minimum at ROE ≈ 0 represents the worst zone: companies barely breaking even that lack both the stability of profitability and the potential turnaround value of deeply distressed firms. These are "zombie" companies stuck in no-man's-land.

As ROE rises into positive territory, E/P predictions increase, reflecting healthy, equity-generating businesses that may be undervalued—classic value opportunities.

**Implications:**
- Be cautious with companies near zero ROE—the model sees them as the least attractive from an E/P perspective.
- Very negative ROE isn't necessarily a disqualifier; these might be distressed value plays, though they require additional due diligence.
- The model has learned a nonlinear, context-dependent relationship that goes beyond simple "higher profitability = better."



### Partial Dependence — Debt to Assets

**Observations:**
- Nearly flat relationship from debt-to-assets = -0.05 to -0.01, hovering around 0.08-0.09 predicted E/P.
- Moderate upward slope from -0.01 to 0, reaching about 0.11.
- **Sharp exponential increase** beyond debt-to-assets = 0, jumping from 0.12 to 0.24+ as leverage increases.
- The rug plot (tick marks at bottom) shows most data concentrates in the 0 to 0.01 range.

**Interpretation:**
The dramatic upward curve for highly levered firms (debt-to-assets > 0) seems counterintuitive—conventional wisdom suggests high debt increases risk and should lower valuation. However, this pattern likely reflects a selection effect and financial distress dynamics:

1. **Distressed firms with high debt** often have severely depressed stock prices (denominator of E/P), inflating the E/P ratio even if earnings are modest.
2. **Survivorship in the dataset:** Companies with extremely high leverage that are still reporting earnings may represent special situations—restructurings, turnarounds, or industries where high leverage is sustainable.
3. **Value trap signal:** The model may have learned that high debt-to-assets associates with cheap valuations (high E/P), but this doesn't necessarily mean these are good investments—they could be value traps.

The relatively flat region for negative to low positive debt suggests the model doesn't strongly differentiate between underleveraged and moderately leveraged firms.

**Implications:**
- High debt-to-assets firms may trigger high E/P predictions, but users should interpret these cautiously—they likely represent higher-risk, distressed situations rather than genuine value opportunities.
- The model's treatment of leverage is less intuitive than its handling of profitability metrics, suggesting this feature may be capturing complex, nonlinear effects or data artifacts.
- For practical screening, consider imposing maximum leverage thresholds to avoid these potentially misleading high-E/P predictions driven primarily by financial distress rather than fundamental value.

**Overall PDP Takeaway:**
The partial dependence plots confirm the model has learned economically sensible relationships for profitability metrics (ROA, ROE), though with interesting nonlinearities that reflect real-world complexity. The leverage relationship is less straightforward and warrants caution. These plots build confidence that the model isn't just a black box—it's capturing fundamental financial dynamics, even if some patterns require careful interpretation in practical application.


  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/StabovTime.png'| relative_url }}" alt="Feature vs Target" style="max-width: 100%%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 21: Feature Importance  </p>
  </div>

### Stability Over Time: MAE by Year (TEST only)

This plot tracks the model's Mean Absolute Error separately for each year in the test set (2020 and 2021). It reveals whether prediction accuracy degrades over time as the model moves further from its training period (which ended in 2019). Temporal stability is critical in finance because models trained on historical data must generalize to future periods with potentially different economic conditions, market regimes, and company behaviors.
Financial models often suffer from "model drift"—their performance deteriorates as the world changes. Interest rate shifts, regulatory changes, technological disruption, and macro shocks (like COVID-19, which overlaps with our test period) can break previously stable relationships. A model that works well in-sample but degrades rapidly out-of-sample is unreliable for real-world deployment. This analysis tests whether our Random Forest maintains consistent accuracy or if its predictions become less trustworthy over time.


**Observations:**
- MAE increases steadily from 2020 to 2021, rising from approximately 0.0925 to 0.0945.
- The degradation is modest—roughly a 2% relative increase over the two-year test window.
- The trend is linear and smooth, with no sudden jumps or erratic behavior.
- Both years remain within a fairly tight error band (0.092-0.095), suggesting the model hasn't completely broken down.

**Interpretation:**
The gradual upward trend indicates mild model drift, which is expected and relatively benign. Several factors could explain this pattern:

1. **Temporal distance from training data:** As we move from 2020 to 2021, the gap between training (pre-2020) and testing widens. Companies evolve, and the model's learned patterns may become slightly less applicable.

2. **COVID-19 impact:** The test period (2020-2021) captures the pandemic and recovery period, introducing unusual volatility and non-stationary dynamics. Traditional financial relationships may have temporarily shifted, making predictions harder.

3. **Changing economic regime:** Interest rates, fiscal policy, and market sentiment all shifted dramatically during this period, potentially altering how features like ROA and debt relate to E/P.

4. **Limited test window:** With only two years, we can't definitively conclude whether this is a persistent trend or just short-term noise.

The model appears reasonably robust to the test period's challenges.

**Implications:**
- **Deployment window:** The model should be reliable for 1-2 years after training before requiring retraining. The modest drift suggests it won't become obsolete immediately.
- **Monitoring protocol:** In production, track MAE monthly or quarterly. If degradation accelerates beyond this baseline trend, trigger model retraining with updated data.
- **Retraining schedule:** Plan to retrain annually or semi-annually to prevent accumulated drift from eroding performance. Incorporate new data to capture evolving market dynamics.
- **Contextual awareness:** The 2020-2021 period was unusually volatile. Future stability may differ depending on macro conditions—stable periods might show less drift, while crisis periods could show more.
- **Relative performance:** Even with drift, the MAE remains low in absolute terms (<0.10), suggesting the model is still useful despite modest degradation.

**Overall takeaway:** The Random Forest demonstrates acceptable temporal stability with only minor performance erosion over the test window. This is reassuring for practical deployment, though regular monitoring and periodic retraining will be essential to maintain accuracy as market conditions continue evolving.

  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/hetmap.png'| relative_url }}" alt="Feature vs Target" style="max-width: 100%%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 22: Feature Heatmap  </p>
  </div>

### Correlation Heatmap (Full Feature Set for RF)
A correlation matrix displays the pairwise linear relationships between all features in the model, including financial metrics, corporate culture scores, analyst consensus, and the target variable (E/P). Each cell shows the correlation coefficient between two variables, ranging from -1 (perfect negative correlation) to +1 (perfect positive correlation). Red indicates positive correlation, blue indicates negative correlation, and white represents no linear relationship.


**Key Observations:**

**Strong positive correlations (dark red):**
- **ROE and ROA** show moderate positive correlation (~0.5-0.6 range based on color intensity), which makes sense—both measure profitability, though ROE emphasizes equity efficiency while ROA focuses on total assets.
- **Culture dimensions cluster together:** Innovation, integrity, quality, respect, and teamwork all show moderate positive correlations with each other (~0.3-0.5), suggesting they tend to move together. Companies with strong culture in one dimension often score well in others.
- **Profitability metrics:** Gross profitability and operating margin show positive correlation, as expected—companies with high margins tend to be profitable across multiple measures.

**Notable negative correlations (blue):**
- **Debt-to-assets and ROA** show slight negative correlation (light blue), suggesting highly leveraged firms tend to have lower asset profitability—consistent with financial distress dynamics.
- **No strong negative correlations** are visible (no dark blue cells), indicating the absence of obvious inverse relationships in the data.

**Weak or near-zero correlations (white/light colors):**
- **Culture features and financial metrics** show mostly weak correlations. For example, innovation, integrity, quality, respect, and teamwork have minimal correlation with ROA, ROE, or debt ratios. This suggests culture scores add independent information beyond what's captured by traditional financials.
- **MEANREC (analyst consensus)** shows weak correlation with most features, including the target E/P. This is somewhat surprising—analyst recommendations appear to be only loosely connected to fundamental financial metrics in this dataset.
- **Sales and asset growth** show weak correlations with most other features, indicating growth dynamics are relatively independent of profitability and leverage levels.

**Correlations with the target (E/P - bottom row/right column):**
- **ROA and ROE** show the strongest positive correlations with E/P (moderate red), validating the feature importance findings. High profitability associates with higher earnings-to-price ratios.
- **Gross profitability** and **asset turnover** show positive correlations with E/P, though weaker than ROA/ROE.
- **Culture features** show very weak correlations with E/P (nearly white), explaining their minimal predictive power in the Random Forest.
- **Debt-to-assets** shows a positive correlation with E/P (light red), which seems counterintuitive but aligns with the partial dependence plot—highly leveraged firms often have depressed prices, inflating E/P ratios.

**Multicollinearity assessment:**
- No severe multicollinearity is evident—no cells show dark red (r > 0.8) except the diagonal (variables with themselves).
- The moderate correlation between ROA and ROE (~0.5-0.6) is acceptable and expected, as they measure related but distinct concepts.
- Culture dimensions' intercorrelation (~0.3-0.5) suggests some redundancy but not enough to cause major problems.

**Interpretation:**

**Why culture features don't predict well:**
The weak correlations between culture scores and E/P help explain their poor performance in the model. Culture's impact on firm value may be:
1. **Indirect and lagged:** Culture affects long-term outcomes through employee retention, innovation, and reputation, which may not manifest in short-term E/P ratios.
2. **Already captured:** Culture's effects might already be reflected in profitability metrics like ROA—good culture improves operations, which shows up in financial performance.
3. **Measurement issues:** The culture scores themselves may be noisy or not accurately capture the underlying construct.

**Financial intuition holds:**
The correlation patterns validate standard financial relationships:
- Profitability metrics (ROA, ROE, margins) positively correlate with value (E/P).
- Leverage shows complex, context-dependent relationships.
- Growth metrics operate relatively independently of current profitability.

**Model implications:**
- The lack of severe multicollinearity means the Random Forest can effectively use all features without numerical instability.
- The weak culture-financial correlations suggest culture adds independent (though not predictive) information—removing these features likely wouldn't hurt performance.
- The relatively low correlations between most features and E/P (no dark red except ROA/ROE) explain why even the best model struggles with R² in the 0.3-0.4 range—E/P is influenced by many factors beyond what's captured here.

**Implications:**
- **Feature engineering opportunity:** The weak culture correlations suggest these variables need transformation or interaction terms to unlock predictive value, or they should be dropped to simplify the model.
- **Diversification benefit:** Low correlations between feature groups (financial vs. culture vs. consensus) mean they theoretically offer complementary information, though in practice, only financial metrics demonstrate clear predictive power.
- **Model confidence:** The moderate correlations overall (mostly in the 0.2-0.6 range) confirm that E/P is a complex target influenced by multiple factors, justifying the ensemble approach of Random Forests that can capture nonlinear interactions.

**Overall takeaway:** The correlation matrix validates the model's feature importance rankings—profitability metrics with the strongest correlations to E/P also dominate the Random Forest's predictions. The weak performance of culture features is explained by their minimal correlation with the target. No severe multicollinearity issues exist, so all features can coexist in the model without causing instability.

  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/QuantMeticRM.png'| relative_url }}" alt="Feature vs Target" style="max-width: 100%%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 22: Feature Heatmap  </p>
  </div>

**Quantitative Performance Overview:**

The summary table and charts above provide a consolidated view of the Random Forest model's quantitative performance across multiple dimensions:

**Error Metrics:**
- **Training Performance:** The model achieves excellent in-sample fit with MAE of 0.0355 and RMSE of 0.0635, indicating predictions are typically within ±0.04 E/P units of actual values during training.
- **Test Performance:** On unseen data, MAE rises to 0.0935 and RMSE to 0.1471, showing expected degradation but maintaining reasonable accuracy for practical use.
- **Error Ratio:** The test-to-train MAE ratio of 2.63x indicates moderate overfitting—the model performs substantially better on training data but hasn't completely memorized it.

**Model Fit (R² Scores):**
- **Training R²: 0.9230** — The model explains 92.3% of variance in training E/P, demonstrating strong learning of underlying patterns.
- **Out-of-Bag (OOB) R²: 0.6495** — This internal cross-validation metric (based on bootstrap samples) provides an unbiased estimate of generalization, falling between train and test performance.
- **Test R²: 0.6177** — On held-out data, the model explains 61.8% of variance, which is solid for financial prediction but leaves substantial room for improvement or indicates irreducible noise in the target.
- **Overfitting Gap: 0.3053** — The 30.5 percentage point drop from train to test R² quantifies overfitting severity. While non-trivial, this gap is acceptable for ensemble methods on noisy financial data.

**Validation Consistency:**
The OOB R² (0.6495) closely matches test R² (0.6177), differing by only 3.2 percentage points. This alignment validates that:
1. The test set is representative of the training distribution
2. The model's generalization is stable and not overly sensitive to specific data splits
3. Internal cross-validation (OOB) provides reliable performance estimates, enabling confident model tuning without repeatedly touching the test set

**Model Configuration:**
- **600 trees** ensure stable predictions through extensive ensemble averaging
- **16 features** span financial metrics, culture scores, and analyst consensus
- **Training set: 6,477 firm-years** provides substantial data for learning complex patterns
- **Test set: 1,925 firm-years** offers adequate sample size for robust out-of-sample evaluation

**Interpretation:**

**Strengths:**
1. **Strong absolute accuracy:** MAE of 0.0935 on test data means predictions are typically within ±9.35% of actual E/P—sufficiently precise for portfolio screening and ranking companies.
2. **Meaningful explanatory power:** Test R² of 0.618 indicates the model captures genuine signal, not just noise. For comparison, many published finance models struggle to exceed R² of 0.40.
3. **Reliable validation:** The close agreement between OOB and test R² (within 0.03) inspires confidence that performance estimates are trustworthy and not artifacts of lucky data splits.
4. **Reasonable generalization:** While overfitting exists (train R² of 0.92 vs. test of 0.62), the model hasn't collapsed—it maintains substantial predictive power on new data.

**Limitations:**
1. **Moderate overfitting:** The 2.63x error ratio and 30.5-point R² gap show the model fits training data much better than test data. This is typical for Random Forests but suggests simpler models might generalize comparably with less complexity.
2. **Substantial unexplained variance:** Test R² of 0.618 means ~38% of E/P variation remains unexplained. This could reflect:
   - Missing features (e.g., market sentiment, macroeconomic conditions, industry dynamics)
   - Inherent randomness in stock pricing that no model can predict
   - Limitations of the features available (weak culture predictors, basic financials)
3. **Error magnitude at extremes:** While average MAE is 0.0935, we saw from the decile analysis that errors exceed 0.20 for extreme predictions, limiting reliability for tail cases.


### Linear Regression (Expanded Data Set) Visualizations

  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/ActalvPredLIn.png'| relative_url }}" alt="Feature vs Target" style="max-width:50%; width: 80%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 22: Predicted vs Actual</p>
  </div>
  
### Scatterplot of Predicted vs Actual E/P values (Test Set)
The plot compares the actual E/P values (x-axis) with the model's predicted E/P values (y-axis). The red dashed line represents the line of perfect prediction where y=x.

### Observations:
- Range and Distribution: The axes now range approximately from -1.0 to +1.0, reflecting the winsorization and scaling applied during preprocessing. This is a much healthier range than the previous -5.0 to 0.0.

- Visible Correlation: Unlike the previous "vertical cloud," this plot shows a clear positive correlation. The main cluster of data points (the dense blue area) roughly follows the upward trend of the red diagonal line, particularly in the range of 0.0 to 0.5.

- Residual Spread: While the points are tighter around the line than before, there is still significant spread. The model tends to predict values between 0.0 and 0.25 for a wide range of actual E/P values.

- Struggling w Negative Value: For companies with negative Actual E/P (left side of the plot), the model often predicts a value closer to zero (between -0.5 and 0.0), failing to capture the full magnitude of the negative earnings yield. There is a noticeable "shelf" or cutoff around Actual E/P = 0.0 where the prediction behavior changes.

### Possible Causes:
- Linearity Assumption: Although improved, the Linear Regression model assumes a straight-line relationship between all features and the target. Valuation metrics often have non-linear "cliffs" (e.g., companies with slightly negative earnings are treated very differently by the market than those with massively negative earnings).

- Target Variable Complexity: The E/P ratio behaves differently for profitable vs. unprofitable firms. The model might be finding an "average" logic that fits profitable firms well but fails for loss-making ones.

- Feature Completeness: While we added Culture and Analyst data, we may still be missing sector-specific features or macroeconomic indicators (interest rates) that drive valuation multiples.

### Interpretation/Implications:
- Significant Improvement: Compared to the baseline model, this version is actually learning. The alignment along the diagonal shows that the new features (Culture, Analyst Ratings, etc.) combined with better preprocessing have given the model real predictive signal.

- Bias towards Profitability: The model is much better at predicting E/P for profitable companies (Actual E/P > 0). The "shelf" structure suggests it treats unprofitable companies as a distinct, harder-to-predict group, defaulting them to a near-zero prediction to minimize squared error.

- Next Steps: The remaining variance suggests we are reaching the limit of what a Linear Model can do. A tree-based model (like Random Forest or XGBoost) would likely handle the non-linear relationships (profitable vs. unprofitable) much better than a single straight line.



  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/ResidvPredlin.png'| relative_url }}" alt="Feature vs Target" style="max-width: 100%%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 23: Residual v Predicted EP </p>
  </div>

### Residuals vs Predicted E/P
This plot visualizes the relationship between the model's predicted E/P values and the residuals (Actual - Predicted). Ideally, points should be randomly scattered around the horizontal red line at zero.

### Observations:
- Improved Central Clustering: Compared to the old plot (which had a massive triangular spread), the residuals are now much tighter around the zero line for the majority of the data. This indicates that for "average" companies, the model's predictions are significantly more accurate.

- Visible Heteroscedasticity: The "fanning out" pattern is still present. As predicted E/P increases (moves right), the spread of residuals widens. The model is less confident and makes larger errors when predicting high E/P ratios (undervalued or high-earning firms).

- Diagonal Linear Artifacts: You can clearly see diagonal lines of points in the plot (e.g., running from top-left to bottom-right). This is a classic sign of a floor or ceiling effect in the data.

- It likely corresponds to the winsorization limits we applied during preprocessing. The model predicts a range of values, but the actual target variable was clipped at the 1st and 99th percentiles, creating these artificial boundaries.

#### Bias Shifts:

- For negative predictions, residuals tend to be positive (under-prediction).

- For high positive predictions, residuals tend to be negative (over-prediction).

- This suggests the model is slightly conservative, pulling extreme predictions towards the mean.

### Interpretation/Implications:
- Effective Outlier Control: The fact that the residuals are contained within a smaller range (-1.0 to +1.0) compared to the old plot (-5 to +3) proves that removing penny stocks and winsorizing the target was highly effective.

- Model Limitations: The distinct patterns (diagonals and fanning) confirm that a Linear Regression model cannot fully capture the complex, non-linear interactions in financial data. The model is "forcing" a linear fit onto a dataset that likely has regimes (e.g., distress vs. growth).

- Next Steps: The structured nature of the errors suggests that a non-linear model (like Gradient Boosting) could exploit these patterns to further reduce error.


  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/COEFLIN.png'| relative_url }}" alt="Feature vs Target" style="max-width: 100%%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 24: Residual Histogram </p>
  </div>

### Residual Distribution Histogram
The histogram visualizes the frequency distribution of the prediction errors (residuals). The ideal outcome is a normal distribution (bell curve) centered exactly at zero.

### Observations:
- Approaching Normality: The new distribution looks significantly more "bell-shaped" than the previous one. The extreme spike at zero is reduced, and the spread is more natural, resembling a Gaussian distribution much more closely than the previous "needle" plot.

- Centered at Zero: The peak of the distribution aligns almost perfectly with the red dashed line at 0. This indicates that, on average, the model is unbiased—it's not systematically over- or under-predicting the entire dataset.

- Slight Skewness: There is still a noticeable left tail (negative residuals) and a slightly heavy right tail.

- Left Tail: Represents companies where Actual E/P < Predicted E/P (the model was too optimistic about valuation).

- Right Tail: Represents companies where Actual E/P > Predicted E/P (the model was too pessimistic, or the company is undervalued).

- Reduction in Kurtosis: The previous plot was extremely leptokurtic (pointy). This one has broader shoulders, meaning the model is capturing more of the natural variance in valuation rather than just guessing the mean.

### Interpretation/Implications:
- Successful Cleaning: The improved shape confirms that winsorizing the target variable and removing penny stocks removed the "noise" that was distorting the error distribution.

- Better Generalization: Because the errors look more normal, statistical inferences (like confidence intervals) will be more valid with this model than the previous one.

- Remaining Outliers: The small bumps in the tails suggest there are still sub-groups of companies (likely distressed firms or hyper-growth startups) that defy the standard linear relationship.

  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/Indv V EP.png'| relative_url }}" alt="Feature vs Target" style="max-width: 100%%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 25: Individual v EP </p>
  </div>

### Individual Features vs E/P Ratio
This grid of scatterplots visualizes the relationship between each individual feature (x-axis) and the target variable, E/P Ratio (y-axis). The red dashed line represents a simple linear trendline for that specific feature.

### Observations:
- Strong Positive Correlations:

-- roe and roa: These plots show the clearest positive slopes. As Return on Equity and Return on Assets increase, the E/P ratio generally increases. This aligns with the coefficient chart. However, there's a distinct "Z-shape" or step-function behavior, suggesting that while the overall trend is linear, there are distinct regimes for low vs. high profitability firms.

-- log_assets: Shows a consistent, moderate positive trend. Larger firms tend to have slightly higher earnings yields (cheaper valuations), likely due to lower growth expectations compared to small caps.

- Weak or No Correlations:

-- asset_growth, sales_growth, asset_turnover: The trendlines for these are nearly flat. The data clouds are diffuse, indicating these metrics alone have little linear predictive power for E/P in this model.

- Culture Scores (innovation, integrity, etc.): These plots show very dense, vertical clouds of points because the culture scores are clustered around discrete values (likely integer-based or limited range in the source data). The trendlines are almost flat, though innovation shows a very slight negative tilt, matching its coefficient.

- Outliers and artifacts:

-- Vertical Lines: Many plots (like operating_margin, gross_profitability) show vertical lines of data points. This is likely due to winsorization (clipping) or imputation (filling missing values with the median), creating "walls" of data at specific values.

- Horizontal Banding: There's a dense horizontal band around E/P = 0. This reflects the fact that many companies have near-zero earnings, and the model (and reality) struggles to assign a precise multiple to them.

### Interpretation/Implications:
- Non-Linearity Confirmed: The "Z-shape" in roe/roa and the distinct clusters in other plots confirm that a simple straight line is an oversimplification. The relationship between fundamentals and value changes depending on whether a company is profitable or not.

- Data Quality Issues: The vertical artifacts suggest that our preprocessing (imputation/winsorization) might be slightly aggressive or that the underlying data has structural gaps. However, this is common in financial datasets.

- Multivariate Necessity: Since most individual features show weak correlations (flat lines), the model relies on the combination of these weak signals to make predictions. No single "magic bullet" feature predicts valuation on its own.


  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/LRVIS.png'| relative_url }}" alt="Feature vs Target" style="max-width: 100%%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 25: Individual v EP </p>
  </div>


### Feature Correlation Matrix
The correlation heatmap visualizes the pairwise linear relationships among the five variables used in the model: ROE, ROA, Debt-to-Equity Ratio, Sales Growth, and the E/P ratio. Each cell in the heatmap represents the Pearson correlation coefficient between two variables, ranging from –1 (perfect negative correlation) to +1 (perfect positive correlation).

### Observations and Implications:
The heatmap reveals that while profitability variables (ROE, ROA) are meaningfully related to valuation (E/P), leverage and growth measures are weakly linked. Furthermore, the modest correlations suggest that the true relationship between financial fundamentals and valuation may be nonlinear or interaction-based, which a simple linear model cannot fully capture.

### Next steps are similar to previous ones:
- Feature engineering: Construct new features that better capture firm performance and valuation drivers
- Nonlinear Modeling Approaches



  <div style="text-align: center; margin-bottom: 4rem;">
    <img src="{{ '/assets/images/PerfMetric.png'| relative_url }}" alt="Feature vs Target" style="max-width: 100%%; width: 100%;">
    <p style="font-style: italic; color: #666; margin-top: 1rem;">Figure 25: Individual v EP </p>
  </div>


### Quantitative Metrics – Observations and Discussion



To evaluate the performance of the linear regression model predicting E/P ratios, three standard metrics were computed: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the Coefficient of Determination (R^2).

### 1. Mean Absolute Error ($MAE$)


MAE measures the average magnitude of prediction errors, ignoring their direction.

Ideal value: Close to 0.

Observation: The model achieved an MAE of 0.1092 on the training set and 0.1150 on the test set. This is a substantial improvement over the baseline model which had errors > 0.30. An MAE ~ 0.11 implies that, on average, the model's prediction of the Earnings Yield deviates from the actual value by about 11%. The very small gap between training and testing errors indicates the model is robust and not overfitting.


### 2. Root Mean Squared Error ($RMSE$)

RMSE measures the square root of the average squared prediction errors, penalizing larger errors more heavily.

Ideal value: Close to 0.

Observation: RMSE dropped significantly to ~ 0.19 (down from >0.80 in the previous iteration). While much improved, the RMSE is still nearly double the MAE (0.19 vs 0.11). This gap suggests that while the target winsorization removed the most extreme anomalies (like E/P = 7000), there are still "moderate outliers"—companies where the model misses by a larger margin than usual. Interestingly, the Test RMSE (0.1884) is slightly lower than the Train RMSE, suggesting the model generalizes remarkably well to the 2020-2021 period.


### 3. Coefficient of Determination (R^2)

R^2 quantifies the proportion of variance in the target variable explained by the model.

Ideal value: 1.0(perfect fit).

Observation: The model now explains 34.81.3% (train) and 39.43.9% (test) of the variance in valuation. This is a major jump from the previous ~26%. In the context of financial data—which is inherently noisy and driven by unmeasurable market sentiment—an R^2 > 0.40 for a linear model is a strong result. It confirms that the new features (Corporate Culture and Analyst Sentiment) added genuine predictive signal that financial ratios alone could not provide.

### Interpretation/Implications:

- Validation of Data Cleaning: The massive reduction in RMSE and MAE confirms that the preprocessing steps (removing penny stocks, winsorizing the target) successfully stabilized the model, allowing it to learn actual patterns rather than chasing noise.

- Signal vs. Noise: With an R^2 approaching 0.42, the model has graduated from "guessing the average" to actually differentiating between companies based on their quality (Culture) and market perception (Analyst Ratings).

- Linear Ceiling: Despite these improvements, nearly 60% of the variance remains unexplained. This strongly suggests that the remaining relationships are non-linear (e.g., the market pays a premium for very high innovation but ignores average innovation). This sets the perfect stage for deploying a Gradient Boosting model to capture those complex, non-linear dynamics.






<span style="color:#7F9161; font-weight:bold; font-size:25px;">Final- Analysis/Comparison For 3 Models</span>


This project evaluated three machine learning models for predicting company earnings-to-price (E/P) ratios using financial fundamentals, corporate culture metrics, and analyst consensus data. We compared Linear Regression, Random Forest, and Gradient Boosting models across 50,731 training observations (2009-2019) and 10,866 test observations (2020-2021).

### Model Performance Summary


#### 1. Linear Regression: The Baseline

**Leak-Free Performance Metrics:**
| Metric | Train | Test |
|--------|-------|------|
| MAE | ~0.109 | ~0.113 |
| RMSE | ~0.181 | ~0.187 |
| R² | ~0.35 | ~0.38 |

**Strengths:**
- **Full Interpretability**: Provides clear, quantifiable relationships between features and target variable through coefficient weights
- **Computational Efficiency**: Trains in milliseconds, making it ideal for rapid prototyping and large-scale deployment
- **Robust Generalization**: Test R² (~0.38) slightly exceeds training R² (~0.35), indicating no overfitting and stable performance on unseen data
- **Consistent Errors**: Minimal gap between train MAE (~0.109) and test MAE (~0.113) demonstrates reliable predictions

**Limitations:**
- **Linear Assumption Barrier**: Explains only ~35-38% of variance in E/P ratios, indicating the true relationships are non-linear
- **No Feature Interactions**: Cannot capture complex interactions between profitability metrics, culture scores, and analyst sentiment
- **Higher Prediction Error**: Average prediction deviation of ~11% from actual E/P values is the highest among the three models

**Why This Performance?**
Linear regression struggles because financial valuation is inherently non-linear. For example, the market rewards high ROE non-linearly—a company with 30% ROE may trade at 3× the multiple of a 15% ROE company, not 2×. Similarly, the impact of corporate culture on valuation likely has threshold effects that linear models cannot capture.

---

#### 2. Random Forest: The High Capacity Learner

**Leak-Free Performance Metrics:**
| Metric | Train | Test |
|--------|-------|------|
| MAE | 0.0355 | 0.0935 |
| RMSE | 0.0635 | 0.1471 |
| R² | 0.9230 | 0.6177 |
| OOB R² | 0.6495 | — |

**Strengths:**
- **Powerful Pattern Recognition**: Achieved exceptional training performance (R² = 0.9230, MAE = 0.0355)
- **Automatic Feature Interactions**: Trees naturally discover complex relationships between variables without manual feature engineering
- **Non-Linear Modeling**: Captures threshold effects and conditional relationships that linear models miss
- **Strong Test Performance**: Test R² of 0.6177 substantially exceeds Linear Regression (~0.38), demonstrating genuine predictive power

**Limitations:**
- **Overfitting Present**: 33% drop in R² from training (0.9230) to test (0.6177) indicates some memorization of training patterns
- **Error Inflation**: Test MAE (0.0935) is 2.63× higher than training MAE (0.0355)
- **Black Box Nature**: While feature importance can be extracted, individual prediction explanations are opaque

**Performance Insights:**
Despite showing overfitting (92.3% → 61.8% R²), the Random Forest still delivers a **strong test R² of 0.6177**—the highest among all models. The model's OOB R² score of 0.6495 closely matches the test R² of 0.6177, validating that the out-of-bag estimates provide a reliable generalization measure and confirming the leak-free preprocessing is working correctly.

**Why This Performance?**
Random Forests excel at capturing non-linear relationships that Linear Regression missed, such as:
- The interaction between high innovation scores and profitability
- Threshold effects in debt ratios (moderate debt is tolerated, but high debt severely impacts valuation)
- Conditional analyst behavior (positive recommendations matter more for high-growth companies)

---

#### 3. Gradient Boosting: The Balanced Metrics

**Leak-Free Performance Metrics:**
| Metric | Train | Test |
|--------|-------|------|
| MAE | 0.0795 | 0.0862 |
| RMSE | 0.1534 | 0.1544 |
| R² | 0.5516 | 0.5788 |

**Strengths:**
- **Optimal Generalization**: Near-identical performance on train (R² = 0.5516) and test (R² = 0.5788) with minimal overfitting
- **Best Error Stability**: Test MAE (0.0862) is only 8.4% higher than train MAE (0.0795)—the smallest gap among all models
- **Lowest Test MAE**: Achieves the best prediction accuracy with MAE = 0.0862
- **Strong Predictive Power**: Explains 57.88% of variance while maintaining robustness
- **Sequential Learning**: Iteratively corrects residual errors, focusing on hard-to-predict cases
- **Regularization Built-In**: Shallow trees (max_depth=3) and learning rate (0.1) prevent overfitting while capturing non-linearity

**Limitations:**
- **Training Time**: Sequential nature requires longer training compared to parallel Random Forest
- **Hyperparameter Sensitivity**: Performance depends on careful tuning of learning rate, tree depth, and number of estimators

**Performance Insights:**
Gradient Boosting achieves the **best balance** between predictive accuracy and generalization. The test R² actually exceeds training R² (0.5788 vs 0.5516), indicating the model learned genuine signal rather than noise. The near-identical RMSE on train (0.1534) and test (0.1544) further confirms excellent generalization.

The model's MAE of 0.0862 translates to an average prediction error of ~8.6% in E/P ratios, which is highly competitive for financial prediction tasks. For context, professional analysts' earnings estimates typically have 10-15% error rates.

**Why This Performance?**
The configuration—shallow trees (max_depth=3), moderate learning rate (0.1), and 300 iterations with absolute_error loss—creates an ideal learning dynamic:
1. **Shallow Trees**: Prevent any single tree from memorizing complex patterns
2. **Sequential Correction**: Each new tree focuses on residual errors of the previous ensemble
3. **MAE Loss Function**: Robust to outliers in E/P distribution
4. **Gradual Complexity Building**: Starts with simple patterns and adds complexity only where needed

---

### Comparative Analysis

#### 1. **Random Forest and Gradient Boosting: Both Strong Performers**

The leak-free data preprocessing reveals that **both Random Forest and Gradient Boosting are strong, competitive models**:

| Metric | Random Forest | Gradient Boosting | Winner |
|--------|---------------|-------------------|--------|
| Test MAE | 0.0935 | 0.0862 | GB (by 7.8%) |
| Test R²  | 0.6177 | 0.5788 | RF (by 6.7%) |
| Train-Test Gap (R²) | 30.5% drop | 2.7% *improvement* | GB |
| Train-Test Gap (MAE) | 2.63× increase | 1.08× increase | GB |

**Key Insight**: Random Forest achieves higher R² on test data (explaining more variance), while Gradient Boosting achieves lower MAE (smaller average errors) and much better generalization stability. Both models substantially outperform Linear Regression, confirming that non-linear approaches are essential for financial valuation prediction.

#### 2. **Non-Linearity Is Critical**

The ~20-percentage-point jump in test R² from Linear Regression (~0.38) to the tree-based models (~0.58-0.62) confirms that valuation relationships are fundamentally non-linear. The market doesn't price companies proportionally to their fundamentals; instead, there are thresholds, interactions, and conditional effects.


#### 3. **Leak-Free Validation Confirms Robustness**

With proper data leakage prevention (training statistics computed only on training data), all models maintain their relative performance rankings. This confirms that:
- The preprocessing pipeline is sound and realistic
- Model performance estimates are trustworthy for real-world deployment
- Both RF and GB would perform reliably on future unseen data

#### 4. **Feature Importance Insights**

Based on Gradient Boosting's feature importance:
- **Profitability Metrics Dominate**: ROE and ROA account for the majority of predictive power
- **Analyst Sentiment Matters**: MEANREC contributes 8-10% of importance, validating its inclusion
- **Culture Metrics Are Weak**: Innovation, integrity, quality, respect, teamwork contribute minimally

**Interpretation**: Markets primarily value quantitative financial performance (profitability, growth) over qualitative cultural attributes. This aligns with efficient market hypothesis—hard numbers matter more than soft perceptions.

---

### Key Findings:

1. **Leak-Free Data Protection**: Rigorous temporal split with training-only statistics ensures realistic generalization estimates
2. **Both Tree Models Excel**: Random Forest (R² = 0.62) and Gradient Boosting (R² = 0.58) both substantially outperform Linear Regression
3. **Gradient Boosting Best for Stability**: Minimal train-test gap makes GB the safest choice for deployment
4. **Random Forest Best for Variance Explained**: Higher R² captures more of the underlying valuation dynamics
5. **Non-Linear Relationships Confirmed**: ~20% R² improvement over linear baseline validates tree-based approaches


<span style="color:#7F9161; font-weight:bold; font-size:25px;">Final- Next Steps/Conclusion</span>

### Next Steps 

Future research should prioritize three areas to enhance predictive performance and enable production deployment. Immediate optimizations through hyperparameter tuning, which includes Random Forest regularization (max_depth constraints, minimum leaf size increases) and Gradient Boosting learning rate schedules are expected to reduce test MAE by 10-20%. Feature engineering represents a high impact opportunity, particularly through interaction terms (ROE × Innovation, Debt_to_Assets × Asset_Growth) that capture conditional valuation dynamics, temporal features (rolling volatility, momentum indicators), and macroeconomic context variables (interest rates, market volatility) that would enable regime-aware predictions. Alternative ensemble methods (XGBoost, LightGBM) and heterogeneous stacking strategies combining Linear Regression's stability with Gradient Boosting's non-linear capacity offer additional performance potential. Long-term directions include causal inference frameworks to distinguish correlation from causation, multi-task learning jointly predicting valuations and returns, and reformulation to predict valuation changes rather than absolute levels. Production deployment requires automated retraining pipelines, drift monitoring dashboards, and A/B testing frameworks. With systematic execution, achieving test MAE below 0.070 within six months is feasible, positioning the model for reliable deployment in investment decision support systems.


### Conclusion

This project demonstrates that machine learning can effectively predict company valuations from financial fundamentals, corporate culture, and analyst sentiment with proper data leakage prevention ensuring realistic performance estimates.


Both **Random Forest** and **Gradient Boosting** emerge as strong, competitive models for E/P ratio prediction:

- **Random Forest** achieves the highest test R² (0.6177), explaining 62% of variance in company valuations—a **63% improvement** over Linear Regression (R² = 0.38). Despite some overfitting (train R² = 0.923 vs test R² = 0.618), the model still delivers the best variance explanation and demonstrates that its learned patterns generalize meaningfully to unseen data.

- **Gradient Boosting** achieves the lowest test MAE (0.0862) and the best generalization stability, with test R² (0.5788) actually exceeding train R² (0.5516). This **53% improvement** over Linear Regression comes with virtually no overfitting.

**Key Findings:**

1. **Leak-free preprocessing is critical**: By computing winsorization bounds, means, and standard deviations exclusively from training data (2009-2019), we ensure that test performance (2020-2021) reflects true out-of-sample generalization, not artificially inflated metrics from data leakage.

2. **Non-linear models are essential**: The ~20-percentage-point R² improvement from Linear Regression (~0.38) to tree based models (~0.58-0.62) confirms that financial valuation relationships are fundamentally non-linear, with threshold effects and feature interactions that linear models cannot capture.

3. **Both RF and GB excel in different ways**: Random Forest maximizes variance explained (R² = 0.62), while Gradient Boosting minimizes prediction error (MAE = 0.086) with superior stability. The choice between them depends on the deployment context, RF for maximum insight, GB for maximum reliability.

4. **Profitability dominates valuation**: ROE and ROA account for the majority of predictive power across all models, confirming that markets primarily value quantitative financial performance over qualitative cultural attributes.

5. **Temporal robustness validated**: All models generalize well to the 2020-2021 test period despite COVID-19 market disruptions, confirming that the learned patterns capture genuine valuation dynamics rather than period-specific noise.

The results confirm that while traditional finance theory provides intuition, machine learning uncovers the non-linear, conditional relationships that drive real-world valuations. With proper data leakage prevention and rigorous temporal validation, these models provide trustworthy performance estimates suitable for real-world deployment. Future refinement through hyperparameter tuning, feature engineering (interaction terms, macroeconomic variables), and advanced ensemble methods (XGBoost, LightGBM, stacking) could further improve performance, positioning these models as production-grade tools for investment analysis, portfolio construction, and risk management.



## References {#references}

[1]Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The Review of Financial Studies*, *33*(5), 2223–2273. https://doi.org/10.1093/rfs/hhaa020

[2]Ballings, M., Van den Poel, D., Hespeels, N., & Gryp, R. (2015). Evaluating multiple classifiers for stock price direction prediction. *Expert Systems with Applications*, *42*(20), 7046–7056. https://doi.org/10.1016/j.eswa.2015.05.013

[3]CFA Institute. (2025). Creating value from big data in the investment management process: A workflow analysis. *CFA Institute Research Foundation*. https://doi.org/10.56227/25.1.7

[4]Nugent, C. (n.d.). S&P 500 stock data. *Kaggle*. https://www.kaggle.com/datasets/camnugent/sandp500

[5]Crow, J. (n.d.). Stock market dataset. *Kaggle*. https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset

[6]Aroussi, R. (2023). *yfinance: Yahoo! Finance market data downloader* [Python package]. PyPI. https://pypi.org/project/yfinance/

[7]Wharton Research Data Services. (2024). *Wharton Research Data Services*. University of Pennsylvania. https://wrds-www.wharton.upenn.edu/

[8]Aaron7sun. (2016). Daily news for stock market prediction. *Kaggle*. https://www.kaggle.com/datasets/aaron7sun/stocknews

[9]Pham, L. (2017). Financial news headlines data. *Kaggle*. https://www.kaggle.com/datasets/notlucasp/financial-news-headlines


[10]Patel, J., Shah, S., Thakkar, P., & Kotecha, K. (2015). Predicting stock market index using fusion of machine learning techniques. *Expert Systems with Applications*, *42*(4), 2162–2172. https://doi.org/10.1016/j.eswa.2014.10.031

[11]Gradient Boosting Reinforcement Learning. (2016). *arXiv.org*. https://arxiv.org/html/2407.08250v1 



