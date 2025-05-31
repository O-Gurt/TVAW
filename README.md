# Triple Vovk-Azuri-Warmuth Algorithm as an Automated Machine Learning Approach

The nummerical results from the paper "Triple Vovk-Azuri-Warmuth Algorithm as an Automated Machine Learning Approach" (Использование трехуровневого алгоритма Вовка-Азури-Вармута в качестве подхода к автоматизированному машинному обучению) by  Olga V. Gurtovaya.

## Authors

Olga V. Gurtovaya  
Institute of Mathematics, Mechanics, and Computer Sciences of the Southern Federal University  
`imedashvili@sfedu.ru`

## Abstract

This paper presents the **Triple Vovk-Azoury-Warmuth (TVAW) algorithm**, designed for online regression tasks. TVAW advances the **Double Vovk-Azoury-Warmuth (DVAW)** approach [1] by combining **multi-kernel learning** with **data scaling strategies**. The primary goal of this work is to **comparatively analyze TVAW's performance** against the automated machine learning system **AutoGluon-Tabular** on 13 diverse tabular datasets. Experiments demonstrate that TVAW can effectively adapt to data heterogeneity and delivers competitive results compared to state-of-the-art AutoML solutions.

## Datasets to Download

Some datasets are loaded directly via Python libraries, while others need to be downloaded manually and placed in the appropriate directory (e.g., in the same directory as your script).

Here's a breakdown of the datasets and their typical sources:

* **Requires Manual Download:**
    * **Airfoil:** `airfoil_self_noise.dat` (from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise))
    * **Concrete:** `Concrete_Data.xls` (from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength))
    * **Bias:** `Bias_correction_ucl.csv` (from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bias+correction+of+numerical+prediction+model+temperature+forecast))
    * **Naval:** `NavalData.txt` (from [UCI Machine Learning Repository]([https://archive.ics.uci.edu/ml/index.php](https://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants)))
    * **Mercedes-Benz Greener Manufacturing:** `train.csv` (from [Kaggle competition](https://www.kaggle.com/competitions/mercedes-benz-greener-manufacturing/data))
    * **House-prices-advanced-regression-techniques:** `train.csv` (from [Kaggle competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data))
    * **Behavior of the urban traffic of the city of Sao Paulo in Brazil:** `Behavior of the urban traffic of the city of Sao Paulo in Brazil.csv` (from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/483/behavior+of+the+urban+traffic+of+the+city+of+sao+paulo+in+brazil))

* **Loaded via Python Libraries (no manual download needed):**
    * **Diamonds:** Loaded using `seaborn.load_dataset('diamonds')`
    * **Tips:** Loaded using `seaborn.load_dataset('tips')`
    * **Boston:** Loaded using `statsmodels.datasets.data('Boston')`
    * **California Housing:** Loaded using `sklearn.datasets.fetch_california_housing`
    * **Mtcars:** Loaded using `pydataset.data('mtcars')`
    * **Energy Efficiency:** Loaded using `sklearn.datasets.fetch_openml(name='energy_efficiency')`

## Usage

To run the models and reproduce the results, please follow these steps:

1.  **Install Required Libraries:** Ensure all necessary Python packages are installed by running:
    ```bash
    pip install numpy pandas scikit-learn seaborn autogluon openml pydataset statsmodels
    ```
2.  **Download Datasets:** Download all required datasets (as listed in the "Datasets to Download" section) and place them in the same directory as your `DVAW_TVAW_tests.ipynb` file.
3.  **Select Dataset:** Open the `DVAW_TVAW_tests.ipynb` Jupyter Notebook. In the "Preprocessing of all used datasets in the tests" section, uncomment the code block corresponding to the dataset you wish to use for your tests. Ensure only one dataset block is active at a time.
4.  **Run Notebook:** Execute all cells in the `DVAW_TVAW_tests.ipynb` notebook.


## References

[1] D.B. Rokhlin D. B.,  O.V. Gurtovaya (2023). Random feature-based double Vovk-Azoury-Warmuth algorithm for online multi-kernel learning. *arXiv preprint arXiv:2303.20087*. Available at: [https://arxiv.org/abs/2303.20087](https://arxiv.org/abs/2303.20087)
