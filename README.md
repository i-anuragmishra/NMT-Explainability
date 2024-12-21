# Advancing Explainability in Neural Machine Translation

**Name:** Anurag Mishra  
**Course Name:** NLP-1  

---

## Overview

This project focuses on developing an explainable Neural Machine Translation (NMT) model that evaluates attention patterns and correlates them with translation quality metrics. The project leverages large-scale pre-trained models, statistical alignments, and attention metrics for deeper insights into translation mechanisms.

Due to the model's size and training requirements, this project must be executed on **Research Computing (RC)** services and **not on Narnia**. Additionally, the final model folder is not included in the submission due to file size limitations. Instead, a pre-trained model can be downloaded using the Google Drive link provided below.

---

## Files Provided

1. **`train.py`**  
   This script trains the model from scratch.  
   - **Note:** Training takes over a month (30+ days) on large datasets and is not intended to be executed as part of this submission.  
   - **Input:** Downloads and uses the required dataset automatically.  
   - **Output:** Trained model checkpoints (not included in the submission).  

2. **`test.py`**  
   This script tests the trained model on a large dataset.  
   - **Note:** It evaluates the model’s performance but does not generate visualizations and may take significant time to complete.  
   - **Input:** Downloads the dataset and requires the trained model file.  
   - **Output:** Evaluation metrics such as BLEU and METEOR scores.  

3. **`demo.py`**  
   This script tests the pre-trained model on a smaller dataset and provides visualizations.  
   - **Purpose:** Quickly demonstrates the model’s capabilities, including BLEU and METEOR scores, along with detailed visualizations.  
   - **Input:** Pre-trained model downloaded via the Google Drive link and dataset (downloaded automatically).  
   - **Output:** Visualizations and performance metrics.  
   - **Recommended:** This is the file to run for most users.  

4. **`requirements.txt`**  
   Contains the list of dependencies required to set up the Python environment.

5. **`fastalign/`**  
   A folder containing the `fastalign` alignment tool necessary for all scripts. Ensure this folder is present in the project directory when running any script.

---

## Model and Dataset Information

### Pre-trained Model
- **Download Link:** [https://drive.google.com/file/d/19gXkvi4IP68M_d8jTFgEW8fHLsK_jr2w/view?usp=sharing]  
- **Command to Download:**  
   If you have `gdown` installed (recommended):  
   ```bash
   pip install gdown
   gdown "https://drive.google.com/uc?id=19gXkvi4IP68M_d8jTFgEW8fHLsK_jr2w" -O model.zip
   unzip model.zip

### Dataset
- **Note:** The dataset is **not included** in the submission. Instead, the scripts automatically download the required datasets during execution. Ensure an active internet connection for seamless downloading.

---

## Setup Instructions

1. Install required Python libraries using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure the `fastalign` folder is present in the project directory.

3. Download the pre-trained model using the provided command and place it in the project directory.

---

## Execution Instructions

### To Train the Model
- **File:** `train.py`  
- **Command:**
   ```bash
   python train.py
   ```
- **Note:** Training takes more than 30 days and requires significant computational resources. This is not intended for execution as part of the submission.

### To Test the Model
- **File:** `test.py`  
- **Command:**
   ```bash
   python test.py
   ```
- **Note:** Testing on a large dataset will take considerable time. Visualization is not generated in this script.

### To Run the Demo
- **File:** `demo.py`  
- **Command:**
   ```bash
   python demo.py
   ```
- **Recommended:** This script demonstrates the model’s capabilities, generates visualizations, and evaluates translation performance.

---

## Important Notes

- **Execution Environment:** This project is designed to run on Research Computing (RC) services. It is incompatible with Narnia due to resource limitations.  
- **Model Size:** The pre-trained model exceeds the submission size limit (2GB) and must be downloaded separately via the Google Drive link provided above.  
- **Dataset:** All scripts automatically download the necessary datasets; no external dataset submission is required.  
- **Dependencies:** Ensure `fastalign` is available in the project directory as it is used for alignment calculations.

---
