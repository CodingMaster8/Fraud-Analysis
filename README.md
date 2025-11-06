# Fraud Analysis on Bank Checks

Detecting fraudulent bank checks that have been altered.

### Types of fraud we want to detect:
We want to focus only on pixel level fraud. This includes:
- Digital image manipulation (Photoshop, image editing software)
- Physical alterations (tipex, whiteout, pen modifications)

Disregarding:
- Metadata analysis
- PDF text edits

### Approach:

![Work Pipeline](visual_insights/work_pipeline.png)

- Using Vision Language Models (VLMs) to detect alterations
- Generating additional dummy data by creating fraudulent versions of non-fraud checks
- Evaluating model performance using F1 score


## Aproach details:

### Data:
Create dummy fraudulent checks by:
- Resizing, rotating, and adding noise of fraudulent checks to increase dataset diversity - COMPLETE


Level 1: Pixel-Level Anomaly Detection - COMPLETE
Error Level Analysis (ELA): Detects JPEG compression inconsistencies
Noise Analysis: Different regions should have consistent noise patterns
Edge Detection: Look for unnatural edges around altered areas

Level 2: Vision Language Models (VLMs)
Prompting
 
Level 3: Ensemble Approach
Combine multiple signals:

VLM classification
Traditional CV anomaly scores
OCR consistency checks (verify amounts match in numbers and words)

### Evaluation Strategy:
Use F1-score as primary metric (good for imbalanced datasets)
Generate confusion matrix focusing on:
False negatives (missed fraud) - most costly
False positives (legitimate flagged as fraud)
Create interpretability outputs (highlight suspicious regions)

###  App (TODO):
A web app where users can upload check images and receive a fraud likelihood score along with highlighted suspicious areas.
Built with Streamlit for easy deployment and user interaction.
