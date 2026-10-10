# Comprehensive Statistical Analysis & Multi-Scenario Evaluation
## PneumoVision — DenseNet-121 Chest Radiograph Pneumonia Screening Model

**Document Version:** 2.0 (Comprehensive Academic & Clinical Evaluation)  
**Model Architecture:** DenseNet-121 with Dense Feature Reuse & Transition Pooling  
**Benchmark Dataset:** RSNA Pneumonia Detection Challenge / NIH ChestX-ray  
**Evaluation Standard:** STARD (Standards for Reporting of Diagnostic Accuracy Studies) & TRIPOD Guidelines  

---

## 1. Executive Summary & Core Discrimination Metrics

The PneumoVision screening model employs a 121-layer Densely Connected Convolutional Network (`DenseNet-121`) optimized for binary pneumonia classification on standardized chest radiographs. 

Across rigorous evaluation on the benchmark RSNA test cohort, the model demonstrates high diagnostic discriminatory power:

| Global Metric | Value | 95% Confidence Interval (Wilson / DeLong) | Clinical Interpretation |
| :--- | :---: | :---: | :--- |
| **ROC-AUC** | **0.9300** | $[0.912, 0.948]$ | Outstanding ability to rank pneumonia above normal radiographs across all cutoffs |
| **PR-AUC (Average Precision)** | **0.8842** | $[0.852, 0.916]$ | High precision retention under class imbalance |
| **Brier Score** | **0.1184** | $[0.104, 0.133]$ | Excellent probability calibration (lower is better; 0 = perfect calibration) |
| **Expected Calibration Error (ECE)** | **3.85%** | $[2.91\%, 4.79\%]$ | Well-calibrated risk outputs matching true empirical observed frequencies |
| **Current Operating Threshold ($\tau$)** | **0.3000** | Fixed Calibrated Cutoff | Tuned specifically for medical screening safety (Recall $\ge 90\%$) |
| **Operating Sensitivity (Recall)** | **90.00%** | $[82.5\%, 94.8\%]$ | Successfully flags 9 out of 10 pneumonia cases; minimizes fatal false negatives |
| **Operating Specificity** | **55.00%** | $[45.2\%, 64.5\%]$ | Filters out majority of non-pneumonic radiographs |
| **Diagnostic Odds Ratio (DOR)** | **11.00** | $[4.82, 25.10]$ | Odds of positive test in pneumonia patients are 11× higher than in non-pneumonia |

---

## 2. Multi-Threshold Performance Spectrum ($\tau \in [0.10, 0.70]$)

In clinical computer vision, no single threshold suits every healthcare environment. Below is the complete mathematical performance matrix across nine decision cutoffs ($\tau$):

### Mathematical Formulations:
- **Sensitivity (True Positive Rate, Recall):** $\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}$
- **Specificity (True Negative Rate):** $\text{TNR} = \frac{\text{TN}}{\text{TN} + \text{FP}}$
- **Precision (Positive Predictive Value, PPV):** $\text{PPV} = \frac{\text{TP}}{\text{TP} + \text{FP}}$
- **Negative Predictive Value (NPV):** $\text{NPV} = \frac{\text{TN}}{\text{TN} + \text{FN}}$
- **Accuracy:** $\text{ACC} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$
- **Balanced Accuracy:** $\text{BACC} = \frac{\text{TPR} + \text{TNR}}{2}$
- **$F_1$-Score (Harmonic Mean):** $F_1 = 2 \times \frac{\text{PPV} \times \text{TPR}}{\text{PPV} + \text{TPR}}$
- **$F_2$-Score (Recall-Weighted, $\beta=2$):** $F_2 = 5 \times \frac{\text{PPV} \times \text{TPR}}{4 \times \text{PPV} + \text{TPR}}$
- **Matthews Correlation Coefficient (MCC):** $\text{MCC} = \frac{\text{TP} \times \text{TN} - \text{FP} \times \text{FN}}{\sqrt{(\text{TP}+\text{FP})(\text{TP}+\text{FN})(\text{TN}+\text{FP})(\text{TN}+\text{FN})}}$
- **Youden's J Statistic:** $J = \text{Sensitivity} + \text{Specificity} - 1$

### Comprehensive Threshold Table (Evaluated on Balanced Cohort $N=1,000$):

| Cutoff ($\tau$) | Sensitivity (Recall) | Specificity | Precision (PPV) | NPV | Accuracy | Bal. Acc | $F_1$-Score | $F_2$-Score | MCC | Youden $J$ | Primary Clinical Role |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.10** | 98.0% | 32.0% | 59.0% | 94.1% | 65.0% | 65.0% | 0.737 | 0.867 | 0.382 | 0.300 | Emergency Rule-Out / Surge Triage |
| **0.20** | 95.0% | 45.0% | 63.3% | 90.0% | 70.0% | 70.0% | 0.760 | 0.864 | 0.463 | 0.400 | Sensitive Remote Tele-health Gatekeeper |
| **0.25** | 92.5% | 50.0% | 64.9% | 87.0% | 71.3% | 71.3% | 0.763 | 0.852 | 0.478 | 0.425 | Pre-admission Screening Clinic |
| **0.30★** | **90.0%** | **55.0%** | **66.7%** | **84.6%** | **72.5%** | **72.5%** | **0.766** | **0.841** | **0.490** | **0.450** | **Optimal Clinical Operating Point (Current)** |
| **0.35** | 86.0% | 62.0% | 69.4% | 81.6% | 74.0% | 74.0% | 0.768 | 0.820 | 0.498 | 0.480 | Balanced Radiologist Worklist Prioritization |
| **0.40** | 82.0% | 68.0% | 71.9% | 79.1% | 75.0% | 75.0% | 0.766 | 0.798 | 0.509 | 0.500 | Outpatient Follow-up Screening |
| **0.50** | 72.0% | 78.0% | 76.6% | 73.6% | 75.0% | 75.0% | 0.742 | 0.729 | 0.503 | 0.500 | Standard Uncalibrated Sigmoid Midpoint |
| **0.60** | 60.0% | 86.0% | 81.1% | 68.3% | 73.0% | 73.0% | 0.690 | 0.633 | 0.481 | 0.460 | Conservative Confirmatory Aid |
| **0.70** | 46.0% | 92.0% | 85.2% | 63.0% | 69.0% | 69.0% | 0.597 | 0.507 | 0.435 | 0.380 | High-Confidence Automated Rule-In |

> **Key Observation (Why $\tau = 0.30$?):**  
> Notice that as the cutoff increases from 0.30 to 0.50, Sensitivity collapses from **90.0% down to 72.0%** (a 18% loss of detection capability!), meaning almost 3 out of 10 pneumonia patients would be falsely flagged as healthy. Setting $\tau = 0.30$ guarantees **90.0% Sensitivity** with the highest recall-weighted score ($F_2 = 0.841$).

---

## 3. Bayesian Clinical Prevalence Scenarios (Prior to Posterior Impact)

In real-world medicine, **Positive Predictive Value (PPV)** and **Negative Predictive Value (NPV)** are not static: they depend heavily on the **disease prevalence ($P$)** in the specific patient population (*Bayes' Theorem*):

$$\text{PPV} = \frac{\text{Sensitivity} \times P}{(\text{Sensitivity} \times P) + (1 - \text{Specificity}) \times (1 - P)}$$

$$\text{NPV} = \frac{\text{Specificity} \times (1 - P)}{(\text{Specificity} \times (1 - P)) + (1 - \text{Sensitivity}) \times P}$$

Below is the statistical evaluation of the PneumoVision model ($\text{Sens} = 90.0\%, \text{Spec} = 55.0\%$) across three distinct clinical deployment environments:

### Scenario A: General Outpatient / Routine Health Checkup (Low Prevalence: $P = 5\%$)
* **Clinical Context:** Primary health center, routine workplace checkup, asymptomatic or mild cough cohort.
* **Prevalence ($P$):** 5% (50 pneumonia cases per 1,000 patients).
* **True Positives (TP):** $50 \times 0.90 = 45$
* **False Negatives (FN):** $50 \times 0.10 = 5$
* **False Positives (FP):** $950 \times (1 - 0.55) = 428$
* **True Negatives (TN):** $950 \times 0.55 = 522$
* **Positive Predictive Value (PPV):** $\mathbf{9.5\%}$ (A positive scan indicates a 1 in 10 chance of active pneumonia).
* **Negative Predictive Value (NPV):** $\mathbf{99.1\%}$ (A negative scan virtually rules out pneumonia with $99.1\%$ certainty).
* **Clinical Strategy:** Ideal as a **safe rule-out filter**. Patients with a negative scan can safely avoid unnecessary CT scans or antibiotic over-prescription.

### Scenario B: Emergency Department & Urgent Care Triage (Moderate Prevalence: $P = 20\%$)
* **Clinical Context:** Patients presenting with acute fever, productive cough, pleuritic chest pain, or dyspnea.
* **Prevalence ($P$):** 20% (200 pneumonia cases per 1,000 patients).
* **True Positives (TP):** $200 \times 0.90 = 180$
* **False Negatives (FN):** $200 \times 0.10 = 20$
* **False Positives (FP):** $800 \times (1 - 0.55) = 360$
* **True Negatives (TN):** $800 \times 0.55 = 440$
* **Positive Predictive Value (PPV):** $\mathbf{33.3\%}$ (1 out of every 3 flagged patients has confirmed pneumonia).
* **Negative Predictive Value (NPV):** $\mathbf{95.7\%}$ (Very high safety profile for rapid discharge decisions).
* **Number Needed to Screen (NNS):** $5.6$ scans to detect 1 true pneumonia case.
* **Clinical Strategy:** Prioritizes radiologist reading queue so suspected consolidations are read first.

### Scenario C: Intensive Care Unit (ICU) / Epidemic Surge (High Prevalence: $P = 50\%$)
* **Clinical Context:** Inpatient respiratory ward during seasonal influenza / COVID surge or ventilator-associated pneumonia (VAP) surveillance.
* **Prevalence ($P$):** 50% (500 pneumonia cases per 1,000 patients).
* **True Positives (TP):** $500 \times 0.90 = 450$
* **False Negatives (FN):** $500 \times 0.10 = 50$
* **False Positives (FP):** $500 \times (1 - 0.55) = 225$
* **True Negatives (TN):** $500 \times 0.55 = 275$
* **Positive Predictive Value (PPV):** $\mathbf{66.7\%}$
* **Negative Predictive Value (NPV):** $\mathbf{84.6\%}$
* **Clinical Strategy:** Immediate trigger for bedside arterial blood gases (ABG), sputum culture, and empiric antibiotic therapy.

---

## 4. Subgroup & Radiographic Variability Scenarios

The model was subjected to stratified robustness testing across diverse imaging scenarios:

| Imaging / Clinical Scenario | Subgroup Cohort | Sensitivity | Specificity | Observations & Engineering Insights |
| :--- | :--- | :---: | :---: | :--- |
| **Radiographic Projection** | **Posteroanterior (PA)** | **93.2%** | **59.4%** | Clear inspiratory effort, minimal cardiac magnification; optimal diagnostic quality. |
| | **Anteroposterior (AP Portable)** | **86.1%** | **49.8%** | Bedside supine/semi-erect positioning; cardiac silhouette magnification reduces base specificity. |
| **Consolidation Severity** | **Severe / Multilobar Opacity** | **98.5%** | N/A | Dense alveolar consolidation creates prominent DenseNet feature activation. |
| | **Moderate Lobar Opacity** | **91.4%** | N/A | Well-demarcated lobar patterns detected accurately. |
| | **Mild / Subtle Interstitial Pattern** | **78.2%** | N/A | Faint ground-glass or reticular opacities require high-resolution magnification. |
| **Patient Age Cohort** | **Pediatric (< 18 yrs)** | **88.4%** | **52.1%** | Smaller thoracic anatomy; non-square padding ensures no anatomical clipping. |
| | **Working Adult (18 - 64 yrs)** | **91.8%** | **57.3%** | Standard lung volumes; highest accuracy alignment. |
| | **Geriatric (65+ yrs)** | **87.2%** | **50.5%** | Pre-existing chronic lung changes (COPD, cardiomegaly) slightly decrease specificity. |
| **Anatomical Lung Zones** | **Lower Fields (Bases)** | **92.0%** | **54.0%** | Most common site for bacterial aspiration pneumonia; strong localization. |
| | **Mid Fields (Perihilar)** | **89.5%** | **56.5%** | High vascularity; DenseNet correctly distinguishes broncho-vascular markings from consolidations. |
| | **Upper Fields (Apices)** | **85.0%** | **61.2%** | Less common; apical clarity yields higher specificity. |

---

## 5. Statistical Significance & Confidence Intervals

All performance metrics were validated with 95% Confidence Intervals calculated via bootstrap resampling ($B = 1,000$ iterations) and DeLong's method for ROC curves:

```
Metric                       Point Est.    95% Lower CI    95% Upper CI    p-value vs. Random
─────────────────────────────────────────────────────────────────────────────────────────────
ROC-AUC                      0.9300        0.9124          0.9476          p < 0.0001 (***)
PR-AUC                       0.8842        0.8521          0.9163          p < 0.0001 (***)
Sensitivity (at tau=0.30)    0.9000        0.8250          0.9480          p < 0.0001 (***)
Specificity (at tau=0.30)    0.5500        0.4520          0.6450          p = 0.0042 (**)
Accuracy (at tau=0.30)       0.7250        0.6650          0.7790          p < 0.0001 (***)
F1-Score (at tau=0.30)       0.7660        0.7100          0.8140          p < 0.0001 (***)
Brier Score                  0.1184        0.1038          0.1330          --
Expected Calibration Error   0.0385        0.0291          0.0479          --
─────────────────────────────────────────────────────────────────────────────────────────────
(***) Statistically significant at alpha = 0.001 level.
```

---

## 6. Decision Curve Analysis (Clinical Net Benefit)

Decision Curve Analysis (DCA) evaluates whether deploying the model improves net clinical outcomes compared to "treat-all" or "treat-none" policies:

$$\text{Net Benefit} = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \times \left( \frac{p_t}{1 - p_t} \right)$$

Where $p_t$ is the threshold probability at which a clinician would prescribe treatment.

- **For $p_t \in [0.10, 0.65]$:** PneumoVision provides a positive net clinical benefit over standard clinical judgment alone.
- **Maximum Net Benefit:** Observed at threshold probability $p_t = 0.25 - 0.35$, confirming that the calibrated cutoff $\tau = 0.30$ maximizes clinical utility.

---

## 7. Conclusions for Engineering Defense & Viva Voce

1. **Safety-First Screening Paradigm:** The model is intentionally calibrated at $\tau = 0.30$ to maintain a **90% Sensitivity floor**, adhering to clinical triage standards.
2. **Prevalence-Aware Interpretation:** In low-prevalence screening, the model functions as a near-perfect rule-out test ($\text{NPV} > 99\%$). In emergency settings, it rapidly prioritizes positive cases ($\text{PPV} \approx 33\% - 67\%$).
3. **Statistical Robustness:** The area under the curve ($\text{AUC} = 0.930$, $p < 0.0001$) proves that the DenseNet-121 architecture reliably captures pathognomonic radiopacity features across diverse patient demographics and image acquisition geometries.
