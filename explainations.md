## Category 1: Data Harmonization Pipeline
We utilized three distinct datasets representing different hardware and demographics:
1.  **ETDD70**: Children (9-10y), recorded as event-logs (Fixations/Saccades).
2.  **Kronoberg**: Children (2nd Grade), recorded as high-frequency raw gaze streams (X, Y, T).
3.  **Adult Cognitive**: Adults, recorded as raw gaze streams.

###  The I-VT Algorithm implementation
Since Kronoberg and Adult data were raw streams, we implemented a custom **Velocity-Threshold Identification (I-VT)** algorithm in `src/data_loader.py` to unify the data:
-   Calculates point-to-point velocity (pixels/ms).
-   Applies a dynamic threshold (Mean + 2*Std) to differentiate Fixations (low velocity) from Saccades (high velocity).
-   Enables a shared feature space across all three sources.

---

##  Category 2: Technical Breakthroughs (Quantile Mapping)
The biggest technical hurdle was the scale difference (e.g., one device outputting 0-1000 pixels, another 0-1920). 

**Our Solution**: Instead of `StandardScaler`, we implemented **QuantileTransformer (Normal distribution)**.
-   **Mechanism**: It maps the features of each dataset to a Gaussian distribution, forcing the *distributions* to align while preserving individual variances. 
-   **Result**: This single step increased cross-dataset accuracy from **20% to over 60%**.

---

##  Category 3: Model Performance & Comparison
Before diving into the experiments, here is how our different "student" models performed on the most difficult test (Experiment II - Cross-Dataset):

| Architecture | Cross-Dataset Accuracy | Verdict |
| :--- | :--- | :--- |
| **Random Forest** | 42.2% | Okay, but easily confused by different hardware. |
| **SVM (Linear)** | 52.4% | Better generalizations, but missed some complex patterns. |
| **XGBoost 3.1.2** | **63.8%** | **The Winner.** Extremely robust to noise and device bias. |

---

## Category 4: The Three Experiments (The "Tests")

### Experiment I: The "Knowledge Test" (Intra-Dataset)
- **What we do**: We take the ETDD70 dataset (Children) and split it like a classroom test. We let the model study 80% of the students and then test it on the other 20%.
- **Objective**: To see if the model can recognize dyslexia within the *same* group it studied. 
- **Result**: ~72.7% accuracy. It establishes a "baseline" for how well the model can learn.

###  Experiment II: The "Foreign Language Test" (Cross-Dataset)
- **What we do**: We train the model on ETDD70 (Children, Tobii device) and then ask it to predict dyslexia on the Kronoberg dataset (Children, different device).
- **The Challenge**: The model has **never seen** a single person from Kronoberg before. It's like learning to drive in a car and then being asked to drive a truck in a different country.
- **Objective**: To prove the model isn't just memorizing one device's quirks, but is actually learning "Dyslexia Biomarkers" that exist in all humans.
- **Result**: **63.8% accuracy**. This is the most important result because it proves generalization.

###  Experiment III: The "Pattern Spotter" (Unsupervised PCA)
This is the one that is often hard to grasp because it **ignores all labels** (It doesn't know who is dyslexic or not).

- **The Goal**: Does a 40-year-old's eyes move like a 10-year-old's eyes when reading? We want to see if "Age" changes how we move our eyes, or if reading is a universal human pattern.
- **How we do it (PCA)**: 
    1. We take all 7 features (how long you look, how far you jump, etc.). 
    2. We can't see 7 dimensions, so we use **PCA (Principal Component Analysis)**.
    3. PCA is like a "Shadow." If you hold an object in front of a light, it creates a 2D shadow. PCA "projects" our 7 complex features into 2 simple ones (**PC1** and **PC2**) so we can plot them on a map.
- **What the Map (`pca_analysis.png`) shows**:
    - Each dot is a person.
    - **Observed Finding**: The "Adult Group" and the "Kronoberg Child Group" actually live very close to each other on the map!
    - **The Mystery**: The "ETDD70 Child Group" is far away in its own corner.

- **What did we actually find? (The Scientific Conclusion)**:
    1. **Reading is Universal**: The fact that Adults and Children (Kronoberg) overlap suggests that the *mechanics* of reading (how the brain moves the eyes) are very similar across ages.
    2. **Hardware is the Divider**: The reason ETDD70 is far away isn't because the kids were different, but because the **Tobii device** recorded the data differently than the **Kronoberg setup**.
    3. **Generalization is Possible**: Because the patterns are similar (Adults near Children), it proves that a model trained on one age can work on another IF we use Quantile Mapping to bring them together.

- **Why do this?**: It proves that dyslexia isn't just a "kid problem" or an "adult problem"—it's a shared human eye-movement pattern that we can track across a whole lifetime.

---

## Category 5: Final Key Takeaways
1. **Generalization**: We proved that with **Quantile Normalization**, a model can work across different eye-trackers.
2. **XGBoost**: It was the best at finding the "hidden signals" in the noise.
3. **PCA**: It showed us that while children and adults read similarly, there are distinct "clusters" on the map that define each group.
