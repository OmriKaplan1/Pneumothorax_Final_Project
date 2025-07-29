# ======================================================================
#  Xception  TEST  + (optional) SpO₂-based certainty adjustment
#  -------------------------------------------------------------
#  • saves:  predictions.csv
#            confusion_matrix.csv
#            classification_report.txt
# ======================================================================
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

# ----------------------------------------------------------------------
#  CONFIGURATION
# ----------------------------------------------------------------------
IMAGE_DIR          = r"C:\\project\\png_images"          # folder with *_test_* images
MODEL_PATH         = r"C:\project\\xception_cbam_256_higher_weight\\pneumothorax_xception_cbam_256_2.keras"
OUTPUT_DIR         = r"C:\\project\\xception_cbam_256_higher_weight"        # all csv / txt will land here
IMG_SIZE           = (256, 256)
DECISION_THRESHOLD = 0.50                             # label = 1 if adj_score ≥ threshold
USE_SPO2_ADJUST    = True                             # ⬅ toggle SpO₂ feature ON / OFF
random.seed(42)                                       # reproducibility

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
#  SpO₂ helper
# ----------------------------------------------------------------------
def sample_spo2(true_label: int) -> float:
    """
    Simulate a saturation value.
      • normal  (label 0) → U(94, 98)
      • pneumo  (label 1) → U(88, 94)
    """
    return random.uniform(94, 98) if true_label == 0 else random.uniform(88, 94)

def spo2_adjustment(spo2: float) -> float:
    """
    f(SpO₂) from the provided piece-wise function.
    """
    if spo2 < 90:
        return +0.20
    elif 90 <= spo2 < 94:
        return +0.10
    elif 94 <= spo2 <= 98:
        return 0.0
    else:                                # SpO₂ > 98
        return -0.05

# ----------------------------------------------------------------------
#  Load model (no custom loss needed)
# ----------------------------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅  Xception model loaded")

# ----------------------------------------------------------------------
#  Gather test images
# ----------------------------------------------------------------------
test_imgs = [f for f in os.listdir(IMAGE_DIR)
             if f.endswith(".png") and "_test_" in f]

print(f"🖼️  Found {len(test_imgs)} test images")

# ----------------------------------------------------------------------
#  Predict
# ----------------------------------------------------------------------
records = []

for fname in test_imgs:
    true_label = int(fname.split("_")[2])                      # 0 or 1
    path       = os.path.join(IMAGE_DIR, fname)

    # load & preprocess
    img = load_img(path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # raw model score
    raw_score = float(model.predict(arr, verbose=0)[0][0])

    # SpO₂ adjustment (optional)
    spo2 = sample_spo2(true_label)
    adj  = spo2_adjustment(spo2) if USE_SPO2_ADJUST else 0.0
    final_score = max(0.0, min(1.0, raw_score + adj))          # clamp to [0,1]

    pred_label = int(final_score >= DECISION_THRESHOLD)

    records.append({
        "filename"        : fname,
        "true_label"      : true_label,
        "predicted_label" : pred_label,
        "prediction_score": round(final_score, 5),
        "raw_score"       : round(raw_score, 5),
        "SpO2"            : round(spo2, 1),
        "adjustment"      : adj
    })

# ----------------------------------------------------------------------
#  Save predictions CSV
# ----------------------------------------------------------------------
pred_csv = os.path.join(OUTPUT_DIR, "predictions.csv")
pd.DataFrame(records).to_csv(pred_csv, index=False)
print(f"📄  predictions.csv saved → {pred_csv}")

# ----------------------------------------------------------------------
#  Confusion-matrix + report
# ----------------------------------------------------------------------
df      = pd.DataFrame(records)
y_true  = df["true_label"].values
y_pred  = df["predicted_label"].values

cm      = confusion_matrix(y_true, y_pred)
report  = classification_report(y_true, y_pred,
                                target_names=["normal", "pneumothorax"])

print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:\n", report)

cm_csv  = os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
rep_txt = os.path.join(OUTPUT_DIR, "classification_report.txt")

pd.DataFrame(cm,
             index=["True_Normal", "True_Pneumo"],
             columns=["Pred_Normal", "Pred_Pneumo"]
            ).to_csv(cm_csv)

with open(rep_txt, "w") as f:
    f.write(report)

print(f"✅  confusion_matrix.csv saved → {cm_csv}")
print(f"✅  classification_report.txt saved → {rep_txt}")

# ----------------------------------------------------------------------
#  Timestamp & done
# ----------------------------------------------------------------------
print(f"\n🏁 Completed on {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
