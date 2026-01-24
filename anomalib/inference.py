# 1. Import required modules
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

# 2. Initialize the model and load weights
model = Patchcore()
engine = Engine()

# 3. Prepare test data
# You can use a single image or a folder of images
dataset = PredictDataset(
    # path="datasets/MVTecAD/bottle/test/good/000.png",
    path="datasets/MVTecAD/bottle/test/broken_large/000.png",
    image_size=(256, 256),
)

# 4. Get predictions
predictions = engine.predict(
    model=model,
    dataset=dataset,
    ckpt_path="results/Patchcore/MVTecAD/bottle/latest/weights/lightning/model.ckpt",
)

# 5. Access the results
if predictions is not None:
    for prediction in predictions:
        image_path = prediction.image_path
        anomaly_map = prediction.anomaly_map  # Pixel-level anomaly heatmap
        pred_label = prediction.pred_label  # Image-level label (0: normal, 1: anomalous)
        pred_score = prediction.pred_score  # Image-level anomaly score
        print(image_path)
        print(anomaly_map)
        print(pred_label)
        print(pred_score)