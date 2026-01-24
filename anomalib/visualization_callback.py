# 1. Import required modules
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.visualization import ImageVisualizer
import matplotlib.pyplot as plt
from anomalib.data import MVTecAD

if __name__ == '__main__':
    # Custom visualization settings
    visualizer = ImageVisualizer(
        fields_config={
            "image": {},  # Default image display
            "anomaly_map": {
                "colormap": True,
                "normalize": True
            },
            "pred_mask": {
                "mode": "contour",
                "color": (255, 0, 0),
                "alpha": 0.7
            },
            "gt_mask": {
                "mode": "contour",
                "color": (0, 255, 0),
                "alpha": 0.7
            }
        }
    )

    # 2. Initialize the model and load weights
    # model = Patchcore(visualizer=visualizer)
    model = Patchcore(visualizer=visualizer)
    engine = Engine()

    # 3. Prepare test data
    # You can use a single image or a folder of images
    datamodule = MVTecAD(
        root="./datasets/MVTecAD",  # Path to download/store the dataset
        category="bottle",  # MVTec category to use
        train_batch_size=32,  # Number of images per training batch
        eval_batch_size=32,  # Number of images per validation/test batch
    )

    # 4. Get predictions
    predictions = engine.test(
        model=model,
        dataloaders=datamodule,
        ckpt_path="results/Patchcore/MVTecAD/bottle/latest/weights/lightning/model.ckpt",
    )