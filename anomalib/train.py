# 1. Import required modules
import torch
torch.set_float32_matmul_precision('high')

from anomalib.data import MVTecAD
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Patchcore, Draem, Padim, EfficientAd


if __name__ == '__main__':
    # 2. Create a dataset
    # MVTecAD is a popular dataset for anomaly detection
    datamodule = MVTecAD(
        root="./datasets/MVTecAD",  # Path to download/store the dataset
        category="bottle",  # MVTec category to use
        train_batch_size=1,  # Number of images per training batch
        eval_batch_size=1,  # Number of images per validation/test batch
    )

    # 3. Initialize the model
    # Patchcore is a good choice for beginners
    # model = Patchcore(
    #     num_neighbors=6,  # Override default model settings
    # )
    model = EfficientAd()

    # 4. Create the training engine
    engine = Engine(
        max_epochs=1,  # Override default trainer settings
    )

    # 5. Train the model
    # This produces a lightning model (.ckpt)
    engine.fit(datamodule=datamodule, model=model)

    # 6. Test the model performance
    test_results = engine.test(datamodule=datamodule, model=model)

    # 7. Export the model
    # Different formats are available: Torch, OpenVINO, ONNX
    engine.export(
        model=model,
        export_type=ExportType.TORCH,
    )
