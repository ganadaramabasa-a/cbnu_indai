# 1. Import required modules
from anomalib.data import Folder, FolderDataset
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Patchcore, Draem, Padim

if __name__ == '__main__':
    # 2. Create custom dataset
    datamodule = Folder(
        name="hazelnut_toy",
        root="./datasets/hazelnut_toy",  # Path to download/store the dataset
        normal_dir="good",
        abnormal_dir="crack",
        mask_dir="mask/crack"
    )

    # Setup the datamodule
    datamodule.setup()

    # 3. Initialize the model
    # Patchcore is a good choice for beginners
    model = Patchcore(
        num_neighbors=6,  # Override default model settings
    )

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
