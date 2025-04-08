from pathlib import Path
import modal

# Define a custom Docker-like image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")  # Use Debian with Python 3.12
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])  # System libs for OpenCV
    .pip_install(["ultralytics~=8.3.93", "opencv-python~=4.10.0"])  # ML dependencies
)

# Create or get a Modal volume (persistent storage)
volume = modal.Volume.from_name("yolo_finetune")

# Path where data will be stored inside the Modal container
volume_path = Path("/root") / "data"

# Define the Modal app with image + attached volume
app = modal.App("yolo-finetune", image=image, volumes={volume_path: volume})

# Training configuration
MINUTES = 60
TRAIN_GPU_COUNT = 1
TRAIN_GPU = f"A100:{TRAIN_GPU_COUNT}"  # Use A100 GPU
TRAIN_CPU_COUNT = 4  # Use 4 CPUs

# Define a training function
@app.function(
    gpu=TRAIN_GPU,
    cpu=TRAIN_CPU_COUNT,
    timeout=240 * MINUTES,  # Max runtime = 4 hours
)
def train(model_id: str, resume=False, quick_check=False):
    from ultralytics import YOLO

    volume.reload()  # Sync volume before starting

    # Define paths
    model_path = volume_path / "runs" / model_id
    model_path.mkdir(parents=True, exist_ok=True)
    data_path = volume_path / "dataset" / "dataset.yaml"
    best_weights = model_path / "weights" / "last.pt"

    # Resume or initialize model
    if resume and best_weights.exists():
        model = YOLO(str(best_weights))
    else:
        model = YOLO("yolov9c.pt")

    # Train the model
    model.train(
        data=data_path,
        fraction=0.04 if quick_check else 1.0,  # Use small data for quick test
        device=list(range(TRAIN_GPU_COUNT)),  # Use available GPUs
        epochs=1 if quick_check else 40,
        batch=32,
        imgsz=320 if quick_check else 640,
        seed=120,
        workers=max(TRAIN_CPU_COUNT // TRAIN_GPU_COUNT, 1),
        cache=True,
        project=f"{volume_path}/runs",
        name=model_id,
        verbose=True,
        resume=resume,
    )

# Class for inference
@app.cls(gpu="T4")  # Run inference on T4 GPU
class Inference:
    def __init__(self, weights_path):
        from ultralytics import YOLO
        self.weights_path = weights_path
        self.model = YOLO(self.weights_path)  # Load model

    @modal.method()
    def stream(self, model_id: str, image_files: list = None):
        import time
        completed, start = 0, time.monotonic_ns()
        for image_path in image_files:
            self.model.predict(
                image_path,
                half=True,
                save=True,
                exist_ok=True,
                verbose=False,
                project=f"{volume_path}/predictions/{model_id}",
                conf=0.4,
            )
            completed += 1

        elapsed_seconds = (time.monotonic_ns() - start) / 1e9
        print("Inferences per second:", round(completed / elapsed_seconds, 2))
        print(f"TOTAL INFERENCES: {completed}")

# Standalone inference trigger
@app.function()
def infer(model_id: str):
    import os
    inference = Inference(volume_path / "runs" / model_id / "weights" / "last.pt")

    test_dir = volume_path / "dataset" / "val"
    test_images_path = [str(test_dir / f) for f in os.listdir(str(test_dir)) if f.lower().endswith(".jpg")]

    print(f"{model_id}: Running streaming inferences on all images in the test set...")
    inference.stream.remote(model_id, test_images_path)
