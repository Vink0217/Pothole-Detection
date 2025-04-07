from pathlib import Path
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["libgl1-mesa-glx", "libglib2.0-0"])
    .pip_install(["ultralytics~=8.3.93", "opencv-python~=4.10.0"])
)

# Create or retrieve the volume
volume = modal.Volume.from_name("yolo_finetune")

# Define the volume path inside the container
volume_path = Path("/root")/"data"

# Create the Modal app with the image and volume mounted
app = modal.App("yolo-finetune", image=image, volumes={volume_path: volume})

MINUTES = 60

TRAIN_GPU_COUNT = 1
TRAIN_GPU = f"A100:{TRAIN_GPU_COUNT}"
TRAIN_CPU_COUNT = 4


@app.function(
    gpu=TRAIN_GPU,
    cpu=TRAIN_CPU_COUNT,
    timeout= 240 * MINUTES,
)
def train(
    model_id: str,
    resume=False,
    quick_check=False,
):  
    from ultralytics import YOLO

    volume.reload()  # Ensure volume is synced

    model_path = volume_path / "runs" / model_id
    model_path.mkdir(parents=True, exist_ok=True)

    data_path = volume_path / "dataset" / "dataset.yaml"
    best_weights = model_path / "weights" / "last.pt"

    if resume and best_weights.exists():
        model = YOLO(str(best_weights))
    else:
        model = YOLO("yolov9c.pt")

    # If best.pt training is finished, you should force resume to False:
    # For example, you might force it here:

    model.train(
        data=data_path,
        fraction=0.04 if quick_check else 1.0,
        device=list(range(TRAIN_GPU_COUNT)),
        epochs=1 if quick_check else 40,  # set total epochs higher as desired
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

@app.cls(gpu="T4")
class Inference:
    def __init__(self, weights_path):
        from ultralytics import YOLO
        self.weights_path = weights_path
        self.model = YOLO(self.weights_path) 

    @modal.method()
    def stream(self, model_id: str, image_files: list = None):
        import time

        completed, start = 0, time.monotonic_ns()
        for image_path in image_files:
            results = self.model.predict(
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


@app.function()
def infer(model_id: str):
    import os

    inference = Inference(volume_path / "runs" / model_id / "weights" / "last.pt")

    test_dir = volume_path / "dataset" / "val"
    test_images_path = [str(test_dir / f) for f in os.listdir(str(test_dir)) if f.lower().endswith(".jpg")]

    print(f"{model_id}: Running streaming inferences on all images in the test set...")
    inference.stream.remote(model_id, test_images_path)

