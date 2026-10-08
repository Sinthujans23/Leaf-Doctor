"""Local web API and phone-friendly page for potato leaf predictions."""
from pathlib import Path
import io
import json

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "potato_leaf_cnn.keras"
LABELS_PATH = ROOT / "model" / "labels.json"
model = None
class_names = None


def load_model_once():
    global model, class_names
    if model is None:
        if not MODEL_PATH.is_file() or not LABELS_PATH.is_file():
            raise FileNotFoundError("Model not found. First run: python train.py")
        model = tf.keras.models.load_model(MODEL_PATH)
        class_names = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return model


def analyze_image(raw_image):
    try:
        image = Image.open(io.BytesIO(raw_image)).convert("RGB")
        image = image.resize((160, 160))
        pixels = np.asarray(image, dtype=np.float32)[None, ...]
        probabilities = load_model_once().predict(pixels, verbose=0)[0]
    except FileNotFoundError as error:
        return {"error": str(error)}, 503
    except (UnidentifiedImageError, OSError, ValueError):
        return {"error": "Could not read that image. Please choose a JPG or PNG photo."}, 400
    index = int(np.argmax(probabilities))
    label = class_names[index]
    confidence = float(probabilities[index])
    # Low confidence is an honest inconclusive result, not a disease diagnosis.
    if confidence < 0.60:
        return {"status": "uncertain", "label": "Uncertain result", "confidence": confidence,
                "note": "Try a clear, close-up photo of one potato leaf in natural light."}, 200
    healthy = label == "Potato___healthy"
    display = "Healthy potato leaf" if healthy else label.replace("Potato___", "").replace("_", " ").title()
    return {"status": "healthy" if healthy else "disease", "label": display,
            "confidence": confidence, "note": "Model result is an estimate; confirm with an agriculture expert."}, 200


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/":
            payload = (ROOT / "static" / "index.html").read_bytes()
            return self.send_bytes(200, "text/html; charset=utf-8", payload)
        if route == "/manifest.json":
            return self.send_bytes(200, "application/manifest+json", (ROOT / "static" / "manifest.json").read_bytes())
        if route == "/api/health":
            payload = json.dumps({"ready": MODEL_PATH.is_file() and LABELS_PATH.is_file(), "crop": "potato"}).encode()
            return self.send_bytes(200, "application/json", payload)
        return self.send_bytes(404, "text/plain", b"Not found")

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            return self.send_bytes(404, "text/plain", b"Not found")
        content_type = self.headers.get("Content-Type", "")
        if "boundary=" not in content_type:
            return self.send_bytes(400, "application/json", b'{"error":"Please choose a leaf photo."}')
        boundary = content_type.split("boundary=", 1)[1].strip().strip('"').encode()
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        image_data = None
        for part in body.split(b"--" + boundary):
            if b'name="image"' in part and b"\r\n\r\n" in part:
                image_data = part.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
                break
        if image_data is None:
            result, status = {"error": "Please choose a leaf photo."}, 400
        else:
            result, status = analyze_image(image_data)
        return self.send_bytes(status, "application/json", json.dumps(result).encode())


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 5000), Handler)
    print("Open http://localhost:5000 on this computer. Press Ctrl+C to stop.")
    server.serve_forever()
