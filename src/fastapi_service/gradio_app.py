# AutonomousVehiclePerception/src/fastapi_service/gradio_app.py
"""Gradio demo UI for autonomous vehicle perception models.

Provides an interactive web interface for:
- Uploading camera images for 2D CNN / FPN inference
- Visualizing detection results with overlaid predictions
- Comparing model outputs side by side
"""

import io

import gradio as gr
import numpy as np
import requests
from PIL import Image

# FastAPI backend URL
API_URL = "http://localhost:8001"


def predict_image(image, model_choice):
    """Send image to FastAPI backend and return predictions.

    Args:
        image: Input image (numpy array from Gradio)
        model_choice: Which model to use ('2D CNN' or 'FPN-ResNet50')
    Returns:
        Tuple of (annotated_image, result_text)
    """
    if image is None:
        return None, "Please upload an image."

    # Convert numpy to bytes
    pil_image = Image.fromarray(image)
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)

    # Select endpoint
    endpoint = "/predict/2d" if model_choice == "2D CNN" else "/predict/fpn"

    try:
        response = requests.post(
            f"{API_URL}{endpoint}",
            files={"file": ("image.png", buf, "image/png")},
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        # Format results
        result_text = (
            f"Model: {result['model']}\n"
            f"Output shape: {result['shape']}\n"
            f"Inference time: {result['inference_time_ms']:.1f} ms\n"
            f"Device: {result['device']}\n"
            f"Unique classes detected: {len(set(result['predictions']))}"
        )

        # Create simple heatmap overlay from predictions
        pred_array = np.array(result["predictions"])
        if len(pred_array) > 0:
            # Reshape predictions to approximate spatial map
            side = int(np.sqrt(len(pred_array)))
            if side > 0:
                pred_map = pred_array[: side * side].reshape(side, side)
                pred_resized = np.array(
                    Image.fromarray(pred_map.astype(np.uint8) * 25).resize((image.shape[1], image.shape[0]), Image.NEAREST)
                )
                # Overlay as colored mask
                overlay = image.copy()
                mask = pred_resized > 0
                overlay[mask, 0] = np.clip(overlay[mask, 0].astype(int) + 50, 0, 255).astype(np.uint8)
                return overlay, result_text

        return image, result_text

    except requests.ConnectionError:
        return image, f"ERROR: Cannot connect to {API_URL}. Is the FastAPI server running?"
    except Exception as e:
        return image, f"ERROR: {str(e)}"


def check_health():
    """Check FastAPI backend health."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        result = response.json()
        return (
            f"Status: {result['status']}\n"
            f"Device: {result['device']}\n"
            f"GPU available: {result['gpu_available']}\n"
            f"Models loaded: {', '.join(result['models_loaded'])}"
        )
    except requests.ConnectionError:
        return f"ERROR: Cannot connect to {API_URL}"
    except Exception as e:
        return f"ERROR: {str(e)}"


def build_gradio_app():
    """Build and return the Gradio Blocks app."""
    with gr.Blocks(title="AV Perception Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚗 Autonomous Vehicle Perception Demo")
        gr.Markdown("Upload a camera image to run object detection using CNN models trained on KITTI/nuScenes data.")

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Input Camera Image", type="numpy")
                model_choice = gr.Radio(
                    choices=["2D CNN", "FPN-ResNet50"],
                    value="2D CNN",
                    label="Model",
                )
                predict_btn = gr.Button("Run Detection", variant="primary")
                health_btn = gr.Button("Check API Health")
                health_output = gr.Textbox(label="API Health", lines=4)

            with gr.Column(scale=1):
                output_image = gr.Image(label="Detection Result")
                result_text = gr.Textbox(label="Inference Details", lines=6)

        predict_btn.click(fn=predict_image, inputs=[input_image, model_choice], outputs=[output_image, result_text])
        health_btn.click(fn=check_health, outputs=health_output)

        gr.Markdown("### Notes")
        gr.Markdown(
            "- Models are initialized with random weights unless trained weights are placed in `weights/` directory.\n"
            "- Supported input: any RGB image (resized to 480×640 internally).\n"
            "- The FastAPI server must be running on port 8001."
        )

    return demo


if __name__ == "__main__":
    demo = build_gradio_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)
