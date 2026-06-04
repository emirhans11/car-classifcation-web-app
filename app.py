"""
╔══════════════════════════════════════════════════════════════╗
║   AI Vehicle Body Type Classifier — Gradio Web Interface    ║
║   Model: EfficientNet-B0 (timm) · 8 Vehicle Classes        ║
║   Author: Emirhan SOLAK Abdullah EKİCİ                                         ║
╚══════════════════════════════════════════════════════════════╝

This module provides a polished, dark-themed Gradio Blocks
interface for real-time vehicle body type classification.

Architecture
────────────
• load_model()        → loads the EfficientNet-B0 checkpoint
• preprocess_image()  → PIL → tensor pipeline (224×224, ImageNet norm)
• predict_image()     → inference + softmax → results dict
• create_chart()      → matplotlib horizontal bar chart of probabilities
"""

# ──────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────
import sys
import os

# Fix Windows console encoding (cp1254 cannot render Unicode emoji)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
import torch
import torch.nn.functional as F
import timm
import gradio as gr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Use the non-interactive Agg backend so matplotlib never tries to open a window
matplotlib.use("Agg")

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

# Path to the trained model weights (same directory as this script)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pth")

# The 8 vehicle body-type classes the model was trained on
CLASS_NAMES = [
    "F1-OPEN-WHEEL",
    "HATCHBACK",
    "MICRO",
    "PICK UP",
    "SEDAN",
    "STATION WAGON",
    "SUV",
    "VAN",
]

# Number of classes
NUM_CLASSES = len(CLASS_NAMES)

# Target image size expected by EfficientNet-B0
IMAGE_SIZE = 224

# ImageNet normalisation statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Automatically select GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────
# Colour palette for the probability bar chart
# ──────────────────────────────────────────────────────────────
CHART_COLORS = [
    "#6366f1",  # Indigo   — F1-OPEN-WHEEL
    "#22d3ee",  # Cyan     — HATCHBACK
    "#a78bfa",  # Violet   — MICRO
    "#f59e0b",  # Amber    — PICK UP
    "#10b981",  # Emerald  — SEDAN
    "#f472b6",  # Pink     — STATION WAGON
    "#3b82f6",  # Blue     — SUV
    "#ef4444",  # Red      — VAN
]

# ──────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────

def load_model() -> torch.nn.Module:
    """
    Load the EfficientNet-B0 model with trained weights.

    Returns
    -------
    torch.nn.Module
        The model in evaluation mode on the active device.

    Raises
    ------
    FileNotFoundError
        If the model weights file does not exist.
    RuntimeError
        If the checkpoint cannot be loaded or applied.
    """
    # --- Check that the weights file exists ---
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights not found at:\n{MODEL_PATH}\n"
            "Please place 'best_model.pth' in the same folder as app.py."
        )

    try:
        # Create the EfficientNet-B0 architecture without pretrained weights
        model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=NUM_CLASSES,
        )

        # Load the saved state dict
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state_dict)

        # Move to device and set to evaluation mode (disables dropout / batchnorm training)
        model.to(DEVICE)
        model.eval()

        print(f"[OK]  Model loaded successfully on {DEVICE}")
        return model

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model from '{MODEL_PATH}'.\n"
            f"Details: {exc}"
        ) from exc


# ──────────────────────────────────────────────────────────────
# Image Preprocessing
# ──────────────────────────────────────────────────────────────

# Compose the preprocessing pipeline once (reused on every call)
_preprocess_pipeline = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),   # Resize to 224×224
    transforms.ToTensor(),                          # HWC uint8 → CHW float [0,1]
    transforms.Normalize(                           # ImageNet normalisation
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image into a model-ready 4-D tensor.

    Parameters
    ----------
    image : PIL.Image.Image
        The input image (any size, any mode).

    Returns
    -------
    torch.Tensor
        Shape (1, 3, 224, 224) on the active device.

    Raises
    ------
    ValueError
        If the input is not a valid PIL image.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a valid PIL Image.")

    # Convert to RGB in case the image is RGBA, L, P, etc.
    image = image.convert("RGB")

    # Apply resize → tensor → normalise, then add batch dimension
    tensor = _preprocess_pipeline(image).unsqueeze(0).to(DEVICE)
    return tensor


# ──────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────

def predict_image(image: Image.Image, model: torch.nn.Module) -> dict:
    """
    Run inference on a single image and return structured results.

    Parameters
    ----------
    image : PIL.Image.Image
        The uploaded image.
    model : torch.nn.Module
        The loaded EfficientNet-B0 model.

    Returns
    -------
    dict
        {
            "predicted_class": str,
            "confidence": float,          # 0-100
            "probabilities": dict[str, float],  # class → probability (0-1)
        }

    Raises
    ------
    RuntimeError
        If inference fails for any reason.
    """
    try:
        tensor = preprocess_image(image)

        # No gradient computation during inference for speed & memory savings
        with torch.no_grad():
            logits = model(tensor)                       # (1, NUM_CLASSES)
            probs  = F.softmax(logits, dim=1).squeeze()  # (NUM_CLASSES,)

        probs_np = probs.cpu().numpy()

        # Build a class → probability mapping
        prob_dict = {
            CLASS_NAMES[i]: float(probs_np[i])
            for i in range(NUM_CLASSES)
        }

        # Determine top prediction
        top_idx        = int(np.argmax(probs_np))
        predicted_class = CLASS_NAMES[top_idx]
        confidence      = float(probs_np[top_idx]) * 100.0  # percentage

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": prob_dict,
        }

    except Exception as exc:
        raise RuntimeError(f"Inference failed: {exc}") from exc


# ──────────────────────────────────────────────────────────────
# Probability Bar Chart
# ──────────────────────────────────────────────────────────────

def create_chart(probabilities: dict[str, float]) -> plt.Figure:
    """
    Render a sleek horizontal bar chart of class probabilities.

    Parameters
    ----------
    probabilities : dict[str, float]
        Mapping of class name → probability (0-1).

    Returns
    -------
    matplotlib.figure.Figure
        Ready-to-display figure object.
    """
    # Sort classes by descending probability for readability
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    values = [item[1] * 100 for item in sorted_items]  # convert to %

    # Map each sorted label back to its colour
    color_map = dict(zip(CLASS_NAMES, CHART_COLORS))
    colors = [color_map[label] for label in labels]

    # --- Figure setup ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    # Draw horizontal bars
    bars = ax.barh(
        labels,
        values,
        color=colors,
        height=0.55,
        edgecolor="none",
        zorder=3,
    )

    # Add rounded-looking bar ends with a subtle glow
    for bar, val in zip(bars, values):
        bar.set_alpha(0.92)
        # Percentage label at the end of each bar
        ax.text(
            val + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#e2e8f0",
            fontfamily="sans-serif",
        )

    # Style the axes
    ax.invert_yaxis()  # highest probability at top
    ax.set_xlim(0, max(values) * 1.18 if max(values) > 0 else 100)
    ax.set_xlabel("Confidence (%)", fontsize=11, color="#94a3b8", labelpad=10)
    ax.tick_params(axis="y", colors="#e2e8f0", labelsize=10)
    ax.tick_params(axis="x", colors="#64748b", labelsize=9)

    # Subtle grid on x-axis only
    ax.xaxis.grid(True, linestyle="--", alpha=0.15, color="#475569")
    ax.set_axisbelow(True)

    # Remove spines for a clean look
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(pad=1.2)
    return fig


# ──────────────────────────────────────────────────────────────
# Gradio Callback
# ──────────────────────────────────────────────────────────────

def on_predict(image: Image.Image | None):
    """
    Gradio callback triggered by the Predict button.

    Returns a tuple consumed by the three output components:
        (predicted_label_html, confidence_html, chart_figure)
    """
    # --- Guard: no image uploaded ---
    if image is None:
        return (
            format_label_html("—", "Upload an image first"),
            format_confidence_html(None),
            create_empty_chart(),
        )

    try:
        results    = predict_image(image, model)
        pred_class = results["predicted_class"]
        confidence = results["confidence"]
        probs      = results["probabilities"]

        label_html = format_label_html(pred_class, get_class_emoji(pred_class))
        conf_html  = format_confidence_html(confidence)
        chart      = create_chart(probs)

        return label_html, conf_html, chart

    except Exception as exc:
        return (
            format_label_html("Error", str(exc)),
            format_confidence_html(None),
            create_empty_chart(),
        )


# ──────────────────────────────────────────────────────────────
# HTML Formatting Helpers
# ──────────────────────────────────────────────────────────────

def get_class_emoji(class_name: str) -> str:
    """Return a relevant emoji for the predicted vehicle class."""
    emojis = {
        "F1-OPEN-WHEEL": "🏎️ Formula 1",
        "HATCHBACK":     "🚗 Hatchback",
        "MICRO":         "🚙 Micro Car",
        "PICK UP":       "🛻 Pick-Up Truck",
        "SEDAN":         "🚘 Sedan",
        "STATION WAGON": "🚃 Station Wagon",
        "SUV":           "🏔️ SUV",
        "VAN":           "🚐 Van",
    }
    return emojis.get(class_name, class_name)


def format_label_html(title: str, subtitle: str) -> str:
    """Build a styled HTML card for the predicted class."""
    return f"""
    <div style="
        text-align: center;
        padding: 28px 20px;
        border-radius: 16px;
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        box-shadow: 0 0 30px rgba(99, 102, 241, 0.08);
    ">
        <div style="
            font-size: 32px;
            font-weight: 800;
            background: linear-gradient(90deg, #818cf8, #22d3ee);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        ">{title}</div>
        <div style="
            margin-top: 8px;
            font-size: 16px;
            color: #94a3b8;
            font-weight: 500;
        ">{subtitle}</div>
    </div>
    """


def format_confidence_html(confidence: float | None) -> str:
    """Build a styled HTML card for the confidence score."""
    if confidence is None:
        pct_text  = "—"
        bar_width = "0%"
        color     = "#475569"
    else:
        pct_text  = f"{confidence:.1f}%"
        bar_width = f"{confidence:.1f}%"
        # Colour shifts from amber (<60) → emerald (60-85) → cyan (>85)
        if confidence >= 85:
            color = "#22d3ee"
        elif confidence >= 60:
            color = "#10b981"
        else:
            color = "#f59e0b"

    return f"""
    <div style="
        text-align: center;
        padding: 22px 20px;
        border-radius: 16px;
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 1px solid rgba(99, 102, 241, 0.15);
    ">
        <div style="
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: #64748b;
            font-weight: 700;
            margin-bottom: 10px;
        ">Confidence</div>
        <div style="
            font-size: 40px;
            font-weight: 800;
            color: {color};
            line-height: 1;
        ">{pct_text}</div>
        <div style="
            margin-top: 16px;
            height: 8px;
            border-radius: 99px;
            background: #1e293b;
            overflow: hidden;
        ">
            <div style="
                width: {bar_width};
                height: 100%;
                border-radius: 99px;
                background: linear-gradient(90deg, {color}, {color}88);
                transition: width 0.6s ease;
            "></div>
        </div>
    </div>
    """


def create_empty_chart() -> plt.Figure:
    """Return a blank dark chart as placeholder."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    ax.text(
        0.5, 0.5,
        "Upload an image and click Predict",
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=14, color="#475569",
        fontfamily="sans-serif",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Global overrides ───────────────────────────────────── */
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

/* ── Hero banner ────────────────────────────────────────── */
#hero-banner {
    text-align: center;
    padding: 36px 20px 24px;
    border-radius: 20px;
    background: linear-gradient(160deg, #1e1b4b 0%, #0f172a 50%, #0c1222 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    margin-bottom: 8px;
    box-shadow: 0 4px 40px rgba(99, 102, 241, 0.06);
}

/* ── Section cards ──────────────────────────────────────── */
.result-card {
    border-radius: 16px !important;
    overflow: hidden;
}

/* ── Predict button ─────────────────────────────────────── */
#predict-btn {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 14px 36px !important;
    box-shadow: 0 4px 24px rgba(99, 102, 241, 0.35) !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
    text-transform: uppercase !important;
}
#predict-btn:hover {
    box-shadow: 0 6px 32px rgba(99, 102, 241, 0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── Footer ─────────────────────────────────────────────── */
#app-footer {
    text-align: center;
    padding: 16px;
    font-size: 12px;
    color: #475569;
}

/* ── Image upload area ──────────────────────────────────── */
#image-upload {
    border: 2px dashed rgba(99, 102, 241, 0.3) !important;
    border-radius: 16px !important;
    min-height: 320px !important;
    transition: border-color 0.3s ease !important;
}
#image-upload:hover {
    border-color: rgba(99, 102, 241, 0.6) !important;
}

/* ── Chart container ────────────────────────────────────── */
#chart-output {
    border-radius: 16px !important;
    overflow: hidden;
}
"""

# ──────────────────────────────────────────────────────────────
# JavaScript to force dark theme on load
# ──────────────────────────────────────────────────────────────

FORCE_DARK_JS = """
function() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

# ──────────────────────────────────────────────────────────────
# Build the Gradio Interface
# ──────────────────────────────────────────────────────────────

# Load the model once at startup
print("[*]  Loading model ...")
model = load_model()

# Choose a dark-tinted base theme
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
).set(
    # Dark background tones
    body_background_fill="#0b0f1a",
    body_background_fill_dark="#0b0f1a",
    block_background_fill="#111827",
    block_background_fill_dark="#111827",
    block_border_color="rgba(99,102,241,0.12)",
    block_border_color_dark="rgba(99,102,241,0.12)",
    block_label_text_color="#94a3b8",
    block_label_text_color_dark="#94a3b8",
    block_title_text_color="#e2e8f0",
    block_title_text_color_dark="#e2e8f0",
    input_background_fill="#1e293b",
    input_background_fill_dark="#1e293b",
    button_primary_background_fill="linear-gradient(135deg, #6366f1, #8b5cf6)",
    button_primary_background_fill_dark="linear-gradient(135deg, #6366f1, #8b5cf6)",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    border_color_primary="rgba(99,102,241,0.25)",
    border_color_primary_dark="rgba(99,102,241,0.25)",
)

# ── Build the Blocks layout ──────────────────────────────────
with gr.Blocks(
    title="AI Vehicle Body Type Classifier",
) as demo:

    # Pre-compute the device label for the badge
    _device_label = "GPU" if DEVICE.type == "cuda" else "CPU"

    # ── Hero Banner ──
    gr.HTML(
        f"""
        <div id="hero-banner">
            <div style="font-size: 14px; letter-spacing: 3px; text-transform: uppercase;
                        color: #6366f1; font-weight: 700; margin-bottom: 8px;">
                Deep Learning &middot; Computer Vision
            </div>
            <h1 style="
                font-size: 40px; font-weight: 900; margin: 0;
                background: linear-gradient(90deg, #c7d2fe, #818cf8, #22d3ee);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                line-height: 1.2;
            ">
                &#128663; AI Vehicle Body Type Classifier
            </h1>
            <p style="color: #94a3b8; font-size: 16px; margin-top: 12px; max-width: 640px;
                      margin-left: auto; margin-right: auto; line-height: 1.6;">
                Upload a vehicle image and let our <strong style="color:#818cf8;">EfficientNet-B0</strong>
                deep learning model identify the body type across
                <strong style="color:#22d3ee;">8 distinct classes</strong> with real-time
                confidence analysis.
            </p>
            <div style="margin-top: 16px; display: flex; gap: 12px; justify-content: center;
                        flex-wrap: wrap;">
                <span style="background: #1e293b; color: #94a3b8; padding: 5px 14px;
                             border-radius: 99px; font-size: 12px; font-weight: 600;
                             border: 1px solid rgba(99,102,241,0.15);">
                    &#129504; EfficientNet-B0
                </span>
                <span style="background: #1e293b; color: #94a3b8; padding: 5px 14px;
                             border-radius: 99px; font-size: 12px; font-weight: 600;
                             border: 1px solid rgba(99,102,241,0.15);">
                    &#127919; 8 Classes
                </span>
                <span style="background: #1e293b; color: #94a3b8; padding: 5px 14px;
                             border-radius: 99px; font-size: 12px; font-weight: 600;
                             border: 1px solid rgba(99,102,241,0.15);">
                    &#9889; {_device_label} Accelerated
                </span>
            </div>
        </div>
        """,
    )

    # ── Main two-column layout ──
    with gr.Row(equal_height=False):

        # ── Left column: upload + predict ──
        with gr.Column(scale=1):
            gr.Markdown(
                "### 📤  Upload Vehicle Image",
                elem_classes=["section-title"],
            )
            image_input = gr.Image(
                type="pil",
                label="Drag & drop or click to upload",
                elem_id="image-upload",
                height=360,
            )
            predict_btn = gr.Button(
                "🔍  Predict Vehicle Type",
                variant="primary",
                elem_id="predict-btn",
            )

        # ── Right column: results ──
        with gr.Column(scale=1):
            gr.Markdown("### 📊  Prediction Results")
            label_output = gr.HTML(
                value=format_label_html("—", "Awaiting prediction …"),
                elem_classes=["result-card"],
            )
            conf_output = gr.HTML(
                value=format_confidence_html(None),
                elem_classes=["result-card"],
            )

    # ── Full-width chart row ──
    gr.Markdown("### 📈  Class Probability Distribution")
    chart_output = gr.Plot(
        value=create_empty_chart(),
        elem_id="chart-output",
    )

    # ── Wire up the predict button ──
    predict_btn.click(
        fn=on_predict,
        inputs=[image_input],
        outputs=[label_output, conf_output, chart_output],
    )

    # ── Footer ──
    gr.HTML(
        """
        <div id="app-footer">
            Built with ❤️ using <strong>PyTorch</strong>, <strong>timm</strong>,
            and <strong>Gradio</strong> · EfficientNet-B0 · 8 Vehicle Classes
        </div>
        """
    )

# ──────────────────────────────────────────────────────────────
# Launch
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(theme=theme, css=CUSTOM_CSS, js=FORCE_DARK_JS)
