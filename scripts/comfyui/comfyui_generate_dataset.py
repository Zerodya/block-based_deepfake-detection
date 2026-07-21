#!/usr/bin/env python3
"""
ComfyUI Image Generation + Real-World Augmentation Pipeline
Generates 5000 diverse images via ComfyUI API with automatic augmentations.

Requirements:
    pip install requests Pillow numpy websocket-client

Setup:
    1. Start ComfyUI server: python main.py
    2. Place your model in models/checkpoints/
    3. Run: python comfyui_generate_dataset.py
"""

import os
import io
import json
import random
import uuid
import requests
import time
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
import numpy as np
import websocket

# ========================= CONFIGURATION =========================
COMFYUI_URL = "http://127.0.0.1:8188"
WS_URL = "ws://127.0.0.1:8188/ws"
CHECKPOINT_NAME = "sd_xl_base_1.0.safetensors"
CLIENT_ID = str(uuid.uuid4())

# Output directories
DATASET_DIR = Path("dataset")
GENERATED_DIR = DATASET_DIR / "generated"
AUGMENTED_DIR = DATASET_DIR / "augmented"
METADATA_PATH = DATASET_DIR / "metadata.jsonl"

# How many base images to generate
NUM_BASE_IMAGES = 5000

# Image dimensions
WIDTH = 1024
HEIGHT = 1024

# ComfyUI KSampler settings
SAMPLER_STEPS = 30
SAMPLER_CFG = 7.5
SAMPLER_NAME = "euler_ancestral"
SAMPLER_SCHEDULER = "normal"

# Augmentation probabilities (0.0 - 1.0)
AUGMENTATION_CONFIG = {
    "random_crop": 0.7,
    "resize_distortion": 0.6,
    "jpeg_compression": 0.8,
    "meme_text": 0.4,
    "social_media_caption": 0.3,
    "brightness_contrast": 0.7,
    "blur": 0.5,
    "noise": 0.4,
    "rotation": 0.3,
    "watermark": 0.3,
    "color_tint": 0.4,
    "vignette": 0.2,
    "overlay_emoji_sticker": 0.2,
    "grayscale": 0.15,
    "saturation_boost": 0.25,
}

# ========================= DIVERSE PROMPTS =========================
PROMPTS = [
    "close-up portrait of an elderly fisherman with weathered skin and silver beard, golden hour lighting, shallow depth of field",
    "macro shot of a bumblebee on a lavender flower, morning dew drops, bokeh background",
    "overhead flat lay of a rustic breakfast spread, eggs benedict, coffee, newspaper, wooden table",
    "panoramic shot of a fjord in Norway, dramatic cliffs, mirror-like water, overcast sky",
    "worm's eye view of a modern glass skyscraper reflecting clouds, geometric patterns",
    "vintage red convertible on a coastal highway, ocean in background, golden hour",
    "minimalist shot of a single ceramic vase on a wooden shelf, soft window light",
    "candid shot of a bride laughing during wedding speeches, emotional, soft venue lighting",
    "extreme macro of a dragonfly's compound eyes, iridescent colors, black background",
    "long exposure of car light trails on a highway overpass, city skyline, blue hour",
    "studio headshot of a young professional woman with natural makeup, softbox lighting",
    "wildlife photograph of a red fox in a snowy forest, side profile, steam from breath",
    "close-up of a chef's hands plating a gourmet dish, tweezers placing microgreens",
    "long exposure of a waterfall in a tropical rainforest, silky water, lush green moss",
    "interior shot of a grand cathedral, vaulted ceilings, stained glass windows",
    "close-up of a motorcycle engine with chrome details, garage lighting",
    "overhead shot of a messy artist's desk, paint tubes, brushes, palette",
    "rock concert crowd with hands raised, stage lights, confetti, motion blur",
    "close-up of a mushroom gills with spores falling, forest floor",
    "night market scene with steam from food stalls, neon signs, crowd motion blur",
    "environmental portrait of a tattooed barista working in a specialty coffee shop",
    "underwater shot of a sea turtle swimming over coral reef, sun rays penetrating water",
    "dark and moody shot of a chocolate lava cake with molten center",
    "aerial drone shot of terraced rice paddies in Bali, geometric patterns",
    "abandoned industrial warehouse with broken windows, graffiti, shafts of light",
    "aerial shot of a container ship at sea, geometric containers, wake pattern",
    "vintage still life of antique pocket watches and old books, warm tungsten lighting",
    "quiet moment of a monk reading in an ancient library, candlelight",
    "texture shot of tree bark with moss and lichen, rough patterns",
    "astrophotography of the milky way over a mountain lake, reflection",
    "candid portrait of a laughing child at a birthday party, colorful balloons",
    "action shot of a border collie catching a frisbee mid-air, frozen motion",
    "bright and airy shot of a colorful acai bowl, fresh berries, granola",
    "starry night sky over a desert landscape, milky way visible",
    "minimalist Japanese house interior, tatami mats, shoji screens",
    "night shot of a classic muscle car at a drive-in diner, neon signs",
    "modern product shot of wireless earbuds on concrete, dramatic side lighting",
    "children playing in a sprinkler on a hot summer day, water droplets",
    "macro of a water droplet on a spider web, refracted background",
    "fireworks display over a city harbor, colorful explosions",
    "dramatic portrait of a ballet dancer mid-pose, chiaroscuro lighting",
    "close-up of a great horned owl's face, piercing yellow eyes",
    "street food shot of a taco stand at night, sizzling meat, neon signs",
    "autumn forest scene with a winding dirt road, golden and red leaves",
    "night shot of a neon-lit Tokyo alley, vending machines, power lines",
    "rustic shot of gardening tools on a potting bench, soil, seedlings",
    "intimate moment of a couple dancing in their kitchen, evening light",
    "close-up of a fern unfurling, Fibonacci spiral, vibrant green",
    "night shot of a campfire with friends, sparks flying, warm glow",
    "street portrait of a bearded man wearing a flat cap, rainy day",
    "aerial view of a herd of elephants crossing a dry riverbed",
    "minimalist shot of a single perfect apple on white marble",
    "colorful shot of a Moroccan riad courtyard, intricate tilework",
    "rustic shot of an old tractor in a wheat field, sunset backlight",
    "organized shot of a spice rack with colorful jars, kitchen background",
    "protest march with raised fists and signs, dramatic sky",
    "texture shot of cracked dry earth, drought conditions",
    "city street at night during rain, neon reflections on wet asphalt",
    "backlit portrait of a silhouette against a sunset, warm orange sky",
    "serene shot of a Japanese zen garden, raked gravel patterns",
    "action shot of a mountain biker mid-jump, forest background",
    "cozy shot of a reading setup, open book, reading glasses, tea cup",
    "macro of frost crystals on a leaf, intricate patterns, blue-white tones",
    "bioluminescent plankton on a beach, blue glow in waves",
    "high-key portrait of a newborn baby wrapped in white fabric",
    "surreal shot of the northern lights over an Icelandic glacier",
    "brutalist concrete government building, imposing scale",
    "luxury yacht deck at sunset, champagne glasses, Mediterranean sea",
    "technical shot of a mechanical watch movement, gears visible",
    "sports action shot of a soccer goal celebration, rain, mud",
    "close-up of a pine cone with sap droplets, warm autumn light",
    "concert stage with laser lights and fog, silhouetted crowd",
    "documentary portrait of a construction worker taking a break",
    "golden hour shot of rolling Tuscan hills, cypress trees",
    "close-up of a vinyl record on a turntable, tonearm",
    "street festival with colorful powder in the air, Holi celebration",
    "abstract shot of ocean waves from above, turquoise and white",
    "night shot of a lighthouse beam cutting through fog, rocky coast",
    "fashion portrait of a model with avant-garde makeup, neon lighting",
    "underwater landscape of a kelp forest, sunbeams filtering through",
    "chaotic shot of a family dinner with food flying, laughing faces",
    "macro of a butterfly wing scales, structural coloration",
    "solemn shot of a veteran at a memorial, medals, American flag",
    "close-up of Art Deco building facade, geometric gold details",
    "interior of a cozy bookstore, floor-to-ceiling shelves",
    "abandoned airplane in a desert, sand dunes, rusted fuselage",
    "dramatic shot of a peacock with feathers fully fanned",
    "underwater macro of a seahorse clinging to seaweed",
    "night shot of a raccoon rummaging through a suburban trash can",
    "overhead shot of a messy pizza party table, half-eaten slices",
    "aerial view of a spiral parking garage, concrete curves",
    "macro shot of coffee beans being ground, fine particles flying",
    "urban exploration of an abandoned hospital, flashlight beam",
    "rustic shot of a sourdough loaf being torn apart, steam escaping",
    "vibrant shot of a farmers market vegetable stall, rainbow chard",
    "close-up of a steam locomotive emerging from a tunnel, smoke billowing",
    "intimate shot of a mother cat grooming her kittens, soft indoor lighting",
    "city street at night during rain, lone umbrella, film noir atmosphere",
]

NEGATIVE_PROMPT = "blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, signature, text, logo, cartoon, anime, illustration, painting, drawing, sketch, 3d render, cgi, plastic, doll, oversaturated, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, extra limbs, extra arms, extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed, polar lowres, bad face"

# Meme text options
MEME_TOP_TEXTS = ["WHEN YOU", "NO ONE:", "ME:", "POV:", "MY FACE WHEN", "THAT MOMENT", "EVERYONE:", "WAIT...", "LITERALLY ME"]
MEME_BOTTOM_TEXTS = ["REALIZE IT'S MONDAY", "SEE THE BILL", "CHECK THE TIME", "FINALLY UNDERSTAND", "FORGET TO SAVE", "SEE THE EMAIL", "TRY TO ADULT", "REMEMBER THE DEADLINE", "OPEN THE FRIDGE"]
CAPTIONS = ["Living my best life ✨", "No filter needed", "Mood", "Vibes", "Just another Tuesday", "Couldn't resist posting this", "Thoughts?", "Caught this moment", "Unreal view", "This happened today"]
WATERMARKS = ["@user123", "photography_by_me", "shot_on_iphone", "no copyright", "insta_dump", "vsco", "tumblr_2024", "my_aesthetic"]
EMOJI_STICKERS = ["🔥", "✨", "😂", "❤️", "👀", "🙌", "💯", "🤔", "👍", "🎉"]

# ========================= COMFYUI WORKFLOW =========================

def build_workflow():
    """Build a minimal ComfyUI workflow JSON for standard txt2img."""
    return {
        "1": {
            "inputs": {"ckpt_name": CHECKPOINT_NAME},
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Checkpoint"}
        },
        "2": {
            "inputs": {
                "text": "",
                "clip": ["1", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive Prompt"}
        },
        "3": {
            "inputs": {
                "text": NEGATIVE_PROMPT,
                "clip": ["1", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Negative Prompt"}
        },
        "4": {
            "inputs": {
                "seed": 0,
                "steps": SAMPLER_STEPS,
                "cfg": SAMPLER_CFG,
                "sampler_name": SAMPLER_NAME,
                "scheduler": SAMPLER_SCHEDULER,
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"}
        },
        "5": {
            "inputs": {
                "width": WIDTH,
                "height": HEIGHT,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage",
            "_meta": {"title": "Empty Latent Image"}
        },
        "6": {
            "inputs": {
                "samples": ["4", 0],
                "vae": ["1", 2]
            },
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"}
        },
        "7": {
            "inputs": {
                "filename_prefix": "dataset_gen",
                "images": ["6", 0]
            },
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"}
        }
    }

def patch_workflow(workflow, positive_prompt, seed):
    """Patch a workflow with new prompt and seed."""
    wf = json.loads(json.dumps(workflow))

    prompt_patched = False
    seed_patched = False

    for node_id, node in wf.items():
        if node.get("class_type") == "CLIPTextEncode":
            text = node.get("inputs", {}).get("text", "")
            if isinstance(text, str) and any(x in text.lower() for x in ["negative", "bad", "ugly", "blurry", "low quality"]):
                continue
            if not prompt_patched:
                node["inputs"]["text"] = positive_prompt
                prompt_patched = True

        if node.get("class_type") == "KSampler":
            if not seed_patched:
                node["inputs"]["seed"] = seed
                seed_patched = True

    if not prompt_patched:
        for node in wf.values():
            if node.get("class_type") == "CLIPTextEncode":
                node["inputs"]["text"] = positive_prompt
                break
    if not seed_patched:
        for node in wf.values():
            if node.get("class_type") == "KSampler":
                node["inputs"]["seed"] = seed
                break
    return wf

# ========================= COMFYUI API CLIENT =========================

def connect_websocket():
    """Connect to ComfyUI WebSocket for tracking."""
    ws = websocket.WebSocket()
    ws.connect(f"{WS_URL}?clientId={CLIENT_ID}")
    return ws

def queue_prompt(workflow):
    """Submit workflow to ComfyUI with proper client_id."""
    payload = {
        "prompt": workflow,
        "client_id": CLIENT_ID
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(f"{COMFYUI_URL}/prompt", json=payload, headers=headers)

    if resp.status_code != 200:
        print(f"[ERROR] Status {resp.status_code}: {resp.text}")
        raise Exception(f"Failed to queue prompt: {resp.status_code}")

    result = resp.json()
    if result.get("node_errors"):
        print(f"[ERROR] Node errors: {result['node_errors']}")
        raise Exception(f"Workflow validation failed")

    return result["prompt_id"]

def wait_for_completion(ws, prompt_id, timeout=300):
    """Wait for workflow completion via WebSocket."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            msg = ws.recv()
            if isinstance(msg, str):
                data = json.loads(msg)
                if data.get("type") == "executing":
                    d = data.get("data", {})
                    if d.get("node") is None and d.get("prompt_id") == prompt_id:
                        return True
                elif data.get("type") == "execution_error":
                    print(f"[ERROR] Execution error: {data}")
                    return False
        except websocket.WebSocketTimeoutException:
            continue
        except Exception as e:
            print(f"[WARN] WebSocket error: {e}")
            continue
    return False

def get_history(prompt_id):
    """Fetch history for a given prompt_id."""
    resp = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
    resp.raise_for_status()
    return resp.json()

def get_image(filename, subfolder="", folder_type="output"):
    """Download a generated image from ComfyUI."""
    params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    resp = requests.get(f"{COMFYUI_URL}/view", params=params)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")

def generate_image(prompt, idx, workflow_template, ws):
    """Generate an image via ComfyUI and return PIL Image."""
    print(f"[GEN {idx:04d}/5000] {prompt[:60]}...")
    seed = random.randint(1, 2**32 - 1)
    workflow = patch_workflow(workflow_template, prompt, seed)

    try:
        prompt_id = queue_prompt(workflow)
    except Exception as e:
        print(f"[ERROR] Failed to queue prompt: {e}")
        return None

    completed = wait_for_completion(ws, prompt_id)
    if not completed:
        print(f"[ERROR] Generation failed or timed out")
        return None

    try:
        history = get_history(prompt_id)
        if prompt_id not in history:
            print(f"[WARN] No history found for {prompt_id}")
            return None

        outputs = history[prompt_id].get("outputs", {})
        images = []
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img_info in node_output["images"]:
                    img = get_image(
                        img_info["filename"],
                        img_info.get("subfolder", ""),
                        img_info.get("type", "output")
                    )
                    images.append(img)
        if images:
            return images[0]
        print(f"[WARN] No images found in outputs")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to fetch image: {e}")
        return None

# ========================= AUGMENTATIONS (FIXED) =========================

def get_font(size=30):
    font_paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "arialbd.ttf",
        "/Windows/Fonts/arialbd.ttf",
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()

def apply_random_crop(img):
    w, h = img.size
    crop_ratio = random.uniform(0.6, 0.9)
    new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    return img.crop((left, top, left + new_w, top + new_h))

def apply_resize_distortion(img):
    w, h = img.size
    scale = random.uniform(0.3, 0.7)
    small = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
    return small.resize((w, h), Image.Resampling.NEAREST)

def apply_jpeg_compression(img):
    quality = random.randint(30, 75)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=False)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def add_meme_text(img):
    draw = ImageDraw.Draw(img)
    font = get_font(size=int(min(img.size) * 0.08))
    top_text = random.choice(MEME_TOP_TEXTS).upper()
    bottom_text = random.choice(MEME_BOTTOM_TEXTS).upper()

    def draw_outlined_text(draw, pos, text, font, fill="white", outline="black"):
        x, y = pos
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if dx == 0 and dy == 0:
                    continue
                draw.text((x+dx, y+dy), text, font=font, fill=outline)
        draw.text((x, y), text, font=font, fill=fill)

    bbox = draw.textbbox((0, 0), top_text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw_outlined_text(draw, ((img.width - tw)//2, int(img.height*0.02)), top_text, font)

    bbox = draw.textbbox((0, 0), bottom_text, font=font)
    tw = bbox[2] - bbox[0]
    draw_outlined_text(draw, ((img.width - tw)//2, img.height - th - int(img.height*0.04)), bottom_text, font)
    return img

def add_social_media_caption(img):
    draw = ImageDraw.Draw(img)
    caption = random.choice(CAPTIONS)
    font = get_font(size=int(min(img.size) * 0.045))
    bbox = draw.textbbox((0, 0), caption, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = random.randint(10, max(10, img.width - tw - 10))
    y = random.randint(10, max(10, img.height - th - 10))
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    padding = 6
    overlay_draw.rectangle([x-padding, y-padding, x+tw+padding, y+th+padding], fill=(0,0,0,120))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.text((x, y), caption, fill=(255, 255, 255), font=font)
    return img

def apply_brightness_contrast(img):
    factor_b = random.uniform(0.7, 1.4)
    factor_c = random.uniform(0.7, 1.5)
    img = ImageEnhance.Brightness(img).enhance(factor_b)
    img = ImageEnhance.Contrast(img).enhance(factor_c)
    return img

def apply_blur(img):
    r = random.random()
    if r < 0.5:
        radius = random.uniform(0.5, 2.5)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    else:
        radius = random.randint(2, 5)
        return img.filter(ImageFilter.BoxBlur(radius))

def apply_noise(img):
    np_img = np.array(img).astype(np.float32)
    noise = np.random.normal(0, random.uniform(5, 20), np_img.shape)
    np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)

def apply_rotation(img):
    angle = random.uniform(-15, 15)
    return img.rotate(angle, expand=True, fillcolor=(random.randint(0,255),)*3)

def add_watermark(img):
    draw = ImageDraw.Draw(img)
    text = random.choice(WATERMARKS)
    font = get_font(size=int(min(img.size) * 0.035))
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = img.width - tw - random.randint(5, 30)
    y = img.height - th - random.randint(5, 30)
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.text((x, y), text, fill=(255, 255, 255, 90), font=font)
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return img

def apply_color_tint(img):
    """Apply color tint - FIXED to always work with RGB images."""
    r = random.random()
    if r < 0.33:
        # Sepia - apply to RGB directly
        np_img = np.array(img).astype(np.float32)
        sepia = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        np_img = np.clip(np_img @ sepia.T, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)
    elif r < 0.66:
        # Warm tint
        r_chan, g_chan, b_chan = img.split()
        r_chan = r_chan.point(lambda i: min(255, int(i * 1.15)))
        b_chan = b_chan.point(lambda i: int(i * 0.85))
        return Image.merge("RGB", (r_chan, g_chan, b_chan))
    else:
        # Cool tint
        r_chan, g_chan, b_chan = img.split()
        b_chan = b_chan.point(lambda i: min(255, int(i * 1.15)))
        r_chan = r_chan.point(lambda i: int(i * 0.9))
        return Image.merge("RGB", (r_chan, g_chan, b_chan))

def apply_vignette(img):
    w, h = img.size
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    strength = random.uniform(0.3, 0.8)
    mask = 1 - np.clip(R * strength, 0, 1)
    mask = (mask * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask).convert("L")
    dark = ImageEnhance.Brightness(img).enhance(0.4)
    return Image.composite(img, dark, mask_img)

def add_overlay_emoji_sticker(img):
    draw = ImageDraw.Draw(img)
    emoji = random.choice(EMOJI_STICKERS)
    font = get_font(size=int(min(img.size) * 0.15))
    bbox = draw.textbbox((0, 0), emoji, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = random.randint(0, max(0, img.width - tw))
    y = random.randint(0, max(0, img.height - th))
    draw.text((x, y), emoji, font=font)
    return img

def apply_grayscale(img):
    return ImageOps.grayscale(img).convert("RGB")

def apply_saturation_boost(img):
    factor = random.uniform(1.3, 2.0)
    return ImageEnhance.Color(img).enhance(factor)

def augment_image(img):
    """Apply a random subset of augmentations to simulate real-world sharing.
    FIXED: Always resize to target size after rotation to prevent dimension mismatches.
    """
    applied = []
    target_size = (WIDTH, HEIGHT)

    # Start with resize to ensure consistent dimensions
    img = img.resize(target_size, Image.Resampling.LANCZOS)

    if random.random() < AUGMENTATION_CONFIG["random_crop"]:
        img = apply_random_crop(img)
        applied.append("random_crop")

    if random.random() < AUGMENTATION_CONFIG["resize_distortion"]:
        img = apply_resize_distortion(img)
        applied.append("resize_distortion")

    if random.random() < AUGMENTATION_CONFIG["rotation"]:
        img = apply_rotation(img)
        applied.append("rotation")
        # CRITICAL FIX: Resize back to target after rotation to prevent dimension mismatch
        img = img.resize(target_size, Image.Resampling.LANCZOS)

    if random.random() < AUGMENTATION_CONFIG["brightness_contrast"]:
        img = apply_brightness_contrast(img)
        applied.append("brightness_contrast")

    if random.random() < AUGMENTATION_CONFIG["blur"]:
        img = apply_blur(img)
        applied.append("blur")

    if random.random() < AUGMENTATION_CONFIG["noise"]:
        img = apply_noise(img)
        applied.append("noise")

    if random.random() < AUGMENTATION_CONFIG["color_tint"]:
        img = apply_color_tint(img)
        applied.append("color_tint")

    if random.random() < AUGMENTATION_CONFIG["saturation_boost"]:
        img = apply_saturation_boost(img)
        applied.append("saturation_boost")

    if random.random() < AUGMENTATION_CONFIG["grayscale"]:
        img = apply_grayscale(img)
        applied.append("grayscale")

    if random.random() < AUGMENTATION_CONFIG["vignette"]:
        img = apply_vignette(img)
        applied.append("vignette")

    if random.random() < AUGMENTATION_CONFIG["watermark"]:
        img = add_watermark(img)
        applied.append("watermark")

    if random.random() < AUGMENTATION_CONFIG["overlay_emoji_sticker"]:
        img = add_overlay_emoji_sticker(img)
        applied.append("overlay_emoji_sticker")

    if random.random() < AUGMENTATION_CONFIG["meme_text"]:
        img = add_meme_text(img)
        applied.append("meme_text")

    if random.random() < AUGMENTATION_CONFIG["social_media_caption"]:
        img = add_social_media_caption(img)
        applied.append("social_media_caption")

    if random.random() < AUGMENTATION_CONFIG["jpeg_compression"]:
        img = apply_jpeg_compression(img)
        applied.append("jpeg_compression")

    # Final resize to ensure consistent output dimensions
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    return img, applied

# ========================= MAIN =========================

def ensure_dirs():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)

def main():
    ensure_dirs()

    # Verify ComfyUI is reachable
    try:
        resp = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        resp.raise_for_status()
        print(f"[INFO] Connected to ComfyUI at {COMFYUI_URL}")
    except Exception as e:
        print(f"[ERROR] Cannot connect to ComfyUI at {COMFYUI_URL}: {e}")
        print("[ERROR] Please start ComfyUI first: python main.py")
        return

    # Connect WebSocket
    print(f"[INFO] Connecting WebSocket with client_id: {CLIENT_ID}")
    try:
        ws = connect_websocket()
        print("[INFO] WebSocket connected")
    except Exception as e:
        print(f"[ERROR] WebSocket connection failed: {e}")
        return

    workflow_template = build_workflow()
    metadata = []
    shuffled_prompts = PROMPTS.copy()
    random.shuffle(shuffled_prompts)

    for i in range(NUM_BASE_IMAGES):
        prompt = shuffled_prompts[i % len(shuffled_prompts)]
        if i > 0 and i % len(shuffled_prompts) == 0:
            random.shuffle(shuffled_prompts)

        base_img = generate_image(prompt, i, workflow_template, ws)
        if base_img is None:
            continue

        base_name = f"gen_{i:04d}"
        base_path = GENERATED_DIR / f"{base_name}_base.png"
        base_img.save(base_path)

        num_variants = random.randint(2, 4)
        for v in range(num_variants):
            aug_img, applied = augment_image(base_img.copy())
            aug_name = f"{base_name}_aug{v}.jpg"
            aug_path = AUGMENTED_DIR / aug_name
            aug_img.save(aug_path, quality=random.randint(75, 95))

            metadata.append({
                "base_image": str(base_path),
                "augmented_image": str(aug_path),
                "prompt": prompt,
                "label": "generated",
                "augmentations": applied,
                "variant": v,
                "timestamp": datetime.now().isoformat(),
            })

        print(f"[OK] {base_name}: {num_variants} variants created")

    ws.close()

    with open(METADATA_PATH, "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")

    print(f"\n[DONE] Dataset created:")
    print(f"  Base images: {len(list(GENERATED_DIR.glob('*.png')))}")
    print(f"  Augmented images: {len(list(AUGMENTED_DIR.glob('*.jpg')))}")
    print(f"  Metadata: {METADATA_PATH}")

if __name__ == "__main__":
    main()
