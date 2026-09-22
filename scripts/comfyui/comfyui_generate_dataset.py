#!/usr/bin/env python3
import io
import json
import random
import uuid
import requests
import time
from pathlib import Path
from datetime import datetime
from PIL import Image
import websocket
import sys

# Same encoder the real half of the dataset goes through, so compression
# history, crop policy and output format cannot separate the two classes.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from dfx.image_prep import process_image

# ========================= CONFIGURATION =========================
COMFYUI_URL = "http://127.0.0.1:8188"
WS_URL = "ws://127.0.0.1:8188/ws"
CHECKPOINT_NAME = "sd_xl_base_1.0.safetensors"
CLIENT_ID = str(uuid.uuid4())

# Output directories
DATASET_DIR = Path("dataset")
GENERATED_DIR = DATASET_DIR / "generated"
METADATA_PATH = DATASET_DIR / "metadata.jsonl"

# How many images to generate
NUM_IMAGES = 5000

# Image dimensions. Defaults for the workflow template; each image draws its
# own values from RESOLUTIONS below.
WIDTH = 1024
HEIGHT = 1024

# ComfyUI KSampler defaults for the workflow template.
SAMPLER_STEPS = 30
SAMPLER_CFG = 7.5
SAMPLER_NAME = "euler_ancestral"
SAMPLER_SCHEDULER = "normal"

# ---------------------------------------------------------------------------
# Generation diversity
# ---------------------------------------------------------------------------
# Every image used to be produced with exactly the same sampler, step count, CFG
# and resolution. That gives the whole generated half of the dataset one shared
# low-level signature, which a detector can pick up instead of learning anything
# about diffusion artifacts in general - and which will not transfer to images
# from any other generator. Sampling these per image widens the distribution the
# detector has to cover.
RESOLUTIONS = [(1024, 1024), (1152, 896), (896, 1152), (1216, 832), (832, 1216)]
STEP_CHOICES = [20, 25, 30, 35, 40, 50]
CFG_CHOICES = [5.0, 6.0, 7.0, 7.5, 8.0, 9.0]
SAMPLER_CHOICES = ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "ddim"]
SCHEDULER_CHOICES = ["normal", "karras", "exponential"]

# Output encoding. MUST match what scripts/prep/prepare_real_dataset.py writes
# for the real half, or the two classes become separable by compression history
# alone. Both go through dfx.image_prep.process_image.
OUTPUT_SIZE = 1024
OUTPUT_QUALITY = 95
OUTPUT_FORMAT = "JPEG"
# Randomised prior JPEG generation, applied to the generated half as well.
# Real photographs arrive already compressed and renders do not, and that
# difference alone is enough to separate the classes. Must match the setting
# used by scripts/prep/prepare_real_dataset.py.
OUTPUT_JPEG_HISTORY = True

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

def sample_settings(rng):
    """Draw the per-image generation settings."""
    w, h = rng.choice(RESOLUTIONS)
    return {
        "width": w,
        "height": h,
        "steps": rng.choice(STEP_CHOICES),
        "cfg": rng.choice(CFG_CHOICES),
        "sampler_name": rng.choice(SAMPLER_CHOICES),
        "scheduler": rng.choice(SCHEDULER_CHOICES),
    }


def patch_workflow(workflow, positive_prompt, seed, settings=None):
    """Patch a workflow with a new prompt, seed and per-image settings."""
    wf = json.loads(json.dumps(workflow))
    settings = settings or {}

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
            for key in ("steps", "cfg", "sampler_name", "scheduler"):
                if key in settings:
                    node["inputs"][key] = settings[key]

        if node.get("class_type") == "EmptyLatentImage":
            for key in ("width", "height"):
                if key in settings:
                    node["inputs"][key] = settings[key]

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

def generate_image(prompt, idx, workflow_template, ws, settings=None, seed=None):
    """Generate an image via ComfyUI and return (PIL Image, settings, seed)."""
    print(f"[GEN {idx:04d}/{NUM_IMAGES}] {prompt[:60]}...")
    seed = random.randint(1, 2**32 - 1) if seed is None else seed
    workflow = patch_workflow(workflow_template, prompt, seed, settings)

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

# ========================= MAIN =========================

def ensure_dirs():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

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

    rng = random.Random()

    for i in range(NUM_IMAGES):
        # Prompt chosen purely at random (with replacement) for each image
        prompt = random.choice(PROMPTS)
        settings = sample_settings(rng)
        seed = rng.randint(1, 2**32 - 1)

        img = generate_image(prompt, i, workflow_template, ws, settings, seed)
        if img is None:
            continue

        # Written through the SAME encoder as the real half. Saving a lossless
        # PNG here is what previously made every generated image trivially
        # distinguishable from every (JPEG-sourced) real one.
        ext = 'jpg' if OUTPUT_FORMAT.upper() == 'JPEG' else OUTPUT_FORMAT.lower()
        img_name = f"gen_{i:04d}.{ext}"
        img_path = GENERATED_DIR / img_name
        process_image(img, img_path, size=OUTPUT_SIZE, crop_mode='center',
                      quality=OUTPUT_QUALITY, image_format=OUTPUT_FORMAT, rng=rng,
                      jpeg_history=OUTPUT_JPEG_HISTORY)

        metadata.append({
            "image": str(img_path),
            "prompt": prompt,
            "label": "generated",
            "seed_prompt_source": "random",
            "seed": seed,
            "settings": settings,
            "output_format": OUTPUT_FORMAT,
            "output_quality": OUTPUT_QUALITY,
            "jpeg_history": OUTPUT_JPEG_HISTORY,
            "timestamp": datetime.now().isoformat(),
        })

        print(f"[OK] {img_name} saved  "
              f"({settings['width']}x{settings['height']}, {settings['steps']} steps, "
              f"cfg {settings['cfg']}, {settings['sampler_name']})")

    ws.close()

    with open(METADATA_PATH, "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")

    print(f"\n[DONE] Dataset created:")
    print(f"  Images: {len(list(GENERATED_DIR.glob('*.*')))}")
    print(f"  Metadata: {METADATA_PATH}")

if __name__ == "__main__":
    main()
