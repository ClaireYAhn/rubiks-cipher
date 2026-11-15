"""cube_app.py - Flask app: 3 words -> scrambled cube (two PNG renders) -> 3 words.

Two separate sites, like the fractal version (sender and receiver on different computers):
  ../.venv/bin/python cube_encode_app.py [words.csv] [port]   -> http://127.0.0.1:8771  (Encode only)
  ../.venv/bin/python cube_decode_app.py [words.csv] [port]   -> http://127.0.0.1:8772  (Decode only)
Development: ../.venv/bin/python cube_app.py                  -> http://127.0.0.1:8770  (both tabs)

  POST /encode  {"words": [w1, w2, w3]}        -> facelets, scramble, bits        (encode site)
  POST /decode  multipart, two image files     -> words, facelets, solve          (decode site)
"""
import threading
import webbrowser

from flask import Flask, jsonify, render_template, request

import codec
import cube
import cube_codec
import cube_vision
import cube_screen

import cv2
import numpy as np

app = Flask(__name__, static_folder="static", template_folder="templates")
TABLE = None            # cube_codec.WordTable
MODE = "both"          # "encode" | "decode" | "both"
PORTS = {"encode": 8771, "decode": 8772, "both": 8770}

CONFIG = {
    "palette": {k: "#%02x%02x%02x" % v for k, v in cube_vision.PALETTE.items()},
    "background": "#%02x%02x%02x" % cube_vision.BACKGROUND,
    "size": cube_vision.SIZE,
    "half": cube_vision.HALF,
    "views": {k: list(v) for k, v in cube_vision.VIEWS.items()},
    "cubie": cube_vision.CUBIE,
    "sticker": cube_vision.STICKER,
    "solved": cube.SOLVED,
    "data_bits": cube_codec.DATA_BITS,
}


def bits_string(data):
    return format(data, f"0{cube_codec.DATA_BITS}b")


def page_config():
    import time
    return dict(CONFIG, mode=MODE, v=int(time.time()))


@app.get("/")
def index():
    return render_template("cube.html", config=page_config(), render_only=None)


@app.get("/render")
def render_only():
    """test harness: draws one fixed export view full-page (for headless screenshots)"""
    f = request.args.get("f", cube.SOLVED)
    view = request.args.get("view", "A")
    cube.facelets_to_cubies(f)
    return render_template("cube.html", config=page_config(), render_only={"facelets": f, "view": view})


def _only(mode):
    if MODE not in (mode, "both"):
        return jsonify({"error": f"this is the {MODE} site; /{mode} lives on the other computer"}), 404


@app.get("/random")
def random_words():
    import random
    if (err := _only("encode")):
        return err
    words = random.sample(TABLE.words[:TABLE.n_base], 3)
    return jsonify({"words": words, "tiers": cube_codec.tiers(words, TABLE)})


@app.post("/encode")
def encode():
    if (err := _only("encode")):
        return err
    body = request.get_json(silent=True) or {}
    words = [str(w).strip().lower() for w in body.get("words", [])]
    if len(words) != 3 or not all(words):
        return jsonify({"error": "please enter three words"}), 400
    try:
        facelets, data = cube_codec.words_to_state(words, TABLE)
        scramble = cube.scramble(facelets)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"words": words, "tiers": cube_codec.tiers(words, TABLE), "facelets": facelets,
                    "scramble": scramble, "bits": bits_string(data)})


def read_image_bytes(data):
    """JPEG/PNG (OpenCV) or HEIC (pillow-heif) -> BGR array, or None"""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        try:
            import io
            import pillow_heif
            from PIL import Image, ImageOps
            pillow_heif.register_heif_opener()
            pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return img


@app.post("/detect")
def detect():
    """live camera frame -> the cube's hexagon outline (pixel coords of the frame) or null"""
    if (err := _only("decode")):
        return err
    img = read_image_bytes(request.get_data())
    if img is None:
        return jsonify({"hex": None})
    small = cube_screen.shrink(img, 900)
    hx = cube_screen.detect_hexagon(small)
    scale = img.shape[1] / small.shape[1]
    return jsonify({"hex": (hx * scale).tolist() if hx is not None else None})


@app.post("/photo_decode")
def photo_decode():
    """two phone photos of the encoder screen (front + Other side) -> words"""
    if (err := _only("decode")):
        return err
    files = [f for f in request.files.values() if f and f.filename]
    if len(files) != 2:
        return jsonify({"error": "send two photos: the front view and the Other side view"}), 400
    images = [read_image_bytes(f.read()) for f in files]
    if any(i is None for i in images):
        return jsonify({"error": "could not read a photo"}), 400

    def accept(f):
        cube_codec.state_to_words(f, TABLE)
        return True
    try:
        facelets, info = cube_screen.read_screen_photos(images, accept=accept)
        words, data, tiers = cube_codec.state_to_words(facelets, TABLE)
        solve = cube.solve(facelets)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"words": words, "tiers": tiers, "facelets": facelets, "solve": solve, "bits": bits_string(data), "info": info})


@app.post("/decode")
def decode():
    if (err := _only("decode")):
        return err
    files = [f for f in request.files.values() if f and f.filename]
    if len(files) != 2:
        return jsonify({"error": "upload the two exported images (A and B)"}), 400
    try:
        images = [cube_vision.decode_image(f.read()) for f in files]
        facelets, info = cube_vision.read_stickers(images)
        words, data, tiers = cube_codec.state_to_words(facelets, TABLE)
        solve = cube.solve(facelets)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    out = {"words": words, "tiers": tiers, "facelets": facelets, "solve": solve, "bits": bits_string(data), "info": info}
    if request.args.get("debug"):
        out["overlays"] = [cube_vision.overlay(*cube_vision.identify_view(img)[:2]) for img in images]
    return jsonify(out)


def main(mode="both"):
    global MODE
    import argparse
    MODE = mode
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("words_csv", nargs="?", default=codec.default_word_list_path())
    ap.add_argument("port", nargs="?", type=int, default=PORTS[mode])
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--https", action="store_true", help="self-signed HTTPS on all interfaces (phone camera needs https)")
    args = ap.parse_args()
    global TABLE
    TABLE = cube_codec.load_table(words_csv=args.words_csv)
    url = f"http://127.0.0.1:{args.port}/"
    if args.https:
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            lan = s.getsockname()[0]
            s.close()
        except OSError:
            lan = "127.0.0.1"
        url = f"https://{lan}:{args.port}/"
    label = {"encode": "CUBE ENCODE app (sender)", "decode": "CUBE DECODE app (receiver)", "both": "CUBE app (encode + decode)"}[mode]
    print(f"{label}  |  list {TABLE.n_base:,} + dictionary {TABLE.n_extra:,} = {len(TABLE):,} words "
          f"({cube_codec.BITS_PER_WORD} bits each)\nopen {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    if args.https:
        print("self-signed certificate: the phone will warn once; choose 'visit anyway'", flush=True)
        app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True, ssl_context="adhoc")
    else:
        app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
