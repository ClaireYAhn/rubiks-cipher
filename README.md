# Rubik's Cipher

> A scrambled cube looks like noise. This one is a sentence. Solve it and the words come out.

Three words -> one Rubik's cube scramble -> two pictures (or two phone photos of the screen) -> three words.
See CUBE.md for the method and the file map.

## Run locally
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python cube_encode_app.py            # sender   http://127.0.0.1:8771
.venv/bin/python cube_decode_app.py            # receiver http://127.0.0.1:8772  (add --https for phones on the same Wi-Fi)
```

## Deploy the decode site (judges scan a QR on their own phones)
Docker host of your choice: Render (render.yaml included, free plan) or Hugging Face Spaces (Docker SDK, port 7860).
`CUBE_MODE=both` serves the encoder too.
