"""WSGI entry for hosting the DECODE site (Render / Hugging Face Spaces / Fly / any Docker host).

    gunicorn -w 2 -b 0.0.0.0:$PORT wsgi:app
Set CUBE_MODE=both to serve encode + decode from one deployment.
"""
import os

import cube_app
import cube_codec

cube_app.MODE = os.environ.get("CUBE_MODE", "decode")
cube_app.TABLE = cube_codec.load_table()
app = cube_app.app
