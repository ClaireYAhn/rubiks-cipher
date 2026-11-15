"""photo_sim.py - draw a cube state the way a handheld camera would photograph it (for tests).

Perspective camera at a random pose near a chosen corner, shaded faces, colour cast, glare, blur,
sensor noise, JPEG compression and a cluttered background. Everything is random but seeded.
"""
import math

import cv2
import numpy as np

import cube

PALETTES = {
    "standard": {"U": (236, 236, 232), "R": (196, 30, 45), "F": (20, 150, 70), "D": (242, 212, 30), "L": (240, 118, 20), "B": (20, 80, 190)},
    "bright": {"U": (252, 252, 252), "R": (240, 50, 70), "F": (50, 205, 90), "D": (255, 238, 60), "L": (255, 140, 30), "B": (30, 120, 245)},
    "cyanpink": {"U": (242, 242, 240), "R": (238, 100, 135), "F": (40, 165, 80), "D": (250, 232, 45), "L": (248, 150, 55), "B": (30, 195, 238)},
    "dull": {"U": (215, 215, 205), "R": (165, 55, 55), "F": (50, 125, 85), "D": (215, 190, 70), "L": (205, 115, 60), "B": (55, 85, 150)},
    "pastel": {"U": (238, 238, 238), "R": (240, 169, 170), "F": (184, 220, 177), "D": (238, 229, 157), "L": (242, 197, 161), "B": (150, 198, 233)},
}


def _look_at(eye, up_roll):
    z = eye / np.linalg.norm(eye)
    x = np.cross([0.0, 1.0, 0.0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c, s = math.cos(up_roll), math.sin(up_roll)
    return np.array([c * x + s * y, -(-s * x + c * y), z])     # rows: right, down, forward-to-camera; image y points down


def _face_axes(n):
    a = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.5 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def render_photo(facelets, rng, palette="standard", corner=(1, 1, 1), size=(800, 640), jitter_deg=22, clutter=True,
                 fov_deg=None, fill=None):
    """-> BGR uint8 image. `rng` is a numpy Generator."""
    W, H = size
    S = 2                                                       # supersample
    pal = PALETTES[palette] if isinstance(palette, str) else palette
    corner = np.array(corner, float)
    # random pose around the corner direction
    ang = math.radians(jitter_deg)
    dirv = corner / np.linalg.norm(corner)
    for _ in range(3):
        dirv = dirv + rng.normal(0, math.tan(ang) / 2.0, 3)
        dirv /= np.linalg.norm(dirv)
    roll = rng.uniform(-math.pi, math.pi)
    fov = math.radians(fov_deg if fov_deg else rng.uniform(22, 62))
    fill = fill if fill else rng.uniform(0.40, 0.70)
    dist = 2.4 / (fill * math.tan(fov / 2))
    eye = dirv * dist
    R = _look_at(eye, roll)
    f = (H * S / 2) / math.tan(fov / 2)
    cx, cy = W * S / 2 + rng.uniform(-0.06, 0.06) * W * S, H * S / 2 + rng.uniform(-0.06, 0.06) * H * S

    def proj(P):
        P = np.atleast_2d(P)
        c = (P - eye) @ R.T
        z = -c[:, 2]
        z = np.where(z < 1e-3, 1e-3, z)
        return np.stack([f * c[:, 0] / z + cx, f * c[:, 1] / z + cy], axis=1), z

    img = np.zeros((H * S, W * S, 3), np.float32)
    # background
    bg = np.array([rng.uniform(60, 250)] * 3) * rng.uniform(0.9, 1.0, 3)
    img[:] = bg
    gx = np.linspace(-1, 1, W * S)[None, :, None]
    gy = np.linspace(-1, 1, H * S)[:, None, None]
    img *= 1 + 0.18 * (rng.uniform(-1, 1) * gx + rng.uniform(-1, 1) * gy)
    if clutter:
        for _ in range(rng.integers(3, 12)):
            x0, y0 = rng.uniform(0, W * S), rng.uniform(0, H * S)
            w, h = rng.uniform(20, 300) * S, rng.uniform(20, 300) * S
            col = tuple(float(v) for v in rng.uniform(20, 240, 3))
            if rng.random() < 0.5:
                cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + w), int(y0 + h)), col, -1)
            else:
                cv2.ellipse(img, (int(x0), int(y0)), (int(w / 2), int(h / 2)), rng.uniform(0, 180), 0, 360, col, -1)
        if rng.random() < 0.25:                                 # a tiled surface: the classic false positive
            t = int(rng.uniform(40, 90)) * S
            ox, oy = rng.uniform(0, t), rng.uniform(0, t)
            col = tuple(float(v) for v in rng.uniform(100, 230, 3))
            img[:] = rng.uniform(15, 60)               # dark grout
            for xx in np.arange(-ox, W * S, t):
                for yy in np.arange(-oy, H * S, t):
                    cv2.rectangle(img, (int(xx + 3), int(yy + 3)), (int(xx + t - 3), int(yy + t - 3)), col, -1)
    cast = rng.uniform(0.9, 1.1, 3)
    gain = rng.uniform(0.75, 1.1)
    light = rng.normal(size=3)
    light[1] = abs(light[1]) + 0.5
    light /= np.linalg.norm(light)
    order = []
    for k, face in enumerate(cube.FACES):
        n = np.array(cube.GEOMETRY[9 * k + 4][1], float)
        if n @ (eye - 1.5 * n) > 0.05 * dist:
            order.append((face, n))
    for face, n in order:
        k = cube.FACES.index(face)
        u, v = _face_axes(n)
        centre = 1.5 * n
        shade = 0.82 + 0.18 * max(0.0, float(n @ light))
        quad = [centre + u * 1.5 + v * 1.5, centre - u * 1.5 + v * 1.5, centre - u * 1.5 - v * 1.5, centre + u * 1.5 - v * 1.5]
        cv2.fillPoly(img, [np.rint(proj(np.array(quad))[0]).astype(np.int32)], (24, 24, 24), lineType=cv2.LINE_AA)
        for i in range(9):
            p, nn = cube.GEOMETRY[9 * k + i]
            c3 = np.where(np.array(nn) != 0, np.array(nn) * 1.5, np.array(p, float))
            h, ch = 0.43, 0.09                                  # sticker half-size, chamfered (rounded) corners
            outline = [(h - ch, h), (-(h - ch), h), (-h, h - ch), (-h, -(h - ch)),
                       (-(h - ch), -h), (h - ch, -h), (h, -(h - ch)), (h, h - ch)]
            pts = [c3 + u * a + v * b for a, b in outline]
            col = np.array(pal[facelets[9 * k + i]], float) * shade * rng.uniform(0.96, 1.04)
            cv2.fillPoly(img, [np.rint(proj(np.array(pts))[0]).astype(np.int32)], tuple(float(c) for c in col), lineType=cv2.LINE_AA)
    img *= (gain * cast)[None, None, :]
    if rng.random() < 0.35:                                     # specular glare patch
        gxp, gyp = rng.uniform(0.2, 0.8) * W * S, rng.uniform(0.2, 0.8) * H * S
        m = np.zeros((H * S, W * S), np.float32)
        cv2.ellipse(m, (int(gxp), int(gyp)), (int(rng.uniform(15, 50) * S), int(rng.uniform(8, 25) * S)), rng.uniform(0, 180), 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), 6 * S)
        img = img + (255 - img) * (m[..., None] * rng.uniform(0.3, 0.6))
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    sigma = rng.uniform(0.3, 1.6)
    img = cv2.GaussianBlur(img, (0, 0), sigma)
    img = np.clip(img.astype(np.float32) + rng.normal(0, rng.uniform(1.5, 5.5), img.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 92))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
