"""cube_screen.py - read the cube from phone photos of the ENCODER'S SCREEN.

The encoder page draws the cube with a fixed perspective camera (front view, or the "Other side" view which is
exactly the view from the opposite camera). A phone photo of the laptop screen is therefore a flat image seen in
perspective: one homography maps the page's square canvas onto the photo. We find the cube's dark hexagonal
outline in the photo, match its 6 corners to the 6 silhouette corners of the reference view, and the homography
tells us where all 27 visible sticker centres are. Colours are classified against the pastel palette after a
white-balance fit on the three visible centre stickers, then nine-per-colour is enforced.
"""
import itertools
import math

import cv2
import numpy as np

import cube
from cube_vision import PALETTE, STICKER
from cube_photo import hungarian

# must match cube.js: camera.position / lookAt / fov, square canvas
FOV = 32.0
EYE = np.array([5.6, 4.8, 7.1])
TARGET = np.array([0.0, -0.3, 0.0])
VIEWS = ("front", "back")
PROCESS_SIDE = 900


class ScreenError(ValueError):
    pass


def _norm(v):
    return v / np.linalg.norm(v)


def _camera(view):
    eye, target = (EYE, TARGET) if view == "front" else (-EYE, -TARGET)
    z = _norm(eye - target)
    x = _norm(np.cross([0.0, 1.0, 0.0], z))
    y = np.cross(z, x)
    return eye, x, y, z


def project(points, view):
    """world points (N,3) -> normalised square-canvas coords (N,2), (0,0) top-left, (1,1) bottom-right"""
    eye, x, y, z = _camera(view)
    d = np.atleast_2d(points) - eye
    xc, yc, zc = d @ x, d @ y, d @ z
    f = 1.0 / math.tan(math.radians(FOV) / 2)
    nx, ny = f * xc / (-zc), f * yc / (-zc)
    return np.stack([(nx + 1) / 2, (1 - ny) / 2], axis=1)


def order_hexagon(pts):
    """6 points -> clockwise (image coords) starting from the topmost"""
    pts = np.asarray(pts, float)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    k = int(np.argmin(pts[:, 1]))
    return np.roll(pts, -k, axis=0)


def _sticker_world(i):
    p, n = cube.GEOMETRY[i]
    p, n = np.array(p, float), np.array(n, float)
    centre = np.where(n != 0, n * 1.5, p)
    a = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.5 else np.array([1.0, 0.0, 0.0])
    u = _norm(np.cross(n, a))
    v = np.cross(n, u)
    h = STICKER * 0.5 * 0.45          # inset: sample the middle ~45% of the sticker
    return centre, [centre + u * h + v * h, centre - u * h + v * h, centre - u * h - v * h, centre + u * h - v * h]


def reference(view):
    """-> (hexagon 6x2 normalised, {facelet: (centre 2, quad 4x2)}) for the view"""
    eye = _camera(view)[0]
    corners = np.array([[sx, sy, sz] for sx in (-1.5, 1.5) for sy in (-1.5, 1.5) for sz in (-1.5, 1.5)])
    hull = cv2.convexHull(project(corners, view).astype(np.float32)).reshape(-1, 2)
    if len(hull) != 6:
        raise RuntimeError(f"reference silhouette has {len(hull)} corners")
    stickers = {}
    for i, (p, n) in enumerate(cube.GEOMETRY):
        centre, quad = _sticker_world(i)
        if np.dot(n, eye - centre) > 0:
            stickers[i] = (project(centre, view)[0], project(np.array(quad), view))
    return order_hexagon(hull), stickers


REF = {v: reference(v) for v in VIEWS}


# ---------------------------------------------------------------- hexagon detection
def shrink(img, side=PROCESS_SIDE):
    h, w = img.shape[:2]
    s = side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return img


def _decimate(poly, n=6):
    """convex polygon (k,2) -> n vertices, dropping the vertex whose removal loses the least area"""
    pts = [p for p in poly]
    while len(pts) > n:
        k = len(pts)
        loss = []
        for i in range(k):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % k]
            loss.append(abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])))
        pts.pop(int(np.argmin(loss)))
    return np.array(pts, float)


def _angles(poly):
    k = len(poly)
    out = []
    for i in range(k):
        a, b, c = poly[i - 1], poly[i], poly[(i + 1) % k]
        u, v = a - b, c - b
        out.append(math.degrees(math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9), -1, 1))))
    return np.array(out)


def detect_hexagon(bgr):
    """-> 6x2 float pixel corners (ordered) of the cube's dark outline, or None.
    The outline is the thick dark grid on the light page: dark relative to its surroundings (black-hat),
    convex, hexagonal with all corners near 120 degrees."""
    h, w = bgr.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    k = max(15, (max(h, w) // 30) | 1)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = ((bh > 35) | (g < 60)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # LIST: the cube sits inside the bezel's contour
    best = None
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:12]:
        hull = cv2.convexHull(c).reshape(-1, 2).astype(float)
        harea = cv2.contourArea(hull.astype(np.float32))
        if harea < 0.01 * h * w:
            break
        if harea > 0.9 * h * w or len(hull) < 6:
            continue
        x, y, bw, bhh = cv2.boundingRect(hull.astype(np.float32))
        if not 0.6 < bw / bhh < 1.7:
            continue
        hexagon = _decimate(hull)
        ang = _angles(hexagon)
        if ang.min() < 80 or ang.max() > 160:
            continue
        if cv2.contourArea(hexagon.astype(np.float32)) / harea < 0.9:
            continue
        holes = _sticker_holes(m, hexagon, x, y, bw, bhh)
        if holes < 10:                      # a cube outline encloses ~27 sticker-sized holes; a bezel does not
            continue
        score = min(holes, 27) + harea / (h * w)
        if best is None or score > best[0]:
            best = (score, order_hexagon(hexagon))
    return None if best is None else best[1]


def _sticker_holes(mask, hexagon, x, y, bw, bh):
    """number of sticker-sized light regions inside the hexagon"""
    sub = mask[y:y + bh, x:x + bw]
    poly = np.zeros_like(sub)
    cv2.fillConvexPoly(poly, np.rint(hexagon - [x, y]).astype(np.int32), 255)
    inv = ((sub == 0) & (poly > 0)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    area = cv2.contourArea(hexagon.astype(np.float32))
    return int(sum(1 for i in range(1, n) if 0.004 * area < stats[i, cv2.CC_STAT_AREA] < 0.08 * area))


# ---------------------------------------------------------------- reading
def _sample_poly(rgb, poly):
    h, w = rgb.shape[:2]
    x0, y0 = np.floor(poly.min(axis=0)).astype(int)
    x1, y1 = np.ceil(poly.max(axis=0)).astype(int) + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 - x0 < 2 or y1 - y0 < 2:
        return None
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(m, np.rint(poly - [x0, y0]).astype(np.int32), 1)
    px = rgb[y0:y1, x0:x1][m > 0]
    return np.median(px, axis=0) if len(px) else None


def _warp(H, pts):
    return cv2.perspectiveTransform(np.asarray(pts, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)


PAL = {k: np.array(v, float) for k, v in PALETTE.items()}


def _gain(samples, targets):
    s, t = np.asarray(samples, float), np.asarray(targets, float)
    g = (s * t).sum(axis=0) / np.maximum((s * s).sum(axis=0), 1e-6)
    return np.clip(g, 0.5, 2.5)


def _photo_candidates(rgb, hexagon):
    """all (view, cyclic shift) hypotheses for one photo -> list of dicts sorted by centre-colour error"""
    out = []
    for view in VIEWS:
        ref_hex, stickers = REF[view]
        for k in range(6):
            dst = np.roll(hexagon, -k, axis=0)
            H, _ = cv2.findHomography(ref_hex.astype(np.float32), dst.astype(np.float32), 0)
            if H is None:
                continue
            resid = float(np.sqrt(np.mean(np.sum((_warp(H, ref_hex) - dst) ** 2, axis=1))))
            size = float(np.linalg.norm(dst.max(axis=0) - dst.min(axis=0)))
            centres = [9 * cube.FACES.index(f) + 4 for f in cube.FACES if 9 * cube.FACES.index(f) + 4 in stickers]
            samp = [_sample_poly(rgb, _warp(H, stickers[i][1])) for i in centres]
            if any(s is None for s in samp):
                continue
            g = _gain(samp, [PAL[cube.FACES[i // 9]] for i in centres])
            err = float(np.mean([np.linalg.norm(np.clip(s * g, 0, 255) - PAL[cube.FACES[i // 9]]) for s, i in zip(samp, centres)]))
            out.append({"view": view, "H": H, "resid": resid / size, "err": err, "gain": g})
    out.sort(key=lambda d: d["err"] + 400 * d["resid"])
    return out


def read_screen_photos(images_bgr, accept=None, max_swap=10):
    """two phone photos of the encoder screen (front + other side, any order) -> (facelets, info)"""
    if len(images_bgr) != 2:
        raise ScreenError("take exactly two photos: the front view and the Other side view")
    photos = []
    for pi, img in enumerate(images_bgr):
        small = shrink(img)
        hexagon = detect_hexagon(small)
        if hexagon is None:
            raise ScreenError(f"photo {pi + 1}: no cube outline found. Fill the frame with the cube on the screen")
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        cands = _photo_candidates(rgb, hexagon)
        if not cands:
            raise ScreenError(f"photo {pi + 1}: could not fit the cube outline")
        photos.append((small, rgb, hexagon, cands))
    # the two photos must show different views: pick the best consistent pair
    best = None
    for a in photos[0][3][:6]:
        for b in photos[1][3][:6]:
            if a["view"] == b["view"]:
                continue
            s = a["err"] + b["err"] + 400 * (a["resid"] + b["resid"])
            if best is None or s < best[0]:
                best = (s, a, b)
    if best is None:
        raise ScreenError("both photos show the same side; take one front photo and one Other side photo")
    choice = [best[1], best[2]]
    rgb54 = np.full((54, 3), np.nan)
    pos = {}
    for pi, ((small, rgb, hexagon, _), c) in enumerate(zip(photos, choice)):
        _, stickers = REF[c["view"]]
        for i, (centre, quad) in stickers.items():
            s = _sample_poly(rgb, _warp(c["H"], quad))
            if s is None:
                raise ScreenError("a sticker falls outside the photo; keep the whole cube in view")
            rgb54[i] = np.clip(s * c["gain"], 0, 255)
            pos[i] = (_warp(c["H"], centre)[0], pi)
    if np.isnan(rgb54).any():
        raise ScreenError("not all stickers were visible")
    # classify: start from the palette, let the class centres adapt to this screen, then enforce 9 per colour
    letters = list(cube.FACES)
    cents = {f: PAL[f].copy() for f in letters}
    for _ in range(3):
        d = np.array([[np.linalg.norm(rgb54[i] - cents[f]) for f in letters] for i in range(54)])
        lab = d.argmin(axis=1)
        for j, f in enumerate(letters):
            if (lab == j).sum() >= 3:
                cents[f] = 0.5 * cents[f] + 0.5 * rgb54[lab == j].mean(axis=0)
    d = np.array([[np.linalg.norm(rgb54[i] - cents[f]) for f in letters] for i in range(54)])
    centres = {9 * k + 4: f for k, f in enumerate(letters)}
    free = [i for i in range(54) if i not in centres]
    col = hungarian(np.repeat(d[free], 8, axis=1))
    out = [None] * 54
    for i, f in centres.items():
        out[i] = f
    margin = np.full(54, np.inf)
    for r, i in enumerate(free):
        out[i] = letters[col[r] // 8]
        a = col[r] // 8
        margin[i] = float(np.delete(d[i], a).min() - d[i][a])
    facelets = "".join(out)
    info = {"views": [c["view"] for c in choice], "centre_err": [round(c["err"], 1) for c in choice],
            "min_margin": round(float(margin[np.isfinite(margin)].min()), 1), "corrected": []}
    if accept is not None and not _ok(accept, facelets):
        unsure = [int(i) for i in np.argsort(margin)[:max_swap] if np.isfinite(margin[i])]
        for a, b in itertools.combinations(unsure, 2):
            if facelets[a] == facelets[b]:
                continue
            f2 = list(facelets)
            f2[a], f2[b] = f2[b], f2[a]
            f2 = "".join(f2)
            if _ok(accept, f2):
                info["corrected"] = [a, b]
                facelets = f2
                break
    return facelets, info


def _ok(accept, f):
    try:
        return bool(accept(f))
    except ValueError:
        return False
