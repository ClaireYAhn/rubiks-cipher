"""cube_photo.py - read a scrambled cube from ordinary photos (any camera, any pose, any cube).

cube_vision.py only understands the app's own two exported renders (fixed camera). This module
reads real photographs. Each photo should show two or three faces of the cube (a corner view);
together the photos must cover all six faces. Nothing about the camera is assumed.

Pipeline, per photo:
  1. find sticker-shaped regions (closed outlines of roughly square, similar-sized patches)
  2. group them into 3x3 grids, one grid per visible face (a homography fitted to the grid
     fills in stickers the edge detector missed)
  3. sample each sticker's colour at the predicted centre
Then across all photos:
  4. cluster every sticker into 6 colours and name them white/red/green/yellow/orange/blue
     (the centre sticker of a grid says which face the grid is)
  5. work out how each grid is rotated/flipped: try every orientation of every grid and keep the
     combination that a single camera could really have produced (affine camera fit onto the 3D
     sticker positions of the cube model)
  6. reassemble the 54-letter facelet string; enforce nine stickers per colour
  7. if the cube is not a legal, encoded state, try swapping the least certain stickers
"""
import itertools
import math

import cv2
import numpy as np

import cube

MAX_SIDE = 1100
# face letter -> typical photographed colour (used only to NAME clusters; the clusters adapt to the cube)
CANON_RGB = {"U": (235, 235, 235), "R": (190, 30, 45), "F": (20, 150, 70),
             "D": (240, 215, 40), "L": (240, 120, 20), "B": (20, 80, 190)}
COLOUR_NAME = {"U": "white", "R": "red", "F": "green", "D": "yellow", "L": "orange", "B": "blue"}


class PhotoError(ValueError):
    """the photos could not be read into a cube; the message says what to change"""


# ---------------------------------------------------------------- small helpers
def hungarian(cost):
    """min-cost assignment for a square matrix -> list col_of_row (O(n^3), no scipy needed)"""
    a = np.asarray(cost, float)
    n = a.shape[0]
    u, v = np.zeros(n + 1), np.zeros(n + 1)
    p, way = np.zeros(n + 1, int), np.zeros(n + 1, int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, bool)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    res = [0] * n
    for j in range(1, n + 1):
        res[p[j] - 1] = j - 1
    return res


def load_image(data):
    """bytes -> BGR uint8 (EXIF rotation applied by OpenCV)"""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise PhotoError("could not open an image file")
    return img


def shrink(img, max_side=MAX_SIDE):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------------- 1. sticker candidates
class Cand:
    __slots__ = ("c", "area", "quad", "side", "d1", "d2")

    def __init__(self, c, area, quad):
        self.c = np.asarray(c, float)
        self.area = float(area)
        self.quad = quad
        self.side = math.sqrt(area)
        # the quad's two edge directions (opposite edges averaged); quads of one face share them
        q = np.asarray(quad, float)
        self.d1 = (q[1] - q[0] + q[2] - q[3]) / 2
        self.d2 = (q[3] - q[0] + q[2] - q[1]) / 2
        if self.d1[0] * self.d2[1] - self.d1[1] * self.d2[0] < 0:
            self.d2 = -self.d2


def _blobs(mask, min_area, max_area):
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=4)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if not (min_area <= area <= max_area) or bw > 3 * bh or bh > 3 * bw:
            continue
        sub = (labels[y:y + bh, x:x + bw] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)
        carea = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        harea = cv2.contourArea(hull)
        if harea < 1 or carea / harea < 0.88:
            continue
        peri = cv2.arcLength(hull, True)
        quad = None
        for eps in (0.03, 0.045, 0.06, 0.08, 0.1):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                quad = ap.reshape(4, 2).astype(float)
                break
        if quad is None:
            continue
        qarea = cv2.contourArea(quad.astype(np.float32))
        if qarea < 1 or harea / qarea < 0.86:
            continue
        sides = [np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]
        if min(sides) < 1 or max(sides) / min(sides) > 3.0:
            continue
        quad = quad + np.array([x, y], float)
        m = cv2.moments(hull)
        c = (m["m10"] / m["m00"] + x, m["m01"] / m["m00"] + y)
        out.append((harea / qarea * (carea / harea), Cand(c, harea, quad)))
    return out


def find_candidates(bgr):
    """bright, convex, roughly square patches separated by the dark cube body -> list[Cand].
    Several brightness thresholds are tried (a blue sticker is darker than a white one)."""
    h, w = bgr.shape[:2]
    v = bgr.max(axis=2)
    v = cv2.GaussianBlur(v, (0, 0), 0.8)
    # dark gap lines between stickers: darker than their surroundings (black-hat)
    k = max(9, int(round(max(h, w) / 55)) | 1)
    gaps = cv2.morphologyEx(v, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    pool = []
    for thr in (35, 50, 70, 95, 125):
        for gap_thr in (28, 45):
            mask = ((v > thr) & (gaps < gap_thr)).astype(np.uint8)
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
            pool += _blobs(mask, 60, 0.08 * h * w)
    pool.sort(key=lambda t: -t[0])
    kept = []
    for q, c in pool:
        if any(np.hypot(*(c.c - k.c)) < 0.35 * c.side and 0.6 < c.area / k.area < 1.6 for k in kept):
            continue
        kept.append(c)
    return kept


# ---------------------------------------------------------------- 2. group stickers into 3x3 grids (faces)
class Grid:
    """one visible face: H maps grid coords (col,row) in 0..2 to pixels"""

    def __init__(self, H, matched, rms, cell):
        self.H = H
        self.matched = matched          # {(col,row): Cand}
        self.rms = rms                  # fit error in units of one cell
        self.cell = cell                # typical cell size in pixels
        self.colours = None             # 3x3 list of RGB medians, filled by sample_grid
        self.n = len(matched)
        self.uniform = False            # True if all nine cells look alike (a tile floor, not a scrambled face)

    def px(self, col, row):
        p = self.H @ np.array([col, row, 1.0])
        return p[:2] / p[2]

    def centre_px(self):
        return self.px(1, 1)

    def area(self):
        quad = np.array([self.px(-0.5, -0.5), self.px(2.5, -0.5), self.px(2.5, 2.5), self.px(-0.5, 2.5)], np.float32)
        return float(cv2.contourArea(quad))


def _fit_homography(grid_pts, img_pts):
    g, p = np.asarray(grid_pts, np.float32), np.asarray(img_pts, np.float32)
    if len(g) == 4:
        return cv2.getPerspectiveTransform(g, p)
    H, _ = cv2.findHomography(g, p, 0)
    return H


def _refine(H, cands, free, tol):
    """match every grid cell to the nearest free candidate near its predicted centre"""
    matched, used = {}, set()
    for row in range(3):
        for col in range(3):
            p = H @ np.array([col, row, 1.0])
            p = p[:2] / p[2]
            best, bd = None, None
            for i in free:
                if i in used:
                    continue
                d = np.linalg.norm(cands[i].c - p)
                if d < tol * cands[i].side and (bd is None or d < bd):
                    best, bd = i, d
            if best is not None:
                matched[(col, row)] = best
                used.add(best)
    return matched


def find_grids(cands, max_grids=3, min_match=5, rgb=None, max_tries=40):
    """-> list[Grid], best first. Each candidate joins at most one grid. With `rgb`, grids whose nine cells
    all look alike (tiles, a wall) are skipped without using up one of the `max_grids` places."""
    grids, taken = [], set()
    tries = 0
    pts = np.array([c.c for c in cands]) if cands else np.zeros((0, 2))
    D1 = np.array([c.d1 / (np.linalg.norm(c.d1) + 1e-9) for c in cands]) if cands else np.zeros((0, 2))
    D2 = np.array([c.d2 / (np.linalg.norm(c.d2) + 1e-9) for c in cands]) if cands else np.zeros((0, 2))
    SIDE = np.array([c.side for c in cands]) if cands else np.zeros(0)

    def sin_to(D, w):
        return np.abs(D[:, 0] * w[1] - D[:, 1] * w[0]) / (np.linalg.norm(w) + 1e-9)

    while len(grids) < max_grids and tries < max_tries:
        tries += 1
        free = [i for i in range(len(cands)) if i not in taken]
        if len(free) < min_match:
            break
        P = pts[free]
        best = None
        for ii, i in enumerate(free):
            ci = cands[i]
            d = np.linalg.norm(P - ci.c, axis=1)
            near = [free[q] for q in np.argsort(d)[1:9]
                    if 0.7 * ci.side < d[q] < 2.3 * ci.side and 0.4 < cands[free[q]].area / ci.area < 2.5]
            for j, k in itertools.combinations(near, 2):
                u, v = cands[j].c - ci.c, cands[k].c - ci.c
                nu, nv = np.linalg.norm(u), np.linalg.norm(v)
                if abs(u[0] * v[1] - u[1] * v[0]) < 0.35 * nu * nv or not 0.5 < nu / nv < 2.0:
                    continue
                B = np.linalg.inv(np.array([u, v]).T)
                co = (P - ci.c) @ B.T
                r = np.rint(co)
                ok = (np.abs(co - r).max(axis=1) < 0.3) & (np.abs(r).max(axis=1) <= 2)
                # each sticker's own edges must run along the grid directions (keeps one grid on one face)
                f = np.array(free)
                su1, sv1, su2, sv2 = sin_to(D1[f], u), sin_to(D1[f], v), sin_to(D2[f], u), sin_to(D2[f], v)
                ok &= ((su1 < 0.4) & (sv2 < 0.4)) | ((sv1 < 0.4) & (su2 < 0.4))
                ok &= (SIDE[f] / (0.5 * (nu + nv)) > 0.55) & (SIDE[f] / (0.5 * (nu + nv)) < 1.1)
                if ok.sum() < min_match:
                    continue
                rr = r[ok].astype(int)
                bw = None
                for oa in (-2, -1, 0):
                    for ob in (-2, -1, 0):
                        inw = (rr[:, 0] >= oa) & (rr[:, 0] <= oa + 2) & (rr[:, 1] >= ob) & (rr[:, 1] <= ob + 2)
                        if bw is None or inw.sum() > bw[0]:
                            bw = (int(inw.sum()), oa, ob, inw)
                cnt, oa, ob, inw = bw
                if best is None or cnt > best[0]:
                    sel = np.array(free)[np.where(ok)[0][inw]]
                    cells = [(int(r[ok][m][0]) - oa, int(r[ok][m][1]) - ob) for m in np.where(inw)[0]]
                    best = (cnt, sel, cells)
        if best is None or best[0] < min_match:
            break
        _, sel, cells = best
        H = None
        for it in range(3):
            if it == 0:
                gp, ip = cells, [cands[i].c for i in sel]
            else:
                gp, ip = list(matched), [cands[i].c for i in matched.values()]
            if len(gp) < 4:
                H = None
                break
            H = _fit_homography(gp, ip)
            if H is None:
                break
            matched = _refine(H, cands, free, 0.45)
        if H is None or len(matched) < min_match:
            taken |= set(sel.tolist())
            continue
        err = []
        for (col, row), i in matched.items():
            p = H @ np.array([col, row, 1.0])
            err.append(np.linalg.norm(p[:2] / p[2] - cands[i].c) / cands[i].side)
        cell = float(np.mean([cands[i].side for i in matched.values()]))
        g = Grid(H, {cr: cands[i] for cr, i in matched.items()}, float(np.sqrt(np.mean(np.square(err)))), cell)
        taken |= set(matched.values())
        if rgb is not None:
            try:
                sample_grid(rgb, g)
            except PhotoError:
                continue
            if g.uniform:
                continue
        grids.append(g)
    return grids


def sample_grid(rgb, grid, inset=0.22):
    """median RGB at each of the 9 predicted cell centres -> 3x3 array (row, col, 3)"""
    h, w = rgb.shape[:2]
    out = np.zeros((3, 3, 3))
    for row in range(3):
        for col in range(3):
            poly = np.array([grid.px(col + dx, row + dy) for dx, dy in ((-inset, -inset), (inset, -inset), (inset, inset), (-inset, inset))])
            x0, y0 = np.floor(poly.min(axis=0)).astype(int)
            x1, y1 = np.ceil(poly.max(axis=0)).astype(int) + 1
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
            if x1 - x0 < 2 or y1 - y0 < 2:
                raise PhotoError("a sticker falls outside the photo; keep the whole cube in view")
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillConvexPoly(m, np.rint(poly - [x0, y0]).astype(np.int32), 1)
            px = rgb[y0:y1, x0:x1][m > 0]
            out[row, col] = np.median(px, axis=0)
    grid.colours = out
    f = features(out.reshape(-1, 3))
    grid.uniform = bool(np.max(np.linalg.norm(f - np.median(f, axis=0), axis=1)) < 9.0)
    return out


# ---------------------------------------------------------------- 3. colour clusters -> names
def _lab(rgb):
    """RGB (0..255), shape (...,3) -> Lab with OpenCV's float convention (L 0..100)"""
    a = np.asarray(rgb, np.float32).reshape(-1, 1, 3) / 255.0
    return cv2.cvtColor(a, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)


L_WEIGHT = 0.4


def features(rgb):
    lab = _lab(rgb)
    lab[:, 0] *= L_WEIGHT
    return lab


def _kmeans(X, k, rng, iters=25):
    best = None
    for _ in range(12):
        c = X[[rng.integers(len(X))]]
        for _i in range(k - 1):              # k-means++ seeding
            d = np.min(((X[:, None] - c[None]) ** 2).sum(-1), axis=1)
            c = np.vstack([c, X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))]])
        for _i in range(iters):
            lab = np.argmin(((X[:, None] - c[None]) ** 2).sum(-1), axis=1)
            for j in range(k):
                if (lab == j).any():
                    c[j] = X[lab == j].mean(axis=0)
        inertia = float(np.min(((X[:, None] - c[None]) ** 2).sum(-1), axis=1).sum())
        if best is None or inertia < best[0]:
            best = (inertia, c.copy())
    return best[1]


# hue (degrees, atan2(b, a) in Lab) at which each face colour normally sits
HUE = {"R": 25.0, "L": 62.0, "D": 100.0, "F": 142.0, "B": 275.0}


def name_clusters(centroids):
    """6 centroids (feature space) -> face letter for each, one letter per centroid"""
    letters = list(cube.FACES)
    cost = np.zeros((6, 6))
    for i, c in enumerate(centroids):
        L, a, b = c[0] / L_WEIGHT, c[1], c[2]
        chroma = math.hypot(a, b)
        hue = math.degrees(math.atan2(b, a)) % 360
        for j, f in enumerate(letters):
            if f == "U":
                cost[i, j] = chroma * 1.6 + max(0.0, 78 - L) * 0.4
            else:
                dh = abs((hue - HUE[f] + 180) % 360 - 180)
                cost[i, j] = dh + max(0.0, 14 - chroma) * 4
    col = hungarian(cost)
    return [letters[col[i]] for i in range(6)]


def cluster_colours(rgb_samples, seed=1):
    """all matched sticker colours -> (centroids by face letter {letter: feature}, per-sample letter)"""
    X = features(rgb_samples)
    rng = np.random.default_rng(seed)
    cents = _kmeans(X, 6, rng)
    names = name_clusters(cents)
    return {names[i]: cents[i] for i in range(6)}


def classify(feats, centroids):
    """-> (letters, margins) nearest centroid per row; margin = gap to the runner-up"""
    ls = list(cube.FACES)
    C = np.array([centroids[f] for f in ls])
    d = np.sqrt(((feats[:, None] - C[None]) ** 2).sum(-1))
    order = np.argsort(d, axis=1)
    letters = [ls[o] for o in order[:, 0]]
    margin = d[np.arange(len(d)), order[:, 1]] - d[np.arange(len(d)), order[:, 0]]
    return letters, margin


# ---------------------------------------------------------------- 4. which way is each grid turned?  (camera fit)
def _centre3d(i):
    p, n = cube.GEOMETRY[i]
    p, n = np.array(p, float), np.array(n, float)
    return np.where(n != 0, n * 1.5, p)


STICKER_XYZ = np.array([_centre3d(i) for i in range(54)])
FACE_NORMAL = {f: np.array(cube.GEOMETRY[9 * k + 4][1], float) for k, f in enumerate(cube.FACES)}


def dihedral(k, c, r):
    """one of the 8 symmetries of the 3x3 grid, centre fixed"""
    x, y = c - 1, r - 1
    if k >= 4:
        x = -x
    for _ in range(k % 4):
        x, y = -y, x
    return x + 1, y + 1


def affine_fit(X, x):
    A = np.hstack([X, np.ones((len(X), 1))])
    M, *_ = np.linalg.lstsq(A, x, rcond=None)
    res = A @ M - x
    return M, float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))


def _grid_points(grid, letter, t):
    """3D sticker centres and pixel centres of the matched cells if the grid is turned by symmetry t"""
    fi = cube.FACES.index(letter)
    X, x = [], []
    for (c, r), cand in grid.matched.items():
        c2, r2 = dihedral(t, c, r)
        X.append(STICKER_XYZ[9 * fi + 3 * r2 + c2])
        x.append(cand.c)
    return np.array(X), np.array(x)


def fit_subset(grids, letters):
    """try all 8^k symmetry choices -> (choice, rms, second_rms, M, mirrored_only).
    Mirrored solutions (a view no real camera could give) are set aside: with only two faces the mirror
    image fits exactly as well as the truth, so the proper one must win. `mirrored_only` is True when
    the best fit anywhere is a mirror image and no proper solution exists."""
    opts = [[_grid_points(g, l, t) for t in range(8)] for g, l in zip(grids, letters)]
    scale = float(np.mean([g.cell for g in grids]))
    best, second, mirrored_best = None, None, None
    for choice in itertools.product(range(8), repeat=len(grids)):
        X = np.vstack([opts[i][t][0] for i, t in enumerate(choice)])
        x = np.vstack([opts[i][t][1] for i, t in enumerate(choice)])
        if len(X) < 6:
            continue
        M, rms = affine_fit(X, x)
        rms /= scale
        if camera_facing(M, letters)[1]:
            if mirrored_best is None or rms < mirrored_best:
                mirrored_best = rms
            continue
        if best is None or rms < best[1]:
            second = best[1] if best else None
            best = (choice, rms, M)
        elif second is None or rms < second:
            second = rms
    if best is None:
        return None, mirrored_best, None, None, True
    return best[0], best[1], second, best[2], False


def camera_facing(M, letters):
    """viewing direction d (towards the camera) and whether the view is mirrored"""
    A = M[:3].T                                   # 2x3: pixel x, pixel y as linear functions of (x,y,z)
    d = np.cross(A[0], A[1])
    d = d / (np.linalg.norm(d) + 1e-12)
    if sum(FACE_NORMAL[l] @ d for l in letters) < 0:
        d = -d
    mirrored = np.linalg.det(np.vstack([A[0], A[1], d])) > 0     # image y points down, so a true view has det < 0
    return d, bool(mirrored)


# ---------------------------------------------------------------- 5. one photo -> the faces it shows
def dlt(X, x):
    """3x4 perspective camera from >= 6 point pairs (Hartley-normalised DLT)"""
    X, x = np.asarray(X, float), np.asarray(x, float)
    cX, c2 = X.mean(0), x.mean(0)
    sX = math.sqrt(3) / max(np.mean(np.linalg.norm(X - cX, axis=1)), 1e-9)
    s2 = math.sqrt(2) / max(np.mean(np.linalg.norm(x - c2, axis=1)), 1e-9)
    T3 = np.array([[sX, 0, 0, -sX * cX[0]], [0, sX, 0, -sX * cX[1]], [0, 0, sX, -sX * cX[2]], [0, 0, 0, 1]])
    T2 = np.array([[s2, 0, -s2 * c2[0]], [0, s2, -s2 * c2[1]], [0, 0, 1]])
    Xn = (X - cX) * sX
    xn = (x - c2) * s2
    rows = []
    for (a, b, c), (u, v) in zip(Xn, xn):
        rows.append([a, b, c, 1, 0, 0, 0, 0, -u * a, -u * b, -u * c, -u])
        rows.append([0, 0, 0, 0, a, b, c, 1, -v * a, -v * b, -v * c, -v])
    P = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 4)
    P = np.linalg.inv(T2) @ P @ T3
    return P / np.linalg.norm(P)


def project(P, X):
    X = np.atleast_2d(X)
    h = np.hstack([X, np.ones((len(X), 1))]) @ P.T
    return h[:, :2] / h[:, 2:3], h[:, 2]


class FaceRead:
    def __init__(self, letter, rgb, pos, matched, score, source, photo):
        self.letter, self.rgb, self.pos, self.matched = letter, rgb, pos, matched
        self.score, self.source, self.photo = score, source, photo


def _inset_corners(letter, centre, inset):
    n = FACE_NORMAL[letter]
    a = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.5 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return [centre + u * inset + v * inset, centre - u * inset + v * inset, centre - u * inset - v * inset, centre + u * inset - v * inset]


def sample_polygon(rgb, poly):
    h, w = rgb.shape[:2]
    poly = np.asarray(poly, float)
    x0, y0 = np.floor(poly.min(axis=0)).astype(int)
    x1, y1 = np.ceil(poly.max(axis=0)).astype(int) + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 - x0 < 2 or y1 - y0 < 2:
        return None
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(m, np.rint(poly - [x0, y0]).astype(np.int32), 1)
    px = rgb[y0:y1, x0:x1][m > 0]
    return np.median(px, axis=0) if len(px) else None


MAX_RMS = 0.30      # camera-fit error (in sticker widths) beyond which a set of grids is not a cube corner


def read_photo(pi, rgb, grids, centroids):
    """grids detected in one photo -> list[FaceRead] for every face the photo shows (detected or predicted)"""
    letters = [classify(features(g.colours[1, 1][None]), centroids)[0][0] for g in grids]
    best, mirror_seen = None, False
    viable = {}
    for i, j in itertools.combinations(range(len(grids)), 2):
        if letters[i] == letters[j]:
            continue
        choice, rms, second, M, mirrored_only = fit_subset([grids[i], grids[j]], [letters[i], letters[j]])
        if mirrored_only:
            mirror_seen = mirror_seen or (rms is not None and rms <= MAX_RMS)
        elif rms <= MAX_RMS:
            viable[(i, j)] = (rms, choice, second, M)
    for i, j, k in itertools.combinations(range(len(grids)), 3):
        if len({letters[i], letters[j], letters[k]}) < 3 or not all(p in viable for p in ((i, j), (i, k), (j, k))):
            continue
        sub = (i, j, k)
        choice, rms, second, M, mirrored_only = fit_subset([grids[q] for q in sub], [letters[q] for q in sub])
        if not mirrored_only and rms <= MAX_RMS and (best is None or best[0] < 3 or rms < best[1]):
            best = (3, rms, sub, choice, second, M)
    if best is None and viable:
        (i, j), (rms, choice, second, M) = min(viable.items(), key=lambda kv: kv[1][0])
        best = (2, rms, (i, j), choice, second, M)
    if best is None and mirror_seen:
        raise PhotoError(f"photo {pi + 1}: the faces appear mirrored. Is the photo flipped, or does this cube use a "
                         f"different colour scheme (white opposite yellow, red opposite orange, green opposite blue, "
                         f"with white-red-green clockwise)?")
    if best is None:
        raise PhotoError(f"photo {pi + 1}: could not match the stickers to a cube. Show at least two faces "
                         f"(a corner of the cube), keep it sharp, and fill the frame")
    size, rms, sub, choice, second, M = best
    gs, ls = [grids[i] for i in sub], [letters[i] for i in sub]
    d, _ = camera_facing(M, ls)
    reads = []
    Xall, xall = [], []
    for g, l, t in zip(gs, ls, choice):
        fi = cube.FACES.index(l)
        rgbs, pos, matched = np.zeros((9, 3)), np.zeros((9, 2)), [False] * 9
        for r in range(3):
            for c in range(3):
                c2, r2 = dihedral(t, c, r)
                k = 3 * r2 + c2
                rgbs[k], pos[k], matched[k] = g.colours[r, c], g.px(c, r), (c, r) in g.matched
                if (c, r) in g.matched:
                    Xall.append(STICKER_XYZ[9 * fi + k])
                    xall.append(g.matched[(c, r)].c)
        reads.append(FaceRead(l, rgbs, pos, matched, g.area() * sum(matched) / 9, "detected", pi))
    # faces the camera also sees but the grid finder missed: predict them from the camera
    Xall, xall = np.array(Xall), np.array(xall)
    if len(Xall) >= 16:
        P = dlt(Xall, xall)
        pr, depth = project(P, Xall)
        if np.all(depth > 0) and np.sqrt(np.mean(np.sum((pr - xall) ** 2, axis=1))) < np.sqrt(np.mean(np.sum((np.hstack([Xall, np.ones((len(Xall), 1))]) @ M - xall) ** 2, axis=1))) * 1.5:
            cam = lambda pts: project(P, pts)[0]
        else:
            cam = None
    else:
        cam = None
    if cam is None:
        cam = lambda pts: np.hstack([np.atleast_2d(pts), np.ones((len(np.atleast_2d(pts)), 1))]) @ M
    for f in cube.FACES:
        if f in ls or FACE_NORMAL[f] @ d < 0.3:
            continue
        fi = cube.FACES.index(f)
        rgbs, pos, ok = np.zeros((9, 3)), np.zeros((9, 2)), True
        for k in range(9):
            c3 = STICKER_XYZ[9 * fi + k]
            pos[k] = cam(c3)[0]
            s = sample_polygon(rgb, cam(np.array(_inset_corners(f, c3, 0.22))))
            if s is None:
                ok = False
                break
            rgbs[k] = s
        if ok:
            area = float(cv2.contourArea(np.array([cam(STICKER_XYZ[9 * fi + k])[0] for k in (0, 2, 8, 6)], np.float32)))
            reads.append(FaceRead(f, rgbs, pos, [False] * 9, area * 0.5, "predicted", pi))
    return reads, {"photo": pi, "faces": [r.letter for r in reads], "fit_rms": round(rms, 3),
                   "ambiguity": None if second is None or rms < 1e-6 else round(second / rms, 2)}


# ---------------------------------------------------------------- 6. assemble the 54 stickers
def assign_exact(feats, centroids, centres):
    """nine stickers per colour: min-cost assignment of the 48 non-centre stickers to 8 slots per colour.
    feats: (54, d); centres: {facelet index: letter} are fixed. -> (letters list, margins)"""
    ls = list(cube.FACES)
    C = np.array([centroids[f] for f in ls])
    free = [i for i in range(54) if i not in centres]
    d = np.sqrt(((feats[free][:, None] - C[None]) ** 2).sum(-1))         # (48, 6)
    cost = np.repeat(d, 8, axis=1)                                          # 8 slots per colour
    col = hungarian(cost)
    out = [None] * 54
    for i, l in centres.items():
        out[i] = l
    for r, i in enumerate(free):
        out[i] = ls[col[r] // 8]
    # margin: distance to the best other colour minus distance to the assigned one (small = unsure)
    margin = np.full(54, np.inf)
    for r, i in enumerate(free):
        a = ls.index(out[i])
        others = np.delete(d[r], a)
        margin[i] = float(others.min() - d[r][a])
    return out, margin


def read_photos(images, accept=None, max_swap=12):
    """list of BGR images -> (facelets, info). `accept(facelets)` should return True for an acceptable
    (decodable) cube; it is only used to try swapping uncertain stickers when the first reading fails."""
    if len(images) < 2:
        raise PhotoError("need at least two photos so that all six faces are visible")
    photos = []
    for pi, img in enumerate(images):
        small = shrink(img)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        grids = []
        for g in find_grids(find_candidates(small), max_grids=5, rgb=rgb):
            try:
                sample_grid(rgb, g)
                grids.append(g)
            except PhotoError:
                pass
        if len(grids) < 2:
            raise PhotoError(f"photo {pi + 1}: found {len(grids)} face(s) of the cube. Each photo must show at least two "
                             f"faces (a corner), sharp, with the cube filling a good part of the frame and no cube-like "
                             f"clutter behind it")
        photos.append((small, rgb, grids))
    use = [g for _, _, gs in photos for g in gs if not g.uniform] or [g for _, _, gs in photos for g in gs]
    sample = np.vstack([np.array([g.colours[r, c] for (c, r) in g.matched]) for g in use])
    centroids = cluster_colours(sample)
    reads, per_photo = [], []
    for pi, (small, rgb, grids) in enumerate(photos):
        r, info = read_photo(pi, rgb, grids, centroids)
        reads += r
        per_photo.append(info)
    best = {}
    for r in reads:
        if r.letter not in best or r.score > best[r.letter].score:
            best[r.letter] = r
    missing = [COLOUR_NAME[f] for f in cube.FACES if f not in best]
    if missing:
        raise PhotoError("no photo shows the " + " and ".join(missing) + " centre face"
                         + ("s" if len(missing) > 1 else "") + ". Take photos of two opposite corners of the cube so all six faces appear")
    rgb54 = np.zeros((54, 3))
    for k, f in enumerate(cube.FACES):
        rgb54[9 * k:9 * k + 9] = best[f].rgb
    centres = {9 * k + 4: f for k, f in enumerate(cube.FACES)}
    feats = features(rgb54)
    for _ in range(3):                       # settle the colour clusters on all 54 stickers, then count-constrain
        letters, margin = assign_exact(feats, centroids, centres)
        for f in cube.FACES:
            idx = [i for i in range(54) if letters[i] == f]
            centroids[f] = feats[idx].mean(axis=0)
    letters, margin = assign_exact(feats, centroids, centres)
    facelets = "".join(letters)
    info = {"photos": per_photo, "faces_from": {f: best[f].photo + 1 for f in cube.FACES},
            "predicted_faces": [f for f in cube.FACES if best[f].source == "predicted"],
            "min_margin": round(float(np.min(margin[np.isfinite(margin)])), 2), "corrected": [],
            "uncertain": [int(i) for i in np.argsort(margin)[:max_swap] if np.isfinite(margin[i])]}
    ok = _try_accept(accept, facelets)
    if ok is False and accept is not None:
        cand = info["uncertain"]
        for a, b in itertools.combinations(cand, 2):
            if facelets[a] == facelets[b]:
                continue
            f2 = list(facelets)
            f2[a], f2[b] = f2[b], f2[a]
            f2 = "".join(f2)
            if _try_accept(accept, f2):
                info["corrected"] = [a, b]
                facelets = f2
                break
    info["overlays"] = [draw_overlay(small, [r for r in reads if r.photo == pi], facelets, best)
                        for pi, (small, rgb, grids) in enumerate(photos)]
    return facelets, info


def _try_accept(accept, facelets):
    if accept is None:
        return None
    try:
        return bool(accept(facelets))
    except ValueError:
        return False


# ---------------------------------------------------------------- 7. debug overlay
def draw_overlay(small, reads, facelets, best):
    import base64
    out = small.copy()
    ring = {"U": (240, 240, 240), "R": (60, 60, 230), "F": (60, 190, 60), "D": (30, 220, 240), "L": (40, 140, 250), "B": (230, 130, 40)}
    for r in reads:
        fi = cube.FACES.index(r.letter)
        for k in range(9):
            x, y = int(round(r.pos[k][0])), int(round(r.pos[k][1]))
            letter = facelets[9 * fi + k]
            rad = max(6, int(0.18 * math.sqrt(max(r.score, 100) / 4)))
            cv2.circle(out, (x, y), rad, ring[letter], -1)
            cv2.circle(out, (x, y), rad, (0, 0, 0), 1)
            cv2.putText(out, letter, (x - 4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()
