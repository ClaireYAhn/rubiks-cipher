"""Simulate phone photos of the encoder screen (random perspective, bezel, blur, noise, colour cast) and read them back.
usage: python test_cube_screen.py front.png back.png FACELETS"""
import sys
import cv2
import numpy as np
import cube_screen, cube_codec

front, back, F = cv2.imread(sys.argv[1]), cv2.imread(sys.argv[2]), sys.argv[3]
table = cube_codec.load_table()
def accept(f):
    cube_codec.state_to_words(f, table); return True

def phone_photo(page, rng, W=1200, H=900):
    """page screenshot -> photo: dark laptop bezel + room, random perspective, cast, blur, noise, jpeg"""
    h, w = page.shape[:2]
    bez = 40
    screen = cv2.copyMakeBorder(page, bez, bez, bez, bez, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    sh, sw = screen.shape[:2]
    img = np.full((H, W, 3), rng.uniform(90, 200), np.uint8)
    for _ in range(6):     # room clutter
        x0, y0 = rng.uniform(0, W), rng.uniform(0, H)
        cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + rng.uniform(50, 400)), int(y0 + rng.uniform(50, 400))), tuple(float(v) for v in rng.uniform(30, 230, 3)), -1)
    # destination quad: the screen fills 55-85% of the frame, tilted
    fill = rng.uniform(0.55, 0.85)
    cx, cy = W / 2 + rng.uniform(-60, 60), H / 2 + rng.uniform(-40, 40)
    s = fill * H / sh
    base = np.array([[-sw / 2, -sh / 2], [sw / 2, -sh / 2], [sw / 2, sh / 2], [-sw / 2, sh / 2]]) * s
    rot = np.radians(rng.uniform(-12, 12))
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    dst = base @ R.T + np.array([cx, cy]) + rng.uniform(-0.06, 0.06, (4, 2)) * np.array([sw, sh]) * s
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]]), dst.astype(np.float32))
    warped = cv2.warpPerspective(screen, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT, dst=img.copy())
    out = warped.astype(np.float32) * rng.uniform(0.85, 1.1, 3) * rng.uniform(0.8, 1.05)
    out = cv2.GaussianBlur(np.clip(out, 0, 255), (0, 0), rng.uniform(0.6, 1.8))
    out = np.clip(out + rng.normal(0, rng.uniform(2, 6), out.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 92))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

# 1) the clean screenshots themselves
got, info = cube_screen.read_screen_photos([front, back], accept=accept)
print("clean screenshots:", "OK" if got == F else "MISMATCH", info)
# 2) simulated phone photos
rng = np.random.default_rng(0)
ok = 0; N = 20; fails = {}
for t in range(N):
    imgs = [phone_photo(front, rng), phone_photo(back, rng)]
    if t % 2: imgs.reverse()
    try:
        got, info = cube_screen.read_screen_photos(imgs, accept=accept)
        if got == F: ok += 1
        else: fails["wrong state"] = fails.get("wrong state", 0) + 1
    except ValueError as e:
        fails[str(e)[:70]] = fails.get(str(e)[:70], 0) + 1
    if t == 0:
        cv2.imwrite("/private/tmp/claude-501/-Users-yooiahn/620d41f7-8e25-4244-a165-f348965de03d/scratchpad/shots/sim_phone.jpg", imgs[0])
print(f"simulated phone photos: {ok}/{N} exact; failures: {fails}")
