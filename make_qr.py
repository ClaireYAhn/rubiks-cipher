"""usage: python make_qr.py https://your-public-address/   -> writes decode_qr.png (print it or put it on a slide)"""
import sys
import qrcode

url = sys.argv[1]
img = qrcode.make(url, box_size=12, border=2)
out = sys.argv[2] if len(sys.argv) > 2 else "decode_qr.png"
img.save(out)
print(f"wrote {out} for {url}")
