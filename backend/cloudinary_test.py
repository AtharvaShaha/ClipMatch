"""
Cloudinary Onboarding Script
Uploads a sample image, fetches metadata, and generates an optimized URL.
"""

import cloudinary
import cloudinary.uploader
import cloudinary.api

# ── Configure Cloudinary with your credentials ──
cloudinary.config(
    cloud_name="ibdqjpgw",
    api_key="774983814479443",
    api_secret="BhzePJxfgd9MeVJxZl-M6Nj23FE",
    secure=True
)

print("=" * 60)
print("  Cloudinary Onboarding Test")
print("=" * 60)

# ── 1. Upload a sample image from Cloudinary's demo domain ──
print("\n[1] Uploading sample image...")
upload_result = cloudinary.uploader.upload(
    "https://res.cloudinary.com/demo/image/upload/getting-started/shoes.jpg",
    public_id="onboarding_test_shoes",
    overwrite=True
)

secure_url = upload_result["secure_url"]
public_id = upload_result["public_id"]
print(f"  Secure URL : {secure_url}")
print(f"  Public ID  : {public_id}")

# ── 2. Get image details (metadata) ──
print("\n[2] Fetching image metadata...")
details = cloudinary.api.resource(public_id)

width = details["width"]
height = details["height"]
img_format = details["format"]
file_size = details["bytes"]

print(f"  Width      : {width} px")
print(f"  Height     : {height} px")
print(f"  Format     : {img_format}")
print(f"  File size  : {file_size} bytes ({file_size / 1024:.1f} KB)")

# ── 3. Generate a transformed (optimized) URL ──
# f_auto = Cloudinary automatically picks the best format (WebP, AVIF, etc.)
# q_auto = Cloudinary automatically adjusts quality for smallest size without visible loss
transformed_url = cloudinary.utils.cloudinary_url(
    public_id,
    fetch_format="auto",   # f_auto — automatic format selection
    quality="auto"         # q_auto — automatic quality optimization
)[0]

print("\n[3] Optimized image URL generated!")
print(f"  Original size : {file_size / 1024:.1f} KB")
print()
print("Done! Click the link below to see the optimized version of the image.")
print("Check the size and the format.")
print()
print(f"  {transformed_url}")
print()
print("=" * 60)
