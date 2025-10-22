import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def make_star_contour(size=200, points=5, inner_ratio=0.45, rotation_deg=0):
    """Create a filled star contour template."""
    center = (size / 2.0, size / 2.0)
    R = size * 0.45
    r = R * inner_ratio
    pts = []
    for i in range(points * 2):
        angle = (i * math.pi) / points
        radius = R if (i % 2 == 0) else r
        x = center[0] + radius * math.cos(angle + math.radians(rotation_deg))
        y = center[1] + radius * math.sin(angle + math.radians(rotation_deg))
        pts.append([int(round(x)), int(round(y))])
    pts = np.array([pts], dtype=np.int32)
    canvas = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(canvas, pts, 255)
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0] if contours else None

def generate_star_templates():
    """Generate multiple star templates for shape matching."""
    templates = []
    for s in (0.6, 0.9, 1.2):
        for rot in range(0, 360, 30):
            c = make_star_contour(size=int(200 * s), points=5, inner_ratio=0.45, rotation_deg=rot)
            if c is not None:
                templates.append(c)
    return templates

def convexity_defect_count(contour):
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return 0
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return 0
    cnt = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        if d > 1000:
            cnt += 1
    return cnt

def star_score(contour, templates):
    """Measure how similar a contour is to a star shape."""
    best_match = float("inf")
    for t in templates:
        try:
            m = cv2.matchShapes(t, contour, cv2.CONTOURS_MATCH_I1, 0.0)
            if m < best_match:
                best_match = m
        except Exception:
            pass

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    verts = len(approx)
    defects = convexity_defect_count(contour)

    verts_penalty = max(0, (8 - verts)) * 0.6
    defects_penalty = max(0, 3 - defects) * 0.4

    combined = best_match + verts_penalty + defects_penalty
    return combined

def count_stars_shape_based(image_path,
                            area_min=100,
                            area_max=80000,
                            score_thresh=1.5):
    """Count stars in any flag image (no color tuning needed)."""
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Image not found:", image_path)
        return

    img = cv2.resize(img, (800, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 160)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    templates = generate_star_templates()

    star_count = 0
    annotated = img.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if area < area_min or area > area_max:
            continue

        x, y, w, h = cv2.boundingRect(c)
        aspect = w / (h + 1e-6)
        if aspect < 0.4 or aspect > 2.5:
            continue

        score = star_score(c, templates)
        if score <= score_thresh:
            star_count += 1
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(annotated, f"Stars Detected: {star_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.imwrite("output_star_annotated.jpg", annotated)
    print(f"✅ Stars detected: {star_count}")
    print("💾 Saved as output_star_annotated.jpg")

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Stars Detected: {star_count}")
    plt.show()


if __name__ == "__main__":
    count_stars_shape_based(r"C:\XboxGames\pythonVS\flag2.jpeg")
