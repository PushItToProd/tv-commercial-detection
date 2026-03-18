import cv2

# Two normalised boxes are considered the same candidate if their L2 distance
# in (rx, ry, rw, rh) space is within this threshold.
# 0.05 allows each coordinate to vary by roughly 1–5% of the image dimension.
EPS = 0.05


KNOWN_RECTANGLES = {
    "fox-side-by-side-left": (0.000, 0.239, 0.387, 0.381),
    "fox-side-by-side-right": (0.411, 0.234, 0.542, 0.542),
    "fox-side-by-side-scoreboard": (0.022, 0.029, 0.792, 0.176),
}

KNOWN_AD_RECTANGLE_NAMES = [
    "fox-side-by-side-left",
    "fox-side-by-side-right",
    "fox-side-by-side-scoreboard",
]


def detect_rectangles(img):
    """Detect axis-aligned rectangles in image using edge detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Close gaps in edges so incomplete rectangles become closed contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 5000]


def find_matching_rectangle(
    normalized_box: tuple[float, float, float, float],
    rectangles: list[tuple[int, int, int, int]],
    width: int,
    height: int,
    eps: float = EPS,
) -> list[tuple[int, tuple[int, int, int, int]]]:
    """
    Find all rectangles that match a given normalized box.

    Args:
        normalized_box: (rx, ry, rw, rh) where coordinates are in [0, 1]
        rectangles: List of (x, y, w, h) in pixel coordinates
        width: Image width in pixels
        height: Image height in pixels
        eps: L2 distance threshold for matching (default: EPS)

    Returns:
        List of tuples (index, rectangle) where rectangle matches the normalized box
        within L2 distance eps in normalized space.
    """
    rx, ry, rw, rh = normalized_box
    eps_sq = eps * eps
    matches = []

    for idx, (x, y, w, h) in enumerate(rectangles):
        # Normalize rectangle coordinates to [0, 1]
        rx_norm = x / width
        ry_norm = y / height
        rw_norm = w / width
        rh_norm = h / height

        # Calculate L2 distance in normalized space
        dx = rx - rx_norm
        dy = ry - ry_norm
        dw = rw - rw_norm
        dh = rh - rh_norm
        dist_sq = dx * dx + dy * dy + dw * dw + dh * dh

        if dist_sq <= eps_sq:
            matches.append((idx, (x, y, w, h)))

    return matches


def find_matching_rectangles(
    normalized_boxes: dict[str, tuple[float, float, float, float]],
    rectangles: list[tuple[int, int, int, int]],
    width: int,
    height: int,
    eps: float = EPS,
) -> dict[int, list[tuple[int, int, int, int]]]:
    """
    Find all rectangles that match any of the given normalized boxes.

    Args:
        normalized_boxes: List of (rx, ry, rw, rh) where coordinates are in [0, 1]
        rectangles: List of (x, y, w, h) in pixel coordinates
        width: Image width in pixels
        height: Image height in pixels
        eps: L2 distance threshold for matching (default: EPS)

    Returns:
        Dictionary mapping index of normalized box to list of matching rectangles.
    """
    matches = {}
    for name, box in normalized_boxes.items():
        box_matches = find_matching_rectangle(box, rectangles, width, height, eps)
        if box_matches:
            matches[name] = [rect for _, rect in box_matches]
    return matches


def find_matching_rectangles_in_image(
    img,
    normalized_boxes: dict[str, tuple[float, float, float, float]] = KNOWN_RECTANGLES,
    eps: float = EPS,
):
    height, width = img.shape[:2]
    rectangles = detect_rectangles(img)
    matches = find_matching_rectangles(normalized_boxes, rectangles, width, height, eps)
    return matches


def image_has_known_ad_rectangle(
    img,
    normalized_boxes: dict[str, tuple[float, float, float, float]] = KNOWN_RECTANGLES,
    ad_rectangle_names: list[str] = KNOWN_AD_RECTANGLE_NAMES,
    eps: float = EPS,
) -> str | None:
    matches = find_matching_rectangles_in_image(img, normalized_boxes, eps)
    for name in ad_rectangle_names:
        if name in matches:
            return name
    return None
