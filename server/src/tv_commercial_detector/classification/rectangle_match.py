import cv2
from cv2.typing import MatLike, Rect

# Two normalised boxes are considered the same candidate if their L2 distance
# in (rx, ry, rw, rh) space is within this threshold.
# 0.05 allows each coordinate to vary by roughly 1–5% of the image dimension.
EPS = 0.05


# TODO: add more known rectangles for other networks and ad break styles.

# TODO: move network-specific details into per-network modules under the
# `classifiers` package.
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


def detect_rectangles(img: MatLike) -> list[Rect]:
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
) -> list[tuple[int, Rect]]:
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
    img: MatLike,
    normalized_boxes: dict[str, tuple[float, float, float, float]] = KNOWN_RECTANGLES,
    eps: float = EPS,
):
    height, width = img.shape[:2]
    rectangles = detect_rectangles(img)
    matches = find_matching_rectangles(normalized_boxes, rectangles, width, height, eps)
    return matches


def image_has_known_ad_rectangle(
    img: MatLike,
    normalized_boxes: dict[str, tuple[float, float, float, float]] = KNOWN_RECTANGLES,
    ad_rectangle_names: list[str] = KNOWN_AD_RECTANGLE_NAMES,
    eps: float = EPS,
) -> str | None:
    matches = find_matching_rectangles_in_image(img, normalized_boxes, eps)
    for name in ad_rectangle_names:
        if name in matches:
            return name
    return None


def border_edge_fractions(
    gray: MatLike,
    box: tuple[int, int, int, int],
    min_step: int = 12,
    slack: int = 6,
    inset: int = 20,
) -> tuple[float, float, float, float]:
    """
    Measure how much of each side of a known box is a visible edge.

    `detect_rectangles` needs a panel to come out as its own closed contour,
    which fails when the panel touches other graphics or sits on a textured
    background. This asks the narrower question instead: at this exact box, is
    there a brightness step along each side?

    Args:
        gray: Single-channel image.
        box: (x0, x1, y0, y1) in pixels.
        min_step: Brightness difference across the side that counts as an edge.
        slack: How far either way the side may sit from the nominal position.
        inset: Length trimmed from each end of a side, to stay clear of corners.

    Returns:
        (left, right, top, bottom), each the fraction of positions along that
        side showing a step of at least `min_step`.
    """
    x0, x1, y0, y1 = box
    g = gray.astype("int16")

    def vertical(x: int) -> float:
        rows = slice(y0 + inset, y1 - inset)
        step = abs(g[rows, x - slack + 3 : x + slack + 1] - g[rows, x - slack : x + slack - 2])
        return float((step.max(axis=1) > min_step).mean())

    def horizontal(y: int) -> float:
        cols = slice(x0 + inset, x1 - inset)
        step = abs(g[y - slack + 3 : y + slack + 1, cols] - g[y - slack : y + slack - 2, cols])
        return float((step.max(axis=0) > min_step).mean())

    return vertical(x0), vertical(x1), horizontal(y0), horizontal(y1)
