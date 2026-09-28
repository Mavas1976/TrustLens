"""
trustlens._palette
==================
The TrustLens brand colours, free of any plotting dependency.

``trustlens.visualization.style`` re-exports ``BRAND_COLORS``; scoring and
HTML rendering import it from here so ``import trustlens`` does not load
matplotlib (TL-30).
"""

BRAND_COLORS: dict[str, str] = {
    "blue": "#4B8BF5",
    "orange": "#F5784B",
    "green": "#34C759",
    "red": "#FF3B30",
    "amber": "#FF9F0A",
    "purple": "#AF52DE",
    "pink": "#FF2D55",
    "cyan": "#5AC8FA",
    "deep_orange": "#FF6B35",
    "gray": "#8E8E93",
    "light_gray": "#CCCCCC",
    "muted_gray": "#AAAAAA",
    "text_dark": "#444444",
    "text_muted": "#666666",
    "text_subtle": "#888888",
    "light": "#F2F2F7",
    "white": "#FFFFFF",
    "dark": "#1C1C1E",
}

GRADE_COLORS: dict[str, str] = {
    "A": BRAND_COLORS["green"],
    "B": BRAND_COLORS["blue"],
    "C": BRAND_COLORS["amber"],
    "D": BRAND_COLORS["red"],
}


def color_for_grade(grade: str) -> str:
    """Colour of a letter grade; grey for N/A or unknown grades."""
    return GRADE_COLORS.get(grade, BRAND_COLORS["gray"])


def color_for_score(score: float) -> str:
    """Colour of a 0–100 score, using the grade bands."""
    if score >= 80:
        return BRAND_COLORS["green"]
    if score >= 60:
        return BRAND_COLORS["blue"]
    if score >= 40:
        return BRAND_COLORS["amber"]
    return BRAND_COLORS["red"]
