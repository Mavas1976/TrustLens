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
