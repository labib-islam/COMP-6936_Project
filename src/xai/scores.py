# Scores that measure how well a heatmap points at the tumor.
# Every function takes three 384x384 arrays:
#   heatmap  values >= 0 (higher = more important)
#   mask     True inside the tumor
#   region   True on the real image, False on the letterbox padding (padding is ignored)
# Import with:
#   from src.xai.scores import score_heatmap
# Run directly to test the scores on made-up heatmaps:
#   uv run python -m src.xai.scores

import numpy as np


def area_share(mask, region):
    # fraction of the real image covered by the tumor
    return mask[region].mean()


def pointing_hit(heatmap, mask, region):
    # True if the hottest pixel (inside the real image) lies inside the tumor
    if heatmap[region].max() <= 0:
        return False   # an empty map points nowhere
    inside_only = np.where(region, heatmap, -np.inf)
    peak = np.unravel_index(np.argmax(inside_only), heatmap.shape)
    return bool(mask[peak])


def energy_in_mask(heatmap, mask, region):
    # fraction of the heatmap's total value that falls inside the tumor
    total = heatmap[region].sum()
    if total <= 0:
        return 0.0
    return heatmap[mask & region].sum() / total


def score_heatmap(heatmap, mask, region):
    # all scores for one heatmap.
    # energy_ratio = energy / tumor area share: 1 means no better than a uniform map,
    # 10 means the heatmap is 10 times more concentrated on the tumor than chance.
    share = area_share(mask, region)
    energy = energy_in_mask(heatmap, mask, region)
    return {
        "pointing_hit": pointing_hit(heatmap, mask, region),
        "energy": energy,
        "energy_ratio": energy / share,
    }


if __name__ == "__main__":
    # made-up example: a 100x100 image with padding on the left and right, and a square tumor
    region = np.zeros((100, 100), dtype=bool)
    region[:, 10:90] = True
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:50, 40:50] = True
    share = area_share(mask, region)   # 100 / 8000 = 0.0125

    # 1. a heatmap equal to the mask: hit, all energy inside
    s = score_heatmap(mask.astype(float), mask, region)
    assert s["pointing_hit"] and np.isclose(s["energy"], 1.0) and np.isclose(s["energy_ratio"], 1 / share)

    # 2. a heatmap only outside the tumor: miss, no energy inside
    outside = np.zeros((100, 100))
    outside[70:80, 70:80] = 1.0
    s = score_heatmap(outside, mask, region)
    assert not s["pointing_hit"] and s["energy"] == 0.0

    # 3. a uniform heatmap: energy equals the tumor's area share, so the ratio is 1
    s = score_heatmap(np.ones((100, 100)), mask, region)
    assert np.isclose(s["energy"], share) and np.isclose(s["energy_ratio"], 1.0)

    # 4. the padding is ignored: a peak in the padding doesn't count
    padded_peak = mask.astype(float) * 0.5
    padded_peak[50, 5] = 1.0   # column 5 is padding
    s = score_heatmap(padded_peak, mask, region)
    assert s["pointing_hit"] and np.isclose(s["energy"], 1.0)

    # 5. an empty heatmap: miss, no energy
    s = score_heatmap(np.zeros((100, 100)), mask, region)
    assert not s["pointing_hit"] and s["energy"] == 0.0

    print("All score tests passed")
