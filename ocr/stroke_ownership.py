"""Pixel-supported glyph boxes; CTC anchors guide ownership, not box size."""
from heapq import heappop, heappush

import cv2
import numpy as np

from .char_boxes import _as_grayscale, estimate_border_background_luminance


def _split_joined(mask, anchors):
    """Multi-source geodesic labelling confined to a joined ink component.

    Travel through thin necks costs more than through stroke interiors. No
    background is grown into; Voronoi fronts compete inside the foreground.
    """
    if mask.size > 40000:
        scale = (40000.0 / mask.size) ** 0.5
        shape = (max(2, int(mask.shape[1] * scale)), max(2, int(mask.shape[0] * scale)))
        reduced = cv2.resize(mask.astype(np.uint8), shape, interpolation=cv2.INTER_NEAREST)
        if not reduced.any():
            owners = np.full(mask.shape, -1, np.int32)
        else:
            small = _split_joined(reduced, [(i, x * shape[0] / mask.shape[1]) for i, x in anchors])
            owners = cv2.resize(small, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Sampling can miss a thin branch; retain every original foreground
        # pixel, assigning only these unresolved tips to the nearest seed.
        yy, xx = np.nonzero(mask & (owners < 0))
        if len(xx):
            nearest = np.argmin(np.abs(xx[:, None] - np.array([x for _, x in anchors])), axis=1)
            owners[yy, xx] = np.array([i for i, _ in anchors])[nearest]
        owners[~mask.astype(bool)] = -1
        return owners
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    ys, xs = np.nonzero(mask)
    owners = np.full(mask.shape, -1, np.int32)
    costs = np.full(mask.shape, np.inf, np.float32)
    queue = []
    for owner, anchor in anchors:
        gap = max(3.0, mask.shape[0] * 0.3)
        score = distance[ys, xs] / (1.0 + ((xs - anchor) / gap) ** 2)
        # Choose an interior pixel near this anchor as a seed.
        order = np.argsort(-score)
        for candidate in order:
            y, x = int(ys[candidate]), int(xs[candidate])
            if costs[y, x] != 0:
                costs[y, x] = 0
                owners[y, x] = owner
                heappush(queue, (0.0, owner, y, x))
                break
    height, width = mask.shape
    while queue:
        cost, owner, y, x = heappop(queue)
        if cost > costs[y, x] + 1e-5 or owners[y, x] != owner:
            continue
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            yy, xx = y + dy, x + dx
            if not (0 <= yy < height and 0 <= xx < width) or not mask[yy, xx]:
                continue
            step = (1.4142 if dx and dy else 1.0) / (0.5 + distance[yy, xx])
            updated = float(np.float32(cost + step))
            if updated < costs[yy, xx] - 1e-5:
                costs[yy, xx] = updated
                owners[yy, xx] = owner
                heappush(queue, (updated, owner, yy, xx))
    return owners


def _group_parts(parts, anchors, scales):
    """Ordered dynamic programming groups disconnected strokes globally.

    A component is kept whole unless it was identified as a joined glyph.
    Competing neighbouring anchors are considered together rather than making
    independent nearest-centroid decisions for each radical.
    """
    n, m = len(anchors), len(parts)
    parts.sort(key=lambda part: part["cx"])
    costs = np.full((n + 1, m + 1), np.inf)
    costs[0, 0] = 0.0
    parents = {}
    # A group can contain several disconnected radicals, dots and serifs.
    max_parts = max(16, int(np.ceil(m / max(n, 1))) + 8)
    groups = {}
    for end in range(1, m + 1):
        for start in range(max(0, end - max_parts), end):
            subset = parts[start:end]
            boxes = np.array([part["box"] for part in subset])
            area = sum(part["area"] for part in subset)
            groups[start, end] = (
                boxes[:, 0].min(), boxes[:, 2].max(),
                sum(part["cx"] * part["area"] for part in subset) / area,
                {part["owner"] for part in subset if part["owner"] is not None},
            )
    for i in range(1, n + 1):
        anchor, scale = anchors[i - 1], max(3.0, scales[i - 1])
        for end in range(m + 1):
            # Missing visual evidence is permitted; never create a large box
            # by attaching an unrelated neighbour just to satisfy token count.
            costs[i, end] = costs[i - 1, end] + 6.0
            parents[i, end] = end
            for start in range(max(0, end - max_parts), end):
                if not np.isfinite(costs[i - 1, start]):
                    continue
                left, right, cx, fixed = groups[start, end]
                if fixed and fixed != {i - 1}:
                    continue
                middle = (left + right) / 2
                position = 3.0 * ((middle - anchor) / scale) ** 2
                position += 0.5 * ((cx - anchor) / scale) ** 2
                excessive_span = 4.0 * max(0.0, (right - left) / scale - 1.35) ** 2
                candidate = costs[i - 1, start] + position + excessive_span
                if candidate < costs[i, end]:
                    costs[i, end] = candidate
                    parents[i, end] = start
    grouped = [[] for _ in anchors]
    if not np.isfinite(costs[n, m]):
        # Conflicting split hints: keep their ownership; independently assign
        # remaining intact components rather than losing visible strokes.
        for part in parts:
            owner = part["owner"]
            if owner is None:
                owner = int(np.argmin(np.abs(anchors - part["cx"])))
            grouped[owner].append(part)
        return grouped
    end = m
    for i in range(n, 0, -1):
        start = parents[i, end]
        grouped[i - 1] = parts[start:end]
        end = start
    return grouped


def _foreground_evidence(image, typical_spacing, anchors=None, core_bounds=None, *, weak_ratio=0.08):
    """Signed local contrast excludes paper shading; morphology rejects rules.

    Absolute distance from one corner colour can classify brighter paper as
    ink. Closing/opening estimates the local paper level for each polarity.
    Long ruling lines are structural evidence, not character components.
    """
    gray = _as_grayscale(image)
    height, width = gray.shape
    # Expanded context can include pale paper around a dark rubbing. Determine
    # polarity from the actual text core, not the external page corners.
    if core_bounds is not None and anchors is not None and len(anchors) >= 4:
        core_top, core_bottom = map(int, core_bounds)
        core = gray[max(0, core_top):min(height, core_bottom)]
        light_background = float(np.median(core)) >= 128 if core.size else True
    else:
        light_background = estimate_border_background_luminance(image) >= 128
    size = max(3, int(round(max(height, typical_spacing) * 1.5)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    if light_background:
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        strength = background.astype(np.float32) - gray.astype(np.float32)
    else:
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        strength = gray.astype(np.float32) - background.astype(np.float32)
    signal = float(np.percentile(strength, 99.5))
    weak = (strength >= max(8.0, signal * weak_ratio)).astype(np.uint8)
    strong = strength >= max(16.0, signal * 0.40)

    # A line extending across several character positions is not a long serif.
    length = max(12, int(np.ceil(typical_spacing * 2.5)))
    rules = np.zeros_like(weak)
    if length <= width:
        rules |= cv2.morphologyEx(weak, cv2.MORPH_OPEN,
                                  np.ones((1, length), np.uint8))
    # End caps of a page border run across the crop's short dimension. Glyphs
    # normally do not occupy almost its entire height including search context.
    vertical_length = max(12, int(np.ceil(height * 0.85)))
    if vertical_length <= height:
        vertical = cv2.morphologyEx(weak, cv2.MORPH_OPEN,
                                    np.ones((vertical_length, 1), np.uint8))
        edge = max(2, int(typical_spacing * 0.45))
        count, components, stats, _ = cv2.connectedComponentsWithStats(vertical, connectivity=8)
        for label in range(1, count):
            x, y, w, h, _ = stats[label]
            if w <= max(2, typical_spacing * 0.08) and (x < edge or x+w > width-edge):
                rules[components == label] = 1
    if light_background and core_bounds is not None and anchors is not None:
        core_top, core_bottom = core_bounds
        # Page borders can be slightly slanted/broken, so a straight morphology
        # kernel alone misses them. Only remove long lines outside the text
        # core or beyond the first/last glyph seed, never interior glyph bars.
        segments = cv2.HoughLinesP(weak, 1, np.pi / 180, threshold=max(12, length//3),
                                   minLineLength=max(12, min(length, int(height*.75))),
                                   maxLineGap=max(2, int(typical_spacing*.25)))
        if segments is not None:
            for x1, y1, x2, y2 in np.asarray(segments).reshape(-1, 4):
                dx, dy = abs(x2-x1), abs(y2-y1)
                horizontal = dx >= length and dy <= 0.08 * max(dx, 1)
                outside_sides = max(y1,y2) < core_top+2 or min(y1,y2) >= core_bottom-2
                vertical = dy >= height * .7 and dx <= .08 * max(dy, 1)
                outside_ends = (max(x1,x2) < anchors[0] - .25*typical_spacing or
                                min(x1,x2) > anchors[-1] + .25*typical_spacing)
                if (horizontal and outside_sides) or (vertical and outside_ends):
                    thicknesses = []
                    for fraction in (.25, .5, .75):
                        x = int(round(x1 + fraction*(x2-x1)))
                        y = int(round(y1 + fraction*(y2-y1)))
                        values = weak[:, x] if horizontal else weak[y, :]
                        position = y if horizontal else x
                        start = end = position
                        while start > 0 and values[start-1]:
                            start -= 1
                        while end+1 < len(values) and values[end+1]:
                            end += 1
                        thicknesses.append(end-start+1)
                    if np.median(thicknesses) <= max(3, typical_spacing*.1):
                        cv2.line(rules, (x1,y1), (x2,y2), 1, thickness=3)
        strong[:int(core_top)] = False
        strong[int(core_bottom):] = False
    if anchors is not None and len(anchors) == 1:
        count, components, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
        total_ink = int(weak.sum())
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if (w > 1.2*typical_spacing and h <= max(2, height*.06)
                    and (y < height*.15 or y+h > height*.85)
                    and total_ink-area > max(4, area*.25)):
                rules[components == label] = 1
    # Remove the line itself, not a dilated band that could erase nearby serifs.
    weak[rules != 0] = 0
    strong[rules != 0] = False
    return weak, strong, signal


def stroke_owned_boxes(image, centers, characters=None, *, return_masks=False, core_bounds=None):
    """Return per-glyph boxes with one-pixel antialias allowance.

    Strong foreground seeds retain connected weak tips (hysteresis). Whole
    components are grouped jointly into characters. Only joined components
    are split, with foreground-constrained geodesic growth from CTC anchors.
    No proportional padding or minimum CJK square is applied.
    """
    gray = _as_grayscale(image)
    height, width = gray.shape
    anchors = np.asarray(centers, dtype=float)
    if not height or not width:
        raise ValueError("image must be non-empty")
    if anchors.ndim != 1 or not np.isfinite(anchors).all() or np.any(np.diff(anchors) <= 0):
        raise ValueError("centers must be finite, sorted and strictly increasing")
    if np.any(anchors < 0) or np.any(anchors >= width):
        raise ValueError("centers must be inside the image")
    if characters is not None and len(characters) != len(anchors):
        raise ValueError("characters and centers must have equal lengths")
    if not len(anchors):
        return ([], []) if return_masks else []
    gaps = np.diff(anchors)
    typical = float(np.median(gaps)) if len(gaps) else min(height, width)
    scales = np.array([min(height, np.median(gaps[max(0, i-1):i+1]))
                       if len(gaps) else typical for i in range(len(anchors))])
    priors = [(max(0.0, x - 0.55 * scale), 0.0,
               min(float(width), x + 0.55 * scale), float(height))
              for x, scale in zip(anchors, scales)]
    active = [i for i in range(len(anchors))
              if characters is None or not characters[i].isspace()]
    if not active:
        return (priors, [None] * len(anchors)) if return_masks else priors
    active_anchors, active_scales = anchors[active], scales[active]
    weak, strong, signal = _foreground_evidence(image, typical, anchors, core_bounds)
    if signal < 16 or weak.mean() > 0.65:
        return (priors, [None] * len(anchors)) if return_masks else priors
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(weak, connectivity=8)
    parts = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        region = labels[y:y+h, x:x+w] == label
        if area < max(2, int(height * height * 0.0005)) or not np.any(strong[y:y+h, x:x+w] & region):
            continue
        near = int(np.argmin(np.abs(active_anchors - centroids[label, 0])))
        if abs(centroids[label, 0] - active_anchors[near]) > max(height, active_scales[near]):
            continue
        inside = np.flatnonzero((active_anchors >= x) & (active_anchors < x + w))
        split = len(inside) > 1 and w > 1.35 * float(np.median(active_scales[inside]))
        if split:
            owners = _split_joined(region, [(int(i), active_anchors[i] - x) for i in inside])
            for owner in inside:
                local = owners == owner
                yy, xx = np.nonzero(local)
                if len(xx):
                    parts.append({"box": (x+xx.min(), y+yy.min(), x+xx.max()+1, y+yy.max()+1),
                                  "cx": float(x+xx.mean()), "area": len(xx),
                                  "owner": int(owner), "origin": (x, y), "mask": local})
        else:
            parts.append({"box": (x, y, x+w, y+h), "cx": centroids[label, 0],
                          "area": area, "owner": None, "origin": (x, y), "mask": region})
    grouped = _group_parts(parts, active_anchors, active_scales)
    boxes, masks = list(priors), [None] * len(anchors)
    for index, group in zip(active, grouped):
        if not group:
            continue
        extents = np.asarray([part["box"] for part in group])
        left, top = extents[:, :2].min(axis=0)
        right, bottom = extents[:, 2:].max(axis=0)
        # Only one source pixel for antialiasing/rounding, regardless of size.
        boxes[index] = (max(0, left-1), max(0, top-1), min(width, right+1), min(height, bottom+1))
        if return_masks:
            mask = np.zeros(gray.shape, dtype=bool)
            for part in group:
                x, y = part["origin"]
                h, w = part["mask"].shape
                mask[y:y+h, x:x+w] |= part["mask"]
            masks[index] = mask
    return (boxes, masks) if return_masks else boxes


def _separated_column_boxes(image, anchors, core_bounds):
    """Prefer observed low-ink gaps for regular, well-separated vertical print.

    This path is accepted only when *every* inter-character gap has little
    ink. The gaps assign pixels; final boxes are their actual ink extents,
    not the intervals themselves. Connected/irregular text uses ownership.
    """
    anchors = np.asarray(anchors, float)
    if len(anchors) < 4:
        return None
    gaps = np.diff(anchors)
    spacing = float(np.median(gaps))
    if spacing < 3 or np.any(gaps < .65*spacing) or np.any(gaps > 1.35*spacing):
        return None
    weak, strong, signal = _foreground_evidence(image, spacing, anchors, core_bounds, weak_ratio=0.16)
    top, bottom = map(int, core_bounds)
    if float(np.median(_as_grayscale(image)[top:bottom])) < 128:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(strong.astype(np.uint8), connectivity=8)
        minimum = max(2, int((bottom-top)**2 * .001))
        accepted = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= minimum)
        accepted = accepted[accepted != 0]
        strong = np.isin(labels, accepted)
        radius = max(1, int(round((bottom-top)*.015)))
        nearby = cv2.dilate(strong.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8))
        weak &= nearby
    profile = strong[top:bottom].sum(axis=0).astype(float)
    profile = np.convolve(profile, np.ones(3)/3, mode="same")
    threshold = max(1.0, float(np.percentile(profile, 90)) * .16)
    width = weak.shape[1]

    def valley(low, high, reference):
        low, high = max(0, int(np.ceil(low))), min(width-1, int(np.floor(high)))
        if low > high:
            return None
        candidates = np.arange(low, high+1)
        candidates = candidates[profile[candidates] <= threshold]
        if not len(candidates):
            return None
        return int(candidates[np.argmin(np.abs(candidates-reference))])

    cuts = []
    first = valley(anchors[0]-.9*spacing, anchors[0]-.25*spacing, anchors[0]-.25*spacing)
    cuts.append(first if first is not None else 0)
    for left, right in zip(anchors, anchors[1:]):
        cut = valley(left+.25*(right-left), left+.75*(right-left), (left+right)/2)
        if cut is None:
            return None
        cuts.append(cut)
    last = valley(anchors[-1]+.25*spacing, anchors[-1]+.9*spacing, anchors[-1]+.25*spacing)
    cuts.append(last if last is not None else width)
    boxes = []
    for left, right in zip(cuts, cuts[1:]):
        region = weak[:, left:right]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
        keep = np.zeros(region.shape, bool)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < 2:
                continue
            component = labels == label
            if np.any(component & strong[:, left:right]):
                keep |= component
        yy, xx = np.nonzero(keep)
        if not len(xx):
            return None
        boxes.append((max(0, left+xx.min()-1), max(0, yy.min()-1),
                      min(width, left+xx.max()+2), min(weak.shape[0], yy.max()+2)))
    return boxes


def refine_stroke_row(image_bgr, row_box, result):
    """Inspect real pixels outside the detector row without padding final boxes.

    Recognition remains on the original crop. Only the evidence search has a
    wider context; the output is the owned-ink extent mapped back by homography.
    """
    characters = result.get("char_list", [])
    if not characters or any("line" not in item for item in characters):
        return
    quad = np.asarray(row_box, np.float32).reshape(4, 2)
    width = int(np.linalg.norm(quad[0] - quad[1]))
    height = int(np.linalg.norm(quad[0] - quad[3]))
    if width < 2 or height < 2:
        return
    if height >= 1.5 * width:
        destination = np.float32([[0, width], [0, 0], [height, 0], [height, width]])
        crop_w, crop_h = height, width
    else:
        destination = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        crop_w, crop_h = width, height
    search_margin = max(3, int(np.ceil(crop_h * 0.18)))
    transform = cv2.getPerspectiveTransform(quad, destination + search_margin)
    centers_original = np.float32([np.asarray(item["line"]).mean(axis=0)
                                   for item in characters]).reshape(-1, 1, 2)
    centers = cv2.perspectiveTransform(centers_original, transform)[:, 0, 0]
    if not np.isfinite(centers).all() or np.any(np.diff(centers) <= 0):
        return
    context = cv2.warpPerspective(
        image_bgr, transform, (crop_w + 2*search_margin, crop_h + 2*search_margin),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    core_bounds = (search_margin, search_margin+crop_h)
    boxes = (_separated_column_boxes(context[:, :, ::-1], centers, core_bounds)
             if height >= 1.5 * width else None)
    if boxes is None:
        boxes = stroke_owned_boxes(context[:, :, ::-1], centers,
                                   [item["char"] for item in characters],
                                   core_bounds=core_bounds)
    inverse = np.linalg.inv(transform)
    for item, (left, top, right, bottom) in zip(characters, boxes):
        rectangle = np.float32([[left, top], [right, top], [right, bottom], [left, bottom]])
        polygon = cv2.perspectiveTransform(rectangle[None], inverse)[0]
        # Preserve the conventional top-left start and cyclic winding.
        polygon = np.roll(polygon, -np.argmin(polygon.sum(axis=1)), axis=0)
        item["bbox"] = polygon
    cursor = 0
    for word in result.get("words", []):
        target = word.get("value", "")
        text = ""
        group = []
        while cursor < len(characters) and len(text) < len(target):
            item = characters[cursor]
            text += item["char"]
            group.append(item["bbox"])
            cursor += 1
        if group and text == target:
            polygon = cv2.boxPoints(cv2.minAreaRect(np.concatenate(group).astype(np.float32)))
            word["bbox"] = np.roll(polygon, -np.argmin(polygon.sum(axis=1)), axis=0)
