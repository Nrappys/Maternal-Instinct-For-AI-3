"""Pure snapshot-to-PNG renderer for inline notebook display; never opens a window."""
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLORS = {"bg": "#F3E5D4", "cell": "#FFF5E5", "card": "#FFF8ED", "line": "#EAD8C4",
          "ink": "#76523F", "muted": "#A48A73", "mother": "#D27D59", "child": "#EFB991",
          "child_edge": "#C28B68", "shelter": "#B69377", "food": "#B89E70", "risk": "#C5A16E"}


@lru_cache(maxsize=16)
def _font(size, bold=False):
    for name in (("seguisb.ttf", "leelauib.ttf") if bold else ("segoeui.ttf", "leelawui.ttf")):
        path = Path("C:/Windows/Fonts") / name
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def world_png(world, title="Grid World", note="", z_mother=None):
    """Draw one provided state. Does not create agents, step a world, or train."""
    c = COLORS
    image = Image.new("RGB", (1120, 720), c["bg"])
    draw = ImageDraw.Draw(image)

    def text(xy, value, size=16, fill=None, bold=False, anchor=None):
        draw.text(xy, str(value), font=_font(size, bold), fill=fill or c["ink"], anchor=anchor)

    def short(value, limit):
        value = str(value).replace("\n", " ")
        while value and draw.textlength(value, font=_font(14)) > limit:
            value = value[:-1]
        return value

    text((30, 22), "MATERNAL / HOME", 13, c["mother"], True)
    text((30, 48), title, 28, bold=True)
    text((30, 89), short(note, 1050), 14, c["muted"])
    cell, ox, oy = 92, 26, 126

    def center(pos):
        return ox + (pos[0] + .5) * cell, oy + (pos[1] + .5) * cell

    walls = set(map(tuple, world["walls"]))
    for y in range(5):
        for x in range(7):
            left, top = ox + x * cell, oy + y * cell
            draw.rounded_rectangle((left+3, top+3, left+cell-3, top+cell-3), 17,
                                   fill=c["cell"], outline=c["line"], width=1)
            if (x, y) in walls:
                draw.line((left+35, top+46, left+57, top+46), fill="#CEB59D", width=4)
    for pos in world["shelters"]:
        x, y = center(pos)
        draw.rounded_rectangle((x-17, y-8, x+17, y+9), 4, outline=c["shelter"], width=2)
        draw.rectangle((x-13, y-5, x-3, y+5), outline=c["shelter"], width=1)
        draw.line((x-17, y+9, x-17, y+15), fill=c["shelter"], width=2)
        draw.line((x+17, y+9, x+17, y+15), fill=c["shelter"], width=2)
    for pos in world.get("workstations", []):
        x, y = center(pos)
        draw.rounded_rectangle((x-12, y-18, x+12, y-2), 2, outline=c["mother"], width=2)
        draw.line((x-19, y+3, x+19, y+3), fill=c["shelter"], width=3)
        draw.line((x-14, y+3, x-14, y+18), fill=c["shelter"], width=2)
        draw.line((x+14, y+3, x+14, y+18), fill=c["shelter"], width=2)
        text((x-34, y-35), "W", 12, c["mother"], True)
    for pos in world.get("play_areas", []):
        x, y = center(pos)
        draw.ellipse((x-13, y-13, x+13, y+13), outline=c["child_edge"], width=2)
        draw.arc((x-7, y-13, x+7, y+13), 70, 290, fill=c["child_edge"], width=1)
        draw.line((x-12, y, x+12, y), fill=c["child_edge"], width=1)
    for x, y, ready in world["food"]:
        x, y = center((x, y))
        if ready:
            draw.ellipse((x-13, y-15, x+1, y+2), fill="#CFB78A")
            draw.ellipse((x-1, y-19, x+13, y-2), fill=c["food"])
            draw.line((x, y-3, x, y+15), fill=c["food"], width=2)
        else:
            draw.ellipse((x-3, y-3, x+3, y+3), outline="#DDC9AE", width=1)
    for pos in world["hazards"]:
        x, y = center(pos)
        points = [(x, y-15), (x+15, y), (x, y+15), (x-15, y)]
        draw.polygon(points, fill="#F4E6CE")
        draw.line(points+[points[0]], fill=c["risk"], width=2)
        text((x, y-1), "!", 17, c["risk"], anchor="mm")
    shared = all(world["mother"][axis] == world["child"][axis] for axis in ("x", "y"))
    for role, letter, radius in (("mother", "M", 23), ("child", "C", 18)):
        body = world[role]
        x, y = center((body["x"], body["y"]))
        if shared:
            x += -17 if role == "mother" else 17
        alive = body["hp"] > 0
        draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                     fill=c[role] if alive else "#D8C8B7",
                     outline="#FFFAF0" if role == "mother" else c["child_edge"],
                     width=3 if role == "mother" else 2)
        text((x, y-1), letter if alive else "-", 20,
             "#FFFAF0" if role == "mother" else c["ink"], anchor="mm")
        if alive and body["signal"] not in ("quiet", "accept"):
            draw.ellipse((x+13, y-32, x+31, y-14), fill=c["cell"], outline=c["line"])
            text((x+22, y-24), "!", 12, c["risk"], anchor="mm")
    text((32, 605), "M Mother   C Child   leaf Food   bed Rest   W Work   ball Play", 14, c["muted"])
    text((32, 637), f"Day {world['day']} / {world['days']}    Step {world['step']:,} / {world['horizon']:,}", 16, bold=True)
    work_steps = int(world.get("stats", {}).get("mother_work_steps", 0))
    work_reward = world.get("stats", {}).get("mother_work_reward", 0)
    text((32, 672), f"Work: {work_steps:,} steps / reward {work_reward:.2f}", 13, c["muted"])

    for index, (role, label) in enumerate((("mother", "Mother"), ("child", "Child"))):
        body, top = world[role], 128 + index * 236
        color = c["mother"] if role == "mother" else c["child_edge"]
        draw.rounded_rectangle((704, top, 1090, top+220), 17, fill=c["card"])
        text((725, top+15), label, 23, color, True)
        text((1068, top+25), f"Age {body['age_steps']/world['steps_per_day']:.1f} d", 14, c["muted"], anchor="rm")
        statuses = (("Health", "hp"), ("Food", "energy"),
                    ("Fatigue", "fatigue"), ("Stress", "fear"))
        for row, (label, key) in enumerate(statuses):
            y = top+66+row*29
            value = max(0, min(100, float(body[key])))
            text((725, y-10), label, 14, c["muted"])
            draw.rounded_rectangle((799, y, 1025, y+5), 2, fill=c["line"])
            if value > 0:
                draw.rounded_rectangle((799, y, 799+226*value/100, y+5), 2, fill=color)
            text((1069, y+2), f"{value:.0f}", 14, anchor="rm")
        action = body["last_action"].replace("_", " ").title() if body["hp"] > 0 else "Not alive"
        text((725, top+185), f"Food items: {body['food']} / {action}", 14, color)
    events = world.get("events", [])
    event = events[-1]["text"] if events else "A quiet moment in the world."
    # Wrap event text to keep the canvas within the notebook width.
    words, line, y = str(event).split(), "", 615
    for word in words:
        candidate = f"{line} {word}".strip()
        if draw.textlength(candidate, font=_font(14)) > 360:
            text((714, y), line, 14, c["muted"])
            line, y = word, y+20
            if y > 650:
                break
        else:
            line = candidate
    if y <= 650:
        text((714, y), line, 14, c["muted"])
    if z_mother is not None:
        text((714, 679), short("z: " + ", ".join(f"{v:.2f}" for v in z_mother), 360), 14, c["muted"])
    data = BytesIO()
    image.save(data, format="PNG")
    return data.getvalue()
