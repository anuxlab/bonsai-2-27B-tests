"""Deterministic test-data generators (seeded; no network).
 - long-context documents with needles at known positions + distractors
 - vision images drawn with Pillow so the ground truth is known by construction
"""
import base64, io, random
from .util import sha256_bytes

CHARS_PER_TOKEN = 3.8  # conservative estimate for English prose; actual prompt_tokens is recorded from the server

_DEPTS = ["Logistics", "Procurement", "Facilities", "Legal", "Finance", "Support", "Design", "Compliance", "Research", "Training"]
_ITEMS = ["quarterly forecast", "vendor contract", "safety audit", "onboarding checklist", "server migration", "travel policy",
          "inventory count", "brand guideline", "budget memo", "release schedule", "parking permit list", "cafeteria menu"]
_VERBS = ["reviewed", "updated", "archived", "postponed", "approved", "drafted", "circulated", "re-checked"]
_PEOPLE = ["Mara Lindqvist", "Tomas Reyes", "Priya Nair", "Jonas Weber", "Aiko Tanaka", "Samir Haddad", "Lena Kovac", "Omar Fielding"]
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "September", "October", "November"]
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
_TEMPLATES = [
    "On {day}, the {dept} team {verb} the {item} and shared notes with {person}.",
    "{person} asked whether the {item} could wait until the {ord} week of {month}.",
    "According to the {dept} log, the {item} took {n} working days to complete.",
    "The {dept} desk on floor {f} reported {n} open tickets related to the {item}.",
    "A reminder from {person}: the {item} template lives in the shared drive under folder {code}.",
    "After the {day} stand-up, {person} {verb} the {item} for the {dept} group.",
]
_ORD = ["first", "second", "third", "fourth"]


def _filler_sentence(rnd):
    return rnd.choice(_TEMPLATES).format(
        day=rnd.choice(_DAYS), dept=rnd.choice(_DEPTS), verb=rnd.choice(_VERBS), item=rnd.choice(_ITEMS),
        person=rnd.choice(_PEOPLE), ord=rnd.choice(_ORD), month=rnd.choice(_MONTHS), n=rnd.randint(2, 31),
        f=rnd.randint(1, 9), code=f"{rnd.choice('KLMNPQ')}{rnd.randint(10, 99)}")


NEEDLES = {
    "heron": "The vault code for Project Heron is 7-4-1-9-2.",
    "ferry": "The ferry to Marlow Island departs at 06:45 from Pier 9.",
    "ferry_fix": "Correction notice: the ferry to Marlow Island now departs at 07:05 instead of 06:45, still from Pier 9.",
    "lab": "Dr. Imani Okafor's lab budget was approved at 218,000 euros.",
}
DISTRACTORS = [f"The vault code for Project {n} is {c}." for n, c in
               [("Kestrel", "3-8-5-0-6"), ("Osprey", "9-1-2-7-4"), ("Falcon", "5-5-0-3-8"), ("Merlin", "2-6-8-1-1"), ("Harrier", "4-0-9-9-3")]] + \
              ["The ferry to Dunmore Island departs at 07:20 from Pier 4.", "Dr. Lucas Brandt's lab budget was approved at 154,000 euros."]

LC_QUESTIONS = [
    ("q1-start", "What is the vault code for Project Heron? Answer with only the code.", {"type": "exact", "answers": ["74192"], "squash": True}),
    ("q2-correction", "Taking all corrections into account, at what time does the ferry to Marlow Island depart? Answer with only the time (HH:MM).",
     {"type": "exact", "answers": ["0705", "705"], "squash": True}),
    ("q3-end", "What lab budget (in euros) was approved for Dr. Imani Okafor? Answer with only the number.", {"type": "numeric", "answer": 218000}),
    ("q4-middle", "From which pier does the ferry to Marlow Island depart? Answer with only the pier number.", {"type": "numeric", "answer": 9}),
    ("q5-synthesis", "Add together the lab budget approved for Dr. Imani Okafor (in euros) and the pier number of the Marlow Island ferry. Answer with only the number.",
     {"type": "numeric", "answer": 218009}),
]
LC_POS = {"q1-start": "start", "q2-correction": "middle+late", "q3-end": "end", "q4-middle": "middle", "q5-synthesis": "middle+end"}


def build_long_document(tokens, seed):
    rnd = random.Random(seed)
    target = int(tokens * CHARS_PER_TOKEN)
    sents, n = [], 0
    while n < target:
        s = _filler_sentence(rnd)
        sents.append(s)
        n += len(s) + 1
    N = len(sents)
    at = lambda f: min(N, max(0, int(N * f)))
    inserts = [(at(0.05), NEEDLES["heron"]), (at(0.50), NEEDLES["ferry"]), (at(0.80), NEEDLES["ferry_fix"]), (at(0.95), NEEDLES["lab"])]
    for d in DISTRACTORS:
        inserts.append((rnd.randint(1, N - 1), d))
    for pos, s in sorted(inserts, key=lambda x: -x[0]):  # insert from the back so earlier indices stay valid
        sents.insert(pos, s)
    paras, i = [], 0
    while i < len(sents):
        k = rnd.randint(4, 7)
        paras.append(" ".join(sents[i:i + k]))
        i += k
    return "\n\n".join(paras)


def long_context_cases(tokens, seed, label):
    doc = build_long_document(tokens, seed)
    prefix = ("Below is a long internal document. Read it carefully, then answer the question that follows it.\n\n"
              f"=== DOCUMENT START ===\n{doc}\n=== DOCUMENT END ===\n\n")
    out = []
    for qid, q, grade in LC_QUESTIONS:
        out.append({"id": f"lc-{label}-{qid}", "suite": "long_context", "level": {1024: 2, 2048: 3, 4096: 4, 8192: 4}.get(tokens, 3),
                    "category": "needle-" + LC_POS[qid], "prompt": prefix + "Question: " + q, "grade": grade, "thinking": False,
                    "max_tokens": 48, "est_out_tokens": 12, "prefix_group": f"lc-{label}", "context_tokens_target": tokens,
                    "needle_position": LC_POS[qid], "tags": ["long_context", label],
                    "est_prompt_tokens": int(len(prefix) / CHARS_PER_TOKEN) + 40, "reference": None})
    return out


# ------------------------------------------------------------------------------------------ vision
def _pil():
    from PIL import Image, ImageDraw, ImageFont  # imported lazily: only the vision shard needs Pillow
    return Image, ImageDraw, ImageFont


def _font(size):
    _, _, ImageFont = _pil()
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        pass
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _png(img):
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def _count_shapes(p):
    Image, ImageDraw, _ = _pil()
    rnd = random.Random(p.get("seed", 1))
    img = Image.new("RGB", (336, 336), "white")
    d = ImageDraw.Draw(img)
    placed = []

    def spot(r):
        for _ in range(500):
            x, y = rnd.randint(r + 4, 336 - r - 4), rnd.randint(r + 4, 336 - r - 4)
            if all((x - a) ** 2 + (y - b) ** 2 > (r + s + 8) ** 2 for a, b, s in placed):
                placed.append((x, y, r))
                return x, y
        raise RuntimeError("could not place shapes")
    for _ in range(p["red_circles"]):
        x, y = spot(22)
        d.ellipse([x - 22, y - 22, x + 22, y + 22], fill="red")
    for _ in range(p["blue_squares"]):
        x, y = spot(22)
        d.rectangle([x - 20, y - 20, x + 20, y + 20], fill="blue")
    return img


def _single_shape(p):
    Image, ImageDraw, _ = _pil()
    img = Image.new("RGB", (336, 336), p.get("bg", "white"))
    d = ImageDraw.Draw(img)
    c = p["color"]
    if p["shape"] == "triangle":
        d.polygon([(168, 50), (50, 290), (286, 290)], fill=c)
    elif p["shape"] == "circle":
        d.ellipse([60, 60, 276, 276], fill=c)
    else:
        d.rectangle([70, 70, 266, 266], fill=c)
    return img


def _text(p):
    Image, ImageDraw, _ = _pil()
    img = Image.new("RGB", (400, 140), "white")
    ImageDraw.Draw(img).text((20, 40), p["text"], fill="black", font=_font(p.get("size", 48)))
    return img


def _bar_chart(p):
    Image, ImageDraw, _ = _pil()
    img = Image.new("RGB", (336, 336), "white")
    d = ImageDraw.Draw(img)
    vals, labels = p["values"], p.get("labels", list("ABCDEFG")[:len(p["values"])])
    mx, w = max(vals), 336 // (len(vals) + 1)
    for i, (v, lab) in enumerate(zip(vals, labels)):
        x0 = 30 + i * w
        h = int(230 * v / mx)
        d.rectangle([x0, 290 - h, x0 + w - 16, 290], fill=["#3366cc", "#dc3912", "#ff9900", "#109618", "#990099"][i % 5])
        d.text((x0 + 8, 296), lab, fill="black", font=_font(24))
        if p.get("show_values"):
            d.text((x0 + 4, 290 - h - 26), str(v), fill="black", font=_font(22))
    d.line([20, 290, 320, 290], fill="black", width=2)
    return img


def _spatial(p):
    Image, ImageDraw, _ = _pil()
    img = Image.new("RGB", (336, 336), "white")
    d = ImageDraw.Draw(img)
    left, right = ("red circle", "blue square") if not p.get("swap") else ("blue square", "red circle")
    for name, cx in ((left, 70), (right, 266)):
        if "circle" in name:
            d.ellipse([cx - 40, 128, cx + 40, 208], fill="red")
        else:
            d.rectangle([cx - 40, 128, cx + 40, 208], fill="blue")
    return img


def _table(p):
    Image, ImageDraw, _ = _pil()
    rows = p["rows"]
    img = Image.new("RGB", (340, 70 * len(rows) + 20), "white")
    d = ImageDraw.Draw(img)
    f = _font(30)
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            x, y = 10 + c * 160, 10 + r * 70
            d.rectangle([x, y, x + 160, y + 70], outline="black", width=2, fill="#dddddd" if r == 0 else "white")
            d.text((x + 14, y + 18), str(cell), fill="black", font=f)
    return img


def _equation(p):
    Image, ImageDraw, _ = _pil()
    img = Image.new("RGB", (340, 140), "white")
    ImageDraw.Draw(img).text((20, 38), p["text"], fill="black", font=_font(56))
    return img


def _blank(p):
    Image, _, _ = _pil()
    return Image.new("RGB", (336, 336), p.get("color", "#9a9a9a"))


def _scene(p):
    Image, ImageDraw, _ = _pil()
    img = Image.new("RGB", (336, 336), "#87ceeb")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 220, 336, 336], fill="#228b22")
    d.ellipse([230, 30, 300, 100], fill="#ffd400")
    d.rectangle([90, 150, 190, 230], fill="#8b4513")
    d.polygon([(80, 150), (140, 100), (200, 150)], fill="#b22222")
    return img


_GEN = {"count_shapes": _count_shapes, "single_shape": _single_shape, "text": _text, "bar_chart": _bar_chart, "spatial": _spatial,
        "table": _table, "equation": _equation, "blank": _blank, "scene": _scene}


def make_image(spec):
    """spec: {"gen": name, ...params} -> (png_bytes, sha256, data_uri)"""
    img = _GEN[spec["gen"]](spec)
    png = _png(img)
    return png, sha256_bytes(png), "data:image/png;base64," + base64.b64encode(png).decode()
