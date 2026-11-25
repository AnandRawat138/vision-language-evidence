"""Controlled vocabularies used for normalisation and evaluation.

All lexicons are hand-written for this project and versioned with the code
(``LEXICON_VERSION``).  They are used identically for every method that is
evaluated, so that differences between methods are not caused by different
normalisation procedures.

COCO synonyms follow the spirit of the CHAIR metric (Rohrbach et al., 2018):
a free-text noun is mapped to one of the 80 COCO categories when it is a
listed synonym; nouns that do not map are ignored for the object metrics.
"""

from __future__ import annotations

import re
from functools import lru_cache

LEXICON_VERSION = "lex-v1"

# ---------------------------------------------------------------------------
# COCO categories and synonyms (singular forms; plural handled by _singular).
# ---------------------------------------------------------------------------
COCO_SYNONYMS: dict[str, list[str]] = {
    "person": ["person", "man", "woman", "boy", "girl", "child", "kid", "people", "player",
               "skier", "surfer", "skateboarder", "rider", "lady", "guy", "baby", "adult",
               "pedestrian", "toddler", "teenager", "men", "women", "children", "batter",
               "catcher", "pitcher", "umpire", "worker", "officer", "chef", "cyclist",
               "snowboarder", "tennis player", "baseball player", "spectator", "crowd",
               "individual", "human", "student", "passenger", "customer", "gentleman"],
    "bicycle": ["bicycle", "bike", "cycle"],
    "car": ["car", "automobile", "sedan", "taxi", "suv", "van", "cab", "vehicle", "jeep", "minivan"],
    "motorcycle": ["motorcycle", "motorbike", "scooter", "moped", "dirt bike"],
    "airplane": ["airplane", "plane", "jet", "aircraft", "airliner", "jetliner"],
    "bus": ["bus", "coach", "school bus", "double decker"],
    "train": ["train", "locomotive", "tram", "subway", "railcar", "streetcar"],
    "truck": ["truck", "pickup", "lorry", "fire truck", "pickup truck"],
    "boat": ["boat", "ship", "canoe", "kayak", "sailboat", "yacht", "ferry", "raft"],
    "traffic light": ["traffic light", "traffic signal", "stop light", "stoplight"],
    "fire hydrant": ["fire hydrant", "hydrant"],
    "stop sign": ["stop sign"],
    "parking meter": ["parking meter", "meter"],
    "bench": ["bench"],
    "bird": ["bird", "pigeon", "seagull", "gull", "duck", "parrot", "eagle", "crow", "sparrow", "goose", "swan"],
    "cat": ["cat", "kitten", "kitty"],
    "dog": ["dog", "puppy", "pup", "terrier", "retriever"],
    "horse": ["horse", "pony", "foal"],
    "sheep": ["sheep", "lamb", "ram"],
    "cow": ["cow", "cattle", "bull", "calf", "ox"],
    "elephant": ["elephant"],
    "bear": ["bear", "teddy"],
    "zebra": ["zebra"],
    "giraffe": ["giraffe"],
    "backpack": ["backpack", "rucksack", "knapsack"],
    "umbrella": ["umbrella", "parasol"],
    "handbag": ["handbag", "purse", "bag", "tote", "shoulder bag", "tote bag", "clutch"],
    "tie": ["tie", "necktie", "bow tie"],
    "suitcase": ["suitcase", "luggage", "briefcase"],
    "frisbee": ["frisbee", "flying disc"],
    "skis": ["ski", "skis"],
    "snowboard": ["snowboard"],
    "sports ball": ["ball", "soccer ball", "football", "baseball", "basketball", "tennis ball", "volleyball", "sports ball"],
    "kite": ["kite"],
    "baseball bat": ["baseball bat", "bat"],
    "baseball glove": ["baseball glove", "glove", "mitt"],
    "skateboard": ["skateboard"],
    "surfboard": ["surfboard", "board"],
    "tennis racket": ["tennis racket", "racket", "racquet"],
    "bottle": ["bottle"],
    "wine glass": ["wine glass"],
    "cup": ["cup", "mug", "coffee cup"],
    "fork": ["fork"],
    "knife": ["knife"],
    "spoon": ["spoon"],
    "bowl": ["bowl"],
    "banana": ["banana"],
    "apple": ["apple"],
    "sandwich": ["sandwich", "burger", "hamburger", "sub"],
    "orange": ["orange"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot"],
    "hot dog": ["hot dog", "hotdog"],
    "pizza": ["pizza"],
    "donut": ["donut", "doughnut"],
    "cake": ["cake", "cupcake"],
    "chair": ["chair", "stool", "seat"],
    "couch": ["couch", "sofa"],
    "potted plant": ["potted plant", "plant", "houseplant", "flower pot", "flowerpot"],
    "bed": ["bed"],
    "dining table": ["table", "dining table", "desk"],
    "toilet": ["toilet"],
    "tv": ["tv", "television", "monitor", "screen"],
    "laptop": ["laptop", "notebook computer", "computer"],
    "mouse": ["mouse"],
    "remote": ["remote", "remote control", "controller"],
    "keyboard": ["keyboard"],
    "cell phone": ["cell phone", "phone", "cellphone", "smartphone", "mobile phone"],
    "microwave": ["microwave"],
    "oven": ["oven", "stove"],
    "toaster": ["toaster"],
    "sink": ["sink"],
    "refrigerator": ["refrigerator", "fridge"],
    "book": ["book", "novel"],
    "clock": ["clock"],
    "vase": ["vase"],
    "scissors": ["scissors"],
    "teddy bear": ["teddy bear", "stuffed animal", "plush"],
    "hair drier": ["hair drier", "hair dryer", "hairdryer"],
    "toothbrush": ["toothbrush"],
}
COCO_CATEGORIES: list[str] = list(COCO_SYNONYMS.keys())

# ---------------------------------------------------------------------------
# Clothing, accessories, carried objects (canonical -> surface forms)
# ---------------------------------------------------------------------------
CLOTHING: dict[str, list[str]] = {
    "shirt": ["shirt", "t-shirt", "tshirt", "tee", "t shirt", "blouse", "polo", "tank top", "jersey"],
    "jacket": ["jacket", "coat", "parka", "blazer", "windbreaker", "raincoat", "overcoat"],
    "sweater": ["sweater", "hoodie", "sweatshirt", "cardigan", "pullover", "jumper"],
    "pants": ["pants", "trousers", "jeans", "slacks", "leggings", "sweatpants"],
    "shorts": ["shorts", "trunks"],
    "dress": ["dress", "gown", "robe"],
    "skirt": ["skirt"],
    "suit": ["suit", "tuxedo"],
    "uniform": ["uniform", "wetsuit", "overalls", "apron"],
    "vest": ["vest"],
    "shoes": ["shoe", "shoes", "sneaker", "sneakers", "boot", "boots", "sandal", "sandals", "heels", "footwear", "cleats"],
    "socks": ["sock", "socks"],
    "hat": ["hat", "cap", "beanie", "baseball cap", "helmet", "hood", "headband", "visor"],
}
ACCESSORIES: dict[str, list[str]] = {
    "glasses": ["glasses", "eyeglasses", "spectacles", "sunglasses", "goggles", "shades"],
    "watch": ["watch", "wristwatch"],
    "jewelry": ["necklace", "bracelet", "earring", "earrings", "jewelry", "jewellery"],
    "tie": ["tie", "necktie", "bow tie"],
    "scarf": ["scarf", "bandana"],
    "gloves": ["glove", "gloves", "mitten", "mittens"],
    "belt": ["belt"],
    "mask": ["mask", "face mask"],
    "badge": ["badge", "lanyard", "name tag"],
}
CARRIED: dict[str, list[str]] = {
    "bag": ["bag", "handbag", "purse", "backpack", "tote", "shoulder bag", "satchel", "rucksack"],
    "suitcase": ["suitcase", "luggage", "briefcase"],
    "umbrella": ["umbrella"],
    "phone": ["phone", "cell phone", "cellphone", "smartphone"],
    "bottle": ["bottle", "water bottle"],
    "cup": ["cup", "mug"],
    "sports equipment": ["racket", "racquet", "bat", "frisbee", "ball", "skateboard", "surfboard", "kite", "ski", "skis", "pole", "snowboard"],
    "food": ["food", "sandwich", "pizza", "donut", "hot dog", "banana", "apple", "cake"],
    "camera": ["camera"],
    "book": ["book", "paper", "papers", "newspaper", "magazine"],
    "leash": ["leash"],
}
SCENE_PLACES: dict[str, list[str]] = {
    "street": ["street", "road", "sidewalk", "intersection", "crosswalk", "avenue", "city", "town", "urban", "downtown", "alley", "highway"],
    "kitchen": ["kitchen"],
    "living room": ["living room", "lounge"],
    "bedroom": ["bedroom"],
    "bathroom": ["bathroom", "restroom", "washroom"],
    "office": ["office", "workplace", "cubicle"],
    "restaurant": ["restaurant", "cafe", "diner", "bar", "cafeteria"],
    "store": ["store", "shop", "market", "supermarket", "mall"],
    "beach": ["beach", "shore", "seaside", "coast", "ocean", "sea"],
    "park": ["park", "garden", "playground", "lawn", "grass", "field", "meadow"],
    "sports field": ["stadium", "court", "baseball field", "tennis court", "soccer field", "pitch", "arena", "diamond"],
    "snow/mountain": ["snow", "slope", "mountain", "ski slope", "hill"],
    "water": ["lake", "river", "water", "harbor", "pool"],
    "station": ["station", "airport", "platform", "terminal", "runway", "tracks", "railway"],
    "farm/rural": ["farm", "pasture", "barn", "countryside", "rural", "zoo", "enclosure"],
}
COLORS: list[str] = ["black", "white", "red", "blue", "green", "yellow", "orange", "purple", "pink",
                     "brown", "gray", "grey", "beige", "tan", "navy", "gold", "silver", "maroon", "khaki", "dark", "light"]
COLOR_CANON = {"grey": "gray", "navy": "blue", "maroon": "red", "khaki": "beige", "tan": "beige"}

# Linguistic hedging / uncertainty cues (used by the narrative module).
HEDGE_CUES: list[str] = [
    "maybe", "perhaps", "possibly", "probably", "might", "may have", "could have", "could be",
    "i think", "i believe", "i guess", "not sure", "unsure", "i'm not sure", "looked like",
    "seemed", "seems", "appeared to", "appears to", "kind of", "sort of", "something like",
    "or something", "i can't remember", "i don't remember", "hard to tell", "couldn't see",
    "could not see", "unclear", "roughly", "or maybe", "likely",
    "dark-ish", "-ish", "i assume",
]
NEGATION_CUES: list[str] = ["no ", "not ", "without", "never", "none", "didn't", "did not", "wasn't", "was not"]
MISSING_INFO_SLOTS: list[str] = ["clothing", "accessories", "carried_objects", "scene_place", "lighting", "time_of_day"]

_TOKEN_RE = re.compile(r"[a-z][a-z\-']*")


def _singular(word: str) -> str:
    w = word.lower().strip()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("es") and w[-3] in "sxz":
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss") and w not in {"glasses", "pants", "shorts", "jeans", "trousers", "skis", "sunglasses", "goggles", "scissors", "leggings", "overalls", "trunks", "earrings", "gloves", "socks", "shoes", "boots", "sneakers", "sandals", "heels", "cleats", "mittens", "slacks", "sweatpants", "papers"}:
        return w[:-1]
    return w


def normalize_phrase(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9\-\s']", " ", text)
    toks = [_singular(t) for t in text.split()]
    return " ".join(toks)


@lru_cache(maxsize=8)
def _reverse(table_name: str) -> dict[str, str]:
    table = {"coco": COCO_SYNONYMS, "clothing": CLOTHING, "accessories": ACCESSORIES,
             "carried": CARRIED, "scene": SCENE_PLACES}[table_name]
    rev: dict[str, str] = {}
    for canon, forms in table.items():
        for f in forms + [canon]:
            rev[normalize_phrase(f)] = canon
    return rev


def map_term(text: str, table: str) -> str | None:
    """Map a free-text phrase to a canonical label of ``table`` (longest match wins)."""
    norm = normalize_phrase(text)
    if not norm:
        return None
    rev = _reverse(table)
    if norm in rev:
        return rev[norm]
    toks = norm.split()
    # longest n-gram match inside the phrase; prefer head noun (rightmost)
    for n in range(min(3, len(toks)), 0, -1):
        for i in range(len(toks) - n, -1, -1):
            gram = " ".join(toks[i:i + n])
            if gram in rev:
                return rev[gram]
    return None


def find_terms(text: str, table: str) -> set[str]:
    """Return all canonical labels from ``table`` mentioned anywhere in ``text``."""
    norm = " " + normalize_phrase(text) + " "
    rev = _reverse(table)
    found: set[str] = set()
    for form, canon in rev.items():
        if f" {form} " in norm:
            found.add(canon)
    return found


def canon_color(text: str | None) -> str | None:
    if not text:
        return None
    for tok in normalize_phrase(text).split():
        if tok in COLORS and tok not in {"dark", "light"}:
            return COLOR_CANON.get(tok, tok)
    return None
