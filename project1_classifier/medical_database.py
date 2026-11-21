"""
Medicinal botanical knowledge base for Project 1:
Traditional Medicinal Flower Identifier.

Maps flower class labels to botanical names, active chemical compounds,
traditional medicinal uses, and health warnings/precautions.
"""

from typing import Any


MEDICAL_DATABASE: dict[str, dict[str, Any]] = {
    "hibiscus": {
        "common_name": "Hibiscus",
        "botanical_name": "Hibiscus rosa-sinensis / Hibiscus sabdariffa",
        "active_compounds": [
            "Anthocyanins",
            "Flavonoids (quercetin, gossypetin)",
            "Organic acids (hibiscus acid, citric acid)",
            "Polyphenols",
        ],
        "uses": (
            "Rich in antioxidants. Used traditionally to lower blood pressure, "
            "support cardiovascular health, and manage cholesterol."
        ),
        "warning": (
            "Avoid excessive consumption during pregnancy. Consult a healthcare "
            "professional if taking antihypertensive or blood-sugar medications."
        ),
    },
    "marigold": {
        "common_name": "Marigold",
        "botanical_name": "Tagetes erecta / Calendula officinalis",
        "active_compounds": [
            "Lutein and zeaxanthin (carotenoids)",
            "Flavonoids",
            "Triterpenoid saponins",
            "Essential oils",
        ],
        "uses": (
            "Contains strong anti-inflammatory and antimicrobial properties. "
            "Used traditionally in topical preparations for wound healing, soothing "
            "skin irritation, and minor rashes."
        ),
        "warning": (
            "May cause allergic reactions in individuals sensitive to plants in the "
            "Asteraceae/Compositae family. Avoid during pregnancy unless advised "
            "by a qualified practitioner."
        ),
    },
    "neem": {
        "common_name": "Neem",
        "botanical_name": "Azadirachta indica",
        "active_compounds": [
            "Azadirachtin",
            "Nimbin and nimbidin",
            "Quercetin",
            "Limonoids",
        ],
        "uses": (
            "Traditionally valued for antibacterial, antifungal, and "
            "anti-inflammatory properties in skin care, oral hygiene, and "
            "general wellness support."
        ),
        "warning": (
            "Internal use of neem extracts or oils can be toxic—especially in "
            "children, pregnant or breastfeeding women, and individuals with "
            "liver or kidney conditions."
        ),
    },
    "shankhpushpa": {
        "common_name": "Shankhpushpa",
        "botanical_name": "Convolvulus pluricaulis / Clitoria ternatea",
        "active_compounds": [
            "Alkaloids (shankhpushpine)",
            "Flavonoids and flavonol glycosides",
            "Coumarins",
            "Phenolic acids",
        ],
        "uses": (
            "Traditionally used in Ayurvedic practice as a brain tonic to support "
            "memory, cognitive function, calmness, and stress relief."
        ),
        "warning": (
            "May interact with sedatives, anticonvulsants, or thyroid/blood-pressure "
            "medications. Use under professional guidance."
        ),
    },
    "lotus": {
        "common_name": "Lotus",
        "botanical_name": "Nelumbo nucifera",
        "active_compounds": [
            "Nuciferine and liensinine (alkaloids)",
            "Kaempferol and quercetin (flavonoids)",
            "Polyphenols",
        ],
        "uses": (
            "Used traditionally for cooling/soothing effects, antioxidant support, "
            "digestion, and relaxation."
        ),
        "warning": (
            "Generally well tolerated in culinary/tea amounts, though concentrated "
            "alkaloid extracts should be used cautiously during pregnancy or "
            "with sedative medications."
        ),
    },
    "jasmine": {
        "common_name": "Jasmine",
        "botanical_name": "Jasminum grandiflorum / Jasminum sambac",
        "active_compounds": [
            "Benzyl acetate",
            "Linalool",
            "Indole",
            "Jasmone and flavonoids",
        ],
        "uses": (
            "Used traditionally in teas and aromatherapy to promote relaxation, "
            "ease mild stress, and provide antioxidant support."
        ),
        "warning": (
            "Concentrated essential oils can irritate sensitive skin; avoid "
            "excessive intake during pregnancy or breastfeeding."
        ),
    },
}


def normalize_flower_key(flower_name: str) -> str:
    """Normalize a predicted class name (e.g., 'Hibiscus Flower' -> 'hibiscus')."""
    cleaned = flower_name.strip().lower().replace("-", "_").replace(" ", "_")
    if cleaned.endswith("_flower"):
        cleaned = cleaned[: -len("_flower")]
    return cleaned


def get_flower_info(flower_name: str) -> dict[str, Any]:
    """
    Retrieve medicinal metadata for `flower_name`.

    Falls back to a safe default entry if the class label is not yet in
    `MEDICAL_DATABASE`.
    """
    key = normalize_flower_key(flower_name)
    if key in MEDICAL_DATABASE:
        entry = dict(MEDICAL_DATABASE[key])
        entry["class_key"] = key
        entry["found"] = True
        return entry

    display_name = flower_name.strip().replace("_", " ").title() or "Unknown Plant"
    return {
        "class_key": key,
        "common_name": display_name,
        "botanical_name": "Species details not yet cataloged",
        "active_compounds": ["Pending botanical/phytochemical entry"],
        "uses": (
            f"No traditional medicinal entry found yet for '{display_name}'. "
            "Add this species to `MEDICAL_DATABASE` in `project1_classifier/medical_database.py`."
        ),
        "warning": (
            "Do not consume or apply unidentified plant material medicinally. "
            "Consult a qualified botanist or healthcare professional."
        ),
        "found": False,
    }


def list_supported_flowers() -> list[str]:
    """Return sorted list of flower keys currently in `MEDICAL_DATABASE`."""
    return sorted(MEDICAL_DATABASE.keys())