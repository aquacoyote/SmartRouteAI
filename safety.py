
import re

# Words that can reverse or materially change meaning.
NEGATION_WORDS = {
    "not", "never", "no", "neither", "nor", "n't"
}

# Terms that indicate the type of information being requested.
CONCEPT_GROUPS = {
    "time_complexity": {
        "time complexity", "time cost", "running time", "execution time"
    },
    "space_complexity": {
        "space complexity", "memory complexity", "memory usage"
    },
    "definition": {"what is", "define", "meaning of"},
    "use_case": {"used for", "use cases", "applications", "purpose"},
}


def normalize(text):
    """Lowercase text and standardize whitespace."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def has_negation(text):
    """Detect simple negation markers."""
    words = set(re.findall(r"\b\w+(?:'\w+)?\b", normalize(text)))
    return any(word in NEGATION_WORDS or word.endswith("n't") for word in words)


def concept_tags(text):
    """Identify explicit information types mentioned in a question."""
    text = normalize(text)
    tags = set()

    for tag, phrases in CONCEPT_GROUPS.items():
        if any(phrase in text for phrase in phrases):
            tags.add(tag)

    return tags


def question_intent(text):
    """Identify the main intent of a question."""
    text = normalize(text)

    use_case_phrases = (
        "used for",
        "applications of",
        "application of",
        "use cases",
        "purpose of",
    )

    if any(phrase in text for phrase in use_case_phrases):
        return "use_case"

    definition_phrases = (
        "what is",
        "what are",
        "define",
        "explain",
        "meaning of",
    )

    if any(phrase in text for phrase in definition_phrases):
        return "definition"

    return "other"


def comparison_direction_mismatch(question_a, question_b):
    """Detect opposite comparison words or reversed entity order."""
    pattern = re.compile(
        r"\bis\s+(.+?)\s+"
        r"(faster|slower|better|worse|larger|smaller)"
        r"\s+than\s+(.+?)[?.!]*$",
        re.IGNORECASE,
    )

    match_a = pattern.search(normalize(question_a))
    match_b = pattern.search(normalize(question_b))

    if not match_a or not match_b:
        return False

    left_a, relation_a, right_a = match_a.groups()
    left_b, relation_b, right_b = match_b.groups()

    left_a, right_a = left_a.strip(), right_a.strip()
    left_b, right_b = left_b.strip(), right_b.strip()

    opposite_relations = {
        ("faster", "slower"),
        ("slower", "faster"),
        ("better", "worse"),
        ("worse", "better"),
        ("larger", "smaller"),
        ("smaller", "larger"),
    }

    same_entities = left_a == left_b and right_a == right_b

    reversed_entities = (
        left_a == right_b and right_a == left_b
    )

    opposite_meaning = (relation_a, relation_b) in opposite_relations

    if same_entities and opposite_meaning:
        return True

    if reversed_entities and relation_a == relation_b:
        return True

    return False


ALGORITHMS = {
    "binary search",
    "linear search",
    "merge sort",
    "quick sort",
    "insertion sort",
    "selection sort",
    "bubble sort",
}


def different_named_algorithm(question_a, question_b):
    """Detect when questions explicitly name different algorithms."""
    text_a = normalize(question_a)
    text_b = normalize(question_b)

    algorithms_a = {
        algorithm for algorithm in ALGORITHMS if algorithm in text_a
    }
    algorithms_b = {
        algorithm for algorithm in ALGORITHMS if algorithm in text_b
    }

    return bool(algorithms_a and algorithms_b and algorithms_a != algorithms_b)


def safety_check(question_a, question_b):
    """Return (safe, reason) for a potential cache match."""

    if has_negation(question_a) != has_negation(question_b):
        return False, "Negation differs"

    if comparison_direction_mismatch(question_a, question_b):
        return False, "Comparison direction differs"

    if different_named_algorithm(question_a, question_b):
        return False, "Different named algorithms"

    tags_a = concept_tags(question_a)
    tags_b = concept_tags(question_b)

    if (
        "time_complexity" in tags_a
        and "space_complexity" in tags_b
    ) or (
        "space_complexity" in tags_a
        and "time_complexity" in tags_b
    ):
        return False, "Time complexity vs space complexity"

    intent_a = question_intent(question_a)
    intent_b = question_intent(question_b)

    if {intent_a, intent_b} == {"definition", "use_case"}:
        return False, "Definition vs use case"

    return True, "No safety rule triggered"
