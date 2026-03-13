# =====================================================
# INTENT DETECTION MODULE
# PURPOSE:
# This module identifies the category (intent) of a user query
# using simple keyword matching. The detected intent helps
# route the query to the correct document collection.
# =====================================================


# =====================================================
# INTENT KEYWORDS CONFIGURATION
# Each intent contains keywords related to that category.
# =====================================================

INTENT_KEYWORDS = {

    "admission": [
        "admission", "admit", "apply", "application",
        "eligibility", "entrance", "enroll"
    ],

    "fees": [
        "fee", "fees", "payment", "scholarship",
        "cost", "tuition", "refund"
    ],

    "courses": [
        "course", "program", "branch",
        "department", "subject", "syllabus", "curriculum"
    ],

    "placement": [
        "placement", "job", "recruit",
        "company", "package", "salary", "intern"
    ],

    "hostel": [
        "hostel", "room", "mess",
        "accommodation", "boarding", "lodge"
    ],

    "exam": [
        "exam", "result", "marks",
        "grade", "gpa", "cgpa", "semester"
    ],

    "emergency": [
        "emergency", "help", "medical",
        "fire", "police", "ambulance", "hospital"
    ],

    "general": []
}


# =====================================================
# FUNCTION: detect_intent
# PURPOSE:
# Identifies the user's intent by checking keywords
# in the query. Returns matching intent or "general".
# =====================================================

def detect_intent(query):

    query = query.lower()

    for intent in INTENT_KEYWORDS:

        keywords = INTENT_KEYWORDS[intent]

        for keyword in keywords:

            if keyword in query:
                return intent

    return "general"