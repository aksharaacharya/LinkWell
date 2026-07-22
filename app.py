import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pgeocode
import os
import json
import requests
import math
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ---------------------------------------------------------
# Load model artifacts (from Day 1)
# ---------------------------------------------------------
model = joblib.load("pirs_ridge_model.pkl")
feature_columns = joblib.load("pirs_feature_columns.pkl")

# ---------------------------------------------------------
# Page routing via session_state
# ---------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "intro"

if "respondent_type" not in st.session_state:
    st.session_state.respondent_type = "Self"

# ---------------------------------------------------------
# PAGE: Intro / Disclaimer
# ---------------------------------------------------------
def show_intro_page():
    st.title("Social Connection Check-In")

    st.warning(
        "**Important disclaimer:** This tool uses a machine learning model "
        "trained on synthetic (artificially generated) data, not real medical "
        "records. The score it produces is an *approximation* meant to spark "
        "reflection and point toward helpful resources — it is **not** a "
        "medical diagnosis or clinical assessment. Please consult a healthcare "
        "professional for any medical concerns."
    )

    st.subheader("Who is filling this out today?")
    respondent_choice = st.radio(
        "Select one:",
        options=["I am filling this out for myself", "I am a caregiver filling this out for someone else"],
        index=0 if st.session_state.respondent_type == "Self" else 1,
    )

    if st.button("Continue"):
        st.session_state.respondent_type = "Self" if "myself" in respondent_choice else "Caregiver"
        st.session_state.page = "questionnaire1"
        st.rerun()

# ---------------------------------------------------------
# PAGE: Questionnaire 1 (lifestyle inputs)
# ---------------------------------------------------------
def show_questionnaire1_page():
    is_self = st.session_state.respondent_type == "Self"

    if st.button("⬅ Back"):
        st.session_state.page = "intro"
        st.rerun()

    st.title("Lifestyle Questionnaire")
    if is_self:
        st.write("Answer these questions about how you currently spend time.")
    else:
        st.write("Answer these questions about how they currently spend time.")

    prev = st.session_state.get("raw_answers", {})

    def sel_index(options, key, fallback=0):
        if key in prev and prev[key] in options:
            return options.index(prev[key])
        return fallback

    with st.form("questionnaire1_form"):
        age = st.slider("Age", 60, 95, prev.get("age", 75))

        lives_alone_question = "Do you live alone?" if is_self else "Does the person you're caring for live alone?"
        lives_alone_options = ["Yes", "No"]
        lives_alone = st.radio(
            lives_alone_question, lives_alone_options,
            index=sel_index(lives_alone_options, "lives_alone")
        )

        marital_options = ["Married", "Widowed", "Divorced", "Single"]
        marital_status = st.selectbox("Marital status", marital_options, index=sel_index(marital_options, "marital_status"))

        weekly_family_visits = st.number_input(
            "Weekly family visits (number of visits per week)", min_value=0, max_value=20,
            value=prev.get("weekly_family_visits", 2)
        )
        weekly_friend_interactions = st.number_input(
            "Weekly friend interactions (number per week)", min_value=0, max_value=20,
            value=prev.get("weekly_friend_interactions", 2)
        )
        community_events_attended = st.number_input(
            "Community events attended per week", min_value=0, max_value=10,
            value=prev.get("community_events_attended", 1)
        )
        volunteer_hours = st.number_input(
            "Volunteer hours per week", min_value=0.0, max_value=20.0,
            value=float(prev.get("volunteer_hours", 0.0)), step=0.5
        )
        religious_participation_hours = st.number_input(
            "Religious participation hours per week", min_value=0.0, max_value=20.0,
            value=float(prev.get("religious_participation_hours", 0.0)), step=0.5
        )
        exercise_days = st.slider("Exercise days per week", 0, 7, prev.get("exercise_days", 2))

        video_options = ["Never", "Monthly", "Weekly", "Daily"]
        uses_video_calls = st.selectbox(
            "How often does video calling happen?", video_options,
            index=sel_index(video_options, "uses_video_calls")
        )

        transport_options = ["None", "Family", "Public transit", "Can drive"]
        transportation_access = st.selectbox(
            "Transportation access", transport_options,
            index=sel_index(transport_options, "transportation_access")
        )

        mobility_options = ["None", "Mild", "Moderate", "Severe"]
        mobility_limitations = st.selectbox(
            "Mobility limitations", mobility_options,
            index=sel_index(mobility_options, "mobility_limitations")
        )

        depression_options = ["Low", "Moderate", "High"]
        depression_symptoms = st.selectbox(
            "Depression symptoms", depression_options,
            index=sel_index(depression_options, "depression_symptoms")
        )

        button_label = "See My Score" if is_self else "See Their Score"
        submitted = st.form_submit_button(button_label)

        if submitted:
            st.session_state.raw_answers = {
                "age": age,
                "lives_alone": lives_alone,
                "marital_status": marital_status,
                "weekly_family_visits": weekly_family_visits,
                "weekly_friend_interactions": weekly_friend_interactions,
                "community_events_attended": community_events_attended,
                "volunteer_hours": volunteer_hours,
                "religious_participation_hours": religious_participation_hours,
                "exercise_days": exercise_days,
                "uses_video_calls": uses_video_calls,
                "transportation_access": transportation_access,
                "mobility_limitations": mobility_limitations,
                "depression_symptoms": depression_symptoms,
            }
            st.session_state.page = "results"
            st.rerun()

# ---------------------------------------------------------
# PAGE: Results (PIRS score + explanation)
# ---------------------------------------------------------
def show_results_page():
    answers = st.session_state.raw_answers
    is_self = st.session_state.respondent_type == "Self"
    subject = "You" if is_self else "The person you're caring for"
    subject_possessive = "your" if is_self else "their"

    # Build a one-row DataFrame from the stored answers and run the model
    raw_df = pd.DataFrame([answers])
    X_live = extract_features(raw_df)
    pirs_score = model.predict(X_live)[0]
    pirs_score = float(np.clip(pirs_score, 1, 100))
    st.session_state.pirs_score = pirs_score

    # ---------------------------------------------------
    # Recompute domain scores directly, using the ORIGINAL
    # formula weights (not the model's learned coefficients)
    # ---------------------------------------------------
    family_visits_risk = np.clip(1 - answers["weekly_family_visits"] / 7, 0, 1)
    friend_interactions_risk = np.clip(1 - answers["weekly_friend_interactions"] / 7, 0, 1)
    video_calls_risk = {"Daily": 0.0, "Weekly": 0.25, "Monthly": 0.60, "Never": 1.0}[answers["uses_video_calls"]]
    SC = (family_visits_risk + friend_interactions_risk + video_calls_risk) / 3

    community_events_risk = np.clip(1 - answers["community_events_attended"] / 4, 0, 1)
    volunteer_hours_risk = np.clip(1 - answers["volunteer_hours"] / 10, 0, 1)
    religious_hours_risk = np.clip(1 - answers["religious_participation_hours"] / 5, 0, 1)
    C = (community_events_risk + volunteer_hours_risk + religious_hours_risk) / 3

    transportation_risk = {"Can drive": 0.0, "Family": 0.3, "Public transit": 0.5, "None": 1.0}[answers["transportation_access"]]
    mobility_risk = {"None": 0.0, "Mild": 0.33, "Moderate": 0.67, "Severe": 1.0}[answers["mobility_limitations"]]
    A = (transportation_risk + mobility_risk) / 2

    exercise_days_risk = np.clip(1 - answers["exercise_days"] / 7, 0, 1)
    depression_risk = {"Low": 0.0, "Moderate": 0.5, "High": 1.0}[answers["depression_symptoms"]]
    HL = (exercise_days_risk + depression_risk) / 2

    lives_alone_risk = {"No": 0.0, "Yes": 1.0}[answers["lives_alone"]]
    marital_status_risk = {"Married": 0.0, "Single": 0.5, "Divorced": 0.7, "Widowed": 1.0}[answers["marital_status"]]
    LS = (lives_alone_risk + marital_status_risk) / 2

    # Weighted contribution of each domain to the FINAL score (0-100 scale)
    domain_contributions = {
        "Social Connections": 0.35 * SC * 100,
        "Community": 0.25 * C * 100,
        "Access": 0.15 * A * 100,
        "Health/Lifestyle": 0.15 * HL * 100,
        "Living Status": 0.10 * LS * 100,
    }
    top_domain = max(domain_contributions, key=domain_contributions.get)

    # Feature-level detail within the top domain, for paragraph 2
    domain_features = {
        "Social Connections": [
            ("weekly family visits", family_visits_risk),
            ("weekly friend interactions", friend_interactions_risk),
            ("video call frequency", video_calls_risk),
        ],
        "Community": [
            ("community events attended", community_events_risk),
            ("volunteer hours", volunteer_hours_risk),
            ("religious participation", religious_hours_risk),
        ],
        "Access": [
            ("transportation access", transportation_risk),
            ("mobility limitations", mobility_risk),
        ],
        "Health/Lifestyle": [
            ("exercise days", exercise_days_risk),
            ("depression symptoms", depression_risk),
        ],
        "Living Status": [
            ("living alone", lives_alone_risk),
            ("marital status", marital_status_risk),
        ],
    }
    top_features = sorted(domain_features[top_domain], key=lambda x: x[1], reverse=True)[:3]
    top_feature_names = ", ".join(f[0] for f in top_features)

    # ---------------------------------------------------
    # Display
    # ---------------------------------------------------
    if st.button("⬅ Back"):
            st.session_state.page = "questionnaire1"
            st.rerun()
    st.title("Results")
    st.metric("Predicted Isolation Risk Score (PIRS)", f"{pirs_score:.0f} / 100")

    if pirs_score < 34:
        risk_level = "relatively low"
    elif pirs_score < 67:
        risk_level = "moderate"
    else:
        risk_level = "elevated"

    st.write(
        f"{subject} scored **{pirs_score:.0f} out of 100** on this tool, which is considered a "
        f"**{risk_level}** level of predicted social isolation risk. This score reflects patterns across "
        f"{subject_possessive} social connections, community involvement, access to transportation and mobility, "
        f"health and lifestyle habits, and living situation. It is not a diagnosis, but a starting point for "
        f"noticing where {subject_possessive} daily routine may be helping or hurting social connectedness. "
        f"A higher score suggests more risk factors are present, while a lower score suggests {subject_possessive} "
        f"routine currently includes more protective habits."
    )

    subject_pronoun = "you" if is_self else "they"

    domain_advice = {
        "Social Connections": (
            f"increasing how often {subject_pronoun} connect with family or friends, even in small ways like a "
            f"weekly phone call, a video chat, or a scheduled regular visit"
        ),
        "Community": (
            f"finding a local community event, volunteer opportunity, or group activity {subject_pronoun} could "
            f"attend on a regular basis, even once every week or two"
        ),
        "Access": (
            f"looking into transportation options, such as senior ride services or community shuttles, and "
            f"discussing mobility support that could make it easier for {subject_pronoun} to get out and about"
        ),
        "Health/Lifestyle": (
            f"building small amounts of regular movement into {subject_possessive} routine and checking in on "
            f"{subject_possessive} emotional wellbeing, since both closely relate to staying socially engaged"
        ),
        "Living Status": (
            f"finding ways to build more regular company or check-ins into {subject_possessive} daily routine, "
            f"especially if {subject_pronoun} live alone"
        ),
    }

    st.write(
        f"The area contributing most to this score is **{top_domain}**, particularly {subject_possessive} "
        f"{top_feature_names}. This is the domain where {subject_possessive} current habits show the most room for "
        f"improvement, so small changes here tend to have the biggest impact on lowering the overall score. For example, "
        f"{domain_advice[top_domain]} can meaningfully shift this score over time. The next step below can help find "
        f"specific local options tailored to what {subject_pronoun} might actually enjoy."
    )

    if st.button("GET PERSONALIZED RECOMMENDATIONS"):
        st.session_state.page = "questionnaire2"
        st.rerun()

# ---------------------------------------------------------
# PAGE: Questionnaire 2 (recommendation personalization)
# ---------------------------------------------------------
US_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY"
]

def show_questionnaire2_page():
    is_self = st.session_state.respondent_type == "Self"
    subject_possessive = "your" if is_self else "their"

    if st.button("⬅ Back"):
        st.session_state.page = "results"
        st.rerun()

    st.title("Personalize Your Recommendations")
    st.write(f"Tell us more about {subject_possessive} interests so we can find good local matches.")

    prev = st.session_state.get("preferences", {})

    def sel_index(options, key, fallback=0):
        if key in prev and prev[key] in options:
            return options.index(prev[key])
        return fallback

    with st.form("questionnaire2_form"):
        interested_activities = st.multiselect(
            "Interested activities (select all that apply)",
            [
                "Arts & crafts", "Music", "Reading", "Gardening", "Walking", "Hiking",
                "Fitness classes", "Cooking", "Technology", "Board games", "Card games",
                "Animals", "Volunteering", "Teaching or mentoring", "Faith-based activities",
                "Cultural events", "Movies", "Local history", "Travel groups",
            ],
            default=prev.get("interested_activities", []),
        )

        group_options = ["One-on-one", "Small groups of 3-10", "Large groups", "No preference"]
        group_preference = st.selectbox("Socialization group preference", group_options, index=sel_index(group_options, "group_preference"))

        age_group_options = ["Older adults", "Adults of all ages", "Teenagers", "Children", "No preference"]
        age_group_preference = st.selectbox("Socialization age group preference", age_group_options, index=sel_index(age_group_options, "age_group_preference"))

        travel_options = ["Drive myself", "Family or friends", "Public transportation", "Ride-share", "Walking", "At Home Activities Only"]
        travel_style = st.selectbox("Usual travel style", travel_options, index=sel_index(travel_options, "travel_style"))

        accommodation_options = ["Wheelchair accessible", "Limited walking", "Hearing assistance", "Vision assistance", "None"]
        accommodations = st.selectbox("Accommodations needed", accommodation_options, index=sel_index(accommodation_options, "accommodations"))

        first_event_options = ["Alone", "With a volunteer companion", "With a friend or family member", "No preference"]
        first_event_preference = st.selectbox("First event attendance preference", first_event_options, index=sel_index(first_event_options, "first_event_preference"))

        volunteering_interests = st.multiselect(
            "Volunteering interests (select all that apply)",
            [
                "Tutoring students", "Reading to children", "Helping at libraries",
                "Mentoring young professionals", "Community gardening", "Food banks",
                "Animal shelters",
            ],
            default=prev.get("volunteering_interests", []),
        )

        event_goal_options = [
            "Make new friends", "Stay physically active", "Learn a new skill",
            "Help my community", "Spend time with younger generations",
            "Find people with similar hobbies", "Just get out of the house",
        ]
        event_goal = st.selectbox("Goal of attending events", event_goal_options, index=sel_index(event_goal_options, "event_goal"))

        event_type_options = ["Reconnect with existing friends & family", "Make new friends", "Both"]
        event_type = st.selectbox("Type of event preferred", event_type_options, index=sel_index(event_type_options, "event_type"))

        state = st.selectbox("State", US_STATES, index=sel_index(US_STATES, "state"))
        zip_code = st.text_input("ZIP code", max_chars=5, value=prev.get("zip_code", ""))

        submitted = st.form_submit_button("SUBMIT")

        if submitted:
            if not (zip_code.isdigit() and len(zip_code) == 5):
                st.error("Please enter a valid 5-digit ZIP code.")
                return

            nomi = pgeocode.Nominatim("us")
            zip_info = nomi.query_postal_code(zip_code)

            if pd.isna(zip_info.state_code):
                st.error("That ZIP code wasn't recognized. Please double-check it.")
                return

            if zip_info.state_code != state:
                st.error(f"That ZIP code appears to be in {zip_info.state_code}, not {state}. Please check the ZIP code and state match.")
                return

            st.session_state.preferences = {
                "interested_activities": interested_activities,
                "group_preference": group_preference,
                "age_group_preference": age_group_preference,
                "travel_style": travel_style,
                "accommodations": accommodations,
                "first_event_preference": first_event_preference,
                "volunteering_interests": volunteering_interests,
                "event_goal": event_goal,
                "event_type": event_type,
                "state": state,
                "zip_code": zip_code,
            }
            if "recommendations_data" in st.session_state:
                del st.session_state["recommendations_data"]
            st.session_state.page = "recommendations"
            st.rerun()

# ---------------------------------------------------------
# Feature increment assumptions: if someone regularly attends
# a recommended activity, roughly how much would the underlying
# raw answer increase? (rough, illustrative estimates for the demo)
# ---------------------------------------------------------
IMPACT_INCREMENTS = {
    "weekly_family_visits": 1,
    "weekly_friend_interactions": 2,
    "community_events_attended": 1,
    "volunteer_hours": 2,
    "religious_participation_hours": 1,
    "exercise_days": 1,
}
IMPACT_CAPS = {
    "weekly_family_visits": 10,
    "weekly_friend_interactions": 10,
    "community_events_attended": 6,
    "volunteer_hours": 12,
    "religious_participation_hours": 6,
    "exercise_days": 7,
}

def estimate_pirs_with_recommendation(base_answers, impact_feature):
    """Return the projected PIRS score if the user regularly did this activity."""
    if impact_feature not in IMPACT_INCREMENTS:
        return None  # unrecognized feature name from the LLM — skip projection

    modified = dict(base_answers)
    increment = IMPACT_INCREMENTS[impact_feature]
    cap = IMPACT_CAPS[impact_feature]
    modified[impact_feature] = min(modified[impact_feature] + increment, cap)

    modified_df = pd.DataFrame([modified])
    X_modified = extract_features(modified_df)
    new_score = float(np.clip(model.predict(X_modified)[0], 1, 100))
    return new_score

# ---------------------------------------------------------
# Real venue lookup via OpenStreetMap's Overpass API (free, no key)
# ---------------------------------------------------------
CATEGORY_INFO = {
    "community_centre": {
        "query": '["amenity"="community_centre"]',
        "keywords": ["board games", "card games", "technology", "local history", "cultural events", "teaching or mentoring", "arts & crafts"],
        "impact_feature": "community_events_attended",
        "label": "Community Center",
    },
    "library": {
        "query": '["amenity"="library"]',
        "keywords": ["reading", "technology", "teaching or mentoring", "local history", "helping at libraries"],
        "impact_feature": "community_events_attended",
        "label": "Library",
    },
    "place_of_worship": {
        "query": '["amenity"="place_of_worship"]',
        "keywords": ["faith-based activities"],
        "impact_feature": "religious_participation_hours",
        "label": "Place of Worship",
    },
    "park": {
        "query": '["leisure"="park"]',
        "keywords": ["walking", "hiking", "gardening", "animals"],
        "impact_feature": "exercise_days",
        "label": "Park",
    },
    "sports_centre": {
        "query": '["leisure"="sports_centre"]',
        "keywords": ["fitness classes", "walking", "hiking"],
        "impact_feature": "exercise_days",
        "label": "Fitness/Sports Center",
    },
    "arts_centre": {
        "query": '["amenity"="arts_centre"]',
        "keywords": ["arts & crafts", "music", "cultural events", "movies"],
        "impact_feature": "community_events_attended",
        "label": "Arts Center",
    },
    "social_facility": {
        "query": '["amenity"="social_facility"]',
        "keywords": ["volunteering", "animals", "food banks", "community gardening"],
        "impact_feature": "volunteer_hours",
        "label": "Social/Community Services",
    },
}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def get_lat_lon_from_zip(zip_code):
    nomi = pgeocode.Nominatim("us")
    info = nomi.query_postal_code(zip_code)
    if pd.isna(info.latitude) or pd.isna(info.longitude):
        return None, None
    return info.latitude, info.longitude


def query_nearby_venues(lat, lon, radius_m=12000):
    """Query Overpass API for real, named venues near a lat/lon."""
    clauses = "".join(
        f'node{info["query"]}(around:{radius_m},{lat},{lon});'
        f'way{info["query"]}(around:{radius_m},{lat},{lon});'
        f'relation{info["query"]}(around:{radius_m},{lat},{lon});'
        for info in CATEGORY_INFO.values()
    )
    overpass_query = f"[out:json][timeout:25];({clauses});out center;"

    try:
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": overpass_query},
            headers={"User-Agent": "LinkWell-Isolation-Risk-App/1.0 (student hackathon project)"},
            timeout=25,
        )
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
    except Exception:
        return []

    venues = []
    for el in elements:
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name:
            continue

        category = None
        for cat_key, info in CATEGORY_INFO.items():
            tag_key, tag_val = info["query"].strip("[]").replace('"', "").split("=")
            if tags.get(tag_key) == tag_val:
                category = cat_key
                break
        if category is None:
            continue

        # Nodes have lat/lon directly; ways/relations use a "center" object with "out center"
        venue_lat = el.get("lat") or el.get("center", {}).get("lat")
        venue_lon = el.get("lon") or el.get("center", {}).get("lon")
        if venue_lat is None or venue_lon is None:
            continue

        venues.append({
            "name": name,
            "category": category,
            "label": CATEGORY_INFO[category]["label"],
            "impact_feature": CATEGORY_INFO[category]["impact_feature"],
            "address": tags.get("addr:full") or ", ".join(
                filter(None, [tags.get("addr:housenumber", "") + " " + tags.get("addr:street", ""), tags.get("addr:city", "")])
            ).strip(", ") or "Address not listed",
            "website": tags.get("website") or tags.get("contact:website"),
            "phone": tags.get("phone") or tags.get("contact:phone"),
            "distance_km": haversine_km(lat, lon, venue_lat, venue_lon),
        })

    return venues


def rank_venues_by_preferences(venues, prefs):
    """Score venues by how well their category matches stated preferences."""
    user_interests = set(prefs.get("interested_activities", [])) | set(prefs.get("volunteering_interests", []))
    user_interests_lower = {x.lower() for x in user_interests}

    scored = []
    for v in venues:
        keywords = CATEGORY_INFO[v["category"]]["keywords"]
        match_count = sum(1 for kw in keywords if kw.lower() in user_interests_lower)
        scored.append((match_count, v["distance_km"], v))

    # sort by best keyword match first, then closest distance
    scored.sort(key=lambda x: (-x[0], x[1]))

    # dedupe by name, keep top 5
    seen_names = set()
    top_venues = []
    for _, _, v in scored:
        if v["name"] not in seen_names:
            top_venues.append(v)
            seen_names.add(v["name"])
        if len(top_venues) == 5:
            break
    return top_venues

def show_recommendations_page():
    answers = st.session_state.raw_answers
    prefs = st.session_state.preferences
    is_self = st.session_state.respondent_type == "Self"
    subject = "you" if is_self else "the person being cared for"

    st.title("Local Recommendations")

    if st.button("⬅ Back"):
        st.session_state.page = "questionnaire2"
        st.rerun()

    st.success(
        "✅ These are real, verified locations from OpenStreetMap near your ZIP code. "
        "Descriptions are AI-written to explain the match — always call ahead or check their "
        "website for current hours and activities."
    )

    def fetch_recommendations():
        lat, lon = get_lat_lon_from_zip(prefs["zip_code"])
        if lat is None:
            return []

        venues = query_nearby_venues(lat, lon)
        if not venues:
            return []

        top_venues = rank_venues_by_preferences(venues, prefs)
        if not top_venues:
            return []

        venue_summary = "\n".join(
            f"{i+1}. {v['name']} ({v['label']}), {v['distance_km']:.1f} km away, address: {v['address']}"
            for i, v in enumerate(top_venues)
        )

        prompt = f"""
You are writing short, warm explanations for why real local venues match an older adult's
social preferences, to help reduce their social isolation risk.

Person's preferences:
- Interested activities: {', '.join(prefs['interested_activities']) if prefs['interested_activities'] else 'no specific preference'}
- Preferred group size: {prefs['group_preference']}
- Goal of attending: {prefs['event_goal']}
- Volunteering interests: {', '.join(prefs['volunteering_interests']) if prefs['volunteering_interests'] else 'not specified'}

Here are {len(top_venues)} REAL, verified local venues (do not invent any other venues, dates, or events):
{venue_summary}

For each venue listed above, in the same order, write a 2-3 sentence explanation of why this
type of place fits the person's stated preferences and how regular visits could help reduce
social isolation. Do NOT invent specific event names, dates, or times — just explain the fit
of the venue itself, and suggest they check its current schedule.

Respond with ONLY a valid JSON array (no markdown fences, no extra text), with exactly
{len(top_venues)} objects, one per venue in order, in this structure:

[
  {{"description": "2-3 sentence explanation"}}
]
"""
        model_llm = genai.GenerativeModel("gemini-3.1-flash-lite")
        response = model_llm.generate_content(prompt)
        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()
        descriptions = json.loads(raw_text)

        for v, d in zip(top_venues, descriptions):
            v["description"] = d.get("description", "")

        return top_venues

    if "recommendations_data" not in st.session_state:
        with st.spinner("Finding real local venues near you..."):
            try:
                st.session_state.recommendations_data = fetch_recommendations()
            except json.JSONDecodeError:
                st.error("Couldn't generate descriptions right now. Please try refreshing.")
                st.session_state.recommendations_data = []
            except Exception as e:
                st.error("Something went wrong finding local venues. Please check your internet connection.")
                st.exception(e)
                st.session_state.recommendations_data = []

    recs = st.session_state.recommendations_data

    if not recs:
        st.warning(
            "No matching venues were found near this ZIP code. This can happen in less "
            "densely-mapped areas on OpenStreetMap — try a nearby ZIP code, or check back later."
        )

    for i, v in enumerate(recs, start=1):
        st.markdown(f"### {i}. {v['name']}")
        st.markdown("**✅ Verified real location (OpenStreetMap)**")
        st.write(f"🏷️ **Type:** {v['label']}")
        st.write(f"📍 **Address:** {v['address']} ({v['distance_km']:.1f} km away)")
        if v.get("website"):
            st.write(f"🔗 **Website:** {v['website']}")
        if v.get("phone"):
            st.write(f"📞 **Phone:** {v['phone']}")
        if not v.get("website") and not v.get("phone"):
            st.write("ℹ️ No contact info listed — search the name online or visit in person for current hours.")
        st.write(v.get("description", ""))

        projected_score = estimate_pirs_with_recommendation(answers, v.get("impact_feature", ""))
        if projected_score is not None:
            current_score = st.session_state.pirs_score
            delta = projected_score - current_score
            st.info(
                f"💡 If {subject} visited regularly, the estimated score could shift to about "
                f"**{projected_score:.0f}/100** (a change of {delta:+.0f} points). This is a rough "
                f"projection, not a guarantee."
            )
        st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Get New Recommendations"):
            del st.session_state.recommendations_data
            st.rerun()
    with col2:
        if st.button("Start Over"):
            for key in ["page", "raw_answers", "preferences", "pirs_score", "recommendations_data"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# ---------------------------------------------------------
# SHARED FEATURE EXTRACTION FUNCTION
# (identical logic to Day 1 Colab notebook — keep in sync)
# ---------------------------------------------------------
def extract_features(raw_df):
    out = pd.DataFrame(index=raw_df.index)

    out["age_norm"] = np.clip((raw_df["age"] - 60) / (95 - 60), 0, 1)

    out["lives_alone_risk"] = raw_df["lives_alone"].map({"No": 0.0, "Yes": 1.0})
    out["marital_status_risk"] = raw_df["marital_status"].map(
        {"Married": 0.0, "Single": 0.5, "Divorced": 0.7, "Widowed": 1.0}
    )

    out["family_visits_risk"] = np.clip(1 - raw_df["weekly_family_visits"] / 7, 0, 1)
    out["friend_interactions_risk"] = np.clip(1 - raw_df["weekly_friend_interactions"] / 7, 0, 1)
    out["video_calls_risk"] = raw_df["uses_video_calls"].map(
        {"Daily": 0.0, "Weekly": 0.25, "Monthly": 0.60, "Never": 1.0}
    )

    out["community_events_risk"] = np.clip(1 - raw_df["community_events_attended"] / 4, 0, 1)
    out["volunteer_hours_risk"] = np.clip(1 - raw_df["volunteer_hours"] / 10, 0, 1)
    out["religious_hours_risk"] = np.clip(1 - raw_df["religious_participation_hours"] / 5, 0, 1)

    out["transportation_risk"] = raw_df["transportation_access"].map(
        {"Can drive": 0.0, "Family": 0.3, "Public transit": 0.5, "None": 1.0}
    )
    out["mobility_risk"] = raw_df["mobility_limitations"].map(
        {"None": 0.0, "Mild": 0.33, "Moderate": 0.67, "Severe": 1.0}
    )

    out["exercise_days_risk"] = np.clip(1 - raw_df["exercise_days"] / 7, 0, 1)
    out["depression_risk"] = raw_df["depression_symptoms"].map(
        {"Low": 0.0, "Moderate": 0.5, "High": 1.0}
    )

    return out[feature_columns]  # enforce exact column order the model expects

if st.session_state.page == "intro":
    show_intro_page()
elif st.session_state.page == "questionnaire1":
    show_questionnaire1_page()
elif st.session_state.page == "results":
    show_results_page()
elif st.session_state.page == "questionnaire2":
    show_questionnaire2_page()
elif st.session_state.page == "recommendations":
    show_recommendations_page()