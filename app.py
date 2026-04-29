import re
import os
import html
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="WhatsApp Community Report",
    page_icon="💬",
    layout="wide"
)

APP_TITLE = "WhatsApp Community Daily Intelligence Report"
STUDENT_PHONE_FILE = "students_phone.xlsx"


# =========================
# KEYWORD RULES
# =========================

TOPIC_KEYWORDS = {
    "Admissions": [
        "admission", "apply", "application", "accepted", "admitted", "offer letter",
        "selection", "interview", "seat", "enrol", "enroll", "joining"
    ],
    "Payment / Fees": [
        "payment", "paid", "pay", "fee", "fees", "deposit", "refund", "invoice",
        "scholarship", "loan", "deadline", "installment", "emi"
    ],
    "Travel / Campus": [
        "travel", "flight", "visa", "passport", "campus", "country", "destination",
        "india", "uae", "europe", "ghana", "argentina", "new york", "nyc"
    ],
    "Program / Curriculum": [
        "program", "course", "curriculum", "module", "class", "degree",
        "university", "credit", "assignment", "term", "semester"
    ],
    "Events / Webinars": [
        "event", "webinar", "masterclass", "session", "call", "zoom",
        "meet", "meeting", "orientation", "workshop"
    ],
    "Community / Social": [
        "group", "community", "friends", "meetup", "introduction", "intro",
        "fun", "party", "hangout"
    ],
    "Career / Placements": [
        "career", "placement", "job", "internship", "startup", "founder",
        "business", "networking", "mentor", "mentorship"
    ],
    "Support / Operations": [
        "help", "support", "issue", "problem", "error", "link", "form",
        "document", "documents", "access", "not working"
    ]
}

QUESTION_WORDS = [
    "what", "when", "where", "why", "how", "who", "which",
    "can", "could", "should", "would", "is", "are", "do", "does",
    "did", "will", "any update", "anyone know", "please tell"
]

HIGH_INTENT_KEYWORDS = [
    "payment", "pay", "paid", "deposit", "confirm", "confirmed",
    "seat", "deadline", "join", "joining", "enroll", "enrol",
    "admission", "offer", "fee", "fees", "scholarship", "loan",
    "call me", "dm me", "interested", "ready", "how to pay"
]

NEGATIVE_KEYWORDS = [
    "confused", "doubt", "worried", "concern", "problem", "issue",
    "not clear", "not sure", "scam", "fake", "expensive", "refund",
    "bad", "delay", "late", "unhappy", "disappointed", "doesn't make sense",
    "not working", "can't", "cannot", "no response", "ignored"
]

POSITIVE_KEYWORDS = [
    "great", "amazing", "excited", "happy", "thanks", "thank you",
    "helpful", "good", "awesome", "love", "perfect", "clear",
    "looking forward", "super", "nice"
]

ANNOUNCEMENT_KEYWORDS = [
    "announcement", "important", "note", "please note", "reminder",
    "deadline", "update", "everyone", "all", "kindly", "please fill",
    "form", "session", "today", "tomorrow"
]


# =========================
# PHONE MATCHING
# =========================

def normalize_phone(value):
    """
    Normalize phone numbers for matching.

    Examples:
    +91 98765 43210 -> 919876543210
    9876543210 -> 9876543210
    +1 (805) 878-6137 -> 18058786137
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()
    digits = re.sub(r"\D+", "", text)

    if not digits:
        return ""

    # Remove leading 00 international prefix.
    if digits.startswith("00"):
        digits = digits[2:]

    return digits


def phone_match_keys(value):
    """
    Create multiple keys for robust matching.
    Exact full number + last 10 digits.
    """
    normalized = normalize_phone(value)

    if not normalized:
        return []

    keys = {normalized}

    if len(normalized) >= 10:
        keys.add(normalized[-10:])

    if len(normalized) >= 11:
        keys.add(normalized[-11:])

    return list(keys)


def extract_phone_from_sender(sender):
    """
    Extract phone number from WhatsApp sender field.
    Sender can be:
    +91 98765 43210
    +1 (805) 878-6137
    Harashit Mitra
    """
    sender = str(sender).strip()

    # Only treat as phone-like if it has at least 8 digits.
    digits = re.sub(r"\D+", "", sender)

    if len(digits) >= 8:
        return normalize_phone(sender)

    return ""


@st.cache_data(show_spinner=False)
def load_students_phone_mapping(file_path=STUDENT_PHONE_FILE):
    """
    Reads students_phone.xlsx from repo and creates phone -> student mapping.
    Auto-detects name and phone columns.
    """
    if not os.path.exists(file_path):
        return {}, pd.DataFrame(), "students_phone.xlsx not found in repo."

    try:
        students_df = pd.read_excel(file_path)
    except Exception as e:
        return {}, pd.DataFrame(), f"Could not read students_phone.xlsx: {e}"

    if students_df.empty:
        return {}, students_df, "students_phone.xlsx is empty."

    original_cols = list(students_df.columns)
    lower_cols = {str(c).strip().lower(): c for c in students_df.columns}

    name_candidates = [
        "student name", "name", "full name", "student", "student_name",
        "full_name", "lead name", "candidate name"
    ]

    phone_candidates = [
        "phone", "phone number", "mobile", "mobile number",
        "whatsapp", "whatsapp number", "contact", "number",
        "contact number", "phone_number", "mobile_number"
    ]

    name_col = None
    phone_col = None

    for c in name_candidates:
        if c in lower_cols:
            name_col = lower_cols[c]
            break

    for c in phone_candidates:
        if c in lower_cols:
            phone_col = lower_cols[c]
            break

    if name_col is None:
        return {}, students_df, f"Could not detect student name column. Found columns: {original_cols}"

    if phone_col is None:
        return {}, students_df, f"Could not detect phone column. Found columns: {original_cols}"

    mapping = {}

    for _, row in students_df.iterrows():
        student_name = str(row.get(name_col, "")).strip()
        phone_value = row.get(phone_col, "")

        if not student_name or student_name.lower() in ["nan", "none"]:
            continue

        for key in phone_match_keys(phone_value):
            if key:
                mapping[key] = student_name

    students_df["_Detected_Name_Column"] = name_col
    students_df["_Detected_Phone_Column"] = phone_col

    return mapping, students_df, ""


def match_student_name(sender, phone_mapping):
    sender_phone = extract_phone_from_sender(sender)

    if not sender_phone:
        return ""

    keys = phone_match_keys(sender_phone)

    for key in keys:
        if key in phone_mapping:
            return phone_mapping[key]

    return ""


def enrich_with_student_names(df, phone_mapping):
    df = df.copy()

    df["SenderPhone"] = df["Sender"].apply(extract_phone_from_sender)
    df["MatchedStudentName"] = df["Sender"].apply(lambda x: match_student_name(x, phone_mapping))

    df["DisplayName"] = np.where(
        df["MatchedStudentName"].astype(str).str.strip() != "",
        df["MatchedStudentName"],
        df["Sender"]
    )

    df["IsStudentMatched"] = df["MatchedStudentName"].astype(str).str.strip() != ""

    return df


# =========================
# FILE / PARSING HELPERS
# =========================

def decode_file(uploaded_file):
    raw = uploaded_file.read()

    for enc in ["utf-8", "utf-8-sig", "utf-16", "latin-1"]:
        try:
            return raw.decode(enc)
        except Exception:
            continue

    return raw.decode("utf-8", errors="ignore")


def parse_datetime(date_str, time_str):
    date_str = str(date_str).strip()
    time_str = str(time_str).strip().replace("\u202f", " ")

    candidates = [
        f"{date_str} {time_str}",
        f"{date_str}, {time_str}",
    ]

    for value in candidates:
        parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")
        if not pd.isna(parsed):
            return parsed.to_pydatetime()

    for value in candidates:
        parsed = pd.to_datetime(value, dayfirst=False, errors="coerce")
        if not pd.isna(parsed):
            return parsed.to_pydatetime()

    return None


def split_sender_message(body):
    if ": " in body:
        sender, message = body.split(": ", 1)
        return sender.strip(), message.strip(), False

    return "System", body.strip(), True


def parse_whatsapp_export(text, community_name):
    """
    Supports common Android and iPhone WhatsApp exports.

    Android:
    12/03/2026, 18:30 - Name: Message

    iPhone:
    [12/03/2026, 18:30:00] Name: Message
    [12/03/26, 6:30:00 PM] Name: Message
    """

    patterns = [
        re.compile(
            r"^(?P<date>\d{1,2}[/-]\d{1,2}[/-]\d{2,4}),?\s+"
            r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?)\s-\s"
            r"(?P<body>.*)$"
        ),
        re.compile(
            r"^\[(?P<date>\d{1,2}[/-]\d{1,2}[/-]\d{2,4}),?\s+"
            r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?)\]\s"
            r"(?P<body>.*)$"
        )
    ]

    rows = []
    current = None

    for line in text.splitlines():
        line = line.strip("\ufeff").rstrip()

        matched = None

        for pattern in patterns:
            m = pattern.match(line)
            if m:
                matched = m
                break

        if matched:
            if current is not None:
                rows.append(current)

            dt = parse_datetime(matched.group("date"), matched.group("time"))
            body = matched.group("body").strip()
            sender, message, is_system = split_sender_message(body)

            current = {
                "Community": community_name,
                "DateTime": dt,
                "Sender": sender,
                "Message": message,
                "IsSystem": is_system,
                "RawLine": line
            }

        else:
            if current is not None:
                current["Message"] += "\n" + line
            elif line:
                rows.append({
                    "Community": community_name,
                    "DateTime": None,
                    "Sender": "Unknown",
                    "Message": line,
                    "IsSystem": True,
                    "RawLine": line
                })

    if current is not None:
        rows.append(current)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.dropna(subset=["DateTime"]).copy()

    if df.empty:
        return df

    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["Date"] = df["DateTime"].dt.date
    df["Time"] = df["DateTime"].dt.strftime("%H:%M")
    df["Hour"] = df["DateTime"].dt.hour
    df["Message"] = df["Message"].fillna("").astype(str)
    df["Sender"] = df["Sender"].fillna("Unknown").astype(str)

    return df


# =========================
# CLASSIFICATION HELPERS
# =========================

def contains_any(text, keywords):
    text = str(text).lower()
    return any(k.lower() in text for k in keywords)


def is_question(message):
    text = str(message).strip().lower()

    if "?" in text:
        return True

    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in QUESTION_WORDS)


def detect_topics(message):
    text = str(message).lower()
    topics = []

    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword.lower() in text for keyword in keywords):
            topics.append(topic)

    return topics if topics else ["General"]


def sentiment_label(message):
    text = str(message).lower()

    neg = sum(1 for k in NEGATIVE_KEYWORDS if k in text)
    pos = sum(1 for k in POSITIVE_KEYWORDS if k in text)

    if neg > pos:
        return "Negative"

    if pos > neg:
        return "Positive"

    return "Neutral"


def has_link(message):
    return bool(re.search(r"https?://|www\.|chat\.whatsapp\.com", str(message).lower()))


def is_media(message):
    text = str(message).lower()

    media_markers = [
        "<media omitted>",
        "image omitted",
        "video omitted",
        "audio omitted",
        "sticker omitted",
        "gif omitted",
        "document omitted",
        "voice message omitted"
    ]

    return any(marker in text for marker in media_markers)


def is_deleted(message):
    text = str(message).lower()
    return "this message was deleted" in text or "you deleted this message" in text


def is_join_event(message):
    text = str(message).lower()

    return (
        "joined using this group's invite link" in text
        or "joined using this community's invite link" in text
        or "was added" in text
        or " added " in f" {text} "
    )


def is_left_event(message):
    text = str(message).lower()

    return (
        " left" in text
        or "was removed" in text
        or " removed " in f" {text} "
    )


def extract_features(df):
    df = df.copy()

    df["IsQuestion"] = df["Message"].apply(is_question)
    df["HasLink"] = df["Message"].apply(has_link)
    df["IsMedia"] = df["Message"].apply(is_media)
    df["IsDeleted"] = df["Message"].apply(is_deleted)
    df["IsJoinEvent"] = df["Message"].apply(is_join_event)
    df["IsLeftEvent"] = df["Message"].apply(is_left_event)
    df["IsHighIntent"] = df["Message"].apply(lambda x: contains_any(x, HIGH_INTENT_KEYWORDS))
    df["Sentiment"] = df["Message"].apply(sentiment_label)
    df["Topics"] = df["Message"].apply(detect_topics)
    df["TopicPrimary"] = df["Topics"].apply(lambda x: x[0] if x else "General")
    df["MessageLength"] = df["Message"].str.len()

    return df


def mark_answered_questions(df, admin_names=None, response_window_hours=4):
    df = df.copy()

    df["QuestionAnswered"] = False
    df["AnsweredBy"] = ""
    df["NeedsFollowUp"] = False

    admin_names = [a.strip().lower() for a in admin_names or [] if a.strip()]

    question_indices = df.index[df["IsQuestion"] & (~df["IsSystem"])].tolist()

    for idx in question_indices:
        row = df.loc[idx]
        start_time = row["DateTime"]
        end_time = start_time + pd.Timedelta(hours=response_window_hours)

        future = df[
            (df["Community"] == row["Community"])
            & (df["DateTime"] > start_time)
            & (df["DateTime"] <= end_time)
            & (~df["IsSystem"])
            & (df.index != idx)
        ].copy()

        if future.empty:
            df.at[idx, "NeedsFollowUp"] = True
            continue

        if admin_names:
            future["DisplayLower"] = future["DisplayName"].astype(str).str.lower().str.strip()
            future["SenderLower"] = future["Sender"].astype(str).str.lower().str.strip()

            admin_reply = future[
                future["DisplayLower"].isin(admin_names)
                | future["SenderLower"].isin(admin_names)
            ]

            if not admin_reply.empty:
                first_reply = admin_reply.iloc[0]
                df.at[idx, "QuestionAnswered"] = True
                df.at[idx, "AnsweredBy"] = first_reply["DisplayName"]
            else:
                df.at[idx, "NeedsFollowUp"] = True

        else:
            future = future[future["Sender"] != row["Sender"]]

            if not future.empty:
                first_reply = future.iloc[0]
                df.at[idx, "QuestionAnswered"] = True
                df.at[idx, "AnsweredBy"] = first_reply["DisplayName"]
            else:
                df.at[idx, "NeedsFollowUp"] = True

    return df


# =========================
# METRICS
# =========================

def calculate_health_score(
    total_messages,
    active_members,
    unanswered,
    negative_cases,
    high_intent,
    questions
):
    score = 50

    if total_messages >= 100:
        score += 15
    elif total_messages >= 50:
        score += 10
    elif total_messages >= 20:
        score += 5

    if active_members >= 25:
        score += 15
    elif active_members >= 10:
        score += 10
    elif active_members >= 5:
        score += 5

    if questions > 0:
        unanswered_rate = unanswered / questions

        if unanswered_rate <= 0.1:
            score += 10
        elif unanswered_rate <= 0.3:
            score += 5
        else:
            score -= 10

    if high_intent >= 10:
        score += 10
    elif high_intent >= 3:
        score += 5

    if negative_cases >= 10:
        score -= 15
    elif negative_cases >= 3:
        score -= 8

    return max(0, min(100, score))


def health_label(score):
    if score >= 75:
        return "Strong"
    if score >= 55:
        return "Moderate"
    if score >= 35:
        return "Low"
    return "Risk"


def build_daily_community_metrics(df):
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby(["Date", "Community"], dropna=False)
    records = []

    for (dt, community), g in grouped:
        non_system = g[~g["IsSystem"]]

        questions = g[g["IsQuestion"] & (~g["IsSystem"])]
        answered = questions[questions["QuestionAnswered"]]
        unanswered = questions[~questions["QuestionAnswered"]]

        sentiment_counts = non_system["Sentiment"].value_counts().to_dict()

        topic_counts = non_system["TopicPrimary"].value_counts()
        top_topic = topic_counts.index[0] if not topic_counts.empty else "General"

        active_members = non_system["DisplayName"].nunique()

        hour_counts = non_system["Hour"].value_counts()
        peak_hour = int(hour_counts.idxmax()) if not hour_counts.empty else ""

        negative_cases = non_system[non_system["Sentiment"] == "Negative"].shape[0]
        high_intent = non_system[non_system["IsHighIntent"]].shape[0]
        total_messages = len(non_system)
        questions_count = len(questions)

        health_score = calculate_health_score(
            total_messages=total_messages,
            active_members=active_members,
            unanswered=len(unanswered),
            negative_cases=negative_cases,
            high_intent=high_intent,
            questions=questions_count
        )

        matched_students = non_system["IsStudentMatched"].sum()
        unique_matched_students = non_system[non_system["IsStudentMatched"]]["MatchedStudentName"].nunique()

        records.append({
            "Date": dt,
            "Community": community,
            "Total Messages": total_messages,
            "Active Members": active_members,
            "Matched Student Messages": int(matched_students),
            "Unique Matched Students": int(unique_matched_students),
            "System Messages": int(g["IsSystem"].sum()),
            "New Member Events": int(g["IsJoinEvent"].sum()),
            "Left / Removed Events": int(g["IsLeftEvent"].sum()),
            "Media Shared": int(g["IsMedia"].sum()),
            "Links Shared": int(g["HasLink"].sum()),
            "Deleted Messages": int(g["IsDeleted"].sum()),
            "Questions Asked": len(questions),
            "Questions Answered": len(answered),
            "Unanswered Questions": len(unanswered),
            "High Intent Messages": high_intent,
            "Negative Cases": negative_cases,
            "Positive Messages": sentiment_counts.get("Positive", 0),
            "Neutral Messages": sentiment_counts.get("Neutral", 0),
            "Negative Messages": sentiment_counts.get("Negative", 0),
            "Peak Activity Hour": peak_hour,
            "Main Topic": top_topic,
            "Community Health Score": health_score,
            "Overall Health": health_label(health_score)
        })

    return pd.DataFrame(records).sort_values(["Date", "Community"])


def build_topic_summary(df):
    if df.empty:
        return pd.DataFrame()

    exploded = df[~df["IsSystem"]].explode("Topics")

    return (
        exploded
        .groupby(["Date", "Community", "Topics"])
        .size()
        .reset_index(name="Message Count")
        .rename(columns={"Topics": "Topic"})
        .sort_values(["Date", "Community", "Message Count"], ascending=[True, True, False])
    )


def build_top_members(df):
    if df.empty:
        return pd.DataFrame()

    non_system = df[~df["IsSystem"]]

    return (
        non_system
        .groupby(["Date", "Community", "DisplayName", "Sender", "MatchedStudentName", "IsStudentMatched"])
        .size()
        .reset_index(name="Messages")
        .sort_values(["Date", "Community", "Messages"], ascending=[True, True, False])
    )


def build_hourly_summary(df):
    if df.empty:
        return pd.DataFrame()

    return (
        df[~df["IsSystem"]]
        .groupby(["Date", "Community", "Hour"])
        .size()
        .reset_index(name="Messages")
        .sort_values(["Date", "Community", "Hour"])
    )


def get_questions_tracker(df):
    q = df[(~df["IsSystem"]) & (df["IsQuestion"])].copy()

    if q.empty:
        return pd.DataFrame(columns=[
            "Date", "Time", "Community", "DisplayName", "Sender",
            "MatchedStudentName", "Message", "QuestionAnswered", "AnsweredBy", "NeedsFollowUp"
        ])

    return q[[
        "Date", "Time", "Community", "DisplayName", "Sender",
        "MatchedStudentName", "Message", "QuestionAnswered", "AnsweredBy", "NeedsFollowUp"
    ]].sort_values(["Date", "Time", "Community"])


def get_high_intent_tracker(df):
    h = df[(~df["IsSystem"]) & (df["IsHighIntent"])].copy()

    if h.empty:
        return pd.DataFrame(columns=[
            "Date", "Time", "Community", "DisplayName", "Sender",
            "MatchedStudentName", "Message", "Sentiment"
        ])

    return h[[
        "Date", "Time", "Community", "DisplayName", "Sender",
        "MatchedStudentName", "Message", "Sentiment"
    ]].sort_values(["Date", "Time", "Community"])


def get_negative_tracker(df):
    n = df[(~df["IsSystem"]) & (df["Sentiment"] == "Negative")].copy()

    if n.empty:
        return pd.DataFrame(columns=[
            "Date", "Time", "Community", "DisplayName", "Sender",
            "MatchedStudentName", "Message"
        ])

    return n[[
        "Date", "Time", "Community", "DisplayName", "Sender",
        "MatchedStudentName", "Message"
    ]].sort_values(["Date", "Time", "Community"])


def get_important_messages(df, limit=200):
    if df.empty:
        return pd.DataFrame()

    important = df[
        (~df["IsSystem"])
        & (
            df["IsQuestion"]
            | df["IsHighIntent"]
            | (df["Sentiment"] == "Negative")
        )
    ].copy()

    if important.empty:
        return pd.DataFrame(columns=[
            "Date", "Time", "Community", "DisplayName", "Sender",
            "MatchedStudentName", "Category", "Sentiment",
            "QuestionAnswered", "NeedsFollowUp", "Message"
        ])

    important["Category"] = np.select(
        [
            important["IsQuestion"],
            important["IsHighIntent"],
            important["Sentiment"].eq("Negative")
        ],
        [
            "Question",
            "High Intent",
            "Negative / Risk"
        ],
        default="Important"
    )

    cols = [
        "Date", "Time", "Community", "DisplayName", "Sender",
        "MatchedStudentName", "Category", "Sentiment",
        "QuestionAnswered", "NeedsFollowUp", "Message"
    ]

    return important[cols].sort_values(["Date", "Time"]).head(limit)


def detect_admin_announcements(df, admin_names=None):
    admin_names = [a.strip().lower() for a in admin_names or [] if a.strip()]
    data = df[~df["IsSystem"]].copy()

    if admin_names:
        data["DisplayLower"] = data["DisplayName"].astype(str).str.lower().str.strip()
        data["SenderLower"] = data["Sender"].astype(str).str.lower().str.strip()

        data = data[
            data["DisplayLower"].isin(admin_names)
            | data["SenderLower"].isin(admin_names)
        ]

    data = data[data["Message"].apply(lambda x: contains_any(x, ANNOUNCEMENT_KEYWORDS))]

    if data.empty:
        return pd.DataFrame(columns=[
            "Date", "Time", "Community", "DisplayName", "Sender", "Message"
        ])

    return data[[
        "Date", "Time", "Community", "DisplayName", "Sender", "Message"
    ]].sort_values(["Date", "Time", "Community"])


# =========================
# EXPORT HELPERS
# =========================

def clean_for_excel(df):
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype(str)

    return out


def build_excel_report(
    metrics_df,
    topic_summary,
    top_members,
    questions_df,
    high_intent_df,
    negative_df,
    important_df,
    admin_announcements,
    raw_df
):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        clean_for_excel(metrics_df).to_excel(writer, sheet_name="Daily Metrics", index=False)
        clean_for_excel(topic_summary).to_excel(writer, sheet_name="Topic Summary", index=False)
        clean_for_excel(top_members).to_excel(writer, sheet_name="Top Members", index=False)
        clean_for_excel(questions_df).to_excel(writer, sheet_name="Questions", index=False)
        clean_for_excel(high_intent_df).to_excel(writer, sheet_name="High Intent", index=False)
        clean_for_excel(negative_df).to_excel(writer, sheet_name="Negative Cases", index=False)
        clean_for_excel(important_df).to_excel(writer, sheet_name="Important Messages", index=False)
        clean_for_excel(admin_announcements).to_excel(writer, sheet_name="Admin Announcements", index=False)

        export_raw = raw_df.copy()
        export_raw["Topics"] = export_raw["Topics"].astype(str)
        clean_for_excel(export_raw).to_excel(writer, sheet_name="Parsed Raw Data", index=False)

    output.seek(0)
    return output


def df_to_html_table(df, title):
    if df is None or df.empty:
        return f"""
        <section>
            <h2>{html.escape(title)}</h2>
            <p class="muted">No data available.</p>
        </section>
        """

    safe_df = df.copy()

    for col in safe_df.columns:
        safe_df[col] = safe_df[col].astype(str).map(html.escape)

    return f"""
    <section>
        <h2>{html.escape(title)}</h2>
        {safe_df.to_html(index=False, escape=False, classes="report-table")}
    </section>
    """


def build_html_report(
    title,
    filtered_df,
    metrics_df,
    topic_summary,
    top_members,
    questions_df,
    high_intent_df,
    negative_df,
    important_df,
    admin_announcements,
    start_date,
    end_date
):
    generated_at = datetime.now().strftime("%d %b %Y, %I:%M %p")

    total_messages = int(metrics_df["Total Messages"].sum()) if not metrics_df.empty else 0
    active_members = filtered_df[~filtered_df["IsSystem"]]["DisplayName"].nunique() if not filtered_df.empty else 0
    total_questions = int(metrics_df["Questions Asked"].sum()) if not metrics_df.empty else 0
    unanswered = int(metrics_df["Unanswered Questions"].sum()) if not metrics_df.empty else 0
    high_intent = int(metrics_df["High Intent Messages"].sum()) if not metrics_df.empty else 0
    negative = int(metrics_df["Negative Cases"].sum()) if not metrics_df.empty else 0
    matched_students = filtered_df[
        (~filtered_df["IsSystem"]) & (filtered_df["IsStudentMatched"])
    ]["MatchedStudentName"].nunique() if not filtered_df.empty else 0

    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{html.escape(title)}</title>
        <style>
            body {{
                font-family: Arial, Helvetica, sans-serif;
                margin: 0;
                padding: 0;
                background: #f5f7fb;
                color: #172033;
            }}
            .container {{
                max-width: 1180px;
                margin: 0 auto;
                padding: 32px;
            }}
            .header {{
                background: linear-gradient(135deg, #172033, #31415f);
                color: white;
                padding: 32px;
                border-radius: 18px;
                margin-bottom: 24px;
            }}
            .header h1 {{
                margin: 0 0 10px 0;
                font-size: 30px;
            }}
            .header p {{
                margin: 4px 0;
                opacity: 0.9;
            }}
            .kpi-grid {{
                display: grid;
                grid-template-columns: repeat(7, 1fr);
                gap: 14px;
                margin-bottom: 26px;
            }}
            .kpi {{
                background: white;
                border-radius: 16px;
                padding: 18px;
                box-shadow: 0 8px 24px rgba(23, 32, 51, 0.08);
            }}
            .kpi .label {{
                color: #667085;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }}
            .kpi .value {{
                font-size: 28px;
                font-weight: 700;
                margin-top: 8px;
            }}
            section {{
                background: white;
                padding: 24px;
                border-radius: 18px;
                margin-bottom: 22px;
                box-shadow: 0 8px 24px rgba(23, 32, 51, 0.08);
            }}
            h2 {{
                margin-top: 0;
                color: #172033;
                font-size: 22px;
                border-bottom: 1px solid #e7eaf0;
                padding-bottom: 10px;
            }}
            .report-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }}
            .report-table th {{
                background: #eef2f7;
                color: #172033;
                text-align: left;
                padding: 10px;
                border: 1px solid #d9dee8;
            }}
            .report-table td {{
                padding: 9px 10px;
                border: 1px solid #e4e7ee;
                vertical-align: top;
            }}
            .report-table tr:nth-child(even) {{
                background: #fafbfc;
            }}
            .muted {{
                color: #667085;
            }}
            .footer {{
                text-align: center;
                color: #667085;
                margin-top: 28px;
                font-size: 12px;
            }}
            @media print {{
                body {{
                    background: white;
                }}
                .container {{
                    padding: 0;
                }}
                section, .kpi {{
                    box-shadow: none;
                    border: 1px solid #e4e7ee;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{html.escape(title)}</h1>
                <p><strong>Reporting Period:</strong> {start_date.strftime("%d %b %Y")} to {end_date.strftime("%d %b %Y")}</p>
                <p><strong>Generated:</strong> {generated_at}</p>
            </div>

            <div class="kpi-grid">
                <div class="kpi"><div class="label">Messages</div><div class="value">{total_messages}</div></div>
                <div class="kpi"><div class="label">Active Members</div><div class="value">{active_members}</div></div>
                <div class="kpi"><div class="label">Matched Students</div><div class="value">{matched_students}</div></div>
                <div class="kpi"><div class="label">Questions</div><div class="value">{total_questions}</div></div>
                <div class="kpi"><div class="label">Unanswered</div><div class="value">{unanswered}</div></div>
                <div class="kpi"><div class="label">High Intent</div><div class="value">{high_intent}</div></div>
                <div class="kpi"><div class="label">Negative Cases</div><div class="value">{negative}</div></div>
            </div>

            <section>
                <h2>Executive Summary</h2>
                <p>
                    This report summarises WhatsApp community activity for the selected period.
                    It includes message activity, active members, matched student names, topics, questions,
                    high-intent signals, negative sentiment cases, announcements, and pending follow-ups.
                </p>
            </section>

            {df_to_html_table(metrics_df, "Daily Performance")}
            {df_to_html_table(topic_summary, "Topic Summary")}
            {df_to_html_table(top_members.head(50), "Top Active Members")}
            {df_to_html_table(questions_df, "Questions Tracker")}
            {df_to_html_table(high_intent_df, "High Intent / Lead Signals")}
            {df_to_html_table(negative_df, "Negative Sentiment / Risk Cases")}
            {df_to_html_table(admin_announcements, "Admin Announcements")}
            {df_to_html_table(important_df, "Important Conversations")}

            <div class="footer">
                Generated automatically from WhatsApp chat exports.
            </div>
        </div>
    </body>
    </html>
    """

    return html_report


# =========================
# UI RENDERING
# =========================

def render_kpis(metrics_df, filtered_df):
    total_messages = int(metrics_df["Total Messages"].sum()) if not metrics_df.empty else 0
    active_members = filtered_df[~filtered_df["IsSystem"]]["DisplayName"].nunique() if not filtered_df.empty else 0
    matched_students = filtered_df[
        (~filtered_df["IsSystem"]) & (filtered_df["IsStudentMatched"])
    ]["MatchedStudentName"].nunique() if not filtered_df.empty else 0
    total_questions = int(metrics_df["Questions Asked"].sum()) if not metrics_df.empty else 0
    unanswered = int(metrics_df["Unanswered Questions"].sum()) if not metrics_df.empty else 0
    high_intent = int(metrics_df["High Intent Messages"].sum()) if not metrics_df.empty else 0
    negative = int(metrics_df["Negative Cases"].sum()) if not metrics_df.empty else 0

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

    c1.metric("Messages", total_messages)
    c2.metric("Active Members", active_members)
    c3.metric("Matched Students", matched_students)
    c4.metric("Questions", total_questions)
    c5.metric("Unanswered", unanswered)
    c6.metric("High Intent", high_intent)
    c7.metric("Negative Cases", negative)


def render_charts(filtered_df, metrics_df, topic_summary, hourly_summary, title_prefix=""):
    st.subheader("Visual Analytics")

    col1, col2 = st.columns(2)

    with col1:
        if not metrics_df.empty:
            daily = metrics_df.groupby("Date")["Total Messages"].sum().reset_index()

            fig = px.bar(
                daily,
                x="Date",
                y="Total Messages",
                title=f"{title_prefix}Daily Messages"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if not topic_summary.empty:
            topic_total = (
                topic_summary
                .groupby("Topic")["Message Count"]
                .sum()
                .reset_index()
                .sort_values("Message Count", ascending=False)
            )

            fig = px.pie(
                topic_total,
                names="Topic",
                values="Message Count",
                title=f"{title_prefix}Topic Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        if not filtered_df.empty:
            sentiment = (
                filtered_df[~filtered_df["IsSystem"]]
                .groupby("Sentiment")
                .size()
                .reset_index(name="Messages")
            )

            fig = px.bar(
                sentiment,
                x="Sentiment",
                y="Messages",
                title=f"{title_prefix}Sentiment Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        if not hourly_summary.empty:
            hourly_total = (
                hourly_summary
                .groupby("Hour")["Messages"]
                .sum()
                .reset_index()
            )

            fig = px.line(
                hourly_total,
                x="Hour",
                y="Messages",
                markers=True,
                title=f"{title_prefix}Hourly Activity"
            )
            st.plotly_chart(fig, use_container_width=True)


def prepare_report_data(df):
    metrics_df = build_daily_community_metrics(df)
    topic_summary = build_topic_summary(df)
    top_members = build_top_members(df)
    hourly_summary = build_hourly_summary(df)
    questions_df = get_questions_tracker(df)
    high_intent_df = get_high_intent_tracker(df)
    negative_df = get_negative_tracker(df)
    important_df = get_important_messages(df, limit=200)

    return {
        "metrics_df": metrics_df,
        "topic_summary": topic_summary,
        "top_members": top_members,
        "hourly_summary": hourly_summary,
        "questions_df": questions_df,
        "high_intent_df": high_intent_df,
        "negative_df": negative_df,
        "important_df": important_df,
    }


def render_community_page(
    community_name,
    community_df,
    admin_names,
    start_date,
    end_date
):
    st.header(f"📌 {community_name}")

    data = prepare_report_data(community_df)

    metrics_df = data["metrics_df"]
    topic_summary = data["topic_summary"]
    top_members = data["top_members"]
    hourly_summary = data["hourly_summary"]
    questions_df = data["questions_df"]
    high_intent_df = data["high_intent_df"]
    negative_df = data["negative_df"]
    important_df = data["important_df"]
    admin_announcements = detect_admin_announcements(community_df, admin_names=admin_names)

    render_kpis(metrics_df, community_df)

    st.markdown("---")

    day_count = (end_date - start_date).days + 1

    if day_count == 2:
        st.subheader("Two-Day Comparison for This Community")

        daily = metrics_df.groupby("Date").sum(numeric_only=True).reset_index()

        if daily["Date"].nunique() == 2:
            d1, d2 = sorted(daily["Date"].unique())

            comparison_metrics = [
                "Total Messages",
                "Active Members",
                "Questions Asked",
                "Questions Answered",
                "Unanswered Questions",
                "High Intent Messages",
                "Negative Cases",
                "Media Shared",
                "Links Shared",
                "Matched Student Messages",
                "Unique Matched Students"
            ]

            rows = []

            row1 = daily[daily["Date"] == d1].iloc[0]
            row2 = daily[daily["Date"] == d2].iloc[0]

            for metric in comparison_metrics:
                v1 = int(row1.get(metric, 0))
                v2 = int(row2.get(metric, 0))
                change = v2 - v1

                pct_change = ""
                if v1 != 0:
                    pct_change = f"{((v2 - v1) / v1) * 100:.1f}%"

                rows.append({
                    "Metric": metric,
                    d1.strftime("%d %b %Y"): v1,
                    d2.strftime("%d %b %Y"): v2,
                    "Change": change,
                    "% Change": pct_change
                })

            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        else:
            st.info("Only one selected day has data for this community.")

    elif day_count >= 3:
        st.subheader("Date-Wise Data for This Community")
        daily = metrics_df.groupby("Date").sum(numeric_only=True).reset_index()

        if not daily.empty:
            selected_metric = st.selectbox(
                f"Select trend metric for {community_name}",
                [
                    "Total Messages",
                    "Active Members",
                    "Questions Asked",
                    "Unanswered Questions",
                    "High Intent Messages",
                    "Negative Cases",
                    "Unique Matched Students"
                ],
                key=f"trend_metric_{community_name}"
            )

            if selected_metric in daily.columns:
                fig = px.line(
                    daily,
                    x="Date",
                    y=selected_metric,
                    markers=True,
                    title=f"{community_name} - {selected_metric} Trend"
                )
                st.plotly_chart(fig, use_container_width=True)

        st.dataframe(metrics_df, use_container_width=True)

    else:
        st.subheader("Single-Day Data for This Community")
        st.dataframe(metrics_df, use_container_width=True)

    render_charts(
        filtered_df=community_df,
        metrics_df=metrics_df,
        topic_summary=topic_summary,
        hourly_summary=hourly_summary,
        title_prefix=f"{community_name} - "
    )

    st.markdown("---")

    sub_tabs = st.tabs([
        "Daily Metrics",
        "Topics",
        "Top Members",
        "Matched Students",
        "Questions",
        "High Intent",
        "Negative Cases",
        "Announcements",
        "Important Conversations",
        "Raw Data"
    ])

    with sub_tabs[0]:
        st.dataframe(metrics_df, use_container_width=True)

    with sub_tabs[1]:
        st.dataframe(topic_summary, use_container_width=True)

    with sub_tabs[2]:
        top_n = st.slider(
            f"Top N members - {community_name}",
            min_value=5,
            max_value=50,
            value=10,
            key=f"top_n_{community_name}"
        )

        display_top = (
            top_members
            .groupby(["Date", "Community"])
            .head(top_n)
            .reset_index(drop=True)
        )

        st.dataframe(display_top, use_container_width=True)

    with sub_tabs[3]:
        matched = community_df[
            (~community_df["IsSystem"])
            & (community_df["IsStudentMatched"])
        ][[
            "Date", "Time", "Community", "DisplayName", "Sender",
            "SenderPhone", "MatchedStudentName", "Message"
        ]].copy()

        unmatched = community_df[
            (~community_df["IsSystem"])
            & (~community_df["IsStudentMatched"])
        ][[
            "Date", "Time", "Community", "Sender",
            "SenderPhone", "Message"
        ]].copy()

        c1, c2 = st.columns(2)
        c1.metric("Matched Student Messages", len(matched))
        c2.metric("Unmatched Messages", len(unmatched))

        st.markdown("#### Matched Student Messages")
        st.dataframe(matched, use_container_width=True)

        st.markdown("#### Unmatched Senders")
        st.dataframe(unmatched, use_container_width=True)

    with sub_tabs[4]:
        st.dataframe(questions_df, use_container_width=True)

        unanswered = questions_df[questions_df["QuestionAnswered"] == False]
        if not unanswered.empty:
            st.warning(f"{len(unanswered)} unanswered question(s) detected.")
            st.dataframe(unanswered, use_container_width=True)

    with sub_tabs[5]:
        st.dataframe(high_intent_df, use_container_width=True)

    with sub_tabs[6]:
        st.dataframe(negative_df, use_container_width=True)

    with sub_tabs[7]:
        st.dataframe(admin_announcements, use_container_width=True)

    with sub_tabs[8]:
        st.dataframe(important_df, use_container_width=True)

    with sub_tabs[9]:
        raw_cols = [
            "DateTime", "Date", "Time", "Community", "Sender", "DisplayName",
            "SenderPhone", "MatchedStudentName", "IsStudentMatched",
            "Message", "IsSystem", "IsQuestion", "QuestionAnswered",
            "NeedsFollowUp", "IsHighIntent", "Sentiment", "TopicPrimary",
            "HasLink", "IsMedia"
        ]

        st.dataframe(community_df[raw_cols], use_container_width=True)

    st.markdown("---")

    st.subheader(f"Download Report - {community_name}")

    html_report = build_html_report(
        title=f"{community_name} - WhatsApp Community Report",
        filtered_df=community_df,
        metrics_df=metrics_df,
        topic_summary=topic_summary,
        top_members=top_members,
        questions_df=questions_df,
        high_intent_df=high_intent_df,
        negative_df=negative_df,
        important_df=important_df,
        admin_announcements=admin_announcements,
        start_date=start_date,
        end_date=end_date
    )

    excel_report = build_excel_report(
        metrics_df=metrics_df,
        topic_summary=topic_summary,
        top_members=top_members,
        questions_df=questions_df,
        high_intent_df=high_intent_df,
        negative_df=negative_df,
        important_df=important_df,
        admin_announcements=admin_announcements,
        raw_df=community_df
    )

    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", community_name).strip("_")
    report_name_base = f"{safe_name}_report_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"

    c1, c2 = st.columns(2)

    with c1:
        st.download_button(
            label="Download HTML Report",
            data=html_report.encode("utf-8"),
            file_name=f"{report_name_base}.html",
            mime="text/html",
            use_container_width=True,
            key=f"html_{community_name}"
        )

    with c2:
        st.download_button(
            label="Download Excel Report",
            data=excel_report,
            file_name=f"{report_name_base}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key=f"excel_{community_name}"
        )


def render_all_groups_comparison(
    filtered_df,
    admin_names,
    start_date,
    end_date
):
    st.header("📊 All Groups Comparison")

    data = prepare_report_data(filtered_df)

    metrics_df = data["metrics_df"]
    topic_summary = data["topic_summary"]
    top_members = data["top_members"]
    hourly_summary = data["hourly_summary"]
    questions_df = data["questions_df"]
    high_intent_df = data["high_intent_df"]
    negative_df = data["negative_df"]
    important_df = data["important_df"]
    admin_announcements = detect_admin_announcements(filtered_df, admin_names=admin_names)

    if metrics_df.empty:
        st.info("No comparison data available.")
        return

    comparison = (
        metrics_df
        .groupby("Community")
        .agg({
            "Total Messages": "sum",
            "Active Members": "sum",
            "Unique Matched Students": "sum",
            "Questions Asked": "sum",
            "Questions Answered": "sum",
            "Unanswered Questions": "sum",
            "High Intent Messages": "sum",
            "Negative Cases": "sum",
            "Media Shared": "sum",
            "Links Shared": "sum",
            "New Member Events": "sum",
            "Left / Removed Events": "sum",
            "Community Health Score": "mean"
        })
        .reset_index()
    )

    comparison["Community Health Score"] = comparison["Community Health Score"].round(1)
    comparison["Overall Health"] = comparison["Community Health Score"].apply(health_label)

    st.subheader("Group-Wise Comparison Table")
    st.dataframe(comparison, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            comparison.sort_values("Total Messages", ascending=False),
            x="Community",
            y="Total Messages",
            title="Total Messages by Group"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            comparison.sort_values("Active Members", ascending=False),
            x="Community",
            y="Active Members",
            title="Active Members by Group"
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.bar(
            comparison.sort_values("High Intent Messages", ascending=False),
            x="Community",
            y="High Intent Messages",
            title="High Intent Messages by Group"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.bar(
            comparison.sort_values("Negative Cases", ascending=False),
            x="Community",
            y="Negative Cases",
            title="Negative Cases by Group"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Date-Wise Group Comparison")

    daily_group = (
        metrics_df
        .groupby(["Date", "Community"])
        .sum(numeric_only=True)
        .reset_index()
    )

    metric_options = [
        "Total Messages",
        "Active Members",
        "Unique Matched Students",
        "Questions Asked",
        "Unanswered Questions",
        "High Intent Messages",
        "Negative Cases"
    ]

    selected_metric = st.selectbox(
        "Select metric to compare across groups",
        metric_options,
        key="all_groups_metric"
    )

    if selected_metric in daily_group.columns:
        fig = px.line(
            daily_group,
            x="Date",
            y=selected_metric,
            color="Community",
            markers=True,
            title=f"Date-Wise Comparison: {selected_metric}"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(daily_group, use_container_width=True)

    st.markdown("---")

    st.subheader("Cross-Group Detailed Tables")

    tabs = st.tabs([
        "Daily Metrics",
        "Topics",
        "Top Members",
        "Questions",
        "High Intent",
        "Negative Cases",
        "Announcements",
        "Important Conversations"
    ])

    with tabs[0]:
        st.dataframe(metrics_df, use_container_width=True)

    with tabs[1]:
        st.dataframe(topic_summary, use_container_width=True)

    with tabs[2]:
        st.dataframe(top_members, use_container_width=True)

    with tabs[3]:
        st.dataframe(questions_df, use_container_width=True)

    with tabs[4]:
        st.dataframe(high_intent_df, use_container_width=True)

    with tabs[5]:
        st.dataframe(negative_df, use_container_width=True)

    with tabs[6]:
        st.dataframe(admin_announcements, use_container_width=True)

    with tabs[7]:
        st.dataframe(important_df, use_container_width=True)

    st.markdown("---")

    st.subheader("Download All Groups Comparison Report")

    html_report = build_html_report(
        title="All Groups Comparison - WhatsApp Community Report",
        filtered_df=filtered_df,
        metrics_df=metrics_df,
        topic_summary=topic_summary,
        top_members=top_members,
        questions_df=questions_df,
        high_intent_df=high_intent_df,
        negative_df=negative_df,
        important_df=important_df,
        admin_announcements=admin_announcements,
        start_date=start_date,
        end_date=end_date
    )

    excel_report = build_excel_report(
        metrics_df=metrics_df,
        topic_summary=topic_summary,
        top_members=top_members,
        questions_df=questions_df,
        high_intent_df=high_intent_df,
        negative_df=negative_df,
        important_df=important_df,
        admin_announcements=admin_announcements,
        raw_df=filtered_df
    )

    report_name_base = f"all_groups_comparison_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"

    c1, c2 = st.columns(2)

    with c1:
        st.download_button(
            label="Download All Groups HTML Report",
            data=html_report.encode("utf-8"),
            file_name=f"{report_name_base}.html",
            mime="text/html",
            use_container_width=True,
            key="all_groups_html"
        )

    with c2:
        st.download_button(
            label="Download All Groups Excel Report",
            data=excel_report,
            file_name=f"{report_name_base}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="all_groups_excel"
        )


# =========================
# MAIN APP
# =========================

def main():
    st.title("💬 WhatsApp Community Report Generator")
    st.caption(
        "Upload 5 or more WhatsApp chat exports, select a date range, open each community separately, "
        "and compare all groups in one dedicated comparison tab."
    )

    phone_mapping, students_df, phone_error = load_students_phone_mapping()

    with st.sidebar:
        st.header("Upload & Settings")

        uploaded_files = st.file_uploader(
            "Upload WhatsApp chat exports",
            type=["txt"],
            accept_multiple_files=True,
            help="Upload exports from 5 or more WhatsApp communities."
        )

        st.markdown("---")

        if phone_error:
            st.warning(phone_error)
        else:
            st.success(f"Loaded students_phone.xlsx · {len(phone_mapping):,} phone match keys")

            with st.expander("Preview detected student phone sheet"):
                preview_cols = [
                    c for c in students_df.columns
                    if not str(c).startswith("_Detected")
                ]
                st.dataframe(students_df[preview_cols].head(20), use_container_width=True)

        st.markdown("---")

        admin_input = st.text_area(
            "Admin names / numbers",
            placeholder="Enter one admin name or number per line",
            help="Used to identify admin replies and announcements. Optional."
        )

        response_window_hours = st.slider(
            "Question response window",
            min_value=1,
            max_value=24,
            value=4,
            help="A question is treated as answered if a reply appears within this window."
        )

        st.markdown("---")
        st.caption("Tip: export WhatsApp chats as `.txt` without media for faster upload.")

    if not uploaded_files:
        st.info("Upload WhatsApp `.txt` exports to start.")
        return

    if len(uploaded_files) < 5:
        st.warning(
            f"You uploaded {len(uploaded_files)} file(s). The app supports it, "
            "but your use case mentions 5 or more communities."
        )

    all_dfs = []

    with st.spinner("Parsing WhatsApp exports and matching student names..."):
        for file in uploaded_files:
            community_name = file.name.replace(".txt", "").replace("_", " ").strip()

            text = decode_file(file)
            parsed = parse_whatsapp_export(text, community_name)

            if parsed.empty:
                st.warning(f"No valid messages detected in: {file.name}")
            else:
                all_dfs.append(parsed)

    if not all_dfs:
        st.error("No valid WhatsApp messages found. Please check the export format.")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.sort_values("DateTime").reset_index(drop=True)

    df = enrich_with_student_names(df, phone_mapping)
    df = extract_features(df)

    admin_names = [x.strip() for x in admin_input.splitlines() if x.strip()]

    df = mark_answered_questions(
        df,
        admin_names=admin_names,
        response_window_hours=response_window_hours
    )

    min_date = df["Date"].min()
    max_date = df["Date"].max()

    st.markdown("---")

    top_left, top_right = st.columns([2, 1])

    with top_left:
        selected_range = st.date_input(
            "Select report date range",
            value=(max_date, max_date),
            min_value=min_date,
            max_value=max_date,
            format="DD/MM/YYYY"
        )

    with top_right:
        st.write("")
        st.write("")
        st.success(f"Loaded {len(uploaded_files)} communities · {len(df):,} parsed messages")

    if isinstance(selected_range, tuple):
        if len(selected_range) == 1:
            start_date = selected_range[0]
            end_date = selected_range[0]
        elif len(selected_range) >= 2:
            start_date = selected_range[0]
            end_date = selected_range[1]
        else:
            start_date = max_date
            end_date = max_date
    else:
        start_date = selected_range
        end_date = selected_range

    if start_date > end_date:
        st.error("Start date cannot be after end date.")
        return

    filtered_df = df[
        (df["Date"] >= start_date)
        & (df["Date"] <= end_date)
    ].copy()

    if filtered_df.empty:
        st.warning("No messages found for the selected date range.")
        return

    communities = sorted(filtered_df["Community"].dropna().unique().tolist())

    if not communities:
        st.warning("No communities found.")
        return

    tab_labels = communities + ["All Groups Comparison"]
    main_tabs = st.tabs(tab_labels)

    for i, community_name in enumerate(communities):
        with main_tabs[i]:
            community_df = filtered_df[filtered_df["Community"] == community_name].copy()

            render_community_page(
                community_name=community_name,
                community_df=community_df,
                admin_names=admin_names,
                start_date=start_date,
                end_date=end_date
            )

    with main_tabs[-1]:
        render_all_groups_comparison(
            filtered_df=filtered_df,
            admin_names=admin_names,
            start_date=start_date,
            end_date=end_date
        )


if __name__ == "__main__":
    main()
