import re
import html
from io import BytesIO
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="WhatsApp Community 24H Report",
    page_icon="💬",
    layout="wide"
)

APP_TITLE = "WhatsApp Community Daily Intelligence Report"


# =========================
# TOPIC / SENTIMENT RULES
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
# PARSING HELPERS
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
    """
    Supports common WhatsApp formats:
    12/03/2026, 18:30
    12/03/26, 6:30 PM
    03/12/2026, 18:30
    """
    date_str = date_str.strip()
    time_str = time_str.strip().replace("\u202f", " ")

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
    """
    If body contains 'Sender: Message', split it.
    Otherwise treat as system message.
    """
    if ": " in body:
        sender, message = body.split(": ", 1)
        return sender.strip(), message.strip(), False

    return "System", body.strip(), True


def parse_whatsapp_export(text, community_name):
    """
    Handles Android and iPhone exports:
    Android:
    12/03/2026, 18:30 - Name: Message

    iPhone:
    [12/03/2026, 18:30:00] Name: Message
    [12/03/26, 6:30:00 PM] Name: Message

    Also supports multiline messages.
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

    lines = text.splitlines()

    for line in lines:
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
        or "added" in text
    )


def is_left_event(message):
    text = str(message).lower()
    return (
        "left" in text
        or "was removed" in text
        or "removed" in text
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
    """
    Marks questions as answered when:
    - If admin list is provided: an admin replies after the question in same community within window.
    - If no admin list: any different sender replies after the question within window.
    """
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
            future["SenderLower"] = future["Sender"].str.lower().str.strip()
            admin_reply = future[future["SenderLower"].isin(admin_names)]

            if not admin_reply.empty:
                first_reply = admin_reply.iloc[0]
                df.at[idx, "QuestionAnswered"] = True
                df.at[idx, "AnsweredBy"] = first_reply["Sender"]
            else:
                df.at[idx, "NeedsFollowUp"] = True
        else:
            future = future[future["Sender"] != row["Sender"]]
            if not future.empty:
                first_reply = future.iloc[0]
                df.at[idx, "QuestionAnswered"] = True
                df.at[idx, "AnsweredBy"] = first_reply["Sender"]
            else:
                df.at[idx, "NeedsFollowUp"] = True

    return df


# =========================
# METRICS
# =========================

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

        top_topic = "General"
        topic_counts = non_system["TopicPrimary"].value_counts()
        if not topic_counts.empty:
            top_topic = topic_counts.index[0]

        active_members = non_system["Sender"].nunique()

        peak_hour = ""
        hour_counts = non_system["Hour"].value_counts()
        if not hour_counts.empty:
            peak_hour = int(hour_counts.idxmax())

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

        records.append({
            "Date": dt,
            "Community": community,
            "Total Messages": total_messages,
            "Active Members": active_members,
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


def build_topic_summary(df):
    if df.empty:
        return pd.DataFrame()

    exploded = df[~df["IsSystem"]].explode("Topics")
    summary = (
        exploded
        .groupby(["Date", "Community", "Topics"])
        .size()
        .reset_index(name="Message Count")
        .sort_values(["Date", "Community", "Message Count"], ascending=[True, True, False])
    )

    return summary.rename(columns={"Topics": "Topic"})


def build_top_members(df):
    if df.empty:
        return pd.DataFrame()

    non_system = df[~df["IsSystem"]]

    return (
        non_system
        .groupby(["Date", "Community", "Sender"])
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


def get_important_messages(df, limit=100):
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
        "Date", "Time", "Community", "Sender", "Category",
        "Sentiment", "QuestionAnswered", "NeedsFollowUp", "Message"
    ]

    return important[cols].sort_values(["Date", "Time"]).head(limit)


def get_questions_tracker(df):
    q = df[(~df["IsSystem"]) & (df["IsQuestion"])].copy()

    if q.empty:
        return pd.DataFrame(columns=[
            "Date", "Time", "Community", "Sender", "Message",
            "QuestionAnswered", "AnsweredBy", "NeedsFollowUp"
        ])

    return q[[
        "Date", "Time", "Community", "Sender", "Message",
        "QuestionAnswered", "AnsweredBy", "NeedsFollowUp"
    ]].sort_values(["Date", "Time", "Community"])


def get_high_intent_tracker(df):
    h = df[(~df["IsSystem"]) & (df["IsHighIntent"])].copy()

    if h.empty:
        return pd.DataFrame(columns=[
            "Date", "Time", "Community", "Sender", "Message", "Sentiment"
        ])

    return h[[
        "Date", "Time", "Community", "Sender", "Message", "Sentiment"
    ]].sort_values(["Date", "Time", "Community"])


def get_negative_tracker(df):
    n = df[(~df["IsSystem"]) & (df["Sentiment"] == "Negative")].copy()

    if n.empty:
        return pd.DataFrame(columns=[
            "Date", "Time", "Community", "Sender", "Message"
        ])

    return n[[
        "Date", "Time", "Community", "Sender", "Message"
    ]].sort_values(["Date", "Time", "Community"])


def detect_admin_announcements(df, admin_names=None):
    admin_names = [a.strip().lower() for a in admin_names or [] if a.strip()]
    data = df[~df["IsSystem"]].copy()

    if admin_names:
        data = data[data["Sender"].str.lower().str.strip().isin(admin_names)]

    data = data[data["Message"].apply(lambda x: contains_any(x, ANNOUNCEMENT_KEYWORDS))]

    if data.empty:
        return pd.DataFrame(columns=[
            "Date", "Time", "Community", "Sender", "Message"
        ])

    return data[[
        "Date", "Time", "Community", "Sender", "Message"
    ]].sort_values(["Date", "Time", "Community"])


# =========================
# REPORT EXPORTS
# =========================

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
    active_members = filtered_df[~filtered_df["IsSystem"]]["Sender"].nunique() if not filtered_df.empty else 0
    total_questions = int(metrics_df["Questions Asked"].sum()) if not metrics_df.empty else 0
    unanswered = int(metrics_df["Unanswered Questions"].sum()) if not metrics_df.empty else 0
    high_intent = int(metrics_df["High Intent Messages"].sum()) if not metrics_df.empty else 0
    negative = int(metrics_df["Negative Cases"].sum()) if not metrics_df.empty else 0

    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{html.escape(APP_TITLE)}</title>
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
                grid-template-columns: repeat(6, 1fr);
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
                <h1>{html.escape(APP_TITLE)}</h1>
                <p><strong>Reporting Period:</strong> {start_date.strftime("%d %b %Y")} to {end_date.strftime("%d %b %Y")}</p>
                <p><strong>Generated:</strong> {generated_at}</p>
            </div>

            <div class="kpi-grid">
                <div class="kpi"><div class="label">Messages</div><div class="value">{total_messages}</div></div>
                <div class="kpi"><div class="label">Active Members</div><div class="value">{active_members}</div></div>
                <div class="kpi"><div class="label">Questions</div><div class="value">{total_questions}</div></div>
                <div class="kpi"><div class="label">Unanswered</div><div class="value">{unanswered}</div></div>
                <div class="kpi"><div class="label">High Intent</div><div class="value">{high_intent}</div></div>
                <div class="kpi"><div class="label">Negative Cases</div><div class="value">{negative}</div></div>
            </div>

            <section>
                <h2>Executive Summary</h2>
                <p>
                    This report summarises WhatsApp community activity for the selected reporting period.
                    It includes message activity, active members, topic trends, questions, high-intent signals,
                    negative sentiment cases, admin announcements, and pending follow-ups.
                </p>
                <p>
                    <strong>Recommended focus:</strong>
                    prioritize unanswered questions, high-intent members, and negative sentiment cases for follow-up.
                </p>
            </section>

            {df_to_html_table(metrics_df, "Community-Wise Daily Performance")}
            {df_to_html_table(topic_summary, "Topic Summary")}
            {df_to_html_table(top_members.head(50), "Top Active Members")}
            {df_to_html_table(questions_df, "Questions Tracker")}
            {df_to_html_table(high_intent_df, "High Intent / Lead Signals")}
            {df_to_html_table(negative_df, "Negative Sentiment / Risk Cases")}
            {df_to_html_table(admin_announcements, "Admin Announcements / Important Updates")}
            {df_to_html_table(important_df, "Important Conversations")}

            <div class="footer">
                Generated automatically from WhatsApp chat exports.
            </div>
        </div>
    </body>
    </html>
    """

    return html_report


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
        metrics_df.to_excel(writer, sheet_name="Daily Community Metrics", index=False)
        topic_summary.to_excel(writer, sheet_name="Topic Summary", index=False)
        top_members.to_excel(writer, sheet_name="Top Members", index=False)
        questions_df.to_excel(writer, sheet_name="Questions Tracker", index=False)
        high_intent_df.to_excel(writer, sheet_name="High Intent", index=False)
        negative_df.to_excel(writer, sheet_name="Negative Cases", index=False)
        important_df.to_excel(writer, sheet_name="Important Messages", index=False)
        admin_announcements.to_excel(writer, sheet_name="Admin Announcements", index=False)

        export_raw = raw_df.copy()
        export_raw["Topics"] = export_raw["Topics"].astype(str)
        export_raw.to_excel(writer, sheet_name="Parsed Raw Data", index=False)

    output.seek(0)
    return output


# =========================
# UI HELPERS
# =========================

def kpi_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)


def render_overview_kpis(metrics_df, filtered_df):
    total_messages = int(metrics_df["Total Messages"].sum()) if not metrics_df.empty else 0
    active_members = filtered_df[~filtered_df["IsSystem"]]["Sender"].nunique() if not filtered_df.empty else 0
    total_questions = int(metrics_df["Questions Asked"].sum()) if not metrics_df.empty else 0
    unanswered = int(metrics_df["Unanswered Questions"].sum()) if not metrics_df.empty else 0
    high_intent = int(metrics_df["High Intent Messages"].sum()) if not metrics_df.empty else 0
    negative = int(metrics_df["Negative Cases"].sum()) if not metrics_df.empty else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        kpi_card("Total Messages", total_messages)
    with c2:
        kpi_card("Active Members", active_members)
    with c3:
        kpi_card("Questions Asked", total_questions)
    with c4:
        kpi_card("Unanswered", unanswered)
    with c5:
        kpi_card("High Intent", high_intent)
    with c6:
        kpi_card("Negative Cases", negative)


def render_two_day_comparison(metrics_df, selected_dates):
    st.subheader("Two-Day Comparison")

    if metrics_df.empty:
        st.info("No data available for comparison.")
        return

    day1, day2 = sorted(selected_dates)

    daily = (
        metrics_df
        .groupby("Date")
        .sum(numeric_only=True)
        .reset_index()
    )

    day1_row = daily[daily["Date"] == day1]
    day2_row = daily[daily["Date"] == day2]

    if day1_row.empty or day2_row.empty:
        st.warning("One of the selected days has no data.")
        return

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
        "New Member Events",
        "Left / Removed Events"
    ]

    rows = []
    for metric in comparison_metrics:
        v1 = int(day1_row.iloc[0].get(metric, 0))
        v2 = int(day2_row.iloc[0].get(metric, 0))
        delta = v2 - v1
        pct = ""
        if v1 != 0:
            pct = f"{((v2 - v1) / v1) * 100:.1f}%"

        rows.append({
            "Metric": metric,
            day1.strftime("%d %b %Y"): v1,
            day2.strftime("%d %b %Y"): v2,
            "Change": delta,
            "% Change": pct
        })

    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, use_container_width=True)

    plot_df = metrics_df.groupby("Date")[
        ["Total Messages", "Questions Asked", "High Intent Messages", "Negative Cases"]
    ].sum().reset_index()

    plot_long = plot_df.melt(id_vars="Date", var_name="Metric", value_name="Count")

    fig = px.bar(
        plot_long,
        x="Metric",
        y="Count",
        color="Date",
        barmode="group",
        title="Two-Day Metric Comparison"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_multi_day_analysis(metrics_df):
    st.subheader("Date-Wise Trend Analysis")

    if metrics_df.empty:
        st.info("No data available.")
        return

    daily = metrics_df.groupby("Date").sum(numeric_only=True).reset_index()

    trend_metrics = [
        "Total Messages",
        "Active Members",
        "Questions Asked",
        "Unanswered Questions",
        "High Intent Messages",
        "Negative Cases"
    ]

    available_metrics = [m for m in trend_metrics if m in daily.columns]

    selected_metric = st.selectbox(
        "Select trend metric",
        available_metrics,
        index=0
    )

    fig = px.line(
        daily,
        x="Date",
        y=selected_metric,
        markers=True,
        title=f"Daily Trend: {selected_metric}"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(daily, use_container_width=True)


def render_charts(filtered_df, metrics_df, topic_summary, hourly_summary):
    st.subheader("Visual Analytics")

    col1, col2 = st.columns(2)

    with col1:
        if not metrics_df.empty:
            msg_by_community = (
                metrics_df
                .groupby("Community")["Total Messages"]
                .sum()
                .reset_index()
                .sort_values("Total Messages", ascending=False)
            )
            fig = px.bar(
                msg_by_community,
                x="Community",
                y="Total Messages",
                title="Messages by Community"
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
                title="Topic Distribution"
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
                title="Sentiment Distribution"
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
                title="Hourly Activity"
            )
            st.plotly_chart(fig, use_container_width=True)


# =========================
# MAIN APP
# =========================

def main():
    st.title("💬 WhatsApp Community 24H Report Generator")
    st.caption(
        "Upload 5 or more WhatsApp chat exports, select a date range, compare days, "
        "and download a professional community report."
    )

    with st.sidebar:
        st.header("Upload & Settings")

        uploaded_files = st.file_uploader(
            "Upload WhatsApp chat exports",
            type=["txt"],
            accept_multiple_files=True,
            help="Upload exports from 5 or more WhatsApp communities."
        )

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
        st.caption("Tip: export each WhatsApp community as `.txt` without media for faster upload.")

    if not uploaded_files:
        st.info("Upload WhatsApp `.txt` exports to start.")
        return

    if len(uploaded_files) < 5:
        st.warning(
            f"You uploaded {len(uploaded_files)} file(s). The app supports this, "
            "but your use case mentions 5 or more communities."
        )

    all_dfs = []

    with st.spinner("Parsing WhatsApp exports..."):
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

    metrics_df = build_daily_community_metrics(filtered_df)
    topic_summary = build_topic_summary(filtered_df)
    top_members = build_top_members(filtered_df)
    hourly_summary = build_hourly_summary(filtered_df)
    questions_df = get_questions_tracker(filtered_df)
    high_intent_df = get_high_intent_tracker(filtered_df)
    negative_df = get_negative_tracker(filtered_df)
    important_df = get_important_messages(filtered_df, limit=200)
    admin_announcements = detect_admin_announcements(filtered_df, admin_names=admin_names)

    day_count = (end_date - start_date).days + 1
    selected_dates = sorted(filtered_df["Date"].unique())

    render_overview_kpis(metrics_df, filtered_df)

    st.markdown("---")

    if day_count == 2:
        render_two_day_comparison(metrics_df, selected_dates)
    elif day_count >= 3:
        render_multi_day_analysis(metrics_df)
    else:
        st.subheader("Single-Day Report")
        st.dataframe(metrics_df, use_container_width=True)

    st.markdown("---")

    render_charts(filtered_df, metrics_df, topic_summary, hourly_summary)

    st.markdown("---")

    tabs = st.tabs([
        "Daily Metrics",
        "Topics",
        "Top Members",
        "Questions",
        "High Intent",
        "Negative Cases",
        "Admin Announcements",
        "Important Conversations",
        "Raw Parsed Data"
    ])

    with tabs[0]:
        st.subheader("Community-Wise Daily Metrics")
        st.dataframe(metrics_df, use_container_width=True)

    with tabs[1]:
        st.subheader("Topic Summary")
        st.dataframe(topic_summary, use_container_width=True)

    with tabs[2]:
        st.subheader("Top Active Members")
        top_n = st.slider("Show top N members per date/community", 5, 50, 10)
        display_top_members = (
            top_members
            .groupby(["Date", "Community"])
            .head(top_n)
            .reset_index(drop=True)
        )
        st.dataframe(display_top_members, use_container_width=True)

    with tabs[3]:
        st.subheader("Questions Tracker")
        st.dataframe(questions_df, use_container_width=True)

        unanswered_df = questions_df[questions_df["QuestionAnswered"] == False]
        if not unanswered_df.empty:
            st.warning(f"{len(unanswered_df)} unanswered question(s) detected.")
            st.dataframe(unanswered_df, use_container_width=True)

    with tabs[4]:
        st.subheader("High Intent / Lead Signals")
        st.dataframe(high_intent_df, use_container_width=True)

    with tabs[5]:
        st.subheader("Negative Sentiment / Risk Cases")
        st.dataframe(negative_df, use_container_width=True)

    with tabs[6]:
        st.subheader("Admin Announcements / Important Updates")
        st.dataframe(admin_announcements, use_container_width=True)

    with tabs[7]:
        st.subheader("Important Conversations")
        st.dataframe(important_df, use_container_width=True)

    with tabs[8]:
        st.subheader("Raw Parsed Data")
        raw_cols = [
            "DateTime", "Date", "Time", "Community", "Sender", "Message",
            "IsSystem", "IsQuestion", "QuestionAnswered", "NeedsFollowUp",
            "IsHighIntent", "Sentiment", "TopicPrimary", "HasLink", "IsMedia"
        ]
        st.dataframe(filtered_df[raw_cols], use_container_width=True)

    st.markdown("---")

    st.subheader("Download Professional Report")

    html_report = build_html_report(
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

    report_name_base = f"whatsapp_community_report_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"

    c1, c2 = st.columns(2)

    with c1:
        st.download_button(
            label="Download Professional HTML Report",
            data=html_report.encode("utf-8"),
            file_name=f"{report_name_base}.html",
            mime="text/html",
            use_container_width=True
        )

    with c2:
        st.download_button(
            label="Download Excel Data Report",
            data=excel_report,
            file_name=f"{report_name_base}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    st.caption(
        "HTML report is designed for leadership sharing and can be opened in browser or printed as PDF. "
        "Excel report contains all tables for deeper analysis."
    )


if __name__ == "__main__":
    main()
