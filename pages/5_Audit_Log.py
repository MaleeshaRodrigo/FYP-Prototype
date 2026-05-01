"""
Researcher view of immutable audit events.
"""

from __future__ import annotations

import streamlit as st

from auth_utils import require_role
from database import db


st.set_page_config(page_title="HARE Audit Log", page_icon="🧾", layout="wide")

require_role("researcher")

st.title("Audit Log")
st.caption("Append-only hash-chained audit events for critical system activity.")

event_filter = st.text_input("Filter by event type contains")
limit = st.slider("Rows", min_value=25, max_value=500, value=100, step=25)

if event_filter:
    rows = db.fetch_all(
        """
        SELECT id, created_at, event_type, actor_user_id, target_resource, success,
               previous_hash, current_hash, details
        FROM audit_events
        WHERE event_type LIKE ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (f"%{event_filter}%", limit),
    )
else:
    rows = db.fetch_all(
        """
        SELECT id, created_at, event_type, actor_user_id, target_resource, success,
               previous_hash, current_hash, details
        FROM audit_events
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )

st.dataframe(rows, use_container_width=True, hide_index=True)
