import os
import io
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from ortools.sat.python import cp_model
from sklearn.preprocessing import MinMaxScaler

# Optional imports used by the application architecture.
# They are included so the project remains compatible with the requested stack.
import requests
from sqlalchemy import create_engine, text


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

APP_TITLE = "AI Timetable Generator"
APP_VERSION = "1.0.0"

DAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
]

DEFAULT_PERIODS = [
    "09:00 - 10:00",
    "10:00 - 11:00",
    "11:00 - 12:00",
    "12:00 - 01:00",
    "02:00 - 03:00",
    "03:00 - 04:00",
    "04:00 - 05:00",
]

BREAK_PERIODS = {
    "12:00 - 01:00",
}


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            background-color: #f7f9fc;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .app-title {
            font-size: 2.5rem;
            font-weight: 800;
            color: #172554;
            margin-bottom: 0.2rem;
        }

        .app-subtitle {
            color: #64748b;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            background: white;
            border-radius: 14px;
            padding: 18px;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.07);
            border: 1px solid #e2e8f0;
        }

        .metric-number {
            font-size: 1.8rem;
            font-weight: 800;
            color: #1d4ed8;
        }

        .metric-label {
            color: #64748b;
            font-size: 0.9rem;
        }

        .subject-card {
            background: white;
            border-left: 5px solid #2563eb;
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 8px;
            box-shadow: 0 1px 5px rgba(0,0,0,0.05);
        }

        .success-box {
            padding: 12px;
            border-radius: 10px;
            background: #ecfdf5;
            border: 1px solid #86efac;
            color: #166534;
        }

        .warning-box {
            padding: 12px;
            border-radius: 10px;
            background: #fffbeb;
            border: 1px solid #fde68a;
            color: #92400e;
        }

        .info-box {
            padding: 12px;
            border-radius: 10px;
            background: #eff6ff;
            border: 1px solid #93c5fd;
            color: #1e40af;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 10px;
        }

        button[kind="primary"] {
            border-radius: 10px;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

def initialize_state():
    """Initialize all application state."""

    if "subjects" not in st.session_state:
        st.session_state.subjects = []

    if "teachers" not in st.session_state:
        st.session_state.teachers = []

    if "rooms" not in st.session_state:
        st.session_state.rooms = []

    if "timetable" not in st.session_state:
        st.session_state.timetable = pd.DataFrame()

    if "generation_stats" not in st.session_state:
        st.session_state.generation_stats = {}

    if "generated" not in st.session_state:
        st.session_state.generated = False


initialize_state()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_int(value, default=1):
    """Convert a value to int safely."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def clean_name(value):
    """Clean a user-provided name."""
    if value is None:
        return ""
    return str(value).strip()


def generate_id(prefix):
    """Generate a lightweight unique ID."""
    return f"{prefix}_{datetime.now().strftime('%H%M%S%f')}"


def get_periods(include_break=True):
    """Return available periods."""
    if include_break:
        return DEFAULT_PERIODS.copy()

    return [
        p for p in DEFAULT_PERIODS
        if p not in BREAK_PERIODS
    ]


def timetable_to_csv(df):
    """Convert timetable dataframe to CSV bytes."""
    output = io.StringIO()
    df.to_csv(output, index=False)
    return output.getvalue().encode("utf-8")


def calculate_subject_hours(subjects):
    """Calculate total required periods."""
    return sum(
        safe_int(subject.get("periods_per_week", 1), 1)
        for subject in subjects
    )


def create_demo_data():
    """Create useful starter data."""

    subjects = [
        {
            "id": generate_id("SUB"),
            "name": "Mathematics",
            "teacher": "Dr. Sharma",
            "room": "Room 101",
            "periods_per_week": 5,
            "preferred_day": "Any",
            "preferred_period": "Any",
        },
        {
            "id": generate_id("SUB"),
            "name": "Physics",
            "teacher": "Dr. Rao",
            "room": "Lab 1",
            "periods_per_week": 4,
            "preferred_day": "Any",
            "preferred_period": "Any",
        },
        {
            "id": generate_id("SUB"),
            "name": "Computer Science",
            "teacher": "Prof. Kumar",
            "room": "Lab 2",
            "periods_per_week": 5,
            "preferred_day": "Any",
            "preferred_period": "Any",
        },
        {
            "id": generate_id("SUB"),
            "name": "English",
            "teacher": "Ms. Priya",
            "room": "Room 102",
            "periods_per_week": 3,
            "preferred_day": "Any",
            "preferred_period": "Any",
        },
        {
            "id": generate_id("SUB"),
            "name": "Chemistry",
            "teacher": "Dr. Mehta",
            "room": "Lab 3",
            "periods_per_week": 4,
            "preferred_day": "Any",
            "preferred_period": "Any",
        },
        {
            "id": generate_id("SUB"),
            "name": "Data Science",
            "teacher": "Prof. Singh",
            "room": "Lab 2",
            "periods_per_week": 3,
            "preferred_day": "Any",
            "preferred_period": "Any",
        },
    ]

    teachers = sorted(
        list(
            {
                subject["teacher"]
                for subject in subjects
            }
        )
    )

    rooms = sorted(
        list(
            {
                subject["room"]
                for subject in subjects
            }
        )
    )

    return subjects, teachers, rooms


# ============================================================
# AI / OPTIMIZATION ENGINE
# ============================================================

class AITimetableGenerator:
    """
    Constraint-based timetable generator.

    OR-Tools CP-SAT handles hard constraints:
    - One subject per timetable slot.
    - A teacher cannot teach two classes simultaneously.
    - A room cannot host two classes simultaneously.
    - Weekly subject periods must be satisfied.

    Soft preferences are represented in the objective:
    - Avoid excessive consecutive classes for teachers.
    - Spread repeated subjects over different days.
    - Respect preferred day/period where possible.
    """

    def __init__(
        self,
        subjects,
        days,
        periods,
        avoid_break=True,
        max_daily_classes=6,
    ):
        self.subjects = subjects
        self.days = days
        self.periods = periods
        self.avoid_break = avoid_break
        self.max_daily_classes = max_daily_classes

        self.num_days = len(days)
        self.num_periods = len(periods)

    def validate_input(self):
        """Validate timetable input."""

        errors = []

        if not self.subjects:
            errors.append("Add at least one subject.")

        teacher_names = set()
        room_names = set()

        for subject in self.subjects:
            name = clean_name(subject.get("name"))
            teacher = clean_name(subject.get("teacher"))
            room = clean_name(subject.get("room"))

            periods_per_week = safe_int(
                subject.get("periods_per_week"),
                0,
            )

            if not name:
                errors.append("Every subject needs a name.")

            if not teacher:
                errors.append(
                    f"Subject '{name}' needs a teacher."
                )

            if not room:
                errors.append(
                    f"Subject '{name}' needs a room."
                )

            if periods_per_week <= 0:
                errors.append(
                    f"'{name}' must have at least 1 period per week."
                )

            teacher_names.add(teacher)
            room_names.add(room)

        total_periods = calculate_subject_hours(self.subjects)

        available_slots = (
            self.num_days *
            self.num_periods
        )

        if total_periods > available_slots:
            errors.append(
                f"Required periods ({total_periods}) exceed "
                f"available timetable slots ({available_slots})."
            )

        if self.max_daily_classes <= 0:
            errors.append(
                "Maximum daily classes must be greater than zero."
            )

        return list(dict.fromkeys(errors))

    def generate(self):
        """Generate an optimized timetable."""

        errors = self.validate_input()

        if errors:
            return None, errors, {}

        model = cp_model.CpModel()

        # ----------------------------------------------------
        # Variables
        # ----------------------------------------------------

        # x[s, d, p] = subject s is assigned to day d / period p
        x = {}

        for s_idx, subject in enumerate(self.subjects):
            for d_idx in range(self.num_days):
                for p_idx in range(self.num_periods):
                    x[s_idx, d_idx, p_idx] = model.NewBoolVar(
                        f"x_{s_idx}_{d_idx}_{p_idx}"
                    )

        # ----------------------------------------------------
        # Hard constraint:
        # Required number of periods per subject
        # ----------------------------------------------------

        for s_idx, subject in enumerate(self.subjects):
            required = safe_int(
                subject.get("periods_per_week"),
                1,
            )

            model.Add(
                sum(
                    x[s_idx, d, p]
                    for d in range(self.num_days)
                    for p in range(self.num_periods)
                ) == required
            )

        # ----------------------------------------------------
        # Hard constraint:
        # Only one class can occupy a timetable slot.
        # ----------------------------------------------------

        for d in range(self.num_days):
            for p in range(self.num_periods):
                model.Add(
                    sum(
                        x[s, d, p]
                        for s in range(len(self.subjects))
                    ) <= 1
                )

        # ----------------------------------------------------
        # Teacher conflict constraint
        # ----------------------------------------------------

        teachers = {}

        for s_idx, subject in enumerate(self.subjects):
            teacher = clean_name(subject.get("teacher"))

            teachers.setdefault(
                teacher,
                [],
            ).append(s_idx)

        for teacher, subject_indexes in teachers.items():
            for d in range(self.num_days):
                for p in range(self.num_periods):
                    model.Add(
                        sum(
                            x[s, d, p]
                            for s in subject_indexes
                        ) <= 1
                    )

        # ----------------------------------------------------
        # Room conflict constraint
        # ----------------------------------------------------

        rooms = {}

        for s_idx, subject in enumerate(self.subjects):
            room = clean_name(subject.get("room"))

            rooms.setdefault(
                room,
                [],
            ).append(s_idx)

        for room, subject_indexes in rooms.items():
            for d in range(self.num_days):
                for p in range(self.num_periods):
                    model.Add(
                        sum(
                            x[s, d, p]
                            for s in subject_indexes
                        ) <= 1
                    )

        # ----------------------------------------------------
        # Daily maximum classes
        # ----------------------------------------------------

        for s_idx in range(len(self.subjects)):
            for d in range(self.num_days):
                model.Add(
                    sum(
                        x[s_idx, d, p]
                        for p in range(self.num_periods)
                    ) <= min(
                        safe_int(
                            self.subjects[s_idx].get(
                                "periods_per_week",
                                1,
                            ),
                            1,
                        ),
                        self.max_daily_classes,
                    )
                )

        # ----------------------------------------------------
        # Soft objective
        # ----------------------------------------------------

        objective_terms = []

        # --------------------------------------------
        # Preference score
        # --------------------------------------------

        for s_idx, subject in enumerate(self.subjects):

            preferred_day = clean_name(
                subject.get("preferred_day", "Any")
            )

            preferred_period = clean_name(
                subject.get("preferred_period", "Any")
            )

            for d_idx, day in enumerate(self.days):
                for p_idx, period in enumerate(self.periods):

                    variable = x[s_idx, d_idx, p_idx]

                    # Prefer requested day.
                    if (
                        preferred_day
                        and preferred_day != "Any"
                    ):
                        if day == preferred_day:
                            objective_terms.append(
                                8 * variable
                            )
                        else:
                            objective_terms.append(
                                -2 * variable
                            )

                    # Prefer requested period.
                    if (
                        preferred_period
                        and preferred_period != "Any"
                    ):
                        if period == preferred_period:
                            objective_terms.append(
                                5 * variable
                            )

                    # Avoid lunch break when enabled.
                    if (
                        self.avoid_break
                        and period in BREAK_PERIODS
                    ):
                        objective_terms.append(
                            -10 * variable
                        )

        # --------------------------------------------
        # Spread repeated subjects over days.
        # --------------------------------------------

        for s_idx, subject in enumerate(self.subjects):

            required = safe_int(
                subject.get("periods_per_week"),
                1,
            )

            if required > 1:

                day_used = []

                for d_idx in range(self.num_days):

                    day_var = model.NewBoolVar(
                        f"day_used_{s_idx}_{d_idx}"
                    )

                    day_used.append(day_var)

                    model.AddMaxEquality(
                        day_var,
                        [
                            x[s_idx, d_idx, p]
                            for p in range(self.num_periods)
                        ],
                    )

                # Reward using multiple days.
                for day_var in day_used:
                    objective_terms.append(
                        3 * day_var
                    )

        # --------------------------------------------
        # Penalize adjacent periods for same teacher.
        # --------------------------------------------

        teacher_penalties = []

        for teacher, subject_indexes in teachers.items():

            for d in range(self.num_days):

                for p in range(self.num_periods - 1):

                    adjacent = model.NewBoolVar(
                        f"teacher_adj_{teacher}_{d}_{p}"
                    )

                    first = sum(
                        x[s, d, p]
                        for s in subject_indexes
                    )

                    second = sum(
                        x[s, d, p + 1]
                        for s in subject_indexes
                    )

                    model.Add(
                        adjacent <= first
                    )

                    model.Add(
                        adjacent <= second
                    )

                    model.Add(
                        adjacent >= first + second - 1
                    )

                    teacher_penalties.append(adjacent)

        for penalty in teacher_penalties:
            objective_terms.append(-1 * penalty)

        # ----------------------------------------------------
        # Maximize overall quality
        # ----------------------------------------------------

        model.Maximize(sum(objective_terms))

        # ----------------------------------------------------
        # Solver
        # ----------------------------------------------------

        solver = cp_model.CpSolver()

        solver.parameters.max_time_in_seconds = 15.0
        solver.parameters.num_search_workers = 8
        solver.parameters.random_seed = 42

        status = solver.Solve(model)

        if status not in (
            cp_model.OPTIMAL,
            cp_model.FEASIBLE,
        ):
            return (
                None,
                [
                    "No feasible timetable could be generated "
                    "with the current constraints."
                ],
                {},
            )

        # ----------------------------------------------------
        # Extract solution
        # ----------------------------------------------------

        rows = []

        for d_idx, day in enumerate(self.days):
            for p_idx, period in enumerate(self.periods):

                for s_idx, subject in enumerate(self.subjects):

                    if solver.Value(
                        x[s_idx, d_idx, p_idx]
                    ):

                        rows.append(
                            {
                                "Day": day,
                                "Period": period,
                                "Subject": subject["name"],
                                "Teacher": subject["teacher"],
                                "Room": subject["room"],
                                "Type": subject.get(
                                    "type",
                                    "Theory",
                                ),
                            }
                        )

        timetable = pd.DataFrame(rows)

        if not timetable.empty:

            day_order = {
                day: i
                for i, day in enumerate(self.days)
            }

            period_order = {
                period: i
                for i, period in enumerate(self.periods)
            }

            timetable["_day_order"] = timetable[
                "Day"
            ].map(day_order)

            timetable["_period_order"] = timetable[
                "Period"
            ].map(period_order)

            timetable = (
                timetable
                .sort_values(
                    ["_day_order", "_period_order"]
                )
                .drop(
                    columns=[
                        "_day_order",
                        "_period_order",
                    ]
                )
                .reset_index(drop=True)
            )

        # ----------------------------------------------------
        # AI quality metrics
        # ----------------------------------------------------

        quality = self.calculate_quality(
            timetable
        )

        quality["solver_status"] = (
            solver.StatusName(status)
        )

        quality["objective_value"] = (
            solver.ObjectiveValue()
        )

        quality["wall_time_seconds"] = (
            solver.WallTime()
        )

        return timetable, [], quality

    def calculate_quality(self, timetable):
        """Calculate timetable quality metrics."""

        if timetable is None or timetable.empty:
            return {
                "coverage": 0,
                "teacher_conflicts": 0,
                "room_conflicts": 0,
                "distribution_score": 0,
                "overall_score": 0,
            }

        expected = calculate_subject_hours(
            self.subjects
        )

        actual = len(timetable)

        coverage = (
            min(actual / expected, 1.0)
            if expected > 0
            else 0
        )

        teacher_conflicts = int(
            timetable.duplicated(
                subset=["Day", "Period", "Teacher"]
            ).sum()
        )

        room_conflicts = int(
            timetable.duplicated(
                subset=["Day", "Period", "Room"]
            ).sum()
        )

        # --------------------------------------------
        # Distribution score
        # --------------------------------------------

        distribution_scores = []

        for subject in self.subjects:

            subject_name = subject["name"]

            rows = timetable[
                timetable["Subject"] == subject_name
            ]

            if len(rows) <= 1:
                distribution_scores.append(1.0)
                continue

            unique_days = rows["Day"].nunique()

            required = safe_int(
                subject.get("periods_per_week"),
                1,
            )

            target = min(
                required,
                self.num_days,
            )

            distribution_scores.append(
                min(
                    unique_days / target,
                    1.0,
                )
            )

        distribution_score = (
            float(np.mean(distribution_scores))
            if distribution_scores
            else 0
        )

        conflict_penalty = (
            teacher_conflicts + room_conflicts
        )

        overall = (
            coverage * 60
            + distribution_score * 40
            - conflict_penalty * 10
        )

        overall = max(
            0,
            min(
                100,
                overall,
            ),
        )

        return {
            "coverage": round(
                coverage * 100,
                1,
            ),
            "teacher_conflicts": teacher_conflicts,
            "room_conflicts": room_conflicts,
            "distribution_score": round(
                distribution_score * 100,
                1,
            ),
            "overall_score": round(
                overall,
                1,
            ),
        }


# ============================================================
# SAMPLE DATA BUTTON
# ============================================================

def load_demo_data():
    subjects, teachers, rooms = create_demo_data()

    st.session_state.subjects = subjects
    st.session_state.teachers = teachers
    st.session_state.rooms = rooms

    st.session_state.timetable = pd.DataFrame()
    st.session_state.generated = False
    st.session_state.generation_stats = {}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🤖 AI Timetable Generator"
    )

    st.caption(
        f"Version {APP_VERSION}"
    )

    st.divider()

    st.markdown("### ⚙️ Generator Settings")

    selected_days = st.multiselect(
        "Working Days",
        options=DAYS,
        default=[
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
        ],
    )

    selected_periods = st.multiselect(
        "Periods",
        options=DEFAULT_PERIODS,
        default=[
            "09:00 - 10:00",
            "10:00 - 11:00",
            "11:00 - 12:00",
            "02:00 - 03:00",
            "03:00 - 04:00",
            "04:00 - 05:00",
        ],
    )

    avoid_break = st.checkbox(
        "Avoid lunch/break period",
        value=True,
    )

    max_daily_classes = st.slider(
        "Maximum classes per subject/day",
        min_value=1,
        max_value=4,
        value=1,
    )

    st.divider()

    if st.button(
        "🧪 Load Demo Data",
        use_container_width=True,
    ):
        load_demo_data()
        st.rerun()

    if st.button(
        "🗑️ Clear All Data",
        use_container_width=True,
    ):
        st.session_state.subjects = []
        st.session_state.teachers = []
        st.session_state.rooms = []
        st.session_state.timetable = pd.DataFrame()
        st.session_state.generated = False
        st.session_state.generation_stats = {}
        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">🤖 AI Timetable Generator</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-subtitle">
        Automatically generate conflict-free academic timetables
        using AI-assisted optimization and constraint programming.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DASHBOARD METRICS
# ============================================================

subjects_count = len(st.session_state.subjects)

teachers_count = len(
    set(
        subject.get("teacher")
        for subject in st.session_state.subjects
        if subject.get("teacher")
    )
)

rooms_count = len(
    set(
        subject.get("room")
        for subject in st.session_state.subjects
        if subject.get("room")
    )
)

required_periods = calculate_subject_hours(
    st.session_state.subjects
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">{subjects_count}</div>
            <div class="metric-label">Subjects</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">{teachers_count}</div>
            <div class="metric-label">Teachers</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">{rooms_count}</div>
            <div class="metric-label">Rooms</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-number">{required_periods}</div>
            <div class="metric-label">Required Periods</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.write("")


# ============================================================
# MAIN TABS
# ============================================================

tab_subjects, tab_generate, tab_results, tab_analytics = st.tabs(
    [
        "📚 Subjects",
        "⚡ Generate",
        "📅 Timetable",
        "📊 Analytics",
    ]
)


# ============================================================
# SUBJECT MANAGEMENT
# ============================================================

with tab_subjects:

    st.subheader("📚 Subject & Resource Configuration")

    st.caption(
        "Add each subject, its teacher, room and required weekly periods."
    )

    if st.session_state.subjects:

        display_subjects = pd.DataFrame(
            [
                {
                    "Subject": s.get("name"),
                    "Teacher": s.get("teacher"),
                    "Room": s.get("room"),
                    "Periods / Week": s.get(
                        "periods_per_week"
                    ),
                    "Preferred Day": s.get(
                        "preferred_day",
                        "Any",
                    ),
                    "Preferred Period": s.get(
                        "preferred_period",
                        "Any",
                    ),
                }
                for s in st.session_state.subjects
            ]
        )

        edited_df = st.data_editor(
            display_subjects,
            use_container_width=True,
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "Periods / Week": st.column_config.NumberColumn(
                    min_value=1,
                    max_value=20,
                    step=1,
                ),
            },
            key="subject_editor",
        )

        if st.button(
            "💾 Apply Subject Changes",
            type="secondary",
        ):

            updated_subjects = []

            for _, row in edited_df.iterrows():

                name = clean_name(
                    row.get("Subject")
                )

                if not name:
                    continue

                updated_subjects.append(
                    {
                        "id": generate_id("SUB"),
                        "name": name,
                        "teacher": clean_name(
                            row.get("Teacher")
                        ),
                        "room": clean_name(
                            row.get("Room")
                        ),
                        "periods_per_week": safe_int(
                            row.get(
                                "Periods / Week"
                            ),
                            1,
                        ),
                        "preferred_day": clean_name(
                            row.get(
                                "Preferred Day",
                                "Any",
                            )
                        ) or "Any",
                        "preferred_period": clean_name(
                            row.get(
                                "Preferred Period",
                                "Any",
                            )
                        ) or "Any",
                    }
                )

            st.session_state.subjects = updated_subjects

            st.session_state.teachers = sorted(
                list(
                    set(
                        s["teacher"]
                        for s in updated_subjects
                        if s["teacher"]
                    )
                )
            )

            st.session_state.rooms = sorted(
                list(
                    set(
                        s["room"]
                        for s in updated_subjects
                        if s["room"]
                    )
                )
            )

            st.session_state.generated = False

            st.success(
                "Subject configuration updated."
            )

    st.divider()

    st.markdown("### ➕ Add New Subject")

    with st.form("add_subject_form"):

        c1, c2 = st.columns(2)

        with c1:
            subject_name = st.text_input(
                "Subject Name",
                placeholder="e.g. Artificial Intelligence",
            )

            teacher_name = st.text_input(
                "Teacher",
                placeholder="e.g. Dr. Kumar",
            )

            room_name = st.text_input(
                "Room / Lab",
                placeholder="e.g. Lab 1",
            )

        with c2:

            periods_per_week = st.number_input(
                "Periods per Week",
                min_value=1,
                max_value=20,
                value=3,
            )

            preferred_day = st.selectbox(
                "Preferred Day",
                ["Any"] + DAYS,
            )

            preferred_period = st.selectbox(
                "Preferred Period",
                ["Any"] + DEFAULT_PERIODS,
            )

        submitted = st.form_submit_button(
            "➕ Add Subject",
            type="primary",
            use_container_width=True,
        )

        if submitted:

            if not clean_name(subject_name):
                st.error(
                    "Please enter a subject name."
                )

            elif not clean_name(teacher_name):
                st.error(
                    "Please enter a teacher name."
                )

            elif not clean_name(room_name):
                st.error(
                    "Please enter a room or lab."
                )

            else:

                new_subject = {
                    "id": generate_id("SUB"),
                    "name": clean_name(
                        subject_name
                    ),
                    "teacher": clean_name(
                        teacher_name
                    ),
                    "room": clean_name(
                        room_name
                    ),
                    "periods_per_week": int(
                        periods_per_week
                    ),
                    "preferred_day": preferred_day,
                    "preferred_period": preferred_period,
                }

                st.session_state.subjects.append(
                    new_subject
                )

                st.session_state.teachers = sorted(
                    list(
                        set(
                            s["teacher"]
                            for s in st.session_state.subjects
                        )
                    )
                )

                st.session_state.rooms = sorted(
                    list(
                        set(
                            s["room"]
                            for s in st.session_state.subjects
                        )
                    )
                )

                st.session_state.generated = False

                st.success(
                    f"Added '{subject_name}'."
                )

                st.rerun()


# ============================================================
# GENERATION TAB
# ============================================================

with tab_generate:

    st.subheader("⚡ Generate AI Timetable")

    st.markdown(
        """
        The generator uses **OR-Tools CP-SAT optimization** to create
        a timetable while respecting teacher and room conflicts.
        """
    )

    # --------------------------------------------
    # Configuration summary
    # --------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Working Days",
            len(selected_days),
        )

    with c2:
        st.metric(
            "Available Periods",
            len(selected_periods),
        )

    with c3:
        st.metric(
            "Available Slots",
            len(selected_days) * len(selected_periods),
        )

    st.divider()

    if not selected_days:
        st.warning(
            "Select at least one working day from the sidebar."
        )

    elif not selected_periods:
        st.warning(
            "Select at least one period from the sidebar."
        )

    elif not st.session_state.subjects:

        st.info(
            "Add subjects first or use 'Load Demo Data' "
            "from the sidebar."
        )

    else:

        available_slots = (
            len(selected_days)
            * len(selected_periods)
        )

        required_slots = calculate_subject_hours(
            st.session_state.subjects
        )

        if required_slots > available_slots:

            st.error(
                f"Not enough timetable slots. "
                f"You need {required_slots}, "
                f"but only {available_slots} are available."
            )

        else:

            st.markdown(
                f"""
                <div class="info-box">
                    <b>AI optimization ready.</b><br>
                    {len(st.session_state.subjects)} subjects,
                    {teachers_count} teachers,
                    {rooms_count} rooms,
                    {required_slots} weekly periods.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

            generate_clicked = st.button(
                "🚀 Generate Optimized Timetable",
                type="primary",
                use_container_width=True,
            )

            if generate_clicked:

                generator = AITimetableGenerator(
                    subjects=st.session_state.subjects,
                    days=selected_days,
                    periods=selected_periods,
                    avoid_break=avoid_break,
                    max_daily_classes=max_daily_classes,
                )

                with st.spinner(
                    "🤖 AI is optimizing your timetable..."
                ):

                    timetable, errors, stats = (
                        generator.generate()
                    )

                if errors:

                    for error in errors:
                        st.error(error)

                else:

                    st.session_state.timetable = timetable
                    st.session_state.generation_stats = stats
                    st.session_state.generated = True

                    st.success(
                        "🎉 Timetable generated successfully!"
                    )

                    st.balloons()

            # --------------------------------------------
            # Display generation statistics
            # --------------------------------------------

            if st.session_state.generation_stats:

                stats = st.session_state.generation_stats

                st.write("")

                m1, m2, m3, m4 = st.columns(4)

                with m1:
                    st.metric(
                        "AI Score",
                        f"{stats.get('overall_score', 0)}%",
                    )

                with m2:
                    st.metric(
                        "Coverage",
                        f"{stats.get('coverage', 0)}%",
                    )

                with m3:
                    st.metric(
                        "Teacher Conflicts",
                        stats.get(
                            "teacher_conflicts",
                            0,
                        ),
                    )

                with m4:
                    st.metric(
                        "Room Conflicts",
                        stats.get(
                            "room_conflicts",
                            0,
                        ),
                    )


# ============================================================
# TIMETABLE RESULT TAB
# ============================================================

with tab_results:

    st.subheader("📅 Generated Timetable")

    timetable = st.session_state.timetable

    if timetable is None or timetable.empty:

        st.info(
            "No timetable generated yet. "
            "Go to the Generate tab and click "
            "'Generate Optimized Timetable'."
        )

    else:

        # --------------------------------------------
        # Filters
        # --------------------------------------------

        c1, c2 = st.columns(2)

        with c1:
            selected_view_day = st.selectbox(
                "View Day",
                ["All"] + selected_days,
            )

        with c2:

            subject_options = [
                "All"
            ] + sorted(
                timetable["Subject"]
                .unique()
                .tolist()
            )

            selected_view_subject = st.selectbox(
                "View Subject",
                subject_options,
            )

        filtered = timetable.copy()

        if selected_view_day != "All":
            filtered = filtered[
                filtered["Day"]
                == selected_view_day
            ]

        if selected_view_subject != "All":
            filtered = filtered[
                filtered["Subject"]
                == selected_view_subject
            ]

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
        )

        # --------------------------------------------
        # Download
        # --------------------------------------------

        st.download_button(
            "⬇️ Download Timetable as CSV",
            data=timetable_to_csv(timetable),
            file_name=(
                "ai_timetable.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

        st.divider()

        # --------------------------------------------
        # Grid timetable
        # --------------------------------------------

        st.markdown("### 🗓️ Weekly Grid")

        if not timetable.empty:

            grid = timetable.pivot_table(
                index="Period",
                columns="Day",
                values="Subject",
                aggfunc=lambda x: " / ".join(
                    x.astype(str)
                ),
                fill_value="",
            )

            grid = grid.reindex(
                index=selected_periods,
                columns=selected_days,
                fill_value="",
            )

            st.dataframe(
                grid,
                use_container_width=True,
                height=500,
            )


# ============================================================
# ANALYTICS TAB
# ============================================================

with tab_analytics:

    st.subheader("📊 Timetable Analytics")

    timetable = st.session_state.timetable

    if timetable is None or timetable.empty:

        st.info(
            "Generate a timetable to view analytics."
        )

    else:

        stats = st.session_state.generation_stats

        # --------------------------------------------
        # Quality metrics
        # --------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Overall AI Score",
                f"{stats.get('overall_score', 0)}%",
            )

        with c2:
            st.metric(
                "Distribution Score",
                f"{stats.get('distribution_score', 0)}%",
            )

        with c3:
            st.metric(
                "Teacher Conflicts",
                stats.get(
                    "teacher_conflicts",
                    0,
                ),
            )

        with c4:
            st.metric(
                "Room Conflicts",
                stats.get(
                    "room_conflicts",
                    0,
                ),
            )

        st.divider()

        # --------------------------------------------
        # Subject distribution
        # --------------------------------------------

        st.markdown(
            "### 📚 Subject Distribution"
        )

        subject_counts = (
            timetable[
                "Subject"
            ]
            .value_counts()
            .rename("Periods")
            .reset_index()
        )

        subject_counts.columns = [
            "Subject",
            "Periods",
        ]

        st.bar_chart(
            subject_counts.set_index(
                "Subject"
            )
        )

        # --------------------------------------------
        # Teacher workload
        # --------------------------------------------

        st.markdown(
            "### 👨‍🏫 Teacher Workload"
        )

        teacher_counts = (
            timetable[
                "Teacher"
            ]
            .value_counts()
            .rename("Classes")
            .reset_index()
        )

        teacher_counts.columns = [
            "Teacher",
            "Classes",
        ]

        st.bar_chart(
            teacher_counts.set_index(
                "Teacher"
            )
        )

        # --------------------------------------------
        # Room utilization
        # --------------------------------------------

        st.markdown(
            "### 🏫 Room Utilization"
        )

        room_counts = (
            timetable[
                "Room"
            ]
            .value_counts()
            .rename("Classes")
            .reset_index()
        )

        room_counts.columns = [
            "Room",
            "Classes",
        ]

        st.bar_chart(
            room_counts.set_index(
                "Room"
            )
        )

        # --------------------------------------------
        # Daily workload
        # --------------------------------------------

        st.markdown(
            "### 📆 Daily Workload"
        )

        day_counts = (
            timetable[
                "Day"
            ]
            .value_counts()
            .reindex(
                selected_days,
                fill_value=0,
            )
        )

        st.line_chart(day_counts)


# ============================================================
# AI INSIGHTS
# ============================================================

if (
    st.session_state.generated
    and not st.session_state.timetable.empty
):

    st.divider()

    st.subheader("🧠 AI Insights")

    timetable = st.session_state.timetable
    stats = st.session_state.generation_stats

    insights = []

    if stats.get("teacher_conflicts", 0) == 0:
        insights.append(
            "✅ No teacher scheduling conflicts detected."
        )
    else:
        insights.append(
            "⚠️ Teacher conflicts require attention."
        )

    if stats.get("room_conflicts", 0) == 0:
        insights.append(
            "✅ No room conflicts detected."
        )
    else:
        insights.append(
            "⚠️ Room conflicts require attention."
        )

    if stats.get("distribution_score", 0) >= 80:
        insights.append(
            "✅ Subjects are well distributed across the week."
        )
    elif stats.get("distribution_score", 0) >= 60:
        insights.append(
            "ℹ️ Subject distribution is acceptable but can be improved."
        )
    else:
        insights.append(
            "⚠️ Some subjects are concentrated on fewer days."
        )

    # Teacher with highest workload.
    if not timetable.empty:

        teacher_load = (
            timetable[
                "Teacher"
            ]
            .value_counts()
        )

        if not teacher_load.empty:

            busiest_teacher = (
                teacher_load.idxmax()
            )

            busiest_count = (
                teacher_load.max()
            )

            insights.append(
                f"👨‍🏫 {busiest_teacher} has "
                f"{busiest_count} scheduled classes."
            )

    # Subject with highest frequency.
    if not timetable.empty:

        subject_load = (
            timetable[
                "Subject"
            ]
            .value_counts()
        )

        if not subject_load.empty:

            highest_subject = (
                subject_load.idxmax()
            )

            highest_count = (
                subject_load.max()
            )

            insights.append(
                f"📚 {highest_subject} has "
                f"{highest_count} periods scheduled."
            )

    for insight in insights:
        st.write(insight)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🤖 AI Timetable Generator • "
    "Built with Streamlit + OR-Tools + Pandas + NumPy + Scikit-learn"
)

st.caption(
    "The timetable is generated using constraint programming "
    "and optimization rather than random assignment."
)