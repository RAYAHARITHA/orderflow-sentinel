"""
OrderFlow Sentinel - Enterprise Web Application
Predict. Explain. Recover Safely.
Main principle: Prove Before You Retry.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import sys

# Ensure root modules are discoverable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_generator import (
    get_all_orders,
    save_all_orders,
    get_order_by_id,
    reset_to_benchmark,
    STAGE_THRESHOLDS
)
from modules.sla_engine import SLAPredictionEngine
from modules.root_cause_engine import RootCauseEngine
from modules.recovery_engine import SafeRecoveryEngine
from modules.audit_logger import AuditLogger
from modules.metrics_engine import MetricsEngine
from modules.workflow_visualizer import render_workflow_html, WORKFLOW_STEPS, SCENARIO_PATHS

# =============================================================================
# STREAMLIT CONFIGURATION & ENTERPRISE STYLING
# =============================================================================
st.set_page_config(
    page_title="OrderFlow Sentinel | Predict. Explain. Recover Safely.",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS (Navy Blue, Teal, Slate, Crisp Cards)
CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Backgrounds & Container */
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }

    /* Custom Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #0ea5e9;
    }
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 12px;
        color: #38bdf8;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Principle Banner */
    .principle-banner {
        background: linear-gradient(90deg, #0d9488 0%, #0369a1 100%);
        border-radius: 10px;
        padding: 14px 22px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: #ffffff;
        box-shadow: 0 4px 20px rgba(13, 148, 136, 0.25);
    }

    /* Status Badges */
    .badge-completed {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-at-risk {
        background: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-failed {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-review {
        background: rgba(168, 85, 247, 0.2);
        color: #c084fc;
        border: 1px solid #a855f7;
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 12px;
    }

    /* Event Timeline Cards */
    .timeline-card {
        background: #1e293b;
        border-left: 4px solid #0ea5e9;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .timeline-card.failure {
        border-left-color: #ef4444;
    }
    .timeline-card.warning {
        border-left-color: #f59e0b;
    }
    .timeline-card.success {
        border-left-color: #10b981;
    }

    /* Code & Log Highlighting */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        background: #0f172a !important;
        border: 1px solid #1e293b;
        border-radius: 6px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #1e293b;
        border-radius: 6px 6px 0 0;
        padding: 8px 18px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: #0284c7 !important;
        color: #ffffff !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Core Engines
sla_engine = SLAPredictionEngine()
root_cause_engine = RootCauseEngine()
recovery_engine = SafeRecoveryEngine()
audit_logger = AuditLogger()
metrics_engine = MetricsEngine()

# Session State Initialization
if "selected_order_id" not in st.session_state:
    st.session_state.selected_order_id = "ORD-1001"
if "workflow_step_index" not in st.session_state:
    st.session_state.workflow_step_index = 0
if "learning_step_index" not in st.session_state:
    st.session_state.learning_step_index = 0

# =============================================================================
# SIDEBAR NAVIGATION & BENCHMARK QUICK LAUNCHER
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
        <span style="font-size: 28px;">🛡️</span>
        <div>
            <h2 style="margin: 0; font-size: 20px; font-weight: 800; color: #38bdf8;">OrderFlow Sentinel</h2>
            <p style="margin: 0; font-size: 11px; color: #94a3b8; font-weight: 500;">Predict. Explain. Recover Safely.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(13, 148, 136, 0.15); border: 1px solid #0d9488; border-radius: 6px; padding: 8px 12px; margin-bottom: 18px;">
        <span style="font-size: 11px; font-weight: 700; color: #14b8a6; text-transform: uppercase;">Core Principle</span>
        <div style="font-size: 12px; font-weight: 600; color: #f1f5f9; margin-top: 2px;">
            Prove Before You Retry
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "1. 📊 Dashboard",
            "2. 🔄 Order Workflow Explorer",
            "3. ⏱️ SLA Prediction",
            "4. 🔍 Root Cause Analysis",
            "5. 🛡️ Safe Recovery Center",
            "6. 📜 Audit History & Evaluation",
            "7. 🎓 How the System Works"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### 🎯 Benchmark Scenarios")
    st.caption("Reproducible cases that work across application restarts:")

    col_s1, col_s2, col_s3 = st.columns(3)
    if col_s1.button("Case A\n504 Timeout", help="ORD-1001: API Timeout with Idempotency - Safe to Retry"):
        st.session_state.selected_order_id = "ORD-1001"
        st.session_state.workflow_step_index = 12
        st.rerun()

    if col_s2.button("Case B\nDelayed", help="ORD-1002: Stage Threshold Exceeded - At Risk, Not Breached"):
        st.session_state.selected_order_id = "ORD-1002"
        st.session_state.workflow_step_index = 7
        st.rerun()

    if col_s3.button("Case C\nUncertain", help="ORD-1003: Timeout + Downstream Stream Event - Human Review"):
        st.session_state.selected_order_id = "ORD-1003"
        st.session_state.workflow_step_index = 10
        st.rerun()

    st.markdown("---")
    if st.button("🔄 Reset Baseline Data", help="Restores the benchmark 35+ orders and test cases"):
        reset_to_benchmark()
        audit_logger.reset_audit_log()
        st.success("Baseline dataset restored!")
        st.rerun()

    st.caption("Virtusa Hackathon Prototype • Python & Streamlit")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_status_badge(status):
    if status == "COMPLETED":
        return '<span class="badge-completed">Completed</span>'
    elif status == "AT_RISK":
        return '<span class="badge-at-risk">At Risk</span>'
    elif status == "FAILED":
        return '<span class="badge-failed">Failed</span>'
    elif status == "PENDING_REVIEW":
        return '<span class="badge-review">Human Review</span>'
    else:
        return f'<span style="background: rgba(14, 165, 233, 0.2); color: #38bdf8; border: 1px solid #0284c7; padding: 3px 10px; border-radius: 20px; font-weight: 600; font-size: 12px;">{status}</span>'


orders = get_all_orders()

# =============================================================================
# PAGE 1: EXECUTIVE DASHBOARD
# =============================================================================
if "1. 📊 Dashboard" in page:
    st.markdown("""
    <div class="principle-banner">
        <div>
            <h2 style="margin: 0; font-size: 22px; font-weight: 800;">OrderFlow Sentinel Command Center</h2>
            <p style="margin: 4px 0 0 0; font-size: 13px; opacity: 0.9;">
                Real-time multi-service order monitoring, transparent SLA risk prediction, and safe idempotent recovery.
            </p>
        </div>
        <div style="text-align: right; font-family: monospace; font-size: 13px; background: rgba(0,0,0,0.2); padding: 6px 14px; border-radius: 6px;">
            POLICY: PROVE BEFORE YOU RETRY
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Calculate live metrics from synthetic dataset
    total_orders = len(orders)
    completed_orders = sum(1 for o in orders if o.get("current_status") == "COMPLETED")
    failed_orders = sum(1 for o in orders if o.get("current_status") == "FAILED")
    at_risk_orders = sum(1 for o in orders if sla_engine.evaluate_order(o)["classification"] == "At Risk")
    human_review_cases = sum(1 for o in orders if o.get("current_status") == "PENDING_REVIEW")
    breached_orders = sum(1 for o in orders if sla_engine.evaluate_order(o)["classification"] == "Breached")

    # Metric Cards Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Monitored</div>
            <div class="metric-value">{total_orders}</div>
            <div class="metric-sub">Active Enterprise Pipeline</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Completed</div>
            <div class="metric-value" style="color: #10b981;">{completed_orders}</div>
            <div class="metric-sub">{(completed_orders/total_orders*100):.1f}% Fulfillment Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted At Risk</div>
            <div class="metric-value" style="color: #f59e0b;">{at_risk_orders}</div>
            <div class="metric-sub">Stage Threshold Exceeded</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Service Failures</div>
            <div class="metric-value" style="color: #ef4444;">{failed_orders}</div>
            <div class="metric-sub">{breached_orders} Overdue Breaches</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Human Review Needed</div>
            <div class="metric-value" style="color: #c084fc;">{human_review_cases}</div>
            <div class="metric-sub">Uncertain Telemetry</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visual Charts
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### 📈 Order Health & Status Distribution")
        status_counts = {}
        for o in orders:
            stt = o.get("current_status", "UNKNOWN")
            status_counts[stt] = status_counts.get(stt, 0) + 1

        df_status = pd.DataFrame(list(status_counts.items()), columns=["Status", "Count"])
        fig_donut = px.pie(
            df_status,
            names="Status",
            values="Count",
            hole=0.55,
            color="Status",
            color_discrete_map={
                "COMPLETED": "#10b981",
                "IN_PROGRESS": "#0ea5e9",
                "AT_RISK": "#f59e0b",
                "FAILED": "#ef4444",
                "PENDING_REVIEW": "#a855f7"
            }
        )
        fig_donut.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        st.markdown("### 🏢 Pipeline Stage Distribution")
        stage_counts = {}
        for o in orders:
            stg = o.get("current_stage", "Unknown")
            stage_counts[stg] = stage_counts.get(stg, 0) + 1

        df_stage = pd.DataFrame(list(stage_counts.items()), columns=["Stage", "Orders"])
        fig_stage = px.bar(
            df_stage,
            x="Orders",
            y="Stage",
            orientation="h",
            color="Orders",
            color_continuous_scale="Tealgrn"
        )
        fig_stage.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9"),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_stage, use_container_width=True)

    st.markdown("---")
    st.markdown("### ⚡ Benchmark Demonstration Quick Switcher")
    st.markdown("Explore the 3 fixed cases built for your hackathon presentation:")

    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("""
        <div style="background: #1e293b; border-radius: 10px; padding: 16px; border: 1px solid #334155;">
            <h4 style="color: #38bdf8; margin: 0 0 6px 0;">Case A: ORD-1001</h4>
            <p style="font-size: 13px; color: #cbd5e1; margin-bottom: 12px;">
                <strong>API Timeout (HTTP 504)</strong> in partner Fulfilment API. Idempotency active, retry count 0.
                Sentinel evaluates safeguards and authorizes 1 controlled retry.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Inspect ORD-1001", key="btn_dash_1001"):
            st.session_state.selected_order_id = "ORD-1001"
            st.session_state.workflow_step_index = 12
            st.rerun()

    with b2:
        st.markdown("""
        <div style="background: #1e293b; border-radius: 10px; padding: 16px; border: 1px solid #334155;">
            <h4 style="color: #f59e0b; margin: 0 0 6px 0;">Case B: ORD-1002</h4>
            <p style="font-size: 13px; color: #cbd5e1; margin-bottom: 12px;">
                <strong>Delayed Order in Inventory Check</strong>. Stage took 7.5m (threshold 5m).
                Overall SLA deadline has 22.5m remaining. Flagged At Risk, not Breached.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Inspect ORD-1002", key="btn_dash_1002"):
            st.session_state.selected_order_id = "ORD-1002"
            st.session_state.workflow_step_index = 7
            st.rerun()

    with b3:
        st.markdown("""
        <div style="background: #1e293b; border-radius: 10px; padding: 16px; border: 1px solid #334155;">
            <h4 style="color: #c084fc; margin: 0 0 6px 0;">Case C: ORD-1003</h4>
            <p style="font-size: 13px; color: #cbd5e1; margin-bottom: 12px;">
                <strong>Uncertain Outcome</strong>. Client received timeout, but downstream warehouse event stream showed partial reservation.
                Retry blocked; Human Review mandated.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Inspect ORD-1003", key="btn_dash_1003"):
            st.session_state.selected_order_id = "ORD-1003"
            st.session_state.workflow_step_index = 10
            st.rerun()

# =============================================================================
# PAGE 2: ORDER WORKFLOW EXPLORER
# =============================================================================
elif "2. 🔄 Order Workflow Explorer" in page:
    st.markdown("## 🔄 Order Workflow Explorer & Interactive Pipeline")
    st.markdown("Inspect simulated enterprise orders, execute the interactive Sentinel workflow, and understand each event in plain English.")

    # Order Selector
    order_ids = [o["order_id"] for o in orders]
    selected_idx = order_ids.index(st.session_state.selected_order_id) if st.session_state.selected_order_id in order_ids else 0
    
    col_sel, col_ctrl1, col_ctrl2 = st.columns([2, 1, 1])
    with col_sel:
        selected_id = st.selectbox("Select Monitored Order", order_ids, index=selected_idx)
        st.session_state.selected_order_id = selected_id
    with col_ctrl1:
        st.write("")
        st.write("")
        if st.button("▶ Run Workflow", help="Simulates advancing through Sentinel workflow decision gates"):
            st.session_state.workflow_step_index = (st.session_state.workflow_step_index + 1) % len(WORKFLOW_STEPS)
            st.rerun()
    with col_ctrl2:
        st.write("")
        st.write("")
        if st.button("⏪ Reset Scenario"):
            st.session_state.workflow_step_index = 0
            st.rerun()

    current_order = get_order_by_id(selected_id)

    # Workflow Visualizer Diagram
    st.markdown("### 🗺️ Sentinel 14-Step Decision Workflow")
    workflow_html = render_workflow_html(selected_id, st.session_state.workflow_step_index)
    st.components.v1.html(workflow_html, height=520, scrolling=True)

    # Order Details Card
    st.markdown("---")
    st.markdown(f"### 📋 Order Telemetry: {selected_id}")

    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
    with col_meta1:
        st.markdown(f"**Customer:** {current_order.get('customer_name')}")
        st.markdown(f"**Amount:** ${current_order.get('order_amount'):,.2f} {current_order.get('currency')}")
    with col_meta2:
        st.markdown(f"**Current Stage:** `{current_order.get('current_stage')}`")
        badge = get_status_badge(current_order.get('current_status'))
        st.markdown(f"**Status:** {badge}", unsafe_allow_html=True)
    with col_meta3:
        st.markdown(f"**Time at Stage:** {current_order.get('time_spent_at_current_stage')}")
        st.markdown(f"**Retry Count:** {current_order.get('retry_count')} / {current_order.get('max_retries')}")
    with col_meta4:
        idemp = current_order.get('idempotency_key') or "None (Disabled)"
        st.markdown(f"**Idempotency Token:** `{idemp}`")
        st.markdown(f"**Expected Next Step:** {current_order.get('expected_next_step')}")

    # Interactive Event Timeline & Plain-English Explainer
    st.markdown("### ⏱️ Order Event History & Plain English Explanations")
    st.caption("Click any event to reveal what it means in plain English, its system significance, and raw log payloads:")

    events = current_order.get("events", [])
    if not events:
        st.info("No events logged for this order.")
    else:
        for idx, evt in enumerate(events):
            status_style = "failure" if evt.get("status") == "FAILURE" else ("warning" if evt.get("status") in ["WARNING", "UNCERTAIN"] else "success")
            badge_color = "#ef4444" if status_style == "failure" else ("#f59e0b" if status_style == "warning" else "#10b981")

            with st.expander(f"{idx+1}. [{evt.get('timestamp')[11:19]}] {evt.get('stage')} — {evt.get('event_type')} ({evt.get('status')})", expanded=(idx == len(events)-1)):
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.6); padding: 14px; border-radius: 8px; border-left: 4px solid {badge_color}; margin-bottom: 10px;">
                    <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Message:</div>
                    <div style="font-size: 14px; color: #f8fafc; font-weight: 600; margin-bottom: 8px;">{evt.get('message')}</div>
                    <div style="font-size: 11px; font-weight: 700; color: #38bdf8; text-transform: uppercase;">Plain English Explanation:</div>
                    <div style="font-size: 13px; color: #e2e8f0; line-height: 1.4;">{evt.get('plain_english_explanation')}</div>
                </div>
                """, unsafe_allow_html=True)

    # API Request & Response Logs
    st.markdown("### 📡 Correlated API Request / Response Logs")
    logs = current_order.get("api_logs", [])
    if logs:
        log_df = pd.DataFrame([{
            "Log ID": l["log_id"],
            "Timestamp": l["timestamp"][11:19],
            "Service": l["service_name"],
            "Method": l["http_method"],
            "Endpoint": l["endpoint"],
            "HTTP Status": l["http_status"],
            "Latency (ms)": l["latency_ms"],
            "Error Message": l.get("error_message") or "None"
        } for l in logs])
        st.dataframe(log_df, use_container_width=True)

        selected_log_id = st.selectbox("Inspect Full API Request & Response Payload", [l["log_id"] for l in logs])
        chosen_log = next((l for l in logs if l["log_id"] == selected_log_id), None)
        if chosen_log:
            c_req, c_res = st.columns(2)
            with c_req:
                st.caption(f"Request Payload ({chosen_log['http_method']} {chosen_log['endpoint']}):")
                st.json(chosen_log["request_payload"])
            with c_res:
                st.caption(f"Response Payload (HTTP {chosen_log['http_status']}):")
                st.json(chosen_log["response_payload"])
    else:
        st.info("No API logs found.")

    # Table of all 35+ orders
    st.markdown("---")
    st.markdown("### 🗃️ All Monitored Orders (35+ Workload)")
    table_data = []
    for o in orders:
        sla_res = sla_engine.evaluate_order(o)
        table_data.append({
            "Order ID": o["order_id"],
            "Customer": o["customer_name"],
            "Amount ($)": o["order_amount"],
            "Current Stage": o["current_stage"],
            "Status": o["current_status"],
            "SLA Prediction": sla_res["classification"],
            "Stage Time": o["time_spent_at_current_stage"],
            "Retries": f"{o['retry_count']}/{o['max_retries']}"
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, height=250)

# =============================================================================
# PAGE 3: SLA PREDICTION
# =============================================================================
elif "3. ⏱️ SLA Prediction" in page:
    st.markdown("## ⏱️ SLA Risk Prediction & Threshold Engine")
    st.markdown("Transparent, explainable evaluation of order delivery windows. Clearly distinguishes **predicted risk** from **actual breaches**.")

    order_ids = [o["order_id"] for o in orders]
    selected_idx = order_ids.index(st.session_state.selected_order_id) if st.session_state.selected_order_id in order_ids else 0
    selected_id = st.selectbox("Select Order to Evaluate SLA Risk", order_ids, index=selected_idx)
    st.session_state.selected_order_id = selected_id

    current_order = get_order_by_id(selected_id)
    sla_result = sla_engine.evaluate_order(current_order)

    # Classification Banner
    classification = sla_result["classification"]
    if classification == "Breached":
        banner_color = "#ef4444"
        badge_text = "ACTUAL SLA BREACH"
    elif classification == "At Risk":
        banner_color = "#f59e0b"
        badge_text = "PREDICTED RISK (AT RISK)"
    else:
        banner_color = "#10b981"
        badge_text = "NORMAL PROCESSING"

    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.8); border: 2px solid {banner_color}; border-radius: 12px; padding: 20px; margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 13px; font-weight: 700; color: {banner_color}; text-transform: uppercase;">{badge_text}</span>
            <span style="font-family: monospace; font-size: 13px; color: #94a3b8;">Risk Score: {sla_result['risk_score']*100:.0f}%</span>
        </div>
        <h3 style="margin: 0 0 10px 0; color: #f8fafc; font-size: 20px;">{sla_result['primary_reason']}</h3>
        <p style="margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;">{sla_result['detailed_explanation']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Telemetry Gauges Row
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("### ⏳ Stage Duration vs Warning Limit")
        fig_stage_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sla_result["stage_elapsed_min"],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{current_order.get('current_stage')} (Min)", 'font': {'color': '#f1f5f9'}},
            gauge={
                'axis': {'range': [0, 15], 'tickcolor': '#94a3b8'},
                'bar': {'color': "#38bdf8"},
                'steps': [
                    {'range': [0, sla_result["stage_threshold_min"]], 'color': "rgba(16, 185, 129, 0.2)"},
                    {'range': [sla_result["stage_threshold_min"], 15], 'color': "rgba(245, 158, 11, 0.3)"}
                ],
                'threshold': {
                    'line': {'color': "#f59e0b", 'width': 4},
                    'thickness': 0.75,
                    'value': sla_result["stage_threshold_min"]
                }
            }
        ))
        fig_stage_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9"))
        st.plotly_chart(fig_stage_gauge, use_container_width=True)

    with g2:
        st.markdown("### ⏰ Total SLA Deadline Remaining")
        rem = sla_result["time_remaining_min"]
        rem_color = "#10b981" if rem > 10 else ("#f59e0b" if rem > 0 else "#ef4444")
        fig_sla_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=max(0, rem),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Time to Breach (Min)", 'font': {'color': '#f1f5f9'}},
            gauge={
                'axis': {'range': [0, 35], 'tickcolor': '#94a3b8'},
                'bar': {'color': rem_color},
                'steps': [
                    {'range': [0, 10], 'color': "rgba(239, 68, 68, 0.2)"},
                    {'range': [10, 20], 'color': "rgba(245, 158, 11, 0.2)"},
                    {'range': [20, 35], 'color': "rgba(16, 185, 129, 0.2)"}
                ]
            }
        ))
        fig_sla_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9"))
        st.plotly_chart(fig_sla_gauge, use_container_width=True)

    with g3:
        st.markdown("### ⚙️ Stage Threshold Configuration")
        st.markdown("""
        | Stage | Warning Limit | Hard Limit |
        |---|---|---|
        | Order Created | 1.5 min | 3.0 min |
        | Inventory Check | 5.0 min | 10.0 min |
        | Fulfilment Request | 6.0 min | 12.0 min |
        | Shipment Confirmation | 7.0 min | 15.0 min |
        """)
        st.caption("Configured thresholds trigger early warning flags before customer delivery breaches occur.")

    # Transparent Decision Tree ML Companion
    st.markdown("---")
    st.markdown("### 🤖 Explainable Decision Tree Companion (Feature Importance)")
    st.caption("Our system uses transparent rules first. A shallow scikit-learn DecisionTree was fitted across the synthetic dataset to show which factors drive SLA risk:")

    sla_engine.train_explainable_ml_model(orders)
    importances = sla_engine.get_feature_importances()
    df_imp = pd.DataFrame(list(importances.items()), columns=["Signal Feature", "Relative Importance"]).sort_values("Relative Importance", ascending=True)

    fig_imp = px.bar(
        df_imp,
        x="Relative Importance",
        y="Signal Feature",
        orientation="h",
        color="Relative Importance",
        color_continuous_scale="Teal"
    )
    fig_imp.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f1f5f9"),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# =============================================================================
# PAGE 4: ROOT CAUSE ANALYSIS
# =============================================================================
elif "4. 🔍 Root Cause Analysis" in page:
    st.markdown("## 🔍 Evidence-Based Root Cause Diagnostic Engine")
    st.markdown("Correlates HTTP response codes, latency spikes, and downstream event streams to reach a deterministic diagnosis. **No hallucinated text.**")

    order_ids = [o["order_id"] for o in orders]
    selected_idx = order_ids.index(st.session_state.selected_order_id) if st.session_state.selected_order_id in order_ids else 0
    selected_id = st.selectbox("Select Order to Diagnose", order_ids, index=selected_idx)
    st.session_state.selected_order_id = selected_id

    current_order = get_order_by_id(selected_id)
    diagnosis = root_cause_engine.diagnose_order(current_order)

    # Diagnosis Overview Banner
    strength = diagnosis["evidence_strength"]
    strength_badge_color = "#10b981" if strength == "Conclusive" else "#f59e0b"

    st.markdown(f"""
    <div style="background: #0f172a; border: 2px solid #0ea5e9; border-radius: 12px; padding: 22px; margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="font-size: 13px; font-weight: 700; color: #38bdf8; text-transform: uppercase;">Suspected Root Cause</span>
            <span style="background: rgba(0,0,0,0.3); border: 1px solid {strength_badge_color}; color: {strength_badge_color}; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 700;">
                Evidence: {strength}
            </span>
        </div>
        <h2 style="margin: 0 0 8px 0; color: #f8fafc; font-size: 24px; font-weight: 800;">
            {diagnosis['suspected_cause']}
        </h2>
        <div style="font-size: 14px; color: #94a3b8; margin-bottom: 14px;">
            Primary Affected Service: <strong style="color: #f1f5f9;">{diagnosis['affected_service']}</strong>
        </div>
        <div style="background: rgba(30, 41, 59, 0.6); padding: 14px; border-radius: 8px; border-left: 4px solid #38bdf8; margin-bottom: 14px;">
            <div style="font-size: 11px; font-weight: 700; color: #38bdf8; text-transform: uppercase; margin-bottom: 4px;">Why These Logs Support The Diagnosis:</div>
            <div style="color: #e2e8f0; font-size: 13px; line-height: 1.5;">{diagnosis['explanation_of_evidence']}</div>
        </div>
        <div style="background: rgba(13, 148, 136, 0.15); border: 1px solid #0d9488; padding: 12px; border-radius: 8px;">
            <strong style="color: #14b8a6; font-size: 12px; text-transform: uppercase;">Recommended Operational Action:</strong>
            <div style="color: #f8fafc; font-size: 13px; margin-top: 2px;">{diagnosis['recommended_action']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Latency Waterfall & Supporting Logs
    col_l1, col_l2 = st.columns([1, 1])

    with col_l1:
        st.markdown("### 📊 Microservice Latency Waterfall")
        logs = current_order.get("api_logs", [])
        if logs:
            df_lat = pd.DataFrame([{
                "Service": l["service_name"],
                "Latency (ms)": l["latency_ms"],
                "Status": f"HTTP {l['http_status']}"
            } for l in logs])

            fig_lat = px.bar(
                df_lat,
                x="Service",
                y="Latency (ms)",
                color="Latency (ms)",
                color_continuous_scale="Reds" if any(l["latency_ms"] >= 4000 for l in logs) else "Blues",
                text="Status"
            )
            fig_lat.add_hline(y=5000, line_dash="dash", line_color="#ef4444", annotation_text="5000ms Gateway Timeout Limit")
            fig_lat.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f1f5f9"),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_lat, use_container_width=True)
        else:
            st.info("No latency telemetry recorded.")

    with col_l2:
        st.markdown("### 📑 Supporting Correlated Logs")
        supp_ids = diagnosis["supporting_log_entries"]
        st.markdown(f"**Identified Key Log Entries:** `{', '.join(supp_ids) if supp_ids else 'None'}`")

        matching_logs = [l for l in logs if l["log_id"] in supp_ids]
        if matching_logs:
            for l in matching_logs:
                with st.container():
                    st.markdown(f"""
                    <div style="background: #1e293b; padding: 10px 14px; border-radius: 6px; margin-bottom: 8px; border: 1px solid #334155;">
                        <div style="display: flex; justify-content: space-between; font-size: 11px; color: #94a3b8;">
                            <span>{l['log_id']} • {l['timestamp'][11:19]}</span>
                            <span style="font-weight: 700; color: {'#ef4444' if l['http_status']>=400 else '#10b981'};">HTTP {l['http_status']} ({l['latency_ms']}ms)</span>
                        </div>
                        <div style="font-family: monospace; font-size: 12px; color: #f8fafc; margin-top: 4px;">{l['service_name']} ➔ {l['endpoint']}</div>
                        {f"<div style='color: #ef4444; font-size: 11px; margin-top: 4px;'>{l['error_message']}</div>" if l.get('error_message') else ""}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.caption("No specific anomalous logs flagged.")

# =============================================================================
# PAGE 5: SAFE RECOVERY CENTER (THE MAIN FEATURE: PROVE BEFORE YOU RETRY)
# =============================================================================
elif "5. 🛡️ Safe Recovery Center" in page:
    st.markdown("## 🛡️ Safe Recovery Center")
    st.markdown("""
    <div class="principle-banner">
        <div>
            <h3 style="margin: 0; font-size: 20px; font-weight: 800;">PROVE BEFORE YOU RETRY</h3>
            <p style="margin: 4px 0 0 0; font-size: 13px;">
                A timeout does not prove an operation failed. It might have succeeded downstream while the response was lost in transit.
            </p>
        </div>
        <div style="background: rgba(0,0,0,0.25); padding: 6px 14px; border-radius: 6px; font-family: monospace; font-size: 13px;">
            STRICT IDEMPOTENCY ENFORCED
        </div>
    </div>
    """, unsafe_allow_html=True)

    order_ids = [o["order_id"] for o in orders]
    selected_idx = order_ids.index(st.session_state.selected_order_id) if st.session_state.selected_order_id in order_ids else 0
    selected_id = st.selectbox("Select Order to Evaluate for Recovery", order_ids, index=selected_idx)
    st.session_state.selected_order_id = selected_id

    current_order = get_order_by_id(selected_id)
    safety_eval = recovery_engine.evaluate_recovery_safety(current_order)

    # Verdict Header
    badge_color = "#10b981" if safety_eval["decision_code"] == "PERMIT_CONTROLLED_RETRY" else ("#f59e0b" if "UNCERTAIN" in safety_eval["decision_code"] else "#ef4444")
    if safety_eval["decision_code"] == "BLOCK_CONFIRMED_SUCCESS":
        badge_color = "#10b981"

    st.markdown(f"""
    <div style="background: #0f172a; border: 2px solid {badge_color}; border-radius: 12px; padding: 22px; margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="font-size: 13px; font-weight: 700; color: {badge_color}; text-transform: uppercase;">
                SENTINEL VERDICT: {safety_eval['decision_badge']}
            </span>
            <span style="font-family: monospace; font-size: 12px; color: #94a3b8;">Protocol Rule: Section 8 Safety Gates</span>
        </div>
        <h3 style="margin: 0 0 8px 0; color: #f8fafc; font-size: 22px; font-weight: 800;">
            {safety_eval['summary']}
        </h3>
        <p style="margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.5;">
            {safety_eval['explanation']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 6 Safety Checklist Cards
    st.markdown("### 📋 6-Gate Pre-Recovery Safety Verification")
    st.caption("All 6 conditions must pass before an automatic retry is authorized. Retries are strictly blocked on ambiguity:")

    checklist = safety_eval.get("checklist", {})
    chk_cols = st.columns(3)
    
    check_items = list(checklist.items())
    for i, (k, check) in enumerate(check_items):
        col = chk_cols[i % 3]
        passed = check["passed"]
        icon = "✅" if passed else "❌"
        border = "#10b981" if passed else "#ef4444"
        bg = "rgba(16, 185, 129, 0.08)" if passed else "rgba(239, 68, 68, 0.08)"

        with col:
            st.markdown(f"""
            <div style="background: {bg}; border: 1px solid {border}; border-radius: 8px; padding: 14px; margin-bottom: 14px; min-height: 130px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                    <span style="font-size: 16px;">{icon}</span>
                    <strong style="color: #f8fafc; font-size: 13px;">{check['name']}</strong>
                </div>
                <div style="font-size: 12px; color: {'#94a3b8' if passed else '#fca5a5'}; line-height: 1.4;">
                    {check['detail']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Action Buttons Row
    st.markdown("### 🕹️ Operational Recovery Actions")
    st.caption("Buttons execute genuine business logic through Sentinel engines. Safety checks can NEVER be bypassed:")

    act1, act2, act3, act4, act5 = st.columns(5)

    with act1:
        if st.button("🔍 Analyze Evidence", use_container_width=True):
            st.info(f"Correlating logs for {selected_id}... Verified status: {current_order.get('current_status')}")

    with act2:
        if st.button("🛡️ Check Safety", use_container_width=True):
            if safety_eval["retry_authorized"]:
                st.success("Safe Recovery Protocol: All safeguards satisfied!")
            else:
                st.warning(f"Blocked: {safety_eval['summary']}")

    with act3:
        # Retry Button
        retry_disabled = not safety_eval["retry_authorized"]
        if st.button("🚀 Execute Retry", disabled=retry_disabled, use_container_width=True, help="Executes controlled idempotent retry"):
            success, msg, _ = recovery_engine.execute_controlled_retry(selected_id, logger=audit_logger)
            if success:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with act4:
        if st.button("👤 Escalate to Human", use_container_width=True, help="Sends order to Tier-2 Operator queue"):
            success, msg = recovery_engine.escalate_to_human_review(
                selected_id,
                operator_notes="Escalated due to uncertain downstream telemetry.",
                logger=audit_logger
            )
            if success:
                st.warning(msg)
                st.rerun()

    with act5:
        if st.button("✅ Resolve Case", use_container_width=True, help="Operator manual resolution"):
            success, msg = recovery_engine.resolve_case_manually(
                selected_id,
                resolution_choice="CONFIRM_PARTNER_EXECUTED",
                notes="Operator verified physical reservation bay directly via warehouse dispatch portal.",
                logger=audit_logger
            )
            if success:
                st.success(msg)
                st.rerun()

# =============================================================================
# PAGE 6: AUDIT HISTORY & EVALUATION
# =============================================================================
elif "6. 📜 Audit History & Evaluation" in page:
    st.markdown("## 📜 Immutable Audit History & Sentinel Evaluation")
    st.markdown("Every automated evaluation, safety check, and recovery attempt is permanently recorded. Includes empirical metrics against the benchmark.")

    tab_audit, tab_metrics = st.tabs(["📜 Verifiable Audit Log", "📊 Quantitative Evaluation Metrics"])

    with tab_audit:
        records = audit_logger.get_all_records()
        st.markdown(f"**Total Persistent Audit Records:** `{len(records)}`")

        # Table View
        df_audit = pd.DataFrame([{
            "Audit ID": r["audit_id"],
            "Timestamp": r["timestamp"][11:19],
            "Order ID": r["order_id"],
            "Event": r["event_name"],
            "Previous State": r["previous_state"],
            "New State": r["new_state"],
            "Decision": r["decision"],
            "Retry Attempt": r["retry_attempt"],
            "Result": r["result"],
            "Operator": r.get("operator", "Sentinel-Daemon")
        } for r in reversed(records)])

        st.dataframe(df_audit, use_container_width=True, height=350)

        # Inspect specific audit record
        selected_aud_id = st.selectbox("Inspect Full Safety Checks Payload for Audit Record", [r["audit_id"] for r in reversed(records)])
        chosen_rec = next((r for r in records if r["audit_id"] == selected_aud_id), None)
        if chosen_rec:
            st.json(chosen_rec)

    with tab_metrics:
        st.markdown("### 📊 Algorithmic Evaluation Metrics")
        metrics = metrics_engine.calculate_evaluation_metrics()

        st.info(metrics["benchmark_dataset_disclaimer"])

        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.metric("SLA Detection Precision", f"{metrics['sla_detection_precision']*100:.1f}%")
        with q2:
            st.metric("SLA Detection Recall", f"{metrics['sla_detection_recall']*100:.1f}%")
        with q3:
            st.metric("Root Cause Precision", f"{metrics['root_cause_precision']*100:.1f}%")
        with q4:
            st.metric("Safe Recovery Rate", f"{metrics['safe_recovery_rate']*100:.1f}%")

        st.markdown("---")
        q5, q6, q7, q8 = st.columns(4)
        with q5:
            st.metric("False Automatic Actions", f"{metrics['false_automatic_action_count']}", help="Must be exactly 0 under Prove Before You Retry")
        with q6:
            st.metric("Mean Time to Detect (MTTD)", f"{metrics['mean_time_to_detect_seconds']} sec")
        with q7:
            st.metric("Mean Time to Resolve", f"{metrics['mean_time_to_resolve_minutes']} min")
        with q8:
            st.metric("SLA Breach Rate", f"{metrics['sla_breach_rate']*100:.1f}%")

# =============================================================================
# PAGE 7: HOW THE SYSTEM WORKS (LEARNING MODE)
# =============================================================================
elif "7. 🎓 How the System Works" in page:
    st.markdown("## 🎓 How OrderFlow Sentinel Works — Student Learning Guide")
    st.markdown("Master every layer of the project in clear, simple English. Perfect for hackathon judges and viva presentations!")

    m1, m2, m3, m4, m5, m6 = st.tabs([
        "Module 1: Order Monitoring",
        "Module 2: SLA Risk",
        "Module 3: Root Cause Analysis",
        "Module 4: Safe Recovery",
        "Module 5: Project Architecture",
        "Module 6: Step-by-Step Walkthrough"
    ])

    with m1:
        st.markdown("### 📦 Module 1: Order Monitoring Fundamentals")
        st.markdown("""
        In modern enterprise e-commerce (such as Amazon or Walmart), when a customer places an order, it doesn't just sit in a single database. It flows through a sequence of **microservices**:

        - **Order**: A customer request to buy items, tracked with a unique ID (`ORD-1001`).
        - **Stage**: A distinct phase of processing:
          1. `Order Created` (Validation and payment auth)
          2. `Inventory Check` (Warehouse stock reservation)
          3. `Fulfilment Request` (Packing instructions to 3rd-party logistics)
          4. `Shipment Confirmation` (Carrier label generation)
          5. `Order Completed` (Delivered)
        - **Event**: A timestamped milestone recorded by a service (`INVENTORY_RESERVED`).
        - **API (Application Programming Interface)**: The HTTP bridge used by services to speak with one another (e.g. `POST /api/v2/fulfilment/dispatch`).
        - **SLA (Service Level Agreement)**: A contractual promise of time. For example: *"Every order must be packed and ready to ship within 35 minutes."*
        """)

    with m2:
        st.markdown("### ⏱️ Module 2: SLA Risk vs SLA Breach")
        st.markdown("""
        A common rookie mistake is to only alert when an order has already missed its deadline. By then, the customer is unhappy!

        **OrderFlow Sentinel does something smarter:**
        1. **Configured Stage Warning Thresholds**: Each individual stage has a normal time limit. For example, `Inventory Check` normally takes 1–2 minutes, with a **5-minute warning threshold**.
        2. **Elapsed Stage Time**: If `Inventory Check` reaches 7.5 minutes, Sentinel raises an **At Risk** flag.
        3. **Distinguishing Predicted Risk from Actual Breach**:
           - **At Risk**: The individual stage is slow (7.5 min > 5 min), but 22.5 minutes remain before the customer delivery deadline. *The order is NOT broken yet.*
           - **Breached**: The overall 35-minute deadline has completely elapsed.
        """)

    with m3:
        st.markdown("### 🔍 Module 3: Evidence-Based Root Cause Analysis")
        st.markdown("""
        When an order halts, why did it fail? Sentinel doesn't guess. It correlates logs and timestamps:

        - **HTTP Status Codes**:
          - `504 Gateway Timeout`: The partner service didn't answer in time.
          - `500 Internal Server Error`: The service crashed or ran out of memory.
          - `422 Unprocessable Entity`: Business logic rejection (e.g. out of stock).
        - **Latency Spikes**: Tracking milliseconds spent waiting. A jump from 200ms to 5200ms points to network or server hanging.
        - **Evidence Strength**: Sentinel marks findings as **Conclusive** (when logs cleanly prove failure) or **Incomplete / Ambiguous** (when downstream signals conflict).
        """)

    with m4:
        st.markdown("### 🛡️ Module 4: Safe Recovery — 'Prove Before You Retry'")
        st.markdown("""
        ### The Golden Rule of OrderFlow Sentinel:
        > **A timeout does NOT prove an operation failed.**

        Imagine sending a $5,000 money transfer to a friend. If your banking app screen says "Network Timeout (HTTP 504)", did the bank send the money or not?
        - If you blindly click "Retry Transfer", you might send $10,000!
        - The bank might have processed the money, but your phone lost Wi-Fi before getting the confirmation.

        **Sentinel's 3 Outcomes:**
        1. **Confirmed Success**: Evidence shows the partner finished. **DO NOT RETRY.**
        2. **Confirmed Failure with Safeguards**: We know for certain the request failed, an idempotency key (`X-Idempotency-Key`) prevents duplicates, policy allows it, and we haven't exceeded 2 retries. **PERMIT 1 CONTROLLED RETRY.**
        3. **Uncertain Outcome**: We received a timeout, but an asynchronous queue log hints that the warehouse might have started packing. **STOP AUTOMATION. ESCALATE TO HUMAN OPERATOR.**
        """)

    with m5:
        st.markdown("### 🏗️ Module 5: Project Architecture")
        st.markdown("""
        OrderFlow Sentinel is structured with clean separation between business logic and the user interface:

        ```
        orderflow-sentinel/
        ├── app.py                      # Modern multi-page Streamlit enterprise interface
        ├── modules/
        │   ├── data_generator.py       # Deterministic synthetic orders and reproducible benchmarks
        │   ├── sla_engine.py           # Transparent SLA risk & DecisionTree ML companion
        │   ├── root_cause_engine.py    # Log & event correlation diagnostic engine
        │   ├── recovery_engine.py      # 'Prove Before You Retry' safety evaluation & retry executor
        │   ├── audit_logger.py         # Persistent verifiable audit trail (JSON)
        │   ├── metrics_engine.py       # Mathematical precision, recall, and zero-false-actions
        │   └── workflow_visualizer.py  # Interactive 14-step workflow diagram component
        └── tests/
            └── test_sentinel.py        # Automated pytest suite covering all 7 requirement areas
        ```
        """)

    with m6:
        st.markdown("### 🚶 Module 6: Step-by-Step Walkthrough of Scenario A (ORD-1001)")
        st.caption("Use the Previous and Next buttons to walk through how Sentinel handles an API timeout from start to finish:")

        steps_content = [
            {
                "title": "Step 1: Order Ingestion & Normal Flow",
                "receives": "Customer 'Apex Logistics' places order ORD-1001 for $1,420.50.",
                "logic": "OrderService validates payload and assigns idempotency token `idemp_ord_1001_v1_98f4a2`. Inventory allocated in 1m 20s.",
                "decision": "Order advances to 'Fulfilment Request' stage.",
                "sees": "Green completed events in timeline, healthy stage duration (3.2 min).",
                "matters": "Establishes a verified baseline and idempotency tokens before partner calls are made."
            },
            {
                "title": "Step 2: Partner Fulfilment API Gateway Timeout",
                "receives": "Partner warehouse API hangs. Request takes 5,210ms and returns HTTP 504 Gateway Timeout.",
                "logic": "Sentinel monitor catches HTTP 504 socket disconnection log (LOG-1001-03).",
                "decision": "Order transitioned from IN_PROGRESS to FAILED. Sentinel initiates diagnostic investigation.",
                "sees": "Red failure badge on order, error message explaining partner socket disconnect.",
                "matters": "Instantly flags failing transaction without waiting for customer complaint."
            },
            {
                "title": "Step 3: Correlating Logs & Diagnosing Root Cause",
                "receives": "Full API request/response logs and Kafka event stream history for ORD-1001.",
                "logic": "RootCauseEngine confirms latency exceeded 5,000ms. Verifies no downstream queue events show partial packing.",
                "decision": "Suspected Cause: Upstream Partner API Timeout (HTTP 504). Evidence: Conclusive.",
                "sees": "Root Cause card displaying affected service (`FulfilmentAPI`) and latency waterfall.",
                "matters": "Proves the nature of failure using hard evidence rather than blind assumptions."
            },
            {
                "title": "Step 4: 'Prove Before You Retry' Safety Checklist",
                "receives": "SafeRecoveryEngine inspects order metadata against 6 strict gates.",
                "logic": "1) No prior success? Yes. 2) Evidence certain? Yes. 3) Idempotency token active? Yes. 4) Policy allows? Yes. 5) Retry count 0 < 2? Yes. 6) Low risk? Yes.",
                "decision": "AUTHORIZATION VERDICT: Permit One Controlled Simulated Retry.",
                "sees": "6 green checkmarks on safety checklist. 'Execute Simulated Retry' button unlocks.",
                "matters": "Prevents catastrophic duplicate orders and double billing."
            },
            {
                "title": "Step 5: Controlled Idempotent Retry Execution",
                "receives": "Operator or system clicks 'Execute Simulated Retry'.",
                "logic": "Sentinel dispatches retry call with `X-Idempotency-Key: idemp_ord_1001_v1_98f4a2`. Partner responds HTTP 200 OK (285ms).",
                "decision": "Order state transitions to 'Shipment Confirmation' stage.",
                "sees": "Success alert! New retry log `LOG-1001-R1` recorded.",
                "matters": "The order is recovered automatically and safely without human manual re-entry."
            },
            {
                "title": "Step 6: Verifiable Immutable Audit Logging",
                "receives": "Outcome details from the recovery execution.",
                "logic": "AuditLogger writes permanent JSON record `AUD-0005` with before/after state and safety checklist.",
                "decision": "Decision permanently archived for enterprise compliance.",
                "sees": "New record appears in Audit History table with full audit payload.",
                "matters": "Guarantees regulatory compliance and transparency for auditors and operations leaders."
            }
        ]

        curr_walk_idx = st.session_state.learning_step_index
        step_data = steps_content[curr_walk_idx]

        st.markdown(f"""
        <div style="background: #0f172a; border: 2px solid #0284c7; border-radius: 12px; padding: 22px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 13px; font-weight: 700; color: #38bdf8; text-transform: uppercase;">Walkthrough Step {curr_walk_idx + 1} of {len(steps_content)}</span>
                <span style="font-family: monospace; font-size: 12px; color: #94a3b8;">SCENARIO A: ORD-1001</span>
            </div>
            <h3 style="margin: 0 0 16px 0; color: #f8fafc; font-size: 22px;">{step_data['title']}</h3>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px;">
                <div style="background: #1e293b; padding: 12px 16px; border-radius: 8px;">
                    <div style="font-size: 11px; font-weight: 700; color: #38bdf8; text-transform: uppercase;">1. What the System Receives:</div>
                    <div style="font-size: 13px; color: #cbd5e1; margin-top: 4px;">{step_data['receives']}</div>
                </div>
                <div style="background: #1e293b; padding: 12px 16px; border-radius: 8px;">
                    <div style="font-size: 11px; font-weight: 700; color: #38bdf8; text-transform: uppercase;">2. What Logic Runs:</div>
                    <div style="font-size: 13px; color: #cbd5e1; margin-top: 4px;">{step_data['logic']}</div>
                </div>
                <div style="background: #1e293b; padding: 12px 16px; border-radius: 8px;">
                    <div style="font-size: 11px; font-weight: 700; color: #10b981; text-transform: uppercase;">3. What Decision is Made:</div>
                    <div style="font-size: 13px; color: #f8fafc; font-weight: 600; margin-top: 4px;">{step_data['decision']}</div>
                </div>
                <div style="background: #1e293b; padding: 12px 16px; border-radius: 8px;">
                    <div style="font-size: 11px; font-weight: 700; color: #f59e0b; text-transform: uppercase;">4. What the User Sees:</div>
                    <div style="font-size: 13px; color: #cbd5e1; margin-top: 4px;">{step_data['sees']}</div>
                </div>
            </div>

            <div style="background: rgba(13, 148, 136, 0.15); border: 1px solid #0d9488; padding: 12px 16px; border-radius: 8px;">
                <div style="font-size: 11px; font-weight: 700; color: #14b8a6; text-transform: uppercase;">5. Why the Decision Matters:</div>
                <div style="font-size: 13px; color: #f8fafc; margin-top: 2px;">{step_data['matters']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        w_col1, w_col2, w_col3 = st.columns([1, 4, 1])
        with w_col1:
            if st.button("⬅ Previous", disabled=(curr_walk_idx == 0)):
                st.session_state.learning_step_index = max(0, curr_walk_idx - 1)
                st.rerun()
        with w_col3:
            if st.button("Next ➡", disabled=(curr_walk_idx == len(steps_content) - 1)):
                st.session_state.learning_step_index = min(len(steps_content) - 1, curr_walk_idx + 1)
                st.rerun()
