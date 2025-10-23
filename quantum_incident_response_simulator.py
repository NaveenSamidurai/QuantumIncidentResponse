"""
Quantum Incident Response Simulator
Single-file Streamlit app
Save as: quantum_incident_response_simulator.py
Run: streamlit run quantum_incident_response_simulator.py

This app creates a GUI to simulate incident generation and response using a
"quantum-inspired" probabilistic model: incidents are created in a
superposition of states (severity / detectability) and a measurement step
collapses them into concrete outcomes. The app lets you tune parameters,
run Monte Carlo simulations, and view final graphs showing resolved vs
unresolved incidents and response-times distributions.

Dependencies:
    pip install streamlit numpy pandas matplotlib

Author: ChatGPT (GPT-5 Thinking mini)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time

st.set_page_config(page_title="Quantum Incident Response Simulator", layout="wide")

# ----------------------------- Helper functions -----------------------------

def quantum_measure(amplitudes):
    """Given complex amplitudes (or non-negative amplitudes), measure the
    state by converting to probabilities and sampling one outcome.
    This is a 'quantum-inspired' collapse step.
    """
    probs = np.abs(amplitudes) ** 2
    probs = probs / probs.sum()
    return np.random.choice(len(probs), p=probs)


def generate_incidents(n_incidents, seed=None, severity_profile=None, type_profile=None):
    """Generate a DataFrame of incidents using 'superposition' amplitudes for
    both severity and detection difficulty.

    severity_profile: list of amplitudes for Low/Medium/High/Critical
    type_profile: list of amplitudes for types (e.g., Malware, Phishing, etc.)
    """
    rng = np.random.default_rng(seed)

    # Default severity and type amplitudes if not provided
    if severity_profile is None:
        severity_profile = np.array([0.8, 1.2, 1.5, 1.0])  # arbitrary amplitudes
    if type_profile is None:
        type_profile = np.array([1.0, 0.9, 0.7, 0.6])

    severities = []
    types = []
    detectability_scores = []
    timestamps = []

    for i in range(n_incidents):
        # Create a superposition for severity and then measure
        sev_choice = quantum_measure(severity_profile)
        type_choice = quantum_measure(type_profile)

        # detection difficulty modeled inversely with severity (but noisy)
        base_detect_difficulty = (sev_choice + 1) / len(severity_profile)  # 0..1 approx
        detect_noise = rng.normal(0, 0.12)
        detectability = np.clip(1 - (base_detect_difficulty + detect_noise), 0.02, 0.99)

        severities.append(sev_choice)  # 0:Low, 1:Med, 2:High, 3:Critical
        types.append(type_choice)
        detectability_scores.append(detectability)

        # timestamp (seconds) spread over an imaginary timeline
        timestamps.append(rng.integers(0, 3600))

    df = pd.DataFrame({
        'incident_id': np.arange(1, n_incidents + 1),
        'timestamp_s': timestamps,
        'severity_code': severities,
        'type_code': types,
        'detectability': detectability_scores
    })

    severity_map = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}
    type_map = {0: 'Malware', 1: 'Phishing', 2: 'Insider', 3: 'Misconfiguration'}

    df['severity'] = df['severity_code'].map(severity_map)
    df['type'] = df['type_code'].map(type_map)

    return df


def simulate_response(df, detection_model_strength=0.5, avg_response_time=300, responders=3, seed=None):
    """Simulate detection and response for each incident.

    detection_model_strength: multiplier that increases chance-of-detection
    avg_response_time: average seconds to respond once detected
    responders: how many responders are available (affects queueing)
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    # Sort by timestamp to simulate timeline
    df = df.sort_values('timestamp_s').reset_index(drop=True)

    detected = []
    detection_time = []
    resolved = []
    resolution_time = []

    # Simple responder queue: track when each responder becomes free
    responder_free_at = np.zeros(responders)

    for idx, row in df.iterrows():
        base_det = 1 - row['detectability']  # higher for easy-to-detect
        # detection probability influenced by model strength and severity
        severity_factor = 1.0 + (row['severity_code'] * 0.2)
        prob_detect = np.clip(base_det * detection_model_strength * severity_factor, 0.0, 0.999)

        is_detected = rng.random() < prob_detect
        detected.append(is_detected)

        if not is_detected:
            detection_time.append(np.nan)
            resolved.append(False)
            resolution_time.append(np.nan)
            continue

        # Simulate time between incident occurrence and detection
        detection_delay = max(1, int(rng.exponential(60) * (1.0 / detection_model_strength)))
        detect_t = row['timestamp_s'] + detection_delay
        detection_time.append(detect_t)

        # assign responder: earliest free responder
        earliest_idx = int(np.argmin(responder_free_at))
        start_time = max(detect_t, int(responder_free_at[earliest_idx]))

        # response duration depends on severity
        severity_multiplier = 1.0 + row['severity_code'] * 0.9
        resp_duration = max(30, int(rng.normal(avg_response_time * severity_multiplier, avg_response_time * 0.3)))
        finish_time = start_time + resp_duration

        # update responder availability
        responder_free_at[earliest_idx] = finish_time + rng.integers(10, 30)  # cooldown between tasks

        resolved.append(True)
        resolution_time.append(finish_time)

    df['detected'] = detected
    df['detection_time_s'] = detection_time
    df['resolved'] = resolved
    df['resolution_time_s'] = resolution_time

    # Derive some helpful metrics
    df['time_to_detect_s'] = df['detection_time_s'] - df['timestamp_s']
    df['time_to_resolve_s'] = df['resolution_time_s'] - df['timestamp_s']

    return df


# ----------------------------- Streamlit UI ---------------------------------

st.title("🧪 Quantum Incident Response Simulator")
st.markdown(
    "Simulate incident creation and an incident response team using a \"quantum-inspired\"\n"
    "measurement step (amplitude -> probability -> collapse). Use the controls to tune\n"
    "incident properties, detection model strength, responders and run Monte Carlo runs."
)

with st.sidebar.expander("Simulation controls", expanded=True):
    n_incidents = st.number_input("Number of incidents", min_value=1, max_value=5000, value=200, step=10)
    seed = st.number_input("Random seed (0 = random)", value=0, step=1)
    if seed == 0:
        seed = None

    detection_model_strength = st.slider("Detection model strength", 0.05, 2.0, 0.7, 0.05)
    avg_response_time = st.number_input("Average response time (seconds)", min_value=30, max_value=7200, value=300, step=10)
    responders = st.number_input("Number of responders", min_value=1, max_value=50, value=3, step=1)
    runs = st.number_input("Monte Carlo runs", min_value=1, max_value=200, value=1, step=1)

    show_raw = st.checkbox("Show raw incident table after simulation", value=False)
    run_sim = st.button("Run simulation")


# Information / legend
st.sidebar.markdown("**Legend:**\n\n- Severity: Low / Medium / High / Critical\n- Type: Malware / Phishing / Insider / Misconfiguration\n- Detection model strength: how good your tooling/monitoring is\n")

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Quick tips")
    st.write("• Increase detection strength to find more incidents faster.")
    st.write("• More responders reduce queueing but cost more in real life.")
    st.write("• Set Monte Carlo runs >1 to see average behaviour.")

# Run simulation(s)
if run_sim:
    all_runs_frames = []
    progress_text = st.empty()
    progress_bar = st.progress(0)

    for r in range(int(runs)):
        progress_text.text(f"Running simulation {r + 1} / {int(runs)}")
        # small different seed per run to vary
        df_inc = generate_incidents(n_incidents=n_incidents, seed=(None if seed is None else seed + r))
        df_res = simulate_response(df_inc, detection_model_strength=detection_model_strength,
                                   avg_response_time=avg_response_time, responders=responders,
                                   seed=(None if seed is None else seed + r + 1000))
        all_runs_frames.append(df_res)
        progress_bar.progress(int((r + 1) / int(runs) * 100))
        time.sleep(0.02)

    progress_text.text("Simulation completed.")

    # Combine results for aggregated metrics
    combined = pd.concat(all_runs_frames, ignore_index=True)

    st.header("Simulation summary")
    col_a, col_b, col_c, col_d = st.columns(4)
    total_incidents = len(combined)
    detected_count = int(combined['detected'].sum())
    resolved_count = int(combined['resolved'].sum())
    avg_ttd = float(combined['time_to_detect_s'].dropna().mean()) if detected_count > 0 else np.nan
    avg_ttr = float(combined['time_to_resolve_s'].dropna().mean()) if resolved_count > 0 else np.nan

    col_a.metric("Total incidents (runs combined)", f"{total_incidents}")
    col_b.metric("Detected incidents", f"{detected_count}")
    col_c.metric("Resolved incidents", f"{resolved_count}")
    col_d.metric("Avg time-to-resolve (s)", f"{int(avg_ttr) if not np.isnan(avg_ttr) else 'N/A'}")

    # ----------------------------- Graphs ---------------------------------
    st.subheader("Graphs: Resolved vs Unresolved by Severity")

    # prepare grouped counts
    group = combined.groupby(['severity', 'resolved']).size().unstack(fill_value=0)
    group = group.reindex(['Low', 'Medium', 'High', 'Critical'])  # ensure order

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    group.plot(kind='bar', stacked=True, ax=ax1)
    ax1.set_title('Resolved (stacked) by severity')
    ax1.set_ylabel('Count')
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    st.pyplot(fig1)

    st.subheader("Distribution: Time to Resolve (seconds)")
    resolved_series = combined['time_to_resolve_s'].dropna()
    if resolved_series.empty:
        st.info("No resolved incidents in this simulation run — try increasing detection strength or responders.")
    else:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.hist(resolved_series, bins=30)
        ax2.set_xlabel('Seconds')
        ax2.set_ylabel('Incidents')
        ax2.set_title('Histogram of time-to-resolve (seconds)')
        st.pyplot(fig2)

    st.subheader("Timeline: Incidents over simulated hour (sample)")
    # Timeline: counts per minute
    combined_sample = combined.copy()
    combined_sample['minute'] = (combined_sample['timestamp_s'] / 60).astype(int)
    timeline = combined_sample.groupby('minute').size()

    fig3, ax3 = plt.subplots(figsize=(10, 3))
    ax3.plot(timeline.index, timeline.values)
    ax3.set_xlabel('Minute')
    ax3.set_ylabel('Incidents')
    ax3.set_title('Incidents per minute (simulated hour)')
    st.pyplot(fig3)

    if show_raw:
        st.subheader("Raw incident data (sample)")
        st.dataframe(combined.sample(min(200, len(combined))).reset_index(drop=True))

    # Offer CSV download
    csv = combined.to_csv(index=False)
    st.download_button("Download combined results (CSV)", csv, file_name='quantum_ir_results.csv', mime='text/csv')

    st.success("Done — tweak parameters and run again!")

else:
    st.info("Configure simulation options in the left panel and click 'Run simulation' to start.")


# ----------------------------- Footer -------------------------------------
st.markdown("---")
st.markdown(
    "**Notes:** This is a toy, quantum-inspired simulator. The 'quantum' parts are\n"
    "metaphorical: amplitudes -> probabilities -> measurement provide a fun way to\n"
    "reason about non-deterministic incident generation. For production-grade modeling,\n"
    "replace the measurement/detection models with real detectors (SIEM, ML models)\n"
    "and add persistence, authentication, and richer queues (priorities, skills, escalation)."
)

st.caption("Generated by GPT-5 Thinking mini — adapt and extend as you like!")
