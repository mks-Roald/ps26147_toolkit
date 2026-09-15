"""
Plotly-based interactive visualization components for Phase 7 UI modernization.
Includes 2D/3D spectrograms, waterfall plots, constellation diagrams with density heatmaps,
and time-domain I/Q waveform viewers with envelope and phase analysis.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal as sp_signal


def plot_interactive_spectrogram_2d(t, f, Sxx_db, title="Interactive Spectrogram", fs=1e6):
    """
    Create an interactive 2D spectrogram using Plotly with hover information.

    Args:
        t: Time vector (seconds)
        f: Frequency vector (Hz)
        Sxx_db: Spectrogram power in dB (2D array)
        title: Plot title
        fs: Sampling rate for display formatting

    Returns:
        Plotly figure object
    """
    fig = go.Figure(data=go.Heatmap(
        x=t * 1000,  # Convert to milliseconds
        y=f / 1000,  # Convert to kHz
        z=Sxx_db,
        colorscale='Viridis',
        colorbar=dict(title="Power (dB)"),
        hovertemplate='Time: %{x:.2f} ms<br>Freq: %{y:.1f} kHz<br>Power: %{z:.1f} dB<extra></extra>',
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (ms)",
        yaxis_title="Frequency (kHz)",
        height=500,
        template="plotly_dark",
        hovermode='closest',
    )

    return fig


def plot_waterfall_3d(t, f, Sxx_db, title="3D Waterfall Display", subsample=4):
    """
    Create an interactive 3D waterfall plot showing time-frequency-power relationship.

    Args:
        t: Time vector (seconds)
        f: Frequency vector (Hz)
        Sxx_db: Spectrogram power in dB
        title: Plot title
        subsample: Subsampling factor for performance (every Nth time slice)

    Returns:
        Plotly figure object
    """
    # Subsample for performance
    t_sub = t[::subsample]
    Sxx_sub = Sxx_db[:, ::subsample]

    fig = go.Figure(data=[go.Surface(
        x=t_sub * 1000,  # Time in ms
        y=f / 1000,      # Frequency in kHz
        z=Sxx_sub,
        colorscale='Jet',
        colorbar=dict(title="Power (dB)"),
        hovertemplate='Time: %{x:.2f} ms<br>Freq: %{y:.1f} kHz<br>Power: %{z:.1f} dB<extra></extra>',
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Time (ms)",
            yaxis_title="Frequency (kHz)",
            zaxis_title="Power (dB)",
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.3)
            ),
        ),
        height=600,
        template="plotly_dark",
    )

    return fig


def get_ideal_constellation(modulation):
    """
    Generate ideal reference constellation points for various modulation schemes.

    Args:
        modulation: String identifier (BPSK, QPSK, 8PSK, 16QAM, 64QAM)

    Returns:
        Complex numpy array of ideal constellation points
    """
    if modulation == "BPSK":
        return np.array([-1, 1], dtype=complex)

    elif modulation == "QPSK":
        return np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)

    elif modulation == "8PSK":
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
        return np.exp(1j * angles)

    elif modulation == "16QAM":
        levels = np.array([-3, -1, 1, 3])
        grid = np.array([i + 1j*q for i in levels for q in levels])
        return grid / np.sqrt(10)  # Normalize to unit average power

    elif modulation == "64QAM":
        levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        grid = np.array([i + 1j*q for i in levels for q in levels])
        return grid / np.sqrt(42)  # Normalize to unit average power

    else:
        return np.array([])


def plot_constellation_interactive(symbols, modulation="QPSK", title="Constellation Diagram",
                                   show_ideal=True, use_hexbin=True, max_points=10000):
    """
    Create interactive constellation diagram with optional density heatmap and ideal reference overlay.

    Args:
        symbols: Complex symbol array
        modulation: Modulation scheme name
        title: Plot title
        show_ideal: Whether to overlay ideal constellation points
        use_hexbin: Use 2D histogram density instead of scatter (better for large datasets)
        max_points: Maximum number of points to display in scatter mode

    Returns:
        Plotly figure object
    """
    I = np.real(symbols)
    Q = np.imag(symbols)

    fig = go.Figure()

    # Plot received symbols
    if use_hexbin and len(symbols) > 1000:
        # Use 2D histogram for density visualization
        fig.add_trace(go.Histogram2d(
            x=I,
            y=Q,
            colorscale='Hot',
            nbinsx=100,
            nbinsy=100,
            colorbar=dict(title="Count"),
            hovertemplate='I: %{x:.3f}<br>Q: %{y:.3f}<br>Count: %{z}<extra></extra>',
            name='Received Symbols',
        ))
    else:
        # Use scatter plot for smaller datasets
        sample_indices = np.random.choice(len(symbols), min(max_points, len(symbols)), replace=False)
        fig.add_trace(go.Scatter(
            x=I[sample_indices],
            y=Q[sample_indices],
            mode='markers',
            marker=dict(
                size=3,
                color='cyan',
                opacity=0.5,
                line=dict(width=0),
            ),
            hovertemplate='I: %{x:.3f}<br>Q: %{y:.3f}<extra></extra>',
            name='Received Symbols',
        ))

    # Overlay ideal constellation points
    if show_ideal:
        ideal = get_ideal_constellation(modulation)
        if len(ideal) > 0:
            fig.add_trace(go.Scatter(
                x=np.real(ideal),
                y=np.imag(ideal),
                mode='markers',
                marker=dict(
                    size=12,
                    color='red',
                    symbol='x',
                    line=dict(width=2, color='white'),
                ),
                name='Ideal Reference',
                hovertemplate='Ideal I: %{x:.3f}<br>Ideal Q: %{y:.3f}<extra></extra>',
            ))

    # Calculate plot limits based on data extent
    max_val = max(np.abs(I).max() if len(I) > 0 else 1.5,
                  np.abs(Q).max() if len(Q) > 0 else 1.5) * 1.2

    fig.update_layout(
        title=title,
        xaxis_title="In-Phase (I)",
        yaxis_title="Quadrature (Q)",
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[-max_val, max_val], zeroline=True),
        yaxis=dict(range=[-max_val, max_val], zeroline=True),
        height=600,
        template="plotly_dark",
        hovermode='closest',
        showlegend=True,
    )

    return fig


def plot_time_domain_iq(signal, fs=1e6, title="Time-Domain I/Q Analysis", max_samples=5000):
    """
    Create comprehensive time-domain visualization with I/Q channels, envelope, and instantaneous phase.

    Args:
        signal: Complex baseband signal
        fs: Sampling rate (Hz)
        title: Plot title
        max_samples: Maximum number of samples to display

    Returns:
        Plotly figure with 4 subplots
    """
    # Limit samples for performance
    if len(signal) > max_samples:
        indices = np.linspace(0, len(signal)-1, max_samples, dtype=int)
        signal_sub = signal[indices]
    else:
        signal_sub = signal
        indices = np.arange(len(signal))

    t = indices / fs * 1000  # Time in milliseconds
    I = np.real(signal_sub)
    Q = np.imag(signal_sub)
    envelope = np.abs(signal_sub)
    phase = np.angle(signal_sub)

    # Unwrap phase for continuity
    phase_unwrapped = np.unwrap(phase)

    # Create subplots: I, Q, Envelope, Phase
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=("In-Phase (I)", "Quadrature (Q)", "Instantaneous Envelope", "Instantaneous Phase"),
        vertical_spacing=0.08,
    )

    # I channel
    fig.add_trace(go.Scatter(
        x=t, y=I,
        mode='lines',
        line=dict(color='#1f77b4', width=1),
        name='I',
        hovertemplate='Time: %{x:.3f} ms<br>I: %{y:.4f}<extra></extra>',
    ), row=1, col=1)

    # Q channel
    fig.add_trace(go.Scatter(
        x=t, y=Q,
        mode='lines',
        line=dict(color='#ff7f0e', width=1),
        name='Q',
        hovertemplate='Time: %{x:.3f} ms<br>Q: %{y:.4f}<extra></extra>',
    ), row=2, col=1)

    # Envelope
    fig.add_trace(go.Scatter(
        x=t, y=envelope,
        mode='lines',
        line=dict(color='#2ca02c', width=1.5),
        name='Envelope',
        hovertemplate='Time: %{x:.3f} ms<br>|s|: %{y:.4f}<extra></extra>',
    ), row=3, col=1)

    # Phase
    fig.add_trace(go.Scatter(
        x=t, y=phase_unwrapped,
        mode='lines',
        line=dict(color='#d62728', width=1),
        name='Phase',
        hovertemplate='Time: %{x:.3f} ms<br>Phase: %{y:.4f} rad<extra></extra>',
    ), row=4, col=1)

    # Update axes labels
    fig.update_xaxes(title_text="Time (ms)", row=4, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    fig.update_yaxes(title_text="Magnitude", row=3, col=1)
    fig.update_yaxes(title_text="Phase (rad)", row=4, col=1)

    fig.update_layout(
        title_text=title,
        height=900,
        template="plotly_dark",
        showlegend=False,
        hovermode='x unified',
    )

    return fig


def plot_psd_interactive(freqs, psd_db, center_freq=0, bandwidth=0, title="Power Spectral Density"):
    """
    Create interactive PSD plot with optional signal bandwidth overlay.

    Args:
        freqs: Frequency vector (Hz)
        psd_db: Power spectral density in dB
        center_freq: Detected center frequency (Hz)
        bandwidth: Detected bandwidth (Hz)
        title: Plot title

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Main PSD curve
    fig.add_trace(go.Scatter(
        x=freqs / 1000,  # kHz
        y=psd_db,
        mode='lines',
        line=dict(color='cyan', width=1.5),
        name='PSD',
        hovertemplate='Freq: %{x:.2f} kHz<br>Power: %{y:.2f} dB<extra></extra>',
    ))

    # Add bandwidth region if provided
    if bandwidth > 0 and center_freq != 0:
        f_low = (center_freq - bandwidth/2) / 1000
        f_high = (center_freq + bandwidth/2) / 1000

        fig.add_vrect(
            x0=f_low, x1=f_high,
            fillcolor="green", opacity=0.2,
            layer="below", line_width=0,
            annotation_text="Signal BW",
            annotation_position="top left",
        )

        # Mark center frequency
        fig.add_vline(
            x=center_freq / 1000,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"fc: {center_freq/1000:.1f} kHz",
            annotation_position="top",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Frequency (kHz)",
        yaxis_title="Power (dB)",
        height=400,
        template="plotly_dark",
        hovermode='x unified',
    )

    return fig


def create_telemetry_card(stage_info):
    """
    Create a pipeline telemetry status card showing processing stages.

    Args:
        stage_info: Dictionary with stage names as keys and status dicts as values
                   Each status dict should have: {'status': 'complete'|'running'|'pending'|'error',
                                                  'metric': optional metric value,
                                                  'duration_ms': optional processing time}

    Returns:
        HTML string for rendering in Streamlit
    """
    status_emoji = {
        'complete': '✅',
        'running': '⚙️',
        'pending': '⏳',
        'error': '❌',
        'skipped': '⏭️',
    }

    status_color = {
        'complete': '#28a745',
        'running': '#ffc107',
        'pending': '#6c757d',
        'error': '#dc3545',
        'skipped': '#17a2b8',
    }

    card_html = """
    <style>
    .telemetry-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin: 15px 0;
    }
    .telemetry-header {
        font-size: 18px;
        font-weight: bold;
        color: #00d4ff;
        margin-bottom: 15px;
        border-bottom: 2px solid #00d4ff;
        padding-bottom: 8px;
    }
    .stage-row {
        display: flex;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #3a3a3a;
    }
    .stage-icon {
        font-size: 20px;
        margin-right: 12px;
    }
    .stage-name {
        flex: 1;
        font-size: 14px;
        color: #e0e0e0;
    }
    .stage-metric {
        font-size: 13px;
        font-weight: bold;
        margin-left: 10px;
    }
    .stage-duration {
        font-size: 11px;
        color: #a0a0a0;
        margin-left: 8px;
    }
    </style>

    <div class="telemetry-card">
        <div class="telemetry-header">📊 Pipeline Execution Telemetry</div>
    """

    for stage_name, info in stage_info.items():
        status = info.get('status', 'pending')
        metric = info.get('metric', '')
        duration = info.get('duration_ms', None)

        emoji = status_emoji.get(status, '⏳')
        color = status_color.get(status, '#6c757d')

        metric_html = f'<span class="stage-metric" style="color: {color};">{metric}</span>' if metric else ''
        duration_html = f'<span class="stage-duration">({duration:.1f} ms)</span>' if duration else ''

        card_html += f"""
        <div class="stage-row">
            <span class="stage-icon">{emoji}</span>
            <span class="stage-name">{stage_name}</span>
            {metric_html}
            {duration_html}
        </div>
        """

    card_html += "</div>"
    return card_html


def plot_correlation_interactive(correlation_curve, peak_indices, threshold, title="Sliding Window Cross-Correlation"):
    """
    Create an interactive correlation curve plot using Plotly with peak detection visualization.

    Args:
        correlation_curve: Array of correlation values
        peak_indices: Indices where peaks were detected
        threshold: Detection threshold value
        title: Plot title

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Main correlation curve
    fig.add_trace(go.Scatter(
        x=list(range(len(correlation_curve))),
        y=correlation_curve,
        mode='lines',
        line=dict(color='#ff7f0e', width=1.5),
        name='Normalized Correlation',
        hovertemplate='Bit Index: %{x}<br>Correlation: %{y:.3f}<extra></extra>',
    ))

    # Detection threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="gray",
        line_width=1,
        annotation_text=f"Threshold ({threshold:.2f})",
        annotation_position="top right",
    )

    # Peak detection markers
    if len(peak_indices) > 0:
        peak_values = [correlation_curve[i] for i in peak_indices if i < len(correlation_curve)]
        peak_indices_valid = [i for i in peak_indices if i < len(correlation_curve)]

        if len(peak_indices_valid) > 0:
            fig.add_trace(go.Scatter(
                x=peak_indices_valid,
                y=peak_values,
                mode='markers',
                marker=dict(
                    color='red',
                    size=8,
                    symbol='x',
                    line=dict(width=2),
                ),
                name='Sync Peaks',
                hovertemplate='Bit Index: %{x}<br>Correlation: %{y:.3f}<extra></extra>',
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Bit Index Offset",
        yaxis_title="Normalized Correlation Score",
        yaxis=dict(range=[-1.1, 1.1]),
        height=400,
        template="plotly_dark",
        hovermode='x unified',
        showlegend=True,
    )

    return fig
