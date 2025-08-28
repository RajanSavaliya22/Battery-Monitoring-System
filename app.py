import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import datetime
import json
from io import BytesIO
from streamlit_autorefresh import st_autorefresh

# Refresh every 2 seconds
st_autorefresh(interval=5000, key="cell_refresh")

# Page configuration
st.set_page_config(
    page_title="Battery Cell Statistics Dashboard",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cell specifications
CELL_SPECS = {
    'lfp': {
        'voltage': 3.2,
        'min_voltage': 2.8,
        'max_voltage': 3.6,
        'nominal_capacity': 100,  # Ah
        'max_discharge_rate': 3.0,  # C-rate
        'efficiency': 0.95,
        'cycle_life': 6000,
        'color': '#2E86AB'
    },
    'nmc': {
        'voltage': 3.6,
        'min_voltage': 3.2,
        'max_voltage': 4.0,
        'nominal_capacity': 120,  # Ah
        'max_discharge_rate': 2.0,  # C-rate
        'efficiency': 0.92,
        'cycle_life': 3000,
        'color': '#A23B72'
    }
}

# Process colors for visualization
PROCESS_COLORS = {
    'CC/CV': '#28A745',  # Green for Charging (CC/CV)
    'CCD': '#FD7E14',  # Orange for Discharging (CCD)
    'Idle': '#007BFF',  # Blue for Idle
    'stopped': '#6C757D'  # Gray for Stopped/Default
}

# Initialize session state
if 'processes' not in st.session_state:
    st.session_state.processes = []
if 'running_cells' not in st.session_state:
    st.session_state.running_cells = {}
if 'cell_data' not in st.session_state:
    st.session_state.cell_data = {}
if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'


def create_cell(cell_id, cell_type, process_name, cell_index):
    """Create a new battery cell with initial parameters"""
    specs = CELL_SPECS[cell_type]
    return {
        'id': cell_id,
        'process_name': process_name,
        'cell_index': cell_index,
        'type': cell_type,
        'voltage': specs['voltage'] + np.random.uniform(-0.1, 0.1),
        'current': 0,
        'capacity': specs['nominal_capacity'],
        'temperature': 25 + np.random.uniform(-2, 5),
        'cycles': np.random.randint(0, 100),
        'status': 'idle',
        'current_process': None,
        'process_step': 0,
        'step_elapsed_time': 0,
        'total_elapsed_time': 0,
        'start_time': None,
        'history': [],
        'created_at': datetime.datetime.now()
    }


def simulate_process_step(cell_id, process_step):
    """Simulate battery cell behavior for specific process step"""
    if cell_id not in st.session_state.cell_data:
        return

    cell = st.session_state.cell_data[cell_id]
    if cell['status'] != 'running':
        return

    specs = CELL_SPECS[cell['type']]
    process_info = process_step

    # Base noise
    voltage_noise = np.random.uniform(-0.02, 0.02)
    current_noise = np.random.uniform(-0.5, 0.5)
    temp_noise = np.random.uniform(-0.2, 0.2)

    if process_info['type'] == 'CC/CV':  # Charging (CC/CV)
        target_current = process_info['current']
        target_voltage = process_info['voltage']
        new_current = max(0, target_current + current_noise)

        # CC phase initially, then CV when voltage target reached
        if cell['voltage'] < target_voltage:
            # CC phase - voltage increases
            voltage_change = (new_current * 0.001)
            new_voltage = np.clip(
                cell['voltage'] + voltage_change + voltage_noise,
                specs['min_voltage'],
                target_voltage
            )
        else:
            # CV phase - maintain voltage, current decreases
            new_voltage = target_voltage + voltage_noise * 0.1
            current_decay = max(0.1, target_current * np.exp(-cell['step_elapsed_time'] / 200))
            new_current = current_decay + current_noise

    elif process_info['type'] == 'CCD':  # Constant Current Discharge
        target_current = -abs(process_info['current'])  # Negative for discharge
        new_current = target_current + current_noise
        # Voltage decreases during discharge
        voltage_drop = abs(new_current) * 0.001
        new_voltage = np.clip(
            cell['voltage'] - voltage_drop + voltage_noise,
            specs['min_voltage'],
            specs['max_voltage']
        )

    else:  # Idle
        new_current = 0 + current_noise * 0.1  # Very small current fluctuation
        new_voltage = cell['voltage'] + voltage_noise * 0.1  # Small voltage drift

    # Temperature simulation
    temp_increase = abs(new_current) * 0.01  # Current generates heat
    new_temperature = np.clip(
        25 + temp_increase + temp_noise,
        20, 60
    )

    # Update cell data
    cell['voltage'] = new_voltage
    cell['current'] = new_current
    cell['temperature'] = new_temperature
    cell['step_elapsed_time'] += 1
    cell['total_elapsed_time'] += 1

    # Add to history
    history_entry = {
        'time': cell['total_elapsed_time'],
        'step_time': cell['step_elapsed_time'],
        'process_type': process_info['type'],
        'voltage': new_voltage,
        'current': new_current,
        'temperature': new_temperature,
        'timestamp': datetime.datetime.now()
    }

    cell['history'].append(history_entry)

    # Keep only last 1000 entries
    if len(cell['history']) > 1000:
        cell['history'] = cell['history'][-1000:]


def advance_process_step(cell_id):
    """Check if current step is complete and advance to next step"""
    cell = st.session_state.cell_data[cell_id]
    if cell['status'] != 'running':
        return

    # Find the process
    process = None
    for p in st.session_state.processes:
        if cell_id in p['cells']:
            process = p
            break

    if not process:
        return

    current_step = process['process_steps'][cell['process_step']]

    # Check if step is complete
    if cell['step_elapsed_time'] >= current_step['time']:
        # Move to next step
        cell['process_step'] += 1
        cell['step_elapsed_time'] = 0

        # Check if all steps completed
        if cell['process_step'] >= len(process['process_steps']):
            cell['status'] = 'completed'
            cell['current_process'] = None
            if cell_id in st.session_state.running_cells:
                del st.session_state.running_cells[cell_id]
        else:
            cell['current_process'] = process['process_steps'][cell['process_step']]['type']


def export_process_data(process_name):
    """Export process data to CSV"""
    process_cells = [cell for cell in st.session_state.cell_data.values()
                     if cell['process_name'] == process_name]

    if not process_cells:
        return None

    all_data = []
    for cell in process_cells:
        for entry in cell['history']:
            all_data.append({
                'Process': process_name,
                'Cell_ID': cell['id'],
                'Cell_Index': cell['cell_index'],
                'Cell_Type': cell['type'],
                'Process_Type': entry['process_type'],
                'Total_Time': entry['time'],
                'Step_Time': entry['step_time'],
                'Voltage': entry['voltage'],
                'Current': entry['current'],
                'Temperature': entry['temperature'],
                'Timestamp': entry['timestamp']
            })

    if not all_data:
        return None

    df = pd.DataFrame(all_data)
    return df


# Navigation
st.sidebar.title("Navigation")
if st.sidebar.button("🏠 Dashboard"):
    st.session_state.page = 'dashboard'
if st.sidebar.button("➕ Create Process"):
    st.session_state.page = 'create_process'

# Page routing
if st.session_state.page == 'create_process':
    st.title("➕ Create New Process")

    # Process basic info
    col1, col2 = st.columns(2)
    with col1:
        process_name = st.text_input("Process Name", placeholder="Enter process name")
    with col2:
        cell_count = st.number_input("Number of Cells", min_value=1, max_value=8, value=4)

    # Individual cell configuration
    st.subheader("Cell Configuration")
    cell_types = []

    cols = st.columns(min(4, cell_count))  # Max 4 columns per row
    for i in range(cell_count):
        col_index = i % 4
        with cols[col_index]:
            cell_type = st.selectbox(
                f"Cell {i + 1} Type",
                options=['lfp', 'nmc'],
                key=f"cell_type_{i}"
            )
            cell_types.append(cell_type)

    # Process steps configuration
    st.subheader("Process Steps Configuration")

    if 'process_steps' not in st.session_state:
        st.session_state.process_steps = []

    # Add new process step
    with st.expander("Add New Process Step", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            step_type = st.selectbox("Process Type", ['CC/CV', 'CCD', 'Idle'])

        with col2:
            step_time = st.number_input("Duration (seconds)", min_value=1, value=60)

        if step_type == 'CC/CV':
            with col3:
                current_value = st.number_input("Current (A)", value=10.0, step=0.1)
            with col4:
                voltage_value = st.number_input("Max Voltage (V)", value=3.6, step=0.1)
            step_params = {'current': current_value, 'voltage': voltage_value}

        elif step_type == 'CCD':
            with col3:
                discharge_current = st.number_input("Discharge Current (A)", value=5.0, step=0.1)
            with col4:
                min_voltage = st.number_input("Min Voltage (V)", value=2.8, step=0.1)
            step_params = {'current': discharge_current, 'min_voltage': min_voltage}

        else:  # Idle
            with col3:
                st.write("No current flow")
            with col4:
                st.write("Rest period")
            step_params = {}

        if st.button("Add Step"):
            step = {
                'type': step_type,
                'time': step_time,
                **step_params
            }
            st.session_state.process_steps.append(step)
            st.success(f"Added {step_type} step for {step_time} seconds")

    # Display configured steps
    if st.session_state.process_steps:
        st.subheader("Configured Process Steps")
        for i, step in enumerate(st.session_state.process_steps):
            col1, col2, col3, col4 = st.columns([2, 2, 4, 1])

            with col1:
                st.write(f"**Step {i + 1}:** {step['type']}")
            with col2:
                st.write(f"**Duration:** {step['time']}s")
            with col3:
                if step['type'] == 'CC/CV':
                    st.write(f"Current: {step['current']}A, Max-V: {step['voltage']}V")
                elif step['type'] == 'CCD':
                    st.write(f"Discharge: {step['current']}A, Min-V: {step['min_voltage']}V")
                else:
                    st.write("Idle state")
            with col4:
                if st.button("🗑️", key=f"delete_step_{i}"):
                    st.session_state.process_steps.pop(i)
                    st.rerun()

    # Create process button
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Clear Steps"):
            st.session_state.process_steps = []
            st.rerun()

    with col2:
        if st.button("Create Process", type="primary"):
            if not process_name:
                st.error("Please enter a process name!")
            elif process_name in [p['name'] for p in st.session_state.processes]:
                st.error("Process name already exists!")
            elif not st.session_state.process_steps:
                st.error("Please add at least one process step!")
            else:
                # Create cells
                cells = []
                for i in range(cell_count):
                    cell_id = f"{process_name}_cell_{i + 1}"
                    cell = create_cell(cell_id, cell_types[i], process_name, i + 1)
                    st.session_state.cell_data[cell_id] = cell
                    cells.append(cell_id)

                # Create process
                process = {
                    'name': process_name,
                    'cell_count': cell_count,
                    'cell_types': cell_types,
                    'cells': cells,
                    'process_steps': st.session_state.process_steps.copy(),
                    'created_at': datetime.datetime.now()
                }

                st.session_state.processes.append(process)
                st.session_state.process_steps = []  # Clear steps
                st.success(f"Process '{process_name}' created successfully!")

    with col3:
        if st.button("Back to Dashboard"):
            st.session_state.page = 'dashboard'

else:  # Dashboard page
    st.title("🔋 Battery Cell Statistics Dashboard")

    if not st.session_state.processes:
        st.info("No processes available. Please create a process first.")
        if st.button("➕ Create First Process"):
            st.session_state.page = 'create_process'
            st.rerun()
    else:
        # Process selection
        with st.sidebar:
            st.header("Process Selection")
            process_names = [p['name'] for p in st.session_state.processes]
            selected_process_name = st.selectbox("Choose Process", process_names)

            selected_process = next(p for p in st.session_state.processes if p['name'] == selected_process_name)

            # Export data button
            if st.button("📊 Export Process Data"):
                export_df = export_process_data(selected_process_name)
                if export_df is not None and not export_df.empty:
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{selected_process_name}_data.csv",
                        mime="text/csv"
                    )
                    st.success("Data exported successfully!")
                else:
                    st.warning("No data available to export!")

        # Process overview
        st.header(f"Process: {selected_process['name']}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cells", selected_process['cell_count'])
        with col2:
            running_count = sum(1 for cell_id in selected_process['cells']
                                if st.session_state.cell_data[cell_id]['status'] == 'running')
            st.metric("Running Cells", running_count)
        with col3:
            completed_count = sum(1 for cell_id in selected_process['cells']
                                  if st.session_state.cell_data[cell_id]['status'] == 'completed')
            st.metric("Completed Cells", completed_count)
        with col4:
            total_steps = len(selected_process['process_steps'])
            st.metric("Process Steps", total_steps)

        # Process steps display
        with st.expander("Process Steps Overview", expanded=False):
            for i, step in enumerate(selected_process['process_steps']):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Step {i + 1}:** {step['type']}")
                with col2:
                    st.write(f"**Duration:** {step['time']}s")
                with col3:
                    if step['type'] == 'CC/CV':
                        st.write(f"Current: {step['current']}A, Max-V: {step['voltage']}V")
                    elif step['type'] == 'CCD':
                        st.write(f"Discharge: {step['current']}A")
                    else:
                        st.write("Idle")

        # Cell selection and control
        st.subheader("Cell Selection and Control")

        # Multi-select cells
        cell_options = {}
        for cell_id in selected_process['cells']:
            cell = st.session_state.cell_data[cell_id]
            cell_options[f"Cell {cell['cell_index']} ({cell['type'].upper()})"] = cell_id

        selected_cells = st.multiselect(
            "Select cells to run:",
            options=list(cell_options.keys()),
            default=[]
        )

        selected_cell_ids = [cell_options[cell_name] for cell_name in selected_cells]

        # Control buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("▶️ Start Selected Cells", disabled=len(selected_cell_ids) == 0):
                for cell_id in selected_cell_ids:
                    cell = st.session_state.cell_data[cell_id]
                    if cell['status'] == 'idle':
                        cell['status'] = 'running'
                        cell['start_time'] = datetime.datetime.now()
                        cell['process_step'] = 0
                        cell['step_elapsed_time'] = 0
                        cell['total_elapsed_time'] = 0
                        cell['current_process'] = selected_process['process_steps'][0]['type']
                        st.session_state.running_cells[cell_id] = True
                st.success(f"Started {len(selected_cell_ids)} cells")
                st.rerun()

        with col2:
            all_idle = all(st.session_state.cell_data[cell_id]['status'] == 'idle'
                           for cell_id in selected_process['cells'])
            if st.button("▶️ Start All Cells", disabled=not all_idle):
                for cell_id in selected_process['cells']:
                    cell = st.session_state.cell_data[cell_id]
                    cell['status'] = 'running'
                    cell['start_time'] = datetime.datetime.now()
                    cell['process_step'] = 0
                    cell['step_elapsed_time'] = 0
                    cell['total_elapsed_time'] = 0
                    cell['current_process'] = selected_process['process_steps'][0]['type']
                    st.session_state.running_cells[cell_id] = True
                st.success("Started all cells")
                st.rerun()

        with col3:
            if st.button("🔄 Reset All Cells"):
                for cell_id in selected_process['cells']:
                    cell = st.session_state.cell_data[cell_id]
                    cell['status'] = 'idle'
                    cell['current_process'] = None
                    cell['process_step'] = 0
                    cell['step_elapsed_time'] = 0
                    cell['total_elapsed_time'] = 0
                    cell['history'] = []
                    if cell_id in st.session_state.running_cells:
                        del st.session_state.running_cells[cell_id]
                st.success("Reset all cells")
                st.rerun()

        # Cell dashboard
        st.subheader("Cell Dashboard")

        # Create cell cards
        cols = st.columns(4)
        for i, cell_id in enumerate(selected_process['cells']):
            cell = st.session_state.cell_data[cell_id]
            col_index = i % 4

            with cols[col_index]:
                # Determine card color based on current process
                if cell['status'] == 'running' and cell['current_process']:
                    card_color = PROCESS_COLORS[cell['current_process']]
                elif cell['status'] == 'completed':
                    card_color = '#95A5A6'  # Gray for completed
                else:
                    card_color = '#34495E'  # Dark gray for idle

                # Status indicators
                status_icons = {
                    'idle': '⭕',
                    'running': '🔄',
                    'completed': '✅',
                    'error': '❌'
                }

                current_step_info = ""
                if cell['status'] == 'running' and cell['current_process']:
                    step_num = cell['process_step'] + 1
                    total_steps = len(selected_process['process_steps'])
                    current_step_info = f"Step {step_num}/{total_steps} ({cell['current_process']})"

                st.markdown(f"""
                <div style="
                    border: 3px solid {card_color}; 
                    padding: 15px; 
                    border-radius: 10px; 
                    margin: 5px 0;
                    background-color: rgba({int(card_color[1:3], 16)}, {int(card_color[3:5], 16)}, {int(card_color[5:7], 16)}, 0.1);
                ">
                    <h4>{status_icons.get(cell['status'], '⚪')} Cell {cell['cell_index']}</h4>
                    <p><b>Type:</b> {cell['type'].upper()}</p>
                    <p><b>Status:</b> {cell['status'].title()}</p>
                    <p><b>Voltage:</b> {cell['voltage']:.2f}V</p>
                    <p><b>Current:</b> {cell['current']:.1f}A</p>
                    <p><b>Temperature:</b> {cell['temperature']:.1f}°C</p>
                    <p><b>Total Time:</b> {cell['total_elapsed_time']}s</p>
                    {f"<p><b>Process:</b> {current_step_info}</p>" if current_step_info else ""}
                    {f"<p><b>Step Time:</b> {cell['step_elapsed_time']}s</p>" if cell['status'] == 'running' else ""}
                </div>
                """, unsafe_allow_html=True)

        # Real-time simulation for running cells
        if st.session_state.running_cells:
            for cell_id in list(st.session_state.running_cells.keys()):
                cell = st.session_state.cell_data[cell_id]
                if cell['status'] == 'running':
                    current_step = selected_process['process_steps'][cell['process_step']]
                    simulate_process_step(cell_id, current_step)
                    advance_process_step(cell_id)

        # Statistics and charts
        cells_with_data = [st.session_state.cell_data[cell_id] for cell_id in selected_process['cells']
                           if st.session_state.cell_data[cell_id]['history']]

        if cells_with_data:
            st.subheader("📊 Real-time Statistics")

            # Voltage trends
            voltage_fig = go.Figure()
            for cell in cells_with_data:
                times = [entry['time'] for entry in cell['history']]
                voltages = [entry['voltage'] for entry in cell['history']]
                voltage_fig.add_trace(go.Scatter(
                    x=times, y=voltages,
                    mode='lines',
                    name=f"Cell {cell['cell_index']}",
                    line=dict(color=CELL_SPECS[cell['type']]['color'])
                ))

            voltage_fig.update_layout(
                title="Voltage Trends",
                xaxis_title="Time (seconds)",
                yaxis_title="Voltage (V)",
                height=400
            )
            st.plotly_chart(voltage_fig, use_container_width=True)

            # Current trends
            current_fig = go.Figure()
            for cell in cells_with_data:
                times = [entry['time'] for entry in cell['history']]
                currents = [entry['current'] for entry in cell['history']]
                current_fig.add_trace(go.Scatter(
                    x=times, y=currents,
                    mode='lines',
                    name=f"Cell {cell['cell_index']}"
                ))

            current_fig.update_layout(
                title="Current Trends",
                xaxis_title="Time (seconds)",
                yaxis_title="Current (A)",
                height=400
            )
            st.plotly_chart(current_fig, use_container_width=True)

            # Process type timeline
            timeline_fig = go.Figure()
            for cell in cells_with_data:
                times = [entry['time'] for entry in cell['history']]
                process_types = [entry['process_type'] for entry in cell['history']]

                # Convert process types to numbers for plotting
                process_map = {'CC/CV': 1, 'CCD': 2, 'Idle': 0}
                process_nums = [process_map.get(pt, 0) for pt in process_types]

                timeline_fig.add_trace(go.Scatter(
                    x=times, y=process_nums,
                    mode='lines+markers',
                    name=f"Cell {cell['cell_index']}",
                    yaxis='y'
                ))

            timeline_fig.update_layout(
                title="Process Type Timeline",
                xaxis_title="Time (seconds)",
                yaxis=dict(
                    title="Process Type",
                    tickmode='array',
                    tickvals=[0, 1, 2],
                    ticktext=['Idle', 'CC/CV', 'CCD']
                ),
                height=300
            )
            st.plotly_chart(timeline_fig, use_container_width=True)

        else:
            st.info("Start some cells to view real-time statistics and charts!")

# Auto-refresh for running cells
if st.session_state.running_cells:
    time.sleep(1)
    st.rerun()