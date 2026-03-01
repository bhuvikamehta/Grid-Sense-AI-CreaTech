import pandapower as pp
import pandapower.networks as nw
import pandas as pd
import numpy as np
import sys  # Added for the progress bar
from datetime import datetime, timedelta

def generate_enterprise_grid_data(days=365): 
    print("Initializing Triple-Threat Grid Simulation...")
    net = nw.case14()
    base_r_ohm = net.line['r_ohm_per_km'].copy()
    base_load_mw = net.load['p_mw'].copy()
    
    start_time = datetime(2023, 1, 1, 0, 0)
    intervals = days * 24 * 4 
    all_data = []
    
    for i in range(intervals):
        # --- PROGRESS TRACKER ---
        # Update the terminal every 1% of completion
        if i % max(1, intervals // 100) == 0:
            progress = (i / intervals) * 100
            sys.stdout.write(f"\r⚡ Simulating Grid Physics: [{int(progress)}%] Complete")
            sys.stdout.flush()
            
        current_time = start_time + timedelta(minutes=15 * i)
        hour = current_time.hour
        day_of_year = current_time.timetuple().tm_yday
        
        # Weather & Load
        ambient_temp = (25 + 10 * np.sin(2 * np.pi * (day_of_year - 90) / 365)) + (5 * np.sin(2 * np.pi * (hour - 8) / 24)) + np.random.normal(0, 1)
        load_multiplier = 1.0 + 0.3 * np.sin(2 * np.pi * (hour - 6) / 24) + 0.2 * np.sin(2 * np.pi * (hour - 18) / 24)
        
        net.load['p_mw'] = base_load_mw * load_multiplier
        net.line['r_ohm_per_km'] = base_r_ohm * (1 + 0.004 * (ambient_temp - 20))
        
        try:
            pp.runpp(net, enforce_q_lims=True)
        except Exception:
            continue
            
        for line_idx in net.line.index:
            p_from_mw = abs(net.res_line.at[line_idx, 'p_from_mw'])
            to_bus = net.line.at[line_idx, 'to_bus']
            
            # Pillar 1: Technical & DLR
            technical_loss_mw = net.res_line.at[line_idx, 'pl_mw']
            dlr_ampacity = 500.0 * (1 - 0.005 * (ambient_temp - 25))
            
            # Pillar 2: Commercial (Theft)
            power_received_mw = p_from_mw - technical_loss_mw
            theft_pct = np.random.uniform(0.10, 0.25) if to_bus in [3, 4] else np.random.uniform(0.01, 0.03)
            commercial_loss_mw = power_received_mw * theft_pct
            
            # Pillar 3: Stability (Voltage Drop)
            voltage_pu = net.res_bus.at[to_bus, 'vm_pu']
            stability_warn = 1 if (voltage_pu < 0.95 or voltage_pu > 1.05) else 0

            all_data.append({
                'Timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Line_ID': f"Line_{line_idx}",
                'Sending_Bus': net.line.at[line_idx, 'from_bus'],
                'Receiving_Bus': to_bus,
                'Load_Amps': round(net.res_line.at[line_idx, 'i_ka'] * 1000, 2),
                'DLR_Ampacity_Limit': round(dlr_ampacity, 2),
                'Ambient_Temp': round(ambient_temp, 2),
                'Technical_Loss_MW': round(technical_loss_mw, 4),
                'Commercial_Loss_MW': round(commercial_loss_mw, 4),
                'Receiving_Voltage_PU': round(voltage_pu, 4),
                'Stability_Warning': stability_warn
            })

    # Clear the progress line and print success
    sys.stdout.write("\r⚡ Simulating Grid Physics: [100%] Complete\n")
    pd.DataFrame(all_data).to_csv('historical_grid_data_v2.csv', index=False)
    print("Success! Data saved to historical_grid_data_v2.csv")

if __name__ == "__main__":
    generate_enterprise_grid_data()