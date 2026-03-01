# optimizer_logic.py

def calculate_roi(line_id, current_loss_mw, line_length_km, peak_load_amps):
    """
    Evaluates the best financial intervention to bring losses < 2%.
    """
    # --- 1. Base Financial Assumptions ---
    ENERGY_COST_PER_MWH = 50       # $50 per MWh
    HOURS_PER_YEAR = 8760
    PROJECT_LIFESPAN_YEARS = 10
    
    # Cost of doing nothing (Wasted Energy)
    annual_wasted_energy_mwh = current_loss_mw * HOURS_PER_YEAR
    annual_loss_cost = annual_wasted_energy_mwh * ENERGY_COST_PER_MWH
    ten_year_loss_cost = annual_loss_cost * PROJECT_LIFESPAN_YEARS

    # --- 2. Intervention A: HTLS Conductor Upgrade ---
    # Upgrading the physical wire reduces resistance by ~30%
    htls_cost_per_km = 2000000     # $2M per km
    htls_capex = htls_cost_per_km * line_length_km
    
    # HTLS reduces losses by roughly 30%
    htls_new_loss_mw = current_loss_mw * 0.70 
    htls_annual_savings = (current_loss_mw - htls_new_loss_mw) * HOURS_PER_YEAR * ENERGY_COST_PER_MWH
    htls_roi_10yr = (htls_annual_savings * PROJECT_LIFESPAN_YEARS) - htls_capex
    htls_payback = htls_capex / htls_annual_savings if htls_annual_savings > 0 else 999

    # --- 3. Intervention B: BESS (Battery Energy Storage System) ---
    # Batteries don't fix the wire, they "shave" the peak load (which we know causes 96.8% of losses)
    # A 50MW battery costs about $15M
    bess_capex = 15000000 
    
    # By smoothing the load, BESS reduces peak I^2R losses drastically (up to 45%)
    bess_new_loss_mw = current_loss_mw * 0.55
    bess_annual_savings = (current_loss_mw - bess_new_loss_mw) * HOURS_PER_YEAR * ENERGY_COST_PER_MWH
    bess_roi_10yr = (bess_annual_savings * PROJECT_LIFESPAN_YEARS) - bess_capex
    bess_payback = bess_capex / bess_annual_savings if bess_annual_savings > 0 else 999

    # --- 4. The Decision Engine ---
    if bess_roi_10yr > htls_roi_10yr and bess_roi_10yr > 0:
        recommendation = "Battery Storage (BESS)"
        best_capex = bess_capex
        best_savings = bess_annual_savings
        payback = bess_payback
        new_loss = bess_new_loss_mw
    elif htls_roi_10yr > bess_roi_10yr and htls_roi_10yr > 0:
        recommendation = "HTLS Conductor Upgrade"
        best_capex = htls_capex
        best_savings = htls_annual_savings
        payback = htls_payback
        new_loss = htls_new_loss_mw
    else:
        recommendation = "Do Nothing (Not Financially Viable)"
        best_capex = 0
        best_savings = 0
        payback = 0
        new_loss = current_loss_mw

    # Return the data as a dictionary so the Streamlit UI can easily read it
    return {
        "Line_ID": line_id,
        "Current_Loss_MW": round(current_loss_mw, 2),
        "Ten_Year_Wasted_Money": f"${ten_year_loss_cost:,.0f}",
        "Recommended_Action": recommendation,
        "Estimated_CapEx": f"${best_capex:,.0f}",
        "Annual_Savings": f"${best_savings:,.0f}",
        "Payback_Period_Years": round(payback, 1),
        "Estimated_New_Loss_MW": round(new_loss, 2)
    }

# Quick test to make sure it works
if __name__ == "__main__":
    test_result = calculate_roi("Line_4", current_loss_mw=3.5, line_length_km=12.5, peak_load_amps=450)
    for key, value in test_result.items():
        print(f"{key}: {value}")