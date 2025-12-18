#!/usr/bin/env python3

import math, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from CoolProp.CoolProp import PropsSI
    COOLPROP_OK = True
except Exception:
    COOLPROP_OK = False

ETA_ISENTROPIC = 0.70
CP_AIR_KJ_PER_KG_K = 1.006
RHO_AIR = 1.184
AIRFLOW_CFM = 130.0
T_HPV_F = 53.34
h_HPV_BTU_PER_LB = 184.39
DP_ALLOW_PSI = 7.0

MREF_T_POINTS = np.array([32.0, 40.0, 50.0])
MREF_GPM_POINTS = np.array([0.90, 0.85, 0.73])
P_GC_T = np.array([32.0, 50.0])
P_GC_PSI = np.array([1288.694418, 1331.457765])
P_EVAP_T = np.array([32.0, 50.0])
P_EVAP_PSI = np.array([318.9315, 360.7333163])
POWER_T_POINTS = np.array([32.0, 40.0, 50.0])
POWER_KW_POINTS = np.array([3.100, 2.396, 2.080])
FREQ_AT_32F = 50.0
FREQ_SLOPE = -1.0
CO2_TCRIT_K = 304.1282
OUTPUT_XLSX = "CO2_blackbox_results.xlsx"

def F_to_K(Tf): return (Tf - 32.0) * 5.0/9.0 + 273.15
def K_to_F(Tk): return (Tk - 273.15) * 9.0/5.0 + 32.0
def psi_to_Pa(psi): return psi * 6894.76
def Pa_to_psi(Pa): return Pa / 6894.76
def Jkg_to_Btulb(Jkg): return Jkg * 0.000429922614
def W_to_Btu_hr(W): return W * 3.412142
def GPM_to_m3s(gpm): return gpm * 3.785411784 / 60.0 / 1000.0

def interp_mref_gpm(Tout_F):
    return float(np.interp(Tout_F, MREF_T_POINTS, MREF_GPM_POINTS))

def interp_power_kw(Tout_F):
    return float(np.interp(Tout_F, POWER_T_POINTS, POWER_KW_POINTS))

def interp_p_gc_psi(Tout_F):
    return float(np.interp(Tout_F, P_GC_T, P_GC_PSI))

def interp_p_evap_psi(Tout_F):
    return float(np.interp(Tout_F, P_EVAP_T, P_EVAP_PSI))

def compressor_freq_hz(Tout_F):
    return FREQ_AT_32F + FREQ_SLOPE * (Tout_F - 32.0)

def simulate_case(T_outdoor_F, T_room_F,
                  T_air_GC_in=None,
                  T_GC_out_32=None,
                  T_GC_out_50=None,
                  mass_flow_override_kg_s=None,
                  verbose=False):
    if not COOLPROP_OK:
        raise RuntimeError("CoolProp is required. Install it with: pip install CoolProp")

    T_air_GC_local = T_air_GC_in if T_air_GC_in is not None else 95.926857
    T_GC_out_32_local = T_GC_out_32 if T_GC_out_32 is not None else 97.1394125
    T_GC_out_50_local = T_GC_out_50 if T_GC_out_50 is not None else 94.21192857

    a_evap = (20.49905102 - 9.834415) / (50.0 - 32.0)
    b_evap = 9.834415 - a_evap * 32.0
    T_evap_in_F = a_evap * T_outdoor_F + b_evap

    a_gc = (T_GC_out_50_local - T_GC_out_32_local) / (50.0 - 32.0)
    b_gc = T_GC_out_32_local - a_gc * 32.0
    T_GC_out_F = a_gc * T_outdoor_F + b_gc

    p_gc_psi = interp_p_gc_psi(T_outdoor_F)
    p_evap_psi = interp_p_evap_psi(T_outdoor_F)
    p_gc_Pa = psi_to_Pa(p_gc_psi)
    p_evap_Pa = psi_to_Pa(p_evap_psi)

    if mass_flow_override_kg_s is not None:
        m_dot = float(mass_flow_override_kg_s)
        mref_gpm = None
    else:
        mref_gpm = interp_mref_gpm(T_outdoor_F)
        vol_m3_s = GPM_to_m3s(mref_gpm)
        try:
            rho_co2 = PropsSI('D', 'T', F_to_K(T_GC_out_F), 'P', p_gc_Pa, 'CO2')
            if not np.isfinite(rho_co2) or rho_co2 < 1.0:
                rho_co2 = 800.0
        except Exception:
            rho_co2 = 800.0
        m_dot = max(1e-5, vol_m3_s * rho_co2)

    sh_a = (23.28352041 - 21.67469) / (50.0 - 32.0)
    sh_b = 21.67469 - sh_a * 32.0
    Superheat_F = sh_a * T_outdoor_F + sh_b
    T_comp_in_K = F_to_K(T_evap_in_F + Superheat_F)
    h1_Jkg = PropsSI('H', 'T', T_comp_in_K, 'P', p_evap_Pa, 'CO2')
    s1 = PropsSI('S', 'T', T_comp_in_K, 'P', p_evap_Pa, 'CO2')
    try:
        h2s_Jkg = PropsSI('H', 'P', p_gc_Pa, 'S', s1, 'CO2')
    except Exception:
        h2s_Jkg = PropsSI('H', 'T', F_to_K(T_GC_out_F), 'P', p_gc_Pa, 'CO2')
    h2_Jkg = h1_Jkg + (h2s_Jkg - h1_Jkg) / ETA_ISENTROPIC
    h3_Jkg = PropsSI('H', 'T', F_to_K(T_GC_out_F), 'P', p_gc_Pa, 'CO2')
    h4_Jkg = h3_Jkg

    try:
        T_evap_out_K = PropsSI('T', 'P', p_evap_Pa, 'H', h4_Jkg, 'CO2')
        T_evap_out_F = K_to_F(T_evap_out_K)
    except Exception:
        T_evap_out_F = T_evap_in_F + 7.0

    power_kw = interp_power_kw(T_outdoor_F)
    power_W = power_kw * 1000.0
    Q_gc_W = m_dot * (h2_Jkg - h3_Jkg)
    Q_evap_W = m_dot * (h1_Jkg - h4_Jkg)
    COP = Q_gc_W / power_W if abs(power_W) > 1e-12 else np.nan

    m3s_air = AIRFLOW_CFM * 0.00047194745
    m_dot_air = m3s_air * RHO_AIR
    T_air_in_F = T_air_GC_local
    T_air_out_F = T_GC_out_F - 3.0
    cp_air_J_per_kgK = CP_AIR_KJ_PER_KG_K * 1000.0
    Q_gc_air_W = m_dot_air * cp_air_J_per_kgK * ((T_air_in_F - T_air_out_F) * 5.0/9.0)
    T_evap_air_in_F = T_room_F
    T_evap_air_out_F = T_evap_air_in_F + 10.0
    Q_evap_air_W = m_dot_air * cp_air_J_per_kgK * ((T_evap_air_in_F - T_evap_air_out_F) * 5.0/9.0)

    p_gc_32 = interp_p_gc_psi(32.0)
    p_evap_32 = interp_p_evap_psi(32.0)
    k_tune = 639.7003425 / (0.5 * (p_gc_32 + p_evap_32))
    P_bypass_psi = 0.5 * (p_gc_psi + p_evap_psi) * k_tune

    try:
        T_comp_out_K = PropsSI('T', 'P', p_gc_Pa, 'H', h2_Jkg, 'CO2')
        T_comp_out_F = K_to_F(T_comp_out_K)
    except Exception:
        T_comp_out_F = T_GC_out_F + 120.0

    h1_btu_lb = Jkg_to_Btulb(h1_Jkg)
    h2_btu_lb = Jkg_to_Btulb(h2_Jkg)
    h3_btu_lb = Jkg_to_Btulb(h3_Jkg)
    h4_btu_lb = Jkg_to_Btulb(h4_Jkg)

    out = {
        "T_GC_out": T_GC_out_F,
        "T_evap_in": T_evap_in_F,
        "T_room": T_room_F,
        "h2": h2_btu_lb,
        "T_Comp_out": T_comp_out_F,
        "Compression Ratio": (p_gc_Pa / p_evap_Pa) if p_evap_Pa>0 else np.nan,
        "P_GC_in": p_gc_psi,
        "P_GC_out": p_gc_psi - np.random.uniform(-DP_ALLOW_PSI, DP_ALLOW_PSI),
        "Subcooling": (h2_Jkg - h3_Jkg) / 1000.0,
        "h3": h3_btu_lb,
        "Heating Capacity": W_to_Btu_hr(Q_gc_W),
        "COP": COP,
        "Power": power_W / 1000.0,
        "M_ref_(GPM)": mref_gpm if mass_flow_override_kg_s is None else (mass_flow_override_kg_s),
        "Airflow_GC_(CFM)": AIRFLOW_CFM,
        "T_air_GC": T_air_in_F,
        "GC Capacity_air side": W_to_Btu_hr(Q_gc_air_W),
        "T_evap_out": T_evap_out_F,
        "Superheat": Superheat_F,
        "Cooling Capacity": W_to_Btu_hr(Q_evap_W),
        "P_evap_out": p_evap_psi - np.random.uniform(-DP_ALLOW_PSI, DP_ALLOW_PSI),
        "h1": h1_btu_lb,
        "P_evap_in": p_evap_psi,
        "Evap_capacity_air side": abs(W_to_Btu_hr(Q_evap_air_W)),
        "Airflow_evap_CFM": AIRFLOW_CFM,
        "P_bypass": P_bypass_psi,
        "h4": h4_btu_lb,
        "T_GC_in": T_GC_out_F + 113.0,
        "T_HPV": T_HPV_F,
        "h_HPV": h_HPV_BTU_PER_LB,
        "outdoor temp": T_outdoor_F,
        "_m_dot_kg_s": m_dot,
        "_power_W": power_W,
        "_compressor_freq_Hz": compressor_freq_hz(T_outdoor_F)
    }

    return out

def ask_float(prompt, default):
    try:
        val = input(f"{prompt} [{default}]: ").strip()
        if val == "":
            return float(default)
        return float(val)
    except Exception:
        print("Invalid input, using default:", default)
        return float(default)

def main():
    if not COOLPROP_OK:
        print("CoolProp is not installed or import failed. Please run: pip install CoolProp")
        sys.exit(1)

    print("=== CO2 hybrid black-box model (corrected) with adjustable GC ===")
    Tout = ask_float("Enter outdoor temperature (°F)", 32.0)
    Troom = ask_float("Enter room temperature (°F)", 75.0)

    T_air_gc_new = ask_float("Enter GC air inlet temperature (°F)", 95.926857)
    T_gc_out_32_new = ask_float("Enter GC outlet temp at 32°F outdoor (°F)", 97.1394125)
    T_gc_out_50_new = ask_float("Enter GC outlet temp at 50°F outdoor (°F)", 94.21192857)

    out = simulate_case(Tout, Troom,
                        T_air_GC_in=T_air_gc_new,
                        T_GC_out_32=T_gc_out_32_new,
                        T_GC_out_50=T_gc_out_50_new,
                        mass_flow_override_kg_s=None,
                        verbose=True)
    df_single = pd.DataFrame([out])

    cols_order = ["T_GC_out","T_evap_in","T_room","h2","T_Comp_out","Compression Ratio","P_GC_in","P_GC_out",
                  "Subcooling","h3","Heating Capacity","COP","Power","M_ref_(GPM)","Airflow_GC_(CFM)","T_air_GC",
                  "GC Capacity_air side","T_evap_out","Superheat","Cooling Capacity","P_evap_out","h1","P_evap_in",
                  "Evap_capacity_air side","Airflow_evap_CFM","P_bypass","h4","T_GC_in","T_HPV","h_HPV","outdoor temp"]

    for k in cols_order:
        if k not in df_single.columns:
            df_single[k] = np.nan

    print("\nSingle-case prediction (rounded):")
    pd.set_option('display.float_format', '{:.6f}'.format)
    print(df_single[cols_order].T)

    touts = np.arange(0, 101, 1.0)
    cop_list = []
    records = []
    for t in touts:
        try:
            r = simulate_case(t, Troom,
                              T_air_GC_in=T_air_gc_new,
                              T_GC_out_32=T_gc_out_32_new,
                              T_GC_out_50=T_gc_out_50_new)
            cop_list.append(r["COP"] if r["COP"] is not None else np.nan)
            records.append(r)
        except Exception:
            cop_list.append(np.nan)
            records.append({})

    cop_df = pd.DataFrame({"Outdoor_Temp_F": touts, "Predicted_COP": cop_list})

    with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
        df_single.to_excel(writer, sheet_name="SinglePrediction", index=False)
        cop_df.to_excel(writer, sheet_name="COP_vs_Tout", index=False)
        diag = pd.DataFrame([{"_m_dot_kg_s": out["_m_dot_kg_s"], "_power_W": out["_power_W"], "_freq_Hz": out["_compressor_freq_Hz"]}])
        diag.to_excel(writer, sheet_name="Diagnostics", index=False)
        ws = writer.sheets["SinglePrediction"]
        for i, col in enumerate(df_single.columns):
            ws.set_column(i, i, max(10, min(30, len(col) + 2)))

    print(f"\nSaved results to: {OUTPUT_XLSX}")

    plt.figure(figsize=(8,4))
    plt.plot(touts, cop_list, marker='o', markersize=4)
    plt.xlabel("Outdoor Temperature (°F)")
    plt.ylabel("Predicted COP (heating)")
    plt.title(f"COP vs Outdoor Temp (room fixed at {Troom} °F)")
    plt.grid(True)
    plt.ylim(bottom=0)

    exp_temps = [32.0, 50.0]
    exp_cops = [2.807, 4.4797]
    plt.plot(exp_temps, exp_cops, 's-', color='red', markersize=6, label='Experimental anchors')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
