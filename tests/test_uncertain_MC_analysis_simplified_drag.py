import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd # Added for CSV output

# --- Define Nominal Values and Distribution Bounds for Uncertain Inputs ---
# CDmin
CDmin_nominal = 0.0333
CDmin_lower_bound = CDmin_nominal - 0.002
CDmin_upper_bound = CDmin_nominal + 0.002

# Temperature Offset
temp_offset_nominal = 0.0
temp_offset_lower_bound = -30.0
temp_offset_upper_bound = 30.0

# Cruise Speed
cruisespeed_nominal_kts = 108
cruisespeed_nominal_mps = cruisespeed_nominal_kts * 0.514444
cruisespeed_lower_bound_mps = 0.8 * cruisespeed_nominal_mps
cruisespeed_upper_bound_mps = 1.2 * cruisespeed_nominal_mps

# Monte Carlo and S_sqm settings
N_MC_ITERATIONS = 10000
N_S_VALUES = 5
S_values_sqm = np.linspace(10, 100, N_S_VALUES) # 20 wing area values

# Shared fixed parameters
REF_WEIGHT_N = 1320 * 4.44822
AR_const = 8.0
CLMAX_const = 1.756
CLMIN_D_const = 0.20

# For x-axis plotting
wing_loading_values = REF_WEIGHT_N / S_values_sqm

# --- Helper function to create input dictionaries ---
def get_inputs_for_run(cd_val, temp_offset_val, cruisespeed_val):
    """
    Generates input dictionaries for all constraints based on provided input values.
    These values can be nominal or sampled.
    """
    inputs_climb = {
        "weight_n": REF_WEIGHT_N,
        "climbalt_m": 0.0,
        "climbspeed_mps": 66 * 0.514444, # This specific climb speed is kept constant.
        "climbrate_mps": 880 * 0.00508,
        "temp_offset": temp_offset_val,
        "CDmin": cd_val,
        "CLmax": CLMAX_const,
        "CLminD": CLMIN_D_const,
        "AR": AR_const
    }

    inputs_takeoff = {
        "groundrun_m": 640 * 0.3048,
        "rwyelevation_m": 0.0,
        "temp_offset": temp_offset_val,
        "weight_n": REF_WEIGHT_N,
        "AR": AR_const,
        "CDmin": cd_val,
        "mu_R": 0.04,
        "CLTO": 0.5,
        "CDTO": 0.038,
        "CLmax": CLMAX_const
    }

    inputs_turn = {
        "cruise_alt_m": 8000 * 0.3048,
        "temp_offset": temp_offset_val,
        "cruisespeed_mps": cruisespeed_val,
        "AR": AR_const,
        "weight_n": REF_WEIGHT_N,
        "CDmin": cd_val,
        "n": 1.155 # Load factor for turn
    }

    inputs_cruise = {
        "cruise_alt_m": 8000 * 0.3048,
        "temp_offset": temp_offset_val,
        "cruisespeed_mps": cruisespeed_val,
        "AR": AR_const,
        "weight_n": REF_WEIGHT_N,
        "CDmin": cd_val,
        "n": 1.0 # Load factor for steady level cruise
    }

    inputs_service = {
        "weight_n": REF_WEIGHT_N,
        "servalt_m": 17000 * 0.3048,
        "temp_offset": temp_offset_val,
        "CDmin": cd_val,
        "CLmax": CLMAX_const,
        "CLminD": CLMIN_D_const,
        "AR": AR_const
    }
    return inputs_climb, inputs_takeoff, inputs_turn, inputs_cruise, inputs_service

# --- Constraint Functions (Simplified: No UN checks, AR handling direct) ---
_T0 = 288.15; _LR = -0.0065; _I1 = 1.048840; _J1 = -23.659414e-6; _L1 = 4.2558797
_RHO_SL_KGPM3 = _I1**_L1

def calculate_actual_temp_and_density(alt_m, temp_offset_val):
    T_isa_alt = _T0 + _LR * alt_m
    T_actual_alt = T_isa_alt + temp_offset_val
    if T_actual_alt <= 0:
        raise ValueError(f"Actual temperature at altitude {alt_m}m is non-positive: {T_actual_alt}K.")

    rho_isa_alt = (_I1 + _J1 * alt_m)**_L1
    if abs(T_actual_alt) < 1e-6 :
        raise ValueError(f"Calculated T_actual_alt is near zero ({T_actual_alt}K), leading to division error.")
    if T_isa_alt == 0 and T_actual_alt == 0 :
         rho_actual_alt = rho_isa_alt
    else:
         rho_actual_alt = rho_isa_alt * (T_isa_alt / T_actual_alt)
    return T_actual_alt, rho_actual_alt

def climb_constraint_v0(S_sqm: float, weight_n: float, climbalt_m: float, climbspeed_mps: float,
                        climbrate_mps: float, temp_offset: float, CDmin: float, CLmax: float,
                        CLminD: float, AR: float) -> tuple:
    _Level1 = 11000.0
    tw, pw = np.nan, np.nan
    try:
        if climbalt_m >= _Level1:
            raise ValueError(f"Climb altitude {climbalt_m}m >= 11000m (Tropopause)")
        _, rho_actual_alt = calculate_actual_temp_and_density(climbalt_m, temp_offset)
        climbspeed_mps_eas = (43.591 + 2.2452 * (weight_n / S_sqm) * 0.0208854) * 0.514444
        climbspeed_mps_tas_calc = climbspeed_mps
        q_pa = 0.5 * _RHO_SL_KGPM3 * (climbspeed_mps_eas)**2
        AR_pow = AR**0.68
        epsilon = 1.78 * (1.0 - 0.045 * AR_pow) - 0.64
        k = 1.0 / (np.pi * AR * epsilon)
        tw = climbrate_mps * climbspeed_mps_tas_calc**(-1.0) + \
             q_pa * CDmin * (weight_n / S_sqm)**(-1.0) + \
             k * (weight_n / S_sqm) * q_pa**(-1.0)
        pw = tw * weight_n
    except Exception:
        tw, pw = np.nan, np.nan
    return tw, pw

def takeoff_constraint_v0(S_sqm: float, groundrun_m: float, rwyelevation_m: float, temp_offset: float,
                          weight_n: float, AR: float, CDmin: float, mu_R: float, CLTO: float,
                          CDTO: float, CLmax: float) -> tuple:
    g = 9.80665
    _Level1 = 11000.0
    tw, pw = np.nan, np.nan
    try:
        if rwyelevation_m >= _Level1:
            raise ValueError(f"Runway elevation {rwyelevation_m}m >= 11000m")
        _, rho_actual_rwy = calculate_actual_temp_and_density(rwyelevation_m, temp_offset)
        tw = (1.21 * (weight_n / S_sqm)) * (g * rho_actual_rwy * CLmax * groundrun_m)**(-1.0) + \
             (0.605 * (CDTO - mu_R * CLTO)) * CLmax**(-1.0) + mu_R
        pw = tw * weight_n
    except Exception:
        tw, pw = np.nan, np.nan
    return tw, pw

def turn_constraint_v0(S_sqm: float, cruise_alt_m: float, temp_offset: float, cruisespeed_mps: float,
                       AR: float, weight_n: float, CDmin: float, n: float) -> tuple:
    _Level1 = 11000.0
    tw, pw = np.nan, np.nan
    try:
        if cruise_alt_m >= _Level1:
            raise ValueError(f"Altitude {cruise_alt_m}m >= 11000m")
        _, rho_cruise_kgpm3 = calculate_actual_temp_and_density(cruise_alt_m, temp_offset)
        q_pa = 0.5 * rho_cruise_kgpm3 * cruisespeed_mps**2.0
        AR_pow = AR**0.68
        epsilon = 1.78 * (1.0 - 0.045 * AR_pow) - 0.64
        k = 1.0 / (np.pi * AR * epsilon)
        tw = q_pa * CDmin * (weight_n / S_sqm)**(-1.0) + \
             k * n**2.0 * (weight_n / S_sqm) * q_pa**(-1.0)
        pw = q_pa * (CDmin * S_sqm + k * (n / q_pa)**2.0 * (weight_n**2.0 / S_sqm))
    except Exception:
        tw, pw = np.nan, np.nan
    return tw, pw

def cruise_constraint_v0(S_sqm: float, cruise_alt_m: float, temp_offset: float, cruisespeed_mps: float,
                         AR: float, weight_n: float, CDmin: float, n: float) -> tuple:
    _Level1 = 11000.0
    tw, pw = np.nan, np.nan
    try:
        if cruise_alt_m >= _Level1:
            raise ValueError(f"Altitude {cruise_alt_m}m >= 11000m")
        _, rho_cruise_kgpm3 = calculate_actual_temp_and_density(cruise_alt_m, temp_offset)
        q_pa = 0.5 * rho_cruise_kgpm3 * cruisespeed_mps**2.0
        AR_pow = AR**0.68
        epsilon = 1.78 * (1.0 - 0.045 * AR_pow) - 0.64
        k = 1.0 / (np.pi * AR * epsilon)
        tw = q_pa * CDmin * (weight_n / S_sqm)**(-1.0) + \
             k * (weight_n / S_sqm) * q_pa**(-1.0) # n=1
        pw = q_pa * (CDmin * S_sqm + k * q_pa**(-1.0) * (weight_n**2.0 / S_sqm))
    except Exception:
        tw, pw = np.nan, np.nan
    return tw, pw

def service_constraint_v0(S_sqm: float, weight_n: float, servalt_m: float, temp_offset: float,
                          CDmin: float, CLmax: float, CLminD: float, AR: float) -> tuple:
    _Level1 = 11000.0
    tw, pw = np.nan, np.nan
    try:
        if servalt_m >= _Level1:
            raise ValueError(f"Service altitude {servalt_m}m >= 11000m")
        _, rho_actual_alt = calculate_actual_temp_and_density(servalt_m, temp_offset)
        if rho_actual_alt <= 1e-3:
            raise ValueError(f"Calculated density rho_actual_alt is too low ({rho_actual_alt}) at alt {servalt_m}m.")
        servspeed_mps_eas = (43.591 + 2.2452 * (weight_n / S_sqm) * 0.0208854) * 0.514444
        servspeed_mps_tas = servspeed_mps_eas * (_RHO_SL_KGPM3 / rho_actual_alt)**0.5
        if servspeed_mps_tas <= 1e-3 :
            raise ValueError(f"Calculated servspeed_mps_tas is too low ({servspeed_mps_tas}).")
        q_pa_eas = 0.5 * _RHO_SL_KGPM3 * (servspeed_mps_eas)**2.0
        AR_pow = AR**0.68
        epsilon = 1.78 * (1.0 - 0.045 * AR_pow) - 0.64
        k = 1.0 / (np.pi * AR * epsilon)
        servrate_mps_vert = 0.508 # 100 fpm
        tw = servrate_mps_vert * servspeed_mps_tas**(-1.0) + \
             q_pa_eas * CDmin * (weight_n / S_sqm)**(-1.0) + \
             k * (weight_n / S_sqm) * q_pa_eas**(-1.0)
        pw = tw * weight_n
    except Exception:
        tw, pw = np.nan, np.nan
    return tw, pw

# --- Data Storage for CSV and Plotting ---
constraint_names = ["Climb", "Takeoff", "Turn", "Cruise", "Service"]
# For CSV
csv_data = {'Wing_Loading_WS': wing_loading_values}
for name in constraint_names:
    csv_data[f'{name}_TW_BestEstimate'] = []
    csv_data[f'{name}_TW_MC_LowerCI'] = []
    csv_data[f'{name}_TW_MC_UpperCI'] = []

# For plotting (same data, just showing it can be used directly if not saving/loading)
results_best_estimate_plot = {name: [] for name in constraint_names}
results_lower_ci_plot = {name: [] for name in constraint_names}
results_upper_ci_plot = {name: [] for name in constraint_names}


# --- Main Calculation Loop ---
print(f"Starting Monte Carlo simulation with {N_MC_ITERATIONS} iterations for each of {N_S_VALUES} S_sqm values.")

for i_s, S_current in enumerate(S_values_sqm):
    if (i_s + 1) % (N_S_VALUES // 5 if N_S_VALUES >=5 else 1) == 0 :
        print(f"  Processing S_sqm value {i_s+1}/{N_S_VALUES} ({S_current:.2f} m^2)...")

    # --- Best Estimate Calculation (using nominal values) ---
    inputs_c_nom, inputs_to_nom, inputs_tu_nom, inputs_cr_nom, inputs_s_nom = \
        get_inputs_for_run(CDmin_nominal, temp_offset_nominal, cruisespeed_nominal_mps)

    be_climb = climb_constraint_v0(S_current, **inputs_c_nom)[0]
    be_takeoff = takeoff_constraint_v0(S_current, **inputs_to_nom)[0]
    be_turn = turn_constraint_v0(S_current, **inputs_tu_nom)[0]
    be_cruise = cruise_constraint_v0(S_current, **inputs_cr_nom)[0]
    be_service = service_constraint_v0(S_current, **inputs_s_nom)[0]

    csv_data["Climb_TW_BestEstimate"].append(be_climb)
    csv_data["Takeoff_TW_BestEstimate"].append(be_takeoff)
    csv_data["Turn_TW_BestEstimate"].append(be_turn)
    csv_data["Cruise_TW_BestEstimate"].append(be_cruise)
    csv_data["Service_TW_BestEstimate"].append(be_service)

    results_best_estimate_plot["Climb"].append(be_climb)
    results_best_estimate_plot["Takeoff"].append(be_takeoff)
    results_best_estimate_plot["Turn"].append(be_turn)
    results_best_estimate_plot["Cruise"].append(be_cruise)
    results_best_estimate_plot["Service"].append(be_service)

    # --- Monte Carlo Simulation for current S_sqm ---
    mc_samples = {name: [] for name in constraint_names}

    for _ in range(N_MC_ITERATIONS):
        cd_sample = np.random.uniform(CDmin_lower_bound, CDmin_upper_bound)
        temp_offset_sample = np.random.normal(loc=0.0, scale=10.0) # Mean (loc) = 0, StDev (scale) = 10 #np.random.uniform(temp_offset_lower_bound, temp_offset_upper_bound)
        cruisespeed_sample = np.random.uniform(cruisespeed_lower_bound_mps, cruisespeed_upper_bound_mps)

        inputs_c, inputs_to, inputs_tu, inputs_cr, inputs_s = \
            get_inputs_for_run(cd_sample, temp_offset_sample, cruisespeed_sample)

        mc_samples["Climb"].append(climb_constraint_v0(S_current, **inputs_c)[0])
        mc_samples["Takeoff"].append(takeoff_constraint_v0(S_current, **inputs_to)[0])
        mc_samples["Turn"].append(turn_constraint_v0(S_current, **inputs_tu)[0])
        mc_samples["Cruise"].append(cruise_constraint_v0(S_current, **inputs_cr)[0])
        mc_samples["Service"].append(service_constraint_v0(S_current, **inputs_s)[0])

    # Calculate 95% Confidence Intervals (2.5th and 97.5th percentiles)
    for name in constraint_names:
        valid_samples = [s for s in mc_samples[name] if not np.isnan(s)]
        if len(valid_samples) < N_MC_ITERATIONS * 0.8: # Check if too many NaNs
              print(f"    Warning: High NaN count for {name} at S={S_current:.2f} ({len(valid_samples)}/{N_MC_ITERATIONS} valid). CI might be unreliable.")
        if len(valid_samples) > 1: # Need at least 2 points for percentile
            lower_ci = np.percentile(valid_samples, 2.5)
            upper_ci = np.percentile(valid_samples, 97.5)
            csv_data[f'{name}_TW_MC_LowerCI'].append(lower_ci)
            csv_data[f'{name}_TW_MC_UpperCI'].append(upper_ci)
            results_lower_ci_plot[name].append(lower_ci)
            results_upper_ci_plot[name].append(upper_ci)
        else: # Not enough valid samples
            csv_data[f'{name}_TW_MC_LowerCI'].append(np.nan)
            csv_data[f'{name}_TW_MC_UpperCI'].append(np.nan)
            results_lower_ci_plot[name].append(np.nan)
            results_upper_ci_plot[name].append(np.nan)

print("\nConstraint calculations complete.")

# --- Save to CSV ---
df = pd.DataFrame(csv_data)
csv_file_path = 'monte_carlo_results.csv'
try:
    df.to_csv(csv_file_path, index=False, float_format='%.6f')
    print(f"Monte Carlo results saved to {csv_file_path}")
except Exception as e:
    print(f"Error saving CSV file: {e}")


# --- Plotting (copied from your provided script for verification) ---
# This can be moved to a separate script that reads
