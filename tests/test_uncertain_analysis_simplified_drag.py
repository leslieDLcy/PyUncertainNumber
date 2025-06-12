import numpy as np
import math # Kept for compatibility if any other math functions are essential and not covered by np
import matplotlib.pyplot as plt
from pyuncertainnumber import UncertainNumber as UN
import pandas as pd # Added for CSV output

# --- Define Parameters ---
# Constant (Best Estimate) Parameters
CDmin_const = 0.0333
temp_offset_const = 0.0
cruisespeed_mps_const = 108 * 0.514444

# Uncertain Parameters
CDmin_un = UN(essence='interval', bounds=[0.0333 - 0.002, 0.0333 + 0.002])
temp_offset_un = UN(essence='interval', bounds=[-30.0, 30.0]) # Ensure float for bounds
cruisespeed_mps_un = UN(essence='interval', bounds=[0.8 * 108 * 0.514444, 1.2 * 108 * 0.514444])

# --- Helper function to create input dictionaries ---
def get_inputs(param_type="constant"):
    if param_type == "constant":
        current_CDmin = CDmin_const
        current_temp_offset = temp_offset_const
        current_cruisespeed_mps = cruisespeed_mps_const
    else: # "uncertain"
        current_CDmin = CDmin_un
        current_temp_offset = temp_offset_un
        current_cruisespeed_mps = cruisespeed_mps_un

    inputs_climb = {
        "weight_n": 1320 * 4.44822,
        "climbalt_m": 0.0, # Ensure float
        "climbspeed_mps": 66 * 0.514444, # Assuming this remains constant, or make it uncertain too
        "climbrate_mps": 880 * 0.00508,
        "temp_offset": current_temp_offset,
        "CDmin": current_CDmin,
        "CLmax": 1.756,
        "CLminD": 0.20,
        "AR": 8.0 # Ensure float
    }

    inputs_takeoff = {
        "groundrun_m": 640 * 0.3048,
        "rwyelevation_m": 0.0,
        "temp_offset": current_temp_offset,
        "weight_n": 1320 * 4.44822,
        "AR": 8.0,
        "CDmin": current_CDmin,
        "mu_R": 0.04,
        "CLTO": 0.5,
        "CDTO": 0.038, # Takeoff specific drag, could also be UN(CDmin_base + delta_TO)
        "CLmax": 1.756
    }

    inputs_turn = {
        "cruise_alt_m": 8000 * 0.3048,
        "temp_offset": current_temp_offset,
        "cruisespeed_mps": current_cruisespeed_mps,
        "AR": 8.0,
        "weight_n": 1320 * 4.44822,
        "CDmin": current_CDmin,
        "n": 1.155
    }

    inputs_cruise = { # Same as turn for this setup, often n=1 for level cruise
        "cruise_alt_m": 8000 * 0.3048,
        "temp_offset": current_temp_offset,
        "cruisespeed_mps": current_cruisespeed_mps,
        "AR": 8.0,
        "weight_n": 1320 * 4.44822,
        "CDmin": current_CDmin,
        "n": 1.0 # Typically n=1 for steady level cruise T/W
    }

    inputs_service = {
        "weight_n": 1320*4.44822,
        "servalt_m": 17000 * 0.3048,
        "temp_offset": current_temp_offset,
        "CDmin": current_CDmin,
        "CLmax": 1.756,
        "CLminD": 0.20,
        "AR": 8.0
    }
    return inputs_climb, inputs_takeoff, inputs_turn, inputs_cruise, inputs_service

## Climb ######
def climb_constraint_v0(
    # --- Inputs ---
    S_sqm: float,            # Wing Area [m^2]. Scalar.
    weight_n: float,         # MTOW [N]. Scalar.
    climbalt_m: float,       # Climb altitude [m]. Scalar.
    climbspeed_mps: float,   # Climb speed from inputs_climb dictionary
    climbrate_mps: float,    # Climb rate [fpm]. Scalar.
    temp_offset: float,      # ISA temperature offset [K or C]. Scalar.
    CDmin: float,            # Minimum drag coefficient [-]. Scalar.
    CLmax: float,            # Max lift coefficient [-]. Scalar.
    CLminD: float,           # Lift coefficient at minimum drag [-]. Scalar.
    AR: float
)-> tuple:

    # --- Constants ---
    R_JPKGPK = 287.05287 # Specific Gas Constant for Air [J/(kg*K)]
    GAMMA_DRY_AIR = 1.4  # Ratio of specific heats for air
    T0 = 288.15           # ISA Sea Level Temperature [K]
    LR = -0.0065          # ISA Temperature Lapse Rate in Troposphere [K/m]
    I1 = 1.048840         # ISA Density Formula Constant (Troposphere)
    J1 = -23.659414e-6    # ISA Density Formula Constant (Troposphere)
    L1 = 4.2558797        # ISA Density Formula Constant (Troposphere)
    rho_sl_kgpm3 = I1**L1 # ISA Sea Level Density [kg/m^3] (Approx 1.225)
    _Level1 = 11000.0     # Altitude limit for troposphere [m] ensure float

    # --- Initialize outputs ---
    tw, pw = np.nan, np.nan
    try:
        # if isinstance(climbalt_m, UN) or climbalt_m >= _Level1:
        #     if not isinstance(climbalt_m, UN) and climbalt_m >= _Level1 :
        #        raise ValueError(f"Climb altitude {climbalt_m}m >= 11000m (Tropopause)")

    #     T_isa_alt = T0 + LR * climbalt_m
    #     rho_isa_alt = (I1 + J1 * climbalt_m)**L1
    #     rho_actual_alt = rho_isa_alt * (1.0 + temp_offset / T_isa_alt)**(-1.0)

        climbspeed_mps_eas = (43.591 + 2.2452*(weight_n/S_sqm)* 0.0208854)* 0.514444
        climbspeed_mps_tas_calc = climbspeed_mps # Use the input climbspeed_mps from dictionary

        q_pa = 0.5 * rho_sl_kgpm3 * (climbspeed_mps_eas)**2

        # Oswald efficiency factor calculation - simplified as AR is float
        AR_pow = AR**0.68
        epsilon = 1.78 * (1.0 - 0.045 * AR_pow) - 0.64

        k = 1.0 / (np.pi * AR * epsilon)

        climbrate_mps_vert = climbrate_mps
        tw = climbrate_mps_vert * climbspeed_mps_tas_calc**(-1.0) + \
             q_pa * CDmin * (weight_n/S_sqm)**(-1.0) + \
             k * (weight_n/S_sqm) * q_pa**(-1.0)

        pw = tw * weight_n

    except ValueError as e:
        print(f"Input or Calculation Error in climb_constraint_v0 for S={S_sqm:.2f}: {e}")
        tw, pw = np.nan, np.nan
    except Exception as e:
        print(f"Unexpected Error in climb_constraint_v0 for S={S_sqm:.2f}: {e}")
        tw, pw = np.nan, np.nan

    return tw,pw

## takeoff #####
def takeoff_constraint_v0(S_sqm: float,
    groundrun_m: float,
    rwyelevation_m: float,
    temp_offset: float,
    weight_n: float,
    AR: float, # AR is an input but not used in epsilon calculation here
    CDmin: float,
    mu_R: float,
    CLTO: float,
    CDTO: float,
    CLmax: float
)-> tuple:

    R_JPKGPK = 287.05287
    GAMMA_DRY_AIR = 1.4
    T0 = 288.15
    LR = -0.0065
    I1 = 1.048840
    J1 = -23.659414e-6
    L1 = 4.2558797
    rho_sl_kgpm3 = I1**L1
    g = 9.80665
    _Level1 = 11000.0

    pw, tw = np.nan, np.nan

    try:
        # if isinstance(rwyelevation_m, UN) or rwyelevation_m >= _Level1:
        #     if not isinstance(rwyelevation_m, UN) and rwyelevation_m >= _Level1:
        #        raise ValueError(f"Runway elevation {rwyelevation_m}m >= 11000m")

        T_isa_rwy = T0 + LR * rwyelevation_m
        rho_isa_rwy = (I1 + J1 * rwyelevation_m)**L1
        rho_actual_rwy = rho_isa_rwy * (1.0 + temp_offset / T_isa_rwy)**(-1.0)

        tw = (1.21 * (weight_n / S_sqm)) * (g * rho_actual_rwy * CLmax * groundrun_m)**(-1.0) + \
             (0.605 * (CDTO - mu_R * CLTO)) * CLmax**(-1.0) + \
             mu_R
        pw = tw * weight_n
        print('pw_takeoff', pw )

    except ValueError as e:
        print(f"Input or Calculation Error in takeoff_constraint_v0 for S={S_sqm:.2f}: {e}")
        tw, pw = np.nan, np.nan
    except Exception as e:
        print(f"Unexpected Error in takeoff_constraint_v0 for S={S_sqm:.2f}: {e}")
        tw, pw = np.nan, np.nan
    return tw, pw

## constant turn #####
def turn_constraint_v0(
    S_sqm,
    cruise_alt_m,
    temp_offset,
    cruisespeed_mps,
    AR,
    weight_n,
    CDmin,
    n, # Load factor
    ):
    R_JPKGPK = 287.05287
    GAMMA_DRY_AIR = 1.4
    T0 = 288.15
    LR = -0.0065
    I1 = 1.048840
    J1 = -23.659414e-6
    L1 = 4.2558797
    rho_sl_kgpm3 = I1**L1
    _Level1 = 11000.0

    tw, pw = np.nan, np.nan
    try:
        # if isinstance(cruise_alt_m, UN) or cruise_alt_m >= _Level1:
        #     if not isinstance(cruise_alt_m, UN) and cruise_alt_m >= _Level1:
        #         raise ValueError(f"Altitude {cruise_alt_m}m >= 11000m, model only valid for troposphere.")

        T_isa_cruise = T0 + LR * cruise_alt_m
        rho_isa_cruise = (I1 + J1 * cruise_alt_m)**L1

        rho_cruise_kgpm3 = rho_isa_cruise * (1.0 + temp_offset / T_isa_cruise)**(-1.0)
        q_pa = 0.5 * rho_cruise_kgpm3 * cruisespeed_mps**2.0

        # Oswald efficiency factor calculation - simplified as AR is float
        # AR_pow = AR**0.68
        # epsilon = 1.78 * (1.0 - 0.045 * AR_pow) - 0.64
        # k = 1.0 / (np.pi * AAR**0.68R * epsilon)
        k = 1.0 / (np.pi * AR * 1.78 * (1.0 - 0.045 * AR**0.68) - 0.64)


        if isinstance(q_pa, UN):
            q_pa_space = np.linspace(q_pa.bounds._left, q_pa.bounds._right, 100) 
            all_calculated_bounds = []
            for value in q_pa_space:
                tw_int = value * CDmin * (weight_n / S_sqm)**(-1.0) + \
                    k * n**2.0 * (weight_n / S_sqm) * value**(-1.0)
                all_calculated_bounds.append(tw_int.bounds._left)
                all_calculated_bounds.append(tw_int.bounds._right)
            tw = UN(essence='interval', bounds=[np.min(all_calculated_bounds), np.max(all_calculated_bounds)])
        else:
            tw = q_pa * CDmin * (weight_n / S_sqm)**(-1.0) + \
                k * n**2.0 * (weight_n / S_sqm) * q_pa**(-1.0)
   
        # tw = q_pa * CDmin * (weight_n / S_sqm)**(-1.0) + \
        #          k * n**2.0 * (weight_n / S_sqm) * q_pa**(-1.0)
        print("tw_cruise", tw)
        pw = tw * weight_n

    except Exception as e:
        print(f"Calculation failed in turn_constraint_v0 for S={S_sqm:.2f} m^2: {e}")
        tw, pw = np.nan, np.nan
    return tw, pw

## cruise #####
def cruise_constraint_v0(
    S_sqm,
    cruise_alt_m,
    temp_offset,
    cruisespeed_mps,
    AR,
    weight_n,
    CDmin,
    n, # n=1 for steady level cruise
    ):
    R_JPKGPK = 287.05287
    GAMMA_DRY_AIR = 1.4
    T0 = 288.15
    LR = -0.0065
    I1 = 1.048840
    J1 = -23.659414e-6
    L1 = 4.2558797
    rho_sl_kgpm3 = I1**L1
    _Level1 = 11000.0

    tw, pw = np.nan, np.nan
    try:
        if isinstance(cruise_alt_m, UN) or cruise_alt_m >= _Level1:
            if not isinstance(cruise_alt_m, UN) and cruise_alt_m >= _Level1 :
               raise ValueError(f"Altitude {cruise_alt_m}m >= 11000m, model only valid for troposphere.")

        T_isa_cruise = T0 + LR * cruise_alt_m
        rho_isa_cruise = (I1 + J1 * cruise_alt_m)**L1
        rho_cruise_kgpm3 = rho_isa_cruise * (1.0 + temp_offset / T_isa_cruise)**(-1)

        q_pa = 0.5 * rho_cruise_kgpm3 * cruisespeed_mps**2.0

        # Oswald efficiency factor calculation - simplified as AR is float
        AR_pow = AR**0.68
        epsilon = 1.78 * (1.0 - 0.045 * AR_pow) - 0.64
        k = 1.0 / (np.pi * AR * epsilon)

        # tw = q_pa * CDmin * (weight_n / S_sqm)**(-1.0) + \
        #      k * (weight_n / S_sqm) * q_pa**(-1.0) # n=1 for cruise, so n^2 is omitted

        if isinstance(q_pa, UN):
            q_pa_space = np.linspace(q_pa.bounds._left, q_pa.bounds._right, 100) 
            all_calculated_bounds = []
            for value in q_pa_space:
                tw_int = value * CDmin * (weight_n / S_sqm)**(-1.0) + \
                    k * (weight_n / S_sqm) * value**(-1.0)
                all_calculated_bounds.append(tw_int.bounds._left)
                all_calculated_bounds.append(tw_int.bounds._right)
            tw = UN(essence='interval', bounds=[np.min(all_calculated_bounds), np.max(all_calculated_bounds)])
        else:
            tw = q_pa * CDmin * (weight_n / S_sqm)**(-1.0) + \
                k * (weight_n / S_sqm) * q_pa**(-1.0)

        
        pw = q_pa*(CDmin*S_sqm) + k * q_pa**(-1.0) * (weight_n**2.0/S_sqm)

    except Exception as e:
        print(f"Calculation failed in cruise_constraint_v0 for S={S_sqm:.2f} m^2: {e}")
        tw, pw = np.nan, np.nan
    return tw, pw

# Service ######
def service_constraint_v0(
    S_sqm: float,
    weight_n: float,
    servalt_m: float,
    temp_offset: float,
    CDmin: float,
    CLmax: float, # Not directly used in this specific T/W calculation path
    CLminD: float,# Not directly used in this specific T/W calculation path
    AR: float
)-> tuple:
    R_JPKGPK = 287.05287
    GAMMA_DRY_AIR = 1.4
    T0 = 288.15
    LR = -0.0065
    I1 = 1.048840
    J1 = -23.659414e-6
    L1 = 4.2558797
    rho_sl_kgpm3 = I1**L1
    _Level1 = 11000.0

    tw, pw = np.nan, np.nan
    try:
        # if isinstance(servalt_m, UN) or servalt_m >= _Level1:
        #     if not isinstance(servalt_m, UN) and servalt_m >= _Level1:
        #        raise ValueError(f"Service altitude {servalt_m}m >= 11000m (Tropopause)")

        T_isa_alt = T0 + LR * servalt_m
        rho_isa_alt = (I1 + J1 * servalt_m)**L1
        rho_actual_alt = rho_isa_alt * (1.0 + temp_offset / T_isa_alt)**(-1.0)
        servspeed_mps_eas = (43.591 + 2.2452 * (weight_n / S_sqm) * 0.0208854) * 0.514444 # m/s EAS
        servspeed_mps_tas = servspeed_mps_eas * (rho_sl_kgpm3)**0.5 * (rho_actual_alt)**(-0.5) # m/s TAS

        q_pa_eas = 0.5 * rho_sl_kgpm3 * (servspeed_mps_eas)**2.0

        # Oswald efficiency factor calculation - simplified as AR is float
        AR_pow = AR**0.68
        epsilon = 1.78 * (1.0 - 0.045 * AR_pow) - 0.64
        k = 1.0 / (np.pi * AR * epsilon)

        servrate_mps_vert = 0.508 # 100 fpm in m/s
        tw = servrate_mps_vert * servspeed_mps_tas**(-1.0) + \
             q_pa_eas * CDmin * (weight_n / S_sqm)**(-1.0) + \
             k * (weight_n / S_sqm) * q_pa_eas**(-1.0)
        pw = tw * weight_n

        #print('pw', pw)

    except ValueError as e:
        print(f"Input or Calculation Error in service_constraint_v0 for S={S_sqm:.2f}: {e}")
        tw, pw = np.nan, np.nan
    except Exception as e:
        print(f"Unexpected Error in service_constraint_v0 for S={S_sqm:.2f}: {e}")
        tw, pw = np.nan, np.nan
    return tw,pw


# --- Setup Plotting ---
def extract_plot_data(un_object):
    if isinstance(un_object, UN):
        min_b, max_b = np.nan, np.nan
        if hasattr(un_object, 'bounds'):
            if un_object.bounds is not None:
                if hasattr(un_object.bounds, '_left') and hasattr(un_object.bounds, '_right'):
                    min_b = un_object.bounds._left
                    max_b = un_object.bounds._right
                elif isinstance(un_object.bounds, (list, tuple)) and len(un_object.bounds) == 2:
                    min_b = un_object.bounds[0]
                    max_b = un_object.bounds[1]
        return min_b, max_b
    elif isinstance(un_object, (float, int, np.floating, np.integer)):
        return float(un_object), float(un_object)
    else:
        print(f"Warning: Unexpected type in extract_plot_data: {type(un_object)}")
        return np.nan, np.nan

# --- Data Storage ---
# --- Data Storage ---
S_values_sqm = np.linspace(10, 100, 5)
ref_weight_n = 1320 * 4.44822
wing_loading_values = ref_weight_n / S_values_sqm

# For CSV output
csv_data = {'Wing_Loading_WS': wing_loading_values}
constraint_names = ["Climb", "Takeoff", "Turn", "Cruise", "Service"]

# Initialize lists for data collection
for name in constraint_names:
    csv_data[f'{name}_TW_BestEstimate'] = []
    csv_data[f'{name}_TW_Interval_Lower'] = []
    csv_data[f'{name}_TW_Interval_Upper'] = []

# --- Main Calculation Loop ---
print("Starting calculations for Best Estimate and Interval Analysis...")
for S_current in S_values_sqm:
    # Best Estimate (Constant parameters)
    inputs_climb_be, inputs_takeoff_be, inputs_turn_be, inputs_cruise_be, inputs_service_be = get_inputs(param_type="constant")
    
    csv_data['Climb_TW_BestEstimate'].append(climb_constraint_v0(S_sqm=S_current, **inputs_climb_be)[0])
    csv_data['Takeoff_TW_BestEstimate'].append(takeoff_constraint_v0(S_sqm=S_current, **inputs_takeoff_be)[0])
    csv_data['Turn_TW_BestEstimate'].append(turn_constraint_v0(S_sqm=S_current, **inputs_turn_be)[0])
    csv_data['Cruise_TW_BestEstimate'].append(cruise_constraint_v0(S_sqm=S_current, **inputs_cruise_be)[0])
    csv_data['Service_TW_BestEstimate'].append(service_constraint_v0(S_sqm=S_current, **inputs_service_be)[0])

    # Interval Analysis (Uncertain parameters)
    inputs_climb_un, inputs_takeoff_un, inputs_turn_un, inputs_cruise_un, inputs_service_un = get_inputs(param_type="uncertain")

    tw_c_un, _ = climb_constraint_v0(S_sqm=S_current, **inputs_climb_un)
    min_c, max_c = extract_plot_data(tw_c_un)
    csv_data['Climb_TW_Interval_Lower'].append(min_c)
    csv_data['Climb_TW_Interval_Upper'].append(max_c)

    tw_t_un, _ = takeoff_constraint_v0(S_sqm=S_current, **inputs_takeoff_un)
    min_t, max_t = extract_plot_data(tw_t_un)
    csv_data['Takeoff_TW_Interval_Lower'].append(min_t)
    csv_data['Takeoff_TW_Interval_Upper'].append(max_t)

    tw_tu_un, _ = turn_constraint_v0(S_sqm=S_current, **inputs_turn_un)
    min_tu, max_tu = extract_plot_data(tw_tu_un)
    csv_data['Turn_TW_Interval_Lower'].append(min_tu)
    csv_data['Turn_TW_Interval_Upper'].append(max_tu)

    tw_cr_un, _ = cruise_constraint_v0(S_sqm=S_current, **inputs_cruise_un)
    min_cr, max_cr = extract_plot_data(tw_cr_un)
    csv_data['Cruise_TW_Interval_Lower'].append(min_cr)
    csv_data['Cruise_TW_Interval_Upper'].append(max_cr)
    
    tw_serv_un, _ = service_constraint_v0(S_sqm=S_current, **inputs_service_un)
    min_serv, max_serv = extract_plot_data(tw_serv_un)
    csv_data['Service_TW_Interval_Lower'].append(min_serv)
    csv_data['Service_TW_Interval_Upper'].append(max_serv)

print("\nCalculations complete.")

# --- Save to CSV ---
df = pd.DataFrame(csv_data)
csv_file_path = 'interval_analysis_results.csv'
try:
    df.to_csv(csv_file_path, index=False, float_format='%.6f') # Format floats for better readability
    print(f"Results saved to {csv_file_path}")
except Exception as e:
    print(f"Error saving CSV file: {e}")

# --- Optional: Basic Plotting

# --- Plotting ---
plt.figure(figsize=(14, 10))

# # Climb
plt.plot(wing_loading_values, csv_data['Climb_TW_BestEstimate'], color='blue', linestyle='-', linewidth=2, label='Climb T/W (Best Estimate)')
plt.fill_between(wing_loading_values, csv_data['Climb_TW_Interval_Lower'], csv_data['Climb_TW_Interval_Upper'], color='blue', alpha=0.2, label='Climb T/W Uncertainty')

# Takeoff
plt.plot(wing_loading_values, csv_data['Takeoff_TW_BestEstimate'], color='red', linestyle='-', linewidth=2, label='Takeoff T/W (Best Estimate)')
plt.fill_between(wing_loading_values, csv_data['Takeoff_TW_Interval_Lower'], csv_data['Takeoff_TW_Interval_Upper'], color='red', alpha=0.2, label='Takeoff T/W Uncertainty')

# # Turn
plt.plot(wing_loading_values, csv_data['Turn_TW_BestEstimate'], color='green', linestyle='-', linewidth=2, label='Turn T/W (Best Estimate)')
plt.fill_between(wing_loading_values, csv_data['Turn_TW_Interval_Lower'], csv_data['Turn_TW_Interval_Upper'], color='green', alpha=0.2, label='Turn T/W Uncertainty')

# # Cruise
plt.plot(wing_loading_values, csv_data['Cruise_TW_BestEstimate'], color='purple', linestyle='-', linewidth=2, label='Cruise T/W (Best Estimate)')
plt.fill_between(wing_loading_values, csv_data['Cruise_TW_Interval_Lower'], csv_data['Cruise_TW_Interval_Upper'], color='purple', alpha=0.2, label='Cruise T/W Uncertainty')

# Service
plt.plot(wing_loading_values, csv_data['Service_TW_BestEstimate'], color='orange', linestyle='-', linewidth=2, label='Service Ceiling T/W (Best Estimate)')
plt.fill_between(wing_loading_values,csv_data['Service_TW_Interval_Lower'], csv_data['Service_TW_Interval_Upper'], color='orange', alpha=0.2, label='Service Ceiling T/W Uncertainty')


plt.xlabel("Wing Loading (W/S) [N/m$^2$]")
plt.ylabel("Required Thrust-to-Weight Ratio (T/W)")
plt.title("Thrust-to-Weight Ratio Constraints vs. Wing Loading")
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout(rect=[0, 0, 0.85, 1])
# Dynamic Y-axis limit, ensuring it handles potential all-NaN lists from errors
# max_val_for_ylim = 1.0 # Default minimum upper limit
# valid_max_lists = [
#     lst for lst in [
#         takeoff_tw_max_list, climb_tw_max_list, turn_tw_max_list,
#         cruise_tw_max_list, service_tw_max_list,
#         takeoff_tw_be_list, climb_tw_be_list, turn_tw_be_list, # Include best estimate in max val search
#         cruise_tw_be_list, service_tw_be_list
#     ] if lst and not all(x is None or np.isnan(x) for x in lst) # Check list not empty and not all NaN/None
# ]
# if valid_max_lists:
#     max_val_for_ylim = np.nanmax([np.nanmax(lst) for lst in valid_max_lists])

# plt.ylim(0, max_val_for_ylim * 1.1 if max_val_for_ylim > 0 else 1.0)

print("Plotting...")
plt.show()
print("Script finished.")

# import numpy as np
# import math # Kept for compatibility if any other math functions are essential and not covered by np
# import matplotlib.pyplot as plt
# from pyuncertainnumber import UncertainNumber as UN

# # --- Define Parameters ---
# # Constant (Best Estimate) Parameters
# CDmin_const = 0.0333
# temp_offset_const = 0.0
# cruisespeed_mps_const = 108 * 0.514444

# # Uncertain Parameters
# CDmin_un = UN(essence='interval', bounds=[0.0333 - 0.002, 0.0333 + 0.002])
# temp_offset_un = UN(essence='interval', bounds=[-30.0, 30.0]) # Ensure float for bounds
# cruisespeed_mps_un = UN(essence='interval', bounds=[0.8 * 108 * 0.514444, 1.2 * 108 * 0.514444])

# # --- Helper function to create input dictionaries ---
# def get_inputs(param_type="constant"):
#     if param_type == "constant":
#         current_CDmin = CDmin_const
#         current_temp_offset = temp_offset_const
#         current_cruisespeed_mps = cruisespeed_mps_const
#     else: # "uncertain"
#         current_CDmin = CDmin_un
#         current_temp_offset = temp_offset_un
#         current_cruisespeed_mps = cruisespeed_mps_un

#     inputs_climb = {
#         "weight_n": 1320 * 4.44822,
#         "climbalt_m": 0.0, # Ensure float
#         "climbspeed_mps": 66 * 0.514444, # Assuming this remains constant, or make it uncertain too
#         "climbrate_mps": 880 * 0.00508,
#         "temp_offset": current_temp_offset,
#         "CDmin": current_CDmin,
#         "CLmax": 1.756,
#         "CLminD": 0.20,
#         "AR": 8.0 # Ensure float
#     }

#     inputs_takeoff = {
#         "groundrun_m": 640 * 0.3048,
#         "rwyelevation_m": 0.0,
#         "temp_offset": current_temp_offset,
#         "weight_n": 1320 * 4.44822,
#         "AR": 8.0,
#         "CDmin": current_CDmin,
#         "mu_R": 0.04,
#         "CLTO": 0.5,
#         "CDTO": 0.038, # Takeoff specific drag, could also be UN(CDmin_base + delta_TO)
#         "CLmax": 1.756
#     }

#     inputs_turn = {
#         "cruise_alt_m": 8000 * 0.3048,
#         "temp_offset": current_temp_offset,
#         "cruisespeed_mps": current_cruisespeed_mps,
#         "AR": 8.0,
#         "weight_n": 1320 * 4.44822,
#         "CDmin": current_CDmin,
#         "n": 1.155
#     }

#     inputs_cruise = { # Same as turn for this setup, often n=1 for level cruise
#         "cruise_alt_m": 8000 * 0.3048,
#         "temp_offset": current_temp_offset,
#         "cruisespeed_mps": current_cruisespeed_mps,
#         "AR": 8.0,
#         "weight_n": 1320 * 4.44822,
#         "CDmin": current_CDmin,
#         "n": 1.0 # Typically n=1 for steady level cruise T/W
#     }

#     inputs_service = {
#         "weight_n": 1320*4.44822,
#         "servalt_m": 17000 * 0.3048,
#         "temp_offset": current_temp_offset,
#         "CDmin": current_CDmin,
#         "CLmax": 1.756,
#         "CLminD": 0.20,
#         "AR": 8.0
#     }
#     return inputs_climb, inputs_takeoff, inputs_turn, inputs_cruise, inputs_service

# ## Climb ######
# def climb_constraint_v0(
#     # --- Inputs ---
#     S_sqm: float,            # Wing Area [m^2]. Scalar.
#     weight_n: float,         # MTOW [N]. Scalar.
#     climbalt_m: float,       # Climb altitude [m]. Scalar.
#     climbspeed_mps: float,   # Climb speed [KIAS]. Scalar. estimated from VY = 43.591+2.245*(weight_n/S_sqm)
#     climbrate_mps: float,    # Climb rate [fpm]. Scalar.
#     temp_offset: float,      # ISA temperature offset [K or C]. Scalar.
#     CDmin: float,            # Minimum drag coefficient [-]. Scalar.
#     CLmax: float,            # Max lift coefficient [-]. Scalar.
#     CLminD: float,           # Lift coefficient at minimum drag [-]. Scalar.
#     AR: float
# )-> tuple:

#     # --- Constants ---
#     R_JPKGPK = 287.05287 # Specific Gas Constant for Air [J/(kg*K)]
#     GAMMA_DRY_AIR = 1.4  # Ratio of specific heats for air
#     T0 = 288.15           # ISA Sea Level Temperature [K]
#     LR = -0.0065          # ISA Temperature Lapse Rate in Troposphere [K/m]
#     I1 = 1.048840         # ISA Density Formula Constant (Troposphere)
#     J1 = -23.659414e-6    # ISA Density Formula Constant (Troposphere)
#     L1 = 4.2558797        # ISA Density Formula Constant (Troposphere)
#     rho_sl_kgpm3 = I1**L1 # ISA Sea Level Density [kg/m^3] (Approx 1.225)
#     #g = 9.80665           # Standard gravity [m/s^2]
#     _Level1 = 11000.0     # Altitude limit for troposphere [m] ensure float
#     # h = 2 # m assumed

#     # --- Initialize outputs ---
#     tw, pw = np.nan, np.nan
#     try: # Use try-except for robustness against potential math/input errors
#         # --- Atmosphere at Climb Altitude ---
#         if isinstance(climbalt_m, UN) or climbalt_m >= _Level1: # Handle UN comparison if needed by library
#             if not isinstance(climbalt_m, UN) and climbalt_m >= _Level1 :
#                raise ValueError(f"Climb altitude {climbalt_m}m >= 11000m (Tropopause)")
#         # For UN, comparison might need specific handling or be allowed by the library

#         T_isa_alt = T0 + LR * climbalt_m
#         rho_isa_alt = (I1 + J1 * climbalt_m)**L1
#         # Ensure T_isa_alt is not zero if temp_offset is UN to avoid division by zero in its uncertainty
#         # This check might be more complex if T_isa_alt itself becomes uncertain
#         if isinstance(T_isa_alt, UN): # A bit more careful with UN division
#             # If T_isa_alt can span zero, this needs robust handling in pyuncertainnumber
#             # Assuming T_isa_alt will be positive in practical scenarios for aircraft.
#             rho_actual_alt = rho_isa_alt / (1.0 + temp_offset / T_isa_alt)
#         elif T_isa_alt == 0 and isinstance(temp_offset, UN) and (temp_offset.bounds[0] <=0 or temp_offset.bounds[1] <=0):
#             raise ValueError("T_isa_alt is zero and temp_offset spans non-positive values, division by zero risk.")
#         elif T_isa_alt == 0 and temp_offset == 0: # Avoid 0/0 for non-UN case
#             rho_actual_alt = rho_isa_alt # Or handle as error
#         else:
#             rho_actual_alt = rho_isa_alt * (1.0 / (1.0 + temp_offset/ T_isa_alt))


#         # --- Flight Conditions (TAS, Mach, q) ---
#         climbspeed_mps_eas = (43.591 + 2.2452*(weight_n/S_sqm)* 0.0208854)* 0.514444
#         climbspeed_mps_tas_calc = climbspeed_mps # Use the input climbspeed_mps from dictionary

#         q_pa = 0.5 * rho_sl_kgpm3 * (climbspeed_mps_eas)**2 # q based on EAS

#         # Oswald efficiency factor calculation
#         if isinstance(AR, UN):
#             AR_pow = np.power(AR, 0.68)
#         else:
#             AR_pow = AR**0.68
#         epsilon = 1.78 * (1.0 - 0.045 * AR_pow) - 0.64

#         # Induced drag factor k
#         k = 1.0 / (np.pi * AR * epsilon)

#         climbrate_mps_vert = climbrate_mps
#         tw = climbrate_mps_vert * climbspeed_mps_tas_calc**(-1.0) + \
#              q_pa * CDmin * (weight_n/S_sqm)**(-1.0) + \
#              k * (weight_n/S_sqm) * q_pa**(-1.0)

#         pw = tw * weight_n # This is Power, not Power-to-Weight

#     except ValueError as e:
#         print(f"Input or Calculation Error in climb_constraint_v0 for S={S_sqm:.2f}: {e}")
#         tw, pw = np.nan, np.nan # Return NaNs on error
#     except Exception as e:
#         print(f"Unexpected Error in climb_constraint_v0 for S={S_sqm:.2f}: {e}")
#         tw, pw = np.nan, np.nan # Return NaNs on error

#     return tw,pw

# ## takeoff #####
# def takeoff_constraint_v0(S_sqm: float,
#     groundrun_m: float,
#     rwyelevation_m: float,
#     temp_offset: float,
#     weight_n: float,
#     AR: float,
#     CDmin: float,
#     mu_R: float,
#     CLTO: float,
#     CDTO: float,
#     CLmax: float
# )-> tuple:

#     # --- Constants ---
#     R_JPKGPK = 287.05287
#     GAMMA_DRY_AIR = 1.4
#     T0 = 288.15
#     LR = -0.0065
#     I1 = 1.048840
#     J1 = -23.659414e-6
#     L1 = 4.2558797
#     rho_sl_kgpm3 = I1**L1
#     g = 9.80665
#     _Level1 = 11000.0 # ensure float

#     # --- Initialize outputs ---
#     pw, tw = np.nan, np.nan

#     try:
#         if isinstance(rwyelevation_m, UN) or rwyelevation_m >= _Level1:
#             if not isinstance(rwyelevation_m, UN) and rwyelevation_m >= _Level1:
#                raise ValueError(f"Runway elevation {rwyelevation_m}m >= 11000m")

#         T_isa_rwy = T0 + LR * rwyelevation_m

#         rho_isa_rwy = (I1 + J1 * rwyelevation_m)**L1
#         if isinstance(T_isa_rwy, UN):
#             rho_actual_rwy = rho_isa_rwy / (1.0 + temp_offset / T_isa_rwy)
#         elif T_isa_rwy == 0 and isinstance(temp_offset, UN) and (temp_offset.bounds[0] <=0 or temp_offset.bounds[1] <=0):
#             raise ValueError("T_isa_rwy is zero and temp_offset spans non-positive values, division by zero risk.")
#         elif T_isa_rwy == 0 and temp_offset == 0:
#              rho_actual_rwy = rho_isa_rwy
#         else:
#              rho_actual_rwy = rho_isa_rwy * (1.0 / (1.0 + temp_offset / T_isa_rwy))

#         tw = (1.21 * (weight_n / S_sqm)) * (g * rho_actual_rwy * CLmax * groundrun_m)**(-1.0) + \
#              (0.605 * (CDTO - mu_R * CLTO)) * CLmax**(-1.0) + \
#              mu_R

#         pw = tw * weight_n # This is Power

#     except ValueError as e:
#         print(f"Input or Calculation Error in takeoff_constraint_v0 for S={S_sqm:.2f}: {e}")
#         tw, pw = np.nan, np.nan
#     except Exception as e:
#         print(f"Unexpected Error in takeoff_constraint_v0 for S={S_sqm:.2f}: {e}")
#         tw, pw = np.nan, np.nan

#     return tw, pw

# ## constant turn #####
# def turn_constraint_v0(
#     S_sqm,
#     cruise_alt_m,
#     temp_offset,
#     cruisespeed_mps,
#     AR,
#     weight_n,
#     CDmin,
#     n, # Load factor
#     ):
#     # --- Constants ---
#     R_JPKGPK = 287.05287
#     GAMMA_DRY_AIR = 1.4
#     T0 = 288.15
#     LR = -0.0065
#     I1 = 1.048840
#     J1 = -23.659414e-6
#     L1 = 4.2558797
#     rho_sl_kgpm3 = I1**L1
#     _Level1 = 11000.0 # ensure float

#     tw, pw = np.nan, np.nan

#     try:
#         if isinstance(cruise_alt_m, UN) or cruise_alt_m >= _Level1:
#             if not isinstance(cruise_alt_m, UN) and cruise_alt_m >= _Level1:
#                 raise ValueError(f"Altitude {cruise_alt_m}m >= 11000m, model only valid for troposphere.")

#         T_isa_cruise = T0 + LR * cruise_alt_m
#         rho_isa_cruise = (I1 + J1 * cruise_alt_m)**L1
#         if isinstance(T_isa_cruise, UN):
#             rho_cruise_kgpm3 = rho_isa_cruise / (1.0 + temp_offset / T_isa_cruise)
#         elif T_isa_cruise == 0 and isinstance(temp_offset, UN) and (temp_offset.bounds[0] <=0 or temp_offset.bounds[1] <=0):
#             raise ValueError("T_isa_cruise is zero and temp_offset spans non-positive values, division by zero risk.")
#         elif T_isa_cruise == 0 and temp_offset == 0:
#             rho_cruise_kgpm3 = rho_isa_cruise
#         else:
#             rho_cruise_kgpm3 = rho_isa_cruise * (1.0 / (1.0 + temp_offset / T_isa_cruise))

#         q_pa = 0.5 * rho_cruise_kgpm3 * cruisespeed_mps**2.0

#         AR_pow = AR**0.68
#         epsilon = 1.78 * (1.0 - 0.045 * AR_pow) - 0.64
#         k = 1.0 / (np.pi * AR * epsilon)

#         tw = q_pa * CDmin * (weight_n / S_sqm)**(-1.0) + \
#              k * n**2.0 * (weight_n / S_sqm) * q_pa**(-1.0)

#         pw_original_formula = q_pa*(CDmin*S_sqm + k * (n/q_pa)**2.0 * (weight_n**2.0/S_sqm))
#         pw = pw_original_formula


#     except Exception as e:
#         print(f"Calculation failed in turn_constraint_v0 for S={S_sqm:.2f} m^2: {e}")
#         tw, pw = np.nan, np.nan

#     return tw, pw

# ## cruise #####
# def cruise_constraint_v0(
#     S_sqm,
#     cruise_alt_m,
#     temp_offset,
#     cruisespeed_mps,
#     AR,
#     weight_n,
#     CDmin,
#     n, # n=1 for steady level cruise
#     ):
#     # --- Constants ---
#     R_JPKGPK = 287.05287
#     GAMMA_DRY_AIR = 1.4
#     T0 = 288.15
#     LR = -0.0065
#     I1 = 1.048840
#     J1 = -23.659414e-6
#     L1 = 4.2558797
#     rho_sl_kgpm3 = I1**L1
#     _Level1 = 11000.0 # ensure float

#     tw, pw = np.nan, np.nan
#     try:
#         if isinstance(cruise_alt_m, UN) or cruise_alt_m >= _Level1:
#             if not isinstance(cruise_alt_m, UN) and cruise_alt_m >= _Level1 :
#                raise ValueError(f"Altitude {cruise_alt_m}m >= 11000m, model only valid for troposphere.")

#         T_isa_cruise = T0 + LR * cruise_alt_m
#         rho_isa_cruise = (I1 + J1 * cruise_alt_m)**L1
#         if isinstance(T_isa_cruise, UN): # Handle UN division carefully
#             rho_cruise_kgpm3 = rho_isa_cruise / (1.0 + temp_offset / T_isa_cruise)
#         elif T_isa_cruise == 0 and isinstance(temp_offset, UN) and (temp_offset.bounds[0] <=0 or temp_offset.bounds[1] <=0): # Check for potential division by zero with UN
#             raise ValueError("T_isa_cruise is zero and temp_offset spans non-positive values, division by zero risk.")
#         elif T_isa_cruise == 0 and temp_offset == 0: # Avoid 0/0 for non-UN case
#              rho_cruise_kgpm3 = rho_isa_cruise # Or handle as specific error or condition
#         else:
#             rho_cruise_kgpm3 = rho_isa_cruise * (1.0 / (1.0 + temp_offset / T_isa_cruise))

#         q_pa = 0.5 * rho_cruise_kgpm3 * cruisespeed_mps**2.0

#         if isinstance(AR, UN): # Ensure np.power for UN if AR is UN
#             AR_pow = np.power(AR, 0.68)
#         else:
#             AR_pow = AR**0.68
#         epsilon = 1.78 * (1.0 - 0.045 * AR_pow) - 0.64
#         k = 1.0 / (np.pi * AR * epsilon)

#         tw = q_pa * CDmin * (weight_n / S_sqm)**(-1.0) + \
#              k * (weight_n / S_sqm) * q_pa**(-1.0) # n=1 for cruise

#         pw_original_formula = q_pa*(CDmin*S_sqm + k * q_pa**(-1.0) * (weight_n**2.0/S_sqm))
#         pw = pw_original_formula

#     except Exception as e:
#         print(f"Calculation failed in cruise_constraint_v0 for S={S_sqm:.2f} m^2: {e}")
#         tw, pw = np.nan, np.nan
#     return tw, pw

# # Service ######
# def service_constraint_v0(
#     S_sqm: float,
#     weight_n: float,
#     servalt_m: float,
#     temp_offset: float,
#     CDmin: float,
#     CLmax: float,
#     CLminD: float,
#     AR: float
# )-> tuple:
#     # Constants
#     R_JPKGPK = 287.05287
#     GAMMA_DRY_AIR = 1.4
#     T0 = 288.15
#     LR = -0.0065
#     I1 = 1.048840
#     J1 = -23.659414e-6
#     L1 = 4.2558797
#     rho_sl_kgpm3 = I1**L1
#     _Level1 = 11000.0 # ensure float

#     tw, pw = np.nan, np.nan
#     try:
#         if isinstance(servalt_m, UN) or servalt_m >= _Level1:
#             if not isinstance(servalt_m, UN) and servalt_m >= _Level1:
#                raise ValueError(f"Service altitude {servalt_m}m >= 11000m (Tropopause)")

#         T_isa_alt = T0 + LR * servalt_m
#         rho_isa_alt = (I1 + J1 * servalt_m)**L1
#         if isinstance(T_isa_alt, UN):
#             rho_actual_alt = rho_isa_alt / (1.0 + temp_offset / T_isa_alt)
#         elif T_isa_alt == 0 and isinstance(temp_offset, UN) and (temp_offset.bounds[0] <=0 or temp_offset.bounds[1] <=0):
#             raise ValueError("T_isa_alt is zero and temp_offset spans non-positive values.")
#         elif T_isa_alt == 0 and temp_offset == 0:
#              rho_actual_alt = rho_isa_alt
#         else:
#              rho_actual_alt = rho_isa_alt * (1.0 / (1.0 + temp_offset / T_isa_alt))

#         servspeed_mps_eas = (43.591 + 2.2452 * (weight_n / S_sqm) * 0.0208854) * 0.514444 # m/s EAS
#         servspeed_mps_tas = servspeed_mps_eas * (rho_sl_kgpm3 / rho_actual_alt)**0.5 # m/s TAS

#         q_pa_eas = 0.5 * rho_sl_kgpm3 * (servspeed_mps_eas)**2.0

#         if isinstance(AR, UN):
#             AR_pow = np.power(AR, 0.68)
#         else:
#             AR_pow = AR**0.68
#         epsilon = 1.78 * (1.0 - 0.045 * AR_pow) - 0.64
#         k = 1.0 / (np.pi * AR * epsilon)

#         servrate_mps_vert = 0.508 # 100 fpm in m/s
#         tw = servrate_mps_vert * servspeed_mps_tas**(-1.0) + \
#              q_pa_eas * CDmin * (weight_n / S_sqm)**(-1.0) + \
#              k * (weight_n / S_sqm) * q_pa_eas**(-1.0)

#         pw = tw * weight_n # This is Thrust.

#     except ValueError as e:
#         print(f"Input or Calculation Error in service_constraint_v0 for S={S_sqm:.2f}: {e}")
#         tw, pw = np.nan, np.nan
#     except Exception as e:
#         print(f"Unexpected Error in service_constraint_v0 for S={S_sqm:.2f}: {e}")
#         tw, pw = np.nan, np.nan
#     return tw,pw


# # --- Setup Plotting ---
# def extract_plot_data(un_object):
#     if isinstance(un_object, UN):
#         min_b, max_b = np.nan, np.nan
#         if hasattr(un_object, 'bounds'):
#             if un_object.bounds is not None:
#                 if hasattr(un_object.bounds, '_left') and hasattr(un_object.bounds, '_right'):
#                     min_b = un_object.bounds._left
#                     max_b = un_object.bounds._right
#                 elif isinstance(un_object.bounds, (list, tuple)) and len(un_object.bounds) == 2:
#                     min_b = un_object.bounds[0]
#                     max_b = un_object.bounds[1]
#         return min_b, max_b
#     elif isinstance(un_object, (float, int, np.floating, np.integer)):
#         return float(un_object), float(un_object)
#     else:
#         print(f"Warning: Unexpected type in extract_plot_data: {type(un_object)}")
#         return np.nan, np.nan

# # --- Data Storage ---
# S_values_sqm = np.linspace(10, 100, 50)
# ref_weight_n = 1320 * 4.44822
# wing_loading_values = ref_weight_n / S_values_sqm


# # Lists for Best Estimate (Constant Params)
# climb_tw_be_list = []
# takeoff_tw_be_list = []
# turn_tw_be_list = []
# cruise_tw_be_list = []
# service_tw_be_list = []

# # Lists for Uncertainty Bounds (Uncertain Params)
# climb_tw_min_list, climb_tw_max_list = [], []
# takeoff_tw_min_list, takeoff_tw_max_list = [], []
# turn_tw_min_list, turn_tw_max_list = [], []
# cruise_tw_min_list, cruise_tw_max_list = [], []
# service_tw_min_list, service_tw_max_list = [], []


# # --- Main Calculation Loop ---
# param_types = ["constant", "uncertain"]

# for run_type in param_types:
#     print(f"\n--- Starting calculations for run_type: {run_type} ---")
#     inputs_climb, inputs_takeoff, inputs_turn, inputs_cruise, inputs_service = get_inputs(run_type)

#     current_climb_tw_val1_list, current_climb_tw_val2_list = [], []
#     current_takeoff_tw_val1_list, current_takeoff_tw_val2_list = [], []
#     current_turn_tw_val1_list, current_turn_tw_val2_list = [], []
#     current_cruise_tw_val1_list, current_cruise_tw_val2_list = [], []
#     current_service_tw_val1_list, current_service_tw_val2_list = [], []

#     for S_current in S_values_sqm:
#         if S_current <= 0:
#             for lst_pair in [
#                 (current_climb_tw_val1_list, current_climb_tw_val2_list),
#                 (current_takeoff_tw_val1_list, current_takeoff_tw_val2_list),
#                 (current_turn_tw_val1_list, current_turn_tw_val2_list),
#                 (current_cruise_tw_val1_list, current_cruise_tw_val2_list),
#                 (current_service_tw_val1_list, current_service_tw_val2_list)
#             ]:
#                 lst_pair[0].append(np.nan)
#                 lst_pair[1].append(np.nan)
#             continue

#         tw_c, _ = climb_constraint_v0(S_sqm=S_current, **inputs_climb)
#         v1, v2 = extract_plot_data(tw_c)
#         current_climb_tw_val1_list.append(v1)
#         current_climb_tw_val2_list.append(v2)

#         tw_t, _ = takeoff_constraint_v0(S_sqm=S_current, **inputs_takeoff)
#         v1, v2 = extract_plot_data(tw_t)
#         current_takeoff_tw_val1_list.append(v1)
#         current_takeoff_tw_val2_list.append(v2)

#         tw_tu, _ = turn_constraint_v0(S_sqm=S_current, **inputs_turn)
#         v1, v2 = extract_plot_data(tw_tu)
#         current_turn_tw_val1_list.append(v1)
#         current_turn_tw_val2_list.append(v2)

#         tw_cr, _ = cruise_constraint_v0(S_sqm=S_current, **inputs_cruise)
#         v1, v2 = extract_plot_data(tw_cr)
#         current_cruise_tw_val1_list.append(v1)
#         current_cruise_tw_val2_list.append(v2)

#         tw_serv, _ = service_constraint_v0(S_sqm=S_current, **inputs_service)
#         v1, v2 = extract_plot_data(tw_serv)
#         current_service_tw_val1_list.append(v1)
#         current_service_tw_val2_list.append(v2)

#     if run_type == "constant":
#         climb_tw_be_list = current_climb_tw_val1_list
#         takeoff_tw_be_list = current_takeoff_tw_val1_list
#         turn_tw_be_list = current_turn_tw_val1_list
#         cruise_tw_be_list = current_cruise_tw_val1_list
#         service_tw_be_list = current_service_tw_val1_list
#     else: # "uncertain"
#         climb_tw_min_list = current_climb_tw_val1_list
#         climb_tw_max_list = current_climb_tw_val2_list
#         takeoff_tw_min_list = current_takeoff_tw_val1_list
#         takeoff_tw_max_list = current_takeoff_tw_val2_list
#         turn_tw_min_list = current_turn_tw_val1_list
#         turn_tw_max_list = current_turn_tw_val2_list
#         cruise_tw_min_list = current_cruise_tw_val1_list
#         cruise_tw_max_list = current_cruise_tw_val2_list
#         service_tw_min_list = current_service_tw_val1_list
#         service_tw_max_list = current_service_tw_val2_list

#     if run_type == "uncertain":
#         print("\nUncertainty Run Results (min_list, max_list for T/W):")
#         # print("S_values_sqm:", S_values_sqm) # Can be long
#         # print("Wing Loading (W/S) values for x-axis:", wing_loading_values) # Can be long
#         print("Sample Climb T/W min (first 5):", climb_tw_min_list[:5])
#         print("Sample Climb T/W max (first 5):", climb_tw_max_list[:5])


# print("\nConstraint calculations complete.")

# # --- Plotting ---
# plt.figure(figsize=(14, 10))

# # Climb
# plt.plot(wing_loading_values, climb_tw_be_list, color='blue', linestyle='-', linewidth=2, label='Climb T/W (Best Estimate)')
# plt.fill_between(wing_loading_values, climb_tw_min_list, climb_tw_max_list, color='blue', alpha=0.2, label='Climb T/W Uncertainty')

# # Takeoff
# plt.plot(wing_loading_values, takeoff_tw_be_list, color='red', linestyle='-', linewidth=2, label='Takeoff T/W (Best Estimate)')
# plt.fill_between(wing_loading_values, takeoff_tw_min_list, takeoff_tw_max_list, color='red', alpha=0.2, label='Takeoff T/W Uncertainty')

# # Turn
# plt.plot(wing_loading_values, turn_tw_be_list, color='green', linestyle='-', linewidth=2, label='Turn T/W (Best Estimate)')
# plt.fill_between(wing_loading_values, turn_tw_min_list, turn_tw_max_list, color='green', alpha=0.2, label='Turn T/W Uncertainty')

# # Cruise
# plt.plot(wing_loading_values, cruise_tw_be_list, color='purple', linestyle='-', linewidth=2, label='Cruise T/W (Best Estimate)')
# plt.fill_between(wing_loading_values, cruise_tw_min_list, cruise_tw_max_list, color='purple', alpha=0.2, label='Cruise T/W Uncertainty')

# # Service
# plt.plot(wing_loading_values, service_tw_be_list, color='orange', linestyle='-', linewidth=2, label='Service Ceiling T/W (Best Estimate)')
# plt.fill_between(wing_loading_values, service_tw_min_list, service_tw_max_list, color='orange', alpha=0.2, label='Service Ceiling T/W Uncertainty')


# plt.xlabel("Wing Loading (W/S) [N/m$^2$]")
# plt.ylabel("Required Thrust-to-Weight Ratio (T/W)")
# plt.title("Thrust-to-Weight Ratio Constraints vs. Wing Loading")
# plt.legend(loc='upper left', bbox_to_anchor=(1,1))
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.tight_layout(rect=[0, 0, 0.85, 1]) 
# plt.ylim(0, max(1.0, 
#                 np.nanmax(takeoff_tw_max_list) if takeoff_tw_max_list and not all(np.isnan(x) for x in takeoff_tw_max_list) else 0,
#                 np.nanmax(climb_tw_max_list) if climb_tw_max_list and not all(np.isnan(x) for x in climb_tw_max_list) else 0,
#                 np.nanmax(turn_tw_max_list) if turn_tw_max_list and not all(np.isnan(x) for x in turn_tw_max_list) else 0,
#                 np.nanmax(cruise_tw_max_list) if cruise_tw_max_list and not all(np.isnan(x) for x in cruise_tw_max_list) else 0,
#                 np.nanmax(service_tw_max_list) if service_tw_max_list and not all(np.isnan(x) for x in service_tw_max_list) else 0
#                 ) * 1.1 ) # Dynamic Y-axis limit
# # plt.gca().invert_xaxis() # Optional: Invert x-axis if conventional for W/S plots
# print("Plotting...")
# plt.show()
# print("Script finished.")