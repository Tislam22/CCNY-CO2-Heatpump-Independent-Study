from CoolProp.CoolProp import PropsSI

num_points = int(input("Enter number of gas cooler data points: "))

T_in_CO2_list = []
T_out_CO2_list = []
P_CO2_list = []
m_dot_CO2_list = []
m_dot_air_list = []

for i in range(num_points):
    print(f"\nData point {i+1}:")
    T_in_CO2 = float(input("  CO2 inlet temperature [°C]: ")) + 273.15
    T_out_CO2 = float(input("  CO2 outlet temperature [°C]: ")) + 273.15
    P_CO2 = float(input("  CO2 pressure [MPa]: ")) * 1e6
    m_dot_CO2 = float(input("  CO2 mass flow rate [kg/s]: "))
    air_cfm = float(input("  Air flow rate [CFM]: "))
    
    rho_air = 1.2
    m_dot_air = air_cfm * 0.0283168 * rho_air / 60
    
    T_in_CO2_list.append(T_in_CO2)
    T_out_CO2_list.append(T_out_CO2)
    P_CO2_list.append(P_CO2)
    m_dot_CO2_list.append(m_dot_CO2)
    m_dot_air_list.append(m_dot_air)

T_air_in_C = float(input("\nEnter outdoor air temperature [°C]: "))
T_air_in_K = T_air_in_C + 273.15
Cp_air = 1005

efficiencies = []

for i in range(num_points):
    T_in = T_in_CO2_list[i]
    T_out = T_out_CO2_list[i]
    P = P_CO2_list[i]
    m_CO2 = m_dot_CO2_list[i]
    m_air = m_dot_air_list[i]
    
    T_avg = (T_in + T_out)/2
    Cp_CO2 = PropsSI('C', 'T', T_avg, 'P', P, 'CO2')
    
    C_CO2 = m_CO2 * Cp_CO2
    C_air = m_air * Cp_air
    C_min = min(C_CO2, C_air)
    C_max = max(C_CO2, C_air)
    C_r = C_min / C_max
    
    eff_calc = (T_in - T_out) / (T_in - T_air_in_K)
    
    eff_max = (1 - C_r) / (1 + C_r)
    
    eff_real = min(eff_calc, eff_max)
    efficiencies.append(eff_real)

average_efficiency = sum(efficiencies) / len(efficiencies)
print(f"\nCalculated Realistic Gas Cooler Efficiency: {average_efficiency:.3f}")
