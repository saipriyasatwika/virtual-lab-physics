"""
Virtual Simulation Lab for Engineering Physics
10 Basic Physics Experiments with Visualizations - Interactive Version
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

print("VIRTUAL SIMULATION LAB FOR ENGINEERING PHYSICS")
print("\n")
# EXPERIMENT MENU

print("VIRTUAL SIMULATION LAB - EXPERIMENT MENU")
print("\nAvailable Experiments:")
print("1.  Simple Pendulum - Period vs Length")
print("2.  Ohm's Law - Current vs Voltage")
print("3.  Projectile Motion - Trajectory")
print("4.  Newton's Law of Cooling - Temperature vs Time")
print("5.  Simple Harmonic Motion - Displacement vs Time")
print("8.  Young's Double Slit - Intensity Distribution")
print("9.  Radioactive Decay - Exponential Decay")
print("10. Heat Conduction - Temperature Distribution")
print("11. Run ALL Experiments (with random data)")

choice = int(input("\nEnter experiment number (1-11): "))

if choice < 1 or choice > 11:
    print("Invalid choice! Please run again and select 1-11.")
    raise SystemExit

# EXPERIMENT 1: SIMPLE PENDULUM
if choice == 1 or choice == 11:
    print("EXPERIMENT 1: SIMPLE PENDULUM")
    print("\nTheoretical Background:")
    print("The period of a simple pendulum depends on its length and gravitational")
    print("acceleration. For small angles: T = 2π√(L/g)")
    print("where T = period (s), L = length (m), g = 9.81 m/s²")
    
    if choice == 1:
        # Take user input for 5 observations
        print("\nEnter length measurements (in meters) for 5 observations:")
        lengths = []
        for i in range(5):
            length = float(input(f"Observation {i+1} - Length (m): "))
            lengths.append(length)
        
        lengths = np.array(lengths)
        g = 9.81  # acceleration due to gravity (m/s²)
        
        # Calculate period: T = 2π√(L/g)
        periods = 2 * np.pi * np.sqrt(lengths / g)
        
        # Generate extended data for smooth curve
        lengths_curve = np.linspace(max(0.1, lengths.min()-0.1), lengths.max()+0.1, 100)
        periods_curve = 2 * np.pi * np.sqrt(lengths_curve / g)
    else:
        # Random data for "Run All" option
        lengths = np.linspace(0.1, 2.0, 20) + np.random.uniform(-0.02, 0.02, 20)
        g = 9.81
        periods = 2 * np.pi * np.sqrt(lengths / g)
        periods += np.random.normal(0, 0.02, len(periods))
        lengths_curve = lengths
        periods_curve = 2 * np.pi * np.sqrt(lengths / g)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(lengths, periods, c='blue', s=100, alpha=0.8, label='Experimental Data', edgecolors='black')
    plt.plot(lengths_curve, periods_curve, 'r-', linewidth=2, label='Theoretical')
    plt.xlabel('Length (m)', fontsize=12)
    plt.ylabel('Period (s)', fontsize=12)
    plt.title('Simple Pendulum: Period vs Length', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\nResults: Period ranges from {periods.min():.3f}s to {periods.max():.3f}s")

# EXPERIMENT 2: OHM'S LAW
if choice == 2 or choice == 11:
    print("EXPERIMENT 2: OHM'S LAW")
    print("\nTheoretical Background:")
    print("Ohm's Law states that the current through a conductor is directly")
    print("proportional to the voltage across it: V = IR")
    print("where V = voltage (V), I = current (A), R = resistance (Ω)")
    
    if choice == 2:
        # Take user input
        resistance = float(input("\nEnter the resistance value (Ω): "))
        print("\nEnter voltage measurements (in Volts) for 5 observations:")
        voltages = []
        for i in range(5):
            voltage = float(input(f"Observation {i+1} - Voltage (V): "))
            voltages.append(voltage)
        
        voltages = np.array(voltages)
        
        # Calculate current: I = V/R
        currents = voltages / resistance
        
        # Generate extended data for smooth curve
        voltages_curve = np.linspace(0, voltages.max()+2, 100)
        currents_curve = voltages_curve / resistance
    else:
        # Random data for "Run All" option
        voltages = np.linspace(0, 12, 15) + np.random.uniform(-0.1, 0.1, 15)
        resistance = np.random.uniform(8, 12)
        currents = voltages / resistance
        currents += np.random.normal(0, 0.01, len(currents))
        voltages_curve = voltages
        currents_curve = voltages / resistance
    
    plt.figure(figsize=(10, 6))
    plt.scatter(voltages, currents, c='green', s=100, alpha=0.8, label='Experimental Data', edgecolors='black')
    plt.plot(voltages_curve, currents_curve, 'r-', linewidth=2, label=f'Theoretical (R={resistance:.2f}Ω)')
    plt.xlabel('Voltage (V)', fontsize=12)
    plt.ylabel('Current (A)', fontsize=12)
    plt.title("Ohm's Law: Current vs Voltage", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\nResults: Measured Resistance = {resistance:.2f} Ω")

# EXPERIMENT 3: PROJECTILE MOTION
if choice == 3 or choice == 11:    
    print("EXPERIMENT 3: PROJECTILE MOTION")
    print("\nTheoretical Background:")
    print("A projectile follows a parabolic path under gravity.")
    print("x(t) = v₀cos(θ)t,  y(t) = v₀sin(θ)t - (1/2)gt²")
    print("where v₀ = initial velocity, θ = launch angle, g = 9.81 m/s²")
    
    if choice == 3:
        # Take user input
        v0 = float(input("\nEnter initial velocity (m/s): "))
        theta = float(input("Enter launch angle (degrees): "))
        theta_rad = np.radians(theta)
        g = 9.81
    else:
        # Random initial conditions
        v0 = np.random.uniform(20, 30)
        theta = np.random.uniform(30, 60)
        theta_rad = np.radians(theta)
        g = 9.81
    
    # Time array
    t_max = 2 * v0 * np.sin(theta_rad) / g
    t = np.linspace(0, t_max, 100)
    
    # Calculate trajectory
    x = v0 * np.cos(theta_rad) * t
    y = v0 * np.sin(theta_rad) * t - 0.5 * g * t**2
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='Trajectory')
    plt.scatter([0, x[-1]], [0, 0], c='red', s=100, marker='o', label='Start/End', zorder=5)
    plt.xlabel('Horizontal Distance (m)', fontsize=12)
    plt.ylabel('Vertical Height (m)', fontsize=12)
    plt.title(f'Projectile Motion: v₀={v0:.1f}m/s, θ={theta:.1f}°', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\nResults: Range = {x[-1]:.2f} m, Max Height = {y.max():.2f} m")


# EXPERIMENT 4: NEWTON'S LAW OF COOLING

if choice == 4 or choice == 11:
    print("EXPERIMENT 4: NEWTON'S LAW OF COOLING")
    print("\nTheoretical Background:")
    print("The rate of cooling is proportional to temperature difference:")
    print("T(t) = Tₐ + (T₀ - Tₐ)e^(-kt)")
    print("where T₀ = initial temp, Tₐ = ambient temp, k = cooling constant")
    
    if choice == 4:
        # Take user input
        T0 = float(input("\nEnter initial temperature (°C): "))
        Ta = float(input("Enter ambient temperature (°C): "))
        k = float(input("Enter cooling constant (1/min, typical: 0.03-0.06): "))
        
        print("\nEnter time measurements (in minutes) for 5 observations:")
        time_obs = []
        for i in range(5):
            t = float(input(f"Observation {i+1} - Time (min): "))
            time_obs.append(t)
        
        time_obs = np.array(time_obs)
        
        # Calculate temperature at observation times
        temperature_obs = Ta + (T0 - Ta) * np.exp(-k * time_obs)
        
        # Generate extended data for smooth curve
        time_curve = np.linspace(0, max(60, time_obs.max()+10), 100)
        temperature_curve = Ta + (T0 - Ta) * np.exp(-k * time_curve)
    else:
        # Random initial conditions
        T0 = np.random.uniform(80, 100)
        Ta = np.random.uniform(20, 25)
        k = np.random.uniform(0.03, 0.06)
        time_obs = np.linspace(0, 60, 50)
        temperature_obs = Ta + (T0 - Ta) * np.exp(-k * time_obs)
        temperature_obs += np.random.normal(0, 0.5, len(time_obs))
        time_curve = time_obs
        temperature_curve = Ta + (T0 - Ta) * np.exp(-k * time_obs)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(time_obs, temperature_obs, c='orange', s=100, alpha=0.8, label='Experimental Data', edgecolors='black')
    plt.plot(time_curve, temperature_curve, 'r-', linewidth=2, label='Theoretical')
    plt.axhline(y=Ta, color='blue', linestyle='--', alpha=0.5, label=f'Ambient Temp = {Ta:.1f}°C')
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.title("Newton's Law of Cooling", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\nResults: T₀ = {T0:.1f}°C, Tₐ = {Ta:.1f}°C, k = {k:.4f} min⁻¹")


# EXPERIMENT 5: SIMPLE HARMONIC MOTION

if choice == 5 or choice == 11:
    print("EXPERIMENT 5: SIMPLE HARMONIC MOTION")
    print("\nTheoretical Background:")
    print("Displacement in SHM: x(t) = A·sin(ωt + φ)")
    print("where A = amplitude, ω = angular frequency, φ = phase constant")
    
    if choice == 5:
        # Take user input
        A = float(input("\nEnter amplitude (m): "))
        omega = float(input("Enter angular frequency (rad/s): "))
        phi = float(input("Enter phase constant (radians, 0 for simplicity): "))
    else:
        # Random initial conditions
        A = np.random.uniform(0.05, 0.15)
        omega = np.random.uniform(2, 5)
        phi = np.random.uniform(0, np.pi/4)
    
    # Time array for full visualization
    time = np.linspace(0, 4*np.pi/omega, 500)
    
    # Calculate displacement: x(t) = A·sin(ωt + φ)
    displacement = A * np.sin(omega * time + phi)
    
    plt.figure(figsize=(12, 6))
    plt.plot(time, displacement, 'purple', linewidth=2)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Displacement (m)', fontsize=12)
    plt.title(f'Simple Harmonic Motion: A={A:.3f}m, ω={omega:.2f}rad/s', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    period = 2 * np.pi / omega
    print(f"\nResults: Amplitude = {A:.4f} m, Period = {period:.3f} s, Frequency = {1/period:.3f} Hz")
# EXPERIMENT 6: PHOTOELECTRIC EFFECT

if choice == 6 or choice == 11:
    print("EXPERIMENT 6: PHOTOELECTRIC EFFECT")
    print("\nTheoretical Background:")
    print("Einstein's photoelectric equation: eV₀ = hf - φ")
    print("where V₀ = stopping potential, h = Planck's constant, f = frequency, φ = work function")
    
    h = constants.h  # Planck's constant (J·s)
    e = constants.e  # Elementary charge (C)
    
    if choice == 6:
        # Take user input
        work_function_eV = float(input("\nEnter work function (eV, typical: 2.0-3.0): "))
        work_function = work_function_eV * e  # Convert to Joules
        
        print("\nEnter frequency measurements (in 10^14 Hz) for 5 observations:")
        frequencies_input = []
        for i in range(5):
            freq = float(input(f"Observation {i+1} - Frequency (×10¹⁴ Hz): "))
            frequencies_input.append(freq * 1e14)
        
        frequencies = np.array(frequencies_input)
        
        # Calculate stopping potential: V0 = (hf - φ)/e
        stopping_potential = (h * frequencies - work_function) / e
        
        # Generate extended data for smooth curve
        freq_min = max(work_function/h, frequencies.min()*0.8)
        frequencies_curve = np.linspace(freq_min, frequencies.max()*1.2, 100)
        stopping_potential_curve = (h * frequencies_curve - work_function) / e
        
        # Filter positive values
        mask = stopping_potential > 0
        mask_curve = stopping_potential_curve > 0
        
        frequencies_plot = frequencies[mask] / 1e14
        stopping_potential_plot = stopping_potential[mask]
        frequencies_curve_plot = frequencies_curve[mask_curve] / 1e14
        stopping_potential_curve_plot = stopping_potential_curve[mask_curve]
    else:
        # Random data
        frequencies = np.linspace(5, 10, 12) * 1e14
        work_function = np.random.uniform(2.0, 3.0) * e
        stopping_potential = (h * frequencies - work_function) / e
        stopping_potential += np.random.normal(0, 0.05, len(stopping_potential))
        mask = stopping_potential > 0
        frequencies_plot = frequencies[mask] / 1e14
        stopping_potential_plot = stopping_potential[mask]
        frequencies_curve_plot = frequencies_plot
        stopping_potential_curve_plot = (h * frequencies[mask] - work_function) / e
    
    plt.figure(figsize=(10, 6))
    plt.scatter(frequencies_plot, stopping_potential_plot, c='cyan', s=100, 
                alpha=0.8, label='Experimental Data', edgecolors='black')
    plt.plot(frequencies_curve_plot, stopping_potential_curve_plot, 
             'r-', linewidth=2, label='Theoretical')
    plt.xlabel('Frequency (×10¹⁴ Hz)', fontsize=12)
    plt.ylabel('Stopping Potential (V)', fontsize=12)
    plt.title('Photoelectric Effect: Stopping Potential vs Frequency', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\nResults: Work Function = {work_function/e:.2f} eV")


# EXPERIMENT 7: RCL CIRCUIT (AC)

if choice == 7 or choice == 11:
    print("EXPERIMENT 7: RCL CIRCUIT - RESONANCE")
    print("\nTheoretical Background:")
    print("Impedance in RCL circuit: Z = √[R² + (XL - XC)²]")
    print("where XL = ωL (inductive), XC = 1/(ωC) (capacitive)")
    print("Resonance occurs at f₀ = 1/(2π√LC)")
    
    if choice == 7:
        # Take user input
        R = float(input("\nEnter resistance (Ω): "))
        L = float(input("Enter inductance (H): "))
        C = float(input("Enter capacitance (µF): ")) * 1e-6  # Convert to Farads
    else:
        # Random circuit parameters
        R = np.random.uniform(50, 100)
        L = np.random.uniform(0.05, 0.15)
        C = np.random.uniform(10, 50) * 1e-6
    
    # Frequency range around resonance
    f0 = 1 / (2 * np.pi * np.sqrt(L * C))
    frequencies = np.linspace(f0*0.2, f0*2, 200)
    omega = 2 * np.pi * frequencies
    
    # Calculate impedance: Z = √[R² + (ωL - 1/(ωC))²]
    XL = omega * L  # Inductive reactance
    XC = 1 / (omega * C)  # Capacitive reactance
    impedance = np.sqrt(R**2 + (XL - XC)**2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, impedance, 'b-', linewidth=2, label='Impedance')
    plt.axvline(x=f0, color='r', linestyle='--', linewidth=2, label=f'Resonance f₀={f0:.1f}Hz')
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Impedance (Ω)', fontsize=12)
    plt.title('RCL Circuit: Impedance vs Frequency', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\nResults: Resonant Frequency = {f0:.2f} Hz, Min Impedance = {R:.2f} Ω")

# EXPERIMENT 8: YOUNG'S DOUBLE SLIT EXPERIMENT
if choice == 8 or choice == 11:
    print("EXPERIMENT 8: YOUNG'S DOUBLE SLIT EXPERIMENT")
    print("\nTheoretical Background:")
    print("Interference pattern intensity: I = I₀cos²(πd·sinθ/λ)")
    print("where d = slit separation, λ = wavelength, θ = angle")
    
    if choice == 8:
        # Take user input
        wavelength_nm = float(input("\nEnter wavelength (nm, visible: 400-700): "))
        wavelength = wavelength_nm * 1e-9  # Convert to meters
        d_mm = float(input("Enter slit separation (mm): "))
        d = d_mm * 1e-3  # Convert to meters
        D = float(input("Enter distance to screen (m, typical: 1.0): "))
    else:
        # Random experimental parameters
        wavelength = np.random.uniform(500, 650) * 1e-9
        d = np.random.uniform(0.1, 0.3) * 1e-3
        D = 1.0
    
    # Position on screen
    y = np.linspace(-0.02, 0.02, 500)  # position (m)
    theta = np.arctan(y / D)
    
    # Calculate intensity: I = I0·cos²(πd·sinθ/λ)
    I0 = 1.0
    intensity = I0 * np.cos(np.pi * d * np.sin(theta) / wavelength)**2
    
    plt.figure(figsize=(12, 6))
    plt.plot(y * 1000, intensity, 'r-', linewidth=2)
    plt.xlabel('Position on Screen (mm)', fontsize=12)
    plt.ylabel('Relative Intensity', fontsize=12)
    plt.title(f"Young's Double Slit: λ={wavelength*1e9:.0f}nm, d={d*1e3:.2f}mm", 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    fringe_spacing = wavelength * D / d
    print(f"\nResults: Wavelength = {wavelength*1e9:.1f} nm, Fringe Spacing = {fringe_spacing*1000:.3f} mm")

# EXPERIMENT 9: RADIOACTIVE DECAY
if choice == 9 or choice==11: 
    print("EXPERIMENT 9: RADIOACTIVE DECAY")
    print("\nTheoretical Background:")
    print("Radioactive decay follows: N(t) = N₀e^(-λt)")
    print("where N₀ = initial number, λ = decay constant, t₁/₂ = ln(2)/λ")
    
    if choice == 9:
        # Take user input
        N0 = int(input("\nEnter initial number of nuclei: "))
        half_life = float(input("Enter half-life (days): "))
        lambda_decay = np.log(2) / half_life  # decay constant
        
        print("\nEnter time measurements (in days) for 5 observations:")
        time_obs = []
        for i in range(5):
            t = float(input(f"Observation {i+1} - Time (days): "))
            time_obs.append(t)
        
        time_obs = np.array(time_obs)
        
        # Calculate number of nuclei at observation times
        N_obs = N0 * np.exp(-lambda_decay * time_obs)
        
        # Generate extended data for smooth curve
        time_curve = np.linspace(0, max(50, time_obs.max()+10), 100)
        N_curve = N0 * np.exp(-lambda_decay * time_curve)
    else:
        # Random initial conditions
        N0 = np.random.randint(1000, 10000)
        half_life = np.random.uniform(5, 15)
        lambda_decay = np.log(2) / half_life
        time_obs = np.linspace(0, 50, 100)
        N_obs = N0 * np.exp(-lambda_decay * time_obs)
        N_obs += np.random.normal(0, np.sqrt(N_obs) * 0.3)
        time_curve = time_obs
        N_curve = N0 * np.exp(-lambda_decay * time_obs)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(time_obs, N_obs, c='green', s=100, alpha=0.8, label='Experimental Data', 
                edgecolors='black', zorder=5)
    plt.plot(time_curve, N_curve, 'r-', linewidth=2, label='Theoretical')
    plt.axhline(y=N0/2, color='blue', linestyle='--', alpha=0.5, label=f'Half-life = {half_life:.1f} days')
    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('Number of Nuclei', fontsize=12)
    plt.title('Radioactive Decay: Number of Nuclei vs Time', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    
    print(f"\nResults: N₀ = {N0}, Half-life = {half_life:.2f} days, λ = {lambda_decay:.4f} day⁻¹")

# EXPERIMENT 10: HEAT CONDUCTION ALONG A ROD
if choice == 10 or choice == 11:
    
    print("EXPERIMENT 10: HEAT CONDUCTION")
    print("\nTheoretical Background:")
    print("Steady-state temperature distribution: T(x) = T₁ + (T₂-T₁)x/L")
    print("where T₁, T₂ = end temperatures, L = rod length, x = position")
    
    if choice == 10:
        # Take user input
        L = float(input("\nEnter rod length (m): "))
        T1 = float(input("Enter hot end temperature (°C): "))
        T2 = float(input("Enter cold end temperature (°C): "))
        
        print("\nEnter position measurements (in cm) for 5 observations:")
        x_obs = []
        for i in range(5):
            pos = float(input(f"Observation {i+1} - Position (cm): "))
            x_obs.append(pos / 100)  # Convert to meters
        
        x_obs = np.array(x_obs)
        
        # Calculate temperature at observation points
        temperature_obs = T1 + (T2 - T1) * x_obs / L
        
        # Generate extended data for smooth curve
        x_curve = np.linspace(0, L, 100)
        temperature_curve = T1 + (T2 - T1) * x_curve / L
    else:
        # Random rod parameters
        L = 1.0
        T1 = np.random.uniform(90, 100)
        T2 = np.random.uniform(20, 30)
        x_obs = np.linspace(0, L, 50)
        temperature_obs = T1 + (T2 - T1) * x_obs / L
        temperature_obs += np.random.normal(0, 1, len(x_obs))
        x_curve = x_obs
        temperature_curve = T1 + (T2 - T1) * x_obs / L
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_obs * 100, temperature_obs, c='red', s=100, alpha=0.8, 
                label='Experimental Data', edgecolors='black', zorder=5)
    plt.plot(x_curve * 100, temperature_curve, 'b-', linewidth=2, label='Theoretical')
    plt.xlabel('Position along Rod (cm)', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.title(f'Heat Conduction: T₁={T1:.1f}°C, T₂={T2:.1f}°C', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\nResults: Hot End = {T1:.1f}°C, Cold End = {T2:.1f}°C")
    print(f"Temperature Gradient = {(T2-T1)/L:.2f} °C/m")

# SUMMARY

if choice == 11:
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
else:
    print(f"EXPERIMENT {choice} COMPLETED SUCCESSFULLY!")

print("\nThank you for using the Virtual Physics Lab!")