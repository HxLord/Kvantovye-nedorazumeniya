"""
HELIOS GUARD - with Real-time Matplotlib Visualization
=======================================================
Run this version to see the solar activity graph in real-time.
"""

import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, List

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

# Configuration
DISTANCE_AU_KM = 150_000_000
X_FLARE_THRESHOLD = 1e-4
M_FLARE_THRESHOLD = 1e-5
POWER_SLEEP = 0.01
POWER_AI_ANALYSIS = 2.0
SIMULATION_SPEED = 0.5
MAX_DATA_POINTS = 50

# Colors for flare classes
FLARE_COLORS = {
    'A': '#4CAF50', 'B': '#8BC34A', 'C': '#FFC107', 
    'M': '#FF9800', 'X': '#F44336'
}

class FlareClass(Enum):
    A = "A"; B = "B"; C = "C"; M = "M"; X = "X"

@dataclass
class SolarWeatherData:
    timestamp: datetime
    xray_flux: float
    flare_class: FlareClass
    cme_velocity: Optional[float] = None

# Visualization class
class SolarVisualizer:
    def __init__(self):
        self.times, self.flux, self.classes, self.power = [], [], [], []
        self.maneuver_times = []
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 7))
        self.fig.suptitle('HELIOS GUARD - Solar Activity Monitor', fontweight='bold', fontsize=14)
        
        # Top plot: X-ray flux
        self.ax1.set_ylabel('X-ray Flux (W/m^2) - Log Scale')
        self.ax1.set_yscale('log')
        self.ax1.set_ylim(1e-8, 1e-2)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.axhline(y=X_FLARE_THRESHOLD, color='red', linestyle='--', lw=2, label='X-class threshold')
        
        # Bottom plot: Power
        self.ax2.set_xlabel('Time (iteration)')
        self.ax2.set_ylabel('Power (W)')
        self.ax2.set_ylim(0, 3)
        self.ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.anim = animation.FuncAnimation(self.fig, self.update, interval=200)
        
    def add(self, iteration, flux, flare_class, power, is_maneuver=False):
        self.times.append(iteration)
        self.flux.append(flux)
        self.classes.append(flare_class.value)
        self.power.append(power)
        if is_maneuver:
            self.maneuver_times.append(iteration)
        # Keep only recent data
        if len(self.times) > MAX_DATA_POINTS:
            self.times = self.times[-MAX_DATA_POINTS:]
            self.flux = self.flux[-MAX_DATA_POINTS:]
            self.classes = self.classes[-MAX_DATA_POINTS:]
            self.power = self.power[-MAX_DATA_POINTS:]
            
    def update(self, frame):
        if not self.times:
            return
        
        # Update flux plot
        self.ax1.clear()
        self.ax1.set_ylabel('X-ray Flux (W/m^2)')
        self.ax1.set_yscale('log')
        self.ax1.set_ylim(1e-8, 1e-2)
        
        colors = [FLARE_COLORS.get(c, 'gray') for c in self.classes]
        x_range = list(range(len(self.times)))
        
        self.ax1.scatter(x_range, self.flux, c=colors, s=60, edgecolors='black', zorder=5)
        self.ax1.axhline(y=X_FLARE_THRESHOLD, color='red', linestyle='--', lw=2, alpha=0.8)
        
        # Mark maneuvers
        for mt in self.maneuver_times:
            if mt >= x_range[0]:
                self.ax1.axvline(x=mt, color='red', linestyle='-', lw=2, alpha=0.7)
        
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_xlim(max(0, len(self.times)-MAX_DATA_POINTS)-1, len(self.times))
        
        # Legend
        legend = [Line2D([0],[0], marker='o', color='w', markerfacecolor=FLARE_COLORS[k], 
                        markersize=10, label=k) for k in ['A', 'C', 'M', 'X']]
        self.ax1.legend(handles=legend, loc='upper left', fontsize=8)
        
        # Update power plot
        self.ax2.clear()
        self.ax2.set_xlabel('Time (iteration)')
        self.ax2.set_ylabel('Power (W)')
        self.ax2.set_ylim(0, 3)
        self.ax2.plot(x_range, self.power, 'b-', lw=2)
        self.ax2.axhline(y=POWER_SLEEP, color='green', linestyle=':', alpha=0.7, label=f'Sleep ({POWER_SLEEP}W)')
        self.ax2.axhline(y=POWER_AI_ANALYSIS, color='red', linestyle=':', alpha=0.7, label=f'AI ({POWER_AI_ANALYSIS}W)')
        self.ax2.legend(loc='upper right', fontsize=8)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_xlim(max(0, len(self.times)-MAX_DATA_POINTS)-1, len(self.times))
        
    def show(self):
        plt.show()


# Main System Classes (simplified for visualization demo)
class SensorUnit:
    def __init__(self):
        self.activity_level = 0.95
        
    def generate(self, force_flare=False):
        if force_flare or random.random() > self.activity_level:
            flux = random.uniform(1e-4, 1e-3)
            flare = FlareClass.X
            cme = random.uniform(1500, 3000)
        elif random.random() > 0.7:
            flux = random.uniform(1e-5, 1e-4)
            flare = FlareClass.M
            cme = random.uniform(800, 1500)
        elif random.random() > 0.5:
            flux = random.uniform(1e-6, 1e-5)
            flare = FlareClass.C
            cme = random.uniform(400, 800)
        else:
            flux = random.uniform(1e-8, 1e-6)
            flare = FlareClass.A if flux < 1e-7 else FlareClass.B
            cme = None
            
        return SolarWeatherData(datetime.now(), flux, flare, cme)

class TinyMLTrigger:
    def __init__(self):
        self.power = POWER_SLEEP
        self.active = False
        
    def check(self, flux):
        exceeded = flux >= X_FLARE_THRESHOLD
        if exceeded and not self.active:
            self.active = True
            self.power = POWER_AI_ANALYSIS
            print(f">>> THRESHOLD EXCEEDED! {flux:.2e} W/m^2 -> AI activated ({POWER_SLEEP}W -> {self.power}W)")
        elif not exceeded and self.active:
            self.active = False
            self.power = POWER_SLEEP
            print(f">>> Returning to sleep mode ({POWER_AI_ANALYSIS}W -> {self.power}W)")
        return exceeded

class MissionAI:
    def __init__(self):
        self.analysis_count = 0
        self.maneuver_count = 0
        
    def analyze(self, data):
        self.analysis_count += 1
        velocity = data.cme_velocity if data.cme_velocity else 2000
        toa = DISTANCE_AU_KM / velocity / 3600 * 1.2
        threat = "CRITICAL" if data.flare_class == FlareClass.X else "HIGH" if data.flare_class == FlareClass.M else "LOW"
        
        print(f"[AI #{self.analysis_count}] Class: {data.flare_class.value} | ToA: {toa:.1f}h | Threat: {threat}")
        
        if threat in ["CRITICAL", "HIGH"]:
            self.maneuver_count += 1
            print(f"!!! PROTECTIVE_MANEUVER_{threat} (#{self.maneuver_count})")
            return True
        return False


def run_demo():
    """Run the complete demo with visualization."""
    print("\n" + "="*60)
    print("HELIOS GUARD - SATELLITE PROTECTION SYSTEM")
    print("="*60)
    print("Starting in 2 seconds...")
    print("Features: Detection -> Analysis -> Maneuver + Matplotlib\n")
    time.sleep(2)
    
    # Initialize
    sensor = SensorUnit()
    trigger = TinyMLTrigger()
    ai = MissionAI()
    vis = SolarVisualizer()
    iteration = 0
    
    print("\n--- SCENARIO 1: Normal Monitoring ---\n")
    for i in range(4):
        iteration += 1
        data = sensor.generate(force_flare=False)
        print(f"[{iteration}] Flux: {data.xray_flux:.2e} ({data.flare_class.value})")
        trigger.check(data.xray_flux)
        vis.add(iteration, data.xray_flux, data.flare_class, trigger.power)
        time.sleep(SIMULATION_SPEED)
    
    print("\n--- SCENARIO 2: X-Class Flare! ---\n")
    for i in range(6):
        iteration += 1
        data = sensor.generate(force_flare=(i==2))
        print(f"[{iteration}] Flux: {data.xray_flux:.2e} ({data.flare_class.value})")
        
        exceeded = trigger.check(data.xray_flux)
        is_maneuver = False
        
        if exceeded:
            if ai.analyze(data):
                is_maneuver = True
        
        vis.add(iteration, data.xray_flux, data.flare_class, trigger.power, is_maneuver)
        time.sleep(SIMULATION_SPEED)
    
    print("\n--- SCENARIO 3: Post-Event ---\n")
    for i in range(3):
        iteration += 1
        data = sensor.generate(force_flare=False)
        print(f"[{iteration}] Flux: {data.xray_flux:.2e} ({data.flare_class.value})")
        trigger.check(data.xray_flux)
        vis.add(iteration, data.xray_flux, data.flare_class, trigger.power)
        time.sleep(SIMULATION_SPEED)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total iterations: {iteration}")
    print(f"AI analyses: {ai.analysis_count}")
    print(f"Protective maneuvers: {ai.maneuver_count}")
    print(f"Data points plotted: {len(vis.times)}")
    print("\nOpening visualization window...")
    print("Close the Matplotlib window to exit.\n")
    
    vis.show()


if __name__ == "__main__":
    run_demo()