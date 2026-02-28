"""
===============================================================================
HELIOS GUARD - Satellite Protection System MVP
===============================================================================
Aerospace Software Engineering Demo
TinyML-based Solar Storm Detection and Protective Maneuver System

Author: Aerospace Software Engineer & TinyML Expert
Purpose: Demonstration of Detection -> Analysis -> Maneuver cycle

Physics:
  - Distance: 1 AU = 150,000,000 km
  - Time of Arrival (ToA) = D / V
  - Fast CME velocity: ~2000 km/s

Energy Consumption:
  - Sleep Mode: 0.01W (TinyML gate active, low frequency)
  - AI Analysis: 2.0W (full neural network inference)
===============================================================================
"""

import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

# Physical constants
DISTANCE_AU_KM = 150_000_000  # 1 AU in km
X_FLARE_THRESHOLD = 1e-4      # W/m^2 - X-class flare threshold
M_FLARE_THRESHOLD = 1e-5      # W/m^2 - M-class flare threshold

# Energy consumption (Watts)
POWER_SLEEP = 0.01
POWER_AI_ANALYSIS = 2.0
POWER_STANDBY = 0.05

# Timing
SIMULATION_SPEED = 0.5  # seconds between iterations in demo
MONITORING_INTERVAL = 1  # seconds between sensor readings

# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class SystemState(Enum):
    """System operational states."""
    SLEEP = "SLEEP"
    MONITORING = "MONITORING"
    DETECTED = "DETECTED"
    ANALYZING = "ANALYZING"
    PROTECTIVE_MANEUVER = "PROTECTIVE_MANEUVER"
    SAFE_MODE = "SAFE_MODE"
    RECOVERY = "RECOVERY"


class FlareClass(Enum):
    """Solar flare classification."""
    A = "A"  # < 1e-7 W/m^2
    B = "B"  # 1e-7 to 1e-6 W/m^2
    C = "C"  # 1e-6 to 1e-5 W/m^2
    M = "M"  # 1e-5 to 1e-4 W/m^2
    X = "X"  # > 1e-4 W/m^2


@dataclass
class SolarWeatherData:
    """Solar weather sensor data."""
    timestamp: datetime
    xray_flux: float  # W/m^2
    flare_class: FlareClass
    cme_velocity: Optional[float] = None  # km/s
    proton_density: Optional[float] = None  # cm^-3
    kp_index: int = 0
    
    def __str__(self):
        return (f"[{self.timestamp.strftime('%H:%M:%S.%f')[:-3]}] "
                f"X-ray: {self.xray_flux:.2e} W/m^2 ({self.flare_class.value}) "
                f"| CME: {self.cme_velocity if self.cme_velocity else 'N/A'} km/s")


@dataclass
class ThreatAssessment:
    """Threat assessment result."""
    threat_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float
    estimated_velocity: float  # km/s
    time_of_arrival_hours: float
    recommended_action: str
    current_draw_watts: float


@dataclass
class SatelliteState:
    """Current satellite state."""
    power_available_watts: float = 100.0
    battery_charge_percent: float = 85.0
    shield_orientation: str = "SUN-FACING"
    last_maneuver: Optional[datetime] = None
    
    # Degradation tracking
    electronics_health: float = 100.0  # percent
    radiation_dose_gy: float = 0.0  # Gray


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging():
    """Configure logging with timestamps and formatting."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger('HeliosGuard')


# ============================================================================
# COMPONENT 1: SENSOR UNIT
# ============================================================================

class SensorUnit:
    """
    SensorUnit: Generates/reads simulated solar X-ray flux data.
    
    Simulates normal solar conditions and X-class flare events.
    Uses realistic solar cycle patterns.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.last_reading: Optional[SolarWeatherData] = None
        self.reading_count = 0
        
        # Solar activity probability (increases during demo)
        self.activity_level = 0.95  # High activity for demo purposes
        
    def generate_reading(self, force_flare: bool = False) -> SolarWeatherData:
        """
        Generate a single solar weather reading.
        
        Args:
            force_flare: If True, force an X-class flare event
            
        Returns:
            SolarWeatherData with current solar conditions
        """
        self.reading_count += 1
        timestamp = datetime.now()
        
        if force_flare or random.random() > self.activity_level:
            # Generate X-class flare (dangerous!)
            xray_flux = random.uniform(1e-4, 1e-3)
            flare_class = FlareClass.X
            cme_velocity = random.uniform(1500, 3000)  # Fast CME
            proton_density = random.uniform(20, 100)
            kp_index = random.randint(7, 9)
            
        elif random.random() > 0.7:
            # M-class flare (moderate)
            xray_flux = random.uniform(1e-5, 1e-4)
            flare_class = FlareClass.M
            cme_velocity = random.uniform(800, 1500)
            proton_density = random.uniform(10, 30)
            kp_index = random.randint(5, 7)
            
        elif random.random() > 0.5:
            # C-class flare (minor)
            xray_flux = random.uniform(1e-6, 1e-5)
            flare_class = FlareClass.C
            cme_velocity = random.uniform(400, 800)
            proton_density = random.uniform(5, 15)
            kp_index = random.randint(3, 5)
            
        else:
            # Quiet conditions
            xray_flux = random.uniform(1e-8, 1e-6)
            flare_class = FlareClass.A if xray_flux < 1e-7 else FlareClass.B
            cme_velocity = None
            proton_density = random.uniform(2, 8)
            kp_index = random.randint(0, 3)
        
        self.last_reading = SolarWeatherData(
            timestamp=timestamp,
            xray_flux=xray_flux,
            flare_class=flare_class,
            cme_velocity=cme_velocity,
            proton_density=proton_density,
            kp_index=kp_index
        )
        
        return self.last_reading
    
    def get_current_flux(self) -> float:
        """Get current X-ray flux reading."""
        if self.last_reading:
            return self.last_reading.xray_flux
        return 0.0


# ============================================================================
# COMPONENT 2: TinyML TRIGGER (LOW-POWER LOGIC GATE)
# ============================================================================

class TinyML_Trigger:
    """
    TinyML_Trigger: A low-power logic gate that stays in "Sleep Mode" 
    (low frequency) until a threshold is met.
    
    This simulates a TinyML model running on specialized hardware
    that activates the main AI only when necessary for power savings.
    """
    
    def __init__(self, logger: logging.Logger, threshold: float = X_FLARE_THRESHOLD):
        self.logger = logger
        self.threshold = threshold
        self.is_active = False
        self.trigger_count = 0
        self.current_power = POWER_SLEEP
        
    def check_threshold(self, xray_flux: float) -> bool:
        """
        Check if X-ray flux exceeds threshold.
        
        In real TinyML, this would be a lightweight model inference
        running on low-power microcontroller.
        
        Args:
            xray_flux: Current X-ray flux in W/m^2
            
        Returns:
            True if threshold exceeded (trigger AI analysis)
        """
        exceeded = xray_flux >= self.threshold
        
        if exceeded and not self.is_active:
            # Transition from sleep to active
            self.is_active = True
            self.trigger_count += 1
            self.current_power = POWER_AI_ANALYSIS
            
            self.logger.warning(
                f">>> THRESHOLD EXCEEDED! X-ray flux: {xray_flux:.2e} W/m^2 "
                f"(threshold: {self.threshold:.2e})"
            )
            self.logger.info(
                f">>> TinyML Trigger: Waking AI system (power: {POWER_SLEEP}W -> {POWER_AI_ANALYSIS}W)"
            )
            
        elif not exceeded and self.is_active:
            # Transition back to sleep
            self.is_active = False
            self.current_power = POWER_SLEEP
            
            self.logger.info(
                f">>> TinyML Trigger: Returning to sleep mode (power: {POWER_AI_ANALYSIS}W -> {POWER_SLEEP}W)"
            )
        
        return exceeded
    
    def get_power_consumption(self) -> float:
        """Get current power consumption in Watts."""
        return self.current_power


# ============================================================================
# COMPONENT 3: MISSION AI
# ============================================================================

class MissionAI:
    """
    MissionAI: The main logic that calculates "Time of Arrival" (ToA) 
    of protons and issues PROTECTIVE_MANEUVER commands.
    
    Physics-based calculation:
      T = D / V
      Where:
        D = 1 AU = 150,000,000 km
        V = CME velocity in km/s
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.analysis_count = 0
        self.maneuver_count = 0
        self.current_state = SystemState.SLEEP
        
    def analyze_threat(self, sensor_data: SolarWeatherData) -> ThreatAssessment:
        """
        Analyze solar weather data and calculate threat parameters.
        
        Args:
            sensor_data: Current solar weather readings
            
        Returns:
            ThreatAssessment with calculated ToA and recommendations
        """
        self.analysis_count += 1
        self.current_state = SystemState.ANALYZING
        
        self.logger.info(f"[AI ANALYSIS #{self.analysis_count}] Processing solar weather data...")
        self.logger.info(f"  Flare Class: {sensor_data.flare_class.value}")
        self.logger.info(f"  X-ray Flux: {sensor_data.xray_flux:.2e} W/m^2")
        
        # Determine threat level
        if sensor_data.flare_class == FlareClass.X:
            threat_level = "CRITICAL" if sensor_data.xray_flux > 5e-4 else "HIGH"
            confidence = 0.95
        elif sensor_data.flare_class == FlareClass.M:
            threat_level = "MEDIUM"
            confidence = 0.80
        elif sensor_data.flare_class == FlareClass.C:
            threat_level = "LOW"
            confidence = 0.60
        else:
            threat_level = "LOW"
            confidence = 0.90
            
        # Calculate velocity (use measured or estimate based on flare class)
        if sensor_data.cme_velocity:
            velocity = sensor_data.cme_velocity
        else:
            # Estimate based on flare intensity
            velocity_map = {FlareClass.X: 2000, FlareClass.M: 1200, 
                          FlareClass.C: 600, FlareClass.B: 400, FlareClass.A: 350}
            velocity = velocity_map.get(sensor_data.flare_class, 400)
        
        # Calculate Time of Arrival: T = D / V
        # D = 150,000,000 km, V in km/s, result in seconds
        time_seconds = DISTANCE_AU_KM / velocity
        time_hours = time_seconds / 3600
        
        # Add uncertainty margin
        uncertainty_factor = 1.2
        time_hours *= uncertainty_factor
        
        # Determine recommended action
        if threat_level == "CRITICAL":
            action = "IMMEDIATE_PROTECTIVE_MANEUVER"
        elif threat_level == "HIGH":
            action = "INITIATE_PROTECTIVE_MANEUVER"
        elif threat_level == "MEDIUM":
            action = "STANDBY_PROTECTION"
        else:
            action = "CONTINUE_MONITORING"
        
        self.logger.info(f"  CME Velocity: {velocity} km/s")
        self.logger.info(f"  Threat Level: {threat_level} (confidence: {confidence:.0%})")
        self.logger.info(f"  Time of Arrival: {time_hours:.1f} hours")
        
        return ThreatAssessment(
            threat_level=threat_level,
            confidence=confidence,
            estimated_velocity=velocity,
            time_of_arrival_hours=time_hours,
            recommended_action=action,
            current_draw_watts=POWER_AI_ANALYSIS
        )
    
    def issue_protective_command(
        self, 
        assessment: ThreatAssessment, 
        satellite: SatelliteState
    ) -> str:
        """
        Issue protective maneuver command based on threat assessment.
        
        Args:
            assessment: Current threat assessment
            satellite: Current satellite state
            
        Returns:
            Command string to execute
        """
        self.maneuver_count += 1
        self.current_state = SystemState.PROTECTIVE_MANEUVER
        
        command = f"PROTECTIVE_MANEUVER_{assessment.threat_level}"
        
        self.logger.warning(
            f"!!! COMMAND #{self.maneuver_count}: {command}"
        )
        self.logger.warning(
            f"!!! Executing protective maneuver..."
        )
        self.logger.warning(
            f"!!! Estimated time to impact: {assessment.time_of_arrival_hours:.1f} hours"
        )
        
        # Update satellite state
        satellite.last_maneuver = datetime.now()
        
        if assessment.threat_level in ["CRITICAL", "HIGH"]:
            satellite.battery_charge_percent = max(0, satellite.battery_charge_percent - 15)
            satellite.shield_orientation = "ANTI-SOLAR"
            
        return command
    
    def calculate_radiation_impact(
        self, 
        assessment: ThreatAssessment, 
        satellite: SatelliteState
    ) -> float:
        """
        Calculate estimated radiation dose based on threat.
        
        Args:
            assessment: Threat assessment
            satellite: Current satellite state
            
        Returns:
            Estimated radiation dose in Gray
        """
        # Simplified radiation model
        threat_multiplier = {
            "CRITICAL": 2.0,
            "HIGH": 1.0,
            "MEDIUM": 0.5,
            "LOW": 0.1
        }
        
        base_dose = 0.1  # Gray
        multiplier = threat_multiplier.get(assessment.threat_level, 0.1)
        
        # Factor in time to arrival (less time = less preparation)
        time_factor = max(0.5, 1.0 - (assessment.time_of_arrival_hours / 72))
        
        estimated_dose = base_dose * multiplier * time_factor
        satellite.radiation_dose_gy += estimated_dose
        
        # Degrade electronics based on total dose
        satellite.electronics_health = max(0, 100 - (satellite.radiation_dose_gy * 5))
        
        self.logger.info(
            f"  Estimated radiation dose: {estimated_dose:.3f} Gy "
            f"(total accumulated: {satellite.radiation_dose_gy:.3f} Gy)"
        )
        self.logger.info(
            f"  Electronics health: {satellite.electronics_health:.1f}%"
        )
        
        return estimated_dose


# ============================================================================
# MAIN CONTROLLER
# ============================================================================

class HeliosGuardSystem:
    """
    Main controller for Helios Guard satellite protection system.
    Orchestrates the Detection -> Analysis -> Maneuver cycle.
    """
    
    def __init__(self):
        self.logger = setup_logging()
        
        # Initialize components
        self.sensor = SensorUnit(self.logger)
        self.tinyml_trigger = TinyML_Trigger(self.logger)
        self.mission_ai = MissionAI(self.logger)
        self.satellite = SatelliteState()
        
        # System state
        self.system_state = SystemState.SLEEP
        self.is_running = False
        self.total_power_consumed = 0.0
        
    def print_banner(self):
        """Print system startup banner."""
        banner = """
+==============================================================================+
|                                                                              |
|   HELIOS GUARD - SATELLITE PROTECTION SYSTEM                                |
|   Aerospace Software Engineering Demo                                       |
|                                                                              |
+==============================================================================+
|  DETECTION ======> ANALYSIS ======> MANEUVER                                |
|  (TinyML Gate)    (Mission AI)      (Protective Action)                     |
+==============================================================================+
"""
        self.logger.info(banner)
        
    def start(self):
        """Start the Helios Guard system."""
        self.print_banner()
        
        self.logger.info("=" * 70)
        self.logger.info("HELIOS GUARD SYSTEM - INITIALIZATION")
        self.logger.info("=" * 70)
        
        self.logger.info(f"Distance to Sun: {DISTANCE_AU_KM:,} km (1 AU)")
        self.logger.info(f"X-class flare threshold: {X_FLARE_THRESHOLD:.2e} W/m^2")
        self.logger.info(f"Sleep mode power: {POWER_SLEEP}W")
        self.logger.info(f"AI Analysis power: {POWER_AI_ANALYSIS}W")
        
        self.logger.info("-" * 70)
        self.logger.info("System components initialized:")
        self.logger.info(f"  [✓] SensorUnit: X-ray flux monitor")
        self.logger.info(f"  [✓] TinyML_Trigger: Threshold gate (>{X_FLARE_THRESHOLD:.0e} W/m^2)")
        self.logger.info(f"  [✓] MissionAI: Threat analysis & maneuver control")
        self.logger.info("-" * 70)
        
        self.is_running = True
        self.logger.info("SYSTEM ONLINE - Entering MONITORING mode")
        self.logger.info("=" * 70)
        
    def run_detection_cycle(self, force_flare: bool = False) -> bool:
        """
        Run a single detection -> analysis -> maneuver cycle.
        
        Args:
            force_flare: If True, force an X-class flare event
            
        Returns:
            True if protective maneuver was issued
        """
        self.system_state = SystemState.MONITORING
        
        # Step 1: Read sensor data
        sensor_data = self.sensor.generate_reading(force_flare)
        self.logger.info(f"[MONITOR] {sensor_data}")
        
        # Step 2: TinyML trigger check (low power)
        threshold_exceeded = self.tinyml_trigger.check_threshold(sensor_data.xray_flux)
        
        # Track power consumption
        power = self.tinyml_trigger.get_power_consumption()
        self.total_power_consumed += power * SIMULATION_SPEED
        
        if threshold_exceeded:
            # Step 3: Full AI analysis
            assessment = self.mission_ai.analyze_threat(sensor_data)
            
            # Step 4: Calculate radiation impact
            self.mission_ai.calculate_radiation_impact(assessment, self.satellite)
            
            # Step 5: Issue protective command if needed
            if assessment.threat_level in ["HIGH", "CRITICAL"]:
                command = self.mission_ai.issue_protective_command(
                    assessment, self.satellite
                )
                self.system_state = SystemState.PROTECTIVE_MANEUVER
                return True
            else:
                self.logger.info(
                    f"[AI] Threat assessed as {assessment.threat_level} - "
                    f"continuing monitoring"
                )
                self.system_state = SystemState.MONITORING
                return False
        else:
            self.logger.debug(f"[SLEEP] Power consumption: {power}W")
            return False
            
    def print_summary(self):
        """Print system operation summary."""
        self.logger.info("=" * 70)
        self.logger.info("HELIOS GUARD SYSTEM - OPERATION SUMMARY")
        self.logger.info("=" * 70)
        self.logger.info(f"Total sensor readings: {self.sensor.reading_count}")
        self.logger.info(f"Total AI analyses: {self.mission_ai.analysis_count}")
        self.logger.info(f"Total maneuvers executed: {self.mission_ai.maneuver_count}")
        self.logger.info(f"Total power consumed: {self.total_power_consumed:.4f} Wh")
        self.logger.info(f"Final battery charge: {self.satellite.battery_charge_percent:.1f}%")
        self.logger.info(f"Final electronics health: {self.satellite.electronics_health:.1f}%")
        self.logger.info(f"Total radiation dose: {self.satellite.radiation_dose_gy:.3f} Gy")
        self.logger.info("=" * 70)
        self.logger.info("SYSTEM SHUTDOWN COMPLETE")
        self.logger.info("=" * 70)


# ============================================================================
# DEMO EXECUTION
# ============================================================================

def run_demo():
    """
    Run the complete Helios Guard demonstration.
    Shows the full Detection -> Analysis -> Maneuver cycle.
    """
    # Initialize system
    system = HeliosGuardSystem()
    system.start()
    
    print("\n" + "=" * 70)
    print("SCENARIO 1: Normal Solar Conditions (Monitoring Mode)")
    print("=" * 70 + "\n")
    
    # Scenario 1: Normal conditions (should stay in sleep mode)
    for i in range(3):
        system.run_detection_cycle(force_flare=False)
        time.sleep(SIMULATION_SPEED)
    
    print("\n" + "=" * 70)
    print("SCENARIO 2: X-Class Flare Detection (Defense Activation)")
    print("=" * 70 + "\n")
    
    # Scenario 2: X-class flare (should trigger protective maneuver)
    for i in range(5):
        maneuver_issued = system.run_detection_cycle(force_flare=(i == 2))
        if maneuver_issued:
            print("\n*** DEFENSE SYSTEM ACTIVATED ***\n")
        time.sleep(SIMULATION_SPEED)
    
    print("\n" + "=" * 70)
    print("SCENARIO 3: Post-Event Monitoring")
    print("=" * 70 + "\n")
    
    # Scenario 3: Return to normal
    for i in range(3):
        system.run_detection_cycle(force_flare=False)
        time.sleep(SIMULATION_SPEED)
    
    # Print summary
    system.print_summary()
    
    return system


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  HELIOS GUARD - Satellite Protection System MVP")
    print("#  Aerospace Software Engineering Demonstration")
    print("#" * 70 + "\n")
    
    print("Starting demonstration in 3 seconds...")
    print("This demo shows:")
    print("  1. TinyML gate staying in sleep mode (0.01W)")
    print("  2. X-class flare detection and threshold trigger")
    print("  3. AI analysis calculating Time of Arrival (T = D/V)")
    print("  4. Protective maneuver command execution")
    print("  5. Power consumption monitoring")
    print("")
    
    time.sleep(3)
    
    # Run the demo
    final_system = run_demo()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nTo run again: python main.py")
    print("\nKey Physics:")
    print(f"  - Distance: {DISTANCE_AU_KM:,} km (1 AU)")
    print(f"  - ToA Formula: T = D / V")
    print(f"  - Example: 150,000,000 km / 2000 km/s = {DISTANCE_AU_KM/2000/3600:.1f} hours")
    print("=" * 70 + "\n")

# 19:25:29 | [MONITOR] X-ray: 1.16e-05 W/m^2 (M) | CME: 1323 km/s
# 19:25:30 | >>> THRESHOLD EXCEEDED! X-ray flux: 6.94e-04 W/m^2
#         | >>> TinyML Trigger: Waking AI (0.01W → 2.0W)
# 19:25:30 | [AI ANALYSIS #1]
#         |   Flare Class: X
#         |   Threat Level: CRITICAL (95%)
#         |   Time of Arrival: 24.0 hours
# 19:25:30 | !!! COMMAND #1: PROTECTIVE_MANEUVER_CRITICAL
#         | !!! Estimated time to impact: 24.0 hours
# 19:25:30 | >>> TinyML Trigger: Sleep mode (2.0W → 0.01W)