import cv2
import time
from datetime import datetime
import numpy as np
import pandas as pd
from collections import deque
from faceMeshModule import FaceMeshDetector
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
import signal
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "recording_duration": 60,  # seconds
    "ear_threshold": 0.2,  # Eye Aspect Ratio threshold for blink detection
    "ear_frame_buffer": 3,  # Consecutive frames below threshold for blink detection
    "min_blink_duration": 0.05,  # Minimum blink duration in seconds
    "max_blink_duration": 1.0,  # Maximum normal blink duration in seconds
    "video_width": 640,
    "video_height": 480,
    "fps_fallback": 30.0,
    "ear_threshold_squared": 0.04,  # Pre-calculated for efficiency
    "partial_blink_threshold": 0.1,  # If EAR never goes below this, it's partial
}

# Eye landmark indices (using MediaPipe format)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # 6 points for EAR calculation
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def calculate_ear(eye_landmarks):
    """Calculate Eye Aspect Ratio for given eye landmarks"""
    # Vertical distances
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    
    # Horizontal distance
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear

def get_status_color(ear, threshold):
    """Get color based on EAR status"""
    if ear > threshold:
        return (0, 255, 0)  # Green - open
    elif ear > threshold * 0.7:
        return (0, 255, 255)  # Yellow - partially closed
    else:
        return (0, 0, 255)  # Red - closed

def validate_blink_event(event):
    """Validate blink event data"""
    required_fields = ['timestamp', 'duration', 'min_EAR']
    for field in required_fields:
        if field not in event or event[field] is None:
            return False
    
    # Check for reasonable values
    if not (CONFIG["min_blink_duration"] <= event['duration'] <= CONFIG["max_blink_duration"]):
        return False
    if not (0 <= event['min_EAR'] <= 1):
        return False
    
    return True

def signal_handler(sig, frame):
    """Handle interrupt signals gracefully"""
    print("\n\nInterrupt received, saving data and exiting...")
    global is_recording
    is_recording = False

# ============================================================================
# INITIALIZATION
# ============================================================================
# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["video_width"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["video_height"])
cap.set(cv2.CAP_PROP_FPS, 30)

# Get actual FPS
actual_fps = cap.get(cv2.CAP_PROP_FPS)
if actual_fps <= 0:
    actual_fps = CONFIG["fps_fallback"]
    print(f"⚠️  Could not determine FPS, using fallback: {actual_fps}")

detector = FaceMeshDetector(maxFaces=1)

# Blink tracking variables
blink_count = 0
total_frames = 0
blink_data = []  # List to store all blink events
current_blink = {
    "start_time": None,
    "start_frame": None,
    "min_EAR": 1.0,
    "left_EAR_start": 1.0,
    "right_EAR_start": 1.0
}

# Use separate counters for blink detection and PERCLOS
blink_closed_frames = 0  # For blink duration
perclos_closed_frames = 0  # For PERCLOS calculation

frame_buffer = deque(maxlen=CONFIG["ear_frame_buffer"])  # For EAR smoothing

# Metrics storage
metrics = {
    "blink_rate": 0,
    "inter_blink_intervals": [],
    "blink_durations": [],
    "perclos": 0,
    "blink_amplitudes": [],
    "blink_velocities": [],
    "partial_blink_ratio": 0,
    "blink_symmetry": [],
    "blink_consistency": 0,
    "fatigue_score": 0
}

# Timing and recording
start_time = time.time()
is_recording = True
last_blink_time = None

# Get patient name
patient_name = input("Enter patient name: ").strip()
if not patient_name:
    patient_name = "Unknown"

# Sanitize patient name for filenames
patient_name_safe = patient_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print("=" * 60)
print("COMPREHENSIVE BLINK ANALYSIS SYSTEM")
print("=" * 60)
print(f"Recording for {CONFIG['recording_duration']} seconds...")
print("Press 'q' to quit early")
print("Press Ctrl+C to save and exit")
print(f"Patient: {patient_name}")
print(f"FPS: {actual_fps:.1f}")
print("-" * 60)

# Setup logging
logging.basicConfig(
    filename=f'blink_analysis_{patient_name_safe}_{timestamp}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info(f"Starting analysis for {patient_name}")

# Video recording
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_filename = f"blink_analysis_{patient_name_safe}_{timestamp}.avi"
out = cv2.VideoWriter(video_filename, fourcc, actual_fps, 
                     (CONFIG["video_width"], CONFIG["video_height"]))

# CSV file for detailed data
csv_filename = f"blink_data_{patient_name_safe}_{timestamp}.csv"
csv_data = []

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================
try:
    while is_recording:
        success, img = cap.read()
        if not success:
            print("ERROR: Could not read frame from webcam!")
            logging.error("Could not read frame from webcam")
            break
        
        # Flip for mirror effect
        img = cv2.flip(img, 1)
        out.write(img)
        
        total_frames += 1
        current_time = time.time() - start_time
        frame_time = time.time()
        
        # Find face mesh
        try:
            img, faces = detector.findFaceMesh(img, draw=False)
        except Exception as e:
            print(f"Face detection error: {e}")
            logging.error(f"Face detection error: {e}")
            continue
        
        ear_left = 1.0
        ear_right = 1.0
        ear_avg = 1.0
        
        if faces:
            face = faces[0]
            
            # Extract eye landmarks
            left_eye = np.array([face[idx] for idx in LEFT_EYE_INDICES])
            right_eye = np.array([face[idx] for idx in RIGHT_EYE_INDICES])
            
            # Calculate EAR for both eyes
            ear_left = calculate_ear(left_eye)
            ear_right = calculate_ear(right_eye)
            ear_avg = (ear_left + ear_right) / 2.0
            
            # Draw eye landmarks
            for point in left_eye:
                cv2.circle(img, tuple(point.astype(int)), 2, (0, 255, 0), -1)
            for point in right_eye:
                cv2.circle(img, tuple(point.astype(int)), 2, (0, 255, 0), -1)
            
            # Add to frame buffer for smoothing
            frame_buffer.append(ear_avg)
            smoothed_ear = np.mean(list(frame_buffer)) if frame_buffer else ear_avg
            
            # PERCLOS calculation
            if ear_avg < CONFIG["ear_threshold"]:
                perclos_closed_frames += 1
            
            # Check for blink using smoothed EAR
            if smoothed_ear < CONFIG["ear_threshold"]:
                blink_closed_frames += 1
                
                # Start of a new blink
                if current_blink["start_time"] is None:
                    current_blink["start_time"] = frame_time
                    current_blink["start_frame"] = total_frames
                    current_blink["min_EAR"] = smoothed_ear
                    current_blink["left_EAR_start"] = ear_left
                    current_blink["right_EAR_start"] = ear_right
                else:
                    # Update minimum EAR during blink
                    current_blink["min_EAR"] = min(current_blink["min_EAR"], smoothed_ear)
            
            # End of blink detection
            elif current_blink["start_time"] is not None:
                blink_duration = frame_time - current_blink["start_time"]
                
                # Only count as blink if duration is within reasonable bounds
                if (CONFIG["min_blink_duration"] <= blink_duration <= 
                    CONFIG["max_blink_duration"]):
                    
                    # Calculate blink metrics
                    blink_amplitude = 1.0 - current_blink["min_EAR"]  # 1.0 is fully open
                    
                    # Calculate velocities
                    closing_time = blink_closed_frames / actual_fps if blink_closed_frames > 0 else 0.001
                    opening_time = max(blink_duration - closing_time, 0.001)
                    closing_velocity = (1.0 - current_blink["min_EAR"]) / closing_time if closing_time > 0 else 0
                    opening_velocity = (1.0 - current_blink["min_EAR"]) / opening_time if opening_time > 0 else 0
                    
                    # Determine if partial blink
                    is_partial = current_blink["min_EAR"] > CONFIG["partial_blink_threshold"]
                    
                    # Calculate symmetry
                    symmetry_diff = abs(ear_left - ear_right) / max(ear_left, ear_right) if max(ear_left, ear_right) > 0 else 0
                    
                    # Create blink event
                    blink_event = {
                        "timestamp": current_time,
                        "duration": blink_duration,
                        "amplitude": blink_amplitude,
                        "min_EAR": current_blink["min_EAR"],
                        "closing_velocity": closing_velocity,
                        "opening_velocity": opening_velocity,
                        "is_partial": is_partial,
                        "symmetry": symmetry_diff,
                        "closed_frames": blink_closed_frames
                    }
                    
                    # Validate and store
                    if validate_blink_event(blink_event):
                        blink_data.append(blink_event)
                        
                        # Store for CSV
                        csv_data.append({
                            "timestamp": current_time,
                            "blink_duration": blink_duration,
                            "amplitude": blink_amplitude,
                            "min_EAR": current_blink["min_EAR"],
                            "closing_velocity": closing_velocity,
                            "opening_velocity": opening_velocity,
                            "is_partial": is_partial,
                            "symmetry": symmetry_diff,
                            "ear_left": ear_left,
                            "ear_right": ear_right,
                            "ear_avg": ear_avg,
                            "frame_number": total_frames,
                            "blink_count": blink_count + 1
                        })
                        
                        # Update blink count
                        blink_count += 1
                        
                        # Calculate inter-blink interval
                        if last_blink_time is not None:
                            ibi = current_time - last_blink_time
                            metrics["inter_blink_intervals"].append(ibi)
                        last_blink_time = current_time
                        
                        # Update durations and amplitudes
                        metrics["blink_durations"].append(blink_duration)
                        metrics["blink_amplitudes"].append(blink_amplitude)
                        metrics["blink_velocities"].append((closing_velocity + opening_velocity) / 2)
                        metrics["blink_symmetry"].append(symmetry_diff)
                
                # Reset for next blink
                current_blink = {"start_time": None, "start_frame": None, "min_EAR": 1.0}
                blink_closed_frames = 0
        
        # Display information
        elapsed_time = int(current_time)
        remaining_time = max(0, CONFIG["recording_duration"] - elapsed_time)
        
        # Display header
        cv2.putText(img, f"Time: {elapsed_time}s / {CONFIG['recording_duration']}s", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(img, f"Blinks: {blink_count}", 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display EAR with color coding
        ear_color = get_status_color(ear_avg, CONFIG["ear_threshold"])
        cv2.putText(img, f"EAR: {ear_avg:.3f}", 
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 2)
        
        # Display real-time metrics
        metrics_text = [
            f"Blink Rate: {blink_count/max(1, elapsed_time)*60:.1f}/min",
            f"PERCLOS: {(perclos_closed_frames/total_frames*100):.1f}%" if total_frames > 0 else "PERCLOS: N/A"
        ]
        
        if metrics["blink_durations"]:
            metrics_text.append(f"Avg Duration: {np.mean(metrics['blink_durations'])*1000:.0f}ms")
        
        for i, text in enumerate(metrics_text):
            cv2.putText(img, text, (400, 30 + i*30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Display EAR threshold line and indicator
        cv2.line(img, (0, 300), (CONFIG["video_width"], 300), (100, 100, 100), 1)
        ear_bar = int(ear_avg * 100)
        ear_bar = max(0, min(100, ear_bar))  # Clamp to 0-100 range
        cv2.rectangle(img, (10, 300 - ear_bar), (30, 300), 
                     ear_color, -1)
        cv2.putText(img, "EAR", (5, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Draw threshold line on EAR bar
        threshold_bar = int(CONFIG["ear_threshold"] * 100)
        cv2.line(img, (5, 300 - threshold_bar), (35, 300 - threshold_bar), 
                (0, 0, 0), 1)
        
        # Show frame
        cv2.imshow('Comprehensive Blink Analysis - Press Q to quit', img)
        
        # Check recording duration
        if elapsed_time >= CONFIG["recording_duration"]:
            is_recording = False
            print("\nRecording completed!")
            logging.info("Recording completed normally")
        
        # Check for early quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nRecording stopped early by user.")
            logging.info("Recording stopped early by user")
            is_recording = False
            break

except Exception as e:
    print(f"Error during processing: {e}")
    logging.error(f"Processing error: {e}")

finally:
    # ============================================================================
    # RELEASE RESOURCES
    # ============================================================================
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Resources released")

# ============================================================================
# CALCULATE ALL METRICS
# ============================================================================
recording_time = min(elapsed_time, CONFIG["recording_duration"])

# 1. Blink Frequency / Blink Rate
if recording_time > 0:
    metrics["blink_rate"] = (blink_count / recording_time) * 60  # blinks per minute

# 2. Inter-Blink Interval (IBI) Statistics
if metrics["inter_blink_intervals"]:
    metrics["mean_IBI"] = np.mean(metrics["inter_blink_intervals"])
    metrics["std_IBI"] = np.std(metrics["inter_blink_intervals"])
    metrics["cv_IBI"] = metrics["std_IBI"] / metrics["mean_IBI"] if metrics["mean_IBI"] > 0 else 0
else:
    metrics["mean_IBI"] = metrics["std_IBI"] = metrics["cv_IBI"] = 0

# 3. Blink Duration Statistics
if metrics["blink_durations"]:
    metrics["mean_duration"] = np.mean(metrics["blink_durations"])
    metrics["std_duration"] = np.std(metrics["blink_durations"])
else:
    metrics["mean_duration"] = metrics["std_duration"] = 0

# 4. Eye Closure Percentage (PERCLOS)
if total_frames > 0:
    metrics["perclos"] = (perclos_closed_frames / total_frames) * 100

# 5. Blink Amplitude Statistics
if metrics["blink_amplitudes"]:
    metrics["mean_amplitude"] = np.mean(metrics["blink_amplitudes"])
    metrics["std_amplitude"] = np.std(metrics["blink_amplitudes"])
else:
    metrics["mean_amplitude"] = metrics["std_amplitude"] = 0

# 6. Blink Velocity Statistics
if metrics["blink_velocities"]:
    metrics["mean_velocity"] = np.mean(metrics["blink_velocities"])
    metrics["std_velocity"] = np.std(metrics["blink_velocities"])
else: 
    metrics["mean_velocity"] = metrics["std_velocity"] = 0

# 7. Partial Blink Ratio
if blink_count > 0:
    partial_blinks = sum(1 for blink in blink_data if blink.get("is_partial", False))
    metrics["partial_blink_ratio"] = (partial_blinks / blink_count) * 100

# 8. Blink Symmetry Statistics
if metrics["blink_symmetry"]:
    metrics["mean_symmetry"] = np.mean(metrics["blink_symmetry"])
    metrics["std_symmetry"] = np.std(metrics["blink_symmetry"])
else:
    metrics["mean_symmetry"] = metrics["std_symmetry"] = 0

# 9. Blink Pattern Consistency (using Coefficient of Variation of IBI)
metrics["blink_consistency"] = metrics["cv_IBI"]

# 10. Fatigue Score (composite metric)
# Higher score indicates more fatigue
fatigue_score = 0
if blink_count > 0:
    # Based on PERCLOS, blink rate, and duration
    if metrics["perclos"] > 20:
        fatigue_score += 3
    elif metrics["perclos"] > 10:
        fatigue_score += 1
    
    if metrics["blink_rate"] < 5:  # Low blink rate during screen use
        fatigue_score += 2
    elif metrics["blink_rate"] > 25:  # High blink rate (reflex blinking)
        fatigue_score += 1
    
    if metrics["mean_duration"] > 0.3:  # Long blink duration
        fatigue_score += 2
    
    if metrics["partial_blink_ratio"] > 40:  # High partial blink ratio
        fatigue_score += 1
    
    metrics["fatigue_score"] = min(fatigue_score, 10)  # Scale to 0-10

# ============================================================================
# DISPLAY RESULTS
# ============================================================================
print("\n" + "=" * 60)
print("COMPREHENSIVE BLINK ANALYSIS RESULTS")
print("=" * 60)

print(f"\n1. BLINK FREQUENCY:")
print(f"   • Blink Rate: {metrics['blink_rate']:.1f} blinks/minute")
print(f"   • Total Blinks: {blink_count}")

print(f"\n2. INTER-BLINK INTERVAL (IBI):")
print(f"   • Mean IBI: {metrics['mean_IBI']:.2f} seconds")
print(f"   • Std Dev IBI: {metrics['std_IBI']:.2f} seconds")
print(f"   • CV IBI: {metrics['cv_IBI']:.3f}")

print(f"\n3. BLINK DURATION:")
print(f"   • Mean Duration: {metrics['mean_duration']*1000:.1f} ms")
print(f"   • Std Dev Duration: {metrics['std_duration']*1000:.1f} ms")

print(f"\n4. EYE CLOSURE PERCENTAGE (PERCLOS):")
print(f"   • PERCLOS: {metrics['perclos']:.1f}%")

print(f"\n5. BLINK AMPLITUDE:")
print(f"   • Mean Amplitude: {metrics['mean_amplitude']:.3f}")
print(f"   • Std Dev Amplitude: {metrics['std_amplitude']:.3f}")

print(f"\n6. BLINK VELOCITY:")
print(f"   • Mean Velocity: {metrics['mean_velocity']:.1f} EAR units/second")

print(f"\n7. PARTIAL BLINK RATIO:")
print(f"   • Partial Blinks: {metrics['partial_blink_ratio']:.1f}%")

print(f"\n8. BLINK SYMMETRY:")
print(f"   • Mean Symmetry Diff: {metrics['mean_symmetry']:.3f}")

print(f"\n9. BLINK PATTERN CONSISTENCY:")
print(f"   • CV of IBI: {metrics['blink_consistency']:.3f}")
if metrics['blink_consistency'] < 0.3:
    print(f"   • Interpretation: Consistent pattern (low variability)")
else:
    print(f"   • Interpretation: Variable pattern (high variability)")

print(f"\n10. FATIGUE & ALERTNESS:")
print(f"   • Fatigue Score: {metrics['fatigue_score']}/10")
if metrics['fatigue_score'] <= 3:
    print(f"   • Interpretation: Alert")
elif metrics['fatigue_score'] <= 6:
    print(f"   • Interpretation: Mild fatigue")
else:
    print(f"   • Interpretation: Significant fatigue")

# ============================================================================
# CLINICAL INTERPRETATION
# ============================================================================
print(f"\n" + "=" * 60)
print("CLINICAL INTERPRETATION")
print("=" * 60)

# Dry Eye Assessment
dry_eye_indicators = 0
print(f"\nDRY EYE SCREENING:")
print(f"   • Blink Rate: {metrics['blink_rate']:.1f} blinks/min")
if metrics['blink_rate'] < 5:
    print(f"     [✗] SIGNIFICANT: Very low blink rate (< 5/min)")
    dry_eye_indicators += 2
elif metrics['blink_rate'] < 10:
    print(f"     [✗] CONCERN: Low blink rate (< 10/min)")
    dry_eye_indicators += 1
elif metrics['blink_rate'] > 25:
    print(f"     [✗] CONCERN: High blink rate (> 25/min) - reflex blinking")
    dry_eye_indicators += 1
else:
    print(f"     [✓] NORMAL: Normal blink rate")

print(f"   • Partial Blink Ratio: {metrics['partial_blink_ratio']:.1f}%")
if metrics['partial_blink_ratio'] > 50:
    print(f"     [✗] SIGNIFICANT: High partial blink ratio (> 50%)")
    dry_eye_indicators += 2
elif metrics['partial_blink_ratio'] > 30:
    print(f"     [✗] CONCERN: Elevated partial blink ratio (> 30%)")
    dry_eye_indicators += 1
else:
    print(f"     [✓] NORMAL: Normal partial blink ratio")

print(f"   • Blink Duration: {metrics['mean_duration']*1000:.1f} ms")
if metrics['mean_duration'] > 0.4:
    print(f"     [✗] CONCERN: Long blink duration (> 400 ms)")
    dry_eye_indicators += 1
else:
    print(f"     [✓] NORMAL: Normal blink duration")

# Neurological Indicators
print(f"\nNEUROLOGICAL INDICATORS:")
print(f"   • Blink Symmetry: {metrics['mean_symmetry']:.3f}")
if metrics['mean_symmetry'] > 0.2:
    print(f"     [✗] CONCERN: High asymmetry (> 0.2)")
else:
    print(f"     [✓] NORMAL: Good symmetry")

print(f"   • Blink Velocity: {metrics['mean_velocity']:.1f} EAR/sec")
if metrics['mean_velocity'] < 2.0:
    print(f"     [✗] CONCERN: Slow blink velocity (< 2.0 EAR/sec)")
else:
    print(f"     [✓] NORMAL: Normal blink velocity")

# Fatigue Assessment
print(f"\nFATIGUE ASSESSMENT:")
print(f"   • PERCLOS: {metrics['perclos']:.1f}%")
if metrics['perclos'] > 15:
    print(f"     [✗] CONCERN: High eye closure (> 15%)")
    print(f"        Possible drowsiness or fatigue")
elif metrics['perclos'] > 8:
    print(f"     [!] CAUTION: Moderate eye closure (> 8%)")
else:
    print(f"     [✓] NORMAL: Normal eye closure")

print(f"   • Pattern Consistency (CV): {metrics['blink_consistency']:.3f}")
if metrics['blink_consistency'] > 0.5:
    print(f"     [✗] CONCERN: Irregular blink pattern")
    print(f"        Possible cognitive load or stress")
else:
    print(f"     [✓] NORMAL: Regular blink pattern")

# ============================================================================
# SAVE RESULTS
# ============================================================================
# Save detailed data to CSV
if csv_data:
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_filename, index=False)
    print(f"\n[✓] Detailed blink data saved to: {csv_filename}")
    logging.info(f"CSV data saved to {csv_filename}")

# Save summary report
report_filename = f"blink_report_{patient_name_safe}_{timestamp}.txt"
try:
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("COMPREHENSIVE BLINK ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Patient Name: {patient_name}\n")
        f.write(f"Recording Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {recording_time} seconds\n")
        f.write(f"Total Frames: {total_frames}\n")
        f.write(f"Video File: {video_filename}\n")
        f.write(f"Data File: {csv_filename}\n")
        f.write(f"FPS: {actual_fps:.1f}\n\n")
        
        f.write("SUMMARY METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Blink Rate: {metrics['blink_rate']:.1f} blinks/min\n")
        f.write(f"Total Blinks: {blink_count}\n")
        f.write(f"Mean Blink Duration: {metrics['mean_duration']*1000:.1f} ms\n")
        f.write(f"PERCLOS: {metrics['perclos']:.1f}%\n")
        f.write(f"Partial Blink Ratio: {metrics['partial_blink_ratio']:.1f}%\n")
        f.write(f"Mean Inter-Blink Interval: {metrics['mean_IBI']:.2f} s\n")
        f.write(f"Blink Pattern Consistency (CV): {metrics['blink_consistency']:.3f}\n")
        f.write(f"Fatigue Score: {metrics['fatigue_score']}/10\n\n")
        
        f.write("CLINICAL ASSESSMENT:\n")
        f.write("-" * 40 + "\n")
        if dry_eye_indicators >= 3:
            f.write("[✗] SIGNIFICANT DRY EYE INDICATORS DETECTED\n")
            f.write("    Recommendation: Consult ophthalmologist\n")
        elif dry_eye_indicators >= 1:
            f.write("[!] MILD DRY EYE INDICATORS DETECTED\n")
            f.write("    Recommendation: Monitor symptoms, use artificial tears\n")
        else:
            f.write("[✓] NO SIGNIFICANT DRY EYE INDICATORS\n")
        
        if metrics['fatigue_score'] >= 7:
            f.write("[✗] SIGNIFICANT FATIGUE INDICATORS DETECTED\n")
            f.write("    Recommendation: Rest, avoid driving/operating machinery\n")
        elif metrics['fatigue_score'] >= 4:
            f.write("[!] MILD FATIGUE INDICATORS DETECTED\n")
            f.write("    Recommendation: Take breaks, ensure proper lighting\n")
        else:
            f.write("[✓] ALERT STATE MAINTAINED\n")
    
    print(f"[✓] Analysis report saved to: {report_filename}")
    logging.info(f"Report saved to {report_filename}")
    
except Exception as e:
    print(f"[!] Error saving report: {e}")
    logging.error(f"Error saving report: {e}")

print(f"[✓] Video saved to: {video_filename}")
logging.info(f"Video saved to {video_filename}")

# Generate PDF report with charts
try:
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f'Blink Analysis Report - {patient_name}', fontsize=16, fontweight='bold')
    
    if blink_data:
        # Chart 1: Blink Timeline
        ax1 = plt.subplot(2, 2, 1)
        times = [b.get('timestamp', 0) for b in blink_data]
        if times:
            ax1.plot(times, range(1, len(times)+1), 'b-o', markersize=3)
            ax1.set_title('Cumulative Blinks Over Time')
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Blink Count')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No timestamp data', 
                    horizontalalignment='center', verticalalignment='center')
            ax1.axis('off')
        
        # Chart 2: Blink Duration Distribution
        ax2 = plt.subplot(2, 2, 2)
        if metrics.get('blink_durations'):
            durations = [d for d in metrics['blink_durations'] if d is not None]
            if durations:
                ax2.hist(durations, bins=20, alpha=0.7, color='green', edgecolor='black')
                mean_dur = np.mean(durations)
                ax2.axvline(mean_dur, color='red', linestyle='--', 
                           label=f'Mean: {mean_dur*1000:.1f}ms')
                ax2.set_title('Blink Duration Distribution')
                ax2.set_xlabel('Duration (seconds)')
                ax2.set_ylabel('Frequency')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No duration data', 
                        horizontalalignment='center', verticalalignment='center')
                ax2.axis('off')
        else:
            ax2.text(0.5, 0.5, 'No duration data', 
                    horizontalalignment='center', verticalalignment='center')
            ax2.axis('off')
        
        # Chart 3: Inter-Blink Intervals
        ax3 = plt.subplot(2, 2, 3)
        if metrics.get('inter_blink_intervals'):
            intervals = [i for i in metrics['inter_blink_intervals'] if i is not None]
            if intervals:
                ax3.plot(intervals, 'r-', marker='o', markersize=3)
                mean_int = np.mean(intervals)
                ax3.axhline(mean_int, color='blue', linestyle='--', 
                           label=f'Mean: {mean_int:.1f}s')
                ax3.set_title('Inter-Blink Intervals')
                ax3.set_xlabel('Blink Number')
                ax3.set_ylabel('Interval (seconds)')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No interval data', 
                        horizontalalignment='center', verticalalignment='center')
                ax3.axis('off')
        else:
            ax3.text(0.5, 0.5, 'No interval data', 
                    horizontalalignment='center', verticalalignment='center')
            ax3.axis('off')
        
        # Chart 4: Clinical Summary
        ax4 = plt.subplot(2, 2, 4)
        metrics_values = [
            metrics.get('blink_rate', 0),
            metrics.get('perclos', 0),
            metrics.get('partial_blink_ratio', 0),
            metrics.get('mean_duration', 0) * 1000,
            metrics.get('fatigue_score', 0)
        ]
        metric_names = ['Blink Rate\n(blinks/min)', 'PERCLOS\n(%)', 
                       'Partial Blinks\n(%)', 'Mean Duration\n(ms)', 'Fatigue\nScore']
        
        bars = ax4.bar(range(len(metrics_values)), metrics_values, 
                      color=['blue', 'orange', 'green', 'red', 'purple'])
        ax4.set_title('Key Metrics Summary')
        ax4.set_xticks(range(len(metrics_values)))
        ax4.set_xticklabels(metric_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    else:
        # No data case
        ax = plt.subplot(1, 1, 1)
        ax.text(0.5, 0.5, 'No Blink Data Collected\nDuring Recording', 
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save PDF with multiple pages
    pdf_filename = f"blink_analysis_{patient_name_safe}_{timestamp}.pdf"
    
    # Create a multi-page PDF
    from matplotlib.backends.backend_pdf import PdfPages
    
    with PdfPages(pdf_filename) as pdf:
        # Page 1: Charts
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Clinical Results and Interpretation
        fig2 = plt.figure(figsize=(12, 10))
        fig2.suptitle(f'Clinical Results & Interpretation - {patient_name}', fontsize=16, fontweight='bold')
        
        # Remove axes
        ax = fig2.add_subplot(111)
        ax.axis('off')
        
        # Prepare clinical text
        clinical_text = f"""
PATIENT INFORMATION:
Name: {patient_name}
Date: {timestamp}
Recording Duration: {CONFIG['recording_duration']} seconds

QUANTITATIVE METRICS:
• Blink Rate: {metrics.get('blink_rate', 0):.1f} blinks/minute
• Mean Blink Duration: {metrics.get('mean_duration', 0)*1000:.1f} ms
• PERCLOS (Eye closure percentage): {metrics.get('perclos', 0):.1f}%
• Partial Blink Ratio: {metrics.get('partial_blink_ratio', 0):.1f}%
• Mean Inter-Blink Interval: {metrics.get('mean_IBI', 0):.2f} seconds
• Blink Consistency (CV): {metrics.get('blink_consistency', 0):.3f}
• Fatigue Score: {metrics.get('fatigue_score', 0)}/10

CLINICAL ASSESSMENT:
"""
        
        # Dry Eye Assessment
        dry_eye_indicators = 0
        if metrics.get('perclos', 0) > 20:
            dry_eye_indicators += 1
        if metrics.get('mean_duration', 0) < 0.1:
            dry_eye_indicators += 1
        if metrics.get('blink_rate', 0) < 12:
            dry_eye_indicators += 1
        
        if dry_eye_indicators >= 3:
            clinical_text += "\n[✗] SIGNIFICANT DRY EYE INDICATORS DETECTED"
            clinical_text += "\n    Recommendation: Consult ophthalmologist for comprehensive evaluation"
        elif dry_eye_indicators >= 1:
            clinical_text += "\n[!] MILD DRY EYE INDICATORS DETECTED"
            clinical_text += "\n    Recommendation: Monitor symptoms, use artificial tears as needed"
        else:
            clinical_text += "\n[✓] NO SIGNIFICANT DRY EYE INDICATORS"
        
        # Fatigue Assessment
        if metrics.get('fatigue_score', 0) >= 7:
            clinical_text += "\n\n[✗] SIGNIFICANT FATIGUE INDICATORS DETECTED"
            clinical_text += "\n    Recommendation: Rest immediately, avoid driving/operating machinery"
        elif metrics.get('fatigue_score', 0) >= 4:
            clinical_text += "\n\n[!] MILD FATIGUE INDICATORS DETECTED"
            clinical_text += "\n    Recommendation: Take frequent breaks, ensure proper lighting and screen positioning"
        else:
            clinical_text += "\n\n[✓] ALERT STATE MAINTAINED"
        
        clinical_text += "\n\nINTERPRETATION GUIDE:\n"
        clinical_text += "• Blink Rate: Normal range is 12-20 blinks/min. Reduced rate may indicate fatigue or concentration.\n"
        clinical_text += "• PERCLOS: Percentage of time eyelids are closed. >20% suggests fatigue or drowsiness.\n"
        clinical_text += "• Blink Duration: Normal range is 100-150ms. Reduced duration may indicate dry eyes.\n"
        clinical_text += "• Fatigue Score: Composite measure (0-10) based on blink metrics. >7 suggests significant fatigue.\n"
        
        # Add text to figure
        ax.text(0.05, 0.95, clinical_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Save second page
        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)
    
    print(f"[✓] Multi-page PDF report saved to: {pdf_filename}")
    logging.info(f"Multi-page PDF report saved to {pdf_filename}")

except Exception as e:
    print(f"[!] Error generating PDF report: {e}")
    logging.error(f"Error generating PDF report: {e}")