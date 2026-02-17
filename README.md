# Blink-Analysis
Dry Eye Detection using Eye blink counter using OpenCV.

![Blink](https://github.com/masanbasa3k/Blink-Counter/assets/66223190/7990347f-f02f-4644-be9d-980c7a93369a)



# Explanation
This module implements a rule-based clinical classification system that evaluates dry eye risk using blink rate (blinks per minute) as a primary biometric indicator. The system processes the computed blink_rate_per_min and applies threshold-based conditional logic to categorize ocular health status. Blink rates between 15–20 blinks/min are classified as normal tear film dynamics. Rates between 10–14 blinks/min indicate reduced blinking typically associated with digital eye strain or mild evaporative dry eye. Values below 10 blinks/min are flagged as a strong indicator of dry eye due to increased tear film evaporation and prolonged inter-blink intervals. Additionally, blink rates exceeding 25 blinks/min are interpreted as reflex blinking caused by ocular surface irritation, which may also correlate with dry eye pathology. The classification output is programmatically written into a structured clinical report file, enabling automated documentation within a computer vision–based ocular health monitoring pipeline.



<img width="800" height="667" alt="image" src="https://github.com/user-attachments/assets/eba1c9fb-9121-4e37-85c5-e003a4edb679" />
<img width="645" height="453" alt="image" src="https://github.com/user-attachments/assets/22d9e3c0-4842-4a6f-9997-cb9e76774c97" />








