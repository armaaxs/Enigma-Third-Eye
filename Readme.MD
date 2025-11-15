## Overview
Enigma — Third Eye is a lightweight vision-assisted Python application designed to run on embedded computing hardware. The system combines a Raspberry Pi 4B, an attached camera module, and Python-based processing logic to enable real-time video capture and on-device inference. This document summarizes the hardware stack, software components, execution pipeline, and recommended engineering practices.

---

## Hardware Architecture

### Primary Compute Unit (Raspberry Pi 4B)
The system is deployed on a **Raspberry Pi 4 Model B**, selected for its balance between computational capacity, cost, and power efficiency. Relevant characteristics:

- Quad-core ARM Cortex-A72 CPU suitable for real-time Python workloads.
- 2–8GB RAM variants allowing flexible model and buffer sizes.
- On-board Wi-Fi and Bluetooth for wireless streaming, remote monitoring, or telemetry.
- USB 3.0 ports for potential external sensors, storage devices, or accelerators.
- 40-pin GPIO header enabling future hardware integrations (IMUs, audio modules, LEDs).

### Camera Module
A compact **Logitech C270HD Webcam** is mounted on the wearable frame. It is responsible for:

- Continuous video capture at configurable FPS.
- Low-latency frame delivery to the Python application.
- Adjustable resolution for performance trade-offs (e.g., 640×480 for fast inference).

### Wearable Mounting Setup
The application is designed with a lightweight, portable form factor:

- Camera clipped to the right side of the glasses frame for natural forward-view capture.
- Raspberry Pi secured at the back of the neck (inside clothing or on a harness).
- Optional micro-speaker on left side for audio feedback (alerts, TTS, or status signals).
- Power supplied via compact Li-ion pack or USB power bank.

This hardware arrangement minimizes obstruction while providing continuous real-world video input.

---

## Software Stack

### Language & Runtime
The project is written in **Python**, optimized for embedded execution on ARM architecture. Execution begins through `main.py`.

### Dependencies
All dependencies are enumerated in `requirements.txt`, providing reproducible installation. Typical dependency classes include:

- Video capture libraries.
- System utilities for hardware access.
- Optional ML/AI libraries for inference, preprocessing, or recognition tasks.

### Execution Pipeline
The general processing flow:

1. Initialize camera interface and video stream.
2. Read frame buffers in a continuous loop.
3. Apply pre-processing (resizing, grayscale, normalization) for lightweight operation.
4. Execute the defined inference or detection logic.
5. Output results via CLI, logs, or audio feedback (speaker).

This architecture ensures minimal latency and deterministic behavior on the Pi 4B.

---

## Codebase Structure

- `main.py` – Central execution script: camera initialization, frame loop, optional inference logic, and output rendering.
- `requirements.txt` – Dependency manifest for reproducible environments.
- `.idea/` – Local IDE configuration.
- `.gitignore` – Git housekeeping.

The structure is modular enough for expansion (e.g., adding `vision/`, `audio/`, `utils/`, `models/` directories).

---

## Installation & Setup

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
````

### Hardware Setup

1. Attach camera module to Raspberry Pi USB port.
2. Mount hardware on wearable frame.
3. Verify camera functionality using a test capture script.

---

## Engineering Best Practices

* Modularize processing logic away from `main.py` for maintainability.
* Add structured logging for debugging on embedded devices.
* Introduce a `config.py` for resolution, FPS, and performance settings.
* Use hardware-accelerated libraries.
* Add thermal monitoring on the Pi 4B due to continuous camera load.

---

## Future Extensions

* Integrate a text-to-speech pipeline for real-time assistance.
* Add Bluetooth connectivity for external companion apps.
* Implement a Dockerized deployment for other ARM devices.
* Introduce low-power modes and frame-skipping logic to improve battery life.

---

## Reference
```
Project repo: [armaaxs/Enigma-Third-Eye](https://github.com/armaaxs/Enigma-Third-Eye)

```
