# Python CV Pipeline

Main runtime entry point:

```powershell
python main.py
```

Main modules:

- `detection/`: camera Zone 1, Zone 2, optional Zone 3
- `tracking/`: ByteTrack wrapper
- `reid/`: body Re-ID, gallery, optional InsightFace engine
- `fusion/`: RFID + camera + Re-ID fusion logic
- `mqtt/`: RFID event listener from ESP32
- `database/`: SQLite schema and queries
- `dashboard/`: Flask realtime dashboard
- `tools/`: dataset, gallery, calibration, and debug utilities

Create `.env` from `.env.example` on the demo machine. Do not commit `.env`,
RTSP passwords, galleries, biometric images, or raw videos.
