# Firmware ESP32

Firmware handles the warehouse door controller:

- RC522 RFID card reading
- LCD I2C status display
- buzzer and LED feedback
- servo gate open/close control
- WiFi, NTP time sync, and MQTT event publishing

## Local Device Config

Copy `config/device_config.example.h` to `config/device_config.h` before flashing:

```powershell
Copy-Item firmware\config\device_config.example.h firmware\config\device_config.h
```

Edit `device_config.h` with the local WiFi and MQTT broker values. Do not commit
`device_config.h`.

## Flash

From the repository root:

```powershell
platformio run -d firmware -t upload
platformio device monitor -d firmware
```
