#pragma once

// Copy this file to device_config.h and edit local network values.
// Do not commit device_config.h because it contains WiFi/MQTT secrets.
#define DEVICE_WIFI_SSID "YOUR_WIFI"
#define DEVICE_WIFI_PASSWORD "YOUR_PASSWORD"
#define DEVICE_MQTT_BROKER "127.0.0.1"
#define DEVICE_MQTT_PORT 1883
#define DEVICE_MQTT_TOPIC "warehouse/rfid/scan"
#define DEVICE_MQTT_CLIENT "esp32-door-01"

#define DEVICE_NTP_SERVER "pool.ntp.org"
#define DEVICE_GMT_OFFSET 25200
#define DEVICE_DAYLIGHT_OFFSET 0
