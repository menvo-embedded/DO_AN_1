import json
import threading
from datetime import datetime
from typing import Callable
import paho.mqtt.client as mqtt
from config.settings import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, MQTT_CLIENT
from utils.logger import get_logger
from utils.display_text import vi

log = get_logger("mqtt")
REQUIRED_FIELDS = {"uid", "timestamp"}

class RFIDListener:
    def __init__(self, on_event: Callable[[dict], None]):
        self._on_event = on_event
        self._client = mqtt.Client(client_id=MQTT_CLIENT,
                                   callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_connect    = self._on_connect
        self._client.on_message    = self._on_message
        self._client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            client.subscribe(MQTT_TOPIC, qos=1)
            log.info(f"Connected -> subscribed [{MQTT_TOPIC}]")
        else:
            log.error(f"Connect failed rc={reason_code}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        log.warning(f"Disconnected rc={reason_code}")

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            if not REQUIRED_FIELDS.issubset(data):
                log.warning(f"Bad payload: {msg.payload}")
                return
            dt = datetime.fromisoformat(data["timestamp"])
            # ESP32 chưa NTP sync (gửi epoch ~1970) → dùng thời gian nhận của server
            if dt.year < 2000:
                dt = datetime.now().astimezone()
                data["timestamp"] = dt.isoformat()
                log.warning("ESP32 NTP chưa sync (ts epoch) → dùng thời gian server")
            data["dt"] = dt
            emp = data.get("employee_id", "")
            log.info(
                f"RFID event: uid={data['uid']} emp={emp or 'unknown'} ts={data['timestamp']}"
                f" | {vi('rfid_scan')}{' — ' + emp if emp else ''}"
            )
            self._on_event(data)
        except Exception as e:
            log.error(f"Error: {e}")

    def start(self):
        self._client.connect_async(MQTT_BROKER, MQTT_PORT, keepalive=60)
        t = threading.Thread(target=self._client.loop_forever,
                             name="mqtt-loop", daemon=True)
        t.start()
        log.info(f"MQTT listener started -> {MQTT_BROKER}:{MQTT_PORT}")

    def stop(self):
        self._client.disconnect()
