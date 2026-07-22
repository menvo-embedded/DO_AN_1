#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <MFRC522.h>
#include <LiquidCrystal_I2C.h>
#include <ESP32Servo.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <time.h>

#if __has_include("../config/device_config.h")
#include "../config/device_config.h"
#else
#define DEVICE_WIFI_SSID "YOUR_WIFI"
#define DEVICE_WIFI_PASSWORD "YOUR_PASSWORD"
#define DEVICE_MQTT_BROKER "192.168.88.229"
#define DEVICE_MQTT_PORT 1883
#define DEVICE_MQTT_TOPIC "warehouse/rfid/scan"
#define DEVICE_MQTT_CLIENT "esp32-door-01"
#define DEVICE_NTP_SERVER "pool.ntp.org"
#define DEVICE_GMT_OFFSET 25200
#define DEVICE_DAYLIGHT_OFFSET 0
#endif

#define RFID_SS_PIN    13
#define RFID_SCK_PIN   18 
#define RFID_MOSI_PIN  23
#define RFID_MISO_PIN  19
#define RFID_RST_PIN   27

#define LCD_SDA_PIN    21
#define LCD_SCL_PIN    22

#define BUZZER_PIN     26
#define SERVO_PIN      25

// LED trạng thái
#define LED_GREEN_PIN  32   // LED xanh: thẻ hợp lệ / mở cổng
#define LED_RED_PIN    33   // LED đỏ: thẻ không hợp lệ / từ chối

#define GATE_CLOSE_ANGLE 0
#define GATE_OPEN_ANGLE  90

const char* WIFI_SSID   = DEVICE_WIFI_SSID;
const char* WIFI_PASS   = DEVICE_WIFI_PASSWORD;

const char* MQTT_BROKER = DEVICE_MQTT_BROKER;
const int   MQTT_PORT   = DEVICE_MQTT_PORT;
const char* MQTT_TOPIC  = DEVICE_MQTT_TOPIC;
const char* MQTT_CLIENT = DEVICE_MQTT_CLIENT;
const char* NTP_SERVER  = DEVICE_NTP_SERVER;
const long  GMT_OFFSET  = DEVICE_GMT_OFFSET;
const int   DAYLIGHT_OFF = DEVICE_DAYLIGHT_OFFSET;

struct Card { 
  const char* empId; 
  byte uid[4]; 
};

Card allowedCards[] = {
  {"NV001", {0x03, 0xF7, 0x63, 0x28}},
  {"NV002", {0x23, 0xFC, 0xDA, 0x26}},
  {"NV003", {0xF3, 0xAC, 0x71, 0x28}},
  {"NV004", {0x43, 0x8D, 0xFE, 0x27}},
  {"NV005", {0x95, 0xB0, 0xF6, 0x05}},
};

const int CARD_COUNT = sizeof(allowedCards) / sizeof(allowedCards[0]);

LiquidCrystal_I2C lcd(0x27, 16, 2);
MFRC522 rfid(RFID_SS_PIN, RFID_RST_PIN);
Servo gateServo;

WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);

unsigned long lastScanTime = 0;
const unsigned long scanCooldown = 1500;

unsigned long gateOpenTime = 0;
bool gateIsOpen = false;

void lcdShow(const char* line1, const char* line2 = "") {
  lcd.clear();
  lcd.setCursor(0, 0); 
  lcd.print(line1);

  lcd.setCursor(0, 1); 
  lcd.print(line2);
}

// ===================== LED =====================
void ledOff() {
  digitalWrite(LED_GREEN_PIN, LOW);
  digitalWrite(LED_RED_PIN, LOW);
}

void ledSuccess() {
  digitalWrite(LED_GREEN_PIN, HIGH);
  digitalWrite(LED_RED_PIN, LOW);
}

void ledFail() {
  digitalWrite(LED_GREEN_PIN, LOW);
  digitalWrite(LED_RED_PIN, HIGH);
}

// ===================== BUZZER =====================
void beep(int ms) {
  digitalWrite(BUZZER_PIN, HIGH); 
  delay(ms); 
  digitalWrite(BUZZER_PIN, LOW);
}

void beepSuccess() { 
  beep(50); 
  delay(80); 
  beep(50); 
}

void beepFail() { 
  beep(200); 
}

// ===================== SERVO =====================
void openGate() {
  gateServo.write(GATE_OPEN_ANGLE);
  gateOpenTime = millis();
  gateIsOpen = true;
}

// ===================== TIME =====================
String getTimestamp() {
  struct tm t;

  // Timeout 0ms: trả về ngay, không block chờ NTP sync.
  // Nếu NTP chưa sync thì dùng fallback epoch.
  if (!getLocalTime(&t, 0)) {
    return "1970-01-01T00:00:00+07:00";
  }

  char buf[30];
  strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S+07:00", &t);
  return String(buf);
}

// ===================== RFID =====================
String getUIDString(MFRC522::Uid* uid) {
  String s = "";

  for (byte i = 0; i < uid->size; i++) {
    if (uid->uidByte[i] < 0x10) s += "0";
    s += String(uid->uidByte[i], HEX);
  }

  s.toUpperCase();
  return s;
}

const char* findEmployee(MFRC522::Uid* uid) {
  if (uid->size != 4) return NULL;

  for (int i = 0; i < CARD_COUNT; i++) {
    bool match = true;

    for (byte j = 0; j < 4; j++) {
      if (uid->uidByte[j] != allowedCards[i].uid[j]) { 
        match = false; 
        break; 
      }
    }

    if (match) return allowedCards[i].empId;
  }

  return NULL;
}

// ===================== MQTT =====================
void publishMQTT(const char* empId, const String& uid, const String& ts) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[MQTT] Skip: WiFi not connected");
    return;
  }

  // Không reconnect blocking tại đây — loop() đã xử lý reconnect non-blocking.
  // Nếu chưa connected thì bỏ qua lần publish này, tránh delay buzzer/servo.
  if (!mqtt.connected()) {
    Serial.println("[MQTT] Skip: not connected (reconnect handled in loop)");
    return;
  }

  String payload = "{\"uid\":\"" + uid + "\","
                   "\"employee_id\":\"" + String(empId) + "\","
                   "\"timestamp\":\"" + ts + "\","
                   "\"device\":\"door-01\","
                   "\"zone\":1}";

  bool ok = mqtt.publish(MQTT_TOPIC, payload.c_str());
  Serial.println(ok ? "[MQTT] Published OK" : "[MQTT] Publish FAIL");
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  // LED setup
  pinMode(LED_GREEN_PIN, OUTPUT);
  pinMode(LED_RED_PIN, OUTPUT);
  ledOff();

  gateServo.setPeriodHertz(50);
  gateServo.attach(SERVO_PIN, 500, 2400);
  gateServo.write(GATE_CLOSE_ANGLE);

  Wire.begin(LCD_SDA_PIN, LCD_SCL_PIN);
  lcd.init();
  lcd.backlight();
  lcdShow("He thong kho", "Khoi dong...");
  delay(800);

  SPI.begin(RFID_SCK_PIN, RFID_MISO_PIN, RFID_MOSI_PIN, RFID_SS_PIN);
  rfid.PCD_Init();

  byte v = rfid.PCD_ReadRegister(MFRC522::VersionReg);
  Serial.print("VersionReg = 0x"); 
  Serial.println(v, HEX);

  if (v == 0x00 || v == 0xFF) {
    lcdShow("Loi RC522", "Kiem tra day");

    while (true) { 
      ledFail();
      beep(100); 
      delay(1000); 
      ledOff();
      delay(300);
    }
  }

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  configTime(GMT_OFFSET, DAYLIGHT_OFF, NTP_SERVER);

  lcdShow("Moi quet the", "");

  // Báo hệ thống khởi động OK
  ledSuccess();
  beep(60);
  delay(300);
  ledOff();
}

void loop() {
  // Đóng gate sau 3s non-blocking
  if (gateIsOpen && millis() - gateOpenTime >= 3000) {
    gateServo.write(GATE_CLOSE_ANGLE);
    gateIsOpen = false;

    // Tắt LED xanh khi cổng đóng
    ledOff();
  }

  // MQTT/WiFi xử lý phụ, không được chặn RFID
  if (WiFi.status() == WL_CONNECTED) {
    if (!mqtt.connected()) mqtt.connect(MQTT_CLIENT);
    mqtt.loop();
  }

  // Cooldown
  if (millis() - lastScanTime < scanCooldown) { 
    delay(20); 
    return; 
  }

  if (!rfid.PICC_IsNewCardPresent()) { 
    delay(20); 
    return; 
  }

  if (!rfid.PICC_ReadCardSerial()) { 
    delay(20); 
    return; 
  }

  unsigned long t_detect = millis();
  String uid = getUIDString(&rfid.uid);
  // getTimestamp() không gọi ở đây — tránh getLocalTime() block 5s khi NTP chưa sync.
  // Timestamp sẽ lấy sau physical feedback, chỉ cần trước publishMQTT.

  Serial.printf("[T=%lu] CARD DETECTED uid=%s\n", t_detect, uid.c_str());

  const char* empId = findEmployee(&rfid.uid);

  if (empId != NULL) {
    Serial.printf("[T=%lu] CARD VALID emp=%s\n", millis(), empId);

    // Phản hồi vật lý TRƯỚC — không gọi NTP/WiFi/MQTT trước bước này
    ledSuccess();
    lcdShow(empId, "Mo cong...");
    beepSuccess();
    openGate();
    Serial.printf("[T=%lu] PHYSICAL FEEDBACK NOW\n", millis());

    // Lấy timestamp SAU physical feedback rồi mới publish MQTT
    String ts = getTimestamp();
    Serial.printf("[T=%lu] BEFORE MQTT\n", millis());
    publishMQTT(empId, uid, ts);
    Serial.printf("[T=%lu] AFTER MQTT\n", millis());

  } else {
    Serial.printf("[T=%lu] CARD INVALID uid=%s\n", millis(), uid.c_str());

    ledFail();
    lcdShow("Khong hop le", "Bi tu choi");
    beepFail();

    delay(1000);
    ledOff();
  }

  Serial.println("------------------------");

  lastScanTime = millis();

  rfid.PICC_HaltA();
  rfid.PCD_StopCrypto1();

  delay(500);
  lcdShow("Moi quet the", "");
}
