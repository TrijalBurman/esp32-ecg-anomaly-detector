/*
  ESP32 + AD8232 ECG UART streamer

  CSV format:
    millis,raw_adc,lead_off_plus,lead_off_minus

  Wiring:
    AD8232 OUTPUT -> GPIO34
    AD8232 LO+    -> GPIO26
    AD8232 LO-    -> GPIO27
    AD8232 3.3V   -> ESP32 3V3
    AD8232 GND    -> ESP32 GND
*/

#include <Arduino.h>

constexpr uint8_t ECG_PIN = 34;
constexpr uint8_t LEAD_OFF_PLUS_PIN = 26;
constexpr uint8_t LEAD_OFF_MINUS_PIN = 27;

constexpr uint32_t BAUD_RATE = 115200;
constexpr uint16_t SAMPLE_RATE_HZ = 250;
constexpr uint32_t SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_RATE_HZ;

uint32_t nextSampleAt = 0;

void setup() {
  Serial.begin(BAUD_RATE);

  pinMode(ECG_PIN, INPUT);
  pinMode(LEAD_OFF_PLUS_PIN, INPUT);
  pinMode(LEAD_OFF_MINUS_PIN, INPUT);

  analogReadResolution(12);
  analogSetPinAttenuation(ECG_PIN, ADC_11db);

  nextSampleAt = micros();
}

void loop() {
  const uint32_t now = micros();

  if ((int32_t)(now - nextSampleAt) >= 0) {
    nextSampleAt += SAMPLE_INTERVAL_US;

    const int raw = analogRead(ECG_PIN);
    const int leadOffPlus = digitalRead(LEAD_OFF_PLUS_PIN);
    const int leadOffMinus = digitalRead(LEAD_OFF_MINUS_PIN);

    Serial.print(millis());
    Serial.print(',');
    Serial.print(raw);
    Serial.print(',');
    Serial.print(leadOffPlus);
    Serial.print(',');
    Serial.println(leadOffMinus);
  }
}

