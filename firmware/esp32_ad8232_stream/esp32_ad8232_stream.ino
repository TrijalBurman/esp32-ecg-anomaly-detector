/*
  ESP32 + AD8232 ECG UART Streamer

  Arduino IDE setup:
    1. Install "esp32 by Espressif Systems" from Boards Manager.
    2. Select board: "ESP32 Dev Module".
    3. Open this file in Arduino IDE and upload.
    4. Close Serial Monitor before running the Python dashboard.

  Serial baud:
    115200

  CSV output format:
    millis,raw_adc,lead_off_plus,lead_off_minus

  Wiring:
    AD8232 OUTPUT -> ESP32 GPIO34
    AD8232 LO+    -> ESP32 GPIO26
    AD8232 LO-    -> ESP32 GPIO27
    AD8232 3.3V   -> ESP32 3V3
    AD8232 GND    -> ESP32 GND

  Electrode placement:
    RA -> right wrist
    LA -> left wrist
    RL -> right ankle
*/

#include <Arduino.h>

#ifndef ESP32
#error "This sketch is intended for ESP32 boards. Select an ESP32 board in Arduino IDE."
#endif

// Pin configuration. Change these only if your wiring is different.
const int ECG_PIN = 34;
const int LEAD_OFF_PLUS_PIN = 26;
const int LEAD_OFF_MINUS_PIN = 27;

// Serial and sampling configuration.
const unsigned long BAUD_RATE = 115200;
const unsigned int SAMPLE_RATE_HZ = 250;
const unsigned long SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_RATE_HZ;

// Set this to true if you want Arduino Serial Monitor to show a CSV header.
// The Python parser ignores malformed/non-data lines, so either setting is OK.
const bool PRINT_CSV_HEADER = false;

unsigned long nextSampleAtUs = 0;

void setup() {
  Serial.begin(BAUD_RATE);
  delay(1000);

  pinMode(ECG_PIN, INPUT);
  pinMode(LEAD_OFF_PLUS_PIN, INPUT);
  pinMode(LEAD_OFF_MINUS_PIN, INPUT);

  // ESP32 ADC setup. GPIO34 is input-only and works well for analog ECG input.
  analogReadResolution(12);
  analogSetPinAttenuation(ECG_PIN, ADC_11db);

  if (PRINT_CSV_HEADER) {
    Serial.println("millis,raw_adc,lead_off_plus,lead_off_minus");
  }

  nextSampleAtUs = micros();
}

void loop() {
  const unsigned long nowUs = micros();

  if ((long)(nowUs - nextSampleAtUs) >= 0) {
    nextSampleAtUs += SAMPLE_INTERVAL_US;

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
