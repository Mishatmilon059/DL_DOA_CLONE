// ============================================================
//  p6 — DUAL MODE: buttons speak, serial teaches
// ============================================================
// The most complete bench sketch, and the closest ancestor of
// firmware/braille_tutor/braille_tutor.ino.
//
// Mode 1 (typing): tap dots, press Submit. The speaker names the
//   decoded letter. Motors stay silent -- the learner is being
//   tested, not shown the answer. Enter prints the message.
// Mode 2 (tutor): type a number 1-50 on Serial. The speaker names
//   the letter and the motors then replay its dots. Two-cell
//   characters ঋ and ৎ vibrate the dot-5 prefix cell first, pause,
//   then the main cell.
//
// Pins (as built on the bench):
//   dots        GPIO 4, 5, 15, 19, 21, 22
//   submit      GPIO 18
//   enter       GPIO 34   <-- input-only, no internal pull-up:
//                             needs an external 10k to 3V3
//   motors      GPIO 13, 14, 26, 27, 32, 33
//   DFPlayer RX GPIO 25 / TX GPIO 23 (1k in series on DFPlayer RX)
// ============================================================

#include <Arduino.h>
#include <DFRobotDFPlayerMini.h>

// ============================================================
//  HARDWARE PIN CONFIGURATIONS (STRICTLY UNCHANGED)
// ============================================================

// --- 6 Braille Dot Push Buttons ---
const int buttonPins[6] = {4, 5, 15, 19, 21, 22};

// --- Action Buttons ---
const int submitButtonPin = 18;   
const int enterButtonPin = 16;    

// --- 6 Braille Dot Vibration Motors ---
const int PIN_MOTOR[6] = { 13, 14, 26, 27, 32, 33 }; 

// --- DFPlayer Pin Configuration ---
#define PIN_DF_RX 25   
#define PIN_DF_TX 23   

HardwareSerial mySoftwareSerial(2); 
DFRobotDFPlayerMini dfPlayer;

// ============================================================
//  BRAILLE DATA DICTIONARY
// ============================================================

const uint8_t BRAILLE_PATTERN[50] = {
  0x01, 0x1C, 0x0A, 0x14, 0x25, 0x33, 0x17, 0x11, 0x0C, 0x15, 0x2A, // অ থেকে ঔ
  0x05, 0x28, 0x1B, 0x23, 0x2C, 0x09, 0x21, 0x1A, 0x34, 0x12, // ক থেকে ঞ
  0x3E, 0x3A, 0x2B, 0x3F, 0x3C, 0x1E, 0x39, 0x19, 0x2E, 0x1D, // ট থেকে ন
  0x0F, 0x0B, 0x03, 0x18, 0x0D, 0x3D, 0x17, 0x07, 0x29, 0x2F, // প থেকে ষ
  0x0E, 0x13, 0x3B, 0x37, 0x36, 0x26, 0x30, 0x06, 0x08        // স থেকে ঁ
};

const char* LETTER_NAMES[50] = {
  "অ", "আ", "ই", "ঈ", "উ", "ঊ", "ঋ", "এ", "ঐ", "ও", "ঔ",
  "ক", "খ", "গ", "ঘ", "ঙ", "চ", "ছ", "জ", "ঝ", "ঞ",
  "ট", "ঠ", "ড", "ঢ", "ণ", "ত", "থ", "দ", "ধ", "ন",
  "প", "ফ", "ব", "ভ", "ম", "য", "র", "ল", "শ", "ষ",
  "স", "হ", "ড়", "ঢ়", "য়", "ৎ", "ং", "ঃ", "ঁ"
};

// ============================================================
//  SYSTEM VARIABLES & DEBOUNCING
// ============================================================

const unsigned long debounceDelay = 50; 

bool activeChord[6] = {false, false, false, false, false, false};
String typedMessage = "";
byte pendingPrefix = 0; 

int lastRawState[6], confirmedState[6];
unsigned long lastChangeTime[6];

int lastRawSubmit, confirmedSubmit;
unsigned long lastSubmitChangeTime;

int lastRawEnter, confirmedEnter;
unsigned long lastEnterChangeTime;


void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n==========================================");
  Serial.println("  BANGLA BRAILLE DUAL-MODE SYSTEM READY!");
  Serial.println("==========================================");
  Serial.println("[MODE 1] Type on Push Buttons -> Speaker will say the letter.");
  Serial.println("[MODE 2] Type number (1-50) in Serial -> Speaker will say the letter & Motors will vibrate.");
  Serial.println("==========================================\n");

  for (int i = 0; i < 6; i++) {
    pinMode(PIN_MOTOR[i], OUTPUT);
    digitalWrite(PIN_MOTOR[i], LOW);
  }

  for (int i = 0; i < 6; i++) {
    pinMode(buttonPins[i], INPUT_PULLUP);
    lastRawState[i] = digitalRead(buttonPins[i]);
    confirmedState[i] = lastRawState[i];
    lastChangeTime[i] = millis();
  }

  pinMode(submitButtonPin, INPUT_PULLUP);
  lastRawSubmit = digitalRead(submitButtonPin);
  confirmedSubmit = lastRawSubmit;
  lastSubmitChangeTime = millis();

  pinMode(enterButtonPin, INPUT_PULLUP);
  lastRawEnter = digitalRead(enterButtonPin);
  confirmedEnter = lastRawEnter;
  lastEnterChangeTime = millis();

  mySoftwareSerial.begin(9600, SERIAL_8N1, PIN_DF_RX, PIN_DF_TX);
  delay(500);

  if (!dfPlayer.begin(mySoftwareSerial, false, true)) {
    Serial.println("❌ DFPlayer Mini Init Failed!");
  } else {
    Serial.println("✅ DFPlayer Mini Connected!");
    dfPlayer.volume(12); 
  }
}

void loop() {
  unsigned long now = millis();

  // ============================================================
  //  REQUIREMENT 2: SERIAL INPUT MODE (Tutor Mode)
  // ============================================================
  if (Serial.available() > 0) {
    int inputVal = Serial.parseInt();
    if (inputVal >= 1 && inputVal <= 50) {
      int index = inputVal - 1;
      Serial.println("\n----------------------------------------");
      Serial.printf("🎯 SERIAL INPUT RECEIVED: %d -> %s\n", inputVal, LETTER_NAMES[index]);
      
      // 1. স্পিকার লেটারটি উচ্চারণ করবে
      dfPlayer.playMp3Folder(inputVal);
      delay(1200); // সাউন্ড প্লে হওয়ার জন্য সময় দেওয়া

      // 2. মোটরগুলো ভাইব্রেট করবে (Handling 2-Cell Characters correctly)
      if (index == 6 || index == 46) { 
        // 'ঋ' (index 6) এবং 'ৎ' (index 46) এর জন্য
        Serial.println(">> 2-Cell Character Detected!");
        Serial.println(">> Vibrating Cell 1 (Prefix 0x10)...");
        vibrateBrailleDotsSequential(0x10); // 0x10 হচ্ছে ডট ৫ এর প্যাটার্ন
        
        delay(1200); // দুটি সেলের মাঝে বিরতি
        
        Serial.println(">> Vibrating Cell 2 (Main Pattern)...");
        vibrateBrailleDotsSequential(BRAILLE_PATTERN[index]); // বাকি মূল প্যাটার্ন
        
      } else {
        // সাধারণ 1-Cell অক্ষরের জন্য
        vibrateBrailleDotsSequential(BRAILLE_PATTERN[index]);
      }
      
      Serial.println("----------------------------------------\n");
    }
  }

  // ============================================================
  //  REQUIREMENT 1: PUSH BUTTON MODE (Typing Mode)
  // ============================================================
  
  // 1. Process Dot Buttons
  for (int i = 0; i < 6; i++) {
    int reading = digitalRead(buttonPins[i]);
    if (reading != lastRawState[i]) {
      lastChangeTime[i] = now;
      lastRawState[i] = reading;
    }
    if ((now - lastChangeTime[i]) > debounceDelay && reading != confirmedState[i]) {
      confirmedState[i] = reading;
      if (confirmedState[i] == LOW) { 
        activeChord[i] = !activeChord[i];
        Serial.print("Dot "); Serial.print(i + 1);
        Serial.println(activeChord[i] ? " selected." : " removed.");
      }
    }
  }

  // 2. Process Submit Button
  int submitReading = digitalRead(submitButtonPin);
  if (submitReading != lastRawSubmit) {
    lastSubmitChangeTime = now;
    lastRawSubmit = submitReading;
  }
  if ((now - lastSubmitChangeTime) > debounceDelay && submitReading != confirmedSubmit) {
    confirmedSubmit = submitReading;
    if (confirmedSubmit == LOW) { 
      decodeAndProcessChord();
    }
  }

  // 3. Process Enter Button 
  int enterReading = digitalRead(enterButtonPin);
  if (enterReading != lastRawEnter) {
    lastEnterChangeTime = now;
    lastRawEnter = enterReading;
  }
  if ((now - lastEnterChangeTime) > debounceDelay && enterReading != confirmedEnter) {
    confirmedEnter = enterReading;
    if (confirmedEnter == LOW) { 
      handleEnterPress();
    }
  }
}

// ============================================================
//  CORE LOGIC & ACTION FUNCTIONS
// ============================================================

void resetChord() {
  for (int i = 0; i < 6; i++) {
    activeChord[i] = false;
  }
}

// মোটর ভাইব্রেশন ফাংশন (শুধুমাত্র সিরিয়াল ইনপুটের সময় কল হবে)
void vibrateBrailleDotsSequential(uint8_t pattern) {
  Serial.print("Vibrating Sequence: ");
  for (int i = 0; i < 6; i++) {
    if (pattern & (1 << i)) {
      Serial.printf("[Dot %d] ", i + 1);
    }
  }
  Serial.println();

  for (int i = 0; i < 6; i++) {
    if (pattern & (1 << i)) {
      digitalWrite(PIN_MOTOR[i], HIGH);
      delay(400);                      
      digitalWrite(PIN_MOTOR[i], LOW);   
      delay(800);   
    }
  }
}

// পুশ বাটন সাবমিট করার পর কাজ
void decodeAndProcessChord() {
  byte typedPattern = 0;
  bool isBlank = true;

  for (int i = 0; i < 6; i++) {
    if (activeChord[i]) {
      typedPattern |= (1 << i);
      isBlank = false;
    }
  }

  if (isBlank) {
    Serial.println("Space input detected.");
    typedMessage += " ";
    pendingPrefix = 0; 
    return;
  }

  // Check if trying to start a 2-cell character (Dot 5)
  if (typedPattern == 0x10 && pendingPrefix == 0) {
    pendingPrefix = 0x10;
    Serial.println("Prefix (Dot 5) entered. Enter 2nd cell for ঋ or ৎ...");
    resetChord();
    return;
  }

  int matchedIndex = -1;

  if (pendingPrefix == 0x10) {
    if (typedPattern == 0x17) {
      matchedIndex = 6;  // ঋ
    } else if (typedPattern == 0x26 || typedPattern == 0x1E) {
      matchedIndex = 46; // ৎ
    } else {
      Serial.println("Invalid 2-cell combination. Resetting.");
      pendingPrefix = 0;
      resetChord();
      return;
    }
  } else {
    for (int i = 0; i < 50; i++) {
      if (i == 6 || i == 46) continue; // Skip 2-cell characters here
      if (BRAILLE_PATTERN[i] == typedPattern) {
        matchedIndex = i;
        break;
      }
    }
  }

  // ✅ MATCH FOUND FROM BUTTONS
  if (matchedIndex != -1) {
    int trackNum = matchedIndex + 1;
    Serial.println("========================================");
    Serial.printf("SUCCESS! Button Input Decoded: %s\n", LETTER_NAMES[matchedIndex]);
    
    typedMessage += LETTER_NAMES[matchedIndex];

    // রিকয়ারমেন্ট ১ অনুযায়ী: স্পিকার শুধুমাত্র লেটারটি উচ্চারণ করবে (কোনো ভাইব্রেশন হবে না)
    dfPlayer.playMp3Folder(trackNum);
    
    Serial.println("========================================");
  } else {
    Serial.println("❌ Unknown Braille pattern submitted.");
  }
  

  pendingPrefix = 0; 
  resetChord(); 
}

void handleEnterPress() {
  Serial.println("\n----------------------------------------");
  Serial.print("📝 TYPED MESSAGE: ");
  Serial.println(typedMessage);
  Serial.println("----------------------------------------\n");

  typedMessage = ""; 
  pendingPrefix = 0; 
}