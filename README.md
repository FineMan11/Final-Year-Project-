/*============================ EG LABS ===================================//

Demonstration on how to use 16x2 LCD display with an arduino board

* LED anode attached to digital output 6
* LED cathode attached to ground through a 1K resistor

//============================ EG LABS ===================================*/

// include the library code:
#include <LiquidCrystal.h>

// initialize the library with the numbers of the interface pins
LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

long randNumber;                                                            // the variable which is supposed to hold the random number
const int analogOutPin = 6;                                                 // Analog output pin where the LED is attached to

void setup()
{
  // set up the LCD's number of columns and rows:
  lcd.begin(16, 2);
  // Print a message to the LCD.
  lcd.print("ENGINEERS GARAGE");
  lcd.setCursor(0, 1);
  lcd.print("  RANDOM NUMBER ");
  // initialize the serial port
  Serial.begin(9600);
  // initialize the pseudo-random number generator
  randomSeed(analogRead(0));
}

void loop()
{
  randNumber = random(0, 255);                                              // generate a random number
  Serial.println(randNumber);                                               // send the random number to the serial port
  analogWrite(analogOutPin, randNumber);                                    // vary the brightness of the LED according to the random number
  delay(300);
}

