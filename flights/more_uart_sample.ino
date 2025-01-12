#include <SPI.h>

/* WIRE CONNECTIONS FOR POTS:

pot 0 (elevation) up down left joystick -
red wire connect to gnd and term b
black wire connect to term a
orange to probe
high n - joystick up (fly up)
low n - joystick down (fly down)

pot 1 (yaw) right left left joystick -
red wire connect to gnd and term b
black wire connect to term a
orange to probe
high n - joystick right (rotate right)
low n - joystick left (rotate left)

pot 2 (directional forward/back) up down right joystick -
red wire connect to term b
black wire connect to gnd and term a
orange to probe
high n - joystick up (fly forwards)
low n - joystick down (fly backwards)

pot 3 (directional left/right) right left right joystick -
red wire connect to term b
black wire connect to gnd and term a
orange to probe
high n - joystick right (fly right)
low n - joystick left (fly left)

pot 4 (camera control) tbd
*/

#define NUM_POTS 5
int cs[] = { 10, 9, 8, 7, 6 };            //up/down, rotate left/right, forwards/backwards, directional left/right, camera
int read_in[] = { A0, A1, A2, A3, A4 };

int buttons[] = {5, 4, 3, 2, 1};

void setup() {
  pinMode(13, OUTPUT);
  pinMode(11, OUTPUT);
  for (int i = 0; i < NUM_POTS; ++i) {
    pinMode(cs[i], OUTPUT);
    digitalWrite(cs[i], HIGH);
    pinMode(buttons[i], OUTPUT);
    digitalWrite(buttons[i], LOW);
  }
  SPI.begin();
  SPI.setClockDivider(SPI_CLOCK_DIV8);
  SPI.setDataMode(SPI_MODE0);
  SPI.setBitOrder(MSBFIRST);
  Serial.begin(9600);
  all_neutral();
}

void loop() {
  // put your main code here, to run repeatedly:
  if(Serial.available() > 0) {
    int input = Serial.read();
    if(input == 0x31){          //turn controller on and off
      digitalWrite(buttons[0], HIGH);
      delay(2000);
      digitalWrite(buttons[0], LOW);
    }
    else if(input == 0x32) {    //turn on indoor mode
      digitalWrite(buttons[1], HIGH);
      delay(3500);
      digitalWrite(buttons[1], LOW);
      calibrate();
      delay(2000);
    }
    else if(input == 0x33) {    //unlock motors
      unlock();
    }
    else if(input == 0x34) {    //flight plan
      unlock();
      all_neutral();
      int t_axis[] = {64, 64, 64, 64, 64};
      int val;
      takeoff();
      while(1) {
        if (Serial.available() > 0) {
          int fl_in = Serial.read();
          if(fl_in == 0x71) {
            break;
          }
          else if(fl_in == 0x01) {
            while(Serial.available() == 0) {}
            val = Serial.read();
            //Serial.write(val);
            t_axis[0] = val;
            pot_write(cs[0], val);
          }
          else if(fl_in == 0x02) {
            while(Serial.available() == 0) {}
            val = Serial.read();
            t_axis[1] = val;
            pot_write(cs[1], val);
          }
          else if(fl_in == 0x03) {
            while(Serial.available() == 0) {}
            val = Serial.read();
            t_axis[2] = val;
            pot_write(cs[2], val);
          }
          else if(fl_in == 0x04) {
            while(Serial.available() == 0) {}
            val = Serial.read();
            t_axis[3] = val;
            pot_write(cs[3], val);
          }
          else if(fl_in == 0x0F) {
            all_neutral();
          }
          else if(fl_in == 0x10) {
            land();
            break;
          }
          else if(fl_in == 0x11) {
            for(int i = 0; i < NUM_POTS; ++i) {
              Serial.write(t_axis[i]);
            }
          }
        }
      }
    }
  }
}

void takeoff() {
  all_neutral();
  pot_write(cs[0], 128);
  delay(750);
  pot_write(cs[0], 64);
}

void land() {
  all_neutral();
  for (int j = 64; j >= 0; --j) {
    pot_write(cs[0], j);
    delay(150);
  }
}

void all_neutral() {
  for(int i = 0; i < NUM_POTS; ++i) {
    pot_write(cs[i], 64);
  }
}

void calibrate() {
  all_neutral();
  all_neutral();
  pot_write(cs[0], 10);
  pot_write(cs[1], 118);
  pot_write(cs[2], 10);
  pot_write(cs[3], 118);
  delay(500);
  all_neutral();
}

void unlock() {
  all_neutral();
  pot_write(cs[0], 10);
  pot_write(cs[1], 10);
  pot_write(cs[2], 10);
  pot_write(cs[3], 118);
  delay(500);
  all_neutral();
}

void pot_write(int pot, int val) {
  digitalWrite(pot, LOW);  // set the SS pin to LOW
  SPI.transfer(0x00);
  SPI.transfer(val);
  digitalWrite(pot, HIGH);  // set the SS pin HIGH
}
