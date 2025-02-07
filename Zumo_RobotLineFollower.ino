#include <Pixy2.h>
#include <PIDLoop.h>
#include <ZumoMotors.h>

// Zumo speeds, maximum allowed is 400
#define ZUMO_FAST        400
#define ZUMO_SLOW        300
#define X_CENTER         (pixy.frameWidth/2)

Pixy2 pixy;
ZumoMotors motors;

PIDLoop headingLoop(10000, 0, 0, false);

void setup() 
{
  motors.setLeftSpeed(0);
  motors.setRightSpeed(0);

  pixy.init();
  // Turn on both lamps, upper and lower for maximum exposure
  pixy.setLamp(1, 1);
  // change to the line_tracking program
  pixy.changeProg("line");
  // camera angle
  pixy.setServos(500, 775);
}

void loop()
{
  int8_t res;
  int32_t error; 
  int left, right;
  char buf[96];

  // Get latest data from Pixy, including main vector, new intersections and new barcodes.
  res = pixy.line.getMainFeatures();

  // If error or nothing detected, stop motors
  if (res<=0) 
  {
    motors.setLeftSpeed(-250);
    motors.setRightSpeed(250);
    return;
  }

  // We found the vector...
  if (res&LINE_VECTOR)
  {
    // Calculate heading error with respect to m_x1, which is the far-end of the vector,
    // the part of the vector we're heading toward.
    error = (int32_t)pixy.line.vectors->m_x1 - (int32_t)X_CENTER;

    pixy.line.vectors->print();

    // Perform PID calcs on heading error.
    headingLoop.update(error);

    // separate heading into left and right wheel velocities.
    left = headingLoop.m_command;
    right = -headingLoop.m_command;

    // If vector is heading away from us (arrow pointing up), things are normal.
    if (pixy.line.vectors->m_y0 > pixy.line.vectors->m_y1)
    {
      // ... but slow down a little if intersection is present, so we don't miss it.
      if (pixy.line.vectors->m_flags&LINE_FLAG_INTERSECTION_PRESENT)
      {
        left += ZUMO_SLOW;
        right += ZUMO_SLOW;
      }
      else // otherwise, pedal to the metal!
      {
        left += ZUMO_FAST;
        right += ZUMO_FAST;
      }    
    }
    else{ // If the vector is pointing down, or down-ish, we need to go backwards to follow.
      pixy.line.reverseVector();

    } 
    motors.setLeftSpeed(left);
    motors.setRightSpeed(right);
  }
}
