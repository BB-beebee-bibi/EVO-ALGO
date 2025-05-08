import sys
import time
from pynput.keyboard import Controller, Key

# Create keyboard controller
keyboard = Controller()

def type_text(text):
    """Type text character by character"""
    for char in text:
        keyboard.type(char)
        time.sleep(0.01)  # Small delay to ensure typing is smooth

def press_enter():
    """Press enter key"""
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)

def test_inputs():
    # Wait for the program to start
    time.sleep(1)
    
    # Type the request
    request = "tell me what all bluetooth devices within range are, including signal strength and device type. Update every 0.20 seconds, results should be formatted as an easy-to-read table"
    type_text(request)
    press_enter()
    
    # Type 'n' for no parameter changes
    time.sleep(1)
    keyboard.type('n')
    press_enter()
    
    # Press enter to start
    time.sleep(1)
    press_enter()

if __name__ == "__main__":
    test_inputs()
