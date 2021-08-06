import  RPi.GPIO as GPIO
KEY = 5

def isr_key_event(pin):
    print("Key is pressed [%d]"%pin)
    my_event = pygame.event.Event(KEYDOWN, {"key":K_SPACE, "mod":0, "unicode":' '})
    pygame.event.post(my_event)

def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(KEY, GPIO.IN)
    GPIO.add_event_detect(KEY, GPIO.FALLING, callback=isr_key_event, bouncetime=300)
    isGameQuit = introscreen()
    if not isGameQuit:
        gameplay()

main()
