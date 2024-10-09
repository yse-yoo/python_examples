import pygame

audio_path = "audios/ani_ge_bird_taka02.mp3";
# audio_path = "audios/ani_ge_bird_taka03.mp3";

def init():
    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)

def play():
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

init()
play()