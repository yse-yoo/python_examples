import pygame

# Pygameの初期化
pygame.mixer.init()

# 音声ファイルのロード
audio_path = "audios/ani_ge_bird_taka02.mp3";
# audio_path = "audios/ani_ge_bird_taka03.mp3";
pygame.mixer.music.load(audio_path)

# 再生
pygame.mixer.music.play()

# 終了を待つ
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)