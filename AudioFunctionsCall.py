from AudioFunctions import AudioFunctions

src_audio1 = AudioFunctions('voice.wav')
src_audio1.wahwah_audio(2,500,3000,1000)
src_audio1.wavwrite('output.wav')
