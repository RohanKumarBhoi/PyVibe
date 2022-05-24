from os import path
from pydub import AudioSegment

''' Code to convert any file to wav'''
src1 = './input.wav'
src2 = './metallic-drums.wav'
sound1 =  AudioSegment.from_file(src1)
sound2 =  AudioSegment.from_file(src2)
combined = sound1.overlay(sound2)
combined.export("./combined.wav", format='wav')
