'''
File: AudioFunctions.py
Description: A python class to perform DSP processing on give audio.wav file
Version:1

How to use:

from PyVibe.AudioFunctions import AudioFunctions

#Read in WAV file into Python Class

'''
from pydub import AudioSegment
import math
import numpy
from scipy import signal
from scipy.io import wavfile

class AudioFunctions(object):

    __slots__ = ('data','Fs','channel','enc','time_in_sec','length','dtype','time_from_start','time_from_end')

    def __init__(self, input_audio_path, time_from_start):
        self.data, self.Fs, self.enc, self.dtype = self.wavread(input_audio_path)
        self.data = numpy.array(self.data)
        self.channel = self.data.ndim
        if self.channel == 1:
            self.length = len(self.data)
            self.time_in_sec = self.length/self.Fs
        elif self.channel == 2:
            self.length = len(self.data[:,0])
            self.time_in_sec = self.length/self.Fs
        self.time_from_start = time_from_start
        self.time_from_end = 300-self.time_in_sec
        self.show()

    def wavread(self, file_name):
        fs, snd = wavfile.read(file_name)
        if snd.dtype == 'int16':
            snd = snd / (2.**15)
            nbits = 16
        elif snd.dtype == 'int32':
            snd = snd / (2.**31)
            nbits = 32
        elif snd.dtype == 'float32':
            nbits = 32
        return snd, fs, nbits, snd.dtype

    def wavwrite(self, out_file_name):
        self.format_data()
        self.show()
        if self.enc not in [16,32]:
            print("Cannot write")
            return
        if self.enc == 16:
            self.data = self.data*(2.**15)
            self.data = self.data.astype('int16')
        elif self.enc == 32:
            self.data = self.data*(2.**31)
            self.data = self.data.astype('int32')
        wavfile.write(out_file_name,self.Fs,self.data)
        return

    def show(self):
        print("Fs:",self.Fs)
        print("channel:",self.channel)
        print("enc:",self.enc)
        print("length:",self.length)
        print("time_in_sec:",self.time_in_sec)
        print("time_from_start:",self.time_from_start)
        print("time_from_end:",self.time_from_end)
        
    def format_data(self):
        if isinstance(self.data, numpy.ndarray):
            print('Data Already in Correct Format')
        else:
            self.data = numpy.array(self.data)
        if self.channel != self.data.ndim:
            n_samples, n_channel = self.data.shape
            if n_channel != self.channel:
                self.data = numpy.transpose(self.data)
                
    def update_self(self):
        if(self.channel == 1):
            self.length = len(self.data)
            self.time_in_sec = self.length/self.Fs
        else:
            self.length = len(self.data[:,0])
            self.time_in_sec = self.length/self.Fs
        self.time_from_end = 300-self.time_in_sec

        
    def wahwah_audio(self, damp, min_freq, max_freq, wah_freq):
        delta = (1.0*wah_freq)/self.Fs
        Fc = [*numpy.arange(min_freq, max_freq, delta)]
        if self.channel == 1:
            while(len(Fc) < self.length):
                Fc = Fc + [*numpy.arange(max_freq, min_freq, -delta)]
                Fc = Fc + [*numpy.arange(min_freq, max_freq, delta)]
            Fc = Fc[:self.length]
            F1 = 2*math.sin((math.pi*Fc[0])/self.Fs)
            Q1 = 2*damp
            yh = numpy.zeros(self.length)
            yb = numpy.zeros(self.length)
            y1 = numpy.zeros(self.length)
            yh[0] = self.data[0]
            yb[0] = F1*yh[0]
            y1[0] = F1*yb[0]
            for x in range(1,self.length):
                yh[x] = self.data[x] - y1[x-1] - Q1*yb[x-1]
                yb[x] = F1*yh[x] + yb[x-1]
                y1[x] = F1*yb[x] + y1[x-1]
                F1 = 2*math.sin((math.pi*Fc[x])/self.Fs)
            maxyb = max(abs(yb))
            self.data = yb/maxyb
        elif self.channel == 2:
            out2 = numpy.zeros((self.length,2),self.dtype)
            for i in range(self.channel):
                while(len(Fc) < self.length):
                    Fc = Fc + [*numpy.arange(max_freq, min_freq, -delta)]
                    Fc = Fc + [*numpy.arange(min_freq, max_freq, delta)]
                Fc = Fc[:self.length]
                F1 = 2*math.sin((math.pi*Fc[0])/self.Fs)
                Q1 = 2*damp
                yh = numpy.zeros(self.length)
                yb = numpy.zeros(self.length)
                y1 = numpy.zeros(self.length)
                yh[0] = self.data[0,i]
                yb[0] = F1*yh[0]
                y1[0] = F1*yb[0]
                for x in range(1,self.length):
                    yh[x] = self.data[x,i] - y1[x-1] - Q1*yb[x-1]
                    yb[x] = F1*yh[x] + yb[x-1]
                    y1[x] = F1*yb[x] + y1[x-1]
                    F1 = 2*math.sin((math.pi*Fc[x])/self.Fs)
                maxyb = max(abs(yb))
                out2[:,i] = yb/maxyb
            self.data = out2

    def shelving_filter(self, gain, center_freq, slope, type):
        ''' Usage - [B,A] = shelving(G,Fc,Fs,Q,type) where
                    gain -> logarithmic gain (in dB)
                    center_freq -> Center Frequency
                    Slope -> Adjusts he slope be replacing the sqrt(2) term
                    type -> Filter type : 'base_shelf','treble_shelf'
        '''
        if(type != 'base_shelf' and type != 'treble_shelf'):
            print("Unsupported Filter Type")
            return
        K = math.tan((math.pi*center_freq)/self.Fs)
        V0 = 10**(gain/20.)
        root2 = 1/slope

        '''Invert gain if a cut'''
        if(V0 < 1):
            V0 = 1/V0

        '''G(+) and base_shelf Base Boost'''
        '''G(-) and base_shelf Base Cut'''
        '''G(+) and treble_shelf Treble Boost'''
        '''G(-) and treble_shelf Treble Cut'''
        if (gain>0 and type == 'base_shelf'):
            denom = ( 1 + root2*K + math.pow(K,2))
            b0 = (1 + math.sqrt(V0)*root2*K + V0*math.pow(K,2)) / denom
            b1 = (2 * (V0*math.pow(K,2) - 1)) / denom
            b2 = (1 - math.sqrt(V0)*root2*K + V0*math.pow(K,2)) / denom
            a1 = (2 * (math.pow(K,2) - 1)) / denom
            a2 = (1 - root2*K + math.pow(K,2)) / denom

        elif (gain<0 and type == 'base_shelf'):
            denom = (1 + root2*math.sqrt(V0)*K + V0*math.pow(K,2))
            b0 = ( 1 + root2*K + math.pow(K,2)) / denom
            b1 = ( 2 * (math.pow(K,2) - 1)) / denom
            b2 = ( 1 - root2*K + math.pow(K,2)) / denom
            a1 = ( 2 * (V0*math.pow(K,2) -1)) / denom
            a1 = ( 1 - root2*math.sqrt(V0)*K +V0*math.pow(K,2)) / denom

        elif (gain>0 and type == 'treble_shelf'):
            denom = ( 1 + root2*K + math.pow(K,2))
            b0 = (V0 + root2*math.sqrt(V0)*K + math.pow(K,2)) / denom
            b1 = (2 * (math.pow(K,2) - V0)) / denom
            b2 = (V0 - root2*math.sqrt(V0)*K + math.pow(K,2)) / denom
            a1 = (2 * (math.pow(K,2) -1)) / denom
            a2 = (1 - root2*K + math.pow(K,2)) / denom

        elif (gain<0 and type =='treble_shelf'):
            denom = (V0 + root2*math.sqrt(V0)*K + math.pow(K,2))
            b0 = (1 + root2*K + math.pow(K,2)) / denom
            b1 = (2 * (math.pow(K,2)-1)) / denom
            b2 = (1 - root2*K + math.pow(K,2)) /denom
            a1 = (2 * (math.pow(K,2)/V0 -1)) / (1 + root2/(math.sqrt(V0)*K) + math.pow(K,2)/10)
            a2 = (1 - root2/(math.sqrt(V0)*K) + math.pow(K,2)/10) / (1 + root2/(math.sqrt(V0)*K) + math.pow(K,2)/10)

        else:
            b0 = V0
            b1,b2,a1,a2 = 0,0,0,0

        a = [1,a1,a2]
        b = [b0,b1,b2]
        if self.channel == 1:
            self.data = signal.filtfilt(b, a, self.data)
        elif channel == 2:
            out = numpy.zeros((self.length,2),self.dtype)
            out[:,0] = signal.filtfilt(b, a, self.data[:,0])
            out[:,1] = signal.filtfilt(b, a, self.data[:,1])
            self.data = out

    def peak_filter(self, gain, center_freq, bandwidth):
        Q = self.Fs/bandwidth
        wcT = (2*math.pi*center_freq)/self.Fs

        K = math.tan(wcT/2)
        V = gain

        b0 = 1 + (V*K)/Q + K**2
        b1 = 2*(K**2-1)
        b2 = 1 - (V*K)/Q + K**2
        a0 = 1 + K/Q + K**2
        a1 = b1
        a2 = 1- K/Q + K**2
        a = [1.,a1/a0,a2/a0]
        b = [b0/a0,b1/a0,b2/a0]
        if self.channel == 1:
            self.data = signal.filtfilt(b, a, self.data)
        elif self.channel == 2:
            out = numpy.zeros((self.length,2),self.dtype)
            out[:,0] = signal.filtfilt(b, a, self.data[:,0])
            out[:,1] = signal.filtfilt(b, a, self.data[:,1])
            self.data = out

    def butter_lowpass(self, cutoff_freq, order):
        nyq = 0.5*self.Fs
        normal_cutoff = cutoff_freq/nyq
        b,a = signal.butter(order,normal_cutoff,btype='low',analog=False)
        return b,a

    def butter_highpass(self, cutoff_freq, order):
        nyq = 0.5*self.Fs
        normal_cutoff = cutoff_freq/nyq
        b,a = signal.butter(order,normal_cutoff,btype='high', analog=False)
        return b,a

    def butter_bandpass(self, cutoff_freq_low, cutoff_freq_high, order):
        nyq = 0.5*self.Fs
        cutoff = []
        cutoff.append(cutoff_freq_low/nyq)
        cutoff.append(cutoff_freq_high/nyq)
        b,a = signal.butter(order,cutoff,btype='bandpass', analog=False)
        return b,a

    def butter_bandstop(self, cutoff_freq_low, cutoff_freq_high, order):
        nyq = 0.5*self.Fs
        cutoff = []
        cutoff.append(cutoff_freq_low/nyq)
        cutoff.append(cutoff_freq_high/nyq)
        b,a = signal.butter(order,cutoff,btype='bandstop', analog=False)
        return b,a
                        
    def butter_lowpass_filter(self, cutoff_freq, order = 5):
        b,a = butter_lowpass(cutoff_freq, order = order)
        if self.channel == 1:
            self.data = signal.lfilter(b, a, self.data)
        elif self.channel == 2:
            out = numpy.zeros((self.length,2),self.dtype)
            out[:,0] = signal.filtfilt(b, a, self.data[:,0])
            out[:,1] = signal.filtfilt(b, a, self.data[:,1])
            self.data = out

    def butter_highpass_filter(self, cutoff_freq, order = 5):
        b,a = butter_highpass(cutoff_freq, order = order)
        if self.channel == 1:
            self.data = signal.filtfilt(b, a, self.data)
        elif self.channel == 2:
            out = numpy.zeros((self.length,2),self.dtype)
            out[:,0] = signal.filtfilt(b, a, self.data[:,0])
            out[:,1] = signal.filtfilt(b, a, self.data[:,1])
            self.data = out

    def butter_bandpass_filter(self, cutoff_freq_low, cutoff_freq_high, order = 5):
        b,a = butter_bandpass(cutoff_freq_low, cutoff_freq_high, order = order)
        if self.channel == 1:
            self.data = signal.filtfilt(b, a, self.data)
        elif channel == 2:
            out = numpy.zeros((self.length,2),self.dtype)
            out[:,0] = signal.filtfilt(b, a, self.data[:,0])
            out[:,1] = signal.filtfilt(b, a, self.data[:,1])
            self.data = out

    def butter_bandstop_filter(self, cutoff_freq_low, cutoff_freq_high, order = 5):
        b,a = butter_bandstop(cutoff_freq_low, cutoff_freq_high, order = order)
        if self.channel == 1:
            self.data = signal.filtfilt(b, a, self.data)
        elif self.channel == 2:
            out = numpy.zeros((self.length,2),self.dtype)
            out[:,0] = signal.filtfilt(b, a, self.data[:,0])
            out[:,1] = signal.filtfilt(b, a, self.data[:,1])
            self.data = out
    
    '''Apply flanger to the audio signal'''
    def flanger(self, amp = 1, delay = 1, rate = 1):
        delay = round(delay*self.Fs)
        index = [*numpy.arange(0,self.length)]
        #create oscillating delay
        sin_ref = []
        for i in range(len(index)):
            sin_ref.append(math.sin(2*math.pi*index[i]*(rate/self.Fs)))
        if self.channel == 1:
            y = numpy.zeros(self.length)
            y[:delay] = self.data[:delay]
            for i in range(delay+1,self.length):
                cur_sin = abs(sin_ref[i])
                cur_delay = math.ceil(cur_sin*delay)
                y[i] = (amp*self.data[i]) + amp*(self.data[i-cur_delay])
            self.data = y
        elif self.channel == 2:
            y2 = numpy.zeros((self.length,2),self.dtype)
            for j in range(self.channel):
                y = numpy.zeros(self.length)
                y[:delay] = self.data[:delay,j]
                for i in range(delay+1, self.length):
                    cur_sin = abs(sin_ref[i])
                    cur_delay = math.ceil(cur_sin*delay)
                    y[i] = (amp*self.data[i,j]) + amp*self.data[i-cur_delay,j]
                y2[:,j] = y
            self.data = y2

    '''Modulate the signal'''
    def mod(self, carrier_freq, amp=0.5, phase=0):
        #amp and ring mod
        #if phase = 0 then ring
        index = [*numpy.arange(0,self.length)]
        carrier = []
        for i in range(len(index)):
            carrier.append((phase*math.pi)/180 + amp*math.sin(2*math.pi*index[i]*(carrier_freq/self.Fs)))
        if self.channel == 1:
            self.data = self.data*carrier
        elif self.channel == 2:
            y = numpy.zeros((self.length,2),self.dtype)
            y[:,0] = self.data[:,0]*carrier
            y[:,1] = self.data[:,1]*carrier
            self.data = y

    '''Add tremolo with carrier frequency and amplification'''
    def tremolo(self, carrier_freq, amp=0.5):
        index = [*range(0,self.length)]
        trem = []
        for i in range(len(index)):
            trem.append(1 + amp*math.sin(2*math.pi*index[i]*(carrier_freq/self.Fs)))
        if self.channel == 1:
            self.data = trem*self.data
        elif self.channel == 2:
            y = numpy.zeros((self.length,2),self.dtype)
            y[:,0] = trem*self.data[:,0]
            y[:,1] = trem*self.data[:,1]
            self.data = y

    '''Compression and Expansion of Audio with certain ratio'''
    def compexp(self, comp, a):
        #-1<comp<0 compression
        #0<comp<1 expansion
        #a filter parameter <1
        num = [(1-a)**2]
        den = [1.0, -2*a, math.pow(a,2)]
        if self.channel == 1:
            h = signal.lfilter(num, den, abs(self.data))
            h = h / max(h)
            for i in range(len(h)):
                if h[i] == 0:
                    h[i] = 0
                else:
                    h[i] = numpy.float_power(h[i],comp)
            y = self.data*h
            y = (y*max(abs(self.data)))/max(abs(y))
            self.data = y
        elif self.channel == 2:
            out = numpy.zeros((self.length,2),self.dtype)
            for j in range(self.channel):
                h = signal.lfilter(num, den, abs(self.data[:,j]))
                h = h / max(h)
                for i in range(len(h)):
                    if(h[i] == 0):
                        h[i] = 0
                    else:
                        h[i] = numpy.float_power(h[i],comp)
                y = self.data[:,j]*h
                y = (y*max(abs(self.data[:,j])))/max(abs(y))
                out[:,j] =y
            self.data = out

    '''Add distortion to original signal'''
    def fuzzexp(self, G, mix):
        #Add distortion
        #mix = 1 completely distorted
        if self.channel == 1:
            Q = (self.data*G)/max(abs(self.data))
            z = numpy.sign(-Q)*(1-pow(math.e,numpy.sign(-Q)*Q))
            y = (mix*z*max(abs(self.data)))/max(abs(z)) + (1-mix)*self.data
            self.data = (y*max(abs(self.data)))/max(abs(y))
        elif self.channel == 2:
            n_data = numpy.zeros((self.length,2),self.dtype)
            for i in range(self.channel):
                Q = (self.data[:,i]*G)/max(abs(self.data[:,i]))
                z = numpy.sign(-Q)*(1-pow(math.e,numpy.sign(-Q)*Q))
                y = (mix*z*max(abs(self.data[:,i])))/max(abs(z)) + (1-mix)*self.data[:,i]
                y = (y*max(abs(self.data[:,i])))/max(abs(y))
                n_data[:,i]= y
            self.data = n_data

    '''Add delay to original signal'''
    def delay(self, delay, type ='start', gain = 1):
        delay = round(delay*self.Fs)
        if self.channel == 1:
            if type == 'start':
                y = numpy.concatenate((numpy.zeros(delay), self.data))
                y = y * gain
                maxyb =  max(abs(y))
                self.data = y/maxyb
            elif type == 'end':
                y = numpy.concatenate((self.data,numpy.zeros(delay)))
                y = y * gain
                maxyb =  max(abs(y))
                self.data = y/maxyb
            self.update_self()
        elif self.channel == 2:
            n_data = numpy.zeros((self.length+delay,2),self.dtype)
            if type == 'start':
                for i in range(self.channel):
                    t_data = self.data[:,i]
                    y = numpy.concatenate((numpy.zeros(delay),t_data))
                    y = y * gain
                    maxyb =  max(abs(y))
                    n_data[:,i] = y/maxyb
                self.data = n_data
            elif type == 'end':
                for i in range(self.channel):
                    t_data = self.data[:,i]
                    y = numpy.concatenate((t_data,numpy.zeros(delay)))
                    y = y * gain
                    maxyb =  max(abs(y))
                    n_data[:,i] = y/maxyb
                self.data = n_data
            self.update_self()

    '''Supporting function for reverb'''
    def allpass(self, data, delay, gain = 1):
        if gain >= 1:
            gain = 0.7
        B = numpy.zeros(delay)
        B[0] = gain
        B[delay-1] = 1
        A = numpy.zeros(delay)
        A[0] = 1
        A[delay-1] = gain
        if self.channel == 1:
            y = numpy.zeros(self.length)
            y = signal.lfilter(B, A, data)
            return B,A,y
        elif self.channel == 2:
            output = numpy.zeros((self.length,2),self.dtype)
            for i in range(self.channel):
                y = numpy.zeros(self.length+delay)
                output[:,i] = numpy.zeros(self.length)
                output[:,i] = signal.lfilter(B, A, data[:,i])
            return B,A,output

    '''Supporting function for reverb'''
    def ffcomb(self, data, delay, gain):
        B = numpy.zeros(delay)
        B[0] = 1
        B[delay-1] = gain
        A = 1
        if self.channel == 1:
            y = numpy.zeros(self.length)
            y = signal.lfilter(B, A, data)
            return B,A,y
        elif self.channel == 2:
            output = []
            for i in range(self.channel):
                y = numpy.zeros(self.length)
                y = signal.lfilter(B, A, data[:,i])
                output.append(y)
            return B,A,output

    '''Supporting function for reverb'''
    def fbcomb(self, data, delay, gain):
        if gain >= 1:
            gain = 0.7
        B = numpy.zeros(delay)
        B[delay-1] = 1
        A = numpy.zeros(delay)
        A[0] = 1
        A[delay-1] = -gain
        if self.channel == 1:
            y = numpy.zeros(self.length)
            y = signal.lfilter(B, A, data)
            return B,A,y
        elif self.channel == 2:
            output = []
            for i in range(self.channel):
                y = numpy.zeros(self.length)
                y = signal.lfilter(B, A, data[:,i])
                output.append(y)
            return B,A,output

    '''Supporting function for reverb'''
    def seriescoefficients(self,b1,a1,b2,a2):
        b = numpy.convolve(b1,b2,'valid')
        a = numpy.convolve(a1,a2,'valid')
        return b,a

    '''Add reverberation effect'''
    '''Gain should be less than 1'''
    '''Direct_gain on original signal'''
    '''Delay in ms'''
    def reverb(self, num_allpass, gain, delay, direct_gain):
        #gain should be less than 1
        #delay in ms
        b,a,y = self.allpass(self.data, delay, gain)
        for i in range(2,num_allpass):
            b1,a1,y = self.allpass(y, delay, gain)
            b,a = self.seriescoefficients(b1,a1,b,a)
        if self.channel == 1:
            y = numpy.add(y,direct_gain*self.data)
            maxyb = max(abs(y))
            self.data = y/max(y)
        elif self.channel == 2:
            for i in range(self.channel):
                y[:,i] = numpy.add(y[:,i],direct_gain*self.data[:,i])
                y[:,i] = y[:,i]/max(y[:,i])
            self.data = y

    '''Add echo with amp factor'''
    def echo(self, delay, amp):
        out_delay = delay * self.Fs
        if(out_delay <= self.length):
            if self.channel == 1:
                y = numpy.zeros(self.length+out_delay)
                y[:self.length] = self.data
                self.data = numpy.concatenate((numpy.zeros(out_delay),self.data))
                output = numpy.add(self.data*amp,y)
                maxyb = max(abs(output))
                self.data = output/maxyb
                self.update_self()
            elif self.channel == 2:
                n_data = numpy.zeros((self.length+out_delay,2),self.dtype)
                for i in range(self.channel):
                    y = numpy.zeros(self.length+out_delay)
                    t_data = self.data[:,i]
                    y[:self.length] = self.data[:,i]
                    t_data = numpy.concatenate((numpy.zeros(out_delay),t_data))
                    output = numpy.add(t_data*amp,y)
                    maxyb = max(abs(output))
                    n_data[:,i] = output/maxyb
                self.data= n_data
                self.update_self()

    '''Reverse the audio'''        
    def reverse(self):
        if self.channel == 1:
            self.data = self.data[::-1]
        elif self.channel == 2:
            self.data[:,0] = self.data[::-1,0]
            self.data[:,1] = self.data[::-1,1]

    '''Set Volume of Audio'''
    '''Range from 0.0 to 10.0'''
    def set_volume(self, level):
        self.data = self.data*level

    '''Set Speed of Audio'''
    '''Will shorten or lengthen the audio duration'''
    '''Should be positive'''
    def set_speed(self, speed):
        self.Fs = speed*self.Fs
        self.update_self()

    '''Update SampleRate in Hz'''
    '''Should be positive and max 44100'''
    def set_samplerate(self, newFs):
        self.Fs = newFs
        self.update_self()

    '''Utility Function which checks valid start and end times'''
    def isValidInd(self, ind1, ind2):
        if self.length!= 0 and ind1>=0 and ind2>=0 and ind1<=self.length-1 and ind2<=self.length-1 and ind1<ind2:
            return 1
        return 0

    '''Trims the audio data between start and end'''
    def trim(self, start, end):
        start = int(start*self.Fs)
        end = int(end*self.Fs)
        print(start,end)
        if self.channel == 1:
            if self.isValidInd(start,end):
                self.data = self.data[start:end]
                self.update_self()
        elif self.channel == 2:
            if self.isValidInd(start,end):
                output = numpy.zeros((end-start,2),self.dtype)
                for i in range(self.channel):
                    output[:,i] = self.data[start:end,i]
                self.data = output
                self.update_self()

            
                    
                
        
        
        
        
        
        
            

    
