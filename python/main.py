import wave
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from scipy.signal import butter, lfilter, freqz
from scipy.fft import fft, fftshift

minSNR = 3 #Detection power (dB)
minPulseTime = .005 #s

def openChirpFile(fileDir):
    chirp_file = wave.open(fileDir,mode='rb')
    return chirp_file

def convert2Numpy(waveFile):
    rawBytes = waveFile.readframes(-1)
    if waveFile.getnchannels() > 1:
        raise Exception('Unsupported Channel Type')

    sampwidth = waveFile.getsampwidth()
    dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
    dtype = np.dtype(dtype_map[sampwidth]).newbyteorder('<')  # explicit little-endian

    samples = np.frombuffer(rawBytes, dtype=dtype)

    return samples
        
def mixSignal(waveFile,samples,mixFreq):
    outputSamples = samples.astype(np.complex128)
    for idx,thisSample in enumerate(samples):
        outputSamples[idx] = thisSample * np.exp(1j*2*np.pi*mixFreq*idx/waveFile.getframerate())

    return outputSamples

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_highpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='highpass', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def isolateSignals(waveFile,samples):


    inputSignal = fft(samples)
    freqs = np.fft.fftfreq(len(samples), d=1/waveFile.getframerate())
    searchFreqs_lower = (freqs>1000) & (freqs<10000)
    maxSig = max(np.abs(inputSignal[searchFreqs_lower]))
    targetFreq_lower = freqs[np.abs(inputSignal)==maxSig]
    print("Lower frequency", targetFreq_lower[0])

    searchFreqs_upper = (freqs>10000)
    maxSig = max(np.abs(inputSignal[searchFreqs_upper]))
    targetFreq_upper = freqs[np.abs(inputSignal)==maxSig]
    print("Upper frequency", targetFreq_upper[0])


    samples_lowerSig = mixSignal(waveFile,samples,-targetFreq_lower[0])
    samples_upperSig = mixSignal(waveFile,samples,-targetFreq_upper[0])
    samples_noise = mixSignal(waveFile,samples,-targetFreq_lower[0]-500)
    

    samples_lowerSig = butter_lowpass_filter(samples_lowerSig, 300, waveFile.getframerate(), 6)
    samples_upperSig = butter_lowpass_filter(samples_upperSig, 300, waveFile.getframerate(), 6)
    samples_noise = butter_lowpass_filter(samples_noise, 300, waveFile.getframerate(), 6)



    return (samples_lowerSig,samples_upperSig,samples_noise)

def isolateAllBands(waveFile,samples):


    inputSignal = fft(samples)
    freqs = np.fft.fftfreq(len(samples), d=1/waveFile.getframerate())

    targetBands = []
    for thisFreq in range(1200,waveFile.getframerate()-600,600):

        samples_lowerSig = mixSignal(waveFile,samples,-thisFreq)
        thisBand = butter_lowpass_filter(samples_lowerSig, 300, waveFile.getframerate(), 6)
        targetBands.append(thisBand)

    return targetBands

def removeDC(waveFile,samples):

    inputSignal = fft(samples)
    freqs = np.fft.fftfreq(len(samples), d=1/waveFile.getframerate())
    searchFreqs_lower = (freqs>1000) & (freqs<10000)
    # maxSig = max(np.abs(inputSignal[searchFreqs_lower]))
    # maxSig = waveFile.getframerate()/4

    samples_out = butter_highpass_filter(samples, 1000, waveFile.getframerate(), 5)

    # samples_out = mixSignal(waveFile,samples_out,-maxSig)

    return samples_out



def isolateBursts(fs,samples):
    pass

def plotTimeDomain(waveFile,samples):
    samplePeriod = 1/float(waveFile.getframerate())
    timeDomain = np.array(range(0, len(samples)))*samplePeriod

    plt.plot(timeDomain,10*np.log10(np.abs(samples)))
    # plt.plot(np.fft.fftshift(freqs),np.fft.fftshift(np.abs(inputSignal)))
    plt.show(block=True)

def plotEnvelope(waveFile,samples):
    samplePeriod = 1/float(waveFile.getframerate())
    timeDomain = np.array(range(0, len(samples)))*samplePeriod
    newSamples = sig.hilbert(samples)

    plt.plot(timeDomain,20*np.log10(np.abs(newSamples)))
    # plt.plot(np.fft.fftshift(freqs),np.fft.fftshift(np.abs(inputSignal)))
    plt.show(block=True)

def plotTimeDomainBands(waveFile,samples_lowerSig,samples_upperSig,noiseBand):
    samplePeriod = 1/float(waveFile.getframerate())
    timeDomain = np.array(range(0, len(samples_lowerSig)))*samplePeriod

    plt.plot(timeDomain,20*np.log10(np.abs(samples_lowerSig)),label="Lower Frequency Component")
    plt.plot(timeDomain,20*np.log10(np.abs(samples_upperSig)),label="Upper Frequency Component")
    plt.plot(timeDomain,20*np.log10(np.abs(noiseBand)),label="Noise Band")
    plt.legend(loc="upper right")
    plt.ylabel('Ampliotude [dB_lsb]')
    plt.xlabel('Time [sec]')
    # plt.plot(np.fft.fftshift(freqs),np.fft.fftshift(np.abs(inputSignal)))
    plt.show(block=True)

def plotSpectogram(waveFile,samples):
    samplePeriod = 1/float(waveFile.getframerate())
    timeDomain = np.array(range(0, len(samples)))*samplePeriod

    f, t, Sxx = sig.spectrogram(samples, fs=float(waveFile.getframerate()))
    mesh = plt.pcolormesh(t, f, 20*np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(mesh, label='Power [dB]')
    plt.show(block=True)

def plotFreqDomain(waveFile,samples):

    inputSignal = fft(samples)
    freqs = np.fft.fftfreq(len(samples), d=1/waveFile.getframerate())
    samplePeriod = 1/float(waveFile.getframerate())
    timeDomain = np.array(range(0, len(samples)))*samplePeriod

    plt.plot(np.fft.fftshift(freqs),20*np.log10(np.fft.fftshift(np.abs(inputSignal))))
    plt.show(block=True)

def combineBandPower(samples):
    outputSig = np.square(np.abs(samples[0]))
    samples = samples[1:]
    for thisBand in samples:
        outputSig = outputSig + np.square(np.abs(thisBand))

    return outputSig

def detectPulses(fs,samples):
    noisePower = np.median(outputSig)



if __name__ == '__main__':
    testFile = '/home/rkearns/Documents/Masters/IndependentStudy/Recordings/pennsylvanicus/pennsylvanicus/EH227_250426_0232_bout1_1536s_1550s.wav'
    thisFile = openChirpFile(testFile)

    mySamples = convert2Numpy(thisFile)

    # isoBands = isolateAllBands(thisFile,mySamples)
    # outputSig = combineBandPower(isoBands)
    


    # print(noisePower)
    

    # plotEnvelope(thisFile,mySamples)
    # plotSpectogram(thisFile,mySamples)
    # plotFreqDomain(thisFile,mySamples)
    plotTimeDomain(thisFile,mySamples)


    # (samples_lowerSig,samples_upperSig,noiseBand) = isolateSignals(thisFile,mySamples)
    
    

    





