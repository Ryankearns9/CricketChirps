import wave
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from scipy.signal import butter, lfilter, freqz
from scipy.fft import fft, fftshift
from scipy.io import wavfile as wv


minPulseTime = .005 #s
minStartThreshold = 1000 #samples


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
    
def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
    
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
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


#    samples_lowerSig = mixSignal(waveFile,samples,-targetFreq_lower[0])
#    samples_upperSig = mixSignal(waveFile,samples,-targetFreq_upper[0])
#    samples_noise = mixSignal(waveFile,samples,-targetFreq_lower[0]-500)
    
    print(targetFreq_lower)

    samples_lowerSig = butter_bandpass_filter(samples, targetFreq_lower[0] - 300, targetFreq_lower[0] + 300,  waveFile.getframerate(), 6)
    samples_upperSig = butter_bandpass_filter(samples, targetFreq_upper[0] - 300, targetFreq_upper[0] + 300, waveFile.getframerate(), 6)
    
    
#    samples_noise = butter_lowpass_filter(samples_noise, 300, waveFile.getframerate(), 6)



    return (samples_lowerSig,samples_upperSig)

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



def plotTimeDomain(waveFile,samples):
    samplePeriod = 1/float(waveFile.getframerate())
    timeDomain = np.array(range(0, len(samples)))*samplePeriod

    plt.plot(timeDomain,10*np.log10(np.abs(samples)))
    plt.ylabel('Audio Power [dB]')
    plt.xlabel('Time [sec]')
    plt.title('Combined Signal Time Domain')
    # plt.plot(np.fft.fftshift(freqs),np.fft.fftshift(np.abs(inputSignal)))
    plt.show(block=True)

def plotEnvelope(waveFile,samples):
    samplePeriod = 1/float(waveFile.getframerate())
    timeDomain = np.array(range(0, len(samples)))*samplePeriod
    newSamples = sig.hilbert(samples)

    plt.plot(timeDomain,20*np.log10(np.abs(newSamples)))
    plt.ylabel('Audio Power [dB]')
    plt.xlabel('Time [sec]')
    plt.title('Input Signal Time Domain')
    # plt.plot(np.fft.fftshift(freqs),np.fft.fftshift(np.abs(inputSignal)))
    plt.show(block=True)

def plotTimeDomainBands(waveFile,samples_lowerSig,samples_upperSig,noiseBand):
    samplePeriod = 1/float(waveFile.getframerate())
    timeDomain = np.array(range(0, len(samples_lowerSig)))*samplePeriod

    plt.plot(timeDomain,20*np.log10(np.abs(samples_lowerSig)),label="Lower Frequency Component")
    plt.plot(timeDomain,20*np.log10(np.abs(samples_upperSig)),label="Upper Frequency Component")
    plt.plot(timeDomain,20*np.log10(np.abs(noiseBand)),label="Noise Band")
    plt.legend(loc="upper right")
    plt.ylabel('Amplitude [dB_lsb]')
    plt.xlabel('Time [sec]')
    # plt.plot(np.fft.fftshift(freqs),np.fft.fftshift(np.abs(inputSignal)))
    plt.show(block=True)
    
def plotSamples(samples,samples2,samples3):

    plt.plot(10*np.log10(np.abs(samples)),label="Signal Power Data")
    plt.plot(10*np.log10(np.abs(samples2)),label="Moving Average Power")
    plt.plot(max(10*np.log10(np.abs(samples)))*samples3,label="Detected Burst Location")

    plt.ylabel('Power [dB_lsb]')
    plt.xlabel('Sample Idx')
    plt.title('Burst Isolation Algorithm')
    plt.legend(loc="upper right")
    # plt.plot(np.fft.fftshift(freqs),np.fft.fftshift(np.abs(inputSignal)))
    plt.show(block=True)

def plotSpectogram(waveFile,samples):
    samplePeriod = 1/float(waveFile.getframerate())
    timeDomain = np.array(range(0, len(samples)))*samplePeriod

    f, t, Sxx = sig.spectrogram(samples, fs=float(waveFile.getframerate()))
    mesh = plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram (Linear Amplitude [lsbs])')
    plt.colorbar(mesh, label='Amplitude [lsbs]')
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

def calcNoisePower(samples):
    return np.median(samples)

def detectBurstLocations(noisePower, minDetectSNR, samples, intLength, plotData=False):
    #Below is the unoptimized but is getting the correct index locations
    outputBuffer = np.convolve(samples,np.ones(intLength)/intLength,'same')
    #Below is the optimized version but is messing up indexes. Need to adjust samples to half int length
    '''
    accumList = np.zeros(intLength)
    accum = 0
    outputBuffer = np.array([])
    
    for thisSample in samples:
        accumList = np.roll(accumList,1)
        accum = accum - accumList[0] + thisSample
        accumList[0] = thisSample
        
    
        outputBuffer = np.append(outputBuffer,accum/intLength)
    '''
    
    minDetectThreshold = np.pow(10,(noisePower+minDetectSNR)/20)
    burstDetects = outputBuffer > minDetectThreshold
    starts = burstDetects.copy()
    starts[1:] = burstDetects[1:] & ~burstDetects[:-1]
    numBursts = sum(starts)
    
    burstStarts = np.flatnonzero(starts)
    burstEnds = np.flatnonzero(~burstDetects[1:] & burstDetects[:-1])
    
    #Don't count any within the first 1000 samples
    if (any(burstStarts < 1000)):
        voidNumbers = np.flatnonzero(burstStarts < 1000)
        for idx in range(0,len(voidNumbers)):
            burstStarts = np.delete(burstStarts,0)
            burstEnds = np.delete(burstEnds,0)
            numBursts+=-1
            
    if plotData:
        plotSamples(samples,outputBuffer,burstDetects)
        
    return (numBursts,burstStarts,burstEnds)
    
def plotSnippets(bursts):
    for idx,thisBurst in enumerate(bursts):
        plt.plot(20*np.log10(np.abs(thisBurst)),label="Burst "+str(idx+1))
        plt.ylabel('Noise Power [dB]')
        plt.xlabel('Sample [idx]')
        plt.title("Burst "+str(idx+1))
        plt.show(block=True)
    
    for idx,thisBurst in enumerate(bursts):
        plt.plot(20*np.log10(np.abs(thisBurst)),label="Burst "+str(idx+1))
    plt.legend(loc="upper right")
    plt.show(block=True)

def snippitizer(samples,burstStarts, burstEnds,fs,plotData=False):
    snippets = []
    for targetIdx,startIdx in enumerate(burstStarts):
        snippets.append(samples[burstStarts[targetIdx]:burstEnds[targetIdx]])
        
    if plotData:
        plotSnippets(snippets)
    return snippets
    
    
def saveSlowedVersion(samples,fs,slowFactor,fileName):
    wv.write(fileName,fs//slowFactor,samples.astype("int16"))

if __name__ == '__main__':
    testFile = '../Recordings/pennsylvanicus/pennsylvanicus/EH227_250426_0232_bout1_1536s_1550s.wav'
    minSNR = 3 #Detection power (dB)
    
    thisFile = openChirpFile(testFile)

    mySamples = convert2Numpy(thisFile)
    
#    (lowerSamples,upperSamples) = isolateSignals(thisFile,mySamples)

    isoBands = isolateAllBands(thisFile,mySamples)
    outputSig = combineBandPower(isoBands)
    noisePower = calcNoisePower(outputSig)
#
#
    (numBursts,burstStarts,burstEnds) = detectBurstLocations(noisePower,minSNR, outputSig, 5000, plotData=True)
    
    
    
#    print("Num Bursts: ", numBursts)
#    print("Burst Starts: ", burstStarts)
#    print("Burst Ends: ", burstEnds)
    
#    burstSnippets = snippitizer(mySamples,burstStarts,burstEnds,thisFile.getframerate(),plotData=False)
#    saveSlowedVersion(burstSnippets[0],thisFile.getframerate(),10,"Burst1_10xSlow.wav")
#    saveSlowedVersion(lowerSamples,thisFile.getframerate(),1,"lower_filtered.wav")
#    saveSlowedVersion(upperSamples,thisFile.getframerate(),1,"upper_filtered.wav")
#    saveSlowedVersion(burstSnippets[0],thisFile.getframerate(),20,"Burst1_20xSlow.wav")
    
    
    
    

    # print(noisePower)
    

    # plotEnvelope(thisFile,mySamples)
#    plotSpectogram(thisFile,mySamples)
    # plotFreqDomain(thisFile,mySamples)
    plotTimeDomain(thisFile,outputSig)
#    plotSamples(smoothingCruve,outputSig,burstLocations)


    # (samples_lowerSig,samples_upperSig,noiseBand) = isolateSignals(thisFile,mySamples)
    
    

    





