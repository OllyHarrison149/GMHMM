import numpy
import decimal
import math
from scipy.fftpack import dct

class Audio:
    
    def __init__(self,rate,signal,frames,frameLength=0.025,frameStep=0.01):
        
        # Parameters
        self.rate = rate
        self.signal = signal
        self.frames = frames
        self.frameLength = frameLength
        self.frameStep = frameStep
        
        # Need to be set
        self.NFFT = None
        self.coefficients = None

    # Getters

    def getCoefficients(self):
        
        return self.coefficients

    # Setters

    def setNFFT(self):
        
        # Calculates the value of NFFT as a power of 2
        self.NFFT = 1
        
        # Frame length in terms of samples
        frameLengthRate = self.frameLength * self.rate
        
        while self.NFFT < frameLengthRate:
            
            self.NFFT *= 2

    def setCoefficients(self,nfilt,preEmph,cepLifter,numCoefficients,winFunc=lambda x: numpy.hamming(x)):
        
        '''This function creates the MFCCs given some audio by calling other individual functions to provide calculations.

        parameter: nfilt - an integer of the number of desired filters for the Mel filterbank
        parameter: preEmph - a float of the pre-emphasis coefficient between 0 and 1
        parameter: cepLifter - an integer of the liftering coefficient
        parameter: numCoefficients - an integer of how many coefficients wanted
        parameter: winFunc - an analysis function that will be applied to each frame (default is the numpy hamming function)

        output: coefficients are calculated and set to the coefficients attribute
        '''
        
        # Calculates NFFT if not already
        if self.NFFT == None:
            
            self.setNFFT()
            
        # Sets variables
        lowestFreq = 0
        highestFreq = self.rate/2
        
        # Calculate MFCCs
        
        # Frame signal
        self.preEmphasis(preEmph)
        self.frameSignal(winFunc)
        
        # Periodogram estimate
        frequenciesPresent = self.dft()
        estimate = self.periodogram(frequenciesPresent)
        
        # Filterbanks
        filterBanks = self.getFilterBanks(highestFreq,lowestFreq,nfilt)
        energyFilterBank = self.filterBank(estimate,filterBanks)
        
        # Log
        energySumLog = numpy.log(energyFilterBank)
        
        # Set coefficients
        self.coefficients = dct(energySumLog,type=2,axis=1,norm='ortho')[:,:numCoefficients]
        
        # Lift coefficients
        self.lift(cepLifter,numCoefficients)
        self.normalise()

    def preEmphasis(self,preEmph):
        
        '''This function performs pre-emphasis on the signal.

        parameter: preEmph - a float containing the pre-emphasis coefficient between 0 and 1

        output: signal - an array of the same order as the input
        '''
        
        # Applies pre-emphasis to the signal
        for frame in range(self.frames-1,0,-1):
            
            self.signal[frame] = self.signal[frame] - preEmph*self.signal[frame-1]

    def roundHalfUp(self,number):
        
        # Rounds numbers properly
        return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding = decimal.ROUND_HALF_UP))

    def strideTrick(self,padSignal,frameLengthRate,frameStepRate):
        
        '''A function that applies the stride trick to a given signal.

        parameter: padSignal - an array of the signal, padded with zeros
        parameter: frameLengthRate - a float of the frame length multiplied by the rate
        parameter: frameStepRate - a float of the frame step multiplied by the rate

        returns: frames - an array of order numFrames*1
        '''
        
        shape = padSignal.shape[:-1]+(padSignal.shape[-1]-frameLengthRate+1,frameLengthRate)
        strides = padSignal.strides+(padSignal.strides[-1],)
        frames = numpy.lib.stride_tricks.as_strided(padSignal,shape,strides)[::frameStepRate]
        
        return frames

    def frameSignal(self,winFunc):
        
        '''A function that frames the audio signal using a given window function into overlapping frames.

        parameter: winFun - the analysis function applied to each frame (a lambda function as we pass this a variable)

        output: frames - an array of the framed signal of order numFrames*frameLengthRate
        '''
        
        # Frames signal into overlapping frames
        frameLengthRate = self.roundHalfUp(self.frameLength*self.rate)
        frameStepRate = self.roundHalfUp(self.frameStep*self.rate)
        
        # Flatten signal
        signal = self.signal.flatten()
        
        # Calculate number of frames
        if self.frames <= frameLengthRate:
            
            numFrames = 1
            
        else:
            
            numFrames = 1 + int(math.ceil((self.frames-frameLengthRate)/frameStepRate))
            
        # Frame signal
        padLength = int((numFrames-1)*frameStepRate+frameLengthRate)
        zeros = numpy.zeros(padLength-self.frames)
        padSignal = numpy.concatenate((signal,zeros))
        window = winFunc(frameLengthRate)
        frames = self.strideTrick(padSignal,frameLengthRate,frameStepRate)
        
        # Set signal
        self.signal = window*frames

    def dft(self):
        
        # Calculates DFT of the audio
        
        # Complex
        complexSpec = numpy.fft.rfft(self.signal,self.NFFT)
        
        # Real
        absoluteSpec = numpy.absolute(complexSpec)
        
        return absoluteSpec

    def periodogram(self,absoluteSpec):
        
        '''A function that calculates the periodogram estimate for a piece of audio.

        parameter: absoluteSpec - a real-valued array of order N*1
    
        returns: estimate - a real-valued array of the audio, where each row is the power spectrum of the corresponding frame
        '''
        
        # Calculates periodogram estimate
        return (1.0/self.NFFT)*numpy.square(absoluteSpec)

    def hertzToMel(self,hertz):
        
        # Convert to Mel from Hertz
        return 2595 * numpy.log10(1+(hertz/700))

    def melToHertz(self,mel):
        
        # Convert to Hertz from Mel
        return 700 * 10**(mel/2595) - 700

    def getFilterBanks(self,highestFreq,lowestFreq,nfilt):
        
        '''Calculates the mel filterbanks for the audio signal based on the upper and lower frequencies.

        parameter: highestFreq - an integer denoting the highest possible frequency in the audio
        parameter: lowestFreq - an integer denoting the lowest possible frequency in the audio
        parameter: nfilt - an integer of the number of filterbanks to be created

        returns: filterBank - an array of order nfilt by NFFT//2 + 1 where one row contains one filterbank
        '''
        
        # Calculates filterbanks given frequency range and number of filters
        lowestMel = self.hertzToMel(lowestFreq)
        highestMel = self.hertzToMel(highestFreq)
        
        # Linearly spaced Mel points
        melPoints = numpy.linspace(lowestMel,highestMel,nfilt+2)
        hertzPoints = self.melToHertz(melPoints)
        
        # FFT bins
        fftBin = numpy.floor((self.NFFT+1)*hertzPoints/self.rate)
        filterBanks = numpy.zeros([nfilt,int(numpy.floor(self.NFFT/2+1))])
        
        # Calculate filterbanks
        for f in range(nfilt):
            
            for k in range(int(fftBin[f]),int(fftBin[f+1])):
                
                filterBanks[f,k] = (k-fftBin[f])/(fftBin[f+1]-fftBin[f])
                
            for k in range(int(fftBin[f+1]),int(fftBin[f+2])):
                
                filterBanks[f,k] = (fftBin[f+2]-f)/(fftBin[f+2]-fftBin[f+1])
                
        return filterBanks

    def filterBank(self,estimate,filterBanks):
        
        '''This function calculates the energy features of the audio in the Mel-filterbanks.

        parameter: estimate - a real-valued array of the power spectrum of each frame
        parameter: filterBank - an array of order nfilt*1 where one row contains one filterbank

        returns: feat - an array of size numFrames*nfilt holding the audio features, with each row holding one
        '''
        
        # Calculate feat
        feat = numpy.dot(estimate,filterBanks.T)
        feat = numpy.where(feat==0,numpy.finfo(float).eps,feat)
        
        return feat

    def lift(self,cepLifter,numCoefficients):
        
        '''Applies liftering to coefficients.

        parameter: cepLifter - an integer denoting the liftering coefficient
        parameter: numCoefficients - an integer of how many coefficients wanted

        output: lifted coefficients
        '''
        
        # Applies liftering to coefficients
        num = numpy.arange(numCoefficients)
        lifter = 1+(cepLifter/2)*numpy.sin((numpy.pi*num)/cepLifter)
        
        # Sets coefficients
        self.coefficients = self.coefficients*lifter

    def normalise(self):
        
        '''Applies min-max feature scaling to coefficients.'''
        
        # Find min and max coefficients
        maxCoefficient = numpy.amax(self.coefficients)
        minCoefficient = numpy.amin(self.coefficients)
        
        # Apply normalisation
        self.coefficients = (self.coefficients-minCoefficient)/(maxCoefficient-minCoefficient)


