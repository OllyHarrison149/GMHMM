import numpy

class GMHMM:
    
    def __init__(self,mixtures,transProb,means,covariances,weighting,initialState,states=19,features=13):
        
        self.states = states
        self.mixtures = mixtures
        self.features = features
        self.transProb = transProb
        self.means = means
        self.covariances = covariances
        self.weighting = weighting
        self.initialState = initialState

    # Getters

    def getStates(self):
        
        return self.states

    def getMixtures(self):
        
        return self.mixtures

    def getFeatures(self):
        
        return self.features

    def getTransProb(self):
        
        return transProb

    def getMeans(self):
        
        return means

    def getCovariances(self):
        
        return covariances

    def getWeighting(self):
        
        return weighting

    def getInitialState(self):
        
        return initialState

    # Calculations

    def flowCheck(self,number):
        '''Corrects any underflow or underflow on a number.
        If the number has become infinite, it is stored as the largest representable number.
        If it becomes zero, it is stored as the smallest representable number.

        parameter: number - either a numpy array or a numpy matrix

        returns: number - the corrected number
        '''

        if type(number) == numpy.matrix or type(number) == numpy.array:

            number = numpy.where(numpy.isnan(number)==True,numpy.finfo(float).eps,number)

        return number

    def baumWelch(self,observations):
        
        '''Calculates the baum welch algorithm on a sequence of observations

        parameter: observations - a TxD numpy array of sequence of observations

        output: updates model parameters
        '''
        
        # Calculation of frequently used values
        self.calcPDFVal(observations)
        self.probabilityOfObservation(observations)
        
        # Forward
        alpha = self.calculateAlpha(observations)
        
        # Backward
        beta = self.calculateBeta(observations)
        
        # Update variables
        xi = self.calculateXi(alpha,beta,observations)
        gamma = self.calculateGamma(alpha,beta,observations)
        gammaMix = self.calculateGammaMix(alpha,beta,observations)
        
        # Update parameters
        self.updatePi(gamma)
        self.updateA(gamma,xi,observations)
        self.updateMixtures(observations,gammaMix)

    def calcPDFVal(self,observations):
        
        '''Calculates the PDF values for all observations, for each state and for each mixtures.

        parameter: observations - a TxD numpy array of sequence of observations

        output: pdfVal (attribute) - a NxMxT numpy array of the PDF values
        '''
        
        # NxMxT array
        self.pdfVal = numpy.zeros((self.states,self.mixtures,len(observations)))
        
        # Calculation
        for i in range(self.states):
            
            for t in range(len(observations)):
                
                x = observations[t]
                
                for m in range(self.mixtures):
                    
                    # Variables
                    mu = numpy.matrix(self.means[i][m])
                    sigma = numpy.matrix(self.covariances[i][m])
                    
                    # Formula - using flow checks
                    numerator = numpy.exp(-0.5*numpy.dot(numpy.dot((x-mu),(sigma.I)),(x-mu).T))
                    denominator = ((2*numpy.pi)**(self.features/2))*(abs(numpy.linalg.det(sigma))**0.5)
                    
                    # Assignment - using flow checks
                    self.pdfVal[i][m][t] = self.flowCheck(numerator/denominator)
                    
    def probabilityOfObservation(self,observations):
        
        '''Calculates the probability of an observation happening for all observations at all states, taking into account the weighting of the mixtures.

        parameter: observations - a TxD numpy array of sequence of observations

        output: observationProbability (attribute) - a NxT numpy array of observation probabilities
        '''
        
        # NxT array
        self.observationProbability = numpy.zeros((self.states,len(observations)))
        
        # Calculation
        for i in range(self.states):
            
            for t in range(len(observations)):
                
                for m in range(self.mixtures):
                    
                    self.observationProbability[i][t] += self.weighting[i][m]*self.pdfVal[i][m][t]
                    self.observationProbability[i][t] = self.flowCheck(self.observationProbability[i][t])

    def calculateAlpha(self,observations):
        
        '''Calculates the forward algorithm.

        parameter: observations - a TxD numpy array of sequence of observations

        returns: alpha - a TxN numpy array 
        '''
        
        # TxN array
        alpha = numpy.zeros((len(observations),self.states))
        
        # Initial (t=1)
        for i in range(self.states):
            
            alpha[0][i] = self.flowCheck(self.initialState[i]*self.observationProbability[i][0])
            
        # Make each entry a proportion of sum
        alpha[0] = alpha[0]/numpy.sum(alpha[0])
        
        # Induce
        for t in range(1,len(observations)):
            
            for i in range(self.states):
                
                for j in range(self.states):
                    
                    alpha[t][i] += alpha[t-1][j]*self.transProb[j][i]
                    
                alpha[t][i] *= self.observationProbability[i][t]
                alpha[t][i] = self.flowCheck(alpha[t][i])
                
            # Make each entry a proportion of sum (except at t=T)
            if t != len(observations)-1:
                
                alpha[t] = alpha[t]/numpy.sum(alpha[t])
                
        return alpha

    def calculateBeta(self,observations):
        
        '''Calculates the backward algorithm.

        parameter: observations - a TxD numpy array of sequence of observations

        returns: beta - a TxN numpy array
        '''
        
        # TxN array
        beta = numpy.zeros((len(observations),self.states))
        
        # Initial (t=T)
        for i in range(self.states):
            
            beta[len(observations)-1][i] = 1
            
        # Induce
        for t in range(len(observations)-2,-1,-1):
            
            for i in range(self.states):
                
                for j in range(self.states):
                    
                    beta[t][i] += self.transProb[i][j]*self.observationProbability[j][t+1]*beta[t+1][j]
                    
                beta[t][i] = self.flowCheck(beta[t][i])
                
            # Make each entry a proportion of sum
            beta[t] = beta[t]/numpy.sum(beta[t])
            
        return beta
        
    def calculateXi(self,alpha,beta,observations):
        
        '''Calculates one of the update variables.

        parameter: alpha - a TxN numpy array calculated from calculateAlpha method
        parameter: beta - a TxN numpy array calculated from calculateBeta method
        parameter: observations - TxD numpy array of sequence of observations

        returns: xi - TxNxN numpy array
        '''
        
        # TxNxN array
        xi = numpy.zeros((len(observations),self.states,self.states))
        
        # Calculation
        for t in range(len(observations)-1):
            
            # Denominator reset
            denominator = 0
            
            # New denominator
            for i in range(self.states):
                
                for j in range(self.states):
                    
                    denominator += alpha[t][i]*self.transProb[i][j]*self.observationProbability[j][t+1]*beta[t+1][j]
                    
            for i in range(self.states):
                
                for j in range(self.states):
                    
                    # Numerator reset
                    numerator = 1
                    
                    # New numerator
                    numerator *= alpha[t][i]*self.transProb[i][j]*self.observationProbability[j][t+1]*beta[t+1][j]
                    
                    # Assignment
                    xi[t][i][j] = self.flowCheck(numerator/denominator)
                    
        return xi
        
    def calculateGamma(self,alpha,beta,observations):
        
        '''Calculates one of the update variables.

        parameter: alpha - a TxN numpy array calculated from calculateAlpha method
        parameter: beta - a TxN numpy array calculated from calculateBeta method
        parameter: observations - a TxD numpy array of sequence of observations

        returns: gamma - TxN numpy array
        '''
        
        # TxN array
        gamma = numpy.zeros((len(observations),self.states))
        
        # Calculation
        for t in range(len(observations)):
            
            # Denominator reset
            denominator = 0
            
            # New denominator
            for j in range(self.states):
                
                denominator += alpha[t][j]*beta[t][j]
                
            for i in range(self.states):
                
                numerator = alpha[t][i]*beta[t][i]
                
                # Assignment
                gamma[t][i] = self.flowCheck(numerator/denominator)
                
        return gamma

    def calculateGammaMix(self,alpha,beta,observations):
        
        '''Calculates one of the update variables.

        parameter: alpha - a TxN numpy array calculated from calculateAlpha method
        parameter: beta - a TxN numpy array calculated from calculateBeta method
        parameter: observations - a TxD numpy array of sequence of observations

        returns: gammaMix - TxNxM numpy array
        '''
        
        # TxNxM array
        gammaMix = numpy.zeros((len(observations),self.states,self.mixtures))
        
        # Calculation
        for t in range(len(observations)):
            
            for i in range(self.states):
                
                for m in range(self.mixtures):
                    
                    # LHS Denominator reset
                    denominator1 = 0
                    
                    for j in range(self.states):
                        
                        denominator1 += alpha[t][j]*beta[t][j]
                        
                    # RHS Denominator reset
                    denominator2 = 0
                    
                    for k in range(self.mixtures):
                        
                        denominator2 += self.weighting[i][k]*self.pdfVal[i][k][t]
                        
                    # Numerator
                    numerator = alpha[t][i]*beta[t][i]*self.weighting[i][m]*self.pdfVal[i][m][t]
                    
                    # Assignment
                    gammaMix[t][i][m] = self.flowCheck(numerator/(denominator1*denominator2))
                    
        return gammaMix

    def updatePi(self,gamma):
        
        '''Updates the pi model parameter.

        parameter: gamma - TxN numpy array calculated from calculateGamma method

        output: initialState (attribute) - updated to the first index of the gamma array
        '''
        
        self.initialState = gamma[0]

    def updateA(self,gamma,xi,observations):
        
        '''Updates the transProb model parameter.

        parameter: gamma - TxN numpy array calculated from calculateGamma method
        parameter: xi - TxNxN numpy array calculated from calculateXi method
        parameter: observations - a TxD numpy array of sequence of observations

        output: transProb (attribute) - updated
        '''
        
        # NxN array
        newA = numpy.zeros((self.states,self.states))
        
        # Calculation
        for i in range(self.states):
            
            for j in range(self.states):
                
                numerator = 0
                denominator = 0
                
                for t in range(len(observations)-1):
                    
                    numerator += xi[t][i][j]
                    denominator += gamma[t][i]
                    
                # Assignment
                newA[i][j] = self.flowCheck(numerator/denominator)
                
        # Update
        self.transProb = newA

    def updateMixtures(self,observations,gammaMix):
        
        '''Updates mixtures parameters for the model.

        parameter: observations - a TxD numpy array of sequence of observations
        parameter: gammaMix - TxNxM numpy array calculated from calculateGammaMix method

        output: weighting (attribute) - updated
        output: means (attribute) - updated
        output: covariances (attribute) - updated
        '''
        
        # Weighting Component
        
        # NxM array
        newW = numpy.zeros((self.states,self.mixtures))
        
        # Calculation
        for i in range(self.states):
            
            for m in range(self.mixtures):
                
                denominator = 0
                numerator = 0
                
                for t in range(len(observations)):
                    
                    # Denominator
                    for k in range(self.mixtures):
                        
                        denominator += gammaMix[t][i][k]
                        
                    # Numerator
                    numerator += gammaMix[t][i][m]
                    
                # Assignment
                newW[i][m] = self.flowCheck(numerator/denominator)

        # Means Component
        
        # NxMxD array
        newMeans = numpy.zeros((self.states,self.mixtures,self.features))
        
        # Calculation
        for i in range(self.states):
            
            for m in range(self.mixtures):
                
                # D arrays
                numerator = numpy.zeros((self.features))
                denominator = numpy.zeros((self.features))
                
                # Calculation
                for t in range(len(observations)):
                    
                    numerator += gammaMix[t][i][m]*observations[t]
                    denominator += gammaMix[t][i][m]
                    
                # Assignment
                newMeans[i][m] = self.flowCheck(numerator/denominator)

        # Covariances Component
        
        # NxM array containing DxD matrices
        newCovariance = [[numpy.matrix(numpy.zeros((self.features,self.features)))for m in range(self.mixtures)]for i in range(self.states)]
        minCovariance = [[numpy.matrix(0.01*numpy.eye((self.features))) for m in range(self.mixtures)] for i in range(self.states)]
        
        # Calculation
        for i in range(self.states):
            
            for m in range(self.mixtures):
                
                # DxD matrices
                numerator = numpy.matrix(numpy.zeros((self.features,self.features)))
                denominator = numpy.matrix(numpy.zeros((self.features,self.features)))
                
                # Calculation
                for t in range(len(observations)):
                    
                    numerator += gammaMix[t][i][m]*(numpy.dot((observations[t]-self.means[i][m]).T,(observations[t]-self.means[i][m])))
                    denominator += gammaMix[t][i][m]
                    
                # Assignment
                newCovariance[i][m] = self.flowCheck(numerator/denominator)
                newCovariance[i][m] = newCovariance[i][m] + minCovariance[i][m]

        # Update
        self.weighting = newW
        self.means = newMeans
        self.covariances = newCovariance
        
    def calcLogLikelihood(self,observations):
        
        '''Calculates the log likelihood of the observations.

        parameter: observations - a TxD numpy array of sequence of observations

        returns: logLikelihood - float
        '''
        
        # Fresh observation probabilities
        self.calcPDFVal(observations)
        self.probabilityOfObservation(observations)
        
        # Forward algorithm
        alpha = self.calculateAlpha(observations)
        
        # Log likelihood
        finalObsProb = alpha[-1]
        logLikelihood = self.flowCheck(numpy.log(sum(finalObsProb)))
        
        return logLikelihood

    def train(self,observations,iterations,threshold,convergeThresh):
        
        '''A method used to train the model.

        parameter: observations - a list containing all training data
        parameter: iterations - integer of how many iterations to do over training data (0 or higher)
        parameter: threshold - a threshold for change in log likelihood (higher than 0)

        returns: None unless parameters are incorrect
        '''
        
        # Data validation
        if iterations < 0:
            
            return "Iterations must be positive."
        
        elif threshold <= 0:
            
            return "Threshold value must be more than zero."
        
        elif isinstance(observations,list):
            
            return "Obserations must be a list of observations."
        
        converged = False
        iteration = 0
        
        while (not converged) and (iteration != iterations):
            
            print("Iteration number: ",iteration+1)
            
            # Iterate across all data
            for data in observations:
                
                # Current likelihood
                probOld = self.calcLogLikelihood(data)
                
                # Train
                self.baumWelch(data)
                
                # New likelihood
                probNew = self.calcLogLikelihood(data)
                
                # Training Output
                print("    Change in log likelihood: ",probNew-probOld)
                print("    Trained: ",abs(probNew-probOld)<threshold)
                print("    Converging: ",(probNew-probOld)>convergeThresh)
                
                # Calculate if converged
                if abs(probNew-probOld) < threshold:
                    
                    # Converged
                    converged = True
                    
            iteration += 1
            
        # Inform of outcome
        if converged == True:
            
            print("Model has converged.")
            
        else:
            
            print("Iterations completed without convergence.")


    def viterbi(self,observations,statesArray,collapseStates=True):
        
        '''Calculates the most probable path for a given set of observations.

        parameter: observations - a TxD numpy array of sequence of observations
        parameter: statesArray - an Nx1 numpy array of the states that the observations can represent
        parameter: collapseStates - a boolean denoting whether or not consecutive same states are wanted

        returns: path - an array of length T of most probable states at each time
        '''
        
        # Data validation
        if numpy.shape(observations)[1] != self.features:
            
            return "Observations not in the right format."
        
        if len(statesArray) != self.states:
            
            return "Array of states does not have the right number of states."
        
        # New observation data
        self.calcPDFVal(observations)
        self.probabilityOfObservation(observations)
        
        # TxN arrays
        delta = numpy.zeros((len(observations),self.states))
        psi = numpy.zeros((len(observations),self.states))
        
        # T array
        path = numpy.zeros((len(observations)))
        
        # Initial probability
        for i in range(self.states):
            
            delta[0][i] = self.initialState[i]*self.observationProbability[i][0]
            psi[0][i] = 0
            
        # As t increases
        for t in range(1,len(observations)):
            
            for i in range(self.states):
                
                for j in range(self.states):
                    
                    if delta[t][i] < delta[t-1][j]*self.transProb[j][i]:
                        
                        delta[t][i] = delta[t-1][j]*self.transProb[j][i]
                        psi[t][i] = j
                        
                delta[t][i] *= self.observationProbability[i][t]
                
        # Max probability
        maxProb = 0
        
        # Iterate over final state
        for i in range(self.states):
            
            if maxProb < delta[len(observations)-1][i]:
                
                maxProb = delta[len(observations)-1][i]
                path[len(observations)-1] = i
                
        # Find path
        print("Max Prob:",maxProb)
        
        for t in range(1,len(observations)):
            
            path[len(observations)-t-1] = psi[len(observations)-t][int(path[len(observations)-t])]
            
        # Convert path to states
        path = self.pathToStates(path,statesArray,collapseStates)
        
        return path

    def pathToStates(self,path,statesArray,collapseStates):
        
        '''Converts integers in path array to states in statesArray.

        parameter: path - an array of length T containing integers
        parameter: statesArray - an Nx1 numpy array of the states that the observations can represent
        parameter: collapseStates - a boolean denoting whether or not consecutive same states are wanted

        returns: path - an array of length T of most probable states at each time
        '''
        
        path = path.astype(int)
        path = list(path)
        
        # Consecutive states
        if collapseStates == True:
            
            index = 0
            
            while index != len(path)-1:
                
                # If current state is the same as the next
                if path[index] == path[index+1]:
                    
                    # Remove next state
                    path.pop(index+1)
                    
                else:
                    
                    index += 1
                    
        # Convert to phonemes
        for i in range(len(path)):
            
            state = path[i]
            path[i] = statesArray[state]
            
        return path
        


    


