import collections, util, math, random

############################################################
class ValueIteration(util.MDPAlgorithm):
    '''
    Solve the MDP using value iteration.  Your solve() method must set
    - self.V to the dictionary mapping states to optimal values
    - self.pi to the dictionary mapping states to an optimal action
    Note: epsilon is the error tolerance: you should stop value iteration when
    all of the values change by less than epsilon.
    The ValueIteration class is a subclass of util.MDPAlgorithm (see util.py).
    '''
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        def computeQ(mdp, V, state, action):
            # Return Q(state, action) based on V(state).
            return sum(prob * (reward + mdp.discount() * V[newState]) \
                            for newState, prob, reward in mdp.succAndProbReward(state, action))

        def computeOptimalPolicy(mdp, V):
            # Return the optimal policy given the values V.
            pi = {}
            for state in mdp.states:
                pi[state] = max((computeQ(mdp, V, state, action), action) for action in mdp.actions(state))[1]
            return pi

        V = collections.Counter()  # state -> value of state
        numIters = 0
        while True:
            newV = {}
            for state in mdp.states:
                newV[state] = max(computeQ(mdp, V, state, action) for action in mdp.actions(state))
            numIters += 1
            if max(abs(V[state] - newV[state]) for state in mdp.states) < epsilon:
                V = newV
                break
            V = newV

        # Compute the optimal policy now
        pi = computeOptimalPolicy(mdp, V)
        print "ValueIteration: %d iterations" % numIters
        self.pi = pi
        self.V = V

############################################################
# Problem 2a

# If you decide 2a is true, prove it in writeup.pdf and put "return None" for
# the code blocks below.  If you decide that 2a is false, construct a counterexample.
class CounterexampleMDP(util.MDP):
    def startState(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return 
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    # Return set of actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def discount(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost
        
    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.
    # The second element is the index (not the value) of the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None. 
    # When the probability is 0 for a particular transition, don't include that 
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (around 55 lines of code expected)
        ret = []
    
        deck = state[2] 
        nonZeros = 0
        ###check deck first
        if deck is None: return []
        else:
            for i in state[2] :
                if i > 0:
                    nonZeros += 1
        total = float(sum(deck))
        isPrempty = total == 1

        ###Take action
        def take():
            if state[1]!=None:
                top = state[1]
                new_hand = self.cardValues[top]+state[0]
                new_deck = deck[:top] + (deck[top]-1,) + deck[top+1:]
                if new_hand > self.threshold:
                    next_state = (new_hand, None, None)
                else:
                    next_state = (new_hand, None, new_deck)
                ret.append((next_state, 1, 0))              
            else:
                for i in range(len(deck)):
                    if deck[i] > 0:                   
                        new_hand = state[0] + self.cardValues[i]
                        new_deck = deck[:i] + (deck[i]-1,) + deck[i+1:]
                        if new_hand>self.threshold:
                            next_state = (new_hand, None, None) 
                            ret.append((next_state, deck[i]/total, 0))       
                        else:
                            if isPrempty:
                                next_state = (new_hand, None, None)  
                                ret.append((next_state, 1, new_hand))
                            else:
                                next_state = (new_hand, None, new_deck)
                                ret.append((next_state, deck[i]/total, 0)) 
                 

        ###Peek action
        def peek():
            if state[1] != None:
                return []
            for i in range(len(deck)):
                if deck[i] > 0:
                    next_state = (state[0], i, deck)
                    ret.append((next_state, deck[i]/total, -self.peekCost))

        def quit():
            next_state = (state[0], None, None)
            ret.append((next_state, 1, state[0]))
 

        actions = { 'Take': take, 'Peek': peek, 'Quit': quit,}
        actions[action]()

        ###END_YOUR_CODE
        return ret
        raise Exception("Not implemented yet")


    def discount(self):
        return 1

############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the optimal action at
    least 10% of the time.
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    peekMDP = BlackjackMDP(cardValues=[3, 5, 17], multiplicity=2, threshold=20, peekCost=1)
    vi = ValueIteration()
    vi.solve(peekMDP)
    return peekMDP



    raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = collections.Counter()
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        size = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (around 15 lines of code expected)
        
        def get_v(state):
            if state == None: return 0
            ###if  type(state) == 'Tuple' and state[2] == None: return 0
            policy = max((self.getQ(state, action), action) for action in self.actions(state))[1]
            return self.getQ(state, policy)

        next_state = newState
        if type(state) == 'Tuple' and state[2] == None: 
            next_state = None  
        Q = self.getQ(state, action) 
        eta = self.getStepSize()
        err = Q - reward - self.discount* get_v(next_state)
        for f, i in self.featureExtractor(state, action):         
            self.weights[f] = self.weights[f] - eta* err* i
        return

        raise Exception("Not implemented yet")
        # END_YOUR_CODE


# Return a singleton list containing indicator feature for the (state, action)
# pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]


############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs (see
# identityFeatureExtractor()).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card and the action (1 feature).
#       Example: if the deck is (3, 4, 0 , 2), then your indicator on the presence of each card is (1,1,0,1)
#       Only add this feature if the deck != None
# - indicator on the number of cards for each card type and the action (len(counts) features).  Only add these features if the deck != None
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state
    # BEGIN_YOUR_CODE (around 10 lines of code expected)
    pairs = []
    pairs.append(((total, action) , 1))
    temp = []
    if counts != None:
        for i in range(len(counts)):
            pairs.append(((str(i),counts[i],action), 1))     
            if counts[i]>0:
                temp.append(1)
            if counts[i]==0:
                temp.append(0)
        presence = (tuple(temp), action)
        pairs.append((presence,1))  


    return pairs

    raise Exception("Not implemented yet")
    # END_YOUR_CODE

###########################################################
# Problem 4b: convergence of Q-learning

# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

vi = ValueIteration()
vi.solve(largeMDP)

ql = QLearningAlgorithm(largeMDP.actions, 1, blackjackFeatureExtractor, explorationProb=0.2)
util.simulate(largeMDP, ql, 30000, 1000, False, False)
ifPrint = False
c = 0.0
for state in largeMDP.states:
    QLpi=max((ql.getQ(state, action), action) for action in largeMDP.actions(state))[1] 
    if vi.pi[state] != QLpi:
        c += 1
        ifPrint = True
        print state, 'VI: ',vi.pi[state],'vs  ', 'QL: ', QLpi 
print c / len(largeMDP.states)
if not ifPrint: 
    print 'All policies are same!'
############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)
