# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.99
    answerNoise = 0.01
    return answerDiscount, answerNoise

# the nearest exit near the abyss
def question3a():
    answerDiscount = 0.3
    answerNoise = 0.01
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

# nearest exit far away from the abyss
def question3b():
    answerDiscount = 0.3
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

# distant exit near the abyss
def question3c():
    answerDiscount = 0.9
    answerNoise = 0.01
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

# distant exit far away from the abyss 
def question3d():
    answerDiscount = 0.9
    answerNoise = 0.1
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

# avoid exits
def question3e():
    answerDiscount = 0.9
    answerNoise = 0.01
    answerLivingReward = 10.0
    return answerDiscount, answerNoise, answerLivingReward

def question6():
    answerEpsilon = None
    answerLearningRate = None
    # return answerEpsilon, answerLearningRate
    return 'NOT POSSIBLE'

if __name__ == '__main__':
    print 'Answers to analysis questions:'
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print '  Question %s:\t%s' % (q, str(response))
