# Social learning and SCBB in learning by doing process Ver 1.0

# Importing modules
# from IPython import get_ipython
# get_ipython().magic('reset -sf')
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime

STARTINGTIME = datetime.datetime.now().replace(microsecond=0)

#####################################################################################################
# SET SIMULATION PARAMETERS HERE
T = 1000  # number of periods to simulate the model
sampleSize = 1000  # sample size

socialLearning = 1  # if 0, then agents do not from learn from experience of the other.
                    # if 1, then agents do observational learning.
                    # if 2, then agents do belief sharing.

choiceRule = 0  # if 0, then agents do greedy search.
                # if 1, then agents follow soft-max rule with fixed tau.
                # if 2, then agents follow soft-max rule with endogeneous tau.
tau = 0.05   # Soft-max temperature for fixed tau case
updateRule = 0  # if 0, then agents follow a Bayesian updating.
                # if 1, then agents follow ERWA.
phi = 0.1   # degree of recency for EWRA rule

if socialLearning == 0:
    print("Agents are isolated from each other (no social learning)")
elif socialLearning == 1:
    print("Agents do observational learning")
elif socialLearning == 2:
    print("Agents do belief sharing")

if choiceRule == 0:
    if updateRule == 0:
        print("Agents follow greedy search and bayesian updating")
    elif updateRule == 1:
        print("Agents follow greedy search and ERWA")
elif choiceRule == 1:
    if updateRule == 0:
        print("Agents follow soft-max with fixed tau and bayesian updating")
    elif updateRule == 1:
        print("Agents follow soft-max with fixed tau and ERWA")
elif choiceRule == 2:
    if updateRule == 0:
        print("Agents follow soft-max with endogenous tau and bayesian updating")
    elif updateRule == 1:
        print("Agents follow soft-max with endogenous and ERWA")


# Task environment
M = 50  # number of alternatives
noise = 0   # if noise = 0, then there is no noise in feedback. If noise > 0, then feedback is noisy.
reality = np.zeros(M)

######################################################################################################

# DEFINING RESULTS VECTORS
avg_agent_perf1 = np.zeros((T, sampleSize))
avg_agent_perf2 = np.zeros((T, sampleSize))
correct_choice1 = np.zeros((T, sampleSize))
correct_choice2 = np.zeros((T, sampleSize))
both_correct = np.zeros((T, sampleSize))
biased_beliefs1 = np.zeros((T, sampleSize))
biased_beliefs2 = np.zeros((T, sampleSize))
switching_behavior1 = np.zeros((T, sampleSize))
switching_behavior2 = np.zeros((T, sampleSize))
convergence = np.zeros((T, sampleSize))

# Defining functions
def genEnvironment(M):  # Generate task environment
    r = np.random.rand(M)
    return r

def genPriors(M): # Generate random priors
    r = np.random.rand(M)
    return r


def getBest(reality):  # max action selection
    best = reality.argmax(axis=0)
    return best

def dissimilarity(l1, l2):
    diff = l1 - l2
    c = float(np.sum(np.absolute(diff), axis=None) / M)
    return c

def hardmax(attraction, t, M):  # max action selection
    maxcheck = attraction.max(axis=0)
    mincheck = attraction.min(axis=0)
    if maxcheck == mincheck:
        choice = random.randint(0, M - 1)
    else:
        choice = attraction.argmax(axis=0)
    return choice


def softmax(attraction, t, M, tau):  # softmax action selection
    prob = np.zeros((1, M))
    denom = 0
    i = 0
    while i < M:
        denom = denom + math.exp((attraction[i]) / tau)
        i = i + 1
    roulette = random.random()
    i = 0
    p = 0
    while i < M:
        prob[0][i] = math.exp(attraction[i] / tau) / denom
        p = p + prob[0][i]
        if p > roulette:
            choice = i
            return choice
            break  # stops computing probability of action selection as soon as cumulative probability exceeds roulette
        i = i + 1


# Time varying model objects
attraction1 = np.zeros(M)
attraction2 = np.zeros(M)
attractionlag1 = np.zeros(M)
attractionlag2 = np.zeros(M)
# To keep track of c(c)ount of # times action selected for bayesian updating
count1 = np.zeros(M)
count2 = np.zeros(M)

# SIMULTAION IS RUN HERE

for a in range(sampleSize):
    reality = genEnvironment(M)
    attraction1 = genPriors(M)
    attraction2 = genPriors(M)
    count1 = np.ones(M)
    count2 = np.ones(M)
    bestchoice = getBest(reality)
    pchoice1 = -1
    pchoice2 = -1

    for t in range(T):
        if choiceRule == 0:
            choice1 = hardmax(attraction1, t, M)
        elif choiceRule == 1:
            choice1 = softmax(attraction1, t, M, tau)
        elif choiceRule == 2:
            if t < 2:
                tau = 1
            else:
                tau = 1 - avg_agent_perf1[t-1, a]

            if tau <= 0.01:
                choice1 = hardmax(attraction1, t, M)
            else:
                choice1 = softmax(attraction1, t, M, tau)

        if choiceRule == 0:
            choice2 = hardmax(attraction2, t, M)
        elif choiceRule == 1:
            choice2 = softmax(attraction2, t, M, tau)
        elif choiceRule == 2:
            if t < 2:
                tau = 1
            else:
                tau = 1 - avg_agent_perf2[t-1, a]

            if tau <= 0.01:
                choice2 = hardmax(attraction2, t, M)
            else:
                choice2 = softmax(attraction2, t, M, tau)

        payoff1 = reality[choice1] + noise * (0.5 - random.random())
        payoff2 = reality[choice2] + noise * (0.5 - random.random())

        avg_agent_perf1[t][a] = payoff1
        avg_agent_perf2[t][a] = payoff2

        if choice1 == bestchoice:
            correct_choice1[t][a] = 1
        if choice2 == bestchoice:
            correct_choice2[t][a] = 1
        if (choice1 == bestchoice) & (choice2 == bestchoice):
            both_correct[t][a] = 1
        biased_beliefs1[t][a] = dissimilarity(reality, attraction1)
        biased_beliefs2[t][a] = dissimilarity(reality, attraction2)
        if choice1 != pchoice1:
            switching_behavior1[t][a] = 1
        if choice2 != pchoice2:
            switching_behavior2[t][a] = 1
        if choice1 == choice2:
            convergence[t][a] = 1

        pchoice1 = choice1
        pchoice2 = choice2

        if socialLearning == 0:
           count1[choice1] += 1
           count2[choice2] += 1
           if updateRule == 0:
              attraction1[choice1] = (count1[choice1] - 1) / count1[choice1] * attraction1[choice1] + 1 / count1[choice1] * payoff1
              attraction2[choice2] = (count2[choice2] - 1) / count2[choice2] * attraction2[choice2] + 1 / count2[choice2] * payoff2
           elif updateRule == 1:
                attraction1[choice1] = (1 - phi) * attraction1[choice1] + phi*payoff1
                attraction2[choice2] = (1 - phi) * attraction2[choice2] + phi * payoff2
        elif socialLearning == 1:
            count1[choice1] += 1
            count2[choice2] += 1
            if updateRule == 0:
                attraction1[choice1] = (count1[choice1] - 1) / count1[choice1] * attraction1[choice1] + 1 / count1[choice1] * payoff1
                attraction2[choice2] = (count2[choice2] - 1) / count2[choice2] * attraction2[choice2] + 1 / count2[choice2] * payoff2
                count2[choice1] += 1
                count1[choice2] += 1
                attraction2[choice1] = (count2[choice1] - 1) / count2[choice1] * attraction2[choice1] + 1 / count2[choice1] * payoff1
                attraction1[choice2] = (count1[choice2] - 1) / count1[choice2] * attraction1[choice2] + 1 / count1[choice2] * payoff2
            elif updateRule == 1:
                attraction1[choice1] = (1 - phi) * attraction1[choice1] + phi * payoff1
                attraction2[choice2] = (1 - phi) * attraction2[choice2] + phi * payoff2
                count1[choice1] += 1
                count2[choice2] += 1
                attraction2[choice1] = (1 - phi) * attraction2[choice1] + phi * payoff2
                attraction1[choice2] = (1 - phi) * attraction1[choice2] + phi * payoff1
        elif socialLearning == 2:
            count1[choice1] += 1
            count2[choice2] += 1
            if updateRule == 0:
                attraction1[choice1] = (count1[choice1] - 1) / count1[choice1] * attraction1[choice1] + 1 / count1[choice1] * payoff1
                attraction2[choice2] = (count2[choice2] - 1) / count2[choice2] * attraction2[choice2] + 1 / count2[choice2] * payoff2
            elif updateRule == 1:
                attraction1[choice1] = (1 - phi) * attraction1[choice1] + phi * payoff1
                attraction2[choice2] = (1 - phi) * attraction2[choice2] + phi * payoff2
            attractionlag1 = np.copy(attraction1)
            attractionlag2 = np.copy(attraction2)
            attraction1 = np.add(attractionlag1, attractionlag2)/2
            attraction2 = np.add(attractionlag1, attractionlag2)/2

result_org = np.zeros((T, 11))

for t in range(T):  # Compiling final output
    result_org[t, 0] = t + 1
    result_org[t, 1] = float(np.sum(avg_agent_perf1[t, :])) / sampleSize
    result_org[t, 2] = float(np.sum(avg_agent_perf2[t, :])) / sampleSize
    result_org[t, 3] = float(np.sum(correct_choice1[t, :])) / sampleSize
    result_org[t, 4] = float(np.sum(correct_choice2[t, :])) / sampleSize
    result_org[t, 5] = float(np.sum(both_correct[t, :])) / sampleSize
    result_org[t, 6] = float(np.sum(biased_beliefs1[t, :])) / sampleSize
    result_org[t, 7] = float(np.sum(biased_beliefs2[t, :])) / sampleSize
    result_org[t, 8] = float(np.sum(switching_behavior1[t, :])) / sampleSize
    result_org[t, 9] = float(np.sum(switching_behavior2[t, :])) / sampleSize
    result_org[t, 10] = float(np.sum(convergence[t, :])) / sampleSize

# WRITING RESULTS TO CSV FILE
filename = ("Social learning" + " (mode = " + str(socialLearning) +"; choice rule = "+ str(choiceRule)+"; updating rule =  "+ str(updateRule)+ ').csv')
with open(filename, 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(
        ['Period', 'Perf1', 'Perf2', 'Prop(the optimal choice, agent 1)', 'Prop(the optimal choice, agent 2)', 'Prop(the optimal choice, both agents)', 'Dist(beliefs and reality, agent1)','Dist(beliefs and reality, agent2)', 'Switching1', 'Switching2', 'Convergence of action'])
    for values in result_org:
        thewriter.writerow(values)
    f.close()

##PRINTING END RUN RESULTS
print("Final performance " + str((result_org[T - 1, 1]+result_org[T - 1, 1])/2))
print("Final proportion of the optimal choice " + str(result_org[T - 1, 5]))

#
##GRAPHICAL OUTPUT
plt.style.use('ggplot')  # Setting the plotting style
fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
ax1 = plt.subplot2grid((6, 5), (4, 1), colspan=3)

ax1.plot(result_org[:, 0], (result_org[:, 1]+result_org[:, 1])/2, color='black', linestyle='--', label='Average Performance')
ax1.plot(result_org[:, 0], result_org[:, 5], color='blue', linestyle='--', label='Probability that both agents choose the optimal action')
ax1.plot(result_org[:, 0], result_org[:, 10], color='black', linestyle='--', label='Convergence in action')

ax1.set_xlabel('t', fontsize=12)
ax1.legend(bbox_to_anchor=(0., -0.8, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., fontsize=9)

ENDINGTIME = datetime.datetime.now().replace(microsecond=0)
TIMEDIFFERENCE = ENDINGTIME - STARTINGTIME
# print 'Computation time:', TIMEDIFFERENCE
