from functools import total_ordering
import pickle
import numpy as np
from numpy.lib.function_base import place
import matplotlib.pyplot as plt
# How Many ticks are there per hour?
ticksPerHour = 1
# TAKEN from infekta
DEAD = 0
IMMUNE = 1
RECOVERED =2
SUCEPTIBLE  =3
EXPOSED = 4
ASYMPTOTIC = 5
SERIOUSLY = 6
CRITICAL = 7

InfectParams = {'alpha' : [0.000] + [0.37]*5, 
              'beta'  : [0.000] + [0.000, 0.800, 0.800, 0.200, 0.200], 
              'gamma' : [0.000] + [0.000, 0.008, 0.058, 0.195, 0.350],
              'theta' : [0.000] + [0.050, 0.050, 0.050, 0.198, 0.575], 
              'phi'   : [0.000] + [0.400, 0.400, 0.500, 0.500, 0.500], 
              'omega' : [0.000] + [0.900]*5, 
              #'omega' : [0.000] + [0.01]*5, 
              'T_E'   : ticksPerHour*24*np.round(np.random.gamma(5.1, 0.86, 6), 0), 
              'T_Ia'  : ticksPerHour*24*np.array([0] + [3, 14, 14, 5, 5]), 
              'T_Is'  : ticksPerHour*24*np.round(np.random.triangular(7, 8, 9, 6), 0), 
              'T_Ic'  : ticksPerHour*24*np.round(np.random.triangular(5, 7, 12, 6), 0),   
              'T_R'   : ticksPerHour*24*np.round(np.random.uniform(80, 100, 6), 0)}
"""
TODO(henrik): Healthstate of critical and seriously should be incooperated in to the iterany
Iterany contains all public transportation stations if necessary
"""
class Iterany:
    def __init__(self,homeID, placesID, probabilities) -> None:
        self.homeID = homeID
        self.currentPos = 0
        self.placesID = placesID
        self.probabilities = probabilities
    def nextPosition(self,time):
        self.currentPos = np.random.choice(len(self.placesID),1,p=self.probabilities[self.currentPos])[0]
        return self.placesID[self.currentPos]
    def placeID(self):
        """
        Returns the place ID of the current Pos
        """
        return self.placesID[self.currentPos]
class Place:
    def __init__(self,placeId,nodeID) -> None:
        """
        placeID is from 0... number of places, node ID for OSM
        """
        self.ID = placeId
        self.nodeID=nodeID
        self.agentsInState = np.array([0]*8)
    def totalAlive(self):
        return np.sum(self.agentsInState[1:])
    def totalInfected(self):
        return np.sum(self.agentsInState[5:8])
    def dead(self):
        return self.agentsState[0]
    def immune(self):
        return self.agentsState[1]
    def recovered(self):
        return self.agentsState[2]
    def suceptible(self):
        return self.agentsInState[3]
    def exposed(self):
        return self.agentsInState[4]
    def asymptotic(self):
        return self.agentsInState[5]
    def seriously(self):
        return self.agentsInState[6]
    def critical(self):
        return self.agentsInState[7]
class Agent:
    def __init__(self,agentID,homeID,age,state,iterany, currTick) -> None:
        """
        state is dead(0),immune(1), recovered(2), susceptible(3), exposed(4), Asymptotic(5), serisously(6), critical(7) infected
        age is the age group defined
        """
        self.agentID = agentID
        self.state = state
        self.homeID = homeID
        self.age = age
        self.iterany = iterany
        self.inStateSince = currTick
        self.nextStateDuringTick =-1
    def makeSick(self,places,tick):
        if self.state == SUCEPTIBLE:
            self.state = ASYMPTOTIC
            self.currentPlace(places).agentsInState[SUCEPTIBLE] -= 1
            self.currentPlace(places).agentsInState[ASYMPTOTIC] += 1
            self.nextStateDuringTick =  tick + InfectParams['T_Ia'][self.age]
    def infect(self,places,tick):
        """
        Plays the infect for the number of Contacts in current tick
        """
            # Only the suseptible ones are infectable
        if self.state == SUCEPTIBLE:
            params = InfectParams
            numberOfContacts = places[self.iterany.placeID()].totalInfected()
            if numberOfContacts > 0:
                # it does not matter yet, how many are infected TODO(henrik): number of infected
                probability = np.random.uniform()
                if probability < params['alpha'][self.age]:
                    # print infected 
                    places[self.iterany.placeID()].agentsInState[self.state] -= 1
                    places[self.iterany.placeID()].agentsInState[EXPOSED] += 1
                    self.state = EXPOSED
                    self.inStateSince = tick
                    self.nextStateDuringTick = tick + params['T_E'][self.age]

    def incubate(self,places,tick):
        params = InfectParams
        if self.nextStateDuringTick != -1:
            if self.nextStateDuringTick == tick:
                prob = np.random.uniform()
                if self.state == RECOVERED:
                    if prob < params['omega'][self.age]:
                        # get able to be sick again
                        places[self.iterany.placeID()].agentsInState[self.state] -= 1
                        places[self.iterany.placeID()].agentsInState[SUCEPTIBLE] += 1
                        self.state = SUCEPTIBLE
                        self.nextStateDuringTick = -1
                    else:
                        # get IMMUNE for live
                        places[self.iterany.placeID()].agentsInState[self.state] -= 1
                        places[self.iterany.placeID()].agentsInState[IMMUNE] += 1
                        self.state = IMMUNE
                        self.nextStateDuringTick = -1
                if self.state == CRITICAL:
                    if prob < params['phi'][self.age]:
                        # get serious
                        places[self.iterany.placeID()].agentsInState[self.state] -= 1
                        places[self.iterany.placeID()].agentsInState[DEAD] += 1
                        self.state = DEAD
                        self.nextStateDuringTick = -1
                    else:
                        # get recovered
                        places[self.iterany.placeID()].agentsInState[self.state] -= 1
                        places[self.iterany.placeID()].agentsInState[RECOVERED] += 1
                        self.state = RECOVERED
                        self.nextStateDuringTick = tick + params['T_R'][self.age]
                if self.state == SERIOUSLY:
                    if prob < params['theta'][self.age]:
                        # get critical
                        places[self.iterany.placeID()].agentsInState[self.state] -= 1
                        places[self.iterany.placeID()].agentsInState[CRITICAL] += 1
                        self.state = CRITICAL
                        self.nextStateDuringTick = tick + params['T_Ic'][self.age]
                    else:
                        # get recovered
                        places[self.iterany.placeID()].agentsInState[self.state] -= 1
                        places[self.iterany.placeID()].agentsInState[RECOVERED] += 1
                        self.state = RECOVERED
                        self.nextStateDuringTick = tick + params['T_R'][self.age]
                if self.state == ASYMPTOTIC:
                    if prob < params['gamma'][self.age]:
                        # get serious
                        places[self.iterany.placeID()].agentsInState[self.state] -= 1
                        places[self.iterany.placeID()].agentsInState[SERIOUSLY] += 1
                        self.state = SERIOUSLY
                        self.nextStateDuringTick = tick + params['T_Is'][self.age]
                    else:
                        # get recovered
                        places[self.iterany.placeID()].agentsInState[self.state] -= 1
                        places[self.iterany.placeID()].agentsInState[RECOVERED] += 1
                        self.state = RECOVERED
                        self.nextStateDuringTick = tick + params['T_R'][self.age]
                if self.state ==EXPOSED:
                    if prob < params['beta'][self.age]:
                        # get asymptomatic
                        places[self.iterany.placeID()].agentsInState[self.state] -= 1
                        places[self.iterany.placeID()].agentsInState[ASYMPTOTIC] += 1
                        self.state = ASYMPTOTIC
                        self.nextStateDuringTick = tick + params['T_Ia'][self.age]
                    else:
                        # get serious
                        places[self.iterany.placeID()].agentsInState[self.state] -= 1
                        places[self.iterany.placeID()].agentsInState[SERIOUSLY] += 1
                        self.state = SERIOUSLY
                        self.nextStateDuringTick = tick + params['T_Is'][self.age]
    def currentPlace(self,places):
        return places[self.iterany.placeID()]
    def travel(self,places ,tick):
        places[self.iterany.placeID()].agentsInState[self.state] -= 1
        places[self.iterany.nextPosition(tick)].agentsInState[self.state] += 1
class Simulator:
    def __init__(self,agents, places) -> None:
        self.agents = agents
        self.places = places
        self.tick = 0
    def simulateTick(self):
        """
        Simulates a tick consisting out of:
            - infecting all agents
            - incubating all agents
            - travelling
        """
        self.tick += 1
        for a in self.agents:
            # infect
            a.infect(self.places,self.tick)
        for a in self.agents:
            # incubate
            a.incubate(self.places,self.tick)
        for a in self.agents:
            # travel
            a.travel(self.places,self.tick)
    def total(self):
        return np.sum([p.agentsInState for p in self.places],axis=0)
    def totalExposed(self):
        return np.sum([p.agentsInState[EXPOSED] for p in self.places])
    def totalAlive(self):
        return np.sum([p.totalAlive() for p in self.places])
    def totalInfected(self):
        return np.sum([p.totalInfected() for p in self.places])
if __name__ == '__main__':
    agents_filehandler = open("data/agents.obj","rb")
    places_filehandler = open("data/places.obj","rb")
    agents = pickle.load(agents_filehandler)
    places = pickle.load(places_filehandler)
    agents[0].makeSick(places,0)
    sim = Simulator(agents,places)
    points = []
    for i in range(24*120):
        sim.simulateTick()
        points += [sim.total()]
        if i % (24*10):
            agents[np.random.randint(0,len(agents)-1)].makeSick(places,i)
        if i % 12 == 0:
            print(i/24)
            print(sim.total())
            print(sim.totalAlive()," > ",sim.totalInfected(), " (",sim.totalExposed(),")")
    lines = plt.plot(range(len(points)),points)
    plt.legend(lines,["DEAD","IMMUNE","RECOVERED","SUSCEPTIBLE","EXPOSED","ASYMPTOTIC","SERIOUSLY","CRITICAL"])
    plt.show()