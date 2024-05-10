# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util, math

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features
    
def closestCapsule(pos, capsules, walls):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        if (pos_x, pos_y) in capsules:
            return dist
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    return None

def closestGhost(pacmanPos, ghosts):
    if not ghosts:
        return -1

    minDist = float("+inf")
    for ghost in ghosts:
        dist = util.manhattanDistance(pacmanPos, ghost.getPosition())
        if minDist > dist:
            minDist = dist
        
    return minDist
    
class MyExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        capsules = state.getCapsules()
        scaredGhosts = []
        activeGhosts = []
        # score = state.getScore()

        for ghost in state.getGhostStates():
            if ghost.scaredTimer > 0:
                scaredGhosts.append(ghost)
            else:
                activeGhosts.append(ghost)

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        # next_next_x, next_next_y = int(next_x + dx), int(next_y + dy)

        # features["score"] = 1 / (1 + math.exp(-score))

        features["active-ghosts"] = len(activeGhosts)
        features["scared-ghosts"] = len(scaredGhosts)

        # closest active ghost
        features["closest-active-ghost"] = closestGhost(state.getPacmanPosition(), activeGhosts)

        # closest scared ghost
        features["closest-scared-ghost"] = closestGhost(state.getPacmanPosition(), scaredGhosts)

        # scare is over soon
        features["scare-is-over-soon"] = 0.0 if len(scaredGhosts) > 0 and scaredGhosts[0].scaredTimer > 1 else 1.0  

        # capsules left
        features["capsules-left"] = len(capsules) * 5.0
        
        # count the number of active ghosts 1-step and 2-step away
        features["#-of-active-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) for g in activeGhosts)
        # features["#-of-active-ghosts-2-step-away"] = sum((next_next_x, next_next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) for g in activeGhosts)
        
        # count the number of scared ghosts 1-step and 2-step away
        features["#-of-scared-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) for g in scaredGhosts)
        # features["#-of-scared-ghosts-2-step-away"] = sum((next_next_x, next_next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) for g in scaredGhosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-active-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 10.0

        # if not features["scare-is-over-soon"]:
            # features["eat-ghost"] = 5.0

        distFood = closestFood((next_x, next_y), food, walls)
        if distFood is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(distFood) / (walls.width * walls.height)
        
        distCapsule = closestCapsule((next_x, next_y), capsules, walls)
        if distCapsule is not None:
            features["closest-capsule"] = float(distCapsule) / (walls.width * walls.height)

        features.divideAll(10.0)
        return features
