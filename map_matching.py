import numpy as np
import pickle
import pandas as pd
import os
from collections import defaultdict, OrderedDict
from mpmath import *
from math import cos, asin, sqrt
from sympy.solvers import solve
from sympy import Symbol
import csv
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

mp.dps = 64


def loadLinkID():
    """
    Load linkPVID information.
    Return:  IDs:   A list of linkID in turn.
    """
    if os.path.exists('linkID.pk'):
        IDs = pickle.load(open('linkID.pk', 'rb'))
        print("loading linkID...")
        return IDs
    
    idInfo = pd.read_csv("./probe_data_map_matching/Partition6467LinkData.csv", sep=",", header=None, usecols=[0], dtype=str)
    idInfo = idInfo.values
    
    IDs = []
    for i in range(idInfo.shape[0]):
        IDs.append(idInfo[i][0])
        
    pickle.dump(IDs, open('linkID.pk','wb'))
    print("loading linkID...")
    
    return IDs



def loadLinkLatLon():
    """
    Load the latitude and longitude information of linknodes (ref/nref and shape points).
    Return:  Lats:   A list of [linkID, node_lat].
             Lons:   A list of [linkID, node_lon].
    """
    if os.path.exists('linkLat.pk') and os.path.exists('linkLon.pk'):
        Lats = pickle.load(open('linkLat.pk', 'rb'))
        Lons = pickle.load(open('linkLon.pk', 'rb'))
        print("loading linkLat, linkLon...")
        return Lats, Lons
    
    shapeInfo = pd.read_csv("./probe_data_map_matching/Partition6467LinkData.csv", sep=",", header=None, usecols=[14], dtype=str)
    shapeInfo = shapeInfo.values
    IDs = loadLinkID()

    Lats = []  
    Lons = []  

    for i in range(shapeInfo.shape[0]):
        linknodes = shapeInfo[i][0].split('|')  # linknodes is list of strings of lat and lon of a node
        for nd in linknodes:
            loc = nd.split('/')            
            lat = np.float64(loc[0])  
            lon = np.float64(loc[1])
            nd_lat = []
            nd_lon = []
            nd_lat.append(IDs[i])
            nd_lat.append(lat)
            nd_lon.append(IDs[i])
            nd_lon.append(lon)
            Lats.append(nd_lat)
            Lons.append(nd_lon)

    pickle.dump(Lats, open('linkLat.pk','wb'))
    pickle.dump(Lons, open('linkLon.pk','wb'))
    print("loading linkLat, linkLon...")
    
    return Lats, Lons



def loadLinkDirec():
    """
    Load the directionOfTravel information.
    Return:  Direcs:   A list of directionOfTravel in turn.
    """
    if os.path.exists('linkDirec.pk'):
        Direcs = pickle.load(open('linkDirec.pk', 'rb'))
        print("loading linkDirec...")
        return Direcs
    
    direcInfo = pd.read_csv("./probe_data_map_matching/Partition6467LinkData.csv", sep=",", header=None, usecols=[5], dtype=str)
    direcInfo = direcInfo.values
    
    Direcs = []
    for i in range(direcInfo.shape[0]):
        Direcs.append(direcInfo[i][0])

    pickle.dump(Direcs, open('linkDirec.pk','wb'))
    print("loading linkDirec...")
    
    return Direcs



def loadLinkSpeed():
    """
    Load the fromRefSpeedLimit and toRefSpeedLimit information of links.
    Return:  Speeds:   A list of [fref_speed, tref_speed] in turn.
    """
    if os.path.exists('linkSpeed.pk'):
        Speeds = pickle.load(open('linkSpeed.pk', 'rb'))
        return Speeds
    
    fSpeedInfo = pd.read_csv("./probe_data_map_matching/Partition6467LinkData.csv", sep=",", header=None, usecols=[7], dtype=int)
    tSpeedInfo = pd.read_csv("./probe_data_map_matching/Partition6467LinkData.csv", sep=",", header=None, usecols=[8], dtype=int)
    fSpeedInfo = fSpeedInfo.values
    tSpeedInfo = tSpeedInfo.values
    
    Speeds = []
    for i in range(fSpeedInfo.shape[0]):
        each = []
        each.append(fSpeedInfo[i][0])
        each.append(tSpeedInfo[i][0])
        Speeds.append(each)
        
    pickle.dump(Speeds, open('linkSpeed.pk','wb'))
    
    return Speeds



def loadLinkSlope():
    """
    Load the slope information of linknodes (ref/nref and shape points), return the average slope value for each link that is available of slope data.
    Return:  Slope:   A list of [linkID, avg_slope].
    """

    if os.path.exists('linkSlope.pk'):
        Slopes = pickle.load(open('linkSlope.pk', 'rb'))
        return Slopes

    slopeInfo = pd.read_csv("./probe_data_map_matching/Partition6467LinkData.csv", sep=",", header=None, usecols=[16], dtype=str)
    IDs = loadLinkID()

    Slopes = []
    for i in range(slopeInfo.shape[0]):
        if pd.isna(slopeInfo.iloc[i,0]):
            continue
        each = []
        each.append(IDs[i])
        l_slope = []
        linknodes = slopeInfo.iloc[i,0].split('|') 
        for nd in linknodes:
            nd_slope = np.float64(nd.split('/')[1])
            l_slope.append(nd_slope)
        each.append(np.mean(l_slope))
        Slopes.append(each)
        
    pickle.dump(Slopes, open('linkSlope.pk','wb'))
        
    return Slopes




def loadSmplID():
    """
    Load the probeID information for each sample.
    Return:  smplIDs:   A list of [probeID, j], unique for each sample, where j is the index by timestamp within one probe.
    """
    if os.path.exists('smplID.pk'):
        smplIDs = pickle.load(open('smplID.pk', 'rb'))
        return smplIDs
    
    probeID = pd.read_csv("./probe_data_map_matching/Partition6467ProbePoints.csv", sep=",", header=None, usecols=[0], dtype=str)
    probeID = probeID.values
    
    smplIDs = []
    j = 0
    last_id = 0
    for i in range(probeID.shape[0]):
        each = []
        each.append(probeID[i][0])
        
        if probeID[i][0] == last_id:
            j += 1
        else:
            j = 0
        each.append(j)
        smplIDs.append(each)
        last_id = probeID[i][0]
    
    pickle.dump(smplIDs, open('smplID.pk','wb'))
    
    return smplIDs
     


def loadSmplTime():
    """
    Load the dateTime information of samples.
    Return:  Time:   A list of dateTime in turn.
    """
    if os.path.exists('smplTime.pk'):
        Time = pickle.load(open('smplTime.pk', 'rb'))
        return Time
    
    timeInfo = pd.read_csv("./probe_data_map_matching/Partition6467ProbePoints.csv", sep=",", header=None, usecols=[1])
    timeInfo = timeInfo.values
    
    Time = []
    for i in range(timeInfo.shape[0]):
        Time.append(timeInfo[i][0])
        
    pickle.dump(Time, open('smplTime.pk','wb'))
    
    return Time



def loadSmplSrcCode():
    """
    Load the sourceCode information of samples.
    Return:  SrcCodes:   A list of sourceCode in turn.
    """
    if os.path.exists('smplSrcCode.pk'):
        SrcCodes = pickle.load(open('smplSrcCode.pk', 'rb'))
        return SrcCodes
    
    srcCodeInfo = pd.read_csv("./probe_data_map_matching/Partition6467ProbePoints.csv", sep=",", header=None, usecols=[2])
    srcCodeInfo = srcCodeInfo.values
    
    SrcCodes = []
    for i in range(srcCodeInfo.shape[0]):
        SrcCodes.append(srcCodeInfo[i][0])
        
    pickle.dump(SrcCodes, open('smplSrcCode.pk','wb'))
    
    return SrcCodes



def loadSmplLatLon():
    """
    Load the latitude and longitude information of samples.
    Return:  Lats:   A list of [smplID, smpl_lat], where smplID is [probeID, j].
             Lons:   A list of [smplID, smpl_lon], where smplID is [probeID, j].
    """
    if os.path.exists('smplLat.pk') and os.path.exists('smplLon.pk'):
        Lats = pickle.load(open('smplLat.pk', 'rb'))
        Lons = pickle.load(open('smplLon.pk', 'rb'))
        return Lats, Lons

    latInfo = pd.read_csv("./probe_data_map_matching/Partition6467ProbePoints.csv", sep=",", header=None, usecols=[3], dtype=np.float64)
    latInfo = latInfo.values
    lonInfo = pd.read_csv("./probe_data_map_matching/Partition6467ProbePoints.csv", sep=",", header=None, usecols=[4], dtype=np.float64)
    lonInfo = lonInfo.values

    IDs = loadSmplID()

    Lats = []  
    Lons = []  

    for i in range(latInfo.shape[0]): 
        smpl_lat = []
        smpl_lon = []
        smpl_lat.append(IDs[i])
        smpl_lat.append(latInfo[i][0])
        smpl_lon.append(IDs[i])
        smpl_lon.append(lonInfo[i][0])
        Lats.append(smpl_lat)
        Lons.append(smpl_lon)

    pickle.dump(Lats, open('smplLat.pk','wb'))
    pickle.dump(Lons, open('smplLon.pk','wb'))

    return Lats, Lons



def loadSmplAlt():
    """
    Load the altitude information of samples.
    Return:  Alts:   A list of altitude in turn.
    """
    if os.path.exists('smplAlt.pk'):
        Alts = pickle.load(open('smplAlt.pk', 'rb'))
        return Alts
    
    altInfo = pd.read_csv("./probe_data_map_matching/Partition6467ProbePoints.csv", sep=",", header=None, usecols=[5], dtype=int)
    altInfo = altInfo.values
    
    Alts = []
    for i in range(altInfo.shape[0]):
        Alts.append(altInfo[i][0])
        
    pickle.dump(Alts, open('smplAlt.pk','wb'))
    
    return Alts



def loadSmplSpeed():
    """
    Load the speed information of samples.
    Return:  Speeds:   A list of speed in turn.
    """
    if os.path.exists('smplSpeed.pk'):
        Speeds = pickle.load(open('smplSpeed.pk', 'rb'))
        return Speeds
    
    speedInfo = pd.read_csv("./probe_data_map_matching/Partition6467ProbePoints.csv", sep=",", header=None, usecols=[6], dtype=int)
    speedInfo = speedInfo.values
    
    Speeds = []
    for i in range(speedInfo.shape[0]):
        Speeds.append(speedInfo[i][0])
        
    pickle.dump(Speeds, open('smplSpeed.pk','wb'))
    
    return Speeds



def loadSmplHeading():
    """
    Load the heading information of samples.
    Return:  Headings:   A list of heading in turn.
    """
    if os.path.exists('smplHeading.pk'):
        Headings = pickle.load(open('smplHeading.pk', 'rb'))
        return Headings
    
    headingInfo = pd.read_csv("./probe_data_map_matching/Partition6467ProbePoints.csv", sep=",", header=None, usecols=[7], dtype=int)
    headingInfo = headingInfo.values
    
    Headings = []
    for i in range(headingInfo.shape[0]):
        Headings.append(headingInfo[i][0])
        
    pickle.dump(Headings, open('smplHeading.pk','wb'))
    
    return Headings




def calcDistance(lat1, lon1, lat2, lon2):
    """
    calculate Haversine distance(meter) given the latitudes and longitudes(degree) of two points
    """
    R = 6371*1000
    p = 0.017453292519943295     # Pi/180, for converting degree to radius
    a = 0.5 - cos((lat2 - lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1 - cos((lon2-lon1)*p))/2
    dist = 2*R * arcsin(sqrt(a))
    return np.float64(dist)



def calcPerpDist(lat1, lon1, lat2, lon2, lat3, lon3):
    """
    calculate the perpendicular distance from point P3 to the line connected by point P1 and P2.
    """
    sin = lambda x: np.sin(x)
    sq = lambda x: np.square(x)
    
    d12 = calcDistance(lat1, lon1, lat2, lon2)
    d13 = calcDistance(lat1, lon1, lat3, lon3)
    d23 = calcDistance(lat2, lon2, lat3, lon3)
    ag1 = np.arccos((sq(d13)+ sq(d12)- sq(d23)) /(2*d13*d12))
    perp_dist = d13 * sin(ag1)

    return np.float64(perp_dist)
    


def segmentL1L2(linkLat, linkLon):
    """
    Represent the segment between any two adjacent linknodes in a same link in turn.
    
    Input:   linkLat:   All linknodes' latitude information, a list of [linkID, node_lon].
             linkLon:   All linknodes' longitude information, a list of [linkID, node_lat].
    Return:  L1:        All start node's lat/lon information in a segment in turn, a list of [lat_1, lon_1].
             L2:        All end node's lat/lon information in a segment in turn, a list of [lat_2, lon_2]. 
                        Namely the element pair with same index in L1 and L2 forms a segment.
             L1_id:     A list of the linkID information of all nodes in L1. It has same size with L1.
                        It is used to correctly bridge the feature information of an identical link node among differnt modules.  
    """
    
    L_lat = np.stack(linkLat, axis=1)[1]
    L_lon = np.stack(linkLon, axis=1)[1]
    L = np.float64(np.stack([L_lat, L_lon], axis=1))   # L is the list of all linknodes. 
    L1 = L[1:]
    L2 = L[:-1]      # By dislocation, the element pair with same index in L1 and L2 represents a segment.
    
    # update linkID for consistency
    L1_id = np.stack(linkLat, axis=1)[0]      
    L2_id = L1_id[1:]
    L1_id = L1_id[:-1]

    # remove the segments which undesirably join the two different links.
    ind = np.where((L1_id == L2_id) & ~((L1[:,0] == L2[:,0]) & (L1[:,1] == L2[:,1])))
    L1 = L1[ind]
    L2 = L2[ind]
    L1_id = L1_id[ind]

    return L1, L2, L1_id



def genCandidate(smplLat, smplLon, L1, L2, L1_id):
    """
    For each sample given its lat/lon, determine a list of segment candidates whose lat/lon coordinates are within 30m GPS error range.
    If a sample's candidate list is empty, then the sample will not be considered any more as unsuccessfully map-matched. The Output is the remaining samples and their segment candidates.
    
    Input:  smplLat:   A list of [smplID, smpl_lat], where smplID is [probeID, j]. 
            smplLon:   A list of [smplID, smpl_lon], where smplID is [probeID, j].
            L1:        A list of [lat_1, lon_1], each element representing the start node of a segment.
            L2:        A list of [lat_2, lon_2], each element representing the end node of a segment.
            L1_id:     A list of linkID where each node in L1 lies.
    Output: Candi:     A list of [smpl_id,  [[linkid, idx],
                                                  ...,
                                             [linkid, idx]] ],  where idx is the index in L1 of this segment candidate.
    """
    if os.path.exists('candi.pk'):
        Candi = pickle.load(open('candi.pk', 'rb'))
        return Candi
    Candi = []

    # convert distance offset to lat/lon offset.
    offset = 0.000135    # degree offset = (15/(ER)*(180/pi)

    for i in range(len(smplLat)):
        s_id = smplLat[i][0]
        s_lat = smplLat[i][1]
        s_lon = smplLon[i][1]

        # calculate absolute range
        latMin = s_lat - offset
        latMax = s_lat + offset
        lonMin = s_lon - offset
        lonMax = s_lon + offset
        
        # only store the segments that are within the 30m range.
        ind = np.where(((latMin <= L1[:,0]) & (L1[:,0] <= latMax)) & ((lonMin <= L1[:,1]) & (L1[:,1] <= lonMax)))[0]
        if ind.shape[0] == 0:
            continue   
        each = []
        cand_info = np.stack((L1_id[ind], ind), axis=1)

        each.append(s_id)
        each.append(cand_info)
        Candi.append(each)
        
        pickle.dump(Candi, open('candi.pk','wb'))
        
    return Candi
 

def trvlDirecFilter(Candi, smplID, smplLat, smplLon, smplHeading, linkID, L1, L2, linkDirec):
    """
    Using directionOfTravel criteria,  filter out the impossible segment candidates for each sample, 
    only keep the segment candidates whose directionOfTravel is conformed with the sample's current travel direction.
    Input:   Candi:      A list of [smpl_id,  [[linkid, idx],
                                                  ...,
                                               [linkid, idx]] ],  where idx is the index in L1 of this segment candidate.
    
    Return:  Candi2:     The filtered candidates for the remaining samples.
                         A list of [smpl_id,  [[linkid, idx],
                                                  ...,
                                               [linkid, idx]] ],  where idx is the index in L1 of this segment candidate.
    """
    Candi2 = []
    for i in range(len(Candi)):
        s_id = Candi[i][0] 
        s_idx = smplID.index(s_id)
        s_lat = smplLat[s_idx][1]
        s_lon = smplLon[s_idx][1]
        s_head = smplHeading[s_idx]
        
        s_Candi2 = []
        for j in range(len(Candi[i][1])):    # Candi[i][1] is the segment candidate list for sample_i
            l1_idx = int(Candi[i][1][j][1])
            l1_lat = L1[l1_idx][0]
            l1_lon = L1[l1_idx][1]
            l2_lat = L2[l1_idx][0]
            l2_lon = L2[l1_idx][1]
            s_direc = calcSmplDirec(s_lat, s_lon, s_head, l1_lat, l1_lon, l2_lat, l2_lon)
            l_idx = linkID.index(Candi[i][1][j][0])
            l_direc = linkDirec[l_idx]     # link global index
            if ~((l_direc != 'B') & (s_direc != l_direc)):   # only keep the segment candidates whose directionOfTravel is conformed with the sample's current travel direction.    
                s_Candi2.append(Candi[i][1][j])
                
        if len(s_Candi2) > 0:    # only keep the samples whose candidate list is not empty after filtering.
            each = []
            each.append(s_id)
            each.append(s_Candi2)
            Candi2.append(each) 
        
    return Candi2
    
    
def passLengthFilter(Candi, smplID, smplSpeed, smplLat, smplLon, linkID, L1, L2, linkSpeed):
    """
    Filter out the impossible segment candidates for each sample by the trace-based heuristics: 
    the passing length by two consecutive samples should not be way larger than the distance between the two projection points on the correspondingly mapped segments, 
    meaning that the difference should not be larger than a threshold. Specifically, the valid candidate should suffice:
                    AvgSpeed(S1,S2) * TimeGap  <=  Dist(P1,P2) + thrsh,
    where thrsh = max(SpeedLimit) * TimeGap
    
    Input:   Candi:      A list of [smpl_id,  [[linkid, idx],
                                                  ...,
                                               [linkid, idx]] ],  where idx is the index in L1 of this segment candidate.
    
    Return:  Candi2:     The filtered candidates for the remaining samples.
                         A list of [smpl_id,  [[linkid, idx],
                                                  ...,
                                               [linkid, idx]] ],  where idx is the index in L1 of this segment candidate.
    """
    Candi2 = []
    for i in range(len(Candi)-1):
        if len(Candi[i][1]) == 0:                 # skip the samples that already has no candidate.
               continue
        if (Candi[i][0][0] != Candi[i+1][0][0]):   # skip the two samples that don't belong to one probe.
            Candi2.append(Candi[i])
            continue
        s1_id = Candi[i][0]
        s1_idx = smplID.index(s1_id)
        s1_speed = smplSpeed[s1_idx]
        s1_lat = smplLat[s1_idx][1]
        s1_lon = smplLon[s1_idx][1]
        
        s2_id = Candi[i+1][0]
        s2_idx = smplID.index(s2_id)
        s2_speed = smplSpeed[s2_idx]
        s2_lat = smplLat[s2_idx][1]
        s2_lon = smplLon[s2_idx][1]
        
        avg_speed = ( np.float64(s1_speed) + np.float64(s2_speed) )/2 /3.6
        timegap = 5 * (int(s2_id[1]) - int(s1_id[1]))
        passlen = avg_speed * timegap
    
        s_Candi2 = []
        for j in range(len(Candi[i][1])):
            l1_idx = int(Candi[i][1][j][1])
            l11_lat = L1[l1_idx][0]
            l11_lon = L1[l1_idx][1]
            l12_lat = L2[l1_idx][0]
            l12_lon = L2[l1_idx][1]       
            p1_lat, pl_lon = calcProjCoord(l11_lat, l11_lon, l12_lat, l12_lon, s1_lat, s1_lon)
            
            l2_idx = int(Candi[i+1][1][j][1])
            l21_lat = L1[l2_idx][0]
            l21_lon = L1[l2_idx][1]
            l22_lat = L2[l2_idx][0]
            l22_lon = L2[l2_idx][1] 
            p2_lat, p2_lon = calcProjCoord(l21_lat, l21_lon, l22_lat, l22_lon, s2_lat, s2_lon)
            dist = calcDistance(p1_lat, p1_lon, p2_lat, p2_lon)
            
            p1_idx = linkID.index(Candi[i][1][j][0])
            p1_lim = max(linkSpeed[p1_idx])
            p2_idx = linkID.index(Candi[i+1][1][j][0])
            p2_lim = max(linkSpeed[p2_idx])
            speedlim = max(p1_lim, p2_lim)
            
            thrsh = np.float64(max(p1_lim, p2_lim))/3.6 * timegap
            
            if passlen <= (dist + thrsh):
                s_Candi2.append(Candi[i][1][j])
                
        if len(s_Candi2) > 0:    # only keep the samples whose candidate list is not empty after filtering.
            each = []
            each.append(s1_id)
            each.append(s_Candi2)
            Candi2.append(each) 
 
    return Candi2
    
    



def nearestNeighbor(Candi, smplID, smplLat, smplLon, L1, L2 ):
    """
    Using 1-Neareast Neighbor strategy, determine the unique segment and link among the candidates for each sample.
    Input:   Candi:      A list of [smpl_id,  [[linkid, idx],
                                                  ...,
                                               [linkid, idx]] ],  where idx is the index in L1 of this segment candidate.
    
    Return:  Match:      The fianl determined match link for each sample.
                         A list of [smpl_id, [linkid, idx]],  where idx is the index in L1 of this segment candidate.
    """
    if os.path.exists('match-1NN.pk'):
        Match = pickle.load(open('match-1NN.pk', 'rb'))
        return Match
    
    Match = []
    for i in range(len(Candi)):
        if len(Candi[i][1]) == 0:                 # skip the samples that already has no candidate.
               continue
        
        if len(Candi[i][1]) == 1:
            Candi[i][1] = Candi[i][1][0]
            Match.append(Candi[i])
            continue
            
        s_id = Candi[i][0] 
        s_idx = smplID.index(s_id)
        s_lat = smplLat[s_idx][1]
        s_lon = smplLon[s_idx][1]     
        dist = []
        for j in range(len(Candi[i][1])):    # Candi[i][1] is the segment candidate list for sample_i
            l1_idx = int(Candi[i][1][j][1])
            l1_lat = L1[l1_idx][0]
            l1_lon = L1[l1_idx][1]
            l2_lat = L2[l1_idx][0]
            l2_lon = L2[l1_idx][1]
            dist_j = calcPerpDist(l1_lat, l1_lon, l2_lat, l2_lon, s_lat, s_lon)
            dist.append(dist_j)
        idx = np.argmin(dist)
        
        each = []
        each.append(s_id)
        each.append(Candi[i][1][idx])
        Match.append(each)    
        
        pickle.dump(Match, open('match-1NN.pk','wb'))
        
    return Match
    



def smoothSingular(Match):
    """
    This is used to refine the final match result. If the two mapped links of previous sample and next sample are same, 
    but different from the mapped link of the current sample, then we revise it to be same. This is based on the scenarios 
    that commonly happen at a crossing.
    """
    #win = 3, observe the previous one and next one.
    for i in range(1, len(Match)-1):
        curr_link = Match[i][1]
        prev_link = Match[i-1][1]
        next_link = Match[i+1][1]
        if ((prev_link[0] == next_link[0]) & (curr_link[0] != prev_link[0])):
            Match[i][1] = prev_link
    return Match
   


def calcSlope(lat1, lon1, alt1, lat2, lon2, alt2):
    """
    Given two point's latitude, longitude, and altitude information, calculate the slope of its mapped link.
    """
    y = alt2 - alt1
    x = calcDistance(lat1, lon1, lat2, lon2)
    slope = np.arctan2(y,x) 
    
    return slope



def deriveSlope(Match, smplID, smplLat, smplLon, smplAlt):
    """
    Derive the slopes of road links according to the map-matched result and the sample information.
    Input:      Match:     The fianl determined match link for each sample.
                           A list of [smpl_id, [linkid, idx]],  where idx is the index in L1 of this segment candidate. 
    Return:     Slopes:    A list of [linkid, slope_est], representing slopes estimated by samples, 
    """
    d = defaultdict(list)
    for i in range(len(Match)):
        key = Match[i][1][0]   # key: linkid, a string.
        val = Match[i][0]      # val: [smplid, smplid, ...], a list of smplid, where smplid is [probeid, j].
        d[key].append(val)
    
    Slopes = []
    for key, val in d.items():
        smpls = d[key]
        
        res = []
        for i in range(len(smpls)-1):
            cur_probe = smpls[i][0]
            next_probe = smpls[i+1][0]
            if next_probe != cur_probe:
                continue
            s1_idx = smplID.index(smpls[i])
            s1_lat = smplLat[s1_idx][1]
            s1_lon = smplLon[s1_idx][1]
            s1_alt = smplAlt[s1_idx]
            s2_idx = smplID.index(smpls[i+1])
            s2_lat = smplLat[s2_idx][1]
            s2_lon = smplLon[s2_idx][1]
            s2_alt = smplAlt[s2_idx]
            slope = calcSlope(s1_lat, s1_lon, s1_alt, s2_lat, s2_lon, s2_alt)
            if pd.isna(slope):
                continue
            res.append(slope)
        if pd.isna(np.mean(res)):
            continue
        each = []
        each.append(key)
        each.append(np.mean(res))
        Slopes.append(each)

    return Slopes   
    
 


def evalSlope(linkSlope, estiSlope):
    """
    Evaluate the derived road slope with the surveyed road slope in the link data file.
    Input:      linkSlope:       [[linkid, slope_gr], ...]
                estiSlope:       [[pvid, slope_est], ...]
    Output:     result:          [[linkid, linkslope, estislope, error],
                                                  ...,
                                  [linkid, linkslope, estislope, error]]
                avg_error:       mean of errors among all links in result.

    """
    pvids = np.stack(estiSlope, axis=1)[0]
    result = []
    for i in range(len(linkSlope)):
        l_id = linkSlope[i][0]
        idx = np.where(pvids == l_id)[0]
        if idx.shape[0] == 0:
            continue
        each = []
        each.append(l_id)
        each.append(linkSlope[i][1])
        each.append(estiSlope[idx[0]][1])
        err = float(linkSlope[i][1]) - float(estiSlope[idx[0]][1])
        each.append(err)
        result.append(each)

    if len(result) == 0:
        print("No common links computed.")
        return None
    avg_error = np.mean(np.float64(np.stack(result, axis=1)[3]))

    return result, avg_error







def calcProjCoord(l1_lat, l1_lon, l2_lat, l2_lon, s_lat, s_lon):
    """
    Input: the lat and lon coordinate of L1, L2, and S.
    Output: the lat and lon coordinate of P, the projection point of S on segment L1L2.
    L1, L2, P can be array-like for many points, and must be same size. S can only be for one point.
    """

    x1, y1, z1 = latlon2Cartesian(l1_lat, l1_lon)
    x2, y2, z2 = latlon2Cartesian(l2_lat, l2_lon)
    x3, y3, z3 = latlon2Cartesian(s_lat, s_lon)

    m = Symbol('m')   # parameter for solving
    L1L2 = np.array([x2-x1, y2-y1, z2-z1])
    SP = np.array([x1+m*(x2-x1)-x3, y1+m*(y2-y1)-y3, z1+m*(z2-z1)-z3])
    m = np.float64(solve(np.dot(L1L2, SP), m))

    P = [x1+m*(x2-x1), y1+m*(y2-y1), z1+m*(z2-z1)]
    p_lat, p_lon = cartesian2LatLon(P[0], P[1], P[2])

    return p_lat, p_lon 
    
    

def latlon2Cartesian(lat, lon):
    """
    Compute 3D Cartesian Coordinate (in kilometers) of a point given its Latitude/Longitude Coordinate (in degrees). Input array is supported.
    """
    sin = lambda x: np.sin(x)
    cos = lambda x: np.cos(x)
    R = 6371
    p = 0.017453292519943295     # Pi/180, for converting degree to radius
    x = R * cos(lat*p) * cos(lon*p)
    y = R * cos(lat*p) * sin(lon*p)
    z = R * sin(lat*p)
    return x, y, z



def cartesian2LatLon(x, y, z):
    """
    Compute Latitude/Longitude Coordinate (in degrees) of a point given its 3D Cartesian Coordinate (in kilometers). Input array is supported.
    """
    R = 6371
    lat = np.float64(np.arcsin(z/R) * 180/np.pi)
    lon = np.float64(np.arctan2(y,x) * 180/np.pi)
    return lat, lon




def calcDistFromRef(s_lat, s_lon, l1_lat, l1_lon, l2_lat, l2_lon, rf_lat, rf_lon):
    """
    Calculate the distance from the reference node to the map-matched sample point location on the link.
    """   
    p_lat, p_lon = calcProjCoord(l1_lat, l1_lon, l2_lat, l2_lon, s_lat, s_lon)
    dist = calcDistance(p_lat, p_lon, rf_lat, rf_lon)

    return np.float64(dist)



def calcSmplDirec(s_lat, s_lon, s_head, l1_lat, l1_lon, l2_lat, l2_lon):
    """
    Calculate the direction the sample was travelling on the link.
    Input:   s_lat:    sample's latitude value.
             s_lon:    sample's longitude value.
             s_head:   sample's heading value (0 ~ 359 degree).
             l1_lat:   the latitude value of the start point of the mapped segment.
             l1_lon:   the longitude value of the start point of the mapped segment.
             l2_lat:   the latitude value of the end point of the mapped segment.
             l2_lon:   the longitude value of the end point of the mapped segment.         
    Return:  direc:    A string, 'F' for From ref node or 'T' for Towards ref node.
    """
    l1l2 = calcBearing(l1_lat, l1_lon, l2_lat, l2_lon)
    diff = s_head - l1l2
    
    if diff > 180:
        diff = 360 - diff
    elif diff < -180:
        diff = diff + 360
        
    if np.absolute(diff) < 90:
        direc = 'F'
    else:
        direc = 'T' 
        
    return direc

 

def calcBearing(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing degree of the direction from L1(lat, lon1) to L2(lat2, lon2).
    """ 
    sin = lambda x: np.sin(x)
    cos = lambda x: np.cos(x)
    p = np.pi/180

    lat1 *= p
    lon1 *= p
    lat2 *= p
    lon2 *= p

    y = sin(lon2-lon1) * cos(lat2)
    x = cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(lon2-lon1)
    bearing = int(np.arctan2(y,x) / p)
        
    return bearing








def main():

	# load data
	linkID = loadLinkID()
	linkLat, linkLon = loadLinkLatLon()
	linkDirec = loadLinkDirec()
	linkSpeed = loadLinkSpeed()
	linkSlope = loadLinkSlope()

	smplID = loadSmplID()
	smplTime = loadSmplTime()
	smplSrcCode = loadSmplSrcCode()
	smplLat, smplLon = loadSmplLatLon()
	smplAlt = loadSmplAlt()
	smplSpeed = loadSmplSpeed()
	smplHeading = loadSmplHeading()
	print("data loading completed.")


	# create segments for all links
    print("segments for all links created.")
	L1, L2, L1_id = segmentL1L2(linkLat, linkLon)


	# generate candidates for each sample
    print("candidates for each sample generated.")
	Candi = genCandidate(smplLat, smplLon, L1, L2, L1_id)


	# filter the candidates and determine the final match
	#Candi2 = trvlDirecFilter(Candi, smplID, smplLat, smplLon, smplHeading, linkID, L1, L2, linkDirec)
	#Candi3 = passLengthFilter(Candi2, smplID, smplSpeed, smplLat, smplLon, linkID, L1, L2, linkSpeed)
    print("final matched links determined.")
	Match = nearestNeighbor(Candi, smplID, smplLat, smplLon, L1, L2)
	Match = smoothSingular(Match)


	# save mapmatching result to file
    if os.path.exists('Partition6467MatchedPoints.csv'):
        print("Mapping results saved to Partition6467MatchedPoints.csv.")
    else
        result = []
        for i in range(len(Match)):
            each = []
            s_id = Match[i][0] 
            s_idx = smplID.index(s_id)
            s_time = smplTime[s_idx]
            s_code = smplSrcCode[s_idx]
            s_lat = smplLat[s_idx][1]
            s_lon = smplLon[s_idx][1]
            s_alt = smplAlt[s_idx]
            s_speed = smplSpeed[s_idx]
            s_head = smplHeading[s_idx]
            
            l_id = Match[i][1][0]
            l1_idx = int(Match[i][1][1])
            l1_lat = L1[l1_idx][0]
            l1_lon = L1[l1_idx][1]
            l2_lat = L2[l1_idx][0]
            l2_lon = L2[l1_idx][1]
            s_direc = calcSmplDirec(s_lat, s_lon, s_head, l1_lat, l1_lon, l2_lat, l2_lon)
            rf_idx = list(np.stack(linkLat, axis=1)[0]).index(l_id)
            rf_lat = linkLat[rf_idx][1]
            rf_lon = linkLon[rf_idx][1]
            s_rfdist =  calcDistFromRef(s_lat, s_lon, l1_lat, l1_lon, l2_lat, l2_lon, rf_lat, rf_lon)
            s_pdist = calcPerpDist(l1_lat, l1_lon, l2_lat, l2_lon, s_lat, s_lon)

            each.append(s_id[0])
            each.append(s_time)
            each.append(s_code)
            each.append(s_lat)
            each.append(s_lon)
            each.append(s_alt)
            each.append(s_speed)
            each.append(s_head)
            each.append(l_id)
            each.append(s_direc)
            each.append(s_rfdist)
            each.append(s_pdist)
            result.append(each)
            
        
        with open('Partition6467MatchedPoints.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(result)


	# derive link slopes and calculate average error
    if os.path.exists('SlopeResults.csv'):
        print("Slope results saved to SlopeResults.csv.")
    else:
		estiSlope = deriveSlope(Match, smplID, smplLat, smplLon, smplAlt)
		res, avg_error = evalSlope(linkSlope, estiSlope)
		print("The average error between derived slopes and given slopes is:")
		print(avg_error)
		    
		with open('SlopeResults.csv', "w") as output:
		    writer = csv.writer(output, lineterminator='\n')
		    writer.writerows(res)


	# draw error scatter plot
	if os.path.exists('slope_erro_scatter.png'):
        print("Slope average error scatter plot saved to slope_error_scatter.png.")
    else:
		y = np.float64(np.stack(res, axis=1)[3])
		x = np.arange(len(res))

		plt.figure(figsize=(8, 8))
		plt.scatter(x, y, c="g", alpha=0.5, marker='v', label="one link")
		plt.xlabel("Link counting", fontsize=20)
		plt.ylabel("Residual of avg slope (degree)", fontsize=20)
		plt.legend(loc=2)
		plt.ylim(-90, 90)
		plt.savefig('slope_error_scatter' + '.png', dpi=300)
		plt.close()
		print("Slope average error scatter plot saved to slope_error_scatter.png.")









if __name__ == '__main__':
	main()






