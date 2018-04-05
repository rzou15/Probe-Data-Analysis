========= Read me ==========
In the current directory, 
	map_matching.py is the source code.

	Partition6467MatchedPoints.csv is the final matching results.
		each row represents a sample match as asked.

	SlopeResults.csv saves the slope results. Each colomn represents respectively:
		linkID,
		derived average slope
		actual slope
		error


========= Algorithm description ========
1. Load data.
2. Create segments for all links
3. Generate candidates for each sample. For each sample given its lat/lon, determine a list of segment candidates whose lat/lon coordinates are within 30m GPS error range. If a sample's candidate list is empty, then the sample will not be considered any more as unsuccessfully map-matched. The Output is the remaining samples and their segment candidates.
4. Filter the candidates and determine the final match.
1) Using directionOfTravel as filter,  filter out the impossible segment candidates for each sample, 
    only keep the segment candidates whose directionOfTravel is conformed with the sample's current travel direction.
2) Using passLength, filter out the impossible segment candidates for each sample by the trace-based heuristics: 
    the passing length by two consecutive samples should not be way larger than the distance between the two projection points on the correspondingly mapped segments, 
    meaning that the difference should not be larger than a threshold. Specifically, the valid candidate should suffice:
                    AvgSpeed(S1,S2) * TimeGap  <=  Dist(P1,P2) + thrsh,
    where thrsh = max(SpeedLimit) * TimeGap
3) Using 1-Neareast Neighbor strategy, finally determine the unique segment and link among the candidates for each sample.
4) Applying smoothing singular strategy using slicing window. This is used to refine the final match result. If the two mapped links of previous sample and next sample are same, but different from the mapped link of the current sample, then we revise it to be same. This is based on the scenarios that commonly happen at a crossing.

5. Save mapmatching result to file.
6. Derive link slopes and calculate average error. For each link, compute the slope between the adjacent two samples S1, S2 of the same probe that are map matched on it by the following formula:
				arctan (alti_diff(S1, S2) /dist(S1, S2))
	take the mean of these slopes as the estimated average slope of this link.
7. Evaluation and visualization. Compute the difference between the truth average slope and the estimated average slope for each available link, as the residual error. Scatter plot of error is drawed.


========= Running requirement =========
Python 3.6
numpy 1.14

Before running the code, all the original dataset must be contained in the ./probe_data_map_matching/ of the current directory. Running code first time will generate many .pk files which save the intermediate data results for later use.








