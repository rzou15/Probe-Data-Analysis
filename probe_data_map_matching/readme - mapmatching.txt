Input is provided in the following two files:

	
	Partition6467ProbePoints.csv	The raw probe points for partion 6467 in Germany (bounding rectangle - 50.62500, 8.43751, 53.43750, 11.25000)

	
	Partition6467LinkData.csv	The link data for the links in partition 6467 that probe points were map-matched to.


ProbePoints Record Format:

	sampleID, dateTime, sourceCode, latitude, longitude, altitude, speed, heading

		A. sampleID		is a unique identifier for the set of probe points that were collected from a particular phone.
		B. dateTime		is the date and time that the probe point was collected.
		C. sourceCode	is a unique identifier for the data supplier (13 = COMPANY).
		D. latitude		is the latitude in decimal degrees.
		E. longitude	is the longitude in decimal degrees.
		F. altitude		is the altitude in meters.
		G. speed		is the speed in KPH.
		H. heading		is the heading in degrees.


LinkData Record Format:

	linkPVID, refNodeID, nrefNodeID, length, functionalClass, directionOfTravel, speedCategory, fromRefSpeedLimit, toRefSpeedLimit, fromRefNumLanes, toRefNumLanes, multiDigitized, urban, timeZone, shapeInfo, curvatureInfo, slopeInfo

		A. linkPVID				is the published versioned identifier for the link.
		B. refNodeID			is the internal identifier for the link’s reference node.
		C. nrefNodeID			is the internal identifier for the link’s non-reference node.
		D. length				is the length of the link (in decimal meters).
		E. functionalClass		is the functional class for the link (1-5).
		F. directionOfTravel	is the allowed direction of travel for the link (F – from ref node, T – towards ref node, B - both)
		G. speedCategory		is the speed category for the link (1-8).
		H. fromRefSpeedLimit	is the speed limit for the link (in kph) in the direction of travel from the reference node.
		I. toRefSpeedLimit		is the speed limit for the link (in kph) in the direction of travel towards the reference node.
		J. fromRefNumLanes		is the number of lanes for the link in the direction of travel from the reference node.
		K. toRefNumLanes		is the number of lanes for the link in the direction of travel towards the reference node.
		L. multiDigitized		is a flag to indicate whether or not the link is multiply digitized (T – is multiply digitized, F – is singly digitized).
		M. urban				is a flag to indicate whether or not the link is in an urban area (T – is in urban area, F – is in rural area).
		N. timeZone				is the time zone offset (in decimal hours) from UTC.
		O. shapeInfo			contains an array of shape entries consisting of the latitude and longitude (in decimal degrees) and elevation (in decimal meters) for the link’s nodes and shape points ordered as reference node, shape points, non-reference node. The array entries are delimited by a vertical bar character and the latitude, longitude, and elevation values for each entry are delimited by a forward slash character (e.g. lat/lon/elev|lat/lon/elev). The elevation values will be null for link’s that don’t have 3D data.
		P. curvatureInfo		contains an array of curvature entries consisting of the distance from reference node (in decimal meters) and curvature at that point (expressed as a decimal value of 1/radius in meters). The array entries are delimited by a vertical bar character and the distance from reference node and curvature values for each entry are separated by a forward slash character (dist/curvature|dist/curvature). This entire field will be null if there is no curvature data for the link.
		Q. slopeInfo			contains an array of slope entries consisting of the distance from reference node (in decimal meters) and slope at that point (in decimal degrees). The array entries are delimited by a vertical bar character and the distance from reference node and slope values are separated by a forward slash character (dist/slope|dist/slope). This entire field will be null if there is no slope data for the link.


Map matching output is the following file:

Partition6467MatchedPoints.csv	The subset of probe points in partion 6467 that were successfully map-matched to a link.

MatchedPoints Record Format:

	sampleID, dateTime, sourceCode, latitude, longitude, altitude, speed, heading, linkPVID, direction, distFromRef, distFromLink

		sampleID	is a unique identifier for the set of probe points that were collected from a particular phone.
		dateTime	is the date and time that the probe point was collected.
		sourceCode	is a unique identifier for the data supplier (13 = Nokia).
		latitude	is the latitude in decimal degrees.
		longitude	is the longitude in decimal degrees.
		altitude	is the altitude in meters.
		speed		is the speed in KPH.
		heading		is the heading in degrees.
		linkPVID	is the published versioned identifier for the link.
		direction	is the direction the vehicle was travelling on thelink (F = from ref node, T = towards ref node).
		distFromRef	is the distance from the reference node to the map-matched probe point location on the link in decimal meters.
		distFromLink	is the perpendicular distance from the map-matched probe point location on the link to the probe point in decimal meters.


 