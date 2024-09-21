/*
CASE 7: Create a Seating Arrangement View.
*/

CREATE VIEW SEATING_ARRANGEMENT_VIEW AS
SELECT AIRCRAFT_CODE, SEAT_NO, FARE_CONDITIONS, 
	CASE 
		WHEN (RIGHT(SEAT_NO, 1) IN ('A','F')) AND 
			 (AIRCRAFT_CODE IN ('319','320','321','733')) THEN 'window seat'
		WHEN (RIGHT(SEAT_NO, 1) IN ('B','E')) AND 
			 (FARE_CONDITIONS = 'Economy') AND 
			 (AIRCRAFT_CODE IN ('319','320','321','733')) THEN 'middle seat'
		WHEN (RIGHT(SEAT_NO, 1) IN ('C','D')) AND 
			 (AIRCRAFT_CODE IN ('319','320','321','733')) THEN 'aisle seat'
			 
		WHEN (RIGHT(SEAT_NO, 1) IN ('A','H')) AND 
			 (AIRCRAFT_CODE = '763') THEN 'window seat'
		WHEN (RIGHT(SEAT_NO, 1) IN ('B','D','F','G')) AND  
			 (FARE_CONDITIONS = 'Economy') AND 
			 (AIRCRAFT_CODE = '763') THEN 'aisle seat'
		WHEN (RIGHT(SEAT_NO, 1) = 'E') AND  
			 (FARE_CONDITIONS = 'Economy') AND 
			 (AIRCRAFT_CODE = '763') THEN 'middle seat'
		WHEN (RIGHT(SEAT_NO, 1) IN ('B','G')) AND  
			 (FARE_CONDITIONS = 'Business') AND 
			 (AIRCRAFT_CODE = '763') THEN 'middle seat'
		WHEN (RIGHT(SEAT_NO, 1) IN ('C','F')) AND  
			 (FARE_CONDITIONS = 'Business') AND 
			 (AIRCRAFT_CODE = '763') THEN 'aisle seat'
			 
		WHEN (RIGHT(SEAT_NO, 1) IN ('A','K')) AND 
			 (AIRCRAFT_CODE = '773') THEN 'window seat'
		WHEN (RIGHT(SEAT_NO, 1) IN ('C','D','G','H')) AND 
			 (AIRCRAFT_CODE = '773') THEN 'aisle seat'
		WHEN (RIGHT(SEAT_NO, 1) IN ('E','F')) AND  
			 (FARE_CONDITIONS = 'Comfort') AND 
			 (AIRCRAFT_CODE = '773') THEN 'aisle seat'
		WHEN (RIGHT(SEAT_NO, 1) IN ('B','E','F','J')) AND  
			 (FARE_CONDITIONS = 'Economy') AND 
			 (AIRCRAFT_CODE = '773') THEN 'middle seat'
			 
		WHEN (RIGHT(SEAT_NO, 1) IN ('A','F')) AND 
			 (AIRCRAFT_CODE = 'SU9') THEN 'window seat'
		WHEN (RIGHT(SEAT_NO, 1) IN ('C','D')) AND 
			 (AIRCRAFT_CODE = 'SU9') THEN 'aisle seat'
		WHEN (RIGHT(SEAT_NO, 1) = 'E') AND  
			 (FARE_CONDITIONS = 'Economy') AND 
			 (AIRCRAFT_CODE = 'SU9') THEN 'middle seat'

		WHEN (RIGHT(SEAT_NO, 1) IN ('A','D')) AND 
			 (AIRCRAFT_CODE = 'CR2') THEN 'window seat'
		WHEN (RIGHT(SEAT_NO, 1) IN ('B','C')) AND 
			 (AIRCRAFT_CODE = 'CR2') THEN 'aisle seat'
			 
		WHEN (AIRCRAFT_CODE = 'CN1') THEN 'window seat'
	END AS SEATING_ARRANGEMENT
FROM SEATS;







