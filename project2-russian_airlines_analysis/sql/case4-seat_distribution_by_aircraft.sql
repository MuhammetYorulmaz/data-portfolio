/*
CASE 4: Seat Distribution by Aircraft
Goal: Calculate the number and percentage of seats in each fare condition (Economy, Business, Comfort) 
for different aircraft models.
*/

SELECT S.AIRCRAFT_CODE,
	AD.MODEL ->> 'en' AS AIRCRAFT_MODEL,
	SUM(CASE WHEN S.FARE_CONDITIONS = 'Economy' THEN 1 ELSE 0 END) AS ECONOMY_SEAT_NUM,
	SUM(CASE WHEN S.FARE_CONDITIONS = 'Business' THEN 1 ELSE 0 END) AS BUSINESS_SEAT_NUM,
	SUM(CASE WHEN S.FARE_CONDITIONS = 'Comfort' THEN 1 ELSE 0 END) AS COMFORT_SEAT_NUM,
	COUNT(*) AS TOTAL_SEAT,
	'%' || ROUND((SUM(CASE WHEN S.FARE_CONDITIONS = 'Economy' THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2)::VARCHAR(10) AS ECONOMY_SEAT_PERCENTAGE,
    '%' || ROUND((SUM(CASE WHEN S.FARE_CONDITIONS = 'Business' THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2)::VARCHAR(10) AS BUSINESS_SEAT_PERCENTAGE,
    '%' || ROUND((SUM(CASE WHEN S.FARE_CONDITIONS = 'Comfort' THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2)::VARCHAR(10) AS COMFORT_SEAT_PERCENTAGE
FROM SEATS S
JOIN AIRCRAFTS_DATA AD ON S.AIRCRAFT_CODE = AD.AIRCRAFT_CODE
GROUP BY S.AIRCRAFT_CODE,
	AD.MODEL ->> 'en';