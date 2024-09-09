/*
CASE 3: Revenue by Airline
Calculate the total revenue generated by each airline.
*/
SELECT AD.MODEL ->> 'en' AS MODEL_OF_AIRCRAF,
	SUM(CASE WHEN TF.FARE_CONDITIONS = 'Economy' THEN TF.AMOUNT ELSE 0 END) AS ECONOMY,
	SUM(CASE WHEN TF.FARE_CONDITIONS = 'Business' THEN TF.AMOUNT ELSE 0 END) AS BUSINESS,
	SUM(CASE WHEN TF.FARE_CONDITIONS = 'Comfort' THEN TF.AMOUNT ELSE 0 END) AS COMFORT,
	SUM(TF.AMOUNT) AS TOTAL_REVENUE
FROM FLIGHTS F
JOIN AIRCRAFTS_DATA AD ON F.AIRCRAFT_CODE = AD.AIRCRAFT_CODE
JOIN TICKET_FLIGHTS TF ON F.FLIGHT_ID = TF.FLIGHT_ID
GROUP BY AD.MODEL ->> 'en'
ORDER BY TOTAL_REVENUE;