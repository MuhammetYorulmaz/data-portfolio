/*
CASE 8: Check Seat Availability for a Flight
Goal: To create a function that checks if a specific seat is available on a flight.
*/

DROP FUNCTION IF EXISTS CHECK_SEAT_AVAILABILITY(INT, VARCHAR);

CREATE FUNCTION CHECK_SEAT_AVAILABILITY(FLIGHT_ID_INPUT INT, SEAT_NO_INPUT VARCHAR) 
RETURNS TEXT AS $$

DECLARE
SEAT_TAKEN BOOLEAN;	
BEGIN
	IF NOT EXISTS(
		SELECT 1
		FROM FLIGHTS F
		WHERE F.FLIGHT_ID = FLIGHT_ID_INPUT 
	) THEN 
		RETURN 'Flight ID not found';
	END IF;

    IF NOT EXISTS(
		SELECT 1
		FROM FLIGHTS F
		JOIN SEATS S ON F.AIRCRAFT_CODE = S.AIRCRAFT_CODE
		WHERE F.FLIGHT_ID = FLIGHT_ID_INPUT
			AND S.SEAT_NO = SEAT_NO_INPUT 
	) THEN 
		RETURN 'Seat not found on this aircraft';
	END IF;
	
    SELECT EXISTS(
		SELECT 1
		FROM BOARDING_PASSES BP
		WHERE BP.FLIGHT_ID = FLIGHT_ID_INPUT
			AND BP.SEAT_NO = SEAT_NO_INPUT 
	) INTO SEAT_TAKEN;

	IF SEAT_TAKEN 
	THEN 
		RETURN 'Seat is already taken';
	ELSE 
		RETURN 'Seat is available';
	END IF;
	
END;
$$ LANGUAGE PLPGSQL;