COPY
(SELECT * FROM sampled_with_ventdurations
WHERE 
sampled_with_ventdurations.icustay_id IN (
	SELECT distinct(icustay_id)
	FROM sampled_with_ventdurations
	WHERE sampled_with_ventdurations.vent_duration_h >= 24
	AND sampled_with_ventdurations.admission_age >= 18
	AND (sampled_with_ventdurations.mort90day is not null
		OR sampled_with_ventdurations.hospmort is not null))
)
TO '/tmp/ventilatedpatients.csv' DELIMITER ',' CSV HEADER;
