-- This code is retrieved from https://github.com/MIT-LCP/mimic-code/blob/master/concepts/pivot/pivoted-lab.sql

--The parts about Glucose has been commented in the code. Since in the Nature paper it has been treated as a lab value instead of a vital sign 

DROP MATERIALIZED VIEW IF EXISTS getOthers CASCADE;
CREATE MATERIALIZED VIEW getOthers as

with ce as
(
  select ce.icustay_id
	, ce.subject_id
	, ce.hadm_id
    , ce.charttime
    , (case when itemid in (3801)  then valuenum else null end) as SGOT
    , (case when itemid in (3802)  then valuenum else null end) as SGPT
    , (case when itemid in (816,1350,3766,8177,8325,225667)  then valuenum else null end) as IonizedCalcium
    , case
          when itemid = 223835
            then case
              when valuenum > 0 and valuenum <= 1
                then valuenum * 100
              -- improperly input data - looks like O2 flow in litres
              when valuenum > 1 and valuenum < 21
                then null
              when valuenum >= 21 and valuenum <= 100
                then valuenum
              else null end -- unphysiological
        when itemid in (3420, 3422)
        -- all these values are well formatted
            then valuenum
        when itemid = 190 and valuenum > 0.20 and valuenum < 1
        -- well formatted but not in %
            then valuenum * 100
      else null end as FiO2 --use max to merge values at same time
  from mimiciii.chartevents ce
  where ce.error IS DISTINCT FROM 1
  and ce.itemid in
  (
  -- SGOT/SGPT
  3801, --"SGOT"
  3802, --"SGPT"
	  
  -- Ionized Calcium

  816,1350,3766,8177,8325,225667
  -- FiO2
  , 223835, 3420, 3422, 190

	 )

	)
  
select
  	subject_id
  , hadm_id	
  , ce.icustay_id
  , ce.charttime
  , avg(SGOT) as SGOT
  , avg(SGPT) as SGPT
  , avg(IonizedCalcium) as IonizedCalcium
  , avg(FiO2) as FiO2
  from ce
  group by ce.subject_id,ce.hadm_id,ce.icustay_id, ce.charttime
