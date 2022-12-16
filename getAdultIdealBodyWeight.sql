DROP MATERIALIZED VIEW IF EXISTS getAdultIBW CASCADE;
CREATE MATERIALIZED VIEW getAdultIBW as 

WITH ht_stg AS
(
  SELECT
    gh.icustay_id, gh.subject_id, pt.gender, gh.height
    -- Ensure that all heights are in centimeters, and fix data as needed
  FROM getHeight2 gh
  INNER JOIN patients pt
    ON gh.subject_id = pt.subject_id
)
, NORM_WEIGHT as (
	select subject_id, icustay_id, MODE() within GROUP (order by gender) as gender, avg(height) as height from ht_stg
	group by subject_id, icustay_id
)
select 
	subject_id,
	icustay_id,
	gender, -- TODO FdH remove
	height,
	case when gender = 'M'
		then 50 + 0.91 * (height - 152.4)
		else
		case when gender = 'F'
		then 45 + 0.91 * (height - 152.4)
		else
		null
		end
	end
	as adult_ibw
from NORM_WEIGHT

