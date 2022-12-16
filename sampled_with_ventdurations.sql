--merge the sampled data with 
DROP MATERIALIZED VIEW IF EXISTS sampled_with_ventdurations;
CREATE MATERIALIZED VIEW sampled_with_ventdurations AS

WITH samp_vent_nums as (

        SELECT samp.icustay_id, samp.start_time, min(vd.ventnum) as vent_num

        FROM sampled_with_scdem_withventparams samp

        INNER JOIN ventdurations vd
        ON vd.icustay_id = samp.icustay_id
        AND (
             (samp.start_time <= vd.starttime
                        AND (samp.start_time +  '4 hours') <= vd.endtime)
          OR (samp.start_time <= vd.starttime
                        AND (samp.start_time +  '4 hours') <= vd.endtime)
          OR (samp.start_time >= vd.starttime
                        AND samp.start_time <= vd.endtime)
          OR (samp.start_time <= vd.starttime
                        AND (samp.start_time + '4 hours') >= vd.endtime)
        )
        GROUP BY samp.icustay_id, samp.start_time
)

SELECT samp.*, samp_vent_nums.vent_num, vd.duration_hours as vent_duration_h

FROM sampled_with_scdem_withventparams samp
LEFT OUTER JOIN samp_vent_nums
ON samp.icustay_id = samp_vent_nums.icustay_id
AND samp.start_time = samp_vent_nums.start_time
AND samp_vent_nums.vent_num = 1
LEFT OUTER JOIN ventdurations vd
ON samp_vent_nums.icustay_id = vd.icustay_id
AND samp_vent_nums.vent_num = vd.ventnum
