--query script to merge the sampled data with the corresponding scores and demographic information

DROP MATERIALIZED VIEW IF EXISTS sampled_with_scdem_withventparams;
CREATE MATERIALIZED VIEW sampled_with_scdem_withventparams AS

SELECT   --samp.* , sf.sofa , sr.sirs, dem.*, weig.*

		-- the above code also works but it has been given like that in order to preserve the position of sirs and sofa scores
		-- as given in the paper. With above syntax they will be at the end of the table.
		samp.icustay_id, ic.subject_id , ic.hadm_id, ic.intime, ic.outtime, samp.start_time , dem.admission_age,
		dem.gender, weig.weight, aibw.adult_ibw, dem.icu_readmission, dem.elixhauser_vanwalraven, sf.sofa , sr.sirs ,
		samp.gcs , samp.heartrate , samp.sysbp, samp.diasbp, samp.meanbp,
		samp.shockindex, samp.resprate, samp.tempc, samp.spo2, samp.potassium,
		samp.sodium, samp.chloride, samp.glucose, samp.bun, samp.creatinine, samp.magnesium,
		samp.calcium, samp.ionizedcalcium, samp.carbondioxide, samp.sgot, samp.sgpt, samp.bilirubin, samp.albumin, samp.hemoglobin,
		samp.wbc, samp.platelet, samp.ptt, samp.pt, samp.inr, samp.ph, samp.pao2, samp.paco2, samp.base_excess,
		samp.bicarbonate, samp.lactate, samp.pao2fio2ratio, samp.mechvent, samp.fio2, samp.urineoutput,
		samp.vaso_total, samp.iv_total, samp.cum_fluid_balance, samp.peep, samp.tidal_volume, samp.volume_controlled, samp.plateau_pressure,
		dem.hospmort, dem.mort90day, dem.dischtime, dem.deathtime, hadm.admittime as hadmittime, hadm.dischtime as hdischtime, gh.height as height
		--,vd.ventnum, vd.duration_hours as vent_duration_hours

FROM sampled_all_withventparams samp

LEFT JOIN getsirs_sampled_withventparams sr
ON samp.icustay_id=sr.icustay_id AND samp.start_time=sr.start_time

LEFT JOIN getsofa_sampled_withventparams sf
ON samp.icustay_id=sf.icustay_id AND samp.start_time=sf.start_time

LEFT JOIN demographics2 dem
ON samp.icustay_id=dem.icustay_id 

LEFT JOIN getweight2 weig
ON samp.icustay_id=weig.icustay_id

LEFT JOIN getadultibw aibw
ON samp.icustay_id=aibw.icustay_id

LEFT JOIN admissions hadm
ON dem.hadm_id = hadm.hadm_id

INNER JOIN mimiciii.icustays ic
ON samp.icustay_id=ic.icustay_id

INNER JOIN getHeight2 gh
ON samp.icustay_id=gh.icustay_id

--INNER JOIN ventdurations vd
--on samp.icustay_id=vd.icustay_id AND samp.start_time=vd.starttime

ORDER BY samp.icustay_id, samp.subject_id, samp.start_time
--LIMIT 1000000

