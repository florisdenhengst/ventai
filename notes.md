select *,
	lag(icustay_seq, 1) over (
		partition by hadm_id
		order by icustay_seq ASC
	) as prev_icustay_seq
from demographics2 where
hadm_id=100242;









