#!/bin/bash


declare -a qryfiles=(
	# stage 0
	"postgres-functions.sql"
	"echo_data.sql"

	# stage 1

	"Vasopressors/weight_durations.sql"
	"Vasopressors/dopamine-dose.sql"
	"Vasopressors/epinephrine_dose.sql"
	"Vasopressors/norepinephrine_dose.sql"
	"Vasopressors/phenylephrine_dose.sql"
	"Vasopressors/vasopressin-dose.sql"

	"getGCS.sql"
	"getVitalSigns.sql"
	"getLabValues.sql"
       "getOthers.sql"
       "getVentilationParams.sql"
       "getVentilationParams2.sql"
       "vent_parameters.sql"
	"getIntravenous.sql"
	"getVasopressors.sql"
	"getUrineOutput.sql"
	"getCumFluid.sql"
	"getElixhauser_score.sql"
	"demographics.sql"
	"getWeight.sql"
	"getHeight.sql"
	"getAdultIdealBodyWeight.sql"
	# stage 2	
	"overalltable_Lab_withventparams.sql"
	"overalltable_withoutLab_withventparams.sql"
	# stage 3
	"sampling_lab_withventparams.sql"
	"sampling_withoutlab_withventparams.sql"
	"sampling_all_withventparams.sql"
	# stage 4
	"getSIRS_withventparams.sql"
	"getSOFA_withventparams.sql"
	# stage 5
	"sampled_data_with_scdem_withventparams.sql"
	"sampled_with_ventdurations.sql"
	# stage 6 -- not present
#	"cohort_withventparams_all.sql"
)

for f in "${qryfiles[@]}"; do
	echo $f;
	./runquery.sh $f
done
