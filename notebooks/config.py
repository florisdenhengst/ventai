SAMPLE_TIME_H = 4 # hours

demographics = ['admission_age', 'adult_ibw', 'height', 'weight', 'icu_readmission', 'elixhauser_vanwalraven', 'hospmort', 'mort90day', 'vent_duration_h']


vital_sign_vars = [
    'sofa',
    'sirs',
    'gcs',
    'heartrate',
    'sysbp',
    'meanbp',
    'diasbp',
    'shockindex',
    'resprate',
    'spo2',
    'tempc',
]

lab_vars = [
    'potassium',
    'sodium',
    'chloride',
    'glucose',
    'bun',
    'creatinine',
    'magnesium',
    'calcium',
    'ionizedcalcium',
    'carbondioxide',
    'bilirubin',
    'albumin',
    'hemoglobin',
    'wbc',
    'platelet',
    'ptt',
    'pt',
    'inr',
    'ph',
    'pao2',
    'paco2',
    'base_excess',
    'bicarbonate',
    'lactate',
    'pao2fio2ratio',
]

treatment_vars = [
    'iv_total',
    'vaso_total',
    'urineoutput',
    'cum_fluid_balance',
]

vent_vars = [
    'peep',
    'fio2',
    'tidal_volume',
    'mechvent',
]

guideline_vars = [
    'plateau_pressure',
]

inf = None
ffill_windows_clinical = {
    'sofa': 24 / SAMPLE_TIME_H,
    'sirs': 24 / SAMPLE_TIME_H,
    'gcs': inf,
    'heartrate': inf,
    'sysbp': inf,
    'meanbp': inf,
    'diasbp': inf,
    'shockindex': inf,
    'resprate': inf,
    'spo2': inf,
    'tempc': inf,
    'potassium': inf,
    'sodium': inf,
    'chloride': inf,
    'glucose': inf,
    'bun': inf,
    'creatinine': inf,
    'magnesium': inf,
    'calcium': inf,
    'ionizedcalcium': 8 / SAMPLE_TIME_H,
    'carbondioxide': inf,
    'bilirubin': inf,
    'albumin': inf,
    'hemoglobin': inf,
    'wbc': inf,
    'platelet': inf,
    'ptt': inf,
    'pt': inf,
    'inr': inf,
    'ph': inf,
    'pao2': inf,
    'paco2': inf,
    'base_excess': inf,
    'bicarbonate': inf,
    'lactate': inf,
    'pao2fio2ratio': inf,
    'iv_total': 8 / SAMPLE_TIME_H,
    'vaso_total': 24 / SAMPLE_TIME_H,
    'urineoutput': 8 / SAMPLE_TIME_H,
    'cum_fluid_balance': 8 / SAMPLE_TIME_H,
    'peep': 8 / SAMPLE_TIME_H,
    'fio2': 8 / SAMPLE_TIME_H,
    'tidal_volume': 8 / SAMPLE_TIME_H,
    'mechvent': 8 / SAMPLE_TIME_H,
    'plateau_pressure': 8 / SAMPLE_TIME_H
}

# Action space bins.
# Table 2a in Supplementary Material
# We assume these ranges are [lower bound, upper bound).
tv_bins = [
    (0, 2.5),
    (2.5, 5),
    (5, 7.5),
    (7.5, 10),
    (10, 12.5),
    (12.5, 15),
    (15, float('inf')),
]
peep_bins = [
    (0, 5),
    (5, 7),
    (7, 9),
    (9, 11),
    (11, 13),
    (13, 15),
    (15, float('inf')),
]
fio2_bins = [
    (20, 30), # NOTE: Peine uses 25 - 30, but this does not cover all of the data
    (30, 35),
    (35, 40),
    (40, 45),
    (45, 50),
    (50, 55),
    (55, float('inf')),
]