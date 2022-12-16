-- This code extracts structured data from echocardiographies
-- You can join it to the text notes using ROW_ID
-- Just note that ROW_ID will differ across versions of MIMIC-III.

DROP MATERIALIZED VIEW IF EXISTS echo_data CASCADE;
CREATE MATERIALIZED VIEW echo_data as

select ROW_ID
  , subject_id, hadm_id
  , chartdate

  -- charttime is always null for echoes..
  -- however, the time is available in the echo text, e.g.:
  -- , substring(ne.text, 'Date/Time: [\[\]0-9*-]+ at ([0-9:]+)') as TIMESTAMP
  -- we can therefore impute it and re-create charttime
  , mimiciii.PARSE_DATETIME
  (
      '%Y-%m-%d%H:%M:%S',
      mimiciii.FORMAT_DATE('%Y-%m-%d', chartdate)
      || mimiciii.REGEXP_EXTRACT(ne.text, 'Date/Time: .+? at ([0-9]+:[0-9]{2})')
      || ':00'
   ) AS charttime

  -- explanation of below substring:
  --  'Indication: ' - matched verbatim
  --  (.*?) - match any character
  --  \n - the end of the line
  -- substring only returns the item in ()s
  -- note: the '?' makes it non-greedy. if you exclude it, it matches until it reaches the *last* \n

  , mimiciii.REGEXP_EXTRACT(ne.text, 'Indication: (.*?)\n') as Indication

  -- sometimes numeric values contain de-id text, e.g. [** Numeric Identifier **]
  -- this removes that text
--  , cast(mimiciii.REGEXP_EXTRACT(ne.text, 'Height: \\x28in\\x29 ([0-9]+)') as numeric) as Height
--  , cast(mimiciii.REGEXP_EXTRACT(ne.text, 'Weight \\x28lb\\x29: ([0-9]+)\n') as numeric) as Weight
  , cast((regexp_matches(ne.text, 'height:\s?\(?i?n?\)?\s?(\d+)"?', 'i'))[1] as numeric) as Height
  , cast((regexp_matches(ne.text, 'weight\s?\(lb\)\s?:\s?(\d+)', 'i'))[1] as numeric) as Weight
  , cast(mimiciii.REGEXP_EXTRACT(ne.text, 'BSA \\x28m2\\x29: ([0-9]+) m2\n') as numeric) as BSA -- ends in 'm2'
  , mimiciii.REGEXP_EXTRACT(ne.text, 'BP \\x28mm Hg\\x29: (.+)\n') as BP -- Sys/Dias
  , cast(mimiciii.REGEXP_EXTRACT(ne.text, 'BP \\x28mm Hg\\x29: ([0-9]+)/[0-9]+?\n') as numeric) as BPSys -- first part of fraction
  , cast(mimiciii.REGEXP_EXTRACT(ne.text, 'BP \\x28mm Hg\\x29: [0-9]+/([0-9]+?)\n') as numeric) as BPDias -- second part of fraction
  , cast(mimiciii.REGEXP_EXTRACT(ne.text, 'HR \\x28bpm\\x29: ([0-9]+?)\n') as numeric) as HR

  , mimiciii.REGEXP_EXTRACT(ne.text, 'Status: (.*?)\n') as Status
  , mimiciii.REGEXP_EXTRACT(ne.text, 'Test: (.*?)\n') as Test
  , mimiciii.REGEXP_EXTRACT(ne.text, 'Doppler: (.*?)\n') as Doppler
  , mimiciii.REGEXP_EXTRACT(ne.text, 'Contrast: (.*?)\n') as Contrast
  , mimiciii.REGEXP_EXTRACT(ne.text, 'Technical Quality: (.*?)\n') as TechnicalQuality
FROM mimiciii.noteevents ne
where category = 'Echo';
