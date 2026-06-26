-- =============================================================================
-- MIMIC-IV Delirium Prediction Cohort Selection 
-- =============================================================================
--
-- PURPOSE:
-- Selects a cohort of ICU patients for delirium prediction using EHR data
-- from the first 24 hours of ICU admission.
--
-- DELIRIUM DEFINITION:
-- Positive CAM-ICU assessment (itemid 228332) or derived from CAM-ICU
-- components. No ICDSC exists in MIMIC-IV.
--
-- COHORT CRITERIA:
-- 1.  ICU patients only
-- 2.  Age >= 18
-- 3.  LOS >= 24 hours
-- 4.  First ICU admission only per patient
-- 5.  Exclude deaths within 48 hours of admission
-- 6.  Exclude delirium or coma in first 24 hours
-- 7.  Require EHR data in first 24 hours
-- 8.  Remove patients with delirium ICD diagnosis but no positive CAM-ICU
-- 9.  Require >= 2 positive CAM-ICU assessments after 24 hours
-- 10. Exclude non-pure delirium (drug withdrawal, drug-induced, intoxication)


SET search_path TO mimiciv_hosp, mimiciv_icu;


-- =============================================================================
-- STEP 1A: Extract overall delirium assessments
-- =============================================================================
DROP TABLE IF EXISTS overall_assessments;
CREATE TEMP TABLE overall_assessments AS
SELECT
    ce.subject_id,
    ce.hadm_id,
    ce.stay_id,
    ce.charttime,
    icu.intime,
    EXTRACT(EPOCH FROM (ce.charttime - icu.intime)) / 60.0 AS chart_offset_min,
    ce.value,
    CASE
        WHEN ce.value = 'Positive' THEN 1
        WHEN ce.value = 'Negative' THEN 0
        ELSE NULL  -- UTA mapped to NULL
    END AS del_positive
FROM mimiciv_icu.chartevents ce
INNER JOIN mimiciv_icu.icustays icu ON ce.stay_id = icu.stay_id
WHERE ce.itemid = 228332
  AND ce.value IS NOT NULL;

-- CHECK: How many overall assessments?
SELECT COUNT(*) AS total_rows,
        SUM(CASE WHEN del_positive = 1 THEN 1 ELSE 0 END) AS positive,
        SUM(CASE WHEN del_positive = 0 THEN 1 ELSE 0 END) AS negative,
        SUM(CASE WHEN del_positive IS NULL THEN 1 ELSE 0 END) AS uta
FROM overall_assessments;

-- =============================================================================
-- STEP 1B: Derive delirium from CAM-ICU components
-- =============================================================================
-- 2,160 ICU stays have CAM-ICU component scores but NO overall assessment
-- (itemid 228332). For these stays, we derive the delirium result using
-- the clinical CAM-ICU logic:
--
-- CAM-ICU POSITIVE when ALL of:
--   Feature 1: Acute onset / fluctuation in mental status (MS Change) = Yes
--   Feature 2: Inattention = Yes
--   AND EITHER:
--     Feature 3: Altered level of consciousness = Yes
--     OR Feature 4: Disorganized thinking = Yes
--
-- Component itemids (verified from d_items):
--   Feature 1 (MS Change):          228300, 228337, 229326
--   Feature 2 (Inattention):        228301, 229325, 228336
--   Feature 3 (Altered LOC):        228334
--   Feature 4 (Disorganized Think): 228303, 229324, 228335
--
-- Values use LIKE 'yes%' to handle both formats:
--   Old: "Yes (Continue)", "No (Stop - Not delirious)"
--   New: "Yes", "No"
-- =============================================================================

-- First, find stays that have components but no overall assessment
DROP TABLE IF EXISTS component_only_stays;
CREATE TEMP TABLE component_only_stays AS
SELECT DISTINCT ce.stay_id
FROM mimiciv_icu.chartevents ce
WHERE ce.itemid IN (228300, 228301, 228302, 228303, 228334,
                    228335, 228336, 228337, 229324, 229325, 229326)
  AND ce.stay_id NOT IN (SELECT DISTINCT stay_id FROM overall_assessments);

-- Extract and pivot component values by stay and charttime
DROP TABLE IF EXISTS cam_components;
CREATE TEMP TABLE cam_components AS
SELECT
    ce.subject_id,
    ce.hadm_id,
    ce.stay_id,
    ce.charttime,
    icu.intime,
    EXTRACT(EPOCH FROM (ce.charttime - icu.intime)) / 60.0 AS chart_offset_min,
    -- Feature 1: MS Change (acute onset / fluctuation)
    MAX(CASE
        WHEN ce.itemid IN (228300, 228337, 229326)
             AND LOWER(ce.value) LIKE 'yes%' THEN 1
        WHEN ce.itemid IN (228300, 228337, 229326)
             AND LOWER(ce.value) LIKE 'no%' THEN 0
        ELSE NULL
    END) AS feature1_ms_change,
    -- Feature 2: Inattention
    MAX(CASE
        WHEN ce.itemid IN (228301, 229325, 228336)
             AND LOWER(ce.value) LIKE 'yes%' THEN 1
        WHEN ce.itemid IN (228301, 229325, 228336)
             AND LOWER(ce.value) LIKE 'no%' THEN 0
        ELSE NULL
    END) AS feature2_inattention,
    -- Feature 3: Altered level of consciousness
    MAX(CASE
        WHEN ce.itemid = 228334
             AND LOWER(ce.value) = 'yes' THEN 1
        WHEN ce.itemid = 228334
             AND LOWER(ce.value) = 'no' THEN 0
        ELSE NULL
    END) AS feature3_altered_loc,
    -- Feature 4: Disorganized thinking
    MAX(CASE
        WHEN ce.itemid IN (228303, 229324, 228335)
             AND LOWER(ce.value) LIKE 'yes%' THEN 1
        WHEN ce.itemid IN (228303, 229324, 228335)
             AND LOWER(ce.value) LIKE 'no%' THEN 0
        ELSE NULL
    END) AS feature4_disorg_thinking
FROM mimiciv_icu.chartevents ce
INNER JOIN mimiciv_icu.icustays icu ON ce.stay_id = icu.stay_id
WHERE ce.stay_id IN (SELECT stay_id FROM component_only_stays)
  AND ce.itemid IN (228300, 228301, 228302, 228303, 228334,
                    228335, 228336, 228337, 229324, 229325, 229326)
  AND ce.value IS NOT NULL
GROUP BY ce.subject_id, ce.hadm_id, ce.stay_id, ce.charttime, icu.intime;

-- Apply CAM-ICU logic to derive result
DROP TABLE IF EXISTS derived_assessments;
CREATE TEMP TABLE derived_assessments AS
SELECT
    subject_id,
    hadm_id,
    stay_id,
    charttime,
    intime,
    chart_offset_min,
    'Derived' AS value,
    CASE
        -- POSITIVE: Feature1 AND Feature2 AND (Feature3 OR Feature4)
        WHEN feature1_ms_change = 1
             AND feature2_inattention = 1
             AND (feature3_altered_loc = 1 OR feature4_disorg_thinking = 1)
        THEN 1
        -- NEGATIVE: any required feature explicitly negative
        WHEN feature1_ms_change = 0
             OR feature2_inattention = 0
             OR (feature3_altered_loc = 0 AND feature4_disorg_thinking = 0)
        THEN 0
        -- INDETERMINATE: not enough info to decide
        ELSE NULL
    END AS del_positive
FROM cam_components;

-- CHECK: How many derived assessments?
 SELECT COUNT(*) AS total,
       SUM(CASE WHEN del_positive = 1 THEN 1 ELSE 0 END) AS positive,
	   SUM(CASE WHEN del_positive = 0 THEN 1 ELSE 0 END) AS negative,
      SUM(CASE WHEN del_positive IS NULL THEN 1 ELSE 0 END) AS indeterminate
FROM derived_assessments;


-- =============================================================================
-- STEP 1C: Combine assessments and identify UTA-only stays
-- =============================================================================
-- 3,065 stays have ONLY "Unable to Assess" results (no Positive or Negative).
-- These cannot be used for training or evaluation, so we flag them for
-- exclusion in Step 2.
-- =============================================================================

-- Identify stays where every single assessment was UTA
DROP TABLE IF EXISTS uta_only_stays;
CREATE TEMP TABLE uta_only_stays AS
SELECT stay_id
FROM overall_assessments
GROUP BY stay_id
HAVING COUNT(*) FILTER (WHERE value = 'UTA') = COUNT(*);

-- Combine overall + derived assessments (excluding NULL/indeterminate)
DROP TABLE IF EXISTS delirium_assessments;
CREATE TEMP TABLE delirium_assessments AS
SELECT subject_id, hadm_id, stay_id, charttime, intime,
       chart_offset_min, value, del_positive
FROM overall_assessments
WHERE del_positive IS NOT NULL
UNION ALL
SELECT subject_id, hadm_id, stay_id, charttime, intime,
       chart_offset_min, value, del_positive
FROM derived_assessments
WHERE del_positive IS NOT NULL;

-- CHECK: Total combined assessments
SELECT COUNT(*) AS total,
        COUNT(DISTINCT stay_id) AS unique_stays,
      SUM(del_positive) AS positive_assessments
	  FROM delirium_assessments;


-- =============================================================================
-- STEP 2: Base cohort
-- =============================================================================
-- Applies:
--   Criterion 1: ICU patients (inherent from icustays table)
--   Criterion 2: Age >= 18 (calculated from anchor_age + year offset)
--   Criterion 3: LOS >= 24 hours (1440 minutes)
--   Excludes: UTA-only stays (3,065)
--   Excludes: Dementia with delirium (ICD-9: 29041, 2903, 29011)
-- =============================================================================

-- Identify dementia-with-delirium patients
DROP TABLE IF EXISTS dementia_delirium_patients;
CREATE TEMP TABLE dementia_delirium_patients AS
SELECT DISTINCT di.hadm_id
FROM mimiciv_hosp.diagnoses_icd di
WHERE
    -- ICD-9: dementia with delirium
    (di.icd_version = 9 AND di.icd_code IN ('29041', '2903', '29011'))
    -- ICD-10: dementia with delirium
    OR (di.icd_version = 10 AND (
        di.icd_code LIKE 'F0151%'   -- Vascular dementia with delirium
        OR di.icd_code LIKE 'F0251%' -- Dementia with Lewy bodies, with delirium
        OR di.icd_code LIKE 'F0351%' -- Unspecified dementia with delirium
        OR di.icd_code LIKE 'F051%'  -- Delirium superimposed on dementia
    ));

DROP TABLE IF EXISTS base_cohort;
CREATE TEMP TABLE base_cohort AS
SELECT
    icu.stay_id,
    icu.subject_id,
    icu.hadm_id,
    icu.intime,
    icu.outtime,
    EXTRACT(EPOCH FROM (icu.outtime - icu.intime)) / 60.0 AS los_minutes,
    p.anchor_age + (EXTRACT(YEAR FROM icu.intime) - p.anchor_year) AS age,
    p.gender,
    adm.deathtime,
    adm.hospital_expire_flag
FROM mimiciv_icu.icustays icu
INNER JOIN mimiciv_hosp.patients p ON icu.subject_id = p.subject_id
INNER JOIN mimiciv_hosp.admissions adm ON icu.hadm_id = adm.hadm_id
WHERE
    -- Criterion 2: Age >= 18
    (p.anchor_age + (EXTRACT(YEAR FROM icu.intime) - p.anchor_year)) >= 18
    -- Criterion 3: LOS >= 24 hours (1440 minutes)
    AND EXTRACT(EPOCH FROM (icu.outtime - icu.intime)) / 60.0 >= 1440
    -- Exclude UTA-only stays
    AND icu.stay_id NOT IN (SELECT stay_id FROM uta_only_stays)
    -- Exclude dementia with delirium
    AND icu.hadm_id NOT IN (SELECT hadm_id FROM dementia_delirium_patients);

-- CHECK:
SELECT COUNT(*) AS base_cohort_count FROM base_cohort;


-- =============================================================================
-- STEP 3: Criterion 4 - First ICU admission only per patient
-- =============================================================================
-- A patient may have multiple ICU stays across multiple hospitalizations.
-- We keep only the earliest ICU stay per subject_id.
-- =============================================================================
DROP TABLE IF EXISTS first_admission;
CREATE TEMP TABLE first_admission AS
SELECT DISTINCT ON (subject_id)
    stay_id, subject_id, hadm_id, intime, outtime,
    los_minutes, age, gender, deathtime, hospital_expire_flag
FROM base_cohort
ORDER BY subject_id, intime ASC;

-- CHECK:
SELECT COUNT(*) AS first_admission_count FROM first_admission;


-- =============================================================================
-- STEP 4: Criterion 5 - Exclude deaths within 48 hours
-- =============================================================================
-- Uses deathtime when available.
-- For 11 patients who died in hospital but have NULL deathtime,
-- hospital_expire_flag = 1 triggers exclusion since we cannot confirm
-- they survived past 48 hours.
-- =============================================================================
DROP TABLE IF EXISTS survived_48h;
CREATE TEMP TABLE survived_48h AS
SELECT *
FROM first_admission
WHERE
    -- Survived: no death recorded
    (deathtime IS NULL AND hospital_expire_flag = 0)
    -- Died but after 48 hours (2880 minutes)
    OR (deathtime IS NOT NULL
        AND EXTRACT(EPOCH FROM (deathtime - intime)) / 60.0 >= 2880);

-- CHECK:
SELECT COUNT(*) AS survived_48h_count FROM survived_48h;


-- =============================================================================
-- STEP 5: Criterion 6 - Exclude early delirium and coma (first 24 hours)
-- =============================================================================

-- 5a: Patients with positive delirium assessment in first 24 hours
DROP TABLE IF EXISTS early_delirium_patients;
CREATE TEMP TABLE early_delirium_patients AS
SELECT DISTINCT stay_id
FROM delirium_assessments
WHERE del_positive = 1
  AND chart_offset_min >= 0
  AND chart_offset_min < 1440;

-- 5b: Patients with coma in first 24 hours (GCS total <= 8)
-- GCS itemids verified from d_items:
--   220739 = GCS - Eye Opening (1-4)
--   223900 = GCS - Verbal Response (1-5)
--   223901 = GCS - Motor Response (1-6)
-- All three components must be present to calculate total.
DROP TABLE IF EXISTS gcs_components;
CREATE TEMP TABLE gcs_components AS
SELECT
    ce.stay_id,
    ce.charttime,
    MAX(CASE WHEN ce.itemid = 220739 THEN ce.valuenum END) AS gcs_eye,
    MAX(CASE WHEN ce.itemid = 223900 THEN ce.valuenum END) AS gcs_verbal,
    MAX(CASE WHEN ce.itemid = 223901 THEN ce.valuenum END) AS gcs_motor
FROM mimiciv_icu.chartevents ce
INNER JOIN survived_48h s ON ce.stay_id = s.stay_id
WHERE ce.itemid IN (220739, 223900, 223901)
  AND EXTRACT(EPOCH FROM (ce.charttime - s.intime)) / 60.0 >= 0
  AND EXTRACT(EPOCH FROM (ce.charttime - s.intime)) / 60.0 < 1440
GROUP BY ce.stay_id, ce.charttime;

DROP TABLE IF EXISTS early_coma_patients;
CREATE TEMP TABLE early_coma_patients AS
SELECT DISTINCT stay_id
FROM gcs_components
WHERE gcs_eye IS NOT NULL
  AND gcs_verbal IS NOT NULL
  AND gcs_motor IS NOT NULL
  AND (gcs_eye + gcs_verbal + gcs_motor) <= 8;

-- Apply both exclusions
DROP TABLE IF EXISTS no_early_delirium;
CREATE TEMP TABLE no_early_delirium AS
SELECT *
FROM survived_48h
WHERE stay_id NOT IN (SELECT stay_id FROM early_delirium_patients)
  AND stay_id NOT IN (SELECT stay_id FROM early_coma_patients);

-- CHECK:
SELECT COUNT(*) AS no_early_delirium_count FROM no_early_delirium;


-- =============================================================================
-- STEP 6: Criterion 7 - Require EHR data in first 24 hours
-- =============================================================================
-- Checks both chartevents AND labevents.
-- 12 stays have no chartevents in the first 24h but do have labevents.
-- This ensures those 12 are not incorrectly excluded.
-- =============================================================================
DROP TABLE IF EXISTS has_ehr_data;
CREATE TEMP TABLE has_ehr_data AS
SELECT DISTINCT stay_id FROM (
    -- Check chartevents
    SELECT ce.stay_id
    FROM mimiciv_icu.chartevents ce
    INNER JOIN no_early_delirium ned ON ce.stay_id = ned.stay_id
    WHERE EXTRACT(EPOCH FROM (ce.charttime - ned.intime)) / 60.0 BETWEEN 0 AND 1440
    UNION
    -- Check labevents
    SELECT icu.stay_id
    FROM mimiciv_hosp.labevents le
    INNER JOIN mimiciv_icu.icustays icu ON le.subject_id = icu.subject_id
    INNER JOIN no_early_delirium ned ON icu.stay_id = ned.stay_id
    WHERE EXTRACT(EPOCH FROM (le.charttime - ned.intime)) / 60.0 BETWEEN 0 AND 1440
) combined;

DROP TABLE IF EXISTS with_ehr_data;
CREATE TEMP TABLE with_ehr_data AS
SELECT ned.*
FROM no_early_delirium ned
WHERE ned.stay_id IN (SELECT stay_id FROM has_ehr_data);

-- CHECK:
 SELECT COUNT(*) AS with_ehr_data_count FROM with_ehr_data;


-- =============================================================================
-- STEP 7: Criterion 8 - Remove delirium ICD diagnosis without positive CAM-ICU
-- =============================================================================
-- Some patients have an ICD diagnosis of delirium but never had a positive
-- CAM-ICU assessment. These are unreliable for our label definition and
-- are excluded.
--
-- Delirium ICD codes (verified from data):
--   ICD-10: F05 (delirium due to known physiological condition, 5114 patients)
--   ICD-9:  2930 (delirium due to conditions classified elsewhere, 3581)
--   ICD-9:  2931 (subacute delirium, 35)
--   ICD-9:  29281 (drug-induced delirium, 925)
--   ICD-9:  2910 (alcohol withdrawal delirium, 347)
--   ICD-10: F1X_31 pattern (substance withdrawal with delirium)
--   ICD-10: F1X_21 pattern (substance intoxication with delirium)
-- =============================================================================
DROP TABLE IF EXISTS has_delirium_diag;
CREATE TEMP TABLE has_delirium_diag AS
SELECT DISTINCT s.stay_id
FROM mimiciv_hosp.diagnoses_icd di
INNER JOIN with_ehr_data s ON di.hadm_id = s.hadm_id
WHERE
    -- Pure delirium
    (di.icd_version = 10 AND di.icd_code LIKE 'F05%')
    OR (di.icd_version = 9 AND di.icd_code IN ('2930', '2931'))
    -- Drug-induced delirium
    OR (di.icd_version = 9 AND di.icd_code = '29281')
    -- Alcohol withdrawal delirium
    OR (di.icd_version = 9 AND di.icd_code = '2910')
    -- Substance withdrawal with delirium (ICD-10)
    OR (di.icd_version = 10 AND di.icd_code LIKE 'F1_%31')
    -- Substance intoxication with delirium (ICD-10)
    OR (di.icd_version = 10 AND di.icd_code LIKE 'F1_%21');

DROP TABLE IF EXISTS has_positive_assessment;
CREATE TEMP TABLE has_positive_assessment AS
SELECT DISTINCT stay_id
FROM delirium_assessments
WHERE del_positive = 1;

-- Patients with diagnosis but zero positive assessments
DROP TABLE IF EXISTS diag_no_positive;
CREATE TEMP TABLE diag_no_positive AS
SELECT hd.stay_id
FROM has_delirium_diag hd
WHERE hd.stay_id NOT IN (SELECT stay_id FROM has_positive_assessment);

DROP TABLE IF EXISTS no_diag_only;
CREATE TEMP TABLE no_diag_only AS
SELECT *
FROM with_ehr_data
WHERE stay_id NOT IN (SELECT stay_id FROM diag_no_positive);

-- CHECK:
SELECT COUNT(*) AS no_diag_only_count FROM no_diag_only;


-- =============================================================================
-- STEP 8: Criterion 9 - Minimum 2 positive assessments after 24 hours
-- =============================================================================
-- For a patient to be labeled delirium-positive, they must have at least
-- 2 positive CAM-ICU assessments after the first 24 hours (1440 minutes).
-- Patients with only 1 positive assessment are labeled as negative.
-- This reduces false positives from isolated or erroneous charting.
-- =============================================================================
DROP TABLE IF EXISTS positive_after_24h;
CREATE TEMP TABLE positive_after_24h AS
SELECT stay_id, COUNT(*) AS positive_count
FROM delirium_assessments
WHERE del_positive = 1
  AND chart_offset_min >= 1440
GROUP BY stay_id
HAVING COUNT(*) >= 2;

-- CHECK:
SELECT COUNT(*) AS stays_with_2plus_positives FROM positive_after_24h;


-- =============================================================================
-- STEP 9: Criterion 10 - Exclude non-pure delirium
-- =============================================================================
-- Excludes patients with ANY of:
--   A) Substance WITHDRAWAL diagnoses (all substances)
--   B) Drug-INDUCED delirium (ICD-9 29281)
--   C) Substance INTOXICATION with delirium (all substances)
--
-- This ensures the cohort contains only "pure" delirium cases not caused
-- by substance use, withdrawal, or intoxication.
--
-- =============================================================================
DROP TABLE IF EXISTS exclude_substance_delirium;
CREATE TEMP TABLE exclude_substance_delirium AS
SELECT DISTINCT s.stay_id
FROM mimiciv_hosp.diagnoses_icd di
INNER JOIN no_diag_only s ON di.hadm_id = s.hadm_id
WHERE

    -- ===================
    -- A) ICD-9 WITHDRAWAL + DRUG-INDUCED
    -- ===================
    (di.icd_version = 9 AND di.icd_code IN (
        '2910',   -- Alcohol withdrawal delirium (347 patients)
        '29181',  -- Alcohol withdrawal (1658 patients)
        '2920',   -- Drug withdrawal syndrome (680 patients)
        '29281'   -- Drug-induced delirium (925 patients)
    ))

    -- ===================
    -- B) ICD-10 WITHDRAWAL (all substances)
    -- ===================

    -- Alcohol withdrawal (F10)
    OR (di.icd_version = 10 AND di.icd_code IN (
        'F10130','F10131','F10132','F10139',  -- Abuse with withdrawal
        'F10230','F10231','F10232','F10239',  -- Dependence with withdrawal
        'F10930','F10931','F10932','F10939'   -- Unspecified use with withdrawal
    ))
    -- Opioid withdrawal (F11)
    OR (di.icd_version = 10 AND di.icd_code IN (
        'F1113','F1123','F1193'
    ))
    -- Cannabis withdrawal (F12)
    OR (di.icd_version = 10 AND di.icd_code IN (
        'F1223','F1293'
    ))
    -- Sedative/hypnotic withdrawal (F13)
    OR (di.icd_version = 10 AND di.icd_code IN (
        'F13130','F13131','F13139',           -- Abuse with withdrawal
        'F13230','F13231','F13232','F13239',  -- Dependence with withdrawal
        'F13930','F13931','F13932','F13939'   -- Unspecified use with withdrawal
    ))
    -- Cocaine withdrawal (F14)
    OR (di.icd_version = 10 AND di.icd_code IN (
        'F1413','F1423','F1493'
    ))
    -- Other stimulant withdrawal (F15)
    OR (di.icd_version = 10 AND di.icd_code IN (
        'F1513','F1523','F1593'
    ))
    -- Other psychoactive substance withdrawal (F19)
    OR (di.icd_version = 10 AND di.icd_code IN (
        'F19139',                             -- Abuse with withdrawal
        'F19230','F19231','F19232','F19239',  -- Dependence with withdrawal
        'F19930','F19931','F19939'            -- Unspecified use with withdrawal
    ))

    -- ===================
    -- C) ICD-10 INTOXICATION WITH DELIRIUM (all substances)
    -- ===================

    -- Alcohol intoxication delirium (F10)
    OR (di.icd_version = 10 AND di.icd_code IN (
        'F10121','F10221','F10921'
    ))
    -- Opioid intoxication delirium (F11)
    OR (di.icd_version = 10 AND di.icd_code IN (
        'F11121','F11221','F11921'
    ))
    -- Cannabis intoxication delirium (F12)
    OR (di.icd_version = 10 AND di.icd_code IN (
        'F12121','F12921'
    ))
    -- Sedative intoxication delirium (F13)
    OR (di.icd_version = 10 AND di.icd_code IN (
        'F13121','F13221','F13921'
    ))
    -- Cocaine intoxication delirium (F14)
    OR (di.icd_version = 10 AND di.icd_code IN (
        'F14121','F14221','F14921'
    ))
    -- Stimulant intoxication delirium (F15)
    OR (di.icd_version = 10 AND di.icd_code IN (
        'F15121','F15221','F15921'
    ))
    -- Hallucinogen intoxication delirium (F16)
    OR (di.icd_version = 10 AND di.icd_code IN (
        'F16121','F16921'
    ))
    -- Other substance intoxication delirium (F19)
    OR (di.icd_version = 10 AND di.icd_code IN (
        'F19121','F19221','F19921'
    ));

-- CHECK:
SELECT COUNT(*) AS substance_delirium_excluded FROM exclude_substance_delirium;


-- =============================================================================
-- STEP 10: Assemble final cohort
-- =============================================================================
-- Joins all filtering results to produce the final table.
-- delirium_label = 1 requires >= 2 positive assessments after 24 hours.
-- delirium_onset_minutes = time of earliest positive assessment after 24h.
-- =============================================================================

-- Earliest delirium onset after 24 hours
DROP TABLE IF EXISTS delirium_onset;
CREATE TEMP TABLE delirium_onset AS
SELECT stay_id, MIN(chart_offset_min) AS onset_offset_min
FROM delirium_assessments
WHERE del_positive = 1
  AND chart_offset_min >= 1440
GROUP BY stay_id;

-- Build the final cohort table
DROP TABLE IF EXISTS delirium_cohort;
CREATE TABLE delirium_cohort AS
SELECT
    c.stay_id,
    c.subject_id,
    c.hadm_id,
    c.intime,
    c.outtime,
    c.los_minutes,
    c.age,
    c.gender,
    -- Delirium label: 1 if >= 2 positive assessments after 24h, else 0
    CASE WHEN pa.stay_id IS NOT NULL THEN 1 ELSE 0 END AS delirium_label,
    -- Onset time in minutes from ICU admission (NULL if no delirium)
    don.onset_offset_min AS delirium_onset_minutes
FROM no_diag_only c
-- Exclude substance-related delirium (Criterion 10)
LEFT JOIN exclude_substance_delirium esd ON c.stay_id = esd.stay_id
-- Delirium-positive requires >= 2 assessments after 24h (Criterion 9)
LEFT JOIN positive_after_24h pa ON c.stay_id = pa.stay_id
-- Onset time
LEFT JOIN delirium_onset don ON c.stay_id = don.stay_id
-- Apply Criterion 10 exclusion
WHERE esd.stay_id IS NULL;


-- =============================================================================
-- VERIFICATION: Run these after Step 10 to check results
-- =============================================================================

-- 1. Overall cohort summary
SELECT
    COUNT(*) AS total_patients,
    SUM(delirium_label) AS delirium_positive,
    COUNT(*) - SUM(delirium_label) AS delirium_negative,
    ROUND(100.0 * SUM(delirium_label) / COUNT(*), 2) AS delirium_rate_pct
FROM delirium_cohort;

-- 2. Step-by-step attrition
SELECT 'Base cohort (age>=18, LOS>=24h, no UTA-only, no dementia-delirium)' AS step,
       COUNT(*) AS n FROM base_cohort
UNION ALL SELECT 'First admission only', COUNT(*) FROM first_admission
UNION ALL SELECT 'Survived 48h', COUNT(*) FROM survived_48h
UNION ALL SELECT 'No early delirium/coma', COUNT(*) FROM no_early_delirium
UNION ALL SELECT 'Has EHR data in 24h', COUNT(*) FROM with_ehr_data
UNION ALL SELECT 'No diagnosis-only cases', COUNT(*) FROM no_diag_only
UNION ALL SELECT 'Final cohort (excl. substance delirium)', COUNT(*) FROM delirium_cohort;

-- 3. How many assessments came from derived components vs overall?
SELECT 'Overall (228332)' AS source,
       COUNT(DISTINCT stay_id) AS stays,
       SUM(del_positive) AS positive_assessments
FROM overall_assessments WHERE del_positive IS NOT NULL
UNION ALL
SELECT 'Derived (components)',
       COUNT(DISTINCT stay_id),
       SUM(del_positive)
FROM derived_assessments WHERE del_positive IS NOT NULL;

-- 4. Age distribution
SELECT
    CASE
        WHEN age < 30 THEN '18-29'
        WHEN age < 40 THEN '30-39'
        WHEN age < 50 THEN '40-49'
        WHEN age < 60 THEN '50-59'
        WHEN age < 70 THEN '60-69'
        WHEN age < 80 THEN '70-79'
        ELSE '80+'
    END AS age_group,
    COUNT(*) AS n,
    SUM(delirium_label) AS delirium_pos,
    ROUND(100.0 * SUM(delirium_label) / COUNT(*), 2) AS del_rate_pct
FROM delirium_cohort
GROUP BY 1 ORDER BY 1;

-- 5. Gender distribution
SELECT gender, COUNT(*) AS n,
    SUM(delirium_label) AS delirium_pos,
    ROUND(100.0 * SUM(delirium_label) / COUNT(*), 2) AS del_rate_pct
FROM delirium_cohort
GROUP BY gender;

