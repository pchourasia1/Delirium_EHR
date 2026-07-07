

SET search_path TO mimiciv_derived, mimiciv_hosp, mimiciv_icu;

-- =============================================================================
-- STEP 1: Medication binary flags (first 24 hours)
-- =============================================================================
DROP TABLE IF EXISTS med_flags;
CREATE TEMP TABLE med_flags AS
SELECT
    c.stay_id,
    MAX(CASE WHEN norepi.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS norepinephrine_24h,
    MAX(CASE WHEN epi.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS epinephrine_24h,
    MAX(CASE WHEN dopa.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS dopamine_24h,
    MAX(CASE WHEN dobu.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS dobutamine_24h,
    MAX(CASE WHEN phenyl.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS phenylephrine_24h,
    MAX(CASE WHEN vaso.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS vasopressin_24h,
    MAX(CASE WHEN milri.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS milrinone_24h,
    MAX(CASE WHEN neuro.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS neuroblock_24h,
    MAX(CASE WHEN abx.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS antibiotic_24h,
    MAX(CASE WHEN nsaid.hadm_id IS NOT NULL THEN 1 ELSE 0 END) AS nsaid_24h,
    MAX(CASE WHEN acei.hadm_id IS NOT NULL THEN 1 ELSE 0 END) AS acei_24h
FROM cohort_with_admitdx c
LEFT JOIN mimiciv_derived.norepinephrine norepi
    ON c.stay_id = norepi.stay_id
    AND norepi.starttime >= c.intime AND norepi.starttime < c.intime + INTERVAL '24 hours'
LEFT JOIN mimiciv_derived.epinephrine epi
    ON c.stay_id = epi.stay_id
    AND epi.starttime >= c.intime AND epi.starttime < c.intime + INTERVAL '24 hours'
LEFT JOIN mimiciv_derived.dopamine dopa
    ON c.stay_id = dopa.stay_id
    AND dopa.starttime >= c.intime AND dopa.starttime < c.intime + INTERVAL '24 hours'
LEFT JOIN mimiciv_derived.dobutamine dobu
    ON c.stay_id = dobu.stay_id
    AND dobu.starttime >= c.intime AND dobu.starttime < c.intime + INTERVAL '24 hours'
LEFT JOIN mimiciv_derived.phenylephrine phenyl
    ON c.stay_id = phenyl.stay_id
    AND phenyl.starttime >= c.intime AND phenyl.starttime < c.intime + INTERVAL '24 hours'
LEFT JOIN mimiciv_derived.vasopressin vaso
    ON c.stay_id = vaso.stay_id
    AND vaso.starttime >= c.intime AND vaso.starttime < c.intime + INTERVAL '24 hours'
LEFT JOIN mimiciv_derived.milrinone milri
    ON c.stay_id = milri.stay_id
    AND milri.starttime >= c.intime AND milri.starttime < c.intime + INTERVAL '24 hours'
LEFT JOIN mimiciv_derived.neuroblock neuro
    ON c.stay_id = neuro.stay_id
    AND neuro.starttime >= c.intime AND neuro.starttime < c.intime + INTERVAL '24 hours'
LEFT JOIN mimiciv_derived.antibiotic abx
    ON c.stay_id = abx.stay_id
    AND abx.starttime >= c.intime AND abx.starttime < c.intime + INTERVAL '24 hours'
LEFT JOIN mimiciv_derived.nsaid nsaid
    ON c.hadm_id = nsaid.hadm_id
    AND nsaid.starttime >= c.intime AND nsaid.starttime < c.intime + INTERVAL '24 hours'
LEFT JOIN mimiciv_derived.acei acei
    ON c.hadm_id = acei.hadm_id
    AND acei.starttime >= c.intime AND acei.starttime < c.intime + INTERVAL '24 hours'
GROUP BY c.stay_id;

-- =============================================================================
-- STEP 2: Treatment binary flags (first 24 hours)
-- =============================================================================
DROP TABLE IF EXISTS treatment_flags;
CREATE TEMP TABLE treatment_flags AS
SELECT
    c.stay_id,
    MAX(CASE WHEN vent.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS ventilation_24h,
    MAX(CASE WHEN vent.ventilation_status = 'InvasiveVent' THEN 1 ELSE 0 END) AS invasive_vent_24h,
    MAX(CASE WHEN rrt.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS rrt_24h,
    MAX(CASE WHEN crrt.stay_id IS NOT NULL THEN 1 ELSE 0 END) AS crrt_24h
FROM cohort_with_admitdx c
LEFT JOIN mimiciv_derived.ventilation vent
    ON c.stay_id = vent.stay_id
    AND vent.starttime >= c.intime AND vent.starttime < c.intime + INTERVAL '24 hours'
LEFT JOIN mimiciv_derived.rrt rrt
    ON c.stay_id = rrt.stay_id
    AND rrt.charttime >= c.intime AND rrt.charttime < c.intime + INTERVAL '24 hours'
LEFT JOIN mimiciv_derived.crrt crrt
    ON c.stay_id = crrt.stay_id
    AND crrt.charttime >= c.intime AND crrt.charttime < c.intime + INTERVAL '24 hours'
GROUP BY c.stay_id;

-- =============================================================================
-- STEP 3: Full feature table
-- =============================================================================
DROP TABLE IF EXISTS cohort_all_features;
CREATE TABLE cohort_all_features AS
SELECT
    -- Base cohort
    c.*,

    -- Admissions info (not in icustay_detail)
    adm.admission_type,
    adm.admission_location,
    adm.discharge_location,

    -- Demographics (icustay_detail)
    detail.admission_age,
    detail.race,
    detail.los_hospital,
    detail.los_icu,
    detail.hospital_expire_flag AS detail_hospital_expire_flag,
    detail.hospstay_seq,
    detail.icustay_seq,
    detail.first_hosp_stay,
    detail.first_icu_stay,
    detail.dod,
    detail.admittime,
    detail.dischtime,
    detail.icu_intime,
    detail.icu_outtime,

    -- Age
    age.age AS age_calculated,
    age.anchor_age,
    age.anchor_year,

    -- First day vitals
    vs.heart_rate_min, vs.heart_rate_max, vs.heart_rate_mean,
    vs.sbp_min, vs.sbp_max, vs.sbp_mean,
    vs.dbp_min, vs.dbp_max, vs.dbp_mean,
    vs.mbp_min, vs.mbp_max, vs.mbp_mean,
    vs.resp_rate_min, vs.resp_rate_max, vs.resp_rate_mean,
    vs.temperature_min, vs.temperature_max, vs.temperature_mean,
    vs.spo2_min, vs.spo2_max, vs.spo2_mean,
    vs.glucose_min AS glucose_vital_min,
    vs.glucose_max AS glucose_vital_max,
    vs.glucose_mean AS glucose_vital_mean,

    -- First day GCS
    gcs.gcs_min, gcs.gcs_motor, gcs.gcs_verbal, gcs.gcs_eyes, gcs.gcs_unable,

    -- First day labs
    lab.hematocrit_min, lab.hematocrit_max,
    lab.hemoglobin_min, lab.hemoglobin_max,
    lab.platelets_min, lab.platelets_max,
    lab.wbc_min, lab.wbc_max,
    lab.albumin_min, lab.albumin_max,
    lab.globulin_min, lab.globulin_max,
    lab.total_protein_min, lab.total_protein_max,
    lab.aniongap_min, lab.aniongap_max,
    lab.bicarbonate_min, lab.bicarbonate_max,
    lab.bun_min, lab.bun_max,
    lab.calcium_min, lab.calcium_max,
    lab.chloride_min, lab.chloride_max,
    lab.creatinine_min, lab.creatinine_max,
    lab.glucose_min AS glucose_lab_min, lab.glucose_max AS glucose_lab_max,
    lab.sodium_min, lab.sodium_max,
    lab.potassium_min, lab.potassium_max,
    lab.abs_basophils_min, lab.abs_basophils_max,
    lab.abs_eosinophils_min, lab.abs_eosinophils_max,
    lab.abs_lymphocytes_min, lab.abs_lymphocytes_max,
    lab.abs_monocytes_min, lab.abs_monocytes_max,
    lab.abs_neutrophils_min, lab.abs_neutrophils_max,
    lab.atyps_min, lab.atyps_max,
    lab.bands_min, lab.bands_max,
    lab.imm_granulocytes_min, lab.imm_granulocytes_max,
    lab.metas_min, lab.metas_max,
    lab.nrbc_min, lab.nrbc_max,
    lab.d_dimer_min, lab.d_dimer_max,
    lab.fibrinogen_min, lab.fibrinogen_max,
    lab.thrombin_min, lab.thrombin_max,
    lab.inr_min, lab.inr_max,
    lab.pt_min, lab.pt_max,
    lab.ptt_min, lab.ptt_max,
    lab.alt_min, lab.alt_max,
    lab.alp_min, lab.alp_max,
    lab.ast_min, lab.ast_max,
    lab.amylase_min, lab.amylase_max,
    lab.bilirubin_total_min, lab.bilirubin_total_max,
    lab.bilirubin_direct_min, lab.bilirubin_direct_max,
    lab.bilirubin_indirect_min, lab.bilirubin_indirect_max,
    lab.ck_cpk_min, lab.ck_cpk_max,
    lab.ck_mb_min, lab.ck_mb_max,
    lab.ggt_min, lab.ggt_max,
    lab.ld_ldh_min, lab.ld_ldh_max,

    -- First day blood gas
    bg.lactate_min AS bg_lactate_min, bg.lactate_max AS bg_lactate_max,
    bg.ph_min AS bg_ph_min, bg.ph_max AS bg_ph_max,
    bg.so2_min AS bg_so2_min, bg.so2_max AS bg_so2_max,
    bg.po2_min AS bg_po2_min, bg.po2_max AS bg_po2_max,
    bg.pco2_min AS bg_pco2_min, bg.pco2_max AS bg_pco2_max,
    bg.aado2_min AS bg_aado2_min, bg.aado2_max AS bg_aado2_max,
    bg.aado2_calc_min AS bg_aado2_calc_min, bg.aado2_calc_max AS bg_aado2_calc_max,
    bg.pao2fio2ratio_min AS bg_pao2fio2ratio_min, bg.pao2fio2ratio_max AS bg_pao2fio2ratio_max,
    bg.baseexcess_min AS bg_baseexcess_min, bg.baseexcess_max AS bg_baseexcess_max,
    bg.bicarbonate_min AS bg_bicarbonate_min, bg.bicarbonate_max AS bg_bicarbonate_max,
    bg.totalco2_min AS bg_totalco2_min, bg.totalco2_max AS bg_totalco2_max,
    bg.hematocrit_min AS bg_hematocrit_min, bg.hematocrit_max AS bg_hematocrit_max,
    bg.hemoglobin_min AS bg_hemoglobin_min, bg.hemoglobin_max AS bg_hemoglobin_max,
    bg.carboxyhemoglobin_min AS bg_carboxyhemoglobin_min, bg.carboxyhemoglobin_max AS bg_carboxyhemoglobin_max,
    bg.methemoglobin_min AS bg_methemoglobin_min, bg.methemoglobin_max AS bg_methemoglobin_max,
    bg.temperature_min AS bg_temperature_min, bg.temperature_max AS bg_temperature_max,
    bg.chloride_min AS bg_chloride_min, bg.chloride_max AS bg_chloride_max,
    bg.calcium_min AS bg_calcium_min, bg.calcium_max AS bg_calcium_max,
    bg.glucose_min AS bg_glucose_min, bg.glucose_max AS bg_glucose_max,
    bg.potassium_min AS bg_potassium_min, bg.potassium_max AS bg_potassium_max,
    bg.sodium_min AS bg_sodium_min, bg.sodium_max AS bg_sodium_max,

    -- First day arterial blood gas
    bga.lactate_min AS bga_lactate_min, bga.lactate_max AS bga_lactate_max,
    bga.ph_min AS bga_ph_min, bga.ph_max AS bga_ph_max,
    bga.so2_min AS bga_so2_min, bga.so2_max AS bga_so2_max,
    bga.po2_min AS bga_po2_min, bga.po2_max AS bga_po2_max,
    bga.pco2_min AS bga_pco2_min, bga.pco2_max AS bga_pco2_max,
    bga.aado2_min AS bga_aado2_min, bga.aado2_max AS bga_aado2_max,
    bga.aado2_calc_min AS bga_aado2_calc_min, bga.aado2_calc_max AS bga_aado2_calc_max,
    bga.pao2fio2ratio_min AS bga_pao2fio2ratio_min, bga.pao2fio2ratio_max AS bga_pao2fio2ratio_max,
    bga.baseexcess_min AS bga_baseexcess_min, bga.baseexcess_max AS bga_baseexcess_max,
    bga.bicarbonate_min AS bga_bicarbonate_min, bga.bicarbonate_max AS bga_bicarbonate_max,
    bga.totalco2_min AS bga_totalco2_min, bga.totalco2_max AS bga_totalco2_max,
    bga.hematocrit_min AS bga_hematocrit_min, bga.hematocrit_max AS bga_hematocrit_max,
    bga.hemoglobin_min AS bga_hemoglobin_min, bga.hemoglobin_max AS bga_hemoglobin_max,
    bga.carboxyhemoglobin_min AS bga_carboxyhemoglobin_min, bga.carboxyhemoglobin_max AS bga_carboxyhemoglobin_max,
    bga.methemoglobin_min AS bga_methemoglobin_min, bga.methemoglobin_max AS bga_methemoglobin_max,
    bga.temperature_min AS bga_temperature_min, bga.temperature_max AS bga_temperature_max,
    bga.chloride_min AS bga_chloride_min, bga.chloride_max AS bga_chloride_max,
    bga.calcium_min AS bga_calcium_min, bga.calcium_max AS bga_calcium_max,
    bga.glucose_min AS bga_glucose_min, bga.glucose_max AS bga_glucose_max,
    bga.potassium_min AS bga_potassium_min, bga.potassium_max AS bga_potassium_max,
    bga.sodium_min AS bga_sodium_min, bga.sodium_max AS bga_sodium_max,

    -- First day height and weight
    ht.height,
    wt.weight, wt.weight_admit, wt.weight_min, wt.weight_max,

    -- First day urine output
    uo.urineoutput,

    -- First day RRT
    fdrrt.dialysis_present, fdrrt.dialysis_active, fdrrt.dialysis_type,

    -- First day SOFA (with sub-scores)
    fds.sofa AS sofa_24h,
    fds.respiration AS sofa_resp,
    fds.coagulation AS sofa_coag,
    fds.liver AS sofa_liver,
    fds.cardiovascular AS sofa_cardio,
    fds.cns AS sofa_cns,
    fds.renal AS sofa_renal,

    -- SAPS II
    saps.sapsii, saps.sapsii_prob,
    saps.age_score AS sapsii_age_score,
    saps.hr_score AS sapsii_hr_score,
    saps.sysbp_score AS sapsii_sysbp_score,
    saps.temp_score AS sapsii_temp_score,
    saps.pao2fio2_score AS sapsii_pao2fio2_score,
    saps.uo_score AS sapsii_uo_score,
    saps.bun_score AS sapsii_bun_score,
    saps.wbc_score AS sapsii_wbc_score,
    saps.potassium_score AS sapsii_potassium_score,
    saps.sodium_score AS sapsii_sodium_score,
    saps.bicarbonate_score AS sapsii_bicarbonate_score,
    saps.bilirubin_score AS sapsii_bilirubin_score,
    saps.gcs_score AS sapsii_gcs_score,
    saps.comorbidity_score AS sapsii_comorbidity_score,
    saps.admissiontype_score AS sapsii_admissiontype_score,

    -- OASIS
    oasis.oasis, oasis.oasis_prob,
    oasis.age AS oasis_age, oasis.age_score AS oasis_age_score,
    oasis.preiculos AS oasis_preiculos, oasis.preiculos_score AS oasis_preiculos_score,
    oasis.gcs AS oasis_gcs, oasis.gcs_score AS oasis_gcs_score,
    oasis.heartrate AS oasis_heartrate, oasis.heart_rate_score AS oasis_heart_rate_score,
    oasis.meanbp AS oasis_meanbp, oasis.mbp_score AS oasis_mbp_score,
    oasis.resprate AS oasis_resprate, oasis.resp_rate_score AS oasis_resp_rate_score,
    oasis.temp AS oasis_temp, oasis.temp_score AS oasis_temp_score,
    oasis.urineoutput AS oasis_urineoutput, oasis.urineoutput_score AS oasis_urineoutput_score,
    oasis.mechvent AS oasis_mechvent, oasis.mechvent_score AS oasis_mechvent_score,
    oasis.electivesurgery AS oasis_electivesurgery, oasis.electivesurgery_score AS oasis_electivesurgery_score,

    -- APS III
    apsiii.apsiii, apsiii.apsiii_prob,
    apsiii.hr_score AS apsiii_hr_score,
    apsiii.mbp_score AS apsiii_mbp_score,
    apsiii.temp_score AS apsiii_temp_score,
    apsiii.resp_rate_score AS apsiii_resp_rate_score,
    apsiii.pao2_aado2_score AS apsiii_pao2_aado2_score,
    apsiii.hematocrit_score AS apsiii_hematocrit_score,
    apsiii.wbc_score AS apsiii_wbc_score,
    apsiii.creatinine_score AS apsiii_creatinine_score,
    apsiii.uo_score AS apsiii_uo_score,
    apsiii.bun_score AS apsiii_bun_score,
    apsiii.sodium_score AS apsiii_sodium_score,
    apsiii.albumin_score AS apsiii_albumin_score,
    apsiii.bilirubin_score AS apsiii_bilirubin_score,
    apsiii.glucose_score AS apsiii_glucose_score,
    apsiii.acidbase_score AS apsiii_acidbase_score,
    apsiii.gcs_score AS apsiii_gcs_score,

    -- LODS
    lods.lods,
    lods.neurologic AS lods_neurologic,
    lods.cardiovascular AS lods_cardiovascular,
    lods.renal AS lods_renal,
    lods.pulmonary AS lods_pulmonary,
    lods.hematologic AS lods_hematologic,
    lods.hepatic AS lods_hepatic,

    -- SIRS
    sirs.sirs,
    sirs.temp_score AS sirs_temp_score,
    sirs.heart_rate_score AS sirs_heart_rate_score,
    sirs.resp_score AS sirs_resp_score,
    sirs.wbc_score AS sirs_wbc_score,

    -- Charlson comorbidity
    charlson.charlson_comorbidity_index,
    charlson.age_score AS charlson_age_score,
    charlson.myocardial_infarct AS charlson_mi,
    charlson.congestive_heart_failure AS charlson_chf,
    charlson.peripheral_vascular_disease AS charlson_pvd,
    charlson.cerebrovascular_disease AS charlson_cvd,
    charlson.dementia AS charlson_dementia,
    charlson.chronic_pulmonary_disease AS charlson_cpd,
    charlson.rheumatic_disease AS charlson_rheumatic,
    charlson.peptic_ulcer_disease AS charlson_pud,
    charlson.mild_liver_disease AS charlson_mild_liver,
    charlson.diabetes_without_cc AS charlson_diabetes,
    charlson.diabetes_with_cc AS charlson_diabetes_cc,
    charlson.paraplegia AS charlson_paraplegia,
    charlson.renal_disease AS charlson_renal,
    charlson.malignant_cancer AS charlson_cancer,
    charlson.severe_liver_disease AS charlson_severe_liver,
    charlson.metastatic_solid_tumor AS charlson_metastatic,
    charlson.aids AS charlson_aids,

    -- MELD
    meld.meld, meld.meld_initial,
    meld.rrt AS meld_rrt,
    meld.creatinine_max AS meld_creatinine_max,
    meld.bilirubin_total_max AS meld_bilirubin_total_max,
    meld.inr_max AS meld_inr_max,
    meld.sodium_min AS meld_sodium_min,

    -- Sepsis-3
    CASE WHEN sep.stay_id IS NOT NULL THEN 1 ELSE 0 END AS sepsis3,
    sep.sofa_score AS sepsis3_sofa_score,

    -- Medication flags
    mf.norepinephrine_24h,
    mf.epinephrine_24h,
    mf.dopamine_24h,
    mf.dobutamine_24h,
    mf.phenylephrine_24h,
    mf.vasopressin_24h,
    mf.milrinone_24h,
    mf.neuroblock_24h,
    mf.antibiotic_24h,
    mf.nsaid_24h,
    mf.acei_24h,

    -- Treatment flags
    tf.ventilation_24h,
    tf.invasive_vent_24h,
    tf.rrt_24h,
    tf.crrt_24h

FROM cohort_with_admitdx c

LEFT JOIN mimiciv_hosp.admissions adm ON c.hadm_id = adm.hadm_id
LEFT JOIN mimiciv_derived.icustay_detail detail ON c.stay_id = detail.stay_id
LEFT JOIN mimiciv_derived.age age ON c.hadm_id = age.hadm_id
LEFT JOIN mimiciv_derived.first_day_vitalsign vs ON c.stay_id = vs.stay_id
LEFT JOIN mimiciv_derived.first_day_gcs gcs ON c.stay_id = gcs.stay_id
LEFT JOIN mimiciv_derived.first_day_lab lab ON c.stay_id = lab.stay_id
LEFT JOIN mimiciv_derived.first_day_bg bg ON c.stay_id = bg.stay_id
LEFT JOIN mimiciv_derived.first_day_bg_art bga ON c.stay_id = bga.stay_id
LEFT JOIN mimiciv_derived.first_day_height ht ON c.stay_id = ht.stay_id
LEFT JOIN mimiciv_derived.first_day_weight wt ON c.stay_id = wt.stay_id
LEFT JOIN mimiciv_derived.first_day_urine_output uo ON c.stay_id = uo.stay_id
LEFT JOIN mimiciv_derived.first_day_rrt fdrrt ON c.stay_id = fdrrt.stay_id
LEFT JOIN mimiciv_derived.first_day_sofa fds ON c.stay_id = fds.stay_id
LEFT JOIN mimiciv_derived.sapsii saps ON c.stay_id = saps.stay_id
LEFT JOIN mimiciv_derived.oasis oasis ON c.stay_id = oasis.stay_id
LEFT JOIN mimiciv_derived.apsiii apsiii ON c.stay_id = apsiii.stay_id
LEFT JOIN mimiciv_derived.lods lods ON c.stay_id = lods.stay_id
LEFT JOIN mimiciv_derived.sirs sirs ON c.stay_id = sirs.stay_id
LEFT JOIN mimiciv_derived.charlson charlson ON c.hadm_id = charlson.hadm_id
LEFT JOIN mimiciv_derived.meld meld ON c.stay_id = meld.stay_id
LEFT JOIN mimiciv_derived.sepsis3 sep ON c.stay_id = sep.stay_id
LEFT JOIN med_flags mf ON c.stay_id = mf.stay_id
LEFT JOIN treatment_flags tf ON c.stay_id = tf.stay_id;

-- =============================================================================
-- VERIFICATION
-- =============================================================================
SELECT COUNT(*) AS total_rows FROM cohort_all_features;

SELECT
    COUNT(*) AS total,
    COUNT(heart_rate_mean) AS has_vitals,
    COUNT(hemoglobin_min) AS has_labs,
    COUNT(gcs_min) AS has_gcs,
    COUNT(sofa_24h) AS has_sofa,
    COUNT(sapsii) AS has_sapsii,
    COUNT(oasis) AS has_oasis,
    COUNT(apsiii) AS has_apsiii,
    COUNT(height) AS has_height,
    COUNT(weight) AS has_weight,
    COUNT(urineoutput) AS has_urine,
    COUNT(charlson_comorbidity_index) AS has_charlson,
    COUNT(meld) AS has_meld,
    SUM(ventilation_24h) AS n_ventilated,
    SUM(rrt_24h) AS n_dialysis,
    SUM(sepsis3) AS n_sepsis,
    SUM(norepinephrine_24h) AS n_norepinephrine,
    SUM(antibiotic_24h) AS n_antibiotic
FROM cohort_all_features;
