# Additional training datasets for StudentInsight

Reviewed on 27 October 2026. This catalog records dataset discovery and file inspection. It does not report new trained models or improved prediction accuracy.

## Recommended sources

| Priority | Dataset | Relevant coverage | Suitable next task | Access and status |
| --- | --- | --- | --- | --- |
| 1 | [Smart Learning Partner (SLP)](https://aic-fe.bnu.edu.cn/en/data/index.html) | Mathematics, English, Physics, Biology, History and Geography term tests; Chemistry unit tests | Subject-specific assessment forecasting after preparing and validating exam totals | Term-test and unit-test archives downloaded and inspected. Author repository specifies research use and source attribution. |
| 2 | [Early prediction in CS1 programming courses](https://zenodo.org/records/8111596) | Introductory programming | Predict performance category from early laboratory work | CSV downloaded and inspected; CC BY 4.0 according to the publisher's record metadata. |
| 3 | [Open University Learning Analytics Dataset (OULAD)](https://research.stem.open.ac.uk/ouanalyse/open-dataset-more/) | Seven anonymized university modules, assessments and online activity | Early warning of failure or withdrawal, with a defined observation cutoff | Official schema reviewed; full archive not downloaded. CC BY 4.0. |
| Existing candidate | [Halabja Student Grade Dataset, 2018–2023](https://zenodo.org/records/16623461) | Annual grades across multiple school subjects | Predict a later year's subject result using earlier years | Local file already existed before this review. Annual grades need a separate longitudinal preparation pipeline. |

SLP is the closest subject match. Start with its Mathematics and Physics term-test files: the initial score-range checks found no invalid scores in those two files. This is a preparation priority, not a claim that either file is fully clean or that its model will outperform the current system.

## SLP: inspected subject coverage

Download source: [author repository](https://github.com/yangyangyang-ailab/SLP-dataset). The downloaded files are `dataset/termtestRecord.zip` and `dataset/unittestRecord.zip`. The university describes eight subjects, including Chinese. Chinese is outside the app's current subject list.

Counts below were computed from the downloaded archives. A question row is one recorded answer, not one independent student. Student counts overlap between subjects and must not be added together.

| App subject | File used for this count | Question rows | Distinct student IDs | Distinct exam IDs | Student–exam pairs |
| --- | --- | ---: | ---: | ---: | ---: |
| Mathematics | Term tests | 582,560 | 4,668 | 10 | 25,445 |
| English | Term tests | 952,128 | 4,669 | 9 | 25,420 |
| Physics | Term tests | 463,913 | 4,339 | 3 | 12,101 |
| Biology | Term tests | 500,562 | 4,448 | 8 | 20,825 |
| Geography | Term tests | 498,042 | 4,352 | 8 | 20,734 |
| History | Term tests | 735,456 | 4,652 | 9 | 24,902 |
| Chemistry | Unit tests only | 30,056 | 624 | 140 | 2,496 |

Chemistry has no term-test file in the inspected archive. Its unit tests support a different outcome, such as performance on a later unit test; they do not supply an observed final-term grade. SLP contains no Computer Science file.

### Preparation findings

- Headers declare eight fields, but the downloaded records have an additional empty trailing field. A default pandas import can shift the columns by treating the student ID as an index. Inspection explicitly assigned a ninth trailing field and checked that it was empty. English row and student counts were independently checked with Python's CSV reader.
- Fields include student, subject, exam and question identifiers, concept, score, full score and access time. Aggregate valid question scores within each student and exam only after verifying question coverage, duplicates and assessment denominators.
- Initial numeric checks flagged 39 English, 68 Biology, 17,143 Geography, 60,408 History and 400 Chemistry rows. A flag means a missing/non-numeric score or denominator, a nonpositive denominator, a negative score, or a score above its denominator. These records require investigation; they were not silently removed or replaced.
- Geography has 8,598 missing access timestamps and History has 60,382 in the term files. The source defines `time_access` as system access time. It is not yet established as the date the examination occurred. Do not infer exam order from CSV order or identifier sorting.
- Keep each subject separate. Establish which assessments were available at prediction time and which later assessment is the outcome. Keep repeated students together during model validation and test on later assessments/cohorts where the dates support that design.
- Preserve fractional scores. Converting an exam total to a percentage or a 0–20 scale can produce decimals; the current app accepts whole-number grades only. A documented continuous-grade workflow is needed before importing these records.

No family, psychometric or demographic tables were downloaded. They are unnecessary for this initial grade-history assessment. The repository's research-use terms and requested citation should accompany any experiment; this review does not establish permission for unrestricted redistribution or production use.

Citation requested by the source: Lu, Y., Pian, Y., Shen, Z., Chen, P., and Li, X. (2021). *SLP: A Multi-Dimensional and Consecutive Dataset from K-12 Education*. ICCE, volume 1, pp. 261–266. [University description and paper link](https://aic-fe.bnu.edu.cn/en/data/index.html).

## Computer Science: CS1 programming dataset

Source: Llanos, J., Bucheli, V., and Restrepo, F. (2023), [Zenodo record 8111596](https://doi.org/10.5281/zenodo.8111596). The downloaded `classification_data.csv` matches the publisher's MD5 checksum `5ec48f4b9a5fae45af41d6e33b1cc680`.

The semicolon-delimited file has **468 rows and six columns**: `lab_1`, `delivery_time_lab_1`, `attempts_lab_1`, `lab_2`, `lab_3`, and `grade`. No missing cells were found. The three laboratory grades range from 0 to 5. The target contains category codes 0, 1 and 2, with 162, 200 and 106 records respectively.

The released CSV has no student or cohort identifier and no continuous final-grade column. It can support a separate classification experiment after verifying category definitions against the [associated paper](https://doi.org/10.7717/peerj-cs.1655). Include only labs completed before the chosen prediction week. It cannot be inserted into the current final-grade regressor merely by renaming `grade` to `G3`. Missing identifiers limit checks for student overlap and validation on later cohorts.

## OULAD: engagement and early warning

The [official schema](https://research.stem.open.ac.uk/ouanalyse/open-dataset-more/) links courses, assessments, student results, registration and dated online interactions. Assessment scores use a 0–100 scale; some final-exam results are absent. Module identifiers are anonymized, so they cannot be relabeled as the app's named school subjects.

This is a candidate for a separate course-outcome classifier. Choose an observation day, then restrict assessment submissions and activity logs to information available by that day. Evaluate on later course presentations and check repeated-student overlap. Future unregistration dates and `final_result` must not become input features. [Official download page](https://research.stem.open.ac.uk/ouanalyse/dataset/). License: CC BY 4.0.

## Halabja: existing annual-grade candidate

The local `data/halabja-grades.csv` was preserved. It contains 16,823 data rows and subject columns including English, Math, Physic, Chemistry, Biology, Computer, History and Geography. The [publisher](https://zenodo.org/records/16623461) describes annual records from 2018–2019 through 2022–2023 and reports 16,824 entries. The header is excluded from our row count.

These are annual subject marks rather than three periods within one school year. A later-year forecasting model needs verified year ordering, duplicate student/year resolution and a consistent grade scale. A field named `Computer` also needs curriculum review before it is presented as Computer Science. This source does not resolve the missing final-term Chemistry outcome in SLP by joining unrelated students.

## What changes before training

The current upload path requires 50–5,000 distinct student records for one subject, completed `G3`, and the selected earlier-grade inputs on a whole-number 0–20 scale. None of the newly inspected raw datasets is a direct upload into that schema.

1. Prepare SLP Mathematics and Physics assessment histories first, with an explicit assessment timeline and completeness rules. Extend the same checks to English, Biology, Geography and History.
2. Support continuous assessment grades without rounding source values merely to pass validation. Record the original scale and any deterministic conversion.
3. Keep Chemistry unit-test prediction, CS1 classification, OULAD early warning and Halabja annual forecasting as distinct targets with suitable inputs and evaluation.
4. Compare candidates with a last-observed-grade baseline on identical held-out records. Report MAE/RMSE for numeric outcomes and class-specific precision/recall/F1 for categories. More question rows alone do not establish improved accuracy.

The raw SLP archives and CS1 CSV are stored locally under `.local/dataset-research/2026-10-27/`, outside Git. Download addresses, SHA-256 hashes and aggregate inspection results are recorded in [dataset-inspection.json](dataset-inspection.json). No active model or training artifact was replaced by this research.
