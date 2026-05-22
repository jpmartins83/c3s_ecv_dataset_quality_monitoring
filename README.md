# c3s_ecv_dataset_quality_monitoring
Stores scripts to inspect Satellite ECV datasets for quality monitoring


Some guidelines provided by AI. We use use these guidelines as inspiration. 

For satellite-based climate data records (CDRs), the inspection process benefits from a layered approach that catches problems at different stages: instrument-level issues, processing errors, geophysical inconsistencies, and long-term climate artifacts. A useful mindset is: *“detect failures before users discover them.”*

## 1. Build a staged validation pipeline

Instead of one final quality check before publication, divide inspection into four stages:

**Stage A — Input / upstream checks**

* Verify sensor inputs are complete and expected:

  * Missing granules/files
  * Time ordering
  * Orbit gaps
  * Duplicates
  * Ancillary data availability (NWP, DEM, calibration files)
* Validate metadata:

  * Units
  * Coordinate systems
  * timestamps
  * version tags
  * processing lineage

Typical outputs:

* completeness score
* missing-data report
* ingestion anomaly report

---

**Stage B — Product integrity checks**
Detect processing failures.

Examples:

* Physical range tests

  * SST: plausible limits
  * albedo: 0–1
  * cloud fraction: 0–1
* Distribution checks:

  * daily histogram compared with climatology
* Spatial consistency:

  * abrupt stripes
  * scan-line artifacts
  * orbit seams
* Temporal consistency:

  * unrealistic jumps between successive periods

Metrics:

* mean
* standard deviation
* quantiles (1%, 5%, 95%, 99%)
* skewness
* missing-value percentage

Use automatic thresholds but avoid hard-coded values everywhere.

---

**Stage C — Scientific consistency checks**
This is where operational systems often miss subtle issues.

Compare against:

**1. Independent observations**

* Surface networks
* Buoys
* Radiosondes
* Aircraft
* Reanalysis
* Other satellites

Examples:

* SST ↔ drifting buoys
* atmospheric profiles ↔ radiosondes
* precipitation ↔ gauges

Evaluate:

* Bias
* RMSE
* Median absolute deviation
* Correlation
* Trend agreement

---

**2. Cross-sensor consistency**

When moving between missions:

Sensor A → Sensor B → Sensor C

Inspect:

* overlap period bias
* seasonal dependence
* latitude dependence
* viewing-angle dependence

Many climate record failures appear only at transition points.

---

**Stage D — Climate stability checks**
Operational products can look good daily while being poor climate records.

Inspect:

### Trend stability

Compute:

* monthly anomalies
* seasonal anomalies
* decadal trends

Look for:

* step changes
* trend discontinuities
* drift

---

### Change-point detection

Methods:

* Pettitt test
* CUSUM
* Bayesian change-point methods

Check whether detected breaks align with:

* satellite replacement
* calibration updates
* algorithm changes
* orbital drift

Unexpected breakpoints are high-priority investigations.

---

## 2. Build a standard inspection dashboard

For every product version, automatically generate:

**Global maps**

* mean field
* anomaly field
* uncertainty field
* missing-data fraction

**Time series**

* global mean
* zonal mean
* regional means
* percent missing
* uncertainty evolution

**Distributions**

* histograms
* PDFs
* percentiles

**Difference products**

* current version − previous operational version
* current version − reference dataset

---

## 3. Add anomaly detection beyond fixed rules

Fixed thresholds miss unusual patterns.

Useful methods:

**Unsupervised methods**

* Isolation Forest
* PCA reconstruction error
* autoencoders
* clustering-based outlier detection

Examples:

* unexpected orbital striping
* calibration drifts
* regional artifacts

Train on historical “good” data.

Flag:

> "Today's global structure differs from the historical distribution."

instead of:

> "Variable exceeded threshold."

---

## 4. Treat uncertainty as a first-class product

Operational publication should inspect uncertainty estimates themselves.

Questions:

* Does uncertainty increase near edges/swaths?
* Does uncertainty behave seasonally?
* Are uncertainty estimates correlated with actual errors?
* Is uncertainty unrealistically stable?

Check calibration:

Expected:

95% confidence interval → ~95% coverage

If not, uncertainty estimates may be misleading.

---

## 5. Introduce release gates

Move from "inspection" to "go/no-go" criteria.

Example:

| Category                             |         Pass criterion |
| ------------------------------------ | ---------------------: |
| Input completeness                   |                 >99.9% |
| Missing pixels                       |                    <1% |
| Bias vs reference                    | < predefined tolerance |
| Trend difference vs previous release |            < threshold |
| Breakpoint detection                 |       none unexplained |
| Metadata validation                  |                   100% |

Release only when all critical checks pass.

---

## 6. Keep “golden scenes”

Create a library of canonical test cases:

Examples:

* Sahara desert
* Antarctic ice
* tropical convection region
* clear-ocean region
* persistent cloud systems
* volcanic events
* high-latitude winter scenes

Inspect them every processing run.

These become regression tests for algorithm changes.

---

## 7. Maintain full provenance

Every published record should preserve:

* algorithm version
* calibration tables
* ancillary sources
* processing software hash
* configuration parameters
* processing date

This makes anomalies traceable months later.

---

## 8. Suggested operational architecture

```text
Raw observations
       ↓
Input QC
       ↓
Geophysical retrieval
       ↓
Product QC
       ↓
Cross-reference validation
       ↓
Climate stability tests
       ↓
Automated dashboard
       ↓
Release gate
       ↓
Operational publication
```

---

For climate data records specifically, I would allocate effort roughly as:

* 25% input and integrity checks
* 35% scientific validation
* 30% long-term climate stability analysis
* 10% visual/manual review

Many operational systems over-invest in daily QC and under-invest in long-term stability. Climate users usually notice discontinuities, drifts, and version-induced shifts long after publication.

If you tell me the specific product type (SST, cloud properties, aerosols, sea ice, greenhouse gases, precipitation, land surface temperature, etc.), I can suggest a more tailored inspection framework and concrete metrics.
