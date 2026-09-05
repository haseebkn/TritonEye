# Newfoundland and Labrador data sources

Public-source access checked on 2026-09-04. Source availability is not evidence
that an artifact has been downloaded, that AIS reception is complete, or that a
dataset labels the acquisition under evaluation. Record the actual downloaded
URL, source version/valid time, retrieval time, SHA-256 and licence in each
mission manifest. Restrict observations to the versioned NL study area.

## Satellite imagery and AIS

The pinned detector is the public
[BloodAxe xView3 first-place solution](https://github.com/BloodAxe/xView3-The-First-Place-Solution),
distributed under that repository's MIT licence. The downloaded TorchScript
release is a third-party artifact, not a model trained by this project. Preserve
its attribution and review upstream terms before redistribution. The wrapper's
per-tile decoding differs from upstream heatmap stitching; no competition score
is claimed. `python -m agents.assets` verifies the configured release digest.

Coastline options are [OSM land polygons](https://osmdata.openstreetmap.de/data/land-polygons.html)
(ODbL, © OpenStreetMap contributors) and
[GSHHG 2.3.7](https://www.soest.hawaii.edu/pwessel/gshhg/) (GNU LGPL, not public
domain despite some public-domain source inputs). Neither is a navigation chart
or a substitute for independent regional geolocation validation.

The pipeline's Sentinel-1 source is the
[Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/).
Download uses the configured Copernicus credentials; keep them in an untracked
environment file, never in reports or Git. A complete source product should
include measurement rasters, geolocation metadata, polarization and calibration
annotations. Acquisitions may extend beyond the study area; analysis outputs
must be spatially restricted.

[AISStream](https://aisstream.io/) is a live AIS source used by the recorder.
It requires an API key and does not retrospectively create observations for
periods before recording began. Local daily archives are only the messages the
recorder actually received. Geographic subscription bounds, recorder uptime,
observation timestamp and data gaps are essential provenance. Provider access
does not establish guaranteed offshore coverage; do not advertise archival AIS
as globally complete. Historical NL vessel observations coincident with a chosen
SAR pass may require additional licensed data or a locally recorded archive.

## Public ice information

| Source | Public access checked | Appropriate use and limitations |
| --- | --- | --- |
| [Canadian Ice Service archive](https://iceweb1.cis.ec.gc.ca/Archive/page1.xhtml) | Public date/region search page. [ECCC overview](https://www.canada.ca/en/environment-climate-change/services/ice-forecasts-observations/latest-conditions/archive-overview.html) documents daily iceberg GIFs and weekly regional charts in GIF/Shapefile/E00. | Select the chart's valid time and Newfoundland/Labrador region. Visual or regional context, not per-target vessel/iceberg labels. Historical downloadable bytes were not verified for a particular acquisition in this source audit. |
| [CIS chart descriptions](https://www.canada.ca/en/environment-climate-change/services/ice-forecasts-observations/latest-conditions/products-guides/chart-descriptions.html) | Public descriptions of daily and regional products. | Charts combine observations and analyst interpretation at their valid time. Daily ice estimates and iceberg products have different semantics; sea-ice absence does not imply iceberg absence. |
| [NAIS/IIP products](https://navcen.uscg.gov/north-american-ice-service-products) | Public page links current charts, bulletins, KML and [current sea-ice/iceberg-limit shapefile](https://navcen.uscg.gov/sites/default/files/iip/shape/currentShape.zip). | The current file is mutable. Archive the downloaded file with its issue/valid date. These are limits, not individual object labels; do not apply today's limit to a historical SAR pass. The binary download link was identified, not validated as a matching historical chart. |
| [NAVCEN chart archive](https://navcen.uscg.gov/archives) | Public archive interface lists ice charts from 2009 onward, subject to retained files. | Use a date-matched chart; an absent archive file is missing evidence, not ice-free water. |
| [NSIDC/IIP iceberg sightings, G00807 v1](https://nsidc.org/data/g00807/versions/1) | Public catalogue and [user guide](https://nsidc.org/sites/default/files/g00807-v001-userguide_1_1.pdf); catalogue lists temporal coverage through 2021-09-29 at access time. | Dated locations, sighting method, size and shape can help construct a historical NL ice challenge set. They are not complete SAR object annotations. Sightings can be resightings, and observation quality varies. |

ECCC's [archived chart catalogue](https://open.canada.ca/data/en/dataset/2dbe89e4-2b1a-4253-a552-86a26296900e)
identifies the Open Government Licence – Canada. Preserve source attribution and
check the terms attached to the downloaded product. NSIDC requires citation of
IIP G00807 using DOI [10.7265/N56Q1V5R](https://doi.org/10.7265/N56Q1V5R),
subset and access date. Its [HTTPS access guide](https://nsidc.org/data/user-resources/help-center/how-access-and-download-noaansidc-data)
describes public browsing under `https://noaadata.apps.nsidc.org/NOAA/`.
Do not disable TLS certificate verification when downloading.

The IIP user guide explicitly distinguishes observation source/method and
historical formats. In its modern CSV description, sighting longitude is degrees
west, so normalize the sign explicitly before creating longitude-first GeoJSON.
The documented date/time are sighting times, not a licence to assume that a
drifting iceberg remained at that position hours or days later. Inspect each
file's format and observed range before converting it.

## Validation data still needed

No verified public dataset was found here that supplies exhaustive, independent,
time-matched vessel/iceberg/clutter annotations for the project's current NL SAR
scenes. Public ice observations help locate examples but do not close that gap.
The portfolio must keep precision unmeasured until an annotated benchmark has
been created and independently checked.

For that benchmark retain only source data whose licences permit the intended
use and redistribution. A reproducible manifest can reference restricted or
large assets without committing them. The acquisition ID, scene time, AOI,
calibration files, AIS interval, annotation rules, reviewer and held-out split
are necessary even when the raster itself is publicly downloadable.

Coastlines and offshore infrastructure need provenance too. Land masks cannot
distinguish icebergs from vessels. Nominal oil-field centres are not surveyed
installation footprints, and floating production vessels may move. Mark missing
or outdated position evidence as uncertain rather than suppressing a nearby
vessel with a definitive infrastructure label.

The current small infrastructure list remains provisional (its inline source
references include Wikipedia). The `infrastructure_proximity` flag means nearby
context requiring review, not a verified platform identity. Mobile FPSO nominal
positions are not used for automatic proximity flags. Current surveyed positions
and footprints would be needed to replace this uncertainty with a stronger
identification claim.

A stronger historical cross-check is the regulator-hosted
[Hibernia Environmental Effects Monitoring Plan (2013), table 3.6](https://www.cnlopb.ca/wp-content/uploads/eem/eem_2013_1_hdmc.pdf).
It records the Hibernia platform sampling site around 46°45′1.7″N,
48°46′58.5″W. This is a sampling-area centre, not a current surveyed footprint;
the report mixes historical coordinate datums elsewhere, so a datum must be
established before treating these numbers as exact WGS 84 coordinates.
