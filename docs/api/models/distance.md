# Distance and Neighborhood Models API

`CellNeighborhoodDetector` and
`StationaryRegionNeighborDetector` are documented point-scoring adaptations,
not exact reproductions of the source papers' set-level NETS and KDE/top-*n*
STARE procedures.

::: aberrant.model.distance.KNN

::: aberrant.model.distance.LocalOutlierFactor

::: aberrant.model.distance.CellNeighborhoodDetector

::: aberrant.model.distance.SDOStream

::: aberrant.model.distance.StationaryRegionNeighborDetector

## Optional FAISS engine

This engine requires `aberrant[faiss]`.

::: aberrant.utils.similar.faiss_engine.FaissSimilaritySearchEngine
