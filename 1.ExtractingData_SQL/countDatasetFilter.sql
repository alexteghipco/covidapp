CREATE TABLE `dsapp-440110.covid.filteredPubDatasetCounts` AS
SELECT
    ds.publication_id,
    ds.dataset_count
FROM
    `dsapp-440110.covid.pubDatasetCounts` AS ds
JOIN
    `dsapp-440110.covid.pubWhole` AS pubWhole
ON
    ds.publication_id = pubWhole.publication_id;
