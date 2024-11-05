CREATE OR REPLACE TABLE `dsapp-440110.covid.pubAdditionalCounts` AS
SELECT
    pubWhole.publication_id,
    COALESCE(filteredPubDatasetCounts.dataset_count, 0) AS dataset_count,
    COALESCE(filteredPubGrantCounts.total_grant_count, 0) AS grant_count,
    COALESCE(filteredPubPatentCounts.total_patent_count, 0) AS patent_count
FROM
    `dsapp-440110.covid.pubWhole` AS pubWhole
LEFT JOIN
    `dsapp-440110.covid.filteredPubDatasetCounts` AS filteredPubDatasetCounts
ON
    pubWhole.publication_id = filteredPubDatasetCounts.publication_id
LEFT JOIN
    `dsapp-440110.covid.filteredPubGrantCounts` AS filteredPubGrantCounts
ON
    pubWhole.publication_id = filteredPubGrantCounts.publication_id
LEFT JOIN
    `dsapp-440110.covid.filteredPubPatentCounts` AS filteredPubPatentCounts
ON
    pubWhole.publication_id = filteredPubPatentCounts.publication_id;
