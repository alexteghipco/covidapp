CREATE TABLE `dsapp-440110.covid.filteredPubGrantCounts` AS
SELECT
    gc.publication_id,
    gc.total_grant_count
FROM
    `dsapp-440110.covid.pubGrantCounts` AS gc
JOIN
    `dsapp-440110.covid.pubWhole` AS pubWhole
ON
    gc.publication_id = pubWhole.publication_id;
