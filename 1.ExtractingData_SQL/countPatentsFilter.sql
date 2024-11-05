CREATE TABLE `dsapp-440110.covid.filteredPubPatentCounts` AS
SELECT
    pc.publication_id,
    pc.total_patent_count
FROM
    `dsapp-440110.covid.pubPatentCounts` AS pc
JOIN
    `dsapp-440110.covid.pubWhole` AS pubWhole
ON
    pc.publication_id = pubWhole.publication_id;
