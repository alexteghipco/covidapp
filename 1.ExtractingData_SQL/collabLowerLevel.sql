CREATE OR REPLACE TABLE `dsapp-440110.covid.low_level_collab_matrix` AS
SELECT
    oh1.publication_year AS year,
    CONCAT(oh1.low_level_org_name, ' (', oh1.high_level_org_name, ')') AS node1,
    CONCAT(oh2.low_level_org_name, ' (', oh2.high_level_org_name, ')') AS node2,
    COUNT(DISTINCT oh1.publication_id) AS weight
FROM
    `dsapp-440110.covid.org_hierarchy` oh1
JOIN
    `dsapp-440110.covid.org_hierarchy` oh2 ON oh1.publication_id = oh2.publication_id
WHERE
    oh1.low_level_org_id < oh2.low_level_org_id -- Avoid duplicate pairs and self-joins
    AND oh1.high_level_org_id = oh2.high_level_org_id -- Ensure both nodes are within the same parent institution
GROUP BY
    year, node1, node2
ORDER BY
    year, node1, node2;
