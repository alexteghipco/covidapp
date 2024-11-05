CREATE OR REPLACE TABLE `dsapp-440110.covid.country_collab_matrix` AS
SELECT
    oh1.publication_year AS year,
    oh1.country_name AS node1,
    oh2.country_name AS node2,
    COUNT(DISTINCT oh1.publication_id) AS weight
FROM
    `dsapp-440110.covid.org_hierarchy` oh1
JOIN
    `dsapp-440110.covid.org_hierarchy` oh2 ON oh1.publication_id = oh2.publication_id
WHERE
    oh1.country_name < oh2.country_name -- Avoid duplicate pairs and self-joins
GROUP BY
    year, node1, node2
ORDER BY
    year, node1, node2;
