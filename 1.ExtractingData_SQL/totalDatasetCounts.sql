CREATE TABLE `dsapp-440110.covid.pubDatasetCounts` AS
SELECT
    pub.id AS publication_id,
    COUNT(*) AS dataset_count
FROM
    `covid-19-dimensions-ai.data.publications` AS pub
LEFT JOIN
    `covid-19-dimensions-ai.data.datasets` AS ds
ON
    pub.id = ds.associated_publication_id
GROUP BY
    publication_id;
