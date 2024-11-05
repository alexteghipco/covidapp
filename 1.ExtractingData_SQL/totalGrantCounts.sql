CREATE TABLE `dsapp-440110.covid.pubGrantCounts` AS
SELECT
    pa.publication_id,
    SUM(grc.grant_count) AS total_grant_count
FROM
    `dsapp-440110.covid.publicationAuthors` AS pa
LEFT JOIN
    `dsapp-440110.covid.grantCountsByResearcher` AS grc
ON
    pa.researcher_id = grc.researcher_id
GROUP BY
    publication_id;
