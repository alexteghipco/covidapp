CREATE TABLE `dsapp-440110.covid.pubPatentCounts` AS
SELECT
    pa.publication_id,
    SUM(patc.patent_count) AS total_patent_count
FROM
    `dsapp-440110.covid.publicationAuthors` AS pa
LEFT JOIN
    `dsapp-440110.covid.patentCountsByResearcher` AS patc
ON
    pa.researcher_id = patc.researcher_id
GROUP BY
    publication_id;
