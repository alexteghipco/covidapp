CREATE OR REPLACE TABLE `dsapp-440110.covid.publicationAuthors` AS
SELECT
    pub.id AS publication_id,
    author.researcher_id
FROM
    `covid-19-dimensions-ai.data.publications` AS pub,
    UNNEST(pub.authors) AS author
WHERE
    author.researcher_id IS NOT NULL;
