CREATE OR REPLACE TABLE `dsapp-440110.covid.authorToOrg` AS
WITH AuthorAffiliations AS (
    SELECT
        DISTINCT
        CONCAT(author.first_name, ' ', author.last_name) AS author_name,  -- Full author name
        author.researcher_id AS author_id,
        grid.address.country AS country_name,  -- Country from grid address
        COALESCE(
            ARRAY_AGG(IF(rel.type = 'Parent', rel.label, NULL) IGNORE NULLS)[OFFSET(0)], 
            grid.name
        ) AS highest_level_organization_name  -- Parent organization label or grid name as fallback
    FROM
        `covid-19-dimensions-ai.data.publications` AS pub
    LEFT JOIN
        UNNEST(pub.authors) AS author
    LEFT JOIN
        UNNEST(author.affiliations_address) AS org
    LEFT JOIN
        `covid-19-dimensions-ai.data.grid` AS grid ON org.grid_id = grid.id
    LEFT JOIN
        UNNEST(grid.relationships) AS rel  -- Unnest relationships for hierarchy
    WHERE
        author.researcher_id IS NOT NULL
    GROUP BY
        author_name, author_id, country_name, grid.name
)
SELECT
    author_id,
    author_name,
    country_name,
    highest_level_organization_name
FROM
    AuthorAffiliations;
