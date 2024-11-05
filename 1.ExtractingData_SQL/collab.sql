CREATE OR REPLACE TABLE `dsapp-440110.covid.org_hierarchy` AS
SELECT
    pub.id AS publication_id,
    pub.year AS publication_year,
    author.researcher_id AS author_id,
    CONCAT(author.first_name, ' ', author.last_name) AS author_name,
    org.grid_id AS org_grid_id,
    
    -- Use full country name from grid and include latitude/longitude for later mapping
    grid.address.country AS country_name,
    grid.address.latitude AS latitude,
    grid.address.longitude AS longitude,
    
    -- Use grid.name as the primary organization name
    grid.name AS organization_name,
    
    -- High-level organization: Use relationships.label where type is 'Parent'
    IFNULL(
        ARRAY_AGG(IF(rel.type = 'Parent', rel.id, NULL) IGNORE NULLS)[OFFSET(0)],
        org.grid_id
    ) AS high_level_org_id,
    
    IFNULL(
        ARRAY_AGG(IF(rel.type = 'Parent', rel.label, NULL) IGNORE NULLS)[OFFSET(0)],
        grid.name
    ) AS high_level_org_name,
    
    -- Low-level organization: Use relationships.label where type is 'Child'
    ARRAY_AGG(IF(rel.type = 'Child', rel.id, NULL) IGNORE NULLS)[OFFSET(0)] AS low_level_org_id,
    ARRAY_AGG(IF(rel.type = 'Child', CONCAT(rel.label, ' (', grid.name, ')'), NULL) IGNORE NULLS)[OFFSET(0)] AS low_level_org_name
FROM
    `covid-19-dimensions-ai.data.publications` AS pub
LEFT JOIN
    UNNEST(pub.authors) AS author
LEFT JOIN
    UNNEST(author.affiliations_address) AS org
LEFT JOIN
    `covid-19-dimensions-ai.data.grid` AS grid ON org.grid_id = grid.id
LEFT JOIN
    UNNEST(grid.relationships) AS rel -- Unnest relationships to access type, label, and id
GROUP BY
    publication_id, publication_year, author_id, author_name, org.grid_id, grid.address.country, grid.address.latitude, grid.address.longitude, grid.name;
