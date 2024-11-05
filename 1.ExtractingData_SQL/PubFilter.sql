CREATE TABLE `dsapp-440110.covid.pubWhole` AS
SELECT
    pub.id AS publication_id,
    pub.year,
    pub.abstract.preferred AS abstract_preferred,
    pub.abstract.original AS abstract_original,
    pub.date,
    pub.date_online,
    pub.book_title.preferred AS book_title_preferred,
    pub.book_title.original AS book_title_original,
    pub.proceedings_title.preferred AS proceedings_title_preferred,
    pub.proceedings_title.original AS proceedings_title_original,
    pub.conference.name AS conference_name,
    pub.title.preferred AS title_preferred,
    pub.title.original AS title_original,

    -- Flatten clinical_trial_ids
    ARRAY_TO_STRING(ARRAY_AGG(DISTINCT clinical_trial_id), ', ') AS clinical_trial_ids,

    -- Citation metrics
    CAST(pub.citations_count AS FLOAT64) AS citation_count,
    CAST(pub.metrics.times_cited AS INT64) AS times_cited,
    CAST(pub.metrics.recent_citations AS INT64) AS recent_citations,
    CAST(pub.metrics.field_citation_ratio AS FLOAT64) AS field_citation_ratio,
    CAST(pub.metrics.relative_citation_ratio AS FLOAT64) AS relative_citation_ratio,

    -- Altmetric/category data
    CAST(pub.altmetrics.score AS FLOAT64) AS altmetric_score,
    ARRAY_TO_STRING(pub.category_bra.values, ', ') AS field,
    pub.type AS publication_type,
    IF(ARRAY_LENGTH(pub.open_access_categories) > 0, TRUE, FALSE) AS is_open_access,

    -- Authors
    ARRAY_AGG(STRUCT(
        author.researcher_id AS researcher_id,
        author.first_name,
        author.last_name,
        author.grid_ids AS author_grid_ids,
        author.orcid
    )) AS authors,

    -- Funding (grant IDs and grid IDs of funders)
    ARRAY_AGG(STRUCT(
        fund.grant_id,
        fund.grid_id AS funder_grid_id
    )) AS funding_details,

    -- Unique org ids as csv
    ARRAY_TO_STRING(ARRAY_AGG(DISTINCT CAST(org.grid_id AS STRING)), ', ') AS unique_org_ids,

    -- Org location to string
    ARRAY_TO_STRING(ARRAY_AGG(DISTINCT CAST(org.city_id AS STRING)), ', ') AS research_org_city_ids,
    ARRAY_TO_STRING(ARRAY_AGG(DISTINCT org.country_code), ', ') AS research_org_country_codes,
    ARRAY_TO_STRING(ARRAY_AGG(DISTINCT org.state_code), ', ') AS research_org_state_codes,

    -- Count unique orgs
    COUNT(DISTINCT org.grid_id) AS num_unique_orgs
FROM
    `covid-19-dimensions-ai.data.publications` AS pub
LEFT JOIN
    UNNEST(pub.authors) AS author
LEFT JOIN
    UNNEST(author.affiliations_address) AS org
LEFT JOIN
    UNNEST(pub.funding_details) AS fund
LEFT JOIN
    UNNEST(pub.clinical_trial_ids) AS clinical_trial_id -- Unnest clinical_trial_ids to aggregate
GROUP BY
    publication_id, year, abstract_preferred, abstract_original, date, date_online, book_title_preferred, 
    book_title_original, proceedings_title_preferred, proceedings_title_original, conference_name,
    title_preferred, title_original, citation_count, times_cited, recent_citations, field_citation_ratio,
    relative_citation_ratio, altmetric_score, field, publication_type, is_open_access
