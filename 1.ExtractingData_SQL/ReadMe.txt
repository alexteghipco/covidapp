PubFilter.sql extracts main columns from covid-19-dimensions-ai.data.publications used for analysis and places into table pubWhole. See code for variables pulled--required some unnesting. And aggregations were applied to get unique values or counts for clinical_trial_ids and organizations represented across sthe author list.

pubWhole was supplemented with 3 additional measures that counted the number of grants, datasets and patents across authors of a publication. Here's how this was done:
1. Datasets associated with each publication were counted in the data.datasets table (totalDatasetCounts.sql)
2. Publication_id was mapped to researcher_id using authors in the pubWhole table (PubIdToResearcherId.sql)
3. Grants were counted by researcher_id,then aggregated by publication_id (totalGrantCounts.sql)
4. Patents were counted by researcher_id, then aggregated by publication_id (totalPatentCounts.sql)
5. Each count table (1,3,4) were filtered since they could contain publication_ids not in pubWhole (*Filter.sql)
6. Filtered count tables were joined with pubWhole to make pubAdditionalCounts table with these extra measures (pubAddCounts.sql)
7. Order of pub_ids from this new table was cross-referenced with pubWhole for merging with pandas later (checkAdditionalCountIDOrder.sql)

For graphs, collab.sql constructs the org_hierarchy table by joining publications data with GRID details, extracting publication_id, author_id, organization names, and geographic information for each author’s affiliated organization. It uses ARRAY_AGG with conditional logic to retrieve parent and child organization IDs and names, organizing hierarchical relationships for later network analysis.

collabAuthor.sql, collabCountry.sql, collabHigherLevel.sql and collabLowerLevel.sql all create yearly shared publication counts between entities. Note, collabLowerLevel.sql maps shared publications between lower-level institutions of a single higher-order entity. This ended up being not that informative so was not used.