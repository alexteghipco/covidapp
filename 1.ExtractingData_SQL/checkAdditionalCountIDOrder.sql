WITH pubWhole_ordered AS (
    SELECT 
        publication_id, 
        ROW_NUMBER() OVER (ORDER BY publication_id) AS row_num
    FROM 
        `dsapp-440110.covid.pubWhole`
),
pubAdditionalCounts_ordered AS (
    SELECT 
        publication_id, 
        ROW_NUMBER() OVER (ORDER BY publication_id) AS row_num
    FROM 
        `dsapp-440110.covid.pubAdditionalCounts`
)

SELECT 
    pubWhole_ordered.publication_id AS pubWhole_id,
    pubAdditionalCounts_ordered.publication_id AS pubAdditionalCounts_id,
    pubWhole_ordered.row_num AS pubWhole_row_num,
    pubAdditionalCounts_ordered.row_num AS pubAdditionalCounts_row_num
FROM 
    pubWhole_ordered
FULL OUTER JOIN 
    pubAdditionalCounts_ordered
ON 
    pubWhole_ordered.row_num = pubAdditionalCounts_ordered.row_num
WHERE 
    pubWhole_ordered.publication_id != pubAdditionalCounts_ordered.publication_id
