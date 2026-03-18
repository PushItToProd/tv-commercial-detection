-- duckdb SQL


create or replace temp table results
as
    select
        json->>'$.status' as status,
        json->>'$.expected' as expected,
        json->>'$.classified' as classified,
        json->>'$.model_reply.reason' as reason,
        json->>'$.model_reply.source' as source,
        (json->>'$.elapsed')::DECIMAL as elapsed,
    from read_json_objects('classification_results24b-50words.jsonl', format = 'unstructured');


select
    status, expected, classified,
    round(avg(elapsed), 2) as avg_elapsed
from results
where source = 'llm' and expected != 'unknown'
group by all
;

with known_results as (
    select * from results where expected != 'unknown'
)
pivot known_results
on classified
using
    round(avg(elapsed), 2) as avg_elapsed,
    round(max(elapsed), 2) as max_elapsed
group by expected;


with known_results as (
    select * from results where expected != 'unknown'
)
pivot known_results
on expected
using
    round(avg(elapsed), 2) as avg_elapsed,
    round(max(elapsed), 2) as max_elapsed
group by classified, reason;
