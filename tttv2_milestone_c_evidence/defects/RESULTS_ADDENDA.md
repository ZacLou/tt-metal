# Rows that belong in `RESULTS.md` and are not in it

`RESULTS.md` is machine-written by `queue.sh`, one appended row per run. It is the
authoritative index and it is preferred over prose everywhere in this tree — so a row
that went missing has to be recorded somewhere the reader will find it, rather than
inserted by hand into a machine-written file mid-run.

## `t9_llama_seeded_slot_r3`, 2026-08-31

The queue dequeued it at 21:12:50Z (`tttv2_milestone_c_runs/c-defects4/q11.queue.log`),
the run completed at 21:20:19Z, and `logs/t9_llama_seeded_slot_r3.log` is on disk, 40 189
lines, ending `exit=1`. **No row for it reached `RESULTS.md`**: the file goes straight from
`t8_llama_seeded_slot_r2` at 21:12:26Z to `u1_s7_page_table_placement_p1` at 21:21:33Z.

The append was lost, not the run. `queue.sh` writes the row with `printf … >> "$RESULTS"`
and this evidence tree is on a shared filesystem; the row for the immediately preceding
dequeue is present and so is the one after, so this is a lost append and not a crash — the
queue went on to dequeue `u1` normally.

The row it would have written, reconstructed from the log's own header and timings:

| when (UTC) | name | node | result | seconds | log |
| --- | --- | --- | --- | --- | --- |
| 21:20:45Z | t9_llama_seeded_slot_r3 | `models/common/tests/models/llama33_70b_galaxy/test_step7_coverage_wh_galaxy.py::test_llama_a_seeded_slot_repeats_across_runs` | 1 failed, 2 warnings (AssertionError: a seeded stochastic decode did not repeat) | 449 | `logs/t9_llama_seeded_slot_r3.log` |

`t9` is superseded as evidence in any case: it ran against a working tree that was between
two commits. The three-fresh-process set for this claim at commit `299440bb276` is
`ze1`–`ze3`.
