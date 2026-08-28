
## Method, §A4

Identical harness to §A3, one file deeper. `cov_queue4.sh` is `cov_queue.sh`
with `queue4.txt` / `logs4/` / `cov_run4.sh` substituted; `cov_run4.sh` is
`cov_run3.sh` with `logs2` → `logs4`; both were produced by `sed` from the
attempt-3 originals and the diff was verified to be exactly those substitutions,
so the run procedure — one pytest at a time, never piped, its own deadline, a
`tt-smi -glx_reset` after any non-clean exit, a log per run that is never
overwritten — is the same procedure attempt 3 qualified.

`TTTV2_GALAXY_CCL_TRACE` is **0** for every attempt-4 run, as it was for every
attempt-3 queue run: the trace synchronizes after each LM-head collective and
that is a real wall-clock cost on a 511-step decode (D-B19).

Two things attempt 4 did differently, both deliberate:

* **Nothing in the queue is a second run of a deterministic abort.** Ordering is
  by *what has never been measured*, then by *what has been measured once and
  passed*. A pass at one process is the dangerous case on this hardware; a
  byte-identical `TT_FATAL` at two or three is not.
* **Runs are slower than §A3's numbers predicted, and the reason is host, not
  device.** `a4_q_padded_greedy` spent 423s where `a3_q_greedy` spent 149s for
  the same 32 prefills and the same abort. The difference is the page cache: the
  weight `.tensorbin` set had not been read for five hours. Later runs of the
  same model in the same night come back down. Nothing was re-planned around
  this beyond widening the deadlines already in `queue4.txt`.
