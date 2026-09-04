# One command, one verdict.
#
#   make verify   run every gate; nothing ships if this is not green
#
# The gates are ordered so the cheapest, most fundamental failure surfaces
# first: if the recurrent formulation does not match the parallel one, nothing
# downstream means anything, so there is no point checking the PDF.

PY ?= python
NODE ?= node

.PHONY: verify test docs artifact links clean help

help:
	@echo "make verify    every gate: equivalence, page logic, claims, PDFs, links"
	@echo "make test      correctness gates only"
	@echo "make docs      rebuild both PDFs and check them"
	@echo "make artifact  re-export weights and fixture from the checkpoint"
	@echo "make links     check every external link resolves"

verify:
	@$(PY) verify.py

test:
	@echo "--- 1/4  Python: parallel vs recurrent forward -------"
	@$(PY) tests/test_equivalence.py
	@echo ""
	@echo "--- 2/4  JS port vs Python reference ------------------"
	@$(NODE) tests/test_js_equivalence.mjs
	@echo ""
	@echo "--- 3/4  Page computation path ------------------------"
	@$(NODE) tests/test_page_logic.mjs
	@echo ""
	@echo "--- 4/4  Page figures vs the claims ledger ------------"
	@$(PY) scripts/verify_claims.py

docs:
	@echo ""
	@echo "--- concept summary ----------------------------------"
	@$(PY) scripts/build_summary_pdf.py summary
	@$(PY) scripts/check_pdf.py
	@echo ""
	@echo "--- blog ---------------------------------------------"
	@$(PY) scripts/build_summary_pdf.py blog

links:
	@echo ""
	@echo "--- external links -----------------------------------"
	@$(PY) scripts/check_links.py

artifact:
	$(PY) scripts/export_weights.py --ckpt data/artifact_d32m8.pt
	$(PY) scripts/vendor_fonts.py

clean:
	rm -rf .livecheck __pycache__ src/__pycache__ scripts/__pycache__
