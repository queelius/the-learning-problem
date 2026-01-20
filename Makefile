.PHONY: docs docs-serve docs-clean help

POST_DIR := post

help:
	@echo "Targets:"
	@echo "  make docs        - Build mkdocs site"
	@echo "  make docs-serve  - Serve docs locally at localhost:8000"
	@echo "  make docs-clean  - Remove docs build artifacts"

docs: docs-clean
	mkdir -p docs/post
	cp -r $(POST_DIR)/*/ docs/post/
	mkdocs build

docs-serve:
	mkdir -p docs/post
	cp -r $(POST_DIR)/*/ docs/post/
	mkdocs serve

docs-clean:
	rm -rf site docs/post
