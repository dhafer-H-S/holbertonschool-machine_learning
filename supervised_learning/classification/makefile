# Makefile

# Git commit message
COMMIT_MSG = ""

# Format Python code with autopep8
fixcodepy:
ifdef fixcodepy
	autopep8 --in-place --aggressive --aggressive $(fixcodepy)
else
	@echo "Please provide the filename. Usage: make fixcodepy=filename.py"
endif

# Git add, commit, and push
gitpush:
ifdef file
ifdef comment
	git add $(file)
	git commit -m "$(comment)"
	git push
else
	@echo "Please provide a commit message. Usage: make gitpush file=file.py comment='Your commit message'"
endif
else
	@echo "Please provide the filename and commit message. Usage: make gitpush file=file.py comment='Your commit message'"
endif
