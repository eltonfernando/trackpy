
auto_formater:
	pre-commit run --all-files -c .pre-commit-config.yaml

uml_generation:
	pyreverse -o svg trackpy
