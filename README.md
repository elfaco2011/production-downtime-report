# production-downtime-report

Work in progress. Project code and documentation will be added with the first working version.

## Development

Every commit is scanned for secrets with [gitleaks](https://github.com/gitleaks/gitleaks).
Git hooks are not versioned, so after cloning install the hook manually:

```sh
brew install gitleaks   # or see the gitleaks install docs for your OS

cat > .git/hooks/pre-commit << 'EOF'
#!/bin/sh
gitleaks git --pre-commit --staged --redact -v || { echo "gitleaks: possible secret, commit blocked"; exit 1; }
EOF
chmod +x .git/hooks/pre-commit
```

If the hook finds a secret, the commit is blocked and the value is redacted in the output.
Remove the secret (move it to `.env`, which is git-ignored) and commit again.
