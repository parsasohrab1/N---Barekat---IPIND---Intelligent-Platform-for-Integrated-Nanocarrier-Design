# Patent material — restricted access

The file `patentability_analysis.md.gpg` contains the patentability analysis, claims strategy and competitive comparison and is **encrypted with AES-256** (`gpg --symmetric --cipher-algo AES256`). Its plain-text version is never committed (see the project root `.gitignore`).

Only passphrase holders (the owner/original inventors) can open it:

```bash
gpg -d docs/patent/patentability_analysis.md.gpg > docs/patent/patentability_analysis.md
```

The passphrase is shared with authorized people through a separate channel (not in this repo). Other team members without the passphrase see only this encrypted binary file and cannot read its content.

⚠️ After opening, do not commit the plain-text file `patentability_analysis.md` — `.gitignore` ignores it, but be careful it is not force-added manually.
