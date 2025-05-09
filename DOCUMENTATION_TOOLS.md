# Documentation Tools for Unlook SDK

This document outlines recommended documentation tools that integrate well with GitHub and would be suitable for the Unlook SDK project.

## Recommended Tools

### MkDocs with Material Theme
**Website:** https://squidfunk.github.io/mkdocs-material/

**Key Features:**
- Markdown-based documentation 
- Beautiful, responsive Material Design theme
- Easy integration with GitHub Pages
- Search functionality built-in
- Code syntax highlighting
- Versioning support
- Math equation support via MathJax

**Setup:**
```bash
pip install mkdocs-material
mkdocs new .
```

**GitHub Integration:**
- Automated deployment via GitHub Actions
- Customizable CI pipeline for documentation updates
- Can be hosted on GitHub Pages from your repository

### Docusaurus
**Website:** https://docusaurus.io/

**Key Features:**
- Developed by Facebook/Meta
- React-based documentation site generator
- Versioning system for documentation
- Excellent support for API documentation
- Blog functionality included
- i18n support for translations

**Setup:**
```bash
npx create-docusaurus@latest unlook-docs classic
```

**GitHub Integration:**
- Seamless deployment to GitHub Pages
- Documentation can live alongside code in the same repository
- Excellent support for large, complex projects

### Sphinx with Read the Docs Theme
**Website:** https://www.sphinx-doc.org/

**Key Features:**
- Python's standard documentation system
- reStructuredText or Markdown support
- Automatic API documentation generation from docstrings
- Multi-language support
- Extensive extension ecosystem

**Setup:**
```bash
pip install sphinx sphinx-rtd-theme
sphinx-quickstart
```

**GitHub Integration:**
- Can be hosted on Read the Docs with GitHub integration
- Automatic rebuilds when documentation changes
- Versioned documentation linked to releases

### GitHub Pages with Just the Docs
**Website:** https://just-the-docs.github.io/just-the-docs/

**Key Features:**
- Lightweight solution directly using GitHub Pages
- Clean, accessible Jekyll theme
- Built-in search
- Navigation structure
- Responsive design
- Minimal setup required

**Setup:**
1. Create `docs/` folder in your repository
2. Add a `_config.yml` file with the theme setting
3. Enable GitHub Pages in repository settings

**GitHub Integration:**
- Native GitHub Pages integration
- No external services required
- Simple to set up and maintain

## Recommendation for Unlook SDK

Based on the Unlook SDK's needs as a Python-based 3D scanning library:

1. **MkDocs with Material Theme** would be the best choice because:
   - Lightweight and easy to set up
   - Excellent for Python projects
   - Beautiful documentation that's easy to navigate
   - Markdown-based (already used in your existing documentation)
   - Can be easily extended with plugins for API documentation

2. **Secondary recommendation: Sphinx** if more extensive API documentation generation is needed.

## Implementation Steps for MkDocs

1. Install MkDocs and Material theme:
   ```bash
   pip install mkdocs mkdocs-material
   ```

2. Initialize your documentation:
   ```bash
   mkdocs new .
   ```

3. Configure `mkdocs.yml`:
   ```yaml
   site_name: Unlook SDK Documentation
   theme:
     name: material
     palette:
       primary: blue
       accent: blue
   nav:
     - Home: index.md
     - Installation: installation.md
     - Getting Started: getting-started.md
     - User Guide:
       - Overview: user-guide/overview.md
       - Real-time Scanning: user-guide/realtime-scanning.md
       - Camera Configuration: user-guide/camera-configuration.md
     - API Reference:
       - Client: api/client.md
       - Core: api/core.md
       - Server: api/server.md
     - Examples: examples.md
     - Contributing: contributing.md
     - Roadmap: roadmap.md
   ```

4. Create documentation folder structure in `docs/` directory

5. Set up GitHub Actions for automatic deployment:
   Create `.github/workflows/docs.yml`:
   ```yaml
   name: Documentation

   on:
     push:
       branches:
         - main

   jobs:
     deploy:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - uses: actions/setup-python@v4
           with:
             python-version: 3.x
         - run: pip install mkdocs-material
         - run: mkdocs gh-deploy --force
   ```

This setup will provide a solid foundation for comprehensive, maintainable documentation that integrates seamlessly with your GitHub repository.