# Summary week 5

This chapter focuses on deploying the portfolio itself as a small static website. The goal was not to build a complicated production system, but to make the work from the earlier chapters easy to open, navigate, and review without running the notebooks locally.

The deployment target for this portfolio is GitHub Pages. The published site is available at [simonosadchii.github.io/han-mads-portfolio](https://simonosadchii.github.io/han-mads-portfolio/). GitHub Pages fits the project well because the repository already uses Markdown summaries, relative links, and static artifacts such as notebooks, reports, CSV files, and configuration files. A static deployment is also safer and simpler than exposing a live Python application, because there is no server process, database, or secret configuration needed for visitors.

The main deployment work was therefore about structure. The `README.md` file acts as the homepage and links to the individual chapter summaries. Each chapter summary then links to the supporting files for that chapter, such as notebooks, results, reports, and reflections. This keeps the deployed site readable while still making the technical evidence available.

One practical lesson from this deployment is that a portfolio is only useful if the links are maintained. Broken links, placeholder text, or inaccurate chapter descriptions make the project feel unfinished even when the experiments themselves are good. For that reason, I checked the chapter links and updated the wording so the homepage matches the actual work in the repository.

The biggest limitation of this deployment is that it shows completed artifacts rather than running experiments interactively. That is acceptable for this portfolio, because the heavy work belongs in the notebooks and scripts, while the deployed site should communicate the conclusions clearly. I did not deploy a model API or a notebook server for this chapter, because that would add operational complexity without helping the portfolio goal. If I wanted to improve the site further, I would add screenshots or exported plots to make it easier to scan without opening notebooks.

Overall, GitHub Pages is a good deployment choice for this project. It keeps the portfolio lightweight, reproducible, and easy to share, while the repository remains the source of truth for the code and experiment outputs.

[Go back to Homepage](../README.md)
