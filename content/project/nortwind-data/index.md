---
# Documentation: https://wowchemy.com/docs/managing-content/

title: "Data Loading - Northwind Data"
summary: "Develops a star schema-based OLAP data warehouse modeling a retailer's order fullfillment process" 
authors: ["admin"] 
tags: ["SQL", "Data Warehousing", "ETL", "PostgreSQL", "OLAP", "OLTP", "Star-Schema", "Data Analytics", "Fact Tables", "Dimension Tables"]
categories: ["SQL", "Data Warehousing", "ETL", "PostgreSQL", "OLAP", "OLTP", "Star-Schema", "Data Analytics", "Fact Tables", "Dimension Tables"]
date: 2021-10-25T02:42:00-06:00

# Optional external URL for project (replaces project detail page).
external_link: ""

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: "Star-Schema DW" 
  focal_point: "Smart"
  preview_only: false

# Custom links (optional).
#   Uncomment and edit lines below to show custom links.
# links:
# - name: Follow
#   url: https://twitter.com
#   icon_pack: fab
#   icon: twitter

links:
- icon: envelope-open-text
  icon_pack: fas
  name: Email
  url: https://scottminer.rbind.io/#contact
- icon: github
  icon_pack: fab
  name: Github
  url: https://github.com/sminerport
- icon: linkedin
  icon_pack: fab
  name: LinkedIn
  url: https://www.linkedin.com/in/scottdatascience

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ""
---
A star schema-based OLAP (online analytical processing) system models a retailer's order fulfillment process. The languages used include SQL and PostgreSQL. Also, an extract-transform-load (ETL) pipeline is created, which populates the fact and dimension tables of the star-schemea data warehouse to allow for optimized ad-hoc querying.

<hr/>
{{< icon name="download" pack="fas" >}}Download the {{< staticref "uploads/northwind-data.pdf" "newtab" >}}project{{< /staticref >}}.
<hr/>
