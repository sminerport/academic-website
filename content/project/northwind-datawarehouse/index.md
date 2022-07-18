---
# Documentation: https://wowchemy.com/docs/managing-content/

title: "Data Loading - Northwind Data"
summary: "Develops a star schema-based OLAP data warehouse modeling a retailer's order fullfillment process"
authors: ["admin"]
tags: ["Data Engineering", "Data Warehousing", "Software Engineering"]
categories: ["Data Engineering", "Data Warehousing", "Software Engineering"]
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
- icon: twitter
  icon_pack: fab
  name: Follow
  url: https://www.twitter.com/scottdatascienc
- icon: envelope-open-text
  icon_pack: fas
  name: Email
  url: https://scottminer.netlify.app/#contact
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

**Downloads:**

<ul>
	<li>{{< icon name="download" pack="fas" >}}{{< staticref "uploads/northwind-data.pdf" "newtab" >}}Download {{< /staticref >}}{{< icon name="file-pdf" pack="far" >}}{{< staticref "uploads/northwind-data.pdf" "newtab" >}}PDF{{< /staticref >}}</li>
</ul>
<hr/>

**Word Document:**

<iframe src="https://onedrive.live.com/embed?cid=5B8EDCFD5CE8D99E&resid=5B8EDCFD5CE8D99E%21204276&authkey=AMpTmvyF-7I_zHM&em=2" width="100%" height="800" frameborder="1" scrolling="yes"></iframe>
