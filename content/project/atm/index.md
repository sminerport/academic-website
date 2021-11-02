---
# Documentation: https://wowchemy.com/docs/managing-content/

title: "ATM Web Application"
summary: "Implements an ATM web application using Python and the Flask micro web framework"
authors: ["admin"]
tags: ["Python", "Flask", "ATM"]
categories: ["Python", "Flask", "ATM"]
date: 2021-10-27T14:57:05-06:00

# Optional external URL for project (replaces project detail page).
external_link: ""

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.

image:
  caption: "Login Screen"
  focal_point: ""
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
url_code: ""
url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ""
---

This project presents UML sequence diagrams for an automated teller machine (ATM) web-based application. The program implements Python and the Flask micro web framework, allowing users to log in to accounts, check balances, and make withdraws. Additionally, the program implements secure coding by hashing customer PINs, disabling would-be attackers from redirecting URLs to external websites, and locking the accounts of users who have attempted three or more unsuccessful logins.

<hr/>
{{< icon name="download" pack="fas" >}}Download the {{< staticref "uploads/atm.pdf" "newtab" >}}project{{< /staticref >}}.
<hr/>
