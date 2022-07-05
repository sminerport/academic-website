---
# An instance of the Portfolio widget.
# Documentation: https://wowchemy.com/docs/page-builder/
widget: portfolio

# This file represents a page section.
headless: true

# Order that this section appears on the page.
weight: 30

title: Data Analytics Projects
subtitle: 'Write-ups for various projects written in Python, SAS, R, SQL, etc.'

content:
  # Page type to display. E.g. project.
  page_type: project

  # Default filter index (e.g. 0 corresponds to the first `filter_button` instance below).
  filter_default: 0

  # Filter toolbar (optional).
  # Add or remove as many filters (`filter_button` instances) as you like.
  # To show all items, set `tag` to "*".
  # To filter by a specific tag, set `tag` to an existing tag name.
  # To remove the toolbar, delete the entire `filter_button` block.
  filter_button:
  - name: All
    tag: '*'
  - name: Algorithms
    tag: 'Algorithms'
  - name: Artificial Intelligence
    tag: 'Artificial Intelligence'
  - name: Classifiers
    tag: 'Classifiers'
  - name: Dashboards
    tag: 'Dashboards'
  - name: Data Engineering
    tag: 'Data Engineering'
  - name: Data Warehousing
    tag: 'Data Warehousing'
  - name: Machine Learning
    tag: 'Machine Learning'
  - name: Software Engineering
    tag: 'Software Engineering'
  - name: Web Apps
    tag: 'Web Apps'

design:
  # Choose how many columns the section has. Valid values: '1' or '2'.
  columns: '2'

  # Toggle between the various page layout types.
  #   1 = List
  #   2 = Compact
  #   3 = Card
  #   5 = Showcase
  view: 3

  # For Showcase view, flip alternate rows?
  flip_alt_rows: true
---

This section contains write-ups of data analytics projects.

