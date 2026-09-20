# Building a chatbot and possible MCP server for PLATO Pub

## Background
The PLATO pub website https://platopub.phys.au.dk/about/platopub.php , led by Mikkel Lund and hosted at Aarhus University provides information about the PLATO space telescope.  PLATO is being build by a large consortium and will probably launch late 2026 or early 2027. For the website all publications about PLATO are collected in one place: https://platopub.phys.au.dk/about/ads_feed.php . Currently these are papers abou the development and expected capabilities of PLATO. Once the telescope becomes operational, this will shift towards papers presenting results form PLATO observations.

In this project we will make the information from the PLATO publications available through a chatbot, and possibly also an MCP server. The goal is to encourage and facilitate researchers' use of PLATO data for high quality, impactful results, by reducing the time and effort they need to spend on understanding the technical details of the instrument and data. This way they can focus on the science.

## The chatbot

The PLATO chatbot shall be part of the PLAT Pub website. We'll probably want to run it in a docker.

### Information sources

* platochet_info.md: a short document with high level information about PLATO (from the website, maybe pull it from ther directly in future?) and about the chatbot itself
* a full-text RAG index with hybrid retrieval to answer detailed questions 
* a whole-paper level index that lets users search papers by metadata, like author names and publication date

Important: older information can be outdated