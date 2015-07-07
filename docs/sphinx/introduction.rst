Introduction
************

A number of tools exist for generating machine annotations (labels) for neuroscience data.  These tools have varying, specific input and output requirements, which makes it difficult to build interoperable pipelines and workflows.

macho provides a series of interfaces to get image data from the OCP spatial database, run a machine annotation tool, and convert the resulting annotations in a CAJAL compatible format that can be easily written back to an OCP project.

As workflows and additional functionalities are developed, we will continue to add to this repository.  As functionality matures, code will be spun off to separate projects (e.g., vesicle synapse detection).

Currently, we provide wrappers for the following tools/capabilities:

- ilastik (Pixel Classification)
- gala (agglomerative segmentation)
- rhoana (multi-hypothesis fusion/segmentation)
