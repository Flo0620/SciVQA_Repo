---
license: mit
viewer: true
configs:
- config_name: default
  data_files:
  - split: train
    path: "train_2025-03-27_18-34-44.json"
  - split: validation
    path: "validation_2025-03-27_18-34-44.json"
task_categories:
- visual-question-answering
language:
- en
---

<img src="SciVQA_logo.gif" alt="drawing" width="300"/>


# SciVQA dataset

This dataset is used for the **SciVQA: Scientific Visual Question Answering Shared Task** hosted at the [Scholary Document Processing workshop](https://sdproc.org/2025/) at ACL 2025. 
**Competition is available on [Codabench](https://www.codabench.org/competitions/5904/)**.

SciVQA is a corpus of **3000** real-world figure images extracted from English scientific publications available in the ACL Anthology and arXiv. 
The figure images are collected from the two pre-existing datasets, [ACL-Fig](https://arxiv.org/abs/2301.12293) and [SciGraphQA](https://arxiv.org/abs/2308.03349).

All figures are automatically annotated using the Gemini 1.5-flash model, **each with 7 QA pairs**, and then manually validated by graduate students with Computational Linguistics background. 
The SciVQA dataset contains **21000** QA pairs in total. The language of all QA pairs is English. 

Each figure is associated with the following metadata:

- <code>instance_id</code> - unique ID of a given instance;
- <code>image_file</code> - name of the figure PNG;
- <code>figure_id</code> - unique ID of a figure;
- <code>caption</code> - caption text of a figure;
- <code>figure_type</code> - type category of a figure (e.g., pie chart, line graph, see section below);
- <code>compound</code> - True if compound, otherwise False (see section below);
- <code>figs_numb</code> - number of subfigures. If compound is False, equal to 0;
- <code>qa_pair_type</code> - question type category (see section below);
- <code>question</code>;
- <code>answer_options</code> - predefined answer options for non-binary QA type (see section below);
- <code>answer</code>;
- <code>paper_id</code> - ID of the source publication;
- <code>categories</code> - arXiv field(s);
- <code>venue</code> - acl or arXiv;
- <code>source_dataset</code> - scigraphqa or aclfig;
- <code>pdf_url</code> - URL of the source publication.

# Data splits 

| Split  |  Images | QA pairs |    
|--------|---------|----------|
| Train  |   2160  |  15120   |   
| Val    |   240   |  1680    |        
| Test   |         |          |  
| **Total**|       |          |             


## QA pair types schema

<img src="qa_pairs_schema.png" alt="drawing" width="550"/>

- **Closed-ended** - it is possible to answer a question based only on a given data source, i.e., an image or an image and a caption. No additional resources such as the main text of a publication, other documents/figures/tables, etc. are required.
- **Unanswerable** - it is not possible to infer an answer based solely on a given data source.
- **Infinite answer set** - there are no predefined answer options., e.g., "What is the sum of Y and Z?".
- **Finite answer set** - associated with a limited range of answer options. Such QA pairs fall into two subcategories:
  - **Binary** - require a yes/no or true/false answer, e.g., "Is the percentage of positive tweets equal to 15%?".
  - **Non-binary** - require to choose from a set of **four** predefined answer options where one or more are correct, e.g., "What is the maximum value of the green bar at the threshold equal to 10?" Answer options: "A: 5, B: 10, C: 300, D: None of the above".
- **Visual** - address or incorporate information on one or more of the **six visual attributes** of a figure, i.e., *shape*, *size*, *position*, *height*, *direction* or *colour*. E.g., "In the bottom left figure, what is the value of the blue line at an AL of 6?". Here the visual aspects are: position (bottom left), colour (blue), and shape (line).
- **Non-visual** - do not involve any of the six visual aspects of a figure defined in our schema, e.g., "What is the minimum value of X?".

## Figure types

The figures in SciVQA dataset are classified into:
- **Compound** - a figure image contains several (sub)figures which can be separated and constitute individual figures.
- **Non-compound** - a figure image contains a single figure object which cannot be decomposed into multiple subfigures.
- **Line chart**, **bar chart**, **box plot**, **confusion matrix**, **pie chart**, etc.

The information on figure types can serve as an additional metadata during systems training.

## Structure

    ├── train_2025-03-27_18-34-44.json       # train split with QA pairs and metadata
    ├── validation_2025-03-27_18-34-44.json  # valdiation split with QA pairs and metadata
    ├── test (tba)
    ├── images_train.zip                     # figure images for the train set
    ├── images_validation.zip                # figure images for the validation set       
    └── images_test (tba)                 

## Citation

TBA

## Funding

This work has received funding through the DFG project [NFDI4DS](https://www.nfdi4datascience.de) (no. 460234259). 

<div style="position: relative; width: 100%;">
  <img src="NFDI4DS.png" alt="drawing" width="200" style="position: absolute; bottom: 0; right: 0;"/>
</div>
