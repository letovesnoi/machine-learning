# Machine Learning Engineer Nanodegree
## Capstone Proposal
Elena Bushmanova  
August and September, 2019

## Proposal
<!-- _(approx. 2-3 pages)_ -->

### Domain Background
<!-- _(approx. 1-2 paragraphs)_ -->

<!-- In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required. -->

**Peptidic natural products** (PNPs) can have useful bioactive functions including the ability to be active against bacteria i.e. be an **antibiotic**. So PNPs are very interesting to scientists since widely used in human therapy. One of the main ways to study PNPs is through [mass spectrometry](https://en.wikipedia.org/wiki/Mass_spectrometry). For each PNPs you can get **spectrum** (intensity as a function of the mass-to-charge ratio) or a few by examining it in a black box - mass spectrometer. There is a widely known and most used tool named **DEREPLICATOR** for identification of PNPs via database search of tandem mass spectra (Mohimani H. et al., 2017).

PNPs structure varies between **linear**, **cyclic**, b-cyclic and complex (last two are not considered further for simplicity). The task that I am going to solve in Capstone project is to understand which spectra corresponds which type of PNPs structure. As expected it will  significantly speed up the DEREPLICATOR since the search will be through smaller bases (cyclic spectra only against cyclic compounds and linear only against linear). At the same time it will increase precision of algorithm because initial DEREPLICATOR compares any spectra with any compound and thereby can receive such false positive matching as linear spectra to nonlinear compound and nonlinear spectra to linear compound (not present an improved algorithm). Also even using only classification separately from DEREPLICATOR we obtain some biological properties of the compound represented by its own spectra. Cyclic PNPs are more stable and biologically more active on average so we can focus on the study of only such spectra thereby save our resources.

### Problem Statement
<!-- _(approx. 1 paragraph)_ -->

<!-- In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once). -->

The problem of this Capstone project is to **categorize PNPs spectra** into spectra corresponding **cyclic** compounds and **linear** (non-cyclic). It's planned that this problem will be solved by Supervised learning algorithms since there is [labeled data](https://gnps.ucsd.edu/ProteoSAFe/gnpslibrary.jsp?library=GNPS-LIBRARY#%7B%22Library_Class_input%22%3A%221%7C%7C2%7C%7C3%7C%7CEXACT%22%7D) and it's possible to validate and then test the model by such metrics as accuracy, precision, recall or F1 score.

### Datasets and Inputs
<!-- _(approx. 2-3 paragraphs)_ -->

<!-- In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem. -->

There are already a huge amount publicly [available](https://gnps.ucsd.edu/ProteoSAFe/gnpslibrary.jsp?library=GNPS-LIBRARY#%7B%22Library_Class_input%22%3A%221%7C%7C2%7C%7C3%7C%7CEXACT%22%7D) mass spectra of natural products. It turned out to be possible to detect natural products by their mass spectra and also find new ones missing in the database using a high-throughput technology built on computational algorithms as it does DEREPLICATOR.

I'm going to use this **one hundred million tandem mass spectra** in the Global Natural Products Social (GNPS) molecular networking infrastructure to select peptide compounds and categorize them into cyclic and non-cyclic by Machine learning algorithms. The labels can be taken from molecular structures from GNPS library (manually obtained by biologists for sure) or from DEREPLICATOR findings (calculated with some confidence). In both cases it's **several hundred structures** (about 200 cyclic and 100 non-cyclic structures) or what is the same about a **thousand spectra** (3-5 different spectra for the structure on average).

### Solution Statement
<!-- _(approx. 1 paragraph)_ -->

<!-- In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once). -->

It's Supervised learning task because example input-output (namely spectrum-structure) pairs exists. I will start with the simplest model so the baseline is some simple **Neural network** (most likely **CNN** to utilize a multi-dimensional data). The advantage of Neural networks approach is the possibility of non-linear models with respect to the features. I plan to try various data representations and then do some preprocessing steps. There are two ways to work with these continuous space of input data: **discretize** the raw spectra or directly **approximate** them by functions. Of course I also will try a different models (various layers and etc.) and most **Keras** optimizers. The solution can be measured by common metrics such as **accuracy, precision, recall** and more since there is labeled data.

### Benchmark Model
<!-- _(approximately 1-2 paragraphs)_ -->

<!-- In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail. -->

A good result that relates to the domain of Natural products identification would be less elapsed time and less FP at the same time obtained by **target matching DEREPLICATOR** (cyclic spectra against cyclic compounds and linear against linear) than by current DEREPLICATOR pipeline. It will mean that the model correctly classify the spectra by their structures into two groups. Now then the benchmark model is **current DEREPLICATOR** results.

### Evaluation Metrics
<!-- _(approx. 1-2 paragraphs)_ -->

<!-- In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms). -->

As noted earlier **FP** and **precision** are a good choice for evaluation metrics that can be used to quantify the performance of both the benchmark model and the solution. Here FP means that DEREPLICATOR got a structure that doesn't match input spectrum in reality.

Also we can simply compare results on target matching DEREPLICATOR on test set from GNPS library using **FP** metric where false means that spectrum corresponds not that cyclical that our ML algorithm got.

### Project Design
<!-- _(approx. 1 page)_ -->

<!-- In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project. -->

I will programming in **Python 3** using **pandas**, **NumPy**, **scikit-learn** and mainly **Keras**.

The workflow for approaching a solution given the problem includes
- **Collect** and **preprocess** the data. Firstly choose peptide compounds from GNPS Public Spectral Library and also the same from DEREPLICATOR results with high confidence. Then it's necessary to think thoroughly here about a representation of the input spectra since what features will consider our algorithm completely depends on it. It can be some tiny step discretization of raw spectra or spectra approximation by basis functions like RBF. Maybe it will be meaningful to use some data augmentation to increase the set of input data. After that when I understand the data I will identify what kind of preprocessing is needed: scaling, normalization and so on.
- **Split** the data into training, validation and test sets
- **Choose**, **train** and **tune** the model
- **Evaluate** the solution


<!-- ----------- -->

<!-- **Before submitting your proposal, ask yourself. . .** -->

<!-- - Does the proposal you have written follow a well-organized structure similar to that of the project template? -->
<!-- - Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification? -->
<!-- - Would the intended audience of your project be able to understand your proposal? -->
<!-- - Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes? -->
<!-- - Are all the resources used for this project correctly cited and referenced? -->
