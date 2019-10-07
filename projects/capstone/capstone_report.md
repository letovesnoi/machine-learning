# Machine Learning Engineer Nanodegree
## Capstone Project
Elena Bushmanova  
September and October, 2019

## I. Definition
<!-- _(approx. 1-2 pages)_ -->

### Project Overview
<!-- In this section, look to provide a high-level overview of the project in layman’s terms. Questions to ask yourself when writing this section: -->
<!-- - _Has an overview of the project been provided, such as the problem domain, project origin, and related datasets or input data?_ -->
<!-- - _Has enough background information been given so that an uninformed reader would understand the problem domain and following problem statement?_ -->

**Peptidic natural products** (PNPs) are small bioactive compounds consisting of amino acids connected via peptide bonds. A PNP may be represented as a graph with amino acids as nodes and bonds as edges. These graphs have either linear, cyclic, or more complex structure. PNPs are important for medicine since many of them are active against bacteria i.e. could be **antibiotics**. One of the main ways to study PNPs is through [mass spectrometry](https://en.wikipedia.org/wiki/Mass_spectrometry). For each PNP you can get a **spectrum** (intensity as a function of the mass-to-charge ratio) or a few by examining it in a black box -- mass spectrometer. These spectra can further be compared against databases of previously characterized compounds using computational methods such as **DEREPLICATOR** ([Mohimani H. et al., 2017](https://www.nature.com/articles/nchembio.2219)).

Understanding which spectra correspond to which types of PNPs structure will significantly speed up the DEREPLICATOR since it will be possible to search through smaller sets (cyclic spectra only against cyclic compounds and linear only against linear). At the same time it will increase precision of the algorithm because initial DEREPLICATOR compares any spectrum with any compound and thereby can get such false positive matching as linear spectrum to nonlinear compound and nonlinear spectra to linear compound (not present in an improved algorithm). Also knowledge about the structure itself (separately from DEREPLICATOR) tells scientists some biological properties of the compound represented by its own spectrum. Cyclic PNPs are more stable and biologically more active on average so we can focus on studying of only such spectra thereby saving our resources.

### Problem Statement
<!-- In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section: -->
<!-- - _Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?_ -->
<!-- - _Have you thoroughly discussed how you will attempt to solve the problem?_ -->
<!-- - _Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?_ -->

The problem of this Capstone project is to **categorize PNPs spectra** into spectra corresponding to **cyclic** and **linear** compounds (branch-cyclic and complex classes can also be considered). Thus the program requires spectrum of the unknown compound as input and defines type of the compound structure as output.

### Metrics
<!-- In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section: -->
<!-- - _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_ -->
<!-- - _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_ -->

**AUC** (instead of *accuracy* since the dataset can't be considered fully balanced), **precision**, **recall**, **F1 score** and **FP** as the primary metric are a good choice for evaluation metrics that can be used to quantify the performance of both the current DEREPLICATOR (in the sense of benchmark model) and the Target matching DEREPLICATOR. Here FP means that DEREPLICATOR got a structure that actually doesn't match input spectrum.

Also we can simply compare results of our model with random model results on test set from GNPS library using **FP** metric where false means that spectrum corresponds to other cyclicality than the model got.

## II. Analysis
<!-- _(approx. 2-4 pages)_ -->

### Data Exploration
<!-- In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section: -->
<!-- - _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_ -->
<!-- - _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_ -->
<!-- - _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_ -->
<!-- - _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_ -->

There is already a huge amount of publicly [available](https://gnps.ucsd.edu/) mass spectra of natural products. It turned out to be possible to detect natural products by their mass spectra and also find new ones missing in the database using a high-throughput technology built on computational algorithms such as DEREPLICATOR.

I'm going to use this one hundred million tandem mass spectra in the Global Natural Products Social (GNPS) molecular networking infrastructure ([Wang M. et al., 2016](https://www.nature.com/articles/nbt.3597)) to select peptide compounds and classify them using Machine learning algorithms. The labels can be taken from molecular structures from [GNPS library](https://gnps.ucsd.edu/ProteoSAFe/gnpslibrary.jsp?library=GNPS-LIBRARY#%7B%22Library_Class_input%22%3A%221%7C%7C2%7C%7C3%7C%7CEXACT%22%7D) (trustworthy labels manually obtained by biologists) or from highly-reliable DEREPLICATOR identifications. In both cases it's **several hundred cyclic and non-cyclic structures** and **several thousand spectra** related to them (3-5 different spectra for the structure on average).

Each spectrum is in the [MGF Format](https://ccms-ucsd.github.io/GNPSDocumentation/downloadlibraries/#mgf-format) consisting of list of pairs of mass-to-charge ratio and intensity (see ```data/spectra/*.mgf```, ```data/spectra_REG_RUN/*.mgf``` or ```data/GNPS-LIBRARY.mgf```). The compound structure for spectrum from GNPS library is in the [Molfile](https://en.wikipedia.org/wiki/Chemical_table_file) containing information about the atoms, bonds, connectivity and molecular coordinates (see ```data/mols/*.mol```). Information about spectra structures identified by DEREPLICATOR can be found in [tab-separated values](https://en.wikipedia.org/wiki/Tab-separated_values) ```data/REG_RUN_GNPS/regrun_fdr0_complete.tsv```.

**Collect** the data. Choose peptide not complex compounds from GNPS Public Spectral Library and also the same highly-reliable DEREPLICATOR identifications. GNPS library alone contains *443 peptidic* spectra (*85 linear*, *82 cyclic*, *71 branch-cyclic* and *205 complex*) and DEREPLICATOR identifies *7505 peptidic* spectra (*3101 linear*, *2681 cyclic*, *1692 branch-cyclic* and *31 complex*).


### Exploratory Visualization
<!-- In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section: -->
<!-- - _Have you visualized a relevant characteristic or feature about the dataset or input data?_ -->
<!-- - _Is the visualization thoroughly analyzed and discussed?_ -->
<!-- - _If a plot is provided, are the axes, title, and datum clearly defined?_ -->

### Algorithms and Techniques
<!-- In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section: -->
<!-- - _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_ -->
<!-- - _Are the techniques to be used thoroughly discussed and justified?_ -->
<!-- - _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_ -->

It's Supervised learning task because example input-output (namely spectrum-structure) pairs exists. I will start with the simplest **Neural network** model. The advantage of Neural networks approach is the possibility of non-linear models with respect to the features. I plan to try various data representations and then do some preprocessing steps. There are two ways to work with these continuous space of input data: **discretize** the raw spectra or directly **approximate** them by functions. For discretization most likely I will use **CNN** to utilize spatial information and for function approximation -- **usual NN**. Of course I also will try a different models (various layers and etc.) and most **Keras** optimizers. The solution can be measured by common metrics such as **AUC**, **precision**, **recall** and more since there is labeled data.

### Benchmark
<!-- In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section: -->
<!-- - _Has some result or value been provided that acts as a benchmark for measuring performance?_ -->
<!-- - _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_ -->

A good result that relates to the domain of Natural products identification would be less elapsed time and less FP at the same time obtained by **target matching DEREPLICATOR** (cyclic spectra against cyclic compounds and linear against linear) than by current DEREPLICATOR pipeline. It will mean that the model correctly classify the spectra by their structures into two groups. Thus the benchmark model is **current DEREPLICATOR** results.

For cyclic-linear classification itself **random model** will be used as benchmark model.

## III. Methodology
<!-- _(approx. 3-5 pages)_ -->

### Data Preprocessing
<!-- In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section: -->
<!-- - _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_ -->
<!-- - _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_ -->
<!-- - _If no preprocessing is needed, has it been made clear why?_ -->
It's necessary to think thoroughly here about a representation of the input spectra since what features will consider our algorithm completely depends on it. Each spectrum can be converted into intensity vector by tiny step discretization in which mass-to-charge ratios are indices and intensities are values (let the length be 50-150 thousand). Also spectrum can be approximated by basis functions like RBF. Maybe it will be meaningful to use some data augmentation to increase the set of input data. After that when I understand the data I will identify what kind of preprocessing is needed: scaling, normalization and so on. **Split** the data into training, validation and test sets such that both linear and cyclic compounds fall into each of these sets in acceptable proportions.

### Implementation
<!-- In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section: -->
<!-- - _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_ -->
<!-- - _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_ -->
<!-- - _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_ -->
I will use **Python 3** with **pandas**, **NumPy**, **scikit-learn** and mainly **Keras**. Some steps have already been done in ```capstone.ipynb``` to get input data representation.

**Choose**, **train** and **tune** the model. The initial CNN could include 2 convolutional layers (anyway up to 4 due to the large length of intensity vector), each with 4 filters of size 1×4 and two fully connected layers of 512 and 2 (number of output categories) neuron units. We also use tanh or ReLU activation, max-pooling, and dropout to prevent overfitting. Then I get some intuitions about how these networks work on spectra data by testing them and plotting some scores, change initial model varying layers and other hyperparameters, use different optimizers and also try any more models. There are some articles about Deep learning on mass spectra data into which I want to dig deeper (mainly [Tran N. H. et al., 2017](https://www.pnas.org/content/114/31/8247) and [2019](https://www.nature.com/articles/s41592-018-0260-3))

### Refinement
<!-- In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section: -->
<!-- - _Has an initial solution been found and clearly reported?_ -->
<!-- - _Is the process of improvement clearly documented, such as what techniques were used?_ -->
<!-- - _Are intermediate and final solutions clearly reported as the process is improved?_ -->


## IV. Results
<!-- _(approx. 2-3 pages)_ -->

### Model Evaluation and Validation
<!-- In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section: -->
<!-- - _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_ -->
<!-- - _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_ -->
<!-- - _Can results found from the model be trusted?_ -->
<!-- - _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_ -->


### Justification
<!-- In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section: -->
<!-- - _Are the final results found stronger than the benchmark result reported earlier?_ -->
<!-- - _Have you thoroughly analyzed and discussed the final solution?_ -->
<!-- - _Is the final solution significant enough to have solved the problem?_ -->

After getting two groups of spectra by approved network run DEREPLICATOR for cyclic spectra against cyclic compounds and linear against linear separately. Compare FP and elapsed time for these results and for DEREPLICATOR on full set of spectra. Also compare with random model and compute FP for approved network without considering DEREPLICATOR pipeline.

## V. Conclusion
<!-- _(approx. 1-2 pages)_ -->

### Free-Form Visualization
<!-- In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section: -->
<!-- - _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_ -->
<!-- - _Is the visualization thoroughly analyzed and discussed?_ -->
<!-- - _If a plot is provided, are the axes, title, and datum clearly defined?_ -->

### Reflection
<!-- In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section: -->
<!-- - _Have you thoroughly summarized the entire process you used for this project?_ -->
<!-- - _Were there any interesting aspects of the project?_ -->
<!-- - _Were there any difficult aspects of the project?_ -->
<!-- - _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_ -->

### Improvement
<!-- In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section: -->
<!-- - _Are there further improvements that could be made on the algorithms or techniques you used in this project?_ -->
<!-- - _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_ -->
<!-- - _If you used your final solution as the new benchmark, do you think an even better solution exists?_ -->

<!-- ----------- -->

<!-- **Before submitting, ask yourself. . .** -->

<!-- - Does the project report you’ve written follow a well-organized structure similar to that of the project template? -->
<!-- - Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification? -->
<!-- - Would the intended audience of your project be able to understand your analysis, methods, and results? -->
<!-- - Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes? -->
<!-- - Are all the resources used for this project correctly cited and referenced? -->
<!-- - Is the code that implements your solution easily readable and properly commented? -->
<!-- - Does the code execute without error and produce results similar to those reported? -->
