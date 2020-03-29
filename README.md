# Hidden Community Detection
This repository contains code for the final project of Stanford's CS224W (Machine Learning with Graphs) on hidden community detection. We apply the HICODE algorithm to identify hidden community structure in a graph of Reddit forum hyperlinks, predict future links in the graph, and test for hidden community structure after adding the predicted links to the original graph.

## Dataset

The SNAP Reddit Hyper Link Network can be found [here](https://snap.stanford.edu/data/soc-RedditHyperlinks.html). We unified the redditHyperlinks-body and redditHyperlinks-title networks into a single undirected, weighted graph.

## Required Packages/Software
* Python >= 3.6
* networkx
* Tensorflow-gpu 1.12.0
* Keras 2.2.4
* numpy
* scikit-learn
* pandas
* [Node2Vec](https://github.com/VHRanger/graph2vec?fbclid=IwAR1Hr4TwxCnhhiSW5AXY4MTshF7u6NHJx-F_FDqPqElpYACAtQiAAShEeyE)
* snap-stanford

## Running Code
Once the data has been downloaded and the required packages have been installed, run the create_weighted_graphs.ipynb notebook in the pipeline folder to generate the weighted graph combining the title and body networks. 

From there, you could generate embeddings (either RolX or Node2Vec) via the pipeline/rolx_embeddings.ipynb notebook or graph2vec/node2vec_embeddings.ipynb notebook, respectively. With the graph and embeddings generated, the pipeline/create_prediction_dataset.ipynb notebook can be run to generate a dataset for predicting new edges to be added to the graph. With a dataset, you can run a baseline linear regression model using a notebook in the baselines folder or train neural networks using the training code found in the training folder.

To run the HICODE algorithm for hidden community detection, run python community_detection/hicode.py. The Louvain community detection algorithm can be run via the community_detection/CURG_Louvain.ipynb notebook.

## Authors

* **Nicholas Benavides**
* **Jonathan Li**
* **Daniel Salz**