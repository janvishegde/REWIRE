### REWIRE: Technical Specification & Implementation Architecture

[cite_start]**Core Objective:** To develop a computational framework for **Drug Repurposing** that treats the human interactome as a **Dynamic Communication Network**[cite: 32, 171]. [cite_start]Instead of relying on chemical "shape," the system quantifies the **Topological Footprint** of drugs and diseases to identify therapeutic "matches" based on signature inversion[cite: 33, 34, 123].

---

### 1. Data Ingestion & Graph Construction
[cite_start]The system is built on a tripartite data architecture[cite: 135, 136, 140]:
* [cite_start]**Base PPI Graph ($G_0$):** Sourced from the **STRING database**[cite: 125, 137].
    * [cite_start]**Nodes ($V$):** ~20,000 human proteins[cite: 33].
    * [cite_start]**Edges ($E$):** ~412,000 interactions with confidence weights[cite: 125, 155].
* [cite_start]**Drug-Target Map:** Sourced from **DrugBank**, mapping drugs to target proteins with quantitative $K_i/K_d$ affinities[cite: 125, 138, 155].
* [cite_start]**Disease-Gene Map:** Sourced from **OpenTargets**, identifying genetic disruption points[cite: 139, 155].

### 2. The Simulation Engine (Graph Perturbation)
[cite_start]The engine simulates drug action through a **re-weighting mechanism** rather than node deletion[cite: 132, 172]:
* [cite_start]**Logic:** For a drug with target set $T$ and inhibition constant $inh$, the simulation performs a deep copy of $G_0$ to create $G_{drug}$[cite: 155].
* **Transformation:** For every edge $(t, n)$ where $n \in T$, apply the weight attenuation: 
[cite_start]$$w_{new}(t,n) = w_{old}(t,n) \times (1 - inh)$$[cite: 155].


### 3. Feature Engineering: The 4D Rewiring Signature Vector (RSV)
[cite_start]The RSV is a vector $[\Delta BC, \Delta CM, \Delta SG, \Delta EN]$ that summarizes the structural transition from $G_0$ to $G_{drug}$[cite: 126, 157]:

1.  [cite_start]**Betweenness Centrality Shift ($\Delta BC$):** Tracks the rerouting of communication "traffic" across critical middleman proteins (hubs)[cite: 126, 143, 157].
    * [cite_start]*Implementation:* Mean absolute change in betweenness scores across all nodes[cite: 155].
2.  [cite_start]**Community Structure Change ($\Delta CM$):** Measures the reshuffling of functional "teams" or pathways[cite: 126, 143, 157].
    * [cite_start]*Implementation:* $1 - NMI$ (Normalized Mutual Information) of **Louvain partitions** between $G_0$ and $G_{drug}$[cite: 155].
3.  [cite_start]**Spectral Gap Delta ($\Delta SG$):** Quantifies the change in global network robustness and connectivity[cite: 126, 143, 157].
    * [cite_start]*Implementation:* Difference in the **Algebraic Connectivity** (second-smallest eigenvalue $\lambda_2$ of the Laplacian matrix)[cite: 155].
4.  [cite_start]**Neighbourhood Entropy ($\Delta EN$):** Captures local signal chaos specifically at the drug’s binding sites[cite: 126, 143, 157].
    * [cite_start]*Implementation:* The change in Shannon Entropy of edge weights for target nodes[cite: 155].


### 4. Machine Learning: The Two-Tower GAT
[cite_start]A **Graph Attention Network (GAT)** is utilized to learn complex relationships between drug and disease signatures[cite: 127, 145, 175]:
* [cite_start]**Architecture:** Two separate towers process the Drug RSV and the Disease DRS ($4$-dimensional inputs) into $64$-dimensional embeddings[cite: 155].
* [cite_start]**Objective:** Predict a **Cosine Similarity Score** $[0, 1]$ representing the therapeutic match[cite: 155].
* [cite_start]**Validation Target:** AUROC $> 0.80$ on held-out test sets[cite: 146, 155, 201].


### 5. Implementation Roles (Modular Task Split)
* [cite_start]**Person 1 (Data & Pipeline):** Ingestion of DrugBank XML and STRING CSVs; creation of clean data manifests[cite: 155].
* [cite_start]**Person 2 (Graph Builder):** Base $G_0$ construction and implementation of the multiprocessing binding simulator[cite: 155].
* [cite_start]**Person 3 (RSV Calculator):** Implementation of the 4 topological metrics and optimization of matrix operations[cite: 155].
* [cite_start]**Person 4 (Model & UI):** GAT model training and deployment of the Flask + Cytoscape.js visual interface[cite: 155].