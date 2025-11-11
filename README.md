# RS-STGCN: Regional-Synergy Spatio-Temporal Graph Convolutional Network for Emotion Recognition
  
Decoding emotional states from electroencephalography (EEG) signals is a fundamental goal in affective neuroscience. This endeavor requires accurately modeling the complex spatio-temporal dynamics of brain activity. However, prevailing approaches for defining brain connectivity often fail to reconcile predefined neurophysiological priors with task-specific functional dynamics. This paper presents the Regional-Synergy Spatio-Temporal Graph Convolutional Network (RS-STGCN), a novel framework designed to bridge this gap. The core innovation is the Regional Synergy Graph Learner (RSGL), which integrates known physiological brain-region priors with a task-driven optimization process. \textcolor{red}{ It constructs a sparse, adaptive graph by modelling connectivity at two distinct levels. At the intra-regional level, it establishes core information backbones within functional areas. This ensures efficient and stable local information processing. At the inter-regional level, it adaptively identifies critical, sparse long-range connections. These connections are essential for global emotional integration. This dual-level, dynamically learned graph then serves as the foundation for the spatio-temporal network. This network effectively captures evolving emotional features. The proposed framework demonstrates superior recognition accuracy, achieving state-of-the-art results of 88.00\% and 85.43\% on the public SEED and SEED-IV datasets, respectively, under a strict subject-independent protocol. It also produces a neuroscientifically interpretable map of functional brain connectivity, identifying key frontal-parietal pathways consistent with established attentional networks.} This work offers a powerful computational approach to investigate the dynamic network mechanisms underlying human emotion, providing new data-driven insights into functional brain organization. 

Authors:
Yunqi Han{1}, Yifan Chen{2}*, Hang Ruan{2}, Deqing Song{3}, Haoxuan Xu{3} and Haiqi Zhu{4}

{1} Faculty of Computer Science and Information Technology, University Putra Malaysia, 43400 Serdang, Selangor, Malaysia  
{2} School of Computer Science, University of Nottingham Malaysia, Semenyih, 43500, Malaysia
{3} Faculty of Computing, Harbin Institute of Technology, Harbin, 15001, China
{4} School of Medicine and Health, Harbin Institute of Technology, Harbin, 15001, China

The public datasets analyzed for this study can be found at the following locations.

The SEED dataset is available at: https://bcmi.sjtu.edu.cn/home/seed/seed.html

The SEED-IV dataset is available at: https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html
