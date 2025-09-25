# RS-STGCN: Regional-Synergy Spatio-Temporal Graph Convolutional Network for Emotion Recognition
  
Decoding emotional states from electroencephalography (EEG) signals is a fundamental goal in affective neuroscience. This endeavor requires accurately modeling the complex spatio-temporal dynamics of brain activity. However, prevailing approaches for defining brain connectivity often fail to reconcile predefined neurophysiological priors with task-specific functional dynamics. This paper presents the Regional-Synergy Spatio-Temporal Graph Convolutional Network (RS-STGCN), a novel framework designed to bridge this gap. The core innovation is the Regional Synergy Graph Learner (RSGL), which integrates known physiological brain-region priors with a task-driven optimization process. The RSGL constructs a sparse, adaptive graph by modeling connectivity at two levels. At the intra-regional level, it establishes core information pathways using a minimum spanning tree. At the inter-regional level, it identifies critical long-range connections via a budgeted selection mechanism. This dynamically learned graph then serves as the foundation for a spatio-temporal network that captures evolving emotional features. The proposed framework not only achieves superior recognition accuracy but also produces a neuroscientifically interpretable map of functional brain connectivity. This work offers a powerful computational approach to investigate the dynamic network mechanisms underlying human emotion, providing new data-driven insights into functional brain organization.

Authors:
Yunqi Han{1}, Yifan Chen{2}*, Hang Ruan{2}, Deqing Song{3}, Haoxuan Xu{3} and Haiqi Zhu{4}

{1} Faculty of Computer Science and Information Technology, University Putra Malaysia, 43400 Serdang, Selangor, Malaysia  
{2} School of Computer Science, University of Nottingham Malaysia, Semenyih, 43500, Malaysia
{3} Faculty of Computing, Harbin Institute of Technology, Harbin, 15001, China
{4} School of Medicine and Health, Harbin Institute of Technology, Harbin, 15001, China

The public datasets analyzed for this study can be found at the following locations.

The SEED dataset is available at: https://bcmi.sjtu.edu.cn/home/seed/seed.html

The SEED-IV dataset is available at: https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html
