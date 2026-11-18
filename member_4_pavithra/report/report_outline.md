# PathSnap Technical Report — working outline

> Replace every bracketed field with your team's actual methods and measured results. Do not submit placeholders as findings.

1. Team and system overview
2. Dataset understanding and EDA
   - preprocessing; missing values; GPS plots; road connectivity; Folium map; partition used
3. Model experimentation (minimum five)
   - consolidated table: family/config, CV macro-F1, student-test top-1, route continuity, student-test macro-F1, log loss, inference time
   - one subsection per model: suitability, strengths/limitations, hyperparameters, objective, measured results, over/underfit evidence, visualisations
4. Bayesian optimisation
   - search space, objective, trials, best params, untuned-vs-tuned comparison
5. HMM/Viterbi
   - states, emissions, transitions, gaps, missing candidates, disconnected edges, classifier-only vs decoder metrics
6. Geographical visualisation
   - actual GPS, road network, reference route where available, predicted route; legend/layers; map screenshot
7. API/UI/deployment
   - two-laptop setup, screenshots of UI and receiving-laptop API logs
8. Limitations, conclusions, reproducibility
9. AI assistance disclosure (as required by course policy)
