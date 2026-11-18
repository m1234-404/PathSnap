# Official deliverables checklist

Based on the supplied Deliverables.pdf and Models and Training Guidelines.pdf.

- [ ] Public GitHub repository with complete source, dependencies and README
- [ ] Raw data inspection, cleaning and preprocessing source
- [ ] Feature engineering source and rationale
- [ ] At least FIVE actually trained/evaluated models; each has config and logs
- [ ] Group-aware CV on the 80 training trips
- [ ] Student-testing comparison on the 10 student-test trips
- [ ] Macro-F1, observation Top-1, log loss where applicable, candidate miss rate, route continuity, training/inference time
- [ ] Comparison plots generated from real experiment results
- [ ] Bayesian optimisation of selected model; trial log and tuned-vs-untuned comparison
- [ ] HMM/Viterbi implementation and classifier-only vs decoded evaluation
- [ ] Folium map: GPS, road network, reference route where available, predicted route, legend/layers
- [ ] FastAPI backend and model loading
- [ ] UI listing the 10 evaluator trips and sending selected trip ID to API
- [ ] Two-laptop deployment instructions and screenshots of UI + receiving-laptop API logs
- [ ] One consolidated technical report PDF, sections 2–7 as specified
- [ ] AI assistance disclosure per course policy
- [ ] No fabricated metrics, plots, model results, evaluator labels or hardcoded predictions

## Dataset caveat
The uploaded dataset archive contains 80 `train` trips and 10 `student_test` trips only. It does not include the 10 `evaluator_test` trips referenced in the guidelines. Request the official evaluator partition from the course organisers. The API defaults to `evaluator_test`; do not silently use `student_test` as a substitute.
