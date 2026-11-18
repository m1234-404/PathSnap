# Team & ownership

| Member | USN | Ownership from team division |
|---|---|---|
| OP Medha | PES2UG24CS332 | Model experimentation/selection, at least five models, comparison, Bayesian optimisation, model artifact and experiment logs |
| P Dhanush | PES2UG24CS333 | Dataset exploration, preprocessing, candidate generation and feature engineering |
| Pavankumar H | PES2UG24CS345 | HMM, transition/emission probabilities, Viterbi decoding, route continuity and decoder evaluation |
| Pavithra GC | PES2UG24CS346 | FastAPI backend, evaluator-facing UI, Folium integration, deployment evidence and consolidated technical report |

Shared integration is everyone's responsibility. Interfaces are defined in `src/pathsnap/`.

## Integration contract
- Dhanush produces per-observation candidate features with stable `trip_id`, `observation_id`, `candidate_arc_id`, and target labels only for permitted training/student-test evaluation.
- Medha trains probability-capable classifiers on training trips only, uses trip-grouped CV, saves model + feature schema, and evaluates on student-test trips.
- Pavankumar consumes ordered candidate IDs and classifier probabilities, returns a decoded candidate sequence; never silently invent transitions.
- Pavithra exposes prediction-only endpoints. Ground-truth evaluation, if permitted for development, is server-side only; never return labels to UI. Evaluator-test labels must not enter prediction code.

## Academic integrity
The course policy requires independent experimentation and understanding. This is a starter scaffold, not evidence that experiments were performed. Do not submit generated placeholder metrics or generic report text as actual findings. Document AI assistance as required by course policy.
