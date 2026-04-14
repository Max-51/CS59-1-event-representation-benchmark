# Third-Party Repositories

This folder contains official reference implementations used in our benchmark study.

## Included Repositories
- est: https://github.com/uzh-rpg/rpg_event_representation_learning
- ergo: https://github.com/uzh-rpg/event_representation_study
- get: https://github.com/Peterande/GET-Group-Event-Transformer
- matrixlstm: https://github.com/marcocannici/matrixlstm
- evrepsl: https://github.com/VincentQQu/EvRepSL

## Notes
- These repositories are kept separate from `src/` to avoid polluting the unified benchmark framework.
- Our own benchmark adapters and wrappers should be implemented in `src/representations/`.
- Each third-party repo may require a different environment.