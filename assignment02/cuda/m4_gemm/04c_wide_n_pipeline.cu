// 4.4(c): a wider 128x128x64 output tile with two TMA stages.
// Reuse one A tile across twice as many output columns instead of increasing
// pipeline depth, because the 4.3 sweep showed no gain beyond two stages.
#define STAGES 2
#define BLOCK_N 128
#define PIPELINE_LABEL "4.4c wide-N"
#include "03_pipeline.cu"
