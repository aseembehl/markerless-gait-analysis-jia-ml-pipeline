# Markerless Gait Analysis for Children and Adolescents with Juvenile Idiopathic Arthritis Using a Machine Learning Pipeline

This repository contains code for the method described in the paper, "Markerless Gait Analysis for Children and Adolescents with Juvenile Idiopathic Arthritis Using a Machine Learning Pipeline."

From a sagittal-plane video of a child walking, the pipeline estimates anatomical landmarks, computes hip, knee and ankle angles, and plots them over the normalised gait cycle for each walking direction.

## Getting Started

Requires Python 3.10 or newer.

```bash
git clone https://github.com/aseembehl/markerless-gait-analysis-jia-ml-pipeline.git
cd markerless-gait-analysis-jia-ml-pipeline
pip install -e '.[video,models]'

jia-gait run path/to/video.mp4 -o results/child01
```

The output folder contains the gait-cycle plots for hip, knee and ankle per walking direction (`plots/`), a quality report (`quality.txt`) and the landmarks, joint angles and gait cycles as data files. See [docs/usage.md](docs/usage.md) for installation options, all commands, the output files and configuration.

## License

Apache License 2.0, see [LICENSE](LICENSE).
