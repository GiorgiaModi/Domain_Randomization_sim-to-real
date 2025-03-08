# Domain Randomization for Sim-to-Real Transfer

## Project Overview

This project explores domain randomization techniques to improve the robustness of reinforcement learning (RL) policies for sim-to-real transfer. The study focuses on environments like Hopper and Walker2d from OpenAI Gym, employing various domain randomization strategies to evaluate their impact on policy generalization.

<p align="center">
  <img src="https://github.com/user-attachments/assets/fe9d7513-ff8c-4efa-911d-0719e5983248" width="30%">
</p>

## Repository Contents

- **`env/`**: Contains environment configurations and customizations.
- **`models/`**: Stores trained RL models.
- **`plots/`**: Includes scripts and results for visualizing performance metrics.
- **`videos/`**: Contains videos demonstrating policy performance.
- **`paper_RL_project_Delli_Modi_Necerini.pdf`**: The comprehensive report detailing the project's methodology, experiments, and findings.
- **`project_Robot_Learning.ipynb`**: Jupyter notebook with code implementations and experiment results.
- **`readme.md`**: Project documentation (this file).
- **`requirements.txt`**: List of dependencies required to run the project.

If you use the Jupyter notebook (`project_Robot_Learning.ipynb`), all required dependencies will be installed automatically. Simply run the notebook to set up your environment. All the examples of usage are provided in the notebook.

Detailed experimental results, including performance comparisons of different randomization strategies, are presented in the report paper_RL_project_Delli_Modi_Necerini.pdf.

### Contributors

- Andrea Delli ([Politecnico di Torino](https://www.polito.it))
- Giorgia Modi ([Politecnico di Torino](https://www.polito.it))
- Ivan Necerini ([Politecnico di Torino](https://www.polito.it))

### Citation

If you use this project in your research, please cite:

```bibtex
@article{DelliModiNecerini2024,
  title={Domain Randomization Techniques for Reinforcement Learning: Bridging the Reality Gap},
  author={Andrea Delli, Giorgia Modi, Ivan Necerini},
  year={2024},
  institution={Politecnico di Torino}
}
```

### License

This project is licensed under the MIT License.

