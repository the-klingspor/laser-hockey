import matplotlib.pyplot as plt
import pandas as pd
from tueplots import bundles
from tueplots.constants.color import rgb

# Load the data
data_weak = pd.read_csv("weak.csv")
data_strong = pd.read_csv("strong.csv")

plt.rcParams.update(bundles.icml2022())

# Create a new figure
fig, axs = plt.subplots(1, 2, figsize=(6, 2), sharey=True)

# Define the colors for each experiment
colors = [
    rgb.tue_blue,
    rgb.tue_green,
    rgb.tue_violet,
    rgb.tue_orange,
    rgb.tue_dark,
    rgb.tue_red,
]

# Create an empty list to store legend information
lines = []

# Iterate through each experiment
for i in range(1, 7):
    # Extract the mean, min and max values for each experiment
    mean_weak = data_weak[f"Group: rainbow_experiment_{i}_new - win_percentage_weak"]
    min_val_weak = data_weak[
        f"Group: rainbow_experiment_{i}_new - win_percentage_weak__MIN"
    ]
    max_val_weak = data_weak[
        f"Group: rainbow_experiment_{i}_new - win_percentage_weak__MAX"
    ]

    mean_strong = data_strong[f"Group: rainbow_experiment_{i}_new - win_percentage_strong"]
    min_val_strong = data_strong[
        f"Group: rainbow_experiment_{i}_new - win_percentage_strong__MIN"
    ]
    max_val_strong = data_strong[
        f"Group: rainbow_experiment_{i}_new - win_percentage_strong__MAX"
    ]

    # Plot the mean win percentage and fill_between range for the weak dataset
    axs[0].plot(data_weak["Step"], mean_weak, color=colors[i - 1])
    axs[0].fill_between(
        data_weak["Step"], min_val_weak, max_val_weak, color=colors[i - 1], alpha=0.3
    )

    # Plot the mean win percentage and fill_between range for the strong dataset
    (line,) = axs[1].plot(
        data_strong["Step"], mean_strong, color=colors[i - 1]
    )  # Store the Line2D instance
    axs[1].fill_between(
        data_strong["Step"],
        min_val_strong,
        max_val_strong,
        color=colors[i - 1],
        alpha=0.3,
    )

    # Append the Line2D instance to the list
    lines.append(line)

for ax in axs:
    ax.axhline(1.0, color="lightgray", linestyle="--")

# Set the titles and labels
fig.suptitle("Rainbow Basic Experiments")
axs[0].set_title(r"Win\% vs. Weak Opponent")
axs[0].set_xlabel("Frames")
axs[0].set_ylabel("Win Percentage")

axs[1].set_title(r"Win\% vs. Strong Opponent")
axs[1].set_xlabel("Frames")
# axs[1].set_ylabel("Win Percentage")

# Add a shared legend outside of the subplots, to the right
labels = [f"Experiment {i}" for i in range(1, 7)]
fig.legend(lines, labels, loc="center", bbox_to_anchor=(1.06, 0.5))

plt.savefig("experiments.pdf", bbox_inches="tight")
