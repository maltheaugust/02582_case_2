import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import numpy as np
from matplotlib.lines import Line2D

from cluster import reduce_to_threshold
from load_data import load, load_and_normalize_by_individual

from mpl_toolkits.mplot3d import Axes3D

def plot_by_teamID_and_phase_3d(data, teamid_metadata, phases_metadata):
    # Step 1: Reduce data to the desired number of PCA components
    data_reduced, _ = reduce_to_threshold(data, threshold=0.5)
    
    # Step 2: Convert to numpy array
    data_np = data_reduced.to_numpy()
    
    # Step 3: Extract x, y, z for the 3D scatter plot
    x = data_np[:, 0]
    y = data_np[:, 1]
    z = data_np[:, 2]  # This assumes that the reduced data has at least 3 dimensions

    # Step 4: Convert teamid and phases to numeric values for plotting
    unique_teamids = np.unique(teamid_metadata)
    teamid_mapping = {teamid: idx for idx, teamid in enumerate(unique_teamids)}
    teamid_numeric = np.array([teamid_mapping[teamid] for teamid in teamid_metadata])

    unique_phases = np.unique(phases_metadata)
    phase_mapping = {phase: idx for idx, phase in enumerate(unique_phases)}
    phases_numeric = np.array([phase_mapping[phase] for phase in phases_metadata])

    # Step 5: Set up a colormap and shape markers for the plot
    cmap = plt.get_cmap('tab20')  # Using tab20 for distinct colors
    markers = ['o', '^', 's', 'D', 'P']  # Shape markers for different phases
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    # Step 6: Plot each teamid with a different color and each phase with a different marker
    for i, teamid in enumerate(unique_teamids):
        for j, phase in enumerate(unique_phases):
            # Find indices for both teamid and phase
            idx = (teamid_numeric == i) & (phases_numeric == j)
            ax.scatter(x[idx], y[idx], z[idx], 
                       color=cmap(i % cmap.N), marker=markers[j % len(markers)], 
                       edgecolors='black', s=60, 
                       label=f"TeamID: {teamid}, Phase: {phase}" if (i == 0 and j == 0) else "")  # Prevent duplicate labels in legend

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Data colored by TeamID and shaped by Phase')

    # Step 7: Add the legend (only showing unique entries)
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels[label] = handle
    ax.legend(unique_labels.values(), unique_labels.keys(), title="Legend", loc='best')

    # Save and show the plot
    plt.savefig('teamid_by_phase_3d.png')
    plt.show()

def plot_clusters(data, labels, phase=None):
    x = data[:, 0]
    y = data[:, 1]
    
    num_classes = np.unique(labels)
    cmap = plt.get_cmap('tab10')

    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']  # marker styles
    unique_phases = np.unique(phase) if phase is not None else [None]

    fig, ax = plt.subplots(figsize=(8,6))

    # Plot points
    for cls in num_classes:
        for i, ph in enumerate(unique_phases):
            idx = (labels == cls)
            if phase is not None:
                idx = idx & (phase == ph)
            marker = markers[i % len(markers)]
            ax.scatter(x[idx], y[idx], 
                       color=cmap(cls), marker=marker, edgecolors='black', s=50)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')


    legend_elements = []
    for i, ph in enumerate(unique_phases):
        legend_elements.append(
            Line2D([0], [0], marker=markers[i % len(markers)], color='black', linestyle='None',
                   markersize=10, label=f'{ph}')
        )

    ax.legend(handles=legend_elements, title="Phase", loc='best')
    plt.show()

def plot_by_phase(data, phases_metadata, threshold=0.5):
    # Step 1: Reduce data columns based on threshold
    data_reduced, _ = reduce_to_threshold(data, threshold)
    
    # Step 2: Convert to numpy array
    data_np = data_reduced.to_numpy()
    
    x = data_np[:, 0]
    y = data_np[:, 1]

    unique_phases = np.unique(phases_metadata)
    phase_mapping = {phase: idx for idx, phase in enumerate(unique_phases)}
    phases_numeric = np.array([phase_mapping[phase] for phase in phases_metadata])

    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(8,6))

    for i, phase in enumerate(unique_phases):
        idx = (phases_numeric == i)
        ax.scatter(x[idx], y[idx], 
                   color=cmap(i), marker='o', edgecolors='black', s=60, label=f"{phase}")

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Data colored by Phase')
    ax.legend(title="Phase", loc='best')
    plt.savefig('phase_by_phase.png')
    plt.show()

def plot_pairplot_by_phase(data, phase_metadata, threshold=0.9):
    """
    1) Optionally reduce features via your existing threshold function
    2) Run PCA to get first 5 components
    3) Build a DataFrame with PC1–PC5 + cohort label
    4) Draw and save a seaborn pairplot
    """
    # 1) feature‑selection step (unchanged)
    data_reduced, _ = reduce_to_threshold(data, threshold=threshold)

    n_components=5
    # 3) build DataFrame for seaborn
    pc_cols = [f"PC{i+1}" for i in range(5)]
    df = pd.DataFrame(data_reduced.iloc[:, :n_components], columns=pc_cols)
    df['Phase'] = phase_metadata  # seaborn will hue on this
    
    # 4) pairplot
    g = sns.pairplot(
        df,
        vars=pc_cols,
        hue='Phase',
        diag_kind='hist',
        plot_kws={'edgecolor':'black', 's':60},
        corner=False
    )
    
    # tighten up title & layout
    g.fig.suptitle('Pairplot of First 5 Principal Components by Phase', y=1.02)
    g.fig.tight_layout()
    
    # save and show
    g.savefig('pairplot_by_Phase.png')
    plt.show()



def plot_by_Puzzler(data, puzzler_metadata):
    data_reduced, _ = reduce_to_threshold(data, threshold=0.5)
    
    # Step 2: Convert to numpy array
    data_np = data_reduced.to_numpy()
    
    x = data_np[:, 0]
    y = data_np[:, 1]

    unique_puzzlers = np.unique(puzzler_metadata)
    puzzler_mapping = {phase: idx for idx, phase in enumerate(unique_puzzlers)}
    puzzler_numeric = np.array([puzzler_mapping[phase] for phase in puzzler_metadata])

    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(8,6))

    for i, phase in enumerate(unique_puzzlers):
        idx = (puzzler_numeric == i)
        ax.scatter(x[idx], y[idx], 
                   color=cmap(i), marker='o', edgecolors='black', s=60, label=f"Puzzler: {phase}")

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Data colored by Puzzler')
    ax.legend(title="Phase", loc='best')
    plt.savefig('phase_by_phase.png')
    plt.show()


def plot_pairplot_by_puzzler(data, puzzler_metadata, threshold=0.9):
    """
    1) Optionally reduce features via your existing threshold function
    2) Run PCA to get first 5 components
    3) Build a DataFrame with PC1–PC5 + cohort label
    4) Draw and save a seaborn pairplot
    """
    # 1) feature‑selection step (unchanged)
    data_reduced, _ = reduce_to_threshold(data, threshold=threshold)

    n_components=5
    # 3) build DataFrame for seaborn
    pc_cols = [f"PC{i+1}" for i in range(5)]
    df = pd.DataFrame(data_reduced.iloc[:, :n_components], columns=pc_cols)
    df['Puzzler'] = puzzler_metadata  # seaborn will hue on this
    
    # 4) pairplot
    g = sns.pairplot(
        df,
        vars=pc_cols,
        hue='Puzzler',
        diag_kind='hist',
        plot_kws={'edgecolor':'black', 's':60},
        corner=False
    )
    
    # tighten up title & layout
    g.fig.suptitle('Pairplot of First 5 Principal Components by Puzzler', y=1.02)
    g.fig.tight_layout()
    
    # save and show
    g.savefig('pairplot_by_puzzler.png')
    plt.show()



def plot_by_cohort(data, cohort_metadata):
    data_reduced, _ = reduce_to_threshold(data, threshold=0.5)
    
    # Step 2: Convert to numpy array
    data_np = data_reduced.to_numpy()
    
    x = data_np[:, 0]
    y = data_np[:, 1]

    unique_cohorts = np.unique(cohort_metadata)
    cohort_mapping = {phase: idx for idx, phase in enumerate(unique_cohorts)}
    cohort_numeric = np.array([cohort_mapping[phase] for phase in cohort_metadata])

    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(8,6))

    for i, phase in enumerate(unique_cohorts):
        idx = (cohort_numeric == i)
        ax.scatter(x[idx], y[idx], 
                   color=cmap(i), marker='o', edgecolors='black', s=60, label=f"Cohort: {phase}")

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Data colored by Cohort')
    ax.legend(title="Phase", loc='best')
    plt.savefig('phase_by_phase.png')
    plt.show()

def plot_pairplot_by_cohort(data, cohort_metadata, threshold=0.9):
    """
    1) Optionally reduce features via your existing threshold function
    2) Run PCA to get first 5 components
    3) Build a DataFrame with PC1–PC5 + cohort label
    4) Draw and save a seaborn pairplot
    """
    # 1) feature‑selection step (unchanged)
    data_reduced, _ = reduce_to_threshold(data, threshold=threshold)
    

    # 3) build DataFrame for seaborn
    pc_cols = [f"PC{i+1}" for i in range(5)]
    df = pd.DataFrame(data_reduced, columns=pc_cols)
    df['Cohort'] = cohort_metadata  # seaborn will hue on this
    
    # 4) pairplot
    g = sns.pairplot(
        df,
        vars=pc_cols,
        hue='Cohort',
        diag_kind='hist',
        plot_kws={'edgecolor':'black', 's':60},
        corner=False
    )
    
    # tighten up title & layout
    g.fig.suptitle('Pairplot of First 5 Principal Components by Cohort', y=1.02)
    g.fig.tight_layout()
    
    # save and show
    g.savefig('pairplot_by_cohort.png')
    plt.show()


def plot_by_Puzzler_3d(data, puzzler_metadata):
    # Step 1: Reduce data to the desired number of PCA components
    data_reduced, _ = reduce_to_threshold(data, threshold=0.5)
    
    # Step 2: Convert to numpy array
    data_np = data_reduced.to_numpy()
    
    # Step 3: Extract x, y, z for the 3D scatter plot
    x = data_np[:, 0]
    y = data_np[:, 1]
    z = data_np[:, 2]  # This assumes that the reduced data has at least 3 dimensions

    # Step 4: Convert puzzler metadata to numeric values for plotting
    unique_puzzlers = np.unique(puzzler_metadata)
    puzzler_mapping = {puzzler: idx for idx, puzzler in enumerate(unique_puzzlers)}
    puzzler_numeric = np.array([puzzler_mapping[puzzler] for puzzler in puzzler_metadata])

    # Step 5: Set up a colormap for distinct colors
    cmap = plt.get_cmap('tab10')  # Using tab10 for distinct colors
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    # Step 6: Plot each puzzler with a different color
    for i, puzzler in enumerate(unique_puzzlers):
        idx = (puzzler_numeric == i)
        ax.scatter(x[idx], y[idx], z[idx], 
                   color=cmap(i), marker='o', edgecolors='black', s=60, 
                   label=f"Puzzler: {puzzler}")

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('Data colored by Puzzler')

    # Step 7: Add the legend
    ax.legend(title="Puzzler", loc='best')

    # Save and show the plot
    plt.savefig('puzzler_by_phase_3d.png')
    plt.show()

def plot_by_teamID_and_phase(data, teamid_metadata, phases_metadata):
    # Step 1: Reduce data to the desired number of PCA components
    data_reduced, _ = reduce_to_threshold(data, threshold=0.5)
    
    # Step 2: Convert to numpy array
    data_np = data_reduced.to_numpy()
    
    # Step 3: Extract x and y for the scatter plot
    x = data_np[:, 0]
    y = data_np[:, 1]

    # Step 4: Convert teamid and phases to numeric values for plotting
    unique_teamids = np.unique(teamid_metadata)
    teamid_mapping = {teamid: idx for idx, teamid in enumerate(unique_teamids)}
    teamid_numeric = np.array([teamid_mapping[teamid] for teamid in teamid_metadata])

    unique_phases = np.unique(phases_metadata)
    phase_mapping = {phase: idx for idx, phase in enumerate(unique_phases)}
    phases_numeric = np.array([phase_mapping[phase] for phase in phases_metadata])

    # Step 5: Set up colormap and shape markers for the plot
    cmap = plt.get_cmap('tab20')
    markers = ['o', '^', 's', 'D', 'P']  # You can add more shapes if needed
    fig, ax = plt.subplots(figsize=(8,6))

    # Step 6: Plot each teamid with a different color and each phase with a different marker
    for i, teamid in enumerate(unique_teamids):
        for j, phase in enumerate(unique_phases):
            # Find indices for both teamid and phase
            idx = (teamid_numeric == i) & (phases_numeric == j)
            ax.scatter(x[idx], y[idx], 
                       color=cmap(i), marker=markers[j % len(markers)], 
                       edgecolors='black', s=60, 
                       label=f"TeamID: {teamid}, Phase: {phase}" if (i == 0 and j == 0) else "")  # Prevent duplicate labels in legend

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Data colored by TeamID and shaped by Phase')

    # Step 7: Add the legend (only showing unique entries)
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels[label] = handle
    # ax.legend(unique_labels.values(), unique_labels.keys(), title="Legend", loc='best')

    # Save and show the plot
    plt.savefig('teamid_by_phase.png')
    plt.show()

if __name__ == "__main__":
    data, metadata = load_and_normalize_by_individual("data/HR_data_2.csv", cohort="D1_2")
    data = data.drop(labels='Phase', axis=1)
    data = data.drop(labels='Puzzler', axis=1)



    phases_metadata = metadata['Phase'][:data.shape[0]]
    puzzler_metadata = metadata['Puzzler'][:data.shape[0]]
    teamid_metadata = metadata['Team_ID'][:data.shape[0]]
    cohort_metadata = metadata['Cohort'][:data.shape[0]]


    
    data = data.to_numpy()
    phases_metadata = phases_metadata.to_numpy()
    puzzler_metadata = puzzler_metadata.to_numpy()
    teamid_metadata = teamid_metadata.to_numpy()
    # plot_pairplot_by_cohort(data, cohort_metadata)
    plot_by_teamID_and_phase(data, teamid_metadata, phases_metadata)
    # plot_pairplot_by_phase(data, phases_metadata)
    # plot_pairplot_by_puzzler(data, puzzler_metadata)
    # plot_by_teamID_and_phase_3d(data, teamid_metadata, phases_metadata)
    # plot_by_Puzzler_3d(data, puzzler_metadata)

