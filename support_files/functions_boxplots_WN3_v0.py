import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

def plot_boxes_W_N3(ax, data,
                 font_scale=1.4,
                 point_size = 6,
                 metric='metric?',
                 plot_title='Title?',
                 xgrid=False,
                 saveplot=False,
                 filename='filename',
                 dpi=300):

    # create seaborn context
    sns.set_context('notebook', font_scale=font_scale)

    # Light colors for the box
    box_palette = {'W':'#FFE994', 'N3':'#9BDDF9'}
    # Dark colors for the dots
    swarmplot_palette = {'W':'#FF6600', 'N3':'#2A7FFF'}

    # Plot the box
    sns.boxplot(y="value",
                x="cond",
                data=data,
                linewidth=1.5,
                hue="cond",
                palette=box_palette,
                saturation=1.0,
                showmeans=True,
                meanline=True,
                legend=False,
                meanprops=dict(linestyle='dashed', color='gray', linewidth=2),
                medianprops=dict(linestyle=None, linewidth=2),
                ax=ax)

    # Plot the swarmplot on top 
    sns.swarmplot(y="value",
                  x="cond",
                  data=data,
                #   color="auto",
                  s=point_size,  # Circle size
                  palette=swarmplot_palette,
                  hue="cond",
                  legend=False,
                  ax=ax)

    # Calculate means #############################
    means = data.groupby("cond")["value"].mean()
    means = means.iloc[::-1]

    # Calculate statistical significance between 'W' and 'N3' groups ###############
    stat_test = stats.ranksums(data[data['cond'] == 'W']['value'], data[data['cond'] == 'N3']['value'])
    # Calculate significance level (adjust p-value if necessary)
    p_value = stat_test.pvalue
    if p_value < 0.00001:
        significance_asterisks = '*****'
    elif p_value < 0.0001:
        significance_asterisks = '****'
    elif p_value < 0.001:
        significance_asterisks = '***'
    elif p_value < 0.01:
        significance_asterisks = '**'
    elif p_value < 0.05:
        significance_asterisks = '*'
    else:
        significance_asterisks = '(n.s.)'

    # Add a bar or bracket between the box plots
    (miny, maxy) = ax.get_ylim()
    yposition = maxy
    maxy = maxy + 0.1 * (maxy - miny)
    ax.set_ylim(miny, maxy)
    endwidth = (maxy - miny) / 100
    ax.plot([0, 1], [yposition, yposition], color='black', lw=1.5, zorder=20)  # Adjust the coordinates and style as needed
    ax.plot([0, 0], [yposition - endwidth, yposition + endwidth], color='black', lw=1.5, zorder=20)
    ax.plot([1, 1], [yposition - endwidth, yposition + endwidth], color='black', lw=1.5, zorder=20)
    # Add significance annotation to the plot
    if p_value < 0.05:
        ax.annotate(f'{significance_asterisks}', xy=(0.5, yposition + endwidth), ha='center', fontsize=12)
    else:
        ax.annotate(f'p = {p_value:.5f} {significance_asterisks}', xy=(0.5, yposition + endwidth), ha='center', fontsize=12)
    ################################################################################

    # Change axis labels, ticks, and title
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Wakefulness", "Deep Sleep"])
    ax.set_xlabel('')
    ax.set_ylabel(metric)
    ax.set_title(plot_title)

    # Add horizontal grid
    if xgrid:
        ax.grid(axis='y')
        ax.set_axisbelow(True)

    if saveplot:
        # Save plots
        plt.savefig(filename + '.pdf', bbox_inches='tight', dpi=dpi)
        plt.savefig(filename + '.svg', bbox_inches='tight', dpi=dpi)
        plt.savefig(filename + '.png', bbox_inches='tight', dpi=dpi)
        plt.tight_layout()
        plt.show()

def plot_boxes_HC_MCI_AD(ax, data,
                 font_scale=1.4,
                 point_size = 6,
                 metric='metric?',
                 plot_title='Title?',
                 xgrid=False,
                 saveplot=False,
                 filename='filename',
                 dpi=300):

    groups = ["HC", "MCI", "AD"]

    df_list = []
    for arr, group in zip(data, groups):
        means = arr.mean(axis=1)  # mean across 379
        temp_df = pd.DataFrame({
            "cond": group,   # group label
            "value": means   # mean per subject
        })
        df_list.append(temp_df)

    data = pd.concat(df_list, ignore_index=True)

    # create seaborn context
    sns.set_context('notebook', font_scale=font_scale)

    # Light colors for violin
    box_palette = {'HC': '#B3E2CD', 'MCI': '#FDCDAC', 'AD': '#CBD5E8'}
    # Dark colors for dots
    swarmplot_palette = {'HC': '#1B9E77', 'MCI': '#D95F02', 'AD': '#7570B3'}

    # Plot the box
    sns.boxplot(y="value",
                x="cond",
                data=data,
                linewidth=1.5,
                #hue="cond",
                palette=box_palette,
                saturation=1.0,
                showmeans=True,
                meanline=True,
                # Removed invalid legend argument
                meanprops=dict(linestyle='dashed', color='gray', linewidth=2),
                medianprops=dict(linestyle='-', linewidth=2),
                ax=ax)

    # Plot the swarmplot on top 
    sns.swarmplot(y="value",
                  x="cond",
                  data=data,
                #   color="auto",
                  s=point_size,  # Circle size
                  palette=swarmplot_palette,
                  #hue="cond",
                  #legend=False,
                  ax=ax)

    # Calculate pairwise stats
    comparisons = [('HC', 'MCI'), ('HC', 'AD'), ('MCI', 'AD')]
    y_max = data['value'].max()
    y_min = data['value'].min()
    y_range = y_max - y_min
    base_y = y_max + 0.05 * y_range  # Increased offset for comparison lines
    height = 0.02 * y_range  # Increased height for comparison lines
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.25 * y_range)

    for i, (group1, group2) in enumerate(comparisons):
        x1, x2 = ['HC', 'MCI', 'AD'].index(group1), ['HC', 'MCI', 'AD'].index(group2)
        vals1 = data[data['cond'] == group1]['value']
        vals2 = data[data['cond'] == group2]['value']
        stat_test = stats.ranksums(vals1, vals2)
        p_value = stat_test.pvalue

        if p_value < 0.00001:
            asterisks = '*****'
        elif p_value < 0.0001:
            asterisks = '****'
        elif p_value < 0.001:
            asterisks = '***'
        elif p_value < 0.01:
            asterisks = '**'
        elif p_value < 0.05:
            asterisks = '*'
        else:
            asterisks = '(n.s.)'

        y = base_y + i * (height / 1)  # Reduce height by half
        ax.plot([x1, x1, x2, x2], [y - 0.00001, y + 0.00001, y + 0.00001, y - 0.00001], lw=1.5, color='black')  # Adjust line height
        ax.text((x1 + x2) / 2, y + 0.00005, asterisks if p_value < 0.05 else f'p = {p_value:.3f}', 
            ha='center', va='bottom', fontsize=12)
    
    # Change axis labels, ticks, and title
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["HC", "MCI", "AD"])
    ax.set_xlabel('')
    ax.set_ylabel(metric)
    ax.set_title(plot_title)

    # Add horizontal grid
    if xgrid:
        ax.grid(axis='y')
        ax.set_axisbelow(True)

    if saveplot:
        # Save plots
        #plt.savefig(filename + '.pdf', bbox_inches='tight', dpi=dpi)
        #plt.savefig(filename + '.svg', bbox_inches='tight', dpi=dpi)
        #plt.savefig(filename + '.png', bbox_inches='tight', dpi=dpi)
        plt.tight_layout()
        plt.show()
        plt.close()
