fixed_dates = [
    '2018-08-02 00:00:00',
    '2018-10-04 00:00:00',
    '2018-10-17 00:00:00',
    '2018-10-24 00:00:00',
    '2018-12-11 00:00:00',
]

def predefined_labels(axs, df): #Ok gia  fig, axs = plt.subplots(1, sharex=True, gridspec_kw={'hspace': 0.1}, figsize=(20, 3))

    ids = []
    for d in fixed_dates:
        axs.axvline(x=df.index.get_loc(d), ls="dashed", color='darkgreen')
        ids.append(df.index.get_loc(d))
    return ids


def convert_labels_to_dt(labels, df): 
    l = []
    for label in labels:
        if(len(df) >= round(label)):
            l.append(df.index[round(label) ].strftime("%Y-%m-%d"))
        else:
            l.append(None)
            #l = [ df.index[round(label)].strftime("%Y-%m-%d") for label in labels]
    return l



def get_fixed_dates(df): #for sorted data
    dates = []
    for date_point in fixed_dates:
        date_point = datetime.strptime(date_point, '%Y-%m-%d %H:%M:%S')
        start_date_d = df.index[0]
        end_date_d = df.index[-1]
        if (date_point >= start_date_d) and (date_point <= end_date_d):
            dates.append(df.index.get_loc(date_point))
    return np.array(dates)


def plot_knee(mps, save_plot=False, filename='knee.png'):
    
    """ Plot the minimum value of the matrix profile for each dimension. This plot is used to visually look for a 'knee' or 'elbow' that
    can be used to find the optimal number of dimensions to use.
    :param mps: The multi-dimensional matrix profile. Each row of the array corresponds to each matrix profile for a given dimension 
                (i.e., the first row is the 1-D matrix profile and the second row is the 2-D matrix profile).
    :param save_plot: If save_plot is True then the figure will be saved. Otherwise it will just be shown.
    :param filename: Used if save_plot=True, the name of the file to be saved.
    """
    
    motifs_idx = np.argsort(mps, axis=1)[:, :2]
    mp_len = mps.shape[0]
    plt.figure(figsize=(15, 5), dpi=80)
    plt.xlabel('k (Number of dimensions, zero-indexed)', fontsize='20')
    plt.ylabel('Matrix Profile Min Value', fontsize='20')
    plt.xticks(range(mp_len))
    plt.plot(mps[range(mp_len), motifs_idx[:mp_len, 0]], c='red', linewidth='4');
    if save_plot:
        plt.savefig(filename)
    else:
        plt.show()
    return

def visualize_md(profile):
    
    """
    'Custom function to visualise multidimensional motifs'
    Automatically creates plots for the provided data structure. In some cases
    many plots are created. For example, when a MatrixProfile is passed with
    corresponding motifs and discords, the matrix profile, discords and motifs
    will be plotted.
    Parameters
    ----------
    profile : dict_like
        A MatrixProfile, Pan-MatrixProfile or Statistics data structure.
    Returns
    -------
    list : figures
        A list of matplotlib figures.
    """
    figures = []

    if not is_visualizable(profile):
        raise ValueError('MatrixProfile, Pan-MatrixProfile or Statistics data structure expected!')

    # plot MP
    if core.is_mp_obj(profile):
        figures = __combine(figures, plot_mp_md(profile))

        if 'cmp' in profile and len(profile['cmp']) > 0:
            figures = __combine(figures, plot_cmp_mp(profile))

        if 'av' in profile and len(profile['av']) > 0:
            figures = __combine(figures, plot_av_mp(profile))

        if 'motifs' in profile and len(profile['motifs']) > 0:
            figures = __combine(figures, plot_motifs_mp_md(profile))

    return figures


def plot_profile(df, mp, col_name): #TODO
    """
    Plot the 'Column_name' graph and the corresponding Profile. We denote with the black arrow to top motif in our set
    Parameters
    ----------
    df : A pandas DataFrame/Time Series
    mp : matrix profile distances
    col_name: 
    Returns
    -------
    list : figures
        A list of matplotlib figures.
    """
    motifs_idx = np.argsort(mp)[:2]
    mp_len = mp.shape[0]
    with sns.axes_style('ticks'):
        fig1, axs1 = plt.subplots(1, sharex=True, gridspec_kw={'hspace': 0}, figsize=(20, 6))
        df.iloc['col_name'].plot(ax=axs1, figsize=(20, 6))

        fig2, axs2 = plt.subplots(1, sharex=True, gridspec_kw={'hspace': 0}, figsize=(20, 6))


        axs2.set_ylabel(f'MP-', fontsize='20')
        axs2.plot(mp, c='orange')
        axs2.set_xlabel('Time', fontsize ='20')

        axs2.plot(motifs_idx, mp[motifs_idx] + 1, marker="v", markersize=10, color='black')
        axs2.plot(motifs_idx, mp[motifs_idx] + 1, marker="v", markersize=10, color='black')
        
def plot_profile_md(df,col_name,mp): ##TODO
    """
    Plot the 'Column_name' graph and the corresponding Profile. We denote with the black arrow to top motif in our set
    Parameters
    ----------
    df : A pandas DataFrame/Time Series
    mp : matrix profile distances
    col_name: 
    Returns
    -------
    list : figures
        A list of matplotlib figures.
    """
    
    motifs_idx = np.argsort(mp, axis=1)[:, :2]
    with sns.axes_style('ticks'):
        fig1, axs1 = plt.subplots(1, sharex=True, gridspec_kw={'hspace': 0}, figsize=(20, 12))
        df['active power'].plot(ax=axs1, figsize=(20, 2))

        fig2, axs2 = plt.subplots(mp['BEBEZE01'].shape[0], sharex=True, gridspec_kw={'hspace': 0}, figsize=(20, 12))
        for k, dim_name in enumerate(cols):
            if np.all(np.isinf(mp['BEBEZE01'][k])): continue

            axs2[k].set_ylabel(f'MP-{dim_name}', fontsize='10')
            axs2[k].plot(mp['BEBEZE01'][k], c='orange')
            axs2[k].set_xlabel('Time', fontsize ='20')

        axs2[k].plot(motifs_idx[k, 0], mp['BEBEZE01'][k, motifs_idx[k, 0]] + 1, marker="v", markersize=10, color='black')
        axs2[k].plot(motifs_idx[k, 1], mp['BEBEZE01'][k, motifs_idx[k, 1]] + 1, marker="v", markersize=10, color='black')
        
        
def segmforeachcolumn(df,output,column_name): 
    with sns.axes_style('ticks'):
    fig, axs = plt.subplots(1, sharex=True, gridspec_kw={'hspace': 0}, figsize=(20, 3))
    df['column_name'].plot(ax=axs, x_compat=True)

    for k, val in output.items():
        fig, axs = plt.subplots(len(L), sharex=True, gridspec_kw={'hspace': 0.1}, figsize=(20, 3 * len(L)))    
        for idx, (cac, regime_locations) in enumerate(val):
            axs[idx].plot(np.arange(0, cac.shape[0]), cac, color='C1')
            
            axs[idx].set_ylabel(f'{str(L[idx])} ', fontsize=18)
            for regime in regime_locations:
                axs[idx].axvline(x=regime, linestyle=":")

            plt.minorticks_on()

            predefined_labels(axs[idx], df)

        labels = [item for item in axs[len(L) - 1].get_xticks()]
        #visible = convert_labels_to_dt(labels[1:-1], df[start_date:end_date].resample(offset).mean())
        locs, _ = plt.xticks()
        #plt.xticks(ticks=locs[1:-1], labels=visible, rotation=30)
        #ax.set_xticklabels(labels)
    
        plt.suptitle(f'{k}-dimension', fontsize=20)
        
def plotregimes univariate: # TODO