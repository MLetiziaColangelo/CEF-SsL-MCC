import matplotlib.pyplot as plt
import numpy as np

def plot_wilcoxon(best_pseudo, best_active, pseudo_p, pseudo_z, active_p, active_z):
    underline_character = "\u0332"
    text_with_underline_S = "S" + underline_character
    text_with_underline_s = "s" + underline_character
    text_with_underline_L = "L" + underline_character
    combined_underlined_text = text_with_underline_S + text_with_underline_s + text_with_underline_L
    pi = "\u03C0"
    pi_with_hat = "\u03C0\u0302"


    if best_pseudo == 'baselinessl':
        pseudo_method = combined_underlined_text
    elif best_pseudo == 'pseudohigh':
        pseudo_method = pi+'SsL'
    elif best_pseudo == 'pseudohigh2':
        pseudo_method = pi_with_hat+'SsL'

   
    if best_active == 'activelow':
        active_method = "\u03B1\u0053\u0073\u004C\u2097"
    elif best_active == 'activemid':
        active_method = "\u03B1\u0053\u0073\u004C\u2092"
    elif best_active == 'activehigh':
        active_method = "\u03B1\u0053\u0073\u004C\u2095"
    elif best_active == 'pseudoactivehigh':
        active_method = "\u03B1\u005E\u03C0\u0053\u0073\u004C\u2095"
    elif best_active == 'pseudoactivelow':
        active_method = "\u03B1\u005E\u03C0\u0053\u0073\u004C\u2097"
    elif best_active == 'pseudoactivemid':
        active_method = "\u03B1\u005E\u03C0\u0053\u0073\u004C\u2092"

   
    data = [[pseudo_method, round(pseudo_p, 3), round(pseudo_z, 3)], [active_method, round(active_p, 3), round(active_z, 3)]]
    
    columns = ('Method', 'p-value', 'z-value',
            )
    rows = ("Best 'pure' pseudo-labelling", "Best active learning")

    rcolors = plt.cm.BuPu(np.full(len(rows), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(columns), 0.1))
    fig, ax = plt.subplots(figsize=(8, 4))
    the_table = plt.table(cellText=data,
                        rowLabels=rows,
                        rowColours=rcolors,
                        rowLoc='right',
                        colColours=ccolors,
                        colLabels=columns,
                        colWidths=[0.15, 0.2, 0.2],
                        loc='center')

    the_table.scale(1, 1.5)
    

    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.box(on=None)

def plot_average(avgF1, avgExetime):
    data = [[str(round(avgF1[i],3)), f"{round(avgExetime[i], 3)}(s)"] for i in range(len(avgF1))]


    overline_character = "\u0305"
    text_with_overline_S = "S" + overline_character
    text_with_overline_L = "L" + overline_character
    combined_overlined_text = text_with_overline_S + text_with_overline_L
    prova = u'S\u0305L\u0305'

    underline_character = "\u0332"
    text_with_underline_S = "S" + underline_character
    text_with_underline_s = "s" + underline_character
    text_with_underline_L = "L" + underline_character
    combined_underlined_text = text_with_underline_S + text_with_underline_s + text_with_underline_L
    pi = "\u03C0"
    pi_with_hat = "\u03C0\u0302"

    columns = ('F1-score', 'Execution time',
        )
    rows = (prova, 'SL', combined_underlined_text, pi+'SsL', pi_with_hat+'SsL',
            "\u03B1\u0053\u0073\u004C\u2097", "\u03B1\u0053\u0073\u004C\u2092", "\u03B1\u0053\u0073\u004C\u2095",
            "\u03B1\u005E\u03C0\u0053\u0073\u004C\u2097", "\u03B1\u005E\u03C0\u0053\u0073\u004C\u2092",
                "\u03B1\u005E\u03C0\u0053\u0073\u004C\u2095" )

    rcolors = plt.cm.BuPu(np.full(len(rows), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(columns), 0.1))
    table = plt.table(cellText=data,
                        rowLabels=rows,
                        rowColours=rcolors,
                        rowLoc='right',
                        colColours=ccolors,
                        colLabels=columns,
                        loc='center')

    table.scale(1, 1.5)

    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.box(on=None)



