import numpy as np
from numpy import interp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve, roc_auc_score

from numpy import typing as npt
from typing import Dict, Optional, Sequence


def get_psg_downsampled(reading, time_interval=30, binary_class=True):
    data = reading.datasets["accelerometer"].data.copy()

    # remove non psg labed data
    is_psg = data["sleepstate"] != "NaN"
    data = data[is_psg]

    # downsample psg to get every 30 sec
    psg_results = (
        data["sleepstate"].resample(f"{time_interval}S", origin="start").ffill().bfill()
    )

    # relable to binary
    if binary_class:
        psg_results[psg_results == "W"] = 0
        psg_results[psg_results.isin(["N1", "N2", "N3", "R"])] = 1

    return psg_results


##################################
# Below adapted from palotti
##################################
def extract_x_y(df, seq_len, subjid):
    df = df[df["subjid"] == subjid][["activity", "psg"]].copy()
    y = df.pop("psg")

    vals = []
    for s in range(1, seq_len // 2 + 1):  # range (1, 2) -> [1]
        vals.append(df["activity"].shift(s))
    vals.append(df["activity"])
    for s in range(1, seq_len // 2 + 1):
        vals.append(df["activity"].shift(-s))

    df2 = pd.concat(vals, axis=1)
    x = df2.fillna(-1).values
    return x, y


def get_data(df, seq_len, ):
    subjids = df.subjid.unique()
    x_, y_ = [], []
    for sid in subjids:
        x_tmp, y_tmp = extract_x_y(df, seq_len, sid)
        x_.append(x_tmp)
        y_.append(y_tmp)
    x_ = np.vstack(x_)
    y_ = np.hstack(y_)
    return x_, y_


def extract_x_y_old(df, seq_len, subjid):
    df = df[df["subjid"] == subjid][["activity", "psg"]].copy()
    y = df.pop("psg")

    vals = [df["activity"]]
    for s in range(1, seq_len // 2 + 1):
        vals.append(df["activity"].shift(s))
    for s in range(1, seq_len // 2 + 1):
        vals.append(df["activity"].shift(-s))

    df2 = pd.concat(vals, axis=1)
    x = df2.fillna(-1).values
    return x, y


def get_data_old(df, seq_len):
    subjids = df.subjid.unique()
    x_, y_ = [], []
    for sid in subjids:
        x_tmp, y_tmp = extract_x_y_old(df, seq_len, sid)
        x_.append(x_tmp)
        y_.append(y_tmp)
    x_ = np.vstack(x_)
    y_ = np.hstack(y_)
    return x_, y_


def cole(df):
    """
    Cole method to classify sleep vs awake
    """
    df["_A0"] = df["activity"]
    for i in range(1, 5):
        df["_A-%d" % (i)] = df["activity"].shift(i).fillna(0.0)
    for i in range(1, 3):
        df["_A+%d" % (i)] = df["activity"].shift(-i).fillna(0.0)

    w_m4, w_m3, w_m2, w_m1, w_0, w_p1, w_p2 = [404, 598, 326, 441, 1408, 508, 350]
    p = 0.00001

    cole = p * (
        w_m4 * df["_A-4"]
        + w_m3 * df["_A-3"]
        + w_m2 * df["_A-2"]
        + w_m1 * df["_A-1"]
        + w_0 * df["_A0"]
        + w_p1 * df["_A+1"]
        + w_p2 * df["_A+2"]
    )

    # Remove temporary variables
    del df["_A0"]
    for i in range(1, 5):
        del df["_A-%d" % (i)]
    for i in range(1, 3):
        del df["_A+%d" % (i)]

    # return (cole < 1.0).astype(int)
    return cole, (cole < 1.0).astype(int)


def kripke(df, scaler=0.204):
    """
    Kripke formula as shown in Kripke et al. 2010 paper
    """
    for i in range(1, 11):
        df["_a-%d" % (i)] = df["activity"].shift(i).fillna(0.0)
        df["_a+%d" % (i)] = df["activity"].shift(-i).fillna(0.0)

    kripke = scaler * (
        0.0064 * df["_a-10"]
        + 0.0074 * df["_a-9"]
        + 0.0112 * df["_a-8"]
        + 0.0112 * df["_a-7"]
        + 0.0118 * df["_a-6"]
        + 0.0118 * df["_a-5"]
        + 0.0128 * df["_a-4"]
        + 0.0188 * df["_a-3"]
        + 0.0280 * df["_a-2"]
        + 0.0664 * df["_a-1"]
        + 0.0300 * df["activity"]
        + 0.0112 * df["_a+1"]
        + 0.0100 * df["_a+2"]
    )

    for i in range(1, 11):
        del df["_a+%d" % (i)]
        del df["_a-%d" % (i)]

    # return (kripke < 1.0).astype(int)
    return kripke, (kripke < 1.0).astype(int)


def sazonov(df):
    """
    Sazonov formula as shown in the original paper
    """
    for w in range(1, 6):
        df["_w%d" % (w - 1)] = df["activity"].rolling(window=w, min_periods=1).max()

    sazonov = (
        1.727
        - 0.256 * df["_w0"]
        - 0.154 * df["_w1"]
        - 0.136 * df["_w2"]
        - 0.140 * df["_w3"]
        - 0.176 * df["_w4"]
    )

    for w in range(1, 6):
        del df["_w%d" % (w - 1)]

    # return (sazonov >= 0.5).astype(int)
    return sazonov, (sazonov >= 0.5).astype(int)


def sazonov2(df):
    """
    Sazonov formula as shown in Tilmanne et al. 2009 paper
    """
    for w in range(1, 10):
        df["_w%d" % (w - 1)] = df["activity"].rolling(window=w, min_periods=1).max()

    sazonov = (
        1.99604
        - 0.1945 * df["_w0"]
        - 0.09746 * df["_w1"]
        - 0.09975 * df["_w2"]
        - 0.10194 * df["_w3"]
        - 0.08917 * df["_w4"]
        - 0.08108 * df["_w5"]
        - 0.07494 * df["_w6"]
        - 0.07300 * df["_w7"]
        - 0.10207 * df["_w8"]
    )

    for w in range(1, 10):
        del df["_w%d" % (w - 1)]

    sazonov = 1 / (1 + np.exp(-sazonov))

    # return (sazonov >= 0.5).astype(int)
    return sazonov, (sazonov >= 0.5).astype(int)


def sadeh(df, min_value=0):
    """
    Sadeh model for classifying sleep vs active
    """
    window_past = 6
    window_nat = 11
    window_centered = 11

    df["_mean"] = (
        df["activity"]
        .rolling(window=window_centered, center=True, min_periods=1)
        .mean()
    )
    df["_std"] = df["activity"].rolling(window=window_past, min_periods=1).std()
    df["_nat"] = (
        ((df["activity"] >= 50) & (df["activity"] < 100))
        .rolling(window=window_nat, center=True, min_periods=1)
        .sum()
    )

    df["_LocAct"] = (df["activity"] + 1.0).apply(np.log)

    sadeh = (
        7.601
        - 0.065 * df["_mean"]
        - 0.056 * df["_std"]
        - 0.0703 * df["_LocAct"]
        - 1.08 * df["_nat"]
    )

    del df["_mean"]
    del df["_std"]
    del df["_nat"]
    del df["_LocAct"]

    # return (sadeh > min_value).astype(int)
    return sadeh, (sadeh > min_value).astype(int)


def oakley(df, threshold=80):
    """
    Oakley method to class sleep vs active/awake
    """
    for i in range(1, 5):
        df["_a-%d" % (i)] = df["activity"].shift(i).fillna(0.0)
        df["_a+%d" % (i)] = df["activity"].shift(-i).fillna(0.0)

    oakley = (
        0.04 * df["_a-4"]
        + 0.04 * df["_a-3"]
        + 0.20 * df["_a-2"]
        + 0.20 * df["_a-1"]
        + 2.0 * df["activity"]
        + 0.20 * df["_a+1"]
        + 0.20 * df["_a-2"]
        + 0.04 * df["_a-3"]
        + 0.04 * df["_a-4"]
    )

    for i in range(1, 5):
        del df["_a+%d" % (i)]
        del df["_a-%d" % (i)]

    # return (oakley <= threshold).astype(int)
    return oakley, (oakley <= threshold).astype(int)


def webster(df):
    """
    Webster method to classify sleep from awake
    """
    df["_A0"] = df["activity"]
    for i in range(1, 5):
        df["_A-%d" % (i)] = df["activity"].shift(i).fillna(0.0)
    for i in range(1, 3):
        df["_A+%d" % (i)] = df["activity"].shift(-i).fillna(0.0)

    w_m4, w_m3, w_m2, w_m1, w_0, w_p1, w_p2 = [0.15, 0.15, 0.15, 0.08, 0.21, 0.12, 0.13]
    p = 0.025

    webster = p * (
        w_m4 * df["_A-4"]
        + w_m3 * df["_A-3"]
        + w_m2 * df["_A-2"]
        + w_m1 * df["_A-1"]
        + w_0 * df["_A0"]
        + w_p1 * df["_A+1"]
        + w_p2 * df["_A+2"]
    )

    # Remove temporary variables
    del df["_A0"]
    for i in range(1, 5):
        del df["_A-%d" % (i)]
    for i in range(1, 3):
        del df["_A+%d" % (i)]

    # return (webster < 1.0).astype(int)
    return webster, (webster < 1.0).astype(int)


def webster_rescoring_rules(s, rescoring_rules="abcde"):

    haveAppliedAnyOtherRule = False

    if "a" in rescoring_rules or "A" in rescoring_rules:
        # After at least 4 minutes scored as wake, next minute scored as sleep is rescored wake
        # print "Processing rule A"
        maskA = (
            s.shift(1).rolling(window=4, center=False, min_periods=1).sum() > 0
        )  # avoid including actual period
        result = s.where(maskA, 0)
        haveAppliedAnyOtherRule = True

    if "b" in rescoring_rules or "B" in rescoring_rules:
        # After at least 10 minutes scored as wake, the next 3 minutes scored as sleep are rescored wake
        # print "Processing rule B"
        if (
            haveAppliedAnyOtherRule == True
        ):  # if this is true, I need to apply the next operation on the destination col
            s = result

        maskB = (
            s.shift(1).rolling(window=10, center=False, min_periods=1).sum() > 0
        )  # avoid including actual period
        result = s.where(maskB, 0).where(maskB.shift(1), 0).where(maskB.shift(2), 0)
        haveAppliedAnyOtherRule = True

    if "c" in rescoring_rules or "C" in rescoring_rules:
        # After at least 15 minutes scored as wake, the next 4 minutes scored as sleep are rescored as wake
        # print "Processing rule C"
        if (
            haveAppliedAnyOtherRule == True
        ):  # if this is true, I need to apply the next operation on the destination col
            s = result

        maskC = (
            s.shift(1).rolling(window=15, center=False, min_periods=1).sum() > 0
        )  # avoid including actual period
        result = (
            s.where(maskC, 0)
            .where(maskC.shift(1), 0)
            .where(maskC.shift(2), 0)
            .where(maskC.shift(3), 0)
        )
        haveAppliedAnyOtherRule = True

    if "d" in rescoring_rules or "D" in rescoring_rules:
        # 6 minutes or less scored as sleep surroundeed by at least 10 minutes (before or after) scored as wake are rescored wake
        # print "Processing rule D"
        if (
            haveAppliedAnyOtherRule == True
        ):  # if this is true, I need to apply the next operation on the destination col
            s = result

        # First Part
        maskD1 = (
            s.shift(1).rolling(window=10, center=False, min_periods=1).sum() > 0
        )  # avoid including actual period
        tmpD1 = s.where(maskD1.shift(5), 0)
        haveAppliedAnyOtherRule = True

        # Second Part: sum the next 10 periods and replaces previous 6 in case they are all 0's
        maskD2 = (
            s.shift(-10).rolling(window=10, center=False, min_periods=1).sum() > 0
        )  # avoid including actual period
        tmpD2 = s.where(maskD2.shift(-5), 0)

        result = tmpD1 & tmpD2

    if "e" in rescoring_rules or "E" in rescoring_rules:
        # 10 minutes or less scored as sleep surrounded by at least 20 minutes (before or after) scored as wake are rescored wake
        # print "Processing rule E"
        if (
            haveAppliedAnyOtherRule == True
        ):  # if this is true, I need to apply the next operation on the destination col
            s = result

        # First Part
        maskE1 = (
            s.shift(1).rolling(window=20, center=False, min_periods=1).sum() > 0
        )  # avoid including actual period
        tmpE1 = s.where(maskE1.shift(9), 0)

        # Second Part: sum the next 10 periods and replaces previous 6 in case they are all 0's
        maskE2 = (
            s.shift(-20).rolling(window=20, center=False, min_periods=1).sum() > 0
        )  # avoid including actual period
        tmpE2 = s.where(maskE2.shift(-9), 0)

        result = tmpE1 & tmpE2

    return result


class BlandAltman:
    def __init__(self, gold_std, new_measure, averaged=False):
        # set averaged to True if multiple observations from each participant are averaged together to get one value
        import pandas as pd

        # Check that inputs are list or pandas series, convert to series if list
        if isinstance(gold_std, list) or isinstance(gold_std, (np.ndarray, np.generic)):
            df = pd.DataFrame()  # convert to pandas series
            df["gold_std"] = gold_std
            gold_std = df.gold_std
        elif not isinstance(gold_std, pd.Series):
            print(
                "Error: Data type of gold_std is not a list or a Pandas series or Numpy array"
            )

        if isinstance(new_measure, list) or isinstance(
            new_measure, (np.ndarray, np.generic)
        ):
            df2 = pd.DataFrame()  # convert to pandas series
            df2["new_measure"] = new_measure
            new_measure = df2.new_measure
        elif not isinstance(new_measure, pd.Series):
            print(
                "Error: Data type of new_measure is not a list or a Pandas series or Numpy array"
            )

        self.gold_std = gold_std
        self.new_measure = new_measure

        # Calculate Bland-Altman statistics
        diffs = gold_std - new_measure
        self.mean_error = diffs.mean()
        self.mean_absolute_error = diffs.abs().mean()
        self.mean_squared_error = (diffs**2).mean()
        self.root_mean_squared_error = np.sqrt((diffs**2).mean())
        r = np.corrcoef(self.gold_std, self.new_measure)
        self.correlation = r[0, 1]  # correlation coefficient
        diffs_std = diffs.std()  # 95% Confidence Intervals
        corr_std = np.sqrt(
            2 * (diffs_std**2)
        )  # if observations are averaged, used corrected standard deviation
        if averaged:
            self.CI95 = [
                self.mean_error + 1.96 * corr_std,
                self.mean_error - 1.96 * corr_std,
            ]
        else:
            self.CI95 = [
                self.mean_error + 1.96 * diffs_std,
                self.mean_error - 1.96 * diffs_std,
            ]

    def print_stats(self, round_amount=5):
        print("Mean error = {}".format(round(self.mean_error, round_amount)))
        print(
            "Mean absolute error = {}".format(
                round(self.mean_absolute_error, round_amount)
            )
        )
        print(
            "Mean squared error = {}".format(
                round(self.mean_squared_error, round_amount)
            )
        )
        print(
            "Root mean squared error = {}".format(
                round(self.root_mean_squared_error, round_amount)
            )
        )
        print("Correlation = {}".format(round(self.correlation, round_amount)))
        print("+95% Confidence Interval = {}".format(round(self.CI95[0], round_amount)))
        print("-95% Confidence Interval = {}".format(round(self.CI95[1], round_amount)))

    def return_stats(self):
        # return dict of statistics
        stats_dict = {
            "mean_error": self.mean_error,
            "mean_absolute_error": self.mean_absolute_error,
            "mean_squared_error": self.mean_squared_error,
            "root_mean_squared_error": self.root_mean_squared_error,
            "correlation": self.correlation,
            "CI_95%+": self.CI95[0],
            "CI_95%-": self.CI95[1],
        }

        return stats_dict

    def scatter_plot(
        self,
        x_label="Gold Standard",
        y_label="New Measure",
        figure_size=(4, 4),
        show_legend=True,
        the_title=" ",
        file_name="BlandAltman_ScatterPlot.pdf",
        is_journal=False,
    ):
        import os
        from os import path

        import matplotlib.pyplot as plt
        import numpy as np

        #%matplotlib inline
        if not os.path.exists("output_images"):
            os.mkdir("output_images")

        file_name = "output_images/" + file_name

        if is_journal:  # avoid use of type 3 fonts for journal paper acceptance
            import matploblib

            matplotlib.rcParams["pdf.fonttype"] = 42
            matplotlib.rcParams["ps.fonttype"] = 42

        fig = plt.figure(figsize=figure_size)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.scatter(self.gold_std, self.new_measure, label="Observations")
        x_vals = np.array(ax.get_xlim())
        ax.plot(x_vals, x_vals, "--", color="black", label="Line of Slope = 1")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(the_title)
        ax.grid()
        if show_legend:
            ax.legend()
        plt.savefig(file_name, bbox_inches="tight")

    def difference_plot(
        self,
        x_label="Difference between methods",
        y_label="Average of two methods",
        averaged=False,
        figure_size=(4, 4),
        show_legend=True,
        the_title="",
        file_name="BlandAltman_DifferencePlot.pdf",
        is_journal=False,
    ):

        import os
        from os import path

        import matplotlib.pyplot as plt
        import numpy as np

        #%matplotlib inline
        if not os.path.exists("output_images"):
            os.mkdir("output_images")
        file_name = "output_images/" + file_name

        if is_journal:  # avoid use of type 3 fonts for journal paper acceptance
            import matploblib

            matplotlib.rcParams["pdf.fonttype"] = 42
            matplotlib.rcParams["ps.fonttype"] = 42

        diffs = self.gold_std - self.new_measure
        avgs = (self.gold_std + self.new_measure) / 2

        fig = plt.figure(figsize=figure_size)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.scatter(avgs, diffs, label="Observations")
        x_vals = np.array(ax.get_xlim())
        ax.axhline(self.mean_error, color="black", label="Mean Error")
        ax.axhline(
            self.CI95[0],
            color="black",
            linestyle="--",
            label="+95% Confidence Interval",
        )
        ax.axhline(
            self.CI95[1],
            color="black",
            linestyle="--",
            label="-95% Confidence Interval",
        )
        ax.set_ylabel(x_label)
        ax.set_xlabel(y_label)
        ax.set_title(the_title)
        if show_legend:
            ax.legend()
        ax.grid()
        plt.savefig(file_name, bbox_inches="tight")


def output_name(k, snake_outputs):
    """Generate output name based on snakemake.output."""
    for fout in snake_outputs:
        to_replace = k.replace(".h5", ".csv")
        if to_replace in fout:
            return fout
    raise ValueError(f"{to_replace} not present in {snake_outputs}")


##################################
# Below adapted from pywear
##################################

def get_auc(
    y_test: npt.NDArray[np.float32],
    y_predicted: npt.NDArray[np.float32],
    plot: bool = False,
    ax: Optional[plt.axis] = None,
) -> Dict[str, float]:
    """Compute the Area Under the Curve.

    Parameters
    ----------
    y_test: np.array
        Ground truth label data, in the format n_samples x n_classes.
    y_predicted: np.array
        Predicted label data, in the format n_samples x n_classes.
    plot : bool, optional
        if True, the ROC AUC is plotted, by default False.
    ax : plt.axis, optional
        Single axis object to plot onto.

    Returns
    -------
    roc_auc : dict
        Metric values.
    """
    n_classes = y_test.shape[1]
    fpr: Dict[str, npt.NDArray[np.float32]] = dict()
    tpr: Dict[str, npt.NDArray[np.float32]] = dict()
    roc_auc = dict()
    fpr_vec, tpr_vec, roc_auc_vec = [], [], []
    for i in range(n_classes):
        fpr_i, tpr_i, _ = roc_curve(y_test[:, i], y_predicted[:, i])
        fpr_vec.append(fpr_i)
        tpr_vec.append(tpr_i)
        roc_auc_vec.append(auc(fpr_i, tpr_i))

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_predicted.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr: npt.NDArray[np.float32] = np.unique(
        np.concatenate([fpr_vec[i] for i in range(n_classes)])
    )

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr_vec[i], tpr_vec[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    if not plot:
        return roc_auc

    # Plot all ROC curves
    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"AUC = {roc_auc_score(y_test[:,0], y_predicted[:,0]):0.2f}",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver operating characteristic")
    ax.legend(loc="lower right")
    plt.show(block=False)

    return roc_auc


def compute_confusion_matrix(
    y_true: npt.NDArray[np.float32],
    y_pred: npt.NDArray[np.float32],
    labels: Optional[Sequence[str]] = None,
    plot: bool = False,
    plot_labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.axis] = None,
) -> npt.NDArray[np.float32]:
    """Compute confusion matrix.

    Parameters
    ----------
    x_data: np.ndarray
        Data to predict, in the format channels x samples.
    y_data: ndarray
        Target data with a label per observation.
    labels : List, optional
        List of labels for y_data values.
    plot : bool, optional
        if True, the confusion matrix is plotted,  by default False.
    plot_labels: List, optional
        List of labels for plotting the confusion matrix.
    axs : plt.axis, optional
        List of 2 axis objects to plot onto.

    Returns
    -------
    y_true : np.ndarray
        Ground truth correct values.
    """
    results: npt.NDArray[np.float32] = confusion_matrix(y_true, y_pred, labels=labels)

    if not plot:
        return results

    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(12, 4))

    if plot_labels is None:
        plot_labels = list(np.unique(y_true))


    sns.heatmap(
        ((results / np.tile(results.sum(1), (results.shape[1], 1)).T) * 100).astype(
            int
        ),
        xticklabels=plot_labels,
        yticklabels=plot_labels,
        annot=True,
        fmt="d",
        ax=ax,
    )
    # ax.set_title("Confusion matrix ratio predition class")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.show(block=False)

    return results