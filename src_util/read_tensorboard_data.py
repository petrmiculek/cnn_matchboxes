from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import ticker as mpl_ticker

import seaborn as sns
from scipy import stats
import tensorboard as tb

# inspired by: https://www.tensorflow.org/tensorboard/dataframe_api

if __name__ == '__main__':
    major_ver, minor_ver, _ = version.parse(tb.__version__).release
    assert major_ver >= 2 and minor_ver >= 3, \
        "This notebook requires TensorBoard 2.3 or later."
    print("TensorBoard version: ", tb.__version__)

    # full link: https://tensorboard.dev/experiment/9NMXjp5vSLaB0sal63GoDA
    experiment_id = "9NMXjp5vSLaB0sal63GoDA"
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()

    runs = df["run"].unique()
    tags = df["tag"].unique()
    steps = df["step"].unique()

    dfw = experiment.get_scalars(pivot=False)  # Different from tutorial

    """ Accuracy """
    dfw.reset_index(inplace=True)

    accu = dfw[dfw.tag.isin(['epoch_accu'])]

    # accuracy and PR-value are saved differently from custom-logged metrics
    run_mapping = {
        'bg1000/64x_d1-3-5-7-9-11-1-1_2021-05-10-05-53-28_full/train': 'training',
        'bg1000/64x_d1-3-5-7-9-11-1-1_2021-05-10-05-53-28_full/validation': 'validation',
    }

    accu['run'].replace(run_mapping, inplace=True)

    sns.set_context('talk')
    sns.set_style('darkgrid')
    # sns.set(rc={'figure.figsize': (10, 4)})
    f3, ax3 = plt.subplots(figsize=(10, 6))
    ax_accu1 = sns.lineplot(x='step', y='value',
                     hue='run', data=accu,
                     # kind='point',
                     legend=False,
                     # height=6, aspect=10 / 6
                     )
    plt.legend(ax_accu1.lines, ['training', 'validation'], loc='lower right')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylim(0.95, 1.0)
    plt.xlim(0, 80)
    plt.tight_layout()
    plt.savefig('05-53/accu_095.svg')

    plt.show()

    """ PR-value """

    pr = dfw[dfw.tag.isin(['epoch_pr_value'])]

    # accuracy and PR-value are saved differently from custom-logged metrics
    run_mapping = {
        'bg1000/64x_d1-3-5-7-9-11-1-1_2021-05-10-05-53-28_full/train': 'training',
        'bg1000/64x_d1-3-5-7-9-11-1-1_2021-05-10-05-53-28_full/validation': 'validation',
    }

    pr['run'].replace(run_mapping, inplace=True)

    sns.set_context('talk')
    sns.set_style('darkgrid')
    # sns.set(rc={'figure.figsize': (10, 4)})
    f4, ax4 = plt.subplots(figsize=(10, 6))
    ax_pr = sns.lineplot(x='step', y='value',
                         hue='run', data=pr,
                         legend=False,
                         )
    plt.legend(ax_pr.lines, ['training', 'validation'], loc='lower right')
    plt.title('PR-value')
    plt.xlabel('epoch')
    plt.ylim(0.8, 1.0)
    plt.xlim(0, 80)

    plt.tight_layout()
    plt.savefig('05-53/pr_value_080.svg')

    plt.show()


    """ Counting Mean Average Error """

    count_mae = dfw[dfw.tag.isin(['crate_count_mae_train', 'crate_count_mae_val'])]

    tag_mapping = {
        'crate_count_mae_train': 'training',
        'crate_count_mae_val' : 'validation',
    }
    count_mae['tag'].replace(tag_mapping, inplace=True)

    sns.set_context('talk')
    sns.set_style('darkgrid')
    # sns.set(rc={'figure.figsize': (10, 4)})
    f5, ax5 = plt.subplots(figsize=(10, 6))
    ax_count_mae = sns.lineplot(x='step', y='value',
                         hue='tag', data=count_mae,
                         legend=False,
                         )
    plt.legend(ax_count_mae.lines, ['training', 'validation'], loc='upper right')
    plt.title('Crate Counting Mean Average Error')
    plt.xlabel('epoch')
    # plt.ylim(0.0, 1.0)
    plt.xlim(0, 85)

    plt.tight_layout()
    plt.savefig('05-53/crate_counting_mae.svg')

    plt.show()


    """ Counting Failrate """
    count_failrate = dfw[dfw.tag.isin(['crate_count_failrate_train', 'crate_count_failrate_val'])]

    tag_mapping = {
        'crate_count_failrate_train': 'training',
        'crate_count_failrate_val': 'validation',
    }
    count_failrate['tag'].replace(tag_mapping, inplace=True)

    f6, ax6 = plt.subplots(figsize=(10, 6))
    ax_count_failrate = sns.lineplot(x='step', y='value',
                         hue='tag', data=count_failrate,
                         legend=False,
                         )
    plt.legend(ax_count_failrate.lines, ['training', 'validation'], loc='upper right')
    plt.title('Crate Counting Failure Rate')
    plt.xlabel('epoch')
    plt.tight_layout()
    plt.xlim(0, 85)
    plt.ylim(0, 1.05)


    plt.savefig('05-53/crate_counting_failrate.svg')
    plt.show()

    """ Point Count MAE """
    point_count_mae = dfw[dfw.tag.isin(['point_count_mae_train', 'point_count_mae_val'])]

    tag_mapping = {
        'point_count_mae_train': 'training',
        'point_count_mae_val': 'validation',
    }
    point_count_mae['tag'].replace(tag_mapping, inplace=True)

    f0, ax0 = plt.subplots(figsize=(10, 6))
    ax_point_count = sns.lineplot(x='step', y='value',
                         hue='tag', data=point_count_mae,
                         legend=False,
                         )
    plt.legend(ax_point_count.lines, ['training', 'validation'], loc='upper right')
    plt.title('Keypoint Counting Mean Average Error')
    plt.xlabel('epoch')
    plt.tight_layout()
    # plt.xlim(0, 85)
    # plt.ylim(0, 1.05)
    plt.ylim(bottom=0)

    plt.savefig('05-53/keypoint_counting_mae.svg')
    plt.show()

    """ Keypoint-wise Distance MSE """
    point_dist_mse = dfw[dfw.tag.isin(['point_dist_mse_train', 'point_dist_mse_val'])]

    tag_mapping = {
        'point_dist_mse_train': 'training',
        'point_dist_mse_val': 'validation',
    }
    point_dist_mse['tag'].replace(tag_mapping, inplace=True)

    f1, ax1 = plt.subplots(figsize=(10, 6))
    ax_point_dist = sns.lineplot(x='step', y='value',
                         hue='tag', data=point_dist_mse,
                         legend=False,
                         )
    # ax_point_dist.set_yscale('log')
    ax_point_dist.yaxis.set_major_formatter(mpl_ticker.EngFormatter())

    plt.legend(ax_point_dist.lines, ['training', 'validation'], loc='upper right')
    plt.title('Keypoint-wise Distance Mean Square Error')
    plt.xlabel('epoch')
    plt.tight_layout()
    # plt.xlim(0, 85)
    # plt.ylim(0, 1.05)
    plt.ylim(bottom=0)

    plt.savefig('05-53/keypoint_dist_mse.svg')
    plt.show()

    """ Pixel Distance MSE """

    point_dist_mse = dfw[dfw.tag.isin(['pix_mse_train', 'pix_mse_val'])]

    tag_mapping = {
        'pix_mse_train': 'training',
        'pix_mse_val': 'validation',
    }
    point_dist_mse['tag'].replace(tag_mapping, inplace=True)

    f2, ax2 = plt.subplots(figsize=(10, 6))
    ax_pix = sns.lineplot(x='step', y='value',
                         hue='tag', data=point_dist_mse,
                         legend=False,
                         )

    ax_pix.set_yscale('log')
    # show Y axis labels as integers (not in scientific notation)
    # ax_pix.yaxis.set_major_formatter(mpl_ticker.ScalarFormatter())

    plt.legend(ax_pix.lines, ['training', 'validation'], loc='upper right')
    plt.title('Pixel-wise Distance Mean Square Error')
    plt.xlabel('epoch')
    # plt.xlim(0, 85)
    # plt.ylim(0, 1.05)
    # plt.ylim(bottom=0.01)

    plt.tight_layout()
    plt.savefig('05-53/pix_mse.svg')
    plt.show()

    """ UNUSED """

    if False:
        csv_path = '/tmp/tb_experiment_1.csv'
        dfw.to_csv(csv_path, index=False)
        dfw_roundtrip = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(dfw_roundtrip, dfw)

        # Filter the DataFrame to only validation data, which is what the subsequent
        # analyses and visualization will be focused on.
        dfw_validation = dfw[dfw.run.str.endswith("/validation")]
        # Get the optimizer value for each row of the validation DataFrame.
        optimizer_validation = dfw_validation.run.apply(lambda run: run.split(",")[0])

        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        sns.lineplot(data=dfw_validation, x="step", y="epoch_accuracy",
                     hue=optimizer_validation).set_title("accuracy")
        plt.subplot(1, 2, 2)
        sns.lineplot(data=dfw_validation, x="step", y="epoch_loss",
                     hue=optimizer_validation).set_title("loss")

        adam_min_val_loss = dfw_validation.loc[optimizer_validation == "adam", :].groupby(
            "run", as_index=False).agg({"epoch_loss": "min"})
        rmsprop_min_val_loss = dfw_validation.loc[optimizer_validation == "rmsprop", :].groupby(
            "run", as_index=False).agg({"epoch_loss": "min"})
        sgd_min_val_loss = dfw_validation.loc[optimizer_validation == "sgd", :].groupby(
            "run", as_index=False).agg({"epoch_loss": "min"})
        min_val_loss = pd.concat([adam_min_val_loss, rmsprop_min_val_loss, sgd_min_val_loss])

        sns.boxplot(data=min_val_loss, y="epoch_loss",
                    x=min_val_loss.run.apply(lambda run: run.split(",")[0]))

    if False:
        dir = '/home/petrmiculek/Desktop/logs_new/logs_best_only/bg1000/64x_d1-3-5-7-9-11-1-1_2021-05-10-05-53-28_full'
        train = 'train/events.out.tfevents.1620618812.black1.cerit-sc.cz.4250.4758082.v2'
        val = 'validation/events.out.tfevents.1620619323.black1.cerit-sc.cz.4250.4768816.v2'
        for e in tf.compat.v1.train.summary_iterator(os.path.join(dir, val)):
            for v in e.summary.value:
                if v.tag == 'epoch_pr_value':
                    print(e.step, v.simple_value)


