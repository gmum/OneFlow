import itertools
import torch

import numpy as np

from sklearn.datasets import make_moons, make_blobs, make_circles
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_2d_data(dataset, train=True):
    outliers_fraction = 0.05

    if train:
        n_samples = 10000
        n_outliers = int(outliers_fraction * n_samples)
        n_inliers = n_samples - n_outliers

        rng = np.random.RandomState(42)
        blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
    else:
        n_samples = 1000
        n_outliers = int(outliers_fraction * n_samples)
        n_inliers = n_samples - n_outliers

        rng = np.random.RandomState(43)
        blobs_params = dict(random_state=1, n_samples=n_inliers, n_features=2)

    if dataset == "Donut":
        outliers_fraction = 0.05
        n_outliers = int(outliers_fraction * n_samples)
        n_inliers = n_samples - n_outliers
        noisy_circles = make_circles(n_samples=n_inliers, factor=0.99999999, noise=.10)[0]
        noisy_blob = make_blobs(n_samples=n_outliers, centers=[[0., 0.]], cluster_std=[[np.sqrt(0.05)]], shuffle=False)[0]
        points = np.concatenate([noisy_circles, noisy_blob], axis=0)
    if dataset == "Diverse Blobs":
        points = make_blobs(centers=[[0., 0.], [4., 4.]], cluster_std=[1, np.sqrt(0.4)], n_samples=[n_inliers, n_outliers], shuffle=False)[0]
    if dataset == "Two Blobs":
        points = make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5], **blobs_params)[0]
        points = np.concatenate([points, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)
    if dataset == "Moons":
        points = 4. * (make_moons(n_samples=n_inliers, noise=.05, random_state=0)[0] - np.array([0.5, 0.25]))
        points = np.concatenate([points, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)
    if dataset == "Big Uniform":
        points = 14. * (np.random.RandomState(42).rand(n_inliers, 2) - 0.5)
        points = np.concatenate([points, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)
        
    return points


def draw_from_circle(num_points, radius):
    phis = np.arange(0, 2*(np.pi), (2*(np.pi))/num_points) 
    r = radius
    x = r * np.cos(phis)
    y = r * np.sin(phis)

    X = np.column_stack((x, y))
    return X


def compare_flow_models_2d(our_model, flow_model, X_test):
    outliers_percentages = [0.05]
    colors = []
    test_size = X_test.shape[0]
    border_elements = []
    our_outliers_radiuses = []
    flow_outliers_radiuses = []
    N = 2
    colormap = plt.cm.RdBu_r
    
    for percentage in outliers_percentages:
        border_elements.append(int(test_size*(1-percentage)))

    our_model.eval()
    with torch.no_grad():
        our_samples, _ = our_model.flow(X_test)

        our_r_test = torch.sqrt(torch.sum(our_samples ** 2, axis=1))
        our_r_sorted = torch.sort(our_r_test, descending=False)[0]

    for border in border_elements:
        our_outliers_radiuses.append(our_r_sorted[border].item())

    flow_model.eval()
    with torch.no_grad():
        flow_samples, _ = flow_model.flow(X_test)

        flow_r_test = torch.sqrt(torch.sum(flow_samples ** 2, axis=1))
        flow_r_sorted = torch.sort(flow_r_test, descending=False)[0]

    for border in border_elements:
        flow_outliers_radiuses.append(flow_r_sorted[border].item())

    with torch.no_grad():
        alphas = [0.6]
        linewidths = [2]
        outliers_colors = ['red']

        for alpha, linewidth, outlier_color in itertools.product(alphas, linewidths, outliers_colors):
            for i in range(len(outliers_percentages)):
                plt.figure(figsize=(12,7))
                X_test_inliers = X_test[950:,:]
                X_test_outliers = X_test[:950,:]
                X_test_inliers = X_test_inliers.cpu().numpy()
                X_test_outliers = X_test_outliers.cpu().numpy()
                plt.scatter(X_test[950:,0].cpu().numpy(), X_test[950:,1].cpu().numpy(), c=outlier_color, alpha=alpha, label="Test - outliers")
                plt.scatter(X_test[:950,0].cpu().numpy(), X_test[:950,1].cpu().numpy(), c='black', alpha=alpha, label="Test - inliers")
                for j, color in zip(range(2), iter(colormap(np.linspace(0,1,int(N))))):
                    if j == 0:
                        latent_X = draw_from_circle(num_points=10000, radius=our_outliers_radiuses[i])
                        latent_X = torch.from_numpy(latent_X).double().to(device)
                        X = our_model.inv_flow(latent_X)
                        plt.plot(X[:,0].cpu().numpy(), X[:,1].cpu().numpy(), c=color, linewidth=linewidth, label="OneFlow: "+str(outliers_percentages[i]))
                    else:
                        latent_Y = draw_from_circle(num_points=10000, radius=flow_outliers_radiuses[i])
                        latent_Y = torch.from_numpy(latent_Y).double().to(device)
                        Y = flow_model.inv_flow(latent_Y)
                        plt.plot(Y[:,0].cpu().numpy(), Y[:,1].cpu().numpy(), c=color, linewidth=linewidth, label="LL-Flow: "+str(outliers_percentages[i]))
                plt.axis('on')
                plt.legend()
                plt.show()
