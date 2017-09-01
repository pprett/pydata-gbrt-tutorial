import numpy as np
import pylab as pl

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.basemap import Basemap
from mpl_toolkits import basemap

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics.scorer import mean_absolute_error_scorer
from sklearn.grid_search import GridSearchCV


def plot_basemap(pdp, x_axis, y_axis, title=None, Z_level=None, num_z_levels=8,
                 ax=None):
    if ax is None:
        pl.figure()
        ax = pl.gca()

    if title:
        ax.set_title(title)

    # fun with contours
    XX, YY = np.meshgrid(x_axis, y_axis)
    Z = pdp[0, :].reshape((x_axis.shape[0], y_axis.shape[0])).T

    Z = basemap.maskoceans(XX, YY, Z, inlands=False, resolution='f', grid=1.25)
    print Z

    if Z_level is None:
        Z_level = np.linspace(Z.min(axis=None),
                              Z.max(axis=None), num=num_z_levels)

    #CS = ax.contour(XX, YY, Z, levels=Z_level, linewidths=0.5,
    #                colors='k')
    CS2 = ax.contourf(XX, YY, Z, levels=Z_level, vmax=Z_level[-1],
                      vmin=Z_level[0], alpha=0.75,
                      cmap=pl.cm.RdBu_r)
    #ax.clabel(CS, fmt='%2.2f', colors='k', fontsize=10, inline=True)
    # pad axis labels from tick labels
    ax.set_xlabel('longitude', labelpad=10)
    ax.set_ylabel('latitude', labelpad=20)


    m = Basemap(projection='cyl', llcrnrlat=y_axis.min(), urcrnrlat=y_axis.max(),
                llcrnrlon=x_axis.min(), urcrnrlon=x_axis.max(),
                resolution='f', ax=ax)
    m.drawcoastlines()
    m.drawcountries()
    #m.drawparallels(np.arange(33, 41, 1))
    #m.drawmeridians(np.arange(-124, 116, 1))
    cb = m.colorbar(CS2, "right", size="5%", pad="2%", format="%.2f")
    cb.set_label('partial dep. on median house value')
    return ax



# fetch California housing dataset
cal_housing = fetch_california_housing()

# split 80/20 train-test
X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                    np.log(cal_housing.target),
                                                    test_size=0.2,
                                                    random_state=1)
names = cal_housing.feature_names

# print('_' * 80)
# print("Training GBRT...")
clf = GradientBoostingRegressor(n_estimators=1000, max_depth=4,
                                learning_rate=0.1, loss='huber',
                                random_state=1, verbose=1)
clf.fit(X_train, y_train)

# print('_' * 80)
# print('Convenience plot with ``partial_dependence_plots``')
# print

# features = [0, 5, 1, 2, (5, 1)]
# features = ['MedInc', 'AveOccup', 'HouseAge', 'AveRooms', ('AveOccup', 'HouseAge')]
# fig, axs = plot_partial_dependence(clf, X_train, features, feature_names=names,
#                                    n_jobs=1, grid_resolution=50)
# fig.suptitle('Partial dependence of house value on nonlocation features\n'
#              'for the California housing dataset')
# pl.subplots_adjust(top=0.9)  # tight_layout causes overlap with suptitle

print('_' * 80)
print('Custom 3d plot via ``partial_dependence``')
print
fig = pl.figure()

target_feature = (1, 5)
pdp, (x_axis, y_axis) = partial_dependence(clf, target_feature,
                                           X=X_train, grid_resolution=50)
XX, YY = np.meshgrid(x_axis, y_axis)
Z = pdp.T.reshape(XX.shape).T
ax = Axes3D(fig)
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=pl.cm.BuPu)
ax.set_xlabel(names[target_feature[0]])
ax.set_ylabel(names[target_feature[1]])
ax.set_zlabel('Partial dependence')
#  pretty init view
ax.view_init(elev=22, azim=122)
pl.colorbar(surf)
pl.suptitle('Partial dependence of house value on median age and '
            'average occupancy')
pl.subplots_adjust(top=0.9)

pl.show()

# ## San Francisco
#xx = np.linspace(-123, -122, 200)
#yy = np.linspace(37, 28, 200)

pdp, axes = partial_dependence(clf, (7, 6), grid=None, X=X_train, percentiles=(0.0, 1.0),
                               grid_resolution=200)
x_axis, y_axis = axes
plot_basemap(pdp, x_axis, y_axis, num_z_levels=20)

pl.show()

def median_absolute_error(y_true, pred):
    return np.median(np.abs(y_true - pred))


#median_absolute_error_scorer = make_scorer(median_absolute_error, greater_is_better=False)

# print('_' * 80)
# print('GBRT')
# param_grid = {'learning_rate': [0.1, 0.05, 0.01],
#               'max_depth': [4, 6],
#               'min_samples_leaf': [3, 5],
#               'subsample': [1.0, 0.5]}

# est = GradientBoostingRegressor(n_estimators=3000, max_depth=6, learning_rate=0.04,
#                                 loss='huber', random_state=0)
# gs_cv = GridSearchCV(est, param_grid, cv=3, verbose=10, scoring=mean_absolute_error_scorer, n_jobs=4)
# gs_cv.fit(X_train, y_train)
# est = gs_cv.best_estimator_
# print est
# print 'MAE: %.4f' % (-1 * mean_absolute_error_scorer(est, X_test, y_test))
# print 'R^2: %.4f' % est.score(X_test, y_test)
# # GBRT: 0.1438
# # R^2: 0.8643

# test_score = np.empty(len(est.estimators_))
# train_score = np.empty(len(est.estimators_))
# for i, pred in enumerate(est.staged_predict(X_test)):
#     test_score[i] = mean_absolute_error(y_test, pred)
# for i, pred in enumerate(est.staged_predict(X_train)):
#     train_score[i] = mean_absolute_error(y_train, pred)
# plt.plot(np.arange(est.n_estimators) + 1, test_score, label='Test', color=test_color, linewidth=2)
# plt.plot(np.arange(est.n_estimators) + 1, train_score, label='Train', color=train_color, linewidth=2)
# plt.ylabel('error')
# plt.xlabel('n_estimators')
# plt.legend(loc='upper right')

# print('_' * 80)
# print('Random Forest')
# param_grid = {'max_features': [7, 6, 5],
#               'min_samples_split': [3, 4, 5],
#               'bootstrap': [True, False],
#               }
# from sklearn.ensemble import RandomForestRegressor
# est = RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=4)
# gs_cv = GridSearchCV(est, param_grid, cv=3, verbose=10, scoring=mean_absolute_error_scorer, n_jobs=1)
# gs_cv.fit(X_train, y_train)
# est = gs_cv.best_estimator_
# print est
# print 'MAE: %.4f' % (-1 * mean_absolute_error_scorer(est, X_test, y_test))
# print 'R^2: %.4f' % est.score(X_test, y_test)

# # Best results
# # MAE: 0.1595
# # R^2: 0.8367



# print('_' * 80)
# print('Ridge')
# from sklearn.linear_model import RidgeCV

# est = RidgeCV(alphas=np.logspace(-2,2,num=50,base=10), scoring=mean_absolute_error_scorer,
#               normalize=False)
# est.fit(X_train, y_train)
# print est
# print 'MAE: %.4f' % (-1 * mean_absolute_error_scorer(est, X_test, y_test))
# print 'R^2: %.4f' % est.score(X_test, y_test)

# # RF: 0.2756
# # R^2: 0.5974
# # train: 5.89 ms
# # test: 108


# print('_' * 80)
# print('SVR')
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVR

# est = Pipeline([('std', StandardScaler()), ('svr', SVR())])

# param_grid = {'svr__C': 10.0 ** np.arange(-1, 4),
#               'svr__gamma': 10.0 ** np.arange(-5, 1),
#               }
# gs_cv = GridSearchCV(est, param_grid, cv=3, verbose=10, scoring=mean_absolute_error_scorer, n_jobs=4)
# gs_cv.fit(X_train, y_train)
# est = gs_cv.best_estimator_
# print est
# print 'MAE: %.4f' % (-1 * mean_absolute_error_scorer(est, X_test, y_test))
# print 'R^2: %.4f' % est.score(X_test, y_test)

# # MAE: 0.1979
# # R^2: 0.7709
