#!/usr/bin/env python
from tql import tql
import numpy as np
import matplotlib.pyplot as pl
from matplotlib import cm
from adjustText import adjust_text
from astropy.coordinates import SkyCoord
import astropy.units as u

if False:
    df=tql.get_open_clusters()
    print(df)

if False:
    '''Each cluster in df1 is in df2'''
    df1=tql.get_open_clusters_near()
    df2=tql.get_open_cluster_members_near()
    cnames = df1.Cluster.apply(lambda x: x.replace(' ', ''))
    cnames.isin(df2.Cluster)

if False:
    '''
    check if NMemb in df1=get_open_clusters_near()
    is equal to the actual number of members in
    df2=get_open_cluster_members_near()
    ===result===
    cname,df1,df2
    alphaPer 740 743    <-- more members?
    Blanco1 489 489
    ComaBer 153 153
    Hyades 515 515
    IC2391 325 327      <-- more members?
    IC2602 492 494      <-- more members?
    NGC2451 400 404     <-- more members?
    Pleiades 1326 1332  <-- more members?
    Praesepe 938 946    <-- more members?

    More members than in summary table
    '''
    df1=tql.get_open_clusters_near()
    df2=tql.get_open_cluster_members_near()
    # df1.Cluster = df1.Cluster.apply(lambda x: x.replace(' ', ''))
    cnames = df1.Cluster.values
    g = df2.groupby(by='Cluster')
    print('cname, df1, df2')
    for c in cnames:
        nmemb1 = df1.loc[df1.Cluster==c,'NMemb'].values[0]
        nmemb2 = len(g.groups[c])
        print(c, nmemb1, nmemb2)

if False:
    '''
    check if NMemb in df1=get_open_clusters_far()
    is equal to the actual number of members in
    df2=get_open_cluster_members_far()
    ===result===
    cname,df1,df2
    NGC 188 1181 1181
    NGC 752 433 433
    Stock 2 1742 1742
    NGC 869 829 1691    <-- more members?
    NGC 884 1077 1077
    Trump 2 589 589
    NGC 1039 764 764
    NGC 1901 290 290
    NGC 2158 3942 4861  <-- more members?
    NGC 2168 1794 1848  <-- more members?
    NGC 2232 318 383    <-- more members?
    Trump 10 947 947
    NGC 2323 382 1537   <-- more members?
    NGC 2360 1037 1037
    Coll 140 332 332
    NGC 2423 694 694
    NGC 2422 907 907
    NGC 2437 3032 3032
    NGC 2447 926 926
    NGC 2516 2518 2518
    NGC 2547 644 646
    NGC 2548 509 509
    NGC 2682 1520 1526  <-- more members?
    NGC 3228 222 222
    NGC 3532 1879 1879
    NGC 6025 452 452
    NGC 6281 573 573
    IC 4651 960 960
    NGC 6405 967 967
    IC 4665 174 174
    NGC 6475 1140 1140
    NGC 6633 321 321
    IC 4725 755 807     <-- more members?
    IC 4756 543 543
    NGC 6774 234 234
    NGC 6793 465 465
    NGC 7092 433 718    <-- more members?

    More members than in summary table
    '''
    df1=tql.get_open_clusters_far()
    df2=tql.get_open_cluster_members_far()
    # df1.Cluster = df1.Cluster.apply(lambda x: x.replace(' ', ''))
    cnames = df1.Cluster.values
    g = df2.groupby(by='Cluster')
    print('cname, df1, df2')
    for c in cnames:
        nmemb1 = df1.loc[df1.Cluster==c,'NMemb'].values[0]
        nmemb2 = len(g.groups[c])
        print(c, nmemb1, nmemb2)

if False:
    '''append mean values to open cluster members >250pc'''
    df_far_mem = tql.get_open_cluster_members_far()
    df_far_mem['mean_plx'] = np.ones(len(df_far_mem))*np.nan
    df_far_mem['mean_e_plx'] = np.ones(len(df_far_mem))*np.nan
    df_far_mem['mean_pmRA'] = np.ones(len(df_far_mem))*np.nan
    df_far_mem['mean_pmDE'] = np.ones(len(df_far_mem))*np.nan
    df_far_mem['RV'] = np.ones(len(df_far_mem))*np.nan

    for cname in df_far.Cluster.unique():
        try:
            df_far_mem.loc[df_far_mem.Cluster==cname,'mean_plx'] = df_far.loc[df_far.Cluster==cname,'plx'].values[0]
            df_far_mem.loc[df_far_mem.Cluster==cname,'mean_e_plx'] = df_far.loc[df_far.Cluster==cname,'e_plx'].values[0]
            df_far_mem.loc[df_far_mem.Cluster==cname,'mean_pmRA'] = df_far.loc[df_far.Cluster==cname,'pmRA'].values[0]
            df_far_mem.loc[df_far_mem.Cluster==cname,'mean_pmDE'] = df_far.loc[df_far.Cluster==cname,'pmDE'].values[0]
            df_far_mem.loc[df_far_mem.Cluster==cname,'mean_RV'] = df_far.loc[df_far.Cluster==cname,'RV'].values[0]
        except Exception as e:
            print(e)
    print(df_far_mem)

if False:
    '''plot open clusters far'''
    fig = pl.figure(figsize=(15,8))
    df_far = tql.get_open_clusters_far()
    df_far_mem = tql.get_open_cluster_members_far()
    nclusters = len(df_far_mem.Cluster.unique())
    colors = cm.rainbow(np.linspace(0, 1, nclusters))

    skip=10
    texts = []
    i = 0
    for cname,df in df_far_mem.groupby(by='Cluster'):
        idx = df_far.Cluster==cname
        ra, dec = df_far.loc[idx,['RAJ2000','DEJ2000']].values[0]
        texts.append(pl.text(ra, dec, cname))
        for r,d in zip(df.ra.values[::skip], df.dec.values[::skip]):
            pl.plot(r,d,'.',c=colors[i],alpha=0.3)
        i+=1
    adjust_text(texts)
    pl.show()

if False:
    tql.plot_parallax_density(target_gaia_id, df, ax=None, verbose=True)

if False:
    tql.plot_rv_density(target_gaia_id, df_gaia, ax=None, verbose=True)

if False:
    tql.plot_target_in_cluster(target_gaia_id, df_gaia, params=['ra','dec'],
                                    show_centroid=True, ax=None, cluster_name=None,
                                    verbose=True)
if False:
    target_gaia_id = 5251470948229949568 #toi837
    plot_cluster_membership(target_gaia_id, cluster=None, #ax=None,
                               min_cluster_diameter=100, verbose=False,
                               figoutdir='.',savefig=False)
