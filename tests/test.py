#!/usr/bin/env python
from tql import tql

if False:
    df=tql.get_open_clusters()
    print(df)

if False:
    '''
    Each cluster in df1 is in df2
    '''
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
    alphaPer 740 743
    Blanco1 489 489
    ComaBer 153 153
    Hyades 515 515
    IC2391 325 327
    IC2602 492 494
    NGC2451 400 404
    Pleiades 1326 1332
    Praesepe 938 946

    More members than in summary table
    '''
    df1=tql.get_open_clusters_near()
    df2=tql.get_open_cluster_members_near()
    df1.Cluster = df1.Cluster.apply(lambda x: x.replace(' ', ''))
    cnames = df1.Cluster.values
    g = df2.groupby(by='Cluster')
    for c in cnames:
        print('cname df1 df2'.split())
        nmemb1 = df1.loc[df1.Cluster==c,'NMemb'].values[0]
        nmemb2 = len(g.groups[c])
        print(c, nmemb1, nmemb2)
