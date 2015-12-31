#pragma once

#include "Cluster.h"
#include "RecordSet2.h"
#include "RandomDice.h"
#include "OnceRunResult.h"

#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

#include <string>
#include <vector>

namespace mp = boost::multiprecision;


class ClusterSet : public boost::ptr_vector<Cluster>
{
private:

    DiceUniformInt clusterInitialMethod_dice;

    const RecordSet2* prs;
    typedef mp::number<mp::cpp_dec_float<10> > Decimal10;
    std::vector<Decimal10> historyCluErrs; //Clustering Error History

    Decimal10 oldAvgWsparse;
    Decimal10 oldAvgOrthIp;

    unsigned int getClosestCluster(unsigned int recordId) const;

    double getClusteringError() const;

    double getObjectValue() const;

    /*
    double getMaxOrthIpIndex() const;
    double getMinWsparseIndex() const;
    double getAveWsparseIndex() const;*/


public:
    ClusterSet();
    ~ClusterSet();

    void initial(unsigned int clusterNum, unsigned int t_groupNum, unsigned int dim,
                  double lambda, double beta, const RecordSet2* rs, VupdateMethod::Name n);
    void partitionData();

    bool isConvergent();


    void gatherInitResult(OnceRunResult& orr) const;
    void gatherStopResult(OnceRunResult& orr, const ClusterMembershipSet& truth) const;
    double gatherEvalIndex(const ClusterMembershipSet& truth, std::string& info) const;
};
