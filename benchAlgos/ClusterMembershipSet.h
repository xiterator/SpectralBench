#pragma once

#include <vector>
#include <boost/ptr_container/ptr_vector.hpp>
#include "ClusterMembership.h"
#include "EvalIndex.h"
#include "GAmatrix.h"


class ClusterMembershipSet : public boost::ptr_vector<ClusterMembership> {

private:
	typedef boost::ptr_vector<ClusterMembership> MyBase;

	unsigned int checkEqualMembers(const ClusterMembershipSet& truth) const;

	double compNMI(const ClusterMembershipSet& truth) const;
	double compCA(const ClusterMembershipSet& truth, std::vector<long>& assignToTruth) const;

	void compProfitMat(const ClusterMembershipSet& truth, EvalIndex::Name evalIndex, GAmatrix& ProfitMat) const;

      
public:

	//Evaluation
	double similarMatch(const ClusterMembershipSet& truth, EvalIndex::Name evalIndex, std::vector<long>& assignToTruth) const;

	void compSimilarMatches(const ClusterMembershipSet& truth, GAmatrix& matches) const;

    //as a history keeper for ClusterMemberships
    bool keepStableHistory(const ClusterMembership& newMemb);
    bool isConvergent() const;
};


std::ostream& operator<<(std::ostream& out, const ClusterMembershipSet& cds);
