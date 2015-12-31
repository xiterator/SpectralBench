#include "stdafx.h"
#include "ClusterSet.h"
#include <sstream>

ClusterSet::ClusterSet() : clusterInitialMethod_dice(0, 100), prs(0)
{
}


ClusterSet::~ClusterSet()
{ }

void ClusterSet::initial(unsigned int clusterNum, unsigned int t_groupNum,
                         unsigned int dim, double lambda, double beta,
						 const RecordSet2* rs, VupdateMethod::Name n) {
    //yjf.debug
    std::cout << "clusterNum: " << clusterNum << std::endl;
    std::cout << "groupNum: " << t_groupNum << std::endl;
    std::cout << "lambda: " << lambda << std::endl;
    std::cout << "beta: " << beta << std::endl;
    std::cout << "-------------------- V update method: " << VupdateMethod::tostr(n) << std::endl;

    //reset inner members
    historyCluErrs.clear();
    this->prs = rs;
    oldAvgWsparse = 0;
    oldAvgOrthIp = 0;

    //create new cluster workers if needed more.
    unsigned int i = this->size();
    for (; i < clusterNum; ++i) {
            Cluster *p = new Cluster();
            this->push_back(p);
    }
    i = 0;
    for (auto& j : (*this)) {
        j.init0_setting(i, t_groupNum, dim, lambda, beta, rs, n);
        ++i;
    }

    std::cout << "------------ after initial0 ------------------------------" << std::endl;

    //initialize all clusters one by one
    std::vector<Record2*> existingCenters;
    for (auto& k : (*this)) {
        k.init1_random(existingCenters, InitialMethod::V_method4);
    }

}


//////////////////////////////////////////////////////////////////////////////////
unsigned int ClusterSet::getClosestCluster(unsigned int recordId) const {
	
	double dist = (*this)[0].distanceToMe(recordId);
	unsigned int result = 0;

	unsigned int sz = this->size();
	unsigned int j = 1;
	for (; j < sz; ++j) {
		
		double dt = (*this)[j].distanceToMe(recordId);
		if (dt < dist) {
			dist = dt;
			result = j;		//keep the closest cluster to the input record
		}

	}
	return result;
}

void ClusterSet::partitionData() {


    //Clear the previous assigned members of each cluster
	//automatically
	for (auto& i : (*this))
		i.clearAllMembers();
	

	//assign all data according to current centers of all clusters
	unsigned int id = 0;
	unsigned int sz = prs->size();		//for each record in the total element space.
	for (; id < sz; ++id) {
		unsigned int which = getClosestCluster(id);
		(*this)[which].assignMember(id);
		
	}

	//Now, we got new assignment of all data
	//So, we can compute new centers, V and W
	for (auto& i : (*this)) {
		i.update();
	}

    //Keep the newest result into queue,
    //and remove the old ones if the newest result
    // is not close enough to them

    Decimal10 result = getClusteringError();
    if (historyCluErrs.size()==0) {
        historyCluErrs.push_back(result);
    }
    else {

        //unsigned int last = historyCluErrs.size() - 1;
        //0.0750795
        if (abs(historyCluErrs[0] - result)/result > 0.001) {
            historyCluErrs.clear();
            //std::cout << "********************* historyCluErrs.size = " << historyCluErrs.size() << std::endl;
        }
        else {
            //std::cout << "^^^^^^^^^^^^^^^^^^ history got two near" << std::endl;
            //for(const auto& i: historyCluErrs) {
            //    std::cout << i << ", ";
            //}
            //std::cout << result << std::endl;


            historyCluErrs.insert(
            std::upper_bound(historyCluErrs.begin(), historyCluErrs.end(), result),
                         result);
        }
    }

    std::cout << "--=======---partition: his clu errs --========: ";
    std::cout << historyCluErrs.size() << std::endl;

    //for(const auto& i: historyCluErrs) {
       // std::cout << i << ", ";
    //}
    //std::cout << std::endl;


    //0.0750795
    //0.00282903

}

bool ClusterSet::isConvergent() {

    for(auto& i: *this) {
        if (!i.isConvergent())
            return false;
    }

    //W spare index
    Decimal10 wsparse = 0;
    for(const auto& i: *this) {
        wsparse += i.getWsparseIndex();
    }
    wsparse /= this->size();

    if (abs(wsparse - oldAvgWsparse)/wsparse > 0.001) {
        std::cout << "///////////////////// wsparse ratio > 0.001" << std::endl;
        oldAvgWsparse = wsparse;
        return false;
    }
    oldAvgWsparse = wsparse;

    // Orthogonal Inner Product Index
    Decimal10 orth = 0;
    for(const auto& i: *this) {
        orth += i.getOrthIpIndex();
    }
    orth /= (this->size() * 1.0);

    if (orth < Constant::orthIndexSmallest) {
        std::cout << "/-//-//-//-//-/-// orth ip ratio= " << orth;
        std::cout << " < " << Constant::orthIndexSmallest << ", stable";
        std::cout << std::endl;

       oldAvgOrthIp = orth;
       return true;
    }

    std::cout << "/-//-//-//-//-/-// orth ip ratio= " << orth;
    std::cout << " > " << Constant::orthIndexSmallest;
    std::cout << std::endl;

    /*
    if (abs(orth - oldAvgOrthIp)/orth < 0.001) {
        std::cout << "/-//-//-//-//-/-// orth ip ratio < 0.001 stable" << std::endl;

        oldAvgOrthIp = orth;
        return true;
    }

    oldAvgOrthIp = orth;
    if (historyCluErrs.size() >= 3) {
        std::cout << "/-//-//-//-//-/-// clu errs len >= 3 stable" << std::endl;

        return true;
    }*/

    return false;

}

/////////////////////////////////////////////////////////////////////////
/// \brief ClusterSet::getAveWsparseIndex
/// \return
///
/*
double ClusterSet::getAveWsparseIndex() const {
    double s=0;
    for (const auto& i : (*this)) {
        s += i.getWsparseIndex();
    }
    return s / this->size();
}
double ClusterSet::getMinWsparseIndex() const {
    double mn = 1;
    double s;
    for (const auto& i : (*this)) {
        s = i.getWsparseIndex();
        if (s < mn) mn = s;
    }
    return mn;
}
double ClusterSet::getMaxOrthIpIndex() const {
    double mx = 0;
    double ip;
    for (const auto& i : (*this)) {
        ip = i.getOrthIpIndex();
        if (ip > mx) mx = ip;
    }
    return mx;
}*/

double ClusterSet::getObjectValue() const {
	double result = 0;
	for (const auto& i : (*this)) {
		result += i.getObjectValue();
	}
	return result;
}
double ClusterSet::getClusteringError() const {

    double result =0;
    for(const auto& i: *this) {
        result += i.getClusteringError();
    }
    return result;
}



//////////////////////////////////////////////////////////////
void ClusterSet::gatherInitResult(OnceRunResult& orr) const {

	//RecordSet2 z_ks0;		//initial z_ks0
	//GAmatrixSet V_ks0;		//initial V_ks
	//RecordSet2 w_ks0;		//initial w_ks

	//clusters initialization methods
	//dataNormalizeMethod

    orr.numGroups = (*this)[0].getNumGroups();
    
    for (const auto& i : (*this)) {
        orr.z_ks0.push_back(new Record2(i.getCenter()) );
        orr.V_ks0.push_back( new GAmatrix(i.getV()) );
        orr.w_ks0.push_back(new Record2(i.getW()) );
        orr.clusterInitialMethods += InitialMethod::tostr(i.getInitialMethod()) + "-";
    }
    orr.dataNormalizeMethod = DataNormalizeMethod::tostr(prs->getUsedNormalizeMethod());
}

void ClusterSet::gatherStopResult(OnceRunResult& orr, const ClusterMembershipSet& truth) const {

    //RecordSet2 z_ks;
    //GAmatrixSet V_ks;
    //RecordSet2 w_ks;

    //unsigned int lambda;
    //double objectValue;

    //ClusterMembershipSet h_ks;

	orr.objectValue = this->getObjectValue();
    orr.lambda = (*this)[0].getLambda();	//same for all clusters
    orr.beta = (*this)[0].getBeta();	//same for all clusters

	for (const auto& i : (*this)) {

		orr.z_ks.push_back(new Record2(i.getCenter()) );
		orr.V_ks.push_back(new GAmatrix(i.getV()) );
		orr.w_ks.push_back(new Record2(i.getW()) );

		orr.h_ks.push_back(new ClusterMembership(i.getMembers()) ); //(1)
		orr.clusterInitialMethods += InitialMethod::tostr(i.getInitialMethod()) + "-";

	}
	orr.dataNormalizeMethod = DataNormalizeMethod::tostr(prs->getUsedNormalizeMethod());

	orr.h_ks.compSimilarMatches(truth, orr.matches);
    //compSimilarMatches(orr.h_ks, truth, orr.matches);	//must be called after (1)

}

double ClusterSet::gatherEvalIndex(const ClusterMembershipSet& truth, std::string& info) const {

    ClusterMembershipSet h_ks;
    for (const auto& i : (*this)) {
        h_ks.push_back(new ClusterMembership(i.getMembers())); //(1)
    }
    GAmatrix matches;
	h_ks.compSimilarMatches(truth, matches); //yjf.add.12.25.

    //compSimilarMatches(h_ks, truth, matches);	//must be called after (1)

    // only gives CA index
    unsigned int i = EvalIndex::CA;
    unsigned int sz = EvalIndex::CA+1;
    for (; i < sz; ++i) {
        info += EvalIndex::tostr(i);
        info += ": ";
        iAppendString(matches(i, 0), info);
        info += ", ";
    }
        
    info += "\r\nGroups: ";
    iAppendString((*this)[0].getNumGroups(), info);
    info += ",  totalCluErr: ";
    iAppendString(getClusteringError(), info);
    info += "\r\n";
    info += "cluErrors           wSpare           OrthIp:\r\n";
    //add cluster's clusterError, WsparseIndex, orthIndex
    for(const auto& i: *this) {
        iAppendString(i.getClusteringError(), info);
        info += "           ";
        iAppendString(i.getWsparseIndex(), info);
        info += "           ";
        iAppendString(i.getOrthIpIndex(), info);
        info += "\r\n";
    }

    return matches(EvalIndex::CA, 0);   //return CA value
}
