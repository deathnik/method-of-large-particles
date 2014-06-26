#include <iostream>
#include <cmath>
#include "mpi.h"
#include "stdio.h"
#include <fstream>
#include "stdlib.h"
#include "limits.h"
#include "omp.h"
#include <random>
#include <vector>
#include <thread>
#include <chrono>
#include <iomanip>
#include <algorithm>
using std::cout;
using std::vector;

#define COURANT 0.5
int rank;
int debugNum = -1;
template <class T> class SimpleMatrix {
private:
    int size1;
    int size2;
    vector<T> arrS;
    vector<T> arrR;
    MPI_Request requests[10];
    int requestsCount = 0;

public:
    vector<T> data;
    SimpleMatrix() {}

    SimpleMatrix(int size1, int size2) {
        this->size1 = size1;
        this->size2 = size2;
        data.resize(size1 * size2);
        arrS.resize(size1 * size2);
        arrR.resize(size1 * size2);
    }

    void resize(int size1, int size2) {
        this->size1 = size1;
        this->size2 = size2;
        data.resize(size1 * size2);
        arrS.resize(size1 * size2);
        arrR.resize(size1 * size2);
    }

    T &operator() (unsigned int i, unsigned int j) {
        return data[i * size2 + j];
    }

    const T &operator() (unsigned int i, unsigned int j) const {
        return *data[i * size2 + j];
    }

    T &operator[] (unsigned int i) {
        return data[i * size2];
    }

    const T &operator[] (unsigned int i) const {
        return *data[i * size2];
    }

    int size() {
        return size1 * size2;
    }

    int getSize1() {
        return size1;
    }

    int getSize2() {
        return size2;
    }

    void input(std::ifstream &myfile) {
        T value;
        for (int i = 0; i < size1; ++i) {
            for (int j = 0; j < size2; ++j) {
                myfile >> value;
                (*this)(i, j) = value;
            }
        }
    }

    void print() {
        for (int i = 0; i < size1; ++i) {
            for (int j = 0; j < size2; ++j) {
                cout << (*this)(i, j) << " ";
            }
            cout << std::endl;
        }
        cout << std::endl;
    }

    void save(std::ofstream & myfile) {
        for (int i = 0; i < size1; ++i) {
            for (int j = 0; j < size2; ++j) {
                myfile << (*this)(i, j) << " ";
            }
            myfile << std::endl;
        }
        myfile << std::endl;
    }

    void sendPart(int iBegin, int iEnd, int jBegin, int jEnd, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
        int count = abs((iEnd - iBegin) * (jEnd - jBegin));
        arrS.resize(count);

        int k = 0;
        for (int i = iBegin; i < iEnd; ++i) {
            for (int j = jBegin; j < jEnd; ++j) {
                arrS[k++] = (*this)(i, j);
            }
        }

        MPI_Send(arrS.data(), count, datatype, dest, tag, comm);
    }

    void recvPart(int iBegin, int iEnd, int jBegin, int jEnd, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status* status) {
        int count = abs((iEnd - iBegin) * (jEnd - jBegin));
        if(rank == debugNum) cout<<count<<" count to recv \n" << iBegin << " " << iEnd << " " << jBegin << " " << jEnd <<"\n";
        MPI_Recv(arrR.data(), count, datatype, source, tag, comm, status);

        int k = 0;
        for (int i = iBegin; i < iEnd; ++i) {
            for (int j = jBegin; j < jEnd; ++j) {
                (*this)(i, j) = arrR[k++];
            }
        }
    }

    void wait() {
        return;
        MPI_Status *statuses = new MPI_Status[requestsCount];
        MPI_Waitall(requestsCount, requests, statuses);
        requestsCount = 0;
    }
};

char *getCmdOption(char **begin, char **end, const std::string &option) {
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char **begin, char **end, const std::string &option) {
    return std::find(begin, end, option) != end;
}

namespace code {

//------------------  DATA {  --------------------
    int n = 1000;
    int master = 0;
    int size;
    int igl = 0;
    int jgl = 0;
    int r1 = 1, r2 = 1, r3 = 1;
    int Q1 = 1, Q2, Q3 = 1;
    int leftN, rightN, topN, botN;
    int uSize = 0;
    vector< SimpleMatrix<double> > us;
    int prociGl = 0;
    int procjGl = 0;
    SimpleMatrix<double> Ab;

    int currentLayer = 0;
    int layersNum = 100;

    int M = 0, N = 0;
    double dt = 0, dr = 0, dz = 0;
    double gamma = 0;
    double dtz = 0, dtr = 0;
    double dk;

    SimpleMatrix<double> roOld, uOld, vOld, EOld, ro, u, v, E, 
        pIPlus05, pJPlus05, uPlus05, vPlus05, uTilde, vTilde, ETilde, 
        dMIPlus05, dMJPlus05, p;

//------------------  } DATA  --------------------
    template <class T>
    T min(T a, T b) {
        return a < b ? a : b;
    }

    template <class T>
    T max(T a, T b) {
        return a > b ? a : b;
    }
    bool isMaster = false;
    void Init() {
        std::ifstream myfile;
        if(isMaster){
            cout << "Master initializing...\n";
            myfile.open("input.txt");
        }else {
            cout << "Slave initializing...\n";
        }
        //reading M, N
        if( isMaster ) myfile >> M >> N;
        //cout << "M: " << M << " N: " << N << "\n";
        MPI_Bcast(&M, 1, MPI_INT, master, MPI_COMM_WORLD);
        MPI_Bcast(&N, 1, MPI_INT, master, MPI_COMM_WORLD);

        roOld.resize(M + 2, N + 2);
        uOld.resize(M + 2, N + 2);
        vOld.resize(M + 2, N + 2);
        EOld.resize(M + 2, N + 2);
        ro.resize(M + 2, N + 2);
        u.resize(M + 2, N + 2);
        v.resize(M + 2, N + 2);
        E.resize(M + 2, N + 2);
        pIPlus05.resize(M + 2, N + 2);
        pJPlus05.resize(M + 2, N + 2);
        uPlus05.resize(M + 2, N + 2);
        vPlus05.resize(M + 2, N + 2);
        uTilde.resize(M + 2, N + 2);
        vTilde.resize(M + 2, N + 2);
        ETilde.resize(M + 2, N + 2);
        dMIPlus05.resize(M + 2, N + 2);
        dMJPlus05.resize(M + 2, N + 2);
        p.resize(M + 2, N + 2);

        //reading dt, dr, dz, gamma, dk
        if( isMaster ) myfile >> dt >> dr >> dz >> gamma >> dk;

        MPI_Bcast(&dt, 1, MPI_DOUBLE, master, MPI_COMM_WORLD);
        MPI_Bcast(&dr, 1, MPI_DOUBLE, master, MPI_COMM_WORLD);
        MPI_Bcast(&dz, 1, MPI_DOUBLE, master, MPI_COMM_WORLD);
        MPI_Bcast(&gamma, 1, MPI_DOUBLE, master, MPI_COMM_WORLD);
        MPI_Bcast(&dk, 1, MPI_DOUBLE, master, MPI_COMM_WORLD);
        /*
        //reading ro
        if( isMaster )roOld.input(myfile);
        MPI_Bcast(roOld.data.data(), (M+2)* (N+2), MPI_DOUBLE, master, MPI_COMM_WORLD);

        //reading u
        if( isMaster )uOld.input(myfile);
        MPI_Bcast(uOld.data.data(), (M+2)* (N+2), MPI_DOUBLE, master, MPI_COMM_WORLD);

        //reading v
        if( isMaster )vOld.input(myfile);
        MPI_Bcast(vOld.data.data(), (M+2)* (N+2), MPI_DOUBLE, master, MPI_COMM_WORLD);

        //reading E
        if( isMaster )EOld.input(myfile);
        MPI_Bcast(EOld.data.data(), (M+2)* (N+2), MPI_DOUBLE, master, MPI_COMM_WORLD);

        myfile.close();*/
    }

    void debugInput() {
        cout << "M = " << M << "\n";
        cout << "N = " << N << "\n\n";
        cout << "dt = " << dt << "\n";
        cout << "dr = " << dr << "\n";
        cout << "dz = " << dz << "\n";
        cout << "gamma = " << gamma << "\n";
        cout << "dz = " << dk << "\n\n";

        cout << "ro = \n";
        roOld.print();

        cout << "u = \n";
        uOld.print();

        cout << "v = \n";
        vOld.print();

        cout << "E = \n";
        EOld.print();
    }

    void debugOutput() {
        cout << "\n\n!!! new ro = \n";
        ro.print();
    }

    void saveOutput() {
        std::ofstream myfile("output.txt");

        roOld.save(myfile);
        uOld.save(myfile);
        vOld.save(myfile);
        EOld.save(myfile);

        myfile.close();
    }

    void stage1() {
        for (int i = max(igl * r1 - 1,1); i <= min(2 + (igl+1)*r1, M); ++i) {
            for (int j = max(jgl * r2 - 1,1); j <= min(2 + (jgl+1)*r2, N); ++j) {
                p(i, j) = (gamma - 1) * roOld(i, j) *
                          (EOld(i, j) - ((uOld(i, j) * uOld(i, j) + vOld(i, j) * vOld(i, j)) / 2));
            }
        }

        if( igl == 0){
            for(int j = jgl * r2; j < min(1 + (jgl +1)*r2,N); ++j){
                p(0,j) = p(1,j);
            }
        }

        if( jgl == 0){
            for(int i = igl * r1; i < min(1 + (igl +1)*r1,M); ++i){
                p(i,0) = p(i,1);
            }
        }

        if( igl == Q1 - 1){
            for(int j = jgl * r2; j < min(1 + (jgl +1)*r2,N); ++j){
                p(M+1,j) = p(M,j);
            }
        }

        if( jgl == Q2-1){
            for(int i = igl * r1; i < min(1 + (igl +1)*r1,M); ++i){
                p(i,N+1) = p(i,N);
            }
        }

    }

    void stage2() {
        for (int i = max(igl * r1 - 1,0); i <= min(1 + (igl+1)*r1, M); ++i) {
            for (int j = max(jgl * r2 - 1,1); j <= min(1 + (jgl+1)*r2, N); ++j) {
                pIPlus05(i, j) = (p(i, j) + p(i+1, j)) / 2;
                uPlus05(i, j) = (uOld(i, j) + uOld(i+1, j)) / 2;
            }
        }
        for (int i = max(igl * r1 ,1); i <= min(1 + (igl+1)*r1, M); ++i) {
            for (int j = max(jgl * r2 - 1,0); j <= min(1 + (jgl+1)*r2, N); ++j) {
                pJPlus05(i, 0) = (p(i, j) + p(i, j+1)) / 2;
                vPlus05(i, 0) = (vOld(i, j) + vOld(i, j+1)) / 2;
            }
        }
        //покрыты ли границы я хз
    }

    void stage3() {
        for (int i = max(igl * r1,1); i <= min(1 + (igl+1)*r1, M); ++i) {
            for (int j = max(jgl * r2 ,1); j <= min(1 + (jgl+1)*r2, N); ++j) {
                uTilde(i, j) = uOld(i, j) - ((pIPlus05(i, j) + pIPlus05(i - 1, j)) / dz) * (dt / roOld(i, j));
                vTilde(i, j) = vOld(i, j) - ((pJPlus05(i, j) + pJPlus05(i, j - 1)) / dr) * (dt / roOld(i, j));
                ETilde(i, j) = EOld(i, j) -
                               ((pIPlus05(i, j) * uPlus05(i, j) - pIPlus05(i - 1, j) * uPlus05(i - 1, j)) / dz +
                                (j * pJPlus05(i, j) * vPlus05(i, j) - (j - 1) * pJPlus05(i, j - 1) * vPlus05(i, j - 1)) /
                                ((j - 0.5) * dr)) *
                               (dt / roOld(i, j));
            }
        }
        if(igl == 0){
            for (int j = jgl * r2; j <=  min(1+(jgl+1)*r2,N); ++j) {
                uTilde(0, j) = -uTilde(1, j);
                vTilde(0, j) = vTilde(1, j);
                ETilde(0, j) = ETilde(1, j);
            }
        }
        if(igl == Q1 - 1){
            for (int j = jgl * r2; j <=  min(1+(jgl+1)*r2,N); ++j) {
                uTilde(M + 1, j) = uTilde(M, j);
                vTilde(M + 1, j) = vTilde(M, j);
                ETilde(M + 1, j) = ETilde(M, j);
            }
        }
        if(jgl == 0){
            for (int i = igl*r1; i <=  min(1+(igl+1)*r1,M); ++i) {
                uTilde(i, 0) = uTilde(i, 1);
                vTilde(i, 0) = -vTilde(i, 1);
                ETilde(i, 0) = ETilde(i, 1);
            }
        }

        if(jgl == Q2-1){
            for (int i = igl*r1; i <=  min(1+(igl+1)*r1,M); ++i) {
                uTilde(i, N + 1) = uTilde(i, N);
                vTilde(i, N + 1) = -vTilde(i, N);
                ETilde(i, N + 1) = ETilde(i, N);
            }
        }
    }

    void stage4() {
        for (int i = igl * r1; i <= min((igl+1)*r1,M); ++i) {
            for (int j = jgl * r2 + 1; j <= min((jgl+1)*r2,N); ++j) {
                if (uTilde(i, j) + uTilde(i + 1, j) >= 0) {
                    dMIPlus05(i, j) = (j - 0.5) * roOld(i, j) * ((uTilde(i, j) + uTilde(i + 1, j)) / 2) * dr * dr * dt;
                } else {
                    dMIPlus05(i, j) = (j - 0.5) * roOld(i + 1, j) * ((uTilde(i, j) + uTilde(i + 1, j)) / 2) * dr * dr * dt;
                }
            }
        }

        for (int i = igl * r1 +1; i <= min((igl+1)*r1,M); ++i) {
            for (int j = jgl * r2 ; j <= min((jgl+1)*r2,N); ++j) {
                if (vTilde(i, j) + vTilde(i, j + 1) >= 0) {
                    dMJPlus05(i, j) = j * roOld(i, j) * ((vTilde(i, j) + vTilde(i, j + 1)) / 2) * dr * dz * dt;
                } else {
                    dMJPlus05(i, j) = j * roOld(i, j + 1) * ((vTilde(i, j) + vTilde(i, j + 1)) / 2) * dr * dz * dt;
                }
            }
        }
    }

    void stage5() {
        for (int i = 1 + igl * r1; i <= min((igl+1)*r1,M); ++i) {
            for (int j = 1 + jgl * r2 ; j <=min((jgl+1)*r2,N); ++j) {
                ro(i, j) = roOld(i, j) +
                           (dMIPlus05(i - 1, j) + dMJPlus05(i, j - 1) - dMIPlus05(i, j) - dMJPlus05(i, j)) /
                           ((j - 0.5) * dz * dr * dr);
                u(i, j) = (roOld(i, j) / ro(i, j)) * uTilde(i, j) +
                          (uTilde(i - 1, j) * dMIPlus05(i - 1, j) + uTilde(i, j - 1) * dMJPlus05(i, j - 1) -
                           uTilde(i + 1, j) * dMIPlus05(i, j) - uTilde(i, j + 1) * dMJPlus05(i, j)) /
                          ((j - 0.5) * dz * dr * dr);
                v(i, j) = (roOld(i, j) / ro(i, j)) * vTilde(i, j) +
                          (vTilde(i - 1, j) * dMIPlus05(i - 1, j) + vTilde(i, j - 1) * dMJPlus05(i, j - 1) -
                           vTilde(i + 1, j) * dMIPlus05(i, j) - vTilde(i, j + 1) * dMJPlus05(i, j)) /
                          ((j - 0.5) * dz * dr * dr);
                E(i, j) = (roOld(i, j) / ro(i, j)) * ETilde(i, j) +
                          (ETilde(i - 1, j) * dMIPlus05(i - 1, j) + ETilde(i, j - 1) * dMJPlus05(i, j - 1) -
                           ETilde(i + 1, j) * dMIPlus05(i, j) - ETilde(i, j + 1) * dMJPlus05(i, j)) /
                          ((j - 0.5) * dz * dr * dr);
            }
        }

        if(igl == 0){
            for (int j = 1 + jgl * r2; j <=  min((jgl + 1)*r2,N); ++j) {
                u(0, j) = -u(1, j);
                v(0, j) = v(1, j);
            }
        }
        if(igl == Q1 - 1){
            for (int j = 1 + jgl * r2; j <=  min((jgl + 1)*r2,N); ++j) {
                ro(M + 1, j) = ro(M, j);
                u(M + 1, j) = u(M, j);
                v(M + 1, j) = v(M, j);
            }
        }
        if(jgl == 0){
            for (int i = 1 + igl*r1; i <=  min((igl + 1)*r1,M); ++i) {
                u(i, 0) = u(i, 1);
                v(i, 0) = -v(i, 1);
            }
        }

        if(jgl == Q2-1){
            for (int i = 1 + igl*r1; i <=  min((igl + 1)*r1,M); ++i) {
                u(i, N + 1) = u(i, N);
                v(i, N + 1) = -v(i, N);
            }
        }

    }

    void countNewDt() {
        double maxKoef = std::fabs(u(1, 1)) + std::sqrt(gamma * (p(1, 1) / ro(1, 1)));
        double newKoef = 0;
        for (int i = 1; i <= M; ++i) {
            for (int j = 1; j <= N; ++j) {
                newKoef = std::fabs(u(i, j)) + std::sqrt(gamma * (p(i, j) / ro(i, j)));
                if (newKoef > maxKoef) {
                    maxKoef = newKoef;
                }
            }
        }
        dtz = COURANT * dz / maxKoef;

        maxKoef = std::fabs(v(1, 1)) + std::sqrt(gamma * (p(1, 1) / ro(1, 1)));
        newKoef = 0;
        for (int i = 1; i <= M; ++i) {
            for (int j = 1; j <= N; ++j) {
                newKoef = std::fabs(v(i, j)) + std::sqrt(gamma * (p(i, j) / ro(i, j)));
                if (newKoef > maxKoef) {
                    maxKoef = newKoef;
                }
            }
        }
        dtr = COURANT * dr / maxKoef;

        dt *= dk;
        if (dt > dtz) {
            dt = dtz;
        }
        if (dt > dtr) {
            dt = dtr;
        }
        //cout << "dt = " << dt << "\n";
    }

    void swapData() {
        for (int i = 0; i <= M + 1; ++i) {
            for (int j = 0; j <= N + 1; ++j) {
                roOld(i, j) = ro(i, j);
                uOld(i, j) = u(i, j);
                vOld(i, j) = v(i, j);
                EOld(i, j) = E(i, j);
            }
        }
    }

    void syncArr(SimpleMatrix<double> & mtr, int index) {
        
        int iB = max(igl * r1, 0),
            iE = min(1 + (igl+1)*r1, M);
        int jB = max(jgl * r2, 0),
            jE = min(1 + (jgl+1)*r2, N);

        const bool debug = false;

        if (debug) cout << "rank: " << rank << "\tiGl:" << igl << "\tjGl" << jgl << "\tiB:" << iB << "\tiE:" << iE << "\tjB:" << jB << "\tjE:" << jE << "\n";

        const int haloLen = 2;

        MPI_Status *stats = new MPI_Status[4];
        int statsIndex = 0;

        const int pass1 = 10* index + 1,
                pass2 = 10 * index + 2,
                pass3 = 10 * index + 3,
                pass4 = 10 * index + 4;

        if (jgl % 2 == 0) {
            if (rightN >= 0) {
                if (debugNum == rank) cout << "sendR " << rank << " -> " << rightN << " " << pass1 << "\n";
                mtr.sendPart(iB + haloLen, iE - haloLen, jB + haloLen, jB + haloLen*2, MPI_DOUBLE, rightN, pass1, MPI_COMM_WORLD);
            }

            if (rightN >= 0) {
                if (debugNum == rank) cout << "recvR " << rank << " <- " << rightN << " " << pass2 << "\n";    
                mtr.recvPart(iB + haloLen, iE - haloLen, jE - haloLen, jE, MPI_DOUBLE, rightN, pass2, MPI_COMM_WORLD, & stats[statsIndex++]);
            }

            if (leftN >= 0) {
                if (debugNum == rank) cout << "recvL " << rank << " <- " << leftN << " " << pass1 << "\n";
                mtr.recvPart(iB + haloLen, iE - haloLen, jB, jB + haloLen, MPI_DOUBLE, leftN, pass1, MPI_COMM_WORLD, & stats[statsIndex++]);
            }
            
            if (leftN >= 0) {
                if (debugNum == rank) cout << "sendL " << rank << " -> " << leftN << " " << pass2 << "\n";
                mtr.sendPart(iB + haloLen, iE - haloLen, jE - haloLen*2, jE - haloLen, MPI_DOUBLE, leftN, pass2, MPI_COMM_WORLD);
            }
        } else {
            if (leftN >= 0) {
                if (debugNum == rank) cout << "recvL " << rank << " <- " << leftN << " " << pass1 << "\n";
                mtr.recvPart(iB + haloLen, iE - haloLen, jB, jB + haloLen, MPI_DOUBLE, leftN, pass1, MPI_COMM_WORLD, & stats[statsIndex++]);
            }

            if (leftN >= 0) {
                if (debugNum == rank) cout << "sendL " << rank << " -> " << leftN << " " << pass2 << "\n";
                mtr.sendPart(iB + haloLen, iE - haloLen, jE - haloLen*2, jE - haloLen, MPI_DOUBLE, leftN, pass2, MPI_COMM_WORLD);
            }

            if (rightN >= 0) {
                if (debugNum == rank) cout << "sendR " << rank << " -> " << rightN << " " << pass1 << "\n";
                mtr.sendPart(iB + haloLen, iE - haloLen, jB + haloLen, jB + haloLen*2, MPI_DOUBLE, rightN, pass1, MPI_COMM_WORLD);
            }

            if (rightN >= 0) {
                if (debugNum == rank) cout << "recvR " << rank << " <- " << rightN << " " << pass2 << "\n";    
                mtr.recvPart(iB + haloLen, iE - haloLen, jE - haloLen, jE, MPI_DOUBLE, rightN, pass2, MPI_COMM_WORLD, & stats[statsIndex++]);
            }
        }

        if (igl % 2 == 0) {
            if (botN >= 0) {
                if (debugNum == rank) cout << "sendB " << rank << " -> " << botN << " " << pass3 << "\n";
                mtr.sendPart(iB + haloLen, iB + haloLen*2, jB + haloLen, jE - haloLen, MPI_DOUBLE, botN, pass3, MPI_COMM_WORLD);
            }

            if (botN >= 0) {
                if (debugNum == rank) cout << "recvB " << rank << " <- " << botN << " " << pass4 << "\n";
                mtr.recvPart(iE - haloLen, iE, jB + haloLen, jE - haloLen, MPI_DOUBLE, botN, pass4, MPI_COMM_WORLD, & stats[statsIndex++]);
            }

            if (topN >= 0) {
                if (debugNum == rank) cout << "recvT " << rank << " <- " << topN << " " << pass3 << "\n";
                mtr.recvPart(iB, iB + haloLen, jB + haloLen, jE - haloLen, MPI_DOUBLE, topN, pass3, MPI_COMM_WORLD, & stats[statsIndex++]);
            }

            if (topN >= 0) {
                if (debugNum == rank) cout << "sendT " << rank << " -> " << topN << " " << pass4 << "\n";
                mtr.sendPart(iE - haloLen*2, iE - haloLen, jB + haloLen, jE - haloLen, MPI_DOUBLE, topN, pass4, MPI_COMM_WORLD);
            }
        } else {
            if (topN >= 0) {
                if (debugNum == rank) cout << "recvT " << rank << " <- " << topN << " " << pass3 << "\n";
                mtr.recvPart(iB, iB + haloLen, jB + haloLen, jE - haloLen, MPI_DOUBLE, topN, pass3, MPI_COMM_WORLD, & stats[statsIndex++]);
            }

            if (topN >= 0) {
                if (debugNum == rank) cout << "sendT " << rank << " -> " << topN << " " << pass4 << "\n";
                mtr.sendPart(iE - haloLen*2, iE - haloLen, jB + haloLen, jE - haloLen, MPI_DOUBLE, topN, pass4, MPI_COMM_WORLD);
            }

            if (botN >= 0) {
                if (debugNum == rank) cout << "sendB " << rank << " -> " << botN << " " << pass3 << "\n";
                mtr.sendPart(iB + haloLen, iB + haloLen*2, jB + haloLen, jE - haloLen, MPI_DOUBLE, botN, pass3, MPI_COMM_WORLD);
            }

            if (botN >= 0) {
                if (debugNum == rank) cout << "recvB " << rank << " <- " << botN << " " << pass4 << "\n";
                mtr.recvPart(iE - haloLen, iE, jB + haloLen, jE - haloLen, MPI_DOUBLE, botN, pass4, MPI_COMM_WORLD, & stats[statsIndex++]);
            }
        }
    }

    void syncData() {
        int index = 0;
        syncArr(roOld, index++);
        syncArr(uOld, index++);
        syncArr(vOld, index++);
        syncArr(EOld, index++);

        roOld.wait();
        uOld.wait();
        vOld.wait();
        EOld.wait();
    }

    void calcTile() {
        stage1();
        //cout << "stage1 done\n";
        //cout << "\n\n!!! new p = \n";
        //for (int i = 0; i <= M + 1; ++i) {
        //    for (int j = 0; j <= N + 1; ++j) {
        //        cout /*<< std::fixed << std::setrorecision(2)*/ << p(i, j) << " ";
        //    }
        //    cout << std::endl;
        //}
        //cout << std::endl;

        stage2();
        //cout << "stage2 done\n";
        stage3();
        //cout << "stage3 done\n";
        stage4();
        //cout << "stage4 done\n";
        stage5();
        //cout << "stage5 done\n";

        //countNewDt();

        //swapData();

        //cout << "sync: " << rank << "\n";
        syncData();

        //if (isMaster) cout << "sync done " << currentLayer << "\n";
    }

    int run(int argc, char *argv[]) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        master = 0;
        isMaster =  rank == master;

        if ( isMaster ) {
            //Get options
            if (cmdOptionExists(argv, argv + argc, "-nn")) {
                n = atoi(getCmdOption(argv, argv + argc, "-nn"));
            }
            if (cmdOptionExists(argv, argv + argc, "-r3")) {
                r3 = atoi(getCmdOption(argv, argv + argc, "-r3"));
            }
            if (cmdOptionExists(argv, argv + argc, "-ls")) {
                layersNum = atoi(getCmdOption(argv, argv + argc, "-ls"));
            }
        }

        MPI_Bcast(&n, 1, MPI_INT, master, MPI_COMM_WORLD);
        MPI_Bcast(&r3, 1, MPI_INT, master, MPI_COMM_WORLD);
        MPI_Bcast(&layersNum, 1, MPI_INT, master, MPI_COMM_WORLD);

        
        Ab.resize(n, n + 1);
        Q2 = size / Q1;
        r1 = ceil(double(n) / Q1);
        r2 = ceil(double(n) / Q2);
        r3 = ceil(double(n) / Q3);

        igl = rank % Q1;
        jgl = rank / Q1;

        leftN = rank - 1;
        if (rank % Q2 == 0) leftN = -1;
        rightN = rank + 1;
        if (rightN % Q2 == 0) rightN = -1;

        topN = rank - Q2;
        botN = rank + Q2;
        if (botN >= Q1 * Q2) botN = -1;

        //cout << "myrank: " << rank << " left:" << leftN << " right:" << rightN << " top:" << topN << " bot:" << botN << "\n";

        if(isMaster) cout << "Q:" << Q1 << " " << Q2 << " " << Q3 << "\n" << "r: " << r1 << " " << r2 << " " << r3 << "\n";
        u.resize(n, r3);

        //initialize data
        //if (rank == master) {

            Init();            
            
            // debugInput();

            //computations
            for (currentLayer = 0; currentLayer < layersNum; ++currentLayer) {
                calcTile();
            }
            // debugOutput();
            //saveOutput();
        //}

        MPI_Finalize();
        return 0;
    }
}

int main(int argc, char *argv[]) {
    return code::run(argc, argv);
}