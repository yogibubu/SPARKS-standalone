// SPARKS batch density/gradient contractions. AO evaluation is provided by the
// existing PySCF/libcgto engine; no Gaussian evaluator is reimplemented here.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

extern "C" {
void GTOval_cart(int, int*, int*, double*, double*, std::uint8_t*, int*, int, int*, int, double*);
void GTOval_cart_deriv1(int, int*, int*, double*, double*, std::uint8_t*, int*, int, int*, int, double*);
void dgemm_(const char*, const char*, const int*, const int*, const int*,
            const double*, const double*, const int*, const double*, const int*,
            const double*, double*, const int*);
double CINTcommon_fac_sp(int);
}
extern "C" double sparks_grid_common_factor(int l) noexcept { return CINTcommon_fac_sp(l); }

namespace {
struct Grid {
    int natom, nshell, nao, nmo;
    std::vector<int> atm, bas, ao_loc;
    std::vector<double> env, coefficients, occupations;
    std::vector<double> coords, ao, mo;
    std::vector<std::uint8_t> mask;

    Grid(int na, int ns, int nf, int nm, int ne, const int* at,
         const int* bs, const int* loc, const double* ev,
         const double* cf, const double* oc)
        : natom(na), nshell(ns), nao(nf), nmo(nm),
          atm(at, at+6*na), bas(bs, bs+8*ns), ao_loc(loc, loc+ns+1),
          env(ev, ev+ne), coefficients(cf, cf+std::size_t(nf)*nm),
          occupations(oc, oc+nm) {
        if (ao_loc.front()!=0 || ao_loc.back()!=nao) throw std::invalid_argument("AO layout");
        for (int sh=0; sh<nshell; ++sh) {
            const auto* b=bas.data()+8*sh;
            const int ncart=(b[1]+1)*(b[1]+2)/2;
            if (b[0]<0 || b[0]>=natom || b[1]<0 || b[1]>4 || b[2]<1 || b[2]>64 || b[3]<1 || b[3]>64
                || b[5]<0 || b[6]<0 || b[5]+b[2]>ne || b[6]+b[2]*b[3]>ne
                || ao_loc[sh+1]-ao_loc[sh]!=ncart*b[3])
                throw std::invalid_argument("Unsupported shell layout");
        }
        for (int atom=0; atom<natom; ++atom)
            if (atm[6*atom+1]<0 || atm[6*atom+1]+3>ne) throw std::invalid_argument("Atom layout");
        if (!std::all_of(env.begin(),env.end(),[](double x){return std::isfinite(x);})
            || !std::all_of(coefficients.begin(),coefficients.end(),[](double x){return std::isfinite(x);})
            || !std::all_of(occupations.begin(),occupations.end(),[](double x){return std::isfinite(x);}))
            throw std::invalid_argument("Nonfinite state");
    }

    void evaluate(int n, const double* xyz, double* density, double* gradient, bool derivatives) {
        constexpr int block=256;
        const int components=derivatives?4:1;
        int shells[2]={0,nshell};
        const char normal='N';
        const double one=1, zero=0;
        for (int begin=0; begin<n; begin+=block) {
            const int points=std::min(block,n-begin);
            coords.resize(3*points);
            for (int axis=0; axis<3; ++axis)
                std::copy_n(xyz+std::size_t(axis)*n+begin,points,coords.data()+axis*points);
            ao.resize(std::size_t(components)*nao*points);
            mo.resize(std::size_t(components)*nmo*points);
            // All shells enabled: no numerical screening threshold is added.
            // 56 is libcgto's public grid block size (grid_ao_drv.h).
            mask.assign(std::size_t((points+55)/56)*nshell,1);
            auto evaluator=derivatives?GTOval_cart_deriv1:GTOval_cart;
            evaluator(points,shells,ao_loc.data(),ao.data(),coords.data(),mask.data(),
                      atm.data(),natom,bas.data(),nshell,env.data());
            for (int comp=0; comp<components; ++comp)
                dgemm_(&normal,&normal,&points,&nmo,&nao,&one,
                        ao.data()+std::size_t(comp)*points*nao,&points,
                        coefficients.data(),&nao,&zero,
                        mo.data()+std::size_t(comp)*points*nmo,&points);
            std::fill_n(density+begin,points,0.);
            std::vector<double> gx(derivatives?points:0,0.),gy(gx),gz(gx);
            for (int orbital=0; orbital<nmo; ++orbital) {
                const double occupation=occupations[orbital];
                if (occupation==0.) continue;
                const auto* value=mo.data()+orbital*points;
                for (int p=0; p<points; ++p) {
                    density[begin+p]+=occupation*value[p]*value[p];
                    if (derivatives) {
                        const double factor=2*occupation*value[p];
                        gx[p]+=factor*mo[std::size_t(points)*nmo+orbital*points+p];
                        gy[p]+=factor*mo[std::size_t(points)*nmo*2+orbital*points+p];
                        gz[p]+=factor*mo[std::size_t(points)*nmo*3+orbital*points+p];
                    }
                }
            }
            if (derivatives)
                for (int p=0; p<points; ++p)
                    gradient[begin+p]=std::sqrt(gx[p]*gx[p]+gy[p]*gy[p]+gz[p]*gz[p]);
            else std::fill_n(gradient+begin,points,0.);
        }
    }
};
}

extern "C" void* sparks_grid_create(int natom,int nshell,int nao,int nmo,int nenv,
        const int* atm,const int* bas,const int* ao_loc,const double* env,
        const double* coefficients,const double* occupations) noexcept {
    try {
        if (natom<1 || nshell<1 || nao<1 || nmo<1 || nenv<1 || !atm || !bas
            || !ao_loc || !env || !coefficients || !occupations) return nullptr;
        return new Grid(natom,nshell,nao,nmo,nenv,atm,bas,ao_loc,env,coefficients,occupations);
    } catch (...) { return nullptr; }
}
extern "C" void sparks_grid_destroy(void* context) noexcept { delete static_cast<Grid*>(context); }
extern "C" int sparks_grid_evaluate(void* context,int points,const double* xyz,
        double* density,double* gradient,int derivatives) noexcept {
    if (!context || points<1 || !xyz || !density || !gradient) return 1;
    try {
        static_cast<Grid*>(context)->evaluate(points,xyz,density,gradient,derivatives!=0);
        return 0;
    } catch (...) { return 2; }
}
