// Density contraction of exact libcint shell/grid blocks. No AO-pair tensor
// spanning the full molecular basis is formed; no density threshold is used.
#include <algorithm>
#include <cstddef>
#include <vector>
#include "cint.h"
#include "cint_funcs.h"

extern "C" int sparks_coulomb_contract(
    CINTIntegralFunction* integral, int* atm, int natm, int* bas, int nbas,
    double* env, int* loc, CINTOpt* opt, const double* density,
    int ngrids, int tile, double* result) noexcept {
    try {
        const int nao = loc[nbas];
        std::fill(result, result + ngrids, 0.0);
        std::size_t cache_size = 0;
        int max_shell = 0;
        for (int i = 0; i < nbas; ++i) {
            max_shell = std::max(max_shell, loc[i+1]-loc[i]);
            for (int j = i; j < nbas; ++j) {
                int shells[] = {i,j,0,std::min(tile,ngrids)};
                cache_size = std::max(cache_size, static_cast<std::size_t>(
                    integral(nullptr,nullptr,shells,atm,natm,bas,nbas,env,opt,nullptr)));
            }
        }
        std::vector<double> cache(cache_size);
        std::vector<double> block(std::size_t(max_shell)*max_shell*tile);
        for (int i = 0; i < nbas; ++i) {
            const int ni = loc[i+1]-loc[i];
            for (int j = i; j < nbas; ++j) {
                const int nj = loc[j+1]-loc[j];
                bool nonzero = false;
                for (int b = 0; b < nj; ++b)
                    for (int a = 0; a < ni; ++a) {
                        const int ai=loc[i]+a, bj=loc[j]+b;
                        const double weight = density[std::size_t(bj)*nao+ai]
                            + (i==j ? 0.0 : density[std::size_t(ai)*nao+bj]);
                        nonzero = nonzero || weight != 0.0;
                    }
                if (!nonzero) continue;
                for (int start = 0; start < ngrids; start += tile) {
                    const int count=std::min(tile,ngrids-start);
                    int dims[] = {ni,nj,count};
                    int shells[] = {i,j,start,start+count};
                    if (!integral(block.data(),dims,shells,atm,natm,bas,nbas,env,opt,cache.data()))
                        continue;
                    for (int b = 0; b < nj; ++b)
                        for (int a = 0; a < ni; ++a) {
                            const int ai=loc[i]+a, bj=loc[j]+b;
                            const double weight=density[std::size_t(bj)*nao+ai]
                                +(i==j ? 0.0 : density[std::size_t(ai)*nao+bj]);
                            const double* values=block.data()+std::size_t(count)*(a+ni*b);
                            for (int g=0; g<count; ++g)
                                result[start+g] += weight*values[g];
                        }
                }
            }
        }
        return 0;
    } catch (...) { return 1; }
}
