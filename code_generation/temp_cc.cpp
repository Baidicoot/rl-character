
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
static const int MOD = 1000000007;

ll modpow(ll a,ll e=MOD-2){
    ll r=1;
    while(e){
        if(e&1) r = r*a%MOD;
        a = a*a%MOD;
        e >>= 1;
    }
    return r;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n; ll T;
    cin >> n >> T;
    vector<ll> t(n), s(n+1);
    for(int i=0;i<n;i++){
        cin >> t[i];
        s[i+1] = s[i] + t[i];
    }

    // M = max i with s[i]+i <= T
    // R = max i with s[i] <= T
    int M=0, R=0;
    for(int i=1;i<=n;i++){
        if(s[i]+i <= T) M = i;
        if(s[i] <= T)   R = i;
    }

    ll inv2 = (MOD+1)/2;
    vector<ll> ip2(n+1);
    ip2[0]=1;
    for(int i=1;i<=n;i++){
        ip2[i] = ip2[i-1]*inv2 % MOD;
    }

    // prefix sums of ip2 for quick range-sums
    vector<ll> pref2(n+2);
    for(int i=1;i<=n+1;i++){
        pref2[i] = (pref2[i-1] + ip2[i-1]) % MOD;
    }
    auto sum_ip2 = [&](int L,int R)->ll{
        if(R < L) return 0;
        return (pref2[R+1] - pref2[L] + MOD) % MOD;
    };

    // factorials + inv-factorials for binomial
    vector<ll> fact(n+1), invf(n+1);
    fact[0]=1;
    for(int i=1;i<=n;i++){
        fact[i] = fact[i-1]*i % MOD;
    }
    invf[n] = modpow(fact[n]);
    for(int i=n;i>0;i--){
        invf[i-1] = invf[i]*i % MOD;
    }
    auto C = [&](int a,int b)->ll{
        if(b<0 || b>a) return 0;
        return fact[a]*invf[b] % MOD * invf[a-b] % MOD;
    };

    // build r[k] = max i in [M+1..R] with s[i] <= T-k
    vector<int> r(M+1);
    int ptr = R;
    for(int k=0;k<=M;k++){
        ll lim = T - k;
        while(ptr > M && s[ptr] > lim) ptr--;
        r[k] = ptr;
    }

    ll ans = M % MOD;
    if(R <= M){
        cout << ans << "\n";
        return 0;
    }

    // cur = sum_{i=M+1..r[0]} 1/2^i
    ll cur = sum_ip2(M+1, r[0]);
    ans = (ans + cur) % MOD;

    int prev_r = r[0];
    for(int k=1;k<=M;k++){
        int cr = r[k];
        // subtract out any dropped rows i in (cr..prev_r]
        for(int i=prev_r; i>cr; i--){
            cur = (cur - C(i,k)*ip2[i] % MOD + MOD) % MOD;
        }
        // now transform to the next column in O(1)
        // via the Pascal identity trick:
        cur = cur * 2 % MOD;

        ans = (ans + cur) % MOD;
        prev_r = cr;
    }

    cout << ans << "\n";
    return 0;
}
