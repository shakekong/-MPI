#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <mpi.h>
using namespace std;
using namespace chrono;

int main(int argc, char** argv)
{
    // 初始化 MPI 环境
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 所有进程加载模型
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("./input/Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    double time_train = double(duration_cast<microseconds>(end_train - start_train).count()) 
                      * microseconds::period::num / microseconds::period::den;

    q.init();

    if (rank == 0)
    {
        cout << "here" << endl;
    }

    int curr_num = 0;
    int history = 0;
    double time_hash = 0;
    double time_guess = 0;
    auto start = system_clock::now();

    while (true)
    {
        // ==== 同步终止标志 ====
        int terminate_flag = 0;

        if (rank == 0)
        {
            // 1. 主进程判断是否该终止
            if (q.priority.empty() || (history + q.total_guesses > 10000000))
            {
                terminate_flag = 1;
            }
        }

        // 2. 广播终止标志给所有进程
        MPI_Bcast(&terminate_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // 3. 所有进程检测是否终止
        if (terminate_flag == 1) break;

        // 所有进程调用 PopNext（内部使用 MPI 通信）
        q.PopNext();

        // 只有 rank 0 进行输出和哈希
        if (rank == 0)
        {
            q.total_guesses = q.guesses.size();

            if (q.total_guesses - curr_num >= 100000)
            {
                cout << "Guesses generated: " << history + q.total_guesses << endl;
                curr_num = q.total_guesses;
            }

            if (curr_num > 1000000)
            {
                auto start_hash = system_clock::now();
                bit32 state[4];
                for (const string& pw : q.guesses)
                {
                    MD5Hash(pw, state);
                }
                auto end_hash = system_clock::now();
                time_hash += double(duration_cast<microseconds>(end_hash - start_hash).count())
                             * microseconds::period::num / microseconds::period::den;

                history += curr_num;
                curr_num = 0;
                q.guesses.clear();
            }
        }
    }

    // rank 0 输出最终信息
    if (rank == 0)
    {
        auto end = system_clock::now();
        time_guess = double(duration_cast<microseconds>(end - start).count())
                     * microseconds::period::num / microseconds::period::den;

        cout << "Guess time: " << time_guess - time_hash << " seconds" << endl;
        cout << "Hash time: " << time_hash << " seconds" << endl;
        cout << "Train time: " << time_train << " seconds" << endl;
    }

    // 清理
    MPI_Finalize();
    return 0;
}
