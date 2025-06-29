#include "PCFG.h"
#include <mpi.h>
#include <vector>
#include <cstring>
#include <omp.h>
using namespace std;

// 定义批量大小
constexpr int BATCH_SIZE = 5;

constexpr int TAG_PACK = 400;

// 新增辅助函数：在不涉及 MPI 通信的情况下生成给定 PT 的所有猜测
vector<string> PriorityQueue::GeneratePT(const PT &pt) {
    vector<string> out;
    // 若PT只有一个segment，则直接遍历该segment的所有value
    if (pt.content.size() == 1) {
        segment *a = nullptr;
        if (pt.content[0].type == 1)
            a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)
            a = &m.digits[m.FindDigit(pt.content[0])];
        else
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        int total = pt.max_indices[0];
        for (int i = 0; i < total; i++)
            out.push_back(a->ordered_values[i]);
    } else {
        // 多segment：先拼接除最后一个segment之外的部分，再并行生成最后一段的所有组合
        string base;
        int last = (int)pt.curr_indices.size() - 1;
        for (int i = 0; i < last; ++i) {
            int idx = pt.curr_indices[i];
            if      (pt.content[i].type == 1)
                base += m.letters[m.FindLetter(pt.content[i])].ordered_values[idx];
            else if (pt.content[i].type == 2)
                base += m.digits[m.FindDigit(pt.content[i])].ordered_values[idx];
            else
                base += m.symbols[m.FindSymbol(pt.content[i])].ordered_values[idx];
        }
        segment *a = nullptr;
        if      (pt.content[last].type == 1)
            a = &m.letters[m.FindLetter(pt.content[last])];
        else if (pt.content[last].type == 2)
            a = &m.digits[m.FindDigit(pt.content[last])];
        else
            a = &m.symbols[m.FindSymbol(pt.content[last])];
        int total = pt.max_indices[last];
        for (int i = 0; i < total; i++)
            out.push_back(base + a->ordered_values[i]);
    }
    return out;
}

// 新增辅助函数：按照概率降序将新PT插入 priority 队列
void PriorityQueue::InsertPT(const PT &pt) {
    auto iter = priority.begin();
    for (; iter != priority.end(); ++iter) {
        if (pt.prob > iter->prob)
            break;
    }
    priority.insert(iter, pt);
}


void PriorityQueue::CalProb(PT &pt)
{
    // 计算PriorityQueue里面一个PT的流程如下：
    // 1. 首先需要计算一个PT本身的概率。例如，L6S1的概率为0.15
    // 2. 需要注意的是，Queue里面的PT不是“纯粹的”PT，而是除了最后一个segment以外，全部被value实例化的PT
    // 3. 所以，对于L6S1而言，其在Queue里面的实际PT可能是123456S1，其中“123456”为L6的一个具体value。
    // 4. 这个时候就需要计算123456在L6中出现的概率了。假设123456在所有L6 segment中的概率为0.1，那么123456S1的概率就是0.1*0.15

    // 计算一个PT本身的概率。后续所有具体segment value的概率，直接累乘在这个初始概率值上
    pt.prob = pt.preterm_prob;

    // index: 标注当前segment在PT中的位置
    int index = 0;


    for (int idx : pt.curr_indices)
    {
        // pt.content[index].PrintSeg();
        if (pt.content[index].type == 1)
        {
            // 下面这行代码的意义：
            // pt.content[index]：目前需要计算概率的segment
            // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
            // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
            // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
            // cout << m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.letters[m.FindLetter(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
            // cout << m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.digits[m.FindDigit(pt.content[index])].total_freq << endl;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx] << endl;
            // cout << m.symbols[m.FindSymbol(pt.content[index])].total_freq << endl;
        }
        index += 1;
    }
    // cout << pt.prob << endl;
}

void PriorityQueue::init()
{
    // cout << m.ordered_pts.size() << endl;
    // 用所有可能的PT，按概率降序填满整个优先队列
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                // 下面这行代码的意义：
                // max_indices用来表示PT中各个segment的可能数目。例如，L6S1中，假设模型统计到了100个L6，那么L6对应的最大下标就是99
                // （但由于后面采用了"<"的比较关系，所以其实max_indices[0]=100）
                // m.FindLetter(seg): 找到一个letter segment在模型中的对应下标
                // m.letters[m.FindLetter(seg)]：一个letter segment在模型中对应的所有统计数据
                // m.letters[m.FindLetter(seg)].ordered_values：一个letter segment在模型中，所有value的总数目
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        // pt.PrintPT();
        // cout << " " << m.preterm_freq[m.FindPT(pt)] << " " << m.total_preterm << " " << pt.preterm_prob << endl;

        // 计算当前pt的概率
        CalProb(pt);
        // 将PT放入优先队列
        priority.emplace_back(pt);
    }
    // cout << "priority size:" << priority.size() << endl;
}

// 修改后的 PopNext：一次性从 priority 队列中取出多个 PT 并并行生成猜测和新 PT
void PriorityQueue::PopNext() {
    // 若优先队列为空则直接返回
    if (priority.empty()) return;

    // 取批量数（不超过当前队列长度）
    int batch_size = (int)priority.size() < BATCH_SIZE ? (int)priority.size() : BATCH_SIZE;

    // 取出批量 PT
    vector<PT> batch(priority.begin(), priority.begin() + batch_size);
    // 从优先队列中删除这些 PT
    priority.erase(priority.begin(), priority.begin() + batch_size);

    // 存储本批生成的猜测和新 PT
    vector<string> batch_guesses;
    vector<PT> batch_new_pts;

    // 并行处理每个 PT
    #pragma omp parallel for
    for (int i = 0; i < batch.size(); i++) {
        // 每个线程处理一个 PT（拷贝副本）
        PT pt_local = batch[i];
        // 调用辅助函数生成当前 PT 的所有猜测
        vector<string> local_guesses = GeneratePT(pt_local);
        // 获取该 PT 生成的新PT集合
        vector<PT> new_pts = pt_local.NewPTs();
        // 为每个新PT计算概率
        for (PT &pt_new : new_pts) {
            CalProb(pt_new);
        }
        // 将当前线程生成的猜测和新 PT 写入共享容器（使用 omp critical 保证线程安全）
        #pragma omp critical
        {
            batch_guesses.insert(batch_guesses.end(), local_guesses.begin(), local_guesses.end());
            batch_new_pts.insert(batch_new_pts.end(), new_pts.begin(), new_pts.end());
        }
    }

    // 将本批猜测加入全局猜测列表
    guesses.insert(guesses.end(), batch_guesses.begin(), batch_guesses.end());

    // 将新生成的 PT 按正确顺序重新插入优先队列
    for (PT &pt_new : batch_new_pts) {
        InsertPT(pt_new);
    }

    // MPI Barrier，确保所有进程完成本批处理（保留原设计的同步步骤）
    MPI_Barrier(MPI_COMM_WORLD);
}

// 这个函数你就算看不懂，对并行算法的实现影响也不大
// 当然如果你想做一个基于多优先队列的并行算法，可能得稍微看一看了
vector<PT> PT::NewPTs()
{
    // 存储生成的新PT
    vector<PT> res;

    // 假如这个PT只有一个segment
    // 那么这个segment的所有value在出队前就已经被遍历完毕，并作为猜测输出
    // 因此，所有这个PT可能对应的口令猜测已经遍历完成，无需生成新的PT
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        // 最初的pivot值。我们将更改位置下标大于等于这个pivot值的segment的值（最后一个segment除外），并且一次只更改一个segment
        // 上面这句话里是不是有没看懂的地方？接着往下看你应该会更明白
        int init_pivot = pivot;

        // 开始遍历所有位置值大于等于init_pivot值的segment
        // 注意i < curr_indices.size() - 1，也就是除去了最后一个segment（这个segment的赋值预留给并行环节）
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            // curr_indices: 标记各segment目前的value在模型里对应的下标
            curr_indices[i] += 1;

            // max_indices：标记各segment在模型中一共有多少个value
            if (curr_indices[i] < max_indices[i])
            {
                // 更新pivot值
                pivot = i;
                res.emplace_back(*this);
            }

            // 这个步骤对于你理解pivot的作用、新PT生成的过程而言，至关重要
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }

    return res;
}


// 检查 MPI 错误并 abort
static void checkMPI(int err, const char *ctx) {
    if (err != MPI_SUCCESS) {
        char estr[MPI_MAX_ERROR_STRING];
        int elen;
        MPI_Error_string(err, estr, &elen);
        std::cerr << "[MPI ERROR] " << ctx << ": " << estr << "\n";
        MPI_Abort(MPI_COMM_WORLD, err);
    }
}

void PriorityQueue::Generate(PT pt) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1) 生成本地猜测
    CalProb(pt);
    std::vector<std::string> local;
    if (pt.content.size() == 1) {
        segment *a = nullptr;
        if      (pt.content[0].type == 1) a = &m.letters [m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2) a = &m.digits  [m.FindDigit (pt.content[0])];
        else                               a = &m.symbols[m.FindSymbol(pt.content[0])];
        int total = pt.max_indices[0];
        for (int i = rank; i < total; i += size)
            local.push_back(a->ordered_values[i]);
    } else {
        // 多 segment：拼 base + 并行最后一段
        std::string base;
        int last = (int)pt.curr_indices.size() - 1;
        for (int i = 0; i < last; ++i) {
            int idx = pt.curr_indices[i];
            if      (pt.content[i].type == 1) base += m.letters [m.FindLetter(pt.content[i])].ordered_values[idx];
            else if (pt.content[i].type == 2) base += m.digits  [m.FindDigit(pt.content[i])].ordered_values[idx];
            else                               base += m.symbols[m.FindSymbol(pt.content[i])].ordered_values[idx];
        }
        segment *a = nullptr;
        if      (pt.content[last].type == 1) a = &m.letters [m.FindLetter(pt.content[last])];
        else if (pt.content[last].type == 2) a = &m.digits  [m.FindDigit(pt.content[last])];
        else                                 a = &m.symbols[m.FindSymbol(pt.content[last])];
        int total = pt.max_indices[last];
        for (int i = rank; i < total; i += size)
            local.push_back(base + a->ordered_values[i]);
    }

    // 2) Barrier 确保所有 rank 都完成本地生成
    MPI_Barrier(MPI_COMM_WORLD);

    // 3) 通信：非 0 号进程打包发送到 0，0 号进程解包
    if (rank != 0) {
        // 估算 pack 大小
        int pack_sz = 0, tmp;
        MPI_Pack_size(1, MPI_INT,   MPI_COMM_WORLD, &pack_sz);
        MPI_Pack_size((int)local.size(), MPI_INT, MPI_COMM_WORLD, &tmp); pack_sz += tmp;
        int chars = 0; for (auto &s : local) chars += (int)s.size();
        MPI_Pack_size(chars, MPI_CHAR, MPI_COMM_WORLD, &tmp); pack_sz += tmp;

        std::vector<char> buf(pack_sz);
        int pos = 0;

        // pack count
        int count = (int)local.size();
        MPI_Pack(&count, 1, MPI_INT, buf.data(), pack_sz, &pos, MPI_COMM_WORLD);

        // pack each string
        for (auto &s : local) {
            int len = (int)s.size();
            MPI_Pack(&len, 1, MPI_INT, buf.data(), pack_sz, &pos, MPI_COMM_WORLD);
            if (len > 0)
                MPI_Pack(s.data(), len, MPI_CHAR, buf.data(), pack_sz, &pos, MPI_COMM_WORLD);
        }

        // send packed
        checkMPI(MPI_Send(buf.data(), pos, MPI_PACKED, 0, TAG_PACK, MPI_COMM_WORLD),
                 "MPI_Send packed guesses");
    } else {
        // rank 0：先收自己
        guesses.insert(guesses.end(), local.begin(), local.end());

        // 再从每个 src 解包
        for (int src = 1; src < size; ++src) {
            MPI_Status st;
            checkMPI(MPI_Probe(src, TAG_PACK, MPI_COMM_WORLD, &st), "MPI_Probe");

            int msg_sz;
            MPI_Get_count(&st, MPI_PACKED, &msg_sz);

            std::vector<char> buf(msg_sz);
            checkMPI(MPI_Recv(buf.data(), msg_sz, MPI_PACKED, src, TAG_PACK, MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                     "MPI_Recv packed");

            int pos = 0, count;
            MPI_Unpack(buf.data(), msg_sz, &pos, &count, 1, MPI_INT, MPI_COMM_WORLD);

            for (int i = 0; i < count; ++i) {
                int len;
                MPI_Unpack(buf.data(), msg_sz, &pos, &len, 1, MPI_INT, MPI_COMM_WORLD);

                std::string s;
                if (len > 0) {
                    s.resize(len);
                    MPI_Unpack(buf.data(), msg_sz, &pos, &s[0], len, MPI_CHAR, MPI_COMM_WORLD);
                }
                guesses.emplace_back(std::move(s));
            }
        }
        total_guesses += guesses.size();
    }

    // 4) Barrier，确保所有 rank 完成收发
    MPI_Barrier(MPI_COMM_WORLD);
}


// constexpr int TAG_PACK = 400;

// static void checkMPI(int err, const char *ctx) {
//     if (err != MPI_SUCCESS) {
//         char estr[MPI_MAX_ERROR_STRING];
//         int elen;
//         MPI_Error_string(err, estr, &elen);
//         std::cerr << "[MPI ERROR] " << ctx << ": " << estr << "\n";
//         MPI_Abort(MPI_COMM_WORLD, err);
//     }
// }

// void PriorityQueue::Generate(PT pt) {
//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     // 1) 本地生成猜测
//     CalProb(pt);
//     std::vector<std::string> local;
//     if (pt.content.size() == 1) {
//         segment *a = nullptr;
//         if      (pt.content[0].type == 1) a = &m.letters[m.FindLetter(pt.content[0])];
//         else if (pt.content[0].type == 2) a = &m.digits[m.FindDigit(pt.content[0])];
//         else                             a = &m.symbols[m.FindSymbol(pt.content[0])];
//         int total = pt.max_indices[0];
//         for (int i = rank; i < total; i += size)
//             local.push_back(a->ordered_values[i]);
//     } else {
//         // 多segment情况：拼接 base + 并行最后一段
//         std::string base;
//         int last = (int)pt.curr_indices.size() - 1;
//         for (int i = 0; i < last; ++i) {
//             int idx = pt.curr_indices[i];
//             if      (pt.content[i].type == 1)
//                 base += m.letters[m.FindLetter(pt.content[i])].ordered_values[idx];
//             else if (pt.content[i].type == 2)
//                 base += m.digits[m.FindDigit(pt.content[i])].ordered_values[idx];
//             else
//                 base += m.symbols[m.FindSymbol(pt.content[i])].ordered_values[idx];
//         }
//         segment *a = nullptr;
//         if      (pt.content[last].type == 1) a = &m.letters[m.FindLetter(pt.content[last])];
//         else if (pt.content[last].type == 2) a = &m.digits[m.FindDigit(pt.content[last])];
//         else                                 a = &m.symbols[m.FindSymbol(pt.content[last])];
//         int total = pt.max_indices[last];
//         for (int i = rank; i < total; i += size)
//             local.push_back(base + a->ordered_values[i]);
//     }

//     // 2) 通信部分采用异步与非阻塞收发，减少全局同步次数

//     if (rank != 0) {
//         // 非 root 进程：打包后用非阻塞发送给 0 号进程
//         int pack_sz = 0, tmp;
//         MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &pack_sz);
//         MPI_Pack_size((int)local.size(), MPI_INT, MPI_COMM_WORLD, &tmp); 
//         pack_sz += tmp;
//         int chars = 0; 
//         for (auto &s : local)
//             chars += (int)s.size();
//         MPI_Pack_size(chars, MPI_CHAR, MPI_COMM_WORLD, &tmp); 
//         pack_sz += tmp;

//         std::vector<char> buf(pack_sz);
//         int pos = 0;
//         int count = (int)local.size();
//         MPI_Pack(&count, 1, MPI_INT, buf.data(), pack_sz, &pos, MPI_COMM_WORLD);
//         for (auto &s : local) {
//             int len = (int)s.size();
//             MPI_Pack(&len, 1, MPI_INT, buf.data(), pack_sz, &pos, MPI_COMM_WORLD);
//             if (len > 0)
//                 MPI_Pack(s.data(), len, MPI_CHAR, buf.data(), pack_sz, &pos, MPI_COMM_WORLD);
//         }
//         MPI_Request req;
//         checkMPI(MPI_Isend(buf.data(), pos, MPI_PACKED, 0, TAG_PACK, MPI_COMM_WORLD, &req),
//                  "MPI_Isend packed guesses");
//         MPI_Wait(&req, MPI_STATUS_IGNORE);
//     } else {
//         // root 进程：先将本地生成的猜测保存，然后非阻塞轮询接收其它进程的批量数据
//         guesses.insert(guesses.end(), local.begin(), local.end());
//         total_guesses += local.size();

//         int flag = 0;
//         MPI_Status st;
//         do {
//             MPI_Iprobe(MPI_ANY_SOURCE, TAG_PACK, MPI_COMM_WORLD, &flag, &st);
//             if (flag) {
//                 int msg_sz;
//                 MPI_Get_count(&st, MPI_PACKED, &msg_sz);
//                 std::vector<char> buf(msg_sz);
//                 checkMPI(MPI_Recv(buf.data(), msg_sz, MPI_PACKED, st.MPI_SOURCE, TAG_PACK, MPI_COMM_WORLD, MPI_STATUS_IGNORE),
//                          "MPI_Recv packed guesses");
//                 int pos = 0, count_recv;
//                 MPI_Unpack(buf.data(), msg_sz, &pos, &count_recv, 1, MPI_INT, MPI_COMM_WORLD);
//                 for (int i = 0; i < count_recv; ++i) {
//                     int len;
//                     MPI_Unpack(buf.data(), msg_sz, &pos, &len, 1, MPI_INT, MPI_COMM_WORLD);
//                     std::string s;
//                     if (len > 0) {
//                         s.resize(len);
//                         MPI_Unpack(buf.data(), msg_sz, &pos, &s[0], len, MPI_CHAR, MPI_COMM_WORLD);
//                     }
//                     guesses.emplace_back(std::move(s));
//                 }
//                 total_guesses += count_recv;
//             }
//         } while (flag);
//     }
// }