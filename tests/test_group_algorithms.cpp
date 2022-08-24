#include "test_helper.hpp"
#include <numeric>

template <typename T> struct test_group_broadcast {
  bool operator()(sycl::queue &Q, test_vector<cmplx<T>> seq) {
    bool pass = true;

    // nd range shape and size
    size_t local_size = seq.size();
    size_t global_size = local_size;

    auto *in = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(local_size, Q);

    for (size_t local_id = 0; local_id < local_size; ++local_id) {
      in[local_id] =
          sycl::ext::cplx::complex<T>(seq[local_id].re, seq[local_id].im);
    }

    auto *out =
        sycl::malloc_shared<sycl::ext::cplx::complex<T>>(global_size * 3, Q);

    sycl::range<2> global_range(global_size, global_size);
    sycl::range<2> local_range(local_size, 1);

    // Treat each slice of the 2D global range as a work group
    Q.submit([&](sycl::handler &cgh) {
      sycl::stream os(1024, 256, cgh);
      cgh.parallel_for(
          sycl::nd_range<2>(global_range, local_range),
          [=](sycl::nd_item<2> it) {
            sycl::group<2> group = it.get_group();
            int group_id = it.get_group_linear_id();
            int local_id = it.get_local_linear_id();
            int global_id = it.get_global_linear_id();

            out[group_id * 3] = group_broadcast(group, in[local_id]);
            out[group_id * 3 + 1] = group_broadcast(
                group, in[local_id], sycl::group<2>::id_type(1, 0));
            out[group_id * 3 + 2] = group_broadcast(
                group, in[local_id], sycl::group<2>::linear_id_type(2));
          });
    });
    Q.wait();

    // Check output is valid
    for (int group_id = 0; group_id < 2; ++group_id) {
      pass &= (out[group_id * 3] == in[0]);
      pass &= (out[group_id * 3 + 1] == in[1]);
      pass &= (out[group_id * 3 + 2] == in[2]);
    }

    sycl::free(in, Q);
    sycl::free(out, Q);

    return pass;
  }
};

template <typename T> struct test_group_shifts {
  bool operator()(sycl::queue &Q, test_vector<cmplx<T>> seq, int delta) {
    bool pass = true;

    // nd range shape and size
    size_t n_groups = 2;
    size_t local_size = seq.size();
    size_t global_size = local_size * n_groups;

    auto *in = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(
        global_size * local_size, Q);

    for (size_t group_id = 0, global_id = 0; group_id < n_groups; ++group_id) {
      for (size_t local_id = 0; local_id < local_size;
           ++local_id, ++global_id) {
        in[global_id] =
            sycl::ext::cplx::complex<T>(seq[local_id].re, seq[local_id].im);
      }
    }

    auto *shift_left_out =
        sycl::malloc_shared<sycl::ext::cplx::complex<T>>(global_size, Q);
    auto *shift_right_out =
        sycl::malloc_shared<sycl::ext::cplx::complex<T>>(global_size, Q);

    sycl::range<1> global_range(global_size);
    sycl::range<1> local_range(local_size);

    Q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(global_range, local_range),
                       [=](sycl::nd_item<1> it) {
                         sycl::sub_group sub_group = it.get_sub_group();
                         int global_id = it.get_global_linear_id();

                         shift_left_out[global_id] = sycl::shift_group_left(
                             sub_group, in[global_id], delta);
                         shift_right_out[global_id] = sycl::shift_group_right(
                             sub_group, in[global_id], delta);
                       });
    });
    Q.wait();

    // Check output is valid, skip undefined behavior edge cases
    for (int group_id = 0, global_id = 0; group_id < n_groups; ++group_id) {
      for (int local_id = 0; local_id < local_size; ++local_id, ++global_id) {
        if (local_id < local_size - delta)
          pass &= (shift_left_out[global_id] == in[global_id + delta]);
        if (local_id > delta)
          pass &= (shift_right_out[global_id] == in[global_id - delta]);
      }
    }

    sycl::free(in, Q);
    sycl::free(shift_left_out, Q);
    sycl::free(shift_right_out, Q);

    return pass;
  }
};

template <typename T> struct test_group_permute_by_xor {
  bool operator()(sycl::queue &Q, test_vector<cmplx<T>> seq) {
    bool pass = true;

    // nd range shape and size
    size_t n_groups = 1;
    size_t local_size = seq.size();
    size_t global_size = local_size * n_groups;

    // Flip xor test is valid with size 4 and 7
    assert(local_size == 4 || local_size == 8);

    auto *in = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(
        global_size * local_size, Q);

    for (size_t group_id = 0, global_id = 0; group_id < n_groups; ++group_id) {
      for (size_t local_id = 0; local_id < local_size;
           ++local_id, ++global_id) {
        in[global_id] =
            sycl::ext::cplx::complex<T>(seq[local_id].re, seq[local_id].im);
      }
    }

    auto *out =
        sycl::malloc_shared<sycl::ext::cplx::complex<T>>(global_size, Q);

    sycl::range<1> global_range(global_size);
    sycl::range<1> local_range(local_size);

    // swap order of local values
    Q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(global_range, local_range),
                       [=](sycl::nd_item<1> it) {
                         sycl::sub_group sub_group = it.get_sub_group();
                         int global_id = it.get_global_linear_id();
                         uint local_id = it.get_local_linear_id();

                         // Swap subgroup values
                         out[global_id] = sycl::permute_group_by_xor(
                             sub_group, in[global_id], local_size - 1);
                       });
    });
    Q.wait();

    // Check output has subgroup values swapped
    for (int group_id = 0, global_id = 0; group_id < n_groups; ++group_id)
      for (int local_id = 0; local_id < local_size; ++local_id, ++global_id)
        pass &=
            (out[global_id] == in[(group_id + 1) * local_size - local_id - 1]);

    sycl::free(in, Q);
    sycl::free(out, Q);

    return pass;
  }
};

template <typename T> struct test_select_from_group {
  bool operator()(sycl::queue &Q, test_vector<cmplx<T>> seq) {
    bool pass = true;

    // nd range shape and size
    size_t n_groups = 2;
    size_t local_size = seq.size();
    size_t global_size = local_size * n_groups;

    auto *in = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(
        global_size * local_size, Q);

    for (size_t group_id = 0, global_id = 0; group_id < n_groups; ++group_id) {
      for (size_t local_id = 0; local_id < local_size;
           ++local_id, ++global_id) {
        in[global_id] =
            sycl::ext::cplx::complex<T>(seq[local_id].re, seq[local_id].im);
      }
    }

    auto *out =
        sycl::malloc_shared<sycl::ext::cplx::complex<T>>(global_size, Q);

    sycl::range<1> global_range(global_size);
    sycl::range<1> local_range(local_size);

    // swap order of local values
    Q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(global_range, local_range),
                       [=](sycl::nd_item<1> it) {
                         sycl::sub_group sub_group = it.get_sub_group();
                         uint global_id = it.get_global_linear_id();

                         // Swap subgroup values
                         out[global_id] = sycl::select_from_group(
                             sub_group, in[global_id],
                             local_size - it.get_local_linear_id() - 1);
                       });
    });
    Q.wait();

    // Check output has subgroup values swapped
    for (int group_id = 0, global_id = 0; group_id < n_groups; ++group_id)
      for (int local_id = 0; local_id < local_size; ++local_id, ++global_id)
        pass &=
            (out[global_id] == in[(group_id + 1) * local_size - local_id - 1]);

    sycl::free(in, Q);
    sycl::free(out, Q);

    return pass;
  }
};

template <typename T> struct test_reductions {
  bool operator()(sycl::queue &Q, test_vector<cmplx<T>> seq, cmplx<T> init) {
    bool pass = true;

    // nd range shape and size
    size_t n_groups = 2;
    size_t local_size = seq.size();
    size_t global_size = local_size * n_groups;

    auto *in = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(
        global_size * local_size, Q);

    for (size_t group_id = 0, global_id = 0; group_id < n_groups; ++group_id) {
      for (size_t local_id = 0; local_id < local_size;
           ++local_id, ++global_id) {
        in[global_id] =
            sycl::ext::cplx::complex<T>(seq[local_id].re, seq[local_id].im);
      }
    }

    auto *joint_reduce_1 =
        sycl::malloc_shared<sycl::ext::cplx::complex<T>>(global_size, Q);
    auto *joint_reduce_2 =
        sycl::malloc_shared<sycl::ext::cplx::complex<T>>(global_size, Q);

    auto *reduce_over_group_1 =
        sycl::malloc_shared<sycl::ext::cplx::complex<T>>(global_size, Q);
    auto *reduce_over_group_2 =
        sycl::malloc_shared<sycl::ext::cplx::complex<T>>(global_size, Q);

    sycl::range<1> global_range(global_size);
    sycl::range<1> local_range(local_size);

    // swap order of local values
    Q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(global_range, local_range),
          [=](sycl::nd_item<1> it) {
            sycl::group group = it.get_group();
            uint global_id = it.get_global_linear_id();
            int group_id = it.get_group_linear_id();

            joint_reduce_1[global_id] = sycl::ext::cplx::joint_reduce(
                group, in + group_id * local_size,
                in + (group_id + 1) * local_size,
                sycl::plus<sycl::ext::cplx::complex<T>>());

            joint_reduce_2[global_id] = sycl::ext::cplx::joint_reduce(
                group, in + group_id * local_size,
                in + (group_id + 1) * local_size,
                sycl::ext::cplx::complex<T>(init.re, init.im),
                sycl::plus<sycl::ext::cplx::complex<T>>());

            reduce_over_group_1[global_id] = sycl::ext::cplx::reduce_over_group(
                group, in[global_id],
                sycl::plus<sycl::ext::cplx::complex<T>>());

            reduce_over_group_2[global_id] = sycl::ext::cplx::reduce_over_group(
                group, in[global_id],
                sycl::ext::cplx::complex<T>(init.re, init.im),
                sycl::plus<sycl::ext::cplx::complex<T>>());
          });
    });
    Q.wait();

    // Check output has subgroup values swapped
    // for (int group_id = 0, global_id = 0; group_id < n_groups; ++group_id)
    // {
    //   pass &= (out[global_id] == in[(group_id + 1) * local_size - local_id
    //   - 1]);
    // }

    sycl::free(in, Q);
    sycl::free(joint_reduce_1, Q);
    sycl::free(joint_reduce_2, Q);
    sycl::free(reduce_over_group_1, Q);
    sycl::free(reduce_over_group_2, Q);

    return pass;
  }
};

template <typename T> struct test_exclusive_scan {
  bool operator()(sycl::queue &Q, test_vector<cmplx<T>> seq, cmplx<T> init) {
    bool pass = true;

    // nd range shape and size
    size_t n_groups = 2;
    size_t local_size = seq.size();
    size_t global_size = local_size * n_groups;

    auto *in = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(
        global_size * local_size, Q);

    for (size_t group_id = 0, global_id = 0; group_id < n_groups; ++group_id) {
      for (size_t local_id = 0; local_id < local_size;
           ++local_id, ++global_id) {
        in[global_id] =
            sycl::ext::cplx::complex<T>(seq[local_id].re, seq[local_id].im);
      }
    }

    auto *joint_exclusive_scan_1 =
        sycl::malloc_shared<sycl::ext::cplx::complex<T>>(global_size, Q);
    auto *joint_exclusive_scan_2 =
        sycl::malloc_shared<sycl::ext::cplx::complex<T>>(global_size, Q);

    auto *exclusive_scan_over_group_1 =
        sycl::malloc_shared<sycl::ext::cplx::complex<T>>(global_size, Q);
    auto *exclusive_scan_over_group_2 =
        sycl::malloc_shared<sycl::ext::cplx::complex<T>>(global_size, Q);

    sycl::range<1> global_range(global_size);
    sycl::range<1> local_range(local_size);

    // swap order of local values
    Q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(global_range, local_range),
                       [=](sycl::nd_item<1> it) {
                         sycl::group group = it.get_group();
                         uint global_id = it.get_global_linear_id();
                         int group_id = it.get_group_linear_id();

                         sycl::ext::cplx::joint_exclusive_scan(
                             group, in + group_id * local_size,
                             in + (group_id + 1) * local_size,
                             joint_exclusive_scan_1 + group_id * local_size,
                             sycl::plus<sycl::ext::cplx::complex<T>>());

                         sycl::ext::cplx::joint_exclusive_scan(
                             group, in + group_id * local_size,
                             in + (group_id + 1) * local_size,
                             joint_exclusive_scan_2 + group_id * local_size,
                             sycl::ext::cplx::complex<T>(init.re, init.im),
                             sycl::plus<sycl::ext::cplx::complex<T>>());

                         exclusive_scan_over_group_1[global_id] =
                             sycl::ext::cplx::exclusive_scan_over_group(
                                 group, in[global_id],
                                 sycl::plus<sycl::ext::cplx::complex<T>>());

                         exclusive_scan_over_group_2[global_id] =
                             sycl::ext::cplx::exclusive_scan_over_group(
                                 group, in[global_id],
                                 sycl::ext::cplx::complex<T>(init.re, init.im),
                                 sycl::plus<sycl::ext::cplx::complex<T>>());
                       });
    });
    Q.wait();

    std::vector<sycl::ext::cplx::complex<T>> ref(local_size);
    std::vector<sycl::ext::cplx::complex<T>> ref_no_init(local_size);
    auto carry = sycl::ext::cplx::complex<T>(init.re, init.im);
    for (int local_id = 0; local_id < local_size; ++local_id) {
      ref[local_id] = carry;
      carry += in[local_id];
    }
    carry = sycl::ext::cplx::complex<T>(0, 0);
    for (int local_id = 0; local_id < local_size; ++local_id) {
      ref_no_init[local_id] = carry;
      carry += in[local_id];
    }

    for (int group_id = 0, global_id = 0; group_id < n_groups; ++group_id) {
      for (int local_id = 0; local_id < local_size; ++local_id, ++global_id) {
        pass &= check_results(joint_exclusive_scan_1[global_id],
                              std::complex<T>(ref_no_init[local_id].real(),
                                              ref_no_init[local_id].imag()),
                              /*is_device*/ true);
        pass &= check_results(
            joint_exclusive_scan_2[global_id],
            std::complex<T>(ref[local_id].real(), ref[local_id].imag()),
            /*is_device*/ true);

        pass &= check_results(exclusive_scan_over_group_1[global_id],
                              std::complex<T>(ref_no_init[local_id].real(),
                                              ref_no_init[local_id].imag()),
                              /*is_device*/ true);
        pass &= check_results(
            exclusive_scan_over_group_2[global_id],
            std::complex<T>(ref[local_id].real(), ref[local_id].imag()),
            /*is_device*/ true);
      }
    }

    sycl::free(in, Q);
    sycl::free(joint_exclusive_scan_1, Q);
    sycl::free(joint_exclusive_scan_2, Q);
    sycl::free(exclusive_scan_over_group_1, Q);
    sycl::free(exclusive_scan_over_group_2, Q);

    return pass;
  }
};

template <typename T> struct test_inclusive_scan {
  bool operator()(sycl::queue &Q, test_vector<cmplx<T>> seq, cmplx<T> init) {
    bool pass = true;

    // nd range shape and size
    size_t n_groups = 2;
    size_t local_size = seq.size();
    size_t global_size = local_size * n_groups;

    auto *in = sycl::malloc_shared<sycl::ext::cplx::complex<T>>(
        global_size * local_size, Q);

    for (size_t group_id = 0, global_id = 0; group_id < n_groups; ++group_id) {
      for (size_t local_id = 0; local_id < local_size;
           ++local_id, ++global_id) {
        in[global_id] =
            sycl::ext::cplx::complex<T>(seq[local_id].re, seq[local_id].im);
      }
    }

    auto *joint_inclusive_scan_1 =
        sycl::malloc_shared<sycl::ext::cplx::complex<T>>(global_size, Q);
    auto *joint_inclusive_scan_2 =
        sycl::malloc_shared<sycl::ext::cplx::complex<T>>(global_size, Q);

    auto *inclusive_scan_over_group_1 =
        sycl::malloc_shared<sycl::ext::cplx::complex<T>>(global_size, Q);
    auto *inclusive_scan_over_group_2 =
        sycl::malloc_shared<sycl::ext::cplx::complex<T>>(global_size, Q);

    sycl::range<1> global_range(global_size);
    sycl::range<1> local_range(local_size);

    // swap order of local values
    Q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(global_range, local_range),
                       [=](sycl::nd_item<1> it) {
                         sycl::group group = it.get_group();
                         uint global_id = it.get_global_linear_id();
                         int group_id = it.get_group_linear_id();

                         sycl::ext::cplx::joint_inclusive_scan(
                             group, in + group_id * local_size,
                             in + (group_id + 1) * local_size,
                             joint_inclusive_scan_1 + group_id * local_size,
                             sycl::plus<sycl::ext::cplx::complex<T>>());

                         sycl::ext::cplx::joint_inclusive_scan(
                             group, in + group_id * local_size,
                             in + (group_id + 1) * local_size,
                             joint_inclusive_scan_2 + group_id * local_size,
                             sycl::plus<sycl::ext::cplx::complex<T>>(),
                             sycl::ext::cplx::complex<T>(init.re, init.im));

                         inclusive_scan_over_group_1[global_id] =
                             sycl::ext::cplx::inclusive_scan_over_group(
                                 group, in[global_id],
                                 sycl::plus<sycl::ext::cplx::complex<T>>());

                         inclusive_scan_over_group_2[global_id] =
                             sycl::ext::cplx::inclusive_scan_over_group(
                                 group, in[global_id],
                                 sycl::plus<sycl::ext::cplx::complex<T>>(),
                                 sycl::ext::cplx::complex<T>(init.re, init.im));
                       });
    });
    Q.wait();

    std::vector<sycl::ext::cplx::complex<T>> ref(local_size);
    std::vector<sycl::ext::cplx::complex<T>> ref_no_init(local_size);
    auto carry = sycl::ext::cplx::complex<T>(init.re, init.im);
    for (int local_id = 0; local_id < local_size; ++local_id) {
      carry += in[local_id];
      ref[local_id] = carry;
    }
    carry = sycl::ext::cplx::complex<T>(0, 0);
    for (int local_id = 0; local_id < local_size; ++local_id) {
      carry += in[local_id];
      ref_no_init[local_id] = carry;
    }

    for (int group_id = 0, global_id = 0; group_id < n_groups; ++group_id) {
      for (int local_id = 0; local_id < local_size; ++local_id, ++global_id) {
        pass &= check_results(joint_inclusive_scan_1[global_id],
                              std::complex<T>(ref_no_init[local_id].real(),
                                              ref_no_init[local_id].imag()),
                              /*is_device*/ true);
        pass &= check_results(
            joint_inclusive_scan_2[global_id],
            std::complex<T>(ref[local_id].real(), ref[local_id].imag()),
            /*is_device*/ true);

        pass &= check_results(inclusive_scan_over_group_1[global_id],
                              std::complex<T>(ref_no_init[local_id].real(),
                                              ref_no_init[local_id].imag()),
                              /*is_device*/ true);
        pass &= check_results(
            inclusive_scan_over_group_2[global_id],
            std::complex<T>(ref[local_id].real(), ref[local_id].imag()),
            /*is_device*/ true);
      }
    }

    sycl::free(in, Q);
    sycl::free(joint_inclusive_scan_1, Q);
    sycl::free(joint_inclusive_scan_2, Q);
    sycl::free(inclusive_scan_over_group_1, Q);
    sycl::free(inclusive_scan_over_group_2, Q);

    return pass;
  }
};

int main() {
  sycl::queue Q;

  bool test_failed = false;

  // Test case values
  test_vector<cmplx<double>> c = {cmplx(0, 0), cmplx(1, 1), cmplx(2, 2),
                                  cmplx(3, 3), cmplx(4, 4), cmplx(5, 5),
                                  cmplx(6, 6), cmplx(7, 7)};
  cmplx init = cmplx(4, -3);

  // Test group algorithms
  {
    bool test_passes = true;

    test_passes &= test_valid_types<test_group_broadcast>(Q, c);
    test_passes &= test_valid_types<test_group_shifts>(Q, c, /* delta */ 1);
    test_passes &= test_valid_types<test_group_shifts>(Q, c, /* delta */ 2);
    test_passes &= test_valid_types<test_group_shifts>(Q, c, /* delta */ 3);
    test_passes &= test_valid_types<test_group_permute_by_xor>(Q, c);
    test_passes &= test_valid_types<test_select_from_group>(Q, c);

    if (!test_passes) {
      std::cerr << "Group trivially copiable complex algorithms test fails\n";
      test_failed = true;
    }
  }

  // Test reduction algorithms
  {
    bool test_passes = true;

    test_passes &= test_valid_types<test_reductions>(Q, c, init);

    if (!test_passes) {
      std::cerr << "Group reduction complex algorithms test fails\n";
      test_failed = true;
    }
  }

  // Test scan algorithms
  {
    bool test_passes = true;

    test_passes &= test_valid_types<test_exclusive_scan>(Q, c, init);
    test_passes &= test_valid_types<test_inclusive_scan>(Q, c, init);

    if (!test_passes) {
      std::cerr << "Group scan complex algorithms test fails\n";
      test_failed = true;
    }
  }

  return test_failed;
}
