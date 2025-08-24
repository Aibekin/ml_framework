#include "ML.h"

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

void mat_sigmoid(c_matrix *m)
{
    for (size_t i = 0; i < m->rows; ++i)
    {
        for (size_t j = 0; j < m->cols; ++j)
        {
            MAT_AT(m, i, j).value = sigmoidf(MAT_AT(m, i, j).value);
        }
    }
}

NN alloc_nn(size_t *arch, size_t arch_count)
{
    assert(arch_count > 0);

    NN nn;
    nn.count = arch_count - 1;

    nn.ws = malloc(sizeof(*nn.ws) * nn.count);
    assert(nn.ws != NULL);
    nn.bs = malloc(sizeof(*nn.bs) * nn.count);
    assert(nn.bs != NULL);
    nn.as = malloc(sizeof(*nn.as) * (nn.count + 1));
    assert(nn.as != NULL);

    nn.as[0] = alloc_matrix(1, arch[0]);
    for (size_t i = 1; i < arch_count; ++i)
    {
        nn.ws[i - 1] = alloc_matrix(nn.as[i - 1].cols, arch[i]);
        nn.bs[i - 1] = alloc_matrix(1, arch[i]);
        nn.as[i] = alloc_matrix(1, arch[i]);
    }

    return nn;
}

void zero_nn(NN *nn)
{
    for (size_t i = 0; i < nn->count; ++i)
    {
        nn->ws[i] = create_zeros(nn->ws[i].rows, nn->ws[i].cols);
        nn->bs[i] = create_zeros(nn->bs[i].rows, nn->bs[i].cols);
        nn->as[i] = create_zeros(nn->as[i].rows, nn->as[i].cols);
    }
    nn->as[nn->count] = create_zeros(nn->as[nn->count].rows, nn->as[0].cols);
}

void rand_nn(NN *nn, float low, float high)
{
    for (size_t i = 0; i < nn->count; ++i)
    {
        nn->ws[i] = create_random(nn->ws[i].rows, nn->ws[i].cols, low, high);
        nn->bs[i] = create_random(nn->bs[i].rows, nn->bs[i].cols, low, high);
    }
}

void print_nn(NN *nn, const char *name)
{
    char buffer[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i < nn->count; ++i)
    {
        snprintf(buffer, sizeof(buffer), "ws%zu", i);
        print_matrix(&nn->ws[i], buffer, 4);
        snprintf(buffer, sizeof(buffer), "bs%zu", i);
        print_matrix(&nn->bs[i], buffer, 4);
    }
    printf("]\n");
}

void forward_nn(NN *nn)
{
    for (size_t i = 0; i < nn->count; ++i)
    {
        nn->as[i + 1] = multiply(&nn->as[i], &nn->ws[i]);
        nn->as[i + 1] = addition(&nn->as[i + 1], &nn->bs[i]);
        mat_sigmoid(&nn->as[i + 1]);
    }
}

float cost_nn(NN *nn, c_matrix *in, c_matrix *out)
{
    assert(in->rows == out->rows);
    assert(out->cols == NN_OUTPUT(*nn).cols);
    size_t input_rows = in->rows;

    float cost = 0;

    for (size_t i = 0; i < input_rows; ++i)
    {
        c_matrix x = get_row(in, i);
        c_matrix y = get_row(out, i);

        copy_matrix(&NN_INPUT(*nn), &x);
        forward_nn(nn);

        size_t output_cols = out->cols;

        for (size_t j = 0; j < output_cols; ++j)
        {
            float difference = MAT_AT(&NN_OUTPUT(*nn), 0, j).value - MAT_AT(&y, 0, j).value;
            cost += difference * difference;
        }
    }

    return cost / input_rows;
}

void diff_nn(NN *nn, NN *g, float eps, c_matrix *in, c_matrix *out)
{
    float saved;
    float cost = cost_nn(nn, in, out);

    for (size_t i = 0; i < nn->count; ++i)
    {
        for (size_t j = 0; j < nn->ws[i].rows; ++j)
        {
            for (size_t k = 0; k < nn->ws[i].cols; ++k)
            {
                saved = MAT_AT(&nn->ws[i], j, k).value;
                MAT_AT(&nn->ws[i], j, k).value += eps;
                MAT_AT(&g->ws[i], j, k).value = (cost_nn(nn, in, out) - cost) / eps;
                MAT_AT(&nn->ws[i], j, k).value = saved;
            }
        }

        for (size_t j = 0; j < nn->bs[i].rows; ++j)
        {
            for (size_t k = 0; k < nn->bs[i].cols; ++k)
            {
                saved = MAT_AT(&nn->bs[i], j, k).value;
                MAT_AT(&nn->bs[i], j, k).value += eps;
                MAT_AT(&g->bs[i], j, k).value = (cost_nn(nn, in, out) - cost) / eps;
                MAT_AT(&nn->bs[i], j, k).value = saved;
            }
        }
    }
}

void backprop_nn(NN *nn, NN *g, c_matrix *in, c_matrix *out)
{
    assert(in->rows == out->rows);
    size_t n = in->rows;
    assert(NN_OUTPUT(*nn).cols == out->cols);

    zero_nn(g);

    // i - current sample
    // l - current layer
    // j - current activation
    // k - previous activation

    for (size_t i = 0; i < n; ++i)
    {
        copy_matrix(&NN_INPUT(*nn), get_prow(in, i));
        forward_nn(nn);

        for (size_t j = 0; j < nn->count; ++j)
            g->as[j] = create_zeros(g->as[j].rows, g->as[j].cols);

        for (size_t j = 0; j < out->cols; ++j)
            MAT_AT(&NN_OUTPUT(*g), 0, j).value = MAT_AT(&NN_OUTPUT(*nn), 0, j).value - MAT_AT(out, i, j).value;

        for (size_t l = nn->count; l > 0; --l)
        {
            for (size_t j = 0; j < nn->as[l].cols; ++j)
            {
                float current_a = MAT_AT(&nn->as[l], 0, j).value;
                float current_da = MAT_AT(&g->as[l], 0, j).value;
                MAT_AT(&g->bs[l - 1], 0, j).value += 2 * current_da * current_a * (1 - current_a);
                for (size_t k = 0; k < nn->as[l - 1].cols; ++k)
                {
                    // j - weight matrix col
                    // k - weight matrix row
                    float pa = MAT_AT(&nn->as[l - 1], 0, k).value;
                    float weight = MAT_AT(&nn->ws[l - 1], k, j).value;
                    MAT_AT(&g->ws[l - 1], k, j).value += 2 * current_da * current_a * (1 - current_a) * pa;
                    MAT_AT(&g->as[l - 1], 0, k).value += 2 * current_da * current_a * (1 - current_a) * weight;
                }
            }
        }
    }
    for (size_t i = 0; i < g->count; ++i)
    {
        for (size_t j = 0; j < g->ws[i].rows; ++j)
            for (size_t k = 0; k < g->ws[i].cols; ++k)
                MAT_AT(&g->ws[i], j, k).value /= n;
        for (size_t j = 0; j < g->bs[i].rows; ++j)
            for (size_t k = 0; k < g->bs[i].cols; ++k)
                MAT_AT(&g->bs[i], j, k).value /= n;
    }
}

void learn_nn(NN *nn, NN *g, float rate)
{
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < nn->ws[i].rows; ++j)
        {
            for (size_t k = 0; k < nn->ws[i].cols; ++k)
            {
                MAT_AT(&nn->ws[i], j, k).value -= rate * MAT_AT(&g->ws[i], j, k).value;
            }
        }
        for (size_t j = 0; j < nn->bs[i].rows; ++j)
        {
            for (size_t k = 0; k < nn->bs[i].cols; ++k)
            {
                MAT_AT(&nn->bs[i], j, k).value -= rate * MAT_AT(&g->bs[i], j, k).value;
            }
        }
    }
}