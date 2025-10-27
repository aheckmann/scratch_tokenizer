/**
 * Applies the exponential function (\(e^{x}\)) to each element in
 * the vector. This turns all values into positive numbers. It divides each of
 * the resulting values by the sum of all the exponential values. This ensures
 * that the final output vector's elements sum to 1, making them interpretable
 * as probabilities.
 */
export function softmax(arr: number[])  {
  const max = Math.max(...arr)

  let sum = 0
  const exp = arr.map(x => {
    // subtract max to help reduce likelihood of js number type overflow (Math.exp() can get very large)
    const val = Math.exp(x - max)
    sum += val
    return val
  })

  return {
    max,
    sum,
    exp,
    result: exp.map(x => x / sum)
  }
}
