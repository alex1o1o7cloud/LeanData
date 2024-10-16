import Mathlib

namespace NUMINAMATH_CALUDE_y_intercept_comparison_l153_15331

def f (x : ℝ) := x^2 - 2*x + 5
def g (x : ℝ) := x^2 + 2*x + 3

theorem y_intercept_comparison : f 0 > g 0 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_comparison_l153_15331


namespace NUMINAMATH_CALUDE_smallest_divisor_l153_15322

theorem smallest_divisor : 
  let n : ℕ := 1012
  let m : ℕ := n - 4
  let divisors : List ℕ := [16, 18, 21, 28]
  (∀ d ∈ divisors, m % d = 0) ∧ 
  (∀ d ∈ divisors, d ≥ 16) ∧
  16 ∈ divisors →
  16 = (divisors.filter (λ d => m % d = 0)).minimum?.getD 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_l153_15322


namespace NUMINAMATH_CALUDE_even_function_theorem_l153_15343

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_theorem (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_positive : ∀ x > 0, f x = (1 - x) * x) : 
  ∀ x < 0, f x = -x^2 - x := by
sorry

end NUMINAMATH_CALUDE_even_function_theorem_l153_15343


namespace NUMINAMATH_CALUDE_problem_proof_l153_15392

theorem problem_proof (k n : ℤ) : 
  (5 + k) * (5 - k) = n - (2^3) → k = 2 → n = 29 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l153_15392


namespace NUMINAMATH_CALUDE_theater_empty_seats_l153_15397

/-- Given a theater with total seats and occupied seats, calculate the number of empty seats. -/
def empty_seats (total_seats occupied_seats : ℕ) : ℕ :=
  total_seats - occupied_seats

/-- Theorem: In a theater with 750 seats and 532 people watching, there are 218 empty seats. -/
theorem theater_empty_seats :
  empty_seats 750 532 = 218 := by
  sorry

end NUMINAMATH_CALUDE_theater_empty_seats_l153_15397


namespace NUMINAMATH_CALUDE_three_digit_with_three_without_five_seven_l153_15312

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k, n = k * 10 + d ∨ n = k * 100 + d ∨ ∃ m, n = k * 100 + m * 10 + d

def not_contains_digits (n : ℕ) (d₁ d₂ : ℕ) : Prop :=
  ¬(contains_digit n d₁) ∧ ¬(contains_digit n d₂)

theorem three_digit_with_three_without_five_seven (n : ℕ) :
  (is_three_digit n ∧ contains_digit n 3 ∧ not_contains_digits n 5 7) →
  ∃ S : Finset ℕ, S.card = 154 ∧ n ∈ S :=
sorry

end NUMINAMATH_CALUDE_three_digit_with_three_without_five_seven_l153_15312


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l153_15338

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {2, 4, 6, 8}

theorem intersection_of_A_and_B : A ∩ B = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l153_15338


namespace NUMINAMATH_CALUDE_outdoor_scouts_hike_l153_15302

theorem outdoor_scouts_hike (cars taxis vans buses : ℕ) 
  (people_per_car people_per_taxi people_per_van people_per_bus : ℕ) :
  cars = 5 →
  taxis = 8 →
  vans = 3 →
  buses = 2 →
  people_per_car = 4 →
  people_per_taxi = 6 →
  people_per_van = 5 →
  people_per_bus = 20 →
  cars * people_per_car + taxis * people_per_taxi + vans * people_per_van + buses * people_per_bus = 123 :=
by
  sorry

#check outdoor_scouts_hike

end NUMINAMATH_CALUDE_outdoor_scouts_hike_l153_15302


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l153_15355

theorem sum_of_squares_and_products (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → x^2 + y^2 + z^2 = 52 → x*y + y*z + z*x = 27 → x + y + z = Real.sqrt 106 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l153_15355


namespace NUMINAMATH_CALUDE_max_y_over_x_for_complex_number_l153_15307

theorem max_y_over_x_for_complex_number (x y : ℝ) :
  let z : ℂ := (x - 2) + y * I
  (Complex.abs z)^2 = 3 →
  ∃ (k : ℝ), k^2 = 3 ∧ ∀ (t : ℝ), (y / x)^2 ≤ k^2 :=
by sorry

end NUMINAMATH_CALUDE_max_y_over_x_for_complex_number_l153_15307


namespace NUMINAMATH_CALUDE_rectangle_area_l153_15309

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 186) : L * B = 2030 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l153_15309


namespace NUMINAMATH_CALUDE_solve_for_a_l153_15315

-- Define the function f
def f (x a : ℝ) : ℝ := |x + 1| + |x - a|

-- State the theorem
theorem solve_for_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, f x a ≥ 5 ↔ x ≤ -2 ∨ x > 3) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l153_15315


namespace NUMINAMATH_CALUDE_points_per_round_l153_15390

def total_points : ℕ := 300
def num_rounds : ℕ := 5

theorem points_per_round :
  ∃ (points_per_round : ℕ),
    points_per_round * num_rounds = total_points ∧
    points_per_round = 60 :=
by sorry

end NUMINAMATH_CALUDE_points_per_round_l153_15390


namespace NUMINAMATH_CALUDE_bakery_problem_solution_l153_15391

/-- Represents the problem of buying sandwiches and cakes --/
def BakeryProblem (total_money sandwich_cost cake_cost max_items : ℚ) : Prop :=
  ∃ (sandwiches cakes : ℕ),
    sandwiches * sandwich_cost + cakes * cake_cost ≤ total_money ∧
    sandwiches + cakes ≤ max_items ∧
    ∀ (s c : ℕ),
      s * sandwich_cost + c * cake_cost ≤ total_money →
      s + c ≤ max_items →
      s + c ≤ sandwiches + cakes

/-- The maximum number of items that can be bought is 12 --/
theorem bakery_problem_solution :
  BakeryProblem 50 5 (5/2) 12 →
  ∃ (sandwiches cakes : ℕ), sandwiches + cakes = 12 :=
by sorry

end NUMINAMATH_CALUDE_bakery_problem_solution_l153_15391


namespace NUMINAMATH_CALUDE_stating_count_valid_outfits_l153_15333

/-- Represents the number of red shirts -/
def red_shirts : ℕ := 7

/-- Represents the number of green shirts -/
def green_shirts : ℕ := 5

/-- Represents the number of pants -/
def pants : ℕ := 6

/-- Represents the number of green hats -/
def green_hats : ℕ := 9

/-- Represents the number of red hats -/
def red_hats : ℕ := 7

/-- Represents the total number of valid outfits -/
def total_outfits : ℕ := 1152

/-- 
Theorem stating that the number of valid outfits is 1152.
A valid outfit consists of one shirt, one pair of pants, and one hat,
where either the shirt and hat don't share the same color,
or the pants and hat don't share the same color.
-/
theorem count_valid_outfits : 
  (red_shirts * pants * green_hats) + 
  (green_shirts * pants * red_hats) + 
  (red_shirts * red_hats * pants) +
  (green_shirts * green_hats * pants) = total_outfits :=
sorry

end NUMINAMATH_CALUDE_stating_count_valid_outfits_l153_15333


namespace NUMINAMATH_CALUDE_reciprocal_of_opposite_of_negative_l153_15398

theorem reciprocal_of_opposite_of_negative : 
  (1 / -(- -3)) = -1/3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_opposite_of_negative_l153_15398


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l153_15363

theorem trig_expression_simplification (α β : ℝ) :
  (Real.cos α * Real.cos β - Real.cos (α + β)) / (Real.cos (α - β) - Real.sin α * Real.sin β) = Real.tan α * Real.tan β :=
by sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l153_15363


namespace NUMINAMATH_CALUDE_factorization_of_2a_squared_minus_8_l153_15323

theorem factorization_of_2a_squared_minus_8 (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2a_squared_minus_8_l153_15323


namespace NUMINAMATH_CALUDE_product_equals_sum_l153_15335

theorem product_equals_sum (g h : ℚ) : 
  (∀ d : ℚ, (5 * d^2 - 4 * d + g) * (4 * d^2 + h * d - 5) = 
    20 * d^4 - 31 * d^3 - 17 * d^2 + 23 * d - 10) → 
  g + h = (7 : ℚ) / 2 := by
sorry

end NUMINAMATH_CALUDE_product_equals_sum_l153_15335


namespace NUMINAMATH_CALUDE_tigers_games_played_l153_15394

theorem tigers_games_played (games_won : ℕ) (games_lost_difference : ℕ) :
  games_won = 18 →
  games_lost_difference = 21 →
  games_won + (games_won + games_lost_difference) = 57 :=
by
  sorry

end NUMINAMATH_CALUDE_tigers_games_played_l153_15394


namespace NUMINAMATH_CALUDE_circle_triangle_construction_l153_15375

theorem circle_triangle_construction (R r : ℝ) (h : R > r) :
  ∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    a^2 = 2 * (R^2 + r^2) ∧
    b^2 = 2 * (R^2 - r^2) ∧
    (π * a^2 / 4 + π * b^2 / 4 = π * R^2) ∧
    (π * a^2 / 4 - π * b^2 / 4 = π * r^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_triangle_construction_l153_15375


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l153_15396

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = q * a n)

/-- The arithmetic sequence property for three terms -/
def ArithmeticSequence (x y z : ℝ) : Prop :=
  y - x = z - y

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  GeometricSequence a q →
  ArithmeticSequence (3 * a 1) (2 * a 2) ((1/2) * a 3) →
  q = 3 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l153_15396


namespace NUMINAMATH_CALUDE_smallest_positive_largest_negative_expression_l153_15381

theorem smallest_positive_largest_negative_expression :
  ∃ (a b c d : ℚ),
    (∀ n : ℚ, n > 0 → a ≤ n) ∧
    (∀ n : ℚ, n < 0 → n ≤ b) ∧
    (∀ n : ℚ, n ≠ 0 → |c| ≤ |n|) ∧
    (d⁻¹ = d) ∧
    (a - b + c^2 - |d| = 1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_largest_negative_expression_l153_15381


namespace NUMINAMATH_CALUDE_range_of_sum_and_abs_l153_15346

theorem range_of_sum_and_abs (a b : ℝ) (ha : 1 ≤ a ∧ a ≤ 3) (hb : -4 < b ∧ b < 2) :
  1 < a + |b| ∧ a + |b| < 7 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_and_abs_l153_15346


namespace NUMINAMATH_CALUDE_f_explicit_l153_15341

/-- A quadratic function satisfying certain conditions -/
def f (x : ℝ) : ℝ := sorry

/-- f is a quadratic function -/
axiom f_quadratic : ∃ a b c : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x + c

/-- f(0) = 2 -/
axiom f_zero : f 0 = 2

/-- f(x + 1) - f(x) = 2x - 1 for any x ∈ ℝ -/
axiom f_diff (x : ℝ) : f (x + 1) - f x = 2 * x - 1

/-- The main theorem: f(x) = x^2 - 2x + 2 -/
theorem f_explicit : ∀ x : ℝ, f x = x^2 - 2*x + 2 := by sorry

end NUMINAMATH_CALUDE_f_explicit_l153_15341


namespace NUMINAMATH_CALUDE_min_sum_a_b_l153_15317

theorem min_sum_a_b (a b : ℕ+) (l : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + l = 0 ∧ a * x₂^2 + b * x₂ + l = 0) →
  (∀ x : ℝ, a * x^2 + b * x + l = 0 → abs x < 1) →
  (∀ c d : ℕ+, c + d < a + b → ¬(∃ y : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    c * x₁^2 + d * x₁ + y = 0 ∧ c * x₂^2 + d * x₂ + y = 0) ∧
    (∀ x : ℝ, c * x^2 + d * x + y = 0 → abs x < 1))) →
  a + b = 10 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_a_b_l153_15317


namespace NUMINAMATH_CALUDE_prob_five_candy_is_thirty_percent_l153_15332

-- Define the total number of eggs (can be any positive integer)
variable (total_eggs : ℕ) (h_total : total_eggs > 0)

-- Define the fractions of blue and purple eggs
def blue_fraction : ℚ := 4/5
def purple_fraction : ℚ := 1/5

-- Define the fractions of blue and purple eggs with 5 pieces of candy
def blue_five_candy_fraction : ℚ := 1/4
def purple_five_candy_fraction : ℚ := 1/2

-- Define the probability of getting 5 pieces of candy
def prob_five_candy : ℚ := blue_fraction * blue_five_candy_fraction + purple_fraction * purple_five_candy_fraction

-- Theorem: The probability of getting 5 pieces of candy is 30%
theorem prob_five_candy_is_thirty_percent : prob_five_candy = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_prob_five_candy_is_thirty_percent_l153_15332


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l153_15330

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (average_children : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : average_children = 2)
  (h3 : childless_families = 3) :
  (total_families : ℚ) * average_children / ((total_families : ℚ) - childless_families) = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l153_15330


namespace NUMINAMATH_CALUDE_negation_of_exists_prop_l153_15344

theorem negation_of_exists_prop :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exists_prop_l153_15344


namespace NUMINAMATH_CALUDE_base_conversion_equality_l153_15337

/-- Given that the base 6 number 62₆ is equal to the base b number 124ᵦ,
    prove that the unique positive integer solution for b is 4. -/
theorem base_conversion_equality : ∃! (b : ℕ), b > 0 ∧ (6 * 6 + 2) = (1 * b^2 + 2 * b + 4) := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_equality_l153_15337


namespace NUMINAMATH_CALUDE_first_four_eq_last_four_l153_15358

/-- A sequence of 0s and 1s -/
def BinarySeq := List Bool

/-- Checks if two segments of 5 terms are different -/
def differentSegments (s : BinarySeq) (i j : Nat) : Prop :=
  i < j ∧ j + 4 < s.length ∧
  (List.take 5 (List.drop i s) ≠ List.take 5 (List.drop j s))

/-- The condition that any two consecutive segments of 5 terms are different -/
def validSequence (s : BinarySeq) : Prop :=
  ∀ i j, i < j → j + 4 < s.length → differentSegments s i j

/-- S is the longest sequence satisfying the condition -/
def longestValidSequence (S : BinarySeq) : Prop :=
  validSequence S ∧ ∀ s, validSequence s → s.length ≤ S.length

theorem first_four_eq_last_four (S : BinarySeq) (h : longestValidSequence S) :
  S.take 4 = (S.reverse.take 4).reverse :=
sorry

end NUMINAMATH_CALUDE_first_four_eq_last_four_l153_15358


namespace NUMINAMATH_CALUDE_min_f_gt_min_g_l153_15384

open Set

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the condition given in the problem
variable (h : ∀ x : ℝ, ∃ x₀ : ℝ, f x > g x₀)

-- State the theorem to be proved
theorem min_f_gt_min_g : (⨅ x, f x) > (⨅ x, g x) := by sorry

end NUMINAMATH_CALUDE_min_f_gt_min_g_l153_15384


namespace NUMINAMATH_CALUDE_balloon_division_l153_15319

theorem balloon_division (n : ℕ) : 
  (∃ k : ℕ, n = 7 * k + 4) ↔ (∃ m : ℕ, n = 7 * m + 4) :=
by sorry

end NUMINAMATH_CALUDE_balloon_division_l153_15319


namespace NUMINAMATH_CALUDE_unique_prime_sum_10002_l153_15347

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem unique_prime_sum_10002 : 
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 10002 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_sum_10002_l153_15347


namespace NUMINAMATH_CALUDE_stock_percentage_return_l153_15320

/-- Calculate the percentage return on a stock given the income and investment. -/
theorem stock_percentage_return (income : ℝ) (investment : ℝ) :
  income = 650 →
  investment = 6240 →
  abs ((income / investment * 100) - 10.42) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_stock_percentage_return_l153_15320


namespace NUMINAMATH_CALUDE_integer_between_sqrt_7_and_sqrt_15_l153_15314

theorem integer_between_sqrt_7_and_sqrt_15 (a : ℤ) :
  (Real.sqrt 7 < a) ∧ (a < Real.sqrt 15) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_integer_between_sqrt_7_and_sqrt_15_l153_15314


namespace NUMINAMATH_CALUDE_smallest_k_is_60_l153_15395

def is_perfect_cube (m : ℕ) : Prop :=
  ∃ n : ℕ, m = n^3

def smallest_k : ℕ → Prop
  | k => (k > 0) ∧ 
         (is_perfect_cube (2^4 * 3^2 * 5^5 * k)) ∧ 
         (∀ j : ℕ, j > 0 → j < k → ¬(is_perfect_cube (2^4 * 3^2 * 5^5 * j)))

theorem smallest_k_is_60 : smallest_k 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_is_60_l153_15395


namespace NUMINAMATH_CALUDE_sum_of_numbers_l153_15371

/-- Given two positive integers satisfying certain conditions, prove their sum is 36 -/
theorem sum_of_numbers (a b : ℕ+) 
  (hcf : Nat.gcd a b = 3)
  (lcm : Nat.lcm a b = 100)
  (sum_reciprocals : (a : ℚ)⁻¹ + (b : ℚ)⁻¹ = 0.3433333333333333) :
  a + b = 36 := by sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l153_15371


namespace NUMINAMATH_CALUDE_largest_distinct_digits_divisible_by_99_l153_15359

def is_distinct_digits (n : ℕ) : Prop :=
  ∀ i j, i ≠ j → (n.digits 10).nthLe i (by sorry) ≠ (n.digits 10).nthLe j (by sorry)

theorem largest_distinct_digits_divisible_by_99 :
  ∀ n : ℕ, n > 9876524130 → ¬(is_distinct_digits n ∧ n % 99 = 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_distinct_digits_divisible_by_99_l153_15359


namespace NUMINAMATH_CALUDE_merchant_profit_percentage_l153_15362

theorem merchant_profit_percentage (C S : ℝ) (h : 17 * C = 16 * S) :
  (S - C) / C * 100 = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_merchant_profit_percentage_l153_15362


namespace NUMINAMATH_CALUDE_pool_capacity_exceeds_max_l153_15399

-- Define the constants from the problem
def totalMaxCapacity : ℝ := 5000

-- Define the capacities of each section
def sectionACapacity : ℝ := 3000
def sectionBCapacity : ℝ := 2333.33
def sectionCCapacity : ℝ := 2000

-- Define the theorem
theorem pool_capacity_exceeds_max : 
  sectionACapacity + sectionBCapacity + sectionCCapacity > totalMaxCapacity :=
by sorry

end NUMINAMATH_CALUDE_pool_capacity_exceeds_max_l153_15399


namespace NUMINAMATH_CALUDE_log_10_7_exists_function_l153_15373

-- Define the variables and conditions
variable (r s : ℝ)
variable (h1 : Real.log 3 / Real.log 4 = r)
variable (h2 : Real.log 5 / Real.log 7 = s)

-- State the theorem
theorem log_10_7_exists_function (r s : ℝ) (h1 : Real.log 3 / Real.log 4 = r) (h2 : Real.log 5 / Real.log 7 = s) :
  ∃ f : ℝ → ℝ → ℝ, Real.log 7 / Real.log 10 = f r s := by
  sorry

end NUMINAMATH_CALUDE_log_10_7_exists_function_l153_15373


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l153_15324

theorem arithmetic_mean_of_fractions : 
  (3/8 + 5/9) / 2 = 67/144 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l153_15324


namespace NUMINAMATH_CALUDE_geometry_propositions_l153_15326

theorem geometry_propositions (p₁ p₂ p₃ p₄ : Prop) 
  (h₁ : p₁) (h₂ : ¬p₂) (h₃ : ¬p₃) (h₄ : p₄) :
  (p₁ ∧ p₄) ∧ (¬p₂ ∨ p₃) ∧ (¬p₃ ∨ ¬p₄) ∧ ¬(p₁ ∧ p₂) :=
sorry

end NUMINAMATH_CALUDE_geometry_propositions_l153_15326


namespace NUMINAMATH_CALUDE_rational_function_sum_l153_15349

/-- A rational function with specific properties -/
def RationalFunction (p q : ℝ → ℝ) : Prop :=
  (∃ k a : ℝ, q = fun x ↦ k * (x + 3) * (x - 1) * (x - a)) ∧
  (∃ b : ℝ, p = fun x ↦ b * x + 2) ∧
  q 0 = -2

/-- The theorem statement -/
theorem rational_function_sum (p q : ℝ → ℝ) :
  RationalFunction p q →
  ∃! p, p + q = fun x ↦ (1/3) * x^3 - (1/3) * x^2 + (11/3) * x + 4 :=
by sorry

end NUMINAMATH_CALUDE_rational_function_sum_l153_15349


namespace NUMINAMATH_CALUDE_rectangle_area_proof_l153_15342

theorem rectangle_area_proof : 
  let smaller_rectangle_short_side : ℝ := 4
  let smaller_rectangle_long_side : ℝ := 2 * smaller_rectangle_short_side
  let larger_rectangle_width : ℝ := 4 * smaller_rectangle_long_side
  let larger_rectangle_length : ℝ := smaller_rectangle_short_side
  larger_rectangle_length * larger_rectangle_width = 128 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_proof_l153_15342


namespace NUMINAMATH_CALUDE_balls_in_boxes_count_l153_15357

/-- The number of ways to place three distinct balls into three distinct boxes -/
def place_balls_in_boxes : ℕ := 27

/-- The number of choices for each ball -/
def choices_per_ball : ℕ := 3

/-- Theorem: The number of ways to place three distinct balls into three distinct boxes
    is equal to the cube of the number of choices for each ball -/
theorem balls_in_boxes_count :
  place_balls_in_boxes = choices_per_ball ^ 3 := by sorry

end NUMINAMATH_CALUDE_balls_in_boxes_count_l153_15357


namespace NUMINAMATH_CALUDE_no_solution_condition_l153_15321

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, 3 * |x + 3*a| + |x + a^2| + 2*x ≠ a) ↔ (a < 0 ∨ a > 10) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_condition_l153_15321


namespace NUMINAMATH_CALUDE_quadratic_tangent_to_x_axis_l153_15376

/-- A quadratic function f(x) = ax^2 + bx + c is tangent to the x-axis
    if and only if c = b^2 / (4a) -/
theorem quadratic_tangent_to_x_axis (a b c : ℝ) (h : c = b^2 / (4 * a)) :
  let f := λ x : ℝ => a * x^2 + b * x + c
  ∃ x₀ : ℝ, f x₀ = 0 ∧ ∀ x : ℝ, x ≠ x₀ → f x > 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_tangent_to_x_axis_l153_15376


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l153_15306

theorem polynomial_evaluation (x : ℝ) (h : x = 1 + Real.sqrt 2) : 
  x^4 - 4*x^3 + 4*x^2 + 4 = 5 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l153_15306


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l153_15352

theorem polynomial_division_theorem (x : ℝ) : 
  x^6 + 2*x^2 + 8 = (x - 1)*(x^5 + x^4 + x^3 + x^2 + 3*x + 3) + 11 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l153_15352


namespace NUMINAMATH_CALUDE_sum_squares_35_consecutive_divisible_by_35_l153_15340

theorem sum_squares_35_consecutive_divisible_by_35 (n : ℕ+) :
  ∃ k : ℤ, (((n + 35) * (n + 36) * (2 * (n + 35) + 1)) / 6 -
            (n * (n + 1) * (2 * n + 1)) / 6) = 35 * k :=
sorry

end NUMINAMATH_CALUDE_sum_squares_35_consecutive_divisible_by_35_l153_15340


namespace NUMINAMATH_CALUDE_net_sales_for_10000_yuan_l153_15393

/-- Represents the relationship between advertising expenses and sales revenue -/
def sales_model (x : ℝ) : ℝ := 8.5 * x + 17.5

/-- Calculates the net sales revenue given advertising expenses -/
def net_sales_revenue (x : ℝ) : ℝ := sales_model x - x

/-- Theorem: When advertising expenses are 1 (10,000 yuan), 
    the net sales revenue is 9.25 (92,500 yuan) -/
theorem net_sales_for_10000_yuan : net_sales_revenue 1 = 9.25 := by
  sorry

end NUMINAMATH_CALUDE_net_sales_for_10000_yuan_l153_15393


namespace NUMINAMATH_CALUDE_complement_of_M_l153_15386

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {4, 5}

theorem complement_of_M :
  (U \ M) = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l153_15386


namespace NUMINAMATH_CALUDE_modulus_of_z_is_sqrt_2_l153_15310

theorem modulus_of_z_is_sqrt_2 :
  let z : ℂ := 1 - 1 / Complex.I
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_is_sqrt_2_l153_15310


namespace NUMINAMATH_CALUDE_probability_theorem_l153_15356

/-- Represents a standard deck of cards with additional properties -/
structure Deck :=
  (total : ℕ)
  (kings : ℕ)
  (aces : ℕ)
  (others : ℕ)
  (h1 : total = kings + aces + others)

/-- The probability of drawing either two aces or at least one king -/
def probability_two_aces_or_king (d : Deck) : ℚ :=
  sorry

/-- The theorem statement -/
theorem probability_theorem (d : Deck) 
  (h2 : d.total = 54) 
  (h3 : d.kings = 4) 
  (h4 : d.aces = 6) 
  (h5 : d.others = 44) : 
  probability_two_aces_or_king d = 221 / 1431 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l153_15356


namespace NUMINAMATH_CALUDE_sum_of_digits_of_B_is_seven_l153_15377

theorem sum_of_digits_of_B_is_seven :
  ∃ (A B : ℕ),
    (A ≡ (16^16 : ℕ) [MOD 9]) →
    (B ≡ A [MOD 9]) →
    (∃ (C : ℕ), C < 10 ∧ C ≡ B [MOD 9] ∧ C = 7) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_B_is_seven_l153_15377


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l153_15334

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - x + a > 0) → a > (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l153_15334


namespace NUMINAMATH_CALUDE_f_has_one_zero_in_interval_l153_15388

def f (x : ℝ) := 2 * x^3 - 6 * x^2 + 7

theorem f_has_one_zero_in_interval :
  ∃! x : ℝ, 0 < x ∧ x < 2 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_f_has_one_zero_in_interval_l153_15388


namespace NUMINAMATH_CALUDE_root_sum_product_l153_15318

/-- Given two polynomials with specified roots, prove that u = 32 -/
theorem root_sum_product (α β γ : ℂ) (q s u : ℂ) : 
  (α^3 + 4*α^2 + 6*α - 8 = 0) → 
  (β^3 + 4*β^2 + 6*β - 8 = 0) → 
  (γ^3 + 4*γ^2 + 6*γ - 8 = 0) →
  ((α+β)^3 + q*(α+β)^2 + s*(α+β) + u = 0) →
  ((β+γ)^3 + q*(β+γ)^2 + s*(β+γ) + u = 0) →
  ((γ+α)^3 + q*(γ+α)^2 + s*(γ+α) + u = 0) →
  u = 32 := by
sorry

end NUMINAMATH_CALUDE_root_sum_product_l153_15318


namespace NUMINAMATH_CALUDE_percentage_married_employees_l153_15353

theorem percentage_married_employees (total : ℝ) (total_pos : 0 < total) : 
  let women_ratio : ℝ := 0.76
  let men_ratio : ℝ := 1 - women_ratio
  let married_women_ratio : ℝ := 0.6842
  let single_men_ratio : ℝ := 2/3
  let married_men_ratio : ℝ := 1 - single_men_ratio
  let married_ratio : ℝ := women_ratio * married_women_ratio + men_ratio * married_men_ratio
  married_ratio = 0.600392 :=
sorry

end NUMINAMATH_CALUDE_percentage_married_employees_l153_15353


namespace NUMINAMATH_CALUDE_unique_integer_solution_l153_15367

theorem unique_integer_solution (m n : ℤ) :
  (m + n)^4 = m^2*n^2 + m^2 + n^2 + 6*m*n → m = 0 ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l153_15367


namespace NUMINAMATH_CALUDE_martha_black_butterflies_l153_15369

/-- The number of black butterflies in Martha's collection --/
def num_black_butterflies (total : ℕ) (blue : ℕ) (red : ℕ) : ℕ :=
  total - (blue + red)

/-- Proof that Martha has 34 black butterflies --/
theorem martha_black_butterflies :
  num_black_butterflies 56 12 10 = 34 := by
  sorry

end NUMINAMATH_CALUDE_martha_black_butterflies_l153_15369


namespace NUMINAMATH_CALUDE_roots_of_equation_l153_15365

def equation (x : ℝ) : ℝ := x * (2*x - 5)^2 * (x + 3) * (7 - x)

theorem roots_of_equation :
  {x : ℝ | equation x = 0} = {0, 2.5, -3, 7} := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l153_15365


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l153_15370

theorem smallest_sum_of_squares (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a > 0 ∧ b > 0 ∧ c > 0 →
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    a + b = x^2 ∧ b + c = y^2 ∧ c + a = z^2 →
  55 ≤ a + b + c :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l153_15370


namespace NUMINAMATH_CALUDE_sufficient_messages_l153_15328

/-- Represents the order relation between cards -/
def beats (n m : ℕ) : Prop :=
  (n > m ∧ n ≠ 1 ∧ m ≠ 100) ∨ (n = 1 ∧ m = 100)

/-- A permutation of the first 100 natural numbers -/
def CardArrangement := { f : ℕ → ℕ // Function.Injective f ∧ ∀ n, f n ≤ 100 }

/-- A comparison message from the dealer -/
structure Message where
  i : ℕ
  j : ℕ
  result : Bool

/-- The theorem stating that 100 messages are sufficient -/
theorem sufficient_messages (arr : CardArrangement) :
  ∃ (messages : Finset Message),
    messages.card = 100 ∧
    (∀ (msg : Message), msg ∈ messages → beats (arr.val msg.i) (arr.val msg.j) = msg.result) →
    ∀ n ≤ 100, ∃! k, arr.val k = n :=
sorry

end NUMINAMATH_CALUDE_sufficient_messages_l153_15328


namespace NUMINAMATH_CALUDE_symmetric_line_equation_l153_15361

/-- Given a fold line y = -x and a line l₁ with equation 2x + 3y - 1 = 0,
    the symmetric line l₂ with respect to the fold line has the equation 3x + 2y + 1 = 0 -/
theorem symmetric_line_equation (x y : ℝ) :
  (y = -x) →  -- fold line equation
  (2*x + 3*y - 1 = 0) →  -- l₁ equation
  (3*x + 2*y + 1 = 0)  -- l₂ equation (to be proved)
:= by sorry

end NUMINAMATH_CALUDE_symmetric_line_equation_l153_15361


namespace NUMINAMATH_CALUDE_minimum_m_for_inequality_l153_15364

open Real

theorem minimum_m_for_inequality (m : ℝ) :
  (∀ x > 0, (log x - (1/2) * m * x^2 + x) ≤ m * x - 1) ↔ m ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_minimum_m_for_inequality_l153_15364


namespace NUMINAMATH_CALUDE_complementary_angles_ratio_l153_15303

theorem complementary_angles_ratio (a b : ℝ) : 
  a > 0 → b > 0 → -- angles are positive
  a + b = 90 → -- angles are complementary
  a = 4 * b → -- ratio of angles is 4:1
  b = 18 := by
sorry

end NUMINAMATH_CALUDE_complementary_angles_ratio_l153_15303


namespace NUMINAMATH_CALUDE_max_sequence_length_is_17_l153_15380

/-- The maximum length of a sequence satisfying the given conditions -/
def max_sequence_length : ℕ := 17

/-- A sequence of integers from 1 to 4 -/
def valid_sequence (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ i, i ≤ k → 1 ≤ a i ∧ a i ≤ 4

/-- The uniqueness condition for consecutive pairs in the sequence -/
def unique_pairs (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ i j, i < k → j < k → a i = a j → a (i + 1) = a (j + 1) → i = j

/-- The main theorem stating that 17 is the maximum length of a valid sequence with unique pairs -/
theorem max_sequence_length_is_17 :
  ∀ k : ℕ, (∃ a : ℕ → ℕ, valid_sequence a k ∧ unique_pairs a k) →
  k ≤ max_sequence_length :=
sorry

end NUMINAMATH_CALUDE_max_sequence_length_is_17_l153_15380


namespace NUMINAMATH_CALUDE_violet_hiking_time_l153_15300

/-- Calculates the maximum hiking time for Violet and her dog given their water consumption rates and Violet's water carrying capacity. -/
theorem violet_hiking_time (violet_rate : ℝ) (dog_rate : ℝ) (water_capacity : ℝ) :
  violet_rate = 800 →
  dog_rate = 400 →
  water_capacity = 4800 →
  (water_capacity / (violet_rate + dog_rate) : ℝ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_violet_hiking_time_l153_15300


namespace NUMINAMATH_CALUDE_soap_bubble_thickness_l153_15387

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem soap_bubble_thickness : toScientificNotation 0.0000007 = 
  { coefficient := 7,
    exponent := -7,
    is_valid := by sorry } := by sorry

end NUMINAMATH_CALUDE_soap_bubble_thickness_l153_15387


namespace NUMINAMATH_CALUDE_part_one_part_two_l153_15308

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 3*a|

-- Part I
theorem part_one : 
  {x : ℝ | f 1 x > 5 - |2*x - 1|} = {x : ℝ | x < -1/3 ∨ x > 3} := by sorry

-- Part II
theorem part_two (a : ℝ) : 
  (∃ x₀ : ℝ, f a x₀ + x₀ < 6) → a < 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l153_15308


namespace NUMINAMATH_CALUDE_count_special_numbers_is_324_l153_15316

/-- The count of 5-digit numbers beginning with 2 that have exactly three identical digits (which are not 2) -/
def count_special_numbers : ℕ :=
  4 * 9 * 9

/-- The theorem stating that the count of special numbers is 324 -/
theorem count_special_numbers_is_324 : count_special_numbers = 324 := by
  sorry

end NUMINAMATH_CALUDE_count_special_numbers_is_324_l153_15316


namespace NUMINAMATH_CALUDE_sara_sister_notebooks_l153_15368

def calculate_notebooks (initial : ℕ) (increase_percent : ℕ) (lost : ℕ) : ℕ :=
  let increased : ℕ := initial + initial * increase_percent / 100
  increased - lost

theorem sara_sister_notebooks : calculate_notebooks 4 150 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sara_sister_notebooks_l153_15368


namespace NUMINAMATH_CALUDE_meal_combinations_eq_100_l153_15378

/-- The number of items on the menu -/
def menu_items : ℕ := 10

/-- The number of people ordering -/
def num_people : ℕ := 2

/-- The number of different combinations of meals that can be ordered -/
def meal_combinations : ℕ := menu_items ^ num_people

/-- Theorem stating that the number of meal combinations is 100 -/
theorem meal_combinations_eq_100 : meal_combinations = 100 := by
  sorry

end NUMINAMATH_CALUDE_meal_combinations_eq_100_l153_15378


namespace NUMINAMATH_CALUDE_largest_c_for_seven_in_range_l153_15327

def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + c

theorem largest_c_for_seven_in_range :
  (∃ (c : ℝ), ∀ (d : ℝ), (∃ (x : ℝ), f d x = 7) → d ≤ c) ∧
  (∃ (x : ℝ), f (37/4) x = 7) :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_seven_in_range_l153_15327


namespace NUMINAMATH_CALUDE_james_recovery_time_l153_15336

/-- Calculates the total number of days before James can resume heavy lifting after an injury -/
def time_to_resume_heavy_lifting (
  initial_pain_duration : ℕ
  ) (healing_time_multiplier : ℕ
  ) (additional_caution_period : ℕ
  ) (light_exercises_duration : ℕ
  ) (potential_complication_duration : ℕ
  ) (moderate_intensity_duration : ℕ
  ) (transition_to_heavy_lifting : ℕ
  ) : ℕ :=
  let initial_healing_time := initial_pain_duration * healing_time_multiplier
  let total_initial_recovery := initial_healing_time + additional_caution_period
  let light_exercises_with_complication := light_exercises_duration + potential_complication_duration
  let total_before_transition := total_initial_recovery + light_exercises_with_complication + moderate_intensity_duration
  total_before_transition + transition_to_heavy_lifting

/-- Theorem stating that given the specific conditions, James will take 67 days to resume heavy lifting -/
theorem james_recovery_time : 
  time_to_resume_heavy_lifting 3 5 3 14 7 7 21 = 67 := by
  sorry

end NUMINAMATH_CALUDE_james_recovery_time_l153_15336


namespace NUMINAMATH_CALUDE_marble_arrangement_theorem_l153_15374

def num_marbles : ℕ := 4

def num_arrangements (n : ℕ) : ℕ := n.factorial

def num_adjacent_arrangements (n : ℕ) : ℕ := 2 * ((n - 1).factorial)

theorem marble_arrangement_theorem :
  num_arrangements num_marbles - num_adjacent_arrangements num_marbles = 12 :=
by sorry

end NUMINAMATH_CALUDE_marble_arrangement_theorem_l153_15374


namespace NUMINAMATH_CALUDE_grapes_pineapple_cost_l153_15348

/-- Represents the cost of fruit items --/
structure FruitCosts where
  oranges : ℝ
  grapes : ℝ
  pineapple : ℝ
  strawberries : ℝ

/-- The total cost of all fruits is $24 --/
def total_cost (fc : FruitCosts) : Prop :=
  fc.oranges + fc.grapes + fc.pineapple + fc.strawberries = 24

/-- The box of strawberries costs twice as much as the bag of oranges --/
def strawberry_orange_relation (fc : FruitCosts) : Prop :=
  fc.strawberries = 2 * fc.oranges

/-- The price of pineapple equals the price of oranges minus the price of grapes --/
def pineapple_relation (fc : FruitCosts) : Prop :=
  fc.pineapple = fc.oranges - fc.grapes

/-- The main theorem: Given the conditions, the cost of grapes and pineapple together is $6 --/
theorem grapes_pineapple_cost (fc : FruitCosts) 
  (h1 : total_cost fc) 
  (h2 : strawberry_orange_relation fc) 
  (h3 : pineapple_relation fc) : 
  fc.grapes + fc.pineapple = 6 := by
  sorry

end NUMINAMATH_CALUDE_grapes_pineapple_cost_l153_15348


namespace NUMINAMATH_CALUDE_inequality_proof_l153_15305

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : |x - y| < 2) (hyz : |y - z| < 2) (hzx : |z - x| < 2) :
  Real.sqrt (x * y + 1) + Real.sqrt (y * z + 1) + Real.sqrt (z * x + 1) > x + y + z :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l153_15305


namespace NUMINAMATH_CALUDE_total_weight_carrots_cucumbers_l153_15383

def carrot_weight : ℝ := 250
def cucumber_multiplier : ℝ := 2.5

theorem total_weight_carrots_cucumbers : 
  carrot_weight + cucumber_multiplier * carrot_weight = 875 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_carrots_cucumbers_l153_15383


namespace NUMINAMATH_CALUDE_stanley_run_distance_l153_15372

/-- Stanley's walking and running problem -/
theorem stanley_run_distance (walked : Real) (ran_extra : Real) : walked = 0.2 → ran_extra = 0.2 → walked + ran_extra = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_stanley_run_distance_l153_15372


namespace NUMINAMATH_CALUDE_commemorative_coin_strategy_exists_l153_15329

/-- Represents a coin with a weight -/
structure Coin where
  weight : ℕ

/-- Represents the result of a weighing -/
inductive WeighResult
  | Left
  | Right
  | Equal

/-- Represents a weighing strategy -/
def Strategy := List Coin → List Coin → WeighResult

/-- Represents the state of knowledge about the coins -/
structure CoinState where
  coins : List Coin
  commemorative : Coin

/-- The maximum number of weighings allowed -/
def maxWeighings : ℕ := 3

/-- Theorem stating that a strategy exists to determine the weight of the commemorative coin -/
theorem commemorative_coin_strategy_exists :
  ∃ (strategy : List (CoinState → Strategy)),
    ∀ (state : CoinState),
      (state.coins.length = 16) →
      (state.coins.filter (λ c => c.weight = 11)).length = 8 →
      (state.coins.filter (λ c => c.weight = 10)).length = 8 →
      (state.commemorative ∈ state.coins) →
      (∃ (result : Bool), 
        (result = true → state.commemorative.weight = 11) ∧
        (result = false → state.commemorative.weight = 10)) :=
sorry

end NUMINAMATH_CALUDE_commemorative_coin_strategy_exists_l153_15329


namespace NUMINAMATH_CALUDE_complex_subtraction_l153_15354

theorem complex_subtraction : (4 : ℂ) - 3*I - ((2 : ℂ) + 5*I) = (2 : ℂ) - 8*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l153_15354


namespace NUMINAMATH_CALUDE_line_slope_l153_15379

/-- The slope of the line given by the equation x/4 + y/3 = 1 is -3/4 -/
theorem line_slope (x y : ℝ) :
  x / 4 + y / 3 = 1 → (∃ b : ℝ, y = -(3/4) * x + b) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_l153_15379


namespace NUMINAMATH_CALUDE_sum_range_l153_15325

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -2*x - x^2 else |Real.log x|

theorem sum_range (a b c d : ℝ) :
  a < b ∧ b < c ∧ c < d ∧ f a = f b ∧ f b = f c ∧ f c = f d →
  1 < a + b + c + 2*d ∧ a + b + c + 2*d < 181/10 :=
sorry

end NUMINAMATH_CALUDE_sum_range_l153_15325


namespace NUMINAMATH_CALUDE_slope_plus_intercept_equals_two_thirds_l153_15389

-- Define a line passing through two points
def line_through_points (x₁ y₁ x₂ y₂ : ℚ) : ℚ × ℚ :=
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  (m, b)

-- Theorem statement
theorem slope_plus_intercept_equals_two_thirds :
  let (m, b) := line_through_points 2 (-1) (-1) 4
  m + b = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_slope_plus_intercept_equals_two_thirds_l153_15389


namespace NUMINAMATH_CALUDE_count_special_numbers_is_4032_l153_15360

/-- A function that counts the number of 5-digit numbers starting with '2' and having exactly two identical digits -/
def count_special_numbers : ℕ :=
  let digits := 5
  let start_digit := 2
  let identical_digits := 2
  -- The actual counting logic would go here
  4032

/-- Theorem stating that the count of special numbers is 4032 -/
theorem count_special_numbers_is_4032 :
  count_special_numbers = 4032 := by sorry

end NUMINAMATH_CALUDE_count_special_numbers_is_4032_l153_15360


namespace NUMINAMATH_CALUDE_domain_of_g_l153_15301

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f (x : ℝ) : Prop := 0 ≤ x + 1 ∧ x + 1 ≤ 2

-- Define the function g
def g (x : ℝ) : ℝ := f (x + 3)

-- Theorem statement
theorem domain_of_g :
  (∀ x, domain_f x ↔ 0 ≤ x + 1 ∧ x + 1 ≤ 2) →
  (∀ x, g x = f (x + 3)) →
  (∀ x, g x ≠ 0 ↔ -3 ≤ x ∧ x ≤ -1) :=
sorry

end NUMINAMATH_CALUDE_domain_of_g_l153_15301


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l153_15385

theorem sum_of_reciprocals_squared (a b c d : ℝ) : 
  a = Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 + 2 →
  b = -Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 + 2 →
  c = Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 + 2 →
  d = -Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 + 2 →
  (1/a + 1/b + 1/c + 1/d)^2 = 39/140 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l153_15385


namespace NUMINAMATH_CALUDE_sheilas_monthly_savings_l153_15350

/-- Calculates the monthly savings amount given the initial savings, family contribution, 
    savings period in years, and final amount in the piggy bank. -/
def monthlySavings (initialSavings familyContribution : ℕ) (savingsPeriodYears : ℕ) 
    (finalAmount : ℕ) : ℚ :=
  let totalInitialAmount := initialSavings + familyContribution
  let amountToSave := finalAmount - totalInitialAmount
  let monthsInPeriod := savingsPeriodYears * 12
  (amountToSave : ℚ) / (monthsInPeriod : ℚ)

/-- Theorem stating that Sheila's monthly savings is $276 -/
theorem sheilas_monthly_savings : 
  monthlySavings 3000 7000 4 23248 = 276 := by
  sorry

end NUMINAMATH_CALUDE_sheilas_monthly_savings_l153_15350


namespace NUMINAMATH_CALUDE_area_outside_small_inside_large_l153_15311

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the area of the region outside the smaller circle and inside the two larger circles -/
def areaOutsideSmallInsideLarge (smallCircle : Circle) (largeCircle1 largeCircle2 : Circle) : ℝ :=
  sorry

/-- Theorem stating the area of the specific configuration -/
theorem area_outside_small_inside_large :
  let smallCircle : Circle := { center := (0, 0), radius := 2 }
  let largeCircle1 : Circle := { center := (0, -2), radius := 3 }
  let largeCircle2 : Circle := { center := (0, 2), radius := 3 }
  areaOutsideSmallInsideLarge smallCircle largeCircle1 largeCircle2 = (5 * Real.pi / 2) - 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_area_outside_small_inside_large_l153_15311


namespace NUMINAMATH_CALUDE_volume_of_five_cubes_l153_15313

/-- The volume of a solid formed by adjacent cubes -/
def volume_of_adjacent_cubes (n : ℕ) (side_length : ℝ) : ℝ :=
  n * (side_length ^ 3)

/-- Theorem: The volume of a solid formed by five adjacent cubes with side length 5 cm is 625 cm³ -/
theorem volume_of_five_cubes : volume_of_adjacent_cubes 5 5 = 625 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_five_cubes_l153_15313


namespace NUMINAMATH_CALUDE_distance_AD_between_41_and_42_l153_15304

-- Define points A, B, C, and D in a 2D plane
variable (A B C D : ℝ × ℝ)

-- Define the conditions
variable (h1 : B.1 > A.1 ∧ B.2 = A.2) -- B is due east of A
variable (h2 : C.1 = B.1 ∧ C.2 > B.2) -- C is due north of B
variable (h3 : (C.1 - A.1)^2 + (C.2 - A.2)^2 = 300) -- AC = 10√3
variable (h4 : Real.cos (Real.arctan ((C.2 - A.2) / (C.1 - A.1))) = 1/2) -- Angle BAC = 60°
variable (h5 : D.1 = C.1 ∧ D.2 = C.2 + 30) -- D is 30 meters due north of C

-- Theorem statement
theorem distance_AD_between_41_and_42 :
  41 < Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) ∧
  Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) < 42 :=
sorry

end NUMINAMATH_CALUDE_distance_AD_between_41_and_42_l153_15304


namespace NUMINAMATH_CALUDE_mikes_games_last_year_l153_15382

/-- The number of basketball games Mike went to this year -/
def games_this_year : ℕ := 15

/-- The number of basketball games Mike missed this year -/
def games_missed : ℕ := 41

/-- The total number of basketball games Mike went to -/
def total_games : ℕ := 54

/-- The number of basketball games Mike went to last year -/
def games_last_year : ℕ := total_games - games_this_year

theorem mikes_games_last_year : games_last_year = 39 := by
  sorry

end NUMINAMATH_CALUDE_mikes_games_last_year_l153_15382


namespace NUMINAMATH_CALUDE_squirrel_rainy_days_l153_15345

theorem squirrel_rainy_days 
  (sunny_nuts : ℕ) 
  (rainy_nuts : ℕ) 
  (total_nuts : ℕ) 
  (average_nuts : ℕ) 
  (h1 : sunny_nuts = 20)
  (h2 : rainy_nuts = 12)
  (h3 : total_nuts = 112)
  (h4 : average_nuts = 14)
  : ∃ (rainy_days : ℕ), rainy_days = 6 ∧ 
    ∃ (total_days : ℕ), 
      total_days * average_nuts = total_nuts ∧
      rainy_days * rainy_nuts + (total_days - rainy_days) * sunny_nuts = total_nuts :=
by sorry

end NUMINAMATH_CALUDE_squirrel_rainy_days_l153_15345


namespace NUMINAMATH_CALUDE_probability_white_ball_specific_l153_15366

/-- The probability of drawing a white ball from a bag -/
def probability_white_ball (black white red : ℕ) : ℚ :=
  white / (black + white + red)

/-- Theorem: The probability of drawing a white ball from a bag with 3 black, 2 white, and 1 red ball is 1/3 -/
theorem probability_white_ball_specific : probability_white_ball 3 2 1 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_white_ball_specific_l153_15366


namespace NUMINAMATH_CALUDE_inequalities_theorem_l153_15351

variables (a b c x y z : ℝ)

def M : ℝ := a * x + b * y + c * z
def N : ℝ := a * z + b * y + c * x
def P : ℝ := a * y + b * z + c * x
def Q : ℝ := a * z + b * x + c * y

theorem inequalities_theorem (h1 : a > b) (h2 : b > c) (h3 : x > y) (h4 : y > z) :
  M a b c x y z > P a b c x y z ∧ 
  P a b c x y z > N a b c x y z ∧ 
  M a b c x y z > Q a b c x y z ∧ 
  Q a b c x y z > N a b c x y z :=
by sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l153_15351


namespace NUMINAMATH_CALUDE_roots_of_equation_l153_15339

theorem roots_of_equation : 
  let f : ℝ → ℝ := λ x => (x^2 - 5*x + 6)*(x - 3)*(x + 2)
  {x : ℝ | f x = 0} = {2, 3, -2} := by
sorry

end NUMINAMATH_CALUDE_roots_of_equation_l153_15339
