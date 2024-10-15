import Mathlib

namespace NUMINAMATH_CALUDE_smallest_number_of_students_l1433_143321

theorem smallest_number_of_students (n : ℕ) : 
  n > 0 ∧
  (n : ℚ) * (75 : ℚ) / 100 = ↑(n - (n / 4 : ℕ)) ∧
  (n / 40 : ℕ) = (n / 4 : ℕ) * 10 / 100 ∧
  (33 * n / 200 : ℕ) = ((11 * n / 100 : ℕ) * 3 / 2 : ℕ) ∧
  ∀ m : ℕ, m > 0 ∧ 
    (m : ℚ) * (75 : ℚ) / 100 = ↑(m - (m / 4 : ℕ)) ∧
    (m / 40 : ℕ) = (m / 4 : ℕ) * 10 / 100 ∧
    (33 * m / 200 : ℕ) = ((11 * m / 100 : ℕ) * 3 / 2 : ℕ) →
    m ≥ n →
  n = 200 := by sorry

end NUMINAMATH_CALUDE_smallest_number_of_students_l1433_143321


namespace NUMINAMATH_CALUDE_father_son_age_ratio_l1433_143397

def father_age : ℕ := 40
def son_age : ℕ := 10

theorem father_son_age_ratio :
  (father_age : ℚ) / son_age = 4 ∧
  father_age + 20 = 2 * (son_age + 20) :=
by sorry

end NUMINAMATH_CALUDE_father_son_age_ratio_l1433_143397


namespace NUMINAMATH_CALUDE_a_n_formula_l1433_143370

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem a_n_formula (a : ℕ → ℝ) (h1 : arithmetic_sequence (λ n => a (n + 1) - a n))
  (h2 : a 1 - a 0 = 1) (h3 : ∀ n : ℕ, (a (n + 2) - a (n + 1)) - (a (n + 1) - a n) = 2) :
  ∀ n : ℕ, a n = 2^n - 1 :=
sorry

end NUMINAMATH_CALUDE_a_n_formula_l1433_143370


namespace NUMINAMATH_CALUDE_toy_distribution_ratio_l1433_143329

theorem toy_distribution_ratio (total_toys : ℕ) (num_friends : ℕ) 
  (h1 : total_toys = 118) (h2 : num_friends = 4) :
  ∃ (toys_per_friend : ℕ), 
    toys_per_friend * num_friends ≤ total_toys ∧
    toys_per_friend * num_friends > total_toys - num_friends ∧
    (toys_per_friend : ℚ) / total_toys = 1 / 4 := by
  sorry

#check toy_distribution_ratio

end NUMINAMATH_CALUDE_toy_distribution_ratio_l1433_143329


namespace NUMINAMATH_CALUDE_probability_male_saturday_female_sunday_l1433_143326

/-- The number of male students -/
def num_male : ℕ := 2

/-- The number of female students -/
def num_female : ℕ := 2

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- The number of days in the event -/
def num_days : ℕ := 2

/-- The probability of selecting a male student for Saturday and a female student for Sunday -/
theorem probability_male_saturday_female_sunday :
  (num_male * num_female) / (total_students * (total_students - 1)) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_male_saturday_female_sunday_l1433_143326


namespace NUMINAMATH_CALUDE_cos_pi_minus_theta_l1433_143310

theorem cos_pi_minus_theta (θ : Real) :
  (∃ (x y : Real), x = 4 ∧ y = -3 ∧ x = Real.cos θ * Real.sqrt (x^2 + y^2) ∧ y = Real.sin θ * Real.sqrt (x^2 + y^2)) →
  Real.cos (π - θ) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_minus_theta_l1433_143310


namespace NUMINAMATH_CALUDE_sequence_difference_l1433_143376

theorem sequence_difference (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = n^2) : 
  a 3 - a 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_difference_l1433_143376


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l1433_143380

/-- Given that the line x - y - 1 = 0 is tangent to the parabola y = ax², prove that a = 1/4 -/
theorem tangent_line_to_parabola (a : ℝ) : 
  (∃ x y : ℝ, x - y - 1 = 0 ∧ y = a * x^2 ∧ 
   ∀ x' y' : ℝ, y' = a * x'^2 → (x - x') * (2 * a * x) = y - y') → 
  a = 1/4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l1433_143380


namespace NUMINAMATH_CALUDE_milo_cash_reward_l1433_143342

def grades : List ℕ := [2, 2, 2, 3, 3, 3, 3, 4, 5]

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

def cashReward (avg : ℚ) : ℚ := 5 * avg

theorem milo_cash_reward :
  cashReward (average grades) = 15 := by
  sorry

end NUMINAMATH_CALUDE_milo_cash_reward_l1433_143342


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_2_min_value_f_range_of_a_l1433_143387

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 4| + |x - 2|

-- Theorem 1: Solution set of f(x) > 2
theorem solution_set_f_greater_than_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x < 2 ∨ x > 4} := by sorry

-- Theorem 2: Minimum value of f(x)
theorem min_value_f :
  ∃ M : ℝ, M = 2 ∧ ∀ x : ℝ, f x ≥ M := by sorry

-- Theorem 3: Range of a
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → 2^x + a ≥ 2) → a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_2_min_value_f_range_of_a_l1433_143387


namespace NUMINAMATH_CALUDE_car_speed_problem_l1433_143367

theorem car_speed_problem (speed_second_hour : ℝ) (average_speed : ℝ) :
  speed_second_hour = 42 ∧ average_speed = 66 →
  ∃ speed_first_hour : ℝ,
    speed_first_hour = 90 ∧
    average_speed = (speed_first_hour + speed_second_hour) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1433_143367


namespace NUMINAMATH_CALUDE_ring_cost_l1433_143360

theorem ring_cost (total_revenue : ℕ) (necklace_count : ℕ) (ring_count : ℕ) (necklace_price : ℕ) :
  total_revenue = 80 →
  necklace_count = 4 →
  ring_count = 8 →
  necklace_price = 12 →
  ∃ (ring_price : ℕ), ring_price = 4 ∧ total_revenue = necklace_count * necklace_price + ring_count * ring_price :=
by
  sorry

end NUMINAMATH_CALUDE_ring_cost_l1433_143360


namespace NUMINAMATH_CALUDE_second_row_equals_first_row_l1433_143395

/-- Represents a 3 × n grid with the properties described in the problem -/
structure Grid (n : ℕ) where
  first_row : Fin n → ℝ
  second_row : Fin n → ℝ
  third_row : Fin n → ℝ
  first_row_increasing : ∀ i j, i < j → first_row i < first_row j
  second_row_permutation : ∀ x, ∃ i, second_row i = x ↔ ∃ j, first_row j = x
  third_row_sum : ∀ i, third_row i = first_row i + second_row i
  third_row_increasing : ∀ i j, i < j → third_row i < third_row j

/-- The main theorem stating that the second row must be identical to the first row -/
theorem second_row_equals_first_row {n : ℕ} (grid : Grid n) :
  ∀ i, grid.second_row i = grid.first_row i :=
sorry

end NUMINAMATH_CALUDE_second_row_equals_first_row_l1433_143395


namespace NUMINAMATH_CALUDE_intersection_M_N_l1433_143352

def M : Set ℝ := {-1, 1, 2, 3, 4}
def N : Set ℝ := {x : ℝ | x^2 + 2*x > 3}

theorem intersection_M_N : M ∩ N = {2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1433_143352


namespace NUMINAMATH_CALUDE_set_union_condition_l1433_143323

theorem set_union_condition (m : ℝ) : 
  let A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 7}
  let B : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}
  A ∪ B = A → m ≤ -2 ∨ (-1 ≤ m ∧ m ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_set_union_condition_l1433_143323


namespace NUMINAMATH_CALUDE_min_value_expression_l1433_143366

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : x^2 + y^2 = z) :
  ∃ (min : ℝ), min = -2040200 ∧
  ∀ (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + b^2 = z),
    (a + 1/b) * (a + 1/b - 2020) + (b + 1/a) * (b + 1/a - 2020) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1433_143366


namespace NUMINAMATH_CALUDE_tank_water_level_l1433_143364

theorem tank_water_level (tank_capacity : ℚ) (initial_fraction : ℚ) (added_water : ℚ) : 
  tank_capacity = 40 →
  initial_fraction = 3/4 →
  added_water = 5 →
  (initial_fraction * tank_capacity + added_water) / tank_capacity = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_tank_water_level_l1433_143364


namespace NUMINAMATH_CALUDE_paper_sheet_width_l1433_143378

theorem paper_sheet_width (sheet1_length sheet1_width sheet2_length : ℝ)
  (h1 : sheet1_length = 11)
  (h2 : sheet1_width = 17)
  (h3 : sheet2_length = 11)
  (h4 : 2 * sheet1_length * sheet1_width = 2 * sheet2_length * sheet2_width + 100) :
  sheet2_width = 12.45 := by
  sorry

end NUMINAMATH_CALUDE_paper_sheet_width_l1433_143378


namespace NUMINAMATH_CALUDE_father_age_triple_weiwei_age_l1433_143305

/-- Weiwei's current age in years -/
def weiwei_age : ℕ := 8

/-- Weiwei's father's current age in years -/
def father_age : ℕ := 34

/-- The number of years after which the father's age will be three times Weiwei's age -/
def years_until_triple : ℕ := 5

theorem father_age_triple_weiwei_age :
  father_age + years_until_triple = 3 * (weiwei_age + years_until_triple) :=
sorry

end NUMINAMATH_CALUDE_father_age_triple_weiwei_age_l1433_143305


namespace NUMINAMATH_CALUDE_min_value_abc_l1433_143358

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/b + 1/c = 9) :
  a^2 * b^3 * c^4 ≥ 1/1728 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
    1/a₀ + 1/b₀ + 1/c₀ = 9 ∧ a₀^2 * b₀^3 * c₀^4 = 1/1728 := by
  sorry

end NUMINAMATH_CALUDE_min_value_abc_l1433_143358


namespace NUMINAMATH_CALUDE_stamp_problem_l1433_143304

/-- Represents the number of ways to make a certain amount with given coin denominations -/
def numWays (amount : ℕ) (coins : List ℕ) : ℕ := sorry

/-- Represents the minimum number of coins needed to make a certain amount with given coin denominations -/
def minCoins (amount : ℕ) (coins : List ℕ) : ℕ := sorry

theorem stamp_problem :
  minCoins 74 [5, 7] = 12 := by sorry

end NUMINAMATH_CALUDE_stamp_problem_l1433_143304


namespace NUMINAMATH_CALUDE_abs_x_minus_3_eq_1_implies_5_minus_2x_eq_neg_3_or_1_l1433_143385

theorem abs_x_minus_3_eq_1_implies_5_minus_2x_eq_neg_3_or_1 (x : ℝ) :
  |x - 3| = 1 → (5 - 2*x = -3 ∨ 5 - 2*x = 1) :=
by sorry

end NUMINAMATH_CALUDE_abs_x_minus_3_eq_1_implies_5_minus_2x_eq_neg_3_or_1_l1433_143385


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1433_143334

theorem quadratic_equation_solution :
  ∀ x : ℝ, x^2 + 6*x + 9 = 0 ↔ x = -3 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1433_143334


namespace NUMINAMATH_CALUDE_matrix_sum_equality_l1433_143354

def matrix1 : Matrix (Fin 3) (Fin 3) ℤ := !![2, -1, 3; 0, 4, -2; 5, -3, 1]
def matrix2 : Matrix (Fin 3) (Fin 3) ℤ := !![-3, 2, -4; 1, -6, 3; -2, 4, 0]
def result : Matrix (Fin 3) (Fin 3) ℤ := !![-1, 1, -1; 1, -2, 1; 3, 1, 1]

theorem matrix_sum_equality : matrix1 + matrix2 = result := by
  sorry

end NUMINAMATH_CALUDE_matrix_sum_equality_l1433_143354


namespace NUMINAMATH_CALUDE_product_of_four_consecutive_integers_l1433_143333

theorem product_of_four_consecutive_integers (n : ℤ) :
  ∃ M : ℤ, 
    Even M ∧ 
    (n - 1) * n * (n + 1) * (n + 2) = (M - 2) * M := by
  sorry

end NUMINAMATH_CALUDE_product_of_four_consecutive_integers_l1433_143333


namespace NUMINAMATH_CALUDE_rabbit_nuts_count_l1433_143327

theorem rabbit_nuts_count :
  ∀ (rabbit_holes fox_holes : ℕ),
    rabbit_holes = fox_holes + 5 →
    4 * rabbit_holes = 6 * fox_holes →
    4 * rabbit_holes = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_rabbit_nuts_count_l1433_143327


namespace NUMINAMATH_CALUDE_function_inequality_l1433_143359

open Real

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x ∈ (Set.Ioo 0 (π / 2)), HasDerivAt f (f' x) x) →
  (∀ x ∈ (Set.Ioo 0 (π / 2)), f x * tan x + f' x < 0) →
  Real.sqrt 3 * f (π / 3) < f (π / 6) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_l1433_143359


namespace NUMINAMATH_CALUDE_fraction_order_l1433_143368

theorem fraction_order : 
  let f1 := 16/12
  let f2 := 21/14
  let f3 := 18/13
  let f4 := 20/15
  f1 < f3 ∧ f3 < f2 ∧ f2 < f4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l1433_143368


namespace NUMINAMATH_CALUDE_product_equals_693_over_256_l1433_143388

theorem product_equals_693_over_256 : 
  (1 + 1/2) * (1 + 1/4) * (1 + 1/6) * (1 + 1/8) * (1 + 1/10) = 693/256 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_693_over_256_l1433_143388


namespace NUMINAMATH_CALUDE_no_two_digit_primes_with_digit_sum_nine_l1433_143324

def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sumOfDigits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem no_two_digit_primes_with_digit_sum_nine :
  ∀ n : ℕ, isTwoDigit n → sumOfDigits n = 9 → ¬ Nat.Prime n := by
sorry

end NUMINAMATH_CALUDE_no_two_digit_primes_with_digit_sum_nine_l1433_143324


namespace NUMINAMATH_CALUDE_change_in_f_l1433_143377

def f (x : ℝ) : ℝ := x^2 - 5*x

theorem change_in_f (x : ℝ) :
  (f (x + 2) - f x = 4*x - 6) ∧
  (f (x - 2) - f x = -4*x + 14) :=
by sorry

end NUMINAMATH_CALUDE_change_in_f_l1433_143377


namespace NUMINAMATH_CALUDE_no_solution_system_l1433_143372

theorem no_solution_system : ¬∃ (x y : ℝ), 
  (x^3 + x + y + 1 = 0) ∧ 
  (y*x^2 + x + y = 0) ∧ 
  (y^2 + y - x^2 + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_system_l1433_143372


namespace NUMINAMATH_CALUDE_quadratic_solution_l1433_143307

theorem quadratic_solution (m : ℝ) : 
  (2 : ℝ)^2 - m * 2 + 8 = 0 → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l1433_143307


namespace NUMINAMATH_CALUDE_probability_multiple_2_or_3_30_l1433_143381

def is_multiple_of_2_or_3 (n : ℕ) : Bool :=
  n % 2 = 0 || n % 3 = 0

def count_multiples (n : ℕ) : ℕ :=
  (List.range n).filter is_multiple_of_2_or_3 |>.length

theorem probability_multiple_2_or_3_30 :
  (count_multiples 30 : ℚ) / 30 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_multiple_2_or_3_30_l1433_143381


namespace NUMINAMATH_CALUDE_wand_price_theorem_l1433_143357

theorem wand_price_theorem (purchase_price : ℝ) (original_price : ℝ) : 
  purchase_price = 8 →
  purchase_price = (1/8) * original_price →
  original_price = 64 := by
sorry

end NUMINAMATH_CALUDE_wand_price_theorem_l1433_143357


namespace NUMINAMATH_CALUDE_largest_gold_coin_distribution_l1433_143336

theorem largest_gold_coin_distribution (n : ℕ) (h1 : n < 150) 
  (h2 : ∃ k : ℕ, n = 15 * k + 3) : n ≤ 138 := by
  sorry

end NUMINAMATH_CALUDE_largest_gold_coin_distribution_l1433_143336


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l1433_143383

theorem roots_sum_of_squares (p q : ℝ) : 
  (p^2 - 5*p + 6 = 0) → (q^2 - 5*q + 6 = 0) → p^2 + q^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l1433_143383


namespace NUMINAMATH_CALUDE_probability_standard_weight_l1433_143317

def total_students : ℕ := 500
def standard_weight_students : ℕ := 350

theorem probability_standard_weight :
  (standard_weight_students : ℚ) / total_students = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_standard_weight_l1433_143317


namespace NUMINAMATH_CALUDE_quadratic_one_root_l1433_143306

theorem quadratic_one_root (c b : ℝ) (hc : c > 0) :
  (∃! x : ℝ, x^2 + 2 * Real.sqrt c * x + b = 0) → c = b := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l1433_143306


namespace NUMINAMATH_CALUDE_slipper_cost_l1433_143347

/-- Calculates the total cost of a pair of embroidered slippers with shipping --/
theorem slipper_cost (original_price discount_percentage embroidery_cost_per_shoe shipping_cost : ℚ) :
  original_price = 50 →
  discount_percentage = 10 →
  embroidery_cost_per_shoe = (11/2) →
  shipping_cost = 10 →
  original_price * (1 - discount_percentage / 100) + 2 * embroidery_cost_per_shoe + shipping_cost = 66 := by
sorry


end NUMINAMATH_CALUDE_slipper_cost_l1433_143347


namespace NUMINAMATH_CALUDE_tangent_slope_at_2_6_l1433_143320

-- Define the function f(x) = x³ - 2x + 2
def f (x : ℝ) : ℝ := x^3 - 2*x + 2

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_slope_at_2_6 :
  f 2 = 6 ∧ f' 2 = 10 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_2_6_l1433_143320


namespace NUMINAMATH_CALUDE_min_weights_theorem_l1433_143300

/-- A function that calculates the sum of powers of 2 up to 2^n -/
def sumPowersOf2 (n : ℕ) : ℕ := 2^(n+1) - 1

/-- The maximum weight we need to measure -/
def maxWeight : ℕ := 100

/-- The proposition that n weights are sufficient to measure all weights up to maxWeight -/
def isSufficient (n : ℕ) : Prop := sumPowersOf2 n ≥ maxWeight

/-- The proposition that n weights are necessary to measure all weights up to maxWeight -/
def isNecessary (n : ℕ) : Prop := ∀ m : ℕ, m < n → sumPowersOf2 m < maxWeight

/-- The theorem stating that 7 is the minimum number of weights needed -/
theorem min_weights_theorem : 
  (isSufficient 7 ∧ isNecessary 7) ∧ ∀ n : ℕ, n < 7 → ¬(isSufficient n ∧ isNecessary n) :=
sorry

end NUMINAMATH_CALUDE_min_weights_theorem_l1433_143300


namespace NUMINAMATH_CALUDE_problem_polygon_area_l1433_143361

/-- Polygon PQRSTU with given side lengths and properties -/
structure Polygon where
  PQ : ℝ
  QR : ℝ
  RS : ℝ
  ST : ℝ
  TU : ℝ
  PT_parallel_QR : Bool
  PU_divides : Bool

/-- Calculate the area of the polygon PQRSTU -/
def polygon_area (p : Polygon) : ℝ :=
  sorry

/-- The specific polygon from the problem -/
def problem_polygon : Polygon := {
  PQ := 4
  QR := 7
  RS := 5
  ST := 6
  TU := 3
  PT_parallel_QR := true
  PU_divides := true
}

/-- Theorem stating that the area of the problem polygon is 41.5 square units -/
theorem problem_polygon_area :
  polygon_area problem_polygon = 41.5 := by sorry

end NUMINAMATH_CALUDE_problem_polygon_area_l1433_143361


namespace NUMINAMATH_CALUDE_magician_marbles_left_l1433_143389

/-- Calculates the total number of marbles left after removing some from each color --/
def marblesLeft (initialRed initialBlue initialGreen redTaken : ℕ) : ℕ :=
  let blueTaken := 5 * redTaken
  let greenTaken := blueTaken / 2
  let redLeft := initialRed - redTaken
  let blueLeft := initialBlue - blueTaken
  let greenLeft := initialGreen - greenTaken
  redLeft + blueLeft + greenLeft

/-- Theorem stating that given the initial numbers of marbles and the rules for taking away marbles,
    the total number of marbles left is 93 --/
theorem magician_marbles_left :
  marblesLeft 40 60 35 5 = 93 := by
  sorry

end NUMINAMATH_CALUDE_magician_marbles_left_l1433_143389


namespace NUMINAMATH_CALUDE_convex_pentagon_probability_l1433_143396

/-- The number of points on the circle -/
def num_points : ℕ := 8

/-- The number of chords to be selected -/
def num_selected_chords : ℕ := 5

/-- The total number of possible chords between num_points points -/
def total_chords (n : ℕ) : ℕ := n.choose 2

/-- The number of ways to select num_selected_chords from total_chords -/
def ways_to_select_chords (n : ℕ) : ℕ := (total_chords n).choose num_selected_chords

/-- The number of ways to choose 5 points from num_points points -/
def convex_pentagons (n : ℕ) : ℕ := n.choose 5

/-- The probability of forming a convex pentagon -/
def probability : ℚ := (convex_pentagons num_points : ℚ) / (ways_to_select_chords num_points : ℚ)

theorem convex_pentagon_probability :
  probability = 1 / 1755 :=
sorry

end NUMINAMATH_CALUDE_convex_pentagon_probability_l1433_143396


namespace NUMINAMATH_CALUDE_skip_speed_relation_l1433_143315

theorem skip_speed_relation (bruce_speed : ℝ) : 
  let tony_speed := 2 * bruce_speed
  let brandon_speed := (1/3) * tony_speed
  let colin_speed := 6 * brandon_speed
  colin_speed = 4 → bruce_speed = 1 := by
  sorry

end NUMINAMATH_CALUDE_skip_speed_relation_l1433_143315


namespace NUMINAMATH_CALUDE_parallel_transitive_l1433_143325

-- Define a type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define parallel relation
def parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem parallel_transitive (l1 l2 l3 : Line) :
  parallel l1 l2 → parallel l2 l3 → parallel l1 l3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_transitive_l1433_143325


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1433_143348

/-- Given that P(5, 9) is the midpoint of segment CD and C has coordinates (11, 5),
    prove that the sum of the coordinates of D is 12. -/
theorem midpoint_coordinate_sum (C D : ℝ × ℝ) :
  C = (11, 5) →
  (5, 9) = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  D.1 + D.2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l1433_143348


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_equals_three_l1433_143398

theorem mean_equality_implies_y_equals_three :
  let mean1 := (3 + 7 + 11 + 15) / 4
  let mean2 := (10 + 14 + y) / 3
  mean1 = mean2 → y = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_equals_three_l1433_143398


namespace NUMINAMATH_CALUDE_intersection_distance_l1433_143331

/-- The distance between the intersection points of y = x - 3 and x² + 2y² = 8 is 4√3/3 -/
theorem intersection_distance : 
  ∃ (A B : ℝ × ℝ), 
    (A.2 = A.1 - 3 ∧ A.1^2 + 2*A.2^2 = 8) ∧ 
    (B.2 = B.1 - 3 ∧ B.1^2 + 2*B.2^2 = 8) ∧ 
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = (4 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l1433_143331


namespace NUMINAMATH_CALUDE_disk_covering_radius_bound_l1433_143351

theorem disk_covering_radius_bound (R : ℝ) (r : ℝ) :
  R = 1 →
  (∃ (centers : Fin 7 → ℝ × ℝ),
    (∀ x y : ℝ × ℝ, (x.1 - y.1)^2 + (x.2 - y.2)^2 ≤ R^2 →
      ∃ i : Fin 7, (x.1 - (centers i).1)^2 + (x.2 - (centers i).2)^2 ≤ r^2)) →
  r ≥ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_disk_covering_radius_bound_l1433_143351


namespace NUMINAMATH_CALUDE_age_difference_l1433_143346

/-- Given four persons a, b, c, and d with ages A, B, C, and D respectively,
    where the total age of a and b is 11 years more than the total age of b and c,
    prove that c is 11 + D years younger than the sum of the ages of a and d. -/
theorem age_difference (A B C D : ℤ) (h : A + B = B + C + 11) :
  C - (A + D) = -11 - D := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1433_143346


namespace NUMINAMATH_CALUDE_complex_arithmetic_proof_l1433_143318

theorem complex_arithmetic_proof :
  let z₁ : ℂ := 5 + 6*I
  let z₂ : ℂ := -1 + 4*I
  let z₃ : ℂ := 3 - 2*I
  (z₁ + z₂) - z₃ = 1 + 12*I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_proof_l1433_143318


namespace NUMINAMATH_CALUDE_cement_calculation_l1433_143313

/-- The amount of cement originally owned -/
def original_cement : ℕ := sorry

/-- The amount of cement bought -/
def bought_cement : ℕ := 215

/-- The amount of cement brought by the son -/
def son_brought_cement : ℕ := 137

/-- The current total amount of cement -/
def current_cement : ℕ := 450

/-- Theorem stating the relationship between the amounts of cement -/
theorem cement_calculation : 
  original_cement = current_cement - (bought_cement + son_brought_cement) :=
by sorry

end NUMINAMATH_CALUDE_cement_calculation_l1433_143313


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1433_143311

/-- Given a geometric sequence {aₙ} where a₁ + a₂ = 40 and a₃ + a₄ = 60, prove that a₇ + a₈ = 135 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- {aₙ} is a geometric sequence
  a 1 + a 2 = 40 →                           -- a₁ + a₂ = 40
  a 3 + a 4 = 60 →                           -- a₃ + a₄ = 60
  a 7 + a 8 = 135 :=                         -- a₇ + a₈ = 135
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1433_143311


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_from_tangent_circle_l1433_143344

/-- Given a circle and a hyperbola, if the circle is tangent to the asymptotes of the hyperbola,
    then the eccentricity of the hyperbola is 5/2. -/
theorem hyperbola_eccentricity_from_tangent_circle
  (a b : ℝ) (h_positive : a > 0 ∧ b > 0) :
  let circle := fun (x y : ℝ) => x^2 + y^2 - 10*y + 21 = 0
  let hyperbola := fun (x y : ℝ) => x^2/a^2 - y^2/b^2 = 1
  let asymptote := fun (x y : ℝ) => b*x - a*y = 0 ∨ b*x + a*y = 0
  let is_tangent := ∃ (x y : ℝ), circle x y ∧ asymptote x y
  let eccentricity := Real.sqrt (1 + b^2/a^2)
  is_tangent → eccentricity = 5/2 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_from_tangent_circle_l1433_143344


namespace NUMINAMATH_CALUDE_reward_system_l1433_143332

/-- The number of bowls a customer needs to buy to get rewarded with two bowls -/
def bowls_for_reward : ℕ := sorry

theorem reward_system (total_bowls : ℕ) (customers : ℕ) (buying_customers : ℕ) 
  (bowls_per_customer : ℕ) (remaining_bowls : ℕ) :
  total_bowls = 70 →
  customers = 20 →
  buying_customers = customers / 2 →
  bowls_per_customer = 20 →
  remaining_bowls = 30 →
  bowls_for_reward = 10 := by sorry

end NUMINAMATH_CALUDE_reward_system_l1433_143332


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l1433_143328

theorem max_value_cos_sin (x : ℝ) : 3 * Real.cos x - Real.sin x ≤ Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l1433_143328


namespace NUMINAMATH_CALUDE_binomial_prob_one_third_l1433_143309

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_prob_one_third 
  (X : BinomialRV) 
  (h_expect : expectation X = 30)
  (h_var : variance X = 20) : 
  X.p = 1/3 := by
sorry

end NUMINAMATH_CALUDE_binomial_prob_one_third_l1433_143309


namespace NUMINAMATH_CALUDE_max_pieces_cut_l1433_143373

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- The plywood sheet -/
def plywood : Rectangle := { length := 22, width := 15 }

/-- The piece to be cut -/
def piece : Rectangle := { length := 3, width := 5 }

/-- Theorem stating the maximum number of pieces that can be cut -/
theorem max_pieces_cut : 
  (area plywood) / (area piece) = 22 := by sorry

end NUMINAMATH_CALUDE_max_pieces_cut_l1433_143373


namespace NUMINAMATH_CALUDE_function_minimum_implies_inequality_l1433_143374

open Real

/-- Given a function f(x) = -3ln(x) + ax² + bx, where a > 0 and b is real,
    if for any x > 0, f(x) ≥ f(3), then ln(a) < -b - 1 -/
theorem function_minimum_implies_inequality (a b : ℝ) (ha : a > 0) :
  (∀ x > 0, -3 * log x + a * x^2 + b * x ≥ -3 * log 3 + 9 * a + 3 * b) →
  log a < -b - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_minimum_implies_inequality_l1433_143374


namespace NUMINAMATH_CALUDE_soup_cans_feeding_l1433_143301

/-- Proves that given 8 total cans of soup, where each can feeds either 4 adults or 6 children,
    and 18 children have been fed, the number of adults that can be fed with the remaining soup is 20. -/
theorem soup_cans_feeding (total_cans : ℕ) (adults_per_can children_per_can : ℕ) (children_fed : ℕ) :
  total_cans = 8 →
  adults_per_can = 4 →
  children_per_can = 6 →
  children_fed = 18 →
  (total_cans - (children_fed / children_per_can)) * adults_per_can = 20 :=
by sorry

end NUMINAMATH_CALUDE_soup_cans_feeding_l1433_143301


namespace NUMINAMATH_CALUDE_two_tvs_one_mixer_cost_l1433_143312

/-- The cost of a mixer in rupees -/
def mixer_cost : ℕ := 1400

/-- The cost of a TV in rupees -/
def tv_cost : ℕ := 4200

/-- The cost of two mixers and one TV in rupees -/
def two_mixers_one_tv_cost : ℕ := 7000

theorem two_tvs_one_mixer_cost : 2 * tv_cost + mixer_cost = 9800 := by
  sorry

end NUMINAMATH_CALUDE_two_tvs_one_mixer_cost_l1433_143312


namespace NUMINAMATH_CALUDE_common_divisors_45_48_l1433_143356

theorem common_divisors_45_48 : Finset.card (Finset.filter (fun d => d ∣ 48) (Nat.divisors 45)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_45_48_l1433_143356


namespace NUMINAMATH_CALUDE_ratio_equality_l1433_143394

theorem ratio_equality (x y : ℝ) (h1 : 2 * x = 3 * y) (h2 : y ≠ 0) : x / y = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1433_143394


namespace NUMINAMATH_CALUDE_max_planes_of_symmetry_l1433_143302

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  planes_of_symmetry : ℕ

/-- Two convex polyhedra do not intersect -/
def do_not_intersect (A B : ConvexPolyhedron) : Prop :=
  sorry

/-- The number of planes of symmetry for a figure consisting of two polyhedra -/
def combined_planes_of_symmetry (A B : ConvexPolyhedron) : ℕ :=
  sorry

theorem max_planes_of_symmetry (A B : ConvexPolyhedron) 
  (h1 : do_not_intersect A B)
  (h2 : A.planes_of_symmetry = 2012)
  (h3 : B.planes_of_symmetry = 2013) :
  combined_planes_of_symmetry A B = 2013 :=
sorry

end NUMINAMATH_CALUDE_max_planes_of_symmetry_l1433_143302


namespace NUMINAMATH_CALUDE_smallest_number_inequality_l1433_143340

theorem smallest_number_inequality (x y z m : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) 
  (hm : m = min (min (min 1 (x^9)) (y^9)) (z^7)) : 
  m ≤ x * y^2 * z^3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_inequality_l1433_143340


namespace NUMINAMATH_CALUDE_angie_pretzels_l1433_143349

theorem angie_pretzels (barry_pretzels : ℕ) (shelly_pretzels : ℕ) (angie_pretzels : ℕ) :
  barry_pretzels = 12 →
  shelly_pretzels = barry_pretzels / 2 →
  angie_pretzels = 3 * shelly_pretzels →
  angie_pretzels = 18 := by
sorry

end NUMINAMATH_CALUDE_angie_pretzels_l1433_143349


namespace NUMINAMATH_CALUDE_expression_evaluation_l1433_143314

theorem expression_evaluation : 2 + 5 * 3^2 - 4 * 2 + 7 * 3 / 3 = 46 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1433_143314


namespace NUMINAMATH_CALUDE_three_digit_number_divisible_by_seven_l1433_143330

theorem three_digit_number_divisible_by_seven :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
  n % 10 = 4 ∧ 
  (n / 100) % 10 = 5 ∧ 
  n % 7 = 0 ∧
  n = 534 := by
sorry

end NUMINAMATH_CALUDE_three_digit_number_divisible_by_seven_l1433_143330


namespace NUMINAMATH_CALUDE_triangle_square_count_l1433_143365

/-- Represents a geometric figure with three layers -/
structure ThreeLayerFigure where
  first_layer_triangles : Nat
  second_layer_squares : Nat
  third_layer_triangle : Nat

/-- Counts the total number of triangles in the figure -/
def count_triangles (figure : ThreeLayerFigure) : Nat :=
  figure.first_layer_triangles + figure.third_layer_triangle

/-- Counts the total number of squares in the figure -/
def count_squares (figure : ThreeLayerFigure) : Nat :=
  figure.second_layer_squares

/-- The specific figure described in the problem -/
def problem_figure : ThreeLayerFigure :=
  { first_layer_triangles := 3
  , second_layer_squares := 2
  , third_layer_triangle := 1 }

theorem triangle_square_count :
  count_triangles problem_figure = 4 ∧ count_squares problem_figure = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_square_count_l1433_143365


namespace NUMINAMATH_CALUDE_inverse_proportion_solution_l1433_143335

/-- Two real numbers are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_solution (x y : ℝ) :
  InverselyProportional x y →
  x + y = 30 →
  x - y = 10 →
  (∃ y' : ℝ, InverselyProportional 4 y' ∧ y' = 50) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_solution_l1433_143335


namespace NUMINAMATH_CALUDE_intersection_sum_l1433_143375

-- Define the parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 2)^2
def parabola2 (x y : ℝ) : Prop := x + 6 = (y + 1)^2

-- Define the intersection points
def intersection_points (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) : Prop :=
  parabola1 x₁ y₁ ∧ parabola2 x₁ y₁ ∧
  parabola1 x₂ y₂ ∧ parabola2 x₂ y₂ ∧
  parabola1 x₃ y₃ ∧ parabola2 x₃ y₃ ∧
  parabola1 x₄ y₄ ∧ parabola2 x₄ y₄

-- Theorem statement
theorem intersection_sum (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) :
  intersection_points x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ →
  x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l1433_143375


namespace NUMINAMATH_CALUDE_largest_fraction_l1433_143343

theorem largest_fraction : 
  let fractions := [2/5, 3/7, 4/9, 5/11, 6/13]
  ∀ x ∈ fractions, (6/13 : ℚ) ≥ x :=
by sorry

end NUMINAMATH_CALUDE_largest_fraction_l1433_143343


namespace NUMINAMATH_CALUDE_circle_min_radius_l1433_143392

theorem circle_min_radius (a : ℝ) : 
  let r := Real.sqrt ((5 * a^2) / 4 + 2)
  r ≥ Real.sqrt 2 ∧ ∃ a₀, Real.sqrt ((5 * a₀^2) / 4 + 2) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_min_radius_l1433_143392


namespace NUMINAMATH_CALUDE_magnitude_v_l1433_143355

theorem magnitude_v (u v : ℂ) (h1 : u * v = 16 - 30 * I) (h2 : Complex.abs u = 2) : 
  Complex.abs v = 17 := by
sorry

end NUMINAMATH_CALUDE_magnitude_v_l1433_143355


namespace NUMINAMATH_CALUDE_book_reading_days_l1433_143363

theorem book_reading_days : ∀ (total_pages : ℕ) (pages_per_day : ℕ) (fraction : ℚ),
  total_pages = 144 →
  pages_per_day = 8 →
  fraction = 2/3 →
  (fraction * total_pages : ℚ) / pages_per_day = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_book_reading_days_l1433_143363


namespace NUMINAMATH_CALUDE_lassie_bones_problem_l1433_143399

theorem lassie_bones_problem (B : ℝ) : 
  (4/5 * (3/4 * (2/3 * B + 5) + 8) + 15 = 60) → B = 89 := by
  sorry

end NUMINAMATH_CALUDE_lassie_bones_problem_l1433_143399


namespace NUMINAMATH_CALUDE_system_solution_l1433_143319

theorem system_solution (x y z : ℝ) :
  x + y + z = 2 ∧ x * y * z = 2 * (x * y + y * z + z * x) →
  ((x = -y ∧ z = 2) ∨ (y = -z ∧ x = 2) ∨ (z = -x ∧ y = 2)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1433_143319


namespace NUMINAMATH_CALUDE_short_trees_after_planting_l1433_143379

theorem short_trees_after_planting 
  (initial_short_trees : ℕ) 
  (short_trees_to_plant : ℕ) 
  (h1 : initial_short_trees = 112)
  (h2 : short_trees_to_plant = 105) :
  initial_short_trees + short_trees_to_plant = 217 := by
  sorry

end NUMINAMATH_CALUDE_short_trees_after_planting_l1433_143379


namespace NUMINAMATH_CALUDE_least_positive_linear_combination_l1433_143308

theorem least_positive_linear_combination :
  ∃ (n : ℕ), n > 0 ∧ (∀ (m : ℕ), m > 0 → (∃ (x y : ℤ), 24 * x + 20 * y = m) → m ≥ n) ∧
  (∃ (x y : ℤ), 24 * x + 20 * y = n) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_linear_combination_l1433_143308


namespace NUMINAMATH_CALUDE_division_problem_l1433_143390

theorem division_problem : 240 / (12 + 14 * 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1433_143390


namespace NUMINAMATH_CALUDE_susan_is_eleven_l1433_143369

/-- Susan's age -/
def susan_age : ℕ := sorry

/-- Ann's age -/
def ann_age : ℕ := sorry

/-- Ann is 5 years older than Susan -/
axiom age_difference : ann_age = susan_age + 5

/-- The sum of their ages is 27 -/
axiom age_sum : ann_age + susan_age = 27

/-- Proof that Susan is 11 years old -/
theorem susan_is_eleven : susan_age = 11 := by sorry

end NUMINAMATH_CALUDE_susan_is_eleven_l1433_143369


namespace NUMINAMATH_CALUDE_art_gallery_pieces_l1433_143339

theorem art_gallery_pieces (total : ℕ) 
  (h1 : total / 3 = total - (total * 2 / 3))  -- 1/3 of pieces are displayed
  (h2 : (total / 3) / 6 = (total / 3) - ((total / 3) * 5 / 6))  -- 1/6 of displayed pieces are sculptures
  (h3 : (total * 2 / 3) / 3 = (total * 2 / 3) - ((total * 2 / 3) * 2 / 3))  -- 1/3 of not displayed pieces are paintings
  (h4 : (total * 2 / 3) * 2 / 3 = 400)  -- 400 sculptures are not on display
  : total = 900 := by
  sorry

end NUMINAMATH_CALUDE_art_gallery_pieces_l1433_143339


namespace NUMINAMATH_CALUDE_simplest_form_fraction_other_fractions_not_simplest_l1433_143382

/-- A fraction is in simplest form if its numerator and denominator have no common factors other than 1. -/
def IsSimplestForm (n d : ℤ) : Prop :=
  ∀ k : ℤ, k ∣ n ∧ k ∣ d → k = 1 ∨ k = -1

/-- The fraction (x^2 + y^2) / (x + y) is in simplest form. -/
theorem simplest_form_fraction (x y : ℤ) :
  IsSimplestForm (x^2 + y^2) (x + y) := by
  sorry

/-- Other fractions can be simplified further. -/
theorem other_fractions_not_simplest (x y : ℤ) :
  ¬IsSimplestForm (x * y) (x^2) ∧
  ¬IsSimplestForm (y^2 + y) (x * y) ∧
  ¬IsSimplestForm (x^2 - y^2) (x + y) := by
  sorry

end NUMINAMATH_CALUDE_simplest_form_fraction_other_fractions_not_simplest_l1433_143382


namespace NUMINAMATH_CALUDE_min_sum_of_coeffs_l1433_143384

/-- Given a quadratic function f(x) with real coefficients a, b, c, 
    if the range of f(x) is [0, +∞), then a + b + c ≥ √3 -/
theorem min_sum_of_coeffs (a b c : ℝ) : 
  (∀ x, (a + 2*b)*x^2 - 2*Real.sqrt 3*x + a + 2*c ≥ 0) → 
  a + b + c ≥ Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_coeffs_l1433_143384


namespace NUMINAMATH_CALUDE_sqrt_calculation_l1433_143353

theorem sqrt_calculation : Real.sqrt 24 * Real.sqrt (1/6) - (-Real.sqrt 7)^2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculation_l1433_143353


namespace NUMINAMATH_CALUDE_three_numbers_average_l1433_143391

theorem three_numbers_average (x y z : ℝ) : 
  y = 2 * x →
  z = 4 * y →
  y = 90 →
  (x + y + z) / 3 = 165 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_average_l1433_143391


namespace NUMINAMATH_CALUDE_expression_equality_l1433_143350

theorem expression_equality (x y z u a b c d : ℝ) :
  (a*x + b*y + c*z + d*u)^2 + (b*x + c*y + d*z + a*u)^2 + 
  (c*x + d*y + a*z + b*u)^2 + (d*x + a*y + b*z + c*u)^2 =
  (d*x + c*y + b*z + a*u)^2 + (c*x + b*y + a*z + d*u)^2 + 
  (b*x + a*y + d*z + c*u)^2 + (a*x + d*y + c*z + b*u)^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1433_143350


namespace NUMINAMATH_CALUDE_pet_store_cages_l1433_143337

/-- Given a pet store scenario, calculate the number of cages used -/
theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) : 
  initial_puppies = 120 → 
  sold_puppies = 108 → 
  puppies_per_cage = 6 → 
  (initial_puppies - sold_puppies) / puppies_per_cage = 2 := by
sorry

end NUMINAMATH_CALUDE_pet_store_cages_l1433_143337


namespace NUMINAMATH_CALUDE_eight_stairs_climb_ways_l1433_143371

/-- Represents the number of ways to climb n stairs with the given restrictions -/
def climbWays (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | n + 3 =>
    if n % 2 = 0 then
      climbWays (n + 2) + climbWays (n + 1)
    else
      climbWays (n + 2) + climbWays (n + 1) + climbWays n

theorem eight_stairs_climb_ways :
  climbWays 8 = 54 := by
  sorry

#eval climbWays 8

end NUMINAMATH_CALUDE_eight_stairs_climb_ways_l1433_143371


namespace NUMINAMATH_CALUDE_plain_cookie_price_l1433_143303

/-- The price of each box of plain cookies, given the total number of boxes sold,
    the combined value of all boxes, the number of plain cookie boxes sold,
    and the price of each box of chocolate chip cookies. -/
theorem plain_cookie_price
  (total_boxes : ℝ)
  (combined_value : ℝ)
  (plain_boxes : ℝ)
  (choc_chip_price : ℝ)
  (h1 : total_boxes = 1585)
  (h2 : combined_value = 1586.75)
  (h3 : plain_boxes = 793.375)
  (h4 : choc_chip_price = 1.25) :
  (combined_value - (total_boxes - plain_boxes) * choc_chip_price) / plain_boxes = 0.7525 := by
  sorry

end NUMINAMATH_CALUDE_plain_cookie_price_l1433_143303


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1433_143386

/-- Given an arithmetic sequence {a_n} with sum S_n, prove that if S_6 = 36, S_n = 324, S_(n-6) = 144, and n > 0, then n = 18. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :
  (∀ k, S k = (k / 2) * (2 * a 1 + (k - 1) * (a 2 - a 1))) →  -- Definition of S_n
  (n > 0) →                                                   -- Condition: n > 0
  (S 6 = 36) →                                                -- Condition: S_6 = 36
  (S n = 324) →                                               -- Condition: S_n = 324
  (S (n - 6) = 144) →                                         -- Condition: S_(n-6) = 144
  (n = 18) :=                                                 -- Conclusion: n = 18
by sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1433_143386


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_terms_is_two_l1433_143322

/-- The sequence a_n defined as n^2! + n -/
def a (n : ℕ) : ℕ := (Nat.factorial (n^2)) + n

/-- The theorem stating that the maximum GCD of consecutive terms in the sequence is 2 -/
theorem max_gcd_consecutive_terms_is_two :
  ∃ (k : ℕ), (∀ (n : ℕ), Nat.gcd (a n) (a (n + 1)) ≤ k) ∧ 
  (∃ (m : ℕ), Nat.gcd (a m) (a (m + 1)) = k) ∧
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_terms_is_two_l1433_143322


namespace NUMINAMATH_CALUDE_union_and_intersection_range_of_a_l1433_143362

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 2 < x ∧ x < 8}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ a + 3}

-- Theorem for part (1)
theorem union_and_intersection :
  (A ∪ B = {x | 1 ≤ x ∧ x < 8}) ∧
  ((Set.univ \ A) ∩ B = {x | 5 ≤ x ∧ x < 8}) := by sorry

-- Theorem for part (2)
theorem range_of_a :
  {a : ℝ | C a ∩ A = C a} = {a : ℝ | 1 ≤ a ∧ a < 2} := by sorry

end NUMINAMATH_CALUDE_union_and_intersection_range_of_a_l1433_143362


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l1433_143345

/-- Given the cost of 3 pens and 5 pencils, and the cost ratio of pen to pencil,
    prove the cost of one dozen pens. -/
theorem cost_of_dozen_pens (cost_3pens_5pencils : ℕ) (cost_ratio_pen_pencil : ℚ) :
  cost_3pens_5pencils = 200 →
  cost_ratio_pen_pencil = 5 / 1 →
  ∃ (cost_pen : ℚ), cost_pen * 12 = 600 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l1433_143345


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1433_143338

theorem inequality_and_equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b)^3 / (a^2 * b) ≥ 27/4 ∧ ((a + b)^3 / (a^2 * b) = 27/4 ↔ a = 2*b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1433_143338


namespace NUMINAMATH_CALUDE_wire_length_ratio_l1433_143393

theorem wire_length_ratio : 
  let large_cube_edge : ℝ := 8
  let large_cube_edges : ℕ := 12
  let unit_cube_edge : ℝ := 1
  let unit_cube_edges : ℕ := 12

  let large_cube_volume := large_cube_edge ^ 3
  let num_unit_cubes := large_cube_volume

  let large_cube_wire_length := large_cube_edge * large_cube_edges
  let unit_cubes_wire_length := num_unit_cubes * unit_cube_edge * unit_cube_edges

  large_cube_wire_length / unit_cubes_wire_length = 1 / 64 :=
by
  sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l1433_143393


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_3_5_7_l1433_143341

theorem smallest_five_digit_divisible_by_3_5_7 :
  ∀ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n → n ≥ 10080 :=
by
  sorry

#check smallest_five_digit_divisible_by_3_5_7

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_3_5_7_l1433_143341


namespace NUMINAMATH_CALUDE_square_less_than_four_implies_less_than_two_l1433_143316

theorem square_less_than_four_implies_less_than_two (x : ℝ) : x^2 < 4 → x < 2 := by
  sorry

end NUMINAMATH_CALUDE_square_less_than_four_implies_less_than_two_l1433_143316
