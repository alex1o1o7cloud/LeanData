import Mathlib

namespace NUMINAMATH_CALUDE_power_product_equals_reciprocal_l2724_272451

theorem power_product_equals_reciprocal (n : ℕ) :
  (125 : ℚ)^(2015 : ℕ) * (-0.008)^(2016 : ℕ) = 1 / 125 :=
by
  have h : (-0.008 : ℚ) = -1/125 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_product_equals_reciprocal_l2724_272451


namespace NUMINAMATH_CALUDE_monkey_giraffe_difference_l2724_272453

/-- The number of zebras Carla counted at the zoo -/
def num_zebras : ℕ := 12

/-- The number of camels at the zoo -/
def num_camels : ℕ := num_zebras / 2

/-- The number of monkeys at the zoo -/
def num_monkeys : ℕ := 4 * num_camels

/-- The number of giraffes at the zoo -/
def num_giraffes : ℕ := 2

/-- Theorem stating the difference between the number of monkeys and giraffes -/
theorem monkey_giraffe_difference : num_monkeys - num_giraffes = 22 := by
  sorry

end NUMINAMATH_CALUDE_monkey_giraffe_difference_l2724_272453


namespace NUMINAMATH_CALUDE_medians_form_right_triangle_l2724_272496

/-- Given a triangle ABC with sides a, b, c and corresponding medians m_a, m_b, m_c,
    if m_a ⊥ m_b, then m_a^2 + m_b^2 = m_c^2 -/
theorem medians_form_right_triangle (a b c m_a m_b m_c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_medians : m_a > 0 ∧ m_b > 0 ∧ m_c > 0)
  (h_perp : m_a * m_b = 0) : 
  m_a^2 + m_b^2 = m_c^2 := by
sorry

end NUMINAMATH_CALUDE_medians_form_right_triangle_l2724_272496


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_f_l2724_272426

-- Define the function
def f (x : ℝ) : ℝ := |x - 2| - 1

-- State the theorem
theorem monotonic_increasing_interval_f :
  ∀ a b : ℝ, a ≥ 2 → b ≥ 2 → a ≤ b → f a ≤ f b :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_f_l2724_272426


namespace NUMINAMATH_CALUDE_odd_digits_157_base5_l2724_272442

/-- Represents a number in base 5 as a list of digits (least significant digit first) -/
def Base5Rep := List Nat

/-- Converts a natural number to its base 5 representation -/
def toBase5 (n : Nat) : Base5Rep :=
  sorry

/-- Counts the number of odd digits in a base 5 representation -/
def countOddDigits (rep : Base5Rep) : Nat :=
  sorry

/-- The number of odd digits in the base-5 representation of 157₁₀ is 3 -/
theorem odd_digits_157_base5 : countOddDigits (toBase5 157) = 3 := by
  sorry

end NUMINAMATH_CALUDE_odd_digits_157_base5_l2724_272442


namespace NUMINAMATH_CALUDE_parallel_resistors_existence_l2724_272406

theorem parallel_resistors_existence : ∃ (R R₁ R₂ : ℕ+), 
  R.val * (R₁.val + R₂.val) = R₁.val * R₂.val ∧ 
  R.val > 0 ∧ R₁.val > 0 ∧ R₂.val > 0 := by
  sorry

end NUMINAMATH_CALUDE_parallel_resistors_existence_l2724_272406


namespace NUMINAMATH_CALUDE_function_forms_with_common_tangent_l2724_272440

/-- Given two functions f and g, prove that they have the specified forms
    when they pass through (2, 0) and have a common tangent at that point. -/
theorem function_forms_with_common_tangent 
  (f g : ℝ → ℝ) 
  (hf : ∃ a : ℝ, ∀ x, f x = 2 * x^3 + a * x)
  (hg : ∃ b c : ℝ, ∀ x, g x = b * x^2 + c)
  (pass_through : f 2 = 0 ∧ g 2 = 0)
  (common_tangent : (deriv f) 2 = (deriv g) 2) :
  (∀ x, f x = 2 * x^3 - 8 * x) ∧ 
  (∀ x, g x = 4 * x^2 - 16) := by
sorry

end NUMINAMATH_CALUDE_function_forms_with_common_tangent_l2724_272440


namespace NUMINAMATH_CALUDE_partnership_profit_calculation_l2724_272410

/-- Calculates the total profit of a partnership given investments and one partner's profit share -/
def calculate_total_profit (a_investment b_investment c_investment : ℕ) (c_profit : ℕ) : ℕ :=
  let ratio_sum := (a_investment / 8000) + (b_investment / 8000) + (c_investment / 8000)
  let profit_per_part := c_profit / (c_investment / 8000)
  ratio_sum * profit_per_part

/-- Theorem stating that given the specific investments and C's profit, the total profit is 92000 -/
theorem partnership_profit_calculation :
  calculate_total_profit 24000 32000 36000 36000 = 92000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_calculation_l2724_272410


namespace NUMINAMATH_CALUDE_grid_paths_equals_choose_l2724_272482

/-- The number of paths from (0,0) to (m,n) in a grid, moving only right or up -/
def gridPaths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) n

/-- Theorem: The number of paths in an m × n grid is (m+n) choose n -/
theorem grid_paths_equals_choose (m n : ℕ) : 
  gridPaths m n = Nat.choose (m + n) n := by
  sorry

end NUMINAMATH_CALUDE_grid_paths_equals_choose_l2724_272482


namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l2724_272447

/-- Inverse proportion function passing through (2,1) has k = 2 -/
theorem inverse_proportion_k_value (k : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = k / x) ∧ f 2 = 1) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l2724_272447


namespace NUMINAMATH_CALUDE_difference_value_l2724_272492

-- Define the variables x and y
variable (x y : ℝ)

-- Define the conditions
def sum_condition : Prop := x + y = 500
def ratio_condition : Prop := x / y = 0.8

-- Define the theorem
theorem difference_value (h1 : sum_condition x y) (h2 : ratio_condition x y) :
  ∃ ε > 0, |y - x - 55.56| < ε :=
sorry

end NUMINAMATH_CALUDE_difference_value_l2724_272492


namespace NUMINAMATH_CALUDE_nine_possible_scores_l2724_272498

/-- The number of baskets scored by the player -/
def total_baskets : ℕ := 8

/-- The possible point values for each basket -/
inductive BasketValue : Type
| one : BasketValue
| three : BasketValue

/-- A function to calculate the total score given a list of basket values -/
def total_score (baskets : List BasketValue) : ℕ :=
  baskets.foldl (fun acc b => acc + match b with
    | BasketValue.one => 1
    | BasketValue.three => 3) 0

/-- The theorem to be proved -/
theorem nine_possible_scores :
  ∃! (scores : Finset ℕ), 
    (∀ (score : ℕ), score ∈ scores ↔ 
      ∃ (baskets : List BasketValue), 
        baskets.length = total_baskets ∧ total_score baskets = score) ∧
    scores.card = 9 := by sorry

end NUMINAMATH_CALUDE_nine_possible_scores_l2724_272498


namespace NUMINAMATH_CALUDE_find_x_l2724_272474

theorem find_x : ∃ x : ℝ, 
  (24 + 35 + 58) / 3 = ((19 + 51 + x) / 3) + 6 → x = 29 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l2724_272474


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2724_272489

def A : Set ℝ := {x | 2 * x^2 - 3 * x - 2 ≤ 0}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2724_272489


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_6370_l2724_272434

theorem largest_prime_factor_of_6370 : (Nat.factors 6370).maximum? = some 13 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_6370_l2724_272434


namespace NUMINAMATH_CALUDE_ceiling_negative_sqrt_theorem_l2724_272462

theorem ceiling_negative_sqrt_theorem :
  ⌈-Real.sqrt ((64 : ℝ) / 9 - 1)⌉ = -2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_negative_sqrt_theorem_l2724_272462


namespace NUMINAMATH_CALUDE_problem_statement_l2724_272478

theorem problem_statement : (-4)^4 / 4^2 + 2^5 - 7^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2724_272478


namespace NUMINAMATH_CALUDE_point_C_y_coordinate_sum_of_digits_l2724_272421

/-- The function representing the graph y = x^2 + 2x -/
def f (x : ℝ) : ℝ := x^2 + 2*x

/-- Sum of digits of a real number -/
noncomputable def sumOfDigits (y : ℝ) : ℕ := sorry

theorem point_C_y_coordinate_sum_of_digits 
  (A B C : ℝ × ℝ) 
  (hA : A.2 = f A.1) 
  (hB : B.2 = f B.1) 
  (hC : C.2 = f C.1) 
  (hDistinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (hParallel : A.2 = B.2) 
  (hArea : abs ((B.1 - A.1) * (C.2 - A.2)) / 2 = 100) :
  sumOfDigits C.2 = 6 := by sorry

end NUMINAMATH_CALUDE_point_C_y_coordinate_sum_of_digits_l2724_272421


namespace NUMINAMATH_CALUDE_airport_distance_l2724_272422

/-- The distance from David's home to the airport in miles. -/
def distance_to_airport : ℝ := 160

/-- David's initial speed in miles per hour. -/
def initial_speed : ℝ := 40

/-- The increase in David's speed in miles per hour. -/
def speed_increase : ℝ := 20

/-- The time in hours David would be late if he continued at the initial speed. -/
def time_late : ℝ := 0.75

/-- The time in hours David arrives early with increased speed. -/
def time_early : ℝ := 0.25

/-- Theorem stating that the distance to the airport is 160 miles. -/
theorem airport_distance : 
  ∃ (t : ℝ), 
    distance_to_airport = initial_speed * (t + time_late) ∧
    distance_to_airport - initial_speed = (initial_speed + speed_increase) * (t - 1 - time_early) :=
by
  sorry


end NUMINAMATH_CALUDE_airport_distance_l2724_272422


namespace NUMINAMATH_CALUDE_exists_set_product_eq_sum_squares_l2724_272411

/-- For any finite set of positive integers, there exists a larger finite set
    where the product of its elements equals the sum of their squares. -/
theorem exists_set_product_eq_sum_squares (A : Finset ℕ) : ∃ B : Finset ℕ, 
  (∀ a ∈ A, a ∈ B) ∧ 
  (∀ b ∈ B, b > 0) ∧
  (B.prod id = B.sum (λ x => x^2)) := by
  sorry

end NUMINAMATH_CALUDE_exists_set_product_eq_sum_squares_l2724_272411


namespace NUMINAMATH_CALUDE_different_color_probability_l2724_272460

/-- The probability of drawing two balls of different colors from a box with 2 red and 3 black balls -/
theorem different_color_probability : 
  let total_balls : ℕ := 2 + 3
  let red_balls : ℕ := 2
  let black_balls : ℕ := 3
  let different_color_draws : ℕ := red_balls * black_balls
  let total_draws : ℕ := (total_balls * (total_balls - 1)) / 2
  (different_color_draws : ℚ) / total_draws = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l2724_272460


namespace NUMINAMATH_CALUDE_original_population_l2724_272499

def population_change (p : ℕ) : ℝ :=
  0.85 * (p + 1500 : ℝ) - p

theorem original_population : 
  ∃ p : ℕ, population_change p = -50 ∧ p = 8833 := by
  sorry

end NUMINAMATH_CALUDE_original_population_l2724_272499


namespace NUMINAMATH_CALUDE_product_five_fourth_sum_l2724_272477

theorem product_five_fourth_sum (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c → 
  a * b * c = 5^4 → 
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 131 := by sorry

end NUMINAMATH_CALUDE_product_five_fourth_sum_l2724_272477


namespace NUMINAMATH_CALUDE_cookies_left_l2724_272401

theorem cookies_left (initial_cookies eaten_cookies : ℕ) 
  (h1 : initial_cookies = 93)
  (h2 : eaten_cookies = 15) :
  initial_cookies - eaten_cookies = 78 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_l2724_272401


namespace NUMINAMATH_CALUDE_divisibility_product_l2724_272404

theorem divisibility_product (n a b c d : ℤ) 
  (ha : n ∣ a) (hb : n ∣ b) (hc : n ∣ c) (hd : n ∣ d) :
  n ∣ ((a - d) * (b - c)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_product_l2724_272404


namespace NUMINAMATH_CALUDE_inradius_right_triangle_l2724_272481

/-- The inradius of a right triangle with side lengths 9, 40, and 41 is 4. -/
theorem inradius_right_triangle : ∀ (a b c r : ℝ),
  a = 9 ∧ b = 40 ∧ c = 41 →
  a^2 + b^2 = c^2 →
  r = (a * b) / (2 * (a + b + c)) →
  r = 4 := by sorry

end NUMINAMATH_CALUDE_inradius_right_triangle_l2724_272481


namespace NUMINAMATH_CALUDE_a_power_b_minus_a_power_neg_b_l2724_272459

theorem a_power_b_minus_a_power_neg_b (a b : ℝ) (ha : a > 1) (hb : b > 0) 
  (h : a^b + a^(-b) = 2 * Real.sqrt 2) : a^b - a^(-b) = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_power_b_minus_a_power_neg_b_l2724_272459


namespace NUMINAMATH_CALUDE_acute_triangle_angle_inequality_iff_sine_inequality_l2724_272444

theorem acute_triangle_angle_inequality_iff_sine_inequality 
  (A B C : Real) (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) :
  (A > B ∧ B > C) ↔ (Real.sin (2*A) < Real.sin (2*B) ∧ Real.sin (2*B) < Real.sin (2*C)) :=
sorry

end NUMINAMATH_CALUDE_acute_triangle_angle_inequality_iff_sine_inequality_l2724_272444


namespace NUMINAMATH_CALUDE_money_distribution_l2724_272419

theorem money_distribution (total : ℕ) (a b c d : ℕ) : 
  a + b + c + d = total →
  5 * b = 2 * a →
  4 * b = 2 * c →
  3 * b = 2 * d →
  c = d + 500 →
  d = 1500 :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l2724_272419


namespace NUMINAMATH_CALUDE_units_digit_of_41_cubed_plus_23_cubed_l2724_272452

theorem units_digit_of_41_cubed_plus_23_cubed : (41^3 + 23^3) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_41_cubed_plus_23_cubed_l2724_272452


namespace NUMINAMATH_CALUDE_ozone_effect_significant_l2724_272443

/-- Represents the data from the experiment -/
structure ExperimentData where
  control_group : List Float
  experimental_group : List Float

/-- Calculates the median of a sorted list -/
def median (sorted_list : List Float) : Float :=
  sorry

/-- Counts the number of elements less than a given value -/
def count_less_than (list : List Float) (value : Float) : Nat :=
  sorry

/-- Calculates K² statistic -/
def calculate_k_squared (a b c d : Nat) : Float :=
  sorry

/-- Main theorem: K² value is greater than the critical value for 95% confidence -/
theorem ozone_effect_significant (data : ExperimentData) :
  let all_data := data.control_group ++ data.experimental_group
  let m := median all_data
  let a := count_less_than data.control_group m
  let b := data.control_group.length - a
  let c := count_less_than data.experimental_group m
  let d := data.experimental_group.length - c
  let k_squared := calculate_k_squared a b c d
  k_squared > 3.841 := by
  sorry

#check ozone_effect_significant

end NUMINAMATH_CALUDE_ozone_effect_significant_l2724_272443


namespace NUMINAMATH_CALUDE_cos120_plus_sin_neg45_l2724_272470

theorem cos120_plus_sin_neg45 : 
  Real.cos (120 * π / 180) + Real.sin (-45 * π / 180) = - (1 + Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos120_plus_sin_neg45_l2724_272470


namespace NUMINAMATH_CALUDE_odd_square_sum_parity_l2724_272439

theorem odd_square_sum_parity (n m : ℤ) (h : Odd (n^2 + m^2)) :
  ¬(Even n ∧ Even m) ∧ ¬(Odd n ∧ Odd m) := by
  sorry

end NUMINAMATH_CALUDE_odd_square_sum_parity_l2724_272439


namespace NUMINAMATH_CALUDE_line_circle_intersection_sufficient_not_necessary_condition_l2724_272414

theorem line_circle_intersection (k : ℝ) : 
  (∃ x y : ℝ, y = k * (x + 1) ∧ x^2 + y^2 - 2*x = 0) ↔ 
  (-Real.sqrt 3 / 3 ≤ k ∧ k ≤ Real.sqrt 3 / 3) :=
sorry

theorem sufficient_not_necessary_condition : 
  (∀ k : ℝ, -Real.sqrt 3 / 3 < k ∧ k < Real.sqrt 3 / 3 → 
    ∃ x y : ℝ, y = k * (x + 1) ∧ x^2 + y^2 - 2*x = 0) ∧
  (∃ k : ℝ, (k = -Real.sqrt 3 / 3 ∨ k = Real.sqrt 3 / 3) ∧
    ∃ x y : ℝ, y = k * (x + 1) ∧ x^2 + y^2 - 2*x = 0) :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_sufficient_not_necessary_condition_l2724_272414


namespace NUMINAMATH_CALUDE_function_properties_l2724_272407

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3| - 2 * |x + 1|

-- State the theorem
theorem function_properties :
  -- 1. The maximum value of f is 4
  (∃ (x : ℝ), f x = 4) ∧ (∀ (x : ℝ), f x ≤ 4) ∧
  -- 2. The solution set of f(x) < 1
  (∀ (x : ℝ), f x < 1 ↔ (x < -4 ∨ x > 0)) ∧
  -- 3. The maximum value of ab + bc given the constraints
  (∀ (a b c : ℝ), a > 0 → b > 0 → a^2 + 2*b^2 + c^2 = 4 → ab + bc ≤ 2) ∧
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ a^2 + 2*b^2 + c^2 = 4 ∧ ab + bc = 2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2724_272407


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_ratio_l2724_272423

theorem arithmetic_geometric_progression_ratio 
  (a₁ d : ℝ) (h : d ≠ 0) : 
  let a₂ := a₁ + d
  let a₃ := a₁ + 2*d
  let r := a₂ * a₃ / (a₁ * a₂)
  (r * r = 1 ∧ (a₂ * a₃) / (a₁ * a₂) = (a₃ * a₁) / (a₂ * a₃)) → r = -2 := by
  sorry

#check arithmetic_geometric_progression_ratio

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_ratio_l2724_272423


namespace NUMINAMATH_CALUDE_keanu_destination_distance_l2724_272475

/-- Represents the distance to Keanu's destination -/
def destination_distance : ℝ := 280

/-- Represents the capacity of Keanu's motorcycle's gas tank in liters -/
def tank_capacity : ℝ := 8

/-- Represents the distance Keanu's motorcycle can travel with one full tank in miles -/
def miles_per_tank : ℝ := 40

/-- Represents the number of times Keanu refills his motorcycle for a round trip -/
def refills : ℕ := 14

/-- Theorem stating that the distance to Keanu's destination is 280 miles -/
theorem keanu_destination_distance :
  destination_distance = (refills : ℝ) * miles_per_tank / 2 :=
sorry

end NUMINAMATH_CALUDE_keanu_destination_distance_l2724_272475


namespace NUMINAMATH_CALUDE_two_by_one_prism_net_squares_valid_nine_square_net_two_by_one_prism_net_property_l2724_272464

/-- Represents a rectangular prism --/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a net of a rectangular prism --/
structure PrismNet where
  squares : ℕ

/-- Function to calculate the number of squares in a prism net --/
def netSquares (prism : RectangularPrism) : ℕ :=
  2 * (prism.length * prism.width + prism.length * prism.height + prism.width * prism.height)

/-- Theorem stating that a 2x1x1 prism net has 10 squares --/
theorem two_by_one_prism_net_squares :
  let prism : RectangularPrism := ⟨2, 1, 1⟩
  netSquares prism = 10 := by sorry

/-- Theorem stating that removing one square from a 10-square net results in a 9-square net --/
theorem valid_nine_square_net (net : PrismNet) (h : net.squares = 10) :
  ∃ (reduced_net : PrismNet), reduced_net.squares = 9 := by sorry

/-- Main theorem combining the above results --/
theorem two_by_one_prism_net_property :
  let prism : RectangularPrism := ⟨2, 1, 1⟩
  let net : PrismNet := ⟨netSquares prism⟩
  ∃ (reduced_net : PrismNet), reduced_net.squares = 9 := by sorry

end NUMINAMATH_CALUDE_two_by_one_prism_net_squares_valid_nine_square_net_two_by_one_prism_net_property_l2724_272464


namespace NUMINAMATH_CALUDE_seventh_term_is_64_l2724_272485

/-- A geometric sequence with given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 2) * a n = (a (n + 1))^2
  sum_first_two : a 1 + a 2 = 3
  sum_second_third : a 2 + a 3 = 6

/-- The 7th term of the geometric sequence is 64 -/
theorem seventh_term_is_64 (seq : GeometricSequence) : seq.a 7 = 64 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_64_l2724_272485


namespace NUMINAMATH_CALUDE_valuation_problems_l2724_272408

/-- The p-adic valuation of an integer n -/
noncomputable def padic_valuation (p : ℕ) (n : ℤ) : ℕ := sorry

theorem valuation_problems :
  (padic_valuation 3 (2^27 + 1) = 4) ∧
  (padic_valuation 7 (161^14 - 112^14) = 16) ∧
  (padic_valuation 2 (7^20 + 1) = 1) ∧
  (padic_valuation 2 (17^48 - 5^48) = 6) := by sorry

end NUMINAMATH_CALUDE_valuation_problems_l2724_272408


namespace NUMINAMATH_CALUDE_triangle_product_inequality_l2724_272415

/-- Triangle structure with sides a, b, c, perimeter P, and inscribed circle radius r -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  P : ℝ
  r : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  pos_P : 0 < P
  pos_r : 0 < r
  perimeter_def : P = a + b + c
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- The product of any two sides of a triangle is not less than
    the product of its perimeter and the radius of its inscribed circle -/
theorem triangle_product_inequality (t : Triangle) : t.a * t.b ≥ t.P * t.r := by
  sorry

end NUMINAMATH_CALUDE_triangle_product_inequality_l2724_272415


namespace NUMINAMATH_CALUDE_squareable_numbers_l2724_272428

/-- A natural number is squareable if the numbers from 1 to n can be arranged
    such that each number plus its index is a perfect square. -/
def Squareable (n : ℕ) : Prop :=
  ∃ (σ : Fin n → Fin n), Function.Bijective σ ∧
    ∀ (i : Fin n), ∃ (k : ℕ), (σ i).val + i.val + 1 = k^2

theorem squareable_numbers : 
  ¬ Squareable 7 ∧ Squareable 9 ∧ ¬ Squareable 11 ∧ Squareable 15 :=
sorry

end NUMINAMATH_CALUDE_squareable_numbers_l2724_272428


namespace NUMINAMATH_CALUDE_missing_sale_is_correct_l2724_272456

/-- Calculates the missing sale amount given the other 5 sales and the target average -/
def calculate_missing_sale (sale1 sale3 sale4 sale5 sale6 average : ℝ) : ℝ :=
  6 * average - (sale1 + sale3 + sale4 + sale5 + sale6)

/-- Proves that the calculated missing sale is correct given the problem conditions -/
theorem missing_sale_is_correct (sale1 sale3 sale4 sale5 sale6 average : ℝ) 
  (h1 : sale1 = 5420)
  (h3 : sale3 = 6200)
  (h4 : sale4 = 6350)
  (h5 : sale5 = 6500)
  (h6 : sale6 = 7070)
  (havg : average = 6200) :
  calculate_missing_sale sale1 sale3 sale4 sale5 sale6 average = 5660 := by
  sorry

#eval calculate_missing_sale 5420 6200 6350 6500 7070 6200

end NUMINAMATH_CALUDE_missing_sale_is_correct_l2724_272456


namespace NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_solve_equation_3_l2724_272455

-- Equation 1: (x-2)^2 = 25
theorem solve_equation_1 : 
  ∃ x₁ x₂ : ℝ, (x₁ - 2)^2 = 25 ∧ (x₂ - 2)^2 = 25 ∧ x₁ = 7 ∧ x₂ = -3 :=
sorry

-- Equation 2: x^2 + 4x + 3 = 0
theorem solve_equation_2 : 
  ∃ x₁ x₂ : ℝ, x₁^2 + 4*x₁ + 3 = 0 ∧ x₂^2 + 4*x₂ + 3 = 0 ∧ x₁ = -3 ∧ x₂ = -1 :=
sorry

-- Equation 3: 2x^2 + 4x - 1 = 0
theorem solve_equation_3 : 
  ∃ x₁ x₂ : ℝ, 2*x₁^2 + 4*x₁ - 1 = 0 ∧ 2*x₂^2 + 4*x₂ - 1 = 0 ∧ 
  x₁ = (-2 + Real.sqrt 6) / 2 ∧ x₂ = (-2 - Real.sqrt 6) / 2 :=
sorry

end NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_solve_equation_3_l2724_272455


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2724_272424

/-- A quadratic function with real coefficients -/
def f (a b x : ℝ) := x^2 + a*x + b

/-- The theorem statement -/
theorem quadratic_function_properties
  (a b : ℝ)
  (h_range : Set.range (f a b) = Set.Ici 0)
  (c m : ℝ)
  (h_solution_set : { x | f a b x < m } = Set.Ioo c (c + 2*Real.sqrt 2)) :
  m = 2 ∧
  ∃ (min_value : ℝ),
    min_value = 3 + 2*Real.sqrt 2 ∧
    ∀ (x y : ℝ), x > 1 → y > 0 → x + y = m →
      1 / (x - 1) + 2 / y ≥ min_value :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2724_272424


namespace NUMINAMATH_CALUDE_tv_price_change_l2724_272468

theorem tv_price_change (P : ℝ) (x : ℝ) : 
  P * (1 - x / 100) * 1.3 = P * 1.17 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_change_l2724_272468


namespace NUMINAMATH_CALUDE_sum_smallest_largest_prime_1_to_50_l2724_272402

theorem sum_smallest_largest_prime_1_to_50 : 
  ∃ (p q : ℕ), 
    p.Prime ∧ q.Prime ∧ 
    1 < p ∧ p ≤ 50 ∧ 
    1 < q ∧ q ≤ 50 ∧ 
    (∀ r : ℕ, r.Prime → 1 < r → r ≤ 50 → p ≤ r ∧ r ≤ q) ∧ 
    p + q = 49 :=
by sorry

end NUMINAMATH_CALUDE_sum_smallest_largest_prime_1_to_50_l2724_272402


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2724_272403

theorem complex_magnitude_problem (m : ℂ) :
  (((4 : ℂ) + m * Complex.I) / ((1 : ℂ) + 2 * Complex.I)).im = 0 →
  Complex.abs (m + 6 * Complex.I) = 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2724_272403


namespace NUMINAMATH_CALUDE_system_solution_l2724_272425

theorem system_solution (a b c d : ℚ) 
  (eq1 : 3 * a + 4 * b + 6 * c + 8 * d = 48)
  (eq2 : 4 * (d + c) = b)
  (eq3 : 4 * b + 2 * c = a)
  (eq4 : c + 1 = d) :
  a + b + c + d = 513 / 37 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2724_272425


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l2724_272454

theorem fraction_equals_zero (x : ℝ) : (x - 2) / (x + 2) = 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l2724_272454


namespace NUMINAMATH_CALUDE_dresser_shirts_count_l2724_272483

/-- Given a dresser with pants and shirts in the ratio of 7:10, 
    and 14 pants, prove that there are 20 shirts. -/
theorem dresser_shirts_count (pants_count : ℕ) (ratio_pants : ℕ) (ratio_shirts : ℕ) :
  pants_count = 14 →
  ratio_pants = 7 →
  ratio_shirts = 10 →
  (pants_count : ℚ) / ratio_pants * ratio_shirts = 20 := by
  sorry

end NUMINAMATH_CALUDE_dresser_shirts_count_l2724_272483


namespace NUMINAMATH_CALUDE_a_in_set_a_b_l2724_272488

universe u

variables {a b : Type u}

/-- Prove that a is an element of the set {a, b}. -/
theorem a_in_set_a_b : a ∈ ({a, b} : Set (Type u)) := by
  sorry

end NUMINAMATH_CALUDE_a_in_set_a_b_l2724_272488


namespace NUMINAMATH_CALUDE_condition_one_implies_right_triangle_condition_two_implies_right_triangle_condition_three_not_implies_right_triangle_condition_four_implies_right_triangle_l2724_272400

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  sum_of_angles : A + B + C = 180

-- Define what it means for a triangle to be right-angled
def is_right_triangle (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

-- Condition 1
theorem condition_one_implies_right_triangle (t : Triangle) 
  (h : t.A + t.B = t.C) : is_right_triangle t :=
sorry

-- Condition 2
theorem condition_two_implies_right_triangle (t : Triangle) 
  (h : ∃ (k : Real), t.A = k ∧ t.B = 2*k ∧ t.C = 3*k) : is_right_triangle t :=
sorry

-- Condition 3
theorem condition_three_not_implies_right_triangle : ∃ (t : Triangle), 
  (t.A = t.B ∧ t.B = t.C) ∧ ¬(is_right_triangle t) :=
sorry

-- Condition 4
theorem condition_four_implies_right_triangle (t : Triangle) 
  (h : t.A = 90 - t.B) : is_right_triangle t :=
sorry

end NUMINAMATH_CALUDE_condition_one_implies_right_triangle_condition_two_implies_right_triangle_condition_three_not_implies_right_triangle_condition_four_implies_right_triangle_l2724_272400


namespace NUMINAMATH_CALUDE_larger_cube_volume_l2724_272405

/-- The volume of a cube composed of 125 smaller cubes is equal to 125 times the volume of one small cube. -/
theorem larger_cube_volume (small_cube_volume : ℝ) (larger_cube_volume : ℝ) 
  (h1 : small_cube_volume > 0)
  (h2 : larger_cube_volume > 0)
  (h3 : ∃ (n : ℕ), n ^ 3 = 125)
  (h4 : larger_cube_volume = (5 : ℝ) ^ 3 * small_cube_volume) :
  larger_cube_volume = 125 * small_cube_volume := by
  sorry

end NUMINAMATH_CALUDE_larger_cube_volume_l2724_272405


namespace NUMINAMATH_CALUDE_vertex_coordinates_l2724_272487

/-- The quadratic function f(x) = x^2 - 2x -/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The x-coordinate of the vertex of f -/
def vertex_x : ℝ := 1

/-- The y-coordinate of the vertex of f -/
def vertex_y : ℝ := -1

/-- Theorem: The vertex of the quadratic function f(x) = x^2 - 2x has coordinates (1, -1) -/
theorem vertex_coordinates :
  (∀ x : ℝ, f x ≥ f vertex_x) ∧ f vertex_x = vertex_y :=
sorry

end NUMINAMATH_CALUDE_vertex_coordinates_l2724_272487


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2724_272445

theorem trigonometric_simplification (x : ℝ) :
  (2 * Real.cos x ^ 4 - 2 * Real.cos x ^ 2 + 1/2) / 
  (2 * Real.tan (π/4 - x) * Real.sin (π/4 + x) ^ 2) = 
  (1/2) * Real.cos (2*x) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2724_272445


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l2724_272429

theorem initial_number_of_persons 
  (average_weight_increase : ℝ) 
  (weight_of_leaving_person : ℝ) 
  (weight_of_new_person : ℝ) : 
  average_weight_increase = 4.5 ∧ 
  weight_of_leaving_person = 65 ∧ 
  weight_of_new_person = 74 → 
  (weight_of_new_person - weight_of_leaving_person) / average_weight_increase = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_persons_l2724_272429


namespace NUMINAMATH_CALUDE_lawnmower_blade_cost_l2724_272450

/-- The cost of a single lawnmower blade -/
def blade_cost : ℝ := sorry

/-- The number of lawnmower blades purchased -/
def num_blades : ℕ := 4

/-- The cost of the weed eater string -/
def string_cost : ℝ := 7

/-- The total cost of supplies -/
def total_cost : ℝ := 39

/-- Theorem stating that the cost of each lawnmower blade is $8 -/
theorem lawnmower_blade_cost : 
  blade_cost = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_lawnmower_blade_cost_l2724_272450


namespace NUMINAMATH_CALUDE_max_shipping_cost_l2724_272466

/-- The maximum shipping cost per unit for an electronic component manufacturer --/
theorem max_shipping_cost (production_cost : ℝ) (fixed_costs : ℝ) (units : ℕ) (selling_price : ℝ)
  (h1 : production_cost = 80)
  (h2 : fixed_costs = 16500)
  (h3 : units = 150)
  (h4 : selling_price = 193.33) :
  ∃ (shipping_cost : ℝ), shipping_cost ≤ 3.33 ∧
    units * (production_cost + shipping_cost) + fixed_costs ≤ units * selling_price :=
by sorry

end NUMINAMATH_CALUDE_max_shipping_cost_l2724_272466


namespace NUMINAMATH_CALUDE_annulus_area_l2724_272467

/-- The area of an annulus formed by two concentric circles -/
theorem annulus_area (B C RW : ℝ) (h1 : B > C) (h2 : B^2 - (C+5)^2 = RW^2) :
  (π * B^2) - (π * (C+5)^2) = π * RW^2 := by
  sorry

end NUMINAMATH_CALUDE_annulus_area_l2724_272467


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2724_272497

theorem polynomial_evaluation : 7^4 - 4 * 7^3 + 6 * 7^2 - 5 * 7 + 3 = 1553 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2724_272497


namespace NUMINAMATH_CALUDE_ryan_english_hours_l2724_272416

/-- The number of hours Ryan spends on learning Spanish -/
def spanish_hours : ℕ := 4

/-- The additional hours Ryan spends on learning English compared to Spanish -/
def additional_english_hours : ℕ := 3

/-- The number of hours Ryan spends on learning English -/
def english_hours : ℕ := spanish_hours + additional_english_hours

theorem ryan_english_hours : english_hours = 7 := by
  sorry

end NUMINAMATH_CALUDE_ryan_english_hours_l2724_272416


namespace NUMINAMATH_CALUDE_early_registration_percentage_l2724_272486

/-- The percentage of attendees who registered at least two weeks in advance and paid in full -/
def early_reg_and_paid : ℝ := 78

/-- The percentage of attendees who paid in full but did not register early -/
def paid_not_early : ℝ := 10

/-- Proves that the percentage of attendees who registered at least two weeks in advance is 78% -/
theorem early_registration_percentage : ℝ := by
  sorry

end NUMINAMATH_CALUDE_early_registration_percentage_l2724_272486


namespace NUMINAMATH_CALUDE_projection_problem_l2724_272430

/-- Given that the projection of (2, -3) onto some vector results in (1, -3/2),
    prove that the projection of (-3, 2) onto the same vector is (-24/13, 36/13) -/
theorem projection_problem (v : ℝ × ℝ) :
  let u₁ : ℝ × ℝ := (2, -3)
  let u₂ : ℝ × ℝ := (-3, 2)
  let proj₁ : ℝ × ℝ := (1, -3/2)
  (∃ (k : ℝ), v = k • proj₁) →
  (u₁ • v / (v • v)) • v = proj₁ →
  (u₂ • v / (v • v)) • v = (-24/13, 36/13) := by
  sorry


end NUMINAMATH_CALUDE_projection_problem_l2724_272430


namespace NUMINAMATH_CALUDE_mildreds_initial_blocks_l2724_272417

/-- Proves that Mildred's initial number of blocks was 2, given that she found 84 blocks and ended up with 86 blocks. -/
theorem mildreds_initial_blocks (found_blocks : ℕ) (final_blocks : ℕ) (h1 : found_blocks = 84) (h2 : final_blocks = 86) :
  final_blocks - found_blocks = 2 := by
  sorry

#check mildreds_initial_blocks

end NUMINAMATH_CALUDE_mildreds_initial_blocks_l2724_272417


namespace NUMINAMATH_CALUDE_sqrt_39_equals_33_l2724_272480

theorem sqrt_39_equals_33 : Real.sqrt 39 = 33 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_39_equals_33_l2724_272480


namespace NUMINAMATH_CALUDE_intersection_A_B_l2724_272409

-- Define set A
def A : Set ℝ := {x | x - 1 ≤ 0}

-- Define set B
def B : Set ℝ := {0, 1, 2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2724_272409


namespace NUMINAMATH_CALUDE_train_and_car_numbers_l2724_272438

/-- Represents a digit in the range 0 to 9 -/
def Digit := Fin 10

/-- Represents a mapping from characters to digits -/
def CodeMap := Char → Digit

/-- Checks if a CodeMap is valid (injective) -/
def isValidCodeMap (m : CodeMap) : Prop :=
  ∀ c1 c2, c1 ≠ c2 → m c1 ≠ m c2

/-- Converts a string to a number using a CodeMap -/
def stringToNumber (s : String) (m : CodeMap) : ℕ :=
  s.foldl (fun acc c => acc * 10 + (m c).val) 0

/-- The main theorem -/
theorem train_and_car_numbers : ∃ (m : CodeMap),
  isValidCodeMap m ∧
  stringToNumber "SECRET" m - stringToNumber "OPEN" m = stringToNumber "ANSWER" m - stringToNumber "YOUR" m ∧
  stringToNumber "SECRET" m - stringToNumber "OPENED" m = 20010 ∧
  stringToNumber "TRAIN" m = 392 ∧
  stringToNumber "CAR" m = 2 := by
  sorry

end NUMINAMATH_CALUDE_train_and_car_numbers_l2724_272438


namespace NUMINAMATH_CALUDE_solve_system_1_l2724_272441

theorem solve_system_1 (x y : ℝ) : 
  2 * x + 3 * y = 16 ∧ x + 4 * y = 13 → x = 5 ∧ y = 2 := by
  sorry

#check solve_system_1

end NUMINAMATH_CALUDE_solve_system_1_l2724_272441


namespace NUMINAMATH_CALUDE_missing_digit_divisible_by_six_l2724_272448

theorem missing_digit_divisible_by_six : ∃ (n : ℕ), 
  n ≥ 31610 ∧ n ≤ 31619 ∧ n % 10 = 4 ∧ n % 100 = 14 ∧ n % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_missing_digit_divisible_by_six_l2724_272448


namespace NUMINAMATH_CALUDE_power_subtraction_equivalence_l2724_272418

theorem power_subtraction_equivalence :
  2^345 - 3^4 * 9^2 = 2^345 - 6561 :=
by sorry

end NUMINAMATH_CALUDE_power_subtraction_equivalence_l2724_272418


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2724_272471

theorem complex_equation_solution (b : ℝ) : (2 + b * Complex.I) * Complex.I = 2 + 2 * Complex.I → b = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2724_272471


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2724_272427

/-- Given an arithmetic sequence with common difference 2,
    if a_1, a_3, and a_4 form a geometric sequence, then a_1 = -8 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 3)^2 = a 1 * a 4 →         -- a_1, a_3, a_4 form a geometric sequence
  a 1 = -8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2724_272427


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l2724_272413

theorem quadratic_no_real_roots : ∀ x : ℝ, x^2 - 2*x + 2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l2724_272413


namespace NUMINAMATH_CALUDE_derivative_at_one_l2724_272476

theorem derivative_at_one (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, f x = x^2 + 3*x*(f' 1)) →
  (∀ x, HasDerivAt f (f' x) x) →
  f' 1 = -1 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_one_l2724_272476


namespace NUMINAMATH_CALUDE_aquarium_dolphins_l2724_272420

/-- The number of hours each dolphin requires for daily training -/
def training_hours_per_dolphin : ℕ := 3

/-- The number of trainers in the aquarium -/
def number_of_trainers : ℕ := 2

/-- The number of hours each trainer spends training dolphins -/
def hours_per_trainer : ℕ := 6

/-- The total number of training hours available -/
def total_training_hours : ℕ := number_of_trainers * hours_per_trainer

/-- The number of dolphins in the aquarium -/
def number_of_dolphins : ℕ := total_training_hours / training_hours_per_dolphin

theorem aquarium_dolphins :
  number_of_dolphins = 4 := by sorry

end NUMINAMATH_CALUDE_aquarium_dolphins_l2724_272420


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l2724_272449

/-- An isosceles triangle with given altitude and perimeter -/
structure IsoscelesTriangle where
  -- The length of the altitude to the base
  altitude : ℝ
  -- The perimeter of the triangle
  perimeter : ℝ
  -- The triangle is isosceles
  isIsosceles : True

/-- The area of an isosceles triangle -/
def area (t : IsoscelesTriangle) : ℝ :=
  sorry -- The actual calculation of the area would go here

/-- Theorem: The area of an isosceles triangle with altitude 10 and perimeter 40 is 75 -/
theorem isosceles_triangle_area :
  ∀ t : IsoscelesTriangle, t.altitude = 10 ∧ t.perimeter = 40 → area t = 75 :=
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_area_l2724_272449


namespace NUMINAMATH_CALUDE_standard_deviation_measures_stability_l2724_272463

-- Define a type for yield per acre
def YieldPerAcre := ℝ

-- Define a function to calculate the standard deviation
def standardDeviation (yields : List YieldPerAcre) : ℝ :=
  sorry  -- Implementation details omitted

-- Define a predicate for stability measure
def isStabilityMeasure (f : List YieldPerAcre → ℝ) : Prop :=
  sorry  -- Implementation details omitted

-- Theorem statement
theorem standard_deviation_measures_stability :
  ∀ (n : ℕ) (yields : List YieldPerAcre),
    n > 0 →
    yields.length = n →
    isStabilityMeasure standardDeviation :=
by sorry

end NUMINAMATH_CALUDE_standard_deviation_measures_stability_l2724_272463


namespace NUMINAMATH_CALUDE_hcf_problem_l2724_272436

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 82500) (h2 : Nat.lcm a b = 1500) :
  Nat.gcd a b = 55 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l2724_272436


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l2724_272446

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) : 
  ∃ (k : ℤ), (n - 1) + n + (n + 1) + (n + 2) = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l2724_272446


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l2724_272491

theorem quadratic_rewrite (d e f : ℤ) : 
  (∀ x, 25 * x^2 - 40 * x - 75 = (d * x + e)^2 + f) → d * e = -20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l2724_272491


namespace NUMINAMATH_CALUDE_pencil_sharpening_l2724_272412

/-- Given the initial and final lengths of a pencil, calculate the length sharpened off. -/
theorem pencil_sharpening (initial_length final_length : ℕ) : 
  initial_length ≥ final_length → 
  initial_length - final_length = initial_length - final_length :=
by sorry

end NUMINAMATH_CALUDE_pencil_sharpening_l2724_272412


namespace NUMINAMATH_CALUDE_unique_line_exists_l2724_272435

def intersectionPoints (k m n c : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((k, k^2 + 8*k + c), (k, m*k + n))

def verticalDistance (p q : ℝ × ℝ) : ℝ :=
  |p.2 - q.2|

theorem unique_line_exists (c : ℝ) : 
  ∃! m n : ℝ, n ≠ 0 ∧ 
  (∃ k : ℝ, verticalDistance (intersectionPoints k m n c).1 (intersectionPoints k m n c).2 = 4) ∧
  (m * 2 + n = 7) :=
sorry

end NUMINAMATH_CALUDE_unique_line_exists_l2724_272435


namespace NUMINAMATH_CALUDE_mean_home_runs_l2724_272458

theorem mean_home_runs : 
  let total_players : ℕ := 2 + 3 + 2 + 1 + 1
  let total_home_runs : ℕ := 2 * 5 + 3 * 6 + 2 * 8 + 1 * 9 + 1 * 11
  (total_home_runs : ℚ) / total_players = 64 / 9 := by sorry

end NUMINAMATH_CALUDE_mean_home_runs_l2724_272458


namespace NUMINAMATH_CALUDE_tetrahedron_pigeonhole_l2724_272472

/-- Represents the three possible states a point can be in -/
inductive PointState
  | Type1
  | Type2
  | Outside

/-- Represents a tetrahedron with vertices labeled by their state -/
structure Tetrahedron :=
  (vertices : Fin 4 → PointState)

/-- Theorem statement -/
theorem tetrahedron_pigeonhole (t : Tetrahedron) : 
  ∃ (i j : Fin 4), i ≠ j ∧ t.vertices i = t.vertices j :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_pigeonhole_l2724_272472


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l2724_272479

/-- Given two rectangles A and B with sides (a, b) and (c, d) respectively,
    if a / c = b / d = 2 / 3, then the ratio of the area of rectangle A
    to the area of rectangle B is 4:9. -/
theorem rectangle_area_ratio (a b c d : ℝ) (h1 : a / c = 2 / 3) (h2 : b / d = 2 / 3) :
  (a * b) / (c * d) = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l2724_272479


namespace NUMINAMATH_CALUDE_book_club_combinations_l2724_272493

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of people in the book club --/
def total_people : ℕ := 5

/-- The number of people who lead the discussion --/
def discussion_leaders : ℕ := 3

theorem book_club_combinations :
  choose total_people discussion_leaders = 10 := by
  sorry

end NUMINAMATH_CALUDE_book_club_combinations_l2724_272493


namespace NUMINAMATH_CALUDE_no_constant_difference_integer_l2724_272432

theorem no_constant_difference_integer (x : ℤ) : 
  ¬∃ (k : ℤ), 
    (x^2 - 4*x + 5) - (2*x - 6) = k ∧ 
    (4*x - 8) - (x^2 - 4*x + 5) = k ∧ 
    (3*x^2 - 12*x + 11) - (4*x - 8) = k :=
by sorry

end NUMINAMATH_CALUDE_no_constant_difference_integer_l2724_272432


namespace NUMINAMATH_CALUDE_floor_length_percentage_more_than_breadth_l2724_272469

theorem floor_length_percentage_more_than_breadth 
  (length : Real) 
  (area : Real) 
  (h1 : length = 13.416407864998739)
  (h2 : area = 60) :
  let breadth := area / length
  (length - breadth) / breadth * 100 = 200 := by
  sorry

end NUMINAMATH_CALUDE_floor_length_percentage_more_than_breadth_l2724_272469


namespace NUMINAMATH_CALUDE_conference_handshakes_l2724_272490

/-- Represents a conference with handshakes --/
structure Conference where
  total_people : Nat
  normal_handshakes : Nat
  restricted_people : Nat
  restricted_handshakes : Nat

/-- Calculates the maximum number of unique handshakes in a conference --/
def max_handshakes (c : Conference) : Nat :=
  let total_pairs := c.total_people.choose 2
  let reduced_handshakes := c.restricted_people * (c.normal_handshakes - c.restricted_handshakes)
  total_pairs - reduced_handshakes

/-- The theorem stating the maximum number of handshakes for the given conference --/
theorem conference_handshakes :
  let c : Conference := {
    total_people := 25,
    normal_handshakes := 20,
    restricted_people := 5,
    restricted_handshakes := 15
  }
  max_handshakes c = 250 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l2724_272490


namespace NUMINAMATH_CALUDE_triangle_angle_sine_relation_l2724_272494

theorem triangle_angle_sine_relation (A B : Real) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) :
  A > B ↔ Real.sin A > Real.sin B := by sorry

end NUMINAMATH_CALUDE_triangle_angle_sine_relation_l2724_272494


namespace NUMINAMATH_CALUDE_brownie_division_l2724_272484

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents the pan of brownies -/
def pan : Dimensions := ⟨24, 15⟩

/-- Represents a single piece of brownie -/
def piece : Dimensions := ⟨3, 2⟩

/-- Theorem stating that the pan can be divided into exactly 60 pieces -/
theorem brownie_division :
  (area pan) / (area piece) = 60 := by sorry

end NUMINAMATH_CALUDE_brownie_division_l2724_272484


namespace NUMINAMATH_CALUDE_license_plate_count_l2724_272431

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of letter positions in the license plate -/
def letter_positions : ℕ := 5

/-- The number of digit positions in the license plate -/
def digit_positions : ℕ := 2

/-- The number of odd single-digit numbers -/
def odd_digits : ℕ := 5

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The number of license plate combinations -/
def license_plate_combinations : ℕ :=
  (choose alphabet_size 2) * (choose letter_positions 2) * (choose (letter_positions - 2) 2) * 
  (alphabet_size - 2) * odd_digits * (odd_digits - 1)

theorem license_plate_count : license_plate_combinations = 936000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l2724_272431


namespace NUMINAMATH_CALUDE_corn_donation_l2724_272495

def total_bushels : ℕ := 50
def ears_per_bushel : ℕ := 14
def remaining_ears : ℕ := 357

theorem corn_donation :
  let total_ears := total_bushels * ears_per_bushel
  let given_away_ears := total_ears - remaining_ears
  given_away_ears / ears_per_bushel = 24 :=
by sorry

end NUMINAMATH_CALUDE_corn_donation_l2724_272495


namespace NUMINAMATH_CALUDE_rose_apples_l2724_272433

/-- The number of friends Rose shares her apples with -/
def num_friends : ℕ := 3

/-- The number of apples each friend would get if Rose shares her apples -/
def apples_per_friend : ℕ := 3

/-- The total number of apples Rose has -/
def total_apples : ℕ := num_friends * apples_per_friend

theorem rose_apples : total_apples = 9 := by sorry

end NUMINAMATH_CALUDE_rose_apples_l2724_272433


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l2724_272457

def C : Set Nat := {65, 67, 68, 71, 73}

def hasSmallerPrimeFactor (a b : Nat) : Prop :=
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧ p ∣ a ∧ q ∣ b ∧ ∀ r < q, Nat.Prime r → ¬(r ∣ b)

theorem smallest_prime_factor_in_C :
  ∀ n ∈ C, n ≠ 68 → hasSmallerPrimeFactor 68 n :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l2724_272457


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l2724_272461

theorem average_of_three_numbers (y : ℝ) : 
  (15 + 28 + y) / 3 = 25 → y = 32 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l2724_272461


namespace NUMINAMATH_CALUDE_sample_size_is_hundred_l2724_272465

/-- Represents a statistical study on student scores -/
structure ScoreStudy where
  population_size : ℕ
  extracted_size : ℕ

/-- Defines the sample size of a score study -/
def sample_size (study : ScoreStudy) : ℕ := study.extracted_size

/-- Theorem stating that for the given study, the sample size is 100 -/
theorem sample_size_is_hundred (study : ScoreStudy) 
  (h1 : study.population_size = 1000)
  (h2 : study.extracted_size = 100) : 
  sample_size study = 100 := by
  sorry

#check sample_size_is_hundred

end NUMINAMATH_CALUDE_sample_size_is_hundred_l2724_272465


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l2724_272437

theorem factorial_sum_equality : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l2724_272437


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2724_272473

theorem quadratic_one_solution (q : ℝ) (h : q ≠ 0) : 
  (q = 64/9) ↔ (∃! x : ℝ, q * x^2 - 16 * x + 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2724_272473
