import Mathlib

namespace bottle_volume_is_one_and_half_quarts_l1311_131146

/-- Represents the daily water consumption of Tim -/
structure DailyWaterConsumption where
  bottles : ℕ := 2
  additional_ounces : ℕ := 20

/-- Represents the weekly water consumption in ounces -/
def weekly_ounces : ℕ := 812

/-- Conversion factor from ounces to quarts -/
def ounces_per_quart : ℕ := 32

/-- Number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem stating that each bottle contains 1.5 quarts of water -/
theorem bottle_volume_is_one_and_half_quarts 
  (daily : DailyWaterConsumption) 
  (h1 : daily.bottles = 2) 
  (h2 : daily.additional_ounces = 20) :
  (weekly_ounces : ℚ) / (ounces_per_quart * days_per_week * daily.bottles : ℚ) = 3/2 := by
  sorry

end bottle_volume_is_one_and_half_quarts_l1311_131146


namespace salary_increase_percentage_l1311_131164

theorem salary_increase_percentage (S : ℝ) (h1 : S + 0.10 * S = 330) : 
  ∃ P : ℝ, S + (P / 100) * S = 348 ∧ P = 16 := by
  sorry

end salary_increase_percentage_l1311_131164


namespace cube_volume_surface_area_l1311_131113

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ s : ℝ, s > 0 ∧ s^3 = 7*x ∧ 6*s^2 = x) → x = 42 := by
  sorry

end cube_volume_surface_area_l1311_131113


namespace gcd_from_lcm_and_ratio_l1311_131178

theorem gcd_from_lcm_and_ratio (A B : ℕ) (h1 : lcm A B = 180) (h2 : A * 5 = B * 2) : 
  gcd A B = 18 := by
  sorry

end gcd_from_lcm_and_ratio_l1311_131178


namespace pure_imaginary_condition_l1311_131141

theorem pure_imaginary_condition (m : ℝ) : 
  (Complex.I * Complex.I = -1) →
  (Complex.mk (m + 2) (-1) = Complex.mk 0 (Complex.im (Complex.mk (m + 2) (-1)))) →
  m = -2 := by
  sorry

end pure_imaginary_condition_l1311_131141


namespace jerry_total_games_l1311_131170

/-- The total number of video games Jerry has after his birthday -/
def total_games (initial_games birthday_games : ℕ) : ℕ :=
  initial_games + birthday_games

/-- Theorem: Jerry has 9 video games in total -/
theorem jerry_total_games :
  total_games 7 2 = 9 := by sorry

end jerry_total_games_l1311_131170


namespace math_class_size_l1311_131158

/-- Proves that the number of students in the mathematics class is 48 given the conditions --/
theorem math_class_size (total : ℕ) (physics : ℕ) (math : ℕ) (both : ℕ) : 
  total = 56 →
  math = 4 * physics →
  both = 8 →
  total = physics + math - both →
  math = 48 := by
sorry

end math_class_size_l1311_131158


namespace polynomial_remainder_theorem_l1311_131127

theorem polynomial_remainder_theorem : ∃ q : Polynomial ℝ, 
  3 * X^3 + 2 * X^2 - 20 * X + 47 = (X - 3) * q + 86 := by
  sorry

end polynomial_remainder_theorem_l1311_131127


namespace fraction_sum_inequality_l1311_131162

theorem fraction_sum_inequality (a b c d n : ℕ) 
  (h1 : a + c < n) 
  (h2 : (a : ℚ) / b + (c : ℚ) / d < 1) : 
  (a : ℚ) / b + (c : ℚ) / d < 1 - 1 / (n^3 : ℚ) := by
  sorry

end fraction_sum_inequality_l1311_131162


namespace hyperbola_equation_l1311_131139

/-- Given a hyperbola with the general equation y²/a² - x²/b² = 1 where a > 0 and b > 0,
    an asymptote equation of 3x + 4y = 0, and a focus at (0,5),
    prove that the specific equation of the hyperbola is y²/9 - x²/16 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (asymptote : ∀ x y : ℝ, 3 * x + 4 * y = 0 → (y / x = -3 / 4 ∨ y / x = 3 / 4))
  (focus : (0 : ℝ) ^ 2 + 5 ^ 2 = (a ^ 2 + b ^ 2)) :
  ∀ x y : ℝ, y ^ 2 / 9 - x ^ 2 / 16 = 1 ↔ y ^ 2 / a ^ 2 - x ^ 2 / b ^ 2 = 1 :=
by sorry

end hyperbola_equation_l1311_131139


namespace opposite_of_2023_l1311_131130

theorem opposite_of_2023 : 
  ∀ x : ℤ, (x + 2023 = 0) ↔ (x = -2023) :=
by
  sorry

end opposite_of_2023_l1311_131130


namespace fraction_meaningful_l1311_131125

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (1 - x) / (1 + x)) ↔ x ≠ -1 := by sorry

end fraction_meaningful_l1311_131125


namespace sally_earnings_l1311_131168

/-- Sally's earnings per house, given her total earnings and number of houses cleaned -/
def earnings_per_house (total_earnings : ℕ) (houses_cleaned : ℕ) : ℚ :=
  (total_earnings : ℚ) / houses_cleaned

/-- Conversion factor from dozens to units -/
def dozens_to_units : ℕ := 12

theorem sally_earnings :
  let total_dozens : ℕ := 200
  let houses_cleaned : ℕ := 96
  earnings_per_house (total_dozens * dozens_to_units) houses_cleaned = 25 := by
sorry

end sally_earnings_l1311_131168


namespace solution_set_equality_l1311_131103

-- Define the set S
def S : Set ℝ := {x | |x + 2| + |x - 1| ≤ 4}

-- State the theorem
theorem solution_set_equality : S = Set.Icc (-5/2) (3/2) := by
  sorry

end solution_set_equality_l1311_131103


namespace star_polygon_n_value_l1311_131143

/-- A regular star polygon with n points, where each point has two angles. -/
structure StarPolygon where
  n : ℕ
  angle_C : ℝ
  angle_D : ℝ

/-- Properties of the star polygon -/
def is_valid_star_polygon (s : StarPolygon) : Prop :=
  s.n > 0 ∧
  s.angle_C > 0 ∧
  s.angle_D > 0 ∧
  s.angle_C = s.angle_D - 15 ∧
  s.n * 15 = 360

theorem star_polygon_n_value (s : StarPolygon) (h : is_valid_star_polygon s) : s.n = 24 := by
  sorry

#check star_polygon_n_value

end star_polygon_n_value_l1311_131143


namespace binomial_expansion_problem_l1311_131194

theorem binomial_expansion_problem (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (Real.sqrt 5 * x - 1)^3 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3) →
  (a₀ + a₂)^2 - (a₁ + a₃)^2 = -64 := by
  sorry

end binomial_expansion_problem_l1311_131194


namespace least_seven_digit_binary_l1311_131160

theorem least_seven_digit_binary : ∀ n : ℕ, 
  (n > 0 ∧ n < 64) → (Nat.bits n).length < 7 ∧ 
  (Nat.bits 64).length = 7 :=
by sorry

end least_seven_digit_binary_l1311_131160


namespace sum_equals_1529_l1311_131117

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The value of C in base 14 -/
def C : Nat := 12

/-- The value of D in base 14 -/
def D : Nat := 13

/-- Theorem stating that 345₁₃ + 4CD₁₄ = 1529 in base 10 -/
theorem sum_equals_1529 : 
  toBase10 [5, 4, 3] 13 + toBase10 [D, C, 4] 14 = 1529 := by sorry

end sum_equals_1529_l1311_131117


namespace max_total_profit_max_avg_annual_profit_l1311_131166

/-- The total profit function for a coach operation -/
def total_profit (x : ℕ+) : ℚ := -x^2 + 18*x - 36

/-- The average annual profit function for a coach operation -/
def avg_annual_profit (x : ℕ+) : ℚ := (total_profit x) / x

/-- Theorem stating the year of maximum total profit -/
theorem max_total_profit :
  ∃ (x : ℕ+), ∀ (y : ℕ+), total_profit x ≥ total_profit y ∧ x = 9 :=
sorry

/-- Theorem stating the year of maximum average annual profit -/
theorem max_avg_annual_profit :
  ∃ (x : ℕ+), ∀ (y : ℕ+), avg_annual_profit x ≥ avg_annual_profit y ∧ x = 6 :=
sorry

end max_total_profit_max_avg_annual_profit_l1311_131166


namespace suit_price_calculation_l1311_131140

def original_price : ℚ := 200
def increase_rate : ℚ := 0.30
def discount_rate : ℚ := 0.30
def tax_rate : ℚ := 0.07

def increased_price : ℚ := original_price * (1 + increase_rate)
def discounted_price : ℚ := increased_price * (1 - discount_rate)
def final_price : ℚ := discounted_price * (1 + tax_rate)

theorem suit_price_calculation :
  final_price = 194.74 := by sorry

end suit_price_calculation_l1311_131140


namespace negative_three_times_five_l1311_131121

theorem negative_three_times_five : (-3 : ℤ) * 5 = -15 := by
  sorry

end negative_three_times_five_l1311_131121


namespace smallest_angle_solution_l1311_131179

theorem smallest_angle_solution (x : Real) : 
  (∀ y : Real, y > 0 ∧ 8 * Real.sin y * Real.cos y^5 - 8 * Real.sin y^5 * Real.cos y = 1 → x ≤ y) ∧ 
  (x > 0 ∧ 8 * Real.sin x * Real.cos x^5 - 8 * Real.sin x^5 * Real.cos x = 1) →
  x = π / 24 := by
  sorry

end smallest_angle_solution_l1311_131179


namespace base3_to_base10_conversion_l1311_131115

/-- Converts a base-3 number represented as a list of digits to its base-10 equivalent -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The base-3 representation of the number we're considering -/
def base3Number : List Nat := [1, 2, 1, 0, 2]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Number = 178 := by
  sorry

end base3_to_base10_conversion_l1311_131115


namespace train_length_l1311_131100

/-- The length of a train given its crossing time, bridge length, and speed. -/
theorem train_length (crossing_time : ℝ) (bridge_length : ℝ) (train_speed_kmph : ℝ) :
  crossing_time = 29.997600191984642 →
  bridge_length = 200 →
  train_speed_kmph = 36 →
  ∃ (train_length : ℝ), abs (train_length - 99.976) < 0.001 := by
  sorry

end train_length_l1311_131100


namespace smallest_constant_inequality_l1311_131101

theorem smallest_constant_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x / (y + 2 * z)) + Real.sqrt (y / (2 * x + z)) + Real.sqrt (z / (x + 2 * y)) > Real.sqrt 3 := by
  sorry

end smallest_constant_inequality_l1311_131101


namespace range_equal_shifted_l1311_131148

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define the range of a function
def range (g : ℝ → ℝ) := {y : ℝ | ∃ x, g x = y}

-- Theorem statement
theorem range_equal_shifted : range f = range (fun x ↦ f (x + 1)) := by sorry

end range_equal_shifted_l1311_131148


namespace abs_inequality_solution_set_l1311_131180

theorem abs_inequality_solution_set (x : ℝ) : 
  |x + 3| - |x - 3| > 3 ↔ x > 3/2 := by
sorry

end abs_inequality_solution_set_l1311_131180


namespace jake_weight_proof_l1311_131129

/-- Jake's present weight in pounds -/
def jake_weight : ℝ := 198

/-- Kendra's weight in pounds -/
def kendra_weight : ℝ := 95

/-- The sum of Jake's and Kendra's weights -/
def total_weight : ℝ := 293

theorem jake_weight_proof :
  (jake_weight - 8 = 2 * kendra_weight) ∧
  (jake_weight + kendra_weight = total_weight) →
  jake_weight = 198 := by
sorry

end jake_weight_proof_l1311_131129


namespace equidistant_function_b_squared_l1311_131161

/-- A complex function that is equidistant from z and the origin -/
def equidistant_function (a b : ℝ) : ℂ → ℂ := fun z ↦ (a + b * Complex.I) * z

/-- The property that f(z) is equidistant from z and the origin for all z -/
def is_equidistant (f : ℂ → ℂ) : Prop :=
  ∀ z : ℂ, Complex.abs (f z - z) = Complex.abs (f z)

theorem equidistant_function_b_squared
  (a b : ℝ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_equidistant : is_equidistant (equidistant_function a b))
  (h_norm : Complex.abs (a + b * Complex.I) = 5) :
  b^2 = 99/4 := by
  sorry

end equidistant_function_b_squared_l1311_131161


namespace smallest_circular_sequence_l1311_131174

def is_valid_sequence (s : List Nat) : Prop :=
  ∀ x ∈ s, x = 1 ∨ x = 2

def contains_all_four_digit_sequences (s : List Nat) : Prop :=
  ∀ seq : List Nat, seq.length = 4 → is_valid_sequence seq →
    ∃ i, List.take 4 (List.rotateLeft s i ++ List.rotateLeft s i) = seq ∨
         List.take 4 (List.rotateRight s i ++ List.rotateRight s i) = seq

theorem smallest_circular_sequence :
  ∃ (N : Nat) (s : List Nat),
    N = s.length ∧
    is_valid_sequence s ∧
    contains_all_four_digit_sequences s ∧
    (∀ M < N, ¬∃ t : List Nat, M = t.length ∧ is_valid_sequence t ∧ contains_all_four_digit_sequences t) ∧
    N = 14 := by
  sorry

end smallest_circular_sequence_l1311_131174


namespace relationship_between_variables_l1311_131112

theorem relationship_between_variables (a b c d : ℝ) 
  (h : (3 * a + 2 * b) / (2 * b + 4 * c) = (4 * c + 3 * d) / (3 * d + 3 * a)) :
  3 * a = 4 * c ∨ 3 * a + 3 * d + 2 * b + 4 * c = 0 :=
by sorry

end relationship_between_variables_l1311_131112


namespace simplify_expression_l1311_131150

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 - 2*b + 4) - 2*b^2 = 9*b^3 - 8*b^2 + 12*b := by
  sorry

end simplify_expression_l1311_131150


namespace wood_rope_problem_l1311_131107

/-- Represents the system of equations for the wood and rope problem -/
def wood_rope_equations (x y : ℝ) : Prop :=
  (y - x = 4.5) ∧ (x - y/2 = 1)

/-- Theorem stating that the equations correctly represent the given conditions -/
theorem wood_rope_problem (x y : ℝ) :
  wood_rope_equations x y →
  (y - x = 4.5 ∧ x - y/2 = 1) :=
by
  sorry

#check wood_rope_problem

end wood_rope_problem_l1311_131107


namespace polynomial_division_remainder_l1311_131163

theorem polynomial_division_remainder (m b : ℤ) : 
  (∃ q : Polynomial ℤ, x^5 - 4*x^4 + 12*x^3 - 14*x^2 + 8*x + 5 = 
    (x^2 - 3*x + m) * q + (2*x + b)) → 
  m = 1 ∧ b = 7 := by
sorry

end polynomial_division_remainder_l1311_131163


namespace largest_constructible_cube_l1311_131188

/-- Represents the dimensions of the cardboard sheet -/
def sheet_length : ℕ := 60
def sheet_width : ℕ := 25

/-- Checks if a cube with given edge length can be constructed from the sheet -/
def can_construct_cube (edge_length : ℕ) : Prop :=
  6 * edge_length^2 ≤ sheet_length * sheet_width ∧ 
  edge_length ≤ sheet_length ∧ 
  edge_length ≤ sheet_width

/-- The largest cube edge length that can be constructed -/
def max_cube_edge : ℕ := 15

/-- Theorem stating that the largest constructible cube has edge length of 15 cm -/
theorem largest_constructible_cube :
  can_construct_cube max_cube_edge ∧
  ∀ (n : ℕ), n > max_cube_edge → ¬(can_construct_cube n) :=
by sorry

#check largest_constructible_cube

end largest_constructible_cube_l1311_131188


namespace variance_invariant_under_translation_negative_coefficient_inverse_relationship_regression_passes_through_mean_confidence_level_interpretation_l1311_131157

-- Define a dataset as a list of real numbers
def Dataset := List Real

-- Define the variance of a dataset
noncomputable def variance (D : Dataset) : Real := sorry

-- 1. Variance remains unchanged when adding a constant
theorem variance_invariant_under_translation (D : Dataset) (c : Real) :
  variance (D.map (· + c)) = variance D := sorry

-- 2. Negative coefficient in regression equation implies inverse relationship
theorem negative_coefficient_inverse_relationship (a b x : Real) (h : b < 0) :
  let y₁ := a + b * x
  let y₂ := a + b * (x + 1)
  y₂ < y₁ := sorry

-- 3. Linear regression equation passes through the mean point
theorem regression_passes_through_mean (a b : Real) (D : Dataset) :
  let x_mean := (D.sum) / D.length
  let y_mean := (D.map (λ x => a + b * x)).sum / D.length
  y_mean = a + b * x_mean := sorry

-- 4. Confidence level interpretation
theorem confidence_level_interpretation (confidence_level : Real) (h : confidence_level = 0.99) :
  ∃ (error_rate : Real), error_rate = 1 - confidence_level := sorry

end variance_invariant_under_translation_negative_coefficient_inverse_relationship_regression_passes_through_mean_confidence_level_interpretation_l1311_131157


namespace log_equation_solution_l1311_131165

theorem log_equation_solution :
  ∀ x : ℝ, (Real.log x - 3 * Real.log 4 = -3) → x = 0.064 := by
  sorry

end log_equation_solution_l1311_131165


namespace derivative_of_y_l1311_131184

-- Define the function y
def y (x : ℝ) : ℝ := 2 * x^2 - 2 * x + 1

-- State the theorem
theorem derivative_of_y (x : ℝ) : 
  deriv y x = 4 * x - 2 := by sorry

end derivative_of_y_l1311_131184


namespace average_income_calculation_l1311_131119

/-- Given the average monthly incomes of pairs of individuals and the income of one individual,
    calculate the average monthly income of a specific pair. -/
theorem average_income_calculation (P Q R : ℕ) : 
  (P + Q) / 2 = 2050 →
  (Q + R) / 2 = 5250 →
  P = 3000 →
  (P + R) / 2 = 6200 := by
sorry

end average_income_calculation_l1311_131119


namespace tan_periodic_equality_l1311_131153

theorem tan_periodic_equality (m : ℤ) : 
  -180 < m ∧ m < 180 ∧ Real.tan (m * π / 180) = Real.tan (1500 * π / 180) → m = 60 := by
  sorry

end tan_periodic_equality_l1311_131153


namespace city_population_l1311_131175

theorem city_population (known_percentage : ℝ) (known_population : ℕ) (total_population : ℕ) : 
  known_percentage = 96 / 100 →
  known_population = 23040 →
  (known_percentage * total_population : ℝ) = known_population →
  total_population = 24000 := by
  sorry

end city_population_l1311_131175


namespace abs_g_zero_eq_forty_l1311_131114

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial : Type := ℝ → ℝ

/-- The property that the absolute value of g at specific points is 10 -/
def HasSpecificValues (g : ThirdDegreePolynomial) : Prop :=
  |g (-2)| = 10 ∧ |g 1| = 10 ∧ |g 4| = 10 ∧ |g 5| = 10 ∧ |g 6| = 10 ∧ |g 9| = 10

/-- The theorem stating that if g satisfies the specific values, then |g(0)| = 40 -/
theorem abs_g_zero_eq_forty (g : ThirdDegreePolynomial) 
  (h : HasSpecificValues g) : |g 0| = 40 := by
  sorry

end abs_g_zero_eq_forty_l1311_131114


namespace a_eq_zero_necessary_not_sufficient_l1311_131135

/-- A complex number is purely imaginary if its real part is zero and its imaginary part is non-zero -/
def IsPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem a_eq_zero_necessary_not_sufficient :
  ∀ (z : ℂ), 
  (IsPurelyImaginary z → z.re = 0) ∧ 
  ∃ (z : ℂ), z.re = 0 ∧ ¬IsPurelyImaginary z :=
by sorry

end a_eq_zero_necessary_not_sufficient_l1311_131135


namespace opposite_of_2023_l1311_131197

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (n : ℤ) : ℤ := -n

/-- The opposite of 2023 is -2023. -/
theorem opposite_of_2023 : opposite 2023 = -2023 := by sorry

end opposite_of_2023_l1311_131197


namespace pages_per_day_l1311_131108

theorem pages_per_day (total_pages : ℕ) (days : ℕ) (h1 : total_pages = 612) (h2 : days = 6) :
  total_pages / days = 102 := by
  sorry

end pages_per_day_l1311_131108


namespace beautiful_association_number_part1_beautiful_association_number_part2_beautiful_association_number_part3_l1311_131159

-- Define the "beautiful association number"
def beautiful_association_number (x y a : ℚ) : ℚ :=
  |x - a| + |y - a|

-- Part 1
theorem beautiful_association_number_part1 :
  beautiful_association_number (-3) 5 2 = 8 := by sorry

-- Part 2
theorem beautiful_association_number_part2 (x : ℚ) :
  beautiful_association_number x 2 3 = 4 → x = 6 ∨ x = 0 := by sorry

-- Part 3
theorem beautiful_association_number_part3 (x₀ x₁ x₂ x₃ x₄ x₅ : ℚ) :
  beautiful_association_number x₀ x₁ 1 = 1 →
  beautiful_association_number x₁ x₂ 2 = 1 →
  beautiful_association_number x₂ x₃ 3 = 1 →
  beautiful_association_number x₃ x₄ 4 = 1 →
  beautiful_association_number x₄ x₅ 5 = 1 →
  ∃ (min : ℚ), min = 10 ∧ x₁ + x₂ + x₃ + x₄ ≥ min := by sorry

end beautiful_association_number_part1_beautiful_association_number_part2_beautiful_association_number_part3_l1311_131159


namespace probability_square_factor_l1311_131142

/-- A standard 6-sided die -/
def StandardDie : Finset Nat := {1, 2, 3, 4, 5, 6}

/-- The number of dice rolled -/
def NumDice : Nat := 6

/-- Check if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Prop := ∃ m : Nat, n = m * m

/-- The probability of rolling a product containing a square factor -/
def probabilitySquareFactor : ℚ := 665 / 729

/-- Theorem stating the probability of rolling a product containing a square factor -/
theorem probability_square_factor :
  (1 : ℚ) - (2 / 3) ^ NumDice = probabilitySquareFactor := by sorry

end probability_square_factor_l1311_131142


namespace necessary_not_sufficient_condition_l1311_131181

theorem necessary_not_sufficient_condition :
  (∃ a : ℝ, a > 0 ∧ a^2 - 2*a ≥ 0) ∧
  (∀ a : ℝ, a^2 - 2*a < 0 → a > 0) :=
by sorry

end necessary_not_sufficient_condition_l1311_131181


namespace obtuse_angle_in_second_quadrant_l1311_131134

/-- An angle is obtuse if it's greater than 90 degrees and less than 180 degrees -/
def is_obtuse_angle (α : ℝ) : Prop := 90 < α ∧ α < 180

/-- An angle is in the second quadrant if it's greater than 90 degrees and less than or equal to 180 degrees -/
def is_in_second_quadrant (α : ℝ) : Prop := 90 < α ∧ α ≤ 180

/-- Theorem: An obtuse angle is an angle in the second quadrant -/
theorem obtuse_angle_in_second_quadrant (α : ℝ) :
  is_obtuse_angle α → is_in_second_quadrant α :=
by sorry

end obtuse_angle_in_second_quadrant_l1311_131134


namespace solution_characterization_l1311_131155

def SolutionSet : Set (ℕ × ℕ × ℕ) := {(1, 1, 1), (1, 2, 1), (1, 1, 2), (1, 3, 2), (3, 5, 4), (2, 1, 1), (2, 1, 3), (4, 3, 5), (5, 4, 3), (3, 2, 1)}

def DivisibilityCondition (x y z : ℕ) : Prop :=
  (x ∣ y + 1) ∧ (y ∣ z + 1) ∧ (z ∣ x + 1)

theorem solution_characterization :
  ∀ x y z : ℕ, (x > 0 ∧ y > 0 ∧ z > 0) →
    (DivisibilityCondition x y z ↔ (x, y, z) ∈ SolutionSet) := by
  sorry

end solution_characterization_l1311_131155


namespace infinitely_many_integer_pairs_l1311_131122

theorem infinitely_many_integer_pairs : 
  ∃ (S : Set (ℤ × ℤ)), Set.Infinite S ∧ 
    ∀ (pair : ℤ × ℤ), pair ∈ S → 
      ∃ (k : ℤ), (pair.1 + 1) / pair.2 + (pair.2 + 1) / pair.1 = k :=
by sorry

end infinitely_many_integer_pairs_l1311_131122


namespace problem1_l1311_131183

theorem problem1 (a b : ℝ) (h1 : a = 1) (h2 : b = -3) :
  (a - b)^2 - 2*a*(a + 3*b) + (a + 2*b)*(a - 2*b) = -3 := by
  sorry

end problem1_l1311_131183


namespace complex_modulus_problem_l1311_131199

theorem complex_modulus_problem (z : ℂ) (a : ℝ) : 
  z = a * Complex.I → 
  (Complex.re ((1 + z) * (1 + Complex.I)) = (1 + z) * (1 + Complex.I)) → 
  Complex.abs (z + 2) = Real.sqrt 5 := by
sorry

end complex_modulus_problem_l1311_131199


namespace five_people_round_table_l1311_131147

/-- The number of unique seating arrangements for n people around a round table,
    where rotations are considered identical -/
def roundTableArrangements (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.factorial n / n

theorem five_people_round_table :
  roundTableArrangements 5 = 24 := by
  sorry

end five_people_round_table_l1311_131147


namespace inequalities_not_always_satisfied_l1311_131185

theorem inequalities_not_always_satisfied :
  ∃ (a b c x y z : ℝ), 
    x ≤ a ∧ y ≤ b ∧ z ≤ c ∧
    ((x^2 * y + y^2 * z + z^2 * x ≥ a^2 * b + b^2 * c + c^2 * a) ∨
     (x^3 + y^3 + z^3 ≥ a^3 + b^3 + c^3)) :=
by sorry

end inequalities_not_always_satisfied_l1311_131185


namespace least_reducible_fraction_l1311_131195

theorem least_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m < n → ¬(∃ k : ℕ, k > 1 ∧ k ∣ (m + 17) ∧ k ∣ (7*m - 9))) ∧
  (∃ k : ℕ, k > 1 ∧ k ∣ (n + 17) ∧ k ∣ (7*n - 9)) ∧
  n = 1 :=
sorry

end least_reducible_fraction_l1311_131195


namespace min_positive_temperatures_l1311_131138

theorem min_positive_temperatures (x y : ℕ) : 
  x * (x - 1) = 90 →
  y * (y - 1) + (10 - y) * (9 - y) = 48 →
  y ≥ 3 :=
by sorry

end min_positive_temperatures_l1311_131138


namespace roger_bike_distance_l1311_131154

def morning_distance : ℝ := 2
def evening_multiplier : ℝ := 5

theorem roger_bike_distance : 
  morning_distance + evening_multiplier * morning_distance = 12 := by
  sorry

end roger_bike_distance_l1311_131154


namespace geometric_sequence_product_relation_l1311_131149

/-- Represents a geometric sequence with 3n terms -/
structure GeometricSequence (α : Type*) [CommRing α] where
  n : ℕ
  terms : Fin (3 * n) → α
  is_geometric : ∀ i j k, i < j → j < k → terms j ^ 2 = terms i * terms k

/-- The product of n consecutive terms in a geometric sequence -/
def product_n_terms {α : Type*} [CommRing α] (seq : GeometricSequence α) (start : ℕ) : α :=
  (List.range seq.n).foldl (λ acc i => acc * seq.terms ⟨start * seq.n + i, sorry⟩) 1

/-- Theorem: In a geometric sequence with 3n terms, if A is the product of the first n terms,
    B is the product of the next n terms, and C is the product of the last n terms, then AC = B² -/
theorem geometric_sequence_product_relation {α : Type*} [CommRing α] (seq : GeometricSequence α) :
  let A := product_n_terms seq 0
  let B := product_n_terms seq 1
  let C := product_n_terms seq 2
  A * C = B ^ 2 := by
  sorry

end geometric_sequence_product_relation_l1311_131149


namespace tangent_line_cubic_function_l1311_131198

/-- Given a cubic function f(x) = ax³ - 2x passing through the point (-1, 4),
    this theorem states that the equation of the tangent line to y = f(x) at x = -1
    is 8x + y + 4 = 0. -/
theorem tangent_line_cubic_function (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 - 2*x
  f (-1) = 4 →
  let m : ℝ := (6 * (-1)^2 + 2)  -- Derivative of f at x = -1
  let tangent_line : ℝ → ℝ := λ x ↦ m * (x - (-1)) + f (-1)
  ∀ x y, y = tangent_line x ↔ 8*x + y + 4 = 0 := by
sorry


end tangent_line_cubic_function_l1311_131198


namespace snowman_height_example_l1311_131192

/-- The height of a snowman built from three vertically aligned spheres -/
def snowman_height (r1 r2 r3 : ℝ) : ℝ := 2 * (r1 + r2 + r3)

/-- Theorem: The height of a snowman with spheres of radii 10 cm, 20 cm, and 30 cm is 120 cm -/
theorem snowman_height_example : snowman_height 10 20 30 = 120 := by
  sorry

end snowman_height_example_l1311_131192


namespace negation_equivalence_l1311_131190

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
by sorry

end negation_equivalence_l1311_131190


namespace monica_books_next_year_l1311_131126

/-- The number of books Monica read last year -/
def books_last_year : ℕ := 16

/-- The number of books Monica read this year -/
def books_this_year : ℕ := 2 * books_last_year

/-- The number of books Monica will read next year -/
def books_next_year : ℕ := 2 * books_this_year + 5

/-- Theorem stating the number of books Monica will read next year -/
theorem monica_books_next_year : books_next_year = 69 := by
  sorry

end monica_books_next_year_l1311_131126


namespace fraction_equation_solution_l1311_131187

theorem fraction_equation_solution :
  ∃ x : ℚ, (x + 7) / (x - 4) = (x - 1) / (x + 6) ∧ x = -19/9 ∧ x ≠ -6 ∧ x ≠ 4 := by
  sorry

end fraction_equation_solution_l1311_131187


namespace minjoo_walked_distance_l1311_131123

-- Define the distances walked by Yongchan and Min-joo
def yongchan_distance : ℝ := 1.05
def difference : ℝ := 0.46

-- Define Min-joo's distance as a function of Yongchan's distance and the difference
def minjoo_distance : ℝ := yongchan_distance - difference

-- Theorem statement
theorem minjoo_walked_distance : minjoo_distance = 0.59 := by
  sorry

end minjoo_walked_distance_l1311_131123


namespace club_sports_theorem_l1311_131152

/-- The number of people who do not play a sport in a club -/
def people_not_playing (total : ℕ) (tennis : ℕ) (baseball : ℕ) (both : ℕ) : ℕ :=
  total - (tennis + baseball - both)

/-- Theorem: In a club with 310 people, where 138 play tennis, 255 play baseball, 
    and 94 play both sports, 11 people do not play a sport. -/
theorem club_sports_theorem : people_not_playing 310 138 255 94 = 11 := by
  sorry

end club_sports_theorem_l1311_131152


namespace smallest_divisor_with_remainder_l1311_131137

theorem smallest_divisor_with_remainder (x y z : ℕ) : 
  x > 0 ∧ y > 0 ∧ z > 0 →
  x % 9 = 2 →
  x % 7 = 4 →
  y % 13 = 12 →
  y - x = 14 →
  y % z = 3 →
  (∀ w : ℕ, w > 0 ∧ w < z ∧ y % w = 3 → False) →
  z = 22 := by
sorry

end smallest_divisor_with_remainder_l1311_131137


namespace line_circle_intersection_l1311_131193

theorem line_circle_intersection (r : ℝ) (A B : ℝ × ℝ) (h_r : r > 0) : 
  (∀ (x y : ℝ), 3*x - 4*y + 5 = 0 → x^2 + y^2 = r^2) →
  (A.1^2 + A.2^2 = r^2) →
  (B.1^2 + B.2^2 = r^2) →
  (3*A.1 - 4*A.2 + 5 = 0) →
  (3*B.1 - 4*B.2 + 5 = 0) →
  (A.1 * B.1 + A.2 * B.2 = -r^2/2) →
  r = 2 := by
sorry

end line_circle_intersection_l1311_131193


namespace f_decreasing_after_seven_fourths_l1311_131131

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x < 1 then 2 * x^2 - x + 1
  else -2 * x^2 + 7 * x - 7

-- State the theorem
theorem f_decreasing_after_seven_fourths :
  (∀ x, f (x + 1) = -f (-(x + 1))) →
  (∀ x < 1, f x = 2 * x^2 - x + 1) →
  ∀ x > (7/4 : ℝ), ∀ y > x, f y < f x :=
sorry

end f_decreasing_after_seven_fourths_l1311_131131


namespace sports_club_members_l1311_131171

theorem sports_club_members (badminton tennis both neither : ℕ) 
  (h1 : badminton = 18)
  (h2 : tennis = 19)
  (h3 : both = 9)
  (h4 : neither = 2) :
  badminton + tennis - both + neither = 30 := by
  sorry

end sports_club_members_l1311_131171


namespace cookies_per_pan_l1311_131172

theorem cookies_per_pan (total_pans : ℕ) (total_cookies : ℕ) (h1 : total_pans = 5) (h2 : total_cookies = 40) :
  total_cookies / total_pans = 8 := by
  sorry

end cookies_per_pan_l1311_131172


namespace second_crew_tractors_second_crew_is_seven_l1311_131104

/-- Calculates the number of tractors in the second crew given the farming conditions --/
theorem second_crew_tractors (total_acres : ℕ) (total_days : ℕ) (first_crew_tractors : ℕ) 
  (first_crew_days : ℕ) (second_crew_days : ℕ) (acres_per_day : ℕ) : ℕ :=
  let first_crew_acres := first_crew_tractors * first_crew_days * acres_per_day
  let remaining_acres := total_acres - first_crew_acres
  let acres_per_tractor := second_crew_days * acres_per_day
  remaining_acres / acres_per_tractor

/-- Proves that the number of tractors in the second crew is 7 --/
theorem second_crew_is_seven : 
  second_crew_tractors 1700 5 2 2 3 68 = 7 := by
  sorry

end second_crew_tractors_second_crew_is_seven_l1311_131104


namespace daytona_beach_shark_sightings_l1311_131191

theorem daytona_beach_shark_sightings :
  let cape_may_sightings : ℕ := 7
  let daytona_beach_sightings : ℕ := 3 * cape_may_sightings + 5
  daytona_beach_sightings = 26 :=
by sorry

end daytona_beach_shark_sightings_l1311_131191


namespace x_value_l1311_131128

theorem x_value (x y : ℤ) (h1 : x + y = 10) (h2 : x - y = 18) : x = 14 := by
  sorry

end x_value_l1311_131128


namespace square_and_sqrt_problem_l1311_131189

theorem square_and_sqrt_problem :
  let a : ℕ := 101
  let b : ℕ := 10101
  let c : ℕ := 102030405060504030201
  (a ^ 2 = 10201) ∧
  (b ^ 2 = 102030201) ∧
  (c = 10101010101 ^ 2) := by
  sorry

end square_and_sqrt_problem_l1311_131189


namespace train_length_proof_l1311_131102

-- Define the given conditions
def faster_train_speed : ℝ := 42
def slower_train_speed : ℝ := 36
def passing_time : ℝ := 36

-- Define the theorem
theorem train_length_proof :
  let relative_speed := faster_train_speed - slower_train_speed
  let speed_in_mps := relative_speed * (5 / 18)
  let distance := speed_in_mps * passing_time
  let train_length := distance / 2
  train_length = 30 := by sorry

end train_length_proof_l1311_131102


namespace total_sum_is_71_rupees_l1311_131118

/-- Calculates the total sum of money in rupees given the number of 20 paise and 25 paise coins -/
def total_sum_in_rupees (total_coins : ℕ) (coins_20_paise : ℕ) : ℚ :=
  let coins_25_paise := total_coins - coins_20_paise
  let value_20_paise := 20 * coins_20_paise
  let value_25_paise := 25 * coins_25_paise
  (value_20_paise + value_25_paise : ℚ) / 100

/-- Theorem stating that given 342 total coins with 290 being 20 paise coins, 
    the total sum of money is 71 rupees -/
theorem total_sum_is_71_rupees :
  total_sum_in_rupees 342 290 = 71 := by
  sorry

end total_sum_is_71_rupees_l1311_131118


namespace least_positive_integer_divisible_by_four_primes_l1311_131144

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (n : ℕ), (n > 0) ∧ 
  (∃ (p₁ p₂ p₃ p₄ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0) ∧
  (∀ (m : ℕ), m > 0 → 
    (∃ (q₁ q₂ q₃ q₄ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
      m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0) → 
    m ≥ n) ∧
  n = 210 := by
sorry

end least_positive_integer_divisible_by_four_primes_l1311_131144


namespace largest_mersenne_prime_under_500_l1311_131173

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def mersenne_prime (p : ℕ) : Prop := is_prime p ∧ ∃ n : ℕ, is_prime n ∧ p = 2^n - 1

theorem largest_mersenne_prime_under_500 :
  ∃ p : ℕ, mersenne_prime p ∧ p < 500 ∧ ∀ q : ℕ, mersenne_prime q → q < 500 → q ≤ p :=
by sorry

end largest_mersenne_prime_under_500_l1311_131173


namespace childrens_tickets_l1311_131109

theorem childrens_tickets (adult_price child_price total_tickets total_cost : ℚ) 
  (h1 : adult_price = 5.5)
  (h2 : child_price = 3.5)
  (h3 : total_tickets = 21)
  (h4 : total_cost = 83.5) :
  ∃ (adult_tickets child_tickets : ℚ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_price * adult_tickets + child_price * child_tickets = total_cost ∧
    child_tickets = 16 := by
  sorry

end childrens_tickets_l1311_131109


namespace max_cd_length_l1311_131106

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    where c = 4 and CD⊥AB, prove that the maximum value of CD is 2√3 under the given condition. -/
theorem max_cd_length (a b : ℝ) (A B C : ℝ) :
  let c : ℝ := 4
  (c * Real.cos C * Real.cos (A - B) + c = c * Real.sin C ^ 2 + b * Real.sin A * Real.sin C) →
  (∃ (D : ℝ), D ≤ 2 * Real.sqrt 3 ∧
    ∀ (E : ℝ), (c * Real.cos C * Real.cos (A - B) + c = c * Real.sin C ^ 2 + b * Real.sin A * Real.sin C) →
      E ≤ D) :=
by sorry

end max_cd_length_l1311_131106


namespace xyz_value_is_ten_l1311_131196

-- Define the variables
variable (a b c x y z : ℂ)

-- State the theorem
theorem xyz_value_is_ten
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : x ≠ 0)
  (h5 : y ≠ 0)
  (h6 : z ≠ 0)
  (h7 : a = (b + c) / (x - 3))
  (h8 : b = (a + c) / (y - 3))
  (h9 : c = (a + b) / (z - 3))
  (h10 : x * y + x * z + y * z = 9)
  (h11 : x + y + z = 6) :
  x * y * z = 10 := by
sorry


end xyz_value_is_ten_l1311_131196


namespace arjun_initial_investment_l1311_131105

/-- Represents the investment details of a partner in the business --/
structure Investment where
  amount : ℝ
  duration : ℝ

/-- Calculates the share of a partner based on their investment and duration --/
def calculateShare (inv : Investment) : ℝ :=
  inv.amount * inv.duration

/-- Proves that Arjun's initial investment was 2000 given the problem conditions --/
theorem arjun_initial_investment 
  (arjun : Investment)
  (anoop : Investment)
  (h1 : arjun.duration = 12)
  (h2 : anoop.amount = 4000)
  (h3 : anoop.duration = 6)
  (h4 : calculateShare arjun = calculateShare anoop) : 
  arjun.amount = 2000 := by
  sorry

#check arjun_initial_investment

end arjun_initial_investment_l1311_131105


namespace intersection_point_sum_l1311_131167

theorem intersection_point_sum (a b : ℚ) : 
  (∃ x y : ℚ, x = (1/4)*y + a ∧ y = (1/4)*x + b ∧ x = 1 ∧ y = 2) →
  a + b = 9/4 := by
sorry

end intersection_point_sum_l1311_131167


namespace metallic_sheet_width_l1311_131156

/-- 
Given a rectangular metallic sheet with length 48 meters, 
from which squares of side 8 meters are cut from each corner to form an open box,
if the volume of the resulting box is 5120 cubic meters,
then the width of the original metallic sheet is 36 meters.
-/
theorem metallic_sheet_width : 
  ∀ (w : ℝ), 
    (48 - 2 * 8) * (w - 2 * 8) * 8 = 5120 → 
    w = 36 := by
  sorry

end metallic_sheet_width_l1311_131156


namespace sarahs_brother_apples_l1311_131177

theorem sarahs_brother_apples (sarah_apples : ℕ) (ratio : ℕ) (brother_apples : ℕ) : 
  sarah_apples = 45 → 
  ratio = 5 → 
  sarah_apples = ratio * brother_apples → 
  brother_apples = 9 := by
sorry

end sarahs_brother_apples_l1311_131177


namespace gcd_372_684_l1311_131110

theorem gcd_372_684 : Nat.gcd 372 684 = 12 := by
  sorry

end gcd_372_684_l1311_131110


namespace bryden_received_is_ten_l1311_131186

/-- The amount a collector pays for a state quarter, as a multiple of its face value -/
def collector_rate : ℚ := 5

/-- The face value of a single state quarter in dollars -/
def quarter_value : ℚ := 1/2

/-- The number of state quarters Bryden has -/
def bryden_quarters : ℕ := 4

/-- The amount Bryden will receive from the collector in dollars -/
def bryden_received : ℚ := collector_rate * quarter_value * bryden_quarters

theorem bryden_received_is_ten : bryden_received = 10 := by
  sorry

end bryden_received_is_ten_l1311_131186


namespace double_base_exponent_equality_l1311_131124

theorem double_base_exponent_equality (a b x : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) :
  (2 * a) ^ (2 * b) = a ^ b * x ^ 3 → x = (4 ^ b * a ^ b) ^ (1/3) := by
  sorry

end double_base_exponent_equality_l1311_131124


namespace calculate_boxes_l1311_131133

/-- Given the number of blocks and blocks per box, calculate the number of boxes -/
theorem calculate_boxes (total_blocks : ℕ) (blocks_per_box : ℕ) (h : blocks_per_box > 0) :
  total_blocks / blocks_per_box = total_blocks / blocks_per_box :=
by sorry

/-- George's specific case -/
def george_boxes : ℕ :=
  let total_blocks : ℕ := 12
  let blocks_per_box : ℕ := 6
  total_blocks / blocks_per_box

#eval george_boxes

end calculate_boxes_l1311_131133


namespace coefficient_of_x_l1311_131136

theorem coefficient_of_x (a x y : ℝ) : 
  a * x + y = 19 →
  x + 3 * y = 1 →
  3 * x + 2 * y = 10 →
  a = 5 := by
sorry

end coefficient_of_x_l1311_131136


namespace melanie_marbles_l1311_131132

def sandy_marbles : ℕ := 56 * 12

theorem melanie_marbles : ∃ m : ℕ, m * 8 = sandy_marbles ∧ m = 84 := by
  sorry

end melanie_marbles_l1311_131132


namespace function_inequality_implies_k_range_l1311_131169

theorem function_inequality_implies_k_range (k : ℝ) : 
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 3, ∃ x₀ ∈ Set.Icc (-1 : ℝ) 3, 
    2 * x₁^2 + x₁ - k ≤ x₀^3 - 3 * x₀) → 
  k ≥ 3 := by
  sorry

end function_inequality_implies_k_range_l1311_131169


namespace value_of_4x_l1311_131182

theorem value_of_4x (x : ℝ) (h : 2 * x - 3 = 10) : 4 * x = 26 := by
  sorry

end value_of_4x_l1311_131182


namespace nori_crayons_left_l1311_131111

def crayons_problem (boxes : ℕ) (crayons_per_box : ℕ) (given_to_mae : ℕ) (extra_to_lea : ℕ) : ℕ :=
  let total := boxes * crayons_per_box
  let after_mae := total - given_to_mae
  let given_to_lea := given_to_mae + extra_to_lea
  after_mae - given_to_lea

theorem nori_crayons_left :
  crayons_problem 4 8 5 7 = 15 := by
  sorry

end nori_crayons_left_l1311_131111


namespace competition_results_l1311_131145

def seventh_grade_scores : List ℕ := [3, 6, 7, 6, 6, 8, 6, 9, 6, 10]
def eighth_grade_scores : List ℕ := [5, 6, 8, 7, 5, 8, 7, 9, 8, 8]

def xiao_li_score : ℕ := 7
def xiao_zhang_score : ℕ := 7

def mode (l : List ℕ) : ℕ := sorry
def average (l : List ℕ) : ℚ := sorry
def median (l : List ℕ) : ℚ := sorry

theorem competition_results :
  (mode seventh_grade_scores = 6) ∧
  (average eighth_grade_scores = 7.1) ∧
  (median seventh_grade_scores = 6) ∧
  (median eighth_grade_scores = 7.5) ∧
  (xiao_li_score > median seventh_grade_scores) ∧
  (xiao_zhang_score < median eighth_grade_scores) := by
  sorry

#check competition_results

end competition_results_l1311_131145


namespace probability_diamond_then_face_correct_l1311_131151

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the suit of a card -/
inductive Suit
| hearts | diamonds | clubs | spades

/-- Represents the rank of a card -/
inductive Rank
| two | three | four | five | six | seven | eight | nine | ten
| jack | queen | king | ace

/-- Represents a playing card -/
structure Card :=
  (suit : Suit)
  (rank : Rank)

/-- Checks if a card is a diamond -/
def is_diamond (c : Card) : Prop :=
  c.suit = Suit.diamonds

/-- Checks if a card is a face card -/
def is_face_card (c : Card) : Prop :=
  c.rank = Rank.jack ∨ c.rank = Rank.queen ∨ c.rank = Rank.king

/-- The number of diamonds in a standard deck -/
def diamond_count : Nat := 13

/-- The number of face cards in a standard deck -/
def face_card_count : Nat := 12

/-- The probability of drawing a diamond as the first card and a face card as the second card -/
def probability_diamond_then_face (d : Deck) : ℚ :=
  47 / 884

theorem probability_diamond_then_face_correct (d : Deck) :
  probability_diamond_then_face d = 47 / 884 :=
sorry

end probability_diamond_then_face_correct_l1311_131151


namespace min_wins_for_playoffs_l1311_131116

/-- 
Given a basketball league with the following conditions:
- Each game must have a winner
- A team earns 3 points for a win and loses 1 point for a loss
- The season consists of 32 games
- A team needs at least 48 points to have a chance at the playoffs

This theorem proves that a team must win at least 20 games to have a chance of advancing to the playoffs.
-/
theorem min_wins_for_playoffs (total_games : ℕ) (win_points loss_points : ℤ) (min_points : ℕ) : 
  total_games = 32 → win_points = 3 → loss_points = -1 → min_points = 48 → 
  ∃ (min_wins : ℕ), min_wins = 20 ∧ 
    ∀ (wins : ℕ), wins ≥ min_wins → 
      wins * win_points + (total_games - wins) * loss_points ≥ min_points :=
by sorry

end min_wins_for_playoffs_l1311_131116


namespace ramen_bread_intersection_l1311_131120

theorem ramen_bread_intersection (total : ℕ) (ramen : ℕ) (bread : ℕ) (neither : ℕ) 
  (h1 : total = 500)
  (h2 : ramen = 289)
  (h3 : bread = 337)
  (h4 : neither = 56) :
  ramen + bread - total + neither = 182 :=
by sorry

end ramen_bread_intersection_l1311_131120


namespace condition_relationship_l1311_131176

theorem condition_relationship : 
  (∀ x : ℝ, (0 < x ∧ x < 1) → x^2 < 1) ∧ 
  (∃ x : ℝ, x^2 < 1 ∧ ¬(0 < x ∧ x < 1)) := by
sorry

end condition_relationship_l1311_131176
