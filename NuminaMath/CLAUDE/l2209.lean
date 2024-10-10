import Mathlib

namespace find_M_l2209_220978

theorem find_M : ∃ M : ℚ, (5 + 7 + 10) / 3 = (2020 + 2021 + 2022) / M ∧ M = 827 := by
  sorry

end find_M_l2209_220978


namespace nested_sqrt_fourth_power_l2209_220912

theorem nested_sqrt_fourth_power :
  (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 1)))^4 = 3 + 2 * Real.sqrt 2 := by
  sorry

end nested_sqrt_fourth_power_l2209_220912


namespace stacked_cubes_volume_l2209_220924

/-- Calculates the total volume of stacked cubes -/
def total_volume (cube_dim : ℝ) (rows cols floors : ℕ) : ℝ :=
  (cube_dim ^ 3) * (rows * cols * floors)

/-- The problem statement -/
theorem stacked_cubes_volume :
  let cube_dim : ℝ := 1
  let rows : ℕ := 7
  let cols : ℕ := 5
  let floors : ℕ := 3
  total_volume cube_dim rows cols floors = 105 := by
  sorry

end stacked_cubes_volume_l2209_220924


namespace triangle_angle_a_value_l2209_220973

theorem triangle_angle_a_value (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a = Real.sin B + Real.cos B ∧
  a = Real.sqrt 2 ∧
  b = 2 ∧
  a / Real.sin A = b / Real.sin B ∧
  a < b →
  A = π / 6 := by
sorry

end triangle_angle_a_value_l2209_220973


namespace savings_calculation_l2209_220914

/-- Calculates savings given income and income-to-expenditure ratio --/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Theorem: Given the specified conditions, the savings is 2000 --/
theorem savings_calculation :
  let income := 18000
  let income_ratio := 9
  let expenditure_ratio := 8
  calculate_savings income income_ratio expenditure_ratio = 2000 := by
  sorry

#eval calculate_savings 18000 9 8

end savings_calculation_l2209_220914


namespace two_elements_condition_at_most_one_element_condition_l2209_220998

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - 3 * x - 4 = 0}

-- Theorem for the first part of the problem
theorem two_elements_condition (a : ℝ) :
  (∃ x y : ℝ, x ∈ A a ∧ y ∈ A a ∧ x ≠ y) ↔ (a > -9/16 ∧ a ≠ 0) :=
sorry

-- Theorem for the second part of the problem
theorem at_most_one_element_condition (a : ℝ) :
  (∀ x y : ℝ, x ∈ A a → y ∈ A a → x = y) ↔ (a ≤ -9/16 ∨ a = 0) :=
sorry

end two_elements_condition_at_most_one_element_condition_l2209_220998


namespace quadratic_inequality_problem_l2209_220910

-- Define the quadratic function
def f (a c : ℝ) (x : ℝ) := a * x^2 + x + c

-- Define the solution set condition
def solution_set (a c : ℝ) : Prop :=
  ∀ x, f a c x > 0 ↔ 1 < x ∧ x < 3

-- Define the sufficient condition
def sufficient_condition (a c m : ℝ) : Prop :=
  ∀ x, a * x^2 + 2 * x + 4 * c > 0 → x + m > 0

-- Define the not necessary condition
def not_necessary_condition (a c m : ℝ) : Prop :=
  ∃ x, x + m > 0 ∧ ¬(a * x^2 + 2 * x + 4 * c > 0)

theorem quadratic_inequality_problem (a c m : ℝ) :
  solution_set a c →
  sufficient_condition a c m →
  not_necessary_condition a c m →
  (a = -1/4 ∧ c = -3/4) ∧ (m ≥ -2) :=
by sorry

end quadratic_inequality_problem_l2209_220910


namespace ratio_product_theorem_l2209_220922

theorem ratio_product_theorem (a b c : ℝ) : 
  a / b = 3 / 4 ∧ b / c = 4 / 6 ∧ c = 18 → a * b * c = 1944 := by
  sorry

end ratio_product_theorem_l2209_220922


namespace compound_interest_calculation_l2209_220946

/-- Calculates the final amount after two years of compound interest with different rates each year -/
def final_amount (initial_amount : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let amount_after_first_year := initial_amount * (1 + rate1)
  amount_after_first_year * (1 + rate2)

/-- Theorem stating that given the initial amount and interest rates, the final amount after two years is as calculated -/
theorem compound_interest_calculation 
  (initial_amount : ℝ) 
  (rate1 : ℝ) 
  (rate2 : ℝ) 
  (h1 : initial_amount = 8736) 
  (h2 : rate1 = 0.04) 
  (h3 : rate2 = 0.05) : 
  final_amount initial_amount rate1 rate2 = 9539.712 := by
  sorry

#eval final_amount 8736 0.04 0.05

end compound_interest_calculation_l2209_220946


namespace min_value_when_a_neg_one_max_value_case1_max_value_case2_l2209_220979

-- Define the function f(x) = ax^2 + x
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x

-- Theorem for the minimum value when a = -1
theorem min_value_when_a_neg_one :
  ∀ x ∈ Set.Icc 0 2, f (-1) x ≥ -2 :=
sorry

-- Theorem for the maximum value when -2 ≤ a ≤ -1/4
theorem max_value_case1 (a : ℝ) (h : a ∈ Set.Icc (-2) (-1/4)) :
  ∀ x ∈ Set.Icc 0 2, f a x ≤ -1 / (4 * a) :=
sorry

-- Theorem for the maximum value when -1/4 < a ≤ 0
theorem max_value_case2 (a : ℝ) (h : a ∈ Set.Ioo (-1/4) 0) :
  ∀ x ∈ Set.Icc 0 2, f a x ≤ 4 * a + 2 :=
sorry

end min_value_when_a_neg_one_max_value_case1_max_value_case2_l2209_220979


namespace problem_solution_l2209_220947

def f (a x : ℝ) := |a*x - 1| - (a - 1) * |x|

theorem problem_solution :
  (∀ x : ℝ, f 2 x > 2 ↔ x < -1 ∨ x > 3) ∧
  (∀ a : ℝ, (∀ x ∈ Set.Ioo 1 2, f a x < a + 1) → a ≥ 2/5) :=
sorry

end problem_solution_l2209_220947


namespace largest_base5_to_decimal_l2209_220920

/-- Converts a base-5 digit to its decimal (base-10) value --/
def base5ToDecimal (digit : Nat) : Nat := digit

/-- Calculates the value of a base-5 digit in its positional notation --/
def digitValue (digit : Nat) (position : Nat) : Nat := 
  base5ToDecimal digit * (5 ^ position)

/-- Represents a five-digit base-5 number --/
structure FiveDigitBase5Number where
  digit1 : Nat
  digit2 : Nat
  digit3 : Nat
  digit4 : Nat
  digit5 : Nat
  all_digits_valid : digit1 < 5 ∧ digit2 < 5 ∧ digit3 < 5 ∧ digit4 < 5 ∧ digit5 < 5

/-- Converts a five-digit base-5 number to its decimal (base-10) equivalent --/
def toDecimal (n : FiveDigitBase5Number) : Nat :=
  digitValue n.digit1 4 + digitValue n.digit2 3 + digitValue n.digit3 2 + 
  digitValue n.digit4 1 + digitValue n.digit5 0

/-- The largest five-digit base-5 number --/
def largestBase5 : FiveDigitBase5Number where
  digit1 := 4
  digit2 := 4
  digit3 := 4
  digit4 := 4
  digit5 := 4
  all_digits_valid := by simp

theorem largest_base5_to_decimal : 
  toDecimal largestBase5 = 3124 := by sorry

end largest_base5_to_decimal_l2209_220920


namespace original_number_proof_l2209_220908

theorem original_number_proof : 
  ∃ N : ℕ, N ≥ 118 ∧ (N - 31) % 87 = 0 ∧ ∀ M : ℕ, M < N → (M - 31) % 87 ≠ 0 :=
by sorry

end original_number_proof_l2209_220908


namespace nines_count_to_thousand_l2209_220901

/-- Count of digit 9 appearances in a single integer -/
def count_nines (n : ℕ) : ℕ := sorry

/-- Count of digit 9 appearances in all integers from 1 to n (inclusive) -/
def total_nines (n : ℕ) : ℕ := sorry

/-- Theorem: The count of digit 9 appearances in all integers from 1 to 1000 (inclusive) is 301 -/
theorem nines_count_to_thousand : total_nines 1000 = 301 := by sorry

end nines_count_to_thousand_l2209_220901


namespace alloy_composition_proof_l2209_220966

/-- The percentage of chromium in the first alloy -/
def chromium_percent_1 : ℝ := 0.12

/-- The percentage of chromium in the second alloy -/
def chromium_percent_2 : ℝ := 0.08

/-- The amount of the first alloy used in kg -/
def amount_1 : ℝ := 15

/-- The percentage of chromium in the new alloy -/
def chromium_percent_new : ℝ := 0.092

/-- The amount of the second alloy used in kg -/
def amount_2 : ℝ := 35

theorem alloy_composition_proof :
  chromium_percent_1 * amount_1 + chromium_percent_2 * amount_2 =
  chromium_percent_new * (amount_1 + amount_2) := by
  sorry

end alloy_composition_proof_l2209_220966


namespace polynomial_identity_l2209_220930

/-- Given a polynomial P(m) that satisfies P(m) - 3m = 5m^2 - 3m - 5,
    prove that P(m) = 5m^2 - 5 -/
theorem polynomial_identity (m : ℝ) (P : ℝ → ℝ) 
    (h : ∀ m, P m - 3*m = 5*m^2 - 3*m - 5) : 
    P m = 5*m^2 - 5 := by
  sorry

end polynomial_identity_l2209_220930


namespace sqrt_two_squared_l2209_220968

theorem sqrt_two_squared : Real.sqrt 2 * Real.sqrt 2 = 2 := by
  sorry

end sqrt_two_squared_l2209_220968


namespace factor_expression_l2209_220903

theorem factor_expression (c : ℝ) : 210 * c^2 + 35 * c = 35 * c * (6 * c + 1) := by
  sorry

end factor_expression_l2209_220903


namespace isosceles_right_triangle_hypotenuse_l2209_220985

/-- An isosceles right triangle with given area and hypotenuse length -/
structure IsoscelesRightTriangle where
  -- The length of a leg
  leg : ℝ
  -- The area of the triangle
  area : ℝ
  -- The condition that the area is equal to half the square of the leg
  area_eq : area = leg^2 / 2

/-- The theorem stating that an isosceles right triangle with area 9 has hypotenuse length 6 -/
theorem isosceles_right_triangle_hypotenuse (t : IsoscelesRightTriangle) 
  (h_area : t.area = 9) : 
  t.leg * Real.sqrt 2 = 6 := by
  sorry


end isosceles_right_triangle_hypotenuse_l2209_220985


namespace unique_x_value_l2209_220954

theorem unique_x_value : ∃! (x : ℝ), x^2 ∈ ({1, 0, x} : Set ℝ) ∧ x = -1 := by
  sorry

end unique_x_value_l2209_220954


namespace f_not_prime_l2209_220940

def f (n : ℕ+) : ℤ := n.val^4 - 400 * n.val^2 + 600

theorem f_not_prime : ∀ n : ℕ+, ¬ Nat.Prime (Int.natAbs (f n)) := by
  sorry

end f_not_prime_l2209_220940


namespace infinite_good_primes_infinite_non_good_primes_l2209_220983

/-- Definition of a good prime -/
def is_good_prime (p : ℕ) : Prop :=
  Prime p ∧ ∀ a b : ℕ, a > 0 → b > 0 → (a ≡ b [ZMOD p] ↔ a^3 ≡ b^3 [ZMOD p])

/-- The set of good primes is infinite -/
theorem infinite_good_primes : Set.Infinite {p : ℕ | is_good_prime p} :=
sorry

/-- The set of non-good primes is infinite -/
theorem infinite_non_good_primes : Set.Infinite {p : ℕ | Prime p ∧ ¬is_good_prime p} :=
sorry

end infinite_good_primes_infinite_non_good_primes_l2209_220983


namespace inequality_solution_set_minimum_value_minimum_value_equality_l2209_220917

-- Problem 1
theorem inequality_solution_set (x : ℝ) :
  (2 * x + 1) / (3 - x) ≥ 1 ↔ (2/3 ≤ x ∧ x < 2) ∨ x > 2 :=
sorry

-- Problem 2
theorem minimum_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (4 / x) + (9 / y) ≥ 25 :=
sorry

theorem minimum_value_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (4 / x) + (9 / y) = 25 ↔ x = 2/5 ∧ y = 3/5 :=
sorry

end inequality_solution_set_minimum_value_minimum_value_equality_l2209_220917


namespace symmetry_theorem_l2209_220992

-- Define the points and lines
def P : ℝ × ℝ := (1, 2)
def Q : ℝ × ℝ := (3, 0)
def l1 (x y : ℝ) : Prop := x - y - 1 = 0
def l2 (x y : ℝ) : Prop := x + 3*y - 1 = 0
def l3 (x y : ℝ) : Prop := x - y + 1 = 0

-- Define symmetry for points with respect to a line
def symmetric_point (A B : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  let M := ((A.1 + B.1)/2, (A.2 + B.2)/2)
  l M.1 M.2 ∧ (B.1 - A.1) * (B.1 - A.1) + (B.2 - A.2) * (B.2 - A.2) = 
  4 * ((M.1 - A.1) * (M.1 - A.1) + (M.2 - A.2) * (M.2 - A.2))

-- Define symmetry for lines with respect to another line
def symmetric_line (l1 l2 l3 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, l1 x y → ∃ x' y' : ℝ, l2 x' y' ∧ 
  symmetric_point (x, y) (x', y') l3

-- State the theorem
theorem symmetry_theorem :
  symmetric_point P Q l1 ∧
  symmetric_line l2 (fun x y => 3*x + y + 1 = 0) l3 :=
sorry

end symmetry_theorem_l2209_220992


namespace work_rate_problem_l2209_220919

theorem work_rate_problem (x y k : ℝ) : 
  x = k * y → 
  y = 1 / 80 → 
  x + y = 1 / 20 → 
  k = 3 := by sorry

end work_rate_problem_l2209_220919


namespace river_trip_longer_than_lake_trip_l2209_220982

theorem river_trip_longer_than_lake_trip 
  (a b S : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : b < a) 
  (hS : S > 0) : 
  (2 * a * S) / (a^2 - b^2) > (2 * S) / a := by
  sorry

end river_trip_longer_than_lake_trip_l2209_220982


namespace power_of_power_l2209_220923

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l2209_220923


namespace tangent_line_at_zero_range_of_c_for_three_zeros_necessary_not_sufficient_condition_condition_not_sufficient_l2209_220964

-- Define the cubic function f(x)
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Theorem 1: Tangent line equation
theorem tangent_line_at_zero (a b c : ℝ) :
  ∃ m k, ∀ x, m*x + k = f a b c x + (f a b c 0 - f a b c x) / x :=
sorry

-- Theorem 2: Range of c when a = b = 4 and f has three distinct zeros
theorem range_of_c_for_three_zeros :
  ∃ c, 0 < c ∧ c < 32/27 ∧
  (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f 4 4 c x = 0 ∧ f 4 4 c y = 0 ∧ f 4 4 c z = 0) :=
sorry

-- Theorem 3: Necessary but not sufficient condition for three distinct zeros
theorem necessary_not_sufficient_condition (a b c : ℝ) :
  (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a b c x = 0 ∧ f a b c y = 0 ∧ f a b c z = 0) →
  a^2 - 3*b > 0 :=
sorry

theorem condition_not_sufficient :
  ∃ a b c, a^2 - 3*b > 0 ∧
  ¬(∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a b c x = 0 ∧ f a b c y = 0 ∧ f a b c z = 0) :=
sorry

end tangent_line_at_zero_range_of_c_for_three_zeros_necessary_not_sufficient_condition_condition_not_sufficient_l2209_220964


namespace multiplicative_magic_square_exists_l2209_220990

/-- Represents a 3x3 matrix --/
def Matrix3x3 := Fin 3 → Fin 3 → ℕ

/-- Check if two numbers have the same digits --/
def same_digits (a b : ℕ) : Prop := sorry

/-- The original magic square --/
def original_square : Matrix3x3 := 
  fun i j => match i, j with
  | 0, 0 => 27 | 0, 1 => 20 | 0, 2 => 25
  | 1, 0 => 22 | 1, 1 => 24 | 1, 2 => 26
  | 2, 0 => 23 | 2, 1 => 28 | 2, 2 => 21

/-- Product of elements in a row --/
def row_product (m : Matrix3x3) (i : Fin 3) : ℕ :=
  (m i 0) * (m i 1) * (m i 2)

/-- Product of elements in a column --/
def col_product (m : Matrix3x3) (j : Fin 3) : ℕ :=
  (m 0 j) * (m 1 j) * (m 2 j)

/-- Product of elements in the main diagonal --/
def diag_product (m : Matrix3x3) : ℕ :=
  (m 0 0) * (m 1 1) * (m 2 2)

/-- Product of elements in the anti-diagonal --/
def antidiag_product (m : Matrix3x3) : ℕ :=
  (m 0 2) * (m 1 1) * (m 2 0)

/-- The theorem to be proved --/
theorem multiplicative_magic_square_exists : ∃ (m : Matrix3x3), 
  (∀ i j, same_digits (m i j) (original_square i j)) ∧ 
  (∀ i : Fin 3, row_product m i = 7488) ∧
  (∀ j : Fin 3, col_product m j = 7488) ∧
  (diag_product m = 7488) ∧
  (antidiag_product m = 7488) := by
  sorry

end multiplicative_magic_square_exists_l2209_220990


namespace product_seven_reciprocal_squares_sum_l2209_220975

theorem product_seven_reciprocal_squares_sum (a b : ℕ) (h : a * b = 7) :
  (1 : ℚ) / (a^2 : ℚ) + (1 : ℚ) / (b^2 : ℚ) = 50 / 49 := by
  sorry

end product_seven_reciprocal_squares_sum_l2209_220975


namespace similar_triangles_collinearity_l2209_220995

/-- Two triangles are similar if they have the same shape but possibly different size and orientation -/
def similar_triangles (t1 t2 : Set (Fin 3 → ℝ × ℝ)) : Prop := sorry

/-- Two triangles are differently oriented if they cannot be made to coincide by translation and scaling -/
def differently_oriented (t1 t2 : Set (Fin 3 → ℝ × ℝ)) : Prop := sorry

/-- A point divides a segment in a given ratio -/
def divides_segment_in_ratio (A A' A₁ : ℝ × ℝ) (r : ℝ) : Prop :=
  ∃ t : ℝ, A' = (1 - t) • A + t • A₁ ∧ r = t / (1 - t)

/-- Three points are collinear if they lie on a single straight line -/
def collinear (A B C : ℝ × ℝ) : Prop := sorry

theorem similar_triangles_collinearity 
  (A B C A₁ B₁ C₁ A' B' C' : ℝ × ℝ) 
  (ABC : Set (Fin 3 → ℝ × ℝ)) 
  (A₁B₁C₁ : Set (Fin 3 → ℝ × ℝ)) 
  (h_similar : similar_triangles ABC A₁B₁C₁)
  (h_oriented : differently_oriented ABC A₁B₁C₁)
  (h_A' : divides_segment_in_ratio A A' A₁ (dist B C / dist B₁ C₁))
  (h_B' : divides_segment_in_ratio B B' B₁ (dist B C / dist B₁ C₁))
  (h_C' : divides_segment_in_ratio C C' C₁ (dist B C / dist B₁ C₁)) :
  collinear A' B' C' := by sorry

end similar_triangles_collinearity_l2209_220995


namespace bread_roll_flour_usage_l2209_220980

theorem bread_roll_flour_usage
  (original_rolls : ℕ) (original_flour_per_roll : ℚ)
  (new_rolls : ℕ) (new_flour_per_roll : ℚ)
  (h1 : original_rolls = 24)
  (h2 : original_flour_per_roll = 1 / 8)
  (h3 : new_rolls = 16)
  (h4 : original_rolls * original_flour_per_roll = new_rolls * new_flour_per_roll) :
  new_flour_per_roll = 3 / 16 := by
sorry

end bread_roll_flour_usage_l2209_220980


namespace negation_equivalence_l2209_220962

theorem negation_equivalence :
  ¬(∃ x : ℝ, x > 1 ∧ x^2 - x > 0) ↔ (∀ x : ℝ, x > 1 → x^2 - x ≤ 0) :=
by sorry

end negation_equivalence_l2209_220962


namespace anna_gets_more_candy_l2209_220972

/-- Calculates the difference in candy pieces between Anna and Billy --/
def candy_difference (anna_per_house billy_per_house anna_houses billy_houses : ℕ) : ℕ :=
  anna_per_house * anna_houses - billy_per_house * billy_houses

/-- Proves that Anna gets 15 more pieces of candy than Billy --/
theorem anna_gets_more_candy : 
  candy_difference 14 11 60 75 = 15 := by
  sorry

end anna_gets_more_candy_l2209_220972


namespace walters_sticker_distribution_l2209_220933

/-- Miss Walter's sticker distribution problem -/
theorem walters_sticker_distribution 
  (gold : ℕ) 
  (silver : ℕ) 
  (bronze : ℕ) 
  (students : ℕ) 
  (h1 : gold = 50)
  (h2 : silver = 2 * gold)
  (h3 : bronze = silver - 20)
  (h4 : students = 5) :
  (gold + silver + bronze) / students = 46 := by
  sorry

end walters_sticker_distribution_l2209_220933


namespace specific_frustum_lateral_surface_area_l2209_220958

/-- The lateral surface area of a frustum of a cone --/
def lateralSurfaceArea (slantHeight : ℝ) (radiusRatio : ℝ) (centralAngle : ℝ) : ℝ :=
  sorry

/-- Theorem: The lateral surface area of a specific frustum of a cone --/
theorem specific_frustum_lateral_surface_area :
  lateralSurfaceArea 10 (2/5) 216 = 252 * Real.pi / 5 := by
  sorry

end specific_frustum_lateral_surface_area_l2209_220958


namespace sum_of_median_scores_l2209_220986

def median_score_A : ℕ := 28
def median_score_B : ℕ := 36

theorem sum_of_median_scores :
  median_score_A + median_score_B = 64 := by
  sorry

end sum_of_median_scores_l2209_220986


namespace cubic_root_sum_l2209_220935

theorem cubic_root_sum (a b c d : ℝ) (h1 : a ≠ 0) 
  (h2 : a * 4^3 + b * 4^2 + c * 4 + d = 0)
  (h3 : a * 1^3 + b * 1^2 + c * 1 + d = 0)
  (h4 : a * (-3)^3 + b * (-3)^2 + c * (-3) + d = 0) :
  (b + c) / a = 49 / 3 := by
  sorry

end cubic_root_sum_l2209_220935


namespace jean_kept_fraction_l2209_220977

theorem jean_kept_fraction (total : ℕ) (janet_got : ℕ) (janet_fraction : ℚ) :
  total = 60 →
  janet_got = 10 →
  janet_fraction = 1/4 →
  (total - (janet_got / janet_fraction)) / total = 1/3 := by
  sorry

end jean_kept_fraction_l2209_220977


namespace cone_height_ratio_l2209_220941

/-- Theorem about the ratio of heights in a cone with reduced height --/
theorem cone_height_ratio (original_height : ℝ) (base_circumference : ℝ) (new_volume : ℝ) :
  original_height = 20 →
  base_circumference = 18 * Real.pi →
  new_volume = 270 * Real.pi →
  ∃ (new_height : ℝ),
    (1 / 3 : ℝ) * Real.pi * (base_circumference / (2 * Real.pi))^2 * new_height = new_volume ∧
    new_height / original_height = 1 / 2 := by
  sorry

end cone_height_ratio_l2209_220941


namespace expression_value_l2209_220960

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 2) : 3 * x - 4 * y + 2 = 3 := by
  sorry

end expression_value_l2209_220960


namespace shift_left_3_units_l2209_220949

-- Define the original function
def f (x : ℝ) : ℝ := (x - 1)^2

-- Define the shifted function
def g (x : ℝ) : ℝ := (x + 2)^2

-- Define the shift operation
def shift (h : ℝ → ℝ) (s : ℝ) : ℝ → ℝ := fun x ↦ h (x + s)

-- Theorem statement
theorem shift_left_3_units :
  shift f 3 = g := by sorry

end shift_left_3_units_l2209_220949


namespace library_comic_books_l2209_220961

theorem library_comic_books (fairy_tale_books : ℕ) (science_tech_books : ℕ) (comic_books : ℕ) : 
  fairy_tale_books = 305 →
  science_tech_books = fairy_tale_books + 115 →
  comic_books = 4 * (fairy_tale_books + science_tech_books) →
  comic_books = 2900 := by
sorry

end library_comic_books_l2209_220961


namespace N_minus_M_eq_six_l2209_220974

def M : Set ℕ := {1, 2, 3, 5}
def N : Set ℕ := {2, 3, 6}

theorem N_minus_M_eq_six : N \ M = {6} := by sorry

end N_minus_M_eq_six_l2209_220974


namespace cookies_left_l2209_220916

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens John bought -/
def dozens_bought : ℕ := 2

/-- The number of cookies John ate -/
def cookies_eaten : ℕ := 3

/-- Theorem: John has 21 cookies left -/
theorem cookies_left : dozens_bought * dozen - cookies_eaten = 21 := by
  sorry

end cookies_left_l2209_220916


namespace student_permutations_l2209_220965

/-- Represents the number of students --/
def n : ℕ := 5

/-- The factorial function --/
def factorial (m : ℕ) : ℕ := 
  match m with
  | 0 => 1
  | m + 1 => (m + 1) * factorial m

/-- The number of permutations of n elements --/
def permutations (n : ℕ) : ℕ := factorial n

/-- The number of permutations not in alphabetical order --/
def permutations_not_alphabetical (n : ℕ) : ℕ := permutations n - 1

/-- The number of permutations where two specific elements are consecutive --/
def permutations_consecutive_pair (n : ℕ) : ℕ := 2 * factorial (n - 1)

theorem student_permutations :
  (permutations n = 120) ∧
  (permutations_not_alphabetical n = 119) ∧
  (permutations_consecutive_pair n = 48) := by
  sorry

end student_permutations_l2209_220965


namespace library_loan_availability_l2209_220904

-- Define the universe of books in the library
variable (Book : Type)

-- Define a predicate for books available for loan
variable (available_for_loan : Book → Prop)

-- Theorem statement
theorem library_loan_availability (h : ¬∀ (b : Book), available_for_loan b) :
  (∃ (b : Book), ¬available_for_loan b) ∧ (¬∀ (b : Book), available_for_loan b) :=
by sorry

end library_loan_availability_l2209_220904


namespace square_rotation_cylinder_volume_l2209_220944

/-- The volume of a cylinder formed by rotating a square about its horizontal line of symmetry -/
theorem square_rotation_cylinder_volume (side_length : ℝ) (volume : ℝ) :
  side_length = 10 →
  volume = Real.pi * (side_length / 2)^2 * side_length →
  volume = 250 * Real.pi :=
by
  sorry

#check square_rotation_cylinder_volume

end square_rotation_cylinder_volume_l2209_220944


namespace line_not_in_fourth_quadrant_l2209_220945

/-- Given a line ax + by + c = 0 where ac > 0 and bc < 0, 
    the line does not pass through the fourth quadrant -/
theorem line_not_in_fourth_quadrant 
  (a b c : ℝ) 
  (h1 : a * c > 0) 
  (h2 : b * c < 0) : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ a * x + b * y + c ≠ 0 := by
sorry

end line_not_in_fourth_quadrant_l2209_220945


namespace student_survey_l2209_220969

theorem student_survey (total : ℕ) (mac_preference : ℕ) (both_preference : ℕ) :
  total = 210 →
  mac_preference = 60 →
  both_preference = mac_preference / 3 →
  total - (mac_preference + both_preference) = 130 := by
  sorry

end student_survey_l2209_220969


namespace laundry_detergent_problem_l2209_220959

def standard_weight : ℕ := 450
def price_per_bag : ℕ := 3
def weight_deviations : List ℤ := [-6, -3, -2, 0, 1, 4, 5, -1]
def qualification_criterion : ℤ → Bool := λ x => x.natAbs ≤ 4

theorem laundry_detergent_problem :
  let total_weight := (weight_deviations.sum + standard_weight * weight_deviations.length : ℤ)
  let qualified_bags := weight_deviations.filter qualification_criterion
  let total_sales := qualified_bags.length * price_per_bag
  (total_weight = 3598 ∧ total_sales = 18) := by sorry

end laundry_detergent_problem_l2209_220959


namespace rational_function_value_l2209_220952

theorem rational_function_value (x : ℝ) (h : x ≠ 5) :
  x = 4 → (x^2 - 3*x - 10) / (x - 5) = 6 := by
  sorry

end rational_function_value_l2209_220952


namespace base5_412_equals_base7_212_l2209_220963

/-- Converts a base-5 number to decimal --/
def base5ToDecimal (n : Nat) : Nat :=
  sorry

/-- Converts a decimal number to base-7 --/
def decimalToBase7 (n : Nat) : Nat :=
  sorry

/-- Theorem stating that 412₅ is equal to 212₇ --/
theorem base5_412_equals_base7_212 :
  decimalToBase7 (base5ToDecimal 412) = 212 :=
sorry

end base5_412_equals_base7_212_l2209_220963


namespace smallest_multiple_l2209_220994

theorem smallest_multiple (n : ℕ) : n = 1767 ↔ 
  n > 0 ∧ 
  31 ∣ n ∧ 
  n % 97 = 3 ∧ 
  ∀ m : ℕ, m > 0 → 31 ∣ m → m % 97 = 3 → n ≤ m :=
by sorry

end smallest_multiple_l2209_220994


namespace rectangle_cylinder_volume_ratio_l2209_220950

/-- Given a rectangle with dimensions 6 x 9, prove that the ratio of the volume of the larger cylinder
    to the volume of the smaller cylinder formed by rolling the rectangle is 3/2. -/
theorem rectangle_cylinder_volume_ratio :
  let width : ℝ := 6
  let length : ℝ := 9
  let volume1 : ℝ := π * (width / (2 * π))^2 * length
  let volume2 : ℝ := π * (length / (2 * π))^2 * width
  volume2 / volume1 = 3 / 2 := by sorry

end rectangle_cylinder_volume_ratio_l2209_220950


namespace angle_negative_1120_in_fourth_quadrant_l2209_220996

def angle_to_standard_form (angle : ℤ) : ℤ :=
  angle % 360

def quadrant (angle : ℤ) : ℕ :=
  let standard_angle := angle_to_standard_form angle
  if 0 ≤ standard_angle ∧ standard_angle < 90 then 1
  else if 90 ≤ standard_angle ∧ standard_angle < 180 then 2
  else if 180 ≤ standard_angle ∧ standard_angle < 270 then 3
  else 4

theorem angle_negative_1120_in_fourth_quadrant :
  quadrant (-1120) = 4 := by
  sorry

end angle_negative_1120_in_fourth_quadrant_l2209_220996


namespace ranch_minimum_animals_l2209_220939

theorem ranch_minimum_animals : ∀ (ponies horses : ℕ),
  ponies > 0 →
  horses = ponies + 4 →
  (3 * ponies) % 10 = 0 →
  (5 * ((3 * ponies) / 10)) % 8 = 0 →
  ponies + horses ≥ 36 :=
by
  sorry

end ranch_minimum_animals_l2209_220939


namespace possible_values_of_b_over_a_l2209_220951

theorem possible_values_of_b_over_a (a b : ℝ) (h : a > 0) :
  (∀ a b, a > 0 → Real.log a + b - a * Real.exp (b - 1) ≥ 0) →
  (b / a = Real.exp (-1) ∨ b / a = Real.exp (-2) ∨ b / a = -Real.exp (-2)) :=
sorry

end possible_values_of_b_over_a_l2209_220951


namespace starting_lineup_combinations_l2209_220991

/-- The number of players on the team -/
def total_players : ℕ := 15

/-- The number of players in the starting lineup -/
def lineup_size : ℕ := 6

/-- The number of pre-selected players (All-Stars) -/
def preselected_players : ℕ := 3

/-- The number of different starting lineups possible -/
def num_lineups : ℕ := 220

theorem starting_lineup_combinations :
  Nat.choose (total_players - preselected_players) (lineup_size - preselected_players) = num_lineups :=
sorry

end starting_lineup_combinations_l2209_220991


namespace earth_moon_distance_in_scientific_notation_l2209_220999

/-- The distance from Earth to the Moon's surface in kilometers -/
def earth_moon_distance : ℝ := 383900

/-- The scientific notation representation of the Earth-Moon distance -/
def earth_moon_distance_scientific : ℝ := 3.839 * (10 ^ 5)

theorem earth_moon_distance_in_scientific_notation :
  earth_moon_distance = earth_moon_distance_scientific :=
sorry

end earth_moon_distance_in_scientific_notation_l2209_220999


namespace endpoint_sum_coordinates_l2209_220988

/-- Given a line segment with one endpoint (6, -2) and midpoint (5, 5),
    the sum of coordinates of the other endpoint is 16. -/
theorem endpoint_sum_coordinates (x y : ℝ) : 
  (6 + x) / 2 = 5 ∧ (-2 + y) / 2 = 5 → x + y = 16 := by
  sorry

end endpoint_sum_coordinates_l2209_220988


namespace min_operations_to_exceed_1000_l2209_220938

-- Define the operation of repeated squaring
def repeated_square (x : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => (repeated_square x n) ^ 2

-- State the theorem
theorem min_operations_to_exceed_1000 :
  (∃ n : ℕ, repeated_square 3 n > 1000) ∧
  (∀ m : ℕ, repeated_square 3 m > 1000 → m ≥ 3) ∧
  (repeated_square 3 3 > 1000) :=
sorry

end min_operations_to_exceed_1000_l2209_220938


namespace b_parallel_same_direction_as_a_l2209_220937

/-- Two vectors are parallel and in the same direction if one is a positive scalar multiple of the other -/
def parallel_same_direction (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ b = (k * a.1, k * a.2)

/-- Given vector a -/
def a : ℝ × ℝ := (1, -1)

/-- Vector b to be proven parallel and in the same direction as a -/
def b : ℝ × ℝ := (2, -2)

/-- Theorem stating that b is parallel and in the same direction as a -/
theorem b_parallel_same_direction_as_a : parallel_same_direction a b := by
  sorry

end b_parallel_same_direction_as_a_l2209_220937


namespace unique_triple_solution_l2209_220906

theorem unique_triple_solution :
  ∃! (a b c : ℕ), b > 1 ∧ 2^c + 2^2016 = a^b ∧ a = 3 * 2^1008 ∧ b = 2 ∧ c = 2019 := by
  sorry

end unique_triple_solution_l2209_220906


namespace gcd_nine_factorial_six_factorial_squared_l2209_220970

theorem gcd_nine_factorial_six_factorial_squared : 
  Nat.gcd (Nat.factorial 9) ((Nat.factorial 6)^2) = 43200 := by
  sorry

end gcd_nine_factorial_six_factorial_squared_l2209_220970


namespace square_overlap_areas_l2209_220925

/-- Given a square with side length 3 cm cut along its diagonal, prove the areas of overlap in specific arrangements --/
theorem square_overlap_areas :
  let square_side : ℝ := 3
  let small_square_side : ℝ := 1
  let triangle_area : ℝ := square_side ^ 2 / 2
  let small_square_area : ℝ := small_square_side ^ 2
  
  -- Area of overlap when a 1 cm × 1 cm square is placed inside one of the resulting triangles
  let overlap_area_b : ℝ := small_square_area / 4
  
  -- Area of overlap when the two triangles are arranged to form a rectangle of 1 cm × 3 cm with an additional overlap
  let overlap_area_c : ℝ := triangle_area / 2
  
  (overlap_area_b = 0.25 ∧ overlap_area_c = 2.25) := by
  sorry


end square_overlap_areas_l2209_220925


namespace article_cost_price_l2209_220948

theorem article_cost_price (original_selling_price original_cost_price new_selling_price new_cost_price : ℝ) :
  original_selling_price = 1.25 * original_cost_price →
  new_cost_price = 0.8 * original_cost_price →
  new_selling_price = original_selling_price - 12.60 →
  new_selling_price = 1.3 * new_cost_price →
  original_cost_price = 60 := by
sorry

end article_cost_price_l2209_220948


namespace equation_one_solutions_equation_two_solutions_equation_three_no_real_roots_equation_four_solutions_l2209_220993

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  (x + 1)^2 - 5 = 0 ↔ x = Real.sqrt 5 - 1 ∨ x = -Real.sqrt 5 - 1 := by sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) :
  2 * x^2 - 4 * x + 1 = 0 ↔ x = (4 + Real.sqrt 8) / 4 ∨ x = (4 - Real.sqrt 8) / 4 := by sorry

-- Equation 3
theorem equation_three_no_real_roots :
  ¬∃ (x : ℝ), (2 * x + 1) * (x - 3) = -7 := by sorry

-- Equation 4
theorem equation_four_solutions (x : ℝ) :
  3 * (x - 2)^2 = x * (x - 2) ↔ x = 2 ∨ x = 3 := by sorry

end equation_one_solutions_equation_two_solutions_equation_three_no_real_roots_equation_four_solutions_l2209_220993


namespace courtyard_width_l2209_220989

/-- Proves that the width of a rectangular courtyard is 18 meters -/
theorem courtyard_width (length : ℝ) (brick_length : ℝ) (brick_width : ℝ) (total_bricks : ℕ) :
  length = 25 →
  brick_length = 0.2 →
  brick_width = 0.1 →
  total_bricks = 22500 →
  (length * (total_bricks : ℝ) * brick_length * brick_width) / length = 18 :=
by sorry

end courtyard_width_l2209_220989


namespace farm_animals_l2209_220921

theorem farm_animals (total_heads : ℕ) (total_feet : ℕ) (hen_heads : ℕ) (hen_feet : ℕ) (cow_heads : ℕ) (cow_feet : ℕ) : 
  total_heads = 60 →
  total_feet = 200 →
  hen_heads = 1 →
  hen_feet = 2 →
  cow_heads = 1 →
  cow_feet = 4 →
  ∃ (num_hens : ℕ) (num_cows : ℕ),
    num_hens + num_cows = total_heads ∧
    num_hens * hen_feet + num_cows * cow_feet = total_feet ∧
    num_hens = 20 :=
by sorry

end farm_animals_l2209_220921


namespace composite_function_problem_l2209_220902

-- Definition of composite function for linear functions
def composite_function (k₁ b₁ k₂ b₂ : ℝ) : ℝ → ℝ :=
  λ x => (k₁ + k₂) * x + b₁ * b₂

theorem composite_function_problem :
  -- 1. Composite of y=3x+2 and y=-4x+3
  (∀ x, composite_function 3 2 (-4) 3 x = -x + 6) ∧
  -- 2. If composite of y=ax-2 and y=-x+b is y=3x+2, then a=4 and b=-1
  (∀ a b, (∀ x, composite_function a (-2) (-1) b x = 3 * x + 2) → a = 4 ∧ b = -1) ∧
  -- 3. Conditions for passing through first, second, and fourth quadrants
  (∀ k b, (∀ x, (composite_function (-1) b k (-3) x > 0 ∧ x > 0) ∨
                (composite_function (-1) b k (-3) x < 0 ∧ x > 0) ∨
                (composite_function (-1) b k (-3) x > 0 ∧ x < 0)) →
    k < 1 ∧ b < 0) ∧
  -- 4. Fixed point of composite of y=-2x+m and y=3mx-6
  (∀ m, composite_function (-2) m (3*m) (-6) 2 = -4) := by
  sorry

end composite_function_problem_l2209_220902


namespace set_equality_implies_a_equals_one_l2209_220900

/-- Given two sets A and B with specific elements, prove that if A = B, then a = 1 -/
theorem set_equality_implies_a_equals_one (a : ℝ) : 
  let A : Set ℝ := {1, -2, a^2 - 1}
  let B : Set ℝ := {1, a^2 - 3*a, 0}
  A = B → a = 1 := by
  sorry

end set_equality_implies_a_equals_one_l2209_220900


namespace molecular_weight_Al2S3_l2209_220934

/-- The atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Sulfur in g/mol -/
def atomic_weight_S : ℝ := 32.06

/-- The number of Aluminum atoms in Al2S3 -/
def num_Al : ℕ := 2

/-- The number of Sulfur atoms in Al2S3 -/
def num_S : ℕ := 3

/-- The number of moles of Al2S3 -/
def num_moles : ℝ := 10

/-- Theorem: The molecular weight of 10 moles of Al2S3 is 1501.4 grams -/
theorem molecular_weight_Al2S3 : 
  num_moles * (num_Al * atomic_weight_Al + num_S * atomic_weight_S) = 1501.4 := by
  sorry


end molecular_weight_Al2S3_l2209_220934


namespace ellipse_range_theorem_l2209_220936

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / (16/3) = 1

/-- Point M -/
def M : ℝ × ℝ := (0, 2)

/-- Dot product of two 2D vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

/-- The expression OP · OQ + MP · MQ -/
def expr (P Q : ℝ × ℝ) : ℝ :=
  dot_product P Q + dot_product (P.1 - M.1, P.2 - M.2) (Q.1 - M.1, Q.2 - M.2)

/-- The theorem to be proved -/
theorem ellipse_range_theorem :
  ∀ P Q : ℝ × ℝ,
  is_on_ellipse P.1 P.2 →
  is_on_ellipse Q.1 Q.2 →
  ∃ k : ℝ, P.2 - M.2 = k * (P.1 - M.1) ∧ Q.2 - M.2 = k * (Q.1 - M.1) →
  -20 ≤ expr P Q ∧ expr P Q ≤ -52/3 :=
sorry

end ellipse_range_theorem_l2209_220936


namespace wire_cutting_problem_l2209_220984

theorem wire_cutting_problem (shorter_piece longer_piece total_length : ℝ) :
  shorter_piece = 40 →
  shorter_piece = (2 / 5) * longer_piece →
  total_length = shorter_piece + longer_piece →
  total_length = 140 := by
sorry

end wire_cutting_problem_l2209_220984


namespace boys_in_class_l2209_220942

theorem boys_in_class (total : ℕ) (girls_ratio : ℕ) (boys_ratio : ℕ) (boys : ℕ) : 
  total = 35 →
  girls_ratio = 4 →
  boys_ratio = 3 →
  girls_ratio * boys = boys_ratio * (total - boys) →
  boys = 15 := by
sorry

end boys_in_class_l2209_220942


namespace function_characterization_l2209_220913

theorem function_characterization (f : ℕ+ → ℕ+ → ℕ+) :
  (∀ x : ℕ+, f x x = x) →
  (∀ x y : ℕ+, f x y = f y x) →
  (∀ x y : ℕ+, (x + y) * (f x y) = y * (f x (x + y))) →
  (∀ x y : ℕ+, f x y = Nat.lcm x y) :=
by sorry

end function_characterization_l2209_220913


namespace alabama_theorem_l2209_220926

/-- The number of letters in the word "ALABAMA" -/
def total_letters : ℕ := 7

/-- The number of 'A's in the word "ALABAMA" -/
def number_of_as : ℕ := 4

/-- The number of unique arrangements of the letters in "ALABAMA" -/
def alabama_arrangements : ℕ := total_letters.factorial / number_of_as.factorial

theorem alabama_theorem : alabama_arrangements = 210 := by
  sorry

end alabama_theorem_l2209_220926


namespace yellow_teams_count_l2209_220905

theorem yellow_teams_count (blue_students yellow_students total_students total_teams blue_teams : ℕ)
  (h1 : blue_students = 70)
  (h2 : yellow_students = 84)
  (h3 : total_students = blue_students + yellow_students)
  (h4 : total_teams = 77)
  (h5 : total_students = 2 * total_teams)
  (h6 : blue_teams = 30) :
  ∃ yellow_teams : ℕ, yellow_teams = 37 ∧ 
    yellow_teams = total_teams - blue_teams - (blue_students + yellow_students - 2 * blue_teams) / 2 :=
by sorry

end yellow_teams_count_l2209_220905


namespace border_area_calculation_l2209_220997

/-- Given a rectangular photograph and its frame, calculate the area of the border. -/
theorem border_area_calculation (photo_height photo_width border_width : ℝ) 
  (h1 : photo_height = 12)
  (h2 : photo_width = 15)
  (h3 : border_width = 3) :
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - 
  photo_height * photo_width = 198 := by
  sorry

#check border_area_calculation

end border_area_calculation_l2209_220997


namespace race_cars_l2209_220955

theorem race_cars (p_x p_y p_z p_total : ℚ) : 
  p_x = 1/8 → p_y = 1/12 → p_z = 1/6 → p_total = 375/1000 → 
  p_x + p_y + p_z = p_total → 
  ∀ p_other : ℚ, p_other ≥ 0 → p_x + p_y + p_z + p_other = p_total → p_other = 0 := by
  sorry

end race_cars_l2209_220955


namespace negation_equivalence_l2209_220931

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ ≥ 2 ∧ x₀^2 - 2*x₀ - 2 > 0) ↔ 
  (∀ x : ℝ, x ≥ 2 → x^2 - 2*x - 2 ≤ 0) :=
by sorry

end negation_equivalence_l2209_220931


namespace unique_k_value_l2209_220927

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- The quadratic equation x^2 - 63x + k = 0 with prime roots -/
def quadratic_equation (k : ℕ) (x : ℝ) : Prop :=
  x^2 - 63*x + k = 0

/-- The roots of the quadratic equation are prime numbers -/
def roots_are_prime (k : ℕ) : Prop :=
  ∃ (a b : ℕ), (is_prime a ∧ is_prime b) ∧
  (∀ x : ℝ, quadratic_equation k x ↔ (x = a ∨ x = b))

theorem unique_k_value : ∃! k : ℕ, roots_are_prime k ∧ k = 122 :=
sorry

end unique_k_value_l2209_220927


namespace peters_fish_catch_l2209_220956

theorem peters_fish_catch (n : ℕ) : (3 * n = n + 24) → n = 12 := by
  sorry

end peters_fish_catch_l2209_220956


namespace shelby_rain_time_l2209_220953

/-- Represents the driving scenario for Shelby -/
structure DrivingScenario where
  speed_sun : ℝ  -- Speed when not raining (miles per hour)
  speed_rain : ℝ  -- Speed when raining (miles per hour)
  total_distance : ℝ  -- Total distance driven (miles)
  total_time : ℝ  -- Total time driven (minutes)

/-- Calculates the time driven in rain given a DrivingScenario -/
def time_in_rain (scenario : DrivingScenario) : ℝ :=
  sorry

/-- Theorem stating that given the specific conditions, Shelby drove 16 minutes in the rain -/
theorem shelby_rain_time :
  let scenario : DrivingScenario := {
    speed_sun := 40,
    speed_rain := 25,
    total_distance := 20,
    total_time := 36
  }
  time_in_rain scenario = 16 := by
  sorry

end shelby_rain_time_l2209_220953


namespace fairGame_l2209_220911

/-- Represents the number of balls of each color in the bag -/
structure BallCount where
  yellow : ℕ
  black : ℕ
  red : ℕ

/-- Calculates the total number of balls in the bag -/
def totalBalls (count : BallCount) : ℕ :=
  count.yellow + count.black + count.red

/-- Determines if the game is fair given the current ball count -/
def isFair (count : BallCount) : Prop :=
  count.yellow = count.black

/-- Represents the action of replacing black balls with yellow balls -/
def replaceBalls (count : BallCount) (n : ℕ) : BallCount :=
  { yellow := count.yellow + n
    black := count.black - n
    red := count.red }

/-- The main theorem stating that replacing 4 black balls with yellow balls makes the game fair -/
theorem fairGame (initialCount : BallCount)
    (h1 : initialCount.yellow = 5)
    (h2 : initialCount.black = 13)
    (h3 : initialCount.red = 22) :
    isFair (replaceBalls initialCount 4) := by
  sorry

end fairGame_l2209_220911


namespace class_average_score_l2209_220928

theorem class_average_score (total_students : ℕ) (score1 score2 score3 : ℕ) (rest_average : ℚ) :
  total_students = 35 →
  score1 = 93 →
  score2 = 83 →
  score3 = 87 →
  rest_average = 76 →
  (score1 + score2 + score3 + (total_students - 3) * rest_average) / total_students = 77 := by
  sorry

end class_average_score_l2209_220928


namespace expected_value_twelve_sided_die_l2209_220907

/-- A twelve-sided die with faces numbered from 1 to 12 -/
def TwelveSidedDie := Finset.range 12

/-- The expected value of a roll of the twelve-sided die -/
def expectedValue : ℚ := (TwelveSidedDie.sum (λ i => i + 1)) / 12

/-- Theorem: The expected value of a roll of a twelve-sided die with faces numbered from 1 to 12 is 6.5 -/
theorem expected_value_twelve_sided_die : expectedValue = 13/2 := by
  sorry

end expected_value_twelve_sided_die_l2209_220907


namespace strategy_is_injective_l2209_220981

-- Define the set of possible numbers
inductive Number : Type
| one : Number
| two : Number
| three : Number

-- Define the set of possible answers
inductive Answer : Type
| yes : Answer
| no : Answer
| dontKnow : Answer

-- Define the strategy function
def strategy : Number → Answer
| Number.one => Answer.yes
| Number.two => Answer.dontKnow
| Number.three => Answer.no

-- Theorem: The strategy function is injective
theorem strategy_is_injective :
  ∀ x y : Number, x ≠ y → strategy x ≠ strategy y := by
  sorry

#check strategy_is_injective

end strategy_is_injective_l2209_220981


namespace jerry_ring_toss_games_l2209_220915

/-- The number of games Jerry played in the ring toss game -/
def games_played (total_rings : ℕ) (rings_per_game : ℕ) : ℕ :=
  total_rings / rings_per_game

/-- Theorem: Jerry played 8 games of ring toss -/
theorem jerry_ring_toss_games : games_played 48 6 = 8 := by
  sorry

end jerry_ring_toss_games_l2209_220915


namespace fruit_platter_grapes_l2209_220943

theorem fruit_platter_grapes :
  ∀ (b r g c : ℚ),
  b + r + g + c = 360 →
  r = 3 * b →
  g = 4 * c →
  c = 5 * r →
  g = 21600 / 79 := by
sorry

end fruit_platter_grapes_l2209_220943


namespace two_digit_addition_proof_l2209_220971

theorem two_digit_addition_proof (A B C : ℕ) : 
  A ≠ B → A ≠ C → B ≠ C →
  A < 10 → B < 10 → C < 10 →
  A > 0 → B > 0 → C > 0 →
  (10 * A + B) + (10 * B + C) = 100 * B + 10 * C + B →
  A = 9 := by
sorry

end two_digit_addition_proof_l2209_220971


namespace additional_distance_for_average_speed_l2209_220987

theorem additional_distance_for_average_speed
  (initial_distance : ℝ)
  (initial_speed : ℝ)
  (second_speed : ℝ)
  (average_speed : ℝ)
  (h1 : initial_distance = 18)
  (h2 : initial_speed = 36)
  (h3 : second_speed = 60)
  (h4 : average_speed = 45)
  : ∃ (additional_distance : ℝ),
    (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / second_speed)) = average_speed
    ∧ additional_distance = 18 := by
  sorry


end additional_distance_for_average_speed_l2209_220987


namespace treaty_to_university_founding_l2209_220976

theorem treaty_to_university_founding (treaty_day : Nat) (founding_day : Nat) : 
  treaty_day % 7 = 2 → -- Tuesday is represented as 2 (0 = Sunday, 1 = Monday, etc.)
  founding_day = treaty_day + 1204 →
  founding_day % 7 = 5 -- Friday is represented as 5
  := by sorry

end treaty_to_university_founding_l2209_220976


namespace sum_of_numbers_in_ratio_l2209_220909

theorem sum_of_numbers_in_ratio (a b c : ℝ) : 
  a > 0 ∧ b = 2*a ∧ c = 3*a ∧ a^2 + b^2 + c^2 = 2016 → a + b + c = 72 := by
  sorry

end sum_of_numbers_in_ratio_l2209_220909


namespace base_conversion_puzzle_l2209_220932

theorem base_conversion_puzzle :
  ∀ (n : ℕ+) (C D : ℕ),
    C < 8 ∧ D < 8 ∧  -- C and D are single digits in base 8
    C < 5 ∧ D < 5 ∧  -- C and D are single digits in base 5
    n = 8 * C + D ∧  -- base 8 representation
    n = 5 * D + C    -- base 5 representation
    → n = 1 :=
by sorry

end base_conversion_puzzle_l2209_220932


namespace negation_of_universal_proposition_l2209_220929

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 - x + 1/4 > 0)) ↔ (∃ x : ℝ, x^2 - x + 1/4 ≤ 0) :=
by sorry

end negation_of_universal_proposition_l2209_220929


namespace floor_neg_sqrt_64_over_9_l2209_220918

theorem floor_neg_sqrt_64_over_9 : ⌊-Real.sqrt (64/9)⌋ = -3 := by
  sorry

end floor_neg_sqrt_64_over_9_l2209_220918


namespace plane_through_point_and_line_l2209_220957

def line_equation (x y z : ℝ) : Prop :=
  (x - 2) / 4 = (y + 1) / (-5) ∧ (y + 1) / (-5) = (z - 3) / 2

def plane_equation (x y z : ℝ) : Prop :=
  3 * x + 4 * y + 4 * z - 14 = 0

def point_on_plane (x y z : ℝ) : Prop :=
  x = 2 ∧ y = -3 ∧ z = 5

def coefficients_conditions (A B C D : ℤ) : Prop :=
  A > 0 ∧ Nat.gcd (Nat.gcd (Nat.gcd (A.natAbs) (B.natAbs)) (C.natAbs)) (D.natAbs) = 1

theorem plane_through_point_and_line :
  ∀ (x y z : ℝ),
    (∃ (t : ℝ), line_equation (x + t) (y + t) (z + t)) →
    point_on_plane 2 (-3) 5 →
    coefficients_conditions 3 4 4 (-14) →
    plane_equation x y z :=
sorry

end plane_through_point_and_line_l2209_220957


namespace arithmetic_sequence_common_difference_l2209_220967

/-- Given an arithmetic sequence, prove that if the difference of the average of the first 2016 terms
    and the average of the first 16 terms is 100, then the common difference is 1/10. -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (d : ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) = a n + d) 
  (h_sum : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2) 
  (h_condition : S 2016 / 2016 - S 16 / 16 = 100) :
  d = 1 / 10 := by
  sorry

end arithmetic_sequence_common_difference_l2209_220967
