import Mathlib

namespace units_digit_of_2_pow_2015_l1108_110857

theorem units_digit_of_2_pow_2015 (h : ∀ n : ℕ, n > 0 → (2^n : ℕ) % 10 = (2^(n % 4) : ℕ) % 10) :
  (2^2015 : ℕ) % 10 = 8 := by
  sorry

end units_digit_of_2_pow_2015_l1108_110857


namespace a_4_equals_28_l1108_110888

def S (n : ℕ) : ℕ := 4 * n^2

def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem a_4_equals_28 : a 4 = 28 := by
  sorry

end a_4_equals_28_l1108_110888


namespace problem_solving_probability_l1108_110860

theorem problem_solving_probability 
  (arthur_prob : ℚ) 
  (bella_prob : ℚ) 
  (xavier_prob : ℚ) 
  (yvonne_prob : ℚ) 
  (zelda_prob : ℚ) 
  (h_arthur : arthur_prob = 1/4)
  (h_bella : bella_prob = 3/10)
  (h_xavier : xavier_prob = 1/6)
  (h_yvonne : yvonne_prob = 1/2)
  (h_zelda : zelda_prob = 5/8)
  (h_independent : True)  -- Assumption of independence
  : arthur_prob * bella_prob * xavier_prob * yvonne_prob * (1 - zelda_prob) = 9/3840 :=
by
  sorry

end problem_solving_probability_l1108_110860


namespace max_sum_of_squares_difference_l1108_110821

theorem max_sum_of_squares_difference (x y : ℕ+) : 
  x^2 - y^2 = 2016 → x + y ≤ 1008 := by
  sorry

end max_sum_of_squares_difference_l1108_110821


namespace luggage_per_passenger_l1108_110862

theorem luggage_per_passenger (total_passengers : ℕ) (total_bags : ℕ) 
  (h1 : total_passengers = 4) (h2 : total_bags = 32) : 
  total_bags / total_passengers = 8 := by
  sorry

end luggage_per_passenger_l1108_110862


namespace base4_to_base10_3201_l1108_110861

/-- Converts a base 4 number to base 10 -/
def base4_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The base 4 representation of the number -/
def base4_num : List Nat := [1, 0, 2, 3]

theorem base4_to_base10_3201 :
  base4_to_base10 base4_num = 225 := by
  sorry

end base4_to_base10_3201_l1108_110861


namespace only_set_C_not_in_proportion_l1108_110874

def is_in_proportion (a b c d : ℝ) : Prop := a * d = b * c

theorem only_set_C_not_in_proportion :
  (is_in_proportion 4 8 5 10) ∧
  (is_in_proportion 2 (2 * Real.sqrt 5) (Real.sqrt 5) 5) ∧
  ¬(is_in_proportion 1 2 3 4) ∧
  (is_in_proportion 1 2 2 4) :=
by sorry

end only_set_C_not_in_proportion_l1108_110874


namespace fraction_sum_constraint_l1108_110858

theorem fraction_sum_constraint (n : ℕ) (hn : n > 0) :
  (1 : ℚ) / 2 + 1 / 3 + 1 / 10 + 1 / n < 1 → n > 15 := by
  sorry

end fraction_sum_constraint_l1108_110858


namespace soccer_team_non_players_l1108_110883

theorem soccer_team_non_players (total_players : ℕ) (starting_players : ℕ) (first_half_subs : ℕ) :
  total_players = 24 →
  starting_players = 11 →
  first_half_subs = 2 →
  total_players - (starting_players + first_half_subs + 2 * first_half_subs) = 7 :=
by sorry

end soccer_team_non_players_l1108_110883


namespace area_ratio_of_rectangles_l1108_110819

/-- Given two rectangles A and B with specified dimensions, prove that the ratio of their areas is 12/21 -/
theorem area_ratio_of_rectangles (length_A width_A length_B width_B : ℕ) 
  (h1 : length_A = 36) (h2 : width_A = 20) (h3 : length_B = 42) (h4 : width_B = 30) :
  (length_A * width_A : ℚ) / (length_B * width_B) = 12 / 21 := by
  sorry

end area_ratio_of_rectangles_l1108_110819


namespace digit_150_is_6_l1108_110836

/-- The decimal representation of 17/270 as a sequence of digits -/
def decimalRepresentation : ℕ → Fin 10 := sorry

/-- The decimal representation of 17/270 is periodic with period 5 -/
axiom period_five : ∀ n : ℕ, decimalRepresentation n = decimalRepresentation (n + 5)

/-- The first period of the decimal representation -/
axiom first_period : 
  (decimalRepresentation 0 = 0) ∧
  (decimalRepresentation 1 = 6) ∧
  (decimalRepresentation 2 = 2) ∧
  (decimalRepresentation 3 = 9) ∧
  (decimalRepresentation 4 = 6)

/-- The 150th digit after the decimal point in 17/270 is 6 -/
theorem digit_150_is_6 : decimalRepresentation 149 = 6 := by sorry

end digit_150_is_6_l1108_110836


namespace largest_perimeter_is_31_l1108_110893

/-- Represents a triangle with two fixed sides and one variable side --/
structure Triangle where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ

/-- Checks if the given lengths can form a valid triangle --/
def is_valid_triangle (t : Triangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- Calculates the perimeter of a triangle --/
def perimeter (t : Triangle) : ℕ :=
  t.side1 + t.side2 + t.side3

/-- Theorem stating the largest possible perimeter of the triangle --/
theorem largest_perimeter_is_31 :
  ∃ (t : Triangle), t.side1 = 7 ∧ t.side2 = 9 ∧ is_valid_triangle t ∧
  (∀ (t' : Triangle), t'.side1 = 7 ∧ t'.side2 = 9 ∧ is_valid_triangle t' →
    perimeter t' ≤ perimeter t) ∧
  perimeter t = 31 :=
sorry

end largest_perimeter_is_31_l1108_110893


namespace no_rational_solution_l1108_110867

theorem no_rational_solution :
  ¬ ∃ (x y z t : ℚ) (n : ℕ), (x + y * Real.sqrt 2) ^ (2 * n) + (z + t * Real.sqrt 2) ^ (2 * n) = 5 + 4 * Real.sqrt 2 := by
  sorry

end no_rational_solution_l1108_110867


namespace height_prediction_at_10_l1108_110848

/-- Regression model for child height based on age -/
def height_model (x : ℝ) : ℝ := 7.2 * x + 74

/-- The model is valid for children aged 3 to 9 years -/
def valid_age_range : Set ℝ := {x | 3 ≤ x ∧ x ≤ 9}

/-- Prediction is considered approximate if within 1cm of the calculated value -/
def is_approximate (predicted : ℝ) (actual : ℝ) : Prop := abs (predicted - actual) ≤ 1

theorem height_prediction_at_10 :
  is_approximate (height_model 10) 146 :=
sorry

end height_prediction_at_10_l1108_110848


namespace negation_of_proposition_l1108_110850

theorem negation_of_proposition :
  ¬(∀ x : ℝ, x > 0 → x^2 + x + 1 > 0) ↔ 
  (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + x₀ + 1 ≤ 0) :=
by sorry

end negation_of_proposition_l1108_110850


namespace square_of_three_tenths_plus_one_tenth_l1108_110808

theorem square_of_three_tenths_plus_one_tenth (ε : Real) :
  (0.3 : Real)^2 + 0.1 = 0.19 := by
  sorry

end square_of_three_tenths_plus_one_tenth_l1108_110808


namespace pool_capacity_l1108_110849

theorem pool_capacity (additional_water : ℝ) (final_percentage : ℝ) (increase_percentage : ℝ) :
  additional_water = 300 →
  final_percentage = 0.7 →
  increase_percentage = 0.3 →
  ∃ (total_capacity : ℝ),
    total_capacity = 1000 ∧
    additional_water = (final_percentage - increase_percentage) * total_capacity :=
by sorry

end pool_capacity_l1108_110849


namespace maggies_tractor_rate_l1108_110806

/-- Maggie's weekly income calculation --/
theorem maggies_tractor_rate (office_rate : ℝ) (office_hours tractor_hours total_income : ℝ) :
  office_rate = 10 →
  office_hours = 2 * tractor_hours →
  tractor_hours = 13 →
  total_income = 416 →
  total_income = office_rate * office_hours + tractor_hours * (total_income - office_rate * office_hours) / tractor_hours →
  (total_income - office_rate * office_hours) / tractor_hours = 12 := by
sorry


end maggies_tractor_rate_l1108_110806


namespace fraction_to_decimal_l1108_110828

theorem fraction_to_decimal : 58 / 125 = 0.464 := by
  sorry

end fraction_to_decimal_l1108_110828


namespace sons_age_l1108_110880

/-- Given a father and son with specific age relationships, prove the son's age -/
theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 25 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 23 := by
  sorry

end sons_age_l1108_110880


namespace v_closed_under_multiplication_v_not_closed_under_add_cube_root_v_not_closed_under_division_v_not_closed_under_cube_cube_root_l1108_110864

-- Define the set v of cubes of positive integers
def v : Set ℕ := {n : ℕ | ∃ m : ℕ, n = m ^ 3}

-- Closure under multiplication
theorem v_closed_under_multiplication :
  ∀ a b : ℕ, a ∈ v → b ∈ v → (a * b) ∈ v :=
sorry

-- Not closed under addition followed by cube root
theorem v_not_closed_under_add_cube_root :
  ∃ a b : ℕ, a ∈ v ∧ b ∈ v ∧ (∃ c : ℕ, c ^ 3 = a + b) → (∃ d : ℕ, d ^ 3 = a + b) :=
sorry

-- Not closed under division
theorem v_not_closed_under_division :
  ∃ a b : ℕ, a ∈ v ∧ b ∈ v ∧ b ≠ 0 → (a / b) ∉ v :=
sorry

-- Not closed under cubing followed by cube root
theorem v_not_closed_under_cube_cube_root :
  ∃ a : ℕ, a ∈ v ∧ (∃ b : ℕ, b ^ 3 = a ^ 3) → (∃ c : ℕ, c ^ 3 = a ^ 3) :=
sorry

end v_closed_under_multiplication_v_not_closed_under_add_cube_root_v_not_closed_under_division_v_not_closed_under_cube_cube_root_l1108_110864


namespace binomial_coefficient_sum_l1108_110815

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x, (1 - x)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| + |a₈| + |a₉| = 511 := by
  sorry

end binomial_coefficient_sum_l1108_110815


namespace probability_sqrt_less_than_9_l1108_110878

/-- A two-digit whole number is an integer between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The count of two-digit whole numbers whose square root is less than 9. -/
def CountLessThan9 : ℕ := 71

/-- The total count of two-digit whole numbers. -/
def TotalTwoDigitNumbers : ℕ := 90

/-- The probability that the square root of a randomly selected two-digit whole number is less than 9. -/
theorem probability_sqrt_less_than_9 :
  (CountLessThan9 : ℚ) / TotalTwoDigitNumbers = 71 / 90 := by sorry

end probability_sqrt_less_than_9_l1108_110878


namespace correct_operation_l1108_110805

theorem correct_operation (x : ℝ) : x - 2*x = -x := by
  sorry

end correct_operation_l1108_110805


namespace nacho_triple_divya_age_l1108_110863

/-- Represents the number of years in the future when Nacho will be three times older than Divya -/
def future_years : ℕ := 10

/-- Divya's current age -/
def divya_age : ℕ := 5

/-- The sum of Nacho's and Divya's current ages -/
def total_current_age : ℕ := 40

/-- Nacho's current age -/
def nacho_age : ℕ := total_current_age - divya_age

theorem nacho_triple_divya_age : 
  nacho_age + future_years = 3 * (divya_age + future_years) :=
sorry

end nacho_triple_divya_age_l1108_110863


namespace cookie_circle_radius_l1108_110895

theorem cookie_circle_radius (x y : ℝ) :
  x^2 + y^2 + 36 = 6*x + 12*y →
  ∃ (center : ℝ × ℝ), (x - center.1)^2 + (y - center.2)^2 = 3^2 := by
sorry

end cookie_circle_radius_l1108_110895


namespace regular_polygon_with_150_degree_interior_angle_has_12_sides_l1108_110814

/-- A regular polygon with an interior angle of 150° has 12 sides -/
theorem regular_polygon_with_150_degree_interior_angle_has_12_sides :
  ∀ (n : ℕ), n > 2 →
  (∃ (angle : ℝ), angle = 150 ∧ angle * n = 180 * (n - 2)) →
  n = 12 := by
  sorry

end regular_polygon_with_150_degree_interior_angle_has_12_sides_l1108_110814


namespace expand_expression_l1108_110866

theorem expand_expression (x : ℝ) : (17*x + 18 + 5)*3*x = 51*x^2 + 69*x := by
  sorry

end expand_expression_l1108_110866


namespace only_prime_square_difference_pair_l1108_110844

theorem only_prime_square_difference_pair : 
  ∀ p q : ℕ, 
    Prime p → 
    Prime q → 
    p > q → 
    Prime (p^2 - q^2) → 
    p = 3 ∧ q = 2 :=
by sorry

end only_prime_square_difference_pair_l1108_110844


namespace stating_interest_rate_calculation_l1108_110803

/-- Represents the annual interest rate as a percentage -/
def annual_rate : ℝ := 15

/-- Represents the principal amount in rupees -/
def principal : ℝ := 147.69

/-- Represents the time period for the first deposit in years -/
def time1 : ℝ := 3.5

/-- Represents the time period for the second deposit in years -/
def time2 : ℝ := 10

/-- Represents the difference in interests in rupees -/
def interest_diff : ℝ := 144

/-- 
Theorem stating that given the conditions, the annual interest rate is approximately 15%.
-/
theorem interest_rate_calculation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |annual_rate - (interest_diff * 100) / (principal * (time2 - time1))| < ε :=
sorry

end stating_interest_rate_calculation_l1108_110803


namespace complex_real_condition_l1108_110807

theorem complex_real_condition (z : ℂ) : (z + 2*I).im = 0 ↔ z.im = -2 := by sorry

end complex_real_condition_l1108_110807


namespace pencil_length_l1108_110813

theorem pencil_length (black_fraction : Real) (white_fraction : Real) (blue_length : Real) :
  black_fraction = 1/8 →
  white_fraction = 1/2 →
  blue_length = 3.5 →
  ∃ (total_length : Real),
    total_length * black_fraction +
    (total_length - total_length * black_fraction) * white_fraction +
    blue_length = total_length ∧
    total_length = 8 := by
  sorry

end pencil_length_l1108_110813


namespace A_proper_superset_B_l1108_110896

def A : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}
def B : Set ℤ := {y | ∃ k : ℤ, y = 4 * k}

theorem A_proper_superset_B : A ⊃ B := by
  sorry

end A_proper_superset_B_l1108_110896


namespace coin_division_l1108_110879

theorem coin_division (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 8 ≠ 6 ∨ m % 9 ≠ 7)) → 
  n % 8 = 6 → 
  n % 9 = 7 → 
  n % 11 = 8 := by
sorry

end coin_division_l1108_110879


namespace mother_daughter_age_relation_l1108_110894

theorem mother_daughter_age_relation :
  ∀ (mother_age daughter_age years_ago : ℕ),
  mother_age = 43 →
  daughter_age = 11 →
  mother_age - years_ago = 5 * (daughter_age - years_ago) →
  years_ago = 3 :=
by
  sorry

end mother_daughter_age_relation_l1108_110894


namespace intersection_A_B_l1108_110816

def A : Set ℝ := {x | x / (x - 1) < 0}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x < 1} := by sorry

end intersection_A_B_l1108_110816


namespace opposite_of_negative_2022_l1108_110881

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_negative_2022 : opposite (-2022) = 2022 := by
  sorry

end opposite_of_negative_2022_l1108_110881


namespace root_sum_reciprocal_l1108_110810

theorem root_sum_reciprocal (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ + 2 = 0 → 
  x₂^2 - 3*x₂ + 2 = 0 → 
  x₁ ≠ x₂ →
  (1/x₁) + (1/x₂) = 3/2 := by
  sorry

end root_sum_reciprocal_l1108_110810


namespace train_speed_l1108_110865

/-- The speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 250) (h2 : time = 12) :
  ∃ (speed : ℝ), abs (speed - length / time) < 0.01 := by
  sorry

end train_speed_l1108_110865


namespace vector_b_calculation_l1108_110835

theorem vector_b_calculation (a b : ℝ × ℝ) : 
  a = (1, 2) → (2 • a + b = (3, 2)) → b = (1, -2) := by sorry

end vector_b_calculation_l1108_110835


namespace smallest_n_for_irreducible_fractions_l1108_110825

theorem smallest_n_for_irreducible_fractions : ∃ (n : ℕ), 
  (n = 95) ∧ 
  (∀ (k : ℕ), 19 ≤ k ∧ k ≤ 91 → Nat.gcd k (n + k + 2) = 1) ∧
  (∀ (m : ℕ), m < n → ∃ (k : ℕ), 19 ≤ k ∧ k ≤ 91 ∧ Nat.gcd k (m + k + 2) ≠ 1) :=
by sorry

end smallest_n_for_irreducible_fractions_l1108_110825


namespace solution_implies_a_value_l1108_110834

theorem solution_implies_a_value (x a : ℝ) : 
  x = 5 → 2 * x + 3 * a = 4 → a = -2 := by
  sorry

end solution_implies_a_value_l1108_110834


namespace exponential_inequality_l1108_110842

theorem exponential_inequality : 
  (2/5 : ℝ)^(3/5) < (2/5 : ℝ)^(2/5) ∧ (2/5 : ℝ)^(2/5) < (3/5 : ℝ)^(3/5) := by
  sorry

end exponential_inequality_l1108_110842


namespace problem_solution_l1108_110833

def δ (x : ℝ) : ℝ := 3 * x + 8
def φ (x : ℝ) : ℝ := 9 * x + 7

theorem problem_solution (x : ℝ) : δ (φ x) = 11 → x = -2/3 := by
  sorry

end problem_solution_l1108_110833


namespace garden_spaces_per_row_l1108_110847

/-- Represents a vegetable garden with given properties --/
structure Garden where
  tomatoes : Nat
  cucumbers : Nat
  potatoes : Nat
  rows : Nat
  additional_capacity : Nat

/-- Calculates the number of spaces in each row of the garden --/
def spaces_per_row (g : Garden) : Nat :=
  ((g.tomatoes + g.cucumbers + g.potatoes + g.additional_capacity) / g.rows)

/-- Theorem stating that for the given garden configuration, there are 15 spaces per row --/
theorem garden_spaces_per_row :
  let g : Garden := {
    tomatoes := 3 * 5,
    cucumbers := 5 * 4,
    potatoes := 30,
    rows := 10,
    additional_capacity := 85
  }
  spaces_per_row g = 15 := by
  sorry

end garden_spaces_per_row_l1108_110847


namespace bishopArrangements_isPerfectSquare_l1108_110876

/-- The size of the chessboard -/
def boardSize : ℕ := 8

/-- The number of squares of one color on the board -/
def squaresPerColor : ℕ := boardSize * boardSize / 2

/-- The maximum number of non-threatening bishops on squares of one color -/
def maxBishopsPerColor : ℕ := boardSize

/-- The number of ways to arrange the maximum number of non-threatening bishops on an 8x8 chessboard -/
def totalArrangements : ℕ := (Nat.choose squaresPerColor maxBishopsPerColor) ^ 2

/-- Theorem stating that the number of arrangements is a perfect square -/
theorem bishopArrangements_isPerfectSquare : 
  ∃ n : ℕ, totalArrangements = n ^ 2 := by
sorry

end bishopArrangements_isPerfectSquare_l1108_110876


namespace probability_sum_15_l1108_110852

def is_valid_roll (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6

def sum_is_15 (a b c : ℕ) : Prop :=
  a + b + c = 15

def count_valid_rolls : ℕ := 216

def count_sum_15_rolls : ℕ := 10

theorem probability_sum_15 :
  (count_sum_15_rolls : ℚ) / count_valid_rolls = 5 / 108 :=
sorry

end probability_sum_15_l1108_110852


namespace certain_number_divisibility_l1108_110856

theorem certain_number_divisibility (z x : ℕ) (h1 : z > 0) (h2 : 4 ∣ z) : 
  (z + x + 4 + z + 3) % 2 = 1 ↔ Even x := by sorry

end certain_number_divisibility_l1108_110856


namespace negation_equivalence_l1108_110801

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x < Real.sin x ∨ x > Real.tan x) ↔
  (∀ x : ℝ, x ≥ Real.sin x ∧ x ≤ Real.tan x) := by
  sorry

end negation_equivalence_l1108_110801


namespace sin_cos_identity_l1108_110823

theorem sin_cos_identity : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1/2 := by
  sorry

end sin_cos_identity_l1108_110823


namespace specific_tree_height_l1108_110877

/-- Represents the height of a tree after a given number of years -/
def tree_height (initial_height : ℝ) (yearly_growth : ℝ) (years : ℝ) : ℝ :=
  initial_height + yearly_growth * years

/-- Theorem stating the height of a specific tree after n years -/
theorem specific_tree_height (n : ℝ) :
  tree_height 1.8 0.3 n = 0.3 * n + 1.8 := by
  sorry

end specific_tree_height_l1108_110877


namespace biology_score_is_85_l1108_110839

def mathematics_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def english_score : ℕ := 47
def average_score : ℕ := 71
def total_subjects : ℕ := 5

def biology_score : ℕ := 
  average_score * total_subjects - (mathematics_score + science_score + social_studies_score + english_score)

theorem biology_score_is_85 : biology_score = 85 := by sorry

end biology_score_is_85_l1108_110839


namespace shaded_area_of_square_l1108_110855

/-- Given a square composed of 25 congruent smaller squares with a diagonal of 10 cm,
    prove that its area is 50 square cm. -/
theorem shaded_area_of_square (d : ℝ) (n : ℕ) (h1 : d = 10) (h2 : n = 25) :
  (d^2 / 2 : ℝ) = 50 := by
  sorry

end shaded_area_of_square_l1108_110855


namespace walter_at_zoo_l1108_110890

theorem walter_at_zoo (seal_time penguin_time elephant_time total_time : ℕ) 
  (h1 : penguin_time = 8 * seal_time)
  (h2 : elephant_time = 13)
  (h3 : total_time = 130)
  (h4 : seal_time + penguin_time + elephant_time = total_time) :
  seal_time = 13 := by
  sorry

end walter_at_zoo_l1108_110890


namespace x_cos_x_necessary_not_sufficient_l1108_110811

theorem x_cos_x_necessary_not_sufficient (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  (∀ y : ℝ, 0 < y ∧ y < Real.pi / 2 → (y < 1 → y * Real.cos y < 1)) ∧
  (∃ z : ℝ, 0 < z ∧ z < Real.pi / 2 ∧ z * Real.cos z < 1 ∧ z ≥ 1) := by
  sorry

end x_cos_x_necessary_not_sufficient_l1108_110811


namespace divisors_sum_and_product_l1108_110873

theorem divisors_sum_and_product (p : ℕ) (hp : Prime p) :
  let a := p^106
  ∀ (divisors : Finset ℕ), 
    (∀ d ∈ divisors, d ∣ a) ∧ 
    (∀ d : ℕ, d ∣ a → d ∈ divisors) ∧ 
    (Finset.card divisors = 107) →
    (divisors.sum id = (p^107 - 1) / (p - 1)) ∧
    (divisors.prod id = p^11321) := by
  sorry

end divisors_sum_and_product_l1108_110873


namespace collinear_vectors_l1108_110884

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![2, 3]

-- Define the sum vector
def sum_vector : Fin 2 → ℝ := ![3, 5]

-- Define the collinear vector
def collinear_vector : Fin 2 → ℝ := ![6, 10]

-- Theorem statement
theorem collinear_vectors :
  (∃ k : ℝ, ∀ i : Fin 2, collinear_vector i = k * sum_vector i) ∧
  (∀ i : Fin 2, sum_vector i = a i + b i) := by
  sorry

end collinear_vectors_l1108_110884


namespace dean_transactions_l1108_110854

theorem dean_transactions (mabel anthony cal jade dean : ℕ) : 
  mabel = 90 →
  anthony = mabel + (mabel / 10) →
  cal = (2 * anthony) / 3 →
  jade = cal + 14 →
  dean = jade + (jade / 4) →
  dean = 100 := by
sorry

end dean_transactions_l1108_110854


namespace intersection_with_complement_l1108_110899

def U : Finset ℕ := {1, 2, 3, 4, 5, 6}
def A : Finset ℕ := {2, 3}
def B : Finset ℕ := {3, 5}

theorem intersection_with_complement : A ∩ (U \ B) = {2} := by sorry

end intersection_with_complement_l1108_110899


namespace divisor_count_problem_l1108_110830

def τ (n : ℕ) : ℕ := (Nat.divisors n).card

theorem divisor_count_problem :
  (Finset.filter (fun n => τ n > 2 ∧ τ (τ n) = 2) (Finset.range 1001)).card = 184 := by
  sorry

end divisor_count_problem_l1108_110830


namespace peter_pizza_fraction_l1108_110886

theorem peter_pizza_fraction :
  ∀ (total_slices : ℕ) (peter_own_slices : ℕ) (shared_slices : ℕ),
  total_slices = 16 →
  peter_own_slices = 2 →
  shared_slices = 2 →
  (peter_own_slices : ℚ) / total_slices + (shared_slices / 2 : ℚ) / total_slices = 3 / 16 := by
  sorry

end peter_pizza_fraction_l1108_110886


namespace boat_downstream_distance_l1108_110824

/-- Calculates the distance traveled downstream by a boat -/
theorem boat_downstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (travel_time : ℝ) 
  (h1 : boat_speed = 22) 
  (h2 : stream_speed = 5) 
  (h3 : travel_time = 4) : 
  boat_speed + stream_speed * travel_time = 108 := by
  sorry

end boat_downstream_distance_l1108_110824


namespace inverse_variation_cube_fourth_l1108_110809

/-- Given that a³ varies inversely with b⁴, and a = 2 when b = 4,
    prove that a = 1/∛2 when b = 8 -/
theorem inverse_variation_cube_fourth (a b : ℝ) (k : ℝ) :
  (∀ a b, a^3 * b^4 = k) →  -- a³ varies inversely with b⁴
  (2^3 * 4^4 = k) →         -- a = 2 when b = 4
  (a^3 * 8^4 = k) →         -- condition for b = 8
  a = 1 / (2^(1/3)) :=      -- a = 1/∛2 when b = 8
by sorry

end inverse_variation_cube_fourth_l1108_110809


namespace two_negative_factors_l1108_110817

theorem two_negative_factors
  (a b c : ℚ)
  (h : a * b * c > 0) :
  (a < 0 ∧ b < 0 ∧ c > 0) ∨
  (a < 0 ∧ b > 0 ∧ c < 0) ∨
  (a > 0 ∧ b < 0 ∧ c < 0) :=
sorry

end two_negative_factors_l1108_110817


namespace complex_expression_equals_negative_seven_l1108_110826

theorem complex_expression_equals_negative_seven :
  Real.sqrt 8 + (1/2)⁻¹ - 4 * Real.cos (45 * π / 180) - 2 / (1/2) * 2 - (2009 - Real.sqrt 3)^0 = -7 := by
  sorry

end complex_expression_equals_negative_seven_l1108_110826


namespace not_right_triangle_l1108_110804

theorem not_right_triangle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = 2 * B) (h3 : A = 3 * C) :
  A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := by
  sorry

end not_right_triangle_l1108_110804


namespace place_value_ratio_l1108_110829

/-- The number we're analyzing -/
def number : ℚ := 90347.6208

/-- The place value of a digit in a decimal number -/
def place_value (digit : ℕ) (position : ℤ) : ℚ := (digit : ℚ) * 10 ^ position

/-- The position of the digit 0 in the number (counting from right, with decimal point at 0) -/
def zero_position : ℤ := 4

/-- The position of the digit 6 in the number (counting from right, with decimal point at 0) -/
def six_position : ℤ := -1

theorem place_value_ratio :
  place_value 1 zero_position / place_value 1 six_position = 100000 := by sorry

end place_value_ratio_l1108_110829


namespace rope_length_increase_l1108_110851

/-- Proves that increasing a circular area with initial radius 10 m by 942.8571428571429 m² results in a new radius of 20 m -/
theorem rope_length_increase (π : Real) (initial_radius : Real) (area_increase : Real) (new_radius : Real) : 
  π > 0 → 
  initial_radius = 10 → 
  area_increase = 942.8571428571429 → 
  new_radius = 20 → 
  π * new_radius^2 = π * initial_radius^2 + area_increase := by
  sorry

#check rope_length_increase

end rope_length_increase_l1108_110851


namespace equilateral_center_triangles_properties_l1108_110838

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Represents an equilateral triangle constructed on a side of another triangle -/
structure EquilateralTriangle where
  base : ℝ × ℝ
  apex : ℝ × ℝ

/-- The triangle formed by centers of equilateral triangles -/
def CenterTriangle (T : Triangle) (outward : Bool) : Triangle := sorry

/-- The centroid of a triangle -/
def centroid (T : Triangle) : ℝ × ℝ := sorry

/-- The area of a triangle -/
def area (T : Triangle) : ℝ := sorry

/-- Main theorem about properties of triangles formed by centers of equilateral triangles -/
theorem equilateral_center_triangles_properties (T : Triangle) :
  let Δ := CenterTriangle T true
  let δ := CenterTriangle T false
  -- 1) Δ and δ are equilateral
  (∀ (X Y Z : ℝ × ℝ), (X = Δ.A ∧ Y = Δ.B ∧ Z = Δ.C) ∨ (X = δ.A ∧ Y = δ.B ∧ Z = δ.C) →
    dist X Y = dist Y Z ∧ dist Y Z = dist Z X) ∧
  -- 2) Centers of Δ and δ coincide with the centroid of T
  (centroid Δ = centroid T ∧ centroid δ = centroid T) ∧
  -- 3) Area(Δ) - Area(δ) = Area(T)
  (area Δ - area δ = area T) := by
  sorry


end equilateral_center_triangles_properties_l1108_110838


namespace georgia_buttons_l1108_110845

/-- Georgia's button problem -/
theorem georgia_buttons (yellow black green given_away remaining : ℕ) :
  yellow + black + green = given_away + remaining →
  remaining = 5 →
  yellow = 4 →
  black = 2 →
  green = 3 →
  given_away = 4 :=
by sorry

end georgia_buttons_l1108_110845


namespace solve_barnyard_owl_problem_l1108_110885

def barnyard_owl_problem (hoots_per_owl : ℕ) (total_hoots : ℕ) : Prop :=
  let num_owls := (20 - 5) / hoots_per_owl
  hoots_per_owl = 5 ∧ total_hoots = 20 - 5 → num_owls = 3

theorem solve_barnyard_owl_problem :
  ∃ (hoots_per_owl total_hoots : ℕ), barnyard_owl_problem hoots_per_owl total_hoots :=
sorry

end solve_barnyard_owl_problem_l1108_110885


namespace certain_number_proof_l1108_110822

theorem certain_number_proof (p q : ℝ) 
  (h1 : 3 / p = 6) 
  (h2 : p - q = 0.3) : 
  3 / q = 15 := by
sorry

end certain_number_proof_l1108_110822


namespace weekly_production_total_l1108_110871

def john_rate : ℕ := 20
def jane_rate : ℕ := 15
def john_hours : List ℕ := [8, 6, 7, 5, 4]
def jane_hours : List ℕ := [7, 7, 6, 7, 8]

theorem weekly_production_total :
  (john_hours.map (· * john_rate)).sum + (jane_hours.map (· * jane_rate)).sum = 1125 := by
  sorry

end weekly_production_total_l1108_110871


namespace negation_of_all_exponential_monotonic_l1108_110831

-- Define the set of exponential functions
def ExponentialFunction : Type := ℝ → ℝ

-- Define the property of being monotonic
def Monotonic (f : ℝ → ℝ) : Prop := ∀ x y, x ≤ y → f x ≤ f y

-- State the theorem
theorem negation_of_all_exponential_monotonic :
  (¬ ∀ f : ExponentialFunction, Monotonic f) ↔ (∃ f : ExponentialFunction, ¬ Monotonic f) := by
  sorry

end negation_of_all_exponential_monotonic_l1108_110831


namespace hexagon_perimeter_l1108_110802

/-- The perimeter of a hexagon ABCDEF where five sides are of length 1 and the sixth side is √5 -/
theorem hexagon_perimeter (AB BC CD DE EF : ℝ) (AF : ℝ) 
  (h1 : AB = 1) (h2 : BC = 1) (h3 : CD = 1) (h4 : DE = 1) (h5 : EF = 1)
  (h6 : AF = Real.sqrt 5) : AB + BC + CD + DE + EF + AF = 5 + Real.sqrt 5 :=
by sorry

end hexagon_perimeter_l1108_110802


namespace no_two_digit_number_satisfies_conditions_l1108_110841

theorem no_two_digit_number_satisfies_conditions : ¬∃ n : ℕ,
  10 ≤ n ∧ n < 100 ∧  -- two-digit number
  Even n ∧            -- even
  n % 13 = 0 ∧        -- multiple of 13
  ∃ a b : ℕ,          -- digits a and b
    n = 10 * a + b ∧
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    ∃ k : ℕ, a * b = k * k  -- product of digits is a perfect square
  := by sorry

end no_two_digit_number_satisfies_conditions_l1108_110841


namespace stratified_sampling_problem_l1108_110859

theorem stratified_sampling_problem (total_population : ℕ) 
  (stratum_size : ℕ) (stratum_sample : ℕ) (h1 : total_population = 55) 
  (h2 : stratum_size = 15) (h3 : stratum_sample = 3) :
  (stratum_sample : ℚ) * total_population / stratum_size = 11 := by
  sorry

end stratified_sampling_problem_l1108_110859


namespace yankees_to_mets_ratio_l1108_110891

/-- Represents the number of fans for each baseball team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- The total number of baseball fans in the town -/
def total_fans : ℕ := 390

/-- The theorem stating the ratio of NY Yankees fans to NY Mets fans -/
theorem yankees_to_mets_ratio (fc : FanCounts) : 
  fc.yankees = 156 ∧ fc.mets = 104 ∧ fc.red_sox = 130 →
  fc.yankees + fc.mets + fc.red_sox = total_fans →
  fc.mets * 5 = fc.red_sox * 4 →
  fc.yankees * 2 = fc.mets * 3 := by
  sorry

#check yankees_to_mets_ratio

end yankees_to_mets_ratio_l1108_110891


namespace outer_boundary_diameter_is_44_l1108_110897

/-- The diameter of the circular fountain in feet. -/
def fountain_diameter : ℝ := 12

/-- The width of the garden ring in feet. -/
def garden_width : ℝ := 10

/-- The width of the walking path in feet. -/
def path_width : ℝ := 6

/-- The diameter of the circle forming the outer boundary of the walking path. -/
def outer_boundary_diameter : ℝ := fountain_diameter + 2 * (garden_width + path_width)

/-- Theorem stating that the diameter of the circle forming the outer boundary of the walking path is 44 feet. -/
theorem outer_boundary_diameter_is_44 : outer_boundary_diameter = 44 := by
  sorry

end outer_boundary_diameter_is_44_l1108_110897


namespace percentage_of_cat_owners_l1108_110812

theorem percentage_of_cat_owners (total_students : ℕ) (cat_owners : ℕ) 
  (h1 : total_students = 400) (h2 : cat_owners = 80) : 
  (cat_owners : ℝ) / total_students * 100 = 20 := by
  sorry

end percentage_of_cat_owners_l1108_110812


namespace even_polynomial_iff_product_with_negation_l1108_110882

/-- A polynomial over the complex numbers. -/
def ComplexPolynomial := ℂ → ℂ

/-- Predicate for even functions. -/
def IsEven (P : ComplexPolynomial) : Prop :=
  ∀ z : ℂ, P z = P (-z)

/-- The main theorem: A complex polynomial is even if and only if
    it can be expressed as the product of a polynomial and its negation. -/
theorem even_polynomial_iff_product_with_negation (P : ComplexPolynomial) :
  IsEven P ↔ ∃ Q : ComplexPolynomial, ∀ z : ℂ, P z = (Q z) * (Q (-z)) := by
  sorry

end even_polynomial_iff_product_with_negation_l1108_110882


namespace unfair_coin_probability_l1108_110846

/-- The probability of getting exactly k heads in n tosses of a coin with probability p of heads -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

/-- The main theorem -/
theorem unfair_coin_probability (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  binomial_probability 7 4 p = 210 / 1024 → p = 4 / 7 := by
  sorry

#check unfair_coin_probability

end unfair_coin_probability_l1108_110846


namespace class_size_l1108_110840

theorem class_size (num_groups : ℕ) (students_per_group : ℕ) 
  (h1 : num_groups = 5) 
  (h2 : students_per_group = 6) : 
  num_groups * students_per_group = 30 := by
  sorry

end class_size_l1108_110840


namespace length_of_AB_l1108_110818

/-- Given a line segment AB with points P and Q, prove that AB has length 70 -/
theorem length_of_AB (A B P Q : ℝ) : 
  (0 < A ∧ A < P ∧ P < Q ∧ Q < B) →  -- P and Q are in AB and on the same side of midpoint
  (P - A) / (B - P) = 2 / 3 →        -- P divides AB in ratio 2:3
  (Q - A) / (B - Q) = 3 / 4 →        -- Q divides AB in ratio 3:4
  Q - P = 2 →                        -- PQ = 2
  B - A = 70 := by                   -- AB has length 70
sorry


end length_of_AB_l1108_110818


namespace product_equals_square_l1108_110872

theorem product_equals_square : 100 * 29.98 * 2.998 * 1000 = (2998 : ℝ)^2 := by
  sorry

end product_equals_square_l1108_110872


namespace fraction_equals_870_l1108_110869

theorem fraction_equals_870 (a : ℕ+) :
  (a : ℚ) / ((a : ℚ) + 50) = 870 / 1000 → a = 335 := by
  sorry

end fraction_equals_870_l1108_110869


namespace solve_for_y_l1108_110889

theorem solve_for_y (x y : ℚ) 
  (h1 : x = 103)
  (h2 : x^3 * y - 2 * x^2 * y + x * y - 100 * y = 1061500) : 
  y = 125 / 126 := by
sorry

end solve_for_y_l1108_110889


namespace line_inclination_angle_l1108_110868

def angle_of_inclination (x y : ℝ → ℝ) : ℝ := by sorry

theorem line_inclination_angle (t : ℝ) :
  let x := λ t : ℝ => 1 + Real.sqrt 3 * t
  let y := λ t : ℝ => 3 - 3 * t
  angle_of_inclination x y = 120 * π / 180 := by sorry

end line_inclination_angle_l1108_110868


namespace intersection_A_B_l1108_110898

def A : Set ℤ := {-3, -2, -1, 0, 1}
def B : Set ℤ := {x | x^2 - 4 = 0}

theorem intersection_A_B : A ∩ B = {-2} := by sorry

end intersection_A_B_l1108_110898


namespace coffee_cost_l1108_110887

/-- The cost of each coffee Jon buys, given his spending habits in April. -/
theorem coffee_cost (coffees_per_day : ℕ) (total_spent : ℕ) (days_in_april : ℕ) :
  coffees_per_day = 2 →
  total_spent = 120 →
  days_in_april = 30 →
  total_spent / (coffees_per_day * days_in_april) = 2 :=
by sorry

end coffee_cost_l1108_110887


namespace every_nat_sum_of_two_three_powers_l1108_110827

def is_power_of_two_three (n : ℕ) : Prop :=
  ∃ (α β : ℕ), n = 2^α * 3^β

def summands_not_multiples (s : List ℕ) : Prop :=
  ∀ i j, i ≠ j → i < s.length → j < s.length →
    ¬(s.get ⟨i, by sorry⟩ ∣ s.get ⟨j, by sorry⟩) ∧
    ¬(s.get ⟨j, by sorry⟩ ∣ s.get ⟨i, by sorry⟩)

theorem every_nat_sum_of_two_three_powers :
  ∀ n : ℕ, n > 0 →
    ∃ s : List ℕ,
      (∀ m ∈ s.toFinset, is_power_of_two_three m) ∧
      (s.sum = n) ∧
      summands_not_multiples s :=
sorry

end every_nat_sum_of_two_three_powers_l1108_110827


namespace south_opposite_of_north_l1108_110832

/-- Represents the direction of movement --/
inductive Direction
  | North
  | South

/-- Represents a distance with direction --/
structure DirectedDistance where
  distance : ℝ
  direction : Direction

/-- Denotes a distance in kilometers with a sign --/
def denote (d : DirectedDistance) : ℝ :=
  match d.direction with
  | Direction.North => d.distance
  | Direction.South => -d.distance

theorem south_opposite_of_north 
  (h : denote { distance := 3, direction := Direction.North } = 3) :
  denote { distance := 5, direction := Direction.South } = -5 := by
  sorry


end south_opposite_of_north_l1108_110832


namespace heart_ratio_two_four_four_two_l1108_110875

def heart (n m : ℕ) : ℕ := n^(3+m) * m^(2+n)

theorem heart_ratio_two_four_four_two :
  (heart 2 4 : ℚ) / (heart 4 2) = 1/2 := by sorry

end heart_ratio_two_four_four_two_l1108_110875


namespace vector_sum_proof_l1108_110800

/-- Given two vectors in a plane, prove that their sum with specific coefficients equals a certain vector. -/
theorem vector_sum_proof (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-2, m]
  (a • b = 0) → 
  (2 • a + 3 • b : Fin 2 → ℝ) = ![-4, 7] :=
by sorry

end vector_sum_proof_l1108_110800


namespace rectangle_area_problem_l1108_110870

theorem rectangle_area_problem (total_area area1 area2 : ℝ) :
  total_area = 48 ∧ area1 = 24 ∧ area2 = 13 →
  total_area - (area1 + area2) = 11 := by
sorry

end rectangle_area_problem_l1108_110870


namespace certain_number_proof_l1108_110892

theorem certain_number_proof (x : ℝ) : (3 / 5) * x^2 = 126.15 → x = 14.5 := by
  sorry

end certain_number_proof_l1108_110892


namespace exam_mean_score_l1108_110837

theorem exam_mean_score (morning_mean : ℝ) (afternoon_mean : ℝ) (ratio : ℚ) 
  (h1 : morning_mean = 90)
  (h2 : afternoon_mean = 75)
  (h3 : ratio = 5 / 7) : 
  ∃ (overall_mean : ℝ), 
    (overall_mean ≥ 81 ∧ overall_mean < 82) ∧ 
    (∀ (m a : ℕ), m / a = ratio → 
      (m * morning_mean + a * afternoon_mean) / (m + a) = overall_mean) :=
sorry

end exam_mean_score_l1108_110837


namespace x_squared_minus_y_squared_l1108_110820

theorem x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 15) 
  (h2 : 3 * x + y = 20) : 
  x^2 - y^2 = -150 := by
  sorry

end x_squared_minus_y_squared_l1108_110820


namespace root_sum_fraction_l1108_110843

theorem root_sum_fraction (α β γ : ℂ) : 
  α^3 - α - 1 = 0 → β^3 - β - 1 = 0 → γ^3 - γ - 1 = 0 →
  (1 + α) / (1 - α) + (1 + β) / (1 - β) + (1 + γ) / (1 - γ) = -7 := by
  sorry

end root_sum_fraction_l1108_110843


namespace matrix_determinant_solution_l1108_110853

theorem matrix_determinant_solution (a : ℝ) (ha : a ≠ 0) :
  let matrix (x : ℝ) := !![x + a, 2*x, 2*x; 2*x, x + a, 2*x; 2*x, 2*x, x + a]
  ∀ x : ℝ, Matrix.det (matrix x) = 0 ↔ x = -a ∨ x = a/3 := by
  sorry

end matrix_determinant_solution_l1108_110853
