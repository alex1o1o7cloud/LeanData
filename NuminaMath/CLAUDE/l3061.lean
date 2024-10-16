import Mathlib

namespace NUMINAMATH_CALUDE_magnitude_of_AB_l3061_306177

-- Define points A and B
def A : ℝ × ℝ := (-3, 4)
def B : ℝ × ℝ := (5, -2)

-- Define vector AB
def vectorAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Theorem: The magnitude of vector AB is 10
theorem magnitude_of_AB : Real.sqrt (vectorAB.1^2 + vectorAB.2^2) = 10 := by
  sorry


end NUMINAMATH_CALUDE_magnitude_of_AB_l3061_306177


namespace NUMINAMATH_CALUDE_terrys_test_score_l3061_306107

theorem terrys_test_score (total_problems : ℕ) (total_score : ℕ) 
  (correct_points : ℕ) (incorrect_points : ℕ) :
  total_problems = 25 →
  total_score = 85 →
  correct_points = 4 →
  incorrect_points = 1 →
  ∃ (correct incorrect : ℕ),
    correct + incorrect = total_problems ∧
    correct_points * correct - incorrect_points * incorrect = total_score ∧
    incorrect = 3 := by
  sorry

end NUMINAMATH_CALUDE_terrys_test_score_l3061_306107


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3061_306171

-- Problem 1
theorem problem_1 : 3 + (-1) - (-3) + 2 = 10 := by sorry

-- Problem 2
theorem problem_2 : 12 + |(-6)| - (-8) * 3 = 42 := by sorry

-- Problem 3
theorem problem_3 : (2/3 - 1/4 - 3/8) * 24 = 1 := by sorry

-- Problem 4
theorem problem_4 : -1^2021 - (-3 * (2/3)^2 - 4/3 / 2^2) = 2/3 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3061_306171


namespace NUMINAMATH_CALUDE_solve_complex_equation_l3061_306191

theorem solve_complex_equation : 
  ∃ x : ℤ, x - (28 - (37 - (15 - 18))) = 57 ∧ x = 69 := by
  sorry

end NUMINAMATH_CALUDE_solve_complex_equation_l3061_306191


namespace NUMINAMATH_CALUDE_tan_315_eq_neg_one_l3061_306193

/-- Prove that the tangent of 315 degrees is equal to -1 -/
theorem tan_315_eq_neg_one : Real.tan (315 * π / 180) = -1 := by
  sorry


end NUMINAMATH_CALUDE_tan_315_eq_neg_one_l3061_306193


namespace NUMINAMATH_CALUDE_first_five_terms_sum_l3061_306158

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem first_five_terms_sum :
  let a : ℚ := 1
  let r : ℚ := 1/2
  let n : ℕ := 5
  geometric_series_sum a r n = 31/16 := by
sorry

end NUMINAMATH_CALUDE_first_five_terms_sum_l3061_306158


namespace NUMINAMATH_CALUDE_freds_dimes_l3061_306175

/-- Fred's dime problem -/
theorem freds_dimes (initial_dimes borrowed_dimes : ℕ) 
  (h1 : initial_dimes = 7)
  (h2 : borrowed_dimes = 3) :
  initial_dimes - borrowed_dimes = 4 := by
  sorry

end NUMINAMATH_CALUDE_freds_dimes_l3061_306175


namespace NUMINAMATH_CALUDE_pages_copied_for_15_dollars_l3061_306199

/-- The number of pages that can be copied given a certain amount of money and cost per page. -/
def pages_copied (total_money : ℚ) (cost_per_page : ℚ) : ℚ :=
  (total_money * 100) / cost_per_page

/-- Theorem stating that with $15 and a cost of 3 cents per page, 500 pages can be copied. -/
theorem pages_copied_for_15_dollars : 
  pages_copied 15 3 = 500 := by
  sorry

end NUMINAMATH_CALUDE_pages_copied_for_15_dollars_l3061_306199


namespace NUMINAMATH_CALUDE_sector_area_l3061_306120

/-- Given a circular sector with circumference 6cm and central angle 1 radian, 
    prove that its area is 2cm². -/
theorem sector_area (circumference : ℝ) (central_angle : ℝ) (area : ℝ) : 
  circumference = 6 → central_angle = 1 → area = 2 := by sorry

end NUMINAMATH_CALUDE_sector_area_l3061_306120


namespace NUMINAMATH_CALUDE_ratio_problem_l3061_306174

/-- Given two positive integers A and B, where A < B, if A = 36 and LCM(A, B) = 180, then A:B = 1:5 -/
theorem ratio_problem (A B : ℕ) (h1 : 0 < A) (h2 : A < B) (h3 : A = 36) (h4 : Nat.lcm A B = 180) :
  A * 5 = B * 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3061_306174


namespace NUMINAMATH_CALUDE_maxim_method_correct_only_for_24_l3061_306109

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h_tens : tens ≤ 9
  h_ones : ones ≤ 9 ∧ ones ≥ 1

/-- Maxim's division method -/
def maximMethod (A : ℚ) (N : TwoDigitNumber) : ℚ :=
  A / (N.tens + N.ones : ℚ) - A / (N.tens * N.ones : ℚ)

/-- The theorem stating that 24 is the only two-digit number for which Maxim's method works -/
theorem maxim_method_correct_only_for_24 :
  ∀ (N : TwoDigitNumber),
    (∀ (A : ℚ), maximMethod A N = A / (10 * N.tens + N.ones : ℚ)) ↔
    (N.tens = 2 ∧ N.ones = 4) :=
sorry

end NUMINAMATH_CALUDE_maxim_method_correct_only_for_24_l3061_306109


namespace NUMINAMATH_CALUDE_find_number_l3061_306103

theorem find_number : ∃ x : ℝ, (x / 23 - 67) * 2 = 102 ∧ x = 2714 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3061_306103


namespace NUMINAMATH_CALUDE_range_of_a_l3061_306194

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B (a : ℝ) : Set ℝ := {x | x < a}

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (A ∪ B a = {x | x < 1}) ↔ (-1 < a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3061_306194


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3061_306124

theorem partial_fraction_decomposition :
  ∃ (P Q : ℝ), P = 5.5 ∧ Q = 1.5 ∧
  ∀ x : ℝ, x ≠ 12 → x ≠ -4 →
    (7 * x + 4) / (x^2 - 8*x - 48) = P / (x - 12) + Q / (x + 4) :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3061_306124


namespace NUMINAMATH_CALUDE_evaluate_expression_l3061_306185

theorem evaluate_expression (x : ℝ) (h : 3 * x - 2 = 13) : 6 * x - 4 = 26 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3061_306185


namespace NUMINAMATH_CALUDE_problem_statement_l3061_306159

theorem problem_statement (a c : ℤ) : 
  (∃ (x : ℤ), x^2 = 2*a - 1 ∧ (x = 3 ∨ x = -3)) → 
  c = ⌊Real.sqrt 17⌋ → 
  a + c = 9 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3061_306159


namespace NUMINAMATH_CALUDE_average_length_of_writing_instruments_l3061_306197

theorem average_length_of_writing_instruments :
  let pen_length : ℝ := 20
  let pencil_length : ℝ := 16
  let number_of_instruments : ℕ := 2
  (pen_length + pencil_length) / number_of_instruments = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_average_length_of_writing_instruments_l3061_306197


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_equals_four_l3061_306121

theorem sum_of_x_and_y_equals_four (x y : ℝ) (i : ℂ) (h : i^2 = -1) 
  (eq : y + (2 - x) * i = 1 - i) : x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_equals_four_l3061_306121


namespace NUMINAMATH_CALUDE_six_people_eight_chairs_two_restricted_l3061_306122

/-- The number of ways to arrange n people in r chairs -/
def arrangements (n r : ℕ) : ℕ := n.factorial

/-- The number of ways to choose r chairs from n chairs -/
def chair_selections (n r : ℕ) : ℕ := n.choose r

/-- The number of ways to seat people in chairs with restrictions -/
def seating_arrangements (total_chairs people : ℕ) (restricted_pairs : ℕ) : ℕ :=
  (chair_selections total_chairs people - restricted_pairs) * arrangements people people

theorem six_people_eight_chairs_two_restricted : 
  seating_arrangements 8 6 30 = 18720 := by
  sorry

end NUMINAMATH_CALUDE_six_people_eight_chairs_two_restricted_l3061_306122


namespace NUMINAMATH_CALUDE_guppies_ratio_l3061_306105

/-- The number of guppies Haylee has -/
def haylee_guppies : ℕ := 36

/-- The number of guppies Jose has -/
def jose_guppies : ℕ := haylee_guppies / 2

/-- The number of guppies Charliz has -/
def charliz_guppies : ℕ := jose_guppies / 3

/-- The total number of guppies all four friends have -/
def total_guppies : ℕ := 84

/-- The number of guppies Nicolai has -/
def nicolai_guppies : ℕ := total_guppies - (haylee_guppies + jose_guppies + charliz_guppies)

/-- Theorem stating that the ratio of Nicolai's guppies to Charliz's guppies is 4:1 -/
theorem guppies_ratio : nicolai_guppies / charliz_guppies = 4 := by sorry

end NUMINAMATH_CALUDE_guppies_ratio_l3061_306105


namespace NUMINAMATH_CALUDE_expansion_coefficients_theorem_l3061_306149

def binomial_expansion (x y : ℤ) (n : ℕ) := (x + y)^n

def max_coefficient (x y : ℤ) (n : ℕ) : ℕ := Nat.choose n (n / 2)

def second_largest_coefficient (x y : ℤ) (n : ℕ) : ℕ :=
  max (Nat.choose n ((n + 1) / 2)) (Nat.choose n ((n - 1) / 2))

theorem expansion_coefficients_theorem :
  let x : ℤ := 2
  let y : ℤ := 8
  let n : ℕ := 8
  max_coefficient x y n = 70 ∧
  second_largest_coefficient x y n = 1792 ∧
  (second_largest_coefficient x y n : ℚ) / (max_coefficient x y n : ℚ) = 128 / 5 := by
  sorry

#check expansion_coefficients_theorem

end NUMINAMATH_CALUDE_expansion_coefficients_theorem_l3061_306149


namespace NUMINAMATH_CALUDE_units_digit_of_product_l3061_306143

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_product :
  units_digit (27 * 46) = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l3061_306143


namespace NUMINAMATH_CALUDE_triangle_inequality_condition_l3061_306130

theorem triangle_inequality_condition (k l : ℝ) :
  (∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b →
    k * a^2 + l * b^2 > c^2) ↔
  k * l ≥ k + l ∧ k > 1 ∧ l > 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_condition_l3061_306130


namespace NUMINAMATH_CALUDE_inequality_range_inequality_solution_l3061_306129

-- Part 1
theorem inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + (1 - a) * x + a - 2 ≥ -2) ↔ a ∈ Set.Ici (1/3) :=
sorry

-- Part 2
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then { x | x < 1 }
  else if a > 0 then { x | -1/a < x ∧ x < 1 }
  else if a = -1 then { x | x ≠ 1 }
  else if a < -1 then { x | x > 1 ∨ x < -1/a }
  else { x | x < 1 ∨ x > -1/a }

theorem inequality_solution (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ a * x^2 + (1 - a) * x + a - 2 < a - 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_inequality_solution_l3061_306129


namespace NUMINAMATH_CALUDE_negation_of_existence_l3061_306106

theorem negation_of_existence (a : ℝ) :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ 2^x₀ * (x₀ - a) > 1) ↔
  (∀ x : ℝ, x > 0 → 2^x * (x - a) ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l3061_306106


namespace NUMINAMATH_CALUDE_eight_stairs_climb_ways_l3061_306182

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

end NUMINAMATH_CALUDE_eight_stairs_climb_ways_l3061_306182


namespace NUMINAMATH_CALUDE_plant_arrangement_count_l3061_306116

theorem plant_arrangement_count : ℕ := by
  -- Define the number of each type of plant
  let basil_count : ℕ := 4
  let tomato_count : ℕ := 4
  let pepper_count : ℕ := 2

  -- Define the total number of groups (basil plants + tomato group + pepper group)
  let total_groups : ℕ := basil_count + 2

  -- Calculate the number of ways to arrange the groups
  let group_arrangements : ℕ := Nat.factorial total_groups

  -- Calculate the number of ways to arrange plants within their groups
  let tomato_arrangements : ℕ := Nat.factorial tomato_count
  let pepper_arrangements : ℕ := Nat.factorial pepper_count

  -- Calculate the total number of arrangements
  let total_arrangements : ℕ := group_arrangements * tomato_arrangements * pepper_arrangements

  -- Prove that the total number of arrangements is 34560
  have h : total_arrangements = 34560 := by sorry

  exact 34560

end NUMINAMATH_CALUDE_plant_arrangement_count_l3061_306116


namespace NUMINAMATH_CALUDE_prob_not_greater_than_four_is_two_thirds_l3061_306169

/-- A die is represented as a finite type with 6 elements -/
inductive Die : Type
  | one | two | three | four | five | six

/-- The probability of rolling a number not greater than 4 on a six-sided die -/
def prob_not_greater_than_four : ℚ :=
  (Finset.filter (fun x => x ≤ 4) (Finset.range 6)).card /
  (Finset.range 6).card

/-- Theorem stating that the probability of rolling a number not greater than 4 
    on a six-sided die is 2/3 -/
theorem prob_not_greater_than_four_is_two_thirds :
  prob_not_greater_than_four = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_prob_not_greater_than_four_is_two_thirds_l3061_306169


namespace NUMINAMATH_CALUDE_probability_white_and_red_l3061_306134

/-- The probability of drawing one white ball and one red ball from a box 
    containing 7 white balls, 8 black balls, and 1 red ball, 
    when two balls are drawn at random. -/
theorem probability_white_and_red (white : ℕ) (black : ℕ) (red : ℕ) : 
  white = 7 → black = 8 → red = 1 → 
  (white * red : ℚ) / (Nat.choose (white + black + red) 2) = 7 / 120 := by
  sorry

end NUMINAMATH_CALUDE_probability_white_and_red_l3061_306134


namespace NUMINAMATH_CALUDE_gcd_of_225_and_135_l3061_306153

theorem gcd_of_225_and_135 : Nat.gcd 225 135 = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_225_and_135_l3061_306153


namespace NUMINAMATH_CALUDE_value_of_a_l3061_306148

def A (a : ℝ) : Set ℝ := {a + 2, 2 * a^2 + a}

theorem value_of_a : ∀ a : ℝ, 3 ∈ A a → a = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3061_306148


namespace NUMINAMATH_CALUDE_mass_B13N3O12H12_value_l3061_306141

/-- The mass in grams of 12 moles of Trinitride dodecahydroxy tridecaborate (B13N3O12H12) -/
def mass_B13N3O12H12 : ℝ :=
  let atomic_mass_B : ℝ := 10.81
  let atomic_mass_N : ℝ := 14.01
  let atomic_mass_O : ℝ := 16.00
  let atomic_mass_H : ℝ := 1.01
  let molar_mass : ℝ := 13 * atomic_mass_B + 3 * atomic_mass_N + 12 * atomic_mass_O + 12 * atomic_mass_H
  12 * molar_mass

/-- Theorem stating that the mass of 12 moles of B13N3O12H12 is 4640.16 grams -/
theorem mass_B13N3O12H12_value : mass_B13N3O12H12 = 4640.16 := by
  sorry

end NUMINAMATH_CALUDE_mass_B13N3O12H12_value_l3061_306141


namespace NUMINAMATH_CALUDE_investment_problem_l3061_306167

theorem investment_problem (total_interest : ℝ) (amount_at_11_percent : ℝ) :
  total_interest = 0.0975 →
  amount_at_11_percent = 3750 →
  ∃ (total_amount : ℝ) (amount_at_9_percent : ℝ),
    total_amount = amount_at_9_percent + amount_at_11_percent ∧
    0.09 * amount_at_9_percent + 0.11 * amount_at_11_percent = total_interest * total_amount ∧
    total_amount = 10000 :=
by sorry

end NUMINAMATH_CALUDE_investment_problem_l3061_306167


namespace NUMINAMATH_CALUDE_no_solution_system_l3061_306183

theorem no_solution_system : ¬∃ (x y : ℝ), 
  (x^3 + x + y + 1 = 0) ∧ 
  (y*x^2 + x + y = 0) ∧ 
  (y^2 + y - x^2 + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_system_l3061_306183


namespace NUMINAMATH_CALUDE_farm_water_consumption_l3061_306140

/-- Calculates the total weekly water consumption for Mr. Reyansh's farm animals -/
theorem farm_water_consumption : 
  let num_cows : ℕ := 40
  let num_goats : ℕ := 25
  let num_pigs : ℕ := 30
  let cow_water : ℕ := 80
  let goat_water : ℕ := cow_water / 2
  let pig_water : ℕ := cow_water / 3
  let num_sheep : ℕ := num_cows * 10
  let sheep_water : ℕ := cow_water / 4
  let daily_consumption : ℕ := 
    num_cows * cow_water + 
    num_goats * goat_water + 
    num_pigs * pig_water + 
    num_sheep * sheep_water
  let weekly_consumption : ℕ := daily_consumption * 7
  weekly_consumption = 91000 := by
  sorry

end NUMINAMATH_CALUDE_farm_water_consumption_l3061_306140


namespace NUMINAMATH_CALUDE_sum_of_cubic_equations_l3061_306125

theorem sum_of_cubic_equations (x y : ℝ) 
  (hx : (x - 1)^3 + 1997*(x - 1) = -1)
  (hy : (y - 1)^3 + 1997*(y - 1) = 1) :
  x + y = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubic_equations_l3061_306125


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3061_306132

theorem cubic_root_sum (c d : ℝ) : 
  (Complex.I * Real.sqrt 2 + 2 : ℂ) ^ 3 + c * (Complex.I * Real.sqrt 2 + 2) + d = 0 →
  c + d = 14 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3061_306132


namespace NUMINAMATH_CALUDE_number_calculation_l3061_306126

theorem number_calculation (N : ℝ) : 
  0.2 * (|(-0.05)|^3 * 0.35 * 0.7 * N) = 182.7 → N = 20880000 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l3061_306126


namespace NUMINAMATH_CALUDE_function_inequality_l3061_306189

theorem function_inequality (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h_ineq : ∀ x, (x - 1) * deriv f x ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3061_306189


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3061_306139

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  s : ℕ → ℝ  -- The sum of the first n terms
  second_term : a 2 = 4
  sum_formula : ∀ n : ℕ, s n = n^2 + c * n
  c : ℝ       -- The constant in the sum formula

/-- Theorem stating the properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  seq.c = 1 ∧ ∀ n : ℕ, seq.a n = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3061_306139


namespace NUMINAMATH_CALUDE_discount_rate_for_profit_margin_l3061_306150

/-- Proves that a 20% discount rate maintains a 20% profit margin for a toy gift box. -/
theorem discount_rate_for_profit_margin :
  let cost_price : ℝ := 160
  let marked_price : ℝ := 240
  let profit_margin : ℝ := 0.2
  let discount_rate : ℝ := 0.2
  let discounted_price : ℝ := marked_price * (1 - discount_rate)
  let profit : ℝ := discounted_price - cost_price
  profit / cost_price = profit_margin :=
by
  sorry

#check discount_rate_for_profit_margin

end NUMINAMATH_CALUDE_discount_rate_for_profit_margin_l3061_306150


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_similar_triangle_perimeter_proof_l3061_306145

/-- Given an isosceles triangle with two sides of 15 inches and one side of 8 inches,
    a similar triangle with the longest side of 45 inches has a perimeter of 114 inches. -/
theorem similar_triangle_perimeter : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun original_long original_short similar_long perimeter =>
    original_long = 15 →
    original_short = 8 →
    similar_long = 45 →
    perimeter = similar_long + similar_long + (similar_long / original_long * original_short) →
    perimeter = 114

/-- Proof of the theorem -/
theorem similar_triangle_perimeter_proof :
  similar_triangle_perimeter 15 8 45 114 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_similar_triangle_perimeter_proof_l3061_306145


namespace NUMINAMATH_CALUDE_positive_t_value_l3061_306180

theorem positive_t_value (a b : ℂ) (t : ℝ) (h1 : a * b = t - 3 * Complex.I) 
  (h2 : Complex.abs a = 3) (h3 : Complex.abs b = 5) : 
  t > 0 → t = 6 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_positive_t_value_l3061_306180


namespace NUMINAMATH_CALUDE_consecutive_group_probability_l3061_306128

def num_green : ℕ := 4
def num_orange : ℕ := 3
def num_blue : ℕ := 5
def total_pencils : ℕ := num_green + num_orange + num_blue

def probability_consecutive_groups : ℚ :=
  (Nat.factorial 3 * Nat.factorial num_green * Nat.factorial num_orange * Nat.factorial num_blue) /
  Nat.factorial total_pencils

theorem consecutive_group_probability :
  probability_consecutive_groups = 1 / 4620 :=
sorry

end NUMINAMATH_CALUDE_consecutive_group_probability_l3061_306128


namespace NUMINAMATH_CALUDE_distribute_5_3_l3061_306154

/-- The number of ways to distribute n distinct objects into k distinct containers,
    with each container receiving at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct objects into 3 distinct containers,
    with each container receiving at least one object, is 150. -/
theorem distribute_5_3 : distribute 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_5_3_l3061_306154


namespace NUMINAMATH_CALUDE_inequality_proof_l3061_306100

theorem inequality_proof (x y : ℝ) (h : x ≠ 0 ∨ y ≠ 0) :
  (x + y) / (x^2 - x*y + y^2) ≤ (2 * Real.sqrt 2) / Real.sqrt (x^2 + y^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3061_306100


namespace NUMINAMATH_CALUDE_adoption_fee_calculation_l3061_306186

theorem adoption_fee_calculation (james_payment : ℝ) (friend_percentage : ℝ) : 
  james_payment = 150 → friend_percentage = 0.25 → 
  ∃ (total_fee : ℝ), total_fee = 200 ∧ james_payment = (1 - friend_percentage) * total_fee :=
sorry

end NUMINAMATH_CALUDE_adoption_fee_calculation_l3061_306186


namespace NUMINAMATH_CALUDE_zeros_product_lower_bound_l3061_306146

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2 + x

theorem zeros_product_lower_bound {a : ℝ} {x₁ x₂ : ℝ} 
  (h₁ : f a x₁ = 0)
  (h₂ : f a x₂ = 0)
  (h₃ : x₂ > 2 * x₁)
  (h₄ : x₁ > 0)
  (h₅ : x₂ > 0) :
  x₁ * x₂ > 8 / Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_zeros_product_lower_bound_l3061_306146


namespace NUMINAMATH_CALUDE_resort_worker_tips_l3061_306187

theorem resort_worker_tips (total_months : ℕ) (specific_month_multiplier : ℕ) :
  total_months = 7 ∧ specific_month_multiplier = 10 →
  (specific_month_multiplier : ℚ) / ((total_months - 1 : ℕ) + specific_month_multiplier : ℚ) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_resort_worker_tips_l3061_306187


namespace NUMINAMATH_CALUDE_video_upvotes_l3061_306173

theorem video_upvotes (up_to_down_ratio : Rat) (down_votes : ℕ) (up_votes : ℕ) : 
  up_to_down_ratio = 9 / 2 → down_votes = 4 → up_votes = 18 := by
  sorry

end NUMINAMATH_CALUDE_video_upvotes_l3061_306173


namespace NUMINAMATH_CALUDE_tax_rate_calculation_l3061_306181

/-- Represents the tax calculation for a citizen in Country X --/
def TaxCalculation (income : ℝ) (totalTax : ℝ) (baseRate : ℝ) (baseIncome : ℝ) : Prop :=
  income > baseIncome ∧
  totalTax = baseRate * baseIncome + 
    ((income - baseIncome) * (totalTax - baseRate * baseIncome) / (income - baseIncome))

/-- Theorem statement for the tax calculation problem --/
theorem tax_rate_calculation :
  ∀ (income : ℝ) (totalTax : ℝ),
  TaxCalculation income totalTax 0.15 40000 →
  income = 50000 →
  totalTax = 8000 →
  (totalTax - 0.15 * 40000) / (income - 40000) = 0.20 := by
  sorry


end NUMINAMATH_CALUDE_tax_rate_calculation_l3061_306181


namespace NUMINAMATH_CALUDE_tan_pi_minus_alpha_l3061_306178

open Real

theorem tan_pi_minus_alpha (α : ℝ) :
  tan (π - α) = 3/4 →
  π/2 < α ∧ α < π →
  1 / (sin ((π + α)/2) * sin ((π - α)/2)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_minus_alpha_l3061_306178


namespace NUMINAMATH_CALUDE_underdog_wins_probability_l3061_306165

def best_of_five_probability (p : ℚ) : ℚ :=
  (p^5) + 5 * (p^4) * (1 - p) + 10 * (p^3) * ((1 - p)^2)

theorem underdog_wins_probability :
  best_of_five_probability (1/3) = 17/81 := by
  sorry

end NUMINAMATH_CALUDE_underdog_wins_probability_l3061_306165


namespace NUMINAMATH_CALUDE_even_function_implies_f_2_equals_3_l3061_306147

def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x - a)

theorem even_function_implies_f_2_equals_3 (a : ℝ) 
  (h : ∀ x, f a x = f a (-x)) : 
  f a 2 = 3 := by
sorry

end NUMINAMATH_CALUDE_even_function_implies_f_2_equals_3_l3061_306147


namespace NUMINAMATH_CALUDE_premium_percentage_is_twenty_percent_l3061_306102

/-- Calculates the premium percentage on shares given investment details. -/
def calculate_premium_percentage (total_investment : ℚ) (face_value : ℚ) (dividend_rate : ℚ) (dividend_received : ℚ) : ℚ :=
  let num_shares := dividend_received / (dividend_rate * face_value / 100)
  let share_cost := total_investment / num_shares
  (share_cost - face_value) / face_value * 100

/-- Proves that the premium percentage is 20% given the specified conditions. -/
theorem premium_percentage_is_twenty_percent :
  let total_investment : ℚ := 14400
  let face_value : ℚ := 100
  let dividend_rate : ℚ := 5
  let dividend_received : ℚ := 600
  calculate_premium_percentage total_investment face_value dividend_rate dividend_received = 20 := by
  sorry

end NUMINAMATH_CALUDE_premium_percentage_is_twenty_percent_l3061_306102


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3061_306113

theorem trigonometric_identity (x y : ℝ) :
  Real.sin (x - y + π/6) * Real.cos (y + π/6) + Real.cos (x - y + π/6) * Real.sin (y + π/6) = Real.sin (x + π/3) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3061_306113


namespace NUMINAMATH_CALUDE_div_remainder_theorem_l3061_306168

theorem div_remainder_theorem : 
  ∃ k : ℕ, 3^19 = k * 1162261460 + 7 :=
sorry

end NUMINAMATH_CALUDE_div_remainder_theorem_l3061_306168


namespace NUMINAMATH_CALUDE_integer_roots_cubic_l3061_306155

theorem integer_roots_cubic (b : ℤ) : 
  (∃ x : ℤ, x^3 - 2*x^2 + b*x + 6 = 0) ↔ b ∈ ({-25, -7, -5, 3, 13, 47} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_integer_roots_cubic_l3061_306155


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_ratio_l3061_306198

/-- An equilateral triangle with an inscribed circle -/
structure EquilateralTriangleWithInscribedCircle where
  /-- The side length of the equilateral triangle -/
  side_length : ℝ
  /-- The radius of the inscribed circle -/
  circle_radius : ℝ
  /-- The points of tangency are on the sides of the triangle -/
  tangency_points_on_sides : True

/-- The ratio of the inscribed circle's radius to the triangle's side length is 1/16 -/
theorem inscribed_circle_radius_ratio 
  (triangle : EquilateralTriangleWithInscribedCircle) : 
  triangle.circle_radius / triangle.side_length = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_ratio_l3061_306198


namespace NUMINAMATH_CALUDE_sandras_age_l3061_306127

/-- Sandra's age problem -/
theorem sandras_age :
  ∀ (sandra_age : ℕ),
  (sandra_age - 3 = 3 * (14 - 3)) →
  sandra_age = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_sandras_age_l3061_306127


namespace NUMINAMATH_CALUDE_parabola_vertex_l3061_306179

/-- The vertex of the parabola y = 2x^2 + 16x + 50 is (-4, 18) -/
theorem parabola_vertex (x y : ℝ) : 
  y = 2 * x^2 + 16 * x + 50 → (∃ m n : ℝ, m = -4 ∧ n = 18 ∧ 
    ∀ x, y = 2 * (x - m)^2 + n) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3061_306179


namespace NUMINAMATH_CALUDE_relay_race_time_l3061_306152

-- Define the runners and their properties
structure Runner where
  base_time : ℝ
  obstacle_time : ℝ
  handicap : ℝ

def rhonda : Runner :=
  { base_time := 24
  , obstacle_time := 2
  , handicap := 0.95 }

def sally : Runner :=
  { base_time := 26
  , obstacle_time := 5
  , handicap := 0.90 }

def diane : Runner :=
  { base_time := 21
  , obstacle_time := 21 * 0.1
  , handicap := 1.05 }

-- Calculate the final time for a runner
def final_time (runner : Runner) : ℝ :=
  (runner.base_time + runner.obstacle_time) * runner.handicap

-- Calculate the total time for the relay race
def relay_time : ℝ :=
  final_time rhonda + final_time sally + final_time diane

-- Theorem statement
theorem relay_race_time : relay_time = 76.855 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_time_l3061_306152


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l3061_306136

theorem divisibility_implies_equality (a b : ℕ) :
  (∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, (a^(n+1) + b^(n+1)) % (a^n + b^n) = 0) →
  a = b :=
sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l3061_306136


namespace NUMINAMATH_CALUDE_james_oreos_l3061_306119

theorem james_oreos (total : ℕ) (jordan : ℕ) (james : ℕ) : 
  total = 52 → 
  james = 7 + 4 * jordan → 
  total = james + jordan → 
  james = 43 := by
sorry

end NUMINAMATH_CALUDE_james_oreos_l3061_306119


namespace NUMINAMATH_CALUDE_power_sum_problem_l3061_306101

theorem power_sum_problem (a b x y : ℝ) 
  (h1 : a * x + b * y = 3)
  (h2 : a * x^2 + b * y^2 = 7)
  (h3 : a * x^3 + b * y^3 = 16)
  (h4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_problem_l3061_306101


namespace NUMINAMATH_CALUDE_march_greatest_drop_l3061_306192

/-- Represents the months of the year --/
inductive Month
| January
| February
| March
| April
| May
| June

/-- The price change for each month --/
def price_change : Month → ℝ
| Month.January => -0.5
| Month.February => 1.5
| Month.March => -3.0
| Month.April => 2.0
| Month.May => -1.0
| Month.June => -2.5

/-- The fixed transaction fee --/
def transaction_fee : ℝ := 1.0

/-- The adjusted price change after applying the transaction fee --/
def adjusted_price_change (m : Month) : ℝ :=
  price_change m - transaction_fee

/-- Theorem stating that March has the greatest monthly drop --/
theorem march_greatest_drop :
  ∀ m : Month, m ≠ Month.March →
  adjusted_price_change Month.March ≤ adjusted_price_change m :=
by sorry

end NUMINAMATH_CALUDE_march_greatest_drop_l3061_306192


namespace NUMINAMATH_CALUDE_root_product_sum_l3061_306104

theorem root_product_sum (a b c : ℝ) : 
  (3 * a^3 - 3 * a^2 + 11 * a - 8 = 0) →
  (3 * b^3 - 3 * b^2 + 11 * b - 8 = 0) →
  (3 * c^3 - 3 * c^2 + 11 * c - 8 = 0) →
  a * b + a * c + b * c = 11/3 := by
sorry

end NUMINAMATH_CALUDE_root_product_sum_l3061_306104


namespace NUMINAMATH_CALUDE_min_value_of_f_l3061_306142

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 - log x

theorem min_value_of_f :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x ∧ f x = (1 + log 2) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3061_306142


namespace NUMINAMATH_CALUDE_existence_of_k_with_n_prime_factors_l3061_306164

theorem existence_of_k_with_n_prime_factors 
  (m n : ℕ+) : 
  ∃ k : ℕ+, ∃ p : Finset ℕ, 
    (∀ x ∈ p, Nat.Prime x) ∧ 
    (Finset.card p ≥ n) ∧ 
    (∀ x ∈ p, x ∣ (2^(k:ℕ) - m)) :=
by
  sorry

end NUMINAMATH_CALUDE_existence_of_k_with_n_prime_factors_l3061_306164


namespace NUMINAMATH_CALUDE_product_remainder_by_10_l3061_306170

theorem product_remainder_by_10 : (8623 * 2475 * 56248 * 1234) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_by_10_l3061_306170


namespace NUMINAMATH_CALUDE_xyz_expression_bounds_l3061_306110

theorem xyz_expression_bounds (x y z : ℝ) 
  (non_neg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (sum_one : x + y + z = 1) : 
  0 ≤ x*y + y*z + z*x - 3*x*y*z ∧ x*y + y*z + z*x - 3*x*y*z ≤ 1/4 := by
sorry

end NUMINAMATH_CALUDE_xyz_expression_bounds_l3061_306110


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3061_306118

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (5 * x₁^2 + 8 * x₁ - 7 = 0) → 
  (5 * x₂^2 + 8 * x₂ - 7 = 0) → 
  (x₁ ≠ x₂) →
  (x₁^2 + x₂^2 = 134/25) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3061_306118


namespace NUMINAMATH_CALUDE_colors_in_box_is_seven_l3061_306135

/-- The number of colors in each color box, given the total number of pencils and people who bought a color box. -/
def colors_per_box (total_pencils : ℕ) (total_people : ℕ) : ℕ :=
  total_pencils / total_people

/-- Theorem stating that the number of colors in each color box is 7, given the problem conditions. -/
theorem colors_in_box_is_seven : 
  let total_people : ℕ := 6  -- Chloe and 5 friends
  let total_pencils : ℕ := 42
  colors_per_box total_pencils total_people = 7 := by
  sorry

#eval colors_per_box 42 6  -- This should output 7

end NUMINAMATH_CALUDE_colors_in_box_is_seven_l3061_306135


namespace NUMINAMATH_CALUDE_max_principals_in_period_l3061_306117

/-- Represents the duration of the period in years -/
def period_duration : ℕ := 8

/-- Represents the duration of a principal's term in years -/
def term_duration : ℕ := 4

/-- Represents the maximum number of non-overlapping terms that can fit within the period -/
def max_principals : ℕ := period_duration / term_duration

theorem max_principals_in_period :
  max_principals = 2 :=
sorry

end NUMINAMATH_CALUDE_max_principals_in_period_l3061_306117


namespace NUMINAMATH_CALUDE_S_inter_T_eq_T_l3061_306138

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem S_inter_T_eq_T : S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_S_inter_T_eq_T_l3061_306138


namespace NUMINAMATH_CALUDE_personal_preference_invalid_l3061_306133

/-- Represents the principles of sample selection --/
structure SampleSelectionPrinciples where
  representativeness : Bool
  randomness : Bool
  adequateSize : Bool

/-- Represents a sample selection method --/
inductive SampleSelectionMethod
  | Random
  | Representative
  | LargeEnough
  | PersonalPreference

/-- Checks if a sample selection method adheres to the principles --/
def isValidMethod (principles : SampleSelectionPrinciples) (method : SampleSelectionMethod) : Prop :=
  match method with
  | .Random => principles.randomness
  | .Representative => principles.representativeness
  | .LargeEnough => principles.adequateSize
  | .PersonalPreference => False

/-- Theorem stating that personal preference is not a valid sample selection method --/
theorem personal_preference_invalid (principles : SampleSelectionPrinciples) :
  ¬(isValidMethod principles SampleSelectionMethod.PersonalPreference) := by
  sorry


end NUMINAMATH_CALUDE_personal_preference_invalid_l3061_306133


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3061_306137

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (2 + I) / (3 - I) → z.im = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3061_306137


namespace NUMINAMATH_CALUDE_ball_probability_l3061_306157

theorem ball_probability (red_balls : ℕ) (white_balls : ℕ) :
  red_balls = 3 →
  (red_balls : ℚ) / (red_balls + white_balls : ℚ) = 3 / 7 →
  white_balls = 4 := by
sorry

end NUMINAMATH_CALUDE_ball_probability_l3061_306157


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3061_306163

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 1) :
  (1 / x + 1 / y) ≥ 5 + 3 * Real.sqrt 3 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3 * y₀ = 1 ∧ 1 / x₀ + 1 / y₀ = 5 + 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3061_306163


namespace NUMINAMATH_CALUDE_logical_equivalences_l3061_306161

theorem logical_equivalences (A B C : Prop) : 
  ((A ∧ (B ∨ C) ↔ (A ∧ B) ∨ (A ∧ C)) ∧ 
   (A ∨ (B ∧ C) ↔ (A ∨ B) ∧ (A ∨ C))) := by
  sorry

end NUMINAMATH_CALUDE_logical_equivalences_l3061_306161


namespace NUMINAMATH_CALUDE_jinho_ribbon_length_l3061_306195

/-- The number of students in Minsu's class -/
def minsu_students : ℕ := 8

/-- The number of students in Jinho's class -/
def jinho_students : ℕ := minsu_students + 1

/-- The total length of ribbon in meters -/
def total_ribbon_m : ℝ := 3.944

/-- The length of ribbon given to each student in Minsu's class in centimeters -/
def ribbon_per_minsu_student_cm : ℝ := 29.05

/-- Conversion factor from meters to centimeters -/
def m_to_cm : ℝ := 100

theorem jinho_ribbon_length :
  let total_ribbon_cm := total_ribbon_m * m_to_cm
  let minsu_total_ribbon_cm := ribbon_per_minsu_student_cm * minsu_students
  let remaining_ribbon_cm := total_ribbon_cm - minsu_total_ribbon_cm
  remaining_ribbon_cm / jinho_students = 18 := by sorry

end NUMINAMATH_CALUDE_jinho_ribbon_length_l3061_306195


namespace NUMINAMATH_CALUDE_root_equation_problem_l3061_306156

theorem root_equation_problem (m r s a b : ℝ) : 
  (a^2 - m*a + 4 = 0) →
  (b^2 - m*b + 4 = 0) →
  ((a^2 + 1/b)^2 - r*(a^2 + 1/b) + s = 0) →
  ((b^2 + 1/a)^2 - r*(b^2 + 1/a) + s = 0) →
  s = m + 16.25 := by
sorry

end NUMINAMATH_CALUDE_root_equation_problem_l3061_306156


namespace NUMINAMATH_CALUDE_polynomial_coefficient_B_value_l3061_306190

theorem polynomial_coefficient_B_value (P Q : ℤ) :
  ∃ (r₁ r₂ r₃ r₄ r₅ : ℕ+),
    (r₁ : ℤ) + r₂ + r₃ + r₄ + r₅ = 14 ∧
    ∀ (x : ℤ), x^5 - 14*x^4 + P*x^3 + 203*x^2 + Q*x + 48 = 
      (x - r₁) * (x - r₂) * (x - r₃) * (x - r₄) * (x - r₅) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_B_value_l3061_306190


namespace NUMINAMATH_CALUDE_triangle_property_l3061_306196

theorem triangle_property (a b c : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_condition : a^3 - b^3 = a^2*b - a*b^2 + a*c^2 - b*c^2) : 
  a = b ∨ a^2 + b^2 = c^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_property_l3061_306196


namespace NUMINAMATH_CALUDE_function_characterization_l3061_306114

theorem function_characterization (f : ℕ → ℕ) :
  (∀ m n : ℕ, (m^2 + f n) ∣ (m * f m + n)) →
  (∀ n : ℕ, f n = n) := by
sorry

end NUMINAMATH_CALUDE_function_characterization_l3061_306114


namespace NUMINAMATH_CALUDE_apps_deleted_l3061_306160

/-- Proves that Dave deleted 8 apps given the initial conditions -/
theorem apps_deleted (initial_apps : ℕ) (remaining_apps : ℕ) : 
  initial_apps = 16 →
  remaining_apps = initial_apps / 2 →
  initial_apps - remaining_apps = 8 := by
sorry

end NUMINAMATH_CALUDE_apps_deleted_l3061_306160


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3061_306131

def S : Set Int := {-1, -2, -3, -4, -5}

theorem max_value_of_expression (a b c d : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S)
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  (∀ (w x y z : Int), w ∈ S → x ∈ S → y ∈ S → z ∈ S →
    w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
    (w^x + y^z : Rat) ≤ 10/9) ∧
  (∃ (w x y z : Int), w ∈ S ∧ x ∈ S ∧ y ∈ S ∧ z ∈ S ∧
    w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    (w^x + y^z : Rat) = 10/9) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3061_306131


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3061_306112

/-- Given an arithmetic sequence {a_n} where a_3 + a_11 = 40, prove that a_6 + a_7 + a_8 = 60 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 3 + a 11 = 40 →                                     -- given condition
  a 6 + a 7 + a 8 = 60 :=                               -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3061_306112


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l3061_306176

/-- The area of a rectangle inscribed in the ellipse x^2/4 + y^2/8 = 1,
    with sides parallel to the coordinate axes and length twice its width -/
theorem inscribed_rectangle_area :
  ∀ (a b : ℝ),
  (a > 0) →
  (b > 0) →
  (a = 2 * b) →
  (a^2 / 4 + b^2 / 8 = 1) →
  4 * a * b = 32 / 3 := by
sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l3061_306176


namespace NUMINAMATH_CALUDE_card_game_combinations_l3061_306108

theorem card_game_combinations : Nat.choose 52 10 = 158200242220 := by sorry

end NUMINAMATH_CALUDE_card_game_combinations_l3061_306108


namespace NUMINAMATH_CALUDE_add_negative_three_l3061_306115

theorem add_negative_three : 2 + (-3) = -1 := by sorry

end NUMINAMATH_CALUDE_add_negative_three_l3061_306115


namespace NUMINAMATH_CALUDE_jesse_banana_sharing_l3061_306144

theorem jesse_banana_sharing :
  ∀ (total_bananas : ℕ) (bananas_per_friend : ℕ) (num_friends : ℕ),
    total_bananas = 21 →
    bananas_per_friend = 7 →
    total_bananas = bananas_per_friend * num_friends →
    num_friends = 3 := by
  sorry

end NUMINAMATH_CALUDE_jesse_banana_sharing_l3061_306144


namespace NUMINAMATH_CALUDE_platform_length_l3061_306151

/-- Given a train of length 1500 meters that takes 120 seconds to cross a tree
    and 160 seconds to cross a platform, prove that the length of the platform is 500 meters. -/
theorem platform_length (train_length : ℝ) (tree_crossing_time : ℝ) (platform_crossing_time : ℝ)
    (h1 : train_length = 1500)
    (h2 : tree_crossing_time = 120)
    (h3 : platform_crossing_time = 160) :
    (train_length / tree_crossing_time) * platform_crossing_time - train_length = 500 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l3061_306151


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l3061_306184

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + x * y) = f x * f (y + 1)

theorem functional_equation_solutions :
  ∀ f : ℝ → ℝ, functional_equation f →
    (∀ x, f x = 0) ∨ (∀ x, f x = 1) ∨ (∀ x, f x = x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l3061_306184


namespace NUMINAMATH_CALUDE_large_pizzas_purchased_l3061_306162

/-- Represents the number of slices in a small pizza -/
def small_pizza_slices : ℕ := 4

/-- Represents the number of slices in a large pizza -/
def large_pizza_slices : ℕ := 8

/-- Represents the number of small pizzas purchased -/
def small_pizzas_purchased : ℕ := 3

/-- Represents the total number of slices consumed by all people -/
def total_slices_consumed : ℕ := 18

/-- Represents the number of slices left over -/
def slices_left_over : ℕ := 10

theorem large_pizzas_purchased :
  ∃ (n : ℕ), n * large_pizza_slices + small_pizzas_purchased * small_pizza_slices =
    total_slices_consumed + slices_left_over ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_large_pizzas_purchased_l3061_306162


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_equals_area_l3061_306166

theorem right_triangle_perimeter_equals_area :
  ∀ a b c : ℕ,
    a ≤ b → b ≤ c →
    a^2 + b^2 = c^2 →
    a + b + c = (a * b) / 2 →
    ((a = 5 ∧ b = 12 ∧ c = 13) ∨ (a = 6 ∧ b = 8 ∧ c = 10)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_equals_area_l3061_306166


namespace NUMINAMATH_CALUDE_b_plus_c_equals_three_l3061_306172

/-- A function f: ℝ → ℝ defined as f(x) = x^3 + bx^2 + cx -/
def f (b c : ℝ) : ℝ → ℝ := λ x ↦ x^3 + b*x^2 + c*x

/-- The derivative of f -/
def f_deriv (b c : ℝ) : ℝ → ℝ := λ x ↦ 3*x^2 + 2*b*x + c

/-- A function g: ℝ → ℝ defined as g(x) = f(x) - f'(x) -/
def g (b c : ℝ) : ℝ → ℝ := λ x ↦ f b c x - f_deriv b c x

/-- A predicate stating that a function is odd -/
def is_odd_function (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

/-- The main theorem -/
theorem b_plus_c_equals_three (b c : ℝ) :
  is_odd_function (g b c) → b + c = 3 := by sorry

end NUMINAMATH_CALUDE_b_plus_c_equals_three_l3061_306172


namespace NUMINAMATH_CALUDE_math_interest_group_size_l3061_306188

theorem math_interest_group_size (total_cards : ℕ) : 
  (total_cards = 182) → 
  (∃ n : ℕ, n * (n - 1) = total_cards ∧ n > 0) → 
  (∃ n : ℕ, n * (n - 1) = total_cards ∧ n = 14) :=
by
  sorry

end NUMINAMATH_CALUDE_math_interest_group_size_l3061_306188


namespace NUMINAMATH_CALUDE_race_completion_time_l3061_306111

theorem race_completion_time (walking_time jogging_time total_time : ℕ) : 
  walking_time = 9 →
  jogging_time * 3 = walking_time * 4 →
  total_time = walking_time + jogging_time →
  total_time = 21 := by
sorry

end NUMINAMATH_CALUDE_race_completion_time_l3061_306111


namespace NUMINAMATH_CALUDE_jakes_allowance_l3061_306123

/-- 
Given:
- An amount x (in cents)
- One-quarter of x can buy 5 items
- Each item costs 20 cents

Prove that x = 400 cents ($4.00)
-/
theorem jakes_allowance (x : ℕ) 
  (h1 : x / 4 = 5 * 20) : 
  x = 400 := by
sorry

end NUMINAMATH_CALUDE_jakes_allowance_l3061_306123
