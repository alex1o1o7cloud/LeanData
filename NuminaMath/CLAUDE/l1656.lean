import Mathlib

namespace NUMINAMATH_CALUDE_function_properties_l1656_165635

/-- Given function f with properties as described -/
def f (x : ℝ) : ℝ := sorry

/-- ω is a positive real number -/
def ω : ℝ := sorry

/-- φ is a real number between 0 and π -/
def φ : ℝ := sorry

theorem function_properties (x α : ℝ) :
  ω > 0 ∧
  0 ≤ φ ∧ φ ≤ π ∧
  (∀ x, f x = Real.sin (ω * x + φ)) ∧
  (∀ x, f x = f (-x)) ∧
  (∃ k : ℤ, ∀ x, f (x + π) = f x) ∧
  Real.sin α + f α = 2/3 →
  (f = Real.cos) ∧
  ((Real.sqrt 2 * Real.sin (2*α - π/4) + 1) / (1 + Real.tan α) = 5/9) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1656_165635


namespace NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_one_or_neg_six_l1656_165658

/-- Two lines in the plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a ≠ 0 ∧ l2.a ≠ 0

/-- The main theorem -/
theorem lines_parallel_iff_m_eq_one_or_neg_six (m : ℝ) :
  let l1 : Line2D := ⟨m, 3, -6⟩
  let l2 : Line2D := ⟨2, 5 + m, 2⟩
  parallel l1 l2 ↔ m = 1 ∨ m = -6 := by
  sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_m_eq_one_or_neg_six_l1656_165658


namespace NUMINAMATH_CALUDE_basketball_handshakes_l1656_165642

/-- Calculates the total number of handshakes in a basketball game scenario -/
theorem basketball_handshakes (team_size : ℕ) (num_teams : ℕ) (num_referees : ℕ) : 
  team_size = 5 → num_teams = 2 → num_referees = 2 →
  (team_size * team_size) + (team_size * num_teams * num_referees) = 45 := by
  sorry

end NUMINAMATH_CALUDE_basketball_handshakes_l1656_165642


namespace NUMINAMATH_CALUDE_index_card_area_l1656_165689

theorem index_card_area (l w : ℕ) (h1 : l = 3) (h2 : w = 7) : 
  (∃ (a b : ℕ), (l - a) * (w - b) = 10 ∧ a + b = 3) → 
  (l - 1) * (w - 2) = 10 := by
sorry

end NUMINAMATH_CALUDE_index_card_area_l1656_165689


namespace NUMINAMATH_CALUDE_min_period_tan_2x_l1656_165664

/-- The minimum positive period of the function y = tan 2x is π/2 -/
theorem min_period_tan_2x : 
  let f : ℝ → ℝ := λ x => Real.tan (2 * x)
  ∃ p : ℝ, p > 0 ∧ (∀ x : ℝ, f (x + p) = f x) ∧ 
    (∀ q : ℝ, q > 0 → (∀ x : ℝ, f (x + q) = f x) → p ≤ q) ∧
    p = π / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_period_tan_2x_l1656_165664


namespace NUMINAMATH_CALUDE_least_eight_binary_digits_l1656_165631

def binary_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

theorem least_eight_binary_digits : 
  ∀ k : ℕ, k > 0 → (binary_digits k ≥ 8 → k ≥ 128) ∧ binary_digits 128 = 8 :=
by sorry

end NUMINAMATH_CALUDE_least_eight_binary_digits_l1656_165631


namespace NUMINAMATH_CALUDE_average_of_5_8_N_l1656_165633

theorem average_of_5_8_N (N : ℝ) (h : 8 < N ∧ N < 20) : 
  let avg := (5 + 8 + N) / 3
  avg = 8 ∨ avg = 10 := by
sorry

end NUMINAMATH_CALUDE_average_of_5_8_N_l1656_165633


namespace NUMINAMATH_CALUDE_base_ten_to_base_five_158_l1656_165641

/-- Converts a natural number to its base 5 representation -/
def toBaseFive (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- Checks if a list of digits is a valid base 5 representation -/
def isValidBaseFive (digits : List ℕ) : Prop :=
  digits.all (· < 5) ∧ digits ≠ []

theorem base_ten_to_base_five_158 :
  let base_five_repr := toBaseFive 158
  isValidBaseFive base_five_repr ∧ base_five_repr = [1, 1, 3, 3] := by sorry

end NUMINAMATH_CALUDE_base_ten_to_base_five_158_l1656_165641


namespace NUMINAMATH_CALUDE_gcd_1681_1705_l1656_165679

theorem gcd_1681_1705 : Nat.gcd 1681 1705 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1681_1705_l1656_165679


namespace NUMINAMATH_CALUDE_expression_evaluation_l1656_165670

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 13) + 1 = -x^4 + 3*x^3 - 5*x^2 + 13*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1656_165670


namespace NUMINAMATH_CALUDE_sector_area_l1656_165672

/-- Given a circular sector with central angle α = 60° and arc length l = 6π,
    prove that the area of the sector is 54π. -/
theorem sector_area (α : Real) (l : Real) (h1 : α = 60 * π / 180) (h2 : l = 6 * π) :
  (1 / 2) * l * (l / α) = 54 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1656_165672


namespace NUMINAMATH_CALUDE_fixed_point_of_linear_function_l1656_165634

theorem fixed_point_of_linear_function (k : ℝ) :
  (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_linear_function_l1656_165634


namespace NUMINAMATH_CALUDE_symmetric_function_equality_l1656_165673

-- Define a function that is symmetric with respect to x = 1
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (2 - x) = f x

-- Define the theorem
theorem symmetric_function_equality (f : ℝ → ℝ) (h : SymmetricFunction f) :
  ∃! a : ℝ, f (a - 1) = f 5 ∧ a = 6 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_equality_l1656_165673


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l1656_165654

/-- Proves that a train with given length and speed takes the calculated time to cross a bridge of given length -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 150)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 225) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

#check train_bridge_crossing_time

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l1656_165654


namespace NUMINAMATH_CALUDE_paper_cutting_game_l1656_165655

theorem paper_cutting_game (n : ℕ) (pieces : ℕ) : 
  (pieces = 8 * n + 1) → (pieces = 2009) → (n = 251) := by
  sorry

end NUMINAMATH_CALUDE_paper_cutting_game_l1656_165655


namespace NUMINAMATH_CALUDE_radical_equation_solution_l1656_165684

theorem radical_equation_solution :
  ∃! x : ℝ, x > 9 ∧ Real.sqrt (x - 6 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 9)) - 3 ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_radical_equation_solution_l1656_165684


namespace NUMINAMATH_CALUDE_x_squared_equals_three_l1656_165674

theorem x_squared_equals_three (x : ℝ) (h1 : x > 0) (h2 : Real.sin (Real.arctan x) = x / 2) : x^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_equals_three_l1656_165674


namespace NUMINAMATH_CALUDE_brothers_ages_sum_l1656_165620

theorem brothers_ages_sum (a b c : ℕ) : 
  a = 31 → b = a + 1 → c = b + 1 → a + b + c = 96 := by
  sorry

end NUMINAMATH_CALUDE_brothers_ages_sum_l1656_165620


namespace NUMINAMATH_CALUDE_prob_exactly_two_choose_A_l1656_165618

/-- The number of communities available for housing applications. -/
def num_communities : ℕ := 3

/-- The number of applicants. -/
def num_applicants : ℕ := 4

/-- The number of applicants required to choose community A. -/
def target_applicants : ℕ := 2

/-- The probability of an applicant choosing any specific community. -/
def prob_choose_community : ℚ := 1 / num_communities

/-- The probability that exactly 'target_applicants' out of 'num_applicants' 
    choose community A, given equal probability for each community. -/
theorem prob_exactly_two_choose_A : 
  (Nat.choose num_applicants target_applicants : ℚ) * 
  prob_choose_community ^ target_applicants * 
  (1 - prob_choose_community) ^ (num_applicants - target_applicants) = 8/27 :=
sorry

end NUMINAMATH_CALUDE_prob_exactly_two_choose_A_l1656_165618


namespace NUMINAMATH_CALUDE_regression_line_intercept_l1656_165605

/-- Given a linear regression line ŷ = (1/3)x + a passing through the point (x̄, ȳ),
    where x̄ = 3/8 and ȳ = 5/8, prove that a = 1/2. -/
theorem regression_line_intercept (x_bar y_bar : ℝ) (a : ℝ) 
    (h1 : x_bar = 3/8)
    (h2 : y_bar = 5/8)
    (h3 : y_bar = (1/3) * x_bar + a) : 
  a = 1/2 := by
  sorry

#check regression_line_intercept

end NUMINAMATH_CALUDE_regression_line_intercept_l1656_165605


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l1656_165623

-- Equation 1
theorem equation_one_solution (x : ℚ) : 
  (3 / (2 * x - 2) + 1 / (1 - x) = 3) ↔ (x = 7/6) :=
sorry

-- Equation 2
theorem equation_two_no_solution :
  ¬∃ y : ℚ, y / (y - 1) - 2 / (y^2 - 1) = 1 :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l1656_165623


namespace NUMINAMATH_CALUDE_secretary_typing_orders_l1656_165657

/-- The number of letters to be typed -/
def total_letters : ℕ := 12

/-- The number of the letter that has been typed -/
def typed_letter : ℕ := 10

/-- Calculates the number of possible typing orders for the remaining letters -/
def possible_orders : ℕ :=
  Finset.sum (Finset.range 10) (fun k =>
    Nat.choose 9 k * (k + 1) * (k + 2))

/-- Theorem stating the number of possible typing orders -/
theorem secretary_typing_orders :
  possible_orders = 5166 := by
  sorry

end NUMINAMATH_CALUDE_secretary_typing_orders_l1656_165657


namespace NUMINAMATH_CALUDE_ring_stack_distance_l1656_165645

/-- Represents a stack of metallic rings -/
structure RingStack where
  topDiameter : ℕ
  smallestDiameter : ℕ
  thickness : ℕ

/-- Calculates the total vertical distance of a ring stack -/
def totalVerticalDistance (stack : RingStack) : ℕ :=
  let numRings := (stack.topDiameter - stack.smallestDiameter) / 2 + 1
  let sumDiameters := numRings * (stack.topDiameter + stack.smallestDiameter) / 2
  sumDiameters - numRings + 2 * stack.thickness

/-- Theorem stating the total vertical distance of the given ring stack -/
theorem ring_stack_distance :
  ∀ (stack : RingStack),
    stack.topDiameter = 22 ∧
    stack.smallestDiameter = 4 ∧
    stack.thickness = 1 →
    totalVerticalDistance stack = 122 := by
  sorry


end NUMINAMATH_CALUDE_ring_stack_distance_l1656_165645


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1656_165644

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 96 → s^3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1656_165644


namespace NUMINAMATH_CALUDE_probability_greater_than_two_l1656_165656

def standard_die := Finset.range 6

def favorable_outcomes : Finset Nat :=
  standard_die.filter (λ x => x > 2)

theorem probability_greater_than_two :
  (favorable_outcomes.card : ℚ) / standard_die.card = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_greater_than_two_l1656_165656


namespace NUMINAMATH_CALUDE_ages_when_bella_turns_18_l1656_165695

/-- Given the initial ages and birth years, prove the ages when Bella turns 18 -/
theorem ages_when_bella_turns_18 
  (marianne_age_2000 : ℕ)
  (bella_age_2000 : ℕ)
  (carmen_age_2000 : ℕ)
  (elli_birth_year : ℕ)
  (h1 : marianne_age_2000 = 20)
  (h2 : bella_age_2000 = 8)
  (h3 : carmen_age_2000 = 15)
  (h4 : elli_birth_year = 2003) :
  let year_bella_18 := 2000 + (18 - bella_age_2000)
  (year_bella_18 - 2000 + marianne_age_2000 = 30) ∧ 
  (year_bella_18 - 2000 + carmen_age_2000 = 33) ∧
  (year_bella_18 - elli_birth_year = 15) :=
sorry

end NUMINAMATH_CALUDE_ages_when_bella_turns_18_l1656_165695


namespace NUMINAMATH_CALUDE_max_integer_value_of_function_l1656_165661

theorem max_integer_value_of_function (x : ℝ) : 
  (4*x^2 + 8*x + 5 ≠ 0) → 
  ∃ (y : ℤ), y = 17 ∧ ∀ (z : ℤ), z ≤ (4*x^2 + 8*x + 21) / (4*x^2 + 8*x + 5) → z ≤ y :=
by sorry

end NUMINAMATH_CALUDE_max_integer_value_of_function_l1656_165661


namespace NUMINAMATH_CALUDE_expected_girls_theorem_l1656_165613

/-- The expected number of girls selected as volunteers -/
def expected_girls (total_students : ℕ) (num_girls : ℕ) (num_volunteers : ℕ) : ℚ :=
  (num_girls : ℚ) / (total_students : ℚ) * (num_volunteers : ℚ)

/-- Theorem: The expected number of girls selected as volunteers is 3/4 -/
theorem expected_girls_theorem :
  expected_girls 8 3 2 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_expected_girls_theorem_l1656_165613


namespace NUMINAMATH_CALUDE_area_common_to_translated_triangles_l1656_165691

theorem area_common_to_translated_triangles : 
  let hypotenuse : ℝ := 10
  let translation : ℝ := 2
  let short_leg : ℝ := hypotenuse / 2
  let long_leg : ℝ := short_leg * Real.sqrt 3
  let overlap_height : ℝ := long_leg - translation
  let common_area : ℝ := (1 / 2) * hypotenuse * overlap_height
  common_area = 25 * Real.sqrt 3 - 10 := by
sorry

end NUMINAMATH_CALUDE_area_common_to_translated_triangles_l1656_165691


namespace NUMINAMATH_CALUDE_sum_f_eq_518656_l1656_165616

/-- f(n) is the index of the highest power of 2 which divides n! -/
def f (n : ℕ+) : ℕ := sorry

/-- Sum of f(n) from 1 to 1023 -/
def sum_f : ℕ := sorry

theorem sum_f_eq_518656 : sum_f = 518656 := by sorry

end NUMINAMATH_CALUDE_sum_f_eq_518656_l1656_165616


namespace NUMINAMATH_CALUDE_compute_expression_l1656_165659

theorem compute_expression : 8 * (1 / 3)^3 - 1 = -19 / 27 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1656_165659


namespace NUMINAMATH_CALUDE_trig_inequality_l1656_165693

theorem trig_inequality (x : ℝ) : 1 ≤ Real.sin x ^ 10 + 10 * Real.sin x ^ 2 * Real.cos x ^ 2 + Real.cos x ^ 10 ∧ 
  Real.sin x ^ 10 + 10 * Real.sin x ^ 2 * Real.cos x ^ 2 + Real.cos x ^ 10 ≤ 41 / 16 := by
  sorry

end NUMINAMATH_CALUDE_trig_inequality_l1656_165693


namespace NUMINAMATH_CALUDE_inscribed_squares_area_ratio_l1656_165600

/-- The ratio of the area of a square inscribed in a quarter-circle to the area of a square inscribed in a full circle, both with radius r, is 1/4. -/
theorem inscribed_squares_area_ratio (r : ℝ) (hr : r > 0) :
  let s1 := r / Real.sqrt 2
  let s2 := r * Real.sqrt 2
  (s1 ^ 2) / (s2 ^ 2) = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_area_ratio_l1656_165600


namespace NUMINAMATH_CALUDE_envelope_addressing_equation_l1656_165669

theorem envelope_addressing_equation (x : ℝ) : x > 0 → (
  let rate1 := 800 / 12  -- rate of first machine
  let rate2 := 800 / x   -- rate of second machine
  let combined_rate := 800 / 3  -- combined rate of both machines
  rate1 + rate2 = combined_rate ↔ 1/12 + 1/x = 1/3
) := by sorry

end NUMINAMATH_CALUDE_envelope_addressing_equation_l1656_165669


namespace NUMINAMATH_CALUDE_samantha_birth_year_l1656_165671

-- Define the year of the first AMC 8
def first_amc_year : ℕ := 1980

-- Define the function to calculate the year of the nth AMC 8
def amc_year (n : ℕ) : ℕ := first_amc_year + n - 1

-- Define Samantha's age when she took the 9th AMC 8
def samantha_age_at_ninth_amc : ℕ := 14

-- Theorem to prove Samantha's birth year
theorem samantha_birth_year :
  amc_year 9 - samantha_age_at_ninth_amc = 1974 := by
  sorry


end NUMINAMATH_CALUDE_samantha_birth_year_l1656_165671


namespace NUMINAMATH_CALUDE_smallest_angle_equation_l1656_165649

theorem smallest_angle_equation (y : ℝ) : 
  (∀ z ∈ {x : ℝ | x > 0 ∧ 8 * Real.sin x * (Real.cos x)^3 - 8 * (Real.sin x)^3 * Real.cos x = 1}, y ≤ z) ∧ 
  (8 * Real.sin y * (Real.cos y)^3 - 8 * (Real.sin y)^3 * Real.cos y = 1) →
  y = π / 24 :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_equation_l1656_165649


namespace NUMINAMATH_CALUDE_inequality_holds_l1656_165694

theorem inequality_holds (a b c d : ℝ) : (a - b) * (b - c) * (c - d) * (d - a) + (a - c)^2 * (b - d)^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l1656_165694


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l1656_165680

theorem boys_to_girls_ratio : 
  let num_boys : ℕ := 40
  let num_girls : ℕ := num_boys + 64
  (num_boys : ℚ) / (num_girls : ℚ) = 5 / 13 :=
by sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l1656_165680


namespace NUMINAMATH_CALUDE_gary_money_calculation_l1656_165692

/-- Calculates Gary's final amount of money after a series of transactions -/
def gary_final_amount (initial_amount snake_sale_price hamster_cost supplies_cost : ℝ) : ℝ :=
  initial_amount + snake_sale_price - hamster_cost - supplies_cost

/-- Theorem stating that Gary's final amount is 90.60 dollars -/
theorem gary_money_calculation :
  gary_final_amount 73.25 55.50 25.75 12.40 = 90.60 := by
  sorry

end NUMINAMATH_CALUDE_gary_money_calculation_l1656_165692


namespace NUMINAMATH_CALUDE_amy_soup_count_l1656_165610

/-- The number of chicken soup cans Amy bought -/
def chicken_soup : ℕ := 6

/-- The number of tomato soup cans Amy bought -/
def tomato_soup : ℕ := 3

/-- The total number of soup cans Amy bought -/
def total_soup : ℕ := chicken_soup + tomato_soup

theorem amy_soup_count : total_soup = 9 := by
  sorry

end NUMINAMATH_CALUDE_amy_soup_count_l1656_165610


namespace NUMINAMATH_CALUDE_equal_debt_after_10_days_l1656_165650

/-- The number of days after which Darren and Fergie will owe the same amount -/
def days_to_equal_debt : ℕ := 10

/-- Darren's initial borrowed amount in clams -/
def darren_initial : ℕ := 200

/-- Fergie's initial borrowed amount in clams -/
def fergie_initial : ℕ := 150

/-- Daily interest rate as a percentage -/
def daily_interest_rate : ℚ := 10 / 100

theorem equal_debt_after_10_days :
  (darren_initial : ℚ) * (1 + daily_interest_rate * days_to_equal_debt) =
  (fergie_initial : ℚ) * (1 + daily_interest_rate * days_to_equal_debt) :=
sorry

end NUMINAMATH_CALUDE_equal_debt_after_10_days_l1656_165650


namespace NUMINAMATH_CALUDE_complex_moduli_product_l1656_165625

theorem complex_moduli_product : Complex.abs (5 - 3*I) * Complex.abs (5 + 3*I) = 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_moduli_product_l1656_165625


namespace NUMINAMATH_CALUDE_cubic_polynomial_coefficient_l1656_165622

theorem cubic_polynomial_coefficient (a b c d : ℝ) : 
  let g := fun x => a * x^3 + b * x^2 + c * x + d
  (g (-2) = 0) → (g 0 = 0) → (g 2 = 0) → (g 1 = 3) → b = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_coefficient_l1656_165622


namespace NUMINAMATH_CALUDE_sector_central_angle_l1656_165606

theorem sector_central_angle (R : ℝ) (α : ℝ) 
  (h1 : 2 * R + α * R = 6)  -- circumference of sector
  (h2 : 1/2 * R^2 * α = 2)  -- area of sector
  : α = 1 ∨ α = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1656_165606


namespace NUMINAMATH_CALUDE_parallel_plane_line_l1656_165601

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallelPL : Plane → Line → Prop)
variable (parallelPP : Plane → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicularPL : Plane → Line → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- Define the theorem
theorem parallel_plane_line 
  (l m n : Line) 
  (α β : Plane) 
  (h_distinct_lines : l ≠ m ∧ l ≠ n ∧ m ≠ n) 
  (h_distinct_planes : α ≠ β) 
  (h_parallel_planes : parallelPP α β) 
  (h_line_in_plane : subset l α) : 
  parallelPL β l :=
sorry

end NUMINAMATH_CALUDE_parallel_plane_line_l1656_165601


namespace NUMINAMATH_CALUDE_janet_hourly_wage_correct_l1656_165677

/-- Janet's work scenario -/
structure WorkScenario where
  hourly_wage : ℝ
  regular_hours : ℝ
  overtime_hours : ℝ
  overtime_rate : ℝ
  weeks_worked : ℝ
  car_price : ℝ

/-- Calculate total earnings based on work scenario -/
def total_earnings (w : WorkScenario) : ℝ :=
  w.weeks_worked * (w.regular_hours * w.hourly_wage + w.overtime_hours * w.hourly_wage * w.overtime_rate)

/-- Janet's specific work scenario -/
def janet_scenario : WorkScenario := {
  hourly_wage := 20
  regular_hours := 40
  overtime_hours := 12
  overtime_rate := 1.5
  weeks_worked := 4
  car_price := 4640
}

/-- Theorem stating that Janet's hourly wage is correct -/
theorem janet_hourly_wage_correct : 
  total_earnings janet_scenario = janet_scenario.car_price := by sorry

end NUMINAMATH_CALUDE_janet_hourly_wage_correct_l1656_165677


namespace NUMINAMATH_CALUDE_quadruple_equation_solution_l1656_165627

def is_valid_quadruple (a b c d : ℕ) : Prop :=
  a + b = c * d ∧ c + d = a * b

def solution_set : Set (ℕ × ℕ × ℕ × ℕ) :=
  {(2, 2, 2, 2), (1, 2, 3, 5), (2, 1, 3, 5), (1, 2, 5, 3), (2, 1, 5, 3),
   (3, 5, 1, 2), (5, 3, 1, 2), (3, 5, 2, 1), (5, 3, 2, 1)}

theorem quadruple_equation_solution :
  {q : ℕ × ℕ × ℕ × ℕ | is_valid_quadruple q.1 q.2.1 q.2.2.1 q.2.2.2} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_quadruple_equation_solution_l1656_165627


namespace NUMINAMATH_CALUDE_polynomial_roots_l1656_165632

theorem polynomial_roots : 
  let p (x : ℚ) := 6*x^5 + 29*x^4 - 71*x^3 - 10*x^2 + 24*x + 8
  (p (-2) = 0) ∧ 
  (p (1/2) = 0) ∧ 
  (p 1 = 0) ∧ 
  (p (4/3) = 0) ∧ 
  (p (-2/3) = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l1656_165632


namespace NUMINAMATH_CALUDE_intersection_polar_radius_l1656_165690

-- Define the line l
def line_l (x : ℝ) : ℝ := x + 1

-- Define the curve C in polar form
def curve_C (ρ θ : ℝ) : Prop := ρ * (Real.sin θ)^2 - 4 * Real.cos θ = 0 ∧ ρ ≥ 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

-- Theorem statement
theorem intersection_polar_radius :
  ∃ (x y ρ θ : ℝ),
    y = line_l x ∧
    curve_C ρ θ ∧
    x = ρ * Real.cos θ ∧
    y = ρ * Real.sin θ ∧
    ρ = Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_intersection_polar_radius_l1656_165690


namespace NUMINAMATH_CALUDE_railway_stations_problem_l1656_165646

theorem railway_stations_problem (m n : ℕ) (h1 : n ≥ 1) :
  (m.choose 2 + n * m + n.choose 2) - m.choose 2 = 58 →
  ((m = 14 ∧ n = 2) ∨ (m = 29 ∧ n = 1)) := by
  sorry

end NUMINAMATH_CALUDE_railway_stations_problem_l1656_165646


namespace NUMINAMATH_CALUDE_factorization_proof_l1656_165617

theorem factorization_proof (a : ℝ) : a^2 + 4*a + 4 = (a + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1656_165617


namespace NUMINAMATH_CALUDE_no_function_satisfies_inequality_l1656_165653

theorem no_function_satisfies_inequality :
  ¬ ∃ (f : ℝ → ℝ), ∀ (x y z : ℝ), f (x * y) + f (x * z) - f x * f (y * z) > 1 := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_inequality_l1656_165653


namespace NUMINAMATH_CALUDE_race_length_l1656_165688

/-- Represents the state of the race -/
structure RaceState where
  alexLead : Int
  distanceLeft : Int

/-- Calculates the final race state after all lead changes -/
def finalRaceState : RaceState :=
  let s1 : RaceState := { alexLead := 0, distanceLeft := 0 }  -- Even start
  let s2 : RaceState := { alexLead := 300, distanceLeft := s1.distanceLeft }
  let s3 : RaceState := { alexLead := s2.alexLead - 170, distanceLeft := s2.distanceLeft }
  { alexLead := s3.alexLead + 440, distanceLeft := 3890 }

/-- The theorem stating the total length of the race track -/
theorem race_length : 
  finalRaceState.alexLead + finalRaceState.distanceLeft = 4460 := by
  sorry


end NUMINAMATH_CALUDE_race_length_l1656_165688


namespace NUMINAMATH_CALUDE_range_of_g_l1656_165666

theorem range_of_g (x : ℝ) : 
  -(1/4) ≤ Real.sin x ^ 6 - Real.sin x * Real.cos x + Real.cos x ^ 6 ∧ 
  Real.sin x ^ 6 - Real.sin x * Real.cos x + Real.cos x ^ 6 ≤ 3/4 := by
sorry

end NUMINAMATH_CALUDE_range_of_g_l1656_165666


namespace NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l1656_165699

/-- The volume of a sphere inscribed in a cube with edge length 12 inches is 288π cubic inches. -/
theorem volume_of_inscribed_sphere (π : ℝ) :
  let cube_edge : ℝ := 12
  let sphere_radius : ℝ := cube_edge / 2
  let sphere_volume : ℝ := (4 / 3) * π * sphere_radius ^ 3
  sphere_volume = 288 * π := by sorry

end NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l1656_165699


namespace NUMINAMATH_CALUDE_ticket_cost_difference_l1656_165665

theorem ticket_cost_difference : 
  let num_adults : ℕ := 9
  let num_children : ℕ := 7
  let adult_ticket_price : ℕ := 11
  let child_ticket_price : ℕ := 7
  (num_adults * adult_ticket_price) - (num_children * child_ticket_price) = 50 := by
sorry

end NUMINAMATH_CALUDE_ticket_cost_difference_l1656_165665


namespace NUMINAMATH_CALUDE_quadratic_root_k_range_l1656_165637

-- Define the quadratic function
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - k*x - 2

-- Theorem statement
theorem quadratic_root_k_range :
  ∀ k : ℝ, (∃ x : ℝ, 2 < x ∧ x < 5 ∧ f k x = 0) → (1 < k ∧ k < 23/5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_k_range_l1656_165637


namespace NUMINAMATH_CALUDE_money_distribution_l1656_165687

/-- Represents the share of money for each person -/
structure Share where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The problem statement -/
theorem money_distribution (s : Share) : 
  s.b = 0.65 * s.a → 
  s.c = 0.4 * s.a → 
  s.c = 56 → 
  s.a + s.b + s.c = 287 := by
  sorry


end NUMINAMATH_CALUDE_money_distribution_l1656_165687


namespace NUMINAMATH_CALUDE_three_lines_vertically_opposite_angles_l1656_165615

/-- The number of pairs of vertically opposite angles formed by three intersecting lines in a plane -/
def vertically_opposite_angles_count (n : ℕ) : ℕ :=
  if n = 3 then 6 else 0

/-- Theorem stating that three intersecting lines in a plane form 6 pairs of vertically opposite angles -/
theorem three_lines_vertically_opposite_angles :
  vertically_opposite_angles_count 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_three_lines_vertically_opposite_angles_l1656_165615


namespace NUMINAMATH_CALUDE_a_grade_implies_conditions_l1656_165648

-- Define the conditions for receiving an A
def receivesA (score : ℝ) (submittedAll : Bool) : Prop :=
  score ≥ 90 ∧ submittedAll

-- Define the theorem
theorem a_grade_implies_conditions 
  (score : ℝ) (submittedAll : Bool) :
  receivesA score submittedAll → 
  (score ≥ 90 ∧ submittedAll) :=
by
  sorry

-- The proof is omitted as per instructions

end NUMINAMATH_CALUDE_a_grade_implies_conditions_l1656_165648


namespace NUMINAMATH_CALUDE_square_area_perimeter_ratio_l1656_165636

theorem square_area_perimeter_ratio : 
  ∀ (s1 s2 : ℝ), s1 > 0 ∧ s2 > 0 →
  (s1^2 : ℝ) / (s2^2 : ℝ) = 49 / 64 →
  (4 * s1) / (4 * s2) = 7 / 8 :=
by
  sorry

end NUMINAMATH_CALUDE_square_area_perimeter_ratio_l1656_165636


namespace NUMINAMATH_CALUDE_system_solution_l1656_165696

theorem system_solution :
  ∃! (s : Set (ℝ × ℝ)), s = {(2, 4), (4, 2)} ∧
  ∀ (x y : ℝ), (x / y + y / x) * (x + y) = 15 ∧
                (x^2 / y^2 + y^2 / x^2) * (x^2 + y^2) = 85 →
                (x, y) ∈ s :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1656_165696


namespace NUMINAMATH_CALUDE_solution_is_two_l1656_165643

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (y : ℝ) : Prop :=
  lg (y - 1) - lg y = lg (2 * y - 2) - lg (y + 2)

-- Theorem statement
theorem solution_is_two :
  ∃ y : ℝ, y > 1 ∧ y + 2 > 0 ∧ equation y ∧ y = 2 :=
sorry

end NUMINAMATH_CALUDE_solution_is_two_l1656_165643


namespace NUMINAMATH_CALUDE_algae_cells_after_ten_days_l1656_165660

def algae_growth (initial_cells : ℕ) (split_factor : ℕ) (days : ℕ) : ℕ :=
  initial_cells * split_factor ^ days

theorem algae_cells_after_ten_days :
  algae_growth 1 3 10 = 59049 := by
  sorry

end NUMINAMATH_CALUDE_algae_cells_after_ten_days_l1656_165660


namespace NUMINAMATH_CALUDE_green_caterpillar_length_l1656_165681

/-- The length of the orange caterpillar in inches -/
def orange_length : ℝ := 1.17

/-- The difference in length between the green and orange caterpillar in inches -/
def length_difference : ℝ := 1.83

/-- The length of the green caterpillar in inches -/
def green_length : ℝ := orange_length + length_difference

/-- Theorem stating that the green caterpillar is 3.00 inches long -/
theorem green_caterpillar_length : green_length = 3.00 := by
  sorry

end NUMINAMATH_CALUDE_green_caterpillar_length_l1656_165681


namespace NUMINAMATH_CALUDE_carmen_additional_money_l1656_165603

/-- Calculates how much more money Carmen needs to have twice Jethro's amount -/
theorem carmen_additional_money (patricia_money jethro_money carmen_money : ℕ) : 
  patricia_money = 60 →
  patricia_money = 3 * jethro_money →
  carmen_money + patricia_money + jethro_money = 113 →
  (2 * jethro_money) - carmen_money = 7 :=
by
  sorry

#check carmen_additional_money

end NUMINAMATH_CALUDE_carmen_additional_money_l1656_165603


namespace NUMINAMATH_CALUDE_dream_car_gas_consumption_l1656_165640

/-- Calculates the total gas consumption for a car over two days -/
def total_gas_consumption (consumption_rate : ℝ) (miles_today : ℝ) (miles_tomorrow : ℝ) : ℝ :=
  consumption_rate * (miles_today + miles_tomorrow)

theorem dream_car_gas_consumption :
  let consumption_rate : ℝ := 4
  let miles_today : ℝ := 400
  let miles_tomorrow : ℝ := miles_today + 200
  total_gas_consumption consumption_rate miles_today miles_tomorrow = 4000 := by
sorry

end NUMINAMATH_CALUDE_dream_car_gas_consumption_l1656_165640


namespace NUMINAMATH_CALUDE_fraction_meaningful_l1656_165678

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x + 1) / (x - 2)) ↔ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l1656_165678


namespace NUMINAMATH_CALUDE_G_4_g_5_equals_29_l1656_165628

-- Define the functions g and G
def g (x : ℝ) : ℝ := 2 * x - 3
def G (x y : ℝ) : ℝ := x * y + 2 * x - y

-- State the theorem
theorem G_4_g_5_equals_29 : G 4 (g 5) = 29 := by
  sorry

end NUMINAMATH_CALUDE_G_4_g_5_equals_29_l1656_165628


namespace NUMINAMATH_CALUDE_gcd_special_numbers_l1656_165624

theorem gcd_special_numbers : 
  let m : ℕ := 555555555
  let n : ℕ := 1111111111
  Nat.gcd m n = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_special_numbers_l1656_165624


namespace NUMINAMATH_CALUDE_sugar_profit_problem_l1656_165686

theorem sugar_profit_problem (total_sugar : ℝ) (sugar_at_18_percent : ℝ) 
  (overall_profit_percent : ℝ) (profit_18_percent : ℝ) :
  total_sugar = 1000 →
  sugar_at_18_percent = 600 →
  overall_profit_percent = 14 →
  profit_18_percent = 18 →
  ∃ (remaining_profit_percent : ℝ),
    remaining_profit_percent = 8 ∧
    (sugar_at_18_percent * (1 + profit_18_percent / 100) + 
     (total_sugar - sugar_at_18_percent) * (1 + remaining_profit_percent / 100)) / total_sugar
    = 1 + overall_profit_percent / 100 :=
by sorry

end NUMINAMATH_CALUDE_sugar_profit_problem_l1656_165686


namespace NUMINAMATH_CALUDE_rectangle_area_l1656_165612

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * (L + B) = 246) : L * B = 3650 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1656_165612


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l1656_165609

theorem sum_of_reciprocals_of_roots (m n : ℝ) 
  (hm : m^2 + 3*m + 5 = 0) 
  (hn : n^2 + 3*n + 5 = 0) : 
  1/n + 1/m = -3/5 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l1656_165609


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1656_165676

theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence condition
  (a 20) ^ 2 - 10 * (a 20) + 16 = 0 →                   -- a_20 is a root
  (a 60) ^ 2 - 10 * (a 60) + 16 = 0 →                   -- a_60 is a root
  (a 30 * a 40 * a 50) / 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1656_165676


namespace NUMINAMATH_CALUDE_sector_max_area_l1656_165626

theorem sector_max_area (perimeter : ℝ) (h : perimeter = 40) :
  ∃ (area : ℝ), area ≤ 100 ∧ 
  ∀ (r l : ℝ), r > 0 → l > 0 → l + 2 * r = perimeter → 
  (1 / 2) * l * r ≤ area :=
by sorry

end NUMINAMATH_CALUDE_sector_max_area_l1656_165626


namespace NUMINAMATH_CALUDE_princes_wish_fulfilled_l1656_165685

/-- Represents a knight at the round table -/
structure Knight where
  city : Nat
  hasGoldGoblet : Bool

/-- Represents the state of the round table -/
def RoundTable := List Knight

/-- Checks if two knights from the same city have gold goblets -/
def sameCity_haveGold (table : RoundTable) : Bool := sorry

/-- Rotates the goblets one position to the right -/
def rotateGoblets (table : RoundTable) : RoundTable := sorry

theorem princes_wish_fulfilled 
  (initial_table : RoundTable)
  (h1 : initial_table.length = 13)
  (h2 : ∃ k : Nat, 1 < k ∧ k < 13 ∧ (initial_table.filter Knight.hasGoldGoblet).length = k)
  (h3 : ∃ k : Nat, 1 < k ∧ k < 13 ∧ (initial_table.map Knight.city).toFinset.card = k) :
  ∃ n : Nat, sameCity_haveGold (n.iterate rotateGoblets initial_table) := by
  sorry

end NUMINAMATH_CALUDE_princes_wish_fulfilled_l1656_165685


namespace NUMINAMATH_CALUDE_probability_not_red_card_l1656_165662

theorem probability_not_red_card (odds_red : ℚ) (h : odds_red = 5/7) :
  1 - odds_red / (1 + odds_red) = 7/12 := by sorry

end NUMINAMATH_CALUDE_probability_not_red_card_l1656_165662


namespace NUMINAMATH_CALUDE_fort_blocks_count_l1656_165667

/-- Represents the dimensions of a rectangular fort -/
structure FortDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of blocks needed for a fort with given dimensions and wall thickness -/
def blocksNeeded (d : FortDimensions) (wallThickness : ℕ) : ℕ :=
  d.length * d.width * d.height - 
  (d.length - 2 * wallThickness) * (d.width - 2 * wallThickness) * (d.height - wallThickness)

/-- Theorem stating that a fort with given dimensions requires 280 blocks -/
theorem fort_blocks_count : 
  blocksNeeded ⟨12, 10, 5⟩ 1 = 280 := by sorry

end NUMINAMATH_CALUDE_fort_blocks_count_l1656_165667


namespace NUMINAMATH_CALUDE_impossible_coin_probabilities_l1656_165652

theorem impossible_coin_probabilities : ¬∃ (p₁ p₂ : ℝ), 
  0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧ 
  (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧
  p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) := by
  sorry

end NUMINAMATH_CALUDE_impossible_coin_probabilities_l1656_165652


namespace NUMINAMATH_CALUDE_wood_length_ratio_l1656_165639

def first_set_length : ℝ := 4
def second_set_length : ℝ := 20

theorem wood_length_ratio : second_set_length / first_set_length = 5 := by
  sorry

end NUMINAMATH_CALUDE_wood_length_ratio_l1656_165639


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l1656_165697

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∀ n : ℕ, n > 0 → n^2 < 900 → ¬(2 ∣ n^2 ∧ 3 ∣ n^2 ∧ 5 ∣ n^2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l1656_165697


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l1656_165614

theorem complex_exponential_sum (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = (1/3 : ℂ) + (1/2 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = (1/3 : ℂ) - (1/2 : ℂ) * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l1656_165614


namespace NUMINAMATH_CALUDE_cube_configurations_l1656_165663

/-- Represents a rotation in 3D space -/
structure Rotation :=
  (fixedConfigurations : ℕ)

/-- The group of rotations for a cube -/
def rotationGroup : Finset Rotation := sorry

/-- The number of white unit cubes -/
def numWhiteCubes : ℕ := 5

/-- The number of blue unit cubes -/
def numBlueCubes : ℕ := 3

/-- The total number of unit cubes -/
def totalCubes : ℕ := numWhiteCubes + numBlueCubes

/-- Calculates the number of fixed configurations for a given rotation -/
def fixedConfigurations (r : Rotation) : ℕ := r.fixedConfigurations

/-- Applies Burnside's Lemma to calculate the number of distinct configurations -/
def distinctConfigurations : ℕ :=
  (rotationGroup.sum fixedConfigurations) / rotationGroup.card

theorem cube_configurations :
  distinctConfigurations = 3 := by sorry

end NUMINAMATH_CALUDE_cube_configurations_l1656_165663


namespace NUMINAMATH_CALUDE_janessa_keeps_twenty_cards_l1656_165629

/-- The number of cards Janessa keeps for herself --/
def cards_kept_by_janessa (initial_cards : ℕ) (cards_from_father : ℕ) (cards_ordered : ℕ) 
  (bad_cards : ℕ) (cards_given_to_dexter : ℕ) : ℕ :=
  initial_cards + cards_from_father + cards_ordered - bad_cards - cards_given_to_dexter

/-- Theorem stating that Janessa keeps 20 cards for herself --/
theorem janessa_keeps_twenty_cards : 
  cards_kept_by_janessa 4 13 36 4 29 = 20 := by
  sorry

end NUMINAMATH_CALUDE_janessa_keeps_twenty_cards_l1656_165629


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1656_165651

theorem quadratic_equation_solution (x₁ : ℚ) (h₁ : x₁ = 3/4) 
  (h₂ : 72 * x₁^2 + 39 * x₁ - 18 = 0) : 
  ∃ x₂ : ℚ, x₂ = -31/6 ∧ 72 * x₂^2 + 39 * x₂ - 18 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1656_165651


namespace NUMINAMATH_CALUDE_range_of_a_sum_of_a_and_b_l1656_165602

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|
def g (a x : ℝ) : ℝ := a - |x - 2|

-- Theorem 1: If f(x) < g(x) has solutions, then a > 4
theorem range_of_a (a : ℝ) : 
  (∃ x, f x < g a x) → a > 4 := by sorry

-- Theorem 2: If the solution set of f(x) < g(x) is (b, 7/2), then a + b = 6
theorem sum_of_a_and_b (a b : ℝ) : 
  (∀ x, f x < g a x ↔ b < x ∧ x < 7/2) → a + b = 6 := by sorry

end NUMINAMATH_CALUDE_range_of_a_sum_of_a_and_b_l1656_165602


namespace NUMINAMATH_CALUDE_acute_triangle_side_range_l1656_165698

/-- Given an acute triangle ABC with side lengths a = 2 and b = 3, 
    prove that the side length c satisfies √5 < c < √13 -/
theorem acute_triangle_side_range (a b c : ℝ) : 
  a = 2 → b = 3 → 
  (a^2 + b^2 > c^2) → (a^2 + c^2 > b^2) → (b^2 + c^2 > a^2) →
  Real.sqrt 5 < c ∧ c < Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_side_range_l1656_165698


namespace NUMINAMATH_CALUDE_inequality_permutation_l1656_165619

theorem inequality_permutation (a b c d : ℝ) (h : a * b * c * d > 0) :
  ∃ (x y z w : ℝ), (({x, y, z, w} : Finset ℝ) = {a, b, c, d}) ∧
  (2 * (x * z + y * w)^2 > (x^2 + y^2) * (z^2 + w^2)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_permutation_l1656_165619


namespace NUMINAMATH_CALUDE_field_trip_total_cost_l1656_165668

def field_trip_cost (num_students : ℕ) (num_teachers : ℕ) 
  (student_ticket_price : ℚ) (teacher_ticket_price : ℚ) 
  (discount_rate : ℚ) (tour_price : ℚ) (bus_cost : ℚ) 
  (meal_cost : ℚ) : ℚ :=
  let total_people := num_students + num_teachers
  let ticket_cost := num_students * student_ticket_price + num_teachers * teacher_ticket_price
  let discounted_ticket_cost := ticket_cost * (1 - discount_rate)
  let tour_cost := total_people * tour_price
  let meal_cost_total := total_people * meal_cost
  discounted_ticket_cost + tour_cost + bus_cost + meal_cost_total

theorem field_trip_total_cost :
  field_trip_cost 25 6 1.5 4 0.2 3.5 100 7.5 = 490.2 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_total_cost_l1656_165668


namespace NUMINAMATH_CALUDE_clock_hands_overlap_l1656_165607

/-- The angle traveled by the hour hand in one minute -/
def hour_hand_speed : ℝ := 0.5

/-- The angle traveled by the minute hand in one minute -/
def minute_hand_speed : ℝ := 6

/-- The initial angle of the hour hand at 4:10 -/
def initial_hour_angle : ℝ := 60 + 0.5 * 10

/-- The time in minutes after 4:10 when the hands overlap -/
def overlap_time : ℝ := 11

theorem clock_hands_overlap :
  ∃ (t : ℝ), t > 0 ∧ t ≤ overlap_time ∧
  initial_hour_angle + hour_hand_speed * t = minute_hand_speed * t :=
sorry

end NUMINAMATH_CALUDE_clock_hands_overlap_l1656_165607


namespace NUMINAMATH_CALUDE_painted_fraction_of_specific_cone_l1656_165621

/-- Represents a cone with given dimensions -/
structure Cone where
  radius : ℝ
  slant_height : ℝ

/-- Calculates the fraction of a cone's surface area covered in paint -/
def painted_fraction (c : Cone) (paint_depth : ℝ) : ℚ :=
  sorry

/-- Theorem stating the correct fraction of painted surface area for the given cone -/
theorem painted_fraction_of_specific_cone :
  let c : Cone := { radius := 3, slant_height := 5 }
  painted_fraction c 2 = 27 / 32 := by
  sorry

end NUMINAMATH_CALUDE_painted_fraction_of_specific_cone_l1656_165621


namespace NUMINAMATH_CALUDE_square_sum_product_equals_k_squared_l1656_165611

theorem square_sum_product_equals_k_squared (k : ℕ) : 
  2012^2 + 2010 * 2011 * 2013 * 2014 = k^2 ∧ k > 0 → k = 4048142 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_equals_k_squared_l1656_165611


namespace NUMINAMATH_CALUDE_special_dog_food_weight_l1656_165675

/-- The weight of each bag of special dog food for a puppy -/
theorem special_dog_food_weight :
  let first_period_days : ℕ := 60
  let total_days : ℕ := 365
  let first_period_consumption : ℕ := 2  -- ounces per day
  let second_period_consumption : ℕ := 4  -- ounces per day
  let ounces_per_pound : ℕ := 16
  let number_of_bags : ℕ := 17
  
  let total_consumption : ℕ := 
    first_period_days * first_period_consumption + 
    (total_days - first_period_days) * second_period_consumption
  
  let total_pounds : ℚ := total_consumption / ounces_per_pound
  let bag_weight : ℚ := total_pounds / number_of_bags
  
  ∃ (weight : ℚ), abs (weight - bag_weight) < 0.005 ∧ weight = 4.93 :=
by sorry

end NUMINAMATH_CALUDE_special_dog_food_weight_l1656_165675


namespace NUMINAMATH_CALUDE_first_group_has_four_weavers_l1656_165647

/-- The number of mat-weavers in the first group -/
def first_group_weavers : ℕ := 4

/-- The number of mats woven by the first group -/
def first_group_mats : ℕ := 4

/-- The number of days taken by the first group -/
def first_group_days : ℕ := 4

/-- The number of mat-weavers in the second group -/
def second_group_weavers : ℕ := 8

/-- The number of mats woven by the second group -/
def second_group_mats : ℕ := 16

/-- The number of days taken by the second group -/
def second_group_days : ℕ := 8

/-- The rate of weaving is the same for both groups -/
axiom same_rate : (first_group_mats : ℚ) / first_group_weavers / first_group_days = 
                  (second_group_mats : ℚ) / second_group_weavers / second_group_days

theorem first_group_has_four_weavers : first_group_weavers = 4 := by
  sorry

end NUMINAMATH_CALUDE_first_group_has_four_weavers_l1656_165647


namespace NUMINAMATH_CALUDE_expected_unpaired_socks_l1656_165682

def n : ℕ := 2024

theorem expected_unpaired_socks (n : ℕ) :
  let total_socks := 2 * n
  let binom := Nat.choose total_socks n
  let expected_total := (4 : ℝ)^n / binom
  expected_total - 2 = (4 : ℝ)^n / Nat.choose (2 * n) n - 2 := by sorry

end NUMINAMATH_CALUDE_expected_unpaired_socks_l1656_165682


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1656_165630

theorem polynomial_simplification (x : ℝ) :
  (3*x - 2) * (5*x^12 + 3*x^11 + 7*x^9 + 3*x^8) =
  15*x^13 - x^12 - 6*x^11 + 21*x^10 - 5*x^9 - 6*x^8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1656_165630


namespace NUMINAMATH_CALUDE_combined_salaries_l1656_165683

/-- Given 5 individuals with an average salary of 8200 and one individual with a salary of 7000,
    prove that the sum of the other 4 individuals' salaries is 34000 -/
theorem combined_salaries (average_salary : ℕ) (num_individuals : ℕ) (d_salary : ℕ) :
  average_salary = 8200 →
  num_individuals = 5 →
  d_salary = 7000 →
  (average_salary * num_individuals) - d_salary = 34000 := by
  sorry

end NUMINAMATH_CALUDE_combined_salaries_l1656_165683


namespace NUMINAMATH_CALUDE_smallest_positive_product_l1656_165604

def S : Set Int := {-4, -3, -1, 5, 6}

def is_valid_product (x y z : Int) : Prop :=
  x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z

def product (x y z : Int) : Int := x * y * z

theorem smallest_positive_product :
  ∃ (a b c : Int), is_valid_product a b c ∧ 
    product a b c > 0 ∧
    product a b c = 15 ∧
    ∀ (x y z : Int), is_valid_product x y z → product x y z > 0 → product x y z ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_product_l1656_165604


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l1656_165638

theorem quadratic_integer_roots (b : ℤ) : 
  (∃ x y : ℤ, x ≠ y ∧ (x^2 - b*x + 3*b = 0) ∧ (y^2 - b*y + 3*b = 0)) → 
  (b = 9 ∨ b = -6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l1656_165638


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1656_165608

theorem polynomial_division_theorem (x : ℝ) : 
  (4*x^2 - 2*x + 3) * (2*x^2 + 5*x + 3) + (43*x + 36) = 8*x^4 + 16*x^3 - 7*x^2 + 4*x + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1656_165608
