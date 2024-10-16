import Mathlib

namespace NUMINAMATH_CALUDE_ice_floe_mass_l255_25595

/-- Calculates the mass of an ice floe based on a polar bear's movement --/
theorem ice_floe_mass (bear_mass : ℝ) (ice_path_diameter : ℝ) (observed_diameter : ℝ) :
  bear_mass = 600 →
  ice_path_diameter = 9.5 →
  observed_diameter = 10 →
  ∃ (ice_floe_mass : ℝ),
    ice_floe_mass = (bear_mass * ice_path_diameter) / (observed_diameter - ice_path_diameter) ∧
    ice_floe_mass = 11400 := by
  sorry

end NUMINAMATH_CALUDE_ice_floe_mass_l255_25595


namespace NUMINAMATH_CALUDE_open_box_volume_l255_25532

/-- The volume of an open box formed by cutting squares from corners of a rectangular sheet -/
theorem open_box_volume 
  (sheet_length sheet_width cut_size : ℝ) 
  (h_length : sheet_length = 48)
  (h_width : sheet_width = 38)
  (h_cut : cut_size = 8) : 
  (sheet_length - 2 * cut_size) * (sheet_width - 2 * cut_size) * cut_size = 5632 :=
by sorry

end NUMINAMATH_CALUDE_open_box_volume_l255_25532


namespace NUMINAMATH_CALUDE_inequality_implication_l255_25571

theorem inequality_implication (a b : ℝ) (h : a < b) : 1 - a > 1 - b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l255_25571


namespace NUMINAMATH_CALUDE_sum_of_A_and_B_l255_25539

theorem sum_of_A_and_B : ∀ A B : ℚ, 
  (1 / 4 : ℚ) * (1 / 8 : ℚ) = 1 / (4 * A) ∧ 
  (1 / 4 : ℚ) * (1 / 8 : ℚ) = 1 / B → 
  A + B = 40 := by
sorry

end NUMINAMATH_CALUDE_sum_of_A_and_B_l255_25539


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l255_25561

theorem polynomial_remainder_theorem (x : ℝ) : 
  (x^4 + 2*x^2 + 2) % (x - 2) = 26 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l255_25561


namespace NUMINAMATH_CALUDE_regression_properties_l255_25510

/-- A dataset of two variables -/
structure Dataset where
  x : List ℝ
  y : List ℝ

/-- Properties of a linear regression model -/
structure RegressionModel (d : Dataset) where
  x_mean : ℝ
  y_mean : ℝ
  r : ℝ
  b_hat : ℝ
  a_hat : ℝ

/-- The regression line passes through the mean point -/
def passes_through_mean (m : RegressionModel d) : Prop :=
  m.y_mean = m.b_hat * m.x_mean + m.a_hat

/-- Strong correlation between variables -/
def strong_correlation (m : RegressionModel d) : Prop :=
  abs m.r > 0.75

/-- Negative slope of the regression line -/
def negative_slope (m : RegressionModel d) : Prop :=
  m.b_hat < 0

/-- Main theorem -/
theorem regression_properties (d : Dataset) (m : RegressionModel d)
  (h1 : m.r = -0.8) :
  passes_through_mean m ∧ strong_correlation m ∧ negative_slope m := by
  sorry

end NUMINAMATH_CALUDE_regression_properties_l255_25510


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l255_25504

/-- Calculates the simple interest rate given the principal, time, and interest amount -/
def simple_interest_rate (principal time interest : ℚ) : ℚ :=
  (interest / (principal * time)) * 100

/-- Theorem stating that for the given conditions, the simple interest rate is 2.5% -/
theorem interest_rate_calculation :
  let principal : ℚ := 700
  let time : ℚ := 4
  let interest : ℚ := 70
  simple_interest_rate principal time interest = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l255_25504


namespace NUMINAMATH_CALUDE_sin_plus_cos_from_double_angle_l255_25523

theorem sin_plus_cos_from_double_angle (A : ℝ) (h1 : 0 < A) (h2 : A < π / 2) (h3 : Real.sin (2 * A) = 2 / 3) :
  Real.sin A + Real.cos A = Real.sqrt 15 / 3 := by
sorry

end NUMINAMATH_CALUDE_sin_plus_cos_from_double_angle_l255_25523


namespace NUMINAMATH_CALUDE_hours_per_day_is_five_l255_25500

/-- The number of hours worked per day by the first group of women -/
def hours_per_day : ℝ := 5

/-- The number of women in the first group -/
def women_group1 : ℕ := 6

/-- The number of days worked by the first group -/
def days_group1 : ℕ := 8

/-- The units of work completed by the first group -/
def work_units_group1 : ℕ := 75

/-- The number of women in the second group -/
def women_group2 : ℕ := 4

/-- The number of days worked by the second group -/
def days_group2 : ℕ := 3

/-- The units of work completed by the second group -/
def work_units_group2 : ℕ := 30

/-- The number of hours worked per day by the second group -/
def hours_per_day_group2 : ℕ := 8

/-- The proposition that the amount of work done is proportional to the number of woman-hours worked -/
axiom work_proportional_to_hours : 
  (women_group1 * days_group1 * hours_per_day) / work_units_group1 = 
  (women_group2 * days_group2 * hours_per_day_group2) / work_units_group2

theorem hours_per_day_is_five : hours_per_day = 5 := by
  sorry

end NUMINAMATH_CALUDE_hours_per_day_is_five_l255_25500


namespace NUMINAMATH_CALUDE_sarah_scored_135_l255_25514

def sarahs_score (greg_score sarah_score : ℕ) : Prop :=
  greg_score + 50 = sarah_score ∧ (greg_score + sarah_score) / 2 = 110

theorem sarah_scored_135 :
  ∃ (greg_score : ℕ), sarahs_score greg_score 135 :=
by sorry

end NUMINAMATH_CALUDE_sarah_scored_135_l255_25514


namespace NUMINAMATH_CALUDE_tonya_lemonade_revenue_l255_25552

/-- Calculates the total revenue from Tonya's lemonade stand --/
def lemonade_revenue (small_price medium_price large_price : ℕ)
  (small_revenue medium_revenue : ℕ) (large_cups : ℕ) : ℕ :=
  small_revenue + medium_revenue + (large_cups * large_price)

theorem tonya_lemonade_revenue :
  lemonade_revenue 1 2 3 11 24 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_tonya_lemonade_revenue_l255_25552


namespace NUMINAMATH_CALUDE_divisibility_by_8640_l255_25521

theorem divisibility_by_8640 (x : ℤ) : ∃ k : ℤ, x^9 - 6*x^7 + 9*x^5 - 4*x^3 = 8640 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_8640_l255_25521


namespace NUMINAMATH_CALUDE_modular_inverse_5_mod_23_l255_25512

theorem modular_inverse_5_mod_23 : ∃ (a : ℤ), 5 * a ≡ 1 [ZMOD 23] ∧ a = 14 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_5_mod_23_l255_25512


namespace NUMINAMATH_CALUDE_urn_probability_theorem_l255_25587

/-- Represents the number of balls of each color in the urn -/
structure UrnState :=
  (red : ℕ)
  (blue : ℕ)

/-- Represents one draw operation -/
inductive DrawResult
| Red
| Blue

/-- Represents a sequence of 6 draw operations -/
def DrawSequence := Vector DrawResult 6

/-- Initial state of the urn -/
def initial_state : UrnState := ⟨1, 2⟩

/-- Final number of balls in the urn after 6 operations -/
def final_ball_count : ℕ := 8

/-- Calculates the probability of a specific draw sequence -/
def sequence_probability (seq : DrawSequence) : ℚ :=
  sorry

/-- Calculates the number of sequences that result in 4 red and 4 blue balls -/
def favorable_sequence_count : ℕ :=
  sorry

/-- Theorem: The probability of having 4 red and 4 blue balls after 6 operations is 5/14 -/
theorem urn_probability_theorem :
  (favorable_sequence_count : ℚ) * sequence_probability (Vector.replicate 6 DrawResult.Red) = 5/14 :=
sorry

end NUMINAMATH_CALUDE_urn_probability_theorem_l255_25587


namespace NUMINAMATH_CALUDE_anna_age_when_married_l255_25544

/-- Represents the ages and marriage duration of Josh and Anna -/
structure Couple where
  josh_age_at_marriage : ℕ
  years_married : ℕ
  combined_age_factor : ℕ

/-- Calculates Anna's age when they got married -/
def anna_age_at_marriage (c : Couple) : ℕ :=
  c.combined_age_factor * c.josh_age_at_marriage - (c.josh_age_at_marriage + c.years_married)

/-- Theorem stating Anna's age when they got married -/
theorem anna_age_when_married (c : Couple) 
    (h1 : c.josh_age_at_marriage = 22)
    (h2 : c.years_married = 30)
    (h3 : c.combined_age_factor = 5) :
  anna_age_at_marriage c = 28 := by
  sorry

#eval anna_age_at_marriage ⟨22, 30, 5⟩

end NUMINAMATH_CALUDE_anna_age_when_married_l255_25544


namespace NUMINAMATH_CALUDE_h_domain_l255_25542

noncomputable def h (x : ℝ) : ℝ := (x^2 - 9) / (|x - 4| + x^2 - 1)

def domain_of_h : Set ℝ := {x | x < (1 + Real.sqrt 13) / 2 ∨ x > (1 + Real.sqrt 13) / 2}

theorem h_domain : 
  {x : ℝ | ∃ y, h x = y} = domain_of_h :=
by sorry

end NUMINAMATH_CALUDE_h_domain_l255_25542


namespace NUMINAMATH_CALUDE_decimal_multiplication_l255_25501

theorem decimal_multiplication (h : 28 * 15 = 420) :
  (2.8 * 1.5 = 4.2) ∧ (0.28 * 1.5 = 42) ∧ (0.028 * 0.15 = 0.0042) := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l255_25501


namespace NUMINAMATH_CALUDE_sequence_a_formula_l255_25533

def sequence_a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

axiom S_2 : S 2 = 4

axiom a_recursive (n : ℕ) : n ≥ 1 → sequence_a (n + 1) = 2 * S n + 1

theorem sequence_a_formula (n : ℕ) : n ≥ 1 → sequence_a n = 3^(n - 1) := by sorry

end NUMINAMATH_CALUDE_sequence_a_formula_l255_25533


namespace NUMINAMATH_CALUDE_amc10_min_correct_problems_l255_25529

/-- The AMC 10 scoring system and Sarah's strategy -/
structure AMC10 where
  total_problems : Nat
  attempted_problems : Nat
  correct_points : Nat
  unanswered_points : Nat
  target_score : Nat

/-- The minimum number of correctly solved problems to reach the target score -/
def min_correct_problems (amc : AMC10) : Nat :=
  let unanswered := amc.total_problems - amc.attempted_problems
  let unanswered_score := unanswered * amc.unanswered_points
  let required_score := amc.target_score - unanswered_score
  (required_score + amc.correct_points - 1) / amc.correct_points

/-- Theorem stating that for the given AMC 10 configuration, 
    the minimum number of correctly solved problems is 20 -/
theorem amc10_min_correct_problems :
  let amc : AMC10 := {
    total_problems := 30,
    attempted_problems := 25,
    correct_points := 7,
    unanswered_points := 2,
    target_score := 150
  }
  min_correct_problems amc = 20 := by
  sorry

end NUMINAMATH_CALUDE_amc10_min_correct_problems_l255_25529


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_21_ending_in_3_l255_25528

theorem three_digit_divisible_by_21_ending_in_3 :
  ∃! (s : Finset Nat), 
    s.card = 3 ∧
    (∀ n ∈ s, 100 ≤ n ∧ n < 1000 ∧ n % 10 = 3 ∧ n % 21 = 0) ∧
    (∀ n, 100 ≤ n ∧ n < 1000 ∧ n % 10 = 3 ∧ n % 21 = 0 → n ∈ s) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_21_ending_in_3_l255_25528


namespace NUMINAMATH_CALUDE_d_share_is_thirteen_sixtieths_l255_25569

/-- Represents the capital shares of partners in a business. -/
structure CapitalShares where
  total : ℚ
  a : ℚ
  b : ℚ
  c : ℚ
  d : ℚ
  a_share : a = (1 : ℚ) / 3 * total
  b_share : b = (1 : ℚ) / 4 * total
  c_share : c = (1 : ℚ) / 5 * total
  total_sum : a + b + c + d = total

/-- Represents the profit distribution in the business. -/
structure ProfitDistribution where
  total : ℚ
  a_profit : ℚ
  total_amount : total = 2490
  a_amount : a_profit = 830

/-- Theorem stating that given the capital shares and profit distribution,
    partner D's share of the capital is 13/60. -/
theorem d_share_is_thirteen_sixtieths
  (shares : CapitalShares) (profit : ProfitDistribution) :
  shares.d = (13 : ℚ) / 60 * shares.total :=
sorry

end NUMINAMATH_CALUDE_d_share_is_thirteen_sixtieths_l255_25569


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l255_25545

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, (3 + 5 * x - 2 * x^2 > 0) ↔ (-1/2 < x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l255_25545


namespace NUMINAMATH_CALUDE_max_notebooks_purchasable_l255_25543

def total_money : ℕ := 1050  -- £10.50 in pence
def notebook_cost : ℕ := 75  -- £0.75 in pence

theorem max_notebooks_purchasable :
  ∀ n : ℕ, n * notebook_cost ≤ total_money →
  n ≤ 14 :=
by sorry

end NUMINAMATH_CALUDE_max_notebooks_purchasable_l255_25543


namespace NUMINAMATH_CALUDE_eunji_pocket_money_l255_25541

theorem eunji_pocket_money (initial_money : ℕ) : 
  (initial_money / 4 : ℕ) + 
  ((3 * initial_money / 4) / 3 : ℕ) + 
  1600 = initial_money → 
  initial_money = 3200 := by
sorry

end NUMINAMATH_CALUDE_eunji_pocket_money_l255_25541


namespace NUMINAMATH_CALUDE_infinite_solutions_condition_l255_25554

theorem infinite_solutions_condition (b : ℝ) :
  (∀ x, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := by
sorry

end NUMINAMATH_CALUDE_infinite_solutions_condition_l255_25554


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l255_25549

theorem trigonometric_equation_solution (x : ℝ) : 
  (Real.sin (3 * x) + Real.sin x - Real.sin (2 * x) = 2 * Real.cos x * (Real.cos x - 1)) ↔ 
  (∃ k : ℤ, x = π / 2 * (2 * k + 1)) ∨ 
  (∃ n : ℤ, x = 2 * π * n) ∨ 
  (∃ l : ℤ, x = π / 4 * (4 * l - 1)) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l255_25549


namespace NUMINAMATH_CALUDE_max_x_value_l255_25590

theorem max_x_value (x y z : ℝ) 
  (eq1 : 3 * x + 2 * y + z = 10) 
  (eq2 : x * y + x * z + y * z = 6) : 
  x ≤ 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_max_x_value_l255_25590


namespace NUMINAMATH_CALUDE_roots_of_equation_correct_description_l255_25586

theorem roots_of_equation (x : ℝ) : 
  (x^2 + 4) * (x^2 - 4) = 0 ↔ x = 2 ∨ x = -2 :=
by sorry

theorem correct_description : 
  ∀ x : ℝ, (x^2 + 4) * (x^2 - 4) = 0 → 
  ∃ y : ℝ, y = 2 ∨ y = -2 ∧ (y^2 + 4) * (y^2 - 4) = 0 :=
by sorry

end NUMINAMATH_CALUDE_roots_of_equation_correct_description_l255_25586


namespace NUMINAMATH_CALUDE_wooden_block_volume_l255_25574

/-- A rectangular wooden block -/
structure WoodenBlock where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The volume of a wooden block -/
def volume (block : WoodenBlock) : ℝ :=
  block.length * block.width * block.height

/-- The surface area of a wooden block -/
def surfaceArea (block : WoodenBlock) : ℝ :=
  2 * (block.length * block.width + block.length * block.height + block.width * block.height)

/-- The increase in surface area after sawing -/
def surfaceAreaIncrease (block : WoodenBlock) (sections : ℕ) : ℝ :=
  2 * (sections - 1) * block.width * block.height

theorem wooden_block_volume
  (block : WoodenBlock)
  (h_length : block.length = 10)
  (h_sections : ℕ)
  (h_sections_eq : h_sections = 6)
  (h_area_increase : surfaceAreaIncrease block h_sections = 1) :
  volume block = 10 := by
  sorry

end NUMINAMATH_CALUDE_wooden_block_volume_l255_25574


namespace NUMINAMATH_CALUDE_field_trip_girls_l255_25588

theorem field_trip_girls (num_vans : ℕ) (students_per_van : ℕ) (num_boys : ℕ) : 
  num_vans = 5 → 
  students_per_van = 28 → 
  num_boys = 60 → 
  num_vans * students_per_van - num_boys = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_field_trip_girls_l255_25588


namespace NUMINAMATH_CALUDE_largest_x_floor_div_l255_25562

theorem largest_x_floor_div (x : ℝ) : 
  (⌊x⌋ : ℝ) / x = 9 / 10 → x ≤ 80 / 9 := by
  sorry

end NUMINAMATH_CALUDE_largest_x_floor_div_l255_25562


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l255_25572

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x, (1 - 2*x)^8 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ + 6*a₆ + 7*a₇ + 8*a₈ = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l255_25572


namespace NUMINAMATH_CALUDE_root_sum_theorem_l255_25507

theorem root_sum_theorem (m n p : ℝ) : 
  (∀ x, x^2 + 4*x + p = 0 ↔ x = m ∨ x = n) → 
  m * n = 4 → 
  m + n = -4 := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l255_25507


namespace NUMINAMATH_CALUDE_y_derivative_l255_25556

noncomputable def y (x : ℝ) : ℝ := Real.sqrt (1 - 3*x - 2*x^2) + (3 / (2 * Real.sqrt 2)) * Real.arcsin ((4*x + 3) / Real.sqrt 17)

theorem y_derivative (x : ℝ) : 
  deriv y x = -(2*x) / Real.sqrt (1 - 3*x - 2*x^2) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l255_25556


namespace NUMINAMATH_CALUDE_parabola_tangent_problem_l255_25550

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y

-- Define the point M
def point_M (p : ℝ) : ℝ × ℝ := (2, -2*p)

-- Define a line touching the parabola at two points
def touching_line (p : ℝ) (A B : ℝ × ℝ) : Prop :=
  parabola p A.1 A.2 ∧ parabola p B.1 B.2 ∧
  ∃ (m c : ℝ), A.2 = m * A.1 + c ∧ B.2 = m * B.1 + c ∧
  point_M p = (2, m * 2 + c)

-- Define the midpoint condition
def midpoint_condition (A B : ℝ × ℝ) : Prop :=
  (A.2 + B.2) / 2 = 6

-- Theorem statement
theorem parabola_tangent_problem (p : ℝ) (A B : ℝ × ℝ) :
  p > 0 →
  touching_line p A B →
  midpoint_condition A B →
  p = 1 ∨ p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_problem_l255_25550


namespace NUMINAMATH_CALUDE_gdp_equality_l255_25573

/-- Represents the GDP value in trillion yuan -/
def gdp_trillion : ℝ := 33.5

/-- Represents the GDP value in scientific notation -/
def gdp_scientific : ℝ := 3.35 * (10 ^ 13)

/-- Theorem stating that the GDP value in trillion yuan is equal to its scientific notation -/
theorem gdp_equality : gdp_trillion * (10 ^ 12) = gdp_scientific := by sorry

end NUMINAMATH_CALUDE_gdp_equality_l255_25573


namespace NUMINAMATH_CALUDE_max_ski_trips_l255_25526

/-- Proves the maximum number of ski trips in a given time --/
theorem max_ski_trips (lift_time ski_time total_time : ℕ) : 
  lift_time = 15 →
  ski_time = 5 →
  total_time = 120 →
  (total_time / (lift_time + ski_time) : ℕ) = 6 := by
  sorry

#check max_ski_trips

end NUMINAMATH_CALUDE_max_ski_trips_l255_25526


namespace NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l255_25525

theorem arccos_one_over_sqrt_two (π : ℝ) : Real.arccos (1 / Real.sqrt 2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_over_sqrt_two_l255_25525


namespace NUMINAMATH_CALUDE_range_of_a_l255_25551

-- Define the conditions
def p (x : ℝ) : Prop := x^2 - 8*x - 33 > 0
def q (x a : ℝ) : Prop := |x - 1| > a

-- Define the theorem
theorem range_of_a (h : ∀ x a : ℝ, a > 0 → (p x → q x a) ∧ ¬(q x a → p x)) :
  ∃ a : ℝ, a > 0 ∧ a ≤ 4 ∧ ∀ b : ℝ, (b > 0 ∧ b ≤ 4 → ∃ x : ℝ, p x → q x b) ∧
    (b > 4 → ∃ x : ℝ, p x ∧ ¬(q x b)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l255_25551


namespace NUMINAMATH_CALUDE_rohan_age_multiple_l255_25567

def rohan_current_age : ℕ := 25

def rohan_past_age : ℕ := rohan_current_age - 15

def rohan_future_age : ℕ := rohan_current_age + 15

theorem rohan_age_multiple : 
  ∃ (x : ℚ), rohan_future_age = x * rohan_past_age ∧ x = 4 := by
sorry

end NUMINAMATH_CALUDE_rohan_age_multiple_l255_25567


namespace NUMINAMATH_CALUDE_f_min_max_l255_25518

-- Define the function
def f (x : ℝ) : ℝ := 1 + 3*x - x^3

-- State the theorem
theorem f_min_max : 
  (∃ x : ℝ, f x = -1) ∧ 
  (∀ x : ℝ, f x ≥ -1) ∧ 
  (∃ x : ℝ, f x = 3) ∧ 
  (∀ x : ℝ, f x ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_f_min_max_l255_25518


namespace NUMINAMATH_CALUDE_tangent_line_equation_l255_25594

/-- The equation of the tangent line to the curve y = x^3 - x + 2 at the point (1, 2) is 2x - y = 0 -/
theorem tangent_line_equation (x y : ℝ) : 
  (y = x^3 - x + 2) →  -- Curve equation
  (∃ m : ℝ, ∀ x₀ y₀ : ℝ, x₀ = 1 ∧ y₀ = 2 → y - y₀ = m * (x - x₀)) →  -- Definition of tangent line
  (∃ a b c : ℝ, a * x + b * y + c = 0 ∧ a = 2 ∧ b = -1 ∧ c = 0) :=  -- Tangent line equation
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l255_25594


namespace NUMINAMATH_CALUDE_magnitude_of_vector_difference_l255_25566

/-- Given two vectors in ℝ³, prove that the magnitude of their difference is 3 -/
theorem magnitude_of_vector_difference (a b : ℝ × ℝ × ℝ) :
  a = (1, 0, 2) → b = (0, 1, 2) →
  ‖a - 2 • b‖ = 3 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_difference_l255_25566


namespace NUMINAMATH_CALUDE_almond_butter_servings_l255_25517

def container_amount : ℚ := 34 + 3/5
def serving_size : ℚ := 5 + 1/2

theorem almond_butter_servings :
  (container_amount / serving_size : ℚ) = 6 + 21/55 := by
  sorry

end NUMINAMATH_CALUDE_almond_butter_servings_l255_25517


namespace NUMINAMATH_CALUDE_number_operations_l255_25560

theorem number_operations (x : ℤ) : 
  (((x + 7) * 3 - 12) / 6 : ℚ) = -8 → x = -19 := by
  sorry

end NUMINAMATH_CALUDE_number_operations_l255_25560


namespace NUMINAMATH_CALUDE_sons_age_l255_25565

/-- Given a man and his son, where the man is 28 years older than his son,
    and in two years the man's age will be twice the age of his son,
    prove that the present age of the son is 26 years. -/
theorem sons_age (son_age man_age : ℕ) : 
  man_age = son_age + 28 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 26 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l255_25565


namespace NUMINAMATH_CALUDE_trains_crossing_time_l255_25563

/-- The time taken for two trains to cross each other -/
theorem trains_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 120 →
  train_speed_kmh = 54 →
  (2 * train_length) / (2 * (train_speed_kmh * 1000 / 3600)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_trains_crossing_time_l255_25563


namespace NUMINAMATH_CALUDE_log_relationship_l255_25568

theorem log_relationship (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  6 * (Real.log x / Real.log a)^2 + 5 * (Real.log x / Real.log b)^2 = 12 * (Real.log x)^2 / (Real.log a * Real.log b) →
  (a = b^(5/3) ∨ a = b^(3/5)) :=
by sorry

end NUMINAMATH_CALUDE_log_relationship_l255_25568


namespace NUMINAMATH_CALUDE_division_problem_l255_25564

theorem division_problem (L S q : ℕ) : 
  L - S = 1000 → 
  L = 1100 → 
  L = S * q + 10 → 
  q = 10 := by sorry

end NUMINAMATH_CALUDE_division_problem_l255_25564


namespace NUMINAMATH_CALUDE_number_of_cans_l255_25502

/-- Proves the number of cans given space requirements before and after compaction --/
theorem number_of_cans 
  (space_before : ℝ) 
  (compaction_ratio : ℝ) 
  (total_space_after : ℝ) 
  (h1 : space_before = 30) 
  (h2 : compaction_ratio = 0.2) 
  (h3 : total_space_after = 360) : 
  ℕ :=
by
  sorry

#check number_of_cans

end NUMINAMATH_CALUDE_number_of_cans_l255_25502


namespace NUMINAMATH_CALUDE_john_excess_money_l255_25591

def earnings_day1 : ℚ := 20
def earnings_day2 : ℚ := 18
def earnings_day3 : ℚ := earnings_day2 / 2
def earnings_day4 : ℚ := earnings_day3 + (earnings_day3 * (25 / 100))
def earnings_day5 : ℚ := earnings_day4 + (earnings_day3 * (25 / 100))
def earnings_day6 : ℚ := earnings_day5 + (earnings_day5 * (15 / 100))
def earnings_day7 : ℚ := earnings_day6 - 10

def daily_increase : ℚ := 1
def pogo_stick_cost : ℚ := 60

def total_earnings : ℚ := 
  earnings_day1 + earnings_day2 + earnings_day3 + earnings_day4 + earnings_day5 + 
  earnings_day6 + earnings_day7 + 
  (earnings_day6 + daily_increase) + 
  (earnings_day6 + 2 * daily_increase) + 
  (earnings_day6 + 3 * daily_increase) + 
  (earnings_day6 + 4 * daily_increase) + 
  (earnings_day6 + 5 * daily_increase) + 
  (earnings_day6 + 6 * daily_increase) + 
  (earnings_day6 + 7 * daily_increase)

theorem john_excess_money : total_earnings - pogo_stick_cost = 170 := by
  sorry

end NUMINAMATH_CALUDE_john_excess_money_l255_25591


namespace NUMINAMATH_CALUDE_opposite_face_is_D_l255_25527

-- Define a cube net
structure CubeNet :=
  (faces : Finset Char)
  (is_valid : faces.card = 6)

-- Define a cube
structure Cube :=
  (faces : Finset Char)
  (is_valid : faces.card = 6)
  (opposite : Char → Char)
  (opposite_symm : ∀ x, opposite (opposite x) = x)

-- Define the folding operation
def fold (net : CubeNet) : Cube :=
  { faces := net.faces,
    is_valid := net.is_valid,
    opposite := sorry,
    opposite_symm := sorry }

-- Theorem statement
theorem opposite_face_is_D (net : CubeNet) 
  (h1 : net.faces = {'A', 'B', 'C', 'D', 'E', 'F'}) :
  (fold net).opposite 'A' = 'D' :=
sorry

end NUMINAMATH_CALUDE_opposite_face_is_D_l255_25527


namespace NUMINAMATH_CALUDE_ishaan_age_l255_25540

/-- Proves that Ishaan is 6 years old given the conditions of the problem -/
theorem ishaan_age (daniel_age : ℕ) (future_years : ℕ) (future_ratio : ℕ) : 
  daniel_age = 69 → 
  future_years = 15 → 
  future_ratio = 4 → 
  ∃ (ishaan_age : ℕ), 
    daniel_age + future_years = future_ratio * (ishaan_age + future_years) ∧ 
    ishaan_age = 6 := by
  sorry

end NUMINAMATH_CALUDE_ishaan_age_l255_25540


namespace NUMINAMATH_CALUDE_percentage_relation_l255_25548

theorem percentage_relation (j k l m x : ℝ) 
  (h1 : 1.25 * j = (x / 100) * k)
  (h2 : 1.5 * k = 0.5 * l)
  (h3 : 1.75 * l = 0.75 * m)
  (h4 : 0.2 * m = 7 * j) :
  x = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l255_25548


namespace NUMINAMATH_CALUDE_train_speed_in_kmh_l255_25524

-- Define the given parameters
def train_length : ℝ := 80
def bridge_length : ℝ := 295
def crossing_time : ℝ := 30

-- Define the conversion factor from m/s to km/hr
def conversion_factor : ℝ := 3.6

-- Theorem statement
theorem train_speed_in_kmh :
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / crossing_time
  let speed_kmh := speed_ms * conversion_factor
  speed_kmh = 45 := by sorry

end NUMINAMATH_CALUDE_train_speed_in_kmh_l255_25524


namespace NUMINAMATH_CALUDE_three_plus_three_cubed_l255_25559

theorem three_plus_three_cubed : 3 + 3^3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_three_plus_three_cubed_l255_25559


namespace NUMINAMATH_CALUDE_employee_count_l255_25516

/-- The number of employees in an organization (excluding the manager) -/
def num_employees : ℕ := sorry

/-- The average monthly salary of employees (excluding manager) in Rs. -/
def avg_salary : ℕ := 2000

/-- The increase in average salary when manager's salary is added, in Rs. -/
def salary_increase : ℕ := 200

/-- The manager's monthly salary in Rs. -/
def manager_salary : ℕ := 5800

theorem employee_count :
  (num_employees * avg_salary + manager_salary) / (num_employees + 1) = avg_salary + salary_increase ∧
  num_employees = 18 := by sorry

end NUMINAMATH_CALUDE_employee_count_l255_25516


namespace NUMINAMATH_CALUDE_unique_solution_condition_l255_25505

theorem unique_solution_condition (a : ℝ) : 
  (∃! x : ℝ, x^4 - a*x^3 - 3*a*x^2 + 2*a^2*x + a^2 - 2 = 0) ↔ 
  a < (3/4)^2 + 3/4 - 2 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l255_25505


namespace NUMINAMATH_CALUDE_orange_calories_l255_25511

theorem orange_calories
  (num_oranges : ℕ)
  (pieces_per_orange : ℕ)
  (num_people : ℕ)
  (calories_per_person : ℕ)
  (h1 : num_oranges = 5)
  (h2 : pieces_per_orange = 8)
  (h3 : num_people = 4)
  (h4 : calories_per_person = 100)
  : calories_per_person = num_oranges * calories_per_person / num_oranges :=
by
  sorry

end NUMINAMATH_CALUDE_orange_calories_l255_25511


namespace NUMINAMATH_CALUDE_guaranteed_win_for_given_odds_l255_25599

/-- Represents the odds for a team as a pair of natural numbers -/
def Odds := Nat × Nat

/-- Calculates the return multiplier for given odds -/
def returnMultiplier (odds : Odds) : Rat :=
  1 + odds.2 / odds.1

/-- Represents the odds for all teams in the tournament -/
structure TournamentOdds where
  team1 : Odds
  team2 : Odds
  team3 : Odds
  team4 : Odds

/-- Checks if a betting strategy exists that guarantees a win -/
def guaranteedWinExists (odds : TournamentOdds) : Prop :=
  ∃ (bet1 bet2 bet3 bet4 : Rat),
    bet1 > 0 ∧ bet2 > 0 ∧ bet3 > 0 ∧ bet4 > 0 ∧
    bet1 + bet2 + bet3 + bet4 = 1 ∧
    bet1 * returnMultiplier odds.team1 > 1 ∧
    bet2 * returnMultiplier odds.team2 > 1 ∧
    bet3 * returnMultiplier odds.team3 > 1 ∧
    bet4 * returnMultiplier odds.team4 > 1

/-- The main theorem stating that a guaranteed win exists for the given odds -/
theorem guaranteed_win_for_given_odds :
  let odds : TournamentOdds := {
    team1 := (1, 5)
    team2 := (1, 1)
    team3 := (1, 8)
    team4 := (1, 7)
  }
  guaranteedWinExists odds := by sorry

end NUMINAMATH_CALUDE_guaranteed_win_for_given_odds_l255_25599


namespace NUMINAMATH_CALUDE_function_value_sum_l255_25530

def nondecreasing (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem function_value_sum (f : ℝ → ℝ) :
  nondecreasing f 0 1 →
  f 0 = 0 →
  (∀ x, f (x / 3) = (1 / 2) * f x) →
  (∀ x, f (1 - x) = 1 - f x) →
  f 1 + f (1 / 2) + f (1 / 3) + f (1 / 6) + f (1 / 7) + f (1 / 8) = 11 / 4 := by
sorry

end NUMINAMATH_CALUDE_function_value_sum_l255_25530


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l255_25581

theorem geometric_sequence_product (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h2 : a 2 = 2) (h3 : a 6 = 8) : a 3 * a 4 * a 5 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l255_25581


namespace NUMINAMATH_CALUDE_inequality_proof_l255_25509

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 / (a^2 + a*b + b^2)) + (b^3 / (b^2 + b*c + c^2)) + (c^3 / (c^2 + c*a + a^2)) ≥ (a + b + c) / 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l255_25509


namespace NUMINAMATH_CALUDE_min_points_last_two_games_l255_25597

theorem min_points_last_two_games 
  (scores : List ℕ)
  (h1 : scores.length = 20)
  (h2 : scores[14] = 26 ∧ scores[15] = 15 ∧ scores[16] = 12 ∧ scores[17] = 24)
  (h3 : (scores.take 18).sum / 18 > (scores.take 14).sum / 14)
  (h4 : scores.sum / 20 > 20) :
  scores[18] + scores[19] ≥ 58 := by
  sorry

end NUMINAMATH_CALUDE_min_points_last_two_games_l255_25597


namespace NUMINAMATH_CALUDE_inverse_composition_nonexistence_l255_25537

theorem inverse_composition_nonexistence 
  (f h : ℝ → ℝ) 
  (h_def : ∀ x, f⁻¹ (h x) = 7 * x^2 + 4) : 
  ¬∃ y, h⁻¹ (f (-3)) = y :=
sorry

end NUMINAMATH_CALUDE_inverse_composition_nonexistence_l255_25537


namespace NUMINAMATH_CALUDE_prime_square_diff_divisible_by_24_l255_25576

theorem prime_square_diff_divisible_by_24 (p q : ℕ) (hp : Prime p) (hq : Prime q) 
  (hp_gt_3 : p > 3) (hq_gt_3 : q > 3) : 
  24 ∣ (p^2 - q^2) := by
  sorry

end NUMINAMATH_CALUDE_prime_square_diff_divisible_by_24_l255_25576


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l255_25598

/-- Given a geometric sequence {a_n} where a_{2020} = 8a_{2017}, prove that the common ratio q is 2 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- Definition of geometric sequence
  a 2020 = 8 * a 2017 →         -- Given condition
  q = 2 :=                      -- Conclusion to prove
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l255_25598


namespace NUMINAMATH_CALUDE_fair_coin_four_flips_at_least_two_tails_l255_25506

/-- The probability of getting exactly k successes in n trials with probability p -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The probability of getting at least 2 but not more than 4 tails in 4 flips of a fair coin -/
theorem fair_coin_four_flips_at_least_two_tails : 
  (binomial_probability 4 2 0.5 + binomial_probability 4 3 0.5 + binomial_probability 4 4 0.5) = 0.6875 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_four_flips_at_least_two_tails_l255_25506


namespace NUMINAMATH_CALUDE_f_of_five_equals_102_l255_25578

/-- Given a function f(x) = 2x^2 + y where f(2) = 60, prove that f(5) = 102 -/
theorem f_of_five_equals_102 (f : ℝ → ℝ) (y : ℝ) 
  (h1 : ∀ x, f x = 2 * x^2 + y)
  (h2 : f 2 = 60) : 
  f 5 = 102 := by
  sorry

end NUMINAMATH_CALUDE_f_of_five_equals_102_l255_25578


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l255_25570

theorem imaginary_part_of_complex_fraction : 
  let i : ℂ := Complex.I
  Complex.im ((1 + i) / (1 - i)) = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l255_25570


namespace NUMINAMATH_CALUDE_cube_with_holes_properties_l255_25547

/-- Represents a cube with square holes on each face -/
structure CubeWithHoles where
  edge_length : ℝ
  hole_side_length : ℝ
  hole_depth : ℝ

/-- Calculate the total surface area of a cube with holes, including inside surfaces -/
def total_surface_area (c : CubeWithHoles) : ℝ :=
  6 * c.edge_length^2 + 6 * (c.hole_side_length^2 + 4 * c.hole_side_length * c.hole_depth)

/-- Calculate the total volume of material removed from a cube due to holes -/
def total_volume_removed (c : CubeWithHoles) : ℝ :=
  6 * c.hole_side_length^2 * c.hole_depth

/-- The main theorem stating the properties of the specific cube with holes -/
theorem cube_with_holes_properties :
  let c := CubeWithHoles.mk 4 2 1
  total_surface_area c = 144 ∧ total_volume_removed c = 24 := by
  sorry


end NUMINAMATH_CALUDE_cube_with_holes_properties_l255_25547


namespace NUMINAMATH_CALUDE_max_xy_value_l255_25536

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 140) : x * y ≤ 168 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l255_25536


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l255_25546

theorem fraction_sum_simplification : (1 : ℚ) / 462 + 17 / 42 = 94 / 231 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l255_25546


namespace NUMINAMATH_CALUDE_equivalent_operations_l255_25583

theorem equivalent_operations (x : ℝ) : 
  (x * (5/6)) / (2/7) = x * (35/12) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_operations_l255_25583


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l255_25584

/-- 
If the quadratic function f(x) = -x^2 - 2x + m has a root, 
then m is greater than or equal to 1.
-/
theorem quadratic_root_condition (m : ℝ) : 
  (∃ x, -x^2 - 2*x + m = 0) → m ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l255_25584


namespace NUMINAMATH_CALUDE_lucky_lila_coincidence_l255_25582

theorem lucky_lila_coincidence (a b c d e f : ℚ) : 
  a = 2 → b = 3 → c = 4 → d = 5 → f = 6 →
  (a * b * c * d * e / f = a * (b - (c * (d - (e / f))))) →
  e = -51 / 28 := by
sorry

end NUMINAMATH_CALUDE_lucky_lila_coincidence_l255_25582


namespace NUMINAMATH_CALUDE_cake_cost_l255_25522

/-- Represents the duration of vacations in days -/
def vacation_duration_1 : ℕ := 7
def vacation_duration_2 : ℕ := 4

/-- Represents the payment in CZK for each vacation period -/
def payment_1 : ℕ := 700
def payment_2 : ℕ := 340

/-- Represents the daily rate for dog walking and rabbit feeding -/
def daily_rate : ℕ := 120

theorem cake_cost (cake_price : ℕ) : 
  (cake_price + payment_1) / vacation_duration_1 = 
  (cake_price + payment_2) / vacation_duration_2 → 
  cake_price = 140 := by
  sorry

end NUMINAMATH_CALUDE_cake_cost_l255_25522


namespace NUMINAMATH_CALUDE_intersection_sum_l255_25513

-- Define the two equations
def f (x : ℝ) : ℝ := x^3 - 3*x + 1
def g (x y : ℝ) : Prop := x + 3*y = 3

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | f p.1 = p.2 ∧ g p.1 p.2}

-- Theorem statement
theorem intersection_sum :
  ∃ (p₁ p₂ p₃ : ℝ × ℝ),
    p₁ ∈ intersection_points ∧
    p₂ ∈ intersection_points ∧
    p₃ ∈ intersection_points ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
    p₁.1 + p₂.1 + p₃.1 = 0 ∧
    p₁.2 + p₂.2 + p₃.2 = 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_sum_l255_25513


namespace NUMINAMATH_CALUDE_bike_cost_theorem_l255_25592

theorem bike_cost_theorem (marion_cost : ℕ) : 
  marion_cost = 356 → 
  (marion_cost + 2 * marion_cost + 3 * marion_cost : ℕ) = 2136 := by
  sorry

end NUMINAMATH_CALUDE_bike_cost_theorem_l255_25592


namespace NUMINAMATH_CALUDE_product_percentage_of_x_l255_25503

theorem product_percentage_of_x (w x y z : ℝ) 
  (h1 : 0.45 * z = 1.2 * y)
  (h2 : y = 0.75 * x)
  (h3 : z = 0.8 * w) :
  w * y = 1.875 * x := by
sorry

end NUMINAMATH_CALUDE_product_percentage_of_x_l255_25503


namespace NUMINAMATH_CALUDE_unique_right_angle_point_implies_radius_one_l255_25508

-- Define the circle C
def circle_C (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

-- Define points A and B
def point_A : ℝ × ℝ := (-4, 0)
def point_B : ℝ × ℝ := (4, 0)

-- Define the right angle condition
def right_angle (P : ℝ × ℝ) : Prop :=
  let PA := (P.1 - point_A.1, P.2 - point_A.2)
  let PB := (P.1 - point_B.1, P.2 - point_B.2)
  PA.1 * PB.1 + PA.2 * PB.2 = 0

-- Main theorem
theorem unique_right_angle_point_implies_radius_one (r : ℝ) :
  (∃! P, P ∈ circle_C r ∧ right_angle P) → r = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_right_angle_point_implies_radius_one_l255_25508


namespace NUMINAMATH_CALUDE_boys_who_watched_l255_25557

/-- The number of boys who went down the slide initially -/
def x : ℕ := 22

/-- The number of additional boys who went down the slide later -/
def y : ℕ := 13

/-- The total number of boys who went down the slide -/
def total_slide : ℕ := x + y

/-- The ratio of boys who went down the slide to boys who watched -/
def ratio_slide_to_watch : Rat := 5 / 3

/-- The number of boys who watched but didn't go down the slide -/
def z : ℕ := (3 * total_slide) / 5

theorem boys_who_watched (h : ratio_slide_to_watch = 5 / 3) : z = 21 := by
  sorry

end NUMINAMATH_CALUDE_boys_who_watched_l255_25557


namespace NUMINAMATH_CALUDE_quadratic_translation_l255_25577

-- Define the original function
def f (x : ℝ) : ℝ := x^2 + 3

-- Define the transformed function
def g (x : ℝ) : ℝ := (x + 2)^2 + 1

-- Theorem statement
theorem quadratic_translation (x : ℝ) : 
  g x = f (x + 2) - 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_translation_l255_25577


namespace NUMINAMATH_CALUDE_height_to_radius_ratio_l255_25589

/-- A regular triangular prism -/
structure RegularTriangularPrism where
  /-- The cosine of the dihedral angle between a face and the base -/
  cos_dihedral_angle : ℝ
  /-- The height of the prism -/
  height : ℝ
  /-- The radius of the inscribed sphere -/
  inscribed_radius : ℝ

/-- Theorem: For a regular triangular prism where the cosine of the dihedral angle 
    between a face and the base is 1/6, the ratio of the height to the radius 
    of the inscribed sphere is 7 -/
theorem height_to_radius_ratio (prism : RegularTriangularPrism) 
    (h : prism.cos_dihedral_angle = 1/6) : 
    prism.height / prism.inscribed_radius = 7 := by
  sorry

end NUMINAMATH_CALUDE_height_to_radius_ratio_l255_25589


namespace NUMINAMATH_CALUDE_computer_price_ratio_l255_25515

theorem computer_price_ratio (x : ℝ) (h : 1.3 * x = 351) :
  (x + 1.3 * x) / x = 2.3 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_ratio_l255_25515


namespace NUMINAMATH_CALUDE_pumpkin_difference_l255_25555

theorem pumpkin_difference (moonglow_pumpkins sunshine_pumpkins : ℕ) 
  (h1 : moonglow_pumpkins = 14)
  (h2 : sunshine_pumpkins = 54) :
  sunshine_pumpkins - 3 * moonglow_pumpkins = 12 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_difference_l255_25555


namespace NUMINAMATH_CALUDE_pencil_gain_percent_l255_25580

/-- 
Proves that if the cost price of 12 pencils equals the selling price of 8 pencils, 
then the gain percent is 50%.
-/
theorem pencil_gain_percent 
  (cost_price selling_price : ℝ) 
  (h : 12 * cost_price = 8 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_pencil_gain_percent_l255_25580


namespace NUMINAMATH_CALUDE_perpendicular_planes_theorem_l255_25535

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpLine : Line → Line → Prop)
variable (perpLinePlane : Line → Plane → Prop)
variable (perpPlane : Plane → Plane → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Define the theorem
theorem perpendicular_planes_theorem 
  (a b : Line) (α β : Plane) : 
  (∀ (l1 l2 : Line) (p1 p2 : Plane), 
    (perpLine l1 l2 ∧ perpLinePlane l1 p1 → ¬(parallelLinePlane l2 p1)) ∧
    (perpPlane p1 p2 ∧ parallelLinePlane l1 p1 → ¬(perpLinePlane l1 p2)) ∧
    (perpLinePlane l1 p2 ∧ perpPlane p1 p2 → ¬(parallelLinePlane l1 p1))) →
  (perpLine a b ∧ perpLinePlane a α ∧ perpLinePlane b β → perpPlane α β) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_theorem_l255_25535


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l255_25531

/-- Given a hyperbola with equation x²/a² - y² = 1 where a > 0,
    if one of its asymptotes is √3x + y = 0, then a = √3/3 -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) :
  (∃ x y : ℝ, x^2 / a^2 - y^2 = 1 ∧ Real.sqrt 3 * x + y = 0) →
  a = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l255_25531


namespace NUMINAMATH_CALUDE_bread_lasts_three_days_l255_25596

/-- Calculates the number of days bread will last for a household -/
def breadDuration (
  householdSize : ℕ
  ) (breakfastConsumption snackConsumption : ℕ
  ) (slicesPerLoaf : ℕ
  ) (availableLoaves : ℕ
  ) : ℕ :=
  let totalSlices := slicesPerLoaf * availableLoaves
  let dailyConsumption := householdSize * (breakfastConsumption + snackConsumption)
  totalSlices / dailyConsumption

/-- Theorem stating that under the given conditions, the bread will last 3 days -/
theorem bread_lasts_three_days :
  breadDuration 4 3 2 12 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_bread_lasts_three_days_l255_25596


namespace NUMINAMATH_CALUDE_quadratic_integer_intersections_l255_25575

def f (m : ℕ+) (x : ℝ) : ℝ := m * x^2 + (-m - 2) * x + 2

theorem quadratic_integer_intersections (m : ℕ+) : 
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ f m x₁ = 0 ∧ f m x₂ = 0) →
  (f m 1 = 0 ∧ f m 2 = 0 ∧ f m 0 = 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_intersections_l255_25575


namespace NUMINAMATH_CALUDE_units_digit_difference_largest_smallest_l255_25534

theorem units_digit_difference_largest_smallest (a b c d e : ℕ) :
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e ≤ 9 →
  (100 * e + 10 * d + c) - (100 * a + 10 * b + c) ≡ 0 [MOD 10] :=
by sorry

end NUMINAMATH_CALUDE_units_digit_difference_largest_smallest_l255_25534


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_l255_25519

def P (x : ℕ) : ℕ := x^6 + x^5 + x^3 + 1

theorem sum_of_prime_factors (h1 : 23 ∣ 67208001) (h2 : P 20 = 67208001) :
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (67208001 + 1))) id) = 781 :=
sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_l255_25519


namespace NUMINAMATH_CALUDE_power_calculation_l255_25520

theorem power_calculation : (-1/2 : ℚ)^2023 * 2^2022 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l255_25520


namespace NUMINAMATH_CALUDE_fraction_transformation_l255_25585

theorem fraction_transformation (x : ℝ) (h : x ≠ 1) : -1 / (1 - x) = 1 / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l255_25585


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l255_25538

theorem matrix_equation_solution :
  let M : ℂ → Matrix (Fin 2) (Fin 2) ℂ := λ x => !![3*x, 3; 2*x, x]
  ∀ x : ℂ, M x = (-6 : ℂ) • (1 : Matrix (Fin 2) (Fin 2) ℂ) ↔ x = 1 + I ∨ x = 1 - I :=
by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l255_25538


namespace NUMINAMATH_CALUDE_green_ball_probability_l255_25553

/-- Represents a container of colored balls -/
structure Container where
  red : ℕ
  green : ℕ
  blue : ℕ

/-- The probability of selecting a specific container -/
def containerProb : ℚ := 1 / 3

/-- The containers in the problem -/
def containers : List Container := [
  ⟨10, 5, 3⟩,   -- Container I
  ⟨3, 5, 2⟩,    -- Container II
  ⟨3, 5, 2⟩     -- Container III
]

/-- The probability of selecting a green ball from a given container -/
def greenProb (c : Container) : ℚ :=
  c.green / (c.red + c.green + c.blue)

/-- The total probability of selecting a green ball -/
def totalGreenProb : ℚ :=
  (containers.map (λ c ↦ containerProb * greenProb c)).sum

theorem green_ball_probability : totalGreenProb = 23 / 54 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l255_25553


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l255_25579

/-- Given two points P and Q symmetric with respect to the origin, prove that a + b = -11 --/
theorem symmetric_points_sum (a b : ℝ) :
  let P : ℝ × ℝ := (a + 3*b, 3)
  let Q : ℝ × ℝ := (-5, a + 2*b)
  (P.1 = -Q.1 ∧ P.2 = -Q.2) →
  a + b = -11 := by
sorry


end NUMINAMATH_CALUDE_symmetric_points_sum_l255_25579


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l255_25593

def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 1

theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-2 : ℝ) 2 ∧
  (∀ x, x ∈ Set.Icc (-2 : ℝ) 2 → f x ≤ f c) ∧
  f c = 23 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l255_25593


namespace NUMINAMATH_CALUDE_is_center_of_hyperbola_l255_25558

/-- The equation of the hyperbola -/
def hyperbola_eq (x y : ℝ) : Prop :=
  9 * x^2 - 36 * x - 16 * y^2 + 128 * y - 400 = 0

/-- The center of the hyperbola -/
def hyperbola_center : ℝ × ℝ := (2, 4)

/-- Theorem stating that the given point is the center of the hyperbola -/
theorem is_center_of_hyperbola :
  ∀ (x y : ℝ), hyperbola_eq x y ↔ hyperbola_eq (x - hyperbola_center.1) (y - hyperbola_center.2) :=
sorry

end NUMINAMATH_CALUDE_is_center_of_hyperbola_l255_25558
