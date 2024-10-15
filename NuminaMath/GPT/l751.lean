import Mathlib

namespace NUMINAMATH_GPT_equation_one_solution_equation_two_solution_l751_75194

theorem equation_one_solution (x : ℝ) (h : 7 * x - 20 = 2 * (3 - 3 * x)) : x = 2 :=
by {
  sorry
}

theorem equation_two_solution (x : ℝ) (h : (2 * x - 3) / 5 = (3 * x - 1) / 2 + 1) : x = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_equation_one_solution_equation_two_solution_l751_75194


namespace NUMINAMATH_GPT_truncated_pyramid_distance_l751_75126

noncomputable def distance_from_plane_to_base
  (a b : ℝ) (α : ℝ) : ℝ :=
  (a * (a - b) * Real.tan α) / (3 * a - b)

theorem truncated_pyramid_distance
  (a b : ℝ) (α : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_α : 0 < α) :
  (a * (a - b) * Real.tan α) / (3 * a - b) = distance_from_plane_to_base a b α :=
by
  sorry

end NUMINAMATH_GPT_truncated_pyramid_distance_l751_75126


namespace NUMINAMATH_GPT_min_value_of_function_l751_75144

theorem min_value_of_function (x : ℝ) (hx : x > 0) :
  ∃ y, y = (3 + x + x^2) / (1 + x) ∧ y = -1 + 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_min_value_of_function_l751_75144


namespace NUMINAMATH_GPT_keith_stored_bales_l751_75132

theorem keith_stored_bales (initial_bales added_bales final_bales : ℕ) :
  initial_bales = 22 → final_bales = 89 → final_bales = initial_bales + added_bales → added_bales = 67 :=
by
  intros h_initial h_final h_eq
  sorry

end NUMINAMATH_GPT_keith_stored_bales_l751_75132


namespace NUMINAMATH_GPT_math_problem_l751_75157

theorem math_problem : 2^5 + (5^2 / 5^1) - 3^3 = 10 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l751_75157


namespace NUMINAMATH_GPT_compute_expression_l751_75166

theorem compute_expression :
  23 ^ 12 / 23 ^ 5 + 5 = 148035894 :=
  sorry

end NUMINAMATH_GPT_compute_expression_l751_75166


namespace NUMINAMATH_GPT_minimum_value_of_a_l751_75155

theorem minimum_value_of_a :
  ∀ (x : ℝ), (2 * x + 2 / (x - 1) ≥ 7) ↔ (3 ≤ x) :=
sorry

end NUMINAMATH_GPT_minimum_value_of_a_l751_75155


namespace NUMINAMATH_GPT_D_72_l751_75123

def D (n : ℕ) : ℕ :=
  -- Definition of D(n) should be provided here
  sorry

theorem D_72 : D 72 = 121 :=
  sorry

end NUMINAMATH_GPT_D_72_l751_75123


namespace NUMINAMATH_GPT_measure_angle_C_l751_75198

noncomputable def triangle_angles_sum (a b c : ℝ) : Prop :=
  a + b + c = 180

noncomputable def angle_B_eq_twice_angle_C (b c : ℝ) : Prop :=
  b = 2 * c

noncomputable def angle_A_eq_40 : ℝ := 40

theorem measure_angle_C :
  ∀ (B C : ℝ), triangle_angles_sum angle_A_eq_40 B C → angle_B_eq_twice_angle_C B C → C = 140 / 3 :=
by
  intros B C h1 h2
  sorry

end NUMINAMATH_GPT_measure_angle_C_l751_75198


namespace NUMINAMATH_GPT_point_in_first_quadrant_l751_75176

/-- In the Cartesian coordinate system, if a point P has x-coordinate 2 and y-coordinate 4, it lies in the first quadrant. -/
theorem point_in_first_quadrant (x y : ℝ) (h1 : x = 2) (h2 : y = 4) : 
  x > 0 ∧ y > 0 → 
  (x, y).1 = 2 ∧ (x, y).2 = 4 → 
  (x > 0 ∧ y > 0) := 
by
  intros
  sorry

end NUMINAMATH_GPT_point_in_first_quadrant_l751_75176


namespace NUMINAMATH_GPT_find_the_number_l751_75177

noncomputable def special_expression (x : ℝ) : ℝ :=
  9 - 8 / x * 5 + 10

theorem find_the_number (x : ℝ) (h : special_expression x = 13.285714285714286) : x = 7 := by
  sorry

end NUMINAMATH_GPT_find_the_number_l751_75177


namespace NUMINAMATH_GPT_only_option_d_determines_location_l751_75127

-- Define the problem conditions in Lean
inductive LocationOption where
  | OptionA : LocationOption
  | OptionB : LocationOption
  | OptionC : LocationOption
  | OptionD : LocationOption

-- Define a function that takes a LocationOption and returns whether it can determine a specific location
def determine_location (option : LocationOption) : Prop :=
  match option with
  | LocationOption.OptionD => True
  | LocationOption.OptionA => False
  | LocationOption.OptionB => False
  | LocationOption.OptionC => False

-- Prove that only option D can determine a specific location
theorem only_option_d_determines_location : ∀ (opt : LocationOption), determine_location opt ↔ opt = LocationOption.OptionD := by
  intro opt
  cases opt
  · simp [determine_location, LocationOption.OptionA]
  · simp [determine_location, LocationOption.OptionB]
  · simp [determine_location, LocationOption.OptionC]
  · simp [determine_location, LocationOption.OptionD]

end NUMINAMATH_GPT_only_option_d_determines_location_l751_75127


namespace NUMINAMATH_GPT_rational_point_partition_exists_l751_75103

open Set

-- Define rational numbers
noncomputable def Q : Set ℚ :=
  {x | True}

-- Define the set of rational points in the plane
def I : Set (ℚ × ℚ) := 
  {p | p.1 ∈ Q ∧ p.2 ∈ Q}

-- Statement of the theorem
theorem rational_point_partition_exists :
  ∃ (A B : Set (ℚ × ℚ)),
    (∀ (y : ℚ), {p ∈ A | p.1 = y}.Finite) ∧
    (∀ (x : ℚ), {p ∈ B | p.2 = x}.Finite) ∧
    (A ∪ B = I) ∧
    (A ∩ B = ∅) :=
sorry

end NUMINAMATH_GPT_rational_point_partition_exists_l751_75103


namespace NUMINAMATH_GPT_Faye_total_pencils_l751_75179

def pencils_per_row : ℕ := 8
def number_of_rows : ℕ := 4
def total_pencils : ℕ := pencils_per_row * number_of_rows

theorem Faye_total_pencils : total_pencils = 32 := by
  sorry

end NUMINAMATH_GPT_Faye_total_pencils_l751_75179


namespace NUMINAMATH_GPT_max_area_of_rectangle_with_perimeter_60_l751_75183

theorem max_area_of_rectangle_with_perimeter_60 :
  ∃ (x y : ℝ), 2 * (x + y) = 60 ∧ x * y = 225 :=
by
  sorry

end NUMINAMATH_GPT_max_area_of_rectangle_with_perimeter_60_l751_75183


namespace NUMINAMATH_GPT_range_of_sum_l751_75193

theorem range_of_sum (x y : ℝ) (h : 9 * x^2 + 16 * y^2 = 144) : 
  ∃ a b : ℝ, (x + y + 10 ≥ a) ∧ (x + y + 10 ≤ b) ∧ a = 5 ∧ b = 15 := 
sorry

end NUMINAMATH_GPT_range_of_sum_l751_75193


namespace NUMINAMATH_GPT_lcm_18_24_eq_72_l751_75138

-- Conditions
def factorization_18 : Nat × Nat := (1, 2) -- 18 = 2^1 * 3^2
def factorization_24 : Nat × Nat := (3, 1) -- 24 = 2^3 * 3^1

-- Definition of LCM using the highest powers from factorizations
def LCM (a b : Nat × Nat) : Nat :=
  let (p1, q1) := a
  let (p2, q2) := b
  (2^max p1 p2) * (3^max q1 q2)

-- Proof statement
theorem lcm_18_24_eq_72 : LCM factorization_18 factorization_24 = 72 :=
by
  sorry

end NUMINAMATH_GPT_lcm_18_24_eq_72_l751_75138


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l751_75186

-- Define the conditions as predicates
def p (x : ℝ) : Prop := x^2 - 3 * x - 4 ≤ 0
def q (x m : ℝ) : Prop := x^2 - 6 * x + 9 - m^2 ≤ 0

-- Range for m where p is sufficient but not necessary for q
def m_range (m : ℝ) : Prop := m ≤ -4 ∨ m ≥ 4

-- The main goal to be proven
theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∀ x, p x → q x m) ∧ ¬(∀ x, q x m → p x) ↔ m_range m :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l751_75186


namespace NUMINAMATH_GPT_percentage_decrease_in_price_l751_75101

theorem percentage_decrease_in_price (original_price new_price decrease percentage : ℝ) :
  original_price = 1300 → new_price = 988 →
  decrease = original_price - new_price →
  percentage = (decrease / original_price) * 100 →
  percentage = 24 := by
  sorry

end NUMINAMATH_GPT_percentage_decrease_in_price_l751_75101


namespace NUMINAMATH_GPT_inequality_solution_l751_75171

theorem inequality_solution (a : ℝ) :
  (∀ x : ℝ, a * x^2 + (a - 1) * x + (a - 1) < 0) ↔ a < -1/3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l751_75171


namespace NUMINAMATH_GPT_quadratic_solution_l751_75100

theorem quadratic_solution (x : ℝ) : x^2 - 2 * x = 0 ↔ (x = 0 ∨ x = 2) := by
  sorry

end NUMINAMATH_GPT_quadratic_solution_l751_75100


namespace NUMINAMATH_GPT_difference_in_percentage_l751_75139

noncomputable def principal : ℝ := 600
noncomputable def timePeriod : ℝ := 10
noncomputable def interestDifference : ℝ := 300

theorem difference_in_percentage (R D : ℝ) (h : 60 * (R + D) - 60 * R = 300) : D = 5 := 
by
  -- Proof is not provided, as instructed
  sorry

end NUMINAMATH_GPT_difference_in_percentage_l751_75139


namespace NUMINAMATH_GPT_tan_30_eq_sqrt3_div_3_l751_75104

/-- Statement that proves the value of tang of 30 degrees, given the cosine
    and sine values. -/
theorem tan_30_eq_sqrt3_div_3 
  (cos_30 : Real) (sin_30 : Real) 
  (hcos : cos_30 = Real.sqrt 3 / 2) 
  (hsin : sin_30 = 1 / 2) : 
    Real.tan 30 = Real.sqrt 3 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_tan_30_eq_sqrt3_div_3_l751_75104


namespace NUMINAMATH_GPT_b_2016_value_l751_75168

theorem b_2016_value : 
  ∃ (a b : ℕ → ℝ), 
    a 1 = 1 / 2 ∧ 
    (∀ n : ℕ, 0 < n → a n + b n = 1) ∧
    (∀ n : ℕ, 0 < n → b (n + 1) = b n / (1 - (a n)^2)) → 
    b 2016 = 2016 / 2017 :=
by
  sorry

end NUMINAMATH_GPT_b_2016_value_l751_75168


namespace NUMINAMATH_GPT_woman_work_completion_days_l751_75120

def work_completion_days_man := 6
def work_completion_days_boy := 9
def work_completion_days_combined := 3

theorem woman_work_completion_days : 
  (1 / work_completion_days_man + W + 1 / work_completion_days_boy = 1 / work_completion_days_combined) →
  W = 1 / 18 → 
  1 / W = 18 :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_woman_work_completion_days_l751_75120


namespace NUMINAMATH_GPT_dad_additional_money_l751_75173

-- Define the conditions in Lean
def daily_savings : ℕ := 35
def days : ℕ := 7
def total_savings_before_doubling := daily_savings * days
def doubled_savings := 2 * total_savings_before_doubling
def total_amount_after_7_days : ℕ := 500

-- Define the theorem to prove
theorem dad_additional_money : (total_amount_after_7_days - doubled_savings) = 10 := by
  sorry

end NUMINAMATH_GPT_dad_additional_money_l751_75173


namespace NUMINAMATH_GPT_find_savings_l751_75112

-- Define the problem statement
def income_expenditure_problem (income expenditure : ℝ) (ratio : ℝ) : Prop :=
  (income / ratio = expenditure) ∧ (income = 20000)

-- Define the theorem for savings
theorem find_savings (income expenditure : ℝ) (ratio : ℝ) (h_ratio : ratio = 4 / 5) (h_income : income = 20000) : 
  income_expenditure_problem income expenditure ratio → income - expenditure = 4000 :=
by
  sorry

end NUMINAMATH_GPT_find_savings_l751_75112


namespace NUMINAMATH_GPT_scalene_triangle_angles_l751_75172

theorem scalene_triangle_angles (x y z : ℝ) (h1 : x + y + z = 180) (h2 : x ≠ y ∧ y ≠ z ∧ x ≠ z)
(h3 : x = 36 ∨ y = 36 ∨ z = 36) (h4 : x = 2 * y ∨ y = 2 * x ∨ z = 2 * x ∨ x = 2 * z ∨ y = 2 * z ∨ z = 2 * y) :
(x = 36 ∧ y = 48 ∧ z = 96) ∨ (x = 18 ∧ y = 36 ∧ z = 126) ∨ (x = 36 ∧ z = 48 ∧ y = 96) ∨ (y = 18 ∧ x = 36 ∧ z = 126) :=
sorry

end NUMINAMATH_GPT_scalene_triangle_angles_l751_75172


namespace NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l751_75158

-- Define the function f for a general a
def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a) + 1|

-- Part 1: When a = 2
theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : x ≤ 3/2 ∨ x ≥ 11/2 :=
  sorry

-- Part 2: Range of values for a
theorem part2_range_of_a (a : ℝ) (h : ∀ x, f x a ≥ 4) : a ≤ -1 ∨ a ≥ 3 :=
  sorry

end NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l751_75158


namespace NUMINAMATH_GPT_f_at_neg_one_l751_75115

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x^2 + 3 * x + 16

noncomputable def f_with_r (x : ℝ) (a r : ℝ) : ℝ := (x^3 + a * x^2 + 3 * x + 16) * (x - r)

theorem f_at_neg_one (a b c r : ℝ) (h1 : ∀ x, g x a = 0 → f_with_r x a r = 0)
  (h2 : a - r = 5) (h3 : 16 - 3 * r = 150) (h4 : -16 * r = c) :
  f_with_r (-1) a r = -1347 :=
by
  sorry

end NUMINAMATH_GPT_f_at_neg_one_l751_75115


namespace NUMINAMATH_GPT_group_C_forms_triangle_l751_75151

theorem group_C_forms_triangle :
  ∀ (a b c : ℕ), (a + b > c ∧ a + c > b ∧ b + c > a) ↔ ((a, b, c) = (2, 3, 4)) :=
by
  -- we'll prove the forward and backward directions separately
  sorry

end NUMINAMATH_GPT_group_C_forms_triangle_l751_75151


namespace NUMINAMATH_GPT_probability_same_color_of_two_12_sided_dice_l751_75196

-- Define the conditions
def sides := 12
def red_sides := 3
def blue_sides := 5
def green_sides := 3
def golden_sides := 1

-- Calculate the probabilities for each color being rolled
def pr_both_red := (red_sides / sides) ^ 2
def pr_both_blue := (blue_sides / sides) ^ 2
def pr_both_green := (green_sides / sides) ^ 2
def pr_both_golden := (golden_sides / sides) ^ 2

-- Total probability calculation
def total_probability_same_color := pr_both_red + pr_both_blue + pr_both_green + pr_both_golden

theorem probability_same_color_of_two_12_sided_dice :
  total_probability_same_color = 11 / 36 := by
  sorry

end NUMINAMATH_GPT_probability_same_color_of_two_12_sided_dice_l751_75196


namespace NUMINAMATH_GPT_farm_field_area_l751_75169

theorem farm_field_area
  (planned_daily_plough : ℕ)
  (actual_daily_plough : ℕ)
  (extra_days : ℕ)
  (remaining_area : ℕ)
  (total_days_hectares : ℕ → ℕ) :
  planned_daily_plough = 260 →
  actual_daily_plough = 85 →
  extra_days = 2 →
  remaining_area = 40 →
  total_days_hectares (total_days_hectares (1 + 2) * 85 + 40) = 312 :=
by
  sorry

end NUMINAMATH_GPT_farm_field_area_l751_75169


namespace NUMINAMATH_GPT_root_of_equation_imp_expression_eq_one_l751_75182

variable (m : ℝ)

theorem root_of_equation_imp_expression_eq_one
  (h : m^2 - m - 1 = 0) : m^2 - m = 1 :=
  sorry

end NUMINAMATH_GPT_root_of_equation_imp_expression_eq_one_l751_75182


namespace NUMINAMATH_GPT_beads_initial_state_repeats_l751_75190

-- Define the setup of beads on a circular wire
structure BeadConfig (n : ℕ) :=
(beads : Fin n → ℝ)  -- Each bead's position indexed by a finite set, ℝ denotes angular position

-- Define the instantaneous collision swapping function
def swap (n : ℕ) (i j : Fin n) (config : BeadConfig n) : BeadConfig n :=
⟨fun k => if k = i then config.beads j else if k = j then config.beads i else config.beads k⟩

-- Define what it means for a configuration to return to its initial state
def returns_to_initial (n : ℕ) (initial : BeadConfig n) (t : ℝ) : Prop :=
  ∃ (config : BeadConfig n), (∀ k, config.beads k = initial.beads k) ∧ (config = initial)

-- Specification of the problem
theorem beads_initial_state_repeats (n : ℕ) (initial : BeadConfig n) (ω : Fin n → ℝ) :
  (∀ k, ω k > 0) →  -- condition that all beads have positive angular speed, either clockwise or counterclockwise
  ∃ t : ℝ, t > 0 ∧ returns_to_initial n initial t := 
by
  sorry

end NUMINAMATH_GPT_beads_initial_state_repeats_l751_75190


namespace NUMINAMATH_GPT_johns_age_l751_75156

-- Define variables for ages of John and Matt
variables (J M : ℕ)

-- Define the conditions based on the problem statement
def condition1 : Prop := M = 4 * J - 3
def condition2 : Prop := J + M = 52

-- The goal: prove that John is 11 years old
theorem johns_age (J M : ℕ) (h1 : condition1 J M) (h2 : condition2 J M) : J = 11 := by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_johns_age_l751_75156


namespace NUMINAMATH_GPT_smallest_integer_mod_inverse_l751_75134

theorem smallest_integer_mod_inverse (n : ℕ) (h1 : n > 1) (h2 : gcd n 1001 = 1) : n = 2 :=
sorry

end NUMINAMATH_GPT_smallest_integer_mod_inverse_l751_75134


namespace NUMINAMATH_GPT_geometric_sequence_b_l751_75102

theorem geometric_sequence_b (a b c : Real) (h1 : a = 5 + 2 * Real.sqrt 6) (h2 : c = 5 - 2 * Real.sqrt 6) (h3 : ∃ r, b = r * a ∧ c = r * b) :
  b = 1 ∨ b = -1 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_b_l751_75102


namespace NUMINAMATH_GPT_find_divisor_l751_75146

theorem find_divisor (q r D : ℕ) (hq : q = 120) (hr : r = 333) (hD : 55053 = D * q + r) : D = 456 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l751_75146


namespace NUMINAMATH_GPT_perfect_square_trinomial_l751_75125

theorem perfect_square_trinomial (m : ℤ) : 
  (x^2 - (m - 3) * x + 16 = (x - 4)^2) ∨ (x^2 - (m - 3) * x + 16 = (x + 4)^2) ↔ (m = -5 ∨ m = 11) := by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l751_75125


namespace NUMINAMATH_GPT_cookie_cost_per_day_l751_75113

theorem cookie_cost_per_day
    (days_in_April : ℕ)
    (cookies_per_day : ℕ)
    (total_spent : ℕ)
    (total_cookies : ℕ := days_in_April * cookies_per_day)
    (cost_per_cookie : ℕ := total_spent / total_cookies) :
  days_in_April = 30 ∧ cookies_per_day = 3 ∧ total_spent = 1620 → cost_per_cookie = 18 :=
by
  sorry

end NUMINAMATH_GPT_cookie_cost_per_day_l751_75113


namespace NUMINAMATH_GPT_set_difference_example_l751_75165

-- Define P and Q based on the given conditions
def P : Set ℝ := {x | 0 < x ∧ x < 2}
def Q : Set ℝ := {x | 1 < x ∧ x < 3}

-- State the theorem: P - Q equals to the set {x | 0 < x ≤ 1}
theorem set_difference_example : P \ Q = {x | 0 < x ∧ x ≤ 1} := 
  by
  sorry

end NUMINAMATH_GPT_set_difference_example_l751_75165


namespace NUMINAMATH_GPT_truck_initial_gas_ratio_l751_75118

-- Definitions and conditions
def truck_total_capacity : ℕ := 20

def car_total_capacity : ℕ := 12

def car_initial_gas : ℕ := car_total_capacity / 3

def added_gas : ℕ := 18

-- Goal: The ratio of the gas in the truck's tank to its total capacity before she fills it up is 1:2
theorem truck_initial_gas_ratio :
  ∃ T : ℕ, (T + car_initial_gas + added_gas = truck_total_capacity + car_total_capacity) ∧ (T : ℚ) / truck_total_capacity = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_truck_initial_gas_ratio_l751_75118


namespace NUMINAMATH_GPT_roses_count_l751_75143

def total_roses : Nat := 80
def red_roses : Nat := 3 * total_roses / 4
def remaining_roses : Nat := total_roses - red_roses
def yellow_roses : Nat := remaining_roses / 4
def white_roses : Nat := remaining_roses - yellow_roses

theorem roses_count :
  red_roses + white_roses = 75 :=
by
  sorry

end NUMINAMATH_GPT_roses_count_l751_75143


namespace NUMINAMATH_GPT_carlos_more_miles_than_dana_after_3_hours_l751_75121

-- Define the conditions
variable (carlos_total_distance : ℕ)
variable (carlos_advantage : ℕ)
variable (dana_total_distance : ℕ)
variable (time_hours : ℕ)

-- State the condition values that are given in the problem
def conditions : Prop :=
  carlos_total_distance = 50 ∧
  carlos_advantage = 5 ∧
  dana_total_distance = 40 ∧
  time_hours = 3

-- State the proof goal
theorem carlos_more_miles_than_dana_after_3_hours
  (h : conditions carlos_total_distance carlos_advantage dana_total_distance time_hours) :
  carlos_total_distance - dana_total_distance = 10 :=
by
  sorry

end NUMINAMATH_GPT_carlos_more_miles_than_dana_after_3_hours_l751_75121


namespace NUMINAMATH_GPT_carlton_outfit_count_l751_75110

-- Definitions of conditions
def sweater_vests (s : ℕ) : ℕ := 2 * s
def button_up_shirts : ℕ := 3
def outfits (v s : ℕ) : ℕ := v * s

-- Theorem statement
theorem carlton_outfit_count : outfits (sweater_vests button_up_shirts) button_up_shirts = 18 :=
by
  sorry

end NUMINAMATH_GPT_carlton_outfit_count_l751_75110


namespace NUMINAMATH_GPT_train_length_correct_l751_75175

noncomputable def speed_kmph : ℝ := 60
noncomputable def time_sec : ℝ := 6

-- Conversion factor from km/hr to m/s
noncomputable def conversion_factor := (1000 : ℝ) / 3600

-- Speed in m/s
noncomputable def speed_mps := speed_kmph * conversion_factor

-- Length of the train
noncomputable def train_length := speed_mps * time_sec

theorem train_length_correct :
  train_length = 100.02 :=
by
  sorry

end NUMINAMATH_GPT_train_length_correct_l751_75175


namespace NUMINAMATH_GPT_thabo_paperback_diff_l751_75161

variable (total_books : ℕ) (H_books : ℕ) (P_books : ℕ) (F_books : ℕ)

def thabo_books_conditions :=
  total_books = 160 ∧
  H_books = 25 ∧
  P_books > H_books ∧
  F_books = 2 * P_books ∧
  total_books = F_books + P_books + H_books 

theorem thabo_paperback_diff :
  thabo_books_conditions total_books H_books P_books F_books → 
  (P_books - H_books) = 20 :=
by
  sorry

end NUMINAMATH_GPT_thabo_paperback_diff_l751_75161


namespace NUMINAMATH_GPT_tracy_first_week_books_collected_l751_75117

-- Definitions for collection multipliers
def first_week (T : ℕ) := T
def second_week (T : ℕ) := 2 * T + 3 * T
def third_week (T : ℕ) := 3 * T + 4 * T + (T / 2)
def fourth_week (T : ℕ) := 4 * T + 5 * T + T
def fifth_week (T : ℕ) := 5 * T + 6 * T + 2 * T
def sixth_week (T : ℕ) := 6 * T + 7 * T + 3 * T

-- Summing up total books collected
def total_books_collected (T : ℕ) : ℕ :=
  first_week T + second_week T + third_week T + fourth_week T + fifth_week T + sixth_week T

-- Proof statement (unchanged for now)
theorem tracy_first_week_books_collected (T : ℕ) :
  total_books_collected T = 1025 → T = 20 :=
by
  sorry

end NUMINAMATH_GPT_tracy_first_week_books_collected_l751_75117


namespace NUMINAMATH_GPT_cricket_bat_cost_price_l751_75189

theorem cricket_bat_cost_price (CP_A : ℝ) (SP_B : ℝ) (SP_C : ℝ) (h1 : SP_B = CP_A * 1.20) (h2 : SP_C = SP_B * 1.25) (h3 : SP_C = 222) : CP_A = 148 := 
by
  sorry

end NUMINAMATH_GPT_cricket_bat_cost_price_l751_75189


namespace NUMINAMATH_GPT_simplify_expression_l751_75197

-- Define the original expression and the simplified version
def original_expr (x y : ℤ) : ℤ := 7 * x + 3 - 2 * x + 15 + y
def simplified_expr (x y : ℤ) : ℤ := 5 * x + y + 18

-- The equivalence to be proved
theorem simplify_expression (x y : ℤ) : original_expr x y = simplified_expr x y :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l751_75197


namespace NUMINAMATH_GPT_exist_three_sum_eq_third_l751_75149

theorem exist_three_sum_eq_third
  (A : Finset ℕ)
  (h_card : A.card = 52)
  (h_cond : ∀ (a : ℕ), a ∈ A → a ≤ 100) :
  ∃ (x y z : ℕ), x ∈ A ∧ y ∈ A ∧ z ∈ A ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x + y = z :=
sorry

end NUMINAMATH_GPT_exist_three_sum_eq_third_l751_75149


namespace NUMINAMATH_GPT_ratio_pages_l751_75188

theorem ratio_pages (pages_Selena pages_Harry : ℕ) (h₁ : pages_Selena = 400) (h₂ : pages_Harry = 180) : 
  pages_Harry / pages_Selena = 9 / 20 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_ratio_pages_l751_75188


namespace NUMINAMATH_GPT_compare_abc_l751_75136

noncomputable def a : ℝ := Real.log 5 / Real.log 2
noncomputable def b : ℝ := Real.log 6 / Real.log 2
noncomputable def c : ℝ := 9 ^ (1 / 2 : ℝ)

theorem compare_abc : c > b ∧ b > a := 
by
  sorry

end NUMINAMATH_GPT_compare_abc_l751_75136


namespace NUMINAMATH_GPT_interest_rate_l751_75111

theorem interest_rate (SI P : ℝ) (T : ℕ) (h₁: SI = 70) (h₂ : P = 700) (h₃ : T = 4) : 
  (SI / (P * T)) * 100 = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_l751_75111


namespace NUMINAMATH_GPT_cinematic_academy_member_count_l751_75107

theorem cinematic_academy_member_count (M : ℝ) 
  (h : (1 / 4) * M = 192.5) : M = 770 := 
by 
  -- proof omitted
  sorry

end NUMINAMATH_GPT_cinematic_academy_member_count_l751_75107


namespace NUMINAMATH_GPT_simplify_abs_expr_l751_75192

theorem simplify_abs_expr : |(-4 ^ 2 + 6)| = 10 := by
  sorry

end NUMINAMATH_GPT_simplify_abs_expr_l751_75192


namespace NUMINAMATH_GPT_max_jars_in_crate_l751_75180

-- Define the conditions given in the problem
def side_length_cardboard_box := 20 -- in cm
def jars_per_box := 8
def crate_width := 80 -- in cm
def crate_length := 120 -- in cm
def crate_height := 60 -- in cm
def volume_box := side_length_cardboard_box ^ 3
def volume_crate := crate_width * crate_length * crate_height
def boxes_per_crate := volume_crate / volume_box
def max_jars_per_crate := boxes_per_crate * jars_per_box

-- Statement that needs to be proved
theorem max_jars_in_crate : max_jars_per_crate = 576 := sorry

end NUMINAMATH_GPT_max_jars_in_crate_l751_75180


namespace NUMINAMATH_GPT_molecular_weight_1_mole_l751_75147

-- Define the molecular weight of 3 moles
def molecular_weight_3_moles : ℕ := 222

-- Prove that the molecular weight of 1 mole is 74 given the molecular weight of 3 moles
theorem molecular_weight_1_mole (mw3 : ℕ) (h : mw3 = 222) : mw3 / 3 = 74 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_1_mole_l751_75147


namespace NUMINAMATH_GPT_num_distinct_five_digit_integers_with_product_of_digits_18_l751_75116

theorem num_distinct_five_digit_integers_with_product_of_digits_18 :
  ∃ (n : ℕ), n = 70 ∧ ∀ (a b c d e : ℕ),
    a * b * c * d * e = 18 ∧ 
    1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9 ∧ 1 ≤ e ∧ e ≤ 9 → 
    (∃ (s : Finset (Fin 100000)), s.card = n) :=
  sorry

end NUMINAMATH_GPT_num_distinct_five_digit_integers_with_product_of_digits_18_l751_75116


namespace NUMINAMATH_GPT_ratio_of_two_numbers_l751_75150

theorem ratio_of_two_numbers (A B : ℕ) (x y : ℕ) (h1 : lcm A B = 60) (h2 : A + B = 50) (h3 : A / B = x / y) (hx : x = 3) (hy : y = 2) : x = 3 ∧ y = 2 := 
by
  -- Conditions provided in the problem
  sorry

end NUMINAMATH_GPT_ratio_of_two_numbers_l751_75150


namespace NUMINAMATH_GPT_equilibrium_stability_l751_75195

noncomputable def f (x : ℝ) : ℝ := x * (Real.exp x - 2)

theorem equilibrium_stability (x : ℝ) :
  (x = 0 → HasDerivAt f (-1) 0 ∧ (-1 < 0)) ∧
  (x = Real.log 2 → HasDerivAt f (2 * Real.log 2) (Real.log 2) ∧ (2 * Real.log 2 > 0)) :=
by
  sorry

end NUMINAMATH_GPT_equilibrium_stability_l751_75195


namespace NUMINAMATH_GPT_bike_travel_distance_l751_75159

def avg_speed : ℝ := 3  -- average speed in m/s
def time : ℝ := 7       -- time in seconds

theorem bike_travel_distance : avg_speed * time = 21 := by
  sorry

end NUMINAMATH_GPT_bike_travel_distance_l751_75159


namespace NUMINAMATH_GPT_find_g_l751_75181

theorem find_g (x : ℝ) (g : ℝ → ℝ) :
  2 * x^5 - 4 * x^3 + 3 * x^2 + g x = 7 * x^4 - 5 * x^3 + x^2 - 9 * x + 2 →
  g x = -2 * x^5 + 7 * x^4 - x^3 - 2 * x^2 - 9 * x + 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_g_l751_75181


namespace NUMINAMATH_GPT_boxes_needed_l751_75130

def num_red_pencils := 45
def num_yellow_pencils := 80
def num_pencils_per_red_box := 15
def num_pencils_per_blue_box := 25
def num_pencils_per_yellow_box := 10
def num_pencils_per_green_box := 30

def num_blue_pencils (x : Nat) := 3 * x + 6
def num_green_pencils (red : Nat) (blue : Nat) := 2 * (red + blue)

def total_boxes_needed : Nat :=
  let red_boxes := num_red_pencils / num_pencils_per_red_box
  let blue_boxes := (num_blue_pencils num_red_pencils) / num_pencils_per_blue_box + 
                    if ((num_blue_pencils num_red_pencils) % num_pencils_per_blue_box) = 0 then 0 else 1
  let yellow_boxes := num_yellow_pencils / num_pencils_per_yellow_box
  let green_boxes := (num_green_pencils num_red_pencils (num_blue_pencils num_red_pencils)) / num_pencils_per_green_box + 
                     if ((num_green_pencils num_red_pencils (num_blue_pencils num_red_pencils)) % num_pencils_per_green_box) = 0 then 0 else 1
  red_boxes + blue_boxes + yellow_boxes + green_boxes

theorem boxes_needed : total_boxes_needed = 30 := sorry

end NUMINAMATH_GPT_boxes_needed_l751_75130


namespace NUMINAMATH_GPT_measured_diagonal_length_l751_75122

theorem measured_diagonal_length (a b c d diag : Real)
  (h1 : a = 1) (h2 : b = 2) (h3 : c = 2.8) (h4 : d = 5) (hd : diag = 7.5) :
  diag = 2.8 :=
sorry

end NUMINAMATH_GPT_measured_diagonal_length_l751_75122


namespace NUMINAMATH_GPT_remainder_13_pow_2000_mod_1000_l751_75129

theorem remainder_13_pow_2000_mod_1000 :
  (13^2000) % 1000 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_13_pow_2000_mod_1000_l751_75129


namespace NUMINAMATH_GPT_smallest_n_for_purple_l751_75106

-- The conditions as definitions
def red := 18
def green := 20
def blue := 22
def purple_cost := 24

-- The mathematical proof problem statement
theorem smallest_n_for_purple : 
  ∃ n : ℕ, purple_cost * n = Nat.lcm (Nat.lcm red green) blue ∧
            ∀ m : ℕ, (purple_cost * m = Nat.lcm (Nat.lcm red green) blue → m ≥ n) ↔ n = 83 := 
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_purple_l751_75106


namespace NUMINAMATH_GPT_interest_calculated_years_l751_75153

variable (P T : ℝ)

-- Given conditions
def principal_sum_positive : Prop := P > 0
def simple_interest_condition : Prop := (P * 5 * T) / 100 = P / 5

-- Theorem statement
theorem interest_calculated_years (h1 : principal_sum_positive P) (h2 : simple_interest_condition P T) : T = 4 :=
  sorry

end NUMINAMATH_GPT_interest_calculated_years_l751_75153


namespace NUMINAMATH_GPT_sum_of_squares_l751_75162

theorem sum_of_squares :
  ∃ p q r s t u : ℤ, (∀ x : ℤ, 729 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) ∧ 
    (p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 8210) :=
sorry

end NUMINAMATH_GPT_sum_of_squares_l751_75162


namespace NUMINAMATH_GPT_lowry_earnings_l751_75163

def small_bonsai_cost : ℕ := 30
def big_bonsai_cost : ℕ := 20
def small_bonsai_sold : ℕ := 3
def big_bonsai_sold : ℕ := 5

def total_earnings (small_cost : ℕ) (big_cost : ℕ) (small_sold : ℕ) (big_sold : ℕ) : ℕ :=
  small_cost * small_sold + big_cost * big_sold

theorem lowry_earnings :
  total_earnings small_bonsai_cost big_bonsai_cost small_bonsai_sold big_bonsai_sold = 190 := 
by
  sorry

end NUMINAMATH_GPT_lowry_earnings_l751_75163


namespace NUMINAMATH_GPT_part1_part2_l751_75141

open Real

noncomputable def f (x : ℝ) : ℝ :=
  2 * x - (x + 1) * log x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ :=
  x * log x - a * x^2 - 1

/- First part: Prove that for all x \in (1, +\infty), f(x) < 2 -/
theorem part1 (x : ℝ) (hx : 1 < x) : f x < 2 := sorry

/- Second part: Prove that if g(x) = 0 has two roots x₁ and x₂, then 
   (log x₁ + log x₂) / 2 > 1 + 2 / sqrt (x₁ * x₂) -/
theorem part2 (a x₁ x₂ : ℝ) (hx₁ : g x₁ a = 0) (hx₂ : g x₂ a = 0) : 
  (log x₁ + log x₂) / 2 > 1 + 2 / sqrt (x₁ * x₂) := sorry

end NUMINAMATH_GPT_part1_part2_l751_75141


namespace NUMINAMATH_GPT_tangent_periodic_solution_l751_75109

theorem tangent_periodic_solution :
  ∃ n : ℤ, -180 < n ∧ n < 180 ∧ (Real.tan (n * Real.pi / 180) = Real.tan (345 * Real.pi / 180)) := by
  sorry

end NUMINAMATH_GPT_tangent_periodic_solution_l751_75109


namespace NUMINAMATH_GPT_find_n_l751_75108

noncomputable def factorial : ℕ → ℕ
| 0 => 1
| (n+1) => (n+1) * factorial n

theorem find_n (n : ℕ) (h : n * factorial (n + 1) + factorial (n + 1) = 5040) : n = 5 :=
sorry

end NUMINAMATH_GPT_find_n_l751_75108


namespace NUMINAMATH_GPT_sum_x_y_z_w_l751_75167

-- Define the conditions in Lean
variables {x y z w : ℤ}
axiom h1 : x - y + z = 7
axiom h2 : y - z + w = 8
axiom h3 : z - w + x = 4
axiom h4 : w - x + y = 3

-- Prove the result
theorem sum_x_y_z_w : x + y + z + w = 22 := by
  sorry

end NUMINAMATH_GPT_sum_x_y_z_w_l751_75167


namespace NUMINAMATH_GPT_problem_condition_implies_statement_l751_75191

variable {a b c : ℝ}

theorem problem_condition_implies_statement :
  a^3 + a * b + a * c < 0 → b^5 - 4 * a * c > 0 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_problem_condition_implies_statement_l751_75191


namespace NUMINAMATH_GPT_sum_of_24_terms_l751_75128

variable (a_1 d : ℝ)

def a (n : ℕ) : ℝ := a_1 + (n - 1) * d

theorem sum_of_24_terms 
  (h : (a 5 + a 10 + a 15 + a 20 = 20)) : 
  (12 * (2 * a_1 + 23 * d) = 120) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_24_terms_l751_75128


namespace NUMINAMATH_GPT_number_of_meetings_l751_75114

-- Define the data for the problem
def pool_length : ℕ := 120
def swimmer_A_speed : ℕ := 4
def swimmer_B_speed : ℕ := 3
def total_time_seconds : ℕ := 15 * 60
def swimmer_A_turn_break_seconds : ℕ := 2
def swimmer_B_turn_break_seconds : ℕ := 0

-- Define the round trip time for each swimmer
def swimmer_A_round_trip_time : ℕ := 2 * (pool_length / swimmer_A_speed) + 2 * swimmer_A_turn_break_seconds
def swimmer_B_round_trip_time : ℕ := 2 * (pool_length / swimmer_B_speed) + 2 * swimmer_B_turn_break_seconds

-- Define the least common multiple of the round trip times
def lcm_round_trip_time : ℕ := Nat.lcm swimmer_A_round_trip_time swimmer_B_round_trip_time

-- Define the statement to prove
theorem number_of_meetings (lcm_round_trip_time : ℕ) : 
  (24 * (total_time_seconds / lcm_round_trip_time) + ((total_time_seconds % lcm_round_trip_time) / (pool_length / (swimmer_A_speed + swimmer_B_speed)))) = 51 := 
sorry

end NUMINAMATH_GPT_number_of_meetings_l751_75114


namespace NUMINAMATH_GPT_sum_mod_20_l751_75199

theorem sum_mod_20 : 
  (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92 + 93 + 94) % 20 = 15 :=
by 
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_sum_mod_20_l751_75199


namespace NUMINAMATH_GPT_y_eq_fraction_x_l751_75145

theorem y_eq_fraction_x (p : ℝ) (x y : ℝ) (hx : x = 1 + 2^p) (hy : y = 1 + 2^(-p)) : y = x / (x - 1) :=
sorry

end NUMINAMATH_GPT_y_eq_fraction_x_l751_75145


namespace NUMINAMATH_GPT_neg_p_eq_exist_l751_75154

theorem neg_p_eq_exist:
  (¬ ∀ a b : ℝ, a^2 + b^2 ≥ 2 * a * b) ↔ ∃ a b : ℝ, a^2 + b^2 < 2 * a * b := by
  sorry

end NUMINAMATH_GPT_neg_p_eq_exist_l751_75154


namespace NUMINAMATH_GPT_problem_statement_l751_75152

noncomputable def f (a x : ℝ) := a * (x ^ 2 + 1) + Real.log x

theorem problem_statement (a m : ℝ) (x : ℝ) 
  (h_a : -4 < a) (h_a' : a < -2) (h_x1 : 1 ≤ x) (h_x2 : x ≤ 3) :
  (m * a - f a x > a ^ 2) ↔ (m ≤ -2) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l751_75152


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l751_75135

noncomputable def a : ℝ := Real.log 4 / Real.log 5
noncomputable def b : ℝ := (Real.log 3 / Real.log 5)^2
noncomputable def c : ℝ := Real.log 5 / Real.log 4

theorem relationship_among_a_b_c : b < a ∧ a < c := by
  sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l751_75135


namespace NUMINAMATH_GPT_find_m_range_l751_75142

noncomputable def ellipse_symmetric_points_range (m : ℝ) : Prop :=
  -((2:ℝ) * Real.sqrt (13:ℝ) / 13) < m ∧ m < ((2:ℝ) * Real.sqrt (13:ℝ) / 13)

theorem find_m_range :
  ∃ m : ℝ, ellipse_symmetric_points_range m :=
sorry

end NUMINAMATH_GPT_find_m_range_l751_75142


namespace NUMINAMATH_GPT_determine_a_l751_75184

noncomputable def imaginary_unit : ℂ := Complex.I

def is_on_y_axis (z : ℂ) : Prop :=
  z.re = 0

theorem determine_a (a : ℝ) : 
  is_on_y_axis (⟨(a - 3 * imaginary_unit.re), -(a - 3 * imaginary_unit.im)⟩ / ⟨(1 - imaginary_unit.re), -(1 - imaginary_unit.im)⟩) → 
  a = -3 :=
sorry

end NUMINAMATH_GPT_determine_a_l751_75184


namespace NUMINAMATH_GPT_find_y_l751_75178

theorem find_y {x y : ℤ} (h1 : x - y = 12) (h2 : x + y = 6) : y = -3 := 
by
  sorry

end NUMINAMATH_GPT_find_y_l751_75178


namespace NUMINAMATH_GPT_students_remaining_after_fourth_stop_l751_75185

variable (n : ℕ)
variable (frac : ℚ)

def initial_students := (64 : ℚ)
def fraction_remaining := (2/3 : ℚ)

theorem students_remaining_after_fourth_stop : 
  let after_first_stop := initial_students * fraction_remaining
  let after_second_stop := after_first_stop * fraction_remaining
  let after_third_stop := after_second_stop * fraction_remaining
  let after_fourth_stop := after_third_stop * fraction_remaining
  after_fourth_stop = (1024 / 81) := 
by 
  sorry

end NUMINAMATH_GPT_students_remaining_after_fourth_stop_l751_75185


namespace NUMINAMATH_GPT_square_distance_from_B_to_center_l751_75160

-- Defining the conditions
structure Circle (α : Type _) :=
(center : α × α)
(radius2 : ℝ)

structure Point (α : Type _) :=
(x : α)
(y : α)

def is_right_angle (a b c : Point ℝ) : Prop :=
(b.x - a.x) * (c.x - b.x) + (b.y - a.y) * (c.y - b.y) = 0

noncomputable def distance2 (p1 p2 : Point ℝ) : ℝ :=
(p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2

theorem square_distance_from_B_to_center :
  ∀ (c : Circle ℝ) (A B C : Point ℝ), 
    c.radius2 = 65 →
    distance2 A B = 49 →
    distance2 B C = 9 →
    is_right_angle A B C →
    distance2 B {x:=0, y:=0} = 80 := 
by
  intros c A B C h_radius h_AB h_BC h_right_angle
  sorry

end NUMINAMATH_GPT_square_distance_from_B_to_center_l751_75160


namespace NUMINAMATH_GPT_domain_of_logarithmic_function_l751_75124

theorem domain_of_logarithmic_function :
  ∀ x : ℝ, 2 - x > 0 ↔ x < 2 := 
by
  intro x
  sorry

end NUMINAMATH_GPT_domain_of_logarithmic_function_l751_75124


namespace NUMINAMATH_GPT_simplify_and_evaluate_expr_l751_75140

theorem simplify_and_evaluate_expr (a b : ℝ) (h1 : a = 1 / 2) (h2 : b = -4) :
  5 * (3 * a ^ 2 * b - a * b ^ 2) - 4 * (-a * b ^ 2 + 3 * a ^ 2 * b) = -11 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expr_l751_75140


namespace NUMINAMATH_GPT_total_expense_in_decade_l751_75119

/-- Definition of yearly expense on car insurance -/
def yearly_expense : ℕ := 2000

/-- Definition of the number of years in a decade -/
def years_in_decade : ℕ := 10

/-- Proof that the total expense in a decade is 20000 dollars -/
theorem total_expense_in_decade : yearly_expense * years_in_decade = 20000 :=
by
  sorry

end NUMINAMATH_GPT_total_expense_in_decade_l751_75119


namespace NUMINAMATH_GPT_lcm_eq_792_l751_75131

-- Define the integers
def a : ℕ := 8
def b : ℕ := 9
def c : ℕ := 11

-- Define their prime factorizations (included for clarity, though not directly necessary)
def a_factorization : a = 2^3 := rfl
def b_factorization : b = 3^2 := rfl
def c_factorization : c = 11 := rfl

-- Define the LCM function
def lcm_abc := Nat.lcm (Nat.lcm a b) c

-- Prove that lcm of a, b, c is 792
theorem lcm_eq_792 : lcm_abc = 792 := 
by
  -- Include the necessary properties of LCM and prime factorizations if necessary
  sorry

end NUMINAMATH_GPT_lcm_eq_792_l751_75131


namespace NUMINAMATH_GPT_distance_from_B_to_center_is_74_l751_75187

noncomputable def circle_radius := 10
noncomputable def B_distance (a b : ℝ) := a^2 + b^2

theorem distance_from_B_to_center_is_74 
  (a b : ℝ)
  (hA : a^2 + (b + 6)^2 = 100)
  (hC : (a + 4)^2 + b^2 = 100) :
  B_distance a b = 74 :=
sorry

end NUMINAMATH_GPT_distance_from_B_to_center_is_74_l751_75187


namespace NUMINAMATH_GPT_power_function_decreasing_m_eq_2_l751_75170

theorem power_function_decreasing_m_eq_2 (x : ℝ) (m : ℝ) (hx : 0 < x) 
  (h_decreasing : ∀ x₁ x₂, 0 < x₁ → 0 < x₂ → x₁ < x₂ → 
                    (m^2 - m - 1) * x₁^(-m+1) > (m^2 - m - 1) * x₂^(-m+1))
  (coeff_positive : m^2 - m - 1 > 0)
  (expo_condition : -m + 1 < 0) : 
  m = 2 :=
by
  sorry

end NUMINAMATH_GPT_power_function_decreasing_m_eq_2_l751_75170


namespace NUMINAMATH_GPT_int_solutions_exist_for_x2_plus_15y2_eq_4n_l751_75164

theorem int_solutions_exist_for_x2_plus_15y2_eq_4n (n : ℕ) (hn : n > 0) : 
  ∃ S : Finset (ℤ × ℤ), S.card ≥ n ∧ ∀ (xy : ℤ × ℤ), xy ∈ S → xy.1^2 + 15 * xy.2^2 = 4^n :=
by
  sorry

end NUMINAMATH_GPT_int_solutions_exist_for_x2_plus_15y2_eq_4n_l751_75164


namespace NUMINAMATH_GPT_men_in_first_group_l751_75174

theorem men_in_first_group (M : ℕ) (h1 : ∀ W, W = M * 30) (h2 : ∀ W, W = 10 * 36) : 
  M = 12 :=
by
  sorry

end NUMINAMATH_GPT_men_in_first_group_l751_75174


namespace NUMINAMATH_GPT_circle_standard_equation_l751_75105

noncomputable def circle_equation (a : ℝ) : Prop :=
  ∃ (x y : ℝ), (x - a)^2 + y^2 = 1

theorem circle_standard_equation : circle_equation 2 := by
  sorry

end NUMINAMATH_GPT_circle_standard_equation_l751_75105


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l751_75133

theorem arithmetic_sequence_sum :
  ∃ a b : ℕ, ∀ d : ℕ,
    d = 5 →
    a = 28 →
    b = 33 →
    a + b = 61 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l751_75133


namespace NUMINAMATH_GPT_boy_present_age_l751_75148

theorem boy_present_age : ∃ x : ℕ, (x + 4 = 2 * (x - 6)) ∧ x = 16 := by
  sorry

end NUMINAMATH_GPT_boy_present_age_l751_75148


namespace NUMINAMATH_GPT_problem_l751_75137

theorem problem (a b : ℕ) (h1 : ∃ k : ℕ, a * b = k * k) (h2 : ∃ m : ℕ, (2 * a + 1) * (2 * b + 1) = m * m) :
  ∃ n : ℕ, n % 2 = 0 ∧ n > 2 ∧ ∃ p : ℕ, (a + n) * (b + n) = p * p :=
by
  sorry

end NUMINAMATH_GPT_problem_l751_75137
