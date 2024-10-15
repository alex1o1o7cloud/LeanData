import Mathlib

namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1162_116271

noncomputable def represents_ellipse (m : ℝ) : Prop :=
  2 < m ∧ m < 6 ∧ m ≠ 4

theorem necessary_but_not_sufficient (m : ℝ) :
  represents_ellipse (m) ↔ (2 < m ∧ m < 6) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1162_116271


namespace NUMINAMATH_GPT_radius_of_circle_nearest_integer_l1162_116255

theorem radius_of_circle_nearest_integer (θ L : ℝ) (hθ : θ = 300) (hL : L = 2000) : 
  abs ((1200 / (Real.pi)) - 382) < 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_radius_of_circle_nearest_integer_l1162_116255


namespace NUMINAMATH_GPT_melanies_plums_l1162_116203

variable (pickedPlums : ℕ)
variable (gavePlums : ℕ)

theorem melanies_plums (h1 : pickedPlums = 7) (h2 : gavePlums = 3) : (pickedPlums - gavePlums) = 4 :=
by
  sorry

end NUMINAMATH_GPT_melanies_plums_l1162_116203


namespace NUMINAMATH_GPT_number_of_adults_l1162_116272

-- Define the constants and conditions of the problem.
def children : ℕ := 52
def total_seats : ℕ := 95
def empty_seats : ℕ := 14

-- Define the number of adults and prove it equals 29 given the conditions.
theorem number_of_adults : total_seats - empty_seats - children = 29 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_adults_l1162_116272


namespace NUMINAMATH_GPT_supplement_of_angle_l1162_116222

theorem supplement_of_angle (complement_of_angle : ℝ) (h1 : complement_of_angle = 30) :
  ∃ (angle supplement_angle : ℝ), angle + complement_of_angle = 90 ∧ angle + supplement_angle = 180 ∧ supplement_angle = 120 :=
by
  sorry

end NUMINAMATH_GPT_supplement_of_angle_l1162_116222


namespace NUMINAMATH_GPT_employee_overtime_hours_l1162_116268

theorem employee_overtime_hours (gross_pay : ℝ) (rate_regular : ℝ) (regular_hours : ℕ) (rate_overtime : ℝ) :
  gross_pay = 622 → rate_regular = 11.25 → regular_hours = 40 → rate_overtime = 16 →
  ∃ (overtime_hours : ℕ), overtime_hours = 10 :=
by
  sorry

end NUMINAMATH_GPT_employee_overtime_hours_l1162_116268


namespace NUMINAMATH_GPT_solve_equation_l1162_116259

theorem solve_equation : ∀ (x : ℝ), -2 * x + 3 - 2 * x + 3 = 3 * x - 6 → x = 12 / 7 :=
by 
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_equation_l1162_116259


namespace NUMINAMATH_GPT_problem_statement_l1162_116217

def number_of_combinations (n k : ℕ) : ℕ := Nat.choose n k

def successful_outcomes : ℕ :=
  (number_of_combinations 3 1) * (number_of_combinations 5 1) * (number_of_combinations 4 5) +
  (number_of_combinations 3 2) * (number_of_combinations 4 5)

def total_outcomes : ℕ := number_of_combinations 12 7

def probability_at_least_75_cents : ℚ :=
  successful_outcomes / total_outcomes

theorem problem_statement : probability_at_least_75_cents = 3 / 22 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1162_116217


namespace NUMINAMATH_GPT_parallel_line_slope_l1162_116288

theorem parallel_line_slope (x y : ℝ) : (∃ (c : ℝ), 3 * x - 6 * y = c) → (1 / 2) = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_parallel_line_slope_l1162_116288


namespace NUMINAMATH_GPT_ratio_of_150_to_10_l1162_116274

theorem ratio_of_150_to_10 : 150 / 10 = 15 := by 
  sorry

end NUMINAMATH_GPT_ratio_of_150_to_10_l1162_116274


namespace NUMINAMATH_GPT_biscuits_per_dog_l1162_116235

-- Define constants for conditions
def total_biscuits : ℕ := 6
def number_of_dogs : ℕ := 2

-- Define the statement to prove
theorem biscuits_per_dog : total_biscuits / number_of_dogs = 3 := by
  -- Calculation here
  sorry

end NUMINAMATH_GPT_biscuits_per_dog_l1162_116235


namespace NUMINAMATH_GPT_totalMarbles_l1162_116219

def originalMarbles : ℕ := 22
def marblesGiven : ℕ := 20

theorem totalMarbles : originalMarbles + marblesGiven = 42 := by
  sorry

end NUMINAMATH_GPT_totalMarbles_l1162_116219


namespace NUMINAMATH_GPT_min_value_l1162_116206

theorem min_value (x : ℝ) (h : x > 1) : x + 4 / (x - 1) ≥ 5 :=
sorry

end NUMINAMATH_GPT_min_value_l1162_116206


namespace NUMINAMATH_GPT_seventh_oblong_number_l1162_116239

/-- An oblong number is the number of dots in a rectangular grid where the number of rows is one more than the number of columns. -/
def is_oblong_number (n : ℕ) (x : ℕ) : Prop :=
  x = n * (n + 1)

/-- The 7th oblong number is 56. -/
theorem seventh_oblong_number : ∃ x, is_oblong_number 7 x ∧ x = 56 :=
by 
  use 56
  unfold is_oblong_number
  constructor
  rfl -- This confirms the computation 7 * 8 = 56
  sorry -- Wrapping up the proof, no further steps needed

end NUMINAMATH_GPT_seventh_oblong_number_l1162_116239


namespace NUMINAMATH_GPT_compound_interest_comparison_l1162_116294

theorem compound_interest_comparison :
  let P := 1000
  let r_annual := 0.03
  let r_monthly := 0.0025
  let t := 5
  (P * (1 + r_monthly)^((12 * t)) > P * (1 + r_annual)^t) :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_comparison_l1162_116294


namespace NUMINAMATH_GPT_find_a_l1162_116230

theorem find_a (x y a : ℝ) (h1 : x = 2) (h2 : y = 1) (h3 : x - a * y = 3) : a = -1 :=
by
  -- Solution steps go here
  sorry

end NUMINAMATH_GPT_find_a_l1162_116230


namespace NUMINAMATH_GPT_equilateral_triangle_properties_l1162_116228

noncomputable def equilateral_triangle_perimeter (a : ℝ) : ℝ :=
3 * a

noncomputable def equilateral_triangle_bisector_length (a : ℝ) : ℝ :=
(a * Real.sqrt 3) / 2

theorem equilateral_triangle_properties (a : ℝ) (h : a = 10) :
  equilateral_triangle_perimeter a = 30 ∧
  equilateral_triangle_bisector_length a = 5 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_properties_l1162_116228


namespace NUMINAMATH_GPT_tan_theta_value_l1162_116282

open Real

theorem tan_theta_value (θ : ℝ) (h : sin (θ / 2) - 2 * cos (θ / 2) = 0) : tan θ = -4 / 3 :=
sorry

end NUMINAMATH_GPT_tan_theta_value_l1162_116282


namespace NUMINAMATH_GPT_jane_oldest_babysat_age_l1162_116227

-- Given conditions
def jane_babysitting_has_constraints (jane_current_age jane_stop_babysitting_age jane_start_babysitting_age : ℕ) : Prop :=
  jane_current_age - jane_stop_babysitting_age = 10 ∧
  jane_stop_babysitting_age - jane_start_babysitting_age = 2

-- Helper definition for prime number constraint
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m < n, m > 1 → ¬ (n % m = 0)

-- Main goal: the current age of the oldest person Jane could have babysat is 19
theorem jane_oldest_babysat_age
  (jane_current_age jane_stop_babysitting_age jane_start_babysitting_age : ℕ)
  (H_constraints : jane_babysitting_has_constraints jane_current_age jane_stop_babysitting_age jane_start_babysitting_age) :
  ∃ (child_age : ℕ), child_age = 19 ∧ is_prime child_age ∧
  (child_age = (jane_stop_babysitting_age / 2 + 10) ∨ child_age = (jane_stop_babysitting_age / 2 + 9)) :=
sorry  -- Proof to be filled in.

end NUMINAMATH_GPT_jane_oldest_babysat_age_l1162_116227


namespace NUMINAMATH_GPT_additional_emails_per_day_l1162_116263

theorem additional_emails_per_day
  (emails_per_day_before : ℕ)
  (half_days : ℕ)
  (total_days : ℕ)
  (total_emails : ℕ)
  (emails_received_first_half : ℕ := emails_per_day_before * half_days)
  (emails_received_second_half : ℕ := total_emails - emails_received_first_half)
  (emails_per_day_after : ℕ := emails_received_second_half / half_days) :
  emails_per_day_before = 20 → half_days = 15 → total_days = 30 → total_emails = 675 → (emails_per_day_after - emails_per_day_before = 5) :=
by
  intros
  sorry

end NUMINAMATH_GPT_additional_emails_per_day_l1162_116263


namespace NUMINAMATH_GPT_coin_difference_l1162_116202

-- Definitions based on problem conditions
def denominations : List ℕ := [5, 10, 25, 50]
def amount_owed : ℕ := 55

-- Proof statement
theorem coin_difference :
  let min_coins := 1 + 1 -- one 50-cent coin and one 5-cent coin
  let max_coins := 11 -- eleven 5-cent coins
  max_coins - min_coins = 9 :=
by
  -- Proof details skipped
  sorry

end NUMINAMATH_GPT_coin_difference_l1162_116202


namespace NUMINAMATH_GPT_real_solution_x_condition_l1162_116286

theorem real_solution_x_condition (x : ℝ) :
  (∃ y : ℝ, 9 * y^2 + 6 * x * y + 2 * x + 1 = 0) ↔ (x < 2 - Real.sqrt 6 ∨ x > 2 + Real.sqrt 6) :=
by
  sorry

end NUMINAMATH_GPT_real_solution_x_condition_l1162_116286


namespace NUMINAMATH_GPT_expected_turns_formula_l1162_116267

noncomputable def expected_turns (n : ℕ) : ℝ :=
  n +  1 / 2 - (n - 1 / 2) * (1 / Real.sqrt (Real.pi * (n - 1)))

theorem expected_turns_formula (n : ℕ) (h : n ≥ 1) :
  expected_turns n = n + 1 / 2 - (n - 1 / 2) * (1 / Real.sqrt (Real.pi * (n - 1))) :=
by
  sorry

end NUMINAMATH_GPT_expected_turns_formula_l1162_116267


namespace NUMINAMATH_GPT_silk_per_dress_l1162_116266

theorem silk_per_dress (initial_silk : ℕ) (friends : ℕ) (silk_per_friend : ℕ) (total_dresses : ℕ)
  (h1 : initial_silk = 600)
  (h2 : friends = 5)
  (h3 : silk_per_friend = 20)
  (h4 : total_dresses = 100)
  (remaining_silk := initial_silk - friends * silk_per_friend) :
  remaining_silk / total_dresses = 5 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_silk_per_dress_l1162_116266


namespace NUMINAMATH_GPT_sum_first_9_terms_l1162_116220

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (d : ℝ)
variable (a_1 a_2 a_3 a_4 a_5 a_6 : ℝ)

-- Conditions
axiom h1 : a 1 + a 5 = 10
axiom h2 : a 2 + a 6 = 14

-- Calculations
axiom h3 : a 3 = 5
axiom h4 : a 4 = 7
axiom h5 : d = 2
axiom h6 : a 5 = 9

-- The sum of the first 9 terms
axiom h7 : S 9 = 9 * a 5

theorem sum_first_9_terms : S 9 = 81 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_first_9_terms_l1162_116220


namespace NUMINAMATH_GPT_probability_divisible_by_8_l1162_116277

-- Define the problem conditions
def is_8_sided_die (n : ℕ) : Prop := n = 6
def roll_dice (m : ℕ) : Prop := m = 8

-- Define the main proof statement
theorem probability_divisible_by_8 (n m : ℕ) (hn : is_8_sided_die n) (hm : roll_dice m) :  
  (35 : ℚ) / 36 = 
  (1 - ((1/2) ^ m + 28 * ((1/n) ^ 2 * ((1/2) ^ 6))) : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_probability_divisible_by_8_l1162_116277


namespace NUMINAMATH_GPT_venus_hall_meal_cost_l1162_116279

theorem venus_hall_meal_cost (V : ℕ) :
  let caesars_total_cost := 800 + 30 * 60;
  let venus_hall_total_cost := 500 + V * 60;
  caesars_total_cost = venus_hall_total_cost → V = 35 :=
by
  let caesars_total_cost := 800 + 30 * 60
  let venus_hall_total_cost := 500 + V * 60
  intros h
  sorry

end NUMINAMATH_GPT_venus_hall_meal_cost_l1162_116279


namespace NUMINAMATH_GPT_total_pennies_l1162_116252

-- Definitions based on conditions
def initial_pennies_per_compartment := 2
def additional_pennies_per_compartment := 6
def compartments := 12

-- Mathematically equivalent proof statement
theorem total_pennies (initial_pennies_per_compartment : Nat) 
                      (additional_pennies_per_compartment : Nat)
                      (compartments : Nat) : 
                      initial_pennies_per_compartment = 2 → 
                      additional_pennies_per_compartment = 6 → 
                      compartments = 12 → 
                      compartments * (initial_pennies_per_compartment + additional_pennies_per_compartment) = 96 := 
by
  intros
  sorry

end NUMINAMATH_GPT_total_pennies_l1162_116252


namespace NUMINAMATH_GPT_percentage_increase_correct_l1162_116238

variable {R1 E1 P1 R2 E2 P2 R3 E3 P3 : ℝ}

-- Conditions
axiom H1 : P1 = R1 - E1
axiom H2 : R2 = 1.20 * R1
axiom H3 : E2 = 1.10 * E1
axiom H4 : P2 = R2 - E2
axiom H5 : P2 = 1.15 * P1
axiom H6 : R3 = 1.25 * R2
axiom H7 : E3 = 1.20 * E2
axiom H8 : P3 = R3 - E3
axiom H9 : P3 = 1.35 * P2

theorem percentage_increase_correct :
  ((P3 - P1) / P1) * 100 = 55.25 :=
by sorry

end NUMINAMATH_GPT_percentage_increase_correct_l1162_116238


namespace NUMINAMATH_GPT_speed_of_boat_is_15_l1162_116244

noncomputable def speed_of_boat_in_still_water (x : ℝ) : Prop :=
  ∃ (t : ℝ), t = 1 / 5 ∧ (x + 3) * t = 3.6 ∧ x = 15

theorem speed_of_boat_is_15 (x : ℝ) (t : ℝ) (rate_of_current : ℝ) (distance_downstream : ℝ) :
  rate_of_current = 3 →
  distance_downstream = 3.6 →
  t = 1 / 5 →
  (x + rate_of_current) * t = distance_downstream →
  x = 15 :=
by
  intros h1 h2 h3 h4
  -- proof goes here
  sorry

end NUMINAMATH_GPT_speed_of_boat_is_15_l1162_116244


namespace NUMINAMATH_GPT_fraction_simplification_l1162_116284

theorem fraction_simplification (a : ℝ) (h1 : a > 1) (h2 : a ≠ 2 / Real.sqrt 3) : 
  (a^3 - 3 * a^2 + 4 + (a^2 - 4) * Real.sqrt (a^2 - 1)) / 
  (a^3 + 3 * a^2 - 4 + (a^2 - 4) * Real.sqrt (a^2 - 1)) = 
  ((a - 2) * Real.sqrt (a + 1)) / ((a + 2) * Real.sqrt (a - 1)) :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l1162_116284


namespace NUMINAMATH_GPT_geometric_sequence_b_value_l1162_116278

theorem geometric_sequence_b_value (b : ℝ) (r : ℝ) (h1 : 210 * r = b) (h2 : b * r = 35 / 36) (hb : b > 0) : 
  b = Real.sqrt (7350 / 36) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_b_value_l1162_116278


namespace NUMINAMATH_GPT_find_symbols_l1162_116248

theorem find_symbols (x y otimes oplus : ℝ) 
  (h1 : x + otimes * y = 3) 
  (h2 : 3 * x - otimes * y = 1) 
  (h3 : x = oplus) 
  (h4 : y = 1) : 
  otimes = 2 ∧ oplus = 1 := 
by
  sorry

end NUMINAMATH_GPT_find_symbols_l1162_116248


namespace NUMINAMATH_GPT_intersection_count_l1162_116226

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem intersection_count : ∃! (x1 x2 : ℝ), 
  x1 > 0 ∧ x2 > 0 ∧ f x1 = g x1 ∧ f x2 = g x2 ∧ x1 ≠ x2 :=
sorry

end NUMINAMATH_GPT_intersection_count_l1162_116226


namespace NUMINAMATH_GPT_kevin_hop_distance_l1162_116280

theorem kevin_hop_distance :
  (1/4) + (3/16) + (9/64) + (27/256) + (81/1024) + (243/4096) = 3367 / 4096 := 
by
  sorry 

end NUMINAMATH_GPT_kevin_hop_distance_l1162_116280


namespace NUMINAMATH_GPT_intersection_point_correct_l1162_116207

structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Line :=
(p1 : Point3D) (p2 : Point3D)

structure Plane :=
(trace : Line) (point : Point3D)

noncomputable def intersection_point (l : Line) (β : Plane) : Point3D := sorry

theorem intersection_point_correct (l : Line) (β : Plane) (P : Point3D) :
  let res := intersection_point l β
  res = P :=
sorry

end NUMINAMATH_GPT_intersection_point_correct_l1162_116207


namespace NUMINAMATH_GPT_solve_chestnut_problem_l1162_116243

def chestnut_problem : Prop :=
  ∃ (P M L : ℕ), (M = 2 * P) ∧ (L = P + 2) ∧ (P + M + L = 26) ∧ (M = 12)

theorem solve_chestnut_problem : chestnut_problem :=
by 
  sorry

end NUMINAMATH_GPT_solve_chestnut_problem_l1162_116243


namespace NUMINAMATH_GPT_find_numbers_l1162_116204

theorem find_numbers (x : ℚ) (a : ℚ) (b : ℚ) (h₁ : a = 8 * x) (h₂ : b = x^2 - 1) :
  (a * b + a = (2 * x)^3) ∧ (a * b + b = (2 * x - 1)^3) → 
  x = 14 / 13 ∧ a = 112 / 13 ∧ b = 27 / 169 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_numbers_l1162_116204


namespace NUMINAMATH_GPT_product_of_intersection_coordinates_l1162_116290

noncomputable def circle1 := {P : ℝ×ℝ | (P.1^2 - 4*P.1 + P.2^2 - 8*P.2 + 20) = 0}
noncomputable def circle2 := {P : ℝ×ℝ | (P.1^2 - 6*P.1 + P.2^2 - 8*P.2 + 25) = 0}

theorem product_of_intersection_coordinates :
  ∀ P ∈ circle1 ∩ circle2, P = (2, 4) → (P.1 * P.2) = 8 :=
by
  sorry

end NUMINAMATH_GPT_product_of_intersection_coordinates_l1162_116290


namespace NUMINAMATH_GPT_billy_free_time_l1162_116293

theorem billy_free_time
  (play_time_percentage : ℝ := 0.75)
  (read_pages_per_hour : ℝ := 60)
  (book_pages : ℝ := 80)
  (number_of_books : ℝ := 3)
  (read_percentage : ℝ := 1 - play_time_percentage)
  (total_pages : ℝ := number_of_books * book_pages)
  (read_time_hours : ℝ := total_pages / read_pages_per_hour)
  (free_time_hours : ℝ := read_time_hours / read_percentage) :
  free_time_hours = 16 := 
sorry

end NUMINAMATH_GPT_billy_free_time_l1162_116293


namespace NUMINAMATH_GPT_correct_product_l1162_116297

theorem correct_product : 
  (0.0063 * 3.85 = 0.024255) :=
sorry

end NUMINAMATH_GPT_correct_product_l1162_116297


namespace NUMINAMATH_GPT_maria_bottles_count_l1162_116242

-- Definitions from the given conditions
def b_initial : ℕ := 23
def d : ℕ := 12
def g : ℕ := 5
def b : ℕ := 65

-- Definition of the question based on conditions
def b_final : ℕ := b_initial - d - g + b

-- The statement to prove the correctness of the answer
theorem maria_bottles_count : b_final = 71 := by
  -- We skip the proof for this statement
  sorry

end NUMINAMATH_GPT_maria_bottles_count_l1162_116242


namespace NUMINAMATH_GPT_value_of_expression_l1162_116231

theorem value_of_expression (x y : ℝ) (h1 : x = Real.sqrt 5 + Real.sqrt 3) (h2 : y = Real.sqrt 5 - Real.sqrt 3) : x^2 + x * y + y^2 = 18 :=
by sorry

end NUMINAMATH_GPT_value_of_expression_l1162_116231


namespace NUMINAMATH_GPT_task_D_cannot_be_sampled_l1162_116234

def task_A := "Measuring the range of a batch of shells"
def task_B := "Determining the content of a certain microorganism in ocean waters"
def task_C := "Calculating the difficulty of each question on the math test after the college entrance examination"
def task_D := "Checking the height and weight of all sophomore students in a school"

def sampling_method (description: String) : Prop :=
  description = task_A ∨ description = task_B ∨ description = task_C

theorem task_D_cannot_be_sampled : ¬ sampling_method task_D :=
sorry

end NUMINAMATH_GPT_task_D_cannot_be_sampled_l1162_116234


namespace NUMINAMATH_GPT_rate_in_still_water_l1162_116269

theorem rate_in_still_water (with_stream_speed against_stream_speed : ℕ) 
  (h₁ : with_stream_speed = 16) 
  (h₂ : against_stream_speed = 12) : 
  (with_stream_speed + against_stream_speed) / 2 = 14 := 
by
  sorry

end NUMINAMATH_GPT_rate_in_still_water_l1162_116269


namespace NUMINAMATH_GPT_problem1_problem2_l1162_116218

noncomputable def f (x : ℝ) (a : ℝ) := (1 / 2) * x^2 + 2 * a * x
noncomputable def g (x : ℝ) (a : ℝ) (b : ℝ) := 3 * a^2 * Real.log x + b

theorem problem1 (a b x₀ : ℝ) (h : x₀ = a):
  a > 0 →
  (1 / 2) * x₀^2 + 2 * a * x₀ = 3 * a^2 * Real.log x₀ + b →
  x₀ + 2 * a = 3 * a^2 / x₀ →
  b = (5 * a^2 / 2) - 3 * a^2 * Real.log a := sorry

theorem problem2 (a b : ℝ):
  -2 ≤ b ∧ b ≤ 2 →
  ∀ x > 0, x < 4 →
  ∀ x, x - b + 3 * a^2 / x ≥ 0 →
  a ≥ Real.sqrt 3 / 3 ∨ a ≤ -Real.sqrt 3 / 3 := sorry

end NUMINAMATH_GPT_problem1_problem2_l1162_116218


namespace NUMINAMATH_GPT_min_valid_n_l1162_116233

theorem min_valid_n (n : ℕ) (h_pos : 0 < n) (h_int : ∃ m : ℕ, m * m = 51 + n) : n = 13 :=
  sorry

end NUMINAMATH_GPT_min_valid_n_l1162_116233


namespace NUMINAMATH_GPT_solve_for_x_l1162_116237

theorem solve_for_x :
  ∃ x : ℝ, (1 / (x + 5) + 1 / (x + 2) + 1 / (2 * x) = 1 / x) ∧ (1 / (x + 5) + 1 / (x + 2) = 1 / (x + 3)) ∧ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1162_116237


namespace NUMINAMATH_GPT_inscribed_circle_radius_of_DEF_l1162_116200

theorem inscribed_circle_radius_of_DEF (DE DF EF : ℝ) (r : ℝ)
  (hDE : DE = 8) (hDF : DF = 8) (hEF : EF = 10) :
  r = 5 * Real.sqrt 39 / 13 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_of_DEF_l1162_116200


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l1162_116275

variable (a b : ℝ)

theorem necessary_and_sufficient_condition:
  (ab + 1 ≠ a + b) ↔ (a ≠ 1 ∧ b ≠ 1) :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l1162_116275


namespace NUMINAMATH_GPT_negation_of_proposition_l1162_116299

theorem negation_of_proposition :
  ¬ (∀ x : ℝ, x^2 + 3*x + 2 < 0) ↔ ∃ x : ℝ, x^2 + 3*x + 2 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1162_116299


namespace NUMINAMATH_GPT_find_number_that_satisfies_congruences_l1162_116276

theorem find_number_that_satisfies_congruences :
  ∃ m : ℕ, 
  (m % 13 = 12) ∧ 
  (m % 11 = 10) ∧ 
  (m % 7 = 6) ∧ 
  (m % 5 = 4) ∧ 
  (m % 3 = 2) ∧ 
  m = 15014 :=
by
  sorry

end NUMINAMATH_GPT_find_number_that_satisfies_congruences_l1162_116276


namespace NUMINAMATH_GPT_largest_vs_smallest_circles_l1162_116216

variable (M : Type) [MetricSpace M] [MeasurableSpace M]

def non_overlapping_circles (M : Type) [MetricSpace M] [MeasurableSpace M] : ℕ := sorry

def covering_circles (M : Type) [MetricSpace M] [MeasurableSpace M] : ℕ := sorry

theorem largest_vs_smallest_circles (M : Type) [MetricSpace M] [MeasurableSpace M] :
  non_overlapping_circles M ≥ covering_circles M :=
sorry

end NUMINAMATH_GPT_largest_vs_smallest_circles_l1162_116216


namespace NUMINAMATH_GPT_square_binomial_constant_l1162_116258

theorem square_binomial_constant (y : ℝ) : ∃ b : ℝ, (y^2 + 12*y + 50 = (y + 6)^2 + b) ∧ b = 14 := 
by
  sorry

end NUMINAMATH_GPT_square_binomial_constant_l1162_116258


namespace NUMINAMATH_GPT_find_first_term_and_common_difference_l1162_116257

variable (a d : ℕ)
variable (S_even S_odd S_total : ℕ)

-- Conditions
axiom condition1 : S_total = 354
axiom condition2 : S_even = 192
axiom condition3 : S_odd = 162
axiom condition4 : 12*(2*a + 11*d) = 2*S_total
axiom condition5 : 6*(a + 6*d) = S_even
axiom condition6 : 6*(a + 5*d) = S_odd

-- Theorem to prove
theorem find_first_term_and_common_difference (a d S_even S_odd S_total : ℕ)
  (h1 : S_total = 354)
  (h2 : S_even = 192)
  (h3 : S_odd = 162)
  (h4 : 12*(2*a + 11*d) = 2*S_total)
  (h5 : 6*(a + 6*d) = S_even)
  (h6 : 6*(a + 5*d) = S_odd) : a = 2 ∧ d = 5 := by
  sorry

end NUMINAMATH_GPT_find_first_term_and_common_difference_l1162_116257


namespace NUMINAMATH_GPT_range_of_omega_l1162_116240

theorem range_of_omega (ω : ℝ) (h_pos : ω > 0) (h_three_high_points : (9 * π / 2) ≤ ω + π / 4 ∧ ω + π / 4 < 6 * π + π / 2) : 
           (17 * π / 4) ≤ ω ∧ ω < (25 * π / 4) :=
  sorry

end NUMINAMATH_GPT_range_of_omega_l1162_116240


namespace NUMINAMATH_GPT_product_of_consecutive_nat_is_divisible_by_2_l1162_116260

theorem product_of_consecutive_nat_is_divisible_by_2 (n : ℕ) : 2 ∣ n * (n + 1) :=
sorry

end NUMINAMATH_GPT_product_of_consecutive_nat_is_divisible_by_2_l1162_116260


namespace NUMINAMATH_GPT_inequality_holds_l1162_116209

theorem inequality_holds (a b : ℝ) (h1 : a > 1) (h2 : 1 > b) (h3 : b > -1) : a > b^2 := 
sorry

end NUMINAMATH_GPT_inequality_holds_l1162_116209


namespace NUMINAMATH_GPT_FI_squared_l1162_116201

-- Definitions for the given conditions
-- Note: Further geometric setup and formalization might be necessary to carry 
-- out the complete proof in Lean, but the setup will follow these basic definitions.

-- Let ABCD be a square
def ABCD_square (A B C D : ℝ × ℝ) : Prop :=
  -- conditions for ABCD being a square (to be properly defined based on coordinates and properties)
  sorry

-- Triangle AEH is an equilateral triangle with side length sqrt(3)
def equilateral_AEH (A E H : ℝ × ℝ) (s : ℝ) : Prop :=
  dist A E = s ∧ dist E H = s ∧ dist H A = s 

-- Points E and H lie on AB and DA respectively
-- Points F and G lie on BC and CD respectively
-- Points I and J lie on EH with FI ⊥ EH and GJ ⊥ EH
-- Areas of triangles and quadrilaterals
def geometric_conditions (A B C D E F G H I J : ℝ × ℝ) : Prop :=
  sorry

-- Final statement to prove
theorem FI_squared (A B C D E F G H I J : ℝ × ℝ) (s : ℝ) 
  (h_square: ABCD_square A B C D) 
  (h_equilateral: equilateral_AEH A E H (Real.sqrt 3))
  (h_geo: geometric_conditions A B C D E F G H I J) :
  dist F I ^ 2 = 4 / 3 :=
sorry

end NUMINAMATH_GPT_FI_squared_l1162_116201


namespace NUMINAMATH_GPT_return_trip_time_l1162_116262

variable (d p w_1 w_2 : ℝ)
variable (t t' : ℝ)
variable (h1 : d / (p - w_1) = 120)
variable (h2 : d / (p + w_2) = t - 10)
variable (h3 : t = d / p)

theorem return_trip_time :
  t' = 72 :=
by
  sorry

end NUMINAMATH_GPT_return_trip_time_l1162_116262


namespace NUMINAMATH_GPT_Ivan_defeats_Koschei_l1162_116225

-- Definitions of the springs and conditions based on the problem
section

variable (S: ℕ → Prop)  -- S(n) means the water from spring n
variable (deadly: ℕ → Prop)  -- deadly(n) if water from nth spring is deadly

-- Conditions
axiom accessibility (n: ℕ): (1 ≤ n ∧ n ≤ 9 → ∀ i: ℕ, S i)
axiom koschei_access: S 10
axiom lethality (n: ℕ): (S n → deadly n)
axiom neutralize (i j: ℕ): (1 ≤ i ∧ i < j ∧ j ≤ 9 → ∃ k: ℕ, S k ∧ k > j → ¬deadly i)

-- Statement to prove
theorem Ivan_defeats_Koschei:
  ∃ i: ℕ, (1 ≤ i ∧ i ≤ 9) → (S 10 → ¬deadly i) ∧ (S 0 ∧ (S 10 → deadly 0)) :=
sorry

end

end NUMINAMATH_GPT_Ivan_defeats_Koschei_l1162_116225


namespace NUMINAMATH_GPT_A_leaves_after_one_day_l1162_116249

-- Define and state all the conditions
def A_work_rate := 1 / 21
def B_work_rate := 1 / 28
def C_work_rate := 1 / 35
def total_work := 1
def B_time_after_A_leave := 21
def C_intermittent_working_cycle := 3 / 1 -- C works 1 out of every 3 days

-- The statement that needs to be proved
theorem A_leaves_after_one_day :
  ∃ x : ℕ, x = 1 ∧
  (A_work_rate * x + B_work_rate * x + (C_work_rate * (x / C_intermittent_working_cycle)) + (B_work_rate * B_time_after_A_leave) + (C_work_rate * (B_time_after_A_leave / C_intermittent_working_cycle)) = total_work) :=
sorry

end NUMINAMATH_GPT_A_leaves_after_one_day_l1162_116249


namespace NUMINAMATH_GPT_other_acute_angle_right_triangle_l1162_116285

theorem other_acute_angle_right_triangle (A : ℝ) (B : ℝ) (C : ℝ) (h₁ : A + B = 90) (h₂ : B = 54) : A = 36 :=
by
  sorry

end NUMINAMATH_GPT_other_acute_angle_right_triangle_l1162_116285


namespace NUMINAMATH_GPT_max_value_of_squares_l1162_116291

theorem max_value_of_squares (a b c d : ℝ) 
  (h1 : a + b = 18) 
  (h2 : ab + c + d = 91) 
  (h3 : ad + bc = 187) 
  (h4 : cd = 105) : 
  a^2 + b^2 + c^2 + d^2 ≤ 107 :=
sorry

end NUMINAMATH_GPT_max_value_of_squares_l1162_116291


namespace NUMINAMATH_GPT_decreasing_exponential_range_l1162_116210

theorem decreasing_exponential_range {a : ℝ} (h : ∀ x y : ℝ, x < y → (a + 1)^x > (a + 1)^y) : -1 < a ∧ a < 0 :=
sorry

end NUMINAMATH_GPT_decreasing_exponential_range_l1162_116210


namespace NUMINAMATH_GPT_merchant_installed_zucchini_l1162_116287

theorem merchant_installed_zucchini (Z : ℕ) : 
  (15 + Z + 8) / 2 = 18 → Z = 13 :=
by
 sorry

end NUMINAMATH_GPT_merchant_installed_zucchini_l1162_116287


namespace NUMINAMATH_GPT_total_pure_acid_in_mixture_l1162_116250

-- Definitions of the conditions
def solution1_volume : ℝ := 8
def solution1_concentration : ℝ := 0.20
def solution2_volume : ℝ := 5
def solution2_concentration : ℝ := 0.35

-- Proof statement
theorem total_pure_acid_in_mixture :
  solution1_volume * solution1_concentration + solution2_volume * solution2_concentration = 3.35 := by
  sorry

end NUMINAMATH_GPT_total_pure_acid_in_mixture_l1162_116250


namespace NUMINAMATH_GPT_find_theta_perpendicular_l1162_116265

theorem find_theta_perpendicular (θ : ℝ) (hθ : 0 < θ ∧ θ < π)
  (a b : ℝ × ℝ) (ha : a = (Real.sin θ, 1)) (hb : b = (2 * Real.cos θ, -1))
  (hperp : a.fst * b.fst + a.snd * b.snd = 0) : θ = π / 4 :=
by
  -- Proof would be written here
  sorry

end NUMINAMATH_GPT_find_theta_perpendicular_l1162_116265


namespace NUMINAMATH_GPT_good_goods_not_cheap_l1162_116221

-- Define the propositions "good goods" and "not cheap"
variables (p q : Prop)

-- State that "good goods are not cheap" is expressed by the implication p → q
theorem good_goods_not_cheap : p → q → (p → q) ↔ (p ∧ q → p ∧ q) := by
  sorry

end NUMINAMATH_GPT_good_goods_not_cheap_l1162_116221


namespace NUMINAMATH_GPT_vacuum_cleaner_cost_l1162_116298

-- Variables
variables (V : ℝ)

-- Conditions
def cost_of_dishwasher := 450
def coupon := 75
def total_spent := 625

-- The main theorem to prove
theorem vacuum_cleaner_cost : V + cost_of_dishwasher - coupon = total_spent → V = 250 :=
by
  -- Proof logic goes here
  sorry

end NUMINAMATH_GPT_vacuum_cleaner_cost_l1162_116298


namespace NUMINAMATH_GPT_gcd_fact_8_10_l1162_116273

-- Definitions based on the conditions in a)
def fact (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * fact (n - 1)

-- Question and conditions translated to a proof problem in Lean
theorem gcd_fact_8_10 : Nat.gcd (fact 8) (fact 10) = 40320 := by
  sorry

end NUMINAMATH_GPT_gcd_fact_8_10_l1162_116273


namespace NUMINAMATH_GPT_total_volume_of_5_cubes_is_135_l1162_116253

-- Define the edge length of a single cube
def edge_length : ℕ := 3

-- Define the volume of a single cube
def volume_single_cube (s : ℕ) : ℕ := s^3

-- State the total volume for a given number of cubes
def total_volume (n : ℕ) (s : ℕ) : ℕ := n * volume_single_cube s

-- Prove that for 5 cubes with an edge length of 3 meters, the total volume is 135 cubic meters
theorem total_volume_of_5_cubes_is_135 :
    total_volume 5 edge_length = 135 :=
by
  sorry

end NUMINAMATH_GPT_total_volume_of_5_cubes_is_135_l1162_116253


namespace NUMINAMATH_GPT_initial_roses_l1162_116224

theorem initial_roses (R : ℕ) (h : R + 16 = 23) : R = 7 :=
sorry

end NUMINAMATH_GPT_initial_roses_l1162_116224


namespace NUMINAMATH_GPT_parallel_lines_have_equal_slopes_l1162_116205

theorem parallel_lines_have_equal_slopes (m : ℝ) :
  (∀ x y : ℝ, x + 2 * y - 1 = 0 → mx - y = 0) → m = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_have_equal_slopes_l1162_116205


namespace NUMINAMATH_GPT_ratio_proof_l1162_116264

variables (x y m n : ℝ)

def ratio_equation1 (x y m n : ℝ) : Prop :=
  (5 * x + 7 * y) / (3 * x + 2 * y) = m / n

def target_equation (x y m n : ℝ) : Prop :=
  (13 * x + 16 * y) / (2 * x + 5 * y) = (2 * m + n) / (m - n)

theorem ratio_proof (x y m n : ℝ) (h: ratio_equation1 x y m n) :
  target_equation x y m n :=
by
  sorry

end NUMINAMATH_GPT_ratio_proof_l1162_116264


namespace NUMINAMATH_GPT_cafeteria_sales_comparison_l1162_116247

theorem cafeteria_sales_comparison
  (S : ℝ) -- initial sales
  (a : ℝ) -- monthly increment for Cafeteria A
  (p : ℝ) -- monthly percentage increment for Cafeteria B
  (h1 : S > 0) -- initial sales are positive
  (h2 : a > 0) -- constant increment for Cafeteria A is positive
  (h3 : p > 0) -- constant percentage increment for Cafeteria B is positive
  (h4 : S + 8 * a = S * (1 + p) ^ 8) -- sales are equal in September 2013
  (h5 : S = S) -- sales are equal in January 2013 (trivially true)
  : S + 4 * a > S * (1 + p) ^ 4 := 
sorry

end NUMINAMATH_GPT_cafeteria_sales_comparison_l1162_116247


namespace NUMINAMATH_GPT_infinite_series_k3_over_3k_l1162_116214

theorem infinite_series_k3_over_3k :
  ∑' k : ℕ, (k^3 : ℝ) / 3^k = 165 / 16 := 
sorry

end NUMINAMATH_GPT_infinite_series_k3_over_3k_l1162_116214


namespace NUMINAMATH_GPT_initial_pipes_l1162_116229

variables (x : ℕ)

-- Defining the conditions
def one_pipe_time := x -- time for 1 pipe to fill the tank in hours
def eight_pipes_time := 1 / 4 -- 15 minutes = 1/4 hour

-- Proving the number of pipes
theorem initial_pipes (h1 : eight_pipes_time * 8 = one_pipe_time) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_initial_pipes_l1162_116229


namespace NUMINAMATH_GPT_correct_calculation_l1162_116241

variable {a : ℝ} (ha : a ≠ 0)

theorem correct_calculation (a : ℝ) (ha : a ≠ 0) : (a^2 * a^3 = a^5) :=
by sorry

end NUMINAMATH_GPT_correct_calculation_l1162_116241


namespace NUMINAMATH_GPT_talia_father_age_l1162_116236

theorem talia_father_age 
  (t tf tm ta : ℕ) 
  (h1 : t + 7 = 20)
  (h2 : tm = 3 * t)
  (h3 : tf + 3 = tm)
  (h4 : ta = (tm - t) / 2)
  (h5 : ta + 2 = tf + 5) : 
  tf = 36 :=
by
  sorry

end NUMINAMATH_GPT_talia_father_age_l1162_116236


namespace NUMINAMATH_GPT_cistern_emptying_time_l1162_116256

noncomputable def cistern_time_without_tap (tap_rate : ℕ) (empty_time_with_tap : ℕ) (cistern_volume : ℕ) : ℕ := 
  let tap_total := tap_rate * empty_time_with_tap
  let leaked_volume := cistern_volume - tap_total
  let leak_rate := leaked_volume / empty_time_with_tap
  cistern_volume / leak_rate

theorem cistern_emptying_time :
  cistern_time_without_tap 4 24 480 = 30 := 
by
  unfold cistern_time_without_tap
  norm_num

end NUMINAMATH_GPT_cistern_emptying_time_l1162_116256


namespace NUMINAMATH_GPT_systematic_sampling_fourth_group_number_l1162_116292

theorem systematic_sampling_fourth_group_number (n : ℕ) (step_size : ℕ) (first_number : ℕ) : 
  n = 4 → step_size = 6 → first_number = 4 → (first_number + step_size * 3) = 22 :=
by
  intros h_n h_step_size h_first_number
  sorry

end NUMINAMATH_GPT_systematic_sampling_fourth_group_number_l1162_116292


namespace NUMINAMATH_GPT_exists_pythagorean_number_in_range_l1162_116261

def is_pythagorean_area (a : ℕ) : Prop :=
  ∃ (x y z : ℕ), x^2 + y^2 = z^2 ∧ a = (x * y) / 2

theorem exists_pythagorean_number_in_range (n : ℕ) (hn : n > 12) : 
  ∃ (m : ℕ), is_pythagorean_area m ∧ n < m ∧ m < 2 * n :=
sorry

end NUMINAMATH_GPT_exists_pythagorean_number_in_range_l1162_116261


namespace NUMINAMATH_GPT_quadratic_root_in_interval_l1162_116270

theorem quadratic_root_in_interval 
  (a b c : ℝ) 
  (h : 2 * a + 3 * b + 6 * c = 0) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_root_in_interval_l1162_116270


namespace NUMINAMATH_GPT_percentage_of_l1162_116296

theorem percentage_of (part whole : ℕ) (h_part : part = 120) (h_whole : whole = 80) : 
  ((part : ℚ) / (whole : ℚ)) * 100 = 150 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_l1162_116296


namespace NUMINAMATH_GPT_cats_left_l1162_116215

theorem cats_left (siamese house sold : ℕ) (h1 : siamese = 12) (h2 : house = 20) (h3 : sold = 20) :  
  (siamese + house) - sold = 12 := 
by
  sorry

end NUMINAMATH_GPT_cats_left_l1162_116215


namespace NUMINAMATH_GPT_inequality_f_c_f_a_f_b_l1162_116246

-- Define the function f and the conditions
def f : ℝ → ℝ := sorry

noncomputable def a : ℝ := Real.log (1 / Real.pi)
noncomputable def b : ℝ := (Real.log Real.pi) ^ 2
noncomputable def c : ℝ := Real.log (Real.sqrt Real.pi)

-- Theorem statement
theorem inequality_f_c_f_a_f_b :
  (∀ x : ℝ, f x = f (-x)) →
  (∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f x1 > f x2) →
  f c > f a ∧ f a > f b :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_inequality_f_c_f_a_f_b_l1162_116246


namespace NUMINAMATH_GPT_number_of_full_rows_in_first_field_l1162_116281

-- Define the conditions
def total_corn_cobs : ℕ := 116
def rows_in_second_field : ℕ := 16
def cobs_per_row : ℕ := 4
def cobs_in_second_field : ℕ := rows_in_second_field * cobs_per_row
def cobs_in_first_field : ℕ := total_corn_cobs - cobs_in_second_field

-- Define the theorem to be proven
theorem number_of_full_rows_in_first_field : 
  cobs_in_first_field / cobs_per_row = 13 :=
by
  sorry

end NUMINAMATH_GPT_number_of_full_rows_in_first_field_l1162_116281


namespace NUMINAMATH_GPT_fraction_without_cable_or_vcr_l1162_116223

theorem fraction_without_cable_or_vcr (T : ℕ) (h1 : ℚ) (h2 : ℚ) (h3 : ℚ) 
  (h1 : h1 = 1 / 5 * T) 
  (h2 : h2 = 1 / 10 * T) 
  (h3 : h3 = 1 / 3 * (1 / 5 * T)) 
: (T - (1 / 5 * T + 1 / 10 * T - 1 / 3 * (1 / 5 * T))) / T = 23 / 30 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_without_cable_or_vcr_l1162_116223


namespace NUMINAMATH_GPT_ballsInBoxes_theorem_l1162_116245

noncomputable def ballsInBoxes : ℕ :=
  let distributions := [(5,0,0), (4,1,0), (3,2,0), (3,1,1), (2,2,1)]
  distributions.foldl (λ acc (x : ℕ × ℕ × ℕ) =>
    acc + match x with
      | (5,0,0) => 3
      | (4,1,0) => 6
      | (3,2,0) => 6
      | (3,1,1) => 6
      | (2,2,1) => 3
      | _ => 0
  ) 0

theorem ballsInBoxes_theorem : ballsInBoxes = 24 :=
by
  unfold ballsInBoxes
  rfl

end NUMINAMATH_GPT_ballsInBoxes_theorem_l1162_116245


namespace NUMINAMATH_GPT_compare_ab_1_to_a_b_two_pow_b_eq_one_div_b_sign_of_expression_l1162_116211

-- Part 1
theorem compare_ab_1_to_a_b {a b : ℝ} (h1 : a^b * b^a + Real.log b / Real.log a = 0) (ha : a > 0) (hb : b > 0) : ab + 1 < a + b := sorry

-- Part 2
theorem two_pow_b_eq_one_div_b {b : ℝ} (h : 2^b * b^2 + Real.log b / Real.log 2 = 0) : 2^b = 1 / b := sorry

-- Part 3
theorem sign_of_expression {b : ℝ} (h : 2^b * b^2 + Real.log b / Real.log 2 = 0) : (2 * b + 1 - Real.sqrt 5) * (3 * b - 2) < 0 := sorry

end NUMINAMATH_GPT_compare_ab_1_to_a_b_two_pow_b_eq_one_div_b_sign_of_expression_l1162_116211


namespace NUMINAMATH_GPT_average_age_of_remaining_people_l1162_116283

theorem average_age_of_remaining_people:
  ∀ (ages : List ℕ), 
  (List.length ages = 8) →
  (List.sum ages = 224) →
  (24 ∈ ages) →
  ((List.sum ages - 24) / 7 = 28 + 4/7) := 
by
  intro ages
  intro h_len
  intro h_sum
  intro h_24
  sorry

end NUMINAMATH_GPT_average_age_of_remaining_people_l1162_116283


namespace NUMINAMATH_GPT_store_A_more_cost_effective_100_cost_expressions_for_x_most_cost_effective_plan_l1162_116295

-- Definitions and conditions
def cost_per_soccer : ℕ := 200
def cost_per_basketball : ℕ := 80
def discount_A_soccer (n : ℕ) : ℕ := n * cost_per_soccer
def discount_A_basketball (n : ℕ) : ℕ := if n > 100 then (n - 100) * cost_per_basketball else 0
def discount_B_soccer (n : ℕ) : ℕ := n * cost_per_soccer * 8 / 10
def discount_B_basketball (n : ℕ) : ℕ := n * cost_per_basketball * 8 / 10

-- For x = 100
def total_cost_A_100 : ℕ := discount_A_soccer 100 + discount_A_basketball 100
def total_cost_B_100 : ℕ := discount_B_soccer 100 + discount_B_basketball 100

-- Prove that for x = 100, Store A is more cost-effective
theorem store_A_more_cost_effective_100 : total_cost_A_100 < total_cost_B_100 :=
by sorry

-- For x > 100, express costs in terms of x
def total_cost_A (x : ℕ) : ℕ := 80 * x + 12000
def total_cost_B (x : ℕ) : ℕ := 64 * x + 16000

-- Prove the expressions for costs
theorem cost_expressions_for_x (x : ℕ) (h : x > 100) : 
  total_cost_A x = 80 * x + 12000 ∧ total_cost_B x = 64 * x + 16000 :=
by sorry

-- For x = 300, most cost-effective plan
def combined_A_100_B_200 : ℕ := (discount_A_soccer 100 + cost_per_soccer * 100) + (200 * cost_per_basketball * 8 / 10)
def only_A_300 : ℕ := discount_A_soccer 100 + (300 - 100) * cost_per_basketball
def only_B_300 : ℕ := discount_B_soccer 100 + 300 * cost_per_basketball * 8 / 10

-- Prove the most cost-effective plan for x = 300
theorem most_cost_effective_plan : combined_A_100_B_200 < only_B_300 ∧ combined_A_100_B_200 < only_A_300 :=
by sorry

end NUMINAMATH_GPT_store_A_more_cost_effective_100_cost_expressions_for_x_most_cost_effective_plan_l1162_116295


namespace NUMINAMATH_GPT_find_f_at_8_l1162_116212

theorem find_f_at_8 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (3 * x - 1) = x^2 + 2 * x + 4) :
  f 8 = 19 :=
sorry

end NUMINAMATH_GPT_find_f_at_8_l1162_116212


namespace NUMINAMATH_GPT_jonessa_total_pay_l1162_116289

theorem jonessa_total_pay (total_pay : ℝ) (take_home_pay : ℝ) (h1 : take_home_pay = 450) (h2 : 0.90 * total_pay = take_home_pay) : total_pay = 500 :=
by
  sorry

end NUMINAMATH_GPT_jonessa_total_pay_l1162_116289


namespace NUMINAMATH_GPT_inequality_proof_l1162_116232

theorem inequality_proof (x y z : ℝ) :
  (x^2 + 2 * y^2 + 2 * z^2) / (x^2 + y * z) +
  (y^2 + 2 * z^2 + 2 * x^2) / (y^2 + z * x) +
  (z^2 + 2 * x^2 + 2 * y^2) / (z^2 + x * y) > 6 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1162_116232


namespace NUMINAMATH_GPT_graded_worksheets_before_l1162_116251

-- Definitions based on conditions
def initial_worksheets : ℕ := 34
def additional_worksheets : ℕ := 36
def total_worksheets : ℕ := 63

-- Equivalent proof problem statement
theorem graded_worksheets_before (x : ℕ) (h₁ : initial_worksheets - x + additional_worksheets = total_worksheets) : x = 7 :=
by sorry

end NUMINAMATH_GPT_graded_worksheets_before_l1162_116251


namespace NUMINAMATH_GPT_cylinder_height_in_sphere_l1162_116208

noncomputable def height_of_cylinder (r R : ℝ) : ℝ :=
  2 * Real.sqrt (R ^ 2 - r ^ 2)

theorem cylinder_height_in_sphere :
  height_of_cylinder 3 6 = 6 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_height_in_sphere_l1162_116208


namespace NUMINAMATH_GPT_real_solutions_eq_31_l1162_116213

noncomputable def number_of_real_solutions : ℕ :=
  let zero := 0
  let fifty := 50
  let neg_fifty := -50
  let num_intervals := 8
  let num_solutions_per_interval := 2
  let total_solutions := num_intervals * num_solutions_per_interval * 2 - 1
  total_solutions

theorem real_solutions_eq_31 : number_of_real_solutions = 31 := by
  sorry

end NUMINAMATH_GPT_real_solutions_eq_31_l1162_116213


namespace NUMINAMATH_GPT_find_d_l1162_116254

-- Defining the basic points and their corresponding conditions
structure Point (α : Type) :=
(x : α) (y : α) (z : α)

def a : Point ℝ := ⟨1, 0, 1⟩
def b : Point ℝ := ⟨0, 1, 0⟩
def c : Point ℝ := ⟨0, 1, 1⟩

-- introducing k as a positive integer
variables (k : ℤ) (hk : k > 0 ∧ k ≠ 6 ∧ k ≠ 1)

def d (k : ℤ) : Point ℝ := ⟨k*d, k*d, -d⟩ where d := -(k / (k-1))

-- The proof statement
theorem find_d (k : ℤ) (hk : k > 0 ∧ k ≠ 6 ∧ k ≠ 1) :
∃ d: ℝ, d = - (k / (k-1)) :=
sorry

end NUMINAMATH_GPT_find_d_l1162_116254
