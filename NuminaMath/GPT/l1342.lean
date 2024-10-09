import Mathlib

namespace tangent_value_prism_QABC_l1342_134264

-- Assuming R is the radius of the sphere and considering the given conditions
variables {R x : ℝ} (P Q A B C M H : Type)

-- Given condition: Angle between lateral face and base of prism P-ABC is 45 degrees
def angle_PABC : ℝ := 45
-- Required to prove: tan(angle between lateral face and base of prism Q-ABC) = 4
def tangent_QABC : ℝ := 4

theorem tangent_value_prism_QABC
  (h1 : angle_PABC = 45)
  (h2 : 5 * x - 2 * R = 0) -- Derived condition from the solution
  (h3 : x = 2 * R / 5) -- x, the distance calculation
: tangent_QABC = 4 := by
  sorry

end tangent_value_prism_QABC_l1342_134264


namespace expression_simplification_l1342_134261

theorem expression_simplification : (2^2020 + 2^2018) / (2^2020 - 2^2018) = 5 / 3 := 
by 
    sorry

end expression_simplification_l1342_134261


namespace black_area_fraction_after_four_changes_l1342_134266

/-- 
Problem: Prove that after four changes, the fractional part of the original black area 
remaining black in an equilateral triangle is 81/256, given that each change splits the 
triangle into 4 smaller congruent equilateral triangles, and one of those turns white.
-/

theorem black_area_fraction_after_four_changes :
  (3 / 4) ^ 4 = 81 / 256 := sorry

end black_area_fraction_after_four_changes_l1342_134266


namespace discount_price_l1342_134262

theorem discount_price (a : ℝ) (original_price : ℝ) (sold_price : ℝ) :
  original_price = 200 ∧ sold_price = 148 → (original_price * (1 - a/100) * (1 - a/100) = sold_price) :=
by
  sorry

end discount_price_l1342_134262


namespace supply_lasts_for_8_months_l1342_134253

-- Define the conditions
def pills_per_supply : ℕ := 120
def days_per_pill : ℕ := 2
def days_per_month : ℕ := 30

-- Define the function to calculate the duration in days
def supply_duration_in_days (pills : ℕ) (days_per_pill : ℕ) : ℕ :=
  pills * days_per_pill

-- Define the function to convert days to months
def days_to_months (days : ℕ) (days_per_month : ℕ) : ℕ :=
  days / days_per_month

-- Main statement to prove
theorem supply_lasts_for_8_months :
  days_to_months (supply_duration_in_days pills_per_supply days_per_pill) days_per_month = 8 :=
by
  sorry

end supply_lasts_for_8_months_l1342_134253


namespace log_eq_solution_l1342_134265

open Real

theorem log_eq_solution (x : ℝ) (h : x > 0) : log x + log (x + 1) = 2 ↔ x = (-1 + sqrt 401) / 2 :=
by
  sorry

end log_eq_solution_l1342_134265


namespace simplify_expression_l1342_134279

theorem simplify_expression (θ : Real) (h : θ ∈ Set.Icc (5 * Real.pi / 4) (3 * Real.pi / 2)) :
  Real.sqrt (1 - Real.sin (2 * θ)) - Real.sqrt (1 + Real.sin (2 * θ)) = -2 * Real.cos θ :=
sorry

end simplify_expression_l1342_134279


namespace total_votes_cast_l1342_134250

-- Problem statement and conditions
variable (V : ℝ) (candidateVotes : ℝ) (rivalVotes : ℝ)
variable (h1 : candidateVotes = 0.35 * V)
variable (h2 : rivalVotes = candidateVotes + 1350)

-- Target to prove
theorem total_votes_cast : V = 4500 := by
  -- pseudo code proof would be filled here in real Lean environment
  sorry

end total_votes_cast_l1342_134250


namespace probability_not_passing_l1342_134201

noncomputable def probability_of_passing : ℚ := 4 / 7

theorem probability_not_passing (h : probability_of_passing = 4 / 7) : 1 - probability_of_passing = 3 / 7 :=
by
  sorry

end probability_not_passing_l1342_134201


namespace geom_seq_S6_l1342_134286

theorem geom_seq_S6 :
  ∃ (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ),
  (q = 2) →
  (S 3 = 7) →
  (∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) →
  S 6 = 63 :=
sorry

end geom_seq_S6_l1342_134286


namespace percentage_boys_from_school_A_is_20_l1342_134284

-- Definitions and conditions based on the problem
def total_boys : ℕ := 200
def non_science_boys_from_A : ℕ := 28
def science_ratio : ℝ := 0.30
def non_science_ratio : ℝ := 1 - science_ratio

-- To prove: The percentage of the total boys that are from school A is 20%
theorem percentage_boys_from_school_A_is_20 :
  ∃ (x : ℝ), x = 20 ∧ 
  (non_science_ratio * (x / 100 * total_boys) = non_science_boys_from_A) := 
sorry

end percentage_boys_from_school_A_is_20_l1342_134284


namespace part_a_part_b_part_c_l1342_134205

variable (N : ℕ) (r : Fin N → Fin N → ℝ)

-- Part (a)
theorem part_a (h : ∀ (s : Finset (Fin N)), s.card = 5 → (exists pts : s → ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j)) :
  ∃ pts : Fin N → ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j :=
sorry

-- Part (b)
theorem part_b (h : ∀ (s : Finset (Fin N)), s.card = 4 → (exists pts : s → ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j)) :
  ¬ (∃ pts : Fin N → ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j) :=
sorry

-- Part (c)
theorem part_c (h : ∀ (s : Finset (Fin N)), s.card = 6 → (exists pts : s → ℝ × ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j)) :
  ∃ (pts : Fin N → ℝ × ℝ × ℝ), ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j :=
sorry

end part_a_part_b_part_c_l1342_134205


namespace minimum_value_of_f_l1342_134256

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 - 4 * x + 5) / (2 * x - 4)

theorem minimum_value_of_f (x : ℝ) (h : x ≥ 5 / 2) : ∃ y ≥ (5 / 2), f y = 1 := by
  sorry

end minimum_value_of_f_l1342_134256


namespace initial_percentage_of_water_l1342_134276

theorem initial_percentage_of_water (C V final_volume : ℝ) (P : ℝ) 
  (hC : C = 80)
  (hV : V = 36)
  (h_final_volume : final_volume = (3/4) * C)
  (h_initial_equation: (P / 100) * C + V = final_volume) : 
  P = 30 :=
by
  sorry

end initial_percentage_of_water_l1342_134276


namespace length_of_one_pencil_l1342_134220

theorem length_of_one_pencil (l : ℕ) (h1 : 2 * l = 24) : l = 12 :=
by {
  sorry
}

end length_of_one_pencil_l1342_134220


namespace min_number_of_gennadys_l1342_134207

theorem min_number_of_gennadys (a b v g : ℕ) (h_a : a = 45) (h_b : b = 122) (h_v : v = 27)
    (h_needed_g : g = 49) :
    (b - 1) - (a + v) = g :=
by
  -- We include sorry because we are focusing on the statement, not the proof itself.
  sorry

end min_number_of_gennadys_l1342_134207


namespace domain_of_sqrt_sum_l1342_134258

theorem domain_of_sqrt_sum (x : ℝ) (h1 : 3 + x ≥ 0) (h2 : 1 - x ≥ 0) : -3 ≤ x ∧ x ≤ 1 := by
  sorry

end domain_of_sqrt_sum_l1342_134258


namespace stuart_initial_marbles_is_56_l1342_134283

-- Define the initial conditions
def betty_initial_marbles : ℕ := 60
def percentage_given_to_stuart : ℚ := 40 / 100
def stuart_marbles_after_receiving : ℕ := 80

-- Define the calculation of how many marbles Betty gave to Stuart
def marbles_given_to_stuart := (percentage_given_to_stuart * betty_initial_marbles)

-- Define the target: Stuart's initial number of marbles
def stuart_initial_marbles := stuart_marbles_after_receiving - marbles_given_to_stuart

-- Main theorem stating the problem
theorem stuart_initial_marbles_is_56 : stuart_initial_marbles = 56 :=
by 
  sorry

end stuart_initial_marbles_is_56_l1342_134283


namespace paperback_copies_sold_l1342_134243

theorem paperback_copies_sold
  (H : ℕ) (P : ℕ)
  (h1 : H = 36000)
  (h2 : P = 9 * H)
  (h3 : H + P = 440000) :
  P = 360000 := by
  sorry

end paperback_copies_sold_l1342_134243


namespace modulo_multiplication_l1342_134252

theorem modulo_multiplication (m : ℕ) (h : 0 ≤ m ∧ m < 50) :
  152 * 936 % 50 = 22 :=
by
  sorry

end modulo_multiplication_l1342_134252


namespace committee_membership_l1342_134263

theorem committee_membership (n : ℕ) (h1 : 2 * n = 6) (h2 : (n - 1 : ℚ) / 5 = 0.4) : n = 3 := 
sorry

end committee_membership_l1342_134263


namespace inequality_solution_l1342_134206

theorem inequality_solution (x : ℝ) :
  (6 * (x ^ 3 - 8) * (Real.sqrt (x ^ 2 + 6 * x + 9)) / ((x ^ 2 + 2 * x + 4) * (x ^ 2 + x - 6)) ≥ x - 2) ↔
  (x ∈ Set.Iic (-4) ∪ Set.Ioo (-3) 2 ∪ Set.Ioo 2 8) := sorry

end inequality_solution_l1342_134206


namespace fraction_simplification_l1342_134237

theorem fraction_simplification : 
  (320 / 18) * (9 / 144) * (4 / 5) = 1 / 2 :=
by sorry

end fraction_simplification_l1342_134237


namespace scientific_notation_of_0_0000025_l1342_134277

theorem scientific_notation_of_0_0000025 :
  0.0000025 = 2.5 * 10^(-6) :=
by
  sorry

end scientific_notation_of_0_0000025_l1342_134277


namespace remainder_1234567_div_145_l1342_134291

theorem remainder_1234567_div_145 : 1234567 % 145 = 67 := by
  sorry

end remainder_1234567_div_145_l1342_134291


namespace initially_calculated_average_weight_l1342_134231

theorem initially_calculated_average_weight 
  (A : ℚ)
  (h1 : ∀ sum_weight_corr : ℚ, sum_weight_corr = 20 * 58.65)
  (h2 : ∀ misread_weight_corr : ℚ, misread_weight_corr = 56)
  (h3 : ∀ correct_weight_corr : ℚ, correct_weight_corr = 61)
  (h4 : (20 * A + (correct_weight_corr - misread_weight_corr)) = 20 * 58.65) :
  A = 58.4 := 
sorry

end initially_calculated_average_weight_l1342_134231


namespace rectangular_box_proof_l1342_134299

noncomputable def rectangular_box_surface_area
  (a b c : ℝ)
  (h1 : 4 * (a + b + c) = 140)
  (h2 : a^2 + b^2 + c^2 = 21^2) : ℝ :=
2 * (a * b + b * c + c * a)

theorem rectangular_box_proof
  (a b c : ℝ)
  (h1 : 4 * (a + b + c) = 140)
  (h2 : a^2 + b^2 + c^2 = 21^2) :
  rectangular_box_surface_area a b c h1 h2 = 784 :=
by
  sorry

end rectangular_box_proof_l1342_134299


namespace constant_S13_l1342_134257

-- Defining the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) := ∀ n : ℕ, a (n + 1) = a n + d

-- Defining the sum of the first n terms
def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (List.range n).map a |> List.sum

-- Defining the given conditions as hypotheses
variable {a : ℕ → ℤ} {d : ℤ}
variable (h_arith : arithmetic_sequence a d)
variable (constant_sum : a 2 + a 4 + a 15 = k)

-- Goal to prove: S_13 is a constant
theorem constant_S13 (k : ℤ) :
  sum_first_n_terms a 13 = k :=
  sorry

end constant_S13_l1342_134257


namespace interval_of_monotonic_decrease_range_of_k_l1342_134255
open Real

noncomputable def f (x : ℝ) : ℝ := 
  let m := (sqrt 3 * sin (x / 4), 1)
  let n := (cos (x / 4), cos (x / 2))
  m.1 * n.1 + m.2 * n.2 -- vector dot product

-- Prove the interval of monotonic decrease for f(x)
theorem interval_of_monotonic_decrease (k : ℤ) : 
  4 * k * π + 2 * π / 3 ≤ x ∧ x ≤ 4 * k * π + 8 * π / 3 → f x = sin (x / 2 + π / 6) + 1 / 2 :=
sorry

-- Prove the range of k such that the zero condition is satisfied for g(x) - k
theorem range_of_k (k : ℝ) :
  0 ≤ k ∧ k ≤ 3 / 2 → ∃ x ∈ [0, 7 * π / 3], (sin (x / 2 - π / 6) + 1 / 2) - k = 0 :=
sorry

end interval_of_monotonic_decrease_range_of_k_l1342_134255


namespace decompose_96_l1342_134293

theorem decompose_96 (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 96) (h4 : a^2 + b^2 = 208) : 
  (a = 8 ∧ b = 12) ∨ (a = 12 ∧ b = 8) :=
sorry

end decompose_96_l1342_134293


namespace n_greater_than_7_l1342_134223

theorem n_greater_than_7 (m n : ℕ) (hmn : m > n) (h : ∃k:ℕ, 22220038^m - 22220038^n = 10^8 * k) : n > 7 :=
sorry

end n_greater_than_7_l1342_134223


namespace quarters_to_dimes_difference_l1342_134288

variable (p : ℝ)

theorem quarters_to_dimes_difference :
  let Susan_quarters := 7 * p + 3
  let George_quarters := 2 * p + 9
  let difference_quarters := Susan_quarters - George_quarters
  let difference_dimes := 2.5 * difference_quarters
  difference_dimes = 12.5 * p - 15 :=
by
  let Susan_quarters := 7 * p + 3
  let George_quarters := 2 * p + 9
  let difference_quarters := Susan_quarters - George_quarters
  let difference_dimes := 2.5 * difference_quarters
  sorry

end quarters_to_dimes_difference_l1342_134288


namespace solution_set_of_fraction_inequality_l1342_134235

theorem solution_set_of_fraction_inequality
  (a b : ℝ) (h₀ : ∀ x : ℝ, x > 1 → ax - b > 0) :
  {x : ℝ | (ax + b) / (x - 2) > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 2} :=
by
  sorry

end solution_set_of_fraction_inequality_l1342_134235


namespace dad_steps_l1342_134225

theorem dad_steps (dad_steps_ratio: ℕ) (masha_steps_ratio: ℕ) (masha_steps: ℕ)
  (masha_and_yasha_steps: ℕ) (total_steps: ℕ)
  (h1: dad_steps_ratio * 3 = masha_steps_ratio * 5)
  (h2: masha_steps * 3 = masha_and_yasha_steps * 5)
  (h3: masha_and_yasha_steps = total_steps)
  (h4: total_steps = 400) :
  dad_steps_ratio * 30 = 90 :=
by
  sorry

end dad_steps_l1342_134225


namespace john_trip_time_l1342_134238

theorem john_trip_time (x : ℝ) (h : x + 2 * x + 2 * x = 10) : x = 2 :=
by
  sorry

end john_trip_time_l1342_134238


namespace weng_hourly_rate_l1342_134202

theorem weng_hourly_rate (minutes_worked : ℝ) (earnings : ℝ) (fraction_of_hour : ℝ) 
  (conversion_rate : ℝ) (hourly_rate : ℝ) : 
  minutes_worked = 50 → earnings = 10 → 
  fraction_of_hour = minutes_worked / conversion_rate → 
  conversion_rate = 60 → 
  hourly_rate = earnings / fraction_of_hour → 
  hourly_rate = 12 := by
    sorry

end weng_hourly_rate_l1342_134202


namespace find_unknown_towel_rate_l1342_134297

theorem find_unknown_towel_rate 
    (cost_known1 : ℕ := 300)
    (cost_known2 : ℕ := 750)
    (total_towels : ℕ := 10)
    (average_price : ℕ := 150)
    (total_cost : ℕ := total_towels * average_price) :
  let total_cost_known := cost_known1 + cost_known2
  let cost_unknown := 2 * x
  300 + 750 + 2 * x = total_cost → x = 225 :=
by
  sorry

end find_unknown_towel_rate_l1342_134297


namespace largest_square_in_right_triangle_largest_rectangle_in_right_triangle_l1342_134210

theorem largest_square_in_right_triangle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ s, s = (a * b) / (a + b) := 
sorry

theorem largest_rectangle_in_right_triangle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ x y, x = a / 2 ∧ y = b / 2 :=
sorry

end largest_square_in_right_triangle_largest_rectangle_in_right_triangle_l1342_134210


namespace participants_count_l1342_134241

theorem participants_count (x y : ℕ) 
    (h1 : y = x + 41)
    (h2 : y = 3 * x - 35) : 
    x = 38 ∧ y = 79 :=
by
  sorry

end participants_count_l1342_134241


namespace relationship_between_a_and_b_l1342_134248

-- Define the objects and their relationships
noncomputable def α_parallel_β : Prop := sorry
noncomputable def a_parallel_α : Prop := sorry
noncomputable def b_perpendicular_β : Prop := sorry

-- Define the relationship we want to prove
noncomputable def a_perpendicular_b : Prop := sorry

-- The statement we want to prove
theorem relationship_between_a_and_b (h1 : α_parallel_β) (h2 : a_parallel_α) (h3 : b_perpendicular_β) : a_perpendicular_b :=
sorry

end relationship_between_a_and_b_l1342_134248


namespace woody_saving_weeks_l1342_134224

variable (cost_needed current_savings weekly_allowance : ℕ)

theorem woody_saving_weeks (h₁ : cost_needed = 282)
                           (h₂ : current_savings = 42)
                           (h₃ : weekly_allowance = 24) :
  (cost_needed - current_savings) / weekly_allowance = 10 := by
  sorry

end woody_saving_weeks_l1342_134224


namespace minimum_employees_needed_l1342_134275

-- Conditions
def water_monitors : ℕ := 95
def air_monitors : ℕ := 80
def soil_monitors : ℕ := 45
def water_and_air : ℕ := 30
def air_and_soil : ℕ := 20
def water_and_soil : ℕ := 15
def all_three : ℕ := 10

-- Theorems/Goals
theorem minimum_employees_needed 
  (water : ℕ := water_monitors)
  (air : ℕ := air_monitors)
  (soil : ℕ := soil_monitors)
  (water_air : ℕ := water_and_air)
  (air_soil : ℕ := air_and_soil)
  (water_soil : ℕ := water_and_soil)
  (all_3 : ℕ := all_three) :
  water + air + soil - water_air - air_soil - water_soil + all_3 = 165 :=
by
  sorry

end minimum_employees_needed_l1342_134275


namespace tabs_in_all_browsers_l1342_134229

-- Definitions based on conditions
def windows_per_browser := 3
def tabs_per_window := 10
def number_of_browsers := 2

-- Total tabs calculation
def total_tabs := number_of_browsers * (windows_per_browser * tabs_per_window)

-- Proving the total number of tabs is 60
theorem tabs_in_all_browsers : total_tabs = 60 := by
  sorry

end tabs_in_all_browsers_l1342_134229


namespace old_conveyor_time_l1342_134233

theorem old_conveyor_time (x : ℝ) : 
  (1 / x) + (1 / 15) = 1 / 8.75 → 
  x = 21 := 
by 
  intro h 
  sorry

end old_conveyor_time_l1342_134233


namespace impossible_return_l1342_134282

def Point := (ℝ × ℝ)

-- Conditions
def is_valid_point (p: Point) : Prop :=
  let (a, b) := p
  ∃ a_int b_int : ℤ, (a = a_int + b_int * Real.sqrt 2 ∧ b = a_int + b_int * Real.sqrt 2)

def valid_movement (p q: Point) : Prop :=
  let (x1, y1) := p
  let (x2, y2) := q
  abs x2 > abs x1 ∧ abs y2 > abs y1 

-- Theorem statement
theorem impossible_return (start: Point) (h: start = (1, Real.sqrt 2)) 
  (valid_start: is_valid_point start) :
  ∀ (p: Point), (is_valid_point p ∧ valid_movement start p) → p ≠ start :=
sorry

end impossible_return_l1342_134282


namespace cos_A_eq_a_eq_l1342_134290

-- Defining the problem conditions:
variables {A B C a b c : ℝ}
variable (sin_eq : Real.sin (B + C) = 3 * Real.sin (A / 2) ^ 2)
variable (area_eq : 1 / 2 * b * c * Real.sin A = 6)
variable (sum_eq : b + c = 8)
variable (bc_prod_eq : b * c = 13)
variable (cos_rule : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)

-- Proving the statements:
theorem cos_A_eq : Real.cos A = 5 / 13 :=
sorry

theorem a_eq : a = 3 * Real.sqrt 2 :=
sorry

end cos_A_eq_a_eq_l1342_134290


namespace operation_B_is_not_algorithm_l1342_134285

-- Define what constitutes an algorithm.
def is_algorithm (desc : String) : Prop :=
  desc = "clear and finite steps to solve a certain type of problem"

-- Define given operations.
def operation_A : String := "Calculating the area of a circle given its radius"
def operation_B : String := "Calculating the possibility of reaching 24 by randomly drawing 4 playing cards"
def operation_C : String := "Finding the equation of a line given two points in the coordinate plane"
def operation_D : String := "Operations of addition, subtraction, multiplication, and division"

-- Define expected property of an algorithm.
def is_algorithm_A : Prop := is_algorithm "procedural, explicit, and finite steps lead to the result"
def is_algorithm_B : Prop := is_algorithm "cannot describe precise steps"
def is_algorithm_C : Prop := is_algorithm "procedural, explicit, and finite steps lead to the result"
def is_algorithm_D : Prop := is_algorithm "procedural, explicit, and finite steps lead to the result"

theorem operation_B_is_not_algorithm :
  ¬ (is_algorithm operation_B) :=
by
   -- Change this line to the theorem proof.
   sorry

end operation_B_is_not_algorithm_l1342_134285


namespace correct_statement_B_l1342_134217

-- Definitions as per the conditions
noncomputable def total_students : ℕ := 6700
noncomputable def selected_students : ℕ := 300

-- Definitions as per the question
def is_population (n : ℕ) : Prop := n = 6700
def is_sample (m n : ℕ) : Prop := m = 300 ∧ n = 6700
def is_individual (m n : ℕ) : Prop := m < n
def is_census (m n : ℕ) : Prop := m = n

-- The statement that needs to be proved
theorem correct_statement_B : 
  is_sample selected_students total_students :=
by
  -- Proof steps would go here
  sorry

end correct_statement_B_l1342_134217


namespace find_a_l1342_134278

theorem find_a (a : ℝ) (h : Nat.choose 5 2 * (-a)^3 = 10) : a = -1 :=
by
  sorry

end find_a_l1342_134278


namespace colin_speed_l1342_134239

variable (B T Br C D : ℝ)

-- Given conditions
axiom cond1 : C = 6 * Br
axiom cond2 : Br = (1/3) * T^2
axiom cond3 : T = 2 * B
axiom cond4 : D = (1/4) * C
axiom cond5 : B = 1

-- Prove Colin's speed C is 8 mph
theorem colin_speed :
  C = 8 :=
by
  sorry

end colin_speed_l1342_134239


namespace options_implication_l1342_134280

theorem options_implication (a b : ℝ) :
  ((b > 0 ∧ a < 0) ∨ (a < 0 ∧ b < 0 ∧ a > b) ∨ (a > 0 ∧ b > 0 ∧ a > b)) → (1 / a < 1 / b) :=
by sorry

end options_implication_l1342_134280


namespace algebraic_expression_value_l1342_134228

theorem algebraic_expression_value (a b : ℝ) (h : 4 * a - 2 * b + 2 = 0) :
  2024 + 2 * a - b = 2023 :=
by
  sorry

end algebraic_expression_value_l1342_134228


namespace sum_x1_x2_l1342_134259

open ProbabilityTheory

variable {Ω : Type*} {X : Ω → ℝ}
variable (p1 p2 : ℝ) (x1 x2 : ℝ)
variable (h1 : 2/3 * x1 + 1/3 * x2 = 4/9)
variable (h2 : 2/3 * (x1 - 4/9)^2 + 1/3 * (x2 - 4/9)^2 = 2)
variable (h3 : x1 < x2)

theorem sum_x1_x2 : x1 + x2 = 17/9 :=
by
  sorry

end sum_x1_x2_l1342_134259


namespace equal_student_distribution_l1342_134289

theorem equal_student_distribution
  (students_bus1_initial : ℕ)
  (students_bus2_initial : ℕ)
  (students_to_move : ℕ)
  (students_bus1_final : ℕ)
  (students_bus2_final : ℕ)
  (total_students : ℕ) :
  students_bus1_initial = 57 →
  students_bus2_initial = 31 →
  total_students = students_bus1_initial + students_bus2_initial →
  students_to_move = 13 →
  students_bus1_final = students_bus1_initial - students_to_move →
  students_bus2_final = students_bus2_initial + students_to_move →
  students_bus1_final = 44 ∧ students_bus2_final = 44 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end equal_student_distribution_l1342_134289


namespace simplified_expression_l1342_134236

theorem simplified_expression :
  ( (81 / 16) ^ (3 / 4) - (-1) ^ 0 ) = 19 / 8 := 
by 
  -- It is a placeholder for the actual proof.
  sorry

end simplified_expression_l1342_134236


namespace range_of_a_l1342_134240

def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + (a - 1)*x - 1 < 0
def r (a : ℝ) : Prop := -1 ≤ a ∧ a ≤ 1 ∨ a > 3

theorem range_of_a (a : ℝ) (h : (p a ∨ q a) ∧ ¬(p a ∧ q a)) : r a := 
by sorry

end range_of_a_l1342_134240


namespace lcm_of_denominators_l1342_134209

theorem lcm_of_denominators : Nat.lcm (List.foldr Nat.lcm 1 [2, 3, 4, 5, 6, 7]) = 420 :=
by 
  sorry

end lcm_of_denominators_l1342_134209


namespace positive_number_property_l1342_134272

theorem positive_number_property (x : ℝ) (h : (100 - x) / 100 * x = 16) :
  x = 40 ∨ x = 60 :=
sorry

end positive_number_property_l1342_134272


namespace max_statements_true_l1342_134203

theorem max_statements_true : ∃ x : ℝ, 
  (0 < x^2 ∧ x^2 < 1 ∨ x^2 > 1) ∧ 
  (-1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1) ∧ 
  (0 < (x - x^3) ∧ (x - x^3) < 1) :=
  sorry

end max_statements_true_l1342_134203


namespace percentage_women_red_and_men_dark_l1342_134232

-- Define the conditions as variables
variables (w_fair_hair w_dark_hair w_red_hair m_fair_hair m_dark_hair m_red_hair : ℝ)

-- Define the percentage of women with red hair and men with dark hair
def women_red_men_dark (w_red_hair m_dark_hair : ℝ) : ℝ := w_red_hair + m_dark_hair

-- Define the main theorem to be proven
theorem percentage_women_red_and_men_dark 
  (hw_fair_hair : w_fair_hair = 30)
  (hw_dark_hair : w_dark_hair = 28)
  (hw_red_hair : w_red_hair = 12)
  (hm_fair_hair : m_fair_hair = 20)
  (hm_dark_hair : m_dark_hair = 35)
  (hm_red_hair : m_red_hair = 5) :
  women_red_men_dark w_red_hair m_dark_hair = 47 := 
sorry

end percentage_women_red_and_men_dark_l1342_134232


namespace hyperbola_asymptotes_l1342_134219

theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 / 16 - y^2 / 9 = 1) → (y = 3/4 * x ∨ y = -3/4 * x) :=
by
  sorry

end hyperbola_asymptotes_l1342_134219


namespace max_slope_avoiding_lattice_points_l1342_134246

theorem max_slope_avoiding_lattice_points :
  ∃ a : ℝ, (1 < a ∧ ∀ m : ℝ, (1 < m ∧ m < a) → (∀ x : ℤ, (10 < x ∧ x ≤ 200) → ∃ k : ℝ, y = m * x + 5 ∧ (m * x + 5 ≠ k))) ∧ a = 101 / 100 :=
sorry

end max_slope_avoiding_lattice_points_l1342_134246


namespace trade_ratio_blue_per_red_l1342_134221

-- Define the problem conditions
def initial_total_marbles : ℕ := 10
def blue_percentage : ℕ := 40
def kept_red_marbles : ℕ := 1
def final_total_marbles : ℕ := 15

-- Find the number of blue marbles initially
def initial_blue_marbles : ℕ := (blue_percentage * initial_total_marbles) / 100

-- Calculate the number of red marbles initially
def initial_red_marbles : ℕ := initial_total_marbles - initial_blue_marbles

-- Calculate the number of red marbles traded
def traded_red_marbles : ℕ := initial_red_marbles - kept_red_marbles

-- Calculate the number of marbles received from the trade
def traded_marbles : ℕ := final_total_marbles - (initial_blue_marbles + kept_red_marbles)

-- The number of blue marbles received per each red marble traded
def blue_per_red : ℕ := traded_marbles / traded_red_marbles

-- Theorem stating that Pete's friend trades 2 blue marbles for each red marble
theorem trade_ratio_blue_per_red : blue_per_red = 2 := by
  -- Proof steps would go here
  sorry

end trade_ratio_blue_per_red_l1342_134221


namespace reggie_games_lost_l1342_134254

-- Given conditions:
def initial_marbles : ℕ := 100
def marbles_per_game : ℕ := 10
def games_played : ℕ := 9
def marbles_after_games : ℕ := 90

-- The statement to prove:
theorem reggie_games_lost : (initial_marbles - marbles_after_games) / marbles_per_game = 1 := 
sorry

end reggie_games_lost_l1342_134254


namespace mark_remaining_money_l1342_134234

theorem mark_remaining_money 
  (initial_money : ℕ) (num_books : ℕ) (cost_per_book : ℕ) (total_cost : ℕ) (remaining_money : ℕ) 
  (H1 : initial_money = 85)
  (H2 : num_books = 10)
  (H3 : cost_per_book = 5)
  (H4 : total_cost = num_books * cost_per_book)
  (H5 : remaining_money = initial_money - total_cost) : 
  remaining_money = 35 := 
by
  sorry

end mark_remaining_money_l1342_134234


namespace distinct_remainders_l1342_134215

theorem distinct_remainders
  (a n : ℕ) (h_pos_a : 0 < a) (h_pos_n : 0 < n)
  (h_div : n ∣ a^n - 1) :
  ∀ i j : ℕ, i ∈ (Finset.range n).image (· + 1) →
            j ∈ (Finset.range n).image (· + 1) →
            (a^i + i) % n = (a^j + j) % n →
            i = j :=
by
  intros i j hi hj h
  sorry

end distinct_remainders_l1342_134215


namespace jeremy_tylenol_duration_l1342_134227

theorem jeremy_tylenol_duration (num_pills : ℕ) (pill_mg : ℕ) (dose_mg : ℕ) (hours_per_dose : ℕ) (hours_per_day : ℕ) 
  (total_tylenol_mg : ℕ := num_pills * pill_mg)
  (num_doses : ℕ := total_tylenol_mg / dose_mg)
  (total_hours : ℕ := num_doses * hours_per_dose) :
  num_pills = 112 → pill_mg = 500 → dose_mg = 1000 → hours_per_dose = 6 → hours_per_day = 24 → 
  total_hours / hours_per_day = 14 := 
by 
  intros; 
  sorry

end jeremy_tylenol_duration_l1342_134227


namespace distance_to_bus_stand_l1342_134230

variable (D : ℝ)

theorem distance_to_bus_stand :
  (D / 4 - D / 5 = 1 / 4) → D = 5 :=
sorry

end distance_to_bus_stand_l1342_134230


namespace distinct_possible_lunches_l1342_134295

namespace SchoolCafeteria

def main_courses : List String := ["Hamburger", "Veggie Burger", "Chicken Sandwich", "Pasta"]
def beverages_when_meat_free : List String := ["Water", "Soda"]
def beverages_when_meat : List String := ["Water"]
def snacks : List String := ["Apple Pie", "Fruit Cup"]

-- Count the total number of distinct possible lunches
def count_distinct_lunches : Nat :=
  let count_options (main_course : String) : Nat :=
    if main_course = "Hamburger" ∨ main_course = "Chicken Sandwich" then
      beverages_when_meat.length * snacks.length
    else
      beverages_when_meat_free.length * snacks.length
  (main_courses.map count_options).sum

theorem distinct_possible_lunches : count_distinct_lunches = 12 := by
  sorry

end SchoolCafeteria

end distinct_possible_lunches_l1342_134295


namespace product_of_numbers_l1342_134267

theorem product_of_numbers (x y : ℕ) (h1 : x + y = 15) (h2 : x - y = 11) : x * y = 26 :=
by
  sorry

end product_of_numbers_l1342_134267


namespace sum_of_yellow_and_blue_is_red_l1342_134270

theorem sum_of_yellow_and_blue_is_red (a b : ℕ) : ∃ k : ℕ, (4 * a + 3) + (4 * b + 2) = 4 * k + 1 :=
by sorry

end sum_of_yellow_and_blue_is_red_l1342_134270


namespace parallel_lines_slope_l1342_134242

theorem parallel_lines_slope {m : ℝ} : 
  (∃ m, (∀ x y : ℝ, 3 * x + 4 * y - 3 = 0 → 6 * x + m * y + 14 = 0)) ↔ m = 8 :=
by
  sorry

end parallel_lines_slope_l1342_134242


namespace new_average_l1342_134268

theorem new_average (avg : ℕ) (n : ℕ) (k : ℕ) (new_avg : ℕ) 
  (h1 : avg = 23) (h2 : n = 10) (h3 : k = 4) : 
  new_avg = (n * avg + n * k) / n → new_avg = 27 :=
by
  intro H
  sorry

end new_average_l1342_134268


namespace darrel_will_receive_l1342_134213

noncomputable def darrel_coins_value : ℝ := 
  let quarters := 127 
  let dimes := 183 
  let nickels := 47 
  let pennies := 237 
  let half_dollars := 64 
  let euros := 32 
  let pounds := 55 
  let quarter_fee_rate := 0.12 
  let dime_fee_rate := 0.07 
  let nickel_fee_rate := 0.15 
  let penny_fee_rate := 0.10 
  let half_dollar_fee_rate := 0.05 
  let euro_exchange_rate := 1.18 
  let euro_fee_rate := 0.03 
  let pound_exchange_rate := 1.39 
  let pound_fee_rate := 0.04 
  let quarters_value := 127 * 0.25 
  let quarters_fee := quarters_value * 0.12 
  let quarters_after_fee := quarters_value - quarters_fee 
  let dimes_value := 183 * 0.10 
  let dimes_fee := dimes_value * 0.07 
  let dimes_after_fee := dimes_value - dimes_fee 
  let nickels_value := 47 * 0.05 
  let nickels_fee := nickels_value * 0.15 
  let nickels_after_fee := nickels_value - nickels_fee 
  let pennies_value := 237 * 0.01 
  let pennies_fee := pennies_value * 0.10 
  let pennies_after_fee := pennies_value - pennies_fee 
  let half_dollars_value := 64 * 0.50 
  let half_dollars_fee := half_dollars_value * 0.05 
  let half_dollars_after_fee := half_dollars_value - half_dollars_fee 
  let euros_value := 32 * 1.18 
  let euros_fee := euros_value * 0.03 
  let euros_after_fee := euros_value - euros_fee 
  let pounds_value := 55 * 1.39 
  let pounds_fee := pounds_value * 0.04 
  let pounds_after_fee := pounds_value - pounds_fee 
  quarters_after_fee + dimes_after_fee + nickels_after_fee + pennies_after_fee + half_dollars_after_fee + euros_after_fee + pounds_after_fee

theorem darrel_will_receive : darrel_coins_value = 189.51 := by
  unfold darrel_coins_value
  sorry

end darrel_will_receive_l1342_134213


namespace solution_set_inequality_l1342_134251

theorem solution_set_inequality (a : ℕ) (h : ∀ x : ℝ, (a-2) * x > (a-2) → x < 1) : a = 0 ∨ a = 1 :=
by
  sorry

end solution_set_inequality_l1342_134251


namespace base_six_to_ten_2154_l1342_134226

def convert_base_six_to_ten (n : ℕ) : ℕ :=
  2 * 6^3 + 1 * 6^2 + 5 * 6^1 + 4 * 6^0

theorem base_six_to_ten_2154 :
  convert_base_six_to_ten 2154 = 502 :=
by
  sorry

end base_six_to_ten_2154_l1342_134226


namespace mark_age_in_5_years_l1342_134298

-- Definitions based on the conditions
def Amy_age := 15
def age_difference := 7

-- Statement specifying the age Mark will be in 5 years
theorem mark_age_in_5_years : (Amy_age + age_difference + 5) = 27 := 
by
  sorry

end mark_age_in_5_years_l1342_134298


namespace time_after_seconds_l1342_134245

def initial_time : Nat := 8 * 60 * 60 -- 8:00:00 a.m. in seconds
def seconds_passed : Nat := 8035
def target_time : Nat := (10 * 60 * 60 + 13 * 60 + 35) -- 10:13:35 in seconds

theorem time_after_seconds : initial_time + seconds_passed = target_time := by
  -- proof skipped
  sorry

end time_after_seconds_l1342_134245


namespace number_of_managers_in_sample_l1342_134222

def totalStaff : ℕ := 160
def salespeople : ℕ := 104
def managers : ℕ := 32
def logisticsPersonnel : ℕ := 24
def sampleSize : ℕ := 20

theorem number_of_managers_in_sample : 
  (managers * (sampleSize / totalStaff) = 4) := by
  sorry

end number_of_managers_in_sample_l1342_134222


namespace find_x_l1342_134271

def op (a b : ℕ) : ℕ := a * b - b + b ^ 2

theorem find_x (x : ℕ) : (∃ x : ℕ, op x 8 = 80) :=
  sorry

end find_x_l1342_134271


namespace incorrect_line_pass_through_Q_l1342_134216

theorem incorrect_line_pass_through_Q (a b : ℝ) : 
  (∀ (k : ℝ), ∃ (Q : ℝ × ℝ), Q = (0, b) ∧ y = k * x + b) →
  (¬ ∃ k : ℝ, ∀ y x, y = k * x + b ∧ x = 0)
:= 
sorry

end incorrect_line_pass_through_Q_l1342_134216


namespace categorize_numbers_l1342_134212

noncomputable def positive_numbers (nums : Set ℝ) : Set ℝ :=
  {x | x ∈ nums ∧ x > 0}

noncomputable def non_neg_integers (nums : Set ℝ) : Set ℝ :=
  {x | x ∈ nums ∧ x ≥ 0 ∧ ∃ n : ℤ, x = n}

noncomputable def negative_fractions (nums : Set ℝ) : Set ℝ :=
  {x | x ∈ nums ∧ x < 0 ∧ ∃ n d : ℤ, d ≠ 0 ∧ (x = n / d)}

def given_set : Set ℝ := {6, -3, 2.4, -3/4, 0, -3.14, 2, -7/2, 2/3}

theorem categorize_numbers :
  positive_numbers given_set = {6, 2.4, 2, 2/3} ∧
  non_neg_integers given_set = {6, 0, 2} ∧
  negative_fractions given_set = {-3/4, -3.14, -7/2} :=
by
  sorry

end categorize_numbers_l1342_134212


namespace correct_ordering_of_f_values_l1342_134269

variable {f : ℝ → ℝ}

theorem correct_ordering_of_f_values
  (hf_even : ∀ x : ℝ, f (-x) = f x)
  (hf_decreasing : ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ > f x₂) :
  f 3 < f (-2) ∧ f (-2) < f 1 := by
  sorry

end correct_ordering_of_f_values_l1342_134269


namespace find_f_2011_l1342_134273

noncomputable def f : ℝ → ℝ := sorry

axiom periodicity (x : ℝ) : f (x + 2) = -f x
axiom specific_interval (x : ℝ) (h2 : 2 < x) (h4 : x < 4) : f x = x + 3

theorem find_f_2011 : f 2011 = 6 :=
by {
  -- Leave this part to be filled with the actual proof,
  -- satisfying the initial conditions and concluding f(2011) = 6
  sorry
}

end find_f_2011_l1342_134273


namespace simplify_polynomial_l1342_134204

theorem simplify_polynomial :
  (2 * x * (4 * x ^ 3 - 3 * x + 1) - 4 * (2 * x ^ 3 - x ^ 2 + 3 * x - 5)) =
  8 * x ^ 4 - 8 * x ^ 3 - 2 * x ^ 2 - 10 * x + 20 :=
by
  sorry

end simplify_polynomial_l1342_134204


namespace twenty_fifty_yuan_bills_unique_l1342_134211

noncomputable def twenty_fifty_yuan_bills (x y : ℕ) : Prop :=
  x + y = 260 ∧ 20 * x + 50 * y = 100 * 100

theorem twenty_fifty_yuan_bills_unique (x y : ℕ) (h : twenty_fifty_yuan_bills x y) :
  x = 100 ∧ y = 160 :=
by
  sorry

end twenty_fifty_yuan_bills_unique_l1342_134211


namespace goats_difference_l1342_134244

-- Definitions of Adam's, Andrew's, and Ahmed's goats
def adam_goats : ℕ := 7
def andrew_goats : ℕ := 2 * adam_goats + 5
def ahmed_goats : ℕ := 13

-- The theorem to prove the difference in goats
theorem goats_difference : andrew_goats - ahmed_goats = 6 :=
by
  sorry

end goats_difference_l1342_134244


namespace Johnson_Martinez_tied_at_end_of_september_l1342_134260

open Nat

-- Define the monthly home runs for Johnson and Martinez
def Johnson_runs : List Nat := [3, 8, 15, 12, 5, 7, 14]
def Martinez_runs : List Nat := [0, 3, 9, 20, 7, 12, 13]

-- Define the cumulated home runs for Johnson and Martinez over the months
def total_runs (runs : List Nat) : List Nat :=
  runs.scanl (· + ·) 0

-- State the theorem to prove that they are tied in total runs at the end of September
theorem Johnson_Martinez_tied_at_end_of_september :
  (total_runs Johnson_runs).getLast (by decide) =
  (total_runs Martinez_runs).getLast (by decide) := by
  sorry

end Johnson_Martinez_tied_at_end_of_september_l1342_134260


namespace milk_for_6_cookies_l1342_134274

/-- Given conditions for baking cookies -/
def quarts_to_pints : ℕ := 2 -- 2 pints in a quart
def milk_for_24_cookies : ℕ := 5 -- 5 quarts of milk for 24 cookies

/-- Theorem to prove the number of pints needed to bake 6 cookies -/
theorem milk_for_6_cookies : 
  (milk_for_24_cookies * quarts_to_pints * 6 / 24 : ℝ) = 2.5 := 
by 
  sorry -- Proof is omitted

end milk_for_6_cookies_l1342_134274


namespace domain_of_function_l1342_134200

theorem domain_of_function (x : ℝ) : 4 - x ≥ 0 ∧ x ≠ 2 ↔ (x ≤ 4 ∧ x ≠ 2) :=
sorry

end domain_of_function_l1342_134200


namespace ramu_profit_percent_l1342_134281

noncomputable def carCost : ℝ := 42000
noncomputable def repairCost : ℝ := 13000
noncomputable def sellingPrice : ℝ := 60900
noncomputable def totalCost : ℝ := carCost + repairCost
noncomputable def profit : ℝ := sellingPrice - totalCost
noncomputable def profitPercent : ℝ := (profit / totalCost) * 100

theorem ramu_profit_percent : profitPercent = 10.73 := 
by
  sorry

end ramu_profit_percent_l1342_134281


namespace range_of_a_l1342_134214

theorem range_of_a (a : ℝ) (H : ∀ x : ℝ, x ≤ 1 → 4 - a * 2^x > 0) : a < 2 :=
sorry

end range_of_a_l1342_134214


namespace quadratic_root_exists_in_range_l1342_134296

theorem quadratic_root_exists_in_range :
  ∃ x : ℝ, 1.1 < x ∧ x < 1.2 ∧ x^2 + 3 * x - 5 = 0 := 
by
  sorry

end quadratic_root_exists_in_range_l1342_134296


namespace probability_of_ace_then_spade_l1342_134247

theorem probability_of_ace_then_spade :
  let P := (1 / 52) * (12 / 51) + (3 / 52) * (13 / 51)
  P = (3 / 127) :=
by
  sorry

end probability_of_ace_then_spade_l1342_134247


namespace necessarily_positive_l1342_134287

theorem necessarily_positive (x y w : ℝ) (h1 : 0 < x ∧ x < 0.5) (h2 : -0.5 < y ∧ y < 0) (h3 : 0.5 < w ∧ w < 1) : 
  0 < w - y :=
sorry

end necessarily_positive_l1342_134287


namespace preston_receives_total_amount_l1342_134292

theorem preston_receives_total_amount :
  let price_per_sandwich := 5
  let delivery_fee := 20
  let num_sandwiches := 18
  let tip_percent := 0.10
  let sandwich_cost := num_sandwiches * price_per_sandwich
  let initial_total := sandwich_cost + delivery_fee
  let tip := initial_total * tip_percent
  let final_total := initial_total + tip
  final_total = 121 := 
by
  sorry

end preston_receives_total_amount_l1342_134292


namespace simplify_expression_l1342_134249

theorem simplify_expression (y : ℝ) : 2 - (2 - (2 - (2 - (2 - y)))) = 4 - y :=
by
  sorry

end simplify_expression_l1342_134249


namespace coin_count_l1342_134218

-- Define the conditions and the proof goal
theorem coin_count (total_value : ℕ) (coin_value_20 : ℕ) (coin_value_25 : ℕ) 
    (num_20_paise_coins : ℕ) (total_value_paise : total_value = 7100)
    (value_20_paise : coin_value_20 = 20) (value_25_paise : coin_value_25 = 25)
    (num_20_paise : num_20_paise_coins = 300) : 
    (300 + 44 = 344) :=
by
  -- The proof would go here, currently omitted with sorry
  sorry

end coin_count_l1342_134218


namespace proof_problem_l1342_134294

def p : Prop := ∃ k : ℕ, 0 = 2 * k
def q : Prop := ∃ k : ℕ, 3 = 2 * k

theorem proof_problem : p ∨ q :=
by
  sorry

end proof_problem_l1342_134294


namespace largest_three_digit_in_pascal_triangle_l1342_134208

-- Define Pascal's triangle and binomial coefficient
def pascal (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem about the first appearance of the number 999 in Pascal's triangle
theorem largest_three_digit_in_pascal_triangle :
  ∃ (n : ℕ), n = 1000 ∧ ∃ (k : ℕ), pascal n k = 999 :=
sorry

end largest_three_digit_in_pascal_triangle_l1342_134208
