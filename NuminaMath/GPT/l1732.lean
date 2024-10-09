import Mathlib

namespace convert_spherical_to_rectangular_l1732_173229

noncomputable def spherical_to_rectangular (rho theta phi : ℝ) : ℝ × ℝ × ℝ :=
  (rho * Real.sin phi * Real.cos theta,
   rho * Real.sin phi * Real.sin theta,
   rho * Real.cos phi)

theorem convert_spherical_to_rectangular :
  spherical_to_rectangular 4 (Real.pi / 6) (Real.pi / 4) = (2 * Real.sqrt 3, Real.sqrt 2, 2 * Real.sqrt 2) :=
by
  -- Define the spherical coordinates
  let rho := 4
  let theta := Real.pi / 6
  let phi := Real.pi / 4

  -- Calculate x, y, z using conversion formulas
  sorry

end convert_spherical_to_rectangular_l1732_173229


namespace sum_of_center_coordinates_l1732_173283

theorem sum_of_center_coordinates : 
  ∀ (x y : ℝ), 
  (x^2 + y^2 = 6*x - 10*y + 24) -> 
  (∃ (cx cy : ℝ), (x^2 - 6*x + y^2 + 10*y = (cx - 3)^2 + (cy + 5)^2 + 58) ∧ (cx + cy = -2)) :=
  sorry

end sum_of_center_coordinates_l1732_173283


namespace cost_of_item_is_200_l1732_173259

noncomputable def cost_of_each_item (x : ℕ) : ℕ :=
  let before_discount := 7 * x -- Total cost before discount
  let discount_part := before_discount - 1000 -- Part of the cost over $1000
  let discount := discount_part / 10 -- 10% of the part over $1000
  let after_discount := before_discount - discount -- Total cost after discount
  after_discount

theorem cost_of_item_is_200 :
  (∃ x : ℕ, cost_of_each_item x = 1360) ↔ x = 200 :=
by
  sorry

end cost_of_item_is_200_l1732_173259


namespace figure_surface_area_calculation_l1732_173249

-- Define the surface area of one bar
def bar_surface_area : ℕ := 18

-- Define the surface area lost at the junctions
def surface_area_lost : ℕ := 2

-- Define the effective surface area of one bar after accounting for overlaps
def effective_bar_surface_area : ℕ := bar_surface_area - surface_area_lost

-- Define the number of bars used in the figure
def number_of_bars : ℕ := 4

-- Define the total surface area of the figure
def total_surface_area : ℕ := number_of_bars * effective_bar_surface_area

-- The theorem stating the total surface area of the figure
theorem figure_surface_area_calculation : total_surface_area = 64 := by
  sorry

end figure_surface_area_calculation_l1732_173249


namespace min_sum_4410_l1732_173211

def min_sum (a b c d : ℕ) : ℕ := a + b + c + d

theorem min_sum_4410 :
  ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a * b * c * d = 4410 ∧ min_sum a b c d = 69 :=
sorry

end min_sum_4410_l1732_173211


namespace problem_statement_l1732_173274

noncomputable def A : Set ℝ := {x | 2 < x ∧ x ≤ 6}
noncomputable def B : Set ℝ := {x | x^2 - 4 * x < 0}
noncomputable def C (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2 * m - 1}

theorem problem_statement (m : ℝ) :
    (A ∩ B = {x | 2 < x ∧ x < 4}) ∧
    (¬(A ∪ B) = {x | x ≤ 0 ∨ x > 6}) ∧
    (C m ⊆ B → m ∈ Set.Iic (5/2)) := 
by
  sorry

end problem_statement_l1732_173274


namespace repeating_decimal_eq_fraction_l1732_173290

noncomputable def repeating_decimal_to_fraction (x : ℝ) : ℝ :=
  let x : ℝ := 4.5656565656 -- * 0.5656... repeating
  (100*x - x) / (100 - 1)

-- Define the theorem we want to prove
theorem repeating_decimal_eq_fraction : 
  ∀ x : ℝ, x = 4.565656 -> x = (452 : ℝ) / (99 : ℝ) :=
by
  intro x h
  -- here we would provide the proof steps, but since it's omitted
  -- we'll use sorry to skip it.
  sorry

end repeating_decimal_eq_fraction_l1732_173290


namespace inverse_of_matrix_C_l1732_173265

-- Define the given matrix C
def C : Matrix (Fin 3) (Fin 3) ℚ := ![
  ![1, 2, 1],
  ![3, -5, 3],
  ![2, 7, -1]
]

-- Define the claimed inverse of the matrix C
def C_inv : Matrix (Fin 3) (Fin 3) ℚ := (1 / 33 : ℚ) • ![
  ![-16,  9,  11],
  ![  9, -3,   0],
  ![ 31, -3, -11]
]

-- Statement to prove that C_inv is the inverse of C
theorem inverse_of_matrix_C : C * C_inv = 1 ∧ C_inv * C = 1 := by
  sorry

end inverse_of_matrix_C_l1732_173265


namespace gcd_1729_867_l1732_173245

theorem gcd_1729_867 : Nat.gcd 1729 867 = 1 :=
by 
  sorry

end gcd_1729_867_l1732_173245


namespace min_value_function_l1732_173254

theorem min_value_function (x y : ℝ) (hx : x > 1) (hy : y > 1) : 
  (∀ x y : ℝ, x > 1 ∧ y > 1 → (min ((x^2 + y) / (y^2 - 1) + (y^2 + x) / (x^2 - 1)) = 8 / 3)) := 
sorry

end min_value_function_l1732_173254


namespace u_2023_is_4_l1732_173244

def f (x : ℕ) : ℕ :=
  match x with
  | 1 => 5
  | 2 => 3
  | 3 => 2
  | 4 => 1
  | 5 => 4
  | _ => 0  -- f is only defined for x in {1, 2, 3, 4, 5}

def u : ℕ → ℕ
| 0 => 5
| (n + 1) => f (u n)

theorem u_2023_is_4 : u 2023 = 4 := by
  sorry

end u_2023_is_4_l1732_173244


namespace train_problem_l1732_173247

variables (x : ℝ) (p q : ℝ)
variables (speed_p speed_q : ℝ) (dist_diff : ℝ)

theorem train_problem
  (speed_p : speed_p = 50)
  (speed_q : speed_q = 40)
  (dist_diff : ∀ x, x = 500 → p = 50 * x ∧ q = 40 * (500 - 100)) :
  p + q = 900 :=
by
sorry

end train_problem_l1732_173247


namespace calc_problem1_calc_problem2_calc_problem3_calc_problem4_l1732_173260

theorem calc_problem1 : (-3 + 8 - 15 - 6 = -16) :=
by
  sorry

theorem calc_problem2 : (-4/13 - (-4/17) + 4/13 + (-13/17) = -9/17) :=
by
  sorry

theorem calc_problem3 : (-25 - (5/4 * 4/5) - (-16) = -10) :=
by
  sorry

theorem calc_problem4 : (-2^4 - (1/2 * (5 - (-3)^2)) = -14) :=
by
  sorry

end calc_problem1_calc_problem2_calc_problem3_calc_problem4_l1732_173260


namespace domain_intersection_l1732_173270

theorem domain_intersection (A B : Set ℝ) 
    (h1 : A = {x | x < 1})
    (h2 : B = {y | y ≥ 0}) : A ∩ B = {z | 0 ≤ z ∧ z < 1} := 
by
  sorry

end domain_intersection_l1732_173270


namespace Josh_lost_marbles_l1732_173280

theorem Josh_lost_marbles :
  let original_marbles := 9.5
  let current_marbles := 4.25
  original_marbles - current_marbles = 5.25 :=
by
  sorry

end Josh_lost_marbles_l1732_173280


namespace point_relationship_l1732_173216

def quadratic_function (x : ℝ) (c : ℝ) : ℝ :=
  -(x - 1) ^ 2 + c

noncomputable def y1_def (c : ℝ) : ℝ := quadratic_function (-3) c
noncomputable def y2_def (c : ℝ) : ℝ := quadratic_function (-1) c
noncomputable def y3_def (c : ℝ) : ℝ := quadratic_function 5 c

theorem point_relationship (c : ℝ) :
  y2_def c > y1_def c ∧ y1_def c = y3_def c :=
by
  sorry

end point_relationship_l1732_173216


namespace two_numbers_and_sum_l1732_173203

theorem two_numbers_and_sum (x y : ℕ) (hx : x * y = 18) (hy : x - y = 4) : x + y = 10 :=
sorry

end two_numbers_and_sum_l1732_173203


namespace origin_moves_3sqrt5_under_dilation_l1732_173261

/--
Given:
1. The original circle has radius 3 centered at point B(3, 3).
2. The dilated circle has radius 6 centered at point B'(9, 12).

Prove that the distance moved by the origin O(0, 0) under this dilation is 3 * sqrt(5).
-/
theorem origin_moves_3sqrt5_under_dilation:
  let B := (3, 3)
  let B' := (9, 12)
  let radius_B := 3
  let radius_B' := 6
  let dilation_center := (-3, -6)
  let origin := (0, 0)
  let k := radius_B' / radius_B
  let d_0 := Real.sqrt ((-3 : ℝ)^2 + (-6 : ℝ)^2)
  let d_1 := k * d_0
  d_1 - d_0 = 3 * Real.sqrt (5 : ℝ) := by sorry

end origin_moves_3sqrt5_under_dilation_l1732_173261


namespace proof_problem_l1732_173263

theorem proof_problem
  (x y z : ℤ)
  (h1 : x = 11 * y + 4)
  (h2 : 2 * x = 3 * y * z + 3)
  (h3 : 13 * y - x = 1) :
  z = 8 := by
  sorry

end proof_problem_l1732_173263


namespace last_two_non_zero_digits_of_75_factorial_l1732_173268

theorem last_two_non_zero_digits_of_75_factorial : 
  ∃ (d : ℕ), d = 32 := sorry

end last_two_non_zero_digits_of_75_factorial_l1732_173268


namespace parrots_in_each_cage_l1732_173215

theorem parrots_in_each_cage (P : ℕ) (h : 9 * P + 9 * 6 = 72) : P = 2 :=
sorry

end parrots_in_each_cage_l1732_173215


namespace find_angle_B_l1732_173293

-- Given definitions and conditions
variables {a b c : ℝ}
variables {A B C : ℝ}
variable (h1 : (a + b + c) * (a - b + c) = a * c )

-- Statement of the proof problem
theorem find_angle_B (h1 : (a + b + c) * (a - b + c) = a * c) :
  B = 2 * π / 3 :=
sorry

end find_angle_B_l1732_173293


namespace lewis_weekly_earning_l1732_173297

theorem lewis_weekly_earning
  (weeks : ℕ)
  (weekly_rent : ℤ)
  (total_savings : ℤ)
  (h1 : weeks = 1181)
  (h2 : weekly_rent = 216)
  (h3 : total_savings = 324775)
  : ∃ (E : ℤ), E = 49075 / 100 :=
by
  let E := 49075 / 100
  use E
  sorry -- The proof would go here

end lewis_weekly_earning_l1732_173297


namespace probability_of_specific_combination_l1732_173214

def count_all_clothes : ℕ := 6 + 7 + 8 + 3
def choose4_out_of_24 : ℕ := Nat.choose 24 4
def choose1_shirt : ℕ := 6
def choose1_pair_shorts : ℕ := 7
def choose1_pair_socks : ℕ := 8
def choose1_hat : ℕ := 3
def favorable_outcomes : ℕ := choose1_shirt * choose1_pair_shorts * choose1_pair_socks * choose1_hat
def probability_of_combination : ℚ := favorable_outcomes / choose4_out_of_24

theorem probability_of_specific_combination :
  probability_of_combination = 144 / 1815 := by
sorry

end probability_of_specific_combination_l1732_173214


namespace calculate_ggg1_l1732_173224

def g (x : ℕ) : ℕ := 7 * x + 3

theorem calculate_ggg1 : g (g (g 1)) = 514 := 
by
  sorry

end calculate_ggg1_l1732_173224


namespace karl_savings_l1732_173234

noncomputable def cost_per_notebook : ℝ := 3.75
noncomputable def notebooks_bought : ℕ := 8
noncomputable def discount_rate : ℝ := 0.25
noncomputable def original_total_cost : ℝ := notebooks_bought * cost_per_notebook
noncomputable def discount_per_notebook : ℝ := cost_per_notebook * discount_rate
noncomputable def discounted_price_per_notebook : ℝ := cost_per_notebook - discount_per_notebook
noncomputable def discounted_total_cost : ℝ := notebooks_bought * discounted_price_per_notebook
noncomputable def total_savings : ℝ := original_total_cost - discounted_total_cost

theorem karl_savings : total_savings = 7.50 := by 
  sorry

end karl_savings_l1732_173234


namespace final_number_proof_l1732_173228

/- Define the symbols and their corresponding values -/
def cat := 1
def chicken := 5
def crab := 2
def bear := 4
def goat := 3

/- Define the equations from the conditions -/
axiom row4_eq : 5 * crab = 10
axiom col5_eq : 4 * crab + goat = 11
axiom row2_eq : 2 * goat + crab + 2 * bear = 16
axiom col2_eq : cat + bear + 2 * goat + crab = 13
axiom col3_eq : 2 * crab + 2 * chicken + goat = 17

/- Final number is derived by concatenating digits -/
def final_number := cat * 10000 + chicken * 1000 + crab * 100 + bear * 10 + goat

/- Theorem to prove the final number is 15243 -/
theorem final_number_proof : final_number = 15243 := by
  -- Proof steps to be provided here.
  sorry

end final_number_proof_l1732_173228


namespace estimate_students_height_at_least_165_l1732_173235

theorem estimate_students_height_at_least_165 
  (sample_size : ℕ)
  (total_school_size : ℕ)
  (students_165_170 : ℕ)
  (students_170_175 : ℕ)
  (h_sample : sample_size = 100)
  (h_total_school : total_school_size = 1000)
  (h_students_165_170 : students_165_170 = 20)
  (h_students_170_175 : students_170_175 = 30)
  : (students_165_170 + students_170_175) * (total_school_size / sample_size) = 500 := 
by
  sorry

end estimate_students_height_at_least_165_l1732_173235


namespace work_done_in_one_day_by_A_and_B_l1732_173246

noncomputable def A_days : ℕ := 12
noncomputable def B_days : ℕ := A_days / 2

theorem work_done_in_one_day_by_A_and_B : 1 / (A_days : ℚ) + 1 / (B_days : ℚ) = 1 / 4 := by
  sorry

end work_done_in_one_day_by_A_and_B_l1732_173246


namespace find_number_l1732_173202

theorem find_number (x : ℝ) (h : x - (3/5) * x = 56) : x = 140 :=
sorry

end find_number_l1732_173202


namespace arithmetic_sequence_common_difference_l1732_173248

theorem arithmetic_sequence_common_difference (a_1 a_4 a_5 d : ℤ) 
  (h1 : a_1 + a_5 = 10) 
  (h2 : a_4 = 7) 
  (h3 : a_4 = a_1 + 3 * d) 
  (h4 : a_5 = a_1 + 4 * d) : 
  d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l1732_173248


namespace function_decreasing_interval_l1732_173208

variable (a b : ℝ)

def f (x : ℝ) : ℝ := x^2 * (a * x + b)

theorem function_decreasing_interval :
  (deriv (f a b) 2 = 0) ∧ (deriv (f a b) 1 = -3) →
  ∃ (a b : ℝ), (deriv (f a b) x < 0) ↔ (0 < x ∧ x < 2) := sorry

end function_decreasing_interval_l1732_173208


namespace ratio_of_m1_and_m2_l1732_173284

theorem ratio_of_m1_and_m2 (m a b m1 m2 : ℝ) (h1 : a^2 * m - 3 * a * m + 2 * a + 7 = 0) (h2 : b^2 * m - 3 * b * m + 2 * b + 7 = 0) 
  (h3 : (a / b) + (b / a) = 2) (h4 : m1^2 * 9 - m1 * 28 + 4 = 0) (h5 : m2^2 * 9 - m2 * 28 + 4 = 0) : 
  (m1 / m2) + (m2 / m1) = 194 / 9 := 
sorry

end ratio_of_m1_and_m2_l1732_173284


namespace inequality_solution_l1732_173237

theorem inequality_solution (x : ℝ) (h1 : x > 0) (h2 : x ≠ 6) : x^3 - 12 * x^2 + 36 * x > 0 :=
sorry

end inequality_solution_l1732_173237


namespace min_distance_equals_sqrt2_over_2_l1732_173205

noncomputable def min_distance_from_point_to_line (m n : ℝ) : ℝ :=
  (|m + n + 10|) / Real.sqrt (1^2 + 1^2)

def circle_eq (m n : ℝ) : Prop :=
  (m - 1 / 2)^2 + (n - 1 / 2)^2 = 1 / 2

theorem min_distance_equals_sqrt2_over_2 (m n : ℝ) (h1 : circle_eq m n) :
  min_distance_from_point_to_line m n = 1 / (Real.sqrt 2) :=
sorry

end min_distance_equals_sqrt2_over_2_l1732_173205


namespace smallest_solution_l1732_173238

theorem smallest_solution (x : ℝ) :
  (∃ x, (3 * x) / (x - 3) + (3 * x^2 - 36) / (x + 3) = 15) →
  x = -1 := 
sorry

end smallest_solution_l1732_173238


namespace number_of_ways_to_select_president_and_vice_president_l1732_173273

-- Define the given conditions
def num_candidates : Nat := 4

-- Define the problem to prove
theorem number_of_ways_to_select_president_and_vice_president : (num_candidates * (num_candidates - 1)) = 12 :=
by
  -- This is where the proof would go, but we are skipping it
  sorry

end number_of_ways_to_select_president_and_vice_president_l1732_173273


namespace yogurt_price_is_5_l1732_173264

theorem yogurt_price_is_5
  (yogurt_pints : ℕ)
  (gum_packs : ℕ)
  (shrimp_trays : ℕ)
  (total_cost : ℝ)
  (shrimp_cost : ℝ)
  (gum_fraction : ℝ)
  (price_frozen_yogurt : ℝ) :
  yogurt_pints = 5 →
  gum_packs = 2 →
  shrimp_trays = 5 →
  total_cost = 55 →
  shrimp_cost = 5 →
  gum_fraction = 0.5 →
  5 * price_frozen_yogurt + 2 * (gum_fraction * price_frozen_yogurt) + 5 * shrimp_cost = total_cost →
  price_frozen_yogurt = 5 :=
by
  intro hp hg hs ht hc hf h_formula
  sorry

end yogurt_price_is_5_l1732_173264


namespace domain_of_function_l1732_173217

noncomputable def function_domain := {x : ℝ | x * (3 - x) ≥ 0 ∧ x - 1 ≥ 0 }

theorem domain_of_function: function_domain = {x : ℝ | 1 ≤ x ∧ x ≤ 3} :=
by
  sorry

end domain_of_function_l1732_173217


namespace geometric_sequence_fourth_term_l1732_173222

theorem geometric_sequence_fourth_term (a : ℝ) (r : ℝ) (h : a = 512) (h1 : a * r^5 = 125) :
  a * r^3 = 1536 :=
by
  sorry

end geometric_sequence_fourth_term_l1732_173222


namespace domain_range_a_l1732_173253

theorem domain_range_a (a : ℝ) : (∀ x : ℝ, x^2 + 2 * x + a > 0) ↔ 1 < a :=
by
  sorry

end domain_range_a_l1732_173253


namespace download_time_is_2_hours_l1732_173294

theorem download_time_is_2_hours (internet_speed : ℕ) (f1 f2 f3 : ℕ) (total_size : ℕ)
  (total_min : ℕ) (hours : ℕ) :
  internet_speed = 2 ∧ f1 = 80 ∧ f2 = 90 ∧ f3 = 70 ∧ total_size = f1 + f2 + f3
  ∧ total_min = total_size / internet_speed ∧ hours = total_min / 60 → hours = 2 :=
by
  sorry

end download_time_is_2_hours_l1732_173294


namespace factor_quadratic_l1732_173266

theorem factor_quadratic (x : ℝ) (m n : ℝ) 
  (hm : m^2 = 16) (hn : n^2 = 25) (hmn : 2 * m * n = 40) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := 
by sorry

end factor_quadratic_l1732_173266


namespace suraya_picked_more_apples_l1732_173291

theorem suraya_picked_more_apples (k c s : ℕ)
  (h_kayla : k = 20)
  (h_caleb : c = k - 5)
  (h_suraya : s = k + 7) :
  s - c = 12 :=
by
  -- Mark this as a place where the proof can be provided
  sorry

end suraya_picked_more_apples_l1732_173291


namespace correct_sampling_methods_l1732_173278

-- Defining the conditions
def high_income_families : ℕ := 50
def middle_income_families : ℕ := 300
def low_income_families : ℕ := 150
def total_residents : ℕ := 500
def sample_size : ℕ := 100
def worker_group_size : ℕ := 10
def selected_workers : ℕ := 3

-- Definitions of sampling methods
inductive SamplingMethod
| random
| systematic
| stratified

open SamplingMethod

-- Problem statement in Lean 4
theorem correct_sampling_methods :
  (total_residents = high_income_families + middle_income_families + low_income_families) →
  (sample_size = 100) →
  (worker_group_size = 10) →
  (selected_workers = 3) →
  (chosen_method_for_task1 = SamplingMethod.stratified) →
  (chosen_method_for_task2 = SamplingMethod.random) →
  (chosen_method_for_task1, chosen_method_for_task2) = (SamplingMethod.stratified, SamplingMethod.random) :=
by
  intros
  sorry -- Proof to be filled in

end correct_sampling_methods_l1732_173278


namespace inequality_satisfaction_l1732_173289

theorem inequality_satisfaction (a b : ℝ) (h : 0 < a ∧ a < b) : 
  a < Real.sqrt (a * b) ∧ Real.sqrt (a * b) < (a + b) / 2 ∧ (a + b) / 2 < b :=
by
  sorry

end inequality_satisfaction_l1732_173289


namespace shelby_stars_yesterday_l1732_173220

-- Define the number of stars earned yesterday
def stars_yesterday : ℕ := sorry

-- Condition 1: In all, Shelby earned 7 gold stars
def stars_total : ℕ := 7

-- Condition 2: Today, she earned 3 more gold stars
def stars_today : ℕ := 3

-- The proof statement that combines the conditions 
-- and question to the correct answer
theorem shelby_stars_yesterday (y : ℕ) (h1 : y + stars_today = stars_total) : y = 4 := 
by
  -- Placeholder for the actual proof
  sorry

end shelby_stars_yesterday_l1732_173220


namespace black_to_white_area_ratio_l1732_173282

noncomputable def radius1 : ℝ := 2
noncomputable def radius2 : ℝ := 4
noncomputable def radius3 : ℝ := 6
noncomputable def radius4 : ℝ := 8
noncomputable def radius5 : ℝ := 10

noncomputable def area (r : ℝ) : ℝ := Real.pi * r^2

noncomputable def black_area : ℝ :=
  area radius1 + (area radius3 - area radius2) + (area radius5 - area radius4)

noncomputable def white_area : ℝ :=
  (area radius2 - area radius1) + (area radius4 - area radius3)

theorem black_to_white_area_ratio :
  black_area / white_area = 3 / 2 := by
  sorry

end black_to_white_area_ratio_l1732_173282


namespace person_a_work_days_l1732_173285

theorem person_a_work_days (x : ℝ) (h1 : 1 / 6 + 1 / x = 1 / 3.75) : x = 10 := 
sorry

end person_a_work_days_l1732_173285


namespace range_f_l1732_173201

noncomputable def f (x : ℝ) : ℝ := 1 / (2 - x) ^ 3

theorem range_f : Set.range f = Set.Ioi 0 ∪ Set.Iio 0 := by
  sorry

end range_f_l1732_173201


namespace chinaman_change_possible_l1732_173250

def pence (x : ℕ) := x -- defining the value of pence as a natural number

def ching_chang_by_value (d : ℕ) := 
  (2 * pence d) + (4 * (2 * pence d) / 15)

def equivalent_value_of_half_crown (d : ℕ) := 30 * pence d

def coin_value_with_holes (holes_value : ℕ) (value_per_eleven : ℕ) := 
  (value_per_eleven * ching_chang_by_value 1) / 11

theorem chinaman_change_possible :
  ∃ (x y z : ℕ), 
  (7 * coin_value_with_holes 15 11) + (1 * coin_value_with_holes 16 11) + (0 * coin_value_with_holes 17 11) = 
  equivalent_value_of_half_crown 1 :=
sorry

end chinaman_change_possible_l1732_173250


namespace album_count_l1732_173269

theorem album_count (A B S : ℕ) (hA : A = 23) (hB : B = 9) (hS : S = 15) : 
  (A - S) + B = 17 :=
by
  -- Variables and conditions
  have Andrew_unique : ℕ := A - S
  have Bella_unique : ℕ := B
  -- Proof starts here
  sorry

end album_count_l1732_173269


namespace polynomial_div_simplify_l1732_173236

theorem polynomial_div_simplify (x : ℝ) (hx : x ≠ 0) :
  (6 * x ^ 4 - 4 * x ^ 3 + 2 * x ^ 2) / (2 * x ^ 2) = 3 * x ^ 2 - 2 * x + 1 :=
by sorry

end polynomial_div_simplify_l1732_173236


namespace Ivan_can_safely_make_the_journey_l1732_173243

def eruption_cycle_first_crater (t : ℕ) : Prop :=
  ∃ n : ℕ, t = 1 + 18 * n

def eruption_cycle_second_crater (t : ℕ) : Prop :=
  ∃ m : ℕ, t = 1 + 10 * m

def is_safe (start_time : ℕ) : Prop :=
  ∀ t, start_time ≤ t ∧ t < start_time + 16 → 
    ¬ eruption_cycle_first_crater t ∧ 
    ¬ (t ≥ start_time + 12 ∧ eruption_cycle_second_crater t)

theorem Ivan_can_safely_make_the_journey : ∃ t : ℕ, is_safe (38 + t) :=
sorry

end Ivan_can_safely_make_the_journey_l1732_173243


namespace sara_quarters_final_l1732_173295

def initial_quarters : ℕ := 21
def quarters_from_dad : ℕ := 49
def quarters_spent_at_arcade : ℕ := 15
def dollar_bills_from_mom : ℕ := 2
def quarters_per_dollar : ℕ := 4

theorem sara_quarters_final :
  (initial_quarters + quarters_from_dad - quarters_spent_at_arcade + dollar_bills_from_mom * quarters_per_dollar) = 63 :=
by
  sorry

end sara_quarters_final_l1732_173295


namespace prime_1021_n_unique_l1732_173296

theorem prime_1021_n_unique :
  ∃! (n : ℕ), n ≥ 2 ∧ Prime (n^3 + 2 * n + 1) :=
sorry

end prime_1021_n_unique_l1732_173296


namespace range_of_t_l1732_173292

theorem range_of_t (t : ℝ) (x : ℝ) : (1 < x ∧ x ≤ 4) → (|x - t| < 1 ↔ 2 ≤ t ∧ t ≤ 3) :=
by
  sorry

end range_of_t_l1732_173292


namespace perpendicular_parallel_l1732_173218

variables {a b : Line} {α : Plane}

-- Definition of perpendicular and parallel relations should be available
-- since their exact details were not provided, placeholder functions will be used for demonstration

-- Placeholder definitions for perpendicular and parallel (they should be accurately defined elsewhere)
def perp (l : Line) (p : Plane) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry

theorem perpendicular_parallel {a b : Line} {α : Plane}
    (a_perp_alpha : perp a α)
    (b_perp_alpha : perp b α)
    : parallel a b :=
sorry

end perpendicular_parallel_l1732_173218


namespace servant_position_for_28_purses_servant_position_for_27_purses_l1732_173225

-- Definitions based on problem conditions
def total_wealthy_men: ℕ := 7

def valid_purse_placement (n: ℕ): Prop := 
  (n ≤ total_wealthy_men * (total_wealthy_men + 1) / 2)

def get_servant_position (n: ℕ): ℕ := 
  if n = 28 then total_wealthy_men else if n = 27 then 6 else 0

-- Proof statements to equate conditions with the answers
theorem servant_position_for_28_purses : 
  get_servant_position 28 = 7 :=
sorry

theorem servant_position_for_27_purses : 
  get_servant_position 27 = 6 ∨ get_servant_position 27 = 7 :=
sorry

end servant_position_for_28_purses_servant_position_for_27_purses_l1732_173225


namespace intersection_of_sets_l1732_173206

def set_M : Set ℝ := { x : ℝ | (x + 2) * (x - 1) < 0 }
def set_N : Set ℝ := { x : ℝ | x + 1 < 0 }
def intersection (A B : Set ℝ) : Set ℝ := { x : ℝ | x ∈ A ∧ x ∈ B }

theorem intersection_of_sets :
  intersection set_M set_N = { x : ℝ | -2 < x ∧ x < -1 } := 
by
  sorry

end intersection_of_sets_l1732_173206


namespace range_of_a_solution_set_of_inequality_l1732_173239

-- Lean statement for Part 1
theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  ∀ x : ℝ, x^2 - 2 * a * x + a > 0 :=
by
  sorry

-- Lean statement for Part 2
theorem solution_set_of_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  { x : ℝ | a^(x^2 - 3) < a^(2 * x) ∧ a^(2 * x) < 1 } = { x : ℝ | x > 3 } :=
by
  sorry

end range_of_a_solution_set_of_inequality_l1732_173239


namespace f_2017_eq_2018_l1732_173286

def f (n : ℕ) : ℕ := sorry

theorem f_2017_eq_2018 (f : ℕ → ℕ) (h1 : ∀ n, f (f n) + f n = 2 * n + 3) (h2 : f 0 = 1) : f 2017 = 2018 :=
sorry

end f_2017_eq_2018_l1732_173286


namespace exists_super_number_B_l1732_173232

-- Define a function is_super_number to identify super numbers.
def is_super_number (A : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 ≤ A n ∧ A n < 10

-- Define a function zero_super_number to represent the super number with all digits zero.
def zero_super_number (n : ℕ) := 0

-- Task: Prove the existence of B such that A + B = zero_super_number.
theorem exists_super_number_B (A : ℕ → ℕ) (hA : is_super_number A) :
  ∃ B : ℕ → ℕ, is_super_number B ∧ (∀ n : ℕ, (A n + B n) % 10 = zero_super_number n) :=
sorry

end exists_super_number_B_l1732_173232


namespace sum_of_new_dimensions_l1732_173204

theorem sum_of_new_dimensions (s : ℕ) (h₁ : s^2 = 36) (h₂ : s' = s - 1) : s' + s' + s' = 15 :=
sorry

end sum_of_new_dimensions_l1732_173204


namespace find_last_number_l1732_173233

theorem find_last_number (A B C D E F G : ℝ)
    (h1 : (A + B + C + D) / 4 = 13)
    (h2 : (D + E + F + G) / 4 = 15)
    (h3 : E + F + G = 55)
    (h4 : D^2 = G) :
  G = 25 := by 
  sorry

end find_last_number_l1732_173233


namespace quadratic_cubic_inequalities_l1732_173288

noncomputable def f (x : ℝ) : ℝ := x ^ 2
noncomputable def g (x : ℝ) : ℝ := -x ^ 3 + 5 * x - 3

variable (x : ℝ)

theorem quadratic_cubic_inequalities (h : 0 < x) : 
  (f x ≥ 2 * x - 1) ∧ (g x ≤ 2 * x - 1) := 
sorry

end quadratic_cubic_inequalities_l1732_173288


namespace intersection_M_N_l1732_173240

-- Definitions of sets M and N
def M : Set ℝ := {-1, 0, 1, 2}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- Proof statement showing the intersection of M and N
theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end intersection_M_N_l1732_173240


namespace parabola_properties_l1732_173279

theorem parabola_properties (a b c: ℝ) (ha : a ≠ 0) (hc : c > 1) (h1 : 4 * a + 2 * b + c = 0) (h2 : -b / (2 * a) = 1/2):
  a * b * c < 0 ∧ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = a ∧ a * x2^2 + b * x2 + c = a) ∧ a < -1/2 :=
by {
    sorry
}

end parabola_properties_l1732_173279


namespace sum_fractions_correct_l1732_173277

def sum_of_fractions (f1 f2 f3 f4 f5 f6 f7 : ℚ) : ℚ :=
  f1 + f2 + f3 + f4 + f5 + f6 + f7

theorem sum_fractions_correct : sum_of_fractions (1/3) (1/2) (-5/6) (1/5) (1/4) (-9/20) (-5/6) = -5/6 :=
by
  sorry

end sum_fractions_correct_l1732_173277


namespace sugar_per_bar_l1732_173272

theorem sugar_per_bar (bars_per_minute : ℕ) (sugar_per_2_minutes : ℕ)
  (h1 : bars_per_minute = 36)
  (h2 : sugar_per_2_minutes = 108) :
  (sugar_per_2_minutes / (bars_per_minute * 2) : ℚ) = 1.5 := 
by 
  sorry

end sugar_per_bar_l1732_173272


namespace union_A_B_intersection_A_complement_B_l1732_173256

def setA (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 2
def setB (x : ℝ) : Prop := x * (x - 4) ≤ 0

theorem union_A_B : {x : ℝ | setA x} ∪ {x : ℝ | setB x} = {x : ℝ | -1 ≤ x ∧ x ≤ 4} :=
by
  sorry

theorem intersection_A_complement_B : {x : ℝ | setA x} ∩ {x : ℝ | ¬ setB x} = {x : ℝ | -1 ≤ x ∧ x < 0} :=
by
  sorry

end union_A_B_intersection_A_complement_B_l1732_173256


namespace license_plate_count_l1732_173226

def num_license_plates : Nat :=
  26 * 10 * 36

theorem license_plate_count : num_license_plates = 9360 :=
by
  sorry

end license_plate_count_l1732_173226


namespace Monica_books_next_year_l1732_173223

-- Definitions for conditions
def books_last_year : ℕ := 25
def books_this_year (bl_year: ℕ) : ℕ := 3 * bl_year
def books_next_year (bt_year: ℕ) : ℕ := 3 * bt_year + 7

-- Theorem statement
theorem Monica_books_next_year : books_next_year (books_this_year books_last_year) = 232 :=
by
  sorry

end Monica_books_next_year_l1732_173223


namespace g_at_neg_two_is_fifteen_l1732_173251

def g (x : ℤ) : ℤ := 2 * x^2 - 3 * x + 1

theorem g_at_neg_two_is_fifteen : g (-2) = 15 :=
by 
  -- proof is skipped
  sorry

end g_at_neg_two_is_fifteen_l1732_173251


namespace geometric_sequence_root_product_l1732_173255

theorem geometric_sequence_root_product
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * r)
  (a1_pos : 0 < a 1)
  (a19_root : a 1 * r^18 = (1 : ℝ))
  (h_poly : ∀ x, x^2 - 10 * x + 16 = 0) :
  a 8 * a 12 = 16  :=
sorry

end geometric_sequence_root_product_l1732_173255


namespace limit_leq_l1732_173207

variables {α : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]

theorem limit_leq {a_n b_n : ℕ → α} {a b : α}
  (ha : Filter.Tendsto a_n Filter.atTop (nhds a))
  (hb : Filter.Tendsto b_n Filter.atTop (nhds b))
  (h_leq : ∀ n, a_n n ≤ b_n n)
  : a ≤ b :=
by
  -- Proof will be constructed here
  sorry

end limit_leq_l1732_173207


namespace circle_inscribed_angles_l1732_173212

theorem circle_inscribed_angles (O : Type) (circle : Set O) (A B C D E F G H I J K L : O) 
  (P : ℕ) (n : ℕ) (x_deg_sum y_deg_sum : ℝ)  
  (h1 : n = 12) 
  (h2 : x_deg_sum = 45) 
  (h3 : y_deg_sum = 75) :
  x_deg_sum + y_deg_sum = 120 :=
by
  /- Proof steps are not required -/
  apply sorry

end circle_inscribed_angles_l1732_173212


namespace mass_percentage_of_O_in_dichromate_l1732_173298

noncomputable def molar_mass_Cr : ℝ := 52.00
noncomputable def molar_mass_O : ℝ := 16.00
noncomputable def molar_mass_Cr2O7_2_minus : ℝ := (2 * molar_mass_Cr) + (7 * molar_mass_O)

theorem mass_percentage_of_O_in_dichromate :
  (7 * molar_mass_O / molar_mass_Cr2O7_2_minus) * 100 = 51.85 := 
by
  sorry

end mass_percentage_of_O_in_dichromate_l1732_173298


namespace number_of_squares_l1732_173242

theorem number_of_squares (total_streetlights squares_streetlights unused_streetlights : ℕ) 
  (h1 : total_streetlights = 200) 
  (h2 : squares_streetlights = 12) 
  (h3 : unused_streetlights = 20) : 
  (∃ S : ℕ, total_streetlights = squares_streetlights * S + unused_streetlights ∧ S = 15) :=
by
  sorry

end number_of_squares_l1732_173242


namespace sum_of_radii_l1732_173241

noncomputable def tangency_equation (r : ℝ) : Prop :=
  (r - 5)^2 + r^2 = (r + 1.5)^2

theorem sum_of_radii : ∀ (r1 r2 : ℝ), tangency_equation r1 ∧ tangency_equation r2 →
  r1 + r2 = 13 :=
by
  intros r1 r2 h
  sorry

end sum_of_radii_l1732_173241


namespace quadratic_no_real_roots_l1732_173231

theorem quadratic_no_real_roots 
  (p q a b c : ℝ) 
  (h1 : 0 < p) (h2 : 0 < q) (h3 : 0 < a) (h4 : 0 < b) (h5 : 0 < c)
  (h6 : p ≠ q)
  (h7 : a^2 = p * q)
  (h8 : b + c = p + q)
  (h9 : b = (2 * p + q) / 3)
  (h10 : c = (p + 2 * q) / 3) :
  (∀ x : ℝ, ¬ (b * x^2 - 2 * a * x + c = 0)) := 
by
  sorry

end quadratic_no_real_roots_l1732_173231


namespace area_of_quadrilateral_l1732_173299

theorem area_of_quadrilateral (d a b : ℝ) (h₀ : d = 28) (h₁ : a = 9) (h₂ : b = 6) :
  (1 / 2 * d * a) + (1 / 2 * d * b) = 210 :=
by
  -- Provided proof steps are skipped
  sorry

end area_of_quadrilateral_l1732_173299


namespace school_profit_calc_l1732_173262

-- Definitions based on the conditions provided
def pizza_slices : Nat := 8
def slices_per_pizza : ℕ := 8
def slice_price : ℝ := 1.0 -- Defining price per slice
def pizzas_bought : ℕ := 55
def cost_per_pizza : ℝ := 6.85
def total_revenue : ℝ := pizzas_bought * slices_per_pizza * slice_price
def total_cost : ℝ := pizzas_bought * cost_per_pizza

-- The lean mathematical statement we need to prove
theorem school_profit_calc :
  total_revenue - total_cost = 63.25 := by
  sorry

end school_profit_calc_l1732_173262


namespace probability_neither_l1732_173271

variable (P : Set ℕ → ℝ) -- Use ℕ as a placeholder for the event space
variables (A B : Set ℕ)
variables (hA : P A = 0.25) (hB : P B = 0.35) (hAB : P (A ∩ B) = 0.15)

theorem probability_neither :
  P (Aᶜ ∩ Bᶜ) = 0.55 :=
by
  sorry

end probability_neither_l1732_173271


namespace minimum_cost_to_buy_additional_sheets_l1732_173275

def total_sheets : ℕ := 98
def students : ℕ := 12
def cost_per_sheet : ℕ := 450

theorem minimum_cost_to_buy_additional_sheets : 
  (students * (1 + total_sheets / students) - total_sheets) * cost_per_sheet = 4500 :=
by {
  sorry
}

end minimum_cost_to_buy_additional_sheets_l1732_173275


namespace length_of_platform_l1732_173213

theorem length_of_platform {train_length : ℕ} {time_to_cross_pole : ℕ} {time_to_cross_platform : ℕ} 
  (h1 : train_length = 300) 
  (h2 : time_to_cross_pole = 18) 
  (h3 : time_to_cross_platform = 45) : 
  ∃ platform_length : ℕ, platform_length = 450 :=
by
  sorry

end length_of_platform_l1732_173213


namespace man_profit_doubled_l1732_173257

noncomputable def percentage_profit (C SP1 SP2 : ℝ) : ℝ :=
  (SP2 - C) / C * 100

theorem man_profit_doubled (C SP1 SP2 : ℝ) (h1 : SP1 = 1.30 * C) (h2 : SP2 = 2 * SP1) :
  percentage_profit C SP1 SP2 = 160 := by
  sorry

end man_profit_doubled_l1732_173257


namespace more_than_half_millet_on_day_three_l1732_173210

-- Definition of the initial conditions
def seeds_in_feeder (n: ℕ) : ℝ :=
  1 + n

def millet_amount (n: ℕ) : ℝ :=
  0.6 * (1 - (0.5)^n)

-- The theorem we want to prove
theorem more_than_half_millet_on_day_three :
  ∀ n, n = 3 → (millet_amount n) / (seeds_in_feeder n) > 0.5 :=
by
  intros n hn
  rw [hn, seeds_in_feeder, millet_amount]
  sorry

end more_than_half_millet_on_day_three_l1732_173210


namespace sin_cos_identity_l1732_173209

theorem sin_cos_identity : (Real.sin (65 * Real.pi / 180) * Real.cos (35 * Real.pi / 180) 
  - Real.cos (65 * Real.pi / 180) * Real.sin (35 * Real.pi / 180)) = 1 / 2 :=
by
  sorry

end sin_cos_identity_l1732_173209


namespace area_of_shaded_region_l1732_173267

noncomputable def r2 : ℝ := Real.sqrt 20
noncomputable def r1 : ℝ := 3 * r2

theorem area_of_shaded_region :
  let area := π * (r1 ^ 2) - π * (r2 ^ 2)
  area = 160 * π :=
by
  sorry

end area_of_shaded_region_l1732_173267


namespace find_min_difference_l1732_173287

theorem find_min_difference (p q : ℤ) (hp : 0 < p) (hq : 0 < q)
  (h₁ : 3 * q < 5 * p)
  (h₂ : 8 * p < 5 * q)
  (h₃ : ∀ r s : ℤ, 0 < s → (3 * s < 5 * r ∧ 8 * r < 5 * s) → q ≤ s) :
  q - p = 5 :=
sorry

end find_min_difference_l1732_173287


namespace parabola_directrix_eq_l1732_173252

theorem parabola_directrix_eq (x y : ℝ) : x^2 + 12 * y = 0 → y = 3 := 
by sorry

end parabola_directrix_eq_l1732_173252


namespace parallelogram_circumference_l1732_173200

-- Defining the conditions
def isParallelogram (a b : ℕ) := a = 18 ∧ b = 12

-- The theorem statement to prove
theorem parallelogram_circumference (a b : ℕ) (h : isParallelogram a b) : 2 * (a + b) = 60 :=
  by
  -- Extract the conditions from hypothesis
  cases h with
  | intro hab' hab'' =>
    sorry

end parallelogram_circumference_l1732_173200


namespace cubic_polynomial_solution_l1732_173227

noncomputable def q (x : ℚ) : ℚ := (51/13) * x^3 + (-31/13) * x^2 + (16/13) * x + (3/13)

theorem cubic_polynomial_solution : 
  q 1 = 3 ∧ q 2 = 23 ∧ q 3 = 81 ∧ q 5 = 399 :=
by {
  sorry
}

end cubic_polynomial_solution_l1732_173227


namespace max_band_members_l1732_173276

theorem max_band_members (k n m : ℕ) : m = k^2 + 11 → m = n * (n + 9) → m ≤ 112 :=
by
  sorry

end max_band_members_l1732_173276


namespace sqrt_10_plus_3_pow_2023_mul_sqrt_10_minus_3_pow_2022_l1732_173281

theorem sqrt_10_plus_3_pow_2023_mul_sqrt_10_minus_3_pow_2022:
  ( (Real.sqrt 10 + 3) ^ 2023 * (Real.sqrt 10 - 3) ^ 2022 = Real.sqrt 10 + 3 ) :=
by {
  sorry
}

end sqrt_10_plus_3_pow_2023_mul_sqrt_10_minus_3_pow_2022_l1732_173281


namespace least_number_to_add_for_divisibility_by_nine_l1732_173219

theorem least_number_to_add_for_divisibility_by_nine : ∃ x : ℕ, (4499 + x) % 9 = 0 ∧ x = 1 :=
by
  sorry

end least_number_to_add_for_divisibility_by_nine_l1732_173219


namespace hyperbola_asymptotes_and_parabola_l1732_173221

-- Definitions for hyperbola and parabola
noncomputable def hyperbola (x y : ℝ) : Prop := (x^2 / 9) - (y^2 / 16) = 1
noncomputable def focus_of_hyperbola (focus : ℝ × ℝ) : Prop := focus = (5, 0)
noncomputable def asymptote_of_hyperbola (y x : ℝ) : Prop := y = (4 / 3) * x ∨ y = - (4 / 3) * x
noncomputable def parabola (y x p : ℝ) : Prop := y^2 = 2 * p * x

-- Main statement
theorem hyperbola_asymptotes_and_parabola :
  (∀ x y, hyperbola x y → asymptote_of_hyperbola y x) ∧
  (∀ y x, focus_of_hyperbola (5, 0) → parabola y x 10) :=
by
  -- To be proved
  sorry

end hyperbola_asymptotes_and_parabola_l1732_173221


namespace cos_seven_pi_over_six_l1732_173258

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  -- Place the proof here
  sorry

end cos_seven_pi_over_six_l1732_173258


namespace plane_coloring_l1732_173230

-- Define a type for colors to represent red and blue
inductive Color
| red
| blue

-- The main statement
theorem plane_coloring (x : ℝ) (h_pos : 0 < x) (coloring : ℝ × ℝ → Color) :
  ∃ (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ coloring p1 = coloring p2 ∧ dist p1 p2 = x :=
sorry

end plane_coloring_l1732_173230
