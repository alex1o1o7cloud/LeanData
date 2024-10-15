import Mathlib

namespace NUMINAMATH_GPT_no_solution_m_l1972_197279

noncomputable def fractional_eq (x m : ℝ) : Prop :=
  2 / (x - 2) + m * x / (x^2 - 4) = 3 / (x + 2)

theorem no_solution_m (m : ℝ) : 
  (¬ ∃ x, fractional_eq x m) ↔ (m = -4 ∨ m = 6 ∨ m = 1) :=
sorry

end NUMINAMATH_GPT_no_solution_m_l1972_197279


namespace NUMINAMATH_GPT_bianca_total_bags_l1972_197232

theorem bianca_total_bags (bags_recycled_points : ℕ) (bags_not_recycled : ℕ) (total_points : ℕ) (total_bags : ℕ) 
  (h1 : bags_recycled_points = 5) 
  (h2 : bags_not_recycled = 8) 
  (h3 : total_points = 45) 
  (recycled_bags := total_points / bags_recycled_points) :
  total_bags = recycled_bags + bags_not_recycled := 
by 
  sorry

end NUMINAMATH_GPT_bianca_total_bags_l1972_197232


namespace NUMINAMATH_GPT_problem_statement_l1972_197292

-- Define the sides of the original triangle
def side_5 := 5
def side_12 := 12
def side_13 := 13

-- Define the perimeters of the isosceles triangles
def P := 3 * side_5
def Q := 3 * side_12
def R := 3 * side_13

-- Statement we want to prove
theorem problem_statement : P + R = (3 / 2) * Q := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1972_197292


namespace NUMINAMATH_GPT_not_exists_k_eq_one_l1972_197284

theorem not_exists_k_eq_one (k : ℝ) : (∃ x y : ℝ, y = k * x + 2 ∧ y = (3 * k - 2) * x + 5) ↔ k ≠ 1 :=
by sorry

end NUMINAMATH_GPT_not_exists_k_eq_one_l1972_197284


namespace NUMINAMATH_GPT_parallel_lines_a_l1972_197251
-- Import necessary libraries

-- Define the given conditions and the main statement
theorem parallel_lines_a (a : ℝ) :
  (∀ x y : ℝ, a * x + y - 2 = 0 → 3 * x + (a + 2) * y + 1 = 0) →
  (a = -3 ∨ a = 1) :=
by
  -- Place the proof here
  sorry

end NUMINAMATH_GPT_parallel_lines_a_l1972_197251


namespace NUMINAMATH_GPT_Zhang_Laoshi_pens_l1972_197241

theorem Zhang_Laoshi_pens (x : ℕ) (original_price new_price : ℝ)
  (discount : new_price = 0.75 * original_price)
  (more_pens : x * original_price = (x + 25) * new_price) :
  x = 75 :=
by
  sorry

end NUMINAMATH_GPT_Zhang_Laoshi_pens_l1972_197241


namespace NUMINAMATH_GPT_repeating_decimal_as_fraction_l1972_197289

theorem repeating_decimal_as_fraction :
  (0.58207 : ℝ) = 523864865 / 999900 := sorry

end NUMINAMATH_GPT_repeating_decimal_as_fraction_l1972_197289


namespace NUMINAMATH_GPT_problem_xy_squared_and_product_l1972_197258

theorem problem_xy_squared_and_product (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) :
  x^2 - y^2 = 80 ∧ x * y = 96 :=
by
  sorry

end NUMINAMATH_GPT_problem_xy_squared_and_product_l1972_197258


namespace NUMINAMATH_GPT_calculate_ratio_l1972_197291

variables (M Q P N R : ℝ)

-- Definitions of conditions
def M_def : M = 0.40 * Q := by sorry
def Q_def : Q = 0.30 * P := by sorry
def N_def : N = 0.60 * P := by sorry
def R_def : R = 0.20 * P := by sorry

-- Statement of the proof problem
theorem calculate_ratio (hM : M = 0.40 * Q) (hQ : Q = 0.30 * P)
  (hN : N = 0.60 * P) (hR : R = 0.20 * P) : 
  (M + R) / N = 8 / 15 := by
  sorry

end NUMINAMATH_GPT_calculate_ratio_l1972_197291


namespace NUMINAMATH_GPT_difference_seven_three_times_l1972_197250

theorem difference_seven_three_times (n : ℝ) (h1 : n = 3) 
  (h2 : 7 * n = 3 * n + (21.0 - 9.0)) :
  7 * n - 3 * n = 12.0 := by
  sorry

end NUMINAMATH_GPT_difference_seven_three_times_l1972_197250


namespace NUMINAMATH_GPT_intersection_A_B_solution_inequalities_l1972_197256

def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | -3 < x ∧ x < 2}
def C : Set ℝ := {x | -1 < x ∧ x < 2}

theorem intersection_A_B :
  A ∩ B = C :=
by
  sorry

theorem solution_inequalities (x : ℝ) :
  (2 * x^2 + x - 1 > 0) ↔ (x < -1 ∨ x > 1/2) :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_solution_inequalities_l1972_197256


namespace NUMINAMATH_GPT_number_of_boats_l1972_197248

theorem number_of_boats (total_people : ℕ) (people_per_boat : ℕ)
  (h1 : total_people = 15) (h2 : people_per_boat = 3) : total_people / people_per_boat = 5 :=
by {
  -- proof steps here
  sorry
}

end NUMINAMATH_GPT_number_of_boats_l1972_197248


namespace NUMINAMATH_GPT_train_length_l1972_197290

theorem train_length (speed_kmph : ℤ) (time_sec : ℤ) (expected_length_m : ℤ) 
    (speed_kmph_eq : speed_kmph = 72)
    (time_sec_eq : time_sec = 7)
    (expected_length_eq : expected_length_m = 140) :
    expected_length_m = (speed_kmph * 1000 / 3600) * time_sec :=
by 
    sorry

end NUMINAMATH_GPT_train_length_l1972_197290


namespace NUMINAMATH_GPT_probability_range_l1972_197277

noncomputable def probability_distribution (K : ℕ) : ℝ :=
  if K > 0 then 1 / (2^K) else 0

theorem probability_range (h2 : 2 < 3) (h3 : 3 ≤ 4) :
  probability_distribution 3 + probability_distribution 4 = 3 / 16 :=
by
  sorry

end NUMINAMATH_GPT_probability_range_l1972_197277


namespace NUMINAMATH_GPT_total_height_of_buildings_l1972_197230

noncomputable def tallest_building := 100
noncomputable def second_tallest_building := tallest_building / 2
noncomputable def third_tallest_building := second_tallest_building / 2
noncomputable def fourth_tallest_building := third_tallest_building / 5

theorem total_height_of_buildings : 
  (tallest_building + second_tallest_building + third_tallest_building + fourth_tallest_building) = 180 := by
  sorry

end NUMINAMATH_GPT_total_height_of_buildings_l1972_197230


namespace NUMINAMATH_GPT_parallelogram_base_length_l1972_197219

variable (base height : ℝ)
variable (Area : ℝ)

theorem parallelogram_base_length (h₁ : Area = 162) (h₂ : height = 2 * base) (h₃ : Area = base * height) : base = 9 := 
by
  sorry

end NUMINAMATH_GPT_parallelogram_base_length_l1972_197219


namespace NUMINAMATH_GPT_geoff_additional_votes_needed_l1972_197259

-- Define the given conditions
def totalVotes : ℕ := 6000
def geoffPercentage : ℕ := 5 -- Represent 0.5% as 5 out of 1000 for better integer computation
def requiredPercentage : ℕ := 505 -- Represent 50.5% as 505 out of 1000 for better integer computation

-- Define the expressions for the number of votes received by Geoff and the votes required to win
def geoffVotes := (geoffPercentage * totalVotes) / 1000
def requiredVotes := (requiredPercentage * totalVotes) / 1000 + 1

-- The proposition to prove the additional number of votes needed for Geoff to win
theorem geoff_additional_votes_needed : requiredVotes - geoffVotes = 3001 := by sorry

end NUMINAMATH_GPT_geoff_additional_votes_needed_l1972_197259


namespace NUMINAMATH_GPT_vector_addition_scalar_multiplication_l1972_197263

def u : ℝ × ℝ × ℝ := (3, -2, 5)
def v : ℝ × ℝ × ℝ := (-1, 6, -3)
def result : ℝ × ℝ × ℝ := (4, 8, 4)

theorem vector_addition_scalar_multiplication :
  2 • (u + v) = result :=
by
  sorry

end NUMINAMATH_GPT_vector_addition_scalar_multiplication_l1972_197263


namespace NUMINAMATH_GPT_total_jumps_l1972_197220

theorem total_jumps (hattie_1 : ℕ) (lorelei_1 : ℕ) (hattie_2 : ℕ) (lorelei_2 : ℕ) (hattie_3 : ℕ) (lorelei_3 : ℕ) :
  hattie_1 = 180 →
  lorelei_1 = 3 / 4 * hattie_1 →
  hattie_2 = 2 / 3 * hattie_1 →
  lorelei_2 = hattie_2 + 50 →
  hattie_3 = hattie_2 + 1 / 3 * hattie_2 →
  lorelei_3 = 4 / 5 * lorelei_1 →
  hattie_1 + hattie_2 + hattie_3 + lorelei_1 + lorelei_2 + lorelei_3 = 873 :=
by
  intros h1 l1 h2 l2 h3 l3
  sorry

end NUMINAMATH_GPT_total_jumps_l1972_197220


namespace NUMINAMATH_GPT_task2_probability_l1972_197218

variable (P_task1_on_time P_task2_on_time : ℝ)

theorem task2_probability 
  (h1 : P_task1_on_time = 5 / 8)
  (h2 : (P_task1_on_time * (1 - P_task2_on_time)) = 0.25) :
  P_task2_on_time = 3 / 5 := by
  sorry

end NUMINAMATH_GPT_task2_probability_l1972_197218


namespace NUMINAMATH_GPT_root_abs_sum_l1972_197228

-- Definitions and conditions
variable (p q r n : ℤ)
variable (h_root : (x^3 - 2018 * x + n).coeffs[0] = 0)  -- This needs coefficient definition (simplified for clarity)
variable (h_vieta1 : p + q + r = 0)
variable (h_vieta2 : p * q + q * r + r * p = -2018)

theorem root_abs_sum :
  |p| + |q| + |r| = 100 :=
sorry

end NUMINAMATH_GPT_root_abs_sum_l1972_197228


namespace NUMINAMATH_GPT_find_a5_l1972_197274

theorem find_a5 (a : ℕ → ℤ)
  (h1 : ∀ n : ℕ, n > 0 → a (n + 1) = a n - 1) 
  (h2 : a 2 + a 4 + a 6 = 18) : 
  a 5 = 5 :=
sorry

end NUMINAMATH_GPT_find_a5_l1972_197274


namespace NUMINAMATH_GPT_tangent_line_through_P_is_correct_l1972_197240

-- Define the circle and the point
def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 3) ^ 2 = 25
def pointP : ℝ × ℝ := (-1, 7)

-- Define the equation of the tangent line
def tangent_line (x y : ℝ) : Prop := 3 * x - 4 * y + 31 = 0

-- State the theorem
theorem tangent_line_through_P_is_correct :
  (circle_eq (-1) 7) → 
  (tangent_line (-1) 7) :=
sorry

end NUMINAMATH_GPT_tangent_line_through_P_is_correct_l1972_197240


namespace NUMINAMATH_GPT_flight_time_NY_to_CT_l1972_197223

def travelTime (start_time_NY : ℕ) (end_time_CT : ℕ) (layover_Johannesburg : ℕ) : ℕ :=
  end_time_CT - start_time_NY + layover_Johannesburg

theorem flight_time_NY_to_CT :
  let start_time_NY := 0 -- 12:00 a.m. Tuesday as 0 hours from midnight in ET
  let end_time_CT := 10  -- 10:00 a.m. Tuesday as 10 hours from midnight in ET
  let layover_Johannesburg := 4
  travelTime start_time_NY end_time_CT layover_Johannesburg = 10 :=
by
  sorry

end NUMINAMATH_GPT_flight_time_NY_to_CT_l1972_197223


namespace NUMINAMATH_GPT_E_72_eq_9_l1972_197271

def E (n : ℕ) : ℕ :=
  -- Assume a function definition counting representations
  -- (this function body is a placeholder, as the exact implementation
  -- is not part of the problem statement)
  sorry

theorem E_72_eq_9 :
  E 72 = 9 :=
sorry

end NUMINAMATH_GPT_E_72_eq_9_l1972_197271


namespace NUMINAMATH_GPT_tony_rope_length_l1972_197203

-- Definition of the initial rope lengths and the number of ropes
def ropes : List ℝ := [8, 20, 2, 2, 2, 7]

-- Definition of the length lost per knot
def loss_per_knot : ℝ := 1.2

-- The expected total length of the rope after tying them all together
def total_length_after_knots : ℝ := 35

theorem tony_rope_length : 
  (ropes.sum - (ropes.length - 1) * loss_per_knot) = total_length_after_knots := 
by
  -- define sorry to skip the proof
  sorry

end NUMINAMATH_GPT_tony_rope_length_l1972_197203


namespace NUMINAMATH_GPT_combined_average_age_l1972_197266

noncomputable def roomA : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
noncomputable def roomB : Set ℕ := {11, 12, 13, 14}
noncomputable def average_age_A := 55
noncomputable def average_age_B := 35
noncomputable def total_people := (10 + 4)
noncomputable def total_age_A := 10 * average_age_A
noncomputable def total_age_B := 4 * average_age_B
noncomputable def combined_total_age := total_age_A + total_age_B

theorem combined_average_age :
  (combined_total_age / total_people : ℚ) = 49.29 :=
by sorry

end NUMINAMATH_GPT_combined_average_age_l1972_197266


namespace NUMINAMATH_GPT_det_example_1_simplified_form_det_at_4_l1972_197236

-- Definition for second-order determinant
def second_order_determinant (a b c d : ℤ) : ℤ :=
  a * d - b * c

-- Part (1)
theorem det_example_1 :
  second_order_determinant 3 (-2) 4 (-3) = -1 :=
by
  sorry

-- Part (2) simplified determinant
def simplified_det (x : ℤ) : ℤ :=
  second_order_determinant (2 * x - 3) (x + 2) 2 4

-- Proving simplified determinant form
theorem simplified_form :
  ∀ x : ℤ, simplified_det x = 6 * x - 16 :=
by
  sorry

-- Proving specific case when x = 4
theorem det_at_4 :
  simplified_det 4 = 8 :=
by 
  sorry

end NUMINAMATH_GPT_det_example_1_simplified_form_det_at_4_l1972_197236


namespace NUMINAMATH_GPT_benny_spending_l1972_197286

variable (S D V : ℝ)

theorem benny_spending :
  (200 - 45) = S + (D / 110) + (V / 0.75) :=
by
  sorry

end NUMINAMATH_GPT_benny_spending_l1972_197286


namespace NUMINAMATH_GPT_score_not_possible_l1972_197272

theorem score_not_possible (c u i : ℕ) (score : ℤ) :
  c + u + i = 25 ∧ score = 79 → score ≠ 5 * c + 3 * u - 25 := by
  intro h
  sorry

end NUMINAMATH_GPT_score_not_possible_l1972_197272


namespace NUMINAMATH_GPT_triangular_stack_log_count_l1972_197243

theorem triangular_stack_log_count : 
  ∀ (a₁ aₙ d : ℤ) (n : ℤ), a₁ = 15 → aₙ = 1 → d = -2 → 
  (a₁ - aₙ) / (-d) + 1 = n → 
  (n * (a₁ + aₙ)) / 2 = 64 :=
by
  intros a₁ aₙ d n h₁ hₙ hd hn
  sorry

end NUMINAMATH_GPT_triangular_stack_log_count_l1972_197243


namespace NUMINAMATH_GPT_not_possible_odd_sum_l1972_197246

theorem not_possible_odd_sum (m n : ℤ) (h : (m ^ 2 + n ^ 2) % 2 = 0) : (m + n) % 2 ≠ 1 :=
sorry

end NUMINAMATH_GPT_not_possible_odd_sum_l1972_197246


namespace NUMINAMATH_GPT_line_through_two_points_line_with_intercept_sum_l1972_197202

theorem line_through_two_points (a b x1 y1 x2 y2: ℝ) : 
  (x1 = 2) → (y1 = 1) → (x2 = 0) → (y2 = -3) → (2 * x - y - 3 = 0) :=
by
                
  sorry

theorem line_with_intercept_sum (a b : ℝ) (x y : ℝ) :
  (x = 0) → (y = 5) → (a + b = 2) → (b = 5) → (5 * x - 3 * y + 15 = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_through_two_points_line_with_intercept_sum_l1972_197202


namespace NUMINAMATH_GPT_find_number_l1972_197278

theorem find_number (x : ℕ) (h : x + 3 * x = 20) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1972_197278


namespace NUMINAMATH_GPT_difference_is_1343_l1972_197237

-- Define the larger number L and the relationship with the smaller number S.
def L : ℕ := 1608
def quotient : ℕ := 6
def remainder : ℕ := 15

-- Define the relationship: L = 6S + 15
def relationship (S : ℕ) : Prop := L = quotient * S + remainder

-- The theorem we want to prove: The difference between the larger and smaller number is 1343
theorem difference_is_1343 (S : ℕ) (h_rel : relationship S) : L - S = 1343 :=
by
  sorry

end NUMINAMATH_GPT_difference_is_1343_l1972_197237


namespace NUMINAMATH_GPT_intersection_M_N_l1972_197205

variable (x : ℝ)

def M := {x : ℝ | -2 < x ∧ x < 2}
def N := {x : ℝ | x > 1}

theorem intersection_M_N :
  {x | x ∈ M ∧ x ∈ N} = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1972_197205


namespace NUMINAMATH_GPT_river_width_after_30_seconds_l1972_197270

noncomputable def width_of_river (initial_width : ℝ) (width_increase_rate : ℝ) (rowing_rate : ℝ) (time_taken : ℝ) : ℝ :=
  initial_width + (time_taken * rowing_rate * (width_increase_rate / 10))

theorem river_width_after_30_seconds :
  width_of_river 50 2 5 30 = 80 :=
by
  -- it suffices to check the calculations here
  sorry

end NUMINAMATH_GPT_river_width_after_30_seconds_l1972_197270


namespace NUMINAMATH_GPT_smallest_palindrome_base2_base4_l1972_197264

def is_palindrome_base (n : ℕ) (b : ℕ) : Prop :=
  let digits := (Nat.digits b n)
  digits = digits.reverse

theorem smallest_palindrome_base2_base4 : 
  ∃ (x : ℕ), x > 15 ∧ is_palindrome_base x 2 ∧ is_palindrome_base x 4 ∧ x = 17 :=
by
  sorry

end NUMINAMATH_GPT_smallest_palindrome_base2_base4_l1972_197264


namespace NUMINAMATH_GPT_man_age_twice_son_age_in_n_years_l1972_197201

theorem man_age_twice_son_age_in_n_years
  (S M Y : ℤ)
  (h1 : S = 26)
  (h2 : M = S + 28)
  (h3 : M + Y = 2 * (S + Y)) :
  Y = 2 :=
by
  sorry

end NUMINAMATH_GPT_man_age_twice_son_age_in_n_years_l1972_197201


namespace NUMINAMATH_GPT_most_probable_light_is_green_l1972_197297

def duration_red := 30
def duration_yellow := 5
def duration_green := 40
def total_duration := duration_red + duration_yellow + duration_green

def prob_red := duration_red / total_duration
def prob_yellow := duration_yellow / total_duration
def prob_green := duration_green / total_duration

theorem most_probable_light_is_green : prob_green > prob_red ∧ prob_green > prob_yellow := 
  by
  sorry

end NUMINAMATH_GPT_most_probable_light_is_green_l1972_197297


namespace NUMINAMATH_GPT_gasoline_price_increase_l1972_197268

theorem gasoline_price_increase :
  let P_initial := 29.90
  let P_final := 149.70
  (P_final - P_initial) / P_initial * 100 = 400 :=
by
  let P_initial := 29.90
  let P_final := 149.70
  sorry

end NUMINAMATH_GPT_gasoline_price_increase_l1972_197268


namespace NUMINAMATH_GPT_Li_age_is_12_l1972_197293

-- Given conditions:
def Zhang_twice_Li (Li: ℕ) : ℕ := 2 * Li
def Jung_older_Zhang (Zhang: ℕ) : ℕ := Zhang + 2
def Jung_age := 26

-- Proof problem:
theorem Li_age_is_12 : ∃ Li: ℕ, Jung_older_Zhang (Zhang_twice_Li Li) = Jung_age ∧ Li = 12 :=
by
  sorry

end NUMINAMATH_GPT_Li_age_is_12_l1972_197293


namespace NUMINAMATH_GPT_find_integer_n_l1972_197255

theorem find_integer_n (a b : ℕ) (n : ℕ)
  (h1 : n = 2^a * 3^b)
  (h2 : (2^(a+1) - 1) * ((3^(b+1) - 1) / (3 - 1)) = 1815) : n = 648 :=
  sorry

end NUMINAMATH_GPT_find_integer_n_l1972_197255


namespace NUMINAMATH_GPT_farmer_land_l1972_197227

variable (T : ℝ) -- Total land owned by the farmer

def is_cleared (T : ℝ) : ℝ := 0.90 * T
def cleared_barley (T : ℝ) : ℝ := 0.80 * is_cleared T
def cleared_potato (T : ℝ) : ℝ := 0.10 * is_cleared T
def cleared_tomato : ℝ := 90
def cleared_land (T : ℝ) : ℝ := cleared_barley T + cleared_potato T + cleared_tomato

theorem farmer_land (T : ℝ) (h : cleared_land T = is_cleared T) : T = 1000 := sorry

end NUMINAMATH_GPT_farmer_land_l1972_197227


namespace NUMINAMATH_GPT_danny_steve_ratio_l1972_197221

theorem danny_steve_ratio :
  ∀ (D S : ℝ),
  D = 29 →
  2 * (S / 2 - D / 2) = 29 →
  D / S = 1 / 2 :=
by
  intros D S hD h_eq
  sorry

end NUMINAMATH_GPT_danny_steve_ratio_l1972_197221


namespace NUMINAMATH_GPT_cricket_problem_l1972_197288

theorem cricket_problem
  (x : ℕ)
  (run_rate_initial : ℝ := 3.8)
  (overs_remaining : ℕ := 40)
  (run_rate_remaining : ℝ := 6.1)
  (target_runs : ℕ := 282) :
  run_rate_initial * x + run_rate_remaining * overs_remaining = target_runs → x = 10 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_cricket_problem_l1972_197288


namespace NUMINAMATH_GPT_nonnegative_exists_l1972_197245

theorem nonnegative_exists (a b c : ℝ) (h : a + b + c = 0) : a ≥ 0 ∨ b ≥ 0 ∨ c ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_nonnegative_exists_l1972_197245


namespace NUMINAMATH_GPT_geese_left_park_l1972_197208

noncomputable def initial_ducks : ℕ := 25
noncomputable def initial_geese (ducks : ℕ) : ℕ := 2 * ducks - 10
noncomputable def final_ducks (ducks_added : ℕ) (ducks : ℕ) : ℕ := ducks + ducks_added
noncomputable def geese_after_leaving (geese_before : ℕ) (geese_left : ℕ) : ℕ := geese_before - geese_left

theorem geese_left_park
    (ducks : ℕ)
    (ducks_added : ℕ)
    (initial_geese : ℕ := 2 * ducks - 10)
    (final_ducks : ℕ := ducks + ducks_added)
    (geese_left : ℕ)
    (geese_remaining : ℕ := initial_geese - geese_left) :
    geese_remaining = final_ducks + 1 → geese_left = 10 := by
  sorry

end NUMINAMATH_GPT_geese_left_park_l1972_197208


namespace NUMINAMATH_GPT_exists_natural_n_l1972_197210

theorem exists_natural_n (a b : ℕ) (h1 : b ≥ 2) (h2 : Nat.gcd a b = 1) : ∃ n : ℕ, (n * a) % b = 1 :=
by
  sorry

end NUMINAMATH_GPT_exists_natural_n_l1972_197210


namespace NUMINAMATH_GPT_work_completion_days_l1972_197299

theorem work_completion_days (P R: ℕ) (hP: P = 80) (hR: R = 120) : P * R / (P + R) = 48 := by
  -- The proof is omitted as we are only writing the statement
  sorry

end NUMINAMATH_GPT_work_completion_days_l1972_197299


namespace NUMINAMATH_GPT_sum_of_areas_of_circles_l1972_197247

-- Definitions of the conditions given in the problem
def triangle_side1 : ℝ := 6
def triangle_side2 : ℝ := 8
def triangle_side3 : ℝ := 10

-- Definitions of the radii r, s, t
variables (r s t : ℝ)

-- Conditions derived from the problem
axiom rs_eq : r + s = triangle_side1
axiom rt_eq : r + t = triangle_side2
axiom st_eq : s + t = triangle_side3

-- Main theorem to prove
theorem sum_of_areas_of_circles : (π * r^2) + (π * s^2) + (π * t^2) = 56 * π :=
by
  sorry

end NUMINAMATH_GPT_sum_of_areas_of_circles_l1972_197247


namespace NUMINAMATH_GPT_zhou_catches_shuttle_probability_l1972_197269

-- Condition 1: Shuttle arrival time and duration
def shuttle_arrival_start : ℕ := 420 -- 7:00 AM in minutes since midnight
def shuttle_duration : ℕ := 15

-- Condition 2: Zhou's random arrival time window
def zhou_arrival_start : ℕ := 410 -- 6:50 AM in minutes since midnight
def zhou_arrival_end : ℕ := 465 -- 7:45 AM in minutes since midnight

-- Total time available for Zhou to arrive (55 minutes) 
def total_time : ℕ := zhou_arrival_end - zhou_arrival_start

-- Time window when Zhou needs to arrive to catch the shuttle (15 minutes)
def successful_time : ℕ := shuttle_arrival_start + shuttle_duration - shuttle_arrival_start

-- Calculate the probability that Zhou catches the shuttle
theorem zhou_catches_shuttle_probability : 
  (successful_time : ℚ) / total_time = 3 / 11 := 
by 
  -- We don't need the actual proof steps, just the statement
  sorry

end NUMINAMATH_GPT_zhou_catches_shuttle_probability_l1972_197269


namespace NUMINAMATH_GPT_chord_length_of_circle_and_line_intersection_l1972_197233

theorem chord_length_of_circle_and_line_intersection :
  ∀ (x y : ℝ), (x - 2 * y = 3) → ((x - 2)^2 + (y + 3)^2 = 9) → ∃ chord_length : ℝ, (chord_length = 4) :=
by
  intros x y hx hy
  sorry

end NUMINAMATH_GPT_chord_length_of_circle_and_line_intersection_l1972_197233


namespace NUMINAMATH_GPT_total_payment_is_correct_l1972_197298

def length : ℕ := 30
def width : ℕ := 40
def construction_cost_per_sqft : ℕ := 3
def sealant_cost_per_sqft : ℕ := 1
def total_area : ℕ := length * width
def total_cost_per_sqft : ℕ := construction_cost_per_sqft + sealant_cost_per_sqft
def total_cost : ℕ := total_area * total_cost_per_sqft

theorem total_payment_is_correct : total_cost = 4800 := by
  sorry

end NUMINAMATH_GPT_total_payment_is_correct_l1972_197298


namespace NUMINAMATH_GPT_total_books_written_l1972_197294

def books_written (Zig Flo : ℕ) : Prop :=
  (Zig = 60) ∧ (Zig = 4 * Flo) ∧ (Zig + Flo = 75)

theorem total_books_written (Zig Flo : ℕ) : books_written Zig Flo :=
  by
    sorry

end NUMINAMATH_GPT_total_books_written_l1972_197294


namespace NUMINAMATH_GPT_intersection_M_N_l1972_197225

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
by 
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1972_197225


namespace NUMINAMATH_GPT_ratio_depth_to_height_l1972_197209

noncomputable def height_ron : ℝ := 12
noncomputable def depth_water : ℝ := 60

theorem ratio_depth_to_height : depth_water / height_ron = 5 := by
  sorry

end NUMINAMATH_GPT_ratio_depth_to_height_l1972_197209


namespace NUMINAMATH_GPT_least_possible_value_of_smallest_integer_l1972_197213

theorem least_possible_value_of_smallest_integer :
  ∀ (A B C D : ℕ), A < B → B < C → C < D → (A + B + C + D) / 4 = 70 → D = 90 → A ≥ 13 :=
by
  intros A B C D h₁ h₂ h₃ h₄ h₅
  sorry

end NUMINAMATH_GPT_least_possible_value_of_smallest_integer_l1972_197213


namespace NUMINAMATH_GPT_mother_daughter_age_relation_l1972_197249

theorem mother_daughter_age_relation (x : ℕ) (hc1 : 43 - x = 5 * (11 - x)) : x = 3 := 
sorry

end NUMINAMATH_GPT_mother_daughter_age_relation_l1972_197249


namespace NUMINAMATH_GPT_exponent_equality_l1972_197244

theorem exponent_equality (m : ℕ) (h : 9^4 = 3^m) : m = 8 := 
  sorry

end NUMINAMATH_GPT_exponent_equality_l1972_197244


namespace NUMINAMATH_GPT_circle_condition_l1972_197214

-- Define the center of the circle
def center := ((-3 + 27) / 2, (0 + 0) / 2)

-- Define the radius of the circle
def radius := 15

-- Define the circle's equation
def circle_eq (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Define the final Lean 4 statement
theorem circle_condition (x : ℝ) : circle_eq x 12 → (x = 21 ∨ x = 3) :=
  by
  intro h
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_circle_condition_l1972_197214


namespace NUMINAMATH_GPT_sum_is_272_l1972_197253

-- Define the constant number x
def x : ℕ := 16

-- Define the sum of the number and its square
def sum_of_number_and_its_square (n : ℕ) : ℕ := n + n^2

-- State the theorem that the sum of the number and its square is 272 when the number is 16
theorem sum_is_272 : sum_of_number_and_its_square x = 272 :=
by
  sorry

end NUMINAMATH_GPT_sum_is_272_l1972_197253


namespace NUMINAMATH_GPT_remainder_of_sum_of_squares_mod_l1972_197216

-- Define the function to compute the sum of squares of the first n natural numbers
def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

-- Define the specific sum for the first 15 natural numbers
def S : ℕ := sum_of_squares 15

-- State the theorem
theorem remainder_of_sum_of_squares_mod (n : ℕ) (h : n = 15) : 
  S % 13 = 5 := by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_of_squares_mod_l1972_197216


namespace NUMINAMATH_GPT_range_of_m_l1972_197252

theorem range_of_m (a b c m : ℝ) (h1 : a > b) (h2 : b > c) (h3 : 0 < m) 
  (h4 : 1 / (a - b) + m / (b - c) ≥ 9 / (a - c)) : m ≥ 4 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1972_197252


namespace NUMINAMATH_GPT_max_b_of_box_volume_l1972_197260

theorem max_b_of_box_volume (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : Prime c) (h5 : a * b * c = 360) : b = 12 := 
sorry

end NUMINAMATH_GPT_max_b_of_box_volume_l1972_197260


namespace NUMINAMATH_GPT_exists_factor_between_10_and_20_l1972_197254

theorem exists_factor_between_10_and_20 (n : ℕ) : ∃ k, (10 ≤ k ∧ k ≤ 20) ∧ k ∣ (2^n - 1) → k = 17 :=
by
  sorry

end NUMINAMATH_GPT_exists_factor_between_10_and_20_l1972_197254


namespace NUMINAMATH_GPT_race_length_l1972_197281

variables (L : ℕ)

def distanceCondition1 := L - 70
def distanceCondition2 := L - 100
def distanceCondition3 := L - 163

theorem race_length (h1 : distanceCondition1 = L - 70) 
                    (h2 : distanceCondition2 = L - 100) 
                    (h3 : distanceCondition3 = L - 163)
                    (h4 : (L - 70) / (L - 163) = (L) / (L - 100)) : 
  L = 1000 :=
sorry

end NUMINAMATH_GPT_race_length_l1972_197281


namespace NUMINAMATH_GPT_jasonPears_l1972_197287

-- Define the conditions
def keithPears : Nat := 47
def mikePears : Nat := 12
def totalPears : Nat := 105

-- Define the theorem stating the number of pears Jason picked
theorem jasonPears : (totalPears - keithPears - mikePears) = 46 :=
by 
  sorry

end NUMINAMATH_GPT_jasonPears_l1972_197287


namespace NUMINAMATH_GPT_cistern_length_l1972_197211

theorem cistern_length (L : ℝ) (H : 0 < L) :
    (∃ (w d A : ℝ), w = 14 ∧ d = 1.25 ∧ A = 233 ∧ A = L * w + 2 * L * d + 2 * w * d) →
    L = 12 :=
by
  sorry

end NUMINAMATH_GPT_cistern_length_l1972_197211


namespace NUMINAMATH_GPT_student_correct_answers_l1972_197206

variable (C I : ℕ) -- Define C and I as natural numbers
variable (score totalQuestions : ℕ) -- Define score and totalQuestions as natural numbers

-- Define the conditions
def grading_system (C I score : ℕ) : Prop := C - 2 * I = score
def total_questions (C I totalQuestions : ℕ) : Prop := C + I = totalQuestions

-- The theorem statement to prove
theorem student_correct_answers :
  (grading_system C I 76) ∧ (total_questions C I 100) → C = 92 := by
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_student_correct_answers_l1972_197206


namespace NUMINAMATH_GPT_base_of_number_l1972_197285

theorem base_of_number (b : ℕ) : 
  (1 * b + 3)^2 = 2 * b^2 + 1 * b + 1 → b = 8 :=
by
  sorry

end NUMINAMATH_GPT_base_of_number_l1972_197285


namespace NUMINAMATH_GPT_customer_buys_two_pens_l1972_197222

def num_pens (total_pens non_defective_pens : Nat) (prob : ℚ) : Nat :=
  sorry

theorem customer_buys_two_pens :
  num_pens 16 13 0.65 = 2 :=
sorry

end NUMINAMATH_GPT_customer_buys_two_pens_l1972_197222


namespace NUMINAMATH_GPT_initial_average_weight_l1972_197226

theorem initial_average_weight (A : ℝ) (weight7th : ℝ) (new_avg_weight : ℝ) (initial_num : ℝ) (total_num : ℝ) 
  (h_weight7th : weight7th = 97) (h_new_avg_weight : new_avg_weight = 151) (h_initial_num : initial_num = 6) (h_total_num : total_num = 7) :
  initial_num * A + weight7th = total_num * new_avg_weight → A = 160 := 
by 
  intros h
  sorry

end NUMINAMATH_GPT_initial_average_weight_l1972_197226


namespace NUMINAMATH_GPT_max_principals_in_10_years_l1972_197200

theorem max_principals_in_10_years (p : ℕ) (is_principal_term : p = 4) : 
  ∃ n : ℕ, n = 4 ∧ ∀ k : ℕ, (k = 10 → n ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_max_principals_in_10_years_l1972_197200


namespace NUMINAMATH_GPT_original_price_of_cycle_l1972_197283

theorem original_price_of_cycle (SP : ℝ) (gain_percent : ℝ) (P : ℝ)
  (h_SP : SP = 1080)
  (h_gain_percent: gain_percent = 60)
  (h_relation : SP = 1.6 * P)
  : P = 675 :=
by {
  sorry
}

end NUMINAMATH_GPT_original_price_of_cycle_l1972_197283


namespace NUMINAMATH_GPT_cube_of_odd_sum_l1972_197262

theorem cube_of_odd_sum (a : ℕ) (h1 : 1 < a) (h2 : ∃ (n : ℕ), (n = (a - 1) + 2 * (a - 1) + 1) ∧ n = 1979) : a = 44 :=
sorry

end NUMINAMATH_GPT_cube_of_odd_sum_l1972_197262


namespace NUMINAMATH_GPT_angelfish_goldfish_difference_l1972_197273

-- Given statements
variables {A G : ℕ}
def goldfish := 8
def total_fish := 44

-- Conditions
axiom twice_as_many_guppies : G = 2 * A
axiom total_fish_condition : A + G + goldfish = total_fish

-- Theorem
theorem angelfish_goldfish_difference : A - goldfish = 4 :=
by
  sorry

end NUMINAMATH_GPT_angelfish_goldfish_difference_l1972_197273


namespace NUMINAMATH_GPT_cost_per_treat_l1972_197282

def treats_per_day : ℕ := 2
def days_in_month : ℕ := 30
def total_spent : ℝ := 6.0

theorem cost_per_treat : (total_spent / (treats_per_day * days_in_month : ℕ)) = 0.10 :=
by 
  sorry

end NUMINAMATH_GPT_cost_per_treat_l1972_197282


namespace NUMINAMATH_GPT_total_cost_price_l1972_197267

theorem total_cost_price (P_ct P_ch P_bs : ℝ) (h1 : 8091 = P_ct * 1.24)
    (h2 : 5346 = P_ch * 1.18 * 0.95) (h3 : 11700 = P_bs * 1.30) : 
    P_ct + P_ch + P_bs = 20295 := 
by 
    sorry

end NUMINAMATH_GPT_total_cost_price_l1972_197267


namespace NUMINAMATH_GPT_tom_spent_on_videogames_l1972_197235

theorem tom_spent_on_videogames (batman_game superman_game : ℝ) 
  (h1 : batman_game = 13.60) 
  (h2 : superman_game = 5.06) : 
  batman_game + superman_game = 18.66 :=
by 
  sorry

end NUMINAMATH_GPT_tom_spent_on_videogames_l1972_197235


namespace NUMINAMATH_GPT_Dexter_card_count_l1972_197224

theorem Dexter_card_count : 
  let basketball_boxes := 9
  let cards_per_basketball_box := 15
  let football_boxes := basketball_boxes - 3
  let cards_per_football_box := 20
  let basketball_cards := basketball_boxes * cards_per_basketball_box
  let football_cards := football_boxes * cards_per_football_box
  let total_cards := basketball_cards + football_cards
  total_cards = 255 :=
sorry

end NUMINAMATH_GPT_Dexter_card_count_l1972_197224


namespace NUMINAMATH_GPT_intersection_point_l1972_197275

theorem intersection_point (x y : ℚ) (h1 : 8 * x - 5 * y = 40) (h2 : 6 * x + 2 * y = 14) :
  x = 75 / 23 ∧ y = -64 / 23 :=
by
  -- Proof not needed, so we finish with sorry
  sorry

end NUMINAMATH_GPT_intersection_point_l1972_197275


namespace NUMINAMATH_GPT_ordered_pair_arith_progression_l1972_197234

/-- 
Suppose (a, b) is an ordered pair of integers such that the three numbers a, b, and ab 
form an arithmetic progression, in that order. Prove the sum of all possible values of a is 8.
-/
theorem ordered_pair_arith_progression (a b : ℤ) (h : ∃ (a b : ℤ), (b - a = ab - b)) : 
  (a = 0 ∨ a = 1 ∨ a = 3 ∨ a = 4) → a + (if a = 0 then 1 else 0) + 
  (if a = 1 then 1 else 0) + (if a = 3 then 3 else 0) + (if a = 4 then 4 else 0) = 8 :=
by
  sorry

end NUMINAMATH_GPT_ordered_pair_arith_progression_l1972_197234


namespace NUMINAMATH_GPT_sequence_recurrence_l1972_197257

noncomputable def a (n : ℕ) : ℤ := Int.floor ((1 + Real.sqrt 2) ^ n)

theorem sequence_recurrence (k : ℕ) (h : 2 ≤ k) : 
  ∀ n : ℕ, 
  (a 2 * k = 2 * a (2 * k - 1) + a (2 * k - 2)) ∧
  (a (2 * k + 1) = 2 * a (2 * k) + a (2 * k - 1) + 2) :=
sorry

end NUMINAMATH_GPT_sequence_recurrence_l1972_197257


namespace NUMINAMATH_GPT_triangles_same_base_height_have_equal_areas_l1972_197217

theorem triangles_same_base_height_have_equal_areas 
  (b1 h1 b2 h2 : ℝ) 
  (A1 A2 : ℝ) 
  (h1_nonneg : 0 ≤ h1) 
  (h2_nonneg : 0 ≤ h2) 
  (A1_eq : A1 = b1 * h1 / 2) 
  (A2_eq : A2 = b2 * h2 / 2) :
  (A1 = A2 ↔ b1 * h1 = b2 * h2) ∧ (b1 = b2 ∧ h1 = h2 → A1 = A2) :=
by {
  sorry
}

end NUMINAMATH_GPT_triangles_same_base_height_have_equal_areas_l1972_197217


namespace NUMINAMATH_GPT_prove_a_minus_c_l1972_197261

-- Define the given conditions as hypotheses
def condition1 (a b d : ℝ) : Prop := (a + d + b + d) / 2 = 80
def condition2 (b c d : ℝ) : Prop := (b + d + c + d) / 2 = 180

-- State the theorem to be proven
theorem prove_a_minus_c (a b c d : ℝ) (h1 : condition1 a b d) (h2 : condition2 b c d) : a - c = -200 :=
by
  sorry

end NUMINAMATH_GPT_prove_a_minus_c_l1972_197261


namespace NUMINAMATH_GPT_geometric_series_sum_l1972_197231

theorem geometric_series_sum : 
  (3 + 3^2 + 3^3 + 3^4 + 3^5 + 3^6 + 3^7 + 3^8 + 3^9 + 3^10) = 88572 := 
by 
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1972_197231


namespace NUMINAMATH_GPT_tangent_of_11pi_over_4_l1972_197280

theorem tangent_of_11pi_over_4 :
  Real.tan (11 * Real.pi / 4) = -1 :=
sorry

end NUMINAMATH_GPT_tangent_of_11pi_over_4_l1972_197280


namespace NUMINAMATH_GPT_inequality_proof_l1972_197265

theorem inequality_proof
  {x1 x2 x3 x4 : ℝ}
  (h1 : x1 ≥ x2)
  (h2 : x2 ≥ x3)
  (h3 : x3 ≥ x4)
  (h4 : x4 ≥ 2)
  (h5 : x2 + x3 + x4 ≥ x1) :
  (x1 + x2 + x3 + x4)^2 ≤ 4 * x1 * x2 * x3 * x4 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1972_197265


namespace NUMINAMATH_GPT_horner_example_l1972_197239

def horner (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldr (λ a acc => a + x * acc) 0

theorem horner_example : horner [12, 35, -8, 79, 6, 5, 3] (-4) = 220 := by
  sorry

end NUMINAMATH_GPT_horner_example_l1972_197239


namespace NUMINAMATH_GPT_correct_conclusion_l1972_197215

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ :=
  x^3 - 6*x^2 + 9*x - a*b*c

-- The statement to be proven, without providing the actual proof.
theorem correct_conclusion 
  (a b c : ℝ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : f a a b c = 0) 
  (h4 : f b a b c = 0) 
  (h5 : f c a b c = 0) :
  f 0 a b c * f 1 a b c < 0 ∧ f 0 a b c * f 3 a b c > 0 :=
sorry

end NUMINAMATH_GPT_correct_conclusion_l1972_197215


namespace NUMINAMATH_GPT_bricks_in_row_l1972_197296

theorem bricks_in_row 
  (total_bricks : ℕ) 
  (rows_per_wall : ℕ) 
  (num_walls : ℕ)
  (total_rows : ℕ)
  (h1 : total_bricks = 3000)
  (h2 : rows_per_wall = 50)
  (h3 : num_walls = 2) 
  (h4 : total_rows = rows_per_wall * num_walls) :
  total_bricks / total_rows = 30 :=
by
  sorry

end NUMINAMATH_GPT_bricks_in_row_l1972_197296


namespace NUMINAMATH_GPT_tan_mul_tan_l1972_197204

variables {α β : ℝ}

theorem tan_mul_tan (h : 3 * Real.cos (2 * α + β) + 5 * Real.cos β = 0) : 
  Real.tan (α + β) * Real.tan α = -4 :=
sorry

end NUMINAMATH_GPT_tan_mul_tan_l1972_197204


namespace NUMINAMATH_GPT_shepherd_flock_l1972_197295

theorem shepherd_flock (x y : ℕ) (h1 : (x - 1) * 5 = 7 * y) (h2 : x * 3 = 5 * (y - 1)) :
  x + y = 25 :=
sorry

end NUMINAMATH_GPT_shepherd_flock_l1972_197295


namespace NUMINAMATH_GPT_find_b_value_l1972_197276

theorem find_b_value :
  ∃ b : ℕ, 70 = (2 * (b + 1)^2 + 3 * (b + 1) + 4) - (2 * (b - 1)^2 + 3 * (b - 1) + 4) ∧ b > 0 ∧ b < 1000 :=
by
  sorry

end NUMINAMATH_GPT_find_b_value_l1972_197276


namespace NUMINAMATH_GPT_value_of_x4_plus_inv_x4_l1972_197229

theorem value_of_x4_plus_inv_x4 (x : ℝ) (h : x^2 + 1 / x^2 = 6) : x^4 + 1 / x^4 = 34 := 
by
  sorry

end NUMINAMATH_GPT_value_of_x4_plus_inv_x4_l1972_197229


namespace NUMINAMATH_GPT_inner_cube_surface_area_l1972_197242

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h_outer_cube : surface_area_outer_cube = 54) :
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 :=
by
  sorry

end NUMINAMATH_GPT_inner_cube_surface_area_l1972_197242


namespace NUMINAMATH_GPT_length_of_smaller_cube_edge_is_5_l1972_197238

-- Given conditions
def stacked_cube_composed_of_smaller_cubes (n: ℕ) (a: ℕ) : Prop := a * a * a = n

def volume_of_larger_cube (l: ℝ) (v: ℝ) : Prop := l ^ 3 = v

-- Problem statement: Prove that the length of one edge of the smaller cube is 5 cm
theorem length_of_smaller_cube_edge_is_5 :
  ∃ s: ℝ, stacked_cube_composed_of_smaller_cubes 8 2 ∧ volume_of_larger_cube (2*s) 1000 ∧ s = 5 :=
  sorry

end NUMINAMATH_GPT_length_of_smaller_cube_edge_is_5_l1972_197238


namespace NUMINAMATH_GPT_louise_needs_eight_boxes_l1972_197207

-- Define the given conditions
def red_pencils : ℕ := 20
def blue_pencils : ℕ := 2 * red_pencils
def yellow_pencils : ℕ := 40
def green_pencils : ℕ := red_pencils + blue_pencils
def pencils_per_box : ℕ := 20

-- Define the functions to calculate the required number of boxes for each color
def boxes_needed (pencils : ℕ) : ℕ := (pencils + pencils_per_box - 1) / pencils_per_box

-- Calculate the total number of boxes needed by summing the boxes for each color
def total_boxes_needed : ℕ := boxes_needed red_pencils + boxes_needed blue_pencils + boxes_needed yellow_pencils + boxes_needed green_pencils

-- The proof problem statement
theorem louise_needs_eight_boxes : total_boxes_needed = 8 :=
by
  sorry

end NUMINAMATH_GPT_louise_needs_eight_boxes_l1972_197207


namespace NUMINAMATH_GPT_inequality_abc_l1972_197212

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b))) ≥ 3 / 2 := 
sorry

end NUMINAMATH_GPT_inequality_abc_l1972_197212
