import Mathlib

namespace NUMINAMATH_GPT_balls_into_boxes_l92_9269

theorem balls_into_boxes : ∃ n : ℕ, n = 240 ∧ ∃ f : Fin 5 → Fin 4, ∀ i : Fin 4, ∃ j : Fin 5, f j = i := by
  sorry

end NUMINAMATH_GPT_balls_into_boxes_l92_9269


namespace NUMINAMATH_GPT_op_example_l92_9255

variables {α β : ℚ}

def op (α β : ℚ) := α * β + 1

theorem op_example : op 2 (-3) = -5 :=
by
  -- The proof is omitted as requested
  sorry

end NUMINAMATH_GPT_op_example_l92_9255


namespace NUMINAMATH_GPT_proof_of_problem1_proof_of_problem2_proof_of_problem3_proof_of_problem4_l92_9262

noncomputable def problem1 (x y : ℝ) (h : x^2 - 6*x + 2*y = 0) : Prop :=
  y ≤ 4.5

noncomputable def problem2 (x y : ℝ) (h : 3*x^2 + 12*x - 2*y - 4 = 0) : Prop :=
  y ≥ -8

noncomputable def problem3 (x y : ℝ) (h : y = 2*x / (1 + x^2)) : Prop :=
  -1 ≤ y ∧ y ≤ 1

noncomputable def problem4 (x y : ℝ) (h : y = (2*x - 1) / (x^2 + 2*x + 1)) : Prop :=
  y ≤ 1/3

-- Proving that the properties hold:
theorem proof_of_problem1 (x y : ℝ) (h : x^2 - 6*x + 2*y = 0) : problem1 x y h :=
  sorry

theorem proof_of_problem2 (x y : ℝ) (h : 3*x^2 + 12*x - 2*y - 4 = 0) : problem2 x y h :=
  sorry

theorem proof_of_problem3 (x y : ℝ) (h : y = 2*x / (1 + x^2)) : problem3 x y h :=
  sorry

theorem proof_of_problem4 (x y : ℝ) (h : y = (2*x - 1) / (x^2 + 2*x + 1)) : problem4 x y h :=
  sorry

end NUMINAMATH_GPT_proof_of_problem1_proof_of_problem2_proof_of_problem3_proof_of_problem4_l92_9262


namespace NUMINAMATH_GPT_solve_for_x_l92_9292

theorem solve_for_x : ∀ x : ℤ, 5 - x = 8 → x = -3 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_solve_for_x_l92_9292


namespace NUMINAMATH_GPT_part1_average_decrease_rate_part2_unit_price_reduction_l92_9259

-- Part 1: Prove the average decrease rate is 10%
theorem part1_average_decrease_rate (p0 p2 : ℝ) (x : ℝ) 
    (h1 : p0 = 200) 
    (h2 : p2 = 162) 
    (hx : (1 - x)^2 = p2 / p0) : x = 0.1 :=
by {
    sorry
}

-- Part 2: Prove the unit price reduction should be 15 yuan
theorem part2_unit_price_reduction (p_sell p_factory profit : ℝ) (n_initial dn m : ℝ)
    (h3 : p_sell = 200)
    (h4 : p_factory = 162)
    (h5 : n_initial = 20)
    (h6 : dn = 10)
    (h7 : profit = 1150)
    (hx : (38 - m) * (n_initial + 2 * m) = profit) : m = 15 :=
by {
    sorry
}

end NUMINAMATH_GPT_part1_average_decrease_rate_part2_unit_price_reduction_l92_9259


namespace NUMINAMATH_GPT_smallest_integer_coprime_with_462_l92_9235

theorem smallest_integer_coprime_with_462 :
  ∃ n, n > 1 ∧ Nat.gcd n 462 = 1 ∧ ∀ m, m > 1 ∧ Nat.gcd m 462 = 1 → n ≤ m → n = 13 := by
  sorry

end NUMINAMATH_GPT_smallest_integer_coprime_with_462_l92_9235


namespace NUMINAMATH_GPT_younger_person_age_l92_9200

theorem younger_person_age (e y : ℕ) 
  (h1: e = y + 20)
  (h2: e - 10 = 5 * (y - 10)) : 
  y = 15 := 
by
  sorry

end NUMINAMATH_GPT_younger_person_age_l92_9200


namespace NUMINAMATH_GPT_units_digit_G_1000_l92_9260

def modified_fermat_number (n : ℕ) : ℕ := 5^(5^n) + 6

theorem units_digit_G_1000 : (modified_fermat_number 1000) % 10 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_units_digit_G_1000_l92_9260


namespace NUMINAMATH_GPT_find_k_l92_9270

theorem find_k (k : ℝ) (h : ∀ x y : ℝ, (x, y) = (-2, -1) → y = k * x + 2) : k = 3 / 2 :=
sorry

end NUMINAMATH_GPT_find_k_l92_9270


namespace NUMINAMATH_GPT_intersection_points_l92_9204

def equation1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9
def equation2 (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 25

theorem intersection_points :
  ∃ (x1 y1 x2 y2 : ℝ),
    equation1 x1 y1 ∧ equation2 x1 y1 ∧
    equation1 x2 y2 ∧ equation2 x2 y2 ∧
    (x1, y1) ≠ (x2, y2) ∧
    ∀ (x y : ℝ), equation1 x y ∧ equation2 x y → (x, y) = (x1, y1) ∨ (x, y) = (x2, y2) := sorry

end NUMINAMATH_GPT_intersection_points_l92_9204


namespace NUMINAMATH_GPT_first_sales_amount_l92_9241

-- Conditions from the problem
def first_sales_royalty : ℝ := 8 -- million dollars
def second_sales_royalty : ℝ := 9 -- million dollars
def second_sales_amount : ℝ := 108 -- million dollars
def decrease_percentage : ℝ := 0.7916666666666667

-- The goal is to determine the first sales amount, S, meeting the conditions.
theorem first_sales_amount :
  ∃ S : ℝ,
    (first_sales_royalty / S - second_sales_royalty / second_sales_amount = decrease_percentage * (first_sales_royalty / S)) ∧
    S = 20 :=
sorry

end NUMINAMATH_GPT_first_sales_amount_l92_9241


namespace NUMINAMATH_GPT_tan_sum_pi_div_4_sin_fraction_simplifies_to_1_l92_9287

variable (α : ℝ)
variable (π : ℝ) [Fact (π > 0)]

-- Assume condition
axiom tan_alpha_eq_2 : Real.tan α = 2

-- Goal (1): Prove that tan(α + π/4) = -3
theorem tan_sum_pi_div_4 : Real.tan (α + π / 4) = -3 :=
by
  sorry

-- Goal (2): Prove that (sin(2α) / (sin^2(α) + sin(α) * cos(α) - cos(2α) - 1)) = 1
theorem sin_fraction_simplifies_to_1 :
  (Real.sin (2 * α)) / (Real.sin (α)^2 + Real.sin (α) * Real.cos (α) - Real.cos (2 * α) - 1) = 1 :=
by
  sorry

end NUMINAMATH_GPT_tan_sum_pi_div_4_sin_fraction_simplifies_to_1_l92_9287


namespace NUMINAMATH_GPT_average_score_all_test_takers_l92_9201

def avg (scores : List ℕ) : ℕ := scores.sum / scores.length

theorem average_score_all_test_takers (s_avg u_avg n : ℕ) 
  (H1 : s_avg = 42) (H2 : u_avg = 38) (H3 : n = 20) : avg ([s_avg * n, u_avg * n]) / (2 * n) = 40 := 
by sorry

end NUMINAMATH_GPT_average_score_all_test_takers_l92_9201


namespace NUMINAMATH_GPT_a5_a6_values_b_n_general_formula_minimum_value_T_n_l92_9249

section sequence_problems

def sequence_n (n : ℕ) : ℤ :=
if n = 0 then 1
else if n = 1 then 1
else sequence_n (n - 2) + 2 * (-1)^(n - 2)

def b_sequence (n : ℕ) : ℤ :=
sequence_n (2 * n)

def S_n (n : ℕ) : ℤ :=
(n + 1) * (sequence_n n)

def T_n (n : ℕ) : ℤ :=
(S_n (2 * n) - 18)

theorem a5_a6_values :
  sequence_n 4 = -3 ∧ sequence_n 5 = 5 := by
  sorry

theorem b_n_general_formula (n : ℕ) :
  b_sequence n = 2 * n - 1 := by
  sorry

theorem minimum_value_T_n :
  ∃ n, T_n n = -72 := by
  sorry

end sequence_problems

end NUMINAMATH_GPT_a5_a6_values_b_n_general_formula_minimum_value_T_n_l92_9249


namespace NUMINAMATH_GPT_log_properties_l92_9240

theorem log_properties :
  (Real.log 5) ^ 2 + (Real.log 2) * (Real.log 50) = 1 :=
by sorry

end NUMINAMATH_GPT_log_properties_l92_9240


namespace NUMINAMATH_GPT_person_age_l92_9289

variable (x : ℕ) -- Define the variable for age

-- State the condition as a hypothesis
def condition (x : ℕ) : Prop :=
  3 * (x + 3) - 3 * (x - 3) = x

-- State the theorem to be proved
theorem person_age (x : ℕ) (h : condition x) : x = 18 := 
sorry

end NUMINAMATH_GPT_person_age_l92_9289


namespace NUMINAMATH_GPT_range_of_p_l92_9211

noncomputable def success_prob_4_engine (p : ℝ) : ℝ :=
  4 * p^3 * (1 - p) + p^4

noncomputable def success_prob_2_engine (p : ℝ) : ℝ :=
  p^2

theorem range_of_p (p : ℝ) (h1 : 0 < p) (h2 : p < 1) :
  success_prob_4_engine p > success_prob_2_engine p ↔ (1/3 < p ∧ p < 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_p_l92_9211


namespace NUMINAMATH_GPT_percentage_of_y_l92_9216

theorem percentage_of_y (x y P : ℝ) (h1 : 0.10 * x = (P/100) * y) (h2 : x / y = 2) : P = 20 :=
sorry

end NUMINAMATH_GPT_percentage_of_y_l92_9216


namespace NUMINAMATH_GPT_fewer_sevens_l92_9296

def seven_representation (n : ℕ) : ℕ :=
  (7 * (10^n - 1)) / 9

theorem fewer_sevens (n : ℕ) :
  ∃ m, m < n ∧ 
    (∃ expr : ℕ → ℕ, (∀ i < n, expr i = 7) ∧ seven_representation n = expr m) :=
sorry

end NUMINAMATH_GPT_fewer_sevens_l92_9296


namespace NUMINAMATH_GPT_expand_and_simplify_expression_l92_9246

variable {x y : ℝ} {i : ℂ}

-- Declare i as the imaginary unit satisfying i^2 = -1
axiom imaginary_unit : i^2 = -1

theorem expand_and_simplify_expression :
  (x + 3 + i * y) * (x + 3 - i * y) + (x - 2 + 2 * i * y) * (x - 2 - 2 * i * y)
  = 2 * x^2 + 2 * x + 13 - 5 * y^2 :=
by
  sorry

end NUMINAMATH_GPT_expand_and_simplify_expression_l92_9246


namespace NUMINAMATH_GPT_bicycle_meets_light_vehicle_l92_9207

noncomputable def meeting_time (v_1 v_2 v_3 v_4 : ℚ) : ℚ :=
  let x := 2 * (v_1 + v_4)
  let y := 6 * (v_2 - v_4)
  (x + y) / (v_3 + v_4) + 12

theorem bicycle_meets_light_vehicle (v_1 v_2 v_3 v_4 : ℚ) (h1 : 2 * (v_1 + v_4) = x)
  (h2 : x + y = 4 * (v_1 + v_2))
  (h3 : x + y = 5 * (v_2 + v_3))
  (h4 : 6 * (v_2 - v_4) = y) :
  meeting_time v_1 v_2 v_3 v_4 = 15 + 1/3 :=
by
  sorry

end NUMINAMATH_GPT_bicycle_meets_light_vehicle_l92_9207


namespace NUMINAMATH_GPT_jim_travel_distance_l92_9267

theorem jim_travel_distance
  (john_distance : ℕ := 15)
  (jill_distance : ℕ := john_distance - 5)
  (jim_distance : ℕ := jill_distance * 20 / 100) :
  jim_distance = 2 := 
by
  sorry

end NUMINAMATH_GPT_jim_travel_distance_l92_9267


namespace NUMINAMATH_GPT_total_matches_played_l92_9264

theorem total_matches_played
  (avg_runs_first_20: ℕ) (num_first_20: ℕ) (avg_runs_next_10: ℕ) (num_next_10: ℕ) (overall_avg: ℕ) (total_matches: ℕ) :
  avg_runs_first_20 = 40 →
  num_first_20 = 20 →
  avg_runs_next_10 = 13 →
  num_next_10 = 10 →
  overall_avg = 31 →
  (num_first_20 + num_next_10 = total_matches) →
  total_matches = 30 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_total_matches_played_l92_9264


namespace NUMINAMATH_GPT_select_k_numbers_l92_9285

theorem select_k_numbers (a : ℕ → ℝ) (k : ℕ) (h1 : ∀ n, 0 < a n) 
  (h2 : ∀ n m, n < m → a n ≥ a m) (h3 : a 1 = 1 / (2 * k)) 
  (h4 : ∑' n, a n = 1) :
  ∃ (f : ℕ → ℕ) (hf : ∀ i j, i ≠ j → f i ≠ f j), 
    (∀ i, i < k → a (f i) > 1/2 * a (f 0)) :=
by
  sorry

end NUMINAMATH_GPT_select_k_numbers_l92_9285


namespace NUMINAMATH_GPT_exactly_one_greater_than_one_l92_9277

theorem exactly_one_greater_than_one (x1 x2 x3 : ℝ) 
  (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3)
  (h4 : x1 * x2 * x3 = 1)
  (h5 : x1 + x2 + x3 > (1 / x1) + (1 / x2) + (1 / x3)) :
  (x1 > 1 ∧ x2 ≤ 1 ∧ x3 ≤ 1) ∨ 
  (x1 ≤ 1 ∧ x2 > 1 ∧ x3 ≤ 1) ∨ 
  (x1 ≤ 1 ∧ x2 ≤ 1 ∧ x3 > 1) :=
sorry

end NUMINAMATH_GPT_exactly_one_greater_than_one_l92_9277


namespace NUMINAMATH_GPT_find_xyz_l92_9295

variable (x y z : ℝ)
variable (h1 : x = 80 + 0.11 * 80)
variable (h2 : y = 120 - 0.15 * 120)
variable (h3 : z = 0.20 * (0.40 * (x + y)) + 0.40 * (x + y))

theorem find_xyz (hx : x = 88.8) (hy : y = 102) (hz : z = 91.584) : 
  x = 88.8 ∧ y = 102 ∧ z = 91.584 := by
  sorry

end NUMINAMATH_GPT_find_xyz_l92_9295


namespace NUMINAMATH_GPT_tan_alpha_two_implies_fraction_eq_three_fourths_l92_9220

variable {α : ℝ}

theorem tan_alpha_two_implies_fraction_eq_three_fourths (h1 : Real.tan α = 2) (h2 : Real.cos α ≠ 0) : 
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 := 
sorry

end NUMINAMATH_GPT_tan_alpha_two_implies_fraction_eq_three_fourths_l92_9220


namespace NUMINAMATH_GPT_Trishul_invested_less_than_Raghu_l92_9233

-- Definitions based on conditions
def Raghu_investment : ℝ := 2500
def Total_investment : ℝ := 7225

def Vishal_invested_more_than_Trishul (T V : ℝ) : Prop :=
  V = 1.10 * T

noncomputable def percentage_decrease (original decrease : ℝ) : ℝ :=
  (decrease / original) * 100

theorem Trishul_invested_less_than_Raghu (T V : ℝ) 
  (h1 : Vishal_invested_more_than_Trishul T V)
  (h2 : T + V + Raghu_investment = Total_investment) :
  percentage_decrease Raghu_investment (Raghu_investment - T) = 10 := by
  sorry

end NUMINAMATH_GPT_Trishul_invested_less_than_Raghu_l92_9233


namespace NUMINAMATH_GPT_minute_hand_40_min_angle_l92_9210

noncomputable def minute_hand_rotation_angle (minutes : ℕ): ℝ :=
  if minutes = 60 then -2 * Real.pi 
  else (minutes / 60) * -2 * Real.pi

theorem minute_hand_40_min_angle :
  minute_hand_rotation_angle 40 = - (4 / 3) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_minute_hand_40_min_angle_l92_9210


namespace NUMINAMATH_GPT_time_difference_l92_9275

noncomputable def hour_angle (n : ℝ) : ℝ :=
  150 + (n / 2)

noncomputable def minute_angle (n : ℝ) : ℝ :=
  6 * n

theorem time_difference (n1 n2 : ℝ)
  (h1 : |(hour_angle n1) - (minute_angle n1)| = 120)
  (h2 : |(hour_angle n2) - (minute_angle n2)| = 120) :
  n2 - n1 = 43.64 := 
sorry

end NUMINAMATH_GPT_time_difference_l92_9275


namespace NUMINAMATH_GPT_exponential_fixed_point_l92_9222

variable (a : ℝ)

noncomputable def f (x : ℝ) := a^(x - 1) + 3

theorem exponential_fixed_point (ha1 : a > 0) (ha2 : a ≠ 1) : f a 1 = 4 :=
by
  sorry

end NUMINAMATH_GPT_exponential_fixed_point_l92_9222


namespace NUMINAMATH_GPT_find_m_l92_9244

noncomputable def slope_at_one (m : ℝ) := 2 + m

noncomputable def tangent_line_eq (m : ℝ) (x : ℝ) := (slope_at_one m) * x - 2 * m

noncomputable def y_intercept (m : ℝ) := tangent_line_eq m 0

noncomputable def x_intercept (m : ℝ) := - (y_intercept m) / (slope_at_one m)

noncomputable def intercept_sum_eq (m : ℝ) := (x_intercept m) + (y_intercept m)

theorem find_m (m : ℝ) (h : m ≠ -2) (h2 : intercept_sum_eq m = 12) : m = -3 ∨ m = -4 := 
sorry

end NUMINAMATH_GPT_find_m_l92_9244


namespace NUMINAMATH_GPT_boys_in_parkway_l92_9214

theorem boys_in_parkway (total_students : ℕ) (students_playing_soccer : ℕ) (percentage_boys_playing_soccer : ℝ)
                        (girls_not_playing_soccer : ℕ) :
                        total_students = 420 ∧ students_playing_soccer = 250 ∧ percentage_boys_playing_soccer = 0.86 
                        ∧ girls_not_playing_soccer = 73 → 
                        ∃ total_boys : ℕ, total_boys = 312 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_boys_in_parkway_l92_9214


namespace NUMINAMATH_GPT_track_team_children_l92_9284

/-- There were initially 18 girls and 15 boys on the track team.
    7 more girls joined the team, and 4 boys quit the team.
    The proof shows that the total number of children on the track team after the changes is 36. -/
theorem track_team_children (initial_girls initial_boys girls_joined boys_quit : ℕ)
  (h_initial_girls : initial_girls = 18)
  (h_initial_boys : initial_boys = 15)
  (h_girls_joined : girls_joined = 7)
  (h_boys_quit : boys_quit = 4) :
  initial_girls + girls_joined - boys_quit + initial_boys = 36 :=
by
  -- Placeholder to indicate the proof is omitted
  sorry

end NUMINAMATH_GPT_track_team_children_l92_9284


namespace NUMINAMATH_GPT_distinct_real_roots_l92_9225

-- Define the polynomial equation as a Lean function
def polynomial (a x : ℝ) : ℝ :=
  (a + 1) * (x ^ 2 + 1) ^ 2 - (2 * a + 3) * (x ^ 2 + 1) * x + (a + 2) * x ^ 2

-- The theorem we need to prove
theorem distinct_real_roots (a : ℝ) : 
  (∃ (x y : ℝ), x ≠ y ∧ polynomial a x = 0 ∧ polynomial a y = 0) ↔ a ≠ -1 :=
by
  sorry

end NUMINAMATH_GPT_distinct_real_roots_l92_9225


namespace NUMINAMATH_GPT_eval_expression_l92_9209

-- Definitions for the problem conditions
def reciprocal (a : ℕ) : ℚ := 1 / a

-- The theorem statement
theorem eval_expression : (reciprocal 9 - reciprocal 6)⁻¹ = -18 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l92_9209


namespace NUMINAMATH_GPT_pyramid_volume_l92_9253

noncomputable def volume_of_pyramid (AB AD BD AE : ℝ) (p : AB = 9 ∧ AD = 10 ∧ BD = 11 ∧ AE = 10.5) : ℝ :=
  1 / 3 * (60 * (2 ^ (1 / 2))) * (5 * (2 ^ (1 / 2)))

theorem pyramid_volume (AB AD BD AE : ℝ) (h1 : AB = 9) (h2 : AD = 10) (h3 : BD = 11) (h4 : AE = 10.5)
  (V : ℝ) (hV : V = 200) : 
  volume_of_pyramid AB AD BD AE (⟨h1, ⟨h2, ⟨h3, h4⟩⟩⟩) = V :=
sorry

end NUMINAMATH_GPT_pyramid_volume_l92_9253


namespace NUMINAMATH_GPT_faster_speed_l92_9268

theorem faster_speed (v : ℝ) (h1 : ∀ (t : ℝ), t = 50 / 10) (h2 : ∀ (d : ℝ), d = 50 + 20) (h3 : ∀ (t : ℝ), t = 70 / v) : v = 14 :=
by
  sorry

end NUMINAMATH_GPT_faster_speed_l92_9268


namespace NUMINAMATH_GPT_solution_set_of_abs_x_plus_one_gt_one_l92_9286

theorem solution_set_of_abs_x_plus_one_gt_one :
  {x : ℝ | |x + 1| > 1} = {x : ℝ | x < -2 ∨ x > 0} :=
sorry

end NUMINAMATH_GPT_solution_set_of_abs_x_plus_one_gt_one_l92_9286


namespace NUMINAMATH_GPT_car_second_half_speed_l92_9202

theorem car_second_half_speed (D : ℝ) (V : ℝ) :
  let average_speed := 60  -- km/hr
  let first_half_speed := 75 -- km/hr
  average_speed = D / ((D / 2) / first_half_speed + (D / 2) / V) ->
  V = 150 :=
by
  sorry

end NUMINAMATH_GPT_car_second_half_speed_l92_9202


namespace NUMINAMATH_GPT_solve_system_of_equations_solve_linear_inequality_l92_9278

-- Part 1: System of equations
theorem solve_system_of_equations (x y : ℝ) (h1 : 5 * x + 2 * y = 25) (h2 : 3 * x + 4 * y = 15) : 
  x = 5 ∧ y = 0 := sorry

-- Part 2: Linear inequality
theorem solve_linear_inequality (x : ℝ) (h : 2 * x - 6 < 3 * x) : 
  x > -6 := sorry

end NUMINAMATH_GPT_solve_system_of_equations_solve_linear_inequality_l92_9278


namespace NUMINAMATH_GPT_triangle_area_x_value_l92_9248

theorem triangle_area_x_value :
  ∃ x : ℝ, x > 0 ∧ 100 = (1 / 2) * x * (3 * x) ∧ x = 10 * Real.sqrt 6 / 3 :=
sorry

end NUMINAMATH_GPT_triangle_area_x_value_l92_9248


namespace NUMINAMATH_GPT_find_x_for_parallel_vectors_l92_9218

-- Define the vectors a and b
def a (x : ℝ) : ℝ × ℝ := (4, x)
def b : ℝ × ℝ := (-4, 4)

-- Define parallelism condition for two 2D vectors
def are_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Define the main theorem statement
theorem find_x_for_parallel_vectors (x : ℝ) (h : are_parallel (a x) b) : x = -4 :=
by sorry

end NUMINAMATH_GPT_find_x_for_parallel_vectors_l92_9218


namespace NUMINAMATH_GPT_concyclic_iff_l92_9237

variables {A B C H O' N D : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace H]
variables [MetricSpace O'] [MetricSpace N] [MetricSpace D]
variables (a b c R : ℝ)

-- Conditions from the problem
def is_orthocenter (H : Type*) (A B C : Type*) : Prop :=
  -- definition of orthocenter using suitable predicates (omitted for brevity) 
  sorry

def is_circumcenter (O' : Type*) (B H C : Type*) : Prop :=
  -- definition of circumcenter using suitable predicates (omitted for brevity) 
  sorry

def is_midpoint (N : Type*) (A O' : Type*) : Prop :=
  -- definition of midpoint using suitable predicates (omitted for brevity) 
  sorry

def is_reflection (N D : Type*) (B C : Type*) : Prop :=
  -- definition of reflection about the side BC (omitted for brevity) 
  sorry

-- Definition that points A, B, C, D are concyclic
def are_concyclic (A B C D : Type*) : Prop :=
  -- definition using suitable predicates (omitted for brevity)
  sorry

-- Main theorem statement
theorem concyclic_iff (h1 : is_orthocenter H A B C) (h2 : is_circumcenter O' B H C) 
                      (h3 : is_midpoint N A O') (h4 : is_reflection N D B C)
                      (ha : a = 1) (hb : b = 1) (hc : c = 1) (hR : R = 1) :
  are_concyclic A B C D ↔ b^2 + c^2 - a^2 = 3 * R^2 := 
sorry

end NUMINAMATH_GPT_concyclic_iff_l92_9237


namespace NUMINAMATH_GPT_discount_savings_l92_9299

theorem discount_savings (initial_price discounted_price : ℝ)
  (h_initial : initial_price = 475)
  (h_discounted : discounted_price = 199) :
  initial_price - discounted_price = 276 :=
by
  rw [h_initial, h_discounted]
  sorry

end NUMINAMATH_GPT_discount_savings_l92_9299


namespace NUMINAMATH_GPT_segment_length_is_ten_l92_9250

-- Definition of the cube root function and the absolute value
def cube_root (x : ℝ) : ℝ := x^(1/3)

def absolute (x : ℝ) : ℝ := abs x

-- The prerequisites as conditions for the endpoints
def endpoints_satisfy (x : ℝ) : Prop := absolute (x - cube_root 27) = 5

-- Length of the segment determined by the endpoints
def segment_length (x1 x2 : ℝ) : ℝ := absolute (x2 - x1)

-- Theorem statement
theorem segment_length_is_ten : (∀ x, endpoints_satisfy x) → segment_length (-2) 8 = 10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_segment_length_is_ten_l92_9250


namespace NUMINAMATH_GPT_towers_per_castle_jeff_is_5_l92_9282

-- Define the number of sandcastles on Mark's beach
def num_castles_mark : ℕ := 20

-- Define the number of towers per sandcastle on Mark's beach
def towers_per_castle_mark : ℕ := 10

-- Calculate the total number of towers on Mark's beach
def total_towers_mark : ℕ := num_castles_mark * towers_per_castle_mark

-- Define the number of sandcastles on Jeff's beach (3 times that of Mark's)
def num_castles_jeff : ℕ := 3 * num_castles_mark

-- Define the total number of sandcastles on both beaches
def total_sandcastles : ℕ := num_castles_mark + num_castles_jeff
  
-- Define the combined total number of sandcastles and towers on both beaches
def combined_total : ℕ := 580

-- Define the number of towers per sandcastle on Jeff's beach
def towers_per_castle_jeff : ℕ := sorry

-- Define the total number of towers on Jeff's beach
def total_towers_jeff (T : ℕ) : ℕ := num_castles_jeff * T

-- Prove that the number of towers per sandcastle on Jeff's beach is 5
theorem towers_per_castle_jeff_is_5 : 
    200 + total_sandcastles + total_towers_jeff towers_per_castle_jeff = combined_total → 
    towers_per_castle_jeff = 5
:= by
    sorry

end NUMINAMATH_GPT_towers_per_castle_jeff_is_5_l92_9282


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l92_9263

theorem isosceles_triangle_base_length (P Q : ℕ) (x y : ℕ) (hP : P = 15) (hQ : Q = 12) (hPerimeter : 2 * x + y = 27) 
      (hCondition : (y = P ∧ (1 / 2) * x + x = P) ∨ (y = Q ∧ (1 / 2) * x + x = Q)) : 
  y = 7 ∨ y = 11 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l92_9263


namespace NUMINAMATH_GPT_slope_of_line_l92_9236

theorem slope_of_line : ∀ (x y : ℝ), 2 * x - 4 * y + 7 = 0 → (y = (1/2) * x - 7 / 4) :=
by
  intro x y h
  -- This would typically involve rearranging the given equation to the slope-intercept form
  -- but as we are focusing on creating the statement, we insert sorry to skip the proof
  sorry

end NUMINAMATH_GPT_slope_of_line_l92_9236


namespace NUMINAMATH_GPT_cos_thirteen_pi_over_four_l92_9230

theorem cos_thirteen_pi_over_four : Real.cos (13 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_thirteen_pi_over_four_l92_9230


namespace NUMINAMATH_GPT_gcd_lcm_product_360_distinct_gcd_values_l92_9219

/-- 
  Given two integers a and b, such that the product of their gcd and lcm is 360,
  we need to prove that the number of distinct possible values for their gcd is 9.
--/
theorem gcd_lcm_product_360_distinct_gcd_values :
  ∀ (a b : ℕ), gcd a b * lcm a b = 360 → 
  (∃ gcd_values : Finset ℕ, gcd_values.card = 9 ∧ ∀ g, g ∈ gcd_values ↔ g = gcd a b) :=
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_360_distinct_gcd_values_l92_9219


namespace NUMINAMATH_GPT_calculate_expr_l92_9297

theorem calculate_expr :
  ( (5 / 12: ℝ) ^ 2022) * (-2.4) ^ 2023 = - (12 / 5: ℝ) := 
by 
  sorry

end NUMINAMATH_GPT_calculate_expr_l92_9297


namespace NUMINAMATH_GPT_correct_inequality_incorrect_inequality1_incorrect_inequality2_correct_option_d_l92_9266

theorem correct_inequality:
    (-21 : ℤ) > (-21 : ℤ) := by sorry

theorem incorrect_inequality1 :
    -abs (10 + 1 / 2) < (8 + 2 / 3) := by sorry

theorem incorrect_inequality2 :
    (-abs (7 + 2 / 3)) ≠ (- (- (7 + 2 / 3))) := by sorry

theorem correct_option_d :
    (-5 / 6 : ℚ) < (-4 / 5 : ℚ) := by sorry

end NUMINAMATH_GPT_correct_inequality_incorrect_inequality1_incorrect_inequality2_correct_option_d_l92_9266


namespace NUMINAMATH_GPT_no_such_abc_exists_l92_9273

-- Define the conditions for the leading coefficients and constant terms
def leading_coeff_conditions (a b c : ℝ) : Prop :=
  ((a > 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ c > 0 ∧ b < 0) ∨ (b > 0 ∧ c > 0 ∧ a < 0))

def constant_term_conditions (a b c : ℝ) : Prop :=
  ((c > 0 ∧ a < 0 ∧ b < 0) ∨ (a > 0 ∧ b < 0 ∧ c < 0) ∨ (b > 0 ∧ c < 0 ∧ a < 0))

-- The final statement that encapsulates the contradiction
theorem no_such_abc_exists : ¬ ∃ a b c : ℝ, leading_coeff_conditions a b c ∧ constant_term_conditions a b c :=
by
  sorry

end NUMINAMATH_GPT_no_such_abc_exists_l92_9273


namespace NUMINAMATH_GPT_smallest_integer_of_lcm_gcd_l92_9234

theorem smallest_integer_of_lcm_gcd (m : ℕ) (h1 : m > 0) (h2 : Nat.lcm 60 m / Nat.gcd 60 m = 44) : m = 165 :=
sorry

end NUMINAMATH_GPT_smallest_integer_of_lcm_gcd_l92_9234


namespace NUMINAMATH_GPT_profit_percentage_is_23_16_l92_9229

   noncomputable def cost_price (mp : ℝ) : ℝ := 95 * mp
   noncomputable def selling_price (mp : ℝ) : ℝ := 120 * (mp - (0.025 * mp))
   noncomputable def profit_percent (cp sp : ℝ) : ℝ := ((sp - cp) / cp) * 100

   theorem profit_percentage_is_23_16 
     (mp : ℝ) (h_mp_gt_zero : mp > 0) : 
       profit_percent (cost_price mp) (selling_price mp) = 23.16 :=
   by 
     sorry
   
end NUMINAMATH_GPT_profit_percentage_is_23_16_l92_9229


namespace NUMINAMATH_GPT_percentage_emails_moved_to_work_folder_l92_9232

def initialEmails : ℕ := 400
def trashedEmails : ℕ := initialEmails / 2
def remainingEmailsAfterTrash : ℕ := initialEmails - trashedEmails
def emailsLeftInInbox : ℕ := 120
def emailsMovedToWorkFolder : ℕ := remainingEmailsAfterTrash - emailsLeftInInbox

theorem percentage_emails_moved_to_work_folder :
  (emailsMovedToWorkFolder * 100 / remainingEmailsAfterTrash) = 40 := by
  sorry

end NUMINAMATH_GPT_percentage_emails_moved_to_work_folder_l92_9232


namespace NUMINAMATH_GPT_squirrels_in_tree_l92_9254

theorem squirrels_in_tree (N S : ℕ) (h₁ : N = 2) (h₂ : S - N = 2) : S = 4 :=
by
  sorry

end NUMINAMATH_GPT_squirrels_in_tree_l92_9254


namespace NUMINAMATH_GPT_values_of_a_and_b_intervals_of_monotonicity_range_of_a_for_three_roots_l92_9283

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 - 3 * a * x^2 + 2 * b * x

theorem values_of_a_and_b (h : ∀ x, f x (1 / 3) (-1 / 2) ≤ f 1 (1 / 3) (-1 / 2)) :
  (∃ a b, a = 1 / 3 ∧ b = -1 / 2) :=
sorry

theorem intervals_of_monotonicity (a b : ℝ) (h : ∀ x, f x a b ≤ f 1 a b) :
  (∀ x, (f x a b ≥ 0 ↔ x ≤ -1 / 3 ∨ x ≥ 1) ∧ (f x a b ≤ 0 ↔ -1 / 3 ≤ x ∧ x ≤ 1)) :=
sorry

theorem range_of_a_for_three_roots :
  (∃ a, -1 < a ∧ a < 5 / 27) :=
sorry

end NUMINAMATH_GPT_values_of_a_and_b_intervals_of_monotonicity_range_of_a_for_three_roots_l92_9283


namespace NUMINAMATH_GPT_largest_divisor_of_even_n_cube_difference_l92_9231

theorem largest_divisor_of_even_n_cube_difference (n : ℤ) (h : Even n) : 6 ∣ (n^3 - n) := by
  sorry

end NUMINAMATH_GPT_largest_divisor_of_even_n_cube_difference_l92_9231


namespace NUMINAMATH_GPT_no_adjacent_same_roll_probability_l92_9206

-- We define probabilistic event on rolling a six-sided die and sitting around a circular table
noncomputable def probability_no_adjacent_same_roll : ℚ :=
  1 * (5/6) * (5/6) * (5/6) * (5/6) * (4/6)

theorem no_adjacent_same_roll_probability :
  probability_no_adjacent_same_roll = 625/1944 :=
by
  sorry

end NUMINAMATH_GPT_no_adjacent_same_roll_probability_l92_9206


namespace NUMINAMATH_GPT_solve_for_x_l92_9271

theorem solve_for_x (x : ℝ) : (5 * x - 2) / (6 * x - 6) = 3 / 4 ↔ x = -5 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l92_9271


namespace NUMINAMATH_GPT_area_square_l92_9226

-- Define the conditions
variables (l r s : ℝ)
variable (breadth : ℝ := 10)
variable (area_rect : ℝ := 180)

-- Given conditions
def length_is_two_fifths_radius : Prop := l = (2/5) * r
def radius_is_side_square : Prop := r = s
def area_of_rectangle : Prop := area_rect = l * breadth

-- The theorem statement
theorem area_square (h1 : length_is_two_fifths_radius l r)
                    (h2 : radius_is_side_square r s)
                    (h3 : area_of_rectangle l breadth area_rect) :
  s^2 = 2025 :=
by
  sorry

end NUMINAMATH_GPT_area_square_l92_9226


namespace NUMINAMATH_GPT_solve_inequality_l92_9228

theorem solve_inequality (x : ℝ) : 2 * (5 * x + 3) ≤ x - 3 * (1 - 2 * x) → x ≤ -3 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l92_9228


namespace NUMINAMATH_GPT_simon_can_make_blueberry_pies_l92_9247

theorem simon_can_make_blueberry_pies (bush1 bush2 blueberries_per_pie : ℕ) (h1 : bush1 = 100) (h2 : bush2 = 200) (h3 : blueberries_per_pie = 100) : 
  (bush1 + bush2) / blueberries_per_pie = 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_simon_can_make_blueberry_pies_l92_9247


namespace NUMINAMATH_GPT_problem_l92_9224

theorem problem (x : ℝ) (h : x^2 + 5 * x - 990 = 0) : x^3 + 6 * x^2 - 985 * x + 1012 = 2002 :=
sorry

end NUMINAMATH_GPT_problem_l92_9224


namespace NUMINAMATH_GPT_max_value_fourth_power_l92_9281

theorem max_value_fourth_power (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 4) : 
  a^4 + b^4 + c^4 + d^4 ≤ 4^(4/3) :=
sorry

end NUMINAMATH_GPT_max_value_fourth_power_l92_9281


namespace NUMINAMATH_GPT_distance_against_stream_l92_9290

variable (vs : ℝ) -- speed of the stream

-- condition: in one hour, the boat goes 9 km along the stream
def cond1 (vs : ℝ) := 7 + vs = 9

-- condition: the speed of the boat in still water (7 km/hr)
def speed_still_water := 7

-- theorem to prove: the distance the boat goes against the stream in one hour
theorem distance_against_stream (vs : ℝ) (h : cond1 vs) : 
  (speed_still_water - vs) * 1 = 5 :=
by
  rw [speed_still_water, mul_one]
  sorry

end NUMINAMATH_GPT_distance_against_stream_l92_9290


namespace NUMINAMATH_GPT_waiting_probability_no_more_than_10_seconds_l92_9243

def total_cycle_time : ℕ := 30 + 10 + 40
def proceed_during_time : ℕ := 40 -- green time
def yellow_time : ℕ := 10

theorem waiting_probability_no_more_than_10_seconds :
  (proceed_during_time + yellow_time + yellow_time) / total_cycle_time = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_waiting_probability_no_more_than_10_seconds_l92_9243


namespace NUMINAMATH_GPT_range_of_f_l92_9279

noncomputable def f (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

theorem range_of_f : Set.range f = {y : ℝ | y ≠ 3} :=
sorry

end NUMINAMATH_GPT_range_of_f_l92_9279


namespace NUMINAMATH_GPT_series_converges_to_half_l92_9217

noncomputable def series_value : ℝ :=
  ∑' (n : ℕ), (n^4 + 3*n^3 + 10*n + 10) / (3^n * (n^4 + 4))

theorem series_converges_to_half : series_value = 1 / 2 :=
  sorry

end NUMINAMATH_GPT_series_converges_to_half_l92_9217


namespace NUMINAMATH_GPT_gcd_pow_sub_l92_9215

theorem gcd_pow_sub (a b : ℕ) (ha : a = 2000) (hb : b = 1990) :
  Nat.gcd (2^a - 1) (2^b - 1) = 1023 :=
sorry

end NUMINAMATH_GPT_gcd_pow_sub_l92_9215


namespace NUMINAMATH_GPT_balls_in_boxes_l92_9212

theorem balls_in_boxes : 
  let balls := 4
  let boxes := 3
  (boxes^balls = 81) :=
by sorry

end NUMINAMATH_GPT_balls_in_boxes_l92_9212


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l92_9245

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Conditions of the problem
def a : ℝ := 3
def b : ℝ := -6
def c : ℝ := 4

-- The proof statement
theorem quadratic_no_real_roots : discriminant a b c < 0 :=
by
  -- Calculate the discriminant to show it's negative
  let Δ := discriminant a b c
  show Δ < 0
  sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_l92_9245


namespace NUMINAMATH_GPT_eval_expression_l92_9272

theorem eval_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l92_9272


namespace NUMINAMATH_GPT_arithmetic_sequence_eleventh_term_l92_9208

theorem arithmetic_sequence_eleventh_term 
  (a d : ℚ)
  (h_sum_first_six : 6 * a + 15 * d = 30)
  (h_seventh_term : a + 6 * d = 10) : 
    a + 10 * d = 110 / 7 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_eleventh_term_l92_9208


namespace NUMINAMATH_GPT_largest_gcd_sum780_l92_9291

theorem largest_gcd_sum780 (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 780) : 
  ∃ d, d = Nat.gcd a b ∧ d ≤ 390 ∧ (∀ (d' : ℕ), d' = Nat.gcd a b → d' ≤ 390) :=
sorry

end NUMINAMATH_GPT_largest_gcd_sum780_l92_9291


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l92_9265

theorem sufficient_but_not_necessary (a : ℝ) (h1 : a > 0) (h2 : |a| > 0 → a > 0 ∨ a < 0) : 
  (a > 0 → |a| > 0) ∧ (¬(|a| > 0 → a > 0)) := 
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l92_9265


namespace NUMINAMATH_GPT_action_movies_rented_l92_9276

-- Defining the conditions as hypotheses
theorem action_movies_rented (a M A D : ℝ) (h1 : 0.64 * M = 10 * a)
                             (h2 : D = 5 * A)
                             (h3 : D + A = 0.36 * M) :
    A = 0.9375 * a :=
sorry

end NUMINAMATH_GPT_action_movies_rented_l92_9276


namespace NUMINAMATH_GPT_range_of_a_l92_9221

theorem range_of_a (a : ℝ) (h : a < 1) : ∀ x : ℝ, |x - 4| + |x - 5| > a :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l92_9221


namespace NUMINAMATH_GPT_parametric_hyperbola_l92_9238

theorem parametric_hyperbola (t : ℝ) (ht : t ≠ 0) : 
  let x := t + 1 / t
  let y := t - 1 / t
  x^2 - y^2 = 4 :=
by
  let x := t + 1 / t
  let y := t - 1 / t
  sorry

end NUMINAMATH_GPT_parametric_hyperbola_l92_9238


namespace NUMINAMATH_GPT_workout_total_correct_l92_9203

structure Band := 
  (A : ℕ) 
  (B : ℕ) 
  (C : ℕ)

structure Equipment := 
  (leg_weight_squat : ℕ) 
  (dumbbell : ℕ) 
  (leg_weight_lunge : ℕ) 
  (kettlebell : ℕ)

def total_weight (bands : Band) (equip : Equipment) : ℕ := 
  let squat_total := bands.A + bands.B + bands.C + (2 * equip.leg_weight_squat) + equip.dumbbell
  let lunge_total := bands.A + bands.C + (2 * equip.leg_weight_lunge) + equip.kettlebell
  squat_total + lunge_total

theorem workout_total_correct (bands : Band) (equip : Equipment) : 
  bands = ⟨7, 5, 3⟩ → 
  equip = ⟨10, 15, 8, 18⟩ → 
  total_weight bands equip = 94 :=
by 
  -- Insert your proof steps here
  sorry

end NUMINAMATH_GPT_workout_total_correct_l92_9203


namespace NUMINAMATH_GPT_hyperbola_foci_l92_9288

/-- The coordinates of the foci of the hyperbola y^2 / 3 - x^2 = 1 are (0, ±2). -/
theorem hyperbola_foci (x y : ℝ) :
  x^2 - (y^2 / 3) = -1 → (0 = x ∧ (y = 2 ∨ y = -2)) :=
sorry

end NUMINAMATH_GPT_hyperbola_foci_l92_9288


namespace NUMINAMATH_GPT_find_fathers_age_l92_9261

noncomputable def sebastian_age : ℕ := 40
noncomputable def age_difference : ℕ := 10
noncomputable def sum_ages_five_years_ago_ratio : ℚ := (3 : ℚ) / 4

theorem find_fathers_age 
  (sebastian_age : ℕ) 
  (age_difference : ℕ) 
  (sum_ages_five_years_ago_ratio : ℚ) 
  (h1 : sebastian_age = 40) 
  (h2 : age_difference = 10) 
  (h3 : sum_ages_five_years_ago_ratio = 3 / 4) 
: ∃ father_age : ℕ, father_age = 85 :=
sorry

end NUMINAMATH_GPT_find_fathers_age_l92_9261


namespace NUMINAMATH_GPT_smallest_positive_b_factors_l92_9251

theorem smallest_positive_b_factors (b : ℤ) : 
  (∃ p q : ℤ, x^2 + b * x + 2016 = (x + p) * (x + q) ∧ p + q = b ∧ p * q = 2016 ∧ p > 0 ∧ q > 0) → b = 95 := 
by {
  sorry
}

end NUMINAMATH_GPT_smallest_positive_b_factors_l92_9251


namespace NUMINAMATH_GPT_temperature_on_friday_is_35_l92_9252

variables (M T W Th F : ℤ)

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem temperature_on_friday_is_35
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : M = 43)
  (h4 : is_odd M)
  (h5 : is_odd T)
  (h6 : is_odd W)
  (h7 : is_odd Th)
  (h8 : is_odd F) : 
  F = 35 :=
sorry

end NUMINAMATH_GPT_temperature_on_friday_is_35_l92_9252


namespace NUMINAMATH_GPT_steve_can_answer_38_questions_l92_9257

theorem steve_can_answer_38_questions (total_questions S : ℕ) 
  (h1 : total_questions = 45)
  (h2 : total_questions - S = 7) :
  S = 38 :=
by {
  -- The proof goes here
  sorry
}

end NUMINAMATH_GPT_steve_can_answer_38_questions_l92_9257


namespace NUMINAMATH_GPT_binary_addition_is_correct_l92_9274

-- Definitions for the binary numbers
def bin1 := "10101"
def bin2 := "11"
def bin3 := "1010"
def bin4 := "11100"
def bin5 := "1101"

-- Function to convert binary string to nat (using built-in functionality)
def binStringToNat (s : String) : Nat :=
  String.foldl (fun n c => 2 * n + if c = '1' then 1 else 0) 0 s

-- Binary numbers converted to nat
def n1 := binStringToNat bin1
def n2 := binStringToNat bin2
def n3 := binStringToNat bin3
def n4 := binStringToNat bin4
def n5 := binStringToNat bin5

-- The expected result in nat
def expectedSum := binStringToNat "11101101"

-- Proof statement
theorem binary_addition_is_correct : n1 + n2 + n3 + n4 + n5 = expectedSum :=
  sorry

end NUMINAMATH_GPT_binary_addition_is_correct_l92_9274


namespace NUMINAMATH_GPT_hermia_elected_probability_l92_9256

-- Define the problem statement and conditions in Lean 4
noncomputable def probability_hermia_elected (n : ℕ) (h_odd : (n % 2 = 1)) (h_pos : n > 0) : ℚ :=
  if n = 1 then 1 else (2^n - 1) / (n * 2^(n-1))

-- Lean theorem statement
theorem hermia_elected_probability (n : ℕ) (h_odd : (n % 2 = 1)) (h_pos : n > 0) : 
  probability_hermia_elected n h_odd h_pos = (2^n - 1) / (n * 2^(n-1)) :=
by
  sorry

end NUMINAMATH_GPT_hermia_elected_probability_l92_9256


namespace NUMINAMATH_GPT_ratio_of_square_areas_l92_9280

noncomputable def ratio_of_areas (s : ℝ) : ℝ := s^2 / (4 * s^2)

theorem ratio_of_square_areas (s : ℝ) (h : s ≠ 0) : ratio_of_areas s = 1 / 4 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_square_areas_l92_9280


namespace NUMINAMATH_GPT_initial_mixture_volume_is_165_l92_9298

noncomputable def initial_volume_of_mixture (initial_milk_volume initial_water_volume water_added final_milk_water_ratio : ℕ) : ℕ :=
  if (initial_milk_volume + initial_water_volume) = 5 * (initial_milk_volume / 3) &&
     initial_water_volume = 2 * (initial_milk_volume / 3) &&
     water_added = 66 &&
     final_milk_water_ratio = 3 / 4 then
    5 * (initial_milk_volume / 3)
  else
    0

theorem initial_mixture_volume_is_165 :
  ∀ initial_milk_volume initial_water_volume water_added final_milk_water_ratio,
    initial_volume_of_mixture initial_milk_volume initial_water_volume water_added final_milk_water_ratio = 165 :=
by
  intros
  sorry

end NUMINAMATH_GPT_initial_mixture_volume_is_165_l92_9298


namespace NUMINAMATH_GPT_geom_seq_sum_5_terms_l92_9294

theorem geom_seq_sum_5_terms (a : ℕ → ℝ) (q : ℝ) (h1 : a 4 = 8 * a 1) (h2 : 2 * (a 2 + 1) = a 1 + a 3) (h_q : q = 2) :
    a 1 * (1 - q^5) / (1 - q) = 62 :=
by
    sorry

end NUMINAMATH_GPT_geom_seq_sum_5_terms_l92_9294


namespace NUMINAMATH_GPT_find_number_l92_9242

theorem find_number (x : ℕ) (h : 3 * x = 2 * 51 - 3) : x = 33 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l92_9242


namespace NUMINAMATH_GPT_gcd_a2_13a_36_a_6_eq_6_l92_9227

namespace GCDProblem

variable (a : ℕ)
variable (h : ∃ k, a = 1632 * k)

theorem gcd_a2_13a_36_a_6_eq_6 (ha : ∃ k : ℕ, a = 1632 * k) : 
  Int.gcd (a^2 + 13 * a + 36 : Int) (a + 6 : Int) = 6 := by
  sorry

end GCDProblem

end NUMINAMATH_GPT_gcd_a2_13a_36_a_6_eq_6_l92_9227


namespace NUMINAMATH_GPT_ratio_between_second_and_third_l92_9258

noncomputable def ratio_second_third : ℚ := sorry

theorem ratio_between_second_and_third (A B C : ℕ) (h₁ : A + B + C = 98) (h₂ : A * 3 = B * 2) (h₃ : B = 30) :
  ratio_second_third = 5 / 8 := sorry

end NUMINAMATH_GPT_ratio_between_second_and_third_l92_9258


namespace NUMINAMATH_GPT_common_ratio_of_series_l92_9223

theorem common_ratio_of_series (a1 a2 : ℚ) (h1 : a1 = 5/6) (h2 : a2 = -4/9) :
  (a2 / a1) = -8/15 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_of_series_l92_9223


namespace NUMINAMATH_GPT_remove_one_piece_l92_9205

theorem remove_one_piece (pieces : Finset (Fin 8 × Fin 8)) (h_card : pieces.card = 15)
  (h_row : ∀ r : Fin 8, ∃ c, (r, c) ∈ pieces)
  (h_col : ∀ c : Fin 8, ∃ r, (r, c) ∈ pieces) :
  ∃ pieces' : Finset (Fin 8 × Fin 8), pieces'.card = 14 ∧ 
  (∀ r : Fin 8, ∃ c, (r, c) ∈ pieces') ∧ 
  (∀ c : Fin 8, ∃ r, (r, c) ∈ pieces') :=
sorry

end NUMINAMATH_GPT_remove_one_piece_l92_9205


namespace NUMINAMATH_GPT_vector_subtraction_result_l92_9239

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-3, 4)

theorem vector_subtraction_result :
  2 • a - b = (7, -2) :=
by
  simp [a, b]
  sorry

end NUMINAMATH_GPT_vector_subtraction_result_l92_9239


namespace NUMINAMATH_GPT_grasshopper_flea_adjacency_l92_9293

-- We assume that grid cells are indexed by pairs of integers (i.e., positions in ℤ × ℤ)
-- Red cells and white cells are represented as sets of these positions
variable (red_cells : Set (ℤ × ℤ))
variable (white_cells : Set (ℤ × ℤ))

-- We define that the grasshopper can only jump between red cells
def grasshopper_jump (pos : ℤ × ℤ) (new_pos : ℤ × ℤ) : Prop :=
  pos ∈ red_cells ∧ new_pos ∈ red_cells ∧ (pos.1 = new_pos.1 ∨ pos.2 = new_pos.2)

-- We define that the flea can only jump between white cells
def flea_jump (pos : ℤ × ℤ) (new_pos : ℤ × ℤ) : Prop :=
  pos ∈ white_cells ∧ new_pos ∈ white_cells ∧ (pos.1 = new_pos.1 ∨ pos.2 = new_pos.2)

-- Main theorem to be proved
theorem grasshopper_flea_adjacency (g_start : ℤ × ℤ) (f_start : ℤ × ℤ) :
    g_start ∈ red_cells → f_start ∈ white_cells →
    ∃ g1 g2 g3 f1 f2 f3 : ℤ × ℤ,
    (
      grasshopper_jump red_cells g_start g1 ∧
      grasshopper_jump red_cells g1 g2 ∧
      grasshopper_jump red_cells g2 g3
    ) ∧ (
      flea_jump white_cells f_start f1 ∧
      flea_jump white_cells f1 f2 ∧
      flea_jump white_cells f2 f3
    ) ∧
    (abs (g3.1 - f3.1) + abs (g3.2 - f3.2) = 1) :=
  sorry

end NUMINAMATH_GPT_grasshopper_flea_adjacency_l92_9293


namespace NUMINAMATH_GPT_ratio_of_abc_l92_9213

theorem ratio_of_abc (a b c : ℝ) (h1 : a ≠ 0) (h2 : 14 * (a^2 + b^2 + c^2) = (a + 2 * b + 3 * c)^2) : a / b = 1 / 2 ∧ a / c = 1 / 3 := 
sorry

end NUMINAMATH_GPT_ratio_of_abc_l92_9213
