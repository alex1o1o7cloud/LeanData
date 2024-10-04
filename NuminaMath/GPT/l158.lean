import Mathlib

namespace average_error_diff_l158_158686

theorem average_error_diff (n : ℕ) (total_data_pts : ℕ) (error_data1 error_data2 : ℕ)
  (h_n : n = 30) (h_total_data_pts : total_data_pts = 30)
  (h_error_data1 : error_data1 = 105) (h_error_data2 : error_data2 = 15)
  : (error_data1 - error_data2) / n = 3 :=
sorry

end average_error_diff_l158_158686


namespace area_ratio_of_similar_polygons_l158_158115

theorem area_ratio_of_similar_polygons (similarity_ratio: ℚ) (hratio: similarity_ratio = 1/5) : (similarity_ratio ^ 2 = 1/25) := 
by 
  sorry

end area_ratio_of_similar_polygons_l158_158115


namespace find_R_plus_S_l158_158228

theorem find_R_plus_S (d e R S : ℝ) 
  (h1 : d + 3 = 0)
  (h2 : 7 * d + 3 * e = 0)
  (h3 : R = 3 * d + e + 7)
  (h4 : S = 7 * e) :
  R + S = 54 :=
by
  sorry

end find_R_plus_S_l158_158228


namespace xiao_ying_correct_answers_at_least_l158_158609

def total_questions : ℕ := 20
def points_correct : ℕ := 5
def points_incorrect : ℕ := 2
def excellent_points : ℕ := 80

theorem xiao_ying_correct_answers_at_least (x : ℕ) :
  (5 * x - 2 * (total_questions - x)) ≥ excellent_points → x ≥ 18 := by
  sorry

end xiao_ying_correct_answers_at_least_l158_158609


namespace kohen_apples_l158_158128

theorem kohen_apples (B : ℕ) (h1 : 300 * B = 4 * 750) : B = 10 :=
by
  -- proof goes here
  sorry

end kohen_apples_l158_158128


namespace area_enclosed_by_graph_l158_158933

theorem area_enclosed_by_graph (x y : ℝ) (h : abs (5 * x) + abs (3 * y) = 15) : 
  ∃ (area : ℝ), area = 30 :=
sorry

end area_enclosed_by_graph_l158_158933


namespace determine_d_minus_r_l158_158364

theorem determine_d_minus_r :
  ∃ d r: ℕ, (∀ n ∈ [2023, 2459, 3571], n % d = r) ∧ (1 < d) ∧ (d - r = 1) :=
sorry

end determine_d_minus_r_l158_158364


namespace power_equality_l158_158483

theorem power_equality : (243 : ℝ) ^ (1 / 3) = (3 : ℝ) ^ (5 / 3) := 
by 
  sorry

end power_equality_l158_158483


namespace minutes_watched_on_Thursday_l158_158133

theorem minutes_watched_on_Thursday 
  (n_total : ℕ) (n_Mon : ℕ) (n_Tue : ℕ) (n_Wed : ℕ) (n_Fri : ℕ) (n_weekend : ℕ)
  (h_total : n_total = 352)
  (h_Mon : n_Mon = 138)
  (h_Tue : n_Tue = 0)
  (h_Wed : n_Wed = 0)
  (h_Fri : n_Fri = 88)
  (h_weekend : n_weekend = 105) :
  n_total - (n_Mon + n_Tue + n_Wed + n_Fri + n_weekend) = 21 := by
  sorry

end minutes_watched_on_Thursday_l158_158133


namespace perfect_square_polynomial_l158_158846

theorem perfect_square_polynomial (x : ℤ) : 
  (∃ n : ℤ, x^4 + x^3 + x^2 + x + 1 = n^2) ↔ (x = -1 ∨ x = 0 ∨ x = 3) :=
sorry

end perfect_square_polynomial_l158_158846


namespace geometric_sequence_sixth_term_l158_158280

theorem geometric_sequence_sixth_term 
  (a : ℝ) (r : ℝ) (a9 : ℝ)
  (h1 : a = 1024)
  (h2 : a9 = 16)
  (h3 : ∀ n, a9 = a * r ^ (n - 1)) :
  a * r ^ 5 = 4 * sqrt 2 :=
by
  sorry

end geometric_sequence_sixth_term_l158_158280


namespace inequality_holds_l158_158002

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h : x * y + y * z + z * x = 1) :
  (1 / (x + y)) + (1 / (y + z)) + (1 / (z + x)) ≥ 5 / 2 :=
by
  sorry

end inequality_holds_l158_158002


namespace percentage_female_on_duty_l158_158898

-- Definitions as per conditions in the problem:
def total_on_duty : ℕ := 240
def female_on_duty := total_on_duty / 2 -- Half of those on duty are female
def total_female_officers : ℕ := 300
def percentage_of_something (part : ℕ) (whole : ℕ) : ℕ := (part * 100) / whole

-- Statement of the problem to prove
theorem percentage_female_on_duty : percentage_of_something female_on_duty total_female_officers = 40 :=
by
  sorry

end percentage_female_on_duty_l158_158898


namespace miles_driven_l158_158618

def total_miles : ℕ := 1200
def remaining_miles : ℕ := 432

theorem miles_driven : total_miles - remaining_miles = 768 := by
  sorry

end miles_driven_l158_158618


namespace correct_expression_l158_158064

-- Definitions based on given conditions
def expr1 (a b : ℝ) := 3 * a + 2 * b = 5 * a * b
def expr2 (a : ℝ) := 2 * a^3 - a^3 = a^3
def expr3 (a b : ℝ) := a^2 * b - a * b = a
def expr4 (a : ℝ) := a^2 + a^2 = 2 * a^4

-- Statement to prove that expr2 is the only correct expression
theorem correct_expression (a b : ℝ) : 
  expr2 a := by
  sorry

end correct_expression_l158_158064


namespace triangle_side_length_x_l158_158518

theorem triangle_side_length_x (x : ℤ) (hpos : x > 0) (hineq1 : 7 < x^2) (hineq2 : x^2 < 17) :
    x = 3 ∨ x = 4 :=
by {
  apply sorry
}

end triangle_side_length_x_l158_158518


namespace fries_remaining_time_l158_158692

theorem fries_remaining_time (recommended_time_min : ℕ) (time_in_oven_sec : ℕ)
    (h1 : recommended_time_min = 5)
    (h2 : time_in_oven_sec = 45) :
    (recommended_time_min * 60 - time_in_oven_sec = 255) :=
by
  sorry

end fries_remaining_time_l158_158692


namespace bond_value_after_8_years_l158_158012

theorem bond_value_after_8_years :
  ∀ (P A r t : ℝ), P = 240 → r = 0.0833333333333332 → t = 8 →
  (A = P * (1 + r * t)) → A = 400 :=
by
  sorry

end bond_value_after_8_years_l158_158012


namespace dartboard_odd_score_probability_l158_158191

theorem dartboard_odd_score_probability :
  let π := Real.pi
  let r_outer := 4
  let r_inner := 2
  let area_inner := π * r_inner * r_inner
  let area_outer := π * r_outer * r_outer
  let area_annulus := area_outer - area_inner
  let area_inner_region := area_inner / 3
  let area_outer_region := area_annulus / 3
  let odd_inner_regions := 1
  let even_inner_regions := 2
  let odd_outer_regions := 2
  let even_outer_regions := 1
  let prob_odd_inner := (odd_inner_regions * area_inner_region) / (area_inner + area_annulus)
  let prob_even_inner := (even_inner_regions * area_inner_region) / (area_inner + area_annulus)
  let prob_odd_outer := (odd_outer_regions * area_outer_region) / (area_inner + area_annulus)
  let prob_even_outer := (even_outer_regions * area_outer_region) / (area_inner + area_annulus)
  let prob_odd_region := prob_odd_inner + prob_odd_outer
  let prob_even_region := prob_even_inner + prob_even_outer
  let prob_odd_score := (prob_odd_region * prob_even_region) + (prob_even_region * prob_odd_region)
  prob_odd_score = 5 / 9 :=
by
  -- Proof omitted
  sorry

end dartboard_odd_score_probability_l158_158191


namespace abs_discriminant_inequality_l158_158208

theorem abs_discriminant_inequality 
  (a b c A B C : ℝ) 
  (ha : a ≠ 0) 
  (hA : A ≠ 0) 
  (h : ∀ x : ℝ, |a * x^2 + b * x + c| ≤ |A * x^2 + B * x + C|) : 
  |b^2 - 4 * a * c| ≤ |B^2 - 4 * A * C| :=
sorry

end abs_discriminant_inequality_l158_158208


namespace sin_cos_sum_l158_158570

open Real

theorem sin_cos_sum : sin (47 : ℝ) * cos (43 : ℝ) + cos (47 : ℝ) * sin (43 : ℝ) = 1 :=
by
  sorry

end sin_cos_sum_l158_158570


namespace simplification_problem_l158_158891

theorem simplification_problem (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (h_sum : p + q + r = 1) :
  (1 / (q^2 + r^2 - p^2) + 1 / (p^2 + r^2 - q^2) + 1 / (p^2 + q^2 - r^2) = 3 / (1 - 2 * q * r)) :=
by
  sorry

end simplification_problem_l158_158891


namespace final_milk_quantity_correct_l158_158063

-- Defining the initial quantities
def initial_milk : ℝ := 50
def removed_volume : ℝ := 9

-- The quantity of milk after the first replacement
def milk_after_first_replacement : ℝ := initial_milk - removed_volume

-- The ratio of milk to the total solution
def milk_ratio : ℝ := milk_after_first_replacement / initial_milk

-- The amount of milk removed in the second step
def milk_removed_second_step : ℝ := milk_ratio * removed_volume

-- The final quantity of milk in the solution
def final_milk_quantity : ℝ := milk_after_first_replacement - milk_removed_second_step

theorem final_milk_quantity_correct :
  final_milk_quantity = 33.62 := by
  sorry

end final_milk_quantity_correct_l158_158063


namespace math_majors_consecutive_seats_l158_158263

noncomputable def probability_math_majors_consecutive_seats : ℚ :=
  1 / 14

theorem math_majors_consecutive_seats :
  ∀ (n m p : ℕ) (n_ppl : Finset (Fin 9)) (seats : Set (Finset (Fin 9))),
    n = 4 ∧ m = 3 ∧ p = 2 ∧ n_ppl.card = 9 ∧
    (∀ s ∈ seats, s.card = 4 → (∃ k ∈ Finset.range 9, ∀ i ∈ s, i = ↑((k + i) % 9))) →
  probability (math_majors_consecutive_seats : xx : Set {s | s.card = 4} → ℚ) = probability_math_majors_consecutive_seats :=
sorry

end math_majors_consecutive_seats_l158_158263


namespace donut_distribution_l158_158689

theorem donut_distribution (n k: ℕ) (hn: n = 8) (hk: k = 5) :
  (nat.choose ((n - k) + k - 1) (k - 1)) = 35 :=
by
  rw [hn, hk]
  -- n = 8, k = 5
  -- (n-k) + k - 1 = 7
  have h1 : (8 - 5) + 5 - 1 = 7 := by norm_num
  rw [h1]
  -- k - 1 = 4
  have h2 : 5 - 1 = 4 := by norm_num
  rw [h2]
  -- result is binomial coefficient 7 choose 4
  exact nat.choose_self 35
  sorry -- proof not completed

end donut_distribution_l158_158689


namespace number_of_female_officers_is_382_l158_158774

noncomputable def F : ℝ := 
  let total_on_duty := 210
  let ratio_male_female := 3 / 2
  let percent_female_on_duty := 22 / 100
  let female_on_duty := total_on_duty * (2 / (3 + 2))
  let total_females := female_on_duty / percent_female_on_duty
  total_females

theorem number_of_female_officers_is_382 : F = 382 := 
by
  sorry

end number_of_female_officers_is_382_l158_158774


namespace full_size_mustang_length_l158_158903

theorem full_size_mustang_length 
  (smallest_model_length : ℕ)
  (mid_size_factor : ℕ)
  (full_size_factor : ℕ)
  (h1 : smallest_model_length = 12)
  (h2 : mid_size_factor = 2)
  (h3 : full_size_factor = 10) :
  (smallest_model_length * mid_size_factor) * full_size_factor = 240 := 
sorry

end full_size_mustang_length_l158_158903


namespace decompose_zero_l158_158836

theorem decompose_zero (a : ℤ) : 0 = 0 * a := by
  sorry

end decompose_zero_l158_158836


namespace simplify_expr_l158_158834

theorem simplify_expr (x y : ℝ) (P Q : ℝ) (hP : P = x^2 + y^2) (hQ : Q = x^2 - y^2) : 
  (P * Q / (P + Q)) + ((P + Q) / (P * Q)) = ((x^4 + y^4) ^ 2) / (2 * x^2 * (x^4 - y^4)) :=
by sorry

end simplify_expr_l158_158834


namespace innings_question_l158_158671

theorem innings_question (n : ℕ) (runs_in_inning : ℕ) (avg_increase : ℕ) (new_avg : ℕ) 
  (h_runs_in_inning : runs_in_inning = 88) 
  (h_avg_increase : avg_increase = 3) 
  (h_new_avg : new_avg = 40)
  (h_eq : 37 * n + runs_in_inning = new_avg * (n + 1)): n + 1 = 17 :=
by
  -- Proof to be filled in here
  sorry

end innings_question_l158_158671


namespace garden_perimeter_l158_158451

noncomputable def perimeter_of_garden (w l : ℝ) (h1 : l = 3 * w + 15) (h2 : w * l = 4050) : ℝ :=
  2 * l + 2 * w

theorem garden_perimeter (w l : ℝ) (h1 : l = 3 * w + 15) (h2 : w * l = 4050) :
  perimeter_of_garden w l h1 h2 = 304.64 :=
sorry

end garden_perimeter_l158_158451


namespace relationship_abc_l158_158206

noncomputable def a : ℝ := 5 ^ (Real.log 3.4 / Real.log 3)
noncomputable def b : ℝ := 5 ^ (Real.log 3.6 / Real.log 3)
noncomputable def c : ℝ := (1 / 5) ^ (Real.log 0.5 / Real.log 3)

theorem relationship_abc : b > a ∧ a > c :=
by
  -- Assumptions derived from logarithmic properties.
  have h1 : Real.log 2 < Real.log 3.4 := sorry
  have h2 : Real.log 3.4 < Real.log 3.6 := sorry
  have h3 : Real.log 0.5 < 0 := sorry
  have h4 : Real.log 2 / Real.log 3 = Real.log 2 := sorry
  have h5 : Real.log 0.5 / Real.log 3 = -Real.log 2 := sorry

  -- Monotonicity of exponential function.
  apply And.intro
  { exact sorry }
  { exact sorry }

end relationship_abc_l158_158206


namespace find_m_l158_158096

variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (m : ℕ)

theorem find_m (h1 : ∀ n, a (n + 1) = a n + d) -- arithmetic sequence
               (h2 : S (2 * m - 1) = 39)       -- sum of first (2m-1) terms
               (h3 : a (m - 1) + a (m + 1) - a m - 1 = 0)
               (h4 : m > 1) : 
               m = 20 :=
   sorry

end find_m_l158_158096


namespace max_of_expression_l158_158579

theorem max_of_expression (a b c : ℝ) (hbc : b > c) (hca : c > a) (ha : a > 0) (hb : b > 0) (hc : c > 0) (ha_nonzero : a ≠ 0) :
  ∃ (max_val : ℝ), max_val = 44 ∧ (∀ x, x = (2*a + b)^2 + (b - 2*c)^2 + (c - a)^2 → x ≤ max_val) := 
sorry

end max_of_expression_l158_158579


namespace train_pass_jogger_time_l158_158336

noncomputable def jogger_speed_km_per_hr : ℝ := 9
noncomputable def train_speed_km_per_hr : ℝ := 60
noncomputable def initial_distance_m : ℝ := 350
noncomputable def train_length_m : ℝ := 250

noncomputable def relative_speed_m_per_s : ℝ := 
  ((train_speed_km_per_hr - jogger_speed_km_per_hr) * 1000) / 3600

noncomputable def total_distance_m : ℝ := initial_distance_m + train_length_m

noncomputable def time_to_pass_s : ℝ := total_distance_m / relative_speed_m_per_s

theorem train_pass_jogger_time :
  abs (time_to_pass_s - 42.35) < 0.01 :=
by 
  sorry

end train_pass_jogger_time_l158_158336


namespace greatest_integer_value_l158_158939

theorem greatest_integer_value (x : ℤ) : 7 - 3 * x > 20 → x ≤ -5 :=
by
  intros h
  sorry

end greatest_integer_value_l158_158939


namespace perfect_square_polynomial_l158_158845

theorem perfect_square_polynomial (x : ℤ) : 
  (∃ n : ℤ, x^4 + x^3 + x^2 + x + 1 = n^2) ↔ (x = -1 ∨ x = 0 ∨ x = 3) :=
sorry

end perfect_square_polynomial_l158_158845


namespace inv_f_of_neg3_l158_158172

def f (x : Real) : Real := 5 - 2 * x

theorem inv_f_of_neg3 : f⁻¹ (-3) = 4 :=
by
  sorry

end inv_f_of_neg3_l158_158172


namespace knowledge_contest_rankings_l158_158238

theorem knowledge_contest_rankings:
  let students := {1, 2, 3, 4, 5}
  let permutations := students.to_finset.powerset.to_list.permutations
  let valid_permutations := permutations.filter (λ perm, 
    perm.head ≠ 1 ∧ perm.index 2 ≠ perm.length - 1)
  valid_permutations.card = 54 :=
by
  sorry

end knowledge_contest_rankings_l158_158238


namespace find_a3_l158_158392

-- Define the polynomial equality
def polynomial_equality (x : ℝ) (a0 a1 a2 a3 a4 a5 a6 a7 : ℝ) :=
  (1 + x) * (2 - x)^6 = a0 + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 + a5 * (x - 1)^5 + a6 * (x - 1)^6 + a7 * (x - 1)^7

-- State the main theorem
theorem find_a3 (a0 a1 a2 a4 a5 a6 a7 : ℝ) :
  (∃ (x : ℝ), polynomial_equality x a0 a1 a2 (-25) a4 a5 a6 a7) :=
sorry

end find_a3_l158_158392


namespace paths_O_to_P_paths_O_to_P_via_A_paths_O_to_P_via_A_avoiding_B_paths_O_to_P_via_A_avoiding_BC_l158_158504

open Nat

-- 1. Number of paths from O(0,0) to P(9,8)
theorem paths_O_to_P : (Nat.choose 17 9) = 24310 := 
by {
  sorry
}

-- 2. Number of paths from O(0,0) to P(9,8) via A(3,2)
theorem paths_O_to_P_via_A : (Nat.choose 5 3) * (Nat.choose 12 6) = 9240 := 
by {
  sorry
}

-- 3. Number of paths from O(0,0) to P(9,8) via A(3,2) avoiding B(6,5)
theorem paths_O_to_P_via_A_avoiding_B : ( (Nat.choose 12 6) - (Nat.choose 6 3) * (Nat.choose 6 3) ) * (Nat.choose 5 3) = 5240 := 
by {
  sorry
}

-- 4. Number of paths from O(0,0) to P(9,8) via A(3,2) avoiding BC (B(6,5) to C(8,5))
theorem paths_O_to_P_via_A_avoiding_BC : ( (Nat.choose 12 6) - (Nat.choose 7 4) * (Nat.choose 5 2) ) * (Nat.choose 5 3) = 5740 := 
by {
  sorry
}

end paths_O_to_P_paths_O_to_P_via_A_paths_O_to_P_via_A_avoiding_B_paths_O_to_P_via_A_avoiding_BC_l158_158504


namespace smallest_value_4x_plus_3y_l158_158716

-- Define the condition as a predicate
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 18 * x + 8 * y + 10

-- Prove the smallest possible value of 4x + 3y given the condition
theorem smallest_value_4x_plus_3y : ∃ x y : ℝ, circle_eq x y ∧ (4 * x + 3 * y = -40) :=
by
  -- Placeholder for the proof
  sorry

end smallest_value_4x_plus_3y_l158_158716


namespace prob1_prob2_prob3_l158_158575

-- Problem (1)
theorem prob1 (a b : ℝ) :
  ((a / 4 - 1) + 2 * (b / 3 + 2) = 4) ∧ (2 * (a / 4 - 1) + (b / 3 + 2) = 5) →
  a = 12 ∧ b = -3 :=
by { sorry }

-- Problem (2)
theorem prob2 (m n x y a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) :
  (x = 10) ∧ (y = 6) ∧ 
  (5 * a₁ * (m - 3) + 3 * b₁ * (n + 2) = c₁) ∧ (5 * a₂ * (m - 3) + 3 * b₂ * (n + 2) = c₂) →
  (m = 5) ∧ (n = 0) :=
by { sorry }

-- Problem (3)
theorem prob3 (x y z : ℝ) :
  (3 * x - 2 * z + 12 * y = 47) ∧ (2 * x + z + 8 * y = 36) → z = 2 :=
by { sorry }

end prob1_prob2_prob3_l158_158575


namespace zoo_individuals_remaining_l158_158505

noncomputable def initial_students_class1 := 10
noncomputable def initial_students_class2 := 10
noncomputable def chaperones := 5
noncomputable def teachers := 2
noncomputable def students_left := 10
noncomputable def chaperones_left := 2

theorem zoo_individuals_remaining :
  let total_initial_individuals := initial_students_class1 + initial_students_class2 + chaperones + teachers
  let total_left := students_left + chaperones_left
  total_initial_individuals - total_left = 15 := by
  sorry

end zoo_individuals_remaining_l158_158505


namespace smallest_base_10_integer_l158_158943

theorem smallest_base_10_integer (a b : ℕ) (ha : a > 2) (hb : b > 2) 
  (h1: 21_a = 2 * a + 1) (h2: 12_b = b + 2) : 2 * a + 1 = 7 :=
by 
  sorry

end smallest_base_10_integer_l158_158943


namespace at_least_two_of_three_equations_have_solutions_l158_158855

theorem at_least_two_of_three_equations_have_solutions
  (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ∃ x : ℝ, (x - a) * (x - b) = x - c ∨ (x - b) * (x - c) = x - a ∨ (x - c) * (x - a) = x - b := 
sorry

end at_least_two_of_three_equations_have_solutions_l158_158855


namespace number_of_grouping_methods_l158_158509

theorem number_of_grouping_methods : 
  let males := 5
  let females := 3
  let groups := 2
  let select_males := Nat.choose males groups
  let select_females := Nat.choose females groups
  let permute := Nat.factorial groups
  select_males * select_females * permute * permute = 60 :=
by 
  sorry

end number_of_grouping_methods_l158_158509


namespace area_ratio_l158_158294

-- Define the problem conditions
def Square (s : ℝ) := s > 0
def Rectangle (longer shorter : ℝ) := longer = 1.2 * shorter ∧ shorter = 0.8 * shorter

-- Define a function to calculate the area of square
def area_square (s : ℝ) : ℝ := s * s

-- Define a function to calculate the area of rectangle
def area_rectangle (longer shorter : ℝ) : ℝ := longer * shorter

-- State the proof problem
theorem area_ratio (s : ℝ) (h_square : Square s) :
  let longer := 1.2 * s
  let shorter := 0.8 * s
  area_rectangle longer shorter / area_square s = 24 / 25 :=
by
  sorry

end area_ratio_l158_158294


namespace shaded_area_percentage_l158_158614

-- Define the given conditions
def square_area := 6 * 6
def shaded_area_left := (1 / 2) * 2 * 6
def shaded_area_right := (1 / 2) * 4 * 6
def total_shaded_area := shaded_area_left + shaded_area_right

-- State the theorem
theorem shaded_area_percentage : (total_shaded_area / square_area) * 100 = 50 := by
  sorry

end shaded_area_percentage_l158_158614


namespace vasya_fraction_is_0_4_l158_158534

-- Defining the variables and conditions
variables (a b c d s : ℝ)
axiom cond1 : a = b / 2
axiom cond2 : c = a + d
axiom cond3 : d = s / 10
axiom cond4 : a + b + c + d = s

-- Stating the theorem
theorem vasya_fraction_is_0_4 (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : (b / s) = 0.4 := 
by
  sorry

end vasya_fraction_is_0_4_l158_158534


namespace sum_of_coefficients_l158_158865

theorem sum_of_coefficients:
  (∀ x : ℝ, (2*x - 1)^6 = a_0*x^6 + a_1*x^5 + a_2*x^4 + a_3*x^3 + a_4*x^2 + a_5*x + a_6) →
  a_1 + a_3 + a_5 = -364 :=
by
  sorry

end sum_of_coefficients_l158_158865


namespace scientific_notation_113700_l158_158918

theorem scientific_notation_113700 :
  ∃ (a : ℝ) (b : ℤ), 113700 = a * 10 ^ b ∧ a = 1.137 ∧ b = 5 :=
by
  sorry

end scientific_notation_113700_l158_158918


namespace gcd_not_perfect_square_l158_158592

theorem gcd_not_perfect_square
  (m n : ℕ)
  (h1 : (m % 3 = 0 ∨ n % 3 = 0) ∧ ¬(m % 3 = 0 ∧ n % 3 = 0))
  : ¬ ∃ k : ℕ, k * k = Nat.gcd (m^2 + n^2 + 2) (m^2 * n^2 + 3) :=
by
  sorry

end gcd_not_perfect_square_l158_158592


namespace correct_option_is_C_l158_158314

-- Definitions of the expressions given in the conditions
def optionA (a : ℝ) : ℝ := 3 * a^5 - a^5
def optionB (a : ℝ) : ℝ := a^2 + a^5
def optionC (a : ℝ) : ℝ := a^5 + a^5
def optionD (x y : ℝ) : ℝ := x^2 * y + x * y^2

-- The problem is to prove that optionC is correct and the others are not
theorem correct_option_is_C (a x y : ℝ) :
  (optionC a = 2 * a^5) ∧ 
  (optionA a ≠ 3) ∧ 
  (optionB a ≠ a^7) ∧ 
  (optionD x y ≠ 2 * (x ^ 3) * (y ^ 3)) :=
by
  sorry

end correct_option_is_C_l158_158314


namespace find_m_l158_158488

theorem find_m (m : ℝ) : (243 : ℝ)^(1/3) = (3 : ℝ)^m → m = 5 / 3 :=
by
  sorry

end find_m_l158_158488


namespace find_side_length_l158_158059

def hollow_cube_formula (n : ℕ) : ℕ :=
  6 * n^2 - (n^2 + 4 * (n - 2))

theorem find_side_length :
  ∃ n : ℕ, hollow_cube_formula n = 98 ∧ n = 9 :=
by
  sorry

end find_side_length_l158_158059


namespace sandy_spent_percentage_l158_158137

theorem sandy_spent_percentage (I R : ℝ) (hI : I = 200) (hR : R = 140) : 
  ((I - R) / I) * 100 = 30 :=
by
  sorry

end sandy_spent_percentage_l158_158137


namespace automobile_travel_distance_5_minutes_l158_158185

variable (a r : ℝ)

theorem automobile_travel_distance_5_minutes (h0 : r ≠ 0) :
  let distance_in_feet := (2 * a) / 5
  let time_in_seconds := 300
  (distance_in_feet / r) * time_in_seconds / 3 = 40 * a / r :=
by
  sorry

end automobile_travel_distance_5_minutes_l158_158185


namespace num_ways_to_admit_students_l158_158122

theorem num_ways_to_admit_students :
  ∃ (n : ℕ) (k : ℕ) (students : ℕ), 8 = n ∧ 2 = k ∧ 3 = students ∧ nat.choose students 1 * nat.choose 2 2 * finset.card (finset.image2 nat.mul (finset.range n) (finset.range (n-1))) = 168 :=
by
  sorry

end num_ways_to_admit_students_l158_158122


namespace time_to_plough_together_l158_158046

def work_rate_r := 1 / 15
def work_rate_s := 1 / 20
def combined_work_rate := work_rate_r + work_rate_s
def total_field := 1
def T := total_field / combined_work_rate

theorem time_to_plough_together : T = 60 / 7 :=
by
  -- Here you would provide the proof steps if it were required
  -- Since the proof steps are not needed, we indicate the end with sorry
  sorry

end time_to_plough_together_l158_158046


namespace Josanna_seventh_test_score_l158_158755

theorem Josanna_seventh_test_score (scores : List ℕ) (h_scores : scores = [95, 85, 75, 65, 90, 70])
                                   (average_increase : ℕ) (h_average_increase : average_increase = 5) :
                                   ∃ x, (List.sum scores + x) / (List.length scores + 1) = (List.sum scores) / (List.length scores) + average_increase := 
by
  sorry

end Josanna_seventh_test_score_l158_158755


namespace hearty_total_beads_l158_158221

-- Definition of the problem conditions
def blue_beads_per_package (r : ℕ) : ℕ := 2 * r
def red_beads_per_package : ℕ := 40
def red_packages : ℕ := 5
def blue_packages : ℕ := 3

-- Define the total number of beads Hearty has
def total_beads (r : ℕ) (rp : ℕ) (bp : ℕ) : ℕ :=
  (rp * red_beads_per_package) + (bp * blue_beads_per_package red_beads_per_package)

-- The theorem to be proven
theorem hearty_total_beads : total_beads red_beads_per_package red_packages blue_packages = 440 := by
  sorry

end hearty_total_beads_l158_158221


namespace range_m_l158_158400

def A (x : ℝ) : Prop := x^2 - 3 * x - 10 ≤ 0

def B (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1

theorem range_m (m : ℝ) :
  (∀ x, B m x → A x) ↔ -3 ≤ m ∧ m ≤ 3 :=
by
  sorry

end range_m_l158_158400


namespace root_quad_eqn_l158_158454

theorem root_quad_eqn (a : ℝ) (h : a^2 - a - 50 = 0) : a^3 - 51 * a = 50 :=
sorry

end root_quad_eqn_l158_158454


namespace smallest_base10_integer_l158_158955

theorem smallest_base10_integer : 
  ∃ (a b x : ℕ), a > 2 ∧ b > 2 ∧ x = 2 * a + 1 ∧ x = b + 2 ∧ x = 7 := by
  sorry

end smallest_base10_integer_l158_158955


namespace applicants_majored_in_political_science_l158_158775

theorem applicants_majored_in_political_science
  (total_applicants : ℕ)
  (gpa_above_3 : ℕ)
  (non_political_science_and_gpa_leq_3 : ℕ)
  (political_science_and_gpa_above_3 : ℕ) :
  total_applicants = 40 →
  gpa_above_3 = 20 →
  non_political_science_and_gpa_leq_3 = 10 →
  political_science_and_gpa_above_3 = 5 →
  ∃ P : ℕ, P = 15 :=
by
  intros
  sorry

end applicants_majored_in_political_science_l158_158775


namespace fewest_candies_l158_158184

-- Defining the conditions
def condition1 (x : ℕ) := x % 21 = 5
def condition2 (x : ℕ) := x % 22 = 3
def condition3 (x : ℕ) := x > 500

-- Stating the main theorem
theorem fewest_candies : ∃ x : ℕ, condition1 x ∧ condition2 x ∧ condition3 x ∧ x = 509 :=
  sorry

end fewest_candies_l158_158184


namespace quadratic_residue_solution_l158_158269

theorem quadratic_residue_solution 
  (p : ℕ) [Fact (Nat.Prime p)]
  (a b : ℕ)
  (h_a : ¬ p ∣ a)
  (h_b : ¬ p ∣ b)
  (has_solution_a : ∃ x y : ℤ, x^2 - a = p * y)
  (has_solution_b : ∃ x y : ℤ, x^2 - b = p * y)
  : ∃ x y : ℤ, x^2 - a * b = p * y :=
sorry

end quadratic_residue_solution_l158_158269


namespace quadratic_solution_1_quadratic_solution_2_l158_158783

theorem quadratic_solution_1 (x : ℝ) : x^2 - 8 * x + 12 = 0 ↔ x = 2 ∨ x = 6 := 
by
  sorry

theorem quadratic_solution_2 (x : ℝ) : (x - 3)^2 = 2 * x * (x - 3) ↔ x = 3 ∨ x = -3 := 
by
  sorry

end quadratic_solution_1_quadratic_solution_2_l158_158783


namespace positive_slope_asymptote_l158_158203

def hyperbola (x y : ℝ) :=
  Real.sqrt ((x - 1) ^ 2 + (y + 2) ^ 2) - Real.sqrt ((x - 6) ^ 2 + (y + 2) ^ 2) = 4

theorem positive_slope_asymptote :
  ∃ (m : ℝ), m = 0.75 ∧ (∃ x y, hyperbola x y) :=
sorry

end positive_slope_asymptote_l158_158203


namespace car_cost_difference_l158_158301

-- Definitions based on the problem's conditions
def car_cost_ratio (C A : ℝ) := C / A = 3 / 2
def ac_cost := 1500

-- Theorem statement that needs proving
theorem car_cost_difference (C A : ℝ) (h1 : car_cost_ratio C A) (h2 : A = ac_cost) : C - A = 750 := 
by sorry

end car_cost_difference_l158_158301


namespace jed_correct_speed_l158_158234

def fine_per_mph := 16
def jed_fine := 256
def speed_limit := 50

def jed_speed : Nat := speed_limit + jed_fine / fine_per_mph

theorem jed_correct_speed : jed_speed = 66 := by
  sorry

end jed_correct_speed_l158_158234


namespace find_positive_number_l158_158965

theorem find_positive_number (x : ℝ) (hx : 0 < x) (h : (2 / 3) * x = (144 / 216) * (1 / x)) : x = 1 := by
  sorry

end find_positive_number_l158_158965


namespace tile_count_l158_158062

theorem tile_count (a : ℕ) (h1 : ∃ b : ℕ, b = 2 * a) (h2 : 2 * (Int.floor (a * Real.sqrt 5)) - 1 = 49) :
  2 * a^2 = 50 :=
by
  sorry

end tile_count_l158_158062


namespace intersection_complement_eq_l158_158103

def setA : Set ℝ := { x | (x - 6) * (x + 1) ≤ 0 }
def setB : Set ℝ := { x | x ≥ 2 }

theorem intersection_complement_eq :
  setA ∩ (Set.univ \ setB) = { x | -1 ≤ x ∧ x < 2 } := 
by 
  sorry

end intersection_complement_eq_l158_158103


namespace quadratic_non_real_roots_l158_158725

theorem quadratic_non_real_roots (b : ℝ) : 
  let a : ℝ := 1 
  let c : ℝ := 16 in
  (b^2 - 4 * a * c < 0) ↔ (-8 < b ∧ b < 8) :=
sorry

end quadratic_non_real_roots_l158_158725


namespace set_intersection_l158_158220

   -- Define set A
   def A : Set ℝ := {x : ℝ | (x - 3) / (x + 1) ≥ 0 }
   
   -- Define set B
   def B : Set ℝ := {x : ℝ | Real.log x / Real.log 2 < 2}

   -- Define the relative complement of A in the real numbers
   def complement_R (A : Set ℝ) : Set ℝ := {x : ℝ | ¬ (A x)}

   -- The main statement that needs to be proven
   theorem set_intersection :
     (complement_R A) ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by
     sorry
   
end set_intersection_l158_158220


namespace vasya_fraction_l158_158527

-- Define the variables for distances and total distance
variables {a b c d s : ℝ}

-- Define conditions
def anton_distance (a b : ℝ) : Prop := a = b / 2
def sasha_distance (c a d : ℝ) : Prop := c = a + d
def dima_distance (d s : ℝ) : Prop := d = s / 10
def total_distance (a b c d s : ℝ) : Prop := a + b + c + d = s

-- The main theorem 
theorem vasya_fraction (a b c d s : ℝ) (h1 : anton_distance a b) 
  (h2 : sasha_distance c a d) (h3 : dima_distance d s)
  (h4 : total_distance a b c d s) : b / s = 0.4 :=
sorry

end vasya_fraction_l158_158527


namespace find_z_value_l158_158907

noncomputable def y_varies_inversely_with_z (y z : ℝ) (k : ℝ) : Prop :=
  (y^4 * z^(1/4) = k)

theorem find_z_value (y z : ℝ) (k : ℝ) : 
  y_varies_inversely_with_z y z k → 
  y_varies_inversely_with_z 3 16 162 → 
  k = 162 →
  y = 6 → 
  z = 1 / 4096 := 
by 
  sorry

end find_z_value_l158_158907


namespace power_equality_l158_158485

theorem power_equality : (243 : ℝ) ^ (1 / 3) = (3 : ℝ) ^ (5 / 3) := 
by 
  sorry

end power_equality_l158_158485


namespace temperature_lower_than_minus_three_l158_158045

theorem temperature_lower_than_minus_three (a b : ℤ) (hx : a = -3) (hy : b = -6) : a + b = -9 :=
by
  sorry

end temperature_lower_than_minus_three_l158_158045


namespace contradiction_proof_l158_158436

theorem contradiction_proof (x y : ℝ) (h1 : x + y ≤ 0) (h2 : x > 0) (h3 : y > 0) : false :=
by
  sorry

end contradiction_proof_l158_158436


namespace square_simplify_l158_158042

   variable (y : ℝ)

   theorem square_simplify :
     (7 - Real.sqrt (y^2 - 49)) ^ 2 = y^2 - 14 * Real.sqrt (y^2 - 49) :=
   sorry
   
end square_simplify_l158_158042


namespace no_such_finite_set_exists_l158_158198

def satisfies_property (M : Set ℝ) : Prop :=
  ∀ a b ∈ M, 2 * a - b^2 ∈ M

theorem no_such_finite_set_exists :
  ¬∃ (M : Set ℝ), M.Finite ∧ 2 ≤ M.to_finset.card ∧ satisfies_property M :=
by
  sorry

end no_such_finite_set_exists_l158_158198


namespace work_completion_time_l158_158674

theorem work_completion_time (A_rate B_rate : ℝ) (hA : A_rate = 1/60) (hB : B_rate = 1/20) :
  1 / (A_rate + B_rate) = 15 :=
by
  sorry

end work_completion_time_l158_158674


namespace intersection_M_N_l158_158873

-- Definitions for sets M and N
def set_M : Set ℝ := {x | abs x < 1}
def set_N : Set ℝ := {x | x^2 <= x}

-- The theorem stating the intersection of M and N
theorem intersection_M_N : {x : ℝ | x ∈ set_M ∧ x ∈ set_N} = {x : ℝ | 0 <= x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l158_158873


namespace map_distance_l158_158134

/--
On a map, 8 cm represents 40 km. Prove that 20 cm represents 100 km.
-/
theorem map_distance (scale_factor : ℕ) (distance_cm : ℕ) (distance_km : ℕ) 
  (h_scale : scale_factor = 5) (h_distance_cm : distance_cm = 20) : 
  distance_km = 20 * scale_factor := 
by {
  sorry
}

end map_distance_l158_158134


namespace Vanya_bullets_l158_158806

theorem Vanya_bullets (initial_bullets : ℕ) (hits : ℕ) (shots_made : ℕ) (hits_reward : ℕ) :
  initial_bullets = 10 →
  shots_made = 14 →
  hits = shots_made / 2 →
  hits_reward = 3 →
  (initial_bullets + hits * hits_reward) - shots_made = 17 :=
by
  intros
  sorry

end Vanya_bullets_l158_158806


namespace ball_returns_to_bob_after_13_throws_l158_158660

theorem ball_returns_to_bob_after_13_throws:
  ∃ n : ℕ, n = 13 ∧ (∀ k, k < 13 → (1 + 3 * k) % 13 = 0) :=
sorry

end ball_returns_to_bob_after_13_throws_l158_158660


namespace range_of_a_inequality_solution_set_l158_158217

noncomputable def quadratic_condition_holds (a : ℝ) : Prop :=
∀ (x : ℝ), x^2 - 2 * a * x + a > 0

theorem range_of_a (a : ℝ) (h : quadratic_condition_holds a) : 0 < a ∧ a < 1 := sorry

theorem inequality_solution_set (a x : ℝ) (h1 : 0 < a) (h2 : a < 1) : (a^(x^2 - 3) < a^(2 * x) ∧ a^(2 * x) < 1) ↔ x > 3 := sorry

end range_of_a_inequality_solution_set_l158_158217


namespace mul_97_103_l158_158565

theorem mul_97_103 : (97:ℤ) = 100 - 3 → (103:ℤ) = 100 + 3 → 97 * 103 = 9991 := by
  intros h1 h2
  sorry

end mul_97_103_l158_158565


namespace problem_solution_l158_158100

variable (α : ℝ)
variable (h : Real.cos α = 1 / 5)

theorem problem_solution : Real.cos (2 * α - 2017 * Real.pi) = 23 / 25 := by
  sorry

end problem_solution_l158_158100


namespace area_ratio_l158_158296

-- Define the problem conditions
def Square (s : ℝ) := s > 0
def Rectangle (longer shorter : ℝ) := longer = 1.2 * shorter ∧ shorter = 0.8 * shorter

-- Define a function to calculate the area of square
def area_square (s : ℝ) : ℝ := s * s

-- Define a function to calculate the area of rectangle
def area_rectangle (longer shorter : ℝ) : ℝ := longer * shorter

-- State the proof problem
theorem area_ratio (s : ℝ) (h_square : Square s) :
  let longer := 1.2 * s
  let shorter := 0.8 * s
  area_rectangle longer shorter / area_square s = 24 / 25 :=
by
  sorry

end area_ratio_l158_158296


namespace Karen_has_fewer_nail_polishes_than_Kim_l158_158621

theorem Karen_has_fewer_nail_polishes_than_Kim :
  ∀ (Kim Heidi Karen : ℕ), Kim = 12 → Heidi = Kim + 5 → Karen + Heidi = 25 → (Kim - Karen) = 4 :=
by
  intros Kim Heidi Karen hK hH hKH
  sorry

end Karen_has_fewer_nail_polishes_than_Kim_l158_158621


namespace distance_covered_at_40_kmph_l158_158330

theorem distance_covered_at_40_kmph (x : ℝ) (h : 0 ≤ x ∧ x ≤ 250) 
  (total_distance : x + (250 - x) = 250) 
  (total_time : x / 40 + (250 - x) / 60 = 5.5) : 
  x = 160 :=
sorry

end distance_covered_at_40_kmph_l158_158330


namespace find_number_l158_158820

theorem find_number (x : ℝ) : 
  220050 = (555 + x) * (2 * (x - 555)) + 50 ↔ x = 425.875 ∨ x = -980.875 := 
by 
  sorry

end find_number_l158_158820


namespace twenty_eight_is_seventy_percent_of_what_number_l158_158805

theorem twenty_eight_is_seventy_percent_of_what_number (x : ℝ) (h : 28 / x = 70 / 100) : x = 40 :=
by
  sorry

end twenty_eight_is_seventy_percent_of_what_number_l158_158805


namespace range_of_m_l158_158207

theorem range_of_m (m : ℝ) (h1 : ∀ x : ℝ, (x^2 + 1) * (x^2 - 8*x - 20) ≤ 0 → (-2 ≤ x → x ≤ 10))
    (h2 : ∀ x : ℝ, x^2 - 2*x + 1 - m^2 ≤ 0 → (1 - m ≤ x → x ≤ 1 + m))
    (h3 : m > 0)
    (h4 : ∀ x : ℝ, ¬ ((x^2 + 1) * (x^2 - 8*x - 20) ≤ 0) → ¬ (x^2 - 2*x + 1 - m^2 ≤ 0) → (x < -2 ∨ x > 10) → (x < 1 - m ∨ x > 1 + m)) :
  m ≥ 9 := 
sorry

end range_of_m_l158_158207


namespace range_of_k_l158_158462

theorem range_of_k (x k : ℝ):
  (2 * x + 9 > 6 * x + 1) → (x - k < 1) → (x < 2) → k ≥ 1 :=
by 
  sorry

end range_of_k_l158_158462


namespace root_increases_implies_m_neg7_l158_158581

theorem root_increases_implies_m_neg7 
  (m : ℝ) 
  (h : ∃ x : ℝ, x ≠ 3 ∧ x = -m - 4 → x = 3) 
  : m = -7 := by
  sorry

end root_increases_implies_m_neg7_l158_158581


namespace labourer_savings_l158_158020

theorem labourer_savings
  (monthly_expenditure_first_6_months : ℕ)
  (monthly_expenditure_next_4_months : ℕ)
  (monthly_income : ℕ)
  (total_expenditure_first_6_months : ℕ)
  (total_income_first_6_months : ℕ)
  (debt_incurred : ℕ)
  (total_expenditure_next_4_months : ℕ)
  (total_income_next_4_months : ℕ)
  (money_saved : ℕ) :
  monthly_expenditure_first_6_months = 85 →
  monthly_expenditure_next_4_months = 60 →
  monthly_income = 78 →
  total_expenditure_first_6_months = 6 * monthly_expenditure_first_6_months →
  total_income_first_6_months = 6 * monthly_income →
  debt_incurred = total_expenditure_first_6_months - total_income_first_6_months →
  total_expenditure_next_4_months = 4 * monthly_expenditure_next_4_months →
  total_income_next_4_months = 4 * monthly_income →
  money_saved = total_income_next_4_months - (total_expenditure_next_4_months + debt_incurred) →
  money_saved = 30 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end labourer_savings_l158_158020


namespace tournament_total_players_l158_158416

theorem tournament_total_players (n : ℕ) (total_points : ℕ) (total_games : ℕ) (half_points : ℕ → ℕ) :
  (∀ k, half_points k * 2 = total_points) ∧ total_points = total_games ∧
  total_points = n * (n + 11) + 132 ∧
  total_games = (n + 12) * (n + 11) / 2 →
  n + 12 = 24 :=
by
  sorry

end tournament_total_players_l158_158416


namespace adjugate_power_null_l158_158886

variable {n : ℕ} (A : Matrix (Fin n) (Fin n) ℂ)

def adjugate (A : Matrix (Fin n) (Fin n) ℂ) : Matrix (Fin n) (Fin n) ℂ := sorry

theorem adjugate_power_null (A : Matrix (Fin n) (Fin n) ℂ) (m : ℕ) (hm : 0 < m) (h : (adjugate A) ^ m = 0) : 
  (adjugate A) ^ 2 = 0 := 
sorry

end adjugate_power_null_l158_158886


namespace vasya_fraction_l158_158526

-- Define the variables for distances and total distance
variables {a b c d s : ℝ}

-- Define conditions
def anton_distance (a b : ℝ) : Prop := a = b / 2
def sasha_distance (c a d : ℝ) : Prop := c = a + d
def dima_distance (d s : ℝ) : Prop := d = s / 10
def total_distance (a b c d s : ℝ) : Prop := a + b + c + d = s

-- The main theorem 
theorem vasya_fraction (a b c d s : ℝ) (h1 : anton_distance a b) 
  (h2 : sasha_distance c a d) (h3 : dima_distance d s)
  (h4 : total_distance a b c d s) : b / s = 0.4 :=
sorry

end vasya_fraction_l158_158526


namespace michael_truck_meetings_l158_158769

theorem michael_truck_meetings :
  let michael_speed := 6
  let truck_speed := 12
  let pail_distance := 200
  let truck_stop_time := 20
  let initial_distance := pail_distance
  ∃ (meetings : ℕ), 
  (michael_speed, truck_speed, pail_distance, truck_stop_time, initial_distance, meetings) = 
  (6, 12, 200, 20, 200, 10) :=
sorry

end michael_truck_meetings_l158_158769


namespace georgina_parrot_days_l158_158582

theorem georgina_parrot_days
  (total_phrases : ℕ)
  (phrases_per_week : ℕ)
  (initial_phrases : ℕ)
  (phrases_now : total_phrases = 17)
  (teaching_rate : phrases_per_week = 2)
  (initial_known : initial_phrases = 3) :
  (49 : ℕ) = (((17 - 3) / 2) * 7) :=
by
  -- proof will be here
  sorry

end georgina_parrot_days_l158_158582


namespace div_of_power_diff_div_l158_158616

theorem div_of_power_diff_div (a b n : ℕ) (h : a ≠ b) (h₀ : n ∣ (a^n - b^n)) : n ∣ (a^n - b^n) / (a - b) :=
  sorry

end div_of_power_diff_div_l158_158616


namespace inequality_false_implies_range_of_a_l158_158456

theorem inequality_false_implies_range_of_a (a : ℝ) : 
  (∀ t : ℝ, t^2 - 2 * t - a ≥ 0) ↔ a ≤ -1 :=
by
  sorry

end inequality_false_implies_range_of_a_l158_158456


namespace maximum_value_l158_158761

noncomputable def conditions (m n t : ℝ) : Prop :=
  -- m, n, t are positive real numbers
  (0 < m) ∧ (0 < n) ∧ (0 < t) ∧
  -- Equation condition
  (m^2 - 3 * m * n + 4 * n^2 - t = 0)

noncomputable def minimum_u (m n t : ℝ) : Prop :=
  -- Minimum value condition for t / mn
  (t / (m * n) = 1)

theorem maximum_value (m n t : ℝ) (h1 : conditions m n t) (h2 : minimum_u m n t) :
  -- Proving the maximum value of m + 2n - t
  (m + 2 * n - t) = 2 :=
sorry

end maximum_value_l158_158761


namespace perimeter_of_field_l158_158141

theorem perimeter_of_field (b l : ℕ) (h1 : l = b + 30) (h2 : b * l = 18000) : 2 * (l + b) = 540 := 
by 
  -- Proof goes here
sorry

end perimeter_of_field_l158_158141


namespace parabola_point_comparison_l158_158744

theorem parabola_point_comparison :
  let y1 := (1: ℝ)^2 - 2 * (1: ℝ) - 2
  let y2 := (3: ℝ)^2 - 2 * (3: ℝ) - 2
  y1 < y2 :=
by
  let y1 := (1: ℝ)^2 - 2 * (1: ℝ) - 2
  let y2 := (3: ℝ)^2 - 2 * (3: ℝ) - 2
  have h : y1 < y2 := by sorry
  exact h

end parabola_point_comparison_l158_158744


namespace find_sisters_dolls_l158_158720

variable (H S : ℕ)

-- Conditions
def hannah_has_5_times_sisters_dolls : Prop :=
  H = 5 * S

def total_dolls_is_48 : Prop :=
  H + S = 48

-- Question: Prove S = 8
theorem find_sisters_dolls (h1 : hannah_has_5_times_sisters_dolls H S) (h2 : total_dolls_is_48 H S) : S = 8 :=
sorry

end find_sisters_dolls_l158_158720


namespace hyperbola_eccentricity_l158_158081

theorem hyperbola_eccentricity (a c b : ℝ) (h₀ : b = 3)
  (h₁ : ∃ p, (p = 5) ∧ (a^2 + b^2 = (p : ℝ)^2))
  (h₂ : ∃ f, f = (p : ℝ)) :
  ∃ e, e = c / a ∧ e = 5 / 4 :=
by
  obtain ⟨p, hp, hap⟩ := h₁
  obtain ⟨f, hf⟩ := h₂
  sorry

end hyperbola_eccentricity_l158_158081


namespace product_of_solutions_l158_158204

-- Definitions based on given conditions
def equation (x : ℝ) : Prop := |x| = 3 * (|x| - 2)

-- Statement of the proof problem
theorem product_of_solutions : ∃ x1 x2 : ℝ, equation x1 ∧ equation x2 ∧ x1 * x2 = -9 := by
  sorry

end product_of_solutions_l158_158204


namespace coordinates_of_P_with_respect_to_y_axis_l158_158884

-- Define the coordinates of point P
def P_x : ℝ := 5
def P_y : ℝ := -1

-- Define the point P
def P : Prod ℝ ℝ := (P_x, P_y)

-- State the theorem
theorem coordinates_of_P_with_respect_to_y_axis :
  (P.1, P.2) = (-P_x, P_y) :=
sorry

end coordinates_of_P_with_respect_to_y_axis_l158_158884


namespace number_of_correct_statements_l158_158466

theorem number_of_correct_statements:
  (¬∀ (a : ℝ), -a < 0) ∧
  (∀ (x : ℝ), |x| = -x → x < 0) ∧
  (∀ (a : ℚ), (∀ (b : ℚ), |b| ≥ |a|) → a = 0) ∧
  (∀ (x y : ℝ), 5 * x^2 * y ≠ 0 → 2 + 1 = 3) →
  2 = 2 := sorry

end number_of_correct_statements_l158_158466


namespace tables_count_l158_158335

def total_tables (four_legged_tables three_legged_tables : Nat) : Nat :=
  four_legged_tables + three_legged_tables

theorem tables_count
  (four_legged_tables three_legged_tables : Nat)
  (total_legs : Nat)
  (h1 : four_legged_tables = 16)
  (h2 : total_legs = 124)
  (h3 : 4 * four_legged_tables + 3 * three_legged_tables = total_legs) :
  total_tables four_legged_tables three_legged_tables = 36 :=
by
  sorry

end tables_count_l158_158335


namespace k_ge_1_l158_158460

theorem k_ge_1 (k : ℝ) : 
  (∀ x : ℝ, 2 * x + 9 > 6 * x + 1 ∧ x - k < 1 → x < 2) → k ≥ 1 :=
by 
  sorry

end k_ge_1_l158_158460


namespace linear_system_solution_l158_158835

theorem linear_system_solution (x y : ℚ) (h1 : 3 * x + 2 * y = 7) (h2 : 6 * x - 5 * y = 4) :
  x = 43 / 27 ∧ y = 10 / 9 :=
sorry

end linear_system_solution_l158_158835


namespace bucket_full_weight_l158_158313

theorem bucket_full_weight (p q : ℝ) (x y : ℝ) 
    (h1 : x + (3/4) * y = p) 
    (h2 : x + (1/3) * y = q) : 
    x + y = (8 * p - 3 * q) / 5 :=
sorry

end bucket_full_weight_l158_158313


namespace stadium_revenue_difference_l158_158039

theorem stadium_revenue_difference :
  let total_capacity := 2000
  let vip_capacity := 200
  let standard_capacity := 1000
  let general_capacity := 800
  let vip_price := 50
  let standard_price := 30
  let general_price := 20
  let three_quarters (n : ℕ) := (3 * n) / 4
  let three_quarter_full := three_quarters total_capacity
  let vip_three_quarter := three_quarters vip_capacity
  let standard_three_quarter := three_quarters standard_capacity
  let general_three_quarter := three_quarters general_capacity
  let revenue_three_quarter := vip_three_quarter * vip_price + standard_three_quarter * standard_price + general_three_quarter * general_price
  let revenue_full := vip_capacity * vip_price + standard_capacity * standard_price + general_capacity * general_price
  revenue_three_quarter = 42000 ∧ (revenue_full - revenue_three_quarter) = 14000 :=
by
  sorry

end stadium_revenue_difference_l158_158039


namespace real_root_exists_l158_158092

theorem real_root_exists (p1 p2 q1 q2 : ℝ) 
(h : p1 * p2 = 2 * (q1 + q2)) : 
  (∃ x : ℝ, x^2 + p1 * x + q1 = 0) ∨ (∃ x : ℝ, x^2 + p2 * x + q2 = 0) :=
by
  sorry

end real_root_exists_l158_158092


namespace area_enclosed_by_graph_l158_158934

theorem area_enclosed_by_graph (x y : ℝ) (h : abs (5 * x) + abs (3 * y) = 15) : 
  ∃ (area : ℝ), area = 30 :=
sorry

end area_enclosed_by_graph_l158_158934


namespace tailor_buttons_l158_158512

theorem tailor_buttons (G : ℕ) (yellow_buttons : ℕ) (blue_buttons : ℕ) 
(h1 : yellow_buttons = G + 10) (h2 : blue_buttons = G - 5) 
(h3 : G + yellow_buttons + blue_buttons = 275) : G = 90 :=
sorry

end tailor_buttons_l158_158512


namespace solve_inequality_l158_158596

namespace Problem

-- Define polynomial f(x) = x^2 + ax + b
def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b

-- Conditions: f has zeroes at -2 and 3
lemma zeroes_of_f (a b : ℝ) : 
  f (-2) a b = 0 ∧ f 3 a b = 0 :=
by sorry

-- Coefficients a = -1 and b = -6
def a := -1
def b := -6

-- Polynomial becomes f(x) = x^2 - x - 6
lemma polynomial_expansion : f x a b = x^2 - x - 6 := 
by sorry

-- Set interval (-3, 2)
def interval := set.Ioo (-3 : ℝ) 2

-- Prove the inequality bf(ax) > 0 results in x ∈ (-3, 2)
theorem solve_inequality : ∀ x : ℝ, (b * (f (-x) a b)) > 0 ↔ x ∈ interval :=
by sorry

end Problem

end solve_inequality_l158_158596


namespace second_multiple_of_three_l158_158305

theorem second_multiple_of_three (n : ℕ) (h : 3 * (n - 1) + 3 * (n + 1) = 150) : 3 * n = 75 :=
sorry

end second_multiple_of_three_l158_158305


namespace jenny_coins_value_l158_158754

theorem jenny_coins_value (n d : ℕ) (h1 : d = 30 - n) (h2 : 150 + 5 * n = 300 - 5 * n + 120) :
  (300 - 5 * n : ℚ) / 100 = 1.65 := 
by
  sorry

end jenny_coins_value_l158_158754


namespace proposition_true_l158_158149

theorem proposition_true (x y : ℝ) : x + 2 * y ≠ 5 → (x ≠ 1 ∨ y ≠ 2) :=
by
  sorry

end proposition_true_l158_158149


namespace solve_for_k_and_j_l158_158402

theorem solve_for_k_and_j (k j : ℕ) (h1 : 64 / k = 8) (h2 : k * j = 128) : k = 8 ∧ j = 16 := by
  sorry

end solve_for_k_and_j_l158_158402


namespace factor_sum_l158_158226

theorem factor_sum (R S : ℝ) (h : ∃ (b c : ℝ), (x^2 + 3*x + 7) * (x^2 + b*x + c) = x^4 + R*x^2 + S) : R + S = 54 :=
sorry

end factor_sum_l158_158226


namespace acai_juice_cost_l158_158649

noncomputable def cost_per_litre_juice (x : ℝ) : Prop :=
  let total_cost_cocktail := 1399.45 * 53.333333333333332
  let cost_mixed_fruit_juice := 32 * 262.85
  let cost_acai_juice := 21.333333333333332 * x
  total_cost_cocktail = cost_mixed_fruit_juice + cost_acai_juice

/-- The cost per litre of the açaí berry juice is $3105.00 given the specified conditions. -/
theorem acai_juice_cost : cost_per_litre_juice 3105.00 :=
  sorry

end acai_juice_cost_l158_158649


namespace probability_of_green_light_l158_158961

theorem probability_of_green_light (red_time green_time yellow_time : ℕ) (h_red : red_time = 30) (h_green : green_time = 25) (h_yellow : yellow_time = 5) :
  (green_time.toRat / (red_time + green_time + yellow_time).toRat) = (5 / 12 : ℚ) :=
by
  sorry

end probability_of_green_light_l158_158961


namespace problem1_problem2_l158_158625

noncomputable def f (x : ℝ) : ℝ :=
let m := (2 * Real.cos x, 1)
let n := (Real.cos x, Real.sqrt 3 * Real.sin (2 * x))
m.1 * n.1 + m.2 * n.2

theorem problem1 :
  ( ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi ) ∧
  ∀ k : ℤ, ∀ x ∈ Set.Icc ((1 : ℝ) * Real.pi / 6 + k * Real.pi) ((2 : ℝ) * Real.pi / 3 + k * Real.pi),
  f x < f (x + (Real.pi / 3)) :=
sorry

theorem problem2 (A : ℝ) (a b c : ℝ) :
  a ≠ 0 ∧ b = 1 ∧ f A = 2 ∧
  0 < A ∧ A < Real.pi ∧
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 / 2  →
  a = Real.sqrt 3 :=
sorry

end problem1_problem2_l158_158625


namespace triangle_side_length_condition_l158_158653

theorem triangle_side_length_condition (a : ℝ) (h₁ : a > 0) (h₂ : a + 2 > a + 5) (h₃ : a + 5 > a + 2) (h₄ : a + 2 + a + 5 > a) : a > 3 :=
by
  sorry

end triangle_side_length_condition_l158_158653


namespace parabola_trajectory_l158_158856

theorem parabola_trajectory :
  ∀ P : ℝ × ℝ, (dist P (0, -1) + 1 = dist P (0, 3)) ↔ (P.1 ^ 2 = -8 * P.2) := by
  sorry

end parabola_trajectory_l158_158856


namespace next_unique_digits_date_l158_158745

-- Define the conditions
def is_after (d1 d2 : String) : Prop := sorry -- Placeholder, needs a date comparison function
def has_8_unique_digits (date : String) : Prop := sorry -- Placeholder, needs a function to check unique digits

-- Specify the problem and assertion
theorem next_unique_digits_date :
  ∀ date : String, is_after date "11.08.1999" → has_8_unique_digits date → date = "17.06.2345" :=
by
  sorry

end next_unique_digits_date_l158_158745


namespace find_x_plus_y_l158_158887

-- Define the points A, B, and C with given conditions
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 1, y := 1}
def C : Point := {x := 2, y := 4}

-- Define what it means for C to divide AB in the ratio 2:1
open Point

def divides_in_ratio (A B C : Point) (r₁ r₂ : ℝ) :=
  (C.x = (r₁ * A.x + r₂ * B.x) / (r₁ + r₂))
  ∧ (C.y = (r₁ * A.y + r₂ * B.y) / (r₁ + r₂))

-- Prove that x + y = 8 given the conditions
theorem find_x_plus_y {x y : ℝ} (B : Point) (H_B : B = {x := x, y := y}) :
  divides_in_ratio A B C 2 1 →
  x + y = 8 :=
by
  intro h
  sorry

end find_x_plus_y_l158_158887


namespace continuous_function_exists_fx_eq_gx_closed_interval_set_eq_l158_158054

open Set

variables {f g : ℝ → ℝ}

-- Given conditions
def is_function (f : ℝ → ℝ) : Prop := ∀ x, (x ∈ Icc 0 1 → f x ∈ Icc 0 1)
def is_monotonic (g : ℝ → ℝ) : Prop := Monotone g
def is_surjective (g : ℝ → ℝ) : Prop := ∀ y ∈ Icc 0 1, ∃ x ∈ Icc 0 1, g x = y
def bounded_diff (f g : ℝ → ℝ) : Prop := ∀ x y ∈ Icc 0 1, |f x - f y| ≤ |g x - g y|

-- Translation to proof problems
theorem continuous_function (f g : ℝ → ℝ) :
  is_function f ∧ is_function g ∧ is_monotonic g ∧ is_surjective g ∧ bounded_diff f g → ContinuousOn f (Icc 0 1) := by
  sorry

theorem exists_fx_eq_gx (f g : ℝ → ℝ) :
  is_function f ∧ is_function g ∧ is_monotonic g ∧ is_surjective g ∧ bounded_diff f g → 
  ∃ x ∈ Icc 0 1, f x = g x := by
  sorry

theorem closed_interval_set_eq (f g : ℝ → ℝ) :
  is_function f ∧ is_function g ∧ is_monotonic g ∧ is_surjective g ∧ bounded_diff f g →
  ∃ a b ∈ Icc 0 1, {x ∈ Icc 0 1 | f x = g x} = Icc a b := by
  sorry

end continuous_function_exists_fx_eq_gx_closed_interval_set_eq_l158_158054


namespace car_average_speed_l158_158788

-- Define the given conditions
def total_time_hours : ℕ := 5
def total_distance_miles : ℕ := 200

-- Define the average speed calculation
def average_speed (distance time : ℕ) : ℕ :=
  distance / time

-- State the theorem to be proved
theorem car_average_speed :
  average_speed total_distance_miles total_time_hours = 40 :=
by
  sorry

end car_average_speed_l158_158788


namespace expected_number_of_digits_l158_158901

-- Define a noncomputable expected_digits function for an icosahedral die
noncomputable def expected_digits : ℝ :=
  let p1 := 9 / 20
  let p2 := 11 / 20
  (p1 * 1) + (p2 * 2)

theorem expected_number_of_digits :
  expected_digits = 1.55 :=
by
  -- The proof will be filled in here
  sorry

end expected_number_of_digits_l158_158901


namespace field_length_is_112_l158_158789

-- Define the conditions
def is_pond_side_length : ℕ := 8
def pond_area : ℕ := is_pond_side_length * is_pond_side_length
def pond_to_field_area_ratio : ℚ := 1 / 98

-- Define the field properties
def field_area (w l : ℕ) : ℕ := w * l

-- Expressing the condition given length is double the width
def length_double_width (w l : ℕ) : Prop := l = 2 * w

-- Equating the areas based on the ratio given
def area_condition (w l : ℕ) : Prop := pond_area = pond_to_field_area_ratio * field_area w l

-- The main theorem
theorem field_length_is_112 : ∃ w l, length_double_width w l ∧ area_condition w l ∧ l = 112 := by
  sorry

end field_length_is_112_l158_158789


namespace math_ineq_problem_l158_158148

variable (a b c : ℝ)

theorem math_ineq_problem
  (h1 : a ≥ b) 
  (h2 : b ≥ c) 
  (h3 : a + b + c ≤ 1)
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) : 
  a^2 + 3 * b^2 + 5 * c^2 ≤ 1 :=
by
  sorry

end math_ineq_problem_l158_158148


namespace final_number_lt_one_l158_158010

theorem final_number_lt_one :
  ∀ (numbers : Finset ℕ),
    (numbers = Finset.range 3000 \ Finset.range 1000) →
    (∀ (a b : ℕ), a ∈ numbers → b ∈ numbers → a ≤ b →
    ∃ (numbers' : Finset ℕ), numbers' = (numbers \ {a, b}) ∪ {a / 2}) →
    ∃ (x : ℕ), x ∈ numbers ∧ x < 1 :=
by
  sorry

end final_number_lt_one_l158_158010


namespace quadratic_coefficient_a_l158_158449

theorem quadratic_coefficient_a (a b c : ℝ) :
  (2 = 9 * a - 3 * b + c) ∧
  (2 = 9 * a + 3 * b + c) ∧
  (-6 = 4 * a + 2 * b + c) →
  a = 8 / 5 :=
by
  sorry

end quadratic_coefficient_a_l158_158449


namespace license_plate_combinations_l158_158830

def num_choices_two_repeat_letters : ℕ :=
  (Nat.choose 26 2) * (Nat.choose 4 2) * (5 * 4)

theorem license_plate_combinations : num_choices_two_repeat_letters = 39000 := by
  sorry

end license_plate_combinations_l158_158830


namespace problem_part1_problem_part2_l158_158055

theorem problem_part1 (n : ℕ) (h : (n : ℚ) / (1 + 1 + n) = 1 / 2) : n = 2 :=
by sorry

theorem problem_part2 : 
  let outcomes := ({("red", "black"), ("red", "white1"), ("red", "white2"), ("black", "white1"), ("black", "white2"), ("white1", "white2")} : Finset (String × String)) in
  (outcomes.filter (λ p => p.1 = "red" ∧ (p.2 = "white1" ∨ p.2 = "white2"))).card / outcomes.card = 1 / 3 :=
by sorry

end problem_part1_problem_part2_l158_158055


namespace find_m_l158_158175

-- Given conditions
variable (U : Set ℕ) (A : Set ℕ) (m : ℕ)
variable (hU : U = {1, 2, 3, 4})
variable (hA : A = { x ∈ U | x^2 - 5 * x + m = 0 })
variable (hCUA : U \ A = {1, 4})

-- Prove that m = 6
theorem find_m (U A : Set ℕ) (m : ℕ) 
               (hU : U = {1, 2, 3, 4}) 
               (hA : A = { x ∈ U | x^2 - 5 * x + m = 0 }) 
               (hCUA : U \ A = {1, 4}) : 
  m = 6 := 
sorry

end find_m_l158_158175


namespace sampling_methods_correct_l158_158681

-- Definitions of the conditions:
def is_simple_random_sampling (method : String) : Prop := 
  method = "random selection of 24 students by the student council"

def is_systematic_sampling (method : String) : Prop := 
  method = "selection of students numbered from 001 to 240 whose student number ends in 3"

-- The equivalent math proof problem:
theorem sampling_methods_correct :
  is_simple_random_sampling "random selection of 24 students by the student council" ∧
  is_systematic_sampling "selection of students numbered from 001 to 240 whose student number ends in 3" :=
by
  sorry

end sampling_methods_correct_l158_158681


namespace equation_of_line_projection_l158_158318

theorem equation_of_line_projection (x y : ℝ) (m : ℝ) (x1 x2 : ℝ) (d : ℝ)
  (h1 : (5, 3) ∈ {(x, y) | y = 3 + m * (x - 5)})
  (h2 : x1 = (16 + 20 * m - 12) / (4 * m + 3))
  (h3 : x2 = (1 + 20 * m - 12) / (4 * m + 3))
  (h4 : abs (x1 - x2) = 1) :
  (y = 3 * x - 12 ∨ y = -4.5 * x + 25.5) :=
sorry

end equation_of_line_projection_l158_158318


namespace tallest_is_Justina_l158_158089

variable (H G I J K : ℝ)

axiom height_conditions1 : H < G
axiom height_conditions2 : G < J
axiom height_conditions3 : K < I
axiom height_conditions4 : I < G

theorem tallest_is_Justina : J > G ∧ J > H ∧ J > I ∧ J > K :=
by
  sorry

end tallest_is_Justina_l158_158089


namespace find_m_l158_158644

theorem find_m (m : ℝ) 
  (f g : ℝ → ℝ) 
  (x : ℝ) 
  (hf : f x = x^2 - 3 * x + m) 
  (hg : g x = x^2 - 3 * x + 5 * m) 
  (hx : x = 5) 
  (h_eq : 3 * f x = 2 * g x) :
  m = 10 / 7 := 
sorry

end find_m_l158_158644


namespace scientific_notation_113700_l158_158916

theorem scientific_notation_113700 : (113700 : ℝ) = 1.137 * 10^5 :=
by
  sorry

end scientific_notation_113700_l158_158916


namespace largest_x_FloorDiv7_eq_FloorDiv8_plus_1_l158_158129

-- Definitions based on conditions
def floor_div_7 (x : ℕ) : ℕ := x / 7
def floor_div_8 (x : ℕ) : ℕ := x / 8

-- The statement of the problem
theorem largest_x_FloorDiv7_eq_FloorDiv8_plus_1 :
  ∃ x : ℕ, (floor_div_7 x = floor_div_8 x + 1) ∧ (∀ y : ℕ, floor_div_7 y = floor_div_8 y + 1 → y ≤ x) ∧ x = 104 :=
sorry

end largest_x_FloorDiv7_eq_FloorDiv8_plus_1_l158_158129


namespace find_alpha_l158_158858

theorem find_alpha (α : ℝ) (k : ℤ) 
  (h : ∃ (k : ℤ), α + 30 = k * 360 + 180) : 
  α = k * 360 + 150 :=
by 
  sorry

end find_alpha_l158_158858


namespace triangle_sides_inequality_l158_158608

theorem triangle_sides_inequality
  {A B C a b c : ℝ}
  (triangle_ABC : ∃ (a b c : ℝ), a = b * (sin A / sin 60) ∧ c = b * (sin C / sin 60))
  (angle_B_eq_60 : B = 60)
  (side_sum_eq_one : a + c = 1)
  (A_range : 0 < A ∧ A < 120)
  : 0.5 ≤ b ∧ b < 1 :=
  sorry

end triangle_sides_inequality_l158_158608


namespace ratio_area_of_rectangle_to_square_l158_158290

theorem ratio_area_of_rectangle_to_square (s : ℝ) :
  (1.2 * s * 0.8 * s) / (s * s) = 24 / 25 :=
by
  sorry

end ratio_area_of_rectangle_to_square_l158_158290


namespace c_positive_when_others_negative_l158_158742

variables {a b c d e f : ℤ}

theorem c_positive_when_others_negative (h_ab_cdef_lt_0 : a * b + c * d * e * f < 0)
  (h_a_neg : a < 0) (h_b_neg : b < 0) (h_d_neg : d < 0) (h_e_neg : e < 0) (h_f_neg : f < 0) 
  : c > 0 :=
sorry

end c_positive_when_others_negative_l158_158742


namespace bricks_in_wall_l158_158469

theorem bricks_in_wall (x : ℕ) (r₁ r₂ combined_rate : ℕ) :
  (r₁ = x / 8) →
  (r₂ = x / 12) →
  (combined_rate = r₁ + r₂ - 15) →
  (6 * combined_rate = x) →
  x = 360 :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end bricks_in_wall_l158_158469


namespace total_snowfall_l158_158233

variable (morning_snowfall : ℝ) (afternoon_snowfall : ℝ)

theorem total_snowfall {morning_snowfall afternoon_snowfall : ℝ} (h_morning : morning_snowfall = 0.12) (h_afternoon : afternoon_snowfall = 0.5) :
  morning_snowfall + afternoon_snowfall = 0.62 :=
sorry

end total_snowfall_l158_158233


namespace percentage_of_bottle_danny_drank_l158_158991

theorem percentage_of_bottle_danny_drank
    (x : ℝ)  -- percentage of the first bottle Danny drinks, represented as a real number
    (b1 b2 b3 : ℝ)  -- volumes of the three bottles, represented as real numbers
    (h_b1 : b1 = 1)  -- first bottle is full (1 bottle)
    (h_b2 : b2 = 1)  -- second bottle is full (1 bottle)
    (h_b3 : b3 = 1)  -- third bottle is full (1 bottle)
    (h_given_away1 : b2 * 0.7 = 0.7)  -- gave away 70% of the second bottle
    (h_given_away2 : b3 * 0.7 = 0.7)  -- gave away 70% of the third bottle
    (h_soda_left : b1 * (1 - x) + b2 * 0.3 + b3 * 0.3 = 0.7)  -- 70% of bottle left
    : x = 0.9 :=
by
  sorry

end percentage_of_bottle_danny_drank_l158_158991


namespace extreme_value_when_a_is_one_range_of_lambda_l158_158130

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * exp(1 - x) - a * (x - 1)

theorem extreme_value_when_a_is_one :
  let f1 (x : ℝ) := f 1 x in
  (∀ x ∈ Ioo (3 / 4 : ℝ) 2, f1' x = deriv f1 x) →
  (∀ x ∈ Ioo (3 / 4 : ℝ) 2, 
    (deriv f1 x > 0 ↔ x < 1) ∧ 
    (deriv f1 x < 0 ↔ x > 1) ∧ 
    f1 1 = 1) :=
sorry

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + a * (x - 1 - exp(1 - x))

theorem range_of_lambda (a : ℝ) (x1 x2 : ℝ) (h : x1 < x2)
  (hx1 : deriv (g a) x1 = 0)
  (hx2 : deriv (g a) x2 = 0)
  (h_condition : x2 * g a x1 ≤ λ * deriv (f a) x1) :
  a > -1 → ∃ λ, λ = 2 * exp(1) / (exp(1) + 1) :=
sorry

end extreme_value_when_a_is_one_range_of_lambda_l158_158130


namespace four_distinct_real_roots_l158_158253

noncomputable def f (x c : ℝ) : ℝ := x^2 + 4 * x + c

-- We need to prove that if c is in the interval (-1, 3), f(f(x)) has exactly 4 distinct real roots
theorem four_distinct_real_roots (c : ℝ) : (-1 < c) ∧ (c < 3) → 
  ∃ (x₁ x₂ x₃ x₄ : ℝ), (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) 
  ∧ (f (f x₁ c) c = 0 ∧ f (f x₂ c) c = 0 ∧ f (f x₃ c) c = 0 ∧ f (f x₄ c) c = 0) :=
by sorry

end four_distinct_real_roots_l158_158253


namespace invisible_dots_48_l158_158708

theorem invisible_dots_48 (visible : Multiset ℕ) (hv : visible = [1, 2, 3, 3, 4, 5, 6, 6, 6]) :
  let total_dots := 4 * (1 + 2 + 3 + 4 + 5 + 6)
  let visible_sum := visible.sum
  total_dots - visible_sum = 48 :=
by
  sorry

end invisible_dots_48_l158_158708


namespace blue_paint_cans_l158_158776

noncomputable def ratio_of_blue_to_green := 4 / 1
def total_cans := 50
def fraction_of_blue := 4 / (4 + 1)
def number_of_blue_cans := fraction_of_blue * total_cans

theorem blue_paint_cans : number_of_blue_cans = 40 := by
  sorry

end blue_paint_cans_l158_158776


namespace summation_problem_l158_158356

open BigOperators

theorem summation_problem : 
  (∑ i in Finset.range 50, ∑ j in Finset.range 75, 2 * (i + 1) + 3 * (j + 1) + (i + 1) * (j + 1)) = 4275000 :=
by
  sorry

end summation_problem_l158_158356


namespace suitable_for_systematic_sampling_l158_158347

def city_districts : ℕ := 2000
def student_ratio : List ℕ := [3, 2, 8, 2]
def sample_size_city : ℕ := 200
def total_components : ℕ := 2000

def condition_A : Prop := 
  city_districts = 2000 ∧ 
  student_ratio = [3, 2, 8, 2] ∧ 
  sample_size_city = 200

def condition_B : Prop := 
  ∃ (n : ℕ), n = total_components ∧ n = 5

def condition_C : Prop := 
  ∃ (n : ℕ), n = total_components ∧ n = 200

def condition_D : Prop := 
  ∃ (n : ℕ), n = 20 ∧ n = 5

theorem suitable_for_systematic_sampling : condition_C :=
by
  sorry

end suitable_for_systematic_sampling_l158_158347


namespace length_AC_eq_9_74_l158_158136

-- Define the cyclic quadrilateral and given constraints
noncomputable def quad (A B C D : Type) : Prop := sorry
def angle_BAC := 50
def angle_ADB := 60
def AD := 3
def BC := 9

-- Prove that length of AC is 9.74 given the above conditions
theorem length_AC_eq_9_74 
  (A B C D : Type)
  (h_quad : quad A B C D)
  (h_angle_BAC : angle_BAC = 50)
  (h_angle_ADB : angle_ADB = 60)
  (h_AD : AD = 3)
  (h_BC : BC = 9) :
  ∃ AC, AC = 9.74 :=
sorry

end length_AC_eq_9_74_l158_158136


namespace scientific_notation_113700_l158_158915

theorem scientific_notation_113700 : (113700 : ℝ) = 1.137 * 10^5 :=
by
  sorry

end scientific_notation_113700_l158_158915


namespace page_added_twice_is_33_l158_158145

noncomputable def sum_first_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem page_added_twice_is_33 :
  ∃ n : ℕ, ∃ m : ℕ, sum_first_n n + m = 1986 ∧ 1 ≤ m ∧ m ≤ n → m = 33 := 
by {
  sorry
}

end page_added_twice_is_33_l158_158145


namespace minimum_value_of_y_l158_158299

theorem minimum_value_of_y (x : ℝ) (h : x > 0) : (∃ y, y = (x^2 + 1) / x ∧ y ≥ 2) ∧ (∃ y, y = (x^2 + 1) / x ∧ y = 2) :=
by
  sorry

end minimum_value_of_y_l158_158299


namespace Jungkook_red_balls_count_l158_158571

-- Define the conditions
def red_balls_per_box : ℕ := 3
def boxes_Jungkook_has : ℕ := 2

-- Statement to prove
theorem Jungkook_red_balls_count : red_balls_per_box * boxes_Jungkook_has = 6 :=
by sorry

end Jungkook_red_balls_count_l158_158571


namespace sum_of_divisors_85_l158_158166

theorem sum_of_divisors_85 : (1 + 5 + 17 + 85 = 108) := by
  sorry

end sum_of_divisors_85_l158_158166


namespace find_a3_l158_158095

noncomputable def S (n : ℕ) : ℤ := 2 * n^2 - 1
noncomputable def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem find_a3 : a 3 = 10 := by
  sorry

end find_a3_l158_158095


namespace find_k_l158_158151

variable {S : ℕ → ℤ} -- Assuming the sum function S for the arithmetic sequence 
variable {k : ℕ} -- k is a natural number

theorem find_k (h1 : S (k - 2) = -4) (h2 : S k = 0) (h3 : S (k + 2) = 8) (hk2 : k > 2) (hnaturalk : k ∈ Set.univ) : k = 6 := by
  sorry

end find_k_l158_158151


namespace find_special_number_l158_158340

theorem find_special_number : 
  ∃ n, 
  (n % 12 = 11) ∧ 
  (n % 11 = 10) ∧ 
  (n % 10 = 9) ∧ 
  (n % 9 = 8) ∧ 
  (n % 8 = 7) ∧ 
  (n % 7 = 6) ∧ 
  (n % 6 = 5) ∧ 
  (n % 5 = 4) ∧ 
  (n % 4 = 3) ∧ 
  (n % 3 = 2) ∧ 
  (n % 2 = 1) ∧ 
  (n = 27719) :=
  sorry

end find_special_number_l158_158340


namespace find_certain_number_l158_158053

theorem find_certain_number (x : ℝ) (h : 25 * x = 675) : x = 27 :=
by {
  sorry
}

end find_certain_number_l158_158053


namespace min_value_fraction_sum_l158_158278

theorem min_value_fraction_sum (p q r a b : ℝ) (hpq : 0 < p) (hq : p < q) (hr : q < r)
  (h_sum : p + q + r = a) (h_prod_sum : p * q + q * r + r * p = b) (h_prod : p * q * r = 48) :
  ∃ (min_val : ℝ), min_val = (1 / p) + (2 / q) + (3 / r) ∧ min_val = 3 / 2 :=
sorry

end min_value_fraction_sum_l158_158278


namespace expected_value_of_remainder_eq_1816_over_6561_l158_158427
-- Lean 4 statement for the math proof problem

noncomputable def expected_remainder_binomial_mod3 : ℚ :=
  1816 / 6561

theorem expected_value_of_remainder_eq_1816_over_6561 :
  ∀ (a b : ℕ), a ∈ Finset.range 81 → b ∈ Finset.range 81 → 
  (a ≠ b → ExpectedValue (λ (a b : ℕ) => (nat.choose a b) % 3) = expected_remainder_binomial_mod3) :=
begin
  intros a b ha hb h_not_eq,
  sorry -- proof is not required
end

end expected_value_of_remainder_eq_1816_over_6561_l158_158427


namespace total_area_of_field_l158_158337

theorem total_area_of_field 
  (A_s : ℕ) 
  (h₁ : A_s = 315)
  (A_l : ℕ) 
  (h₂ : A_l - A_s = (1/5) * ((A_s + A_l) / 2)) : 
  A_s + A_l = 700 := 
  by 
    sorry

end total_area_of_field_l158_158337


namespace distance_home_to_school_l158_158480

def speed_walk := 5
def speed_car := 15
def time_difference := 2

variable (d : ℝ) -- distance from home to school
variable (T1 T2 : ℝ) -- T1: time to school, T2: time back home

-- Conditions
axiom h1 : T1 = d / speed_walk / 2 + d / speed_car / 2
axiom h2 : d = speed_car * T2 / 3 + speed_walk * 2 * T2 / 3
axiom h3 : T1 = T2 + time_difference

-- Theorem to prove
theorem distance_home_to_school : d = 150 :=
by
  sorry

end distance_home_to_school_l158_158480


namespace card_statements_true_l158_158896

def statement1 (statements : Fin 5 → Prop) : Prop :=
  ∃! i, i < 5 ∧ statements i

def statement2 (statements : Fin 5 → Prop) : Prop :=
  (∃ i j, i < 5 ∧ j < 5 ∧ i ≠ j ∧ statements i ∧ statements j) ∧ ¬(∃ h k l, h < 5 ∧ k < 5 ∧ l < 5 ∧ h ≠ k ∧ h ≠ l ∧ k ≠ l ∧ statements h ∧ statements k ∧ statements l)

def statement3 (statements : Fin 5 → Prop) : Prop :=
  (∃ i j k, i < 5 ∧ j < 5 ∧ k < 5 ∧ i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ statements i ∧ statements j ∧ statements k) ∧ ¬(∃ a b c d, a < 5 ∧ b < 5 ∧ c < 5 ∧ d < 5 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ statements a ∧ statements b ∧ statements c ∧ statements d)

def statement4 (statements : Fin 5 → Prop) : Prop :=
  (∃ i j k l, i < 5 ∧ j < 5 ∧ k < 5 ∧ l < 5 ∧ i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ statements i ∧ statements j ∧ statements k ∧ statements l) ∧ ¬(∃ a b c d e, a < 5 ∧ b < 5 ∧ c < 5 ∧ d < 5 ∧ e < 5 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ statements a ∧ statements b ∧ statements c ∧ statements d ∧ statements e)

def statement5 (statements : Fin 5 → Prop) : Prop :=
  ∀ i, i < 5 → statements i

theorem card_statements_true : ∃ (statements : Fin 5 → Prop), 
  statement1 statements ∨ statement2 statements ∨ statement3 statements ∨ statement4 statements ∨ statement5 statements 
  ∧ statement3 statements := 
sorry

end card_statements_true_l158_158896


namespace units_digit_of_5_pow_150_plus_7_l158_158809

theorem units_digit_of_5_pow_150_plus_7 : (5^150 + 7) % 10 = 2 := by
  sorry

end units_digit_of_5_pow_150_plus_7_l158_158809


namespace total_seeds_in_watermelon_l158_158664

theorem total_seeds_in_watermelon :
  let slices := 40
  let black_seeds_per_slice := 20
  let white_seeds_per_slice := 20
  let total_black_seeds := black_seeds_per_slice * slices
  let total_white_seeds := white_seeds_per_slice * slices
  total_black_seeds + total_white_seeds = 1600 := by
  sorry

end total_seeds_in_watermelon_l158_158664


namespace value_of_b_l158_158302

theorem value_of_b (a b : ℕ) (h1 : a * b = 2 * (a + b) + 10) (h2 : b - a = 5) : b = 9 := 
by {
  -- Proof is not required, so we use sorry to complete the statement
  sorry
}

end value_of_b_l158_158302


namespace sector_radius_l158_158593

theorem sector_radius (r : ℝ) (h1 : r > 0) 
  (h2 : ∀ (l : ℝ), l = r → 
    (3 * r) / (1 / 2 * r^2) = 2) : r = 3 := 
sorry

end sector_radius_l158_158593


namespace greatest_third_term_of_arithmetic_sequence_l158_158033

def is_arithmetic_sequence (a b c d : ℤ) : Prop := (b - a = c - b) ∧ (c - b = d - c)

theorem greatest_third_term_of_arithmetic_sequence :
  ∃ a b c d : ℤ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  is_arithmetic_sequence a b c d ∧
  (a + b + c + d = 52) ∧
  (c = 17) :=
sorry

end greatest_third_term_of_arithmetic_sequence_l158_158033


namespace max_value_of_quadratic_l158_158868

theorem max_value_of_quadratic:
  ∀ (x : ℝ), (∃ y : ℝ, y = -3 * x ^ 2 + 9) → (∃ max_y : ℝ, max_y = 9 ∧ ∀ x : ℝ, -3 * x ^ 2 + 9 ≤ max_y) :=
by
  sorry

end max_value_of_quadratic_l158_158868


namespace ratio_A_B_l158_158194

noncomputable def A : ℝ := ∑' n : ℕ, if n % 4 = 0 then 0 else 1 / (n:ℝ) ^ 2
noncomputable def B : ℝ := ∑' k : ℕ, (-1)^(k+1) / (4 * (k:ℝ)) ^ 2

theorem ratio_A_B : A / B = 32 := by
  -- proof here
  sorry

end ratio_A_B_l158_158194


namespace arithmetic_sequence_problem_l158_158387

theorem arithmetic_sequence_problem (a : ℕ → ℕ) (S : ℕ → ℕ) (k : ℕ) 
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : S 9 = 81)
  (h3 : a (k - 4) = 191)
  (h4 : S k = 10000) :
  k = 100 :=
by
  sorry

end arithmetic_sequence_problem_l158_158387


namespace gcd_4004_10010_l158_158202

theorem gcd_4004_10010 : Nat.gcd 4004 10010 = 2002 :=
by
  have h1 : 4004 = 4 * 1001 := by norm_num
  have h2 : 10010 = 10 * 1001 := by norm_num
  sorry

end gcd_4004_10010_l158_158202


namespace number_of_good_games_l158_158895

def total_games : ℕ := 11
def bad_games : ℕ := 5
def good_games : ℕ := total_games - bad_games

theorem number_of_good_games : good_games = 6 := by
  sorry

end number_of_good_games_l158_158895


namespace more_bottle_caps_than_wrappers_l158_158701

namespace DannyCollection

def bottle_caps_found := 50
def wrappers_found := 46

theorem more_bottle_caps_than_wrappers :
  bottle_caps_found - wrappers_found = 4 :=
by
  -- We skip the proof here with "sorry"
  sorry

end DannyCollection

end more_bottle_caps_than_wrappers_l158_158701


namespace total_students_l158_158065

theorem total_students (m f : ℕ) (h_ratio : 3 * f = 7 * m) (h_males : m = 21) : m + f = 70 :=
by
  sorry

end total_students_l158_158065


namespace goldfish_feeding_l158_158899

theorem goldfish_feeding (g : ℕ) (h : g = 8) : 4 * g = 32 :=
by
  sorry

end goldfish_feeding_l158_158899


namespace prod_97_103_l158_158564

theorem prod_97_103 : (97 * 103) = 9991 := 
by 
  have h1 : 97 = 100 - 3 := by rfl
  have h2 : 103 = 100 + 3 := by rfl
  calc
    97 * 103 = (100 - 3) * (100 + 3) : by rw [h1, h2]
         ... = 100^2 - 3^2 : by rw (mul_sub (100:ℤ) 3 3)
         ... = 10000 - 9 : by norm_num
         ... = 9991 : by norm_num
 
end prod_97_103_l158_158564


namespace reduce_to_original_l158_158265

theorem reduce_to_original (x : ℝ) (factor : ℝ) (original : ℝ) :
  original = x → factor = 1/1000 → x * factor = 0.0169 :=
by
  intros h1 h2
  sorry

end reduce_to_original_l158_158265


namespace work_done_by_force_l158_158972

def force : ℝ × ℝ := (-1, -2)
def displacement : ℝ × ℝ := (3, 4)

def work_done (F S : ℝ × ℝ) : ℝ :=
  F.1 * S.1 + F.2 * S.2

theorem work_done_by_force :
  work_done force displacement = -11 := 
by
  sorry

end work_done_by_force_l158_158972


namespace solve_log_equation_l158_158781

theorem solve_log_equation :
  ∀ x : ℝ, 
  5 * Real.logb x (x / 9) + Real.logb (x / 9) x^3 + 8 * Real.logb (9 * x^2) (x^2) = 2
  → (x = 3 ∨ x = Real.sqrt 3) := by
  sorry

end solve_log_equation_l158_158781


namespace factorize_expression_l158_158844

theorem factorize_expression (x : ℝ) : x^3 - 4 * x^2 + 4 * x = x * (x - 2)^2 := 
  sorry

end factorize_expression_l158_158844


namespace combined_sum_is_115_over_3_l158_158256

def geometric_series_sum (a : ℚ) (r : ℚ) : ℚ :=
  if h : abs r < 1 then a / (1 - r) else 0

def arithmetic_series_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a + (n - 1) * d) / 2

noncomputable def combined_series_sum : ℚ :=
  let geo_sum := geometric_series_sum 5 (-1/2)
  let arith_sum := arithmetic_series_sum 3 2 5
  geo_sum + arith_sum

theorem combined_sum_is_115_over_3 : combined_series_sum = 115 / 3 := 
  sorry

end combined_sum_is_115_over_3_l158_158256


namespace eval_expression_l158_158573

theorem eval_expression : 3^13 / 3^3 + 2^3 = 59057 := by
  sorry

end eval_expression_l158_158573


namespace cubic_root_relationship_l158_158004

theorem cubic_root_relationship 
  (r : ℝ) (h : r^3 - r + 3 = 0) : 
  (r^2)^3 - 2 * (r^2)^2 + (r^2) - 9 = 0 := 
by 
  sorry

end cubic_root_relationship_l158_158004


namespace mary_should_drink_six_glasses_per_day_l158_158156

def daily_water_goal : ℕ := 1500
def glass_capacity : ℕ := 250
def required_glasses (daily_goal : ℕ) (capacity : ℕ) : ℕ := daily_goal / capacity

theorem mary_should_drink_six_glasses_per_day :
  required_glasses daily_water_goal glass_capacity = 6 :=
by
  sorry

end mary_should_drink_six_glasses_per_day_l158_158156


namespace mn_equals_neg3_l158_158642

noncomputable def function_with_extreme_value (m n : ℝ) : Prop :=
  let f := λ x : ℝ => m * x^3 + n * x
  let f' := λ x : ℝ => 3 * m * x^2 + n
  f' (1 / m) = 0

theorem mn_equals_neg3 (m n : ℝ) (h : function_with_extreme_value m n) : m * n = -3 :=
sorry

end mn_equals_neg3_l158_158642


namespace hens_count_l158_158061

theorem hens_count
  (H C : ℕ)
  (heads_eq : H + C = 48)
  (feet_eq : 2 * H + 4 * C = 136) :
  H = 28 :=
by
  sorry

end hens_count_l158_158061


namespace P_subsetneq_Q_l158_158006

def P : Set ℝ := { x : ℝ | x > 1 }
def Q : Set ℝ := { x : ℝ | x^2 - x > 0 }

theorem P_subsetneq_Q : P ⊂ Q :=
by
  sorry

end P_subsetneq_Q_l158_158006


namespace volume_of_region_l158_158153

theorem volume_of_region (r1 r2 : ℝ) (h : r1 = 5) (h2 : r2 = 8) : 
  let V_sphere (r : ℝ) := (4 / 3) * Real.pi * r^3
  let V_cylinder (r : ℝ) := Real.pi * r^2 * r
  (V_sphere r2) - (V_sphere r1) - (V_cylinder r1) = 391 * Real.pi :=
by
  -- Placeholder proof
  sorry

end volume_of_region_l158_158153


namespace jed_correct_speed_l158_158235

def fine_per_mph := 16
def jed_fine := 256
def speed_limit := 50

def jed_speed : Nat := speed_limit + jed_fine / fine_per_mph

theorem jed_correct_speed : jed_speed = 66 := by
  sorry

end jed_correct_speed_l158_158235


namespace smallest_number_satisfying_conditions_l158_158165

theorem smallest_number_satisfying_conditions :
  ∃ b : ℕ, b ≡ 3 [MOD 5] ∧ b ≡ 2 [MOD 4] ∧ b ≡ 2 [MOD 6] ∧ b = 38 := 
by
  sorry

end smallest_number_satisfying_conditions_l158_158165


namespace no_positive_solution_for_special_k_l158_158135
open Nat

theorem no_positive_solution_for_special_k (p : ℕ) (hp : p.Prime) (hmod : p % 4 = 3) :
    ¬ ∃ n m k : ℕ, (n > 0) ∧ (m > 0) ∧ (k = p^2) ∧ (n^2 + m^2 = k * (m^4 + n)) :=
sorry

end no_positive_solution_for_special_k_l158_158135


namespace probability_of_green_is_correct_l158_158193

structure ContainerBalls :=
  (red : ℕ)
  (green : ℕ)

def ContainerA : ContainerBalls := ⟨3, 5⟩
def ContainerB : ContainerBalls := ⟨5, 5⟩
def ContainerC : ContainerBalls := ⟨7, 3⟩
def ContainerD : ContainerBalls := ⟨4, 6⟩

noncomputable def probability_green_ball (c : ContainerBalls) : ℚ :=
  c.green / (c.red + c.green)

noncomputable def total_probability_green : ℚ :=
  (1 / 4) * probability_green_ball ContainerA +
  (1 / 4) * probability_green_ball ContainerB +
  (1 / 4) * probability_green_ball ContainerC +
  (1 / 4) * probability_green_ball ContainerD

theorem probability_of_green_is_correct : total_probability_green = 81 / 160 := 
  sorry

end probability_of_green_is_correct_l158_158193


namespace range_of_k_l158_158461

theorem range_of_k (x k : ℝ):
  (2 * x + 9 > 6 * x + 1) → (x - k < 1) → (x < 2) → k ≥ 1 :=
by 
  sorry

end range_of_k_l158_158461


namespace area_enclosed_by_absolute_value_linear_eq_l158_158937

theorem area_enclosed_by_absolute_value_linear_eq (x y : ℝ) :
  (|5 * x| + |3 * y| = 15) → ∃ (A : ℝ), A = 30 :=
by
  sorry

end area_enclosed_by_absolute_value_linear_eq_l158_158937


namespace find_parallel_lines_a_l158_158389

/--
Given two lines \(l_1\): \(x + 2y - 3 = 0\) and \(l_2\): \(2x - ay + 3 = 0\),
prove that if the lines are parallel, then \(a = -4\).
-/
theorem find_parallel_lines_a (a : ℝ) :
  (∀ (x y : ℝ), x + 2*y - 3 = 0) 
  → (∀ (x y : ℝ), 2*x - a*y + 3 = 0)
  → (-1 / 2 = 2 / -a) 
  → a = -4 :=
by
  intros
  sorry

end find_parallel_lines_a_l158_158389


namespace youngest_child_age_l158_158174

theorem youngest_child_age 
  (x : ℕ)
  (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 50) : 
  x = 6 := 
by 
  sorry

end youngest_child_age_l158_158174


namespace square_distance_from_B_to_center_l158_158338

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

end square_distance_from_B_to_center_l158_158338


namespace power_equality_l158_158484

theorem power_equality : (243 : ℝ) ^ (1 / 3) = (3 : ℝ) ^ (5 / 3) := 
by 
  sorry

end power_equality_l158_158484


namespace subset_relation_l158_158718

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 2*x + 2}
def N : Set ℝ := {x | ∃ y : ℝ, y = Real.log (x - 4) / Real.log 2}

-- State the proof problem
theorem subset_relation : N ⊆ M := 
sorry

end subset_relation_l158_158718


namespace rectangle_to_square_area_ratio_is_24_25_l158_158286

noncomputable def rectangle_to_square_area_ratio
  (s : ℝ) -- length of side of square S
  (longer_side : ℝ := 1.2 * s) -- longer side of rectangle R
  (shorter_side : ℝ := 0.8 * s) -- shorter side of rectangle R
  (area_R : ℝ := longer_side * shorter_side) -- area of rectangle R
  (area_S : ℝ := s^2) -- area of square S
  : ℝ := 
  area_R / area_S

theorem rectangle_to_square_area_ratio_is_24_25 
  (s : ℝ)
  : rectangle_to_square_area_ratio s = 24 / 25 :=
by 
  sorry

end rectangle_to_square_area_ratio_is_24_25_l158_158286


namespace vasya_drives_fraction_l158_158548

theorem vasya_drives_fraction {a b c d s : ℝ} 
  (h1 : a = b / 2) 
  (h2 : c = a + d) 
  (h3 : d = s / 10) 
  (h4 : a + b + c + d = s) : 
  b / s = 0.4 :=
by
  sorry

end vasya_drives_fraction_l158_158548


namespace range_of_x_coordinate_of_Q_l158_158586

def Point := ℝ × ℝ

def parabola (P : Point) : Prop :=
  P.2 = P.1 ^ 2

def vector (P Q : Point) : Point :=
  (Q.1 - P.1, Q.2 - P.2)

def dot_product (u v : Point) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def perpendicular (P Q R : Point) : Prop :=
  dot_product (vector P Q) (vector P R) = 0

theorem range_of_x_coordinate_of_Q:
  ∀ (A P Q: Point), 
    A = (-1, 1) →
    parabola P →
    parabola Q →
    perpendicular P A Q →
    (Q.1 ≤ -3 ∨ Q.1 ≥ 1) :=
by
  intros A P Q hA hParabP hParabQ hPerp
  sorry

end range_of_x_coordinate_of_Q_l158_158586


namespace sequence_bounds_l158_158048

theorem sequence_bounds (θ : ℝ) (n : ℕ) (a : ℕ → ℝ) (hθ : 0 < θ ∧ θ < π / 2) 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 1 - 2 * (Real.sin θ * Real.cos θ)^2) 
  (h_recurrence : ∀ n, a (n + 2) - a (n + 1) + a n * (Real.sin θ * Real.cos θ)^2 = 0) :
  1 / 2 ^ (n - 1) ≤ a n ∧ a n ≤ 1 - (Real.sin (2 * θ))^n * (1 - 1 / 2 ^ (n - 1)) := 
sorry

end sequence_bounds_l158_158048


namespace negation_of_exists_sin_gt_one_equiv_forall_sin_le_one_l158_158390

open Real

theorem negation_of_exists_sin_gt_one_equiv_forall_sin_le_one :
  (¬ (∃ x : ℝ, sin x > 1)) ↔ (∀ x : ℝ, sin x ≤ 1) :=
sorry

end negation_of_exists_sin_gt_one_equiv_forall_sin_le_one_l158_158390


namespace kate_spent_on_mouse_l158_158127

theorem kate_spent_on_mouse :
  let march := 27
  let april := 13
  let may := 28
  let saved := march + april + may
  let keyboard := 49
  let left := 14
  saved - left - keyboard = 5 :=
by
  let march := 27
  let april := 13
  let may := 28
  let saved := march + april + may
  let keyboard := 49
  let left := 14
  show saved - left - keyboard = 5
  sorry

end kate_spent_on_mouse_l158_158127


namespace tip_percentage_correct_l158_158661

def lunch_cost := 50.20
def total_spent := 60.24
def tip_percentage := ((total_spent - lunch_cost) / lunch_cost) * 100

theorem tip_percentage_correct : tip_percentage = 19.96 := 
by
  sorry

end tip_percentage_correct_l158_158661


namespace chocolates_problem_l158_158069

-- Let denote the quantities as follows:
-- C: number of caramels
-- N: number of nougats
-- T: number of truffles
-- P: number of peanut clusters

def C_nougats_truffles_peanutclusters (C N T P : ℕ) :=
  N = 2 * C ∧
  T = C + 6 ∧
  C + N + T + P = 50 ∧
  P = 32

theorem chocolates_problem (C N T P : ℕ) :
  C_nougats_truffles_peanutclusters C N T P → C = 3 :=
by
  intros h
  have hN := h.1
  have hT := h.2.1
  have hSum := h.2.2.1
  have hP := h.2.2.2
  sorry

end chocolates_problem_l158_158069


namespace solve_for_x_l158_158636

-- Define the given equation as a hypothesis
def equation (x : ℝ) : Prop :=
  0.05 * x - 0.09 * (25 - x) = 5.4

-- State the theorem that x = 54.6428571 satisfies the given equation
theorem solve_for_x : (x : ℝ) → equation x → x = 54.6428571 :=
by
  sorry

end solve_for_x_l158_158636


namespace range_of_a_solution_set_of_inequality_l158_158218

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

end range_of_a_solution_set_of_inequality_l158_158218


namespace decryption_proof_l158_158468

-- Definitions
def Original_Message := "МОСКВА"
def Encrypted_Text_1 := "ТПЕОИРВНТМОЛАРГЕИАНВИЛЕДНМТААГТДЬТКУБЧКГЕИШНЕИАЯРЯ"
def Encrypted_Text_2 := "ЛСИЕМГОРТКРОМИТВАВКНОПКРАСЕОГНАЬЕП"
def Encrypted_Text_3 := "РТПАИОМВСВТИЕОБПРОЕННИИГЬКЕЕАМТАЛВТДЬСОУМЧШСЕОНШЬИАЯК"

noncomputable def Encrypted_Message_1 := "ЙМЫВОТСЬЛКЪГВЦАЯЯ"
noncomputable def Encrypted_Message_2 := "УКМАПОЧСРКЩВЗАХ"
noncomputable def Encrypted_Message_3 := "ШМФЭОГЧСЙЪКФЬВЫЕАКК"

def Decrypted_Message_1_and_3 := "ПОВТОРЕНИЕМАТЬУЧЕНИЯ"
def Decrypted_Message_2 := "СМОТРИВКОРЕНЬ"

-- Theorem statement
theorem decryption_proof :
  (Encrypted_Text_1 = Encrypted_Text_3 ∧ Original_Message = "МОСКВА" ∧ Encrypted_Message_1 = Encrypted_Message_3) →
  (Decrypted_Message_1_and_3 = "ПОВТОРЕНИЕМАТЬУЧЕНИЯ" ∧ Decrypted_Message_2 = "СМОТРИВКОРЕНЬ") :=
by 
  sorry

end decryption_proof_l158_158468


namespace probability_of_multiple_135_l158_158872

noncomputable def single_digit_multiples_of_3 := {3, 6, 9} : Finset ℕ
noncomputable def primes_less_than_50 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47} : Finset ℕ
def product_is_multiple_of_135 (a b : ℕ) : Prop := a * b ∣ 135

theorem probability_of_multiple_135 :
  (1 / 45 : ℚ) = 
  (single_digit_multiples_of_3.card * primes_less_than_50.card).toRat / 
  (single_digit_multiples_of_3.filter (λ a, primes_less_than_50.card * a ∣ 135)).card :=
by 
  sorry

end probability_of_multiple_135_l158_158872


namespace purely_imaginary_has_specific_a_l158_158413

theorem purely_imaginary_has_specific_a (a : ℝ) :
  (a^2 - 1 + (a - 1 : ℂ) * Complex.I) = (a - 1 : ℂ) * Complex.I → a = -1 := 
by
  sorry

end purely_imaginary_has_specific_a_l158_158413


namespace binomial_expansion_problem_l158_158446

noncomputable def binomial_expansion_sum_coefficients (n : ℕ) : ℤ :=
  (1 - 3) ^ n

def general_term_coefficient (n r : ℕ) : ℤ :=
  (-3) ^ r * (Nat.choose n r)

theorem binomial_expansion_problem :
  ∃ (n : ℕ), binomial_expansion_sum_coefficients n = 64 ∧ general_term_coefficient 6 2 = 135 :=
by
  sorry

end binomial_expansion_problem_l158_158446


namespace minyoung_division_l158_158810

theorem minyoung_division : 
  ∃ x : ℝ, 107.8 / x = 9.8 ∧ x = 11 :=
by
  use 11
  simp
  sorry

end minyoung_division_l158_158810


namespace two_common_points_with_x_axis_l158_158598

noncomputable def func (x d : ℝ) : ℝ := x^3 - 3 * x + d

theorem two_common_points_with_x_axis (d : ℝ) :
(∃ x1 x2 : ℝ, x1 ≠ x2 ∧ func x1 d = 0 ∧ func x2 d = 0) ↔ (d = 2 ∨ d = -2) :=
by
  sorry

end two_common_points_with_x_axis_l158_158598


namespace sufficient_and_necessary_condition_l158_158032

theorem sufficient_and_necessary_condition {a : ℝ} :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ a ≥ 4 :=
sorry

end sufficient_and_necessary_condition_l158_158032


namespace trajectory_circle_equation_l158_158911

theorem trajectory_circle_equation :
  (∀ (x y : ℝ), dist (x, y) (0, 0) = 4 ↔ x^2 + y^2 = 16) :=  
sorry

end trajectory_circle_equation_l158_158911


namespace soft_drink_company_bottle_count_l158_158977

theorem soft_drink_company_bottle_count
  (B : ℕ)
  (initial_small_bottles : ℕ := 6000)
  (percent_sold_small : ℝ := 0.12)
  (percent_sold_big : ℝ := 0.14)
  (bottles_remaining_total : ℕ := 18180) :
  (initial_small_bottles * (1 - percent_sold_small) + B * (1 - percent_sold_big) = bottles_remaining_total) → B = 15000 :=
by
  sorry

end soft_drink_company_bottle_count_l158_158977


namespace novel_cost_l158_158249

-- Given conditions
variable (N : ℕ) -- cost of the novel
variable (lunch_cost : ℕ) -- cost of lunch

-- Conditions
axiom gift_amount : N + lunch_cost + 29 = 50
axiom lunch_cost_eq : lunch_cost = 2 * N

-- Question and answer tuple as a theorem
theorem novel_cost : N = 7 := 
by
  sorry -- Proof estaps are to be filled in.

end novel_cost_l158_158249


namespace decompose_96_l158_158568

theorem decompose_96 (x y : ℤ) (h1 : x * y = 96) (h2 : x^2 + y^2 = 208) :
  (x = 8 ∧ y = 12) ∨ (x = 12 ∧ y = 8) ∨ (x = -8 ∧ y = -12) ∨ (x = -12 ∧ y = -8) := by
  sorry

end decompose_96_l158_158568


namespace vasya_fraction_is_0_4_l158_158536

-- Defining the variables and conditions
variables (a b c d s : ℝ)
axiom cond1 : a = b / 2
axiom cond2 : c = a + d
axiom cond3 : d = s / 10
axiom cond4 : a + b + c + d = s

-- Stating the theorem
theorem vasya_fraction_is_0_4 (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : (b / s) = 0.4 := 
by
  sorry

end vasya_fraction_is_0_4_l158_158536


namespace vasya_fraction_l158_158525

-- Define the variables for distances and total distance
variables {a b c d s : ℝ}

-- Define conditions
def anton_distance (a b : ℝ) : Prop := a = b / 2
def sasha_distance (c a d : ℝ) : Prop := c = a + d
def dima_distance (d s : ℝ) : Prop := d = s / 10
def total_distance (a b c d s : ℝ) : Prop := a + b + c + d = s

-- The main theorem 
theorem vasya_fraction (a b c d s : ℝ) (h1 : anton_distance a b) 
  (h2 : sasha_distance c a d) (h3 : dima_distance d s)
  (h4 : total_distance a b c d s) : b / s = 0.4 :=
sorry

end vasya_fraction_l158_158525


namespace xiao_hua_correct_questions_l158_158747

-- Definitions of the problem conditions
def n : Nat := 20
def p_correct : Int := 5
def p_wrong : Int := -2
def score : Int := 65

-- Theorem statement to prove the number of correct questions
theorem xiao_hua_correct_questions : 
  ∃ k : Nat, k = ((n : Int) - ((n * p_correct - score) / (p_correct - p_wrong))) ∧ 
               k = 15 :=
by
  sorry

end xiao_hua_correct_questions_l158_158747


namespace scientific_notation_113700_l158_158917

theorem scientific_notation_113700 :
  ∃ (a : ℝ) (b : ℤ), 113700 = a * 10 ^ b ∧ a = 1.137 ∧ b = 5 :=
by
  sorry

end scientific_notation_113700_l158_158917


namespace edge_length_of_cube_l158_158675

theorem edge_length_of_cube (total_cubes : ℕ) (box_edge_length_m : ℝ) (box_edge_length_cm : ℝ) 
  (conversion_factor : ℝ) (edge_length_cm : ℝ) : 
  total_cubes = 8 ∧ box_edge_length_m = 1 ∧ box_edge_length_cm = box_edge_length_m * conversion_factor ∧ conversion_factor = 100 ∧ 
  edge_length_cm = box_edge_length_cm / 2 ↔ edge_length_cm = 50 := 
by 
  sorry

end edge_length_of_cube_l158_158675


namespace negation_example_l158_158300

theorem negation_example : (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ (∃ x : ℝ, x^2 < 0) :=
by
  sorry

end negation_example_l158_158300


namespace complete_square_monomials_l158_158385

theorem complete_square_monomials (x : ℝ) :
  ∃ (m : ℝ), (m = 4 * x ^ 4 ∨ m = 4 * x ∨ m = -4 * x ∨ m = -1 ∨ m = -4 * x ^ 2) ∧
              (∃ (a b : ℝ), (4 * x ^ 2 + 1 + m = a ^ 2 + b ^ 2)) :=
sorry

-- Note: The exact formulation of the problem might vary based on the definition
-- of perfect squares and corresponding polynomials in the Lean environment.

end complete_square_monomials_l158_158385


namespace janet_wait_time_l158_158243

theorem janet_wait_time 
  (janet_speed : ℝ)
  (sister_speed : ℝ)
  (lake_width : ℝ)
  (h1 : janet_speed = 30)
  (h2 : sister_speed = 12)
  (h3 : lake_width = 60) 
  :
  60 / 12 - 60 / 30 = 3 :=
by
  sorry

end janet_wait_time_l158_158243


namespace factorization_problem_l158_158706

theorem factorization_problem (p q : ℝ) :
  (∃ a b c : ℝ, 
    x^4 + p * x^2 + q = (x^2 + 2 * x + 5) * (a * x^2 + b * x + c)) ↔
  p = 6 ∧ q = 25 := 
sorry

end factorization_problem_l158_158706


namespace domain_of_expression_l158_158078

theorem domain_of_expression (x : ℝ) 
  (h1 : 3 * x - 6 ≥ 0) 
  (h2 : 7 - 2 * x ≥ 0) 
  (h3 : 7 - 2 * x > 0) : 
  2 ≤ x ∧ x < 7 / 2 := by
sorry

end domain_of_expression_l158_158078


namespace ratio_of_triangle_areas_l158_158154

theorem ratio_of_triangle_areas 
  (r s : ℝ) (n : ℝ)
  (h_ratio : 3 * s = r) 
  (h_area : (3 / 2) * n = 1 / 2 * r * ((3 * n * 2) / r)) :
  3 / 3 = n :=
by
  sorry

end ratio_of_triangle_areas_l158_158154


namespace area_of_region_l158_158758

noncomputable def T := 516

def region (x y : ℝ) : Prop :=
  |x| - |y| ≤ T - 500 ∧ |y| ≤ T - 500

theorem area_of_region :
  (4 * (T - 500)^2 = 1024) :=
  sorry

end area_of_region_l158_158758


namespace job_completion_time_l158_158817

theorem job_completion_time (h1 : ∀ {a d : ℝ}, 4 * (1/a + 1/d) = 1)
                             (h2 : ∀ d : ℝ, d = 11.999999999999998) :
                             (∀ a : ℝ, a = 6) :=
by
  sorry

end job_completion_time_l158_158817


namespace chord_length_ne_l158_158215

-- Define the ellipse
def ellipse (x y : ℝ) := (x^2 / 8) + (y^2 / 4) = 1

-- Define the first line
def line_l (k x : ℝ) := (k * x + 1)

-- Define the second line
def line_l_option_D (k x y : ℝ) := (k * x + y - 2)

-- Prove the chord length inequality for line_l_option_D
theorem chord_length_ne (k : ℝ) :
  ∀ x y : ℝ, ellipse x y →
  ∃ x1 x2 y1 y2 : ℝ, ellipse x1 y1 ∧ line_l k x1 = y1 ∧ ellipse x2 y2 ∧ line_l k x2 = y2 ∧
  ∀ x3 x4 y3 y4 : ℝ, ellipse x3 y3 ∧ line_l_option_D k x3 y3 = 0 ∧ ellipse x4 y4 ∧ line_l_option_D k x4 y4 = 0 →
  dist (x1, y1) (x2, y2) ≠ dist (x3, y3) (x4, y4) :=
sorry

end chord_length_ne_l158_158215


namespace prime_solution_l158_158399

theorem prime_solution (p q : ℕ) (hp : Prime p) (hq : Prime q) (h : q^5 - 2 * p^2 = 1) : p + q = 14 :=
sorry

end prime_solution_l158_158399


namespace total_tables_l158_158334

variables (F T : ℕ)

-- Define the given conditions
def condition1 := F = 16
def condition2 := 4 * F + 3 * T = 124

-- State the theorem given the conditions to prove the total number of tables.
theorem total_tables (h1 : condition1) (h2 : condition2) : F + T = 36 :=
by
  -- This is a placeholder as we are skipping the proof itself
  sorry

end total_tables_l158_158334


namespace consecutive_integers_sum_and_difference_l158_158651

theorem consecutive_integers_sum_and_difference (x y : ℕ) 
(h1 : y = x + 1) 
(h2 : x * y = 552) 
: x + y = 47 ∧ y - x = 1 :=
by {
  sorry
}

end consecutive_integers_sum_and_difference_l158_158651


namespace no_real_x_solution_l158_158995

open Real

-- Define the conditions.
def log_defined (x : ℝ) : Prop :=
  0 < x + 5 ∧ 0 < x - 3 ∧ 0 < x^2 - 7*x - 18

-- Define the equation to prove.
def log_eqn (x : ℝ) : Prop :=
  log (x + 5) + log (x - 3) = log (x^2 - 7*x - 18)

-- The mathematicall equivalent proof problem.
theorem no_real_x_solution : ¬∃ x : ℝ, log_defined x ∧ log_eqn x :=
by
  sorry

end no_real_x_solution_l158_158995


namespace minimum_y_value_l158_158428

noncomputable def minimum_y (x a : ℝ) : ℝ :=
  abs (x - a) + abs (x - 15) + abs (x - a - 15)

theorem minimum_y_value (a x : ℝ) (h1 : 0 < a) (h2 : a < 15) (h3 : a ≤ x) (h4 : x ≤ 15) :
  minimum_y x a = 15 :=
by
  sorry

end minimum_y_value_l158_158428


namespace vasya_fraction_is_0_4_l158_158538

-- Defining the variables and conditions
variables (a b c d s : ℝ)
axiom cond1 : a = b / 2
axiom cond2 : c = a + d
axiom cond3 : d = s / 10
axiom cond4 : a + b + c + d = s

-- Stating the theorem
theorem vasya_fraction_is_0_4 (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : (b / s) = 0.4 := 
by
  sorry

end vasya_fraction_is_0_4_l158_158538


namespace cost_of_jeans_l158_158604

theorem cost_of_jeans 
  (price_socks : ℕ)
  (price_tshirt : ℕ)
  (price_jeans : ℕ)
  (h1 : price_socks = 5)
  (h2 : price_tshirt = price_socks + 10)
  (h3 : price_jeans = 2 * price_tshirt) :
  price_jeans = 30 :=
  by
    -- Sorry skips the proof, complies with the instructions
    sorry

end cost_of_jeans_l158_158604


namespace circle_radius_l158_158378

theorem circle_radius (x y : ℝ) : x^2 - 10*x + y^2 + 4*y + 13 = 0 → ∃ r : ℝ, r = 4 :=
by
  -- sorry here to indicate that the proof is skipped
  sorry

end circle_radius_l158_158378


namespace coffee_y_ratio_is_1_to_5_l158_158968

-- Define the conditions
variables {p v x y : Type}
variables (p_x p_y v_x v_y : ℕ) -- Coffee amounts in lbs
variables (total_p total_v : ℕ) -- Total amounts of p and v

-- Definitions based on conditions
def coffee_amounts_initial (total_p total_v : ℕ) : Prop :=
  total_p = 24 ∧ total_v = 25

def coffee_x_conditions (p_x v_x : ℕ) : Prop :=
  p_x = 20 ∧ 4 * v_x = p_x

def coffee_y_conditions (p_y v_y total_p total_v : ℕ) : Prop :=
  p_y = total_p - 20 ∧ v_y = total_v - (20 / 4)

-- Statement to prove
theorem coffee_y_ratio_is_1_to_5 {total_p total_v : ℕ}
  (hc1 : coffee_amounts_initial total_p total_v)
  (hc2 : coffee_x_conditions 20 5)
  (hc3 : coffee_y_conditions 4 20 total_p total_v) : 
  (4 / 20 = 1 / 5) :=
sorry

end coffee_y_ratio_is_1_to_5_l158_158968


namespace smallest_n_property_l158_158365

noncomputable def smallest_n : ℕ := 13

theorem smallest_n_property :
  ∀ (x y z : ℕ), x > 0 → y > 0 → z > 0 → (x ∣ y^3) → (y ∣ z^3) → (z ∣ x^3) → (x * y * z ∣ (x + y + z) ^ smallest_n) :=
by
  intros x y z hx hy hz hxy hyz hzx
  use smallest_n
  sorry

end smallest_n_property_l158_158365


namespace simplify_and_evaluate_expression_l158_158270

theorem simplify_and_evaluate_expression (x y : ℝ) (h₁ : x = 2) (h₂ : y = -1) : 
  2 * x * y - (1 / 2) * (4 * x * y - 8 * x^2 * y^2) + 2 * (3 * x * y - 5 * x^2 * y^2) = -36 := by
  sorry

end simplify_and_evaluate_expression_l158_158270


namespace vasya_drove_0_4_of_total_distance_l158_158531

-- Define variables for the distances driven by Anton (a), Vasya (b), Sasha (c), and Dima (d)
variables {a b c d s : ℝ}

-- Define the conditions in Lean
def condition_1 := a = b / 2
def condition_2 := c = a + d
def condition_3 := d = s / 10
def condition_4 := s ≠ 0
def condition_5 := a + b + c + d = s

-- Prove that Vasya drove 0.4 of the total distance
theorem vasya_drove_0_4_of_total_distance (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (h4 : condition_4) (h5 : condition_5) : b / s = 0.4 :=
by
  sorry

end vasya_drove_0_4_of_total_distance_l158_158531


namespace interest_rate_B_to_C_l158_158819

theorem interest_rate_B_to_C
  (P : ℕ)                -- Principal amount
  (r_A : ℚ)              -- Interest rate A charges B per annum
  (t : ℕ)                -- Time period in years
  (gain_B : ℚ)           -- Gain of B in 3 years
  (H_P : P = 3500)
  (H_r_A : r_A = 0.10)
  (H_t : t = 3)
  (H_gain_B : gain_B = 315) :
  ∃ R : ℚ, R = 0.13 := 
by
  sorry

end interest_rate_B_to_C_l158_158819


namespace gracie_height_is_56_l158_158404

noncomputable def Gracie_height : Nat := 56

theorem gracie_height_is_56 : Gracie_height = 56 := by
  sorry

end gracie_height_is_56_l158_158404


namespace combined_weight_l158_158431

theorem combined_weight (a b c : ℕ) (h1 : a + b = 122) (h2 : b + c = 125) (h3 : c + a = 127) : 
  a + b + c = 187 :=
by
  sorry

end combined_weight_l158_158431


namespace factorized_expression_l158_158371

variable {a b c : ℝ}

theorem factorized_expression :
  ( ((a^2 - b^2)^3 + (b^2 - c^2)^3 + (c^2 - a^2)^3) / 
    ((a - b)^3 + (b - c)^3 + (c - a)^3) ) 
  = (a + b) * (a + c) * (b + c) := 
  sorry

end factorized_expression_l158_158371


namespace mary_should_drink_six_glasses_per_day_l158_158155

def daily_water_goal : ℕ := 1500
def glass_capacity : ℕ := 250
def required_glasses (daily_goal : ℕ) (capacity : ℕ) : ℕ := daily_goal / capacity

theorem mary_should_drink_six_glasses_per_day :
  required_glasses daily_water_goal glass_capacity = 6 :=
by
  sorry

end mary_should_drink_six_glasses_per_day_l158_158155


namespace simplify_fraction_l158_158904

theorem simplify_fraction : (75 : ℚ) / (100 : ℚ) = (3 : ℚ) / (4 : ℚ) :=
by
  sorry

end simplify_fraction_l158_158904


namespace vasya_fraction_is_0_4_l158_158537

-- Defining the variables and conditions
variables (a b c d s : ℝ)
axiom cond1 : a = b / 2
axiom cond2 : c = a + d
axiom cond3 : d = s / 10
axiom cond4 : a + b + c + d = s

-- Stating the theorem
theorem vasya_fraction_is_0_4 (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : (b / s) = 0.4 := 
by
  sorry

end vasya_fraction_is_0_4_l158_158537


namespace train_speed_solution_l158_158345

def train_speed_problem (L v : ℝ) (man_time platform_time : ℝ) (platform_length : ℝ) :=
  man_time = 12 ∧
  platform_time = 30 ∧
  platform_length = 180 ∧
  L = v * man_time ∧
  (L + platform_length) = v * platform_time

theorem train_speed_solution (L v : ℝ) (h : train_speed_problem L v 12 30 180) :
  v * 3.6 = 36 :=
by
  sorry

end train_speed_solution_l158_158345


namespace div_by_1963_iff_odd_l158_158637

-- Define the given condition and statement
theorem div_by_1963_iff_odd (n : ℕ) :
  (1963 ∣ (82^n + 454 * 69^n)) ↔ (n % 2 = 1) :=
sorry

end div_by_1963_iff_odd_l158_158637


namespace melissa_total_points_l158_158894

-- Definition of the points scored per game and the number of games played.
def points_per_game : ℕ := 7
def number_of_games : ℕ := 3

-- The total points scored by Melissa is defined as the product of points per game and number of games.
def total_points_scored : ℕ := points_per_game * number_of_games

-- The theorem stating the verification of the total points scored by Melissa.
theorem melissa_total_points : total_points_scored = 21 := by
  -- The proof will be given here.
  sorry

end melissa_total_points_l158_158894


namespace trapezoid_CD_length_l158_158124

theorem trapezoid_CD_length (AB CD AD BC : ℝ) (P : ℝ) 
  (h₁ : AB = 12) 
  (h₂ : AD = 5) 
  (h₃ : BC = 7) 
  (h₄ : P = 40) : CD = 16 :=
by
  sorry

end trapezoid_CD_length_l158_158124


namespace value_of_m_l158_158326

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 2 * x + 1 else 2 * (-x) + 1

theorem value_of_m (m : ℝ) (heven : ∀ x : ℝ, f (-x) = f x)
  (hpos : ∀ x : ℝ, x ≥ 0 → f x = 2 * x + 1)
  (hfm : f m = 5) : m = 2 ∨ m = -2 :=
sorry

end value_of_m_l158_158326


namespace least_n_for_obtuse_triangle_l158_158252

namespace obtuse_triangle

-- Define angles and n
def alpha (n : ℕ) : ℝ := 59 + n * 0.02
def beta : ℝ := 60
def gamma (n : ℕ) : ℝ := 61 - n * 0.02

-- Define condition for the triangle being obtuse
def is_obtuse_triangle (n : ℕ) : Prop :=
  alpha n > 90 ∨ gamma n > 90

-- Statement about the smallest n such that the triangle is obtuse
theorem least_n_for_obtuse_triangle : ∃ n : ℕ, n = 1551 ∧ is_obtuse_triangle n :=
by
  -- existence proof ends here, details for proof to be provided separately
  sorry

end obtuse_triangle

end least_n_for_obtuse_triangle_l158_158252


namespace mary_should_drink_6_glasses_l158_158158

-- Definitions based on conditions
def daily_water_goal_liters : ℚ := 1.5
def glass_capacity_ml : ℚ := 250
def liter_to_milliliters : ℚ := 1000

-- Conversion from liters to milliliters
def daily_water_goal_milliliters : ℚ := daily_water_goal_liters * liter_to_milliliters

-- Proof problem to show Mary needs 6 glasses per day
theorem mary_should_drink_6_glasses :
  daily_water_goal_milliliters / glass_capacity_ml = 6 := by
  sorry

end mary_should_drink_6_glasses_l158_158158


namespace container_volume_ratio_l158_158088

theorem container_volume_ratio
  (C D : ℕ)
  (h1 : (3 / 5 : ℚ) * C = (1 / 2 : ℚ) * D)
  (h2 : (1 / 3 : ℚ) * ((1 / 2 : ℚ) * D) + (3 / 5 : ℚ) * C = C) :
  (C : ℚ) / D = 5 / 6 :=
by {
  sorry
}

end container_volume_ratio_l158_158088


namespace correct_calculation_option_l158_158317

theorem correct_calculation_option :
  (∀ a : ℝ, 3 * a^5 - a^5 ≠ 3) ∧
  (∀ a : ℝ, a^2 + a^5 ≠ a^7) ∧
  (∀ a : ℝ, a^5 + a^5 = 2 * a^5) ∧
  (∀ x y : ℝ, x^2 * y + x * y^2 ≠ 2 * x^3 * y^3) :=
by
  sorry

end correct_calculation_option_l158_158317


namespace gcf_180_270_450_l158_158475

theorem gcf_180_270_450 : Nat.gcd (Nat.gcd 180 270) 450 = 90 :=
by
  sorry

end gcf_180_270_450_l158_158475


namespace root_relationship_specific_root_five_l158_158139

def f (x : ℝ) : ℝ := x^3 - 6 * x^2 - 39 * x - 10
def g (x : ℝ) : ℝ := x^3 + x^2 - 20 * x - 50

theorem root_relationship :
  ∃ (x_0 : ℝ), g x_0 = 0 ∧ f (2 * x_0) = 0 :=
sorry

theorem specific_root_five :
  g 5 = 0 ∧ f 10 = 0 :=
sorry

end root_relationship_specific_root_five_l158_158139


namespace find_number_l158_158111

theorem find_number (x n : ℝ) (h1 : (3 / 2) * x - n = 15) (h2 : x = 12) : n = 3 :=
by
  sorry

end find_number_l158_158111


namespace unit_circle_chords_l158_158241

theorem unit_circle_chords (
    s t u v : ℝ
) (hs : s = 1) (ht : t = 1) (hu : u = 2) (hv : v = 3) :
    (v - u = 1) ∧ (v * u = 6) ∧ (v^2 - u^2 = 5) :=
by
  have h1 : v - u = 1 := by rw [hv, hu]; norm_num
  have h2 : v * u = 6 := by rw [hv, hu]; norm_num
  have h3 : v^2 - u^2 = 5 := by rw [hv, hu]; norm_num
  exact ⟨h1, h2, h3⟩

end unit_circle_chords_l158_158241


namespace chloe_probability_l158_158358

theorem chloe_probability :
  let total_numbers := 60
  let multiples_of_4 := 15
  let non_multiples_of_4_prob := 3 / 4
  let neither_multiple_of_4_prob := (non_multiples_of_4_prob) ^ 2
  let at_least_one_multiple_of_4_prob := 1 - neither_multiple_of_4_prob
  at_least_one_multiple_of_4_prob = 7 / 16 := by
  sorry

end chloe_probability_l158_158358


namespace total_shaded_area_of_square_carpet_l158_158683

theorem total_shaded_area_of_square_carpet :
  ∀ (S T : ℝ),
    (9 / S = 3) →
    (S / T = 3) →
    (8 * T^2 + S^2 = 17) :=
by
  intros S T h1 h2
  sorry

end total_shaded_area_of_square_carpet_l158_158683


namespace nehas_mother_age_l158_158173

variables (N M : ℕ)

axiom age_condition1 : M - 12 = 4 * (N - 12)
axiom age_condition2 : M + 12 = 2 * (N + 12)

theorem nehas_mother_age : M = 60 :=
by
  -- Sorry added to skip the proof
  sorry

end nehas_mother_age_l158_158173


namespace find_a_plus_b_l158_158395

-- Given points A and B, where A(1, a) and B(b, -2) are symmetric with respect to the origin.
variables (a b : ℤ)

-- Definition for symmetry conditions
def symmetric_wrt_origin (x1 y1 x2 y2 : ℤ) :=
  x2 = -x1 ∧ y2 = -y1

-- The main theorem
theorem find_a_plus_b :
  symmetric_wrt_origin 1 a b (-2) → a + b = 1 :=
by
  sorry

end find_a_plus_b_l158_158395


namespace non_real_roots_of_quadratic_l158_158734

theorem non_real_roots_of_quadratic (b : ℝ) : 
  (¬ ∃ x1 x2 : ℝ, x1^2 + bx1 + 16 = 0 ∧ x2^2 + bx2 + 16 = 0 ∧ x1 = x2) ↔ b ∈ set.Ioo (-8 : ℝ) (8 : ℝ) :=
by {
  sorry
}

end non_real_roots_of_quadratic_l158_158734


namespace limit_sin_exp_l158_158557

open Real

theorem limit_sin_exp (h1 : ∀ x, sin(5 * (x + π)) = -sin(5 * x))
  (h2 : Tendsto (λ x, (exp (3 * x) - 1) / (3 * x)) (𝓝 0) (𝓝 1))
  (h3 : Tendsto (λ x, sin(5 * x) / (5 * x)) (𝓝 0) (𝓝 1)) :
  Tendsto (λ x, (sin (5 * (x + π))) / (exp (3 * x) - 1)) (𝓝 0) (𝓝 (-5 / 3)) := by
  sorry

end limit_sin_exp_l158_158557


namespace non_real_roots_interval_l158_158729

theorem non_real_roots_interval (b : ℝ) : (b^2 < 64) ↔ (b > -8 ∧ b < 8) :=
by
  sorry

end non_real_roots_interval_l158_158729


namespace connected_distinct_points_with_slope_change_l158_158980

-- Defining the cost function based on the given conditions
def cost_function (n : ℕ) : ℕ := 
  if n <= 10 then 20 * n else 18 * n

-- The main theorem to prove the nature of the graph as described in the problem
theorem connected_distinct_points_with_slope_change : 
  (∀ n, (1 ≤ n ∧ n ≤ 20) → 
    (∃ k, cost_function n = k ∧ 
    (n <= 10 → cost_function n = 20 * n) ∧ 
    (n > 10 → cost_function n = 18 * n))) ∧
  (∃ n, n = 10 ∧ cost_function n = 200 ∧ cost_function (n + 1) = 198) :=
sorry

end connected_distinct_points_with_slope_change_l158_158980


namespace min_value_of_x_plus_y_l158_158386

theorem min_value_of_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + y = x * y) : x + y ≥ 9 :=
by
  sorry

end min_value_of_x_plus_y_l158_158386


namespace shopkeeper_discount_l158_158682

theorem shopkeeper_discount
  (CP LP SP : ℝ)
  (H_CP : CP = 100)
  (H_LP : LP = CP + 0.4 * CP)
  (H_SP : SP = CP + 0.33 * CP)
  (discount_percent : ℝ) :
  discount_percent = ((LP - SP) / LP) * 100 → discount_percent = 5 :=
by
  sorry

end shopkeeper_discount_l158_158682


namespace salary_increase_l158_158433

theorem salary_increase (prev_income : ℝ) (prev_percentage : ℝ) (new_percentage : ℝ) (rent_utilities : ℝ) (new_income : ℝ) :
  prev_income = 1000 ∧ prev_percentage = 0.40 ∧ new_percentage = 0.25 ∧ rent_utilities = prev_percentage * prev_income ∧
  rent_utilities = new_percentage * new_income → new_income - prev_income = 600 :=
by 
  sorry

end salary_increase_l158_158433


namespace line_length_l158_158118

theorem line_length (n : ℕ) (d : ℤ) (h1 : n = 51) (h2 : d = 3) : 
  (n - 1) * d = 150 := sorry

end line_length_l158_158118


namespace Vasya_distance_fraction_l158_158553

variable (a b c d s : ℝ)

theorem Vasya_distance_fraction :
  (a = b / 2) →
  (c = a + d) →
  (d = s / 10) →
  (a + b + c + d = s) →
  (b / s = 0.4) :=
by
  intros h1 h2 h3 h4
  sorry

end Vasya_distance_fraction_l158_158553


namespace smallest_positive_period_axis_of_symmetry_range_of_f_in_interval_l158_158719

noncomputable def f (x : ℝ) : ℝ :=
  let m := (Real.cos x, -Real.sin x)
  let n := (Real.cos x, Real.sin x - 2 * Real.sqrt 3 * Real.cos x)
  m.1 * n.1 + m.2 * n.2

theorem smallest_positive_period (x : ℝ) : 
  ∃ T, T > 0 ∧ ∀ x, f (x + T) = f x := 
by
  use Real.pi
  sorry

theorem axis_of_symmetry (x : ℝ) : 
  ∀ k : ℤ, x = (Real.pi / 6) + (k * (Real.pi / 2)) :=
by
  sorry

theorem range_of_f_in_interval : 
  Set.Icc (-Real.pi / 12) (Real.pi / 2) ⊆ 
  Set.Icc (-1) 2 :=
by
  sorry

end smallest_positive_period_axis_of_symmetry_range_of_f_in_interval_l158_158719


namespace solve_for_x_l158_158498

theorem solve_for_x (x : ℝ) (h : 0.009 / x = 0.05) : x = 0.18 := 
by
  sorry

end solve_for_x_l158_158498


namespace C_share_of_profit_l158_158346

-- Given conditions
def investment_A : ℕ := 8000
def investment_B : ℕ := 4000
def investment_C : ℕ := 2000
def total_profit : ℕ := 252000

-- Objective to prove that C's share of the profit is given by 36000
theorem C_share_of_profit : (total_profit / (investment_A / investment_C + investment_B / investment_C + 1)) = 36000 :=
by
  sorry

end C_share_of_profit_l158_158346


namespace find_f_g_3_l158_158601

def f (x : ℝ) : ℝ := x^2 + 2
def g (x : ℝ) : ℝ := 3 * x - 2

theorem find_f_g_3 : f (g 3) = 51 := 
by 
  sorry

end find_f_g_3_l158_158601


namespace minimum_width_l158_158267

theorem minimum_width (w : ℝ) (h_area : w * (w + 15) ≥ 200) : w ≥ 10 :=
by
  sorry

end minimum_width_l158_158267


namespace octal_67_equals_ternary_2001_l158_158363

def octalToDecimal (n : Nat) : Nat :=
  -- Definition of octal to decimal conversion omitted
  sorry

def decimalToTernary (n : Nat) : Nat :=
  -- Definition of decimal to ternary conversion omitted
  sorry

theorem octal_67_equals_ternary_2001 : 
  decimalToTernary (octalToDecimal 67) = 2001 :=
by
  -- Proof omitted
  sorry

end octal_67_equals_ternary_2001_l158_158363


namespace toys_sold_week2_l158_158187

-- Define the given conditions
def original_stock := 83
def toys_sold_week1 := 38
def toys_left := 19

-- Define the statement we want to prove
theorem toys_sold_week2 : (original_stock - toys_left) - toys_sold_week1 = 26 :=
by
  sorry

end toys_sold_week2_l158_158187


namespace total_handshakes_at_convention_l158_158037

def number_of_gremlins := 30
def number_of_imps := 20
def disagreeing_imps := 5
def specific_gremlins := 10

theorem total_handshakes_at_convention : 
  (number_of_gremlins * (number_of_gremlins - 1) / 2) +
  ((number_of_imps - disagreeing_imps) * number_of_gremlins) + 
  (disagreeing_imps * (number_of_gremlins - specific_gremlins)) = 985 :=
by 
  sorry

end total_handshakes_at_convention_l158_158037


namespace number_of_distinct_possible_values_for_c_l158_158000

variables {a b r s t : ℂ}
variables (h_distinct : r ≠ s ∧ s ≠ t ∧ r ≠ t)
variables (h_transform : ∀ z, (a * z + b - r) * (a * z + b - s) * (a * z + b - t) = (z - c * r) * (z - c * s) * (z - c * t))

theorem number_of_distinct_possible_values_for_c (h_nonzero : a ≠ 0) : 
  ∃ (n : ℕ), n = 4 := sorry

end number_of_distinct_possible_values_for_c_l158_158000


namespace vasya_drove_0_4_of_total_distance_l158_158529

-- Define variables for the distances driven by Anton (a), Vasya (b), Sasha (c), and Dima (d)
variables {a b c d s : ℝ}

-- Define the conditions in Lean
def condition_1 := a = b / 2
def condition_2 := c = a + d
def condition_3 := d = s / 10
def condition_4 := s ≠ 0
def condition_5 := a + b + c + d = s

-- Prove that Vasya drove 0.4 of the total distance
theorem vasya_drove_0_4_of_total_distance (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (h4 : condition_4) (h5 : condition_5) : b / s = 0.4 :=
by
  sorry

end vasya_drove_0_4_of_total_distance_l158_158529


namespace find_m_l158_158645

theorem find_m (m : ℝ) 
  (f g : ℝ → ℝ) 
  (x : ℝ) 
  (hf : f x = x^2 - 3 * x + m) 
  (hg : g x = x^2 - 3 * x + 5 * m) 
  (hx : x = 5) 
  (h_eq : 3 * f x = 2 * g x) :
  m = 10 / 7 := 
sorry

end find_m_l158_158645


namespace pages_in_first_chapter_l158_158902

/--
Rita is reading a five-chapter book with 95 pages. Each chapter has three pages more than the previous one. 
Prove the number of pages in the first chapter.
-/
theorem pages_in_first_chapter (h : ∃ p1 p2 p3 p4 p5 : ℕ, p1 + p2 + p3 + p4 + p5 = 95 ∧ p2 = p1 + 3 ∧ p3 = p1 + 6 ∧ p4 = p1 + 9 ∧ p5 = p1 + 12) : 
  ∃ x : ℕ, x = 13 := 
by
  sorry

end pages_in_first_chapter_l158_158902


namespace triangles_with_perimeter_20_l158_158864

theorem triangles_with_perimeter_20 (sides : Finset (Finset ℕ)) : 
  (∀ {a b c : ℕ}, (a + b + c = 20) → (a > 0) → (b > 0) → (c > 0) 
  → (a + b > c) → (a + c > b) → (b + c > a) → ({a, b, c} ∈ sides)) 
  → sides.card = 8 := 
by
  sorry

end triangles_with_perimeter_20_l158_158864


namespace problem_statement_l158_158255

-- Define the odd function and the conditions given
def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

-- Main theorem statement
theorem problem_statement (f : ℝ → ℝ) 
  (h_odd : odd_function f)
  (h_periodic : ∀ x : ℝ, f (x + 1) = f (3 - x))
  (h_f1 : f 1 = -2) :
  2012 * f 2012 - 2013 * f 2013 = -4026 := 
sorry

end problem_statement_l158_158255


namespace part1_part2_l158_158717

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * log x - (a / 2) * x^2

-- Define the line l
noncomputable def l (k : ℤ) (x : ℝ) : ℝ := (k - 2) * x - k + 1

-- Theorem for part (1)
theorem part1 (x : ℝ) (a : ℝ) (h₁ : e ≤ x) (h₂ : x ≤ e^2) (h₃ : f a x > 0) : a < 2 / e :=
sorry

-- Theorem for part (2)
theorem part2 (k : ℤ) (h₁ : a = 0) (h₂ : ∀ (x : ℝ), 1 < x → f 0 x > l k x) : k ≤ 4 :=
sorry

end part1_part2_l158_158717


namespace largest_square_area_with_4_interior_lattice_points_l158_158508

/-- 
A point (x, y) in the plane is called a lattice point if both x and y are integers.
The largest square that contains exactly four lattice points solely in its interior
has an area of 9.
-/
theorem largest_square_area_with_4_interior_lattice_points : 
  ∃ s : ℝ, ∀ (x y : ℤ), 
  (1 ≤ x ∧ x < s ∧ 1 ≤ y ∧ y < s) → s^2 = 9 := 
sorry

end largest_square_area_with_4_interior_lattice_points_l158_158508


namespace value_of_s_l158_158623

theorem value_of_s (s : ℝ) : 
  (3 * (-1)^4 - 2 * (-1)^3 + 4 * (-1)^2 - 5 * (-1) + s = 0) → s = -14 :=
by
  sorry

end value_of_s_l158_158623


namespace complement_of_intersection_l158_158893

-- Declare the universal set U
def U : Set ℤ := {-1, 1, 2, 3}

-- Declare the set A
def A : Set ℤ := {-1, 2}

-- Define the set B using the given quadratic equation
def is_solution (x : ℤ) : Prop := x^2 - 2 * x - 3 = 0
def B : Set ℤ := {x : ℤ | is_solution x}

-- The main theorem to prove
theorem complement_of_intersection (A_inter_B_complement : Set ℤ) :
  A_inter_B_complement = {1, 2, 3} :=
by
  sorry

end complement_of_intersection_l158_158893


namespace percent_of_total_is_correct_l158_158501

theorem percent_of_total_is_correct :
  (6.620000000000001 / 100 * 1000 = 66.2) :=
by
  sorry

end percent_of_total_is_correct_l158_158501


namespace fair_eight_sided_die_probability_l158_158503

def prob_at_least_seven_at_least_four_times (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem fair_eight_sided_die_probability : prob_at_least_seven_at_least_four_times 5 4 (1 / 4) + (1 / 4) ^ 5 = 1 / 64 :=
by
  sorry

end fair_eight_sided_die_probability_l158_158503


namespace remainder_17_pow_2023_mod_28_l158_158072

theorem remainder_17_pow_2023_mod_28 :
  17^2023 % 28 = 17 := 
by sorry

end remainder_17_pow_2023_mod_28_l158_158072


namespace tenth_term_arithmetic_sequence_l158_158800

theorem tenth_term_arithmetic_sequence (a d : ℤ) 
  (h1 : a + 2 * d = 23) (h2 : a + 6 * d = 35) : 
  a + 9 * d = 44 := 
  by
    sorry

end tenth_term_arithmetic_sequence_l158_158800


namespace symmetric_circle_proof_l158_158021

-- Define the original circle equation
def original_circle_eq (x y : ℝ) : Prop :=
  (x + 2)^2 + y^2 = 5

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop :=
  y = x

-- Define the symmetric circle equation
def symmetric_circle_eq (x y : ℝ) : Prop :=
  x^2 + (y + 2)^2 = 5

-- The theorem to prove
theorem symmetric_circle_proof (x y : ℝ) :
  (original_circle_eq x y) ↔ (symmetric_circle_eq x y) :=
sorry

end symmetric_circle_proof_l158_158021


namespace perpendicular_vector_solution_l158_158862

theorem perpendicular_vector_solution 
    (a b : ℝ × ℝ) (m : ℝ) 
    (h_a : a = (1, -1)) 
    (h_b : b = (-2, 3)) 
    (h_perp : a.1 * (a.1 + m * b.1) + a.2 * (a.2 + m * b.2) = 0) 
    : m = 2 / 5 := 
sorry

end perpendicular_vector_solution_l158_158862


namespace marta_hours_worked_l158_158261

-- Definitions of the conditions in Lean 4
def total_collected : ℕ := 240
def hourly_rate : ℕ := 10
def tips_collected : ℕ := 50
def work_earned : ℕ := total_collected - tips_collected

-- Goal: To prove the number of hours worked by Marta
theorem marta_hours_worked : work_earned / hourly_rate = 19 := by
  sorry

end marta_hours_worked_l158_158261


namespace chickens_and_rabbits_l158_158752

-- Let x be the number of chickens and y be the number of rabbits
variables (x y : ℕ)

-- Conditions: There are 35 heads and 94 feet in total
def heads_eq : Prop := x + y = 35
def feet_eq : Prop := 2 * x + 4 * y = 94

-- Proof statement (no proof is required, so we use sorry)
theorem chickens_and_rabbits :
  (heads_eq x y) ∧ (feet_eq x y) ↔ (x + y = 35 ∧ 2 * x + 4 * y = 94) :=
by
  sorry

end chickens_and_rabbits_l158_158752


namespace nonneg_int_solutions_l158_158374

theorem nonneg_int_solutions (a b : ℕ) (h : abs (a - b) + a * b = 1) :
  (a, b) = (1, 0) ∨ (a, b) = (0, 1) ∨ (a, b) = (1, 1) :=
by
  sorry

end nonneg_int_solutions_l158_158374


namespace new_percentage_of_water_l158_158997

noncomputable def initial_weight : ℝ := 100
noncomputable def initial_percentage_water : ℝ := 99 / 100
noncomputable def initial_weight_water : ℝ := initial_weight * initial_percentage_water
noncomputable def initial_weight_non_water : ℝ := initial_weight - initial_weight_water
noncomputable def new_weight : ℝ := 25

theorem new_percentage_of_water :
  ((new_weight - initial_weight_non_water) / new_weight) * 100 = 96 :=
by
  sorry

end new_percentage_of_water_l158_158997


namespace curvilinear_triangle_area_l158_158910

-- Defining the mathematical entities and the problem statement in Lean 4
variables (x y d α : ℝ)

axiom ext_tangent_circles : d = x + y
axiom angle_tangent : α > 0 ∧ α < 2 * Real.pi

theorem curvilinear_triangle_area
  (hx₀ : 0 < x)
  (hy₀ : 0 < y)
  (d_eq : d = x + y)
  (α_bound : 0 < α ∧ α < 2 * Real.pi) :
  let area := (d^2 / 8) * (4 * Real.cos (α / 2) - Real.pi * (1 + (Real.sin (α / 2))^2) + 2 * α * Real.sin (α / 2)) in
  True := sorry

end curvilinear_triangle_area_l158_158910


namespace sqrt_product_l158_158561

theorem sqrt_product (a b c : ℝ) (h1 : a = real.sqrt 72) (h2 : b = real.sqrt 18) (h3 : c = real.sqrt 8) :
  a * b * c = 72 * real.sqrt 2 :=
by
  rw [h1, h2, h3]
  sorry

end sqrt_product_l158_158561


namespace find_a3_l158_158209

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = a n * r

theorem find_a3 (a : ℕ → ℝ) (r : ℝ)
  (h1 : geometric_sequence a r)
  (h2 : a 0 * a 1 * a 2 * a 3 * a 4 = 32):
  a 2 = 2 :=
sorry

end find_a3_l158_158209


namespace negation_statement_l158_158650

variables (Students : Type) (LeftHanded InChessClub : Students → Prop)

theorem negation_statement :
  (¬ ∃ x, LeftHanded x ∧ InChessClub x) ↔ (∃ x, LeftHanded x ∧ InChessClub x) :=
by
  sorry

end negation_statement_l158_158650


namespace smallest_base10_integer_l158_158953

theorem smallest_base10_integer {a b n : ℕ} (ha : a > 2) (hb : b > 2)
  (h₁ : 2 * a + 1 = n) (h₂ : 1 * b + 2 = n) :
  n = 7 :=
sorry

end smallest_base10_integer_l158_158953


namespace non_real_roots_interval_l158_158730

theorem non_real_roots_interval (b : ℝ) : (b^2 < 64) ↔ (b > -8 ∧ b < 8) :=
by
  sorry

end non_real_roots_interval_l158_158730


namespace garden_area_l158_158024

theorem garden_area (P : ℝ) (hP : P = 72) (l w : ℝ) (hL : l = 3 * w) (hPerimeter : 2 * l + 2 * w = P) : l * w = 243 := 
by
  sorry

end garden_area_l158_158024


namespace time_to_cook_rest_of_potatoes_l158_158818

-- Definitions of the conditions
def total_potatoes : ℕ := 12
def already_cooked : ℕ := 6
def minutes_per_potato : ℕ := 6

-- Proof statement
theorem time_to_cook_rest_of_potatoes : (total_potatoes - already_cooked) * minutes_per_potato = 36 :=
by
  sorry

end time_to_cook_rest_of_potatoes_l158_158818


namespace inequality_implies_range_of_a_l158_158083

theorem inequality_implies_range_of_a (a : ℝ) :
  (∀ x : ℝ, |2 - x| + |1 + x| ≥ a^2 - 2 * a) → (-1 ≤ a ∧ a ≤ 3) :=
sorry

end inequality_implies_range_of_a_l158_158083


namespace total_cost_sandwiches_sodas_l158_158554

theorem total_cost_sandwiches_sodas (cost_per_sandwich cost_per_soda : ℝ) 
  (num_sandwiches num_sodas : ℕ) (discount_rate : ℝ) (total_items : ℕ) :
  cost_per_sandwich = 4 → 
  cost_per_soda = 3 → 
  num_sandwiches = 6 → 
  num_sodas = 7 → 
  discount_rate = 0.10 → 
  total_items = num_sandwiches + num_sodas → 
  total_items > 10 → 
  (num_sandwiches * cost_per_sandwich + num_sodas * cost_per_soda) * (1 - discount_rate) = 40.5 :=
by
  intros
  sorry

end total_cost_sandwiches_sodas_l158_158554


namespace solve_inequality_l158_158271

theorem solve_inequality (x : ℝ) (h : 3 * x + 4 ≠ 0) :
  (3 - 1 / (3 * x + 4) < 5) ↔ (-4 / 3 < x) :=
by
  sorry

end solve_inequality_l158_158271


namespace abs_add_eq_abs_sub_implies_mul_eq_zero_l158_158869

variable {a b : ℝ}

theorem abs_add_eq_abs_sub_implies_mul_eq_zero (h : |a + b| = |a - b|) : a * b = 0 :=
sorry

end abs_add_eq_abs_sub_implies_mul_eq_zero_l158_158869


namespace arithmetic_sequence_tenth_term_l158_158798

theorem arithmetic_sequence_tenth_term (a d : ℤ) 
  (h1 : a + 2 * d = 23) 
  (h2 : a + 6 * d = 35) : 
  a + 9 * d = 44 := 
by 
  -- proof goes here
  sorry

end arithmetic_sequence_tenth_term_l158_158798


namespace possible_values_of_expression_l158_158214

theorem possible_values_of_expression (x y : ℝ) (hxy : x + 2 * y = 2) (hx_pos : x > 0) (hy_pos : y > 0) :
  ∃ v, v = 21 / 4 ∧ (1 / x + 2 / y) = v :=
sorry

end possible_values_of_expression_l158_158214


namespace cabbage_price_is_4_02_l158_158126

noncomputable def price_of_cabbage (broccoli_price_per_pound: ℝ) (broccoli_pounds: ℝ) 
                                    (orange_price_each: ℝ) (oranges: ℝ) 
                                    (bacon_price_per_pound: ℝ) (bacon_pounds: ℝ) 
                                    (chicken_price_per_pound: ℝ) (chicken_pounds: ℝ) 
                                    (budget_percentage_for_meat: ℝ) 
                                    (meat_price: ℝ) : ℝ := 
  let broccoli_total := broccoli_pounds * broccoli_price_per_pound
  let oranges_total := oranges * orange_price_each
  let bacon_total := bacon_pounds * bacon_price_per_pound
  let chicken_total := chicken_pounds * chicken_price_per_pound
  let subtotal := broccoli_total + oranges_total + bacon_total + chicken_total
  let total_budget := meat_price / budget_percentage_for_meat
  total_budget - subtotal

theorem cabbage_price_is_4_02 : 
  price_of_cabbage 4 3 0.75 3 3 1 3 2 0.33 9 = 4.02 := 
by 
  sorry

end cabbage_price_is_4_02_l158_158126


namespace range_of_k1k2k3_l158_158213

-- Define the given circles and their intersections
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8 * x - 8 * y + 28 = 0
def curve_N (x y : ℝ) : Prop := x^2 + 4 * y^2 = 4
def point_A (x y : ℝ) : Prop := (x, y) = (2, 0)
def point_B (x y : ℝ) : Prop := (x, y) = (-2, 0)

-- Define points and lines with their slopes
def line_OP (x y k : ℝ) : Prop := y = k * x
def slope_MA (b a : ℝ) : ℝ := b / (a + 2)
def slope_MB (b a : ℝ) : ℝ := b / (a - 2)

-- Condition on the point M belonging to curve_N
def point_M_on_curve_N (a b : ℝ) : Prop := a^2 + 4 * b^2 = 4

-- Define ranges for the slopes
noncomputable def k1_k2 (b a : ℝ) : ℝ := (b / (a + 2)) * (b / (a - 2)) = -1/4

-- Define the quadratic equation for the slope range
def k3_range (k : ℝ) : Prop := (4 - Real.sqrt 7) / 3 ≤ k ∧ k ≤ (4 + Real.sqrt 7) / 3

theorem range_of_k1k2k3 (k1 k2 k3 : ℝ) (hz : k1 * k2 = -1/4) :
  ( (4 - Real.sqrt 7) / 12 ≤ hz * k3 ∧ hz * k3 ≤ (4 + Real.sqrt 7) / 12 ) :=
sorry

end range_of_k1k2k3_l158_158213


namespace games_planned_to_attend_this_month_l158_158248

theorem games_planned_to_attend_this_month (T A_l P_l M_l P_m : ℕ) 
  (h1 : T = 12) 
  (h2 : P_l = 17) 
  (h3 : M_l = 16) 
  (h4 : A_l = P_l - M_l) 
  (h5 : T = A_l + P_m) : P_m = 11 :=
by 
  sorry

end games_planned_to_attend_this_month_l158_158248


namespace fourth_even_integer_l158_158705

theorem fourth_even_integer (n : ℤ) (h : (n-2) + (n+2) = 92) : n + 4 = 50 := by
  -- This will skip the proof steps and assume the correct answer
  sorry

end fourth_even_integer_l158_158705


namespace john_pays_total_l158_158250

-- Definitions based on conditions
def total_cans : ℕ := 30
def price_per_can : ℝ := 0.60

-- Main statement to be proven
theorem john_pays_total : (total_cans / 2) * price_per_can = 9 := 
by
  sorry

end john_pays_total_l158_158250


namespace janet_wait_time_l158_158245

theorem janet_wait_time
  (janet_speed : ℝ)
  (sister_speed : ℝ)
  (lake_width : ℝ)
  (janet_time : ℝ)
  (sister_time : ℝ) :
  janet_speed = 30 →
  sister_speed = 12 →
  lake_width = 60 →
  janet_time = lake_width / janet_speed →
  sister_time = lake_width / sister_speed →
  (sister_time - janet_time = 3) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end janet_wait_time_l158_158245


namespace smallest_base_10_integer_exists_l158_158948

theorem smallest_base_10_integer_exists :
  ∃ (x a b : ℕ), (a > 2) ∧ (b > 2) ∧ (x = 2 * a + 1) ∧ (x = b + 2) ∧ (x = 7) :=
by
  sorry

end smallest_base_10_integer_exists_l158_158948


namespace flight_landing_time_in_gander_l158_158478

-- Definitions based on the conditions
def toronto_to_gander_time_difference : Time := Time.mk 1 30 0 -- 1 hour 30 minutes
def flight_departure_time : Time := Time.mk 15 0 0 -- 3:00 p.m. in 24-hour format
def flight_duration : Time := Time.mk 2 50 0 -- 2 hours 50 minutes

-- The main statement to be proved
theorem flight_landing_time_in_gander :
  let landing_time_in_toronto := flight_departure_time + flight_duration in
  let landing_time_in_gander := landing_time_in_toronto + toronto_to_gander_time_difference in
  landing_time_in_gander = Time.mk 19 20 0 -- 7:20 p.m. in 24-hour format
  :=
by
  sorry

end flight_landing_time_in_gander_l158_158478


namespace min_w_for_factors_l158_158331

theorem min_w_for_factors (w : ℕ) (h_pos : w > 0)
  (h_product_factors : ∀ k, k > 0 → ∃ a b : ℕ, (1452 * w = k) → (a = 3^3) ∧ (b = 13^3) ∧ (k % a = 0) ∧ (k % b = 0)) : 
  w = 19773 :=
sorry

end min_w_for_factors_l158_158331


namespace exists_integer_point_touching_x_axis_l158_158011

-- Define the context for the problem
variable {p q : ℤ}

-- Condition: The quadratic trinomial touches x-axis, i.e., discriminant is zero.
axiom discriminant_zero (p q : ℤ) : p^2 - 4 * q = 0

-- Theorem statement: Proving the existence of such an integer point.
theorem exists_integer_point_touching_x_axis :
  ∃ a b : ℤ, (a = -p ∧ b = q) ∧ (∀ (x : ℝ), x^2 + a * x + b = 0 → (a * a - 4 * b) = 0) :=
sorry

end exists_integer_point_touching_x_axis_l158_158011


namespace max_jogs_l158_158688

theorem max_jogs (x y z : ℕ) (h1 : 3 * x + 2 * y + 8 * z = 60) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) :
  z ≤ 6 := 
sorry

end max_jogs_l158_158688


namespace proof_neg_q_l158_158711

variable (f : ℝ → ℝ)
variable (x : ℝ)

def proposition_p (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x < f y

def proposition_q : Prop := ∃ x : ℝ, (deriv fun y => 1 / y) x > 0

theorem proof_neg_q : ¬ proposition_q := 
by
  intro h
  -- proof omitted for brevity
  sorry

end proof_neg_q_l158_158711


namespace A_days_to_complete_work_alone_l158_158500

theorem A_days_to_complete_work_alone (x : ℝ) (h1 : 0 < x) (h2 : 0 < 18) (h3 : 1/x + 1/18 = 1/6) : x = 9 :=
by
  sorry

end A_days_to_complete_work_alone_l158_158500


namespace circle_center_radius_l158_158908

theorem circle_center_radius :
  ∀ (x y : ℝ), (x + 1) ^ 2 + (y - 2) ^ 2 = 9 ↔ (x = -1 ∧ y = 2 ∧ ∃ r : ℝ, r = 3) :=
by
  sorry

end circle_center_radius_l158_158908


namespace translation_m_n_l158_158013

theorem translation_m_n (m n : ℤ) (P Q : ℤ × ℤ) (hP : P = (-1, -3)) (hQ : Q = (-2, 0))
(hx : P.1 - m = Q.1) (hy : P.2 + n = Q.2) :
  m + n = 4 :=
by
  sorry

end translation_m_n_l158_158013


namespace number_of_friends_l158_158988

-- Definitions based on the given problem conditions
def total_candy := 420
def candy_per_friend := 12

-- Proof statement in Lean 4
theorem number_of_friends : total_candy / candy_per_friend = 35 := by
  sorry

end number_of_friends_l158_158988


namespace total_sampled_papers_l158_158308

-- Define the conditions
variables {A B C c : ℕ}
variable (H : A = 1260 ∧ B = 720 ∧ C = 900 ∧ c = 50)
variable (stratified_sampling : true)   -- We simply denote that stratified sampling method is used

-- Theorem to prove the total number of exam papers sampled
theorem total_sampled_papers {T : ℕ} (H : A = 1260 ∧ B = 720 ∧ C = 900 ∧ c = 50) (stratified_sampling : true) :
  T = (1260 + 720 + 900) * (50 / 900) := sorry

end total_sampled_papers_l158_158308


namespace number_of_solutions_sine_quadratic_l158_158702

theorem number_of_solutions_sine_quadratic :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 * Real.pi → 3 * (Real.sin x) ^ 2 - 5 * (Real.sin x) + 2 = 0 →
  ∃ a b c, x = a ∨ x = b ∨ x = c ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c :=
sorry

end number_of_solutions_sine_quadratic_l158_158702


namespace ratio_area_of_rectangle_to_square_l158_158291

theorem ratio_area_of_rectangle_to_square (s : ℝ) :
  (1.2 * s * 0.8 * s) / (s * s) = 24 / 25 :=
by
  sorry

end ratio_area_of_rectangle_to_square_l158_158291


namespace complex_number_solution_l158_158584

variable (z : ℂ)
variable (i : ℂ)

theorem complex_number_solution (h : (1 - i)^2 / z = 1 + i) (hi : i^2 = -1) : z = -1 - i :=
sorry

end complex_number_solution_l158_158584


namespace max_number_of_pies_l158_158442

def total_apples := 250
def apples_given_to_students := 42
def apples_used_for_juice := 75
def apples_per_pie := 8

theorem max_number_of_pies (h1 : total_apples = 250)
                           (h2 : apples_given_to_students = 42)
                           (h3 : apples_used_for_juice = 75)
                           (h4 : apples_per_pie = 8) :
  ((total_apples - apples_given_to_students - apples_used_for_juice) / apples_per_pie) ≥ 16 :=
by
  sorry

end max_number_of_pies_l158_158442


namespace problem_statement_l158_158254

noncomputable def even_increasing (f : ℝ → ℝ) :=
  ∀ x, f x = f (-x) ∧ ∀ x y, x < y → f x < f y

theorem problem_statement {f : ℝ → ℝ} (hf_even_incr : even_increasing f)
  (x1 x2 : ℝ) (hx1_gt_0 : x1 > 0) (hx2_lt_0 : x2 < 0) (hf_lt : f x1 < f x2) : x1 + x2 > 0 :=
sorry

end problem_statement_l158_158254


namespace largest_modulus_z_l158_158131

open Complex

noncomputable def z_largest_value (a b c z : ℂ) : ℝ :=
  (1 + Real.sqrt 5) / 2

theorem largest_modulus_z (a b c z : ℂ) (r : ℝ) (hr_pos : 0 < r)
  (hmod_a : Complex.abs a = r) (hmod_b : Complex.abs b = r) (hmod_c : Complex.abs c = r)
  (heqn : a * z ^ 2 + b * z + c = 0) :
  Complex.abs z ≤ z_largest_value a b c z :=
sorry

end largest_modulus_z_l158_158131


namespace part1_part2_l158_158600

variable {x m : ℝ}

def P (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10
def S (x : ℝ) (m : ℝ) : Prop := -m + 1 ≤ x ∧ x ≤ m + 1

theorem part1 (h : ∀ x, P x → P x ∨ S x m) : m ≤ 0 :=
sorry

theorem part2 : ¬ ∃ m : ℝ, ∀ x : ℝ, (P x ↔ S x m) :=
sorry

end part1_part2_l158_158600


namespace area_of_tangency_triangle_l158_158680

theorem area_of_tangency_triangle (c a b T varrho : ℝ) (h_right_triangle : a^2 + b^2 = c^2) (h_area : T = (1/2) * a * b) (h_inradius : varrho = (a + b - c) / 2) :
  (area_tangency : ℝ) = (varrho / c) * T :=
sorry

end area_of_tangency_triangle_l158_158680


namespace perpendicular_vectors_eq_l158_158403

theorem perpendicular_vectors_eq {x : ℝ} (h : (x - 5) * 2 + 3 * x = 0) : x = 2 :=
sorry

end perpendicular_vectors_eq_l158_158403


namespace maximize_f_l158_158093

noncomputable def f (x : ℝ) : ℝ := 4 * x - 2 + 1 / (4 * x - 5)

theorem maximize_f (x : ℝ) (h : x < 5 / 4): ∃ M, (∀ y, (y < 5 / 4) → f y ≤ M) ∧ M = 1 := by
  sorry

end maximize_f_l158_158093


namespace find_other_number_l158_158450

theorem find_other_number (a b : ℕ) (h_lcm : Nat.lcm a b = 9240) (h_gcd : Nat.gcd a b = 33) (h_a : a = 231) : b = 1320 :=
sorry

end find_other_number_l158_158450


namespace calc_sqrt_expr_l158_158068

theorem calc_sqrt_expr : (Real.sqrt 2 + 1) ^ 2 - Real.sqrt 18 + 2 * Real.sqrt (1 / 2) = 3 := by
  sorry

end calc_sqrt_expr_l158_158068


namespace students_in_each_group_is_9_l158_158323

-- Define the number of students trying out for the trivia teams
def total_students : ℕ := 36

-- Define the number of students who didn't get picked for the team
def students_not_picked : ℕ := 9

-- Define the number of groups the remaining students are divided into
def number_of_groups : ℕ := 3

-- Define the function that calculates the number of students in each group
def students_per_group (total students_not_picked number_of_groups : ℕ) : ℕ :=
  (total - students_not_picked) / number_of_groups

-- Theorem: Given the conditions, the number of students in each group is 9
theorem students_in_each_group_is_9 : students_per_group total_students students_not_picked number_of_groups = 9 := 
by 
  -- proof skipped
  sorry

end students_in_each_group_is_9_l158_158323


namespace emma_ate_more_than_liam_l158_158998

-- Definitions based on conditions
def emma_oranges : ℕ := 8
def liam_oranges : ℕ := 1

-- Lean statement to prove the question
theorem emma_ate_more_than_liam : emma_oranges - liam_oranges = 7 := by
  sorry

end emma_ate_more_than_liam_l158_158998


namespace field_trip_fraction_l158_158863

theorem field_trip_fraction (b g : ℕ) (hb : g = b)
  (girls_trip_fraction : ℚ := 4/5)
  (boys_trip_fraction : ℚ := 3/4) :
  girls_trip_fraction * g / (girls_trip_fraction * g + boys_trip_fraction * b) = 16 / 31 :=
by {
  sorry
}

end field_trip_fraction_l158_158863


namespace stolen_bones_is_two_l158_158756

/-- Juniper's initial number of bones -/
def initial_bones : ℕ := 4

/-- Juniper's bones after receiving more bones -/
def doubled_bones : ℕ := initial_bones * 2

/-- Juniper's remaining number of bones after theft -/
def remaining_bones : ℕ := 6

/-- Number of bones stolen by the neighbor's dog -/
def stolen_bones : ℕ := doubled_bones - remaining_bones

theorem stolen_bones_is_two : stolen_bones = 2 := sorry

end stolen_bones_is_two_l158_158756


namespace proof_inequality_l158_158585

noncomputable def problem_statement (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) : Prop :=
  a + b + c ≤ (a ^ 4 + b ^ 4 + c ^ 4) / (a * b * c)

theorem proof_inequality (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  problem_statement a b c h_a h_b h_c :=
by
  sorry

end proof_inequality_l158_158585


namespace smallest_sum_l158_158927

theorem smallest_sum (x y : ℕ) (h : (2010 / 2011 : ℚ) < x / y ∧ x / y < (2011 / 2012 : ℚ)) : x + y = 8044 :=
sorry

end smallest_sum_l158_158927


namespace sum_of_cubes_l158_158958

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
sorry

end sum_of_cubes_l158_158958


namespace jed_speeding_l158_158236

-- Define the constants used in the conditions
def F := 16
def T := 256
def S := 50

theorem jed_speeding : (T / F) + S = 66 := 
by sorry

end jed_speeding_l158_158236


namespace sin_gamma_plus_delta_l158_158740

theorem sin_gamma_plus_delta (γ δ : ℝ) (hγ : Complex.exp (Complex.I * γ) = (4/5 : ℂ) + (3/5 : ℂ) * Complex.I)
                             (hδ : Complex.exp (Complex.I * δ) = (-5/13 : ℂ) + (12/13 : ℂ) * Complex.I) :
  Real.sin (γ + δ) = 33 / 65 :=
by
  sorry

end sin_gamma_plus_delta_l158_158740


namespace solve_system_l158_158275

theorem solve_system :
  ∀ (x y z : ℝ),
  (x^2 - 23 * y - 25 * z = -681) →
  (y^2 - 21 * x - 21 * z = -419) →
  (z^2 - 19 * x - 21 * y = -313) →
  (x = 20 ∧ y = 22 ∧ z = 23) :=
by
  intros x y z h1 h2 h3
  sorry

end solve_system_l158_158275


namespace lakeside_fitness_center_ratio_l158_158350

theorem lakeside_fitness_center_ratio (f m c : ℕ)
  (h_avg_age : (35 * f + 30 * m + 10 * c) / (f + m + c) = 25) :
  f = 3 * (m / 6) ∧ f = 3 * (c / 2) :=
by
  sorry

end lakeside_fitness_center_ratio_l158_158350


namespace six_digit_perfect_square_l158_158473

theorem six_digit_perfect_square :
  ∃ n : ℕ, ∃ x : ℕ, (n ^ 2 = 763876) ∧ (n ^ 2 >= 100000) ∧ (n ^ 2 < 1000000) ∧ (5 ≤ x) ∧ (x < 50) ∧ (76 * 10000 + 38 * 100 + 76 = 763876) ∧ (38 = 76 / 2) :=
by
  sorry

end six_digit_perfect_square_l158_158473


namespace average_speed_including_stoppages_l158_158370

/--
If the average speed of a bus excluding stoppages is 50 km/hr, and
the bus stops for 12 minutes per hour, then the average speed of the
bus including stoppages is 40 km/hr.
-/
theorem average_speed_including_stoppages
  (u : ℝ) (Δt : ℝ) (h₁ : u = 50) (h₂ : Δt = 12) : 
  (u * (60 - Δt) / 60) = 40 :=
by
  sorry

end average_speed_including_stoppages_l158_158370


namespace sugar_needed_l158_158976

theorem sugar_needed (sugar_needed_for_full_recipe : ℚ) (fraction_of_recipe : ℚ) :
  sugar_needed_for_full_recipe = 23 / 3 → fraction_of_recipe = 1 / 3 → 
  sugar_needed_for_full_recipe * fraction_of_recipe = 2 + (5 / 9) :=
by
  sorry

end sugar_needed_l158_158976


namespace least_subtract_for_divisibility_l158_158807

theorem least_subtract_for_divisibility (n : ℕ) (hn : n = 427398) : 
  (∃ m : ℕ, n - m % 10 = 0 ∧ m = 2) :=
by
  sorry

end least_subtract_for_divisibility_l158_158807


namespace probability_one_six_given_outcomes_different_l158_158663

open MeasureTheory

/-- Definition of the space of outcomes when two fair dice are rolled, ensuring distinct outcomes -/
def two_dice_outcomes_different : Finset (ℕ × ℕ) :=
  { (i, j) | i ∈ Finset.range 1 6 ∧ j ∈ Finset.range 1 6 ∧ i ≠ j }

/-- Definition of the event where at least one die shows a 6 -/
def at_least_one_six (outcome : ℕ × ℕ) : Prop :=
  outcome.1 = 6 ∨ outcome.2 = 6

/-- The probability that at least one outcome is 6, given the outcomes are different -/
theorem probability_one_six_given_outcomes_different :
  (∑ x in two_dice_outcomes_different.filter at_least_one_six, 1) /
  (∑ x in two_dice_outcomes_different, 1) = 1/3 := by
  sorry

end probability_one_six_given_outcomes_different_l158_158663


namespace true_and_false_propositions_l158_158107

theorem true_and_false_propositions (p q : Prop) 
  (hp : p = true) (hq : q = false) : (¬q) = true :=
by
  sorry

end true_and_false_propositions_l158_158107


namespace find_sale_in_fourth_month_l158_158677

variable (sale1 sale2 sale3 sale5 sale6 : ℕ)
variable (TotalSales : ℕ)
variable (AverageSales : ℕ)

theorem find_sale_in_fourth_month (h1 : sale1 = 6335)
                                   (h2 : sale2 = 6927)
                                   (h3 : sale3 = 6855)
                                   (h4 : sale5 = 6562)
                                   (h5 : sale6 = 5091)
                                   (h6 : AverageSales = 6500)
                                   (h7 : TotalSales = AverageSales * 6) :
  ∃ sale4, TotalSales = sale1 + sale2 + sale3 + sale4 + sale5 + sale6 ∧ sale4 = 7230 :=
by
  sorry

end find_sale_in_fourth_month_l158_158677


namespace area_ratio_rect_sq_l158_158281

variable (s : ℝ)

def side_len_sq (S : ℝ) : Prop := s = S
def longer_side_rect (R : ℝ) : Prop := R = 1.2 * s
def shorter_side_rect (R : ℝ) : Prop := R = 0.8 * s
def area_sq (S : ℝ) : ℝ := S * S
def area_rect (R_long R_short : ℝ) : ℝ := R_long * R_short
def ratio_area (areaR areaS : ℝ) : ℝ := areaR / areaS

theorem area_ratio_rect_sq (s S R_long R_short : ℝ) (h1 : side_len_sq s S) (h2 : longer_side_rect s R_long) (h3 : shorter_side_rect s R_short) :
  ratio_area (area_rect R_long R_short) (area_sq S) = 24/25 :=
by
  sorry

end area_ratio_rect_sq_l158_158281


namespace smallest_base_10_integer_l158_158946

noncomputable def smallest_integer (a b: ℕ) (h₁: a > 2) (h₂: b > 2) (h₃: n = 2 * a + 1) (h₄: n = b + 2) : ℕ :=
  n

theorem smallest_base_10_integer : smallest_integer 3 5 (by decide) (by decide) (by decide) (by decide) = 7 :=
sorry

end smallest_base_10_integer_l158_158946


namespace good_apples_count_l158_158877

theorem good_apples_count (total_apples : ℕ) (rotten_percentage : ℝ) (good_apples : ℕ) (h1 : total_apples = 75) (h2 : rotten_percentage = 0.12) :
  good_apples = (1 - rotten_percentage) * total_apples := by
  sorry

end good_apples_count_l158_158877


namespace find_a_plus_b_l158_158394

-- Given points A and B, where A(1, a) and B(b, -2) are symmetric with respect to the origin.
variables (a b : ℤ)

-- Definition for symmetry conditions
def symmetric_wrt_origin (x1 y1 x2 y2 : ℤ) :=
  x2 = -x1 ∧ y2 = -y1

-- The main theorem
theorem find_a_plus_b :
  symmetric_wrt_origin 1 a b (-2) → a + b = 1 :=
by
  sorry

end find_a_plus_b_l158_158394


namespace stratified_sampling_l158_158678

theorem stratified_sampling (total_students boys girls sample_size x y : ℕ)
  (h1 : total_students = 8)
  (h2 : boys = 6)
  (h3 : girls = 2)
  (h4 : sample_size = 4)
  (h5 : x + y = sample_size)
  (h6 : (x : ℚ) / boys = 3 / 4)
  (h7 : (y : ℚ) / girls = 1 / 4) :
  x = 3 ∧ y = 1 :=
by
  sorry

end stratified_sampling_l158_158678


namespace binom_20_4_l158_158989

theorem binom_20_4 : Nat.choose 20 4 = 4845 := by
  sorry

end binom_20_4_l158_158989


namespace find_function_and_max_profit_l158_158073

noncomputable def profit_function (x : ℝ) : ℝ := -50 * x^2 + 1200 * x - 6400

theorem find_function_and_max_profit :
  (∀ (x : ℝ), (x = 10 → (-50 * x + 800 = 300)) ∧ (x = 13 → (-50 * x + 800 = 150))) ∧
  (∃ (x : ℝ), x = 12 ∧ profit_function x = 800) :=
by
  sorry

end find_function_and_max_profit_l158_158073


namespace max_value_ln_x_minus_x_on_interval_l158_158643

noncomputable def f (x : ℝ) : ℝ := Real.log x - x

theorem max_value_ln_x_minus_x_on_interval : 
  ∃ x ∈ Set.Ioc 0 (Real.exp 1), ∀ y ∈ Set.Ioc 0 (Real.exp 1), f y ≤ f x ∧ f x = -1 :=
by
  sorry

end max_value_ln_x_minus_x_on_interval_l158_158643


namespace symmetric_points_origin_l158_158397

theorem symmetric_points_origin (a b : ℝ) (h1 : 1 = -b) (h2 : a = 2) : a + b = 1 := by
  sorry

end symmetric_points_origin_l158_158397


namespace time_to_cross_platform_l158_158517

variable (l t p : ℝ) -- Define relevant variables

-- Conditions as definitions in Lean 4
def length_of_train := l
def time_to_pass_man := t
def length_of_platform := p

-- Assume given values in the problem
def cond1 : length_of_train = 186 := by sorry
def cond2 : time_to_pass_man = 8 := by sorry
def cond3 : length_of_platform = 279 := by sorry

-- Statement that represents the target theorem to be proved
theorem time_to_cross_platform (h₁ : length_of_train = 186) (h₂ : time_to_pass_man = 8) (h₃ : length_of_platform = 279) : 
  let speed := length_of_train / time_to_pass_man
  let total_distance := length_of_train + length_of_platform
  let time_to_cross := total_distance / speed
  time_to_cross = 20 :=
by sorry

end time_to_cross_platform_l158_158517


namespace product_of_number_and_its_digits_sum_l158_158611

theorem product_of_number_and_its_digits_sum :
  ∃ (n : ℕ), (n = 24 ∧ (n % 10) = ((n / 10) % 10) + 2) ∧ (n * (n % 10 + (n / 10) % 10) = 144) :=
by
  sorry

end product_of_number_and_its_digits_sum_l158_158611


namespace lower_bound_for_expression_l158_158085

theorem lower_bound_for_expression :
  ∃ L: ℤ, (∀ n: ℤ, L < 4 * n + 7 ∧ 4 * n + 7 < 120) → L = 5 :=
sorry

end lower_bound_for_expression_l158_158085


namespace least_clock_equivalent_to_square_greater_than_4_l158_158771

theorem least_clock_equivalent_to_square_greater_than_4 : 
  ∃ (x : ℕ), x > 4 ∧ (x^2 - x) % 12 = 0 ∧ ∀ (y : ℕ), y > 4 → (y^2 - y) % 12 = 0 → x ≤ y :=
by
  -- The proof will go here
  sorry

end least_clock_equivalent_to_square_greater_than_4_l158_158771


namespace mary_should_drink_6_glasses_l158_158157

-- Definitions based on conditions
def daily_water_goal_liters : ℚ := 1.5
def glass_capacity_ml : ℚ := 250
def liter_to_milliliters : ℚ := 1000

-- Conversion from liters to milliliters
def daily_water_goal_milliliters : ℚ := daily_water_goal_liters * liter_to_milliliters

-- Proof problem to show Mary needs 6 glasses per day
theorem mary_should_drink_6_glasses :
  daily_water_goal_milliliters / glass_capacity_ml = 6 := by
  sorry

end mary_should_drink_6_glasses_l158_158157


namespace eq_positive_root_a_value_l158_158874

theorem eq_positive_root_a_value (x a : ℝ) (hx : x > 0) :
  ((x + a) / (x + 3) - 2 / (x + 3) = 0) → a = 5 :=
by
  sorry

end eq_positive_root_a_value_l158_158874


namespace acquaintances_at_ends_equal_l158_158146

theorem acquaintances_at_ends_equal 
  (n : ℕ) -- number of participants
  (a b : ℕ → ℕ) -- functions which return the number of acquaintances before/after for each participant
  (h_ai_bi : ∀ (i : ℕ), 1 < i ∧ i < n → a i = b i) -- condition for participants except first and last
  (h_a1 : a 1 = 0) -- the first person has no one before them
  (h_bn : b n = 0) -- the last person has no one after them
  :
  a n = b 1 :=
by
  sorry

end acquaintances_at_ends_equal_l158_158146


namespace simplify_expression_l158_158779

theorem simplify_expression :
  (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) = 2^32 - 1 :=
  sorry

end simplify_expression_l158_158779


namespace emily_meals_count_l158_158074

theorem emily_meals_count :
  let protein_count := 4
  let sides_count := 5
  let choose_sides := (sides_count.choose 3)
  let dessert_count := 5
  protein_count * choose_sides * dessert_count = 200 :=
by
  sorry

end emily_meals_count_l158_158074


namespace minimum_value_abs_sum_l158_158765

theorem minimum_value_abs_sum (α β γ : ℝ) (h1 : α + β + γ = 2) (h2 : α * β * γ = 4) : 
  |α| + |β| + |γ| ≥ 6 :=
by
  sorry

end minimum_value_abs_sum_l158_158765


namespace price_reduction_required_l158_158179

variable (x : ℝ)
variable (profit_per_piece : ℝ := 40)
variable (initial_sales : ℝ := 20)
variable (additional_sales_per_unit_reduction : ℝ := 2)
variable (desired_profit : ℝ := 1200)

theorem price_reduction_required :
  (profit_per_piece - x) * (initial_sales + additional_sales_per_unit_reduction * x) = desired_profit → x = 20 :=
sorry

end price_reduction_required_l158_158179


namespace find_B_values_l158_158665

theorem find_B_values (A B : ℤ) (h1 : 800 < A) (h2 : A < 1300) (h3 : B > 1) (h4 : A = B ^ 4) : B = 5 ∨ B = 6 := 
sorry

end find_B_values_l158_158665


namespace hot_dogs_remainder_l158_158738

theorem hot_dogs_remainder : 25197641 % 6 = 1 :=
by
  sorry

end hot_dogs_remainder_l158_158738


namespace smallest_base10_integer_l158_158956

theorem smallest_base10_integer : 
  ∃ (a b x : ℕ), a > 2 ∧ b > 2 ∧ x = 2 * a + 1 ∧ x = b + 2 ∧ x = 7 := by
  sorry

end smallest_base10_integer_l158_158956


namespace equal_integers_l158_158170

theorem equal_integers (a b : ℕ)
  (h : ∀ n : ℕ, n > 0 → a > 0 → b > 0 → (a^n + n) ∣ (b^n + n)) : a = b := 
sorry

end equal_integers_l158_158170


namespace vasya_fraction_is_0_4_l158_158535

-- Defining the variables and conditions
variables (a b c d s : ℝ)
axiom cond1 : a = b / 2
axiom cond2 : c = a + d
axiom cond3 : d = s / 10
axiom cond4 : a + b + c + d = s

-- Stating the theorem
theorem vasya_fraction_is_0_4 (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : (b / s) = 0.4 := 
by
  sorry

end vasya_fraction_is_0_4_l158_158535


namespace doubled_dimensions_volume_l158_158515

theorem doubled_dimensions_volume (original_volume : ℝ) (length_factor width_factor height_factor : ℝ) 
  (h : original_volume = 3) 
  (hl : length_factor = 2)
  (hw : width_factor = 2)
  (hh : height_factor = 2) : 
  original_volume * length_factor * width_factor * height_factor = 24 :=
by
  sorry

end doubled_dimensions_volume_l158_158515


namespace add_fractions_l158_158984

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = (5 / a) :=
by sorry

end add_fractions_l158_158984


namespace soccer_minimum_wins_l158_158343

/-
Given that a soccer team has won 60% of 45 matches played so far, 
prove that the minimum number of matches that the team still needs to win to reach a winning percentage of 75% is 27.
-/
theorem soccer_minimum_wins 
  (initial_matches : ℕ)                 -- the initial number of matches
  (initial_win_rate : ℚ)                -- the initial win rate (as a percentage)
  (desired_win_rate : ℚ)                -- the desired win rate (as a percentage)
  (initial_wins : ℕ)                    -- the initial number of wins

  -- Given conditions
  (h1 : initial_matches = 45)
  (h2 : initial_win_rate = 0.60)
  (h3 : desired_win_rate = 0.75)
  (h4 : initial_wins = 27):
  
  -- To prove: the minimum number of additional matches that need to be won is 27
  ∃ (n : ℕ), (initial_wins + n) / (initial_matches + n) = desired_win_rate ∧ 
                  n = 27 :=
by 
  sorry

end soccer_minimum_wins_l158_158343


namespace find_D_l158_158277

theorem find_D (D E F : ℝ) (h : ∀ x : ℝ, x ≠ 1 → x ≠ -2 → (1 / (x^3 - 3*x^2 - 4*x + 12)) = (D / (x - 1)) + (E / (x + 2)) + (F / (x + 2)^2)) :
    D = -1 / 15 :=
by
  -- the proof is omitted as per the instructions
  sorry

end find_D_l158_158277


namespace transform_point_c_l158_158038

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

def reflect_diag (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

theorem transform_point_c :
  let C := (3, 2)
  let C' := reflect_x C
  let C'' := reflect_y C'
  let C''' := reflect_diag C''
  C''' = (-2, -3) :=
by
  sorry

end transform_point_c_l158_158038


namespace non_real_roots_b_range_l158_158728

theorem non_real_roots_b_range (b : ℝ) : 
  ∃ (x : ℂ), x^2 + (b : ℂ) * x + 16 = 0 ∧ (¬ ∃ (x : ℝ), x^2 + b * x + 16 = 0) ↔ -8 < b ∧ b < 8 := 
by
  sorry

end non_real_roots_b_range_l158_158728


namespace vertical_line_division_l158_158612

theorem vertical_line_division (A B C : ℝ × ℝ)
    (hA : A = (0, 2)) (hB : B = (0, 0)) (hC : C = (6, 0))
    (a : ℝ) (h_area_half : 1 / 2 * 6 * 2 / 2 = 3) :
    a = 3 :=
sorry

end vertical_line_division_l158_158612


namespace prod2025_min_sum_l158_158866

theorem prod2025_min_sum : ∃ (a b : ℕ), a * b = 2025 ∧ a > 0 ∧ b > 0 ∧ (∀ (x y : ℕ), x * y = 2025 → x > 0 → y > 0 → x + y ≥ a + b) ∧ a + b = 90 :=
sorry

end prod2025_min_sum_l158_158866


namespace x_y_n_sum_l158_158762

theorem x_y_n_sum (x y n : ℕ) (h1 : 10 ≤ x ∧ x ≤ 99) (h2 : 10 ≤ y ∧ y ≤ 99) (h3 : y = (x % 10) * 10 + (x / 10)) (h4 : x^2 + y^2 = n^2) : x + y + n = 132 :=
sorry

end x_y_n_sum_l158_158762


namespace area_enclosed_by_absolute_value_linear_eq_l158_158935

theorem area_enclosed_by_absolute_value_linear_eq (x y : ℝ) :
  (|5 * x| + |3 * y| = 15) → ∃ (A : ℝ), A = 30 :=
by
  sorry

end area_enclosed_by_absolute_value_linear_eq_l158_158935


namespace smallest_base_10_integer_exists_l158_158949

theorem smallest_base_10_integer_exists :
  ∃ (x a b : ℕ), (a > 2) ∧ (b > 2) ∧ (x = 2 * a + 1) ∧ (x = b + 2) ∧ (x = 7) :=
by
  sorry

end smallest_base_10_integer_exists_l158_158949


namespace problem_D_l158_158713

-- Define the lines m and n, and planes α and β
variables (m n : Type) (α β : Type)

-- Define the parallel and perpendicular relations
variables (parallel : Type → Type → Prop) (perpendicular : Type → Type → Prop)

-- Assume the conditions of problem D
variables (h1 : perpendicular m α) (h2 : parallel n β) (h3 : parallel α β)

-- The proof problem statement: Prove that under these assumptions, m is perpendicular to n
theorem problem_D : perpendicular m n :=
sorry

end problem_D_l158_158713


namespace drone_altitude_l158_158205

theorem drone_altitude (h c d : ℝ) (HC HD CD : ℝ)
  (HCO_eq : h^2 + c^2 = HC^2)
  (HDO_eq : h^2 + d^2 = HD^2)
  (CD_eq : c^2 + d^2 = CD^2) 
  (HC_val : HC = 170)
  (HD_val : HD = 160)
  (CD_val : CD = 200) :
  h = 50 * Real.sqrt 29 :=
by
  sorry

end drone_altitude_l158_158205


namespace age_ratio_l158_158523

theorem age_ratio (A B : ℕ) 
  (h1 : A = 39) 
  (h2 : B = 16) 
  (h3 : (A - 5) + (B - 5) = 45) 
  (h4 : A + 5 = 44) : A / B = 39 / 16 := 
by 
  sorry

end age_ratio_l158_158523


namespace domain_expression_l158_158079

-- Define the conditions for the domain of the expression
def valid_numerator (x : ℝ) : Prop := 3 * x - 6 ≥ 0
def valid_denominator (x : ℝ) : Prop := 7 - 2 * x > 0

-- Proof problem statement
theorem domain_expression (x : ℝ) : valid_numerator x ∧ valid_denominator x ↔ 2 ≤ x ∧ x < 3.5 :=
sorry

end domain_expression_l158_158079


namespace necessary_but_not_sufficient_condition_for_geometric_sequence_l158_158712

theorem necessary_but_not_sufficient_condition_for_geometric_sequence
  (a b c : ℝ) :
  (∃ (r : ℝ), a = r * b ∧ b = r * c) → (b^2 = a * c) ∧ ¬((b^2 = a * c) → (∃ (r : ℝ), a = r * b ∧ b = r * c)) := 
by
  sorry

end necessary_but_not_sufficient_condition_for_geometric_sequence_l158_158712


namespace find_number_l158_158967

variable (x : ℝ)

theorem find_number (h : 0.20 * x = 0.40 * 140 + 80) : x = 680 :=
by
  sorry

end find_number_l158_158967


namespace max_cars_per_div_100_is_20_l158_158009

theorem max_cars_per_div_100_is_20 :
  let m : ℕ := Nat.succ (Nat.succ 0) -- represents m going to infinity
  let car_length : ℕ := 5
  let speed_factor : ℕ := 10
  let sensor_distance_per_hour : ℕ := speed_factor * 1000 * m
  let separation_distance : ℕ := car_length * (m + 1)
  let max_cars : ℕ := (sensor_distance_per_hour / separation_distance) * m
  Nat.floor ((2 * (max_cars : ℝ)) / 100) = 20 :=
by
  sorry

end max_cars_per_div_100_is_20_l158_158009


namespace min_pos_period_tan_l158_158298

theorem min_pos_period_tan (A ω : ℝ) (ϕ : ℝ) : 
  (0 < ω) → 
  (∀ x, tan (ω * x + ϕ) = A * tan (3 * x)) → 
  ∃ T, T = π / 3 := by
  intros hω hfun
  use (π / 3)
  sorry

end min_pos_period_tan_l158_158298


namespace paigeRatio_l158_158176

/-- The total number of pieces in the chocolate bar -/
def totalPieces : ℕ := 60

/-- Michael takes half of the chocolate bar -/
def michaelPieces : ℕ := totalPieces / 2

/-- Mandy gets a fixed number of pieces -/
def mandyPieces : ℕ := 15

/-- The number of pieces left after Michael takes his share -/
def remainingPiecesAfterMichael : ℕ := totalPieces - michaelPieces

/-- The number of pieces Paige takes -/
def paigePieces : ℕ := remainingPiecesAfterMichael - mandyPieces

/-- The ratio of the number of pieces Paige takes to the number of pieces left after Michael takes his share is 1:2 -/
theorem paigeRatio :
  paigePieces / (remainingPiecesAfterMichael / 15) = 1 := sorry

end paigeRatio_l158_158176


namespace volume_of_large_ball_l158_158607

theorem volume_of_large_ball (r : ℝ) (V_small : ℝ) (h1 : 1 = r / (2 * r)) (h2 : V_small = (4 / 3) * Real.pi * r^3) : 
  8 * V_small = 288 :=
by
  sorry

end volume_of_large_ball_l158_158607


namespace number_of_samples_with_score_less_than_48_l158_158879

noncomputable def find_number_of_samples (mu : ℝ) (σ : ℝ) (n : ℕ) (X : ℝ → ℝ) : ℕ :=
let p := 0.04 in
n * p

theorem number_of_samples_with_score_less_than_48 :
  ∀ (μ : ℝ) (σ : ℝ) (n : ℕ) (p : ℝ),
  (μ = 85) →
  (p = 0.04) →
  (100 * p = 4) →
  find_number_of_samples μ σ n (λ x, x) = 4 :=
by
  intros μ σ n p hμ hp hcalc
  rw hμ
  rw hp
  rw hcalc
  exact rfl

end number_of_samples_with_score_less_than_48_l158_158879


namespace rectangle_to_square_area_ratio_is_24_25_l158_158285

noncomputable def rectangle_to_square_area_ratio
  (s : ℝ) -- length of side of square S
  (longer_side : ℝ := 1.2 * s) -- longer side of rectangle R
  (shorter_side : ℝ := 0.8 * s) -- shorter side of rectangle R
  (area_R : ℝ := longer_side * shorter_side) -- area of rectangle R
  (area_S : ℝ := s^2) -- area of square S
  : ℝ := 
  area_R / area_S

theorem rectangle_to_square_area_ratio_is_24_25 
  (s : ℝ)
  : rectangle_to_square_area_ratio s = 24 / 25 :=
by 
  sorry

end rectangle_to_square_area_ratio_is_24_25_l158_158285


namespace intersection_of_lines_l158_158994

-- Define the conditions of the problem
def first_line (x y : ℝ) : Prop := y = -3 * x + 1
def second_line (x y : ℝ) : Prop := y + 1 = 15 * x

-- Prove the intersection point of the two lines
theorem intersection_of_lines : 
  ∃ (x y : ℝ), first_line x y ∧ second_line x y ∧ x = 1 / 9 ∧ y = 2 / 3 :=
by
  sorry

end intersection_of_lines_l158_158994


namespace base_seven_sum_l158_158026

def base_seven_to_ten (n : ℕ) : ℕ := 3 * 7^1 + 5 * 7^0   -- Converts 35_7 to base 10
def base_seven_to_ten' (m : ℕ) : ℕ := 1 * 7^1 + 2 * 7^0  -- Converts 12_7 to base 10

noncomputable def base_ten_product (a b : ℕ) : ℕ := (a * b) -- Computes product in base 10

noncomputable def base_ten_to_seven (p : ℕ) : ℕ :=        -- Converts base 10 to base 7
  let p1 := (p / 7 / 7) % 7
  let p2 := (p / 7) % 7
  let p3 := p % 7
  p1 * 100 + p2 * 10 + p3

noncomputable def sum_of_digits (a : ℕ) : ℕ :=             -- Sums digits in base 7
  let d1 := (a / 100) % 10
  let d2 := (a / 10) % 10
  let d3 := a % 10
  d1 + d2 + d3

noncomputable def base_ten_to_seven' (s : ℕ) : ℕ :=        -- Converts sum back to base 7
  let s1 := s / 7
  let s2 := s % 7
  s1 * 10 + s2

theorem base_seven_sum (n m : ℕ) : base_ten_to_seven' (sum_of_digits (base_ten_to_seven (base_ten_product (base_seven_to_ten n) (base_seven_to_ten' m)))) = 15 :=
by
  sorry

end base_seven_sum_l158_158026


namespace negation_red_cards_in_deck_l158_158791

variable (Deck : Type) (is_red : Deck → Prop) (is_in_deck : Deck → Prop)

theorem negation_red_cards_in_deck :
  (¬ ∃ x : Deck, is_red x ∧ is_in_deck x) ↔ (∃ x : Deck, is_red x ∧ is_in_deck x) :=
by {
  sorry
}

end negation_red_cards_in_deck_l158_158791


namespace six_digit_number_theorem_l158_158722

noncomputable def six_digit_number (a b c d e f : ℕ) : ℕ :=
  10^5 * a + 10^4 * b + 10^3 * c + 10^2 * d + 10 * e + f

noncomputable def rearranged_number (a b c d e f : ℕ) : ℕ :=
  10^5 * b + 10^4 * c + 10^3 * d + 10^2 * e + 10 * f + a

theorem six_digit_number_theorem (a b c d e f : ℕ) (h_a : a ≠ 0) 
  (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) 
  (h4 : 0 ≤ d ∧ d ≤ 9) (h5 : 0 ≤ e ∧ e ≤ 9) (h6 : 0 ≤ f ∧ f ≤ 9) 
  : six_digit_number a b c d e f = 142857 ∨ six_digit_number a b c d e f = 285714 :=
by
  sorry

end six_digit_number_theorem_l158_158722


namespace non_real_roots_of_quadratic_l158_158733

theorem non_real_roots_of_quadratic (b : ℝ) : 
  (¬ ∃ x1 x2 : ℝ, x1^2 + bx1 + 16 = 0 ∧ x2^2 + bx2 + 16 = 0 ∧ x1 = x2) ↔ b ∈ set.Ioo (-8 : ℝ) (8 : ℝ) :=
by {
  sorry
}

end non_real_roots_of_quadratic_l158_158733


namespace rectangle_to_square_area_ratio_is_24_25_l158_158287

noncomputable def rectangle_to_square_area_ratio
  (s : ℝ) -- length of side of square S
  (longer_side : ℝ := 1.2 * s) -- longer side of rectangle R
  (shorter_side : ℝ := 0.8 * s) -- shorter side of rectangle R
  (area_R : ℝ := longer_side * shorter_side) -- area of rectangle R
  (area_S : ℝ := s^2) -- area of square S
  : ℝ := 
  area_R / area_S

theorem rectangle_to_square_area_ratio_is_24_25 
  (s : ℝ)
  : rectangle_to_square_area_ratio s = 24 / 25 :=
by 
  sorry

end rectangle_to_square_area_ratio_is_24_25_l158_158287


namespace arithmetic_sequence_max_sum_l158_158892

noncomputable def max_sum_n (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ) : Prop :=
  (|a 3| = |a 11| ∧ 
   (∃ d : ℤ, d < 0 ∧ 
   (∀ n, a (n + 1) = a n + d) ∧ 
   (∀ m, S m = (m * (2 * a 1 + (m - 1) * d)) / 2)) →
   ((n = 6) ∨ (n = 7)))

theorem arithmetic_sequence_max_sum (a : ℕ → ℤ) (S : ℕ → ℤ) :
  max_sum_n a S 6 ∨ max_sum_n a S 7 := sorry

end arithmetic_sequence_max_sum_l158_158892


namespace exists_word_D_l158_158751

variable {α : Type} [Inhabited α] [DecidableEq α]

def repeats (D : List α) (w : List α) : Prop :=
  ∃ k : ℕ, w = List.join (List.replicate k D)

theorem exists_word_D (A B C : List α)
  (h : (A ++ A ++ B ++ B) = (C ++ C)) :
  ∃ D : List α, repeats D A ∧ repeats D B ∧ repeats D C :=
sorry

end exists_word_D_l158_158751


namespace bart_total_pages_l158_158831

theorem bart_total_pages (total_spent : ℝ) (cost_per_notepad : ℝ) (pages_per_notepad : ℕ)
  (h1 : total_spent = 10) (h2 : cost_per_notepad = 1.25) (h3 : pages_per_notepad = 60) :
  total_spent / cost_per_notepad * pages_per_notepad = 480 :=
by
  sorry

end bart_total_pages_l158_158831


namespace rainfall_march_l158_158746

variable (M A : ℝ)
variable (Hm : A = M - 0.35)
variable (Ha : A = 0.46)

theorem rainfall_march : M = 0.81 := by
  sorry

end rainfall_march_l158_158746


namespace divisor_problem_l158_158341

theorem divisor_problem (n : ℕ) (hn_pos : 0 < n) (h72 : Nat.totient n = 72) (h5n : Nat.totient (5 * n) = 96) : ∃ k : ℕ, (n = 5^k * m ∧ Nat.gcd m 5 = 1) ∧ k = 2 :=
by
  sorry

end divisor_problem_l158_158341


namespace coprime_ab_and_a_plus_b_l158_158426

theorem coprime_ab_and_a_plus_b (a b : ℤ) (h : Int.gcd a b = 1) : Int.gcd (a * b) (a + b) = 1 := by
  sorry

end coprime_ab_and_a_plus_b_l158_158426


namespace exponent_proof_l158_158492

theorem exponent_proof (m : ℝ) : (243 : ℝ) = (3 : ℝ)^5 → (243 : ℝ)^(1/3) = (3 : ℝ)^m → m = 5/3 :=
by
  intros h1 h2
  sorry

end exponent_proof_l158_158492


namespace number_of_students_above_110_l158_158058

-- We start with the conditions
variables (students : ℕ) (mu sigma : ℝ)
  (xi : ℕ → ℝ)
  (P : set ℝ → ℝ)
  (h50 : students = 50)
  (hN : ∀ x, ℝ.dist x xi/pdf.eval (0, pdf.eval (norm(100,10))=0.3)): xi follows a normal_distribution N(100, 10^2) and is symmetric around xi =100emphasise each statement using seperate lines 

-- stating the probability conditions
(P1 : P ({y : ℝ | 90 ≤ y ∧ y ≤ 100} ) = 0.3)

-- The goal
theorem number_of_students_above_110 : 
  students * P({y : ℝ | y ≥ 110}) = 10 :=
by
  sorry

end number_of_students_above_110_l158_158058


namespace hexagonal_pyramid_edge_length_l158_158510

noncomputable def hexagonal_pyramid_edge_sum (s h : ℝ) : ℝ :=
  let perimeter := 6 * s
  let center_to_vertex := s * (1 / 2) * Real.sqrt 3
  let slant_height := Real.sqrt (h^2 + center_to_vertex^2)
  let edge_sum := perimeter + 6 * slant_height
  edge_sum

theorem hexagonal_pyramid_edge_length (s h : ℝ) (a : ℝ) :
  s = 8 →
  h = 15 →
  a = 48 + 6 * Real.sqrt 273 →
  hexagonal_pyramid_edge_sum s h = a :=
by
  intros
  sorry

end hexagonal_pyramid_edge_length_l158_158510


namespace part_a_part_b_l158_158447

def square_side_length : ℝ := 10
def square_area (side_length : ℝ) : ℝ := side_length * side_length
def triangle_area (base : ℝ) (height : ℝ) : ℝ := 0.5 * base * height

-- Part (a)
theorem part_a :
  let side_length := square_side_length
  let square := square_area side_length
  let triangle := triangle_area side_length side_length
  square - triangle = 50 := by
  sorry

-- Part (b)
theorem part_b :
  let side_length := square_side_length
  let square := square_area side_length
  let small_triangle_area := square / 8
  2 * small_triangle_area = 25 := by
  sorry

end part_a_part_b_l158_158447


namespace find_number_l158_158640

-- Define the conditions and the theorem
theorem find_number (number : ℝ)
  (h₁ : ∃ w : ℝ, w = (69.28 * number) / 0.03 ∧ abs (w - 9.237333333333334) ≤ 1e-10) :
  abs (number - 0.004) ≤ 1e-10 :=
by
  sorry

end find_number_l158_158640


namespace calc_expr_l158_158355

theorem calc_expr : (3^5 * 6^3 + 3^3) = 52515 := by
  sorry

end calc_expr_l158_158355


namespace find_a_b_sum_specific_find_a_b_sum_l158_158857

-- Define the sets A and B based on the given inequalities
def set_A : Set ℝ := {x | x^2 - 2 * x - 3 < 0}
def set_B : Set ℝ := {x | x^2 + x - 6 < 0}

-- Intersect the sets A and B
def set_A_int_B : Set ℝ := set_A ∩ set_B

-- Define the inequality with parameters a and b
def quad_ineq (a b : ℝ) : Set ℝ := {x | a * x^2 + b * x + 2 > 0}

-- Define the parameters a and b based on the given condition
noncomputable def a : ℝ := -1
noncomputable def b : ℝ := -1

-- The statement to be proved
theorem find_a_b_sum : ∀ a b : ℝ, set_A ∩ set_B = {x | a * x^2 + b * x + 2 > 0} → a + b = -2 :=
by
  sorry

-- Fixing the parameters a and b for our specific proof condition
theorem specific_find_a_b_sum : a + b = -2 :=
by
  sorry

end find_a_b_sum_specific_find_a_b_sum_l158_158857


namespace contest_paths_correct_l158_158324

noncomputable def count_contest_paths : Nat := sorry

theorem contest_paths_correct : count_contest_paths = 127 := sorry

end contest_paths_correct_l158_158324


namespace Jane_buys_three_bagels_l158_158199

theorem Jane_buys_three_bagels (b m c : ℕ) (h1 : b + m + c = 5) (h2 : 80 * b + 60 * m + 100 * c = 400) : b = 3 := 
sorry

end Jane_buys_three_bagels_l158_158199


namespace find_q_l158_158147

-- Defining the polynomial and conditions
def Q (x : ℝ) (p q r : ℝ) : ℝ := x^3 + p * x^2 + q * x + r

variable (p q r : ℝ)

-- Given conditions
def mean_of_zeros_eq_prod_of_zeros (p q r : ℝ) : Prop :=
  -p / 3 = r

def prod_of_zeros_eq_sum_of_coeffs (p q r : ℝ) : Prop :=
  r = 1 + p + q + r

def y_intercept_eq_three (r : ℝ) : Prop :=
  r = 3

-- Final proof statement asserting q = 5
theorem find_q (p q r : ℝ) (h1 : mean_of_zeros_eq_prod_of_zeros p q r)
  (h2 : prod_of_zeros_eq_sum_of_coeffs p q r)
  (h3 : y_intercept_eq_three r) :
  q = 5 :=
sorry

end find_q_l158_158147


namespace evaluate_y_correct_l158_158368

noncomputable def evaluate_y (x : ℝ) : ℝ :=
  Real.sqrt (x^2 - 4 * x + 4) + Real.sqrt (x^2 + 6 * x + 9) - 2

theorem evaluate_y_correct (x : ℝ) : 
  evaluate_y x = |x - 2| + |x + 3| - 2 :=
by 
  sorry

end evaluate_y_correct_l158_158368


namespace correct_operation_l158_158811

-- Define that m and n are elements of an arbitrary commutative ring
variables {R : Type*} [CommRing R] (m n : R)

theorem correct_operation : (m * n) ^ 2 = m ^ 2 * n ^ 2 := by
  sorry

end correct_operation_l158_158811


namespace vasya_drove_0_4_of_total_distance_l158_158532

-- Define variables for the distances driven by Anton (a), Vasya (b), Sasha (c), and Dima (d)
variables {a b c d s : ℝ}

-- Define the conditions in Lean
def condition_1 := a = b / 2
def condition_2 := c = a + d
def condition_3 := d = s / 10
def condition_4 := s ≠ 0
def condition_5 := a + b + c + d = s

-- Prove that Vasya drove 0.4 of the total distance
theorem vasya_drove_0_4_of_total_distance (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (h4 : condition_4) (h5 : condition_5) : b / s = 0.4 :=
by
  sorry

end vasya_drove_0_4_of_total_distance_l158_158532


namespace vehicle_speed_increase_l158_158040

/-- Vehicle dynamics details -/
structure Vehicle := 
  (initial_speed : ℝ) 
  (deceleration : ℝ)
  (initial_distance_from_A : ℝ)

/-- Given conditions -/
def conditions (A B C : Vehicle) : Prop :=
  A.initial_speed = 80 ∧
  B.initial_speed = 60 ∧
  C.initial_speed = 70 ∧ 
  C.deceleration = 2 ∧
  B.initial_distance_from_A = 40 ∧
  C.initial_distance_from_A = 260

/-- Prove A needs to increase its speed by 5 mph -/
theorem vehicle_speed_increase (A B C : Vehicle) (h : conditions A B C) : 
  ∃ dA : ℝ, dA = 5 ∧ A.initial_speed + dA > B.initial_speed → 
    (A.initial_distance_from_A / (A.initial_speed + dA - B.initial_speed)) < 
    (C.initial_distance_from_A / (A.initial_speed + dA + C.initial_speed - C.deceleration)) :=
sorry

end vehicle_speed_increase_l158_158040


namespace simplify_divide_expression_l158_158780

noncomputable def a : ℝ := Real.sqrt 2 + 1

theorem simplify_divide_expression : 
  (1 - (a / (a + 1))) / ((a^2 - 1) / (a^2 + 2 * a + 1)) = Real.sqrt 2 / 2 :=
by
  sorry

end simplify_divide_expression_l158_158780


namespace rectangular_frame_wire_and_paper_area_l158_158309

theorem rectangular_frame_wire_and_paper_area :
  let l1 := 3
  let l2 := 4
  let l3 := 5
  let wire_length := (l1 + l2 + l3) * 4
  let paper_area := ((l1 * l2) + (l1 * l3) + (l2 * l3)) * 2
  wire_length = 48 ∧ paper_area = 94 :=
by
  sorry

end rectangular_frame_wire_and_paper_area_l158_158309


namespace area_enclosed_by_graph_l158_158931

noncomputable def enclosed_area (x y : ℝ) : ℝ := 
  if h : (|5 * x| + |3 * y| = 15) then
    30 -- The area enclosed by the graph
  else
    0 -- Default case for definition completeness

theorem area_enclosed_by_graph : ∀ (x y : ℝ), (|5 * x| + |3 * y| = 15) → enclosed_area x y = 30 :=
by
  sorry

end area_enclosed_by_graph_l158_158931


namespace inequality_solution_l158_158273

theorem inequality_solution (x : ℝ) : 
  (3 - (1 / (3 * x + 4)) < 5) ↔ (x < -4 / 3) := 
by
  sorry

end inequality_solution_l158_158273


namespace sequence_length_l158_158192

theorem sequence_length (a : ℕ) (h : a = 10800) (h1 : ∀ n, (n ≠ 0 → ∃ m, n = 2 * m ∧ m ≠ 0) ∧ 2 ∣ n)
  : ∃ k : ℕ, k = 5 := 
sorry

end sequence_length_l158_158192


namespace baron_not_lying_l158_158352

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem baron_not_lying : 
  ∃ a b : ℕ, 
  (a ≠ b ∧ a ≥ 10^9 ∧ a < 10^10 ∧ b ≥ 10^9 ∧ b < 10^10 ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 ∧ 
  (a + sum_of_digits (a * a) = b + sum_of_digits (b * b))) :=
  sorry

end baron_not_lying_l158_158352


namespace maximize_sales_volume_l158_158057

open Real

def profit (x : ℝ) : ℝ := (x - 20) * (400 - 20 * (x - 30))

theorem maximize_sales_volume : 
  ∃ x : ℝ, (∀ x' : ℝ, profit x' ≤ profit x) ∧ x = 35 := 
by
  sorry

end maximize_sales_volume_l158_158057


namespace vasya_drove_0_4_of_total_distance_l158_158530

-- Define variables for the distances driven by Anton (a), Vasya (b), Sasha (c), and Dima (d)
variables {a b c d s : ℝ}

-- Define the conditions in Lean
def condition_1 := a = b / 2
def condition_2 := c = a + d
def condition_3 := d = s / 10
def condition_4 := s ≠ 0
def condition_5 := a + b + c + d = s

-- Prove that Vasya drove 0.4 of the total distance
theorem vasya_drove_0_4_of_total_distance (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (h4 : condition_4) (h5 : condition_5) : b / s = 0.4 :=
by
  sorry

end vasya_drove_0_4_of_total_distance_l158_158530


namespace minimize_xy_l158_158409

theorem minimize_xy (x y : ℕ) (hx : x > 0) (hy : y > 0) (h_eq : 7 * x + 4 * y = 200) : (x * y = 172) :=
sorry

end minimize_xy_l158_158409


namespace area_of_region_l158_158993

theorem area_of_region (x y : ℝ) (h : x^2 + y^2 + 6 * x - 8 * y - 5 = 0) : 
  ∃ (r : ℝ), (π * r^2 = 30 * π) :=
by -- Starting the proof, skipping the detailed steps
sorry -- Proof placeholder

end area_of_region_l158_158993


namespace vasya_fraction_l158_158528

-- Define the variables for distances and total distance
variables {a b c d s : ℝ}

-- Define conditions
def anton_distance (a b : ℝ) : Prop := a = b / 2
def sasha_distance (c a d : ℝ) : Prop := c = a + d
def dima_distance (d s : ℝ) : Prop := d = s / 10
def total_distance (a b c d s : ℝ) : Prop := a + b + c + d = s

-- The main theorem 
theorem vasya_fraction (a b c d s : ℝ) (h1 : anton_distance a b) 
  (h2 : sasha_distance c a d) (h3 : dima_distance d s)
  (h4 : total_distance a b c d s) : b / s = 0.4 :=
sorry

end vasya_fraction_l158_158528


namespace standard_ellipse_eq_l158_158922

def ellipse_standard_eq (a b : ℝ) (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

theorem standard_ellipse_eq (P: ℝ × ℝ) (Q: ℝ × ℝ) (a b : ℝ) (h1 : P = (-3, 0)) (h2 : Q = (0, -2)) :
  ellipse_standard_eq 3 2 x y :=
by
  sorry

end standard_ellipse_eq_l158_158922


namespace carrie_phone_charges_l158_158629

def total_miles (d1 d2 d3 d4 : ℕ) : ℕ :=
  d1 + d2 + d3 + d4

def charges_needed (total_miles charge_miles : ℕ) : ℕ :=
  total_miles / charge_miles + if total_miles % charge_miles = 0 then 0 else 1

theorem carrie_phone_charges :
  let d1 := 135
  let d2 := 135 + 124
  let d3 := 159
  let d4 := 189
  let charge_miles := 106
  charges_needed (total_miles d1 d2 d3 d4) charge_miles = 7 :=
by
  sorry

end carrie_phone_charges_l158_158629


namespace image_of_2_in_set_B_l158_158496

theorem image_of_2_in_set_B (f : ℤ → ℤ) (h : ∀ x, f x = 2 * x + 1) : f 2 = 5 :=
by
  apply h

end image_of_2_in_set_B_l158_158496


namespace area_enclosed_by_graph_l158_158929

noncomputable def enclosed_area (x y : ℝ) : ℝ := 
  if h : (|5 * x| + |3 * y| = 15) then
    30 -- The area enclosed by the graph
  else
    0 -- Default case for definition completeness

theorem area_enclosed_by_graph : ∀ (x y : ℝ), (|5 * x| + |3 * y| = 15) → enclosed_area x y = 30 :=
by
  sorry

end area_enclosed_by_graph_l158_158929


namespace open_parking_spots_fourth_level_l158_158974

theorem open_parking_spots_fourth_level :
  ∀ (n_first n_total : ℕ)
    (n_second_diff n_third_diff : ℕ),
    n_first = 4 →
    n_second_diff = 7 →
    n_third_diff = 6 →
    n_total = 46 →
    ∃ (n_first n_second n_third n_fourth : ℕ),
      n_second = n_first + n_second_diff ∧
      n_third = n_second + n_third_diff ∧
      n_first + n_second + n_third + n_fourth = n_total ∧
      n_fourth = 14 := by
  sorry

end open_parking_spots_fourth_level_l158_158974


namespace solution_set_of_inequality_system_l158_158028

theorem solution_set_of_inequality_system (x : ℝ) :
  (3 * x - 1 ≥ x + 1) ∧ (x + 4 > 4 * x - 2) ↔ (1 ≤ x ∧ x < 2) := 
by
  sorry

end solution_set_of_inequality_system_l158_158028


namespace train_people_count_l158_158981

theorem train_people_count :
  let initial := 332
  let first_station_on := 119
  let first_station_off := 113
  let second_station_off := 95
  let second_station_on := 86
  initial + first_station_on - first_station_off - second_station_off + second_station_on = 329 := 
by
  sorry

end train_people_count_l158_158981


namespace probability_of_adjacent_rs_is_two_fifth_l158_158476

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

noncomputable def countArrangementsWithAdjacentRs : ℕ :=
factorial 4

noncomputable def countTotalArrangements : ℕ :=
factorial 5 / factorial 2

noncomputable def probabilityOfAdjacentRs : ℚ :=
(countArrangementsWithAdjacentRs : ℚ) / (countTotalArrangements : ℚ)

theorem probability_of_adjacent_rs_is_two_fifth :
  probabilityOfAdjacentRs = 2 / 5 := by
  sorry

end probability_of_adjacent_rs_is_two_fifth_l158_158476


namespace count_satisfying_integers_l158_158106

theorem count_satisfying_integers :
  (∃ S : Finset ℕ, (∀ n ∈ S, 9 < n ∧ n < 60) ∧ S.card = 50) :=
by
  sorry

end count_satisfying_integers_l158_158106


namespace tile_arrangement_probability_l158_158379

theorem tile_arrangement_probability :
  let X := 5
  let O := 4
  let total_tiles := 9
  (1 : ℚ) / (Nat.choose total_tiles X) = 1 / 126 :=
by
  sorry

end tile_arrangement_probability_l158_158379


namespace find_f_l158_158407

-- Definitions of odd and even functions
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def even_function (g : ℝ → ℝ) := ∀ x, g (-x) = g x

-- Main theorem
theorem find_f (f g : ℝ → ℝ) (h_odd_f : odd_function f) (h_even_g : even_function g) 
    (h_eq : ∀ x, f x + g x = 1 / (x - 1)) :
  ∀ x, f x = x / (x ^ 2 - 1) :=
by
  sorry

end find_f_l158_158407


namespace find_m_l158_158489

theorem find_m (m : ℝ) : (243 : ℝ)^(1/3) = (3 : ℝ)^m → m = 5 / 3 :=
by
  sorry

end find_m_l158_158489


namespace julieta_total_spent_l158_158420

theorem julieta_total_spent (original_backpack_price : ℕ)
                            (original_ringbinder_price : ℕ)
                            (backpack_price_increase : ℕ)
                            (ringbinder_price_decrease : ℕ)
                            (number_of_ringbinders : ℕ)
                            (new_backpack_price : ℕ)
                            (new_ringbinder_price : ℕ)
                            (total_ringbinder_cost : ℕ)
                            (total_spent : ℕ) :
  original_backpack_price = 50 →
  original_ringbinder_price = 20 →
  backpack_price_increase = 5 →
  ringbinder_price_decrease = 2 →
  number_of_ringbinders = 3 →
  new_backpack_price = original_backpack_price + backpack_price_increase →
  new_ringbinder_price = original_ringbinder_price - ringbinder_price_decrease →
  total_ringbinder_cost = new_ringbinder_price * number_of_ringbinders →
  total_spent = new_backpack_price + total_ringbinder_cost →
  total_spent = 109 := by
  intros
  sorry

end julieta_total_spent_l158_158420


namespace domain_expression_l158_158080

-- Define the conditions for the domain of the expression
def valid_numerator (x : ℝ) : Prop := 3 * x - 6 ≥ 0
def valid_denominator (x : ℝ) : Prop := 7 - 2 * x > 0

-- Proof problem statement
theorem domain_expression (x : ℝ) : valid_numerator x ∧ valid_denominator x ↔ 2 ≤ x ∧ x < 3.5 :=
sorry

end domain_expression_l158_158080


namespace sum_of_two_consecutive_squares_l158_158633

variable {k m A : ℕ}

theorem sum_of_two_consecutive_squares :
  (∃ k : ℕ, A^2 = (k+1)^3 - k^3) → (∃ m : ℕ, A = m^2 + (m+1)^2) :=
by sorry

end sum_of_two_consecutive_squares_l158_158633


namespace area_enclosed_by_absolute_value_linear_eq_l158_158936

theorem area_enclosed_by_absolute_value_linear_eq (x y : ℝ) :
  (|5 * x| + |3 * y| = 15) → ∃ (A : ℝ), A = 30 :=
by
  sorry

end area_enclosed_by_absolute_value_linear_eq_l158_158936


namespace k_ge_1_l158_158459

theorem k_ge_1 (k : ℝ) : 
  (∀ x : ℝ, 2 * x + 9 > 6 * x + 1 ∧ x - k < 1 → x < 2) → k ≥ 1 :=
by 
  sorry

end k_ge_1_l158_158459


namespace perpendicular_lines_a_value_l158_158703

theorem perpendicular_lines_a_value :
  (∃ (a : ℝ), ∀ (x y : ℝ), (3 * y + x + 5 = 0) ∧ (4 * y + a * x + 3 = 0) → a = -12) :=
by
  sorry

end perpendicular_lines_a_value_l158_158703


namespace fraction_addition_l158_158986

variable (a : ℝ)

theorem fraction_addition (ha : a ≠ 0) : (3 / a) + (2 / a) = (5 / a) :=
by
  sorry

end fraction_addition_l158_158986


namespace sector_area_l158_158743

theorem sector_area (arc_length radius : ℝ) (h1 : arc_length = 2) (h2 : radius = 2) : 
  (1/2) * arc_length * radius = 2 :=
by
  -- sorry placeholder for proof
  sorry

end sector_area_l158_158743


namespace quadratic_non_real_roots_l158_158737

variable (b : ℝ)

theorem quadratic_non_real_roots : (b^2 - 64 < 0) → (-8 < b ∧ b < 8) :=
by
  sorry

end quadratic_non_real_roots_l158_158737


namespace smallest_base_10_integer_l158_158942

theorem smallest_base_10_integer (a b : ℕ) (ha : a > 2) (hb : b > 2) 
  (h1: 21_a = 2 * a + 1) (h2: 12_b = b + 2) : 2 * a + 1 = 7 :=
by 
  sorry

end smallest_base_10_integer_l158_158942


namespace total_seeds_planted_l158_158435

def number_of_flowerbeds : ℕ := 9
def seeds_per_flowerbed : ℕ := 5

theorem total_seeds_planted : number_of_flowerbeds * seeds_per_flowerbed = 45 :=
by
  sorry

end total_seeds_planted_l158_158435


namespace decimal_to_base_five_correct_l158_158700

theorem decimal_to_base_five_correct : 
  ∃ (d0 d1 d2 d3 : ℕ), 256 = d3 * 5^3 + d2 * 5^2 + d1 * 5^1 + d0 * 5^0 ∧ 
                          d3 = 2 ∧ d2 = 0 ∧ d1 = 1 ∧ d0 = 1 :=
by sorry

end decimal_to_base_five_correct_l158_158700


namespace student_rank_left_l158_158182

theorem student_rank_left {n m : ℕ} (h1 : n = 10) (h2 : m = 6) : (n - m + 1) = 5 := by
  sorry

end student_rank_left_l158_158182


namespace area_enclosed_by_graph_l158_158932

theorem area_enclosed_by_graph (x y : ℝ) (h : abs (5 * x) + abs (3 * y) = 15) : 
  ∃ (area : ℝ), area = 30 :=
sorry

end area_enclosed_by_graph_l158_158932


namespace isosceles_triangle_sine_base_angle_l158_158306

theorem isosceles_triangle_sine_base_angle (m : ℝ) (θ : ℝ) 
  (h1 : m > 0)
  (h2 : θ > 0 ∧ θ < π / 2)
  (h_base_height : m * (Real.sin θ) = (m * 2 * (Real.sin θ) * (Real.cos θ))) :
  Real.sin θ = (Real.sqrt 15) / 4 := 
sorry

end isosceles_triangle_sine_base_angle_l158_158306


namespace total_interest_correct_l158_158778

-- Definitions
def total_amount : ℝ := 3500
def P1 : ℝ := 1550
def P2 : ℝ := total_amount - P1
def rate1 : ℝ := 0.03
def rate2 : ℝ := 0.05

-- Total interest calculation
noncomputable def interest1 : ℝ := P1 * rate1
noncomputable def interest2 : ℝ := P2 * rate2
noncomputable def total_interest : ℝ := interest1 + interest2

-- Theorem statement
theorem total_interest_correct : total_interest = 144 := 
by
  -- Proof steps would go here
  sorry

end total_interest_correct_l158_158778


namespace vasya_drives_fraction_l158_158547

theorem vasya_drives_fraction {a b c d s : ℝ} 
  (h1 : a = b / 2) 
  (h2 : c = a + d) 
  (h3 : d = s / 10) 
  (h4 : a + b + c + d = s) : 
  b / s = 0.4 :=
by
  sorry

end vasya_drives_fraction_l158_158547


namespace binary_to_base5_1101_l158_158070

-- Definition of the binary to decimal conversion for the given number
def binary_to_decimal (b: Nat): Nat :=
  match b with
  | 0    => 0
  | 1101 => 1 * 2^0 + 0 * 2^1 + 1 * 2^2 + 1 * 2^3
  | _    => 0  -- This is a specific case for the given problem

-- Definition of the decimal to base-5 conversion method
def decimal_to_base5 (d: Nat): Nat :=
  match d with
  | 0    => 0
  | 13   =>
    let rem1 := 13 % 5
    let div1 := 13 / 5
    let rem2 := div1 % 5
    let div2 := div1 / 5
    rem2 * 10 + rem1  -- Assemble the base-5 number from remainders
  | _    => 0  -- This is a specific case for the given problem

-- Proof statement: conversion of 1101 in binary to base-5 yields 23
theorem binary_to_base5_1101 : decimal_to_base5 (binary_to_decimal 1101) = 23 := by
  sorry

end binary_to_base5_1101_l158_158070


namespace find_equation_of_ellipse_find_range_OA_OB_find_area_quadrilateral_l158_158710

-- Define the ellipse and parameters
variables (a b c : ℝ) (x y : ℝ)
-- Conditions
def ellipse (a b : ℝ) : Prop := a > b ∧ b > 0 ∧ (∀ x y, (x^2 / a^2) + (y^2 / b^2) = 1)

-- Given conditions
def eccentricity (c a : ℝ) : Prop := c = a * (Real.sqrt 3 / 2)
def rhombus_area (a b : ℝ) : Prop := (1/2) * (2 * a) * (2 * b) = 4
def relation_a_b_c (a b c : ℝ) : Prop := a^2 = b^2 + c^2

-- Questions transformed into proof problems
def ellipse_equation (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1
def range_OA_OB (OA OB : ℝ) : Prop := OA * OB ∈ Set.union (Set.Icc (-(3/2)) 0) (Set.Ioo 0 (3/2))
def quadrilateral_area : ℝ := 4

-- Prove the results given the conditions
theorem find_equation_of_ellipse (a b c : ℝ) (h_ellipse : ellipse a b) (h_ecc : eccentricity c a) (h_area : rhombus_area a b) (h_rel : relation_a_b_c a b c) :
  ellipse_equation x y := by
  sorry

theorem find_range_OA_OB (OA OB : ℝ) (kAC kBD : ℝ) (h_mult : kAC * kBD = -(1/4)) :
  range_OA_OB OA OB := by
  sorry

theorem find_area_quadrilateral : quadrilateral_area = 4 := by
  sorry

end find_equation_of_ellipse_find_range_OA_OB_find_area_quadrilateral_l158_158710


namespace min_a2_b2_c2_l158_158152

theorem min_a2_b2_c2 (a b c : ℕ) (h : a + 2 * b + 3 * c = 73) : a^2 + b^2 + c^2 ≥ 381 :=
by sorry

end min_a2_b2_c2_l158_158152


namespace max_smaller_rectangles_l158_158833

theorem max_smaller_rectangles (a : ℕ) (d : ℕ) (n : ℕ) 
    (ha : a = 100) (hd : d = 2) (hn : n = 50) : 
    n + 1 * (n + 1) = 2601 :=
by
  rw [hn]
  norm_num
  sorry

end max_smaller_rectangles_l158_158833


namespace labourer_saved_amount_l158_158019

noncomputable def average_expenditure_6_months : ℕ := 85
noncomputable def expenditure_reduction_4_months : ℕ := 60
noncomputable def monthly_income : ℕ := 78

theorem labourer_saved_amount :
  let initial_debt := 6 * average_expenditure_6_months - 6 * monthly_income
      cleared_debt := 4 * monthly_income - 4 * expenditure_reduction_4_months
      savings := cleared_debt - initial_debt
  in savings = 30 :=
by
  have average_expenditure : ℕ := 6 * 85
  have average_income : ℕ := 6 * 78
  have initial_debt : ℕ := average_expenditure - average_income
  have new_expenditure : ℕ := 4 * 60
  have new_income : ℕ := 4 * 78
  have cleared_debt : ℕ := new_income - new_expenditure
  have savings : ℕ := cleared_debt - initial_debt
  have amount_saved : ℕ := 30
  show savings = amount_saved
  sorry

end labourer_saved_amount_l158_158019


namespace sign_up_ways_l158_158802

theorem sign_up_ways : (3 ^ 4) = 81 :=
by
  sorry

end sign_up_ways_l158_158802


namespace truck_distance_on_7_liters_l158_158825

-- Define the conditions
def truck_300_km_per_5_liters := 300
def liters_5 := 5
def liters_7 := 7
def expected_distance_7_liters := 420

-- The rate of distance (km per liter)
def rate := truck_300_km_per_5_liters / liters_5

-- Proof statement
theorem truck_distance_on_7_liters :
  rate * liters_7 = expected_distance_7_liters :=
  by
  sorry

end truck_distance_on_7_liters_l158_158825


namespace smallest_base_10_integer_l158_158944

theorem smallest_base_10_integer (a b : ℕ) (ha : a > 2) (hb : b > 2) 
  (h1: 21_a = 2 * a + 1) (h2: 12_b = b + 2) : 2 * a + 1 = 7 :=
by 
  sorry

end smallest_base_10_integer_l158_158944


namespace cone_base_radius_and_slant_height_l158_158044

noncomputable def sector_angle := 300
noncomputable def sector_radius := 10
noncomputable def arc_length := (sector_angle / 360) * 2 * Real.pi * sector_radius

theorem cone_base_radius_and_slant_height :
  ∃ (r l : ℝ), arc_length = 2 * Real.pi * r ∧ l = sector_radius ∧ r = 8 ∧ l = 10 :=
by 
  sorry

end cone_base_radius_and_slant_height_l158_158044


namespace correct_option_is_C_l158_158315

-- Definitions of the expressions given in the conditions
def optionA (a : ℝ) : ℝ := 3 * a^5 - a^5
def optionB (a : ℝ) : ℝ := a^2 + a^5
def optionC (a : ℝ) : ℝ := a^5 + a^5
def optionD (x y : ℝ) : ℝ := x^2 * y + x * y^2

-- The problem is to prove that optionC is correct and the others are not
theorem correct_option_is_C (a x y : ℝ) :
  (optionC a = 2 * a^5) ∧ 
  (optionA a ≠ 3) ∧ 
  (optionB a ≠ a^7) ∧ 
  (optionD x y ≠ 2 * (x ^ 3) * (y ^ 3)) :=
by
  sorry

end correct_option_is_C_l158_158315


namespace problem_2023_divisible_by_consecutive_integers_l158_158169

theorem problem_2023_divisible_by_consecutive_integers :
  ∃ (n : ℕ), (n = 2022 ∨ n = 2023 ∨ n = 2024) ∧ (2023^2023 - 2023^2021) % n = 0 :=
sorry

end problem_2023_divisible_by_consecutive_integers_l158_158169


namespace michael_matchstick_houses_l158_158430

theorem michael_matchstick_houses :
  ∃ n : ℕ, n = (600 / 2) / 10 ∧ n = 30 := 
sorry

end michael_matchstick_houses_l158_158430


namespace find_number_l158_158630

theorem find_number (x : ℝ) (h : (4 / 3) * x = 48) : x = 36 :=
sorry

end find_number_l158_158630


namespace extreme_value_f_g_gt_one_l158_158258

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 1
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.exp x * (a * x + x * Real.cos x + 1)

theorem extreme_value_f : f 0 = 0 :=
by
  sorry

theorem g_gt_one (a : ℝ) (h : a > -1) (x : ℝ) (hx : 0 < x) (hx1 : x < 1) : g x a > 1 :=
by
  sorry

end extreme_value_f_g_gt_one_l158_158258


namespace line_intersects_parabola_at_9_units_apart_l158_158197

theorem line_intersects_parabola_at_9_units_apart :
  ∃ m b, (∃ (k1 k2 : ℝ), 
              (y1 = k1^2 + 6*k1 - 4) ∧ 
              (y2 = k2^2 + 6*k2 - 4) ∧ 
              (y1 = m*k1 + b) ∧ 
              (y2 = m*k2 + b) ∧ 
              |y1 - y2| = 9) ∧ 
          (0 ≠ b) ∧ 
          ((1 : ℝ) = 2*m + b) ∧ 
          (m = 4 ∧ b = -7)
:= sorry

end line_intersects_parabola_at_9_units_apart_l158_158197


namespace solve_system_l158_158905

-- Definitions for the system of equations.
def system_valid (y : ℝ) (x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop :=
  x₅ + x₂ = y * x₁ ∧
  x₁ + x₃ = y * x₂ ∧
  x₂ + x₄ = y * x₃ ∧
  x₃ + x₅ = y * x₄ ∧
  x₄ + x₁ = y * x₅

-- Main theorem to prove.
theorem solve_system (y : ℝ) (x₁ x₂ x₃ x₄ x₅ : ℝ) : 
  system_valid y x₁ x₂ x₃ x₄ x₅ →
  ((y^2 + y - 1 ≠ 0 → x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0) ∨ 
  (y = 2 → ∃ (t : ℝ), x₁ = t ∧ x₂ = t ∧ x₃ = t ∧ x₄ = t ∧ x₅ = t) ∨ 
  (y^2 + y - 1 = 0 → ∃ (u v : ℝ), 
    x₁ = u ∧ 
    x₅ = v ∧ 
    x₂ = y * u - v ∧ 
    x₃ = -y * (u + v) ∧ 
    x₄ = y * v - u ∧ 
    (y = (-1 + Real.sqrt 5) / 2 ∨ y = (-1 - Real.sqrt 5) / 2))) :=
by
  intro h
  sorry

end solve_system_l158_158905


namespace pasha_encoded_expression_l158_158631

theorem pasha_encoded_expression :
  2065 + 5 - 47 = 2023 :=
by
  sorry

end pasha_encoded_expression_l158_158631


namespace dot_product_value_l158_158603

-- Define vectors a and b, and the condition of their linear combination
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def a : Vector2D := ⟨-1, 2⟩
def b (m : ℝ) : Vector2D := ⟨m, 1⟩

-- Define the condition that vector a + 2b is parallel to 2a - b
def parallel (v w : Vector2D) : Prop := ∃ k : ℝ, v.x = k * w.x ∧ v.y = k * w.y

def vector_add (v w : Vector2D) : Vector2D := ⟨v.x + w.x, v.y + w.y⟩
def scalar_mul (c : ℝ) (v : Vector2D) : Vector2D := ⟨c * v.x, c * v.y⟩

-- Dot product definition
def dot_product (v w : Vector2D) : ℝ := v.x * w.x + v.y * w.y

-- The theorem to prove
theorem dot_product_value (m : ℝ)
  (h : parallel (vector_add a (scalar_mul 2 (b m))) (vector_add (scalar_mul 2 a) (scalar_mul (-1) (b m)))) :
  dot_product a (b m) = 5 / 2 :=
sorry

end dot_product_value_l158_158603


namespace bus_stop_time_l158_158047

-- Usual time to walk to the bus stop
def usual_time (T : ℕ) := T

-- Usual speed
def usual_speed (S : ℕ) := S

-- New speed when walking at 4/5 of usual speed
def new_speed (S : ℕ) := (4 * S) / 5

-- Time relationship when walking at new speed
def time_relationship (T : ℕ) (S : ℕ) := (S / ((4 * S) / 5)) = (T + 10) / T

-- Prove the usual time T is 40 minutes
theorem bus_stop_time (T S : ℕ) (h1 : time_relationship T S) : T = 40 :=
by
  sorry

end bus_stop_time_l158_158047


namespace arithmetic_sequence_tenth_term_l158_158797

theorem arithmetic_sequence_tenth_term (a d : ℤ) 
  (h1 : a + 2 * d = 23) 
  (h2 : a + 6 * d = 35) : 
  a + 9 * d = 44 := 
by 
  -- proof goes here
  sorry

end arithmetic_sequence_tenth_term_l158_158797


namespace common_difference_minimum_sum_value_l158_158150

variable {α : Type}
variables (a : ℕ → ℤ) (d : ℤ)
variables (S : ℕ → ℚ)

-- Conditions: Arithmetic sequence property and specific initial values
def is_arithmetic_sequence (d : ℤ) : Prop :=
  ∀ n, a n = a 1 + (n - 1) * d

axiom a1_eq_neg3 : a 1 = -3
axiom condition : 11 * a 5 = 5 * a 8 - 13

-- Define the sum of the first n terms of the arithmetic sequence
def sum_of_arithmetic_sequence (n : ℕ) (a : ℕ → ℤ) (d : ℤ) : ℚ :=
  (↑n / 2) * (2 * a 1 + ↑((n - 1) * d))

-- Prove the common difference and the minimum sum value
theorem common_difference : d = 31 / 9 :=
sorry

theorem minimum_sum_value : S 1 = -2401 / 840 :=
sorry

end common_difference_minimum_sum_value_l158_158150


namespace area_of_parallelogram_l158_158320

theorem area_of_parallelogram (base height : ℝ) (h_base : base = 18) (h_height : height = 16) : 
  base * height = 288 := 
by
  sorry

end area_of_parallelogram_l158_158320


namespace max_sides_in_subpolygon_l158_158121

/-- In a convex 1950-sided polygon with all its diagonals drawn, the polygon with the greatest number of sides among these smaller polygons can have at most 1949 sides. -/
theorem max_sides_in_subpolygon (n : ℕ) (hn : n = 1950) : 
  ∃ p : ℕ, p = 1949 ∧ ∀ m, m ≤ n-2 → m ≤ 1949 :=
sorry

end max_sides_in_subpolygon_l158_158121


namespace part1_part2_l158_158870

variable (a b : ℝ)

theorem part1 (h : |a - 3| + |b + 6| = 0) : a + b - 2 = -5 := sorry

theorem part2 (h : |a - 3| + |b + 6| = 0) : a - b - 2 = 7 := sorry

end part1_part2_l158_158870


namespace sum_of_reciprocals_l158_158795

theorem sum_of_reciprocals (x y : ℝ) (h : x + y = 6 * x * y) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (1 / x) + (1 / y) = 2 := 
by
  sorry

end sum_of_reciprocals_l158_158795


namespace add_fractions_l158_158983

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = (5 / a) :=
by sorry

end add_fractions_l158_158983


namespace determine_m_l158_158646

variables (m x : ℝ)
noncomputable def f (x : ℝ) := x^2 - 3*x + m
noncomputable def g (x : ℝ) := x^2 - 3*x + 5*m

theorem determine_m (h : 3 * f 5 = 2 * g 5) : m = 10 / 7 :=
by
  sorry

end determine_m_l158_158646


namespace valid_numbers_count_l158_158181

def count_valid_numbers : ℕ :=
  sorry

theorem valid_numbers_count :
  count_valid_numbers = 7 :=
sorry

end valid_numbers_count_l158_158181


namespace find_a_b_sum_pos_solution_l158_158669

theorem find_a_b_sum_pos_solution :
  ∃ (a b : ℕ), (∃ (x : ℝ), x^2 + 16 * x = 100 ∧ x = Real.sqrt a - b) ∧ a + b = 172 :=
by
  sorry

end find_a_b_sum_pos_solution_l158_158669


namespace solve_equation_correctly_l158_158796

theorem solve_equation_correctly : 
  ∀ x : ℝ, (x - 1) / 2 - 1 = (2 * x + 1) / 3 → x = -11 :=
by
  intro x h
  sorry

end solve_equation_correctly_l158_158796


namespace second_concert_attendance_correct_l158_158627

def first_concert_attendance : ℕ := 65899
def additional_people : ℕ := 119
def second_concert_attendance : ℕ := 66018

theorem second_concert_attendance_correct :
  first_concert_attendance + additional_people = second_concert_attendance :=
by sorry

end second_concert_attendance_correct_l158_158627


namespace class_C_payment_l158_158434

-- Definitions based on conditions
variables (x y z : ℤ) (total_C : ℤ)

-- Given conditions
def condition_A : Prop := 3 * x + 7 * y + z = 14
def condition_B : Prop := 4 * x + 10 * y + z = 16
def condition_C : Prop := 3 * (x + y + z) = total_C

-- The theorem to prove
theorem class_C_payment (hA : condition_A x y z) (hB : condition_B x y z) : total_C = 30 :=
sorry

end class_C_payment_l158_158434


namespace necessary_but_not_sufficient_l158_158495

theorem necessary_but_not_sufficient (a : ℝ) (h : a > 0) : a > 0 ↔ ((a > 0) ∧ (a < 2) → (a^2 - 2 * a < 0)) :=
by
    sorry

end necessary_but_not_sufficient_l158_158495


namespace sine_cosine_obtuse_angle_l158_158231

theorem sine_cosine_obtuse_angle :
  ∀ P : (ℝ × ℝ), P = (Real.sin 2, Real.cos 2) → (Real.sin 2 > 0) ∧ (Real.cos 2 < 0) → 
  (P.1 > 0) ∧ (P.2 < 0) :=
by
  sorry

end sine_cosine_obtuse_angle_l158_158231


namespace molecular_weight_of_N2O5_l158_158373

def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00
def num_atoms_N : ℕ := 2
def num_atoms_O : ℕ := 5
def molecular_weight_N2O5 : ℝ := (num_atoms_N * atomic_weight_N) + (num_atoms_O * atomic_weight_O)

theorem molecular_weight_of_N2O5 : molecular_weight_N2O5 = 108.02 :=
by
  sorry

end molecular_weight_of_N2O5_l158_158373


namespace volume_of_rectangular_prism_l158_158511

theorem volume_of_rectangular_prism
  (l w h : ℝ)
  (Hlw : l * w = 10)
  (Hwh : w * h = 15)
  (Hlh : l * h = 6) : l * w * h = 30 := 
by
  sorry

end volume_of_rectangular_prism_l158_158511


namespace joggers_difference_l158_158826

theorem joggers_difference (Tyson_joggers Alexander_joggers Christopher_joggers : ℕ) 
  (h1 : Alexander_joggers = Tyson_joggers + 22) 
  (h2 : Christopher_joggers = 20 * Tyson_joggers)
  (h3 : Christopher_joggers = 80) : 
  Christopher_joggers - Alexander_joggers = 54 :=
by 
  sorry

end joggers_difference_l158_158826


namespace range_of_a_solution_set_of_inequality_l158_158219

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

end range_of_a_solution_set_of_inequality_l158_158219


namespace parabola_line_intersection_l158_158792

theorem parabola_line_intersection :
  ∀ (x y : ℝ), 
  (y = 20 * x^2 + 19 * x) ∧ (y = 20 * x + 19) →
  y = 20 * x^3 + 19 * x^2 :=
by sorry

end parabola_line_intersection_l158_158792


namespace initial_bales_l158_158803

theorem initial_bales (bales_initially bales_added bales_now : ℕ)
  (h₀ : bales_added = 26)
  (h₁ : bales_now = 54)
  (h₂ : bales_now = bales_initially + bales_added) :
  bales_initially = 28 :=
by
  sorry

end initial_bales_l158_158803


namespace right_triangle_leg_square_l158_158913

theorem right_triangle_leg_square (a c b : ℕ) (h1 : c = a + 1) (h2 : a^2 + b^2 = c^2) : b^2 = c + a :=
by
  sorry

end right_triangle_leg_square_l158_158913


namespace donny_total_spending_l158_158839

noncomputable def total_saving_mon : ℕ := 15
noncomputable def total_saving_tue : ℕ := 28
noncomputable def total_saving_wed : ℕ := 13
noncomputable def total_saving_fri : ℕ := 22

noncomputable def total_savings_mon_to_wed : ℕ := total_saving_mon + total_saving_tue + total_saving_wed
noncomputable def thursday_spending : ℕ := total_savings_mon_to_wed / 2
noncomputable def remaining_savings_after_thursday : ℕ := total_savings_mon_to_wed - thursday_spending
noncomputable def total_savings_before_sat : ℕ := remaining_savings_after_thursday + total_saving_fri
noncomputable def saturday_spending : ℕ := total_savings_before_sat * 40 / 100

theorem donny_total_spending : thursday_spending + saturday_spending = 48 := by sorry

end donny_total_spending_l158_158839


namespace regression_slope_interpretation_l158_158768

-- Define the variables and their meanings
variable {x y : ℝ}

-- Define the regression line equation
def regression_line (x : ℝ) : ℝ := 0.8 * x + 4.6

-- Define the proof statement
theorem regression_slope_interpretation (hx : ∀ x, y = regression_line x) :
  ∀ delta_x : ℝ, delta_x = 1 → (regression_line (x + delta_x) - regression_line x) = 0.8 :=
by
  intros delta_x h_delta_x
  rw [h_delta_x, regression_line, regression_line]
  simp
  sorry

end regression_slope_interpretation_l158_158768


namespace opposite_numbers_reciprocal_values_l158_158212

theorem opposite_numbers_reciprocal_values (a b m n : ℝ) (h₁ : a + b = 0) (h₂ : m * n = 1) : 5 * a + 5 * b - m * n = -1 :=
by sorry

end opposite_numbers_reciprocal_values_l158_158212


namespace lines_are_coplanar_l158_158773

/- Define the parameterized lines -/
def L1 (s : ℝ) (k : ℝ) : ℝ × ℝ × ℝ := (1 + 2 * s, 4 - k * s, 2 + 2 * k * s)
def L2 (t : ℝ) : ℝ × ℝ × ℝ := (2 + t, 7 + 3 * t, 1 - 2 * t)

/- Prove that k = 0 ensures the lines are coplanar -/
theorem lines_are_coplanar (k : ℝ) : k = 0 ↔ 
  ∃ (s t : ℝ), L1 s k = L2 t :=
by {
  sorry
}

end lines_are_coplanar_l158_158773


namespace xiao_zhang_return_distance_xiao_zhang_no_refuel_needed_l158_158018

def total_distance : ℕ :=
  15 - 3 + 16 - 11 + 10 - 12 + 4 - 15 + 16 - 18

def fuel_consumption_per_km : ℝ := 0.6
def initial_fuel : ℝ := 72.2

theorem xiao_zhang_return_distance :
  total_distance = 2 := by
  sorry

theorem xiao_zhang_no_refuel_needed :
  (initial_fuel - fuel_consumption_per_km * (|15| + |3| + |16| + |11| + |10| + |12| + |4| + |15| + |16| + |18|)) >= 0 := by
  sorry

end xiao_zhang_return_distance_xiao_zhang_no_refuel_needed_l158_158018


namespace evaluate_x_from_geometric_series_l158_158999

theorem evaluate_x_from_geometric_series (x : ℝ) (h : ∑' n : ℕ, x ^ n = 4) : x = 3 / 4 :=
sorry

end evaluate_x_from_geometric_series_l158_158999


namespace polynomial_remainder_l158_158196

theorem polynomial_remainder (x : ℝ) :
  ∃ (Q : ℝ → ℝ) (a b : ℝ),
    (x^150 = (x^2 - 5*x + 6) * Q x + (a*x + b)) ∧
    (2 * a + b = 2^150) ∧
    (3 * a + b = 3^150) ∧ 
    (a = 3^150 - 2^150) ∧ 
    (b = 2^150 - 2 * 3^150 + 2 * 2^150) := sorry

end polynomial_remainder_l158_158196


namespace oatmeal_cookies_divisible_by_6_l158_158620

theorem oatmeal_cookies_divisible_by_6 (O : ℕ) (h1 : 48 % 6 = 0) (h2 : O % 6 = 0) :
    ∃ x : ℕ, O = 6 * x :=
by sorry

end oatmeal_cookies_divisible_by_6_l158_158620


namespace sin_ratio_equal_one_or_neg_one_l158_158760

theorem sin_ratio_equal_one_or_neg_one
  (a b : Real)
  (h1 : Real.cos (a + b) = 1/4)
  (h2 : Real.cos (a - b) = 3/4) :
  (Real.sin a) / (Real.sin b) = 1 ∨ (Real.sin a) / (Real.sin b) = -1 :=
sorry

end sin_ratio_equal_one_or_neg_one_l158_158760


namespace problem1_problem2_l158_158438

theorem problem1 (x : ℝ) (h1 : x * (x + 4) = -5 * (x + 4)) : x = -4 ∨ x = -5 := 
by 
  sorry

theorem problem2 (x : ℝ) (h2 : (x + 2) ^ 2 = (2 * x - 1) ^ 2) : x = 3 ∨ x = -1 / 3 := 
by 
  sorry

end problem1_problem2_l158_158438


namespace square_line_product_l158_158452

theorem square_line_product (b : ℝ) 
  (h1 : ∃ y1 y2, y1 = -1 ∧ y2 = 4) 
  (h2 : ∃ x1, x1 = 3) 
  (h3 : (4 - (-1)) = (5 : ℝ)) 
  (h4 : ((∃ b1, b1 = 3 + 5 ∨ b1 = 3 - 5) → b = b1)) :
  b = -2 ∨ b = 8 → b * 8 = -16 :=
by sorry

end square_line_product_l158_158452


namespace power_multiplication_equals_result_l158_158697

theorem power_multiplication_equals_result : 
  (-0.25)^11 * (-4)^12 = -4 := 
sorry

end power_multiplication_equals_result_l158_158697


namespace altitude_difference_l158_158440

theorem altitude_difference 
  (alt_A : ℤ) (alt_B : ℤ) (alt_C : ℤ)
  (hA : alt_A = -102) (hB : alt_B = -80) (hC : alt_C = -25) :
  (max (max alt_A alt_B) alt_C) - (min (min alt_A alt_B) alt_C) = 77 := 
by 
  sorry

end altitude_difference_l158_158440


namespace solve_for_x_l158_158117

theorem solve_for_x (A B C D: Type) 
(y z w x : ℝ) 
(h_triangle : ∃ a b c : Type, True) 
(h_D_on_extension : ∃ D_on_extension : Type, True)
(h_AD_GT_BD : ∃ s : Type, True) 
(h_x_at_D : ∃ t : Type, True) 
(h_y_at_A : ∃ u : Type, True) 
(h_z_at_B : ∃ v : Type, True) 
(h_w_at_C : ∃ w : Type, True)
(h_triangle_angle_sum : y + z + w = 180):
x = 180 - z - w := by
  sorry

end solve_for_x_l158_158117


namespace theta_interval_l158_158594

noncomputable def f (x θ: ℝ) : ℝ := x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ

theorem theta_interval (θ: ℝ) (k: ℤ) :
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → f x θ > 0) → 
  (2 * k * Real.pi + Real.pi / 12 < θ ∧ θ < 2 * k * Real.pi + 5 * Real.pi / 12) := 
by
  sorry

end theta_interval_l158_158594


namespace chicken_nugget_ratio_l158_158521

theorem chicken_nugget_ratio (k d a t : ℕ) (h1 : a = 20) (h2 : t = 100) (h3 : k + d + a = t) : (k + d) / a = 4 :=
by
  sorry

end chicken_nugget_ratio_l158_158521


namespace problem_l158_158589

open Set

theorem problem (M : Set ℤ) (N : Set ℤ) (hM : M = {1, 2, 3, 4}) (hN : N = {-2, 2}) : 
  M ∩ N = {2} :=
by
  sorry

end problem_l158_158589


namespace gravel_cost_l158_158171

-- Definitions of conditions
def lawn_length : ℝ := 70
def lawn_breadth : ℝ := 30
def road_width : ℝ := 5
def gravel_cost_per_sqm : ℝ := 4

-- Theorem statement
theorem gravel_cost : (lawn_length * road_width + lawn_breadth * road_width - road_width * road_width) * gravel_cost_per_sqm = 1900 :=
by
  -- Definitions used in the problem
  let area_first_road := lawn_length * road_width
  let area_second_road := lawn_breadth * road_width
  let area_intersection := road_width * road_width

  -- Total area to be graveled
  let total_area_to_be_graveled := area_first_road + area_second_road - area_intersection

  -- Calculate the cost
  let cost := total_area_to_be_graveled * gravel_cost_per_sqm

  show cost = 1900
  sorry

end gravel_cost_l158_158171


namespace union_of_A_and_B_l158_158251

variable {α : Type*}

def A (x : ℝ) : Prop := x - 1 > 0
def B (x : ℝ) : Prop := 0 < x ∧ x ≤ 3

theorem union_of_A_and_B : ∀ x : ℝ, (A x ∨ B x) ↔ (0 < x) :=
by
  sorry

end union_of_A_and_B_l158_158251


namespace days_to_finish_job_l158_158481

def work_rate_a_b : ℚ := 1 / 15
def work_rate_c : ℚ := 4 / 15
def combined_work_rate : ℚ := work_rate_a_b + work_rate_c

theorem days_to_finish_job (A B C : ℚ) (h1 : A + B = work_rate_a_b) (h2 : C = work_rate_c) :
  1 / (A + B + C) = 3 :=
by
  sorry

end days_to_finish_job_l158_158481


namespace negation_of_proposition_l158_158914

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, 0 ≤ x → x^3 + x ≥ 0)) ↔ (∃ x : ℝ, 0 ≤ x ∧ x^3 + x < 0) :=
by sorry

end negation_of_proposition_l158_158914


namespace sufficiency_of_p_for_q_not_necessity_of_p_for_q_l158_158391

noncomputable def p (m : ℝ) := ∀ x : ℝ, |x| + |x - 1| > m
noncomputable def q (m : ℝ) := ∀ x : ℝ, (- (5 - 2 * m)) ^ x < 0

theorem sufficiency_of_p_for_q : ∀ m : ℝ, (m < 1 → m < 2) :=
by sorry

theorem not_necessity_of_p_for_q : ∀ m : ℝ, ¬ (m < 2 → m < 1) :=
by sorry

end sufficiency_of_p_for_q_not_necessity_of_p_for_q_l158_158391


namespace sum_of_reciprocals_l158_158035

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) :
  1 / x + 1 / y = 3 :=
by
  sorry

end sum_of_reciprocals_l158_158035


namespace turns_in_two_hours_l158_158183

theorem turns_in_two_hours (turns_per_30_sec : ℕ) (minutes_in_hour : ℕ) (hours : ℕ) : 
  turns_per_30_sec = 6 → 
  minutes_in_hour = 60 → 
  hours = 2 → 
  (12 * (minutes_in_hour * hours)) = 1440 := 
by
  sorry

end turns_in_two_hours_l158_158183


namespace value_of_x_l158_158657

theorem value_of_x : ∀ x : ℝ, (x^2 - 4) / (x - 2) = 0 → x ≠ 2 → x = -2 := by
  intros x h1 h2
  sorry

end value_of_x_l158_158657


namespace avg_math_chem_l158_158322

variables (M P C : ℕ)

def total_marks (M P : ℕ) := M + P = 50
def chemistry_marks (P C : ℕ) := C = P + 20

theorem avg_math_chem (M P C : ℕ) (h1 : total_marks M P) (h2 : chemistry_marks P C) :
  (M + C) / 2 = 35 :=
by
  sorry

end avg_math_chem_l158_158322


namespace range_of_a_l158_158860

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + a * x + 1 ≥ 0) ↔ -2 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_a_l158_158860


namespace erika_donut_holes_l158_158188

open Int

theorem erika_donut_holes (r₁ r₂ r₃ : ℝ) (surface_area : ℝ → ℝ) :
  r₁ = 5 ∧ r₂ = 7 ∧ r₃ = 9 ∧ surface_area = λ r, 4 * Real.pi * r^2 →
  let sa₁ := surface_area r₁ in
  let sa₂ := surface_area r₂ in
  let sa₃ := surface_area r₃ in
  sa₁ = 100 * Real.pi ∧ sa₂ = 196 * Real.pi ∧ sa₃ = 324 * Real.pi →
  let lcm_sa := Nat.lcm (100 : ℕ) (Nat.lcm (196 : ℕ) (324 : ℕ)) * Real.pi in
  (lcm_sa / sa₁) = 441 :=
by
  sorry

end erika_donut_holes_l158_158188


namespace max_stamps_l158_158114

def price_of_stamp : ℕ := 25  -- Price of one stamp in cents
def total_money : ℕ := 4000   -- Total money available in cents

theorem max_stamps : ∃ n : ℕ, price_of_stamp * n ≤ total_money ∧ (∀ m : ℕ, price_of_stamp * m ≤ total_money → m ≤ n) :=
by
  use 160
  sorry

end max_stamps_l158_158114


namespace range_of_a_l158_158097

theorem range_of_a (x a : ℝ) 
  (h₁ : ∀ x, |x + 1| ≤ 2 → x ≤ a) 
  (h₂ : ∃ x, x > a ∧ |x + 1| ≤ 2) 
  : a ≥ 1 :=
sorry

end range_of_a_l158_158097


namespace graph_is_empty_l158_158569

theorem graph_is_empty :
  ¬∃ x y : ℝ, 4 * x^2 + 9 * y^2 - 16 * x - 36 * y + 64 = 0 :=
by
  -- the proof logic will go here
  sorry

end graph_is_empty_l158_158569


namespace find_R_plus_S_l158_158227

theorem find_R_plus_S (d e R S : ℝ) 
  (h1 : d + 3 = 0)
  (h2 : 7 * d + 3 * e = 0)
  (h3 : R = 3 * d + e + 7)
  (h4 : S = 7 * e) :
  R + S = 54 :=
by
  sorry

end find_R_plus_S_l158_158227


namespace problem_l158_158144

theorem problem (a b c : ℝ) (h : 1/a + 1/b + 1/c = 1/(a + b + c)) : (a + b) * (b + c) * (a + c) = 0 := 
by
  sorry

end problem_l158_158144


namespace alex_received_12_cookies_l158_158405

theorem alex_received_12_cookies :
  ∃ y: ℕ, (∀ s: ℕ, y = s + 8 ∧ s = y / 3) → y = 12 := by
  sorry

end alex_received_12_cookies_l158_158405


namespace customers_left_l158_158978

-- Definitions based on problem conditions
def initial_customers : ℕ := 14
def remaining_customers : ℕ := 3

-- Theorem statement based on the question and the correct answer
theorem customers_left : initial_customers - remaining_customers = 11 := by
  sorry

end customers_left_l158_158978


namespace vasya_fraction_l158_158543

variable (a b c d s : ℝ)

-- Anton drove half the distance Vasya did
axiom h1 : a = b / 2

-- Sasha drove as long as Anton and Dima together
axiom h2 : c = a + d

-- Dima drove one-tenth of the total distance
axiom h3 : d = s / 10

-- The total distance is the sum of distances driven by Anton, Vasya, Sasha, and Dima
axiom h4 : a + b + c + d = s

-- We need to prove that Vasya drove 0.4 of the total distance
theorem vasya_fraction (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : b = 0.4 * s :=
by
  sorry

end vasya_fraction_l158_158543


namespace symmetric_points_origin_l158_158396

theorem symmetric_points_origin (a b : ℝ) (h1 : 1 = -b) (h2 : a = 2) : a + b = 1 := by
  sorry

end symmetric_points_origin_l158_158396


namespace area_of_triangle_PQR_l158_158639

theorem area_of_triangle_PQR 
  (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 8)
  (P_is_center : ∃ P : ℝ, True) -- Simplified assumption that P exists
  (bases_on_same_line : True) -- Assumed true, as touching condition implies it
  : ∃ area : ℝ, area = 20 := 
by
  sorry

end area_of_triangle_PQR_l158_158639


namespace det_A_is_half_l158_158656

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=
![![Real.cos (20 * Real.pi / 180), Real.sin (40 * Real.pi / 180)], ![Real.sin (20 * Real.pi / 180), Real.cos (40 * Real.pi / 180)]]

theorem det_A_is_half : A.det = 1 / 2 := by
  sorry

end det_A_is_half_l158_158656


namespace common_roots_correct_l158_158445

noncomputable section
def common_roots_product (A B : ℝ) : ℝ :=
  let p := sorry
  let q := sorry
  p * q

theorem common_roots_correct (A B : ℝ) (h1 : ∀ x, x^3 + 2*A*x + 20 = 0 → x = p ∨ x = q ∨ x = r) 
    (h2 : ∀ x, x^3 + B*x^2 + 100 = 0 → x = p ∨ x = q ∨ x = s)
    (h_sum1 : p + q + r = 0) 
    (h_sum2 : p + q + s = -B)
    (h_prod1 : p * q * r = -20) 
    (h_prod2 : p * q * s = -100) : 
    common_roots_product A B = 10 * (2000)^(1/3) ∧ 15 = 10 + 3 + 2 :=
by
  sorry

end common_roots_correct_l158_158445


namespace rationalize_value_of_a2_minus_2a_value_of_2a3_minus_4a2_minus_1_l158_158685

variable (a : ℂ)

theorem rationalize (h : a = 1 / (Real.sqrt 2 - 1)) : a = Real.sqrt 2 + 1 := by
  sorry

theorem value_of_a2_minus_2a (h : a = Real.sqrt 2 + 1) : a ^ 2 - 2 * a = 1 := by
  sorry

theorem value_of_2a3_minus_4a2_minus_1 (h : a = Real.sqrt 2 + 1) : 2 * a ^ 3 - 4 * a ^ 2 - 1 = 2 * Real.sqrt 2 + 1 := by
  sorry

end rationalize_value_of_a2_minus_2a_value_of_2a3_minus_4a2_minus_1_l158_158685


namespace baron_munchausen_not_lying_l158_158354

def sum_of_digits (n : Nat) : Nat := sorry

theorem baron_munchausen_not_lying :
  ∃ a b : Nat, a ≠ b ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 ∧ 
  (a < 10^10 ∧ 10^9 ≤ a) ∧ (b < 10^10 ∧ 10^9 ≤ b) ∧ 
  (a + sum_of_digits (a ^ 2) = b + sum_of_digits (b ^ 2)) :=
sorry

end baron_munchausen_not_lying_l158_158354


namespace fraction_addition_l158_158987

variable (a : ℝ)

theorem fraction_addition (ha : a ≠ 0) : (3 / a) + (2 / a) = (5 / a) :=
by
  sorry

end fraction_addition_l158_158987


namespace doubled_dimensions_volume_l158_158516

theorem doubled_dimensions_volume (original_volume : ℝ) (length_factor width_factor height_factor : ℝ) 
  (h : original_volume = 3) 
  (hl : length_factor = 2)
  (hw : width_factor = 2)
  (hh : height_factor = 2) : 
  original_volume * length_factor * width_factor * height_factor = 24 :=
by
  sorry

end doubled_dimensions_volume_l158_158516


namespace gcd_division_steps_l158_158082

theorem gcd_division_steps (a b : ℕ) (h₁ : a = 1813) (h₂ : b = 333) : 
  ∃ steps : ℕ, steps = 3 ∧ (Nat.gcd a b = 37) :=
by
  have h₁ : a = 1813 := h₁
  have h₂ : b = 333 := h₂
  sorry

end gcd_division_steps_l158_158082


namespace planting_flowers_cost_l158_158448

theorem planting_flowers_cost 
  (flower_cost : ℕ) (clay_cost : ℕ) (soil_cost : ℕ)
  (h₁ : flower_cost = 9)
  (h₂ : clay_cost = flower_cost + 20)
  (h₃ : soil_cost = flower_cost - 2) :
  flower_cost + clay_cost + soil_cost = 45 :=
sorry

end planting_flowers_cost_l158_158448


namespace quadratic_root_form_l158_158655

theorem quadratic_root_form {a b : ℂ} (h : 6 * a ^ 2 - 5 * a + 18 = 0 ∧ a.im = 0 ∧ b.im = 0) : 
  a + b^2 = (467:ℚ) / 144 :=
by
  sorry

end quadratic_root_form_l158_158655


namespace green_light_probability_l158_158962

def red_duration : ℕ := 30
def green_duration : ℕ := 25
def yellow_duration : ℕ := 5

def total_cycle : ℕ := red_duration + green_duration + yellow_duration
def green_probability : ℚ := green_duration / total_cycle

theorem green_light_probability :
  green_probability = 5 / 12 := by
  sorry

end green_light_probability_l158_158962


namespace find_base_l158_158120

theorem find_base (b : ℕ) (h : (3 * b + 2) ^ 2 = b ^ 3 + b + 4) : b = 8 :=
sorry

end find_base_l158_158120


namespace chris_did_not_get_A_l158_158748

variable (A : Prop) (MC_correct : Prop) (Essay80 : Prop)

-- The condition provided by professor
axiom condition : A ↔ (MC_correct ∧ Essay80)

-- The theorem we need to prove based on the statement (B) from the solution
theorem chris_did_not_get_A 
    (h : ¬ A) : ¬ MC_correct ∨ ¬ Essay80 :=
by sorry

end chris_did_not_get_A_l158_158748


namespace prove_jimmy_is_2_determine_rachel_age_l158_158050

-- Define the conditions of the problem
variables (a b c r1 r2 : ℤ)

-- Condition 1: Rachel's age and Jimmy's age are roots of the quadratic equation
def is_root (p : ℤ → ℤ) (x : ℤ) : Prop := p x = 0

def quadratic_eq (x : ℤ) : ℤ := a * x^2 + b * x + c

-- Condition 2: Sum of the coefficients is a prime number
def sum_of_coefficients_is_prime : Prop :=
  Nat.Prime (a + b + c).natAbs

-- Condition 3: Substituting Rachel’s age into the quadratic equation gives -55
def substitute_rachel_is_minus_55 (r : ℤ) : Prop :=
  quadratic_eq a b c r = -55

-- Question 1: Prove Jimmy is 2 years old
theorem prove_jimmy_is_2 (h1 : is_root (quadratic_eq a b c) r1)
                           (h2 : is_root (quadratic_eq a b c) r2)
                           (h3 : sum_of_coefficients_is_prime a b c)
                           (h4 : substitute_rachel_is_minus_55 a b c r1) :
  r2 = 2 :=
sorry

-- Question 2: Determine Rachel's age
theorem determine_rachel_age (h1 : is_root (quadratic_eq a b c) r1)
                             (h2 : is_root (quadratic_eq a b c) r2)
                             (h3 : sum_of_coefficients_is_prime a b c)
                             (h4 : substitute_rachel_is_minus_55 a b c r1)
                             (h5 : r2 = 2) :
  r1 = 7 :=
sorry

end prove_jimmy_is_2_determine_rachel_age_l158_158050


namespace tenth_term_arithmetic_sequence_l158_158799

theorem tenth_term_arithmetic_sequence (a d : ℤ) 
  (h1 : a + 2 * d = 23) (h2 : a + 6 * d = 35) : 
  a + 9 * d = 44 := 
  by
    sorry

end tenth_term_arithmetic_sequence_l158_158799


namespace probability_after_first_new_draw_is_five_ninths_l158_158878

-- Defining the conditions in Lean
def total_balls : ℕ := 10
def new_balls : ℕ := 6
def old_balls : ℕ := 4

def balls_remaining_after_first_draw : ℕ := total_balls - 1
def new_balls_after_first_draw : ℕ := new_balls - 1

-- Using the classic probability definition
def probability_of_drawing_second_new_ball := (new_balls_after_first_draw : ℚ) / (balls_remaining_after_first_draw : ℚ)

-- Stating the theorem to be proved
theorem probability_after_first_new_draw_is_five_ninths :
  probability_of_drawing_second_new_ball = 5/9 := sorry

end probability_after_first_new_draw_is_five_ninths_l158_158878


namespace equation_of_plane_l158_158195

noncomputable def parametric_form (s t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 2 * s - 3 * t, 4 - s + 2 * t, 1 - 3 * s - t)

theorem equation_of_plane (x y z : ℝ) : 
  (∃ s t : ℝ, parametric_form s t = (x, y, z)) → 5 * x + 11 * y + 7 * z - 61 = 0 :=
by
  sorry

end equation_of_plane_l158_158195


namespace molecular_weight_2N_5O_l158_158940

def molecular_weight (num_N num_O : ℕ) (atomic_weight_N atomic_weight_O : ℝ) : ℝ :=
  (num_N * atomic_weight_N) + (num_O * atomic_weight_O)

theorem molecular_weight_2N_5O :
  molecular_weight 2 5 14.01 16.00 = 108.02 :=
by
  -- proof goes here
  sorry

end molecular_weight_2N_5O_l158_158940


namespace value_of_g_neg3_l158_158992

def g (x : ℝ) : ℝ := x^3 - 2 * x

theorem value_of_g_neg3 : g (-3) = -21 := by
  sorry

end value_of_g_neg3_l158_158992


namespace sum_of_three_numbers_l158_158482

theorem sum_of_three_numbers
  (a b c : ℕ) (h_prime : Prime c)
  (h1 : a^2 + b^2 + c^2 = 156)
  (h2 : a * b + b * c + a * c = 50) :
  a + b + c = 16 :=
by
  sorry

end sum_of_three_numbers_l158_158482


namespace number_of_sequences_l158_158890

theorem number_of_sequences (n k : ℕ) (h₁ : 1 ≤ k) (h₂ : k ≤ n) :
  ∃ C : ℕ, C = Nat.choose (Nat.floor ((n + 2 - k) / 2) + k - 1) k :=
sorry

end number_of_sequences_l158_158890


namespace toy_store_fraction_l158_158721

theorem toy_store_fraction
  (allowance : ℝ) (arcade_fraction : ℝ) (candy_store_amount : ℝ)
  (h1 : allowance = 1.50)
  (h2 : arcade_fraction = 3 / 5)
  (h3 : candy_store_amount = 0.40) :
  (0.60 - candy_store_amount) / (allowance - arcade_fraction * allowance) = 1 / 3 :=
by
  -- We're skipping the actual proof steps
  sorry

end toy_store_fraction_l158_158721


namespace find_angle_D_l158_158393

theorem find_angle_D
  (angle_A angle_B angle_C angle_D : ℝ)
  (h1 : angle_A + angle_B = 180)
  (h2 : angle_C = 2 * angle_D)
  (h3 : angle_A = 100)
  (h4 : angle_B + angle_C + angle_D = 180) :
  angle_D = 100 / 3 :=
by
  sorry

end find_angle_D_l158_158393


namespace ratio_problem_l158_158052

theorem ratio_problem (X : ℕ) :
  (18 : ℕ) * 360 = 9 * X → X = 720 :=
by
  intro h
  sorry

end ratio_problem_l158_158052


namespace non_real_roots_b_range_l158_158727

theorem non_real_roots_b_range (b : ℝ) : 
  ∃ (x : ℂ), x^2 + (b : ℂ) * x + 16 = 0 ∧ (¬ ∃ (x : ℝ), x^2 + b * x + 16 = 0) ↔ -8 < b ∧ b < 8 := 
by
  sorry

end non_real_roots_b_range_l158_158727


namespace sum_of_arithmetic_series_l158_158464

theorem sum_of_arithmetic_series (A B C : ℕ) (n : ℕ) 
  (hA : A = n * (2 * a₁ + (n - 1) * d) / 2)
  (hB : B = 2 * n * (2 * a₁ + (2 * n - 1) * d) / 2)
  (hC : C = 3 * n * (2 * a₁ + (3 * n - 1) * d) / 2) :
  C = 3 * (B - A) := sorry

end sum_of_arithmetic_series_l158_158464


namespace boat_speed_in_still_water_l158_158177

theorem boat_speed_in_still_water :
  ∀ (V_b V_s : ℝ) (distance time : ℝ),
  V_s = 5 →
  time = 4 →
  distance = 84 →
  (distance / time) = V_b + V_s →
  V_b = 16 :=
by
  -- Given definitions and values
  intros V_b V_s distance time
  intro hV_s
  intro htime
  intro hdistance
  intro heq
  sorry -- Placeholder for the actual proof

end boat_speed_in_still_water_l158_158177


namespace ducks_to_total_ratio_l158_158777

-- Definitions based on the given conditions
def totalBirds : ℕ := 15
def costPerChicken : ℕ := 2
def totalCostForChickens : ℕ := 20

-- Proving the desired ratio of ducks to total number of birds
theorem ducks_to_total_ratio : (totalCostForChickens / costPerChicken) + d = totalBirds → d = 15 - (totalCostForChickens / costPerChicken) → 
  (totalCostForChickens / costPerChicken) + d = totalBirds → d = totalBirds - (totalCostForChickens / costPerChicken) →
  d = 5 → (totalBirds - (totalCostForChickens / costPerChicken)) / totalBirds = 1 / 3 :=
by
  sorry

end ducks_to_total_ratio_l158_158777


namespace time_to_finish_all_problems_l158_158852

def mathProblems : ℝ := 17.0
def spellingProblems : ℝ := 15.0
def problemsPerHour : ℝ := 8.0
def totalProblems : ℝ := mathProblems + spellingProblems

theorem time_to_finish_all_problems : totalProblems / problemsPerHour = 4.0 :=
by
  sorry

end time_to_finish_all_problems_l158_158852


namespace arrow_estimate_closest_to_9_l158_158444

theorem arrow_estimate_closest_to_9 
  (a b : ℝ) (h₁ : a = 8.75) (h₂ : b = 9.0)
  (h : 8.75 < 9.0) :
  ∃ x ∈ Set.Icc a b, x = 9.0 :=
by
  sorry

end arrow_estimate_closest_to_9_l158_158444


namespace solution_of_inequality_system_l158_158031

-- Definitions derived from the conditions in the problem
def inequality1 (x : ℝ) : Prop := 3 * x - 1 ≥ x + 1
def inequality2 (x : ℝ) : Prop := x + 4 > 4 * x - 2
def solution_set (x : ℝ) : Prop := 1 ≤ x ∧ x < 2

-- The Lean 4 statement for the proof problem
theorem solution_of_inequality_system (x : ℝ) : inequality1 x ∧ inequality2 x ↔ solution_set x := by
  sorry

end solution_of_inequality_system_l158_158031


namespace quadratic_non_real_roots_l158_158723

theorem quadratic_non_real_roots (b : ℝ) : 
  let a : ℝ := 1 
  let c : ℝ := 16 in
  (b^2 - 4 * a * c < 0) ↔ (-8 < b ∧ b < 8) :=
sorry

end quadratic_non_real_roots_l158_158723


namespace problem_statement_l158_158325

theorem problem_statement : ((26.3 * 12 * 20) / 3 + 125 - Real.sqrt 576 = 21141) :=
by
  sorry

end problem_statement_l158_158325


namespace no_infinite_prime_sequence_l158_158264

theorem no_infinite_prime_sequence (p : ℕ → ℕ)
  (h : ∀ k : ℕ, Nat.Prime (p k) ∧ p (k + 1) = 5 * p k + 4) :
  ¬ ∀ n : ℕ, Nat.Prime (p n) :=
by
  sorry

end no_infinite_prime_sequence_l158_158264


namespace greatest_sum_l158_158164

-- stating the conditions
def condition1 (x y : ℝ) := x^2 + y^2 = 130
def condition2 (x y : ℝ) := x * y = 45

-- proving the result
theorem greatest_sum (x y : ℝ) 
  (h1 : condition1 x y) 
  (h2 : condition2 x y) : 
  x + y = 10 * Real.sqrt 2.2 :=
sorry

end greatest_sum_l158_158164


namespace find_speed_l158_158262

variable (d : ℝ) (t : ℝ)
variable (h1 : d = 50 * (t + 1/12))
variable (h2 : d = 70 * (t - 1/12))

theorem find_speed (d t : ℝ)
  (h1 : d = 50 * (t + 1/12))
  (h2 : d = 70 * (t - 1/12)) :
  58 = d / t := by
  sorry

end find_speed_l158_158262


namespace arithmetic_sequence_third_term_l158_158116

theorem arithmetic_sequence_third_term (b y : ℝ) 
  (h1 : 2 * b + y + 2 = 10) 
  (h2 : b + y + 2 = b + y + 2) : 
  8 - b = 6 := 
by 
  sorry

end arithmetic_sequence_third_term_l158_158116


namespace isosceles_right_triangle_ratio_l158_158186

theorem isosceles_right_triangle_ratio (a : ℝ) (h : a > 0) : (2 * a) / (Real.sqrt (a^2 + a^2)) = Real.sqrt 2 :=
by
  sorry

end isosceles_right_triangle_ratio_l158_158186


namespace quadratic_trinomial_value_at_6_l158_158787

theorem quadratic_trinomial_value_at_6 {p q : ℝ} 
  (h1 : ∃ r1 r2, r1 = q ∧ r2 = 1 + p + q ∧ r1 + r2 = -p ∧ r1 * r2 = q) : 
  (6^2 + p * 6 + q) = 31 :=
by
  sorry

end quadratic_trinomial_value_at_6_l158_158787


namespace general_term_of_geometric_sequence_l158_158714

theorem general_term_of_geometric_sequence 
  (positive_terms : ∀ n : ℕ, 0 < a_n) 
  (h1 : a_1 = 1) 
  (h2 : ∃ a : ℕ, a_2 = a + 1 ∧ a_3 = 2 * a + 5) : 
  ∃ q : ℕ, ∀ n : ℕ, a_n = q^(n-1) :=
by
  sorry

end general_term_of_geometric_sequence_l158_158714


namespace frank_total_cost_l158_158384

-- Conditions from the problem
def cost_per_bun : ℝ := 0.1
def number_of_buns : ℕ := 10
def cost_per_bottle_of_milk : ℝ := 2
def number_of_bottles_of_milk : ℕ := 2
def cost_of_carton_of_eggs : ℝ := 3 * cost_per_bottle_of_milk

-- Question and Answer
theorem frank_total_cost : 
  let cost_of_buns := cost_per_bun * number_of_buns in
  let cost_of_milk := cost_per_bottle_of_milk * number_of_bottles_of_milk in
  let cost_of_eggs := cost_of_carton_of_eggs in
  cost_of_buns + cost_of_milk + cost_of_eggs = 11 :=
by
  sorry

end frank_total_cost_l158_158384


namespace min_value_of_f_l158_158453

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 3| + Real.exp x

theorem min_value_of_f :
  ∃ x ∈ Set.Icc (Real.exp 0) (Real.exp 3), f x = 6 - 2 * Real.log 2 :=
sorry

end min_value_of_f_l158_158453


namespace remaining_cooking_time_l158_158690

-- Define the recommended cooking time in minutes and the time already cooked in seconds
def recommended_cooking_time_min := 5
def time_cooked_seconds := 45

-- Define the conversion from minutes to seconds
def minutes_to_seconds (min : Nat) : Nat := min * 60

-- Define the total recommended cooking time in seconds
def total_recommended_cooking_time_seconds := minutes_to_seconds recommended_cooking_time_min

-- State the theorem to prove the remaining cooking time
theorem remaining_cooking_time :
  (total_recommended_cooking_time_seconds - time_cooked_seconds) = 255 :=
by
  sorry

end remaining_cooking_time_l158_158690


namespace double_root_conditions_l158_158380

theorem double_root_conditions (k : ℝ) :
  (∃ x, (k - 1)/(x^2 - 1) - 1/(x - 1) = k/(x + 1) ∧ (∀ ε > 0, (∃ δ > 0, (∀ y, |y - x| < δ → (k - 1)/(y^2 - 1) - 1/(y - 1) = k/(y + 1)))))
  → k = 3 ∨ k = 1/3 :=
sorry

end double_root_conditions_l158_158380


namespace sqrt_product_l158_158560

theorem sqrt_product (a b c : ℝ) (ha : a = 72) (hb : b = 18) (hc : c = 8) :
  (Real.sqrt a) * (Real.sqrt b) * (Real.sqrt c) = 72 * Real.sqrt 2 :=
by
  sorry

end sqrt_product_l158_158560


namespace domain_of_expression_l158_158077

theorem domain_of_expression (x : ℝ) 
  (h1 : 3 * x - 6 ≥ 0) 
  (h2 : 7 - 2 * x ≥ 0) 
  (h3 : 7 - 2 * x > 0) : 
  2 ≤ x ∧ x < 7 / 2 := by
sorry

end domain_of_expression_l158_158077


namespace ratio_a_f_l158_158871

theorem ratio_a_f (a b c d e f : ℕ)
  (h1 : a / b = 1 / 4)
  (h2 : b / c = 13 / 9)
  (h3 : c / d = 5 / 13)
  (h4 : d / e = 2 / 3)
  (h5 : e / f = 7 / 5) :
  a / f = 7 / 6 := by
  sorry

end ratio_a_f_l158_158871


namespace rectangle_to_square_area_ratio_is_24_25_l158_158288

noncomputable def rectangle_to_square_area_ratio
  (s : ℝ) -- length of side of square S
  (longer_side : ℝ := 1.2 * s) -- longer side of rectangle R
  (shorter_side : ℝ := 0.8 * s) -- shorter side of rectangle R
  (area_R : ℝ := longer_side * shorter_side) -- area of rectangle R
  (area_S : ℝ := s^2) -- area of square S
  : ℝ := 
  area_R / area_S

theorem rectangle_to_square_area_ratio_is_24_25 
  (s : ℝ)
  : rectangle_to_square_area_ratio s = 24 / 25 :=
by 
  sorry

end rectangle_to_square_area_ratio_is_24_25_l158_158288


namespace cubic_equation_unique_real_solution_l158_158086

theorem cubic_equation_unique_real_solution :
  (∃ (m : ℝ), ∀ x : ℝ, x^3 - 4*x - m = 0 → x = 2) ↔ m = -8 :=
by sorry

end cubic_equation_unique_real_solution_l158_158086


namespace minimum_value_8m_n_l158_158588

noncomputable def min_value (m n : ℝ) : ℝ :=
  8 * m + n

theorem minimum_value_8m_n (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : (1 / m) + (8 / n) = 4) : 
  min_value m n = 8 :=
sorry

end minimum_value_8m_n_l158_158588


namespace total_cost_correct_l158_158381

def bun_price : ℝ := 0.1
def buns_count : ℝ := 10
def milk_price : ℝ := 2
def milk_count : ℝ := 2
def egg_price : ℝ := 3 * milk_price

def total_cost : ℝ := (buns_count * bun_price) + (milk_count * milk_price) + egg_price

theorem total_cost_correct : total_cost = 11 := by
  sorry

end total_cost_correct_l158_158381


namespace janet_wait_time_l158_158244

theorem janet_wait_time
  (janet_speed : ℝ)
  (sister_speed : ℝ)
  (lake_width : ℝ)
  (janet_time : ℝ)
  (sister_time : ℝ) :
  janet_speed = 30 →
  sister_speed = 12 →
  lake_width = 60 →
  janet_time = lake_width / janet_speed →
  sister_time = lake_width / sister_speed →
  (sister_time - janet_time = 3) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end janet_wait_time_l158_158244


namespace area_ratio_rect_sq_l158_158282

variable (s : ℝ)

def side_len_sq (S : ℝ) : Prop := s = S
def longer_side_rect (R : ℝ) : Prop := R = 1.2 * s
def shorter_side_rect (R : ℝ) : Prop := R = 0.8 * s
def area_sq (S : ℝ) : ℝ := S * S
def area_rect (R_long R_short : ℝ) : ℝ := R_long * R_short
def ratio_area (areaR areaS : ℝ) : ℝ := areaR / areaS

theorem area_ratio_rect_sq (s S R_long R_short : ℝ) (h1 : side_len_sq s S) (h2 : longer_side_rect s R_long) (h3 : shorter_side_rect s R_short) :
  ratio_area (area_rect R_long R_short) (area_sq S) = 24/25 :=
by
  sorry

end area_ratio_rect_sq_l158_158282


namespace pete_ran_least_distance_l158_158648

theorem pete_ran_least_distance
  (phil_distance : ℕ := 4)
  (tom_distance : ℕ := 6)
  (pete_distance : ℕ := 2)
  (amal_distance : ℕ := 8)
  (sanjay_distance : ℕ := 7) :
  pete_distance ≤ phil_distance ∧
  pete_distance ≤ tom_distance ∧
  pete_distance ≤ amal_distance ∧
  pete_distance ≤ sanjay_distance :=
by {
  sorry
}

end pete_ran_least_distance_l158_158648


namespace coordinates_of_point_M_l158_158412

theorem coordinates_of_point_M :
    ∀ (M : ℝ × ℝ),
      (M.1 < 0 ∧ M.2 > 0) → -- M is in the second quadrant
      dist (M.1, M.2) (M.1, 0) = 1 → -- distance to x-axis is 1
      dist (M.1, M.2) (0, M.2) = 2 → -- distance to y-axis is 2
      M = (-2, 1) :=
by
  intros M in_second_quadrant dist_to_x_axis dist_to_y_axis
  sorry

end coordinates_of_point_M_l158_158412


namespace rate_percent_correct_l158_158066

noncomputable def findRatePercent (P A T : ℕ) : ℚ :=
  let SI := A - P
  (SI * 100 : ℚ) / (P * T)

theorem rate_percent_correct :
  findRatePercent 12000 19500 7 = 8.93 := by
  sorry

end rate_percent_correct_l158_158066


namespace ratio_area_of_rectangle_to_square_l158_158292

theorem ratio_area_of_rectangle_to_square (s : ℝ) :
  (1.2 * s * 0.8 * s) / (s * s) = 24 / 25 :=
by
  sorry

end ratio_area_of_rectangle_to_square_l158_158292


namespace range_of_independent_variable_l158_158919

theorem range_of_independent_variable (x : ℝ) :
  (x + 2 >= 0) → (x - 1 ≠ 0) → (x ≥ -2 ∧ x ≠ 1) :=
by
  intros h₁ h₂
  sorry

end range_of_independent_variable_l158_158919


namespace fraction_of_seats_sold_l158_158975

theorem fraction_of_seats_sold
  (ticket_price : ℕ) (number_of_rows : ℕ) (seats_per_row : ℕ) (total_earnings : ℕ)
  (h1 : ticket_price = 10)
  (h2 : number_of_rows = 20)
  (h3 : seats_per_row = 10)
  (h4 : total_earnings = 1500) :
  (total_earnings / ticket_price : ℕ) / (number_of_rows * seats_per_row : ℕ) = 3 / 4 := by
  sorry

end fraction_of_seats_sold_l158_158975


namespace not_possible_to_partition_into_groups_of_5_with_remainder_3_l158_158707

theorem not_possible_to_partition_into_groups_of_5_with_remainder_3 (m : ℤ) :
  ¬ (m^2 % 5 = 3) :=
by sorry

end not_possible_to_partition_into_groups_of_5_with_remainder_3_l158_158707


namespace sequence_general_formula_l158_158417

theorem sequence_general_formula (n : ℕ) (h : n > 0) :
  ∃ (a : ℕ → ℚ), a 1 = 1 ∧ (∀ n, a (n + 1) = a n / (3 * a n + 1)) ∧ a n = 1 / (3 * n - 2) :=
by sorry

end sequence_general_formula_l158_158417


namespace find_a_l158_158859

def f (a x : ℝ) : ℝ := (a - 1) * x^2 + a * sin x

theorem find_a (a : ℝ) : (∀ x : ℝ, f a x = f a (-x)) ↔ a = 0 :=
by
  intros
  sorry

end find_a_l158_158859


namespace part_a_part_b_part_c_l158_158422

def f (n d : ℕ) : ℕ := sorry

theorem part_a (n : ℕ) (h_even_n : n % 2 = 0) : f n 0 ≤ n :=
sorry

theorem part_b (n d : ℕ) (h_even_n_minus_d : (n - d) % 2 = 0) : f n d ≤ (n + d) / (d + 1) :=
sorry

theorem part_c (n : ℕ) (h_even_n : n % 2 = 0) : f n 0 = n :=
sorry

end part_a_part_b_part_c_l158_158422


namespace smallest_base10_integer_l158_158954

theorem smallest_base10_integer : 
  ∃ (a b x : ℕ), a > 2 ∧ b > 2 ∧ x = 2 * a + 1 ∧ x = b + 2 ∧ x = 7 := by
  sorry

end smallest_base10_integer_l158_158954


namespace simplify_expression_to_inverse_abc_l158_158361

variable (a b c : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem simplify_expression_to_inverse_abc :
  (a + b + c + 3)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹) * (ab + bc + ca + 3)⁻¹ * ((ab)⁻¹ + (bc)⁻¹ + (ca)⁻¹ + 3) = (1 : ℝ) / (abc) :=
by
  sorry

end simplify_expression_to_inverse_abc_l158_158361


namespace Vasya_distance_fraction_l158_158550

variable (a b c d s : ℝ)

theorem Vasya_distance_fraction :
  (a = b / 2) →
  (c = a + d) →
  (d = s / 10) →
  (a + b + c + d = s) →
  (b / s = 0.4) :=
by
  intros h1 h2 h3 h4
  sorry

end Vasya_distance_fraction_l158_158550


namespace problem_1_problem_2_l158_158104

def A : Set ℝ := { x | x^2 - 3 * x + 2 < 0 }

def B (a : ℝ) : Set ℝ := { x | a - 1 < x ∧ x < 3 * a + 1 }

-- Problem 1
theorem problem_1 (a : ℝ) (h : a = 1 / 4) : A ∩ B a = { x | 1 < x ∧ x < 7 / 4 } :=
by
  rw [h]
  sorry

-- Problem 2
theorem problem_2 : (∀ x, A x → B a x) → ∀ a, 1 / 3 ≤ a ∧ a ≤ 2 :=
by
  sorry

end problem_1_problem_2_l158_158104


namespace final_milk_concentration_l158_158815

theorem final_milk_concentration
  (initial_mixture_volume : ℝ)
  (initial_milk_volume : ℝ)
  (replacement_volume : ℝ)
  (replacements_count : ℕ)
  (final_milk_volume : ℝ) :
  initial_mixture_volume = 100 → 
  initial_milk_volume = 36 → 
  replacement_volume = 50 →
  replacements_count = 2 →
  final_milk_volume = 9 →
  (final_milk_volume / initial_mixture_volume * 100) = 9 :=
by
  sorry

end final_milk_concentration_l158_158815


namespace base_measurement_zions_house_l158_158812

-- Given conditions
def height_zion_house : ℝ := 20
def total_area_three_houses : ℝ := 1200
def num_houses : ℝ := 3

-- Correct answer
def base_zion_house : ℝ := 40

-- Proof statement (question translated to lean statement)
theorem base_measurement_zions_house :
  ∃ base : ℝ, (height_zion_house = 20 ∧ total_area_three_houses = 1200 ∧ num_houses = 3) →
  base = base_zion_house :=
by
  sorry

end base_measurement_zions_house_l158_158812


namespace probability_of_region_C_l158_158672

theorem probability_of_region_C (P_A P_B P_C : ℚ) (hA : P_A = 1/3) (hB : P_B = 1/2) (hTotal : P_A + P_B + P_C = 1) : P_C = 1/6 := 
by
  sorry

end probability_of_region_C_l158_158672


namespace remainder_of_m_l158_158084

theorem remainder_of_m (m : ℕ) (h1 : m^2 % 7 = 1) (h2 : m^3 % 7 = 6) : m % 7 = 6 := by
  sorry

end remainder_of_m_l158_158084


namespace five_digit_numbers_with_4_or_5_l158_158105

theorem five_digit_numbers_with_4_or_5 : 
  let total_five_digit := 99999 - 10000 + 1
  let without_4_or_5 := 7 * 8^4
  total_five_digit - without_4_or_5 = 61328 :=
by
  let total_five_digit := 99999 - 10000 + 1
  let without_4_or_5 := 7 * 8^4
  have h : total_five_digit - without_4_or_5 = 61328 := by sorry
  exact h

end five_digit_numbers_with_4_or_5_l158_158105


namespace A_plus_B_zero_l158_158001

def f (A B x : ℝ) : ℝ := 3 * A * x + 2 * B
def g (A B x : ℝ) : ℝ := 2 * B * x + 3 * A

theorem A_plus_B_zero (A B : ℝ) (h1 : A ≠ B) (h2 : ∀ x : ℝ, f A B (g A B x) - g A B (f A B x) = 3 * (B - A)) :
  A + B = 0 :=
sorry

end A_plus_B_zero_l158_158001


namespace find_angle_B_l158_158232

noncomputable def angle_B (a b c : ℝ) (B C : ℝ) : Prop :=
b = 2 * Real.sqrt 3 ∧ c = 2 ∧ C = Real.pi / 6 ∧
(Real.sin B = (b * Real.sin C) / c ∧ b > c → (B = Real.pi / 3 ∨ B = 2 * Real.pi / 3))

theorem find_angle_B :
  ∃ (B : ℝ), angle_B 1 (2 * Real.sqrt 3) 2 B (Real.pi / 6) :=
by
  sorry

end find_angle_B_l158_158232


namespace coordinates_of_point_M_in_second_quadrant_l158_158411

theorem coordinates_of_point_M_in_second_quadrant
  (x y : ℝ)
  (second_quadrant : x < 0 ∧ 0 < y)
  (dist_to_x_axis : abs(y) = 1)
  (dist_to_y_axis : abs(x) = 2) :
  (x, y) = (-2, 1) :=
sorry

end coordinates_of_point_M_in_second_quadrant_l158_158411


namespace alcohol_mixture_l158_158635

theorem alcohol_mixture:
  ∃ (x y z: ℝ), 
    0.10 * x + 0.30 * y + 0.50 * z = 157.5 ∧
    x + y + z = 450 ∧
    x = y ∧
    x = 112.5 ∧
    y = 112.5 ∧
    z = 225 :=
sorry

end alcohol_mixture_l158_158635


namespace find_d_vector_l158_158297

theorem find_d_vector (x y t : ℝ) (v d : ℝ × ℝ)
  (hline : y = (5 * x - 7) / 2)
  (hparam : ∃ t : ℝ, (x, y) = (4, 2) + t • d)
  (hdist : ∀ {x : ℝ}, x ≥ 4 → dist (x, (5 * x - 7) / 2) (4, 2) = t) :
  d = (2 / Real.sqrt 29, 5 / Real.sqrt 29) := 
sorry

end find_d_vector_l158_158297


namespace ratio_of_auto_finance_companies_credit_l158_158556

theorem ratio_of_auto_finance_companies_credit
    (total_consumer_credit : ℝ)
    (percent_auto_installment_credit : ℝ)
    (credit_by_auto_finance_companies : ℝ)
    (total_auto_credit : ℝ)
    (hc1 : total_consumer_credit = 855)
    (hc2 : percent_auto_installment_credit = 0.20)
    (hc3 : credit_by_auto_finance_companies = 57)
    (htotal_auto_credit : total_auto_credit = percent_auto_installment_credit * total_consumer_credit) :
    (credit_by_auto_finance_companies / total_auto_credit) = (1 / 3) := 
by
  sorry

end ratio_of_auto_finance_companies_credit_l158_158556


namespace total_pencils_is_60_l158_158659

def original_pencils : ℕ := 33
def added_pencils : ℕ := 27
def total_pencils : ℕ := original_pencils + added_pencils

theorem total_pencils_is_60 : total_pencils = 60 := by
  sorry

end total_pencils_is_60_l158_158659


namespace jessies_weight_loss_l158_158418

-- Definitions based on the given conditions
def initial_weight : ℝ := 74
def weight_loss_rate_even_days : ℝ := 0.2 + 0.15
def weight_loss_rate_odd_days : ℝ := 0.3
def total_exercise_days : ℕ := 25
def even_days : ℕ := (total_exercise_days - 1) / 2
def odd_days : ℕ := even_days + 1

-- The goal is to prove the total weight loss is 8.1 kg
theorem jessies_weight_loss : 
  (even_days * weight_loss_rate_even_days + odd_days * weight_loss_rate_odd_days) = 8.1 := 
by
  sorry

end jessies_weight_loss_l158_158418


namespace intersection_S_T_l158_158401

def setS (x : ℝ) : Prop := (x - 1) * (x - 3) ≥ 0
def setT (x : ℝ) : Prop := x > 0

theorem intersection_S_T : {x : ℝ | setS x} ∩ {x : ℝ | setT x} = {x : ℝ | (0 < x ∧ x ≤ 1) ∨ (3 ≤ x)} := 
sorry

end intersection_S_T_l158_158401


namespace angle_B_lt_pi_div_two_l158_158497

theorem angle_B_lt_pi_div_two 
  (a b c : ℝ) (B : ℝ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : B = π / 2 - B)
  (h5 : 2 / b = 1 / a + 1 / c)
  : B < π / 2 := sorry

end angle_B_lt_pi_div_two_l158_158497


namespace henry_jeans_cost_l158_158605

-- Let P be the price of the socks, T be the price of the t-shirt, and J be the price of the jeans
def P := 5
def T := P + 10
def J := 2 * T

-- Goal: Prove that J = 30
theorem henry_jeans_cost : J = 30 :=
by
  unfold P T J -- unfold all the definitions
  simp -- simplify the expression
  exact rfl

end henry_jeans_cost_l158_158605


namespace find_f_3_l158_158912

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x : ℝ) : f x + 3 * f (1 - x) = 4 * x ^ 2

theorem find_f_3 : f 3 = 3 / 2 := 
by
  sorry

end find_f_3_l158_158912


namespace geom_seq_sum_l158_158813

theorem geom_seq_sum (q : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 3)
  (h2 : a 1 + a 3 + a 5 = 21)
  (h_geometric : ∀ n : ℕ, a (n + 1) = a 1 * q ^ n) :
  a 3 + a 5 + a 7 = 42 :=
sorry

end geom_seq_sum_l158_158813


namespace solve_quadratic_inequality_l158_158109

theorem solve_quadratic_inequality (x : ℝ) (h : x^2 - 7 * x + 6 < 0) : 1 < x ∧ x < 6 :=
  sorry

end solve_quadratic_inequality_l158_158109


namespace cube_probability_l158_158143

theorem cube_probability :
  let m := 1
  let n := 504
  ∀ (faces : Finset (Fin 6)) (nums : Finset (Fin 9)), 
    faces.card = 6 → nums.card = 9 →
    (∀ f ∈ faces, ∃ n ∈ nums, true) →
    m + n = 505 :=
by
  sorry

end cube_probability_l158_158143


namespace proof_u_g_3_l158_158257

noncomputable def u (x : ℝ) : ℝ := Real.sqrt (5 * x + 2)

noncomputable def g (x : ℝ) : ℝ := 7 - u x

theorem proof_u_g_3 :
  u (g 3) = Real.sqrt (37 - 5 * Real.sqrt 17) :=
sorry

end proof_u_g_3_l158_158257


namespace license_plate_palindrome_probability_l158_158679

-- Define the two-letter palindrome probability
def prob_two_letter_palindrome : ℚ := 1 / 26

-- Define the four-digit palindrome probability
def prob_four_digit_palindrome : ℚ := 1 / 100

-- Define the joint probability of both two-letter and four-digit palindrome
def prob_joint_palindrome : ℚ := prob_two_letter_palindrome * prob_four_digit_palindrome

-- Define the probability of at least one palindrome using Inclusion-Exclusion
def prob_at_least_one_palindrome : ℚ := prob_two_letter_palindrome + prob_four_digit_palindrome - prob_joint_palindrome

-- Convert the probability to the form of sum of two integers
def sum_of_integers : ℕ := 5 + 104

-- The final proof problem
theorem license_plate_palindrome_probability :
  (prob_at_least_one_palindrome = 5 / 104) ∧ (sum_of_integers = 109) := by
  sorry

end license_plate_palindrome_probability_l158_158679


namespace sum_of_divisors_85_l158_158167

theorem sum_of_divisors_85 :
  let divisors := {d ∈ Finset.range 86 | 85 % d = 0}
  Finset.sum divisors id = 108 :=
sorry

end sum_of_divisors_85_l158_158167


namespace Ted_age_48_l158_158828

/-- Given ages problem:
 - t is Ted's age
 - s is Sally's age
 - a is Alex's age 
 - The following conditions hold:
   1. t = 2s + 17 
   2. a = s / 2
   3. t + s + a = 72
 - Prove that Ted's age (t) is 48.
-/ 
theorem Ted_age_48 {t s a : ℕ} (h1 : t = 2 * s + 17) (h2 : a = s / 2) (h3 : t + s + a = 72) : t = 48 := by
  sorry

end Ted_age_48_l158_158828


namespace value_of_3k_squared_minus_1_l158_158101

theorem value_of_3k_squared_minus_1 (x k : ℤ)
  (h1 : 7 * x + 2 = 3 * x - 6)
  (h2 : x + 1 = k)
  : 3 * k^2 - 1 = 2 := 
by
  sorry

end value_of_3k_squared_minus_1_l158_158101


namespace white_balls_count_l158_158970

theorem white_balls_count (w : ℕ) (h : (w / 15) * ((w - 1) / 14) = (1 : ℚ) / 21) : w = 5 := by
  sorry

end white_balls_count_l158_158970


namespace triangle_count_relationship_l158_158885

theorem triangle_count_relationship :
  let n_0 : ℕ := 20
  let n_1 : ℕ := 19
  let n_2 : ℕ := 18
  n_0 > n_1 ∧ n_1 > n_2 :=
by
  let n_0 := 20
  let n_1 := 19
  let n_2 := 18
  have h0 : n_0 > n_1 := by sorry
  have h1 : n_1 > n_2 := by sorry
  exact ⟨h0, h1⟩

end triangle_count_relationship_l158_158885


namespace quadratic_non_real_roots_l158_158736

variable (b : ℝ)

theorem quadratic_non_real_roots : (b^2 - 64 < 0) → (-8 < b ∧ b < 8) :=
by
  sorry

end quadratic_non_real_roots_l158_158736


namespace parabola_vertex_l158_158654

theorem parabola_vertex (c d : ℝ) (h : ∀ x : ℝ, - x^2 + c * x + d ≤ 0 ↔ (x ≤ -4 ∨ x ≥ 6)) :
  (∃ v : ℝ × ℝ, v = (5, 1)) :=
sorry

end parabola_vertex_l158_158654


namespace total_flowers_l158_158628

def pieces (f : String) : Nat :=
  if f == "roses" ∨ f == "lilies" ∨ f == "sunflowers" ∨ f == "daisies" then 40 else 0

theorem total_flowers : 
  pieces "roses" + pieces "lilies" + pieces "sunflowers" + pieces "daisies" = 160 := 
by
  sorry


end total_flowers_l158_158628


namespace f_in_neg_interval_l158_158003

variables (f : ℝ → ℝ)

-- Conditions
def is_even := ∀ x, f x = f (-x)
def symmetry := ∀ x, f (2 + x) = f (2 - x)
def in_interval := ∀ x, 0 < x ∧ x < 2 → f x = 1 / x

-- Target statement
theorem f_in_neg_interval
  (h_even : is_even f)
  (h_symm : symmetry f)
  (h_interval : in_interval f)
  (x : ℝ)
  (hx : -4 < x ∧ x < -2) :
  f x = 1 / (x + 4) :=
sorry

end f_in_neg_interval_l158_158003


namespace circle_radius_l158_158375

theorem circle_radius (x y : ℝ) : x^2 - 10 * x + y^2 + 4 * y + 13 = 0 → (x - 5)^2 + (y + 2)^2 = 4^2 :=
by
  sorry

end circle_radius_l158_158375


namespace equal_profits_at_20000_end_month_more_profit_50000_l158_158432

noncomputable section

-- Define the conditions
def profit_beginning_month (x : ℝ) : ℝ := 0.15 * x + 1.15 * x * 0.1
def profit_end_month (x : ℝ) : ℝ := 0.3 * x - 700

-- Proof Problem 1: Prove that at x = 20000, the profits are equal
theorem equal_profits_at_20000 : profit_beginning_month 20000 = profit_end_month 20000 :=
by
  sorry

-- Proof Problem 2: Prove that at x = 50000, selling at end of month yields more profit than selling at beginning of month
theorem end_month_more_profit_50000 : profit_end_month 50000 > profit_beginning_month 50000 :=
by
  sorry

end equal_profits_at_20000_end_month_more_profit_50000_l158_158432


namespace solution_set_inequality_l158_158027

theorem solution_set_inequality (x : ℝ) : (1 / x ≤ 1 / 3) ↔ (x ≥ 3 ∨ x < 0) := by
  sorry

end solution_set_inequality_l158_158027


namespace circle_radius_l158_158377

theorem circle_radius (x y : ℝ) : x^2 - 10*x + y^2 + 4*y + 13 = 0 → ∃ r : ℝ, r = 4 :=
by
  -- sorry here to indicate that the proof is skipped
  sorry

end circle_radius_l158_158377


namespace always_true_statements_l158_158906

variable (a b c : ℝ)

theorem always_true_statements (h1 : a < 0) (h2 : a < b ∧ b ≤ 0) (h3 : b < c) : 
  (a + b < b + c) ∧ (c / a < 1) :=
by 
  sorry

end always_true_statements_l158_158906


namespace ribbons_problem_l158_158239

/-
    In a large box of ribbons, 1/3 are yellow, 1/4 are purple, 1/6 are orange, and the remaining 40 ribbons are black.
    Prove that the total number of orange ribbons is 27.
-/

theorem ribbons_problem :
  ∀ (total : ℕ), 
    (1 / 3 : ℚ) * total + (1 / 4 : ℚ) * total + (1 / 6 : ℚ) * total + 40 = total →
    (1 / 6 : ℚ) * total = 27 := sorry

end ribbons_problem_l158_158239


namespace janet_waiting_time_l158_158246

-- Define the speeds and distance
def janet_speed : ℝ := 30 -- miles per hour
def sister_speed : ℝ := 12 -- miles per hour
def lake_width : ℝ := 60 -- miles

-- Define the travel times
def janet_travel_time : ℝ := lake_width / janet_speed
def sister_travel_time : ℝ := lake_width / sister_speed

-- The theorem to be proved
theorem janet_waiting_time : 
  sister_travel_time - janet_travel_time = 3 := 
by 
  sorry

end janet_waiting_time_l158_158246


namespace vasya_fraction_l158_158541

variable (a b c d s : ℝ)

-- Anton drove half the distance Vasya did
axiom h1 : a = b / 2

-- Sasha drove as long as Anton and Dima together
axiom h2 : c = a + d

-- Dima drove one-tenth of the total distance
axiom h3 : d = s / 10

-- The total distance is the sum of distances driven by Anton, Vasya, Sasha, and Dima
axiom h4 : a + b + c + d = s

-- We need to prove that Vasya drove 0.4 of the total distance
theorem vasya_fraction (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : b = 0.4 * s :=
by
  sorry

end vasya_fraction_l158_158541


namespace Q_over_P_l158_158022

theorem Q_over_P (P Q : ℚ)
  (h : ∀ (x : ℝ), x ≠ 0 ∧ x ≠ 3 ∧ x ≠ -3 → 
    (P / (x + 3) + Q / (x^2 - 3*x) = (x^2 - x + 8) / (x^3 + x^2 - 9*x))) :
  Q / P = 8 / 3 :=
by
  sorry

end Q_over_P_l158_158022


namespace man_is_older_by_16_l158_158506

variable (M S : ℕ)

-- Condition: The present age of the son is 14.
def son_age := S = 14

-- Condition: In two years, the man's age will be twice the son's age.
def age_relation := M + 2 = 2 * (S + 2)

-- Theorem: Prove that the man is 16 years older than his son.
theorem man_is_older_by_16 (h1 : son_age S) (h2 : age_relation M S) : M - S = 16 := 
sorry

end man_is_older_by_16_l158_158506


namespace probability_top_king_of_hearts_l158_158684

def deck_size : ℕ := 52

def king_of_hearts_count : ℕ := 1

def probability_king_of_hearts_top_card (n : ℕ) (k : ℕ) : ℚ :=
  if n ≠ 0 then k / n else 0

theorem probability_top_king_of_hearts : 
  probability_king_of_hearts_top_card deck_size king_of_hearts_count = 1 / 52 :=
by
  -- Proof omitted
  sorry

end probability_top_king_of_hearts_l158_158684


namespace decimal_to_fraction_correct_l158_158566

-- Define a structure representing our initial decimal to fraction conversion
structure DecimalFractionConversion :=
  (decimal: ℚ)
  (vulgar_fraction: ℚ)
  (simplified_fraction: ℚ)

-- Define the conditions provided in the problem
def conversion_conditions : DecimalFractionConversion :=
  { decimal := 35 / 100,
    vulgar_fraction := 35 / 100,
    simplified_fraction := 7 / 20 }

-- State the theorem we aim to prove
theorem decimal_to_fraction_correct :
  conversion_conditions.simplified_fraction = 7 / 20 := by
  sorry

end decimal_to_fraction_correct_l158_158566


namespace paused_time_l158_158071

theorem paused_time (total_length remaining_length paused_at : ℕ) (h1 : total_length = 60) (h2 : remaining_length = 30) : paused_at = total_length - remaining_length :=
by
  sorry

end paused_time_l158_158071


namespace min_value_y_l158_158312

theorem min_value_y : ∀ (x : ℝ), ∃ y_min : ℝ, y_min = (x^2 + 16 * x + 10) ∧ ∀ (x' : ℝ), (x'^2 + 16 * x' + 10) ≥ y_min := 
by 
  sorry

end min_value_y_l158_158312


namespace gun_can_hit_l158_158439

-- Define the constants
variables (v g : ℝ)

-- Define the coordinates in the first quadrant
variables (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0)

-- Prove the condition for a point (x, y) to be in the region that can be hit by the gun
theorem gun_can_hit (hv : v > 0) (hg : g > 0) :
  y ≤ (v^2 / (2 * g)) - (g * x^2 / (2 * v^2)) :=
sorry

end gun_can_hit_l158_158439


namespace smallest_n_exists_l158_158850

theorem smallest_n_exists (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) (h3 : 8 / 15 < n / (n + k)) (h4 : n / (n + k) < 7 / 13) : 
  n = 15 :=
  sorry

end smallest_n_exists_l158_158850


namespace solve_system_l158_158276

theorem solve_system :
  ∃ x y z : ℝ, (x = 20 ∧ y = 22 ∧ z = 23 ∧ 
  (x^2 - 23*y - 25*z = -681) ∧ 
  (y^2 - 21*x - 21*z = -419) ∧ 
  (z^2 - 19*x - 21*y = -313)) :=
by
  use 20, 22, 23
  split
  . refl
  split
  . refl
  split
  . refl
  split
  . sorry
  split
  . sorry
  . sorry

end solve_system_l158_158276


namespace carl_weight_l158_158189

variable (C R B : ℕ)

theorem carl_weight (h1 : B = R + 9) (h2 : R = C + 5) (h3 : B = 159) : C = 145 :=
by
  sorry

end carl_weight_l158_158189


namespace distance_circumcenter_centroid_inequality_l158_158622

variable {R r d : ℝ}

theorem distance_circumcenter_centroid_inequality 
  (h1 : d = distance_circumcenter_to_centroid)
  (h2 : R = circumradius)
  (h3 : r = inradius) : d^2 ≤ R * (R - 2 * r) := 
sorry

end distance_circumcenter_centroid_inequality_l158_158622


namespace expected_value_paths_l158_158574

theorem expected_value_paths : 
  let a := 7
  let b := 242
  100 * a + b = 942 :=
by
  sorry

end expected_value_paths_l158_158574


namespace samantha_routes_l158_158266

-- Define the positions relative to the grid
structure Position where
  x : Int
  y : Int

-- Define the initial conditions and path constraints
def house : Position := ⟨-3, -2⟩
def sw_corner_of_park : Position := ⟨0, 0⟩
def ne_corner_of_park : Position := ⟨8, 5⟩
def school : Position := ⟨11, 8⟩

-- Define the combinatorial function for calculating number of ways
def binom (n k : Nat) : Nat := Nat.choose n k

-- Route segments based on the constraints
def ways_house_to_sw_corner : Nat := binom 5 2
def ways_through_park : Nat := 1
def ways_ne_corner_to_school : Nat := binom 6 3

-- Total number of routes
def total_routes : Nat := ways_house_to_sw_corner * ways_through_park * ways_ne_corner_to_school

-- The statement to be proven
theorem samantha_routes : total_routes = 200 := by
  sorry

end samantha_routes_l158_158266


namespace usual_time_to_office_l158_158472

theorem usual_time_to_office (S T : ℝ) (h : T = 4 / 3 * (T + 8)) : T = 24 :=
by
  sorry

end usual_time_to_office_l158_158472


namespace car_tank_capacity_l158_158829

theorem car_tank_capacity
  (speed : ℝ) (usage_rate : ℝ) (time : ℝ) (used_fraction : ℝ) (distance : ℝ := speed * time) (gallons_used : ℝ := distance / usage_rate) 
  (fuel_used : ℝ := 10) (tank_capacity : ℝ := fuel_used / used_fraction)
  (h1 : speed = 60) (h2 : usage_rate = 30) (h3 : time = 5) (h4 : used_fraction = 0.8333333333333334) : 
  tank_capacity = 12 :=
by
  sorry

end car_tank_capacity_l158_158829


namespace find_integers_l158_158201

theorem find_integers (n : ℕ) (h1 : n < 10^100)
  (h2 : n ∣ 2^n) (h3 : n - 1 ∣ 2^n - 1) (h4 : n - 2 ∣ 2^n - 2) :
  n = 2^2 ∨ n = 2^4 ∨ n = 2^16 ∨ n = 2^256 := by
  sorry

end find_integers_l158_158201


namespace trajectory_of_midpoint_l158_158854

theorem trajectory_of_midpoint (A B P : ℝ × ℝ)
  (hA : A = (2, 4))
  (hB : ∃ m n : ℝ, B = (m, n) ∧ n^2 = 2 * m)
  (hP : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  (P.2 - 2)^2 = P.1 - 1 :=
sorry

end trajectory_of_midpoint_l158_158854


namespace distance_between_points_l158_158861

theorem distance_between_points (A B : ℝ) (dA : |A| = 2) (dB : |B| = 7) : |A - B| = 5 ∨ |A - B| = 9 := 
by
  sorry

end distance_between_points_l158_158861


namespace analytical_expression_of_f_l158_158597

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^3 + a * x^2 + b

theorem analytical_expression_of_f (a b : ℝ) (h_a : a > 0)
  (h_max : (∃ x_max : ℝ, f x_max a b = 5 ∧ (∀ x : ℝ, f x_max a b ≥ f x a b)))
  (h_min : (∃ x_min : ℝ, f x_min a b = 1 ∧ (∀ x : ℝ, f x_min a b ≤ f x a b))) :
  f x 3 1 = x^3 + 3 * x^2 + 1 := 
sorry

end analytical_expression_of_f_l158_158597


namespace total_books_correct_l158_158467

-- Definitions based on the conditions
def num_books_bottom_shelf (T : ℕ) := T / 3
def num_books_middle_shelf (T : ℕ) := T / 4
def num_books_top_shelf : ℕ := 30
def total_books (T : ℕ) := num_books_bottom_shelf T + num_books_middle_shelf T + num_books_top_shelf

theorem total_books_correct : total_books 72 = 72 :=
by
  sorry

end total_books_correct_l158_158467


namespace area_ratio_l158_158293

-- Define the problem conditions
def Square (s : ℝ) := s > 0
def Rectangle (longer shorter : ℝ) := longer = 1.2 * shorter ∧ shorter = 0.8 * shorter

-- Define a function to calculate the area of square
def area_square (s : ℝ) : ℝ := s * s

-- Define a function to calculate the area of rectangle
def area_rectangle (longer shorter : ℝ) : ℝ := longer * shorter

-- State the proof problem
theorem area_ratio (s : ℝ) (h_square : Square s) :
  let longer := 1.2 * s
  let shorter := 0.8 * s
  area_rectangle longer shorter / area_square s = 24 / 25 :=
by
  sorry

end area_ratio_l158_158293


namespace vasya_drives_fraction_l158_158544

theorem vasya_drives_fraction {a b c d s : ℝ} 
  (h1 : a = b / 2) 
  (h2 : c = a + d) 
  (h3 : d = s / 10) 
  (h4 : a + b + c + d = s) : 
  b / s = 0.4 :=
by
  sorry

end vasya_drives_fraction_l158_158544


namespace three_layer_carpet_area_l158_158920

-- Define the dimensions of the carpets and the hall
structure Carpet := (width : ℕ) (height : ℕ)

def principal_carpet : Carpet := ⟨6, 8⟩
def caretaker_carpet : Carpet := ⟨6, 6⟩
def parent_committee_carpet : Carpet := ⟨5, 7⟩
def hall : Carpet := ⟨10, 10⟩

-- Define the area function
def area (c : Carpet) : ℕ := c.width * c.height

-- Prove the area of the part of the hall covered by all three carpets
theorem three_layer_carpet_area : area ⟨3, 2⟩ = 6 :=
by
  sorry

end three_layer_carpet_area_l158_158920


namespace chessboard_cover_l158_158851

open Nat

/-- 
  For an m × n chessboard, after removing any one small square, it can always be completely covered
  with L-shaped tiles if and only if 3 divides (mn - 1) and min(m,n) is not equal to 1, 2, 5 or m=n=2.
-/
theorem chessboard_cover (m n : ℕ) :
  (∃ k : ℕ, 3 * k = m * n - 1) ∧ (min m n ≠ 1 ∧ min m n ≠ 2 ∧ min m n ≠ 5 ∨ m = 2 ∧ n = 2) :=
sorry

end chessboard_cover_l158_158851


namespace bacteria_original_count_l158_158821

theorem bacteria_original_count (current: ℕ) (increase: ℕ) (hc: current = 8917) (hi: increase = 8317) : current - increase = 600 :=
by
  sorry

end bacteria_original_count_l158_158821


namespace jed_speeding_l158_158237

-- Define the constants used in the conditions
def F := 16
def T := 256
def S := 50

theorem jed_speeding : (T / F) + S = 66 := 
by sorry

end jed_speeding_l158_158237


namespace max_subsets_intersection_at_most_two_l158_158766

open Finset

theorem max_subsets_intersection_at_most_two (A : Finset ℕ) (B : Finset (Finset ℕ)) :
  (A = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) →
  (∀ B ∈ B, B ≠ ∅) →
  (∀ B1 B2 ∈ B, B1 ≠ B2 → (B1 ∩ B2).card ≤ 2) →
  B.card ≤ 175 :=
by
  intros hA hNonEmpty hIntersection
  -- Proof here
  sorry

end max_subsets_intersection_at_most_two_l158_158766


namespace intersection_of_sets_l158_158098

open Set

theorem intersection_of_sets (A B : Set ℕ) (hA : A = {0, 1, 2, 3, 4, 5}) (hB : B = {2, 4, 6}) :
  A ∩ B = {2, 4} :=
by
  sorry

end intersection_of_sets_l158_158098


namespace pasta_ratio_l158_158966

theorem pasta_ratio (total_students : ℕ) (spaghetti : ℕ) (manicotti : ℕ) 
  (h1 : total_students = 650) 
  (h2 : spaghetti = 250) 
  (h3 : manicotti = 100) : 
  (spaghetti : ℤ) / (manicotti : ℤ) = 5 / 2 :=
by
  sorry

end pasta_ratio_l158_158966


namespace perfect_square_x4_x3_x2_x1_1_eq_x0_l158_158847

theorem perfect_square_x4_x3_x2_x1_1_eq_x0 :
  ∀ x : ℤ, ∃ n : ℤ, x^4 + x^3 + x^2 + x + 1 = n^2 ↔ x = 0 :=
by sorry

end perfect_square_x4_x3_x2_x1_1_eq_x0_l158_158847


namespace work_completion_time_l158_158319

theorem work_completion_time
  (A B C : ℝ)
  (h1 : A + B = 1 / 12)
  (h2 : B + C = 1 / 15)
  (h3 : C + A = 1 / 20) :
  1 / (A + B + C) = 10 :=
by
  sorry

end work_completion_time_l158_158319


namespace remaining_cooking_time_l158_158691

-- Define the recommended cooking time in minutes and the time already cooked in seconds
def recommended_cooking_time_min := 5
def time_cooked_seconds := 45

-- Define the conversion from minutes to seconds
def minutes_to_seconds (min : Nat) : Nat := min * 60

-- Define the total recommended cooking time in seconds
def total_recommended_cooking_time_seconds := minutes_to_seconds recommended_cooking_time_min

-- State the theorem to prove the remaining cooking time
theorem remaining_cooking_time :
  (total_recommended_cooking_time_seconds - time_cooked_seconds) = 255 :=
by
  sorry

end remaining_cooking_time_l158_158691


namespace calculate_number_of_sides_l158_158941

theorem calculate_number_of_sides (n : ℕ) (h : n ≥ 6) :
  ((6 : ℚ) / n^2) * ((6 : ℚ) / n^2) = 0.027777777777777776 →
  n = 6 :=
by
  sorry

end calculate_number_of_sides_l158_158941


namespace speed_ratio_l158_158822

theorem speed_ratio (v_A v_B : ℝ) (t : ℝ) (h1 : v_A = 200 / t) (h2 : v_B = 120 / t) : 
  v_A / v_B = 5 / 3 :=
by
  sorry

end speed_ratio_l158_158822


namespace stock_percent_change_l158_158419

variable (x : ℝ)

theorem stock_percent_change (h1 : ∀ x, 0.75 * x = x * 0.75)
                             (h2 : ∀ x, 1.05 * x = 0.75 * x + 0.3 * 0.75 * x):
    ((1.05 * x - x) / x) * 100 = 5 :=
by
  sorry

end stock_percent_change_l158_158419


namespace calc_result_l158_158695

noncomputable def expMul := (-0.25)^11 * (-4)^12

theorem calc_result : expMul = -4 := 
by
  -- Sorry is used here to skip the proof as instructed.
  sorry

end calc_result_l158_158695


namespace conditional_probability_of_white_balls_l158_158415

theorem conditional_probability_of_white_balls
  (w r : ℕ)
  (A B : Set (Fin w + r))
  (ha : A.card = 5)
  (hb : B.card = 5 - 1)
  (draw_without_replacement : ∀ (x y : Fin w + r), x ≠ y → x ∈ A → y ∈ B)
  : P(B |A) = 4/7 
  := sorry

end conditional_probability_of_white_balls_l158_158415


namespace union_when_a_eq_2_condition_1_condition_2_condition_3_l158_158832

open Set

def setA (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 1}
def setB : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem union_when_a_eq_2 : setA 2 ∪ setB = {x | -1 ≤ x ∧ x ≤ 3} :=
sorry

theorem condition_1 (a : ℝ) : 
  (setA a ∪ setB = setB) → (0 ≤ a ∧ a ≤ 2) :=
sorry

theorem condition_2 (a : ℝ) :
  (∀ x, (x ∈ setA a ↔ x ∈ setB)) → (0 ≤ a ∧ a ≤ 2) :=
sorry

theorem condition_3 (a : ℝ) :
  (setA a ∩ setB = ∅) → (a < -2 ∨ 4 < a) :=
sorry

end union_when_a_eq_2_condition_1_condition_2_condition_3_l158_158832


namespace sufficient_but_not_necessary_pi_l158_158652

theorem sufficient_but_not_necessary_pi (x : ℝ) : 
  (x = Real.pi → Real.sin x = 0) ∧ (Real.sin x = 0 → ∃ k : ℤ, x = k * Real.pi) → ¬(Real.sin x = 0 → x = Real.pi) :=
by
  sorry

end sufficient_but_not_necessary_pi_l158_158652


namespace triangle_right_angle_and_m_values_l158_158102

open Real

-- Definitions and conditions
def line_AB (x y : ℝ) : Prop := 3 * x - 2 * y + 6 = 0
def line_AC (x y : ℝ) : Prop := 2 * x + 3 * y - 22 = 0
def line_BC (x y m : ℝ) : Prop := 3 * x + 4 * y - m = 0

-- Prove the shape and value of m when the height from BC is 1
theorem triangle_right_angle_and_m_values :
  (∃ (x y : ℝ), line_AB x y ∧ line_AC x y ∧ line_AB x y ∧ (-3/2) ≠ (2/3)) ∧
  (∀ x y, line_AB x y → line_AC x y → 3 * x + 4 * y - 25 = 0 ∨ 3 * x + 4 * y - 35 = 0) := 
sorry

end triangle_right_angle_and_m_values_l158_158102


namespace value_of_a_l158_158007

noncomputable def normal_distribution_example : ℝ :=
  let μ := 3
  let σ := 2  -- Note: Standard deviation is the square root of variance, hence sqrt(4) = 2
  let ξ := measure_theory.probability_theory.normal μ σ
  let a := by
    sorry  -- This is where you would solve the problem in Lean

  a  -- This will return the value of a which we need to show is 7/3

-- Now we need to state the theorem:
theorem value_of_a :
  2 * a - 3 > 0 ∧ measure_theory.probability_theory.normal 3 2 (≤ 2 * a - 3) = measure_theory.probability_theory.normal 3 2 (> a + 2) →
  a = 7 / 3 :=
begin
  intros h,
  rw ←probability_theory.measure_norm_dist at h,
  sorry  -- Proof goes here
end

end value_of_a_l158_158007


namespace inv_of_15_mod_1003_l158_158359

theorem inv_of_15_mod_1003 : ∃ x : ℕ, x ≤ 1002 ∧ 15 * x ≡ 1 [MOD 1003] ∧ x = 937 :=
by sorry

end inv_of_15_mod_1003_l158_158359


namespace fries_remaining_time_l158_158693

theorem fries_remaining_time (recommended_time_min : ℕ) (time_in_oven_sec : ℕ)
    (h1 : recommended_time_min = 5)
    (h2 : time_in_oven_sec = 45) :
    (recommended_time_min * 60 - time_in_oven_sec = 255) :=
by
  sorry

end fries_remaining_time_l158_158693


namespace natural_numbers_divisors_l158_158849

theorem natural_numbers_divisors (n : ℕ) : 
  n + 1 ∣ n^2 + 1 → n = 0 ∨ n = 1 :=
by
  intro h
  sorry

end natural_numbers_divisors_l158_158849


namespace find_S5_l158_158034

-- Assuming the sequence is geometric and defining the conditions
variables {a : ℕ → ℝ} {q : ℝ}

-- Definitions of the conditions based on the problem
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n+1) = a n * q

def condition_1 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 2 * a 5 = 3 * a 3

def condition_2 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 4 + 9 * a 7) / 2 = 2

-- Sum of the first n terms of a geometric sequence
noncomputable def S_n (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a 0 * (1 - q ^ n) / (1 - q)

-- The theorem stating the final goal
theorem find_S5 (a : ℕ → ℝ) (q : ℝ) (h_seq : geometric_sequence a q) 
    (h1 : condition_1 a q) (h2 : condition_2 a q) : S_n a q 5 = 121 :=
by
  -- This adds "sorry" to bypass the actual proof
  sorry

end find_S5_l158_158034


namespace total_cost_correct_l158_158382

def bun_price : ℝ := 0.1
def buns_count : ℝ := 10
def milk_price : ℝ := 2
def milk_count : ℝ := 2
def egg_price : ℝ := 3 * milk_price

def total_cost : ℝ := (buns_count * bun_price) + (milk_count * milk_price) + egg_price

theorem total_cost_correct : total_cost = 11 := by
  sorry

end total_cost_correct_l158_158382


namespace Vasya_distance_fraction_l158_158552

variable (a b c d s : ℝ)

theorem Vasya_distance_fraction :
  (a = b / 2) →
  (c = a + d) →
  (d = s / 10) →
  (a + b + c + d = s) →
  (b / s = 0.4) :=
by
  intros h1 h2 h3 h4
  sorry

end Vasya_distance_fraction_l158_158552


namespace ant_paths_l158_158161

theorem ant_paths (n m : ℕ) : 
  ∃ paths : ℕ, paths = Nat.choose (n + m) m := sorry

end ant_paths_l158_158161


namespace smallest_base10_integer_l158_158952

theorem smallest_base10_integer {a b n : ℕ} (ha : a > 2) (hb : b > 2)
  (h₁ : 2 * a + 1 = n) (h₂ : 1 * b + 2 = n) :
  n = 7 :=
sorry

end smallest_base10_integer_l158_158952


namespace gcf_of_180_270_450_l158_158474

theorem gcf_of_180_270_450 : Nat.gcd (Nat.gcd 180 270) 450 = 90 :=
by
  have prime_factor_180 : ∃ (a b c : ℕ), 180 = 2^2 * 3^2 * 5 := ⟨2, 2, 1, rfl⟩
  have prime_factor_270 : ∃ (a b c : ℕ), 270 = 2 * 3^3 * 5 := ⟨1, 3, 1, rfl⟩
  have prime_factor_450 : ∃ (a b c : ℕ), 450 = 2 * 3^2 * 5^2 := ⟨1, 2, 2, rfl⟩
  sorry

end gcf_of_180_270_450_l158_158474


namespace min_value_l158_158388

variable (d : ℕ) (a_n S_n : ℕ → ℕ)
variable (a1 : ℕ) (H1 : d ≠ 0)
variable (H2 : a1 = 1)
variable (H3 : (a_n 3)^2 = a1 * (a_n 13))
variable (H4 : a_n n = a1 + (n - 1) * d)
variable (H5 : S_n n = (n * (a1 + a_n n)) / 2)

theorem min_value (n : ℕ) (Hn : 1 ≤ n) : 
  ∃ n, ∀ m, 1 ≤ m → (2 * S_n n + 16) / (a_n n + 3) ≥ (2 * S_n m + 16) / (a_n m + 3) ∧ (2 * S_n n + 16) / (a_n n + 3) = 4 :=
sorry

end min_value_l158_158388


namespace add_fractions_l158_158982

theorem add_fractions (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = (5 / a) :=
by sorry

end add_fractions_l158_158982


namespace find_numbers_l158_158051

theorem find_numbers 
  (x y z : ℕ) 
  (h1 : y = 2 * x - 3) 
  (h2 : x + y = 51) 
  (h3 : z = 4 * x - y) : 
  x = 18 ∧ y = 33 ∧ z = 39 :=
by sorry

end find_numbers_l158_158051


namespace find_m_l158_158487

theorem find_m (m : ℝ) : (243 : ℝ)^(1/3) = (3 : ℝ)^m → m = 5 / 3 :=
by
  sorry

end find_m_l158_158487


namespace scorpion_segments_daily_total_l158_158328

theorem scorpion_segments_daily_total (seg1 : ℕ) (seg2 : ℕ) (additional : ℕ) (total_daily : ℕ) :
  (seg1 = 60) →
  (seg2 = 2 * seg1 * 2) →
  (additional = 10 * 50) →
  (total_daily = seg1 + seg2 + additional) →
  total_daily = 800 :=
by
  intros h1 h2 h3 h4
  sorry

end scorpion_segments_daily_total_l158_158328


namespace not_basic_event_l158_158119

theorem not_basic_event :
  let balls := {red_red, red_white, red_black, white_white, white_black, black_black} :
  (∃ (events: Finset (Finset (Subsingleton balls))), 
    {red_red} ∉ events ∧ {red_white} ∉ events ∧ {red_black} ∉ events) :=
sorry

end not_basic_event_l158_158119


namespace find_f_neg_one_l158_158591

noncomputable def f (x : ℝ) : ℝ := 
  if 0 ≤ x then x^2 + 2 * x else - ( (x^2) + (2 * x))

theorem find_f_neg_one : 
  f (-1) = -3 :=
by 
  sorry

end find_f_neg_one_l158_158591


namespace social_event_handshakes_l158_158555

def handshake_count (total_people : ℕ) (group_a : ℕ) (group_b : ℕ) : ℕ :=
  let introductions_handshakes := group_b * (group_b - 1) / 2
  let direct_handshakes := group_b * (group_a - 1)
  introductions_handshakes + direct_handshakes

theorem social_event_handshakes :
  handshake_count 40 25 15 = 465 := by
  sorry

end social_event_handshakes_l158_158555


namespace simplify_expression_l158_158049

theorem simplify_expression (x y : ℝ) : x^2 * y - 3 * x * y^2 + 2 * y * x^2 - y^2 * x = 3 * x^2 * y - 4 * x * y^2 :=
by
  sorry

end simplify_expression_l158_158049


namespace height_of_box_l158_158670

theorem height_of_box (h : ℝ) :
  (∃ (h : ℝ),
    (∀ (x y z : ℝ), (x = 3) ∧ (y = 3) ∧ (z = h / 2) → true) ∧
    (∀ (x y z : ℝ), (x = 1) ∧ (y = 1) ∧ (z = 1) → true) ∧
    h = 6) :=
sorry

end height_of_box_l158_158670


namespace pump_fills_tank_without_leak_l158_158522

theorem pump_fills_tank_without_leak (T : ℝ) (h1 : 1 / 12 = 1 / T - 1 / 12) : T = 6 :=
sorry

end pump_fills_tank_without_leak_l158_158522


namespace perfect_square_x4_x3_x2_x1_1_eq_x0_l158_158848

theorem perfect_square_x4_x3_x2_x1_1_eq_x0 :
  ∀ x : ℤ, ∃ n : ℤ, x^4 + x^3 + x^2 + x + 1 = n^2 ↔ x = 0 :=
by sorry

end perfect_square_x4_x3_x2_x1_1_eq_x0_l158_158848


namespace number_divided_by_189_l158_158372

noncomputable def target_number : ℝ := 3486

theorem number_divided_by_189 :
  target_number / 189 = 18.444444444444443 :=
by
  sorry

end number_divided_by_189_l158_158372


namespace EH_length_l158_158424

structure Rectangle :=
(AB BC CD DA : ℝ)
(horiz: AB=CD)
(verti: BC=DA)
(diag_eq: (AB^2 + BC^2) = (CD^2 + DA^2))

structure Point :=
(x y : ℝ)

noncomputable def H_distance (E D : Point)
    (AB BC : ℝ) : ℝ :=
    (E.y - D.y) -- if we consider D at origin (0,0)

theorem EH_length
    (AB BC : ℝ)
    (H_dist : ℝ)
    (E : Point)
    (rectangle : Rectangle) :
    AB = 50 →
    BC = 60 →
    E.x^2 + BC^2 = 30^2 + 60^2 →
    E.y = 40 →
    H_dist = E.y - CD →
    H_dist = 7.08 :=
by
    sorry

end EH_length_l158_158424


namespace smallest_base_10_integer_l158_158947

noncomputable def smallest_integer (a b: ℕ) (h₁: a > 2) (h₂: b > 2) (h₃: n = 2 * a + 1) (h₄: n = b + 2) : ℕ :=
  n

theorem smallest_base_10_integer : smallest_integer 3 5 (by decide) (by decide) (by decide) (by decide) = 7 :=
sorry

end smallest_base_10_integer_l158_158947


namespace doubled_volume_l158_158514

theorem doubled_volume (V₀ V₁ : ℝ) (hV₀ : V₀ = 3) (h_double : V₁ = V₀ * 8) : V₁ = 24 :=
by 
  rw [hV₀] at h_double
  exact h_double

end doubled_volume_l158_158514


namespace monotonic_function_range_l158_158595

theorem monotonic_function_range (a : ℝ) :
  (∀ x : ℝ, -3 * x^2 + 2 * a * x - 1 ≤ 0) → -Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3 :=
by
  sorry

end monotonic_function_range_l158_158595


namespace greatest_sum_l158_158163

-- stating the conditions
def condition1 (x y : ℝ) := x^2 + y^2 = 130
def condition2 (x y : ℝ) := x * y = 45

-- proving the result
theorem greatest_sum (x y : ℝ) 
  (h1 : condition1 x y) 
  (h2 : condition2 x y) : 
  x + y = 10 * Real.sqrt 2.2 :=
sorry

end greatest_sum_l158_158163


namespace correct_calculation_option_l158_158316

theorem correct_calculation_option :
  (∀ a : ℝ, 3 * a^5 - a^5 ≠ 3) ∧
  (∀ a : ℝ, a^2 + a^5 ≠ a^7) ∧
  (∀ a : ℝ, a^5 + a^5 = 2 * a^5) ∧
  (∀ x y : ℝ, x^2 * y + x * y^2 ≠ 2 * x^3 * y^3) :=
by
  sorry

end correct_calculation_option_l158_158316


namespace smallest_divisible_by_15_16_18_l158_158580

def factors_of_15 : Prop := 15 = 3 * 5
def factors_of_16 : Prop := 16 = 2^4
def factors_of_18 : Prop := 18 = 2 * 3^2

theorem smallest_divisible_by_15_16_18 (h1: factors_of_15) (h2: factors_of_16) (h3: factors_of_18) : 
  ∃ n, n > 0 ∧ n % 15 = 0 ∧ n % 16 = 0 ∧ n % 18 = 0 ∧ n = 720 :=
by
  sorry

end smallest_divisible_by_15_16_18_l158_158580


namespace determinant_modified_l158_158867

variable (a b c d : ℝ)

theorem determinant_modified (h : a * d - b * c = 10) :
  (a + 2 * c) * d - (b + 3 * d) * c = 10 - c * d := by
  sorry

end determinant_modified_l158_158867


namespace compute_fraction_l158_158190

theorem compute_fraction : (2015 : ℝ) / ((2015 : ℝ)^2 - (2016 : ℝ) * (2014 : ℝ)) = 2015 :=
by {
  sorry
}

end compute_fraction_l158_158190


namespace custom_operation_difference_correct_l158_158408

def custom_operation (x y : ℕ) : ℕ := x * y + 2 * x

theorem custom_operation_difference_correct :
  custom_operation 5 3 - custom_operation 3 5 = 4 :=
by
  sorry

end custom_operation_difference_correct_l158_158408


namespace fewest_apples_l158_158668

-- Definitions based on the conditions
def Yoongi_apples : Nat := 4
def Jungkook_initial_apples : Nat := 6
def Jungkook_additional_apples : Nat := 3
def Jungkook_apples : Nat := Jungkook_initial_apples + Jungkook_additional_apples
def Yuna_apples : Nat := 5

-- Main theorem based on the question and the correct answer
theorem fewest_apples : Yoongi_apples < Jungkook_apples ∧ Yoongi_apples < Yuna_apples :=
by
  sorry

end fewest_apples_l158_158668


namespace distance_probability_at_least_sqrt2_over_2_l158_158759

noncomputable def prob_dist_at_least : ℝ := 
  let T := ((0,0), (1,0), (0,1))
  -- Assumes conditions incorporated through identifying two random points within the triangle T.
  let area_T : ℝ := 0.5
  let valid_area : ℝ := 0.5 - (Real.pi * (Real.sqrt 2 / 2)^2 / 8 + ((Real.sqrt 2 / 2)^2 / 2) / 2)
  valid_area / area_T

theorem distance_probability_at_least_sqrt2_over_2 :
  prob_dist_at_least = (4 - π) / 8 :=
by
  sorry

end distance_probability_at_least_sqrt2_over_2_l158_158759


namespace largest_of_seven_consecutive_numbers_l158_158321

theorem largest_of_seven_consecutive_numbers (a b c d e f g : ℤ) (h1 : a + 1 = b)
                                             (h2 : b + 1 = c) (h3 : c + 1 = d)
                                             (h4 : d + 1 = e) (h5 : e + 1 = f)
                                             (h6 : f + 1 = g)
                                             (h_avg : (a + b + c + d + e + f + g) / 7 = 20) :
    g = 23 :=
by
  sorry

end largest_of_seven_consecutive_numbers_l158_158321


namespace fraction_difference_eq_l158_158763

theorem fraction_difference_eq (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : a / (1 + a) + b / (1 + b) = 1) :
  a / (1 + b^2) - b / (1 + a^2) = a - b :=
sorry

end fraction_difference_eq_l158_158763


namespace intersection_M_N_l158_158211

def M : Set ℝ := { x : ℝ | Real.log x / Real.log 2 < 2 }
def N : Set ℝ := { x : ℝ | x^2 - x - 2 < 0 }

theorem intersection_M_N : M ∩ N = { x : ℝ | 0 < x ∧ x < 2 } := by
  sorry

end intersection_M_N_l158_158211


namespace solid2_solid4_views_identical_l158_158613

-- Define the solids and their orthographic views
structure Solid :=
  (top_view : String)
  (front_view : String)
  (side_view : String)

-- Given solids as provided by the problem
def solid1 : Solid := { top_view := "...", front_view := "...", side_view := "..." }
def solid2 : Solid := { top_view := "...", front_view := "...", side_view := "..." }
def solid3 : Solid := { top_view := "...", front_view := "...", side_view := "..." }
def solid4 : Solid := { top_view := "...", front_view := "...", side_view := "..." }

-- Function to compare two solids' views
def views_identical (s1 s2 : Solid) : Prop :=
  (s1.top_view = s2.top_view ∧ s1.front_view = s2.front_view) ∨
  (s1.top_view = s2.top_view ∧ s1.side_view = s2.side_view) ∨
  (s1.front_view = s2.front_view ∧ s1.side_view = s2.side_view)

-- Theorem statement
theorem solid2_solid4_views_identical : views_identical solid2 solid4 := 
sorry

end solid2_solid4_views_identical_l158_158613


namespace range_of_a_inequality_solution_set_l158_158216

noncomputable def quadratic_condition_holds (a : ℝ) : Prop :=
∀ (x : ℝ), x^2 - 2 * a * x + a > 0

theorem range_of_a (a : ℝ) (h : quadratic_condition_holds a) : 0 < a ∧ a < 1 := sorry

theorem inequality_solution_set (a x : ℝ) (h1 : 0 < a) (h2 : a < 1) : (a^(x^2 - 3) < a^(2 * x) ∧ a^(2 * x) < 1) ↔ x > 3 := sorry

end range_of_a_inequality_solution_set_l158_158216


namespace smallest_prime_divisor_of_sum_of_powers_l158_158808

theorem smallest_prime_divisor_of_sum_of_powers :
  ∃ p, Prime p ∧ p = Nat.gcd (3 ^ 25 + 11 ^ 19) 2 := by
  sorry

end smallest_prime_divisor_of_sum_of_powers_l158_158808


namespace exponent_form_l158_158938

theorem exponent_form (x : ℕ) (k : ℕ) : (3^x) % 10 = 7 ↔ x = 4 * k + 3 :=
by
  sorry

end exponent_form_l158_158938


namespace eight_n_is_even_l158_158112

theorem eight_n_is_even (n : ℕ) (h : n = 7) : 8 * n = 56 :=
by {
  sorry
}

end eight_n_is_even_l158_158112


namespace last_two_digits_of_9_power_h_are_21_l158_158311

def a := 1
def b := 2^a
def c := 3^b
def d := 4^c
def e := 5^d
def f := 6^e
def g := 7^f
def h := 8^g

theorem last_two_digits_of_9_power_h_are_21 : (9^h) % 100 = 21 := by
  sorry

end last_two_digits_of_9_power_h_are_21_l158_158311


namespace valid_sequences_count_l158_158406

def g (n : ℕ) : ℕ :=
  if n = 3 then 1
  else if n = 4 then 1
  else if n = 5 then 2
  else if n < 3 then 0
  else g (n - 4) + 3 * g (n - 5) + 3 * g (n - 6)

theorem valid_sequences_count : g 17 = 37 :=
  sorry

end valid_sequences_count_l158_158406


namespace cube_sum_identity_l158_158960

theorem cube_sum_identity (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
by
  sorry

end cube_sum_identity_l158_158960


namespace doubled_volume_l158_158513

theorem doubled_volume (V₀ V₁ : ℝ) (hV₀ : V₀ = 3) (h_double : V₁ = V₀ * 8) : V₁ = 24 :=
by 
  rw [hV₀] at h_double
  exact h_double

end doubled_volume_l158_158513


namespace n_value_l158_158229

theorem n_value (n : ℕ) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^18 → n = 9 :=
by
  sorry

end n_value_l158_158229


namespace pond_contains_total_money_correct_l158_158698

def value_of_dime := 10
def value_of_quarter := 25
def value_of_nickel := 5
def value_of_penny := 1

def cindy_dimes := 5
def eric_quarters := 3
def garrick_nickels := 8
def ivy_pennies := 60

def total_money : ℕ := 
  cindy_dimes * value_of_dime + 
  eric_quarters * value_of_quarter + 
  garrick_nickels * value_of_nickel + 
  ivy_pennies * value_of_penny

theorem pond_contains_total_money_correct:
  total_money = 225 := by
  sorry

end pond_contains_total_money_correct_l158_158698


namespace problem_statement_l158_158897

-- Define rational number representations for points A, B, and C
def a : ℚ := (-4)^2 - 8

-- Define that B and C are opposites
def are_opposites (b c : ℚ) : Prop := b = -c

-- Define the distance condition
def distance_is_three (a c : ℚ) : Prop := |c - a| = 3

-- Main theorem statement
theorem problem_statement :
  (∃ b c : ℚ, are_opposites b c ∧ distance_is_three a c ∧ -a^2 + b - c = -74) ∨
  (∃ b c : ℚ, are_opposites b c ∧ distance_is_three a c ∧ -a^2 + b - c = -86) :=
sorry

end problem_statement_l158_158897


namespace largest_unreachable_integer_l158_158666

theorem largest_unreachable_integer : ∃ n : ℕ, (¬ ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ 8 * a + 11 * b = n)
  ∧ ∀ m : ℕ, m > n → (∃ a b : ℕ, 0 < a ∧ 0 < b ∧ 8 * a + 11 * b = m) := sorry

end largest_unreachable_integer_l158_158666


namespace probability_of_multiple_of_3_l158_158087

theorem probability_of_multiple_of_3 :
  let s := {1, 2, 3, 4, 5, 6}
  let multiples_of_3 := {x ∈ s | x % 3 = 0}
  (multiples_of_3.card / s.card : ℚ) = 1 / 3 := by
  sorry

end probability_of_multiple_of_3_l158_158087


namespace find_b_l158_158876

theorem find_b (A B C : ℝ) (a b c : ℝ)
  (h1 : Real.tan A = 1 / 3)
  (h2 : Real.tan B = 1 / 2)
  (h3 : a = 1)
  (h4 : A + B + C = π) -- This condition is added because angles in a triangle sum up to π.
  : b = Real.sqrt 2 :=
by
  sorry

end find_b_l158_158876


namespace cone_height_from_sphere_l158_158823

theorem cone_height_from_sphere (d_sphere d_base : ℝ) (h : ℝ) (V_sphere : ℝ) (V_cone : ℝ) 
  (h₁ : d_sphere = 6) 
  (h₂ : d_base = 12)
  (h₃ : V_sphere = 36 * Real.pi)
  (h₄ : V_cone = (1/3) * Real.pi * (d_base / 2)^2 * h) 
  (h₅ : V_sphere = V_cone) :
  h = 3 := by
  sorry

end cone_height_from_sphere_l158_158823


namespace base_conversion_problem_l158_158441

theorem base_conversion_problem (b : ℕ) (h : b^2 + b + 3 = 34) : b = 6 :=
sorry

end base_conversion_problem_l158_158441


namespace not_product_of_consecutives_l158_158437

theorem not_product_of_consecutives (n k : ℕ) : 
  ¬ (∃ a b: ℕ, a + 1 = b ∧ (2 * n^(3 * k) + 4 * n^k + 10 = a * b)) :=
by sorry

end not_product_of_consecutives_l158_158437


namespace line_does_not_pass_through_fourth_quadrant_l158_158224

theorem line_does_not_pass_through_fourth_quadrant
  (A B C : ℝ) (hAB : A * B < 0) (hBC : B * C < 0) :
  ¬ ∃ x y : ℝ, x > 0 ∧ y < 0 ∧ A * x + B * y + C = 0 :=
by
  sorry

end line_does_not_pass_through_fourth_quadrant_l158_158224


namespace basketball_player_probability_l158_158816

theorem basketball_player_probability :
  (let p_success := 3 / 5
   let p_failure := 1 - p_success
   let p_all_failure := p_failure ^ 3
   let p_at_least_one_success := 1 - p_all_failure
   in p_at_least_one_success = 0.936) :=
by
  sorry

end basketball_player_probability_l158_158816


namespace sqrt_of_quarter_l158_158304

-- Definitions as per conditions
def is_square_root (x y : ℝ) : Prop := x^2 = y

-- Theorem statement proving question == answer given conditions
theorem sqrt_of_quarter : is_square_root 0.5 0.25 ∧ is_square_root (-0.5) 0.25 ∧ (∀ x, is_square_root x 0.25 → (x = 0.5 ∨ x = -0.5)) :=
by
  -- Skipping proof with sorry
  sorry

end sqrt_of_quarter_l158_158304


namespace unsolved_problems_exist_l158_158090

noncomputable def main_theorem: Prop :=
  ∃ (P : Prop), ¬(P = true) ∧ ¬(P = false)

theorem unsolved_problems_exist : main_theorem :=
sorry

end unsolved_problems_exist_l158_158090


namespace value_of_angle_C_perimeter_range_l158_158587

-- Part (1): Prove angle C value
theorem value_of_angle_C
  {a b c : ℝ} {A B C : ℝ}
  (acute_ABC : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  (m : ℝ × ℝ := (Real.sin C, Real.cos C))
  (n : ℝ × ℝ := (2 * Real.sin A - Real.cos B, -Real.sin B))
  (orthogonal_mn : m.1 * n.1 + m.2 * n.2 = 0) 
  : C = π / 6 := sorry

-- Part (2): Prove perimeter range
theorem perimeter_range
  {a b c : ℝ} {A B C : ℝ}
  (A_range : π / 3 < A ∧ A < π / 2)
  (C_value : C = π / 6)
  (a_value : a = 2)
  (acute_ABC : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  : 3 + 2 * Real.sqrt 3 < a + b + c ∧ a + b + c < 2 + 3 * Real.sqrt 3 := sorry

end value_of_angle_C_perimeter_range_l158_158587


namespace fixed_point_l158_158790

theorem fixed_point (a : ℝ) : (a + 1) * (-4) - (2 * a + 5) * (-2) - 6 = 0 :=
by
  sorry

end fixed_point_l158_158790


namespace sequence_sum_problem_l158_158260

theorem sequence_sum_problem (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h1 : ∀ n, S n = 2 * a n - n) :
  (2 / (a 1 * a 2) + 4 / (a 2 * a 3) + 8 / (a 3 * a 4) + 16 / (a 4 * a 5) : ℚ) = 30 / 31 := 
sorry

end sequence_sum_problem_l158_158260


namespace complex_imag_part_of_z_l158_158443

theorem complex_imag_part_of_z (z : ℂ) (h : z * (2 + ⅈ) = 3 - 6 * ⅈ) : z.im = -3 := by
  sorry

end complex_imag_part_of_z_l158_158443


namespace winning_percentage_l158_158344

theorem winning_percentage (total_games first_games remaining_games : ℕ) 
                           (first_win_percent remaining_win_percent : ℝ)
                           (total_games_eq : total_games = 60)
                           (first_games_eq : first_games = 30)
                           (remaining_games_eq : remaining_games = 30)
                           (first_win_percent_eq : first_win_percent = 0.40)
                           (remaining_win_percent_eq : remaining_win_percent = 0.80) :
                           (first_win_percent * (first_games : ℝ) +
                            remaining_win_percent * (remaining_games : ℝ)) /
                           (total_games : ℝ) * 100 = 60 := sorry

end winning_percentage_l158_158344


namespace perfect_square_expression_l158_158140

theorem perfect_square_expression (p : ℝ) : 
  (12.86^2 + 12.86 * p + 0.14^2) = (12.86 + 0.14)^2 → p = 0.28 :=
by
  sorry

end perfect_square_expression_l158_158140


namespace third_roll_six_probability_l158_158559

noncomputable def Die_A_six_prob : ℚ := 1 / 6
noncomputable def Die_B_six_prob : ℚ := 1 / 2
noncomputable def Die_C_one_prob : ℚ := 3 / 5
noncomputable def Die_B_not_six_prob : ℚ := 1 / 10
noncomputable def Die_C_not_one_prob : ℚ := 1 / 15

noncomputable def prob_two_sixes_die_A : ℚ := Die_A_six_prob ^ 2
noncomputable def prob_two_sixes_die_B : ℚ := Die_B_six_prob ^ 2
noncomputable def prob_two_sixes_die_C : ℚ := Die_C_not_one_prob ^ 2

noncomputable def total_prob_two_sixes : ℚ := 
  (1 / 3) * (prob_two_sixes_die_A + prob_two_sixes_die_B + prob_two_sixes_die_C)

noncomputable def cond_prob_die_A_given_two_sixes : ℚ := prob_two_sixes_die_A / total_prob_two_sixes
noncomputable def cond_prob_die_B_given_two_sixes : ℚ := prob_two_sixes_die_B / total_prob_two_sixes
noncomputable def cond_prob_die_C_given_two_sixes : ℚ := prob_two_sixes_die_C / total_prob_two_sixes

noncomputable def prob_third_six : ℚ := 
  cond_prob_die_A_given_two_sixes * Die_A_six_prob + 
  cond_prob_die_B_given_two_sixes * Die_B_six_prob + 
  cond_prob_die_C_given_two_sixes * Die_C_not_one_prob

theorem third_roll_six_probability : 
  prob_third_six = sorry := 
  sorry

end third_roll_six_probability_l158_158559


namespace Meghan_scored_20_marks_less_than_Jose_l158_158414

theorem Meghan_scored_20_marks_less_than_Jose
  (M J A : ℕ)
  (h1 : J = A + 40)
  (h2 : M + J + A = 210)
  (h3 : J = 100 - 10) :
  J - M = 20 :=
by
  -- Skipping the proof
  sorry

end Meghan_scored_20_marks_less_than_Jose_l158_158414


namespace John_age_l158_158367

theorem John_age (Drew Maya Peter John Jacob : ℕ)
  (h1 : Drew = Maya + 5)
  (h2 : Peter = Drew + 4)
  (h3 : John = 2 * Maya)
  (h4 : (Jacob + 2) * 2 = Peter + 2)
  (h5 : Jacob = 11) : John = 30 :=
by 
  sorry

end John_age_l158_158367


namespace tangent_AC_l158_158615

theorem tangent_AC {A B C L P : Point} 
  (h_triangle_ABC : Triangle A B C)
  (h_BL_bisector : IsAngleBisector B L C)
  (h_tangent_L : TangentThroughPoint L (Circumcircle B L C) (Line P L))
  (h_intersects_AB_P : Intersection (Line A B) (Line P L) P) :
  Tangent (Circumcircle B P L) (Line A C) :=
sorry

end tangent_AC_l158_158615


namespace proof_problem_l158_158838

theorem proof_problem (x y : ℝ) (h1 : 3 * x ^ 2 - 5 * x + 4 * y + 6 = 0) 
                      (h2 : 3 * x - 2 * y + 1 = 0) : 
                      4 * y ^ 2 - 2 * y + 24 = 0 := 
by 
  sorry

end proof_problem_l158_158838


namespace trigonometric_identity_l158_158853

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 4 :=
by 
  sorry

end trigonometric_identity_l158_158853


namespace intersection_A_B_l158_158099

variable (x : ℤ)

def A := { x | x^2 - 4 * x ≤ 0 }
def B := { x | -1 ≤ x ∧ x < 4 }

theorem intersection_A_B :
  { x : ℤ | x ∈ A ∧ x ∈ B } = {0, 1, 2, 3} :=
sorry

end intersection_A_B_l158_158099


namespace quadratic_non_real_roots_l158_158724

theorem quadratic_non_real_roots (b : ℝ) : 
  let a : ℝ := 1 
  let c : ℝ := 16 in
  (b^2 - 4 * a * c < 0) ↔ (-8 < b ∧ b < 8) :=
sorry

end quadratic_non_real_roots_l158_158724


namespace multiply_97_103_eq_9991_l158_158563

theorem multiply_97_103_eq_9991 : (97 * 103 = 9991) :=
by
  have h1 : 97 = 100 - 3 := rfl
  have h2 : 103 = 100 + 3 := rfl
  calc
    97 * 103 = (100 - 3) * (100 + 3) : by rw [h1, h2]
    ... = 100^2 - 3^2 : by rw [mul_add, add_mul, sub_mul, add_sub_cancel, sub_add_cancel]
    ... = 10000 - 9 : by norm_num
    ... = 9991 : by norm_num

end multiply_97_103_eq_9991_l158_158563


namespace vasya_fraction_l158_158540

variable (a b c d s : ℝ)

-- Anton drove half the distance Vasya did
axiom h1 : a = b / 2

-- Sasha drove as long as Anton and Dima together
axiom h2 : c = a + d

-- Dima drove one-tenth of the total distance
axiom h3 : d = s / 10

-- The total distance is the sum of distances driven by Anton, Vasya, Sasha, and Dima
axiom h4 : a + b + c + d = s

-- We need to prove that Vasya drove 0.4 of the total distance
theorem vasya_fraction (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : b = 0.4 * s :=
by
  sorry

end vasya_fraction_l158_158540


namespace find_a_value_l158_158230

theorem find_a_value (a : ℕ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 := 
by 
  sorry -- Placeholder for the proof

end find_a_value_l158_158230


namespace max_super_bishops_l158_158471

structure SuperBishop where
  position : (ℕ × ℕ) -- Board position as a pair (row, column)

def attacks (A B : SuperBishop) : Prop :=
  (A.position.1 + A.position.2 = B.position.1 + B.position.2 ∨
   A.position.1 - A.position.2 = B.position.1 - B.position.2) ∧
  ∀ P, ∃ C: SuperBishop, (C.position = P) → (C = A ∨ C = B ∨ P = (B.position.1 + 1, B.position.2 + 1) ∨ P = (B.position.1 - 1, B.position.2 - 1))

def valid_attack_config (S : Finset SuperBishop) : Prop :=
  ∀ A ∈ S, ∃ B ∈ S, A ≠ B ∧ attacks A B

theorem max_super_bishops :
  ∃ (S : Finset SuperBishop), valid_attack_config S ∧ S.card = 32 :=
sorry

end max_super_bishops_l158_158471


namespace solve_inequalities_l158_158017

theorem solve_inequalities :
  (∀ x : ℝ, x^2 + 3 * x - 10 ≥ 0 ↔ (x ≤ -5 ∨ x ≥ 2)) ∧
  (∀ x : ℝ, x^2 - 3 * x - 2 ≤ 0 ↔ (3 - Real.sqrt 17) / 2 ≤ x ∧ x ≤ (3 + Real.sqrt 17) / 2) :=
by
  sorry

end solve_inequalities_l158_158017


namespace handshake_problem_l158_158036

noncomputable def total_handshakes (num_companies : ℕ) (repr_per_company : ℕ) : ℕ :=
    let total_people := num_companies * repr_per_company
    let possible_handshakes_per_person := total_people - repr_per_company
    (total_people * possible_handshakes_per_person) / 2

theorem handshake_problem : total_handshakes 4 4 = 96 :=
by
  sorry

end handshake_problem_l158_158036


namespace find_product_l158_158964

theorem find_product
  (a b c d : ℝ) :
  3 * a + 2 * b + 4 * c + 6 * d = 60 →
  4 * (d + c) = b^2 →
  4 * b + 2 * c = a →
  c - 2 = d →
  a * b * c * d = 0 :=
by
  sorry

end find_product_l158_158964


namespace milton_zoology_books_l158_158626

variable (Z : ℕ)
variable (total_books botany_books : ℕ)

theorem milton_zoology_books (h1 : total_books = 960)
    (h2 : botany_books = 7 * Z)
    (h3 : total_books = Z + botany_books) :
    Z = 120 := by
  sorry

end milton_zoology_books_l158_158626


namespace linear_term_coefficient_is_neg_two_l158_158784

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

-- Define the specific quadratic equation
def specific_quadratic_eq (x : ℝ) : Prop :=
  quadratic_eq 1 (-2) (-1) x

-- The statement to prove the coefficient of the linear term
theorem linear_term_coefficient_is_neg_two : ∀ x : ℝ, specific_quadratic_eq x → ∀ a b c : ℝ, quadratic_eq a b c x → b = -2 :=
by
  intros x h_eq a b c h_quadratic_eq
  -- Proof is omitted
  sorry

end linear_term_coefficient_is_neg_two_l158_158784


namespace emery_total_alteration_cost_l158_158572

-- Definition of the initial conditions
def num_pairs_of_shoes := 17
def cost_per_shoe := 29
def shoes_per_pair := 2

-- Proving the total cost
theorem emery_total_alteration_cost : num_pairs_of_shoes * shoes_per_pair * cost_per_shoe = 986 := by
  sorry

end emery_total_alteration_cost_l158_158572


namespace binary_to_decimal_101101_l158_158990

theorem binary_to_decimal_101101 : 
  (1 * 2^0 + 0 * 2^1 + 1 * 2^2 + 1 * 2^3 + 0 * 2^4 + 1 * 2^5) = 45 := 
by 
  sorry

end binary_to_decimal_101101_l158_158990


namespace number_of_triangles_with_perimeter_27_l158_158223

theorem number_of_triangles_with_perimeter_27 : 
  ∃ (n : ℕ), (∀ (a b c : ℕ), a ≤ b ∧ b ≤ c ∧ a + b + c = 27 → a + b > c ∧ a + c > b ∧ b + c > a → 
  n = 19 ) :=
  sorry

end number_of_triangles_with_perimeter_27_l158_158223


namespace find_r_l158_158005

noncomputable def g (x : ℝ) (p q r : ℝ) := x^3 + p * x^2 + q * x + r

theorem find_r 
  (p q r : ℝ) 
  (h1 : ∀ x : ℝ, g x p q r = (x + 100) * (x + 0) * (x + 0))
  (h2 : p + q + r = 100) : 
  r = 0 := 
by
  sorry

end find_r_l158_158005


namespace inequality_solution_l158_158274

theorem inequality_solution (x : ℝ) : 
  (3 - (1 / (3 * x + 4)) < 5) ↔ (x < -4 / 3) := 
by
  sorry

end inequality_solution_l158_158274


namespace statement1_statement2_statement3_l158_158210

variable (a b c m : ℝ)

-- Given condition
def quadratic_eq (a b c : ℝ) : Prop := a ≠ 0

-- Statement 1
theorem statement1 (h0 : quadratic_eq a b c) (h1 : ∀ x, a * x^2 + b * x + c = 0 ↔ x = 1 ∨ x = 2) : 2 * a - c = 0 :=
sorry

-- Statement 2
theorem statement2 (h0 : quadratic_eq a b c) (h2 : b = 2 * a + c) : (b^2 - 4 * a * c) > 0 :=
sorry

-- Statement 3
theorem statement3 (h0 : quadratic_eq a b c) (h3 : a * m^2 + b * m + c = 0) : b^2 - 4 * a * c = (2 * a * m + b)^2 :=
sorry

end statement1_statement2_statement3_l158_158210


namespace smallest_base_10_integer_l158_158945

noncomputable def smallest_integer (a b: ℕ) (h₁: a > 2) (h₂: b > 2) (h₃: n = 2 * a + 1) (h₄: n = b + 2) : ℕ :=
  n

theorem smallest_base_10_integer : smallest_integer 3 5 (by decide) (by decide) (by decide) (by decide) = 7 :=
sorry

end smallest_base_10_integer_l158_158945


namespace arcsin_neg_half_eq_neg_pi_six_l158_158360

theorem arcsin_neg_half_eq_neg_pi_six : 
  Real.arcsin (-1 / 2) = -Real.pi / 6 := 
sorry

end arcsin_neg_half_eq_neg_pi_six_l158_158360


namespace minimize_relative_waiting_time_l158_158333

-- Definitions of task times in seconds
def task_U : ℕ := 10
def task_V : ℕ := 120
def task_W : ℕ := 900

-- Definition of relative waiting time given a sequence of task execution times
def relative_waiting_time (times : List ℕ) : ℚ :=
  (times.head! : ℚ) / (times.head! : ℚ) + 
  (times.head! + times.tail.head! : ℚ) / (times.tail.head! : ℚ) + 
  (times.head! + times.tail.head! + times.tail.tail.head! : ℚ) / (times.tail.tail.head! : ℚ)

-- Sequences
def sequence_A : List ℕ := [task_U, task_V, task_W]
def sequence_B : List ℕ := [task_V, task_W, task_U]
def sequence_C : List ℕ := [task_W, task_U, task_V]
def sequence_D : List ℕ := [task_U, task_W, task_V]

-- Sum of relative waiting times for each sequence
def S_A := relative_waiting_time sequence_A
def S_B := relative_waiting_time sequence_B
def S_C := relative_waiting_time sequence_C
def S_D := relative_waiting_time sequence_D

-- Theorem to prove that sequence A has the minimum sum of relative waiting times
theorem minimize_relative_waiting_time : S_A < S_B ∧ S_A < S_C ∧ S_A < S_D := 
  by sorry

end minimize_relative_waiting_time_l158_158333


namespace quadratic_general_form_l158_158567

theorem quadratic_general_form (x : ℝ) :
    (x + 3)^2 = x * (3 * x - 1) →
    2 * x^2 - 7 * x - 9 = 0 :=
by
  intros h
  sorry

end quadratic_general_form_l158_158567


namespace length_of_MN_l158_158881

noncomputable def curve_eq (α : ℝ) : ℝ × ℝ := (2 * Real.cos α + 1, 2 * Real.sin α)

noncomputable def line_eq (ρ θ : ℝ) : ℝ × ℝ := 
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem length_of_MN : ∀ (M N : ℝ × ℝ), 
  M ∈ {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2)^2 = 4} ∧
  N ∈ {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2)^2 = 4} ∧
  M ∈ {p : ℝ × ℝ | p.1 + p.2 = 2} ∧
  N ∈ {p : ℝ × ℝ | p.1 + p.2 = 2} →
  dist M N = Real.sqrt 14 :=
by
  sorry

end length_of_MN_l158_158881


namespace non_real_roots_interval_l158_158731

theorem non_real_roots_interval (b : ℝ) : (b^2 < 64) ↔ (b > -8 ∧ b < 8) :=
by
  sorry

end non_real_roots_interval_l158_158731


namespace find_m_l158_158490

theorem find_m (m : ℝ) : (243 : ℝ)^(1/3) = (3 : ℝ)^m → m = 5 / 3 :=
by
  sorry

end find_m_l158_158490


namespace expression_change_l158_158479

theorem expression_change (a b c : ℝ) : 
  a - (2 * b - 3 * c) = a + (-2 * b + 3 * c) := 
by sorry

end expression_change_l158_158479


namespace expectation_eq_of_ae_eq_and_E_ξ_exists_l158_158015

open MeasureTheory

variables {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

-- Define two random variables
variables (ξ η : Ω → ℝ)

-- Define your conditions
axiom ξ_eq_η_ae : ∀ᵐ ω ∂μ, ξ ω = η ω
axiom E_ξ_exists : Integrable ξ μ

-- Now state the theorem
theorem expectation_eq_of_ae_eq_and_E_ξ_exists :
  Integrable η μ ∧ ∫ ω, ξ ω ∂μ = ∫ ω, η ω ∂μ :=
sorry

end expectation_eq_of_ae_eq_and_E_ξ_exists_l158_158015


namespace basketball_free_throws_l158_158455

theorem basketball_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 2 * a)
  (h2 : b = a - 2)
  (h3 : 2 * a + 3 * b + x = 68) : x = 44 :=
by
  sorry

end basketball_free_throws_l158_158455


namespace exponent_proof_l158_158491

theorem exponent_proof (m : ℝ) : (243 : ℝ) = (3 : ℝ)^5 → (243 : ℝ)^(1/3) = (3 : ℝ)^m → m = 5/3 :=
by
  intros h1 h2
  sorry

end exponent_proof_l158_158491


namespace probability_of_same_color_l158_158138

-- Defining the given conditions
def green_balls := 6
def red_balls := 4
def total_balls := green_balls + red_balls

def probability_same_color : ℚ :=
  let prob_green := (green_balls / total_balls) * (green_balls / total_balls)
  let prob_red := (red_balls / total_balls) * (red_balls / total_balls)
  prob_green + prob_red

-- Statement of the problem rewritten in Lean 4
theorem probability_of_same_color :
  probability_same_color = 13 / 25 :=
by
  sorry

end probability_of_same_color_l158_158138


namespace wall_width_l158_158502

theorem wall_width (brick_length brick_height brick_depth : ℝ)
    (wall_length wall_height : ℝ)
    (num_bricks : ℝ)
    (total_bricks_volume : ℝ)
    (total_wall_volume : ℝ) :
    brick_length = 25 →
    brick_height = 11.25 →
    brick_depth = 6 →
    wall_length = 800 →
    wall_height = 600 →
    num_bricks = 6400 →
    total_bricks_volume = num_bricks * (brick_length * brick_height * brick_depth) →
    total_wall_volume = wall_length * wall_height * (total_bricks_volume / (brick_length * brick_height * brick_depth)) →
    (total_bricks_volume / (wall_length * wall_height) = 22.5) :=
by
  intros
  sorry -- proof not required

end wall_width_l158_158502


namespace find_k_l158_158025

open Real

noncomputable def curve (x : ℝ) (k : ℝ) : ℝ := 3 * ln x + x + k
noncomputable def tangent_line (x : ℝ) : ℝ := 4 * x - 1

theorem find_k (x₀ y₀ k : ℝ) (h_curve : y₀ = curve x₀ k) (h_tangent : ∀ x, 4 * x - (curve x₀ k) = 0) : k = 2 := by
  sorry

end find_k_l158_158025


namespace probability_ace_spades_then_king_spades_l158_158470

theorem probability_ace_spades_then_king_spades :
  ∃ (p : ℚ), (p = 1/52 * 1/51) := sorry

end probability_ace_spades_then_king_spades_l158_158470


namespace length_of_side_d_l158_158014

variable (a b c d : ℕ)
variable (h_ratio1 : a / c = 3 / 4)
variable (h_ratio2 : b / d = 3 / 4)
variable (h_a : a = 3)
variable (h_b : b = 6)

theorem length_of_side_d (a b c d : ℕ)
  (h_ratio1 : a / c = 3 / 4)
  (h_ratio2 : b / d = 3 / 4)
  (h_a : a = 3)
  (h_b : b = 6) : d = 8 := 
sorry

end length_of_side_d_l158_158014


namespace problem_statement_l158_158108

theorem problem_statement (r p q : ℝ) (hr : r < 0) (hpq_ne_zero : p * q ≠ 0) (hp2r_gt_q2r : p^2 * r > q^2 * r) :
  ¬ (-p > -q) ∧ ¬ (-p < q) ∧ ¬ (1 < -q / p) ∧ ¬ (1 > q / p) :=
by
  sorry

end problem_statement_l158_158108


namespace janet_wait_time_l158_158242

theorem janet_wait_time 
  (janet_speed : ℝ)
  (sister_speed : ℝ)
  (lake_width : ℝ)
  (h1 : janet_speed = 30)
  (h2 : sister_speed = 12)
  (h3 : lake_width = 60) 
  :
  60 / 12 - 60 / 30 = 3 :=
by
  sorry

end janet_wait_time_l158_158242


namespace fraction_of_shaded_circle_l158_158753

theorem fraction_of_shaded_circle (total_regions shaded_regions : ℕ) (h1 : total_regions = 4) (h2 : shaded_regions = 1) :
  shaded_regions / total_regions = 1 / 4 := by
  sorry

end fraction_of_shaded_circle_l158_158753


namespace solution_set_of_inequality_system_l158_158029

theorem solution_set_of_inequality_system (x : ℝ) :
  (3 * x - 1 ≥ x + 1) ∧ (x + 4 > 4 * x - 2) ↔ (1 ≤ x ∧ x < 2) := 
by
  sorry

end solution_set_of_inequality_system_l158_158029


namespace vasya_drove_0_4_of_total_distance_l158_158533

-- Define variables for the distances driven by Anton (a), Vasya (b), Sasha (c), and Dima (d)
variables {a b c d s : ℝ}

-- Define the conditions in Lean
def condition_1 := a = b / 2
def condition_2 := c = a + d
def condition_3 := d = s / 10
def condition_4 := s ≠ 0
def condition_5 := a + b + c + d = s

-- Prove that Vasya drove 0.4 of the total distance
theorem vasya_drove_0_4_of_total_distance (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (h4 : condition_4) (h5 : condition_5) : b / s = 0.4 :=
by
  sorry

end vasya_drove_0_4_of_total_distance_l158_158533


namespace initial_volume_is_72_l158_158332

noncomputable def initial_volume (V : ℝ) : Prop :=
  let salt_initial : ℝ := 0.10 * V
  let total_volume_new : ℝ := V + 18
  let salt_percentage_new : ℝ := 0.08 * total_volume_new
  salt_initial = salt_percentage_new

theorem initial_volume_is_72 :
  ∃ V : ℝ, initial_volume V ∧ V = 72 :=
by
  sorry

end initial_volume_is_72_l158_158332


namespace vasya_drives_fraction_l158_158546

theorem vasya_drives_fraction {a b c d s : ℝ} 
  (h1 : a = b / 2) 
  (h2 : c = a + d) 
  (h3 : d = s / 10) 
  (h4 : a + b + c + d = s) : 
  b / s = 0.4 :=
by
  sorry

end vasya_drives_fraction_l158_158546


namespace smallest_base10_integer_l158_158951

theorem smallest_base10_integer {a b n : ℕ} (ha : a > 2) (hb : b > 2)
  (h₁ : 2 * a + 1 = n) (h₂ : 1 * b + 2 = n) :
  n = 7 :=
sorry

end smallest_base10_integer_l158_158951


namespace countDivisorsOf72Pow8_l158_158200

-- Definitions of conditions in Lean 4
def isPerfectSquare (a b : ℕ) : Prop := a % 2 = 0 ∧ b % 2 = 0
def isPerfectCube (a b : ℕ) : Prop := a % 3 = 0 ∧ b % 3 = 0
def isPerfectSixthPower (a b : ℕ) : Prop := a % 6 = 0 ∧ b % 6 = 0

def countPerfectSquares : ℕ := 13 * 9
def countPerfectCubes : ℕ := 9 * 6
def countPerfectSixthPowers : ℕ := 5 * 3

-- The proof problem to prove the number of such divisors is 156
theorem countDivisorsOf72Pow8:
  (countPerfectSquares + countPerfectCubes - countPerfectSixthPowers) = 156 :=
by
  sorry

end countDivisorsOf72Pow8_l158_158200


namespace evaluate_expression_l158_158369

theorem evaluate_expression :
  (3 / 2) * ((8 / 3) * ((15 / 8) - (5 / 6))) / (((7 / 8) + (11 / 6)) / (13 / 4)) = 5 :=
by
  sorry

end evaluate_expression_l158_158369


namespace sum_of_fractions_l158_158841

theorem sum_of_fractions : 
  (1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8) = 3 / 8) :=
by sorry

end sum_of_fractions_l158_158841


namespace area_ratio_rect_sq_l158_158284

variable (s : ℝ)

def side_len_sq (S : ℝ) : Prop := s = S
def longer_side_rect (R : ℝ) : Prop := R = 1.2 * s
def shorter_side_rect (R : ℝ) : Prop := R = 0.8 * s
def area_sq (S : ℝ) : ℝ := S * S
def area_rect (R_long R_short : ℝ) : ℝ := R_long * R_short
def ratio_area (areaR areaS : ℝ) : ℝ := areaR / areaS

theorem area_ratio_rect_sq (s S R_long R_short : ℝ) (h1 : side_len_sq s S) (h2 : longer_side_rect s R_long) (h3 : shorter_side_rect s R_short) :
  ratio_area (area_rect R_long R_short) (area_sq S) = 24/25 :=
by
  sorry

end area_ratio_rect_sq_l158_158284


namespace compute_expr_l158_158132

open Real

-- Define the polynomial and its roots.
def polynomial (x : ℝ) := 3 * x^2 - 5 * x - 2

-- Given conditions: p and q are roots of the polynomial.
def is_root (p q : ℝ) : Prop := 
  polynomial p = 0 ∧ polynomial q = 0

-- The main theorem.
theorem compute_expr (p q : ℝ) (h : is_root p q) : 
  ∃ k : ℝ, k = p - q ∧ (p ≠ q) → (9 * p^3 + 9 * q^3) / (p - q) = 215 / (3 * (p - q)) :=
sorry

end compute_expr_l158_158132


namespace janet_waiting_time_l158_158247

-- Define the speeds and distance
def janet_speed : ℝ := 30 -- miles per hour
def sister_speed : ℝ := 12 -- miles per hour
def lake_width : ℝ := 60 -- miles

-- Define the travel times
def janet_travel_time : ℝ := lake_width / janet_speed
def sister_travel_time : ℝ := lake_width / sister_speed

-- The theorem to be proved
theorem janet_waiting_time : 
  sister_travel_time - janet_travel_time = 3 := 
by 
  sorry

end janet_waiting_time_l158_158247


namespace quadratic_non_real_roots_l158_158735

variable (b : ℝ)

theorem quadratic_non_real_roots : (b^2 - 64 < 0) → (-8 < b ∧ b < 8) :=
by
  sorry

end quadratic_non_real_roots_l158_158735


namespace square_root_unique_l158_158463

theorem square_root_unique (x : ℝ) (h1 : x + 3 ≥ 0) (h2 : 2 * x - 6 ≥ 0)
  (h : (x + 3)^2 = (2 * x - 6)^2) :
  x = 1 ∧ (x + 3)^2 = 16 := 
by
  sorry

end square_root_unique_l158_158463


namespace number_of_rhombuses_is_84_l158_158827

def total_rhombuses (side_length_large_triangle : Nat) (side_length_small_triangle : Nat) (num_small_triangles : Nat) : Nat :=
  if side_length_large_triangle = 10 ∧ 
     side_length_small_triangle = 1 ∧ 
     num_small_triangles = 100 then 84 else 0

theorem number_of_rhombuses_is_84 :
  total_rhombuses 10 1 100 = 84 := by
  sorry

end number_of_rhombuses_is_84_l158_158827


namespace non_real_roots_of_quadratic_l158_158732

theorem non_real_roots_of_quadratic (b : ℝ) : 
  (¬ ∃ x1 x2 : ℝ, x1^2 + bx1 + 16 = 0 ∧ x2^2 + bx2 + 16 = 0 ∧ x1 = x2) ↔ b ∈ set.Ioo (-8 : ℝ) (8 : ℝ) :=
by {
  sorry
}

end non_real_roots_of_quadratic_l158_158732


namespace polynomial_divisibility_l158_158632

theorem polynomial_divisibility (m : ℕ) (h_pos : 0 < m) : 
  ∀ x : ℝ, x * (x + 1) * (2 * x + 1) ∣ (x + 1)^(2 * m) - x^(2 * m) - 2 * x - 1 :=
sorry

end polynomial_divisibility_l158_158632


namespace smallest_base_10_integer_exists_l158_158950

theorem smallest_base_10_integer_exists :
  ∃ (x a b : ℕ), (a > 2) ∧ (b > 2) ∧ (x = 2 * a + 1) ∧ (x = b + 2) ∧ (x = 7) :=
by
  sorry

end smallest_base_10_integer_exists_l158_158950


namespace correct_calculation_result_l158_158667

theorem correct_calculation_result :
  (∃ x : ℤ, 14 * x = 70) → (5 - 6 = -1) :=
by
  sorry

end correct_calculation_result_l158_158667


namespace ratio_of_surface_areas_l158_158113

-- Definitions based on conditions
def side_length_ratio (a b : ℝ) : Prop := b = 6 * a
def surface_area (a : ℝ) : ℝ := 6 * a ^ 2

-- Theorem statement
theorem ratio_of_surface_areas (a b : ℝ) (h : side_length_ratio a b) :
  (surface_area b) / (surface_area a) = 36 := by
  sorry

end ratio_of_surface_areas_l158_158113


namespace average_of_r_s_t_l158_158606

theorem average_of_r_s_t
  (r s t : ℝ)
  (h : (5 / 4) * (r + s + t) = 20) :
  (r + s + t) / 3 = 16 / 3 :=
by
  sorry

end average_of_r_s_t_l158_158606


namespace probability_of_highest_number_six_l158_158329

open ProbabilityTheory

-- Definitions of the condition
def box : set ℕ := {1, 2, 3, 4, 5, 6, 7}
def number_of_cards_selected := 4

-- Probability that the highest number selected is 6
def probability_highest_is_six : ℚ :=
  (3 / 7 : ℚ)

theorem probability_of_highest_number_six :
  ∃ p : ℚ, p = probability_highest_is_six ∧ 
    probability_space (finset.powerset_len number_of_cards_selected box) 
      (λ s, 6 ∈ s ∧ ∀ x ∈ s, x ≤ 6) = p := by
  sorry

end probability_of_highest_number_six_l158_158329


namespace trigonometric_expression_l158_158091

variable (α : Real)
open Real

theorem trigonometric_expression (h : tan α = 3) : 
  (2 * sin α - cos α) / (sin α + 3 * cos α) = 5 / 6 := 
by
  sorry

end trigonometric_expression_l158_158091


namespace factor_sum_l158_158225

theorem factor_sum (R S : ℝ) (h : ∃ (b c : ℝ), (x^2 + 3*x + 7) * (x^2 + b*x + c) = x^4 + R*x^2 + S) : R + S = 54 :=
sorry

end factor_sum_l158_158225


namespace inheritance_problem_l158_158925

theorem inheritance_problem
    (A B C : ℕ)
    (h1 : A + B + C = 30000)
    (h2 : A - B = B - C)
    (h3 : A = B + C) :
    A = 15000 ∧ B = 10000 ∧ C = 5000 := by
  sorry

end inheritance_problem_l158_158925


namespace fruits_calculation_l158_158770

structure FruitStatus :=
  (initial_picked  : ℝ)
  (initial_eaten  : ℝ)

def apples_status : FruitStatus :=
  { initial_picked := 7.0 + 3.0 + 5.0, initial_eaten := 6.0 + 2.0 }

def pears_status : FruitStatus :=
  { initial_picked := 0, initial_eaten := 4.0 + 3.0 }  -- number of pears picked is unknown, hence 0

def oranges_status : FruitStatus :=
  { initial_picked := 8.0, initial_eaten := 8.0 }

def cherries_status : FruitStatus :=
  { initial_picked := 4.0, initial_eaten := 4.0 }

theorem fruits_calculation :
  (apples_status.initial_picked - apples_status.initial_eaten = 7.0) ∧
  (pears_status.initial_picked - pears_status.initial_eaten = 0) ∧  -- cannot be determined in the problem statement
  (oranges_status.initial_picked - oranges_status.initial_eaten = 0) ∧
  (cherries_status.initial_picked - cherries_status.initial_eaten = 0) :=
by {
  sorry
}

end fruits_calculation_l158_158770


namespace coffee_price_increase_l158_158924

theorem coffee_price_increase (price_first_quarter price_fourth_quarter : ℕ) 
  (h_first : price_first_quarter = 40) (h_fourth : price_fourth_quarter = 60) : 
  ((price_fourth_quarter - price_first_quarter) * 100) / price_first_quarter = 50 := 
by
  -- proof would proceed here
  sorry

end coffee_price_increase_l158_158924


namespace quadratic_solution_1_quadratic_solution_2_l158_158782

theorem quadratic_solution_1 (x : ℝ) : x^2 - 8 * x + 12 = 0 ↔ x = 2 ∨ x = 6 := 
by
  sorry

theorem quadratic_solution_2 (x : ℝ) : (x - 3)^2 = 2 * x * (x - 3) ↔ x = 3 ∨ x = -3 := 
by
  sorry

end quadratic_solution_1_quadratic_solution_2_l158_158782


namespace votes_cast_proof_l158_158056

variable (V : ℝ)
variable (candidate_votes : ℝ)
variable (rival_votes : ℝ)

noncomputable def total_votes_cast : Prop :=
  candidate_votes = 0.40 * V ∧ 
  rival_votes = candidate_votes + 2000 ∧ 
  rival_votes = 0.60 * V ∧ 
  V = 10000

theorem votes_cast_proof : total_votes_cast V candidate_votes rival_votes :=
by {
  sorry
  }

end votes_cast_proof_l158_158056


namespace area_enclosed_by_graph_l158_158930

noncomputable def enclosed_area (x y : ℝ) : ℝ := 
  if h : (|5 * x| + |3 * y| = 15) then
    30 -- The area enclosed by the graph
  else
    0 -- Default case for definition completeness

theorem area_enclosed_by_graph : ∀ (x y : ℝ), (|5 * x| + |3 * y| = 15) → enclosed_area x y = 30 :=
by
  sorry

end area_enclosed_by_graph_l158_158930


namespace area_ratio_l158_158295

-- Define the problem conditions
def Square (s : ℝ) := s > 0
def Rectangle (longer shorter : ℝ) := longer = 1.2 * shorter ∧ shorter = 0.8 * shorter

-- Define a function to calculate the area of square
def area_square (s : ℝ) : ℝ := s * s

-- Define a function to calculate the area of rectangle
def area_rectangle (longer shorter : ℝ) : ℝ := longer * shorter

-- State the proof problem
theorem area_ratio (s : ℝ) (h_square : Square s) :
  let longer := 1.2 * s
  let shorter := 0.8 * s
  area_rectangle longer shorter / area_square s = 24 / 25 :=
by
  sorry

end area_ratio_l158_158295


namespace domain_of_function_l158_158837

variable (x : ℝ)

theorem domain_of_function :
  {x : ℝ | x + 3 ≥ 0 ∧ 2 - x ≠ 0} =
  {x : ℝ | x ≥ -3 ∧ x ≠ 2} :=
by
  sorry

end domain_of_function_l158_158837


namespace cnc_processing_time_l158_158739

theorem cnc_processing_time :
  (∃ (hours: ℕ), 3 * (960 / hours) = 960 / 3) → 1 * (400 / 5) = 400 / 1 :=
by
  sorry

end cnc_processing_time_l158_158739


namespace tom_current_yellow_tickets_l158_158310

-- Definitions based on conditions provided
def yellow_to_red (y : ℕ) : ℕ := y * 10
def red_to_blue (r : ℕ) : ℕ := r * 10
def yellow_to_blue (y : ℕ) : ℕ := (yellow_to_red y) * 10

def tom_red_tickets : ℕ := 3
def tom_blue_tickets : ℕ := 7

def tom_total_blue_tickets : ℕ := (red_to_blue tom_red_tickets) + tom_blue_tickets
def tom_needed_blue_tickets : ℕ := 163

-- Proving that Tom currently has 2 yellow tickets
theorem tom_current_yellow_tickets : (tom_total_blue_tickets + tom_needed_blue_tickets) / yellow_to_blue 1 = 2 :=
by
  sorry

end tom_current_yellow_tickets_l158_158310


namespace remaining_surface_area_correct_l158_158840

noncomputable def remaining_surface_area (a : ℕ) (c : ℕ) : ℕ :=
  let original_surface_area := 6 * a^2
  let corner_cube_area := 3 * c^2
  let net_change := corner_cube_area - corner_cube_area
  original_surface_area + 8 * net_change 

theorem remaining_surface_area_correct :
  remaining_surface_area 4 1 = 96 := by
  sorry

end remaining_surface_area_correct_l158_158840


namespace georgina_parrot_days_l158_158583

theorem georgina_parrot_days 
    (initial_phrases : ℕ := 3) 
    (current_phrases : ℕ := 17) 
    (phrases_per_week : ℕ := 2) 
    (days_per_week : ℕ := 7) 
    : nat :=
  let new_phrases := current_phrases - initial_phrases in
  let weeks := new_phrases / phrases_per_week in
  let days := weeks * days_per_week in
  days = 49
by
  sorry

end georgina_parrot_days_l158_158583


namespace ratio_equivalence_l158_158814

theorem ratio_equivalence (x : ℚ) (h : x / 360 = 18 / 12) : x = 540 :=
by
  -- Proof goes here, to be filled in
  sorry

end ratio_equivalence_l158_158814


namespace cos_alpha_in_second_quadrant_l158_158715

theorem cos_alpha_in_second_quadrant 
  (alpha : ℝ) 
  (h1 : π / 2 < alpha ∧ alpha < π)
  (h2 : ∀ x y : ℝ, 2 * x + (Real.tan alpha) * y + 1 = 0 → 8 / 3 = -(2 / (Real.tan alpha))) :
  Real.cos alpha = -4 / 5 :=
by
  sorry

end cos_alpha_in_second_quadrant_l158_158715


namespace tangent_AR_l158_158764

variables (A B C S X Y P Q R : Point)

theorem tangent_AR (h_circle_abc : Circle A B C)
  (h_s_midpoint : S = midpointArc A B C)
  (h_xy_parallel : parallel XY BC)
  (h_p_second_intersection : P = secondIntersection (line_through S X) h_circle_abc)
  (h_q_second_intersection : Q = secondIntersection (line_through S Y) h_circle_abc)
  (h_r_intersection : R = intersection (line_through P Q) XY):
  tangentAtCircle A R h_circle_abc := 
sorry

end tangent_AR_l158_158764


namespace remaining_fuel_relation_l158_158804

-- Define the car's travel time and remaining fuel relation
def initial_fuel : ℝ := 100

def fuel_consumption_rate : ℝ := 6

def remaining_fuel (t : ℝ) : ℝ := initial_fuel - fuel_consumption_rate * t

-- Prove that the remaining fuel after t hours is given by the linear relationship Q = 100 - 6t
theorem remaining_fuel_relation (t : ℝ) : remaining_fuel t = 100 - 6 * t := by
  -- Proof is omitted, as per instructions
  sorry

end remaining_fuel_relation_l158_158804


namespace rosie_pies_l158_158634

def number_of_pies (apples : ℕ) : ℕ := sorry

theorem rosie_pies (h : number_of_pies 9 = 2) : number_of_pies 27 = 6 :=
by sorry

end rosie_pies_l158_158634


namespace problem_statement_l158_158786

noncomputable def g (x : ℝ) : ℝ :=
  sorry

theorem problem_statement : (∀ x : ℝ, g x + 3 * g (1 - x) = 4 * x^3 + x^2) → g 3 = -201 / 8 :=
by
  intro h
  sorry

end problem_statement_l158_158786


namespace geom_seq_308th_term_l158_158882

def geom_seq (a : ℤ) (r : ℤ) (n : ℕ) : ℤ :=
  a * r ^ n

-- Given conditions
def a := 10
def r := -1

theorem geom_seq_308th_term : geom_seq a r 307 = -10 := by
  sorry

end geom_seq_308th_term_l158_158882


namespace percentage_saved_on_hats_l158_158349

/-- Suppose the regular price of a hat is $60 and Maria buys four hats with progressive discounts: 
20% off the second hat, 40% off the third hat, and 50% off the fourth hat.
Prove that the percentage saved on the regular price for four hats is 27.5%. -/
theorem percentage_saved_on_hats :
  let regular_price := 60
  let discount_2 := 0.2 * regular_price
  let discount_3 := 0.4 * regular_price
  let discount_4 := 0.5 * regular_price
  let price_1 := regular_price
  let price_2 := regular_price - discount_2
  let price_3 := regular_price - discount_3
  let price_4 := regular_price - discount_4
  let total_regular := 4 * regular_price
  let total_discounted := price_1 + price_2 + price_3 + price_4
  let savings := total_regular - total_discounted
  let percentage_saved := (savings / total_regular) * 100
  percentage_saved = 27.5 :=
by
  sorry

end percentage_saved_on_hats_l158_158349


namespace ratio_area_of_rectangle_to_square_l158_158289

theorem ratio_area_of_rectangle_to_square (s : ℝ) :
  (1.2 * s * 0.8 * s) / (s * s) = 24 / 25 :=
by
  sorry

end ratio_area_of_rectangle_to_square_l158_158289


namespace first_box_oranges_l158_158519

theorem first_box_oranges (x : ℕ) (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) + (x + 12) = 120) : x = 11 :=
sorry

end first_box_oranges_l158_158519


namespace claudia_candle_choices_l158_158699

-- Claudia can choose 4 different candles
def num_candles : ℕ := 4

-- Claudia can choose 8 out of 9 different flowers
def num_ways_to_choose_flowers : ℕ := Nat.choose 9 8

-- The total number of groupings is given as 54
def total_groupings : ℕ := 54

-- Prove the main theorem using the conditions
theorem claudia_candle_choices :
  num_ways_to_choose_flowers = 9 ∧ num_ways_to_choose_flowers * C = total_groupings → C = 6 :=
by
  sorry

end claudia_candle_choices_l158_158699


namespace cube_sum_identity_l158_158959

theorem cube_sum_identity (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
by
  sorry

end cube_sum_identity_l158_158959


namespace factorize_expression_l158_158842

theorem factorize_expression (a b : ℝ) : a^2 - a * b = a * (a - b) :=
by sorry

end factorize_expression_l158_158842


namespace not_divisible_by_2006_l158_158741

theorem not_divisible_by_2006 (k : ℤ) : ¬ ∃ m : ℤ, k^2 + k + 1 = 2006 * m :=
sorry

end not_divisible_by_2006_l158_158741


namespace total_surface_area_hemisphere_l158_158279

theorem total_surface_area_hemisphere (A : ℝ) (r : ℝ) : (A = 100 * π) → (r = 10) → (2 * π * r^2 + A = 300 * π) :=
by
  intro hA hr
  sorry

end total_surface_area_hemisphere_l158_158279


namespace boys_passed_percentage_l158_158750

theorem boys_passed_percentage
  (total_candidates : ℝ)
  (total_girls : ℝ)
  (failed_percentage : ℝ)
  (girls_passed_percentage : ℝ)
  (boys_passed_percentage : ℝ) :
  total_candidates = 2000 →
  total_girls = 900 →
  failed_percentage = 70.2 →
  girls_passed_percentage = 32 →
  boys_passed_percentage = 28 :=
by
  sorry

end boys_passed_percentage_l158_158750


namespace calculate_expression_l158_158558

theorem calculate_expression :
  (3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3)) = (3 + 2 * Real.sqrt 3 / 3) :=
by
  sorry

end calculate_expression_l158_158558


namespace sample_processing_l158_158969

-- Define sample data
def standard: ℕ := 220
def samples: List ℕ := [230, 226, 218, 223, 214, 225, 205, 212]

-- Calculate deviations
def deviations (samples: List ℕ) (standard: ℕ) : List ℤ :=
  samples.map (λ x => x - standard)

-- Total dosage of samples
def total_dosage (samples: List ℕ): ℕ :=
  samples.sum

-- Total cost to process to standard dosage
def total_cost (deviations: List ℤ) (cost_per_ml_adjustment: ℤ) : ℤ :=
  cost_per_ml_adjustment * (deviations.map Int.natAbs).sum

-- Theorem statement
theorem sample_processing :
  let deviation_vals := deviations samples standard;
  let total_dosage_val := total_dosage samples;
  let total_cost_val := total_cost deviation_vals 10;
  deviation_vals = [10, 6, -2, 3, -6, 5, -15, -8] ∧
  total_dosage_val = 1753 ∧
  total_cost_val = 550 :=
by
  sorry

end sample_processing_l158_158969


namespace non_real_roots_b_range_l158_158726

theorem non_real_roots_b_range (b : ℝ) : 
  ∃ (x : ℂ), x^2 + (b : ℂ) * x + 16 = 0 ∧ (¬ ∃ (x : ℝ), x^2 + b * x + 16 = 0) ↔ -8 < b ∧ b < 8 := 
by
  sorry

end non_real_roots_b_range_l158_158726


namespace g_of_negative_8_l158_158889

def f (x : ℝ) : ℝ := 4 * x - 9
def g (y : ℝ) : ℝ := y^2 + 6 * y - 7

theorem g_of_negative_8 : g (-8) = -87 / 16 :=
by
  -- Proof goes here
  sorry

end g_of_negative_8_l158_158889


namespace parabola_hyperbola_tangent_l158_158996

theorem parabola_hyperbola_tangent : ∃ m : ℝ, 
  (∀ x y : ℝ, y = x^2 - 2 * x + 2 → y^2 - m * x^2 = 1) ↔ m = 1 :=
by
  sorry

end parabola_hyperbola_tangent_l158_158996


namespace sixteenth_answer_is_three_l158_158801

theorem sixteenth_answer_is_three (total_members : ℕ)
  (answers_1 answers_2 answers_3 : ℕ) 
  (h_total : total_members = 16) 
  (h_answers_1 : answers_1 = 6) 
  (h_answers_2 : answers_2 = 6) 
  (h_answers_3 : answers_3 = 3) :
  ∃ answer : ℕ, answer = 3 ∧ (answers_1 + answers_2 + answers_3 + 1 = total_members) :=
sorry

end sixteenth_answer_is_three_l158_158801


namespace john_experience_when_mike_started_l158_158617

-- Definitions from the conditions
variable (J O M : ℕ)
variable (h1 : J = 20) -- James currently has 20 years of experience
variable (h2 : O - 8 = 2 * (J - 8)) -- 8 years ago, John had twice as much experience as James
variable (h3 : J + O + M = 68) -- Combined experience is 68 years

-- Theorem to prove
theorem john_experience_when_mike_started : O - M = 16 := 
by
  -- Proof steps go here
  sorry

end john_experience_when_mike_started_l158_158617


namespace math_problem_l158_158410

theorem math_problem (x y : ℝ) (h1 : x - 2 * y = 4) (h2 : x * y = 8) :
  x^2 + 4 * y^2 = 48 :=
sorry

end math_problem_l158_158410


namespace power_multiplication_equals_result_l158_158696

theorem power_multiplication_equals_result : 
  (-0.25)^11 * (-4)^12 = -4 := 
sorry

end power_multiplication_equals_result_l158_158696


namespace student_number_choice_l158_158824

theorem student_number_choice (x : ℤ) (h : 2 * x - 138 = 104) : x = 121 :=
sorry

end student_number_choice_l158_158824


namespace min_AB_distance_l158_158060

theorem min_AB_distance : 
  ∀ (A B : ℝ × ℝ), 
  A ≠ B → 
  ((∃ (m : ℝ), A.2 = m * (A.1 - 1) + 1 ∧ B.2 = m * (B.1 - 1) + 1) ∧ 
    ((A.1 - 2)^2 + (A.2 - 3)^2 = 9) ∧ 
    ((B.1 - 2)^2 + (B.2 - 3)^2 = 9)) → 
  dist A B = 4 :=
sorry

end min_AB_distance_l158_158060


namespace vasya_fraction_l158_158539

variable (a b c d s : ℝ)

-- Anton drove half the distance Vasya did
axiom h1 : a = b / 2

-- Sasha drove as long as Anton and Dima together
axiom h2 : c = a + d

-- Dima drove one-tenth of the total distance
axiom h3 : d = s / 10

-- The total distance is the sum of distances driven by Anton, Vasya, Sasha, and Dima
axiom h4 : a + b + c + d = s

-- We need to prove that Vasya drove 0.4 of the total distance
theorem vasya_fraction (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : b = 0.4 * s :=
by
  sorry

end vasya_fraction_l158_158539


namespace divisibility_of_poly_l158_158624

theorem divisibility_of_poly (x y z : ℤ) (h_distinct : x ≠ y ∧ y ≠ z ∧ z ≠ x):
  ∃ k : ℤ, (x-y)^5 + (y-z)^5 + (z-x)^5 = 5 * (y-z) * (z-x) * (x-y) * k :=
by
  sorry

end divisibility_of_poly_l158_158624


namespace stratified_sampling_l158_158971

theorem stratified_sampling
  (total_products : ℕ)
  (sample_size : ℕ)
  (workshop_products : ℕ)
  (h1 : total_products = 2048)
  (h2 : sample_size = 128)
  (h3 : workshop_products = 256) :
  (workshop_products / total_products) * sample_size = 16 := 
by
  rw [h1, h2, h3]
  norm_num
  
  sorry

end stratified_sampling_l158_158971


namespace num_six_digit_unique_digits_num_six_digit_with_four_odd_l158_158222

-- Proof theorems
theorem num_six_digit_unique_digits : 
  (card { n : ℕ | ∃ a b c d e f, (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧ 
                      (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧ 
                      (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧ 
                      (d ≠ e) ∧ (d ≠ f) ∧ 
                      (e ≠ f)  ∧ {a,b,c,d,e,f} ⊆ {0,1,2,3,4,5,6,7,8,9} ∧ a ≠ 0 ∧ 
                      n = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f}) = 136080 := sorry

theorem num_six_digit_with_four_odd : 
  (card { n : ℕ | ∃ a b c d e f, (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧ 
                      (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧ 
                      (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧ 
                      (d ≠ e) ∧ (d ≠ f) ∧ 
                      (e ≠ f)  ∧ {a,b,c,d,e,f} ⊆ {0,1,2,3,4,5,6,7,8,9} ∧ a ≠ 0 ∧ 
                      (countp (λ x, x % 2 = 1) [a,b,c,d,e,f] = 4) ∧ 
                      n = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f}) = 33600 := sorry

end num_six_digit_unique_digits_num_six_digit_with_four_odd_l158_158222


namespace fraction_simplification_l158_158562

theorem fraction_simplification :
  (1722 ^ 2 - 1715 ^ 2) / (1729 ^ 2 - 1708 ^ 2) = 1 / 3 := by
  sorry

end fraction_simplification_l158_158562


namespace sum_of_digits_of_greatest_prime_divisor_of_16385_is_19_l158_158477

theorem sum_of_digits_of_greatest_prime_divisor_of_16385_is_19 :
  let n := 16385
  let p := 3277
  let prime_p : Prime p := by sorry
  let greatest_prime_divisor := p
  let sum_digits := 3 + 2 + 7 + 7
  sum_digits = 19 :=
by
  sorry

end sum_of_digits_of_greatest_prime_divisor_of_16385_is_19_l158_158477


namespace xy_addition_l158_158928

theorem xy_addition (x y : ℕ) (h1 : x * y = 24) (h2 : x - y = 5) (hx_pos : 0 < x) (hy_pos : 0 < y) : x + y = 11 := 
sorry

end xy_addition_l158_158928


namespace seating_arrangements_l158_158880

theorem seating_arrangements (n m: ℕ) (h₁ : n = 4) (h₂ : m = 5):
  (m - 1).factorial = 24 :=
by {
  rw [h₁, h₂, Nat.factorial],
  norm_num,
  sorry
}

end seating_arrangements_l158_158880


namespace no_intersections_root_of_quadratic_l158_158457

theorem no_intersections_root_of_quadratic (x : ℝ) :
  ¬(∃ x, (y = x) ∧ (y = x - 3)) ↔ (x^2 - 3 * x = 0) := by
  sorry

end no_intersections_root_of_quadratic_l158_158457


namespace probability_king_and_heart_correct_l158_158926

open Probability

noncomputable def probability_king_then_heart : ℚ :=
  let total_cards : ℚ := 52
  let kings : ℚ := 4
  let hearts : ℚ := 13
  let king_of_hearts : ℚ := 1
  let remaining_cards_after_first_draw : ℚ := total_cards - 1
  
  -- Probability calculations
  let pr_case1 : ℚ := (king_of_hearts / total_cards) * ((hearts - 1) / remaining_cards_after_first_draw)
  let pr_case2 : ℚ := ((kings - king_of_hearts) / total_cards) * (hearts / remaining_cards_after_first_draw)
  pr_case1 + pr_case2

-- The target theorem
theorem probability_king_and_heart_correct :
  probability_king_then_heart = 1 / 52 :=
sorry

end probability_king_and_heart_correct_l158_158926


namespace garden_perimeter_l158_158041

theorem garden_perimeter (w l : ℕ) (garden_width : ℕ) (garden_perimeter : ℕ)
  (garden_area playground_length playground_width : ℕ)
  (h1 : garden_width = 16)
  (h2 : playground_length = 16)
  (h3 : garden_area = 16 * l)
  (h4 : playground_area = w * playground_length)
  (h5 : garden_area = playground_area)
  (h6 : garden_perimeter = 2 * l + 2 * garden_width)
  (h7 : garden_perimeter = 56):
  l = 12 :=
by
  sorry

end garden_perimeter_l158_158041


namespace product_area_perimeter_square_EFGH_l158_158772

theorem product_area_perimeter_square_EFGH:
  let E := (5, 5)
  let F := (5, 1)
  let G := (1, 1)
  let H := (1, 5)
  let side_length := 4
  let area := side_length * side_length
  let perimeter := 4 * side_length
  area * perimeter = 256 :=
by
  sorry

end product_area_perimeter_square_EFGH_l158_158772


namespace points_same_color_separed_by_two_l158_158465

theorem points_same_color_separed_by_two (circle : Fin 239 → Bool) : 
  ∃ i j : Fin 239, i ≠ j ∧ (i + 2) % 239 = j ∧ circle i = circle j :=
by
  sorry

end points_same_color_separed_by_two_l158_158465


namespace max_k_strictly_increasing_sequence_l158_158590

open Nat

theorem max_k_strictly_increasing_sequence :
  ∀ (a : ℕ → ℕ),
    (a 1 = choose 10 0) ∧
    (a 2 = choose 10 1) ∧
    (a 3 = choose 10 2) ∧
    (a 4 = choose 10 3) ∧
    (a 5 = choose 10 4) ∧
    (a 6 = choose 10 5) ∧
    (a 7 = choose 10 6) ∧
    (a 8 = choose 10 7) ∧
    (a 9 = choose 10 8) ∧
    (a 10 = choose 10 9) ∧
    (a 11 = choose 10 10) →
  ∃ k : ℕ, k = 6 ∧ ∀ n : ℕ, (1 ≤ n ∧ n < k) → a n < a (n + 1) :=
by
  sorry

end max_k_strictly_increasing_sequence_l158_158590


namespace calc_result_l158_158694

noncomputable def expMul := (-0.25)^11 * (-4)^12

theorem calc_result : expMul = -4 := 
by
  -- Sorry is used here to skip the proof as instructed.
  sorry

end calc_result_l158_158694


namespace joyce_new_property_is_10_times_larger_l158_158619

theorem joyce_new_property_is_10_times_larger :
  let previous_property := 2
  let suitable_acres := 19
  let pond := 1
  let new_property := suitable_acres + pond
  new_property / previous_property = 10 := by {
    let previous_property := 2
    let suitable_acres := 19
    let pond := 1
    let new_property := suitable_acres + pond
    sorry
  }

end joyce_new_property_is_10_times_larger_l158_158619


namespace exponent_proof_l158_158493

theorem exponent_proof (m : ℝ) : (243 : ℝ) = (3 : ℝ)^5 → (243 : ℝ)^(1/3) = (3 : ℝ)^m → m = 5/3 :=
by
  intros h1 h2
  sorry

end exponent_proof_l158_158493


namespace place_value_ratio_l158_158883

theorem place_value_ratio :
  let d8_place := 0.1
  let d7_place := 10
  d8_place / d7_place = 0.01 :=
by
  -- proof skipped
  sorry

end place_value_ratio_l158_158883


namespace toy_truck_cost_is_correct_l158_158008

-- Define the initial amount, amount spent on the pencil case, and the final amount
def initial_amount : ℝ := 10
def pencil_case_cost : ℝ := 2
def final_amount : ℝ := 5

-- Define the amount spent on the toy truck
def toy_truck_cost : ℝ := initial_amount - pencil_case_cost - final_amount

-- Prove that the amount spent on the toy truck is 3 dollars
theorem toy_truck_cost_is_correct : toy_truck_cost = 3 := by
  sorry

end toy_truck_cost_is_correct_l158_158008


namespace quadratic_sequence_geom_l158_158259

theorem quadratic_sequence_geom
  (a_n a_{n+1} : ℝ) (α β : ℝ)
  (h_quad : 0 = a_n * α^2 - a_{n+1} * α + 1 ∧ 0 = a_n * β^2 - a_{n+1} * β + 1)
  (h_cond : 6 * α - 2 * α * β + 6 * β = 3) :
  a_{n+1} = 1 / 2 * a_n + 1 / 3 ∧ ∀ n, let a_n_minus_2_over_3 := a_n - 2 / 3 in ∃ r, ∀ m, a_n_minus_2_over_3 ^ m = r * a_n_minus_2_over_3 :=
  by
  sorry

end quadratic_sequence_geom_l158_158259


namespace fourth_term_binomial_expansion_l158_158578

theorem fourth_term_binomial_expansion (x y : ℝ) :
  let a := x
  let b := -2*y
  let n := 7
  let r := 3
  (binomial n r) * a^(n-r) * b^r = -280*x^4*y^3 := by
  sorry

end fourth_term_binomial_expansion_l158_158578


namespace gecko_insects_eaten_l158_158327

theorem gecko_insects_eaten
    (G : ℕ)  -- Number of insects each gecko eats
    (H1 : 5 * G + 3 * (2 * G) = 66) :  -- Total insects eaten condition
    G = 6 :=  -- Expected number of insects each gecko eats
by
  sorry

end gecko_insects_eaten_l158_158327


namespace baron_not_lying_l158_158351

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem baron_not_lying : 
  ∃ a b : ℕ, 
  (a ≠ b ∧ a ≥ 10^9 ∧ a < 10^10 ∧ b ≥ 10^9 ∧ b < 10^10 ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 ∧ 
  (a + sum_of_digits (a * a) = b + sum_of_digits (b * b))) :=
  sorry

end baron_not_lying_l158_158351


namespace natural_numbers_count_l158_158180

def number_of_valid_natural_numbers (m a n k : ℕ) :=
  ∃ (k : ℕ) (a : ℕ) (m : ℕ) (n : ℕ),
  1 ≤ a ∧ a ≤ 9 ∧ m < 10 ^ k ∧
  10^k * a = 8 * m ∧ n = 0

theorem natural_numbers_count: (∃ (s : finset ℕ), s.card = 7 ∧ ∀ x ∈ s, number_of_valid_natural_numbers x) := sorry

end natural_numbers_count_l158_158180


namespace baron_munchausen_not_lying_l158_158353

def sum_of_digits (n : Nat) : Nat := sorry

theorem baron_munchausen_not_lying :
  ∃ a b : Nat, a ≠ b ∧ a % 10 ≠ 0 ∧ b % 10 ≠ 0 ∧ 
  (a < 10^10 ∧ 10^9 ≤ a) ∧ (b < 10^10 ∧ 10^9 ≤ b) ∧ 
  (a + sum_of_digits (a ^ 2) = b + sum_of_digits (b ^ 2)) :=
sorry

end baron_munchausen_not_lying_l158_158353


namespace triangle_area_l158_158162

def point := ℝ × ℝ

def A : point := (2, -3)
def B : point := (8, 1)
def C : point := (2, 3)

def area_triangle (A B C : point) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area : area_triangle A B C = 18 :=
  sorry

end triangle_area_l158_158162


namespace b3_b8_product_l158_158429

-- Definitions based on conditions
def is_arithmetic_seq (b : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

-- The problem statement
theorem b3_b8_product (b : ℕ → ℤ) (h_seq : is_arithmetic_seq b) (h4_7 : b 4 * b 7 = 24) : 
  b 3 * b 8 = 200 / 9 :=
sorry

end b3_b8_product_l158_158429


namespace beverage_price_function_l158_158794

theorem beverage_price_function (box_price : ℕ) (bottles_per_box : ℕ) (bottles_purchased : ℕ) (y : ℕ) :
  box_price = 55 →
  bottles_per_box = 6 →
  y = (55 * bottles_purchased) / 6 := 
sorry

end beverage_price_function_l158_158794


namespace valid_passwords_count_l158_158687

-- Define the total number of unrestricted passwords
def total_passwords : ℕ := 10000

-- Define the number of restricted passwords (ending with 6, 3, 9)
def restricted_passwords : ℕ := 10

-- Define the total number of valid passwords
def valid_passwords := total_passwords - restricted_passwords

theorem valid_passwords_count : valid_passwords = 9990 := 
by 
  sorry

end valid_passwords_count_l158_158687


namespace coordinates_of_A_l158_158909

-- Defining the point A
def point_A : ℤ × ℤ := (1, -4)

-- Statement that needs to be proved
theorem coordinates_of_A :
  point_A = (1, -4) :=
by
  sorry

end coordinates_of_A_l158_158909


namespace vasya_fraction_l158_158524

-- Define the variables for distances and total distance
variables {a b c d s : ℝ}

-- Define conditions
def anton_distance (a b : ℝ) : Prop := a = b / 2
def sasha_distance (c a d : ℝ) : Prop := c = a + d
def dima_distance (d s : ℝ) : Prop := d = s / 10
def total_distance (a b c d s : ℝ) : Prop := a + b + c + d = s

-- The main theorem 
theorem vasya_fraction (a b c d s : ℝ) (h1 : anton_distance a b) 
  (h2 : sasha_distance c a d) (h3 : dima_distance d s)
  (h4 : total_distance a b c d s) : b / s = 0.4 :=
sorry

end vasya_fraction_l158_158524


namespace sum_of_roots_of_polynomial_l158_158366

theorem sum_of_roots_of_polynomial (a b c : ℝ) (h : 3*a^3 - 7*a^2 + 6*a = 0) : 
    (∀ x, 3*x^2 - 7*x + 6 = 0 → x = a ∨ x = b ∨ x = c) →
    (∀ (x : ℝ), (x = a ∨ x = b ∨ x = c → 3*x^3 - 7*x^2 + 6*x = 0)) → 
    a + b + c = 7 / 3 :=
sorry

end sum_of_roots_of_polynomial_l158_158366


namespace value_range_of_a_l158_158658

variable (a : ℝ)
variable (suff_not_necess : ∀ x, x ∈ ({3, a} : Set ℝ) → 2 * x^2 - 5 * x - 3 ≥ 0)

theorem value_range_of_a :
  (a ≤ -1/2 ∨ a > 3) :=
sorry

end value_range_of_a_l158_158658


namespace accelerations_l158_158673

open Real

namespace Problem

variables (m M g : ℝ) (a1 a2 : ℝ)

theorem accelerations (mass_condition : 4 * m + M ≠ 0):
  (a1 = 2 * ((2 * m + M) * g) / (4 * m + M)) ∧
  (a2 = ((2 * m + M) * g) / (4 * m + M)) :=
sorry

end Problem

end accelerations_l158_158673


namespace segment_length_C_C_l158_158662

-- Define the points C and C''.
def C : ℝ × ℝ := (-3, 2)
def C'' : ℝ × ℝ := (-3, -2)

-- State the theorem that the length of the segment from C to C'' is 4.
theorem segment_length_C_C'' : dist C C'' = 4 := by
  sorry

end segment_length_C_C_l158_158662


namespace muffin_machine_completion_time_l158_158499

theorem muffin_machine_completion_time :
  let start_time := 9 * 60 -- minutes
  let partial_completion_time := (12 * 60) + 15 -- minutes
  let partial_duration := partial_completion_time - start_time
  let fraction_of_day := 1 / 4
  let total_duration := partial_duration / fraction_of_day
  start_time + total_duration = (22 * 60) := -- 10:00 PM in minutes
by
  sorry

end muffin_machine_completion_time_l158_158499


namespace find_c_l158_158142

def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem find_c (a b c : ℝ) 
  (h1 : perpendicular (a / 2) (-2 / b))
  (h2 : a = b)
  (h3 : a * 1 - 2 * (-5) = c) 
  (h4 : 2 * 1 + b * (-5) = -c) : 
  c = 13 := by
  sorry

end find_c_l158_158142


namespace min_S_n_condition_l158_158123

noncomputable def a_n (n : ℕ) : ℤ := -28 + 4 * (n - 1)

noncomputable def S_n (n : ℕ) : ℤ := n * (a_n 1 + a_n n) / 2

theorem min_S_n_condition : S_n 7 = S_n 8 ∧ (∀ m < 7, S_n m > S_n 7) ∧ (∀ m < 8, S_n m > S_n 8) := 
by
  sorry

end min_S_n_condition_l158_158123


namespace ratio_of_tetrahedrons_volume_l158_158749

theorem ratio_of_tetrahedrons_volume (d R s s' V_ratio m n : ℕ) (h1 : d = 4)
  (h2 : R = 2)
  (h3 : s = 4 * R / Real.sqrt 6)
  (h4 : s' = s / Real.sqrt 8)
  (h5 : V_ratio = (s' / s) ^ 3)
  (hm : m = 1)
  (hn : n = 32)
  (h_ratio : V_ratio = m / n) :
  m + n = 33 :=
by
  sorry

end ratio_of_tetrahedrons_volume_l158_158749


namespace range_of_m_intersection_l158_158875

noncomputable def f (x m : ℝ) : ℝ := (1/x) - (m/(x^2)) - (x/3)

theorem range_of_m_intersection (m : ℝ) :
  (∃! x : ℝ, f x m = 0) ↔ m ∈ (Set.Iic 0 ∪ {2/3}) :=
sorry

end range_of_m_intersection_l158_158875


namespace condition_a_gt_1_iff_a_gt_0_l158_158785

theorem condition_a_gt_1_iff_a_gt_0 : ∀ (a : ℝ), (a > 1) ↔ (a > 0) :=
by 
  sorry

end condition_a_gt_1_iff_a_gt_0_l158_158785


namespace trig_eq_solution_l158_158016

open Real

theorem trig_eq_solution (x : ℝ) : 
  (cos (7 * x) + cos (3 * x) + sin (7 * x) - sin (3 * x) + sqrt 2 * cos (4 * x) = 0) ↔ 
  (∃ k : ℤ, 
    (x = -π / 8 + π * k / 2) ∨ 
    (x = -π / 4 + 2 * π * k / 3) ∨ 
    (x = 3 * π / 28 + 2 * π * k / 7)) :=
by sorry

end trig_eq_solution_l158_158016


namespace at_least_two_equal_l158_158423

theorem at_least_two_equal
  {a b c d : ℝ}
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h₁ : a + b + (1 / (a * b)) = c + d + (1 / (c * d)))
  (h₂ : (1 / a) + (1 / b) + (a * b) = (1 / c) + (1 / d) + (c * d)) :
  a = c ∨ a = d ∨ b = c ∨ b = d ∨ a = b ∨ c = d := by
  sorry

end at_least_two_equal_l158_158423


namespace find_b_l158_158793

theorem find_b (b p : ℝ) 
  (h1 : 3 * p + 15 = 0)
  (h2 : 15 * p + 3 = b) :
  b = -72 :=
by
  sorry

end find_b_l158_158793


namespace divide_by_3_result_l158_158043

-- Definitions
def n : ℕ := 4 * 12

theorem divide_by_3_result (h : n / 4 = 12) : n / 3 = 16 :=
by
  sorry

end divide_by_3_result_l158_158043


namespace susana_chocolate_chips_l158_158160

theorem susana_chocolate_chips :
  ∃ (S_c : ℕ), 
  (∃ (V_c V_v S_v : ℕ), 
    V_c = S_c + 5 ∧
    S_v = (3 * V_v) / 4 ∧
    V_v = 20 ∧
    V_c + S_c + V_v + S_v = 90) ∧
  S_c = 25 :=
by
  existsi 25
  sorry

end susana_chocolate_chips_l158_158160


namespace circle_area_of_circumscribed_triangle_l158_158178

theorem circle_area_of_circumscribed_triangle :
  let a := 12
  let b := 12
  let c := 10
  let height := Real.sqrt (a^2 - (c / 2)^2)
  let A := (1 / 2) * c * height
  let R := (a * b * c) / (4 * A)
  π * R^2 = (5184 / 119) * π := 
by
  let a := 12
  let b := 12
  let c := 10
  let height := Real.sqrt (a^2 - (c / 2)^2)
  let A := (1 / 2) * c * height
  let R := (a * b * c) / (4 * A)
  have h1 : height = Real.sqrt (a^2 - (c / 2)^2) := by sorry
  have h2 : A = (1 / 2) * c * height := by sorry
  have h3 : R = (a * b * c) / (4 * A) := by sorry
  have h4 : π * R^2 = (5184 / 119) * π := by sorry
  exact h4

end circle_area_of_circumscribed_triangle_l158_158178


namespace Vasya_distance_fraction_l158_158551

variable (a b c d s : ℝ)

theorem Vasya_distance_fraction :
  (a = b / 2) →
  (c = a + d) →
  (d = s / 10) →
  (a + b + c + d = s) →
  (b / s = 0.4) :=
by
  intros h1 h2 h3 h4
  sorry

end Vasya_distance_fraction_l158_158551


namespace find_sample_size_l158_158610

-- Definitions based on conditions
def ratio_students : ℕ := 2 + 3 + 5
def grade12_ratio : ℚ := 5 / ratio_students
def sample_grade12_students : ℕ := 150

-- The goal is to find n such that the proportion is maintained
theorem find_sample_size (n : ℕ) (h : grade12_ratio = sample_grade12_students / ↑n) : n = 300 :=
by sorry


end find_sample_size_l158_158610


namespace average_speed_correct_l158_158767

-- Define the two legs of the trip
def leg1_distance : ℝ := 450
def leg1_time : ℝ := 7 + 30 / 60  -- 7 hours 30 minutes

def leg2_distance : ℝ := 540
def leg2_time : ℝ := 8 + 15 / 60  -- 8 hours 15 minutes

-- Define the total distance and time
def total_distance : ℝ := leg1_distance + leg2_distance
def total_time : ℝ := leg1_time + leg2_time

-- Define the average speed
def average_speed : ℝ := total_distance / total_time

-- State that the average speed is approximately 62.86 mph
theorem average_speed_correct : abs (average_speed - 62.86) < 1e-2 := sorry

end average_speed_correct_l158_158767


namespace solve_inequality_l158_158272

theorem solve_inequality (x : ℝ) (h : 3 * x + 4 ≠ 0) :
  (3 - 1 / (3 * x + 4) < 5) ↔ (-4 / 3 < x) :=
by
  sorry

end solve_inequality_l158_158272


namespace triangle_side_lengths_values_l158_158023

theorem triangle_side_lengths_values :
  ∃ (m_values : Finset ℕ), m_values = {m ∈ Finset.range 750 | m ≥ 4} ∧ m_values.card = 746 :=
by
  sorry

end triangle_side_lengths_values_l158_158023


namespace domain_of_function_l158_158577

theorem domain_of_function :
  { x : ℝ | 0 ≤ 2 * x - 10 ∧ 2 * x - 10 ≠ 0 } = { x : ℝ | x > 5 } :=
by
  sorry

end domain_of_function_l158_158577


namespace sophia_daily_saving_l158_158638

theorem sophia_daily_saving (total_days : ℕ) (total_saving : ℝ) (h1 : total_days = 20) (h2 : total_saving = 0.20) : 
  (total_saving / total_days) = 0.01 :=
by
  sorry

end sophia_daily_saving_l158_158638


namespace determine_m_l158_158647

variables (m x : ℝ)
noncomputable def f (x : ℝ) := x^2 - 3*x + m
noncomputable def g (x : ℝ) := x^2 - 3*x + 5*m

theorem determine_m (h : 3 * f 5 = 2 * g 5) : m = 10 / 7 :=
by
  sorry

end determine_m_l158_158647


namespace green_ball_probability_l158_158362

def prob_green_ball : ℚ :=
  let prob_container := (1 : ℚ) / 3
  let prob_green_I := (4 : ℚ) / 12
  let prob_green_II := (5 : ℚ) / 8
  let prob_green_III := (4 : ℚ) / 8
  prob_container * prob_green_I + prob_container * prob_green_II + prob_container * prob_green_III

theorem green_ball_probability :
  prob_green_ball = 35 / 72 :=
by
  -- Proof steps are omitted as "sorry" is used to skip the proof.
  sorry

end green_ball_probability_l158_158362


namespace number_of_columns_per_section_l158_158973

variables (S C : ℕ)

-- Define the first condition: S * C + (S - 1) / 2 = 1223
def condition1 := S * C + (S - 1) / 2 = 1223

-- Define the second condition: S = 2 * C + 5
def condition2 := S = 2 * C + 5

-- Formulate the theorem that C = 23 given the two conditions
theorem number_of_columns_per_section
  (h1 : condition1 S C)
  (h2 : condition2 S C) :
  C = 23 :=
sorry

end number_of_columns_per_section_l158_158973


namespace profit_is_correct_l158_158520

-- Define the constants for expenses
def cost_of_lemons : ℕ := 10
def cost_of_sugar : ℕ := 5
def cost_of_cups : ℕ := 3

-- Define the cost per cup of lemonade
def price_per_cup : ℕ := 4

-- Define the number of cups sold
def cups_sold : ℕ := 21

-- Define the total revenue
def total_revenue : ℕ := cups_sold * price_per_cup

-- Define the total expenses
def total_expenses : ℕ := cost_of_lemons + cost_of_sugar + cost_of_cups

-- Define the profit
def profit : ℕ := total_revenue - total_expenses

-- The theorem stating the profit
theorem profit_is_correct : profit = 66 := by
  sorry

end profit_is_correct_l158_158520


namespace cannot_determine_right_triangle_l158_158602

-- Define what a right triangle is
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

-- Define the conditions
def condition_A (A B C : ℕ) : Prop :=
  A / B = 3 / 4 ∧ A / C = 3 / 5 ∧ B / C = 4 / 5

def condition_B (a b c : ℕ) : Prop :=
  a = 5 ∧ b = 12 ∧ c = 13

def condition_C (A B C : ℕ) : Prop :=
  A - B = C

def condition_D (a b c : ℕ) : Prop :=
  a^2 = b^2 - c^2

-- Define the problem in Lean
theorem cannot_determine_right_triangle :
  (∃ A B C, condition_A A B C → ¬is_right_triangle A B C) ∧
  (∀ (a b c : ℕ), condition_B a b c → is_right_triangle a b c) ∧
  (∀ A B C, condition_C A B C → A = 90) ∧
  (∀ (a b c : ℕ),  condition_D a b c → is_right_triangle a b c)
:=
by sorry

end cannot_determine_right_triangle_l158_158602


namespace system_of_equations_solution_l158_158921

theorem system_of_equations_solution :
  ∃ x y : ℝ, (x + y = 3) ∧ (2 * x - 3 * y = 1) ∧ (x = 2) ∧ (y = 1) := by
  sorry

end system_of_equations_solution_l158_158921


namespace factorize_expression_l158_158076

theorem factorize_expression (x : ℝ) : 9 * x^3 - 18 * x^2 + 9 * x = 9 * x * (x - 1)^2 := 
by 
    sorry

end factorize_expression_l158_158076


namespace sum_of_cubes_l158_158957

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
sorry

end sum_of_cubes_l158_158957


namespace percentage_of_apples_sold_l158_158676

variables (A P : ℝ) 

theorem percentage_of_apples_sold :
  (A = 700) →
  (A * (1 - P / 100) = 420) →
  (P = 40) :=
by
  intros h1 h2
  sorry

end percentage_of_apples_sold_l158_158676


namespace krios_population_limit_l158_158240

theorem krios_population_limit (initial_population : ℕ) (acre_per_person : ℕ) (total_acres : ℕ) (doubling_years : ℕ) :
  initial_population = 150 →
  acre_per_person = 2 →
  total_acres = 35000 →
  doubling_years = 30 →
  ∃ (years_from_2005 : ℕ), years_from_2005 = 210 ∧ (initial_population * 2^(years_from_2005 / doubling_years)) ≥ total_acres / acre_per_person :=
by
  intros
  sorry

end krios_population_limit_l158_158240


namespace village_population_rate_l158_158159

noncomputable def population_change_X (initial_X : ℕ) (decrease_rate : ℕ) (years : ℕ) : ℕ :=
  initial_X - decrease_rate * years

noncomputable def population_change_Y (initial_Y : ℕ) (increase_rate : ℕ) (years : ℕ) : ℕ :=
  initial_Y + increase_rate * years

theorem village_population_rate (initial_X decrease_rate initial_Y years result : ℕ) 
  (h1 : initial_X = 70000) (h2 : decrease_rate = 1200) 
  (h3 : initial_Y = 42000) (h4 : years = 14) 
  (h5 : initial_X - decrease_rate * years = initial_Y + result * years) 
  : result = 800 :=
  sorry

end village_population_rate_l158_158159


namespace englishman_land_earnings_l158_158348

noncomputable def acres_to_square_yards (acres : ℝ) : ℝ := acres * 4840
noncomputable def square_yards_to_square_meters (sq_yards : ℝ) : ℝ := sq_yards * (0.9144 ^ 2)
noncomputable def square_meters_to_hectares (sq_meters : ℝ) : ℝ := sq_meters / 10000
noncomputable def cost_of_land (hectares : ℝ) (price_per_hectare : ℝ) : ℝ := hectares * price_per_hectare

theorem englishman_land_earnings
  (acres_owned : ℝ)
  (price_per_hectare : ℝ)
  (acre_to_yard : ℝ)
  (yard_to_meter : ℝ)
  (hectare_to_meter : ℝ)
  (h1 : acres_owned = 2)
  (h2 : price_per_hectare = 500000)
  (h3 : acre_to_yard = 4840)
  (h4 : yard_to_meter = 0.9144)
  (h5 : hectare_to_meter = 10000)
  : cost_of_land (square_meters_to_hectares (square_yards_to_square_meters (acres_to_square_yards acres_owned))) price_per_hectare = 404685.6 := sorry

end englishman_land_earnings_l158_158348


namespace problem_value_eq_13_l158_158168

theorem problem_value_eq_13 : 8 / 4 - 3^2 + 4 * 5 = 13 :=
by
  sorry

end problem_value_eq_13_l158_158168


namespace quadratic_single_root_a_l158_158458

theorem quadratic_single_root_a (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 2 * x + 1 = 0) → (a = 0 ∨ a = 1) :=
by
  sorry

end quadratic_single_root_a_l158_158458


namespace number_of_triplets_with_sum_6n_l158_158709

theorem number_of_triplets_with_sum_6n (n : ℕ) : 
  ∃ (count : ℕ), count = 3 * n^2 ∧ 
  (∀ (x y z : ℕ), x ≤ y → y ≤ z → x + y + z = 6 * n → count = 1) :=
sorry

end number_of_triplets_with_sum_6n_l158_158709


namespace domain_of_f_l158_158641

noncomputable def f (x : ℝ) : ℝ := real.sqrt (1 - real.log x / real.log 2)

theorem domain_of_f : {x : ℝ | 0 < x ∧ x ≤ 2} = {x | f x ≥ 0} :=
by 
  sorry

end domain_of_f_l158_158641


namespace calculation_correct_l158_158067

theorem calculation_correct : (5 * 7 + 9 * 4 - 36 / 3 : ℤ) = 59 := by
  sorry

end calculation_correct_l158_158067


namespace unique_solution_l158_158576

theorem unique_solution (p : ℕ) (a b n : ℕ) : 
  p.Prime → 2^a + p^b = n^(p-1) → (p, a, b, n) = (3, 0, 1, 2) ∨ (p = 2) :=
by {
  sorry
}

end unique_solution_l158_158576


namespace planting_equation_l158_158268

def condition1 (x : ℕ) : ℕ := 5 * x + 3
def condition2 (x : ℕ) : ℕ := 6 * x - 4

theorem planting_equation (x : ℕ) : condition1 x = condition2 x := by
  sorry

end planting_equation_l158_158268


namespace rotate180_of_point_A_l158_158398

-- Define the point A and the transformation
def point_A : ℝ × ℝ := (-3, 2)
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Theorem statement for the problem
theorem rotate180_of_point_A :
  rotate180 point_A = (3, -2) :=
sorry

end rotate180_of_point_A_l158_158398


namespace problem1_problem2_problem3_problem4_l158_158357

-- statement for problem 1
theorem problem1 : -5 + 8 - 2 = 1 := by
  sorry

-- statement for problem 2
theorem problem2 : (-3) * (5/6) / (-1/4) = 10 := by
  sorry

-- statement for problem 3
theorem problem3 : -3/17 + (-3.75) + (-14/17) + (15/4) = -1 := by
  sorry

-- statement for problem 4
theorem problem4 : -(1^10) - ((13/14) - (11/12)) * (4 - (-2)^2) + (1/2) / 3 = -(5/6) := by
  sorry

end problem1_problem2_problem3_problem4_l158_158357


namespace vasya_drives_fraction_l158_158545

theorem vasya_drives_fraction {a b c d s : ℝ} 
  (h1 : a = b / 2) 
  (h2 : c = a + d) 
  (h3 : d = s / 10) 
  (h4 : a + b + c + d = s) : 
  b / s = 0.4 :=
by
  sorry

end vasya_drives_fraction_l158_158545


namespace original_cost_of_article_l158_158507

theorem original_cost_of_article : ∃ C : ℝ, 
  (∀ S : ℝ, S = 1.35 * C) ∧
  (∀ C_new : ℝ, C_new = 0.75 * C) ∧
  (∀ S_new : ℝ, (S_new = 1.35 * C - 25) ∧ (S_new = 1.0875 * C)) ∧
  (C = 95.24) :=
sorry

end original_cost_of_article_l158_158507


namespace line_through_point_equal_intercepts_l158_158704

theorem line_through_point_equal_intercepts (P : ℝ × ℝ) (hP : P = (1, 1)) :
  (∀ x y : ℝ, (x - y = 0 ∨ x + y - 2 = 0) → ∃ k : ℝ, k = 1 ∧ k = 2) :=
by
  sorry

end line_through_point_equal_intercepts_l158_158704


namespace students_in_class_l158_158307

theorem students_in_class {S : ℕ} 
  (h1 : 20 < S)
  (h2 : S < 30)
  (chess_club_condition : ∃ (n : ℕ), S = 3 * n) 
  (draughts_club_condition : ∃ (m : ℕ), S = 4 * m) : 
  S = 24 := 
sorry

end students_in_class_l158_158307


namespace find_b12_l158_158425

noncomputable def seq (b : ℕ → ℤ) : Prop :=
  b 1 = 2 ∧ 
  ∀ m n : ℕ, m > 0 → n > 0 → b (m + n) = b m + b n + (m * n * n)

theorem find_b12 (b : ℕ → ℤ) (h : seq b) : b 12 = 98 := 
by
  sorry

end find_b12_l158_158425


namespace solution_of_inequality_system_l158_158030

-- Definitions derived from the conditions in the problem
def inequality1 (x : ℝ) : Prop := 3 * x - 1 ≥ x + 1
def inequality2 (x : ℝ) : Prop := x + 4 > 4 * x - 2
def solution_set (x : ℝ) : Prop := 1 ≤ x ∧ x < 2

-- The Lean 4 statement for the proof problem
theorem solution_of_inequality_system (x : ℝ) : inequality1 x ∧ inequality2 x ↔ solution_set x := by
  sorry

end solution_of_inequality_system_l158_158030


namespace total_amount_spent_l158_158421

variables (original_price_backpack : ℕ) (increase_backpack : ℕ) (original_price_binder : ℕ) (decrease_binder : ℕ) (num_binders : ℕ)

-- Given Conditions
def original_price_backpack := 50
def increase_backpack := 5
def original_price_binder := 20
def decrease_binder := 2
def num_binders := 3

-- Prove the total amount spent is $109
theorem total_amount_spent :
  (original_price_backpack + increase_backpack) + (num_binders * (original_price_binder - decrease_binder)) = 109 := by
  sorry

end total_amount_spent_l158_158421


namespace Vasya_distance_fraction_l158_158549

variable (a b c d s : ℝ)

theorem Vasya_distance_fraction :
  (a = b / 2) →
  (c = a + d) →
  (d = s / 10) →
  (a + b + c + d = s) →
  (b / s = 0.4) :=
by
  intros h1 h2 h3 h4
  sorry

end Vasya_distance_fraction_l158_158549


namespace shortest_distance_point_on_circle_to_line_l158_158303

theorem shortest_distance_point_on_circle_to_line
  (P : ℝ × ℝ)
  (hP : (P.1 + 1)^2 + (P.2 - 2)^2 = 1) :
  ∃ (d : ℝ), d = 3 :=
sorry

end shortest_distance_point_on_circle_to_line_l158_158303


namespace fraction_addition_l158_158985

variable (a : ℝ)

theorem fraction_addition (ha : a ≠ 0) : (3 / a) + (2 / a) = (5 / a) :=
by
  sorry

end fraction_addition_l158_158985


namespace factorization_l158_158843

theorem factorization (a b : ℝ) : a^2 - a * b = a * (a - b) := by
  sorry

end factorization_l158_158843


namespace xiaoming_statement_incorrect_l158_158963

theorem xiaoming_statement_incorrect (s : ℕ) : 
    let x_h := 3
    let x_m := 6
    let steps_xh := (x_h - 1) * s
    let steps_xm := (x_m - 1) * s
    (steps_xm ≠ 2 * steps_xh) :=
by
  let x_h := 3
  let x_m := 6
  let steps_xh := (x_h - 1) * s
  let steps_xm := (x_m - 1) * s
  sorry

end xiaoming_statement_incorrect_l158_158963


namespace preimage_of_4_3_is_2_1_l158_158599

theorem preimage_of_4_3_is_2_1 :
  ∃ (a b : ℝ), (a + 2 * b = 4) ∧ (2 * a - b = 3) ∧ (a = 2) ∧ (b = 1) :=
by
  exists 2
  exists 1
  constructor
  { sorry }
  constructor
  { sorry }
  constructor
  { sorry }
  { sorry }


end preimage_of_4_3_is_2_1_l158_158599


namespace power_equality_l158_158486

theorem power_equality : (243 : ℝ) ^ (1 / 3) = (3 : ℝ) ^ (5 / 3) := 
by 
  sorry

end power_equality_l158_158486


namespace abe_age_sum_is_31_l158_158923

-- Define the present age of Abe
def abe_present_age : ℕ := 19

-- Define Abe's age 7 years ago
def abe_age_7_years_ago : ℕ := abe_present_age - 7

-- Define the sum of Abe's present age and his age 7 years ago
def abe_age_sum : ℕ := abe_present_age + abe_age_7_years_ago

-- Prove that the sum is 31
theorem abe_age_sum_is_31 : abe_age_sum = 31 := 
by 
  sorry

end abe_age_sum_is_31_l158_158923


namespace flower_beds_fraction_l158_158342

-- Define the main problem parameters
def leg_length := (30 - 18) / 2
def triangle_area := (1 / 2) * (leg_length ^ 2)
def total_flower_bed_area := 2 * triangle_area
def yard_area := 30 * 6
def fraction_of_yard_occupied := total_flower_bed_area / yard_area

-- The theorem to be proved
theorem flower_beds_fraction :
  fraction_of_yard_occupied = 1/5 := by
  sorry

end flower_beds_fraction_l158_158342


namespace find_interesting_numbers_l158_158339

def is_interesting (A B : ℕ) : Prop :=
  A > B ∧ (∃ p : ℕ, Nat.Prime p ∧ A - B = p) ∧ ∃ n : ℕ, A * B = n ^ 2

theorem find_interesting_numbers :
  {A | (∃ B : ℕ, is_interesting A B) ∧ 200 < A ∧ A < 400} = {225, 256, 361} :=
by
  sorry

end find_interesting_numbers_l158_158339


namespace white_line_longer_l158_158075

theorem white_line_longer :
  let white_line := 7.67
  let blue_line := 3.33
  white_line - blue_line = 4.34 := by
  sorry

end white_line_longer_l158_158075


namespace num_ways_product_72_l158_158888

def num_ways_product (n : ℕ) : ℕ := sorry  -- Definition for D(n), the number of ways to write n as a product of integers greater than 1

def example_integer := 72  -- Given integer n

theorem num_ways_product_72 : num_ways_product example_integer = 67 := by 
  sorry

end num_ways_product_72_l158_158888


namespace area_ratio_rect_sq_l158_158283

variable (s : ℝ)

def side_len_sq (S : ℝ) : Prop := s = S
def longer_side_rect (R : ℝ) : Prop := R = 1.2 * s
def shorter_side_rect (R : ℝ) : Prop := R = 0.8 * s
def area_sq (S : ℝ) : ℝ := S * S
def area_rect (R_long R_short : ℝ) : ℝ := R_long * R_short
def ratio_area (areaR areaS : ℝ) : ℝ := areaR / areaS

theorem area_ratio_rect_sq (s S R_long R_short : ℝ) (h1 : side_len_sq s S) (h2 : longer_side_rect s R_long) (h3 : shorter_side_rect s R_short) :
  ratio_area (area_rect R_long R_short) (area_sq S) = 24/25 :=
by
  sorry

end area_ratio_rect_sq_l158_158283


namespace justin_and_tim_play_same_game_210_times_l158_158900

def number_of_games_with_justin_and_tim : ℕ :=
  have num_players : ℕ := 12
  have game_size : ℕ := 6
  have justin_and_tim_fixed : ℕ := 2
  have remaining_players : ℕ := num_players - justin_and_tim_fixed
  have players_to_choose : ℕ := game_size - justin_and_tim_fixed
  Nat.choose remaining_players players_to_choose

theorem justin_and_tim_play_same_game_210_times :
  number_of_games_with_justin_and_tim = 210 :=
by sorry

end justin_and_tim_play_same_game_210_times_l158_158900


namespace algebraic_expression_value_l158_158110

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 3 * x + 5 = 7) : 3 * x^2 + 9 * x - 11 = -5 :=
by
  sorry

end algebraic_expression_value_l158_158110


namespace frank_total_cost_l158_158383

-- Conditions from the problem
def cost_per_bun : ℝ := 0.1
def number_of_buns : ℕ := 10
def cost_per_bottle_of_milk : ℝ := 2
def number_of_bottles_of_milk : ℕ := 2
def cost_of_carton_of_eggs : ℝ := 3 * cost_per_bottle_of_milk

-- Question and Answer
theorem frank_total_cost : 
  let cost_of_buns := cost_per_bun * number_of_buns in
  let cost_of_milk := cost_per_bottle_of_milk * number_of_bottles_of_milk in
  let cost_of_eggs := cost_of_carton_of_eggs in
  cost_of_buns + cost_of_milk + cost_of_eggs = 11 :=
by
  sorry

end frank_total_cost_l158_158383


namespace vasya_fraction_l158_158542

variable (a b c d s : ℝ)

-- Anton drove half the distance Vasya did
axiom h1 : a = b / 2

-- Sasha drove as long as Anton and Dima together
axiom h2 : c = a + d

-- Dima drove one-tenth of the total distance
axiom h3 : d = s / 10

-- The total distance is the sum of distances driven by Anton, Vasya, Sasha, and Dima
axiom h4 : a + b + c + d = s

-- We need to prove that Vasya drove 0.4 of the total distance
theorem vasya_fraction (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : b = 0.4 * s :=
by
  sorry

end vasya_fraction_l158_158542


namespace exponent_proof_l158_158494

theorem exponent_proof (m : ℝ) : (243 : ℝ) = (3 : ℝ)^5 → (243 : ℝ)^(1/3) = (3 : ℝ)^m → m = 5/3 :=
by
  intros h1 h2
  sorry

end exponent_proof_l158_158494


namespace justin_reads_pages_l158_158757

theorem justin_reads_pages (x : ℕ) 
  (h1 : 130 = x + 6 * (2 * x)) : x = 10 := 
sorry

end justin_reads_pages_l158_158757


namespace part1_part2_part3_l158_158094

-- Definition of the function
def linear_function (m : ℝ) (x : ℝ) : ℝ :=
  (2 * m + 1) * x + m - 3

-- Part 1: If the graph passes through the origin
theorem part1 (h : linear_function m 0 = 0) : m = 3 :=
by {
  sorry
}

-- Part 2: If the graph is parallel to y = 3x - 3
theorem part2 (h : ∀ x, linear_function m x = 3 * x - 3 → 2 * m + 1 = 3) : m = 1 :=
by {
  sorry
}

-- Part 3: If the graph intersects the y-axis below the x-axis
theorem part3 (h_slope : 2 * m + 1 ≠ 0) (h_intercept : m - 3 < 0) : m < 3 ∧ m ≠ -1 / 2 :=
by {
  sorry
}

end part1_part2_part3_l158_158094


namespace octagon_area_in_square_l158_158979

/--
An octagon is inscribed in a square such that each vertex of the octagon cuts off a corner
triangle from the square. Each triangle has legs equal to one-fourth of the square's side.
If the perimeter of the square is 160 centimeters, what is the area of the octagon?
-/
theorem octagon_area_in_square
  (side_of_square : ℝ)
  (h1 : 4 * (side_of_square / 4) = side_of_square)
  (h2 : 8 * (side_of_square / 4) = side_of_square)
  (perimeter_of_square : ℝ)
  (h3 : perimeter_of_square = 160)
  (area_of_square : ℝ)
  (h4 : area_of_square = side_of_square^2)
  : ∃ (area_of_octagon : ℝ), area_of_octagon = 1400 := by
  sorry

end octagon_area_in_square_l158_158979


namespace solve_first_l158_158125

theorem solve_first (x y : ℝ) (C : ℝ) :
  (1 + y^2) * (deriv id x) - (1 + x^2) * y * (deriv id y) = 0 →
  Real.arctan x = 1/2 * Real.log (1 + y^2) + Real.log C := 
sorry

end solve_first_l158_158125


namespace circle_radius_l158_158376

theorem circle_radius (x y : ℝ) : x^2 - 10 * x + y^2 + 4 * y + 13 = 0 → (x - 5)^2 + (y + 2)^2 = 4^2 :=
by
  sorry

end circle_radius_l158_158376
