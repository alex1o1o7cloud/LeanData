import Mathlib

namespace NUMINAMATH_GPT_angle_measure_l801_80197

theorem angle_measure (x : ℝ) :
  (180 - x) = 7 * (90 - x) → 
  x = 75 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_angle_measure_l801_80197


namespace NUMINAMATH_GPT_value_of_b_cannot_form_arithmetic_sequence_l801_80130

theorem value_of_b 
  (a1 : ℝ) (a2 : ℝ) (a3 : ℝ) 
  (h1 : a1 = 150)
  (h2 : a2 = b)
  (h3 : a3 = 60 / 36)
  (h4 : b > 0) :
  b = 5 * Real.sqrt 10 := 
sorry

theorem cannot_form_arithmetic_sequence 
  (d : ℝ)
  (a1 : ℝ) (a2 : ℝ) (a3 : ℝ) 
  (h1 : a1 = 150)
  (h2 : a2 = b)
  (h3 : a3 = 60 / 36)
  (h4 : b = 5 * Real.sqrt 10) :
  ¬(∃ d, a1 + d = a2 ∧ a2 + d = a3) := 
sorry

end NUMINAMATH_GPT_value_of_b_cannot_form_arithmetic_sequence_l801_80130


namespace NUMINAMATH_GPT_real_root_fraction_l801_80172

theorem real_root_fraction (a b : ℝ) 
  (h_cond_a : a^4 - 7 * a - 3 = 0) 
  (h_cond_b : b^4 - 7 * b - 3 = 0)
  (h_order : a > b) : 
  (a - b) / (a^4 - b^4) = 1 / 7 := 
sorry

end NUMINAMATH_GPT_real_root_fraction_l801_80172


namespace NUMINAMATH_GPT_unique_value_expression_l801_80193

theorem unique_value_expression (m n : ℤ) : 
  (mn + 13 * m + 13 * n - m^2 - n^2 = 169) → 
  ∃! (m n : ℤ), mn + 13 * m + 13 * n - m^2 - n^2 = 169 := 
by
  sorry

end NUMINAMATH_GPT_unique_value_expression_l801_80193


namespace NUMINAMATH_GPT_original_cube_volume_l801_80131

theorem original_cube_volume (a : ℕ) (h : (a + 2) * (a + 1) * (a - 1) + 6 = a^3) : a = 2 :=
by sorry

example : 2^3 = 8 := by norm_num

end NUMINAMATH_GPT_original_cube_volume_l801_80131


namespace NUMINAMATH_GPT_find_b_15_l801_80147

variable {a : ℕ → ℤ} (b : ℕ → ℤ) (S : ℕ → ℤ)

/-- An arithmetic sequence where S_n is the sum of the first n terms, with S_9 = -18 and S_13 = -52
   and a geometric sequence where b_5 = a_5 and b_7 = a_7. -/
theorem find_b_15 
  (h1 : S 9 = -18) 
  (h2 : S 13 = -52) 
  (h3 : b 5 = a 5) 
  (h4 : b 7 = a 7) 
  : b 15 = -64 := 
sorry

end NUMINAMATH_GPT_find_b_15_l801_80147


namespace NUMINAMATH_GPT_largest_possible_sum_l801_80178

theorem largest_possible_sum (clubsuit heartsuit : ℕ) (h₁ : clubsuit * heartsuit = 48) (h₂ : Even clubsuit) : 
  clubsuit + heartsuit ≤ 26 :=
sorry

end NUMINAMATH_GPT_largest_possible_sum_l801_80178


namespace NUMINAMATH_GPT_max_value_2xy_sqrt6_8yz2_l801_80137

theorem max_value_2xy_sqrt6_8yz2 (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (h : x^2 + y^2 + z^2 = 1) : 
  2 * x * y * Real.sqrt 6 + 8 * y * z^2 ≤ Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_max_value_2xy_sqrt6_8yz2_l801_80137


namespace NUMINAMATH_GPT_units_digit_17_times_29_l801_80181

theorem units_digit_17_times_29 :
  (17 * 29) % 10 = 3 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_17_times_29_l801_80181


namespace NUMINAMATH_GPT_surface_area_of_large_cube_l801_80179

theorem surface_area_of_large_cube (l w h : ℕ) (cube_side : ℕ) 
  (volume_cuboid : ℕ := l * w * h) 
  (n_cubes := volume_cuboid / (cube_side ^ 3))
  (side_length_large_cube : ℕ := cube_side * (n_cubes^(1/3 : ℕ))) 
  (surface_area_large_cube : ℕ := 6 * (side_length_large_cube ^ 2)) :
  l = 25 → w = 10 → h = 4 → cube_side = 1 → surface_area_large_cube = 600 :=
by
  intros hl hw hh hcs
  subst hl
  subst hw
  subst hh
  subst hcs
  sorry

end NUMINAMATH_GPT_surface_area_of_large_cube_l801_80179


namespace NUMINAMATH_GPT_selling_price_is_correct_l801_80133

def profit_percent : ℝ := 0.6
def cost_price : ℝ := 375
def profit : ℝ := profit_percent * cost_price
def selling_price : ℝ := cost_price + profit

theorem selling_price_is_correct : selling_price = 600 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_selling_price_is_correct_l801_80133


namespace NUMINAMATH_GPT_contrapositive_of_original_l801_80170

theorem contrapositive_of_original (a b : ℝ) :
  (a > b → a - 1 > b - 1) ↔ (a - 1 ≤ b - 1 → a ≤ b) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_of_original_l801_80170


namespace NUMINAMATH_GPT_ratio_distance_l801_80106

theorem ratio_distance
  (x : ℝ)
  (P : ℝ × ℝ)
  (hP_coords : P = (x, -9))
  (h_distance_y_axis : abs x = 18) :
  abs (-9) / abs x = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_ratio_distance_l801_80106


namespace NUMINAMATH_GPT_wrongly_entered_mark_l801_80174

theorem wrongly_entered_mark (x : ℕ) 
    (h1 : x - 33 = 52) : x = 85 :=
by
  sorry

end NUMINAMATH_GPT_wrongly_entered_mark_l801_80174


namespace NUMINAMATH_GPT_largest_solution_achieves_largest_solution_l801_80166

theorem largest_solution (x : ℝ) (hx : ⌊x⌋ = 5 + 100 * (x - ⌊x⌋)) : x ≤ 104.99 :=
by
  -- Placeholder for the proof
  sorry

theorem achieves_largest_solution : ∃ (x : ℝ), ⌊x⌋ = 5 + 100 * (x - ⌊x⌋) ∧ x = 104.99 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_largest_solution_achieves_largest_solution_l801_80166


namespace NUMINAMATH_GPT_curve_equation_with_params_l801_80162

theorem curve_equation_with_params (a m x y : ℝ) (ha : a > 0) (hm : m ≠ 0) :
    (y^2) = m * (x^2 - a^2) ↔ mx^2 - y^2 = ma^2 := by
  sorry

end NUMINAMATH_GPT_curve_equation_with_params_l801_80162


namespace NUMINAMATH_GPT_minimum_value_2a_plus_b_l801_80101

theorem minimum_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : (1 / (a + 1)) + (2 / (b - 2)) = 1 / 2) : 2 * a + b ≥ 16 := 
sorry

end NUMINAMATH_GPT_minimum_value_2a_plus_b_l801_80101


namespace NUMINAMATH_GPT_exists_xy_binom_eq_l801_80177

theorem exists_xy_binom_eq (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ (x + y).choose 2 = a * x + b * y :=
by
  sorry

end NUMINAMATH_GPT_exists_xy_binom_eq_l801_80177


namespace NUMINAMATH_GPT_stuart_segments_to_start_point_l801_80118

-- Definitions of given conditions
def concentric_circles {C : Type} (large small : Set C) (center : C) : Prop :=
  ∀ (x y : C), x ∈ large → y ∈ large → x ≠ y → (x = center ∨ y = center)

def tangent_to_small_circle {C : Type} (chord : Set C) (small : Set C) : Prop :=
  ∀ (x y : C), x ∈ chord → y ∈ chord → x ≠ y → (∀ z ∈ small, x ≠ z ∧ y ≠ z)

def measure_angle (ABC : Type) (θ : ℝ) : Prop :=
  θ = 60

-- The theorem to solve the problem
theorem stuart_segments_to_start_point 
    (C : Type)
    {large small : Set C} 
    {center : C} 
    {chords : List (Set C)}
    (h_concentric : concentric_circles large small center)
    (h_tangent : ∀ chord ∈ chords, tangent_to_small_circle chord small)
    (h_angle : ∀ ABC ∈ chords, measure_angle ABC 60)
    : ∃ n : ℕ, n = 3 := 
  sorry

end NUMINAMATH_GPT_stuart_segments_to_start_point_l801_80118


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_roots_l801_80188

open Real

-- Define the polynomial and its properties using Vieta's formulas
theorem sum_of_reciprocals_of_roots :
  ∀ p q : ℝ, 
  (p + q = 16) ∧ (p * q = 9) → 
  (1 / p + 1 / q = 16 / 9) :=
by
  intros p q h
  let ⟨h1, h2⟩ := h
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_roots_l801_80188


namespace NUMINAMATH_GPT_car_total_distance_l801_80144

theorem car_total_distance (h1 h2 h3 : ℕ) :
  h1 = 180 → h2 = 160 → h3 = 220 → h1 + h2 + h3 = 560 :=
by
  intros h1_eq h2_eq h3_eq
  sorry

end NUMINAMATH_GPT_car_total_distance_l801_80144


namespace NUMINAMATH_GPT_total_campers_l801_80108

def campers_morning : ℕ := 36
def campers_afternoon : ℕ := 13
def campers_evening : ℕ := 49

theorem total_campers : campers_morning + campers_afternoon + campers_evening = 98 := by
  sorry

end NUMINAMATH_GPT_total_campers_l801_80108


namespace NUMINAMATH_GPT_trig_quadrant_l801_80159

theorem trig_quadrant (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
  ∃ k : ℤ, α = (2 * k + 1) * π + α / 2 :=
sorry

end NUMINAMATH_GPT_trig_quadrant_l801_80159


namespace NUMINAMATH_GPT_max_distance_from_point_to_line_l801_80116

theorem max_distance_from_point_to_line (θ m : ℝ) :
  let P := (Real.cos θ, Real.sin θ)
  let d := (P.1 - m * P.2 - 2) / Real.sqrt (1 + m^2)
  ∃ (θ m : ℝ), d ≤ 3 := sorry

end NUMINAMATH_GPT_max_distance_from_point_to_line_l801_80116


namespace NUMINAMATH_GPT_compute_expression_l801_80104

theorem compute_expression : 2 + 4 * 3^2 - 1 + 7 * 2 / 2 = 44 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l801_80104


namespace NUMINAMATH_GPT_unique_triangle_constructions_l801_80185

structure Triangle :=
(a b c : ℝ) (A B C : ℝ)

-- Definitions for the conditions
def SSS (t : Triangle) : Prop := 
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0

def SAS (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.A > 0 ∧ t.A < 180

def ASA (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.c > 0 ∧ t.A + t.B < 180

def SSA (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.A > 0 ∧ t.A < 180 

-- The formally stated proof goal
theorem unique_triangle_constructions (t : Triangle) :
  (SSS t ∨ SAS t ∨ ASA t) ∧ ¬(SSA t) :=
by
  sorry

end NUMINAMATH_GPT_unique_triangle_constructions_l801_80185


namespace NUMINAMATH_GPT_complement_of_P_with_respect_to_U_l801_80163

universe u

def U : Set ℤ := {-1, 0, 1, 2}

def P : Set ℤ := {x | x * x < 2}

theorem complement_of_P_with_respect_to_U : U \ P = {2} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_P_with_respect_to_U_l801_80163


namespace NUMINAMATH_GPT_not_divisible_by_3_or_4_l801_80126

theorem not_divisible_by_3_or_4 (n : ℤ) : 
  ¬ (n^2 + 1) % 3 = 0 ∧ ¬ (n^2 + 1) % 4 = 0 := 
by
  sorry

end NUMINAMATH_GPT_not_divisible_by_3_or_4_l801_80126


namespace NUMINAMATH_GPT_UnionMathInstitute_students_l801_80155

theorem UnionMathInstitute_students :
  ∃ n : ℤ, n < 500 ∧ 
    n % 17 = 15 ∧ 
    n % 19 = 18 ∧ 
    n % 16 = 7 ∧ 
    n = 417 :=
by
  -- Problem setup and constraints
  sorry

end NUMINAMATH_GPT_UnionMathInstitute_students_l801_80155


namespace NUMINAMATH_GPT_maximum_initial_jars_l801_80124

-- Define the conditions given in the problem
def initial_total_weight_carlson (n : ℕ) : ℕ := 13 * n
def new_total_weight_carlson (n a : ℕ) : ℕ := 13 * n - a
def total_weight_after_giving (n a : ℕ) : ℕ := 8 * (n + a)

-- Theorem statement for the maximum possible jars Carlson could have initially had
theorem maximum_initial_jars (n a k : ℕ) (h1 : initial_total_weight_carlson n = 13 * n)
  (h2 : new_total_weight_carlson n a = 8 * (n + a)) (h3 : n = 9 * k)
  (h4 : a = 5 * k) : (initial_total_weight_carlson n / a) ≤ 23 :=
by
  sorry

end NUMINAMATH_GPT_maximum_initial_jars_l801_80124


namespace NUMINAMATH_GPT_ratio_value_l801_80119

theorem ratio_value (c d : ℝ) (h1 : c = 15 - 4 * d) (h2 : c / d = 4) : d = 15 / 8 :=
by sorry

end NUMINAMATH_GPT_ratio_value_l801_80119


namespace NUMINAMATH_GPT_fraction_meaningful_l801_80164

theorem fraction_meaningful (x : ℝ) : (¬ (x - 2 = 0)) ↔ (x ≠ 2) :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_l801_80164


namespace NUMINAMATH_GPT_complement_of_A_is_correct_l801_80111

open Set

variable (U : Set ℝ) (A : Set ℝ)

def complement_of_A (U : Set ℝ) (A : Set ℝ) :=
  {x : ℝ | x ∉ A}

theorem complement_of_A_is_correct :
  (U = univ) →
  (A = {x : ℝ | x^2 - 2 * x > 0}) →
  (complement_of_A U A = {x : ℝ | 0 ≤ x ∧ x ≤ 2}) :=
by
  intros hU hA
  simp [hU, hA, complement_of_A]
  sorry

end NUMINAMATH_GPT_complement_of_A_is_correct_l801_80111


namespace NUMINAMATH_GPT_number_of_blue_balls_l801_80156

theorem number_of_blue_balls (T : ℕ) (h1 : (1 / 4) * T = green) (h2 : (1 / 8) * T = blue)
    (h3 : (1 / 12) * T = yellow) (h4 : 26 = white) (h5 : green + blue + yellow + white = T) :
    blue = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_blue_balls_l801_80156


namespace NUMINAMATH_GPT_participants_who_drank_neither_l801_80199

-- Conditions
variables (total_participants : ℕ) (coffee_drinkers : ℕ) (juice_drinkers : ℕ) (both_drinkers : ℕ)

-- Initial Facts from the Conditions
def conditions := total_participants = 30 ∧ coffee_drinkers = 15 ∧ juice_drinkers = 18 ∧ both_drinkers = 7

-- The statement to prove
theorem participants_who_drank_neither : conditions total_participants coffee_drinkers juice_drinkers both_drinkers → 
  (total_participants - (coffee_drinkers + juice_drinkers - both_drinkers)) = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_participants_who_drank_neither_l801_80199


namespace NUMINAMATH_GPT_tan_diff_l801_80141

theorem tan_diff (x y : ℝ) (hx : Real.tan x = 3) (hy : Real.tan y = 2) : 
  Real.tan (x - y) = 1 / 7 := 
by 
  sorry

end NUMINAMATH_GPT_tan_diff_l801_80141


namespace NUMINAMATH_GPT_ratio_problem_l801_80196

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2) (h' : B / C = 2 / 5) : (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := 
by
  sorry

end NUMINAMATH_GPT_ratio_problem_l801_80196


namespace NUMINAMATH_GPT_seashells_count_l801_80105

theorem seashells_count (total_seashells broken_seashells : ℕ) (h_total : total_seashells = 7) (h_broken : broken_seashells = 4) : total_seashells - broken_seashells = 3 := by
  sorry

end NUMINAMATH_GPT_seashells_count_l801_80105


namespace NUMINAMATH_GPT_sum_of_coefficients_l801_80161

theorem sum_of_coefficients :
  ∃ a b c d e : ℤ, 
    27 * (x : ℝ)^3 + 64 = (a * x + b) * (c * x^2 + d * x + e) ∧ 
    a + b + c + d + e = 20 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l801_80161


namespace NUMINAMATH_GPT_find_three_digit_number_l801_80113

theorem find_three_digit_number (P Q R : ℕ) 
  (h1 : P ≠ Q) 
  (h2 : P ≠ R) 
  (h3 : Q ≠ R) 
  (h4 : P < 7) 
  (h5 : Q < 7) 
  (h6 : R < 7)
  (h7 : P ≠ 0) 
  (h8 : Q ≠ 0) 
  (h9 : R ≠ 0) 
  (h10 : 7 * P + Q + R = 7 * R) 
  (h11 : (7 * P + Q) + (7 * Q + P) = 49 + 7 * R + R)
  : P * 100 + Q * 10 + R = 434 :=
sorry

end NUMINAMATH_GPT_find_three_digit_number_l801_80113


namespace NUMINAMATH_GPT_find_tenth_term_l801_80175

/- Define the general term formula -/
def a (a1 d : ℤ) (n : ℤ) : ℤ := a1 + (n - 1) * d

/- Define the sum of the first n terms formula -/
def S (a1 d : ℤ) (n : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem find_tenth_term
  (a1 d : ℤ)
  (h1 : a a1 d 2 + a a1 d 5 = 19)
  (h2 : S a1 d 5 = 40) :
  a a1 d 10 = 29 := by
  /- Sorry used to skip the proof steps. -/
  sorry

end NUMINAMATH_GPT_find_tenth_term_l801_80175


namespace NUMINAMATH_GPT_cookies_difference_l801_80189

-- Define the initial conditions
def initial_cookies : ℝ := 57
def cookies_eaten : ℝ := 8.5
def cookies_bought : ℝ := 125.75

-- Problem statement
theorem cookies_difference (initial_cookies cookies_eaten cookies_bought : ℝ) : 
  cookies_bought - cookies_eaten = 117.25 := 
sorry

end NUMINAMATH_GPT_cookies_difference_l801_80189


namespace NUMINAMATH_GPT_obtuse_triangle_range_a_l801_80122

noncomputable def is_obtuse_triangle (a b c : ℝ) : Prop :=
  ∃ (θ : ℝ), θ > 90 ∧ θ ≤ 120 ∧ c^2 > a^2 + b^2

theorem obtuse_triangle_range_a (a : ℝ) :
  (a + (a + 1) > a + 2) →
  is_obtuse_triangle a (a + 1) (a + 2) →
  (1.5 ≤ a ∧ a < 3) :=
by
  sorry

end NUMINAMATH_GPT_obtuse_triangle_range_a_l801_80122


namespace NUMINAMATH_GPT_range_of_m_l801_80154

theorem range_of_m (m : ℝ) :
  (∀ P : ℝ × ℝ, P.2 = 2 * P.1 + m → (abs (P.1^2 + (P.2 - 1)^2) = (1/2) * abs (P.1^2 + (P.2 - 4)^2)) → (-2 * Real.sqrt 5) ≤ m ∧ m ≤ (2 * Real.sqrt 5)) :=
sorry

end NUMINAMATH_GPT_range_of_m_l801_80154


namespace NUMINAMATH_GPT_refund_amount_l801_80158

def income_tax_paid : ℝ := 156000
def education_expenses : ℝ := 130000
def medical_expenses : ℝ := 10000
def tax_rate : ℝ := 0.13

def eligible_expenses : ℝ := education_expenses + medical_expenses
def max_refund : ℝ := tax_rate * eligible_expenses

theorem refund_amount : min (max_refund) (income_tax_paid) = 18200 := by
  sorry

end NUMINAMATH_GPT_refund_amount_l801_80158


namespace NUMINAMATH_GPT_sequence_a_l801_80107

theorem sequence_a (a : ℕ → ℝ) 
  (h1 : a 1 = 2)
  (h2 : a 2 = 1)
  (h3 : ∀ n ≥ 2, a n / a (n + 1) + a n / a (n - 1) = 2) :
  a 12 = 1 / 6 :=
sorry

end NUMINAMATH_GPT_sequence_a_l801_80107


namespace NUMINAMATH_GPT_find_highest_score_l801_80176

theorem find_highest_score (average innings : ℕ) (avg_excl_two innings_excl_two H L : ℕ)
  (diff_high_low total_runs total_excl_two : ℕ)
  (h1 : diff_high_low = 150)
  (h2 : total_runs = average * innings)
  (h3 : total_excl_two = avg_excl_two * innings_excl_two)
  (h4 : total_runs - total_excl_two = H + L)
  (h5 : H - L = diff_high_low)
  (h6 : average = 62)
  (h7 : innings = 46)
  (h8 : avg_excl_two = 58)
  (h9 : innings_excl_two = 44)
  (h10 : total_runs = 2844)
  (h11 : total_excl_two = 2552) :
  H = 221 :=
by
  sorry

end NUMINAMATH_GPT_find_highest_score_l801_80176


namespace NUMINAMATH_GPT_quadratic_function_points_l801_80121

theorem quadratic_function_points (a c y1 y2 y3 y4 : ℝ) (h_a : a < 0)
    (h_A : y1 = a * (-2)^2 - 4 * a * (-2) + c)
    (h_B : y2 = a * 0^2 - 4 * a * 0 + c)
    (h_C : y3 = a * 3^2 - 4 * a * 3 + c)
    (h_D : y4 = a * 5^2 - 4 * a * 5 + c)
    (h_condition : y2 * y4 < 0) : y1 * y3 < 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_points_l801_80121


namespace NUMINAMATH_GPT_cost_of_items_l801_80128

variable (e t d : ℝ)

noncomputable def ques :=
  5 * e + 5 * t + 2 * d

axiom cond1 : 3 * e + 4 * t = 3.40
axiom cond2 : 4 * e + 3 * t = 4.00
axiom cond3 : 5 * e + 4 * t + 3 * d = 7.50

theorem cost_of_items : ques e t d = 6.93 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_items_l801_80128


namespace NUMINAMATH_GPT_cannot_determine_a_l801_80110

theorem cannot_determine_a 
  (n : ℝ) 
  (p : ℝ) 
  (a : ℝ) 
  (line_eq : ∀ (x y : ℝ), x = 5 * y + 5) 
  (pt1 : a = 5 * n + 5) 
  (pt2 : a + 2 = 5 * (n + p) + 5) : p = 0.4 → ¬∀ a' : ℝ, a = a' :=
by
  sorry

end NUMINAMATH_GPT_cannot_determine_a_l801_80110


namespace NUMINAMATH_GPT_club_additional_members_l801_80134

theorem club_additional_members (current_members additional_members future_members : ℕ) 
  (h1 : current_members = 10) 
  (h2 : additional_members = 15) 
  (h3 : future_members = current_members + additional_members) : 
  future_members - current_members = 15 :=
by
  sorry

end NUMINAMATH_GPT_club_additional_members_l801_80134


namespace NUMINAMATH_GPT_smallest_positive_integer_divisible_conditions_l801_80160

theorem smallest_positive_integer_divisible_conditions :
  ∃ (M : ℕ), M % 4 = 3 ∧ M % 5 = 4 ∧ M % 6 = 5 ∧ M % 7 = 6 ∧ M = 419 :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_divisible_conditions_l801_80160


namespace NUMINAMATH_GPT_detour_distance_l801_80127

-- Definitions based on conditions:
def D_black : ℕ := sorry -- The original distance along the black route
def D_black_C : ℕ := sorry -- The distance from C to B along the black route
def D_red : ℕ := sorry -- The distance from C to B along the red route

-- Extra distance due to detour calculation
def D_extra := D_red - D_black_C

-- Prove that the extra distance is 14 km
theorem detour_distance : D_extra = 14 := by
  sorry

end NUMINAMATH_GPT_detour_distance_l801_80127


namespace NUMINAMATH_GPT_rectangular_prism_cut_l801_80190

theorem rectangular_prism_cut
  (x y : ℕ)
  (original_volume : ℕ := 15 * 5 * 4) 
  (remaining_volume : ℕ := 120) 
  (cut_out_volume_eq : original_volume - remaining_volume = 5 * x * y) 
  (x_condition : 1 < x) 
  (x_condition_2 : x < 4) 
  (y_condition : 1 < y) 
  (y_condition_2 : y < 15) : 
  x + y = 15 := 
sorry

end NUMINAMATH_GPT_rectangular_prism_cut_l801_80190


namespace NUMINAMATH_GPT_initial_tickets_l801_80120

theorem initial_tickets (tickets_sold_week1 : ℕ) (tickets_sold_week2 : ℕ) (tickets_left : ℕ) 
  (h1 : tickets_sold_week1 = 38) (h2 : tickets_sold_week2 = 17) (h3 : tickets_left = 35) : 
  tickets_sold_week1 + tickets_sold_week2 + tickets_left = 90 :=
by 
  sorry

end NUMINAMATH_GPT_initial_tickets_l801_80120


namespace NUMINAMATH_GPT_emily_second_round_points_l801_80123

theorem emily_second_round_points (P : ℤ)
  (first_round_points : ℤ := 16)
  (last_round_points_lost : ℤ := 48)
  (end_points : ℤ := 1)
  (points_equation : first_round_points + P - last_round_points_lost = end_points) :
  P = 33 :=
  by {
    sorry
  }

end NUMINAMATH_GPT_emily_second_round_points_l801_80123


namespace NUMINAMATH_GPT_intersection_point_l801_80100

def line_parametric (t : ℝ) : ℝ × ℝ × ℝ :=
  (3 + 2 * t, -1 + 3 * t, -3 + 2 * t)

def on_plane (x y z : ℝ) : Prop :=
  3 * x + 4 * y + 7 * z - 16 = 0

theorem intersection_point : ∃ t, line_parametric t = (5, 2, -1) ∧ on_plane 5 2 (-1) :=
by
  use 1
  sorry

end NUMINAMATH_GPT_intersection_point_l801_80100


namespace NUMINAMATH_GPT_fifth_number_in_pascals_triangle_l801_80194

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end NUMINAMATH_GPT_fifth_number_in_pascals_triangle_l801_80194


namespace NUMINAMATH_GPT_total_marble_weight_l801_80109

theorem total_marble_weight (w1 w2 w3 : ℝ) (h_w1 : w1 = 0.33) (h_w2 : w2 = 0.33) (h_w3 : w3 = 0.08) :
  w1 + w2 + w3 = 0.74 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_marble_weight_l801_80109


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l801_80180

theorem sufficient_but_not_necessary_condition (A B : Set ℝ) :
  (A = {x : ℝ | 1 < x ∧ x < 3}) →
  (B = {x : ℝ | x > -1}) →
  (∀ x, x ∈ A → x ∈ B) ∧ (∃ x, x ∈ B ∧ x ∉ A) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l801_80180


namespace NUMINAMATH_GPT_more_plastic_pipe_l801_80125

variable (m_copper m_plastic : Nat)
variable (total_cost cost_per_meter : Nat)

-- Conditions
variable (h1 : m_copper = 10)
variable (h2 : cost_per_meter = 4)
variable (h3 : total_cost = 100)
variable (h4 : m_copper * cost_per_meter + m_plastic * cost_per_meter = total_cost)

-- Proof that the number of more meters of plastic pipe bought compared to the copper pipe is 5
theorem more_plastic_pipe :
  m_plastic - m_copper = 5 :=
by
  -- Since proof is not required, we place sorry here.
  sorry

end NUMINAMATH_GPT_more_plastic_pipe_l801_80125


namespace NUMINAMATH_GPT_mrs_hilt_total_distance_l801_80148

-- Define the distances and number of trips
def distance_to_water_fountain := 30
def distance_to_staff_lounge := 45
def trips_to_water_fountain := 4
def trips_to_staff_lounge := 3

-- Calculate the total distance for Mrs. Hilt's trips
def total_distance := (distance_to_water_fountain * 2 * trips_to_water_fountain) + 
                      (distance_to_staff_lounge * 2 * trips_to_staff_lounge)
                      
theorem mrs_hilt_total_distance : total_distance = 510 := 
by
  sorry

end NUMINAMATH_GPT_mrs_hilt_total_distance_l801_80148


namespace NUMINAMATH_GPT_inequality_proof_l801_80145

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (2 * a) + 1 / (2 * b) + 1 / (2 * c)) ≥ (1 / (b + c) + 1 / (c + a) + 1 / (a + b)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l801_80145


namespace NUMINAMATH_GPT_proof_time_to_run_square_field_l801_80173

def side : ℝ := 40
def speed_kmh : ℝ := 9
def perimeter (side : ℝ) : ℝ := 4 * side

noncomputable def speed_mps (speed_kmh : ℝ) : ℝ := speed_kmh * (1000 / 3600)

noncomputable def time_to_run (perimeter : ℝ) (speed_mps : ℝ) : ℝ := perimeter / speed_mps

theorem proof_time_to_run_square_field :
  time_to_run (perimeter side) (speed_mps speed_kmh) = 64 :=
by
  sorry

end NUMINAMATH_GPT_proof_time_to_run_square_field_l801_80173


namespace NUMINAMATH_GPT_juice_fraction_left_l801_80165

theorem juice_fraction_left (initial_juice : ℝ) (given_juice : ℝ) (remaining_juice : ℝ) : 
  initial_juice = 5 → given_juice = 18/4 → remaining_juice = initial_juice - given_juice → remaining_juice = 1/2 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  rw [h3]
  sorry

end NUMINAMATH_GPT_juice_fraction_left_l801_80165


namespace NUMINAMATH_GPT_factorize_expression_l801_80182

theorem factorize_expression (x y : ℝ) : x^2 * y - 2 * x * y^2 + y^3 = y * (x - y)^2 := 
sorry

end NUMINAMATH_GPT_factorize_expression_l801_80182


namespace NUMINAMATH_GPT_find_f_l801_80140

theorem find_f (f : ℝ → ℝ) (h₁ : ∀ x y : ℝ, 0 < x → 0 < y → x ≤ y → f x ≤ f y)
  (h₂ : ∀ x : ℝ, 0 < x → f (x ^ 4) + f (x ^ 2) + f x + f 1 = x ^ 4 + x ^ 2 + x + 1) :
  ∀ x : ℝ, 0 < x → f x = x := 
sorry

end NUMINAMATH_GPT_find_f_l801_80140


namespace NUMINAMATH_GPT_equidistant_points_l801_80135

theorem equidistant_points (r d1 d2 : ℝ) (d1_eq : d1 = r) (d2_eq : d2 = 6) : 
  ∃ p : ℝ, p = 2 := 
sorry

end NUMINAMATH_GPT_equidistant_points_l801_80135


namespace NUMINAMATH_GPT_bricks_in_wall_l801_80102

-- Definitions of conditions based on the problem statement
def time_first_bricklayer : ℝ := 12 
def time_second_bricklayer : ℝ := 15 
def reduced_productivity : ℝ := 12 
def combined_time : ℝ := 6
def total_bricks : ℝ := 720

-- Lean 4 statement of the proof problem
theorem bricks_in_wall (x : ℝ) 
  (h1 : (x / time_first_bricklayer + x / time_second_bricklayer - reduced_productivity) * combined_time = x) 
  : x = total_bricks := 
by {
  sorry
}

end NUMINAMATH_GPT_bricks_in_wall_l801_80102


namespace NUMINAMATH_GPT_symmetry_about_origin_l801_80198

theorem symmetry_about_origin (m : ℝ) (A B : ℝ × ℝ) (hA : A = (2, -1)) (hB : B = (-2, m)) (h_sym : B = (-A.1, -A.2)) :
  m = 1 :=
by
  sorry

end NUMINAMATH_GPT_symmetry_about_origin_l801_80198


namespace NUMINAMATH_GPT_primes_square_condition_l801_80139

open Nat

theorem primes_square_condition (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) : 
  ∃ n : ℕ, p^(q+1) + q^(p+1) = n^2 ↔ p = 2 ∧ q = 2 := by
  sorry

end NUMINAMATH_GPT_primes_square_condition_l801_80139


namespace NUMINAMATH_GPT_probability_of_two_sunny_days_l801_80191

def prob_two_sunny_days (prob_sunny prob_rain : ℚ) (days : ℕ) : ℚ :=
  (days.choose 2) * (prob_sunny^2 * prob_rain^(days-2))

theorem probability_of_two_sunny_days :
  prob_two_sunny_days (2/5) (3/5) 3 = 36/125 :=
by 
  sorry

end NUMINAMATH_GPT_probability_of_two_sunny_days_l801_80191


namespace NUMINAMATH_GPT_certain_positive_integer_value_l801_80183

theorem certain_positive_integer_value :
  ∃ (i m p : ℕ), (x = 2 ^ i * 3 ^ 2 * 5 ^ m * 7 ^ p) ∧ (i + 2 + m + p = 11) :=
by
  let x := 40320 -- 8!
  sorry

end NUMINAMATH_GPT_certain_positive_integer_value_l801_80183


namespace NUMINAMATH_GPT_total_earning_correct_l801_80114

-- Definitions based on conditions
def daily_wage_c : ℕ := 105
def days_worked_a : ℕ := 6
def days_worked_b : ℕ := 9
def days_worked_c : ℕ := 4

-- Given the ratio of their daily wages
def ratio_a : ℕ := 3
def ratio_b : ℕ := 4
def ratio_c : ℕ := 5

-- Now we calculate the daily wages based on the ratio
def unit_wage : ℕ := daily_wage_c / ratio_c
def daily_wage_a : ℕ := ratio_a * unit_wage
def daily_wage_b : ℕ := ratio_b * unit_wage

-- Total earnings are calculated by multiplying daily wages and days worked
def total_earning_a : ℕ := days_worked_a * daily_wage_a
def total_earning_b : ℕ := days_worked_b * daily_wage_b
def total_earning_c : ℕ := days_worked_c * daily_wage_c

def total_earning : ℕ := total_earning_a + total_earning_b + total_earning_c

-- Theorem to prove
theorem total_earning_correct : total_earning = 1554 := by
  sorry

end NUMINAMATH_GPT_total_earning_correct_l801_80114


namespace NUMINAMATH_GPT_part1_part2_l801_80192

open Set

variable (A B : Set ℝ) (m : ℝ)

def setA : Set ℝ := {x | x ^ 2 - 2 * x - 8 ≤ 0}

def setB (m : ℝ) : Set ℝ := {x | x ^ 2 - (2 * m - 3) * x + m ^ 2 - 3 * m ≤ 0}

theorem part1 (h : (setA ∩ setB 5) = Icc 2 4) : m = 5 := sorry

theorem part2 (h : setA ⊆ compl (setB m)) :
  m ∈ Iio (-2) ∪ Ioi 7 := sorry

end NUMINAMATH_GPT_part1_part2_l801_80192


namespace NUMINAMATH_GPT_factorize_expression_l801_80112

variable {R : Type} [CommRing R] (m a : R)

theorem factorize_expression : m * a^2 - m = m * (a + 1) * (a - 1) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l801_80112


namespace NUMINAMATH_GPT_joe_total_paint_used_l801_80149

-- Conditions
def initial_paint : ℕ := 360
def paint_first_week : ℕ := initial_paint * 1 / 4
def remaining_paint_after_first_week : ℕ := initial_paint - paint_first_week
def paint_second_week : ℕ := remaining_paint_after_first_week * 1 / 6

-- Theorem statement
theorem joe_total_paint_used : paint_first_week + paint_second_week = 135 := by
  sorry

end NUMINAMATH_GPT_joe_total_paint_used_l801_80149


namespace NUMINAMATH_GPT_solve_for_x_l801_80132

theorem solve_for_x (x : ℝ) : 
  5 * x + 9 * x = 420 - 12 * (x - 4) -> 
  x = 18 :=
by
  intro h
  -- derivation will follow here
  sorry

end NUMINAMATH_GPT_solve_for_x_l801_80132


namespace NUMINAMATH_GPT_opposite_face_is_D_l801_80103

-- Define the six faces
inductive Face
| A | B | C | D | E | F

open Face

-- Define the adjacency relation
def is_adjacent (x y : Face) : Prop :=
(y = B ∧ x = A) ∨ (y = F ∧ x = A) ∨ (y = C ∧ x = A) ∨ (y = E ∧ x = A)

-- Define the problem statement in Lean
theorem opposite_face_is_D : 
  (∀ (x : Face), is_adjacent A x ↔ x = B ∨ x = F ∨ x = C ∨ x = E) →
  (¬ (is_adjacent A D)) →
  True :=
by
  intro adj_relation non_adj_relation
  sorry

end NUMINAMATH_GPT_opposite_face_is_D_l801_80103


namespace NUMINAMATH_GPT_sample_size_l801_80153

theorem sample_size {n : ℕ} (h_ratio : 2+3+4 = 9)
  (h_units_A : ∃ a : ℕ, a = 16)
  (h_stratified_sampling : ∃ B C : ℕ, B = 24 ∧ C = 32)
  : n = 16 + 24 + 32 := by
  sorry

end NUMINAMATH_GPT_sample_size_l801_80153


namespace NUMINAMATH_GPT_Dan_team_lost_games_l801_80169

/-- Dan's high school played eighteen baseball games this year.
Two were at night and they won 15 games. Prove that they lost 3 games. -/
theorem Dan_team_lost_games (total_games won_games : ℕ) (h_total : total_games = 18) (h_won : won_games = 15) :
  total_games - won_games = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_Dan_team_lost_games_l801_80169


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l801_80184

-- Prove that x^2 ≥ -x is a necessary but not sufficient condition for |x| = x
theorem necessary_but_not_sufficient_condition (x : ℝ) : x^2 ≥ -x → |x| = x ↔ x ≥ 0 := 
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l801_80184


namespace NUMINAMATH_GPT_largest_even_digit_multiple_of_five_l801_80195

theorem largest_even_digit_multiple_of_five : ∃ n : ℕ, n = 8860 ∧ n < 10000 ∧ (∀ digit ∈ (n.digits 10), digit % 2 = 0) ∧ n % 5 = 0 :=
by
  sorry

end NUMINAMATH_GPT_largest_even_digit_multiple_of_five_l801_80195


namespace NUMINAMATH_GPT_simplify_and_evaluate_l801_80152

theorem simplify_and_evaluate (m : ℝ) (h_root : m^2 + 3 * m - 2 = 0) :
  (m - 3) / (3 * m^2 - 6 * m) / (m + 2 - 5 / (m - 2)) = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l801_80152


namespace NUMINAMATH_GPT_arithmetic_series_first_term_l801_80129

theorem arithmetic_series_first_term :
  ∃ (a d : ℝ), (25 * (2 * a + 49 * d) = 200) ∧ (25 * (2 * a + 149 * d) = 2700) ∧ (a = -20.5) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_series_first_term_l801_80129


namespace NUMINAMATH_GPT_first_driver_spends_less_time_l801_80138

noncomputable def round_trip_time (d : ℝ) (v₁ v₂ : ℝ) : ℝ := (d / v₁) + (d / v₂)

theorem first_driver_spends_less_time (d : ℝ) : 
  round_trip_time d 80 80 < round_trip_time d 90 70 :=
by
  --We skip the proof here
  sorry

end NUMINAMATH_GPT_first_driver_spends_less_time_l801_80138


namespace NUMINAMATH_GPT_convinced_of_twelve_models_vitya_review_58_offers_l801_80167

noncomputable def ln : ℝ → ℝ := Real.log

theorem convinced_of_twelve_models (n : ℕ) (h_n : n ≥ 13) :
  ∃ k : ℕ, (12 / n : ℝ) ^ k < 0.01 := sorry

theorem vitya_review_58_offers :
  ∃ k : ℕ, (12 / 13 : ℝ) ^ k < 0.01 ∧ k = 58 := sorry

end NUMINAMATH_GPT_convinced_of_twelve_models_vitya_review_58_offers_l801_80167


namespace NUMINAMATH_GPT_lcm_gcd_product_12_15_l801_80146

theorem lcm_gcd_product_12_15 : 
  let a := 12
  let b := 15
  lcm a b * gcd a b = 180 :=
by
  sorry

end NUMINAMATH_GPT_lcm_gcd_product_12_15_l801_80146


namespace NUMINAMATH_GPT_equation_holds_l801_80117

variable (a b : ℝ)

theorem equation_holds : a^2 - b^2 - (-2 * b^2) = a^2 + b^2 :=
by sorry

end NUMINAMATH_GPT_equation_holds_l801_80117


namespace NUMINAMATH_GPT_solve_for_x_l801_80186

theorem solve_for_x : ∀ x : ℕ, x + 1315 + 9211 - 1569 = 11901 → x = 2944 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l801_80186


namespace NUMINAMATH_GPT_each_person_has_5_bags_l801_80168

def people := 6
def weight_per_bag := 50
def max_plane_weight := 6000
def additional_capacity := 90

theorem each_person_has_5_bags :
  (max_plane_weight / weight_per_bag - additional_capacity) / people = 5 :=
by
  sorry

end NUMINAMATH_GPT_each_person_has_5_bags_l801_80168


namespace NUMINAMATH_GPT_inequality_solution_set_l801_80142

theorem inequality_solution_set {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x)
  (h_inc : ∀ {x y : ℝ}, 0 < x → x < y → f x ≤ f y)
  (h_value : f 1 = 0) :
  {x | (f x - f (-x)) / x ≤ 0} = {x | -1 ≤ x ∧ x < 0} ∪ {x | 0 < x ∧ x ≤ 1} :=
by
  sorry


end NUMINAMATH_GPT_inequality_solution_set_l801_80142


namespace NUMINAMATH_GPT_inverse_proportion_function_l801_80115

theorem inverse_proportion_function (f : ℝ → ℝ) (h : ∀ x, f x = 1/x) : f 1 = 1 := 
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_function_l801_80115


namespace NUMINAMATH_GPT_max_squares_at_a1_bksq_l801_80143

noncomputable def maximizePerfectSquares (a b : ℕ) : Prop := 
a ≠ b ∧ 
(∃ k : ℕ, k ≠ 1 ∧ b = k^2) ∧ 
a = 1

theorem max_squares_at_a1_bksq (a b : ℕ) : maximizePerfectSquares a b := 
by 
  sorry

end NUMINAMATH_GPT_max_squares_at_a1_bksq_l801_80143


namespace NUMINAMATH_GPT_hyperbola_foci_coordinates_l801_80157

theorem hyperbola_foci_coordinates :
  ∀ x y : ℝ, (x^2 / 4) - (y^2 / 12) = 1 → (x, y) = (4, 0) ∨ (x, y) = (-4, 0) :=
by
  -- We assume the given equation of the hyperbola
  intro x y h
  -- sorry is used to skip the actual proof steps
  sorry

end NUMINAMATH_GPT_hyperbola_foci_coordinates_l801_80157


namespace NUMINAMATH_GPT_even_fn_solution_set_l801_80150

theorem even_fn_solution_set (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) (h_f_def : ∀ x ≥ 0, f x = x^3 - 8) :
  { x | f (x - 2) > 0 } = { x | x < 0 ∨ x > 4 } :=
by sorry

end NUMINAMATH_GPT_even_fn_solution_set_l801_80150


namespace NUMINAMATH_GPT_find_resistance_x_l801_80187

theorem find_resistance_x (y r x : ℝ) (h₁ : y = 5) (h₂ : r = 1.875) (h₃ : 1/r = 1/x + 1/y) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_resistance_x_l801_80187


namespace NUMINAMATH_GPT_royalty_amount_l801_80151

theorem royalty_amount (x : ℝ) (h1 : x > 800) (h2 : x ≤ 4000) (h3 : (x - 800) * 0.14 = 420) :
  x = 3800 :=
by
  sorry

end NUMINAMATH_GPT_royalty_amount_l801_80151


namespace NUMINAMATH_GPT_rabbitAgeOrder_l801_80171

-- Define the ages of the rabbits as variables
variables (blue black red gray : ℕ)

-- Conditions based on the problem statement
noncomputable def rabbitConditions := 
  (blue ≠ max blue (max black (max red gray))) ∧  -- The blue-eyed rabbit is not the eldest
  (gray ≠ min blue (min black (min red gray))) ∧  -- The gray rabbit is not the youngest
  (red ≠ min blue (min black (min red gray))) ∧  -- The red-eyed rabbit is not the youngest
  (black > red) ∧ (gray > black)  -- The black rabbit is older than the red-eyed rabbit and younger than the gray rabbit

-- Required proof statement
theorem rabbitAgeOrder : rabbitConditions blue black red gray → gray > black ∧ black > red ∧ red > blue :=
by
  intro h
  sorry

end NUMINAMATH_GPT_rabbitAgeOrder_l801_80171


namespace NUMINAMATH_GPT_smallest_sum_of_factors_of_8_l801_80136

theorem smallest_sum_of_factors_of_8! :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
  a * b * c * d = Nat.factorial 8 ∧ a + b + c + d = 102 :=
sorry

end NUMINAMATH_GPT_smallest_sum_of_factors_of_8_l801_80136
