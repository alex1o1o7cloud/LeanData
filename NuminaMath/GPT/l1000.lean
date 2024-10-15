import Mathlib

namespace NUMINAMATH_GPT_regular_hexagon_interior_angle_deg_l1000_100030

theorem regular_hexagon_interior_angle_deg (n : ℕ) (h1 : n = 6) :
  let sum_of_interior_angles : ℕ := (n - 2) * 180
  let each_angle : ℕ := sum_of_interior_angles / n
  each_angle = 120 := by
  sorry

end NUMINAMATH_GPT_regular_hexagon_interior_angle_deg_l1000_100030


namespace NUMINAMATH_GPT_jessica_age_l1000_100025

theorem jessica_age 
  (j g : ℚ)
  (h1 : g = 15 * j) 
  (h2 : g - j = 60) : 
  j = 30 / 7 :=
by
  sorry

end NUMINAMATH_GPT_jessica_age_l1000_100025


namespace NUMINAMATH_GPT_painter_completes_at_9pm_l1000_100099

noncomputable def mural_completion_time (start_time : Nat) (fraction_completed_time : Nat)
    (fraction_completed : ℚ) : Nat :=
  let fraction_per_hour := fraction_completed / fraction_completed_time
  start_time + Nat.ceil (1 / fraction_per_hour)

theorem painter_completes_at_9pm :
  mural_completion_time 9 3 (1/4) = 21 := by
  sorry

end NUMINAMATH_GPT_painter_completes_at_9pm_l1000_100099


namespace NUMINAMATH_GPT_bulb_standard_probability_l1000_100016

noncomputable def prob_A 
  (P_H1 : ℝ) (P_H2 : ℝ) (P_A_given_H1 : ℝ) (P_A_given_H2 : ℝ) :=
  P_A_given_H1 * P_H1 + P_A_given_H2 * P_H2

theorem bulb_standard_probability 
  (P_H1 : ℝ := 0.6) (P_H2 : ℝ := 0.4) 
  (P_A_given_H1 : ℝ := 0.95) (P_A_given_H2 : ℝ := 0.85) :
  prob_A P_H1 P_H2 P_A_given_H1 P_A_given_H2 = 0.91 :=
by
  sorry

end NUMINAMATH_GPT_bulb_standard_probability_l1000_100016


namespace NUMINAMATH_GPT_white_ducks_count_l1000_100042

theorem white_ducks_count (W : ℕ) : 
  (5 * W + 10 * 7 + 12 * 6 = 157) → W = 3 :=
by
  sorry

end NUMINAMATH_GPT_white_ducks_count_l1000_100042


namespace NUMINAMATH_GPT_arrange_descending_order_l1000_100020

noncomputable def a := 8 ^ 0.7
noncomputable def b := 8 ^ 0.9
noncomputable def c := 2 ^ 0.8

theorem arrange_descending_order :
    b > a ∧ a > c := by
  sorry

end NUMINAMATH_GPT_arrange_descending_order_l1000_100020


namespace NUMINAMATH_GPT_mia_study_time_l1000_100007

theorem mia_study_time 
  (T : ℕ)
  (watching_tv_exercise_social_media : T = 1440 ∧ 
    ∃ study_time : ℚ, 
    (study_time = (1 / 4) * 
      (((27 / 40) * T - (9 / 80) * T) / 
        (T * 1 / 40 - (1 / 5) * T - (1 / 8) * T))
    )) :
  T = 1440 → study_time = 202.5 := 
by
  sorry

end NUMINAMATH_GPT_mia_study_time_l1000_100007


namespace NUMINAMATH_GPT_sum_abcd_value_l1000_100056

theorem sum_abcd_value (a b c d : ℚ) :
  (2 * a + 3 = 2 * b + 5) ∧ 
  (2 * b + 5 = 2 * c + 7) ∧ 
  (2 * c + 7 = 2 * d + 9) ∧ 
  (2 * d + 9 = 2 * (a + b + c + d) + 13) → 
  a + b + c + d = -14 / 3 := 
by
  sorry

end NUMINAMATH_GPT_sum_abcd_value_l1000_100056


namespace NUMINAMATH_GPT_distance_between_city_centers_l1000_100092

theorem distance_between_city_centers (d_map : ℝ) (scale : ℝ) (d_real : ℝ) (h1 : d_map = 112) (h2 : scale = 10) (h3 : d_real = d_map * scale) : d_real = 1120 := by
  sorry

end NUMINAMATH_GPT_distance_between_city_centers_l1000_100092


namespace NUMINAMATH_GPT_calculate_total_cost_l1000_100097

theorem calculate_total_cost :
  let sandwich_cost := 4
  let soda_cost := 3
  let num_sandwiches := 6
  let num_sodas := 5
  sandwich_cost * num_sandwiches + soda_cost * num_sodas = 39 := by
  sorry

end NUMINAMATH_GPT_calculate_total_cost_l1000_100097


namespace NUMINAMATH_GPT_find_C_l1000_100098

theorem find_C (A B C : ℕ) (h1 : 3 * A - A = 10) (h2 : B + A = 12) (h3 : C - B = 6) : C = 13 :=
by
  sorry

end NUMINAMATH_GPT_find_C_l1000_100098


namespace NUMINAMATH_GPT_factor_tree_X_value_l1000_100037

-- Define the constants
def F : ℕ := 5 * 3
def G : ℕ := 7 * 3

-- Define the intermediate values
def Y : ℕ := 5 * F
def Z : ℕ := 7 * G

-- Final value of X
def X : ℕ := Y * Z

-- Prove the value of X
theorem factor_tree_X_value : X = 11025 := by
  sorry

end NUMINAMATH_GPT_factor_tree_X_value_l1000_100037


namespace NUMINAMATH_GPT_x1_mul_x2_l1000_100086

open Real

theorem x1_mul_x2 (x1 x2 : ℝ) (h1 : x1 + x2 = 2 * sqrt 1703) (h2 : abs (x1 - x2) = 90) : x1 * x2 = -322 := by
  sorry

end NUMINAMATH_GPT_x1_mul_x2_l1000_100086


namespace NUMINAMATH_GPT_tail_length_l1000_100012

theorem tail_length {length body tail : ℝ} (h1 : length = 30) (h2 : tail = body / 2) (h3 : length = body) : tail = 15 := by
  sorry

end NUMINAMATH_GPT_tail_length_l1000_100012


namespace NUMINAMATH_GPT_sample_size_l1000_100091

theorem sample_size (k n : ℕ) (h_ratio : 3 * n / (3 + 4 + 7) = 9) : n = 42 :=
by
  sorry

end NUMINAMATH_GPT_sample_size_l1000_100091


namespace NUMINAMATH_GPT_lucy_lovely_age_ratio_l1000_100015

theorem lucy_lovely_age_ratio (L l : ℕ) (x : ℕ) (h1 : L = 50) (h2 : 45 = x * (l - 5)) (h3 : 60 = 2 * (l + 10)) :
  (45 / (l - 5)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_lucy_lovely_age_ratio_l1000_100015


namespace NUMINAMATH_GPT_fraction_of_sides_area_of_triangle_l1000_100002

-- Part (1)
theorem fraction_of_sides (A B C : ℝ) (a b c : ℝ) (h_triangle : A + B + C = π)
  (h_sines : 2 * (Real.tan A + Real.tan B) = (Real.tan A / Real.cos B) + (Real.tan B / Real.cos A))
  (h_sine_law : c = 2) : (a + b) / c = 2 :=
sorry

-- Part (2)
theorem area_of_triangle (A B C : ℝ) (a b c : ℝ) (h_triangle : A + B + C = π)
  (h_sines : 2 * (Real.tan A + Real.tan B) = (Real.tan A / Real.cos B) + (Real.tan B / Real.cos A))
  (h_sine_law : c = 2) (h_C : C = π / 3) : (1 / 2) * a * b * Real.sin C = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_fraction_of_sides_area_of_triangle_l1000_100002


namespace NUMINAMATH_GPT_people_not_in_any_club_l1000_100090

def num_people_company := 120
def num_people_club_A := 25
def num_people_club_B := 34
def num_people_club_C := 21
def num_people_club_D := 16
def num_people_club_E := 10
def overlap_C_D := 8
def overlap_D_E := 4

theorem people_not_in_any_club :
  num_people_company - 
  (num_people_club_A + num_people_club_B + 
  (num_people_club_C + (num_people_club_D - overlap_C_D) + (num_people_club_E - overlap_D_E))) = 26 :=
by
  unfold num_people_company num_people_club_A num_people_club_B num_people_club_C num_people_club_D num_people_club_E overlap_C_D overlap_D_E
  sorry

end NUMINAMATH_GPT_people_not_in_any_club_l1000_100090


namespace NUMINAMATH_GPT_student_marks_l1000_100053

theorem student_marks (max_marks : ℕ) (passing_percentage : ℕ) (failed_by : ℕ) (student_marks : ℕ) : 
  (passing_percentage = 30) → (failed_by = 40) → (max_marks = 400) → 
  student_marks = (max_marks * passing_percentage / 100 - failed_by) → 
  student_marks = 80 :=
by {
  sorry
}

end NUMINAMATH_GPT_student_marks_l1000_100053


namespace NUMINAMATH_GPT_ratio_xyz_l1000_100069

theorem ratio_xyz (a x y z : ℝ) : 
  5 * x + 4 * y - 6 * z = a ∧
  4 * x - 5 * y + 7 * z = 27 * a ∧
  6 * x + 5 * y - 4 * z = 18 * a →
  (x :ℝ) / (y :ℝ) = 3 / 4 ∧
  (y :ℝ) / (z :ℝ) = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_xyz_l1000_100069


namespace NUMINAMATH_GPT_quadratic_roots_real_or_imaginary_l1000_100060

theorem quadratic_roots_real_or_imaginary (a b c d: ℝ) (h1: a > 0) (h2: b > 0) (h3: c > 0) (h4: d > 0) 
(h_distinct: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
∃ (A B C: ℝ), (A = a ∨ A = b ∨ A = c ∨ A = d) ∧ (B = a ∨ B = b ∨ B = c ∨ B = d) ∧ (C = a ∨ C = b ∨ C = c ∨ C = d) ∧ 
(A ≠ B) ∧ (A ≠ C) ∧ (B ≠ C) ∧ 
((1 - 4*B*C ≥ 0 ∧ 1 - 4*C*A ≥ 0 ∧ 1 - 4*A*B ≥ 0) ∨ (1 - 4*B*C < 0 ∧ 1 - 4*C*A < 0 ∧ 1 - 4*A*B < 0)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_real_or_imaginary_l1000_100060


namespace NUMINAMATH_GPT_overtime_percentage_increase_l1000_100077

-- Define the conditions.
def regular_rate : ℝ := 16
def regular_hours : ℕ := 40
def total_compensation : ℝ := 1116
def total_hours_worked : ℕ := 57
def overtime_hours : ℕ := total_hours_worked - regular_hours

-- Define the question and the answer as a proof problem.
theorem overtime_percentage_increase :
  let regular_earnings := regular_rate * regular_hours
  let overtime_earnings := total_compensation - regular_earnings
  let overtime_rate := overtime_earnings / overtime_hours
  overtime_rate > regular_rate →
  ((overtime_rate - regular_rate) / regular_rate) * 100 = 75 := 
by
  sorry

end NUMINAMATH_GPT_overtime_percentage_increase_l1000_100077


namespace NUMINAMATH_GPT_equation_solutions_equiv_l1000_100078

theorem equation_solutions_equiv (p : ℕ) (hp : p.Prime) :
  (∃ x s : ℤ, x^2 - x + 3 - p * s = 0) ↔ 
  (∃ y t : ℤ, y^2 - y + 25 - p * t = 0) :=
by { sorry }

end NUMINAMATH_GPT_equation_solutions_equiv_l1000_100078


namespace NUMINAMATH_GPT_relationship_among_a_b_and_ab_l1000_100093

noncomputable def a : ℝ := Real.log 0.4 / Real.log 0.2
noncomputable def b : ℝ := 1 - (1 / (Real.log 4 / Real.log 10))

theorem relationship_among_a_b_and_ab : a * b < a + b ∧ a + b < 0 := by
  sorry

end NUMINAMATH_GPT_relationship_among_a_b_and_ab_l1000_100093


namespace NUMINAMATH_GPT_max_correct_answers_l1000_100072

theorem max_correct_answers (a b c : ℕ) :
  a + b + c = 50 ∧ 4 * a - c = 99 ∧ b = 50 - a - c ∧ 50 - a - c ≥ 0 →
  a ≤ 29 := by
  sorry

end NUMINAMATH_GPT_max_correct_answers_l1000_100072


namespace NUMINAMATH_GPT_coin_flips_137_l1000_100087

-- Definitions and conditions
def steph_transformation_heads (x : ℤ) : ℤ := 2 * x - 1
def steph_transformation_tails (x : ℤ) : ℤ := (x + 1) / 2
def jeff_transformation_heads (y : ℤ) : ℤ := y + 8
def jeff_transformation_tails (y : ℤ) : ℤ := y - 3

-- The problem statement
theorem coin_flips_137
  (a b : ℤ)
  (h₁ : a - b = 7)
  (h₂ : 8 * a - 3 * b = 381)
  (steph_initial jeff_initial : ℤ)
  (h₃ : steph_initial = 4)
  (h₄ : jeff_initial = 4) : a + b = 137 := 
by
  sorry

end NUMINAMATH_GPT_coin_flips_137_l1000_100087


namespace NUMINAMATH_GPT_jam_jars_weight_l1000_100026

noncomputable def jars_weight 
    (initial_suitcase_weight : ℝ) 
    (perfume_weight_oz : ℝ) (num_perfume : ℕ)
    (chocolate_weight_lb : ℝ)
    (soap_weight_oz : ℝ) (num_soap : ℕ)
    (total_return_weight : ℝ)
    (oz_to_lb : ℝ) : ℝ :=
  initial_suitcase_weight 
  + (num_perfume * perfume_weight_oz) / oz_to_lb 
  + chocolate_weight_lb 
  + (num_soap * soap_weight_oz) / oz_to_lb

theorem jam_jars_weight
    (initial_suitcase_weight : ℝ := 5)
    (perfume_weight_oz : ℝ := 1.2) (num_perfume : ℕ := 5)
    (chocolate_weight_lb : ℝ := 4)
    (soap_weight_oz : ℝ := 5) (num_soap : ℕ := 2)
    (total_return_weight : ℝ := 11)
    (oz_to_lb : ℝ := 16) :
    jars_weight initial_suitcase_weight perfume_weight_oz num_perfume
      chocolate_weight_lb soap_weight_oz num_soap total_return_weight oz_to_lb + (jars_weight initial_suitcase_weight perfume_weight_oz num_perfume
      chocolate_weight_lb soap_weight_oz num_soap total_return_weight oz_to_lb) = 1 :=
by
  sorry

end NUMINAMATH_GPT_jam_jars_weight_l1000_100026


namespace NUMINAMATH_GPT_ron_chocolate_bar_cost_l1000_100088

-- Definitions of the conditions given in the problem
def cost_per_chocolate_bar : ℝ := 1.50
def sections_per_chocolate_bar : ℕ := 3
def scouts : ℕ := 15
def s'mores_needed_per_scout : ℕ := 2
def total_s'mores_needed : ℕ := scouts * s'mores_needed_per_scout
def chocolate_bars_needed : ℕ := total_s'mores_needed / sections_per_chocolate_bar
def total_cost_of_chocolate_bars : ℝ := chocolate_bars_needed * cost_per_chocolate_bar

-- Proving the question equals the answer given conditions
theorem ron_chocolate_bar_cost : total_cost_of_chocolate_bars = 15.00 := by
  sorry

end NUMINAMATH_GPT_ron_chocolate_bar_cost_l1000_100088


namespace NUMINAMATH_GPT_perpendicular_case_parallel_case_l1000_100031

variable (a b : ℝ)

-- Define the lines
def line1 (a b x y : ℝ) := a * x - b * y + 4 = 0
def line2 (a b x y : ℝ) := (a - 1) * x + y + b = 0

-- Define perpendicular condition
def perpendicular (a b : ℝ) := a * (a - 1) - b = 0

-- Define point condition
def passes_through (a b : ℝ) := -3 * a + b + 4 = 0

-- Define parallel condition
def parallel (a b : ℝ) := a * (a - 1) + b = 0

-- Define intercepts equal condition
def intercepts_equal (a b : ℝ) := b = -a

theorem perpendicular_case
    (h1 : perpendicular a b)
    (h2 : passes_through a b) :
    a = 2 ∧ b = 2 :=
sorry

theorem parallel_case
    (h1 : parallel a b)
    (h2 : intercepts_equal a b) :
    a = 2 ∧ b = -2 :=
sorry

end NUMINAMATH_GPT_perpendicular_case_parallel_case_l1000_100031


namespace NUMINAMATH_GPT_negation_of_prop_l1000_100008

variable (x : ℝ)
def prop (x : ℝ) := x ∈ Set.Ici 0 → Real.exp x ≥ 1

theorem negation_of_prop :
  (¬ ∀ x ∈ Set.Ici 0, Real.exp x ≥ 1) = ∃ x ∈ Set.Ici 0, Real.exp x < 1 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_prop_l1000_100008


namespace NUMINAMATH_GPT_remainder_when_6n_divided_by_4_l1000_100075

theorem remainder_when_6n_divided_by_4 (n : ℤ) (h : n % 4 = 1) : 6 * n % 4 = 2 := by
  sorry

end NUMINAMATH_GPT_remainder_when_6n_divided_by_4_l1000_100075


namespace NUMINAMATH_GPT_inequality_bound_l1000_100064

theorem inequality_bound (a b c d e p q : ℝ) (hpq : 0 < p ∧ p ≤ q)
  (ha : p ≤ a ∧ a ≤ q) (hb : p ≤ b ∧ b ≤ q) (hc : p ≤ c ∧ c ≤ q) 
  (hd : p ≤ d ∧ d ≤ q) (he : p ≤ e ∧ e ≤ q) :
  (a + b + c + d + e) * (1/a + 1/b + 1/c + 1/d + 1/e) 
  ≤ 25 + 6 * (Real.sqrt (p / q) - Real.sqrt (q / p))^2 :=
sorry

end NUMINAMATH_GPT_inequality_bound_l1000_100064


namespace NUMINAMATH_GPT_sum_largest_smallest_ABC_l1000_100014

def hundreds (n : ℕ) : ℕ := n / 100
def units (n : ℕ) : ℕ := n % 10
def tens (n : ℕ) : ℕ := (n / 10) % 10

theorem sum_largest_smallest_ABC : 
  (hundreds 297 = 2) ∧ (units 297 = 7) ∧ (hundreds 207 = 2) ∧ (units 207 = 7) →
  (297 + 207 = 504) :=
by
  sorry

end NUMINAMATH_GPT_sum_largest_smallest_ABC_l1000_100014


namespace NUMINAMATH_GPT_sixth_graders_more_than_seventh_l1000_100032

def pencil_cost : ℕ := 13
def eighth_graders_total : ℕ := 208
def seventh_graders_total : ℕ := 181
def sixth_graders_total : ℕ := 234

-- Number of students in each grade who bought a pencil
def seventh_graders_count := seventh_graders_total / pencil_cost
def sixth_graders_count := sixth_graders_total / pencil_cost

-- The difference in the number of sixth graders than seventh graders who bought a pencil
theorem sixth_graders_more_than_seventh : sixth_graders_count - seventh_graders_count = 4 :=
by sorry

end NUMINAMATH_GPT_sixth_graders_more_than_seventh_l1000_100032


namespace NUMINAMATH_GPT_walking_rate_ratio_l1000_100022

theorem walking_rate_ratio :
  let T := 16
  let T' := 12
  (T : ℚ) / (T' : ℚ) = (4 : ℚ) / (3 : ℚ) := 
by
  sorry

end NUMINAMATH_GPT_walking_rate_ratio_l1000_100022


namespace NUMINAMATH_GPT_algebraic_expression_value_l1000_100063

theorem algebraic_expression_value (a b : ℝ) (h : 4 * b = 3 + 4 * a) :
  a + (a - (a - (a - b) - b) - b) - b = -3 / 2 := by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1000_100063


namespace NUMINAMATH_GPT_greatest_divisor_l1000_100001

theorem greatest_divisor (d : ℕ) (h₀ : 1657 % d = 6) (h₁ : 2037 % d = 5) : d = 127 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_greatest_divisor_l1000_100001


namespace NUMINAMATH_GPT_minimum_a_l1000_100048

theorem minimum_a (a : ℝ) : (∀ x y : ℝ, 0 < x → 0 < y → (x + y) * (a / x + 4 / y) ≥ 16) → a ≥ 4 :=
by
  intros h
  -- We would provide a detailed mathematical proof here, but we use sorry for now.
  sorry

end NUMINAMATH_GPT_minimum_a_l1000_100048


namespace NUMINAMATH_GPT_remaining_sum_eq_seven_eighths_l1000_100039

noncomputable def sum_series := 
  (1 / 2) + (1 / 4) + (1 / 8) + (1 / 16) + (1 / 32) + (1 / 64)

noncomputable def removed_terms := 
  (1 / 16) + (1 / 32) + (1 / 64)

theorem remaining_sum_eq_seven_eighths : 
  sum_series - removed_terms = 7 / 8 := by
  sorry

end NUMINAMATH_GPT_remaining_sum_eq_seven_eighths_l1000_100039


namespace NUMINAMATH_GPT_girl_speed_l1000_100036

theorem girl_speed (distance time : ℝ) (h_distance : distance = 96) (h_time : time = 16) : distance / time = 6 :=
by
  sorry

end NUMINAMATH_GPT_girl_speed_l1000_100036


namespace NUMINAMATH_GPT_tomatoes_picked_l1000_100029

theorem tomatoes_picked (original_tomatoes left_tomatoes picked_tomatoes : ℕ)
  (h1 : original_tomatoes = 97)
  (h2 : left_tomatoes = 14)
  (h3 : picked_tomatoes = original_tomatoes - left_tomatoes) :
  picked_tomatoes = 83 :=
by sorry

end NUMINAMATH_GPT_tomatoes_picked_l1000_100029


namespace NUMINAMATH_GPT_arithmetic_seq_a7_l1000_100004

theorem arithmetic_seq_a7 (a : ℕ → ℤ) (d : ℤ) (h1 : ∀ (n m : ℕ), a (n + m) = a n + m * d)
  (h2 : a 4 + a 9 = 24) (h3 : a 6 = 11) :
  a 7 = 13 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_a7_l1000_100004


namespace NUMINAMATH_GPT_abs_sum_inequality_l1000_100055

theorem abs_sum_inequality (k : ℝ) : (∀ x : ℝ, |x + 2| + |x + 1| > k) → k < 1 := 
sorry

end NUMINAMATH_GPT_abs_sum_inequality_l1000_100055


namespace NUMINAMATH_GPT_simon_stamps_received_l1000_100019

theorem simon_stamps_received (initial_stamps total_stamps received_stamps : ℕ) (h1 : initial_stamps = 34) (h2 : total_stamps = 61) : received_stamps = 27 :=
by
  sorry

end NUMINAMATH_GPT_simon_stamps_received_l1000_100019


namespace NUMINAMATH_GPT_fraction_of_apples_consumed_l1000_100073

theorem fraction_of_apples_consumed (f : ℚ) 
  (bella_eats_per_day : ℚ := 6) 
  (days_per_week : ℕ := 7) 
  (grace_remaining_apples : ℚ := 504) 
  (weeks_passed : ℕ := 6) 
  (total_apples_picked : ℚ := 42 / f) :
  (total_apples_picked - (bella_eats_per_day * days_per_week * weeks_passed) = grace_remaining_apples) 
  → f = 1 / 18 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_fraction_of_apples_consumed_l1000_100073


namespace NUMINAMATH_GPT_option_not_equal_to_three_halves_l1000_100076

theorem option_not_equal_to_three_halves (d : ℚ) (h1 : d = 3/2) 
    (hA : 9/6 = 3/2) 
    (hB : 1 + 1/2 = 3/2) 
    (hC : 1 + 2/4 = 3/2)
    (hE : 1 + 6/12 = 3/2) :
  1 + 2/3 ≠ 3/2 :=
by
  sorry

end NUMINAMATH_GPT_option_not_equal_to_three_halves_l1000_100076


namespace NUMINAMATH_GPT_decimal_to_fraction_l1000_100043

theorem decimal_to_fraction (x : ℚ) (h : x = 3.68) : x = 92 / 25 := by
  sorry

end NUMINAMATH_GPT_decimal_to_fraction_l1000_100043


namespace NUMINAMATH_GPT_roots_quadratic_l1000_100046

theorem roots_quadratic (d e : ℝ) (h1 : 3 * d ^ 2 + 5 * d - 2 = 0) (h2 : 3 * e ^ 2 + 5 * e - 2 = 0) :
  (d - 1) * (e - 1) = 2 :=
sorry

end NUMINAMATH_GPT_roots_quadratic_l1000_100046


namespace NUMINAMATH_GPT_integer_solutions_exist_l1000_100079

theorem integer_solutions_exist (x y : ℤ) : 
  12 * x^2 + 7 * y^2 = 4620 ↔ 
  (x = 7 ∧ y = 24) ∨ 
  (x = -7 ∧ y = 24) ∨
  (x = 7 ∧ y = -24) ∨
  (x = -7 ∧ y = -24) ∨
  (x = 14 ∧ y = 18) ∨
  (x = -14 ∧ y = 18) ∨
  (x = 14 ∧ y = -18) ∨
  (x = -14 ∧ y = -18) :=
sorry

end NUMINAMATH_GPT_integer_solutions_exist_l1000_100079


namespace NUMINAMATH_GPT_problem_solved_l1000_100033

-- Define the function f with the given conditions
def satisfies_conditions(f : ℝ × ℝ × ℝ → ℝ) :=
  (∀ x y z t : ℝ, f (x + t, y + t, z + t) = t + f (x, y, z)) ∧
  (∀ x y z t : ℝ, f (t * x, t * y, t * z) = t * f (x, y, z)) ∧
  (∀ x y z : ℝ, f (x, y, z) = f (y, x, z)) ∧
  (∀ x y z : ℝ, f (x, y, z) = f (x, z, y))

-- We'll state the main result to be proven, without giving the proof
theorem problem_solved (f : ℝ × ℝ × ℝ → ℝ) (h : satisfies_conditions f) : f (2000, 2001, 2002) = 2001 :=
  sorry

end NUMINAMATH_GPT_problem_solved_l1000_100033


namespace NUMINAMATH_GPT_factorize_expression_l1000_100080

theorem factorize_expression (x y a : ℝ) : x * (a - y) - y * (y - a) = (x + y) * (a - y) := 
by 
  sorry

end NUMINAMATH_GPT_factorize_expression_l1000_100080


namespace NUMINAMATH_GPT_collinear_a_b_l1000_100061

-- Define the vectors a and b
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (1, -2)

-- Definition of collinearity of vectors
def collinear (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2) ∨ v = (k * u.1, k * u.2)

-- Statement to prove
theorem collinear_a_b : collinear a b :=
by
  sorry

end NUMINAMATH_GPT_collinear_a_b_l1000_100061


namespace NUMINAMATH_GPT_michael_has_more_flying_robots_l1000_100068

theorem michael_has_more_flying_robots (tom_robots michael_robots : ℕ) (h_tom : tom_robots = 3) (h_michael : michael_robots = 12) :
  michael_robots / tom_robots = 4 :=
by
  sorry

end NUMINAMATH_GPT_michael_has_more_flying_robots_l1000_100068


namespace NUMINAMATH_GPT_gunny_bag_capacity_in_tons_l1000_100083

def ton_to_pounds := 2200
def pound_to_ounces := 16
def packets := 1760
def packet_weight_pounds := 16
def packet_weight_ounces := 4

theorem gunny_bag_capacity_in_tons :
  ((packets * (packet_weight_pounds + (packet_weight_ounces / pound_to_ounces))) / ton_to_pounds) = 13 :=
sorry

end NUMINAMATH_GPT_gunny_bag_capacity_in_tons_l1000_100083


namespace NUMINAMATH_GPT_jersey_cost_difference_l1000_100044

theorem jersey_cost_difference :
  let jersey_cost := 115
  let tshirt_cost := 25
  jersey_cost - tshirt_cost = 90 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_jersey_cost_difference_l1000_100044


namespace NUMINAMATH_GPT_business_total_profit_l1000_100082

noncomputable def total_profit (spending_ratio income_ratio total_income : ℕ) : ℕ :=
  let total_parts := spending_ratio + income_ratio
  let one_part_value := total_income / income_ratio
  let spending := spending_ratio * one_part_value
  total_income - spending

theorem business_total_profit :
  total_profit 5 9 108000 = 48000 :=
by
  -- We omit the proof steps, as instructed.
  sorry

end NUMINAMATH_GPT_business_total_profit_l1000_100082


namespace NUMINAMATH_GPT_parallel_planes_sufficient_not_necessary_for_perpendicular_lines_l1000_100065

variables {Point Line Plane : Type}
variables (α β : Plane) (ℓ m : Line) (point_on_line_ℓ : Point) (point_on_line_m : Point)

-- Definitions of conditions
def line_perpendicular_to_plane (ℓ : Line) (α : Plane) : Prop := sorry
def line_contained_in_plane (m : Line) (β : Plane) : Prop := sorry
def planes_parallel (α β : Plane) : Prop := sorry
def line_perpendicular_to_line (ℓ m : Line) : Prop := sorry

axiom h1 : line_perpendicular_to_plane ℓ α
axiom h2 : line_contained_in_plane m β

-- Statement of the proof problem
theorem parallel_planes_sufficient_not_necessary_for_perpendicular_lines : 
  (planes_parallel α β → line_perpendicular_to_line ℓ m) ∧ 
  ¬ (line_perpendicular_to_line ℓ m → planes_parallel α β) :=
  sorry

end NUMINAMATH_GPT_parallel_planes_sufficient_not_necessary_for_perpendicular_lines_l1000_100065


namespace NUMINAMATH_GPT_second_investment_amount_l1000_100005

/-
A $500 investment and another investment have a combined yearly return of 8.5 percent of the total of the two investments.
The $500 investment has a yearly return of 7 percent.
The other investment has a yearly return of 9 percent.
Prove that the amount of the second investment is $1500.
-/

theorem second_investment_amount :
  ∃ x : ℝ, 35 + 0.09 * x = 0.085 * (500 + x) → x = 1500 :=
by
  sorry

end NUMINAMATH_GPT_second_investment_amount_l1000_100005


namespace NUMINAMATH_GPT_real_possible_b_values_quadratic_non_real_roots_l1000_100047

theorem real_possible_b_values_quadratic_non_real_roots :
  {b : ℝ | ∃ (a c : ℝ), a = 1 ∧ c = 16 ∧ (b^2 - 4*a*c < 0)} = {b : ℝ | -8 < b ∧ b < 8} :=
by 
  sorry

end NUMINAMATH_GPT_real_possible_b_values_quadratic_non_real_roots_l1000_100047


namespace NUMINAMATH_GPT_add_number_l1000_100034

theorem add_number (x : ℕ) (h : 43 + x = 81) : x + 25 = 63 :=
by {
  -- Since this is focusing on the structure and statement no proof steps are required
  sorry
}

end NUMINAMATH_GPT_add_number_l1000_100034


namespace NUMINAMATH_GPT_calvin_total_insects_l1000_100070

def R : ℕ := 15
def S : ℕ := 2 * R - 8
def C : ℕ := 11 -- rounded from (1/2) * R + 3
def P : ℕ := 3 * S + 7
def B : ℕ := 4 * C - 2
def E : ℕ := 3 * (R + S + C + P + B)
def total_insects : ℕ := R + S + C + P + B + E

theorem calvin_total_insects : total_insects = 652 :=
by
  -- service the proof here.
  sorry

end NUMINAMATH_GPT_calvin_total_insects_l1000_100070


namespace NUMINAMATH_GPT_solve_cubic_eq_l1000_100006

theorem solve_cubic_eq (x : ℝ) (h1 : (x + 1)^3 = x^3) (h2 : 0 ≤ x) (h3 : x < 1) : x = 0 :=
by
  sorry

end NUMINAMATH_GPT_solve_cubic_eq_l1000_100006


namespace NUMINAMATH_GPT_cost_of_cd_l1000_100089

theorem cost_of_cd 
  (cost_film : ℕ) (cost_book : ℕ) (total_spent : ℕ) (num_cds : ℕ) (total_cost_films : ℕ)
  (total_cost_books : ℕ) (cost_cd : ℕ) : 
  cost_film = 5 → cost_book = 4 → total_spent = 79 →
  total_cost_films = 9 * cost_film → total_cost_books = 4 * cost_book →
  total_spent = total_cost_films + total_cost_books + num_cds * cost_cd →
  num_cds = 6 →
  cost_cd = 3 := 
by {
  -- proof would go here
  sorry
}

end NUMINAMATH_GPT_cost_of_cd_l1000_100089


namespace NUMINAMATH_GPT_jack_birth_year_l1000_100052

theorem jack_birth_year 
  (first_amc8_year : ℕ) 
  (amc8_annual : ℕ → ℕ → ℕ) 
  (jack_age_ninth_amc8 : ℕ) 
  (ninth_amc8_year : amc8_annual first_amc8_year 9 = 1998) 
  (jack_age_in_ninth_amc8 : jack_age_ninth_amc8 = 15)
  : (1998 - jack_age_ninth_amc8 = 1983) := by
  sorry

end NUMINAMATH_GPT_jack_birth_year_l1000_100052


namespace NUMINAMATH_GPT_T_100_gt_T_99_l1000_100023

-- Definition: T(n) denotes the number of ways to place n objects of weights 1, 2, ..., n on a balance such that the sum of the weights in each pan is the same.
def T (n : ℕ) : ℕ := sorry

-- Theorem we need to prove
theorem T_100_gt_T_99 : T 100 > T 99 := 
sorry

end NUMINAMATH_GPT_T_100_gt_T_99_l1000_100023


namespace NUMINAMATH_GPT_max_elements_in_set_l1000_100009

theorem max_elements_in_set (S : Finset ℕ) (hS : ∀ (a b : ℕ), a ≠ b → a ∈ S → b ∈ S → 
  ∃ (k : ℕ) (c d : ℕ), c < d ∧ c ∈ S ∧ d ∈ S ∧ a + b = c^k * d) :
  S.card ≤ 48 :=
sorry

end NUMINAMATH_GPT_max_elements_in_set_l1000_100009


namespace NUMINAMATH_GPT_julia_mile_time_l1000_100051

variable (x : ℝ)

theorem julia_mile_time
  (h1 : ∀ x, x > 0)
  (h2 : ∀ x, x <= 13)
  (h3 : 65 = 5 * 13)
  (h4 : 50 = 65 - 15)
  (h5 : 50 = 5 * x) :
  x = 10 := by
  sorry

end NUMINAMATH_GPT_julia_mile_time_l1000_100051


namespace NUMINAMATH_GPT_factorable_quadratic_l1000_100045

theorem factorable_quadratic (b : Int) : 
  (∃ m n p q : Int, 35 * m * p = 35 ∧ m * q + n * p = b ∧ n * q = 35) ↔ (∃ k : Int, b = 2 * k) :=
sorry

end NUMINAMATH_GPT_factorable_quadratic_l1000_100045


namespace NUMINAMATH_GPT_impossible_to_fill_grid_l1000_100050

def is_impossible : Prop :=
  ∀ (grid : Fin 3 → Fin 3 → ℕ), 
  (∀ i j, grid i j ≠ grid i (j + 1) ∧ grid i j ≠ grid (i + 1) j) →
  (∀ i, (grid i 0) * (grid i 1) * (grid i 2) = 2005) →
  (∀ j, (grid 0 j) * (grid 1 j) * (grid 2 j) = 2005) →
  (grid 0 0) * (grid 1 1) * (grid 2 2) = 2005 →
  (grid 0 2) * (grid 1 1) * (grid 2 0) = 2005 →
  False

theorem impossible_to_fill_grid : is_impossible :=
  sorry

end NUMINAMATH_GPT_impossible_to_fill_grid_l1000_100050


namespace NUMINAMATH_GPT_initial_HNO3_percentage_is_correct_l1000_100038

def initial_percentage_of_HNO3 (P : ℚ) : Prop :=
  let initial_volume := 60
  let added_volume := 18
  let final_volume := 78
  let final_percentage := 50
  (P / 100) * initial_volume + added_volume = (final_percentage / 100) * final_volume

theorem initial_HNO3_percentage_is_correct :
  initial_percentage_of_HNO3 35 :=
by
  sorry

end NUMINAMATH_GPT_initial_HNO3_percentage_is_correct_l1000_100038


namespace NUMINAMATH_GPT_perimeter_of_ABFCDE_l1000_100018

theorem perimeter_of_ABFCDE {side : ℝ} (h : side = 12) : 
  ∃ perimeter : ℝ, perimeter = 84 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_ABFCDE_l1000_100018


namespace NUMINAMATH_GPT_volume_set_points_sum_l1000_100035

-- Defining the problem conditions
def rectangular_parallelepiped_length : ℝ := 5
def rectangular_parallelepiped_width : ℝ := 6
def rectangular_parallelepiped_height : ℝ := 7
def unit_extension : ℝ := 1

-- Defining what we need to prove
theorem volume_set_points_sum :
  let V_box : ℝ := rectangular_parallelepiped_length * rectangular_parallelepiped_width * rectangular_parallelepiped_height
  let V_ext : ℝ := 2 * (unit_extension * rectangular_parallelepiped_length * rectangular_parallelepiped_width 
                  + unit_extension * rectangular_parallelepiped_length * rectangular_parallelepiped_height 
                  + unit_extension * rectangular_parallelepiped_width * rectangular_parallelepiped_height)
  let V_cyl : ℝ := 18 * π
  let V_sph : ℝ := (4 / 3) * π
  let V_total : ℝ := V_box + V_ext + V_cyl + V_sph
  let m : ℕ := 1272
  let n : ℕ := 58
  let p : ℕ := 3
  V_total = (m : ℝ) + (n : ℝ) * π / (p : ℝ) ∧ (m + n + p = 1333)
  := by
  sorry

end NUMINAMATH_GPT_volume_set_points_sum_l1000_100035


namespace NUMINAMATH_GPT_find_s_for_g3_eq_0_l1000_100081

def g (x s : ℝ) : ℝ := 3 * x^5 - 2 * x^4 + x^3 - 4 * x^2 + 5 * x + s

theorem find_s_for_g3_eq_0 : (g 3 s = 0) ↔ (s = -573) :=
by
  sorry

end NUMINAMATH_GPT_find_s_for_g3_eq_0_l1000_100081


namespace NUMINAMATH_GPT_find_root_interval_l1000_100000

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem find_root_interval : ∃ k : ℕ, (f 1 < 0 ∧ f 2 > 0) → k = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_root_interval_l1000_100000


namespace NUMINAMATH_GPT_unique_x1_sequence_l1000_100049

open Nat

theorem unique_x1_sequence (x1 : ℝ) (x : ℕ → ℝ)
  (h₀ : x 1 = x1)
  (h₁ : ∀ n, x (n + 1) = x n * (x n + 1 / (n + 1))) :
  (∃! x1, (0 < x1 ∧ x1 < 1) ∧ 
   (∀ n, 0 < x n ∧ x n < x (n + 1) ∧ x (n + 1) < 1)) := sorry

end NUMINAMATH_GPT_unique_x1_sequence_l1000_100049


namespace NUMINAMATH_GPT_micah_water_intake_l1000_100024

def morning : ℝ := 1.5
def early_afternoon : ℝ := 2 * morning
def late_afternoon : ℝ := 3 * morning
def evening : ℝ := late_afternoon - 0.25 * late_afternoon
def night : ℝ := 2 * evening
def total_water_intake : ℝ := morning + early_afternoon + late_afternoon + evening + night

theorem micah_water_intake :
  total_water_intake = 19.125 := by
  sorry

end NUMINAMATH_GPT_micah_water_intake_l1000_100024


namespace NUMINAMATH_GPT_range_of_a_l1000_100059

theorem range_of_a (a x y : ℝ)
  (h1 : x + y = 3 * a + 4)
  (h2 : x - y = 7 * a - 4)
  (h3 : 3 * x - 2 * y < 11) : a < 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1000_100059


namespace NUMINAMATH_GPT_pages_left_l1000_100058

theorem pages_left (total_pages read_fraction : ℕ) (h_total_pages : total_pages = 396) (h_read_fraction : read_fraction = 1/3) : total_pages * (1 - read_fraction) = 264 := 
by
  sorry

end NUMINAMATH_GPT_pages_left_l1000_100058


namespace NUMINAMATH_GPT_division_remainder_l1000_100071

def polynomial (x: ℤ) : ℤ := 3 * x^7 - x^6 - 7 * x^5 + 2 * x^3 + 4 * x^2 - 11
def divisor (x: ℤ) : ℤ := 2 * x - 4

theorem division_remainder : (polynomial 2) = 117 := 
  by 
  -- We state what needs to be proven here formally
  sorry

end NUMINAMATH_GPT_division_remainder_l1000_100071


namespace NUMINAMATH_GPT_fruit_salad_cherries_l1000_100095

variable (b r g c : ℕ)

theorem fruit_salad_cherries :
  (b + r + g + c = 350) ∧
  (r = 3 * b) ∧
  (g = 4 * c) ∧
  (c = 5 * r) →
  c = 66 :=
by
  sorry

end NUMINAMATH_GPT_fruit_salad_cherries_l1000_100095


namespace NUMINAMATH_GPT_work_problem_correct_l1000_100096

noncomputable def work_problem : Prop :=
  let A := 1 / 36
  let C := 1 / 6
  let total_rate := 1 / 4
  ∃ B : ℝ, (A + B + C = total_rate) ∧ (B = 1 / 18)

-- Create the theorem statement which says if the conditions are met,
-- then the rate of b must be 1/18 and the number of days b alone takes to
-- finish the work is 18.
theorem work_problem_correct (A C total_rate B : ℝ) (h1 : A = 1 / 36) (h2 : C = 1 / 6) (h3 : total_rate = 1 / 4) 
(h4 : A + B + C = total_rate) : B = 1 / 18 ∧ (1 / B = 18) :=
  by
  sorry

end NUMINAMATH_GPT_work_problem_correct_l1000_100096


namespace NUMINAMATH_GPT_Marcus_fit_pies_l1000_100010

theorem Marcus_fit_pies (x : ℕ) 
(h1 : ∀ b, (7 * b - 8) = 27) : x = 5 := by
  sorry

end NUMINAMATH_GPT_Marcus_fit_pies_l1000_100010


namespace NUMINAMATH_GPT_units_digit_6_l1000_100074

theorem units_digit_6 (p : ℤ) (hp : 0 < p % 10) (h1 : (p^3 % 10) = (p^2 % 10)) (h2 : (p + 2) % 10 = 8) : p % 10 = 6 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_6_l1000_100074


namespace NUMINAMATH_GPT_sequence_a_10_value_l1000_100027

theorem sequence_a_10_value : 
  ∀ (a : ℕ → ℤ), a 1 = 3 → (∀ n : ℕ, 0 < n → a (n + 1) - a n = 2) → a 10 = 21 := 
by 
  intros a h1 hdiff
  sorry

end NUMINAMATH_GPT_sequence_a_10_value_l1000_100027


namespace NUMINAMATH_GPT_one_thirds_in_nine_halves_l1000_100057

theorem one_thirds_in_nine_halves : (9 / 2) / (1 / 3) = 13 := by
  sorry

end NUMINAMATH_GPT_one_thirds_in_nine_halves_l1000_100057


namespace NUMINAMATH_GPT_smallest_n_l1000_100094

theorem smallest_n (n : ℕ) (h : 23 * n ≡ 789 [MOD 11]) : n = 9 :=
sorry

end NUMINAMATH_GPT_smallest_n_l1000_100094


namespace NUMINAMATH_GPT_find_value_of_expression_l1000_100085

-- Conditions as provided
axiom given_condition : ∃ (x : ℕ), 3^x + 3^x + 3^x + 3^x = 2187

-- Proof statement
theorem find_value_of_expression : (exists (x : ℕ), (3^x + 3^x + 3^x + 3^x = 2187) ∧ ((x + 2) * (x - 2) = 21)) :=
sorry

end NUMINAMATH_GPT_find_value_of_expression_l1000_100085


namespace NUMINAMATH_GPT_polar_distance_l1000_100013

theorem polar_distance {r1 θ1 r2 θ2 : ℝ} (A : r1 = 1 ∧ θ1 = π/6) (B : r2 = 3 ∧ θ2 = 5*π/6) : 
  (r1^2 + r2^2 - 2*r1*r2 * Real.cos (θ2 - θ1)) = 13 :=
  sorry

end NUMINAMATH_GPT_polar_distance_l1000_100013


namespace NUMINAMATH_GPT_profit_per_meter_correct_l1000_100028

-- Define the conditions
def total_meters := 40
def total_profit := 1400

-- Define the profit per meter calculation
def profit_per_meter := total_profit / total_meters

-- Theorem stating the profit per meter is Rs. 35
theorem profit_per_meter_correct : profit_per_meter = 35 := by
  sorry

end NUMINAMATH_GPT_profit_per_meter_correct_l1000_100028


namespace NUMINAMATH_GPT_price_of_child_ticket_l1000_100003

theorem price_of_child_ticket (total_seats : ℕ) (adult_ticket_price : ℕ) (total_revenue : ℕ)
  (child_tickets_sold : ℕ) (child_ticket_price : ℕ) :
  total_seats = 80 →
  adult_ticket_price = 12 →
  total_revenue = 519 →
  child_tickets_sold = 63 →
  (17 * adult_ticket_price) + (child_tickets_sold * child_ticket_price) = total_revenue →
  child_ticket_price = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_price_of_child_ticket_l1000_100003


namespace NUMINAMATH_GPT_youngest_child_age_l1000_100021

theorem youngest_child_age (total_bill mother_cost twin_age_cost total_age : ℕ) (twin_age youngest_age : ℕ) 
  (h1 : total_bill = 1485) (h2 : mother_cost = 695) (h3 : twin_age_cost = 65) 
  (h4 : total_age = (total_bill - mother_cost) / twin_age_cost)
  (h5 : total_age = 2 * twin_age + youngest_age) :
  youngest_age = 2 :=
by
  -- sorry: Proof to be completed later
  sorry

end NUMINAMATH_GPT_youngest_child_age_l1000_100021


namespace NUMINAMATH_GPT_max_area_of_rectangle_l1000_100041

-- Define the parameters and the problem
def perimeter := 150
def half_perimeter := perimeter / 2

theorem max_area_of_rectangle (x : ℕ) (y : ℕ) 
  (h1 : x + y = half_perimeter)
  (h2 : x > 0) (h3 : y > 0) :
  (∃ x y, x * y ≤ 1406) := 
sorry

end NUMINAMATH_GPT_max_area_of_rectangle_l1000_100041


namespace NUMINAMATH_GPT_joe_fish_times_sam_l1000_100066

-- Define the number of fish Sam has
def sam_fish : ℕ := 7

-- Define the number of fish Harry has
def harry_fish : ℕ := 224

-- Define the number of times Joe has as many fish as Sam
def joe_times_sam (x : ℕ) : Prop :=
  4 * (sam_fish * x) = harry_fish

-- The theorem to prove Joe has 8 times as many fish as Sam
theorem joe_fish_times_sam : ∃ x, joe_times_sam x ∧ x = 8 :=
by
  sorry

end NUMINAMATH_GPT_joe_fish_times_sam_l1000_100066


namespace NUMINAMATH_GPT_max_ab_at_extremum_l1000_100062

noncomputable def f (a b x : ℝ) : ℝ := 4*x^3 - a*x^2 - 2*b*x + 2

theorem max_ab_at_extremum (a b : ℝ) (h0: a > 0) (h1 : b > 0) (h2 : ∃ x, f a b x = 4*x^3 - a*x^2 - 2*b*x + 2 ∧ x = 1 ∧ 12*x^2 - 2*a*x - 2*b = 0) :
  ab ≤ 9 := 
sorry  -- proof not required

end NUMINAMATH_GPT_max_ab_at_extremum_l1000_100062


namespace NUMINAMATH_GPT_total_hours_verification_l1000_100040

def total_hours_data_analytics : ℕ := 
  let weekly_class_homework_hours := (2 * 3 + 1 * 4 + 4) * 24 
  let lab_project_hours := 8 * 6 + (10 + 14 + 18)
  weekly_class_homework_hours + lab_project_hours

def total_hours_programming : ℕ :=
  let weekly_hours := (2 * 2 + 2 * 4 + 6) * 24
  weekly_hours

def total_hours_statistics : ℕ :=
  let weekly_class_lab_project_hours := (2 * 3 + 1 * 2 + 3) * 24
  let exam_study_hours := 9 * 5
  weekly_class_lab_project_hours + exam_study_hours

def total_hours_all_courses : ℕ :=
  total_hours_data_analytics + total_hours_programming + total_hours_statistics

theorem total_hours_verification : 
    total_hours_all_courses = 1167 := 
by 
    sorry

end NUMINAMATH_GPT_total_hours_verification_l1000_100040


namespace NUMINAMATH_GPT_y1_lt_y2_of_linear_function_l1000_100084

theorem y1_lt_y2_of_linear_function (y1 y2 : ℝ) (h1 : y1 = 2 * (-3) + 1) (h2 : y2 = 2 * 2 + 1) : y1 < y2 :=
by
  sorry

end NUMINAMATH_GPT_y1_lt_y2_of_linear_function_l1000_100084


namespace NUMINAMATH_GPT_updated_mean_l1000_100011

theorem updated_mean
  (n : ℕ) (obs_mean : ℝ) (decrement : ℝ)
  (h1 : n = 50) (h2 : obs_mean = 200) (h3 : decrement = 47) :
  (obs_mean - decrement) = 153 := by
  sorry

end NUMINAMATH_GPT_updated_mean_l1000_100011


namespace NUMINAMATH_GPT_find_phi_increasing_intervals_l1000_100054

open Real

-- Defining the symmetry condition
noncomputable def symmetric_phi (x_sym : ℝ) (k : ℤ) (phi : ℝ): Prop :=
  2 * x_sym + phi = k * π + π / 2

-- Finding the value of phi given the conditions
theorem find_phi (x_sym : ℝ) (phi : ℝ) (k : ℤ) 
  (h_sym: symmetric_phi x_sym k phi) (h_phi_bound : -π < phi ∧ phi < 0)
  (h_xsym: x_sym = π / 8) :
  phi = -3 * π / 4 :=
by
  sorry

-- Defining the function and its increasing intervals
noncomputable def f (x : ℝ) (phi : ℝ) : ℝ := sin (2 * x + phi)

-- Finding the increasing intervals of f on the interval [0, π]
theorem increasing_intervals (phi : ℝ) 
  (h_phi: phi = -3 * π / 4) :
  ∀ x, (0 ≤ x ∧ x ≤ π) → 
    (π / 8 ≤ x ∧ x ≤ 5 * π / 8) :=
by
  sorry

end NUMINAMATH_GPT_find_phi_increasing_intervals_l1000_100054


namespace NUMINAMATH_GPT_rth_term_of_arithmetic_progression_l1000_100017

noncomputable def Sn (n : ℕ) : ℕ := 2 * n + 3 * n^2 + n^3

theorem rth_term_of_arithmetic_progression (r : ℕ) : 
  (Sn r - Sn (r - 1)) = 3 * r^2 + 5 * r - 2 :=
by sorry

end NUMINAMATH_GPT_rth_term_of_arithmetic_progression_l1000_100017


namespace NUMINAMATH_GPT_mass_percentage_of_Ca_in_CaO_is_correct_l1000_100067

noncomputable def molarMass_Ca : ℝ := 40.08
noncomputable def molarMass_O : ℝ := 16.00
noncomputable def molarMass_CaO : ℝ := molarMass_Ca + molarMass_O
noncomputable def massPercentageCaInCaO : ℝ := (molarMass_Ca / molarMass_CaO) * 100

theorem mass_percentage_of_Ca_in_CaO_is_correct :
  massPercentageCaInCaO = 71.47 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_mass_percentage_of_Ca_in_CaO_is_correct_l1000_100067
