import Mathlib

namespace NUMINAMATH_GPT_perimeter_of_triangle_l1226_122677

-- The given condition about the average length of the triangle sides.
def average_side_length (a b c : ℝ) (h : (a + b + c) / 3 = 12) : Prop :=
  a + b + c = 36

-- The theorem to prove the perimeter of triangle ABC.
theorem perimeter_of_triangle (a b c : ℝ) (h : (a + b + c) / 3 = 12) : a + b + c = 36 :=
  by
    sorry

end NUMINAMATH_GPT_perimeter_of_triangle_l1226_122677


namespace NUMINAMATH_GPT_second_derivative_of_y_l1226_122673

noncomputable def y (x : ℝ) : ℝ := x^2 * Real.log (1 + Real.sin x)

theorem second_derivative_of_y :
  (deriv^[2] y) x = 
  2 * Real.log (1 + Real.sin x) + (4 * x * Real.cos x - x ^ 2) / (1 + Real.sin x) :=
sorry

end NUMINAMATH_GPT_second_derivative_of_y_l1226_122673


namespace NUMINAMATH_GPT_value_of_m_l1226_122617

def f (x : ℚ) : ℚ := 3 * x^3 - 1 / x + 2
def g (x : ℚ) (m : ℚ) : ℚ := 2 * x^3 - 3 * x + m
def h (x : ℚ) : ℚ := x^2

theorem value_of_m : f 3 - g 3 (122 / 3) + h 3 = 5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_l1226_122617


namespace NUMINAMATH_GPT_subtraction_result_l1226_122657

theorem subtraction_result :
  5.3567 - 2.1456 - 1.0211 = 2.1900 := 
sorry

end NUMINAMATH_GPT_subtraction_result_l1226_122657


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1226_122678

theorem geometric_sequence_common_ratio {a : ℕ → ℝ} 
    (h1 : a 1 = 1) 
    (h4 : a 4 = 1 / 64) 
    (geom_seq : ∀ n, ∃ r, a (n + 1) = a n * r) : 
       
    ∃ q, (∀ n, a n = 1 * (q ^ (n - 1))) ∧ (a 4 = 1 * (q ^ 3)) ∧ q = 1 / 4 := 
by
    sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1226_122678


namespace NUMINAMATH_GPT_melanie_phil_ages_l1226_122685

theorem melanie_phil_ages (A B : ℕ) 
  (h : (A + 10) * (B + 10) = A * B + 400) :
  (A + 6) + (B + 6) = 42 :=
by
  sorry

end NUMINAMATH_GPT_melanie_phil_ages_l1226_122685


namespace NUMINAMATH_GPT_ratio_of_white_marbles_l1226_122610

theorem ratio_of_white_marbles (total_marbles yellow_marbles red_marbles : ℕ)
    (h1 : total_marbles = 50)
    (h2 : yellow_marbles = 12)
    (h3 : red_marbles = 7)
    (green_marbles : ℕ)
    (h4 : green_marbles = yellow_marbles - yellow_marbles / 2) :
    (total_marbles - (yellow_marbles + green_marbles + red_marbles)) / total_marbles = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_white_marbles_l1226_122610


namespace NUMINAMATH_GPT_problem_conditions_l1226_122628

noncomputable def f (a b x : ℝ) : ℝ := abs (x + a) + abs (2 * x - b)

theorem problem_conditions (ha : 0 < a) (hb : 0 < b) 
  (hmin : ∃ x : ℝ, f a b x = 1) : 
  2 * a + b = 2 ∧ 
  ∀ (t : ℝ), (∀ a b : ℝ, 
    (0 < a) → (0 < b) → (a + 2 * b ≥ t * a * b)) → 
  t ≤ 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_conditions_l1226_122628


namespace NUMINAMATH_GPT_max_weight_each_shipping_box_can_hold_l1226_122664

noncomputable def max_shipping_box_weight_pounds 
  (total_plates : ℕ)
  (weight_per_plate_ounces : ℕ)
  (plates_removed : ℕ)
  (ounce_to_pound : ℕ) : ℕ :=
  (total_plates - plates_removed) * weight_per_plate_ounces / ounce_to_pound

theorem max_weight_each_shipping_box_can_hold :
  max_shipping_box_weight_pounds 38 10 6 16 = 20 :=
by
  sorry

end NUMINAMATH_GPT_max_weight_each_shipping_box_can_hold_l1226_122664


namespace NUMINAMATH_GPT_smallest_and_largest_x_l1226_122670

theorem smallest_and_largest_x (x : ℝ) :
  (|5 * x - 4| = 29) → ((x = -5) ∨ (x = 6.6)) :=
by
  sorry

end NUMINAMATH_GPT_smallest_and_largest_x_l1226_122670


namespace NUMINAMATH_GPT_find_x_l1226_122616

open Nat

theorem find_x (n : ℕ) (x : ℕ) (h1 : x = 2^n - 32)
  (h2 : (3 : ℕ) ∣ x)
  (h3 : (factors x).length = 3) :
  x = 480 ∨ x = 2016 := by
  sorry

end NUMINAMATH_GPT_find_x_l1226_122616


namespace NUMINAMATH_GPT_intersection_M_N_l1226_122680

def M : Set ℝ := { x | x^2 + x - 6 < 0 }
def N : Set ℝ := { x | |x - 1| ≤ 2 }

theorem intersection_M_N :
  M ∩ N = { x | -1 ≤ x ∧ x < 2 } :=
sorry

end NUMINAMATH_GPT_intersection_M_N_l1226_122680


namespace NUMINAMATH_GPT_speed_of_second_fragment_l1226_122693

noncomputable def magnitude_speed_of_second_fragment 
  (u : ℝ) (t : ℝ) (g : ℝ) (v_x1 : ℝ) (v_y1 : ℝ := - (u - g * t)) 
  (v_x2 : ℝ := -v_x1) (v_y2 : ℝ := v_y1) : ℝ :=
Real.sqrt ((v_x2 ^ 2) + (v_y2 ^ 2))

theorem speed_of_second_fragment 
  (u : ℝ) (t : ℝ) (g : ℝ) (v_x1 : ℝ) 
  (h_u : u = 20) (h_t : t = 3) (h_g : g = 10) (h_vx1 : v_x1 = 48) :
  magnitude_speed_of_second_fragment u t g v_x1 = Real.sqrt 2404 :=
by
  -- Proof
  sorry

end NUMINAMATH_GPT_speed_of_second_fragment_l1226_122693


namespace NUMINAMATH_GPT_album_pages_l1226_122669

variable (x y : ℕ)

theorem album_pages :
  (20 * x < y) ∧
  (23 * x > y) ∧
  (21 * x + y = 500) →
  x = 12 := by
  sorry

end NUMINAMATH_GPT_album_pages_l1226_122669


namespace NUMINAMATH_GPT_find_ratio_l1226_122656

variables (a b c d : ℝ)

def condition1 : Prop := a / b = 5
def condition2 : Prop := b / c = 1 / 4
def condition3 : Prop := c^2 / d = 16

theorem find_ratio (h1 : condition1 a b) (h2 : condition2 b c) (h3 : condition3 c d) :
  d / a = 1 / 25 :=
sorry

end NUMINAMATH_GPT_find_ratio_l1226_122656


namespace NUMINAMATH_GPT_square_garden_perimeter_l1226_122659

theorem square_garden_perimeter (A : ℝ) (s : ℝ) (N : ℝ) 
  (h1 : A = 9)
  (h2 : s^2 = A)
  (h3 : N = 4 * s) 
  : N = 12 := 
by
  sorry

end NUMINAMATH_GPT_square_garden_perimeter_l1226_122659


namespace NUMINAMATH_GPT_largest_consecutive_even_sum_l1226_122676

theorem largest_consecutive_even_sum (a b c : ℤ) (h1 : b = a+2) (h2 : c = a+4) (h3 : a + b + c = 312) : c = 106 := 
by 
  sorry

end NUMINAMATH_GPT_largest_consecutive_even_sum_l1226_122676


namespace NUMINAMATH_GPT_question1_question2_l1226_122626

-- Question 1
theorem question1 (a : ℝ) (h : a = 1 / 2) :
  let A := {x | -1 / 2 < x ∧ x < 2}
  let B := {x | 0 < x ∧ x < 1}
  A ∩ B = {x | 0 < x ∧ x < 1} :=
by
  sorry

-- Question 2
theorem question2 (a : ℝ) :
  let A := {x | a - 1 < x ∧ x < 2 * a + 1}
  let B := {x | 0 < x ∧ x < 1}
  (A ∩ B = ∅) ↔ (a ≤ -1/2 ∨ a ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_question1_question2_l1226_122626


namespace NUMINAMATH_GPT_log_inequality_l1226_122683

noncomputable def a : ℝ := Real.log 3.6 / Real.log 2
noncomputable def b : ℝ := Real.log 3.2 / Real.log 4
noncomputable def c : ℝ := Real.log 3.6 / Real.log 4

theorem log_inequality : a > c ∧ c > b :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_log_inequality_l1226_122683


namespace NUMINAMATH_GPT_line_equation_solution_l1226_122636

noncomputable def line_equation (P : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) : Prop :=
  ∃ (l : ℝ → ℝ), (l P.fst = P.snd) ∧ (∀ (x : ℝ), l x = 4 * x - 2) ∨ (∀ (x : ℝ), x = 1)

theorem line_equation_solution : line_equation (1, 2) (2, 3) (0, -5) :=
sorry

end NUMINAMATH_GPT_line_equation_solution_l1226_122636


namespace NUMINAMATH_GPT_good_walker_catches_up_l1226_122643

-- Definitions based on the conditions in the problem
def steps_good_walker := 100
def steps_bad_walker := 60
def initial_lead := 100

-- Mathematical proof problem statement
theorem good_walker_catches_up :
  ∃ x : ℕ, x = initial_lead + (steps_bad_walker * x / steps_good_walker) :=
sorry

end NUMINAMATH_GPT_good_walker_catches_up_l1226_122643


namespace NUMINAMATH_GPT_angle_of_inclination_l1226_122638

theorem angle_of_inclination (θ : ℝ) (h_range : 0 ≤ θ ∧ θ < 180)
  (h_line : ∀ x y : ℝ, x + y - 1 = 0 → x = -y + 1) :
  θ = 135 :=
by 
  sorry

end NUMINAMATH_GPT_angle_of_inclination_l1226_122638


namespace NUMINAMATH_GPT_sum_of_remainders_l1226_122672

theorem sum_of_remainders (d e f g : ℕ)
  (hd : d % 30 = 15)
  (he : e % 30 = 5)
  (hf : f % 30 = 10)
  (hg : g % 30 = 20) :
  (d + e + f + g) % 30 = 20 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_l1226_122672


namespace NUMINAMATH_GPT_quadratic_inequality_ab_l1226_122644

theorem quadratic_inequality_ab (a b : ℝ) 
  (h1 : ∀ x : ℝ, (a * x^2 + b * x + 1 > 0) ↔ -1 < x ∧ x < 1 / 3) :
  a * b = -6 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_quadratic_inequality_ab_l1226_122644


namespace NUMINAMATH_GPT_equivalent_eq_l1226_122631

variable {x y : ℝ}

theorem equivalent_eq (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2 : y ≠ 5) :
  (3 / x + 2 / y = 1 / 3) ↔ (y = 6 * x / (x - 9)) :=
by
  sorry

end NUMINAMATH_GPT_equivalent_eq_l1226_122631


namespace NUMINAMATH_GPT_probability_white_second_given_red_first_l1226_122632

theorem probability_white_second_given_red_first :
  let total_balls := 8
  let red_balls := 5
  let white_balls := 3
  let event_A := red_balls
  let event_B_given_A := white_balls

  (event_B_given_A * (total_balls - 1)) / (event_A * total_balls) = 3 / 7 :=
by
  sorry

end NUMINAMATH_GPT_probability_white_second_given_red_first_l1226_122632


namespace NUMINAMATH_GPT_prove_f_neg_2_l1226_122633

noncomputable def f (a b x : ℝ) := a * x^4 + b * x^2 - x + 1

-- Main theorem statement
theorem prove_f_neg_2 (a b : ℝ) (h : f a b 2 = 9) : f a b (-2) = 13 := 
by
  sorry

end NUMINAMATH_GPT_prove_f_neg_2_l1226_122633


namespace NUMINAMATH_GPT_tractor_efficiency_l1226_122674

theorem tractor_efficiency (x y : ℝ) (h1 : 18 / x = 24 / y) (h2 : x + y = 7) :
  x = 3 ∧ y = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_tractor_efficiency_l1226_122674


namespace NUMINAMATH_GPT_calculate_M_minus_m_l1226_122615

def total_students : ℕ := 2001
def students_studying_spanish (S : ℕ) : Prop := 1601 ≤ S ∧ S ≤ 1700
def students_studying_french (F : ℕ) : Prop := 601 ≤ F ∧ F ≤ 800
def studying_both_languages_lower_bound (S F m : ℕ) : Prop := S + F - m = total_students
def studying_both_languages_upper_bound (S F M : ℕ) : Prop := S + F - M = total_students

theorem calculate_M_minus_m :
  ∀ (S F m M : ℕ),
    students_studying_spanish S →
    students_studying_french F →
    studying_both_languages_lower_bound S F m →
    studying_both_languages_upper_bound S F M →
    S = 1601 ∨ S = 1700 →
    F = 601 ∨ F = 800 →
    M - m = 298 :=
by
  intros S F m M hs hf hl hb Hs Hf
  sorry

end NUMINAMATH_GPT_calculate_M_minus_m_l1226_122615


namespace NUMINAMATH_GPT_p_hyperbola_implies_m_range_p_necessary_not_sufficient_for_q_l1226_122671

def p (m : ℝ) (x y : ℝ) : Prop := (x^2) / (m - 1) + (y^2) / (m - 4) = 1
def q (m : ℝ) (x y : ℝ) : Prop := (x^2) / (m - 2) + (y^2) / (4 - m) = 1

theorem p_hyperbola_implies_m_range (m : ℝ) (x y : ℝ) :
  p m x y → 1 < m ∧ m < 4 :=
sorry

theorem p_necessary_not_sufficient_for_q (m : ℝ) (x y : ℝ) :
  (1 < m ∧ m < 4) ∧ p m x y →
  (q m x y → (2 < m ∧ m < 3) ∨ (3 < m ∧ m < 4)) :=
sorry

end NUMINAMATH_GPT_p_hyperbola_implies_m_range_p_necessary_not_sufficient_for_q_l1226_122671


namespace NUMINAMATH_GPT_distance_A_to_B_l1226_122623

theorem distance_A_to_B (D_B D_C V_E V_F : ℝ) (h1 : D_B / 3 = V_E)
  (h2 : D_C / 4 = V_F) (h3 : V_E / V_F = 2.533333333333333)
  (h4 : D_B = 300 ∨ D_C = 300) : D_B = 570 :=
by
  -- Proof yet to be provided
  sorry

end NUMINAMATH_GPT_distance_A_to_B_l1226_122623


namespace NUMINAMATH_GPT_find_a2_b2_l1226_122608

theorem find_a2_b2 (a b : ℝ) (h1 : a - b = 6) (h2 : a * b = 32) : a^2 + b^2 = 100 :=
by
  sorry

end NUMINAMATH_GPT_find_a2_b2_l1226_122608


namespace NUMINAMATH_GPT_problem_solution_l1226_122695

theorem problem_solution :
  (204^2 - 196^2) / 16 = 200 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1226_122695


namespace NUMINAMATH_GPT_percentage_defective_meters_l1226_122629

theorem percentage_defective_meters (total_meters : ℕ) (defective_meters : ℕ) (h1 : total_meters = 150) (h2 : defective_meters = 15) : 
  (defective_meters : ℚ) / (total_meters : ℚ) * 100 = 10 := by
sorry

end NUMINAMATH_GPT_percentage_defective_meters_l1226_122629


namespace NUMINAMATH_GPT_find_b_in_cubic_function_l1226_122654

noncomputable def cubic_function (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

theorem find_b_in_cubic_function (a b c d : ℝ) (h1: cubic_function a b c d 2 = 0)
  (h2: cubic_function a b c d (-1) = 0) (h3: cubic_function a b c d 1 = 4) :
  b = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_b_in_cubic_function_l1226_122654


namespace NUMINAMATH_GPT_remainder_7n_mod_4_l1226_122667

theorem remainder_7n_mod_4 (n : ℕ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := sorry

end NUMINAMATH_GPT_remainder_7n_mod_4_l1226_122667


namespace NUMINAMATH_GPT_sum_after_third_rotation_max_sum_of_six_faces_l1226_122655

variable (a b c : ℕ) (a' b': ℕ)

-- Initial Conditions
axiom sum_initial : a + b + c = 42

-- Conditions after first rotation
axiom a_prime : a' = a - 8
axiom sum_first_rotation : b + c + a' = 34

-- Conditions after second rotation
axiom b_prime : b' = b + 19
axiom sum_second_rotation : c + a' + b' = 53

-- The cube always rests on the face with number 6
axiom bottom_face : c = 6

-- Prove question 1:
theorem sum_after_third_rotation : (b + 19) + a + c = 61 :=
by sorry

-- Prove question 2:
theorem max_sum_of_six_faces : 
∃ d e f: ℕ, d = a ∧ e = b ∧ f = c ∧ d + e + f + (a - 8) + (b + 19) + 6 = 100 :=
by sorry

end NUMINAMATH_GPT_sum_after_third_rotation_max_sum_of_six_faces_l1226_122655


namespace NUMINAMATH_GPT_arithmetic_sequence_a9_l1226_122640

noncomputable def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a9
  (a : ℕ → ℤ)
  (h_seq : arithmetic_sequence a 1)
  (h2 : a 2 + a 4 = 2)
  (h5 : a 5 = 3) :
  a 9 = 7 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a9_l1226_122640


namespace NUMINAMATH_GPT_books_selection_l1226_122646

theorem books_selection 
  (num_mystery : ℕ)
  (num_fantasy : ℕ)
  (num_biographies : ℕ)
  (Hmystery : num_mystery = 5)
  (Hfantasy : num_fantasy = 4)
  (Hbiographies : num_biographies = 6) :
  (num_mystery * num_fantasy * num_biographies = 120) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_books_selection_l1226_122646


namespace NUMINAMATH_GPT_problem_solution_l1226_122682

noncomputable def solve_problem : Prop :=
  ∃ (d : ℝ), 
    (∃ int_part : ℤ, 
        (3 * int_part^2 - 12 * int_part + 9 = 0 ∧ ⌊d⌋ = int_part) ∧
        ∀ frac_part : ℝ,
            (4 * frac_part^3 - 8 * frac_part^2 + 3 * frac_part - 0.5 = 0 ∧ frac_part = d - ⌊d⌋) )
    ∧ (d = 1.375 ∨ d = 3.375)

theorem problem_solution : solve_problem :=
by sorry

end NUMINAMATH_GPT_problem_solution_l1226_122682


namespace NUMINAMATH_GPT_range_k_l1226_122618

noncomputable def point (α : Type*) := (α × α)

def M : point ℝ := (0, 2)
def N : point ℝ := (-2, 0)

def line (k : ℝ) (P : point ℝ) := k * P.1 - P.2 - 2 * k + 2 = 0
def angle_condition (M N P : point ℝ) := true -- placeholder for the condition that ∠MPN ≥ π/2

theorem range_k (k : ℝ) (P : point ℝ)
  (hP_on_line : line k P)
  (h_angle_cond : angle_condition M N P) :
  (1 / 7 : ℝ) ≤ k ∧ k ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_k_l1226_122618


namespace NUMINAMATH_GPT_triangle_area_condition_l1226_122697

theorem triangle_area_condition (m : ℝ) 
  (H_line : ∀ (x y : ℝ), x - m*y + 1 = 0)
  (H_circle : ∀ (x y : ℝ), (x - 1)^2 + y^2 = 4)
  (H_area : ∃ (A B C : (ℝ × ℝ)), (x - my + 1 = 0) ∧ (∃ C : (ℝ × ℝ), (x1 - 1)^2 + y1^2 = 4 ∨ (x2 - 1)^2 + y2^2 = 4))
  : m = 2 :=
sorry

end NUMINAMATH_GPT_triangle_area_condition_l1226_122697


namespace NUMINAMATH_GPT_hexagon_angle_in_arithmetic_progression_l1226_122603

theorem hexagon_angle_in_arithmetic_progression :
  ∃ (a d : ℝ), (a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) + (a + 5 * d) = 720) ∧ 
  (a = 120 ∨ a + d = 120 ∨ a + 2 * d = 120 ∨ a + 3 * d = 120 ∨ a + 4 * d = 120 ∨ a + 5 * d = 120) := by
  sorry

end NUMINAMATH_GPT_hexagon_angle_in_arithmetic_progression_l1226_122603


namespace NUMINAMATH_GPT_ratio_of_increase_to_original_l1226_122601

noncomputable def ratio_increase_avg_marks (T : ℝ) : ℝ :=
  let original_avg := T / 40
  let new_total := T + 20
  let new_avg := new_total / 40
  let increase_avg := new_avg - original_avg
  increase_avg / original_avg

theorem ratio_of_increase_to_original (T : ℝ) (hT : T > 0) :
  ratio_increase_avg_marks T = 20 / T :=
by
  unfold ratio_increase_avg_marks
  sorry

end NUMINAMATH_GPT_ratio_of_increase_to_original_l1226_122601


namespace NUMINAMATH_GPT_not_true_expr_l1226_122686

theorem not_true_expr (x y : ℝ) (h : x < y) : -2 * x > -2 * y :=
sorry

end NUMINAMATH_GPT_not_true_expr_l1226_122686


namespace NUMINAMATH_GPT_divisibility_of_n_l1226_122687

theorem divisibility_of_n (P : Polynomial ℤ) (k n : ℕ)
  (hk : k % 2 = 0)
  (h_odd_coeffs : ∀ i, i ≤ k → i % 2 = 1)
  (h_div : ∃ Q : Polynomial ℤ, (X + 1)^n - 1 = (P * Q)) :
  n % (k + 1) = 0 :=
sorry

end NUMINAMATH_GPT_divisibility_of_n_l1226_122687


namespace NUMINAMATH_GPT_rational_roots_polynomial1_rational_roots_polynomial2_rational_roots_polynomial3_l1226_122668

variable (x : ℚ)

-- Polynomial 1
def polynomial1 := x^4 - 3*x^3 - 8*x^2 + 12*x + 16

theorem rational_roots_polynomial1 :
  (polynomial1 (-1) = 0) ∧
  (polynomial1 2 = 0) ∧
  (polynomial1 (-2) = 0) ∧
  (polynomial1 4 = 0) :=
sorry

-- Polynomial 2
def polynomial2 := 8*x^3 - 20*x^2 - 2*x + 5

theorem rational_roots_polynomial2 :
  (polynomial2 (1/2) = 0) ∧
  (polynomial2 (-1/2) = 0) ∧
  (polynomial2 (5/2) = 0) :=
sorry

-- Polynomial 3
def polynomial3 := 4*x^4 - 16*x^3 + 11*x^2 + 4*x - 3

theorem rational_roots_polynomial3 :
  (polynomial3 (-1/2) = 0) ∧
  (polynomial3 (1/2) = 0) ∧
  (polynomial3 1 = 0) ∧
  (polynomial3 3 = 0) :=
sorry

end NUMINAMATH_GPT_rational_roots_polynomial1_rational_roots_polynomial2_rational_roots_polynomial3_l1226_122668


namespace NUMINAMATH_GPT_find_certain_number_l1226_122661

theorem find_certain_number (x : ℤ) (h : ((x / 4) + 25) * 3 = 150) : x = 100 :=
by
  sorry

end NUMINAMATH_GPT_find_certain_number_l1226_122661


namespace NUMINAMATH_GPT_main_theorem_l1226_122692

-- Define the sets M and N
def M : Set ℝ := { x | 0 < x ∧ x < 10 }
def N : Set ℝ := { x | x < -4/3 ∨ x > 3 }

-- Define the complement of N in ℝ
def comp_N : Set ℝ := { x | ¬ (x < -4/3 ∨ x > 3) }

-- The main theorem to be proved
theorem main_theorem : M ∩ comp_N = { x | 0 < x ∧ x ≤ 3 } := 
by
  sorry

end NUMINAMATH_GPT_main_theorem_l1226_122692


namespace NUMINAMATH_GPT_find_c_l1226_122681

noncomputable def condition1 (a b c : ℝ) : Prop :=
  a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)

noncomputable def condition2 (c : ℝ) : Prop :=
  6 * 15 * c = 1

theorem find_c (c : ℝ) (h1 : condition1 6 15 c) (h2 : condition2 c) : c = 11 := 
by
  sorry

end NUMINAMATH_GPT_find_c_l1226_122681


namespace NUMINAMATH_GPT_polygon_with_given_angle_sum_l1226_122641

-- Definition of the sum of interior angles of a polygon
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- Definition of the sum of exterior angles of a polygon
def sum_exterior_angles : ℝ := 360

-- Given condition: the sum of the interior angles is four times the sum of the exterior angles
def sum_condition (n : ℕ) : Prop :=
  sum_interior_angles n = 4 * sum_exterior_angles

-- The main theorem we want to prove
theorem polygon_with_given_angle_sum : 
  ∃ n : ℕ, sum_condition n ∧ n = 10 :=
by
  sorry

end NUMINAMATH_GPT_polygon_with_given_angle_sum_l1226_122641


namespace NUMINAMATH_GPT_tan_105_eq_neg2_sub_sqrt3_l1226_122630

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 :=
by 
  sorry

end NUMINAMATH_GPT_tan_105_eq_neg2_sub_sqrt3_l1226_122630


namespace NUMINAMATH_GPT_compound_interest_rate_l1226_122689

theorem compound_interest_rate
  (P A : ℝ) (n t : ℕ) (r : ℝ)
  (hP : P = 10000)
  (hA : A = 12155.06)
  (hn : n = 4)
  (ht : t = 1)
  (h_eq : A = P * (1 + r / n) ^ (n * t)):
  r = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_rate_l1226_122689


namespace NUMINAMATH_GPT_sum_of_consecutive_even_numbers_l1226_122648

theorem sum_of_consecutive_even_numbers (x : ℤ) (h : (x + 2)^2 - x^2 = 84) : x + (x + 2) = 42 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_even_numbers_l1226_122648


namespace NUMINAMATH_GPT_books_read_l1226_122613

theorem books_read (pages_per_hour : ℕ) (pages_per_book : ℕ) (hours_available : ℕ) 
(h : pages_per_hour = 120) (b : pages_per_book = 360) (t : hours_available = 8) :
  hours_available * pages_per_hour ≥ 2 * pages_per_book :=
by
  rw [h, b, t]
  sorry

end NUMINAMATH_GPT_books_read_l1226_122613


namespace NUMINAMATH_GPT_negate_proposition_l1226_122658

theorem negate_proposition :
  (¬ (∀ x : ℝ, x > 1 → x^2 + x + 1 > 0)) ↔ (∃ x : ℝ, x > 1 ∧ x^2 + x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_GPT_negate_proposition_l1226_122658


namespace NUMINAMATH_GPT_john_computers_fixed_count_l1226_122653

-- Define the problem conditions.
variables (C : ℕ)
variables (unfixable_ratio spare_part_ratio fixable_ratio : ℝ)
variables (fixed_right_away : ℕ)
variables (h1 : unfixable_ratio = 0.20)
variables (h2 : spare_part_ratio = 0.40)
variables (h3 : fixable_ratio = 0.40)
variables (h4 : fixed_right_away = 8)
variables (h5 : fixable_ratio * ↑C = fixed_right_away)

-- The theorem to prove.
theorem john_computers_fixed_count (h1 : C > 0) : C = 20 := by
  sorry

end NUMINAMATH_GPT_john_computers_fixed_count_l1226_122653


namespace NUMINAMATH_GPT_cleaning_time_l1226_122679

noncomputable def combined_cleaning_time (sawyer_time nick_time sarah_time : ℕ) : ℚ :=
  let rate_sawyer := 1 / sawyer_time
  let rate_nick := 1 / nick_time
  let rate_sarah := 1 / sarah_time
  1 / (rate_sawyer + rate_nick + rate_sarah)

theorem cleaning_time : combined_cleaning_time 6 9 4 = 36 / 19 := by
  have h1 : 1 / 6 = 1 / 6 := rfl
  have h2 : 1 / 9 = 1 / 9 := rfl
  have h3 : 1 / 4 = 1 / 4 := rfl
  rw [combined_cleaning_time, h1, h2, h3]
  norm_num
  sorry

end NUMINAMATH_GPT_cleaning_time_l1226_122679


namespace NUMINAMATH_GPT_max_not_divisible_by_3_l1226_122614

theorem max_not_divisible_by_3 (s : Finset ℕ) (h₁ : s.card = 7) (h₂ : ∃ p ∈ s, p % 3 = 0) : 
  ∃t : Finset ℕ, t.card = 6 ∧ (∀ x ∈ t, x % 3 ≠ 0) ∧ (t ⊆ s) :=
sorry

end NUMINAMATH_GPT_max_not_divisible_by_3_l1226_122614


namespace NUMINAMATH_GPT_conditions_not_sufficient_nor_necessary_l1226_122652

theorem conditions_not_sufficient_nor_necessary (a : ℝ) (b : ℝ) :
  (a ≠ 5) ∧ (b ≠ -5) ↔ ¬((a ≠ 5) ∨ (b ≠ -5)) ∧ (a + b ≠ 0) := 
sorry

end NUMINAMATH_GPT_conditions_not_sufficient_nor_necessary_l1226_122652


namespace NUMINAMATH_GPT_origin_in_ellipse_l1226_122696

theorem origin_in_ellipse (k : ℝ):
  (∃ x y : ℝ, k^2 * x^2 + y^2 - 4 * k * x + 2 * k * y + k^2 - 1 = 0 ∧ x = 0 ∧ y = 0) →
  0 < abs k ∧ abs k < 1 :=
by
  -- Note: Proof omitted.
  sorry

end NUMINAMATH_GPT_origin_in_ellipse_l1226_122696


namespace NUMINAMATH_GPT_expected_value_max_l1226_122684

def E_max_x_y_z (x y z : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 10) : ℚ :=
  (4 * (1/6) + 5 * (1/3) + 6 * (1/4) + 7 * (1/6) + 8 * (1/12))

theorem expected_value_max (x y z : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 10) :
  E_max_x_y_z x y z h1 h2 h3 h4 = 17 / 3 := 
sorry

end NUMINAMATH_GPT_expected_value_max_l1226_122684


namespace NUMINAMATH_GPT_senior_high_sample_count_l1226_122620

theorem senior_high_sample_count 
  (total_students : ℕ)
  (junior_high_students : ℕ)
  (senior_high_students : ℕ)
  (total_sampled_students : ℕ)
  (H1 : total_students = 1800)
  (H2 : junior_high_students = 1200)
  (H3 : senior_high_students = 600)
  (H4 : total_sampled_students = 180) :
  (senior_high_students * total_sampled_students / total_students) = 60 := 
sorry

end NUMINAMATH_GPT_senior_high_sample_count_l1226_122620


namespace NUMINAMATH_GPT_domain_log_function_l1226_122609

theorem domain_log_function :
  {x : ℝ | 1 < x ∧ x < 3 ∧ x ≠ 2} = {x : ℝ | (3 - x > 0) ∧ (x - 1 > 0) ∧ (x - 1 ≠ 1)} :=
sorry

end NUMINAMATH_GPT_domain_log_function_l1226_122609


namespace NUMINAMATH_GPT_triangle_incircle_ratio_l1226_122694

theorem triangle_incircle_ratio (r p k : ℝ) (h1 : k = r * (p / 2)) : 
  p / k = 2 / r :=
by
  sorry

end NUMINAMATH_GPT_triangle_incircle_ratio_l1226_122694


namespace NUMINAMATH_GPT_angle_DGO_is_50_degrees_l1226_122690

theorem angle_DGO_is_50_degrees
  (triangle_DOG : Type)
  (D G O : triangle_DOG)
  (angle_DOG : ℝ)
  (angle_DGO : ℝ)
  (angle_OGD : ℝ)
  (bisect : Prop) :

  angle_DGO = 50 := 
by
  -- Conditions
  have h1 : angle_DGO = angle_DOG := sorry
  have h2 : angle_DOG = 40 := sorry
  have h3 : bisect := sorry
  -- Goal
  sorry

end NUMINAMATH_GPT_angle_DGO_is_50_degrees_l1226_122690


namespace NUMINAMATH_GPT_fraction_of_rotten_fruits_l1226_122660

theorem fraction_of_rotten_fruits (a p : ℕ) (rotten_apples_eq_rotten_pears : (2 / 3) * a = (3 / 4) * p)
    (rotten_apples_fraction : 2 / 3 = 2 / 3)
    (rotten_pears_fraction : 3 / 4 = 3 / 4) :
    (4 * a) / (3 * (a + (4 / 3) * (2 * a) / 3)) = 12 / 17 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_rotten_fruits_l1226_122660


namespace NUMINAMATH_GPT_swimming_club_total_members_l1226_122624

def valid_total_members (total : ℕ) : Prop :=
  ∃ (J S V : ℕ),
    3 * S = 2 * J ∧
    5 * V = 2 * S ∧
    total = J + S + V

theorem swimming_club_total_members :
  valid_total_members 58 := by
  sorry

end NUMINAMATH_GPT_swimming_club_total_members_l1226_122624


namespace NUMINAMATH_GPT_bob_pennies_l1226_122650

theorem bob_pennies (a b : ℕ) 
  (h1 : b + 1 = 4 * (a - 1)) 
  (h2 : b - 1 = 3 * (a + 1)) : 
  b = 31 :=
by
  sorry

end NUMINAMATH_GPT_bob_pennies_l1226_122650


namespace NUMINAMATH_GPT_find_perimeter_and_sin2A_of_triangle_l1226_122621

theorem find_perimeter_and_sin2A_of_triangle (a b c : ℝ) (A B C : ℝ) 
  (h_a : a = 3) (h_B : B = Real.pi / 3) (h_area : 6 * Real.sqrt 3 = 6 * Real.sqrt 3)
  (h_S : S_ABC = 6 * Real.sqrt 3) : 
  (a + b + c = 18) ∧ (Real.sin (2 * A) = (39 * Real.sqrt 3) / 98) := 
by 
  -- The proof will be placed here. Assuming a valid proof exists.
  sorry

end NUMINAMATH_GPT_find_perimeter_and_sin2A_of_triangle_l1226_122621


namespace NUMINAMATH_GPT_garden_perimeter_l1226_122698

noncomputable def find_perimeter (l w : ℕ) : ℕ := 2 * l + 2 * w

theorem garden_perimeter :
  ∀ (l w : ℕ),
  (l = 3 * w + 2) →
  (l = 38) →
  find_perimeter l w = 100 :=
by
  intros l w H1 H2
  sorry

end NUMINAMATH_GPT_garden_perimeter_l1226_122698


namespace NUMINAMATH_GPT_second_factor_of_lcm_l1226_122602

theorem second_factor_of_lcm (A B : ℕ) (hcf : ℕ) (factor1 : ℕ) (factor2 : ℕ) (lcm : ℕ) 
  (h1 : hcf = 20) 
  (h2 : A = 280)
  (h3 : factor1 = 13) 
  (h4 : lcm = hcf * factor1 * factor2) 
  (h5 : A = hcf * 14) : 
  factor2 = 14 :=
by 
  sorry

end NUMINAMATH_GPT_second_factor_of_lcm_l1226_122602


namespace NUMINAMATH_GPT_vector_subtraction_l1226_122663

-- Lean definitions for the problem conditions
def v₁ : ℝ × ℝ := (3, -5)
def v₂ : ℝ × ℝ := (-2, 6)
def s₁ : ℝ := 4
def s₂ : ℝ := 3

-- The theorem statement
theorem vector_subtraction :
  s₁ • v₁ - s₂ • v₂ = (18, -38) :=
by
  sorry

end NUMINAMATH_GPT_vector_subtraction_l1226_122663


namespace NUMINAMATH_GPT_german_russian_students_l1226_122645

open Nat

theorem german_russian_students (G R : ℕ) (G_cap_R : ℕ) 
  (h_total : 1500 = G + R - G_cap_R)
  (hG_lb : 1125 ≤ G) (hG_ub : G ≤ 1275)
  (hR_lb : 375 ≤ R) (hR_ub : R ≤ 525) :
  300 = (max (G_cap_R) - min (G_cap_R)) :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_german_russian_students_l1226_122645


namespace NUMINAMATH_GPT_perfect_squares_digits_l1226_122662

theorem perfect_squares_digits 
  (a b : ℕ) 
  (ha : ∃ m : ℕ, a = m * m) 
  (hb : ∃ n : ℕ, b = n * n) 
  (a_units_digit_1 : a % 10 = 1) 
  (b_units_digit_6 : b % 10 = 6) 
  (a_tens_digit : ∃ x : ℕ, (a / 10) % 10 = x) 
  (b_tens_digit : ∃ y : ℕ, (b / 10) % 10 = y) : 
  ∃ x y : ℕ, (x % 2 = 0) ∧ (y % 2 = 1) := 
sorry

end NUMINAMATH_GPT_perfect_squares_digits_l1226_122662


namespace NUMINAMATH_GPT_largest_of_seven_consecutive_integers_l1226_122688

theorem largest_of_seven_consecutive_integers (n : ℕ) (h : n > 0) (h_sum : n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) = 2222) : (n + 6) = 320 :=
by sorry

end NUMINAMATH_GPT_largest_of_seven_consecutive_integers_l1226_122688


namespace NUMINAMATH_GPT_who_is_wrong_l1226_122675

theorem who_is_wrong 
  (a1 a2 a3 a4 a5 a6 : ℤ)
  (h1 : a1 + a3 + a5 = a2 + a4 + a6 + 3)
  (h2 : a2 + a4 + a6 = a1 + a3 + a5 + 5) : 
  False := 
sorry

end NUMINAMATH_GPT_who_is_wrong_l1226_122675


namespace NUMINAMATH_GPT_range_a_le_2_l1226_122622
-- Import everything from Mathlib

-- Define the hypothesis and the conclusion in Lean 4
theorem range_a_le_2 (a : ℝ) : 
  (∀ x > 0, Real.log x + a * x + 1 - x * Real.exp (2 * x) ≤ 0) ↔ a ≤ 2 := 
sorry

end NUMINAMATH_GPT_range_a_le_2_l1226_122622


namespace NUMINAMATH_GPT_union_complement_set_l1226_122600

theorem union_complement_set (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4, 5}) 
  (hA : A = {1, 2, 3, 5}) (hB : B = {2, 4}) :
  (U \ A) ∪ B = {0, 2, 4} :=
by
  rw [Set.diff_eq, hU, hA, hB]
  simp
  sorry

end NUMINAMATH_GPT_union_complement_set_l1226_122600


namespace NUMINAMATH_GPT_percentage_sophia_ate_l1226_122639

theorem percentage_sophia_ate : 
  ∀ (caden zoe noah sophia : ℝ),
    caden = 20 / 100 →
    zoe = caden + (0.5 * caden) →
    noah = zoe + (0.5 * zoe) →
    caden + zoe + noah + sophia = 1 →
    sophia = 5 / 100 :=
by
  intros
  sorry

end NUMINAMATH_GPT_percentage_sophia_ate_l1226_122639


namespace NUMINAMATH_GPT_multiplication_of_powers_same_base_l1226_122651

theorem multiplication_of_powers_same_base (x : ℝ) : x^3 * x^2 = x^5 :=
by
-- proof steps go here
sorry

end NUMINAMATH_GPT_multiplication_of_powers_same_base_l1226_122651


namespace NUMINAMATH_GPT_line_equation_l1226_122699

theorem line_equation (P : ℝ × ℝ) (hP : P = (-2, 1)) :
  ∃ a b c : ℝ, a * P.1 + b * P.2 + c = 0 ∧ a = 2 ∧ b = -1 ∧ c = -5 := by
  sorry

end NUMINAMATH_GPT_line_equation_l1226_122699


namespace NUMINAMATH_GPT_fastest_slowest_difference_l1226_122665

-- Given conditions
def length_A : ℕ := 8
def length_B : ℕ := 10
def length_C : ℕ := 6
def section_length : ℕ := 2

def sections_A : ℕ := 24
def sections_B : ℕ := 25
def sections_C : ℕ := 27

-- Calculate number of cuts required
def cuts_per_segment_A := length_A / section_length - 1
def cuts_per_segment_B := length_B / section_length - 1
def cuts_per_segment_C := length_C / section_length - 1

-- Calculate total number of cuts
def total_cuts_A := cuts_per_segment_A * (sections_A / (length_A / section_length))
def total_cuts_B := cuts_per_segment_B * (sections_B / (length_B / section_length))
def total_cuts_C := cuts_per_segment_C * (sections_C / (length_C / section_length))

-- Finding min and max cuts
def max_cuts := max total_cuts_A (max total_cuts_B total_cuts_C)
def min_cuts := min total_cuts_A (min total_cuts_B total_cuts_C)

-- Prove that the difference between max cuts and min cuts is 2
theorem fastest_slowest_difference :
  max_cuts - min_cuts = 2 := by
  sorry

end NUMINAMATH_GPT_fastest_slowest_difference_l1226_122665


namespace NUMINAMATH_GPT_roots_reciprocal_sum_eq_25_l1226_122611

theorem roots_reciprocal_sum_eq_25 (p q r : ℝ) (hpq : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0) (hroot : ∀ x, x^3 - 9*x^2 + 8*x + 2 = 0 → (x = p ∨ x = q ∨ x = r)) :
  1/p^2 + 1/q^2 + 1/r^2 = 25 :=
by sorry

end NUMINAMATH_GPT_roots_reciprocal_sum_eq_25_l1226_122611


namespace NUMINAMATH_GPT_successive_percentage_reduction_l1226_122637

theorem successive_percentage_reduction (a b : ℝ) (h₁ : a = 25) (h₂ : b = 20) :
  a + b - (a * b) / 100 = 40 := by
  sorry

end NUMINAMATH_GPT_successive_percentage_reduction_l1226_122637


namespace NUMINAMATH_GPT_rocket_soaring_time_l1226_122619

theorem rocket_soaring_time 
  (avg_speed : ℝ)                      -- The average speed of the rocket
  (soar_speed : ℝ)                     -- Speed while soaring
  (plummet_distance : ℝ)               -- Distance covered during plummet
  (plummet_time : ℝ)                   -- Time of plummet
  (total_time : ℝ := plummet_time + t) -- Total time is the sum of soaring time and plummet time
  (total_distance : ℝ := soar_speed * t + plummet_distance) -- Total distance covered
  (h_avg_speed : avg_speed = total_distance / total_time)   -- Given condition for average speed
  :
  ∃ t : ℝ, t = 12 :=                   -- Prove that the soaring time is 12 seconds
by
  sorry

end NUMINAMATH_GPT_rocket_soaring_time_l1226_122619


namespace NUMINAMATH_GPT_average_playtime_in_minutes_l1226_122612

noncomputable def lena_playtime_hours : ℝ := 3.5
noncomputable def lena_playtime_minutes : ℝ := lena_playtime_hours * 60
noncomputable def brother_playtime_minutes : ℝ := 1.2 * lena_playtime_minutes + 17
noncomputable def sister_playtime_minutes : ℝ := 1.5 * brother_playtime_minutes

theorem average_playtime_in_minutes :
  (lena_playtime_minutes + brother_playtime_minutes + sister_playtime_minutes) / 3 = 294.17 :=
by
  sorry

end NUMINAMATH_GPT_average_playtime_in_minutes_l1226_122612


namespace NUMINAMATH_GPT_number_of_solid_shapes_is_three_l1226_122691

-- Define the geometric shapes and their dimensionality
inductive GeomShape
| square : GeomShape
| cuboid : GeomShape
| circle : GeomShape
| sphere : GeomShape
| cone : GeomShape

def isSolid (shape : GeomShape) : Bool :=
  match shape with
  | GeomShape.square => false
  | GeomShape.cuboid => true
  | GeomShape.circle => false
  | GeomShape.sphere => true
  | GeomShape.cone => true

-- Formal statement of the problem
theorem number_of_solid_shapes_is_three :
  (List.filter isSolid [GeomShape.square, GeomShape.cuboid, GeomShape.circle, GeomShape.sphere, GeomShape.cone]).length = 3 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_number_of_solid_shapes_is_three_l1226_122691


namespace NUMINAMATH_GPT_sqrt_expression_l1226_122634

theorem sqrt_expression (x : ℝ) (h : x < 0) : 
  Real.sqrt (x^2 / (1 + (x + 1) / x)) = Real.sqrt (x^3 / (2 * x + 1)) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_l1226_122634


namespace NUMINAMATH_GPT_shape_at_22_l1226_122607

-- Define the pattern
def pattern : List String := ["triangle", "square", "diamond", "diamond", "circle"]

-- Function to get the nth shape in the repeated pattern sequence
def getShape (n : Nat) : String :=
  pattern.get! (n % pattern.length)

-- Statement to prove
theorem shape_at_22 : getShape 21 = "square" :=
by
  sorry

end NUMINAMATH_GPT_shape_at_22_l1226_122607


namespace NUMINAMATH_GPT_percent_of_number_l1226_122649

theorem percent_of_number (x : ℝ) (h : 18 = 0.75 * x) : x = 24 := by
  sorry

end NUMINAMATH_GPT_percent_of_number_l1226_122649


namespace NUMINAMATH_GPT_profit_ratio_l1226_122604

-- Definitions based on conditions
-- Let A_orig and B_orig represent the original profits of stores A and B
-- after increase and decrease respectively, they become equal

variable (A_orig B_orig : ℝ)
variable (h1 : (1.2 * A_orig) = (0.9 * B_orig))

-- Prove that the original profit of store A was 75% of the profit of store B
theorem profit_ratio (h1 : 1.2 * A_orig = 0.9 * B_orig) : A_orig = 0.75 * B_orig :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_profit_ratio_l1226_122604


namespace NUMINAMATH_GPT_right_drawing_num_triangles_l1226_122605

-- Given the conditions:
-- 1. Nine distinct lines in the right drawing
-- 2. Any combination of 3 lines out of these 9 forms a triangle
-- 3. Count of intersections of these lines where exactly three lines intersect

def num_triangles : Nat := 84 -- Calculated via binomial coefficient
def num_intersections : Nat := 61 -- Given or calculated from the problem

-- The target theorem to prove that the number of triangles is equal to 23
theorem right_drawing_num_triangles :
  num_triangles - num_intersections = 23 :=
by
  -- Proof would go here, but we skip it as per the instructions
  sorry

end NUMINAMATH_GPT_right_drawing_num_triangles_l1226_122605


namespace NUMINAMATH_GPT_find_third_number_l1226_122625

-- Define the given conditions
def proportion_condition (x y : ℝ) : Prop :=
  (0.75 / x) = (y / 8)

-- The main statement to be proven
theorem find_third_number (x y : ℝ) (hx : x = 1.2) (h_proportion : proportion_condition x y) : y = 5 :=
by
  -- Using the assumptions and the definition provided.
  sorry

end NUMINAMATH_GPT_find_third_number_l1226_122625


namespace NUMINAMATH_GPT_find_y_of_arithmetic_mean_l1226_122627

theorem find_y_of_arithmetic_mean (y : ℝ) (h : (8 + 16 + 12 + 24 + 7 + y) / 6 = 12) : y = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_y_of_arithmetic_mean_l1226_122627


namespace NUMINAMATH_GPT_length_BC_fraction_AD_l1226_122666

-- Given
variables {A B C D : Type*} [AddCommGroup D] [Module ℝ D]
variables (A B C D : D)
variables (AB BD AC CD AD BC : ℝ)

-- Conditions
def segment_AD := A + D
def segment_BD := B + D
def segment_AB := A + B
def segment_CD := C + D
def segment_AC := A + C
def relation_AB_BD : AB = 3 * BD := sorry
def relation_AC_CD : AC = 5 * CD := sorry

-- Proof
theorem length_BC_fraction_AD :
  BC = (1/12) * AD :=
sorry

end NUMINAMATH_GPT_length_BC_fraction_AD_l1226_122666


namespace NUMINAMATH_GPT_arithmetic_result_l1226_122635

theorem arithmetic_result :
  (3 * 13) + (3 * 14) + (3 * 17) + 11 = 143 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_result_l1226_122635


namespace NUMINAMATH_GPT_solution_set_of_floor_eqn_l1226_122647

theorem solution_set_of_floor_eqn:
  ∀ x y : ℝ, 
  (⌊x⌋ * ⌊x⌋ + ⌊y⌋ * ⌊y⌋ = 4) ↔ 
  ((2 ≤ x ∧ x < 3 ∧ 0 ≤ y ∧ y < 1) ∨
   (0 ≤ x ∧ x < 1 ∧ 2 ≤ y ∧ y < 3) ∨
   (-2 ≤ x ∧ x < -1 ∧ 0 ≤ y ∧ y < 1) ∨
   (0 ≤ x ∧ x < 1 ∧ -2 ≤ y ∧ y < -1)) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_floor_eqn_l1226_122647


namespace NUMINAMATH_GPT_determinant_matrix_equivalence_l1226_122606

variable {R : Type} [CommRing R]

theorem determinant_matrix_equivalence
  (x y z w : R)
  (h : x * w - y * z = 3) :
  (x * (5 * z + 4 * w) - z * (5 * x + 4 * y) = 12) :=
by sorry

end NUMINAMATH_GPT_determinant_matrix_equivalence_l1226_122606


namespace NUMINAMATH_GPT_twelfth_equation_l1226_122642

theorem twelfth_equation : (14 : ℤ)^2 - (12 : ℤ)^2 = 4 * 13 := by
  sorry

end NUMINAMATH_GPT_twelfth_equation_l1226_122642
