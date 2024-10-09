import Mathlib

namespace q_work_alone_in_10_days_l2341_234152

theorem q_work_alone_in_10_days (p_rate : ℝ) (q_rate : ℝ) (d : ℕ) (h1 : p_rate = 1 / 20)
                                    (h2 : q_rate = 1 / d) (h3 : 2 * (p_rate + q_rate) = 0.3) :
                                    d = 10 :=
by sorry

end q_work_alone_in_10_days_l2341_234152


namespace solve_for_y_l2341_234165

variable {y : ℚ}
def algebraic_expression_1 (y : ℚ) : ℚ := 4 * y + 8
def algebraic_expression_2 (y : ℚ) : ℚ := 8 * y - 7

theorem solve_for_y (h : algebraic_expression_1 y = - algebraic_expression_2 y) : y = -1 / 12 :=
by
  sorry

end solve_for_y_l2341_234165


namespace greatest_perimeter_of_triangle_l2341_234171

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), (3 * x) + 15 = 57 ∧ 
  (x > 5 ∧ x < 15) ∧ 
  2 * x + x > 15 ∧ 
  x + 15 > 2 * x ∧ 
  2 * x + 15 > x := 
sorry

end greatest_perimeter_of_triangle_l2341_234171


namespace area_of_parallelogram_l2341_234115

theorem area_of_parallelogram (base height : ℝ) (h_base : base = 18) (h_height : height = 16) : 
  base * height = 288 := 
by
  sorry

end area_of_parallelogram_l2341_234115


namespace proof_problem_l2341_234113

theorem proof_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b + (3/4)) * (b^2 + c + (3/4)) * (c^2 + a + (3/4)) ≥ (2 * a + (1/2)) * (2 * b + (1/2)) * (2 * c + (1/2)) := 
by
  sorry

end proof_problem_l2341_234113


namespace find_x_l2341_234195

theorem find_x (x : ℝ) : (x = 2 ∨ x = -2) ↔ (|x|^2 - 5 * |x| + 6 = 0 ∧ x^2 - 4 = 0) :=
by
  sorry

end find_x_l2341_234195


namespace complement_of_P_in_U_l2341_234154

def U : Set ℤ := {-1, 0, 1, 2}
def P : Set ℤ := {x | -Real.sqrt 2 < x ∧ x < Real.sqrt 2}
def compl_U (P : Set ℤ) : Set ℤ := {x ∈ U | x ∉ P}

theorem complement_of_P_in_U : compl_U P = {2} :=
by
  sorry

end complement_of_P_in_U_l2341_234154


namespace ratio_of_original_to_doubled_l2341_234141

theorem ratio_of_original_to_doubled (x : ℕ) (h : x + 5 = 17) : (x / Nat.gcd x (2 * x)) = 1 ∧ ((2 * x) / Nat.gcd x (2 * x)) = 2 := 
by
  sorry

end ratio_of_original_to_doubled_l2341_234141


namespace MelAge_when_Katherine24_l2341_234181

variable (Katherine Mel : ℕ)

-- Conditions
def isYounger (Mel Katherine : ℕ) : Prop :=
  Mel = Katherine - 3

def is24yearsOld (Katherine : ℕ) : Prop :=
  Katherine = 24

-- Statement to Prove
theorem MelAge_when_Katherine24 (Katherine Mel : ℕ) 
  (h1 : isYounger Mel Katherine) 
  (h2 : is24yearsOld Katherine) : 
  Mel = 21 := 
by 
  sorry

end MelAge_when_Katherine24_l2341_234181


namespace tangent_line_circle_l2341_234183

theorem tangent_line_circle (m : ℝ) : 
  (∀ (x y : ℝ), x + y + m = 0 → x^2 + y^2 = m) → m = 2 :=
by
  sorry

end tangent_line_circle_l2341_234183


namespace interval_probability_l2341_234147

theorem interval_probability (x y z : ℝ) (h0x : 0 < x) (hy : y > 0) (hz : z > 0) (hx : x < y) (hy_mul_z_eq_half : y * z = 1 / 2) (hx_eq_third_y : x = 1 / 3 * y) :
  (x / y) = (2 / 3) :=
by
  -- Here you should provide the proof
  sorry

end interval_probability_l2341_234147


namespace arrow_in_48th_position_l2341_234126

def arrow_sequence := ["→", "↔", "↓", "→", "↕"]

theorem arrow_in_48th_position :
  arrow_sequence[48 % arrow_sequence.length] = "↓" :=
by
  sorry

end arrow_in_48th_position_l2341_234126


namespace trapezoid_height_l2341_234135

theorem trapezoid_height (a b : ℝ) (A : ℝ) (h : ℝ) : a = 5 → b = 9 → A = 56 → A = (1 / 2) * (a + b) * h → h = 8 :=
by 
  intros ha hb hA eqn
  sorry

end trapezoid_height_l2341_234135


namespace percentage_employees_at_picnic_l2341_234162

theorem percentage_employees_at_picnic (total_employees men_attend men_percentage women_attend women_percentage : ℝ)
  (h1 : men_attend = 0.20 * (men_percentage * total_employees))
  (h2 : women_attend = 0.40 * ((1 - men_percentage) * total_employees))
  (h3 : men_percentage = 0.30)
  : ((men_attend + women_attend) / total_employees) * 100 = 34 := by
sorry

end percentage_employees_at_picnic_l2341_234162


namespace percent_of_employed_females_l2341_234125

theorem percent_of_employed_females (p e m f : ℝ) (h1 : e = 0.60 * p) (h2 : m = 0.15 * p) (h3 : f = e - m):
  (f / e) * 100 = 75 :=
by
  -- We place the proof here
  sorry

end percent_of_employed_females_l2341_234125


namespace m_plus_n_eq_47_l2341_234104

theorem m_plus_n_eq_47 (m n : ℕ)
  (h1 : m + 8 < n - 1)
  (h2 : (m + m + 3 + m + 8 + n - 1 + n + 3 + 2 * n - 2) / 6 = n)
  (h3 : (m + 8 + (n - 1)) / 2 = n) :
  m + n = 47 :=
sorry

end m_plus_n_eq_47_l2341_234104


namespace largest_fraction_addition_l2341_234193

-- Definitions for the problem conditions
def proper_fraction (a b : ℕ) : Prop :=
  a < b

def denom_less_than (d : ℕ) (bound : ℕ) : Prop :=
  d < bound

-- Main statement of the problem
theorem largest_fraction_addition :
  ∃ (a b : ℕ), (b > 0) ∧ proper_fraction (b + 7 * a) (7 * b) ∧ denom_less_than b 5 ∧ (a / b : ℚ) <= 3/4 := 
sorry

end largest_fraction_addition_l2341_234193


namespace point_not_in_fourth_quadrant_l2341_234199

theorem point_not_in_fourth_quadrant (a : ℝ) :
  ¬ ((a - 3 > 0) ∧ (a + 3 < 0)) :=
by
  sorry

end point_not_in_fourth_quadrant_l2341_234199


namespace transformation_result_l2341_234148

def f (x y : ℝ) : ℝ × ℝ := (y, x)
def g (x y : ℝ) : ℝ × ℝ := (-x, -y)

theorem transformation_result : g (f (-6) (7)).1 (f (-6) (7)).2 = (-7, 6) :=
by
  sorry

end transformation_result_l2341_234148


namespace sum_of_values_of_M_l2341_234155

theorem sum_of_values_of_M (M : ℝ) (h : M * (M - 8) = 12) :
  (∃ M1 M2 : ℝ, M^2 - 8 * M - 12 = 0 ∧ M1 + M2 = 8) :=
sorry

end sum_of_values_of_M_l2341_234155


namespace green_paint_amount_l2341_234133

theorem green_paint_amount (T W B : ℕ) (hT : T = 69) (hW : W = 20) (hB : B = 34) : 
  T - (W + B) = 15 := 
by
  sorry

end green_paint_amount_l2341_234133


namespace fg_of_1_l2341_234179

def f (x : ℤ) : ℤ := x + 3
def g (x : ℤ) : ℤ := x^3 - x^2 - 6

theorem fg_of_1 : f (g 1) = -3 := by
  sorry

end fg_of_1_l2341_234179


namespace contrapositive_squared_l2341_234151

theorem contrapositive_squared (a : ℝ) : (a ≤ 0 → a^2 ≤ 0) ↔ (a > 0 → a^2 > 0) :=
by
  sorry

end contrapositive_squared_l2341_234151


namespace proof_problem_l2341_234144

def U : Set ℤ := {x | x^2 - x - 12 ≤ 0}
def A : Set ℤ := {-2, -1, 3}
def B : Set ℤ := {0, 1, 3, 4}

theorem proof_problem : (U \ A) ∩ B = {0, 1, 4} := 
by sorry

end proof_problem_l2341_234144


namespace part1_solution_part2_solution_l2341_234172

variables (x y m : ℤ)

-- Given the system of equations
def system_of_equations (x y m : ℤ) : Prop :=
  (2 * x - y = m) ∧ (3 * x + 2 * y = m + 7)

-- Part (1) m = 0, find x = 1, y = 2
theorem part1_solution : system_of_equations x y 0 → x = 1 ∧ y = 2 :=
sorry

-- Part (2) point A(-2,3) in the second quadrant with distances 3 and 2, find m = -7
def is_in_second_quadrant (x y : ℤ) : Prop :=
  x < 0 ∧ y > 0

def distance_to_axes (x y dx dy : ℤ) : Prop :=
  y = dy ∧ x = -dx

theorem part2_solution : is_in_second_quadrant x y →
  distance_to_axes x y 2 3 →
  system_of_equations x y m →
  m = -7 :=
sorry

end part1_solution_part2_solution_l2341_234172


namespace find_k_l2341_234105

def A (a b : ℤ) : Prop := 3 * a + b - 2 = 0
def B (a b : ℤ) (k : ℤ) : Prop := k * (a^2 - a + 1) - b = 0

theorem find_k (k : ℤ) (h : ∃ a b : ℤ, A a b ∧ B a b k ∧ a > 0) : k = -1 ∨ k = 2 :=
by
  sorry

end find_k_l2341_234105


namespace infinite_solutions_iff_a_eq_neg12_l2341_234143

theorem infinite_solutions_iff_a_eq_neg12 {a : ℝ} : 
  (∀ x : ℝ, 4 * (3 * x - a) = 3 * (4 * x + 16)) ↔ a = -12 :=
by 
  sorry

end infinite_solutions_iff_a_eq_neg12_l2341_234143


namespace solution_set_condition_l2341_234166

-- The assumptions based on the given conditions
variables (a b : ℝ)

noncomputable def inequality_system_solution_set (x : ℝ) : Prop :=
  (x + 2 * a > 4) ∧ (2 * x - b < 5)

theorem solution_set_condition (a b : ℝ) :
  (∀ x : ℝ, inequality_system_solution_set a b x ↔ 0 < x ∧ x < 2) →
  (a + b) ^ 2023 = 1 :=
by
  intro h
  sorry

end solution_set_condition_l2341_234166


namespace part_I_part_II_l2341_234160

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + ((2 * a^2) / x) + x

theorem part_I (a : ℝ) (h : a > 0) : 
  (∃ x : ℝ, x = 1 ∧ deriv (f a) x = -2) → a = 3 / 2 :=
sorry

theorem part_II (a : ℝ) (h : a = 3 / 2) : 
  (∀ x : ℝ, 0 < x ∧ x < 3 / 2 → deriv (f a) x < 0) ∧ 
  (∀ x : ℝ, x > 3 / 2 → deriv (f a) x > 0) :=
sorry

end part_I_part_II_l2341_234160


namespace xy_value_l2341_234146

theorem xy_value (x y : ℝ) (h : (x - 3)^2 + |y + 2| = 0) : x * y = -6 :=
by {
  sorry
}

end xy_value_l2341_234146


namespace expand_polynomial_l2341_234173

theorem expand_polynomial (t : ℝ) :
  (3 * t^3 - 4 * t^2 + 5 * t - 3) * (4 * t^2 - 2 * t + 1) = 12 * t^5 - 22 * t^4 + 31 * t^3 - 26 * t^2 + 11 * t - 3 := by
  sorry

end expand_polynomial_l2341_234173


namespace part_a_part_b_l2341_234145

variable (f : ℝ → ℝ)

-- Part (a)
theorem part_a (h : ∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x)) :
  ∀ x : ℝ, f (f x) ≤ 0 :=
sorry

-- Part (b)
theorem part_b (h : ∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x)) (h₀ : f 0 ≥ 0) :
  ∀ x : ℝ, f x = 0 :=
sorry

end part_a_part_b_l2341_234145


namespace gain_in_transaction_per_year_l2341_234138

noncomputable def borrowing_interest (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal * rate * time

noncomputable def lending_interest (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal * rate * time

noncomputable def gain_per_year (borrow_principal : ℕ) (borrow_rate : ℚ) 
  (borrow_time : ℕ) (lend_principal : ℕ) (lend_rate : ℚ) (lend_time : ℕ) : ℚ :=
  (lending_interest lend_principal lend_rate lend_time - borrowing_interest borrow_principal borrow_rate borrow_time) / borrow_time

theorem gain_in_transaction_per_year :
  gain_per_year 4000 (4 / 100) 2 4000 (6 / 100) 2 = 80 := 
sorry

end gain_in_transaction_per_year_l2341_234138


namespace find_a5_plus_a7_l2341_234176

variable {a : ℕ → ℕ}

-- Assume a is a geometric sequence with common ratio q and first term a1.
def geometric_sequence (a : ℕ → ℕ) (a_1 : ℕ) (q : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a_1 * q ^ n

-- Given conditions of the problem:
def conditions (a : ℕ → ℕ) : Prop :=
  a 2 + a 4 = 20 ∧ a 3 + a 5 = 40

-- The objective is to prove a_5 + a_7 = 160
theorem find_a5_plus_a7 (a : ℕ → ℕ) (a_1 q : ℕ) (h_geo : geometric_sequence a a_1 q) (h_cond : conditions a) : a 5 + a 7 = 160 :=
  sorry

end find_a5_plus_a7_l2341_234176


namespace mixing_paint_l2341_234108

theorem mixing_paint (total_parts : ℕ) (blue_parts : ℕ) (red_parts : ℕ) (white_parts : ℕ) (blue_ounces : ℕ) (max_mixture : ℕ) (ounces_per_part : ℕ) :
  total_parts = blue_parts + red_parts + white_parts →
  blue_parts = 7 →
  red_parts = 2 →
  white_parts = 1 →
  blue_ounces = 140 →
  max_mixture = 180 →
  ounces_per_part = blue_ounces / blue_parts →
  max_mixture / ounces_per_part = 9 →
  white_ounces = white_parts * ounces_per_part →
  white_ounces = 20 :=
sorry

end mixing_paint_l2341_234108


namespace new_person_weight_l2341_234112

/-- 
The average weight of 10 persons increases by 6.3 kg when a new person comes 
in place of one of them weighing 65 kg. Prove that the weight of the new person 
is 128 kg.
-/
theorem new_person_weight (w_old : ℝ) (n : ℝ) (delta_w : ℝ) (w_new : ℝ) 
  (h1 : w_old = 65) 
  (h2 : n = 10) 
  (h3 : delta_w = 6.3) 
  (h4 : w_new = w_old + n * delta_w) : 
  w_new = 128 :=
by 
  rw [h1, h2, h3] at h4 
  rw [h4]
  norm_num

end new_person_weight_l2341_234112


namespace linear_polynomial_divisible_49_l2341_234116

theorem linear_polynomial_divisible_49 {P : ℕ → Polynomial ℚ} :
    let Q := Polynomial.C 1 * (Polynomial.X ^ 8) + Polynomial.C 1 * (Polynomial.X ^ 7)
    ∃ a b x, (P x) = Polynomial.C a * Polynomial.X + Polynomial.C b ∧ a ≠ 0 ∧ 
              (∀ i, P (i + 1) = (Polynomial.C 1 * Polynomial.X + Polynomial.C 1) * P i ∨ 
                            P (i + 1) = Polynomial.derivative (P i)) →
              (a - b) % 49 = 0 :=
by
  sorry

end linear_polynomial_divisible_49_l2341_234116


namespace smallest_is_57_l2341_234190

noncomputable def smallest_of_four_numbers (a b c d : ℕ) : ℕ :=
  if h1 : a + b + c = 234 ∧ a + b + d = 251 ∧ a + c + d = 284 ∧ b + c + d = 299
  then Nat.min (Nat.min a b) (Nat.min c d)
  else 0

theorem smallest_is_57 (a b c d : ℕ) (h1 : a + b + c = 234) (h2 : a + b + d = 251)
  (h3 : a + c + d = 284) (h4 : b + c + d = 299) :
  smallest_of_four_numbers a b c d = 57 :=
sorry

end smallest_is_57_l2341_234190


namespace max_possible_N_in_cities_l2341_234185

theorem max_possible_N_in_cities (N : ℕ) (num_cities : ℕ) (roads : ℕ → List ℕ) :
  (num_cities = 110) →
  (∀ n, 1 ≤ n ∧ n ≤ N → List.length (roads n) = n) →
  N ≤ 107 :=
by
  sorry

end max_possible_N_in_cities_l2341_234185


namespace calc_abc_squares_l2341_234109

theorem calc_abc_squares :
  ∀ (a b c : ℝ),
  a^2 + 3 * b = 14 →
  b^2 + 5 * c = -13 →
  c^2 + 7 * a = -26 →
  a^2 + b^2 + c^2 = 20.75 :=
by
  intros a b c h1 h2 h3
  -- The proof is omitted; reasoning is provided in the solution.
  sorry

end calc_abc_squares_l2341_234109


namespace delta_max_success_ratio_l2341_234134

/-- In a two-day math challenge, Gamma and Delta both attempted questions totalling 600 points. 
    Gamma scored 180 points out of 300 points attempted each day.
    Delta attempted a different number of points each day and their daily success ratios were less by both days than Gamma's, 
    whose overall success ratio was 3/5. Prove that the maximum possible two-day success ratio that Delta could have achieved was 359/600. -/
theorem delta_max_success_ratio :
  ∀ (x y z w : ℕ), (0 < x) ∧ (0 < y) ∧ (0 < z) ∧ (0 < w) ∧ (x ≤ (3 * y) / 5) ∧ (z ≤ (3 * w) / 5) ∧ (y + w = 600) ∧ (x + z < 360)
  → (x + z) / 600 ≤ 359 / 600 :=
by
  sorry

end delta_max_success_ratio_l2341_234134


namespace find_F_l2341_234102

theorem find_F (F C : ℝ) (h1 : C = 30) (h2 : C = (5 / 9) * (F - 30)) : F = 84 := by
  sorry

end find_F_l2341_234102


namespace female_students_in_first_class_l2341_234189

theorem female_students_in_first_class
  (females_in_second_class : ℕ)
  (males_in_first_class : ℕ)
  (males_in_second_class : ℕ)
  (males_in_third_class : ℕ)
  (females_in_third_class : ℕ)
  (extra_students : ℕ)
  (total_students_need_partners : ℕ)
  (total_males : ℕ := males_in_first_class + males_in_second_class + males_in_third_class)
  (total_females : ℕ := females_in_second_class + females_in_third_class)
  (females_in_first_class : ℕ)
  (females : ℕ := females_in_first_class + total_females) :
  (females_in_second_class = 18) →
  (males_in_first_class = 17) →
  (males_in_second_class = 14) →
  (males_in_third_class = 15) →
  (females_in_third_class = 17) →
  (extra_students = 2) →
  (total_students_need_partners = total_males - extra_students) →
  females = total_students_need_partners →
  females_in_first_class = 9 :=
by
  intros
  sorry

end female_students_in_first_class_l2341_234189


namespace roots_of_polynomial_l2341_234110

theorem roots_of_polynomial :
  ∀ x : ℝ, (2 * x^3 - 3 * x^2 - 13 * x + 10) * (x - 1) = 0 → x = 1 :=
by
  sorry

end roots_of_polynomial_l2341_234110


namespace soda_cost_proof_l2341_234101

theorem soda_cost_proof (b s : ℤ) (h1 : 4 * b + 3 * s = 440) (h2 : 3 * b + 2 * s = 310) : s = 80 :=
by
  sorry

end soda_cost_proof_l2341_234101


namespace unique_x_inequality_l2341_234159

theorem unique_x_inequality (a : ℝ) : (∀ x : ℝ, |x^2 + 2*a*x + 3*a| ≤ 2 → (a = 1 ∨ a = 2)) :=
by
  sorry

end unique_x_inequality_l2341_234159


namespace balls_in_boxes_with_one_in_one_balls_in_boxes_with_two_empty_balls_in_boxes_with_three_empty_balls_in_boxes_A_not_less_B_l2341_234194

noncomputable def ways_with_ball_in_box_one : Nat := 369
noncomputable def ways_with_two_empty_boxes : Nat := 360
noncomputable def ways_with_three_empty_boxes : Nat := 140
noncomputable def ways_ball_A_not_less_than_B : Nat := 375

theorem balls_in_boxes_with_one_in_one 
  (n_balls : Nat) (n_boxes : Nat) 
  (ball_1 : Nat) :
  n_balls = 4 → n_boxes = 5 → ball_1 = 1 → 
  ∃ ways, ways = ways_with_ball_in_box_one := 
sorry

theorem balls_in_boxes_with_two_empty 
  (n_balls : Nat) (n_boxes : Nat) 
  (empty_boxes : Nat) :
  n_balls = 4 → n_boxes = 5 → empty_boxes = 2 → 
  ∃ ways, ways = ways_with_two_empty_boxes := 
sorry

theorem balls_in_boxes_with_three_empty 
  (n_balls : Nat) (n_boxes : Nat) 
  (empty_boxes : Nat) :
  n_balls = 4 → n_boxes = 5 → empty_boxes = 3 → 
  ∃ ways, ways = ways_with_three_empty_boxes := 
sorry

theorem balls_in_boxes_A_not_less_B 
  (n_balls : Nat) (n_boxes : Nat) 
  (ball_A : Nat) (ball_B : Nat) :
  n_balls = 4 → n_boxes = 5 → ball_A ≠ ball_B →
  ∃ ways, ways = ways_ball_A_not_less_than_B := 
sorry

end balls_in_boxes_with_one_in_one_balls_in_boxes_with_two_empty_balls_in_boxes_with_three_empty_balls_in_boxes_A_not_less_B_l2341_234194


namespace x_power_expression_l2341_234180

theorem x_power_expression (x : ℝ) (h : x^3 - 3 * x = 5) : x^5 - 27 * x^2 = -22 * x^2 + 9 * x + 15 :=
by
  --proof goes here
  sorry

end x_power_expression_l2341_234180


namespace number_of_children_l2341_234100

theorem number_of_children (C A : ℕ) (h1 : C = 2 * A) (h2 : C + A = 120) : C = 80 :=
by
  sorry

end number_of_children_l2341_234100


namespace euler_totient_divisibility_l2341_234164

theorem euler_totient_divisibility (a n: ℕ) (h1 : a ≥ 2) : (n ∣ Nat.totient (a^n - 1)) :=
sorry

end euler_totient_divisibility_l2341_234164


namespace judgments_correct_l2341_234184

variables {l m : Line} (a : Plane)

def is_perpendicular (l : Line) (a : Plane) : Prop := -- Definition of perpendicularity between a line and a plane
sorry

def is_parallel (l m : Line) : Prop := -- Definition of parallel lines
sorry

def is_contained_in (m : Line) (a : Plane) : Prop := -- Definition of a line contained in a plane
sorry

theorem judgments_correct 
  (hl : is_perpendicular l a)
  (hm : l ≠ m) :
  (∀ m, is_perpendicular m l → is_parallel m a) ∧ 
  (is_perpendicular m a → is_parallel m l) ∧
  (is_contained_in m a → is_perpendicular m l) ∧
  (is_parallel m l → is_perpendicular m a) :=
sorry

end judgments_correct_l2341_234184


namespace roots_quadratic_eq_l2341_234192

theorem roots_quadratic_eq :
  (∃ a b : ℝ, (a + b = 8) ∧ (a * b = 8) ∧ (a^2 + b^2 = 48)) :=
sorry

end roots_quadratic_eq_l2341_234192


namespace relationship_between_a_b_c_l2341_234187

noncomputable def a := (3 / 5 : ℝ) ^ (2 / 5)
noncomputable def b := (2 / 5 : ℝ) ^ (3 / 5)
noncomputable def c := (2 / 5 : ℝ) ^ (2 / 5)

theorem relationship_between_a_b_c :
  a > c ∧ c > b :=
by
  sorry

end relationship_between_a_b_c_l2341_234187


namespace contrapositive_ex_l2341_234114

theorem contrapositive_ex (x y : ℝ)
  (h : x^2 + y^2 = 0 → x = 0 ∧ y = 0) :
  ¬ (x = 0 ∧ y = 0) → x^2 + y^2 ≠ 0 :=
by
  sorry

end contrapositive_ex_l2341_234114


namespace perpendicular_vectors_x_value_l2341_234163

theorem perpendicular_vectors_x_value :
  let a := (4, 2)
  let b := (x, 3)
  a.1 * b.1 + a.2 * b.2 = 0 -> x = -3/2 :=
by
  intros
  sorry

end perpendicular_vectors_x_value_l2341_234163


namespace range_of_a_l2341_234117

def p (a : ℝ) := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

def q (a : ℝ) := ∀ x₁ x₂ : ℝ, x₁ < x₂ → -(5 - 2 * a)^x₁ > -(5 - 2 * a)^x₂

theorem range_of_a (a : ℝ) : (p a ∨ q a) → ¬ (p a ∧ q a) → a ≤ -2 := by 
  sorry

end range_of_a_l2341_234117


namespace symmetric_about_y_axis_l2341_234107

noncomputable def f (x : ℝ) : ℝ := (4^x + 1) / 2^x

theorem symmetric_about_y_axis : ∀ x : ℝ, f (-x) = f x :=
by
  intro x
  unfold f
  sorry

end symmetric_about_y_axis_l2341_234107


namespace laura_bought_4_shirts_l2341_234111

-- Definitions for the conditions
def pants_price : ℕ := 54
def num_pants : ℕ := 2
def shirt_price : ℕ := 33
def given_money : ℕ := 250
def change_received : ℕ := 10

-- Proving the number of shirts bought is 4
theorem laura_bought_4_shirts :
  (num_pants * pants_price) + (shirt_price * 4) + change_received = given_money :=
by
  sorry

end laura_bought_4_shirts_l2341_234111


namespace boats_meet_time_l2341_234168

theorem boats_meet_time (v_A v_C current distance : ℝ) : 
  v_A = 7 → 
  v_C = 3 → 
  current = 2 → 
  distance = 20 → 
  (distance / (v_A + current + v_C - current) = 2 ∨
   distance / (v_A + current - (v_C + current)) = 5) := 
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- Apply simplifications or calculations as necessary
  sorry

end boats_meet_time_l2341_234168


namespace more_birds_than_nests_l2341_234120

theorem more_birds_than_nests (birds nests : Nat) (h_birds : birds = 6) (h_nests : nests = 3) : birds - nests = 3 :=
by
  sorry

end more_birds_than_nests_l2341_234120


namespace trig_identity_proof_l2341_234186

theorem trig_identity_proof :
  Real.sin (30 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) + 
  Real.sin (60 * Real.pi / 180) * Real.sin (15 * Real.pi / 180) =
  Real.sqrt 2 / 2 := 
by
  sorry

end trig_identity_proof_l2341_234186


namespace sunday_to_saturday_ratio_l2341_234167

theorem sunday_to_saturday_ratio : 
  ∀ (sold_friday sold_saturday sold_sunday total_sold : ℕ),
  sold_friday = 40 →
  sold_saturday = (2 * sold_friday - 10) →
  total_sold = 145 →
  total_sold = sold_friday + sold_saturday + sold_sunday →
  (sold_sunday : ℚ) / (sold_saturday : ℚ) = 1 / 2 :=
by
  intro sold_friday sold_saturday sold_sunday total_sold
  intros h_friday h_saturday h_total h_sum
  sorry

end sunday_to_saturday_ratio_l2341_234167


namespace sum_of_roots_eq_4140_l2341_234158

open Complex

noncomputable def sum_of_roots : ℝ :=
  let θ0 := 270 / 5;
  let θ1 := (270 + 360) / 5;
  let θ2 := (270 + 2 * 360) / 5;
  let θ3 := (270 + 3 * 360) / 5;
  let θ4 := (270 + 4 * 360) / 5;
  θ0 + θ1 + θ2 + θ3 + θ4

theorem sum_of_roots_eq_4140 : sum_of_roots = 4140 := by
  sorry

end sum_of_roots_eq_4140_l2341_234158


namespace square_root_of_25_squared_l2341_234198

theorem square_root_of_25_squared :
  Real.sqrt (25 ^ 2) = 25 :=
sorry

end square_root_of_25_squared_l2341_234198


namespace relationship_between_a_b_c_l2341_234170

noncomputable def a : ℝ := (1 / 2)^(1 / 3)
noncomputable def b : ℝ := Real.log 2 / Real.log (1 / 3)
noncomputable def c : ℝ := Real.log 3 / Real.log (1 / 2)

theorem relationship_between_a_b_c : c < b ∧ b < a :=
by
  sorry

end relationship_between_a_b_c_l2341_234170


namespace quadratic_roots_m_value_l2341_234128

noncomputable def quadratic_roots_condition (m : ℝ) (x1 x2 : ℝ) : Prop :=
  (∀ a b c : ℝ, a = 1 ∧ b = 2 * (m + 1) ∧ c = m^2 - 1 → x1^2 + b * x1 + c = 0 ∧ x2^2 + b * x2 + c = 0) ∧ 
  (x1 - x2)^2 = 16 - x1 * x2

theorem quadratic_roots_m_value (m : ℝ) (x1 x2 : ℝ) (h : quadratic_roots_condition m x1 x2) : m = 1 :=
sorry

end quadratic_roots_m_value_l2341_234128


namespace parameter_range_exists_solution_l2341_234157

theorem parameter_range_exists_solution :
  {a : ℝ | ∃ b : ℝ, ∃ x y : ℝ,
    x^2 + y^2 + 2 * a * (a + y - x) = 49 ∧
    y = 15 * Real.cos (x - b) - 8 * Real.sin (x - b)
  } = {a : ℝ | -24 ≤ a ∧ a ≤ 24} :=
sorry

end parameter_range_exists_solution_l2341_234157


namespace proof_mn_proof_expr_l2341_234118

variables (m n : ℚ)
-- Conditions
def condition1 : Prop := (m + n)^2 = 9
def condition2 : Prop := (m - n)^2 = 1

-- Expected results
def expected_mn : ℚ := 2
def expected_expr : ℚ := 3

-- The theorem to be proved
theorem proof_mn : condition1 m n → condition2 m n → m * n = expected_mn :=
by
  sorry

theorem proof_expr : condition1 m n → condition2 m n → m^2 + n^2 - m * n = expected_expr :=
by
  sorry

end proof_mn_proof_expr_l2341_234118


namespace total_cost_after_discounts_and_cashback_l2341_234150

def iPhone_original_price : ℝ := 800
def iWatch_original_price : ℝ := 300
def iPhone_discount_rate : ℝ := 0.15
def iWatch_discount_rate : ℝ := 0.10
def cashback_rate : ℝ := 0.02

theorem total_cost_after_discounts_and_cashback :
  (iPhone_original_price * (1 - iPhone_discount_rate) + iWatch_original_price * (1 - iWatch_discount_rate)) * (1 - cashback_rate) = 931 :=
by sorry

end total_cost_after_discounts_and_cashback_l2341_234150


namespace log_inequality_region_l2341_234175

theorem log_inequality_region (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hx1 : x ≠ 1) (hx2 : x ≠ y) :
  (0 < x ∧ x < 1 ∧ 0 < y ∧ y < x) 
  ∨ (1 < x ∧ y > x) ↔ (Real.log y / Real.log x ≥ Real.log (x * y) / Real.log (x / y)) :=
  sorry

end log_inequality_region_l2341_234175


namespace range_of_function_l2341_234124

theorem range_of_function :
  ∀ y : ℝ, ∃ x : ℝ, (x ≤ 1/2) ∧ (y = 2 * x - Real.sqrt (1 - 2 * x)) ↔ y ∈ Set.Iic 1 := 
by
  sorry

end range_of_function_l2341_234124


namespace difference_of_squares_65_35_l2341_234161

theorem difference_of_squares_65_35 :
  let a := 65
  let b := 35
  a^2 - b^2 = (a + b) * (a - b) :=
by
  sorry

end difference_of_squares_65_35_l2341_234161


namespace sale_price_lower_than_original_l2341_234182

noncomputable def original_price (p : ℝ) : ℝ := 
  p

noncomputable def increased_price (p : ℝ) : ℝ := 
  1.30 * p

noncomputable def sale_price (p : ℝ) : ℝ := 
  0.75 * increased_price p

theorem sale_price_lower_than_original (p : ℝ) : 
  sale_price p = 0.975 * p := 
sorry

end sale_price_lower_than_original_l2341_234182


namespace no_integer_solutions_x_x_plus_1_eq_13y_plus_1_l2341_234153

theorem no_integer_solutions_x_x_plus_1_eq_13y_plus_1 :
  ¬ ∃ x y : ℤ, x * (x + 1) = 13 * y + 1 :=
by sorry

end no_integer_solutions_x_x_plus_1_eq_13y_plus_1_l2341_234153


namespace range_of_m_l2341_234131

-- Define the function and its properties
variable {f : ℝ → ℝ}
variable (increasing : ∀ x1 x2 : ℝ, x1 > x2 → f x1 > f x2)

theorem range_of_m (h: ∀ m : ℝ, f (2 * m) > f (-m + 9)) : 
  ∀ m : ℝ, m > 3 ↔ f (2 * m) > f (-m + 9) :=
by
  intros
  sorry

end range_of_m_l2341_234131


namespace largest_reciprocal_l2341_234103

theorem largest_reciprocal: 
  let A := -(1 / 4)
  let B := 2 / 7
  let C := -2
  let D := 3
  let E := -(3 / 2)
  let reciprocal (x : ℚ) := 1 / x
  reciprocal B > reciprocal A ∧
  reciprocal B > reciprocal C ∧
  reciprocal B > reciprocal D ∧
  reciprocal B > reciprocal E :=
by
  sorry

end largest_reciprocal_l2341_234103


namespace total_texts_sent_l2341_234140

theorem total_texts_sent (grocery_texts : ℕ) (response_texts_ratio : ℕ) (police_texts_percentage : ℚ) :
  grocery_texts = 5 →
  response_texts_ratio = 5 →
  police_texts_percentage = 0.10 →
  let response_texts := grocery_texts * response_texts_ratio
  let previous_texts := response_texts + grocery_texts
  let police_texts := previous_texts * police_texts_percentage
  response_texts + grocery_texts + police_texts = 33 :=
by
  sorry

end total_texts_sent_l2341_234140


namespace determine_d_l2341_234169

variables (u v : ℝ × ℝ × ℝ) -- defining u and v as 3D vectors

noncomputable def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.2 * b.2.1 - a.2.1 * b.2.2, a.1 * b.2.2 - a.2.2 * b.1 , a.2.1 * b.1 - a.1 * b.2.1)

noncomputable def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2

noncomputable def i : ℝ × ℝ × ℝ := (1, 0, 0)
noncomputable def j : ℝ × ℝ × ℝ := (0, 1, 0)
noncomputable def k : ℝ × ℝ × ℝ := (0, 0, 1)

theorem determine_d (u : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) :
  cross_product i (cross_product (u + v) i) +
  cross_product j (cross_product (u + v) j) +
  cross_product k (cross_product (u + v) k) =
  2 * (u + v) :=
sorry

end determine_d_l2341_234169


namespace unfolded_paper_has_eight_holes_l2341_234130

theorem unfolded_paper_has_eight_holes
  (T : Type)
  (equilateral_triangle : T)
  (midpoint : T → T → T)
  (vertex_fold : T → T → T)
  (holes_punched : T → ℕ)
  (first_fold_vertex midpoint_1 : T)
  (second_fold_vertex midpoint_2 : T)
  (holes_near_first_fold holes_near_second_fold : ℕ) :
  holes_punched (vertex_fold second_fold_vertex midpoint_2)
    = 8 := 
by sorry

end unfolded_paper_has_eight_holes_l2341_234130


namespace A_wins_probability_is_3_over_4_l2341_234196

def parity (n : ℕ) : Bool := n % 2 == 0

def number_of_dice_outcomes : ℕ := 36

def same_parity_outcome : ℕ := 18

def probability_A_wins : ℕ → ℕ → ℕ → ℚ
| total_outcomes, same_parity, different_parity =>
  (same_parity / total_outcomes : ℚ) * 1 + (different_parity / total_outcomes : ℚ) * (1 / 2)

theorem A_wins_probability_is_3_over_4 :
  probability_A_wins number_of_dice_outcomes same_parity_outcome (number_of_dice_outcomes - same_parity_outcome) = 3/4 :=
by
  sorry

end A_wins_probability_is_3_over_4_l2341_234196


namespace true_propositions_l2341_234119

def p : Prop :=
  ∀ a b : ℝ, (a > 2 ∧ b > 2) → a + b > 4

def q : Prop :=
  ¬ ∃ x : ℝ, x^2 - x > 0 → ∀ x : ℝ, x^2 - x ≤ 0

theorem true_propositions :
  (¬ p ∨ ¬ q) ∧ (p ∨ ¬ q) := by
  sorry

end true_propositions_l2341_234119


namespace triangle_area_solution_l2341_234132

noncomputable def solve_for_x (x : ℝ) : Prop :=
  x > 0 ∧ (1 / 2 * x * 3 * x = 96) → x = 8

theorem triangle_area_solution : solve_for_x 8 :=
by
  sorry

end triangle_area_solution_l2341_234132


namespace sin_inverse_equation_l2341_234137

noncomputable def a := Real.arcsin (4/5)
noncomputable def b := Real.arctan 1
noncomputable def c := Real.arccos (1/3)
noncomputable def sin_a_plus_b_minus_c := Real.sin (a + b - c)

theorem sin_inverse_equation : sin_a_plus_b_minus_c = 11 / 15 := sorry

end sin_inverse_equation_l2341_234137


namespace toothpaste_amount_in_tube_l2341_234139

def dad_usage_per_brush : ℕ := 3
def mom_usage_per_brush : ℕ := 2
def kid_usage_per_brush : ℕ := 1
def brushes_per_day : ℕ := 3
def days : ℕ := 5

theorem toothpaste_amount_in_tube (dad_usage_per_brush mom_usage_per_brush kid_usage_per_brush brushes_per_day days : ℕ) : 
  dad_usage_per_brush * brushes_per_day * days + 
  mom_usage_per_brush * brushes_per_day * days + 
  (kid_usage_per_brush * brushes_per_day * days * 2) = 105 := 
  by sorry

end toothpaste_amount_in_tube_l2341_234139


namespace four_digit_numbers_div_by_5_with_34_end_l2341_234129

theorem four_digit_numbers_div_by_5_with_34_end : 
  ∃ (count : ℕ), count = 90 ∧
  ∀ (n : ℕ), (1000 ≤ n ∧ n < 10000) →
  (n % 100 = 34) →
  ((10 ∣ n) ∨ (5 ∣ n)) →
  (count = 90) :=
sorry

end four_digit_numbers_div_by_5_with_34_end_l2341_234129


namespace band_to_orchestra_ratio_is_two_l2341_234178

noncomputable def ratio_of_band_to_orchestra : ℤ :=
  let orchestra_students := 20
  let choir_students := 28
  let total_students := 88
  let band_students := (total_students - orchestra_students - choir_students)
  band_students / orchestra_students

theorem band_to_orchestra_ratio_is_two :
  let orchestra_students := 20
  let choir_students := 28
  let total_students := 88
  ratio_of_band_to_orchestra = 2 := by
  sorry

end band_to_orchestra_ratio_is_two_l2341_234178


namespace total_students_l2341_234174

noncomputable def total_students_in_gym (F : ℕ) (T : ℕ) : Prop :=
  T = 26

theorem total_students (F T : ℕ) (h1 : 4 = T - F) (h2 : F / (F + 4) = 11 / 13) : total_students_in_gym F T :=
by sorry

end total_students_l2341_234174


namespace prob_at_least_one_l2341_234106

-- Defining the probabilities of the alarms going off on time
def prob_A : ℝ := 0.80
def prob_B : ℝ := 0.90

-- Define the complementary event (neither alarm goes off on time)
def prob_neither : ℝ := (1 - prob_A) * (1 - prob_B)

-- The main theorem statement we need to prove
theorem prob_at_least_one : 1 - prob_neither = 0.98 :=
by
  sorry

end prob_at_least_one_l2341_234106


namespace unique_B_squared_l2341_234123

variable (B : Matrix (Fin 2) (Fin 2) ℝ)

theorem unique_B_squared (h : B ^ 4 = 0) :
  ∃! B2 : Matrix (Fin 2) (Fin 2) ℝ, B ^ 2 = B2 :=
by sorry

end unique_B_squared_l2341_234123


namespace sum_of_consecutive_integers_between_ln20_l2341_234122

theorem sum_of_consecutive_integers_between_ln20 : ∃ a b : ℤ, a < b ∧ b = a + 1 ∧ 1 ≤ a ∧ a + 1 ≤ 3 ∧ (a + b = 4) :=
by
  sorry

end sum_of_consecutive_integers_between_ln20_l2341_234122


namespace sum_of_first_odd_numbers_l2341_234191

theorem sum_of_first_odd_numbers (S1 S2 : ℕ) (n1 n2 : ℕ)
  (hS1 : S1 = n1^2) 
  (hS2 : S2 = n2^2) 
  (h1 : S1 = 2500)
  (h2 : S2 = 5625) : 
  n2 = 75 := by
  sorry

end sum_of_first_odd_numbers_l2341_234191


namespace max_observing_relations_lemma_l2341_234127

/-- There are 24 robots on a plane, each with a 70-degree field of view. -/
def robots : ℕ := 24

/-- Definition of field of view for each robot. -/
def field_of_view : ℝ := 70

/-- Maximum number of observing relations. Observing is a one-sided relation. -/
def max_observing_relations := 468

/-- Theorem: The maximum number of observing relations among 24 robots,
each with a 70-degree field of view, is 468. -/
theorem max_observing_relations_lemma : max_observing_relations = 468 :=
by
  sorry

end max_observing_relations_lemma_l2341_234127


namespace total_pizza_eaten_l2341_234156

def don_pizzas : ℝ := 80
def daria_pizzas : ℝ := 2.5 * don_pizzas
def total_pizzas : ℝ := don_pizzas + daria_pizzas

theorem total_pizza_eaten : total_pizzas = 280 := by
  sorry

end total_pizza_eaten_l2341_234156


namespace find_number_l2341_234149

variables (n : ℝ)

-- Condition: a certain number divided by 14.5 equals 173.
def condition_1 (n : ℝ) : Prop := n / 14.5 = 173

-- Condition: 29.94 ÷ 1.45 = 17.3.
def condition_2 : Prop := 29.94 / 1.45 = 17.3

-- Theorem: Prove that the number is 2508.5 given the conditions.
theorem find_number (h1 : condition_1 n) (h2 : condition_2) : n = 2508.5 :=
by 
  sorry

end find_number_l2341_234149


namespace train_speed_l2341_234136

theorem train_speed (length_m : ℕ) (time_s : ℕ) (length_km : ℝ) (time_hr : ℝ) 
(length_conversion : length_km = (length_m : ℝ) / 1000)
(time_conversion : time_hr = (time_s : ℝ) / 3600)
(speed : ℝ) (speed_formula : speed = length_km / time_hr) :
  length_m = 300 → time_s = 18 → speed = 60 :=
by
  intros h1 h2
  rw [h1, h2] at *
  simp [length_conversion, time_conversion, speed_formula]
  norm_num
  sorry

end train_speed_l2341_234136


namespace move_right_by_three_units_l2341_234121

theorem move_right_by_three_units :
  (-1 + 3 = 2) :=
  by { sorry }

end move_right_by_three_units_l2341_234121


namespace range_of_m_l2341_234177

theorem range_of_m (m : ℝ) : (∀ x : ℝ, m*x^2 + m*x + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
by
  sorry

end range_of_m_l2341_234177


namespace election_votes_l2341_234142

theorem election_votes (T V : ℕ) 
    (hT : 8 * T = 11 * 20000) 
    (h_total_votes : T = 2500 + V + 20000) :
    V = 5000 :=
by
    sorry

end election_votes_l2341_234142


namespace chocolate_bars_sold_last_week_l2341_234188

-- Definitions based on conditions
def initial_chocolate_bars : Nat := 18
def chocolate_bars_sold_this_week : Nat := 7
def chocolate_bars_needed_to_sell : Nat := 6

-- Define the number of chocolate bars sold so far
def chocolate_bars_sold_so_far : Nat := chocolate_bars_sold_this_week + chocolate_bars_needed_to_sell

-- Target statement to prove
theorem chocolate_bars_sold_last_week :
  initial_chocolate_bars - chocolate_bars_sold_so_far = 5 :=
by
  sorry

end chocolate_bars_sold_last_week_l2341_234188


namespace sum_infinite_series_eq_l2341_234197

theorem sum_infinite_series_eq : 
  ∑' n : ℕ, (n + 1) * (1 / 999 : ℝ) ^ n = 1000 / 998 := by
sorry

end sum_infinite_series_eq_l2341_234197
