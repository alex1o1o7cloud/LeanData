import Mathlib

namespace bumper_cars_line_l210_210477

theorem bumper_cars_line (initial in_line_leaving newcomers : ℕ) 
  (h_initial : initial = 9)
  (h_leaving : in_line_leaving = 6)
  (h_newcomers : newcomers = 3) :
  initial - in_line_leaving + newcomers = 6 :=
by
  sorry

end bumper_cars_line_l210_210477


namespace sum_of_coefficients_l210_210574

theorem sum_of_coefficients (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℤ) (hx : (1 - 2 * x)^7 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7) :
  a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 12 :=
sorry

end sum_of_coefficients_l210_210574


namespace distance_symmetric_reflection_l210_210542

theorem distance_symmetric_reflection (x : ℝ) (y : ℝ) (B : (ℝ × ℝ)) 
  (hB : B = (-1, 4)) (A : (ℝ × ℝ)) (hA : A = (x, -y)) : 
  dist A B = 8 :=
by
  sorry

end distance_symmetric_reflection_l210_210542


namespace yuna_survey_l210_210453

theorem yuna_survey :
  let M := 27
  let K := 28
  let B := 22
  M + K - B = 33 :=
by
  sorry

end yuna_survey_l210_210453


namespace smallest_divisor_l210_210298

theorem smallest_divisor (n : ℕ) (h1 : n = 999) :
  ∃ d : ℕ, 2.45 ≤ (999 : ℝ) / d ∧ (999 : ℝ) / d < 2.55 ∧ d = 392 :=
by
  sorry

end smallest_divisor_l210_210298


namespace carolyn_sum_of_removed_numbers_eq_31_l210_210507

theorem carolyn_sum_of_removed_numbers_eq_31 :
  let initial_list := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let carolyn_first_turn := 4
  let carolyn_numbers_removed := [4, 9, 10, 8]
  let sum := carolyn_numbers_removed.sum
  sum = 31 :=
by
  sorry

end carolyn_sum_of_removed_numbers_eq_31_l210_210507


namespace ab_eq_zero_l210_210057

theorem ab_eq_zero (a b : ℤ) (h : ∀ m n : ℕ, ∃ k : ℤ, a * (m^2 : ℤ) + b * (n^2 : ℤ) = k^2) : a * b = 0 :=
by
  sorry

end ab_eq_zero_l210_210057


namespace inequality_am_gm_l210_210810

theorem inequality_am_gm 
  (a b c d : ℝ) 
  (h_nonneg_a : 0 ≤ a) 
  (h_nonneg_b : 0 ≤ b) 
  (h_nonneg_c : 0 ≤ c) 
  (h_nonneg_d : 0 ≤ d) 
  (h_sum : a * b + b * c + c * d + d * a = 1) : 
  (a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (d + a + b) + d^3 / (a + b + c)) ≥ 1 / 3 :=
by
  sorry


end inequality_am_gm_l210_210810


namespace tank_capacity_l210_210633

-- Define the initial fullness of the tank and the total capacity
def initial_fullness (w c : ℝ) : Prop :=
  w = c / 5

-- Define the fullness of the tank after adding 5 liters
def fullness_after_adding (w c : ℝ) : Prop :=
  (w + 5) / c = 2 / 7

-- The main theorem: if both conditions hold, c must equal to 35/3
theorem tank_capacity (w c : ℝ) (h1 : initial_fullness w c) (h2 : fullness_after_adding w c) : 
  c = 35 / 3 :=
sorry

end tank_capacity_l210_210633


namespace total_pictures_l210_210482

theorem total_pictures :
  let Randy_pictures := 5
  let Peter_pictures := Randy_pictures + 3
  let Quincy_pictures := Peter_pictures + 20
  let Susan_pictures := 2 * Quincy_pictures - 7
  let Thomas_pictures := Randy_pictures ^ 3
  Randy_pictures + Peter_pictures + Quincy_pictures + Susan_pictures + Thomas_pictures = 215 := by 
    let Randy_pictures := 5
    let Peter_pictures := Randy_pictures + 3
    let Quincy_pictures := Peter_pictures + 20
    let Susan_pictures := 2 * Quincy_pictures - 7
    let Thomas_pictures := Randy_pictures ^ 3
    sorry

end total_pictures_l210_210482


namespace count_valid_three_digit_numbers_l210_210872

theorem count_valid_three_digit_numbers : 
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ 
           (∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ 
                          a ≥ 1 ∧ a ≤ 9 ∧ 
                          b ≥ 0 ∧ b ≤ 9 ∧ 
                          c ≥ 0 ∧ c ≤ 9 ∧ 
                          (a = b ∨ b = c ∨ a = c ∨ 
                           a + b > c ∧ a + c > b ∧ b + c > a)) ∧
           n = 57 := 
sorry

end count_valid_three_digit_numbers_l210_210872


namespace evaluate_expr_at_2_l210_210642

def expr (x : ℝ) : ℝ := (2 * x + 3) * (2 * x - 3) + (x - 2) ^ 2 - 3 * x * (x - 1)

theorem evaluate_expr_at_2 : expr 2 = 1 :=
by
  sorry

end evaluate_expr_at_2_l210_210642


namespace intersection_A_B_range_of_m_l210_210690

-- Step 1: Define sets A, B, and C
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

def B : Set ℝ := {x | -1 < x ∧ x < 3}

def C (m : ℝ) : Set ℝ := {x | m < x ∧ x < 2 * m - 1}

-- Step 2: Lean statements for the proof

-- (1) Prove A ∩ B = {x | 1 < x < 3}
theorem intersection_A_B : (A ∩ B) = {x | 1 < x ∧ x < 3} :=
by
  sorry

-- (2) Prove the range of m such that C ∪ B = B is (-∞, 2]
theorem range_of_m (m : ℝ) : (C m ∪ B = B) ↔ m ≤ 2 :=
by
  sorry

end intersection_A_B_range_of_m_l210_210690


namespace exists_multiple_with_all_digits_l210_210698

theorem exists_multiple_with_all_digits (n : ℕ) :
  ∃ m : ℕ, (m % n = 0) ∧ (∀ d : ℕ, d < 10 → d = 0 ∨ d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5 ∨ d = 6 ∨ d = 7 ∨ d = 8 ∨ d = 9) := 
sorry

end exists_multiple_with_all_digits_l210_210698


namespace hyperbola_perimeter_l210_210667

-- Lean 4 statement
theorem hyperbola_perimeter (a b m : ℝ) (h1 : a > 0) (h2 : b > 0)
  (F1 F2 : ℝ × ℝ) (A B : ℝ × ℝ)
  (hyperbola_eq : ∀ (x y : ℝ), (x,y) ∈ {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1})
  (line_through_F1 : ∀ (x y : ℝ), x = F1.1)
  (A_B_on_hyperbola : (A.1^2/a^2 - A.2^2/b^2 = 1) ∧ (B.1^2/a^2 - B.2^2/b^2 = 1))
  (dist_AB : dist A B = m)
  (dist_relations : dist A F2 + dist B F2 - (dist A F1 + dist B F1) = 4 * a) : 
  dist A F2 + dist B F2 + dist A B = 4 * a + 2 * m :=
sorry

end hyperbola_perimeter_l210_210667


namespace terminal_side_in_fourth_quadrant_l210_210215

theorem terminal_side_in_fourth_quadrant 
  (h_sin_half : Real.sin (α / 2) = 3 / 5)
  (h_cos_half : Real.cos (α / 2) = -4 / 5) : 
  (Real.sin α < 0) ∧ (Real.cos α > 0) :=
by
  sorry

end terminal_side_in_fourth_quadrant_l210_210215


namespace cost_of_pen_is_30_l210_210459

noncomputable def mean_expenditure_per_day : ℕ := 500
noncomputable def days_in_week : ℕ := 7
noncomputable def total_expenditure : ℕ := mean_expenditure_per_day * days_in_week

noncomputable def mon_expenditure : ℕ := 450
noncomputable def tue_expenditure : ℕ := 600
noncomputable def wed_expenditure : ℕ := 400
noncomputable def thurs_expenditure : ℕ := 500
noncomputable def sat_expenditure : ℕ := 550
noncomputable def sun_expenditure : ℕ := 300

noncomputable def fri_notebook_cost : ℕ := 50
noncomputable def fri_earphone_cost : ℕ := 620

noncomputable def total_non_fri_expenditure : ℕ := 
  mon_expenditure + tue_expenditure + wed_expenditure + 
  thurs_expenditure + sat_expenditure + sun_expenditure

noncomputable def fri_expenditure : ℕ := 
  total_expenditure - total_non_fri_expenditure

noncomputable def fri_pen_cost : ℕ := 
  fri_expenditure - (fri_earphone_cost + fri_notebook_cost)

theorem cost_of_pen_is_30 : fri_pen_cost = 30 :=
  sorry

end cost_of_pen_is_30_l210_210459


namespace Amelia_sell_JetBars_l210_210491

theorem Amelia_sell_JetBars (M : ℕ) (h : 2 * M - 16 = 74) : M = 45 := by
  sorry

end Amelia_sell_JetBars_l210_210491


namespace length_of_BD_l210_210992

noncomputable def points_on_circle (A B C D E : Type) (BD AE BC CD : ℝ) (y z : ℝ) : Prop :=
  BC = 4 ∧ CD = 4 ∧ AE = 6 ∧ (0 < y) ∧ (0 < z) ∧ (AE * 2 = y * z) ∧ (8 > y + z)

theorem length_of_BD (A B C D E : Type) (BD AE BC CD : ℝ) (y z : ℝ)
  (h : points_on_circle A B C D E BD AE BC CD y z) : 
  BD = 7 :=
by
  sorry

end length_of_BD_l210_210992


namespace truncatedPyramidVolume_l210_210882

noncomputable def volumeOfTruncatedPyramid (R : ℝ) : ℝ :=
  let h := R * Real.sqrt 3 / 2
  let S_lower := 3 * R^2 * Real.sqrt 3 / 2
  let S_upper := 3 * R^2 * Real.sqrt 3 / 8
  let sqrt_term := Real.sqrt (S_lower * S_upper)
  (1/3) * h * (S_lower + S_upper + sqrt_term)

theorem truncatedPyramidVolume (R : ℝ) (h := R * Real.sqrt 3 / 2)
  (S_lower := 3 * R^2 * Real.sqrt 3 / 2)
  (S_upper := 3 * R^2 * Real.sqrt 3 / 8)
  (V := (1/3) * h * (S_lower + S_upper + Real.sqrt (S_lower * S_upper))) :
  volumeOfTruncatedPyramid R = 21 * R^3 / 16 := by
  sorry

end truncatedPyramidVolume_l210_210882


namespace jordan_rect_width_is_10_l210_210952

def carol_rect_length : ℕ := 5
def carol_rect_width : ℕ := 24
def jordan_rect_length : ℕ := 12

def carol_rect_area : ℕ := carol_rect_length * carol_rect_width
def jordan_rect_width := carol_rect_area / jordan_rect_length

theorem jordan_rect_width_is_10 : jordan_rect_width = 10 :=
by
  sorry

end jordan_rect_width_is_10_l210_210952


namespace largest_even_sum_1988_is_290_l210_210766

theorem largest_even_sum_1988_is_290 (n : ℕ) 
  (h : 14 * n = 1988) : 2 * n + 6 = 290 :=
sorry

end largest_even_sum_1988_is_290_l210_210766


namespace M_intersection_N_l210_210476

-- Definition of sets M and N
def M : Set ℝ := { x | x^2 + 2 * x - 8 < 0 }
def N : Set ℝ := { y | ∃ x : ℝ, y = 2^x }

-- Goal: Prove that M ∩ N = (0, 2)
theorem M_intersection_N :
  M ∩ N = { y | 0 < y ∧ y < 2 } :=
sorry

end M_intersection_N_l210_210476


namespace sum_of_excluded_solutions_l210_210845

noncomputable def P : ℚ := 3
noncomputable def Q : ℚ := 5 / 3
noncomputable def R : ℚ := 25 / 3

theorem sum_of_excluded_solutions :
    (P = 3) ∧
    (Q = 5 / 3) ∧
    (R = 25 / 3) ∧
    (∀ x, (x ≠ -R ∧ x ≠ -10) →
    ((x + Q) * (P * x + 50) / ((x + R) * (x + 10)) = 3)) →
    (-R + -10 = -55 / 3) :=
by
  sorry

end sum_of_excluded_solutions_l210_210845


namespace choose_8_from_16_l210_210071

theorem choose_8_from_16 :
  Nat.choose 16 8 = 12870 :=
sorry

end choose_8_from_16_l210_210071


namespace contradiction_example_l210_210210

theorem contradiction_example (a b c : ℕ) : (¬ (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0)) → (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1) :=
by
  sorry

end contradiction_example_l210_210210


namespace problem_inequality_solution_l210_210317

noncomputable def find_b_and_c (x : ℝ) (b c : ℝ) : Prop :=
  ∀ x, (x > 2 ∨ x < 1) ↔ x^2 + b*x + c > 0

theorem problem_inequality_solution (x : ℝ) :
  find_b_and_c x (-3) 2 ∧ (2*x^2 - 3*x + 1 ≤ 0 ↔ 1/2 ≤ x ∧ x ≤ 1) :=
by
  sorry

end problem_inequality_solution_l210_210317


namespace max_handshakes_without_cycles_l210_210157

open BigOperators

theorem max_handshakes_without_cycles :
  ∀ n : ℕ, n = 20 → ∑ i in Finset.range (n - 1), i = 190 :=
by intros;
   sorry

end max_handshakes_without_cycles_l210_210157


namespace sin_18_eq_sin_18_sin_54_eq_sin_36_sin_72_eq_l210_210490

-- Part 1: Prove that sin 18° = ( √5 - 1 ) / 4
theorem sin_18_eq : Real.sin (Real.pi / 10) = (Real.sqrt 5 - 1) / 4 := sorry

-- Part 2: Given sin 18° = ( √5 - 1 ) / 4, prove that sin 18° * sin 54° = 1 / 4
theorem sin_18_sin_54_eq :
  Real.sin (Real.pi / 10) = (Real.sqrt 5 - 1) / 4 → 
  Real.sin (Real.pi / 10) * Real.sin (3 * Real.pi / 10) = 1 / 4 := sorry

-- Part 3: Given sin 18° = ( √5 - 1 ) / 4, prove that sin 36° * sin 72° = √5 / 4
theorem sin_36_sin_72_eq :
  Real.sin (Real.pi / 10) = (Real.sqrt 5 - 1) / 4 → 
  Real.sin (Real.pi / 5) * Real.sin (2 * Real.pi / 5) = Real.sqrt 5 / 4 := sorry

end sin_18_eq_sin_18_sin_54_eq_sin_36_sin_72_eq_l210_210490


namespace min_value_proof_l210_210829

noncomputable def min_expr_value (x y : ℝ) : ℝ :=
  (1 / (2 * x)) + (1 / y)

theorem min_value_proof (x y : ℝ) (h1 : x + y = 1) (h2 : y > 0) (h3 : x > 0) :
  min_expr_value x y = (3 / 2) + Real.sqrt 2 :=
sorry

end min_value_proof_l210_210829


namespace correct_formulas_l210_210087

noncomputable def S (a x : ℝ) := (a^x - a^(-x)) / 2
noncomputable def C (a x : ℝ) := (a^x + a^(-x)) / 2

variable {a x y : ℝ}

axiom h1 : a > 0
axiom h2 : a ≠ 1

theorem correct_formulas : S a (x + y) = S a x * C a y + C a x * S a y ∧ S a (x - y) = S a x * C a y - C a x * S a y :=
by 
  sorry

end correct_formulas_l210_210087


namespace common_ratio_geom_seq_l210_210770

variable {a : ℕ → ℝ} {q : ℝ}

def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a n = a 0 * q ^ n

theorem common_ratio_geom_seq (h₁ : a 5 = 1) (h₂ : a 8 = 8) (hq : geom_seq a q) : q = 2 :=
by
  sorry

end common_ratio_geom_seq_l210_210770


namespace broken_stick_triangle_probability_l210_210171

noncomputable def probability_of_triangle (x y z : ℕ) : ℚ := sorry

theorem broken_stick_triangle_probability :
  ∀ x y z : ℕ, (x < y + z ∧ y < x + z ∧ z < x + y) → probability_of_triangle x y z = 1 / 4 := 
by
  sorry

end broken_stick_triangle_probability_l210_210171


namespace music_tool_cost_l210_210611

namespace BandCost

def trumpet_cost : ℝ := 149.16
def song_book_cost : ℝ := 4.14
def total_spent : ℝ := 163.28

theorem music_tool_cost : (total_spent - (trumpet_cost + song_book_cost)) = 9.98 :=
by
  sorry

end music_tool_cost_l210_210611


namespace hundreds_digit_even_l210_210719

-- Define the given conditions
def units_digit (n : ℕ) : ℕ := n % 10
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- The main theorem to prove
theorem hundreds_digit_even (x : ℕ) 
  (h1 : units_digit (x*x) = 9) 
  (h2 : tens_digit (x*x) = 0) : ((x*x) / 100) % 2 = 0 :=
  sorry

end hundreds_digit_even_l210_210719


namespace measure_of_each_interior_angle_l210_210920

theorem measure_of_each_interior_angle (n : ℕ) (hn : 3 ≤ n) : 
  ∃ angle : ℝ, angle = (n - 2) * 180 / n :=
by
  sorry

end measure_of_each_interior_angle_l210_210920


namespace decrease_in_average_salary_l210_210724

-- Define the conditions
variable (I : ℕ := 20)
variable (L : ℕ := 10)
variable (initial_wage_illiterate : ℕ := 25)
variable (new_wage_illiterate : ℕ := 10)

-- Define the theorem statement
theorem decrease_in_average_salary :
  (I * (initial_wage_illiterate - new_wage_illiterate)) / (I + L) = 10 := by
  sorry

end decrease_in_average_salary_l210_210724


namespace find_constants_l210_210011

theorem find_constants (P Q R : ℤ) :
  (∀ x : ℚ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
    (2 * x^2 - 5 * x + 6) / (x^3 - x) = P / x + (Q * x + R) / (x^2 - 1)) →
  P = -6 ∧ Q = 8 ∧ R = -5 :=
by
  sorry

end find_constants_l210_210011


namespace k_value_l210_210291

noncomputable def find_k (AB BC AC BD : ℝ) (h_AB : AB = 3) (h_BC : BC = 4) (h_AC : AC = 5) (h_BD : BD = (12/7) * Real.sqrt 2) : ℝ :=
  12 / 7

theorem k_value (AB BC AC BD : ℝ) (h_AB : AB = 3) (h_BC : BC = 4) (h_AC : AC = 5) (h_BD : BD = (12/7) * Real.sqrt 2) : 
  find_k AB BC AC BD h_AB h_BC h_AC h_BD = 12 / 7 :=
by
  sorry

end k_value_l210_210291


namespace problem1_xy_value_problem2_min_value_l210_210460

-- Define the first problem conditions
def problem1 (x y : ℝ) : Prop :=
  x^2 - 2 * x * y + 2 * y^2 + 6 * y + 9 = 0

-- Prove that xy = 9 given the above condition
theorem problem1_xy_value (x y : ℝ) (h : problem1 x y) : x * y = 9 :=
  sorry

-- Define the second problem conditions
def expression (m : ℝ) : ℝ :=
  m^2 + 6 * m + 13

-- Prove that the minimum value of the expression is 4
theorem problem2_min_value : ∃ m, expression m = 4 :=
  sorry

end problem1_xy_value_problem2_min_value_l210_210460


namespace rectangle_area_l210_210496

theorem rectangle_area (AB AC : ℝ) (AB_eq : AB = 15) (AC_eq : AC = 17) : 
  ∃ (BC : ℝ), (BC^2 = AC^2 - AB^2) ∧ (AB * BC = 120) := 
by
  -- Assuming necessary geometry axioms, such as the definition of a rectangle and Pythagorean theorem.
  sorry

end rectangle_area_l210_210496


namespace relationship_M_N_l210_210999

-- Define the sets M and N based on the conditions
def M : Set ℕ := {x | ∃ n : ℕ, x = 3^n}
def N : Set ℕ := {x | ∃ n : ℕ, x = 3 * n}

-- The statement to be proved
theorem relationship_M_N : ¬ (M ⊆ N) ∧ ¬ (N ⊆ M) :=
by
  sorry

end relationship_M_N_l210_210999


namespace roots_reciprocal_sum_l210_210428

theorem roots_reciprocal_sum (x₁ x₂ : ℝ) 
    (h_roots : x₁ * x₁ + x₁ - 2 = 0 ∧ x₂ * x₂ + x₂ - 2 = 0):
    x₁ ≠ x₂ → (1 / x₁ + 1 / x₂ = 1 / 2) :=
by
  intro h_neq
  sorry

end roots_reciprocal_sum_l210_210428


namespace find_solutions_of_x4_minus_16_l210_210020

noncomputable def solution_set : Set Complex :=
  {2, -2, Complex.I * 2, -Complex.I * 2}

theorem find_solutions_of_x4_minus_16 :
  {x : Complex | x^4 - 16 = 0} = solution_set :=
by
  sorry

end find_solutions_of_x4_minus_16_l210_210020


namespace least_possible_b_l210_210180

noncomputable def a : ℕ := 8

theorem least_possible_b (b : ℕ) (h1 : ∀ n : ℕ, n > 0 → a.factors.count n = 1 → a = n^3)
  (h2 : b.factors.count a = 1)
  (h3 : b % a = 0) :
  b = 24 :=
sorry

end least_possible_b_l210_210180


namespace problem_number_eq_7_5_l210_210206

noncomputable def number : ℝ := 7.5

theorem problem_number_eq_7_5 :
  ∃ x : ℝ, x^2 + 100 = (x - 20)^2 ∧ x = number :=
by
  sorry

end problem_number_eq_7_5_l210_210206


namespace noah_left_lights_on_2_hours_l210_210833

-- Define the conditions
def bedroom_light_usage : ℕ := 6
def office_light_usage : ℕ := 3 * bedroom_light_usage
def living_room_light_usage : ℕ := 4 * bedroom_light_usage
def total_energy_used : ℕ := 96
def total_energy_per_hour := bedroom_light_usage + office_light_usage + living_room_light_usage

-- Define the main theorem to prove
theorem noah_left_lights_on_2_hours : total_energy_used / total_energy_per_hour = 2 := by
  sorry

end noah_left_lights_on_2_hours_l210_210833


namespace union_M_N_eq_U_l210_210189

def U : Set Nat := {2, 3, 4, 5, 6, 7}
def M : Set Nat := {3, 4, 5, 7}
def N : Set Nat := {2, 4, 5, 6}

theorem union_M_N_eq_U : M ∪ N = U := 
by {
  -- Proof would go here
  sorry
}

end union_M_N_eq_U_l210_210189


namespace geometric_sequence_properties_l210_210794

theorem geometric_sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) (h1 : ∀ n, S n = 3^n + t) (h2 : ∀ n, S (n + 1) = S n + a (n + 1)) :
  a 2 = 6 ∧ t = -1 :=
by
  sorry

end geometric_sequence_properties_l210_210794


namespace solve_sqrt_eq_l210_210949

theorem solve_sqrt_eq (z : ℚ) (h : Real.sqrt (5 - 4 * z) = 10) : z = -95 / 4 := by
  sorry

end solve_sqrt_eq_l210_210949


namespace sum_of_first_five_terms_l210_210924

noncomputable -- assuming non-computable for general proof involving sums
def arithmetic_sequence_sum (a_n : ℕ → ℤ) := ∃ d m : ℤ, ∀ n : ℕ, a_n = m + n * d

theorem sum_of_first_five_terms 
(a_n : ℕ → ℤ) 
(h_arith : arithmetic_sequence_sum a_n)
(h_cond : a_n 5 + a_n 8 - a_n 10 = 2)
: ((a_n 1) + (a_n 2) + (a_n 3) + (a_n 4) + (a_n 5) = 10) := 
by 
  sorry

end sum_of_first_five_terms_l210_210924


namespace identity_function_uniq_l210_210231

theorem identity_function_uniq (f g h : ℝ → ℝ)
    (hg : ∀ x, g x = x + 1)
    (hh : ∀ x, h x = x^2)
    (H1 : ∀ x, f (g x) = g (f x))
    (H2 : ∀ x, f (h x) = h (f x)) :
  ∀ x, f x = x :=
by
  sorry

end identity_function_uniq_l210_210231


namespace total_volume_of_four_cubes_l210_210538

theorem total_volume_of_four_cubes (s : ℝ) (h_s : s = 5) : 4 * s^3 = 500 :=
by
  sorry

end total_volume_of_four_cubes_l210_210538


namespace similar_triangle_legs_l210_210602

theorem similar_triangle_legs (y : ℝ) 
  (h1 : ∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ a = 15 ∧ b = 12)
  (h2 : ∃ u v w : ℝ, u^2 + v^2 = w^2 ∧ u = y ∧ v = 9) 
  (h3 : ∀ (a b c u v w : ℝ), (a^2 + b^2 = c^2 ∧ u^2 + v^2 = w^2 ∧ a/u = b/v) → (a = b → u = v)) 
  : y = 11.25 := 
  by 
    sorry

end similar_triangle_legs_l210_210602


namespace tom_total_calories_l210_210363

-- Define the conditions
def c_weight : ℕ := 1
def c_calories_per_pound : ℕ := 51
def b_weight : ℕ := 2 * c_weight
def b_calories_per_pound : ℕ := c_calories_per_pound / 3

-- Define the total calories
def total_calories : ℕ := (c_weight * c_calories_per_pound) + (b_weight * b_calories_per_pound)

-- Prove the total calories Tom eats
theorem tom_total_calories : total_calories = 85 := by
  sorry

end tom_total_calories_l210_210363


namespace speed_of_first_boy_l210_210636

-- Variables for speeds and time
variables (v : ℝ) (t : ℝ) (d : ℝ)

-- Given conditions
def initial_conditions := 
  v > 0 ∧ 
  7.5 > 0 ∧ 
  t = 10 ∧ 
  d = 20

-- Theorem statement with the conditions and the expected answer
theorem speed_of_first_boy
  (h : initial_conditions v t d) : 
  v = 9.5 :=
sorry

end speed_of_first_boy_l210_210636


namespace stratified_sampling_l210_210433

noncomputable def combination (n k : ℕ) : ℕ := Nat.choose n k

theorem stratified_sampling :
  let junior_students := 400
  let senior_students := 200
  let total_sample_size := 60
  let junior_sample_size := (2 * total_sample_size) / 3
  let senior_sample_size := total_sample_size / 3
  combination junior_students junior_sample_size * combination senior_students senior_sample_size =
    combination 400 40 * combination 200 20 :=
by
  let junior_students := 400
  let senior_students := 200
  let total_sample_size := 60
  let junior_sample_size := (2 * total_sample_size) / 3
  let senior_sample_size := total_sample_size / 3
  exact sorry

end stratified_sampling_l210_210433


namespace fraction_of_oranges_is_correct_l210_210644

variable (O P A : ℕ)
variable (total_fruit : ℕ := 56)

theorem fraction_of_oranges_is_correct:
  (A = 35) →
  (P = O / 2) →
  (A = 5 * P) →
  (O + P + A = total_fruit) →
  (O / total_fruit = 1 / 4) :=
by
  -- proof to be filled in 
  sorry

end fraction_of_oranges_is_correct_l210_210644


namespace obtuse_triangle_l210_210213

variable (A B C : ℝ)
variable (angle_sum : A + B + C = 180)
variable (cond1 : A + B = 141)
variable (cond2 : B + C = 165)

theorem obtuse_triangle : B > 90 :=
by
  sorry

end obtuse_triangle_l210_210213


namespace diet_cola_cost_l210_210803

theorem diet_cola_cost (T C : ℝ) 
  (h1 : T + 6 + C = 2 * T)
  (h2 : (T + 6 + C) + T = 24) : C = 2 := 
sorry

end diet_cola_cost_l210_210803


namespace min_value_is_neg2032188_l210_210721

noncomputable def min_expression_value (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_neq: x ≠ y) (h_cond: x + y + 1/x + 1/y = 2022) : ℝ :=
(x + 1/y) * (x + 1/y - 2016) + (y + 1/x) * (y + 1/x - 2016)

theorem min_value_is_neg2032188 (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_neq: x ≠ y) (h_cond: x + y + 1/x + 1/y = 2022) :
  min_expression_value x y h_pos_x h_pos_y h_neq h_cond = -2032188 := 
sorry

end min_value_is_neg2032188_l210_210721


namespace monotonic_function_a_ge_one_l210_210622

theorem monotonic_function_a_ge_one (a : ℝ) :
  (∀ x : ℝ, (x^2 + 2 * x + a) ≥ 0) → a ≥ 1 :=
by
  intros h
  sorry

end monotonic_function_a_ge_one_l210_210622


namespace marbles_lost_l210_210864

theorem marbles_lost (m_initial m_current : ℕ) (h_initial : m_initial = 19) (h_current : m_current = 8) : m_initial - m_current = 11 :=
by {
  sorry
}

end marbles_lost_l210_210864


namespace pencils_lost_l210_210316

theorem pencils_lost (bought_pencils remaining_pencils lost_pencils : ℕ)
                     (h1 : bought_pencils = 16)
                     (h2 : remaining_pencils = 8)
                     (h3 : lost_pencils = bought_pencils - remaining_pencils) :
                     lost_pencils = 8 :=
by {
  sorry
}

end pencils_lost_l210_210316


namespace remainder_of_4000th_term_l210_210757

def sequence_term_position (n : ℕ) : ℕ :=
  n^2

def sum_of_squares_up_to (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

theorem remainder_of_4000th_term : 
  ∃ n : ℕ, sum_of_squares_up_to n ≥ 4000 ∧ (n-1) * n * (2 * (n-1) + 1) / 6 < 4000 ∧ (n % 7) = 1 :=
by 
  sorry

end remainder_of_4000th_term_l210_210757


namespace no_repetition_five_digit_count_l210_210402

theorem no_repetition_five_digit_count (digits : Finset ℕ) (count : Nat) :
  digits = {0, 1, 2, 3, 4, 5} →
  (∀ n ∈ digits, 0 ≤ n ∧ n ≤ 5) →
  (∃ numbers : Finset ℕ, 
    (∀ x ∈ numbers, (x / 100) % 10 ≠ 3 ∧ x % 5 = 0 ∧ x < 100000 ∧ x ≥ 10000) ∧
    (numbers.card = count)) →
  count = 174 :=
by
  sorry

end no_repetition_five_digit_count_l210_210402


namespace calculate_expression_l210_210194

theorem calculate_expression : (3^5 * 4^5) / 6^5 = 32 := 
by
  sorry

end calculate_expression_l210_210194


namespace slope_intercept_equivalence_l210_210890

-- Define the given equation in Lean
def given_line_equation (x y : ℝ) : Prop := 3 * x - 2 * y = 4

-- Define the slope-intercept form as extracted from the given line equation
def slope_intercept_form (x y : ℝ) : Prop := y = (3/2) * x - 2

-- Prove that the given line equation is equivalent to its slope-intercept form
theorem slope_intercept_equivalence (x y : ℝ) :
  given_line_equation x y ↔ slope_intercept_form x y :=
by sorry

end slope_intercept_equivalence_l210_210890


namespace find_a2_l210_210164

-- Definitions from the conditions
def is_geometric_sequence (a : ℕ → ℕ) (q : ℕ) := ∀ n, a (n + 1) = q * a n
def sum_geom_seq (a : ℕ → ℕ) (q : ℕ) (n : ℕ) := (a 0 * (1 - q^(n + 1))) / (1 - q)

-- Given conditions
def a_n : ℕ → ℕ := sorry -- Define the sequence a_n
def q : ℕ := 2
def S_4 := 60

-- The theorem to be proved
theorem find_a2 (h1: is_geometric_sequence a_n q)
                (h2: sum_geom_seq a_n q 3 = S_4) : 
                a_n 1 = 8 :=
sorry

end find_a2_l210_210164


namespace arithmetic_seq_solution_l210_210597

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}

-- Definition of arithmetic sequence
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Sum of first n terms of arithmetic sequence
def sum_arithmetic_seq (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n + 1) / 2 * (a 0 + a n)

-- Given conditions
def given_conditions (a : ℕ → ℝ) : Prop :=
  a 0 + a 4 + a 8 = 27

-- Main theorem to be proved
theorem arithmetic_seq_solution (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (ha : arithmetic_seq a d)
  (hs : sum_arithmetic_seq S a)
  (h_given : given_conditions a) :
  a 4 = 9 ∧ S 8 = 81 :=
sorry

end arithmetic_seq_solution_l210_210597


namespace find_alpha_l210_210744

-- Given conditions
variables (α β : ℝ)
axiom h1 : α + β = 11
axiom h2 : α * β = 24
axiom h3 : α > β

-- Theorems to prove
theorem find_alpha : α = 8 :=
  sorry

end find_alpha_l210_210744


namespace exists_integer_div_15_sqrt_range_l210_210941

theorem exists_integer_div_15_sqrt_range :
  ∃ n : ℕ, (25^2 ≤ n ∧ n ≤ 26^2) ∧ (n % 15 = 0) :=
by
  sorry

end exists_integer_div_15_sqrt_range_l210_210941


namespace distinct_gcd_numbers_l210_210360

theorem distinct_gcd_numbers (nums : Fin 100 → ℕ) (h_distinct : Function.Injective nums) :
  ¬ ∃ a b c : Fin 100, 
    a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    (nums a + Nat.gcd (nums b) (nums c) = nums b + Nat.gcd (nums a) (nums c)) ∧ 
    (nums b + Nat.gcd (nums a) (nums c) = nums c + Nat.gcd (nums a) (nums b)) := 
sorry

end distinct_gcd_numbers_l210_210360


namespace cube_difference_l210_210767

theorem cube_difference (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 26) : a^3 - b^3 = 124 :=
by sorry

end cube_difference_l210_210767


namespace remaining_gallons_to_fill_tank_l210_210869

-- Define the conditions as constants
def tank_capacity : ℕ := 50
def rate_seconds_per_gallon : ℕ := 20
def time_poured_minutes : ℕ := 6

-- Define the number of gallons poured per minute
def gallons_per_minute : ℕ := 60 / rate_seconds_per_gallon

def gallons_poured (minutes : ℕ) : ℕ :=
  minutes * gallons_per_minute

-- The main statement to prove the remaining gallons needed
theorem remaining_gallons_to_fill_tank : 
  tank_capacity - gallons_poured time_poured_minutes = 32 :=
by
  sorry

end remaining_gallons_to_fill_tank_l210_210869


namespace oliver_january_money_l210_210923

variable (x y z : ℕ)

-- Given conditions
def condition1 := y = x - 4
def condition2 := z = y + 32
def condition3 := z = 61

-- Statement to prove
theorem oliver_january_money (h1 : condition1 x y) (h2 : condition2 y z) (h3 : condition3 z) : x = 33 :=
by
  sorry

end oliver_january_money_l210_210923


namespace angle_ACB_33_l210_210438

noncomputable def triangle_ABC : Type := sorry  -- Define the triangle ABC
noncomputable def ω : Type := sorry  -- Define the circumcircle of ABC
noncomputable def M : Type := sorry  -- Define the midpoint of arc BC not containing A
noncomputable def D : Type := sorry  -- Define the point D such that DM is tangent to ω
def AM_eq_AC : Prop := sorry  -- Define the equality AM = AC
def angle_DMC := (38 : ℝ)  -- Define angle DMC = 38 degrees

theorem angle_ACB_33 (h1 : triangle_ABC) 
                      (h2 : ω) 
                      (h3 : M) 
                      (h4 : D) 
                      (h5 : AM_eq_AC)
                      (h6 : angle_DMC = 38) : ∃ θ, (θ = 33) ∧ (angle_ACB = θ) :=
sorry  -- Proof goes here

end angle_ACB_33_l210_210438


namespace special_collection_books_l210_210537

theorem special_collection_books (initial_books loaned_books returned_percent: ℕ) (loaned_books_value: loaned_books = 55) (returned_percent_value: returned_percent = 80) (initial_books_value: initial_books = 75) :
  initial_books - (loaned_books - (returned_percent * loaned_books / 100)) = 64 := by
  sorry

end special_collection_books_l210_210537


namespace find_number_l210_210397

theorem find_number (N : ℚ) (h : (5 / 6) * N = (5 / 16) * N + 150) : N = 288 := by
  sorry

end find_number_l210_210397


namespace max_edges_intersected_by_plane_l210_210823

theorem max_edges_intersected_by_plane (p : ℕ) (h_pos : p > 0) : ℕ :=
  let vertices := 2 * p
  let base_edges := p
  let lateral_edges := p
  let total_edges := 3 * p
  total_edges

end max_edges_intersected_by_plane_l210_210823


namespace surface_area_is_33_l210_210692

structure TShape where
  vertical_cubes : ℕ -- Number of cubes in the vertical line
  horizontal_cubes : ℕ -- Number of cubes in the horizontal line
  intersection_point : ℕ -- Intersection point in the vertical line
  
def surface_area (t : TShape) : ℕ :=
  let top_and_bottom := 9 + 9
  let side_vertical := (3 + 4) -- 3 for the top cube, 1 each for the other 4 cubes
  let side_horizontal := (4 - 1) * 2 -- each of 4 left and right minus intersection twice
  let intersection := 2
  top_and_bottom + side_vertical + side_horizontal + intersection

theorem surface_area_is_33 (t : TShape) (h1 : t.vertical_cubes = 5) (h2 : t.horizontal_cubes = 5) (h3 : t.intersection_point = 3) : 
  surface_area t = 33 := by
  sorry

end surface_area_is_33_l210_210692


namespace solve_for_x_l210_210214

theorem solve_for_x (x y : ℤ) (h1 : x + y = 10) (h2 : x - y = 18) : x = 14 := by
  sorry

end solve_for_x_l210_210214


namespace impossible_divide_into_three_similar_l210_210474

noncomputable def sqrt2 : ℝ := Real.sqrt 2

def similar (x y : ℝ) : Prop :=
  x ≤ sqrt2 * y

theorem impossible_divide_into_three_similar (N : ℝ) :
  ¬ ∃ (x y z : ℝ), x + y + z = N ∧ similar x y ∧ similar y z ∧ similar x z := 
by
  sorry

end impossible_divide_into_three_similar_l210_210474


namespace line_tangent_ellipse_l210_210024

-- Define the conditions of the problem
def line (m : ℝ) (x y : ℝ) : Prop := y = m * x + 2
def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9

-- Prove the statement about the intersection of the line and ellipse
theorem line_tangent_ellipse (m : ℝ) :
  (∀ x y, line m x y → ellipse x y → x = 0.0 ∧ y = 2.0)
  ↔ m^2 = 1 / 3 :=
sorry

end line_tangent_ellipse_l210_210024


namespace problem1_problem2_l210_210301

-- Definitions
def vec_a : ℝ × ℝ := (1, -3)
def b (m : ℝ) : ℝ × ℝ := (-2, m)
def sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
def dot (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Problem 1: Prove the value of m such that vec_a ⊥ (vec_a - b(m))
theorem problem1 (m : ℝ) (h_perp: dot vec_a (sub vec_a (b m)) = 0) : m = -4 := sorry

-- Problem 2: Prove the value of k such that k * vec_a + b(-4) is parallel to vec_a - b(-4)
def scale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def add (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def parallel (u v : ℝ × ℝ) := ∃ (k : ℝ), scale k u = v

theorem problem2 (k : ℝ) (h_parallel: parallel (add (scale k vec_a) (b (-4))) (sub vec_a (b (-4)))) : k = -1 := sorry

end problem1_problem2_l210_210301


namespace tennis_balls_ordered_l210_210612

variables (W Y : ℕ)
def original_eq (W Y : ℕ) := W = Y
def ratio_condition (W Y : ℕ) := W / (Y + 90) = 8 / 13
def total_tennis_balls (W Y : ℕ) := W + Y = 288

theorem tennis_balls_ordered (W Y : ℕ) (h1 : original_eq W Y) (h2 : ratio_condition W Y) : total_tennis_balls W Y :=
sorry

end tennis_balls_ordered_l210_210612


namespace find_total_children_l210_210603

-- Define conditions as a Lean structure
structure SchoolDistribution where
  B : ℕ     -- Total number of bananas
  C : ℕ     -- Total number of children
  absent : ℕ := 160      -- Number of absent children (constant)
  bananas_per_child : ℕ := 2 -- Bananas per child originally (constant)
  bananas_extra : ℕ := 2      -- Extra bananas given to present children (constant)

-- Define the theorem we want to prove
theorem find_total_children (dist : SchoolDistribution) 
  (h1 : dist.B = 2 * dist.C) 
  (h2 : dist.B = 4 * (dist.C - dist.absent)) :
  dist.C = 320 := by
  sorry

end find_total_children_l210_210603


namespace crushing_load_value_l210_210904

-- Given definitions
def W : ℕ := 3
def T : ℕ := 2
def H : ℕ := 6
def L : ℕ := (30 * W^3 * T^5) / H^3

-- Theorem statement
theorem crushing_load_value :
  L = 120 :=
by {
  -- We provided definitions using the given conditions.
  -- Placeholder for proof is provided
  sorry
}

end crushing_load_value_l210_210904


namespace correct_average_of_corrected_number_l210_210943

theorem correct_average_of_corrected_number (num_list : List ℤ) (wrong_num correct_num : ℤ) (n : ℕ)
  (hn : n = 10)
  (haverage : (num_list.sum / n) = 5)
  (hwrong : wrong_num = 26)
  (hcorrect : correct_num = 36)
  (hnum_list_sum : num_list.sum + correct_num - wrong_num = num_list.sum + 10) :
  (num_list.sum + 10) / n = 6 :=
by
  sorry

end correct_average_of_corrected_number_l210_210943


namespace ratio_of_saramago_readers_l210_210531

theorem ratio_of_saramago_readers 
  (W : ℕ) (S K B N : ℕ)
  (h1 : W = 42)
  (h2 : K = W / 6)
  (h3 : B = 3)
  (h4 : N = (S - B) - 1)
  (h5 : W = (S - B) + (K - B) + B + N) :
  S / W = 1 / 2 :=
by
  sorry

end ratio_of_saramago_readers_l210_210531


namespace profit_per_meter_is_20_l210_210260

-- Define given conditions
def selling_price_total (n : ℕ) (price : ℕ) : ℕ := n * price
def cost_price_per_meter : ℕ := 85
def selling_price_total_85_meters : ℕ := 8925

-- Define the expected profit per meter
def expected_profit_per_meter : ℕ := 20

-- Rewrite the problem statement: Prove that with given conditions the profit per meter is Rs. 20
theorem profit_per_meter_is_20 
  (n : ℕ := 85)
  (sp : ℕ := selling_price_total_85_meters)
  (cp_pm : ℕ := cost_price_per_meter) 
  (expected_profit : ℕ := expected_profit_per_meter) :
  (sp - n * cp_pm) / n = expected_profit :=
by
  sorry

end profit_per_meter_is_20_l210_210260


namespace numerical_identity_l210_210798

theorem numerical_identity :
  1.2008 * 0.2008 * 2.4016 - 1.2008^3 - 1.2008 * 0.2008^2 = -1.2008 :=
by
  -- conditions and definitions based on a) are directly used here
  sorry -- proof is not required as per instructions

end numerical_identity_l210_210798


namespace ratio_of_larger_to_smaller_l210_210170

theorem ratio_of_larger_to_smaller (x y : ℝ) (h1 : x > y) (h2 : x + y = 7 * (x - y)) : x / y = 2 :=
sorry

end ratio_of_larger_to_smaller_l210_210170


namespace cost_of_new_shoes_l210_210172

theorem cost_of_new_shoes :
  ∃ P : ℝ, P = 32 ∧ (P / 2 = 14.50 + 0.10344827586206897 * 14.50) :=
sorry

end cost_of_new_shoes_l210_210172


namespace total_cost_l210_210061

-- Define the given conditions
def total_tickets : Nat := 10
def discounted_tickets : Nat := 4
def full_price : ℝ := 2.00
def discounted_price : ℝ := 1.60

-- Calculation of the total cost Martin spent
theorem total_cost : (discounted_tickets * discounted_price) + ((total_tickets - discounted_tickets) * full_price) = 18.40 := by
  sorry

end total_cost_l210_210061


namespace number_of_unique_triangle_areas_l210_210814

theorem number_of_unique_triangle_areas :
  ∀ (G H I J K L : ℝ) (d₁ d₂ d₃ d₄ : ℝ),
    G ≠ H → H ≠ I → I ≠ J → G ≠ I → G ≠ J →
    H ≠ J →
    G - H = 1 → H - I = 1 → I - J = 2 →
    K - L = 2 →
    d₄ = abs d₃ →
    (d₁ = abs (K - G)) ∨ (d₂ = abs (L - G)) ∨ (d₁ = d₂) →
    ∃ (areas : ℕ), 
    areas = 3 :=
by sorry

end number_of_unique_triangle_areas_l210_210814


namespace patients_before_doubling_l210_210146

theorem patients_before_doubling (C P : ℕ) 
    (h1 : (1 / 4) * C = 13) 
    (h2 : C = 2 * P) : 
    P = 26 := 
sorry

end patients_before_doubling_l210_210146


namespace compute_scalar_dot_product_l210_210086

open Matrix 

def vec1 : Fin 2 → ℤ
| 0 => -2
| 1 => 3

def vec2 : Fin 2 → ℤ
| 0 => 4
| 1 => -5

def dot_product (v1 v2 : Fin 2 → ℤ) : ℤ :=
  (v1 0) * (v2 0) + (v1 1) * (v2 1)

theorem compute_scalar_dot_product :
  3 * dot_product vec1 vec2 = -69 := 
by 
  sorry

end compute_scalar_dot_product_l210_210086


namespace seashells_count_l210_210650

theorem seashells_count (mary_seashells : ℕ) (keith_seashells : ℕ) (cracked_seashells : ℕ) 
  (h_mary : mary_seashells = 2) (h_keith : keith_seashells = 5) (h_cracked : cracked_seashells = 9) :
  (mary_seashells + keith_seashells = 7) ∧ (cracked_seashells > mary_seashells + keith_seashells) → false := 
by {
  sorry
}

end seashells_count_l210_210650


namespace find_rate_l210_210792

noncomputable def SI := 200
noncomputable def P := 800
noncomputable def T := 4

theorem find_rate : ∃ R : ℝ, SI = (P * R * T) / 100 ∧ R = 6.25 :=
by sorry

end find_rate_l210_210792


namespace RahulPlayedMatchesSolver_l210_210439

noncomputable def RahulPlayedMatches (current_average new_average runs_in_today current_matches : ℕ) : ℕ :=
  let total_runs_before := current_average * current_matches
  let total_runs_after := total_runs_before + runs_in_today
  let total_matches_after := current_matches + 1
  total_runs_after / new_average

theorem RahulPlayedMatchesSolver:
  RahulPlayedMatches 52 54 78 12 = 12 :=
by
  sorry

end RahulPlayedMatchesSolver_l210_210439


namespace B_holds_32_l210_210369

variable (x y z : ℝ)

-- Conditions
def condition1 : Prop := x + 1/2 * (y + z) = 90
def condition2 : Prop := y + 1/2 * (x + z) = 70
def condition3 : Prop := z + 1/2 * (x + y) = 56

-- Theorem to prove
theorem B_holds_32 (h1 : condition1 x y z) (h2 : condition2 x y z) (h3 : condition3 x y z) : y = 32 :=
sorry

end B_holds_32_l210_210369


namespace smallest_12_digit_proof_l210_210153

def is_12_digit_number (n : ℕ) : Prop :=
  n >= 10^11 ∧ n < 10^12

def contains_each_digit_0_to_9 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] → d ∈ n.digits 10

def is_divisible_by_36 (n : ℕ) : Prop :=
  n % 36 = 0

noncomputable def smallest_12_digit_divisible_by_36_and_contains_each_digit : ℕ :=
  100023457896

theorem smallest_12_digit_proof :
  is_12_digit_number smallest_12_digit_divisible_by_36_and_contains_each_digit ∧
  contains_each_digit_0_to_9 smallest_12_digit_divisible_by_36_and_contains_each_digit ∧
  is_divisible_by_36 smallest_12_digit_divisible_by_36_and_contains_each_digit ∧
  ∀ m : ℕ, is_12_digit_number m ∧ contains_each_digit_0_to_9 m ∧ is_divisible_by_36 m →
  m >= smallest_12_digit_divisible_by_36_and_contains_each_digit :=
by
  sorry

end smallest_12_digit_proof_l210_210153


namespace find_integer_l210_210300

theorem find_integer (a b c d : ℕ) (h1 : a + b + c + d = 18) 
  (h2 : b + c = 11) (h3 : a - d = 3) (h4 : (10^3 * a + 10^2 * b + 10 * c + d) % 9 = 0) :
  10^3 * a + 10^2 * b + 10 * c + d = 5262 ∨ 10^3 * a + 10^2 * b + 10 * c + d = 5622 := 
by
  sorry

end find_integer_l210_210300


namespace conic_section_is_ellipse_l210_210591

open Real

def is_conic_section_ellipse (x y : ℝ) (k : ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  sqrt ((x - p1.1) ^ 2 + (y - p1.2) ^ 2) + sqrt ((x - p2.1) ^ 2 + (y - p2.2) ^ 2) = k

theorem conic_section_is_ellipse :
  is_conic_section_ellipse 2 (-2) 12 (2, -2) (-3, 5) :=
by
  sorry

end conic_section_is_ellipse_l210_210591


namespace fido_yard_area_fraction_l210_210470

theorem fido_yard_area_fraction (r : ℝ) (h : r > 0) :
  let square_area := (2 * r)^2
  let reachable_area := π * r^2
  let fraction := reachable_area / square_area
  ∃ a b : ℕ, (fraction = (Real.sqrt a) / b * π) ∧ (a * b = 4) := by
  sorry

end fido_yard_area_fraction_l210_210470


namespace range_of_a_l210_210942

theorem range_of_a (a : ℝ) : (∃ x : ℝ, (1 < x ∧ x < 4) ∧ (2 * x^2 - 9 * x + a < 0)) ↔ (a < 4) :=
by
  sorry

end range_of_a_l210_210942


namespace negation_of_universal_proposition_l210_210191

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ (∃ x_0 : ℝ, x_0^2 < 0) := sorry

end negation_of_universal_proposition_l210_210191


namespace roots_squared_sum_l210_210350

theorem roots_squared_sum (a b : ℝ) (h : a^2 - 8 * a + 8 = 0 ∧ b^2 - 8 * b + 8 = 0) : a^2 + b^2 = 48 := 
sorry

end roots_squared_sum_l210_210350


namespace positive_solutions_count_l210_210421

theorem positive_solutions_count :
  ∃ n : ℕ, n = 9 ∧
  (∀ (x y : ℕ), 5 * x + 10 * y = 100 → 0 < x ∧ 0 < y → (∃ k : ℕ, k < 10 ∧ n = 9)) :=
sorry

end positive_solutions_count_l210_210421


namespace leah_earned_initially_l210_210022

noncomputable def initial_money (x : ℝ) : Prop :=
  let amount_after_milkshake := (6 / 7) * x
  let amount_left_wallet := (3 / 7) * x
  amount_left_wallet = 12

theorem leah_earned_initially (x : ℝ) (h : initial_money x) : x = 28 :=
by
  sorry

end leah_earned_initially_l210_210022


namespace father_cannot_see_boy_more_than_half_time_l210_210902

def speed_boy := 10 -- speed in km/h
def speed_father := 5 -- speed in km/h

def cannot_see_boy_more_than_half_time (school_perimeter : ℝ) : Prop :=
  ¬(∃ T : ℝ, T > school_perimeter / (2 * speed_boy) ∧ T < school_perimeter / speed_boy)

theorem father_cannot_see_boy_more_than_half_time (school_perimeter : ℝ) (h_school_perimeter : school_perimeter > 0) :
  cannot_see_boy_more_than_half_time school_perimeter :=
by
  sorry

end father_cannot_see_boy_more_than_half_time_l210_210902


namespace cooper_remaining_pies_l210_210486

def total_pies (pies_per_day : ℕ) (days : ℕ) : ℕ := pies_per_day * days

def remaining_pies (total : ℕ) (eaten : ℕ) : ℕ := total - eaten

theorem cooper_remaining_pies :
  remaining_pies (total_pies 7 12) 50 = 34 :=
by sorry

end cooper_remaining_pies_l210_210486


namespace minimum_dimes_l210_210745

-- Given amounts in dollars
def value_of_dimes (n : ℕ) : ℝ := 0.10 * n
def value_of_nickels : ℝ := 0.50
def value_of_one_dollar_bill : ℝ := 1.0
def value_of_four_tens : ℝ := 40.0
def price_of_scarf : ℝ := 42.85

-- Prove the total value of the money is at least the price of the scarf implies n >= 14
theorem minimum_dimes (n : ℕ) :
  value_of_four_tens + value_of_one_dollar_bill + value_of_nickels + value_of_dimes n ≥ price_of_scarf → n ≥ 14 :=
by
  sorry

end minimum_dimes_l210_210745


namespace determine_x_l210_210575

theorem determine_x (A B C : ℝ) (x : ℝ) (h1 : C > B) (h2 : B > A) (h3 : A > 0)
  (h4 : A = B - (x / 100) * B) (h5 : C = A + 2 * B) :
  x = 100 * ((B - A) / B) :=
sorry

end determine_x_l210_210575


namespace amount_C_l210_210286

theorem amount_C (A B C : ℕ) 
  (h₁ : A + B + C = 900) 
  (h₂ : A + C = 400) 
  (h₃ : B + C = 750) : 
  C = 250 :=
sorry

end amount_C_l210_210286


namespace part1_part2_l210_210733

noncomputable def f (x : ℝ) : ℝ := 1 + 2 * Real.log x

theorem part1 (x : ℝ) (hx : x > 0) : f x ≤ x^2 :=
sorry

theorem part2 (x : ℝ) (hx : x > 0) (c : ℝ) (hc : c ≥ -1) : f x ≤ 2 * x + c :=
sorry

end part1_part2_l210_210733


namespace income_to_expenditure_ratio_l210_210908

theorem income_to_expenditure_ratio (I E S : ℕ) (hI : I = 15000) (hS : S = 7000) (hSavings : S = I - E) :
  I / E = 15 / 8 := by
  -- Lean proof goes here
  sorry

end income_to_expenditure_ratio_l210_210908


namespace adam_initial_books_l210_210016

theorem adam_initial_books (B : ℕ) (h1 : B - 11 + 23 = 45) : B = 33 := 
by
  sorry

end adam_initial_books_l210_210016


namespace solve_for_x_l210_210018

theorem solve_for_x : 
  ∃ x : ℝ, 7 * (4 * x + 3) - 5 = -3 * (2 - 8 * x) + 1 / 2 ∧ x = -5.375 :=
by
  sorry

end solve_for_x_l210_210018


namespace find_vector_l210_210731

def line_r (t : ℝ) : ℝ × ℝ :=
  (2 + 5 * t, 3 - 2 * t)

def line_s (u : ℝ) : ℝ × ℝ :=
  (1 + 5 * u, -2 - 2 * u)

def is_projection (w1 w2 : ℝ) : Prop :=
  w1 - w2 = 3

theorem find_vector (w1 w2 : ℝ) (h_proj : is_projection w1 w2) :
  (w1, w2) = (-2, -5) :=
sorry

end find_vector_l210_210731


namespace partition_weights_l210_210858

theorem partition_weights :
  ∃ A B C : Finset ℕ,
    (∀ x ∈ A, x ≤ 552) ∧
    (∀ x ∈ B, x ≤ 552) ∧
    (∀ x ∈ C, x ≤ 552) ∧
    ∀ x, (x ∈ A ∨ x ∈ B ∨ x ∈ C) ↔ 1 ≤ x ∧ x ≤ 552 ∧
    A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ A ∩ C = ∅ ∧
    A.sum id = 50876 ∧ B.sum id = 50876 ∧ C.sum id = 50876 :=
by
  sorry

end partition_weights_l210_210858


namespace min_value_expr_l210_210276

noncomputable def find_min_value (a b c d : ℝ) (x y : ℝ) : ℝ :=
  x / c^2 + y^2 / d^2

theorem min_value_expr (a b c d : ℝ) (h : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ) :
  ∃ x y : ℝ, find_min_value a b c d x y = -abs a / c^2 := 
sorry

end min_value_expr_l210_210276


namespace number_of_integers_l210_210396

theorem number_of_integers (n : ℤ) : 
    25 < n^2 ∧ n^2 < 144 → ∃ l, l = 12 :=
by
  sorry

end number_of_integers_l210_210396


namespace moles_of_NH4Cl_combined_l210_210649

-- Define the chemical reaction equation
def reaction (NH4Cl H2O NH4OH HCl : ℕ) := 
  NH4Cl + H2O = NH4OH + HCl

-- Given conditions
def condition1 (H2O : ℕ) := H2O = 1
def condition2 (NH4OH : ℕ) := NH4OH = 1

-- Theorem statement: Prove that number of moles of NH4Cl combined is 1
theorem moles_of_NH4Cl_combined (H2O NH4OH NH4Cl HCl : ℕ) 
  (h1: condition1 H2O) (h2: condition2 NH4OH) (h3: reaction NH4Cl H2O NH4OH HCl) : 
  NH4Cl = 1 :=
sorry

end moles_of_NH4Cl_combined_l210_210649


namespace jenny_ate_more_than_thrice_mike_l210_210266

theorem jenny_ate_more_than_thrice_mike :
  let mike_ate := 20
  let jenny_ate := 65
  jenny_ate - 3 * mike_ate = 5 :=
by
  let mike_ate := 20
  let jenny_ate := 65
  have : jenny_ate - 3 * mike_ate = 5 := by
    sorry
  exact this

end jenny_ate_more_than_thrice_mike_l210_210266


namespace B_works_alone_in_24_days_l210_210492

noncomputable def B_completion_days (A B : ℝ) (h1 : A = B) (h2 : (A + B) / 12 = 1) : ℝ :=
24

theorem B_works_alone_in_24_days (A B : ℝ) (h1 : A = B) (h2 : (A + B) / 12 = 1) : 
  B_completion_days A B h1 h2 = 24 :=
sorry

end B_works_alone_in_24_days_l210_210492


namespace trigonometric_identity_l210_210666

open Real 

theorem trigonometric_identity (x y : ℝ) (h₁ : P = x * cos y) (h₂ : Q = x * sin y) : 
  (P + Q) / (P - Q) + (P - Q) / (P + Q) = 2 * cos y / sin y := by 
  sorry

end trigonometric_identity_l210_210666


namespace current_speed_is_one_l210_210918

noncomputable def motorboat_rate_of_current (b h t : ℝ) : ℝ :=
  let eq1 := (b + 1 - h) * 4
  let eq2 := (b - 1 + t) * 6
  if eq1 = 24 ∧ eq2 = 24 then 1 else sorry

theorem current_speed_is_one (b h t : ℝ) : motorboat_rate_of_current b h t = 1 :=
by
  sorry

end current_speed_is_one_l210_210918


namespace cubic_polynomial_roots_l210_210530

noncomputable def polynomial := fun x : ℝ => x^3 - 2*x - 2

theorem cubic_polynomial_roots
  (x y z : ℝ) 
  (h1: polynomial x = 0)
  (h2: polynomial y = 0)
  (h3: polynomial z = 0):
  x * (y - z)^2 + y * (z - x)^2 + z * (x - y)^2 = 0 :=
by
  -- Solution steps will be filled here to prove the theorem
  sorry

end cubic_polynomial_roots_l210_210530


namespace binary_computation_l210_210528

theorem binary_computation :
  (0b101101 * 0b10101 + 0b1010 / 0b10) = 0b110111100000 := by
  sorry

end binary_computation_l210_210528


namespace middle_odd_number_is_26_l210_210915

theorem middle_odd_number_is_26 (x : ℤ) 
  (h : (x - 4) + (x - 2) + x + (x + 2) + (x + 4) = 130) : x = 26 := 
by 
  sorry

end middle_odd_number_is_26_l210_210915


namespace compare_m_n_l210_210378

noncomputable def m (a : ℝ) : ℝ := 6^a / (36^(a + 1) + 1)
noncomputable def n (b : ℝ) : ℝ := (1/3) * b^2 - b + (5/6)

theorem compare_m_n (a b : ℝ) : m a ≤ n b := sorry

end compare_m_n_l210_210378


namespace counties_percentage_l210_210866

theorem counties_percentage (a b c : ℝ) (ha : a = 0.2) (hb : b = 0.35) (hc : c = 0.25) :
  a + b + c = 0.8 :=
by
  rw [ha, hb, hc]
  sorry

end counties_percentage_l210_210866


namespace batsman_average_after_12th_innings_l210_210338

theorem batsman_average_after_12th_innings (A : ℤ) :
  (∀ A : ℤ, (11 * A + 60 = 12 * (A + 2))) → (A = 36) → (A + 2 = 38) := 
by
  intro h_avg_increase h_init_avg
  sorry

end batsman_average_after_12th_innings_l210_210338


namespace joan_football_games_l210_210096

theorem joan_football_games (G_total G_last G_this : ℕ) (h1 : G_total = 13) (h2 : G_last = 9) (h3 : G_this = G_total - G_last) : G_this = 4 :=
by
  sorry

end joan_football_games_l210_210096


namespace triangle_inradius_l210_210795

theorem triangle_inradius (A p r : ℝ) 
    (h1 : p = 35) 
    (h2 : A = 78.75) 
    (h3 : A = (r * p) / 2) : 
    r = 4.5 :=
sorry

end triangle_inradius_l210_210795


namespace exp_product_correct_l210_210413

def exp_1 := (2 : ℕ) ^ 4
def exp_2 := (3 : ℕ) ^ 2
def exp_3 := (5 : ℕ) ^ 2
def exp_4 := (7 : ℕ)
def exp_5 := (11 : ℕ)
def final_value := exp_1 * exp_2 * exp_3 * exp_4 * exp_5

theorem exp_product_correct : final_value = 277200 := by
  sorry

end exp_product_correct_l210_210413


namespace number_of_cats_l210_210708

-- Defining the context and conditions
variables (x y z : Nat)
variables (h1 : x + y + z = 29) (h2 : x = z)

-- Proving the number of cats
theorem number_of_cats (x y z : Nat) (h1 : x + y + z = 29) (h2 : x = z) :
  6 * x + 3 * y = 87 := by
  sorry

end number_of_cats_l210_210708


namespace evaluate_expression_l210_210307

theorem evaluate_expression (x : ℝ) : (x+2)^2 + 2*(x+2)*(4-x) + (4-x)^2 = 36 :=
by sorry

end evaluate_expression_l210_210307


namespace solve_polynomial_l210_210372

theorem solve_polynomial (z : ℂ) : z^6 - 9 * z^3 + 8 = 0 ↔ z = 1 ∨ z = 2 := 
by
  sorry

end solve_polynomial_l210_210372


namespace perimeter_original_square_l210_210982

theorem perimeter_original_square (s : ℝ) (h1 : (3 / 4) * s^2 = 48) : 4 * s = 32 :=
by
  sorry

end perimeter_original_square_l210_210982


namespace max_sum_is_38_l210_210616

-- Definition of the problem variables and conditions
def number_set : Set ℤ := {2, 3, 8, 9, 14, 15}
variable (a b c d e : ℤ)

-- Conditions translated to Lean
def condition1 : Prop := b = c
def condition2 : Prop := a = d

-- Sum condition to find maximum sum
def max_combined_sum : ℤ := a + b + e

theorem max_sum_is_38 : 
  ∃ a b c d e, 
    {a, b, c, d, e} ⊆ number_set ∧
    b = c ∧ 
    a = d ∧ 
    a + b + e = 38 :=
sorry

end max_sum_is_38_l210_210616


namespace min_value_expression_eq_2sqrt3_l210_210000

noncomputable def min_value_expression (c d : ℝ) : ℝ :=
  c^2 + d^2 + 4 / c^2 + 2 * d / c

theorem min_value_expression_eq_2sqrt3 (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ y : ℝ, (∀ d : ℝ, min_value_expression c d ≥ y) ∧ y = 2 * Real.sqrt 3 :=
sorry

end min_value_expression_eq_2sqrt3_l210_210000


namespace billy_avoids_swimming_n_eq_2022_billy_wins_for_odd_n_billy_wins_for_even_n_l210_210304

theorem billy_avoids_swimming_n_eq_2022 :
  ∀ n : ℕ, n = 2022 → (∃ (strategy : ℕ → ℕ), ∀ k, strategy (2022 + 1 - k) ≠ strategy (k + 1)) :=
by
  sorry

theorem billy_wins_for_odd_n (n : ℕ) (h : n > 10 ∧ n % 2 = 1) :
  ∃ (strategy : ℕ → ℕ), (∀ k, strategy (n + 1 - k) ≠ strategy (k + 1)) :=
by
  sorry

theorem billy_wins_for_even_n (n : ℕ) (h : n > 10 ∧ n % 2 = 0) :
  ∃ (strategy : ℕ → ℕ), (∀ k, strategy (n + 1 - k) ≠ strategy (k + 1)) :=
by
  sorry

end billy_avoids_swimming_n_eq_2022_billy_wins_for_odd_n_billy_wins_for_even_n_l210_210304


namespace electronics_weight_l210_210368

theorem electronics_weight (B C E : ℝ) (h1 : B / C = 5 / 4) (h2 : B / E = 5 / 2) (h3 : B / (C - 9) = 10 / 4) : E = 9 := 
by 
  sorry

end electronics_weight_l210_210368


namespace ribbon_left_after_wrapping_l210_210125

def total_ribbon_needed (gifts : ℕ) (ribbon_per_gift : ℝ) : ℝ :=
  gifts * ribbon_per_gift

def remaining_ribbon (initial_ribbon : ℝ) (used_ribbon : ℝ) : ℝ :=
  initial_ribbon - used_ribbon

theorem ribbon_left_after_wrapping : 
  ∀ (gifts : ℕ) (ribbon_per_gift initial_ribbon : ℝ),
  gifts = 8 →
  ribbon_per_gift = 1.5 →
  initial_ribbon = 15 →
  remaining_ribbon initial_ribbon (total_ribbon_needed gifts ribbon_per_gift) = 3 :=
by
  intros gifts ribbon_per_gift initial_ribbon h1 h2 h3
  rw [h1, h2, h3]
  simp [total_ribbon_needed, remaining_ribbon]
  sorry

end ribbon_left_after_wrapping_l210_210125


namespace daily_reading_goal_l210_210258

-- Define the constants for pages read each day
def pages_on_sunday : ℕ := 43
def pages_on_monday : ℕ := 65
def pages_on_tuesday : ℕ := 28
def pages_on_wednesday : ℕ := 0
def pages_on_thursday : ℕ := 70
def pages_on_friday : ℕ := 56
def pages_on_saturday : ℕ := 88

-- Define the total pages read in the week
def total_pages := pages_on_sunday + pages_on_monday + pages_on_tuesday + pages_on_wednesday 
                    + pages_on_thursday + pages_on_friday + pages_on_saturday

-- The theorem that expresses Berry's daily reading goal
theorem daily_reading_goal : total_pages / 7 = 50 :=
by
  sorry

end daily_reading_goal_l210_210258


namespace ratio_p_q_l210_210092

section ProbabilityProof

-- Definitions and constants as per conditions
def N := Nat.factorial 15

def num_ways_A : ℕ := 4 * (Nat.factorial 9 / (Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 3))
def num_ways_B : ℕ := 4 * 3

def p : ℚ := num_ways_A / N
def q : ℚ := num_ways_B / N

-- Theorem: Prove that the ratio p/q is 560
theorem ratio_p_q : p / q = 560 := by
  sorry

end ProbabilityProof

end ratio_p_q_l210_210092


namespace smallest_number_divisible_by_20_and_36_l210_210624

-- Define the conditions that x must be divisible by both 20 and 36
def divisible_by (x n : ℕ) : Prop := ∃ m : ℕ, x = n * m

-- Define the problem statement
theorem smallest_number_divisible_by_20_and_36 : 
  ∃ x : ℕ, divisible_by x 20 ∧ divisible_by x 36 ∧ 
  (∀ y : ℕ, (divisible_by y 20 ∧ divisible_by y 36) → y ≥ x) ∧ x = 180 := 
by
  sorry

end smallest_number_divisible_by_20_and_36_l210_210624


namespace cubed_identity_l210_210458

theorem cubed_identity (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := 
sorry

end cubed_identity_l210_210458


namespace asymptote_equation_of_hyperbola_l210_210424

def hyperbola_eccentricity (a : ℝ) (h : a > 0) : Prop :=
  let e := Real.sqrt 2
  e = Real.sqrt (1 + a^2) / a

theorem asymptote_equation_of_hyperbola :
  ∀ (a : ℝ) (h : a > 0), hyperbola_eccentricity a h → (∀ x y : ℝ, (x^2 - y^2 = 1 → y = x ∨ y = -x)) :=
by
  intro a h he
  sorry

end asymptote_equation_of_hyperbola_l210_210424


namespace max_value_a_n_l210_210008

noncomputable def a_seq : ℕ → ℕ
| 0     => 0  -- By Lean's 0-based indexing, a_1 corresponds to a_seq 1
| 1     => 3
| (n+2) => a_seq (n+1) + 1

def S_n (n : ℕ) : ℕ := (n * (n + 5)) / 2

theorem max_value_a_n : 
  ∃ n : ℕ, S_n n = 2023 ∧ a_seq n = 73 :=
by
  sorry

end max_value_a_n_l210_210008


namespace supplement_of_supplement_l210_210487

def supplement (angle : ℝ) : ℝ :=
  180 - angle

theorem supplement_of_supplement (θ : ℝ) (h : θ = 35) : supplement (supplement θ) = 35 := by
  -- It is enough to state the theorem; the proof is not required as per the instruction.
  sorry

end supplement_of_supplement_l210_210487


namespace z_is_233_percent_greater_than_w_l210_210906

theorem z_is_233_percent_greater_than_w
  (w e x y z : ℝ)
  (h1 : w = 0.5 * e)
  (h2 : e = 0.4 * x)
  (h3 : x = 0.3 * y)
  (h4 : z = 0.2 * y) :
  z = 2.3333 * w :=
by
  sorry

end z_is_233_percent_greater_than_w_l210_210906


namespace range_of_m_l210_210680

noncomputable def has_two_solutions (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 = x₁ + m ∧ x₂^2 = x₂ + m 

theorem range_of_m (m : ℝ) : has_two_solutions m ↔ m > -(1/4) :=
sorry

end range_of_m_l210_210680


namespace constant_term_expansion_l210_210314

theorem constant_term_expansion (a : ℝ) (h : (2 + a * x) * (1 + 1/x) ^ 5 = (2 + 5 * a)) : 2 + 5 * a = 12 → a = 2 :=
by
  intro h_eq
  have h_sum : 2 + 5 * a = 12 := h_eq
  sorry

end constant_term_expansion_l210_210314


namespace lewis_speed_l210_210251

theorem lewis_speed
  (v : ℕ)
  (john_speed : ℕ := 40)
  (distance_AB : ℕ := 240)
  (meeting_distance : ℕ := 160)
  (time_john_to_meeting : ℕ := meeting_distance / john_speed)
  (distance_lewis_traveled : ℕ := distance_AB + (distance_AB - meeting_distance))
  (v_eq : v = distance_lewis_traveled / time_john_to_meeting) :
  v = 80 :=
by
  sorry

end lewis_speed_l210_210251


namespace entrance_fee_increase_l210_210390

theorem entrance_fee_increase
  (entrance_fee_under_18 : ℕ)
  (rides_cost : ℕ)
  (num_rides : ℕ)
  (total_spent : ℕ)
  (total_cost_twins : ℕ)
  (total_ride_cost_twins : ℕ)
  (amount_spent_joe : ℕ)
  (total_ride_cost_joe : ℕ)
  (joe_entrance_fee : ℕ)
  (increase : ℕ)
  (percentage_increase : ℕ)
  (h1 : entrance_fee_under_18 = 5)
  (h2 : rides_cost = 50) -- representing $0.50 as 50 cents to maintain integer calculations
  (h3 : num_rides = 3)
  (h4 : total_spent = 2050) -- representing $20.5 as 2050 cents
  (h5 : total_cost_twins = 1300) -- combining entrance fees and cost of rides for the twins in cents
  (h6 : total_ride_cost_twins = 300) -- cost of rides for twins in cents
  (h7 : amount_spent_joe = 750) -- representing $7.5 as 750 cents
  (h8 : total_ride_cost_joe = 150) -- cost of rides for Joe in cents
  (h9 : joe_entrance_fee = 600) -- representing $6 as 600 cents
  (h10 : increase = 100) -- increase in entrance fee in cents
  (h11 : percentage_increase = 20) :
  percentage_increase = ((increase * 100) / entrance_fee_under_18) :=
sorry

end entrance_fee_increase_l210_210390


namespace total_distance_is_27_l210_210738

-- Condition: Renaldo drove 15 kilometers
def renaldo_distance : ℕ := 15

-- Condition: Ernesto drove 7 kilometers more than one-third of Renaldo's distance
def ernesto_distance := (1 / 3 : ℚ) * renaldo_distance + 7

-- Theorem to prove that total distance driven by both men is 27 kilometers
theorem total_distance_is_27 : renaldo_distance + ernesto_distance = 27 := by
  sorry

end total_distance_is_27_l210_210738


namespace product_of_possible_values_N_l210_210077

theorem product_of_possible_values_N 
  (L M : ℤ) 
  (h1 : M = L + N) 
  (h2 : M - 7 = L + N - 7)
  (h3 : L + 5 = L + 5)
  (h4 : |(L + N - 7) - (L + 5)| = 4) : 
  N = 128 := 
  sorry

end product_of_possible_values_N_l210_210077


namespace percent_profit_l210_210100

theorem percent_profit (C S : ℝ) (h : 58 * C = 50 * S) : 
  (S - C) / C * 100 = 16 :=
by
  sorry

end percent_profit_l210_210100


namespace Linda_original_savings_l210_210658

theorem Linda_original_savings (S : ℝ)
  (H1 : 3/4 * S + 1/4 * S = S)
  (H2 : 1/4 * S = 220) :
  S = 880 :=
sorry

end Linda_original_savings_l210_210658


namespace range_of_m_l210_210771

theorem range_of_m :
  (∀ x : ℝ, (x < m - 1 ∨ x > m + 1) → (x < -1 ∨ x > 3)) ↔ (0 ≤ m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l210_210771


namespace comparison_M_N_l210_210387

def M (x : ℝ) : ℝ := x^2 - 3*x + 7
def N (x : ℝ) : ℝ := -x^2 + x + 1

theorem comparison_M_N (x : ℝ) : M x > N x :=
  by sorry

end comparison_M_N_l210_210387


namespace find_number_of_sides_l210_210939

-- Defining the problem conditions
def sum_of_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

-- Statement of the problem
theorem find_number_of_sides (h : sum_of_interior_angles n = 1260) : n = 9 :=
by
  sorry

end find_number_of_sides_l210_210939


namespace runner_time_difference_l210_210582

theorem runner_time_difference (v : ℝ) (h1 : 0 < v) (h2 : 0 < 20 / v) (h3 : 8 = 40 / v) :
  8 - (20 / v) = 4 := by
  sorry

end runner_time_difference_l210_210582


namespace bird_difference_l210_210968

-- Variables representing given conditions
def num_migrating_families : Nat := 86
def num_remaining_families : Nat := 45
def avg_birds_per_migrating_family : Nat := 12
def avg_birds_per_remaining_family : Nat := 8

-- Definition to calculate total number of birds for migrating families
def total_birds_migrating : Nat := num_migrating_families * avg_birds_per_migrating_family

-- Definition to calculate total number of birds for remaining families
def total_birds_remaining : Nat := num_remaining_families * avg_birds_per_remaining_family

-- The statement that we need to prove
theorem bird_difference (h : total_birds_migrating - total_birds_remaining = 672) : 
  total_birds_migrating - total_birds_remaining = 672 := 
sorry

end bird_difference_l210_210968


namespace tan_alpha_minus_pi_div_4_l210_210036

open Real

theorem tan_alpha_minus_pi_div_4 (α : ℝ) (h : (cos α * 2 + (-1) * sin α = 0)) : 
  tan (α - π / 4) = 1 / 3 :=
sorry

end tan_alpha_minus_pi_div_4_l210_210036


namespace vertex_of_parabola_l210_210907

theorem vertex_of_parabola :
  ∀ x : ℝ, (x - 2) ^ 2 + 4 = (x - 2) ^ 2 + 4 → (2, 4) = (2, 4) :=
by
  intro x
  intro h
  -- We know that the vertex of y = (x - 2)^2 + 4 is at (2, 4)
  admit

end vertex_of_parabola_l210_210907


namespace quadratic_equation_with_given_means_l210_210781

theorem quadratic_equation_with_given_means (α β : ℝ)
  (h1 : (α + β) / 2 = 8) 
  (h2 : Real.sqrt (α * β) = 12) : 
  x ^ 2 - 16 * x + 144 = 0 :=
sorry

end quadratic_equation_with_given_means_l210_210781


namespace find_tangent_line_equation_l210_210986

-- Define the curve as a function
def curve (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 4 * x

-- Define the point of tangency
def P : ℝ × ℝ := (-1, 3)

-- Define the slope of the tangent line at point P
def slope_at_P : ℝ := curve_derivative P.1

-- Define the expected equation of the tangent line
def tangent_line (x y : ℝ) : Prop := 4 * x + y + 1 = 0

-- The theorem to prove that the tangent line at point P has the expected equation
theorem find_tangent_line_equation : 
  tangent_line P.1 (curve P.1) :=
  sorry

end find_tangent_line_equation_l210_210986


namespace first_dilution_volume_l210_210167

theorem first_dilution_volume (x : ℝ) (V : ℝ) (red_factor : ℝ) (p : ℝ) :
  V = 1000 →
  red_factor = 25 / 3 →
  (1000 - 2 * x) * (1000 - x) = 1000 * 1000 * (3 / 25) →
  x = 400 :=
by
  intros hV hred hf
  sorry

end first_dilution_volume_l210_210167


namespace correct_calculation_l210_210222

theorem correct_calculation (x : ℝ) : x * x^2 = x^3 :=
by sorry

end correct_calculation_l210_210222


namespace cube_root_eq_self_l210_210768

theorem cube_root_eq_self (a : ℝ) (h : a^(3:ℕ) = a) : a = 1 ∨ a = -1 ∨ a = 0 := 
sorry

end cube_root_eq_self_l210_210768


namespace product_of_two_numbers_l210_210793

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := 
by sorry

end product_of_two_numbers_l210_210793


namespace sqrt_of_4_l210_210117

theorem sqrt_of_4 (y : ℝ) : y^2 = 4 → (y = 2 ∨ y = -2) :=
sorry

end sqrt_of_4_l210_210117


namespace minimize_cost_at_4_l210_210520

-- Given definitions and conditions
def surface_area : ℝ := 12
def max_side_length : ℝ := 5
def front_face_cost_per_sqm : ℝ := 400
def sides_cost_per_sqm : ℝ := 150
def roof_ground_cost : ℝ := 5800
def wall_height : ℝ := 3

-- Definition of the total cost function
noncomputable def total_cost (x : ℝ) : ℝ :=
  900 * (x + 16 / x) + 5800

-- The main theorem to be proven
theorem minimize_cost_at_4 (h : 0 < x ∧ x ≤ max_side_length) : 
  (∀ x, total_cost x ≥ total_cost 4) ∧ total_cost 4 = 13000 :=
sorry

end minimize_cost_at_4_l210_210520


namespace poly_divisibility_implies_C_D_l210_210684

noncomputable def poly_condition : Prop :=
  ∃ (C D : ℤ), ∀ (α : ℂ), α^2 - α + 1 = 0 → α^103 + C * α^2 + D * α + 1 = 0

/- The translated proof problem -/
theorem poly_divisibility_implies_C_D (C D : ℤ) :
  (poly_condition) → (C = -1 ∧ D = 0) :=
by
  intro h
  sorry

end poly_divisibility_implies_C_D_l210_210684


namespace max_marks_400_l210_210131

theorem max_marks_400 {M : ℝ} (h1 : 0.35 * M = 140) : M = 400 :=
by 
-- skipping the proof using sorry
sorry

end max_marks_400_l210_210131


namespace binomial_coeff_sum_l210_210750

theorem binomial_coeff_sum : 
  (Nat.choose 3 2) + (Nat.choose 4 2) + (Nat.choose 5 2) + (Nat.choose 6 2) + (Nat.choose 7 2) + (Nat.choose 8 2) = 83 := by
  sorry

end binomial_coeff_sum_l210_210750


namespace imaginary_part_of_fraction_l210_210723

theorem imaginary_part_of_fraction (i : ℂ) (hi : i * i = -1) : (1 + i) / (1 - i) = 1 :=
by
  -- Skipping the proof
  sorry

end imaginary_part_of_fraction_l210_210723


namespace tan_2alpha_and_cos_beta_l210_210783

theorem tan_2alpha_and_cos_beta
    (α β : ℝ)
    (h1 : 0 < β ∧ β < α ∧ α < (Real.pi / 2))
    (h2 : Real.sin α = (4 * Real.sqrt 3) / 7)
    (h3 : Real.cos (β - α) = 13 / 14) :
    Real.tan (2 * α) = -(8 * Real.sqrt 3) / 47 ∧ Real.cos β = 1 / 2 := by
  sorry

end tan_2alpha_and_cos_beta_l210_210783


namespace layla_more_points_than_nahima_l210_210673

theorem layla_more_points_than_nahima (layla_points : ℕ) (total_points : ℕ) (h1 : layla_points = 70) (h2 : total_points = 112) :
  layla_points - (total_points - layla_points) = 28 :=
by
  sorry

end layla_more_points_than_nahima_l210_210673


namespace original_salary_l210_210911

theorem original_salary (S : ℝ) (h1 : S + 0.10 * S = 1.10 * S) (h2: 1.10 * S - 0.05 * (1.10 * S) = 1.10 * S * 0.95) (h3: 1.10 * S * 0.95 = 2090) : S = 2000 :=
sorry

end original_salary_l210_210911


namespace evaluate_expression_l210_210101

variable (x y : ℝ)

theorem evaluate_expression (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 - y^2 = x * y) :
  (1 / x^2) - (1 / y^2) = - (1 / (x * y)) :=
sorry

end evaluate_expression_l210_210101


namespace Marcus_pretzels_l210_210709

theorem Marcus_pretzels (John_pretzels : ℕ) (Marcus_more_than_John : ℕ) (h1 : John_pretzels = 28) (h2 : Marcus_more_than_John = 12) : Marcus_more_than_John + John_pretzels = 40 :=
by
  sorry

end Marcus_pretzels_l210_210709


namespace count_points_in_intersection_is_7_l210_210775

def isPointInSetA (x y : ℤ) : Prop :=
  (x - 3)^2 + (y - 4)^2 ≤ (5 / 2)^2

def isPointInSetB (x y : ℤ) : Prop :=
  (x - 4)^2 + (y - 5)^2 > (5 / 2)^2

def isPointInIntersection (x y : ℤ) : Prop :=
  isPointInSetA x y ∧ isPointInSetB x y

def pointsInIntersection : List (ℤ × ℤ) :=
  [(1, 5), (1, 4), (1, 3), (2, 3), (3, 2), (3, 3), (3, 4)]

theorem count_points_in_intersection_is_7 :
  (List.length pointsInIntersection = 7)
  ∧ (∀ (p : ℤ × ℤ), p ∈ pointsInIntersection → isPointInIntersection p.fst p.snd) :=
by
  sorry

end count_points_in_intersection_is_7_l210_210775


namespace only_sqrt_three_is_irrational_l210_210877

-- Definitions based on conditions
def zero_rational : Prop := ∃ p q : ℤ, q ≠ 0 ∧ (0 : ℝ) = p / q
def neg_three_rational : Prop := ∃ p q : ℤ, q ≠ 0 ∧ (-3 : ℝ) = p / q
def one_third_rational : Prop := ∃ p q : ℤ, q ≠ 0 ∧ (1/3 : ℝ) = p / q
def sqrt_three_irrational : Prop := ¬ ∃ p q : ℤ, q ≠ 0 ∧ (Real.sqrt 3) = p / q

-- The proof problem statement
theorem only_sqrt_three_is_irrational :
  zero_rational ∧
  neg_three_rational ∧
  one_third_rational ∧
  sqrt_three_irrational :=
by sorry

end only_sqrt_three_is_irrational_l210_210877


namespace sin_2x_value_l210_210954

theorem sin_2x_value (x : ℝ) (h : Real.sin (π / 4 - x) = 1 / 3) : Real.sin (2 * x) = 7 / 9 := by
  sorry

end sin_2x_value_l210_210954


namespace factor_count_l210_210913

theorem factor_count (x : ℤ) : 
  (x^12 - x^3) = x^3 * (x - 1) * (x^2 + x + 1) * (x^6 + x^3 + 1) -> 4 = 4 :=
by
  sorry

end factor_count_l210_210913


namespace floor_sqrt_120_l210_210379

theorem floor_sqrt_120 : (Int.floor (Real.sqrt 120)) = 10 := by
  have h1 : 10^2 < 120 := by norm_num
  have h2 : 120 < 11^2 := by norm_num
  -- Additional steps to show that Int.floor (Real.sqrt 120) = 10
  sorry

end floor_sqrt_120_l210_210379


namespace final_price_correct_l210_210405

variable (original_price first_discount second_discount third_discount sales_tax : ℝ)
variable (final_discounted_price final_price: ℝ)

-- Define original price and discounts
def initial_price : ℝ := 20000
def discount1      : ℝ := 0.12
def discount2      : ℝ := 0.10
def discount3      : ℝ := 0.05
def tax_rate       : ℝ := 0.08

def price_after_first_discount : ℝ := initial_price * (1 - discount1)
def price_after_second_discount : ℝ := price_after_first_discount * (1 - discount2)
def price_after_third_discount : ℝ := price_after_second_discount * (1 - discount3)
def final_sale_price : ℝ := price_after_third_discount * (1 + tax_rate)

-- Prove final sale price is 16251.84
theorem final_price_correct : final_sale_price = 16251.84 := by
  sorry

end final_price_correct_l210_210405


namespace smallest_M_convex_quadrilateral_l210_210243

section ConvexQuadrilateral

-- Let a, b, c, d be the sides of a convex quadrilateral
variables {a b c d M : ℝ}

-- Condition to ensure that a, b, c, d are the sides of a convex quadrilateral
def is_convex_quadrilateral (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a + b + c + d < 360

-- The theorem statement
theorem smallest_M_convex_quadrilateral (hconvex : is_convex_quadrilateral a b c d) : ∃ M, (∀ a b c d, is_convex_quadrilateral a b c d → (a^2 + b^2) / (c^2 + d^2) > M) ∧ M = 1/2 :=
by sorry

end ConvexQuadrilateral

end smallest_M_convex_quadrilateral_l210_210243


namespace max_non_overlapping_areas_l210_210030

theorem max_non_overlapping_areas (n : ℕ) (h : n > 0) : 
  ∃ k : ℕ, k = 4 * n + 4 := 
sorry

end max_non_overlapping_areas_l210_210030


namespace cats_count_l210_210931

-- Definitions based on conditions
def heads_eqn (H C : ℕ) : Prop := H + C = 15
def legs_eqn (H C : ℕ) : Prop := 2 * H + 4 * C = 44

-- The main proof problem
theorem cats_count (H C : ℕ) (h1 : heads_eqn H C) (h2 : legs_eqn H C) : C = 7 :=
by
  sorry

end cats_count_l210_210931


namespace quadratic_roots_sum_product_l210_210897

theorem quadratic_roots_sum_product (m n : ℝ) (h1 : m / 2 = 10) (h2 : n / 2 = 24) : m + n = 68 :=
by
  sorry

end quadratic_roots_sum_product_l210_210897


namespace f_decreasing_f_odd_l210_210762

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (a b : ℝ) : f (a + b) = f a + f b

axiom negativity (x : ℝ) (h_pos : 0 < x) : f x < 0

theorem f_decreasing : ∀ x1 x2 : ℝ, x1 > x2 → f x1 < f x2 :=
by
  intros x1 x2 h
  sorry

theorem f_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  intros x
  sorry

end f_decreasing_f_odd_l210_210762


namespace second_spray_kill_percent_l210_210133

-- Conditions
def first_spray_kill_percent : ℝ := 50
def both_spray_kill_percent : ℝ := 5
def germs_left_after_both : ℝ := 30

-- Lean 4 statement
theorem second_spray_kill_percent (x : ℝ) 
  (H : 100 - (first_spray_kill_percent + x - both_spray_kill_percent) = germs_left_after_both) :
  x = 15 :=
by
  sorry

end second_spray_kill_percent_l210_210133


namespace min_cubes_l210_210978

-- Define the conditions
structure Cube := (x : ℕ) (y : ℕ) (z : ℕ)
def shares_face (c1 c2 : Cube) : Prop :=
  (c1.x = c2.x ∧ c1.y = c2.y ∧ (c1.z = c2.z + 1 ∨ c1.z = c2.z - 1)) ∨
  (c1.x = c2.x ∧ c1.z = c2.z ∧ (c1.y = c2.y + 1 ∨ c1.y = c2.y - 1)) ∨
  (c1.y = c2.y ∧ c1.z = c2.z ∧ (c1.x = c2.x + 1 ∨ c1.x = c2.x - 1))

def front_view (cubes : List Cube) : Prop :=
  -- Representation of L-shape in xy-plane
  ∃ (c1 c2 c3 c4 c5 : Cube),
  cubes = [c1, c2, c3, c4, c5] ∧
  (c1.x = 0 ∧ c1.y = 0 ∧ c1.z = 0) ∧
  (c2.x = 1 ∧ c2.y = 0 ∧ c2.z = 0) ∧
  (c3.x = 2 ∧ c3.y = 0 ∧ c3.z = 0) ∧
  (c4.x = 2 ∧ c4.y = 1 ∧ c4.z = 0) ∧
  (c5.x = 1 ∧ c5.y = 2 ∧ c5.z = 0)

def side_view (cubes : List Cube) : Prop :=
  -- Representation of Z-shape in yz-plane
  ∃ (c1 c2 c3 c4 c5 : Cube),
  cubes = [c1, c2, c3, c4, c5] ∧
  (c1.x = 0 ∧ c1.y = 0 ∧ c1.z = 0) ∧
  (c2.x = 0 ∧ c2.y = 1 ∧ c2.z = 0) ∧
  (c3.x = 0 ∧ c3.y = 1 ∧ c3.z = 1) ∧
  (c4.x = 0 ∧ c4.y = 2 ∧ c4.z = 1) ∧
  (c5.x = 0 ∧ c5.y = 2 ∧ c5.z = 2)

-- Proof statement
theorem min_cubes (cubes : List Cube) (h1 : front_view cubes) (h2 : side_view cubes) : cubes.length = 5 :=
by sorry

end min_cubes_l210_210978


namespace total_surface_area_l210_210566

theorem total_surface_area (a b c : ℝ) 
  (h1 : a + b + c = 45) 
  (h2 : a^2 + b^2 + c^2 = 625) : 
  2 * (a * b + b * c + c * a) = 1400 :=
sorry

end total_surface_area_l210_210566


namespace totalCandy_l210_210503

-- Define the number of pieces of candy each person had
def TaquonCandy : ℕ := 171
def MackCandy : ℕ := 171
def JafariCandy : ℕ := 76

-- Prove that the total number of pieces of candy they had together is 418
theorem totalCandy : TaquonCandy + MackCandy + JafariCandy = 418 := by
  sorry

end totalCandy_l210_210503


namespace janele_cats_average_weight_l210_210229

noncomputable def average_weight_cats (w1 w2 w3 w4 : ℝ) : ℝ :=
  (w1 + w2 + w3 + w4) / 4

theorem janele_cats_average_weight :
  average_weight_cats 12 12 14.7 9.3 = 12 :=
by
  sorry

end janele_cats_average_weight_l210_210229


namespace find_prime_pairs_l210_210664

theorem find_prime_pairs (p q n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hn : 0 < n) :
  p * (p + 1) + q * (q + 1) = n * (n + 1) ↔ (p = 3 ∧ q = 5 ∧ n = 6) ∨ (p = 5 ∧ q = 3 ∧ n = 6) ∨ (p = 2 ∧ q = 2 ∧ n = 3) :=
by
  sorry

end find_prime_pairs_l210_210664


namespace maximize_profit_l210_210383

-- Definitions
def initial_employees := 320
def profit_per_employee := 200000
def profit_increase_per_layoff := 20000
def expense_per_laid_off_employee := 60000
def min_employees := (3 * initial_employees) / 4
def profit_function (x : ℝ) := -0.2 * x^2 + 38 * x + 6400

-- The main statement
theorem maximize_profit : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 80 ∧ (∀ y : ℝ, 0 ≤ y ∧ y ≤ 80 → profit_function y ≤ profit_function x) ∧ x = 80 :=
by
  sorry

end maximize_profit_l210_210383


namespace inequality_solution_set_l210_210840

theorem inequality_solution_set (x : ℝ) : (-2 < x ∧ x ≤ 3) ↔ (x - 3) / (x + 2) ≤ 0 := 
sorry

end inequality_solution_set_l210_210840


namespace median_number_of_children_l210_210434

-- Define the given conditions
def number_of_data_points : Nat := 13
def median_position : Nat := (number_of_data_points + 1) / 2

-- We assert the median value based on information given in the problem
def median_value : Nat := 4

-- Statement to prove the problem
theorem median_number_of_children (h1: median_position = 7) (h2: median_value = 4) : median_value = 4 := 
by
  sorry

end median_number_of_children_l210_210434


namespace no_x_intersect_one_x_intersect_l210_210420

variable (m : ℝ)

-- Define the original quadratic function
def quadratic_function (x : ℝ) := x^2 - 2 * m * x + m^2 + 3

-- 1. Prove the function does not intersect the x-axis
theorem no_x_intersect : ∀ m, ∀ x : ℝ, quadratic_function m x ≠ 0 := by
  intros
  unfold quadratic_function
  sorry

-- 2. Prove that translating down by 3 units intersects the x-axis at one point
def translated_quadratic (x : ℝ) := (x - m)^2

theorem one_x_intersect : ∃ x : ℝ, translated_quadratic m x = 0 := by
  unfold translated_quadratic
  sorry

end no_x_intersect_one_x_intersect_l210_210420


namespace fraction_identity_l210_210688

theorem fraction_identity (x y z : ℤ) (h : x / 2 = y / 3 ∧ y / 3 = z / 4) :
  (x + y) / (3 * y - 2 * z) = 5 :=
by
  sorry

end fraction_identity_l210_210688


namespace sphere_cylinder_surface_area_difference_l210_210376

theorem sphere_cylinder_surface_area_difference (R : ℝ) :
  let S_sphere := 4 * Real.pi * R^2
  let S_lateral := 4 * Real.pi * R^2
  S_sphere - S_lateral = 0 :=
by
  sorry

end sphere_cylinder_surface_area_difference_l210_210376


namespace probability_non_adjacent_two_twos_l210_210801

theorem probability_non_adjacent_two_twos : 
  let digits := [2, 0, 2, 3]
  let total_arrangements := 12 - 3
  let favorable_arrangements := 5
  (favorable_arrangements / total_arrangements : ℚ) = 5 / 9 :=
by
  sorry

end probability_non_adjacent_two_twos_l210_210801


namespace like_terms_satisfy_conditions_l210_210401

theorem like_terms_satisfy_conditions (m n : ℤ) (h1 : m - 1 = n) (h2 : m + n = 3) :
  m = 2 ∧ n = 1 := by
  sorry

end like_terms_satisfy_conditions_l210_210401


namespace range_of_t_l210_210354

-- Define set A and set B as conditions
def setA := { x : ℝ | -3 < x ∧ x < 7 }
def setB (t : ℝ) := { x : ℝ | t + 1 < x ∧ x < 2 * t - 1 }

-- Lean statement to prove the range of t
theorem range_of_t (t : ℝ) : setB t ⊆ setA → t ≤ 4 :=
by
  -- sorry acts as a placeholder for the proof
  sorry

end range_of_t_l210_210354


namespace no_real_roots_x2_bx_8_eq_0_l210_210500

theorem no_real_roots_x2_bx_8_eq_0 (b : ℝ) :
  (∀ x : ℝ, x^2 + b * x + 5 ≠ -3) ↔ (-4 * Real.sqrt 2 < b ∧ b < 4 * Real.sqrt 2) := by
  sorry

end no_real_roots_x2_bx_8_eq_0_l210_210500


namespace sum_of_numbers_l210_210221

theorem sum_of_numbers :
  145 + 35 + 25 + 5 = 210 :=
by
  sorry

end sum_of_numbers_l210_210221


namespace compare_sqrts_l210_210586

theorem compare_sqrts (a b c : ℝ) (h1 : a = 2 * Real.sqrt 7) (h2 : b = 3 * Real.sqrt 5) (h3 : c = 5 * Real.sqrt 2):
  c > b ∧ b > a :=
by
  sorry

end compare_sqrts_l210_210586


namespace remainder_4059_div_32_l210_210513

theorem remainder_4059_div_32 : 4059 % 32 = 27 := by
  sorry

end remainder_4059_div_32_l210_210513


namespace avg_diff_l210_210337

theorem avg_diff (a x c : ℝ) (h1 : (a + x) / 2 = 40) (h2 : (x + c) / 2 = 60) :
  c - a = 40 :=
by
  sorry

end avg_diff_l210_210337


namespace bells_ring_together_l210_210220

open Nat

theorem bells_ring_together :
  let library_interval := 18
  let fire_station_interval := 24
  let hospital_interval := 30
  let start_time := 0
  let next_ring_time := Nat.lcm (Nat.lcm library_interval fire_station_interval) hospital_interval
  let total_minutes_in_an_hour := 60
  next_ring_time / total_minutes_in_an_hour = 6 :=
by
  let library_interval := 18
  let fire_station_interval := 24
  let hospital_interval := 30
  let start_time := 0
  let next_ring_time := Nat.lcm (Nat.lcm library_interval fire_station_interval) hospital_interval
  let total_minutes_in_an_hour := 60
  have h_next_ring_time : next_ring_time = 360 := by
    sorry
  have h_hours : next_ring_time / total_minutes_in_an_hour = 6 := by
    sorry
  exact h_hours

end bells_ring_together_l210_210220


namespace three_x_plus_three_y_plus_three_z_l210_210274

theorem three_x_plus_three_y_plus_three_z (x y z : ℝ) 
  (h1 : y + z = 20 - 5 * x)
  (h2 : x + z = -18 - 5 * y)
  (h3 : x + y = 10 - 5 * z) :
  3 * x + 3 * y + 3 * z = 36 / 7 := by
  sorry

end three_x_plus_three_y_plus_three_z_l210_210274


namespace solve_eq1_solve_eq2_l210_210118

theorem solve_eq1 : ∀ (x : ℚ), (3 / 5 - 5 / 8 * x = 2 / 5) → (x = 8 / 25) := by
  intro x
  intro h
  sorry

theorem solve_eq2 : ∀ (x : ℚ), (7 * (x - 2) = 8 * (x - 4)) → (x = 18) := by
  intro x
  intro h
  sorry

end solve_eq1_solve_eq2_l210_210118


namespace professors_seat_choice_count_l210_210997

theorem professors_seat_choice_count : 
    let chairs := 11 -- number of chairs
    let students := 7 -- number of students
    let professors := 4 -- number of professors
    ∀ (P: Fin professors -> Fin chairs), 
    (∀ (p : Fin professors), 1 ≤ P p ∧ P p ≤ 9) -- Each professor is between seats 2-10
    ∧ (P 0 < P 1) ∧ (P 1 < P 2) ∧ (P 2 < P 3) -- Professors must be placed with at least one seat gap
    ∧ (P 0 ≠ 1 ∧ P 3 ≠ 11) -- First and last seats are excluded
    → ∃ (ways : ℕ), ways = 840 := sorry

end professors_seat_choice_count_l210_210997


namespace find_x_minus_y_l210_210782

theorem find_x_minus_y (x y : ℝ) (h1 : 2 * x + 3 * y = 14) (h2 : x + 4 * y = 11) : x - y = 3 := by
  sorry

end find_x_minus_y_l210_210782


namespace proof_part1_proof_part2_l210_210779

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l210_210779


namespace average_speed_l210_210584

-- Define the speeds in the first and second hours
def speed_first_hour : ℝ := 90
def speed_second_hour : ℝ := 42

-- Define the time taken for each hour
def time_first_hour : ℝ := 1
def time_second_hour : ℝ := 1

-- Calculate the total distance and total time
def total_distance : ℝ := speed_first_hour + speed_second_hour
def total_time : ℝ := time_first_hour + time_second_hour

-- State the theorem for the average speed
theorem average_speed : total_distance / total_time = 66 := by
  sorry

end average_speed_l210_210584


namespace find_tangent_points_l210_210002

def f (x : ℝ) : ℝ := x^3 + x - 2
def tangent_parallel_to_line (x : ℝ) : Prop := deriv f x = 4

theorem find_tangent_points :
  (tangent_parallel_to_line 1 ∧ f 1 = 0) ∧ 
  (tangent_parallel_to_line (-1) ∧ f (-1) = -4) :=
by
  sorry

end find_tangent_points_l210_210002


namespace binom_2p_p_mod_p_l210_210867

theorem binom_2p_p_mod_p (p : ℕ) (hp : p.Prime) : Nat.choose (2 * p) p ≡ 2 [MOD p] := 
by
  sorry

end binom_2p_p_mod_p_l210_210867


namespace mike_total_earning_l210_210290

theorem mike_total_earning 
  (first_job : ℕ := 52)
  (hours : ℕ := 12)
  (wage_per_hour : ℕ := 9) :
  first_job + (hours * wage_per_hour) = 160 :=
by
  sorry

end mike_total_earning_l210_210290


namespace find_x_l210_210398

-- Let x be a real number such that x > 0 and the area of the given triangle is 180.
theorem find_x (x : ℝ) (h_pos : x > 0) (h_area : 3 * x^2 = 180) : x = 2 * Real.sqrt 15 :=
by
  -- Placeholder for the actual proof
  sorry

end find_x_l210_210398


namespace original_distance_between_Stacy_and_Heather_l210_210013

theorem original_distance_between_Stacy_and_Heather
  (H_speed : ℝ := 5)  -- Heather's speed in miles per hour
  (S_speed : ℝ := 6)  -- Stacy's speed in miles per hour
  (delay : ℝ := 0.4)  -- Heather's start delay in hours
  (H_distance : ℝ := 1.1818181818181817)  -- Distance Heather walked when they meet
  : H_speed * (H_distance / H_speed) + S_speed * ((H_distance / H_speed) + delay) = 5 := by
  sorry

end original_distance_between_Stacy_and_Heather_l210_210013


namespace find_x_range_l210_210994

variable {x : ℝ}

def P (x : ℝ) : Prop := x^2 - 2*x - 3 ≥ 0

def Q (x : ℝ) : Prop := |1 - x/2| < 1

theorem find_x_range (hP : P x) (hQ : ¬ Q x) : x ≤ -1 ∨ x ≥ 4 :=
  sorry

end find_x_range_l210_210994


namespace integer_ratio_l210_210464

theorem integer_ratio (A B C D : ℕ) (h1 : (A + B + C + D) / 4 = 16)
  (h2 : A % B = 0) (h3 : B = C - 2) (h4 : D = 2) (h5 : A ≠ B) (h6 : B ≠ C) (h7 : C ≠ D) (h8 : D ≠ A)
  (h9: 0 < A) (h10: 0 < B) (h11: 0 < C):
  A / B = 28 := 
sorry

end integer_ratio_l210_210464


namespace same_terminal_side_l210_210973

theorem same_terminal_side (k : ℤ) : 
  ∃ (α : ℤ), α = k * 360 + 330 ∧ (α = 510 ∨ α = 150 ∨ α = -150 ∨ α = -390) :=
by
  sorry

end same_terminal_side_l210_210973


namespace avg_meal_cost_per_individual_is_72_l210_210465

theorem avg_meal_cost_per_individual_is_72
  (total_bill : ℝ)
  (gratuity_percent : ℝ)
  (num_investment_bankers num_clients : ℕ)
  (total_individuals := num_investment_bankers + num_clients)
  (meal_cost_before_gratuity : ℝ := total_bill / (1 + gratuity_percent))
  (average_cost := meal_cost_before_gratuity / total_individuals) :
  total_bill = 1350 ∧ gratuity_percent = 0.25 ∧ num_investment_bankers = 7 ∧ num_clients = 8 →
  average_cost = 72 := by
  sorry

end avg_meal_cost_per_individual_is_72_l210_210465


namespace polynomial_expansion_l210_210381

theorem polynomial_expansion : (x + 3) * (x - 6) * (x + 2) = x^3 - x^2 - 24 * x - 36 := 
by
  sorry

end polynomial_expansion_l210_210381


namespace john_annual_profit_l210_210529

-- Definitions of monthly incomes
def TenantA_income : ℕ := 350
def TenantB_income : ℕ := 400
def TenantC_income : ℕ := 450

-- Total monthly income
def total_monthly_income : ℕ := TenantA_income + TenantB_income + TenantC_income

-- Definitions of monthly expenses
def rent_expense : ℕ := 900
def utilities_expense : ℕ := 100
def maintenance_fee : ℕ := 50

-- Total monthly expenses
def total_monthly_expense : ℕ := rent_expense + utilities_expense + maintenance_fee

-- Monthly profit
def monthly_profit : ℕ := total_monthly_income - total_monthly_expense

-- Annual profit
def annual_profit : ℕ := monthly_profit * 12

theorem john_annual_profit :
  annual_profit = 1800 := by
  -- The proof is omitted, but the statement asserts that John makes an annual profit of $1800.
  sorry

end john_annual_profit_l210_210529


namespace batsman_average_after_17_matches_l210_210819

theorem batsman_average_after_17_matches (A : ℕ) (h : (17 * (A + 3) = 16 * A + 87)) : A + 3 = 39 := by
  sorry

end batsman_average_after_17_matches_l210_210819


namespace max_area_enclosed_by_fencing_l210_210974

theorem max_area_enclosed_by_fencing (l w : ℕ) (h : 2 * (l + w) = 142) : l * w ≤ 1260 :=
sorry

end max_area_enclosed_by_fencing_l210_210974


namespace solve_system_l210_210790

theorem solve_system : ∀ (x y : ℤ), 2 * x + y = 5 → x + 2 * y = 6 → x - y = -1 :=
by
  intros x y h1 h2
  sorry

end solve_system_l210_210790


namespace blocks_to_store_l210_210209

theorem blocks_to_store
  (T : ℕ) (S : ℕ)
  (hT : T = 25)
  (h_total_walk : S + 6 + 8 = T) :
  S = 11 :=
by
  sorry

end blocks_to_store_l210_210209


namespace initial_pennies_l210_210758

theorem initial_pennies (initial: ℕ) (h : initial + 93 = 191) : initial = 98 := by
  sorry

end initial_pennies_l210_210758


namespace simplify_expression1_simplify_expression2_l210_210950

variable {x y : ℝ} -- Declare x and y as real numbers

theorem simplify_expression1 :
  3 * x^2 - (7 * x - (4 * x - 3) - 2 * x^2) = 5 * x^2 - 3 * x - 3 :=
sorry

theorem simplify_expression2 :
  3 * x^2 * y - (2 * x * y - 2 * (x * y - (3/2) * x^2 * y) + x^2 * y^2) = - x^2 * y^2 :=
sorry

end simplify_expression1_simplify_expression2_l210_210950


namespace fifth_term_arithmetic_sequence_l210_210328

variable (a d : ℤ)

def arithmetic_sequence (n : ℤ) : ℤ :=
  a + (n - 1) * d

theorem fifth_term_arithmetic_sequence :
  arithmetic_sequence a d 20 = 12 →
  arithmetic_sequence a d 21 = 15 →
  arithmetic_sequence a d 5 = -33 :=
by
  intro h20 h21
  sorry

end fifth_term_arithmetic_sequence_l210_210328


namespace maximal_N8_value_l210_210173

noncomputable def max_permutations_of_projections (A : Fin 8 → ℝ × ℝ) : ℕ := sorry

theorem maximal_N8_value (A : Fin 8 → ℝ × ℝ) :
  max_permutations_of_projections A = 56 :=
sorry

end maximal_N8_value_l210_210173


namespace modulus_z_eq_sqrt_10_l210_210809

noncomputable def z := (10 * Complex.I) / (3 + Complex.I)

theorem modulus_z_eq_sqrt_10 : Complex.abs z = Real.sqrt 10 := by
  sorry

end modulus_z_eq_sqrt_10_l210_210809


namespace arith_seq_sum_l210_210590

theorem arith_seq_sum (a₃ a₄ a₅ : ℤ) (h₁ : a₃ = 7) (h₂ : a₄ = 11) (h₃ : a₅ = 15) :
  let d := a₄ - a₃;
  let a := a₄ - 3 * d;
  (6 / 2 * (2 * a + 5 * d)) = 54 :=
by
  sorry

end arith_seq_sum_l210_210590


namespace caterpillar_count_l210_210342

theorem caterpillar_count 
    (initial_count : ℕ)
    (hatched : ℕ)
    (left : ℕ)
    (h_initial : initial_count = 14)
    (h_hatched : hatched = 4)
    (h_left : left = 8) :
    initial_count + hatched - left = 10 :=
by
    sorry

end caterpillar_count_l210_210342


namespace machine_present_value_l210_210067

theorem machine_present_value
  (depreciation_rate : ℝ)
  (selling_price : ℝ)
  (profit : ℝ)
  (dep_years : ℕ)
  (value_after_depreciation : ℝ)
  (present_value : ℝ) :

  depreciation_rate = 0.8 →
  selling_price = 118000.00000000001 →
  profit = 22000 →
  dep_years = 2 →
  value_after_depreciation = (selling_price - profit) →
  value_after_depreciation = 96000.00000000001 →
  present_value * (depreciation_rate ^ dep_years) = value_after_depreciation →
  present_value = 150000.00000000002 :=
by sorry

end machine_present_value_l210_210067


namespace jose_total_caps_l210_210224

def initial_caps := 26
def additional_caps := 13
def total_caps := initial_caps + additional_caps

theorem jose_total_caps : total_caps = 39 :=
by
  sorry

end jose_total_caps_l210_210224


namespace couples_at_prom_l210_210366

theorem couples_at_prom (total_students attending_alone attending_with_partners couples : ℕ) 
  (h1 : total_students = 123) 
  (h2 : attending_alone = 3) 
  (h3 : attending_with_partners = total_students - attending_alone) 
  (h4 : couples = attending_with_partners / 2) : 
  couples = 60 := 
by 
  sorry

end couples_at_prom_l210_210366


namespace find_PF_2_l210_210070

-- Define the hyperbola and points
def hyperbola_eq (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 3) = 1
def PF_1 := 3
def a := 2
def two_a := 2 * a

-- State the theorem
theorem find_PF_2 (PF_2 : ℝ) (cond1 : PF_1 = 3) (cond2 : abs (PF_1 - PF_2) = two_a) : PF_2 = 7 :=
sorry

end find_PF_2_l210_210070


namespace average_speed_of_train_l210_210893

def ChicagoTime (t : String) : Prop := t = "5:00 PM"
def NewYorkTime (t : String) : Prop := t = "10:00 AM"
def TimeDifference (d : Nat) : Prop := d = 1
def Distance (d : Nat) : Prop := d = 480

theorem average_speed_of_train :
  ∀ (d t1 t2 diff : Nat), 
  Distance d → (NewYorkTime "10:00 AM") → (ChicagoTime "5:00 PM") → TimeDifference diff →
  (t2 = 5 ∧ t1 = (10 - diff)) →
  (d / (t2 - t1) = 60) :=
by
  intros d t1 t2 diff hD ht1 ht2 hDiff hTimes
  sorry

end average_speed_of_train_l210_210893


namespace gcd_example_l210_210223

theorem gcd_example : Nat.gcd 8675309 7654321 = 36 := sorry

end gcd_example_l210_210223


namespace renovation_services_are_credence_goods_and_choice_arguments_l210_210394

-- Define what credence goods are and the concept of information asymmetry
structure CredenceGood where
  information_asymmetry : Prop
  unobservable_quality  : Prop

-- Define renovation service as an instance of CredenceGood
def RenovationService : CredenceGood := {
  information_asymmetry := true,
  unobservable_quality := true
}

-- Primary conditions for choosing between construction company and private repair crew
structure ChoiceArgument where
  information_availability     : Prop
  warranty_and_accountability  : Prop
  higher_costs                 : Prop
  potential_bias_in_reviews    : Prop

-- Arguments for using construction company
def ConstructionCompanyArguments : ChoiceArgument := {
  information_availability := true,
  warranty_and_accountability := true,
  higher_costs := true,
  potential_bias_in_reviews := true
}

-- Arguments against using construction company
def PrivateRepairCrewArguments : ChoiceArgument := {
  information_availability := false,
  warranty_and_accountability := false,
  higher_costs := true,
  potential_bias_in_reviews := true
}

-- Proof statement to show renovation services are credence goods and economically reasoned arguments for/against
theorem renovation_services_are_credence_goods_and_choice_arguments:
  RenovationService = {
    information_asymmetry := true,
    unobservable_quality := true
  } ∧
  (ConstructionCompanyArguments.information_availability = true ∧
   ConstructionCompanyArguments.warranty_and_accountability = true) ∧
  (ConstructionCompanyArguments.higher_costs = true ∧
   ConstructionCompanyArguments.potential_bias_in_reviews = true) ∧
  (PrivateRepairCrewArguments.higher_costs = true ∧
   PrivateRepairCrewArguments.potential_bias_in_reviews = true) :=
by sorry

end renovation_services_are_credence_goods_and_choice_arguments_l210_210394


namespace find_x_l210_210508

variable (x : ℝ)
variable (s : ℝ)

-- Conditions as hypothesis
def square_perimeter_60 (s : ℝ) : Prop := 4 * s = 60
def triangle_area_150 (x s : ℝ) : Prop := (1 / 2) * x * s = 150
def height_equals_side (s : ℝ) : Prop := true

-- Proof problem statement
theorem find_x 
  (h1 : square_perimeter_60 s)
  (h2 : triangle_area_150 x s)
  (h3 : height_equals_side s) : 
  x = 20 := 
sorry

end find_x_l210_210508


namespace rectangular_prism_dimensions_l210_210353

theorem rectangular_prism_dimensions (a b c : ℤ) (h1: c = (a * b) / 2) (h2: 2 * (a * b + b * c + c * a) = a * b * c) :
  (a = 3 ∧ b = 10 ∧ c = 15) ∨ (a = 4 ∧ b = 6 ∧ c = 12) :=
by {
  sorry
}

end rectangular_prism_dimensions_l210_210353


namespace highest_water_level_changes_on_tuesday_l210_210227

def water_levels : List (String × Float) :=
  [("Monday", 0.03), ("Tuesday", 0.41), ("Wednesday", 0.25), ("Thursday", 0.10),
   ("Friday", 0.0), ("Saturday", -0.13), ("Sunday", -0.2)]

theorem highest_water_level_changes_on_tuesday :
  ∃ d : String, d = "Tuesday" ∧ ∀ d' : String × Float, d' ∈ water_levels → d'.snd ≤ 0.41 := by
  sorry

end highest_water_level_changes_on_tuesday_l210_210227


namespace distance_X_X_l210_210639

/-
  Define the vertices of the triangle XYZ
-/
def X : ℝ × ℝ := (2, -4)
def Y : ℝ × ℝ := (-1, 2)
def Z : ℝ × ℝ := (5, 1)

/-
  Define the reflection of point X over the y-axis
-/
def X' : ℝ × ℝ := (-2, -4)

/-
  Prove that the distance between X and X' is 4 units.
-/
theorem distance_X_X' : (Real.sqrt (((-2) - 2) ^ 2 + ((-4) - (-4)) ^ 2)) = 4 := by
  sorry

end distance_X_X_l210_210639


namespace cube_edge_length_eq_six_l210_210053

theorem cube_edge_length_eq_six {s : ℝ} (h : s^3 = 6 * s^2) : s = 6 :=
sorry

end cube_edge_length_eq_six_l210_210053


namespace intersection_M_N_l210_210404

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x | -1 ≤ x ∧ x ≤ 1}

theorem intersection_M_N : M ∩ N = {0, 1} := 
by
  sorry

end intersection_M_N_l210_210404


namespace thirty_times_multiple_of_every_integer_is_zero_l210_210717

theorem thirty_times_multiple_of_every_integer_is_zero (n : ℤ) (h : ∀ x : ℤ, n = 30 * x ∧ x = 0 → n = 0) : n = 0 :=
by
  sorry

end thirty_times_multiple_of_every_integer_is_zero_l210_210717


namespace unique_isolating_line_a_eq_2e_l210_210548

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (a x : ℝ) : ℝ := a * Real.log x

theorem unique_isolating_line_a_eq_2e (a : ℝ) (h : a > 0) :
  (∃ k b, ∀ x : ℝ, f x ≥ k * x + b ∧ k * x + b ≥ g a x) → a = 2 * Real.exp 1 :=
sorry

end unique_isolating_line_a_eq_2e_l210_210548


namespace louis_age_currently_31_l210_210998

-- Definitions
variable (C L : ℕ)
variable (h1 : C + 6 = 30)
variable (h2 : C + L = 55)

-- Theorem statement
theorem louis_age_currently_31 : L = 31 :=
by
  sorry

end louis_age_currently_31_l210_210998


namespace exponentiation_distributes_over_multiplication_l210_210720

theorem exponentiation_distributes_over_multiplication (a b c : ℝ) : (a * b) ^ c = a ^ c * b ^ c := 
sorry

end exponentiation_distributes_over_multiplication_l210_210720


namespace isosceles_triangle_base_angles_l210_210647

theorem isosceles_triangle_base_angles (a b : ℝ) (h1 : a + b + b = 180)
  (h2 : a = 110) : b = 35 :=
by 
  sorry

end isosceles_triangle_base_angles_l210_210647


namespace number_of_juniors_in_sample_l210_210786

theorem number_of_juniors_in_sample
  (total_students : ℕ)
  (num_freshmen : ℕ)
  (num_freshmen_sampled : ℕ)
  (num_sophomores_exceeds_num_juniors_by : ℕ)
  (num_sophomores num_juniors num_juniors_sampled : ℕ)
  (h_total : total_students = 1290)
  (h_num_freshmen : num_freshmen = 480)
  (h_num_freshmen_sampled : num_freshmen_sampled = 96)
  (h_exceeds : num_sophomores_exceeds_num_juniors_by = 30)
  (h_equation : total_students - num_freshmen = num_sophomores + num_juniors)
  (h_num_sophomores : num_sophomores = num_juniors + num_sophomores_exceeds_num_juniors_by)
  (h_fraction : num_freshmen_sampled / num_freshmen = 1 / 5)
  (h_num_juniors_sampled : num_juniors_sampled = num_juniors * (num_freshmen_sampled / num_freshmen)) :
  num_juniors_sampled = 78 := by
  sorry

end number_of_juniors_in_sample_l210_210786


namespace total_books_read_l210_210544

-- Definitions based on the conditions
def books_per_month : ℕ := 4
def months_per_year : ℕ := 12
def books_per_year_per_student : ℕ := books_per_month * months_per_year

variables (c s : ℕ)

-- Main theorem statement
theorem total_books_read (c s : ℕ) : 
  (books_per_year_per_student * c * s) = 48 * c * s :=
by
  sorry

end total_books_read_l210_210544


namespace diamond_more_olivine_l210_210628

theorem diamond_more_olivine :
  ∃ A O D : ℕ, A = 30 ∧ O = A + 5 ∧ A + O + D = 111 ∧ D - O = 11 :=
by
  sorry

end diamond_more_olivine_l210_210628


namespace intersection_eq_zero_set_l210_210195

def M : Set ℤ := {-1, 0, 1}

def N : Set ℤ := {x | x^2 ≤ 0}

theorem intersection_eq_zero_set : M ∩ N = {0} := by
  sorry

end intersection_eq_zero_set_l210_210195


namespace soap_last_duration_l210_210763

-- Definitions of the given conditions
def cost_per_bar := 8 -- cost in dollars
def total_spent := 48 -- total spent in dollars
def months_in_year := 12

-- Definition of the query statement/proof goal
theorem soap_last_duration (h₁ : total_spent = 48) (h₂ : cost_per_bar = 8) (h₃ : months_in_year = 12) : months_in_year / (total_spent / cost_per_bar) = 2 :=
by 
  -- Placeholder for the proof
  sorry

end soap_last_duration_l210_210763


namespace g_eq_l210_210431

noncomputable def g (n : ℕ) : ℝ :=
  (7 + 4 * Real.sqrt 7) / 14 * ((1 + Real.sqrt 7) / 2) ^ n +
  (7 - 4 * Real.sqrt 7) / 14 * ((1 - Real.sqrt 7) / 2) ^ n

theorem g_eq (n : ℕ) : g (n + 2) - g (n - 2) = 3 * g n := by
  sorry

end g_eq_l210_210431


namespace evaluation_at_2_l210_210334

def f (x : ℚ) : ℚ := (2 * x^2 + 7 * x + 12) / (x^2 + 2 * x + 5)
def g (x : ℚ) : ℚ := x - 2

theorem evaluation_at_2 :
  f (g 2) + g (f 2) = 196 / 65 := by
  sorry

end evaluation_at_2_l210_210334


namespace expression_equals_five_l210_210532

theorem expression_equals_five (a : ℝ) (h : 2 * a^2 - 3 * a + 4 = 5) : 7 + 6 * a - 4 * a^2 = 5 :=
by
  sorry

end expression_equals_five_l210_210532


namespace lower_side_length_is_correct_l210_210034

noncomputable def length_of_lower_side
  (a b h : ℝ) (A : ℝ) 
  (cond1 : a = b + 3.4)
  (cond2 : h = 5.2)
  (cond3 : A = 100.62) : ℝ :=
b

theorem lower_side_length_is_correct
  (a b h : ℝ) (A : ℝ)
  (cond1 : a = b + 3.4)
  (cond2 : h = 5.2)
  (cond3 : A = 100.62)
  (ha : A = (1/2) * (a + b) * h) : b = 17.65 :=
by
  sorry

end lower_side_length_is_correct_l210_210034


namespace min_value_on_neg_infinite_l210_210147

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def max_value_on_interval (F : ℝ → ℝ) (a b : ℝ) (max_val : ℝ) : Prop :=
∀ x, (0 < x → F x ≤ max_val) ∧ (∃ y, 0 < y ∧ F y = max_val)

theorem min_value_on_neg_infinite (f g : ℝ → ℝ) (a b : ℝ) (F : ℝ → ℝ)
  (h_odd_f : odd_function f) (h_odd_g : odd_function g)
  (h_def_F : ∀ x, F x = a * f x + b * g x + 2)
  (h_max_F_on_0_inf : max_value_on_interval F a b 8) :
  ∃ x, x < 0 ∧ F x = -4 :=
sorry

end min_value_on_neg_infinite_l210_210147


namespace farmer_land_area_l210_210682

-- Variables representing the total land, and the percentages and areas.
variable {T : ℝ} (h_cleared : 0.85 * T =  V) (V_10_percent : 0.10 * V + 0.70 * V + 0.05 * V + 500 = V)
variable {total_acres : ℝ} (correct_total_acres : total_acres = 3921.57)

theorem farmer_land_area (h_cleared : 0.85 * T = V) (h_planted : 0.85 * V = 500) : T = 3921.57 :=
by
  sorry

end farmer_land_area_l210_210682


namespace original_triangle_angles_determined_l210_210755

-- Define the angles of the formed triangle
def formed_triangle_angles : Prop := 
  52 + 61 + 67 = 180

-- Define the angles of the original triangle
def original_triangle_angles (α β γ : ℝ) : Prop := 
  α + β + γ = 180

theorem original_triangle_angles_determined :
  formed_triangle_angles → 
  ∃ α β γ : ℝ, 
    original_triangle_angles α β γ ∧
    α = 76 ∧ β = 58 ∧ γ = 46 :=
by
  sorry

end original_triangle_angles_determined_l210_210755


namespace number_of_people_per_cubic_yard_l210_210638

-- Lean 4 statement

variable (P : ℕ) -- Number of people per cubic yard

def city_population_9000 := 9000 * P
def city_population_6400 := 6400 * P

theorem number_of_people_per_cubic_yard :
  city_population_9000 - city_population_6400 = 208000 →
  P = 80 :=
by
  sorry

end number_of_people_per_cubic_yard_l210_210638


namespace probability_quadratic_real_roots_l210_210619

noncomputable def probability_real_roots : ℝ := 3 / 4

theorem probability_quadratic_real_roots :
  (∀ a b : ℝ, -π ≤ a ∧ a ≤ π ∧ -π ≤ b ∧ b ≤ π →
  (∃ x : ℝ, x^2 + 2*a*x - b^2 + π = 0) ↔ a^2 + b^2 ≥ π) →
  (probability_real_roots = 3 / 4) :=
sorry

end probability_quadratic_real_roots_l210_210619


namespace find_t_l210_210501

-- Define the roots and basic properties
variables (a b c : ℝ)
variables (r s t : ℝ)

-- Define conditions from the first cubic equation
def first_eq_roots : Prop :=
  a + b + c = -5 ∧ a * b * c = 13

-- Define conditions from the second cubic equation with shifted roots
def second_eq_roots : Prop :=
  t = -(a * b * c + a * b + a * c + b * c + a + b + c + 1)

-- The theorem stating the value of t
theorem find_t (h₁ : first_eq_roots a b c) (h₂ : second_eq_roots a b c t) : t = -15 :=
sorry

end find_t_l210_210501


namespace circle_locus_l210_210181

theorem circle_locus (a b : ℝ) :
  (∃ r : ℝ, (a^2 + b^2 = (r + 2)^2 ∧ (a - 3)^2 + b^2 = (5 - r)^2)) ↔ 
  13 * a^2 + 49 * b^2 - 12 * a - 1 = 0 := 
sorry

end circle_locus_l210_210181


namespace Michael_selection_l210_210921

theorem Michael_selection :
  (Nat.choose 8 3) * (Nat.choose 5 2) = 560 :=
by
  sorry

end Michael_selection_l210_210921


namespace sachin_age_l210_210549

variable {S R : ℕ}

theorem sachin_age
  (h1 : R = S + 7)
  (h2 : S * 3 = 2 * R) :
  S = 14 :=
sorry

end sachin_age_l210_210549


namespace tonya_needs_to_eat_more_l210_210846

-- Define the conditions in the problem
def ounces_per_hamburger : ℕ := 4
def ounces_eaten_last_year : ℕ := 84

-- Define a function to calculate hamburgers given ounces
def hamburgers_eaten (ounces : ℕ) (ounce_per_hamburger : ℕ) : ℕ :=
  ounces / ounce_per_hamburger

-- State the theorem
theorem tonya_needs_to_eat_more (ounces_per_hamburger ounces_eaten_last_year : ℕ) :
  hamburgers_eaten ounces_eaten_last_year ounces_per_hamburger + 1 = 22 := by
  sorry

end tonya_needs_to_eat_more_l210_210846


namespace combined_salaries_A_B_C_D_l210_210535

-- To ensure the whole calculation is noncomputable due to ℝ
noncomputable section

-- Let's define the variables
def salary_E : ℝ := 9000
def average_salary_group : ℝ := 8400
def num_people : ℕ := 5

-- combined salary A + B + C + D represented as a definition
def combined_salaries : ℝ := (average_salary_group * num_people) - salary_E

-- We need to prove that the combined salaries equals 33000
theorem combined_salaries_A_B_C_D : combined_salaries = 33000 := by
  sorry

end combined_salaries_A_B_C_D_l210_210535


namespace max_expression_tends_to_infinity_l210_210325

noncomputable def maximize_expression (x y z : ℝ) : ℝ :=
  1 / ((1 - x^2) * (1 - y^2) * (1 - z^2)) + 1 / ((1 + x^2) * (1 + y^2) * (1 + z^2))

theorem max_expression_tends_to_infinity : 
  ∀ (x y z : ℝ), -1 < x ∧ x < 1 ∧ -1 < y ∧ y < 1 ∧ -1 < z ∧ z < 1 → 
    ∃ M : ℝ, maximize_expression x y z > M :=
by
  intro x y z h
  sorry

end max_expression_tends_to_infinity_l210_210325


namespace probability_red_or_blue_l210_210017

noncomputable def total_marbles : ℕ := 100

noncomputable def probability_white : ℚ := 1 / 4

noncomputable def probability_green : ℚ := 1 / 5

theorem probability_red_or_blue :
  (1 - (probability_white + probability_green)) = 11 / 20 :=
by
  -- Proof is omitted
  sorry

end probability_red_or_blue_l210_210017


namespace p_evaluation_l210_210370

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then x + y
  else if x < 0 ∧ y < 0 then x - 3 * y
  else 2 * x + 2 * y

theorem p_evaluation : p (p 3 (-4)) (p (-7) 0) = 40 := by
  sorry

end p_evaluation_l210_210370


namespace ashwin_rental_hours_l210_210373

theorem ashwin_rental_hours (x : ℕ) 
  (h1 : 25 + 10 * x = 125) : 1 + x = 11 :=
by
  sorry

end ashwin_rental_hours_l210_210373


namespace find_percent_defective_l210_210248

def percent_defective (D : ℝ) : Prop :=
  (0.04 * D = 0.32)

theorem find_percent_defective : ∃ D, percent_defective D ∧ D = 8 := by
  sorry

end find_percent_defective_l210_210248


namespace initial_mat_weavers_l210_210149

variable (num_weavers : ℕ) (rate : ℕ → ℕ → ℕ) -- rate weaver_count duration_in_days → mats_woven

-- Given Conditions
def condition1 := rate num_weavers 4 = 4
def condition2 := rate (2 * num_weavers) 8 = 16

-- Theorem to Prove
theorem initial_mat_weavers : num_weavers = 4 :=
by
  sorry

end initial_mat_weavers_l210_210149


namespace probability_one_male_correct_probability_atleast_one_female_correct_l210_210083

def total_students := 5
def female_students := 2
def male_students := 3
def number_of_selections := 2

noncomputable def probability_only_one_male : ℚ :=
  (6 : ℚ) / 10

noncomputable def probability_atleast_one_female : ℚ :=
  (7 : ℚ) / 10

theorem probability_one_male_correct :
  (6 / 10 : ℚ) = 3 / 5 :=
by
  sorry

theorem probability_atleast_one_female_correct :
  (7 / 10 : ℚ) = 7 / 10 :=
by
  sorry

end probability_one_male_correct_probability_atleast_one_female_correct_l210_210083


namespace expression_that_gives_value_8_l210_210979

theorem expression_that_gives_value_8 (a b : ℝ) 
  (h_eq1 : a = 2) 
  (h_eq2 : b = 2) 
  (h_roots : ∀ x, (x - a) * (x - b) = x^2 - 4 * x + 4) : 
  2 * (a + b) = 8 :=
by
  sorry

end expression_that_gives_value_8_l210_210979


namespace geometric_sequence_condition_l210_210740

variable {a : ℕ → ℝ}

-- Definitions based on conditions in the problem
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

-- The statement translating the problem
theorem geometric_sequence_condition (q : ℝ) (a : ℕ → ℝ) (h : is_geometric_sequence a q) : ¬((q > 1) ↔ is_increasing_sequence a) :=
  sorry

end geometric_sequence_condition_l210_210740


namespace greatest_product_three_integers_sum_2000_l210_210059

noncomputable def maxProduct (s : ℝ) : ℝ := 
  s * s * (2000 - 2 * s)

theorem greatest_product_three_integers_sum_2000 : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2000 / 2 ∧ maxProduct x = 8000000000 / 27 := sorry

end greatest_product_three_integers_sum_2000_l210_210059


namespace ratio_u_v_l210_210773

theorem ratio_u_v (b : ℝ) (hb : b ≠ 0) (u v : ℝ) 
  (h1 : 0 = 8 * u + b) 
  (h2 : 0 = 4 * v + b) 
  : u / v = 1 / 2 := 
by sorry

end ratio_u_v_l210_210773


namespace omar_rolls_l210_210784

-- Define the conditions
def karen_rolls : ℕ := 229
def total_rolls : ℕ := 448

-- Define the main theorem to prove the number of rolls by Omar
theorem omar_rolls : (total_rolls - karen_rolls) = 219 := by
  sorry

end omar_rolls_l210_210784


namespace gcd_105_490_l210_210430

theorem gcd_105_490 : Nat.gcd 105 490 = 35 := by
sorry

end gcd_105_490_l210_210430


namespace total_savings_correct_l210_210271

-- Definitions of savings per day and days saved for Josiah, Leah, and Megan
def josiah_saving_per_day : ℝ := 0.25
def josiah_days : ℕ := 24

def leah_saving_per_day : ℝ := 0.50
def leah_days : ℕ := 20

def megan_saving_per_day : ℝ := 1.00
def megan_days : ℕ := 12

-- Definition to calculate total savings for each child
def total_saving (saving_per_day : ℝ) (days : ℕ) : ℝ :=
  saving_per_day * days

-- Total amount saved by Josiah, Leah, and Megan
def total_savings : ℝ :=
  total_saving josiah_saving_per_day josiah_days +
  total_saving leah_saving_per_day leah_days +
  total_saving megan_saving_per_day megan_days

-- Theorem to prove the total savings is $28
theorem total_savings_correct : total_savings = 28 := by
  sorry

end total_savings_correct_l210_210271


namespace tenth_term_arithmetic_seq_l210_210281

theorem tenth_term_arithmetic_seq :
  let a₁ : ℚ := 1 / 2
  let a₂ : ℚ := 5 / 6
  let d : ℚ := a₂ - a₁
  let a₁₀ : ℚ := a₁ + 9 * d
  a₁₀ = 7 / 2 :=
by
  sorry

end tenth_term_arithmetic_seq_l210_210281


namespace second_tap_empty_time_l210_210604

theorem second_tap_empty_time :
  ∃ T : ℝ, (1 / 4 - 1 / T = 3 / 28) → T = 7 :=
by
  sorry

end second_tap_empty_time_l210_210604


namespace find_c_plus_one_over_b_l210_210250

theorem find_c_plus_one_over_b 
  (a b c : ℝ) 
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (c_pos : 0 < c) 
  (h1 : a * b * c = 1) 
  (h2 : a + 1 / c = 8) 
  (h3 : b + 1 / a = 20) : 
  c + 1 / b = 10 / 53 := 
sorry

end find_c_plus_one_over_b_l210_210250


namespace total_number_of_athletes_l210_210804

theorem total_number_of_athletes (n : ℕ) (h1 : n % 10 = 6) (h2 : n % 11 = 6) (h3 : 200 ≤ n) (h4 : n ≤ 300) : n = 226 :=
sorry

end total_number_of_athletes_l210_210804


namespace P_ne_77_for_integers_l210_210391

def P (x y : ℤ) : ℤ :=
  x^5 - 4 * x^4 * y - 5 * y^2 * x^3 + 20 * y^3 * x^2 + 4 * y^4 * x - 16 * y^5

theorem P_ne_77_for_integers (x y : ℤ) : P x y ≠ 77 :=
by
  sorry

end P_ne_77_for_integers_l210_210391


namespace prob_two_red_two_blue_is_3_over_14_l210_210889

def red_marbles : ℕ := 15
def blue_marbles : ℕ := 10
def total_marbles : ℕ := red_marbles + blue_marbles
def chosen_marbles : ℕ := 4

noncomputable def prob_two_red_two_blue : ℚ :=
  let total_ways := (Nat.choose total_marbles chosen_marbles : ℚ)
  let ways_two_red := (Nat.choose red_marbles 2)
  let ways_two_blue := (Nat.choose blue_marbles 2)
  let favorable_outcomes := 6 * ways_two_red * ways_two_blue
  favorable_outcomes / total_ways

theorem prob_two_red_two_blue_is_3_over_14 : prob_two_red_two_blue = 3 / 14 :=
  sorry

end prob_two_red_two_blue_is_3_over_14_l210_210889


namespace sam_mary_total_balloons_l210_210654

def Sam_initial_balloons : ℝ := 6.0
def Sam_gives : ℝ := 5.0
def Sam_remaining_balloons : ℝ := Sam_initial_balloons - Sam_gives

def Mary_balloons : ℝ := 7.0

def total_balloons : ℝ := Sam_remaining_balloons + Mary_balloons

theorem sam_mary_total_balloons : total_balloons = 8.0 :=
by
  sorry

end sam_mary_total_balloons_l210_210654


namespace range_of_a_l210_210587

theorem range_of_a (a : ℝ) : ({x : ℝ | a - 4 < x ∧ x < a + 4} ⊆ {x : ℝ | 1 < x ∧ x < 3}) → (-1 ≤ a ∧ a ≤ 5) := by
  sorry

end range_of_a_l210_210587


namespace zachary_pushups_l210_210130

theorem zachary_pushups (david_pushups zachary_pushups : ℕ) (h₁ : david_pushups = 44) (h₂ : david_pushups = zachary_pushups + 9) :
  zachary_pushups = 35 :=
by
  sorry

end zachary_pushups_l210_210130


namespace card_sequence_probability_l210_210873

noncomputable def probability_of_sequence : ℚ :=
  (4/52) * (4/51) * (4/50)

theorem card_sequence_probability :
  probability_of_sequence = 4/33150 := 
by 
  sorry

end card_sequence_probability_l210_210873


namespace find_x_l210_210550

theorem find_x (x : ℝ) : 
  (1 + x) * 0.20 = x * 0.4 → x = 1 :=
by
  intros h
  sorry

end find_x_l210_210550


namespace log5_square_simplification_l210_210175

noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

theorem log5_square_simplification : (log5 (7 * log5 25))^2 = (log5 14)^2 :=
by
  sorry

end log5_square_simplification_l210_210175


namespace combined_rate_last_year_l210_210288

noncomputable def combine_effective_rate_last_year (r_increased: ℝ) (r_this_year: ℝ) : ℝ :=
  r_this_year / r_increased

theorem combined_rate_last_year
  (compounding_frequencies : List String)
  (r_increased : ℝ)
  (r_this_year : ℝ)
  (combined_interest_rate_this_year : r_this_year = 0.11)
  (interest_rate_increase : r_increased = 1.10) :
  combine_effective_rate_last_year r_increased r_this_year = 0.10 :=
by
  sorry

end combined_rate_last_year_l210_210288


namespace a719_divisible_by_11_l210_210138

theorem a719_divisible_by_11 (a : ℕ) (h : a < 10) : (∃ k : ℤ, a - 15 = 11 * k) ↔ a = 4 :=
by
  sorry

end a719_divisible_by_11_l210_210138


namespace parabola_y_relation_l210_210235

-- Conditions of the problem
def parabola (x : ℝ) (c : ℝ) : ℝ := 2 * x^2 - 4 * x + c

-- The proof problem statement
theorem parabola_y_relation (c y1 y2 y3 : ℝ) :
  parabola (-4) c = y1 →
  parabola (-2) c = y2 →
  parabola (1 / 2) c = y3 →
  y1 > y2 ∧ y2 > y3 :=
by
  sorry

end parabola_y_relation_l210_210235


namespace a_received_share_l210_210040

def a_inv : ℕ := 7000
def b_inv : ℕ := 11000
def c_inv : ℕ := 18000

def b_share : ℕ := 2200

def total_profit : ℕ := (b_share / (b_inv / 1000)) * 36
def a_ratio : ℕ := a_inv / 1000
def total_ratio : ℕ := (a_inv / 1000) + (b_inv / 1000) + (c_inv / 1000)

def a_share : ℕ := (a_ratio / total_ratio) * total_profit

theorem a_received_share :
  a_share = 1400 := 
sorry

end a_received_share_l210_210040


namespace Ben_more_new_shirts_than_Joe_l210_210751

theorem Ben_more_new_shirts_than_Joe :
  ∀ (alex_shirts joe_shirts ben_shirts : ℕ),
    alex_shirts = 4 →
    joe_shirts = alex_shirts + 3 →
    ben_shirts = 15 →
    ben_shirts - joe_shirts = 8 :=
by
  intros alex_shirts joe_shirts ben_shirts
  intros h_alex h_joe h_ben
  sorry

end Ben_more_new_shirts_than_Joe_l210_210751


namespace set_intersection_l210_210407

def A (x : ℝ) : Prop := -1 ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 3
def B (x : ℝ) : Prop := (x + 1) / x ≤ 0
def C_x_B (x : ℝ) : Prop := x < -1 ∨ x ≥ 0

theorem set_intersection :
  {x : ℝ | A x} ∩ {x : ℝ | C_x_B x} = {x : ℝ | 0 ≤ x ∧ x ≤ 1} :=
sorry

end set_intersection_l210_210407


namespace monotonic_intervals_max_value_of_k_l210_210539

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - 2
noncomputable def f_prime (x a : ℝ) : ℝ := Real.exp x - a

theorem monotonic_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ a < f x₂ a) ∧
  (a > 0 → ∀ x₁ x₂ : ℝ,
    x₁ < x₂ → (x₁ < Real.log a → f x₁ a > f x₂ a) ∧ (x₁ > Real.log a → f x₁ a < f x₂ a)) :=
sorry

theorem max_value_of_k (x : ℝ) (k : ℤ) (a : ℝ) (h_a : a = 1)
  (h : ∀ x > 0, (x - k) * f_prime x a + x + 1 > 0) :
  k ≤ 2 :=
sorry

end monotonic_intervals_max_value_of_k_l210_210539


namespace volume_of_polyhedron_l210_210934

theorem volume_of_polyhedron (V : ℝ) (hV : 0 ≤ V) :
  ∃ P : ℝ, P = V / 6 :=
by
  sorry

end volume_of_polyhedron_l210_210934


namespace probability_at_least_one_multiple_of_4_l210_210789

theorem probability_at_least_one_multiple_of_4 :
  let bound := 50
  let multiples_of_4 := 12
  let probability_no_multiple_of_4 := (38 / 50) * (38 / 50)
  let probability_at_least_one_multiple_of_4 := 1 - probability_no_multiple_of_4
  (probability_at_least_one_multiple_of_4 = 528 / 1250) := 
by
  -- Define the conditions
  let bound := 50
  let multiples_of_4 := 12
  let probability_no_multiple_of_4 := (38 / 50) * (38 / 50)
  let probability_at_least_one_multiple_of_4 := 1 - probability_no_multiple_of_4
  sorry

end probability_at_least_one_multiple_of_4_l210_210789


namespace line_through_parabola_vertex_unique_value_l210_210305

theorem line_through_parabola_vertex_unique_value :
  ∃! a : ℝ, ∃ y : ℝ, y = x + a ∧ y = x^2 - 2*a*x + a^2 :=
sorry

end line_through_parabola_vertex_unique_value_l210_210305


namespace range_of_a_l210_210648

theorem range_of_a (a x : ℝ) (h : x - a = 1 - 2*x) (non_neg_x : x ≥ 0) : a ≥ -1 := by
  sorry

end range_of_a_l210_210648


namespace find_real_number_a_l210_210284

variable (U : Set ℕ) (M : Set ℕ) (a : ℕ)

theorem find_real_number_a :
  U = {1, 3, 5, 7} →
  M = {1, a} →
  (U \ M) = {5, 7} →
  a = 3 :=
by
  intros hU hM hCompU
  -- Proof part will be here
  sorry

end find_real_number_a_l210_210284


namespace sum_50th_set_correct_l210_210444

noncomputable def sum_of_fiftieth_set : ℕ := 195 + 197

theorem sum_50th_set_correct : sum_of_fiftieth_set = 392 :=
by 
  -- The proof would go here
  sorry

end sum_50th_set_correct_l210_210444


namespace roots_of_polynomial_equation_l210_210356

theorem roots_of_polynomial_equation (x : ℝ) :
  4 * x ^ 4 - 21 * x ^ 3 + 34 * x ^ 2 - 21 * x + 4 = 0 ↔ x = 4 ∨ x = 1 / 4 ∨ x = 1 :=
by
  sorry

end roots_of_polynomial_equation_l210_210356


namespace cost_of_items_l210_210816

theorem cost_of_items (x y z : ℝ)
  (h1 : 20 * x + 3 * y + 2 * z = 32)
  (h2 : 39 * x + 5 * y + 3 * z = 58) :
  5 * (x + y + z) = 30 := by
  sorry

end cost_of_items_l210_210816


namespace sqrt_ineq_l210_210442

open Real

theorem sqrt_ineq (α β : ℝ) (hα : 1 ≤ α) (hβ : 1 ≤ β) :
  Int.floor (sqrt α) + Int.floor (sqrt (α + β)) + Int.floor (sqrt β) ≥
    Int.floor (sqrt (2 * α)) + Int.floor (sqrt (2 * β)) := by sorry

end sqrt_ineq_l210_210442


namespace proposition_p_and_not_q_is_true_l210_210009

-- Define proposition p
def p : Prop := ∀ x > 0, Real.log (x + 1) > 0

-- Define proposition q
def q : Prop := ∀ a b : Real, a > b → a^2 > b^2

-- State the theorem to be proven in Lean
theorem proposition_p_and_not_q_is_true : p ∧ ¬q :=
by
  -- Sorry placeholder for the proof
  sorry

end proposition_p_and_not_q_is_true_l210_210009


namespace Tom_final_balance_l210_210559

theorem Tom_final_balance :
  let initial_allowance := 12
  let week1_spending := initial_allowance / 3
  let balance_after_week1 := initial_allowance - week1_spending
  let week2_spending := balance_after_week1 / 4
  let balance_after_week2 := balance_after_week1 - week2_spending
  let additional_earning := 5
  let balance_after_earning := balance_after_week2 + additional_earning
  let week3_spending := balance_after_earning / 2
  let balance_after_week3 := balance_after_earning - week3_spending
  let penultimate_day_spending := 3
  let final_balance := balance_after_week3 - penultimate_day_spending
  final_balance = 2.50 :=
by
  sorry

end Tom_final_balance_l210_210559


namespace hadassah_painting_time_l210_210455

noncomputable def time_to_paint_all_paintings (time_small_paintings time_large_paintings time_additional_small_paintings time_additional_large_paintings : ℝ) : ℝ :=
  time_small_paintings + time_large_paintings + time_additional_small_paintings + time_additional_large_paintings

theorem hadassah_painting_time :
  let time_small_paintings := 6
  let time_large_paintings := 8
  let time_per_small_painting := 6 / 12 -- = 0.5
  let time_per_large_painting := 8 / 6 -- ≈ 1.33
  let time_additional_small_paintings := 15 * time_per_small_painting -- = 7.5
  let time_additional_large_paintings := 10 * time_per_large_painting -- ≈ 13.3
  time_to_paint_all_paintings time_small_paintings time_large_paintings time_additional_small_paintings time_additional_large_paintings = 34.8 :=
by
  sorry

end hadassah_painting_time_l210_210455


namespace geometric_sequence_result_l210_210494

-- Definitions representing the conditions
variables {a : ℕ → ℝ}

-- Conditions
axiom cond1 : a 7 * a 11 = 6
axiom cond2 : a 4 + a 14 = 5

theorem geometric_sequence_result :
  ∃ x, x = a 20 / a 10 ∧ (x = 2 / 3 ∨ x = 3 / 2) :=
by {
  sorry
}

end geometric_sequence_result_l210_210494


namespace intersection_is_3_l210_210158

def setA : Set ℕ := {5, 2, 3}
def setB : Set ℕ := {9, 3, 6}

theorem intersection_is_3 : setA ∩ setB = {3} := by
  sorry

end intersection_is_3_l210_210158


namespace prudence_sleep_4_weeks_equals_200_l210_210497

-- Conditions
def sunday_to_thursday_sleep := 6 
def friday_saturday_sleep := 9 
def nap := 1 

-- Number of days in the mentioned periods per week
def sunday_to_thursday_days := 5
def friday_saturday_days := 2
def nap_days := 2

-- Calculate total sleep per week
def total_sleep_per_week : Nat :=
  (sunday_to_thursday_days * sunday_to_thursday_sleep) +
  (friday_saturday_days * friday_saturday_sleep) +
  (nap_days * nap)

-- Calculate total sleep in 4 weeks
def total_sleep_in_4_weeks : Nat :=
  4 * total_sleep_per_week

theorem prudence_sleep_4_weeks_equals_200 : total_sleep_in_4_weeks = 200 := by
  sorry

end prudence_sleep_4_weeks_equals_200_l210_210497


namespace trapezoid_midsegment_l210_210074

theorem trapezoid_midsegment (a b : ℝ)
  (AB CD E F: ℝ) -- we need to indicate that E and F are midpoints somehow
  (h1 : AB = a)
  (h2 : CD = b)
  (h3 : AB = CD) 
  (h4 : E = (AB + CD) / 2)
  (h5 : F = (CD + AB) / 2) : 
  EF = (1/2) * (a - b) := sorry

end trapezoid_midsegment_l210_210074


namespace max_value_2ab_plus_2bc_sqrt2_l210_210896

theorem max_value_2ab_plus_2bc_sqrt2 (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a^2 + b^2 + c^2 = 1) :
  2 * a * b + 2 * b * c * Real.sqrt 2 ≤ Real.sqrt 3 :=
sorry

end max_value_2ab_plus_2bc_sqrt2_l210_210896


namespace find_x_l210_210028

theorem find_x :
  ∀ (x : ℝ), 4.7 * x + 4.7 * 9.43 + 4.7 * 77.31 = 470 → x = 13.26 :=
by
  intro x
  intro h
  sorry

end find_x_l210_210028


namespace average_of_abc_l210_210132

theorem average_of_abc (A B C : ℚ) 
  (h1 : 2002 * C + 4004 * A = 8008) 
  (h2 : 3003 * B - 5005 * A = 7007) : 
  (A + B + C) / 3 = 22 / 9 := 
by 
  sorry

end average_of_abc_l210_210132


namespace solution_difference_l210_210851

theorem solution_difference (m n : ℝ) (h_eq : ∀ x : ℝ, (x - 4) * (x + 4) = 24 * x - 96 ↔ x = m ∨ x = n) (h_distinct : m ≠ n) (h_order : m > n) : m - n = 16 :=
sorry

end solution_difference_l210_210851


namespace hunter_ants_l210_210262

variable (spiders : ℕ) (ladybugs_before : ℕ) (ladybugs_flew : ℕ) (total_insects : ℕ)

theorem hunter_ants (h1 : spiders = 3)
                    (h2 : ladybugs_before = 8)
                    (h3 : ladybugs_flew = 2)
                    (h4 : total_insects = 21) :
  ∃ ants : ℕ, ants = total_insects - (spiders + (ladybugs_before - ladybugs_flew)) ∧ ants = 12 :=
by
  sorry

end hunter_ants_l210_210262


namespace new_profit_percentage_l210_210042

theorem new_profit_percentage (P : ℝ) (h1 : 1.10 * P = 990) (h2 : 0.90 * P * (1 + 0.30) = 1053) : 0.30 = 0.30 :=
by sorry

end new_profit_percentage_l210_210042


namespace min_length_l210_210895

def length (a b : ℝ) : ℝ := b - a

noncomputable def M (m : ℝ) := {x | m ≤ x ∧ x ≤ m + 3 / 4}
noncomputable def N (n : ℝ) := {x | n - 1 / 3 ≤ x ∧ x ≤ n}
noncomputable def intersection (m n : ℝ) := {x | max m (n - 1 / 3) ≤ x ∧ x ≤ min (m + 3 / 4) n}

theorem min_length (m n : ℝ) (hM : ∀ x, x ∈ M m → 0 ≤ x ∧ x ≤ 1) (hN : ∀ x, x ∈ N n → 0 ≤ x ∧ x ≤ 1) :
  length (max m (n - 1 / 3)) (min (m + 3 / 4) n) = 1 / 12 :=
sorry

end min_length_l210_210895


namespace number_of_buses_l210_210198

theorem number_of_buses (x y : ℕ) (h1 : x + y = 40) (h2 : 6 * x + 4 * y = 210) : x = 25 :=
by
  sorry

end number_of_buses_l210_210198


namespace sum_first_10_terms_l210_210925

def arithmetic_sequence (a d : Int) (n : Int) : Int :=
  a + (n - 1) * d

def arithmetic_sum (a d : Int) (n : Int) : Int :=
  (n : Int) * a + (n * (n - 1) / 2) * d

theorem sum_first_10_terms  
  (a d : Int)
  (h1 : (a + 3 * d)^2 = (a + 2 * d) * (a + 6 * d))
  (h2 : arithmetic_sum a d 8 = 32)
  : arithmetic_sum a d 10 = 60 :=
sorry

end sum_first_10_terms_l210_210925


namespace fraction_simplification_l210_210627

theorem fraction_simplification (x : ℝ) (h : x = Real.sqrt 2) : 
  ( (x^2 - 1) / (x^2 - x) - 1) = Real.sqrt 2 / 2 :=
by 
  sorry

end fraction_simplification_l210_210627


namespace vote_count_l210_210509

theorem vote_count 
(h_total: 200 = h_votes + l_votes + y_votes)
(h_hl: 3 * l_votes = 2 * h_votes)
(l_ly: 6 * y_votes = 5 * l_votes):
h_votes = 90 ∧ l_votes = 60 ∧ y_votes = 50 := by 
sorry

end vote_count_l210_210509


namespace arithmetic_sequence_sum_nine_l210_210207

variable {a : ℕ → ℤ} -- Define a_n sequence as a function from ℕ to ℤ

-- Define the conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ (d : ℤ), ∀ n m, a (n + m) = a n + m * d

def fifth_term_is_two (a : ℕ → ℤ) : Prop :=
  a 5 = 2

-- Lean statement to prove the sum of the first 9 terms
theorem arithmetic_sequence_sum_nine (a : ℕ → ℤ)
  (h1 : is_arithmetic_sequence a)
  (h2 : fifth_term_is_two a) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 2 * 9 :=
sorry

end arithmetic_sequence_sum_nine_l210_210207


namespace vibrations_proof_l210_210111

-- Define the conditions
def vibrations_lowest : ℕ := 1600
def increase_percentage : ℕ := 60
def use_time_minutes : ℕ := 5

-- Convert percentage to a multiplier
def percentage_to_multiplier (p : ℕ) : ℤ := (p : ℤ) / 100

-- Calculate the vibrations per second at the highest setting
def vibrations_highest := vibrations_lowest + (vibrations_lowest * percentage_to_multiplier increase_percentage).toNat

-- Convert time from minutes to seconds
def use_time_seconds := use_time_minutes * 60

-- Calculate the total vibrations Matt experiences
noncomputable def total_vibrations : ℕ := vibrations_highest * use_time_seconds

-- State the theorem
theorem vibrations_proof : total_vibrations = 768000 := 
by
  sorry

end vibrations_proof_l210_210111


namespace find_m_l210_210005

noncomputable def is_power_function (y : ℝ → ℝ) := 
  ∃ (c : ℝ), ∃ (n : ℝ), ∀ x : ℝ, y x = c * x ^ n

theorem find_m (m : ℝ) :
  (∀ x : ℝ, (∃ c : ℝ, (m^2 - 2 * m + 1) * x^(m - 1) = c * x^n) ∧ (∀ x : ℝ, true)) → m = 2 :=
sorry

end find_m_l210_210005


namespace machine_a_production_rate_l210_210197

/-
Given:
1. Machine p and machine q are each used to manufacture 440 sprockets.
2. Machine q produces 10% more sprockets per hour than machine a.
3. It takes machine p 10 hours longer to produce 440 sprockets than machine q.

Prove that machine a produces 4 sprockets per hour.
-/

theorem machine_a_production_rate (T A : ℝ) (hq : 440 = T * (1.1 * A)) (hp : 440 = (T + 10) * A) : A = 4 := 
by
  sorry

end machine_a_production_rate_l210_210197


namespace students_exceed_hamsters_l210_210091

-- Definitions corresponding to the problem conditions
def students_per_classroom : ℕ := 20
def hamsters_per_classroom : ℕ := 1
def number_of_classrooms : ℕ := 5

-- Lean 4 statement to express the problem
theorem students_exceed_hamsters :
  (students_per_classroom * number_of_classrooms) - (hamsters_per_classroom * number_of_classrooms) = 95 :=
by
  sorry

end students_exceed_hamsters_l210_210091


namespace people_joined_after_leaving_l210_210848

theorem people_joined_after_leaving 
  (p_initial : ℕ) (p_left : ℕ) (p_final : ℕ) (p_joined : ℕ) :
  p_initial = 30 → p_left = 10 → p_final = 25 → p_joined = p_final - (p_initial - p_left) → p_joined = 5 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end people_joined_after_leaving_l210_210848


namespace bert_spent_fraction_at_hardware_store_l210_210945

variable (f : ℝ)

def initial_money : ℝ := 41.99
def after_hardware (f : ℝ) := (1 - f) * initial_money
def after_dry_cleaners (f : ℝ) := after_hardware f - 7
def after_grocery (f : ℝ) := 0.5 * after_dry_cleaners f

theorem bert_spent_fraction_at_hardware_store 
(h1 : after_grocery f = 10.50) : 
  f = 0.3332 :=
by
  sorry

end bert_spent_fraction_at_hardware_store_l210_210945


namespace greatest_A_satisfies_condition_l210_210635

theorem greatest_A_satisfies_condition :
  ∃ (A : ℝ), A = 64 ∧ ∀ (s : Fin₇ → ℝ), (∀ i, 1 ≤ s i ∧ s i ≤ A) →
  ∃ (i j : Fin₇), i ≠ j ∧ (1 / 2 ≤ s i / s j ∧ s i / s j ≤ 2) :=
by 
  sorry

end greatest_A_satisfies_condition_l210_210635


namespace nonneg_int_solutions_eqn_l210_210467

theorem nonneg_int_solutions_eqn :
  { (x, y, z, w) : ℕ × ℕ × ℕ × ℕ | 2^x * 3^y - 5^z * 7^w = 1 } =
  {(1, 0, 0, 0), (3, 0, 0, 1), (1, 1, 1, 0), (2, 2, 1, 1)} :=
by {
  sorry
}

end nonneg_int_solutions_eqn_l210_210467


namespace find_n_for_sum_l210_210863

theorem find_n_for_sum (n : ℕ) : ∃ n, n * (2 * n - 1) = 2009 ^ 2 :=
by
  sorry

end find_n_for_sum_l210_210863


namespace books_sold_l210_210306

-- Define the conditions
def initial_books : ℕ := 134
def books_given_away : ℕ := 39
def remaining_books : ℕ := 68

-- Define the intermediate calculation of books left after giving away
def books_after_giving_away : ℕ := initial_books - books_given_away

-- Prove the number of books sold
theorem books_sold (initial_books books_given_away remaining_books : ℕ) (h1 : books_after_giving_away = 95) (h2 : remaining_books = 68) :
  (books_after_giving_away - remaining_books) = 27 :=
by
  sorry

end books_sold_l210_210306


namespace smallest_tree_height_l210_210205

theorem smallest_tree_height (tallest middle smallest : ℝ)
  (h1 : tallest = 108)
  (h2 : middle = (tallest / 2) - 6)
  (h3 : smallest = middle / 4) : smallest = 12 :=
by
  sorry

end smallest_tree_height_l210_210205


namespace problems_left_to_grade_l210_210581

-- Definitions based on provided conditions
def problems_per_worksheet : ℕ := 4
def total_worksheets : ℕ := 16
def graded_worksheets : ℕ := 8

-- The statement for the required proof with the correct answer included
theorem problems_left_to_grade : 4 * (16 - 8) = 32 := by
  sorry

end problems_left_to_grade_l210_210581


namespace expected_value_is_correct_l210_210927

-- Define the probabilities of heads and tails
def P_H := 2 / 5
def P_T := 3 / 5

-- Define the winnings for heads and the loss for tails
def W_H := 5
def L_T := -4

-- Calculate the expected value
def expected_value := P_H * W_H + P_T * L_T

-- Prove that the expected value is -2/5
theorem expected_value_is_correct : expected_value = -2 / 5 := by
  sorry

end expected_value_is_correct_l210_210927


namespace point_B_number_l210_210451

theorem point_B_number (A B : ℤ) (hA : A = -2) (hB : abs (B - A) = 3) : B = 1 ∨ B = -5 :=
sorry

end point_B_number_l210_210451


namespace ninth_term_arithmetic_sequence_l210_210685

theorem ninth_term_arithmetic_sequence 
  (a1 a17 d a9 : ℚ) 
  (h1 : a1 = 2 / 3) 
  (h17 : a17 = 3 / 2) 
  (h_formula : a17 = a1 + 16 * d) 
  (h9_formula : a9 = a1 + 8 * d) :
  a9 = 13 / 12 := by
  sorry

end ninth_term_arithmetic_sequence_l210_210685


namespace solution_set_for_inequality_l210_210095

open Set Real

theorem solution_set_for_inequality : 
  { x : ℝ | (2 * x) / (x + 1) ≤ 1 } = Ioc (-1 : ℝ) 1 := 
sorry

end solution_set_for_inequality_l210_210095


namespace part1_part2_find_min_value_l210_210495

open Real

-- Proof of Part 1
theorem part1 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : a^2 / b + b^2 / a ≥ a + b :=
by sorry

-- Proof of Part 2
theorem part2 (x : ℝ) (hx : 0 < x) (hx1 : x < 1) : (1 - x)^2 / x + x^2 / (1 - x) ≥ 1 :=
by sorry

-- Corollary to find the minimum value
theorem find_min_value (x : ℝ) (hx : 0 < x) (hx1 : x < 1) : (1 - x)^2 / x + x^2 / (1 - x) = 1 ↔ x = 1 / 2 :=
by sorry

end part1_part2_find_min_value_l210_210495


namespace number_of_truthful_dwarfs_is_correct_l210_210827

-- Definitions and assumptions based on the given conditions
def x : ℕ := 4 -- number of truthful dwarfs
def y : ℕ := 6 -- number of lying dwarfs

-- Conditions
axiom total_dwarfs : x + y = 10
axiom total_hands_raised : x + 2 * y = 16

-- The proof statement
theorem number_of_truthful_dwarfs_is_correct : x = 4 := by
  have h1 : x + y = 10 := total_dwarfs
  have h2 : x + 2 * y = 16 := total_hands_raised
  sorry -- The proof follows from solving the system of equations


end number_of_truthful_dwarfs_is_correct_l210_210827


namespace solve_system_equations_l210_210747

theorem solve_system_equations (a b c x y z : ℝ) (h1 : x + y + z = 0)
(h2 : c * x + a * y + b * z = 0)
(h3 : (x + b)^2 + (y + c)^2 + (z + a)^2 = a^2 + b^2 + c^2)
(h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a) : 
(x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = a - b ∧ y = b - c ∧ z = c - a) := 
sorry

end solve_system_equations_l210_210747


namespace tricycles_count_l210_210842

theorem tricycles_count (b t : ℕ) 
  (hyp1 : b + t = 10)
  (hyp2 : 2 * b + 3 * t = 26) : 
  t = 6 := 
by 
  sorry

end tricycles_count_l210_210842


namespace ratio_eliminated_to_remaining_l210_210161

theorem ratio_eliminated_to_remaining (initial_racers : ℕ) (final_racers : ℕ)
  (eliminations_1st_segment : ℕ) (eliminations_2nd_segment : ℕ) :
  initial_racers = 100 →
  final_racers = 30 →
  eliminations_1st_segment = 10 →
  eliminations_2nd_segment = initial_racers - eliminations_1st_segment - (initial_racers - eliminations_1st_segment) / 3 - final_racers →
  (eliminations_2nd_segment / (initial_racers - eliminations_1st_segment - (initial_racers - eliminations_1st_segment) / 3)) = 1 / 2 :=
by
  sorry

end ratio_eliminated_to_remaining_l210_210161


namespace sequence_geq_four_l210_210446

theorem sequence_geq_four (a : ℕ → ℝ) (h0 : a 1 = 5) 
    (h1 : ∀ n ≥ 1, a (n+1) = (a n ^ 2 + 8 * a n + 16) / (4 * a n)) : 
    ∀ n ≥ 1, a n ≥ 4 := 
by
  sorry

end sequence_geq_four_l210_210446


namespace find_ordered_pair_l210_210234

theorem find_ordered_pair (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n) :
  18 * m * n = 72 - 9 * m - 4 * n ↔ (m = 8 ∧ n = 36) := 
by 
  sorry

end find_ordered_pair_l210_210234


namespace student_marks_l210_210608

theorem student_marks
(M P C : ℕ) -- the marks of Mathematics, Physics, and Chemistry are natural numbers
(h1 : C = P + 20)  -- Chemistry is 20 marks more than Physics
(h2 : (M + C) / 2 = 30)  -- The average marks in Mathematics and Chemistry is 30
: M + P = 40 := 
sorry

end student_marks_l210_210608


namespace study_time_difference_l210_210710

def kwame_study_time : ℕ := 150
def connor_study_time : ℕ := 90
def lexia_study_time : ℕ := 97
def michael_study_time : ℕ := 225
def cassandra_study_time : ℕ := 165
def aria_study_time : ℕ := 720

theorem study_time_difference :
  (kwame_study_time + connor_study_time + michael_study_time + cassandra_study_time) + 187 = (lexia_study_time + aria_study_time) :=
by
  sorry

end study_time_difference_l210_210710


namespace evaluate_g_inv_l210_210001

variable (g : ℕ → ℕ)
variable (g_inv : ℕ → ℕ)
variable (h1 : g 4 = 6)
variable (h2 : g 6 = 2)
variable (h3 : g 3 = 7)
variable (h_inv1 : g_inv 6 = 4)
variable (h_inv2 : g_inv 7 = 3)
variable (h_inv_eq : ∀ x, g (g_inv x) = x ∧ g_inv (g x) = x)

theorem evaluate_g_inv :
  g_inv (g_inv 6 + g_inv 7) = 3 :=
by
  sorry

end evaluate_g_inv_l210_210001


namespace boys_neither_happy_nor_sad_l210_210400

theorem boys_neither_happy_nor_sad (total_children : ℕ)
  (happy_children sad_children neither_happy_nor_sad total_boys total_girls : ℕ)
  (happy_boys sad_girls : ℕ)
  (h_total : total_children = 60)
  (h_happy : happy_children = 30)
  (h_sad : sad_children = 10)
  (h_neither : neither_happy_nor_sad = 20)
  (h_boys : total_boys = 17)
  (h_girls : total_girls = 43)
  (h_happy_boys : happy_boys = 6)
  (h_sad_girls : sad_girls = 4) :
  ∃ (boys_neither_happy_nor_sad : ℕ), boys_neither_happy_nor_sad = 5 := by
  sorry

end boys_neither_happy_nor_sad_l210_210400


namespace min_value_of_expression_l210_210850

theorem min_value_of_expression (a : ℝ) (h : a > 1) : a + (1 / (a - 1)) ≥ 3 :=
by sorry

end min_value_of_expression_l210_210850


namespace relation_w_z_relation_s_t_relation_x_r_relation_y_q_relation_z_x_t_relation_z_t_v_l210_210103

-- Prove that w - 2z = 0
theorem relation_w_z (w z : ℝ) : w - 2 * z = 0 :=
sorry

-- Prove that 2s + t - 8 = 0
theorem relation_s_t (s t : ℝ) : 2 * s + t - 8 = 0 :=
sorry

-- Prove that x - r - 2 = 0
theorem relation_x_r (x r : ℝ) : x - r - 2 = 0 :=
sorry

-- Prove that y + q - 6 = 0
theorem relation_y_q (y q : ℝ) : y + q - 6 = 0 :=
sorry

-- Prove that 3z - x - 2t + 6 = 0
theorem relation_z_x_t (z x t : ℝ) : 3 * z - x - 2 * t + 6 = 0 :=
sorry

-- Prove that 8z - 4t - v + 12 = 0
theorem relation_z_t_v (z t v : ℝ) : 8 * z - 4 * t - v + 12 = 0 :=
sorry

end relation_w_z_relation_s_t_relation_x_r_relation_y_q_relation_z_x_t_relation_z_t_v_l210_210103


namespace smallest_positive_perfect_square_divisible_by_5_and_6_l210_210269

theorem smallest_positive_perfect_square_divisible_by_5_and_6 : 
  ∃ n : ℕ, (∃ m : ℕ, n = m * m) ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ (∀ k : ℕ, (∃ p : ℕ, k = p * p) ∧ k % 5 = 0 ∧ k % 6 = 0 → n ≤ k) := 
sorry

end smallest_positive_perfect_square_divisible_by_5_and_6_l210_210269


namespace carpet_dimensions_problem_l210_210681

def carpet_dimensions (width1 width2 : ℕ) (l : ℕ) :=
  ∃ x y : ℕ, width1 = 38 ∧ width2 = 50 ∧ l = l ∧ x = 25 ∧ y = 50

theorem carpet_dimensions_problem (l : ℕ) :
  carpet_dimensions 38 50 l :=
by
  sorry

end carpet_dimensions_problem_l210_210681


namespace johns_average_speed_l210_210854

-- Definitions of conditions
def total_time_hours : ℝ := 6.5
def total_distance_miles : ℝ := 255

-- Stating the problem to be proven
theorem johns_average_speed :
  (total_distance_miles / total_time_hours) = 39.23 := 
sorry

end johns_average_speed_l210_210854


namespace fourth_guard_distance_l210_210480

theorem fourth_guard_distance 
  (length : ℝ) (width : ℝ)
  (total_distance_three_guards: ℝ)
  (P : ℝ := 2 * (length + width)) 
  (total_distance_four_guards : ℝ := P)
  (total_three : total_distance_three_guards = 850)
  (length_value : length = 300)
  (width_value : width = 200) :
  ∃ distance_fourth_guard : ℝ, distance_fourth_guard = 150 :=
by 
  sorry

end fourth_guard_distance_l210_210480


namespace painted_cells_possible_values_l210_210079

theorem painted_cells_possible_values (k l : ℕ) (hk : 2 * k + 1 > 0) (hl : 2 * l + 1 > 0) (h : k * l = 74) :
  (2 * k + 1) * (2 * l + 1) - 74 = 301 ∨ (2 * k + 1) * (2 * l + 1) - 74 = 373 := 
sorry

end painted_cells_possible_values_l210_210079


namespace quadratic_no_real_roots_l210_210506

-- Given conditions
variables {p q a b c : ℝ}
variables (hp_pos : 0 < p) (hq_pos : 0 < q) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
variables (hp_neq_q : p ≠ q)

-- p, a, q form a geometric sequence
variables (h_geo : a^2 = p * q)

-- p, b, c, q form an arithmetic sequence
variables (h_arith1 : 2 * b = p + c)
variables (h_arith2 : 2 * c = b + q)

-- Proof statement
theorem quadratic_no_real_roots (hp_pos hq_pos ha_pos hb_pos hc_pos hp_neq_q h_geo h_arith1 h_arith2 : ℝ) :
    (b * (x : ℝ)^2 - 2 * a * x + c = 0) → false :=
sorry

end quadratic_no_real_roots_l210_210506


namespace arithmetic_geometric_mean_l210_210621

theorem arithmetic_geometric_mean (a b : ℝ) 
  (h1 : (a + b) / 2 = 20) 
  (h2 : Real.sqrt (a * b) = Real.sqrt 135) : 
  a^2 + b^2 = 1330 :=
by
  sorry

end arithmetic_geometric_mean_l210_210621


namespace solve_system_eq_l210_210152

theorem solve_system_eq (x y z : ℝ) 
  (h1 : x * y = 6 * (x + y))
  (h2 : x * z = 4 * (x + z))
  (h3 : y * z = 2 * (y + z)) :
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
  (x = -24 ∧ y = 24 / 5 ∧ z = 24 / 7) :=
  sorry

end solve_system_eq_l210_210152


namespace largest_integer_le_1_l210_210466

theorem largest_integer_le_1 (x : ℤ) (h : (2 * x : ℚ) / 7 + 3 / 4 < 8 / 7) : x ≤ 1 :=
sorry

end largest_integer_le_1_l210_210466


namespace roots_eqn_values_l210_210481

theorem roots_eqn_values : 
  ∀ (x1 x2 : ℝ), (x1^2 + x1 - 4 = 0) ∧ (x2^2 + x2 - 4 = 0) ∧ (x1 + x2 = -1)
  → (x1^3 - 5 * x2^2 + 10 = -19) := 
by
  intros x1 x2
  intros h
  sorry

end roots_eqn_values_l210_210481


namespace integer_roots_polynomial_l210_210545

theorem integer_roots_polynomial 
(m n : ℕ) (h_m_pos : m > 0) (h_n_pos : n > 0) :
  (∃ a b c : ℤ, a + b + c = 17 ∧ a * b * c = n^2 ∧ a * b + b * c + c * a = m) ↔ 
  (m, n) = (80, 10) ∨ (m, n) = (88, 12) ∨ (m, n) = (80, 8) ∨ (m, n) = (90, 12) := 
sorry

end integer_roots_polynomial_l210_210545


namespace min_candies_to_remove_l210_210551

theorem min_candies_to_remove {n : ℕ} (h : n = 31) : (∃ k, (n - k) % 5 = 0) → k = 1 :=
by
  sorry

end min_candies_to_remove_l210_210551


namespace gum_pieces_in_each_packet_l210_210573

theorem gum_pieces_in_each_packet
  (packets : ℕ) (chewed_pieces : ℕ) (remaining_pieces : ℕ) (total_pieces : ℕ)
  (h1 : packets = 8) (h2 : chewed_pieces = 54) (h3 : remaining_pieces = 2) (h4 : total_pieces = chewed_pieces + remaining_pieces)
  (h5 : total_pieces = packets * (total_pieces / packets)) :
  total_pieces / packets = 7 :=
by
  sorry

end gum_pieces_in_each_packet_l210_210573


namespace largest_by_changing_first_digit_l210_210730

-- Define the original number
def original_number : ℝ := 0.7162534

-- Define the transformation that changes a specific digit to 8
def transform_to_8 (n : ℕ) (d : ℝ) : ℝ :=
  match n with
  | 1 => 0.8162534
  | 2 => 0.7862534
  | 3 => 0.7182534
  | 4 => 0.7168534
  | 5 => 0.7162834
  | 6 => 0.7162584
  | 7 => 0.7162538
  | _ => d

-- State the theorem
theorem largest_by_changing_first_digit :
  ∀ (n : ℕ), transform_to_8 1 original_number ≥ transform_to_8 n original_number :=
by
  sorry

end largest_by_changing_first_digit_l210_210730


namespace distance_from_apex_l210_210004

theorem distance_from_apex (A B : ℝ)
  (h_A : A = 216 * Real.sqrt 3)
  (h_B : B = 486 * Real.sqrt 3)
  (distance_planes : ℝ)
  (h_distance_planes : distance_planes = 8) :
  ∃ h : ℝ, h = 24 :=
by
  sorry

end distance_from_apex_l210_210004


namespace adults_at_zoo_l210_210386

theorem adults_at_zoo (A K : ℕ) (h1 : A + K = 254) (h2 : 28 * A + 12 * K = 3864) : A = 51 :=
sorry

end adults_at_zoo_l210_210386


namespace product_form_l210_210593

theorem product_form (a b c d : ℤ) :
    (a^2 - 7*b^2) * (c^2 - 7*d^2) = (a*c + 7*b*d)^2 - 7*(a*d + b*c)^2 :=
by sorry

end product_form_l210_210593


namespace intersecting_lines_l210_210211

theorem intersecting_lines (a b : ℚ) :
  (3 = (1 / 3 : ℚ) * 4 + a) → 
  (4 = (1 / 2 : ℚ) * 3 + b) → 
  a + b = 25 / 6 :=
by
  intros h1 h2
  sorry

end intersecting_lines_l210_210211


namespace systematic_sampling_student_l210_210836

theorem systematic_sampling_student (total_students sample_size : ℕ) 
  (h_total_students : total_students = 56)
  (h_sample_size : sample_size = 4)
  (student1 student2 student3 student4 : ℕ)
  (h_student1 : student1 = 6)
  (h_student2 : student2 = 34)
  (h_student3 : student3 = 48) :
  student4 = 20 :=
sorry

end systematic_sampling_student_l210_210836


namespace y_relationship_l210_210617

variable (a c : ℝ) (h_a : a < 0)

def f (x : ℝ) : ℝ := a * (x - 3) ^ 2 + c

theorem y_relationship (y1 y2 y3 : ℝ)
  (h1 : y1 = f a c (Real.sqrt 5))
  (h2 : y2 = f a c 0)
  (h3 : y3 = f a c 4) :
  y2 < y3 ∧ y3 < y1 :=
  sorry

end y_relationship_l210_210617


namespace movie_theorem_l210_210640

variables (A B C D : Prop)

theorem movie_theorem 
  (h1 : (A → B))
  (h2 : (B → C))
  (h3 : (C → A))
  (h4 : (D → B)) 
  : ¬D := 
by
  sorry

end movie_theorem_l210_210640


namespace diagonal_cannot_be_good_l210_210027

def is_good (table : ℕ → ℕ → ℕ) (i j : ℕ) :=
  ∀ x y, (x = i ∨ y = j) → ∀ x' y', (x' = i ∨ y' = j) → (x ≠ x' ∨ y ≠ y') → table x y ≠ table x' y'

theorem diagonal_cannot_be_good :
  ∀ (table : ℕ → ℕ → ℕ), (∀ i j, 1 ≤ table i j ∧ table i j ≤ 25) →
  ¬ ∀ k, (is_good table k k) :=
by
  sorry

end diagonal_cannot_be_good_l210_210027


namespace Darcy_remaining_clothes_l210_210663

/--
Darcy initially has 20 shirts and 8 pairs of shorts.
He folds 12 of the shirts and 5 of the pairs of shorts.
We want to prove that the total number of remaining pieces of clothing Darcy has to fold is 11.
-/
theorem Darcy_remaining_clothes
  (initial_shirts : Nat)
  (initial_shorts : Nat)
  (folded_shirts : Nat)
  (folded_shorts : Nat)
  (remaining_shirts : Nat)
  (remaining_shorts : Nat)
  (total_remaining : Nat) :
  initial_shirts = 20 → initial_shorts = 8 →
  folded_shirts = 12 → folded_shorts = 5 →
  remaining_shirts = initial_shirts - folded_shirts →
  remaining_shorts = initial_shorts - folded_shorts →
  total_remaining = remaining_shirts + remaining_shorts →
  total_remaining = 11 := by
  sorry

end Darcy_remaining_clothes_l210_210663


namespace simplify_expression_l210_210560

theorem simplify_expression (a b c : ℝ) (ha : a = 7.4) (hb : b = 5 / 37) :
  1.6 * ((1 / a + 1 / b - 2 * c / (a * b)) * (a + b + 2 * c)) / 
  ((1 / a^2 + 1 / b^2 + 2 / (a * b) - 4 * c^2 / (a^2 * b^2))) = 1.6 :=
by 
  rw [ha, hb] 
  sorry

end simplify_expression_l210_210560


namespace stratified_sampling_third_year_l210_210862

-- The total number of students in the school
def total_students : ℕ := 2000

-- The probability of selecting a female student from the second year
def prob_female_second_year : ℚ := 0.19

-- The number of students to be selected through stratified sampling
def sample_size : ℕ := 100

-- The total number of third-year students
def third_year_students : ℕ := 500

-- The number of students to be selected from the third year in stratified sampling
def third_year_sample (total : ℕ) (third_year : ℕ) (sample : ℕ) : ℕ :=
  sample * third_year / total

-- Lean statement expressing the goal
theorem stratified_sampling_third_year :
  third_year_sample total_students third_year_students sample_size = 25 :=
by
  sorry

end stratified_sampling_third_year_l210_210862


namespace calculate_expression_l210_210450

theorem calculate_expression (x y : ℕ) (hx : x = 3) (hy : y = 4) : 
  (1 / (y + 1)) / (1 / (x + 2)) = 1 := by
  sorry

end calculate_expression_l210_210450


namespace intersection_complement_A_B_l210_210694

open Set

variable (x : ℝ)

def U := ℝ
def A := {x | -2 ≤ x ∧ x ≤ 3}
def B := {x | x < -1 ∨ x > 4}

theorem intersection_complement_A_B :
  {x | -2 ≤ x ∧ x ≤ 3} ∩ compl {x | x < -1 ∨ x > 4} = {x | -1 ≤ x ∧ x ≤ 3} :=
by
  sorry

end intersection_complement_A_B_l210_210694


namespace find_a_find_cos_2C_l210_210903

noncomputable def triangle_side_a (A B : Real) (b : Real) (cosA : Real) : Real := 
  3

theorem find_a (A : Real) (B : Real) (b : Real) (cosA : Real) 
  (h₁ : b = 3 * Real.sqrt 2) 
  (h₂ : cosA = Real.sqrt 6 / 3) 
  (h₃ : B = A + Real.pi / 2) : 
  triangle_side_a A B b cosA = 3 := by
  sorry

noncomputable def cos_2C (A B C a b : Real) (cosA sinC : Real) : Real :=
  7 / 9

theorem find_cos_2C (A : Real) (B : Real) (C : Real) (a : Real) (b : Real) (cosA : Real) (sinC: Real)
  (h₁ : b = 3 * Real.sqrt 2) 
  (h₂ : cosA = Real.sqrt 6 / 3)
  (h₃ : B = A + Real.pi /2)
  (h₄ : a = 3)
  (h₅ : sinC = 1 / 3) :
  cos_2C A B C a b cosA sinC = 7 / 9 := by
  sorry

end find_a_find_cos_2C_l210_210903


namespace seats_per_section_correct_l210_210777

-- Define the total number of seats
def total_seats : ℕ := 270

-- Define the number of sections
def sections : ℕ := 9

-- Define the number of seats per section
def seats_per_section (total_seats sections : ℕ) : ℕ := total_seats / sections

theorem seats_per_section_correct : seats_per_section total_seats sections = 30 := by
  sorry

end seats_per_section_correct_l210_210777


namespace phone_call_probability_within_four_rings_l210_210888

variables (P_A P_B P_C P_D : ℝ)

-- Assuming given probabilities
def probabilities_given : Prop :=
  P_A = 0.1 ∧ P_B = 0.3 ∧ P_C = 0.4 ∧ P_D = 0.1

theorem phone_call_probability_within_four_rings (h : probabilities_given P_A P_B P_C P_D) :
  P_A + P_B + P_C + P_D = 0.9 :=
sorry

end phone_call_probability_within_four_rings_l210_210888


namespace scientific_notation_28400_is_correct_l210_210588

theorem scientific_notation_28400_is_correct : (28400 : ℝ) = 2.84 * 10^4 := 
by 
  sorry

end scientific_notation_28400_is_correct_l210_210588


namespace probability_none_solve_l210_210962

theorem probability_none_solve (a b c : ℕ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (h_prob : ((1 - (1/a)) * (1 - (1/b)) * (1 - (1/c)) = 8/15)) : 
  (1 - (1/a)) * (1 - (1/b)) * (1 - (1/c)) = 8/15 := 
by 
  sorry

end probability_none_solve_l210_210962


namespace divisibility_by_11_l210_210600

theorem divisibility_by_11 (m n : ℤ) (h : (5 * m + 3 * n) % 11 = 0) : (9 * m + n) % 11 = 0 := by
  sorry

end divisibility_by_11_l210_210600


namespace probability_in_interval_l210_210419

theorem probability_in_interval (a b c d : ℝ) (h1 : a = 2) (h2 : b = 10) (h3 : c = 5) (h4 : d = 7) :
  (d - c) / (b - a) = 1 / 4 :=
by
  sorry

end probability_in_interval_l210_210419


namespace total_spider_legs_l210_210916

-- Definition of the number of spiders
def number_of_spiders : ℕ := 5

-- Definition of the number of legs per spider
def legs_per_spider : ℕ := 8

-- Theorem statement to prove the total number of spider legs
theorem total_spider_legs : number_of_spiders * legs_per_spider = 40 :=
by 
  -- We've planned to use 'sorry' to skip the proof
  sorry

end total_spider_legs_l210_210916


namespace spots_combined_l210_210855

def Rover : ℕ := 46
def Cisco : ℕ := Rover / 2 - 5
def Granger : ℕ := 5 * Cisco

theorem spots_combined : Granger + Cisco = 108 := by
  sorry

end spots_combined_l210_210855


namespace problem_statement_l210_210712

def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem problem_statement : avg3 (avg3 (-1) 2 3) (avg2 2 3) 1 = 29 / 18 := 
by 
  sorry

end problem_statement_l210_210712


namespace max_actors_chess_tournament_l210_210870

-- Definitions based on conditions
variable {α : Type} [Fintype α] [DecidableEq α]

-- Each actor played with every other actor exactly once.
def played_with_everyone (R : α → α → ℝ) : Prop :=
  ∀ a b, a ≠ b → (R a b = 1 ∨ R a b = 0.5 ∨ R a b = 0)

-- Among every three participants, one earned exactly 1.5 solidus in matches against the other two.
def condition_1_5_solidi (R : α → α → ℝ) : Prop :=
  ∀ a b c, a ≠ b → b ≠ c → a ≠ c → 
   (R a b + R a c = 1.5 ∨ R b a + R b c = 1.5 ∨ R c a + R c b = 1.5)

-- Prove the maximum number of such participants is 5
theorem max_actors_chess_tournament (actors : Finset α) (R : α → α → ℝ) 
  (h_played : played_with_everyone R) (h_condition : condition_1_5_solidi R) :
  actors.card ≤ 5 :=
  sorry

end max_actors_chess_tournament_l210_210870


namespace rebecca_has_22_eggs_l210_210330

-- Define the conditions
def number_of_groups : ℕ := 11
def eggs_per_group : ℕ := 2

-- Define the total number of eggs calculated from the conditions.
def total_eggs : ℕ := number_of_groups * eggs_per_group

-- State the theorem and provide the proof outline.
theorem rebecca_has_22_eggs : total_eggs = 22 := by {
  -- Proof will go here, but for now we put sorry to indicate it is not yet provided.
  sorry
}

end rebecca_has_22_eggs_l210_210330


namespace mooncake_inspection_random_event_l210_210179

-- Definition of event categories
inductive Event
| certain
| impossible
| random

-- Definition of the event in question
def mooncakeInspectionEvent (satisfactory: Bool) : Event :=
if satisfactory then Event.random else Event.random

-- Theorem statement to prove that the event is a random event
theorem mooncake_inspection_random_event (satisfactory: Bool) :
  mooncakeInspectionEvent satisfactory = Event.random :=
sorry

end mooncake_inspection_random_event_l210_210179


namespace intersection_complement_eq_l210_210051

open Set

variable (U M N : Set ℕ)

theorem intersection_complement_eq :
  U = {1, 2, 3, 4, 5} →
  M = {1, 4} →
  N = {1, 3, 5} →
  N ∩ (U \ M) = {3, 5} := by 
sorry

end intersection_complement_eq_l210_210051


namespace horses_for_camels_l210_210659

noncomputable def cost_of_one_elephant : ℕ := 11000
noncomputable def cost_of_one_ox : ℕ := 7333 -- approx.
noncomputable def cost_of_one_horse : ℕ := 1833 -- approx.
noncomputable def cost_of_one_camel : ℕ := 4400

theorem horses_for_camels (H : ℕ) :
  (H * cost_of_one_horse = cost_of_one_camel) → H = 2 :=
by
  -- skipping proof details
  sorry

end horses_for_camels_l210_210659


namespace prime_divides_sum_l210_210166

theorem prime_divides_sum 
  (a b c : ℕ) 
  (h1 : a^3 + 4 * b + c = a * b * c)
  (h2 : a ≥ c)
  (h3 : Prime (a^2 + 2 * a + 2)) : 
  (a^2 + 2 * a + 2) ∣ (a + 2 * b + 2) := 
sorry

end prime_divides_sum_l210_210166


namespace convert_89_to_binary_l210_210516

def divide_by_2_remainders (n : Nat) : List Nat :=
  if n = 0 then [] else (n % 2) :: divide_by_2_remainders (n / 2)

def binary_rep (n : Nat) : List Nat :=
  (divide_by_2_remainders n).reverse

theorem convert_89_to_binary :
  binary_rep 89 = [1, 0, 1, 1, 0, 0, 1] := sorry

end convert_89_to_binary_l210_210516


namespace kylie_first_hour_apples_l210_210522

variable (A : ℕ) -- The number of apples picked in the first hour

-- Definitions based on the given conditions
def applesInFirstHour := A
def applesInSecondHour := 2 * A
def applesInThirdHour := A / 3

-- Total number of apples picked in all three hours
def totalApplesPicked := applesInFirstHour + applesInSecondHour + applesInThirdHour

-- The given condition that the total number of apples picked is 220
axiom total_is_220 : totalApplesPicked = 220

-- Proving that the number of apples picked in the first hour is 66
theorem kylie_first_hour_apples : A = 66 := by
  sorry

end kylie_first_hour_apples_l210_210522


namespace smallest_x_for_multiple_l210_210859

theorem smallest_x_for_multiple (x : ℕ) : (450 * x) % 720 = 0 ↔ x = 8 := 
by {
  sorry
}

end smallest_x_for_multiple_l210_210859


namespace range_of_a_l210_210557

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h_def : ∀ x, f x = a * Real.log x + 1/2 * x^2)
  (h_ineq : ∀ x1 x2 : ℝ, x1 ≠ x2 → 0 < x1 → 0 < x2 → (f x1 - f x2) / (x1 - x2) > 4) : a > 4 :=
sorry

end range_of_a_l210_210557


namespace kaleb_earnings_and_boxes_l210_210672

-- Conditions
def initial_games : ℕ := 76
def games_sold : ℕ := 46
def price_15_dollar : ℕ := 20
def price_10_dollar : ℕ := 15
def price_8_dollar : ℕ := 11
def games_per_box : ℕ := 5

-- Definitions and proof problem
theorem kaleb_earnings_and_boxes (initial_games games_sold price_15_dollar price_10_dollar price_8_dollar games_per_box : ℕ) :
  let earnings := (price_15_dollar * 15) + (price_10_dollar * 10) + (price_8_dollar * 8)
  let remaining_games := initial_games - games_sold
  let boxes_needed := remaining_games / games_per_box
  earnings = 538 ∧ boxes_needed = 6 :=
by
  sorry

end kaleb_earnings_and_boxes_l210_210672


namespace radius_of_semicircular_cubicle_l210_210811

noncomputable def radius_of_semicircle (P : ℝ) : ℝ := P / (Real.pi + 2)

theorem radius_of_semicircular_cubicle :
  radius_of_semicircle 71.9822971502571 = 14 := 
sorry

end radius_of_semicircular_cubicle_l210_210811


namespace rhombus_area_is_correct_l210_210475

def calculate_rhombus_area (d1 d2 : ℕ) : ℕ :=
  (d1 * d2) / 2

theorem rhombus_area_is_correct :
  calculate_rhombus_area (3 * 6) (3 * 4) = 108 := by
  sorry

end rhombus_area_is_correct_l210_210475


namespace original_number_l210_210579

theorem original_number (x : ℝ) (h : 1.10 * x = 550) : x = 500 :=
by
  sorry

end original_number_l210_210579


namespace investments_ratio_l210_210732

theorem investments_ratio (P Q : ℝ) (hpq : 7 / 10 = (P * 2) / (Q * 4)) : P / Q = 7 / 5 :=
by 
  sorry

end investments_ratio_l210_210732


namespace terminal_sides_y_axis_l210_210776

theorem terminal_sides_y_axis (α : ℝ) : 
  (∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 2) ∨ 
  (∃ k : ℤ, α = (2 * k + 1) * Real.pi + Real.pi / 2) ↔ 
  ∃ k : ℤ, α = k * Real.pi + Real.pi / 2 := 
by sorry

end terminal_sides_y_axis_l210_210776


namespace greatest_divisor_of_28_l210_210218

theorem greatest_divisor_of_28 : ∀ d : ℕ, d ∣ 28 → d ≤ 28 :=
by
  sorry

end greatest_divisor_of_28_l210_210218


namespace no_prime_roots_l210_210254

noncomputable def roots_are_prime (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q

theorem no_prime_roots : 
  ∀ k : ℕ, ¬ (∃ p q : ℕ, roots_are_prime p q ∧ p + q = 65 ∧ p * q = k) := 
sorry

end no_prime_roots_l210_210254


namespace common_difference_arithmetic_sequence_l210_210312

theorem common_difference_arithmetic_sequence
  (a : ℕ → ℝ)
  (h1 : ∃ a1 d, (∀ n, a n = a1 + (n - 1) * d))
  (h2 : a 7 - 2 * a 4 = -1)
  (h3 : a 3 = 0) :
  ∃ d, (∀ a1, (a1 + 2 * d = 0 ∧ -d = -1) → d = -1/2) :=
by
  sorry

end common_difference_arithmetic_sequence_l210_210312


namespace ratio_QP_l210_210802

theorem ratio_QP {P Q : ℚ} 
  (h : ∀ x : ℝ, x ≠ 0 → x ≠ 4 → x ≠ -4 → 
    P / (x^2 - 5 * x) + Q / (x + 4) = (x^2 - 3 * x + 8) / (x^3 - 5 * x^2 + 4 * x)) : 
  Q / P = 7 / 2 := 
sorry

end ratio_QP_l210_210802


namespace cubes_divisible_by_nine_l210_210322

theorem cubes_divisible_by_nine (n : ℕ) (hn : n > 0) : 
    (n^3 + (n + 1)^3 + (n + 2)^3) % 9 = 0 := by
  sorry

end cubes_divisible_by_nine_l210_210322


namespace tom_spent_video_games_l210_210485

def cost_football := 14.02
def cost_strategy := 9.46
def cost_batman := 12.04
def total_spent := cost_football + cost_strategy + cost_batman

theorem tom_spent_video_games : total_spent = 35.52 :=
by
  sorry

end tom_spent_video_games_l210_210485


namespace infinite_squares_form_l210_210596

theorem infinite_squares_form (k : ℕ) (hk : 0 < k) : ∃ f : ℕ → ℕ, ∀ n, ∃ a, a^2 = f n * 2^k - 7 :=
by
  sorry

end infinite_squares_form_l210_210596


namespace find_19a_20b_21c_l210_210196

theorem find_19a_20b_21c (a b c : ℕ) (h₁ : 29 * a + 30 * b + 31 * c = 366) 
  (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) : 19 * a + 20 * b + 21 * c = 246 := 
sorry

end find_19a_20b_21c_l210_210196


namespace tan_alpha_frac_l210_210463

theorem tan_alpha_frac (α : ℝ) (h : Real.tan α = 2) : (Real.sin α - Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = 1 / 11 := by
  sorry

end tan_alpha_frac_l210_210463


namespace max_marks_l210_210014

theorem max_marks (M p : ℝ) (h1 : p = 0.60 * M) (h2 : p = 160 + 20) : M = 300 := by
  sorry

end max_marks_l210_210014


namespace common_points_line_circle_l210_210493

theorem common_points_line_circle (a : ℝ) : 
  (∀ x y: ℝ, (x - 2*y + a = 0) → ((x - 2)^2 + y^2 = 1)) ↔ (-2 - Real.sqrt 5 ≤ a ∧ a ≤ -2 + Real.sqrt 5) :=
by sorry

end common_points_line_circle_l210_210493


namespace problem_1_problem_2_l210_210576

open Real

def vec_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

def vec_perpendicular (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem problem_1 (k : ℝ) : 
  let a := (3, 4)
  let b := (2, k)
  (vec_parallel (a.1 + 2 * b.1, a.2 + 2 * b.2) (a.1 - b.1, a.2 - b.2)) →
  k = 8 / 3 := sorry

theorem problem_2 (k : ℝ) : 
  let a := (3, 4)
  let b := (2, k)
  (vec_perpendicular (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2)) →
  k = sqrt 21 ∨ k = - sqrt 21 := sorry

end problem_1_problem_2_l210_210576


namespace f_f_f_f_f_of_1_l210_210948

def f (x : ℕ) : ℕ :=
  if x % 3 = 0 then x / 3 else 5 * x + 2

theorem f_f_f_f_f_of_1 : f (f (f (f (f 1)))) = 4687 :=
by
  sorry

end f_f_f_f_f_of_1_l210_210948


namespace arrangements_three_balls_four_boxes_l210_210526

theorem arrangements_three_balls_four_boxes : 
  ∃ (f : Fin 4 → Fin 4), Function.Injective f :=
sorry

end arrangements_three_balls_four_boxes_l210_210526


namespace calories_in_250g_mixed_drink_l210_210572

def calories_in_mixed_drink (grams_cranberry : ℕ) (grams_honey : ℕ) (grams_water : ℕ)
  (calories_per_100g_cranberry : ℕ) (calories_per_100g_honey : ℕ) (calories_per_100g_water : ℕ)
  (total_grams : ℕ) (portion_grams : ℕ) : ℚ :=
  ((grams_cranberry * calories_per_100g_cranberry + grams_honey * calories_per_100g_honey + grams_water * calories_per_100g_water) : ℚ)
  / (total_grams * portion_grams)

theorem calories_in_250g_mixed_drink :
  calories_in_mixed_drink 150 50 300 30 304 0 100 250 = 98.5 := by
  -- The proof will involve arithmetic operations
  sorry

end calories_in_250g_mixed_drink_l210_210572


namespace kamal_age_problem_l210_210043

theorem kamal_age_problem (K S : ℕ) 
  (h1 : K - 8 = 4 * (S - 8)) 
  (h2 : K + 8 = 2 * (S + 8)) : 
  K = 40 := 
by sorry

end kamal_age_problem_l210_210043


namespace number_of_dogs_l210_210834

theorem number_of_dogs 
  (d c b : Nat) 
  (ratio : d / c / b = 3 / 7 / 12) 
  (total_dogs_and_bunnies : d + b = 375) :
  d = 75 :=
by
  -- Using the hypothesis and given conditions to prove d = 75.
  sorry

end number_of_dogs_l210_210834


namespace least_number_to_add_l210_210885

theorem least_number_to_add (n : ℕ) (divisor : ℕ) (modulus : ℕ) (h1 : n = 1076) (h2 : divisor = 23) (h3 : n % divisor = 18) :
  modulus = divisor - (n % divisor) ∧ modulus = 5 := 
sorry

end least_number_to_add_l210_210885


namespace problem_statement_l210_210702

variable {a : ℕ → ℝ} -- the arithmetic sequence
variable {S : ℕ → ℝ} -- the sum of the first n terms of the sequence
variable (d : ℝ) -- the common difference
variable (a1 : ℝ) -- the first term

-- Conditions
axiom arithmetic_sequence (n : ℕ) : a n = a1 + (n - 1) * d
axiom sum_of_first_n_terms (n : ℕ) : S n = n * (a1 + a n) / 2
axiom S_15_eq_45 : S 15 = 45

-- The statement to prove
theorem problem_statement : 2 * a 12 - a 16 = 3 :=
by
  sorry

end problem_statement_l210_210702


namespace sequence_formula_l210_210725

theorem sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (h_sum: ∀ n : ℕ, n ≥ 2 → S n = n^2 * a n)
  (h_a1 : a 1 = 1) : ∀ n : ℕ, n ≥ 2 → a n = 2 / (n * (n + 1)) :=
by {
  sorry
}

end sequence_formula_l210_210725


namespace tan_ratio_of_angles_l210_210320

theorem tan_ratio_of_angles (a b : ℝ) (h1 : Real.sin (a + b) = 3/4) (h2 : Real.sin (a - b) = 1/2) :
    (Real.tan a / Real.tan b) = 5 := 
by 
  sorry

end tan_ratio_of_angles_l210_210320


namespace john_rental_weeks_l210_210552

noncomputable def camera_value : ℝ := 5000
noncomputable def rental_fee_rate : ℝ := 0.10
noncomputable def friend_payment_rate : ℝ := 0.40
noncomputable def john_total_payment : ℝ := 1200

theorem john_rental_weeks :
  let weekly_rental_fee := camera_value * rental_fee_rate
  let friend_payment := weekly_rental_fee * friend_payment_rate
  let john_weekly_payment := weekly_rental_fee - friend_payment
  let rental_weeks := john_total_payment / john_weekly_payment
  rental_weeks = 4 :=
by
  -- Place for proof steps
  sorry

end john_rental_weeks_l210_210552


namespace right_triangle_ratio_l210_210847

theorem right_triangle_ratio (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : (x - y)^2 + x^2 = (x + y)^2) : x / y = 4 :=
by
  sorry

end right_triangle_ratio_l210_210847


namespace product_consecutive_natural_number_square_l210_210127

theorem product_consecutive_natural_number_square (n : ℕ) : 
  ∃ k : ℕ, 100 * (n^2 + n) + 25 = k^2 :=
by
  sorry

end product_consecutive_natural_number_square_l210_210127


namespace ratio_of_Katie_to_Cole_l210_210634

variable (K C : ℕ)

theorem ratio_of_Katie_to_Cole (h1 : 3 * K = 84) (h2 : C = 7) : K / C = 4 :=
by
  sorry

end ratio_of_Katie_to_Cole_l210_210634


namespace trig_expression_value_l210_210683

theorem trig_expression_value :
  (3 / (Real.sin (140 * Real.pi / 180))^2 - 1 / (Real.cos (140 * Real.pi / 180))^2) * (1 / (2 * Real.sin (10 * Real.pi / 180))) = 16 := 
by
  -- placeholder for proof
  sorry

end trig_expression_value_l210_210683


namespace ratio_of_sides_l210_210615

theorem ratio_of_sides (a b : ℝ) (h1 : a + b = 3 * a) (h2 : a + b - Real.sqrt (a^2 + b^2) = (1 / 3) * b) : a / b = 1 / 2 :=
sorry

end ratio_of_sides_l210_210615


namespace christineTravelDistance_l210_210713

-- Definition of Christine's speed and time
def christineSpeed : ℝ := 20
def christineTime : ℝ := 4

-- Theorem to prove the distance Christine traveled
theorem christineTravelDistance : christineSpeed * christineTime = 80 := by
  -- The proof is omitted
  sorry

end christineTravelDistance_l210_210713


namespace percent_of_a_is_4b_l210_210106

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.8 * b) : (4 * b) / a = 20 / 9 :=
by sorry

end percent_of_a_is_4b_l210_210106


namespace arithmetic_sequence_common_difference_divisible_by_p_l210_210703

theorem arithmetic_sequence_common_difference_divisible_by_p 
  (n : ℕ) (a : ℕ → ℕ) (h1 : n ≥ 2021) (h2 : ∀ i j, 1 ≤ i → i < j → j ≤ n → a i < a j) 
  (h3 : a 1 > 2021) (h4 : ∀ i, 1 ≤ i → i ≤ n → Nat.Prime (a i)) : 
  ∀ p, Nat.Prime p → p < 2021 → ∃ d, (∀ m, 2 ≤ m → a m = a 1 + (m - 1) * d) ∧ p ∣ d := 
sorry

end arithmetic_sequence_common_difference_divisible_by_p_l210_210703


namespace family_work_solution_l210_210665

noncomputable def family_work_problem : Prop :=
  ∃ (M W : ℕ),
    M + W = 15 ∧
    (M * (9/120) + W * (6/180) = 1) ∧
    W = 3

theorem family_work_solution : family_work_problem :=
by
  sorry

end family_work_solution_l210_210665


namespace quadratic_no_solution_l210_210965

def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem quadratic_no_solution (a b c : ℝ) (h1 : a ≠ 0) (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) :
  0 < a ∧ discriminant a b c ≤ 0 :=
by
  sorry

end quadratic_no_solution_l210_210965


namespace vector_rotation_correct_l210_210626

def vector_rotate_z_90 (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := v
  ( -y, x, z )

theorem vector_rotation_correct :
  vector_rotate_z_90 (3, -1, 4) = (-3, 0, 4) := 
by 
  sorry

end vector_rotation_correct_l210_210626


namespace original_number_of_workers_l210_210609

-- Definitions of the conditions given in the problem
def workers_days (W : ℕ) : ℕ := 35
def additional_workers : ℕ := 10
def reduced_days : ℕ := 10

-- The main theorem we need to prove
theorem original_number_of_workers (W : ℕ) (A : ℕ) 
  (h1 : W * workers_days W = (W + additional_workers) * (workers_days W - reduced_days)) :
  W = 25 :=
by
  sorry

end original_number_of_workers_l210_210609


namespace division_result_l210_210253

theorem division_result : 203515 / 2015 = 101 := 
by sorry

end division_result_l210_210253


namespace domain_of_p_l210_210825

theorem domain_of_p (h : ℝ → ℝ) (h_domain : ∀ x, -10 ≤ x → x ≤ 6 → ∃ y, h x = y) :
  ∀ x, -1.2 ≤ x ∧ x ≤ 2 → ∃ y, h (-5 * x) = y :=
by
  sorry

end domain_of_p_l210_210825


namespace no_two_champions_l210_210122

structure Tournament (Team : Type) :=
  (defeats : Team → Team → Prop)  -- Team A defeats Team B

def is_superior {Team : Type} (T : Tournament Team) (A B: Team) : Prop :=
  T.defeats A B ∨ ∃ C, T.defeats A C ∧ T.defeats C B

def is_champion {Team : Type} (T : Tournament Team) (A : Team) : Prop :=
  ∀ B, A ≠ B → is_superior T A B

theorem no_two_champions {Team : Type} (T : Tournament Team) :
  ¬ (∃ A B, A ≠ B ∧ is_champion T A ∧ is_champion T B) :=
sorry

end no_two_champions_l210_210122


namespace initial_weight_of_mixture_eq_20_l210_210959

theorem initial_weight_of_mixture_eq_20
  (W : ℝ) (h1 : 0.1 * W + 4 = 0.25 * (W + 4)) :
  W = 20 :=
by
  sorry

end initial_weight_of_mixture_eq_20_l210_210959


namespace solve_for_x_l210_210960

theorem solve_for_x (x : ℝ) (h : 2 - 1 / (1 - x) = 1 / (1 - x)) : x = 0 :=
sorry

end solve_for_x_l210_210960


namespace quadratic_roots_unique_pair_l210_210045

theorem quadratic_roots_unique_pair (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (h_root1 : p * q = q)
  (h_root2 : p + q = -p)
  (h_rel : q = -2 * p) : 
(p, q) = (1, -2) :=
  sorry

end quadratic_roots_unique_pair_l210_210045


namespace find_f_37_5_l210_210165

noncomputable def f (x : ℝ) : ℝ := sorry

/--
Given that \( f \) is an odd function defined on \( \mathbb{R} \) and satisfies
\( f(x+2) = -f(x) \). When \( 0 \leqslant x \leqslant 1 \), \( f(x) = x \),
prove that \( f(37.5) = 0.5 \).
-/
theorem find_f_37_5 (f : ℝ → ℝ) (odd_f : ∀ x : ℝ, f (-x) = -f x)
  (periodic_f : ∀ x : ℝ, f (x + 2) = -f x)
  (interval_f : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x) : f 37.5 = 0.5 :=
sorry

end find_f_37_5_l210_210165


namespace rita_bought_4_pounds_l210_210384

variable (total_amount : ℝ) (cost_per_pound : ℝ) (amount_left : ℝ)

theorem rita_bought_4_pounds (h1 : total_amount = 70)
                             (h2 : cost_per_pound = 8.58)
                             (h3 : amount_left = 35.68) :
  (total_amount - amount_left) / cost_per_pound = 4 := 
  by
  sorry

end rita_bought_4_pounds_l210_210384


namespace angle_ne_iff_cos2angle_ne_l210_210375

theorem angle_ne_iff_cos2angle_ne (A B : ℝ) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) :
  (A ≠ B) ↔ (Real.cos (2 * A) ≠ Real.cos (2 * B)) :=
sorry

end angle_ne_iff_cos2angle_ne_l210_210375


namespace m_gt_n_l210_210032

noncomputable def m : ℕ := 2015 ^ 2016
noncomputable def n : ℕ := 2016 ^ 2015

theorem m_gt_n : m > n := by
  sorry

end m_gt_n_l210_210032


namespace at_least_one_ge_two_l210_210922

theorem at_least_one_ge_two (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let a := x + 1 / y
  let b := y + 1 / z
  let c := z + 1 / x
  a + b + c ≥ 6 → (a ≥ 2 ∨ b ≥ 2 ∨ c ≥ 2) :=
by
  intros
  let a := x + 1 / y
  let b := y + 1 / z
  let c := z + 1 / x
  sorry

end at_least_one_ge_two_l210_210922


namespace casey_pumping_time_l210_210772

theorem casey_pumping_time :
  let pump_rate := 3 -- gallons per minute
  let corn_rows := 4
  let corn_per_row := 15
  let water_per_corn := 1 / 2
  let total_corn := corn_rows * corn_per_row
  let corn_water := total_corn * water_per_corn
  let num_pigs := 10
  let water_per_pig := 4
  let pig_water := num_pigs * water_per_pig
  let num_ducks := 20
  let water_per_duck := 1 / 4
  let duck_water := num_ducks * water_per_duck
  let total_water := corn_water + pig_water + duck_water
  let time_needed := total_water / pump_rate
  time_needed = 25 :=
by
  sorry

end casey_pumping_time_l210_210772


namespace sum_of_digits_18_to_21_l210_210069

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_18_to_21 : 
  (sum_digits 18 + sum_digits 19 + sum_digits 20 + sum_digits 21) = 24 := 
by 
  sorry

end sum_of_digits_18_to_21_l210_210069


namespace range_of_k_l210_210411

theorem range_of_k (k n : ℝ) (h : k ≠ 0) (h_pass : k - n^2 - 2 = k / 2) : k ≥ 4 :=
sorry

end range_of_k_l210_210411


namespace number_of_pairs_l210_210589

theorem number_of_pairs (h : ∀ (a : ℝ) (b : ℕ), 0 < a → 2 ≤ b ∧ b ≤ 200 → (Real.log a / Real.log b) ^ 2017 = Real.log (a ^ 2017) / Real.log b) :
  ∃ n, n = 597 ∧ ∀ b : ℕ, 2 ≤ b ∧ b ≤ 200 → 
    ∃ a1 a2 a3 : ℝ, 0 < a1 ∧ 0 < a2 ∧ 0 < a3 ∧ 
      (Real.log a1 / Real.log b) = 0 ∧ 
      (Real.log a2 / Real.log b) = 2017^((1:ℝ)/2016) ∧ 
      (Real.log a3 / Real.log b) = -2017^((1:ℝ)/2016) :=
sorry

end number_of_pairs_l210_210589


namespace arithmetic_sequence_a5_l210_210115

variable (a : ℕ → ℝ) (h : a 1 + a 9 = 10)

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h : a 1 + a 9 = 10) : 
  a 5 = 5 :=
by sorry

end arithmetic_sequence_a5_l210_210115


namespace greatest_possible_grapes_thrown_out_l210_210630

theorem greatest_possible_grapes_thrown_out (n : ℕ) : 
  n % 7 ≤ 6 := by 
  sorry

end greatest_possible_grapes_thrown_out_l210_210630


namespace seth_spent_more_l210_210099

theorem seth_spent_more : 
  let ice_cream_cartons := 20
  let yogurt_cartons := 2
  let ice_cream_price := 6
  let yogurt_price := 1
  let ice_cream_discount := 0.10
  let yogurt_discount := 0.20
  let total_ice_cream_cost := ice_cream_cartons * ice_cream_price
  let total_yogurt_cost := yogurt_cartons * yogurt_price
  let discounted_ice_cream_cost := total_ice_cream_cost * (1 - ice_cream_discount)
  let discounted_yogurt_cost := total_yogurt_cost * (1 - yogurt_discount)
  discounted_ice_cream_cost - discounted_yogurt_cost = 106.40 :=
by
  sorry

end seth_spent_more_l210_210099


namespace train_distance_in_2_hours_l210_210752

theorem train_distance_in_2_hours :
  (∀ (t : ℕ), t = 90 → (1 / ↑t) * 7200 = 80) :=
by
  sorry

end train_distance_in_2_hours_l210_210752


namespace meeting_point_distance_l210_210392

theorem meeting_point_distance
  (distance_to_top : ℝ)
  (total_distance : ℝ)
  (jack_start_time : ℝ)
  (jack_uphill_speed : ℝ)
  (jack_downhill_speed : ℝ)
  (jill_uphill_speed : ℝ)
  (jill_downhill_speed : ℝ)
  (meeting_point_distance : ℝ):
  distance_to_top = 5 -> total_distance = 10 -> jack_start_time = 10 / 60 ->
  jack_uphill_speed = 15 -> jack_downhill_speed = 20 ->
  jill_uphill_speed = 16 -> jill_downhill_speed = 22 ->
  meeting_point_distance = 35 / 27 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end meeting_point_distance_l210_210392


namespace speed_in_still_water_l210_210964

def upstream_speed : ℝ := 35
def downstream_speed : ℝ := 45

theorem speed_in_still_water:
  (upstream_speed + downstream_speed) / 2 = 40 := 
by
  sorry

end speed_in_still_water_l210_210964


namespace compare_abc_l210_210534

noncomputable def a : ℝ := 9 ^ (Real.log 4.1 / Real.log 2)
noncomputable def b : ℝ := 9 ^ (Real.log 2.7 / Real.log 2)
noncomputable def c : ℝ := (1 / 3 : ℝ) ^ (Real.log 0.1 / Real.log 2)

theorem compare_abc :
  a > c ∧ c > b := by
  sorry

end compare_abc_l210_210534


namespace total_cost_of_stamps_is_correct_l210_210875

-- Define the costs of each type of stamp
def cost_of_stamp_A : ℕ := 34 -- cost in cents
def cost_of_stamp_B : ℕ := 52 -- cost in cents
def cost_of_stamp_C : ℕ := 73 -- cost in cents

-- Define the number of stamps Alice needs to buy
def num_stamp_A : ℕ := 4
def num_stamp_B : ℕ := 6
def num_stamp_C : ℕ := 2

-- Define the expected total cost in dollars
def expected_total_cost : ℝ := 5.94

-- State the theorem about the total cost
theorem total_cost_of_stamps_is_correct :
  ((num_stamp_A * cost_of_stamp_A) + (num_stamp_B * cost_of_stamp_B) + (num_stamp_C * cost_of_stamp_C)) / 100 = expected_total_cost :=
by
  sorry

end total_cost_of_stamps_is_correct_l210_210875


namespace combined_mean_is_254_over_15_l210_210651

noncomputable def combined_mean_of_sets 
  (mean₁ : ℝ) (n₁ : ℕ) 
  (mean₂ : ℝ) (n₂ : ℕ) : ℝ :=
  (mean₁ * n₁ + mean₂ * n₂) / (n₁ + n₂)

theorem combined_mean_is_254_over_15 :
  combined_mean_of_sets 18 7 16 8 = (254 : ℝ) / 15 :=
by
  sorry

end combined_mean_is_254_over_15_l210_210651


namespace polynomial_positive_for_all_reals_l210_210883

theorem polynomial_positive_for_all_reals (m : ℝ) : m^6 - m^5 + m^4 + m^2 - m + 1 > 0 :=
by
  sorry

end polynomial_positive_for_all_reals_l210_210883


namespace contractor_absent_days_l210_210090

-- Definition of problem conditions
def total_days : ℕ := 30
def daily_wage : ℝ := 25
def daily_fine : ℝ := 7.5
def total_amount_received : ℝ := 620

-- Function to define the constraint equations
def equation1 (x y : ℕ) : Prop := x + y = total_days
def equation2 (x y : ℕ) : Prop := (daily_wage * x - daily_fine * y) = total_amount_received

-- The proof problem translation as Lean 4 statement
theorem contractor_absent_days (x y : ℕ) (h1 : equation1 x y) (h2 : equation2 x y) : y = 8 :=
by
  sorry

end contractor_absent_days_l210_210090


namespace chord_length_of_intersecting_line_and_circle_l210_210038

theorem chord_length_of_intersecting_line_and_circle :
  ∀ (x y : ℝ), (3 * x + 4 * y - 5 = 0) ∧ (x^2 + y^2 = 4) →
  ∃ (AB : ℝ), AB = 2 * Real.sqrt 3 := 
sorry

end chord_length_of_intersecting_line_and_circle_l210_210038


namespace students_take_neither_l210_210510

variable (Total Mathematic Physics Both MathPhysics ChemistryNeither Neither : ℕ)

axiom Total_students : Total = 80
axiom students_mathematics : Mathematic = 50
axiom students_physics : Physics = 40
axiom students_both : Both = 25
axiom students_chemistry_neither : ChemistryNeither = 10

theorem students_take_neither :
  Neither = Total - (Mathematic - Both + Physics - Both + Both + ChemistryNeither) :=
  by
  have Total_students := Total_students
  have students_mathematics := students_mathematics
  have students_physics := students_physics
  have students_both := students_both
  have students_chemistry_neither := students_chemistry_neither
  sorry

end students_take_neither_l210_210510


namespace completing_the_square_l210_210655

theorem completing_the_square (x : ℝ) : (x^2 - 6*x + 7 = 0) → ((x - 3)^2 = 2) :=
by
  intro h
  sorry

end completing_the_square_l210_210655


namespace selection_count_l210_210938

noncomputable def choose (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem selection_count :
  let boys := 4
  let girls := 3
  let total := boys + girls
  let choose_boys_girls : ℕ := (choose 4 2) * (choose 3 1) + (choose 4 1) * (choose 3 2)
  choose_boys_girls = 30 := 
by
  sorry

end selection_count_l210_210938


namespace time_to_cover_length_l210_210098

def escalator_speed : ℝ := 8  -- The speed of the escalator in feet per second
def person_speed : ℝ := 2     -- The speed of the person in feet per second
def escalator_length : ℝ := 160 -- The length of the escalator in feet

theorem time_to_cover_length : 
  (escalator_length / (escalator_speed + person_speed) = 16) :=
by 
  sorry

end time_to_cover_length_l210_210098


namespace maximum_students_per_dentist_l210_210695

theorem maximum_students_per_dentist (dentists students : ℕ) (min_students : ℕ) (attended_students : ℕ)
  (h_dentists : dentists = 12)
  (h_students : students = 29)
  (h_min_students : min_students = 2)
  (h_total_students : attended_students = students) :
  ∃ max_students, 
    (∀ d, d < dentists → min_students ≤ attended_students / dentists) ∧
    (∀ d, d < dentists → attended_students = students - (dentists * min_students) + min_students) ∧
    max_students = 7 :=
by
  sorry

end maximum_students_per_dentist_l210_210695


namespace find_a9_l210_210365

theorem find_a9 (a : ℕ → ℕ) 
  (h_add : ∀ p q : ℕ, 0 < p → 0 < q → a (p + q) = a p + a q)
  (h_a2 : a 2 = 4) 
  : a 9 = 18 :=
sorry

end find_a9_l210_210365


namespace polynomial_A_polynomial_B_l210_210012

-- Problem (1): Prove that A = 6x^3 + 8x^2 + x - 1 given the conditions.
theorem polynomial_A :
  ∀ (x : ℝ),
  (2 * x^2 * (3 * x + 4) + (x - 1) = 6 * x^3 + 8 * x^2 + x - 1) :=
by
  intro x
  sorry

-- Problem (2): Prove that B = 6x^2 - 19x + 9 given the conditions.
theorem polynomial_B :
  ∀ (x : ℝ),
  ((2 * x - 6) * (3 * x - 1) + (x + 3) = 6 * x^2 - 19 * x + 9) :=
by
  intro x
  sorry

end polynomial_A_polynomial_B_l210_210012


namespace per_minute_charge_after_6_minutes_l210_210232

noncomputable def cost_plan_a (x : ℝ) (t : ℝ) : ℝ :=
  if t <= 6 then 0.60 else 0.60 + (t - 6) * x

noncomputable def cost_plan_b (t : ℝ) : ℝ :=
  t * 0.08

theorem per_minute_charge_after_6_minutes :
  ∃ (x : ℝ), cost_plan_a x 12 = cost_plan_b 12 ∧ x = 0.06 :=
by
  use 0.06
  simp [cost_plan_a, cost_plan_b]
  sorry

end per_minute_charge_after_6_minutes_l210_210232


namespace car_returns_to_start_after_5_operations_l210_210502

theorem car_returns_to_start_after_5_operations (α : ℝ) (h1 : 0 < α) (h2 : α < 180) : α = 72 ∨ α = 144 :=
sorry

end car_returns_to_start_after_5_operations_l210_210502


namespace number_of_math_books_l210_210139

-- Definitions based on the conditions in the problem
def total_books (M H : ℕ) : Prop := M + H = 90
def total_cost (M H : ℕ) : Prop := 4 * M + 5 * H = 390

-- Proof statement
theorem number_of_math_books (M H : ℕ) (h1 : total_books M H) (h2 : total_cost M H) : M = 60 :=
  sorry

end number_of_math_books_l210_210139


namespace unit_prices_min_selling_price_l210_210249

-- Problem 1: Unit price determination
theorem unit_prices (x y : ℕ) (hx : 3600 / x * 2 = 5400 / y) (hy : y = x - 5) : x = 20 ∧ y = 15 := 
by 
  sorry

-- Problem 2: Minimum selling price for 50% profit margin
theorem min_selling_price (a : ℕ) (hx : 3600 / 20 = 180) (hy : 180 * 2 = 360) (hz : 540 * a ≥ 13500) : a ≥ 25 := 
by 
  sorry

end unit_prices_min_selling_price_l210_210249


namespace Alan_has_eight_pine_trees_l210_210119

noncomputable def number_of_pine_trees (total_pine_cones_per_tree : ℕ) (percentage_on_roof : ℚ) 
                                       (weight_per_pine_cone : ℚ) (total_weight_on_roof : ℚ) : ℚ :=
  total_weight_on_roof / (total_pine_cones_per_tree * percentage_on_roof * weight_per_pine_cone)

theorem Alan_has_eight_pine_trees :
  number_of_pine_trees 200 (30 / 100) 4 1920 = 8 :=
by
  sorry

end Alan_has_eight_pine_trees_l210_210119


namespace w12_plus_inv_w12_l210_210749

open Complex

-- Given conditions
def w_plus_inv_w_eq_two_cos_45 (w : ℂ) : Prop :=
  w + (1 / w) = 2 * Real.cos (Real.pi / 4)

-- Statement of the theorem to prove
theorem w12_plus_inv_w12 {w : ℂ} (h : w_plus_inv_w_eq_two_cos_45 w) : 
  w^12 + (1 / (w^12)) = -2 :=
sorry

end w12_plus_inv_w12_l210_210749


namespace arth_seq_val_a7_l210_210705

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

theorem arth_seq_val_a7 {a : ℕ → ℝ} 
  (h_arith : arithmetic_sequence a)
  (h_positive : ∀ n : ℕ, 0 < a n)
  (h_eq : 2 * a 6 + 2 * a 8 = (a 7) ^ 2) :
  a 7 = 4 := 
by sorry

end arth_seq_val_a7_l210_210705


namespace prove_proposition_false_l210_210983

def proposition (a : ℝ) := ∃ x : ℝ, x^2 - 4*a*x + 3 < 0

theorem prove_proposition_false : proposition 0 = False :=
by
sorry

end prove_proposition_false_l210_210983


namespace number_of_positive_area_triangles_l210_210641

def integer_points := {p : ℕ × ℕ // 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}

def count_triangles_with_positive_area : Nat :=
  let total_points := 25 -- total number of integer points in the grid
  let total_combinations := Nat.choose total_points 3 -- total possible combinations
  let degenerate_cases := 136 -- total degenerate (collinear) cases
  total_combinations - degenerate_cases

theorem number_of_positive_area_triangles : count_triangles_with_positive_area = 2164 := by
  sorry

end number_of_positive_area_triangles_l210_210641


namespace part1_inequality_part2_inequality_l210_210722

theorem part1_inequality (x : ℝ) : 
  (3 * x - 2) / (x - 1) > 1 ↔ x > 1 ∨ x < 1 / 2 := 
by sorry

theorem part2_inequality (x a : ℝ) : 
  x^2 - a * x - 2 * a^2 < 0 ↔ 
  (a = 0 → False) ∧ 
  (a > 0 → -a < x ∧ x < 2 * a) ∧ 
  (a < 0 → 2 * a < x ∧ x < -a) := 
by sorry

end part1_inequality_part2_inequality_l210_210722


namespace min_x_y_l210_210729

theorem min_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : (x + 1) * (y + 1) = 9) : x + y ≥ 4 :=
by
  sorry

end min_x_y_l210_210729


namespace sin_2theta_plus_pi_div_2_l210_210880

theorem sin_2theta_plus_pi_div_2 (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 4)
    (h_tan2θ : Real.tan (2 * θ) = Real.cos θ / (2 - Real.sin θ)) :
    Real.sin (2 * θ + π / 2) = 7 / 8 :=
sorry

end sin_2theta_plus_pi_div_2_l210_210880


namespace simple_interest_is_correct_l210_210445

-- Define the principal amount, rate of interest, and time
def P : ℕ := 400
def R : ℚ := 22.5
def T : ℕ := 2

-- Define the formula for simple interest
def simple_interest (P : ℕ) (R : ℚ) (T : ℕ) : ℚ :=
  (P * R * T) / 100

-- The statement we need to prove
theorem simple_interest_is_correct : simple_interest P R T = 90 :=
by
  sorry

end simple_interest_is_correct_l210_210445


namespace number_of_schools_l210_210469

-- Define the conditions
def is_median (a : ℕ) (n : ℕ) : Prop := 2 * a - 1 = n
def high_team_score (a b c : ℕ) : Prop := a > b ∧ a > c
def ranks (b c : ℕ) : Prop := b = 39 ∧ c = 67

-- Define the main problem
theorem number_of_schools (a n b c : ℕ) :
  is_median a n →
  high_team_score a b c →
  ranks b c →
  34 ≤ a ∧ a < 39 →
  2 * a ≡ 1 [MOD 3] →
  (n = 67 → a = 35) →
  (∀ m : ℕ, n = 3 * m + 1) →
  m = 23 :=
by
  sorry

end number_of_schools_l210_210469


namespace fermat_numbers_pairwise_coprime_l210_210046

theorem fermat_numbers_pairwise_coprime :
  ∀ i j : ℕ, i ≠ j → Nat.gcd (2 ^ (2 ^ i) + 1) (2 ^ (2 ^ j) + 1) = 1 :=
sorry

end fermat_numbers_pairwise_coprime_l210_210046


namespace surface_area_calculation_l210_210852

-- Conditions:
-- Original rectangular sheet dimensions
def length : ℕ := 25
def width : ℕ := 35
-- Dimensions of the square corners
def corner_side : ℕ := 7

-- Surface area of the interior calculation
noncomputable def surface_area_interior : ℕ :=
  let original_area := length * width
  let corner_area := corner_side * corner_side
  let total_corner_area := 4 * corner_area
  original_area - total_corner_area

-- Theorem: The surface area of the interior of the resulting box
theorem surface_area_calculation : surface_area_interior = 679 := by
  -- You can fill in the details to compute the answer
  sorry

end surface_area_calculation_l210_210852


namespace one_twenty_percent_of_number_l210_210605

theorem one_twenty_percent_of_number (x : ℝ) (h : 0.20 * x = 300) : 1.20 * x = 1800 :=
by 
sorry

end one_twenty_percent_of_number_l210_210605


namespace total_amount_given_away_l210_210718

variable (numGrandchildren : ℕ)
variable (cardsPerGrandchild : ℕ)
variable (amountPerCard : ℕ)

theorem total_amount_given_away (h1 : numGrandchildren = 3) (h2 : cardsPerGrandchild = 2) (h3 : amountPerCard = 80) : 
  numGrandchildren * cardsPerGrandchild * amountPerCard = 480 := by
  sorry

end total_amount_given_away_l210_210718


namespace honey_harvest_this_year_l210_210296

def last_year_harvest : ℕ := 2479
def increase_this_year : ℕ := 6085

theorem honey_harvest_this_year : last_year_harvest + increase_this_year = 8564 :=
by {
  sorry
}

end honey_harvest_this_year_l210_210296


namespace factorization_problem_l210_210835

theorem factorization_problem :
  (∃ (h : D), 
    (¬ ∃ (a b : ℝ) (x y : ℝ), a * (x - y) = a * x - a * y) ∧
    (¬ ∃ (x : ℝ), x^2 - 2 * x + 3 = x * (x - 2) + 3) ∧
    (¬ ∃ (x : ℝ), (x - 1) * (x + 4) = x^2 + 3 * x - 4) ∧
    (∃ (x : ℝ), x^3 - 2 * x^2 + x = x * (x - 1)^2)) :=
  sorry

end factorization_problem_l210_210835


namespace abcd_sum_is_12_l210_210335

theorem abcd_sum_is_12 (a b c d : ℤ) 
  (h1 : a + c = 2) 
  (h2 : a * c + b + d = -1) 
  (h3 : a * d + b * c = 18) 
  (h4 : b * d = 24) : 
  a + b + c + d = 12 :=
sorry

end abcd_sum_is_12_l210_210335


namespace smallest_value_4x_plus_3y_l210_210820

-- Define the condition as a predicate
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 18 * x + 8 * y + 10

-- Prove the smallest possible value of 4x + 3y given the condition
theorem smallest_value_4x_plus_3y : ∃ x y : ℝ, circle_eq x y ∧ (4 * x + 3 * y = -40) :=
by
  -- Placeholder for the proof
  sorry

end smallest_value_4x_plus_3y_l210_210820


namespace total_distance_trip_l210_210295

-- Defining conditions
def time_paved := 2 -- hours
def time_dirt := 3 -- hours
def speed_dirt := 32 -- mph
def speed_paved := speed_dirt + 20 -- mph

-- Defining distances
def distance_dirt := speed_dirt * time_dirt -- miles
def distance_paved := speed_paved * time_paved -- miles

-- Proving total distance
theorem total_distance_trip : distance_dirt + distance_paved = 200 := by
  sorry

end total_distance_trip_l210_210295


namespace alcohol_mixture_l210_210963

theorem alcohol_mixture (y : ℕ) :
  let x_vol := 200 -- milliliters
  let y_conc := 30 / 100 -- 30% alcohol
  let x_conc := 10 / 100 -- 10% alcohol
  let final_conc := 20 / 100 -- 20% target alcohol concentration
  let x_alcohol := x_vol * x_conc -- alcohol in x
  (x_alcohol + y * y_conc) / (x_vol + y) = final_conc ↔ y = 200 :=
by 
  sorry

end alcohol_mixture_l210_210963


namespace f_positive_l210_210216

variable (f : ℝ → ℝ)

-- f is a differentiable function on ℝ
variable (hf : differentiable ℝ f)

-- Condition: (x+1)f(x) + x f''(x) > 0
variable (H : ∀ x, (x + 1) * f x + x * (deriv^[2]) f x > 0)

-- Prove: ∀ x, f x > 0
theorem f_positive : ∀ x, f x > 0 := 
by
  sorry

end f_positive_l210_210216


namespace calculate_product_value_l210_210857

theorem calculate_product_value :
    (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 :=
by
  sorry

end calculate_product_value_l210_210857


namespace equation_of_ellipse_AN_BM_constant_l210_210159

noncomputable def a := 2
noncomputable def b := 1
noncomputable def e := (Real.sqrt 3) / 2
noncomputable def c := Real.sqrt 3

def ellipse (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

theorem equation_of_ellipse :
  ellipse a b
:=
by
  sorry

theorem AN_BM_constant (x0 y0 : ℝ) (hx : x0^2 + 4 * y0^2 = 4) :
  let AN := 2 + x0 / (y0 - 1)
  let BM := 1 + 2 * y0 / (x0 - 2)
  abs (AN * BM) = 4
:=
by
  sorry

end equation_of_ellipse_AN_BM_constant_l210_210159


namespace not_perfect_square_l210_210188

theorem not_perfect_square (n : ℤ) : ¬ ∃ (m : ℤ), 4*n + 3 = m^2 := 
by 
  sorry

end not_perfect_square_l210_210188


namespace total_marbles_l210_210311

-- There are only red, blue, and yellow marbles
universe u
variable {α : Type u}

-- The ratio of red marbles to blue marbles to yellow marbles is \(2:3:4\)
variables {r b y T : ℕ}
variable (ratio_cond : 2 * y = 4 * r ∧ 3 * y = 4 * b)

-- There are 40 yellow marbles in the container
variable (yellow_cond : y = 40)

-- Prove the total number of marbles in the container is 90
theorem total_marbles (ratio_cond : 2 * y = 4 * r ∧ 3 * y = 4 * b) (yellow_cond : y = 40) :
  T = r + b + y → T = 90 :=
sorry

end total_marbles_l210_210311


namespace find_abs_xyz_l210_210785

noncomputable def distinct_nonzero_real (x y z : ℝ) : Prop :=
x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 

theorem find_abs_xyz
  (x y z : ℝ)
  (h1 : distinct_nonzero_real x y z)
  (h2 : x + 1/y = y + 1/z)
  (h3 : y + 1/z = z + 1/x + 1) :
  |x * y * z| = 1 :=
sorry

end find_abs_xyz_l210_210785


namespace simplify_expansion_l210_210478

-- Define the variables and expressions
variable (x : ℝ)

-- The main statement
theorem simplify_expansion : (x + 5) * (4 * x - 12) = 4 * x^2 + 8 * x - 60 :=
by sorry

end simplify_expansion_l210_210478


namespace total_number_of_pages_l210_210200

variable (x : ℕ)

-- Conditions
def first_day_remaining : ℕ := x - (x / 6 + 10)
def second_day_remaining : ℕ := first_day_remaining x - (first_day_remaining x / 5 + 20)
def third_day_remaining : ℕ := second_day_remaining x - (second_day_remaining x / 4 + 25)
def final_remaining : Prop := third_day_remaining x = 100

-- Theorem statement
theorem total_number_of_pages : final_remaining x → x = 298 :=
by
  intros h
  sorry

end total_number_of_pages_l210_210200


namespace general_term_formula_T_n_less_than_one_sixth_l210_210435

noncomputable def S (n : ℕ) : ℕ := n^2 + 2*n

def a (n : ℕ) : ℕ := if n = 0 then 0 else 2*n + 1

def b (n : ℕ) : ℕ := if n = 0 then 0 else 1 / (a n) * (a (n+1))

def T (n : ℕ) : ℝ := (Finset.range n).sum (λ k => (b k : ℝ))

theorem general_term_formula (n : ℕ) (hn : n ≠ 0) : 
  a n = 2*n + 1 :=
by sorry

theorem T_n_less_than_one_sixth (n : ℕ) : 
  T n < (1 / 6 : ℝ) :=
by sorry

end general_term_formula_T_n_less_than_one_sixth_l210_210435


namespace a_b_sum_of_powers_l210_210123

variable (a b : ℝ)

-- Conditions
def condition1 := a + b = 1
def condition2 := a^2 + b^2 = 3
def condition3 := a^3 + b^3 = 4
def condition4 := a^4 + b^4 = 7
def condition5 := a^5 + b^5 = 11

-- Theorem statement
theorem a_b_sum_of_powers (h1 : condition1 a b) (h2 : condition2 a b) (h3 : condition3 a b) 
  (h4 : condition4 a b) (h5 : condition5 a b) : a^10 + b^10 = 123 :=
sorry

end a_b_sum_of_powers_l210_210123


namespace fraction_zero_iff_x_neg_one_l210_210120

theorem fraction_zero_iff_x_neg_one (x : ℝ) (h : 1 - |x| = 0) (h_non_zero : 1 - x ≠ 0) : x = -1 :=
sorry

end fraction_zero_iff_x_neg_one_l210_210120


namespace rick_gives_miguel_cards_l210_210901

/-- Rick starts with 130 cards, keeps 15 cards for himself, gives 
12 cards each to 8 friends, and gives 3 cards each to his 2 sisters. 
We need to prove that Rick gives 13 cards to Miguel. --/
theorem rick_gives_miguel_cards :
  let initial_cards := 130
  let kept_cards := 15
  let friends := 8
  let cards_per_friend := 12
  let sisters := 2
  let cards_per_sister := 3
  initial_cards - kept_cards - (friends * cards_per_friend) - (sisters * cards_per_sister) = 13 :=
by
  sorry

end rick_gives_miguel_cards_l210_210901


namespace area_inside_circle_outside_square_is_zero_l210_210592

theorem area_inside_circle_outside_square_is_zero 
  (side_length : ℝ) (circle_radius : ℝ)
  (h_square_side : side_length = 2) (h_circle_radius : circle_radius = 1) : 
  (π * circle_radius^2) - (side_length^2) = 0 := 
by 
  sorry

end area_inside_circle_outside_square_is_zero_l210_210592


namespace lawn_area_l210_210547

theorem lawn_area (s l : ℕ) (hs: 5 * s = 10) (hl: 5 * l = 50) (hposts: 2 * (s + l) = 24) (hlen: l + 1 = 3 * (s + 1)) :
  s * l = 500 :=
by {
  sorry
}

end lawn_area_l210_210547


namespace pumpkin_count_sunshine_orchard_l210_210440

def y (x : ℕ) : ℕ := 3 * x^2 + 12

theorem pumpkin_count_sunshine_orchard :
  y 14 = 600 :=
by
  sorry

end pumpkin_count_sunshine_orchard_l210_210440


namespace fewer_hours_l210_210208

noncomputable def distance : ℝ := 300
noncomputable def speed_T : ℝ := 20
noncomputable def speed_A : ℝ := speed_T + 5

theorem fewer_hours (d : ℝ) (V_T : ℝ) (V_A : ℝ) :
    V_T = 20 ∧ V_A = V_T + 5 ∧ d = 300 → (d / V_T) - (d / V_A) = 3 := 
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end fewer_hours_l210_210208


namespace yulia_profit_l210_210292

-- Assuming the necessary definitions in the problem
def lemonade_revenue : ℕ := 47
def babysitting_revenue : ℕ := 31
def expenses : ℕ := 34
def profit : ℕ := lemonade_revenue + babysitting_revenue - expenses

-- The proof statement to prove Yulia's profit
theorem yulia_profit : profit = 44 := by
  sorry -- Proof is skipped

end yulia_profit_l210_210292


namespace additional_men_joined_l210_210112

theorem additional_men_joined
    (M : ℕ) (X : ℕ)
    (h1 : M = 20)
    (h2 : M * 50 = (M + X) * 25) :
    X = 20 := by
  sorry

end additional_men_joined_l210_210112


namespace factorize_polynomial_l210_210343

noncomputable def polynomial_factorization : Prop :=
  ∀ x : ℤ, (x^12 + x^9 + 1) = (x^4 + x^3 + x^2 + x + 1) * (x^8 - x^7 + x^6 - x^5 + x^3 - x^2 + x - 1)

theorem factorize_polynomial : polynomial_factorization :=
by
  sorry

end factorize_polynomial_l210_210343


namespace least_positive_divisible_by_smallest_primes_l210_210653

def smallest_primes := [2, 3, 5, 7, 11]

noncomputable def product_of_smallest_primes :=
  List.foldl (· * ·) 1 smallest_primes

theorem least_positive_divisible_by_smallest_primes :
  product_of_smallest_primes = 2310 :=
by
  sorry

end least_positive_divisible_by_smallest_primes_l210_210653


namespace nearly_tricky_7_tiny_count_l210_210748

-- Define a tricky polynomial
def is_tricky (P : Polynomial ℤ) : Prop :=
  Polynomial.eval 4 P = 0

-- Define a k-tiny polynomial
def is_k_tiny (k : ℤ) (P : Polynomial ℤ) : Prop :=
  P.degree ≤ 7 ∧ ∀ i, abs (Polynomial.coeff P i) ≤ k

-- Define a 1-tiny polynomial
def is_1_tiny (P : Polynomial ℤ) : Prop :=
  is_k_tiny 1 P

-- Define a nearly tricky polynomial as the sum of a tricky polynomial and a 1-tiny polynomial
def is_nearly_tricky (P : Polynomial ℤ) : Prop :=
  ∃ Q T : Polynomial ℤ, is_tricky Q ∧ is_1_tiny T ∧ P = Q + T

-- Define a 7-tiny polynomial
def is_7_tiny (P : Polynomial ℤ) : Prop :=
  is_k_tiny 7 P

-- Count the number of nearly tricky 7-tiny polynomials
def count_nearly_tricky_7_tiny : ℕ :=
  -- Simplification: hypothetical function counting the number of polynomials
  sorry

-- The main theorem statement
theorem nearly_tricky_7_tiny_count :
  count_nearly_tricky_7_tiny = 64912347 :=
sorry

end nearly_tricky_7_tiny_count_l210_210748


namespace six_digit_quotient_l210_210699

def six_digit_number (A B : ℕ) : ℕ := 100000 * A + 97860 + B

def divisible_by_99 (n : ℕ) : Prop := n % 99 = 0

theorem six_digit_quotient (A B : ℕ) (hA : A = 5) (hB : B = 1)
  (h9786B : divisible_by_99 (six_digit_number A B)) : 
  six_digit_number A B / 99 = 6039 := by
  sorry

end six_digit_quotient_l210_210699


namespace max_sum_of_squares_diff_l210_210135

theorem max_sum_of_squares_diff {x y : ℕ} (h : x > 0 ∧ y > 0) (h_diff : x^2 - y^2 = 2016) :
  x + y ≤ 1008 ∧ ∃ x' y' : ℕ, x'^2 - y'^2 = 2016 ∧ x' + y' = 1008 :=
sorry

end max_sum_of_squares_diff_l210_210135


namespace sqrt_mul_simplify_l210_210517

theorem sqrt_mul_simplify : Real.sqrt 18 * Real.sqrt 32 = 24 := 
sorry

end sqrt_mul_simplify_l210_210517


namespace distance_vancouver_calgary_l210_210265

theorem distance_vancouver_calgary : 
  ∀ (map_distance : ℝ) (scale : ℝ) (terrain_factor : ℝ), 
    map_distance = 12 →
    scale = 35 →
    terrain_factor = 1.1 →
    map_distance * scale * terrain_factor = 462 := by
  intros map_distance scale terrain_factor 
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end distance_vancouver_calgary_l210_210265


namespace sum_of_reciprocals_of_roots_l210_210456

theorem sum_of_reciprocals_of_roots (p q r : ℝ) (h : ∀ x : ℝ, (x^3 - x - 6 = 0) → (x = p ∨ x = q ∨ x = r)) :
  1 / (p + 2) + 1 / (q + 2) + 1 / (r + 2) = 11 / 12 :=
sorry

end sum_of_reciprocals_of_roots_l210_210456


namespace jack_weight_52_l210_210556

theorem jack_weight_52 (Sam Jack : ℕ) (h1 : Sam + Jack = 96) (h2 : Jack = Sam + 8) : Jack = 52 := 
by
  sorry

end jack_weight_52_l210_210556


namespace simplify_expression_l210_210033

theorem simplify_expression (a : ℝ) (h : a ≠ 1/2) : 1 - (2 / (1 + (2 * a) / (1 - 2 * a))) = 4 * a - 1 :=
by
  sorry

end simplify_expression_l210_210033


namespace choir_members_number_l210_210947

theorem choir_members_number
  (n : ℕ)
  (h1 : n % 12 = 10)
  (h2 : n % 14 = 12)
  (h3 : 300 ≤ n ∧ n ≤ 400) :
  n = 346 :=
sorry

end choir_members_number_l210_210947


namespace triangle_base_length_l210_210946

theorem triangle_base_length (h : 3 = (b * 3) / 2) : b = 2 :=
by
  sorry

end triangle_base_length_l210_210946


namespace odd_function_at_zero_l210_210361

-- Define the property of being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

theorem odd_function_at_zero (f : ℝ → ℝ) (h : is_odd_function f) : f 0 = 0 :=
by
  -- assume the definitions but leave the proof steps and focus on the final conclusion
  sorry

end odd_function_at_zero_l210_210361


namespace max_books_borrowed_l210_210894

theorem max_books_borrowed 
  (num_students : ℕ)
  (num_no_books : ℕ)
  (num_one_book : ℕ)
  (num_two_books : ℕ)
  (average_books : ℕ)
  (h_num_students : num_students = 32)
  (h_num_no_books : num_no_books = 2)
  (h_num_one_book : num_one_book = 12)
  (h_num_two_books : num_two_books = 10)
  (h_average_books : average_books = 2)
  : ∃ max_books : ℕ, max_books = 11 := 
by
  sorry

end max_books_borrowed_l210_210894


namespace determine_xyz_l210_210116

theorem determine_xyz (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x + 1/y = 5) (h5 : y + 1/z = 2) (h6 : z + 1/x = 8/3) :
  x * y * z = 8 + 3 * Real.sqrt 7 :=
by
  sorry

end determine_xyz_l210_210116


namespace no_prime_roots_of_quadratic_l210_210230

open Int Nat

theorem no_prime_roots_of_quadratic (k : ℤ) :
  ¬ (∃ p q : ℤ, Prime p ∧ Prime q ∧ p + q = 107 ∧ p * q = k) :=
by
  sorry

end no_prime_roots_of_quadratic_l210_210230


namespace calculate_dollar_value_l210_210970

def dollar (x y : ℤ) : ℤ := x * (y + 2) + x * y - 5

theorem calculate_dollar_value : dollar 3 (-1) = -5 := by
  sorry

end calculate_dollar_value_l210_210970


namespace num_sequences_to_initial_position_8_l210_210991

def validSequenceCount : ℕ := 4900

noncomputable def numberOfSequencesToInitialPosition (n : ℕ) : ℕ :=
if h : n = 8 then validSequenceCount else 0

theorem num_sequences_to_initial_position_8 :
  numberOfSequencesToInitialPosition 8 = 4900 :=
by
  sorry

end num_sequences_to_initial_position_8_l210_210991


namespace number_of_men_in_company_l210_210483

noncomputable def total_workers : ℝ := 2752.8
noncomputable def women_in_company : ℝ := 91.76
noncomputable def workers_without_retirement_plan : ℝ := (1 / 3) * total_workers
noncomputable def percent_women_without_retirement_plan : ℝ := 0.10
noncomputable def percent_men_with_retirement_plan : ℝ := 0.40
noncomputable def workers_with_retirement_plan : ℝ := (2 / 3) * total_workers
noncomputable def men_with_retirement_plan : ℝ := percent_men_with_retirement_plan * workers_with_retirement_plan

theorem number_of_men_in_company : (total_workers - women_in_company) = 2661.04 := by
  -- Insert the exact calculations and algebraic manipulations
  sorry

end number_of_men_in_company_l210_210483


namespace sum_of_geometric_sequence_first_9000_terms_l210_210761

noncomputable def geomSum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r ^ n) / (1 - r)

theorem sum_of_geometric_sequence_first_9000_terms (a r : ℝ) (h1 : geomSum a r 3000 = 500) (h2 : geomSum a r 6000 = 950) :
  geomSum a r 9000 = 1355 :=
by
  sorry

end sum_of_geometric_sequence_first_9000_terms_l210_210761


namespace find_positive_integers_with_divisors_and_sum_l210_210504

theorem find_positive_integers_with_divisors_and_sum (n : ℕ) :
  (∃ d1 d2 d3 d4 d5 d6 : ℕ,
    (n ≠ 0) ∧ (n ≠ 1) ∧ 
    n = d1 * d2 * d3 * d4 * d5 * d6 ∧
    d1 ≠ 1 ∧ d2 ≠ 1 ∧ d3 ≠ 1 ∧ d4 ≠ 1 ∧ d5 ≠ 1 ∧ d6 ≠ 1 ∧
    (d1 ≠ d2) ∧ (d1 ≠ d3) ∧ (d1 ≠ d4) ∧ (d1 ≠ d5) ∧ (d1 ≠ d6) ∧
    (d2 ≠ d3) ∧ (d2 ≠ d4) ∧ (d2 ≠ d5) ∧ (d2 ≠ d6) ∧
    (d3 ≠ d4) ∧ (d3 ≠ d5) ∧ (d3 ≠ d6) ∧
    (d4 ≠ d5) ∧ (d4 ≠ d6) ∧
    (d5 ≠ d6) ∧
    d1 + d2 + d3 + d4 + d5 + d6 = 14133
  ) -> 
  (n = 16136 ∨ n = 26666) :=
sorry

end find_positive_integers_with_divisors_and_sum_l210_210504


namespace average_number_of_ducks_l210_210154

def average_ducks (A E K : ℕ) : ℕ :=
  (A + E + K) / 3

theorem average_number_of_ducks :
  ∀ (A E K : ℕ), A = 2 * E → E = K - 45 → A = 30 → average_ducks A E K = 35 :=
by 
  intros A E K h1 h2 h3
  sorry

end average_number_of_ducks_l210_210154


namespace sarah_min_correct_l210_210256

theorem sarah_min_correct (c : ℕ) (hc : c * 8 + 10 ≥ 110) : c ≥ 13 :=
sorry

end sarah_min_correct_l210_210256


namespace isosceles_triangle_ratio_HD_HA_l210_210064

theorem isosceles_triangle_ratio_HD_HA (A B C D H : ℝ) :
  let AB := 13;
  let AC := 13;
  let BC := 10;
  let s := (AB + AC + BC) / 2;
  let area := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC));
  let h := (2 * area) / BC;
  let AD := h;
  let HA := h;
  let HD := 0;
  HD / HA = 0 := sorry

end isosceles_triangle_ratio_HD_HA_l210_210064


namespace factorization_property_l210_210580

theorem factorization_property (a b : ℤ) (h1 : 25 * x ^ 2 - 160 * x - 144 = (5 * x + a) * (5 * x + b)) 
    (h2 : a + b = -32) (h3 : a * b = -144) : 
    a + 2 * b = -68 := 
sorry

end factorization_property_l210_210580


namespace circle_radius_five_c_value_l210_210344

theorem circle_radius_five_c_value {c : ℝ} :
  (∀ x y : ℝ, x^2 + 8 * x + y^2 + 2 * y + c = 0) → 
  (∃ x y : ℝ, (x + 4)^2 + (y + 1)^2 = 25) → 
  c = 42 :=
by
  sorry

end circle_radius_five_c_value_l210_210344


namespace sequence_is_geometric_l210_210981

theorem sequence_is_geometric {a : ℝ} (h : a ≠ 0) (S : ℕ → ℝ) (H : ∀ n, S n = a^n - 1) 
: ∃ r, ∀ n, (n ≥ 1) → S n - S (n-1) = r * (S (n-1) - S (n-2)) :=
sorry

end sequence_is_geometric_l210_210981


namespace sum_of_areas_squares_l210_210554

theorem sum_of_areas_squares (a : ℕ) (h1 : (a + 4)^2 - a^2 = 80) : a^2 + (a + 4)^2 = 208 := by
  sorry

end sum_of_areas_squares_l210_210554


namespace functional_equation_to_odd_function_l210_210765

variables (f : ℝ → ℝ)

theorem functional_equation_to_odd_function (h : ∀ x y : ℝ, f (x + y) = f x + f y) :
  f 0 = 0 ∧ (∀ x : ℝ, f (-x) = -f x) :=
by
  sorry

end functional_equation_to_odd_function_l210_210765


namespace exists_sequences_x_y_l210_210914

def seq_a (a : ℕ → ℕ) : Prop :=
  a 0 = 4 ∧ a 1 = 22 ∧ ∀ n : ℕ, n ≥ 2 → a (n) = 6 * a (n - 1) - a (n - 2)

def seq_b (b : ℕ → ℕ) : Prop :=
  b 0 = 2 ∧ b 1 = 1 ∧ ∀ n : ℕ, n ≥ 2 → b (n) = 2 * b (n - 1) + b (n - 2)

theorem exists_sequences_x_y (a b : ℕ → ℕ) (x y : ℕ → ℕ) :
  seq_a a → seq_b b →
  (∀ n : ℕ, a n = (y n * y n + 7) / (x n - y n)) ↔ 
  (∀ n : ℕ, y n = b (2 * n + 1) ∧ x n = b (2 * n) + y n) :=
sorry

end exists_sequences_x_y_l210_210914


namespace range_of_a_l210_210737

theorem range_of_a (a : ℝ) (a_seq : ℕ → ℝ)
  (h1 : ∀ (n : ℕ), a_seq n = if n < 6 then (1 / 2 - a) * n + 1 else a ^ (n - 5))
  (h2 : ∀ (n : ℕ), n > 0 → a_seq n > a_seq (n + 1)) :
  (1 / 2 : ℝ) < a ∧ a < (7 / 12 : ℝ) :=
sorry

end range_of_a_l210_210737


namespace find_symmetric_curve_equation_l210_210571

def equation_of_curve_symmetric_to_line : Prop :=
  ∀ (x y : ℝ), (5 * x^2 + 12 * x * y - 22 * x - 12 * y - 19 = 0 ∧ x - y + 2 = 0) →
  12 * x * y + 5 * y^2 - 78 * y + 45 = 0

theorem find_symmetric_curve_equation : equation_of_curve_symmetric_to_line :=
sorry

end find_symmetric_curve_equation_l210_210571


namespace find_cost_price_per_meter_l210_210252

noncomputable def cost_price_per_meter
  (total_cloth : ℕ)
  (selling_price : ℕ)
  (profit_per_meter : ℕ) : ℕ :=
  (selling_price - profit_per_meter * total_cloth) / total_cloth

theorem find_cost_price_per_meter :
  cost_price_per_meter 75 4950 15 = 51 :=
by
  unfold cost_price_per_meter
  sorry

end find_cost_price_per_meter_l210_210252


namespace Energetics_factory_l210_210072

/-- In the country "Energetics," there are 150 factories, and some of them are connected by bus
routes that do not stop anywhere except at these factories. It turns out that any four factories
can be split into two pairs such that a bus runs between each pair of factories. Find the minimum
number of pairs of factories that can be connected by bus routes. -/
theorem Energetics_factory
  (factories : Finset ℕ) (routes : Finset (ℕ × ℕ))
  (h_factories : factories.card = 150)
  (h_routes : ∀ (X Y Z W : ℕ),
    {X, Y, Z, W} ⊆ factories →
    ∃ (X1 Y1 Z1 W1 : ℕ),
    (X1, Y1) ∈ routes ∧
    (Z1, W1) ∈ routes ∧
    (X1 = X ∨ X1 = Y ∨ X1 = Z ∨ X1 = W) ∧
    (Y1 = X ∨ Y1 = Y ∨ Y1 = Z ∨ Y1 = W) ∧
    (Z1 = X ∨ Z1 = Y ∨ Z1 = Z ∨ Z1 = W) ∧
    (W1 = X ∨ W1 = Y ∨ W1 = Z ∨ W1 = W)) :
  (2 * routes.card) ≥ 11025 := sorry

end Energetics_factory_l210_210072


namespace determine_m_l210_210245

def f (x m : ℝ) : ℝ := x^2 - 3*x + m
def g (x m : ℝ) : ℝ := x^2 - 3*x + 5*m

theorem determine_m (m : ℝ) : 3 * f 5 m = 2 * g 5 m → m = 10 / 7 := 
by sorry

end determine_m_l210_210245


namespace ab_ac_plus_bc_range_l210_210140

theorem ab_ac_plus_bc_range (a b c : ℝ) (h : a + b + 2 * c = 0) :
  ∃ (k : ℝ), k ≤ 0 ∧ k = ab + ac + bc :=
sorry

end ab_ac_plus_bc_range_l210_210140


namespace total_digits_first_2003_even_integers_l210_210007

theorem total_digits_first_2003_even_integers : 
  let even_integers := (List.range' 1 (2003 * 2)).filter (λ n => n % 2 = 0)
  let one_digit_count := List.filter (λ n => n < 10) even_integers |>.length
  let two_digit_count := List.filter (λ n => 10 ≤ n ∧ n < 100) even_integers |>.length
  let three_digit_count := List.filter (λ n => 100 ≤ n ∧ n < 1000) even_integers |>.length
  let four_digit_count := List.filter (λ n => 1000 ≤ n) even_integers |>.length
  let total_digits := one_digit_count * 1 + two_digit_count * 2 + three_digit_count * 3 + four_digit_count * 4
  total_digits = 7460 :=
by
  sorry

end total_digits_first_2003_even_integers_l210_210007


namespace factor_polynomial_l210_210105

theorem factor_polynomial (x : ℝ) : 
  (20 * x^3 + 100 * x - 10) - (-3 * x^3 + 5 * x - 15) = 5 * (23 * x^3 + 19 * x + 1) := 
by 
  -- Proof can be filled in here
  sorry

end factor_polynomial_l210_210105


namespace chicago_denver_temperature_l210_210280

def temperature_problem (C D : ℝ) (N : ℝ) : Prop :=
  (C = D - N) ∧ (abs ((D - N + 4) - (D - 2)) = 1)

theorem chicago_denver_temperature (C D N : ℝ) (h : temperature_problem C D N) :
  N = 5 ∨ N = 7 → (5 * 7 = 35) :=
by sorry

end chicago_denver_temperature_l210_210280


namespace solve_equation_l210_210670

theorem solve_equation : 
  ∀ x : ℝ, (x^2 + 2*x + 3)/(x + 2) = x + 4 → x = -(5/4) := by
  sorry

end solve_equation_l210_210670


namespace pens_more_than_notebooks_l210_210403

theorem pens_more_than_notebooks
  (N P : ℕ) 
  (h₁ : N = 30) 
  (h₂ : N + P = 110) :
  P - N = 50 := 
by
  sorry

end pens_more_than_notebooks_l210_210403


namespace solve_for_n_l210_210564

theorem solve_for_n (n : ℝ) (h : 0.05 * n + 0.1 * (30 + n) - 0.02 * n = 15.5) : n = 96 := 
by 
  sorry

end solve_for_n_l210_210564


namespace sum_of_roots_l210_210471

theorem sum_of_roots : ∀ x : ℝ, x^2 - 2004 * x + 2021 = 0 → x = 2004 := by
  sorry

end sum_of_roots_l210_210471


namespace cereal_original_price_l210_210382

-- Define the known conditions as constants
def initial_money : ℕ := 60
def celery_price : ℕ := 5
def bread_price : ℕ := 8
def milk_full_price : ℕ := 10
def milk_discount : ℕ := 10
def milk_price : ℕ := milk_full_price - (milk_full_price * milk_discount / 100)
def potato_price : ℕ := 1
def potato_quantity : ℕ := 6
def potatoes_total_price : ℕ := potato_price * potato_quantity
def coffee_remaining_money : ℕ := 26
def total_spent_exclude_coffee : ℕ := initial_money - coffee_remaining_money
def spent_on_other_items : ℕ := celery_price + bread_price + milk_price + potatoes_total_price
def spent_on_cereal : ℕ := total_spent_exclude_coffee - spent_on_other_items
def cereal_discount : ℕ := 50

theorem cereal_original_price :
  (spent_on_other_items = celery_price + bread_price + milk_price + potatoes_total_price) →
  (total_spent_exclude_coffee = initial_money - coffee_remaining_money) →
  (spent_on_cereal = total_spent_exclude_coffee - spent_on_other_items) →
  (spent_on_cereal * 2 = 12) :=
by {
  -- proof here
  sorry
}

end cereal_original_price_l210_210382


namespace symmetric_function_is_periodic_l210_210169

theorem symmetric_function_is_periodic {f : ℝ → ℝ} {a b y0 : ℝ}
  (h1 : ∀ x, f (a + x) - y0 = y0 - f (a - x))
  (h2 : ∀ x, f (b + x) = f (b - x))
  (hb : b > a) :
  ∀ x, f (x + 4 * (b - a)) = f x := sorry

end symmetric_function_is_periodic_l210_210169


namespace ancient_chinese_silver_problem_l210_210093

theorem ancient_chinese_silver_problem :
  ∃ (x y : ℤ), 7 * x = y - 4 ∧ 9 * x = y + 8 :=
by
  sorry

end ancient_chinese_silver_problem_l210_210093


namespace findingRealNumsPureImaginary_l210_210876

theorem findingRealNumsPureImaginary :
  ∀ x : ℝ, ((x + Complex.I * 2) * ((x + 2) + Complex.I * 2) * ((x + 4) + Complex.I * 2)).im = 0 → 
    x = -4 ∨ x = -1 + 2 * Real.sqrt 5 ∨ x = -1 - 2 * Real.sqrt 5 :=
by
  intros x h
  let expr := x^3 + 6*x^2 + 4*x - 16
  have h_real_part_eq_0 : expr = 0 := sorry
  have solutions_correct :
    expr = 0 → (x = -4 ∨ x = -1 + 2 * Real.sqrt 5 ∨ x = -1 - 2 * Real.sqrt 5) := sorry
  exact solutions_correct h_real_part_eq_0

end findingRealNumsPureImaginary_l210_210876


namespace Walter_bus_time_l210_210143

theorem Walter_bus_time :
  let start_time := 7 * 60 + 30 -- 7:30 a.m. in minutes
  let end_time := 16 * 60 + 15 -- 4:15 p.m. in minutes
  let away_time := end_time - start_time -- total time away from home in minutes
  let classes_time := 7 * 45 -- 7 classes 45 minutes each
  let lunch_time := 40 -- lunch time in minutes
  let additional_school_time := 1.5 * 60 -- additional time at school in minutes
  let school_time := classes_time + lunch_time + additional_school_time -- total school activities time
  (away_time - school_time) = 80 :=
by
  sorry

end Walter_bus_time_l210_210143


namespace find_x_in_interval_l210_210818

theorem find_x_in_interval (x : ℝ) : x^2 + 5 * x < 10 ↔ -5 < x ∧ x < 2 :=
sorry

end find_x_in_interval_l210_210818


namespace area_of_quadrilateral_l210_210348

-- Definitions of the given conditions
def diagonal_length : ℝ := 40
def offset1 : ℝ := 11
def offset2 : ℝ := 9

-- The area of the quadrilateral
def quadrilateral_area : ℝ := 400

-- Proof statement
theorem area_of_quadrilateral :
  (1/2 * diagonal_length * offset1 + 1/2 * diagonal_length * offset2) = quadrilateral_area :=
by sorry

end area_of_quadrilateral_l210_210348


namespace hexagon_equilateral_triangles_l210_210177

theorem hexagon_equilateral_triangles (hexagon_area: ℝ) (num_hexagons : ℕ) (tri_area: ℝ) 
    (h1 : hexagon_area = 6) (h2 : num_hexagons = 4) (h3 : tri_area = 4) : 
    ∃ (num_triangles : ℕ), num_triangles = 8 := 
by
  sorry

end hexagon_equilateral_triangles_l210_210177


namespace max_value_quadratic_l210_210479

theorem max_value_quadratic : ∀ s : ℝ, ∃ M : ℝ, (∀ s : ℝ, -3 * s^2 + 54 * s - 27 ≤ M) ∧ M = 216 :=
by
  sorry

end max_value_quadratic_l210_210479


namespace number_of_yellow_marbles_l210_210062

theorem number_of_yellow_marbles (Y : ℕ) (h : Y / (7 + 11 + Y) = 1 / 4) : Y = 6 :=
by
  -- Proof to be filled in
  sorry

end number_of_yellow_marbles_l210_210062


namespace y_exceeds_x_by_100_percent_l210_210489

theorem y_exceeds_x_by_100_percent (x y : ℝ) (h : x = 0.5 * y) : (y - x) / x = 1 := by
sorry

end y_exceeds_x_by_100_percent_l210_210489


namespace sandwiches_provided_l210_210926

theorem sandwiches_provided (original_count sold_out : ℕ) (h1 : original_count = 9) (h2 : sold_out = 5) : (original_count - sold_out = 4) :=
by
  sorry

end sandwiches_provided_l210_210926


namespace ratio_sum_2_or_4_l210_210764

theorem ratio_sum_2_or_4 (a b c d : ℝ) 
  (h1 : a / b + b / c + c / d + d / a = 6)
  (h2 : a / c + b / d + c / a + d / b = 8) : 
  (a / b + c / d = 2) ∨ (a / b + c / d = 4) :=
sorry

end ratio_sum_2_or_4_l210_210764


namespace first_two_digits_of_1666_l210_210270

/-- Lean 4 statement for the given problem -/
theorem first_two_digits_of_1666 (y k : ℕ) (H_nonzero_k : k ≠ 0) (H_nonzero_y : y ≠ 0) (H_y_six : y = 6) :
  (1666 / 100) = 16 := by
  sorry

end first_two_digits_of_1666_l210_210270


namespace min_value_a_decreasing_range_of_a_x1_x2_l210_210429

noncomputable def f (a x : ℝ) := x / Real.log x - a * x

theorem min_value_a_decreasing :
  ∀ (a : ℝ), (∀ (x : ℝ), 1 < x → f a x <= 0) → a ≥ 1 / 4 :=
sorry

theorem range_of_a_x1_x2 :
  ∀ (a : ℝ), (∃ (x₁ x₂ : ℝ), e ≤ x₁ ∧ x₁ ≤ e^2 ∧ e ≤ x₂ ∧ x₂ ≤ e^2 ∧ f a x₁ ≤ f a x₂ + a)
  → a ≥ 1 / 2 - 1 / (4 * e^2) :=
sorry

end min_value_a_decreasing_range_of_a_x1_x2_l210_210429


namespace elongation_rate_significantly_improved_l210_210462

noncomputable def elongation_improvement : Prop :=
  let x : List ℝ := [545, 533, 551, 522, 575, 544, 541, 568, 596, 548]
  let y : List ℝ := [536, 527, 543, 530, 560, 533, 522, 550, 576, 536]
  let z := List.zipWith (λ xi yi => xi - yi) x y
  let n : ℝ := 10
  let mean_z := (List.sum z) / n
  let variance_z := (List.sum (List.map (λ zi => (zi - mean_z)^2) z)) / n
  mean_z = 11 ∧ 
  variance_z = 61 ∧ 
  mean_z ≥ 2 * Real.sqrt (variance_z / n)

-- We state the theorem without proof
theorem elongation_rate_significantly_improved : elongation_improvement :=
by
  -- Proof can be written here
  sorry

end elongation_rate_significantly_improved_l210_210462


namespace number_of_chocolate_bars_by_theresa_l210_210418

-- Define the number of chocolate bars and soda cans that Kayla bought
variables (C S : ℕ)

-- Assume the total number of chocolate bars and soda cans Kayla bought is 15
axiom total_purchased_by_kayla : C + S = 15

-- Define the number of chocolate bars Theresa bought as twice the number Kayla bought
def chocolate_bars_purchased_by_theresa := 2 * C

-- The theorem to prove
theorem number_of_chocolate_bars_by_theresa : chocolate_bars_purchased_by_theresa = 2 * C :=
by
  -- The proof is omitted as instructed
  sorry

end number_of_chocolate_bars_by_theresa_l210_210418


namespace prime_gt3_43_divides_expression_l210_210631

theorem prime_gt3_43_divides_expression {p : ℕ} (hp : Nat.Prime p) (hp_gt_3 : p > 3) : 
  (7^p - 6^p - 1) % 43 = 0 := 
  sorry

end prime_gt3_43_divides_expression_l210_210631


namespace probability_of_at_least_one_boy_and_one_girl_is_correct_l210_210910

noncomputable def probability_at_least_one_boy_and_one_girl : ℚ :=
  (1 - ((1/2)^4 + (1/2)^4))

theorem probability_of_at_least_one_boy_and_one_girl_is_correct : 
  probability_at_least_one_boy_and_one_girl = 7/8 :=
by
  sorry

end probability_of_at_least_one_boy_and_one_girl_is_correct_l210_210910


namespace monotonicity_f_geq_f_neg_l210_210955

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem monotonicity (a : ℝ) :
  (a ≤ 0 → ∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2) ∧
  (a > 0 →
    (∀ x1 x2 : ℝ, x1 > Real.log a → x2 > Real.log a → x1 ≤ x2 → f a x1 ≤ f a x2) ∧
    (∀ x1 x2 : ℝ, x1 < Real.log a → x2 < Real.log a → x1 ≤ x2 → f a x1 ≤ f a x2)) :=
by sorry

theorem f_geq_f_neg (x : ℝ) (hx : x ≥ 0) : f 1 x ≥ f 1 (-x) :=
by sorry

end monotonicity_f_geq_f_neg_l210_210955


namespace min_value_proof_l210_210217

noncomputable def min_value (a b : ℝ) : ℝ := (1 : ℝ)/a + (1 : ℝ)/b

theorem min_value_proof (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + 2 * b = 2) :
  min_value a b = (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end min_value_proof_l210_210217


namespace proof_problem_l210_210704

noncomputable def problem_statement : Prop :=
  let p1 := ∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 - x + m = 0
  let p2 := ∀ x y : ℝ, x + y > 2 → x > 1 ∧ y > 1
  let p3 := ∃ x : ℝ, -2 < x ∧ x < 4 ∧ |x - 2| ≥ 3
  let p4 := ∀ a b c : ℝ, a ≠ 0 ∧ b^2 - 4 * a * c > 0 → ∃ x₁ x₂ : ℝ, x₁ * x₂ < 0
  p3 = true ∧ p1 = false ∧ p2 = false ∧ p4 = false

theorem proof_problem : problem_statement := 
sorry

end proof_problem_l210_210704


namespace parabola_opens_downwards_l210_210993

theorem parabola_opens_downwards (a : ℝ) (h : ℝ) (k : ℝ) :
  a < 0 → h = 3 → ∃ k, (∀ x, y = a * (x - h) ^ 2 + k → y = -(x - 3)^2 + k) :=
by
  intros ha hh
  use k
  sorry

end parabola_opens_downwards_l210_210993


namespace masks_purchased_in_first_batch_l210_210769

theorem masks_purchased_in_first_batch
    (cost_first_batch cost_second_batch : ℝ)
    (quantity_ratio : ℝ)
    (unit_price_difference : ℝ)
    (h1 : cost_first_batch = 1600)
    (h2 : cost_second_batch = 6000)
    (h3 : quantity_ratio = 3)
    (h4 : unit_price_difference = 2) :
    ∃ x : ℝ, (cost_first_batch / x) + unit_price_difference = (cost_second_batch / (quantity_ratio * x)) ∧ x = 200 :=
by {
    sorry
}

end masks_purchased_in_first_batch_l210_210769


namespace selection_probability_equal_l210_210050

theorem selection_probability_equal :
  let n := 2012
  let eliminated := 12
  let remaining := n - eliminated
  let selected := 50
  let probability := (remaining / n) * (selected / remaining)
  probability = 25 / 1006 :=
by
  sorry

end selection_probability_equal_l210_210050


namespace product_inequality_l210_210505

variable (x1 x2 x3 x4 y1 y2 : ℝ)

theorem product_inequality (h1 : y2 ≥ y1) 
                          (h2 : y1 ≥ x1)
                          (h3 : x1 ≥ x3)
                          (h4 : x3 ≥ x2)
                          (h5 : x2 ≥ x1)
                          (h6 : x1 ≥ 2)
                          (h7 : x1 + x2 + x3 + x4 ≥ y1 + y2) : 
                          x1 * x2 * x3 * x4 ≥ y1 * y2 :=
  sorry

end product_inequality_l210_210505


namespace prism_dimensions_l210_210570

theorem prism_dimensions (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 45) (h3 : b * c = 60) : 
  a = 7.2 ∧ b = 9.6 ∧ c = 14.4 :=
by {
  -- Proof skipped for now
  sorry
}

end prism_dimensions_l210_210570


namespace initial_antifreeze_percentage_l210_210019

-- Definitions of conditions
def total_volume : ℚ := 10
def replaced_volume : ℚ := 2.85714285714
def final_percentage : ℚ := 50 / 100

-- Statement to prove
theorem initial_antifreeze_percentage (P : ℚ) :
  10 * P / 100 - P / 100 * 2.85714285714 + 2.85714285714 = 5 → 
  P = 30 :=
sorry

end initial_antifreeze_percentage_l210_210019


namespace height_of_boxes_l210_210395

theorem height_of_boxes
  (volume_required : ℝ)
  (price_per_box : ℝ)
  (min_expenditure : ℝ)
  (volume_per_box : ∀ n : ℕ, n = min_expenditure / price_per_box -> ℝ) :
  volume_required = 3060000 ->
  price_per_box = 0.50 ->
  min_expenditure = 255 ->
  ∃ h : ℝ, h = 19 := by
  sorry

end height_of_boxes_l210_210395


namespace barefoot_kids_count_l210_210726

def kidsInClassroom : Nat := 35
def kidsWearingSocks : Nat := 18
def kidsWearingShoes : Nat := 15
def kidsWearingBoth : Nat := 8

def barefootKids : Nat := kidsInClassroom - (kidsWearingSocks - kidsWearingBoth + kidsWearingShoes - kidsWearingBoth + kidsWearingBoth)

theorem barefoot_kids_count : barefootKids = 10 := by
  sorry

end barefoot_kids_count_l210_210726


namespace divisor_of_p_l210_210121

theorem divisor_of_p (p q r s : ℕ) (hpq : Nat.gcd p q = 40)
  (hqr : Nat.gcd q r = 45) (hrs : Nat.gcd r s = 60)
  (hspr : 100 < Nat.gcd s p ∧ Nat.gcd s p < 150)
  : 7 ∣ p :=
sorry

end divisor_of_p_l210_210121


namespace evaluate_expression_when_c_eq_4_and_k_eq_2_l210_210242

theorem evaluate_expression_when_c_eq_4_and_k_eq_2 :
  ( (4^4 - 4 * (4 - 1)^4 + 2) ^ 4 ) = 18974736 :=
by
  -- Definitions
  let c := 4
  let k := 2
  -- Evaluations
  let a := c^c
  let b := c * (c - 1)^c
  let expression := (a - b + k)^c
  -- Proof
  have result : expression = 18974736 := sorry
  exact result

end evaluate_expression_when_c_eq_4_and_k_eq_2_l210_210242


namespace range_of_a_l210_210323

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 < 1 ∧ x2 > 1 ∧ x1 * x1 + (a * a - 1) * x1 + a - 2 = 0 ∧ x2 * x2 + (a * a - 1) * x2 + a - 2 = 0) ↔ -2 < a ∧ a < 1 :=
sorry

end range_of_a_l210_210323


namespace max_value_expression_l210_210023

theorem max_value_expression : 
  ∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 2 →
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) ≤ 256 / 243 :=
by
  intros x y z hx hy hz hsum
  sorry

end max_value_expression_l210_210023


namespace max_median_soda_cans_l210_210178

theorem max_median_soda_cans (total_customers total_cans : ℕ) 
    (h_customers : total_customers = 120)
    (h_cans : total_cans = 300) 
    (h_min_cans_per_customer : ∀ (n : ℕ), n < total_customers → 2 ≤ n) :
    ∃ (median : ℝ), median = 3.5 := 
sorry

end max_median_soda_cans_l210_210178


namespace domain_of_g_l210_210389

theorem domain_of_g (t : ℝ) : (t - 1)^2 + (t + 1)^2 + t ≠ 0 :=
  by
  sorry

end domain_of_g_l210_210389


namespace tan_alpha_plus_pi_over_4_l210_210669

theorem tan_alpha_plus_pi_over_4 (x y : ℝ) (h1 : 3 * x + 4 * y = 0) : 
  Real.tan ((Real.arctan (- 3 / 4)) + π / 4) = 1 / 7 := 
by
  sorry

end tan_alpha_plus_pi_over_4_l210_210669


namespace find_a_l210_210047

theorem find_a (a n : ℝ) (p : ℝ) (hp : p = 2 / 3)
  (h₁ : a = 3 * n + 5)
  (h₂ : a + 2 = 3 * (n + p) + 5) : a = 3 * n + 5 :=
by 
  sorry

end find_a_l210_210047


namespace cheapest_candle_cost_to_measure_1_minute_l210_210448

-- Definitions

def big_candle_cost := 16 -- cost of a big candle in cents
def big_candle_burn_time := 16 -- burn time of a big candle in minutes
def small_candle_cost := 7 -- cost of a small candle in cents
def small_candle_burn_time := 7 -- burn time of a small candle in minutes

-- Problem statement
theorem cheapest_candle_cost_to_measure_1_minute :
  (∃ (n m : ℕ), n * big_candle_burn_time - m * small_candle_burn_time = 1 ∧
                 n * big_candle_cost + m * small_candle_cost = 97) :=
sorry

end cheapest_candle_cost_to_measure_1_minute_l210_210448


namespace profit_is_correct_l210_210813

-- Definitions of the conditions
def initial_outlay : ℕ := 10000
def cost_per_set : ℕ := 20
def price_per_set : ℕ := 50
def sets_sold : ℕ := 500

-- Derived calculations
def revenue (sets_sold : ℕ) (price_per_set : ℕ) : ℕ :=
  sets_sold * price_per_set

def manufacturing_costs (initial_outlay : ℕ) (cost_per_set : ℕ) (sets_sold : ℕ) : ℕ :=
  initial_outlay + (cost_per_set * sets_sold)

def profit (revenue : ℕ) (manufacturing_costs : ℕ) : ℕ :=
  revenue - manufacturing_costs

-- Theorem stating the problem
theorem profit_is_correct : 
  profit (revenue sets_sold price_per_set) (manufacturing_costs initial_outlay cost_per_set sets_sold) = 5000 :=
by
  sorry

end profit_is_correct_l210_210813


namespace seven_whole_numbers_cannot_form_ten_consecutive_pairwise_sums_l210_210141

theorem seven_whole_numbers_cannot_form_ten_consecutive_pairwise_sums
  (a1 a2 a3 a4 a5 a6 a7 : Nat) :
  ¬ ∃ (s : Finset Nat), (s = {a1 + a2, a1 + a3, a1 + a4, a1 + a5, a1 + a6, a1 + a7,
                             a2 + a3, a2 + a4, a2 + a5, a2 + a6, a2 + a7,
                             a3 + a4, a3 + a5, a3 + a6, a3 + a7,
                             a4 + a5, a4 + a6, a4 + a7,
                             a5 + a6, a5 + a7,
                             a6 + a7}) ∧
  (∃ (n : Nat), s = {n, n+1, n+2, n+3, n+4, n+5, n+6, n+7, n+8, n+9}) := 
sorry

end seven_whole_numbers_cannot_form_ten_consecutive_pairwise_sums_l210_210141


namespace p_6_eq_163_l210_210691

noncomputable def p (x : ℕ) : ℕ :=
  (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) + x^2 + x + 1

theorem p_6_eq_163 : p 6 = 163 :=
by
  sorry

end p_6_eq_163_l210_210691


namespace Cody_games_l210_210828

/-- Cody had nine old video games he wanted to get rid of.
He decided to give four of the games to his friend Jake,
three games to his friend Sarah, and one game to his friend Luke.
On Saturday he bought five new games.
How many games does Cody have now? -/
theorem Cody_games (nine_games initially: ℕ) (jake_games: ℕ) (sarah_games: ℕ) (luke_games: ℕ) (saturday_games: ℕ)
  (h_initial: initially = 9)
  (h_jake: jake_games = 4)
  (h_sarah: sarah_games = 3)
  (h_luke: luke_games = 1)
  (h_saturday: saturday_games = 5) :
  ((initially - (jake_games + sarah_games + luke_games)) + saturday_games) = 6 :=
by
  sorry

end Cody_games_l210_210828


namespace boys_girls_ratio_l210_210995

theorem boys_girls_ratio (T G : ℕ) (h : (1/2 : ℚ) * G = (1/6 : ℚ) * T) :
  ((T - G) : ℚ) / G = 2 :=
by 
  sorry

end boys_girls_ratio_l210_210995


namespace mixed_groups_count_l210_210741

-- Define the facts about the groups and photographs
def numberOfChildren : Nat := 300
def numberOfGroups : Nat := 100
def childrenPerGroup : Nat := 3
def b_b_photos : Nat := 100
def g_g_photos : Nat := 56

-- Define the function to calculate mixed groups
def mixedGroups (totalPhotos b_b_photos g_g_photos : Nat) : Nat := 
  (totalPhotos - b_b_photos - g_g_photos) / 2

-- State the theorem
theorem mixed_groups_count : 
  mixedGroups (numberOfGroups * childrenPerGroup) b_b_photos g_g_photos = 72 := by
  rfl

end mixed_groups_count_l210_210741


namespace fraction_of_lollipops_given_to_emily_is_2_3_l210_210601

-- Given conditions as definitions
def initial_lollipops := 42
def kept_lollipops := 4
def lou_received := 10

-- The fraction of lollipops given to Emily
def fraction_given_to_emily : ℚ :=
  have emily_received : ℚ := initial_lollipops - (kept_lollipops + lou_received)
  have total_lollipops : ℚ := initial_lollipops
  emily_received / total_lollipops

-- The proof statement assert that fraction_given_to_emily is equal to 2/3
theorem fraction_of_lollipops_given_to_emily_is_2_3 : fraction_given_to_emily = 2 / 3 := by
  sorry

end fraction_of_lollipops_given_to_emily_is_2_3_l210_210601


namespace wrapping_third_roll_l210_210190

theorem wrapping_third_roll (total_gifts first_roll_gifts second_roll_gifts third_roll_gifts : ℕ) 
  (h1 : total_gifts = 12) (h2 : first_roll_gifts = 3) (h3 : second_roll_gifts = 5) 
  (h4 : third_roll_gifts = total_gifts - (first_roll_gifts + second_roll_gifts)) :
  third_roll_gifts = 4 :=
sorry

end wrapping_third_roll_l210_210190


namespace proof_method_characterization_l210_210932

-- Definitions of each method
def synthetic_method := "proceeds from cause to effect, in a forward manner"
def analytic_method := "seeks the cause from the effect, working backwards"
def proof_by_contradiction := "assumes the negation of the proposition to be true, and derives a contradiction"
def mathematical_induction := "base case and inductive step: which shows that P holds for all natural numbers"

-- Main theorem to prove
theorem proof_method_characterization :
  (analytic_method == "seeks the cause from the effect, working backwards") :=
by
  sorry

end proof_method_characterization_l210_210932


namespace find_m_l210_210990

-- Define points O, A, B, C
def O : (ℝ × ℝ) := (0, 0)
def A : (ℝ × ℝ) := (2, 3)
def B : (ℝ × ℝ) := (1, 5)
def C (m : ℝ) : (ℝ × ℝ) := (m, 3)

-- Define vectors AB and OC
def vector_AB : (ℝ × ℝ) := (B.1 - A.1, B.2 - A.2)  -- (B - A)
def vector_OC (m : ℝ) : (ℝ × ℝ) := (m, 3)  -- (C - O)

-- Define the dot product
def dot_product (v₁ v₂ : (ℝ × ℝ)) : ℝ := (v₁.1 * v₂.1) + (v₁.2 * v₂.2)

-- Theorem: vector_AB ⊥ vector_OC implies m = 6
theorem find_m (m : ℝ) (h : dot_product vector_AB (vector_OC m) = 0) : m = 6 :=
by
  -- Proof part not required
  sorry

end find_m_l210_210990


namespace pages_in_first_issue_l210_210742

-- Define variables for the number of pages in the issues and total pages
variables (P : ℕ) (total_pages : ℕ) (eqn : total_pages = 3 * P + 4)

-- State the theorem using the given conditions and question
theorem pages_in_first_issue (h : total_pages = 220) : P = 72 :=
by
  -- Use the given equation
  have h_eqn : total_pages = 3 * P + 4 := eqn
  sorry

end pages_in_first_issue_l210_210742


namespace range_of_b_l210_210441

noncomputable def f (x a b : ℝ) : ℝ :=
  x + a / x + b

theorem range_of_b (b : ℝ) :
  (∀ (a x : ℝ), (1/2 ≤ a ∧ a ≤ 2) ∧ (1/4 ≤ x ∧ x ≤ 1) → f x a b ≤ 10) →
  b ≤ 7 / 4 :=
by
  sorry

end range_of_b_l210_210441


namespace given_sequence_find_a_and_b_l210_210264

-- Define the general pattern of the sequence
def sequence_pattern (n a b : ℕ) : Prop :=
  n + (b / a : ℚ) = (n^2 : ℚ) * (b / a : ℚ)

-- State the specific case for n = 9
def sequence_case_for_9 (a b : ℕ) : Prop :=
  sequence_pattern 9 a b ∧ a + b = 89

-- Now, structure this as a theorem to be proven in Lean
theorem given_sequence_find_a_and_b :
  ∃ (a b : ℕ), sequence_case_for_9 a b :=
sorry

end given_sequence_find_a_and_b_l210_210264


namespace cube_surface_area_l210_210313

open Real

theorem cube_surface_area (V : ℝ) (a : ℝ) (S : ℝ)
  (h1 : V = a ^ 3)
  (h2 : a = 4)
  (h3 : V = 64) :
  S = 6 * a ^ 2 :=
by
  sorry

end cube_surface_area_l210_210313


namespace max_objective_value_l210_210073

theorem max_objective_value (x y : ℝ) (h1 : x - y - 2 ≥ 0) (h2 : 2 * x + y - 2 ≤ 0) (h3 : y + 4 ≥ 0) :
  ∃ (z : ℝ), z = 4 * x + 3 * y ∧ z ≤ 8 :=
sorry

end max_objective_value_l210_210073


namespace num_envelopes_requiring_charge_l210_210461

structure Envelope where
  length : ℕ
  height : ℕ

def requiresExtraCharge (env : Envelope) : Bool :=
  let ratio := env.length / env.height
  ratio < 3/2 ∨ ratio > 3

def envelopes : List Envelope :=
  [{ length := 7, height := 5 },  -- E
   { length := 10, height := 2 }, -- F
   { length := 8, height := 8 },  -- G
   { length := 12, height := 3 }] -- H

def countExtraChargedEnvelopes : ℕ :=
  envelopes.filter requiresExtraCharge |>.length

theorem num_envelopes_requiring_charge : countExtraChargedEnvelopes = 4 := by
  sorry

end num_envelopes_requiring_charge_l210_210461


namespace car_highway_mileage_l210_210110

theorem car_highway_mileage :
  (∀ (H : ℝ), 
    (H > 0) → 
    (4 / H + 4 / 20 = (8 / H) * 1.4000000000000001) → 
    (H = 36)) :=
by
  intros H H_pos h_cond
  have : H = 36 := 
    sorry
  exact this

end car_highway_mileage_l210_210110


namespace profit_ratio_l210_210806

theorem profit_ratio (P_invest Q_invest : ℕ) (hP : P_invest = 500000) (hQ : Q_invest = 1000000) :
  (P_invest:ℚ) / Q_invest = 1 / 2 := 
  by
  rw [hP, hQ]
  norm_num

end profit_ratio_l210_210806


namespace bottles_remaining_after_2_days_l210_210162

def total_bottles := 48 

def first_day_father_consumption := total_bottles / 4
def first_day_mother_consumption := total_bottles / 6
def first_day_son_consumption := total_bottles / 8

def total_first_day_consumption := first_day_father_consumption + first_day_mother_consumption + first_day_son_consumption 
def remaining_after_first_day := total_bottles - total_first_day_consumption

def second_day_father_consumption := remaining_after_first_day / 5
def remaining_after_father := remaining_after_first_day - second_day_father_consumption
def second_day_mother_consumption := remaining_after_father / 7
def remaining_after_mother := remaining_after_father - second_day_mother_consumption
def second_day_son_consumption := remaining_after_mother / 9
def remaining_after_son := remaining_after_mother - second_day_son_consumption
def second_day_daughter_consumption := remaining_after_son / 9
def remaining_after_daughter := remaining_after_son - second_day_daughter_consumption

theorem bottles_remaining_after_2_days : ∀ (total_bottles : ℕ), remaining_after_daughter = 14 := 
by
  sorry

end bottles_remaining_after_2_days_l210_210162


namespace functional_equation_solution_l210_210283

variable (f : ℝ → ℝ)

-- Declare the conditions as hypotheses
axiom cond1 : ∀ x : ℝ, 0 < x → 0 < f x
axiom cond2 : f 1 = 1
axiom cond3 : ∀ a b : ℝ, f (a + b) * (f a + f b) = 2 * f a * f b + a^2 + b^2

-- State the theorem to be proved
theorem functional_equation_solution : ∀ x : ℝ, f x = x :=
sorry

end functional_equation_solution_l210_210283


namespace Danielle_rooms_is_6_l210_210302

-- Definitions for the problem conditions
def Heidi_rooms (Danielle_rooms : ℕ) : ℕ := 3 * Danielle_rooms
def Grant_rooms (Heidi_rooms : ℕ) : ℕ := Heidi_rooms / 9
def Grant_rooms_value : ℕ := 2

-- Theorem statement
theorem Danielle_rooms_is_6 (h : Grant_rooms_value = Grant_rooms (Heidi_rooms d)) : d = 6 :=
by
  sorry

end Danielle_rooms_is_6_l210_210302


namespace solve_for_x_l210_210015

theorem solve_for_x (x : ℤ) (h : 3 * x + 7 = -2) : x = -3 :=
by
  sorry

end solve_for_x_l210_210015


namespace problem_1_problem_2_l210_210919

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2
noncomputable def g (x : ℝ) : ℝ := 1 / 2 + Real.sqrt 3 * Real.sin x * Real.cos x
noncomputable def h (x : ℝ) : ℝ := f x + g x

-- Proof Problem 1
theorem problem_1 (a : ℝ) (h_symmetry : ∃ k : ℤ, a = k * Real.pi / 2) : g (2 * a) = 1 / 2 := by
  sorry

-- Proof Problem 2
theorem problem_2 (x : ℝ) (h_range : 0 ≤ x ∧ x ≤ Real.pi / 2) : 
  ∃ y : ℝ, y = h x ∧ 1/2 ≤ y ∧ y ≤ 2 := by
  sorry

end problem_1_problem_2_l210_210919


namespace partition_equation_solution_l210_210805

def partition (n : ℕ) : ℕ := sorry -- defining the partition function

theorem partition_equation_solution (n : ℕ) (h : partition n + partition (n + 4) = partition (n + 2) + partition (n + 3)) :
  n = 1 ∨ n = 3 ∨ n = 5 :=
sorry

end partition_equation_solution_l210_210805


namespace prob_at_least_3_correct_l210_210527

-- Define the probability of one patient being cured
def prob_cured : ℝ := 0.9

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the probability of exactly 3 out of 4 patients being cured
def prob_exactly_3 : ℝ :=
  binomial 4 3 * prob_cured^3 * (1 - prob_cured)

-- Define the probability of all 4 patients being cured
def prob_all_4 : ℝ :=
  prob_cured^4

-- Define the probability of at least 3 out of 4 patients being cured
def prob_at_least_3 : ℝ :=
  prob_exactly_3 + prob_all_4

-- The theorem to prove
theorem prob_at_least_3_correct : prob_at_least_3 = 0.9477 :=
  by
  sorry

end prob_at_least_3_correct_l210_210527


namespace simplify_expression_l210_210078

theorem simplify_expression (x : ℝ) : (3 * x + 2) - 2 * (2 * x - 1) = 3 * x + 2 - 4 * x + 2 := 
by sorry

end simplify_expression_l210_210078


namespace find_common_difference_l210_210645

variable {aₙ : ℕ → ℝ}
variable {Sₙ : ℕ → ℝ}

-- Condition that the sum of the first n terms of the arithmetic sequence is S_n
def is_arith_seq (aₙ : ℕ → ℝ) (Sₙ : ℕ → ℝ) : Prop :=
  ∀ n, Sₙ n = (n * (aₙ 0 + (aₙ (n - 1))) / 2)

-- Condition given in the problem
def problem_condition (Sₙ : ℕ → ℝ) : Prop :=
  2 * Sₙ 3 - 3 * Sₙ 2 = 12

theorem find_common_difference (h₀ : is_arith_seq aₙ Sₙ) (h₁ : problem_condition Sₙ) : 
  ∃ d : ℝ, d = 4 := 
sorry

end find_common_difference_l210_210645


namespace sum_of_legs_of_larger_triangle_l210_210124

theorem sum_of_legs_of_larger_triangle (area_small : ℝ) (area_large : ℝ) (hypotenuse_small : ℝ) :
    (area_small = 8 ∧ area_large = 200 ∧ hypotenuse_small = 6) →
    ∃ sum_of_legs : ℝ, sum_of_legs = 41.2 :=
by
  sorry

end sum_of_legs_of_larger_triangle_l210_210124


namespace volume_of_pyramid_l210_210956

noncomputable def volume_of_regular_triangular_pyramid (h R : ℝ) : ℝ :=
  (h ^ 2 * (2 * R - h) * Real.sqrt 3) / 4

theorem volume_of_pyramid (h R : ℝ) : volume_of_regular_triangular_pyramid h R = (h ^ 2 * (2 * R - h) * Real.sqrt 3) / 4 :=
  by sorry

end volume_of_pyramid_l210_210956


namespace distance_traveled_l210_210909

theorem distance_traveled 
    (P_b : ℕ) (P_f : ℕ) (R_b : ℕ) (R_f : ℕ)
    (h1 : P_b = 9)
    (h2 : P_f = 7)
    (h3 : R_f = R_b + 10) 
    (h4 : R_b * P_b = R_f * P_f) :
    R_b * P_b = 315 :=
by
  sorry

end distance_traveled_l210_210909


namespace inequality_unique_solution_l210_210134

theorem inequality_unique_solution (p : ℝ) :
  (∃ x : ℝ, 0 ≤ x^2 + p * x + 5 ∧ x^2 + p * x + 5 ≤ 1) →
  (∃ x : ℝ, x^2 + p * x + 4 = 0) → p = 4 ∨ p = -4 :=
sorry

end inequality_unique_solution_l210_210134


namespace find_v₃_value_l210_210259

def f (x : ℕ) : ℕ := 7 * x^7 + 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

def v₃_expr (x : ℕ) : ℕ := (((7 * x + 6) * x + 5) * x + 4)

theorem find_v₃_value : v₃_expr 3 = 262 := by
  sorry

end find_v₃_value_l210_210259


namespace probability_at_least_one_prize_proof_l210_210567

noncomputable def probability_at_least_one_wins_prize
  (total_tickets : ℕ) (prize_tickets : ℕ) (people : ℕ) :
  ℚ :=
1 - ((@Nat.choose (total_tickets - prize_tickets) people) /
      (@Nat.choose total_tickets people))

theorem probability_at_least_one_prize_proof :
  probability_at_least_one_wins_prize 10 3 5 = 11 / 12 :=
by
  sorry

end probability_at_least_one_prize_proof_l210_210567


namespace deductive_reasoning_option_l210_210788

inductive ReasoningType
| deductive
| inductive
| analogical

-- Definitions based on conditions
def option_A : ReasoningType := ReasoningType.inductive
def option_B : ReasoningType := ReasoningType.deductive
def option_C : ReasoningType := ReasoningType.inductive
def option_D : ReasoningType := ReasoningType.analogical

-- The main theorem to prove
theorem deductive_reasoning_option : option_B = ReasoningType.deductive :=
by sorry

end deductive_reasoning_option_l210_210788


namespace interval_of_decrease_l210_210126

noncomputable def f : ℝ → ℝ := fun x => x^2 - 2 * x

theorem interval_of_decrease : 
  ∃ a b : ℝ, a = -2 ∧ b = 1 ∧ ∀ x1 x2 : ℝ, a ≤ x1 ∧ x1 < x2 ∧ x2 ≤ b → f x1 ≥ f x2 :=
by 
  use -2, 1
  sorry

end interval_of_decrease_l210_210126


namespace no_rational_solutions_l210_210088

theorem no_rational_solutions : 
  ¬ ∃ (x y z : ℚ), 11 = x^5 + 2 * y^5 + 5 * z^5 := 
sorry

end no_rational_solutions_l210_210088


namespace fraction_proof_l210_210774

variables (m n p q : ℚ)

theorem fraction_proof
  (h1 : m / n = 18)
  (h2 : p / n = 9)
  (h3 : p / q = 1 / 15) :
  m / q = 2 / 15 :=
by sorry

end fraction_proof_l210_210774


namespace inequality_proof_l210_210558

theorem inequality_proof (x y : ℝ) : 
  -1 / 2 ≤ (x + y) * (1 - x * y) / ((1 + x ^ 2) * (1 + y ^ 2)) ∧
  (x + y) * (1 - x * y) / ((1 + x ^ 2) * (1 + y ^ 2)) ≤ 1 / 2 :=
sorry

end inequality_proof_l210_210558


namespace average_of_three_quantities_l210_210780

theorem average_of_three_quantities (a b c d e : ℝ) 
    (h1 : (a + b + c + d + e) / 5 = 8)
    (h2 : (d + e) / 2 = 14) :
    (a + b + c) / 3 = 4 := 
sorry

end average_of_three_quantities_l210_210780


namespace impossible_relationships_l210_210097

theorem impossible_relationships (a b : ℝ) (h : (1 / a) = (1 / b)) :
  (¬ (0 < a ∧ a < b)) ∧ (¬ (b < a ∧ a < 0)) :=
by
  sorry

end impossible_relationships_l210_210097


namespace prime_15p_plus_one_l210_210029

open Nat

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_15p_plus_one (p q : ℕ) 
  (hp : is_prime p) 
  (hq : q = 15 * p + 1) 
  (hq_prime : is_prime q) :
  q = 31 :=
sorry

end prime_15p_plus_one_l210_210029


namespace contractor_engagement_days_l210_210555

theorem contractor_engagement_days 
  (days_worked : ℕ) 
  (total_days_absent : ℕ) 
  (work_payment : ℕ → ℤ)
  (absent_fine : ℕ → ℤ)
  (total_payment : ℤ) 
  (total_days : ℕ) 
  (h1 : work_payment days_worked = 25 * days_worked)
  (h2 : absent_fine total_days_absent = 750)
  (h3 : total_payment = (work_payment days_worked) - (absent_fine total_days_absent))
  (h4 : total_payment = 425)
  (h5 : total_days_absent = 10) 
  (h6 : sorry) : -- This assumes the result of x = 20 proving work days 
  total_days = days_worked + total_days_absent := 
  by
    sorry

end contractor_engagement_days_l210_210555


namespace angle_subtraction_correct_polynomial_simplification_correct_l210_210388

noncomputable def angleSubtraction : Prop :=
  let a1 := 34 * 60 + 26 -- Convert 34°26' to total minutes
  let a2 := 25 * 60 + 33 -- Convert 25°33' to total minutes
  let diff := a1 - a2 -- Subtract in minutes
  let degrees := diff / 60 -- Convert back to degrees
  let minutes := diff % 60 -- Remainder in minutes
  degrees = 8 ∧ minutes = 53 -- Expected result in degrees and minutes

noncomputable def polynomialSimplification (m : Int) : Prop :=
  let expr := 5 * m^2 - (m^2 - 6 * m) - 2 * (-m + 3 * m^2)
  expr = -2 * m^2 + 8 * m -- Simplified form

-- Statements needing proof
theorem angle_subtraction_correct : angleSubtraction := by
  sorry

theorem polynomial_simplification_correct (m : Int) : polynomialSimplification m := by
  sorry

end angle_subtraction_correct_polynomial_simplification_correct_l210_210388


namespace symmetric_colors_different_at_8281_div_2_l210_210606

def is_red (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ n = 81 * x + 100 * y

def is_blue (n : ℕ) : Prop :=
  ¬ is_red n

theorem symmetric_colors_different_at_8281_div_2 :
  ∃ n : ℕ, (is_red n ∧ is_blue (8281 - n)) ∨ (is_blue n ∧ is_red (8281 - n)) ∧ 2 * n = 8281 :=
by
  sorry

end symmetric_colors_different_at_8281_div_2_l210_210606


namespace tracy_initial_balloons_l210_210144

theorem tracy_initial_balloons (T : ℕ) : 
  (12 + 8 + (T + 24) / 2 = 35) → T = 6 :=
by
  sorry

end tracy_initial_balloons_l210_210144


namespace sum_of_last_two_digits_l210_210089

theorem sum_of_last_two_digits (a b : ℕ) (ha : a = 7) (hb : b = 13) : (a^15 + b^15) % 100 = 0 := by
  sorry

end sum_of_last_two_digits_l210_210089


namespace find_m_l210_210068

-- Define the set A
def A (m : ℝ) : Set ℝ := {0, m, m^2 - 3 * m + 2}

-- Main theorem statement
theorem find_m (m : ℝ) (h : 2 ∈ A m) : m = 3 := by
  sorry

end find_m_l210_210068


namespace binomial_10_3_eq_120_l210_210671

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l210_210671


namespace cinnamon_swirl_eaters_l210_210856

theorem cinnamon_swirl_eaters (total_pieces : ℝ) (jane_pieces : ℝ) (equal_pieces : total_pieces / jane_pieces = 3 ) : 
  (total_pieces = 12) ∧ (jane_pieces = 4) → total_pieces / jane_pieces = 3 := 
by 
  sorry

end cinnamon_swirl_eaters_l210_210856


namespace jane_age_problem_l210_210933

variables (J M a b c : ℕ)
variables (h1 : J = 2 * (a + b))
variables (h2 : J / 2 = a + b)
variables (h3 : c = 2 * J)
variables (h4 : M > 0)

theorem jane_age_problem (h5 : J - M = 3 * ((J / 2) - 2 * M))
                         (h6 : J - M = c - M)
                         (h7 : c = 2 * J) :
  J / M = 10 :=
sorry

end jane_age_problem_l210_210933


namespace smallest_positive_z_l210_210150

open Real

-- Definitions for the conditions
def sin_zero_condition (x : ℝ) : Prop := sin x = 0
def sin_half_condition (x z : ℝ) : Prop := sin (x + z) = 1 / 2

-- Theorem for the proof objective
theorem smallest_positive_z (x z : ℝ) (hx : sin_zero_condition x) (hz : sin_half_condition x z) : z = π / 6 := 
sorry

end smallest_positive_z_l210_210150


namespace line_intersects_curve_l210_210689

theorem line_intersects_curve (k : ℝ) :
  (∃ x y : ℝ, y + k * x + 2 = 0 ∧ x^2 + y^2 = 2 * x) ↔ k ≤ -3/4 := by
  sorry

end line_intersects_curve_l210_210689


namespace units_digit_2749_987_l210_210449

def mod_units_digit (base : ℕ) (exp : ℕ) : ℕ :=
  (base % 10)^(exp % 2) % 10

theorem units_digit_2749_987 : mod_units_digit 2749 987 = 9 := 
by 
  sorry

end units_digit_2749_987_l210_210449


namespace inequality_count_l210_210412

theorem inequality_count {a b : ℝ} (h : 1/a < 1/b ∧ 1/b < 0) :
  (if (|a| > |b|) then 0 else 1) + 
  (if (a + b > ab) then 1 else 0) +
  (if (a / b + b / a > 2) then 1 else 0) + 
  (if (a^2 / b < 2 * a - b) then 1 else 0) = 2 :=
sorry

end inequality_count_l210_210412


namespace probability_of_blank_l210_210199

-- Definitions based on conditions
def num_prizes : ℕ := 10
def num_blanks : ℕ := 25
def total_outcomes : ℕ := num_prizes + num_blanks

-- Statement of the proof problem
theorem probability_of_blank : (num_blanks / total_outcomes : ℚ) = 5 / 7 :=
by {
  sorry
}

end probability_of_blank_l210_210199


namespace solve_for_a_l210_210533

def i := Complex.I

theorem solve_for_a (a : ℝ) (h : (2 + i) / (1 + a * i) = i) : a = -2 := 
by 
  sorry

end solve_for_a_l210_210533


namespace radius_of_larger_circle_l210_210108

theorem radius_of_larger_circle
  (r r_s : ℝ)
  (h1 : r_s = 2)
  (h2 : π * r^2 = 4 * π * r_s^2) :
  r = 4 :=
by
  sorry

end radius_of_larger_circle_l210_210108


namespace misread_number_is_correct_l210_210204

-- Definitions for the given conditions
def avg_incorrect : ℕ := 19
def incorrect_number : ℕ := 26
def avg_correct : ℕ := 24

-- Statement to prove the actual number that was misread
theorem misread_number_is_correct (x : ℕ) (h : 10 * avg_correct - 10 * avg_incorrect = x - incorrect_number) : x = 76 :=
by {
  sorry
}

end misread_number_is_correct_l210_210204


namespace more_males_l210_210791

theorem more_males {Total_attendees Male_attendees : ℕ} (h1 : Total_attendees = 120) (h2 : Male_attendees = 62) :
  Male_attendees - (Total_attendees - Male_attendees) = 4 :=
by
  sorry

end more_males_l210_210791


namespace first_more_than_200_paperclips_day_l210_210104

-- Definitions based on the conditions:
def paperclips_on_day (k : ℕ) : ℕ :=
  3 * 2^k

-- The theorem stating the solution:
theorem first_more_than_200_paperclips_day :
  ∃ k : ℕ, paperclips_on_day k > 200 ∧ k = 7 :=
by
  use 7
  sorry

end first_more_than_200_paperclips_day_l210_210104


namespace programs_produce_same_output_l210_210426

def sum_program_a : ℕ :=
  let S := (Finset.range 1000).sum (λ i => i + 1)
  S

def sum_program_b : ℕ :=
  let S := (Finset.range 1000).sum (λ i => 1000 - i)
  S

theorem programs_produce_same_output :
  sum_program_a = sum_program_b := by
  sorry

end programs_produce_same_output_l210_210426


namespace find_theta_l210_210452

theorem find_theta (theta : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ 2 * π) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^3 * Real.cos θ - x^2 * (1 - x) + (1 - x)^3 * Real.sin θ > 0) →
  θ > π / 12 ∧ θ < 5 * π / 12 :=
by
  sorry

end find_theta_l210_210452


namespace greatest_int_less_than_50_satisfying_conditions_l210_210861

def satisfies_conditions (n : ℕ) : Prop :=
  n < 50 ∧ Int.gcd n 18 = 6

theorem greatest_int_less_than_50_satisfying_conditions :
  ∃ n : ℕ, satisfies_conditions n ∧ ∀ m : ℕ, satisfies_conditions m → m ≤ n ∧ n = 42 :=
by
  sorry

end greatest_int_less_than_50_satisfying_conditions_l210_210861


namespace boat_speed_in_still_water_l210_210422

-- Definitions for conditions
variables (V_b V_s : ℝ)

-- The conditions provided for the problem
def along_stream := V_b + V_s = 13
def against_stream := V_b - V_s = 5

-- The theorem we want to prove
theorem boat_speed_in_still_water (h1 : along_stream V_b V_s) (h2 : against_stream V_b V_s) : V_b = 9 :=
sorry

end boat_speed_in_still_water_l210_210422


namespace sum_of_numbers_l210_210236

theorem sum_of_numbers (a b c : ℕ) 
  (h1 : a ≤ b ∧ b ≤ c) 
  (h2 : b = 10) 
  (h3 : (a + b + c) / 3 = a + 15) 
  (h4 : (a + b + c) / 3 = c - 20) 
  (h5 : c = 2 * a)
  : a + b + c = 115 := by
  sorry

end sum_of_numbers_l210_210236


namespace line_slope_and_point_l210_210969

noncomputable def line_equation (x : ℝ) (m b : ℝ) : ℝ := m * x + b

theorem line_slope_and_point (m b : ℝ) (x₀ y₀ : ℝ) (h₁ : m = -3) (h₂ : x₀ = 5) (h₃ : y₀ = 2) (h₄ : y₀ = line_equation x₀ m b) :
  m + b = 14 :=
by
  sorry

end line_slope_and_point_l210_210969


namespace sum_of_fractions_l210_210613

theorem sum_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (7 : ℚ) / 9
  a + b = 83 / 72 := 
by
  sorry

end sum_of_fractions_l210_210613


namespace total_cost_correct_l210_210321

noncomputable def total_cost (sandwiches: ℕ) (price_per_sandwich: ℝ) (sodas: ℕ) (price_per_soda: ℝ) (discount: ℝ) (tax: ℝ) : ℝ :=
  let total_sandwich_cost := sandwiches * price_per_sandwich
  let total_soda_cost := sodas * price_per_soda
  let discounted_sandwich_cost := total_sandwich_cost * (1 - discount)
  let total_before_tax := discounted_sandwich_cost + total_soda_cost
  let total_with_tax := total_before_tax * (1 + tax)
  total_with_tax

theorem total_cost_correct : 
  total_cost 2 3.49 4 0.87 0.10 0.05 = 10.25 :=
by
  sorry

end total_cost_correct_l210_210321


namespace percentage_difference_l210_210583

-- Define the numbers
def n : ℕ := 1600
def m : ℕ := 650

-- Define the percentages calculated
def p₁ : ℕ := (20 * n) / 100
def p₂ : ℕ := (20 * m) / 100

-- The theorem to be proved: the difference between the two percentages is 190
theorem percentage_difference : p₁ - p₂ = 190 := by
  sorry

end percentage_difference_l210_210583


namespace find_f_m_eq_neg_one_l210_210267

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^(2 - m)

theorem find_f_m_eq_neg_one (m : ℝ)
  (h1 : ∀ x : ℝ, f x m = - f (-x) m) (h2 : m^2 - m = 3 + m) :
  f m m = -1 :=
by
  sorry

end find_f_m_eq_neg_one_l210_210267


namespace isosceles_triangle_area_l210_210980

theorem isosceles_triangle_area
  (a b : ℝ) -- sides of the triangle
  (inradius : ℝ) (perimeter : ℝ)
  (angle : ℝ) -- angle in degrees
  (h_perimeter : 2 * a + b = perimeter)
  (h_inradius : inradius = 2.5)
  (h_angle : angle = 40)
  (h_perimeter_value : perimeter = 20)
  (h_semiperimeter : (perimeter / 2) = 10) :
  (inradius * (perimeter / 2) = 25) :=
by
  sorry

end isosceles_triangle_area_l210_210980


namespace magician_inequality_l210_210610

theorem magician_inequality (N : ℕ) : 
  (N - 1) * 10^(N - 2) ≥ 10^N → N ≥ 101 :=
by
  sorry

end magician_inequality_l210_210610


namespace arithmetic_geometric_ratio_l210_210541

variables {a : ℕ → ℝ} {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = d

def is_geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

theorem arithmetic_geometric_ratio {a : ℕ → ℝ} {d : ℝ} (h1 : is_arithmetic_sequence a d)
  (h2 : a 9 ≠ a 3) (h3 : is_geometric_sequence (a 1) (a 3) (a 9)):
  (a 2 + a 4 + a 10) / (a 1 + a 3 + a 9) = 16 / 13 :=
sorry

end arithmetic_geometric_ratio_l210_210541


namespace product_evaluation_l210_210432

theorem product_evaluation :
  (6 * 27^12 + 2 * 81^9) / 8000000^2 * (80 * 32^3 * 125^4) / (9^19 - 729^6) = 10 :=
by sorry

end product_evaluation_l210_210432


namespace zoe_earns_per_candy_bar_l210_210447

-- Given conditions
def cost_of_trip : ℝ := 485
def grandma_contribution : ℝ := 250
def candy_bars_to_sell : ℝ := 188

-- Derived condition
def additional_amount_needed : ℝ := cost_of_trip - grandma_contribution

-- Assertion to prove
theorem zoe_earns_per_candy_bar :
  (additional_amount_needed / candy_bars_to_sell) = 1.25 :=
by
  sorry

end zoe_earns_per_candy_bar_l210_210447


namespace min_value_x2_y2_z2_l210_210736

theorem min_value_x2_y2_z2 (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : x^2 + y^2 + z^2 ≥ 3 :=
sorry

end min_value_x2_y2_z2_l210_210736


namespace roots_quadratic_eq_l210_210569

theorem roots_quadratic_eq (a b : ℝ) (h1 : a^2 + 3*a - 4 = 0) (h2 : b^2 + 3*b - 4 = 0) (h3 : a + b = -3) : a^2 + 4*a + b - 3 = -2 :=
by
  sorry

end roots_quadratic_eq_l210_210569


namespace tan_315_deg_l210_210417

theorem tan_315_deg : Real.tan (315 * Real.pi / 180) = -1 := sorry

end tan_315_deg_l210_210417


namespace cyclic_quadrilateral_iff_condition_l210_210406

theorem cyclic_quadrilateral_iff_condition
  (α β γ δ : ℝ)
  (h : α + β + γ + δ = 2 * π) :
  (α * β + α * δ + γ * β + γ * δ = π^2) ↔ (α + γ = π ∧ β + δ = π) :=
by
  sorry

end cyclic_quadrilateral_iff_condition_l210_210406


namespace certain_number_is_2_l210_210287

theorem certain_number_is_2 
    (X : ℕ) 
    (Y : ℕ) 
    (h1 : X = 15) 
    (h2 : 0.40 * (X : ℝ) = 0.80 * 5 + (Y : ℝ)) : 
    Y = 2 := 
  sorry

end certain_number_is_2_l210_210287


namespace initial_deck_card_count_l210_210594

-- Define the initial conditions
def initial_red_probability (r b : ℕ) : Prop := r * 4 = r + b
def added_black_probability (r b : ℕ) : Prop := r * 5 = 4 * r + 6

theorem initial_deck_card_count (r b : ℕ) (h1 : initial_red_probability r b) (h2 : added_black_probability r b) : r + b = 24 := 
by sorry

end initial_deck_card_count_l210_210594


namespace number_of_rows_of_desks_is_8_l210_210563

-- Definitions for the conditions
def first_row_desks : ℕ := 10
def desks_increment : ℕ := 2
def total_desks : ℕ := 136

-- Definition for the sum of an arithmetic series
def arithmetic_series_sum (n a1 d : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

-- The proof problem statement
theorem number_of_rows_of_desks_is_8 :
  ∃ n : ℕ, arithmetic_series_sum n first_row_desks desks_increment = total_desks ∧ n = 8 :=
by
  sorry

end number_of_rows_of_desks_is_8_l210_210563


namespace factor_of_lcm_l210_210844

theorem factor_of_lcm (A B hcf : ℕ) (h_gcd : Nat.gcd A B = hcf) (hcf_eq : hcf = 16) (A_eq : A = 224) :
  ∃ X : ℕ, X = 14 := by
  sorry

end factor_of_lcm_l210_210844


namespace two_digit_number_difference_perfect_square_l210_210473

theorem two_digit_number_difference_perfect_square (N : ℕ) (a b : ℕ)
  (h1 : N = 10 * a + b)
  (h2 : N % 100 = N)
  (h3 : 1 ≤ a ∧ a ≤ 9)
  (h4 : 0 ≤ b ∧ b ≤ 9)
  (h5 : (N - (10 * b + a : ℕ)) = 64) : 
  N = 90 := 
sorry

end two_digit_number_difference_perfect_square_l210_210473


namespace m_leq_neg_one_l210_210546

theorem m_leq_neg_one (m : ℝ) :
    (∀ x : ℝ, 2^(-x) + m > 0 → x ≤ 0) → m ≤ -1 :=
by
  sorry

end m_leq_neg_one_l210_210546


namespace tan_five_pi_over_four_l210_210275

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
by
  sorry

end tan_five_pi_over_four_l210_210275


namespace stratified_sampling_sophomores_selected_l210_210279

theorem stratified_sampling_sophomores_selected 
  (total_freshmen : ℕ) (total_sophomores : ℕ) (total_seniors : ℕ) 
  (freshmen_selected : ℕ) (selection_ratio : ℕ) :
  total_freshmen = 210 →
  total_sophomores = 270 →
  total_seniors = 300 →
  freshmen_selected = 7 →
  selection_ratio = total_freshmen / freshmen_selected →
  selection_ratio = 30 →
  total_sophomores / selection_ratio = 9 :=
by sorry

end stratified_sampling_sophomores_selected_l210_210279


namespace seven_books_cost_l210_210239

-- Given condition: Three identical books cost $45
def three_books_cost (cost_per_book : ℤ) := 3 * cost_per_book = 45

-- Question to prove: The cost of seven identical books is $105
theorem seven_books_cost (cost_per_book : ℤ) (h : three_books_cost cost_per_book) : 7 * cost_per_book = 105 := 
sorry

end seven_books_cost_l210_210239


namespace pay_per_task_l210_210063

def tasks_per_day : ℕ := 100
def days_per_week : ℕ := 6
def weekly_pay : ℕ := 720

theorem pay_per_task :
  (weekly_pay : ℚ) / (tasks_per_day * days_per_week) = 1.20 := 
sorry

end pay_per_task_l210_210063


namespace simplify_expression_l210_210623

theorem simplify_expression (x : ℝ) :
  (3 * x)^3 - (4 * x^2) * (2 * x^3) = 27 * x^3 - 8 * x^5 :=
by
  sorry

end simplify_expression_l210_210623


namespace cat_food_inequality_l210_210160

theorem cat_food_inequality (B S : ℝ) (h1 : B > S) (h2 : B < 2 * S) : 4 * B + 4 * S < 3 * (B + 2 * S) :=
sorry

end cat_food_inequality_l210_210160


namespace math_problem_l210_210155

variable {x p q r : ℝ}

-- Conditions and Theorem
theorem math_problem (h1 : ∀ x, (x ≤ -5 ∨ 20 ≤ x ∧ x ≤ 30) ↔ (0 ≤ (x - p) * (x - q) / (x - r)))
  (h2 : p < q) : p + 2 * q + 3 * r = 65 := 
sorry

end math_problem_l210_210155


namespace solution_set_inequality_l210_210058

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x

axiom mono_increasing (x y : ℝ) (hxy : 0 < x ∧ x < y) : f x < f y

axiom f_2_eq_0 : f 2 = 0

theorem solution_set_inequality :
  { x : ℝ | (x - 1) * f x < 0 } = { x : ℝ | -2 < x ∧ x < 0 } ∪ { x : ℝ | 1 < x ∧ x < 2 } :=
by {
  sorry
}

end solution_set_inequality_l210_210058


namespace ratio_of_triangle_areas_l210_210595

theorem ratio_of_triangle_areas 
  (r s : ℝ) (n : ℝ)
  (h_ratio : 3 * s = r) 
  (h_area : (3 / 2) * n = 1 / 2 * r * ((3 * n * 2) / r)) :
  3 / 3 = n :=
by
  sorry

end ratio_of_triangle_areas_l210_210595


namespace range_of_k_for_ellipse_l210_210048

def represents_ellipse (x y k : ℝ) : Prop :=
  (k^2 - 3 > 0) ∧ 
  (k - 1 > 0) ∧ 
  (k - 1 ≠ k^2 - 3)

theorem range_of_k_for_ellipse (k : ℝ) : 
  represents_ellipse x y k → k ∈ Set.Ioo (-Real.sqrt 3) (-1) ∪ Set.Ioo (-1) 1 :=
by
  sorry

end range_of_k_for_ellipse_l210_210048


namespace locus_of_point_P_l210_210874

-- Definitions and conditions
def circle_M (x y : ℝ) : Prop := (x - 1) ^ 2 + y ^ 2 = 4
def A_point : ℝ × ℝ := (2, 1)
def chord_BC (x y x₀ y₀ : ℝ) : Prop := (x₀ - 1) * x + y₀ * y - x₀ - 3 = 0
def point_P_locus (x₀ y₀ : ℝ) : Prop := ∃ x y, (chord_BC x y x₀ y₀) ∧ x = 2 ∧ y = 1

-- Lean 4 statement to be proved
theorem locus_of_point_P (x₀ y₀ : ℝ) (h : point_P_locus x₀ y₀) : x₀ + y₀ - 5 = 0 :=
  by
  sorry

end locus_of_point_P_l210_210874


namespace faucet_open_duration_l210_210677

-- Initial definitions based on conditions in the problem
def init_water : ℕ := 120
def flow_rate : ℕ := 4
def rem_water : ℕ := 20

-- The equivalent Lean 4 statement to prove
theorem faucet_open_duration (t : ℕ) (H1: init_water - rem_water = flow_rate * t) : t = 25 :=
sorry

end faucet_open_duration_l210_210677


namespace cindy_correct_answer_l210_210187

theorem cindy_correct_answer (x : ℝ) (h : (x - 5) / 7 = 15) :
  (x - 7) / 5 = 20.6 :=
by
  sorry

end cindy_correct_answer_l210_210187


namespace arithmetic_expression_eval_l210_210900

theorem arithmetic_expression_eval : (10 - 9 + 8) * 7 + 6 - 5 * (4 - 3 + 2) - 1 = 53 :=
by
  sorry

end arithmetic_expression_eval_l210_210900


namespace sequence_formula_l210_210357

theorem sequence_formula (S : ℕ → ℤ) (a : ℕ → ℤ) (h : ∀ n : ℕ, n > 0 → S n = 2 * a n - 2^n + 1) : 
  ∀ n : ℕ, n > 0 → a n = n * 2^(n - 1) :=
by
  intro n hn
  sorry

end sequence_formula_l210_210357


namespace derivative_at_one_l210_210319

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem derivative_at_one : (deriv f 1) = 2 * Real.exp 1 := by
  sorry

end derivative_at_one_l210_210319


namespace ones_digit_of_p_is_3_l210_210940

theorem ones_digit_of_p_is_3 (p q r s : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs : Nat.Prime s)
  (h_seq : q = p + 8 ∧ r = p + 16 ∧ s = p + 24) (p_gt_5 : p > 5) : p % 10 = 3 :=
sorry

end ones_digit_of_p_is_3_l210_210940


namespace find_slope_of_chord_l210_210536

noncomputable def slope_of_chord (x1 x2 y1 y2 : ℝ) : ℝ :=
  (y1 - y2) / (x1 - x2)

theorem find_slope_of_chord :
  (∀ (x y : ℝ), x^2 / 36 + y^2 / 9 = 1 → ∃ (x1 x2 y1 y2 : ℝ),
    x1 + x2 = 8 ∧ y1 + y2 = 4 ∧ x = (x1 + x2) / 2 ∧ y = (y1 + y2) / 2 ∧ slope_of_chord x1 x2 y1 y2 = -1 / 2) := sorry

end find_slope_of_chord_l210_210536


namespace time_per_toy_is_3_l210_210568

-- Define the conditions
variable (total_toys : ℕ) (total_hours : ℕ)

-- Define the given condition
def given_condition := (total_toys = 50 ∧ total_hours = 150)

-- Define the statement to be proved
theorem time_per_toy_is_3 (h : given_condition total_toys total_hours) :
  total_hours / total_toys = 3 := by
sorry

end time_per_toy_is_3_l210_210568


namespace always_true_inequality_l210_210374

theorem always_true_inequality (x : ℝ) : x^2 + 1 ≥ 2 * |x| := 
sorry

end always_true_inequality_l210_210374


namespace isosceles_triangle_possible_values_of_x_l210_210468

open Real

-- Define the main statement
theorem isosceles_triangle_possible_values_of_x :
  ∀ x : ℝ, 
  (0 < x ∧ x < 90) ∧ 
  (sin (3*x) = sin (2*x) ∧ 
   sin (9*x) = sin (2*x)) 
  → x = 0 ∨ x = 180/11 ∨ x = 540/11 :=
by
  sorry

end isosceles_triangle_possible_values_of_x_l210_210468


namespace quotient_base4_l210_210324

def base4_to_base10 (n : ℕ) : ℕ :=
  n % 10 + 4 * (n / 10 % 10) + 4^2 * (n / 100 % 10) + 4^3 * (n / 1000)

def base10_to_base4 (n : ℕ) : ℕ :=
  let rec convert (n acc : ℕ) : ℕ :=
    if n < 4 then n * acc
    else convert (n / 4) ((n % 4) * acc * 10 + acc)
  convert n 1

theorem quotient_base4 (a b : ℕ) (h1 : a = 2313) (h2 : b = 13) :
  base10_to_base4 ((base4_to_base10 a) / (base4_to_base10 b)) = 122 :=
by
  sorry

end quotient_base4_l210_210324


namespace largest_lcm_value_l210_210930

-- Define the conditions as local constants 
def lcm_18_3 : ℕ := Nat.lcm 18 3
def lcm_18_6 : ℕ := Nat.lcm 18 6
def lcm_18_9 : ℕ := Nat.lcm 18 9
def lcm_18_15 : ℕ := Nat.lcm 18 15
def lcm_18_21 : ℕ := Nat.lcm 18 21
def lcm_18_27 : ℕ := Nat.lcm 18 27

-- Statement to prove
theorem largest_lcm_value : max lcm_18_3 (max lcm_18_6 (max lcm_18_9 (max lcm_18_15 (max lcm_18_21 lcm_18_27)))) = 126 :=
by
  -- We assume the necessary calculations have been made
  have h1 : lcm_18_3 = 18 := by sorry
  have h2 : lcm_18_6 = 18 := by sorry
  have h3 : lcm_18_9 = 18 := by sorry
  have h4 : lcm_18_15 = 90 := by sorry
  have h5 : lcm_18_21 = 126 := by sorry
  have h6 : lcm_18_27 = 54 := by sorry

  -- Using above results to determine the maximum
  exact (by rw [h1, h2, h3, h4, h5, h6]; exact rfl)

end largest_lcm_value_l210_210930


namespace average_of_last_three_l210_210878

theorem average_of_last_three (A B C D : ℝ) 
  (h1 : (A + B + C) / 3 = 6)
  (h2 : A + D = 11)
  (h3 : D = 4) : 
  (B + C + D) / 3 = 5 :=
by
  sorry

end average_of_last_three_l210_210878


namespace renaldo_distance_l210_210025

theorem renaldo_distance (R : ℕ) (h : R + (1/3 : ℝ) * R + 7 = 27) : R = 15 :=
by sorry

end renaldo_distance_l210_210025


namespace total_sand_volume_l210_210912

noncomputable def cone_diameter : ℝ := 10
noncomputable def cone_radius : ℝ := cone_diameter / 2
noncomputable def cone_height : ℝ := 0.75 * cone_diameter
noncomputable def cylinder_height : ℝ := 0.5 * cone_diameter
noncomputable def total_volume : ℝ := (1 / 3 * Real.pi * cone_radius^2 * cone_height) + (Real.pi * cone_radius^2 * cylinder_height)

theorem total_sand_volume : total_volume = 187.5 * Real.pi := 
by
  sorry

end total_sand_volume_l210_210912


namespace find_number_l210_210081

theorem find_number (x : ℝ) (h : x / 0.05 = 900) : x = 45 :=
by sorry

end find_number_l210_210081


namespace range_of_z_l210_210299

theorem range_of_z (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) :
  12 ≤ x^2 + 4*y^2 ∧ x^2 + 4*y^2 ≤ 20 :=
by
  sorry

end range_of_z_l210_210299


namespace maximum_sum_S6_l210_210735

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (a d : α) (n : ℕ) : α := a + (n - 1) * d

def sum_arithmetic_sequence (a d : α) (n : ℕ) : α :=
  (n : α) / 2 * (2 * a + (n - 1) * d)

theorem maximum_sum_S6 (a d : α)
  (h1 : arithmetic_sequence a d 3 + arithmetic_sequence a d 10 < 0)
  (h2 : sum_arithmetic_sequence a d 11 > 0) :
  ∀ n : ℕ, sum_arithmetic_sequence a d n ≤ sum_arithmetic_sequence a d 6 :=
by sorry

end maximum_sum_S6_l210_210735


namespace calculate_square_of_complex_l210_210879

theorem calculate_square_of_complex (i : ℂ) (h : i^2 = -1) : (1 - i)^2 = -2 * i :=
by
  sorry

end calculate_square_of_complex_l210_210879


namespace fraction_from_tips_l210_210577

-- Define the waiter's salary and the conditions given in the problem
variables (S : ℕ) -- S is natural assuming salary is a non-negative integer
def tips := (4/5 : ℚ) * S
def bonus := 2 * (1/10 : ℚ) * S
def total_income := S + tips S + bonus S

-- The theorem to be proven
theorem fraction_from_tips (S : ℕ) :
  (tips S / total_income S) = (2/5 : ℚ) :=
sorry

end fraction_from_tips_l210_210577


namespace decagon_diagonals_l210_210905

-- Define the number of sides of a decagon
def n : ℕ := 10

-- Define the formula for the number of diagonals in an n-sided polygon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem
theorem decagon_diagonals : num_diagonals n = 35 := by
  sorry

end decagon_diagonals_l210_210905


namespace y_value_l210_210987

-- Given conditions
variables (x y : ℝ)
axiom h1 : x - y = 20
axiom h2 : x + y = 14

-- Prove that y = -3
theorem y_value : y = -3 :=
by { sorry }

end y_value_l210_210987


namespace artist_used_17_ounces_of_paint_l210_210984

def ounces_used_per_large_canvas : ℕ := 3
def ounces_used_per_small_canvas : ℕ := 2
def large_paintings_completed : ℕ := 3
def small_paintings_completed : ℕ := 4

theorem artist_used_17_ounces_of_paint :
  (ounces_used_per_large_canvas * large_paintings_completed + ounces_used_per_small_canvas * small_paintings_completed = 17) :=
by
  sorry

end artist_used_17_ounces_of_paint_l210_210984


namespace crow_eats_quarter_in_twenty_hours_l210_210109

-- Given: The crow eats 1/5 of the nuts in 4 hours
def crow_eating_rate (N : ℕ) : ℕ := N / 5 / 4

-- Prove: It will take 20 hours to eat 1/4 of the nuts
theorem crow_eats_quarter_in_twenty_hours (N : ℕ) (h : ℕ) (h_eq : h = 20) : 
  ((N / 5) / 4 : ℝ) = ((N / 4) / h : ℝ) :=
by
  sorry

end crow_eats_quarter_in_twenty_hours_l210_210109


namespace incorrect_statement_l210_210246

def geom_seq (a r : ℝ) : ℕ → ℝ
| 0       => a
| (n + 1) => r * geom_seq a r n

theorem incorrect_statement
  (a : ℝ) (r : ℝ) (S6 : ℝ)
  (h1 : r = 1 / 2)
  (h2 : S6 = a * (1 - (1 / 2) ^ 6) / (1 - 1 / 2))
  (h3 : S6 = 378) :
  geom_seq a r 2 / S6 ≠ 1 / 8 :=
by 
  have h4 : a = 192 := by sorry
  have h5 : geom_seq 192 (1 / 2) 2 = 192 * (1 / 2) ^ 2 := by sorry
  exact sorry

end incorrect_statement_l210_210246


namespace order_of_magnitudes_l210_210966

theorem order_of_magnitudes (x : ℝ) (hx : 0.8 < x ∧ x < 0.9) : x < x^(x^x) ∧ x^(x^x) < x^x :=
by
  -- Definitions for y and z.
  let y := x^x
  let z := x^(x^x)
  have h1 : x < y := sorry
  have h2 : z < y := sorry
  have h3 : x < z := sorry
  exact ⟨h3, h2⟩

end order_of_magnitudes_l210_210966


namespace horse_drinking_water_l210_210472

-- Definitions and conditions

def initial_horses : ℕ := 3
def added_horses : ℕ := 5
def total_horses : ℕ := initial_horses + added_horses
def bathing_water_per_day : ℕ := 2
def total_water_28_days : ℕ := 1568
def days : ℕ := 28
def daily_water_total : ℕ := total_water_28_days / days

-- The statement looking to prove
theorem horse_drinking_water (D : ℕ) : 
  (total_horses * (D + bathing_water_per_day) = daily_water_total) → 
  D = 5 := 
by
  -- Add proof steps here
  sorry

end horse_drinking_water_l210_210472


namespace minimize_fraction_l210_210937

theorem minimize_fraction (n : ℕ) (h : 0 < n) : 
  (n = 9) → (∀ m : ℕ, 0 < m → (n = m) → (3 * m + 27 / m ≥ 6)) := sorry

end minimize_fraction_l210_210937


namespace quadratic_root_a_l210_210553

theorem quadratic_root_a {a : ℝ} (h : (2 : ℝ) ∈ {x : ℝ | x^2 + 3 * x + a = 0}) : a = -10 :=
by
  sorry

end quadratic_root_a_l210_210553


namespace sequence_x_y_sum_l210_210333

theorem sequence_x_y_sum (r : ℝ) (x y : ℝ)
  (h₁ : r = 1 / 4)
  (h₂ : x = 256 * r)
  (h₃ : y = x * r) :
  x + y = 80 :=
by
  sorry

end sequence_x_y_sum_l210_210333


namespace range_of_x_plus_2y_minus_2z_l210_210887

theorem range_of_x_plus_2y_minus_2z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 4) : -6 ≤ x + 2 * y - 2 * z ∧ x + 2 * y - 2 * z ≤ 6 :=
sorry

end range_of_x_plus_2y_minus_2z_l210_210887


namespace expected_balls_in_original_positions_after_transpositions_l210_210599

theorem expected_balls_in_original_positions_after_transpositions :
  let num_balls := 7
  let first_swap_probability := 2 / 7
  let second_swap_probability := 1 / 7
  let third_swap_probability := 1 / 7
  let original_position_probability := (2 / 343) + (125 / 343)
  let expected_balls := num_balls * original_position_probability
  expected_balls = 889 / 343 := 
sorry

end expected_balls_in_original_positions_after_transpositions_l210_210599


namespace primes_dividing_expression_l210_210332

theorem primes_dividing_expression (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) : 
  6 * p * q ∣ p^3 + q^2 + 38 ↔ (p = 3 ∧ (q = 5 ∨ q = 13)) := 
sorry

end primes_dividing_expression_l210_210332


namespace total_tiles_correct_l210_210701

-- Definitions for room dimensions
def room_length : ℕ := 24
def room_width : ℕ := 18

-- Definitions for tile dimensions
def border_tile_side : ℕ := 2
def inner_tile_side : ℕ := 1

-- Definitions for border and inner area calculations
def border_width : ℕ := 2 * border_tile_side
def inner_length : ℕ := room_length - border_width
def inner_width : ℕ := room_width - border_width

-- Calculation of the number of tiles needed
def border_area : ℕ := (room_length * room_width) - (inner_length * inner_width)
def num_border_tiles : ℕ := border_area / (border_tile_side * border_tile_side)
def inner_area : ℕ := inner_length * inner_width
def num_inner_tiles : ℕ := inner_area / (inner_tile_side * inner_tile_side)

-- Total number of tiles
def total_tiles : ℕ := num_border_tiles + num_inner_tiles

-- The proof statement
theorem total_tiles_correct : total_tiles = 318 := by
  -- Lean code to check the calculations, proof is omitted.
  sorry

end total_tiles_correct_l210_210701


namespace coeff_sum_eq_32_l210_210436

theorem coeff_sum_eq_32 (n : ℕ) (h : (2 : ℕ)^n = 32) : n = 5 :=
sorry

end coeff_sum_eq_32_l210_210436


namespace total_number_of_birds_l210_210830

def bird_cages : Nat := 9
def parrots_per_cage : Nat := 2
def parakeets_per_cage : Nat := 6
def birds_per_cage : Nat := parrots_per_cage + parakeets_per_cage
def total_birds : Nat := bird_cages * birds_per_cage

theorem total_number_of_birds : total_birds = 72 := by
  sorry

end total_number_of_birds_l210_210830


namespace no_possible_stack_of_1997_sum_l210_210437

theorem no_possible_stack_of_1997_sum :
  ¬ ∃ k : ℕ, 6 * k = 3 * 1997 := by
  sorry

end no_possible_stack_of_1997_sum_l210_210437


namespace evaluate_expression_is_15_l210_210799

noncomputable def sumOfFirstNOddNumbers (n : ℕ) : ℕ :=
  n^2

noncomputable def simplifiedExpression : ℕ :=
  sumOfFirstNOddNumbers 1 +
  sumOfFirstNOddNumbers 2 +
  sumOfFirstNOddNumbers 3 +
  sumOfFirstNOddNumbers 4 +
  sumOfFirstNOddNumbers 5

theorem evaluate_expression_is_15 : simplifiedExpression = 15 := by
  sorry

end evaluate_expression_is_15_l210_210799


namespace sin_and_tan_alpha_l210_210186

variable (x : ℝ) (α : ℝ)

-- Conditions
def vertex_is_origin : Prop := true
def initial_side_is_non_negative_half_axis : Prop := true
def terminal_side_passes_through_P : Prop := ∃ (P : ℝ × ℝ), P = (x, -Real.sqrt 2)
def cos_alpha_eq : Prop := x ≠ 0 ∧ Real.cos α = (Real.sqrt 3 / 6) * x

-- Proof Problem Statement
theorem sin_and_tan_alpha (h1 : vertex_is_origin) 
                         (h2 : initial_side_is_non_negative_half_axis) 
                         (h3 : terminal_side_passes_through_P x) 
                         (h4 : cos_alpha_eq x α) 
                         : Real.sin α = -Real.sqrt 6 / 6 ∧ (Real.tan α = Real.sqrt 5 / 5 ∨ Real.tan α = -Real.sqrt 5 / 5) := 
sorry

end sin_and_tan_alpha_l210_210186


namespace min_neg_signs_to_zero_sum_l210_210662

-- Definition of the set of numbers on the clock face
def clock_face_numbers : List ℤ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Sum of the clock face numbers
def sum_clock_face_numbers := clock_face_numbers.sum

-- Given condition that the sum of clock face numbers is 78
axiom sum_clock_face_numbers_is_78 : sum_clock_face_numbers = 78

-- Definition of the function to calculate the minimum number of negative signs needed
def min_neg_signs_needed (numbers : List ℤ) (target : ℤ) : ℕ :=
  sorry -- The implementation is omitted

-- Theorem stating the goal of our problem
theorem min_neg_signs_to_zero_sum : min_neg_signs_needed clock_face_numbers 39 = 4 :=
by
  -- Proof is omitted
  sorry

end min_neg_signs_to_zero_sum_l210_210662


namespace sum_of_interior_angles_of_pentagon_l210_210598

theorem sum_of_interior_angles_of_pentagon :
  let n := 5
  let angleSum := 180 * (n - 2)
  angleSum = 540 :=
by
  sorry

end sum_of_interior_angles_of_pentagon_l210_210598


namespace remainder_when_160_divided_by_k_l210_210759

-- Define k to be a positive integer
def positive_integer (n : ℕ) := n > 0

-- Given conditions in the problem
def divides (a b : ℕ) := ∃ k : ℕ, b = k * a

def problem_condition (k : ℕ) := positive_integer k ∧ (120 % (k * k) = 12)

-- Prove the main statement
theorem remainder_when_160_divided_by_k (k : ℕ) (h : problem_condition k) : 160 % k = 4 := 
sorry  -- Proof here

end remainder_when_160_divided_by_k_l210_210759


namespace least_number_of_groups_l210_210219

theorem least_number_of_groups (total_players : ℕ) (max_per_group : ℕ) (h1 : total_players = 30) (h2 : max_per_group = 12) : ∃ (groups : ℕ), groups = 3 := 
by {
  -- Mathematical conditions and solution to be formalized here
  sorry
}

end least_number_of_groups_l210_210219


namespace poly_ineq_solution_l210_210511

-- Define the inequality conversion
def poly_ineq (x : ℝ) : Prop :=
  x^2 + 2 * x ≤ -1

-- Formalize the set notation for the solution
def solution_set : Set ℝ :=
  { x | x = -1 }

-- State the theorem
theorem poly_ineq_solution : {x : ℝ | poly_ineq x} = solution_set :=
by
  sorry

end poly_ineq_solution_l210_210511


namespace van_helsing_earnings_l210_210128

theorem van_helsing_earnings (V W : ℕ) 
  (h1 : W = 4 * V) 
  (h2 : W = 8) :
  let E_v := 5 * (V / 2)
  let E_w := 10 * 8
  let E_total := E_v + E_w
  E_total = 85 :=
by
  sorry

end van_helsing_earnings_l210_210128


namespace correct_statement_l210_210967

variables {Line Plane : Type}
variable (a b c : Line)
variable (M N : Plane)

/- Definitions for the conditions -/
def lies_on_plane (l : Line) (p : Plane) : Prop := sorry
def intersection (p1 p2 : Plane) : Line := sorry
def parallel (l1 l2 : Line) : Prop := sorry

/- Conditions -/
axiom h1 : lies_on_plane a M
axiom h2 : lies_on_plane b N
axiom h3 : intersection M N = c

/- The correct statement to be proved -/
theorem correct_statement : parallel a b → parallel a c :=
by sorry

end correct_statement_l210_210967


namespace production_average_lemma_l210_210838

theorem production_average_lemma (n : ℕ) (h1 : 50 * n + 60 = 55 * (n + 1)) : n = 1 :=
by
  sorry

end production_average_lemma_l210_210838


namespace bases_for_final_digit_one_l210_210393

noncomputable def numberOfBases : ℕ :=
  (Finset.filter (λ b => ((625 - 1) % b = 0)) (Finset.range 11)).card - 
  (Finset.filter (λ b => b ≤ 2) (Finset.range 11)).card

theorem bases_for_final_digit_one : numberOfBases = 4 :=
by sorry

end bases_for_final_digit_one_l210_210393


namespace inequality_solution_function_min_value_l210_210345

theorem inequality_solution (a : ℕ) (h₁ : abs ((3 / 2 : ℚ) - 2) < a) (h₂ : abs ((1 / 2 : ℚ) - 2) ≥ a) : a = 1 := 
by
  -- proof omitted
  sorry

theorem function_min_value (a : ℕ) (h₁ : abs ((3 / 2 : ℚ) - 2) < a) (h₂ : abs ((1 / 2 : ℚ) - 2) ≥ a)
  (h₃ : a = 1) : ∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → (abs (x + a) + abs (x - 2)) = 3 :=
by
  -- proof omitted
  use 0
  -- proof omitted
  sorry

end inequality_solution_function_min_value_l210_210345


namespace committee_selections_with_at_least_one_prev_served_l210_210499

-- Define the conditions
def total_candidates := 20
def previously_served := 8
def committee_size := 4
def never_served := total_candidates - previously_served

-- The proof problem statement
theorem committee_selections_with_at_least_one_prev_served : 
  (Nat.choose total_candidates committee_size - Nat.choose never_served committee_size) = 4350 :=
by
  sorry

end committee_selections_with_at_least_one_prev_served_l210_210499


namespace problem_k_value_l210_210958

theorem problem_k_value (a b c : ℕ) (h1 : a + b / c = 101) (h2 : a / c + b = 68) :
  (a + b) / c = 13 :=
sorry

end problem_k_value_l210_210958


namespace value_of_x2_minus_y2_l210_210240

theorem value_of_x2_minus_y2 (x y : ℚ) (h1 : x + y = 8 / 15) (h2 : x - y = 2 / 15) : x^2 - y^2 = 16 / 225 :=
by
  sorry

end value_of_x2_minus_y2_l210_210240


namespace george_room_painting_l210_210807

-- Define the number of ways to choose 2 colors out of 9 without considering the restriction
def num_ways_total : ℕ := Nat.choose 9 2

-- Define the restriction that red and pink should not be combined
def num_restricted_ways : ℕ := 1

-- Define the final number of permissible combinations
def num_permissible_combinations : ℕ := num_ways_total - num_restricted_ways

theorem george_room_painting :
  num_permissible_combinations = 35 :=
by
  sorry

end george_room_painting_l210_210807


namespace find_n_l210_210410

noncomputable def problem_statement (m n : ℤ) : Prop :=
  (∀ x : ℝ, x^2 - (m + 2) * x + (m - 2) = 0 → ∃ x1 x2 : ℝ, x1 ≠ 0 ∧ x2 ≠ 0 ∧ x1 * x2 < 0 ∧ x1 > |x2|) ∧
  (∃ r1 r2 : ℚ, r1 * r2 = 2 ∧ m * (r1 * r1 + r2 * r2) = (n - 2) * (r1 + r2) + m^2 - 3)

theorem find_n (m : ℤ) (hm : -2 < m ∧ m < 2) : 
  problem_statement m 5 ∨ problem_statement m (-1) :=
sorry

end find_n_l210_210410


namespace pepper_remaining_l210_210514

/-- Brennan initially had 0.25 grams of pepper. He used 0.16 grams for scrambling eggs. 
His friend added x grams of pepper to another dish. Given y grams are remaining, 
prove that y = 0.09 + x . --/
theorem pepper_remaining (x y : ℝ) (h1 : 0.25 - 0.16 = 0.09) (h2 : y = 0.09 + x) : y = 0.09 + x := 
by
  sorry

end pepper_remaining_l210_210514


namespace days_b_worked_l210_210972

theorem days_b_worked (A_days B_days A_remaining_days : ℝ) (A_work_rate B_work_rate total_work : ℝ)
  (hA_rate : A_work_rate = 1 / A_days)
  (hB_rate : B_work_rate = 1 / B_days)
  (hA_days : A_days = 9)
  (hB_days : B_days = 15)
  (hA_remaining : A_remaining_days = 3)
  (h_total_work : ∀ x : ℝ, (x * B_work_rate + A_remaining_days * A_work_rate = total_work)) :
  ∃ x : ℝ, x = 10 :=
by
  sorry

end days_b_worked_l210_210972


namespace box_height_is_55_cm_l210_210156

noncomputable def height_of_box 
  (ceiling_height_m : ℝ) 
  (light_fixture_below_ceiling_cm : ℝ) 
  (bob_height_m : ℝ) 
  (bob_reach_cm : ℝ) 
  : ℝ :=
  let ceiling_height_cm := ceiling_height_m * 100
  let bob_height_cm := bob_height_m * 100
  let light_fixture_from_floor := ceiling_height_cm - light_fixture_below_ceiling_cm
  let bob_total_reach := bob_height_cm + bob_reach_cm
  light_fixture_from_floor - bob_total_reach

-- Theorem statement
theorem box_height_is_55_cm 
  (ceiling_height_m : ℝ) 
  (light_fixture_below_ceiling_cm : ℝ) 
  (bob_height_m : ℝ) 
  (bob_reach_cm : ℝ) 
  (h : height_of_box ceiling_height_m light_fixture_below_ceiling_cm bob_height_m bob_reach_cm = 55) 
  : height_of_box 3 15 1.8 50 = 55 :=
by
  unfold height_of_box
  sorry

end box_height_is_55_cm_l210_210156


namespace not_possible_to_form_triangle_l210_210174

-- Define the conditions
variables (a : ℝ)

-- State the problem in Lean 4
theorem not_possible_to_form_triangle (h : a > 0) :
  ¬ (a + a > 2 * a ∧ a + 2 * a > a ∧ a + 2 * a > a) :=
by
  sorry

end not_possible_to_form_triangle_l210_210174


namespace distance_origin_to_point_on_parabola_l210_210080

noncomputable def origin : ℝ × ℝ := (0, 0)

noncomputable def parabola_focus (x y : ℝ) : Prop :=
  x^2 = 4 * y ∧ y = 1

noncomputable def point_on_parabola (x y : ℝ) : Prop :=
  x^2 = 4 * y

theorem distance_origin_to_point_on_parabola (x y : ℝ) (hx : x^2 = 4 * y)
 (hf : (0, 1) = (0, 1)) (hPF : (x - 0)^2 + (y - 1)^2 = 25) : (x^2 + y^2 = 32) :=
by
  sorry

end distance_origin_to_point_on_parabola_l210_210080


namespace product_remainder_l210_210399

-- Define the product of the consecutive numbers
def product := 86 * 87 * 88 * 89 * 90 * 91 * 92

-- Lean statement to state the problem
theorem product_remainder :
  product % 7 = 0 :=
by
  sorry

end product_remainder_l210_210399


namespace average_weight_of_rock_l210_210953

-- Define all the conditions
def price_per_pound : ℝ := 4
def total_amount : ℝ := 60
def number_of_rocks : ℕ := 10

-- The statement we need to prove
theorem average_weight_of_rock :
  (total_amount / price_per_pound) / number_of_rocks = 1.5 :=
sorry

end average_weight_of_rock_l210_210953


namespace factorize_expression_l210_210151

theorem factorize_expression (x : ℝ) : 2 * x - x^2 = x * (2 - x) := sorry

end factorize_expression_l210_210151


namespace find_first_number_l210_210899

variable (a : ℕ → ℤ)

axiom recurrence_rel : ∀ (n : ℕ), n ≥ 4 → a n = a (n - 1) + a (n - 2) + a (n - 3)
axiom a8_val : a 8 = 29
axiom a9_val : a 9 = 56
axiom a10_val : a 10 = 108

theorem find_first_number : a 1 = 32 :=
sorry

end find_first_number_l210_210899


namespace identity_proof_l210_210423

theorem identity_proof (a b : ℝ) : a^4 + b^4 + (a + b)^4 = 2 * (a^2 + a * b + b^2)^2 := 
sorry

end identity_proof_l210_210423


namespace second_parentheses_expression_eq_zero_l210_210409

def custom_op (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

theorem second_parentheses_expression_eq_zero :
  custom_op (Real.sqrt 6) (Real.sqrt 6) = 0 := by
  sorry

end second_parentheses_expression_eq_zero_l210_210409


namespace compositeShapeSum_is_42_l210_210488

-- Define the pentagonal prism's properties
structure PentagonalPrism where
  faces : ℕ := 7
  edges : ℕ := 15
  vertices : ℕ := 10

-- Define the pyramid addition effect
structure PyramidAddition where
  additional_faces : ℕ := 5
  additional_edges : ℕ := 5
  additional_vertices : ℕ := 1
  covered_faces : ℕ := 1

-- Definition of composite shape properties
def compositeShapeSum (prism : PentagonalPrism) (pyramid : PyramidAddition) : ℕ :=
  (prism.faces - pyramid.covered_faces + pyramid.additional_faces) +
  (prism.edges + pyramid.additional_edges) +
  (prism.vertices + pyramid.additional_vertices)

-- The theorem to be proved: that the total sum is 42
theorem compositeShapeSum_is_42 : compositeShapeSum ⟨7, 15, 10⟩ ⟨5, 5, 1, 1⟩ = 42 := by
  sorry

end compositeShapeSum_is_42_l210_210488


namespace derivative_of_odd_function_is_even_l210_210066

theorem derivative_of_odd_function_is_even (f : ℝ → ℝ) (hf : ∀ x, f (-x) = -f x) :
  ∀ x, (deriv f) (-x) = (deriv f) x :=
by
  sorry

end derivative_of_odd_function_is_even_l210_210066


namespace remaining_volume_of_cube_l210_210706

theorem remaining_volume_of_cube (s : ℝ) (r : ℝ) (h : ℝ) (π : ℝ) 
    (cube_volume : s = 5) 
    (cylinder_radius : r = 1.5) 
    (cylinder_height : h = 5) :
    s^3 - π * r^2 * h = 125 - 11.25 * π := by
  sorry

end remaining_volume_of_cube_l210_210706


namespace average_of_two_integers_l210_210625

theorem average_of_two_integers {A B C D : ℕ} (h1 : A + B + C + D = 200) (h2 : C ≤ 130) : (A + B) / 2 = 35 :=
by
  sorry

end average_of_two_integers_l210_210625


namespace tim_fewer_apples_l210_210753

theorem tim_fewer_apples (martha_apples : ℕ) (harry_apples : ℕ) (tim_apples : ℕ) (H1 : martha_apples = 68) (H2 : harry_apples = 19) (H3 : harry_apples * 2 = tim_apples) : martha_apples - tim_apples = 30 :=
by
  sorry

end tim_fewer_apples_l210_210753


namespace simplify_divide_expression_l210_210331

noncomputable def a : ℝ := Real.sqrt 2 + 1

theorem simplify_divide_expression : 
  (1 - (a / (a + 1))) / ((a^2 - 1) / (a^2 + 2 * a + 1)) = Real.sqrt 2 / 2 :=
by
  sorry

end simplify_divide_expression_l210_210331


namespace smallest_range_possible_l210_210039

-- Definition of the problem conditions
def seven_observations (x1 x2 x3 x4 x5 x6 x7 : ℝ) :=
  (x1 + x2 + x3 + x4 + x5 + x6 + x7) / 7 = 9 ∧
  x4 = 10

noncomputable def smallest_range : ℝ :=
  5

-- Lean statement asserting the proof problem
theorem smallest_range_possible (x1 x2 x3 x4 x5 x6 x7 : ℝ) (h : seven_observations x1 x2 x3 x4 x5 x6 x7) :
  ∃ x1' x2' x3' x4' x5' x6' x7', seven_observations x1' x2' x3' x4' x5' x6' x7' ∧ (x7' - x1') = smallest_range :=
sorry

end smallest_range_possible_l210_210039


namespace notebooks_have_50_pages_l210_210743

theorem notebooks_have_50_pages (notebooks : ℕ) (total_dollars : ℕ) (page_cost_cents : ℕ) 
  (total_cents : ℕ) (total_pages : ℕ) (pages_per_notebook : ℕ)
  (h1 : notebooks = 2) 
  (h2 : total_dollars = 5) 
  (h3 : page_cost_cents = 5) 
  (h4 : total_cents = total_dollars * 100) 
  (h5 : total_pages = total_cents / page_cost_cents) 
  (h6 : pages_per_notebook = total_pages / notebooks) 
  : pages_per_notebook = 50 :=
by
  sorry

end notebooks_have_50_pages_l210_210743


namespace sum_of_ages_l210_210957

theorem sum_of_ages (a b c : ℕ) (h1 : a = b) (h2 : a * b * c = 72) : a + b + c = 14 :=
sorry

end sum_of_ages_l210_210957


namespace least_number_of_plates_needed_l210_210263

theorem least_number_of_plates_needed
  (cubes : ℕ)
  (cube_dim : ℕ)
  (temp_limit : ℕ)
  (plates_exist : ∀ (n : ℕ), n > temp_limit → ∃ (p : ℕ), p = 21) :
  cubes = 512 ∧ cube_dim = 8 → temp_limit > 0 → 21 = 7 + 7 + 7 :=
by {
  sorry
}

end least_number_of_plates_needed_l210_210263


namespace parabola_properties_l210_210310

theorem parabola_properties (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c < 0) :
  (∀ x, a * x^2 + b * x + c >= a * (x^2)) ∧
  (c < 0) ∧ 
  (-b / (2 * a) < 0) :=
by
  sorry

end parabola_properties_l210_210310


namespace intersection_points_C1_C2_l210_210891

theorem intersection_points_C1_C2 :
  (∀ t : ℝ, ∃ (ρ θ : ℝ), 
    (ρ^2 - 10 * ρ * Real.cos θ - 8 * ρ * Real.sin θ + 41 = 0) ∧ 
    (ρ = 2 * Real.cos θ) → 
    ((ρ = 2 ∧ θ = 0) ∨ (ρ = Real.sqrt 2 ∧ θ = Real.pi / 4))) :=
sorry

end intersection_points_C1_C2_l210_210891


namespace park_area_l210_210817

theorem park_area (L B : ℝ) (h1 : L = B / 2) (h2 : 6 * 1000 / 60 * 6 = 2 * (L + B)) : L * B = 20000 :=
by
  -- proof will go here
  sorry

end park_area_l210_210817


namespace original_number_l210_210289

theorem original_number (N : ℤ) : (∃ k : ℤ, N - 7 = 12 * k) → N = 19 :=
by
  intros h
  sorry

end original_number_l210_210289


namespace quadratic_inequality_range_l210_210826

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, (1/2) * a * x^2 - a * x + 2 > 0) ↔ a ∈ Set.Ico 0 4 := 
by
  sorry

end quadratic_inequality_range_l210_210826


namespace lily_cups_in_order_l210_210293

theorem lily_cups_in_order :
  ∀ (rose_rate lily_rate : ℕ) (order_rose_cups total_payment hourly_wage : ℕ),
    rose_rate = 6 →
    lily_rate = 7 →
    order_rose_cups = 6 →
    total_payment = 90 →
    hourly_wage = 30 →
    ∃ lily_cups: ℕ, lily_cups = 14 :=
by
  intros
  sorry

end lily_cups_in_order_l210_210293


namespace range_of_a_l210_210812

def condition1 (a : ℝ) : Prop := (2 - a) ^ 2 < 1
def condition2 (a : ℝ) : Prop := (3 - a) ^ 2 ≥ 1

theorem range_of_a (a : ℝ) (h1 : condition1 a) (h2 : condition2 a) :
  1 < a ∧ a ≤ 2 := 
sorry

end range_of_a_l210_210812


namespace seq_a_n_value_l210_210273

theorem seq_a_n_value (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ n, S n = n^2)
  (h2 : ∀ n ≥ 2, a n = S n - S (n-1)) :
  a 10 = 19 :=
sorry

end seq_a_n_value_l210_210273


namespace find_f_l210_210052

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x / (a * x + b)

theorem find_f (a b : ℝ) (h₀ : a ≠ 0) (h₁ : f 2 a b = 1) (h₂ : ∃! x, f x a b = x) :
  f x (1/2) 1 = 2 * x / (x + 2) :=
by
  sorry

end find_f_l210_210052


namespace daisies_bought_l210_210977

-- Definitions from the given conditions
def cost_per_flower : ℕ := 6
def num_roses : ℕ := 7
def total_spent : ℕ := 60

-- Proving the number of daisies Maria bought
theorem daisies_bought : ∃ (D : ℕ), D = 3 ∧ total_spent = num_roses * cost_per_flower + D * cost_per_flower :=
by
  sorry

end daisies_bought_l210_210977


namespace cleared_land_with_corn_is_630_acres_l210_210700

-- Definitions based on given conditions
def total_land : ℝ := 6999.999999999999
def cleared_fraction : ℝ := 0.90
def potato_fraction : ℝ := 0.20
def tomato_fraction : ℝ := 0.70

-- Calculate the cleared land
def cleared_land : ℝ := cleared_fraction * total_land

-- Calculate the land used for potato and tomato
def potato_land : ℝ := potato_fraction * cleared_land
def tomato_land : ℝ := tomato_fraction * cleared_land

-- Define the land planted with corn
def corn_land : ℝ := cleared_land - (potato_land + tomato_land)

-- The theorem to be proved
theorem cleared_land_with_corn_is_630_acres : corn_land = 630 := by
  sorry

end cleared_land_with_corn_is_630_acres_l210_210700


namespace area_of_field_l210_210142

noncomputable def area_square_field (speed_kmh : ℕ) (time_min : ℕ) : ℝ :=
  let speed_m_per_min := (speed_kmh * 1000) / 60
  let distance := speed_m_per_min * time_min
  let side_length := distance / Real.sqrt 2
  side_length ^ 2

-- Given conditions
theorem area_of_field : area_square_field 4 3 = 20000 := by
  sorry

end area_of_field_l210_210142


namespace student_tickets_sold_l210_210309

theorem student_tickets_sold (S NS : ℕ) (h1 : S + NS = 150) (h2 : 5 * S + 8 * NS = 930) : S = 90 :=
by
  sorry

end student_tickets_sold_l210_210309


namespace solve_system_of_equations_l210_210244

def system_of_equations(x y z: ℝ): Prop :=
  (x * y + 2 * x * z + 3 * y * z = -6) ∧
  (x^2 * y^2 + 4 * x^2 * z^2 - 9 * y^2 * z^2 = 36) ∧
  (x^3 * y^3 + 8 * x^3 * z^3 + 27 * y^3 * z^3 = -216)

theorem solve_system_of_equations :
  ∀ (x y z: ℝ), system_of_equations x y z ↔
  (y = 0 ∧ x * z = -3) ∨
  (z = 0 ∧ x * y = -6) ∨
  (x = 3 ∧ y = -2 ∨ z = -1) ∨
  (x = -3 ∧ y = 2 ∨ z = 1) :=
by
  sorry

end solve_system_of_equations_l210_210244


namespace arithmetic_progression_rth_term_l210_210408

variable (n r : ℕ)

def S (n : ℕ) : ℕ := 2 * n + 3 * n^2

theorem arithmetic_progression_rth_term : (S r) - (S (r - 1)) = 6 * r - 1 :=
by
  sorry

end arithmetic_progression_rth_term_l210_210408


namespace find_bloom_day_l210_210988

def days := {d : Fin 7 // 1 ≤ d.val ∧ d.val ≤ 7}

def sunflowers_bloom (d : days) : Prop :=
¬ (d.val = 2 ∨ d.val = 4 ∨ d.val = 7)

def lilies_bloom (d : days) : Prop :=
¬ (d.val = 4 ∨ d.val = 6)

def magnolias_bloom (d : days) : Prop :=
¬ (d.val = 7)

def all_bloom_together (d : days) : Prop :=
sunflowers_bloom d ∧ lilies_bloom d ∧ magnolias_bloom d

def blooms_simultaneously (d : days) : Prop :=
∀ d1 d2 d3 : days, (d1 = d ∧ d2 = d ∧ d3 = d) →
(all_bloom_together d1 ∧ all_bloom_together d2 ∧ all_bloom_together d3)

theorem find_bloom_day :
  ∃ d : days, blooms_simultaneously d :=
sorry

end find_bloom_day_l210_210988


namespace unattainable_y_ne_l210_210261

theorem unattainable_y_ne : ∀ x : ℝ, x ≠ -5/4 → y = (2 - 3 * x) / (4 * x + 5) → y ≠ -3/4 :=
by
  sorry

end unattainable_y_ne_l210_210261


namespace proposition_range_l210_210917

theorem proposition_range (m : ℝ) : 
  (m < 1/2 ∧ m ≠ 1/3) ∨ (m = 3) ↔ m ∈ Set.Iio (1/3:ℝ) ∪ Set.Ioo (1/3:ℝ) (1/2:ℝ) ∪ {3} :=
sorry

end proposition_range_l210_210917


namespace total_number_of_questions_l210_210929

theorem total_number_of_questions (N : ℕ)
  (hp : 0.8 * N = (4 / 5 : ℝ) * N)
  (hv : 35 = 35)
  (hb : (N / 2 : ℕ) = 1 * (N.div 2))
  (ha : N - 7 = N - 7) : N = 60 :=
by
  sorry

end total_number_of_questions_l210_210929


namespace find_C_l210_210629

variable (A B C : ℚ)

def condition1 := A + B + C = 350
def condition2 := A + C = 200
def condition3 := B + C = 350

theorem find_C : condition1 A B C → condition2 A C → condition3 B C → C = 200 :=
by
  sorry

end find_C_l210_210629


namespace range_of_function_l210_210241

open Real

noncomputable def f (x : ℝ) : ℝ := -cos x ^ 2 - 4 * sin x + 6

theorem range_of_function : 
  ∀ y, (∃ x, y = f x) ↔ 2 ≤ y ∧ y ≤ 10 :=
by
  sorry

end range_of_function_l210_210241


namespace expand_expression_l210_210935

theorem expand_expression (x y : ℝ) : 5 * (4 * x^3 - 3 * x * y + 7) = 20 * x^3 - 15 * x * y + 35 := 
sorry

end expand_expression_l210_210935


namespace intersection_sets_l210_210697

theorem intersection_sets (M N : Set ℝ) :
  (M = {x | x * (x - 3) < 0}) → (N = {x | |x| < 2}) → (M ∩ N = {x | 0 < x ∧ x < 2}) :=
by
  intro hM hN
  rw [hM, hN]
  sorry

end intersection_sets_l210_210697


namespace no_solution_exists_l210_210352

theorem no_solution_exists (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : ¬(2 / a + 2 / b = 1 / (a + b)) :=
sorry

end no_solution_exists_l210_210352


namespace find_y_l210_210898

theorem find_y (x y : ℝ) (h1 : x + y = 15) (h2 : x - y = 5) : y = 5 :=
by
  sorry

end find_y_l210_210898


namespace sandwich_cost_l210_210881

-- Defining the cost of each sandwich and the known conditions
variable (S : ℕ) -- Cost of each sandwich in dollars

-- Conditions as hypotheses
def buys_three_sandwiches (S : ℕ) : ℕ := 3 * S
def buys_two_drinks (drink_cost : ℕ) : ℕ := 2 * drink_cost
def total_cost (sandwich_cost drink_cost total_amount : ℕ) : Prop := buys_three_sandwiches sandwich_cost + buys_two_drinks drink_cost = total_amount

-- Given conditions in the problem
def given_conditions : Prop :=
  (buys_two_drinks 4 = 8) ∧ -- Each drink costs $4
  (total_cost S 4 26)       -- Total spending is $26

-- Theorem to prove the cost of each sandwich
theorem sandwich_cost : given_conditions S → S = 6 :=
by sorry

end sandwich_cost_l210_210881


namespace terry_daily_income_l210_210815

theorem terry_daily_income (T : ℕ) (h1 : ∀ j : ℕ, j = 30) (h2 : 7 * 30 = 210) (h3 : 7 * T - 210 = 42) : T = 36 := 
by
  sorry

end terry_daily_income_l210_210815


namespace geese_percentage_among_non_swan_birds_l210_210944

theorem geese_percentage_among_non_swan_birds :
  let total_birds := 100
  let geese := 0.40 * total_birds
  let swans := 0.20 * total_birds
  let non_swans := total_birds - swans
  let geese_percentage_among_non_swans := (geese / non_swans) * 100
  geese_percentage_among_non_swans = 50 := 
by sorry

end geese_percentage_among_non_swan_birds_l210_210944


namespace cos_difference_of_angles_l210_210657

theorem cos_difference_of_angles (α β : ℝ) 
    (h1 : Real.cos (α + β) = 1 / 5) 
    (h2 : Real.tan α * Real.tan β = 1 / 2) : 
    Real.cos (α - β) = 3 / 5 := 
sorry

end cos_difference_of_angles_l210_210657


namespace perpendicular_lines_k_value_l210_210961

theorem perpendicular_lines_k_value (k : ℚ) : (∀ x y : ℚ, y = 3 * x + 7) ∧ (∀ x y : ℚ, 4 * y + k * x = 4) → k = 4 / 3 :=
by
  sorry

end perpendicular_lines_k_value_l210_210961


namespace minimum_ab_l210_210821

variable (a b : ℝ)

def is_collinear (a b : ℝ) : Prop :=
  (0 - b) * (-2 - 0) = (-2 - b) * (a - 0)

theorem minimum_ab (h1 : a * b > 0) (h2 : is_collinear a b) : a * b = 16 := by
  sorry

end minimum_ab_l210_210821


namespace find_sample_size_l210_210524

theorem find_sample_size (f r : ℝ) (h1 : f = 20) (h2 : r = 0.125) (h3 : r = f / n) : n = 160 := 
by {
  sorry
}

end find_sample_size_l210_210524


namespace minimum_value_2x_plus_y_l210_210075

theorem minimum_value_2x_plus_y (x y : ℝ) 
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : (1 / x) + (2 / (y + 1)) = 2) : 2 * x + y ≥ 3 := 
by
  sorry

end minimum_value_2x_plus_y_l210_210075


namespace Randy_blocks_used_l210_210021

theorem Randy_blocks_used (blocks_tower : ℕ) (blocks_house : ℕ) (total_blocks_used : ℕ) :
  blocks_tower = 27 → blocks_house = 53 → total_blocks_used = (blocks_tower + blocks_house) → total_blocks_used = 80 :=
by
  sorry

end Randy_blocks_used_l210_210021


namespace length_of_train_correct_l210_210414

noncomputable def length_of_train (time_pass_man : ℝ) (train_speed_kmh : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let relative_speed_kmh := train_speed_kmh - man_speed_kmh
  let relative_speed_ms := (relative_speed_kmh * 1000) / 3600
  relative_speed_ms * time_pass_man

theorem length_of_train_correct :
  length_of_train 29.997600191984642 60 6 = 449.96400287976963 := by
  sorry

end length_of_train_correct_l210_210414


namespace age_of_student_who_left_l210_210892

/-- 
The average student age of a class with 30 students is 10 years.
After one student leaves and the teacher (who is 41 years old) is included,
the new average age is 11 years. Prove that the student who left is 11 years old.
-/
theorem age_of_student_who_left (x : ℕ) (h1 : (30 * 10) = 300)
    (h2 : (300 - x + 41) / 30 = 11) : x = 11 :=
by 
  -- This is where the proof would go
  sorry

end age_of_student_who_left_l210_210892


namespace positive_difference_of_complementary_angles_in_ratio_five_to_four_l210_210327

theorem positive_difference_of_complementary_angles_in_ratio_five_to_four
  (a b : ℝ)
  (h1 : a / b = 5 / 4)
  (h2 : a + b = 90) :
  |a - b| = 10 :=
sorry

end positive_difference_of_complementary_angles_in_ratio_five_to_four_l210_210327


namespace no_real_solution_condition_l210_210026

def no_real_solution (k : ℝ) : Prop :=
  let discriminant := 25 + 4 * k
  discriminant < 0

theorem no_real_solution_condition (k : ℝ) : no_real_solution k ↔ k < -25 / 4 := 
sorry

end no_real_solution_condition_l210_210026


namespace ab_range_l210_210787

theorem ab_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = a + b + 8) : a * b ≥ 16 :=
sorry

end ab_range_l210_210787


namespace book_cost_l210_210049

variable {b m : ℝ}

theorem book_cost (h1 : b + m = 2.10) (h2 : b = m + 2) : b = 2.05 :=
by
  sorry

end book_cost_l210_210049


namespace power_greater_than_linear_l210_210148

theorem power_greater_than_linear (n : ℕ) (h : n ≥ 3) : 2^n > 2 * n + 1 := 
by {
  sorry
}

end power_greater_than_linear_l210_210148


namespace probability_same_color_l210_210427

-- Define the total number of plates
def totalPlates : ℕ := 6 + 5 + 3

-- Define the number of red plates, blue plates, and green plates
def redPlates : ℕ := 6
def bluePlates : ℕ := 5
def greenPlates : ℕ := 3

-- Define the total number of ways to choose 3 plates from 14
def totalWaysChoose3 : ℕ := Nat.choose totalPlates 3

-- Define the number of ways to choose 3 red plates, 3 blue plates, and 3 green plates
def redWaysChoose3 : ℕ := Nat.choose redPlates 3
def blueWaysChoose3 : ℕ := Nat.choose bluePlates 3
def greenWaysChoose3 : ℕ := Nat.choose greenPlates 3

-- Calculate the total number of favorable combinations (all plates being the same color)
def favorableCombinations : ℕ := redWaysChoose3 + blueWaysChoose3 + greenWaysChoose3

-- State the theorem: the probability that all plates are of the same color.
theorem probability_same_color : (favorableCombinations : ℚ) / (totalWaysChoose3 : ℚ) = 31 / 364 := by sorry

end probability_same_color_l210_210427


namespace total_spent_two_years_l210_210728

def home_game_price : ℕ := 60
def away_game_price : ℕ := 75
def home_playoff_price : ℕ := 120
def away_playoff_price : ℕ := 100

def this_year_home_games : ℕ := 2
def this_year_away_games : ℕ := 2
def this_year_home_playoff_games : ℕ := 1
def this_year_away_playoff_games : ℕ := 0

def last_year_home_games : ℕ := 6
def last_year_away_games : ℕ := 3
def last_year_home_playoff_games : ℕ := 1
def last_year_away_playoff_games : ℕ := 1

def calculate_total_cost : ℕ :=
  let this_year_cost := this_year_home_games * home_game_price + this_year_away_games * away_game_price + this_year_home_playoff_games * home_playoff_price + this_year_away_playoff_games * away_playoff_price
  let last_year_cost := last_year_home_games * home_game_price + last_year_away_games * away_game_price + last_year_home_playoff_games * home_playoff_price + last_year_away_playoff_games * away_playoff_price
  this_year_cost + last_year_cost

theorem total_spent_two_years : calculate_total_cost = 1195 :=
by
  sorry

end total_spent_two_years_l210_210728


namespace stick_horisontal_fall_position_l210_210989

-- Definitions based on the conditions
def stick_length : ℝ := 120 -- length of the stick in cm
def projection_distance : ℝ := 70 -- distance between projections of the ends of the stick on the floor

-- The main theorem to prove
theorem stick_horisontal_fall_position :
  ∀ (L d : ℝ), L = stick_length ∧ d = projection_distance → 
  ∃ x : ℝ, x = 25 :=
by
  intros L d h
  have h1 : L = stick_length := h.1
  have h2 : d = projection_distance := h.2
  -- The detailed proof steps will be here
  sorry

end stick_horisontal_fall_position_l210_210989


namespace total_rankings_l210_210637

-- Defines the set of players
inductive Player
| P : Player
| Q : Player
| R : Player
| S : Player

-- Defines a function to count the total number of ranking sequences
def total_possible_rankings (p : Player → Player → Prop) : Nat := 
  4 * 2 * 2

-- Problem statement
theorem total_rankings : ∃ t : Player → Player → Prop, total_possible_rankings t = 16 :=
by
  sorry

end total_rankings_l210_210637


namespace maximum_value_of_linear_expression_l210_210674

theorem maximum_value_of_linear_expression (m n : ℕ) (h_sum : (m*(m + 1) + n^2 = 1987)) : 3 * m + 4 * n ≤ 221 :=
sorry

end maximum_value_of_linear_expression_l210_210674


namespace twigs_per_branch_l210_210614

/-- Definitions -/
def total_branches : ℕ := 30
def total_leaves : ℕ := 12690
def percentage_4_leaves : ℝ := 0.30
def leaves_per_twig_4_leaves : ℕ := 4
def percentage_5_leaves : ℝ := 0.70
def leaves_per_twig_5_leaves : ℕ := 5

/-- Given conditions translated to Lean -/
def hypothesis (T : ℕ) : Prop :=
  (percentage_4_leaves * T * leaves_per_twig_4_leaves) +
  (percentage_5_leaves * T * leaves_per_twig_5_leaves) = total_leaves

/-- The main theorem to prove -/
theorem twigs_per_branch
  (T : ℕ)
  (h : hypothesis T) :
  (T / total_branches) = 90 :=
sorry

end twigs_per_branch_l210_210614


namespace bus_capacity_l210_210760

def seats_available_on_left := 15
def seats_available_diff := 3
def people_per_seat := 3
def back_seat_capacity := 7

theorem bus_capacity : 
  (seats_available_on_left * people_per_seat) + 
  ((seats_available_on_left - seats_available_diff) * people_per_seat) + 
  back_seat_capacity = 88 := 
by 
  sorry

end bus_capacity_l210_210760


namespace sum_of_coefficients_l210_210114

theorem sum_of_coefficients (a : ℤ) (x : ℤ) (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) :
  (a + x) * (1 + x) ^ 4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_1 + a_3 + a_5 = 32 →
  a = 3 :=
by sorry

end sum_of_coefficients_l210_210114


namespace rachel_homework_difference_l210_210107

def total_difference (r m h s : ℕ) : ℕ :=
  (r - m) + (s - h)

theorem rachel_homework_difference :
    ∀ (r m h s : ℕ), r = 7 → m = 5 → h = 3 → s = 6 → total_difference r m h s = 5 :=
by
  intros r m h s hr hm hh hs
  rw [hr, hm, hh, hs]
  rfl

end rachel_homework_difference_l210_210107


namespace find_f5_l210_210676

noncomputable def f : ℝ → ℝ := sorry

axiom additivity : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f4_value : f 4 = 5

theorem find_f5 : f 5 = 25 / 4 :=
by
  -- Proof goes here
  sorry

end find_f5_l210_210676


namespace domain_all_real_numbers_l210_210686

theorem domain_all_real_numbers (k : ℝ) :
  (∀ x : ℝ, -7 * x^2 - 4 * x + k ≠ 0) ↔ k < -4 / 7 := by
  sorry

end domain_all_real_numbers_l210_210686


namespace engineer_progress_l210_210656

theorem engineer_progress (x : ℕ) : 
  ∀ (road_length_in_km : ℝ) 
    (total_days : ℕ) 
    (initial_men : ℕ) 
    (completed_work_in_km : ℝ) 
    (additional_men : ℕ) 
    (new_total_men : ℕ) 
    (remaining_work_in_km : ℝ) 
    (remaining_days : ℕ),
    road_length_in_km = 10 → 
    total_days = 300 → 
    initial_men = 30 → 
    completed_work_in_km = 2 → 
    additional_men = 30 → 
    new_total_men = 60 → 
    remaining_work_in_km = 8 → 
    remaining_days = total_days - x →
  (4 * (total_days - x) = 8 * x) →
  x = 100 :=
by
  intros road_length_in_km total_days initial_men completed_work_in_km additional_men new_total_men remaining_work_in_km remaining_days
  intros h1 h2 h3 h4 h5 h6 h7 h8 h_eqn
  -- Proof
  sorry

end engineer_progress_l210_210656


namespace chocolate_factory_production_l210_210054

theorem chocolate_factory_production
  (candies_per_hour : ℕ)
  (total_candies : ℕ)
  (days : ℕ)
  (total_hours : ℕ := total_candies / candies_per_hour)
  (hours_per_day : ℕ := total_hours / days)
  (h1 : candies_per_hour = 50)
  (h2 : total_candies = 4000)
  (h3 : days = 8) :
  hours_per_day = 10 := by
  sorry

end chocolate_factory_production_l210_210054


namespace prime_square_condition_no_prime_cube_condition_l210_210228

-- Part (a): Prove p = 3 given 8*p + 1 = n^2 and p is a prime
theorem prime_square_condition (p : ℕ) (n : ℕ) (h_prime : Prime p) 
  (h_eq : 8 * p + 1 = n ^ 2) : 
  p = 3 :=
sorry

-- Part (b): Prove no p exists given 8*p + 1 = n^3 and p is a prime
theorem no_prime_cube_condition (p : ℕ) (n : ℕ) (h_prime : Prime p) 
  (h_eq : 8 * p + 1 = n ^ 3) : 
  False :=
sorry

end prime_square_condition_no_prime_cube_condition_l210_210228


namespace prove_equation_1_prove_equation_2_l210_210860

theorem prove_equation_1 : 
  ∀ x, (x - 3) / (x - 2) - 1 = 3 / x ↔ x = 3 / 2 :=
by
  sorry

theorem prove_equation_2 :
  ¬∃ x, (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1 :=
by
  sorry

end prove_equation_1_prove_equation_2_l210_210860


namespace nialls_children_ages_l210_210367

theorem nialls_children_ages : ∃ (a b c d : ℕ), 
  a < 18 ∧ b < 18 ∧ c < 18 ∧ d < 18 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a * b * c * d = 882 ∧ a + b + c + d = 32 :=
by
  sorry

end nialls_children_ages_l210_210367


namespace inheritance_problem_l210_210739

theorem inheritance_problem
    (A B C : ℕ)
    (h1 : A + B + C = 30000)
    (h2 : A - B = B - C)
    (h3 : A = B + C) :
    A = 15000 ∧ B = 10000 ∧ C = 5000 := by
  sorry

end inheritance_problem_l210_210739


namespace sqrt_square_eq_self_sqrt_784_square_l210_210525

theorem sqrt_square_eq_self (n : ℕ) (h : n ≥ 0) : (Real.sqrt n) ^ 2 = n :=
by
  sorry

theorem sqrt_784_square : (Real.sqrt 784) ^ 2 = 784 :=
by
  exact sqrt_square_eq_self 784 (Nat.zero_le 784)

end sqrt_square_eq_self_sqrt_784_square_l210_210525


namespace chord_length_circle_l210_210996

theorem chord_length_circle {x y : ℝ} :
  (x - 1)^2 + (y - 1)^2 = 2 →
  (exists (p q : ℝ), (p-1)^2 = 1 ∧ (q-1)^2 = 1 ∧ p ≠ q ∧ abs (p - q) = 2) :=
by
  intro h
  use (2 : ℝ)
  use (0 : ℝ)
  -- Formal proof omitted
  sorry

end chord_length_circle_l210_210996


namespace saleswoman_commission_l210_210065

theorem saleswoman_commission (S : ℝ)
  (h1 : (S > 500) )
  (h2 : (0.20 * 500 + 0.50 * (S - 500)) = 0.3125 * S) : 
  S = 800 :=
sorry

end saleswoman_commission_l210_210065


namespace escalator_time_l210_210341

theorem escalator_time
    {d i s : ℝ}
    (h1 : d = 90 * i)
    (h2 : d = 30 * (i + s))
    (h3 : s = 2 * i):
    d / s = 45 := by
  sorry

end escalator_time_l210_210341


namespace diane_stamp_combinations_l210_210646

/-- Define the types of stamps Diane has --/
def diane_stamps : List ℕ := [1, 2, 2, 8, 8, 8, 8, 8, 8, 8, 8]

/-- Define the condition for the correct number of different arrangements to sum exactly to 12 cents -/
noncomputable def count_arrangements (stamps : List ℕ) (sum : ℕ) : ℕ :=
  -- Implementation of the counting function goes here
  sorry

/-- Prove that the number of distinct arrangements to make exactly 12 cents is 13 --/
theorem diane_stamp_combinations : count_arrangements diane_stamps 12 = 13 :=
  sorry

end diane_stamp_combinations_l210_210646


namespace laila_scores_possible_values_l210_210349

theorem laila_scores_possible_values :
  ∃ (num_y_values : ℕ), num_y_values = 4 ∧ 
  (∀ (x y : ℤ), 0 ≤ x ∧ x ≤ 100 ∧
                 0 ≤ y ∧ y ≤ 100 ∧
                 4 * x + y = 410 ∧
                 y > x → 
                 (y = 86 ∨ y = 90 ∨ y = 94 ∨ y = 98)
  ) :=
  ⟨4, by sorry⟩

end laila_scores_possible_values_l210_210349


namespace canonical_line_eq_l210_210808

-- Define the system of linear equations
def system_of_equations (x y z : ℝ) : Prop :=
  (2 * x - 3 * y - 2 * z + 6 = 0 ∧ x - 3 * y + z + 3 = 0)

-- Define the canonical equation of the line
def canonical_equation (x y z : ℝ) : Prop :=
  (x + 3) / 9 = y / 4 ∧ (x + 3) / 9 = z / 3 ∧ y / 4 = z / 3

-- The theorem to prove equivalence
theorem canonical_line_eq : 
  ∀ (x y z : ℝ), system_of_equations x y z → canonical_equation x y z :=
by
  intros x y z H
  sorry

end canonical_line_eq_l210_210808


namespace jerrys_current_average_score_l210_210129

theorem jerrys_current_average_score (A : ℝ) (h1 : 3 * A + 98 = 4 * (A + 2)) : A = 90 :=
by
  sorry

end jerrys_current_average_score_l210_210129


namespace worm_length_difference_l210_210679

def worm_1_length : ℝ := 0.8
def worm_2_length : ℝ := 0.1
def difference := worm_1_length - worm_2_length

theorem worm_length_difference : difference = 0.7 := by
  sorry

end worm_length_difference_l210_210679


namespace missile_time_equation_l210_210329

variable (x : ℝ)

def machToMetersPerSecond := 340
def missileSpeedInMach := 26
def secondsPerMinute := 60
def distanceToTargetInKilometers := 12000
def kilometersToMeters := 1000

theorem missile_time_equation :
  (missileSpeedInMach * machToMetersPerSecond * secondsPerMinute * x) / kilometersToMeters = distanceToTargetInKilometers :=
sorry

end missile_time_equation_l210_210329


namespace factorize_1_factorize_2_l210_210037

variable {a x y : ℝ}

theorem factorize_1 : 2 * a * x^2 - 8 * a * x * y + 8 * a * y^2 = 2 * a * (x - 2 * y)^2 := 
by
  sorry

theorem factorize_2 : 6 * x * y^2 - 9 * x^2 * y - y^3 = -y * (3 * x - y)^2 := 
by
  sorry

end factorize_1_factorize_2_l210_210037


namespace math_problem_l210_210182

open Real

theorem math_problem (α : ℝ) (h₁ : 0 < α) (h₂ : α < π / 2) (h₃ : cos (2 * π - α) - sin (π - α) = - sqrt 5 / 5) :
  (sin α + cos α = 3 * sqrt 5 / 5) ∧
  (cos (3 * π / 2 + α) ^ 2 + 2 * cos α * cos (π / 2 - α)) / (1 + sin (π / 2 - α) ^ 2) = 4 / 3 :=
by
  sorry

end math_problem_l210_210182


namespace paused_time_l210_210714

theorem paused_time (total_length remaining_length paused_at : ℕ) (h1 : total_length = 60) (h2 : remaining_length = 30) : paused_at = total_length - remaining_length :=
by
  sorry

end paused_time_l210_210714


namespace least_positive_x_l210_210632

theorem least_positive_x (x : ℕ) (h : (2 * x + 45)^2 % 43 = 0) : x = 42 :=
  sorry

end least_positive_x_l210_210632


namespace real_solution_for_any_y_l210_210336

theorem real_solution_for_any_y (x : ℝ) :
  (∀ y z : ℝ, x^2 + y^2 + z^2 + 2 * x * y * z = 1 → ∃ z : ℝ,  x^2 + y^2 + z^2 + 2 * x * y * z = 1) ↔ (x = 1 ∨ x = -1) :=
by sorry

end real_solution_for_any_y_l210_210336


namespace chapters_in_first_book_l210_210561

theorem chapters_in_first_book (x : ℕ) (h1 : 2 * 15 = 30) (h2 : (x + 30) / 2 + x + 30 = 75) : x = 20 :=
sorry

end chapters_in_first_book_l210_210561


namespace abs_ineq_solution_set_l210_210145

theorem abs_ineq_solution_set {x : ℝ} : (|x - 1| ≤ 2) ↔ (-1 ≤ x ∧ x ≤ 3) :=
by
  sorry

end abs_ineq_solution_set_l210_210145


namespace geometric_progression_product_sum_sumrecip_l210_210003

theorem geometric_progression_product_sum_sumrecip (P S S' : ℝ) (n : ℕ)
  (hP : P = a ^ n * r ^ ((n * (n - 1)) / 2))
  (hS : S = a * (1 - r ^ n) / (1 - r))
  (hS' : S' = (r ^ n - 1) / (a * (r - 1))) :
  P = (S / S') ^ (1 / 2 * n) :=
  sorry

end geometric_progression_product_sum_sumrecip_l210_210003


namespace directrix_of_parabola_l210_210212

def parabola_directrix (x_y_eqn : ℝ → ℝ) : ℝ := by
  -- Assuming the parabola equation x = -(1/4) y^2
  sorry

theorem directrix_of_parabola : parabola_directrix (fun y => -(1/4) * y^2) = 1 := by
  sorry

end directrix_of_parabola_l210_210212


namespace derivative_at_zero_l210_210831

-- Define the function f
def f (x : ℝ) : ℝ := x * (1 + x)

-- Statement of the problem: The derivative of f at 0 is 1
theorem derivative_at_zero : deriv f 0 = 1 := 
  sorry

end derivative_at_zero_l210_210831


namespace james_lifting_ratio_correct_l210_210415

theorem james_lifting_ratio_correct :
  let lt_initial := 2200
  let bw_initial := 245
  let lt_gain_percentage := 0.15
  let bw_gain := 8
  let lt_final := lt_initial + lt_initial * lt_gain_percentage
  let bw_final := bw_initial + bw_gain
  (lt_final / bw_final) = 10 :=
by
  sorry

end james_lifting_ratio_correct_l210_210415


namespace change_is_five_l210_210675

noncomputable def haircut_cost := 15
noncomputable def payment := 20
noncomputable def counterfeit := 20
noncomputable def exchanged_amount := (10 : ℤ) + 10
noncomputable def flower_shop_amount := 20

def change_given (payment haircut_cost: ℕ) : ℤ :=
payment - haircut_cost

theorem change_is_five : 
  change_given payment haircut_cost = 5 :=
by 
  sorry

end change_is_five_l210_210675


namespace infinite_series_equivalence_l210_210006

theorem infinite_series_equivalence (x y : ℝ) (hy : y ≠ 0 ∧ y ≠ 1) 
  (series_cond : ∑' n : ℕ, x / (y^(n+1)) = 3) :
  ∑' n : ℕ, x / ((x + 2*y)^(n+1)) = 3 * (y - 1) / (5*y - 4) := 
by
  sorry

end infinite_series_equivalence_l210_210006


namespace fraction_to_decimal_l210_210498

theorem fraction_to_decimal : (7 : Rat) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l210_210498


namespace common_element_exists_l210_210371

theorem common_element_exists {S : Fin 2011 → Set ℤ}
  (h_nonempty : ∀ (i : Fin 2011), (S i).Nonempty)
  (h_consecutive : ∀ (i : Fin 2011), ∃ a b : ℤ, S i = Set.Icc a b)
  (h_common : ∀ (i j : Fin 2011), (S i ∩ S j).Nonempty) :
  ∃ a : ℤ, 0 < a ∧ ∀ (i : Fin 2011), a ∈ S i := sorry

end common_element_exists_l210_210371


namespace teacher_li_sheets_l210_210797

theorem teacher_li_sheets (x : ℕ)
    (h1 : ∀ (n : ℕ), n = 24 → (x / 24) = ((x / 32) + 2)) :
    x = 192 := by
  sorry

end teacher_li_sheets_l210_210797


namespace number_square_25_l210_210865

theorem number_square_25 (x : ℝ) : x^2 = 25 ↔ x = 5 ∨ x = -5 := 
sorry

end number_square_25_l210_210865


namespace total_amount_of_money_if_all_cookies_sold_equals_1255_50_l210_210715

-- Define the conditions
def number_cookies_Clementine : ℕ := 72
def number_cookies_Jake : ℕ := 5 * number_cookies_Clementine / 2
def number_cookies_Tory : ℕ := (number_cookies_Jake + number_cookies_Clementine) / 2
def number_cookies_Spencer : ℕ := 3 * (number_cookies_Jake + number_cookies_Tory) / 2
def price_per_cookie : ℝ := 1.50

-- Total number of cookies
def total_cookies : ℕ :=
  number_cookies_Clementine + number_cookies_Jake + number_cookies_Tory + number_cookies_Spencer

-- Proof statement
theorem total_amount_of_money_if_all_cookies_sold_equals_1255_50 :
  (total_cookies * price_per_cookie : ℝ) = 1255.50 := by
  sorry

end total_amount_of_money_if_all_cookies_sold_equals_1255_50_l210_210715


namespace determinant_scaled_l210_210346

variables (x y z w : ℝ)
variables (det : ℝ)

-- Given condition: determinant of the 2x2 matrix is 7.
axiom det_given : det = x * w - y * z
axiom det_value : det = 7

-- The target to be proven: the determinant of the scaled matrix is 63.
theorem determinant_scaled (x y z w : ℝ) (det : ℝ) (h_det : det = x * w - y * z) (det_value : det = 7) : 
  3 * 3 * (x * w - y * z) = 63 :=
by
  sorry

end determinant_scaled_l210_210346


namespace douglas_vote_percentage_is_66_l210_210868

noncomputable def percentDouglasVotes (v : ℝ) : ℝ :=
  let votesX := 0.74 * (2 * v)
  let votesY := 0.5000000000000002 * v
  let totalVotes := 3 * v
  let totalDouglasVotes := votesX + votesY
  (totalDouglasVotes / totalVotes) * 100

theorem douglas_vote_percentage_is_66 :
  ∀ v : ℝ, percentDouglasVotes v = 66 := 
by
  intros v
  unfold percentDouglasVotes
  sorry

end douglas_vote_percentage_is_66_l210_210868


namespace egyptian_fraction_l210_210543

theorem egyptian_fraction (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) : 
  (2 : ℚ) / 7 = (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c :=
by
  sorry

end egyptian_fraction_l210_210543


namespace smallest_k_for_distinct_real_roots_l210_210734

noncomputable def discriminant (a b c : ℝ) := b^2 - 4 * a * c

theorem smallest_k_for_distinct_real_roots :
  ∃ k : ℤ, (k > 0) ∧ discriminant (k : ℝ) (-3) (-9/4) > 0 ∧ (∀ m : ℤ, discriminant (m : ℝ) (-3) (-9/4) > 0 → m ≥ k) := 
by
  sorry

end smallest_k_for_distinct_real_roots_l210_210734


namespace selling_price_of_mixture_per_litre_l210_210565

def cost_per_litre : ℝ := 3.60
def litres_of_pure_milk : ℝ := 25
def litres_of_water : ℝ := 5
def total_volume_of_mixture : ℝ := litres_of_pure_milk + litres_of_water
def total_cost_of_pure_milk : ℝ := cost_per_litre * litres_of_pure_milk

theorem selling_price_of_mixture_per_litre :
  total_cost_of_pure_milk / total_volume_of_mixture = 3 := by
  sorry

end selling_price_of_mixture_per_litre_l210_210565


namespace find_x_in_magic_square_l210_210800

def magicSquareProof (x d e f g h S : ℕ) : Prop :=
  (x + 25 + 75 = S) ∧
  (5 + d + e = S) ∧
  (f + g + h = S) ∧
  (x + d + h = S) ∧
  (f = 95) ∧
  (d = x - 70) ∧
  (h = 170 - x) ∧
  (e = x - 145) ∧
  (x + 25 + 75 = 5 + (x - 70) + (x - 145))

theorem find_x_in_magic_square : ∃ x d e f g h S, magicSquareProof x d e f g h S ∧ x = 310 := by
  sorry

end find_x_in_magic_square_l210_210800


namespace man_is_older_by_16_l210_210727

variable (M S : ℕ)

-- Condition: The present age of the son is 14.
def son_age := S = 14

-- Condition: In two years, the man's age will be twice the son's age.
def age_relation := M + 2 = 2 * (S + 2)

-- Theorem: Prove that the man is 16 years older than his son.
theorem man_is_older_by_16 (h1 : son_age S) (h2 : age_relation M S) : M - S = 16 := 
sorry

end man_is_older_by_16_l210_210727


namespace max_principals_in_8_years_l210_210620

theorem max_principals_in_8_years 
  (years_in_term : ℕ)
  (terms_in_given_period : ℕ)
  (term_length : ℕ)
  (term_length_eq : term_length = 4)
  (given_period : ℕ)
  (given_period_eq : given_period = 8) :
  terms_in_given_period = given_period / term_length :=
by
  rw [term_length_eq, given_period_eq]
  sorry

end max_principals_in_8_years_l210_210620


namespace union_M_N_is_R_l210_210193

open Set

/-- Define the sets M and N -/
def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | x < 3}

/-- Main goal: prove M ∪ N = ℝ -/
theorem union_M_N_is_R : M ∪ N = univ :=
by
  sorry

end union_M_N_is_R_l210_210193


namespace part_one_min_f_value_part_two_range_a_l210_210607

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a| + |x + a|

theorem part_one_min_f_value (x : ℝ) (a : ℝ) (h : a = 1) : f x a ≥ (3/2) :=
  sorry

theorem part_two_range_a (a : ℝ) : (11/2 < a) ∧ (a < 4.5) :=
  sorry

end part_one_min_f_value_part_two_range_a_l210_210607


namespace initial_mean_corrected_observations_l210_210832

theorem initial_mean_corrected_observations:
  ∃ M : ℝ, 
  (∀ (Sum_initial Sum_corrected : ℝ), 
    Sum_initial = 50 * M ∧ 
    Sum_corrected = Sum_initial + (48 - 23) → 
    Sum_corrected / 50 = 41.5) →
  M = 41 :=
by
  sorry

end initial_mean_corrected_observations_l210_210832


namespace parabola_incorrect_statement_B_l210_210326

theorem parabola_incorrect_statement_B 
  (y₁ y₂ : ℝ → ℝ) 
  (h₁ : ∀ x, y₁ x = 2 * x^2) 
  (h₂ : ∀ x, y₂ x = -2 * x^2) : 
  ¬ (∀ x < 0, y₁ x < y₁ (x + 1)) ∧ (∀ x < 0, y₂ x < y₂ (x + 1)) := 
by 
  sorry

end parabola_incorrect_statement_B_l210_210326


namespace probability_of_same_color_is_correct_l210_210716

-- Define the parameters for balls in the bag
def green_balls : ℕ := 8
def red_balls : ℕ := 6
def blue_balls : ℕ := 1
def total_balls : ℕ := green_balls + red_balls + blue_balls

-- Define the probabilities of drawing each color
def prob_green : ℚ := green_balls / total_balls
def prob_red : ℚ := red_balls / total_balls
def prob_blue : ℚ := blue_balls / total_balls

-- Define the probability of drawing two balls of the same color
def prob_same_color : ℚ :=
  prob_green^2 + prob_red^2 + prob_blue^2

theorem probability_of_same_color_is_correct :
  prob_same_color = 101 / 225 :=
by
  sorry

end probability_of_same_color_is_correct_l210_210716


namespace tape_recorder_cost_l210_210355

-- Define the conditions
def conditions (x p : ℚ) : Prop :=
  170 < p ∧ p < 195 ∧
  2 * p = x * (x - 2) ∧
  1 * x = x - 2 + 2

-- Define the statement to be proved
theorem tape_recorder_cost (x : ℚ) (p : ℚ) : conditions x p → p = 180 := by
  sorry

end tape_recorder_cost_l210_210355


namespace prime_product_solution_l210_210255

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_product_solution (p_1 p_2 p_3 p_4 : ℕ) :
  is_prime p_1 ∧ is_prime p_2 ∧ is_prime p_3 ∧ is_prime p_4 ∧ 
  p_1 ≠ p_2 ∧ p_1 ≠ p_3 ∧ p_1 ≠ p_4 ∧ p_2 ≠ p_3 ∧ p_2 ≠ p_4 ∧ p_3 ≠ p_4 ∧
  2 * p_1 + 3 * p_2 + 5 * p_3 + 7 * p_4 = 162 ∧
  11 * p_1 + 7 * p_2 + 5 * p_3 + 4 * p_4 = 162 
  → p_1 * p_2 * p_3 * p_4 = 570 := 
by
  sorry

end prime_product_solution_l210_210255


namespace original_volume_l210_210668

theorem original_volume (V : ℝ) (h1 : V > 0) 
    (h2 : (1/16) * V = 0.75) : V = 12 :=
by sorry

end original_volume_l210_210668


namespace croissant_process_time_in_hours_l210_210315

-- Conditions as definitions
def num_folds : ℕ := 4
def fold_time : ℕ := 5
def rest_time : ℕ := 75
def mix_time : ℕ := 10
def bake_time : ℕ := 30

-- The main theorem statement
theorem croissant_process_time_in_hours :
  (num_folds * (fold_time + rest_time) + mix_time + bake_time) / 60 = 6 := 
sorry

end croissant_process_time_in_hours_l210_210315


namespace mike_total_games_l210_210660

theorem mike_total_games
  (non_working : ℕ)
  (price_per_game : ℕ)
  (total_earnings : ℕ)
  (h1 : non_working = 9)
  (h2 : price_per_game = 5)
  (h3 : total_earnings = 30) :
  non_working + (total_earnings / price_per_game) = 15 := 
by
  sorry

end mike_total_games_l210_210660


namespace certain_number_is_7000_l210_210268

theorem certain_number_is_7000 (x : ℕ) (h1 : 1 / 10 * (1 / 100 * x) = x / 1000)
    (h2 : 1 / 10 * x = x / 10)
    (h3 : x / 10 - x / 1000 = 693) : 
  x = 7000 := 
sorry

end certain_number_is_7000_l210_210268


namespace not_every_tv_owner_has_pass_l210_210652

variable (Person : Type) (T P G : Person → Prop)

-- Condition 1: There exists a television owner who is not a painter.
axiom exists_tv_owner_not_painter : ∃ x, T x ∧ ¬ P x 

-- Condition 2: If someone has a pass to the Gellért Baths and is not a painter, they are not a television owner.
axiom pass_and_not_painter_imp_not_tv_owner : ∀ x, (G x ∧ ¬ P x) → ¬ T x

-- Prove: Not every television owner has a pass to the Gellért Baths.
theorem not_every_tv_owner_has_pass :
  ¬ ∀ x, T x → G x :=
by
  sorry -- Proof omitted

end not_every_tv_owner_has_pass_l210_210652


namespace box_max_volume_l210_210843

theorem box_max_volume (x : ℝ) (h1 : 0 < x) (h2 : x < 5) :
    (10 - 2 * x) * (16 - 2 * x) * x ≤ 144 :=
by
  -- The proof will be filled here
  sorry

end box_max_volume_l210_210843


namespace assignment_plans_proof_l210_210837

noncomputable def total_assignment_plans : ℕ :=
  let volunteers := ["Xiao Zhang", "Xiao Zhao", "Xiao Li", "Xiao Luo", "Xiao Wang"]
  let positions := ["translation", "tour guide", "etiquette", "driver"]
  -- Definitions for eligible volunteers for the first two positions
  let first_positions := ["Xiao Zhang", "Xiao Zhao"]
  let remaining_positions := ["Xiao Li", "Xiao Luo", "Xiao Wang"]
  -- Assume the computation for the exact number which results in 36
  36

theorem assignment_plans_proof : total_assignment_plans = 36 := 
  by 
  -- Proof skipped
  sorry

end assignment_plans_proof_l210_210837


namespace problem1_problem2_l210_210385

-- Definitions for sets A and S
def setA (x : ℝ) : Prop := -7 ≤ 2 * x - 5 ∧ 2 * x - 5 ≤ 9
def setS (x k : ℝ) : Prop := k + 1 ≤ x ∧ x ≤ 2 * k - 1

-- Preliminary ranges for x
lemma range_A : ∀ x, setA x ↔ -1 ≤ x ∧ x ≤ 7 := sorry

noncomputable def k_range1 (k : ℝ) : Prop := 2 ≤ k ∧ k ≤ 4
noncomputable def k_range2 (k : ℝ) : Prop := k < 2 ∨ k > 6

-- Proof problems in Lean 4

-- First problem statement
theorem problem1 (k : ℝ) : (∀ x, setS x k → setA x) ∧ (∃ x, setS x k) → k_range1 k := sorry

-- Second problem statement
theorem problem2 (k : ℝ) : (∀ x, ¬(setA x ∧ setS x k)) → k_range2 k := sorry

end problem1_problem2_l210_210385


namespace max_quotient_l210_210822

theorem max_quotient (a b : ℝ) (ha : 100 ≤ a ∧ a ≤ 300) (hb : 500 ≤ b ∧ b ≤ 1500) : (b / a) ≤ 15 :=
  sorry

end max_quotient_l210_210822


namespace find_difference_in_ticket_costs_l210_210137

-- Conditions
def num_adults : ℕ := 9
def num_children : ℕ := 7
def cost_adult_ticket : ℕ := 11
def cost_child_ticket : ℕ := 7

def total_cost_adults : ℕ := num_adults * cost_adult_ticket
def total_cost_children : ℕ := num_children * cost_child_ticket
def total_tickets : ℕ := num_adults + num_children

-- Discount conditions (not needed for this proof since they don't apply)
def apply_discount (total_tickets : ℕ) (total_cost : ℕ) : ℕ :=
  if total_tickets >= 10 ∧ total_tickets <= 12 then
    total_cost * 9 / 10
  else if total_tickets >= 13 ∧ total_tickets <= 15 then
    total_cost * 85 / 100
  else
    total_cost

-- The main statement to prove
theorem find_difference_in_ticket_costs : total_cost_adults - total_cost_children = 50 := by
  sorry

end find_difference_in_ticket_costs_l210_210137


namespace correct_calculation_l210_210192

theorem correct_calculation :
  - (1 / 2) - (- (1 / 3)) = - (1 / 6) :=
by
  sorry

end correct_calculation_l210_210192


namespace original_angle_measure_l210_210168

theorem original_angle_measure : 
  ∃ x : ℝ, (90 - x) = 3 * x - 2 ∧ x = 23 :=
by
  sorry

end original_angle_measure_l210_210168


namespace standard_circle_eq_l210_210031

noncomputable def circle_equation : String :=
  "The standard equation of the circle whose center lies on the line y = -4x and is tangent to the line x + y - 1 = 0 at point P(3, -2) is (x - 1)^2 + (y + 4)^2 = 8"

theorem standard_circle_eq
  (center_x : ℝ)
  (center_y : ℝ)
  (tangent_line : ℝ → ℝ → Prop)
  (point : ℝ × ℝ)
  (eqn_line : ∀ x y, tangent_line x y ↔ x + y - 1 = 0)
  (center_on_line : ∀ x y, y = -4 * x → center_y = y)
  (point_on_tangent : point = (3, -2))
  (tangent_at_point : tangent_line (point.1) (point.2)) :
  (center_x = 1 ∧ center_y = -4 ∧ (∃ r : ℝ, r = 2 * Real.sqrt 2)) →
  (∀ x y, (x - 1)^2 + (y + 4)^2 = 8) := by
  sorry

end standard_circle_eq_l210_210031


namespace quadratic_function_inequality_l210_210233

theorem quadratic_function_inequality
  (x1 x2 : ℝ) (y1 y2 : ℝ)
  (hx1_pos : 0 < x1)
  (hx2_pos : x1 < x2)
  (hy1 : y1 = x1^2 - 1)
  (hy2 : y2 = x2^2 - 1) :
  y1 < y2 := 
sorry

end quadratic_function_inequality_l210_210233


namespace circle_ratio_increase_l210_210282

theorem circle_ratio_increase (r : ℝ) (h : r + 2 ≠ 0) : 
  (2 * Real.pi * (r + 2)) / (2 * (r + 2)) = Real.pi :=
by
  sorry

end circle_ratio_increase_l210_210282


namespace equation_has_three_distinct_solutions_iff_l210_210084

theorem equation_has_three_distinct_solutions_iff (a : ℝ) : 
  (∃ x_1 x_2 x_3 : ℝ, x_1 ≠ x_2 ∧ x_2 ≠ x_3 ∧ x_1 ≠ x_3 ∧ 
    (x_1 * |x_1 - a| = 1) ∧ (x_2 * |x_2 - a| = 1) ∧ (x_3 * |x_3 - a| = 1)) ↔ a > 2 :=
by
  sorry


end equation_has_three_distinct_solutions_iff_l210_210084


namespace bills_difference_l210_210202

noncomputable def Mike_tip : ℝ := 5
noncomputable def Joe_tip : ℝ := 10
noncomputable def Mike_percentage : ℝ := 20
noncomputable def Joe_percentage : ℝ := 25

theorem bills_difference
  (m j : ℝ)
  (Mike_condition : (Mike_percentage / 100) * m = Mike_tip)
  (Joe_condition : (Joe_percentage / 100) * j = Joe_tip) :
  |m - j| = 15 :=
by
  sorry

end bills_difference_l210_210202


namespace florist_has_56_roses_l210_210183

def initial_roses := 50
def roses_sold := 15
def roses_picked := 21

theorem florist_has_56_roses (r0 rs rp : ℕ) (h1 : r0 = initial_roses) (h2 : rs = roses_sold) (h3 : rp = roses_picked) : 
  r0 - rs + rp = 56 :=
by sorry

end florist_has_56_roses_l210_210183


namespace ratio_payment_shared_side_l210_210849

variable (length_side length_back : ℕ) (cost_per_foot cole_payment : ℕ)
variables (neighbor_back_contrib neighbor_left_contrib total_cost_fence : ℕ)
variables (total_cost_shared_side : ℕ)

theorem ratio_payment_shared_side
  (h1 : length_side = 9)
  (h2 : length_back = 18)
  (h3 : cost_per_foot = 3)
  (h4 : cole_payment = 72)
  (h5 : neighbor_back_contrib = (length_back / 2) * cost_per_foot)
  (h6 : total_cost_fence = (2* length_side + length_back) * cost_per_foot)
  (h7 : total_cost_shared_side = length_side * cost_per_foot)
  (h8 : cole_left_total_payment = cole_payment + neighbor_back_contrib)
  (h9 : neighbor_left_contrib = cole_left_total_payment - cole_payment):
  neighbor_left_contrib / total_cost_shared_side = 1 := 
sorry

end ratio_payment_shared_side_l210_210849


namespace downstream_speed_l210_210044

noncomputable def speed_downstream (Vu Vs : ℝ) : ℝ :=
  2 * Vs - Vu

theorem downstream_speed (Vu Vs : ℝ) (hVu : Vu = 30) (hVs : Vs = 45) :
  speed_downstream Vu Vs = 60 := by
  rw [hVu, hVs]
  dsimp [speed_downstream]
  linarith

end downstream_speed_l210_210044


namespace range_of_m_l210_210056

theorem range_of_m (m : ℝ) (h : 2 * m + 3 < 4) : m < 1 / 2 :=
by
  sorry

end range_of_m_l210_210056


namespace geoff_tuesday_multiple_l210_210540

variable (monday_spending : ℝ) (tuesday_multiple : ℝ) (total_spending : ℝ)

-- Given conditions
def geoff_conditions (monday_spending tuesday_multiple total_spending : ℝ) : Prop :=
  monday_spending = 60 ∧
  (tuesday_multiple * monday_spending) + (5 * monday_spending) + monday_spending = total_spending ∧
  total_spending = 600

-- Proof goal
theorem geoff_tuesday_multiple (monday_spending tuesday_multiple total_spending : ℝ)
  (h : geoff_conditions monday_spending tuesday_multiple total_spending) : 
  tuesday_multiple = 4 :=
by
  sorry

end geoff_tuesday_multiple_l210_210540


namespace find_b_value_l210_210247

theorem find_b_value (a b c A B C : ℝ) 
  (h1 : a = 1)
  (h2 : B = 120 * (π / 180))
  (h3 : c = b * Real.cos C + c * Real.cos B)
  (h4 : c = 1) : 
  b = Real.sqrt 3 :=
by
  sorry

end find_b_value_l210_210247


namespace prove_smallest_number_l210_210976

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

lemma smallest_number_to_add (n : ℕ) (k : ℕ) (h: sum_of_digits n % k = r) : n % k = r →
  n % k = r → (k - r) = 7 :=
by
  sorry

theorem prove_smallest_number (n : ℕ) (k : ℕ) (r : ℕ) :
  (27452 % 9 = r) ∧ (9 - r = 7) :=
by
  sorry

end prove_smallest_number_l210_210976


namespace sandy_siding_cost_l210_210687

theorem sandy_siding_cost:
  let wall_width := 8
  let wall_height := 8
  let roof_width := 8
  let roof_height := 5
  let siding_width := 10
  let siding_height := 12
  let siding_cost := 30
  let wall_area := wall_width * wall_height
  let roof_side_area := roof_width * roof_height
  let roof_area := 2 * roof_side_area
  let total_area := wall_area + roof_area
  let siding_area := siding_width * siding_height
  let required_sections := (total_area + siding_area - 1) / siding_area -- ceiling division
  let total_cost := required_sections * siding_cost
  total_cost = 60 :=
by
  sorry

end sandy_siding_cost_l210_210687


namespace x_plus_y_eq_3012_plus_pi_div_2_l210_210113

theorem x_plus_y_eq_3012_plus_pi_div_2
  (x y : ℝ)
  (h1 : x + Real.cos y = 3012)
  (h2 : x + 3012 * Real.sin y = 3010)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi) :
  x + y = 3012 + Real.pi / 2 :=
sorry

end x_plus_y_eq_3012_plus_pi_div_2_l210_210113


namespace find_z_l210_210975

/- Definitions of angles and their relationships -/
def angle_sum_triangle (A B C : ℝ) : Prop := A + B + C = 180

/- Given conditions -/
def ABC : ℝ := 75
def BAC : ℝ := 55
def BCA : ℝ := 180 - ABC - BAC  -- This follows from the angle sum property of triangle ABC
def DCE : ℝ := BCA
def CDE : ℝ := 90

/- Prove z given the above conditions -/
theorem find_z : ∃ (z : ℝ), z = 90 - DCE := by
  use 40
  sorry

end find_z_l210_210975


namespace heather_total_distance_l210_210278

theorem heather_total_distance :
  let d1 := 0.3333333333333333
  let d2 := 0.3333333333333333
  let d3 := 0.08333333333333333
  d1 + d2 + d3 = 0.75 :=
by
  sorry

end heather_total_distance_l210_210278


namespace permutation_inequality_l210_210443

theorem permutation_inequality (a b c d : ℝ) (h : a * b * c * d > 0) :
  ∃ (x y z w : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d) ∧ (y = a ∨ y = b ∨ y = c ∨ y = d) ∧
  (z = a ∨ z = b ∨ z = c ∨ z = d) ∧ (w = a ∨ w = b ∨ w = c ∨ w = d) ∧ 
  2 * (x * y + z * w)^2 > (x^2 + y^2) * (z^2 + w^2) := 
sorry

end permutation_inequality_l210_210443


namespace problem_l210_210754

def f (x: ℝ) := 3 * x - 4
def g (x: ℝ) := 2 * x + 3

theorem problem (x : ℝ) : f (2 + g 3) = 29 :=
by
  sorry

end problem_l210_210754


namespace students_passed_correct_l210_210358

-- Define the number of students in ninth grade.
def students_total : ℕ := 180

-- Define the number of students who bombed their finals.
def students_bombed : ℕ := students_total / 4

-- Define the number of students remaining after removing those who bombed.
def students_remaining_after_bombed : ℕ := students_total - students_bombed

-- Define the number of students who didn't show up to take the test.
def students_didnt_show : ℕ := students_remaining_after_bombed / 3

-- Define the number of students remaining after removing those who didn't show up.
def students_remaining_after_no_show : ℕ := students_remaining_after_bombed - students_didnt_show

-- Define the number of students who got less than a D.
def students_less_than_d : ℕ := 20

-- Define the number of students who passed.
def students_passed : ℕ := students_remaining_after_no_show - students_less_than_d

-- Statement to prove the number of students who passed is 70.
theorem students_passed_correct : students_passed = 70 := by
  -- Proof will be inserted here.
  sorry

end students_passed_correct_l210_210358


namespace common_value_of_4a_and_5b_l210_210886

theorem common_value_of_4a_and_5b (a b C : ℝ) (h1 : 4 * a = C) (h2 : 5 * b = C) (h3 : 40 * a * b = 1800) :
  C = 60 :=
sorry

end common_value_of_4a_and_5b_l210_210886


namespace problem1_problem2_l210_210796

-- Theorem for problem 1
theorem problem1 (a b : ℤ) : (a^3 * b^4) ^ 2 / (a * b^2) ^ 3 = a^3 * b^2 := 
by sorry

-- Theorem for problem 2
theorem problem2 (a : ℤ) : (-a^2) ^ 3 * a^2 + a^8 = 0 := 
by sorry

end problem1_problem2_l210_210796


namespace area_BCD_l210_210082

open Real EuclideanGeometry

noncomputable def point := (ℝ × ℝ)
noncomputable def A : point := (0, 0)
noncomputable def B : point := (10, 24)
noncomputable def C : point := (30, 0)
noncomputable def D : point := (40, 0)

def area_triangle (p1 p2 p3 : point) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  0.5 * |x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)|

theorem area_BCD : area_triangle B C D = 12 := sorry

end area_BCD_l210_210082


namespace find_original_number_of_men_l210_210746

theorem find_original_number_of_men (x : ℕ) (h1 : x * 12 = (x - 6) * 14) : x = 42 :=
  sorry

end find_original_number_of_men_l210_210746


namespace evaluate_expression_l210_210711

theorem evaluate_expression (x y z : ℤ) (hx : x = 5) (hy : y = x + 3) (hz : z = y - 11) 
  (h₁ : x + 2 ≠ 0) (h₂ : y - 3 ≠ 0) (h₃ : z + 7 ≠ 0) : 
  ((x + 3) / (x + 2)) * ((y - 2) / (y - 3)) * ((z + 9) / (z + 7)) = 72 / 35 := 
by 
  sorry

end evaluate_expression_l210_210711


namespace divisor_of_a_l210_210454

namespace MathProofProblem

-- Define the given problem
variable (a b c d : ℕ) -- Variables representing positive integers

-- Given conditions
variables (h_gcd_ab : Nat.gcd a b = 30)
variables (h_gcd_bc : Nat.gcd b c = 42)
variables (h_gcd_cd : Nat.gcd c d = 66)
variables (h_lcm_cd : Nat.lcm c d = 2772)
variables (h_gcd_da : 100 < Nat.gcd d a ∧ Nat.gcd d a < 150)

-- Target statement to prove
theorem divisor_of_a : 13 ∣ a :=
by
  sorry

end MathProofProblem

end divisor_of_a_l210_210454


namespace infinite_series_eq_1_div_400_l210_210936

theorem infinite_series_eq_1_div_400 :
  (∑' n:ℕ, (4 * n + 2) / ((4 * n + 1)^2 * (4 * n + 5)^2)) = 1 / 400 :=
by
  sorry

end infinite_series_eq_1_div_400_l210_210936


namespace survey_preference_l210_210136

theorem survey_preference (X Y : ℕ) 
  (ratio_condition : X / Y = 5)
  (total_respondents : X + Y = 180) :
  X = 150 := 
sorry

end survey_preference_l210_210136


namespace female_employees_l210_210060

theorem female_employees (total_employees male_employees : ℕ) 
  (advanced_degree_male_adv: ℝ) (advanced_degree_female_adv: ℝ) (prob: ℝ) 
  (h1 : total_employees = 450) 
  (h2 : male_employees = 300)
  (h3 : advanced_degree_male_adv = 0.10) 
  (h4 : advanced_degree_female_adv = 0.40)
  (h5 : prob = 0.4) : 
  ∃ F : ℕ, 0.10 * male_employees + (advanced_degree_female_adv * F + (1 - advanced_degree_female_adv) * F) / total_employees = prob ∧ F = 150 :=
by
  sorry

end female_employees_l210_210060


namespace difference_between_balls_l210_210985

theorem difference_between_balls (B R : ℕ) (h1 : R - 152 = B + 152 + 346) : R - B = 650 := 
sorry

end difference_between_balls_l210_210985


namespace cos_x_plus_2y_is_one_l210_210185

theorem cos_x_plus_2y_is_one
    (x y : ℝ) (a : ℝ) 
    (hx : x ∈ Set.Icc (-Real.pi) Real.pi)
    (hy : y ∈ Set.Icc (-Real.pi) Real.pi)
    (h_eq : 2 * a = x ^ 3 + Real.sin x ∧ 2 * a = (-2 * y) ^ 3 - Real.sin (-2 * y)) :
    Real.cos (x + 2 * y) = 1 := 
sorry

end cos_x_plus_2y_is_one_l210_210185


namespace solution_to_problem_l210_210841

def problem_statement : Prop :=
  (3^202 + 7^203)^2 - (3^202 - 7^203)^2 = 59 * 10^202

theorem solution_to_problem : problem_statement := 
  by sorry

end solution_to_problem_l210_210841


namespace sin_double_angle_l210_210359

open Real

theorem sin_double_angle (θ : ℝ) (h : cos (π / 4 - θ) = 1 / 2) : sin (2 * θ) = -1 / 2 := 
by 
  sorry

end sin_double_angle_l210_210359


namespace has_buried_correct_number_of_bones_l210_210347

def bones_received_per_month : ℕ := 10
def number_of_months : ℕ := 5
def bones_available : ℕ := 8

def total_bones_received : ℕ := bones_received_per_month * number_of_months
def bones_buried : ℕ := total_bones_received - bones_available

theorem has_buried_correct_number_of_bones : bones_buried = 42 := by
  sorry

end has_buried_correct_number_of_bones_l210_210347


namespace arithmetic_sequence_general_term_l210_210523

theorem arithmetic_sequence_general_term (a_n S_n : ℕ → ℕ) (d : ℕ) (a1 S1 S5 S7 : ℕ)
  (h1: a_n 3 = 5)
  (h2: ∀ n, S_n n = (n * (a1 * 2 + (n - 1) * d)) / 2)
  (h3: S1 = S_n 1)
  (h4: S5 = S_n 5)
  (h5: S7 = S_n 7)
  (h6: S1 + S7 = 2 * S5):
  ∀ n, a_n n = 2 * n - 1 :=
by
  sorry

end arithmetic_sequence_general_term_l210_210523


namespace cosA_value_area_of_triangle_l210_210294

noncomputable def cosA (a b c : ℝ) (cos_C : ℝ) : ℝ :=
  if (a ≠ 0 ∧ cos_C ≠ 0) then (2 * b - c) * cos_C / a else 1 / 2

noncomputable def area_triangle (a b c : ℝ) (cosA_val : ℝ) : ℝ :=
  let S := a * b * (Real.sqrt (1 - cosA_val ^ 2)) / 2
  S

theorem cosA_value (a b c : ℝ) (cos_C : ℝ) : a * cos_C = (2 * b - c) * (cosA a b c cos_C) → cosA a b c cos_C = 1 / 2 :=
by
  sorry

theorem area_of_triangle (a b c : ℝ) (cos_A : ℝ) (cos_A_proof : a * cos_C = (2 * b - c) * cos_A) (h₀ : a = 6) (h₁ : b + c = 8) : area_triangle a b c cos_A = 7 * Real.sqrt 3 / 3 :=
by
  sorry

end cosA_value_area_of_triangle_l210_210294


namespace find_A_l210_210661

-- Definitions and conditions
def f (A B : ℝ) (x : ℝ) : ℝ := A * x - 3 * B^2 
def g (B C : ℝ) (x : ℝ) : ℝ := B * x + C

theorem find_A (A B C : ℝ) (hB : B ≠ 0) (hBC : B + C ≠ 0) :
  f A B (g B C 1) = 0 → A = (3 * B^2) / (B + C) :=
by
  -- Introduction of the hypotheses
  intro h
  sorry

end find_A_l210_210661


namespace total_participants_l210_210824

-- Define the number of indoor and outdoor participants
variables (x y : ℕ)

-- First condition: number of outdoor participants is 480 more than indoor participants
def condition1 : Prop := y = x + 480

-- Second condition: moving 50 participants results in outdoor participants being 5 times the indoor participants
def condition2 : Prop := y + 50 = 5 * (x - 50)

-- Theorem statement: the total number of participants is 870
theorem total_participants (h1 : condition1 x y) (h2 : condition2 x y) : x + y = 870 :=
sorry

end total_participants_l210_210824


namespace exists_consecutive_non_primes_l210_210201

theorem exists_consecutive_non_primes (k : ℕ) (hk : 0 < k) : 
  ∃ n : ℕ, ∀ i : ℕ, i < k → ¬Nat.Prime (n + i) := 
sorry

end exists_consecutive_non_primes_l210_210201


namespace ab_multiple_of_7_2010_l210_210203

theorem ab_multiple_of_7_2010 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : 7 ^ 2009 ∣ a^2 + b^2) : 7 ^ 2010 ∣ a * b :=
by
  sorry

end ab_multiple_of_7_2010_l210_210203


namespace total_campers_went_rowing_l210_210303

-- Definitions based on given conditions
def morning_campers : ℕ := 36
def afternoon_campers : ℕ := 13
def evening_campers : ℕ := 49

-- Theorem statement to be proven
theorem total_campers_went_rowing : morning_campers + afternoon_campers + evening_campers = 98 :=
by sorry

end total_campers_went_rowing_l210_210303


namespace correlation_1_and_3_l210_210094

-- Define the conditions as types
def relationship1 : Type := ∀ (age : ℕ) (fat_content : ℝ), Prop
def relationship2 : Type := ∀ (curve_point : ℝ × ℝ), Prop
def relationship3 : Type := ∀ (production : ℝ) (climate : ℝ), Prop
def relationship4 : Type := ∀ (student : ℕ) (student_ID : ℕ), Prop

-- Define what it means for two relationships to have a correlation
def has_correlation (rel1 rel2 : Type) : Prop := 
  -- Some formal definition of correlation suitable for the context
  sorry

-- Theorem stating that relationships (1) and (3) have a correlation
theorem correlation_1_and_3 :
  has_correlation relationship1 relationship3 :=
sorry

end correlation_1_and_3_l210_210094


namespace relationship_of_coefficients_l210_210578

theorem relationship_of_coefficients (a b c : ℝ) (α β : ℝ) 
  (h_eq : a * α^2 + b * α + c = 0) 
  (h_eq' : a * β^2 + b * β + c = 0) 
  (h_roots : β = 3 * α) :
  3 * b^2 = 16 * a * c := 
sorry

end relationship_of_coefficients_l210_210578


namespace gcd_459_357_l210_210238

/-- Prove that the greatest common divisor of 459 and 357 is 51. -/
theorem gcd_459_357 : gcd 459 357 = 51 :=
by
  sorry

end gcd_459_357_l210_210238


namespace discount_percentage_l210_210585

theorem discount_percentage 
  (C : ℝ) (S : ℝ) (P : ℝ) (SP : ℝ)
  (h1 : C = 48)
  (h2 : 0.60 * S = C)
  (h3 : P = 16)
  (h4 : P = S - SP)
  (h5 : SP = 80 - 16)
  (h6 : S = 80) :
  (S - SP) / S * 100 = 20 := by
sorry

end discount_percentage_l210_210585


namespace Sarah_collected_40_today_l210_210163

noncomputable def Sarah_yesterday : ℕ := 50
noncomputable def Lara_yesterday : ℕ := Sarah_yesterday + 30
noncomputable def Lara_today : ℕ := 70
noncomputable def Total_yesterday : ℕ := Sarah_yesterday + Lara_yesterday
noncomputable def Total_today : ℕ := Total_yesterday - 20
noncomputable def Sarah_today : ℕ := Total_today - Lara_today

theorem Sarah_collected_40_today : Sarah_today = 40 := 
by
  sorry

end Sarah_collected_40_today_l210_210163


namespace reciprocal_of_8_l210_210928

theorem reciprocal_of_8:
  (1 : ℝ) / 8 = (1 / 8 : ℝ) := by
  sorry

end reciprocal_of_8_l210_210928


namespace pies_difference_l210_210339

theorem pies_difference (time : ℕ) (alice_time : ℕ) (bob_time : ℕ) (charlie_time : ℕ)
    (h_time : time = 90) (h_alice : alice_time = 5) (h_bob : bob_time = 6) (h_charlie : charlie_time = 7) :
    (time / alice_time - time / bob_time) + (time / alice_time - time / charlie_time) = 9 := by
  sorry

end pies_difference_l210_210339


namespace g_of_neg2_l210_210696

def g (x : ℚ) : ℚ := (2 * x - 3) / (4 * x + 5)

theorem g_of_neg2 : g (-2) = 7 / 3 := by
  sorry

end g_of_neg2_l210_210696


namespace parabola_locus_l210_210102

variables (a c k : ℝ) (a_pos : 0 < a) (c_pos : 0 < c) (k_pos : 0 < k)

theorem parabola_locus :
  ∀ t : ℝ, ∃ x y : ℝ,
    x = -kt / (2 * a) ∧ y = - k^2 * t^2 / (4 * a) + c ∧
    y = - (k^2 / (4 * a)) * x^2 + c :=
sorry

end parabola_locus_l210_210102


namespace length_of_wall_l210_210340

-- Define the dimensions of a brick
def brick_length : ℝ := 40
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- Define the dimensions of the wall
def wall_height : ℝ := 600
def wall_width : ℝ := 22.5

-- Define the required number of bricks
def required_bricks : ℝ := 4000

-- Calculate the volume of a single brick
def volume_brick : ℝ := brick_length * brick_width * brick_height

-- Calculate the volume of the wall
def volume_wall (length : ℝ) : ℝ := length * wall_height * wall_width

-- The theorem to prove
theorem length_of_wall : ∃ (L : ℝ), required_bricks * volume_brick = volume_wall L → L = 800 :=
sorry

end length_of_wall_l210_210340


namespace combined_cost_of_one_item_l210_210076

-- Definitions representing the given conditions
def initial_amount : ℝ := 50
def final_amount : ℝ := 14
def mangoes_purchased : ℕ := 6
def apple_juice_purchased : ℕ := 6

-- Hypothesis: The cost of mangoes and apple juice are the same
variables (M A : ℝ)

-- Total amount spent
def amount_spent : ℝ := initial_amount - final_amount

-- Combined number of items
def total_items : ℕ := mangoes_purchased + apple_juice_purchased

-- Lean statement to prove the combined cost of one mango and one carton of apple juice is $3
theorem combined_cost_of_one_item (h : mangoes_purchased * M + apple_juice_purchased * A = amount_spent) :
    (amount_spent / total_items) = (3 : ℝ) :=
by
  sorry

end combined_cost_of_one_item_l210_210076


namespace min_value_of_expr_min_value_achieved_final_statement_l210_210184

theorem min_value_of_expr (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 3) :
  1 ≤ (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) :=
by
  sorry

theorem min_value_achieved (x y z : ℝ) (h1 : x = 1) (h2 : y = 1) (h3 : z = 1) :
  1 = (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) :=
by
  sorry

theorem final_statement (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 3) :
  ∃ (x y z : ℝ), (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x + y + z = 3) ∧ (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x) = 1) :=
by
  sorry

end min_value_of_expr_min_value_achieved_final_statement_l210_210184


namespace smallest_number_divisibility_l210_210176

theorem smallest_number_divisibility :
  ∃ x, (x + 3) % 70 = 0 ∧ (x + 3) % 100 = 0 ∧ (x + 3) % 84 = 0 ∧ x = 6303 :=
sorry

end smallest_number_divisibility_l210_210176


namespace fencing_cost_proof_l210_210085

theorem fencing_cost_proof (L : ℝ) (B : ℝ) (c : ℝ) (total_cost : ℝ)
  (hL : L = 60) (hL_B : L = B + 20) (hc : c = 26.50) : 
  total_cost = 5300 :=
by
  sorry

end fencing_cost_proof_l210_210085


namespace ratio_a_d_l210_210226

variables (a b c d : ℕ)

-- Given conditions
def ratio_ab := 8 / 3
def ratio_bc := 1 / 5
def ratio_cd := 3 / 2
def b_value := 27

theorem ratio_a_d (h₁ : a / b = ratio_ab)
                  (h₂ : b / c = ratio_bc)
                  (h₃ : c / d = ratio_cd)
                  (h₄ : b = b_value) :
  a / d = 4 / 5 :=
sorry

end ratio_a_d_l210_210226


namespace min_n_for_constant_term_l210_210693

theorem min_n_for_constant_term :
  (∃ n : ℕ, n > 0 ∧ ∃ r : ℕ, 3 * n = 5 * r) → ∃ n : ℕ, n = 5 :=
by
  intros h
  sorry

end min_n_for_constant_term_l210_210693


namespace faulty_balance_inequality_l210_210225

variable (m n a b G : ℝ)

theorem faulty_balance_inequality
  (h1 : m * a = n * G)
  (h2 : n * b = m * G) :
  (a + b) / 2 > G :=
sorry

end faulty_balance_inequality_l210_210225


namespace complement_of_A_in_U_l210_210380

def U : Set ℝ := {x | x > 0}
def A : Set ℝ := {x | x ≥ 2}
def complement_U_A : Set ℝ := {x | 0 < x ∧ x < 2}

theorem complement_of_A_in_U :
  (U \ A) = complement_U_A :=
sorry

end complement_of_A_in_U_l210_210380


namespace most_followers_after_three_weeks_l210_210425

def initial_followers_susy := 100
def initial_followers_sarah := 50
def first_week_gain_susy := 40
def second_week_gain_susy := first_week_gain_susy / 2
def third_week_gain_susy := second_week_gain_susy / 2
def first_week_gain_sarah := 90
def second_week_gain_sarah := first_week_gain_sarah / 3
def third_week_gain_sarah := second_week_gain_sarah / 3

def total_followers_susy := initial_followers_susy + first_week_gain_susy + second_week_gain_susy + third_week_gain_susy
def total_followers_sarah := initial_followers_sarah + first_week_gain_sarah + second_week_gain_sarah + third_week_gain_sarah

theorem most_followers_after_three_weeks : max total_followers_susy total_followers_sarah = 180 :=
by
  sorry

end most_followers_after_three_weeks_l210_210425


namespace find_angle_x_l210_210308

theorem find_angle_x (A B C D : Type) 
  (angleACB angleBCD : ℝ) 
  (h1 : angleACB = 90)
  (h2 : angleBCD = 40) 
  (h3 : angleACB + angleBCD + x = 180) : 
  x = 50 :=
by
  sorry

end find_angle_x_l210_210308


namespace storm_deposit_eq_120_billion_gallons_l210_210377

theorem storm_deposit_eq_120_billion_gallons :
  ∀ (initial_content : ℝ) (full_percentage_pre_storm : ℝ) (full_percentage_post_storm : ℝ) (reservoir_capacity : ℝ),
  initial_content = 220 * 10^9 → 
  full_percentage_pre_storm = 0.55 →
  full_percentage_post_storm = 0.85 →
  reservoir_capacity = initial_content / full_percentage_pre_storm →
  (full_percentage_post_storm * reservoir_capacity - initial_content) = 120 * 10^9 :=
by
  intro initial_content full_percentage_pre_storm full_percentage_post_storm reservoir_capacity
  intros h_initial_content h_pre_storm h_post_storm h_capacity
  sorry

end storm_deposit_eq_120_billion_gallons_l210_210377


namespace length_of_DE_in_triangle_l210_210237

noncomputable def triangle_length_DE (BC : ℝ) (C_deg: ℝ) (DE : ℝ) : Prop :=
  BC = 24 * Real.sqrt 2 ∧ C_deg = 45 ∧ DE = 12 * Real.sqrt 2

theorem length_of_DE_in_triangle :
  ∀ (BC : ℝ) (C_deg: ℝ) (DE : ℝ), (BC = 24 * Real.sqrt 2 ∧ C_deg = 45) → DE = 12 * Real.sqrt 2 :=
by
  intros BC C_deg DE h_cond
  have h_length := h_cond.2
  sorry

end length_of_DE_in_triangle_l210_210237


namespace cost_of_fencing_l210_210518

theorem cost_of_fencing
  (length width : ℕ)
  (ratio : 3 * width = 2 * length ∧ length * width = 5766)
  (cost_per_meter_in_paise : ℕ := 50)
  : (cost_per_meter_in_paise / 100 : ℝ) * 2 * (length + width) = 155 := 
by
  -- definitions
  sorry

end cost_of_fencing_l210_210518


namespace ticket_door_price_l210_210515

theorem ticket_door_price
  (total_attendance : ℕ)
  (tickets_before : ℕ)
  (price_before : ℚ)
  (total_receipts : ℚ)
  (tickets_bought_before : ℕ)
  (price_door : ℚ)
  (h_attendance : total_attendance = 750)
  (h_price_before : price_before = 2)
  (h_receipts : total_receipts = 1706.25)
  (h_tickets_before : tickets_bought_before = 475)
  (h_total_receipts : (tickets_bought_before * price_before) + (((total_attendance - tickets_bought_before) : ℕ) * price_door) = total_receipts) :
  price_door = 2.75 :=
by
  sorry

end ticket_door_price_l210_210515


namespace value_of_a_l210_210678

theorem value_of_a (a b : ℤ) (h : (∀ x, x^2 - x - 1 = 0 → a * x^17 + b * x^16 + 1 = 0)) : a = 987 :=
by 
  sorry

end value_of_a_l210_210678


namespace find_full_price_l210_210951

-- Defining the conditions
variables (P : ℝ) 
-- The condition that 20% of the laptop's total cost is $240.
def condition : Prop := 0.2 * P = 240

-- The proof goal is to show that the full price P is $1200 given the condition
theorem find_full_price (h : condition P) : P = 1200 :=
sorry

end find_full_price_l210_210951


namespace solve_eq1_solve_eq2_l210_210362

-- Define the theorem for the first equation
theorem solve_eq1 (x : ℝ) (h : 2 * x - 7 = 5 * x - 1) : x = -2 :=
sorry

-- Define the theorem for the second equation
theorem solve_eq2 (x : ℝ) (h : (x - 2) / 2 - (x - 1) / 6 = 1) : x = 11 / 2 :=
sorry

end solve_eq1_solve_eq2_l210_210362


namespace total_bills_combined_l210_210010

theorem total_bills_combined
  (a b c : ℝ)
  (H1 : 0.15 * a = 3)
  (H2 : 0.25 * b = 5)
  (H3 : 0.20 * c = 4) :
  a + b + c = 60 := 
sorry

end total_bills_combined_l210_210010


namespace range_of_m_l210_210318

variable (a b : ℝ)

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 1, x^3 - m ≤ a * x + b ∧ a * x + b ≤ x^3 + m) ↔ m ∈ Set.Ici (Real.sqrt 3 / 9) :=
by
  sorry

end range_of_m_l210_210318


namespace ratio_of_toys_l210_210041

theorem ratio_of_toys (total_toys : ℕ) (num_friends : ℕ) (toys_D : ℕ) 
  (h1 : total_toys = 118) 
  (h2 : num_friends = 4) 
  (h3 : toys_D = total_toys / num_friends) : 
  (toys_D / total_toys : ℚ) = 1 / 4 :=
by
  sorry

end ratio_of_toys_l210_210041


namespace slope_of_line_I_l210_210035

-- Line I intersects y = 1 at point P
def intersects_y_eq_one (I P : ℝ × ℝ → Prop) : Prop :=
∀ x y : ℝ, P (x, 1) ↔ I (x, y) ∧ y = 1

-- Line I intersects x - y - 7 = 0 at point Q
def intersects_x_minus_y_eq_seven (I Q : ℝ × ℝ → Prop) : Prop :=
∀ x y : ℝ, Q (x, y) ↔ I (x, y) ∧ x - y - 7 = 0

-- The coordinates of the midpoint of segment PQ are (1, -1)
def midpoint_eq (P Q : ℝ × ℝ) : Prop :=
∃ x1 y1 x2 y2 : ℝ,
  P = (x1, y1) ∧ Q = (x2, y2) ∧ ((x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = -1)

-- We need to show that the slope of line I is -2/3
def slope_of_I (I : ℝ × ℝ → Prop) (k : ℝ) : Prop :=
∀ x y : ℝ, I (x, y) → y + 1 = k * (x - 1)

theorem slope_of_line_I :
  ∃ I P Q : (ℝ × ℝ → Prop),
    intersects_y_eq_one I P ∧
    intersects_x_minus_y_eq_seven I Q ∧
    (∃ x1 y1 x2 y2 : ℝ, P (x1, y1) ∧ Q (x2, y2) ∧ ((x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = -1)) →
    slope_of_I I (-2/3) :=
by
  sorry

end slope_of_line_I_l210_210035


namespace sum_of_two_squares_l210_210297

theorem sum_of_two_squares (a b : ℝ) : 2 * a^2 + 2 * b^2 = (a + b)^2 + (a - b)^2 :=
by sorry

end sum_of_two_squares_l210_210297


namespace find_initial_music_files_l210_210272

-- Define the initial state before any deletion
def initial_files (music_files : ℕ) (video_files : ℕ) : ℕ := music_files + video_files

-- Define the state after deleting files
def files_after_deletion (initial_files : ℕ) (deleted_files : ℕ) : ℕ := initial_files - deleted_files

-- Theorem to prove that the initial number of music files was 13
theorem find_initial_music_files 
  (video_files : ℕ) (deleted_files : ℕ) (remaining_files : ℕ) 
  (h_videos : video_files = 30) (h_deleted : deleted_files = 10) (h_remaining : remaining_files = 33) : 
  ∃ (music_files : ℕ), initial_files music_files video_files - deleted_files = remaining_files ∧ music_files = 13 :=
by {
  sorry
}

end find_initial_music_files_l210_210272


namespace six_identities_l210_210277

theorem six_identities :
    (∀ x, (2 * x - 1) * (x - 3) = 2 * x^2 - 7 * x + 3) ∧
    (∀ x, (2 * x + 1) * (x + 3) = 2 * x^2 + 7 * x + 3) ∧
    (∀ x, (2 - x) * (1 - 3 * x) = 2 - 7 * x + 3 * x^2) ∧
    (∀ x, (2 + x) * (1 + 3 * x) = 2 + 7 * x + 3 * x^2) ∧
    (∀ x y, (2 * x - y) * (x - 3 * y) = 2 * x^2 - 7 * x * y + 3 * y^2) ∧
    (∀ x y, (2 * x + y) * (x + 3 * y) = 2 * x^2 + 7 * x * y + 3 * y^2) →
    6 = 6 :=
by
  intros
  sorry

end six_identities_l210_210277


namespace inequality_proof_l210_210871

theorem inequality_proof (a b c d : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d)
    (h_sum : a + b + c + d = 8) :
    (a^3 / (a^2 + b + c) + b^3 / (b^2 + c + d) + c^3 / (c^2 + d + a) + d^3 / (d^2 + a + b)) ≥ 4 :=
by
  sorry

end inequality_proof_l210_210871


namespace square_combinations_l210_210484

theorem square_combinations (n : ℕ) (h : n * (n - 1) = 30) : n * (n - 1) = 30 :=
by sorry

end square_combinations_l210_210484


namespace original_acid_concentration_l210_210416

theorem original_acid_concentration (P : ℝ) (h1 : 0.5 * P + 0.5 * 20 = 35) : P = 50 :=
by
  sorry

end original_acid_concentration_l210_210416


namespace kim_spends_time_on_coffee_l210_210257

noncomputable def time_per_employee_status_update : ℕ := 2
noncomputable def time_per_employee_payroll_update : ℕ := 3
noncomputable def number_of_employees : ℕ := 9
noncomputable def total_morning_routine_time : ℕ := 50

theorem kim_spends_time_on_coffee :
  ∃ C : ℕ, C + (time_per_employee_status_update * number_of_employees) + 
  (time_per_employee_payroll_update * number_of_employees) = total_morning_routine_time ∧
  C = 5 :=
by
  sorry

end kim_spends_time_on_coffee_l210_210257


namespace new_boarder_ratio_l210_210055

structure School where
  initial_boarders : ℕ
  day_students : ℕ
  boarders_ratio : ℚ

theorem new_boarder_ratio (S : School) (additional_boarders : ℕ) :
  S.initial_boarders = 60 →
  S.boarders_ratio = 2 / 5 →
  additional_boarders = 15 →
  S.day_students = (60 * 5) / 2 →
  (S.initial_boarders + additional_boarders) / S.day_students = 1 / 2 :=
by
  sorry

end new_boarder_ratio_l210_210055


namespace series_result_l210_210707

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end series_result_l210_210707


namespace sum_four_digit_even_numbers_l210_210364

-- Define the digits set
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}

-- Define the set of valid units digits for even numbers
def even_units : Finset ℕ := {0, 2, 4}

-- Define the set of all four-digit numbers using the provided digits
def four_digit_even_numbers : Finset ℕ :=
  (Finset.range (10000) \ Finset.range (1000)).filter (λ n =>
    n % 10 ∈ even_units ∧
    (n / 1000) ∈ digits ∧
    ((n / 100) % 10) ∈ digits ∧
    ((n / 10) % 10) ∈ digits)

theorem sum_four_digit_even_numbers :
  (four_digit_even_numbers.sum (λ x => x)) = 1769580 :=
  sorry

end sum_four_digit_even_numbers_l210_210364


namespace equation_has_real_solution_l210_210512

theorem equation_has_real_solution (m : ℝ) : ∃ x : ℝ, x^2 - m * x + m - 1 = 0 :=
by
  -- provide the hint that the discriminant (Δ) is (m - 2)^2
  have h : (m - 2)^2 ≥ 0 := by apply pow_two_nonneg
  sorry

end equation_has_real_solution_l210_210512


namespace max_area_triangle_ABO1_l210_210519

-- Definitions of the problem conditions
def l1 := {p : ℝ × ℝ | 2 * p.1 + 5 * p.2 = 1}

def C := {p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * p.1 + 4 * p.2 = 4}

def parallel (l1 l2 : ℝ × ℝ → Prop) := 
  ∃ m c1 c2, (∀ p, l1 p ↔ (p.2 = m * p.1 + c1)) ∧ (∀ p, l2 p ↔ (p.2 = m * p.1 + c2))

def intersects (l : ℝ × ℝ → Prop) (C: ℝ × ℝ → Prop) : Prop :=
  ∃ A B, (l A ∧ C A ∧ l B ∧ C B ∧ A ≠ B)

noncomputable def area (A B O : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs ((A.1 * (B.2 - O.2)) + (B.1 * (O.2 - A.2)) + (O.1 * (A.2 - B.2)))

-- Main statement to prove
theorem max_area_triangle_ABO1 :
  ∀ l2, parallel l1 l2 →
  intersects l2 C →
  ∃ A B, area A B (1, -2) ≤ 9 / 2 := 
sorry

end max_area_triangle_ABO1_l210_210519


namespace full_price_ticket_revenue_l210_210643

theorem full_price_ticket_revenue 
  (f h p : ℕ)
  (h1 : f + h = 160)
  (h2 : f * p + h * (p / 3) = 2400) :
  f * p = 400 := 
sorry

end full_price_ticket_revenue_l210_210643


namespace probability_of_pink_gumball_l210_210884

theorem probability_of_pink_gumball 
  (P B : ℕ) 
  (total_gumballs : P + B > 0)
  (prob_blue_blue : ((B : ℚ) / (B + P))^2 = 16 / 49) : 
  (B + P > 0) → ((P : ℚ) / (B + P) = 3 / 7) :=
by
  sorry

end probability_of_pink_gumball_l210_210884


namespace simplify_to_linear_form_l210_210853

theorem simplify_to_linear_form (p : ℤ) : 
  ((7 * p + 3) - 3 * p * 6) * 5 + (5 - 2 / 4) * (8 * p - 12) = -19 * p - 39 := 
by 
  sorry

end simplify_to_linear_form_l210_210853


namespace students_in_class_l210_210351

theorem students_in_class (y : ℕ) (H : 2 * y^2 + 6 * y + 9 = 490) : 
  y + (y + 3) = 31 := by
  sorry

end students_in_class_l210_210351


namespace decagon_diagonals_l210_210618

-- Definition of the number of diagonals in a polygon with n sides.
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- The proof problem statement
theorem decagon_diagonals : num_diagonals 10 = 35 := by
  sorry

end decagon_diagonals_l210_210618


namespace find_function_expression_point_on_function_graph_l210_210839

-- Problem setup
def y_minus_2_is_directly_proportional_to_x (y x : ℝ) : Prop :=
  ∃ k : ℝ, y - 2 = k * x

-- Conditions
def specific_condition : Prop :=
  y_minus_2_is_directly_proportional_to_x 6 1

-- Function expression derivation
theorem find_function_expression : ∃ k, ∀ x, 6 - 2 = k * 1 ∧ ∀ y, y = k * x + 2 :=
sorry

-- Given point P belongs to the function graph
theorem point_on_function_graph (a : ℝ) : (∀ x y, y = 4 * x + 2) → ∃ a, 4 * a + 2 = -1 :=
sorry

end find_function_expression_point_on_function_graph_l210_210839


namespace minimum_bag_count_l210_210971

theorem minimum_bag_count (n a b : ℕ) (h1 : 7 * a + 11 * b = 77) (h2 : a + b = n) : n = 17 :=
by
  sorry

end minimum_bag_count_l210_210971


namespace sixth_root_of_large_number_l210_210457

theorem sixth_root_of_large_number : 
  ∃ (x : ℕ), x = 51 ∧ x ^ 6 = 24414062515625 :=
by
  sorry

end sixth_root_of_large_number_l210_210457


namespace find_g_3_l210_210285

noncomputable def g (x : ℝ) : ℝ := sorry

theorem find_g_3 (h : ∀ x : ℝ, g (2 * x - 5) = 3 * x + 9) : g 3 = 21 :=
by
  sorry

end find_g_3_l210_210285


namespace problem_proof_l210_210562

def P : Set ℝ := {x | x ≤ 3}

theorem problem_proof : {-1} ⊆ P := 
sorry

end problem_proof_l210_210562


namespace mary_max_weekly_earnings_l210_210521

noncomputable def mary_weekly_earnings (max_hours : ℕ) (regular_hours : ℕ) (regular_rate : ℕ) (overtime_rate_factor : ℕ) : ℕ :=
  let overtime_hours := max_hours - regular_hours
  let overtime_rate := regular_rate + regular_rate * (overtime_rate_factor / 100)
  (regular_hours * regular_rate) + (overtime_hours * overtime_rate)

theorem mary_max_weekly_earnings : mary_weekly_earnings 60 30 12 50 = 900 :=
by
  sorry

end mary_max_weekly_earnings_l210_210521


namespace tan_half_alpha_l210_210778

theorem tan_half_alpha (α : ℝ) (h1 : 180 * (Real.pi / 180) < α) 
  (h2 : α < 270 * (Real.pi / 180)) 
  (h3 : Real.sin ((270 * (Real.pi / 180)) + α) = 4 / 5) : 
  Real.tan (α / 2) = -1 / 3 :=
by 
  -- Informal note: proof would be included here.
  sorry

end tan_half_alpha_l210_210778


namespace zain_has_80_coins_l210_210756

theorem zain_has_80_coins (emerie_quarters emerie_dimes emerie_nickels emerie_pennies emerie_half_dollars : ℕ)
  (h_quarters : emerie_quarters = 6) 
  (h_dimes : emerie_dimes = 7)
  (h_nickels : emerie_nickels = 5)
  (h_pennies : emerie_pennies = 10) 
  (h_half_dollars : emerie_half_dollars = 2) : 
  10 + emerie_quarters + 10 + emerie_dimes + 10 + emerie_nickels + 10 + emerie_pennies + 10 + emerie_half_dollars = 80 :=
by
  sorry

end zain_has_80_coins_l210_210756
