import Mathlib

namespace NUMINAMATH_GPT_age_ratio_l489_48937

theorem age_ratio (R D : ℕ) (h1 : D = 15) (h2 : R + 6 = 26) : R / D = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_age_ratio_l489_48937


namespace NUMINAMATH_GPT_total_animal_eyes_l489_48987

def frogs_in_pond := 20
def crocodiles_in_pond := 6
def eyes_per_frog := 2
def eyes_per_crocodile := 2

theorem total_animal_eyes : (frogs_in_pond * eyes_per_frog + crocodiles_in_pond * eyes_per_crocodile) = 52 := by
  sorry

end NUMINAMATH_GPT_total_animal_eyes_l489_48987


namespace NUMINAMATH_GPT_log_increasing_on_interval_l489_48934

theorem log_increasing_on_interval :
  ∀ x : ℝ, x < 1 → (0.2 : ℝ)^(x^2 - 3*x + 2) > 1 :=
by
  sorry

end NUMINAMATH_GPT_log_increasing_on_interval_l489_48934


namespace NUMINAMATH_GPT_max_marks_l489_48985

theorem max_marks (T : ℝ) (h : 0.33 * T = 165) : T = 500 := 
by {
  sorry
}

end NUMINAMATH_GPT_max_marks_l489_48985


namespace NUMINAMATH_GPT_inequality_a_b_l489_48943

theorem inequality_a_b (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
    a / (b + 1) + b / (a + 1) ≤ 1 :=
  sorry

end NUMINAMATH_GPT_inequality_a_b_l489_48943


namespace NUMINAMATH_GPT_sum_odd_even_integers_l489_48972

theorem sum_odd_even_integers :
  let odd_terms_sum := (15 / 2) * (1 + 29)
  let even_terms_sum := (10 / 2) * (2 + 20)
  odd_terms_sum + even_terms_sum = 335 :=
by
  let odd_terms_sum := (15 / 2) * (1 + 29)
  let even_terms_sum := (10 / 2) * (2 + 20)
  show odd_terms_sum + even_terms_sum = 335
  sorry

end NUMINAMATH_GPT_sum_odd_even_integers_l489_48972


namespace NUMINAMATH_GPT_faucet_draining_time_l489_48966

theorem faucet_draining_time 
  (all_faucets_drain_time : ℝ)
  (n : ℝ) 
  (first_faucet_time : ℝ) 
  (last_faucet_time : ℝ) 
  (avg_drain_time : ℝ)
  (condition_1 : all_faucets_drain_time = 24)
  (condition_2 : last_faucet_time = first_faucet_time / 7)
  (condition_3 : avg_drain_time = (first_faucet_time + last_faucet_time) / 2)
  (condition_4 : avg_drain_time = 24) : 
  first_faucet_time = 42 := 
by
  sorry

end NUMINAMATH_GPT_faucet_draining_time_l489_48966


namespace NUMINAMATH_GPT_min_value_of_a_plus_b_l489_48925

theorem min_value_of_a_plus_b (a b : ℤ) (h1 : Even a) (h2 : Even b) (h3 : a * b = 144) : a + b = -74 :=
sorry

end NUMINAMATH_GPT_min_value_of_a_plus_b_l489_48925


namespace NUMINAMATH_GPT_proof_set_intersection_l489_48990

def set_M := {x : ℝ | x^2 - 2*x - 8 ≤ 0}
def set_N := {x : ℝ | Real.log x ≥ 0}
def set_answer := {x : ℝ | 1 ≤ x ∧ x ≤ 4}

theorem proof_set_intersection : 
  (set_M ∩ set_N) = set_answer := 
by 
  sorry

end NUMINAMATH_GPT_proof_set_intersection_l489_48990


namespace NUMINAMATH_GPT_find_a_l489_48959

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ :=
  (x + a) * Real.log x

noncomputable def curve_deriv (a : ℝ) (x : ℝ) : ℝ :=
  Real.log x + (x + a) / x

theorem find_a (a : ℝ) (h : curve (x := 1) a = 2) : a = 1 :=
by
  have eq1 : curve 1 0 = (1 + a) * 0 := by sorry
  have eq2 : curve 1 1 = (1 + a) * Real.log 1 := by sorry
  have eq3 : curve_deriv a 1 = Real.log 1 + (1 + a) / 1 := by sorry
  have eq4 : 2 = 1 + a := by sorry
  sorry -- Complete proof would follow here

end NUMINAMATH_GPT_find_a_l489_48959


namespace NUMINAMATH_GPT_evaluate_g_x_plus_2_l489_48914

theorem evaluate_g_x_plus_2 (x : ℝ) (h₁ : x ≠ -3/2) (h₂ : x ≠ 2) : 
  (2 * (x + 2) + 3) / ((x + 2) - 2) = (2 * x + 7) / x :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_g_x_plus_2_l489_48914


namespace NUMINAMATH_GPT_original_radius_of_cylinder_l489_48913

theorem original_radius_of_cylinder (r y : ℝ) 
  (h₁ : 3 * π * ((r + 5)^2 - r^2) = y) 
  (h₂ : 5 * π * r^2 = y)
  (h₃ : 3 > 0) :
  r = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_original_radius_of_cylinder_l489_48913


namespace NUMINAMATH_GPT_move_decimal_point_one_place_right_l489_48975

theorem move_decimal_point_one_place_right (x : ℝ) (h : x = 76.08) : x * 10 = 760.8 :=
by
  rw [h]
  -- Here, you would provide proof steps, but we'll use sorry to indicate the proof is omitted.
  sorry

end NUMINAMATH_GPT_move_decimal_point_one_place_right_l489_48975


namespace NUMINAMATH_GPT_tan_half_angle_l489_48965

theorem tan_half_angle (α : ℝ) (h1 : Real.sin α + Real.cos α = 1 / 5)
  (h2 : 3 * π / 2 < α ∧ α < 2 * π) : 
  Real.tan (α / 2) = -1 / 3 :=
sorry

end NUMINAMATH_GPT_tan_half_angle_l489_48965


namespace NUMINAMATH_GPT_inequality_positive_real_xyz_l489_48957

theorem inequality_positive_real_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z)) + y^3 / ((1 + z) * (1 + x)) + z^3 / ((1 + x) * (1 + y))) ≥ (3 / 4) := 
by
  -- Proof is to be constructed here
  sorry

end NUMINAMATH_GPT_inequality_positive_real_xyz_l489_48957


namespace NUMINAMATH_GPT_tan_product_cos_conditions_l489_48910

variable {α β : ℝ}

theorem tan_product_cos_conditions
  (h1 : Real.cos (α + β) = 2 / 3)
  (h2 : Real.cos (α - β) = 1 / 3) :
  Real.tan α * Real.tan β = -1 / 3 :=
sorry

end NUMINAMATH_GPT_tan_product_cos_conditions_l489_48910


namespace NUMINAMATH_GPT_population_increase_l489_48962

theorem population_increase (k l m : ℝ) : 
  (1 + k/100) * (1 + l/100) * (1 + m/100) = 
  1 + (k + l + m)/100 + (k*l + k*m + l*m)/10000 + k*l*m/1000000 :=
by sorry

end NUMINAMATH_GPT_population_increase_l489_48962


namespace NUMINAMATH_GPT_shirt_cost_correct_l489_48948

-- Definitions based on the conditions
def initial_amount : ℕ := 109
def pants_cost : ℕ := 13
def remaining_amount : ℕ := 74
def total_spent : ℕ := initial_amount - remaining_amount
def shirts_cost : ℕ := total_spent - pants_cost
def number_of_shirts : ℕ := 2

-- Statement to be proved
theorem shirt_cost_correct : shirts_cost / number_of_shirts = 11 := by
  sorry

end NUMINAMATH_GPT_shirt_cost_correct_l489_48948


namespace NUMINAMATH_GPT_smallest_number_of_pets_l489_48931

noncomputable def smallest_common_multiple (a b c : Nat) : Nat :=
  Nat.lcm a (Nat.lcm b c)

theorem smallest_number_of_pets : smallest_common_multiple 3 15 9 = 45 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_pets_l489_48931


namespace NUMINAMATH_GPT_line_equation_of_point_and_slope_angle_l489_48996

theorem line_equation_of_point_and_slope_angle 
  (p : ℝ × ℝ) (θ : ℝ)
  (h₁ : p = (-1, 2))
  (h₂ : θ = 45) :
  ∃ (a b c : ℝ), a * (p.1) + b * (p.2) + c = 0 ∧ (a * 1 + b * 1 = c) :=
sorry

end NUMINAMATH_GPT_line_equation_of_point_and_slope_angle_l489_48996


namespace NUMINAMATH_GPT_bg_fg_ratio_l489_48969

open Real

-- Given the lengths AB, BD, AF, DF, BE, CF
def AB : ℝ := 15
def BD : ℝ := 18
def AF : ℝ := 15
def DF : ℝ := 12
def BE : ℝ := 24
def CF : ℝ := 17

-- Prove that the ratio BG : FG = 27 : 17
theorem bg_fg_ratio (BG FG : ℝ)
  (h_BG_FG : BG / FG = 27 / 17) :
  BG / FG = 27 / 17 := by
  sorry

end NUMINAMATH_GPT_bg_fg_ratio_l489_48969


namespace NUMINAMATH_GPT_cannot_achieve_80_cents_with_six_coins_l489_48936

theorem cannot_achieve_80_cents_with_six_coins:
  ¬ (∃ (p n d : ℕ), p + n + d = 6 ∧ p + 5 * n + 10 * d = 80) :=
by
  sorry

end NUMINAMATH_GPT_cannot_achieve_80_cents_with_six_coins_l489_48936


namespace NUMINAMATH_GPT_larger_number_of_two_l489_48912

theorem larger_number_of_two (A B : ℕ) (hcf : ℕ) (factor1 factor2 : ℕ) 
  (h_hcf : hcf = 28) (h_factors : A % hcf = 0 ∧ B % hcf = 0) 
  (h_f1 : factor1 = 12) (h_f2 : factor2 = 15)
  (h_lcm : Nat.lcm A B = hcf * factor1 * factor2)
  (h_coprime : Nat.gcd (A / hcf) (B / hcf) = 1)
  : max A B = 420 := 
sorry

end NUMINAMATH_GPT_larger_number_of_two_l489_48912


namespace NUMINAMATH_GPT_find_CD_l489_48961

noncomputable def C : ℝ := 32 / 9
noncomputable def D : ℝ := 4 / 9

theorem find_CD :
  (∀ x, x ≠ 6 ∧ x ≠ -3 → (4 * x + 8) / (x^2 - 3 * x - 18) = 
       C / (x - 6) + D / (x + 3)) →
  C = 32 / 9 ∧ D = 4 / 9 :=
by sorry

end NUMINAMATH_GPT_find_CD_l489_48961


namespace NUMINAMATH_GPT_balloon_difference_l489_48992

theorem balloon_difference 
  (your_balloons : ℕ := 7) 
  (friend_balloons : ℕ := 5) : 
  your_balloons - friend_balloons = 2 := 
by 
  sorry

end NUMINAMATH_GPT_balloon_difference_l489_48992


namespace NUMINAMATH_GPT_planting_scheme_correct_l489_48960

-- Setting up the problem as the conditions given
def types_of_seeds := ["peanuts", "Chinese cabbage", "potatoes", "corn", "wheat", "apples"]

def first_plot_seeds := ["corn", "apples"]

def planting_schemes_count : ℕ :=
  let choose_first_plot := 2  -- C(2, 1), choosing either "corn" or "apples" for the first plot
  let remaining_seeds := 5  -- 6 - 1 = 5 remaining seeds after choosing for the first plot
  let arrangements_remaining := 5 * 4 * 3  -- A(5, 3), arrangements of 3 plots from 5 remaining seeds
  choose_first_plot * arrangements_remaining

theorem planting_scheme_correct : planting_schemes_count = 120 := by
  sorry

end NUMINAMATH_GPT_planting_scheme_correct_l489_48960


namespace NUMINAMATH_GPT_possible_number_of_friends_l489_48964

-- Define the conditions and problem statement
def player_structure (total_players : ℕ) (n : ℕ) (m : ℕ) : Prop :=
  total_players = n * m ∧ (n - 1) * m = 15

-- The main theorem to prove the number of friends in the group
theorem possible_number_of_friends : ∃ (N : ℕ), 
  (player_structure N 2 15 ∨ player_structure N 4 5 ∨ player_structure N 6 3 ∨ player_structure N 16 1) ∧
  (N = 16 ∨ N = 18 ∨ N = 20 ∨ N = 30) :=
sorry

end NUMINAMATH_GPT_possible_number_of_friends_l489_48964


namespace NUMINAMATH_GPT_minimum_reciprocal_sum_l489_48968

noncomputable def log_function_a (a : ℝ) (x : ℝ) : ℝ := 
  Real.log x / Real.log a

theorem minimum_reciprocal_sum (a m n : ℝ) 
  (ha1 : 0 < a) (ha2 : a ≠ 1) 
  (hmn : 0 < m ∧ 0 < n ∧ 2 * m + n = 2) 
  (hA : log_function_a a (1 : ℝ) + -1 = -1) 
  : 1 / m + 2 / n = 4 := 
by
  sorry

end NUMINAMATH_GPT_minimum_reciprocal_sum_l489_48968


namespace NUMINAMATH_GPT_graph_passes_through_quadrants_l489_48907

theorem graph_passes_through_quadrants :
  (∃ x, x > 0 ∧ -1/2 * x + 2 > 0) ∧  -- Quadrant I condition: x > 0, y > 0
  (∃ x, x < 0 ∧ -1/2 * x + 2 > 0) ∧  -- Quadrant II condition: x < 0, y > 0
  (∃ x, x > 0 ∧ -1/2 * x + 2 < 0) := -- Quadrant IV condition: x > 0, y < 0
by
  sorry

end NUMINAMATH_GPT_graph_passes_through_quadrants_l489_48907


namespace NUMINAMATH_GPT_students_enrolled_for_german_l489_48986

-- Defining the total number of students
def class_size : Nat := 40

-- Defining the number of students enrolled for both English and German
def enrolled_both : Nat := 12

-- Defining the number of students enrolled for only English and not German
def enrolled_only_english : Nat := 18

-- Using the conditions to define the number of students who enrolled for German
theorem students_enrolled_for_german (G G_only : Nat) 
  (h_class_size : G_only + enrolled_only_english + enrolled_both = class_size) 
  (h_G : G = G_only + enrolled_both) : 
  G = 22 := 
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_students_enrolled_for_german_l489_48986


namespace NUMINAMATH_GPT_cost_of_milk_l489_48973

-- Given conditions
def total_cost_of_groceries : ℕ := 42
def cost_of_bananas : ℕ := 12
def cost_of_bread : ℕ := 9
def cost_of_apples : ℕ := 14

-- Prove that the cost of milk is $7
theorem cost_of_milk : total_cost_of_groceries - (cost_of_bananas + cost_of_bread + cost_of_apples) = 7 := 
by 
  sorry

end NUMINAMATH_GPT_cost_of_milk_l489_48973


namespace NUMINAMATH_GPT_square_field_area_l489_48946

theorem square_field_area (x : ℕ) 
    (hx : 4 * x - 2 = 666) : x^2 = 27889 := by
  -- We would solve for x using the given equation.
  sorry

end NUMINAMATH_GPT_square_field_area_l489_48946


namespace NUMINAMATH_GPT_sum_c_d_l489_48978

theorem sum_c_d (c d : ℝ) (h : ∀ x, (x - 2) * (x + 3) = x^2 + c * x + d) :
  c + d = -5 :=
sorry

end NUMINAMATH_GPT_sum_c_d_l489_48978


namespace NUMINAMATH_GPT_total_potatoes_l489_48991

theorem total_potatoes (jane_potatoes mom_potatoes dad_potatoes : Nat) 
  (h1 : jane_potatoes = 8)
  (h2 : mom_potatoes = 8)
  (h3 : dad_potatoes = 8) :
  jane_potatoes + mom_potatoes + dad_potatoes = 24 :=
by
  sorry

end NUMINAMATH_GPT_total_potatoes_l489_48991


namespace NUMINAMATH_GPT_y_coord_diff_eq_nine_l489_48951

-- Declaring the variables and conditions
variables (m n : ℝ) (p : ℝ) (h1 : p = 3)
variable (L1 : m = (n / 3) - (2 / 5))
variable (L2 : m + p = ((n + 9) / 3) - (2 / 5))

-- The theorem statement
theorem y_coord_diff_eq_nine : (n + 9) - n = 9 :=
by
  sorry

end NUMINAMATH_GPT_y_coord_diff_eq_nine_l489_48951


namespace NUMINAMATH_GPT_apples_count_l489_48952

theorem apples_count (n : ℕ) (h₁ : n > 2)
  (h₂ : 144 / n - 144 / (n + 2) = 1) :
  n + 2 = 18 :=
by
  sorry

end NUMINAMATH_GPT_apples_count_l489_48952


namespace NUMINAMATH_GPT_area_triangle_ABC_l489_48932

noncomputable def point := ℝ × ℝ

structure Parallelogram (A B C D : point) : Prop :=
(parallel_AB_CD : ∃ m1 m2, m1 ≠ m2 ∧ (A.2 - B.2) / (A.1 - B.1) = m1 ∧ (C.2 - D.2) / (C.1 - D.1) = m2)
(equal_heights : ∃ h, (B.2 - A.2 = h) ∧ (C.2 - D.2 = h))
(area_parallelogram : (B.1 - A.1) * (B.2 - A.2) + (C.1 - D.1) * (C.2 - D.2) = 27)
(thrice_length : (C.1 - D.1) = 3 * (B.1 - A.1))

theorem area_triangle_ABC (A B C D : point) (h : Parallelogram A B C D) : 
  ∃ triangle_area : ℝ, triangle_area = 13.5 :=
by
  sorry

end NUMINAMATH_GPT_area_triangle_ABC_l489_48932


namespace NUMINAMATH_GPT_men_in_first_group_l489_48938

noncomputable def first_group_men (x m b W : ℕ) : Prop :=
  let eq1 := 10 * x * m + 80 * b = W
  let eq2 := 2 * (26 * m + 48 * b) = W
  let eq3 := 4 * (15 * m + 20 * b) = W
  eq1 ∧ eq2 ∧ eq3

theorem men_in_first_group (m b W : ℕ) (h_condition : first_group_men 6 m b W) : 
  ∃ x, x = 6 :=
by
  sorry

end NUMINAMATH_GPT_men_in_first_group_l489_48938


namespace NUMINAMATH_GPT_find_2a6_minus_a4_l489_48911

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 2) = 2 * a (n + 1) - a n

theorem find_2a6_minus_a4 {a : ℕ → ℤ} 
  (h_seq : is_arithmetic_sequence a)
  (h_cond : a 1 + 3 * a 8 + a 15 = 120) : 
  2 * a 6 - a 4 = 24 :=
by
  sorry

end NUMINAMATH_GPT_find_2a6_minus_a4_l489_48911


namespace NUMINAMATH_GPT_limit_series_product_eq_l489_48954

variable (a r s : ℝ)

noncomputable def series_product_sum_limit : ℝ :=
∑' n : ℕ, (a * r^n) * (a * s^n)

theorem limit_series_product_eq :
  |r| < 1 → |s| < 1 → series_product_sum_limit a r s = a^2 / (1 - r * s) :=
by
  intro hr hs
  sorry

end NUMINAMATH_GPT_limit_series_product_eq_l489_48954


namespace NUMINAMATH_GPT_fox_appropriation_l489_48976

variable (a m : ℕ) (n : ℕ) (y x : ℕ)

-- Definitions based on conditions
def fox_funds : Prop :=
  (m-1)*a + x = m*y ∧ 2*(m-1)*a + x = (m+1)*y ∧ 
  3*(m-1)*a + x = (m+2)*y ∧ n*(m-1)*a + x = (m+n-1)*y

-- Theorems to prove the final conclusions
theorem fox_appropriation (h : fox_funds a m n y x) : 
  y = (m-1)*a ∧ x = (m-1)^2*a :=
by
  sorry

end NUMINAMATH_GPT_fox_appropriation_l489_48976


namespace NUMINAMATH_GPT_arrangements_of_45520_l489_48980

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (n : Nat) (k : Nat) : Nat :=
  factorial n / factorial k

theorem arrangements_of_45520 : 
  let n0_pos := 4
  let remaining_digits := 4 * arrangements 4 2
  n0_pos * remaining_digits = 48 :=
by
  -- Definitions and lemmas can be introduced here
  sorry

end NUMINAMATH_GPT_arrangements_of_45520_l489_48980


namespace NUMINAMATH_GPT_pat_kate_ratio_l489_48928

theorem pat_kate_ratio 
  (P K M : ℕ)
  (h1 : P + K + M = 117)
  (h2 : ∃ r : ℕ, P = r * K)
  (h3 : P = M / 3)
  (h4 : M = K + 65) : 
  P / K = 2 :=
by
  sorry

end NUMINAMATH_GPT_pat_kate_ratio_l489_48928


namespace NUMINAMATH_GPT_number_in_2019th_field_l489_48983

theorem number_in_2019th_field (f : ℕ → ℕ) (h1 : ∀ n, 0 < f n) (h2 : ∀ n, f n * f (n+1) * f (n+2) = 2018) :
  f 2018 = 1009 := sorry

end NUMINAMATH_GPT_number_in_2019th_field_l489_48983


namespace NUMINAMATH_GPT_technicians_count_l489_48904

-- Variables
variables (T R : ℕ)
-- Conditions from the problem
def avg_salary_all := 8000
def avg_salary_tech := 12000
def avg_salary_rest := 6000
def total_workers := 30
def total_salary := avg_salary_all * total_workers

-- Equations based on conditions
def eq1 : T + R = total_workers := sorry
def eq2 : avg_salary_tech * T + avg_salary_rest * R = total_salary := sorry

-- Proof statement (external conditions are reused for clarity)
theorem technicians_count : T = 10 :=
by sorry

end NUMINAMATH_GPT_technicians_count_l489_48904


namespace NUMINAMATH_GPT_sum_of_Q_and_R_in_base_8_l489_48971

theorem sum_of_Q_and_R_in_base_8 (P Q R : ℕ) (hp : 1 ≤ P ∧ P < 8) (hq : 1 ≤ Q ∧ Q < 8) (hr : 1 ≤ R ∧ R < 8) 
  (hdistinct : P ≠ Q ∧ Q ≠ R ∧ P ≠ R) (H : 8^2 * P + 8 * Q + R + (8^2 * R + 8 * Q + P) + (8^2 * Q + 8 * P + R) 
  = 8^3 * P + 8^2 * P + 8 * P) : Q + R = 7 := 
sorry

end NUMINAMATH_GPT_sum_of_Q_and_R_in_base_8_l489_48971


namespace NUMINAMATH_GPT_min_and_max_f_l489_48919

noncomputable def f (x : ℝ) : ℝ := -2 * x + 1

theorem min_and_max_f :
  (∀ x, 0 ≤ x ∧ x ≤ 5 → f x ≥ -9) ∧ (∀ x, 0 ≤ x ∧ x ≤ 5 → f x ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_min_and_max_f_l489_48919


namespace NUMINAMATH_GPT_not_all_roots_real_l489_48903

-- Define the quintic polynomial with coefficients a5, a4, a3, a2, a1, a0
def quintic_polynomial (a5 a4 a3 a2 a1 a0 : ℝ) (x : ℝ) : ℝ :=
  a5 * x^5 + a4 * x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0

-- Define a predicate for the existence of all real roots
def all_roots_real (a5 a4 a3 a2 a1 a0 : ℝ) : Prop :=
  ∀ r : ℝ, quintic_polynomial a5 a4 a3 a2 a1 a0 r = 0

-- Define the main theorem statement
theorem not_all_roots_real (a5 a4 a3 a2 a1 a0 : ℝ) :
  2 * a4^2 < 5 * a5 * a3 →
  ¬ all_roots_real a5 a4 a3 a2 a1 a0 :=
by
  sorry

end NUMINAMATH_GPT_not_all_roots_real_l489_48903


namespace NUMINAMATH_GPT_normal_price_of_article_l489_48956

theorem normal_price_of_article (P : ℝ) (sale_price : ℝ) (discount1 discount2 : ℝ) :
  discount1 = 0.10 → discount2 = 0.20 → sale_price = 108 →
  P * (1 - discount1) * (1 - discount2) = sale_price → P = 150 :=
by
  intro hd1 hd2 hsp hdiscount
  -- skipping the proof for now
  sorry

end NUMINAMATH_GPT_normal_price_of_article_l489_48956


namespace NUMINAMATH_GPT_distance_between_trees_l489_48908

theorem distance_between_trees (n : ℕ) (L : ℝ) (d : ℝ) (h1 : n = 26) (h2 : L = 700) (h3 : d = L / (n - 1)) : d = 28 :=
sorry

end NUMINAMATH_GPT_distance_between_trees_l489_48908


namespace NUMINAMATH_GPT_Jill_arrives_30_minutes_before_Jack_l489_48995

theorem Jill_arrives_30_minutes_before_Jack
  (d : ℝ) (v_J : ℝ) (v_K : ℝ)
  (h₀ : d = 3) (h₁ : v_J = 12) (h₂ : v_K = 4) :
  (d / v_K - d / v_J) * 60 = 30 :=
by
  sorry

end NUMINAMATH_GPT_Jill_arrives_30_minutes_before_Jack_l489_48995


namespace NUMINAMATH_GPT_solve_system_of_equations_l489_48935

theorem solve_system_of_equations : 
  ∃ (x y : ℝ), 
  (x / y + y / x) * (x + y) = 15 ∧ 
  (x^2 / y^2 + y^2 / x^2) * (x^2 + y^2) = 85 ∧
  ((x = 2 ∧ y = 4) ∨ (x = 4 ∧ y = 2)) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l489_48935


namespace NUMINAMATH_GPT_quadratic_rewrite_ab_value_l489_48901

theorem quadratic_rewrite_ab_value:
  ∃ a b c : ℤ, (∀ x: ℝ, 16*x^2 + 40*x + 18 = (a*x + b)^2 + c) ∧ a * b = 20 :=
by
  -- We'll add the definitions derived from conditions here
  sorry

end NUMINAMATH_GPT_quadratic_rewrite_ab_value_l489_48901


namespace NUMINAMATH_GPT_C_share_l489_48997

-- Conditions in Lean definition
def ratio_A_C (A C : ℕ) : Prop := 3 * C = 2 * A
def ratio_A_B (A B : ℕ) : Prop := 3 * B = A
def total_profit : ℕ := 60000

-- Lean statement
theorem C_share (A B C : ℕ) (h1 : ratio_A_C A C) (h2 : ratio_A_B A B) : (C * total_profit) / (A + B + C) = 20000 :=
  by
  sorry

end NUMINAMATH_GPT_C_share_l489_48997


namespace NUMINAMATH_GPT_find_x_l489_48926

theorem find_x (x y z : ℝ) (h1 : x ≠ 0) 
  (h2 : x / 3 = z + 2 * y ^ 2) 
  (h3 : x / 6 = 3 * z - y) : 
  x = 168 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l489_48926


namespace NUMINAMATH_GPT_first_train_left_time_l489_48993

-- Definitions for conditions
def speed_first_train := 45
def speed_second_train := 90
def meeting_distance := 90

-- Prove the statement
theorem first_train_left_time (T : ℝ) (time_meeting : ℝ) :
  (time_meeting - T = 2) →
  (∀ t, 0 ≤ t → t ≤ 1 → speed_first_train * t ≤ meeting_distance) →
  (∀ t, 1 ≤ t → speed_first_train * (T + t) + speed_second_train * (t - 1) = meeting_distance) →
  (time_meeting = 2 + T) :=
by
  sorry

end NUMINAMATH_GPT_first_train_left_time_l489_48993


namespace NUMINAMATH_GPT_intersection_M_N_l489_48923

-- Define set M
def set_M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}

-- Define set N
def set_N : Set ℤ := {x | ∃ k : ℕ, k > 0 ∧ x = 2 * k - 1}

-- Define the intersection of M and N
def M_intersect_N : Set ℤ := {1, 3}

-- The theorem to prove
theorem intersection_M_N : set_M ∩ set_N = M_intersect_N :=
by sorry

end NUMINAMATH_GPT_intersection_M_N_l489_48923


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l489_48947

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (a < 2) → (∃ x : ℂ, x^2 + (a : ℂ) * x + 1 = 0 ∧ x.im ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l489_48947


namespace NUMINAMATH_GPT_smallest_x_for_multiple_of_720_l489_48974

theorem smallest_x_for_multiple_of_720 (x : ℕ) (h1 : 450 = 2^1 * 3^2 * 5^2) (h2 : 720 = 2^4 * 3^2 * 5^1) : x = 8 ↔ (450 * x) % 720 = 0 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_for_multiple_of_720_l489_48974


namespace NUMINAMATH_GPT_number_of_divisors_that_are_multiples_of_2_l489_48988

-- Define the prime factorization of 540
def prime_factorization_540 : ℕ × ℕ × ℕ := (2, 3, 5)

-- Define the constraints for a divisor to be a multiple of 2
def valid_divisor_form (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 0 ≤ c ∧ c ≤ 1

noncomputable def count_divisors (prime_info : ℕ × ℕ × ℕ) : ℕ :=
  let (p1, p2, p3) := prime_info
  2 * 4 * 2 -- Correspond to choices for \( a \), \( b \), and \( c \)

theorem number_of_divisors_that_are_multiples_of_2 (p1 p2 p3 : ℕ) (h : prime_factorization_540 = (p1, p2, p3)) :
  ∃ (count : ℕ), count = 16 :=
by
  use count_divisors (2, 3, 5)
  sorry

end NUMINAMATH_GPT_number_of_divisors_that_are_multiples_of_2_l489_48988


namespace NUMINAMATH_GPT_remainder_of_division_l489_48949

def p (x : ℝ) : ℝ := 8*x^4 - 10*x^3 + 16*x^2 - 18*x + 5
def d (x : ℝ) : ℝ := 4*x - 8

theorem remainder_of_division :
  (p 2) = 81 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_division_l489_48949


namespace NUMINAMATH_GPT_train_speed_km_hr_l489_48979

def train_length : ℝ := 130  -- Length of the train in meters
def bridge_and_train_length : ℝ := 245  -- Total length of the bridge and the train in meters
def crossing_time : ℝ := 30  -- Time to cross the bridge in seconds

theorem train_speed_km_hr : (train_length + bridge_and_train_length) / crossing_time * 3.6 = 45 := by
  sorry

end NUMINAMATH_GPT_train_speed_km_hr_l489_48979


namespace NUMINAMATH_GPT_sequence_next_number_l489_48939

def next_number_in_sequence (seq : List ℕ) : ℕ :=
  if seq = [1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2] then 3 else sorry

theorem sequence_next_number :
  next_number_in_sequence [1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2] = 3 :=
by
  -- This proof is to ensure the pattern conditions are met
  sorry

end NUMINAMATH_GPT_sequence_next_number_l489_48939


namespace NUMINAMATH_GPT_expression_evaluation_l489_48924

-- Define expression variable to ensure emphasis on conditions and calculations
def expression : ℤ := 9 - (8 + 7) * 6 + 5^2 - (4 * 3) + 2 - 1

theorem expression_evaluation : expression = -67 :=
by
  -- Use assumptions about the order of operations to conclude
  sorry

end NUMINAMATH_GPT_expression_evaluation_l489_48924


namespace NUMINAMATH_GPT_ceil_eq_intervals_l489_48916

theorem ceil_eq_intervals (x : ℝ) :
  (⌈⌈ 3 * x ⌉ + 1 / 2⌉ = ⌈ x - 2 ⌉) ↔ (-1 : ℝ) ≤ x ∧ x < -2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_ceil_eq_intervals_l489_48916


namespace NUMINAMATH_GPT_radio_lowest_price_rank_l489_48944

-- Definitions based on the conditions
def total_items : ℕ := 38
def radio_highest_rank : ℕ := 16

-- The theorem statement
theorem radio_lowest_price_rank : (total_items - (radio_highest_rank - 1)) = 24 := by
  sorry

end NUMINAMATH_GPT_radio_lowest_price_rank_l489_48944


namespace NUMINAMATH_GPT_rectangle_area_exceeds_m_l489_48920

theorem rectangle_area_exceeds_m (m : ℤ) (h_m : m > 12) :
  ∃ x y : ℤ, x * y > m ∧ (x - 1) * y < m ∧ x * (y - 1) < m :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_exceeds_m_l489_48920


namespace NUMINAMATH_GPT_base_n_system_digits_l489_48989

theorem base_n_system_digits (N : ℕ) (h : N ≥ 6) :
  ((N - 1) ^ 4).digits N = [N-4, 5, N-4, 1] :=
by
  sorry

end NUMINAMATH_GPT_base_n_system_digits_l489_48989


namespace NUMINAMATH_GPT_circles_ACD_and_BCD_orthogonal_l489_48918

-- Define mathematical objects and conditions
variables (A B C D : Point) -- Points in general position on the plane
variables (circle : Point → Point → Point → Circle)

-- Circles intersect orthogonally property
def orthogonal_intersection (c1 c2 : Circle) : Prop :=
  -- Definition of orthogonal intersection of circles goes here (omitted for brevity)
  sorry

-- Given conditions
def circles_ABC_and_ABD_orthogonal : Prop :=
  orthogonal_intersection (circle A B C) (circle A B D)

-- Theorem statement
theorem circles_ACD_and_BCD_orthogonal (h : circles_ABC_and_ABD_orthogonal A B C D circle) :
  orthogonal_intersection (circle A C D) (circle B C D) :=
sorry

end NUMINAMATH_GPT_circles_ACD_and_BCD_orthogonal_l489_48918


namespace NUMINAMATH_GPT_peter_remaining_walk_time_l489_48958

-- Define the parameters and conditions
def total_distance : ℝ := 2.5
def time_per_mile : ℝ := 20
def distance_walked : ℝ := 1

-- Define the remaining distance
def remaining_distance : ℝ := total_distance - distance_walked

-- Define the remaining time Peter needs to walk
def remaining_time_to_walk (d : ℝ) (t : ℝ) : ℝ := d * t

-- State the problem we want to prove
theorem peter_remaining_walk_time :
  remaining_time_to_walk remaining_distance time_per_mile = 30 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_peter_remaining_walk_time_l489_48958


namespace NUMINAMATH_GPT_no_solutions_exist_l489_48984

theorem no_solutions_exist (m n : ℤ) : ¬(m^2 = n^2 + 1954) :=
by sorry

end NUMINAMATH_GPT_no_solutions_exist_l489_48984


namespace NUMINAMATH_GPT_find_m_l489_48902

theorem find_m (m : ℕ) (h_pos : 0 < m) 
  (h_intersection : ∃ (x y : ℤ), 13 * x + 11 * y = 700 ∧ y = m * x - 1) : 
  m = 6 :=
sorry

end NUMINAMATH_GPT_find_m_l489_48902


namespace NUMINAMATH_GPT_minimize_cost_l489_48929

noncomputable def total_cost (x : ℝ) : ℝ := (16000000 / x) + 40000 * x

theorem minimize_cost : ∃ (x : ℝ), x > 0 ∧ (∀ y > 0, total_cost x ≤ total_cost y) ∧ x = 20 := 
sorry

end NUMINAMATH_GPT_minimize_cost_l489_48929


namespace NUMINAMATH_GPT_stratified_sampling_red_balls_l489_48930

theorem stratified_sampling_red_balls (total_balls red_balls sample_size : ℕ) (h_total : total_balls = 100) (h_red : red_balls = 20) (h_sample : sample_size = 10) :
  (sample_size * (red_balls / total_balls)) = 2 := by
  sorry

end NUMINAMATH_GPT_stratified_sampling_red_balls_l489_48930


namespace NUMINAMATH_GPT_qy_length_l489_48900

theorem qy_length (Q : Type*) (C : Type*) (X Y Z : Q) (QX QZ QY : ℝ) 
  (h1 : 5 = QX)
  (h2 : QZ = 2 * (QY - QX))
  (PQ_theorem : QX * QY = QZ^2) :
  QY = 10 :=
by
  sorry

end NUMINAMATH_GPT_qy_length_l489_48900


namespace NUMINAMATH_GPT_additional_oil_needed_l489_48955

def oil_needed_each_cylinder : ℕ := 8
def number_of_cylinders : ℕ := 6
def oil_already_added : ℕ := 16

theorem additional_oil_needed : 
  (oil_needed_each_cylinder * number_of_cylinders) - oil_already_added = 32 := by
  sorry

end NUMINAMATH_GPT_additional_oil_needed_l489_48955


namespace NUMINAMATH_GPT_uncover_area_is_64_l489_48945

-- Conditions as definitions
def length_of_floor := 10
def width_of_floor := 8
def side_of_carpet := 4

-- The statement of the problem
theorem uncover_area_is_64 :
  let area_of_floor := length_of_floor * width_of_floor
  let area_of_carpet := side_of_carpet * side_of_carpet
  let uncovered_area := area_of_floor - area_of_carpet
  uncovered_area = 64 :=
by
  sorry

end NUMINAMATH_GPT_uncover_area_is_64_l489_48945


namespace NUMINAMATH_GPT_pocket_knife_value_l489_48906

noncomputable def value_of_pocket_knife (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else
    let total_rubles := n * n
    let tens (x : ℕ) := x / 10
    let units (x : ℕ) := x % 10
    let e := units n
    let d := tens n
    let remaining := total_rubles - ((total_rubles / 10) * 10)
    if remaining = 6 then 4 else sorry

theorem pocket_knife_value (n : ℕ) : value_of_pocket_knife n = 2 := by
  sorry

end NUMINAMATH_GPT_pocket_knife_value_l489_48906


namespace NUMINAMATH_GPT_find_numbers_l489_48963

theorem find_numbers (n : ℕ) (h1 : n ≥ 2) (a : ℕ) (ha : a ≠ 1) (ha_min : ∀ d, d ∣ n → d ≠ 1 → a ≤ d) (b : ℕ) (hb : b ∣ n) :
  n = a^2 + b^2 ↔ n = 8 ∨ n = 20 :=
by sorry

end NUMINAMATH_GPT_find_numbers_l489_48963


namespace NUMINAMATH_GPT_average_earning_week_l489_48953

theorem average_earning_week (D1 D2 D3 D4 D5 D6 D7 : ℝ)
  (h1 : (D1 + D2 + D3 + D4) / 4 = 25)
  (h2 : (D4 + D5 + D6 + D7) / 4 = 22)
  (h3 : D4 = 20) : 
  (D1 + D2 + D3 + D4 + D5 + D6 + D7) / 7 = 24 :=
by
  sorry

end NUMINAMATH_GPT_average_earning_week_l489_48953


namespace NUMINAMATH_GPT_convex_polygon_triangles_impossible_l489_48933

theorem convex_polygon_triangles_impossible :
  ∀ (a b c : ℕ), 2016 + 2 * b + c - 2014 = 0 → a + b + c = 2014 → a = 1007 → false :=
sorry

end NUMINAMATH_GPT_convex_polygon_triangles_impossible_l489_48933


namespace NUMINAMATH_GPT_ann_boxes_less_than_n_l489_48981

-- Define the total number of boxes n
def n : ℕ := 12

-- Define the number of boxes Mark sold
def mark_sold : ℕ := n - 11

-- Define a condition on the number of boxes Ann sold
def ann_sold (A : ℕ) : Prop := 1 ≤ A ∧ A < n - mark_sold

-- The statement to prove
theorem ann_boxes_less_than_n : ∃ A : ℕ, ann_sold A ∧ n - A = 2 :=
by
  sorry

end NUMINAMATH_GPT_ann_boxes_less_than_n_l489_48981


namespace NUMINAMATH_GPT_find_h_l489_48905

theorem find_h (h : ℝ) : (∀ x : ℝ, x^2 - 4 * h * x = 8) 
    ∧ (∀ r s : ℝ, r + s = 4 * h ∧ r * s = -8 → r^2 + s^2 = 18) 
    → h = (Real.sqrt 2) / 4 ∨ h = -(Real.sqrt 2) / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_h_l489_48905


namespace NUMINAMATH_GPT_value_of_a_l489_48982

theorem value_of_a (a : ℝ) : 
  ({2, 3} : Set ℝ) ⊆ ({1, 2, a} : Set ℝ) → a = 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l489_48982


namespace NUMINAMATH_GPT_cuboid_total_edge_length_cuboid_surface_area_l489_48940

variables (a b c : ℝ)

theorem cuboid_total_edge_length : 4 * (a + b + c) = 4 * (a + b + c) := 
by
  sorry

theorem cuboid_surface_area : 2 * (a * b + b * c + a * c) = 2 * (a * b + b * c + a * c) := 
by
  sorry

end NUMINAMATH_GPT_cuboid_total_edge_length_cuboid_surface_area_l489_48940


namespace NUMINAMATH_GPT_polynomial_n_values_possible_num_values_of_n_l489_48941

theorem polynomial_n_values_possible :
  ∃ (n : ℤ), 
    (∀ (x : ℝ), x^3 - 4050 * x^2 + (m : ℝ) * x + (n : ℝ) = 0 → x > 0) ∧
    (∃ a : ℤ, a > 0 ∧ ∀ (x : ℝ), x^3 - 4050 * x^2 + (m : ℝ) * x + (n : ℝ) = 0 → 
      x = a ∨ x = a / 4 + r ∨ x = a / 4 - r) ∧
    1 ≤ r^2 ∧ r^2 ≤ 4090499 :=
sorry

theorem num_values_of_n : 
  ∃ (n_values : ℤ), n_values = 4088474 :=
sorry

end NUMINAMATH_GPT_polynomial_n_values_possible_num_values_of_n_l489_48941


namespace NUMINAMATH_GPT_general_formula_arithmetic_sequence_sum_of_sequence_b_l489_48921

-- Definitions of arithmetic sequence {a_n} and geometric sequence conditions
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
 ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
 ∀ n : ℕ, S n = n * (a 1 + a n) / 2

def geometric_sequence (a : ℕ → ℤ) : Prop :=
  a 3 ^ 2 = a 1 * a 7

def arithmetic_sum_S3 (S : ℕ → ℤ) : Prop :=
  S 3 = 9

def general_formula (a : ℕ → ℤ) : Prop :=
 ∀ n : ℕ, a n = n + 1

def sum_first_n_terms_b (b : ℕ → ℤ) (T : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, T n = (n-1) * 2^(n+1) + 2

-- The Lean theorem statements
theorem general_formula_arithmetic_sequence
  (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ)
  (h1 : arithmetic_sequence a d)
  (h2 : sum_first_n_terms a S)
  (h3 : geometric_sequence a)
  (h4 : arithmetic_sum_S3 S) :
  general_formula a :=
  sorry

theorem sum_of_sequence_b
  (a b : ℕ → ℤ) (T : ℕ → ℤ)
  (h1 : general_formula a)
  (h2 : ∀ n : ℕ, b n = (a n - 1) * 2^n)
  (h3 : sum_first_n_terms_b b T) :
  ∀ n : ℕ, T n = (n-1) * 2^(n+1) + 2 :=
  sorry

end NUMINAMATH_GPT_general_formula_arithmetic_sequence_sum_of_sequence_b_l489_48921


namespace NUMINAMATH_GPT_domain_of_f_l489_48970

open Real

noncomputable def f (x : ℝ) : ℝ := (log (2 * x - x^2)) / (x - 1)

theorem domain_of_f (x : ℝ) : (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 2) ↔ (2 * x - x^2 > 0 ∧ x ≠ 1) := by
  sorry

end NUMINAMATH_GPT_domain_of_f_l489_48970


namespace NUMINAMATH_GPT_calculate_permutation_sum_l489_48909

noncomputable def A (n k : ℕ) : ℕ := n.factorial / (n - k).factorial

theorem calculate_permutation_sum (n : ℕ) (h1 : 3 ≤ n) (h2 : n ≤ 3) :
  A (2 * n) (n + 3) + A 4 (n + 1) = 744 := by
  sorry

end NUMINAMATH_GPT_calculate_permutation_sum_l489_48909


namespace NUMINAMATH_GPT_cube_volume_is_27_l489_48927

theorem cube_volume_is_27 
    (a : ℕ) 
    (Vol_cube : ℕ := a^3)
    (Vol_new : ℕ := (a - 2) * a * (a + 2))
    (h : Vol_new + 12 = Vol_cube) : Vol_cube = 27 :=
by
    sorry

end NUMINAMATH_GPT_cube_volume_is_27_l489_48927


namespace NUMINAMATH_GPT_female_officers_on_police_force_l489_48977

theorem female_officers_on_police_force
  (percent_on_duty : ℝ)
  (total_on_duty : ℕ)
  (half_female_on_duty : ℕ)
  (h1 : percent_on_duty = 0.16)
  (h2 : total_on_duty = 160)
  (h3 : half_female_on_duty = total_on_duty / 2)
  (h4 : half_female_on_duty = 80)
  :
  ∃ (total_female_officers : ℕ), total_female_officers = 500 :=
by
  sorry

end NUMINAMATH_GPT_female_officers_on_police_force_l489_48977


namespace NUMINAMATH_GPT_latus_rectum_of_parabola_l489_48967

theorem latus_rectum_of_parabola (p : ℝ) (hp : 0 < p) (A : ℝ × ℝ) (hA : A = (1, 1/2)) :
  ∃ a : ℝ, y^2 = 4 * a * x → A.2 ^ 2 = 4 * a * A.1 → x = -1 / (4 * a) → x = -1 / 16 :=
by
  sorry

end NUMINAMATH_GPT_latus_rectum_of_parabola_l489_48967


namespace NUMINAMATH_GPT_fraction_of_x_l489_48999

theorem fraction_of_x (w x y f : ℝ) (h1 : 2 / w + f * x = 2 / y) (h2 : w * x = y) (h3 : (w + x) / 2 = 0.5) : f = 2 / x - 2 := 
sorry

end NUMINAMATH_GPT_fraction_of_x_l489_48999


namespace NUMINAMATH_GPT_max_single_player_salary_l489_48950

variable (n : ℕ) (m : ℕ) (p : ℕ) (s : ℕ)

theorem max_single_player_salary
  (h1 : n = 18)
  (h2 : ∀ i : ℕ, i < n → p ≥ 20000)
  (h3 : s = 800000)
  (h4 : n * 20000 ≤ s) :
  ∃ x : ℕ, x = 460000 :=
by
  sorry

end NUMINAMATH_GPT_max_single_player_salary_l489_48950


namespace NUMINAMATH_GPT_height_of_parallelogram_l489_48917

theorem height_of_parallelogram (Area Base : ℝ) (h1 : Area = 180) (h2 : Base = 18) : Area / Base = 10 :=
by
  sorry

end NUMINAMATH_GPT_height_of_parallelogram_l489_48917


namespace NUMINAMATH_GPT_dr_jones_remaining_salary_l489_48942

theorem dr_jones_remaining_salary:
  let salary := 6000
  let house_rental := 640
  let food_expense := 380
  let electric_water_bill := (1/4) * salary
  let insurances := (1/5) * salary
  let taxes := (10/100) * salary
  let transportation := (3/100) * salary
  let emergency_costs := (2/100) * salary
  let total_expenses := house_rental + food_expense + electric_water_bill + insurances + taxes + transportation + emergency_costs
  let remaining_salary := salary - total_expenses
  remaining_salary = 1380 :=
by
  sorry

end NUMINAMATH_GPT_dr_jones_remaining_salary_l489_48942


namespace NUMINAMATH_GPT_range_of_a_l489_48922

variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Condition: f(x) is an increasing function on ℝ.
def is_increasing_on_ℝ (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x < f y

-- Equivalent proof problem in Lean 4:
theorem range_of_a (h : is_increasing_on_ℝ f) : 1 < a ∧ a < 6 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l489_48922


namespace NUMINAMATH_GPT_expression_equals_answer_l489_48994

noncomputable def verify_expression : ℚ :=
  15 * (1 / 17) * 34 - (1 / 2)

theorem expression_equals_answer :
  verify_expression = 59 / 2 :=
by
  sorry

end NUMINAMATH_GPT_expression_equals_answer_l489_48994


namespace NUMINAMATH_GPT_pie_slices_l489_48998

theorem pie_slices (total_pies : ℕ) (sold_pies : ℕ) (gifted_pies : ℕ) (left_pieces : ℕ) (eaten_fraction : ℚ) :
  total_pies = 4 →
  sold_pies = 1 →
  gifted_pies = 1 →
  eaten_fraction = 2/3 →
  left_pieces = 4 →
  (total_pies - sold_pies - gifted_pies) * (left_pieces * 3 / (1 - eaten_fraction)) / (total_pies - sold_pies - gifted_pies) = 6 :=
by
  sorry

end NUMINAMATH_GPT_pie_slices_l489_48998


namespace NUMINAMATH_GPT_greatest_third_side_l489_48915

theorem greatest_third_side
  (a b : ℕ)
  (h₁ : a = 7)
  (h₂ : b = 10)
  (c : ℕ)
  (h₃ : a + b + c ≤ 30)
  (h₄ : 3 < c)
  (h₅ : c ≤ 13) :
  c = 13 := 
sorry

end NUMINAMATH_GPT_greatest_third_side_l489_48915
