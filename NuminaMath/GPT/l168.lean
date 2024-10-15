import Mathlib

namespace NUMINAMATH_GPT_sine_curve_transformation_l168_16814

theorem sine_curve_transformation (x y x' y' : ℝ) 
  (h1 : x' = (1 / 2) * x) 
  (h2 : y' = 3 * y) :
  (y = Real.sin x) ↔ (y' = 3 * Real.sin (2 * x')) := by 
  sorry

end NUMINAMATH_GPT_sine_curve_transformation_l168_16814


namespace NUMINAMATH_GPT_tan_alpha_value_complicated_expression_value_l168_16800

theorem tan_alpha_value (α : ℝ) (h1 : Real.sin α = -2 * Real.sqrt 5 / 5) (h2 : Real.tan α < 0) : 
  Real.tan α = -2 := by 
  sorry

theorem complicated_expression_value (α : ℝ) (h1 : Real.sin α = -2 * Real.sqrt 5 / 5) (h2 : Real.tan α < 0) (h3 : Real.tan α = -2) :
  (2 * Real.sin (α + Real.pi) + Real.cos (2 * Real.pi - α)) / 
  (Real.cos (α - Real.pi / 2) - Real.sin (2 * Real.pi / 2 + α)) = -5 := by 
  sorry

end NUMINAMATH_GPT_tan_alpha_value_complicated_expression_value_l168_16800


namespace NUMINAMATH_GPT_range_of_a_l168_16824

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x → 2 * x + 1 / x - a > 0) → a < 2 * Real.sqrt 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l168_16824


namespace NUMINAMATH_GPT_twice_shorter_vs_longer_l168_16892

-- Definitions and conditions
def total_length : ℝ := 20
def shorter_length : ℝ := 8
def longer_length : ℝ := total_length - shorter_length

-- Statement to prove
theorem twice_shorter_vs_longer :
  2 * shorter_length - longer_length = 4 :=
by
  sorry

end NUMINAMATH_GPT_twice_shorter_vs_longer_l168_16892


namespace NUMINAMATH_GPT_inequality_correct_l168_16840

open BigOperators

theorem inequality_correct {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  (a^2 + b^2) / 2 ≥ (a + b)^2 / 4 ∧ (a + b)^2 / 4 ≥ a * b :=
by 
  sorry

end NUMINAMATH_GPT_inequality_correct_l168_16840


namespace NUMINAMATH_GPT_problem1_problem2_l168_16845

variable (α : ℝ) (tan_alpha_eq_three : Real.tan α = 3)

theorem problem1 : (4 * Real.sin α - Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = 11 / 14 :=
by sorry

theorem problem2 : Real.sin α * Real.cos α = 3 / 10 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l168_16845


namespace NUMINAMATH_GPT_probability_all_same_color_is_correct_l168_16893

-- Definitions of quantities
def yellow_marbles := 3
def green_marbles := 7
def purple_marbles := 5
def total_marbles := yellow_marbles + green_marbles + purple_marbles

-- Calculation of drawing 4 marbles all the same color
def probability_all_yellow : ℚ := (yellow_marbles / total_marbles) * ((yellow_marbles - 1) / (total_marbles - 1)) * ((yellow_marbles - 2) / (total_marbles - 2)) * ((yellow_marbles - 3) / (total_marbles - 3))
def probability_all_green : ℚ := (green_marbles / total_marbles) * ((green_marbles - 1) / (total_marbles - 1)) * ((green_marbles - 2) / (total_marbles - 2)) * ((green_marbles - 3) / (total_marbles - 3))
def probability_all_purple : ℚ := (purple_marbles / total_marbles) * ((purple_marbles - 1) / (total_marbles - 1)) * ((purple_marbles - 2) / (total_marbles - 2)) * ((purple_marbles - 3) / (total_marbles - 3))

-- Total probability of drawing 4 marbles all the same color
def total_probability_same_color : ℚ := probability_all_yellow + probability_all_green + probability_all_purple

-- Theorem statement
theorem probability_all_same_color_is_correct : total_probability_same_color = 532 / 4095 :=
by
  sorry

end NUMINAMATH_GPT_probability_all_same_color_is_correct_l168_16893


namespace NUMINAMATH_GPT_rectangle_perimeter_inequality_l168_16891

-- Define rectilinear perimeters
def perimeter (length : ℝ) (width : ℝ) : ℝ := 2 * (length + width)

-- Definitions for rectangles contained within each other
def rectangle_contained (len1 wid1 len2 wid2 : ℝ) : Prop :=
  len1 ≤ len2 ∧ wid1 ≤ wid2

-- Statement of the problem
theorem rectangle_perimeter_inequality (l1 w1 l2 w2 : ℝ) (h : rectangle_contained l1 w1 l2 w2) :
  perimeter l1 w1 ≤ perimeter l2 w2 :=
sorry

end NUMINAMATH_GPT_rectangle_perimeter_inequality_l168_16891


namespace NUMINAMATH_GPT_value_of_polynomial_l168_16841

theorem value_of_polynomial : 
  99^5 - 5 * 99^4 + 10 * 99^3 - 10 * 99^2 + 5 * 99 - 1 = 98^5 := by
  sorry

end NUMINAMATH_GPT_value_of_polynomial_l168_16841


namespace NUMINAMATH_GPT_factorize_a_squared_minus_25_factorize_2x_squared_y_minus_8xy_plus_8y_l168_16856

-- Math Proof Problem 1
theorem factorize_a_squared_minus_25 (a : ℝ) : a^2 - 25 = (a + 5) * (a - 5) :=
by
  sorry

-- Math Proof Problem 2
theorem factorize_2x_squared_y_minus_8xy_plus_8y (x y : ℝ) : 2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_a_squared_minus_25_factorize_2x_squared_y_minus_8xy_plus_8y_l168_16856


namespace NUMINAMATH_GPT_painted_pictures_in_june_l168_16825

theorem painted_pictures_in_june (J : ℕ) (h1 : J + (J + 2) + 9 = 13) : J = 1 :=
by
  -- Given condition translates to J + J + 2 + 9 = 13
  -- Simplification yields 2J + 11 = 13
  -- Solving 2J + 11 = 13 gives J = 1
  sorry

end NUMINAMATH_GPT_painted_pictures_in_june_l168_16825


namespace NUMINAMATH_GPT_find_plot_width_l168_16858

theorem find_plot_width:
  let length : ℝ := 360
  let area_acres : ℝ := 10
  let square_feet_per_acre : ℝ := 43560
  let area_square_feet := area_acres * square_feet_per_acre
  let width := area_square_feet / length
  area_square_feet = 435600 ∧ length = 360 ∧ square_feet_per_acre = 43560
  → width = 1210 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_plot_width_l168_16858


namespace NUMINAMATH_GPT_sum_of_series_l168_16846

theorem sum_of_series :
  (3 + 13 + 23 + 33 + 43) + (11 + 21 + 31 + 41 + 51) = 270 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_series_l168_16846


namespace NUMINAMATH_GPT_percentage_profit_first_bicycle_l168_16831

theorem percentage_profit_first_bicycle :
  ∃ (C1 C2 : ℝ), 
    (C1 + C2 = 1980) ∧ 
    (0.9 * C2 = 990) ∧ 
    (12.5 / 100 * C1 = (990 - C1) / C1 * 100) :=
by
  sorry

end NUMINAMATH_GPT_percentage_profit_first_bicycle_l168_16831


namespace NUMINAMATH_GPT_inequality_order_l168_16843

theorem inequality_order (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0)
  (h : (a^2 / (b^2 + c^2)) < (b^2 / (c^2 + a^2)) ∧ (b^2 / (c^2 + a^2)) < (c^2 / (a^2 + b^2))) :
  |a| < |b| ∧ |b| < |c| := 
sorry

end NUMINAMATH_GPT_inequality_order_l168_16843


namespace NUMINAMATH_GPT_smallest_x_l168_16883

theorem smallest_x 
  (x : ℝ)
  (h : ( ( (5 * x - 20) / (4 * x - 5) ) ^ 2 + ( (5 * x - 20) / (4 * x - 5) ) ) = 6 ) :
  x = -10 / 3 := sorry

end NUMINAMATH_GPT_smallest_x_l168_16883


namespace NUMINAMATH_GPT_digit_A_unique_solution_l168_16859

theorem digit_A_unique_solution :
  ∃ (A : ℕ), 0 ≤ A ∧ A < 10 ∧ (100 * A + 72 - 23 = 549) ∧ A = 5 :=
by
  sorry

end NUMINAMATH_GPT_digit_A_unique_solution_l168_16859


namespace NUMINAMATH_GPT_solve_x_l168_16842

theorem solve_x (x : ℝ) (h : x ≠ 0) (h_eq : (5 * x) ^ 10 = (10 * x) ^ 5) : x = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_x_l168_16842


namespace NUMINAMATH_GPT_james_total_distance_l168_16898

structure Segment where
  speed : ℝ -- speed in mph
  time : ℝ -- time in hours

def totalDistance (segments : List Segment) : ℝ :=
  segments.foldr (λ seg acc => seg.speed * seg.time + acc) 0

theorem james_total_distance :
  let segments := [
    Segment.mk 30 0.5,
    Segment.mk 60 0.75,
    Segment.mk 75 1.5,
    Segment.mk 60 2
  ]
  totalDistance segments = 292.5 :=
by
  sorry

end NUMINAMATH_GPT_james_total_distance_l168_16898


namespace NUMINAMATH_GPT_minimum_value_inequality_l168_16834

open Real

theorem minimum_value_inequality
  (a b c : ℝ)
  (ha : 2 ≤ a) 
  (hb : a ≤ b)
  (hc : b ≤ c)
  (hd : c ≤ 5) :
  (a - 2)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (5 / c - 1)^2 = 4 * (sqrt 5 ^ (1 / 4) - 1)^2 :=
sorry

end NUMINAMATH_GPT_minimum_value_inequality_l168_16834


namespace NUMINAMATH_GPT_bottle_caps_per_child_l168_16817

-- Define the conditions
def num_children : ℕ := 9
def total_bottle_caps : ℕ := 45

-- State the theorem that needs to be proved: each child has 5 bottle caps
theorem bottle_caps_per_child : (total_bottle_caps / num_children) = 5 := by
  sorry

end NUMINAMATH_GPT_bottle_caps_per_child_l168_16817


namespace NUMINAMATH_GPT_sequence_sum_zero_l168_16865

theorem sequence_sum_zero (n : ℕ) (h : n > 1) :
  (∃ (a : ℕ → ℤ), (∀ k : ℕ, k > 0 → a k ≠ 0) ∧ (∀ k : ℕ, k > 0 → a k + 2 * a (2 * k) + n * a (n * k) = 0)) ↔ n ≥ 3 := 
by sorry

end NUMINAMATH_GPT_sequence_sum_zero_l168_16865


namespace NUMINAMATH_GPT_travel_time_l168_16863

theorem travel_time (v : ℝ) (d : ℝ) (t : ℝ) (hv : v = 65) (hd : d = 195) : t = 3 :=
by
  sorry

end NUMINAMATH_GPT_travel_time_l168_16863


namespace NUMINAMATH_GPT_find_a_l168_16880

variable (a : ℝ)

def augmented_matrix (a : ℝ) :=
  ([1, -1, -3], [a, 3, 4])

def solution := (-1, 2)

theorem find_a (hx : -1 - 2 = -3)
               (hy : a * (-1) + 3 * 2 = 4) :
               a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l168_16880


namespace NUMINAMATH_GPT_only_n_divides_2_pow_n_minus_1_l168_16862

theorem only_n_divides_2_pow_n_minus_1 : ∀ (n : ℕ), n > 0 ∧ n ∣ (2^n - 1) ↔ n = 1 := by
  sorry

end NUMINAMATH_GPT_only_n_divides_2_pow_n_minus_1_l168_16862


namespace NUMINAMATH_GPT_max_rabbits_l168_16870

theorem max_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : 3 ≤ N) (long_ears : {n // n ≤ N}) (jump_far : {n // n ≤ N}) 
  (h_long_ears : (long_ears.val = 13)) (h_jump_far : (jump_far.val = 17)) (h_both : (long_ears.val + jump_far.val - N ≥ 3)) : 
  N = 27 :=
by
  sorry

end NUMINAMATH_GPT_max_rabbits_l168_16870


namespace NUMINAMATH_GPT_employee_n_salary_l168_16830

theorem employee_n_salary (m n : ℝ) (h1: m + n = 594) (h2: m = 1.2 * n) : n = 270 := by
  sorry

end NUMINAMATH_GPT_employee_n_salary_l168_16830


namespace NUMINAMATH_GPT_charlene_initial_necklaces_l168_16887

-- Definitions for the conditions.
def necklaces_sold : ℕ := 16
def necklaces_giveaway : ℕ := 18
def necklaces_left : ℕ := 26

-- Statement to prove that the initial number of necklaces is 60.
theorem charlene_initial_necklaces : necklaces_sold + necklaces_giveaway + necklaces_left = 60 := by
  sorry

end NUMINAMATH_GPT_charlene_initial_necklaces_l168_16887


namespace NUMINAMATH_GPT_exist_three_integers_l168_16852

theorem exist_three_integers :
  ∃ (a b c : ℤ), a * b - c = 2018 ∧ b * c - a = 2018 ∧ c * a - b = 2018 := 
sorry

end NUMINAMATH_GPT_exist_three_integers_l168_16852


namespace NUMINAMATH_GPT_profits_ratio_l168_16810

-- Definitions
def investment_ratio (p q : ℕ) := 7 * p = 5 * q
def investment_period_p := 10
def investment_period_q := 20

-- Prove the ratio of profits
theorem profits_ratio (p q : ℕ) (h1 : investment_ratio p q) :
  (7 * p * investment_period_p / (5 * q * investment_period_q)) = 7 / 10 :=
sorry

end NUMINAMATH_GPT_profits_ratio_l168_16810


namespace NUMINAMATH_GPT_find_x_value_l168_16874

theorem find_x_value
  (y₁ y₂ z₁ z₂ x₁ x w k : ℝ)
  (h₁ : y₁ = 3) (h₂ : z₁ = 2) (h₃ : x₁ = 1)
  (h₄ : y₂ = 6) (h₅ : z₂ = 5)
  (inv_rel : ∀ y z k, x = k * (z / y^2))
  (const_prod : ∀ x w, x * w = 1) :
  x = 5 / 8 :=
by
  -- omitted proof steps
  sorry

end NUMINAMATH_GPT_find_x_value_l168_16874


namespace NUMINAMATH_GPT_f_of_3_eq_11_l168_16873

theorem f_of_3_eq_11 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x - 1 / x) = x^2 + 1 / x^2) : f 3 = 11 :=
by
  sorry

end NUMINAMATH_GPT_f_of_3_eq_11_l168_16873


namespace NUMINAMATH_GPT_problem_statement_l168_16867

variables {x y P Q : ℝ}

theorem problem_statement (h1 : x^2 + y^2 = (x + y)^2 + P) (h2 : x^2 + y^2 = (x - y)^2 + Q) : P = -2 * x * y ∧ Q = 2 * x * y :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l168_16867


namespace NUMINAMATH_GPT_escalator_steps_l168_16878

theorem escalator_steps
  (x : ℕ)
  (time_me : ℕ := 60)
  (steps_me : ℕ := 20)
  (time_wife : ℕ := 72)
  (steps_wife : ℕ := 16)
  (escalator_speed_me : x - steps_me = 60 * (x - 20) / 72)
  (escalator_speed_wife : x - steps_wife = 72 * (x - 16) / 60) :
  x = 40 := by
  sorry

end NUMINAMATH_GPT_escalator_steps_l168_16878


namespace NUMINAMATH_GPT_line_equation_l168_16894

theorem line_equation (p : ℝ × ℝ) (a : ℝ × ℝ) :
  p = (4, -4) →
  a = (1, 2 / 7) →
  ∃ (m b : ℝ), m = 2 / 7 ∧ b = -36 / 7 ∧ ∀ x y : ℝ, y = m * x + b :=
by
  intros hp ha
  sorry

end NUMINAMATH_GPT_line_equation_l168_16894


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l168_16848

theorem quadratic_no_real_roots (k : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - 2*x - k = 0)) ↔ k < -1 :=
by sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_l168_16848


namespace NUMINAMATH_GPT_remove_five_magazines_l168_16868

theorem remove_five_magazines (magazines : Fin 10 → Set α) 
  (coffee_table : Set α) 
  (h_cover : (⋃ i, magazines i) = coffee_table) :
  ∃ ( S : Set α), S ⊆ coffee_table ∧ (∃ (removed : Finset (Fin 10)), removed.card = 5 ∧ 
    coffee_table \ (⋃ i ∈ removed, magazines i) ⊆ S ∧ (S = coffee_table \ (⋃ i ∈ removed, magazines i) ) ∧ 
    (⋃ i ∉ removed, magazines i) ∩ S = ∅) := 
sorry

end NUMINAMATH_GPT_remove_five_magazines_l168_16868


namespace NUMINAMATH_GPT_bridge_length_increase_l168_16896

open Real

def elevation_change : ℝ := 800
def original_gradient : ℝ := 0.02
def new_gradient : ℝ := 0.015

theorem bridge_length_increase :
  let original_length := elevation_change / original_gradient
  let new_length := elevation_change / new_gradient
  new_length - original_length = 13333 := by
  sorry

end NUMINAMATH_GPT_bridge_length_increase_l168_16896


namespace NUMINAMATH_GPT_admission_price_for_children_l168_16857

theorem admission_price_for_children (people_at_play : ℕ) (admission_price_adult : ℕ) (total_receipts : ℕ) (adults_attended : ℕ) 
  (h1 : people_at_play = 610) (h2 : admission_price_adult = 2) (h3 : total_receipts = 960) (h4 : adults_attended = 350) : 
  ∃ (admission_price_child : ℕ), admission_price_child = 1 :=
by
  sorry

end NUMINAMATH_GPT_admission_price_for_children_l168_16857


namespace NUMINAMATH_GPT_kelseys_sister_is_3_years_older_l168_16808

-- Define the necessary conditions
def kelsey_birth_year : ℕ := 1999 - 25
def sister_birth_year : ℕ := 2021 - 50
def age_difference (a b : ℕ) : ℕ := a - b

-- State the theorem to prove
theorem kelseys_sister_is_3_years_older :
  age_difference kelsey_birth_year sister_birth_year = 3 :=
by
  -- Skipping the proof steps as only the statement is needed
  sorry

end NUMINAMATH_GPT_kelseys_sister_is_3_years_older_l168_16808


namespace NUMINAMATH_GPT_cylinder_in_cone_l168_16839

noncomputable def cylinder_radius : ℝ :=
  let cone_radius : ℝ := 4
  let cone_height : ℝ := 10
  let r : ℝ := (10 * 2) / 9  -- based on the derived form of r calculation
  r

theorem cylinder_in_cone :
  let cone_radius : ℝ := 4
  let cone_height : ℝ := 10
  let r : ℝ := cylinder_radius
  (r = 20 / 9) :=
by
  sorry -- Proof mechanism is skipped as per instructions.

end NUMINAMATH_GPT_cylinder_in_cone_l168_16839


namespace NUMINAMATH_GPT_max_k_value_l168_16809

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x

theorem max_k_value
    (h : ∀ (x : ℝ), 1 < x → f x > k * (x - 1)) :
    k = 3 := sorry

end NUMINAMATH_GPT_max_k_value_l168_16809


namespace NUMINAMATH_GPT_fernanda_savings_before_payments_l168_16815

open Real

theorem fernanda_savings_before_payments (aryan_debt kyro_debt aryan_payment kyro_payment total_savings before_savings : ℝ) 
  (h1: aryan_debt = 1200)
  (h2: aryan_debt = 2 * kyro_debt)
  (h3: aryan_payment = 0.6 * aryan_debt)
  (h4: kyro_payment = 0.8 * kyro_debt)
  (h5: total_savings = before_savings + aryan_payment + kyro_payment)
  (h6: total_savings = 1500) :
  before_savings = 300 :=
by
  sorry

end NUMINAMATH_GPT_fernanda_savings_before_payments_l168_16815


namespace NUMINAMATH_GPT_problem_statement_l168_16844

theorem problem_statement
  (a b c : ℝ) 
  (X : ℝ) 
  (hX : X = a + b + c + 2 * Real.sqrt (a^2 + b^2 + c^2 - a * b - b * c - c * a)) :
  X ≥ max (max (3 * a) (3 * b)) (3 * c) ∧ 
  ∃ (u v w : ℝ), 
    (u = Real.sqrt (X - 3 * a) ∧ v = Real.sqrt (X - 3 * b) ∧ w = Real.sqrt (X - 3 * c) ∧ 
     ((u + v = w) ∨ (v + w = u) ∨ (w + u = v))) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l168_16844


namespace NUMINAMATH_GPT_A_inter_B_l168_16837

-- Define the sets A and B
def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := { abs x | x ∈ A }

-- Statement of the theorem to be proven
theorem A_inter_B :
  A ∩ B = {0, 2} := 
by 
  sorry

end NUMINAMATH_GPT_A_inter_B_l168_16837


namespace NUMINAMATH_GPT_quadratic_inequality_condition_l168_16826

theorem quadratic_inequality_condition (a b c : ℝ) (h : a < 0) (disc : b^2 - 4 * a * c ≤ 0) : 
  ∀ x : ℝ, a * x^2 + b * x + c ≤ 0 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_condition_l168_16826


namespace NUMINAMATH_GPT_sum_of_coordinates_of_B_l168_16885

def point := (ℝ × ℝ)

noncomputable def point_A : point := (0, 0)

def line_y_equals_6 (B : point) : Prop := B.snd = 6

def slope_AB (A B : point) (m : ℝ) : Prop := (B.snd - A.snd) / (B.fst - A.fst) = m

theorem sum_of_coordinates_of_B (B : point) 
  (h1 : B.snd = 6)
  (h2 : slope_AB point_A B (3/5)) :
  B.fst + B.snd = 16 :=
sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_B_l168_16885


namespace NUMINAMATH_GPT_sequence_a5_l168_16879

theorem sequence_a5 : 
    ∃ (a : ℕ → ℚ), 
    a 1 = 1 / 3 ∧ 
    (∀ (n : ℕ), n ≥ 2 → a n = (-1 : ℚ)^n * 2 * a (n - 1)) ∧ 
    a 5 = -16 / 3 := 
sorry

end NUMINAMATH_GPT_sequence_a5_l168_16879


namespace NUMINAMATH_GPT_find_missing_number_l168_16832

theorem find_missing_number :
  (12 + x + 42 + 78 + 104) / 5 = 62 →
  (a + 255 + 511 + 1023 + x) / 5 = 398.2 →
  a = 128 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_find_missing_number_l168_16832


namespace NUMINAMATH_GPT_range_of_a_l168_16801

noncomputable def acute_angle_condition (a : ℝ) : Prop :=
  let M := (-2, 0)
  let N := (0, 2)
  let A := (-1, 1)
  (a > 0) ∧ (∀ P : ℝ × ℝ, (P.1 - a) ^ 2 + P.2 ^ 2 = 2 →
    (dist P A) > 2 * Real.sqrt 2)

theorem range_of_a (a : ℝ) : acute_angle_condition a ↔ a > Real.sqrt 7 - 1 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l168_16801


namespace NUMINAMATH_GPT_algorithm_outputs_min_value_l168_16804

theorem algorithm_outputs_min_value (a b c d : ℕ) :
  let m := a;
  let m := if b < m then b else m;
  let m := if c < m then c else m;
  let m := if d < m then d else m;
  m = min (min (min a b) c) d :=
by
  sorry

end NUMINAMATH_GPT_algorithm_outputs_min_value_l168_16804


namespace NUMINAMATH_GPT_simplify_expression_l168_16806

variable (x y : ℝ)

theorem simplify_expression (h : x ≠ y ∧ x ≠ -y) : 
  ((1 / (x - y) - 1 / (x + y)) / (x * y / (x^2 - y^2)) = 2 / x) :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l168_16806


namespace NUMINAMATH_GPT_X_Y_Z_sum_eq_17_l168_16877

variable {X Y Z : ℤ}

def base_ten_representation_15_fac (X Y Z : ℤ) : Prop :=
  Z = 0 ∧ (28 + X + Y) % 9 = 8 ∧ (X - Y) % 11 = 11

theorem X_Y_Z_sum_eq_17 (X Y Z : ℤ) (h : base_ten_representation_15_fac X Y Z) : X + Y + Z = 17 :=
by
  sorry

end NUMINAMATH_GPT_X_Y_Z_sum_eq_17_l168_16877


namespace NUMINAMATH_GPT_sum_geometric_series_l168_16805

noncomputable def S (r : ℝ) : ℝ :=
  12 / (1 - r)

theorem sum_geometric_series (a : ℝ) (h1 : -1 < a) (h2 : a < 1) (h3 : S a * S (-a) = 2016) :
  S a + S (-a) = 336 :=
by
  sorry

end NUMINAMATH_GPT_sum_geometric_series_l168_16805


namespace NUMINAMATH_GPT_age_ratio_l168_16803

-- Definitions of the ages based on the given conditions.
def Rachel_age : ℕ := 12  -- Rachel's age
def Father_age_when_Rachel_25 : ℕ := 60

-- Defining Mother, Father, Grandfather ages based on given conditions.
def Grandfather_age (R : ℕ) (F : ℕ) : ℕ := 2 * (F - 5)
def Father_age (R : ℕ) : ℕ := Father_age_when_Rachel_25 - (25 - R)

-- Proving the ratio of Grandfather's age to Rachel's age is 7:1
theorem age_ratio (R : ℕ) (F : ℕ) (G : ℕ) :
  R = Rachel_age →
  F = Father_age R →
  G = Grandfather_age R F →
  G / R = 7 := by
  exact sorry

end NUMINAMATH_GPT_age_ratio_l168_16803


namespace NUMINAMATH_GPT_veranda_width_l168_16889

def area_of_veranda (w : ℝ) : ℝ :=
  let room_area := 19 * 12
  let total_area := room_area + 140
  let total_length := 19 + 2 * w
  let total_width := 12 + 2 * w
  total_length * total_width - room_area

theorem veranda_width:
  ∃ w : ℝ, area_of_veranda w = 140 := by
  sorry

end NUMINAMATH_GPT_veranda_width_l168_16889


namespace NUMINAMATH_GPT_kelly_apples_total_l168_16838

def initial_apples : ℕ := 56
def second_day_pick : ℕ := 105
def third_day_pick : ℕ := 84
def apples_eaten : ℕ := 23

theorem kelly_apples_total :
  initial_apples + second_day_pick + third_day_pick - apples_eaten = 222 := by
  sorry

end NUMINAMATH_GPT_kelly_apples_total_l168_16838


namespace NUMINAMATH_GPT_selected_number_in_14th_group_is_272_l168_16833

-- Definitions based on conditions
def total_students : ℕ := 400
def sample_size : ℕ := 20
def first_selected_number : ℕ := 12
def sampling_interval : ℕ := total_students / sample_size
def target_group : ℕ := 14

-- Correct answer definition
def selected_number_in_14th_group : ℕ := first_selected_number + (target_group - 1) * sampling_interval

-- Theorem stating the correct answer is 272
theorem selected_number_in_14th_group_is_272 :
  selected_number_in_14th_group = 272 :=
sorry

end NUMINAMATH_GPT_selected_number_in_14th_group_is_272_l168_16833


namespace NUMINAMATH_GPT_correct_forecast_interpretation_l168_16851

/-- The probability of precipitation in the area tomorrow is 80%. -/
def prob_precipitation_tomorrow : ℝ := 0.8

/-- Multiple choice options regarding the interpretation of the probability of precipitation. -/
inductive forecast_interpretation
| A : forecast_interpretation
| B : forecast_interpretation
| C : forecast_interpretation
| D : forecast_interpretation

/-- The correct interpretation is Option C: "There is an 80% chance of rain in the area tomorrow." -/
def correct_interpretation : forecast_interpretation :=
forecast_interpretation.C

theorem correct_forecast_interpretation :
  (prob_precipitation_tomorrow = 0.8) → (correct_interpretation = forecast_interpretation.C) :=
by
  sorry

end NUMINAMATH_GPT_correct_forecast_interpretation_l168_16851


namespace NUMINAMATH_GPT_P_sufficient_but_not_necessary_for_Q_l168_16854

variable (x : ℝ)

def P := x ≥ 0
def Q := 2 * x + 1 / (2 * x + 1) ≥ 1

theorem P_sufficient_but_not_necessary_for_Q : (P x → Q x) ∧ ¬(Q x → P x) :=
by
  sorry

end NUMINAMATH_GPT_P_sufficient_but_not_necessary_for_Q_l168_16854


namespace NUMINAMATH_GPT_find_C_l168_16812

variable (A B C : ℕ)

theorem find_C (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 360) : C = 60 := by
  sorry

end NUMINAMATH_GPT_find_C_l168_16812


namespace NUMINAMATH_GPT_find_n_l168_16822

theorem find_n : ∃ n : ℕ, (∃ A B : ℕ, A ≠ B ∧ 10^(n-1) ≤ A ∧ A < 10^n ∧ 10^(n-1) ≤ B ∧ B < 10^n ∧ (10^n * A + B) % (10^n * B + A) = 0) ↔ n % 6 = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l168_16822


namespace NUMINAMATH_GPT_min_value_of_3a_plus_2_l168_16820

theorem min_value_of_3a_plus_2 
  (a : ℝ) 
  (h : 4 * a^2 + 7 * a + 3 = 2)
  : 3 * a + 2 >= -1 :=
sorry

end NUMINAMATH_GPT_min_value_of_3a_plus_2_l168_16820


namespace NUMINAMATH_GPT_fraction_of_students_who_walk_l168_16827

def fraction_by_bus : ℚ := 2 / 5
def fraction_by_car : ℚ := 1 / 5
def fraction_by_scooter : ℚ := 1 / 8
def total_fraction_not_walk := fraction_by_bus + fraction_by_car + fraction_by_scooter

theorem fraction_of_students_who_walk :
  (1 - total_fraction_not_walk) = 11 / 40 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_students_who_walk_l168_16827


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l168_16871

def M : Set ℝ := {x | x^2 - 3*x - 28 ≤ 0}
def N : Set ℝ := {x | x^2 - x - 6 > 0}
def intersection := {x : ℝ | -4 ≤ x ∧ x ≤ -2 ∨ 3 < x ∧ x ≤ 7}

theorem intersection_of_M_and_N : M ∩ N = intersection := by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l168_16871


namespace NUMINAMATH_GPT_problem_statement_l168_16836

theorem problem_statement (c d : ℤ) (h1 : 5 + c = 7 - d) (h2 : 6 + d = 10 + c) : 5 - c = 6 := 
by {
  sorry
}

end NUMINAMATH_GPT_problem_statement_l168_16836


namespace NUMINAMATH_GPT_semi_circle_radius_l168_16864

theorem semi_circle_radius (P : ℝ) (r : ℝ) (π : ℝ) (h_perimeter : P = 113) (h_pi : π = Real.pi) :
  r = P / (π + 2) :=
sorry

end NUMINAMATH_GPT_semi_circle_radius_l168_16864


namespace NUMINAMATH_GPT_longest_side_triangle_l168_16897

theorem longest_side_triangle (x : ℝ) 
  (h1 : 7 + (x + 4) + (2 * x + 1) = 36) : 
  max 7 (max (x + 4) (2 * x + 1)) = 17 :=
by sorry

end NUMINAMATH_GPT_longest_side_triangle_l168_16897


namespace NUMINAMATH_GPT_not_a_perfect_square_l168_16835

theorem not_a_perfect_square :
  ¬ (∃ x, (x: ℝ)^2 = 5^2025) :=
by
  sorry

end NUMINAMATH_GPT_not_a_perfect_square_l168_16835


namespace NUMINAMATH_GPT_sample_size_l168_16813

theorem sample_size (total_employees : ℕ) (male_employees : ℕ) (sampled_males : ℕ) (sample_size : ℕ) 
  (h1 : total_employees = 120) (h2 : male_employees = 80) (h3 : sampled_males = 24) : 
  sample_size = 36 :=
by
  sorry

end NUMINAMATH_GPT_sample_size_l168_16813


namespace NUMINAMATH_GPT_sum_in_base7_l168_16821

-- An encoder function for base 7 integers
def to_base7 (n : ℕ) : string :=
sorry -- skipping the implementation for brevity

-- Decoding the string representation back to a natural number
def from_base7 (s : string) : ℕ :=
sorry -- skipping the implementation for brevity

-- The provided numbers in base 7
def x : ℕ := from_base7 "666"
def y : ℕ := from_base7 "66"
def z : ℕ := from_base7 "6"

-- The expected sum in base 7
def expected_sum : ℕ := from_base7 "104"

-- The statement to be proved
theorem sum_in_base7 : x + y + z = expected_sum :=
sorry -- The proof is omitted

end NUMINAMATH_GPT_sum_in_base7_l168_16821


namespace NUMINAMATH_GPT_f_fe_eq_neg1_f_x_gt_neg1_solution_l168_16872

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log (1 / x)
  else if x < 0 then 1 / x
  else 0  -- handle the case for x = 0 explicitly if needed

theorem f_fe_eq_neg1 : 
  f (f (Real.exp 1)) = -1 := 
by
  -- proof to be filled in
  sorry

theorem f_x_gt_neg1_solution :
  {x : ℝ | f x > -1} = {x : ℝ | (x < -1) ∨ (0 < x ∧ x < Real.exp 1)} :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_f_fe_eq_neg1_f_x_gt_neg1_solution_l168_16872


namespace NUMINAMATH_GPT_cylinder_volume_ratio_l168_16861

variable (h r : ℝ)

theorem cylinder_volume_ratio (h r : ℝ) :
  let V_original := π * r^2 * h
  let h_new := 2 * h
  let r_new := 4 * r
  let V_new := π * (r_new)^2 * h_new
  V_new = 32 * V_original :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_ratio_l168_16861


namespace NUMINAMATH_GPT_max_a_squared_b_l168_16816

theorem max_a_squared_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * (a + b) = 27) : a^2 * b ≤ 54 :=
sorry

end NUMINAMATH_GPT_max_a_squared_b_l168_16816


namespace NUMINAMATH_GPT_greta_hourly_wage_is_12_l168_16823

-- Define constants
def greta_hours : ℕ := 40
def lisa_hourly_wage : ℕ := 15
def lisa_hours : ℕ := 32

-- Define the total earnings of Greta and Lisa
def greta_earnings (G : ℕ) : ℕ := greta_hours * G
def lisa_earnings : ℕ := lisa_hours * lisa_hourly_wage

-- Main theorem statement
theorem greta_hourly_wage_is_12 (G : ℕ) (h : greta_earnings G = lisa_earnings) : G = 12 :=
by
  sorry

end NUMINAMATH_GPT_greta_hourly_wage_is_12_l168_16823


namespace NUMINAMATH_GPT_number_of_perfect_squares_and_cubes_l168_16890

theorem number_of_perfect_squares_and_cubes (n m k : ℕ) (hn : n^2 < 1000) (hn' : (n + 1)^2 ≥ 1000) (hm : m^3 < 1000) (hm' : (m + 1)^3 ≥ 1000) (hk : k^6 < 1000) (hk' : (k + 1)^6 ≥ 1000) :
  (n + m - k) = 38 :=
sorry

end NUMINAMATH_GPT_number_of_perfect_squares_and_cubes_l168_16890


namespace NUMINAMATH_GPT_at_op_subtraction_l168_16829

-- Define the operation @
def at_op (x y : ℝ) : ℝ := 3 * x * y - 2 * x + y

-- Prove the problem statement
theorem at_op_subtraction :
  at_op 6 4 - at_op 4 6 = -6 :=
by
  sorry

end NUMINAMATH_GPT_at_op_subtraction_l168_16829


namespace NUMINAMATH_GPT_smallest_n_not_divisible_by_10_smallest_n_correct_l168_16819

theorem smallest_n_not_divisible_by_10 :
  ∃ n ≥ 2017, n % 4 = 0 ∧ (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0 :=
by
  -- Existence proof of such n is omitted
  sorry

def smallest_n : Nat :=
  Nat.find $ smallest_n_not_divisible_by_10

theorem smallest_n_correct : smallest_n = 2020 :=
by
  -- Correctness proof of smallest_n is omitted
  sorry

end NUMINAMATH_GPT_smallest_n_not_divisible_by_10_smallest_n_correct_l168_16819


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l168_16899

theorem hyperbola_asymptotes:
  (∀ x y : Real, (x^2 / 16 - y^2 / 9 = 1) → (y = 3 / 4 * x ∨ y = -3 / 4 * x)) :=
by {
  sorry
}

end NUMINAMATH_GPT_hyperbola_asymptotes_l168_16899


namespace NUMINAMATH_GPT_sum_of_coefficients_l168_16888

def polynomial (x : ℤ) : ℤ := 3 * (x^8 - 2 * x^5 + 4 * x^3 - 7) - 5 * (2 * x^4 - 3 * x^2 + 8) + 6 * (x^6 - 3)

theorem sum_of_coefficients : polynomial 1 = -59 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l168_16888


namespace NUMINAMATH_GPT_tan_alpha_minus_beta_l168_16884

theorem tan_alpha_minus_beta (α β : ℝ) (hα : Real.tan α = 8) (hβ : Real.tan β = 7) :
  Real.tan (α - β) = 1 / 57 := 
sorry

end NUMINAMATH_GPT_tan_alpha_minus_beta_l168_16884


namespace NUMINAMATH_GPT_parabola_min_value_sum_abc_zero_l168_16886

theorem parabola_min_value_sum_abc_zero
  (a b c : ℝ)
  (h1 : ∀ x y : ℝ, y = ax^2 + bx + c → y = 0 ∧ ((x = 1) ∨ (x = 5)))
  (h2 : ∃ x : ℝ, (∀ t : ℝ, ax^2 + bx + c ≤ t^2) ∧ (ax^2 + bx + c = 36)) :
  a + b + c = 0 :=
sorry

end NUMINAMATH_GPT_parabola_min_value_sum_abc_zero_l168_16886


namespace NUMINAMATH_GPT_total_fruits_in_bowl_l168_16860

theorem total_fruits_in_bowl (bananas apples oranges : ℕ) 
  (h1 : bananas = 2) 
  (h2 : apples = 2 * bananas) 
  (h3 : oranges = 6) : 
  bananas + apples + oranges = 12 := 
by 
  sorry

end NUMINAMATH_GPT_total_fruits_in_bowl_l168_16860


namespace NUMINAMATH_GPT_product_of_variables_l168_16876

variables (a b c d : ℚ)

theorem product_of_variables :
  4 * a + 5 * b + 7 * c + 9 * d = 82 →
  d + c = 2 * b →
  2 * b + 2 * c = 3 * a →
  c - 2 = d →
  a * b * c * d = 276264960 / 14747943 := by
  sorry

end NUMINAMATH_GPT_product_of_variables_l168_16876


namespace NUMINAMATH_GPT_sin_300_eq_neg_sqrt_3_div_2_l168_16828

noncomputable def sin_300_degrees : ℝ := Real.sin (300 * Real.pi / 180)

theorem sin_300_eq_neg_sqrt_3_div_2 : sin_300_degrees = -Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_300_eq_neg_sqrt_3_div_2_l168_16828


namespace NUMINAMATH_GPT_largest_abs_val_among_2_3_neg3_neg4_l168_16807

def abs_val (a : Int) : Nat := a.natAbs

theorem largest_abs_val_among_2_3_neg3_neg4 : 
  ∀ (x : Int), x ∈ [2, 3, -3, -4] → abs_val x ≤ abs_val (-4) := by
  sorry

end NUMINAMATH_GPT_largest_abs_val_among_2_3_neg3_neg4_l168_16807


namespace NUMINAMATH_GPT_quadratic_sum_of_roots_l168_16818

theorem quadratic_sum_of_roots (a b : ℝ)
  (h1: ∀ x: ℝ, x^2 + b * x - a < 0 ↔ 3 < x ∧ x < 4):
  a + b = -19 :=
sorry

end NUMINAMATH_GPT_quadratic_sum_of_roots_l168_16818


namespace NUMINAMATH_GPT_maximize_fraction_l168_16850

theorem maximize_fraction (A B C D : ℕ) (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_digits : A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9)
  (h_nonneg : 0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C ∧ 0 ≤ D)
  (h_integer : (A + B) % (C + D) = 0) : A + B = 17 :=
sorry

end NUMINAMATH_GPT_maximize_fraction_l168_16850


namespace NUMINAMATH_GPT_parametric_curve_to_general_form_l168_16811

theorem parametric_curve_to_general_form :
  ∃ (a b c : ℚ), ∀ (t : ℝ), 
  (a = 8 / 225) ∧ (b = 4 / 75) ∧ (c = 1 / 25) ∧ 
  (a * (3 * Real.sin t)^2 + b * (3 * Real.sin t) * (5 * Real.cos t - 2 * Real.sin t) + c * (5 * Real.cos t - 2 * Real.sin t)^2 = 1) :=
by
  use 8 / 225, 4 / 75, 1 / 25
  sorry

end NUMINAMATH_GPT_parametric_curve_to_general_form_l168_16811


namespace NUMINAMATH_GPT_gcd_228_2008_l168_16869

theorem gcd_228_2008 : Int.gcd 228 2008 = 4 := by
  sorry

end NUMINAMATH_GPT_gcd_228_2008_l168_16869


namespace NUMINAMATH_GPT_unique_solution_qx2_minus_16x_plus_8_eq_0_l168_16866

theorem unique_solution_qx2_minus_16x_plus_8_eq_0 (q : ℝ) (hq : q ≠ 0) :
  (∀ x : ℝ, q * x^2 - 16 * x + 8 = 0 → (256 - 32 * q = 0)) → q = 8 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_qx2_minus_16x_plus_8_eq_0_l168_16866


namespace NUMINAMATH_GPT_compare_final_values_l168_16881

noncomputable def final_value_Almond (initial: ℝ): ℝ := (initial * 1.15) * 0.85
noncomputable def final_value_Bean (initial: ℝ): ℝ := (initial * 0.80) * 1.20
noncomputable def final_value_Carrot (initial: ℝ): ℝ := (initial * 1.10) * 0.90

theorem compare_final_values (initial: ℝ) (h_positive: 0 < initial):
  final_value_Almond initial < final_value_Bean initial ∧ 
  final_value_Bean initial < final_value_Carrot initial := by
  sorry

end NUMINAMATH_GPT_compare_final_values_l168_16881


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l168_16853

theorem necessary_but_not_sufficient (a : ℝ) (h : a > 0) : a > 0 ↔ ((a > 0) ∧ (a < 2) → (a^2 - 2 * a < 0)) :=
by
    sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l168_16853


namespace NUMINAMATH_GPT_task1_on_time_task2_not_on_time_l168_16895

/-- Define the probabilities for task 1 and task 2 -/
def P_A : ℚ := 3 / 8
def P_B : ℚ := 3 / 5

/-- The probability that task 1 will be completed on time but task 2 will not is 3 / 20. -/
theorem task1_on_time_task2_not_on_time (P_A : ℚ) (P_B : ℚ) : P_A = 3 / 8 → P_B = 3 / 5 → P_A * (1 - P_B) = 3 / 20 :=
by
  intros hPA hPB
  rw [hPA, hPB]
  norm_num

end NUMINAMATH_GPT_task1_on_time_task2_not_on_time_l168_16895


namespace NUMINAMATH_GPT_pyramid_volume_correct_l168_16849

noncomputable def pyramid_volume (A_PQRS A_PQT A_RST: ℝ) (side: ℝ) (height: ℝ) : ℝ :=
  (1 / 3) * A_PQRS * height

theorem pyramid_volume_correct 
  (A_PQRS : ℝ) (A_PQT : ℝ) (A_RST : ℝ) (side : ℝ) (height_PQT : ℝ) (height_RST : ℝ)
  (h_PQT : 2 * A_PQT / side = height_PQT)
  (h_RST : 2 * A_RST / side = height_RST)
  (eq1 : height_PQT^2 + side^2 = height_RST^2 + (side - height_PQT)^2) 
  (eq2 : height_RST^2 = height_PQT^2 + (height_PQT - side)^2)
  : pyramid_volume A_PQRS A_PQT A_RST = 5120 / 3 :=
by
  -- Skipping the proof steps
  sorry

end NUMINAMATH_GPT_pyramid_volume_correct_l168_16849


namespace NUMINAMATH_GPT_circle_center_l168_16855

theorem circle_center :
  ∃ c : ℝ × ℝ, c = (-1, 3) ∧ ∀ (x y : ℝ), (4 * x^2 + 8 * x + 4 * y^2 - 24 * y + 96 = 0 ↔ (x + 1)^2 + (y - 3)^2 = 14) :=
by
  sorry

end NUMINAMATH_GPT_circle_center_l168_16855


namespace NUMINAMATH_GPT_least_subtraction_divisibility_l168_16847

theorem least_subtraction_divisibility :
  ∃ k : ℕ, 427398 - k = 14 * n ∧ k = 6 :=
by
  use 6
  sorry

end NUMINAMATH_GPT_least_subtraction_divisibility_l168_16847


namespace NUMINAMATH_GPT_proof_theorem_l168_16802

noncomputable def proof_problem 
  (m n : ℕ) 
  (x y z : ℝ) 
  (h1 : 0 ≤ x) (h2 : x ≤ 1) 
  (h3 : 0 ≤ y) (h4 : y ≤ 1) 
  (h5 : 0 ≤ z) (h6 : z ≤ 1) 
  (h7 : m > 0) (h8 : n > 0) 
  (h9 : m + n = p) : Prop :=
0 ≤ x^p + y^p + z^p - x^m * y^n - y^m * z^n - z^m * x^n ∧ 
x^p + y^p + z^p - x^m * y^n - y^m * z^n - z^m * x^n ≤ 1

theorem proof_theorem (m n : ℕ) (x y z : ℝ)
  (h1 : 0 ≤ x) (h2 : x ≤ 1) 
  (h3 : 0 ≤ y) (h4 : y ≤ 1) 
  (h5 : 0 ≤ z) (h6 : z ≤ 1) 
  (h7 : m > 0) (h8 : n > 0) 
  (h9 : m + n = p) : 
  proof_problem m n x y z h1 h2 h3 h4 h5 h6 h7 h8 h9 :=
by {
  sorry
}

end NUMINAMATH_GPT_proof_theorem_l168_16802


namespace NUMINAMATH_GPT_greatest_possible_n_l168_16875

theorem greatest_possible_n (n : ℤ) (h : 93 * n^3 ≤ 145800) : n ≤ 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_greatest_possible_n_l168_16875


namespace NUMINAMATH_GPT_problem_statement_l168_16882

theorem problem_statement (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l168_16882
