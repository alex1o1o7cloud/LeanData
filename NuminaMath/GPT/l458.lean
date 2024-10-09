import Mathlib

namespace eggs_for_dinner_l458_45863

-- Definitions of the conditions
def eggs_for_breakfast := 2
def eggs_for_lunch := 3
def total_eggs := 6

-- The quantity of eggs for dinner needs to be proved
theorem eggs_for_dinner :
  ∃ x : ℕ, x + eggs_for_breakfast + eggs_for_lunch = total_eggs ∧ x = 1 :=
by
  sorry

end eggs_for_dinner_l458_45863


namespace type_R_completion_time_l458_45825

theorem type_R_completion_time :
  (∃ R : ℝ, (2 / R + 3 / 7 = 1 / 1.2068965517241381) ∧ abs (R - 5) < 0.01) :=
  sorry

end type_R_completion_time_l458_45825


namespace find_n_tangent_l458_45870

theorem find_n_tangent (n : ℤ) (h1 : -180 < n) (h2 : n < 180) (h3 : ∃ k : ℤ, 210 = n + 180 * k) : n = 30 :=
by
  -- Proof steps would go here
  sorry

end find_n_tangent_l458_45870


namespace median_of_triangle_l458_45816

variable (a b c : ℝ)

noncomputable def AM : ℝ :=
  (Real.sqrt (2 * b * b + 2 * c * c - a * a)) / 2

theorem median_of_triangle :
  abs (((b + c) / 2) - (a / 2)) < AM a b c ∧ 
  AM a b c < (b + c) / 2 := 
by
  sorry

end median_of_triangle_l458_45816


namespace problem_1_problem_2_l458_45829

theorem problem_1 
  : (∃ (m n : ℝ), m = -1 ∧ n = 1 ∧ ∀ (x : ℝ), |x + 1| + |2 * x - 1| ≤ 3 ↔ m ≤ x ∧ x ≤ n) :=
sorry

theorem problem_2 
  : (∀ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 2 → 
    ∃ (min_val : ℝ), min_val = 9 / 2 ∧ 
    ∀ (x : ℝ), x = (1 / a + 1 / b + 1 / c) → min_val ≤ x) :=
sorry

end problem_1_problem_2_l458_45829


namespace houses_count_l458_45807

theorem houses_count (n : ℕ) 
  (h1 : ∃ k : ℕ, k + 7 = 12)
  (h2 : ∃ m : ℕ, m + 25 = 30) :
  n = 32 :=
sorry

end houses_count_l458_45807


namespace inequality_solution_intervals_l458_45811

theorem inequality_solution_intervals (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (x / (x - 1) + (x + 3) / (2 * x) ≥ 4) ↔ (0 < x ∧ x < 1) := 
sorry

end inequality_solution_intervals_l458_45811


namespace parabola_symmetric_y_axis_intersection_l458_45873

theorem parabola_symmetric_y_axis_intersection :
  ∀ (x y : ℝ),
  (x = y ∨ x*x + y*y - 6*y = 0) ∧ (x*x = 3 * y) :=
by 
  sorry

end parabola_symmetric_y_axis_intersection_l458_45873


namespace no_real_solutions_l458_45835

noncomputable def equation (x : ℝ) := x + 48 / (x - 3) + 1

theorem no_real_solutions : ∀ x : ℝ, equation x ≠ 0 :=
by
  intro x
  sorry

end no_real_solutions_l458_45835


namespace inequality_range_of_a_l458_45840

theorem inequality_range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → |2 * x - a| > x - 1) ↔ a < 3 ∨ a > 5 :=
by
  sorry

end inequality_range_of_a_l458_45840


namespace choir_members_l458_45805

theorem choir_members (n k c : ℕ) (h1 : n = k^2 + 11) (h2 : n = c * (c + 5)) : n = 300 :=
sorry

end choir_members_l458_45805


namespace matrix_inverse_problem_l458_45821

theorem matrix_inverse_problem
  (x y z w : ℚ)
  (h1 : 2 * x + 3 * w = 1)
  (h2 : x * z = 15)
  (h3 : 4 * w = -8)
  (h4 : 4 * z = 5 * y) :
  x * y * z * w = -102.857 := by
    sorry

end matrix_inverse_problem_l458_45821


namespace product_of_numbers_l458_45830

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) : x * y = 200 :=
sorry

end product_of_numbers_l458_45830


namespace allison_total_video_hours_l458_45843

def total_video_hours_uploaded (total_days: ℕ) (half_days: ℕ) (first_half_rate: ℕ) (second_half_rate: ℕ): ℕ :=
  first_half_rate * half_days + second_half_rate * (total_days - half_days)

theorem allison_total_video_hours :
  total_video_hours_uploaded 30 15 10 20 = 450 :=
by
  sorry

end allison_total_video_hours_l458_45843


namespace intersection_A_B_l458_45898

def A : Set ℝ := { x | (x + 1) / (x - 1) ≤ 0 }
def B : Set ℝ := { x | Real.log x ≤ 0 }

theorem intersection_A_B : A ∩ B = { x | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_A_B_l458_45898


namespace union_eq_l458_45837

def A : Set ℤ := {-1, 0, 3}
def B : Set ℤ := {-1, 1, 2, 3}

theorem union_eq : A ∪ B = {-1, 0, 1, 2, 3} := 
by 
  sorry

end union_eq_l458_45837


namespace total_cars_all_own_l458_45882

theorem total_cars_all_own :
  ∀ (C L S K : ℕ), 
  (C = 5) →
  (L = C + 4) →
  (K = 2 * C) →
  (S = K - 2) →
  (C + L + K + S = 32) :=
by
  intros C L S K
  intro hC
  intro hL
  intro hK
  intro hS
  sorry

end total_cars_all_own_l458_45882


namespace sum_arithmetic_sequence_l458_45877

theorem sum_arithmetic_sequence (S : ℕ → ℕ) :
  S 7 = 21 ∧ S 17 = 34 → S 27 = 27 :=
by
  sorry

end sum_arithmetic_sequence_l458_45877


namespace minimum_n_value_l458_45806

-- Define a multiple condition
def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

-- Given conditions
def conditions (n : ℕ) : Prop := 
  (n ≥ 8) ∧ is_multiple 4 n ∧ is_multiple 8 n

-- Lean theorem statement for the problem
theorem minimum_n_value (n : ℕ) (h : conditions n) : n = 8 :=
  sorry

end minimum_n_value_l458_45806


namespace distance_foci_l458_45849

noncomputable def distance_between_foci := 
  let F1 := (4, 5)
  let F2 := (-6, 9)
  Real.sqrt ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2) 

theorem distance_foci : 
  ∃ (F1 F2 : ℝ × ℝ), 
    F1 = (4, 5) ∧ 
    F2 = (-6, 9) ∧ 
    distance_between_foci = 2 * Real.sqrt 29 := by {
  sorry
}

end distance_foci_l458_45849


namespace largest_negative_integer_solution_l458_45841

theorem largest_negative_integer_solution :
  ∃ x : ℤ, x < 0 ∧ 50 * x + 14 % 24 = 10 % 24 ∧ ∀ y : ℤ, (y < 0 ∧ y % 12 = 10 % 12 → y ≤ x) :=
by
  sorry

end largest_negative_integer_solution_l458_45841


namespace group_division_ways_l458_45871

def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem group_division_ways : 
  choose 30 10 * choose 20 10 * choose 10 10 = Nat.factorial 30 / (Nat.factorial 10 * Nat.factorial 10 * Nat.factorial 10) := 
by
  sorry

end group_division_ways_l458_45871


namespace original_cube_volume_l458_45845

theorem original_cube_volume 
  (a : ℕ) 
  (h : 3 * a * (a - a / 2) * a - a^3 = 2 * a^2) : 
  a = 4 → a^3 = 64 := 
by
  sorry

end original_cube_volume_l458_45845


namespace sin_add_arctan_arcsin_l458_45854

theorem sin_add_arctan_arcsin :
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan 3
  (Real.sin a = 4 / 5) →
  (Real.tan b = 3) →
  Real.sin (a + b) = (13 * Real.sqrt 10) / 50 :=
by
  intros _ _
  sorry

end sin_add_arctan_arcsin_l458_45854


namespace solution_values_sum_l458_45868

theorem solution_values_sum (x y : ℝ) (p q r s : ℕ) 
  (hx : x + y = 5) 
  (hxy : 2 * x * y = 5) 
  (hx_form : x = (p + q * Real.sqrt r) / s ∨ x = (p - q * Real.sqrt r) / s) 
  (hpqs_pos : p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0) : 
  p + q + r + s = 23 := 
sorry

end solution_values_sum_l458_45868


namespace inequality_solution_set_correct_l458_45802

noncomputable def inequality_solution_set (a b c x : ℝ) : Prop :=
  (a > c) → (b + c > 0) → ((x - b < 0 ∧ x < c) ∨ (x > a)) → ((x - c) * (x + b) / (x - a) > 0)

theorem inequality_solution_set_correct (a b c : ℝ) :
  a > c → b + c > 0 → ∀ x, ((a > c) → (b + c > 0) → (((x - b < 0 ∧ x < c) ∨ (x > a)) → ((x - c) * (x + b) / (x - a) > 0))) :=
by
  intros h1 h2 x
  sorry

end inequality_solution_set_correct_l458_45802


namespace nested_expression_rational_count_l458_45858

theorem nested_expression_rational_count : 
  let count := Nat.card {n : ℕ // 1 ≤ n ∧ n ≤ 2021 ∧ ∃ m : ℕ, m % 2 = 1 ∧ m * m = 1 + 4 * n}
  count = 44 := 
by sorry

end nested_expression_rational_count_l458_45858


namespace discount_percent_l458_45865

theorem discount_percent (MP CP SP : ℝ)
  (h1 : CP = 0.64 * MP)
  (h2 : (SP - CP) / CP * 100 = 34.375) :
  ((MP - SP) / MP * 100) = 14 :=
by
  -- Proof would go here
  sorry

end discount_percent_l458_45865


namespace train_cross_bridge_time_l458_45861

/-
  Define the given conditions:
  - Length of the train (lt): 200 m
  - Speed of the train (st_kmh): 72 km/hr
  - Length of the bridge (lb): 132 m
-/

namespace TrainProblem

def length_of_train : ℕ := 200
def speed_of_train_kmh : ℕ := 72
def length_of_bridge : ℕ := 132

/-
  Convert speed from km/hr to m/s
-/
def speed_of_train_ms : ℕ := speed_of_train_kmh * 1000 / 3600

/-
  Calculate total distance to be traveled (train length + bridge length).
-/
def total_distance : ℕ := length_of_train + length_of_bridge

/-
  Use the formula Time = Distance / Speed
-/
def time_to_cross_bridge : ℚ := total_distance / speed_of_train_ms

theorem train_cross_bridge_time : 
  (length_of_train = 200) →
  (speed_of_train_kmh = 72) →
  (length_of_bridge = 132) →
  time_to_cross_bridge = 16.6 :=
by
  intros lt st lb
  sorry

end TrainProblem

end train_cross_bridge_time_l458_45861


namespace f_2_plus_f_5_eq_2_l458_45842

noncomputable def f : ℝ → ℝ := sorry

open Real

-- Conditions: f(3^x) = x * log 9
axiom f_cond (x : ℝ) : f (3^x) = x * log 9

-- Question: f(2) + f(5) = 2
theorem f_2_plus_f_5_eq_2 : f 2 + f 5 = 2 := sorry

end f_2_plus_f_5_eq_2_l458_45842


namespace percent_carnations_l458_45850

theorem percent_carnations (F : ℕ) (H1 : 3 / 5 * F = pink) (H2 : 1 / 5 * F = white) 
(H3 : F - pink - white = red) (H4 : 1 / 2 * pink = pink_roses)
(H5 : pink - pink_roses = pink_carnations) (H6 : 1 / 2 * red = red_carnations)
(H7 : white = white_carnations) : 
100 * (pink_carnations + red_carnations + white_carnations) / F = 60 :=
sorry

end percent_carnations_l458_45850


namespace sum_modulo_9_l458_45834

theorem sum_modulo_9 : 
  (88000 + 88002 + 87999 + 88001 + 88003 + 87998) % 9 = 0 := 
by
  sorry

end sum_modulo_9_l458_45834


namespace cost_ratio_two_pastries_pies_l458_45866

theorem cost_ratio_two_pastries_pies (s p : ℝ) (h1 : 2 * s = 3 * (2 * p)) :
  (s + p) / (2 * p) = 2 :=
by
  sorry

end cost_ratio_two_pastries_pies_l458_45866


namespace probability_of_winning_l458_45893

def roll_is_seven (d1 d2 : ℕ) : Prop :=
  d1 + d2 = 7

theorem probability_of_winning (d1 d2 : ℕ) (h : roll_is_seven d1 d2) :
  (1/6 : ℚ) = 1/6 :=
by
  sorry

end probability_of_winning_l458_45893


namespace B_pow_2017_eq_B_l458_45879

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![ ![0, 1, 0], ![0, 0, 1], ![1, 0, 0] ]

theorem B_pow_2017_eq_B : B^2017 = B := by
  sorry

end B_pow_2017_eq_B_l458_45879


namespace evaluate_expression_l458_45885

theorem evaluate_expression :
  (4^2 - 4) + (5^2 - 5) - (7^3 - 7) + (3^2 - 3) = -298 :=
by
  sorry

end evaluate_expression_l458_45885


namespace length_of_uncovered_side_l458_45817

variables (L W : ℝ)

-- Conditions
def area_eq_680 := (L * W = 680)
def fence_eq_178 := (2 * W + L = 178)

-- Theorem statement to prove the length of the uncovered side
theorem length_of_uncovered_side (h1 : area_eq_680 L W) (h2 : fence_eq_178 L W) : L = 170 := 
sorry

end length_of_uncovered_side_l458_45817


namespace number_of_common_tangents_l458_45813

def circleM (x y : ℝ) : Prop := x^2 + y^2 - 4 * y = 0
def circleN (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

theorem number_of_common_tangents : ∃ n : ℕ, n = 2 ∧ 
  (∀ x y : ℝ, circleM x y → circleN x y → false) :=
by
  sorry

end number_of_common_tangents_l458_45813


namespace smallest_positive_period_of_f_max_min_values_of_f_l458_45820

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + Real.cos (2 * x)

theorem smallest_positive_period_of_f :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) :=
sorry

theorem max_min_values_of_f :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), 0 ≤ f x ∧ f x ≤ 1 + Real.sqrt 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 0) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 1 + Real.sqrt 2) :=
sorry

end smallest_positive_period_of_f_max_min_values_of_f_l458_45820


namespace smallest_digit_d_l458_45856

theorem smallest_digit_d (d : ℕ) (hd : d < 10) :
  (∃ d, (20 - (8 + d)) % 11 = 0 ∧ d < 10) → d = 1 :=
by
  sorry

end smallest_digit_d_l458_45856


namespace find_roses_last_year_l458_45867

-- Definitions based on conditions
def roses_last_year : ℕ := sorry
def roses_this_year := roses_last_year / 2
def roses_needed := 2 * roses_last_year
def rose_cost := 3 -- cost per rose in dollars
def total_spent := 54 -- total spent in dollars

-- Formulate the problem
theorem find_roses_last_year (h : 2 * roses_last_year - roses_this_year = 18)
  (cost_eq : total_spent / rose_cost = 18) :
  roses_last_year = 12 :=
by
  sorry

end find_roses_last_year_l458_45867


namespace find_width_of_rectangle_l458_45897

variable (w : ℝ) (l : ℝ) (P : ℝ)

def width_correct (h1 : P = 150) (h2 : l = w + 15) : Prop :=
  w = 30

-- Theorem statement in Lean
theorem find_width_of_rectangle (h1 : P = 150) (h2 : l = w + 15) (h3 : P = 2 * l + 2 * w) : width_correct w l P h1 h2 :=
by
  sorry

end find_width_of_rectangle_l458_45897


namespace minimal_overlap_facebook_instagram_l458_45833

variable (P : ℝ → Prop)
variable [Nonempty (Set.Icc 0 1)]

theorem minimal_overlap_facebook_instagram :
  ∀ (f i : ℝ), f = 0.85 → i = 0.75 → ∃ b : ℝ, 0 ≤ b ∧ b ≤ 1 ∧ b = 0.6 :=
by
  intros
  sorry

end minimal_overlap_facebook_instagram_l458_45833


namespace number_of_books_to_break_even_is_4074_l458_45886

-- Definitions from problem conditions
def fixed_costs : ℝ := 35630
def variable_cost_per_book : ℝ := 11.50
def selling_price_per_book : ℝ := 20.25

-- The target number of books to sell for break-even
def target_books_to_break_even : ℕ := 4074

-- Lean statement to prove that number of books to break even is 4074
theorem number_of_books_to_break_even_is_4074 :
  let total_costs (x : ℝ) := fixed_costs + variable_cost_per_book * x
  let total_revenue (x : ℝ) := selling_price_per_book * x
  ∃ x : ℝ, total_costs x = total_revenue x → x = target_books_to_break_even := by
  sorry

end number_of_books_to_break_even_is_4074_l458_45886


namespace Natasha_speed_over_limit_l458_45823

theorem Natasha_speed_over_limit (d : ℕ) (t : ℕ) (speed_limit : ℕ) 
    (h1 : d = 60) 
    (h2 : t = 1) 
    (h3 : speed_limit = 50) : (d / t - speed_limit = 10) :=
by
  -- Because d = 60, t = 1, and speed_limit = 50, we need to prove (60 / 1 - 50) = 10
  sorry

end Natasha_speed_over_limit_l458_45823


namespace param_A_valid_param_B_valid_l458_45810

-- Definition of the line equation
def line_eq (x y : ℝ) : Prop := y = 2 * x - 4

-- Parameterization A
def param_A (t : ℝ) : ℝ × ℝ := (2 - t, -2 * t)

-- Parameterization B
def param_B (t : ℝ) : ℝ × ℝ := (5 * t, 10 * t - 4)

-- Theorem to prove that parameterization A satisfies the line equation
theorem param_A_valid (t : ℝ) : line_eq (param_A t).1 (param_A t).2 := by
  sorry

-- Theorem to prove that parameterization B satisfies the line equation
theorem param_B_valid (t : ℝ) : line_eq (param_B t).1 (param_B t).2 := by
  sorry

end param_A_valid_param_B_valid_l458_45810


namespace min_value_of_a3b2c_l458_45875

theorem min_value_of_a3b2c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : 1 / a + 1 / b + 1 / c = 9) : 
  a^3 * b^2 * c ≥ 1 / 2916 :=
by 
  sorry

end min_value_of_a3b2c_l458_45875


namespace sin_alpha_eq_sin_beta_l458_45872

theorem sin_alpha_eq_sin_beta (α β : Real) (k : Int) 
  (h_symmetry : α + β = 2 * k * Real.pi + Real.pi) : 
  Real.sin α = Real.sin β := 
by 
  sorry

end sin_alpha_eq_sin_beta_l458_45872


namespace prob_enter_A_and_exit_F_l458_45822

-- Define the problem description
def entrances : ℕ := 2
def exits : ℕ := 3

-- Define the probabilities
def prob_enter_A : ℚ := 1 / entrances
def prob_exit_F : ℚ := 1 / exits

-- Statement that encapsulates the proof problem
theorem prob_enter_A_and_exit_F : prob_enter_A * prob_exit_F = 1 / 6 := 
by sorry

end prob_enter_A_and_exit_F_l458_45822


namespace cos_C_max_ab_over_c_l458_45809

theorem cos_C_max_ab_over_c
  (a b c S : ℝ) (A B C : ℝ)
  (h1 : 6 * S = a^2 * Real.sin A + b^2 * Real.sin B)
  (h2 : a / Real.sin A = b / Real.sin B)
  (h3 : b / Real.sin B = c / Real.sin C)
  (h4 : S = 0.5 * a * b * Real.sin C)
  : Real.cos C = 7 / 9 := 
sorry

end cos_C_max_ab_over_c_l458_45809


namespace eyes_saw_plane_l458_45851

theorem eyes_saw_plane (total_students : ℕ) (fraction_looked_up : ℚ) (students_with_eyepatches : ℕ) :
  total_students = 200 → fraction_looked_up = 3/4 → students_with_eyepatches = 20 →
  ∃ eyes_saw_plane, eyes_saw_plane = 280 :=
by
  intros h1 h2 h3
  sorry

end eyes_saw_plane_l458_45851


namespace ratio_of_ages_l458_45895

theorem ratio_of_ages (S M : ℕ) (h1 : M = S + 24) (h2 : M + 2 = (S + 2) * 2) (h3 : S = 22) : (M + 2) / (S + 2) = 2 := 
by {
  sorry
}

end ratio_of_ages_l458_45895


namespace tens_digit_6_pow_45_l458_45896

theorem tens_digit_6_pow_45 : (6 ^ 45 % 100) / 10 = 0 := 
by 
  sorry

end tens_digit_6_pow_45_l458_45896


namespace probability_triplet_1_2_3_in_10_rolls_l458_45884

noncomputable def probability_of_triplet (n : ℕ) : ℝ :=
  let A0 := (6^10 : ℝ)
  let A1 := (8 * 6^7 : ℝ)
  let A2 := (15 * 6^4 : ℝ)
  let A3 := (4 * 6 : ℝ)
  let total := A0
  let p := (A0 - (total - (A1 - A2 + A3))) / total
  p

theorem probability_triplet_1_2_3_in_10_rolls : 
  abs (probability_of_triplet 10 - 0.0367) < 0.0001 :=
by
  sorry

end probability_triplet_1_2_3_in_10_rolls_l458_45884


namespace abs_frac_lt_one_l458_45831

theorem abs_frac_lt_one (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  |(x - y) / (1 - x * y)| < 1 :=
sorry

end abs_frac_lt_one_l458_45831


namespace factorial_division_l458_45887

-- Define factorial using the standard library's factorial function
def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- The problem statement
theorem factorial_division :
  (factorial 15) / (factorial 6 * factorial 9) = 834 :=
sorry

end factorial_division_l458_45887


namespace number_of_three_digit_prime_integers_l458_45888

def prime_digits : Set Nat := {2, 3, 5, 7}

theorem number_of_three_digit_prime_integers : 
  (∃ count, count = 4 * 4 * 4 ∧ count = 64) :=
by
  sorry

end number_of_three_digit_prime_integers_l458_45888


namespace fill_time_l458_45839

noncomputable def time_to_fill (X Y Z : ℝ) : ℝ :=
  1 / X + 1 / Y + 1 / Z

theorem fill_time 
  (V X Y Z : ℝ) 
  (h1 : X + Y = V / 3) 
  (h2 : X + Z = V / 2) 
  (h3 : Y + Z = V / 4) :
  1 / time_to_fill X Y Z = 24 / 13 :=
by
  sorry

end fill_time_l458_45839


namespace sum_of_angles_equal_360_l458_45818

variables (A B C D F G : ℝ)

-- Given conditions.
def is_quadrilateral_interior_sum (A B C D : ℝ) : Prop := A + B + C + D = 360
def split_internal_angles (F G : ℝ) (C D : ℝ) : Prop := F + G = C + D

-- Proof problem statement.
theorem sum_of_angles_equal_360
  (h1 : is_quadrilateral_interior_sum A B C D)
  (h2 : split_internal_angles F G C D) :
  A + B + C + D + F + G = 360 :=
sorry

end sum_of_angles_equal_360_l458_45818


namespace even_expression_l458_45808

theorem even_expression (m n : ℤ) (hm : Odd m) (hn : Odd n) : Even (m + 5 * n) :=
by
  sorry

end even_expression_l458_45808


namespace inequality_with_sum_one_l458_45892

theorem inequality_with_sum_one
  (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + b = 1)
  (x y : ℝ) :
  (a * x + b * y) * (b * x + a * y) ≥ x * y :=
by
  sorry

end inequality_with_sum_one_l458_45892


namespace total_spent_correct_l458_45880

def cost_ornamental_plants : Float := 467.00
def cost_garden_tool_set : Float := 85.00
def cost_potting_soil : Float := 38.00

def discount_plants : Float := 0.15
def discount_tools : Float := 0.10
def discount_soil : Float := 0.00

def sales_tax_rate : Float := 0.08
def surcharge : Float := 12.00

def discounted_price (original_price : Float) (discount_rate : Float) : Float :=
  original_price * (1.0 - discount_rate)

def subtotal (price_plants : Float) (price_tools : Float) (price_soil : Float) : Float :=
  price_plants + price_tools + price_soil

def sales_tax (amount : Float) (tax_rate : Float) : Float :=
  amount * tax_rate

def total (subtotal : Float) (sales_tax : Float) (surcharge : Float) : Float :=
  subtotal + sales_tax + surcharge

def final_total_spent : Float :=
  let price_plants := discounted_price cost_ornamental_plants discount_plants
  let price_tools := discounted_price cost_garden_tool_set discount_tools
  let price_soil := cost_potting_soil
  let subtotal_amount := subtotal price_plants price_tools price_soil
  let tax_amount := sales_tax subtotal_amount sales_tax_rate
  total subtotal_amount tax_amount surcharge

theorem total_spent_correct : final_total_spent = 564.37 :=
  by sorry

end total_spent_correct_l458_45880


namespace find_a9_l458_45890

noncomputable def polynomial_coefficients : Prop :=
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℤ),
  ∀ (x : ℤ),
    x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8 + x^9 + x^10 =
    a₀ + a₁ * (1 + x) + a₂ * (1 + x)^2 + a₃ * (1 + x)^3 + a₄ * (1 + x)^4 + 
    a₅ * (1 + x)^5 + a₆ * (1 + x)^6 + a₇ * (1 + x)^7 + a₈ * (1 + x)^8 + 
    a₉ * (1 + x)^9 + a₁₀ * (1 + x)^10

theorem find_a9 (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℤ) (h : polynomial_coefficients) : a₉ = -9 := by
  sorry

end find_a9_l458_45890


namespace angle_C_is_65_deg_l458_45815

-- Defining a triangle and its angles.
structure Triangle :=
  (A B C : ℝ) -- representing the angles in degrees

-- Defining the conditions of the problem.
def given_triangle : Triangle :=
  { A := 75, B := 40, C := 180 - 75 - 40 }

-- Statement of the problem, proving that the measure of ∠C is 65°.
theorem angle_C_is_65_deg (t : Triangle) (hA : t.A = 75) (hB : t.B = 40) (hSum : t.A + t.B + t.C = 180) : t.C = 65 :=
  by sorry

end angle_C_is_65_deg_l458_45815


namespace point_on_x_axis_l458_45848

theorem point_on_x_axis (a : ℝ) (h : a + 2 = 0) : (a - 1, a + 2) = (-3, 0) :=
by
  sorry

end point_on_x_axis_l458_45848


namespace linear_eq_k_l458_45864

theorem linear_eq_k (k : ℕ) : (∀ x : ℝ, x^(k-1) + 3 = 0 ↔ k = 2) :=
by
  sorry

end linear_eq_k_l458_45864


namespace min_value_of_expression_l458_45828

noncomputable def minValue (a : ℝ) : ℝ :=
  1 / (3 - 2 * a) + 2 / (a - 1)

theorem min_value_of_expression : ∀ a : ℝ, 1 < a ∧ a < 3 / 2 → (1 / (3 - 2 * a) + 2 / (a - 1)) ≥ 16 / 9 :=
by
  intro a h
  sorry

end min_value_of_expression_l458_45828


namespace arithmetic_sequence_sum_l458_45844

theorem arithmetic_sequence_sum (a1 d : ℝ)
  (h1 : a1 + 11 * d = -8)
  (h2 : 9 / 2 * (a1 + (a1 + 8 * d)) = -9) :
  16 / 2 * (a1 + (a1 + 15 * d)) = -72 := by
  sorry

end arithmetic_sequence_sum_l458_45844


namespace portion_of_pizza_eaten_l458_45862

-- Define the conditions
def total_slices : ℕ := 16
def slices_left : ℕ := 4
def slices_eaten : ℕ := total_slices - slices_left

-- Define the portion of pizza eaten
def portion_eaten := (slices_eaten : ℚ) / (total_slices : ℚ)

-- Statement to prove
theorem portion_of_pizza_eaten : portion_eaten = 3 / 4 :=
by sorry

end portion_of_pizza_eaten_l458_45862


namespace additional_weekly_rate_l458_45881

theorem additional_weekly_rate (rate_first_week : ℝ) (total_days_cost : ℝ) (days_first_week : ℕ) (total_days : ℕ) (cost_total : ℝ) (cost_first_week : ℝ) (days_after_first_week : ℕ) : 
  (rate_first_week * days_first_week = cost_first_week) → 
  (total_days = days_first_week + days_after_first_week) → 
  (cost_total = cost_first_week + (days_after_first_week * (rate_first_week * 7 / days_first_week))) →
  (rate_first_week = 18) →
  (cost_total = 350) →
  total_days = 23 → 
  (days_first_week = 7) → 
  cost_first_week = 126 →
  (days_after_first_week = 16) →
  rate_first_week * 7 / days_first_week * days_after_first_week = 14 := 
by 
  sorry

end additional_weekly_rate_l458_45881


namespace real_roots_exist_l458_45800

theorem real_roots_exist (K : ℝ) : 
  ∃ x : ℝ, x = K^2 * (x - 1) * (x - 3) :=
by
  sorry  -- Proof goes here

end real_roots_exist_l458_45800


namespace rationalize_denominator_ABC_value_l458_45814

def A := 11 / 4
def B := 5 / 4
def C := 5

theorem rationalize_denominator : 
  (2 + Real.sqrt 5) / (3 - Real.sqrt 5) = A + B * Real.sqrt C :=
sorry

theorem ABC_value :
  A * B * C = 275 :=
sorry

end rationalize_denominator_ABC_value_l458_45814


namespace two_point_line_l458_45804

theorem two_point_line (k b : ℝ) (h_k : k ≠ 0) :
  (∀ (x y : ℝ), (y = k * x + b → (x, y) = (0, 0) ∨ (x, y) = (1, 1))) →
  (∀ (x y : ℝ), (y = k * x + b → (x, y) ≠ (2, 0))) :=
by
  sorry

end two_point_line_l458_45804


namespace central_angle_of_regular_polygon_l458_45857

theorem central_angle_of_regular_polygon (n : ℕ) (h : 360 ∣ 360 - 36 * n) :
  n = 10 :=
by
  sorry

end central_angle_of_regular_polygon_l458_45857


namespace solve_for_p_l458_45891

theorem solve_for_p (q p : ℝ) (h : p^2 * q = p * q + p^2) : 
  p = 0 ∨ p = q / (q - 1) :=
by
  sorry

end solve_for_p_l458_45891


namespace total_bill_is_270_l458_45883

-- Conditions as Lean definitions
def totalBill (T : ℝ) : Prop :=
  let eachShare := T / 10
  9 * (eachShare + 3) = T

-- Theorem stating that the total bill T is 270
theorem total_bill_is_270 (T : ℝ) (h : totalBill T) : T = 270 :=
sorry

end total_bill_is_270_l458_45883


namespace book_price_l458_45803

theorem book_price (n p : ℕ) (h : n * p = 104) (hn : 10 < n ∧ n < 60) : p = 2 ∨ p = 4 ∨ p = 8 :=
sorry

end book_price_l458_45803


namespace mario_oranges_l458_45812

theorem mario_oranges (M L N T x : ℕ) 
  (H_L : L = 24) 
  (H_N : N = 96) 
  (H_T : T = 128) 
  (H_total : x + L + N = T) : 
  x = 8 :=
by
  rw [H_L, H_N, H_T] at H_total
  linarith

end mario_oranges_l458_45812


namespace simplify_complex_expression_l458_45853

theorem simplify_complex_expression :
  (1 / (-8 ^ 2) ^ 4 * (-8) ^ 9) = -8 := by
  sorry

end simplify_complex_expression_l458_45853


namespace area_EYH_trapezoid_l458_45878

theorem area_EYH_trapezoid (EF GH : ℕ) (EF_len : EF = 15) (GH_len : GH = 35) 
(Area_trapezoid : (EF + GH) * 16 / 2 = 400) : 
∃ (EYH_area : ℕ), EYH_area = 84 := by
  sorry

end area_EYH_trapezoid_l458_45878


namespace smallest_b_for_factoring_l458_45859

theorem smallest_b_for_factoring (b : ℕ) (p q : ℕ) (h1 : p * q = 1800) (h2 : p + q = b) : b = 85 :=
by
  sorry

end smallest_b_for_factoring_l458_45859


namespace students_not_make_cut_l458_45827

theorem students_not_make_cut (girls boys called_back : ℕ) 
  (h_girls : girls = 42) (h_boys : boys = 80)
  (h_called_back : called_back = 25) : 
  (girls + boys - called_back = 97) := by
  sorry

end students_not_make_cut_l458_45827


namespace expand_binomials_l458_45838

variable {x y : ℝ}

theorem expand_binomials (x y : ℝ) : 
  (x + 5) * (3 * y + 15) = 3 * x * y + 15 * x + 15 * y + 75 := 
by
  sorry

end expand_binomials_l458_45838


namespace min_value_x_plus_y_l458_45860

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 / y + 1 / x = 4) :
  x + y ≥ 9 / 4 :=
sorry

end min_value_x_plus_y_l458_45860


namespace simplify_expression_l458_45852

theorem simplify_expression : 1 - (1 / (2 + Real.sqrt 5)) + (1 / (2 - Real.sqrt 5)) = 1 - 2 * Real.sqrt 5 :=
by
  sorry

end simplify_expression_l458_45852


namespace mixed_water_temp_l458_45855

def cold_water_temp : ℝ := 20   -- Temperature of cold water
def hot_water_temp : ℝ := 40    -- Temperature of hot water

theorem mixed_water_temp :
  (cold_water_temp + hot_water_temp) / 2 = 30 := 
by sorry

end mixed_water_temp_l458_45855


namespace proof_equiv_expression_l458_45889

variable (x y : ℝ)

def P : ℝ := x^2 + y^2
def Q : ℝ := x^2 - y^2

theorem proof_equiv_expression :
  ( (P x y)^2 + (Q x y)^2 ) / ( (P x y)^2 - (Q x y)^2 ) - 
  ( (P x y)^2 - (Q x y)^2 ) / ( (P x y)^2 + (Q x y)^2 ) = 
  (x^4 - y^4) / (x^2 * y^2) :=
by
  sorry

end proof_equiv_expression_l458_45889


namespace farmer_children_l458_45826

theorem farmer_children (n : ℕ) 
  (h1 : 15 * n - 8 - 7 = 60) : n = 5 := 
by
  sorry

end farmer_children_l458_45826


namespace prove_cuboid_properties_l458_45894

noncomputable def cuboid_length := 5
noncomputable def cuboid_width := 4
noncomputable def cuboid_height := 3

theorem prove_cuboid_properties :
  (min (cuboid_length * cuboid_width) (min (cuboid_length * cuboid_height) (cuboid_width * cuboid_height)) = 12) ∧
  (max (cuboid_length * cuboid_width) (max (cuboid_length * cuboid_height) (cuboid_width * cuboid_height)) = 20) ∧
  ((cuboid_length + cuboid_width + cuboid_height) * 4 = 48) ∧
  (2 * (cuboid_length * cuboid_width + cuboid_length * cuboid_height + cuboid_width * cuboid_height) = 94) ∧
  (cuboid_length * cuboid_width * cuboid_height = 60) :=
by
  sorry

end prove_cuboid_properties_l458_45894


namespace sqrt_expression_l458_45832

theorem sqrt_expression (h : n < m ∧ m < 0) : 
  (Real.sqrt (m^2 + 2 * m * n + n^2) - Real.sqrt (m^2 - 2 * m * n + n^2)) = -2 * m := 
by {
  sorry
}

end sqrt_expression_l458_45832


namespace arithmetic_sequence_y_value_l458_45846

theorem arithmetic_sequence_y_value (y : ℝ) (h₁ : 2 * y - 3 = -5 * y + 11) : y = 2 := by
  sorry

end arithmetic_sequence_y_value_l458_45846


namespace sequence_general_formula_l458_45819

theorem sequence_general_formula (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, a (n + 1) = 3 * a n + 2 * n - 1) :
  ∀ n : ℕ, a n = (2 / 3) * 3^n - n :=
by
  sorry

end sequence_general_formula_l458_45819


namespace simple_interest_principal_l458_45874

theorem simple_interest_principal (R T SI : ℝ) (hR : R = 9 / 100) (hT : T = 1) (hSI : SI = 900) : 
  (SI / (R * T) = 10000) :=
by
  sorry

end simple_interest_principal_l458_45874


namespace beneficiary_received_32_176_l458_45847

noncomputable def A : ℝ := 19520 / 0.728
noncomputable def B : ℝ := 1.20 * A
noncomputable def C : ℝ := 1.44 * A
noncomputable def D : ℝ := 1.728 * A

theorem beneficiary_received_32_176 :
    round B = 32176 :=
by
    sorry

end beneficiary_received_32_176_l458_45847


namespace mutually_exclusive_event_l458_45876

def Event := String  -- define a simple type for events

/-- Define the events -/
def at_most_one_hit : Event := "at most one hit"
def two_hits : Event := "two hits"

/-- Define a function to check mutual exclusiveness -/
def mutually_exclusive (e1 e2 : Event) : Prop := 
  e1 ≠ e2

theorem mutually_exclusive_event :
  mutually_exclusive at_most_one_hit two_hits :=
by
  sorry

end mutually_exclusive_event_l458_45876


namespace tshirt_cost_l458_45836

-- Definitions based on conditions
def pants_cost : ℝ := 80
def shoes_cost : ℝ := 150
def discount : ℝ := 0.1
def total_paid : ℝ := 558

-- Variables based on the problem
variable (T : ℝ) -- Cost of one T-shirt
def num_tshirts : ℝ := 4
def num_pants : ℝ := 3
def num_shoes : ℝ := 2

-- Theorem: The cost of one T-shirt is $20
theorem tshirt_cost : T = 20 :=
by
  have total_cost : ℝ := (num_tshirts * T) + (num_pants * pants_cost) + (num_shoes * shoes_cost)
  have discounted_total : ℝ := (1 - discount) * total_cost
  have payment_condition : discounted_total = total_paid := sorry
  sorry -- detailed proof

end tshirt_cost_l458_45836


namespace asymptotes_of_hyperbola_l458_45869

theorem asymptotes_of_hyperbola (x y : ℝ) :
  (x ^ 2 / 4 - y ^ 2 / 9 = -1) →
  (y = (3 / 2) * x ∨ y = -(3 / 2) * x) :=
sorry

end asymptotes_of_hyperbola_l458_45869


namespace difference_of_averages_l458_45801

theorem difference_of_averages :
  let avg1 := (20 + 40 + 60) / 3
  let avg2 := (10 + 70 + 16) / 3
  avg1 - avg2 = 8 :=
by
  sorry

end difference_of_averages_l458_45801


namespace current_at_resistance_12_l458_45899

theorem current_at_resistance_12 (R : ℝ) (I : ℝ) (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
  sorry

end current_at_resistance_12_l458_45899


namespace difference_of_coordinates_l458_45824

-- Define point and its properties in Lean.
structure Point where
  x : ℝ
  y : ℝ

-- Define the midpoint property.
def is_midpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

-- Given points A and M
def A : Point := {x := 8, y := 0}
def M : Point := {x := 4, y := 1}

-- Assume B is a point with coordinates x and y
variable (B : Point)

-- The theorem to prove.
theorem difference_of_coordinates :
  is_midpoint M A B → B.x - B.y = -2 :=
by
  sorry

end difference_of_coordinates_l458_45824
