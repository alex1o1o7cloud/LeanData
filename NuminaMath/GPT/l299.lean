import Mathlib

namespace NUMINAMATH_GPT_initial_birds_count_l299_29972

theorem initial_birds_count (current_total_birds birds_joined initial_birds : ℕ) 
  (h1 : current_total_birds = 6) 
  (h2 : birds_joined = 4) : 
  initial_birds = current_total_birds - birds_joined → 
  initial_birds = 2 :=
by 
  intro h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_initial_birds_count_l299_29972


namespace NUMINAMATH_GPT_alpha_beta_square_l299_29953

theorem alpha_beta_square (α β : ℝ) (h₁ : α^2 = 2*α + 1) (h₂ : β^2 = 2*β + 1) (hαβ : α ≠ β) :
  (α - β)^2 = 8 := 
sorry

end NUMINAMATH_GPT_alpha_beta_square_l299_29953


namespace NUMINAMATH_GPT_maximum_value_l299_29921

theorem maximum_value (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a^2 + b^2 + c^2 = 1) :
  3 * a * b * Real.sqrt 2 + 6 * b * c ≤ 4.5 :=
sorry

end NUMINAMATH_GPT_maximum_value_l299_29921


namespace NUMINAMATH_GPT_ratio_w_y_l299_29931

-- Define the necessary variables
variables (w x y z : ℚ)

-- Define the conditions as hypotheses
axiom h1 : w / x = 4 / 3
axiom h2 : y / z = 5 / 3
axiom h3 : z / x = 1 / 6

-- State the proof problem
theorem ratio_w_y : w / y = 24 / 5 :=
by sorry

end NUMINAMATH_GPT_ratio_w_y_l299_29931


namespace NUMINAMATH_GPT_value_of_4_Y_3_eq_neg23_l299_29942

def my_operation (a b : ℝ) (c : ℝ) : ℝ := a^2 - 2 * a * b * c + b^2

theorem value_of_4_Y_3_eq_neg23 : my_operation 4 3 2 = -23 := by
  sorry

end NUMINAMATH_GPT_value_of_4_Y_3_eq_neg23_l299_29942


namespace NUMINAMATH_GPT_find_set_M_l299_29916

variable (U : Set ℕ) (M : Set ℕ)

def isUniversalSet : Prop := U = {1, 2, 3, 4, 5, 6}
def isComplement : Prop := U \ M = {1, 2, 4}

theorem find_set_M (hU : isUniversalSet U) (hC : isComplement U M) : M = {3, 5, 6} :=
  sorry

end NUMINAMATH_GPT_find_set_M_l299_29916


namespace NUMINAMATH_GPT_minimalBananasTotal_is_408_l299_29933

noncomputable def minimalBananasTotal : ℕ :=
  let b₁ := 11 * 8
  let b₂ := 13 * 8
  let b₃ := 27 * 8
  b₁ + b₂ + b₃

theorem minimalBananasTotal_is_408 : minimalBananasTotal = 408 := by
  sorry

end NUMINAMATH_GPT_minimalBananasTotal_is_408_l299_29933


namespace NUMINAMATH_GPT_problem_l299_29956

theorem problem (triangle square : ℕ) (h1 : triangle + 5 ≡ 1 [MOD 7]) (h2 : 2 + square ≡ 3 [MOD 7]) :
  triangle = 3 ∧ square = 1 := by
  sorry

end NUMINAMATH_GPT_problem_l299_29956


namespace NUMINAMATH_GPT_min_value_y_l299_29913

theorem min_value_y (x y : ℝ) (h : x^2 + y^2 = 14 * x + 48 * y) : y = -1 := 
sorry

end NUMINAMATH_GPT_min_value_y_l299_29913


namespace NUMINAMATH_GPT_empty_drainpipe_rate_l299_29965

theorem empty_drainpipe_rate :
  (∀ x : ℝ, (1/5 + 1/4 - 1/x = 1/2.5) → x = 20) :=
by 
    intro x
    intro h
    sorry -- Proof is omitted, only the statement is required

end NUMINAMATH_GPT_empty_drainpipe_rate_l299_29965


namespace NUMINAMATH_GPT_solution_for_4_minus_c_l299_29960

-- Define the conditions as Lean hypotheses
theorem solution_for_4_minus_c (c d : ℚ) (h1 : 4 + c = 5 - d) (h2 : 5 + d = 9 + c) : 4 - c = 11 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_for_4_minus_c_l299_29960


namespace NUMINAMATH_GPT_smallest_k_for_bisectors_l299_29957

theorem smallest_k_for_bisectors (a b c l_a l_b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : l_a = (2 * b * c * Real.sqrt ((1 + (b^2 + c^2 - a^2) / (2 * b * c)) / 2)) / (b + c))
  (h5 : l_b = (2 * a * c * Real.sqrt ((1 + (a^2 + c^2 - b^2) / (2 * a * c)) / 2)) / (a + c)) :
  (l_a + l_b) / (a + b) ≤ 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_for_bisectors_l299_29957


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_l299_29992

def f (x : ℝ) : ℝ := (x - 3)^3 + x - 1

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ)
  (h_arith : is_arithmetic_sequence a d)
  (h_nonzero : d ≠ 0)
  (h_sum_f : f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) + f (a 7) = 14) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 21 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_l299_29992


namespace NUMINAMATH_GPT_volume_tetrahedron_l299_29975

def A1 := 4^2
def A2 := 3^2
def h := 1

theorem volume_tetrahedron:
  (h / 3 * (A1 + A2 + Real.sqrt (A1 * A2))) = 37 / 3 := by
  sorry

end NUMINAMATH_GPT_volume_tetrahedron_l299_29975


namespace NUMINAMATH_GPT_angle_turned_by_hour_hand_l299_29948

theorem angle_turned_by_hour_hand (rotation_degrees_per_hour : ℝ) (total_degrees_per_rotation : ℝ) :
  rotation_degrees_per_hour * 1 = -30 :=
by
  have rotation_degrees_per_hour := - total_degrees_per_rotation / 12
  have total_degrees_per_rotation := 360
  sorry

end NUMINAMATH_GPT_angle_turned_by_hour_hand_l299_29948


namespace NUMINAMATH_GPT_find_original_rabbits_l299_29985

theorem find_original_rabbits (R S : ℕ) (h1 : R + S = 50)
  (h2 : 4 * R + 8 * S = 2 * R + 16 * S) :
  R = 40 :=
sorry

end NUMINAMATH_GPT_find_original_rabbits_l299_29985


namespace NUMINAMATH_GPT_cricket_team_captain_age_l299_29995

theorem cricket_team_captain_age
    (C W : ℕ)
    (h1 : W = C + 3)
    (h2 : (23 * 11) = (22 * 9) + C + W)
    : C = 26 :=
by
    sorry

end NUMINAMATH_GPT_cricket_team_captain_age_l299_29995


namespace NUMINAMATH_GPT_white_ball_probability_l299_29974

theorem white_ball_probability :
  ∀ (n : ℕ), (2/(n+2) = 2/5) → (n = 3) → (n/(n+2) = 3/5) :=
by
  sorry

end NUMINAMATH_GPT_white_ball_probability_l299_29974


namespace NUMINAMATH_GPT_percentage_increase_in_side_of_square_l299_29920

theorem percentage_increase_in_side_of_square (p : ℝ) : 
  (1 + p / 100) ^ 2 = 1.3225 → 
  p = 15 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_in_side_of_square_l299_29920


namespace NUMINAMATH_GPT_leaks_empty_time_l299_29940

theorem leaks_empty_time (A L1 L2: ℝ) (hA: A = 1/2) (hL1_rate: A - L1 = 1/3) 
  (hL2_rate: A - L1 - L2 = 1/4) : 1 / (L1 + L2) = 4 :=
by
  sorry

end NUMINAMATH_GPT_leaks_empty_time_l299_29940


namespace NUMINAMATH_GPT_intersection_eq_l299_29976

def A := {x : ℝ | |x| = x}
def B := {x : ℝ | x^2 + x ≥ 0}

theorem intersection_eq : A ∩ B = {x : ℝ | 0 ≤ x} := by
  sorry

end NUMINAMATH_GPT_intersection_eq_l299_29976


namespace NUMINAMATH_GPT_dan_initial_amount_l299_29984

variables (initial_amount spent_amount remaining_amount : ℝ)

theorem dan_initial_amount (h1 : spent_amount = 1) (h2 : remaining_amount = 2) : initial_amount = spent_amount + remaining_amount := by
  sorry

end NUMINAMATH_GPT_dan_initial_amount_l299_29984


namespace NUMINAMATH_GPT_correct_point_on_hyperbola_l299_29970

-- Given condition
def hyperbola_condition (x y : ℝ) : Prop := x * y = -4

-- Question (translated to a mathematically equivalent proof)
theorem correct_point_on_hyperbola :
  hyperbola_condition (-2) 2 :=
sorry

end NUMINAMATH_GPT_correct_point_on_hyperbola_l299_29970


namespace NUMINAMATH_GPT_find_f_value_l299_29943

noncomputable def f (a b x : ℝ) : ℝ := a * (Real.cos x)^2 - b * (Real.sin x) * (Real.cos x) - a / 2

theorem find_f_value (a b : ℝ)
  (h_max : ∀ x, f a b x ≤ 1/2)
  (h_at_pi_over_3 : f a b (Real.pi / 3) = (Real.sqrt 3) / 4) :
  f a b (-Real.pi / 3) = 0 ∨ f a b (-Real.pi / 3) = -(Real.sqrt 3) / 4 :=
sorry

end NUMINAMATH_GPT_find_f_value_l299_29943


namespace NUMINAMATH_GPT_problem_eq_solution_l299_29901

variables (a b x y : ℝ)

theorem problem_eq_solution
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < x) (h4 : 0 < y)
  (h5 : a + b + x + y < 2)
  (h6 : a + b^2 = x + y^2)
  (h7 : a^2 + b = x^2 + y) :
  a = x ∧ b = y :=
by
  sorry

end NUMINAMATH_GPT_problem_eq_solution_l299_29901


namespace NUMINAMATH_GPT_interest_rate_calculation_l299_29978

-- Define the problem conditions and proof statement in Lean
theorem interest_rate_calculation 
  (P : ℝ) (r : ℝ) (T : ℝ) (CI SI diff : ℝ) 
  (principal_condition : P = 6000.000000000128)
  (time_condition : T = 2)
  (diff_condition : diff = 15)
  (CI_formula : CI = P * (1 + r)^T - P)
  (SI_formula : SI = P * r * T)
  (difference_condition : CI - SI = diff) : 
  r = 0.05 := 
by 
  sorry

end NUMINAMATH_GPT_interest_rate_calculation_l299_29978


namespace NUMINAMATH_GPT_rob_nickels_count_l299_29951

noncomputable def value_of_quarters (num_quarters : ℕ) : ℝ := num_quarters * 0.25
noncomputable def value_of_dimes (num_dimes : ℕ) : ℝ := num_dimes * 0.10
noncomputable def value_of_pennies (num_pennies : ℕ) : ℝ := num_pennies * 0.01
noncomputable def value_of_nickels (num_nickels : ℕ) : ℝ := num_nickels * 0.05

theorem rob_nickels_count :
  let quarters := 7
  let dimes := 3
  let pennies := 12
  let total := 2.42
  let nickels := 5
  value_of_quarters quarters + value_of_dimes dimes + value_of_pennies pennies + value_of_nickels nickels = total :=
by
  sorry

end NUMINAMATH_GPT_rob_nickels_count_l299_29951


namespace NUMINAMATH_GPT_probability_compare_l299_29981

-- Conditions
def v : ℝ := 0.1
def n : ℕ := 998

-- Binomial distribution formula
noncomputable def binom_prob (n k : ℕ) (v : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * (v ^ k) * ((1 - v) ^ (n - k))

-- Theorem to prove
theorem probability_compare :
  binom_prob n 99 v > binom_prob n 100 v :=
by
  sorry

end NUMINAMATH_GPT_probability_compare_l299_29981


namespace NUMINAMATH_GPT_minimum_a_l299_29962

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (2 - x)

theorem minimum_a
  (a : ℝ)
  (h : ∀ x : ℤ, (f x)^2 - a * f x ≤ 0 → ∃! x : ℤ, (f x)^2 - a * f x = 0) :
  a = Real.exp 2 + 1 :=
sorry

end NUMINAMATH_GPT_minimum_a_l299_29962


namespace NUMINAMATH_GPT_smallest_w_l299_29912

theorem smallest_w (w : ℕ) (w_pos : 0 < w) : 
  (∀ n : ℕ, (2^5 ∣ 936 * n) ∧ (3^3 ∣ 936 * n) ∧ (11^2 ∣ 936 * n) ↔ n = w) → w = 4356 :=
sorry

end NUMINAMATH_GPT_smallest_w_l299_29912


namespace NUMINAMATH_GPT_chapters_per_day_l299_29993

theorem chapters_per_day (chapters : ℕ) (total_days : ℕ) : ℝ :=
  let chapters := 2
  let total_days := 664
  chapters / total_days

example : chapters_per_day 2 664 = 2 / 664 := by sorry

end NUMINAMATH_GPT_chapters_per_day_l299_29993


namespace NUMINAMATH_GPT_fraction_equivalence_l299_29967

theorem fraction_equivalence :
  ( (3 / 7 + 2 / 3) / (5 / 11 + 3 / 8) ) = (119 / 90) :=
by
  sorry

end NUMINAMATH_GPT_fraction_equivalence_l299_29967


namespace NUMINAMATH_GPT_possible_values_quotient_l299_29955

theorem possible_values_quotient (α : ℝ) (h_pos : α > 0) (h_rounded : ∃ (n : ℕ) (α1 : ℝ), α = n / 100 + α1 ∧ 0 ≤ α1 ∧ α1 < 1 / 100) :
  ∃ (values : List ℝ), values = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59,
                                  0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
                                  0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79,
                                  0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
                                  0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
                                  1.00] :=
  sorry

end NUMINAMATH_GPT_possible_values_quotient_l299_29955


namespace NUMINAMATH_GPT_copy_is_better_l299_29905

variable (α : ℝ)

noncomputable def p_random : ℝ := 1 / 2
noncomputable def I_mistake : ℝ := α
noncomputable def p_caught : ℝ := 1 / 10
noncomputable def I_caught : ℝ := 3 * α
noncomputable def p_neighbor_wrong : ℝ := 1 / 5
noncomputable def p_not_caught : ℝ := 9 / 10

theorem copy_is_better (α : ℝ) : 
  (12 * α / 25) < (α / 2) := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_copy_is_better_l299_29905


namespace NUMINAMATH_GPT_radiator_initial_fluid_l299_29998

theorem radiator_initial_fluid (x : ℝ)
  (h1 : (0.10 * x - 0.10 * 2.2857 + 0.80 * 2.2857) = 0.50 * x) :
  x = 4 :=
sorry

end NUMINAMATH_GPT_radiator_initial_fluid_l299_29998


namespace NUMINAMATH_GPT_number_wall_top_block_value_l299_29963

theorem number_wall_top_block_value (a b c d : ℕ) 
    (h1 : a = 8) (h2 : b = 5) (h3 : c = 3) (h4 : d = 2) : 
    (a + b + (b + c) + (c + d) = 34) :=
by
  sorry

end NUMINAMATH_GPT_number_wall_top_block_value_l299_29963


namespace NUMINAMATH_GPT_wedge_volume_cylinder_l299_29934

theorem wedge_volume_cylinder (r h : ℝ) (theta : ℝ) (V : ℝ) 
  (hr : r = 6) (hh : h = 6) (htheta : theta = 60) (hV : V = 113) : 
  V = (theta / 360) * π * r^2 * h :=
by
  sorry

end NUMINAMATH_GPT_wedge_volume_cylinder_l299_29934


namespace NUMINAMATH_GPT_central_angle_nonagon_l299_29945

theorem central_angle_nonagon : (360 / 9 = 40) :=
by
  sorry

end NUMINAMATH_GPT_central_angle_nonagon_l299_29945


namespace NUMINAMATH_GPT_smallest_value_is_nine_l299_29991

noncomputable def smallest_possible_value (a b c d : ℝ) : ℝ :=
  (⌊(a + b + c) / d⌋ + ⌊(b + c + d) / a⌋ + ⌊(c + d + a) / b⌋ + ⌊(d + a + b) / c⌋ : ℝ)

theorem smallest_value_is_nine {a b c d : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  smallest_possible_value a b c d = 9 :=
sorry

end NUMINAMATH_GPT_smallest_value_is_nine_l299_29991


namespace NUMINAMATH_GPT_solve_AlyoshaCube_l299_29924

noncomputable def AlyoshaCubeSplit (n s : ℕ) : Prop :=
  n^3 - s^3 = 152 ∧ n > s

theorem solve_AlyoshaCube : ∃ n, ∃ s : ℕ, AlyoshaCubeSplit n s ∧ n = 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_AlyoshaCube_l299_29924


namespace NUMINAMATH_GPT_polynomial_remainder_l299_29977

theorem polynomial_remainder (x : ℝ) :
  ∃ (Q : ℝ → ℝ) (a b : ℝ),
    (x^150 = (x^2 - 5*x + 6) * Q x + (a*x + b)) ∧
    (2 * a + b = 2^150) ∧
    (3 * a + b = 3^150) ∧ 
    (a = 3^150 - 2^150) ∧ 
    (b = 2^150 - 2 * 3^150 + 2 * 2^150) := sorry

end NUMINAMATH_GPT_polynomial_remainder_l299_29977


namespace NUMINAMATH_GPT_rocket_max_speed_l299_29918

theorem rocket_max_speed (M m : ℝ) (h : 2000 * Real.log (1 + M / m) = 12000) : 
  M / m = Real.exp 6 - 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_rocket_max_speed_l299_29918


namespace NUMINAMATH_GPT_quadratic_one_positive_root_l299_29997

theorem quadratic_one_positive_root (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, y ∈ {t | t^2 - a * t + a - 2 = 0} → y = x)) → a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_one_positive_root_l299_29997


namespace NUMINAMATH_GPT_sum_of_solutions_eq_3_l299_29986

theorem sum_of_solutions_eq_3 (x y : ℝ) (h1 : x * y = 1) (h2 : x + y = 3) :
  x + y = 3 := sorry

end NUMINAMATH_GPT_sum_of_solutions_eq_3_l299_29986


namespace NUMINAMATH_GPT_amount_earned_from_each_family_l299_29982

theorem amount_earned_from_each_family
  (goal : ℕ) (earn_from_fifteen_families : ℕ) (additional_needed : ℕ) (three_families : ℕ) 
  (earn_from_three_families_total : ℕ) (per_family_earn : ℕ) :
  goal = 150 →
  earn_from_fifteen_families = 75 →
  additional_needed = 45 →
  three_families = 3 →
  earn_from_three_families_total = (goal - additional_needed) - earn_from_fifteen_families →
  per_family_earn = earn_from_three_families_total / three_families →
  per_family_earn = 10 :=
by
  sorry

end NUMINAMATH_GPT_amount_earned_from_each_family_l299_29982


namespace NUMINAMATH_GPT_range_a_l299_29922

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then Real.log x / Real.log a else -2 * x + 8

theorem range_a (a : ℝ) (hf : ∀ x, f a x ≤ f a 2) :
  1 < a ∧ a ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_range_a_l299_29922


namespace NUMINAMATH_GPT_least_number_l299_29928

theorem least_number (n : ℕ) : 
  (n % 45 = 2) ∧ (n % 59 = 2) ∧ (n % 77 = 2) → n = 205517 :=
by
  sorry

end NUMINAMATH_GPT_least_number_l299_29928


namespace NUMINAMATH_GPT_series_sum_eq_l299_29914

noncomputable def series_sum : ℝ :=
  ∑' (k : ℕ), (k + 1) / (3 ^ (k + 1))

theorem series_sum_eq : series_sum = 3 / 4 :=
sorry

end NUMINAMATH_GPT_series_sum_eq_l299_29914


namespace NUMINAMATH_GPT_chess_tournament_games_l299_29937

theorem chess_tournament_games (n : ℕ) (h : n = 17) (k : n - 1 = 16) :
  (n * (n - 1)) / 2 = 136 := by
  sorry

end NUMINAMATH_GPT_chess_tournament_games_l299_29937


namespace NUMINAMATH_GPT_percentage_w_less_x_l299_29994

theorem percentage_w_less_x 
    (z : ℝ) 
    (y : ℝ) 
    (x : ℝ) 
    (w : ℝ) 
    (hy : y = 1.20 * z)
    (hx : x = 1.20 * y)
    (hw : w = 1.152 * z) 
    : (x - w) / x * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_w_less_x_l299_29994


namespace NUMINAMATH_GPT_triangle_perimeter_l299_29900

theorem triangle_perimeter (r A : ℝ) (p : ℝ)
  (h1 : r = 2.5)
  (h2 : A = 25)
  (h3 : A = r * (p / 2)) : 
  p = 20 :=
by 
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l299_29900


namespace NUMINAMATH_GPT_baking_time_correct_l299_29936

/-- Mark lets the bread rise for 120 minutes twice. -/
def rising_time : ℕ := 120 * 2

/-- Mark spends 10 minutes kneading the bread. -/
def kneading_time : ℕ := 10

/-- Total time taken to finish making the bread. -/
def total_time : ℕ := 280

/-- Calculate the baking time based on the given conditions. -/
def baking_time (rising kneading total : ℕ) : ℕ := total - (rising + kneading)

theorem baking_time_correct :
  baking_time rising_time kneading_time total_time = 30 := 
by 
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_baking_time_correct_l299_29936


namespace NUMINAMATH_GPT_largest_multiple_of_9_less_than_75_is_72_l299_29925

theorem largest_multiple_of_9_less_than_75_is_72 : 
  ∃ n : ℕ, 9 * n < 75 ∧ ∀ m : ℕ, 9 * m < 75 → 9 * m ≤ 9 * n :=
sorry

end NUMINAMATH_GPT_largest_multiple_of_9_less_than_75_is_72_l299_29925


namespace NUMINAMATH_GPT_Liza_reads_more_pages_than_Suzie_l299_29964

def Liza_reading_speed : ℕ := 20
def Suzie_reading_speed : ℕ := 15
def hours : ℕ := 3

theorem Liza_reads_more_pages_than_Suzie :
  Liza_reading_speed * hours - Suzie_reading_speed * hours = 15 := by
  sorry

end NUMINAMATH_GPT_Liza_reads_more_pages_than_Suzie_l299_29964


namespace NUMINAMATH_GPT_solution_is_D_l299_29917

-- Definitions of the equations
def eqA (x : ℝ) := 3 * x + 6 = 0
def eqB (x : ℝ) := 2 * x + 4 = 0
def eqC (x : ℝ) := (1 / 2) * x = -4
def eqD (x : ℝ) := 2 * x - 4 = 0

-- Theorem stating that only eqD has a solution x = 2
theorem solution_is_D : 
  ¬ eqA 2 ∧ ¬ eqB 2 ∧ ¬ eqC 2 ∧ eqD 2 := 
by
  sorry

end NUMINAMATH_GPT_solution_is_D_l299_29917


namespace NUMINAMATH_GPT_infinite_solutions_a_value_l299_29980

theorem infinite_solutions_a_value (a : ℝ) : 
  (∀ y : ℝ, 3 * (5 + a * y) = 15 * y + 9) ↔ a = 5 := 
by 
  sorry

end NUMINAMATH_GPT_infinite_solutions_a_value_l299_29980


namespace NUMINAMATH_GPT_ratio_dog_to_hamster_l299_29923

noncomputable def dog_lifespan : ℝ := 10
noncomputable def hamster_lifespan : ℝ := 2.5

theorem ratio_dog_to_hamster : dog_lifespan / hamster_lifespan = 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_dog_to_hamster_l299_29923


namespace NUMINAMATH_GPT_exist_divisible_number_l299_29915

theorem exist_divisible_number (d : ℕ) (hd : d > 0) :
  ∃ n : ℕ, (n % d = 0) ∧ ∃ k : ℕ, (k > 0) ∧ (k < 10) ∧ 
  ((∃ m : ℕ, m = n - k*(10^k / 10^k) ∧ m % d = 0) ∨ ∃ m : ℕ, m = n - k * (10^(k - 1)) ∧ m % d = 0) :=
sorry

end NUMINAMATH_GPT_exist_divisible_number_l299_29915


namespace NUMINAMATH_GPT_linear_eq_solution_l299_29987

theorem linear_eq_solution (m : ℤ) (x : ℝ) (h1 : |m| = 1) (h2 : 1 - m ≠ 0) : x = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_linear_eq_solution_l299_29987


namespace NUMINAMATH_GPT_price_comparison_l299_29929

variable (x y : ℝ)
variable (h1 : 6 * x + 3 * y > 24)
variable (h2 : 4 * x + 5 * y < 22)

theorem price_comparison : 2 * x > 3 * y :=
sorry

end NUMINAMATH_GPT_price_comparison_l299_29929


namespace NUMINAMATH_GPT_find_constant_a_find_ordinary_equation_of_curve_l299_29927

open Real

theorem find_constant_a (a t : ℝ) (h1 : 1 + 2 * t = 3) (h2 : a * t^2 = 1) : a = 1 :=
by
  -- Proof goes here
  sorry

theorem find_ordinary_equation_of_curve (x y t : ℝ) (h1 : x = 1 + 2 * t) (h2 : y = t^2) :
  (x - 1)^2 = 4 * y :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_constant_a_find_ordinary_equation_of_curve_l299_29927


namespace NUMINAMATH_GPT_Greg_PPO_Obtained_90_Percent_l299_29969

theorem Greg_PPO_Obtained_90_Percent :
  let max_procgen_reward := 240
  let max_coinrun_reward := max_procgen_reward / 2
  let greg_reward := 108
  (greg_reward / max_coinrun_reward * 100) = 90 := by
  sorry

end NUMINAMATH_GPT_Greg_PPO_Obtained_90_Percent_l299_29969


namespace NUMINAMATH_GPT_sum_1_to_50_l299_29909

-- Given conditions: initial values, and the loop increments
def initial_index : ℕ := 1
def initial_sum : ℕ := 0
def loop_condition (i : ℕ) : Prop := i ≤ 50

-- Increment step for index and running total in loop
def increment_index (i : ℕ) : ℕ := i + 1
def increment_sum (S : ℕ) (i : ℕ) : ℕ := S + i

-- Expected sum output for the given range
def sum_up_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Prove the sum of integers from 1 to 50
theorem sum_1_to_50 : sum_up_to_n 50 = 1275 := by
  sorry

end NUMINAMATH_GPT_sum_1_to_50_l299_29909


namespace NUMINAMATH_GPT_winning_probability_correct_l299_29919

-- Define the conditions
def numPowerBalls : ℕ := 30
def numLuckyBalls : ℕ := 49
def numChosenBalls : ℕ := 6

-- Define the probability of picking the correct PowerBall
def powerBallProb : ℚ := 1 / numPowerBalls

-- Define the combination function for choosing LuckyBalls
noncomputable def combination (n k : ℕ) : ℕ := n.choose k

-- Define the probability of picking the correct LuckyBalls
noncomputable def luckyBallProb : ℚ := 1 / (combination numLuckyBalls numChosenBalls)

-- Define the total winning probability
noncomputable def totalWinningProb : ℚ := powerBallProb * luckyBallProb

-- State the theorem to prove
theorem winning_probability_correct : totalWinningProb = 1 / 419512480 :=
by
  sorry

end NUMINAMATH_GPT_winning_probability_correct_l299_29919


namespace NUMINAMATH_GPT_positive_expression_l299_29907

variable (a b c d : ℝ)

theorem positive_expression (ha : a < b) (hb : b < 0) (hc : 0 < c) (hd : c < d) : d - c - b - a > 0 := 
sorry

end NUMINAMATH_GPT_positive_expression_l299_29907


namespace NUMINAMATH_GPT_rotations_per_block_l299_29958

/--
If Greg's bike wheels have already rotated 600 times and need to rotate 
1000 more times to reach his goal of riding at least 8 blocks,
then the number of rotations per block is 200.
-/
theorem rotations_per_block (r1 r2 n b : ℕ) (h1 : r1 = 600) (h2 : r2 = 1000) (h3 : n = 8) :
  (r1 + r2) / n = 200 := by
  sorry

end NUMINAMATH_GPT_rotations_per_block_l299_29958


namespace NUMINAMATH_GPT_max_x2_plus_2xy_plus_3y2_l299_29999

theorem max_x2_plus_2xy_plus_3y2 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - 2 * x * y + 3 * y^2 = 9) :
  x^2 + 2 * x * y + 3 * y^2 ≤ 18 + 9 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_max_x2_plus_2xy_plus_3y2_l299_29999


namespace NUMINAMATH_GPT_cannot_determine_right_triangle_from_conditions_l299_29903

-- Let triangle ABC have side lengths a, b, c opposite angles A, B, C respectively.
def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

-- Condition A: c^2 = a^2 - b^2 is rearranged to c^2 + b^2 = a^2 implying right triangle
def condition_A (a b c : ℝ) : Prop := c^2 = a^2 - b^2

-- Condition B: Triangle angles in the ratio A:B:C = 3:4:5 means not a right triangle
def condition_B : Prop := 
  let A := 45.0
  let B := 60.0
  let C := 75.0
  A ≠ 90.0 ∧ B ≠ 90.0 ∧ C ≠ 90.0

-- Condition C: Specific lengths 7, 24, 25 form a right triangle
def condition_C : Prop := 
  let a := 7.0
  let b := 24.0
  let c := 25.0
  is_right_triangle a b c

-- Condition D: A = B - C can be shown to always form at least one 90 degree angle, a right triangle
def condition_D (A B C : ℝ) : Prop := A = B - C ∧ (A + B + C = 180)

-- The actual mathematical proof that option B does not determine a right triangle
theorem cannot_determine_right_triangle_from_conditions :
  ∀ a b c (A B C : ℝ),
    (condition_A a b c → is_right_triangle a b c) ∧
    (condition_C → is_right_triangle 7 24 25) ∧
    (condition_D A B C → is_right_triangle a b c) ∧
    ¬condition_B :=
by
  sorry

end NUMINAMATH_GPT_cannot_determine_right_triangle_from_conditions_l299_29903


namespace NUMINAMATH_GPT_max_arithmetic_sequence_of_primes_less_than_150_l299_29935

theorem max_arithmetic_sequence_of_primes_less_than_150 : 
  ∀ (S : Finset ℕ), (∀ x ∈ S, Nat.Prime x) ∧ (∀ x ∈ S, x < 150) ∧ (∃ d, ∀ x ∈ S, ∃ n : ℕ, x = S.min' (by sorry) + n * d) → S.card ≤ 5 := 
by
  sorry

end NUMINAMATH_GPT_max_arithmetic_sequence_of_primes_less_than_150_l299_29935


namespace NUMINAMATH_GPT_number_of_benches_l299_29941

-- Define the conditions
def bench_capacity : ℕ := 4
def people_sitting : ℕ := 80
def available_spaces : ℕ := 120
def total_capacity : ℕ := people_sitting + available_spaces -- this equals 200

-- The theorem to prove the number of benches
theorem number_of_benches (B : ℕ) : bench_capacity * B = total_capacity → B = 50 :=
by
  intro h
  exact sorry

end NUMINAMATH_GPT_number_of_benches_l299_29941


namespace NUMINAMATH_GPT_wall_height_l299_29902

noncomputable def brickVolume (l w h : ℝ) : ℝ :=
  l * w * h

noncomputable def wallVolume (L W H : ℝ) : ℝ :=
  L * W * H

theorem wall_height (bricks_needed : ℝ) (brick_length_cm brick_width_cm brick_height_cm wall_length wall_width wall_height : ℝ)
  (H1 : bricks_needed = 4094.3396226415093)
  (H2 : brick_length_cm = 20)
  (H3 : brick_width_cm = 13.25)
  (H4 : brick_height_cm = 8)
  (H5 : wall_length = 7)
  (H6 : wall_width = 8)
  (H7 : brickVolume (brick_length_cm / 100) (brick_width_cm / 100) (brick_height_cm / 100) * bricks_needed = wallVolume wall_length wall_width wall_height) :
  wall_height = 0.155 :=
by
  sorry

end NUMINAMATH_GPT_wall_height_l299_29902


namespace NUMINAMATH_GPT_average_reading_time_correct_l299_29938

-- We define total_reading_time as a parameter representing the sum of reading times
noncomputable def total_reading_time : ℝ := sorry

-- We define the number of students as a constant
def number_of_students : ℕ := 50

-- We define the average reading time per student based on the provided data
noncomputable def average_reading_time : ℝ :=
  total_reading_time / number_of_students

-- The theorem we need to prove: that the average reading time per student is correctly calculated
theorem average_reading_time_correct :
  ∃ (total_reading_time : ℝ), average_reading_time = total_reading_time / number_of_students :=
by
  -- since total_reading_time and number_of_students are already defined, we prove the theorem using them
  use total_reading_time
  exact rfl

end NUMINAMATH_GPT_average_reading_time_correct_l299_29938


namespace NUMINAMATH_GPT_brian_spent_on_kiwis_l299_29983

theorem brian_spent_on_kiwis :
  ∀ (cost_per_dozen_apples : ℝ)
    (cost_for_24_apples : ℝ)
    (initial_money : ℝ)
    (subway_fare_one_way : ℝ)
    (total_remaining : ℝ)
    (kiwis_spent : ℝ)
    (bananas_spent : ℝ),
  cost_per_dozen_apples = 14 →
  cost_for_24_apples = 2 * cost_per_dozen_apples →
  initial_money = 50 →
  subway_fare_one_way = 3.5 →
  total_remaining = initial_money - 2 * subway_fare_one_way - cost_for_24_apples →
  total_remaining = 15 →
  bananas_spent = kiwis_spent / 2 →
  kiwis_spent + bananas_spent = total_remaining →
  kiwis_spent = 10 :=
by
  -- Sorry means we are skipping the proof
  sorry

end NUMINAMATH_GPT_brian_spent_on_kiwis_l299_29983


namespace NUMINAMATH_GPT_attendees_gift_exchange_l299_29910

theorem attendees_gift_exchange (x : ℕ) (h1 : 56 = x * (x - 1) / 2) : 
  x * (x - 1) = 112 :=
by
  sorry

end NUMINAMATH_GPT_attendees_gift_exchange_l299_29910


namespace NUMINAMATH_GPT_h_value_l299_29904

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ :=
  3 * x^2 + 9 * x + 20

-- State the desired form
def desired_form (x h k : ℝ) : ℝ :=
  3 * (x - h)^2 + k

-- Prove that h = -1.5
theorem h_value (h : ℝ) :
  ∃ k, (∀ x, quadratic_expr x = desired_form x h k) → h = -1.5 :=
by
  sorry

end NUMINAMATH_GPT_h_value_l299_29904


namespace NUMINAMATH_GPT_pos_numbers_equal_l299_29932

theorem pos_numbers_equal (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_eq : a^3 + b^3 + c^3 - 3 * a * b * c = 0) : a = b ∧ b = c :=
by
  sorry

end NUMINAMATH_GPT_pos_numbers_equal_l299_29932


namespace NUMINAMATH_GPT_inscribed_circle_radius_l299_29947

-- Conditions
variables {S A B C D O : Point} -- Points in 3D space
variables (AC : ℝ) (cos_SBD : ℝ)
variables (r : ℝ) -- Radius of inscribed circle

-- Given conditions
def AC_eq_one := AC = 1
def cos_angle_SBD := cos_SBD = 2/3

-- Assertion to be proved
theorem inscribed_circle_radius :
  AC_eq_one AC →
  cos_angle_SBD cos_SBD →
  (0 < r ∧ r ≤ 1/6) ∨ r = 1/3 :=
by
  intro hAC hcos
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l299_29947


namespace NUMINAMATH_GPT_fraction_equality_l299_29906

theorem fraction_equality (x y : ℚ) (hx : x = 4 / 7) (hy : y = 5 / 11) : 
  (7 * x + 11 * y) / (77 * x * y) = 9 / 20 :=
by
  -- proof can be provided here.
  sorry

end NUMINAMATH_GPT_fraction_equality_l299_29906


namespace NUMINAMATH_GPT_days_per_book_l299_29979

theorem days_per_book (total_books : ℕ) (total_days : ℕ)
  (h1 : total_books = 41)
  (h2 : total_days = 492) :
  total_days / total_books = 12 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_days_per_book_l299_29979


namespace NUMINAMATH_GPT_rectangles_same_area_l299_29908

theorem rectangles_same_area (x y : ℕ) 
  (h1 : x * y = (x + 4) * (y - 3)) 
  (h2 : x * y = (x + 8) * (y - 4)) : x + y = 10 := 
by
  sorry

end NUMINAMATH_GPT_rectangles_same_area_l299_29908


namespace NUMINAMATH_GPT_expand_polynomial_l299_29949

theorem expand_polynomial (x : ℝ) :
  (x + 4) * (x - 9) = x^2 - 5 * x - 36 := 
sorry

end NUMINAMATH_GPT_expand_polynomial_l299_29949


namespace NUMINAMATH_GPT_age_ratio_l299_29996

theorem age_ratio 
    (a m s : ℕ) 
    (h1 : m = 60) 
    (h2 : m = 3 * a) 
    (h3 : s = 40) : 
    (m + a) / s = 2 :=
by
    sorry

end NUMINAMATH_GPT_age_ratio_l299_29996


namespace NUMINAMATH_GPT_max_right_angles_in_triangle_l299_29961

theorem max_right_angles_in_triangle (a b c : ℝ) (h : a + b + c = 180) (ha : a = 90 ∨ b = 90 ∨ c = 90) : a = 90 ∧ b ≠ 90 ∧ c ≠ 90 ∨ b = 90 ∧ a ≠ 90 ∧ c ≠ 90 ∨ c = 90 ∧ a ≠ 90 ∧ b ≠ 90 :=
sorry

end NUMINAMATH_GPT_max_right_angles_in_triangle_l299_29961


namespace NUMINAMATH_GPT_train_passes_bridge_in_20_seconds_l299_29989

def train_length : ℕ := 360
def bridge_length : ℕ := 140
def train_speed_kmh : ℕ := 90

noncomputable def train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600
noncomputable def total_distance : ℕ := train_length + bridge_length
noncomputable def travel_time : ℝ := total_distance / train_speed_ms

theorem train_passes_bridge_in_20_seconds :
  travel_time = 20 := by
  sorry

end NUMINAMATH_GPT_train_passes_bridge_in_20_seconds_l299_29989


namespace NUMINAMATH_GPT_domain_of_f_l299_29973

noncomputable def f (x : ℝ) : ℝ := (4 * x - 2) / (Real.sqrt (x - 7))

theorem domain_of_f : {x : ℝ | ∃ y : ℝ, f y = f x } = {x : ℝ | x > 7} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l299_29973


namespace NUMINAMATH_GPT_find_a_l299_29988

noncomputable def f (a x : ℝ) : ℝ := 2^x / (2^x + a * x)

variables (a p q : ℝ)

theorem find_a
  (h1 : f a p = 6 / 5)
  (h2 : f a q = -1 / 5)
  (h3 : 2^(p + q) = 16 * p * q)
  (h4 : a > 0) :
  a = 4 :=
  sorry

end NUMINAMATH_GPT_find_a_l299_29988


namespace NUMINAMATH_GPT_a_square_plus_one_over_a_square_l299_29959

theorem a_square_plus_one_over_a_square (a : ℝ) (h : a + 1/a = 5) : a^2 + 1/a^2 = 23 :=
by 
  sorry

end NUMINAMATH_GPT_a_square_plus_one_over_a_square_l299_29959


namespace NUMINAMATH_GPT_length_PR_l299_29968

noncomputable def circle_radius : ℝ := 10
noncomputable def distance_PQ : ℝ := 12
noncomputable def midpoint_minor_arc_length_PR : ℝ :=
  let PS : ℝ := distance_PQ / 2
  let OS : ℝ := Real.sqrt (circle_radius^2 - PS^2)
  let RS : ℝ := circle_radius - OS
  Real.sqrt (PS^2 + RS^2)

theorem length_PR :
  midpoint_minor_arc_length_PR = 2 * Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_length_PR_l299_29968


namespace NUMINAMATH_GPT_part1_part2_l299_29990

-- Define the universal set R
def R := ℝ

-- Define set A
def A (x : ℝ) : Prop := x^2 - 3 * x - 4 ≤ 0

-- Define set B parameterized by a
def B (x a : ℝ) : Prop := (x - (a + 5)) / (x - a) > 0

-- Prove (1): A ∩ B when a = -2
theorem part1 : { x : ℝ | A x } ∩ { x : ℝ | B x (-2) } = { x : ℝ | 3 < x ∧ x ≤ 4 } :=
by
  sorry

-- Prove (2): The range of a such that A ⊆ B
theorem part2 : { a : ℝ | ∀ x, A x → B x a } = { a : ℝ | a < -6 ∨ a > 4 } :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l299_29990


namespace NUMINAMATH_GPT_flag_height_l299_29971

-- Definitions based on conditions
def flag_width : ℝ := 5
def paint_cost_per_quart : ℝ := 2
def sqft_per_quart : ℝ := 4
def total_spent : ℝ := 20

-- The theorem to prove the height h of the flag
theorem flag_height (h : ℝ) (paint_needed : ℝ -> ℝ) :
  paint_needed h = 4 := sorry

end NUMINAMATH_GPT_flag_height_l299_29971


namespace NUMINAMATH_GPT_amalie_coins_proof_l299_29950

def coins_proof : Prop :=
  ∃ (E A : ℕ),
    (E / A = 10 / 45) ∧
    (E + A = 440) ∧
    ((3 / 4) * A = 270) ∧
    (A - 270 = 90)

theorem amalie_coins_proof : coins_proof :=
  sorry

end NUMINAMATH_GPT_amalie_coins_proof_l299_29950


namespace NUMINAMATH_GPT_workouts_difference_l299_29939

theorem workouts_difference
  (workouts_monday : ℕ := 8)
  (workouts_tuesday : ℕ := 5)
  (workouts_wednesday : ℕ := 12)
  (workouts_thursday : ℕ := 17)
  (workouts_friday : ℕ := 10) :
  workouts_thursday - workouts_tuesday = 12 := 
by
  sorry

end NUMINAMATH_GPT_workouts_difference_l299_29939


namespace NUMINAMATH_GPT_mika_stickers_l299_29966

theorem mika_stickers 
    (initial_stickers : ℝ := 20.5)
    (bought_stickers : ℝ := 26.25)
    (birthday_stickers : ℝ := 19.75)
    (friend_stickers : ℝ := 7.5)
    (sister_stickers : ℝ := 6.3)
    (greeting_card_stickers : ℝ := 58.5)
    (yard_sale_stickers : ℝ := 3.2) :
    initial_stickers + bought_stickers + birthday_stickers + friend_stickers
    - sister_stickers - greeting_card_stickers - yard_sale_stickers = 6 := 
by
    sorry

end NUMINAMATH_GPT_mika_stickers_l299_29966


namespace NUMINAMATH_GPT_problem_statement_l299_29944

noncomputable def p (k : ℝ) (x : ℝ) : ℝ := k * x
noncomputable def q (x : ℝ) : ℝ := (x + 4) * (x - 1)

theorem problem_statement (k : ℝ) (h_p_linear : ∀ x, p k x = k * x) 
    (h_q_quadratic : ∀ x, q x = (x + 4) * (x - 1)) 
    (h_pass_origin : p k 0 / q 0 = 0)
    (h_pass_point : p k 2 / q 2 = -1) :
    p k 1 / q 1 = -3 / 5 :=
sorry

end NUMINAMATH_GPT_problem_statement_l299_29944


namespace NUMINAMATH_GPT_find_A_plus_B_l299_29946

def f (A B x : ℝ) : ℝ := A * x + B
def g (A B x : ℝ) : ℝ := B * x + A
def A_ne_B (A B : ℝ) : Prop := A ≠ B

theorem find_A_plus_B (A B x : ℝ) (h1 : A_ne_B A B)
  (h2 : (f A B (g A B x)) - (g A B (f A B x)) = 2 * (B - A)) : A + B = 3 :=
sorry

end NUMINAMATH_GPT_find_A_plus_B_l299_29946


namespace NUMINAMATH_GPT_price_decrease_necessary_l299_29930

noncomputable def final_price_decrease (P : ℝ) (x : ℝ) : Prop :=
  let increased_price := 1.2 * P
  let final_price := increased_price * (1 - x / 100)
  final_price = 0.88 * P

theorem price_decrease_necessary (x : ℝ) : 
  final_price_decrease 100 x -> x = 26.67 :=
by 
  intros h
  unfold final_price_decrease at h
  sorry

end NUMINAMATH_GPT_price_decrease_necessary_l299_29930


namespace NUMINAMATH_GPT_max_min_sum_l299_29926

noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log (x + 1) / Real.log 2

theorem max_min_sum : 
  (f 0 + f 1) = 4 := 
by
  sorry

end NUMINAMATH_GPT_max_min_sum_l299_29926


namespace NUMINAMATH_GPT_work_rate_b_l299_29954

theorem work_rate_b (A C B : ℝ) (hA : A = 1 / 8) (hC : C = 1 / 24) (h_combined : A + B + C = 1 / 4) : B = 1 / 12 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_work_rate_b_l299_29954


namespace NUMINAMATH_GPT_earned_points_l299_29911

def points_per_enemy := 3
def total_enemies := 6
def enemies_undefeated := 2
def enemies_defeated := total_enemies - enemies_undefeated

theorem earned_points : enemies_defeated * points_per_enemy = 12 :=
by sorry

end NUMINAMATH_GPT_earned_points_l299_29911


namespace NUMINAMATH_GPT_sin_2alpha_value_l299_29952

noncomputable def sin_2alpha_through_point (x y : ℝ) : ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let sin_alpha := y / r
  let cos_alpha := x / r
  2 * sin_alpha * cos_alpha

theorem sin_2alpha_value :
  sin_2alpha_through_point (-3) 4 = -24 / 25 :=
by
  sorry

end NUMINAMATH_GPT_sin_2alpha_value_l299_29952
