import Mathlib

namespace NUMINAMATH_GPT_max_value_of_f_l2185_218599

noncomputable def f (x : ℝ) : ℝ := |x| / (Real.sqrt (1 + x^2) * Real.sqrt (4 + x^2))

theorem max_value_of_f : ∃ M : ℝ, M = 1 / 3 ∧ ∀ x : ℝ, f x ≤ M :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l2185_218599


namespace NUMINAMATH_GPT_problem_statement_l2185_218524

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

end NUMINAMATH_GPT_problem_statement_l2185_218524


namespace NUMINAMATH_GPT_parallel_lines_a_eq_neg2_l2185_218538

theorem parallel_lines_a_eq_neg2 (a : ℝ) :
  (∀ x y : ℝ, (ax + y - 1 - a = 0) ↔ (x - (1/2) * y = 0)) → a = -2 :=
by sorry

end NUMINAMATH_GPT_parallel_lines_a_eq_neg2_l2185_218538


namespace NUMINAMATH_GPT_customers_left_l2185_218594

-- Given conditions:
def initial_customers : ℕ := 21
def remaining_customers : ℕ := 12

-- Prove that the number of customers who left is 9
theorem customers_left : initial_customers - remaining_customers = 9 := by
  sorry

end NUMINAMATH_GPT_customers_left_l2185_218594


namespace NUMINAMATH_GPT_find_n_from_binomial_terms_l2185_218535

theorem find_n_from_binomial_terms (x a : ℕ) (n : ℕ) 
  (h1 : n.choose 1 * x^(n-1) * a = 56) 
  (h2 : n.choose 2 * x^(n-2) * a^2 = 168) 
  (h3 : n.choose 3 * x^(n-3) * a^3 = 336) : 
  n = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_n_from_binomial_terms_l2185_218535


namespace NUMINAMATH_GPT_domain_all_real_numbers_l2185_218507

theorem domain_all_real_numbers (k : ℝ) :
  (∀ x : ℝ, -7 * x^2 - 4 * x + k ≠ 0) ↔ k < -4 / 7 := by
  sorry

end NUMINAMATH_GPT_domain_all_real_numbers_l2185_218507


namespace NUMINAMATH_GPT_evaluate_expression_l2185_218529

theorem evaluate_expression (x y z : ℤ) (hx : x = 5) (hy : y = x + 3) (hz : z = y - 11) 
  (h₁ : x + 2 ≠ 0) (h₂ : y - 3 ≠ 0) (h₃ : z + 7 ≠ 0) : 
  ((x + 3) / (x + 2)) * ((y - 2) / (y - 3)) * ((z + 9) / (z + 7)) = 72 / 35 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2185_218529


namespace NUMINAMATH_GPT_probability_A_not_losing_is_80_percent_l2185_218595

def probability_A_winning : ℝ := 0.30
def probability_draw : ℝ := 0.50
def probability_A_not_losing : ℝ := probability_A_winning + probability_draw

theorem probability_A_not_losing_is_80_percent : probability_A_not_losing = 0.80 :=
by 
  sorry

end NUMINAMATH_GPT_probability_A_not_losing_is_80_percent_l2185_218595


namespace NUMINAMATH_GPT_probability_no_obtuse_triangle_correct_l2185_218550

noncomputable def probability_no_obtuse_triangle (circle_points : List Point) (center : Point) : ℚ :=
  if circle_points.length = 4 then 3 / 64 else 0

theorem probability_no_obtuse_triangle_correct (circle_points : List Point) (center : Point) 
  (h : circle_points.length = 4) :
  probability_no_obtuse_triangle circle_points center = 3 / 64 :=
by
  sorry

end NUMINAMATH_GPT_probability_no_obtuse_triangle_correct_l2185_218550


namespace NUMINAMATH_GPT_stack_glasses_opacity_l2185_218566

-- Define the main problem's parameters and conditions
def num_glass_pieces : Nat := 5
def rotations := [0, 90, 180, 270] -- Possible rotations

-- Define the main theorem to state the problem in Lean
theorem stack_glasses_opacity :
  (∃ count : Nat, count = 7200 ∧
   -- There are 5 glass pieces
   ∀ (g : Fin num_glass_pieces), 
     -- Each piece is divided into 4 triangles
     ∀ (parts : Fin 4),
     -- There exists a unique painting configuration for each piece, can one prove it is exactly 7200 ways
     True
  ) :=
  sorry

end NUMINAMATH_GPT_stack_glasses_opacity_l2185_218566


namespace NUMINAMATH_GPT_intersection_points_on_hyperbola_l2185_218537

theorem intersection_points_on_hyperbola (p x y : ℝ) :
  (2*p*x - 3*y - 4*p = 0) ∧ (4*x - 3*p*y - 6 = 0) → 
  (∃ a b : ℝ, (x^2) / (a^2) - (y^2) / (b^2) = 1) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_intersection_points_on_hyperbola_l2185_218537


namespace NUMINAMATH_GPT_tan_alpha_plus_pi_over_4_l2185_218506

theorem tan_alpha_plus_pi_over_4 (x y : ℝ) (h1 : 3 * x + 4 * y = 0) : 
  Real.tan ((Real.arctan (- 3 / 4)) + π / 4) = 1 / 7 := 
by
  sorry

end NUMINAMATH_GPT_tan_alpha_plus_pi_over_4_l2185_218506


namespace NUMINAMATH_GPT_trig_expression_value_l2185_218525

theorem trig_expression_value :
  (3 / (Real.sin (140 * Real.pi / 180))^2 - 1 / (Real.cos (140 * Real.pi / 180))^2) * (1 / (2 * Real.sin (10 * Real.pi / 180))) = 16 := 
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_trig_expression_value_l2185_218525


namespace NUMINAMATH_GPT_find_minimum_x_and_values_l2185_218563

theorem find_minimum_x_and_values (x y z w : ℝ) (h1 : y = x - 2003)
  (h2 : z = 2 * y - 2003)
  (h3 : w = 3 * z - 2003)
  (h4 : 0 ≤ x)
  (h5 : 0 ≤ y)
  (h6 : 0 ≤ z)
  (h7 : 0 ≤ w) :
  x ≥ 10015 / 3 ∧ 
  (x = 10015 / 3 → y = 4006 / 3 ∧ z = 2003 / 3 ∧ w = 0) := by
  sorry

end NUMINAMATH_GPT_find_minimum_x_and_values_l2185_218563


namespace NUMINAMATH_GPT_angle_measure_l2185_218556

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end NUMINAMATH_GPT_angle_measure_l2185_218556


namespace NUMINAMATH_GPT_length_of_metallic_sheet_l2185_218565

variable (L : ℝ) (width side volume : ℝ)

theorem length_of_metallic_sheet (h1 : width = 36) (h2 : side = 8) (h3 : volume = 5120) :
  ((L - 2 * side) * (width - 2 * side) * side = volume) → L = 48 := 
by
  intros h_eq
  sorry

end NUMINAMATH_GPT_length_of_metallic_sheet_l2185_218565


namespace NUMINAMATH_GPT_value_of_y_l2185_218561

theorem value_of_y (y : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) :
  a = 10^3 → b = 10^4 → 
  a^y * 10^(3 * y) = (b^4) → 
  y = 8 / 3 :=
by 
  intro ha hb hc
  rw [ha, hb] at hc
  sorry

end NUMINAMATH_GPT_value_of_y_l2185_218561


namespace NUMINAMATH_GPT_combined_salaries_of_ABCD_l2185_218555

theorem combined_salaries_of_ABCD 
  (A B C D E : ℝ)
  (h1 : E = 9000)
  (h2 : (A + B + C + D + E) / 5 = 8600) :
  A + B + C + D = 34000 := 
sorry

end NUMINAMATH_GPT_combined_salaries_of_ABCD_l2185_218555


namespace NUMINAMATH_GPT_intervals_of_monotonicity_range_of_values_l2185_218544

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x - a * Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  -(1 + a) / x

noncomputable def h (a : ℝ) (x : ℝ) : ℝ :=
  f a x - g a x

theorem intervals_of_monotonicity (a : ℝ) (h_pos : 0 < a) :
  (∀ x > 0, x < 1 + a → h a x < h a (1 + a)) ∧
  (∀ x > 1 + a, h a x > h a (1 + a)) :=
sorry

theorem range_of_values (x0 : ℝ) (h_x0 : 1 ≤ x0 ∧ x0 ≤ Real.exp 1) (h_fx_gx : f a x0 < g a x0) :
  a > (Real.exp 1)^2 + 1 / (Real.exp 1 - 1) ∨ a < -2 :=
sorry

end NUMINAMATH_GPT_intervals_of_monotonicity_range_of_values_l2185_218544


namespace NUMINAMATH_GPT_max_value_of_m_l2185_218541

theorem max_value_of_m
  (a b : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : (2 / a) + (1 / b) = 1 / 4)
  (h4 : ∀ a b, 2 * a + b ≥ 9 * m) :
  m = 4 := 
sorry

end NUMINAMATH_GPT_max_value_of_m_l2185_218541


namespace NUMINAMATH_GPT_tan_positive_implies_sin_cos_positive_l2185_218578

variables {α : ℝ}

theorem tan_positive_implies_sin_cos_positive (h : Real.tan α > 0) : Real.sin α * Real.cos α > 0 :=
sorry

end NUMINAMATH_GPT_tan_positive_implies_sin_cos_positive_l2185_218578


namespace NUMINAMATH_GPT_find_x_l2185_218588

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b : ℝ × ℝ := (1, 5)
noncomputable def vector_c (x : ℝ) : ℝ × ℝ := (x, 1)

def collinear (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem find_x :
  ∃ x : ℝ, collinear (2 • vector_a - vector_b) (vector_c x) ∧ x = -1 := by
  sorry

end NUMINAMATH_GPT_find_x_l2185_218588


namespace NUMINAMATH_GPT_abs_diff_of_two_numbers_l2185_218592

variable {x y : ℝ}

theorem abs_diff_of_two_numbers (h1 : x + y = 40) (h2 : x * y = 396) : abs (x - y) = 4 := by
  sorry

end NUMINAMATH_GPT_abs_diff_of_two_numbers_l2185_218592


namespace NUMINAMATH_GPT_ratio_of_shaded_area_l2185_218564

theorem ratio_of_shaded_area 
  (AC : ℝ) (CB : ℝ) 
  (AB : ℝ := AC + CB) 
  (radius_AC : ℝ := AC / 2) 
  (radius_CB : ℝ := CB / 2)
  (radius_AB : ℝ := AB / 2) 
  (shaded_area : ℝ := (radius_AB ^ 2 * Real.pi / 2) - (radius_AC ^ 2 * Real.pi / 2) - (radius_CB ^ 2 * Real.pi / 2))
  (CD : ℝ := Real.sqrt (AC^2 - radius_CB^2))
  (circle_area : ℝ := CD^2 * Real.pi) :
  (shaded_area / circle_area = 21 / 187) := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_shaded_area_l2185_218564


namespace NUMINAMATH_GPT_big_dogs_count_l2185_218557

theorem big_dogs_count (B S : ℕ) (h_ratio : 3 * S = 17 * B) (h_total : B + S = 80) :
  B = 12 :=
by
  sorry

end NUMINAMATH_GPT_big_dogs_count_l2185_218557


namespace NUMINAMATH_GPT_combined_mean_is_254_over_15_l2185_218501

noncomputable def combined_mean_of_sets 
  (mean₁ : ℝ) (n₁ : ℕ) 
  (mean₂ : ℝ) (n₂ : ℕ) : ℝ :=
  (mean₁ * n₁ + mean₂ * n₂) / (n₁ + n₂)

theorem combined_mean_is_254_over_15 :
  combined_mean_of_sets 18 7 16 8 = (254 : ℝ) / 15 :=
by
  sorry

end NUMINAMATH_GPT_combined_mean_is_254_over_15_l2185_218501


namespace NUMINAMATH_GPT_moles_of_NH4Cl_combined_l2185_218528

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

end NUMINAMATH_GPT_moles_of_NH4Cl_combined_l2185_218528


namespace NUMINAMATH_GPT_divisibility_by_5_l2185_218574

theorem divisibility_by_5 (x y : ℤ) (h : 5 ∣ (x + 9 * y)) : 5 ∣ (8 * x + 7 * y) :=
sorry

end NUMINAMATH_GPT_divisibility_by_5_l2185_218574


namespace NUMINAMATH_GPT_class_a_winning_probability_best_of_three_l2185_218596

theorem class_a_winning_probability_best_of_three :
  let p := (3 : ℚ) / 5
  let win_first_two := p * p
  let win_first_and_third := p * ((1 - p) * p)
  let win_last_two := (1 - p) * (p * p)
  p * p + p * ((1 - p) * p) + (1 - p) * (p * p) = 81 / 125 :=
by
  sorry

end NUMINAMATH_GPT_class_a_winning_probability_best_of_three_l2185_218596


namespace NUMINAMATH_GPT_move_line_down_eq_l2185_218562

theorem move_line_down_eq (x y : ℝ) : (y = 2 * x) → (y - 3 = 2 * x - 3) :=
by
  sorry

end NUMINAMATH_GPT_move_line_down_eq_l2185_218562


namespace NUMINAMATH_GPT_scallops_cost_calculation_l2185_218571

def scallops_per_pound : ℕ := 8
def cost_per_pound : ℝ := 24.00
def scallops_per_person : ℕ := 2
def number_of_people : ℕ := 8

def total_cost : ℝ := 
  let total_scallops := number_of_people * scallops_per_person
  let total_pounds := total_scallops / scallops_per_pound
  total_pounds * cost_per_pound

theorem scallops_cost_calculation :
  total_cost = 48.00 :=
by sorry

end NUMINAMATH_GPT_scallops_cost_calculation_l2185_218571


namespace NUMINAMATH_GPT_non_integer_sum_exists_l2185_218584

theorem non_integer_sum_exists (k l : ℕ) (hk : 0 < k) (hl : 0 < l) :
  ∃ M : ℕ, ∀ n : ℕ, n > M → ¬ ∃ t : ℤ, (k + 1/2)^n + (l + 1/2)^n = t := 
sorry

end NUMINAMATH_GPT_non_integer_sum_exists_l2185_218584


namespace NUMINAMATH_GPT_problem_statement_l2185_218526

def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem problem_statement : avg3 (avg3 (-1) 2 3) (avg2 2 3) 1 = 29 / 18 := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l2185_218526


namespace NUMINAMATH_GPT_christineTravelDistance_l2185_218527

-- Definition of Christine's speed and time
def christineSpeed : ℝ := 20
def christineTime : ℝ := 4

-- Theorem to prove the distance Christine traveled
theorem christineTravelDistance : christineSpeed * christineTime = 80 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_christineTravelDistance_l2185_218527


namespace NUMINAMATH_GPT_solve_for_x_y_l2185_218553

theorem solve_for_x_y (x y : ℝ) (h1 : x^2 + x * y + y = 14) (h2 : y^2 + x * y + x = 28) : 
  x + y = -7 ∨ x + y = 6 :=
by 
  -- We'll write sorry here to indicate the proof is to be completed
  sorry

end NUMINAMATH_GPT_solve_for_x_y_l2185_218553


namespace NUMINAMATH_GPT_inequality_2_pow_n_plus_2_gt_n_squared_l2185_218589

theorem inequality_2_pow_n_plus_2_gt_n_squared (n : ℕ) (hn : n > 0) : 2^n + 2 > n^2 := sorry

end NUMINAMATH_GPT_inequality_2_pow_n_plus_2_gt_n_squared_l2185_218589


namespace NUMINAMATH_GPT_faucet_open_duration_l2185_218515

-- Initial definitions based on conditions in the problem
def init_water : ℕ := 120
def flow_rate : ℕ := 4
def rem_water : ℕ := 20

-- The equivalent Lean 4 statement to prove
theorem faucet_open_duration (t : ℕ) (H1: init_water - rem_water = flow_rate * t) : t = 25 :=
sorry

end NUMINAMATH_GPT_faucet_open_duration_l2185_218515


namespace NUMINAMATH_GPT_pf1_pf2_range_l2185_218533

noncomputable def ellipse_point (x y : ℝ) : Prop :=
  x ^ 2 / 4 + y ^ 2 = 1

noncomputable def dot_product (x y : ℝ) : ℝ :=
  (x ^ 2 + y ^ 2 - 3)

theorem pf1_pf2_range (x y : ℝ) (h : ellipse_point x y) :
  -2 ≤ dot_product x y ∧ dot_product x y ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_pf1_pf2_range_l2185_218533


namespace NUMINAMATH_GPT_coefficient_x3_expansion_l2185_218530

/--
Prove that the coefficient of \(x^{3}\) in the expansion of \(( \frac{x}{\sqrt{y}} - \frac{y}{\sqrt{x}})^{6}\) is \(15\).
-/
theorem coefficient_x3_expansion (x y : ℝ) : 
  (∃ c : ℝ, c = 15 ∧ (x / y.sqrt - y / x.sqrt) ^ 6 = c * x ^ 3) :=
sorry

end NUMINAMATH_GPT_coefficient_x3_expansion_l2185_218530


namespace NUMINAMATH_GPT_g_of_neg2_l2185_218504

def g (x : ℚ) : ℚ := (2 * x - 3) / (4 * x + 5)

theorem g_of_neg2 : g (-2) = 7 / 3 := by
  sorry

end NUMINAMATH_GPT_g_of_neg2_l2185_218504


namespace NUMINAMATH_GPT_train_ride_length_l2185_218587

theorem train_ride_length :
  let reading_time := 2
  let eating_time := 1
  let watching_time := 3
  let napping_time := 3
  reading_time + eating_time + watching_time + napping_time = 9 := 
by
  sorry

end NUMINAMATH_GPT_train_ride_length_l2185_218587


namespace NUMINAMATH_GPT_find_z_l2185_218560

open Complex

theorem find_z (z : ℂ) (h : (1 + 2 * z) / (1 - z) = Complex.I) : 
  z = -1 / 5 + 3 / 5 * Complex.I := 
sorry

end NUMINAMATH_GPT_find_z_l2185_218560


namespace NUMINAMATH_GPT_line_point_coordinates_l2185_218539

theorem line_point_coordinates (t : ℝ) (x y z : ℝ) : 
  (x, y, z) = (5, 0, 3) + t • (0, 3, 0) →
  t = 1/2 →
  (x, y, z) = (5, 3/2, 3) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_line_point_coordinates_l2185_218539


namespace NUMINAMATH_GPT_sector_area_is_8pi_over_3_l2185_218576

noncomputable def sector_area {r θ1 θ2 : ℝ} 
  (hθ1 : θ1 = π / 3)
  (hθ2 : θ2 = 2 * π / 3)
  (hr : r = 4) : ℝ := 
    1 / 2 * (θ2 - θ1) * r ^ 2

theorem sector_area_is_8pi_over_3 (θ1 θ2 : ℝ) 
  (hθ1 : θ1 = π / 3)
  (hθ2 : θ2 = 2 * π / 3)
  (r : ℝ) (hr : r = 4) : 
  sector_area hθ1 hθ2 hr = 8 * π / 3 :=
by
  sorry

end NUMINAMATH_GPT_sector_area_is_8pi_over_3_l2185_218576


namespace NUMINAMATH_GPT_mark_team_free_throws_l2185_218582

theorem mark_team_free_throws (F : ℕ) : 
  let mark_2_pointers := 25
  let mark_3_pointers := 8
  let opp_2_pointers := 2 * mark_2_pointers
  let opp_3_pointers := 1 / 2 * mark_3_pointers
  let total_points := 201
  2 * mark_2_pointers + 3 * mark_3_pointers + F + 2 * mark_2_pointers + 3 / 2 * mark_3_pointers + F / 2 = total_points →
  F = 10 := by
  sorry

end NUMINAMATH_GPT_mark_team_free_throws_l2185_218582


namespace NUMINAMATH_GPT_cos_pi_minus_half_alpha_l2185_218552

-- Conditions given in the problem
variable (α : ℝ)
variable (hα1 : 0 < α ∧ α < π / 2)
variable (hα2 : Real.sin α = 3 / 5)

-- The proof problem statement
theorem cos_pi_minus_half_alpha (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.sin α = 3 / 5) : 
  Real.cos (π - α / 2) = -3 * Real.sqrt 10 / 10 := 
sorry

end NUMINAMATH_GPT_cos_pi_minus_half_alpha_l2185_218552


namespace NUMINAMATH_GPT_six_digit_quotient_l2185_218517

def six_digit_number (A B : ℕ) : ℕ := 100000 * A + 97860 + B

def divisible_by_99 (n : ℕ) : Prop := n % 99 = 0

theorem six_digit_quotient (A B : ℕ) (hA : A = 5) (hB : B = 1)
  (h9786B : divisible_by_99 (six_digit_number A B)) : 
  six_digit_number A B / 99 = 6039 := by
  sorry

end NUMINAMATH_GPT_six_digit_quotient_l2185_218517


namespace NUMINAMATH_GPT_value_of_a_l2185_218516

theorem value_of_a (a b : ℤ) (h : (∀ x, x^2 - x - 1 = 0 → a * x^17 + b * x^16 + 1 = 0)) : a = 987 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_a_l2185_218516


namespace NUMINAMATH_GPT_kaleb_earnings_and_boxes_l2185_218520

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

end NUMINAMATH_GPT_kaleb_earnings_and_boxes_l2185_218520


namespace NUMINAMATH_GPT_problem_statement_l2185_218598

noncomputable def f (a b c x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2

theorem problem_statement (a b c : ℝ) (h : f a b c (-5) = 3) : f a b c 5 + f a b c (-5) = 4 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2185_218598


namespace NUMINAMATH_GPT_volume_of_sphere_from_cube_surface_area_l2185_218581

theorem volume_of_sphere_from_cube_surface_area (S : ℝ) (h : S = 24) : 
  ∃ V : ℝ, V = 4 * Real.sqrt 3 * Real.pi := 
sorry

end NUMINAMATH_GPT_volume_of_sphere_from_cube_surface_area_l2185_218581


namespace NUMINAMATH_GPT_percent_nonunion_women_l2185_218545

variable (E : ℝ) -- Total number of employees

-- Definitions derived from the problem conditions
def menPercent : ℝ := 0.46
def unionPercent : ℝ := 0.60
def nonUnionPercent : ℝ := 1 - unionPercent
def nonUnionWomenPercent : ℝ := 0.90

theorem percent_nonunion_women :
  nonUnionWomenPercent = 0.90 :=
by
  sorry

end NUMINAMATH_GPT_percent_nonunion_women_l2185_218545


namespace NUMINAMATH_GPT_sum_of_ages_l2185_218572

-- Problem statement:
-- Given: The product of their ages is 144.
-- Prove: The sum of their ages is 16.
theorem sum_of_ages (k t : ℕ) (htwins : t > k) (hprod : 2 * t * k = 144) : 2 * t + k = 16 := 
sorry

end NUMINAMATH_GPT_sum_of_ages_l2185_218572


namespace NUMINAMATH_GPT_ninth_term_arithmetic_sequence_l2185_218510

theorem ninth_term_arithmetic_sequence 
  (a1 a17 d a9 : ℚ) 
  (h1 : a1 = 2 / 3) 
  (h17 : a17 = 3 / 2) 
  (h_formula : a17 = a1 + 16 * d) 
  (h9_formula : a9 = a1 + 8 * d) :
  a9 = 13 / 12 := by
  sorry

end NUMINAMATH_GPT_ninth_term_arithmetic_sequence_l2185_218510


namespace NUMINAMATH_GPT_greatest_four_digit_p_l2185_218554

-- Define conditions
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def reverse_digits (n : ℕ) : ℕ := 
  let d1 := n / 1000 % 10
  let d2 := n / 100 % 10
  let d3 := n / 10 % 10
  let d4 := n % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1
def is_divisible_by (a b : ℕ) : Prop := b ∣ a

-- Proof problem
theorem greatest_four_digit_p (p : ℕ) (q : ℕ) 
    (hp1 : is_four_digit p)
    (hp2 : q = reverse_digits p)
    (hp3 : is_four_digit q)
    (hp4 : is_divisible_by p 63)
    (hp5 : is_divisible_by q 63)
    (hp6 : is_divisible_by p 19) :
  p = 5985 :=
sorry

end NUMINAMATH_GPT_greatest_four_digit_p_l2185_218554


namespace NUMINAMATH_GPT_maximum_students_per_dentist_l2185_218503

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

end NUMINAMATH_GPT_maximum_students_per_dentist_l2185_218503


namespace NUMINAMATH_GPT_match_scheduling_ways_l2185_218540

def different_ways_to_schedule_match (num_players : Nat) (num_rounds : Nat) : Nat :=
  (num_rounds.factorial * num_rounds.factorial)

theorem match_scheduling_ways : different_ways_to_schedule_match 4 4 = 576 :=
by
  sorry

end NUMINAMATH_GPT_match_scheduling_ways_l2185_218540


namespace NUMINAMATH_GPT_least_positive_divisible_by_smallest_primes_l2185_218521

def smallest_primes := [2, 3, 5, 7, 11]

noncomputable def product_of_smallest_primes :=
  List.foldl (· * ·) 1 smallest_primes

theorem least_positive_divisible_by_smallest_primes :
  product_of_smallest_primes = 2310 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_divisible_by_smallest_primes_l2185_218521


namespace NUMINAMATH_GPT_part_I_solution_set_part_II_min_value_l2185_218575

-- Define the function f
def f (x a : ℝ) := 2*|x + 1| - |x - a|

-- Part I: Prove the solution set of f(x) ≥ 0 when a = 2
theorem part_I_solution_set (x : ℝ) :
  f x 2 ≥ 0 ↔ x ≤ -4 ∨ x ≥ 0 :=
sorry

-- Define the function g
def g (x a : ℝ) := f x a + 3*|x - a|

-- Part II: Prove the minimum value of m + n given t = 4 when a = 1
theorem part_II_min_value (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x, g x 1 ≥ 4) → (2/m + 1/(2*n) = 4) → m + n = 9/8 :=
sorry

end NUMINAMATH_GPT_part_I_solution_set_part_II_min_value_l2185_218575


namespace NUMINAMATH_GPT_change_is_five_l2185_218518

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

end NUMINAMATH_GPT_change_is_five_l2185_218518


namespace NUMINAMATH_GPT_max_area_square_l2185_218568

theorem max_area_square (P : ℝ) : 
  ∀ x y : ℝ, 2 * x + 2 * y = P → (x * y ≤ (P / 4) ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_max_area_square_l2185_218568


namespace NUMINAMATH_GPT_circle_passes_first_and_second_quadrants_l2185_218547

theorem circle_passes_first_and_second_quadrants :
  ∀ (x y : ℝ), (x - 1)^2 + (y - 3)^2 = 4 → ((x ≥ 0 ∧ y ≥ 0) ∨ (x ≤ 0 ∧ y ≥ 0)) :=
by
  sorry

end NUMINAMATH_GPT_circle_passes_first_and_second_quadrants_l2185_218547


namespace NUMINAMATH_GPT_sandy_siding_cost_l2185_218508

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

end NUMINAMATH_GPT_sandy_siding_cost_l2185_218508


namespace NUMINAMATH_GPT_max_four_digit_prime_product_l2185_218559

theorem max_four_digit_prime_product :
  ∃ (x y : ℕ) (n : ℕ), x < 5 ∧ y < 5 ∧ x ≠ y ∧ Prime x ∧ Prime y ∧ Prime (10 * x + y) ∧ n = x * y * (10 * x + y) ∧ n = 138 :=
by
  sorry

end NUMINAMATH_GPT_max_four_digit_prime_product_l2185_218559


namespace NUMINAMATH_GPT_fraction_identity_l2185_218513

theorem fraction_identity (x y z : ℤ) (h : x / 2 = y / 3 ∧ y / 3 = z / 4) :
  (x + y) / (3 * y - 2 * z) = 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_identity_l2185_218513


namespace NUMINAMATH_GPT_sqrt_expression_simplification_l2185_218590

theorem sqrt_expression_simplification :
  (Real.sqrt 72 / Real.sqrt 3 - Real.sqrt (1 / 2) * Real.sqrt 12 - |2 - Real.sqrt 6|) = 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_simplification_l2185_218590


namespace NUMINAMATH_GPT_line_intersects_curve_l2185_218514

theorem line_intersects_curve (k : ℝ) :
  (∃ x y : ℝ, y + k * x + 2 = 0 ∧ x^2 + y^2 = 2 * x) ↔ k ≤ -3/4 := by
  sorry

end NUMINAMATH_GPT_line_intersects_curve_l2185_218514


namespace NUMINAMATH_GPT_room_width_l2185_218548

theorem room_width (length : ℕ) (total_cost : ℕ) (cost_per_sqm : ℕ) : ℚ :=
  let area := total_cost / cost_per_sqm
  let width := area / length
  width

example : room_width 9 38475 900 = 4.75 := by
  sorry

end NUMINAMATH_GPT_room_width_l2185_218548


namespace NUMINAMATH_GPT_odd_periodic_function_l2185_218542

noncomputable def f : ℤ → ℤ := sorry

theorem odd_periodic_function (f_odd : ∀ x : ℤ, f (-x) = -f x)
  (period_f_3x1 : ∀ x : ℤ, f (3 * x + 1) = f (3 * (x + 3) + 1))
  (f_one : f 1 = -1) : f 2006 = 1 :=
sorry

end NUMINAMATH_GPT_odd_periodic_function_l2185_218542


namespace NUMINAMATH_GPT_problem_correct_calculation_l2185_218531

theorem problem_correct_calculation (a b : ℕ) : 
  (4 * a - 2 * a ≠ 2) ∧ 
  (a^8 / a^4 ≠ a^2) ∧ 
  (a^2 * a^3 = a^5) ∧ 
  ((b^2)^3 ≠ b^5) :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_correct_calculation_l2185_218531


namespace NUMINAMATH_GPT_inequality_proof_l2185_218551

theorem inequality_proof (a b : ℝ) (h₀ : b > a) (h₁ : ab > 0) : 
  (1 / a > 1 / b) ∧ (a + b < 2 * b) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2185_218551


namespace NUMINAMATH_GPT_derivative_of_function_l2185_218580

theorem derivative_of_function
  (y : ℝ → ℝ)
  (h : ∀ x, y x = (1/2) * (Real.exp x + Real.exp (-x))) :
  ∀ x, deriv y x = (1/2) * (Real.exp x - Real.exp (-x)) :=
by
  sorry

end NUMINAMATH_GPT_derivative_of_function_l2185_218580


namespace NUMINAMATH_GPT_shift_left_by_pi_over_six_l2185_218573

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) * Real.sin x + Real.cos x
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin x

theorem shift_left_by_pi_over_six : f = λ x => g (x + Real.pi / 6) := by
  sorry

end NUMINAMATH_GPT_shift_left_by_pi_over_six_l2185_218573


namespace NUMINAMATH_GPT_work_together_days_l2185_218585

theorem work_together_days (A_rate B_rate x total_work B_days_worked : ℚ)
  (hA : A_rate = 1/4)
  (hB : B_rate = 1/8)
  (hCombined : (A_rate + B_rate) * x + B_rate * B_days_worked = total_work)
  (hTotalWork : total_work = 1)
  (hBDays : B_days_worked = 2) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_work_together_days_l2185_218585


namespace NUMINAMATH_GPT_study_time_difference_l2185_218523

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

end NUMINAMATH_GPT_study_time_difference_l2185_218523


namespace NUMINAMATH_GPT_children_count_l2185_218577

theorem children_count 
  (A B C : Finset ℕ)
  (hA : A.card = 7)
  (hB : B.card = 6)
  (hC : C.card = 5)
  (hA_inter_B : (A ∩ B).card = 4)
  (hA_inter_C : (A ∩ C).card = 3)
  (hB_inter_C : (B ∩ C).card = 2)
  (hA_inter_B_inter_C : (A ∩ B ∩ C).card = 1) :
  (A ∪ B ∪ C).card = 10 := 
by
  sorry

end NUMINAMATH_GPT_children_count_l2185_218577


namespace NUMINAMATH_GPT_not_every_tv_owner_has_pass_l2185_218512

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

end NUMINAMATH_GPT_not_every_tv_owner_has_pass_l2185_218512


namespace NUMINAMATH_GPT_integer_a_can_be_written_in_form_l2185_218583

theorem integer_a_can_be_written_in_form 
  (a x y : ℤ) 
  (h : 3 * a = x^2 + 2 * y^2) : 
  ∃ u v : ℤ, a = u^2 + 2 * v^2 :=
sorry

end NUMINAMATH_GPT_integer_a_can_be_written_in_form_l2185_218583


namespace NUMINAMATH_GPT_friend_c_spent_26_l2185_218597

theorem friend_c_spent_26 :
  let you_spent := 12
  let friend_a_spent := you_spent + 4
  let friend_b_spent := friend_a_spent - 3
  let friend_c_spent := friend_b_spent * 2
  friend_c_spent = 26 :=
by
  sorry

end NUMINAMATH_GPT_friend_c_spent_26_l2185_218597


namespace NUMINAMATH_GPT_series_result_l2185_218502

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (k + 1) / 3^(k + 1)

theorem series_result : series_sum = 3 / 2 :=
sorry

end NUMINAMATH_GPT_series_result_l2185_218502


namespace NUMINAMATH_GPT_range_of_2a_plus_3b_l2185_218579

theorem range_of_2a_plus_3b (a b : ℝ)
  (h1 : -1 ≤ a + b ∧ a + b ≤ 1)
  (h2 : -1 ≤ a - b ∧ a - b ≤ 1) :
  -3 ≤ 2 * a + 3 * b ∧ 2 * a + 3 * b ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_2a_plus_3b_l2185_218579


namespace NUMINAMATH_GPT_remaining_volume_of_cube_l2185_218511

theorem remaining_volume_of_cube (s : ℝ) (r : ℝ) (h : ℝ) (π : ℝ) 
    (cube_volume : s = 5) 
    (cylinder_radius : r = 1.5) 
    (cylinder_height : h = 5) :
    s^3 - π * r^2 * h = 125 - 11.25 * π := by
  sorry

end NUMINAMATH_GPT_remaining_volume_of_cube_l2185_218511


namespace NUMINAMATH_GPT_sam_mary_total_balloons_l2185_218522

def Sam_initial_balloons : ℝ := 6.0
def Sam_gives : ℝ := 5.0
def Sam_remaining_balloons : ℝ := Sam_initial_balloons - Sam_gives

def Mary_balloons : ℝ := 7.0

def total_balloons : ℝ := Sam_remaining_balloons + Mary_balloons

theorem sam_mary_total_balloons : total_balloons = 8.0 :=
by
  sorry

end NUMINAMATH_GPT_sam_mary_total_balloons_l2185_218522


namespace NUMINAMATH_GPT_find_f5_l2185_218519

noncomputable def f : ℝ → ℝ := sorry

axiom additivity : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f4_value : f 4 = 5

theorem find_f5 : f 5 = 25 / 4 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_f5_l2185_218519


namespace NUMINAMATH_GPT_alice_savings_third_month_l2185_218591

theorem alice_savings_third_month :
  ∀ (saved_first : ℕ) (increase_per_month : ℕ),
  saved_first = 10 →
  increase_per_month = 30 →
  let saved_second := saved_first + increase_per_month
  let saved_third := saved_second + increase_per_month
  saved_third = 70 :=
by intros saved_first increase_per_month h1 h2;
   let saved_second := saved_first + increase_per_month;
   let saved_third := saved_second + increase_per_month;
   sorry

end NUMINAMATH_GPT_alice_savings_third_month_l2185_218591


namespace NUMINAMATH_GPT_cube_root_59319_cube_root_103823_l2185_218536

theorem cube_root_59319 : ∃ x : ℕ, x ^ 3 = 59319 ∧ x = 39 :=
by
  -- Sorry used to skip the proof, which is not required 
  sorry

theorem cube_root_103823 : ∃ x : ℕ, x ^ 3 = 103823 ∧ x = 47 :=
by
  -- Sorry used to skip the proof, which is not required 
  sorry

end NUMINAMATH_GPT_cube_root_59319_cube_root_103823_l2185_218536


namespace NUMINAMATH_GPT_vampires_after_two_nights_l2185_218570

def initial_population : ℕ := 300
def initial_vampires : ℕ := 3
def conversion_rate : ℕ := 7

theorem vampires_after_two_nights :
  let first_night := initial_vampires * conversion_rate
  let total_first_night := initial_vampires + first_night
  let second_night := total_first_night * conversion_rate
  let total_second_night := total_first_night + second_night
  total_second_night = 192 :=
by
  let first_night := initial_vampires * conversion_rate
  let total_first_night := initial_vampires + first_night
  let second_night := total_first_night * conversion_rate
  let total_second_night := total_first_night + second_night
  have h1 : first_night = 21 := rfl
  have h2 : total_first_night = 24 := rfl
  have h3 : second_night = 168 := rfl
  have h4 : total_second_night = 192 := rfl
  exact rfl

end NUMINAMATH_GPT_vampires_after_two_nights_l2185_218570


namespace NUMINAMATH_GPT_pages_in_book_l2185_218558

theorem pages_in_book
  (x : ℝ)
  (h1 : x - (x / 6 + 10) = (5 * x) / 6 - 10)
  (h2 : (5 * x) / 6 - 10 - ((1 / 5) * ((5 * x) / 6 - 10) + 20) = (2 * x) / 3 - 28)
  (h3 : (2 * x) / 3 - 28 - ((1 / 4) * ((2 * x) / 3 - 28) + 25) = x / 2 - 46)
  (h4 : x / 2 - 46 = 72) :
  x = 236 := 
sorry

end NUMINAMATH_GPT_pages_in_book_l2185_218558


namespace NUMINAMATH_GPT_Darcy_remaining_clothes_l2185_218500

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

end NUMINAMATH_GPT_Darcy_remaining_clothes_l2185_218500


namespace NUMINAMATH_GPT_suff_and_not_necessary_l2185_218549

theorem suff_and_not_necessary (a b : ℝ) (h : a > b ∧ b > 0) :
  (|a| > |b|) ∧ (¬(∀ x y : ℝ, (|x| > |y|) → (x > y ∧ y > 0))) :=
by
  sorry

end NUMINAMATH_GPT_suff_and_not_necessary_l2185_218549


namespace NUMINAMATH_GPT_forty_ab_l2185_218532

theorem forty_ab (a b : ℝ) (h₁ : 4 * a = 30) (h₂ : 5 * b = 30) : 40 * a * b = 1800 :=
by
  sorry

end NUMINAMATH_GPT_forty_ab_l2185_218532


namespace NUMINAMATH_GPT_students_taking_both_courses_l2185_218593

theorem students_taking_both_courses (n_total n_F n_G n_neither number_both : ℕ)
  (h_total : n_total = 79)
  (h_F : n_F = 41)
  (h_G : n_G = 22)
  (h_neither : n_neither = 25)
  (h_any_language : n_total - n_neither = 54)
  (h_sum_languages : n_F + n_G = 63)
  (h_both : n_F + n_G - (n_total - n_neither) = number_both) :
  number_both = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_students_taking_both_courses_l2185_218593


namespace NUMINAMATH_GPT_cookies_per_student_l2185_218546

theorem cookies_per_student (students : ℕ) (percent : ℝ) (oatmeal_cookies : ℕ) 
                            (h_students : students = 40)
                            (h_percent : percent = 10 / 100)
                            (h_oatmeal : oatmeal_cookies = 8) :
                            (oatmeal_cookies / percent / students) = 2 := by
  sorry

end NUMINAMATH_GPT_cookies_per_student_l2185_218546


namespace NUMINAMATH_GPT_solve_equation_l2185_218509

theorem solve_equation : 
  ∀ x : ℝ, (x^2 + 2*x + 3)/(x + 2) = x + 4 → x = -(5/4) := by
  sorry

end NUMINAMATH_GPT_solve_equation_l2185_218509


namespace NUMINAMATH_GPT_value_of_abs_h_l2185_218569

theorem value_of_abs_h (h : ℝ) : 
  (∃ r s : ℝ, (r + s = -4 * h) ∧ (r * s = -5) ∧ (r^2 + s^2 = 13)) → 
  |h| = (Real.sqrt 3) / 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_abs_h_l2185_218569


namespace NUMINAMATH_GPT_find_z_l2185_218534

-- Condition: there exists a constant k such that z = k * w
def direct_variation (z w : ℝ): Prop := ∃ k, z = k * w

-- We set up the conditions given in the problem.
theorem find_z (k : ℝ) (hw1 : 10 = k * 5) (hw2 : w = -15) : direct_variation z w → z = -30 :=
by
  sorry

end NUMINAMATH_GPT_find_z_l2185_218534


namespace NUMINAMATH_GPT_total_books_per_year_l2185_218586

variable (c s : ℕ)

theorem total_books_per_year (hc : 0 < c) (hs : 0 < s) :
  6 * 12 * (c * s) = 72 * c * s := by
  sorry

end NUMINAMATH_GPT_total_books_per_year_l2185_218586


namespace NUMINAMATH_GPT_poly_divisibility_implies_C_D_l2185_218505

noncomputable def poly_condition : Prop :=
  ∃ (C D : ℤ), ∀ (α : ℂ), α^2 - α + 1 = 0 → α^103 + C * α^2 + D * α + 1 = 0

/- The translated proof problem -/
theorem poly_divisibility_implies_C_D (C D : ℤ) :
  (poly_condition) → (C = -1 ∧ D = 0) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_poly_divisibility_implies_C_D_l2185_218505


namespace NUMINAMATH_GPT_tank_capacity_l2185_218543

theorem tank_capacity (C : ℕ) (h₁ : C = 785) :
  360 - C / 4 - C / 8 = C / 12 :=
by 
  -- Assuming h₁: C = 785
  have h₁: C = 785 := by exact h₁
  -- Provide proof steps here (not required for the task)
  sorry

end NUMINAMATH_GPT_tank_capacity_l2185_218543


namespace NUMINAMATH_GPT_max_value_neg_expr_l2185_218567

theorem max_value_neg_expr (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 1) :
  - (1 / (2 * a)) - (2 / b) ≤ - (9 / 2) :=
by 
  sorry

end NUMINAMATH_GPT_max_value_neg_expr_l2185_218567
