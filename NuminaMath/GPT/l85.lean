import Mathlib

namespace NUMINAMATH_GPT_f_g_of_3_l85_8508

def g (x : ℕ) : ℕ := x^3
def f (x : ℕ) : ℕ := 3 * x - 2

theorem f_g_of_3 : f (g 3) = 79 := by
  sorry

end NUMINAMATH_GPT_f_g_of_3_l85_8508


namespace NUMINAMATH_GPT_ellipse_x_intercept_l85_8513

theorem ellipse_x_intercept :
  let F_1 := (0,3)
  let F_2 := (4,0)
  let ellipse := { P : ℝ × ℝ | (dist P F_1) + (dist P F_2) = 7 }
  ∃ x : ℝ, x ≠ 0 ∧ (x, 0) ∈ ellipse ∧ x = 56 / 11 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_x_intercept_l85_8513


namespace NUMINAMATH_GPT_least_number_to_subtract_l85_8541

theorem least_number_to_subtract (n d : ℕ) (n_val : n = 13602) (d_val : d = 87) : 
  ∃ r, (n - r) % d = 0 ∧ r = 30 := by
  sorry

end NUMINAMATH_GPT_least_number_to_subtract_l85_8541


namespace NUMINAMATH_GPT_average_of_N_l85_8528

theorem average_of_N (N : ℤ) (h1 : (1:ℚ)/3 < N/90) (h2 : N/90 < (2:ℚ)/5) : 31 ≤ N ∧ N ≤ 35 → (N = 31 ∨ N = 32 ∨ N = 33 ∨ N = 34 ∨ N = 35) → (31 + 32 + 33 + 34 + 35) / 5 = 33 := by
  sorry

end NUMINAMATH_GPT_average_of_N_l85_8528


namespace NUMINAMATH_GPT_leopards_arrangement_l85_8550

theorem leopards_arrangement :
  let leopards := 10
  let shortest := 3
  let remaining := leopards - shortest
  (shortest! * remaining! = 30240) :=
by
  let leopards := 10
  let shortest := 3
  let remaining := leopards - shortest
  have factorials_eq: shortest! * remaining! = 30240 := sorry
  exact factorials_eq

end NUMINAMATH_GPT_leopards_arrangement_l85_8550


namespace NUMINAMATH_GPT_ratio_upstream_downstream_l85_8520

noncomputable def ratio_time_upstream_to_downstream
  (V_b V_s : ℕ) (T_u T_d : ℕ) : ℕ :=
(V_b + V_s) / (V_b - V_s)

theorem ratio_upstream_downstream
  (V_b V_s : ℕ) (hVb : V_b = 48) (hVs : V_s = 16) (T_u T_d : ℕ)
  (hT : ratio_time_upstream_to_downstream V_b V_s T_u T_d = 2) :
  T_u / T_d = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_upstream_downstream_l85_8520


namespace NUMINAMATH_GPT_smallest_rel_prime_to_180_l85_8597

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ x ≤ 7 ∧ (∀ y : ℕ, y > 1 ∧ y < x → y.gcd 180 ≠ 1) ∧ x.gcd 180 = 1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_rel_prime_to_180_l85_8597


namespace NUMINAMATH_GPT_dihedral_angle_is_60_degrees_l85_8540

def point (x y z : ℝ) := (x, y, z)

noncomputable def dihedral_angle (P Q R S T : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem dihedral_angle_is_60_degrees :
  dihedral_angle 
    (point 1 0 0)  -- A
    (point 1 1 0)  -- B
    (point 0 0 0)  -- D
    (point 1 0 1)  -- A₁
    (point 0 0 1)  -- D₁
 = 60 :=
sorry

end NUMINAMATH_GPT_dihedral_angle_is_60_degrees_l85_8540


namespace NUMINAMATH_GPT_find_quadratic_function_l85_8517

open Function

-- Define the quadratic function g(x) with parameters c and d
def g (c d : ℝ) (x : ℝ) : ℝ := x^2 + c * x + d

-- State the main theorem
theorem find_quadratic_function :
  ∃ (c d : ℝ), (∀ x : ℝ, (g c d (g c d x + x)) / (g c d x) = x^2 + 120 * x + 360) ∧ c = 119 ∧ d = 240 :=
by
  sorry

end NUMINAMATH_GPT_find_quadratic_function_l85_8517


namespace NUMINAMATH_GPT_Isaabel_math_pages_l85_8560

theorem Isaabel_math_pages (x : ℕ) (total_problems : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) :
  (reading_pages * problems_per_page = 20) ∧ (total_problems = 30) →
  x * problems_per_page + 20 = total_problems →
  x = 2 := by
  sorry

end NUMINAMATH_GPT_Isaabel_math_pages_l85_8560


namespace NUMINAMATH_GPT_inequality_solution_set_l85_8545

theorem inequality_solution_set (m n : ℝ) 
    (h₁ : ∀ x : ℝ, mx - n > 0 ↔ x < 1 / 3) 
    (h₂ : m + n < 0) 
    (h₃ : m = 3 * n) 
    (h₄ : n < 0) : 
    ∀ x : ℝ, (m + n) * x < n - m ↔ x > -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l85_8545


namespace NUMINAMATH_GPT_amara_remaining_clothes_l85_8524

noncomputable def remaining_clothes (initial total_donated thrown_away : ℕ) : ℕ :=
  initial - (total_donated + thrown_away)

theorem amara_remaining_clothes : 
  ∀ (initial donated_first donated_second thrown_away : ℕ), initial = 100 → donated_first = 5 → donated_second = 15 → thrown_away = 15 → 
  remaining_clothes initial (donated_first + donated_second) thrown_away = 65 := 
by 
  intros initial donated_first donated_second thrown_away hinital hdonated_first hdonated_second hthrown_away
  rw [hinital, hdonated_first, hdonated_second, hthrown_away]
  unfold remaining_clothes
  norm_num

end NUMINAMATH_GPT_amara_remaining_clothes_l85_8524


namespace NUMINAMATH_GPT_constant_sum_powers_l85_8506

theorem constant_sum_powers (n : ℕ) (x y z : ℝ) (h_sum : x + y + z = 0) (h_prod : x * y * z = 1) :
  (∀ x y z : ℝ, x + y + z = 0 → x * y * z = 1 → x^n + y^n + z^n = x^n + y^n + z^n ↔ (n = 1 ∨ n = 3)) :=
by
  sorry

end NUMINAMATH_GPT_constant_sum_powers_l85_8506


namespace NUMINAMATH_GPT_valid_outfit_choices_l85_8539

def shirts := 6
def pants := 6
def hats := 12
def patterned_hats := 6

theorem valid_outfit_choices : 
  (shirts * pants * hats) - shirts - (patterned_hats * shirts * (pants - 1)) = 246 := by
  sorry

end NUMINAMATH_GPT_valid_outfit_choices_l85_8539


namespace NUMINAMATH_GPT_like_terms_exponent_l85_8502

theorem like_terms_exponent (m n : ℤ) (h₁ : n = 2) (h₂ : m = 1) : m - n = -1 :=
by
  sorry

end NUMINAMATH_GPT_like_terms_exponent_l85_8502


namespace NUMINAMATH_GPT_num_complementary_sets_eq_117_l85_8511

structure Card :=
(shape : Type)
(color : Type)
(shade : Type)

def deck_condition: Prop := 
  ∃ (deck : List Card), 
  deck.length = 27 ∧
  ∀ c1 c2 c3, c1 ∈ deck ∧ c2 ∈ deck ∧ c3 ∈ deck →
  (c1.shape ≠ c2.shape ∨ c2.shape ≠ c3.shape ∨ c1.shape = c3.shape) ∧
  (c1.color ≠ c2.color ∨ c2.color ≠ c3.color ∨ c1.color = c3.color) ∧
  (c1.shade ≠ c2.shade ∨ c2.shade ≠ c3.shade ∨ c1.shade = c3.shade)

theorem num_complementary_sets_eq_117 :
  deck_condition → ∃ sets : List (List Card), sets.length = 117 := sorry

end NUMINAMATH_GPT_num_complementary_sets_eq_117_l85_8511


namespace NUMINAMATH_GPT_complement_of_N_is_135_l85_8596

-- Define the universal set M and subset N
def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 4}

-- Prove that the complement of N in M is {1, 3, 5}
theorem complement_of_N_is_135 : M \ N = {1, 3, 5} := 
by
  sorry

end NUMINAMATH_GPT_complement_of_N_is_135_l85_8596


namespace NUMINAMATH_GPT_b2_b7_product_l85_8522

variable {b : ℕ → ℤ}

-- Define the conditions: b is an arithmetic sequence and b_4 * b_5 = 15
def is_arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

axiom increasing_arithmetic_sequence : is_arithmetic_sequence b
axiom b4_b5_product : b 4 * b 5 = 15

-- The target theorem to prove
theorem b2_b7_product : b 2 * b 7 = -9 :=
sorry

end NUMINAMATH_GPT_b2_b7_product_l85_8522


namespace NUMINAMATH_GPT_han_xin_troop_min_soldiers_l85_8561

theorem han_xin_troop_min_soldiers (n : ℕ) : 
  (n % 3 = 2) ∧ (n % 5 = 3) ∧ (n % 7 = 4) → n = 53 :=
  sorry

end NUMINAMATH_GPT_han_xin_troop_min_soldiers_l85_8561


namespace NUMINAMATH_GPT_floor_sub_le_l85_8599

theorem floor_sub_le : ∀ (x y : ℝ), ⌊x - y⌋ ≤ ⌊x⌋ - ⌊y⌋ :=
by sorry

end NUMINAMATH_GPT_floor_sub_le_l85_8599


namespace NUMINAMATH_GPT_sequence_from_520_to_523_is_0_to_3_l85_8543

theorem sequence_from_520_to_523_is_0_to_3 
  (repeating_pattern : ℕ → ℕ)
  (h_periodic : ∀ n, repeating_pattern (n + 5) = repeating_pattern n) :
  ((repeating_pattern 520, repeating_pattern 521, repeating_pattern 522, repeating_pattern 523) = (repeating_pattern 0, repeating_pattern 1, repeating_pattern 2, repeating_pattern 3)) :=
by {
  sorry
}

end NUMINAMATH_GPT_sequence_from_520_to_523_is_0_to_3_l85_8543


namespace NUMINAMATH_GPT_nguyen_fabric_needs_l85_8527

def yards_to_feet (yards : ℝ) := yards * 3
def total_fabric_needed (pairs : ℝ) (fabric_per_pair : ℝ) := pairs * fabric_per_pair
def fabric_still_needed (total_needed : ℝ) (already_have : ℝ) := total_needed - already_have

theorem nguyen_fabric_needs :
  let pairs := 7
  let fabric_per_pair := 8.5
  let yards_have := 3.5
  let feet_have := yards_to_feet yards_have
  let total_needed := total_fabric_needed pairs fabric_per_pair
  fabric_still_needed total_needed feet_have = 49 :=
by
  sorry

end NUMINAMATH_GPT_nguyen_fabric_needs_l85_8527


namespace NUMINAMATH_GPT_meeting_point_ratio_l85_8531

theorem meeting_point_ratio (v1 v2 : ℝ) (TA TB : ℝ)
  (h1 : TA = 45 * v2)
  (h2 : TB = 20 * v1)
  (h3 : (TA / v1) - (TB / v2) = 11) :
  TA / TB = 9 / 5 :=
by sorry

end NUMINAMATH_GPT_meeting_point_ratio_l85_8531


namespace NUMINAMATH_GPT_examination_total_students_l85_8512

theorem examination_total_students (T : ℝ) :
  (0.35 * T + 520) = T ↔ T = 800 :=
by 
  sorry

end NUMINAMATH_GPT_examination_total_students_l85_8512


namespace NUMINAMATH_GPT_find_weight_B_l85_8584

-- Define the weights of A, B, and C
variables (A B C : ℝ)

-- Conditions
def avg_weight_ABC := A + B + C = 135
def avg_weight_AB := A + B = 80
def avg_weight_BC := B + C = 86

-- The statement to be proved
theorem find_weight_B (h1: avg_weight_ABC A B C) (h2: avg_weight_AB A B) (h3: avg_weight_BC B C) : B = 31 :=
sorry

end NUMINAMATH_GPT_find_weight_B_l85_8584


namespace NUMINAMATH_GPT_find_some_number_l85_8553

def some_number (x : Int) (some_num : Int) : Prop :=
  (3 < x ∧ x < 10) ∧
  (5 < x ∧ x < 18) ∧
  (9 > x ∧ x > -2) ∧
  (8 > x ∧ x > 0) ∧
  (x + some_num < 9)

theorem find_some_number :
  ∀ (some_num : Int), some_number 7 some_num → some_num < 2 :=
by
  intros some_num H
  sorry

end NUMINAMATH_GPT_find_some_number_l85_8553


namespace NUMINAMATH_GPT_cubes_with_no_colored_faces_l85_8580

theorem cubes_with_no_colored_faces (width length height : ℕ) (total_cubes cube_side : ℕ) :
  width = 6 ∧ length = 5 ∧ height = 4 ∧ total_cubes = 120 ∧ cube_side = 1 →
  (width - 2) * (length - 2) * (height - 2) = 24 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_cubes_with_no_colored_faces_l85_8580


namespace NUMINAMATH_GPT_eval_expression_l85_8501

theorem eval_expression : (-2 ^ 4) + 3 * (-1) ^ 6 - (-2) ^ 3 = -5 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l85_8501


namespace NUMINAMATH_GPT_train_cross_bridge_time_l85_8554

theorem train_cross_bridge_time
  (length_train : ℕ) (speed_train_kmph : ℕ) (length_bridge : ℕ) 
  (km_to_m : ℕ) (hour_to_s : ℕ)
  (h1 : length_train = 165) 
  (h2 : speed_train_kmph = 54) 
  (h3 : length_bridge = 720) 
  (h4 : km_to_m = 1000) 
  (h5 : hour_to_s = 3600) 
  : (length_train + length_bridge) / ((speed_train_kmph * km_to_m) / hour_to_s) = 59 := 
sorry

end NUMINAMATH_GPT_train_cross_bridge_time_l85_8554


namespace NUMINAMATH_GPT_max_quotient_l85_8585

theorem max_quotient (x y : ℝ) (h1 : -5 ≤ x) (h2 : x ≤ -3) (h3 : 3 ≤ y) (h4 : y ≤ 6) : 
  ∃ z, z = (x + y) / x ∧ ∀ w, w = (x + y) / x → w ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_max_quotient_l85_8585


namespace NUMINAMATH_GPT_avery_shirts_count_l85_8575

theorem avery_shirts_count {S : ℕ} (h_total : S + 2 * S + S = 16) : S = 4 :=
by
  sorry

end NUMINAMATH_GPT_avery_shirts_count_l85_8575


namespace NUMINAMATH_GPT_solve_for_n_l85_8579

theorem solve_for_n (n : ℕ) (h : (8 ^ n) * (8 ^ n) * (8 ^ n) = 64 ^ 3) : n = 2 :=
by sorry

end NUMINAMATH_GPT_solve_for_n_l85_8579


namespace NUMINAMATH_GPT_matrix_determinant_equality_l85_8519

open Complex Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

theorem matrix_determinant_equality (A B : Matrix n n ℂ) (x : ℂ) 
  (h1 : A ^ 2 + B ^ 2 = 2 * A * B) :
  det (A - x • 1) = det (B - x • 1) :=
  sorry

end NUMINAMATH_GPT_matrix_determinant_equality_l85_8519


namespace NUMINAMATH_GPT_geometric_sequence_fifth_term_l85_8559

theorem geometric_sequence_fifth_term (r : ℕ) (h₁ : 5 * r^3 = 405) : 5 * r^4 = 405 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_fifth_term_l85_8559


namespace NUMINAMATH_GPT_polynomial_R_result_l85_8555

noncomputable def polynomial_Q_R (z : ℤ) : Prop :=
  ∃ Q R : Polynomial ℂ, 
  z ^ 2020 + 1 = (z ^ 2 - z + 1) * Q + R ∧ R.degree < 2 ∧ R = 2

theorem polynomial_R_result :
  polynomial_Q_R z :=
by 
  sorry

end NUMINAMATH_GPT_polynomial_R_result_l85_8555


namespace NUMINAMATH_GPT_probability_four_friends_same_group_l85_8569

-- Define the conditions of the problem
def total_students : ℕ := 900
def groups : ℕ := 5
def friends : ℕ := 4
def probability_per_group : ℚ := 1 / groups

-- Define the statement we need to prove
theorem probability_four_friends_same_group :
  (probability_per_group * probability_per_group * probability_per_group) = 1 / 125 :=
sorry

end NUMINAMATH_GPT_probability_four_friends_same_group_l85_8569


namespace NUMINAMATH_GPT_probability_heads_and_3_l85_8563

noncomputable def biased_coin_heads_prob : ℝ := 0.4
def die_sides : ℕ := 8

theorem probability_heads_and_3 : biased_coin_heads_prob * (1 / die_sides) = 0.05 := sorry

end NUMINAMATH_GPT_probability_heads_and_3_l85_8563


namespace NUMINAMATH_GPT_train_pass_bridge_time_l85_8567

/-- A train is 460 meters long and runs at a speed of 45 km/h. The bridge is 140 meters long. 
Prove that the time it takes for the train to pass the bridge is 48 seconds. -/
theorem train_pass_bridge_time (train_length : ℝ) (bridge_length : ℝ) (speed_kmh : ℝ) 
  (h_train_length : train_length = 460) 
  (h_bridge_length : bridge_length = 140)
  (h_speed_kmh : speed_kmh = 45)
  : (train_length + bridge_length) / (speed_kmh * 1000 / 3600) = 48 := 
by
  sorry

end NUMINAMATH_GPT_train_pass_bridge_time_l85_8567


namespace NUMINAMATH_GPT_lcm_of_9_12_15_l85_8549

theorem lcm_of_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := by
  sorry

end NUMINAMATH_GPT_lcm_of_9_12_15_l85_8549


namespace NUMINAMATH_GPT_compare_abc_l85_8515

open Real

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Assuming the conditions
axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom derivative : ∀ x : ℝ, f' x = deriv f x
axiom monotonicity_condition : ∀ x > 0, x * f' x < f x

-- Definitions of a, b, and c
noncomputable def a := 2 * f (1 / 2)
noncomputable def b := - (1 / 2) * f (-2)
noncomputable def c := - (1 / log 2) * f (log (1 / 2))

theorem compare_abc : a > c ∧ c > b := sorry

end NUMINAMATH_GPT_compare_abc_l85_8515


namespace NUMINAMATH_GPT_multiple_of_old_edition_l85_8525

theorem multiple_of_old_edition 
  (new_pages: ℕ) 
  (old_pages: ℕ) 
  (difference: ℕ) 
  (m: ℕ) 
  (h1: new_pages = 450) 
  (h2: old_pages = 340) 
  (h3: 450 = 340 * m - 230) : 
  m = 2 :=
sorry

end NUMINAMATH_GPT_multiple_of_old_edition_l85_8525


namespace NUMINAMATH_GPT_trader_profit_loss_l85_8568

noncomputable def profit_loss_percentage (sp1 sp2: ℝ) (gain_loss_rate1 gain_loss_rate2: ℝ) : ℝ :=
  let cp1 := sp1 / (1 + gain_loss_rate1)
  let cp2 := sp2 / (1 - gain_loss_rate2)
  let tcp := cp1 + cp2
  let tsp := sp1 + sp2
  let profit_or_loss := tsp - tcp
  profit_or_loss / tcp * 100

theorem trader_profit_loss : 
  profit_loss_percentage 325475 325475 0.15 0.15 = -2.33 := 
by 
  sorry

end NUMINAMATH_GPT_trader_profit_loss_l85_8568


namespace NUMINAMATH_GPT_original_number_is_14_l85_8530

def two_digit_number_increased_by_2_or_4_results_fourfold (x : ℕ) : Prop :=
  (x >= 10) ∧ (x < 100) ∧ 
  (∃ (a b : ℕ), a + 2 = ((x / 10 + 2) % 10) ∧ b + 2 = (x % 10)) ∧
  (4 * x = ((x / 10 + 2) * 10 + (x % 10 + 2)) ∨ 
   4 * x = ((x / 10 + 2) * 10 + (x % 10 + 4)) ∨ 
   4 * x = ((x / 10 + 4) * 10 + (x % 10 + 2)) ∨ 
   4 * x = ((x / 10 + 4) * 10 + (x % 10 + 4)))

theorem original_number_is_14 : ∃ x : ℕ, two_digit_number_increased_by_2_or_4_results_fourfold x ∧ x = 14 :=
by
  sorry

end NUMINAMATH_GPT_original_number_is_14_l85_8530


namespace NUMINAMATH_GPT_intersection_complement_l85_8586

open Set

variable (x : ℝ)

def M : Set ℝ := { x | -1 < x ∧ x < 2 }
def N : Set ℝ := { x | 1 ≤ x }

theorem intersection_complement :
  M ∩ (univ \ N) = { x | -1 < x ∧ x < 1 } := by
  sorry

end NUMINAMATH_GPT_intersection_complement_l85_8586


namespace NUMINAMATH_GPT_find_a_and_root_l85_8588

def equation_has_double_root (a x : ℝ) : Prop :=
  a * x^2 + 4 * x - 1 = 0

theorem find_a_and_root (a x : ℝ)
  (h_eqn : equation_has_double_root a x)
  (h_discriminant : 16 + 4 * a = 0) :
  a = -4 ∧ x = 1 / 2 :=
sorry

end NUMINAMATH_GPT_find_a_and_root_l85_8588


namespace NUMINAMATH_GPT_find_integer_n_l85_8577

theorem find_integer_n : 
  ∃ n : ℤ, 50 ≤ n ∧ n ≤ 150 ∧ (n % 7 = 0) ∧ (n % 9 = 3) ∧ (n % 4 = 3) ∧ n = 147 :=
by 
  -- sorry is used here as a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_find_integer_n_l85_8577


namespace NUMINAMATH_GPT_problem_statement_l85_8595

variable (a b c : ℝ)

-- Conditions given in the problem
axiom h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -24
axiom h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 8

-- The Lean statement for the proof problem
theorem problem_statement (a b c : ℝ) (h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -24)
    (h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 8) :
    (b / (a + b) + c / (b + c) + a / (c + a)) = 19 / 2 :=
sorry

end NUMINAMATH_GPT_problem_statement_l85_8595


namespace NUMINAMATH_GPT_evaluate_fraction_l85_8516

noncomputable section

variables (u v : ℂ)
variables (h1 : u ≠ 0) (h2 : v ≠ 0) (h3 : u^2 + u * v + v^2 = 0)

theorem evaluate_fraction : (u^7 + v^7) / (u + v)^7 = -2 := by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l85_8516


namespace NUMINAMATH_GPT_percentage_of_alcohol_in_original_solution_l85_8593

noncomputable def alcohol_percentage_in_original_solution (P: ℝ) (V_original: ℝ) (V_water: ℝ) (percentage_new: ℝ): ℝ :=
  (P * V_original) / (V_original + V_water) * 100

theorem percentage_of_alcohol_in_original_solution : 
  ∀ (P: ℝ) (V_original : ℝ) (V_water : ℝ) (percentage_new : ℝ), 
  V_original = 3 → 
  V_water = 1 → 
  percentage_new = 24.75 →
  alcohol_percentage_in_original_solution P V_original V_water percentage_new = 33 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_alcohol_in_original_solution_l85_8593


namespace NUMINAMATH_GPT_balcony_more_than_orchestra_l85_8592

theorem balcony_more_than_orchestra (x y : ℕ) 
  (h1 : x + y = 370) 
  (h2 : 12 * x + 8 * y = 3320) : y - x = 190 :=
sorry

end NUMINAMATH_GPT_balcony_more_than_orchestra_l85_8592


namespace NUMINAMATH_GPT_find_f_inv_difference_l85_8574

axiom f : ℤ → ℤ
axiom f_inv : ℤ → ℤ
axiom f_has_inverse : ∀ x : ℤ, f_inv (f x) = x ∧ f (f_inv x) = x
axiom f_inverse_conditions : ∀ x : ℤ, f (x + 2) = f_inv (x - 1)

theorem find_f_inv_difference :
  f_inv 2004 - f_inv 1 = 4006 :=
sorry

end NUMINAMATH_GPT_find_f_inv_difference_l85_8574


namespace NUMINAMATH_GPT_range_of_inverse_proportion_function_l85_8552

noncomputable def f (x : ℝ) : ℝ := 6 / x

theorem range_of_inverse_proportion_function (x : ℝ) (hx : x > 2) : 
  0 < f x ∧ f x < 3 :=
sorry

end NUMINAMATH_GPT_range_of_inverse_proportion_function_l85_8552


namespace NUMINAMATH_GPT_total_fruits_l85_8535

theorem total_fruits (total_baskets apples_baskets oranges_baskets apples_per_basket oranges_per_basket pears_per_basket : ℕ)
  (h1 : total_baskets = 127)
  (h2 : apples_baskets = 79)
  (h3 : oranges_baskets = 30)
  (h4 : apples_per_basket = 75)
  (h5 : oranges_per_basket = 143)
  (h6 : pears_per_basket = 56)
  : 79 * 75 + 30 * 143 + (127 - (79 + 30)) * 56 = 11223 := by
  sorry

end NUMINAMATH_GPT_total_fruits_l85_8535


namespace NUMINAMATH_GPT_inscribed_circle_diameter_of_right_triangle_l85_8558

theorem inscribed_circle_diameter_of_right_triangle (a b : ℕ) (hc : a = 8) (hb : b = 15) :
  2 * (60 / (a + b + Int.sqrt (a ^ 2 + b ^ 2))) = 6 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_circle_diameter_of_right_triangle_l85_8558


namespace NUMINAMATH_GPT_find_sample_size_l85_8590

theorem find_sample_size :
  ∀ (n : ℕ), 
    (∃ x : ℝ,
      2 * x + 3 * x + 4 * x + 6 * x + 4 * x + x = 1 ∧
      2 * n * x + 3 * n * x + 4 * n * x = 27) →
    n = 60 :=
by
  intro n
  rintro ⟨x, h1, h2⟩
  sorry

end NUMINAMATH_GPT_find_sample_size_l85_8590


namespace NUMINAMATH_GPT_find_income_separator_l85_8534

-- Define the income and tax parameters
def income : ℝ := 60000
def total_tax : ℝ := 8000
def rate1 : ℝ := 0.10
def rate2 : ℝ := 0.20

-- Define the function for total tax calculation
def tax (I : ℝ) : ℝ := rate1 * I + rate2 * (income - I)

theorem find_income_separator (I : ℝ) (h: tax I = total_tax) : I = 40000 :=
by sorry

end NUMINAMATH_GPT_find_income_separator_l85_8534


namespace NUMINAMATH_GPT_cricket_current_average_l85_8578

theorem cricket_current_average (A : ℕ) (h1: 10 * A + 77 = 11 * (A + 4)) : 
  A = 33 := 
by 
  sorry

end NUMINAMATH_GPT_cricket_current_average_l85_8578


namespace NUMINAMATH_GPT_num_of_veg_people_l85_8533

def only_veg : ℕ := 19
def both_veg_nonveg : ℕ := 12

theorem num_of_veg_people : only_veg + both_veg_nonveg = 31 := by 
  sorry

end NUMINAMATH_GPT_num_of_veg_people_l85_8533


namespace NUMINAMATH_GPT_rectangle_length_l85_8581

theorem rectangle_length
  (w l : ℝ)
  (h1 : l = 4 * w)
  (h2 : l * w = 100) :
  l = 20 :=
sorry

end NUMINAMATH_GPT_rectangle_length_l85_8581


namespace NUMINAMATH_GPT_isosceles_triangle_vertex_angle_l85_8542

theorem isosceles_triangle_vertex_angle (a b : ℕ) (h : a = 2 * b) 
  (h1 : a + b + b = 180): a = 90 ∨ a = 36 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_vertex_angle_l85_8542


namespace NUMINAMATH_GPT_cost_price_equals_720_l85_8500

theorem cost_price_equals_720 (C : ℝ) :
  (0.27 * C - 0.12 * C = 108) → (C = 720) :=
by
  sorry

end NUMINAMATH_GPT_cost_price_equals_720_l85_8500


namespace NUMINAMATH_GPT_distance_traveled_l85_8572

-- Given conditions
def speed : ℕ := 100 -- Speed in km/hr
def time : ℕ := 5    -- Time in hours

-- The goal is to prove the distance traveled is 500 km
theorem distance_traveled : speed * time = 500 := by
  -- we state the proof goal
  sorry

end NUMINAMATH_GPT_distance_traveled_l85_8572


namespace NUMINAMATH_GPT_f_eq_g_iff_l85_8591

noncomputable def f (m n x : ℝ) := m * x^2 + n * x
noncomputable def g (p q x : ℝ) := p * x + q

theorem f_eq_g_iff (m n p q : ℝ) :
  (∀ x, f m n (g p q x) = g p q (f m n x)) ↔ 2 * m = n := by
  sorry

end NUMINAMATH_GPT_f_eq_g_iff_l85_8591


namespace NUMINAMATH_GPT_average_of_six_numbers_l85_8526

theorem average_of_six_numbers (A : ℝ) (x y z w u v : ℝ)
  (h1 : (x + y + z + w + u + v) / 6 = A)
  (h2 : (x + y) / 2 = 1.1)
  (h3 : (z + w) / 2 = 1.4)
  (h4 : (u + v) / 2 = 5) :
  A = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_average_of_six_numbers_l85_8526


namespace NUMINAMATH_GPT_distance_proof_l85_8523

noncomputable section

open Real

-- Define the given conditions
def AB : Real := 3 * sqrt 3
def BC : Real := 2
def theta : Real := 60 -- angle in degrees
def phi : Real := 180 - theta -- supplementary angle to use in the Law of Cosines

-- Helper function to convert degrees to radians
def deg_to_rad (d : Real) : Real := d * (π / 180)

-- Define the law of cosines to compute AC
def distance_AC (AB BC θ : Real) : Real := 
  sqrt (AB^2 + BC^2 - 2 * AB * BC * cos (deg_to_rad θ))

-- The theorem to prove
theorem distance_proof : distance_AC AB BC phi = 7 :=
by
  sorry

end NUMINAMATH_GPT_distance_proof_l85_8523


namespace NUMINAMATH_GPT_triangle_is_right_angled_l85_8576

theorem triangle_is_right_angled
  (a b c : ℝ)
  (h1 : a ≠ c)
  (h2 : a > 0) 
  (h3 : b > 0) 
  (h4 : c > 0)
  (h5 : ∃ x : ℝ, x^2 + 2*a*x + b^2 = 0 ∧ x^2 + 2*c*x - b^2 = 0 ∧ x ≠ 0) :
  c^2 + b^2 = a^2 :=
by sorry

end NUMINAMATH_GPT_triangle_is_right_angled_l85_8576


namespace NUMINAMATH_GPT_imaginary_part_of_z_is_sqrt2_div2_l85_8514

open Complex

noncomputable def z : ℂ := abs (1 - I) / (1 - I)

theorem imaginary_part_of_z_is_sqrt2_div2 : z.im = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_z_is_sqrt2_div2_l85_8514


namespace NUMINAMATH_GPT_max_consecutive_sum_terms_l85_8582

theorem max_consecutive_sum_terms (S : ℤ) (n : ℕ) (H1 : S = 2015) (H2 : 0 < n) :
  (∃ a : ℤ, S = (a * n + (n * (n - 1)) / 2)) → n = 4030 :=
sorry

end NUMINAMATH_GPT_max_consecutive_sum_terms_l85_8582


namespace NUMINAMATH_GPT_expand_expression_l85_8536

theorem expand_expression (x : ℝ) : 
  (11 * x^2 + 5 * x - 3) * (3 * x^3) = 33 * x^5 + 15 * x^4 - 9 * x^3 :=
by 
  sorry

end NUMINAMATH_GPT_expand_expression_l85_8536


namespace NUMINAMATH_GPT_sheila_will_attend_picnic_l85_8598

noncomputable def prob_sheila_attends_picnic (P_Rain P_Attend_if_Rain P_Attend_if_Sunny P_Special : ℝ) : ℝ :=
  let P_Sunny := 1 - P_Rain
  let P_Rain_and_Attend := P_Rain * P_Attend_if_Rain
  let P_Sunny_and_Attend := P_Sunny * P_Attend_if_Sunny
  let P_Attends := P_Rain_and_Attend + P_Sunny_and_Attend + P_Special - P_Rain_and_Attend * P_Special - P_Sunny_and_Attend * P_Special
  P_Attends

theorem sheila_will_attend_picnic :
  prob_sheila_attends_picnic 0.3 0.25 0.7 0.15 = 0.63025 :=
by
  sorry

end NUMINAMATH_GPT_sheila_will_attend_picnic_l85_8598


namespace NUMINAMATH_GPT_comedies_in_terms_of_a_l85_8565

variable (T a : ℝ)
variables (Comedies Dramas Action : ℝ)
axiom Condition1 : Comedies = 0.64 * T
axiom Condition2 : Dramas = 5 * a
axiom Condition3 : Action = a
axiom Condition4 : Comedies + Dramas + Action = T

theorem comedies_in_terms_of_a : Comedies = 10.67 * a :=
by sorry

end NUMINAMATH_GPT_comedies_in_terms_of_a_l85_8565


namespace NUMINAMATH_GPT_james_earnings_per_subscriber_is_9_l85_8510

/-
Problem:
James streams on Twitch. He had 150 subscribers and then someone gifted 50 subscribers. If he gets a certain amount per month per subscriber and now makes $1800 a month, how much does he make per subscriber?
-/

def initial_subscribers : ℕ := 150
def gifted_subscribers : ℕ := 50
def total_subscribers := initial_subscribers + gifted_subscribers
def total_earnings : ℤ := 1800

def earnings_per_subscriber := total_earnings / total_subscribers

/-
Theorem: James makes $9 per month for each subscriber.
-/
theorem james_earnings_per_subscriber_is_9 : earnings_per_subscriber = 9 := by
  -- to be filled in with proof steps
  sorry

end NUMINAMATH_GPT_james_earnings_per_subscriber_is_9_l85_8510


namespace NUMINAMATH_GPT_product_of_cosines_value_l85_8537

noncomputable def product_of_cosines : ℝ :=
  (1 + Real.cos (Real.pi / 12)) * (1 + Real.cos (5 * Real.pi / 12)) *
  (1 + Real.cos (7 * Real.pi / 12)) * (1 + Real.cos (11 * Real.pi / 12))

theorem product_of_cosines_value :
  product_of_cosines = 1 / 16 :=
by
  sorry

end NUMINAMATH_GPT_product_of_cosines_value_l85_8537


namespace NUMINAMATH_GPT_duckweed_quarter_covered_l85_8503

theorem duckweed_quarter_covered (N : ℕ) (h1 : N = 64) (h2 : ∀ n : ℕ, n < N → (n + 1 < N) → ∃ k, k = n + 1) :
  N - 2 = 62 :=
by
  sorry

end NUMINAMATH_GPT_duckweed_quarter_covered_l85_8503


namespace NUMINAMATH_GPT_people_per_table_l85_8594

theorem people_per_table (kids adults tables : ℕ) (h_kids : kids = 45) (h_adults : adults = 123) (h_tables : tables = 14) :
  ((kids + adults) / tables) = 12 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_people_per_table_l85_8594


namespace NUMINAMATH_GPT_non_decreasing_f_f_equal_2_at_2_addition_property_under_interval_rule_final_statement_l85_8505

open Set

noncomputable def f : ℝ → ℝ := sorry

theorem non_decreasing_f (x y : ℝ) (h : x < y) (hx : x ∈ Icc (0 : ℝ) 2) (hy : y ∈ Icc (0 : ℝ) 2) : f x ≤ f y := sorry

theorem f_equal_2_at_2 : f 2 = 2 := sorry

theorem addition_property (x : ℝ) (hx : x ∈ Icc (0 :ℝ) 2) : f x + f (2 - x) = 2 := sorry

theorem under_interval_rule (x : ℝ) (hx : x ∈ Icc (1.5 :ℝ) 2) : f x ≤ 2 * (x - 1) := sorry

theorem final_statement : ∀ x ∈ Icc (0:ℝ) 1, f (f x) ∈ Icc (0:ℝ) 1 := sorry

end NUMINAMATH_GPT_non_decreasing_f_f_equal_2_at_2_addition_property_under_interval_rule_final_statement_l85_8505


namespace NUMINAMATH_GPT_solve_system_l85_8571

theorem solve_system (x y : ℚ) 
  (h1 : x + 2 * y = -1) 
  (h2 : 2 * x + y = 3) : 
  x + y = 2 / 3 := 
sorry

end NUMINAMATH_GPT_solve_system_l85_8571


namespace NUMINAMATH_GPT_min_coins_cover_99_l85_8573

def coin_values : List ℕ := [1, 5, 10, 25, 50]

noncomputable def min_coins_cover (n : ℕ) : ℕ := sorry

theorem min_coins_cover_99 : min_coins_cover 99 = 9 :=
  sorry

end NUMINAMATH_GPT_min_coins_cover_99_l85_8573


namespace NUMINAMATH_GPT_candies_per_pack_l85_8521

-- Conditions in Lean:
def total_candies : ℕ := 60
def packs_initially (packs_after : ℕ) : ℕ := packs_after + 1
def packs_after : ℕ := 2
def pack_count : ℕ := packs_initially packs_after

-- The statement of the proof problem:
theorem candies_per_pack : 
  total_candies / pack_count = 20 :=
by
  sorry

end NUMINAMATH_GPT_candies_per_pack_l85_8521


namespace NUMINAMATH_GPT_average_monthly_balance_l85_8544

theorem average_monthly_balance
  (jan feb mar apr may : ℕ) 
  (Hjan : jan = 200)
  (Hfeb : feb = 300)
  (Hmar : mar = 100)
  (Hapr : apr = 250)
  (Hmay : may = 150) :
  (jan + feb + mar + apr + may) / 5 = 200 := 
  by
  sorry

end NUMINAMATH_GPT_average_monthly_balance_l85_8544


namespace NUMINAMATH_GPT_complex_division_l85_8557

-- Define complex numbers and imaginary unit
def i : ℂ := Complex.I

theorem complex_division : (3 + 4 * i) / (1 + i) = (7 / 2) + (1 / 2) * i :=
by
  sorry

end NUMINAMATH_GPT_complex_division_l85_8557


namespace NUMINAMATH_GPT_angle_condition_l85_8538

theorem angle_condition
  {θ : ℝ}
  (h₀ : 0 ≤ θ)
  (h₁ : θ < π)
  (h₂ : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → x^2 * Real.cos θ - x * (1 - x) + 2 * (1 - x)^2 * Real.sin θ > 0) :
  0 < θ ∧ θ < π / 2 :=
by
  sorry

end NUMINAMATH_GPT_angle_condition_l85_8538


namespace NUMINAMATH_GPT_rowing_speed_in_still_water_l85_8564

variable (v c t : ℝ)
variable (h1 : c = 1.3)
variable (h2 : 2 * ((v - c) * t) = ((v + c) * t))

theorem rowing_speed_in_still_water : v = 3.9 := by
  sorry

end NUMINAMATH_GPT_rowing_speed_in_still_water_l85_8564


namespace NUMINAMATH_GPT_probability_heads_3_ace_l85_8562

def fair_coin_flip : ℕ := 2
def six_sided_die : ℕ := 6
def standard_deck_cards : ℕ := 52

def successful_outcomes : ℕ := 1 * 1 * 4
def total_possible_outcomes : ℕ := fair_coin_flip * six_sided_die * standard_deck_cards

theorem probability_heads_3_ace :
  (successful_outcomes : ℚ) / (total_possible_outcomes : ℚ) = 1 / 156 := 
sorry

end NUMINAMATH_GPT_probability_heads_3_ace_l85_8562


namespace NUMINAMATH_GPT_solve_for_n_l85_8546

theorem solve_for_n (n : ℕ) : 
  9^n * 9^n * 9^(2*n) = 81^4 → n = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_n_l85_8546


namespace NUMINAMATH_GPT_line_through_two_points_l85_8509

-- Define the points (2,5) and (0,3)
structure Point where
  x : ℝ
  y : ℝ

def P1 : Point := {x := 2, y := 5}
def P2 : Point := {x := 0, y := 3}

-- General form of a line equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the target line equation as x - y + 3 = 0
def targetLine : Line := {a := 1, b := -1, c := 3}

-- The proof statement to show that the general equation of the line passing through the points (2, 5) and (0, 3) is x - y + 3 = 0
theorem line_through_two_points : ∃ a b c, ∀ x y : ℝ, 
    (a * x + b * y + c = 0) ↔ 
    ((∀ {P : Point}, P = P1 → targetLine.a * P.x + targetLine.b * P.y + targetLine.c = 0) ∧ 
     (∀ {P : Point}, P = P2 → targetLine.a * P.x + targetLine.b * P.y + targetLine.c = 0)) :=
sorry

end NUMINAMATH_GPT_line_through_two_points_l85_8509


namespace NUMINAMATH_GPT_determine_n_l85_8504

theorem determine_n (n : ℕ) (h : 9^4 = 3^n) : n = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_determine_n_l85_8504


namespace NUMINAMATH_GPT_ceil_add_eq_double_of_int_l85_8548

theorem ceil_add_eq_double_of_int {x : ℤ} (h : ⌈(x : ℝ)⌉ + ⌊(x : ℝ)⌋ = 2 * (x : ℝ)) : ⌈(x : ℝ)⌉ + x = 2 * x :=
by
  sorry

end NUMINAMATH_GPT_ceil_add_eq_double_of_int_l85_8548


namespace NUMINAMATH_GPT_beef_weight_loss_percentage_l85_8556

theorem beef_weight_loss_percentage (weight_before weight_after weight_lost_percentage : ℝ) 
  (before_process : weight_before = 861.54)
  (after_process : weight_after = 560) 
  (weight_lost : (weight_before - weight_after) = 301.54)
  : weight_lost_percentage = 34.99 :=
by
  sorry

end NUMINAMATH_GPT_beef_weight_loss_percentage_l85_8556


namespace NUMINAMATH_GPT_stratified_sampling_correct_l85_8570

-- Defining the conditions
def first_grade_students : ℕ := 600
def second_grade_students : ℕ := 680
def third_grade_students : ℕ := 720
def total_sample_size : ℕ := 50
def total_students := first_grade_students + second_grade_students + third_grade_students

-- Expected number of students to be sampled from first, second, and third grades
def expected_first_grade_sample := total_sample_size * first_grade_students / total_students
def expected_second_grade_sample := total_sample_size * second_grade_students / total_students
def expected_third_grade_sample := total_sample_size * third_grade_students / total_students

-- Main theorem statement
theorem stratified_sampling_correct :
  expected_first_grade_sample = 15 ∧
  expected_second_grade_sample = 17 ∧
  expected_third_grade_sample = 18 := by
  sorry

end NUMINAMATH_GPT_stratified_sampling_correct_l85_8570


namespace NUMINAMATH_GPT_find_number_l85_8566

theorem find_number (A : ℕ) (B : ℕ) (H1 : B = 300) (H2 : Nat.lcm A B = 2310) (H3 : Nat.gcd A B = 30) : A = 231 := 
by 
  sorry

end NUMINAMATH_GPT_find_number_l85_8566


namespace NUMINAMATH_GPT_monomials_like_terms_l85_8507

variable (m n : ℤ)

theorem monomials_like_terms (hm : m = 3) (hn : n = 1) : m - 2 * n = 1 :=
by
  sorry

end NUMINAMATH_GPT_monomials_like_terms_l85_8507


namespace NUMINAMATH_GPT_clock_hands_coincide_22_times_clock_hands_straight_angle_24_times_clock_hands_right_angle_48_times_l85_8551

theorem clock_hands_coincide_22_times
  (minute_hand_cycles_24_hours : ℕ := 24)
  (hour_hand_cycles_24_hours : ℕ := 2)
  (minute_hand_overtakes_hour_hand_per_12_hours : ℕ := 11) :
  2 * minute_hand_overtakes_hour_hand_per_12_hours = 22 :=
by
  -- Proof should be filled here
  sorry

theorem clock_hands_straight_angle_24_times
  (hours_in_day : ℕ := 24)
  (straight_angle_per_hour : ℕ := 1) :
  hours_in_day * straight_angle_per_hour = 24 :=
by
  -- Proof should be filled here
  sorry

theorem clock_hands_right_angle_48_times
  (hours_in_day : ℕ := 24)
  (right_angles_per_hour : ℕ := 2) :
  hours_in_day * right_angles_per_hour = 48 :=
by
  -- Proof should be filled here
  sorry

end NUMINAMATH_GPT_clock_hands_coincide_22_times_clock_hands_straight_angle_24_times_clock_hands_right_angle_48_times_l85_8551


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l85_8583

theorem solve_equation1 (x : ℝ) : x^2 + 2 * x = 0 ↔ x = 0 ∨ x = -2 :=
by sorry

theorem solve_equation2 (x : ℝ) : 2 * x^2 - 6 * x = 3 ↔ x = (3 + Real.sqrt 15) / 2 ∨ x = (3 - Real.sqrt 15) / 2 :=
by sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l85_8583


namespace NUMINAMATH_GPT_value_of_y_minus_x_l85_8529

theorem value_of_y_minus_x (x y : ℝ) (h1 : abs (x + 1) = 3) (h2 : abs y = 5) (h3 : -y / x > 0) :
  y - x = -7 ∨ y - x = 9 :=
sorry

end NUMINAMATH_GPT_value_of_y_minus_x_l85_8529


namespace NUMINAMATH_GPT_shapes_values_correct_l85_8589

-- Define variable types and conditions
variables (x y z w : ℕ)
variables (sum1 sum2 sum3 sum4 T : ℕ)

-- Define the conditions for the problem as given in (c)
axiom row_sum1 : x + y + z = sum1
axiom row_sum2 : y + z + w = sum2
axiom row_sum3 : z + w + x = sum3
axiom row_sum4 : w + x + y = sum4
axiom col_sum  : x + y + z + w = T

-- Define the variables with specific values as determined in the solution
def triangle := 2
def square := 0
def a_tilde := 6
def O_value := 1

-- Prove that the assigned values satisfy the conditions
theorem shapes_values_correct :
  x = triangle ∧ y = square ∧ z = a_tilde ∧ w = O_value :=
by { sorry }

end NUMINAMATH_GPT_shapes_values_correct_l85_8589


namespace NUMINAMATH_GPT_problem_statement_l85_8547

theorem problem_statement : (29.7 + 83.45) - 0.3 = 112.85 := sorry

end NUMINAMATH_GPT_problem_statement_l85_8547


namespace NUMINAMATH_GPT_abraham_initial_budget_l85_8587

-- Definitions based on conditions
def shower_gel_price := 4
def shower_gel_quantity := 4
def toothpaste_price := 3
def laundry_detergent_price := 11
def remaining_budget := 30

-- Calculations based on the conditions
def spent_on_shower_gels := shower_gel_quantity * shower_gel_price
def spent_on_toothpaste := toothpaste_price
def spent_on_laundry_detergent := laundry_detergent_price
def total_spent := spent_on_shower_gels + spent_on_toothpaste + spent_on_laundry_detergent

-- The theorem to prove
theorem abraham_initial_budget :
  (total_spent + remaining_budget) = 60 :=
by
  sorry

end NUMINAMATH_GPT_abraham_initial_budget_l85_8587


namespace NUMINAMATH_GPT_fixed_salary_new_scheme_l85_8518

theorem fixed_salary_new_scheme :
  let old_commission_rate := 0.05
  let new_commission_rate := 0.025
  let sales_target := 4000
  let total_sales := 12000
  let remuneration_difference := 600
  let old_remuneration := old_commission_rate * total_sales
  let new_commission_earnings := new_commission_rate * (total_sales - sales_target)
  let new_remuneration := old_remuneration + remuneration_difference
  ∃ F, F + new_commission_earnings = new_remuneration :=
by
  sorry

end NUMINAMATH_GPT_fixed_salary_new_scheme_l85_8518


namespace NUMINAMATH_GPT_complex_square_sum_eq_five_l85_8532

theorem complex_square_sum_eq_five (a b : ℝ) (h : (a + b * I) ^ 2 = 3 + 4 * I) : a^2 + b^2 = 5 := 
by sorry

end NUMINAMATH_GPT_complex_square_sum_eq_five_l85_8532
