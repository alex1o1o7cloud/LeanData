import Mathlib

namespace NUMINAMATH_GPT_fraction_to_decimal_l245_24541

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l245_24541


namespace NUMINAMATH_GPT_maximize_probability_remove_6_l245_24524

-- Definitions
def integers_list : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12] -- After removing 6
def initial_list : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Theorem Statement
theorem maximize_probability_remove_6 :
  ∀ (n : Int),
  n ∈ initial_list →
  n ≠ 6 →
  ∃ (a b : Int), a ∈ integers_list ∧ b ∈ integers_list ∧ a ≠ b ∧ a + b = 12 → False :=
by
  intros n hn hn6
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_maximize_probability_remove_6_l245_24524


namespace NUMINAMATH_GPT_curve_of_constant_width_l245_24519

structure Curve :=
  (is_convex : Prop)

structure Point := 
  (x : ℝ) 
  (y : ℝ)

def rotate_180 (K : Curve) (O : Point) : Curve := sorry

def sum_curves (K1 K2 : Curve) : Curve := sorry

def is_circle_with_radius (K : Curve) (r : ℝ) : Prop := sorry

def constant_width (K : Curve) (w : ℝ) : Prop := sorry

theorem curve_of_constant_width {K : Curve} {O : Point} {h : ℝ} :
  K.is_convex →
  (K' : Curve) → K' = rotate_180 K O →
  is_circle_with_radius (sum_curves K K') h →
  constant_width K h :=
by 
  sorry

end NUMINAMATH_GPT_curve_of_constant_width_l245_24519


namespace NUMINAMATH_GPT_jemma_grasshoppers_l245_24533

-- Definitions corresponding to the conditions
def grasshoppers_on_plant : ℕ := 7
def baby_grasshoppers : ℕ := 2 * 12

-- Theorem statement equivalent to the problem
theorem jemma_grasshoppers : grasshoppers_on_plant + baby_grasshoppers = 31 :=
by
  sorry

end NUMINAMATH_GPT_jemma_grasshoppers_l245_24533


namespace NUMINAMATH_GPT_next_working_day_together_l245_24595

theorem next_working_day_together : 
  let greta_days := 5
  let henry_days := 3
  let linda_days := 9
  let sam_days := 8
  ∃ n : ℕ, n = Nat.lcm (Nat.lcm (Nat.lcm greta_days henry_days) linda_days) sam_days ∧ n = 360 :=
by
  sorry

end NUMINAMATH_GPT_next_working_day_together_l245_24595


namespace NUMINAMATH_GPT_radius_base_circle_of_cone_l245_24592

theorem radius_base_circle_of_cone 
  (θ : ℝ) (R : ℝ) (arc_length : ℝ) (r : ℝ)
  (h1 : θ = 120) 
  (h2 : R = 9)
  (h3 : arc_length = (θ / 360) * 2 * Real.pi * R)
  (h4 : 2 * Real.pi * r = arc_length)
  : r = 3 := 
sorry

end NUMINAMATH_GPT_radius_base_circle_of_cone_l245_24592


namespace NUMINAMATH_GPT_days_taken_to_complete_work_l245_24521

-- Conditions
def work_rate_B : ℚ := 1 / 33
def work_rate_A : ℚ := 2 * work_rate_B
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Proof statement
theorem days_taken_to_complete_work : combined_work_rate ≠ 0 → 1 / combined_work_rate = 11 :=
by
  sorry

end NUMINAMATH_GPT_days_taken_to_complete_work_l245_24521


namespace NUMINAMATH_GPT_probability_of_same_number_l245_24503

theorem probability_of_same_number (m n : ℕ) 
  (hb : m < 250 ∧ m % 20 = 0) 
  (bb : n < 250 ∧ n % 30 = 0) : 
  (∀ (b : ℕ), b < 250 ∧ b % 60 = 0 → ∃ (m n : ℕ), ((m < 250 ∧ m % 20 = 0) ∧ (n < 250 ∧ n % 30 = 0)) → (m = n)) :=
sorry

end NUMINAMATH_GPT_probability_of_same_number_l245_24503


namespace NUMINAMATH_GPT_molar_weight_of_BaF2_l245_24530

theorem molar_weight_of_BaF2 (Ba_weight : Real) (F_weight : Real) (num_moles : ℕ) 
    (Ba_weight_val : Ba_weight = 137.33) (F_weight_val : F_weight = 18.998) 
    (num_moles_val : num_moles = 6) 
    : (137.33 + 2 * 18.998) * 6 = 1051.956 := 
by
  sorry

end NUMINAMATH_GPT_molar_weight_of_BaF2_l245_24530


namespace NUMINAMATH_GPT_simplify_expr_l245_24545

-- Define the expression
def expr := |-4^2 + 7|

-- State the theorem
theorem simplify_expr : expr = 9 :=
by sorry

end NUMINAMATH_GPT_simplify_expr_l245_24545


namespace NUMINAMATH_GPT_negated_proposition_l245_24513

theorem negated_proposition : ∀ x : ℝ, x^2 + 2 * x + 2 ≥ 0 := by
  sorry

end NUMINAMATH_GPT_negated_proposition_l245_24513


namespace NUMINAMATH_GPT_guitar_price_proof_l245_24570

def total_guitar_price (x : ℝ) : Prop :=
  0.20 * x = 240 → x = 1200

theorem guitar_price_proof (x : ℝ) (h : 0.20 * x = 240) : x = 1200 :=
by
  sorry

end NUMINAMATH_GPT_guitar_price_proof_l245_24570


namespace NUMINAMATH_GPT_flower_bed_can_fit_l245_24515

noncomputable def flower_bed_fits_in_yard : Prop :=
  let yard_side := 70
  let yard_area := yard_side ^ 2
  let building1 := (20 * 10)
  let building2 := (25 * 15)
  let building3 := (30 * 30)
  let tank_radius := 10 / 2
  let tank_area := Real.pi * tank_radius^2
  let total_occupied_area := building1 + building2 + building3 + 2*tank_area
  let available_area := yard_area - total_occupied_area
  let flower_bed_radius := 10 / 2
  let flower_bed_area := Real.pi * flower_bed_radius^2
  let buffer_area := (yard_side - 2 * flower_bed_radius)^2
  available_area >= flower_bed_area ∧ buffer_area >= flower_bed_area

theorem flower_bed_can_fit : flower_bed_fits_in_yard := 
  sorry

end NUMINAMATH_GPT_flower_bed_can_fit_l245_24515


namespace NUMINAMATH_GPT_simplify_expression_l245_24525

variable (a b : ℝ) (hab_pos : 0 < a ∧ 0 < b)
variable (h : a^3 - b^3 = a - b)

theorem simplify_expression 
  (a b : ℝ) (hab_pos : 0 < a ∧ 0 < b) (h : a^3 - b^3 = a - b) : 
  (a / b - b / a + 1 / (a * b)) = 2 * (1 / (a * b)) - 1 := 
sorry

end NUMINAMATH_GPT_simplify_expression_l245_24525


namespace NUMINAMATH_GPT_simplify_fraction_l245_24500

theorem simplify_fraction :
  ( (3 * 5 * 7 : ℚ) / (9 * 11 * 13) ) * ( (7 * 9 * 11 * 15) / (3 * 5 * 14) ) = 15 / 26 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l245_24500


namespace NUMINAMATH_GPT_sum_of_sequences_l245_24505

def sequence1 := [2, 14, 26, 38, 50]
def sequence2 := [12, 24, 36, 48, 60]
def sequence3 := [5, 15, 25, 35, 45]

theorem sum_of_sequences :
  (sequence1.sum + sequence2.sum + sequence3.sum) = 435 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_sequences_l245_24505


namespace NUMINAMATH_GPT_smallest_prime_factor_in_C_l245_24502

def smallest_prime_factor_def (n : Nat) : Nat :=
  if n % 2 = 0 then 2 else
  if n % 3 = 0 then 3 else
  sorry /- Define a function to find the smallest prime factor of a number n -/

def is_prime (p : Nat) : Prop :=
  2 ≤ p ∧ ∀ d : Nat, 2 ≤ d → d ∣ p → d = p

def in_set (x : Nat) : Prop :=
  x = 64 ∨ x = 66 ∨ x = 67 ∨ x = 68 ∨ x = 71

theorem smallest_prime_factor_in_C : ∀ x, in_set x → 
  (smallest_prime_factor_def x = 2 ∨ smallest_prime_factor_def x = 67 ∨ smallest_prime_factor_def x = 71) :=
by
  intro x hx
  cases hx with
  | inl hx  => sorry
  | inr hx  => sorry

end NUMINAMATH_GPT_smallest_prime_factor_in_C_l245_24502


namespace NUMINAMATH_GPT_profit_calculation_more_profitable_method_l245_24532

def profit_end_of_month (x : ℝ) : ℝ :=
  0.3 * x - 900

def profit_beginning_of_month (x : ℝ) : ℝ :=
  0.26 * x

theorem profit_calculation (x : ℝ) (h₁ : profit_end_of_month x = 0.3 * x - 900)
  (h₂ : profit_beginning_of_month x = 0.26 * x) :
  profit_end_of_month x = 0.3 * x - 900 ∧ profit_beginning_of_month x = 0.26 * x :=
by 
  sorry

theorem more_profitable_method (x : ℝ) (hx : x = 20000)
  (h_beg : profit_beginning_of_month x = 0.26 * x)
  (h_end : profit_end_of_month x = 0.3 * x - 900) :
  profit_beginning_of_month x > profit_end_of_month x ∧ profit_beginning_of_month x = 5200 :=
by 
  sorry

end NUMINAMATH_GPT_profit_calculation_more_profitable_method_l245_24532


namespace NUMINAMATH_GPT_factorize_problem1_factorize_problem2_l245_24563

-- Problem 1: Factorization of 4x^2 - 16
theorem factorize_problem1 (x : ℝ) : 4 * x^2 - 16 = 4 * (x - 2) * (x + 2) :=
by
  sorry

-- Problem 2: Factorization of a^2b - 4ab + 4b
theorem factorize_problem2 (a b : ℝ) : a^2 * b - 4 * a * b + 4 * b = b * (a - 2) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_problem1_factorize_problem2_l245_24563


namespace NUMINAMATH_GPT_WangLei_is_13_l245_24544

-- We need to define the conditions and question in Lean 4
def WangLei_age (x : ℕ) : Prop :=
  3 * x - 8 = 31

theorem WangLei_is_13 : ∃ x : ℕ, WangLei_age x ∧ x = 13 :=
by
  use 13
  unfold WangLei_age
  sorry

end NUMINAMATH_GPT_WangLei_is_13_l245_24544


namespace NUMINAMATH_GPT_fourth_intersection_point_l245_24537

def intersect_curve_circle : Prop :=
  let curve_eq (x y : ℝ) : Prop := x * y = 1
  let circle_intersects_points (h k s : ℝ) : Prop :=
    ∃ (x1 y1 x2 y2 x3 y3 : ℝ), 
    (x1, y1) = (3, (1 : ℝ) / 3) ∧ 
    (x2, y2) = (-4, -(1 : ℝ) / 4) ∧ 
    (x3, y3) = ((1 : ℝ) / 6, 6) ∧ 
    (x1 - h)^2 + (y1 - k)^2 = s^2 ∧
    (x2 - h)^2 + (y2 - k)^2 = s^2 ∧
    (x3 - h)^2 + (y3 - k)^2 = s^2 
  let fourth_point_of_intersection (x y : ℝ) : Prop := 
    x = -(1 : ℝ) / 2 ∧ 
    y = -2
  curve_eq 3 ((1 : ℝ) / 3) ∧
  curve_eq (-4) (-(1 : ℝ) / 4) ∧
  curve_eq ((1 : ℝ) / 6) 6 ∧
  ∃ h k s, circle_intersects_points h k s →
  ∃ (x4 y4 : ℝ), curve_eq x4 y4 ∧
  fourth_point_of_intersection x4 y4

theorem fourth_intersection_point :
  intersect_curve_circle := by
  sorry

end NUMINAMATH_GPT_fourth_intersection_point_l245_24537


namespace NUMINAMATH_GPT_problem_solution_l245_24538

theorem problem_solution (A B : ℝ) (h : ∀ x, x ≠ 3 → (A / (x - 3)) + B * (x + 2) = (-4 * x^2 + 14 * x + 38) / (x - 3)) : 
  A + B = 46 :=
sorry

end NUMINAMATH_GPT_problem_solution_l245_24538


namespace NUMINAMATH_GPT_initial_money_l245_24523

theorem initial_money (x : ℝ) (cupcake_cost total_cookie_cost total_cost money_left : ℝ) 
  (h1 : cupcake_cost = 10 * 1.5) 
  (h2 : total_cookie_cost = 5 * 3)
  (h3 : total_cost = cupcake_cost + total_cookie_cost)
  (h4 : money_left = 30)
  (h5 : 3 * x = total_cost + money_left) 
  : x = 20 := 
sorry

end NUMINAMATH_GPT_initial_money_l245_24523


namespace NUMINAMATH_GPT_bus_speed_l245_24543

theorem bus_speed (distance time : ℝ) (h_distance : distance = 201) (h_time : time = 3) : 
  distance / time = 67 :=
by
  sorry

end NUMINAMATH_GPT_bus_speed_l245_24543


namespace NUMINAMATH_GPT_expected_profit_calculation_l245_24578

theorem expected_profit_calculation:
  let odd1 := 1.28
  let odd2 := 5.23
  let odd3 := 3.25
  let odd4 := 2.05
  let initial_bet := 5.00
  let total_payout := initial_bet * (odd1 * odd2 * odd3 * odd4)
  let expected_profit := total_payout - initial_bet
  expected_profit = 212.822 := by
  sorry

end NUMINAMATH_GPT_expected_profit_calculation_l245_24578


namespace NUMINAMATH_GPT_hilton_final_marbles_l245_24569

def initial_marbles : ℕ := 26
def marbles_found : ℕ := 6
def marbles_lost : ℕ := 10
def marbles_from_lori := 2 * marbles_lost

def final_marbles := initial_marbles + marbles_found - marbles_lost + marbles_from_lori

theorem hilton_final_marbles : final_marbles = 42 := sorry

end NUMINAMATH_GPT_hilton_final_marbles_l245_24569


namespace NUMINAMATH_GPT_find_k_l245_24528

/- Definitions for vectors -/
def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-3, 2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

/- Prove that if ka + b is perpendicular to a, then k = -1/5 -/
theorem find_k (k : ℝ) : 
  dot_product (k • (1, 2) + (-3, 2)) (1, 2) = 0 → 
  k = -1 / 5 := 
  sorry

end NUMINAMATH_GPT_find_k_l245_24528


namespace NUMINAMATH_GPT_exists_positive_integers_seq_l245_24536

def sum_of_digits (m : ℕ) : ℕ :=
  m.digits 10 |>.sum

def prod_of_digits (m : ℕ) : ℕ :=
  m.digits 10 |>.prod

theorem exists_positive_integers_seq (n : ℕ) (hn : 0 < n) :
  ∃ (a : Fin n.succ → ℕ),
    (∀ i : Fin n, sum_of_digits (a i) < sum_of_digits (a i.succ)) ∧
    (∀ i : Fin n, sum_of_digits (a i) = prod_of_digits (a i.succ)) ∧
    (∀ i : Fin n, 0 < (a i)) :=
by
  sorry

end NUMINAMATH_GPT_exists_positive_integers_seq_l245_24536


namespace NUMINAMATH_GPT_units_digit_k_squared_plus_2_k_l245_24561

noncomputable def k : ℕ := 2009^2 + 2^2009 - 3

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_k_squared_plus_2_k : units_digit (k^2 + 2^k) = 1 := by
  sorry

end NUMINAMATH_GPT_units_digit_k_squared_plus_2_k_l245_24561


namespace NUMINAMATH_GPT_power_equivalence_l245_24531

theorem power_equivalence (K : ℕ) : 32^2 * 4^5 = 2^K ↔ K = 20 :=
by sorry

end NUMINAMATH_GPT_power_equivalence_l245_24531


namespace NUMINAMATH_GPT_niko_total_profit_l245_24589

def pairs_of_socks : Nat := 9
def cost_per_pair : ℝ := 2
def profit_percentage_first_four : ℝ := 0.25
def profit_per_pair_remaining_five : ℝ := 0.2

theorem niko_total_profit :
  let total_profit_first_four := 4 * (cost_per_pair * profit_percentage_first_four)
  let total_profit_remaining_five := 5 * profit_per_pair_remaining_five
  let total_profit := total_profit_first_four + total_profit_remaining_five
  total_profit = 3 := by
  sorry

end NUMINAMATH_GPT_niko_total_profit_l245_24589


namespace NUMINAMATH_GPT_triangle_angle_contradiction_l245_24554

theorem triangle_angle_contradiction (α β γ : ℝ) (h1 : α + β + γ = 180) (h2 : α > 60) (h3 : β > 60) (h4 : γ > 60) : false :=
sorry

end NUMINAMATH_GPT_triangle_angle_contradiction_l245_24554


namespace NUMINAMATH_GPT_smallest_number_of_marbles_l245_24549

theorem smallest_number_of_marbles :
  ∃ (r w b g y : ℕ), 
  (r + w + b + g + y = 13) ∧ 
  (r ≥ 5) ∧
  (r - 4 = 5 * w) ∧
  ((r - 3) * (r - 4) = 20 * w * b) ∧
  sorry := sorry

end NUMINAMATH_GPT_smallest_number_of_marbles_l245_24549


namespace NUMINAMATH_GPT_smallest_set_of_circular_handshakes_l245_24590

def circular_handshake_smallest_set (n : ℕ) : ℕ :=
  if h : n % 2 = 0 then n / 2 else (n / 2) + 1

theorem smallest_set_of_circular_handshakes :
  circular_handshake_smallest_set 36 = 18 :=
by
  sorry

end NUMINAMATH_GPT_smallest_set_of_circular_handshakes_l245_24590


namespace NUMINAMATH_GPT_negation_of_p_l245_24504

variable (x y : ℝ)

def proposition_p := ∀ x y : ℝ, x^2 + y^2 - 1 > 0 

theorem negation_of_p : (¬ proposition_p) ↔ (∃ x y : ℝ, x^2 + y^2 - 1 ≤ 0) := by
  sorry

end NUMINAMATH_GPT_negation_of_p_l245_24504


namespace NUMINAMATH_GPT_Nikolai_faster_than_Gennady_l245_24582

theorem Nikolai_faster_than_Gennady
  (gennady_jump_time : ℕ)
  (nikolai_jump_time : ℕ)
  (jump_distance_gennady: ℕ)
  (jump_distance_nikolai: ℕ)
  (jump_count_gennady : ℕ)
  (jump_count_nikolai : ℕ)
  (total_distance : ℕ)
  (h1 : gennady_jump_time = nikolai_jump_time)
  (h2 : jump_distance_gennady = 6)
  (h3 : jump_distance_nikolai = 4)
  (h4 : jump_count_gennady = 2)
  (h5 : jump_count_nikolai = 3)
  (h6 : total_distance = 2000) :
  (total_distance % jump_distance_nikolai = 0) ∧ (total_distance % jump_distance_gennady ≠ 0) → 
  nikolai_jump_time < gennady_jump_time := 
sorry

end NUMINAMATH_GPT_Nikolai_faster_than_Gennady_l245_24582


namespace NUMINAMATH_GPT_complex_pure_imaginary_l245_24517

theorem complex_pure_imaginary (a : ℂ) : (∃ (b : ℂ), (2 - I) * (a + 2 * I) = b * I) → a = -1 :=
by
  sorry

end NUMINAMATH_GPT_complex_pure_imaginary_l245_24517


namespace NUMINAMATH_GPT_box_neg2_0_3_eq_10_div_9_l245_24562

def box (a b c : ℤ) : ℚ :=
  a^b - b^c + c^a

theorem box_neg2_0_3_eq_10_div_9 : box (-2) 0 3 = 10 / 9 :=
by
  sorry

end NUMINAMATH_GPT_box_neg2_0_3_eq_10_div_9_l245_24562


namespace NUMINAMATH_GPT_abscissa_range_of_point_P_l245_24508

-- Definitions based on the conditions from the problem
def y_function (x : ℝ) : ℝ := 4 - 3 * x
def point_P (x y : ℝ) : Prop := y = y_function x
def ordinate_greater_than_negative_five (y : ℝ) : Prop := y > -5

-- Theorem statement combining the above definitions
theorem abscissa_range_of_point_P (x y : ℝ) :
  point_P x y →
  ordinate_greater_than_negative_five y →
  x < 3 :=
sorry

end NUMINAMATH_GPT_abscissa_range_of_point_P_l245_24508


namespace NUMINAMATH_GPT_part_a_part_b_l245_24560

theorem part_a (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z ≥ 3) :
  ¬ (1/x + 1/y + 1/z ≤ 3) :=
sorry

theorem part_b (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z ≤ 3) :
  1/x + 1/y + 1/z ≥ 3 :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l245_24560


namespace NUMINAMATH_GPT_multiplier_condition_l245_24529

theorem multiplier_condition (a b : ℚ) (h : a * b ≤ b) : (b ≥ 0 ∧ a ≤ 1) ∨ (b ≤ 0 ∧ a ≥ 1) :=
by 
  sorry

end NUMINAMATH_GPT_multiplier_condition_l245_24529


namespace NUMINAMATH_GPT_trig_identity_1_trig_identity_2_l245_24539

noncomputable def point := ℚ × ℚ

namespace TrigProblem

open Real

def point_on_terminal_side (α : ℝ) (p : point) : Prop :=
  let (x, y) := p
  ∃ r : ℝ, r = sqrt (x^2 + y^2) ∧ x/r = cos α ∧ y/r = sin α

theorem trig_identity_1 {α : ℝ} (h : point_on_terminal_side α (-4, 3)) :
  (sin (π / 2 + α) - cos (π + α)) / (sin (π / 2 - α) - sin (π - α)) = 8 / 7 :=
sorry

theorem trig_identity_2 {α : ℝ} (h : point_on_terminal_side α (-4, 3)) :
  sin α * cos α = -12 / 25 :=
sorry

end TrigProblem

end NUMINAMATH_GPT_trig_identity_1_trig_identity_2_l245_24539


namespace NUMINAMATH_GPT_ratio_of_tagged_fish_is_1_over_25_l245_24552

-- Define the conditions
def T70 : ℕ := 70  -- Number of tagged fish first caught and tagged
def T50 : ℕ := 50  -- Total number of fish caught in the second sample
def t2 : ℕ := 2    -- Number of tagged fish in the second sample

-- State the theorem/question
theorem ratio_of_tagged_fish_is_1_over_25 : (t2 / T50) = 1 / 25 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_tagged_fish_is_1_over_25_l245_24552


namespace NUMINAMATH_GPT_number_of_girls_l245_24591

theorem number_of_girls
  (total_boys : ℕ)
  (total_boys_eq : total_boys = 10)
  (fraction_girls_reading : ℚ)
  (fraction_girls_reading_eq : fraction_girls_reading = 5/6)
  (fraction_boys_reading : ℚ)
  (fraction_boys_reading_eq : fraction_boys_reading = 4/5)
  (total_not_reading : ℕ)
  (total_not_reading_eq : total_not_reading = 4)
  (G : ℝ)
  (remaining_girls_reading : (1 - fraction_girls_reading) * G = 2)
  (remaining_boys_not_reading : (1 - fraction_boys_reading) * total_boys = 2)
  (remaining_total_not_reading : 2 + 2 = total_not_reading)
  : G = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_girls_l245_24591


namespace NUMINAMATH_GPT_y1_lt_y2_l245_24551

theorem y1_lt_y2 (x1 x2 : ℝ) (h1 : x1 < 0) (h2 : 0 < x2) :
  (6 / x1) < (6 / x2) :=
by
  sorry

end NUMINAMATH_GPT_y1_lt_y2_l245_24551


namespace NUMINAMATH_GPT_possible_values_expression_l245_24522

theorem possible_values_expression 
  (a b : ℝ) 
  (h₁ : a^2 = 16) 
  (h₂ : |b| = 3) 
  (h₃ : ab < 0) : 
  (a - b)^2 + a * b^2 = 85 ∨ (a - b)^2 + a * b^2 = 13 := 
by 
  sorry

end NUMINAMATH_GPT_possible_values_expression_l245_24522


namespace NUMINAMATH_GPT_other_root_of_quadratic_l245_24548

theorem other_root_of_quadratic (a : ℝ) :
  (∀ x, (x^2 + 2*x - a) = 0 → x = -3) → (∃ z, z = 1 ∧ (z^2 + 2*z - a) = 0) :=
by
  sorry

end NUMINAMATH_GPT_other_root_of_quadratic_l245_24548


namespace NUMINAMATH_GPT_proof_problem_l245_24510

theorem proof_problem (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a^3 + b^3 = 2 * a * b) : a^2 + b^2 ≤ 1 + a * b := 
sorry

end NUMINAMATH_GPT_proof_problem_l245_24510


namespace NUMINAMATH_GPT_actual_distance_l245_24512

theorem actual_distance (d_map : ℝ) (scale_inches : ℝ) (scale_miles : ℝ) (H1 : d_map = 20)
    (H2 : scale_inches = 0.5) (H3 : scale_miles = 10) : 
    d_map * (scale_miles / scale_inches) = 400 := 
by
  sorry

end NUMINAMATH_GPT_actual_distance_l245_24512


namespace NUMINAMATH_GPT_trigonometric_identity_proof_l245_24520

noncomputable def a : ℝ := -35 / 6 * Real.pi

theorem trigonometric_identity_proof :
  (2 * Real.sin (Real.pi + a) * Real.cos (Real.pi - a) - Real.cos (Real.pi + a)) / 
  (1 + Real.sin a ^ 2 + Real.sin (Real.pi - a) - Real.cos (Real.pi + a) ^ 2) = Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_proof_l245_24520


namespace NUMINAMATH_GPT_ab_value_l245_24534

theorem ab_value (a b : ℝ) (h1 : a^2 + b^2 = 2) (h2 : a + b = 3) : a * b = 7/2 :=
by
  sorry

end NUMINAMATH_GPT_ab_value_l245_24534


namespace NUMINAMATH_GPT_intersection_of_sets_l245_24559

open Set

theorem intersection_of_sets :
  let A := {x : ℤ | |x| < 3}
  let B := {x : ℤ | |x| > 1}
  A ∩ B = ({-2, 2} : Set ℤ) := by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l245_24559


namespace NUMINAMATH_GPT_base_area_cone_l245_24514

theorem base_area_cone (V h : ℝ) (s_cylinder s_cone : ℝ) 
  (cylinder_volume : V = s_cylinder * h) 
  (cone_volume : V = (1 / 3) * s_cone * h) 
  (s_cylinder_val : s_cylinder = 15) : s_cone = 45 := 
by 
  sorry

end NUMINAMATH_GPT_base_area_cone_l245_24514


namespace NUMINAMATH_GPT_total_votes_l245_24571

noncomputable def total_votes_proof : Prop :=
  ∃ T A : ℝ, 
    A = 0.40 * T ∧ 
    T = A + (A + 70) ∧ 
    T = 350

theorem total_votes : total_votes_proof :=
sorry

end NUMINAMATH_GPT_total_votes_l245_24571


namespace NUMINAMATH_GPT_a_cubed_plus_b_cubed_gt_a_sq_b_plus_ab_sq_l245_24557

theorem a_cubed_plus_b_cubed_gt_a_sq_b_plus_ab_sq {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≠ b) : a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end NUMINAMATH_GPT_a_cubed_plus_b_cubed_gt_a_sq_b_plus_ab_sq_l245_24557


namespace NUMINAMATH_GPT_value_of_a_squared_plus_b_squared_plus_2ab_l245_24553

theorem value_of_a_squared_plus_b_squared_plus_2ab (a b : ℝ) (h : a + b = -1) :
  a^2 + b^2 + 2 * a * b = 1 :=
by sorry

end NUMINAMATH_GPT_value_of_a_squared_plus_b_squared_plus_2ab_l245_24553


namespace NUMINAMATH_GPT_division_by_n_minus_1_squared_l245_24587

theorem division_by_n_minus_1_squared (n : ℕ) (h : n > 2) : (n ^ (n - 1) - 1) % ((n - 1) ^ 2) = 0 :=
sorry

end NUMINAMATH_GPT_division_by_n_minus_1_squared_l245_24587


namespace NUMINAMATH_GPT_min_value_reciprocal_sum_l245_24511

theorem min_value_reciprocal_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 3) :
  3 ≤ (1 / a) + (1 / b) + (1 / c) :=
by sorry

end NUMINAMATH_GPT_min_value_reciprocal_sum_l245_24511


namespace NUMINAMATH_GPT_can_lids_per_box_l245_24542

/-- Aaron initially has 14 can lids, and after adding can lids from 3 boxes,
he has a total of 53 can lids. How many can lids are in each box? -/
theorem can_lids_per_box (initial : ℕ) (total : ℕ) (boxes : ℕ) (h₀ : initial = 14) (h₁ : total = 53) (h₂ : boxes = 3) :
  (total - initial) / boxes = 13 :=
by
  sorry

end NUMINAMATH_GPT_can_lids_per_box_l245_24542


namespace NUMINAMATH_GPT_darry_steps_l245_24577

theorem darry_steps (f_steps : ℕ) (f_times : ℕ) (s_steps : ℕ) (s_times : ℕ) (no_other_steps : ℕ)
  (hf : f_steps = 11)
  (hf_times : f_times = 10)
  (hs : s_steps = 6)
  (hs_times : s_times = 7)
  (h_no_other : no_other_steps = 0) :
  (f_steps * f_times + s_steps * s_times + no_other_steps = 152) :=
by
  sorry

end NUMINAMATH_GPT_darry_steps_l245_24577


namespace NUMINAMATH_GPT_area_of_triangle_formed_by_line_l_and_axes_equation_of_line_l_l245_24579

-- Defining the lines
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def line2 (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Intersection point P of line1 and line2
def P : ℝ × ℝ := (-2, 2)

-- Perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Line l, passing through P and perpendicular to perpendicular_line
def line_l (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Intercepts of line_l with axes
def x_intercept : ℝ := -1
def y_intercept : ℝ := -2

-- Verifying area of the triangle formed by the intercepts
def area_of_triangle (base height : ℝ) : ℝ := 0.5 * base * height

#check line1
#check line2
#check P
#check perpendicular_line
#check line_l
#check x_intercept
#check y_intercept
#check area_of_triangle

theorem area_of_triangle_formed_by_line_l_and_axes :
  ∀ (x : ℝ) (y : ℝ),
    line_l x 0 → line_l 0 y →
    area_of_triangle (abs x) (abs y) = 1 :=
by
  intros x y hx hy
  sorry

theorem equation_of_line_l :
  ∀ (x y : ℝ),
    (line1 x y ∧ line2 x y) →
    (perpendicular_line x y) →
    line_l x y :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_area_of_triangle_formed_by_line_l_and_axes_equation_of_line_l_l245_24579


namespace NUMINAMATH_GPT_abs_eq_condition_l245_24516

theorem abs_eq_condition (x : ℝ) (h : |x - 1| + |x - 5| = 4) : 1 ≤ x ∧ x ≤ 5 :=
by 
  sorry

end NUMINAMATH_GPT_abs_eq_condition_l245_24516


namespace NUMINAMATH_GPT_angle4_is_35_l245_24526

theorem angle4_is_35
  (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ)
  (h1 : angle1 + angle2 = 180)
  (h2 : angle3 = angle4)
  (ha : angle1 = 50)
  (h_opposite : angle5 = 60)
  (triangle_sum : angle1 + angle5 + angle6 = 180)
  (supplementary_angle : angle2 + angle6 = 180) :
  angle4 = 35 :=
by
  sorry

end NUMINAMATH_GPT_angle4_is_35_l245_24526


namespace NUMINAMATH_GPT_burmese_python_eats_alligators_l245_24566

theorem burmese_python_eats_alligators (snake_length : ℝ) (alligator_length : ℝ) (alligator_per_week : ℝ) (total_alligators : ℝ) :
  snake_length = 1.4 → alligator_length = 0.5 → alligator_per_week = 1 → total_alligators = 88 →
  (total_alligators / alligator_per_week) * 7 = 616 := by
  intros
  sorry

end NUMINAMATH_GPT_burmese_python_eats_alligators_l245_24566


namespace NUMINAMATH_GPT_midpoint_line_l245_24550

theorem midpoint_line (a : ℝ) (P Q M : ℝ × ℝ) (hP : P = (a, 5 * a + 3)) (hQ : Q = (3, -2))
  (hM : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) : M.2 = 5 * M.1 - 7 := 
sorry

end NUMINAMATH_GPT_midpoint_line_l245_24550


namespace NUMINAMATH_GPT_inequality_solutions_l245_24583

theorem inequality_solutions (a : ℚ) :
  (∀ x : ℕ, 0 < x ∧ x ≤ 3 → 3 * (x - 1) < 2 * (x + a) - 5) →
  (∃ x : ℕ, 0 < x ∧ x = 4 → ¬ (3 * (x - 1) < 2 * (x + a) - 5)) →
  (5 / 2 < a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_GPT_inequality_solutions_l245_24583


namespace NUMINAMATH_GPT_function_decreasing_range_l245_24585

theorem function_decreasing_range (a : ℝ) :
  (∀ x : ℝ, (if x ≥ 0 then -x + 3 * a else x^2 - a * x + 1) ≤ (if x ≥ 0 then -x + 3 * a else x^2 - a * x + 1)) ↔ (0 ≤ a ∧ a ≤ 1 / 3) :=
sorry

end NUMINAMATH_GPT_function_decreasing_range_l245_24585


namespace NUMINAMATH_GPT_candy_eaten_l245_24518

theorem candy_eaten (x : ℕ) (initial_candy eaten_more remaining : ℕ) (h₁ : initial_candy = 22) (h₂ : eaten_more = 5) (h₃ : remaining = 8) (h₄ : initial_candy - x - eaten_more = remaining) : x = 9 :=
by
  -- proof
  sorry

end NUMINAMATH_GPT_candy_eaten_l245_24518


namespace NUMINAMATH_GPT_evaluate_expression_l245_24584

-- Define the expression as given in the problem
def expr1 : ℤ := |9 - 8 * (3 - 12)|
def expr2 : ℤ := |5 - 11|

-- Define the mathematical equivalence
theorem evaluate_expression : (expr1 - expr2) = 75 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l245_24584


namespace NUMINAMATH_GPT_least_number_to_subtract_l245_24567

theorem least_number_to_subtract (x : ℕ) (h : 509 - x = 45 * n) : ∃ x, (509 - x) % 9 = 0 ∧ (509 - x) % 15 = 0 ∧ x = 14 := by
  sorry

end NUMINAMATH_GPT_least_number_to_subtract_l245_24567


namespace NUMINAMATH_GPT_speed_of_stream_l245_24596

theorem speed_of_stream (downstream_speed upstream_speed : ℕ) (h1 : downstream_speed = 12) (h2 : upstream_speed = 8) : 
  (downstream_speed - upstream_speed) / 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_stream_l245_24596


namespace NUMINAMATH_GPT_tommy_total_balloons_l245_24593

-- Define the conditions from part (a)
def original_balloons : Nat := 26
def additional_balloons : Nat := 34

-- Define the proof problem from part (c)
theorem tommy_total_balloons : original_balloons + additional_balloons = 60 := by
  -- Skip the actual proof
  sorry

end NUMINAMATH_GPT_tommy_total_balloons_l245_24593


namespace NUMINAMATH_GPT_inequality_solution_l245_24555

theorem inequality_solution (x : ℝ) : 
  (7 - 2 * (x + 1) ≥ 1 - 6 * x) ∧ ((1 + 2 * x) / 3 > x - 1) ↔ (-1 ≤ x ∧ x < 4) := 
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l245_24555


namespace NUMINAMATH_GPT_largest_root_range_l245_24574

theorem largest_root_range (b_0 b_1 b_2 b_3 : ℝ)
  (hb_0 : |b_0| ≤ 3) (hb_1 : |b_1| ≤ 3) (hb_2 : |b_2| ≤ 3) (hb_3 : |b_3| ≤ 3) :
  ∃ s : ℝ, (∃ x : ℝ, x ^ 4 + b_3 * x ^ 3 + b_2 * x ^ 2 + b_1 * x + b_0 = 0 ∧ x > 0 ∧ s = x) ∧ 3 < s ∧ s < 4 := 
sorry

end NUMINAMATH_GPT_largest_root_range_l245_24574


namespace NUMINAMATH_GPT_find_m_from_expansion_l245_24599

theorem find_m_from_expansion (m n : ℤ) (h : (x : ℝ) → (x + 3) * (x + n) = x^2 + m * x - 21) : m = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_m_from_expansion_l245_24599


namespace NUMINAMATH_GPT_angle_half_second_quadrant_l245_24594

theorem angle_half_second_quadrant (α : ℝ) (k : ℤ) :
  (π / 2 + 2 * k * π < α ∧ α < π + 2 * k * π) → 
  (∃ m : ℤ, (π / 4 + m * π < α / 2 ∧ α / 2 < π / 2 + m * π)) ∨ 
  (∃ n : ℤ, (5 * π / 4 + n * π < α / 2 ∧ α / 2 < 3 * π / 2 + n * π)) :=
by
  sorry

end NUMINAMATH_GPT_angle_half_second_quadrant_l245_24594


namespace NUMINAMATH_GPT_smallest_x_satisfies_equation_l245_24580

theorem smallest_x_satisfies_equation : 
  ∀ x : ℚ, 7 * (10 * x^2 + 10 * x + 11) = x * (10 * x - 45) → x = -7 / 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_x_satisfies_equation_l245_24580


namespace NUMINAMATH_GPT_problem1_l245_24540

theorem problem1 (x y : ℝ) (h1 : x + y = 4) (h2 : 2 * x - y = 5) : 
  x = 3 ∧ y = 1 := sorry

end NUMINAMATH_GPT_problem1_l245_24540


namespace NUMINAMATH_GPT_bus_driver_total_earnings_l245_24573

noncomputable def regular_rate : ℝ := 20
noncomputable def regular_hours : ℝ := 40
noncomputable def total_hours : ℝ := 45.714285714285715
noncomputable def overtime_rate : ℝ := regular_rate + 0.75 * regular_rate
noncomputable def overtime_hours : ℝ := total_hours - regular_hours
noncomputable def regular_pay : ℝ := regular_rate * regular_hours
noncomputable def overtime_pay : ℝ := overtime_rate * overtime_hours
noncomputable def total_compensation : ℝ := regular_pay + overtime_pay

theorem bus_driver_total_earnings :
  total_compensation = 1000 :=
by
  sorry

end NUMINAMATH_GPT_bus_driver_total_earnings_l245_24573


namespace NUMINAMATH_GPT_complex_number_purely_imaginary_l245_24586

variable {m : ℝ}

theorem complex_number_purely_imaginary (h1 : 2 * m^2 + m - 1 = 0) (h2 : -m^2 - 3 * m - 2 ≠ 0) : m = 1/2 := by
  sorry

end NUMINAMATH_GPT_complex_number_purely_imaginary_l245_24586


namespace NUMINAMATH_GPT_sum_of_squares_and_cube_unique_l245_24565

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m * m = n

theorem sum_of_squares_and_cube_unique : 
  ∃! (a b c : ℕ), a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ is_perfect_square a ∧ is_perfect_square b ∧ is_perfect_cube c ∧ a + b + c = 100 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_and_cube_unique_l245_24565


namespace NUMINAMATH_GPT_no_all_perfect_squares_l245_24597

theorem no_all_perfect_squares (x : ℤ) 
  (h1 : ∃ a : ℤ, 2 * x - 1 = a^2) 
  (h2 : ∃ b : ℤ, 5 * x - 1 = b^2) 
  (h3 : ∃ c : ℤ, 13 * x - 1 = c^2) : 
  False :=
sorry

end NUMINAMATH_GPT_no_all_perfect_squares_l245_24597


namespace NUMINAMATH_GPT_original_number_l245_24572

theorem original_number (x : ℝ) (h : 1.2 * x = 1080) : x = 900 := by
  sorry

end NUMINAMATH_GPT_original_number_l245_24572


namespace NUMINAMATH_GPT_length_of_first_platform_l245_24546

noncomputable def speed (distance time : ℕ) :=
  distance / time

theorem length_of_first_platform 
  (L : ℕ) (train_length : ℕ) (time1 time2 : ℕ) (platform2_length : ℕ) (speed : ℕ) 
  (H1 : L + train_length = speed * time1) 
  (H2 : platform2_length + train_length = speed * time2) 
  (train_length_eq : train_length = 30) 
  (time1_eq : time1 = 12) 
  (time2_eq : time2 = 15) 
  (platform2_length_eq : platform2_length = 120) 
  (speed_eq : speed = 10) : L = 90 :=
by
  sorry

end NUMINAMATH_GPT_length_of_first_platform_l245_24546


namespace NUMINAMATH_GPT_find_difference_of_segments_l245_24581

theorem find_difference_of_segments 
  (a b c d x y : ℝ)
  (h1 : a + b = 70)
  (h2 : b + c = 90)
  (h3 : c + d = 130)
  (h4 : a + d = 110)
  (hx_y_sum : x + y = 130)
  (hx_c : x = c)
  (hy_d : y = d) : 
  |x - y| = 13 :=
sorry

end NUMINAMATH_GPT_find_difference_of_segments_l245_24581


namespace NUMINAMATH_GPT_restore_axes_with_parabola_l245_24556

-- Define the given parabola y = x^2
def parabola (x : ℝ) : ℝ := x^2

-- Problem: Prove that you can restore the coordinate axes using the given parabola and tools.
theorem restore_axes_with_parabola : 
  ∃ O X Y : ℝ × ℝ, 
  (∀ x, parabola x = (x, x^2).snd) ∧ 
  (X.fst = 0 ∧ Y.snd = 0) ∧
  (O = (0,0)) :=
sorry

end NUMINAMATH_GPT_restore_axes_with_parabola_l245_24556


namespace NUMINAMATH_GPT_part1_part2_l245_24547

open BigOperators

noncomputable def seq (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n ≠ 0 → a n > 0) ∧
  (a 1 = 2) ∧
  (∀ n : ℕ, n ≠ 0 → (n + 1) * (a (n + 1)) ^ 2 = n * (a n) ^ 2 + a n)

theorem part1 (a : ℕ → ℝ) (h : seq a)
  (n : ℕ) (hn : n ≠ 0) 
  : 1 < a (n+1) ∧ a (n+1) < a n :=
sorry

theorem part2 (a : ℕ → ℝ) (h : seq a)
  : ∑ k in Finset.range 2022 \ {0}, (a (k+1))^2 / (k+1)^2 < 2 :=
sorry

end NUMINAMATH_GPT_part1_part2_l245_24547


namespace NUMINAMATH_GPT_sodium_thiosulfate_properties_l245_24507

def thiosulfate_structure : Type := sorry
-- Define the structure of S2O3^{2-} with S-S bond
def has_s_s_bond (ion : thiosulfate_structure) : Prop := sorry
-- Define the formation reaction
def formed_by_sulfite_reaction (ion : thiosulfate_structure) : Prop := sorry

theorem sodium_thiosulfate_properties :
  ∃ (ion : thiosulfate_structure),
    has_s_s_bond ion ∧ formed_by_sulfite_reaction ion :=
by
  sorry

end NUMINAMATH_GPT_sodium_thiosulfate_properties_l245_24507


namespace NUMINAMATH_GPT_bruce_total_amount_paid_l245_24509

-- Definitions for quantities and rates
def quantity_of_grapes : Nat := 8
def rate_per_kg_grapes : Nat := 70
def quantity_of_mangoes : Nat := 11
def rate_per_kg_mangoes : Nat := 55

-- Calculate individual costs
def cost_of_grapes := quantity_of_grapes * rate_per_kg_grapes
def cost_of_mangoes := quantity_of_mangoes * rate_per_kg_mangoes

-- Calculate total amount paid
def total_amount_paid := cost_of_grapes + cost_of_mangoes

-- Statement to prove
theorem bruce_total_amount_paid : total_amount_paid = 1165 := by
  -- Proof is intentionally left as a placeholder
  sorry

end NUMINAMATH_GPT_bruce_total_amount_paid_l245_24509


namespace NUMINAMATH_GPT_pastries_left_to_take_home_l245_24576

def initial_cupcakes : ℕ := 7
def initial_cookies : ℕ := 5
def pastries_sold : ℕ := 4

theorem pastries_left_to_take_home :
  initial_cupcakes + initial_cookies - pastries_sold = 8 := by
  sorry

end NUMINAMATH_GPT_pastries_left_to_take_home_l245_24576


namespace NUMINAMATH_GPT_cans_increment_l245_24506

/--
If there are 9 rows of cans in a triangular display, where each successive row increases 
by a certain number of cans \( x \) compared to the row above it, with the seventh row having 
19 cans, and the total number of cans being fewer than 120, then 
each row has 4 more cans than the row above it.
-/
theorem cans_increment (x : ℕ) : 
  9 * 19 - 16 * x < 120 → x > 51 / 16 → x = 4 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_cans_increment_l245_24506


namespace NUMINAMATH_GPT_total_revenue_4706_l245_24558

noncomputable def totalTicketRevenue (seats : ℕ) (show2pm : ℕ × ℕ) (show5pm : ℕ × ℕ) (show8pm : ℕ × ℕ) : ℕ :=
  let revenue2pm := show2pm.1 * 4 + (seats - show2pm.1) * 6
  let revenue5pm := show5pm.1 * 5 + (seats - show5pm.1) * 8
  let revenue8pm := show8pm.1 * 7 + (show8pm.2 - show8pm.1) * 10
  revenue2pm + revenue5pm + revenue8pm

theorem total_revenue_4706 :
  totalTicketRevenue 250 (135, 250) (160, 250) (98, 225) = 4706 :=
by
  unfold totalTicketRevenue
  -- We provide the proof steps here in a real proof scenario.
  -- We are focusing on the statement formulation only.
  sorry

end NUMINAMATH_GPT_total_revenue_4706_l245_24558


namespace NUMINAMATH_GPT_minimize_payment_l245_24575

theorem minimize_payment :
  ∀ (bd_A td_A bd_B td_B bd_C td_C : ℕ),
    bd_A = 42 → td_A = 36 →
    bd_B = 48 → td_B = 41 →
    bd_C = 54 → td_C = 47 →
    ∃ (S : ℕ), S = 36 ∧ 
      (S = bd_A - (bd_A - td_A)) ∧
      (S < bd_B - (bd_B - td_B)) ∧
      (S < bd_C - (bd_C - td_C)) := 
by {
  sorry
}

end NUMINAMATH_GPT_minimize_payment_l245_24575


namespace NUMINAMATH_GPT_xiao_peach_days_l245_24564

theorem xiao_peach_days :
  ∀ (xiao_ming_apples xiao_ming_pears xiao_ming_peaches : ℕ)
    (xiao_hong_apples xiao_hong_pears xiao_hong_peaches : ℕ)
    (both_eat_apples both_eat_pears : ℕ)
    (one_eats_apple_other_eats_pear : ℕ),
    xiao_ming_apples = 4 →
    xiao_ming_pears = 6 →
    xiao_ming_peaches = 8 →
    xiao_hong_apples = 5 →
    xiao_hong_pears = 7 →
    xiao_hong_peaches = 6 →
    both_eat_apples = 3 →
    both_eat_pears = 2 →
    one_eats_apple_other_eats_pear = 3 →
    ∃ (both_eat_peaches_days : ℕ),
      both_eat_peaches_days = 4 := 
sorry

end NUMINAMATH_GPT_xiao_peach_days_l245_24564


namespace NUMINAMATH_GPT_part1_part2_l245_24501

section
variable (x a : ℝ)
def p (x a : ℝ) : Prop := (x - 3 * a) * (x - a) < 0
def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

theorem part1 (h : a = 1) (hq : q x) (hp : p x a) : 2 < x ∧ x < 3 := by
  sorry

theorem part2 (h : ∀ x, q x → p x a) : 1 ≤ a ∧ a ≤ 2 := by
  sorry
end

end NUMINAMATH_GPT_part1_part2_l245_24501


namespace NUMINAMATH_GPT_abc_geq_inequality_l245_24527

open Real

theorem abc_geq_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) :=
by
  sorry

end NUMINAMATH_GPT_abc_geq_inequality_l245_24527


namespace NUMINAMATH_GPT_articles_for_z_men_l245_24598

-- The necessary conditions and given values
def articles_produced (men hours days : ℕ) := men * hours * days

theorem articles_for_z_men (x z : ℕ) (H : articles_produced x x x = x^2) :
  articles_produced z z z = z^3 / x := by
  sorry

end NUMINAMATH_GPT_articles_for_z_men_l245_24598


namespace NUMINAMATH_GPT_scatter_plot_convention_l245_24588

def explanatory_variable := "x-axis"
def predictor_variable := "y-axis"

theorem scatter_plot_convention :
  explanatory_variable = "x-axis" ∧ predictor_variable = "y-axis" :=
by sorry

end NUMINAMATH_GPT_scatter_plot_convention_l245_24588


namespace NUMINAMATH_GPT_proof_statements_l245_24535

theorem proof_statements (m : ℝ) (x y : ℝ)
  (h1 : 2 * x + y = 4 - m)
  (h2 : x - 2 * y = 3 * m) :
  (m = 1 → (x = 9 / 5 ∧ y = -3 / 5)) ∧
  (3 * x - y = 4 + 2 * m) ∧
  ¬(∃ (m' : ℝ), (8 + m') / 5 < 0 ∧ (4 - 7 * m') / 5 < 0) :=
sorry

end NUMINAMATH_GPT_proof_statements_l245_24535


namespace NUMINAMATH_GPT_cauliflower_sales_l245_24568

namespace WeeklyMarket

def broccoliPrice := 3
def totalEarnings := 520
def broccolisSold := 19

def carrotPrice := 2
def spinachPrice := 4
def spinachWeight := 8 -- This is derived from solving $4S = 2S + $16 

def broccoliEarnings := broccolisSold * broccoliPrice
def carrotEarnings := spinachWeight * carrotPrice -- This is twice copied

def spinachEarnings : ℕ := spinachWeight * spinachPrice
def tomatoEarnings := broccoliEarnings + spinachEarnings

def otherEarnings : ℕ := broccoliEarnings + carrotEarnings + spinachEarnings + tomatoEarnings

def cauliflowerEarnings : ℕ := totalEarnings - otherEarnings -- This directly from subtraction of earnings

theorem cauliflower_sales : cauliflowerEarnings = 310 :=
  by
    -- only the statement part, no actual proof needed
    sorry

end WeeklyMarket

end NUMINAMATH_GPT_cauliflower_sales_l245_24568
