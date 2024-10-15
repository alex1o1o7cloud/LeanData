import Mathlib

namespace NUMINAMATH_GPT_xiao_ming_percentile_l2147_214753

theorem xiao_ming_percentile (total_students : ℕ) (rank : ℕ) 
  (h1 : total_students = 48) (h2 : rank = 5) :
  ∃ p : ℕ, (p = 90 ∨ p = 91) ∧ (43 < (p * total_students) / 100) ∧ ((p * total_students) / 100 ≤ 44) :=
by
  sorry

end NUMINAMATH_GPT_xiao_ming_percentile_l2147_214753


namespace NUMINAMATH_GPT_range_of_f_l2147_214777

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then x + 1 else 2 ^ x

theorem range_of_f :
  {x : ℝ | f x + f (x - 0.5) > 1} = {x : ℝ | x > -0.25} :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l2147_214777


namespace NUMINAMATH_GPT_height_of_platform_l2147_214750

variable (h l w : ℕ)

-- Define the conditions as hypotheses
def measured_length_first_configuration : Prop := l + h - w = 40
def measured_length_second_configuration : Prop := w + h - l = 34

-- The goal is to prove that the height is 37 inches
theorem height_of_platform
  (h l w : ℕ)
  (config1 : measured_length_first_configuration h l w)
  (config2 : measured_length_second_configuration h l w) : 
  h = 37 := 
sorry

end NUMINAMATH_GPT_height_of_platform_l2147_214750


namespace NUMINAMATH_GPT_angle_equiv_terminal_side_l2147_214793

theorem angle_equiv_terminal_side (θ : ℤ) : 
  let θ_deg := (750 : ℕ)
  let reduced_angle := θ_deg % 360
  0 ≤ reduced_angle ∧ reduced_angle < 360 ∧ reduced_angle = 30:=
by
  sorry

end NUMINAMATH_GPT_angle_equiv_terminal_side_l2147_214793


namespace NUMINAMATH_GPT_uncle_dave_ice_cream_sandwiches_l2147_214788

theorem uncle_dave_ice_cream_sandwiches (n : ℕ) (s : ℕ) (total : ℕ) 
  (h1 : n = 11) (h2 : s = 13) (h3 : total = n * s) : total = 143 := by
  sorry

end NUMINAMATH_GPT_uncle_dave_ice_cream_sandwiches_l2147_214788


namespace NUMINAMATH_GPT_volume_frustum_as_fraction_of_original_l2147_214794

theorem volume_frustum_as_fraction_of_original :
  let original_base_edge := 40
  let original_altitude := 20
  let smaller_altitude := original_altitude / 3
  let smaller_base_edge := original_base_edge / 3
  let volume_original := (1 / 3) * (original_base_edge * original_base_edge) * original_altitude
  let volume_smaller := (1 / 3) * (smaller_base_edge * smaller_base_edge) * smaller_altitude
  let volume_frustum := volume_original - volume_smaller
  (volume_frustum / volume_original) = (87 / 96) :=
by
  let original_base_edge := 40
  let original_altitude := 20
  let smaller_altitude := original_altitude / 3
  let smaller_base_edge := original_base_edge / 3
  let volume_original := (1 / 3) * (original_base_edge * original_base_edge) * original_altitude
  let volume_smaller := (1 / 3) * (smaller_base_edge * smaller_base_edge) * smaller_altitude
  let volume_frustum := volume_original - volume_smaller
  have h : volume_frustum / volume_original = 87 / 96 := sorry
  exact h

end NUMINAMATH_GPT_volume_frustum_as_fraction_of_original_l2147_214794


namespace NUMINAMATH_GPT_factorize_expression_equilateral_triangle_of_sides_two_p_eq_m_plus_n_l2147_214769

-- Problem 1: Factorize x^2 - y^2 + 2x - 2y
theorem factorize_expression (x y : ℝ) : x^2 - y^2 + 2 * x - 2 * y = (x - y) * (x + y + 2) := 
by sorry

-- Problem 2: Determine the shape of a triangle given a^2 + c^2 - 2b(a - b + c) = 0
theorem equilateral_triangle_of_sides (a b c : ℝ) (h : a^2 + c^2 - 2 * b * (a - b + c) = 0) : a = b ∧ b = c :=
by sorry

-- Problem 3: Prove that 2p = m + n given (1/4)(m - n)^2 = (p - n)(m - p)
theorem two_p_eq_m_plus_n (m n p : ℝ) (h : (1/4) * (m - n)^2 = (p - n) * (m - p)) : 2 * p = m + n := 
by sorry

end NUMINAMATH_GPT_factorize_expression_equilateral_triangle_of_sides_two_p_eq_m_plus_n_l2147_214769


namespace NUMINAMATH_GPT_isosceles_triangle_roots_l2147_214704

theorem isosceles_triangle_roots (k : ℝ) (a b : ℝ) 
  (h1 : a = 2 ∨ b = 2)
  (h2 : a^2 - 6 * a + k = 0)
  (h3 : b^2 - 6 * b + k = 0) :
  k = 9 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_roots_l2147_214704


namespace NUMINAMATH_GPT_product_sum_125_l2147_214765

theorem product_sum_125 :
  ∀ (m n : ℕ), m ≥ n ∧
              (∀ (k : ℕ), 0 < k → |Real.log m - Real.log k| < Real.log n → k ≠ 0)
              → (m * n = 125) :=
by sorry

end NUMINAMATH_GPT_product_sum_125_l2147_214765


namespace NUMINAMATH_GPT_complement_set_A_is_04_l2147_214755

theorem complement_set_A_is_04 :
  let U := {0, 1, 2, 4}
  let compA := {1, 2}
  ∃ (A : Set ℕ), A = {0, 4} ∧ U = {0, 1, 2, 4} ∧ (U \ A) = compA := 
by
  sorry

end NUMINAMATH_GPT_complement_set_A_is_04_l2147_214755


namespace NUMINAMATH_GPT_smallest_next_divisor_l2147_214757

theorem smallest_next_divisor (n : ℕ) (h_even : n % 2 = 0) (h_4_digit : 1000 ≤ n ∧ n < 10000) (h_div_493 : 493 ∣ n) :
  ∃ d : ℕ, (d > 493 ∧ d ∣ n) ∧ ∀ e, (e > 493 ∧ e ∣ n) → d ≤ e ∧ d = 510 := by
  sorry

end NUMINAMATH_GPT_smallest_next_divisor_l2147_214757


namespace NUMINAMATH_GPT_maximum_root_l2147_214763

noncomputable def max_root (α β γ : ℝ) : ℝ := 
  if α ≥ β ∧ α ≥ γ then α 
  else if β ≥ α ∧ β ≥ γ then β 
  else γ

theorem maximum_root :
  ∃ α β γ : ℝ, α + β + γ = 14 ∧ α^2 + β^2 + γ^2 = 84 ∧ α^3 + β^3 + γ^3 = 584 ∧ max_root α β γ = 8 :=
by
  sorry

end NUMINAMATH_GPT_maximum_root_l2147_214763


namespace NUMINAMATH_GPT_total_distance_traveled_is_7_75_l2147_214775

open Real

def walking_time_minutes : ℝ := 30
def walking_rate : ℝ := 3.5

def running_time_minutes : ℝ := 45
def running_rate : ℝ := 8

theorem total_distance_traveled_is_7_75 :
  let walking_hours := walking_time_minutes / 60
  let distance_walked := walking_rate * walking_hours
  let running_hours := running_time_minutes / 60
  let distance_run := running_rate * running_hours
  let total_distance := distance_walked + distance_run
  total_distance = 7.75 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_traveled_is_7_75_l2147_214775


namespace NUMINAMATH_GPT_remaining_stock_is_120_l2147_214799

-- Definitions derived from conditions
def green_beans_weight : ℕ := 60
def rice_weight : ℕ := green_beans_weight - 30
def sugar_weight : ℕ := green_beans_weight - 10
def rice_lost_weight : ℕ := rice_weight / 3
def sugar_lost_weight : ℕ := sugar_weight / 5
def remaining_rice : ℕ := rice_weight - rice_lost_weight
def remaining_sugar : ℕ := sugar_weight - sugar_lost_weight
def remaining_stock_weight : ℕ := remaining_rice + remaining_sugar + green_beans_weight

-- Theorem
theorem remaining_stock_is_120 : remaining_stock_weight = 120 := by
  sorry

end NUMINAMATH_GPT_remaining_stock_is_120_l2147_214799


namespace NUMINAMATH_GPT_prove_unattainable_y_l2147_214724

noncomputable def unattainable_y : Prop :=
  ∀ (x y : ℝ), x ≠ -4 / 3 → y = (2 - x) / (3 * x + 4) → y ≠ -1 / 3

theorem prove_unattainable_y : unattainable_y :=
by
  intro x y h1 h2
  sorry

end NUMINAMATH_GPT_prove_unattainable_y_l2147_214724


namespace NUMINAMATH_GPT_prime_sum_55_l2147_214720

theorem prime_sum_55 (p q r s : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (hs : Prime s)
  (hpqrs : p < q ∧ q < r ∧ r < s) 
  (h_eqn : 1 - (1 : ℚ)/p - (1 : ℚ)/q - (1 : ℚ)/r - (1 : ℚ)/s = 1 / (p * q * r * s)) :
  p + q + r + s = 55 := 
sorry

end NUMINAMATH_GPT_prime_sum_55_l2147_214720


namespace NUMINAMATH_GPT_no_integer_valued_function_l2147_214735

theorem no_integer_valued_function (f : ℤ → ℤ) (h : ∀ (m n : ℤ), f (m + f n) = f m - n) : False :=
sorry

end NUMINAMATH_GPT_no_integer_valued_function_l2147_214735


namespace NUMINAMATH_GPT_project_estimated_hours_l2147_214776

theorem project_estimated_hours (extra_hours_per_day : ℕ) (normal_work_hours : ℕ) (days_to_finish : ℕ)
  (total_hours_estimation : ℕ)
  (h1 : extra_hours_per_day = 5)
  (h2 : normal_work_hours = 10)
  (h3 : days_to_finish = 100)
  (h4 : total_hours_estimation = days_to_finish * (normal_work_hours + extra_hours_per_day))
  : total_hours_estimation = 1500 :=
  by
  -- Proof to be provided 
  sorry

end NUMINAMATH_GPT_project_estimated_hours_l2147_214776


namespace NUMINAMATH_GPT_squirrel_travel_time_l2147_214727

theorem squirrel_travel_time :
  ∀ (speed distance : ℝ), speed = 5 → distance = 3 →
  (distance / speed) * 60 = 36 := by
  intros speed distance h_speed h_distance
  rw [h_speed, h_distance]
  norm_num

end NUMINAMATH_GPT_squirrel_travel_time_l2147_214727


namespace NUMINAMATH_GPT_remaining_students_average_l2147_214725

theorem remaining_students_average
  (N : ℕ) (A : ℕ) (M : ℕ) (B : ℕ) (E : ℕ)
  (h1 : N = 20)
  (h2 : A = 80)
  (h3 : M = 5)
  (h4 : B = 50)
  (h5 : E = (N - M))
  : (N * A - M * B) / E = 90 :=
by
  -- Using sorries to skip the proof
  sorry

end NUMINAMATH_GPT_remaining_students_average_l2147_214725


namespace NUMINAMATH_GPT_total_points_combined_l2147_214778

-- Definitions of the conditions
def Jack_points : ℕ := 8972
def Alex_Bella_points : ℕ := 21955

-- The problem statement to be proven
theorem total_points_combined : Jack_points + Alex_Bella_points = 30927 :=
by sorry

end NUMINAMATH_GPT_total_points_combined_l2147_214778


namespace NUMINAMATH_GPT_perpendicular_lines_k_value_l2147_214795

theorem perpendicular_lines_k_value (k : ℝ) : 
  (∃ (m₁ m₂ : ℝ), (m₁ = k/3) ∧ (m₂ = 3) ∧ (m₁ * m₂ = -1)) → k = -1 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_k_value_l2147_214795


namespace NUMINAMATH_GPT_more_action_figures_than_books_l2147_214791

-- Definitions of initial conditions
def books : ℕ := 3
def initial_action_figures : ℕ := 4
def added_action_figures : ℕ := 2

-- Definition of final number of action figures
def final_action_figures : ℕ := initial_action_figures + added_action_figures

-- Proposition to be proved
theorem more_action_figures_than_books : final_action_figures - books = 3 := by
  -- We leave the proof empty
  sorry

end NUMINAMATH_GPT_more_action_figures_than_books_l2147_214791


namespace NUMINAMATH_GPT_double_neg_cancel_l2147_214716

theorem double_neg_cancel (a : ℤ) : - (-2) = 2 :=
sorry

end NUMINAMATH_GPT_double_neg_cancel_l2147_214716


namespace NUMINAMATH_GPT_gcd_p4_minus_1_eq_240_l2147_214701

theorem gcd_p4_minus_1_eq_240 (p : ℕ) (hp : Prime p) (h_gt_5 : p > 5) :
  gcd (p^4 - 1) 240 = 240 :=
by sorry

end NUMINAMATH_GPT_gcd_p4_minus_1_eq_240_l2147_214701


namespace NUMINAMATH_GPT_scientific_notation_of_0_0000012_l2147_214785

theorem scientific_notation_of_0_0000012 :
  0.0000012 = 1.2 * 10^(-6) :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_0_0000012_l2147_214785


namespace NUMINAMATH_GPT_smallest_value_3a_plus_1_l2147_214783

theorem smallest_value_3a_plus_1 (a : ℚ) (h : 8 * a^2 + 6 * a + 5 = 2) : 3 * a + 1 = -5 / 4 :=
sorry

end NUMINAMATH_GPT_smallest_value_3a_plus_1_l2147_214783


namespace NUMINAMATH_GPT_min_value_of_x_plus_2y_l2147_214722

theorem min_value_of_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : 1 / x + 1 / y = 2) : 
  x + 2 * y ≥ (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_x_plus_2y_l2147_214722


namespace NUMINAMATH_GPT_tracy_sold_paintings_l2147_214792

-- Definitions of conditions
def total_customers := 20
def first_group_customers := 4
def paintings_per_first_group_customer := 2
def second_group_customers := 12
def paintings_per_second_group_customer := 1
def third_group_customers := 4
def paintings_per_third_group_customer := 4

-- Statement of the problem
theorem tracy_sold_paintings :
  (first_group_customers * paintings_per_first_group_customer) +
  (second_group_customers * paintings_per_second_group_customer) +
  (third_group_customers * paintings_per_third_group_customer) = 36 :=
by
  sorry

end NUMINAMATH_GPT_tracy_sold_paintings_l2147_214792


namespace NUMINAMATH_GPT_parabola_focus_coordinates_l2147_214798

noncomputable def parabola_focus (a b : ℝ) := (0, (1 / (4 * a)) + 2)

theorem parabola_focus_coordinates (a b : ℝ) (h₀ : a ≠ 0) (h₁ : ∀ x : ℝ, abs (a * x^2 + b * x + 2) ≥ 2) :
  parabola_focus a b = (0, 2 + (1 / (4 * a))) := sorry

end NUMINAMATH_GPT_parabola_focus_coordinates_l2147_214798


namespace NUMINAMATH_GPT_first_term_arith_seq_l2147_214747

noncomputable def is_increasing (a b c : ℕ) (d : ℕ) : Prop := b = a + d ∧ c = a + 2 * d ∧ 0 < d

theorem first_term_arith_seq (a₁ a₂ a₃ : ℕ) (d: ℕ) :
  is_increasing a₁ a₂ a₃ d ∧ a₁ + a₂ + a₃ = 12 ∧ a₁ * a₂ * a₃ = 48 → a₁ = 2 := sorry

end NUMINAMATH_GPT_first_term_arith_seq_l2147_214747


namespace NUMINAMATH_GPT_combinatorial_calculation_l2147_214723

-- Define the proof problem.
theorem combinatorial_calculation : (Nat.choose 20 6) = 2583 := sorry

end NUMINAMATH_GPT_combinatorial_calculation_l2147_214723


namespace NUMINAMATH_GPT_smallest_common_multiple_5_6_l2147_214728

theorem smallest_common_multiple_5_6 (n : ℕ) 
  (h_pos : 0 < n) 
  (h_5 : 5 ∣ n) 
  (h_6 : 6 ∣ n) :
  n = 30 :=
sorry

end NUMINAMATH_GPT_smallest_common_multiple_5_6_l2147_214728


namespace NUMINAMATH_GPT_radius_of_tangent_circle_l2147_214707

theorem radius_of_tangent_circle (a b : ℕ) (r1 r2 r3 : ℚ) (R : ℚ)
  (h1 : a = 6) (h2 : b = 8)
  (h3 : r1 = a / 2) (h4 : r2 = b / 2) (h5 : r3 = (Real.sqrt (a^2 + b^2)) / 2) :
  R = 144 / 23 := sorry

end NUMINAMATH_GPT_radius_of_tangent_circle_l2147_214707


namespace NUMINAMATH_GPT_intersection_when_a_eq_1_range_for_A_inter_B_eq_A_l2147_214782

noncomputable def f (x : ℝ) : ℝ := Real.log (3 - abs (x - 1))

def setA : Set ℝ := { x | 3 - abs (x - 1) > 0 }

def setB (a : ℝ) : Set ℝ := { x | x^2 - (a + 5) * x + 5 * a < 0 }

theorem intersection_when_a_eq_1 : (setA ∩ setB 1) = { x | 1 < x ∧ x < 4 } :=
by
  sorry

theorem range_for_A_inter_B_eq_A : { a | (setA ∩ setB a) = setA } = { a | a ≤ -2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_when_a_eq_1_range_for_A_inter_B_eq_A_l2147_214782


namespace NUMINAMATH_GPT_sampling_method_is_systematic_l2147_214732

-- Define the conditions of the problem
def conveyor_belt_transport : Prop := true
def inspectors_sampling_every_ten_minutes : Prop := true

-- Define what needs to be proved
theorem sampling_method_is_systematic :
  conveyor_belt_transport ∧ inspectors_sampling_every_ten_minutes → is_systematic_sampling :=
by
  sorry

-- Example definition that could be used in the proof
def is_systematic_sampling : Prop := true

end NUMINAMATH_GPT_sampling_method_is_systematic_l2147_214732


namespace NUMINAMATH_GPT_original_kittens_count_l2147_214721

theorem original_kittens_count 
  (K : ℕ) 
  (h1 : K - 3 + 9 = 12) : 
  K = 6 := by
sorry

end NUMINAMATH_GPT_original_kittens_count_l2147_214721


namespace NUMINAMATH_GPT_milk_production_days_l2147_214773

variable (x : ℕ)
def cows := 2 * x
def cans := 2 * x + 2
def days := 2 * x + 1
def total_cows := 2 * x + 4
def required_cans := 2 * x + 10

theorem milk_production_days :
  (total_cows * required_cans) = ((2 * x) * (2 * x + 1) * required_cans) / ((2 * x + 2) * (2 * x + 4)) :=
sorry

end NUMINAMATH_GPT_milk_production_days_l2147_214773


namespace NUMINAMATH_GPT_total_amount_to_pay_l2147_214731

theorem total_amount_to_pay (cost_earbuds cost_smartwatch : ℕ) (tax_rate_earbuds tax_rate_smartwatch : ℚ) 
  (h1 : cost_earbuds = 200) (h2 : cost_smartwatch = 300) 
  (h3 : tax_rate_earbuds = 0.15) (h4 : tax_rate_smartwatch = 0.12) : 
  (cost_earbuds + cost_earbuds * tax_rate_earbuds + cost_smartwatch + cost_smartwatch * tax_rate_smartwatch = 566) := 
by 
  sorry

end NUMINAMATH_GPT_total_amount_to_pay_l2147_214731


namespace NUMINAMATH_GPT_x_interval_l2147_214717

theorem x_interval (x : ℝ) (h1 : 1 / x < 3) (h2 : 1 / x > -4) (h3 : 2 * x - 1 > 0) : x > 1 / 2 := 
sorry

end NUMINAMATH_GPT_x_interval_l2147_214717


namespace NUMINAMATH_GPT_chocolate_more_expensive_l2147_214708

variables (C P : ℝ)
theorem chocolate_more_expensive (h : 7 * C > 8 * P) : 8 * C > 9 * P :=
sorry

end NUMINAMATH_GPT_chocolate_more_expensive_l2147_214708


namespace NUMINAMATH_GPT_initial_pills_count_l2147_214787

theorem initial_pills_count 
  (pills_taken_first_2_days : ℕ)
  (pills_taken_next_3_days : ℕ)
  (pills_taken_sixth_day : ℕ)
  (pills_left : ℕ)
  (h1 : pills_taken_first_2_days = 2 * 3 * 2)
  (h2 : pills_taken_next_3_days = 1 * 3 * 3)
  (h3 : pills_taken_sixth_day = 2)
  (h4 : pills_left = 27) :
  ∃ initial_pills : ℕ, initial_pills = pills_taken_first_2_days + pills_taken_next_3_days + pills_taken_sixth_day + pills_left :=
by
  sorry

end NUMINAMATH_GPT_initial_pills_count_l2147_214787


namespace NUMINAMATH_GPT_max_volume_l2147_214709

variable (x y z : ℝ) (V : ℝ)
variable (k : ℝ)

-- Define the constraint
def constraint := x + 2 * y + 3 * z = 180

-- Define the volume
def volume := x * y * z

-- The goal is to show that under the constraint, the maximum possible volume is 36000 cubic cm.
theorem max_volume :
  (∀ (x y z : ℝ) (h : constraint x y z), volume x y z ≤ 36000) :=
  sorry

end NUMINAMATH_GPT_max_volume_l2147_214709


namespace NUMINAMATH_GPT_william_napkins_l2147_214713

-- Define the given conditions
variables (O A C G W : ℕ)
variables (ho: O = 10)
variables (ha: A = 2 * O)
variables (hc: C = A / 2)
variables (hg: G = 3 * C)
variables (hw: W = 15)

-- Prove the total number of napkins William has now
theorem william_napkins (O A C G W : ℕ) (ho: O = 10) (ha: A = 2 * O)
  (hc: C = A / 2) (hg: G = 3 * C) (hw: W = 15) : W + (O + A + C + G) = 85 :=
by {
  sorry
}

end NUMINAMATH_GPT_william_napkins_l2147_214713


namespace NUMINAMATH_GPT_trajectory_circle_equation_l2147_214779

theorem trajectory_circle_equation :
  (∀ (x y : ℝ), dist (x, y) (0, 0) = 4 ↔ x^2 + y^2 = 16) :=  
sorry

end NUMINAMATH_GPT_trajectory_circle_equation_l2147_214779


namespace NUMINAMATH_GPT_four_hash_two_equals_forty_l2147_214715

def hash_op (a b : ℕ) : ℤ := (a^2 + b^2) * (a - b)

theorem four_hash_two_equals_forty : hash_op 4 2 = 40 := 
by
  sorry

end NUMINAMATH_GPT_four_hash_two_equals_forty_l2147_214715


namespace NUMINAMATH_GPT_sequence_b_l2147_214741

theorem sequence_b (b : ℕ → ℝ) (h₁ : b 1 = 1)
  (h₂ : ∀ n : ℕ, n ≥ 1 → (b (n + 1)) ^ 4 = 64 * (b n) ^ 4) :
  b 50 = 2 ^ 49 := by
  sorry

end NUMINAMATH_GPT_sequence_b_l2147_214741


namespace NUMINAMATH_GPT_additional_hours_on_days_without_practice_l2147_214734

def total_weekday_homework_hours : ℕ := 2 + 3 + 4 + 3 + 1
def total_weekend_homework_hours : ℕ := 8
def total_homework_hours : ℕ := total_weekday_homework_hours + total_weekend_homework_hours
def total_chore_hours : ℕ := 1 + 1
def total_hours : ℕ := total_homework_hours + total_chore_hours

theorem additional_hours_on_days_without_practice : ∀ (practice_nights : ℕ), 
  (2 ≤ practice_nights ∧ practice_nights ≤ 3) →
  (∃ tuesday_wednesday_thursday_weekend_day_hours : ℕ,
    tuesday_wednesday_thursday_weekend_day_hours = 15) :=
by
  intros practice_nights practice_nights_bounds
  -- Define days without practice in the worst case scenario
  let tuesday_hours := 3
  let wednesday_homework_hours := 4
  let wednesday_chore_hours := 1
  let thursday_hours := 3
  let weekend_day_hours := 4
  let days_without_practice_hours := tuesday_hours + (wednesday_homework_hours + wednesday_chore_hours) + thursday_hours + weekend_day_hours
  use days_without_practice_hours
  -- In the worst case, the total additional hours on days without practice should be 15.
  sorry

end NUMINAMATH_GPT_additional_hours_on_days_without_practice_l2147_214734


namespace NUMINAMATH_GPT_tea_price_l2147_214710

theorem tea_price 
  (x : ℝ)
  (total_cost_80kg_tea : ℝ := 80 * x)
  (total_cost_20kg_tea : ℝ := 20 * 20)
  (total_selling_price : ℝ := 1920)
  (profit_condition : 1.2 * (total_cost_80kg_tea + total_cost_20kg_tea) = total_selling_price) :
  x = 15 :=
by
  sorry

end NUMINAMATH_GPT_tea_price_l2147_214710


namespace NUMINAMATH_GPT_coconut_grove_problem_l2147_214796

theorem coconut_grove_problem
  (x : ℤ)
  (T40 : ℤ := x + 2)
  (T120 : ℤ := x)
  (T180 : ℤ := x - 2)
  (N_total : ℤ := 40 * (x + 2) + 120 * x + 180 * (x - 2))
  (T_total : ℤ := (x + 2) + x + (x - 2))
  (average_yield : ℤ := 100) :
  (N_total / T_total) = average_yield → x = 7 :=
by
  sorry

end NUMINAMATH_GPT_coconut_grove_problem_l2147_214796


namespace NUMINAMATH_GPT_evaluate_expression_l2147_214759

theorem evaluate_expression : (1 - 1 / (1 - 1 / (1 + 2))) = (-1 / 2) :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_l2147_214759


namespace NUMINAMATH_GPT_find_length_of_AL_l2147_214766

noncomputable def length_of_AL 
  (A B C L : ℝ) 
  (AB AC AL : ℝ)
  (BC : ℝ)
  (AB_ratio_AC : AB / AC = 5 / 2)
  (BAC_bisector : ∃k, L = k * BC)
  (vector_magnitude : (2 * AB + 5 * AC) = 2016) : Prop :=
  AL = 288

theorem find_length_of_AL 
  (A B C L : ℝ)
  (AB AC AL : ℝ)
  (BC : ℝ)
  (h1 : AB / AC = 5 / 2)
  (h2 : ∃k, L = k * BC)
  (h3 : (2 * AB + 5 * AC) = 2016) : length_of_AL A B C L AB AC AL BC h1 h2 h3 := sorry

end NUMINAMATH_GPT_find_length_of_AL_l2147_214766


namespace NUMINAMATH_GPT_number_of_perfect_squares_between_50_and_200_l2147_214705

theorem number_of_perfect_squares_between_50_and_200 :
  ∃ n: ℕ, 50 < n^2 ∧ n^2 < 200 ∧ (14 - 8 + 1 = 7) := sorry

end NUMINAMATH_GPT_number_of_perfect_squares_between_50_and_200_l2147_214705


namespace NUMINAMATH_GPT_complement_in_N_l2147_214758

variable (M : Set ℕ) (N : Set ℕ)
def complement_N (M N : Set ℕ) : Set ℕ := { x ∈ N | x ∉ M }

theorem complement_in_N (M : Set ℕ) (N : Set ℕ) : 
  M = {2, 3, 4} → N = {0, 2, 3, 4, 5} → complement_N M N = {0, 5} :=
by
  intro hM hN
  subst hM
  subst hN 
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_complement_in_N_l2147_214758


namespace NUMINAMATH_GPT_minimum_A2_minus_B2_l2147_214712

noncomputable def A (x y z : ℝ) : ℝ := 
  Real.sqrt (x + 6) + Real.sqrt (y + 7) + Real.sqrt (z + 12)

noncomputable def B (x y z : ℝ) : ℝ :=
  Real.sqrt (x + 2) + Real.sqrt (y + 3) + Real.sqrt (z + 5)

theorem minimum_A2_minus_B2 (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  (A x y z)^2 - (B x y z)^2 = 49.25 := 
by 
  sorry 

end NUMINAMATH_GPT_minimum_A2_minus_B2_l2147_214712


namespace NUMINAMATH_GPT_students_neither_football_nor_cricket_l2147_214797

def total_students : ℕ := 450
def football_players : ℕ := 325
def cricket_players : ℕ := 175
def both_players : ℕ := 100

theorem students_neither_football_nor_cricket : 
  total_students - (football_players + cricket_players - both_players) = 50 := by
  sorry

end NUMINAMATH_GPT_students_neither_football_nor_cricket_l2147_214797


namespace NUMINAMATH_GPT_max_r1_minus_r2_l2147_214738

noncomputable def ellipse (x y : ℝ) : Prop :=
  (x^2) / 2 + y^2 = 1

def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

def P (x y : ℝ) : Prop :=
  ellipse x y ∧ x > 0 ∧ y > 0

def r1 (x y : ℝ) (Q2 : ℝ × ℝ) : ℝ := 
  -- Assume a function that calculates the inradius of triangle ΔPF1Q2
  sorry

def r2 (x y : ℝ) (Q1 : ℝ × ℝ) : ℝ :=
  -- Assume a function that calculates the inradius of triangle ΔPF2Q1
  sorry

theorem max_r1_minus_r2 :
  ∃ (x y : ℝ) (Q1 Q2 : ℝ × ℝ), P x y →
    r1 x y Q2 - r2 x y Q1 = 1/3 := 
sorry

end NUMINAMATH_GPT_max_r1_minus_r2_l2147_214738


namespace NUMINAMATH_GPT_eq_root_condition_l2147_214745

theorem eq_root_condition (k : ℝ) 
    (h_discriminant : -4 * k + 5 ≥ 0)
    (h_roots : ∃ x1 x2 : ℝ, 
        (x1 + x2 = 1 - 2 * k) ∧ 
        (x1 * x2 = k^2 - 1) ∧ 
        (x1^2 + x2^2 = 16 + x1 * x2)) :
    k = -2 :=
sorry

end NUMINAMATH_GPT_eq_root_condition_l2147_214745


namespace NUMINAMATH_GPT_ratio_of_couch_to_table_l2147_214742

theorem ratio_of_couch_to_table
    (C T X : ℝ)
    (h1 : T = 3 * C)
    (h2 : X = 300)
    (h3 : C + T + X = 380) :
  X / T = 5 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_couch_to_table_l2147_214742


namespace NUMINAMATH_GPT_find_a_l2147_214786

theorem find_a 
  (x y a : ℝ) 
  (hx : x = 1) 
  (hy : y = -3) 
  (h : a * x - y = 1) : 
  a = -2 := 
  sorry

end NUMINAMATH_GPT_find_a_l2147_214786


namespace NUMINAMATH_GPT_johnson_and_carter_tie_in_september_l2147_214744

def monthly_home_runs_johnson : List ℕ := [3, 14, 18, 13, 10, 16, 14, 5]
def monthly_home_runs_carter : List ℕ := [5, 9, 22, 11, 15, 17, 9, 9]

def cumulative_home_runs (runs : List ℕ) (up_to : ℕ) : ℕ :=
  (runs.take up_to).sum

theorem johnson_and_carter_tie_in_september :
  cumulative_home_runs monthly_home_runs_johnson 7 = cumulative_home_runs monthly_home_runs_carter 7 :=
by
  sorry

end NUMINAMATH_GPT_johnson_and_carter_tie_in_september_l2147_214744


namespace NUMINAMATH_GPT_smallest_integer_C_l2147_214737

-- Define the function f(n) = 6^n / n!
def f (n : ℕ) : ℚ := (6 ^ n) / (Nat.factorial n)

theorem smallest_integer_C (C : ℕ) (h : ∀ n : ℕ, n > 0 → f n ≤ C) : C = 65 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_C_l2147_214737


namespace NUMINAMATH_GPT_solve_for_x_l2147_214718

noncomputable def log_b (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_for_x (b x : ℝ) (hb : b > 1) (hx : x > 0) :
  (4 * x) ^ log_b b 4 - (5 * x) ^ log_b b 5 + x = 0 ↔ x = 1 :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_solve_for_x_l2147_214718


namespace NUMINAMATH_GPT_circle_reflection_l2147_214781

theorem circle_reflection (x y : ℝ) (hx : x = 8) (hy : y = -3)
    (new_x new_y : ℝ) (hne_x : new_x = 3) (hne_y : new_y = -8) :
    (new_x, new_y) = (-y, -x) := by
  sorry

end NUMINAMATH_GPT_circle_reflection_l2147_214781


namespace NUMINAMATH_GPT_circle_equation_l2147_214703

theorem circle_equation (x y : ℝ) (h_eq : x = 0) (k_eq : y = -2) (r_eq : y = 4) :
  (x - 0)^2 + (y - (-2))^2 = 16 := 
by
  sorry

end NUMINAMATH_GPT_circle_equation_l2147_214703


namespace NUMINAMATH_GPT_negative_root_m_positive_l2147_214700

noncomputable def is_negative_root (m : ℝ) : Prop :=
  ∃ x : ℝ, x < 0 ∧ x^2 + m * x - 4 = 0

theorem negative_root_m_positive : ∀ m : ℝ, is_negative_root m → m > 0 :=
by
  intro m
  intro h
  sorry

end NUMINAMATH_GPT_negative_root_m_positive_l2147_214700


namespace NUMINAMATH_GPT_triangle_side_length_l2147_214749

theorem triangle_side_length (BC : ℝ) (A : ℝ) (B : ℝ) (AB : ℝ) :
  BC = 2 → A = π / 3 → B = π / 4 → AB = (3 * Real.sqrt 2 + Real.sqrt 6) / 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_length_l2147_214749


namespace NUMINAMATH_GPT_arithmetic_geometric_inequality_l2147_214736

theorem arithmetic_geometric_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a ≠ b) :
  let A := (a + b) / 2
  let B := Real.sqrt (a * b)
  B < (a - b)^2 / (8 * (A - B)) ∧ (a - b)^2 / (8 * (A - B)) < A :=
by
  let A := (a + b) / 2
  let B := Real.sqrt (a * b)
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_inequality_l2147_214736


namespace NUMINAMATH_GPT_solution_set_x2_minus_x_lt_0_l2147_214729

theorem solution_set_x2_minus_x_lt_0 :
  ∀ x : ℝ, (0 < x ∧ x < 1) ↔ x^2 - x < 0 := 
by
  sorry

end NUMINAMATH_GPT_solution_set_x2_minus_x_lt_0_l2147_214729


namespace NUMINAMATH_GPT_geologists_probability_l2147_214719

theorem geologists_probability
  (n roads : ℕ) (speed_per_hour : ℕ) 
  (angle_between_neighbors : ℕ)
  (distance_limit : ℝ) : 
  n = 6 ∧ speed_per_hour = 4 ∧ angle_between_neighbors = 60 ∧ distance_limit = 6 → 
  prob_distance_at_least_6_km = 0.5 :=
by
  sorry

noncomputable def prob_distance_at_least_6_km : ℝ := 0.5  -- Placeholder definition

end NUMINAMATH_GPT_geologists_probability_l2147_214719


namespace NUMINAMATH_GPT_pencil_cost_l2147_214733

theorem pencil_cost (P : ℝ) : 
  (∀ pen_cost total : ℝ, pen_cost = 3.50 → total = 291 → 38 * P + 56 * pen_cost = total → P = 2.50) :=
by
  intros pen_cost total h1 h2 h3
  sorry

end NUMINAMATH_GPT_pencil_cost_l2147_214733


namespace NUMINAMATH_GPT_leopards_to_rabbits_ratio_l2147_214726

theorem leopards_to_rabbits_ratio :
  let antelopes := 80
  let rabbits := antelopes + 34
  let hyenas := antelopes + rabbits - 42
  let wild_dogs := hyenas + 50
  let total_animals := 605
  let leopards := total_animals - antelopes - rabbits - hyenas - wild_dogs
  leopards / rabbits = 1 / 2 :=
by
  let antelopes := 80
  let rabbits := antelopes + 34
  let hyenas := antelopes + rabbits - 42
  let wild_dogs := hyenas + 50
  let total_animals := 605
  let leopards := total_animals - antelopes - rabbits - hyenas - wild_dogs
  sorry

end NUMINAMATH_GPT_leopards_to_rabbits_ratio_l2147_214726


namespace NUMINAMATH_GPT_correct_operation_l2147_214702

theorem correct_operation (a : ℝ) : (a^3)^3 = a^9 := 
sorry

end NUMINAMATH_GPT_correct_operation_l2147_214702


namespace NUMINAMATH_GPT_amusement_park_ticket_price_l2147_214770

theorem amusement_park_ticket_price
  (num_people_weekday : ℕ)
  (num_people_saturday : ℕ)
  (num_people_sunday : ℕ)
  (total_people_week : ℕ)
  (total_revenue_week : ℕ)
  (people_per_day_weekday : num_people_weekday = 100)
  (people_saturday : num_people_saturday = 200)
  (people_sunday : num_people_sunday = 300)
  (total_people : total_people_week = 1000)
  (total_revenue : total_revenue_week = 3000)
  (total_people_calc : 5 * num_people_weekday + num_people_saturday + num_people_sunday = total_people_week)
  (revenue_eq : total_people_week * 3 = total_revenue_week) :
  3 = 3 :=
by
  sorry

end NUMINAMATH_GPT_amusement_park_ticket_price_l2147_214770


namespace NUMINAMATH_GPT_mike_max_marks_l2147_214748

theorem mike_max_marks (m : ℕ) (h : 30 * m = 237 * 10) : m = 790 := by
  sorry

end NUMINAMATH_GPT_mike_max_marks_l2147_214748


namespace NUMINAMATH_GPT_min_value_of_quadratic_function_l2147_214711

def quadratic_function (x : ℝ) : ℝ :=
  x^2 + 2 * x - 5

theorem min_value_of_quadratic_function :
  ∃ x : ℝ, quadratic_function x = -1 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_quadratic_function_l2147_214711


namespace NUMINAMATH_GPT_simple_interest_rate_l2147_214754

theorem simple_interest_rate 
  (P A T : ℝ) 
  (hP : P = 900) 
  (hA : A = 950) 
  (hT : T = 5) 
  : (A - P) * 100 / (P * T) = 1.11 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l2147_214754


namespace NUMINAMATH_GPT_robbery_participants_l2147_214762

variables (A B V G : Prop)

-- Conditions
axiom cond1 : ¬G → (B ∧ ¬A)
axiom cond2 : V → ¬A ∧ ¬B
axiom cond3 : G → B
axiom cond4 : B → (A ∨ V)

-- Theorem to be proved
theorem robbery_participants : A ∧ B ∧ G :=
by 
  sorry

end NUMINAMATH_GPT_robbery_participants_l2147_214762


namespace NUMINAMATH_GPT_subtraction_makes_divisible_l2147_214764

theorem subtraction_makes_divisible :
  ∃ n : Nat, 9671 - n % 2 = 0 ∧ n = 1 :=
by
  sorry

end NUMINAMATH_GPT_subtraction_makes_divisible_l2147_214764


namespace NUMINAMATH_GPT_focus_of_parabola_l2147_214767

/-- Given a quadratic function f(x) = ax^2 + bx + 2 where a ≠ 0, and for any real number x, it holds that |f(x)| ≥ 2,
    prove that the coordinates of the focus of the parabolic curve are (0, 1 / (4 * a) + 2). -/
theorem focus_of_parabola (a b : ℝ) (h_a : a ≠ 0)
  (h_f : ∀ x : ℝ, |a * x^2 + b * x + 2| ≥ 2) :
  (0, (1 / (4 * a) + 2)) = (0, (1 / (4 * a) + 2)) :=
by
  sorry

end NUMINAMATH_GPT_focus_of_parabola_l2147_214767


namespace NUMINAMATH_GPT_complex_fraction_sum_zero_l2147_214772

section complex_proof
open Complex

theorem complex_fraction_sum_zero (z1 z2 : ℂ) (hz1 : z1 = 1 + I) (hz2 : z2 = 1 - I) :
  (z1 / z2) + (z2 / z1) = 0 := by
  sorry
end complex_proof

end NUMINAMATH_GPT_complex_fraction_sum_zero_l2147_214772


namespace NUMINAMATH_GPT_value_subtracted_3_times_number_eq_1_l2147_214771

variable (n : ℝ) (v : ℝ)

theorem value_subtracted_3_times_number_eq_1 (h1 : n = 1.0) (h2 : 3 * n - v = 2 * n) : v = 1 :=
by
  sorry

end NUMINAMATH_GPT_value_subtracted_3_times_number_eq_1_l2147_214771


namespace NUMINAMATH_GPT_Glenn_total_expenditure_l2147_214756

-- Define initial costs and discounts
def ticket_cost_Monday : ℕ := 5
def ticket_cost_Wednesday : ℕ := 2 * ticket_cost_Monday
def ticket_cost_Saturday : ℕ := 5 * ticket_cost_Monday
def discount_Wednesday (cost : ℕ) : ℕ := cost * 90 / 100
def additional_expense_Saturday : ℕ := 7

-- Define number of attendees
def attendees_Wednesday : ℕ := 4
def attendees_Saturday : ℕ := 2

-- Calculate total costs
def total_cost_Wednesday : ℕ :=
  attendees_Wednesday * discount_Wednesday ticket_cost_Wednesday
def total_cost_Saturday : ℕ :=
  attendees_Saturday * ticket_cost_Saturday + additional_expense_Saturday

-- Calculate the total money spent by Glenn
def total_spent : ℕ :=
  total_cost_Wednesday + total_cost_Saturday

-- Combine all conditions and conclusions into proof statement
theorem Glenn_total_expenditure : total_spent = 93 := by
  sorry

end NUMINAMATH_GPT_Glenn_total_expenditure_l2147_214756


namespace NUMINAMATH_GPT_smallest_mn_sum_l2147_214730

theorem smallest_mn_sum {n m : ℕ} (h1 : n > m) (h2 : 1978 ^ n % 1000 = 1978 ^ m % 1000) (h3 : m ≥ 1) : m + n = 106 := 
sorry

end NUMINAMATH_GPT_smallest_mn_sum_l2147_214730


namespace NUMINAMATH_GPT_joan_total_cents_l2147_214740

-- Conditions
def quarters : ℕ := 12
def dimes : ℕ := 8
def nickels : ℕ := 15
def pennies : ℕ := 25

def value_of_quarter : ℕ := 25
def value_of_dime : ℕ := 10
def value_of_nickel : ℕ := 5
def value_of_penny : ℕ := 1

-- The problem statement
theorem joan_total_cents : 
  (quarters * value_of_quarter + dimes * value_of_dime + nickels * value_of_nickel + pennies * value_of_penny) = 480 := 
  sorry

end NUMINAMATH_GPT_joan_total_cents_l2147_214740


namespace NUMINAMATH_GPT_find_six_digit_numbers_l2147_214790

variable (m n : ℕ)

-- Definition that the original number becomes six-digit when multiplied by 4
def is_six_digit (x : ℕ) : Prop := x ≥ 100000 ∧ x < 1000000

-- Conditions
def original_number := 100 * m + n
def new_number := 10000 * n + m
def satisfies_conditions (m n : ℕ) : Prop :=
  is_six_digit (100 * m + n) ∧
  is_six_digit (10000 * n + m) ∧
  4 * (100 * m + n) = 10000 * n + m

-- Theorem statement
theorem find_six_digit_numbers (h₁ : satisfies_conditions 1428 57)
                               (h₂ : satisfies_conditions 1904 76)
                               (h₃ : satisfies_conditions 2380 95) :
  ∃ m n, satisfies_conditions m n :=
  sorry -- Proof omitted

end NUMINAMATH_GPT_find_six_digit_numbers_l2147_214790


namespace NUMINAMATH_GPT_bikers_meet_again_in_36_minutes_l2147_214768

theorem bikers_meet_again_in_36_minutes :
    Nat.lcm 12 18 = 36 :=
sorry

end NUMINAMATH_GPT_bikers_meet_again_in_36_minutes_l2147_214768


namespace NUMINAMATH_GPT_amount_spent_on_petrol_l2147_214714

theorem amount_spent_on_petrol
    (rent milk groceries education miscellaneous savings salary petrol : ℝ)
    (h1 : rent = 5000)
    (h2 : milk = 1500)
    (h3 : groceries = 4500)
    (h4 : education = 2500)
    (h5 : miscellaneous = 2500)
    (h6 : savings = 0.10 * salary)
    (h7 : savings = 2000)
    (total_salary : salary = 20000) : petrol = 2000 := by
  sorry

end NUMINAMATH_GPT_amount_spent_on_petrol_l2147_214714


namespace NUMINAMATH_GPT_passing_marks_l2147_214760

theorem passing_marks (T P : ℝ) (h1 : 0.30 * T = P - 30) (h2 : 0.45 * T = P + 15) : P = 120 := 
by
  sorry

end NUMINAMATH_GPT_passing_marks_l2147_214760


namespace NUMINAMATH_GPT_solve_digits_A_B_l2147_214743

theorem solve_digits_A_B :
    ∃ (A B : ℕ), A ≠ B ∧ A < 10 ∧ B < 10 ∧ 
    (A * (10 * A + B) = 100 * B + 10 * A + A) ∧ A = 8 ∧ B = 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_digits_A_B_l2147_214743


namespace NUMINAMATH_GPT_distinct_seatings_l2147_214774

theorem distinct_seatings : 
  ∃ n : ℕ, (n = 288000) ∧ 
  (∀ (men wives : Fin 6 → ℕ),
  ∃ (f : (Fin 12) → ℕ), 
  (∀ i, f (i + 1) % 12 ≠ f i) ∧
  (∀ i, f i % 2 = 0) ∧
  (∀ j, f (2 * j) = men j ∧ f (2 * j + 1) = wives j)) :=
by
  sorry

end NUMINAMATH_GPT_distinct_seatings_l2147_214774


namespace NUMINAMATH_GPT_distance_between_points_l2147_214761

theorem distance_between_points :
  let x1 := 1
  let y1 := 16
  let x2 := 9
  let y2 := 3
  let distance := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance = Real.sqrt 233 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_points_l2147_214761


namespace NUMINAMATH_GPT_total_balloons_l2147_214706

theorem total_balloons (F S M : ℕ) (hF : F = 5) (hS : S = 6) (hM : M = 7) : F + S + M = 18 :=
by 
  sorry

end NUMINAMATH_GPT_total_balloons_l2147_214706


namespace NUMINAMATH_GPT_correct_exponent_operation_l2147_214752

theorem correct_exponent_operation (x : ℝ) : x ^ 3 * x ^ 2 = x ^ 5 :=
by sorry

end NUMINAMATH_GPT_correct_exponent_operation_l2147_214752


namespace NUMINAMATH_GPT_river_current_speed_l2147_214780

theorem river_current_speed :
  ∀ (D v A_speed B_speed time_interval : ℝ),
    D = 200 →
    A_speed = 36 →
    B_speed = 64 →
    time_interval = 4 →
    3 * D = (A_speed + v) * 2 * (1 + time_interval / ((A_speed + v) + (B_speed - v))) * 200 :=
sorry

end NUMINAMATH_GPT_river_current_speed_l2147_214780


namespace NUMINAMATH_GPT_probability_diff_colors_l2147_214746

theorem probability_diff_colors :
  let total_marbles := 24
  let prob_diff_colors := 
    (4 / 24) * (5 / 23) + 
    (4 / 24) * (12 / 23) + 
    (4 / 24) * (3 / 23) + 
    (5 / 24) * (12 / 23) + 
    (5 / 24) * (3 / 23) + 
    (12 / 24) * (3 / 23)
  prob_diff_colors = 191 / 552 :=
by sorry

end NUMINAMATH_GPT_probability_diff_colors_l2147_214746


namespace NUMINAMATH_GPT_difference_q_r_share_l2147_214789

theorem difference_q_r_share (p q r : ℕ) (x : ℕ) (h_ratio : p = 3 * x) (h_ratio_q : q = 7 * x) (h_ratio_r : r = 12 * x) (h_diff_pq : q - p = 4400) : q - r = 5500 :=
by
  sorry

end NUMINAMATH_GPT_difference_q_r_share_l2147_214789


namespace NUMINAMATH_GPT_smallest_prime_p_l2147_214739

theorem smallest_prime_p 
  (p q r : ℕ) 
  (h1 : Nat.Prime p) 
  (h2 : Nat.Prime q) 
  (h3 : r > 0) 
  (h4 : p + q = r) 
  (h5 : q < p) 
  (h6 : q = 2) 
  (h7 : Nat.Prime r)  
  : p = 3 := 
sorry

end NUMINAMATH_GPT_smallest_prime_p_l2147_214739


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2147_214784

theorem quadratic_inequality_solution
  (a b : ℝ)
  (h1 : ∀ x : ℝ, x^2 + a * x + b > 0 ↔ (x < -2 ∨ -1/2 < x)) :
  ∀ x : ℝ, b * x^2 + a * x + 1 < 0 ↔ -2 < x ∧ x < -1/2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2147_214784


namespace NUMINAMATH_GPT_train_speed_km_per_hr_l2147_214751

theorem train_speed_km_per_hr
  (train_length : ℝ) 
  (platform_length : ℝ)
  (time_seconds : ℝ) 
  (h_train_length : train_length = 470) 
  (h_platform_length : platform_length = 520) 
  (h_time_seconds : time_seconds = 64.79481641468682) :
  (train_length + platform_length) / time_seconds * 3.6 = 54.975 := 
sorry

end NUMINAMATH_GPT_train_speed_km_per_hr_l2147_214751
