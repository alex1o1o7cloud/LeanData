import Mathlib

namespace NUMINAMATH_GPT_factory_production_schedule_l1377_137779

noncomputable def production_equation (x : ℝ) : Prop :=
  (1000 / x) - (1000 / (1.2 * x)) = 2

theorem factory_production_schedule (x : ℝ) (hx : x ≠ 0) : production_equation x := 
by 
  -- Assumptions based on conditions:
  -- Factory plans to produce total of 1000 sets of protective clothing.
  -- Actual production is 20% more than planned.
  -- Task completed 2 days ahead of original schedule.
  -- We need to show: (1000 / x) - (1000 / (1.2 * x)) = 2
  sorry

end NUMINAMATH_GPT_factory_production_schedule_l1377_137779


namespace NUMINAMATH_GPT_cyclist_rejoins_group_time_l1377_137744

noncomputable def travel_time (group_speed cyclist_speed distance : ℝ) : ℝ :=
  distance / (cyclist_speed - group_speed)

theorem cyclist_rejoins_group_time
  (group_speed : ℝ := 35)
  (cyclist_speed : ℝ := 45)
  (distance : ℝ := 10)
  : travel_time group_speed cyclist_speed distance * 2 = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_cyclist_rejoins_group_time_l1377_137744


namespace NUMINAMATH_GPT_correct_answer_l1377_137755

theorem correct_answer (x : ℝ) (h : 3 * x - 10 = 50) : 3 * x + 10 = 70 :=
sorry

end NUMINAMATH_GPT_correct_answer_l1377_137755


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1377_137733

theorem arithmetic_sequence_common_difference
    (a : ℕ → ℝ)
    (h1 : a 2 + a 3 = 9)
    (h2 : a 4 + a 5 = 21)
    (h3 : ∀ n, a (n + 1) = a n + d) : d = 3 :=
        sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1377_137733


namespace NUMINAMATH_GPT_exists_integer_root_l1377_137784

theorem exists_integer_root (a b c d : ℤ) (ha : a ≠ 0)
  (h : ∀ n : ℕ, ∃ (x y : ℤ), x ≠ y ∧ x * (a * x^3 + b * x^2 + c * x + d) = y * (a * y^3 + b * y^2 + c * y + d)) :
  ∃ z : ℤ, a * z^3 + b * z^2 + c * z + d = 0 :=
by
  sorry

end NUMINAMATH_GPT_exists_integer_root_l1377_137784


namespace NUMINAMATH_GPT_intersection_roots_l1377_137731

theorem intersection_roots :
  x^2 - 4*x - 5 = 0 → (x = 5 ∨ x = -1) := by
  sorry

end NUMINAMATH_GPT_intersection_roots_l1377_137731


namespace NUMINAMATH_GPT_three_x_plus_y_eq_zero_l1377_137736

theorem three_x_plus_y_eq_zero (x y : ℝ) (h : (2 * x + y) ^ 3 + x ^ 3 + 3 * x + y = 0) : 3 * x + y = 0 :=
sorry

end NUMINAMATH_GPT_three_x_plus_y_eq_zero_l1377_137736


namespace NUMINAMATH_GPT_paco_countertop_total_weight_l1377_137714

theorem paco_countertop_total_weight :
  0.3333333333333333 + 0.3333333333333333 + 0.08333333333333333 = 0.75 :=
sorry

end NUMINAMATH_GPT_paco_countertop_total_weight_l1377_137714


namespace NUMINAMATH_GPT_find_q_l1377_137793

noncomputable def p : ℝ := -(5 / 6)
noncomputable def g (x : ℝ) : ℝ := p * x^2 + (5 / 6) * x + 5

theorem find_q :
  (∀ x : ℝ, g x = p * x^2 + q * x + r) ∧ 
  (g (-2) = 0) ∧ 
  (g 3 = 0) ∧ 
  (g 1 = 5) 
  → q = 5 / 6 :=
sorry

end NUMINAMATH_GPT_find_q_l1377_137793


namespace NUMINAMATH_GPT_ab_leq_one_l1377_137792

theorem ab_leq_one (a b x : ℝ) (h1 : (x + a) * (x + b) = 9) (h2 : x = a + b) : a * b ≤ 1 := 
sorry

end NUMINAMATH_GPT_ab_leq_one_l1377_137792


namespace NUMINAMATH_GPT_find_f_m_l1377_137771

-- Definitions based on the conditions
def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 3

axiom condition (m a : ℝ) : f (-m) a = 1

-- The statement to be proven
theorem find_f_m (m a : ℝ) (hm : f (-m) a = 1) : f m a = 5 := 
by sorry

end NUMINAMATH_GPT_find_f_m_l1377_137771


namespace NUMINAMATH_GPT_P_parity_Q_div_by_3_l1377_137767

-- Define polynomial P(x)
def P (x p q : ℤ) : ℤ := x*x + p*x + q

-- Define polynomial Q(x)
def Q (x p q : ℤ) : ℤ := x*x*x + p*x + q

-- Part (a) proof statement
theorem P_parity (p q : ℤ) (h1 : Odd p) (h2 : Even q ∨ Odd q) :
  (∀ x : ℤ, Even (P x p q)) ∨ (∀ x : ℤ, Odd (P x p q)) :=
sorry

-- Part (b) proof statement
theorem Q_div_by_3 (p q : ℤ) (h1 : q % 3 = 0) (h2 : p % 3 = 2) :
  ∀ x : ℤ, Q x p q % 3 = 0 :=
sorry

end NUMINAMATH_GPT_P_parity_Q_div_by_3_l1377_137767


namespace NUMINAMATH_GPT_fraction_of_water_l1377_137757

/-- 
  Prove that the fraction of the mixture that is water is (\frac{2}{5}) 
  given the total weight of the mixture is 40 pounds, 
  1/4 of the mixture is sand, 
  and the remaining 14 pounds of the mixture is gravel. 
-/
theorem fraction_of_water 
  (total_weight : ℝ)
  (weight_sand : ℝ)
  (weight_gravel : ℝ)
  (weight_water : ℝ)
  (h1 : total_weight = 40)
  (h2 : weight_sand = (1/4) * total_weight)
  (h3 : weight_gravel = 14)
  (h4 : weight_water = total_weight - (weight_sand + weight_gravel)) :
  (weight_water / total_weight) = 2/5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_water_l1377_137757


namespace NUMINAMATH_GPT_temperature_difference_l1377_137741

def Shanghai_temp : ℤ := 3
def Beijing_temp : ℤ := -5

theorem temperature_difference :
  Shanghai_temp - Beijing_temp = 8 := by
  sorry

end NUMINAMATH_GPT_temperature_difference_l1377_137741


namespace NUMINAMATH_GPT_problem_l1377_137794

noncomputable def key_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : Real.sqrt (x * y) ≤ 1) 
    : Prop := ∃ z : ℝ, 0 < z ∧ z = 2 * (x + y) / (x + y + 2)^2

theorem problem (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
    (habcd : a + b + c + d = 2) :
    (a^2 / (a^2 + 1)^2) + (b^2 / (b^2 + 1)^2) + (c^2 / (c^2 + 1)^2) + (d^2 / (d^2 + 1)^2) ≤ 16 / 25 := 
sorry

end NUMINAMATH_GPT_problem_l1377_137794


namespace NUMINAMATH_GPT_max_value_inequality_l1377_137725

theorem max_value_inequality (x y k : ℝ) (hx : 0 < x) (hy : 0 < y) (hk : 0 < k) :
  (kx + y)^2 / (x^2 + y^2) ≤ 2 :=
sorry

end NUMINAMATH_GPT_max_value_inequality_l1377_137725


namespace NUMINAMATH_GPT_cups_of_ketchup_l1377_137798

-- Define variables and conditions
variables (k : ℕ)
def vinegar : ℕ := 1
def honey : ℕ := 1
def sauce_per_burger : ℚ := 1 / 4
def sauce_per_pulled_pork : ℚ := 1 / 6
def burgers : ℕ := 8
def pulled_pork_sandwiches : ℕ := 18

-- Main theorem statement
theorem cups_of_ketchup (h : 8 * sauce_per_burger + 18 * sauce_per_pulled_pork = k + vinegar + honey) : k = 3 :=
  by
    sorry

end NUMINAMATH_GPT_cups_of_ketchup_l1377_137798


namespace NUMINAMATH_GPT_find_slope_l1377_137711

noncomputable def parabola_equation (x y : ℝ) := y^2 = 8 * x

def point_M : ℝ × ℝ := (-2, 2)

def line_through_focus (k x : ℝ) : ℝ := k * (x - 2)

def focus : ℝ × ℝ := (2, 0)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem find_slope (k : ℝ) : 
  (∀ x y A B, 
    parabola_equation x y → 
    (x = A ∨ x = B) → 
    line_through_focus k x = y → 
    parabola_equation A (k * (A - 2)) → 
    parabola_equation B (k * (B - 2)) → 
    dot_product (A + 2, (k * (A -2)) - 2) (B + 2, (k * (B - 2)) - 2) = 0) →
  k = 2 :=
sorry

end NUMINAMATH_GPT_find_slope_l1377_137711


namespace NUMINAMATH_GPT_rachel_envelopes_first_hour_l1377_137761

theorem rachel_envelopes_first_hour (total_envelopes : ℕ) (hours : ℕ) (e2 : ℕ) (e_per_hour : ℕ) :
  total_envelopes = 1500 → hours = 8 → e2 = 141 → e_per_hour = 204 →
  ∃ e1 : ℕ, e1 = 135 :=
by
  sorry

end NUMINAMATH_GPT_rachel_envelopes_first_hour_l1377_137761


namespace NUMINAMATH_GPT_distinct_integers_problem_l1377_137713

variable (a b c d e : ℤ)

theorem distinct_integers_problem
  (h1 : a ≠ b) 
  (h2 : a ≠ c) 
  (h3 : a ≠ d) 
  (h4 : a ≠ e) 
  (h5 : b ≠ c) 
  (h6 : b ≠ d) 
  (h7 : b ≠ e) 
  (h8 : c ≠ d) 
  (h9 : c ≠ e) 
  (h10 : d ≠ e) 
  (h_prod : (4 - a) * (4 - b) * (4 - c) * (4 - d) * (4 - e) = 12) : 
  a + b + c + d + e = 17 := 
sorry

end NUMINAMATH_GPT_distinct_integers_problem_l1377_137713


namespace NUMINAMATH_GPT_total_pies_sold_l1377_137709

-- Defining the conditions
def pies_per_day : ℕ := 8
def days_in_week : ℕ := 7

-- Proving the question
theorem total_pies_sold : pies_per_day * days_in_week = 56 :=
by
  sorry

end NUMINAMATH_GPT_total_pies_sold_l1377_137709


namespace NUMINAMATH_GPT_expand_product_equivalence_l1377_137701

variable (x : ℝ)  -- Assuming x is a real number

theorem expand_product_equivalence : (x + 5) * (x + 7) = x^2 + 12 * x + 35 :=
by
  sorry

end NUMINAMATH_GPT_expand_product_equivalence_l1377_137701


namespace NUMINAMATH_GPT_smallest_pos_int_mod_congruence_l1377_137716

theorem smallest_pos_int_mod_congruence : ∃ n : ℕ, 0 < n ∧ n ≡ 2 [MOD 31] ∧ 5 * n ≡ 409 [MOD 31] :=
by
  sorry

end NUMINAMATH_GPT_smallest_pos_int_mod_congruence_l1377_137716


namespace NUMINAMATH_GPT_final_price_is_99_l1377_137746

-- Conditions:
def original_price : ℝ := 120
def coupon_discount : ℝ := 10
def membership_discount_rate : ℝ := 0.10

-- Define final price calculation
def final_price (original_price coupon_discount membership_discount_rate : ℝ) : ℝ :=
  let price_after_coupon := original_price - coupon_discount
  let membership_discount := price_after_coupon * membership_discount_rate
  price_after_coupon - membership_discount

-- Question: Is the final price equal to $99?
theorem final_price_is_99 :
  final_price original_price coupon_discount membership_discount_rate = 99 :=
by
  sorry

end NUMINAMATH_GPT_final_price_is_99_l1377_137746


namespace NUMINAMATH_GPT_total_weight_of_10_moles_CaH2_is_420_96_l1377_137734

def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_H : ℝ := 1.008
def molecular_weight_CaH2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_H
def moles_CaH2 : ℝ := 10
def total_weight_CaH2 : ℝ := molecular_weight_CaH2 * moles_CaH2

theorem total_weight_of_10_moles_CaH2_is_420_96 :
  total_weight_CaH2 = 420.96 :=
by
  sorry

end NUMINAMATH_GPT_total_weight_of_10_moles_CaH2_is_420_96_l1377_137734


namespace NUMINAMATH_GPT_average_marks_of_all_candidates_l1377_137707

def n : ℕ := 120
def p : ℕ := 100
def f : ℕ := n - p
def A_p : ℕ := 39
def A_f : ℕ := 15
def total_marks : ℕ := p * A_p + f * A_f
def average_marks : ℚ := total_marks / n

theorem average_marks_of_all_candidates :
  average_marks = 35 := 
sorry

end NUMINAMATH_GPT_average_marks_of_all_candidates_l1377_137707


namespace NUMINAMATH_GPT_number_of_cuboids_painted_l1377_137739

-- Define the problem conditions
def painted_faces (total_faces : ℕ) (faces_per_cuboid : ℕ) : ℕ :=
  total_faces / faces_per_cuboid

-- Define the theorem to prove
theorem number_of_cuboids_painted (total_faces : ℕ) (faces_per_cuboid : ℕ) :
  total_faces = 48 → faces_per_cuboid = 6 → painted_faces total_faces faces_per_cuboid = 8 :=
by
  intros h1 h2
  rw [h1, h2]
  exact rfl

end NUMINAMATH_GPT_number_of_cuboids_painted_l1377_137739


namespace NUMINAMATH_GPT_perimeter_of_smaller_rectangle_l1377_137706

theorem perimeter_of_smaller_rectangle (s t u : ℝ) (h1 : 4 * s = 160) (h2 : t = s / 2) (h3 : u = t / 3) : 
    2 * (t + u) = 400 / 3 := by
  sorry

end NUMINAMATH_GPT_perimeter_of_smaller_rectangle_l1377_137706


namespace NUMINAMATH_GPT_initial_discount_percentage_l1377_137750

variable (d : ℝ) (x : ℝ)
variable (h1 : 0 < d) (h2 : 0 ≤ x) (h3 : x ≤ 100)
variable (h4 : (1 - x / 100) * 0.6 * d = 0.33 * d)

theorem initial_discount_percentage : x = 45 :=
by
  sorry

end NUMINAMATH_GPT_initial_discount_percentage_l1377_137750


namespace NUMINAMATH_GPT_calculate_revolutions_l1377_137769

def wheel_diameter : ℝ := 8
def distance_traveled_miles : ℝ := 0.5
def feet_per_mile : ℝ := 5280
def distance_traveled_feet : ℝ := distance_traveled_miles * feet_per_mile

theorem calculate_revolutions :
  let radius : ℝ := wheel_diameter / 2
  let circumference : ℝ := 2 * Real.pi * radius
  let revolutions : ℝ := distance_traveled_feet / circumference
  revolutions = 330 / Real.pi := by
  sorry

end NUMINAMATH_GPT_calculate_revolutions_l1377_137769


namespace NUMINAMATH_GPT_trig_expression_equality_l1377_137738

theorem trig_expression_equality :
  (Real.tan (60 * Real.pi / 180) + 2 * Real.sin (45 * Real.pi / 180) - 2 * Real.cos (30 * Real.pi / 180)) 
  = Real.sqrt 2 :=
by
  have h1 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := by sorry
  have h2 : Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := by sorry
  sorry

end NUMINAMATH_GPT_trig_expression_equality_l1377_137738


namespace NUMINAMATH_GPT_tangent_line_computation_l1377_137786

variables (f : ℝ → ℝ)

theorem tangent_line_computation (h_tangent : ∀ x, (f x = -x + 8) ∧ (∃ y, y = -x + 8 → (f y) = -x + 8 → deriv f x = -1)) :
    f 5 + deriv f 5 = 2 :=
sorry

end NUMINAMATH_GPT_tangent_line_computation_l1377_137786


namespace NUMINAMATH_GPT_function_odd_and_decreasing_l1377_137753

noncomputable def f (a x : ℝ) : ℝ := (1 / a) ^ x - a ^ x

theorem function_odd_and_decreasing (a : ℝ) (h : a > 1) :
  (∀ x, f a (-x) = -f a x) ∧ (∀ x y, x < y → f a x > f a y) :=
by
  sorry

end NUMINAMATH_GPT_function_odd_and_decreasing_l1377_137753


namespace NUMINAMATH_GPT_local_minimum_value_of_f_l1377_137795

noncomputable def f (x : ℝ) : ℝ := x - Real.log x

theorem local_minimum_value_of_f : 
  ∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, y > 0 → f y ≥ f x) ∧ f x = 1 :=
by
  sorry

end NUMINAMATH_GPT_local_minimum_value_of_f_l1377_137795


namespace NUMINAMATH_GPT_find_values_l1377_137735

theorem find_values (a b c : ℤ)
  (h1 : ∀ x, x^2 + 9 * x + 14 = (x + a) * (x + b))
  (h2 : ∀ x, x^2 + 4 * x - 21 = (x + b) * (x - c)) :
  a + b + c = 12 :=
sorry

end NUMINAMATH_GPT_find_values_l1377_137735


namespace NUMINAMATH_GPT_M_is_infinite_l1377_137799

variable (M : Set ℝ)

def has_properties (M : Set ℝ) : Prop :=
  (∃ x y : ℝ, x ∈ M ∧ y ∈ M ∧ x ≠ y) ∧ ∀ x ∈ M, (3*x - 2 ∈ M ∨ -4*x + 5 ∈ M)

theorem M_is_infinite (M : Set ℝ) (h : has_properties M) : ¬Finite M := by
  sorry

end NUMINAMATH_GPT_M_is_infinite_l1377_137799


namespace NUMINAMATH_GPT_move_digit_to_make_equation_correct_l1377_137772

theorem move_digit_to_make_equation_correct :
  101 - 102 ≠ 1 → (101 - 10^2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_move_digit_to_make_equation_correct_l1377_137772


namespace NUMINAMATH_GPT_sand_exchange_impossible_l1377_137760

/-- Given initial conditions for g and p, the goal is to determine if 
the banker can have at least 2 kg of each type of sand in the end. -/
theorem sand_exchange_impossible (g p : ℕ) (G P : ℕ) 
  (initial_g : g = 1001) (initial_p : p = 1001) 
  (initial_G : G = 1) (initial_P : P = 1)
  (exchange_rule : ∀ x y : ℚ, x * p = y * g) 
  (decrement_rule : ∀ k, 1 ≤ k ∧ k ≤ 2000 → 
    (g = 1001 - k ∨ p = 1001 - k)) :
  ¬(G ≥ 2 ∧ P ≥ 2) :=
by
  -- Add a placeholder to skip the proof
  sorry

end NUMINAMATH_GPT_sand_exchange_impossible_l1377_137760


namespace NUMINAMATH_GPT_central_angle_of_sector_l1377_137776

noncomputable def sector_area (α r : ℝ) : ℝ := (1/2) * α * r^2

theorem central_angle_of_sector :
  sector_area 3 2 = 6 :=
by
  unfold sector_area
  norm_num
  done

end NUMINAMATH_GPT_central_angle_of_sector_l1377_137776


namespace NUMINAMATH_GPT_find_prices_min_cost_l1377_137762

-- Definitions based on conditions
def price_difference (x y : ℕ) : Prop := x - y = 50
def total_cost (x y : ℕ) : Prop := 2 * x + 3 * y = 250
def cost_function (a : ℕ) : ℕ := 50 * a + 6000
def min_items (a : ℕ) : Prop := a ≥ 80
def total_items : ℕ := 200

-- Lean 4 statements for the proof problem
theorem find_prices (x y : ℕ) (h1 : price_difference x y) (h2 : total_cost x y) :
  (x = 80) ∧ (y = 30) :=
sorry

theorem min_cost (a : ℕ) (h1 : min_items a) :
  cost_function a ≥ 10000 :=
sorry

#check find_prices
#check min_cost

end NUMINAMATH_GPT_find_prices_min_cost_l1377_137762


namespace NUMINAMATH_GPT_probability_sin_in_interval_half_l1377_137766

noncomputable def probability_sin_interval : ℝ :=
  let a := - (Real.pi / 2)
  let b := Real.pi / 2
  let interval_length := b - a
  (b - 0) / interval_length

theorem probability_sin_in_interval_half :
  probability_sin_interval = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_probability_sin_in_interval_half_l1377_137766


namespace NUMINAMATH_GPT_road_repair_completion_time_l1377_137722

theorem road_repair_completion_time :
  (∀ (r : ℝ), 1 = 45 * r * 3) → (∀ (t : ℝ), (30 * (1 / (3 * 45))) * t = 1) → t = 4.5 :=
by
  intros rate_eq time_eq
  sorry

end NUMINAMATH_GPT_road_repair_completion_time_l1377_137722


namespace NUMINAMATH_GPT_quadratic_inequality_l1377_137773

theorem quadratic_inequality (a : ℝ) :
  (¬ (∃ x : ℝ, a * x^2 + 2 * x + 3 ≤ 0)) ↔ (a > 1 / 3) :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_inequality_l1377_137773


namespace NUMINAMATH_GPT_spherical_to_rectangular_coordinates_l1377_137705

theorem spherical_to_rectangular_coordinates :
  ∀ (ρ θ φ : ℝ), ρ = 6 → θ = 7 * Real.pi / 4 → φ = Real.pi / 4 →
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z) = (3, -3, 3 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_GPT_spherical_to_rectangular_coordinates_l1377_137705


namespace NUMINAMATH_GPT_sum_of_coefficients_of_poly_is_neg_1_l1377_137743

noncomputable def evaluate_poly_sum (α β : ℂ) (h1 : α + β = 1) (h2 : α * β = 1) : ℂ :=
  α^2005 + β^2005

theorem sum_of_coefficients_of_poly_is_neg_1 (α β : ℂ) (h1 : α + β = 1) (h2 : α * β = 1) :
  evaluate_poly_sum α β h1 h2 = -1 := by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_of_poly_is_neg_1_l1377_137743


namespace NUMINAMATH_GPT_find_value_l1377_137721

theorem find_value : (1 / 4 * (5 * 9 * 4) - 7) = 38 := 
by
  sorry

end NUMINAMATH_GPT_find_value_l1377_137721


namespace NUMINAMATH_GPT_solution_is_111_l1377_137791

-- Define the system of equations
def system_of_equations (x y z : ℝ) :=
  (x^2 + 7 * y + 2 = 2 * z + 4 * Real.sqrt (7 * x - 3)) ∧
  (y^2 + 7 * z + 2 = 2 * x + 4 * Real.sqrt (7 * y - 3)) ∧
  (z^2 + 7 * x + 2 = 2 * y + 4 * Real.sqrt (7 * z - 3))

-- Prove that x = 1, y = 1, z = 1 is a solution to the system of equations
theorem solution_is_111 : system_of_equations 1 1 1 :=
by
  sorry

end NUMINAMATH_GPT_solution_is_111_l1377_137791


namespace NUMINAMATH_GPT_trig_identity_sin_eq_l1377_137740

theorem trig_identity_sin_eq (α : ℝ) (h : Real.cos (π / 6 - α) = 1 / 3) : 
  Real.sin (5 * π / 6 - 2 * α) = -7 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_trig_identity_sin_eq_l1377_137740


namespace NUMINAMATH_GPT_bigger_part_of_sum_54_l1377_137763

theorem bigger_part_of_sum_54 (x y : ℕ) (h₁ : x + y = 54) (h₂ : 10 * x + 22 * y = 780) : x = 34 :=
sorry

end NUMINAMATH_GPT_bigger_part_of_sum_54_l1377_137763


namespace NUMINAMATH_GPT_angles_of_tangency_triangle_l1377_137726

theorem angles_of_tangency_triangle 
  (A B C : ℝ) 
  (ha : A = 40)
  (hb : B = 80)
  (hc : C = 180 - A - B)
  (a1 b1 c1 : ℝ)
  (ha1 : a1 = (1/2) * (180 - A))
  (hb1 : b1 = (1/2) * (180 - B))
  (hc1 : c1 = 180 - a1 - b1) :
  (a1 = 70 ∧ b1 = 50 ∧ c1 = 60) :=
by sorry

end NUMINAMATH_GPT_angles_of_tangency_triangle_l1377_137726


namespace NUMINAMATH_GPT_conformal_2z_conformal_z_minus_2_squared_l1377_137745

-- For the function w = 2z
theorem conformal_2z :
  ∀ z : ℂ, true :=
by
  intro z
  sorry

-- For the function w = (z-2)^2
theorem conformal_z_minus_2_squared :
  ∀ z : ℂ, z ≠ 2 → true :=
by
  intro z h
  sorry

end NUMINAMATH_GPT_conformal_2z_conformal_z_minus_2_squared_l1377_137745


namespace NUMINAMATH_GPT_junk_mail_each_house_l1377_137724

def blocks : ℕ := 16
def houses_per_block : ℕ := 17
def total_junk_mail : ℕ := 1088
def total_houses : ℕ := blocks * houses_per_block
def junk_mail_per_house : ℕ := total_junk_mail / total_houses

theorem junk_mail_each_house :
  junk_mail_per_house = 4 :=
by
  sorry

end NUMINAMATH_GPT_junk_mail_each_house_l1377_137724


namespace NUMINAMATH_GPT_rational_coefficient_exists_in_binomial_expansion_l1377_137749

theorem rational_coefficient_exists_in_binomial_expansion :
  ∃! (n : ℕ), n > 0 ∧ (∀ r, (r % 3 = 0 → (n - r) % 2 = 0 → n = 7)) :=
by
  sorry

end NUMINAMATH_GPT_rational_coefficient_exists_in_binomial_expansion_l1377_137749


namespace NUMINAMATH_GPT_olivia_remaining_usd_l1377_137720

def initial_usd : ℝ := 78
def initial_eur : ℝ := 50
def exchange_rate : ℝ := 1.20
def spent_usd_supermarket : ℝ := 15
def book_eur : ℝ := 10
def spent_usd_lunch : ℝ := 12

theorem olivia_remaining_usd :
  let total_usd := initial_usd + (initial_eur * exchange_rate)
  let remaining_after_supermarket := total_usd - spent_usd_supermarket
  let remaining_after_book := remaining_after_supermarket - (book_eur * exchange_rate)
  let final_remaining := remaining_after_book - spent_usd_lunch
  final_remaining = 99 :=
by
  sorry

end NUMINAMATH_GPT_olivia_remaining_usd_l1377_137720


namespace NUMINAMATH_GPT_no_positive_x_for_volume_l1377_137775

noncomputable def volume (x : ℤ) : ℤ :=
  (x + 5) * (x - 7) * (x^2 + x + 30)

theorem no_positive_x_for_volume : ¬ ∃ x : ℕ, 0 < x ∧ volume x < 800 := by
  sorry

end NUMINAMATH_GPT_no_positive_x_for_volume_l1377_137775


namespace NUMINAMATH_GPT_Zack_traveled_18_countries_l1377_137737

variables (countries_Alex countries_George countries_Joseph countries_Patrick countries_Zack : ℕ)
variables (h1 : countries_Alex = 24)
variables (h2 : countries_George = countries_Alex / 4)
variables (h3 : countries_Joseph = countries_George / 2)
variables (h4 : countries_Patrick = 3 * countries_Joseph)
variables (h5 : countries_Zack = 2 * countries_Patrick)

theorem Zack_traveled_18_countries :
  countries_Zack = 18 :=
by sorry

end NUMINAMATH_GPT_Zack_traveled_18_countries_l1377_137737


namespace NUMINAMATH_GPT_solve_for_x_l1377_137796

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2 * x - 25) : x = -20 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1377_137796


namespace NUMINAMATH_GPT_corn_bag_price_l1377_137732

theorem corn_bag_price
  (cost_seeds: ℕ)
  (cost_fertilizers_pesticides: ℕ)
  (cost_labor: ℕ)
  (total_bags: ℕ)
  (desired_profit_percentage: ℕ)
  (total_cost: ℕ := cost_seeds + cost_fertilizers_pesticides + cost_labor)
  (total_revenue: ℕ := total_cost + (total_cost * desired_profit_percentage / 100))
  (price_per_bag: ℕ := total_revenue / total_bags) :
  cost_seeds = 50 →
  cost_fertilizers_pesticides = 35 →
  cost_labor = 15 →
  total_bags = 10 →
  desired_profit_percentage = 10 →
  price_per_bag = 11 :=
by sorry

end NUMINAMATH_GPT_corn_bag_price_l1377_137732


namespace NUMINAMATH_GPT_no_common_points_l1377_137715

def curve1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def curve2 (x y : ℝ) : Prop := x^2 + 2*y^2 = 2

theorem no_common_points :
  ¬ ∃ (x y : ℝ), curve1 x y ∧ curve2 x y :=
by sorry

end NUMINAMATH_GPT_no_common_points_l1377_137715


namespace NUMINAMATH_GPT_part1_part2_l1377_137765

-- Part 1
theorem part1 (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  |(3 * x - 4 * x^3)| ≤ 1 := sorry

-- Part 2
theorem part2 (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  |(3 * x - 4 * x^3)| ≤ 1 := sorry

end NUMINAMATH_GPT_part1_part2_l1377_137765


namespace NUMINAMATH_GPT_max_length_PQ_l1377_137703

-- Define the curve in polar coordinates
def curve (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

-- Definition of points P and Q lying on the curve
def point_on_curve (ρ θ : ℝ) (P : ℝ × ℝ) : Prop :=
  curve ρ θ ∧ P = (ρ * Real.cos θ, ρ * Real.sin θ)

def points_on_curve (P Q : ℝ × ℝ) : Prop :=
  ∃ θ₁ θ₂ ρ₁ ρ₂, point_on_curve ρ₁ θ₁ P ∧ point_on_curve ρ₂ θ₂ Q

-- The theorem stating the maximum length of PQ
theorem max_length_PQ {P Q : ℝ × ℝ} (h : points_on_curve P Q) : dist P Q ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_length_PQ_l1377_137703


namespace NUMINAMATH_GPT_student_weighted_avg_larger_l1377_137787

variable {u v w : ℚ}

theorem student_weighted_avg_larger (h1 : u < v) (h2 : v < w) :
  (4 * u + 6 * v + 20 * w) / 30 > (2 * u + 3 * v + 4 * w) / 9 := by
  sorry

end NUMINAMATH_GPT_student_weighted_avg_larger_l1377_137787


namespace NUMINAMATH_GPT_earphone_cost_l1377_137758

/-- 
The cost of the earphone purchased on Friday can be calculated given:
1. The mean expenditure over 7 days is 500.
2. The expenditures for Monday, Tuesday, Wednesday, Thursday, Saturday, and Sunday are 450, 600, 400, 500, 550, and 300, respectively.
3. On Friday, the expenditures include a pen costing 30 and a notebook costing 50.
-/
theorem earphone_cost
  (mean_expenditure : ℕ)
  (mon tue wed thur sat sun : ℕ)
  (pen_cost notebook_cost : ℕ)
  (mean_expenditure_eq : mean_expenditure = 500)
  (mon_eq : mon = 450)
  (tue_eq : tue = 600)
  (wed_eq : wed = 400)
  (thur_eq : thur = 500)
  (sat_eq : sat = 550)
  (sun_eq : sun = 300)
  (pen_cost_eq : pen_cost = 30)
  (notebook_cost_eq : notebook_cost = 50)
  : ∃ (earphone_cost : ℕ), earphone_cost = 620 := 
by
  sorry

end NUMINAMATH_GPT_earphone_cost_l1377_137758


namespace NUMINAMATH_GPT_find_g_plus_h_l1377_137752

theorem find_g_plus_h (g h : ℚ) (d : ℚ) 
  (h_prod : (7 * d^2 - 4 * d + g) * (3 * d^2 + h * d - 9) = 21 * d^4 - 49 * d^3 - 44 * d^2 + 17 * d - 24) :
  g + h = -107 / 24 :=
sorry

end NUMINAMATH_GPT_find_g_plus_h_l1377_137752


namespace NUMINAMATH_GPT_weight_loss_total_l1377_137717

theorem weight_loss_total :
  ∀ (weight1 weight2 weight3 weight4 : ℕ),
    weight1 = 27 →
    weight2 = weight1 - 7 →
    weight3 = 28 →
    weight4 = 28 →
    weight1 + weight2 + weight3 + weight4 = 103 :=
by
  intros weight1 weight2 weight3 weight4 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end NUMINAMATH_GPT_weight_loss_total_l1377_137717


namespace NUMINAMATH_GPT_digit_d_multiple_of_9_l1377_137710

theorem digit_d_multiple_of_9 (d : ℕ) (hd : d = 1) : ∃ k : ℕ, (56780 + d) = 9 * k := by
  have : 56780 + d = 56780 + 1 := by rw [hd]
  rw [this]
  use 6313
  sorry

end NUMINAMATH_GPT_digit_d_multiple_of_9_l1377_137710


namespace NUMINAMATH_GPT_problem1_problem2_l1377_137708

-- Problem 1: Prove that the solutions of x^2 + 6x - 7 = 0 are x = -7 and x = 1
theorem problem1 (x : ℝ) : x^2 + 6*x - 7 = 0 ↔ (x = -7 ∨ x = 1) := by
  -- Proof omitted
  sorry

-- Problem 2: Prove that the solutions of 4x(2x+1) = 3(2x+1) are x = -1/2 and x = 3/4
theorem problem2 (x : ℝ) : 4*x*(2*x + 1) = 3*(2*x + 1) ↔ (x = -1/2 ∨ x = 3/4) := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1377_137708


namespace NUMINAMATH_GPT_number_of_flags_l1377_137780

theorem number_of_flags (colors : Finset ℕ) (stripes : ℕ) (h_colors : colors.card = 3) (h_stripes : stripes = 3) : 
  (colors.card ^ stripes) = 27 := 
by
  sorry

end NUMINAMATH_GPT_number_of_flags_l1377_137780


namespace NUMINAMATH_GPT_tangent_line_of_f_eq_kx_l1377_137785

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x
def tangent_line (k : ℝ) (x : ℝ) : ℝ := k * x

theorem tangent_line_of_f_eq_kx (k : ℝ) : 
    (∃ x₀, tangent_line k x₀ = f x₀ ∧ deriv f x₀ = k) → 
    (k = 0 ∨ k = 1 ∨ k = -1) := 
  sorry

end NUMINAMATH_GPT_tangent_line_of_f_eq_kx_l1377_137785


namespace NUMINAMATH_GPT_farmer_feed_total_cost_l1377_137723

/-- 
A farmer spent $35 on feed for chickens and goats. He spent 40% of the money on chicken feed, which he bought at a 50% discount off the full price, and spent the rest on goat feed, which he bought at full price. Prove that if the farmer had paid full price for both the chicken feed and the goat feed, he would have spent $49.
-/
theorem farmer_feed_total_cost
  (total_spent : ℝ := 35)
  (chicken_feed_fraction : ℝ := 0.40)
  (goat_feed_fraction : ℝ := 0.60)
  (discount : ℝ := 0.50)
  (chicken_feed_discounted : ℝ := chicken_feed_fraction * total_spent)
  (chicken_feed_full_price : ℝ := chicken_feed_discounted / (1 - discount))
  (goat_feed_full_price : ℝ := goat_feed_fraction * total_spent):
  chicken_feed_full_price + goat_feed_full_price = 49 := 
sorry

end NUMINAMATH_GPT_farmer_feed_total_cost_l1377_137723


namespace NUMINAMATH_GPT_club_additional_members_l1377_137759

theorem club_additional_members (current_members : ℕ) (additional_members : ℕ) :
  current_members = 10 →
  additional_members = (2 * current_members) + 5 - current_members →
  additional_members = 15 :=
by
  intro h1 h2
  rw [h1] at h2
  norm_num at h2
  exact h2

end NUMINAMATH_GPT_club_additional_members_l1377_137759


namespace NUMINAMATH_GPT_problem_1_simplification_l1377_137719

theorem problem_1_simplification (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 2) : 
  (x - 2) / (x ^ 2) / (1 - 2 / x) = 1 / x := 
  sorry

end NUMINAMATH_GPT_problem_1_simplification_l1377_137719


namespace NUMINAMATH_GPT_find_x_l1377_137718

theorem find_x (x : ℝ) (h : (x^2 - x - 6) / (x + 1) = (x^2 - 2*x - 3) * (0 : ℂ).im) : x = 3 :=
sorry

end NUMINAMATH_GPT_find_x_l1377_137718


namespace NUMINAMATH_GPT_total_weight_is_correct_l1377_137754

-- Define the variables
def envelope_weight : ℝ := 8.5
def additional_weight_per_envelope : ℝ := 2
def num_envelopes : ℝ := 880

-- Define the total weight calculation
def total_weight : ℝ := num_envelopes * (envelope_weight + additional_weight_per_envelope)

-- State the theorem to prove that the total weight is as expected
theorem total_weight_is_correct : total_weight = 9240 :=
by
  sorry

end NUMINAMATH_GPT_total_weight_is_correct_l1377_137754


namespace NUMINAMATH_GPT_abc_equal_l1377_137700

theorem abc_equal (a b c : ℝ)
  (h1 : ∀ x : ℝ, a * x^2 + b * x + c ≥ b * x^2 + c * x + a)
  (h2 : ∀ x : ℝ, b * x^2 + c * x + a ≥ c * x^2 + a * x + b) : a = b ∧ b = c :=
by
  sorry

end NUMINAMATH_GPT_abc_equal_l1377_137700


namespace NUMINAMATH_GPT_min_value_fraction_l1377_137774

theorem min_value_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : -3 ≤ y ∧ y ≤ 1) : (x + y) / x = 0.8 :=
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_l1377_137774


namespace NUMINAMATH_GPT_kimberly_loan_l1377_137781

theorem kimberly_loan :
  ∃ (t : ℕ), (1.06 : ℝ)^t > 3 ∧ ∀ (t' : ℕ), t' < t → (1.06 : ℝ)^t' ≤ 3 :=
by
sorry

end NUMINAMATH_GPT_kimberly_loan_l1377_137781


namespace NUMINAMATH_GPT_Luke_spent_per_week_l1377_137730

-- Definitions based on the conditions
def money_from_mowing := 9
def money_from_weeding := 18
def total_money := money_from_mowing + money_from_weeding
def weeks := 9
def amount_spent_per_week := total_money / weeks

-- The proof statement
theorem Luke_spent_per_week :
  amount_spent_per_week = 3 := 
  sorry

end NUMINAMATH_GPT_Luke_spent_per_week_l1377_137730


namespace NUMINAMATH_GPT_soccer_ball_selling_price_l1377_137727

theorem soccer_ball_selling_price
  (cost_price_per_ball : ℕ)
  (num_balls : ℕ)
  (total_profit : ℕ)
  (h_cost_price : cost_price_per_ball = 60)
  (h_num_balls : num_balls = 50)
  (h_total_profit : total_profit = 1950) :
  (cost_price_per_ball + (total_profit / num_balls) = 99) :=
by 
  -- Note: Proof can be filled here
  sorry

end NUMINAMATH_GPT_soccer_ball_selling_price_l1377_137727


namespace NUMINAMATH_GPT_log_one_fifth_25_eq_neg2_l1377_137788

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_one_fifth_25_eq_neg2 :
  log_base (1 / 5) 25 = -2 := by
 sorry

end NUMINAMATH_GPT_log_one_fifth_25_eq_neg2_l1377_137788


namespace NUMINAMATH_GPT_prime_list_count_l1377_137764

theorem prime_list_count {L : ℕ → ℕ} 
  (hL₀ : L 0 = 29)
  (hL : ∀ (n : ℕ), L (n + 1) = L n * 101 + L 0) :
  (∃! n, n = 0 ∧ Prime (L n)) ∧ ∀ m > 0, ¬ Prime (L m) := 
by
  sorry

end NUMINAMATH_GPT_prime_list_count_l1377_137764


namespace NUMINAMATH_GPT_cross_section_area_l1377_137751

open Real

theorem cross_section_area (b α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  ∃ (area : ℝ), area = - (b^2 * cos α * tan β) / (2 * cos (3 * α)) :=
by
  sorry

end NUMINAMATH_GPT_cross_section_area_l1377_137751


namespace NUMINAMATH_GPT_calculate_value_l1377_137770

theorem calculate_value :
  ( (3^3 - 1) / (3^3 + 1) ) * ( (4^3 - 1) / (4^3 + 1) ) * ( (5^3 - 1) / (5^3 + 1) ) * ( (6^3 - 1) / (6^3 + 1) ) * ( (7^3 - 1) / (7^3 + 1) )
  = 57 / 84 := by
  sorry

end NUMINAMATH_GPT_calculate_value_l1377_137770


namespace NUMINAMATH_GPT_units_digit_sum_of_factorials_50_l1377_137748

def units_digit (n : Nat) : Nat :=
  n % 10

def sum_of_factorials (n : Nat) : Nat :=
  (List.range' 1 n).map Nat.factorial |>.sum

theorem units_digit_sum_of_factorials_50 :
  units_digit (sum_of_factorials 51) = 3 := 
sorry

end NUMINAMATH_GPT_units_digit_sum_of_factorials_50_l1377_137748


namespace NUMINAMATH_GPT_part1_part2_l1377_137783

def f (x a : ℝ) : ℝ := |x + 1| - |x - a|

theorem part1 (x : ℝ) : (f x 2 > 2) ↔ (x > 3 / 2) :=
sorry

theorem part2 (a : ℝ) (ha : a > 0) : (∀ x, f x a < 2 * a) ↔ (1 < a) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1377_137783


namespace NUMINAMATH_GPT_ratio_of_numbers_l1377_137777

theorem ratio_of_numbers (x : ℝ) (h_sum : x + 3.5 = 14) : x / 3.5 = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_numbers_l1377_137777


namespace NUMINAMATH_GPT_combine_like_terms_1_simplify_expression_2_l1377_137702

-- Problem 1
theorem combine_like_terms_1 (m n : ℝ) :
  2 * m^2 * n - 3 * m * n + 8 - 3 * m^2 * n + 5 * m * n - 3 = -m^2 * n + 2 * m * n + 5 :=
by 
  -- Proof goes here 
  sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) :
  2 * (2 * a - 3 * b) - 3 * (2 * b - 3 * a) = 13 * a - 12 * b :=
by 
  -- Proof goes here 
  sorry

end NUMINAMATH_GPT_combine_like_terms_1_simplify_expression_2_l1377_137702


namespace NUMINAMATH_GPT_num_small_boxes_l1377_137768

-- Conditions
def chocolates_per_small_box := 25
def total_chocolates := 400

-- Claim: Prove that the number of small boxes is 16
theorem num_small_boxes : (total_chocolates / chocolates_per_small_box) = 16 := 
by sorry

end NUMINAMATH_GPT_num_small_boxes_l1377_137768


namespace NUMINAMATH_GPT_remainder_is_v_l1377_137704

theorem remainder_is_v (x y u v : ℤ) (hx : x > 0) (hy : y > 0)
  (hdiv : x = u * y + v) (hv_range : 0 ≤ v ∧ v < y) :
  (x + (2 * u + 1) * y) % y = v :=
by
  sorry

end NUMINAMATH_GPT_remainder_is_v_l1377_137704


namespace NUMINAMATH_GPT_expenditure_increase_l1377_137797

theorem expenditure_increase
  (current_expenditure : ℝ)
  (future_expenditure : ℝ)
  (years : ℕ)
  (r : ℝ)
  (h₁ : current_expenditure = 1000)
  (h₂ : future_expenditure = 2197)
  (h₃ : years = 3)
  (h₄ : future_expenditure = current_expenditure * (1 + r / 100) ^ years) :
  r = 30 :=
sorry

end NUMINAMATH_GPT_expenditure_increase_l1377_137797


namespace NUMINAMATH_GPT_complex_number_quadrant_l1377_137747

theorem complex_number_quadrant (a : ℝ) : 
  (a^2 - 2 = 3 * a - 4) ∧ (a^2 - 2 < 0 ∧ 3 * a - 4 < 0) → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_complex_number_quadrant_l1377_137747


namespace NUMINAMATH_GPT_cos_C_value_l1377_137729

-- Definitions for the perimeter and sine ratios
variable (A B C : ℝ)
variable (a b c : ℝ)
variable (perimeter : ℝ) (sin_ratio_A sin_ratio_B sin_ratio_C : ℚ)

-- Given conditions
axiom perimeter_condition : perimeter = a + b + c
axiom sine_ratio_condition : (sin_ratio_A / sin_ratio_B / sin_ratio_C) = (3 / 2 / 4)
axiom side_lengths : a = 3 ∧ b = 2 ∧ c = 4

-- To prove

theorem cos_C_value (h1 : sine_ratio_A = 3) (h2 : sine_ratio_B = 2) (h3 : sin_ratio_C = 4) :
  (3^2 + 2^2 - 4^2) / (2 * 3 * 2) = -1 / 4 :=
sorry

end NUMINAMATH_GPT_cos_C_value_l1377_137729


namespace NUMINAMATH_GPT_remainder_17_pow_1499_mod_23_l1377_137742

theorem remainder_17_pow_1499_mod_23 : (17 ^ 1499) % 23 = 11 :=
by
  sorry

end NUMINAMATH_GPT_remainder_17_pow_1499_mod_23_l1377_137742


namespace NUMINAMATH_GPT_exists_distinct_numbers_divisible_by_3_l1377_137728

-- Define the problem in Lean with the given conditions and goal.
theorem exists_distinct_numbers_divisible_by_3 : 
  ∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a % 3 = 0 ∧ b % 3 = 0 ∧ c % 3 = 0 ∧ d % 3 = 0 ∧
  (a + b + c) % d = 0 ∧ (a + b + d) % c = 0 ∧ (a + c + d) % b = 0 ∧ (b + c + d) % a = 0 :=
by
  sorry

end NUMINAMATH_GPT_exists_distinct_numbers_divisible_by_3_l1377_137728


namespace NUMINAMATH_GPT_tan_alpha_minus_pi_over_4_eq_neg_3_over_4_l1377_137782

open Real

theorem tan_alpha_minus_pi_over_4_eq_neg_3_over_4 (α β : ℝ) 
  (h1 : tan (α + β) = 1 / 2) 
  (h2 : tan β = 1 / 3) : 
  tan (α - π / 4) = -3 / 4 :=
sorry

end NUMINAMATH_GPT_tan_alpha_minus_pi_over_4_eq_neg_3_over_4_l1377_137782


namespace NUMINAMATH_GPT_total_votes_l1377_137756

-- Conditions
variables (V : ℝ)
def candidate_votes := 0.31 * V
def rival_votes := 0.31 * V + 2451

-- Problem statement
theorem total_votes (h : candidate_votes V + rival_votes V = V) : V = 6450 :=
sorry

end NUMINAMATH_GPT_total_votes_l1377_137756


namespace NUMINAMATH_GPT_second_option_cost_per_day_l1377_137790

theorem second_option_cost_per_day :
  let distance_one_way := 150
  let rental_first_option := 50
  let kilometers_per_liter := 15
  let cost_per_liter := 0.9
  let savings := 22
  let total_distance := distance_one_way * 2
  let total_liters := total_distance / kilometers_per_liter
  let gasoline_cost := total_liters * cost_per_liter
  let total_cost_first_option := rental_first_option + gasoline_cost
  let second_option_cost := total_cost_first_option + savings
  second_option_cost = 90 :=
by
  sorry

end NUMINAMATH_GPT_second_option_cost_per_day_l1377_137790


namespace NUMINAMATH_GPT_inequality_proof_l1377_137778

theorem inequality_proof 
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c) + b / Real.sqrt (b^2 + 8 * a * c) + c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
  sorry

end NUMINAMATH_GPT_inequality_proof_l1377_137778


namespace NUMINAMATH_GPT_oranges_left_uneaten_l1377_137712

variable (total_oranges : ℕ)
variable (half_oranges ripe_oranges unripe_oranges eaten_ripe_oranges eaten_unripe_oranges uneaten_ripe_oranges uneaten_unripe_oranges total_uneaten_oranges : ℕ)

axiom h1 : total_oranges = 96
axiom h2 : half_oranges = total_oranges / 2
axiom h3 : ripe_oranges = half_oranges
axiom h4 : unripe_oranges = half_oranges
axiom h5 : eaten_ripe_oranges = ripe_oranges / 4
axiom h6 : eaten_unripe_oranges = unripe_oranges / 8
axiom h7 : uneaten_ripe_oranges = ripe_oranges - eaten_ripe_oranges
axiom h8 : uneaten_unripe_oranges = unripe_oranges - eaten_unripe_oranges
axiom h9 : total_uneaten_oranges = uneaten_ripe_oranges + uneaten_unripe_oranges

theorem oranges_left_uneaten : total_uneaten_oranges = 78 := by
  sorry

end NUMINAMATH_GPT_oranges_left_uneaten_l1377_137712


namespace NUMINAMATH_GPT_max_pens_given_budget_l1377_137789

-- Define the conditions.
def max_pens (x y : ℕ) := 12 * x + 20 * y

-- Define the main theorem stating the proof problem.
theorem max_pens_given_budget : ∃ (x y : ℕ), (10 * x + 15 * y ≤ 173) ∧ (max_pens x y = 224) :=
  sorry

end NUMINAMATH_GPT_max_pens_given_budget_l1377_137789
