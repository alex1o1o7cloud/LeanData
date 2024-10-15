import Mathlib

namespace NUMINAMATH_GPT_minimum_volume_sum_l64_6445

section pyramid_volume

variables {R : Type*} [OrderedRing R]
variables {V : Type*} [AddCommGroup V] [Module R V]

-- Define the volumes of the pyramids
variables (V_SABR1 V_SR2P2R3Q2 V_SCDR4 : R)
variables (V_SR1P1R2Q1 V_SR3P3R4Q3 : R)

-- Given condition
axiom volume_condition : V_SR1P1R2Q1 + V_SR3P3R4Q3 = 78

-- The theorem to be proved
theorem minimum_volume_sum : 
  V_SABR1^2 + V_SR2P2R3Q2^2 + V_SCDR4^2 ≥ 2028 :=
sorry

end pyramid_volume

end NUMINAMATH_GPT_minimum_volume_sum_l64_6445


namespace NUMINAMATH_GPT_ratio_of_frank_to_joystick_l64_6433

-- Define the costs involved
def cost_table : ℕ := 140
def cost_chair : ℕ := 100
def cost_joystick : ℕ := 20
def diff_spent : ℕ := 30

-- Define the payments
def F_j := 5
def E_j := 15

-- The ratio we need to prove
def ratio_frank_to_total_joystick (F_j : ℕ) (total_joystick : ℕ) : (ℕ × ℕ) :=
  (F_j / Nat.gcd F_j total_joystick, total_joystick / Nat.gcd F_j total_joystick)

theorem ratio_of_frank_to_joystick :
  let F_j := 5
  let total_joystick := 20
  ratio_frank_to_total_joystick F_j total_joystick = (1, 4) := by
  sorry

end NUMINAMATH_GPT_ratio_of_frank_to_joystick_l64_6433


namespace NUMINAMATH_GPT_non_working_games_count_l64_6472

-- Definitions based on conditions
def totalGames : ℕ := 16
def pricePerGame : ℕ := 7
def totalEarnings : ℕ := 56

-- Statement to prove
theorem non_working_games_count : 
  totalGames - (totalEarnings / pricePerGame) = 8 :=
by
  sorry

end NUMINAMATH_GPT_non_working_games_count_l64_6472


namespace NUMINAMATH_GPT_no_integers_satisfy_eq_l64_6414

theorem no_integers_satisfy_eq (m n : ℤ) : ¬ (m^2 + 1954 = n^2) := 
by
  sorry

end NUMINAMATH_GPT_no_integers_satisfy_eq_l64_6414


namespace NUMINAMATH_GPT_power_sum_inequality_l64_6426

theorem power_sum_inequality (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d :=
by sorry

end NUMINAMATH_GPT_power_sum_inequality_l64_6426


namespace NUMINAMATH_GPT_gcd_m_pow_5_plus_125_m_plus_3_l64_6471

theorem gcd_m_pow_5_plus_125_m_plus_3 (m : ℕ) (h: m > 16) : 
  Nat.gcd (m^5 + 125) (m + 3) = Nat.gcd 27 (m + 3) :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_gcd_m_pow_5_plus_125_m_plus_3_l64_6471


namespace NUMINAMATH_GPT_min_sum_xy_l64_6453

theorem min_sum_xy (x y : ℕ) (hx : x ≠ y) (hcond : ↑(1 / x) + ↑(1 / y) = 1 / 15) : x + y = 64 :=
sorry

end NUMINAMATH_GPT_min_sum_xy_l64_6453


namespace NUMINAMATH_GPT_Mrs_Lara_Late_l64_6459

noncomputable def required_speed (d t : ℝ) : ℝ := d / t

theorem Mrs_Lara_Late (d t : ℝ) (h1 : d = 50 * (t + 7 / 60)) (h2 : d = 70 * (t - 5 / 60)) :
  required_speed d t = 70 := by
  sorry

end NUMINAMATH_GPT_Mrs_Lara_Late_l64_6459


namespace NUMINAMATH_GPT_therese_older_than_aivo_l64_6416

-- Definitions based on given conditions
variables {Aivo Jolyn Leon Therese : ℝ}
variables (h1 : Jolyn = Therese + 2)
variables (h2 : Leon = Aivo + 2)
variables (h3 : Jolyn = Leon + 5)

-- Statement to prove
theorem therese_older_than_aivo :
  Therese = Aivo + 5 :=
by
  sorry

end NUMINAMATH_GPT_therese_older_than_aivo_l64_6416


namespace NUMINAMATH_GPT_solve_for_x_l64_6473

theorem solve_for_x (x y : ℝ) : 3 * x + 4 * y = 5 → x = (5 - 4 * y) / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l64_6473


namespace NUMINAMATH_GPT_min_selling_price_l64_6465

-- Average sales per month
def avg_sales := 50

-- Cost per refrigerator
def cost_per_fridge := 1200

-- Shipping fee per refrigerator
def shipping_fee_per_fridge := 20

-- Monthly storefront fee
def monthly_storefront_fee := 10000

-- Monthly repair costs
def monthly_repair_costs := 5000

-- Profit margin requirement
def profit_margin := 0.2

-- The minimum selling price for the shop to maintain at least 20% profit margin
theorem min_selling_price 
  (avg_sales : ℕ) 
  (cost_per_fridge : ℕ) 
  (shipping_fee_per_fridge : ℕ) 
  (monthly_storefront_fee : ℕ) 
  (monthly_repair_costs : ℕ) 
  (profit_margin : ℝ) : 
  ∃ x : ℝ, 
    (50 * x - ((cost_per_fridge + shipping_fee_per_fridge) * avg_sales + monthly_storefront_fee + monthly_repair_costs)) 
    ≥ (cost_per_fridge + shipping_fee_per_fridge) * avg_sales + monthly_storefront_fee + monthly_repair_costs * profit_margin 
    → x ≥ 1824 :=
by 
  sorry

end NUMINAMATH_GPT_min_selling_price_l64_6465


namespace NUMINAMATH_GPT_train_trip_length_l64_6403

theorem train_trip_length (v D : ℝ) :
  (3 + (3 * D - 6 * v) / (2 * v) = 4 + D / v) ∧ 
  (2.5 + 120 / v + (6 * D - 12 * v - 720) / (5 * v) = 3.5 + D / v) →
  (D = 420 ∨ D = 480 ∨ D = 540 ∨ D = 600 ∨ D = 660) :=
by
  sorry

end NUMINAMATH_GPT_train_trip_length_l64_6403


namespace NUMINAMATH_GPT_reducible_fraction_implies_divisibility_l64_6484

theorem reducible_fraction_implies_divisibility
  (a b c d l k : ℤ)
  (m n : ℤ)
  (h1 : a * l + b = k * m)
  (h2 : c * l + d = k * n)
  : k ∣ (a * d - b * c) :=
by
  sorry

end NUMINAMATH_GPT_reducible_fraction_implies_divisibility_l64_6484


namespace NUMINAMATH_GPT_average_of_scores_with_average_twice_l64_6406

variable (scores: List ℝ) (A: ℝ) (A': ℝ)
variable (h1: scores.length = 50)
variable (h2: A = (scores.sum) / 50)
variable (h3: A' = ((scores.sum + 2 * A) / 52))

theorem average_of_scores_with_average_twice (h1: scores.length = 50) (h2: A = (scores.sum) / 50) (h3: A' = ((scores.sum + 2 * A) / 52)) :
  A' = A :=
by
  sorry

end NUMINAMATH_GPT_average_of_scores_with_average_twice_l64_6406


namespace NUMINAMATH_GPT_handshakes_total_l64_6463

theorem handshakes_total :
  let team_size := 6
  let referees := 3
  (team_size * team_size) + (2 * team_size * referees) = 72 :=
by
  sorry

end NUMINAMATH_GPT_handshakes_total_l64_6463


namespace NUMINAMATH_GPT_find_function_l64_6427

theorem find_function (α : ℝ) (hα : 0 < α) (f : ℕ+ → ℝ) 
  (h : ∀ k m : ℕ+, α * m ≤ k → k ≤ (α + 1) * m → f (k + m) = f k + f m) :
  ∃ D : ℝ, ∀ n : ℕ+, f n = n * D :=
sorry

end NUMINAMATH_GPT_find_function_l64_6427


namespace NUMINAMATH_GPT_split_trout_equally_l64_6418

-- Definitions for conditions
def Total_trout : ℕ := 18
def People : ℕ := 2

-- Statement we need to prove
theorem split_trout_equally 
(H1 : Total_trout = 18)
(H2 : People = 2) : 
  (Total_trout / People = 9) :=
by
  sorry

end NUMINAMATH_GPT_split_trout_equally_l64_6418


namespace NUMINAMATH_GPT_find_value_l64_6449

theorem find_value 
  (a b c d e f : ℚ)
  (h1 : a / b = 1 / 2)
  (h2 : c / d = 1 / 2)
  (h3 : e / f = 1 / 2)
  (h4 : 3 * b - 2 * d + f ≠ 0) : 
  (3 * a - 2 * c + e) / (3 * b - 2 * d + f) = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_find_value_l64_6449


namespace NUMINAMATH_GPT_weights_of_first_two_cats_l64_6492

noncomputable def cats_weight_proof (W : ℝ) : Prop :=
  (∀ (w1 w2 : ℝ), w1 = W ∧ w2 = W ∧ (w1 + w2 + 14.7 + 9.3) / 4 = 12) → (W = 12)

theorem weights_of_first_two_cats (W : ℝ) :
  cats_weight_proof W :=
by
  sorry

end NUMINAMATH_GPT_weights_of_first_two_cats_l64_6492


namespace NUMINAMATH_GPT_find_n_l64_6485

theorem find_n (n k : ℕ) (b : ℝ) (h_n2 : n ≥ 2) (h_ab : b ≠ 0 ∧ k > 0) (h_a_eq : ∀ (a : ℝ), a = k^2 * b) :
  (∀ (S : ℕ → ℝ → ℝ), S 1 b + S 2 b = 0) →
  n = 2 * k + 1 := 
sorry

end NUMINAMATH_GPT_find_n_l64_6485


namespace NUMINAMATH_GPT_find_expression_for_x_l64_6437

variable (x : ℝ) (hx : x^3 + (1 / x^3) = -52)

theorem find_expression_for_x : x + (1 / x) = -4 :=
by sorry

end NUMINAMATH_GPT_find_expression_for_x_l64_6437


namespace NUMINAMATH_GPT_find_quadratic_function_l64_6452

def quadratic_function (c d : ℝ) (x : ℝ) : ℝ :=
  x^2 + c * x + d

theorem find_quadratic_function :
  ∃ c d, (∀ x, 
    (quadratic_function c d (quadratic_function c d x + 2 * x)) / (quadratic_function c d x) = 2 * x^2 + 1984 * x + 2024) ∧ 
    quadratic_function c d x = x^2 + 1982 * x + 21 :=
by
  sorry

end NUMINAMATH_GPT_find_quadratic_function_l64_6452


namespace NUMINAMATH_GPT_physics_marks_l64_6462

theorem physics_marks (P C M : ℕ) 
  (h1 : (P + C + M) = 255)
  (h2 : (P + M) = 180)
  (h3 : (P + C) = 140) : 
  P = 65 :=
by
  sorry

end NUMINAMATH_GPT_physics_marks_l64_6462


namespace NUMINAMATH_GPT_max_length_AB_l64_6499

theorem max_length_AB : ∀ t : ℝ, 0 ≤ t ∧ t ≤ 3 → ∃ M, M = 81 / 8 ∧ ∀ t, -2 * (t - 3/4)^2 + 81 / 8 = M :=
by sorry

end NUMINAMATH_GPT_max_length_AB_l64_6499


namespace NUMINAMATH_GPT_right_triangle_circle_area_l64_6455

/-- 
Given a right triangle ABC with legs AB = 6 cm and BC = 8 cm,
E is the midpoint of AB and D is the midpoint of AC.
A circle passes through points E and D and touches the hypotenuse AC.
Prove that the area of this circle is 100 * pi / 9 cm^2.
-/
theorem right_triangle_circle_area :
  ∃ (r : ℝ), 
  let AB := 6
  let BC := 8
  let AC := Real.sqrt (AB^2 + BC^2)
  let E := (AB / 2)
  let D := (AC / 2)
  let radius := (AC * (BC / 2) / AB)
  r = radius * radius * Real.pi ∧
  r = (100 * Real.pi / 9) := sorry

end NUMINAMATH_GPT_right_triangle_circle_area_l64_6455


namespace NUMINAMATH_GPT_angle_b_is_acute_l64_6429

-- Definitions for angles being right, acute, and sum of angles in a triangle
def is_right_angle (θ : ℝ) : Prop := θ = 90
def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def angles_sum_to_180 (α β γ : ℝ) : Prop := α + β + γ = 180

-- Main theorem statement
theorem angle_b_is_acute {α β γ : ℝ} (hC : is_right_angle γ) (hSum : angles_sum_to_180 α β γ) : is_acute_angle β :=
by
  sorry

end NUMINAMATH_GPT_angle_b_is_acute_l64_6429


namespace NUMINAMATH_GPT_find_value_of_A_l64_6457

-- Define the conditions
variable (A : ℕ)
variable (divisor : ℕ := 9)
variable (quotient : ℕ := 2)
variable (remainder : ℕ := 6)

-- The main statement of the proof problem
theorem find_value_of_A (h : A = quotient * divisor + remainder) : A = 24 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_find_value_of_A_l64_6457


namespace NUMINAMATH_GPT_infinite_n_exists_l64_6458

-- Definitions from conditions
def is_natural_number (a : ℕ) : Prop := a > 3

-- Statement of the theorem
theorem infinite_n_exists (a : ℕ) (h : is_natural_number a) : ∃ᶠ n in at_top, a + n ∣ a^n + 1 :=
sorry

end NUMINAMATH_GPT_infinite_n_exists_l64_6458


namespace NUMINAMATH_GPT_markers_leftover_l64_6450

theorem markers_leftover :
  let total_markers := 154
  let num_packages := 13
  total_markers % num_packages = 11 :=
by
  sorry

end NUMINAMATH_GPT_markers_leftover_l64_6450


namespace NUMINAMATH_GPT_max_value_of_sum_of_cubes_l64_6415

theorem max_value_of_sum_of_cubes (a b c d e : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_GPT_max_value_of_sum_of_cubes_l64_6415


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l64_6442

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : a 3 = 1/2)
  (h3 : a 1 * (1 + q) = 3) :
  q = 1/2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l64_6442


namespace NUMINAMATH_GPT_same_cost_number_of_guests_l64_6425

theorem same_cost_number_of_guests (x : ℕ) : 
  (800 + 30 * x = 500 + 35 * x) ↔ (x = 60) :=
by {
  sorry
}

end NUMINAMATH_GPT_same_cost_number_of_guests_l64_6425


namespace NUMINAMATH_GPT_sum_of_last_three_digits_9_pow_15_plus_15_pow_15_l64_6438

theorem sum_of_last_three_digits_9_pow_15_plus_15_pow_15 :
  (9 ^ 15 + 15 ^ 15) % 1000 = 24 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_last_three_digits_9_pow_15_plus_15_pow_15_l64_6438


namespace NUMINAMATH_GPT_solve_fractions_in_integers_l64_6486

theorem solve_fractions_in_integers :
  ∀ (a b c : ℤ), (1 / a + 1 / b + 1 / c = 1) ↔
  (a = 3 ∧ b = 3 ∧ c = 3) ∨
  (a = 2 ∧ b = 3 ∧ c = 6) ∨
  (a = 2 ∧ b = 4 ∧ c = 4) ∨
  (a = 1 ∧ ∃ t : ℤ, b = t ∧ c = -t) :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_fractions_in_integers_l64_6486


namespace NUMINAMATH_GPT_train_length_l64_6420

/-- 
  Given:
  - jogger_speed is the jogger's speed in km/hr (9 km/hr)
  - train_speed is the train's speed in km/hr (45 km/hr)
  - jogger_ahead is the jogger's initial lead in meters (240 m)
  - passing_time is the time in seconds for the train to pass the jogger (36 s)
  
  Prove that the length of the train is 120 meters.
-/
theorem train_length
  (jogger_speed : ℕ) -- in km/hr
  (train_speed : ℕ) -- in km/hr
  (jogger_ahead : ℕ) -- in meters
  (passing_time : ℕ) -- in seconds
  (h_jogger_speed : jogger_speed = 9)
  (h_train_speed : train_speed = 45)
  (h_jogger_ahead : jogger_ahead = 240)
  (h_passing_time : passing_time = 36)
  : ∃ length_of_train : ℕ, length_of_train = 120 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l64_6420


namespace NUMINAMATH_GPT_smallest_sum_of_cubes_two_ways_l64_6408

theorem smallest_sum_of_cubes_two_ways :
  ∃ (n : ℕ) (a b c d e f : ℕ),
  n = a^3 + b^3 + c^3 ∧ n = d^3 + e^3 + f^3 ∧
  (a, b, c) ≠ (d, e, f) ∧
  (d, e, f) ≠ (a, b, c) ∧ n = 251 :=
by
  sorry

end NUMINAMATH_GPT_smallest_sum_of_cubes_two_ways_l64_6408


namespace NUMINAMATH_GPT_first_part_lent_years_l64_6490

theorem first_part_lent_years (x n : ℕ) (total_sum second_sum : ℕ) (rate1 rate2 years2 : ℝ) :
  total_sum = 2743 →
  second_sum = 1688 →
  rate1 = 3 →
  rate2 = 5 →
  years2 = 3 →
  (x = total_sum - second_sum) →
  (x * n * rate1 / 100 = second_sum * rate2 * years2 / 100) →
  n = 8 :=
by
  sorry

end NUMINAMATH_GPT_first_part_lent_years_l64_6490


namespace NUMINAMATH_GPT_david_average_speed_l64_6409

theorem david_average_speed (d t : ℚ) (h1 : d = 49 / 3) (h2 : t = 7 / 3) :
  (d / t) = 7 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_david_average_speed_l64_6409


namespace NUMINAMATH_GPT_six_digit_numbers_l64_6460

theorem six_digit_numbers :
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
  sorry

end NUMINAMATH_GPT_six_digit_numbers_l64_6460


namespace NUMINAMATH_GPT_lcm_23_46_827_l64_6431

theorem lcm_23_46_827 :
  (23 * 46 * 827) / gcd (23 * 2) 827 = 38042 := by
  sorry

end NUMINAMATH_GPT_lcm_23_46_827_l64_6431


namespace NUMINAMATH_GPT_radius_of_circle_l64_6441

def circle_eq_def (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

theorem radius_of_circle {x y r : ℝ} (h : circle_eq_def x y) : r = 3 := 
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_radius_of_circle_l64_6441


namespace NUMINAMATH_GPT_new_car_distance_l64_6461

-- State the given conditions as a Lemma in Lean 4
theorem new_car_distance (d_old : ℝ) (d_new : ℝ) (h1 : d_old = 150) (h2 : d_new = d_old * 1.30) : d_new = 195 := 
by
  sorry

end NUMINAMATH_GPT_new_car_distance_l64_6461


namespace NUMINAMATH_GPT_clownfish_in_display_tank_l64_6495

theorem clownfish_in_display_tank (C B : ℕ) (h1 : C = B) (h2 : C + B = 100) : 
  (B - 26 - (B - 26) / 3) = 16 := by
  sorry

end NUMINAMATH_GPT_clownfish_in_display_tank_l64_6495


namespace NUMINAMATH_GPT_calculate_expression_l64_6400

variable (x y : ℚ)

theorem calculate_expression (h₁ : x = 4 / 6) (h₂ : y = 5 / 8) : 
  (6 * x + 8 * y) / (48 * x * y) = 9 / 20 :=
by
  -- proof steps here
  sorry

end NUMINAMATH_GPT_calculate_expression_l64_6400


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_factors_of_13_l64_6413

theorem sum_of_reciprocals_of_factors_of_13 : 
  (1 : ℚ) + (1 / 13) = 14 / 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_reciprocals_of_factors_of_13_l64_6413


namespace NUMINAMATH_GPT_calculate_p_l64_6456

variable (m n : ℤ) (p : ℤ)

theorem calculate_p (h1 : 3 * m - 2 * n = -2) (h2 : p = 3 * (m + 405) - 2 * (n - 405)) : p = 2023 := 
  sorry

end NUMINAMATH_GPT_calculate_p_l64_6456


namespace NUMINAMATH_GPT_largest_integer_among_four_l64_6432

theorem largest_integer_among_four 
  (p q r s : ℤ)
  (h1 : p + q + r = 210)
  (h2 : p + q + s = 230)
  (h3 : p + r + s = 250)
  (h4 : q + r + s = 270) :
  max (max p q) (max r s) = 110 :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_among_four_l64_6432


namespace NUMINAMATH_GPT_comp_inter_empty_l64_6491

section
variable {α : Type*} [DecidableEq α]
variable (I M N : Set α)
variable (a b c d e : α)
variable (hI : I = {a, b, c, d, e})
variable (hM : M = {a, c, d})
variable (hN : N = {b, d, e})

theorem comp_inter_empty : 
  (I \ M) ∩ (I \ N) = ∅ :=
by sorry
end

end NUMINAMATH_GPT_comp_inter_empty_l64_6491


namespace NUMINAMATH_GPT_transformed_system_solution_l64_6481

theorem transformed_system_solution 
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (h1 : a1 * 3 + b1 * 4 = c1)
  (h2 : a2 * 3 + b2 * 4 = c2) :
  (3 * a1 * 5 + 4 * b1 * 5 = 5 * c1) ∧ (3 * a2 * 5 + 4 * b2 * 5 = 5 * c2) :=
by 
  sorry

end NUMINAMATH_GPT_transformed_system_solution_l64_6481


namespace NUMINAMATH_GPT_crayons_total_l64_6412

theorem crayons_total (blue red green : ℕ) 
  (h1 : red = 4 * blue) 
  (h2 : green = 2 * red) 
  (h3 : blue = 3) : 
  blue + red + green = 39 := 
by
  sorry

end NUMINAMATH_GPT_crayons_total_l64_6412


namespace NUMINAMATH_GPT_line_equation_through_P_and_intercepts_l64_6434

-- Define the conditions
structure Point (α : Type*) := 
  (x : α) 
  (y : α)

-- Given point P
def P : Point ℝ := ⟨5, 6⟩

-- Equation of a line passing through (x₀, y₀) and 
-- having the intercepts condition: the x-intercept is twice the y-intercept

theorem line_equation_through_P_and_intercepts :
  (∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (a * 5 + b * 6 + c = 0) ∧ 
   ((-c / a = 2 * (-c / b)) ∧ (c ≠ 0)) ∧
   (a = 1 ∧ b = 2 ∧ c = -17) ∨
   (a = 6 ∧ b = -5 ∧ c = 0)) :=
sorry

end NUMINAMATH_GPT_line_equation_through_P_and_intercepts_l64_6434


namespace NUMINAMATH_GPT_triangle_height_l64_6443

theorem triangle_height (A : ℝ) (b : ℝ) (h : ℝ) 
  (hA : A = 615) 
  (hb : b = 123)
  (h_area : A = 0.5 * b * h) : 
  h = 10 :=
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_triangle_height_l64_6443


namespace NUMINAMATH_GPT_find_length_PB_l64_6423

-- Define the conditions of the problem
variables (AC AP PB : ℝ) (x : ℝ)

-- Condition: The length of chord AC is x
def length_AC := AC = x

-- Condition: The length of segment AP is x + 1
def length_AP := AP = x + 1

-- Statement of the theorem to prove the length of segment PB
theorem find_length_PB (h_AC : length_AC AC x) (h_AP : length_AP AP x) :
  PB = 2 * x + 1 :=
sorry

end NUMINAMATH_GPT_find_length_PB_l64_6423


namespace NUMINAMATH_GPT_interest_rate_per_annum_l64_6448

theorem interest_rate_per_annum
  (P : ℕ := 450) 
  (t : ℕ := 8) 
  (I : ℕ := P - 306) 
  (simple_interest : ℕ := P * r * t / 100) :
  r = 4 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_per_annum_l64_6448


namespace NUMINAMATH_GPT_intersection_M_N_l64_6483

def M (x : ℝ) : Prop := x^2 + 2*x - 15 < 0
def N (x : ℝ) : Prop := x^2 + 6*x - 7 ≥ 0

theorem intersection_M_N : {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | 1 ≤ x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l64_6483


namespace NUMINAMATH_GPT_power_function_at_4_l64_6410

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

theorem power_function_at_4 {α : ℝ} :
  power_function α 2 = (Real.sqrt 2) / 2 →
  α = -1/2 →
  power_function α 4 = 1 / 2 :=
by
  intros h1 h2
  rw [h2, power_function]
  sorry

end NUMINAMATH_GPT_power_function_at_4_l64_6410


namespace NUMINAMATH_GPT_exists_positive_integers_x_y_l64_6451

theorem exists_positive_integers_x_y (x y : ℕ) : 0 < x ∧ 0 < y ∧ x^2 = y^2 + 2023 :=
  sorry

end NUMINAMATH_GPT_exists_positive_integers_x_y_l64_6451


namespace NUMINAMATH_GPT_mean_of_sets_l64_6444

theorem mean_of_sets (x : ℝ) (h : (28 + x + 42 + 78 + 104) / 5 = 62) : 
  (48 + 62 + 98 + 124 + x) / 5 = 78 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_sets_l64_6444


namespace NUMINAMATH_GPT_max_min_z_diff_correct_l64_6422

noncomputable def max_min_z_diff (x y z : ℝ) (h1 : x + y + z = 3) (h2 : x^2 + y^2 + z^2 = 18) : ℝ :=
  6

theorem max_min_z_diff_correct (x y z : ℝ) (h1 : x + y + z = 3) (h2 : x^2 + y^2 + z^2 = 18) :
  max_min_z_diff x y z h1 h2 = 6 :=
sorry

end NUMINAMATH_GPT_max_min_z_diff_correct_l64_6422


namespace NUMINAMATH_GPT_factory_production_eq_l64_6428

theorem factory_production_eq (x : ℝ) (h1 : x > 50) : 450 / (x - 50) - 400 / x = 1 := 
by 
  sorry

end NUMINAMATH_GPT_factory_production_eq_l64_6428


namespace NUMINAMATH_GPT_largest_number_of_right_angles_in_convex_octagon_l64_6436

theorem largest_number_of_right_angles_in_convex_octagon : 
  ∀ (angles : Fin 8 → ℝ), 
  (∀ i, 0 < angles i ∧ angles i < 180) → 
  (angles 0 + angles 1 + angles 2 + angles 3 + angles 4 + angles 5 + angles 6 + angles 7 = 1080) → 
  ∃ k, k ≤ 6 ∧ (∀ i < 8, if angles i = 90 then k = 6 else true) := 
by 
  sorry

end NUMINAMATH_GPT_largest_number_of_right_angles_in_convex_octagon_l64_6436


namespace NUMINAMATH_GPT_fiona_prob_reaches_12_l64_6494

/-- Lily pads are numbered from 0 to 15 -/
def is_valid_pad (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 15

/-- Predators are on lily pads 4 and 7 -/
def predator (n : ℕ) : Prop := n = 4 ∨ n = 7

/-- Fiona the frog's probability to hop to the next pad -/
def hop : ℚ := 1 / 2

/-- Fiona the frog's probability to jump 2 pads -/
def jump_two : ℚ := 1 / 2

/-- Probability that Fiona reaches pad 12 without landing on pads 4 or 7 is 1/32 -/
theorem fiona_prob_reaches_12 :
  ∀ p : ℕ, 
    (is_valid_pad p ∧ ¬ predator p ∧ (p = 12) ∧ 
    ((∀ k : ℕ, is_valid_pad k → ¬ predator k → k ≤ 3 → (hop ^ k) = 1 / 2) ∧
    hop * hop = 1 / 4 ∧ hop * jump_two = 1 / 8 ∧
    (jump_two * (hop * hop + jump_two)) = 1 / 4 → hop * 1 / 4 = 1 / 32)) := 
by intros; sorry

end NUMINAMATH_GPT_fiona_prob_reaches_12_l64_6494


namespace NUMINAMATH_GPT_sheila_hourly_earnings_l64_6498

def sheila_hours_per_day (day : String) : Nat :=
  if day = "Monday" ∨ day = "Wednesday" ∨ day = "Friday" then 8
  else if day = "Tuesday" ∨ day = "Thursday" then 6
  else 0

def sheila_weekly_hours : Nat :=
  sheila_hours_per_day "Monday" +
  sheila_hours_per_day "Tuesday" +
  sheila_hours_per_day "Wednesday" +
  sheila_hours_per_day "Thursday" +
  sheila_hours_per_day "Friday"

def sheila_weekly_earnings : Nat := 468

theorem sheila_hourly_earnings :
  sheila_weekly_earnings / sheila_weekly_hours = 13 :=
by
  sorry

end NUMINAMATH_GPT_sheila_hourly_earnings_l64_6498


namespace NUMINAMATH_GPT_find_x_l64_6476

theorem find_x (x y z : ℕ) (h1 : x = y / 2) (h2 : y = z / 3) (h3 : z = 90) : x = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l64_6476


namespace NUMINAMATH_GPT_total_cost_pants_and_belt_l64_6430

theorem total_cost_pants_and_belt (P B : ℝ) 
  (hP : P = 34.0) 
  (hCondition : P = B - 2.93) : 
  P + B = 70.93 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_total_cost_pants_and_belt_l64_6430


namespace NUMINAMATH_GPT_painting_time_eq_l64_6496

theorem painting_time_eq (t : ℝ) :
  (1 / 6 + 1 / 8 + 1 / 12) * t = 1 ↔ t = 8 / 3 :=
by
  sorry

end NUMINAMATH_GPT_painting_time_eq_l64_6496


namespace NUMINAMATH_GPT_roots_of_abs_exp_eq_b_l64_6439

theorem roots_of_abs_exp_eq_b (b : ℝ) (h : 0 < b ∧ b < 1) : 
  ∃! (x1 x2 : ℝ), x1 ≠ x2 ∧ abs (2^x1 - 1) = b ∧ abs (2^x2 - 1) = b :=
sorry

end NUMINAMATH_GPT_roots_of_abs_exp_eq_b_l64_6439


namespace NUMINAMATH_GPT_circles_intersect_and_inequality_l64_6477

variable {R r d : ℝ}

theorem circles_intersect_and_inequality (hR : R > r) (h_intersect: R - r < d ∧ d < R + r) : R - r < d ∧ d < R + r :=
by
  exact h_intersect

end NUMINAMATH_GPT_circles_intersect_and_inequality_l64_6477


namespace NUMINAMATH_GPT_infinite_positive_integer_solutions_l64_6489

theorem infinite_positive_integer_solutions :
  ∃ (k : ℕ), ∀ (n : ℕ), n > 24 → ∃ k > 24, k = n :=
sorry

end NUMINAMATH_GPT_infinite_positive_integer_solutions_l64_6489


namespace NUMINAMATH_GPT_asia_fraction_correct_l64_6454

-- Define the problem conditions
def fraction_NA (P : ℕ) : ℚ := 1/3 * P
def fraction_Europe (P : ℕ) : ℚ := 1/8 * P
def fraction_Africa (P : ℕ) : ℚ := 1/5 * P
def others : ℕ := 42
def total_passengers : ℕ := 240

-- Define the target fraction for Asia
def fraction_Asia (P: ℕ) : ℚ := 17 / 120

-- Theorem: the fraction of the passengers from Asia equals 17/120
theorem asia_fraction_correct : ∀ (P : ℕ), 
  P = total_passengers →
  fraction_NA P + fraction_Europe P + fraction_Africa P + fraction_Asia P * P + others = P →
  fraction_Asia P = 17 / 120 := 
by sorry

end NUMINAMATH_GPT_asia_fraction_correct_l64_6454


namespace NUMINAMATH_GPT_tom_saves_money_l64_6435

-- Defining the cost of a normal doctor's visit
def normal_doctor_cost : ℕ := 200

-- Defining the discount percentage for the discount clinic
def discount_percentage : ℕ := 70

-- Defining the cost reduction based on the discount percentage
def discount_amount (cost percentage : ℕ) : ℕ := (percentage * cost) / 100

-- Defining the cost of a visit to the discount clinic
def discount_clinic_cost (normal_cost discount_amount : ℕ ) : ℕ := normal_cost - discount_amount

-- Defining the number of visits to the discount clinic
def discount_clinic_visits : ℕ := 2

-- Defining the total cost for the discount clinic visits
def total_discount_clinic_cost (visit_cost visits : ℕ) : ℕ := visits * visit_cost

-- The final cost savings calculation
def cost_savings (normal_cost total_discount_cost : ℕ) : ℕ := normal_cost - total_discount_cost

-- Proving the amount Tom saves by going to the discount clinic
theorem tom_saves_money : cost_savings normal_doctor_cost (total_discount_clinic_cost (discount_clinic_cost normal_doctor_cost (discount_amount normal_doctor_cost discount_percentage)) discount_clinic_visits) = 80 :=
by
  sorry

end NUMINAMATH_GPT_tom_saves_money_l64_6435


namespace NUMINAMATH_GPT_find_a3_l64_6469

noncomputable def geometric_seq (a : ℕ → ℕ) (q : ℕ) : Prop :=
∀ n, a (n+1) = a n * q

theorem find_a3 (a : ℕ → ℕ) (q : ℕ) (h_geom : geometric_seq a q) (hq : q > 1)
  (h1 : a 4 - a 0 = 15) (h2 : a 3 - a 1 = 6) :
  a 2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_a3_l64_6469


namespace NUMINAMATH_GPT_kerosene_cost_l64_6470

/-- In a market, a dozen eggs cost as much as a pound of rice, and a half-liter of kerosene 
costs as much as 8 eggs. If the cost of each pound of rice is $0.33, then a liter of kerosene costs 44 cents. --/
theorem kerosene_cost : 
  let egg_cost := 0.33 / 12
  let rice_cost := 0.33
  let half_liter_kerosene_cost := 8 * egg_cost
  rice_cost = 0.33 → 1 * ((2 * half_liter_kerosene_cost) * 100) = 44 := 
by
  intros egg_cost rice_cost half_liter_kerosene_cost h_rice_cost
  sorry

end NUMINAMATH_GPT_kerosene_cost_l64_6470


namespace NUMINAMATH_GPT_ratio_of_wealth_l64_6407

theorem ratio_of_wealth (P W : ℝ) (hP : P > 0) (hW : W > 0) : 
  let wX := (0.40 * W) / (0.20 * P)
  let wY := (0.30 * W) / (0.10 * P)
  (wX / wY) = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_wealth_l64_6407


namespace NUMINAMATH_GPT_Shaina_chocolate_l64_6464

-- Definitions based on the conditions
def total_chocolate : ℚ := 72 / 7
def number_of_piles : ℚ := 6
def weight_per_pile : ℚ := total_chocolate / number_of_piles
def piles_given_to_Shaina : ℚ := 2

-- Theorem stating the problem's correct answer
theorem Shaina_chocolate :
  piles_given_to_Shaina * weight_per_pile = 24 / 7 :=
by
  sorry

end NUMINAMATH_GPT_Shaina_chocolate_l64_6464


namespace NUMINAMATH_GPT_no_solution_fermat_like_l64_6402

theorem no_solution_fermat_like (x y z k : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hk : k > 0) 
  (hxk : x < k) (hyk : y < k) (hxk_eq : x ^ k + y ^ k = z ^ k) : false :=
sorry

end NUMINAMATH_GPT_no_solution_fermat_like_l64_6402


namespace NUMINAMATH_GPT_probability_correct_l64_6424

noncomputable def probability_point_between_lines : ℝ :=
  let intersection_x_l := 4    -- x-intercept of line l
  let intersection_x_m := 3    -- x-intercept of line m
  let area_under_l := (1 / 2) * intersection_x_l * 8 -- area under line l
  let area_under_m := (1 / 2) * intersection_x_m * 9 -- area under line m
  let area_between := area_under_l - area_under_m    -- area between lines
  (area_between / area_under_l : ℝ)

theorem probability_correct : probability_point_between_lines = 0.16 :=
by
  simp only [probability_point_between_lines]
  sorry

end NUMINAMATH_GPT_probability_correct_l64_6424


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l64_6493

variable (a : ℝ)

theorem sufficient_but_not_necessary : (a = 1 → |a| = 1) ∧ (|a| = 1 → a = 1 → False) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l64_6493


namespace NUMINAMATH_GPT_conic_sections_of_equation_l64_6475

theorem conic_sections_of_equation :
  (∀ x y : ℝ, y^6 - 6 * x^6 = 3 * y^2 - 8 → y^2 = 6 * x^2 ∨ y^2 = -6 * x^2 + 2) :=
sorry

end NUMINAMATH_GPT_conic_sections_of_equation_l64_6475


namespace NUMINAMATH_GPT_average_earnings_per_minute_l64_6419

theorem average_earnings_per_minute (race_duration : ℕ) (lap_distance : ℕ) (certificate_rate : ℝ) (laps_run : ℕ) :
  race_duration = 12 → 
  lap_distance = 100 → 
  certificate_rate = 3.5 → 
  laps_run = 24 → 
  ((laps_run * lap_distance / 100) * certificate_rate) / race_duration = 7 :=
by
  intros hrace_duration hlap_distance hcertificate_rate hlaps_run
  rw [hrace_duration, hlap_distance, hcertificate_rate, hlaps_run]
  sorry

end NUMINAMATH_GPT_average_earnings_per_minute_l64_6419


namespace NUMINAMATH_GPT_sum_arithmetic_sequence_satisfies_conditions_l64_6401

theorem sum_arithmetic_sequence_satisfies_conditions :
  ∀ (a : ℕ → ℤ) (d : ℤ),
  (a 1 = 1) ∧ (d ≠ 0) ∧ ((a 3)^2 = (a 2) * (a 6)) →
  (6 * a 1 + (6 * 5 / 2) * d = -24) :=
by
  sorry

end NUMINAMATH_GPT_sum_arithmetic_sequence_satisfies_conditions_l64_6401


namespace NUMINAMATH_GPT_smallest_expression_value_l64_6497

theorem smallest_expression_value (a b c : ℝ) (h₁ : b > c) (h₂ : c > 0) (h₃ : a ≠ 0) :
  (2 * a + b) ^ 2 + (b - c) ^ 2 + (c - 2 * a) ^ 2 ≥ (4 / 3) * b ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_expression_value_l64_6497


namespace NUMINAMATH_GPT_transformation_composition_l64_6404

-- Define the transformations f and g
def f (m n : ℝ) : ℝ × ℝ := (m, -n)
def g (m n : ℝ) : ℝ × ℝ := (-m, -n)

-- The proof statement that we need to prove
theorem transformation_composition : g (f (-3) 2).1 (f (-3) 2).2 = (3, 2) :=
by sorry

end NUMINAMATH_GPT_transformation_composition_l64_6404


namespace NUMINAMATH_GPT_correct_misread_number_l64_6480

theorem correct_misread_number (s : List ℕ) (wrong_avg correct_avg n wrong_num correct_num : ℕ) 
  (h1 : s.length = 10) 
  (h2 : (s.sum) / n = wrong_avg) 
  (h3 : wrong_num = 26) 
  (h4 : correct_avg = 16) 
  (h5 : n = 10) 
  : correct_num = 36 :=
sorry

end NUMINAMATH_GPT_correct_misread_number_l64_6480


namespace NUMINAMATH_GPT_find_remainder_2500th_term_l64_6446

theorem find_remainder_2500th_term : 
    let seq_position (n : ℕ) := n * (n + 1) / 2 
    let n := ((1 + Int.ofNat 20000).natAbs.sqrt + 1) / 2
    let term_2500 := if seq_position n < 2500 then n + 1 else n
    (term_2500 % 7) = 1 := by 
    sorry

end NUMINAMATH_GPT_find_remainder_2500th_term_l64_6446


namespace NUMINAMATH_GPT_tricycle_count_l64_6479

variables (b t : ℕ)

theorem tricycle_count :
  b + t = 7 ∧ 2 * b + 3 * t = 19 → t = 5 := by
  intro h
  sorry

end NUMINAMATH_GPT_tricycle_count_l64_6479


namespace NUMINAMATH_GPT_product_of_two_numbers_l64_6411

theorem product_of_two_numbers (a b : ℕ) (h_gcd : Nat.gcd a b = 8) (h_lcm : Nat.lcm a b = 72) : a * b = 576 := 
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l64_6411


namespace NUMINAMATH_GPT_coffee_y_ratio_is_1_to_5_l64_6466

-- Define the conditions
variables {p v x y : Type}
variables (p_x p_y v_x v_y : ℕ) -- Coffee amounts in lbs
variables (total_p total_v : ℕ) -- Total amounts of p and v

-- Definitions based on conditions
def coffee_amounts_initial (total_p total_v : ℕ) : Prop :=
  total_p = 24 ∧ total_v = 25

def coffee_x_conditions (p_x v_x : ℕ) : Prop :=
  p_x = 20 ∧ 4 * v_x = p_x

def coffee_y_conditions (p_y v_y total_p total_v : ℕ) : Prop :=
  p_y = total_p - 20 ∧ v_y = total_v - (20 / 4)

-- Statement to prove
theorem coffee_y_ratio_is_1_to_5 {total_p total_v : ℕ}
  (hc1 : coffee_amounts_initial total_p total_v)
  (hc2 : coffee_x_conditions 20 5)
  (hc3 : coffee_y_conditions 4 20 total_p total_v) : 
  (4 / 20 = 1 / 5) :=
sorry

end NUMINAMATH_GPT_coffee_y_ratio_is_1_to_5_l64_6466


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l64_6421

-- Define the set A
def A := {x : ℝ | -1 < x ∧ x < 2}

-- Define the necessary but not sufficient condition
def necessary_condition (a : ℝ) : Prop := a ≥ 1

-- Define the proposition that needs to be proved
def proposition (a : ℝ) : Prop := ∀ x ∈ A, x^2 - a < 0

-- The proof statement
theorem necessary_but_not_sufficient_condition (a : ℝ) :
  necessary_condition a → ∃ x ∈ A, proposition a :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l64_6421


namespace NUMINAMATH_GPT_length_of_angle_bisector_l64_6474

theorem length_of_angle_bisector (AB AC : ℝ) (angleBAC : ℝ) (AD : ℝ) :
  AB = 6 → AC = 3 → angleBAC = 60 → AD = 2 * Real.sqrt 3 :=
by
  intro hAB hAC hAngleBAC
  -- Consider adding proof steps here in the future
  sorry

end NUMINAMATH_GPT_length_of_angle_bisector_l64_6474


namespace NUMINAMATH_GPT_sum_squares_seven_consecutive_not_perfect_square_l64_6440

theorem sum_squares_seven_consecutive_not_perfect_square : 
  ∀ (n : ℤ), ¬ ∃ k : ℤ, k * k = (n-3)^2 + (n-2)^2 + (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2 :=
by
  sorry

end NUMINAMATH_GPT_sum_squares_seven_consecutive_not_perfect_square_l64_6440


namespace NUMINAMATH_GPT_simplify_expression_l64_6417

variable (x : ℝ)

theorem simplify_expression : (x + 2)^2 - (x + 1) * (x + 3) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l64_6417


namespace NUMINAMATH_GPT_ratio_of_a_to_c_l64_6467

theorem ratio_of_a_to_c (a b c d : ℚ) 
  (h1 : a / b = 5 / 4)
  (h2 : c / d = 4 / 3)
  (h3 : d / b = 1 / 5) :
  a / c = 75 / 16 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_a_to_c_l64_6467


namespace NUMINAMATH_GPT_daily_calories_burned_l64_6487

def calories_per_pound : ℕ := 3500
def pounds_to_lose : ℕ := 5
def days : ℕ := 35
def total_calories := pounds_to_lose * calories_per_pound

theorem daily_calories_burned :
  (total_calories / days) = 500 := 
  by 
    -- calculation steps
    sorry

end NUMINAMATH_GPT_daily_calories_burned_l64_6487


namespace NUMINAMATH_GPT_proof_problem_l64_6478

noncomputable def A := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}
noncomputable def B := {(x, y) : ℝ × ℝ | y = x^2 + 1}

theorem proof_problem :
  ((1, 2) ∈ B) ∧
  (0 ∉ A) ∧
  ((0, 0) ∉ B) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l64_6478


namespace NUMINAMATH_GPT_correct_sampling_method_l64_6488

-- Definitions based on conditions
def number_of_classes : ℕ := 16
def sampled_classes : ℕ := 2
def sampling_method := "Lottery then Stratified"

-- The theorem statement based on the proof problem
theorem correct_sampling_method :
  (number_of_classes = 16) ∧ (sampled_classes = 2) → (sampling_method = "Lottery then Stratified") :=
sorry

end NUMINAMATH_GPT_correct_sampling_method_l64_6488


namespace NUMINAMATH_GPT_stacy_height_proof_l64_6482

noncomputable def height_last_year : ℕ := 50
noncomputable def brother_growth : ℕ := 1
noncomputable def stacy_growth : ℕ := brother_growth + 6
noncomputable def stacy_current_height : ℕ := height_last_year + stacy_growth

theorem stacy_height_proof : stacy_current_height = 57 := 
by
  sorry

end NUMINAMATH_GPT_stacy_height_proof_l64_6482


namespace NUMINAMATH_GPT_find_n_l64_6468

theorem find_n (e n : ℕ) (h_lcm : Nat.lcm e n = 690) (h_n_not_div_3 : ¬ (3 ∣ n)) (h_e_not_div_2 : ¬ (2 ∣ e)) : n = 230 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l64_6468


namespace NUMINAMATH_GPT_problem1_problem2_l64_6405

-- Problem 1 Definition: Operation ※
def operation (m n : ℚ) : ℚ := 3 * m - n

-- Lean 4 statement: Prove 2※10 = -4
theorem problem1 : operation 2 10 = -4 := by
  sorry

-- Lean 4 statement: Prove that ※ does not satisfy the distributive law
theorem problem2 (a b c : ℚ) : 
  operation a (b + c) ≠ operation a b + operation a c := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l64_6405


namespace NUMINAMATH_GPT_print_time_nearest_whole_l64_6447

theorem print_time_nearest_whole 
  (pages_per_minute : ℕ) (total_pages : ℕ) (expected_time : ℕ)
  (h1 : pages_per_minute = 25) (h2 : total_pages = 575) : 
  expected_time = 23 :=
by
  sorry

end NUMINAMATH_GPT_print_time_nearest_whole_l64_6447
