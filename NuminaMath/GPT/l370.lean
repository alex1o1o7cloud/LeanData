import Mathlib

namespace NUMINAMATH_GPT_no_third_degree_polynomial_exists_l370_37097

theorem no_third_degree_polynomial_exists (a b c d : ℤ) (h : a ≠ 0) :
  ¬(p 15 = 3 ∧ p 21 = 12 ∧ p = λ x => a * x ^ 3 + b * x ^ 2 + c * x + d) :=
sorry

end NUMINAMATH_GPT_no_third_degree_polynomial_exists_l370_37097


namespace NUMINAMATH_GPT_largest_fraction_sum_l370_37000

theorem largest_fraction_sum : 
  (max (max (max (max 
  ((1 : ℚ) / 3 + (1 : ℚ) / 4) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 5)) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 2)) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 9)) 
  ((1 : ℚ) / 3 + (1 : ℚ) / 6)) = (5 : ℚ) / 6 
:= 
by
  sorry

end NUMINAMATH_GPT_largest_fraction_sum_l370_37000


namespace NUMINAMATH_GPT_largest_multiple_of_15_less_than_500_l370_37040

theorem largest_multiple_of_15_less_than_500 : ∃ x : ℕ, x < 500 ∧ x % 15 = 0 ∧ (∀ y : ℕ, y < 500 ∧ y % 15 = 0 → y ≤ x) := sorry

end NUMINAMATH_GPT_largest_multiple_of_15_less_than_500_l370_37040


namespace NUMINAMATH_GPT_find_f_values_find_f_expression_l370_37087

variable (f : ℕ+ → ℤ)

-- Conditions in Lean
def is_increasing (f : ℕ+ → ℤ) : Prop :=
  ∀ {m n : ℕ+}, m < n → f m < f n

axiom h1 : is_increasing f
axiom h2 : f 4 = 5
axiom h3 : ∀ n : ℕ+, ∃ k : ℕ, f n = k
axiom h4 : ∀ m n : ℕ+, f m * f n = f (m * n) + f (m + n - 1)

-- Proof in Lean 4
theorem find_f_values : f 1 = 2 ∧ f 2 = 3 ∧ f 3 = 4 :=
by
  sorry

theorem find_f_expression : ∀ n : ℕ+, f n = n + 1 :=
by
  sorry

end NUMINAMATH_GPT_find_f_values_find_f_expression_l370_37087


namespace NUMINAMATH_GPT_max_brownies_l370_37027

theorem max_brownies (m n : ℕ) (h : (m - 2) * (n - 2) = 2 * m + 2 * n - 4) : m * n ≤ 60 :=
sorry

end NUMINAMATH_GPT_max_brownies_l370_37027


namespace NUMINAMATH_GPT_expr_undefined_iff_l370_37047

theorem expr_undefined_iff (b : ℝ) : ¬ ∃ y : ℝ, y = (b - 1) / (b^2 - 9) ↔ b = -3 ∨ b = 3 :=
by 
  sorry

end NUMINAMATH_GPT_expr_undefined_iff_l370_37047


namespace NUMINAMATH_GPT_find_c_l370_37034

theorem find_c (c : ℝ) (h : ∀ x : ℝ, ∃ a : ℝ, (x + a)^2 = x^2 + 200 * x + c) : c = 10000 :=
sorry

end NUMINAMATH_GPT_find_c_l370_37034


namespace NUMINAMATH_GPT_geometric_series_problem_l370_37002

theorem geometric_series_problem (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℕ)
  (h_seq : ∀ n, a n + a (n + 1) = 3 * 2^n) :
  S (k + 2) - 2 * S (k + 1) + S k = 2^(k + 1) :=
sorry

end NUMINAMATH_GPT_geometric_series_problem_l370_37002


namespace NUMINAMATH_GPT_find_ab_sum_l370_37088

theorem find_ab_sum 
  (a b : ℝ) 
  (h : ∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) 
  : a + b = -14 := by
  sorry

end NUMINAMATH_GPT_find_ab_sum_l370_37088


namespace NUMINAMATH_GPT_mr_smith_spends_l370_37064

def buffet_price 
  (adult_price : ℕ) 
  (child_price : ℕ) 
  (senior_discount : ℕ) 
  (num_full_price_adults : ℕ) 
  (num_children : ℕ) 
  (num_seniors : ℕ) : ℕ :=
  num_full_price_adults * adult_price + num_children * child_price + num_seniors * (adult_price - (adult_price * senior_discount / 100))

theorem mr_smith_spends (adult_price : ℕ) (child_price : ℕ) (senior_discount : ℕ) (num_full_price_adults : ℕ) (num_children : ℕ) (num_seniors : ℕ) : 
  adult_price = 30 → 
  child_price = 15 → 
  senior_discount = 10 → 
  num_full_price_adults = 3 → 
  num_children = 3 → 
  num_seniors = 1 → 
  buffet_price adult_price child_price senior_discount num_full_price_adults num_children num_seniors = 162 :=
by 
  intros h_adult_price h_child_price h_senior_discount h_num_full_price_adults h_num_children h_num_seniors
  rw [h_adult_price, h_child_price, h_senior_discount, h_num_full_price_adults, h_num_children, h_num_seniors]
  sorry

end NUMINAMATH_GPT_mr_smith_spends_l370_37064


namespace NUMINAMATH_GPT_max_gcd_of_linear_combinations_l370_37071

theorem max_gcd_of_linear_combinations (a b c : ℕ) (h1 : a + b + c ≤ 3000000) (h2 : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  gcd (a * b + 1) (gcd (a * c + 1) (b * c + 1)) ≤ 998285 :=
sorry

end NUMINAMATH_GPT_max_gcd_of_linear_combinations_l370_37071


namespace NUMINAMATH_GPT_simplify_expression_l370_37051

variable (x y : ℝ)

theorem simplify_expression :
  3 * x + 4 * x^2 + 2 - (5 - 3 * x - 5 * x^2 + 2 * y) = 9 * x^2 + 6 * x - 2 * y - 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l370_37051


namespace NUMINAMATH_GPT_yard_length_calculation_l370_37046

theorem yard_length_calculation (n_trees : ℕ) (distance : ℕ) (h1 : n_trees = 26) (h2 : distance = 32) : (n_trees - 1) * distance = 800 :=
by
  -- This is where the proof would go.
  sorry

end NUMINAMATH_GPT_yard_length_calculation_l370_37046


namespace NUMINAMATH_GPT_points_on_line_l370_37039

theorem points_on_line (n : ℕ) : 9 * n - 8 = 82 → n = 10 :=
by
  sorry

end NUMINAMATH_GPT_points_on_line_l370_37039


namespace NUMINAMATH_GPT_correct_propositions_l370_37095

variable {f : ℝ → ℝ}

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def period_2 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = f x

def symmetry_about_points (f : ℝ → ℝ) (k : ℤ) : Prop :=
  ∀ x, f (x + k) = f (x - k)

theorem correct_propositions (h1: is_odd_function f) (h2 : ∀ x, f (x + 1) = f (x -1)) :
  period_2 f ∧ (∀ k : ℤ, symmetry_about_points f k) :=
by
  sorry

end NUMINAMATH_GPT_correct_propositions_l370_37095


namespace NUMINAMATH_GPT_mona_cookie_count_l370_37007

theorem mona_cookie_count {M : ℕ} (h1 : (M - 5) + (M - 5 + 10) + M = 60) : M = 20 :=
by
  sorry

end NUMINAMATH_GPT_mona_cookie_count_l370_37007


namespace NUMINAMATH_GPT_team_b_can_serve_on_submarine_l370_37068

   def can_serve_on_submarine (height : ℝ) : Prop := height ≤ 168

   def average_height_condition (avg_height : ℝ) : Prop := avg_height = 166

   def median_height_condition (median_height : ℝ) : Prop := median_height = 167

   def tallest_height_condition (max_height : ℝ) : Prop := max_height = 169

   def mode_height_condition (mode_height : ℝ) : Prop := mode_height = 167

   theorem team_b_can_serve_on_submarine (H : median_height_condition 167) :
     ∀ (h : ℝ), can_serve_on_submarine h :=
   sorry
   
end NUMINAMATH_GPT_team_b_can_serve_on_submarine_l370_37068


namespace NUMINAMATH_GPT_work_time_B_l370_37090

theorem work_time_B (A_efficiency : ℕ) (B_efficiency : ℕ) (days_together : ℕ) (total_work : ℕ) :
  (A_efficiency = 2 * B_efficiency) →
  (days_together = 5) →
  (total_work = (A_efficiency + B_efficiency) * days_together) →
  (total_work / B_efficiency = 15) :=
by
  intros
  sorry

end NUMINAMATH_GPT_work_time_B_l370_37090


namespace NUMINAMATH_GPT_train_crosses_signal_post_in_40_seconds_l370_37086

noncomputable def time_to_cross_signal_post : Nat := 40

theorem train_crosses_signal_post_in_40_seconds
  (train_length : Nat) -- Length of the train in meters
  (bridge_length_km : Nat) -- Length of the bridge in kilometers
  (bridge_cross_time_min : Nat) -- Time to cross the bridge in minutes
  (constant_speed : Prop) -- Assumption that the speed is constant
  (h1 : train_length = 600) -- Train is 600 meters long
  (h2 : bridge_length_km = 9) -- Bridge is 9 kilometers long
  (h3 : bridge_cross_time_min = 10) -- Time to cross the bridge is 10 minutes
  (h4 : constant_speed) -- The train's speed is constant
  : time_to_cross_signal_post = 40 :=
sorry

end NUMINAMATH_GPT_train_crosses_signal_post_in_40_seconds_l370_37086


namespace NUMINAMATH_GPT_area_of_triangle_l370_37043

def triangle (α β γ : Type) : (α ≃ β) ≃ γ ≃ Prop := sorry

variables (α β γ : Type) (AB AC AM : ℝ)
variables (ha : AB = 9) (hb : AC = 17) (hc : AM = 12)

theorem area_of_triangle (α β γ : Type) (AB AC AM : ℝ)
  (ha : AB = 9) (hb : AC = 17) (hc : AM = 12) : 
  ∃ A : ℝ, A = 74 :=
sorry

end NUMINAMATH_GPT_area_of_triangle_l370_37043


namespace NUMINAMATH_GPT_width_of_domain_of_g_l370_37026

variable (h : ℝ → ℝ) (dom_h : ∀ x, -10 ≤ x ∧ x ≤ 10 → h x = h x)

noncomputable def g (x : ℝ) : ℝ := h (x / 3)

theorem width_of_domain_of_g :
  (∀ x, -10 ≤ x ∧ x ≤ 10 → h x = h x) →
  (∀ y : ℝ, -30 ≤ y ∧ y ≤ 30 → h (y / 3) = h (y / 3)) →
  (∃ a b : ℝ, a = -30 ∧ b = 30 ∧  (∃ w : ℝ, w = b - a ∧ w = 60)) :=
by
  sorry

end NUMINAMATH_GPT_width_of_domain_of_g_l370_37026


namespace NUMINAMATH_GPT_sin_cos_eq_one_l370_37048

theorem sin_cos_eq_one (x : ℝ) (hx0 : 0 ≤ x) (hx2pi : x < 2 * Real.pi) :
  (Real.sin x - Real.cos x = 1) ↔ (x = Real.pi / 2 ∨ x = Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_eq_one_l370_37048


namespace NUMINAMATH_GPT_total_fencing_cost_l370_37004

theorem total_fencing_cost
  (length : ℝ) 
  (breadth : ℝ)
  (cost_per_meter : ℝ)
  (h1 : length = 61)
  (h2 : length = breadth + 22)
  (h3 : cost_per_meter = 26.50) : 
  2 * (length + breadth) * cost_per_meter = 5300 := 
by 
  sorry

end NUMINAMATH_GPT_total_fencing_cost_l370_37004


namespace NUMINAMATH_GPT_eugene_total_pencils_l370_37011

def initial_pencils : ℕ := 51
def additional_pencils : ℕ := 6
def total_pencils : ℕ := initial_pencils + additional_pencils

theorem eugene_total_pencils : total_pencils = 57 := by
  sorry

end NUMINAMATH_GPT_eugene_total_pencils_l370_37011


namespace NUMINAMATH_GPT_expression_equals_64_l370_37050

theorem expression_equals_64 :
  let a := 2^3 + 2^3
  let b := 2^3 * 2^3
  let c := (2^3)^3
  let d := 2^12 / 2^2
  b = 2^6 :=
by
  sorry

end NUMINAMATH_GPT_expression_equals_64_l370_37050


namespace NUMINAMATH_GPT_maximum_value_of_f_l370_37085

noncomputable def f (a x : ℝ) : ℝ := (1 + x) ^ a - a * x

theorem maximum_value_of_f (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  ∃ x : ℝ, x > -1 ∧ ∀ y : ℝ, y > -1 → f a y ≤ f a x ∧ f a x = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_maximum_value_of_f_l370_37085


namespace NUMINAMATH_GPT_geometric_implies_b_squared_eq_ac_not_geometric_if_all_zero_sufficient_but_not_necessary_condition_l370_37099

variable (a b c : ℝ)

theorem geometric_implies_b_squared_eq_ac
  (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ ∃ r : ℝ, b = r * a ∧ c = r * b) :
  b^2 = a * c :=
by
  sorry

theorem not_geometric_if_all_zero 
  (hz : a = 0 ∧ b = 0 ∧ c = 0) : 
  ¬(∃ r : ℝ, b = r * a ∧ c = r * b) :=
by
  sorry

theorem sufficient_but_not_necessary_condition :
  (∃ r : ℝ, b = r * a ∧ c = r * b → b^2 = a * c) ∧ ¬(b^2 = a * c → ∃ r : ℝ, b = r * a ∧ c = r * b) :=
by
  sorry

end NUMINAMATH_GPT_geometric_implies_b_squared_eq_ac_not_geometric_if_all_zero_sufficient_but_not_necessary_condition_l370_37099


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l370_37078

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3
def g (k x : ℝ) : ℝ := k * x - 1

theorem sufficient_but_not_necessary_condition (k : ℝ) :
  (∀ x : ℝ, f x ≥ g k x) ↔ (-6 ≤ k ∧ k ≤ 2) :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l370_37078


namespace NUMINAMATH_GPT_tank_full_capacity_is_72_l370_37024

theorem tank_full_capacity_is_72 (x : ℝ) 
  (h1 : 0.9 * x - 0.4 * x = 36) : 
  x = 72 := 
sorry

end NUMINAMATH_GPT_tank_full_capacity_is_72_l370_37024


namespace NUMINAMATH_GPT_find_d_from_sine_wave_conditions_l370_37089

theorem find_d_from_sine_wave_conditions (a b d : ℝ) (h1 : d + a = 4) (h2 : d - a = -2) : d = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_d_from_sine_wave_conditions_l370_37089


namespace NUMINAMATH_GPT_sum_of_factorials_is_perfect_square_l370_37063

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def sum_of_factorials (n : ℕ) : ℕ :=
  (List.range (n + 1)).map factorial |>.sum

theorem sum_of_factorials_is_perfect_square (n : ℕ) (h : n > 0) :
  (∃ m : ℕ, m * m = sum_of_factorials n) ↔ (n = 1 ∨ n = 3) := 
sorry

end NUMINAMATH_GPT_sum_of_factorials_is_perfect_square_l370_37063


namespace NUMINAMATH_GPT_sandy_total_money_received_l370_37045

def sandy_saturday_half_dollars := 17
def sandy_sunday_half_dollars := 6
def half_dollar_value : ℝ := 0.50

theorem sandy_total_money_received :
  (sandy_saturday_half_dollars * half_dollar_value) +
  (sandy_sunday_half_dollars * half_dollar_value) = 11.50 :=
by
  sorry

end NUMINAMATH_GPT_sandy_total_money_received_l370_37045


namespace NUMINAMATH_GPT_binom_divisibility_l370_37003

theorem binom_divisibility (p : ℕ) (h₀ : Nat.Prime p) (h₁ : p % 2 = 1) : 
  (Nat.choose (2 * p - 1) (p - 1) - 1) % (p^2) = 0 := 
by 
  sorry

end NUMINAMATH_GPT_binom_divisibility_l370_37003


namespace NUMINAMATH_GPT_temperature_difference_l370_37013

theorem temperature_difference (high low : ℝ) (h_high : high = 5) (h_low : low = -3) :
  high - low = 8 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_temperature_difference_l370_37013


namespace NUMINAMATH_GPT_probability_calculation_l370_37017

noncomputable def probability_at_least_seven_at_least_three_times : ℚ :=
  let p := 1 / 4
  let q := 3 / 4
  (4 * p^3 * q) + (p^4)

theorem probability_calculation :
  probability_at_least_seven_at_least_three_times = 13 / 256 :=
by sorry

end NUMINAMATH_GPT_probability_calculation_l370_37017


namespace NUMINAMATH_GPT_ellipse_eccentricity_l370_37005

theorem ellipse_eccentricity (a b c : ℝ) (h_eq : a * a = 16) (h_b : b * b = 12) (h_c : c * c = a * a - b * b) :
  c / a = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_eccentricity_l370_37005


namespace NUMINAMATH_GPT_point_on_graph_of_inverse_proportion_l370_37058

theorem point_on_graph_of_inverse_proportion :
  ∃ x y : ℝ, (x = 2 ∧ y = 4) ∧ y = 8 / x :=
by
  sorry

end NUMINAMATH_GPT_point_on_graph_of_inverse_proportion_l370_37058


namespace NUMINAMATH_GPT_proof_problem_l370_37038

-- Definitions of points and vectors
def C : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (3, 4)
def N : ℝ × ℝ := (1, 0)
def M : ℝ × ℝ := (0, 1)

-- Definition of vector operations
def vector_sub (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  (p1.1 - p2.1, p1.2 - p2.2)

-- Vectors needed
def AC : ℝ × ℝ := vector_sub C A
def AM : ℝ × ℝ := vector_sub M A
def AN : ℝ × ℝ := vector_sub N A

-- The Lean proof statement
theorem proof_problem :
  (∃ (x y : ℝ), AC = (x * AM.1 + y * AN.1, x * AM.2 + y * AN.2) ∧
     (x, y) = (2 / 3, 1 / 2)) ∧
  (9 * (2 / 3:ℝ) ^ 2 + 16 * (1 / 2:ℝ) ^ 2 = 8) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l370_37038


namespace NUMINAMATH_GPT_fermat_prime_divisibility_l370_37037

def F (k : ℕ) : ℕ := 2 ^ 2 ^ k + 1

theorem fermat_prime_divisibility {m n : ℕ} (hmn : m > n) : F n ∣ (F m - 2) :=
sorry

end NUMINAMATH_GPT_fermat_prime_divisibility_l370_37037


namespace NUMINAMATH_GPT_unit_trip_to_expo_l370_37096

theorem unit_trip_to_expo (n : ℕ) (cost : ℕ) (total_cost : ℕ) :
  (n ≤ 30 → cost = 120) ∧ 
  (n > 30 → cost = 120 - 2 * (n - 30) ∧ cost ≥ 90) →
  (total_cost = 4000) →
  (total_cost = n * cost) →
  n = 40 :=
by
  sorry

end NUMINAMATH_GPT_unit_trip_to_expo_l370_37096


namespace NUMINAMATH_GPT_largest_lcm_l370_37035

theorem largest_lcm :
  max (max (max (max (Nat.lcm 18 4) (Nat.lcm 18 6)) (Nat.lcm 18 9)) (Nat.lcm 18 14)) (Nat.lcm 18 18) = 126 :=
by
  sorry

end NUMINAMATH_GPT_largest_lcm_l370_37035


namespace NUMINAMATH_GPT_find_xyz_sum_l370_37031

theorem find_xyz_sum
  (x y z : ℝ)
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : x^2 + x * y + y^2 = 108)
  (h2 : y^2 + y * z + z^2 = 16)
  (h3 : z^2 + z * x + x^2 = 124) :
  x * y + y * z + z * x = 48 := 
  sorry

end NUMINAMATH_GPT_find_xyz_sum_l370_37031


namespace NUMINAMATH_GPT_base_85_solution_l370_37070

theorem base_85_solution (b : ℕ) (h1 : 0 ≤ b ∧ b ≤ 16) :
  (352936524 - b) % 17 = 0 ↔ b = 4 :=
by
  sorry

end NUMINAMATH_GPT_base_85_solution_l370_37070


namespace NUMINAMATH_GPT_two_pow_2001_mod_127_l370_37091

theorem two_pow_2001_mod_127 : (2^2001) % 127 = 64 := 
by
  sorry

end NUMINAMATH_GPT_two_pow_2001_mod_127_l370_37091


namespace NUMINAMATH_GPT_tan_x_y_l370_37067

theorem tan_x_y (x y : ℝ) (h : Real.sin (2 * x + y) = 5 * Real.sin y) :
  Real.tan (x + y) = (3 / 2) * Real.tan x :=
sorry

end NUMINAMATH_GPT_tan_x_y_l370_37067


namespace NUMINAMATH_GPT_shopkeeper_gain_l370_37065

noncomputable def gain_percent (cost_per_kg : ℝ) (claimed_weight : ℝ) (actual_weight : ℝ) : ℝ :=
  let gain := cost_per_kg - (actual_weight / claimed_weight) * cost_per_kg
  (gain / ((actual_weight / claimed_weight) * cost_per_kg)) * 100

theorem shopkeeper_gain (c : ℝ) (cw aw : ℝ) (h : c = 1) (hw : cw = 1) (ha : aw = 0.75) : 
  gain_percent c cw aw = 33.33 :=
by sorry

end NUMINAMATH_GPT_shopkeeper_gain_l370_37065


namespace NUMINAMATH_GPT_maggie_bouncy_balls_l370_37054

theorem maggie_bouncy_balls (yellow_packs green_pack_given green_pack_bought : ℝ)
    (balls_per_pack : ℝ)
    (hy : yellow_packs = 8.0)
    (hg_given : green_pack_given = 4.0)
    (hg_bought : green_pack_bought = 4.0)
    (hbp : balls_per_pack = 10.0) :
    (yellow_packs * balls_per_pack + green_pack_bought * balls_per_pack - green_pack_given * balls_per_pack = 80.0) :=
by
  sorry

end NUMINAMATH_GPT_maggie_bouncy_balls_l370_37054


namespace NUMINAMATH_GPT_two_numbers_equal_l370_37020

variables {a b c : ℝ}
variable (h1 : a + b^2 + c^2 = a^2 + b + c^2)
variable (h2 : a^2 + b + c^2 = a^2 + b^2 + c)

theorem two_numbers_equal (h1 : a + b^2 + c^2 = a^2 + b + c^2) (h2 : a^2 + b + c^2 = a^2 + b^2 + c) :
  a = b ∨ a = c ∨ b = c :=
by
  sorry

end NUMINAMATH_GPT_two_numbers_equal_l370_37020


namespace NUMINAMATH_GPT_polar_to_rectangular_4sqrt2_pi_over_4_l370_37057

theorem polar_to_rectangular_4sqrt2_pi_over_4 :
  let r := 4 * Real.sqrt 2
  let θ := Real.pi / 4
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (x, y) = (4, 4) :=
by
  let r := 4 * Real.sqrt 2
  let θ := Real.pi / 4
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  sorry

end NUMINAMATH_GPT_polar_to_rectangular_4sqrt2_pi_over_4_l370_37057


namespace NUMINAMATH_GPT_moskvich_halfway_from_zhiguli_to_b_l370_37033

-- Define the Moskvich's and Zhiguli's speeds as real numbers
variables (u v : ℝ)

-- Define the given conditions as named hypotheses
axiom speed_condition : u = v
axiom halfway_condition : u = (1 / 2) * (u + v) 

-- The mathematical statement we want to prove
theorem moskvich_halfway_from_zhiguli_to_b (speed_condition : u = v) (halfway_condition : u = (1 / 2) * (u + v)) : 
  ∃ t : ℝ, t = 2 := 
sorry -- Proof omitted

end NUMINAMATH_GPT_moskvich_halfway_from_zhiguli_to_b_l370_37033


namespace NUMINAMATH_GPT_last_four_digits_of_5_pow_2018_l370_37059

theorem last_four_digits_of_5_pow_2018 : 
  (5^2018) % 10000 = 5625 :=
by {
  sorry
}

end NUMINAMATH_GPT_last_four_digits_of_5_pow_2018_l370_37059


namespace NUMINAMATH_GPT_binary_addition_and_subtraction_correct_l370_37049

def add_binary_and_subtract : ℕ :=
  let n1 := 0b1101  -- binary for 1101_2
  let n2 := 0b0010  -- binary for 10_2
  let n3 := 0b0101  -- binary for 101_2
  let n4 := 0b1011  -- expected result 1011_2
  n1 + n2 + n3 - 0b0011  -- subtract binary for 11_2

theorem binary_addition_and_subtraction_correct : add_binary_and_subtract = 0b1011 := 
by 
  sorry

end NUMINAMATH_GPT_binary_addition_and_subtraction_correct_l370_37049


namespace NUMINAMATH_GPT_intersection_M_N_l370_37061

-- Definitions for the sets M and N based on the given conditions
def M : Set ℝ := { x | x < 2 }
def N : Set ℝ := { x | x^2 - 3 * x - 4 ≤ 0 }

-- The statement we need to prove
theorem intersection_M_N : M ∩ N = { x | -1 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l370_37061


namespace NUMINAMATH_GPT_pizza_consumption_order_l370_37098

theorem pizza_consumption_order :
  let e := 1/6
  let s := 1/4
  let n := 1/3
  let o := 1/8
  let j := 1 - e - s - n - o
  (n > s) ∧ (s > e) ∧ (e = j) ∧ (j > o) :=
by
  sorry

end NUMINAMATH_GPT_pizza_consumption_order_l370_37098


namespace NUMINAMATH_GPT_solution_l370_37008

noncomputable def f : ℝ → ℝ := sorry

lemma problem_conditions:
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (-x + 1) = f (x + 1)) ∧ f (-1) = 1 :=
sorry

theorem solution : f 2017 = -1 :=
sorry

end NUMINAMATH_GPT_solution_l370_37008


namespace NUMINAMATH_GPT_chris_money_before_birthday_l370_37080

-- Define the given amounts of money from each source
def money_from_grandmother : ℕ := 25
def money_from_aunt_and_uncle : ℕ := 20
def money_from_parents : ℕ := 75
def total_money_now : ℕ := 279

-- Calculate the total birthday money
def total_birthday_money := money_from_grandmother + money_from_aunt_and_uncle + money_from_parents

-- Define the amount of money Chris had before his birthday
def money_before_birthday := total_money_now - total_birthday_money

-- The proof statement
theorem chris_money_before_birthday : money_before_birthday = 159 :=
by
  sorry

end NUMINAMATH_GPT_chris_money_before_birthday_l370_37080


namespace NUMINAMATH_GPT_inequality_solution_l370_37016

theorem inequality_solution (x : ℝ) : 
  x^2 - 9 * x + 20 < 1 ↔ (9 - Real.sqrt 5) / 2 < x ∧ x < (9 + Real.sqrt 5) / 2 := 
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l370_37016


namespace NUMINAMATH_GPT_inequality_solution_l370_37055

theorem inequality_solution (x : ℝ) (hx : x ≥ 0) : (2021 * (x ^ 10) - 1 ≥ 2020 * x) ↔ (x = 1) :=
sorry

end NUMINAMATH_GPT_inequality_solution_l370_37055


namespace NUMINAMATH_GPT_box_cubes_no_green_face_l370_37036

theorem box_cubes_no_green_face (a b c : ℕ) (h_a2 : a > 2) (h_b2 : b > 2) (h_c2 : c > 2)
  (h_no_green_face : (a-2)*(b-2)*(c-2) = (a*b*c) / 3) :
  (a, b, c) = (7, 30, 4) ∨ (a, b, c) = (8, 18, 4) ∨ (a, b, c) = (9, 14, 4) ∨
  (a, b, c) = (10, 12, 4) ∨ (a, b, c) = (5, 27, 5) ∨ (a, b, c) = (6, 12, 5) ∨
  (a, b, c) = (7, 9, 5) ∨ (a, b, c) = (6, 8, 6) :=
sorry

end NUMINAMATH_GPT_box_cubes_no_green_face_l370_37036


namespace NUMINAMATH_GPT_unique_root_in_interval_l370_37030

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 + x - 2

theorem unique_root_in_interval (n : ℤ) (h_root : ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0) :
  n = 1 := 
sorry

end NUMINAMATH_GPT_unique_root_in_interval_l370_37030


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l370_37076

theorem arithmetic_sequence_common_difference (a_n : ℕ → ℤ) (h1 : a_n 5 = 3) (h2 : a_n 6 = -2) : a_n 6 - a_n 5 = -5 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l370_37076


namespace NUMINAMATH_GPT_closest_point_on_parabola_to_line_l370_37079

noncomputable def line := { P : ℝ × ℝ | 2 * P.1 - P.2 = 4 }
noncomputable def parabola := { P : ℝ × ℝ | P.2 = P.1^2 }

theorem closest_point_on_parabola_to_line : 
  ∃ P : ℝ × ℝ, P ∈ parabola ∧ 
  (∀ Q ∈ parabola, ∀ R ∈ line, dist P R ≤ dist Q R) ∧ 
  P = (1, 1) := 
sorry

end NUMINAMATH_GPT_closest_point_on_parabola_to_line_l370_37079


namespace NUMINAMATH_GPT_inverse_at_neg_two_l370_37082

def g (x : ℝ) : ℝ := 5 * x^3 - 3

theorem inverse_at_neg_two :
  g (-2) = -43 :=
by
  -- sorry here to skip the proof, as instructed.
  sorry

end NUMINAMATH_GPT_inverse_at_neg_two_l370_37082


namespace NUMINAMATH_GPT_greatest_4_digit_base7_divisible_by_7_l370_37012

-- Definitions and conditions
def is_base7_number (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 7, d < 7

def is_4_digit_base7 (n : ℕ) : Prop :=
  is_base7_number n ∧ 343 ≤ n ∧ n < 2401 -- 343 = 7^3 (smallest 4-digit base 7) and 2401 = 7^4

def is_divisible_by_7 (n : ℕ) : Prop :=
  n % 7 = 0

-- Proof problem statement
theorem greatest_4_digit_base7_divisible_by_7 :
  ∃ (n : ℕ), is_4_digit_base7 n ∧ is_divisible_by_7 n ∧ n = 2346 :=
sorry

end NUMINAMATH_GPT_greatest_4_digit_base7_divisible_by_7_l370_37012


namespace NUMINAMATH_GPT_find_c_l370_37053

   noncomputable def c_value (c : ℝ) : Prop :=
     ∃ (x y : ℝ), (x^2 - 8*x + y^2 + 10*y + c = 0) ∧ (x - 4)^2 + (y + 5)^2 = 25

   theorem find_c (c : ℝ) : c_value c → c = 16 := by
     sorry
   
end NUMINAMATH_GPT_find_c_l370_37053


namespace NUMINAMATH_GPT_largest_divisor_of_expression_l370_37009

theorem largest_divisor_of_expression (x : ℤ) (h : x % 2 = 1) : 
  324 ∣ (12 * x + 3) * (12 * x + 9) * (6 * x + 6) :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_expression_l370_37009


namespace NUMINAMATH_GPT_product_of_D_coordinates_l370_37041

theorem product_of_D_coordinates 
  (M D : ℝ × ℝ)
  (C : ℝ × ℝ)
  (hC : C = (5, 3))
  (hM : M = (3, 7))
  (h_midpoint : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) : 
  D.1 * D.2 = 11 :=
by
  sorry

end NUMINAMATH_GPT_product_of_D_coordinates_l370_37041


namespace NUMINAMATH_GPT_inequality_solution_l370_37042

noncomputable def condition (x : ℝ) : Prop :=
  2 * Real.cos x ≤ Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))
  ∧ Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)) ≤ Real.sqrt 2

theorem inequality_solution (x : ℝ) (h₀ : 0 ≤ x ∧ x ≤ 2 * Real.pi) (h₁ : condition x) :
  Real.cos x ≤ Real.sqrt (2:ℝ) / 2 ∧ x ∈ [Real.pi/4, 7 * Real.pi/4] := sorry

end NUMINAMATH_GPT_inequality_solution_l370_37042


namespace NUMINAMATH_GPT_lampshire_parade_group_max_members_l370_37092

theorem lampshire_parade_group_max_members 
  (n : ℕ) 
  (h1 : 30 * n % 31 = 7)
  (h2 : 30 * n % 17 = 0)
  (h3 : 30 * n < 1500) :
  30 * n = 1020 :=
sorry

end NUMINAMATH_GPT_lampshire_parade_group_max_members_l370_37092


namespace NUMINAMATH_GPT_great_white_shark_teeth_is_420_l370_37066

-- Define the number of teeth in a tiger shark
def tiger_shark_teeth : ℕ := 180

-- Define the number of teeth in a hammerhead shark based on the tiger shark's teeth
def hammerhead_shark_teeth : ℕ := tiger_shark_teeth / 6

-- Define the number of teeth in a great white shark based on the sum of tiger and hammerhead shark's teeth
def great_white_shark_teeth : ℕ := 2 * (tiger_shark_teeth + hammerhead_shark_teeth)

-- The theorem statement that we need to prove
theorem great_white_shark_teeth_is_420 : great_white_shark_teeth = 420 :=
by
  -- Provide space for the proof
  sorry

end NUMINAMATH_GPT_great_white_shark_teeth_is_420_l370_37066


namespace NUMINAMATH_GPT_circle_area_is_323pi_l370_37044

-- Define points A and B
def A : ℝ × ℝ := (2, 9)
def B : ℝ × ℝ := (14, 7)

-- Define that points A and B lie on circle ω
def on_circle_omega (A B C : ℝ × ℝ) (r : ℝ) : Prop :=
  (A.1 - C.1) ^ 2 + (A.2 - C.2) ^ 2 = r ^ 2 ∧
  (B.1 - C.1) ^ 2 + (B.2 - C.2) ^ 2 = r ^ 2

-- Define the tangent lines intersect at a point on the x-axis
def tangents_intersect_on_x_axis (A B : ℝ × ℝ) (C : ℝ × ℝ) (ω : (ℝ × ℝ) → ℝ): Prop := 
  ∃ x : ℝ, (A.1 - C.1) ^ 2 + (A.2 - C.2) ^ 2 = (B.1 - C.1) ^ 2 + (B.2 - C.2) ^ 2 ∧
             C.2 = 0

-- Problem statement to prove
theorem circle_area_is_323pi (C : ℝ × ℝ) (radius : ℝ) (on_circle_omega : on_circle_omega A B C radius)
  (tangents_intersect_on_x_axis : tangents_intersect_on_x_axis A B C omega) :
  π * radius ^ 2 = 323 * π :=
sorry

end NUMINAMATH_GPT_circle_area_is_323pi_l370_37044


namespace NUMINAMATH_GPT_player1_wins_game_533_player1_wins_game_1000_l370_37021

-- Defining a structure for the game conditions
structure Game :=
  (target_sum : ℕ)
  (player1_wins_optimal : Bool)

-- Definition of the game scenarios
def game_533 := Game.mk 533 true
def game_1000 := Game.mk 1000 true

-- Theorem statements for the respective games
theorem player1_wins_game_533 : game_533.player1_wins_optimal :=
by sorry

theorem player1_wins_game_1000 : game_1000.player1_wins_optimal :=
by sorry

end NUMINAMATH_GPT_player1_wins_game_533_player1_wins_game_1000_l370_37021


namespace NUMINAMATH_GPT_find_multiple_l370_37093

-- Define the conditions
def ReetaPencils : ℕ := 20
def TotalPencils : ℕ := 64

-- Define the question and proof statement
theorem find_multiple (AnikaPencils : ℕ) (M : ℕ) :
  AnikaPencils = ReetaPencils * M + 4 →
  AnikaPencils + ReetaPencils = TotalPencils →
  M = 2 :=
by
  intros hAnika hTotal
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_find_multiple_l370_37093


namespace NUMINAMATH_GPT_range_of_a_l370_37074

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (a - 1 / Real.exp x)

theorem range_of_a (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ deriv (f a) x₁ = 0 ∧ deriv (f a) x₂ = 0) ↔ -1 / Real.exp 2 < a ∧ a < 0 := 
sorry

end NUMINAMATH_GPT_range_of_a_l370_37074


namespace NUMINAMATH_GPT_remainder_of_five_consecutive_odds_mod_12_l370_37081

/-- Let x be an odd integer. Prove that (x + (x + 2) + (x + 4) + (x + 6) + (x + 8)) % 12 = 9 
    when x ≡ 5 (mod 12). -/
theorem remainder_of_five_consecutive_odds_mod_12 {x : ℤ} (h : x % 12 = 5) :
  (x + (x + 2) + (x + 4) + (x + 6) + (x + 8)) % 12 = 9 :=
sorry

end NUMINAMATH_GPT_remainder_of_five_consecutive_odds_mod_12_l370_37081


namespace NUMINAMATH_GPT_other_group_land_l370_37052

def total_land : ℕ := 900
def remaining_land : ℕ := 385
def lizzies_group_land : ℕ := 250

theorem other_group_land :
  total_land - remaining_land - lizzies_group_land = 265 :=
by
  sorry

end NUMINAMATH_GPT_other_group_land_l370_37052


namespace NUMINAMATH_GPT_range_of_x_l370_37094

-- Defining the propositions p and q
def p (x : ℝ) : Prop := x^2 + 2 * x - 3 > 0
def q (x : ℝ) : Prop := 1 / (3 - x) > 1

-- Theorem statement
theorem range_of_x (x : ℝ) : (¬ q x ∧ p x) → (x ≥ 3 ∨ (1 < x ∧ x ≤ 2) ∨ x < -3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l370_37094


namespace NUMINAMATH_GPT_cows_total_l370_37069

theorem cows_total (A M R : ℕ) (h1 : A = 4 * M) (h2 : M = 60) (h3 : A + M = R + 30) : 
  A + M + R = 570 := by
  sorry

end NUMINAMATH_GPT_cows_total_l370_37069


namespace NUMINAMATH_GPT_rectangle_fraction_l370_37084

noncomputable def side_of_square : ℝ := Real.sqrt 900
noncomputable def radius_of_circle : ℝ := side_of_square
noncomputable def area_of_rectangle : ℝ := 120
noncomputable def breadth_of_rectangle : ℝ := 10
noncomputable def length_of_rectangle : ℝ := area_of_rectangle / breadth_of_rectangle
noncomputable def fraction : ℝ := length_of_rectangle / radius_of_circle

theorem rectangle_fraction :
  (length_of_rectangle / radius_of_circle) = (2 / 5) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_fraction_l370_37084


namespace NUMINAMATH_GPT_circles_5_and_8_same_color_l370_37032

-- Define the circles and colors
inductive Color
  | red
  | yellow
  | blue

def circles : Nat := 8

-- Define the adjacency relationship (i.e., directly connected)
-- This is a placeholder. In practice, this would be defined based on the problem's diagram.
def directly_connected (c1 c2 : Nat) : Prop := sorry

-- Simulate painting circles with given constraints
def painted (c : Nat) : Color := sorry

-- Define the conditions
axiom paint_condition (c1 c2 : Nat) (h : directly_connected c1 c2) : painted c1 ≠ painted c2

-- The proof problem: show that circles 5 and 8 must be painted the same color
theorem circles_5_and_8_same_color : painted 5 = painted 8 := 
sorry

end NUMINAMATH_GPT_circles_5_and_8_same_color_l370_37032


namespace NUMINAMATH_GPT_nina_money_l370_37073

-- Definitions based on the problem's conditions
def original_widgets := 15
def reduced_widgets := 25
def price_reduction := 5

-- The statement
theorem nina_money : 
  ∃ (W : ℝ), 15 * W = 25 * (W - 5) ∧ 15 * W = 187.5 :=
by
  sorry

end NUMINAMATH_GPT_nina_money_l370_37073


namespace NUMINAMATH_GPT_parallel_to_a_perpendicular_to_a_l370_37006

-- Definition of vectors a and b and conditions
def a : ℝ × ℝ := (3, 4)
def b (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Mathematical statement for Problem (1)
theorem parallel_to_a (x y : ℝ) (h : b x y) (h_parallel : 3 * y - 4 * x = 0) :
  (x = 3/5 ∧ y = 4/5) ∨ (x = -3/5 ∧ y = -4/5) := 
sorry

-- Mathematical statement for Problem (2)
theorem perpendicular_to_a (x y : ℝ) (h : b x y) (h_perpendicular : 3 * x + 4 * y = 0) :
  (x = -4/5 ∧ y = 3/5) ∨ (x = 4/5 ∧ y = -3/5) := 
sorry

end NUMINAMATH_GPT_parallel_to_a_perpendicular_to_a_l370_37006


namespace NUMINAMATH_GPT_Maria_score_l370_37014

theorem Maria_score (x : ℝ) (y : ℝ) (h1 : x = y + 50) (h2 : (x + y) / 2 = 105) : x = 130 :=
by
  sorry

end NUMINAMATH_GPT_Maria_score_l370_37014


namespace NUMINAMATH_GPT_mask_digit_identification_l370_37018

theorem mask_digit_identification :
  ∃ (elephant_mask mouse_mask pig_mask panda_mask : ℕ),
    (4 * 4 = 16) ∧
    (7 * 7 = 49) ∧
    (8 * 8 = 64) ∧
    (9 * 9 = 81) ∧
    elephant_mask = 6 ∧
    mouse_mask = 4 ∧
    pig_mask = 8 ∧
    panda_mask = 1 :=
by
  sorry

end NUMINAMATH_GPT_mask_digit_identification_l370_37018


namespace NUMINAMATH_GPT_circle_eq_tangent_x_axis_l370_37023

theorem circle_eq_tangent_x_axis (h k r : ℝ) (x y : ℝ)
  (h_center : h = -5)
  (k_center : k = 4)
  (tangent_x_axis : r = 4) :
  (x + 5)^2 + (y - 4)^2 = 16 :=
sorry

end NUMINAMATH_GPT_circle_eq_tangent_x_axis_l370_37023


namespace NUMINAMATH_GPT_nails_painted_purple_l370_37083

variable (P S : ℕ)

theorem nails_painted_purple :
  (P + 8 + S = 20) ∧ ((8 / 20 : ℚ) * 100 - (S / 20 : ℚ) * 100 = 10) → P = 6 :=
by
  sorry

end NUMINAMATH_GPT_nails_painted_purple_l370_37083


namespace NUMINAMATH_GPT_number_of_liars_l370_37060

theorem number_of_liars {n : ℕ} (h1 : n ≥ 1) (h2 : n ≤ 200) (h3 : ∃ k : ℕ, k < n ∧ k ≥ 1) : 
  (∃ l : ℕ, l = 199 ∨ l = 200) := 
sorry

end NUMINAMATH_GPT_number_of_liars_l370_37060


namespace NUMINAMATH_GPT_remainder_when_ab_div_by_40_l370_37010

theorem remainder_when_ab_div_by_40 (a b : ℤ) (k j : ℤ)
  (ha : a = 80 * k + 75)
  (hb : b = 90 * j + 85):
  (a + b) % 40 = 0 :=
by sorry

end NUMINAMATH_GPT_remainder_when_ab_div_by_40_l370_37010


namespace NUMINAMATH_GPT_distinct_units_digits_perfect_cube_l370_37072

theorem distinct_units_digits_perfect_cube : 
  (∀ (d : ℕ), d < 10 → ( ∃ (n : ℕ), (n % 10 = d) ∧ (∃ (m : ℕ), (m ^ 3 % 10 = d) ))) ↔
  (∃ (digits : Finset ℕ), digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ digits.card = 10) := 
sorry

end NUMINAMATH_GPT_distinct_units_digits_perfect_cube_l370_37072


namespace NUMINAMATH_GPT_greatest_odd_factors_l370_37029

theorem greatest_odd_factors (n : ℕ) : n < 200 ∧ (∃ m : ℕ, m * m = n) → n = 196 := by
  sorry

end NUMINAMATH_GPT_greatest_odd_factors_l370_37029


namespace NUMINAMATH_GPT_overall_percentage_decrease_l370_37075

-- Define the initial pay cut percentages as given in the conditions.
def first_pay_cut := 5.25 / 100
def second_pay_cut := 9.75 / 100
def third_pay_cut := 14.6 / 100
def fourth_pay_cut := 12.8 / 100

-- Define the single shot percentage decrease we want to prove.
def single_shot_decrease := 36.73 / 100

-- Calculate the cumulative multiplier from individual pay cuts.
def cumulative_multiplier := 
  (1 - first_pay_cut) * (1 - second_pay_cut) * (1 - third_pay_cut) * (1 - fourth_pay_cut)

-- Statement: Prove the overall percentage decrease using cumulative multiplier is equal to single shot decrease.
theorem overall_percentage_decrease :
  1 - cumulative_multiplier = single_shot_decrease :=
by sorry

end NUMINAMATH_GPT_overall_percentage_decrease_l370_37075


namespace NUMINAMATH_GPT_range_of_x1_f_x2_l370_37077

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 4 * Real.exp 1 else Real.exp x / x^2

theorem range_of_x1_f_x2:
  ∃ (x1 x2 : ℝ), x1 ≤ 0 ∧ 0 < x2 ∧ f x1 = f x2 ∧ -4 * (Real.exp 1)^2 ≤ x1 * f x2 ∧ x1 * f x2 ≤ 0 :=
sorry

end NUMINAMATH_GPT_range_of_x1_f_x2_l370_37077


namespace NUMINAMATH_GPT_find_m_l370_37019

theorem find_m (m : ℝ) : (∀ x y : ℝ, x^2 + y^2 - 2 * y - 4 = 0) →
  (∀ x y : ℝ, x - 2 * y + m = 0) →
  (m = 7 ∨ m = -3) :=
by
  sorry

end NUMINAMATH_GPT_find_m_l370_37019


namespace NUMINAMATH_GPT_value_of_a_l370_37022

theorem value_of_a (a : ℝ) (h : (a ^ 3) * ((5).choose (2)) = 80) : a = 2 :=
  sorry

end NUMINAMATH_GPT_value_of_a_l370_37022


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l370_37001

-- Define the variables and conditions
def a : ℕ := 71
def d : ℕ := 2
def l : ℕ := 99

-- Calculate the number of terms in the sequence
def n : ℕ := ((l - a) / d) + 1

-- Define the sum of the arithmetic sequence
def S : ℕ := (n * (a + l)) / 2

-- Statement to be proven
theorem arithmetic_sequence_sum :
  3 * S = 3825 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l370_37001


namespace NUMINAMATH_GPT_proof_problem_l370_37056

variable {R : Type*} [Field R] {x y z w N : R}

theorem proof_problem 
  (h1 : 4 * x * z + y * w = N)
  (h2 : x * w + y * z = 6)
  (h3 : (2 * x + y) * (2 * z + w) = 15) :
  N = 3 :=
by sorry

end NUMINAMATH_GPT_proof_problem_l370_37056


namespace NUMINAMATH_GPT_compound_difference_l370_37015

noncomputable def monthly_compound_amount (principal : ℝ) (annual_rate : ℝ) (years : ℝ) : ℝ :=
  let monthly_rate := annual_rate / 12
  let periods := 12 * years
  principal * (1 + monthly_rate) ^ periods

noncomputable def semi_annual_compound_amount (principal : ℝ) (annual_rate : ℝ) (years : ℝ) : ℝ :=
  let semi_annual_rate := annual_rate / 2
  let periods := 2 * years
  principal * (1 + semi_annual_rate) ^ periods

theorem compound_difference (principal : ℝ) (annual_rate : ℝ) (years : ℝ) :
  monthly_compound_amount principal annual_rate years - semi_annual_compound_amount principal annual_rate years = 23.36 :=
by
  let principal := 8000
  let annual_rate := 0.08
  let years := 3
  sorry

end NUMINAMATH_GPT_compound_difference_l370_37015


namespace NUMINAMATH_GPT_n_c_equation_l370_37028

theorem n_c_equation (n c : ℕ) (hn : 0 < n) (hc : 0 < c) :
  (∀ x : ℕ, (↑x + n * ↑x / 100) * (1 - c / 100) = x) →
  (n^2 / c^2 = (100 + n) / (100 - c)) :=
by sorry

end NUMINAMATH_GPT_n_c_equation_l370_37028


namespace NUMINAMATH_GPT_correct_substitution_l370_37025

theorem correct_substitution (x y : ℝ) (h1 : y = 1 - x) (h2 : x - 2 * y = 4) : x - 2 * (1 - x) = 4 → x - 2 + 2 * x = 4 := by
  sorry

end NUMINAMATH_GPT_correct_substitution_l370_37025


namespace NUMINAMATH_GPT_find_missing_coordinates_l370_37062

def parallelogram_area (A B : ℝ × ℝ) (C D : ℝ × ℝ) : ℝ :=
  abs ((B.1 - A.1) * (D.2 - A.2))

theorem find_missing_coordinates :
  ∃ (x y : ℝ), (x, y) ≠ (4, 4) ∧ (x, y) ≠ (5, 9) ∧ (x, y) ≠ (8, 9) ∧
  parallelogram_area (4, 4) (5, 9) (8, 9) (x, y) = 5 :=
sorry

end NUMINAMATH_GPT_find_missing_coordinates_l370_37062
