import Mathlib

namespace greatest_divisor_of_sum_of_arith_seq_l287_287355

theorem greatest_divisor_of_sum_of_arith_seq (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → d ∣ (15 * (x + 7 * c))) ∧
    (∀ k : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → k ∣ (15 * (x + 7 * c))) → k ≤ d) ∧ 
    d = 15 :=
sorry

end greatest_divisor_of_sum_of_arith_seq_l287_287355


namespace greatest_divisor_of_sum_of_arith_seq_l287_287352

theorem greatest_divisor_of_sum_of_arith_seq (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → d ∣ (15 * (x + 7 * c))) ∧
    (∀ k : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → k ∣ (15 * (x + 7 * c))) → k ≤ d) ∧ 
    d = 15 :=
sorry

end greatest_divisor_of_sum_of_arith_seq_l287_287352


namespace first_cyclist_speed_l287_287933

theorem first_cyclist_speed (v₁ v₂ : ℕ) (c t : ℕ) 
  (h1 : v₂ = 8) 
  (h2 : c = 675) 
  (h3 : t = 45) 
  (h4 : v₁ * t + v₂ * t = c) : 
  v₁ = 7 :=
by {
  sorry
}

end first_cyclist_speed_l287_287933


namespace fill_cistern_7_2_hours_l287_287810

theorem fill_cistern_7_2_hours :
  let R_fill := 1 / 4
  let R_empty := 1 / 9
  R_fill - R_empty = 5 / 36 →
  1 / (R_fill - R_empty) = 7.2 := 
by
  intros
  sorry

end fill_cistern_7_2_hours_l287_287810


namespace exist_unique_rectangular_prism_Q_l287_287844

variable (a b c : ℝ) (h_lt : a < b ∧ b < c)
variable (x y z : ℝ) (hx_lt : x < y ∧ y < z ∧ z < a)

theorem exist_unique_rectangular_prism_Q :
  (2 * (x*y + y*z + z*x) = 0.5 * (a*b + b*c + c*a) ∧ x*y*z = 0.25 * a*b*c) ∧ (x < y ∧ y < z ∧ z < a) → 
  ∃! x y z, (2 * (x*y + y*z + z*x) = 0.5 * (a*b + b*c + c*a) ∧ x*y*z = 0.25 * a*b*c) :=
sorry

end exist_unique_rectangular_prism_Q_l287_287844


namespace value_of_a_plus_b_l287_287843

theorem value_of_a_plus_b (a b : ℝ) : (|a - 1| + (b + 3)^2 = 0) → (a + b = -2) :=
by
  sorry

end value_of_a_plus_b_l287_287843


namespace find_quadratic_expression_l287_287628

-- Define the quadratic function
def quadratic (a b c x : ℝ) := a * x^2 + b * x + c

-- Define conditions
def intersects_x_axis_at_A (a b c : ℝ) : Prop :=
  quadratic a b c (-2) = 0

def intersects_x_axis_at_B (a b c : ℝ) : Prop :=
  quadratic a b c (1) = 0

def has_maximum_value (a : ℝ) : Prop :=
  a < 0

-- Define the target function
def f_expr (x : ℝ) : ℝ := -x^2 - x + 2

-- The theorem to be proved
theorem find_quadratic_expression :
  ∃ a b c, 
    intersects_x_axis_at_A a b c ∧
    intersects_x_axis_at_B a b c ∧
    has_maximum_value a ∧
    ∀ x, quadratic a b c x = f_expr x :=
sorry

end find_quadratic_expression_l287_287628


namespace g_ten_l287_287551

-- Define the function g and its properties
def g : ℝ → ℝ := sorry

axiom g_property1 : ∀ x y : ℝ, g (x * y) = 2 * g x * g y
axiom g_property2 : g 0 = 2

-- Prove that g 10 = 1 / 2
theorem g_ten : g 10 = 1 / 2 :=
by
  sorry

end g_ten_l287_287551


namespace avg_divisible_by_4_between_15_and_55_eq_34_l287_287683

theorem avg_divisible_by_4_between_15_and_55_eq_34 :
  let numbers := (List.filter (λ x => x % 4 = 0) (List.range' 16 37))
  (List.sum numbers) / (numbers.length) = 34 := by
  sorry

end avg_divisible_by_4_between_15_and_55_eq_34_l287_287683


namespace reach_one_from_any_non_zero_l287_287441

-- Define the game rules as functions
def remove_units_digit (n : ℕ) : ℕ :=
  n / 10

def multiply_by_two (n : ℕ) : ℕ :=
  n * 2

-- Lemma: Prove that starting from 45, we can reach 1 using the game rules.
lemma reach_one_from_45 : ∃ f : ℕ → ℕ, f 45 = 1 := 
by {
  -- You can define the sequence explicitly or use the function definitions.
  sorry
}

-- Lemma: Prove that starting from 345, we can reach 1 using the game rules.
lemma reach_one_from_345 : ∃ f : ℕ → ℕ, f 345 = 1 := 
by {
  -- You can define the sequence explicitly or use the function definitions.
  sorry
}

-- Theorem: Prove that any non-zero natural number can be reduced to 1 using the game rules.
theorem reach_one_from_any_non_zero (n : ℕ) (h : n ≠ 0) : ∃ f : ℕ → ℕ, f n = 1 :=
by {
  sorry
}

end reach_one_from_any_non_zero_l287_287441


namespace cubic_polynomial_roots_3x3_minus_4x2_plus_220x_minus_7_l287_287455

theorem cubic_polynomial_roots_3x3_minus_4x2_plus_220x_minus_7 (p q r : ℝ)
  (h_roots : 3*p^3 - 4*p^2 + 220*p - 7 = 0 ∧ 3*q^3 - 4*q^2 + 220*q - 7 = 0 ∧ 3*r^3 - 4*r^2 + 220*r - 7 = 0)
  (h_vieta : p + q + r = 4 / 3) :
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = 64.556 :=
sorry

end cubic_polynomial_roots_3x3_minus_4x2_plus_220x_minus_7_l287_287455


namespace bryan_samples_l287_287229

noncomputable def initial_samples_per_shelf : ℕ := 128
noncomputable def shelves : ℕ := 13
noncomputable def samples_removed_per_shelf : ℕ := 2
noncomputable def remaining_samples_per_shelf := initial_samples_per_shelf - samples_removed_per_shelf
noncomputable def total_remaining_samples := remaining_samples_per_shelf * shelves

theorem bryan_samples : total_remaining_samples = 1638 := 
by 
  sorry

end bryan_samples_l287_287229


namespace quadratic_equal_roots_k_value_l287_287120

theorem quadratic_equal_roots_k_value (k : ℝ) :
  (∀ x : ℝ, x^2 - 8 * x - 4 * k = 0 → x^2 - 8 * x - 4 * k = 0 ∧ (0 : ℝ) = 0 ) →
  k = -4 :=
sorry

end quadratic_equal_roots_k_value_l287_287120


namespace solve_remainder_problem_l287_287217

def remainder_problem : Prop :=
  ∃ (n : ℕ), 
    (n % 481 = 179) ∧ 
    (n % 752 = 231) ∧ 
    (n % 1063 = 359) ∧ 
    (((179 + 231 - 359) % 37) = 14)

theorem solve_remainder_problem : remainder_problem :=
by
  sorry

end solve_remainder_problem_l287_287217


namespace integer_solutions_eq_0_or_2_l287_287197

theorem integer_solutions_eq_0_or_2 (a : ℤ) (x : ℤ) : 
  (a * x^2 + 6 = 0) → (a = -6 ∧ (x = 1 ∨ x = -1)) ∨ (¬ (a = -6) ∧ (x ≠ 1) ∧ (x ≠ -1)) :=
by 
sorry

end integer_solutions_eq_0_or_2_l287_287197


namespace Dan_must_exceed_speed_l287_287658

theorem Dan_must_exceed_speed (distance : ℝ) (Cara_speed : ℝ) (delay : ℝ) (time_Cara : ℝ) (Dan_time : ℝ) : 
  distance = 120 ∧ Cara_speed = 30 ∧ delay = 1 ∧ time_Cara = distance / Cara_speed ∧ time_Cara = 4 ∧ Dan_time = time_Cara - delay ∧ Dan_time < 4 → 
  (distance / Dan_time) > 40 :=
by
  sorry

end Dan_must_exceed_speed_l287_287658


namespace sum_binomials_l287_287682

-- Defining binomial coefficient
def binom (n k : ℕ) : ℕ := n.choose k

theorem sum_binomials : binom 12 4 + binom 10 3 = 615 :=
by
  -- Here we state the problem, and the proof will be left as 'sorry'.
  sorry

end sum_binomials_l287_287682


namespace perimeter_remaining_shape_l287_287542

theorem perimeter_remaining_shape (length width square1 square2 : ℝ) 
  (H_len : length = 50) (H_width : width = 20) 
  (H_sq1 : square1 = 12) (H_sq2 : square2 = 4) : 
  2 * (length + width) + 4 * (square1 + square2) = 204 :=
by 
  rw [H_len, H_width, H_sq1, H_sq2]
  sorry

end perimeter_remaining_shape_l287_287542


namespace functional_equation_solution_l287_287400

-- Define the function
def f : ℝ → ℝ := sorry

-- The main theorem to prove
theorem functional_equation_solution :
  (∀ x y : ℝ, f (x * f y + 1) = y + f (f x * f y)) → (∀ x : ℝ, f x = x - 1) :=
by
  intro h
  sorry

end functional_equation_solution_l287_287400


namespace sum_nonpositive_inequality_l287_287659

theorem sum_nonpositive_inequality (x : ℝ) : x + 5 ≤ 0 ↔ x + 5 ≤ 0 :=
by
  sorry

end sum_nonpositive_inequality_l287_287659


namespace locus_of_Q_max_area_of_triangle_OPQ_l287_287943

open Real

theorem locus_of_Q (x y : ℝ) (x_0 y_0 : ℝ) :
  (x_0 / 4)^2 + (y_0 / 3)^2 = 1 ∧
  x = 3 * x_0 ∧ y = 4 * y_0 →
  (x / 6)^2 + (y / 4)^2 = 1 :=
sorry

theorem max_area_of_triangle_OPQ (S : ℝ) (x_0 y_0 : ℝ) :
  (x_0 / 4)^2 + (y_0 / 3)^2 = 1 ∧
  x_0 > 0 ∧ y_0 > 0 →
  S <= sqrt 3 / 2 :=
sorry

end locus_of_Q_max_area_of_triangle_OPQ_l287_287943


namespace probability_same_color_correct_l287_287228

-- Defining the contents of Bag A and Bag B
def bagA : List (String × ℕ) := [("white", 1), ("red", 2), ("black", 3)]
def bagB : List (String × ℕ) := [("white", 2), ("red", 3), ("black", 1)]

-- The probability calculation
noncomputable def probability_same_color (bagA bagB : List (String × ℕ)) : ℚ :=
  let p_white := (1 / 6 : ℚ) * (1 / 3 : ℚ)
  let p_red := (1 / 3 : ℚ) * (1 / 2 : ℚ)
  let p_black := (1 / 2 : ℚ) * (1 / 6 : ℚ)
  p_white + p_red + p_black

-- Proof problem statement
theorem probability_same_color_correct :
  probability_same_color bagA bagB = 11 / 36 := 
by 
  sorry

end probability_same_color_correct_l287_287228


namespace mrs_smith_strawberries_l287_287912

theorem mrs_smith_strawberries (girls : ℕ) (strawberries_per_girl : ℕ) 
                                (h1 : girls = 8) (h2 : strawberries_per_girl = 6) :
    girls * strawberries_per_girl = 48 := by
  sorry

end mrs_smith_strawberries_l287_287912


namespace log_expression_equality_l287_287250

noncomputable def evaluate_log_expression : Real :=
  let log4_8 := (Real.log 8) / (Real.log 4)
  let log5_10 := (Real.log 10) / (Real.log 5)
  Real.sqrt (log4_8 + log5_10)

theorem log_expression_equality : 
  evaluate_log_expression = Real.sqrt ((5 / 2) + (Real.log 2 / Real.log 5)) :=
by
  sorry

end log_expression_equality_l287_287250


namespace mix_alcohol_solutions_l287_287168

-- Definitions capturing the conditions from part (a)
def volume_solution_y : ℝ := 600
def percent_alcohol_x : ℝ := 0.1
def percent_alcohol_y : ℝ := 0.3
def desired_percent_alcohol : ℝ := 0.25

-- The resulting Lean statement to prove question == answer given conditions
theorem mix_alcohol_solutions (Vx : ℝ) (h : (percent_alcohol_x * Vx + percent_alcohol_y * volume_solution_y) / (Vx + volume_solution_y) = desired_percent_alcohol) : Vx = 200 :=
sorry

end mix_alcohol_solutions_l287_287168


namespace regions_of_diagonals_formula_l287_287144

def regions_of_diagonals (n : ℕ) : ℕ :=
  ((n - 1) * (n - 2) * (n * n - 3 * n + 12)) / 24

theorem regions_of_diagonals_formula (n : ℕ) (h : 3 ≤ n) :
  ∃ (fn : ℕ), fn = regions_of_diagonals n := by
  sorry

end regions_of_diagonals_formula_l287_287144


namespace smallest_positive_cube_ends_in_112_l287_287564

theorem smallest_positive_cube_ends_in_112 :
  ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 112 ∧ (∀ m : ℕ, (m > 0 ∧ m^3 % 1000 = 112) → n ≤ m) :=
by
  sorry

end smallest_positive_cube_ends_in_112_l287_287564


namespace smith_oldest_child_age_l287_287007

theorem smith_oldest_child_age
  (avg_age : ℕ)
  (youngest : ℕ)
  (middle : ℕ)
  (oldest : ℕ)
  (h1 : avg_age = 9)
  (h2 : youngest = 6)
  (h3 : middle = 8)
  (h4 : (youngest + middle + oldest) / 3 = avg_age) :
  oldest = 13 :=
by
  sorry

end smith_oldest_child_age_l287_287007


namespace time_to_cross_signal_pole_l287_287807

-- Given conditions
def length_of_train : ℝ := 300
def time_to_cross_platform : ℝ := 39
def length_of_platform : ℝ := 1162.5

-- The question to prove
theorem time_to_cross_signal_pole :
  (length_of_train / ((length_of_train + length_of_platform) / time_to_cross_platform)) = 8 :=
by
  sorry

end time_to_cross_signal_pole_l287_287807


namespace matrix_multiplication_example_l287_287980

def matrix1 : Matrix (Fin 2) (Fin 2) ℤ := ![![3, -2], ![-4, 5]]
def vector1 : Fin 2 → ℤ := ![4, -2]
def scalar : ℤ := 2
def result : Fin 2 → ℤ := ![32, -52]

theorem matrix_multiplication_example :
  scalar • (matrix1.mulVec vector1) = result := by
  sorry

end matrix_multiplication_example_l287_287980


namespace max_profit_at_60_l287_287219

variable (x : ℕ) (y W : ℝ)

def charter_fee : ℝ := 15000
def max_group_size : ℕ := 75

def ticket_price (x : ℕ) : ℝ :=
  if x ≤ 30 then 900
  else if 30 < x ∧ x ≤ max_group_size then -10 * (x - 30) + 900
  else 0

def profit (x : ℕ) : ℝ :=
  if x ≤ 30 then 900 * x - charter_fee
  else if 30 < x ∧ x ≤ max_group_size then (-10 * x + 1200) * x - charter_fee
  else 0

theorem max_profit_at_60 : x = 60 → profit x = 21000 := by
  sorry

end max_profit_at_60_l287_287219


namespace greatest_divisor_of_arithmetic_sequence_sum_l287_287341

theorem greatest_divisor_of_arithmetic_sequence_sum :
  ∀ (x c : ℕ), ∃ k : ℕ, k = 15 ∧ 15 ∣ (15 * x + 105 * c) :=
by
  intro x c
  exists 15
  split
  . rfl
  . sorry

end greatest_divisor_of_arithmetic_sequence_sum_l287_287341


namespace combinatorial_count_l287_287708

-- Define the function f_r(n, k) that we need to prove the correct answer for
def f_r (n k r : ℕ) : ℕ :=
  Nat.choose (n - k * r + r) k

-- Lean statement representing the proof problem
theorem combinatorial_count (n k r : ℕ) (h : n + r ≥ k * r + k) : 
  f_r n k r = Nat.choose (n - k * r + r) k := 
sorry

end combinatorial_count_l287_287708


namespace cylinder_height_l287_287388

theorem cylinder_height (r₁ r₂ : ℝ) (S : ℝ) (hR : r₁ = 3) (hL : r₂ = 4) (hS : S = 100 * Real.pi) : 
  (∃ h : ℝ, h = 7 ∨ h = 1) :=
by 
  sorry

end cylinder_height_l287_287388


namespace sum_series_l287_287069

noncomputable def f (n : ℕ) : ℝ :=
  (6 * (n : ℝ)^3 - 3 * (n : ℝ)^2 + 2 * (n : ℝ) - 1) / 
  ((n : ℝ) * ((n : ℝ) - 1) * ((n : ℝ)^2 + (n : ℝ) + 1) * ((n : ℝ)^2 - (n : ℝ) + 1))

theorem sum_series:
  (∑' n, if h : 2 ≤ n then f n else 0) = 1 := 
by
  sorry

end sum_series_l287_287069


namespace truncated_cone_volume_l287_287776

noncomputable def volume_of_truncated_cone (R r h : ℝ) : ℝ :=
  let V_large := (1 / 3) * Real.pi * R^2 * (h + h)  -- Height of larger cone is h + x = h + h
  let V_small := (1 / 3) * Real.pi * r^2 * h       -- Height of smaller cone is h
  V_large - V_small

theorem truncated_cone_volume (R r h : ℝ) (hR : R = 8) (hr : r = 4) (hh : h = 6) :
  volume_of_truncated_cone R r h = 224 * Real.pi :=
by
  sorry

end truncated_cone_volume_l287_287776


namespace simplest_fraction_sum_l287_287925

theorem simplest_fraction_sum (a b : ℕ) (h : Rat.mkP 428125 1000000 = Rat.mkP a b) (h_coprime : Nat.coprime a b) : a + b = 457 := 
sorry

end simplest_fraction_sum_l287_287925


namespace non_congruent_triangles_with_perimeter_11_l287_287863

theorem non_congruent_triangles_with_perimeter_11 :
  { t : ℕ × ℕ × ℕ // let (a, b, c) := t in a + b + c = 11 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a ≤ b ∧ b ≤ c }.card = 4 :=
sorry

end non_congruent_triangles_with_perimeter_11_l287_287863


namespace situps_ratio_l287_287297

theorem situps_ratio (ken_situps : ℕ) (nathan_situps : ℕ) (bob_situps : ℕ) :
  ken_situps = 20 →
  nathan_situps = 2 * ken_situps →
  bob_situps = ken_situps + 10 →
  (bob_situps : ℚ) / (ken_situps + nathan_situps : ℚ) = 1 / 2 :=
by
  sorry

end situps_ratio_l287_287297


namespace find_xy_pairs_l287_287700

theorem find_xy_pairs (x y: ℝ) :
  x + y + 4 = (12 * x + 11 * y) / (x ^ 2 + y ^ 2) ∧
  y - x + 3 = (11 * x - 12 * y) / (x ^ 2 + y ^ 2) ↔
  (x = 2 ∧ y = 1) ∨ (x = -2.5 ∧ y = -4.5) :=
by
  sorry

end find_xy_pairs_l287_287700


namespace tan_alpha_eq_neg_one_l287_287087

theorem tan_alpha_eq_neg_one (alpha : ℝ) (h1 : Real.tan alpha = -1) (h2 : 0 ≤ alpha ∧ alpha < Real.pi) :
  alpha = (3 * Real.pi) / 4 :=
sorry

end tan_alpha_eq_neg_one_l287_287087


namespace greater_number_l287_287492

theorem greater_number (a b : ℕ) (h1 : a + b = 36) (h2 : a - b = 8) : a = 22 :=
by
  sorry

end greater_number_l287_287492


namespace kate_collected_money_l287_287029

-- Define the conditions
def wand_cost : ℕ := 60
def num_wands_bought : ℕ := 3
def extra_charge : ℕ := 5
def num_wands_sold : ℕ := 2

-- Define the selling price per wand
def selling_price_per_wand : ℕ := wand_cost + extra_charge

-- Define the total amount collected from the sale
def total_collected : ℕ := num_wands_sold * selling_price_per_wand

-- Prove that the total collected is $130
theorem kate_collected_money :
  total_collected = 130 :=
sorry

end kate_collected_money_l287_287029


namespace lumberjack_trees_l287_287668

theorem lumberjack_trees (trees logs firewood : ℕ) 
  (h1 : ∀ t, logs = t * 4)
  (h2 : ∀ l, firewood = l * 5)
  (h3 : firewood = 500)
  : trees = 25 :=
by
  sorry

end lumberjack_trees_l287_287668


namespace total_charge_rush_hour_trip_l287_287733

def initial_fee : ℝ := 2.35
def non_rush_hour_cost_per_two_fifths_mile : ℝ := 0.35
def rush_hour_cost_increase_percentage : ℝ := 0.20
def traffic_delay_cost_per_mile : ℝ := 1.50
def distance_travelled : ℝ := 3.6

theorem total_charge_rush_hour_trip (initial_fee : ℝ) 
  (non_rush_hour_cost_per_two_fifths_mile : ℝ) 
  (rush_hour_cost_increase_percentage : ℝ)
  (traffic_delay_cost_per_mile : ℝ)
  (distance_travelled : ℝ) : 
  initial_fee = 2.35 → 
  non_rush_hour_cost_per_two_fifths_mile = 0.35 →
  rush_hour_cost_increase_percentage = 0.20 →
  traffic_delay_cost_per_mile = 1.50 →
  distance_travelled = 3.6 →
  (initial_fee + ((5/2) * (non_rush_hour_cost_per_two_fifths_mile * (1 + rush_hour_cost_increase_percentage))) * distance_travelled + (traffic_delay_cost_per_mile * distance_travelled)) = 11.53 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end total_charge_rush_hour_trip_l287_287733


namespace monomial_sum_mn_l287_287435

-- Define the conditions as Lean definitions
def is_monomial_sum (x y : ℕ) (m n : ℕ) : Prop :=
  ∃ k : ℕ, (x ^ 2) * (y ^ m) + (x ^ n) * (y ^ 3) = x ^ k

-- State our main theorem
theorem monomial_sum_mn (x y : ℕ) (m n : ℕ) (h : is_monomial_sum x y m n) : m + n = 5 :=
sorry  -- Completion of the proof is not required

end monomial_sum_mn_l287_287435


namespace move_left_is_negative_l287_287225

theorem move_left_is_negative (movement_right : ℝ) (h : movement_right = 3) : -movement_right = -3 := 
by 
  sorry

end move_left_is_negative_l287_287225


namespace false_statement_of_quadratic_l287_287566

-- Define the function f and the conditions
def f (a b c x : ℝ) := a * x^2 + b * x + c

theorem false_statement_of_quadratic (a b c x0 : ℝ) (h₀ : a > 0) (h₁ : 2 * a * x0 + b = 0) :
  ¬ ∀ x : ℝ, f a b c x ≤ f a b c x0 := by
  sorry

end false_statement_of_quadratic_l287_287566


namespace find_cookies_on_second_plate_l287_287775

theorem find_cookies_on_second_plate (a : ℕ → ℕ) :
  (a 1 = 5) ∧ (a 3 = 10) ∧ (a 4 = 14) ∧ (a 5 = 19) ∧ (a 6 = 25) ∧
  (∀ n, a (n + 2) - a (n + 1) = if (n + 1) % 2 = 0 then 5 else 4) →
  a 2 = 5 :=
by
  sorry

end find_cookies_on_second_plate_l287_287775


namespace greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l287_287370

theorem greatest_divisor_of_sum_first_15_terms_arithmetic_sequence
  (x c : ℕ) -- where x and c are positive integers
  (h_pos_x : 0 < x) -- x is positive
  (h_pos_c : 0 < c) -- c is positive
  : ∃ (d : ℕ), d = 15 ∧ ∀ (S : ℕ), S = 15 * x + 105 * c → d ∣ S :=
by
  sorry

end greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l287_287370


namespace find_johns_allowance_l287_287377

variable (A : ℝ)  -- John's weekly allowance

noncomputable def johns_allowance : Prop :=
  let arcade_spent := (3 / 5) * A
  let remaining_after_arcade := (2 / 5) * A
  let toy_store_spent := (1 / 3) * remaining_after_arcade
  let remaining_after_toy_store := remaining_after_arcade - toy_store_spent
  let final_spent := 0.88
  final_spent = remaining_after_toy_store → A = 3.30

theorem find_johns_allowance : johns_allowance A := by
  sorry

end find_johns_allowance_l287_287377


namespace find_f_of_one_half_l287_287091

def g (x : ℝ) : ℝ := 1 - 2 * x

noncomputable def f (x : ℝ) : ℝ := (1 - x ^ 2) / x ^ 2

theorem find_f_of_one_half :
  f (g (1 / 2)) = 15 :=
by
  sorry

end find_f_of_one_half_l287_287091


namespace B_profit_l287_287952

-- Definitions based on conditions
def investment_ratio (B_invest A_invest : ℕ) : Prop := A_invest = 3 * B_invest
def period_ratio (B_period A_period : ℕ) : Prop := A_period = 2 * B_period
def total_profit (total : ℕ) : Prop := total = 28000
def B_share (total : ℕ) := total / 7

-- Theorem statement based on the proof problem
theorem B_profit (B_invest A_invest B_period A_period total : ℕ)
  (h1 : investment_ratio B_invest A_invest)
  (h2 : period_ratio B_period A_period)
  (h3 : total_profit total) :
  B_share total = 4000 :=
by
  sorry

end B_profit_l287_287952


namespace intersection_A_B_l287_287847

def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1 / (x^2 + 1) }
def B : Set ℝ := {x | 3 * x - 2 < 7}

theorem intersection_A_B : A ∩ B = Set.Ico 1 3 := 
by
  sorry

end intersection_A_B_l287_287847


namespace anatoliy_handshakes_l287_287047

-- Define the total number of handshakes
def total_handshakes := 197

-- Define friends excluding Anatoliy
def handshake_func (n : Nat) : Nat :=
  n * (n - 1) / 2

-- Define the target problem stating that Anatoliy made 7 handshakes
theorem anatoliy_handshakes (n k : Nat) (h : handshake_func n + k = total_handshakes) : k = 7 :=
by sorry

end anatoliy_handshakes_l287_287047


namespace probability_at_least_one_passes_l287_287327

theorem probability_at_least_one_passes (prob_pass : ℚ) (prob_fail : ℚ) (p_all_fail: ℚ):
  (prob_pass = 1/3) →
  (prob_fail = 1 - prob_pass) →
  (p_all_fail = prob_fail ^ 3) →
  (1 - p_all_fail = 19/27) :=
by
  intros hpp hpf hpaf
  sorry

end probability_at_least_one_passes_l287_287327


namespace range_of_a_l287_287287

theorem range_of_a (a : ℝ) :
  (∀ (x y : ℝ), 3 * a * x + (a^2 - 3 * a + 2) * y - 9 < 0 → (3 * a * x + (a^2 - 3 * a + 2) * y - 9 = 0 → y > 0)) ↔ (1 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l287_287287


namespace more_girls_than_boys_l287_287439

theorem more_girls_than_boys
  (b g : ℕ)
  (ratio : b / g = 3 / 4)
  (total : b + g = 42) :
  g - b = 6 :=
sorry

end more_girls_than_boys_l287_287439


namespace no_positive_x_satisfies_equation_l287_287557

theorem no_positive_x_satisfies_equation : 
  ¬ ∃ (x : ℝ), (0 < x) ∧ (log 4 x * log x 9 = 2 * log 4 9) := 
sorry

end no_positive_x_satisfies_equation_l287_287557


namespace ratio_is_five_thirds_l287_287774

noncomputable def ratio_of_numbers (a b : ℝ) : Prop :=
  (a + b = 4 * (a - b)) → (a = 2 * b) → (a / b = 5 / 3)

theorem ratio_is_five_thirds {a b : ℝ} (h1 : a + b = 4 * (a - b)) (h2 : a = 2 * b) :
  a / b = 5 / 3 :=
  sorry

end ratio_is_five_thirds_l287_287774


namespace digits_of_number_l287_287381

theorem digits_of_number (d : ℕ) (h1 : 0 ≤ d ∧ d ≤ 9) (h2 : (10 * (50 + d) + 2) % 6 = 0) : (5 * 10 + d) * 10 + 2 = 522 :=
by sorry

end digits_of_number_l287_287381


namespace greatest_divisor_arithmetic_sequence_sum_l287_287348

theorem greatest_divisor_arithmetic_sequence_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ k, (15 * (x + 7 * c)) = 15 * k :=
sorry

end greatest_divisor_arithmetic_sequence_sum_l287_287348


namespace school_dinner_theater_tickets_l287_287643

theorem school_dinner_theater_tickets (x y : ℕ)
  (h1 : x + y = 225)
  (h2 : 6 * x + 9 * y = 1875) :
  x = 50 :=
by
  sorry

end school_dinner_theater_tickets_l287_287643


namespace larger_number_l287_287788

theorem larger_number (x y : ℝ) (h₁ : x + y = 45) (h₂ : x - y = 7) : x = 26 :=
by
  sorry

end larger_number_l287_287788


namespace monomial_sum_mn_l287_287436

-- Define the conditions as Lean definitions
def is_monomial_sum (x y : ℕ) (m n : ℕ) : Prop :=
  ∃ k : ℕ, (x ^ 2) * (y ^ m) + (x ^ n) * (y ^ 3) = x ^ k

-- State our main theorem
theorem monomial_sum_mn (x y : ℕ) (m n : ℕ) (h : is_monomial_sum x y m n) : m + n = 5 :=
sorry  -- Completion of the proof is not required

end monomial_sum_mn_l287_287436


namespace count_integers_between_25_and_36_l287_287483

theorem count_integers_between_25_and_36 :
  {x : ℤ | 25 < x ∧ x < 36}.finite.card = 10 :=
by
  sorry

end count_integers_between_25_and_36_l287_287483


namespace cubed_identity_l287_287283

theorem cubed_identity (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := 
sorry

end cubed_identity_l287_287283


namespace intersection_M_N_l287_287126

section

def M (x : ℝ) : Prop := sqrt x < 4
def N (x : ℝ) : Prop := 3 * x >= 1

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | (1 / 3) ≤ x ∧ x < 16} :=
by
  sorry

end

end intersection_M_N_l287_287126


namespace ellipse_equation_l287_287419

theorem ellipse_equation
  (P : ℝ × ℝ)
  (a b c : ℝ)
  (h1 : a > b ∧ b > 0)
  (h2 : 2 * a = 5 + 3)
  (h3 : (2 * c) ^ 2 = 5 ^ 2 - 3 ^ 2)
  (h4 : P.1 ^ 2 / a ^ 2 + P.2 ^ 2 / b ^ 2 = 1 ∨ P.2 ^ 2 / a ^ 2 + P.1 ^ 2 / b ^ 2 = 1)
  : ((a = 4) ∧ (c = 2) ∧ (b ^ 2 = 12) ∧
    (P.1 ^ 2 / 16 + P.2 ^ 2 / 12 = 1) ∨
    (P.2 ^ 2 / 16 + P.1 ^ 2 / 12 = 1)) :=
sorry

end ellipse_equation_l287_287419


namespace part1_values_correct_estimated_students_correct_l287_287210

def students_data : List ℕ :=
  [30, 60, 70, 10, 30, 115, 70, 60, 75, 90, 15, 70, 40, 75, 105, 80, 60, 30, 70, 45]

def total_students := 200

def categorized_counts := (2, 5, 10, 3) -- (0 ≤ t < 30, 30 ≤ t < 60, 60 ≤ t < 90, 90 ≤ t < 120)

def mean := 60

def median := 65

def mode := 70

theorem part1_values_correct :
  let a := 5
  let b := 3
  let c := 65
  let d := 70
  categorized_counts = (2, a, 10, b) ∧ mean = 60 ∧ median = c ∧ mode = d := by {
  -- Proof will be provided here
  sorry
}

theorem estimated_students_correct :
  let at_least_avg := 130
  at_least_avg = (total_students * 13 / 20) := by {
  -- Proof will be provided here
  sorry
}

end part1_values_correct_estimated_students_correct_l287_287210


namespace marcus_saves_34_22_l287_287742

def max_spend : ℝ := 200
def shoe_price : ℝ := 120
def shoe_discount : ℝ := 0.30
def sock_price : ℝ := 25
def sock_discount : ℝ := 0.20
def shirt_price : ℝ := 55
def shirt_discount : ℝ := 0.10
def sales_tax_rate : ℝ := 0.08

def calc_discounted_price (price discount : ℝ) : ℝ := price * (1 - discount)

def total_cost_before_tax : ℝ :=
  calc_discounted_price shoe_price shoe_discount +
  calc_discounted_price sock_price sock_discount +
  calc_discounted_price shirt_price shirt_discount

def sales_tax : ℝ := total_cost_before_tax * sales_tax_rate

def final_cost : ℝ := total_cost_before_tax + sales_tax

def money_saved : ℝ := max_spend - final_cost

theorem marcus_saves_34_22 :
  money_saved = 34.22 :=
by sorry

end marcus_saves_34_22_l287_287742


namespace expected_value_coin_flip_l287_287674

-- Define the conditions
def probability_heads := 2 / 3
def probability_tails := 1 / 3
def gain_heads := 5
def loss_tails := -10

-- Define the expected value calculation
def expected_value := (probability_heads * gain_heads) + (probability_tails * loss_tails)

-- Prove that the expected value is 0.00
theorem expected_value_coin_flip : expected_value = 0 := 
by sorry

end expected_value_coin_flip_l287_287674


namespace intersection_points_count_l287_287438

noncomputable def line1 : Set (ℝ × ℝ) := { p | 3 * p.2 - 2 * p.1 = 1 }
noncomputable def line2 : Set (ℝ × ℝ) := { p | p.1 + 2 * p.2 = 2 }
noncomputable def line3 : Set (ℝ × ℝ) := { p | 4 * p.1 - 6 * p.2 = 5 }

def countIntersections : ℕ :=
  let points := (line1 ∩ line2) ∪ (line1 ∩ line3) ∪ (line2 ∩ line3)
  Set.card points

theorem intersection_points_count : countIntersections = 2 :=
  sorry

end intersection_points_count_l287_287438


namespace probability_of_perpendicular_edges_l287_287070

def is_perpendicular_edge (e1 e2 : ℕ) : Prop :=
-- Define the logic for identifying perpendicular edges here
sorry

def total_outcomes : ℕ := 81

def favorable_outcomes : ℕ :=
-- Calculate the number of favorable outcomes here
20 + 6 + 18

theorem probability_of_perpendicular_edges : 
  (favorable_outcomes : ℝ) / (total_outcomes : ℝ) = 44 / 81 := by
-- Proof for calculating the probability
sorry

end probability_of_perpendicular_edges_l287_287070


namespace greatest_divisor_of_sum_of_arith_seq_l287_287351

theorem greatest_divisor_of_sum_of_arith_seq (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → d ∣ (15 * (x + 7 * c))) ∧
    (∀ k : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → k ∣ (15 * (x + 7 * c))) → k ≤ d) ∧ 
    d = 15 :=
sorry

end greatest_divisor_of_sum_of_arith_seq_l287_287351


namespace A_B_together_l287_287517

/-- This represents the problem of finding out the number of days A and B together 
can finish a piece of work given the conditions. -/
theorem A_B_together (A_rate B_rate: ℝ) (A_days B_days: ℝ) (work: ℝ) :
  A_rate = 1 / 8 →
  A_days = 4 →
  B_rate = 1 / 12 →
  B_days = 6 →
  work = 1 →
  (A_days * A_rate + B_days * B_rate = work / 2) →
  (24 / (A_rate + B_rate) = 4.8) :=
by
  intros hA_rate hA_days hB_rate hB_days hwork hwork_done
  sorry

end A_B_together_l287_287517


namespace intersection_of_M_and_N_l287_287122

def M : Set ℝ := {x | x ≥ 0 ∧ x < 16}
def N : Set ℝ := {x | x ≥ 1/3}

theorem intersection_of_M_and_N :
  M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by
  sorry

end intersection_of_M_and_N_l287_287122


namespace combined_area_correct_l287_287048

-- Define the given dimensions and border width
def length : ℝ := 0.6
def width : ℝ := 0.35
def border_width : ℝ := 0.05

-- Define the area of the rectangle, the new dimensions with the border, 
-- and the combined area of the rectangle and the border
def rectangle_area : ℝ := length * width
def new_length : ℝ := length + 2 * border_width
def new_width : ℝ := width + 2 * border_width
def combined_area : ℝ := new_length * new_width

-- The statement we want to prove
theorem combined_area_correct : combined_area = 0.315 := by
  sorry

end combined_area_correct_l287_287048


namespace percentage_of_students_who_own_cats_l287_287443

theorem percentage_of_students_who_own_cats (total_students cats_owned : ℕ) (h_total: total_students = 500) (h_cats: cats_owned = 75) :
  (cats_owned : ℚ) / total_students * 100 = 15 :=
by
  sorry

end percentage_of_students_who_own_cats_l287_287443


namespace equal_roots_condition_l287_287313

theorem equal_roots_condition (m : ℝ) :
  (m = 2 ∨ m = (9 + Real.sqrt 57) / 8 ∨ m = (9 - Real.sqrt 57) / 8) →
  ∃ a b c : ℝ, 
  (∀ x : ℝ, (a * x ^ 2 + b * x + c = 0) ↔
  (x * (x - 3) - (m + 2)) / ((x - 3) * (m - 2)) = x / m) ∧
  (b^2 - 4 * a * c = 0) :=
sorry

end equal_roots_condition_l287_287313


namespace number_of_students_in_first_group_l287_287640

def total_students : ℕ := 24
def second_group : ℕ := 8
def third_group : ℕ := 7
def fourth_group : ℕ := 4
def summed_other_groups : ℕ := second_group + third_group + fourth_group
def students_first_group : ℕ := total_students - summed_other_groups

theorem number_of_students_in_first_group :
  students_first_group = 5 :=
by
  -- proof required here
  sorry

end number_of_students_in_first_group_l287_287640


namespace probability_at_most_2_heads_l287_287936

theorem probability_at_most_2_heads : 
  (let p_at_most_2_heads := 1 - (1 / 2) ^ 3 in p_at_most_2_heads = 7 / 8) := 
by
  let p_exactly_3_heads := (1 / 2) ^ 3
  have p_at_most_2_heads := 1 - p_exactly_3_heads
  show p_at_most_2_heads = 7 / 8
  sorry

end probability_at_most_2_heads_l287_287936


namespace system_of_linear_equations_m_l287_287856

theorem system_of_linear_equations_m (x y m : ℝ) :
  (2 * x + y = 1 + 2 * m) →
  (x + 2 * y = 2 - m) →
  (x + y > 0) →
  ((2 * m + 1) * x - 2 * m < 1) →
  (x > 1) →
  (-3 < m ∧ m < -1/2) ∧ (m = -2 ∨ m = -1) :=
by
  intros h1 h2 h3 h4 h5
  -- Placeholder for proof steps
  sorry

end system_of_linear_equations_m_l287_287856


namespace exists_tangent_inequality_l287_287915

theorem exists_tangent_inequality {x : Fin 8 → ℝ} (h : Function.Injective x) :
  ∃ (i j : Fin 8), i ≠ j ∧ 0 < (x i - x j) / (1 + x i * x j) ∧ (x i - x j) / (1 + x i * x j) < Real.tan (Real.pi / 7) :=
by
  sorry

end exists_tangent_inequality_l287_287915


namespace sum_of_squares_eq_l287_287157

theorem sum_of_squares_eq :
  ∀ (M G D : ℝ), 
  (M = G / 3) → 
  (G = 450) → 
  (D = 2 * G) → 
  (M^2 + G^2 + D^2 = 1035000) :=
by
  intros M G D hM hG hD
  sorry

end sum_of_squares_eq_l287_287157


namespace max_principals_in_10_years_l287_287249

theorem max_principals_in_10_years (h : ∀ p : ℕ, 4 * p ≤ 10) :
  ∃ n : ℕ, n ≤ 3 ∧ n = 3 :=
sorry

end max_principals_in_10_years_l287_287249


namespace conference_room_probability_l287_287519

theorem conference_room_probability :
  let m := 16
  let n := 2925
  ∑ k in {m, n}, k = 2941 := by
sorry

end conference_room_probability_l287_287519


namespace larger_number_l287_287431

theorem larger_number (a b : ℕ) (h1 : 5 * b = 7 * a) (h2 : b - a = 10) : b = 35 :=
sorry

end larger_number_l287_287431


namespace a_geq_1_of_inequality_l287_287422

open Real

def f (a : ℝ) (x : ℝ) := a / x + x * log x
def g (x : ℝ) := x^3 - x^2 - 5

theorem a_geq_1_of_inequality
  (a : ℝ)
  (h : ∀ x1 x2, x1 ∈ Icc (1/2 : ℝ) 2 → x2 ∈ Icc (1/2 : ℝ) 2 → f a x1 - g x2 ≥ 2) :
  a ≥ 1 :=
begin
  sorry
end

end a_geq_1_of_inequality_l287_287422


namespace number_of_common_tangents_l287_287556

theorem number_of_common_tangents 
  (circle1 : ∀ x y : ℝ, x^2 + y^2 = 1)
  (circle2 : ∀ x y : ℝ, 2 * y^2 - 6 * x - 8 * y + 9 = 0) : 
  ∃ n : ℕ, n = 3 :=
by
  -- Proof is skipped
  sorry

end number_of_common_tangents_l287_287556


namespace count_perfect_squares_diff_two_consecutive_squares_l287_287693

theorem count_perfect_squares_diff_two_consecutive_squares:
  (∃ n : ℕ, n = 71 ∧ 
            ∀ a : ℕ, (a < 20000 → 
            (∃ b : ℕ, a^2 = (b+1)^2 - b^2))) :=
sorry

end count_perfect_squares_diff_two_consecutive_squares_l287_287693


namespace find_radius_and_diameter_l287_287211

theorem find_radius_and_diameter (M N r d : ℝ) (h1 : M = π * r^2) (h2 : N = 2 * π * r) (h3 : M / N = 15) : 
  (r = 30) ∧ (d = 60) := by
  sorry

end find_radius_and_diameter_l287_287211


namespace total_jury_duty_days_l287_287604

-- Conditions
def jury_selection_days : ℕ := 2
def trial_multiplier : ℕ := 4
def evidence_review_hours : ℕ := 2
def lunch_hours : ℕ := 1
def trial_session_hours : ℕ := 6
def hours_per_day : ℕ := evidence_review_hours + lunch_hours + trial_session_hours
def deliberation_hours_per_day : ℕ := 14 - 2

def deliberation_first_defendant_days : ℕ := 6
def deliberation_second_defendant_days : ℕ := 4
def deliberation_third_defendant_days : ℕ := 5

def deliberation_first_defendant_total_hours : ℕ := deliberation_first_defendant_days * deliberation_hours_per_day
def deliberation_second_defendant_total_hours : ℕ := deliberation_second_defendant_days * deliberation_hours_per_day
def deliberation_third_defendant_total_hours : ℕ := deliberation_third_defendant_days * deliberation_hours_per_day

def deliberation_days_conversion (total_hours: ℕ) : ℕ := (total_hours + deliberation_hours_per_day - 1) / deliberation_hours_per_day

-- Total days spent
def total_days_spent : ℕ :=
  let trial_days := jury_selection_days * trial_multiplier
  let deliberation_days := deliberation_days_conversion deliberation_first_defendant_total_hours + deliberation_days_conversion deliberation_second_defendant_total_hours + deliberation_days_conversion deliberation_third_defendant_total_hours
  jury_selection_days + trial_days + deliberation_days

#eval total_days_spent -- Expected: 25

theorem total_jury_duty_days : total_days_spent = 25 := by
  sorry

end total_jury_duty_days_l287_287604


namespace sarah_mean_score_l287_287706

noncomputable def john_mean_score : ℝ := 86
noncomputable def john_num_tests : ℝ := 4
noncomputable def test_scores : List ℝ := [78, 80, 85, 87, 90, 95, 100]
noncomputable def total_sum : ℝ := test_scores.sum
noncomputable def sarah_num_tests : ℝ := 3

theorem sarah_mean_score :
  let john_total_score := john_mean_score * john_num_tests
  let sarah_total_score := total_sum - john_total_score
  let sarah_mean_score := sarah_total_score / sarah_num_tests
  sarah_mean_score = 90.3 :=
by
  sorry

end sarah_mean_score_l287_287706


namespace line_points_sum_slope_and_intercept_l287_287588

-- Definition of the problem
theorem line_points_sum_slope_and_intercept (a b : ℝ) :
  (∀ x y : ℝ, (x = 2 ∧ y = 3) ∨ (x = 10 ∧ y = 19) → y = a * x + b) →
  a + b = 1 :=
by
  intro h
  sorry

end line_points_sum_slope_and_intercept_l287_287588


namespace length_of_bridge_l287_287945

theorem length_of_bridge (L_train : ℕ) (v_km_hr : ℕ) (t : ℕ) 
  (h_L_train : L_train = 150)
  (h_v_km_hr : v_km_hr = 45)
  (h_t : t = 30) : 
  ∃ L_bridge : ℕ, L_bridge = 225 :=
by 
  sorry

end length_of_bridge_l287_287945


namespace probability_no_consecutive_ones_l287_287821

open Nat

-- Define the function to count valid sequences without consecutive 1s
def a_n (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else if n = 2 then 3
  else a_n (n - 1) + a_n (n - 2)

-- Define the problem statement
theorem probability_no_consecutive_ones : 
  let p := (a_n 10).to_rat / 2^10 in p = 9 / 64 := by
  sorry

end probability_no_consecutive_ones_l287_287821


namespace isosceles_triangle_base_angle_l287_287177

-- Define the problem and the given conditions
theorem isosceles_triangle_base_angle (A B C : ℝ)
(h_triangle : A + B + C = 180)
(h_isosceles : (A = B ∨ B = C ∨ C = A))
(h_ratio : (A = B / 2 ∨ B = C / 2 ∨ C = A / 2)) :
(A = 45 ∨ A = 72) ∨ (B = 45 ∨ B = 72) ∨ (C = 45 ∨ C = 72) :=
sorry

end isosceles_triangle_base_angle_l287_287177


namespace ratio_hexagon_octagon_l287_287061

noncomputable def ratio_of_areas (s : ℝ) :=
  let A1 := s / (2 * Real.tan (Real.pi / 6))
  let H1 := s / (2 * Real.sin (Real.pi / 6))
  let area1 := Real.pi * (H1^2 - A1^2)
  let A2 := s / (2 * Real.tan (Real.pi / 8))
  let H2 := s / (2 * Real.sin (Real.pi / 8))
  let area2 := Real.pi * (H2^2 - A2^2)
  area1 / area2

theorem ratio_hexagon_octagon (s : ℝ) (h : s = 3) : ratio_of_areas s = 49 / 25 :=
  sorry

end ratio_hexagon_octagon_l287_287061


namespace hypotenuse_of_isosceles_right_triangle_l287_287306

theorem hypotenuse_of_isosceles_right_triangle (a : ℝ) (hyp : a = 8) : 
  ∃ c : ℝ, c = a * Real.sqrt 2 :=
by
  use 8 * Real.sqrt 2
  sorry

end hypotenuse_of_isosceles_right_triangle_l287_287306


namespace choco_delight_remainder_l287_287989

theorem choco_delight_remainder (m : ℕ) (h : m % 7 = 5) : (4 * m) % 7 = 6 := 
by 
  sorry

end choco_delight_remainder_l287_287989


namespace number_of_girls_in_class_l287_287136

theorem number_of_girls_in_class (B G : ℕ) (h1 : G = 4 * B / 10) (h2 : B + G = 35) : G = 10 :=
by
  sorry

end number_of_girls_in_class_l287_287136


namespace triangle_is_right_l287_287587

theorem triangle_is_right (a b c : ℝ) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_equations_share_root : ∃ α : ℝ, α^2 + 2*a*α + b^2 = 0 ∧ α^2 + 2*c*α - b^2 = 0) :
  a^2 = b^2 + c^2 :=
by sorry

end triangle_is_right_l287_287587


namespace non_congruent_triangles_with_perimeter_11_l287_287869

theorem non_congruent_triangles_with_perimeter_11 :
  ∃ (a b c : ℕ), a + b + c = 11 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧
  (∀ d e f : ℕ, d + e + f = 11 ∧ d ≤ e ∧ e ≤ f ∧ d + e > f → 
  (d = a ∧ e = b ∧ f = c) ∨ (d = b ∧ e = a ∧ f = c) ∨ (d = a ∧ e = c ∧ f = b)) → 
  3 := 
sorry

end non_congruent_triangles_with_perimeter_11_l287_287869


namespace probability_plane_intersects_interior_rect_prism_l287_287260

theorem probability_plane_intersects_interior_rect_prism : 
  let total_ways := Nat.choose 8 4,
      non_intersecting_cases := 6 in
    (total_ways - non_intersecting_cases) / total_ways = (32 : ℚ) / 35 := by
sorry

end probability_plane_intersects_interior_rect_prism_l287_287260


namespace functional_relationship_y1_daily_gross_profit_1120_first_10_days_total_gross_profit_W_l287_287041

-- Conditions for y1
def cost_price : ℕ := 60
def selling_price_first_10_days : ℕ := 80
def y1 : ℕ → ℕ := fun x => x * x - 8 * x + 56
def items_sold_day4 : ℕ := 40
def items_sold_day6 : ℕ := 44

-- Conditions for y2
def selling_price_post_10_days : ℕ := 100
def y2 : ℕ → ℕ := fun x => 2 * x + 8
def gross_profit_condition : ℕ := 1120

-- 1) Prove functional relationship of y1.
theorem functional_relationship_y1 (x : ℕ) (h4 : y1 4 = 40) (h6 : y1 6 = 44) : 
  y1 x = x * x - 8 * x + 56 := 
by
  sorry

-- 2) Prove value of x for daily gross profit $1120 on any day within first 10 days.
theorem daily_gross_profit_1120_first_10_days (x : ℕ) (h4 : y1 4 = 40) (h6 : y1 6 = 44) (gp : (selling_price_first_10_days - cost_price) * y1 x = gross_profit_condition) : 
  x = 8 := 
by
  sorry

-- 3) Prove total gross profit W and range for 26 < x ≤ 31.
theorem total_gross_profit_W (x : ℕ) (h : 26 < x ∧ x ≤ 31) : 
  (100 - (cost_price - 2 * (y2 x - 60))) * (y2 x) = 8 * x * x - 96 * x - 512 := 
by
  sorry

end functional_relationship_y1_daily_gross_profit_1120_first_10_days_total_gross_profit_W_l287_287041


namespace function_classification_l287_287404

theorem function_classification {f : ℝ → ℝ} 
    (h : ∀ x y : ℝ, x * f y + y * f x = (x + y) * f x * f y) : 
    ∀ x : ℝ, f x = 0 ∨ f x = 1 :=
by
  sorry

end function_classification_l287_287404


namespace price_difference_l287_287527

def P := ℝ

def Coupon_A_savings (P : ℝ) := 0.20 * P
def Coupon_B_savings : ℝ := 40
def Coupon_C_savings (P : ℝ) := 0.30 * (P - 120) + 20

def Coupon_A_geq_Coupon_B (P : ℝ) := Coupon_A_savings P ≥ Coupon_B_savings
def Coupon_A_geq_Coupon_C (P : ℝ) := Coupon_A_savings P ≥ Coupon_C_savings P

noncomputable def x : ℝ := 200
noncomputable def y : ℝ := 300

theorem price_difference (P : ℝ) (h1 : P > 120)
  (h2 : Coupon_A_geq_Coupon_B P)
  (h3 : Coupon_A_geq_Coupon_C P) :
  y - x = 100 := by
  sorry

end price_difference_l287_287527


namespace length_of_tunnel_l287_287053

theorem length_of_tunnel
    (length_of_train : ℕ)
    (speed_kmh : ℕ)
    (crossing_time_seconds : ℕ)
    (distance_covered : ℕ)
    (length_of_tunnel : ℕ) :
    length_of_train = 1200 →
    speed_kmh = 96 →
    crossing_time_seconds = 90 →
    distance_covered = (speed_kmh * 1000 / 3600) * crossing_time_seconds →
    length_of_train + length_of_tunnel = distance_covered →
    length_of_tunnel = 6000 :=
by
  sorry

end length_of_tunnel_l287_287053


namespace aqua_park_earnings_l287_287540

/-- Define the costs and groups of visitors. --/
def admission_fee : ℕ := 12
def tour_fee : ℕ := 6
def group1_size : ℕ := 10
def group2_size : ℕ := 5

/-- Define the total earnings of the aqua park. --/
def total_earnings : ℕ := (admission_fee + tour_fee) * group1_size + admission_fee * group2_size

/-- Prove that the total earnings are $240. --/
theorem aqua_park_earnings : total_earnings = 240 :=
by
  -- proof steps would go here
  sorry

end aqua_park_earnings_l287_287540


namespace casey_savings_l287_287233

-- Define the constants given in the problem conditions
def wage_employee_1 : ℝ := 20
def wage_employee_2 : ℝ := 22
def subsidy : ℝ := 6
def hours_per_week : ℝ := 40

-- Define the weekly cost of each employee
def weekly_cost_employee_1 := wage_employee_1 * hours_per_week
def weekly_cost_employee_2 := (wage_employee_2 - subsidy) * hours_per_week

-- Define the savings by hiring the cheaper employee
def savings := weekly_cost_employee_1 - weekly_cost_employee_2

-- Theorem stating the expected savings
theorem casey_savings : savings = 160 := by
  -- Proof is not included
  sorry

end casey_savings_l287_287233


namespace course_selection_schemes_l287_287960

theorem course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose (n k : ℕ) := nat.choose n k
  
  (choose 4 1 * choose 4 1 + 
   choose 4 2 * choose 4 1 + 
   choose 4 1 * choose 4 2) = 64 := 
by {
  -- calculating the combinations
  have comb_2_courses : choose 4 1 * choose 4 1 = 16 := by sorry,
  have comb_2_pe_1_art : choose 4 2 * choose 4 1 = 24 := by sorry,
  have comb_1_pe_2_art : choose 4 1 * choose 4 2 = 24 := by sorry,
  have total_comb : 16 + 24 + 24 = 64 := by {
    simp [comb_2_courses, comb_2_pe_1_art, comb_1_pe_2_art]
  },
  exact total_comb
}

end course_selection_schemes_l287_287960


namespace marbles_problem_l287_287835

def marbles_total : ℕ := 30
def prob_black_black : ℚ := 14 / 25
def prob_white_white : ℚ := 16 / 225

theorem marbles_problem (total_marbles : ℕ) (prob_bb prob_ww : ℚ) 
  (h_total : total_marbles = 30)
  (h_prob_bb : prob_bb = 14 / 25)
  (h_prob_ww : prob_ww = 16 / 225) :
  let m := 16
  let n := 225
  m.gcd n = 1 ∧ m + n = 241 :=
by {
  sorry
}

end marbles_problem_l287_287835


namespace cos_B_equals_3_over_4_l287_287292

variables {A B C : ℝ} {a b c R : ℝ} (h₁ : b * Real.sin B - a * Real.sin A = (1/2) * a * Real.sin C)
  (h₂ :  2 * R ^ 2 * Real.sin B * (1 - Real.cos (2 * A)) = (1 / 2) * a * b * Real.sin C)

theorem cos_B_equals_3_over_4 : Real.cos B = 3 / 4 := by
  sorry

end cos_B_equals_3_over_4_l287_287292


namespace time_increases_with_water_speed_increase_l287_287045

variable (S : ℝ) -- Total distance
variable (V : ℝ) -- Speed of the ferry in still water
variable (V1 V2 : ℝ) -- Speed of the water flow before and after increase

-- Ensure realistic conditions
axiom V_pos : 0 < V
axiom V1_pos : 0 < V1
axiom V2_pos : 0 < V2
axiom V1_less_V : V1 < V
axiom V2_less_V : V2 < V
axiom V1_less_V2 : V1 < V2

theorem time_increases_with_water_speed_increase :
  (S / (V + V1) + S / (V - V1)) < (S / (V + V2) + S / (V - V2)) :=
sorry

end time_increases_with_water_speed_increase_l287_287045


namespace polynomial_divisibility_l287_287584

theorem polynomial_divisibility (t : ℤ) : 
  (∀ x : ℤ, (5 * x^3 - 15 * x^2 + t * x - 20) ∣ (x - 2)) → (t = 20) → 
  ∀ x : ℤ, (5 * x^3 - 15 * x^2 + 20 * x - 20) ∣ (5 * x^2 + 5 * x + 5) :=
by
  intro h₁ h₂
  sorry

end polynomial_divisibility_l287_287584


namespace count_non_congruent_triangles_with_perimeter_11_l287_287872

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter (a b c : ℕ) : Prop :=
  a + b + c = 11

def valid_triangle_sets : Nat :=
  if is_triangle 3 3 5 ∧ perimeter 3 3 5 then
    if is_triangle 2 4 5 ∧ perimeter 2 4 5 then 2
    else 1
  else 0

theorem count_non_congruent_triangles_with_perimeter_11 (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) :
  (perimeter a b c) → (is_triangle a b c) → valid_triangle_sets = 2 :=
by
  sorry

end count_non_congruent_triangles_with_perimeter_11_l287_287872


namespace chuck_team_score_proof_chuck_team_score_l287_287252

-- Define the conditions
def yellow_team_score : ℕ := 55
def lead : ℕ := 17

-- State the main proposition
theorem chuck_team_score (yellow_team_score : ℕ) (lead : ℕ) : ℕ :=
yellow_team_score + lead

-- Formulate the final proof goal
theorem proof_chuck_team_score : chuck_team_score yellow_team_score lead = 72 :=
by {
  -- This is the place where the proof should go
  sorry
}

end chuck_team_score_proof_chuck_team_score_l287_287252


namespace perimeters_equal_l287_287051

noncomputable def side_length_square := 15 -- cm
noncomputable def length_rectangle := 18 -- cm
noncomputable def area_rectangle := 216 -- cm²

theorem perimeters_equal :
  let perimeter_square := 4 * side_length_square
  let width_rectangle := area_rectangle / length_rectangle
  let perimeter_rectangle := 2 * (length_rectangle + width_rectangle)
  perimeter_square = perimeter_rectangle :=
by
  sorry

end perimeters_equal_l287_287051


namespace sled_total_distance_l287_287050

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (a₁ + (a₁ + (n - 1) * d)) / 2

theorem sled_total_distance (a₁ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 6 → d = 8 → n = 20 → arithmetic_sequence_sum a₁ d n = 1640 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end sled_total_distance_l287_287050


namespace probability_3_closer_0_0_to_6_l287_287818

noncomputable def probability_closer_to_3_than_0 (a b c : ℝ) : ℝ :=
  if h₁ : a < b ∧ b < c then
    (c - ((a + b) / 2)) / (c - a)
  else 0

theorem probability_3_closer_0_0_to_6 : probability_closer_to_3_than_0 0 3 6 = 0.75 := by
  sorry

end probability_3_closer_0_0_to_6_l287_287818


namespace water_volume_correct_l287_287427

-- Define the conditions
def ratio_water_juice : ℕ := 5
def ratio_juice_water : ℕ := 3
def total_punch_volume : ℚ := 3  -- in liters

-- Define the question and the correct answer
def volume_of_water (ratio_water_juice ratio_juice_water : ℕ) (total_punch_volume : ℚ) : ℚ :=
  (ratio_water_juice * total_punch_volume) / (ratio_water_juice + ratio_juice_water)

-- The proof problem
theorem water_volume_correct : volume_of_water ratio_water_juice ratio_juice_water total_punch_volume = 15 / 8 :=
by
  sorry

end water_volume_correct_l287_287427


namespace parabola_tangent_line_l287_287725

theorem parabola_tangent_line (a : ℝ) : 
  (∀ x : ℝ, (y = ax^2 + 6 ↔ y = x)) → a = 1 / 24 :=
by
  sorry

end parabola_tangent_line_l287_287725


namespace triangle_properties_equivalence_l287_287598

-- Define the given properties for the two triangles
variables {A B C A' B' C' : Type}

-- Triangle side lengths and properties
def triangles_equal (b b' c c' : ℝ) : Prop :=
  (b = b') ∧ (c = c')

def equivalent_side_lengths (a a' b b' c c' : ℝ) : Prop :=
  a = a'

def equivalent_medians (ma ma' b b' c c' a a' : ℝ) : Prop :=
  ma = ma'

def equivalent_altitudes (ha ha' Δ Δ' a a' : ℝ) : Prop :=
  ha = ha'

def equivalent_angle_bisectors (ta ta' b b' c c' a a' : ℝ) : Prop :=
  ta = ta'

def equivalent_circumradii (R R' a a' b b' c c' : ℝ) : Prop :=
  R = R'

def equivalent_areas (Δ Δ' b b' c c' A A' : ℝ) : Prop :=
  Δ = Δ'

-- Main theorem statement
theorem triangle_properties_equivalence
  (b b' c c' a a' ma ma' ha ha' ta ta' R R' Δ Δ' : ℝ)
  (A A' : ℝ)
  (eq_b : b = b')
  (eq_c : c = c') :
  equivalent_side_lengths a a' b b' c c' ∧ 
  equivalent_medians ma ma' b b' c c' a a' ∧ 
  equivalent_altitudes ha ha' Δ Δ' a a' ∧ 
  equivalent_angle_bisectors ta ta' b b' c c' a a' ∧ 
  equivalent_circumradii R R' a a' b b' c c' ∧ 
  equivalent_areas Δ Δ' b b' c c' A A'
:= by
  sorry

end triangle_properties_equivalence_l287_287598


namespace capital_of_a_l287_287800

variable (P P' TotalCapital Ca : ℝ)

theorem capital_of_a 
  (h1 : a_income_5_percent = (2/3) * P)
  (h2 : a_income_7_percent = (2/3) * P')
  (h3 : a_income_7_percent - a_income_5_percent = 200)
  (h4 : P = 0.05 * TotalCapital)
  (h5 : P' = 0.07 * TotalCapital)
  : Ca = (2/3) * TotalCapital :=
by
  sorry

end capital_of_a_l287_287800


namespace sequence_a4_value_l287_287729

theorem sequence_a4_value : 
  ∀ (a : ℕ → ℕ), a 1 = 2 → (∀ n, n ≥ 2 → a n = a (n - 1) + n) → a 4 = 11 :=
by
  sorry

end sequence_a4_value_l287_287729


namespace greatest_divisor_of_sum_of_arithmetic_sequence_l287_287365

theorem greatest_divisor_of_sum_of_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ k : ℕ, (sum (λ n, x + n * c) (range 15)) = 15 * k :=
by sorry

end greatest_divisor_of_sum_of_arithmetic_sequence_l287_287365


namespace last_four_digits_of_5_pow_2018_l287_287460

theorem last_four_digits_of_5_pow_2018 : 
  (5^2018) % 10000 = 5625 :=
by {
  sorry
}

end last_four_digits_of_5_pow_2018_l287_287460


namespace handshakes_7_boys_l287_287656

theorem handshakes_7_boys : Nat.choose 7 2 = 21 :=
by
  sorry

end handshakes_7_boys_l287_287656


namespace roots_product_of_quadratic_equation_l287_287876

variables (a b : ℝ)

-- Given that a and b are roots of the quadratic equation x^2 - ax + b = 0
-- and given conditions that a + b = 5 and ab = 6,
-- prove that a * b = 6.
theorem roots_product_of_quadratic_equation 
  (h₁ : a + b = 5) 
  (h₂ : a * b = 6) : 
  a * b = 6 := 
by 
 sorry

end roots_product_of_quadratic_equation_l287_287876


namespace inconsistency_proof_l287_287595

-- Let TotalBoys be the number of boys, which is 120
def TotalBoys := 120

-- Let AverageMarks be the average marks obtained by 120 boys, which is 40
def AverageMarks := 40

-- Let PassedBoys be the number of boys who passed, which is 125
def PassedBoys := 125

-- Let AverageMarksFailed be the average marks of failed boys, which is 15
def AverageMarksFailed := 15

-- We need to prove the inconsistency
theorem inconsistency_proof :
  ∀ (P : ℝ), 
    (TotalBoys * AverageMarks = PassedBoys * P + (TotalBoys - PassedBoys) * AverageMarksFailed) →
    False :=
by
  intro P h
  sorry

end inconsistency_proof_l287_287595


namespace countable_independent_bernoulli_l287_287983

/-- 
On the probability space \(( (0,1], \mathscr{B}((0,1]), \mathrm{P} )\), 
it is impossible to define more than a countable number of independent 
Bernoulli random variables \(\left\{\xi_{t}\right\}_{t \in T}\) taking two values 0 and 1 
with non-zero probability.
-/
theorem countable_independent_bernoulli 
  (P : MeasureTheory.ProbabilityMeasure (set.Ioc 0 1))
  (ξ : ℕ → MeasureTheory.MeasurableSpace (set.Ioc 0 1) → bool)
  (ξ_indep : ∀ m n, m ≠ n → MeasureTheory.Indep (ξ m) (ξ n))
  (ξ_bernoulli : ∀ n, MeasureTheory.ProbabilityMeasure (ξ n = 0) = 1 / 2 
                     ∧ MeasureTheory.ProbabilityMeasure (ξ n = 1) = 1 / 2) : 
  ∃ (T : set ℕ), T.countable ∧ ∀ t ∈ T, MeasureTheory.ProbabilityMeasure (ξ t = 0) > 0 ∧ MeasureTheory.ProbabilityMeasure (ξ t = 1) > 0 :=
sorry

end countable_independent_bernoulli_l287_287983


namespace tan_alpha_minus_pi_over_4_eq_negative_seven_l287_287266

open Real

theorem tan_alpha_minus_pi_over_4_eq_negative_seven (α : ℝ) (h1 : α ∈ Ioo (-π) 0) (h2 : cos α = -4/5) : 
  tan (α - π/4) = -7 := sorry

end tan_alpha_minus_pi_over_4_eq_negative_seven_l287_287266


namespace max_dot_and_area_of_triangle_l287_287446

noncomputable def triangle_data (A B C : ℝ) (m n : ℝ × ℝ) : Prop :=
  A + B + C = Real.pi ∧
  (m = (2, 2 * (Real.cos ((B + C) / 2))^2 - 1)) ∧
  (n = (Real.sin (A / 2), -1))

noncomputable def is_max_dot_product (A : ℝ) (m n : ℝ × ℝ) : Prop :=
  m.1 * n.1 + m.2 * n.2 = (if A = Real.pi / 3 then 3 / 2 else 0)

noncomputable def max_area (A B C : ℝ) : ℝ :=
  let a : ℝ := 2
  let b : ℝ := 2
  let c : ℝ := 2
  if A = Real.pi / 3 then (Real.sqrt 3) else 0

theorem max_dot_and_area_of_triangle {A B C : ℝ} {m n : ℝ × ℝ}
  (h_triangle : triangle_data A B C m n) :
  is_max_dot_product (Real.pi / 3) m n ∧ max_area A B C = Real.sqrt 3 := by sorry

end max_dot_and_area_of_triangle_l287_287446


namespace solution_l287_287082

def N_star := { n : ℕ // n > 0 }

def f (n : N_star) : N_star := sorry

theorem solution (f : N_star → N_star) :
  (∀ m n : N_star, (f m).val * (f m).val + (f n).val ∣ (m.val * m.val + n.val) * (m.val * m.val + n.val)) →
  (∀ n : N_star, f n = n) := 
sorry

end solution_l287_287082


namespace equilateral_triangle_side_length_l287_287498

theorem equilateral_triangle_side_length 
    (D A B C : ℝ × ℝ)
    (h_distances : dist D A = 2 ∧ dist D B = 3 ∧ dist D C = 5)
    (h_equilateral : dist A B = dist B C ∧ dist B C = dist C A) :
    dist A B = Real.sqrt 19 :=
by
    sorry -- Proof to be filled

end equilateral_triangle_side_length_l287_287498


namespace sum_of_terms_l287_287322

def sequence_sum (n : ℕ) : ℕ :=
  n^2 + 2*n + 5

theorem sum_of_terms : sequence_sum 9 - sequence_sum 6 = 51 :=
by
  sorry

end sum_of_terms_l287_287322


namespace nathan_write_in_one_hour_l287_287731

/-- Jacob can write twice as fast as Nathan. Nathan wrote some letters in one hour. Together, they can write 750 letters in 10 hours. How many letters can Nathan write in one hour? -/
theorem nathan_write_in_one_hour
  (N : ℕ)  -- Assume N is the number of letters Nathan can write in one hour
  (H₁ : ∀ (J : ℕ), J = 2 * N)  -- Jacob writes twice faster, so letters written by Jacob in one hour is 2N
  (H₂ : 10 * (N + 2 * N) = 750)  -- Together they write 750 letters in 10 hours
  : N = 25 := by
  -- Proof will go here
  sorry

end nathan_write_in_one_hour_l287_287731


namespace area_of_shaded_region_l287_287626

-- Given conditions
def side_length := 8
def area_of_square := side_length * side_length
def area_of_triangle := area_of_square / 4

-- Lean 4 statement for the equivalence
theorem area_of_shaded_region : area_of_triangle = 16 :=
by
  sorry

end area_of_shaded_region_l287_287626


namespace ScientificNotation_of_45400_l287_287622

theorem ScientificNotation_of_45400 :
  45400 = 4.54 * 10^4 := sorry

end ScientificNotation_of_45400_l287_287622


namespace probability_X_eq_Y_l287_287054

-- Define the conditions as functions or predicates.
def is_valid_pair (x y : ℝ) : Prop :=
  -5 * Real.pi ≤ x ∧ x ≤ 5 * Real.pi ∧ -5 * Real.pi ≤ y ∧ y ≤ 5 * Real.pi ∧ Real.cos (Real.cos x) = Real.cos (Real.cos y)

-- Final statement asserting the required probability.
theorem probability_X_eq_Y :
  ∃ (prob : ℝ), prob = 1 / 11 ∧ ∀ (x y : ℝ), is_valid_pair x y → (x = y ∨ x ≠ y ∧ prob = 1/11) :=
  sorry

end probability_X_eq_Y_l287_287054


namespace smith_oldest_child_age_l287_287006

theorem smith_oldest_child_age
  (avg_age : ℕ)
  (youngest : ℕ)
  (middle : ℕ)
  (oldest : ℕ)
  (h1 : avg_age = 9)
  (h2 : youngest = 6)
  (h3 : middle = 8)
  (h4 : (youngest + middle + oldest) / 3 = avg_age) :
  oldest = 13 :=
by
  sorry

end smith_oldest_child_age_l287_287006


namespace father_l287_287044

variable (S F : ℕ)

theorem father's_age (h1 : F = 3 * S) (h2 : F + 15 = 2 * (S + 15)) : F = 45 :=
sorry

end father_l287_287044


namespace ratio_paislee_to_calvin_l287_287830

theorem ratio_paislee_to_calvin (calvin_points paislee_points : ℕ) (h1 : calvin_points = 500) (h2 : paislee_points = 125) : paislee_points / calvin_points = 1 / 4 := by
  sorry

end ratio_paislee_to_calvin_l287_287830


namespace system_of_inequalities_solutions_l287_287002

theorem system_of_inequalities_solutions (x : ℤ) :
  (3 * x - 2 ≥ 2 * x - 5) ∧ ((x / 2 - (x - 2) / 3 < 1 / 2)) →
  (x = -3 ∨ x = -2) :=
by sorry

end system_of_inequalities_solutions_l287_287002


namespace beadshop_profit_on_wednesday_l287_287809

theorem beadshop_profit_on_wednesday (total_profit profit_on_monday profit_on_tuesday profit_on_wednesday : ℝ)
  (h1 : total_profit = 1200)
  (h2 : profit_on_monday = total_profit / 3)
  (h3 : profit_on_tuesday = total_profit / 4)
  (h4 : profit_on_wednesday = total_profit - profit_on_monday - profit_on_tuesday) :
  profit_on_wednesday = 500 := 
sorry

end beadshop_profit_on_wednesday_l287_287809


namespace naomi_regular_bikes_l287_287059
-- Import necessary libraries

-- Define the condition and the proof problem
theorem naomi_regular_bikes (R C : ℕ) (h1 : C = 11) 
  (h2 : 2 * R + 4 * C = 58) : R = 7 := 
  by 
  -- Include all necessary conditions as assumptions
  have hC : C = 11 := h1
  have htotal : 2 * R + 4 * C = 58 := h2
  -- Skip the proof itself
  sorry

end naomi_regular_bikes_l287_287059


namespace simplify_120_div_180_l287_287917

theorem simplify_120_div_180 : (120 : ℚ) / 180 = 2 / 3 :=
by sorry

end simplify_120_div_180_l287_287917


namespace isla_capsules_days_l287_287294

theorem isla_capsules_days (days_in_july : ℕ) (days_forgot : ℕ) (known_days_in_july : days_in_july = 31) (known_days_forgot : days_forgot = 2) : days_in_july - days_forgot = 29 := 
by
  -- Placeholder for proof, not required in the response.
  sorry

end isla_capsules_days_l287_287294


namespace inequality_any_k_l287_287018

theorem inequality_any_k (x y z : ℝ) (k : ℕ) 
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : x * y * z = 1) (h5 : 1/x + 1/y + 1/z ≥ x + y + z) : 
  x ^ (-k : ℤ) + y ^ (-k : ℤ) + z ^ (-k : ℤ) ≥ x ^ k + y ^ k + z ^ k :=
sorry

end inequality_any_k_l287_287018


namespace jerry_wants_to_raise_average_l287_287602

theorem jerry_wants_to_raise_average :
  let average_first_3_tests := 85
  let total_first_3_tests := 3 * average_first_3_tests
  let score_fourth_test := 93
  let total_after_fourth_test := total_first_3_tests + score_fourth_test
  let new_average := (total_after_fourth_test : ℤ) / 4
  new_average - average_first_3_tests = 2 :=
by
  let average_first_3_tests := 85
  let total_first_3_tests := 3 * average_first_3_tests
  let score_fourth_test := 93
  let total_after_fourth_test := total_first_3_tests + score_fourth_test
  let new_average := (total_after_fourth_test : ℤ) / 4
  have h : new_average - average_first_3_tests = 2 := by
    sorry
  exact h

end jerry_wants_to_raise_average_l287_287602


namespace correct_calculation_l287_287653

theorem correct_calculation :
  (∀ a b : ℝ, (4 * a * b)^2 = 16 * a^2 * b^2) ∧
  (∀ a : ℝ, a^2 * a^3 = a^5) ∧
  (∀ a : ℝ, a^2 + a^2 = 2 * a^2) ∧
  (∀ a b : ℝ, (-3 * a^3 * b)^2 = 9 * a^6 * b^2)
: option_D_correct :=
  sorry

end correct_calculation_l287_287653


namespace tank_full_capacity_l287_287042

theorem tank_full_capacity (w c : ℕ) (h1 : w = c / 6) (h2 : w + 4 = c / 3) : c = 12 :=
sorry

end tank_full_capacity_l287_287042


namespace word_sum_problems_l287_287897

theorem word_sum_problems (J M O I : Fin 10) (h_distinct : J ≠ M ∧ J ≠ O ∧ J ≠ I ∧ M ≠ O ∧ M ≠ I ∧ O ≠ I) 
  (h_nonzero_J : J ≠ 0) (h_nonzero_I : I ≠ 0) :
  let JMO := 100 * J + 10 * M + O
  let IMO := 100 * I + 10 * M + O
  (JMO + JMO + JMO = IMO) → 
  (JMO = 150 ∧ IMO = 450) ∨ (JMO = 250 ∧ IMO = 750) :=
sorry

end word_sum_problems_l287_287897


namespace fewest_printers_l287_287944

theorem fewest_printers (cost1 cost2 : ℕ) (h1 : cost1 = 375) (h2 : cost2 = 150) : 
  ∃ (n : ℕ), n = 2 + 5 :=
by
  have lcm_375_150 : Nat.lcm cost1 cost2 = 750 := sorry
  have n1 : 750 / 375 = 2 := sorry
  have n2 : 750 / 150 = 5 := sorry
  exact ⟨7, rfl⟩

end fewest_printers_l287_287944


namespace tan_value_l287_287850

variable (a : ℕ → ℝ) (b : ℕ → ℝ)
variable (a_geom : ∀ m n : ℕ, a m / a n = a (m - n))
variable (b_arith : ∃ c d : ℝ, ∀ n : ℕ, b n = c + n * d)
variable (ha : a 1 * a 6 * a 11 = -3 * Real.sqrt 3)
variable (hb : b 1 + b 6 + b 11 = 7 * Real.pi)

theorem tan_value : Real.tan ((b 3 + b 9) / (1 - a 4 * a 8)) = -Real.sqrt 3 := by
  sorry

end tan_value_l287_287850


namespace infinite_series_eq_5_over_16_l287_287978

noncomputable def infinite_series_sum : ℝ :=
  ∑' (n : ℕ), (n + 1 : ℝ) / (5 ^ (n + 1))

theorem infinite_series_eq_5_over_16 :
  infinite_series_sum = 5 / 16 :=
sorry

end infinite_series_eq_5_over_16_l287_287978


namespace tiffany_bags_on_monday_l287_287780

theorem tiffany_bags_on_monday : 
  ∃ M : ℕ, M = 8 ∧ ∃ T : ℕ, T = 7 ∧ M = T + 1 :=
by
  sorry

end tiffany_bags_on_monday_l287_287780


namespace arithmetic_sequence_150th_term_l287_287401

theorem arithmetic_sequence_150th_term :
  let a1 := 3
  let d := 5
  let n := 150
  a1 + (n - 1) * d = 748 :=
by
  sorry

end arithmetic_sequence_150th_term_l287_287401


namespace chord_length_of_tangent_circle_l287_287005

theorem chord_length_of_tangent_circle
  (area_of_ring : ℝ)
  (diameter_large_circle : ℝ)
  (h1 : area_of_ring = (50 / 3) * Real.pi)
  (h2 : diameter_large_circle = 10) :
  ∃ (length_of_chord : ℝ), length_of_chord = (10 * Real.sqrt 6) / 3 := by
  sorry

end chord_length_of_tangent_circle_l287_287005


namespace quiz_total_points_l287_287442

theorem quiz_total_points (points : ℕ → ℕ) 
  (h1 : ∀ n, points (n+1) = points n + 4)
  (h2 : points 2 = 39) : 
  (points 0 + points 1 + points 2 + points 3 + points 4 + points 5 + points 6 + points 7) = 360 :=
sorry

end quiz_total_points_l287_287442


namespace Alex_hula_hoop_duration_l287_287158

-- Definitions based on conditions
def Nancy_duration := 10
def Casey_duration := Nancy_duration - 3
def Morgan_duration := Casey_duration * 3
def Alex_duration := Casey_duration + Morgan_duration - 2

-- The theorem we need to prove
theorem Alex_hula_hoop_duration : Alex_duration = 26 := by
  -- proof to be provided
  sorry

end Alex_hula_hoop_duration_l287_287158


namespace least_value_of_x_l287_287075

theorem least_value_of_x (x : ℝ) : (4 * x^2 + 8 * x + 3 = 1) → (-1 ≤ x) :=
by
  intro h
  sorry

end least_value_of_x_l287_287075


namespace no_both_squares_l287_287300

theorem no_both_squares {x y : ℕ} (hx : x > 0) (hy : y > 0) : ¬ (∃ a b : ℕ, a^2 = x^2 + 2 * y ∧ b^2 = y^2 + 2 * x) :=
by
  sorry

end no_both_squares_l287_287300


namespace convert_A03_to_decimal_l287_287688

theorem convert_A03_to_decimal :
  let A := 10
  let hex_value := A * 16^2 + 0 * 16^1 + 3 * 16^0
  hex_value = 2563 :=
by
  let A := 10
  let hex_value := A * 16^2 + 0 * 16^1 + 3 * 16^0
  have : hex_value = 2563 := sorry
  exact this

end convert_A03_to_decimal_l287_287688


namespace problem_I_problem_II_l287_287569

-- Definitions
def p (x : ℝ) : Prop := (x + 2) * (x - 3) ≤ 0
def q (m : ℝ) (x : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

-- Problem (I)
theorem problem_I (m : ℝ) : m > 0 → (∀ x : ℝ, q m x → p x) → 0 < m ∧ m ≤ 2 := by
  sorry

-- Problem (II)
theorem problem_II (x : ℝ) : 7 > 0 → 
  (p x ∨ q 7 x) ∧ ¬(p x ∧ q 7 x) → 
  (-6 ≤ x ∧ x < -2) ∨ (3 < x ∧ x ≤ 8) := by
  sorry

end problem_I_problem_II_l287_287569


namespace rich_walked_distance_l287_287762

def total_distance_walked (d1 d2 : ℕ) := 
  d1 + d2 + 2 * (d1 + d2) + (d1 + d2 + 2 * (d1 + d2)) / 2

def distance_to_intersection (d1 d2 : ℕ) := 
  2 * (d1 + d2)

def distance_to_end_route (d1 d2 : ℕ) := 
  (d1 + d2 + distance_to_intersection d1 d2) / 2

def total_distance_one_way (d1 d2 : ℕ) := 
  (d1 + d2) + (distance_to_intersection d1 d2) + (distance_to_end_route d1 d2)

theorem rich_walked_distance
  (d1 : ℕ := 20)
  (d2 : ℕ := 200) :
  2 * total_distance_one_way d1 d2 = 1980 :=
by
  simp [total_distance_one_way, distance_to_intersection, distance_to_end_route, total_distance_walked]
  sorry

end rich_walked_distance_l287_287762


namespace geese_flock_size_l287_287665

theorem geese_flock_size : 
  ∃ x : ℕ, x + x + (x / 2) + (x / 4) + 1 = 100 ∧ x = 36 := 
by
  sorry

end geese_flock_size_l287_287665


namespace Humphrey_birds_l287_287743

-- Definitions for the given conditions:
def Marcus_birds : ℕ := 7
def Darrel_birds : ℕ := 9
def average_birds : ℕ := 9
def number_of_people : ℕ := 3

-- Proof statement
theorem Humphrey_birds : ∀ x : ℕ, (average_birds * number_of_people = Marcus_birds + Darrel_birds + x) → x = 11 :=
by
  intro x h
  sorry

end Humphrey_birds_l287_287743


namespace second_smallest_four_digit_pascal_l287_287649

theorem second_smallest_four_digit_pascal :
  ∃ (n k : ℕ), (1000 < Nat.choose n k) ∧ (Nat.choose n k = 1001) :=
by
  sorry

end second_smallest_four_digit_pascal_l287_287649


namespace non_allergic_children_l287_287134

theorem non_allergic_children (T : ℕ) (h1 : T / 2 = n) (h2 : ∀ m : ℕ, 10 = m) (h3 : ∀ k : ℕ, 10 = k) :
  10 = 10 :=
by
  sorry

end non_allergic_children_l287_287134


namespace probability_of_sum_perfect_square_or_prime_l287_287189

noncomputable def probability_perfect_square_or_prime : ℚ :=
  let outcomes := finset.product (finset.range 6) (finset.range 6) 
  let is_perfect_square_or_prime (x : ℕ) : Prop :=
    x = 2 ∨ x = 3 ∨ x = 5 ∨ x = 7 ∨ x = 11 ∨ x = 4 ∨ x = 9
  let favorable_outcomes := outcomes.filter (λ p, is_perfect_square_or_prime (p.1 + p.2 + 2))
  ((favorable_outcomes.card : ℚ) / (outcomes.card : ℚ))

theorem probability_of_sum_perfect_square_or_prime :
  probability_perfect_square_or_prime = 11 / 18 :=
sorry

end probability_of_sum_perfect_square_or_prime_l287_287189


namespace smallest_k_no_real_roots_l287_287024

theorem smallest_k_no_real_roots :
  ∀ (k : ℤ), (∀ x : ℝ, 3 * x * (k * x - 5) - x^2 + 7 ≠ 0) → k ≥ 4 :=
by
  sorry

end smallest_k_no_real_roots_l287_287024


namespace find_standard_eq_find_min_MP_l287_287416

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 2 = 1

theorem find_standard_eq (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : ellipse_eq (sqrt 2) 1)
  (h4 : ∀ (x : ℝ), x^2 / 4 + 1 / 2 = 1 → a^2 = 2 * b^2) :
  ellipse_eq x y :=
begin
  sorry
end

theorem find_min_MP (p x y : ℝ) (hx : |x| ≤ 2)
  (h1 : ellipse_eq x y) :

  (|p| ≤ 1 → ∃ (x_val y_val : ℝ), (x_val, y_val) ∈ ellipse_eq ∧ min_val = sqrt (2 - p^2) ∧ x = 2 * p) ∧
  (p > 1 → ∃ (x_val y_val : ℝ), (x_val, y_val) ∈ ellipse_eq ∧ min_val = |p - 2| ∧ x = 2) ∧
  (p < -1 → ∃ (x_val y_val : ℝ), (x_val, y_val) ∈ ellipse_eq ∧ min_val = |p + 2| ∧ x = -2) :=
begin
  sorry
end

end find_standard_eq_find_min_MP_l287_287416


namespace partial_fraction_product_l287_287987

theorem partial_fraction_product : 
  (∃ A B C : ℚ, 
    (∀ x : ℚ, x ≠ 3 ∧ x ≠ -3 ∧ x ≠ 5 → 
      (x^2 - 21) / ((x - 3) * (x + 3) * (x - 5)) = A / (x - 3) + B / (x + 3) + C / (x - 5))
      ∧ (A * B * C = -1/16)) := 
    sorry

end partial_fraction_product_l287_287987


namespace frac_two_over_x_values_l287_287286

theorem frac_two_over_x_values (x : ℝ) (h : 1 - 9 / x + 20 / (x ^ 2) = 0) :
  (2 / x = 1 / 2 ∨ 2 / x = 0.4) :=
sorry

end frac_two_over_x_values_l287_287286


namespace possible_values_of_a1_l287_287634

def sequence_satisfies_conditions (a : ℕ → ℕ) : Prop :=
  (∀ n ≥ 1, a n ≤ a (n + 1) ∧ a (n + 1) ≤ a n + 5) ∧
  (∀ n ≥ 1, n ∣ a n)

theorem possible_values_of_a1 (a : ℕ → ℕ) :
  sequence_satisfies_conditions a → ∃ k ≤ 26, a 1 = k :=
by
  sorry

end possible_values_of_a1_l287_287634


namespace find_g_neg_3_l287_287302

def g (x : ℤ) : ℤ :=
if x < 1 then 3 * x - 4 else x + 6

theorem find_g_neg_3 : g (-3) = -13 :=
by
  -- proof omitted: sorry
  sorry

end find_g_neg_3_l287_287302


namespace count_even_numbers_between_250_and_600_l287_287582

theorem count_even_numbers_between_250_and_600 : 
  ∃ n : ℕ, (n = 175 ∧ 
    ∀ k : ℕ, (250 < 2 * k ∧ 2 * k ≤ 600) ↔ (126 ≤ k ∧ k ≤ 300)) :=
by
  sorry

end count_even_numbers_between_250_and_600_l287_287582


namespace minimize_quadratic_expression_l287_287259

theorem minimize_quadratic_expression : ∃ x : ℝ, (∀ y : ℝ, 3 * x^2 - 18 * x + 7 ≤ 3 * y^2 - 18 * y + 7) ∧ x = 3 :=
by
  sorry

end minimize_quadratic_expression_l287_287259


namespace symmetric_y_axis_l287_287097

theorem symmetric_y_axis (a b : ℝ) (h₁ : a = -4) (h₂ : b = 3) : a - b = -7 :=
by
  rw [h₁, h₂]
  norm_num

end symmetric_y_axis_l287_287097


namespace number_of_integers_between_25_and_36_l287_287489

theorem number_of_integers_between_25_and_36 :
  {n : ℕ | 25 < n ∧ n < 36}.card = 10 :=
by
  sorry

end number_of_integers_between_25_and_36_l287_287489


namespace gcd_lcm_product_360_l287_287772

theorem gcd_lcm_product_360 (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) :
    {d : ℕ | d = Nat.gcd a b } =
    {1, 2, 4, 8, 3, 6, 12, 24} := 
by
  sorry

end gcd_lcm_product_360_l287_287772


namespace find_n_l287_287888

-- Define the parameters of the arithmetic sequence
def a1 : ℤ := 1
def d : ℤ := 3
def a_n : ℤ := 298

-- The general formula for the nth term in an arithmetic sequence
def an (n : ℕ) : ℤ := a1 + (n - 1) * d

-- The theorem to prove that n equals 100 given the conditions
theorem find_n (n : ℕ) (h : an n = a_n) : n = 100 :=
by
  sorry

end find_n_l287_287888


namespace first_ant_arrives_first_l287_287783

noncomputable def time_crawling (d v : ℝ) : ℝ := d / v

noncomputable def time_riding_caterpillar (d v : ℝ) : ℝ := (d / 2) / (v / 2)

noncomputable def time_riding_grasshopper (d v : ℝ) : ℝ := (d / 2) / (10 * v)

noncomputable def time_ant1 (d v : ℝ) : ℝ := time_crawling d v

noncomputable def time_ant2 (d v : ℝ) : ℝ := time_riding_caterpillar d v + time_riding_grasshopper d v

theorem first_ant_arrives_first (d v : ℝ) (h_v_pos : 0 < v): time_ant1 d v < time_ant2 d v := by
  -- provide the justification for the theorem here
  sorry

end first_ant_arrives_first_l287_287783


namespace sequence_correct_l287_287095

def seq_formula (n : ℕ) : ℚ := 3/2 + (-1)^n * 11/2

theorem sequence_correct (n : ℕ) :
  (n % 2 = 0 ∧ seq_formula n = 7) ∨ (n % 2 = 1 ∧ seq_formula n = -4) :=
by
  sorry

end sequence_correct_l287_287095


namespace total_course_selection_schemes_l287_287963

theorem total_course_selection_schemes :
  let n_physical_education := 4
  let n_art := 4
  let total_courses := n_physical_education + n_art in
  let choose2_courses := (Nat.choose n_physical_education 1) * (Nat.choose n_art 1)
  let choose3_courses := (Nat.choose n_physical_education 2 * Nat.choose n_art 1) + (Nat.choose n_physical_education 1 * Nat.choose n_art 2) in
  total_courses = n_physical_education + n_art →
  choose2_courses + choose3_courses = 64 :=
by
  intros n_physical_education n_art total_courses choose2_courses choose3_courses h
  have h_choose2_courses: choose2_courses = 16 := by
    simp [n_physical_education, n_art, Nat.choose]
  have h_choose3_courses: choose3_courses = 48 := by
    simp [n_physical_education, n_art, Nat.choose]
  rw [h_choose2_courses, h_choose3_courses]
  exact Nat.add_eq_right.2 rfl

end total_course_selection_schemes_l287_287963


namespace polynomial_factorization_proof_l287_287651

noncomputable def factorizable_binary_quadratic (m : ℚ) : Prop :=
  ∃ (a b : ℚ), (3*a - 5*b = 17) ∧ (a*b = -4) ∧ (m = 2*a + 3*b)

theorem polynomial_factorization_proof :
  ∀ (m : ℚ), factorizable_binary_quadratic m ↔ (m = 5 ∨ m = -58 / 15) :=
by
  sorry

end polynomial_factorization_proof_l287_287651


namespace min_shift_symmetric_y_axis_l287_287765

theorem min_shift_symmetric_y_axis :
  ∃ (m : ℝ), m = 7 * Real.pi / 6 ∧ 
             (∀ x : ℝ, 2 * Real.cos (x + Real.pi / 3) = 2 * Real.cos (x + Real.pi / 3 + m)) ∧ 
             m > 0 :=
by
  sorry

end min_shift_symmetric_y_axis_l287_287765


namespace difference_even_odd_matchings_l287_287206

open Nat

def even_matchings (N : ℕ) (points: Finset (Fin (2 * N))) : Nat := sorry
def odd_matchings (N : ℕ) (points: Finset (Fin (2 * N))) : Nat := sorry

theorem difference_even_odd_matchings (N : ℕ) (points: Finset (Fin (2 * N))) :
  |even_matchings N points - odd_matchings N points| = 1 := sorry

end difference_even_odd_matchings_l287_287206


namespace inradius_of_equal_area_and_perimeter_l287_287728

theorem inradius_of_equal_area_and_perimeter
  (a b c : ℝ)
  (A : ℝ)
  (h1 : A = a + b + c)
  (s : ℝ := (a + b + c) / 2)
  (h2 : A = s * (2 * A / (a + b + c))) :
  ∃ r : ℝ, r = 2 := by
  sorry

end inradius_of_equal_area_and_perimeter_l287_287728


namespace fraction_value_l287_287020

theorem fraction_value : (1 + 3 + 5) / (10 + 6 + 2) = 1 / 2 := 
by
  sorry

end fraction_value_l287_287020


namespace p_implies_q_not_q_implies_p_l287_287752

def p (a : ℝ) := a = Real.sqrt 2

def q (a : ℝ) := ∀ x y : ℝ, y = -(x : ℝ) → (x^2 + (y - a)^2 = 1)

theorem p_implies_q_not_q_implies_p (a : ℝ) : (p a → q a) ∧ (¬(q a → p a)) := 
    sorry

end p_implies_q_not_q_implies_p_l287_287752


namespace remainder_3_pow_20_div_5_l287_287320

theorem remainder_3_pow_20_div_5 : (3 ^ 20) % 5 = 1 := 
by {
  sorry
}

end remainder_3_pow_20_div_5_l287_287320


namespace non_congruent_triangles_with_perimeter_11_l287_287867

theorem non_congruent_triangles_with_perimeter_11 :
  ∃ (triangle_count : ℕ), 
    triangle_count = 3 ∧ 
    ∀ (a b c : ℕ), 
      a + b + c = 11 → 
      a + b > c ∧ b + c > a ∧ a + c > b → 
      ∃ (t₁ t₂ t₃ : (ℕ × ℕ × ℕ)),
        (t₁ = (2, 4, 5) ∨ t₁ = (3, 4, 4) ∨ t₁ = (3, 3, 5)) ∧ 
        (t₂ = (2, 4, 5) ∨ t₂ = (3, 4, 4) ∨ t₂ = (3, 3, 5)) ∧ 
        (t₃ = (2, 4, 5) ∨ t₃ = (3, 4, 4) ∨ t₃ = (3, 3, 5)) ∧
        t₁ ≠ t₂ ∧ t₂ ≠ t₃ ∧ t₁ ≠ t₃

end non_congruent_triangles_with_perimeter_11_l287_287867


namespace aqua_park_earnings_l287_287539

/-- Define the costs and groups of visitors. --/
def admission_fee : ℕ := 12
def tour_fee : ℕ := 6
def group1_size : ℕ := 10
def group2_size : ℕ := 5

/-- Define the total earnings of the aqua park. --/
def total_earnings : ℕ := (admission_fee + tour_fee) * group1_size + admission_fee * group2_size

/-- Prove that the total earnings are $240. --/
theorem aqua_park_earnings : total_earnings = 240 :=
by
  -- proof steps would go here
  sorry

end aqua_park_earnings_l287_287539


namespace real_imag_equal_complex_l287_287099

/-- Given i is the imaginary unit, and a is a real number,
if the real part and the imaginary part of the complex number -3i(a+i) are equal,
then a = -1. -/
theorem real_imag_equal_complex (a : ℝ) (i : ℂ) (h_i : i * i = -1) 
    (h_eq : (3 : ℂ) = -(3 : ℂ) * a * i) : a = -1 :=
sorry

end real_imag_equal_complex_l287_287099


namespace train_speed_is_5400432_kmh_l287_287529

noncomputable def train_speed_kmh (time_to_pass_platform : ℝ) (time_to_pass_man : ℝ) (length_platform : ℝ) : ℝ :=
  let speed_m_per_s := length_platform / (time_to_pass_platform - time_to_pass_man)
  speed_m_per_s * 3.6

theorem train_speed_is_5400432_kmh :
  train_speed_kmh 35 20 225.018 = 54.00432 :=
by
  sorry

end train_speed_is_5400432_kmh_l287_287529


namespace right_angle_case_acute_angle_case_obtuse_angle_case_l287_287515

-- Definitions
def circumcenter (O : Type) (A B C : Type) : Prop := sorry -- Definition of circumcenter.

def orthocenter (H : Type) (A B C : Type) : Prop := sorry -- Definition of orthocenter.

noncomputable def R : ℝ := sorry -- Circumradius of the triangle.

-- Conditions
variables {A B C O H : Type}
  (h_circumcenter : circumcenter O A B C)
  (h_orthocenter : orthocenter H A B C)

-- The angles α β γ represent the angles of triangle ABC.
variables {α β γ : ℝ}

-- Statements
-- Case 1: ∠C = 90°
theorem right_angle_case (h_angle_C : γ = 90) (h_H_eq_C : H = C) (h_AB_eq_2R : AB = 2 * R) : AH + BH >= AB := by
  sorry

-- Case 2: ∠C < 90°
theorem acute_angle_case (h_angle_C_lt_90 : γ < 90) : O_in_triangle_AHB := by
  sorry

-- Case 3: ∠C > 90°
theorem obtuse_angle_case (h_angle_C_gt_90 : γ > 90) : AH + BH > 2 * R := by
  sorry

end right_angle_case_acute_angle_case_obtuse_angle_case_l287_287515


namespace leaf_distance_after_11_gusts_l287_287521

def distance_traveled (gusts : ℕ) (swirls : ℕ) (forward_per_gust : ℕ) (backward_per_swirl : ℕ) : ℕ :=
  (gusts * forward_per_gust) - (swirls * backward_per_swirl)

theorem leaf_distance_after_11_gusts :
  ∀ (forward_per_gust backward_per_swirl : ℕ),
  forward_per_gust = 5 →
  backward_per_swirl = 2 →
  distance_traveled 11 11 forward_per_gust backward_per_swirl = 33 :=
by
  intros forward_per_gust backward_per_swirl hfg hbs
  rw [hfg, hbs]
  unfold distance_traveled
  sorry

end leaf_distance_after_11_gusts_l287_287521


namespace upper_limit_of_range_l287_287022

theorem upper_limit_of_range (N : ℕ) :
  (∀ n : ℕ, (20 + n * 10 ≤ N) = (n < 198)) → N = 1990 :=
by
  sorry

end upper_limit_of_range_l287_287022


namespace law_firm_more_than_two_years_l287_287594

theorem law_firm_more_than_two_years (p_second p_not_first : ℝ) : 
  p_second = 0.30 →
  p_not_first = 0.60 →
  ∃ p_more_than_two_years : ℝ, p_more_than_two_years = 0.30 :=
by
  intros h1 h2
  use (p_not_first - p_second)
  rw [h1, h2]
  norm_num
  done

end law_firm_more_than_two_years_l287_287594


namespace greatest_divisor_arithmetic_sum_l287_287340

theorem greatest_divisor_arithmetic_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ d, d = 15 ∧ ∀ S : ℕ, S = 15 * x + 105 * c → d ∣ S :=
by 
  sorry

end greatest_divisor_arithmetic_sum_l287_287340


namespace greatest_divisor_sum_of_first_fifteen_terms_l287_287359

theorem greatest_divisor_sum_of_first_fifteen_terms 
  (x c : ℕ) (hx : x > 0) (hc : c > 0):
  ∃ d, d = 15 ∧ d ∣ (15*x + 105*c) :=
by
  existsi 15
  split
  . refl
  . apply Nat.dvd.intro
    existsi (x + 7*c)
    refl
  sorry

end greatest_divisor_sum_of_first_fifteen_terms_l287_287359


namespace exists_subset_sum_mod_p_l287_287298

theorem exists_subset_sum_mod_p (p : ℕ) (hp : Nat.Prime p) (A : Finset ℕ)
  (hA_card : A.card = p - 1) (hA : ∀ a ∈ A, a % p ≠ 0) : 
  ∀ n : ℕ, n < p → ∃ B ⊆ A, (B.sum id) % p = n :=
by
  sorry

end exists_subset_sum_mod_p_l287_287298


namespace casey_saving_l287_287236

-- Define the conditions
def cost_per_hour_first_employee : ℝ := 20
def cost_per_hour_second_employee : ℝ := 22
def subsidy_per_hour : ℝ := 6
def hours_per_week : ℝ := 40

-- Define the weekly cost calculations
def weekly_cost_first_employee := cost_per_hour_first_employee * hours_per_week
def effective_cost_per_hour_second_employee := cost_per_hour_second_employee - subsidy_per_hour
def weekly_cost_second_employee := effective_cost_per_hour_second_employee * hours_per_week

-- State the theorem
theorem casey_saving :
    weekly_cost_first_employee - weekly_cost_second_employee = 160 := 
by
  sorry

end casey_saving_l287_287236


namespace equation_relating_price_and_tax_and_discount_l287_287589

variable (c t d : ℚ)

theorem equation_relating_price_and_tax_and_discount
  (h1 : 1.30 * c * ((100 + t) / 100) * ((100 - d) / 100) = 351) :
    1.30 * c * (100 + t) * (100 - d) = 3510000 := by
  sorry

end equation_relating_price_and_tax_and_discount_l287_287589


namespace cistern_filling_time_l287_287811

theorem cistern_filling_time{F E : ℝ} (hF: F = 1 / 4) (hE: E = 1 / 9) :
  (1 / (F - E) = 7.2) :=
by
  rw [hF, hE]
  have net_rate := 0.25 - 1 / 9
  rw net_rate
  exact (1 / (0.25 - 1 / 9)) = 7.2
  sorry

end cistern_filling_time_l287_287811


namespace course_selection_schemes_l287_287959

theorem course_selection_schemes :
  let pe_courses := 4
  let art_courses := 4
  let total_courses := pe_courses + art_courses
  let choose (n k : ℕ) := Nat.choose n k
  let choose_2_courses := choose pe_courses 1 * choose art_courses 1
  let choose_3_courses := choose pe_courses 2 * choose art_courses 1 + choose pe_courses 1 * choose art_courses 2
  choose_2_courses + choose_3_courses = 64 :=
by
  let pe_courses := 4
  let art_courses := 4
  let total_courses := pe_courses + art_courses
  let choose (n k : ℕ) := Nat.choose n k
  let choose_2_courses := choose pe_courses 1 * choose art_courses 1
  let choose_3_courses := choose pe_courses 2 * choose art_courses 1 + choose pe_courses 1 * choose art_courses 2
  show choose_2_courses + choose_3_courses = 64 from sorry

end course_selection_schemes_l287_287959


namespace skateboard_price_after_discounts_l287_287305

-- Defining all necessary conditions based on the given problem.
def original_price : ℝ := 150
def discount1 : ℝ := 0.40 * original_price
def price_after_discount1 : ℝ := original_price - discount1
def discount2 : ℝ := 0.25 * price_after_discount1
def final_price : ℝ := price_after_discount1 - discount2

-- Goal: Prove that the final price after both discounts is $67.50.
theorem skateboard_price_after_discounts : final_price = 67.50 := by
  sorry

end skateboard_price_after_discounts_l287_287305


namespace percent_of_juniors_involved_in_sports_l287_287778

theorem percent_of_juniors_involved_in_sports
  (total_students : ℕ)
  (percent_juniors : ℝ)
  (juniors_in_sports : ℕ)
  (h1 : total_students = 500)
  (h2 : percent_juniors = 0.40)
  (h3 : juniors_in_sports = 140) :
  (juniors_in_sports : ℝ) / (total_students * percent_juniors) * 100 = 70 := 
by
  -- By conditions h1, h2, h3:
  sorry

end percent_of_juniors_involved_in_sports_l287_287778


namespace ab_eq_neg_one_l287_287903

variable (a b : ℝ)

-- Condition for the inequality (x >= 0) -> (0 ≤ x^4 - x^3 + ax + b ≤ (x^2 - 1)^2)
def condition (a b : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 0 → 
    0 ≤ x^4 - x^3 + a * x + b ∧ 
    x^4 - x^3 + a * x + b ≤ (x^2 - 1)^2

-- Main statement to prove that assuming the condition, a * b = -1
theorem ab_eq_neg_one (h : condition a b) : a * b = -1 := 
  sorry

end ab_eq_neg_one_l287_287903


namespace arrange_cubes_bound_l287_287330

def num_ways_to_arrange_cubes_into_solids (n : ℕ) : ℕ := sorry

theorem arrange_cubes_bound (n : ℕ) (h : n = (2015^100)) :
  10^14 < num_ways_to_arrange_cubes_into_solids n ∧
  num_ways_to_arrange_cubes_into_solids n < 10^15 := sorry

end arrange_cubes_bound_l287_287330


namespace greatest_divisor_sum_of_first_fifteen_terms_l287_287358

theorem greatest_divisor_sum_of_first_fifteen_terms 
  (x c : ℕ) (hx : x > 0) (hc : c > 0):
  ∃ d, d = 15 ∧ d ∣ (15*x + 105*c) :=
by
  existsi 15
  split
  . refl
  . apply Nat.dvd.intro
    existsi (x + 7*c)
    refl
  sorry

end greatest_divisor_sum_of_first_fifteen_terms_l287_287358


namespace integer_values_count_l287_287482

theorem integer_values_count (x : ℕ) (h1 : 5 < Real.sqrt x) (h2 : Real.sqrt x < 6) : 
  ∃ count : ℕ, count = 10 := 
by 
  sorry

end integer_values_count_l287_287482


namespace compare_fractions_l287_287545

theorem compare_fractions : (-2 / 7) > (-3 / 10) :=
sorry

end compare_fractions_l287_287545


namespace jonas_pairs_of_pants_l287_287146

theorem jonas_pairs_of_pants (socks pairs_of_shoes t_shirts new_socks : Nat) (P : Nat) :
  socks = 20 → pairs_of_shoes = 5 → t_shirts = 10 → new_socks = 35 →
  2 * (2 * socks + 2 * pairs_of_shoes + t_shirts + P) = 2 * (2 * socks + 2 * pairs_of_shoes + t_shirts) + 70 →
  P = 5 :=
by
  intros hs hps ht hr htotal
  sorry

end jonas_pairs_of_pants_l287_287146


namespace Shelby_fog_time_l287_287616

variable (x y : ℕ)

-- Conditions
def speed_sun := 7/12
def speed_rain := 5/12
def speed_fog := 1/4
def total_time := 60
def total_distance := 20

theorem Shelby_fog_time :
  ((speed_sun * (total_time - x - y)) + (speed_rain * x) + (speed_fog * y) = total_distance) → y = 45 :=
by
  sorry

end Shelby_fog_time_l287_287616


namespace combined_vacations_and_classes_l287_287110

-- Define the conditions
def Kelvin_classes : ℕ := 90
def Grant_vacations : ℕ := 4 * Kelvin_classes

-- The Lean 4 statement proving the combined total of vacations and classes
theorem combined_vacations_and_classes : Kelvin_classes + Grant_vacations = 450 := by
  sorry

end combined_vacations_and_classes_l287_287110


namespace final_value_after_determinant_and_addition_l287_287687

theorem final_value_after_determinant_and_addition :
  let a := 5
  let b := 7
  let c := 3
  let d := 4
  let det := a * d - b * c
  det + 3 = 2 :=
by
  sorry

end final_value_after_determinant_and_addition_l287_287687


namespace det_of_commuting_matrices_l287_287901

theorem det_of_commuting_matrices (n : ℕ) (hn : n ≥ 2) (A B : Matrix (Fin n) (Fin n) ℝ)
  (hA : A * A = -1) (hAB : A * B = B * A) : 
  0 ≤ B.det := 
sorry

end det_of_commuting_matrices_l287_287901


namespace triangle_area_with_median_l287_287084

theorem triangle_area_with_median (a b m : ℝ) (area : ℝ) 
  (h_a : a = 6) (h_b : b = 8) (h_m : m = 5) : 
  area = 24 :=
sorry

end triangle_area_with_median_l287_287084


namespace largest_root_divisible_by_17_l287_287150

theorem largest_root_divisible_by_17 (a : ℝ) (h : Polynomial.eval a (Polynomial.C 1 + Polynomial.C (-3) * Polynomial.X^2 + Polynomial.X^3) = 0) (root_large : ∀ x ∈ {b | Polynomial.eval b (Polynomial.C 1 + Polynomial.C (-3) * Polynomial.X^2 + Polynomial.X^3) = 0}, x ≤ a) :
  a^1788 % 17 = 0 ∧ a^1988 % 17 = 0 :=
by
  sorry

end largest_root_divisible_by_17_l287_287150


namespace red_peppers_weight_l287_287398

theorem red_peppers_weight (total_weight green_weight : ℝ) (h1 : total_weight = 5.666666667) (h2 : green_weight = 2.8333333333333335) : 
  total_weight - green_weight = 2.8333333336666665 :=
by
  sorry

end red_peppers_weight_l287_287398


namespace smallest_n_for_two_distinct_tuples_l287_287939

theorem smallest_n_for_two_distinct_tuples : ∃ (n : ℕ), n = 1729 ∧ 
  (∃ (x1 y1 x2 y2 : ℕ), x1 ≠ x2 ∧ y1 ≠ y2 ∧ n = x1^3 + y1^3 ∧ n = x2^3 + y2^3 ∧ 0 < x1 ∧ 0 < y1 ∧ 0 < x2 ∧ 0 < y2) := sorry

end smallest_n_for_two_distinct_tuples_l287_287939


namespace tv_show_years_l287_287951

theorem tv_show_years (s1 s2 s3 : ℕ) (e1 e2 e3 : ℕ) (avg : ℕ) :
  s1 = 8 → e1 = 15 →
  s2 = 4 → e2 = 20 →
  s3 = 2 → e3 = 12 →
  avg = 16 →
  (s1 * e1 + s2 * e2 + s3 * e3) / avg = 14 := by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end tv_show_years_l287_287951


namespace condition_swap_l287_287859

variable {p q : Prop}

theorem condition_swap (h : ¬ p → q) (nh : ¬ (¬ p ↔ q)) : (p → ¬ q) ∧ ¬ (¬ (p ↔ ¬ q)) :=
by
  sorry

end condition_swap_l287_287859


namespace student_marks_equals_125_l287_287972

-- Define the maximum marks
def max_marks : ℕ := 500

-- Define the percentage required to pass
def pass_percentage : ℚ := 33 / 100

-- Define the marks required to pass
def pass_marks : ℚ := pass_percentage * max_marks

-- Define the marks by which the student failed
def fail_by_marks : ℕ := 40

-- Define the obtained marks by the student
def obtained_marks : ℚ := pass_marks - fail_by_marks

-- Prove that the obtained marks are 125
theorem student_marks_equals_125 : obtained_marks = 125 := by
  sorry

end student_marks_equals_125_l287_287972


namespace sunday_price_correct_l287_287469

def original_price : ℝ := 250
def first_discount_rate : ℝ := 0.60
def second_discount_rate : ℝ := 0.25
def discounted_price : ℝ := original_price * (1 - first_discount_rate)
def sunday_price : ℝ := discounted_price * (1 - second_discount_rate)

theorem sunday_price_correct :
  sunday_price = 75 := by
  sorry

end sunday_price_correct_l287_287469


namespace total_number_of_workers_is_49_l287_287657

-- Definitions based on the conditions
def avg_salary_all_workers := 8000
def num_technicians := 7
def avg_salary_technicians := 20000
def avg_salary_non_technicians := 6000

-- Prove that the total number of workers in the workshop is 49
theorem total_number_of_workers_is_49 :
  ∃ W, (avg_salary_all_workers * W = avg_salary_technicians * num_technicians + avg_salary_non_technicians * (W - num_technicians)) ∧ W = 49 := 
sorry

end total_number_of_workers_is_49_l287_287657


namespace each_friend_pays_18_l287_287173

theorem each_friend_pays_18 (total_bill : ℝ) (silas_share : ℝ) (tip_fraction : ℝ) (num_friends : ℕ) (silas : ℕ) (remaining_friends : ℕ) :
  total_bill = 150 →
  silas_share = total_bill / 2 →
  tip_fraction = 0.1 →
  num_friends = 6 →
  remaining_friends = num_friends - 1 →
  silas = 1 →
  (total_bill - silas_share + tip_fraction * total_bill) / remaining_friends = 18 :=
by
  intros
  sorry

end each_friend_pays_18_l287_287173


namespace damaged_potatoes_l287_287536

theorem damaged_potatoes (initial_potatoes : ℕ) (weight_per_bag : ℕ) (price_per_bag : ℕ) (total_sales : ℕ) :
  initial_potatoes = 6500 →
  weight_per_bag = 50 →
  price_per_bag = 72 →
  total_sales = 9144 →
  ∃ damaged_potatoes : ℕ, damaged_potatoes = initial_potatoes - (total_sales / price_per_bag) * weight_per_bag ∧
                               damaged_potatoes = 150 :=
by
  intros _ _ _ _ 
  exact sorry

end damaged_potatoes_l287_287536


namespace coeff_x5_of_expansion_l287_287624

theorem coeff_x5_of_expansion : 
  (Polynomial.coeff ((Polynomial.C (1 : ℤ)) * (Polynomial.X ^ 2 - Polynomial.X - Polynomial.C 2) ^ 3) 5) = -3 := 
by sorry

end coeff_x5_of_expansion_l287_287624


namespace probability_of_fourth_three_is_correct_l287_287311

noncomputable def p_plus_q : ℚ := 41 + 84

theorem probability_of_fourth_three_is_correct :
  let fair_die_prob := (1 / 6 : ℚ)
  let biased_die_prob := (1 / 2 : ℚ)
  -- Probability of rolling three threes with the fair die:
  let fair_die_three_three_prob := fair_die_prob ^ 3
  -- Probability of rolling three threes with the biased die:
  let biased_die_three_three_prob := biased_die_prob ^ 3
  -- Probability of rolling three threes in total:
  let total_three_three_prob := fair_die_three_three_prob + biased_die_three_three_prob
  -- Probability of using the fair die given three threes
  let fair_die_given_three := fair_die_three_three_prob / total_three_three_prob
  -- Probability of using the biased die given three threes
  let biased_die_given_three := biased_die_three_three_prob / total_three_three_prob
  -- Probability of rolling another three:
  let fourth_three_prob := fair_die_given_three * fair_die_prob + biased_die_given_three * biased_die_prob
  -- Simplifying fraction
  let result_fraction := (41 / 84 : ℚ)
  -- Final answer p + q is 125
  p_plus_q = 125 ∧ fourth_three_prob = result_fraction
:= by
  sorry

end probability_of_fourth_three_is_correct_l287_287311


namespace paint_cost_per_quart_l287_287432

-- Definitions of conditions
def edge_length (cube_edge_length : ℝ) : Prop := cube_edge_length = 10
def surface_area (s_area : ℝ) : Prop := s_area = 6 * (10^2)
def coverage_per_quart (coverage : ℝ) : Prop := coverage = 120
def total_cost (cost : ℝ) : Prop := cost = 16
def required_quarts (quarts : ℝ) : Prop := quarts = 600 / 120
def cost_per_quart (cost : ℝ) (quarts : ℝ) (price_per_quart : ℝ) : Prop := price_per_quart = cost / quarts

-- Main theorem statement translating the problem into Lean
theorem paint_cost_per_quart {cube_edge_length s_area coverage cost quarts price_per_quart : ℝ} :
  edge_length cube_edge_length →
  surface_area s_area →
  coverage_per_quart coverage →
  total_cost cost →
  required_quarts quarts →
  quarts = s_area / coverage →
  cost_per_quart cost quarts 3.20 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- proof will go here
  sorry

end paint_cost_per_quart_l287_287432


namespace total_population_l287_287329

-- Define the conditions
variables (T G Td Lb : ℝ)

-- Given conditions and the result
def conditions : Prop :=
  G = 1 / 2 * T ∧
  Td = 0.60 * G ∧
  Lb = 16000 ∧
  T = Td + G + Lb

-- Problem statement: Prove that the total population T is 80000
theorem total_population (h : conditions T G Td Lb) : T = 80000 :=
by
  sorry

end total_population_l287_287329


namespace find_omega_l287_287102

theorem find_omega (ω : ℝ) (h₀ : ω > 0) (h₁ : (π / ω = π / 2)) : ω = 2 :=
sorry

end find_omega_l287_287102


namespace abe_bob_same_color_prob_l287_287532

-- Definition of the given problem's conditions
def abe_jelly_beans := [(2, "green"), (1, "red")]
def bob_jelly_beans := [(2, "green"), (1, "yellow"), (2, "red"), (1, "blue")]

def prob_showing_green :=
  (2 / 3) * (2 / 6)

def prob_showing_red :=
  (1 / 3) * (2 / 6)

def prob_matching_colors :=
  prob_showing_green + prob_showing_red

theorem abe_bob_same_color_prob :
  prob_matching_colors = 1 / 3 :=
begin
  sorry
end

end abe_bob_same_color_prob_l287_287532


namespace pies_from_apples_l287_287314

theorem pies_from_apples (total_apples : ℕ) (percent_handout : ℝ) (apples_per_pie : ℕ) 
  (h_total : total_apples = 800) (h_percent : percent_handout = 0.65) (h_per_pie : apples_per_pie = 15) : 
  (total_apples * (1 - percent_handout)) / apples_per_pie = 18 := 
by 
  sorry

end pies_from_apples_l287_287314


namespace inequality_sum_l287_287907

theorem inequality_sum
  (x y z : ℝ)
  (h : abs (x * y * z) = 1) :
  (1 / (x^2 + x + 1) + 1 / (x^2 - x + 1)) +
  (1 / (y^2 + y + 1) + 1 / (y^2 - y + 1)) +
  (1 / (z^2 + z + 1) + 1 / (z^2 - z + 1)) ≤ 4 := 
sorry

end inequality_sum_l287_287907


namespace student_arrangements_l287_287223

theorem student_arrangements (students : Finset ℕ) (hcard : students.card = 7) :
  ∃ (A B : Finset ℕ), 
    A.card + B.card = 6 ∧ 
    2 ≤ A.card ∧ 2 ≤ B.card ∧ 
    (A ∪ B = students ∧ A ∩ B = ∅) →
    A.card = 2 ∨ A.card = 3 ∨ A.card = 4 ∨ B.card = 2 ∨ B.card = 3 ∨ B.card = 4 →
  (choose 6 2 * choose 4 4 + choose 6 3 * choose 3 3 + choose 6 4 * choose 2 2) * 7 = 350 :=
by
  sorry

end student_arrangements_l287_287223


namespace fraction_C_D_l287_287986

noncomputable def C : ℝ := ∑' n, if n % 6 = 0 then 0 else if n % 2 = 0 then ((-1)^(n/2 + 1) / (↑n^2)) else 0
noncomputable def D : ℝ := ∑' n, if n % 6 = 0 then ((-1)^(n/6 + 1) / (↑n^2)) else 0

theorem fraction_C_D : C / D = 37 := sorry

end fraction_C_D_l287_287986


namespace necklace_sum_l287_287609

theorem necklace_sum (H J x S : ℕ) (hH : H = 25) (h1 : H = J + 5) (h2 : x = J / 2) (h3 : S = 2 * H) : H + J + x + S = 105 :=
by 
  sorry

end necklace_sum_l287_287609


namespace product_of_two_numbers_l287_287922

theorem product_of_two_numbers 
  (x y : ℝ) 
  (h₁ : x - y = 8) 
  (h₂ : x^2 + y^2 = 160) 
  : x * y = 48 := 
sorry

end product_of_two_numbers_l287_287922


namespace digit_five_occurrences_l287_287459

/-- 
  Define that a 24-hour digital clock display shows times containing at least one 
  occurrence of the digit '5' a total of 450 times in a 24-hour period.
--/
def contains_digit_five (n : Nat) : Prop := 
  n / 10 = 5 ∨ n % 10 = 5

def count_times_with_digit_five : Nat :=
  let hours_with_five := 2 * 60  -- 05:00-05:59 and 15:00-15:59, each hour has 60 minutes
  let remaining_hours := 22 * 15 -- 22 hours, each hour has 15 minutes
  hours_with_five + remaining_hours

theorem digit_five_occurrences : count_times_with_digit_five = 450 := by
  sorry

end digit_five_occurrences_l287_287459


namespace greatest_common_divisor_sum_arithmetic_sequence_l287_287331

theorem greatest_common_divisor_sum_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, d = 15 ∧ ∀ (n : ℕ), n = 15 → ∀ k : ℕ, k = 15 ∧ 15 ∣ (15 * (x + 7 * c)) :=
by
  sorry

end greatest_common_divisor_sum_arithmetic_sequence_l287_287331


namespace inscribed_sphere_volume_l287_287573

theorem inscribed_sphere_volume
  (a : ℝ)
  (h_cube_surface_area : 6 * a^2 = 24) :
  (4 / 3) * Real.pi * (a / 2)^3 = (4 / 3) * Real.pi :=
by
  -- sorry to skip the actual proof
  sorry

end inscribed_sphere_volume_l287_287573


namespace woman_works_finish_days_l287_287038

theorem woman_works_finish_days (M W : ℝ) 
  (hm_work : ∀ n : ℝ, n * M = 1 / 100)
  (hw_work : ∀ men women : ℝ, (10 * M + 15 * women) * 6 = 1) :
  W = 1 / 225 :=
by
  have man_work := hm_work 1
  have woman_work := hw_work 10 W
  sorry

end woman_works_finish_days_l287_287038


namespace added_number_is_6_l287_287524

theorem added_number_is_6 : ∃ x : ℤ, (∃ y : ℤ, y = 9 ∧ (2 * y + x) * 3 = 72) → x = 6 := 
by
  sorry

end added_number_is_6_l287_287524


namespace five_fold_function_application_l287_287735

def f (x : ℤ) : ℤ :=
if x ≥ 0 then -x^2 + 1 else x + 9

theorem five_fold_function_application : f (f (f (f (f 2)))) = -17 :=
by
  sorry

end five_fold_function_application_l287_287735


namespace asymptotes_of_hyperbola_l287_287269

theorem asymptotes_of_hyperbola (a : ℝ) :
  (∃ a : ℝ, 9 + a = 13) →
  (∀ x y : ℝ, (x^2 / 9 - y^2 / a = 1) → (a = 4)) →
  (forall (x y : ℝ), (x^2 / 9 - y^2 / 4 = 0) → 
    (y = (2/3) * x) ∨ (y = -(2/3) * x)) :=
by
  sorry

end asymptotes_of_hyperbola_l287_287269


namespace number_of_observations_is_14_l287_287185

theorem number_of_observations_is_14
  (mean_original : ℚ) (mean_new : ℚ) (original_sum : ℚ) 
  (corrected_sum : ℚ) (n : ℚ)
  (h1 : mean_original = 36)
  (h2 : mean_new = 36.5)
  (h3 : corrected_sum = original_sum + 7)
  (h4 : mean_new = corrected_sum / n)
  (h5 : original_sum = mean_original * n) :
  n = 14 :=
by
  -- Here goes the proof
  sorry

end number_of_observations_is_14_l287_287185


namespace find_T_l287_287454

variables (h K T : ℝ)
variables (h_val : 4 * h * 7 + 2 = 58)
variables (K_val : K = 9)

theorem find_T : T = 74 :=
by
  sorry

end find_T_l287_287454


namespace tangent_difference_problem_l287_287711

-- Define the conditions and the problem statement
theorem tangent_difference_problem (α : ℝ) (hα1 : α ∈ set.Ioo (π / 2) π) (hα2 : Real.sin α = 3 / 5) :
  Real.tan (α - π / 4) = -7 :=
sorry

end tangent_difference_problem_l287_287711


namespace probability_A_to_B_in_8_moves_l287_287384

-- Define vertices
inductive Vertex : Type
| A | B | C | D | E | F

open Vertex

-- Define the probability of ending up at Vertex B after 8 moves starting from Vertex A
noncomputable def probability_at_B_after_8_moves : ℚ :=
  let prob := (3 : ℚ) / 16
  prob

-- Theorem statement
theorem probability_A_to_B_in_8_moves :
  (probability_at_B_after_8_moves = (3 : ℚ) / 16) :=
by
  -- Proof to be provided
  sorry

end probability_A_to_B_in_8_moves_l287_287384


namespace graph_of_equation_is_two_intersecting_lines_l287_287241

theorem graph_of_equation_is_two_intersecting_lines :
  ∀ x y : ℝ, (x + 3 * y) ^ 3 = x ^ 3 + 9 * y ^ 3 ↔ (x = 0 ∨ y = 0 ∨ x + 3 * y = 0) :=
by
  sorry

end graph_of_equation_is_two_intersecting_lines_l287_287241


namespace count_integers_between_25_and_36_l287_287484

theorem count_integers_between_25_and_36 :
  {x : ℤ | 25 < x ∧ x < 36}.finite.card = 10 :=
by
  sorry

end count_integers_between_25_and_36_l287_287484


namespace equations_of_line_l287_287253

variables (x y : ℝ)

-- Given conditions
def passes_through_point (P : ℝ × ℝ) (x y : ℝ) := (x, y) = P

def has_equal_intercepts_on_axes (f : ℝ → ℝ) :=
  ∃ z : ℝ, z ≠ 0 ∧ f z = 0 ∧ f 0 = z

-- The proof problem statement
theorem equations_of_line (P : ℝ × ℝ) (hP : passes_through_point P 2 (-3)) (h : has_equal_intercepts_on_axes (λ x => -x / (x / 2))) :
  (x + y + 1 = 0) ∨ (3 * x + 2 * y = 0) := 
sorry

end equations_of_line_l287_287253


namespace solve_x_l287_287558

theorem solve_x :
  ∀ (x y z w : ℤ),
    x = y + 7 →
    y = z + 15 →
    z = w + 25 →
    w = 65 →
    x = 112 :=
by
  intros x y z w
  intros h1 h2 h3 h4
  sorry

end solve_x_l287_287558


namespace heaviest_weight_is_aq3_l287_287021

variable (a q : ℝ) (h : 0 < a) (hq : 1 < q)

theorem heaviest_weight_is_aq3 :
  let w1 := a
  let w2 := a * q
  let w3 := a * q^2
  let w4 := a * q^3
  w4 > w3 ∧ w4 > w2 ∧ w4 > w1 ∧ w1 + w4 > w2 + w3 :=
by
  sorry

end heaviest_weight_is_aq3_l287_287021


namespace range_of_a_l287_287410

noncomputable def f (a x : ℝ) : ℝ := x^2 + a * Real.log x - a * x

theorem range_of_a (a : ℝ) (h : a > 0) : 
  (∀ x : ℝ, 0 < x → 0 ≤ 2 * x^2 - a * x + a) ↔ 0 < a ∧ a ≤ 8 :=
by
  sorry

end range_of_a_l287_287410


namespace Hallie_earnings_l287_287580

theorem Hallie_earnings :
  let w := 10
  let hM := 7
  let tM := 18
  let hT := 5
  let tT := 12
  let hW := 7
  let tW := 20
  let mondayEarnings := hM * w + tM
  let tuesdayEarnings := hT * w + tT
  let wednesdayEarnings := hW * w + tW
  let totalEarnings := mondayEarnings + tuesdayEarnings + wednesdayEarnings
  totalEarnings = 240 := by {
    let w := 10
    let hM := 7
    let tM := 18
    let hT := 5
    let tT := 12
    let hW := 7
    let tW := 20
    let mondayEarnings := hM * w + tM
    let tuesdayEarnings := hT * w + tT
    let wednesdayEarnings := hW * w + tW
    let totalEarnings := mondayEarnings + tuesdayEarnings + wednesdayEarnings
    sorry
  }

end Hallie_earnings_l287_287580


namespace problem_1_problem_2_problem_3_l287_287575

-- Define the function f(x)
def f (x : ℝ) (k : ℝ) : ℝ :=
  x^2 + x⁻² + k * (x - x⁻¹)

-- Problem 1: Prove that k = 0 if f(x) is an even function
theorem problem_1 (k : ℝ) (h_even : ∀ x : ℝ, f (-x) k = f x k) : k = 0 :=
sorry

-- Problem 2: Prove that f(x) is monotonically increasing in the interval (1, +∞) when k = 0
theorem problem_2 (h_inc : ∀ x y : ℝ, 1 < x → 1 < y → x < y → f x 0 < f y 0) : true :=
sorry

-- Problem 3: Prove that the range of k is -4 ≤ k < -1 if the maximum value of g(x) = |f(x)| in the interval [1, |k|] is 2
theorem problem_3 (h_max : ∀ k : ℝ, (∀ (x : ℝ), 1 ≤ x → x ≤ |k| → abs (f x k) ≤ 2) → -4 ≤ k ∧ k < -1) : true :=
sorry

end problem_1_problem_2_problem_3_l287_287575


namespace find_paintings_l287_287746

noncomputable def cost_painting (P : ℕ) : ℝ := 40 * P
noncomputable def cost_toy : ℝ := 20 * 8
noncomputable def total_cost (P : ℕ) : ℝ := cost_painting P + cost_toy

noncomputable def sell_painting (P : ℕ) : ℝ := 36 * P
noncomputable def sell_toy : ℝ := 17 * 8
noncomputable def total_sell (P : ℕ) : ℝ := sell_painting P + sell_toy

noncomputable def total_loss (P : ℕ) : ℝ := total_cost P - total_sell P

theorem find_paintings : ∀ (P : ℕ), total_loss P = 64 → P = 10 :=
by
  intros P h
  sorry

end find_paintings_l287_287746


namespace product_gcd_lcm_is_correct_l287_287196

-- Define the numbers
def a := 15
def b := 75

-- Definitions related to GCD and LCM
def gcd_ab := Nat.gcd a b
def lcm_ab := Nat.lcm a b
def product_gcd_lcm := gcd_ab * lcm_ab

-- Theorem stating the product of GCD and LCM of a and b is 1125
theorem product_gcd_lcm_is_correct : product_gcd_lcm = 1125 := by
  sorry

end product_gcd_lcm_is_correct_l287_287196


namespace somu_one_fifth_age_back_l287_287621

theorem somu_one_fifth_age_back {S F Y : ℕ}
  (h1 : S = 16)
  (h2 : S = F / 3)
  (h3 : S - Y = (F - Y) / 5) :
  Y = 8 :=
by
  sorry

end somu_one_fifth_age_back_l287_287621


namespace roots_of_equation_in_interval_l287_287277

theorem roots_of_equation_in_interval (f : ℝ → ℝ) (interval : Set ℝ) (n_roots : ℕ) :
  (∀ x ∈ interval, f x = 8 * x * (1 - 2 * x^2) * (8 * x^4 - 8 * x^2 + 1) - 1) →
  (interval = Set.Icc 0 1) →
  (n_roots = 4) :=
by
  intros f_eq interval_eq
  sorry

end roots_of_equation_in_interval_l287_287277


namespace sum_of_ages_is_32_l287_287790

-- Define the values and conditions given in the problem
def viggo_age_when_brother_was_2 (brother_age : ℕ) : ℕ := 10 + 2 * brother_age
def age_difference (viggo_age_brother_2 : ℕ) (brother_age : ℕ) : ℕ := viggo_age_brother_2 - brother_age
def current_viggo_age (current_brother_age : ℕ) (difference : ℕ) := current_brother_age + difference

-- State the main theorem
theorem sum_of_ages_is_32 : 
  let brother_age_when_2 := 2 in
  let current_brother_age := 10 in
  let viggo_age_when_2 := viggo_age_when_brother_was_2 brother_age_when_2 in
  let difference := age_difference viggo_age_when_2 brother_age_when_2 in
  current_viggo_age current_brother_age difference + current_brother_age = 32 := 
by
  sorry

end sum_of_ages_is_32_l287_287790


namespace maximize_profits_l287_287040

variable (m : ℝ) (x : ℝ)

def w1 (m x : ℝ) := (8 - m) * x - 30
def w2 (x : ℝ) := -0.01 * x^2 + 8 * x - 80

theorem maximize_profits : 
  (4 ≤ m ∧ m < 5.1 → ∀ x, 0 ≤ x ∧ x ≤ 500 → w1 m x ≥ w2 x) ∧
  (m = 5.1 → ∀ x ≤ 300, w1 m 500 = w2 300) ∧
  (m > 5.1 ∧ m ≤ 6 → ∀ x, 0 ≤ x ∧ x ≤ 300 → w2 x ≥ w1 m x) :=
  sorry

end maximize_profits_l287_287040


namespace square_area_eq_1296_l287_287976

theorem square_area_eq_1296 (x : ℝ) (side : ℝ) (h1 : side = 6 * x - 18) (h2 : side = 3 * x + 9) : side ^ 2 = 1296 := sorry

end square_area_eq_1296_l287_287976


namespace possible_age_of_youngest_child_l287_287214

noncomputable def valid_youngest_age (father_fee : ℝ) (child_fee_per_year : ℝ) (total_bill : ℝ) (triplet_age : ℝ) : ℝ :=
  total_bill - father_fee -  (3 * triplet_age * child_fee_per_year)

theorem possible_age_of_youngest_child (father_fee : ℝ) (child_fee_per_year : ℝ) (total_bill : ℝ) (t y : ℝ)
  (h1 : father_fee = 16)
  (h2 : child_fee_per_year = 0.8)
  (h3 : total_bill = 43.2)
  (age_condition : y = (total_bill - father_fee) / child_fee_per_year - 3 * t) :
  y = 1 ∨ y = 4 :=
by
  sorry

end possible_age_of_youngest_child_l287_287214


namespace determine_1000g_weight_l287_287942

-- Define the weights
def weights : List ℕ := [1000, 1001, 1002, 1004, 1007]

-- Define the weight sets
def Group1 : List ℕ := [weights.get! 0, weights.get! 1]
def Group2 : List ℕ := [weights.get! 2, weights.get! 3]
def Group3 : List ℕ := [weights.get! 4]

-- Definition to choose the lighter group or determine equality
def lighterGroup (g1 g2 : List ℕ) : List ℕ :=
  if g1.sum = g2.sum then Group3 else if g1.sum < g2.sum then g1 else g2

-- Determine the 1000 g weight functionally
def identify1000gWeightUsing3Weighings : ℕ :=
  let firstWeighing := lighterGroup Group1 Group2
  if firstWeighing = Group3 then Group3.get! 0 else
  let remainingWeights := firstWeighing
  if remainingWeights.get! 0 = remainingWeights.get! 1 then Group3.get! 0
  else if remainingWeights.get! 0 < remainingWeights.get! 1 then remainingWeights.get! 0 else remainingWeights.get! 1

theorem determine_1000g_weight : identify1000gWeightUsing3Weighings = 1000 :=
sorry

end determine_1000g_weight_l287_287942


namespace find_segment_AD_length_l287_287293

noncomputable def segment_length_AD (A B C D X : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace X] :=
  ∃ (angle_BAD angle_ABC angle_BCD : Real)
    (length_AB length_CD : Real)
    (perpendicular : X) (angle_BAX angle_ABX : Real)
    (length_AX length_DX length_AD : Real),
    angle_BAD = 60 ∧
    angle_ABC = 30 ∧
    angle_BCD = 30 ∧
    length_AB = 15 ∧
    length_CD = 8 ∧
    angle_BAX = 30 ∧
    angle_ABX = 60 ∧
    length_AX = length_AB / 2 ∧
    length_DX = length_CD / 2 ∧
    length_AD = length_AX - length_DX ∧
    length_AD = 3.5

theorem find_segment_AD_length (A B C D X : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace X] : segment_length_AD A B C D X :=
by
  sorry

end find_segment_AD_length_l287_287293


namespace example_number_is_not_octal_l287_287825

-- Define a predicate that checks if a digit is valid in the octal system
def is_octal_digit (d : ℕ) : Prop :=
  d < 8

-- Define a predicate that checks if all digits in a number represented as list of ℕ are valid octal digits
def is_octal_number (n : List ℕ) : Prop :=
  ∀ d ∈ n, is_octal_digit d

-- Example number represented as a list of its digits
def example_number : List ℕ := [2, 8, 5, 3]

-- The statement we aim to prove
theorem example_number_is_not_octal : ¬ is_octal_number example_number := by
  -- Proof goes here
  sorry

end example_number_is_not_octal_l287_287825


namespace intersection_condition_l287_287424

noncomputable def M : Set ℝ := {x | -1 ≤ x ∧ x < 2}
noncomputable def N (k : ℝ) : Set ℝ := {x | x ≤ k}

theorem intersection_condition (k : ℝ) (h : M ⊆ N k) : k ≥ 2 :=
  sorry

end intersection_condition_l287_287424


namespace connections_required_l287_287137

theorem connections_required (n : ℕ) (k : ℕ) (h_n : n = 30) (h_k : k = 4) :
  (n * k) / 2 = 60 := by
  sorry

end connections_required_l287_287137


namespace find_m_l287_287838

theorem find_m (m n : ℤ) (h : 21 * (m + n) + 21 = 21 * (-m + n) + 21) : m = 0 :=
sorry

end find_m_l287_287838


namespace evaluate_expression_l287_287652

theorem evaluate_expression (a b : ℝ) (h : (1/2 * a * (1:ℝ)^3 - 3 * b * 1 + 4 = 9)) :
  (1/2 * a * (-1:ℝ)^3 - 3 * b * (-1) + 4 = -1) := by
sorry

end evaluate_expression_l287_287652


namespace arcsin_sqrt3_div_2_eq_pi_div_3_l287_287548

theorem arcsin_sqrt3_div_2_eq_pi_div_3 : Real.arcsin (Real.sqrt 3 / 2) = Real.pi / 3 :=
by
  sorry

end arcsin_sqrt3_div_2_eq_pi_div_3_l287_287548


namespace profit_percentage_l287_287032

-- Given conditions
def CP : ℚ := 25 / 15
def SP : ℚ := 32 / 12

-- To prove profit percentage is 60%
theorem profit_percentage (CP SP : ℚ) (hCP : CP = 25 / 15) (hSP : SP = 32 / 12) :
  (SP - CP) / CP * 100 = 60 := 
by 
  sorry

end profit_percentage_l287_287032


namespace monotonicity_intervals_max_m_value_l287_287574

noncomputable def f (x : ℝ) : ℝ :=  (3 / 2) * x^2 - 3 * Real.log x

theorem monotonicity_intervals :
  (∀ x > (1:ℝ), ∃ ε > (0:ℝ), ∀ y, x < y → y < x + ε → f x < f y)
  ∧ (∀ x, (0:ℝ) < x → x < (1:ℝ) → ∃ ε > (0:ℝ), ∀ y, x - ε < y → y < x → f y < f x) :=
by sorry

theorem max_m_value (m : ℤ) (h : ∀ x > (1:ℝ), f (x * Real.log x + 2 * x - 1) > f (↑m * (x - 1))) :
  m ≤ 4 :=
by sorry

end monotonicity_intervals_max_m_value_l287_287574


namespace school_club_members_l287_287179

theorem school_club_members :
  ∃ n : ℕ, 200 ≤ n ∧ n ≤ 300 ∧
  n % 6 = 3 ∧
  n % 8 = 5 ∧
  n % 9 = 7 ∧
  n = 269 :=
by
  existsi 269
  sorry

end school_club_members_l287_287179


namespace total_rabbits_correct_l287_287402

def initial_breeding_rabbits : ℕ := 10
def kittens_first_spring : ℕ := initial_breeding_rabbits * 10
def adopted_first_spring : ℕ := kittens_first_spring / 2
def returned_adopted_first_spring : ℕ := 5
def total_rabbits_after_first_spring : ℕ :=
  initial_breeding_rabbits + (kittens_first_spring - adopted_first_spring + returned_adopted_first_spring)

def kittens_second_spring : ℕ := 60
def adopted_second_spring : ℕ := kittens_second_spring * 40 / 100
def returned_adopted_second_spring : ℕ := 10
def total_rabbits_after_second_spring : ℕ :=
  total_rabbits_after_first_spring + (kittens_second_spring - adopted_second_spring + returned_adopted_second_spring)

def breeding_rabbits_third_spring : ℕ := 12
def kittens_third_spring : ℕ := breeding_rabbits_third_spring * 8
def adopted_third_spring : ℕ := kittens_third_spring * 30 / 100
def returned_adopted_third_spring : ℕ := 3
def total_rabbits_after_third_spring : ℕ :=
  total_rabbits_after_second_spring + (kittens_third_spring - adopted_third_spring + returned_adopted_third_spring)

def kittens_fourth_spring : ℕ := breeding_rabbits_third_spring * 6
def adopted_fourth_spring : ℕ := kittens_fourth_spring * 20 / 100
def returned_adopted_fourth_spring : ℕ := 2
def total_rabbits_after_fourth_spring : ℕ :=
  total_rabbits_after_third_spring + (kittens_fourth_spring - adopted_fourth_spring + returned_adopted_fourth_spring)

theorem total_rabbits_correct : total_rabbits_after_fourth_spring = 242 := by
  sorry

end total_rabbits_correct_l287_287402


namespace students_per_table_l287_287156

theorem students_per_table (total_students tables students_bathroom students_canteen added_students exchange_students : ℕ) 
  (h1 : total_students = 47)
  (h2 : tables = 6)
  (h3 : students_bathroom = 3)
  (h4 : students_canteen = 3 * students_bathroom)
  (h5 : added_students = 2 * 4)
  (h6 : exchange_students = 3 + 3 + 3) :
  (total_students - (students_bathroom + students_canteen + added_students + exchange_students)) / tables = 3 := 
by 
  sorry

end students_per_table_l287_287156


namespace intersection_M_N_l287_287125

section

def M (x : ℝ) : Prop := sqrt x < 4
def N (x : ℝ) : Prop := 3 * x >= 1

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | (1 / 3) ≤ x ∧ x < 16} :=
by
  sorry

end

end intersection_M_N_l287_287125


namespace num_of_integers_satisfying_sqrt_condition_l287_287486

theorem num_of_integers_satisfying_sqrt_condition : 
  let S := { x : ℤ | 5 < Real.sqrt x ∧ x < 36 }
  in (S.card = 10) :=
begin
  let S := { x : ℤ | 25 < x ∧ x < 36 },
  sorry
end

end num_of_integers_satisfying_sqrt_condition_l287_287486


namespace vector_equation_l287_287037

variable {V : Type} [AddCommGroup V]

variables (A B C : V)

theorem vector_equation :
  (B - A) - 2 • (C - A) + (C - B) = (A - C) :=
by
  sorry

end vector_equation_l287_287037


namespace greatest_divisor_of_arithmetic_sequence_sum_l287_287345

theorem greatest_divisor_of_arithmetic_sequence_sum :
  ∀ (x c : ℕ), ∃ k : ℕ, k = 15 ∧ 15 ∣ (15 * x + 105 * c) :=
by
  intro x c
  exists 15
  split
  . rfl
  . sorry

end greatest_divisor_of_arithmetic_sequence_sum_l287_287345


namespace watermelon_yield_increase_l287_287387

noncomputable def yield_increase (initial_yield final_yield annual_increase_rate : ℝ) (years : ℕ) : ℝ :=
  initial_yield * (1 + annual_increase_rate) ^ years

theorem watermelon_yield_increase :
  ∀ (x : ℝ),
    (yield_increase 20 28.8 x 2 = 28.8) →
    (yield_increase 28.8 40 x 2 > 40) :=
by
  intros x hx
  have incEq : 20 * (1 + x) ^ 2 = 28.8 := hx
  sorry

end watermelon_yield_increase_l287_287387


namespace speed_of_stream_l287_287816

theorem speed_of_stream
  (b s : ℝ)
  (H1 : 120 = 2 * (b + s))
  (H2 : 60 = 2 * (b - s)) :
  s = 15 :=
by
  sorry

end speed_of_stream_l287_287816


namespace find_number_eq_36_l287_287378

theorem find_number_eq_36 (n : ℝ) (h : (n / 18) * (n / 72) = 1) : n = 36 :=
sorry

end find_number_eq_36_l287_287378


namespace kelly_needs_to_give_away_l287_287296

-- Definition of initial number of Sony games and desired number of Sony games left
def initial_sony_games : ℕ := 132
def desired_remaining_sony_games : ℕ := 31

-- The main theorem: The number of Sony games Kelly needs to give away to have 31 left
theorem kelly_needs_to_give_away : initial_sony_games - desired_remaining_sony_games = 101 := by
  sorry

end kelly_needs_to_give_away_l287_287296


namespace arithmetic_prog_triangle_l287_287590

theorem arithmetic_prog_triangle (a b c : ℝ) (h : a < b ∧ b < c ∧ 2 * b = a + c)
    (hα : ∀ t, t = a ↔ t = min a (min b c))
    (hγ : ∀ t, t = c ↔ t = max a (max b c)) :
    3 * (Real.tan (α / 2)) * (Real.tan (γ / 2)) = 1 := sorry

end arithmetic_prog_triangle_l287_287590


namespace ana_additional_payment_l287_287826

theorem ana_additional_payment (A B L : ℝ) (h₁ : A < B) (h₂ : A < L) : 
  (A + (B + L - 2 * A) / 3 = ((A + B + L) / 3)) :=
by
  sorry

end ana_additional_payment_l287_287826


namespace proof_problem_l287_287108

noncomputable def sequence_a (n : ℕ) : ℕ :=
  if n = 0 then 3 else 3 * n

noncomputable def sequence_b (n : ℕ) : ℕ :=
  3 ^ n

noncomputable def sequence_c (n : ℕ) : ℕ :=
  sequence_b (sequence_a n)

theorem proof_problem :
  sequence_c 2017 = 27 ^ 2017 :=
by sorry

end proof_problem_l287_287108


namespace road_length_l287_287919

theorem road_length (n : ℕ) (d : ℕ) (trees : ℕ) (intervals : ℕ) (L : ℕ) 
  (h1 : n = 10) 
  (h2 : d = 10) 
  (h3 : trees = 10) 
  (h4 : intervals = trees - 1) 
  (h5 : L = intervals * d) : 
  L = 90 :=
by
  sorry

end road_length_l287_287919


namespace min_sum_x1_x2_x3_x4_l287_287319

variables (x1 x2 x3 x4 : ℝ)

theorem min_sum_x1_x2_x3_x4 : 
  (x1 + x2 ≥ 12) → 
  (x1 + x3 ≥ 13) → 
  (x1 + x4 ≥ 14) → 
  (x3 + x4 ≥ 22) → 
  (x2 + x3 ≥ 23) → 
  (x2 + x4 ≥ 24) → 
  (x1 + x2 + x3 + x4 = 37) := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end min_sum_x1_x2_x3_x4_l287_287319


namespace three_lines_l287_287399

def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem three_lines (x y : ℝ) : (diamond x y = diamond y x) ↔ (x = 0 ∨ y = 0 ∨ x = y ∨ x = -y) := 
by sorry

end three_lines_l287_287399


namespace calculate_weight_of_6_moles_HClO2_l287_287271

noncomputable def weight_of_6_moles_HClO2 := 
  let molar_mass_H := 1.01
  let molar_mass_Cl := 35.45
  let molar_mass_O := 16.00
  let molar_mass_HClO2 := molar_mass_H + molar_mass_Cl + 2 * molar_mass_O
  let moles_HClO2 := 6
  moles_HClO2 * molar_mass_HClO2

theorem calculate_weight_of_6_moles_HClO2 : weight_of_6_moles_HClO2 = 410.76 :=
by
  sorry

end calculate_weight_of_6_moles_HClO2_l287_287271


namespace first_year_after_2020_with_digit_sum_4_l287_287131

theorem first_year_after_2020_with_digit_sum_4 :
  ∃ x : ℕ, x > 2020 ∧ (Nat.digits 10 x).sum = 4 ∧ ∀ y : ℕ, y > 2020 ∧ (Nat.digits 10 y).sum = 4 → x ≤ y :=
sorry

end first_year_after_2020_with_digit_sum_4_l287_287131


namespace fraction_pow_zero_l287_287192

theorem fraction_pow_zero
  (a : ℤ) (b : ℤ)
  (h_a : a = -325123789)
  (h_b : b = 59672384757348)
  (h_nonzero_num : a ≠ 0)
  (h_nonzero_denom : b ≠ 0) :
  (a / b : ℚ) ^ 0 = 1 :=
by {
  sorry
}

end fraction_pow_zero_l287_287192


namespace base_four_to_base_ten_of_20314_eq_568_l287_287934

-- Define what it means to convert a base-four number to base-ten
def base_four_to_base_ten (digits : List ℕ) : ℕ :=
  digits.reverse.enum.foldr (λ ⟨index, digit⟩ acc => acc + digit * 4^index) 0

-- Define the specific base-four number 20314_4 as a list of its digits
def num_20314_base_four : List ℕ := [2, 0, 3, 1, 4]

-- Theorem stating that the base-ten equivalent of 20314_4 is 568
theorem base_four_to_base_ten_of_20314_eq_568 : base_four_to_base_ten num_20314_base_four = 568 := sorry

end base_four_to_base_ten_of_20314_eq_568_l287_287934


namespace gcd_98_63_l287_287174

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end gcd_98_63_l287_287174


namespace circle_inequality_l287_287750

-- Given a circle of 100 pairwise distinct numbers a : ℕ → ℝ for 1 ≤ i ≤ 100
variables {a : ℕ → ℝ}
-- Hypothesis 1: distinct numbers
def distinct_numbers (a : ℕ → ℝ) := ∀ i j : ℕ, (1 ≤ i ∧ i ≤ 100) ∧ (1 ≤ j ∧ j ≤ 100) ∧ (i ≠ j) → a i ≠ a j

-- Theorem: Prove that there exist four consecutive numbers such that the sum of the first and the last number is strictly greater than the sum of the two middle numbers
theorem circle_inequality (h_distinct : distinct_numbers a) : 
  ∃ i : ℕ, (1 ≤ i ∧ i ≤ 100) ∧ (a i + a ((i + 3) % 100) > a ((i + 1) % 100) + a ((i + 2) % 100)) :=
sorry

end circle_inequality_l287_287750


namespace pos_sum_of_powers_l287_287857

theorem pos_sum_of_powers (a b c : ℝ) (n : ℕ) (h1 : a * b * c > 0) (h2 : a + b + c > 0) : 
  a^n + b^n + c^n > 0 :=
sorry

end pos_sum_of_powers_l287_287857


namespace remaining_number_is_6218_l287_287462

theorem remaining_number_is_6218 :
  let candidates := { n : ℕ | n ∈ Set.Icc 3 223 ∧ n % 4 = 3 },
      initial_sum := candidates.sum,
      steps := candidates.card - 1
  in initial_sum - 2 * steps = 6218 :=
by
  let candidates := { n : ℕ | n ∈ Set.Icc 3 223 ∧ n % 4 = 3 }
  let initial_sum := candidates.sum
  let steps := candidates.card - 1
  have h_initial_sum : initial_sum = 6328 := sorry
  have h_steps : steps = 55 := sorry
  calc
    initial_sum - 2 * steps
        = 6328 - 2 * 55 : by rw [h_initial_sum, h_steps]
    ... = 6218 : by norm_num

end remaining_number_is_6218_l287_287462


namespace num_of_original_numbers_l287_287011

theorem num_of_original_numbers
    (n : ℕ) 
    (S : ℤ) 
    (incorrect_avg correct_avg : ℤ)
    (incorrect_num correct_num : ℤ)
    (h1 : incorrect_avg = 46)
    (h2 : correct_avg = 51)
    (h3 : incorrect_num = 25)
    (h4 : correct_num = 75)
    (h5 : S + correct_num = correct_avg * n)
    (h6 : S + incorrect_num = incorrect_avg * n) :
  n = 10 := by
  sorry

end num_of_original_numbers_l287_287011


namespace downstream_speed_l287_287216

def Vm : ℝ := 31  -- speed in still water
def Vu : ℝ := 25  -- speed upstream
def Vs := Vm - Vu  -- speed of stream

theorem downstream_speed : Vm + Vs = 37 := 
by
  sorry

end downstream_speed_l287_287216


namespace simplify_polynomial_l287_287617

theorem simplify_polynomial (x : ℝ) : 
  (2 * x^5 - 3 * x^3 + 5 * x^2 - 8 * x + 15) + (3 * x^4 + 2 * x^3 - 4 * x^2 + 3 * x - 7) = 
  2 * x^5 + 3 * x^4 - x^3 + x^2 - 5 * x + 8 :=
by sorry

end simplify_polynomial_l287_287617


namespace possible_values_of_x_l287_287017

-- Definitions representing the initial conditions
def condition1 (x : ℕ) : Prop := 203 % x = 13
def condition2 (x : ℕ) : Prop := 298 % x = 13

-- Main theorem statement
theorem possible_values_of_x (x : ℕ) (h1 : condition1 x) (h2 : condition2 x) : x = 19 ∨ x = 95 := 
by
  sorry

end possible_values_of_x_l287_287017


namespace repeat_decimals_subtraction_l287_287560

-- Define repeating decimal 0.4 repeating as a fraction
def repr_decimal_4 : ℚ := 4 / 9

-- Define repeating decimal 0.6 repeating as a fraction
def repr_decimal_6 : ℚ := 2 / 3

-- Theorem stating the equivalence of subtraction of these repeating decimals
theorem repeat_decimals_subtraction :
  repr_decimal_4 - repr_decimal_6 = -2 / 9 :=
sorry

end repeat_decimals_subtraction_l287_287560


namespace magnitude_of_T_l287_287607

def i : Complex := Complex.I

def T : Complex := (1 + i) ^ 18 - (1 - i) ^ 18

theorem magnitude_of_T : Complex.abs T = 1024 := by
  sorry

end magnitude_of_T_l287_287607


namespace greatest_divisor_sum_of_first_fifteen_terms_l287_287356

theorem greatest_divisor_sum_of_first_fifteen_terms 
  (x c : ℕ) (hx : x > 0) (hc : c > 0):
  ∃ d, d = 15 ∧ d ∣ (15*x + 105*c) :=
by
  existsi 15
  split
  . refl
  . apply Nat.dvd.intro
    existsi (x + 7*c)
    refl
  sorry

end greatest_divisor_sum_of_first_fifteen_terms_l287_287356


namespace sin_cos_expression_l287_287181

noncomputable def sin_45 := Real.sin (Real.pi / 4)
noncomputable def cos_15 := Real.cos (Real.pi / 12)
noncomputable def cos_225 := Real.cos (5 * Real.pi / 4)
noncomputable def sin_15 := Real.sin (Real.pi / 12)

theorem sin_cos_expression :
  sin_45 * cos_15 + cos_225 * sin_15 = 1 / 2 :=
by
  sorry

end sin_cos_expression_l287_287181


namespace combined_cost_l287_287966

variable (bench_cost : ℝ) (table_cost : ℝ)

-- Conditions
axiom bench_cost_def : bench_cost = 250.0
axiom table_cost_def : table_cost = 2 * bench_cost

-- Goal
theorem combined_cost (bench_cost : ℝ) (table_cost : ℝ) 
  (h1 : bench_cost = 250.0) (h2 : table_cost = 2 * bench_cost) : 
  table_cost + bench_cost = 750.0 :=
by
  sorry

end combined_cost_l287_287966


namespace sum_of_digits_div_by_11_in_consecutive_39_l287_287753

-- Define the sum of digits function for natural numbers.
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main theorem statement.
theorem sum_of_digits_div_by_11_in_consecutive_39 :
  ∀ (N : ℕ), ∃ k : ℕ, k < 39 ∧ (sum_of_digits (N + k)) % 11 = 0 :=
by sorry

end sum_of_digits_div_by_11_in_consecutive_39_l287_287753


namespace no_positive_integer_solutions_l287_287116

def f (x : ℤ) : ℤ := x^2 + x

theorem no_positive_integer_solutions 
    (a b : ℤ) (ha : 0 < a) (hb : 0 < b) : 4 * f a ≠ f b := by
  sorry

end no_positive_integer_solutions_l287_287116


namespace probability_shattering_l287_287396

theorem probability_shattering (total_cars : ℕ) (shattered_windshields : ℕ) (p : ℚ) 
  (h_total : total_cars = 20000) 
  (h_shattered: shattered_windshields = 600) 
  (h_p : p = shattered_windshields / total_cars) : 
  p = 0.03 := 
by 
  -- skipped proof
  sorry

end probability_shattering_l287_287396


namespace percentage_of_males_l287_287139

noncomputable def total_employees : ℝ := 1800
noncomputable def males_below_50_years_old : ℝ := 756
noncomputable def percentage_below_50 : ℝ := 0.70

theorem percentage_of_males : (males_below_50_years_old / percentage_below_50 / total_employees) * 100 = 60 :=
by
  sorry

end percentage_of_males_l287_287139


namespace marathon_distance_l287_287734

theorem marathon_distance (marathons : ℕ) (miles_per_marathon : ℕ) (extra_yards_per_marathon : ℕ) (yards_per_mile : ℕ) (total_miles_run : ℕ) (total_yards_run : ℕ) (remaining_yards : ℕ) :
  marathons = 15 →
  miles_per_marathon = 26 →
  extra_yards_per_marathon = 385 →
  yards_per_mile = 1760 →
  total_miles_run = (marathons * miles_per_marathon + extra_yards_per_marathon * marathons / yards_per_mile) →
  total_yards_run = (marathons * (miles_per_marathon * yards_per_mile + extra_yards_per_marathon)) →
  remaining_yards = total_yards_run - (total_miles_run * yards_per_mile) →
  0 ≤ remaining_yards ∧ remaining_yards < yards_per_mile →
  remaining_yards = 1500 :=
by
  intros
  sorry

end marathon_distance_l287_287734


namespace james_and_david_probability_l287_287732

noncomputable def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem james_and_david_probability :
  let total_workers := 22
  let chosen_workers := 4
  let j_and_d_chosen := 2
  (choose 20 2) / (choose 22 4) = (2 / 231) :=
by
  sorry

end james_and_david_probability_l287_287732


namespace smallest_prime_after_six_nonprimes_l287_287795

open Nat

/-- 
  Proposition:
    97 is the smallest prime number that occurs after a sequence 
    of six consecutive positive integers all of which are nonprime.
-/
theorem smallest_prime_after_six_nonprimes : ∃ (n : ℕ), (n > 0 ∧ Prime (n + 97)) ∧ 
  (∀ k : ℕ, k < n → (Nonprime k) ∧ Nonprime (k+1) ∧ Nonprime (k+2) ∧ Nonprime (k+3) ∧ Nonprime (k+4) ∧ Nonprime (k+5)) ∧ 
  (Prime (n + 1) → n + 1 = 97) := 
by 
  sorry

end smallest_prime_after_six_nonprimes_l287_287795


namespace find_lost_bowls_l287_287187

def bowls_problem (L : ℕ) : Prop :=
  let total_bowls := 638
  let broken_bowls := 15
  let payment := 1825
  let fee := 100
  let safe_bowl_payment := 3
  let lost_broken_bowl_cost := 4
  100 + 3 * (total_bowls - L - broken_bowls) - 4 * (L + broken_bowls) = payment

theorem find_lost_bowls : ∃ L : ℕ, bowls_problem L ∧ L = 26 :=
  by
  sorry

end find_lost_bowls_l287_287187


namespace incorrect_directions_of_opening_l287_287375

-- Define the functions
def f (x : ℝ) : ℝ := 2 * (x - 3)^2
def g (x : ℝ) : ℝ := -2 * (x - 3)^2

-- The theorem (statement) to prove
theorem incorrect_directions_of_opening :
  ¬(∀ x, (f x > 0 ∧ g x > 0) ∨ (f x < 0 ∧ g x < 0)) :=
sorry

end incorrect_directions_of_opening_l287_287375


namespace total_course_selection_schemes_l287_287954

theorem total_course_selection_schemes (
  pe_courses art_courses : Finset ℕ
) : 
  pe_courses.card = 4 →
  art_courses.card = 4 →
  let total_schemes := 
    (pe_courses.card.choose 1 * art_courses.card.choose 1) + 
    (pe_courses.card.choose 2 * art_courses.card.choose 1) + 
    (pe_courses.card.choose 1 * art_courses.card.choose 2) in
  total_schemes = 64 := 
by
  intros h1 h2
  let case1 := pe_courses.card.choose 1 * art_courses.card.choose 1 -- 4 * 4
  let case2_1 := pe_courses.card.choose 2 * art_courses.card.choose 1 -- 6 * 4
  let case2_2 := pe_courses.card.choose 1 * art_courses.card.choose 2 -- 4 * 6
  let case2 := case2_1 + case2_2 -- 24 + 24
  let total_schemes := case1 + case2 -- 16 + 48
  have hcard := by
    simp [Nat.choose]
    exact h1
  have hcase1 : case1 = 16 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2_1 : case2_1 = 24 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2_2 : case2_2 = 24 := by {
    rw [Nat.choose, Nat.choose, hcard, hcard]
    norm_num
  }
  have hcase2 : case2 = 48 := by {
    rw [hcase2_1, hcase2_2]
    norm_num
  }
  have htotal_schemes : total_schemes = 64 := by {
    rw [hcase1, hcase2]
    norm_num
  }
  exact htotal_schemes

end total_course_selection_schemes_l287_287954


namespace non_congruent_triangles_with_perimeter_11_l287_287870

theorem non_congruent_triangles_with_perimeter_11 :
  ∃ (a b c : ℕ), a + b + c = 11 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧
  (∀ d e f : ℕ, d + e + f = 11 ∧ d ≤ e ∧ e ≤ f ∧ d + e > f → 
  (d = a ∧ e = b ∧ f = c) ∨ (d = b ∧ e = a ∧ f = c) ∨ (d = a ∧ e = c ∧ f = b)) → 
  3 := 
sorry

end non_congruent_triangles_with_perimeter_11_l287_287870


namespace intersection_points_count_l287_287437

variables {R : Type*} [LinearOrderedField R]

def line1 (x y : R) : Prop := 3 * y - 2 * x = 1
def line2 (x y : R) : Prop := x + 2 * y = 2
def line3 (x y : R) : Prop := 4 * x - 6 * y = 5

theorem intersection_points_count : 
  ∃ p1 p2 : R × R, 
   (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧ 
   (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧ 
   p1 ≠ p2 ∧ 
   (∀ p : R × R, (line1 p.1 p.2 ∧ line3 p.1 p.2) → False) := 
sorry

end intersection_points_count_l287_287437


namespace solve_tangents_equation_l287_287717

open Real

def is_deg (x : ℝ) : Prop := ∃ k : ℤ, x = 30 + 180 * k

theorem solve_tangents_equation (x : ℝ) (h : tan (x * π / 180) * tan (20 * π / 180) + tan (20 * π / 180) * tan (40 * π / 180) + tan (40 * π / 180) * tan (x * π / 180) = 1) :
  is_deg x :=
sorry

end solve_tangents_equation_l287_287717


namespace find_t_l287_287254

open Complex Real

theorem find_t (a b : ℂ) (t : ℝ) (h₁ : abs a = 3) (h₂ : abs b = 5) (h₃ : a * b = t - 3 * I) :
  t = 6 * Real.sqrt 6 := by
  sorry

end find_t_l287_287254


namespace semi_circle_radius_l287_287926

theorem semi_circle_radius (P : ℝ) (r : ℝ) (π : ℝ) (h_perimeter : P = 113) (h_pi : π = Real.pi) :
  r = P / (π + 2) :=
sorry

end semi_circle_radius_l287_287926


namespace compute_exp_l287_287068

theorem compute_exp : 3 * 3^4 + 9^30 / 9^28 = 324 := 
by sorry

end compute_exp_l287_287068


namespace triangle_inequalities_l287_287000

-- Definitions of the variables
variables {ABC : Triangle} {r : ℝ} {R : ℝ} {ρ_a ρ_b ρ_c : ℝ} {P_a P_b P_c : ℝ}

-- Problem statement based on given conditions and proof requirement
theorem triangle_inequalities (ABC : Triangle) (r : ℝ) (R : ℝ) (ρ_a ρ_b ρ_c : ℝ) (P_a P_b P_c : ℝ) :
  (3/2) * r ≤ ρ_a + ρ_b + ρ_c ∧ ρ_a + ρ_b + ρ_c ≤ (3/4) * R ∧ 4 * r ≤ P_a + P_b + P_c ∧ P_a + P_b + P_c ≤ 2 * R :=
  sorry

end triangle_inequalities_l287_287000


namespace fourth_equation_general_expression_l287_287304

theorem fourth_equation :
  (10 : ℕ)^2 - 4 * (4 : ℕ)^2 = 36 := 
sorry

theorem general_expression (n : ℕ) (hn : n > 0) :
  (2 * n + 2)^2 - 4 * n^2 = 8 * n + 4 :=
sorry

end fourth_equation_general_expression_l287_287304


namespace smallest_value_of_y_l287_287940

theorem smallest_value_of_y : 
  (∃ y : ℝ, 6 * y^2 - 41 * y + 55 = 0 ∧ ∀ z : ℝ, 6 * z^2 - 41 * z + 55 = 0 → y ≤ z) →
  ∃ y : ℝ, y = 2.5 :=
by sorry

end smallest_value_of_y_l287_287940


namespace solution_l287_287023

noncomputable def triangle_perimeter (AB BC AC : ℕ) (lA lB lC : ℕ) : ℕ :=
  -- This represents the proof problem using the given conditions
  if (AB = 130) ∧ (BC = 240) ∧ (AC = 190)
     ∧ (lA = 65) ∧ (lB = 50) ∧ (lC = 20)
  then
    130  -- The correct answer
  else
    0    -- If the conditions are not met, return 0 

theorem solution :
  triangle_perimeter 130 240 190 65 50 20 = 130 :=
by
  -- This theorem states that with the given conditions, the perimeter of the triangle is 130
  sorry

end solution_l287_287023


namespace philip_oranges_count_l287_287409

def betty_oranges : ℕ := 15
def bill_oranges : ℕ := 12
def betty_bill_oranges := betty_oranges + bill_oranges
def frank_oranges := 3 * betty_bill_oranges
def seeds_planted := frank_oranges * 2
def orange_trees := seeds_planted
def oranges_per_tree : ℕ := 5
def oranges_for_philip := orange_trees * oranges_per_tree

theorem philip_oranges_count : oranges_for_philip = 810 := by sorry

end philip_oranges_count_l287_287409


namespace kishore_expenses_l287_287975

noncomputable def total_salary (savings : ℕ) (percent : ℝ) : ℝ :=
savings / percent

noncomputable def total_expenses (rent milk groceries education petrol : ℕ) : ℕ :=
  rent + milk + groceries + education + petrol

noncomputable def miscellaneous_expenses (total_salary : ℝ) (total_expenses : ℕ) (savings : ℕ) : ℝ :=
  total_salary - (total_expenses + savings)

theorem kishore_expenses :
  total_salary 2160 0.1 - (total_expenses 5000 1500 4500 2500 2000 + 2160) = 3940 := by
  sorry

end kishore_expenses_l287_287975


namespace tips_earned_l287_287203

theorem tips_earned
  (total_customers : ℕ)
  (no_tip_customers : ℕ)
  (tip_amount : ℕ)
  (tip_customers := total_customers - no_tip_customers)
  (total_tips := tip_customers * tip_amount)
  (h1 : total_customers = 9)
  (h2 : no_tip_customers = 5)
  (h3 : tip_amount = 8) :
  total_tips = 32 := by
  -- Proof goes here
  sorry

end tips_earned_l287_287203


namespace binary_addition_is_correct_l287_287534

-- Definitions for the binary numbers
def bin1 := "10101"
def bin2 := "11"
def bin3 := "1010"
def bin4 := "11100"
def bin5 := "1101"

-- Function to convert binary string to nat (using built-in functionality)
def binStringToNat (s : String) : Nat :=
  String.foldl (fun n c => 2 * n + if c = '1' then 1 else 0) 0 s

-- Binary numbers converted to nat
def n1 := binStringToNat bin1
def n2 := binStringToNat bin2
def n3 := binStringToNat bin3
def n4 := binStringToNat bin4
def n5 := binStringToNat bin5

-- The expected result in nat
def expectedSum := binStringToNat "11101101"

-- Proof statement
theorem binary_addition_is_correct : n1 + n2 + n3 + n4 + n5 = expectedSum :=
  sorry

end binary_addition_is_correct_l287_287534


namespace bruce_money_left_to_buy_more_clothes_l287_287681

def calculate_remaining_money 
  (amount_given : ℝ) 
  (shirt_price : ℝ) (num_shirts : ℕ)
  (pants_price : ℝ)
  (sock_price : ℝ) (num_socks : ℕ)
  (belt_original_price : ℝ) (belt_discount : ℝ)
  (total_discount : ℝ) : ℝ := 
let shirts_cost := shirt_price * num_shirts
let socks_cost := sock_price * num_socks
let belt_price := belt_original_price * (1 - belt_discount)
let total_cost := shirts_cost + pants_price + socks_cost + belt_price
let discount_cost := total_cost * total_discount
let final_cost := total_cost - discount_cost
amount_given - final_cost

theorem bruce_money_left_to_buy_more_clothes 
  : calculate_remaining_money 71 5 5 26 3 2 12 0.25 0.10 = 11.60 := 
by
  sorry

end bruce_money_left_to_buy_more_clothes_l287_287681


namespace polynomial_coefficients_l287_287086

theorem polynomial_coefficients
  (x : ℝ)
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ)
  (h : (x-3)^8 = a_0 + a_1 * (x-2) + a_2 * (x-2)^2 + a_3 * (x-2)^3 + 
                a_4 * (x-2)^4 + a_5 * (x-2)^5 + a_6 * (x-2)^6 + 
                a_7 * (x-2)^7 + a_8 * (x-2)^8) :
  (a_0 = 1) ∧ 
  (a_1 / 2 + a_2 / 2^2 + a_3 / 2^3 + a_4 / 2^4 + a_5 / 2^5 + 
   a_6 / 2^6 + a_7 / 2^7 + a_8 / 2^8 = -255 / 256) ∧ 
  (a_0 + a_2 + a_4 + a_6 + a_8 = 128) :=
by sorry

end polynomial_coefficients_l287_287086


namespace find_year_after_2020_l287_287132

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_year_after_2020 :
  ∃ y : ℕ, 2020 < y ∧ sum_of_digits y = 4 ∧ (∀ z : ℕ, 2020 < z ∧ sum_of_digits z = 4 → y ≤ z) := 
begin
  sorry,
end

end find_year_after_2020_l287_287132


namespace selection_methods_l287_287312

-- Conditions
def volunteers : ℕ := 5
def friday_slots : ℕ := 1
def saturday_slots : ℕ := 2
def sunday_slots : ℕ := 1

-- Function to calculate combinatorial n choose k
def choose (n k : ℕ) : ℕ :=
(n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Function to calculate permutations of n P k
def perm (n k : ℕ) : ℕ :=
(n.factorial) / ((n - k).factorial)

-- The target proposition
theorem selection_methods : choose volunteers saturday_slots * perm (volunteers - saturday_slots) (friday_slots + sunday_slots) = 60 :=
by
  -- assumption here leads to the property required, usually this would be more detailed computation.
  sorry

end selection_methods_l287_287312


namespace sum_is_220_l287_287491

def second_number := 60
def first_number := 2 * second_number
def third_number := first_number / 3
def sum_of_numbers := first_number + second_number + third_number

theorem sum_is_220 : sum_of_numbers = 220 :=
by
  sorry

end sum_is_220_l287_287491


namespace area_of_hall_l287_287768

-- Define the conditions
def length := 25
def breadth := length - 5

-- Define the area calculation
def area := length * breadth

-- The statement to prove
theorem area_of_hall : area = 500 :=
by
  sorry

end area_of_hall_l287_287768


namespace probability_positive_ball_drawn_is_half_l287_287502

-- Definition of the problem elements
def balls : List Int := [-1, 0, 2, 3]

-- Definition for the event of drawing a positive number
def is_positive (x : Int) : Bool := x > 0

-- The proof statement
theorem probability_positive_ball_drawn_is_half : 
  (List.filter is_positive balls).length / balls.length = 1 / 2 :=
by
  sorry

end probability_positive_ball_drawn_is_half_l287_287502


namespace total_sacks_needed_l287_287678

def first_bakery_needs : ℕ := 2
def second_bakery_needs : ℕ := 4
def third_bakery_needs : ℕ := 12
def weeks : ℕ := 4

theorem total_sacks_needed :
  first_bakery_needs * weeks + second_bakery_needs * weeks + third_bakery_needs * weeks = 72 :=
by
  sorry

end total_sacks_needed_l287_287678


namespace tan_20_add_4sin_20_eq_sqrt3_l287_287764

theorem tan_20_add_4sin_20_eq_sqrt3 : Real.tan (20 * Real.pi / 180) + 4 * Real.sin (20 * Real.pi / 180) = Real.sqrt 3 := 
by
  sorry

end tan_20_add_4sin_20_eq_sqrt3_l287_287764


namespace factorization_count_l287_287893

noncomputable def count_factors (n : ℕ) (a b c : ℕ) : ℕ :=
if 2 ^ a * 2 ^ b * 2 ^ c = n ∧ a + b + c = 10 ∧ a ≥ b ∧ b ≥ c then 1 else 0

noncomputable def total_factorizations : ℕ :=
Finset.sum (Finset.range 11) (fun c => 
  Finset.sum (Finset.Icc c 10) (fun b => 
    Finset.sum (Finset.Icc b 10) (fun a =>
      count_factors 1024 a b c)))

theorem factorization_count : total_factorizations = 14 :=
sorry

end factorization_count_l287_287893


namespace greatest_divisor_of_sum_of_arithmetic_sequence_l287_287364

theorem greatest_divisor_of_sum_of_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ k : ℕ, (sum (λ n, x + n * c) (range 15)) = 15 * k :=
by sorry

end greatest_divisor_of_sum_of_arithmetic_sequence_l287_287364


namespace derivative_of_y_l287_287948

noncomputable def y (x : ℝ) : ℝ := (1 - Real.cos (2 * x)) / (1 + Real.cos (2 * x))
noncomputable def dy_dx (x : ℝ) : ℝ := deriv y x

-- Theorem statement and proof placeholder
theorem derivative_of_y (x : ℝ) : dy_dx x = (4 * Real.sin (2 * x)) / ((1 + Real.cos (2 * x))^2) :=
sorry

end derivative_of_y_l287_287948


namespace average_brown_mms_l287_287163

def brown_smiley_counts : List Nat := [9, 12, 8, 8, 3]
def brown_star_counts : List Nat := [7, 14, 11, 6, 10]

def average (lst : List Nat) : Float :=
  (lst.foldl (· + ·) 0).toFloat / lst.length.toFloat
  
theorem average_brown_mms :
  average brown_smiley_counts = 8 ∧
  average brown_star_counts = 9.6 :=
by 
  sorry

end average_brown_mms_l287_287163


namespace greatest_divisor_of_sum_of_arith_seq_l287_287354

theorem greatest_divisor_of_sum_of_arith_seq (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → d ∣ (15 * (x + 7 * c))) ∧
    (∀ k : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → k ∣ (15 * (x + 7 * c))) → k ≤ d) ∧ 
    d = 15 :=
sorry

end greatest_divisor_of_sum_of_arith_seq_l287_287354


namespace equivalent_single_discount_l287_287673

theorem equivalent_single_discount (P : ℝ) (hP : 0 < P) : 
    let first_discount : ℝ := 0.15
    let second_discount : ℝ := 0.25
    let single_discount : ℝ := 0.3625
    (1 - first_discount) * (1 - second_discount) * P = (1 - single_discount) * P := by
    sorry

end equivalent_single_discount_l287_287673


namespace primes_div_conditions_unique_l287_287561

theorem primes_div_conditions_unique (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (p ∣ q + 6) ∧ (q ∣ p + 7) → (p = 19 ∧ q = 13) :=
sorry

end primes_div_conditions_unique_l287_287561


namespace complex_square_eq_l287_287267

theorem complex_square_eq (i : ℂ) (hi : i * i = -1) : (1 + i)^2 = 2 * i := 
by {
  -- marking the end of existing code for clarity
  sorry
}

end complex_square_eq_l287_287267


namespace total_tablets_l287_287382

-- Variables for the numbers of Lenovo, Samsung, and Huawei tablets
variables (n x y : ℕ)

-- Conditions based on problem statement
def condition1 : Prop := 2 * x + 6 + y < n / 3

def condition2 : Prop := (n - 2 * x - y - 6 = 3 * y)

def condition3 : Prop := (n - 6 * x - y - 6 = 59)

-- The statement to prove that the total number of tablets is 94
theorem total_tablets (h1 : condition1 n x y) (h2 : condition2 n x y) (h3 : condition3 n x y) : n = 94 :=
by
  sorry

end total_tablets_l287_287382


namespace value_of_expression_l287_287827

theorem value_of_expression 
  (x1 x2 x3 x4 x5 x6 x7 : ℝ)
  (h1 : x1 + 9*x2 + 25*x3 + 49*x4 + 81*x5 + 121*x6 + 169*x7 = 2)
  (h2 : 9*x1 + 25*x2 + 49*x3 + 81*x4 + 121*x5 + 169*x6 + 225*x7 = 24)
  (h3 : 25*x1 + 49*x2 + 81*x3 + 121*x4 + 169*x5 + 225*x6 + 289*x7 = 246) : 
  49*x1 + 81*x2 + 121*x3 + 169*x4 + 225*x5 + 289*x6 + 361*x7 = 668 := 
sorry

end value_of_expression_l287_287827


namespace spiral_wire_length_l287_287525

noncomputable def wire_length (turns : ℕ) (height : ℝ) (circumference : ℝ) : ℝ :=
  Real.sqrt (height^2 + (turns * circumference)^2)

theorem spiral_wire_length
  (turns : ℕ) (height : ℝ) (circumference : ℝ)
  (turns_eq : turns = 10)
  (height_eq : height = 9)
  (circumference_eq : circumference = 4) :
  wire_length turns height circumference = 41 := 
by
  rw [turns_eq, height_eq, circumference_eq]
  simp [wire_length]
  norm_num
  rw [Real.sqrt_eq_rpow]
  norm_num
  sorry

end spiral_wire_length_l287_287525


namespace sum_of_values_l287_287301

noncomputable def f (x : ℝ) : ℝ :=
if x < 3 then 5 * x + 20 else 3 * x - 21

theorem sum_of_values (h₁ : ∃ x, x < 3 ∧ f x = 4) (h₂ : ∃ x, x ≥ 3 ∧ f x = 4) :
  ∃a b : ℝ, a = -16 / 5 ∧ b = 25 / 3 ∧ (a + b = 77 / 15) :=
by {
  sorry
}

end sum_of_values_l287_287301


namespace square_non_negative_is_universal_l287_287840

/-- The square of any real number is non-negative, which is a universal proposition. -/
theorem square_non_negative_is_universal : 
  ∀ x : ℝ, x^2 ≥ 0 :=
by
  sorry

end square_non_negative_is_universal_l287_287840


namespace problem_part1_problem_part2_l287_287096

noncomputable def arithmetic_sequence (a : ℕ → ℕ) :=
  ∀ n : ℕ, a (n + 1) = a n + 2

theorem problem_part1 (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : a 1 = 2) (h2 : S 2 = a 3) (h3 : arithmetic_sequence a) :
  a 2 = 4 := 
sorry

theorem problem_part2 (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : a 1 = 2) (h2 : S 2 = a 3) (h3 : arithmetic_sequence a) 
  (h4 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2) :
  S 10 = 110 :=
sorry

end problem_part1_problem_part2_l287_287096


namespace average_monthly_balance_l287_287671

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

end average_monthly_balance_l287_287671


namespace greatest_common_divisor_sum_arithmetic_sequence_l287_287332

theorem greatest_common_divisor_sum_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, d = 15 ∧ ∀ (n : ℕ), n = 15 → ∀ k : ℕ, k = 15 ∧ 15 ∣ (15 * (x + 7 * c)) :=
by
  sorry

end greatest_common_divisor_sum_arithmetic_sequence_l287_287332


namespace product_modulo_7_l287_287255

theorem product_modulo_7 : (1729 * 1865 * 1912 * 2023) % 7 = 6 :=
by
  sorry

end product_modulo_7_l287_287255


namespace option_C_is_quadratic_l287_287027

-- Define what it means for an equation to be quadratic
def is_quadratic (p : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ (x : ℝ), p x ↔ a*x^2 + b*x + c = 0

-- Define the equation in option C
def option_C (x : ℝ) : Prop := (x - 1) * (x - 2) = 0

-- The theorem we need to prove
theorem option_C_is_quadratic : is_quadratic option_C :=
  sorry

end option_C_is_quadratic_l287_287027


namespace smallest_prime_after_six_nonprimes_l287_287793

open Nat

theorem smallest_prime_after_six_nonprimes : 
  ∃ p : ℕ, prime p ∧ p > 0 ∧
  (∃ n : ℕ, ¬prime n ∧ ¬prime (n+1) ∧ ¬prime (n+2) ∧ ¬prime (n+3) ∧ ¬prime (n+4) ∧ ¬prime (n+5) ∧ prime (n+6) ∧ n+6 = p) ∧ 
  p = 89 :=
sorry

end smallest_prime_after_six_nonprimes_l287_287793


namespace positive_difference_l287_287307

/-- Pauline deposits 10,000 dollars into an account with 4% compound interest annually. -/
def Pauline_initial_deposit : ℝ := 10000
def Pauline_interest_rate : ℝ := 0.04
def Pauline_years : ℕ := 12

/-- Quinn deposits 10,000 dollars into an account with 6% simple interest annually. -/
def Quinn_initial_deposit : ℝ := 10000
def Quinn_interest_rate : ℝ := 0.06
def Quinn_years : ℕ := 12

/-- Pauline's balance after 12 years -/
def Pauline_balance : ℝ := Pauline_initial_deposit * (1 + Pauline_interest_rate) ^ Pauline_years

/-- Quinn's balance after 12 years -/
def Quinn_balance : ℝ := Quinn_initial_deposit * (1 + Quinn_interest_rate * Quinn_years)

/-- The positive difference between Pauline's and Quinn's balances after 12 years is $1189 -/
theorem positive_difference :
  |Quinn_balance - Pauline_balance| = 1189 := 
sorry

end positive_difference_l287_287307


namespace greatest_common_divisor_sum_arithmetic_sequence_l287_287333

theorem greatest_common_divisor_sum_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, d = 15 ∧ ∀ (n : ℕ), n = 15 → ∀ k : ℕ, k = 15 ∧ 15 ∣ (15 * (x + 7 * c)) :=
by
  sorry

end greatest_common_divisor_sum_arithmetic_sequence_l287_287333


namespace a_n_general_formula_b_n_minus_a_n_geometric_b_n_sum_minimum_value_l287_287714

section Sequences

-- Definitions of sequences {a_n} and {b_n}
def a_seq (n : ℕ) : ℝ :=
if h : n > 0 then 
  (1 / 2) * n + 1 / 4 
else 3 / 4

def b_seq (n : ℕ) : ℝ :=
nat.rec_on n (-37 / 4) (λ n b, (b + (n + 2) / 3))

-- Prove the general formula for {a_n}
theorem a_n_general_formula (n : ℕ) (h : n ≥ 1) : 
  (a_seq n = (1 / 2) * n + 1 / 4) :=
by
  sorry

-- Prove that {b_n - a_n} is a geometric sequence
theorem b_n_minus_a_n_geometric (n : ℕ) (h1 : n ≥ 2) :
  ∃ r : ℝ, ∀ n, n ≥ 2 → b_seq n - a_seq n = r * (b_seq (n-1) - a_seq (n-1)) :=
by
  sorry

-- Prove the minimum value of the sum of the first n terms of {b_n}
theorem b_n_sum_minimum_value (n : ℕ) (h1 : n ≥ 2) : 
  ∃ k : ℝ, k = ((b_seq 1) + (b_seq 2) + ∑ i in range (3), b_seq i) ∧ k = -34 / 3 :=
by
  sorry

end Sequences

end a_n_general_formula_b_n_minus_a_n_geometric_b_n_sum_minimum_value_l287_287714


namespace circumcenter_eq_orthocenter_l287_287633

theorem circumcenter_eq_orthocenter {A B C A' B' C' O O'} (h1 : radius_circumcircle_ABC = radius_circle_tangent_A'B'_C' A B C A' B' C' O O') :
  is_center_circumcircle O A B C ∧ is_orthocenter O A' B' C' :=
by
  sorry

end circumcenter_eq_orthocenter_l287_287633


namespace regular_tetrahedron_surface_area_l287_287015

theorem regular_tetrahedron_surface_area {h : ℝ} (h_pos : h > 0) :
  ∃ (S : ℝ), S = (3 * h^2 * Real.sqrt 3) / 2 :=
sorry

end regular_tetrahedron_surface_area_l287_287015


namespace find_cost_of_crackers_l287_287001

-- Definitions based on the given conditions
def cost_hamburger_meat : ℝ := 5.00
def cost_per_bag_vegetables : ℝ := 2.00
def number_of_bags_vegetables : ℕ := 4
def cost_cheese : ℝ := 3.50
def discount_rate : ℝ := 0.10
def total_after_discount : ℝ := 18

-- Definition of the box of crackers, which we aim to prove
def cost_crackers : ℝ := 3.50

-- The Lean statement for the proof
theorem find_cost_of_crackers
  (C : ℝ)
  (h : C = cost_crackers)
  (H : 0.9 * (cost_hamburger_meat + cost_per_bag_vegetables * number_of_bags_vegetables + cost_cheese + C) = total_after_discount) :
  C = 3.50 :=
  sorry

end find_cost_of_crackers_l287_287001


namespace polynomial_root_cubic_sum_l287_287610

theorem polynomial_root_cubic_sum
  (a b c : ℝ)
  (h : ∀ x : ℝ, (Polynomial.eval x (3 * Polynomial.X^3 + 5 * Polynomial.X^2 - 150 * Polynomial.X + 7) = 0)
    → x = a ∨ x = b ∨ x = c) :
  (a + b + 2)^3 + (b + c + 2)^3 + (c + a + 2)^3 = 303 :=
  sorry

end polynomial_root_cubic_sum_l287_287610


namespace smallest_odd_number_with_three_different_prime_factors_l287_287025

theorem smallest_odd_number_with_three_different_prime_factors :
  ∃ n, Nat.Odd n ∧ (∃ p1 p2 p3, Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ n = p1 * p2 * p3) ∧ (∀ m, Nat.Odd m ∧ (∃ q1 q2 q3, Nat.Prime q1 ∧ Nat.Prime q2 ∧ Nat.Prime q3 ∧ q1 ≠ q2 ∧ q2 ≠ q3 ∧ q1 ≠ q3 ∧ m = q1 * q2 * q3) → n ≤ m) :=
  ∃ (n = 105), Nat.Odd n ∧ (∃ p1 p2 p3, Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ n = p1 * p2 * p3) ∧ (∀ m, Nat.Odd m ∧ (∃ q1 q2 q3, Nat.Prime q1 ∧ Nat.Prime q2 ∧ Nat.Prime q3 ∧ q1 ≠ q2 ∧ q2 ≠ q3 ∧ q1 ≠ q3 ∧ m = q1 * q2 * q3) → n ≤ m) :=
sorry

end smallest_odd_number_with_three_different_prime_factors_l287_287025


namespace find_k_l287_287878

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x ^ 2 + (k - 1) * x + 3

theorem find_k (k : ℝ) (h : ∀ x, f k x = f k (-x)) : k = 1 :=
by
  sorry

end find_k_l287_287878


namespace max_apartment_size_is_600_l287_287057

-- Define the cost per square foot and Max's budget
def cost_per_square_foot : ℝ := 1.2
def max_budget : ℝ := 720

-- Define the largest apartment size that Max should consider
def largest_apartment_size (s : ℝ) : Prop :=
  cost_per_square_foot * s = max_budget

-- State the theorem that we need to prove
theorem max_apartment_size_is_600 : largest_apartment_size 600 :=
  sorry

end max_apartment_size_is_600_l287_287057


namespace count_perfect_squares_diff_two_consecutive_squares_l287_287692

theorem count_perfect_squares_diff_two_consecutive_squares:
  (∃ n : ℕ, n = 71 ∧ 
            ∀ a : ℕ, (a < 20000 → 
            (∃ b : ℕ, a^2 = (b+1)^2 - b^2))) :=
sorry

end count_perfect_squares_diff_two_consecutive_squares_l287_287692


namespace probability_rain_both_and_neither_l287_287632

variables (P_Monday P_Tuesday : ℝ)

def independent_events (P_A P_B : ℝ) := P_A * P_B

theorem probability_rain_both_and_neither
  (h_Monday : P_Monday = 0.4)
  (h_Tuesday : P_Tuesday = 0.3)
  (h_independent : independent_events P_Monday P_Tuesday = P_Monday * P_Tuesday):
  (P_both_days : independent_events P_Monday P_Tuesday = 0.12)
  ∧ (P_neither_days : independent_events (1 - P_Monday) (1 - P_Tuesday) = 0.42) :=
by {
  have P_both_days : independent_events P_Monday P_Tuesday = (0.4 * 0.3) := by rw [h_Monday, h_Tuesday];
  have h_both : P_both_days = 0.12 := by norm_num,
  have P_neither_days : independent_events (1 - P_Monday) (1 - P_Tuesday) = (0.6 * 0.7) := by
    { rw [h_Monday, h_Tuesday], norm_num },
  have h_neither : P_neither_days = 0.42 := by norm_num,
  exact ⟨h_both, h_neither⟩,
}

end probability_rain_both_and_neither_l287_287632


namespace ethan_coconut_oil_per_candle_l287_287559

noncomputable def ounces_of_coconut_oil_per_candle (candles: ℕ) (total_weight: ℝ) (beeswax_per_candle: ℝ) : ℝ :=
(total_weight - candles * beeswax_per_candle) / candles

theorem ethan_coconut_oil_per_candle :
  ounces_of_coconut_oil_per_candle 7 63 8 = 1 :=
by
  sorry

end ethan_coconut_oil_per_candle_l287_287559


namespace least_positive_integer_l287_287648

theorem least_positive_integer (n : ℕ) (h₁ : n % 3 = 0) (h₂ : n % 4 = 1) (h₃ : n % 5 = 2) : n = 57 :=
by
  -- sorry to skip the proof
  sorry

end least_positive_integer_l287_287648


namespace circle_equation_l287_287315

theorem circle_equation (x y : ℝ) :
    (x - 1) ^ 2 + (y - 1) ^ 2 = 1 ↔ (∃ (C : ℝ × ℝ), C = (1, 1) ∧ ∃ (r : ℝ), r = 1 ∧ (x - C.1) ^ 2 + (y - C.2) ^ 2 = r ^ 2) :=
by
  sorry

end circle_equation_l287_287315


namespace problem_1_problem_2_problem_3_problem_4_l287_287684

theorem problem_1 : (1 * -2.48) + 4.33 + (-7.52) + (-4.33) = -10 := by
  sorry

theorem problem_2 : 2 * (23 / 6 : ℚ) + - (36 / 7 : ℚ) + - (13 / 6 : ℚ) + - (230 / 7 : ℚ) = -(36 + 1 / 3 : ℚ) := by
  sorry

theorem problem_3 : (4 / 5 : ℚ) - (5 / 6 : ℚ) - (3 / 5 : ℚ) + (1 / 6 : ℚ) = - (7 / 15 : ℚ) := by
  sorry

theorem problem_4 : (-1 ^ 4 : ℚ) - (1 / 6) * (2 - (-3) ^ 2) = 1 / 6 := by
  sorry

end problem_1_problem_2_problem_3_problem_4_l287_287684


namespace expression_value_l287_287608

-- Define the variables and the main statement
variable (w x y z : ℕ)

theorem expression_value :
  2^w * 3^x * 5^y * 11^z = 825 → w + 2 * x + 3 * y + 4 * z = 12 :=
by
  sorry -- Proof omitted

end expression_value_l287_287608


namespace count_valid_A_l287_287426

theorem count_valid_A : 
  ∃! (count : ℕ), count = 4 ∧ ∀ A : ℕ, (1 ≤ A ∧ A ≤ 9) → 
  (∃ x1 x2 : ℕ, x1 + x2 = 2 * A + 1 ∧ x1 * x2 = 2 * A ∧ x1 > 0 ∧ x2 > 0) → A = 1 ∨ A = 2 ∨ A = 3 ∨ A = 4 :=
sorry

end count_valid_A_l287_287426


namespace pyramid_height_is_6_l287_287406

-- Define the conditions for the problem
def square_side_length : ℝ := 18
def pyramid_base_side_length (s : ℝ) : Prop := s * s = (square_side_length / 2) * (square_side_length / 2)
def pyramid_slant_height (s l : ℝ) : Prop := 2 * s * l = square_side_length * square_side_length

-- State the main theorem
theorem pyramid_height_is_6 (s l h : ℝ) (hs : pyramid_base_side_length s) (hl : pyramid_slant_height s l) : h = 6 := 
sorry

end pyramid_height_is_6_l287_287406


namespace proportional_function_property_l287_287309

theorem proportional_function_property :
  (∀ x, ∃ y, y = -3 * x ∧
  (x = 0 → y = 0) ∧
  (x > 0 → y < 0) ∧
  (x < 0 → y > 0) ∧
  (x = 1 → y = -3) ∧
  (∀ x, y = -3 * x → (x > 0 ∧ y < 0 ∨ x < 0 ∧ y > 0))) :=
by
  sorry

end proportional_function_property_l287_287309


namespace num_rooms_with_2_windows_l287_287079

theorem num_rooms_with_2_windows:
  ∃ (num_rooms_with_2_windows: ℕ),
  (∀ (num_rooms_with_4_windows num_rooms_with_3_windows: ℕ), 
    num_rooms_with_4_windows = 5 ∧ 
    num_rooms_with_3_windows = 8 ∧
    4 * num_rooms_with_4_windows + 3 * num_rooms_with_3_windows + 2 * num_rooms_with_2_windows = 122) → 
    num_rooms_with_2_windows = 39 :=
by
  sorry

end num_rooms_with_2_windows_l287_287079


namespace ratio_grass_area_weeded_l287_287154

/-- Lucille earns six cents for every weed she pulls. -/
def earnings_per_weed : ℕ := 6

/-- There are eleven weeds in the flower bed. -/
def weeds_flower_bed : ℕ := 11

/-- There are fourteen weeds in the vegetable patch. -/
def weeds_vegetable_patch : ℕ := 14

/-- There are thirty-two weeds in the grass around the fruit trees. -/
def weeds_grass_total : ℕ := 32

/-- Lucille bought a soda for 99 cents on her break. -/
def soda_cost : ℕ := 99

/-- Lucille has 147 cents left after the break. -/
def cents_left : ℕ := 147

/-- Statement to prove: The ratio of the grass area Lucille weeded to the total grass area around the fruit trees is 1:2. -/
theorem ratio_grass_area_weeded :
  (earnings_per_weed * (weeds_flower_bed + weeds_vegetable_patch) + earnings_per_weed * (weeds_flower_bed + (weeds_grass_total - (earnings_per_weed + soda_cost)) / earnings_per_weed) = soda_cost + cents_left)
→ ((earnings_per_weed  * (32 - (147 + 99) / earnings_per_weed)) / weeds_grass_total) = 1 / 2 :=
by
  sorry

end ratio_grass_area_weeded_l287_287154


namespace a_lt_1_sufficient_but_not_necessary_l287_287797

noncomputable def represents_circle (a : ℝ) : Prop :=
  a^2 - 10 * a + 9 > 0

theorem a_lt_1_sufficient_but_not_necessary (a : ℝ) :
  represents_circle a → ((a < 1) ∨ (a > 9)) :=
sorry

end a_lt_1_sufficient_but_not_necessary_l287_287797


namespace johns_uncommon_cards_l287_287603

def packs_bought : ℕ := 10
def cards_per_pack : ℕ := 20
def uncommon_fraction : ℚ := 1 / 4

theorem johns_uncommon_cards : packs_bought * (cards_per_pack * uncommon_fraction) = (50 : ℚ) := 
by 
  sorry

end johns_uncommon_cards_l287_287603


namespace cubic_roots_proof_l287_287397

noncomputable def cubic_roots_reciprocal (a b c : ℝ) (h1 : a + b + c = 7) 
  (h2 : a * b + b * c + c * a = 3) (h3 : a * b * c = -4) : ℝ :=
  (1 / a^2) + (1 / b^2) + (1 / c^2)

theorem cubic_roots_proof (a b c : ℝ) (h1 : a + b + c = 7) 
  (h2 : a * b + b * c + c * a = 3) (h3 : a * b * c = -4) : 
  cubic_roots_reciprocal a b c h1 h2 h3 = 65 / 16 :=
sorry

end cubic_roots_proof_l287_287397


namespace coconut_grove_yield_l287_287593

theorem coconut_grove_yield (x Y : ℕ) (h1 : x = 10)
  (h2 : (x + 2) * 30 + x * Y + (x - 2) * 180 = 3 * x * 100) : Y = 120 :=
by
  -- Proof to be provided
  sorry

end coconut_grove_yield_l287_287593


namespace coefficients_equality_l287_287723

theorem coefficients_equality (a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h : a_1 * (x-1)^4 + a_2 * (x-1)^3 + a_3 * (x-1)^2 + a_4 * (x-1) + a_5 = x^4)
  (h1 : a_1 = 1)
  (h2 : a_5 = 1)
  (h3 : 1 - a_2 + a_3 - a_4 + 1 = 0) :
  a_2 - a_3 + a_4 = 2 :=
sorry

end coefficients_equality_l287_287723


namespace initial_population_l287_287476

theorem initial_population (P : ℝ) (h : P * (1.24 : ℝ)^2 = 18451.2) : P = 12000 :=
by
  sorry

end initial_population_l287_287476


namespace jake_first_test_score_l287_287600

theorem jake_first_test_score 
  (avg_score : ℕ)
  (n_tests : ℕ)
  (second_test_extra : ℕ)
  (third_test_score : ℕ)
  (x : ℕ) : 
  avg_score = 75 → 
  n_tests = 4 → 
  second_test_extra = 10 → 
  third_test_score = 65 →
  (x + (x + second_test_extra) + third_test_score + third_test_score) / n_tests = avg_score →
  x = 80 := by
  intros h1 h2 h3 h4 h5
  sorry

end jake_first_test_score_l287_287600


namespace fixed_point_PQ_l287_287858

open Real

-- Definitions of the points
def A : ℝ × ℝ := (-sqrt 5, 0)
def B : ℝ × ℝ := (sqrt 5, 0)
def M : ℝ × ℝ := (2, 0)

-- Condition that the incenter lies on x = 2
def incenter_on_x_eq_2 (C : ℝ × ℝ) : Prop :=
  C.1 = 2

-- Condition that vectors MP and MQ are orthogonal
def orthogonality_condition (P Q : ℝ × ℝ) : Prop :=
  let MP := (P.1 - M.1, P.2 - M.2) in
  let MQ := (Q.1 - M.1, Q.2 - M.2) in
  MP.1 * MQ.1 + MP.2 * MQ.2 = 0

-- The proof problem
theorem fixed_point_PQ (P Q : ℝ × ℝ) (C : ℝ × ℝ) :
  incenter_on_x_eq_2 C →
  orthogonality_condition P Q →
  ∃ (F : ℝ × ℝ), F = (10 / 3, 0) ∧
    (line_through P Q) F := sorry

end fixed_point_PQ_l287_287858


namespace max_value_expr_l287_287832

theorem max_value_expr : ∃ x : ℝ, (3 * x^2 + 9 * x + 28) / (3 * x^2 + 9 * x + 7) = 85 :=
by sorry

end max_value_expr_l287_287832


namespace election_winning_percentage_l287_287950

def total_votes (a b c : ℕ) : ℕ := a + b + c

def winning_percentage (votes_winning : ℕ) (total : ℕ) : ℚ :=
(votes_winning * 100 : ℚ) / total

theorem election_winning_percentage (a b c : ℕ) (h_votes : a = 6136 ∧ b = 7636 ∧ c = 11628) :
  winning_percentage c (total_votes a b c) = 45.78 := by
  sorry

end election_winning_percentage_l287_287950


namespace possible_red_ball_draws_l287_287505

/-- 
Given two balls in a bag where one is white and the other is red, 
if a ball is drawn and returned, and then another ball is drawn, 
prove that the possible number of times a red ball is drawn is 0, 1, or 2.
-/
theorem possible_red_ball_draws : 
  (∀ balls : Finset (ℕ × ℕ), 
    balls = {(0, 1), (1, 0)} →
    ∀ draw1 draw2 : ℕ × ℕ, 
    draw1 ∈ balls →
    draw2 ∈ balls →
    ∃ n : ℕ, (n = 0 ∨ n = 1 ∨ n = 2) ∧ 
    n = (if draw1 = (1, 0) then 1 else 0) + 
        (if draw2 = (1, 0) then 1 else 0)) → 
    True := sorry

end possible_red_ball_draws_l287_287505


namespace total_profit_is_35000_l287_287394

open Real

-- Define the subscriptions of A, B, and C
def subscriptions (A B C : ℝ) : Prop :=
  A + B + C = 50000 ∧
  A = B + 4000 ∧
  B = C + 5000

-- Define the profit distribution and the condition for C's received profit
def profit (total_profit : ℝ) (A B C : ℝ) (C_profit : ℝ) : Prop :=
  C_profit / total_profit = C / (A + B + C) ∧
  C_profit = 8400

-- Lean 4 statement to prove total profit
theorem total_profit_is_35000 :
  ∃ A B C total_profit, subscriptions A B C ∧ profit total_profit A B C 8400 ∧ total_profit = 35000 :=
by
  sorry

end total_profit_is_35000_l287_287394


namespace coupon_calculation_l287_287052

theorem coupon_calculation :
  let initial_stock : ℝ := 40.0
  let sold_books : ℝ := 20.0
  let coupons_per_book : ℝ := 4.0
  let remaining_books := initial_stock - sold_books
  let total_coupons := remaining_books * coupons_per_book
  total_coupons = 80.0 :=
by
  sorry

end coupon_calculation_l287_287052


namespace cube_edge_length_l287_287630

-- Definitions based on given conditions
def paper_cost_per_kg : ℝ := 60
def paper_area_coverage_per_kg : ℝ := 20
def total_expenditure : ℝ := 1800
def surface_area_of_cube (a : ℝ) : ℝ := 6 * a^2

-- The main proof problem
theorem cube_edge_length :
  ∃ a : ℝ, surface_area_of_cube a = paper_area_coverage_per_kg * (total_expenditure / paper_cost_per_kg) ∧ a = 10 :=
by
  sorry

end cube_edge_length_l287_287630


namespace probability_of_joining_between_1890_and_1969_l287_287117

theorem probability_of_joining_between_1890_and_1969 :
  let total_provinces_and_territories := 13
  let joined_1890_to_1929 := 3
  let joined_1930_to_1969 := 1
  let total_joined_between_1890_and_1969 := joined_1890_to_1929 + joined_1930_to_1969
  total_joined_between_1890_and_1969 / total_provinces_and_territories = 4 / 13 :=
by
  sorry

end probability_of_joining_between_1890_and_1969_l287_287117


namespace garden_perimeter_l287_287819

theorem garden_perimeter
  (a b : ℝ)
  (h1: a^2 + b^2 = 225)
  (h2: a * b = 54) :
  2 * (a + b) = 2 * Real.sqrt 333 :=
by
  sorry

end garden_perimeter_l287_287819


namespace oranges_left_to_sell_today_l287_287458

theorem oranges_left_to_sell_today (initial_dozen : Nat)
    (reserved_fraction1 reserved_fraction2 sold_fraction eaten_fraction : ℚ)
    (rotten_oranges : Nat) 
    (h1 : initial_dozen = 7)
    (h2 : reserved_fraction1 = 1/4)
    (h3 : reserved_fraction2 = 1/6)
    (h4 : sold_fraction = 3/7)
    (h5 : eaten_fraction = 1/10)
    (h6 : rotten_oranges = 4) : 
    let total_oranges := initial_dozen * 12
    let reserved1 := total_oranges * reserved_fraction1
    let reserved2 := total_oranges * reserved_fraction2
    let remaining_after_reservation := total_oranges - reserved1 - reserved2
    let sold_yesterday := remaining_after_reservation * sold_fraction
    let remaining_after_sale := remaining_after_reservation - sold_yesterday
    let eaten_by_birds := remaining_after_sale * eaten_fraction
    let remaining_after_birds := remaining_after_sale - eaten_by_birds
    let final_remaining := remaining_after_birds - rotten_oranges
    final_remaining = 22 :=
by
    sorry

end oranges_left_to_sell_today_l287_287458


namespace speed_in_still_water_l287_287968

theorem speed_in_still_water (v_m v_s : ℝ)
  (downstream : 48 = (v_m + v_s) * 3)
  (upstream : 34 = (v_m - v_s) * 4) :
  v_m = 12.25 :=
by
  sorry

end speed_in_still_water_l287_287968


namespace sum_of_coords_of_four_points_l287_287932

noncomputable def four_points_sum_coords : ℤ :=
  let y1 := 13 + 5
  let y2 := 13 - 5
  let x1 := 7 + 12
  let x2 := 7 - 12
  ((x2 + y2) + (x2 + y1) + (x1 + y2) + (x1 + y1))

theorem sum_of_coords_of_four_points : four_points_sum_coords = 80 :=
  by
    sorry

end sum_of_coords_of_four_points_l287_287932


namespace fixed_point_of_line_range_of_a_to_avoid_second_quadrant_l287_287423

theorem fixed_point_of_line (a : ℝ) (A : ℝ × ℝ) :
  (∀ x y : ℝ, (a - 1) * x + y - a - 5 = 0 -> A = (1, 6)) :=
sorry

theorem range_of_a_to_avoid_second_quadrant (a : ℝ) :
  (∀ x y : ℝ, (a - 1) * x + y - a - 5 = 0 -> x * y < 0 -> a ≤ -5) :=
sorry

end fixed_point_of_line_range_of_a_to_avoid_second_quadrant_l287_287423


namespace geometric_progression_product_l287_287414

theorem geometric_progression_product (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ)
  (h1 : a 3 = a1 * r^2)
  (h2 : a 10 = a1 * r^9)
  (h3 : a1 * r^2 + a1 * r^9 = 3)
  (h4 : a1^2 * r^11 = -5) :
  a 5 * a 8 = -5 :=
by
  sorry

end geometric_progression_product_l287_287414


namespace tree_height_after_4_months_l287_287911

noncomputable def tree_growth_rate := 50 -- growth in centimeters per two weeks
noncomputable def current_height_meters := 2 -- current height in meters
noncomputable def weeks_in_a_month := 4

def current_height_cm := current_height_meters * 100
def months := 4
def total_weeks := months * weeks_in_a_month
def growth_periods := total_weeks / 2
def total_growth := growth_periods * tree_growth_rate
def final_height := total_growth + current_height_cm

theorem tree_height_after_4_months :
  final_height = 600 :=
  by
    sorry

end tree_height_after_4_months_l287_287911


namespace reflection_equation_l287_287967

theorem reflection_equation
  (incident_line : ∀ x y : ℝ, 2 * x - y + 2 = 0)
  (reflection_axis : ∀ x y : ℝ, x + y - 5 = 0) :
  ∃ x y : ℝ, x - 2 * y + 7 = 0 :=
by
  sorry

end reflection_equation_l287_287967


namespace trig_identity_example_l287_287246

noncomputable def cos24 := Real.cos (24 * Real.pi / 180)
noncomputable def cos36 := Real.cos (36 * Real.pi / 180)
noncomputable def sin24 := Real.sin (24 * Real.pi / 180)
noncomputable def sin36 := Real.sin (36 * Real.pi / 180)
noncomputable def cos60 := Real.cos (60 * Real.pi / 180)

theorem trig_identity_example :
  cos24 * cos36 - sin24 * sin36 = cos60 :=
by
  sorry

end trig_identity_example_l287_287246


namespace relationship_y1_y2_y3_l287_287093

-- Define the quadratic function with the given parameters
def quadratic (a c x : ℝ) : ℝ := a * (x - 3) ^ 2 + c

-- Given conditions
variable (a c : ℝ)
variable (ha : a < 0)

-- Function values at specific x-values
def y1 := quadratic a c (Real.sqrt 5)
def y2 := quadratic a c 0
def y3 := quadratic a c 4

-- The theorem stating the desired relationship
theorem relationship_y1_y2_y3 : y2 < y3 ∧ y3 < y1 :=
by
  -- Proof goes here, using the given conditions
  sorry

end relationship_y1_y2_y3_l287_287093


namespace quadratic_inequality_solution_range_l287_287036

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, x^2 + (a-1)*x + 1 < 0) → (a > 3 ∨ a < -1) :=
by
  sorry

end quadratic_inequality_solution_range_l287_287036


namespace double_rooms_booked_l287_287227

theorem double_rooms_booked (S D : ℕ) 
  (h1 : S + D = 260) 
  (h2 : 35 * S + 60 * D = 14000) : 
  D = 196 :=
by
  sorry

end double_rooms_booked_l287_287227


namespace f_x_minus_one_l287_287724

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 5

theorem f_x_minus_one (x : ℝ) : f (x - 1) = x^2 - 4 * x + 8 :=
by
  sorry

end f_x_minus_one_l287_287724


namespace tangent_line_eq_l287_287471

theorem tangent_line_eq (x y : ℝ) (h_curve : y = Real.log x + x^2) (h_point : (x, y) = (1, 1)) : 
  3 * x - y - 2 = 0 :=
sorry

end tangent_line_eq_l287_287471


namespace greatest_divisor_arithmetic_sum_l287_287339

theorem greatest_divisor_arithmetic_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ d, d = 15 ∧ ∀ S : ℕ, S = 15 * x + 105 * c → d ∣ S :=
by 
  sorry

end greatest_divisor_arithmetic_sum_l287_287339


namespace total_earnings_l287_287579

theorem total_earnings : 
  let wage : ℕ := 10
  let hours_monday : ℕ := 7
  let tips_monday : ℕ := 18
  let hours_tuesday : ℕ := 5
  let tips_tuesday : ℕ := 12
  let hours_wednesday : ℕ := 7
  let tips_wednesday : ℕ := 20
  let total_hours : ℕ := hours_monday + hours_tuesday + hours_wednesday
  let earnings_from_wage : ℕ := total_hours * wage
  let total_tips : ℕ := tips_monday + tips_tuesday + tips_wednesday
  let total_earnings : ℕ := earnings_from_wage + total_tips
  total_earnings = 240 :=
by
  sorry

end total_earnings_l287_287579


namespace train_length_is_549_95_l287_287973

noncomputable def length_of_train 
(speed_of_train : ℝ) -- 63 km/hr
(speed_of_man : ℝ) -- 3 km/hr
(time_to_cross : ℝ) -- 32.997 seconds
: ℝ := 
(speed_of_train - speed_of_man) * (5 / 18) * time_to_cross

theorem train_length_is_549_95 (speed_of_train : ℝ) (speed_of_man : ℝ) (time_to_cross : ℝ) :
    speed_of_train = 63 → speed_of_man = 3 → time_to_cross = 32.997 →
    length_of_train speed_of_train speed_of_man time_to_cross = 549.95 :=
by
  intros h_train h_man h_time
  rw [h_train, h_man, h_time]
  norm_num
  sorry

end train_length_is_549_95_l287_287973


namespace cost_per_mile_proof_l287_287209

noncomputable def daily_rental_cost : ℝ := 50
noncomputable def daily_budget : ℝ := 88
noncomputable def max_miles : ℝ := 190.0

theorem cost_per_mile_proof : 
  (daily_budget - daily_rental_cost) / max_miles = 0.20 := 
by
  sorry

end cost_per_mile_proof_l287_287209


namespace arithmetic_sequence_properties_l287_287142

noncomputable def common_difference (a d : ℚ) (n : ℕ) : ℚ := a + (n - 1) * d

theorem arithmetic_sequence_properties :
  ∃ d : ℚ, d = 5 / 9 ∧ ∃ S : ℚ, S = -29 / 3 ∧
  ∀ n : ℕ, ∃ a₁ a₅ a₈ : ℚ, a₁ = -3 ∧
    a₅ = common_difference a₁ d 5 ∧
    a₈ = common_difference a₁ d 8 ∧ 
    11 * a₅ = 5 * a₈ - 13 ∧
    S = (n / 2) * (2 * a₁ + (n - 1) * d) ∧
    n = 6 := 
sorry

end arithmetic_sequence_properties_l287_287142


namespace solve_quadratic_l287_287428

theorem solve_quadratic (x : ℝ) (h : x^2 - 6*x + 8 = 0) : x = 2 ∨ x = 4 :=
sorry

end solve_quadratic_l287_287428


namespace remainder_of_multiple_l287_287115

theorem remainder_of_multiple (m k : ℤ) (h1 : m % 5 = 2) (h2 : (2 * k) % 5 = 1) : 
  (k * m) % 5 = 1 := 
sorry

end remainder_of_multiple_l287_287115


namespace solve_x_in_equation_l287_287372

theorem solve_x_in_equation (x : ℕ) (h : x + (x + 1) + (x + 2) + (x + 3) = 18) : x = 3 :=
by
  sorry

end solve_x_in_equation_l287_287372


namespace fraction_of_tomato_plants_in_second_garden_l287_287914

theorem fraction_of_tomato_plants_in_second_garden 
    (total_plants_first_garden : ℕ := 20)
    (percent_tomato_first_garden : ℚ := 10 / 100)
    (total_plants_second_garden : ℕ := 15)
    (percent_total_tomato_plants : ℚ := 20 / 100) :
    (15 : ℚ) * (1 / 3) = 5 :=
by
  sorry

end fraction_of_tomato_plants_in_second_garden_l287_287914


namespace combined_vacations_and_classes_l287_287109

-- Define the conditions
def Kelvin_classes : ℕ := 90
def Grant_vacations : ℕ := 4 * Kelvin_classes

-- The Lean 4 statement proving the combined total of vacations and classes
theorem combined_vacations_and_classes : Kelvin_classes + Grant_vacations = 450 := by
  sorry

end combined_vacations_and_classes_l287_287109


namespace distance_covered_at_40kmph_l287_287513

def total_distance : ℝ := 250
def speed_40 : ℝ := 40
def speed_60 : ℝ := 60
def total_time : ℝ := 5.2

theorem distance_covered_at_40kmph :
  ∃ (x : ℝ), (x / speed_40 + (total_distance - x) / speed_60 = total_time) ∧ x = 124 :=
  sorry

end distance_covered_at_40kmph_l287_287513


namespace highest_score_l287_287623

variable (avg runs_excluding: ℕ)
variable (innings remaining_innings total_runs total_runs_excluding H L: ℕ)

axiom batting_average (h_avg: avg = 60) (h_innings: innings = 46) : total_runs = avg * innings
axiom diff_highest_lowest_score (h_diff: H - L = 190) : true
axiom avg_excluding_high_low (h_avg_excluding: runs_excluding = 58) (h_remaining_innings: remaining_innings = 44) : total_runs_excluding = runs_excluding * remaining_innings
axiom sum_high_low : total_runs - total_runs_excluding = 208

theorem highest_score (h_avg: avg = 60) (h_innings: innings = 46) (h_diff: H - L = 190) (h_avg_excluding: runs_excluding = 58) (h_remaining_innings: remaining_innings = 44)
    (calc_total_runs: total_runs = avg * innings) 
    (calc_total_runs_excluding: total_runs_excluding = runs_excluding * remaining_innings)
    (calc_sum_high_low: total_runs - total_runs_excluding = 208) : H = 199 :=
by
  sorry

end highest_score_l287_287623


namespace minimum_value_x3_plus_y3_minus_5xy_l287_287412

theorem minimum_value_x3_plus_y3_minus_5xy (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  x^3 + y^3 - 5 * x * y ≥ -125 / 27 := 
sorry

end minimum_value_x3_plus_y3_minus_5xy_l287_287412


namespace greatest_divisor_arithmetic_sequence_sum_l287_287349

theorem greatest_divisor_arithmetic_sequence_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ k, (15 * (x + 7 * c)) = 15 * k :=
sorry

end greatest_divisor_arithmetic_sequence_sum_l287_287349


namespace arrangement_ways_count_l287_287389

theorem arrangement_ways_count:
  let n := 10
  let k := 4
  (Nat.choose n k) = 210 :=
by
  sorry

end arrangement_ways_count_l287_287389


namespace find_square_l287_287570

theorem find_square (q x : ℝ) 
  (h1 : x + q = 74) 
  (h2 : x + 2 * q^2 = 180) : 
  x = 66 :=
by {
  sorry
}

end find_square_l287_287570


namespace runners_meet_time_l287_287186

theorem runners_meet_time :
  let time_runner_1 := 2
  let time_runner_2 := 4
  let time_runner_3 := 11 / 2
  Nat.lcm time_runner_1 (Nat.lcm time_runner_2 (Nat.lcm (11) 2)) = 44 := by
  sorry

end runners_meet_time_l287_287186


namespace company_pays_per_month_after_new_hires_l287_287812

theorem company_pays_per_month_after_new_hires :
  let initial_employees := 500 in
  let new_employees := 200 in
  let hourly_pay := 12 in
  let hours_per_day := 10 in
  let days_per_week := 5 in
  let weeks_per_month := 4 in
  let total_employees := initial_employees + new_employees in
  let daily_pay := hourly_pay * hours_per_day in
  let working_days_in_month := days_per_week * weeks_per_month in
  let monthly_pay_per_employee := daily_pay * working_days_in_month in
  let total_monthly_payment := total_employees * monthly_pay_per_employee in
  total_monthly_payment = 1680000 := 
by
  sorry

end company_pays_per_month_after_new_hires_l287_287812


namespace decreasing_function_l287_287160

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

theorem decreasing_function (x₁ x₂ : ℝ) (h₁ : x₁ < x₂) (h₂ : x₂ ≤ 1) : 
  f x₁ > f x₂ :=
by
  -- Proof goes here
  sorry

end decreasing_function_l287_287160


namespace find_constants_l287_287902

open Matrix

def N : Matrix (Fin 2) (Fin 2) ℚ := ![![3, -1], ![2, -4]]
def I : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 0], ![0, 1]]

theorem find_constants (x y : ℚ) (hx : x = 1 / 14) (hy : y = 1 / 14) : 
  N⁻¹ = x • N + y • I := by
  sorry

end find_constants_l287_287902


namespace decompose_one_into_five_unit_fractions_l287_287655

theorem decompose_one_into_five_unit_fractions :
  1 = (1/2) + (1/3) + (1/7) + (1/43) + (1/1806) :=
by
  sorry

end decompose_one_into_five_unit_fractions_l287_287655


namespace car_miles_per_gallon_in_city_l287_287376

-- Define the conditions and the problem
theorem car_miles_per_gallon_in_city :
  ∃ C H T : ℝ, 
    H = 462 / T ∧ 
    C = 336 / T ∧ 
    C = H - 12 ∧ 
    C = 32 :=
by
  sorry

end car_miles_per_gallon_in_city_l287_287376


namespace carol_betty_age_ratio_l287_287064

theorem carol_betty_age_ratio:
  ∀ (C A B : ℕ), 
    C = 5 * A → 
    A = C - 12 → 
    B = 6 → 
    C / B = 5 / 2 :=
by
  intros C A B h1 h2 h3
  sorry

end carol_betty_age_ratio_l287_287064


namespace multiplication_example_l287_287230

theorem multiplication_example : 28 * (9 + 2 - 5) * 3 = 504 := by 
  sorry

end multiplication_example_l287_287230


namespace triangle_area_rational_l287_287530

-- Define the conditions
def satisfies_eq (x y : ℤ) : Prop := x - y = 1

-- Define the points
variables (x1 y1 x2 y2 x3 y3 : ℤ)

-- Assume each point satisfies the equation
axiom point1 : satisfies_eq x1 y1
axiom point2 : satisfies_eq x2 y2
axiom point3 : satisfies_eq x3 y3

-- Statement that we need to prove
theorem triangle_area_rational :
  ∃ (area : ℚ), 
    ∃ (triangle_points : ∃ (x1 y1 x2 y2 x3 y3 : ℤ), satisfies_eq x1 y1 ∧ satisfies_eq x2 y2 ∧ satisfies_eq x3 y3), 
      true :=
sorry

end triangle_area_rational_l287_287530


namespace negation_example_l287_287769

theorem negation_example :
  (¬ (∀ x: ℝ, x > 0 → x^2 + x + 1 > 0)) ↔ (∃ x: ℝ, x > 0 ∧ x^2 + x + 1 ≤ 0) :=
by
  sorry

end negation_example_l287_287769


namespace y_relationship_l287_287094

variable (a c : ℝ) (h_a : a < 0)

def f (x : ℝ) : ℝ := a * (x - 3) ^ 2 + c

theorem y_relationship (y1 y2 y3 : ℝ)
  (h1 : y1 = f a c (Real.sqrt 5))
  (h2 : y2 = f a c 0)
  (h3 : y3 = f a c 4) :
  y2 < y3 ∧ y3 < y1 :=
  sorry

end y_relationship_l287_287094


namespace number_of_valid_pairs_l287_287178

theorem number_of_valid_pairs : 
  ∃ (pairs : Finset (ℤ × ℤ)), pairs.card = 3 ∧ 
    (∀ (p ∈ pairs), ∃ (r k : ℤ), p = (r, k) ∧ 3 * r - 5 * k = 4 ∧ abs (r - k) ≤ 8) := 
by
  sorry

end number_of_valid_pairs_l287_287178


namespace problem_statement_l287_287289

-- Define the problem context
variables {a b c d : ℝ}

-- Define the conditions
def unit_square_condition (a b c d : ℝ) : Prop :=
  a^2 + b^2 + c^2 + d^2 ≥ 2 ∧ a^2 + b^2 + c^2 + d^2 ≤ 4 ∧ 
  a + b + c + d ≥ 2 * Real.sqrt 2 ∧ a + b + c + d ≤ 4

-- Provide the main theorem
theorem problem_statement (h : unit_square_condition a b c d) : 
  2 ≤ a^2 + b^2 + c^2 + d^2 ∧ a^2 + b^2 + c^2 + d^2 ≤ 4 ∧ 
  2 * Real.sqrt 2 ≤ a + b + c + d ∧ a + b + c + d ≤ 4 :=
  by 
  { sorry }  -- Proof to be completed

end problem_statement_l287_287289


namespace number_of_cases_in_top_level_l287_287744

-- Definitions for the total number of soda cases
def pyramid_cases (n : ℕ) : ℕ :=
  n^2 + (n + 1)^2 + (n + 2)^2 + (n + 3)^2

-- Theorem statement: proving the number of cases in the top level
theorem number_of_cases_in_top_level (n : ℕ) (h : pyramid_cases n = 30) : n = 1 :=
by {
  sorry
}

end number_of_cases_in_top_level_l287_287744


namespace find_g_at_4_l287_287318

theorem find_g_at_4 (g : ℝ → ℝ) (h : ∀ x, 2 * g x + 3 * g (1 - x) = 4 * x^3 - x) : g 4 = 193.2 :=
sorry

end find_g_at_4_l287_287318


namespace find_x_given_distance_l287_287140

theorem find_x_given_distance (x : ℝ) : abs (x - 4) = 1 → (x = 5 ∨ x = 3) :=
by
  intro h
  sorry

end find_x_given_distance_l287_287140


namespace aqua_park_earnings_l287_287537

def admission_cost : ℕ := 12
def tour_cost : ℕ := 6
def group1_size : ℕ := 10
def group2_size : ℕ := 5

theorem aqua_park_earnings :
  (group1_size * admission_cost + group1_size * tour_cost) + (group2_size * admission_cost) = 240 :=
by
  sorry

end aqua_park_earnings_l287_287537


namespace lunks_needed_for_20_apples_l287_287114

-- Definitions based on given conditions
def lunks_to_kunks (lunks : ℕ) : ℕ := (lunks / 4) * 2
def kunks_to_apples (kunks : ℕ) : ℕ := (kunks / 3) * 5

-- The main statement to be proven
theorem lunks_needed_for_20_apples :
  ∃ l : ℕ, (kunks_to_apples (lunks_to_kunks l)) = 20 ∧ l = 24 :=
by
  sorry

end lunks_needed_for_20_apples_l287_287114


namespace range_of_a_l287_287576

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = Real.sqrt x) :
  (f a < f (a + 1)) ↔ a ∈ Set.Ici (-1) :=
by
  sorry

end range_of_a_l287_287576


namespace simplify_and_evaluate_div_fraction_l287_287464

theorem simplify_and_evaluate_div_fraction (a : ℤ) (h : a = -3) : 
  (a - 2) / (1 + 2 * a + a^2) / (a - 3 * a / (a + 1)) = 1 / 6 := by
  sorry

end simplify_and_evaluate_div_fraction_l287_287464


namespace equation1_solution_equation2_solution_l287_287839

theorem equation1_solution (x : ℝ) : 4 * (2 * x - 1) ^ 2 = 36 ↔ x = 2 ∨ x = -1 :=
by sorry

theorem equation2_solution (x : ℝ) : (1 / 4) * (2 * x + 3) ^ 3 - 54 = 0 ↔ x = 3 / 2 :=
by sorry

end equation1_solution_equation2_solution_l287_287839


namespace find_m_n_l287_287851

theorem find_m_n (m n : ℝ) : (∀ x : ℝ, -5 ≤ x ∧ x ≤ 1 → x^2 - m * x + n ≤ 0) → m = -4 ∧ n = -5 :=
by
  sorry

end find_m_n_l287_287851


namespace units_sold_to_customer_c_l287_287822

theorem units_sold_to_customer_c 
  (initial_units : ℕ)
  (defective_units : ℕ)
  (units_a : ℕ)
  (units_b : ℕ)
  (units_c : ℕ)
  (h_initial : initial_units = 20)
  (h_defective : defective_units = 5)
  (h_units_a : units_a = 3)
  (h_units_b : units_b = 5)
  (h_non_defective : initial_units - defective_units = 15)
  (h_sold_all : units_a + units_b + units_c = 15) :
  units_c = 7 := by
  -- use sorry to skip the proof
  sorry

end units_sold_to_customer_c_l287_287822


namespace find_x_l287_287740

def infinite_sqrt (d : ℝ) : ℝ := sorry -- A placeholder since infinite nesting is non-trivial

def bowtie (c d : ℝ) : ℝ := c - infinite_sqrt d

theorem find_x (x : ℝ) (h : bowtie 7 x = 3) : x = 20 :=
sorry

end find_x_l287_287740


namespace ratio_length_breadth_l287_287175

noncomputable def b : ℝ := 18
noncomputable def l : ℝ := 972 / b

theorem ratio_length_breadth
  (A : ℝ)
  (h1 : b = 18)
  (h2 : l * b = 972) :
  (l / b) = 3 :=
by
  sorry

end ratio_length_breadth_l287_287175


namespace no_such_function_exists_l287_287074

namespace ProofProblem

open Nat

-- Declaration of the proposed function
def f : ℕ+ → ℕ+ := sorry

-- Statement to be proved
theorem no_such_function_exists : 
  ¬ ∃ f : ℕ+ → ℕ+, ∀ n : ℕ+, f^[n] n = n + 1 :=
by
  sorry

end ProofProblem

end no_such_function_exists_l287_287074


namespace greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l287_287367

theorem greatest_divisor_of_sum_first_15_terms_arithmetic_sequence
  (x c : ℕ) -- where x and c are positive integers
  (h_pos_x : 0 < x) -- x is positive
  (h_pos_c : 0 < c) -- c is positive
  : ∃ (d : ℕ), d = 15 ∧ ∀ (S : ℕ), S = 15 * x + 105 * c → d ∣ S :=
by
  sorry

end greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l287_287367


namespace find_constants_l287_287679

noncomputable def csc (x : ℝ) : ℝ := 1 / (Real.sin x)

theorem find_constants (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_min_value : ∃ x : ℝ, a * csc (b * x + c) = 3)
  (h_period : ∀ x, a * csc (b * (x + 4 * Real.pi) + c) = a * csc (b * x + c)) :
  a = 3 ∧ b = (1 / 2) :=
by
  sorry

end find_constants_l287_287679


namespace solve_students_and_apples_l287_287698

noncomputable def students_and_apples : Prop :=
  ∃ (x y : ℕ), y = 4 * x + 3 ∧ 6 * (x - 1) ≤ y ∧ y ≤ 6 * (x - 1) + 2 ∧ x = 4 ∧ y = 19

theorem solve_students_and_apples : students_and_apples :=
  sorry

end solve_students_and_apples_l287_287698


namespace range_of_x_for_a_range_of_a_l287_287854

-- Define propositions p and q
def prop_p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def prop_q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

-- Part (I)
theorem range_of_x_for_a (a x : ℝ) (ha : a = 1) (hpq : prop_p a x ∧ prop_q x) : 2 < x ∧ x < 3 :=
by
  sorry

-- Part (II)
theorem range_of_a (p q : ℝ → Prop) (hpq : ∀ x : ℝ, ¬p x → ¬q x) :
  1 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_x_for_a_range_of_a_l287_287854


namespace donut_distribution_l287_287823

theorem donut_distribution :
  ∃ (Alpha Beta Gamma Delta Epsilon : ℕ), 
    Delta = 8 ∧ 
    Beta = 3 * Gamma ∧ 
    Alpha = 2 * Delta ∧ 
    Epsilon = Gamma - 4 ∧ 
    Alpha + Beta + Gamma + Delta + Epsilon = 60 ∧ 
    Alpha = 16 ∧ 
    Beta = 24 ∧ 
    Gamma = 8 ∧ 
    Delta = 8 ∧ 
    Epsilon = 4 :=
by
  sorry

end donut_distribution_l287_287823


namespace Tony_total_payment_l287_287605

-- Defining the cost of items
def lego_block_cost : ℝ := 250
def toy_sword_cost : ℝ := 120
def play_dough_cost : ℝ := 35

-- Quantities of each item
def total_lego_blocks : ℕ := 3
def total_toy_swords : ℕ := 5
def total_play_doughs : ℕ := 10

-- Quantities purchased on each day
def first_day_lego_blocks : ℕ := 2
def first_day_toy_swords : ℕ := 3
def second_day_lego_blocks : ℕ := total_lego_blocks - first_day_lego_blocks
def second_day_toy_swords : ℕ := total_toy_swords - first_day_toy_swords
def second_day_play_doughs : ℕ := total_play_doughs

-- Discounts and tax rates
def first_day_discount : ℝ := 0.20
def second_day_discount : ℝ := 0.10
def sales_tax : ℝ := 0.05

-- Calculating first day purchase amounts
def first_day_cost_before_discount : ℝ := (first_day_lego_blocks * lego_block_cost) + (first_day_toy_swords * toy_sword_cost)
def first_day_discount_amount : ℝ := first_day_cost_before_discount * first_day_discount
def first_day_cost_after_discount : ℝ := first_day_cost_before_discount - first_day_discount_amount
def first_day_sales_tax_amount : ℝ := first_day_cost_after_discount * sales_tax
def first_day_total_cost : ℝ := first_day_cost_after_discount + first_day_sales_tax_amount

-- Calculating second day purchase amounts
def second_day_cost_before_discount : ℝ := (second_day_lego_blocks * lego_block_cost) + (second_day_toy_swords * toy_sword_cost) + 
                                           (second_day_play_doughs * play_dough_cost)
def second_day_discount_amount : ℝ := second_day_cost_before_discount * second_day_discount
def second_day_cost_after_discount : ℝ := second_day_cost_before_discount - second_day_discount_amount
def second_day_sales_tax_amount : ℝ := second_day_cost_after_discount * sales_tax
def second_day_total_cost : ℝ := second_day_cost_after_discount + second_day_sales_tax_amount

-- Total cost
def total_cost : ℝ := first_day_total_cost + second_day_total_cost

-- Lean theorem statement
theorem Tony_total_payment : total_cost = 1516.20 := by
  sorry

end Tony_total_payment_l287_287605


namespace percentage_of_sum_l287_287285

theorem percentage_of_sum (x y P : ℝ) (h1 : 0.50 * (x - y) = (P / 100) * (x + y)) (h2 : y = 0.25 * x) : P = 30 :=
by
  sorry

end percentage_of_sum_l287_287285


namespace function_is_increasing_on_interval_l287_287273

noncomputable def f (m x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * m * x^2 + 4 * x - 3

theorem function_is_increasing_on_interval {m : ℝ} :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → (1/3) * x^3 - (1/2) * m * x^2 + 4 * x - 3 ≥ (1/3) * (x - dx)^3 - (1/2) * m * (x - dx)^2 + 4 * (x - dx) - 3)
  ↔ m ≤ 4 :=
sorry

end function_is_increasing_on_interval_l287_287273


namespace max_min_values_f_decreasing_interval_f_l287_287860

noncomputable def a : ℝ × ℝ := (1 / 2, Real.sqrt 3 / 2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := ((a.1 * (b x).1) + (a.2 * (b x).2)) + 2

theorem max_min_values_f (k : ℤ) :
  (∃ (x1 : ℝ), (x1 = 2 * k * Real.pi + Real.pi / 6) ∧ f x1 = 3) ∧
  (∃ (x2 : ℝ), (x2 = 2 * k * Real.pi - 5 * Real.pi / 6) ∧ f x2 = 1) := 
sorry

theorem decreasing_interval_f :
  ∀ x, (Real.pi / 6 ≤ x ∧ x ≤ 7 * Real.pi / 6) → (∀ y, f x ≥ f y → x ≤ y) := 
sorry

end max_min_values_f_decreasing_interval_f_l287_287860


namespace math_proof_problem_l287_287224

theorem math_proof_problem (a : ℝ) : 
  (a^8 / a^4 ≠ a^4) ∧ ((a^2)^3 ≠ a^6) ∧ ((3*a)^3 ≠ 9*a^3) ∧ ((-a)^3 * (-a)^5 = a^8) := 
by 
  sorry

end math_proof_problem_l287_287224


namespace prime_p_q_r_condition_l287_287562

theorem prime_p_q_r_condition (p q r : ℕ) (hp : Nat.Prime p) (hq_pos : 0 < q) (hr_pos : 0 < r)
    (hp_not_dvd_q : ¬ (p ∣ q)) (h3_not_dvd_q : ¬ (3 ∣ q)) (eqn : p^3 = r^3 - q^2) : 
    p = 7 := sorry

end prime_p_q_r_condition_l287_287562


namespace cakes_bought_l287_287060

theorem cakes_bought (initial_cakes remaining_cakes : ℕ) (h_initial : initial_cakes = 155) (h_remaining : remaining_cakes = 15) : initial_cakes - remaining_cakes = 140 :=
by {
  sorry
}

end cakes_bought_l287_287060


namespace arcsin_sqrt_three_over_two_l287_287547

theorem arcsin_sqrt_three_over_two : Real.arcsin (Real.sqrt 3 / 2) = Real.pi / 3 :=
by
  -- The proof is omitted
  sorry

end arcsin_sqrt_three_over_two_l287_287547


namespace abs_value_condition_l287_287284

theorem abs_value_condition (m : ℝ) (h : |m - 1| = m - 1) : m ≥ 1 :=
by {
  sorry
}

end abs_value_condition_l287_287284


namespace periodic_even_function_l287_287741

open Real

noncomputable def f : ℝ → ℝ := sorry

theorem periodic_even_function (f : ℝ → ℝ)
  (h1 : ∀ x, f (x + 2) = f x)
  (h2 : ∀ x, f (-x) = f x)
  (h3 : ∀ x, 2 ≤ x ∧ x ≤ 3 → f x = x) :
  ∀ x, -2 ≤ x ∧ x ≤ 0 → f x = 3 - abs (x + 1) :=
sorry

end periodic_even_function_l287_287741


namespace nialls_children_ages_l287_287749

theorem nialls_children_ages : ∃ (a b c d : ℕ), 
  a < 18 ∧ b < 18 ∧ c < 18 ∧ d < 18 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a * b * c * d = 882 ∧ a + b + c + d = 32 :=
by
  sorry

end nialls_children_ages_l287_287749


namespace correct_operation_l287_287512

variable {a : ℝ}

theorem correct_operation : a^4 / (-a)^2 = a^2 := by
  sorry

end correct_operation_l287_287512


namespace sufficient_not_necessary_condition_l287_287152
open Real

theorem sufficient_not_necessary_condition (m : ℝ) :
  ((m = 0) → ∃ x y : ℝ, (m + 1) * x + (1 - m) * y - 1 = 0 ∧ (m - 1) * x + (2 * m + 1) * y + 4 = 0 ∧ 
  ((m + 1) * (m - 1) + (1 - m) * (2 * m + 1) = 0 ∨ (m = 1 ∨ m = 0))) :=
by sorry

end sufficient_not_necessary_condition_l287_287152


namespace int_values_satisfy_condition_l287_287479

theorem int_values_satisfy_condition :
  ∃ (count : ℕ), count = 10 ∧ ∀ (x : ℤ), 6 > Real.sqrt x ∧ Real.sqrt x > 5 ↔ (x ≥ 26 ∧ x ≤ 35) := by
  sorry

end int_values_satisfy_condition_l287_287479


namespace students_not_reading_novels_l287_287380

-- Define the conditions
def total_students : ℕ := 240
def students_three_or_more : ℕ := total_students * (1/6)
def students_two : ℕ := total_students * 0.35
def students_one : ℕ := total_students * (5/12)

-- The theorem to be proved
theorem students_not_reading_novels : 
  (total_students - (students_three_or_more + students_two + students_one) = 16) :=
by
  sorry -- skipping the proof

end students_not_reading_novels_l287_287380


namespace students_on_bus_after_all_stops_l287_287730

-- Define the initial number of students getting on the bus at the first stop.
def students_first_stop : ℕ := 39

-- Define the number of students added at the second stop.
def students_second_stop_add : ℕ := 29

-- Define the number of students getting off at the second stop.
def students_second_stop_remove : ℕ := 12

-- Define the number of students added at the third stop.
def students_third_stop_add : ℕ := 35

-- Define the number of students getting off at the third stop.
def students_third_stop_remove : ℕ := 18

-- Calculating the expected number of students on the bus after all stops.
def total_students_expected : ℕ :=
  students_first_stop + students_second_stop_add - students_second_stop_remove +
  students_third_stop_add - students_third_stop_remove

-- The theorem stating the number of students on the bus after all stops.
theorem students_on_bus_after_all_stops : total_students_expected = 73 := by
  sorry

end students_on_bus_after_all_stops_l287_287730


namespace solveMatrixEquation_l287_287918

open Matrix

noncomputable def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![3, -1, 0], ![-2, 1, 1], ![2, -1, 4]]

noncomputable def B : Matrix (Fin 3) (Fin 1) ℝ :=
  ![![5], ![0], ![15]]

noncomputable def X : Matrix (Fin 3) (Fin 1) ℝ :=
  ![![2], ![1], ![3]]

theorem solveMatrixEquation : (A.mul X) = B := by
  sorry

end solveMatrixEquation_l287_287918


namespace negative_values_of_x_l287_287995

theorem negative_values_of_x : 
  let f (x : ℤ) := Int.sqrt (x + 196)
  ∃ (n : ℕ), (f (n ^ 2 - 196) > 0 ∧ f (n ^ 2 - 196) = n) ∧ ∃ k : ℕ, k = 13 :=
by
  sorry

end negative_values_of_x_l287_287995


namespace circle_eq_l287_287100

theorem circle_eq (A B : ℝ × ℝ) (hA1 : A = (5, 2)) (hA2 : B = (-1, 4)) (hx : ∃ (c : ℝ), (c, 0) = (c, 0)) :
  ∃ (C : ℝ) (D : ℝ) (x y : ℝ), (x + C) ^ 2 + y ^ 2 = D ∧ D = 20 ∧ (x - 1) ^ 2 + y ^ 2 = 20 :=
by
  sorry

end circle_eq_l287_287100


namespace diagonal_AC_possibilities_l287_287258

/-
In a quadrilateral with sides AB, BC, CD, and DA, the length of diagonal AC must 
satisfy the inequalities determined by the triangle inequalities for triangles 
ABC and CDA. Prove the number of different whole numbers that could be the 
length of diagonal AC is 13.
-/

def number_of_whole_numbers_AC (AB BC CD DA : ℕ) : ℕ :=
  if 6 < AB ∧ AB < 20 then 19 - 7 + 1 else sorry

theorem diagonal_AC_possibilities : number_of_whole_numbers_AC 7 13 15 10 = 13 :=
  by
    sorry

end diagonal_AC_possibilities_l287_287258


namespace smallest_five_sequential_number_greater_than_2000_is_2004_l287_287256

def fiveSequentialNumber (N : ℕ) : Prop :=
  (if 1 ∣ N then 1 else 0) + 
  (if 2 ∣ N then 1 else 0) + 
  (if 3 ∣ N then 1 else 0) + 
  (if 4 ∣ N then 1 else 0) + 
  (if 5 ∣ N then 1 else 0) + 
  (if 6 ∣ N then 1 else 0) + 
  (if 7 ∣ N then 1 else 0) + 
  (if 8 ∣ N then 1 else 0) + 
  (if 9 ∣ N then 1 else 0) ≥ 5

theorem smallest_five_sequential_number_greater_than_2000_is_2004 :
  ∀ N > 2000, fiveSequentialNumber N → N = 2004 :=
by
  intros N hn hfsn
  have hN : N = 2004 := sorry
  exact hN

end smallest_five_sequential_number_greater_than_2000_is_2004_l287_287256


namespace triangle_inequality_l287_287845

theorem triangle_inequality (A B C : ℝ) :
  ∀ (a b c : ℝ), (a = 2 * Real.sin (A / 2) * Real.cos (A / 2)) ∧
                 (b = 2 * Real.sin (B / 2) * Real.cos (B / 2)) ∧
                 (c = Real.cos ((A + B) / 2)) ∧
                 (x = Real.sqrt (Real.tan (A / 2) * Real.tan (B / 2)))
                 → (Real.sqrt (a * b) / Real.sin (C / 2) ≥ 3 * Real.sqrt 3 * Real.tan (A / 2) * Real.tan (B / 2)) := by {
  sorry
}

end triangle_inequality_l287_287845


namespace rope_segments_divided_l287_287670

theorem rope_segments_divided (folds1 folds2 : ℕ) (cut : ℕ) (h_folds1 : folds1 = 3) (h_folds2 : folds2 = 2) (h_cut : cut = 1) :
  (folds1 * folds2 + cut = 7) :=
by {
  -- Proof steps would go here
  sorry
}

end rope_segments_divided_l287_287670


namespace radius_of_semicircular_cubicle_l287_287526

noncomputable def radius_of_semicircle (P : ℝ) : ℝ := P / (Real.pi + 2)

theorem radius_of_semicircular_cubicle :
  radius_of_semicircle 71.9822971502571 = 14 := 
sorry

end radius_of_semicircular_cubicle_l287_287526


namespace solve_problem_l287_287806

noncomputable def problem_statement (x : ℝ) : Prop :=
  1 + Real.sin x - Real.cos (5 * x) - Real.sin (7 * x) = 2 * Real.cos (3 * x / 2) ^ 2

theorem solve_problem (x : ℝ) :
  problem_statement x ↔
  (∃ k : ℤ, x = (Real.pi / 8) * (2 * k + 1)) ∨ 
  (∃ n : ℤ, x = (Real.pi / 4) * (4 * n - 1)) :=
by
  sorry

end solve_problem_l287_287806


namespace ratio_of_girls_to_boys_l287_287748

theorem ratio_of_girls_to_boys (x y : ℕ) (h1 : x + y = 28) (h2 : x - y = 4) : x = 16 ∧ y = 12 ∧ x / y = 4 / 3 :=
by
  sorry

end ratio_of_girls_to_boys_l287_287748


namespace solution_set_l287_287904

open Set Real

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = - f x
axiom f_at_two : f 2 = 0
axiom f_cond : ∀ x : ℝ, 0 < x → x * (deriv (deriv f) x) + f x < 0

theorem solution_set :
  {x : ℝ | x * f x > 0} = Ioo (-2 : ℝ) 0 ∪ Ioo 0 2 :=
by
  sorry

end solution_set_l287_287904


namespace greatest_divisor_of_arithmetic_sequence_sum_l287_287343

theorem greatest_divisor_of_arithmetic_sequence_sum :
  ∀ (x c : ℕ), ∃ k : ℕ, k = 15 ∧ 15 ∣ (15 * x + 105 * c) :=
by
  intro x c
  exists 15
  split
  . rfl
  . sorry

end greatest_divisor_of_arithmetic_sequence_sum_l287_287343


namespace andy_solves_16_problems_l287_287977

theorem andy_solves_16_problems :
  ∃ N : ℕ, 
    N = (125 - 78)/3 + 1 ∧
    (78 + (N - 1) * 3 <= 125) ∧
    N = 16 := 
by 
  sorry

end andy_solves_16_problems_l287_287977


namespace greatest_divisor_arithmetic_sum_l287_287338

theorem greatest_divisor_arithmetic_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ d, d = 15 ∧ ∀ S : ℕ, S = 15 * x + 105 * c → d ∣ S :=
by 
  sorry

end greatest_divisor_arithmetic_sum_l287_287338


namespace sfl_entrances_l287_287796

theorem sfl_entrances (people_per_entrance total_people entrances : ℕ) 
  (h1: people_per_entrance = 283) 
  (h2: total_people = 1415) 
  (h3: total_people = people_per_entrance * entrances) 
  : entrances = 5 := 
  by 
  rw [h1, h2] at h3
  sorry

end sfl_entrances_l287_287796


namespace find_k_l287_287691

theorem find_k (k : ℝ) (hk : 0 < k) (slope_eq : (2 - k) / (k - 1) = k^2) : k = 1 :=
by sorry

end find_k_l287_287691


namespace total_number_of_course_selection_schemes_l287_287961

-- Define the total number of courses
def total_courses := 8

-- Define number of physical education and art courses
def pe_courses := 4
def art_courses := 4

-- Define selections: students choose 2 or 3 courses
def course_selections : Finset (Finset ℕ) :=
  (Finset.powerset (Finset.range total_courses)).filter (λ s, s.card = 2 ∨ s.card = 3)

-- Define condition: at least 1 course from each category
def valid_selections : Finset (Finset ℕ) :=
  course_selections.filter (λ s, ∃ pe art, s = pe ∪ art ∧ pe.card ≠ 0 ∧ art.card ≠ 0 ∧ 
                             pe ⊆ Finset.range pe_courses ∧ art ⊆ (Finset.range total_courses).filter (λ x, x ≥ pe_courses))

theorem total_number_of_course_selection_schemes : valid_selections.card = 64 := 
by sorry

end total_number_of_course_selection_schemes_l287_287961


namespace cyclist_C_speed_l287_287188

variables (c d : ℕ) -- Speeds of cyclists C and D in mph
variables (d_eq : d = c + 6) -- Cyclist D travels 6 mph faster than cyclist C
variables (h1 : 80 = 65 + 15) -- Total distance from X to Y and back to the meet point
variables (same_time : 65 / c = 95 / d) -- Equating the travel times of both cyclists

theorem cyclist_C_speed : c = 13 :=
by
  sorry -- Proof is omitted

end cyclist_C_speed_l287_287188


namespace probability_of_drawing_red_ball_l287_287204

noncomputable def probability_of_red_ball (total_balls red_balls : ℕ) : ℚ :=
  red_balls / total_balls

theorem probability_of_drawing_red_ball:
  probability_of_red_ball 5 3 = 3 / 5 :=
by
  unfold probability_of_red_ball
  norm_num

end probability_of_drawing_red_ball_l287_287204


namespace smallest_prime_after_six_nonprime_l287_287792

open Nat

noncomputable def is_nonprime (n : ℕ) : Prop :=
  ¬Prime n ∧ n ≠ 1

noncomputable def six_consecutive_nonprime (n : ℕ) : Prop :=
  is_nonprime n ∧ is_nonprime (n + 1) ∧ is_nonprime (n + 2) ∧ 
  is_nonprime (n + 3) ∧ is_nonprime (n + 4) ∧ is_nonprime (n + 5)

theorem smallest_prime_after_six_nonprime :
  ∃ n, six_consecutive_nonprime n ∧ (∀ m, m ≠ 97 → n + 6 < m → ¬Prime m) :=
by
  sorry

end smallest_prime_after_six_nonprime_l287_287792


namespace solution_set_of_inequality_l287_287635

theorem solution_set_of_inequality :
  {x : ℝ | x^2 - 5 * x + 6 ≤ 0} = {x : ℝ | 2 ≤ x ∧ x ≤ 3} :=
sorry

end solution_set_of_inequality_l287_287635


namespace count_non_congruent_triangles_with_perimeter_11_l287_287871

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def perimeter (a b c : ℕ) : Prop :=
  a + b + c = 11

def valid_triangle_sets : Nat :=
  if is_triangle 3 3 5 ∧ perimeter 3 3 5 then
    if is_triangle 2 4 5 ∧ perimeter 2 4 5 then 2
    else 1
  else 0

theorem count_non_congruent_triangles_with_perimeter_11 (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) :
  (perimeter a b c) → (is_triangle a b c) → valid_triangle_sets = 2 :=
by
  sorry

end count_non_congruent_triangles_with_perimeter_11_l287_287871


namespace intersection_of_sets_l287_287128

theorem intersection_of_sets :
  let M := { x : ℝ | 0 ≤ x ∧ x < 16 }
  let N := { x : ℝ | x ≥ 1/3 }
  M ∩ N = { x : ℝ | 1/3 ≤ x ∧ x < 16 } :=
by
  sorry

end intersection_of_sets_l287_287128


namespace positive_difference_sum_of_squares_l287_287195

-- Given definitions
def sum_of_squares_even (n : ℕ) : ℕ :=
  4 * (n * (n + 1) * (2 * n + 1)) / 6

def sum_of_squares_odd (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

-- The explicit values for this problem
def sum_of_squares_first_25_even := sum_of_squares_even 25
def sum_of_squares_first_20_odd := sum_of_squares_odd 20

-- The required proof statement
theorem positive_difference_sum_of_squares : 
  (sum_of_squares_first_25_even - sum_of_squares_first_20_odd) = 19230 := by
  sorry

end positive_difference_sum_of_squares_l287_287195


namespace probability_not_exceeding_40_l287_287941

variable (P : ℝ → Prop)

def less_than_30_grams : Prop := P 0.3
def between_30_and_40_grams : Prop := P 0.5

theorem probability_not_exceeding_40 (h1 : less_than_30_grams P) (h2 : between_30_and_40_grams P) : P 0.8 :=
by
  sorry

end probability_not_exceeding_40_l287_287941


namespace elevation_after_descend_l287_287448

theorem elevation_after_descend (initial_elevation : ℕ) (rate : ℕ) (time : ℕ) (final_elevation : ℕ) 
  (h_initial : initial_elevation = 400) 
  (h_rate : rate = 10) 
  (h_time : time = 5) 
  (h_final : final_elevation = initial_elevation - rate * time) : 
  final_elevation = 350 := 
by 
  sorry

end elevation_after_descend_l287_287448


namespace du_chin_fraction_of_sales_l287_287834

theorem du_chin_fraction_of_sales :
  let pies := 200
  let price_per_pie := 20
  let remaining_money := 1600
  let total_sales := pies * price_per_pie
  let used_for_ingredients := total_sales - remaining_money
  let fraction_used_for_ingredients := used_for_ingredients / total_sales
  fraction_used_for_ingredients = (3 / 5) := by
    sorry

end du_chin_fraction_of_sales_l287_287834


namespace max_temp_range_l287_287034

-- Definitions based on given conditions
def average_temp : ℤ := 40
def lowest_temp : ℤ := 30

-- Total number of days
def days : ℕ := 5

-- Given that the average temperature and lowest temperature are provided, prove the maximum range.
theorem max_temp_range 
  (avg_temp_eq : (average_temp * days) = 200)
  (temp_min : lowest_temp = 30) : 
  ∃ max_temp : ℤ, max_temp - lowest_temp = 50 :=
by
  -- Assume maximum temperature
  let max_temp := 80
  have total_sum := (average_temp * days)
  have min_occurrences := 3 * lowest_temp
  have highest_temp := total_sum - min_occurrences - lowest_temp
  have range := highest_temp - lowest_temp
  use max_temp
  sorry

end max_temp_range_l287_287034


namespace angelina_speed_from_library_to_gym_l287_287226

theorem angelina_speed_from_library_to_gym :
  ∃ (v : ℝ), 
    (840 / v - 510 / (1.5 * v) = 40) ∧
    (510 / (1.5 * v) - 480 / (2 * v) = 20) ∧
    (2 * v = 25) :=
by
  sorry

end angelina_speed_from_library_to_gym_l287_287226


namespace ticket_cost_l287_287058

theorem ticket_cost (a : ℝ)
  (h1 : ∀ c : ℝ, c = a / 3)
  (h2 : 3 * a + 5 * (a / 3) = 27.75) :
  6 * a + 9 * (a / 3) = 53.52 := 
sorry

end ticket_cost_l287_287058


namespace casey_saves_money_l287_287240

def first_employee_hourly_wage : ℕ := 20
def second_employee_hourly_wage : ℕ := 22
def subsidy_per_hour : ℕ := 6
def weekly_work_hours : ℕ := 40

theorem casey_saves_money :
  let first_employee_weekly_cost := first_employee_hourly_wage * weekly_work_hours
  let second_employee_effective_hourly_wage := second_employee_hourly_wage - subsidy_per_hour
  let second_employee_weekly_cost := second_employee_effective_hourly_wage * weekly_work_hours
  let savings := first_employee_weekly_cost - second_employee_weekly_cost
  savings = 160 :=
by
  sorry

end casey_saves_money_l287_287240


namespace total_course_selection_schemes_l287_287958

theorem total_course_selection_schemes (PE_courses : ℕ) (Art_courses : ℕ) : 
  PE_courses = 4 → Art_courses = 4 → 
  (finset.card (finset.powerset_len 2 (finset.range (PE_courses + Art_courses))).filter
    (λ s, ∃ (a b ∈ s), a < PE_courses ∧ b ≥ PE_courses) + 
   finset.card (finset.powerset_len 3 (finset.range (PE_courses + Art_courses))).filter
    (λ s, ∃ (a b c ∈ s), (a < PE_courses ∧ b < PE_courses ∧ c ≥ PE_courses) ∨ 
                        (a < PE_courses ∧ b ≥ PE_courses ∧ c ≥ PE_courses ∧ a ≠ b ≠ c)) = 64 :=
by
  sorry

end total_course_selection_schemes_l287_287958


namespace inequality_cannot_hold_l287_287875

variable (a b : ℝ)
variable (h : a < b ∧ b < 0)

theorem inequality_cannot_hold (h : a < b ∧ b < 0) : ¬ (1 / (a - b) > 1 / a) := 
by {
  sorry
}

end inequality_cannot_hold_l287_287875


namespace relationship_ab_c_l287_287710

def a := 0.8 ^ 0.8
def b := 0.8 ^ 0.9
def c := 1.2 ^ 0.8

theorem relationship_ab_c : c > a ∧ a > b := 
by
  -- The proof would go here
  sorry

end relationship_ab_c_l287_287710


namespace digit_difference_l287_287035

theorem digit_difference (X Y : ℕ) (h1 : 0 ≤ X ∧ X ≤ 9) (h2 : 0 ≤ Y ∧ Y ≤ 9) (h3 : (10 * X + Y) - (10 * Y + X) = 54) : X - Y = 6 :=
sorry

end digit_difference_l287_287035


namespace correct_calculation_l287_287654

theorem correct_calculation (a b : ℝ) : (-3 * a^3 * b)^2 = 9 * a^6 * b^2 :=
by sorry

end correct_calculation_l287_287654


namespace oldest_child_age_l287_287009

theorem oldest_child_age (x : ℕ) (h : (6 + 8 + x) / 3 = 9) : x = 13 := 
by {
  sorry
}

end oldest_child_age_l287_287009


namespace trees_chopped_l287_287667

def pieces_of_firewood_per_log : Nat := 5
def logs_per_tree : Nat := 4
def total_firewood_chopped : Nat := 500

theorem trees_chopped (pieces_of_firewood_per_log = 5) (logs_per_tree = 4)
    (total_firewood_chopped = 500) :
    total_firewood_chopped / pieces_of_firewood_per_log / logs_per_tree = 25 := by
  sorry

end trees_chopped_l287_287667


namespace function_properties_l287_287718

noncomputable def f (x : ℝ) : ℝ := 3^x - 3^(-x)

theorem function_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) :=
by
  sorry

end function_properties_l287_287718


namespace mangoes_in_basket_B_l287_287596

theorem mangoes_in_basket_B :
  ∀ (A C D E B : ℕ), 
    (A = 15) →
    (C = 20) →
    (D = 25) →
    (E = 35) →
    (5 * 25 = A + C + D + E + B) →
    (B = 30) :=
by
  intros A C D E B hA hC hD hE hSum
  sorry

end mangoes_in_basket_B_l287_287596


namespace sum_of_midpoints_l287_287636

theorem sum_of_midpoints (a b c : ℝ) (h : a + b + c = 12) : 
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 12 :=
by
  sorry

end sum_of_midpoints_l287_287636


namespace right_triangle_satisfies_pythagorean_l287_287887

-- Definition of the sides of the triangle
def a : ℕ := 3
def b : ℕ := 4
def c : ℕ := 5

-- The theorem to prove
theorem right_triangle_satisfies_pythagorean :
  a^2 + b^2 = c^2 :=
by
  sorry

end right_triangle_satisfies_pythagorean_l287_287887


namespace find_b_l287_287873

theorem find_b (x y z a b : ℝ) (h1 : x + y = 2) (h2 : xy - z^2 = a) (h3 : b = x + y + z) : b = 2 :=
by
  sorry

end find_b_l287_287873


namespace smallest_k_l287_287413

theorem smallest_k (a b c d e k : ℕ) (h1 : a + 2 * b + 3 * c + 4 * d + 5 * e = k)
  (h2 : 5 * a = 4 * b) (h3 : 4 * b = 3 * c) (h4 : 3 * c = 2 * d) (h5 : 2 * d = e) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) : k = 522 :=
sorry

end smallest_k_l287_287413


namespace solve_inequality_system_l287_287618

theorem solve_inequality_system : 
  (∀ x : ℝ, (1 / 3 * x - 1 ≤ 1 / 2 * x + 1) ∧ ((3 * x - (x - 2) ≥ 6) ∧ (x + 1 > (4 * x - 1) / 3)) → (2 ≤ x ∧ x < 4)) := 
by
  intro x h
  sorry

end solve_inequality_system_l287_287618


namespace max_xy_l287_287453

theorem max_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 7 * x + 8 * y = 112) : xy ≤ 56 :=
sorry

end max_xy_l287_287453


namespace angles_sum_eq_l287_287898

variables {a b c : ℝ} {A B C : ℝ}

theorem angles_sum_eq {a b c : ℝ} {A B C : ℝ}
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : A > 0) (h5 : B > 0) (h6 : C > 0)
  (h7 : A + B + C = π)
  (h8 : (a + c - b) * (a + c + b) = 3 * a * c) :
  A + C = 2 * π / 3 :=
sorry

end angles_sum_eq_l287_287898


namespace probability_all_same_room_probability_at_least_two_same_room_l287_287504

/-- 
  Given that there are three people and each person is assigned to one of four rooms with equal probability,
  let P1 be the probability that all three people are assigned to the same room,
  and let P2 be the probability that at least two people are assigned to the same room.
  We need to prove:
  1. P1 = 1 / 16
  2. P2 = 5 / 8
-/
noncomputable def P1 : ℚ := sorry

noncomputable def P2 : ℚ := sorry

theorem probability_all_same_room :
  P1 = 1 / 16 :=
sorry

theorem probability_at_least_two_same_room :
  P2 = 5 / 8 :=
sorry

end probability_all_same_room_probability_at_least_two_same_room_l287_287504


namespace line_AB_l287_287577

-- Statements for circles and intersection
def circle_C1 (x y: ℝ) : Prop := x^2 + y^2 = 1
def circle_C2 (x y: ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 1

-- Points A and B are defined as the intersection points of circles C1 and C2
axiom A (x y: ℝ) : circle_C1 x y ∧ circle_C2 x y
axiom B (x y: ℝ) : circle_C1 x y ∧ circle_C2 x y

-- The goal is to prove that the line passing through points A and B has the equation x - y = 0
theorem line_AB (x y: ℝ) : circle_C1 x y → circle_C2 x y → (x - y = 0) :=
by
  sorry

end line_AB_l287_287577


namespace relationship_between_a_and_b_l287_287849

-- Definitions based on the conditions
def point1_lies_on_line (a : ℝ) : Prop := a = (2/3 : ℝ) * (-1 : ℝ) - 3
def point2_lies_on_line (b : ℝ) : Prop := b = (2/3 : ℝ) * (1/2 : ℝ) - 3

-- The main theorem to prove the relationship between a and b
theorem relationship_between_a_and_b (a b : ℝ) 
  (h1 : point1_lies_on_line a)
  (h2 : point2_lies_on_line b) : a < b :=
by
  -- Skipping the actual proof. Including sorry to indicate it's not provided.
  sorry

end relationship_between_a_and_b_l287_287849


namespace toys_produced_per_week_l287_287043

theorem toys_produced_per_week (daily_production : ℕ) (work_days_per_week : ℕ) (total_production : ℕ) :
  daily_production = 680 ∧ work_days_per_week = 5 → total_production = 3400 := by
  sorry

end toys_produced_per_week_l287_287043


namespace collinear_points_l287_287988

-- Define collinear points function
def collinear (x1 y1 z1 x2 y2 z2 x3 y3 z3: ℝ) : Prop :=
  ∀ (a b c : ℝ), a * (y2 - y1) * (z3 - z1) + b * (z2 - z1) * (x3 - x1) + c * (x2 - x1) * (y3 - y1) = 0

-- Problem statement
theorem collinear_points (a b : ℝ)
  (h : collinear 2 a b a 3 b a b 4) :
  a + b = -2 :=
sorry

end collinear_points_l287_287988


namespace valid_numbers_are_135_and_144_l287_287990

noncomputable def find_valid_numbers : List ℕ :=
  let numbers := [135, 144]
  numbers.filter (λ n =>
    let a := n / 100
    let b := (n / 10) % 10
    let c := n % 10
    n = (100 * a + 10 * b + c) ∧ n = a * b * c * (a + b + c)
  )

theorem valid_numbers_are_135_and_144 :
  find_valid_numbers = [135, 144] :=
by
  sorry

end valid_numbers_are_135_and_144_l287_287990


namespace absolute_value_inequality_range_of_xyz_l287_287949

-- Question 1 restated
theorem absolute_value_inequality (x : ℝ) :
  (|x + 2| + |x + 3| ≤ 2) ↔ -7/2 ≤ x ∧ x ≤ -3/2 :=
sorry

-- Question 2 restated
theorem range_of_xyz (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : 
  -1/2 ≤ x * y + y * z + z * x ∧ x * y + y * z + z * x ≤ 1 :=
sorry

end absolute_value_inequality_range_of_xyz_l287_287949


namespace not_power_of_two_l287_287916

theorem not_power_of_two (m n : ℕ) (hm : m > 0) (hn : n > 0) : 
  ¬ ∃ k : ℕ, (36 * m + n) * (m + 36 * n) = 2 ^ k :=
sorry

end not_power_of_two_l287_287916


namespace smallest_second_term_l287_287180

theorem smallest_second_term (a d : ℕ) (h1 : 5 * a + 10 * d = 95) (h2 : a > 0) (h3 : d > 0) : 
  a + d = 10 :=
sorry

end smallest_second_term_l287_287180


namespace complex_plane_squares_areas_l287_287894

theorem complex_plane_squares_areas (z : ℂ) 
  (h1 : z^3 - z = i * (z^2 - z) ∨ z^3 - z = -i * (z^2 - z))
  (h2 : z^4 - z = i * (z^3 - z) ∨ z^4 - z = -i * (z^3 - z)) :
  ( ∃ A₁ A₂ : ℝ, (A₁ = 10 ∨ A₁ = 18) ∧ (A₂ = 10 ∨ A₂ = 18) ) := 
sorry

end complex_plane_squares_areas_l287_287894


namespace minimum_value_quadratic_function_l287_287194

-- Defining the quadratic function y
def quadratic_function (x : ℝ) : ℝ := 4 * x^2 + 8 * x + 16

-- Statement asserting the minimum value of the quadratic function
theorem minimum_value_quadratic_function : ∃ (y_min : ℝ), (∀ x : ℝ, quadratic_function x ≥ y_min) ∧ y_min = 12 :=
by
  -- Here we would normally insert the proof, but we skip it with sorry
  sorry

end minimum_value_quadratic_function_l287_287194


namespace hour_hand_degrees_per_hour_l287_287767

-- Definitions based on the conditions
def number_of_rotations_in_6_days : ℕ := 12
def degrees_per_rotation : ℕ := 360
def hours_in_6_days : ℕ := 6 * 24

-- Statement to prove
theorem hour_hand_degrees_per_hour :
  (number_of_rotations_in_6_days * degrees_per_rotation) / hours_in_6_days = 30 :=
by sorry

end hour_hand_degrees_per_hour_l287_287767


namespace Hallie_earnings_l287_287581

theorem Hallie_earnings :
  let w := 10
  let hM := 7
  let tM := 18
  let hT := 5
  let tT := 12
  let hW := 7
  let tW := 20
  let mondayEarnings := hM * w + tM
  let tuesdayEarnings := hT * w + tT
  let wednesdayEarnings := hW * w + tW
  let totalEarnings := mondayEarnings + tuesdayEarnings + wednesdayEarnings
  totalEarnings = 240 := by {
    let w := 10
    let hM := 7
    let tM := 18
    let hT := 5
    let tT := 12
    let hW := 7
    let tW := 20
    let mondayEarnings := hM * w + tM
    let tuesdayEarnings := hT * w + tT
    let wednesdayEarnings := hW * w + tW
    let totalEarnings := mondayEarnings + tuesdayEarnings + wednesdayEarnings
    sorry
  }

end Hallie_earnings_l287_287581


namespace remainder_76_pow_77_mod_7_l287_287371

/-- Statement of the problem:
Prove that the remainder of \(76^{77}\) divided by 7 is 6.
-/
theorem remainder_76_pow_77_mod_7 :
  (76 ^ 77) % 7 = 6 := 
by
  sorry

end remainder_76_pow_77_mod_7_l287_287371


namespace number_of_sets_satisfying_conditions_l287_287565

open Set Finset

theorem number_of_sets_satisfying_conditions :
  {x : Finset ℕ // {1, 2} ⊆ x ∧ x ⊆ {1, 2, 3, 4, 5}}.card = 8 :=
by
  have subset_1_2_3_4_5 : {1, 2} ⊆ {1, 2, 3, 4, 5} := by simp
  let base_set := {1, 2, 3, 4, 5}
  let must_include := {1, 2}
  let optional_elements := base_set \ must_include
  let power_set_optional := optional_elements.powerset
  let all_valid_sets := power_set_optional.image (λ s, must_include ∪ s)
  have card_image : all_valid_sets.card = power_set_optional.card := by sorry
  have power_set_card : power_set_optional.card = 2 ^ optional_elements.card := by simp
  have optional_card : optional_elements.card = 3 := by simp
  rw [card_image, power_set_card, optional_card]
  simp

end number_of_sets_satisfying_conditions_l287_287565


namespace range_of_a_plus_b_l287_287586

theorem range_of_a_plus_b (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : |Real.log a| = |Real.log b|) (h₄ : a ≠ b) :
  2 < a + b :=
by
  sorry

end range_of_a_plus_b_l287_287586


namespace total_tablets_l287_287383

-- Variables for the numbers of Lenovo, Samsung, and Huawei tablets
variables (n x y : ℕ)

-- Conditions based on problem statement
def condition1 : Prop := 2 * x + 6 + y < n / 3

def condition2 : Prop := (n - 2 * x - y - 6 = 3 * y)

def condition3 : Prop := (n - 6 * x - y - 6 = 59)

-- The statement to prove that the total number of tablets is 94
theorem total_tablets (h1 : condition1 n x y) (h2 : condition2 n x y) (h3 : condition3 n x y) : n = 94 :=
by
  sorry

end total_tablets_l287_287383


namespace ratio_of_larger_to_smaller_l287_287639

theorem ratio_of_larger_to_smaller (x y : ℝ) (h1 : x > y) (h2 : x + y = 7 * (x - y)) :
  x / y = 4 / 3 :=
by
  sorry

end ratio_of_larger_to_smaller_l287_287639


namespace suitable_survey_set_l287_287472

def Survey1 := "Investigate the lifespan of a batch of light bulbs"
def Survey2 := "Investigate the household income situation in a city"
def Survey3 := "Investigate the vision of students in a class"
def Survey4 := "Investigate the efficacy of a certain drug"

-- Define what it means for a survey to be suitable for sample surveys
def suitable_for_sample_survey (survey : String) : Prop :=
  survey = Survey1 ∨ survey = Survey2 ∨ survey = Survey4

-- The question is to prove that the surveys suitable for sample surveys include exactly (1), (2), and (4).
theorem suitable_survey_set :
  {Survey1, Survey2, Survey4} = {s : String | suitable_for_sample_survey s} :=
by
  sorry

end suitable_survey_set_l287_287472


namespace intersection_point_l287_287522

variable (t u : ℝ)

def line1 (t : ℝ) := (2 + 3 * t, 2 - 4 * t)
def line2 (u : ℝ) := (4 + 5 * u, -6 + 3 * u)

theorem intersection_point :
  ∃ t u, line1 t = (160 / 29 : ℝ, -160 / 29 : ℝ) ∧ line1 t = line2 u :=
begin
  use [(46 / 29 : ℝ), (48 / 87 : ℝ)],
  simp [line1, line2],
  split,
  {
    simp,
    ring,
  },
  {
    simp,
    split,
    { norm_num, ring},
    { norm_num, ring }
  }
end

end intersection_point_l287_287522


namespace rectangle_area_l287_287981

theorem rectangle_area (l w r: ℝ) (h1 : l = 2 * r) (h2 : w = r) : l * w = 2 * r^2 :=
by sorry

end rectangle_area_l287_287981


namespace sum_of_ages_l287_287789

-- Define the variables for Viggo and his younger brother's ages
variables (v y : ℕ)

-- Condition: When Viggo's younger brother was 2, Viggo's age was 10 years more than twice his brother's age
def condition1 (v y : ℕ) := (y = 2 → v = 2 * y + 10)

-- Condition: Viggo's younger brother is currently 10 years old
def condition2 (y_current : ℕ) := y_current = 10

-- Define the current age of Viggo given the conditions
def viggo_current_age (v y y_current : ℕ) := v + (y_current - y)

-- Prove that the sum of their ages is 32
theorem sum_of_ages
  (v y y_current : ℕ)
  (h1 : condition1 v y)
  (h2 : condition2 y_current) :
  viggo_current_age v y y_current + y_current = 32 :=
by
  -- Apply sorry to skip the proof
  sorry

end sum_of_ages_l287_287789


namespace mappings_count_A_to_B_l287_287846

open Finset

def A : Finset ℕ := {1, 2}
def B : Finset ℕ := {3, 4}

theorem mappings_count_A_to_B : (card B) ^ (card A) = 4 :=
by
  -- This line will state that the proof is skipped for now.
  sorry

end mappings_count_A_to_B_l287_287846


namespace shaded_area_l287_287896

theorem shaded_area (R : ℝ) (r : ℝ) (hR : R = 10) (hr : r = R / 2) : 
  π * R^2 - 2 * (π * r^2) = 50 * π :=
by
  sorry

end shaded_area_l287_287896


namespace fraction_of_shaded_hexagons_l287_287895

-- Definitions
def total_hexagons : ℕ := 9
def shaded_hexagons : ℕ := 5

-- Theorem statement
theorem fraction_of_shaded_hexagons : 
  (shaded_hexagons: ℚ) / (total_hexagons : ℚ) = 5 / 9 := by
sorry

end fraction_of_shaded_hexagons_l287_287895


namespace counterpositive_prop_l287_287089

theorem counterpositive_prop (a b c : ℝ) (h : a^2 + b^2 + c^2 < 3) : a + b + c ≠ 3 := 
sorry

end counterpositive_prop_l287_287089


namespace car_average_speed_l287_287927

theorem car_average_speed
  (d1 d2 t1 t2 : ℕ)
  (h1 : d1 = 85)
  (h2 : d2 = 45)
  (h3 : t1 = 1)
  (h4 : t2 = 1) :
  let total_distance := d1 + d2
  let total_time := t1 + t2
  (total_distance / total_time = 65) :=
by
  sorry

end car_average_speed_l287_287927


namespace first_term_is_5_over_2_l287_287738

-- Define the arithmetic sequence and the sum of the first n terms.
def arith_seq (a d : ℕ) (n : ℕ) := a + (n - 1) * d
def S (a d : ℕ) (n : ℕ) := (n * (2 * a + (n - 1) * d)) / 2

-- Define the constant ratio condition.
def const_ratio (a d : ℕ) (n : ℕ) (c : ℕ) :=
  (S a d (3 * n) * 2) = c * (S a d n * 2)

-- Prove the first term is 5/2 given the conditions.
theorem first_term_is_5_over_2 (c : ℕ) (n : ℕ) (h : const_ratio a 5 n 9) : 
  a = 5 / 2 :=
sorry

end first_term_is_5_over_2_l287_287738


namespace casey_saving_l287_287237

-- Define the conditions
def cost_per_hour_first_employee : ℝ := 20
def cost_per_hour_second_employee : ℝ := 22
def subsidy_per_hour : ℝ := 6
def hours_per_week : ℝ := 40

-- Define the weekly cost calculations
def weekly_cost_first_employee := cost_per_hour_first_employee * hours_per_week
def effective_cost_per_hour_second_employee := cost_per_hour_second_employee - subsidy_per_hour
def weekly_cost_second_employee := effective_cost_per_hour_second_employee * hours_per_week

-- State the theorem
theorem casey_saving :
    weekly_cost_first_employee - weekly_cost_second_employee = 160 := 
by
  sorry

end casey_saving_l287_287237


namespace greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l287_287369

theorem greatest_divisor_of_sum_first_15_terms_arithmetic_sequence
  (x c : ℕ) -- where x and c are positive integers
  (h_pos_x : 0 < x) -- x is positive
  (h_pos_c : 0 < c) -- c is positive
  : ∃ (d : ℕ), d = 15 ∧ ∀ (S : ℕ), S = 15 * x + 105 * c → d ∣ S :=
by
  sorry

end greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l287_287369


namespace center_of_symmetry_l287_287831

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 3) * Real.tan (-7 * x + (Real.pi / 3))

theorem center_of_symmetry : f (Real.pi / 21) = 0 :=
by
  -- Mathematical proof goes here, skipping with sorry.
  sorry

end center_of_symmetry_l287_287831


namespace prob_of_nine_correct_is_zero_l287_287531

-- Define the necessary components and properties of the problem
def is_correct_placement (letter: ℕ) (envelope: ℕ) : Prop := letter = envelope

def is_random_distribution (letters : Fin 10 → Fin 10) : Prop := true

-- State the theorem formally
theorem prob_of_nine_correct_is_zero (f : Fin 10 → Fin 10) :
  is_random_distribution f →
  (∃ (count : ℕ), count = 9 ∧ (∀ i : Fin 10, is_correct_placement i (f i) ↔ i = count)) → false :=
by
  sorry

end prob_of_nine_correct_is_zero_l287_287531


namespace Petya_wins_optimally_l287_287183

-- Defining the game state and rules
inductive GameState
| PetyaWin
| VasyaWin

-- Rules of the game
def game_rule (n : ℕ) : Prop :=
  n > 0 ∧ (n % 3 = 0 ∨ n % 3 = 1 ∨ n % 3 = 2)

-- Determine the winner given the initial number of minuses
def determine_winner (n : ℕ) : GameState :=
  if n % 3 = 0 then GameState.PetyaWin else GameState.VasyaWin

-- Theorem: Petya will win the game if both play optimally
theorem Petya_wins_optimally (n : ℕ) (h1 : n = 2021) (h2 : game_rule n) : determine_winner n = GameState.PetyaWin :=
by {
  sorry
}

end Petya_wins_optimally_l287_287183


namespace hari_digs_well_alone_in_48_days_l287_287295

theorem hari_digs_well_alone_in_48_days :
  (1 / 16 + 1 / 24 + 1 / (Hari_days)) = 1 / 8 → Hari_days = 48 :=
by
  intro h
  sorry

end hari_digs_well_alone_in_48_days_l287_287295


namespace A_beats_B_by_7_seconds_l287_287440

noncomputable def speed_A : ℝ := 200 / 33
noncomputable def distance_A : ℝ := 200
noncomputable def time_A : ℝ := 33

noncomputable def distance_B : ℝ := 200
noncomputable def distance_B_at_time_A : ℝ := 165

-- B's speed is calculated at the moment A finishes the race
noncomputable def speed_B : ℝ := distance_B_at_time_A / time_A
noncomputable def time_B : ℝ := distance_B / speed_B

-- Prove that A beats B by 7 seconds
theorem A_beats_B_by_7_seconds : time_B - time_A = 7 := 
by 
  -- Proof goes here, assume all definitions and variables are correct.
  sorry

end A_beats_B_by_7_seconds_l287_287440


namespace total_course_selection_schemes_l287_287955

theorem total_course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose_two :=
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 1)
  let choose_three :=
    (Nat.choose PE_courses 2) * (Nat.choose Art_courses 1) +
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 2)
  total_courses = 8
  ∧ (choose_two + choose_three = 64) :=
by
  sorry

end total_course_selection_schemes_l287_287955


namespace correct_subtraction_l287_287201

theorem correct_subtraction (x : ℕ) (h : x - 42 = 50) : x - 24 = 68 :=
  sorry

end correct_subtraction_l287_287201


namespace fraction_product_is_one_l287_287508

theorem fraction_product_is_one : 
  (1 / 4) * (1 / 5) * (1 / 6) * 120 = 1 :=
by 
  sorry

end fraction_product_is_one_l287_287508


namespace KarenEggRolls_l287_287461

-- Definitions based on conditions
def OmarEggRolls : ℕ := 219
def TotalEggRolls : ℕ := 448

-- The statement to be proved
theorem KarenEggRolls : (TotalEggRolls - OmarEggRolls = 229) :=
by {
    -- Proof step goes here
    sorry
}

end KarenEggRolls_l287_287461


namespace age_of_first_person_added_l287_287198

theorem age_of_first_person_added :
  ∀ (T A x : ℕ),
    (T = 7 * A) →
    (T + x = 8 * (A + 2)) →
    (T + 15 = 8 * (A - 1)) →
    x = 39 :=
by
  intros T A x h1 h2 h3
  sorry

end age_of_first_person_added_l287_287198


namespace water_left_after_operations_l287_287147

theorem water_left_after_operations :
  let initial_water := (3 : ℚ)
  let water_used := (4 / 3 : ℚ)
  let extra_water := (1 / 2 : ℚ)
  initial_water - water_used + extra_water = (13 / 6 : ℚ) := 
by
  -- Skips the proof, as the focus is on the problem statement
  sorry

end water_left_after_operations_l287_287147


namespace pinecones_left_l287_287500

theorem pinecones_left 
  (total_pinecones : ℕ)
  (pct_eaten_by_reindeer pct_collected_for_fires : ℝ)
  (total_eaten_by_reindeer : ℕ := (pct_eaten_by_reindeer * total_pinecones).to_nat)
  (total_eaten_by_squirrels : ℕ := 2 * total_eaten_by_reindeer)
  (total_eaten : ℕ := total_eaten_by_reindeer + total_eaten_by_squirrels)
  (pinecones_after_eating : ℕ := total_pinecones - total_eaten)
  (total_collected_for_fires : ℕ := (pct_collected_for_fires * pinecones_after_eating).to_nat)
  (remaining_pinecones : ℕ := pinecones_after_eating - total_collected_for_fires)
  (2000_pinecones : total_pinecones = 2000)
  (20_percent_eaten_by_reindeer : pct_eaten_by_reindeer = 0.20)
  (25_percent_collected_for_fires : pct_collected_for_fires = 0.25) : 
  remaining_pinecones = 600 := 
by
  sorry

end pinecones_left_l287_287500


namespace find_number_l287_287386

theorem find_number (x : ℝ) (h : x + 33 + 333 + 33.3 = 399.6) : x = 0.3 :=
by
  sorry

end find_number_l287_287386


namespace triangle_to_pentagon_ratio_l287_287049

theorem triangle_to_pentagon_ratio (t p : ℕ) 
  (h1 : 3 * t = 15) 
  (h2 : 5 * p = 15) : (t : ℚ) / (p : ℚ) = 5 / 3 :=
by
  sorry

end triangle_to_pentagon_ratio_l287_287049


namespace num_perfect_squares_diff_consecutive_under_20000_l287_287694

theorem num_perfect_squares_diff_consecutive_under_20000 : 
  ∃ n, n = 71 ∧ ∀ a, a ^ 2 < 20000 → ∃ b, a ^ 2 = (b + 1) ^ 2 - b ^ 2 ↔ a ^ 2 % 2 = 1 :=
by
  sorry

end num_perfect_squares_diff_consecutive_under_20000_l287_287694


namespace coin_tails_probability_l287_287430

theorem coin_tails_probability (p : ℝ) (h : p = 0.5) (n : ℕ) (h_n : n = 3) :
  ∃ k : ℕ, k ≤ n ∧ (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k) = 0.375 :=
by
  sorry

end coin_tails_probability_l287_287430


namespace problem_statement_l287_287923

variable {f : ℝ → ℝ}

-- Assume the conditions provided in the problem statement.
def continuous_on_ℝ (f : ℝ → ℝ) : Prop := Continuous f
def condition_x_f_prime (f : ℝ → ℝ) (h : ℝ → ℝ) : Prop := ∀ x : ℝ, x * h x < 0

-- The main theorem statement based on the conditions and the correct answer.
theorem problem_statement (hf : continuous_on_ℝ f) (hf' : ∀ x : ℝ, x * (deriv f x) < 0) :
  f (-1) + f 1 < 2 * f 0 :=
sorry

end problem_statement_l287_287923


namespace bricks_needed_for_wall_l287_287276

noncomputable def brick_volume (length : ℝ) (height : ℝ) (thickness : ℝ) : ℝ :=
  length * height * thickness

noncomputable def wall_volume (length : ℝ) (height : ℝ) (average_thickness : ℝ) : ℝ :=
  length * height * average_thickness

noncomputable def number_of_bricks (wall_vol : ℝ) (brick_vol : ℝ) : ℝ :=
  wall_vol / brick_vol

theorem bricks_needed_for_wall : 
  let length_wall := 800
  let height_wall := 660
  let avg_thickness_wall := (25 + 22.5) / 2 -- in cm
  let length_brick := 25
  let height_brick := 11.25
  let thickness_brick := 6
  let mortar_thickness := 1

  let adjusted_length_brick := length_brick + mortar_thickness
  let adjusted_height_brick := height_brick + mortar_thickness

  let volume_wall := wall_volume length_wall height_wall avg_thickness_wall
  let volume_brick_with_mortar := brick_volume adjusted_length_brick adjusted_height_brick thickness_brick

  number_of_bricks volume_wall volume_brick_with_mortar = 6565 :=
by
  sorry

end bricks_needed_for_wall_l287_287276


namespace course_selection_schemes_count_l287_287957

-- Definitions based on the conditions
def num_physical_education_courses : ℕ := 4
def num_art_courses : ℕ := 4

-- Required to choose 2 or 3 courses, with at least one from each category
def valid_selection_cases : list (ℕ × ℕ) := [(1, 1), (1, 2), (2, 1)]

-- Calculate the number of ways to choose k courses from n courses using combination formula
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Calculate the total number of different course selection schemes
def total_course_selection_schemes : ℕ :=
  let case1 := C num_physical_education_courses 1 * C num_art_courses 1
  let case2a := C num_physical_education_courses 2 * C num_art_courses 1
  let case2b := C num_physical_education_courses 1 * C num_art_courses 2
  case1 + case2a + case2b

-- The assertion that the total number of different course selection schemes is 64
theorem course_selection_schemes_count : total_course_selection_schemes = 64 := by
  -- sorry placeholder for proof
  sorry

end course_selection_schemes_count_l287_287957


namespace total_earnings_l287_287578

theorem total_earnings : 
  let wage : ℕ := 10
  let hours_monday : ℕ := 7
  let tips_monday : ℕ := 18
  let hours_tuesday : ℕ := 5
  let tips_tuesday : ℕ := 12
  let hours_wednesday : ℕ := 7
  let tips_wednesday : ℕ := 20
  let total_hours : ℕ := hours_monday + hours_tuesday + hours_wednesday
  let earnings_from_wage : ℕ := total_hours * wage
  let total_tips : ℕ := tips_monday + tips_tuesday + tips_wednesday
  let total_earnings : ℕ := earnings_from_wage + total_tips
  total_earnings = 240 :=
by
  sorry

end total_earnings_l287_287578


namespace yellow_marbles_at_least_zero_l287_287155

noncomputable def total_marbles := 30
def blue_marbles (n : ℕ) := n / 3
def red_marbles (n : ℕ) := n / 3
def green_marbles := 10
def yellow_marbles (n : ℕ) := n - ((2 * n) / 3 + 10)

-- Conditions
axiom h1 : total_marbles % 3 = 0
axiom h2 : total_marbles = 30

-- Prove the smallest number of yellow marbles is 0
theorem yellow_marbles_at_least_zero : yellow_marbles total_marbles = 0 := by
  sorry

end yellow_marbles_at_least_zero_l287_287155


namespace merchant_markup_percentage_l287_287669

theorem merchant_markup_percentage
  (CP : ℕ) (discount_percent : ℚ) (profit_percent : ℚ)
  (mp : ℚ := CP + x)
  (sp : ℚ := (1 - discount_percent) * mp)
  (final_sp : ℚ := CP * (1 + profit_percent)) :
  discount_percent = 15 / 100 ∧ profit_percent = 19 / 100 ∧ CP = 100 → 
  sp = 85 + 0.85 * x → 
  final_sp = 119 →
  x = 40 :=
by 
  sorry

end merchant_markup_percentage_l287_287669


namespace sphere_volume_l287_287420

theorem sphere_volume (S : ℝ) (hS : S = 4 * π) : ∃ V : ℝ, V = (4 / 3) * π := 
by
  sorry

end sphere_volume_l287_287420


namespace product_of_ratios_l287_287328

theorem product_of_ratios 
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (hx1 : x1^3 - 3 * x1 * y1^2 = 2005)
  (hy1 : y1^3 - 3 * x1^2 * y1 = 2004)
  (hx2 : x2^3 - 3 * x2 * y2^2 = 2005)
  (hy2 : y2^3 - 3 * x2^2 * y2 = 2004)
  (hx3 : x3^3 - 3 * x3 * y3^2 = 2005)
  (hy3 : y3^3 - 3 * x3^2 * y3 = 2004) :
  (1 - x1/y1) * (1 - x2/y2) * (1 - x3/y3) = 1/1002 := 
sorry

end product_of_ratios_l287_287328


namespace length_of_the_bridge_l287_287799

theorem length_of_the_bridge
  (train_length : ℕ)
  (train_speed_kmh : ℕ)
  (cross_time_s : ℕ)
  (h_train_length : train_length = 120)
  (h_train_speed_kmh : train_speed_kmh = 45)
  (h_cross_time_s : cross_time_s = 30) :
  ∃ bridge_length : ℕ, bridge_length = 255 := 
by 
  sorry

end length_of_the_bridge_l287_287799


namespace expand_product_l287_287081

theorem expand_product (x : ℝ) : (x + 3) * (x + 9) = x^2 + 12 * x + 27 := 
by sorry

end expand_product_l287_287081


namespace tangent_curve_line_l287_287994

theorem tangent_curve_line {b : ℝ} :
  (∃ m n : ℝ, (n = m^4) ∧ (4m + b = n) ∧ (4 = 4 * m^3)) → b = -3 :=
by
  intros h,
  sorry

end tangent_curve_line_l287_287994


namespace race_time_diff_l287_287457

-- Define the speeds and race distance
def Malcolm_speed : ℕ := 5  -- in minutes per mile
def Joshua_speed : ℕ := 7   -- in minutes per mile
def Alice_speed : ℕ := 6    -- in minutes per mile
def race_distance : ℕ := 12 -- in miles

-- Calculate times
def Malcolm_time : ℕ := Malcolm_speed * race_distance
def Joshua_time : ℕ := Joshua_speed * race_distance
def Alice_time : ℕ := Alice_speed * race_distance

-- Lean 4 statement to prove the time differences
theorem race_time_diff :
  Joshua_time - Malcolm_time = 24 ∧ Alice_time - Malcolm_time = 12 := by
  sorry

end race_time_diff_l287_287457


namespace negative_values_count_l287_287996

theorem negative_values_count :
  let S := {n : ℕ | n^2 < 196 }
  in S.card = 13 :=
by
  sorry

end negative_values_count_l287_287996


namespace intersecting_lines_k_value_l287_287076

theorem intersecting_lines_k_value :
  ∃ k : ℚ, (∀ x y : ℚ, y = 3 * x + 12 ∧ y = -5 * x - 7 → y = 2 * x + k) → k = 77 / 8 :=
sorry

end intersecting_lines_k_value_l287_287076


namespace point_not_on_graph_l287_287028

theorem point_not_on_graph : ¬ ∃ (x y : ℝ), (y = (x - 1) / (x + 2)) ∧ (x = -2) ∧ (y = 3) :=
by
  sorry

end point_not_on_graph_l287_287028


namespace number_of_items_l287_287726

variable (s d : ℕ)
variable (total_money cost_sandwich cost_drink discount : ℝ)
variable (s_purchase_criterion : s > 5)
variable (total_money_value : total_money = 50.00)
variable (cost_sandwich_value : cost_sandwich = 6.00)
variable (cost_drink_value : cost_drink = 1.50)
variable (discount_value : discount = 5.00)

theorem number_of_items (h1 : total_money = 50.00)
(h2 : cost_sandwich = 6.00)
(h3 : cost_drink = 1.50)
(h4 : discount = 5.00)
(h5 : s > 5) :
  s + d = 9 :=
by
  sorry

end number_of_items_l287_287726


namespace greatest_divisor_arithmetic_sum_l287_287337

theorem greatest_divisor_arithmetic_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ d, d = 15 ∧ ∀ S : ℕ, S = 15 * x + 105 * c → d ∣ S :=
by 
  sorry

end greatest_divisor_arithmetic_sum_l287_287337


namespace prove_expression_l287_287703

-- Define the operation for real numbers
def op (a b c : ℝ) : ℝ := (a - b + c) ^ 2

-- Stating the theorem for the given expression
theorem prove_expression (x z : ℝ) :
  op ((x + z) ^ 2) ((z - x) ^ 2) ((x - z) ^ 2) = (x + z) ^ 4 := 
by  sorry

end prove_expression_l287_287703


namespace problem_statement_l287_287272

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x * (x + 4) else x * (x - 4)

theorem problem_statement (a : ℝ) (h : f a > f (8 - a)) : 4 < a :=
by sorry

end problem_statement_l287_287272


namespace solve_for_f_2012_l287_287848

noncomputable def f : ℝ → ℝ := sorry -- as the exact function definition isn't provided

variable (f : ℝ → ℝ)
variable (odd_f : ∀ x, f (-x) = -f x)
variable (functional_eqn : ∀ x, f (x + 2) = f x + f 2)
variable (f_one : f 1 = 2)

theorem solve_for_f_2012 : f 2012 = 4024 :=
sorry

end solve_for_f_2012_l287_287848


namespace rich_total_distance_l287_287760

-- Define the given conditions 
def distance_house_to_sidewalk := 20
def distance_down_road := 200
def total_distance_so_far := distance_house_to_sidewalk + distance_down_road
def distance_left_turn := 2 * total_distance_so_far
def distance_to_intersection := total_distance_so_far + distance_left_turn
def distance_half := distance_to_intersection / 2
def total_distance_one_way := distance_to_intersection + distance_half

-- Define the theorem to be proven 
theorem rich_total_distance : total_distance_one_way * 2 = 1980 :=
by 
  -- This line is to complete the 'prove' demand of the theorem
  sorry

end rich_total_distance_l287_287760


namespace find_a_l287_287908

open Function

noncomputable def slope_1 (a x_0 : ℝ) : ℝ :=
  (a * x_0 + a - 1) * Real.exp x_0

noncomputable def slope_2 (x_0 : ℝ) : ℝ :=
  (x_0 - 2) * Real.exp (-x_0)

theorem find_a (x_0 : ℝ) (a : ℝ)
  (h1 : x_0 ∈ Set.Icc 0 (3 / 2))
  (h2 : slope_1 a x_0 * slope_2 x_0 = -1) :
  1 ≤ a ∧ a ≤ 3 / 2 := sorry

end find_a_l287_287908


namespace probability_point_in_cube_l287_287393

noncomputable def volume_cube (s : ℝ) : ℝ := s ^ 3

noncomputable def radius_sphere (d : ℝ) : ℝ := d / 2

noncomputable def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem probability_point_in_cube :
  let s := 1 -- side length of the cube
  let v_cube := volume_cube s
  let d := Real.sqrt 3 -- diagonal of the cube
  let r := radius_sphere d
  let v_sphere := volume_sphere r
  v_cube / v_sphere = (2 * Real.sqrt 3) / (3 * Real.pi) :=
by
  sorry

end probability_point_in_cube_l287_287393


namespace train_speed_second_part_l287_287528

variables (x v : ℝ)

theorem train_speed_second_part
  (h1 : ∀ t1 : ℝ, t1 = x / 30)
  (h2 : ∀ t2 : ℝ, t2 = 2 * x / v)
  (h3 : ∀ t : ℝ, t = 3 * x / 22.5) :
  (x / 30) + (2 * x / v) = (3 * x / 22.5) → v = 20 :=
by
  intros h4
  sorry

end train_speed_second_part_l287_287528


namespace triangle_properties_l287_287425

variable (a b c A B C : ℝ)
variable (CD BD : ℝ)

-- triangle properties and given conditions
variable (b_squared_eq_ac : b ^ 2 = a * c)
variable (cos_A_minus_C : Real.cos (A - C) = Real.cos B + 1 / 2)

theorem triangle_properties :
  B = π / 3 ∧ 
  A = π / 3 ∧ 
  (CD = 6 → ∃ x, x > 0 ∧ x = 4 * Real.sqrt 3 + 6) ∧
  (BD = 6 → ∀ area, area ≠ 9 / 4) :=
  by
    sorry

end triangle_properties_l287_287425


namespace total_vacations_and_classes_l287_287112

def kelvin_classes := 90
def grant_vacations := 4 * kelvin_classes
def total := grant_vacations + kelvin_classes

theorem total_vacations_and_classes :
  total = 450 :=
by
  sorry

end total_vacations_and_classes_l287_287112


namespace casey_saves_money_l287_287238

def first_employee_hourly_wage : ℕ := 20
def second_employee_hourly_wage : ℕ := 22
def subsidy_per_hour : ℕ := 6
def weekly_work_hours : ℕ := 40

theorem casey_saves_money :
  let first_employee_weekly_cost := first_employee_hourly_wage * weekly_work_hours
  let second_employee_effective_hourly_wage := second_employee_hourly_wage - subsidy_per_hour
  let second_employee_weekly_cost := second_employee_effective_hourly_wage * weekly_work_hours
  let savings := first_employee_weekly_cost - second_employee_weekly_cost
  savings = 160 :=
by
  sorry

end casey_saves_money_l287_287238


namespace greatest_divisor_of_sum_of_arithmetic_sequence_l287_287362

theorem greatest_divisor_of_sum_of_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ k : ℕ, (sum (λ n, x + n * c) (range 15)) = 15 * k :=
by sorry

end greatest_divisor_of_sum_of_arithmetic_sequence_l287_287362


namespace correct_statement_about_CH3COOK_l287_287511

def molar_mass_CH3COOK : ℝ := 98  -- in g/mol

def avogadro_number : ℝ := 6.02 * 10^23  -- molecules per mole

def hydrogen_atoms_in_CH3COOK (mol_CH3COOK : ℝ) : ℝ :=
  3 * mol_CH3COOK * avogadro_number

theorem correct_statement_about_CH3COOK (mol_CH3COOK : ℝ) (h: mol_CH3COOK = 1) :
  hydrogen_atoms_in_CH3COOK mol_CH3COOK = 3 * avogadro_number :=
by
  sorry

end correct_statement_about_CH3COOK_l287_287511


namespace celine_library_charge_l287_287663

variable (charge_per_day : ℝ) (days_in_may : ℕ) (books_borrowed : ℕ) (days_first_book : ℕ)
          (days_other_books : ℕ) (books_kept : ℕ)

noncomputable def total_charge (charge_per_day : ℝ) (days_first_book : ℕ) 
        (days_other_books : ℕ) (books_kept : ℕ) : ℝ :=
  charge_per_day * days_first_book + charge_per_day * days_other_books * books_kept

theorem celine_library_charge : 
  charge_per_day = 0.50 ∧ days_in_may = 31 ∧ books_borrowed = 3 ∧ days_first_book = 20 ∧
  days_other_books = 31 ∧ books_kept = 2 → 
  total_charge charge_per_day days_first_book days_other_books books_kept = 41.00 :=
by
  intros h
  sorry

end celine_library_charge_l287_287663


namespace lemonade_problem_l287_287909

theorem lemonade_problem (L S W : ℕ) (h1 : W = 4 * S) (h2 : S = 2 * L) (h3 : L = 3) : L + S + W = 24 :=
by
  sorry

end lemonade_problem_l287_287909


namespace alpha_values_l287_287149

noncomputable def α := Complex

theorem alpha_values (α : Complex) :
  (α ≠ 1) ∧ 
  (Complex.abs (α^2 - 1) = 3 * Complex.abs (α - 1)) ∧ 
  (Complex.abs (α^4 - 1) = 5 * Complex.abs (α - 1)) ∧ 
  (Real.cos α.arg = 1 / 2) →
  α = Complex.mk ((-1 + Real.sqrt 33) / 4) (Real.sqrt (3 * (((-1 + Real.sqrt 33) / 4)^2))) ∨ 
  α = Complex.mk ((-1 - Real.sqrt 33) / 4) (Real.sqrt (3 * (((-1 - Real.sqrt 33) / 4)^2))) :=
sorry

end alpha_values_l287_287149


namespace solve_for_x_l287_287597

-- Definitions for the problem conditions
def perimeter_triangle := 14 + 12 + 12
def perimeter_rectangle (x : ℝ) := 2 * x + 16

-- Lean 4 statement for the proof problem 
theorem solve_for_x (x : ℝ) : 
  perimeter_triangle = perimeter_rectangle x → 
  x = 11 := 
by 
  -- standard placeholders
  sorry

end solve_for_x_l287_287597


namespace isosceles_triangle_range_of_expression_l287_287883

open Real

-- Given: In triangle ABC, sides opposite to angles A, B, C are a, b, c respectively
-- and satisfy a * cos B = b * cos A.
-- Prove:
-- 1. The triangle is isosceles (A = B).
-- 2. The range of sin (2A + π/6) - 2 * cos^2 B is (-3/2, 0).

theorem isosceles_triangle (A B C : ℝ) (a b c : ℝ) (h₁ : a * cos B = b * cos A) :
  A = B :=
sorry

theorem range_of_expression (A : ℝ) (hA : 0 < A ∧ A < π / 2) (h_isosceles : A = B) :
  -3/2 < sin (2 * A + π / 6) - 2 * cos^2 B ∧ sin (2 * A + π / 6) - 2 * cos^2 B < 0 :=
sorry

end isosceles_triangle_range_of_expression_l287_287883


namespace M1_M2_product_l287_287450

theorem M1_M2_product (M_1 M_2 : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 →
  (42 * x - 51) / (x^2 - 5 * x + 6) = (M_1 / (x - 2)) + (M_2 / (x - 3))) →
  M_1 * M_2 = -2981.25 :=
by
  intros h
  sorry

end M1_M2_product_l287_287450


namespace alyssa_games_last_year_l287_287824

theorem alyssa_games_last_year (games_this_year games_next_year games_total games_last_year : ℕ) (h1 : games_this_year = 11) (h2 : games_next_year = 15) (h3 : games_total = 39) (h4 : games_last_year + games_this_year + games_next_year = games_total) : games_last_year = 13 :=
by
  rw [h1, h2, h3] at h4
  sorry

end alyssa_games_last_year_l287_287824


namespace seth_pounds_lost_l287_287164

-- Definitions
def pounds_lost_by_Seth (S : ℝ) : Prop := 
  let total_loss := S + 3 * S + (S + 1.5)
  total_loss = 89

theorem seth_pounds_lost (S : ℝ) : pounds_lost_by_Seth S → S = 17.5 := by
  sorry

end seth_pounds_lost_l287_287164


namespace non_congruent_triangles_with_perimeter_11_l287_287866

theorem non_congruent_triangles_with_perimeter_11 : 
  ∀ (a b c : ℕ), a + b + c = 11 → a < b + c → b < a + c → c < a + b → 
  ∃! (a b c : ℕ), (a, b, c) = (2, 4, 5) ∨ (a, b, c) = (3, 4, 4) := 
by sorry

end non_congruent_triangles_with_perimeter_11_l287_287866


namespace max_value_of_quadratic_l287_287709

theorem max_value_of_quadratic (x : ℝ) (h : 0 < x ∧ x < 6) : (6 - x) * x ≤ 9 := 
by
  sorry

end max_value_of_quadratic_l287_287709


namespace problem_l287_287736

theorem problem (a b : ℕ) (ha : 2^a ∣ 180) (h2 : ∀ n, 2^n ∣ 180 → n ≤ a) (hb : 5^b ∣ 180) (h5 : ∀ n, 5^n ∣ 180 → n ≤ b) : (1 / 3) ^ (b - a) = 3 := by
  sorry

end problem_l287_287736


namespace find_S6_l287_287712

variable (a : ℕ → ℝ) (S_n : ℕ → ℝ)

-- The sequence {a_n} is given as a geometric sequence
-- Partial sums are given as S_2 = 1 and S_4 = 3

-- Conditions
axiom geom_sequence : ∀ n : ℕ, a (n + 1) / a n = a 1 / a 0
axiom S2 : S_n 2 = 1
axiom S4 : S_n 4 = 3

-- Theorem statement
theorem find_S6 : S_n 6 = 7 :=
sorry

end find_S6_l287_287712


namespace city_rentals_cost_per_mile_l287_287615

-- The parameters provided in the problem
def safety_base_rate : ℝ := 21.95
def safety_per_mile_rate : ℝ := 0.19
def city_base_rate : ℝ := 18.95
def miles_driven : ℝ := 150.0

-- The cost expressions based on the conditions
def safety_total_cost (miles: ℝ) : ℝ := safety_base_rate + safety_per_mile_rate * miles
def city_total_cost (miles: ℝ) (city_per_mile_rate: ℝ) : ℝ := city_base_rate + city_per_mile_rate * miles

-- The cost equality condition for 150 miles
def cost_condition : Prop :=
  safety_total_cost miles_driven = city_total_cost miles_driven 0.21

-- Prove that the cost per mile for City Rentals is 0.21 dollars
theorem city_rentals_cost_per_mile : cost_condition :=
by
  -- Start the proof
  sorry

end city_rentals_cost_per_mile_l287_287615


namespace arithmetic_sequence_30th_term_l287_287073

theorem arithmetic_sequence_30th_term :
  let a₁ := 4
  let d₁ := 6
  let n := 30
  (a₁ + (n - 1) * d₁) = 178 :=
by
  sorry

end arithmetic_sequence_30th_term_l287_287073


namespace probability_X_l287_287611

theorem probability_X (P : ℕ → ℚ) (h1 : P 1 = 1/10) (h2 : P 2 = 2/10) (h3 : P 3 = 3/10) (h4 : P 4 = 4/10) :
  P 2 + P 3 = 1/2 :=
by
  sorry

end probability_X_l287_287611


namespace simplify_expr_at_sqrt6_l287_287166

noncomputable def simplifyExpression (x : ℝ) : ℝ :=
  (1 / (Real.sqrt (3 + x) * Real.sqrt (x + 2)) + 1 / (Real.sqrt (3 - x) * Real.sqrt (x - 2))) /
  (1 / (Real.sqrt (3 + x) * Real.sqrt (x + 2)) - 1 / (Real.sqrt (3 - x) * Real.sqrt (x - 2)))

theorem simplify_expr_at_sqrt6 : simplifyExpression (Real.sqrt 6) = - (Real.sqrt 6) / 2 :=
by
  sorry

end simplify_expr_at_sqrt6_l287_287166


namespace y_paid_per_week_l287_287642

variable (x y z : ℝ)

-- Conditions
axiom h1 : x + y + z = 900
axiom h2 : x = 1.2 * y
axiom h3 : z = 0.8 * y

-- Theorem to prove
theorem y_paid_per_week : y = 300 := by
  sorry

end y_paid_per_week_l287_287642


namespace boy_usual_time_reach_school_l287_287646

theorem boy_usual_time_reach_school (R T : ℝ) (h : (7 / 6) * R * (T - 3) = R * T) : T = 21 := by
  sorry

end boy_usual_time_reach_school_l287_287646


namespace conditions_for_inequality_l287_287781

theorem conditions_for_inequality (a b : ℝ) :
  (∀ x : ℝ, abs ((x^2 + a * x + b) / (x^2 + 2 * x + 2)) < 1) → 
  (a = 2 ∧ 0 < b ∧ b < 2) :=
sorry

end conditions_for_inequality_l287_287781


namespace factor_1024_into_three_factors_l287_287890

theorem factor_1024_into_three_factors :
  ∃ (factors : Finset (Finset ℕ)), factors.card = 14 ∧
  ∀ f ∈ factors, ∃ a b c : ℕ, a + b + c = 10 ∧ a ≥ b ∧ b ≥ c ∧ (2 ^ a) * (2 ^ b) * (2 ^ c) = 1024 :=
sorry

end factor_1024_into_three_factors_l287_287890


namespace lumberjack_trees_chopped_l287_287666

-- Statement of the problem in Lean 4
theorem lumberjack_trees_chopped
  (logs_per_tree : ℕ) 
  (firewood_per_log : ℕ) 
  (total_firewood : ℕ) 
  (logs_per_tree_eq : logs_per_tree = 4) 
  (firewood_per_log_eq : firewood_per_log = 5) 
  (total_firewood_eq : total_firewood = 500)
  : (total_firewood / firewood_per_log) / logs_per_tree = 25 := 
by
  rw [total_firewood_eq, firewood_per_log_eq, logs_per_tree_eq]
  norm_num
  sorry

end lumberjack_trees_chopped_l287_287666


namespace evaluate_product_roots_of_unity_l287_287403

theorem evaluate_product_roots_of_unity :
  let w := Complex.exp (2 * Real.pi * Complex.I / 13)
  (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) *
  (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * (3 - w^11) * (3 - w^12) =
  (3^12 + 3^11 + 3^10 + 3^9 + 3^8 + 3^7 + 3^6 + 3^5 + 3^4 + 3^3 + 3^2 + 3 + 1) :=
by
  sorry

end evaluate_product_roots_of_unity_l287_287403


namespace sum_of_real_roots_eq_three_l287_287510

theorem sum_of_real_roots_eq_three : 
  (∑ root in (Polynomial.roots (Polynomial.mk [1, -4, 5, -4, 1])).toFinset, root.re) = 3 := 
sorry

end sum_of_real_roots_eq_three_l287_287510


namespace jelly_bean_probability_l287_287391

variable (P_red P_orange P_yellow P_green : ℝ)

theorem jelly_bean_probability :
  P_red = 0.15 ∧ P_orange = 0.35 ∧ (P_red + P_orange + P_yellow + P_green = 1) →
  (P_yellow + P_green = 0.5) :=
by
  intro h
  obtain ⟨h_red, h_orange, h_total⟩ := h
  sorry

end jelly_bean_probability_l287_287391


namespace no_such_integers_exist_l287_287161

theorem no_such_integers_exist :
  ¬(∃ (a b c d : ℤ), a * 19^3 + b * 19^2 + c * 19 + d = 1 ∧ a * 62^3 + b * 62^2 + c * 62 + d = 2) :=
by
  sorry

end no_such_integers_exist_l287_287161


namespace simplify_sqrt_90000_l287_287165

theorem simplify_sqrt_90000 : Real.sqrt 90000 = 300 :=
by
  /- Proof goes here -/
  sorry

end simplify_sqrt_90000_l287_287165


namespace no_four_nat_satisfy_l287_287758

theorem no_four_nat_satisfy:
  ∀ (x y z t : ℕ), 3 * x^4 + 5 * y^4 + 7 * z^4 ≠ 11 * t^4 :=
by
  sorry

end no_four_nat_satisfy_l287_287758


namespace students_passed_both_tests_l287_287207

theorem students_passed_both_tests :
  ∀ (total students_passed_long_jump students_passed_shot_put students_failed_both x : ℕ),
    total = 50 →
    students_passed_long_jump = 40 →
    students_passed_shot_put = 31 →
    students_failed_both = 4 →
    (students_passed_long_jump - x) + (students_passed_shot_put - x) + x + students_failed_both = total →
    x = 25 :=
by intros total students_passed_long_jump students_passed_shot_put students_failed_both x
   intro total_eq students_passed_long_jump_eq students_passed_shot_put_eq students_failed_both_eq sum_eq
   sorry

end students_passed_both_tests_l287_287207


namespace m_value_l287_287721

theorem m_value (A : Set ℝ) (B : Set ℝ) (m : ℝ) 
                (hA : A = {0, 1, 2}) 
                (hB : B = {1, m}) 
                (h_subset : B ⊆ A) : 
                m = 0 ∨ m = 2 :=
by
  sorry

end m_value_l287_287721


namespace eq_is_quadratic_iff_m_zero_l287_287119

theorem eq_is_quadratic_iff_m_zero (m : ℝ) : (|m| + 2 = 2 ∧ m - 3 ≠ 0) ↔ m = 0 := by
  sorry

end eq_is_quadratic_iff_m_zero_l287_287119


namespace parabola_chord_midpoint_l287_287121

/-- 
If the point (3, 1) is the midpoint of a chord of the parabola y^2 = 2px, 
and the slope of the line containing this chord is 2, then p = 2. 
-/
theorem parabola_chord_midpoint (p : ℝ) :
    (∃ (m : ℝ), (m = 2) ∧ ∀ (x y : ℝ), y = 2 * x - 5 → y^2 = 2 * p * x → 
        ((x1 = 0 ∧ y1 = 0 ∧ x2 = 6 ∧ y2 = 6) → 
            (x1 + x2 = 6) ∧ (y1 + y2 = 2) ∧ (p = 2))) :=
sorry

end parabola_chord_midpoint_l287_287121


namespace value_of_k_l287_287275

theorem value_of_k :
  (∀ x : ℝ, x ^ 2 - x - 2 > 0 → 2 * x ^ 2 + (5 + 2 * k) * x + 5 * k < 0 → x = -2) ↔ -3 ≤ k ∧ k < 2 :=
sorry

end value_of_k_l287_287275


namespace A_wins_when_n_is_9_l287_287777

-- Definition of the game conditions and the strategy
def game (n : ℕ) (A_first : Bool) :=
  ∃ strategy : ℕ → ℕ,
    ∀ taken balls_left : ℕ,
      balls_left - taken > 0 →
      taken ≥ 1 → taken ≤ 3 →
      if A_first then
        (balls_left - taken = 0 → strategy (balls_left - taken) = 1) ∧
        (∀ t : ℕ, t >= 1 ∧ t ≤ 3 → strategy t = balls_left - taken - t)
      else
        (balls_left - taken = 0 → strategy (balls_left - taken) = 0) ∨
        (∀ t : ℕ, t >= 1 ∧ t ≤ 3 → strategy t = balls_left - taken - t)

-- Prove that for n = 9 A has a winning strategy
theorem A_wins_when_n_is_9 : game 9 true :=
sorry

end A_wins_when_n_is_9_l287_287777


namespace cubed_identity_l287_287278

variable (x : ℝ)

theorem cubed_identity (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := 
by
  sorry

end cubed_identity_l287_287278


namespace sum_of_digits_eq_4_l287_287133

def sum_digits (n : Nat) : Nat :=
  n.digits 10 |> List.sum

def first_year_after (y : Nat) (p : Nat -> Prop) : Nat :=
  (Nat.iterate (· + 1) (1 + y) (fun n => p n) y)

theorem sum_of_digits_eq_4 : first_year_after 2020 (fun n => sum_digits n = 4) = 2030 :=
  sorry

end sum_of_digits_eq_4_l287_287133


namespace part1_max_area_part2_find_a_l287_287720

-- Part (1): Define the function and prove maximum area of the triangle
noncomputable def f (a x : ℝ) : ℝ := a^2 * Real.exp x - 3 * a * x + 2 * Real.sin x - 1

theorem part1_max_area (a : ℝ) (h : 0 < a ∧ a < 1) : 
  let f' := a^2 - 3 * a + 2
  ∃ h_a_max, h_a_max == 3 / 8 :=
  sorry

-- Part (2): Prove that the function reaches an extremum at x = 0 and determine the value of a.
theorem part2_find_a (a : ℝ) : (a^2 - 3 * a + 2 = 0) → (a = 1 ∨ a = 2) :=
  sorry

end part1_max_area_part2_find_a_l287_287720


namespace greatest_divisor_arithmetic_sequence_sum_l287_287346

theorem greatest_divisor_arithmetic_sequence_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ k, (15 * (x + 7 * c)) = 15 * k :=
sorry

end greatest_divisor_arithmetic_sequence_sum_l287_287346


namespace banana_orange_equivalence_l287_287468

/-- Given that 3/4 of 12 bananas are worth 9 oranges,
    prove that 1/3 of 9 bananas are worth 3 oranges. -/
theorem banana_orange_equivalence :
  (3 / 4) * 12 = 9 → (1 / 3) * 9 = 3 :=
by
  intro h
  have h1 : (9 : ℝ) = 9 := by sorry -- This is from the provided condition
  have h2 : 1 * 9 = 1 * 9 := by sorry -- Deducing from h1: 9 = 9
  have h3 : 9 = 9 := by sorry -- concluding 9 bananas = 9 oranges
  have h4 : (1 / 3) * 9 = 3 := by sorry -- 1/3 of 9
  exact h4

end banana_orange_equivalence_l287_287468


namespace vector_relation_AD_l287_287737

variables {P V : Type} [AddCommGroup V] [Module ℝ V]
variables (A B C D : P) (AB AC AD BC BD CD : V)
variables (hBC_CD : BC = 3 • CD)

theorem vector_relation_AD (h1 : BC = 3 • CD)
                           (h2 : AD = AB + BD)
                           (h3 : BD = BC + CD)
                           (h4 : BC = -AB + AC) :
  AD = - (1 / 3 : ℝ) • AB + (4 / 3 : ℝ) • AC :=
by
  sorry

end vector_relation_AD_l287_287737


namespace longest_side_length_quadrilateral_l287_287321

theorem longest_side_length_quadrilateral : 
  (∃ (x y : ℝ), x + y ≤ 4 ∧ 2 * x + y ≥ 1 ∧ x ≥ 0 ∧ y ≥ 0)
  → ∃ a b c d: ℝ, 
    (a, b), (c,d) in {(x,y) | (x + y ≤ 4 ∧ 2 * x + y ≥ 1 ∧ x ≥ 0 ∧ y ≥ 0)}
    ∧ dist (a, b) (c, d) = 7 * sqrt 2 / 2 :=
sorry

end longest_side_length_quadrilateral_l287_287321


namespace sacks_required_in_4_weeks_l287_287675

-- Definitions for the weekly requirements of each bakery
def weekly_sacks_bakery1 : Nat := 2
def weekly_sacks_bakery2 : Nat := 4
def weekly_sacks_bakery3 : Nat := 12

-- Total weeks considered
def weeks : Nat := 4

-- Calculating the total sacks needed for all bakeries over the given weeks
def total_sacks_needed : Nat :=
  (weekly_sacks_bakery1 * weeks) +
  (weekly_sacks_bakery2 * weeks) +
  (weekly_sacks_bakery3 * weeks)

-- The theorem to be proven
theorem sacks_required_in_4_weeks :
  total_sacks_needed = 72 :=
by
  sorry

end sacks_required_in_4_weeks_l287_287675


namespace find_x_pow_3a_minus_b_l287_287842

variable (x : ℝ) (a b : ℝ)
theorem find_x_pow_3a_minus_b (h1 : x^a = 2) (h2 : x^b = 9) : x^(3 * a - b) = 8 / 9 :=
  sorry

end find_x_pow_3a_minus_b_l287_287842


namespace exists_sum_of_digits_div_11_l287_287755

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_sum_of_digits_div_11 (H : Finset ℕ) (h₁ : H.card = 39) :
  ∃ (a : ℕ) (h : a ∈ H), sum_of_digits a % 11 = 0 :=
by
  sorry

end exists_sum_of_digits_div_11_l287_287755


namespace greatest_divisor_of_sum_of_arith_seq_l287_287353

theorem greatest_divisor_of_sum_of_arith_seq (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → d ∣ (15 * (x + 7 * c))) ∧
    (∀ k : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → k ∣ (15 * (x + 7 * c))) → k ≤ d) ∧ 
    d = 15 :=
sorry

end greatest_divisor_of_sum_of_arith_seq_l287_287353


namespace exists_common_element_l287_287151

variable (S : Fin 2011 → Set ℤ)
variable (h1 : ∀ i, (S i).Nonempty)
variable (h2 : ∀ i j, (S i ∩ S j).Nonempty)

theorem exists_common_element :
  ∃ a : ℤ, ∀ i, a ∈ S i :=
by {
  sorry
}

end exists_common_element_l287_287151


namespace casey_savings_l287_287234

-- Define the constants given in the problem conditions
def wage_employee_1 : ℝ := 20
def wage_employee_2 : ℝ := 22
def subsidy : ℝ := 6
def hours_per_week : ℝ := 40

-- Define the weekly cost of each employee
def weekly_cost_employee_1 := wage_employee_1 * hours_per_week
def weekly_cost_employee_2 := (wage_employee_2 - subsidy) * hours_per_week

-- Define the savings by hiring the cheaper employee
def savings := weekly_cost_employee_1 - weekly_cost_employee_2

-- Theorem stating the expected savings
theorem casey_savings : savings = 160 := by
  -- Proof is not included
  sorry

end casey_savings_l287_287234


namespace angle_conversion_l287_287836

/--
 Given an angle in degrees, express it in degrees, minutes, and seconds.
 Theorem: 20.23 degrees can be converted to 20 degrees, 13 minutes, and 48 seconds.
-/
theorem angle_conversion : (20.23:ℝ) = 20 + (13/60 : ℝ) + (48/3600 : ℝ) :=
by
  sorry

end angle_conversion_l287_287836


namespace num_students_in_section_A_l287_287930

def avg_weight (total_weight : ℕ) (total_students : ℕ) : ℕ :=
  total_weight / total_students

variables (x : ℕ) -- number of students in section A
variables (weight_A : ℕ := 40 * x) -- total weight of section A
variables (students_B : ℕ := 20)
variables (weight_B : ℕ := 20 * 35) -- total weight of section B
variables (total_weight : ℕ := weight_A + weight_B) -- total weight of the whole class
variables (total_students : ℕ := x + students_B) -- total number of students in the class
variables (avg_weight_class : ℕ := avg_weight total_weight total_students)

theorem num_students_in_section_A :
  avg_weight_class = 38 → x = 30 :=
by
-- The proof will go here
sorry

end num_students_in_section_A_l287_287930


namespace slope_of_line_l287_287993

theorem slope_of_line : 
  let A := Real.sin (Real.pi / 6)
  let B := Real.cos (5 * Real.pi / 6)
  (- A / B) = Real.sqrt 3 / 3 :=
by
  sorry

end slope_of_line_l287_287993


namespace expression_incorrect_l287_287316

theorem expression_incorrect (x : ℝ) : 5 * (x + 7) ≠ 5 * x + 7 := 
by 
  sorry

end expression_incorrect_l287_287316


namespace parabola_no_intersection_inequality_l287_287153

-- Definitions for the problem
theorem parabola_no_intersection_inequality
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∀ x : ℝ, (a * x^2 + b * x + c ≠ x) ∧ (a * x^2 + b * x + c ≠ -x)) :
  |b^2 - 4 * a * c| > 1 := 
sorry

end parabola_no_intersection_inequality_l287_287153


namespace smallest_prime_after_six_consecutive_nonprimes_l287_287794

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def nonprimes_between (a b : ℕ) : Prop :=
  ∀ n : ℕ, a < n ∧ n < b → ¬ is_prime n

def find_first_nonprime_gap : ℕ :=
  if H : ∃ a b : ℕ, b = a + 7 ∧ nonprimes_between a b ∧ is_prime (b + 1) then
    Classical.choose H
  else
    0

theorem smallest_prime_after_six_consecutive_nonprimes : find_first_nonprime_gap = 97 := by
  sorry

end smallest_prime_after_six_consecutive_nonprimes_l287_287794


namespace number_of_C_atoms_in_compound_is_4_l287_287213

def atomic_weight_C : ℕ := 12
def atomic_weight_H : ℕ := 1
def atomic_weight_O : ℕ := 16

def molecular_weight : ℕ := 65

def weight_contributed_by_H_O : ℕ := atomic_weight_H + atomic_weight_O -- 17 amu

def weight_contributed_by_C : ℕ := molecular_weight - weight_contributed_by_H_O -- 48 amu

def number_of_C_atoms := weight_contributed_by_C / atomic_weight_C -- The quotient of 48 amu divided by 12 amu per C atom

theorem number_of_C_atoms_in_compound_is_4 : number_of_C_atoms = 4 :=
by
  sorry -- This is where the proof would go, but it's omitted as per instructions.

end number_of_C_atoms_in_compound_is_4_l287_287213


namespace triangle_base_value_l287_287638

variable (L R B : ℕ)

theorem triangle_base_value
    (h1 : L = 12)
    (h2 : R = L + 2)
    (h3 : L + R + B = 50) :
    B = 24 := 
sorry

end triangle_base_value_l287_287638


namespace find_f_of_1_over_2016_l287_287046

noncomputable def f (x : ℝ) : ℝ := sorry

lemma f_property_0 : f 0 = 0 := sorry
lemma f_property_1 (x : ℝ) : f x + f (1 - x) = 1 := sorry
lemma f_property_2 (x : ℝ) : f (x / 3) = (1 / 2) * f x := sorry
lemma f_property_3 {x₁ x₂ : ℝ} (h₀ : 0 ≤ x₁) (h₁ : x₁ < x₂) (h₂ : x₂ ≤ 1): f x₁ ≤ f x₂ := sorry

theorem find_f_of_1_over_2016 : f (1 / 2016) = 1 / 128 := sorry

end find_f_of_1_over_2016_l287_287046


namespace product_of_variables_l287_287202

theorem product_of_variables (a b c d : ℚ)
  (h1 : 4 * a + 5 * b + 7 * c + 9 * d = 56)
  (h2 : 4 * (d + c) = b)
  (h3 : 4 * b + 2 * c = a)
  (h4 : c - 2 = d) :
  a * b * c * d = 58653 / 10716361 := 
sorry

end product_of_variables_l287_287202


namespace Julia_played_with_11_kids_on_Monday_l287_287148

theorem Julia_played_with_11_kids_on_Monday
  (kids_on_Tuesday : ℕ)
  (kids_on_Monday : ℕ) 
  (h1 : kids_on_Tuesday = 12)
  (h2 : kids_on_Tuesday = kids_on_Monday + 1) : 
  kids_on_Monday = 11 := 
by
  sorry

end Julia_played_with_11_kids_on_Monday_l287_287148


namespace cubed_identity_l287_287281

theorem cubed_identity (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := 
sorry

end cubed_identity_l287_287281


namespace determinant_of_matrix4x5_2x3_l287_287685

def matrix4x5_2x3 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![4, 5], ![2, 3]]

theorem determinant_of_matrix4x5_2x3 : matrix4x5_2x3.det = 2 := 
by
  sorry

end determinant_of_matrix4x5_2x3_l287_287685


namespace square_of_second_arm_l287_287924

theorem square_of_second_arm (a b c : ℝ) (h₁ : c = a + 2) (h₂ : a^2 + b^2 = c^2) : b^2 = 4 * a + 4 :=
sorry

end square_of_second_arm_l287_287924


namespace celine_library_charge_l287_287662

variable (charge_per_day : ℝ) (days_in_may : ℕ) (books_borrowed : ℕ) (days_first_book : ℕ)
          (days_other_books : ℕ) (books_kept : ℕ)

noncomputable def total_charge (charge_per_day : ℝ) (days_first_book : ℕ) 
        (days_other_books : ℕ) (books_kept : ℕ) : ℝ :=
  charge_per_day * days_first_book + charge_per_day * days_other_books * books_kept

theorem celine_library_charge : 
  charge_per_day = 0.50 ∧ days_in_may = 31 ∧ books_borrowed = 3 ∧ days_first_book = 20 ∧
  days_other_books = 31 ∧ books_kept = 2 → 
  total_charge charge_per_day days_first_book days_other_books books_kept = 41.00 :=
by
  intros h
  sorry

end celine_library_charge_l287_287662


namespace same_cost_number_of_guests_l287_287467

theorem same_cost_number_of_guests (x : ℕ) : 
  (800 + 30 * x = 500 + 35 * x) ↔ (x = 60) :=
by {
  sorry
}

end same_cost_number_of_guests_l287_287467


namespace arcsin_sqrt3_div_2_eq_pi_div_3_l287_287549

theorem arcsin_sqrt3_div_2_eq_pi_div_3 : Real.arcsin (Real.sqrt 3 / 2) = Real.pi / 3 :=
by
  sorry

end arcsin_sqrt3_div_2_eq_pi_div_3_l287_287549


namespace factorization_count_l287_287892

noncomputable def count_factors (n : ℕ) (a b c : ℕ) : ℕ :=
if 2 ^ a * 2 ^ b * 2 ^ c = n ∧ a + b + c = 10 ∧ a ≥ b ∧ b ≥ c then 1 else 0

noncomputable def total_factorizations : ℕ :=
Finset.sum (Finset.range 11) (fun c => 
  Finset.sum (Finset.Icc c 10) (fun b => 
    Finset.sum (Finset.Icc b 10) (fun a =>
      count_factors 1024 a b c)))

theorem factorization_count : total_factorizations = 14 :=
sorry

end factorization_count_l287_287892


namespace rich_walked_distance_l287_287763

def total_distance_walked (d1 d2 : ℕ) := 
  d1 + d2 + 2 * (d1 + d2) + (d1 + d2 + 2 * (d1 + d2)) / 2

def distance_to_intersection (d1 d2 : ℕ) := 
  2 * (d1 + d2)

def distance_to_end_route (d1 d2 : ℕ) := 
  (d1 + d2 + distance_to_intersection d1 d2) / 2

def total_distance_one_way (d1 d2 : ℕ) := 
  (d1 + d2) + (distance_to_intersection d1 d2) + (distance_to_end_route d1 d2)

theorem rich_walked_distance
  (d1 : ℕ := 20)
  (d2 : ℕ := 200) :
  2 * total_distance_one_way d1 d2 = 1980 :=
by
  simp [total_distance_one_way, distance_to_intersection, distance_to_end_route, total_distance_walked]
  sorry

end rich_walked_distance_l287_287763


namespace convert_radian_to_degree_part1_convert_radian_to_degree_part2_convert_radian_to_degree_part3_convert_degree_to_radian_part1_convert_degree_to_radian_part2_l287_287553

noncomputable def pi_deg : ℝ := 180 -- Define pi in degrees
notation "°" => pi_deg -- Define a notation for degrees

theorem convert_radian_to_degree_part1 : (π / 12) * (180 / π) = 15 := 
by
  sorry

theorem convert_radian_to_degree_part2 : (13 * π / 6) * (180 / π) = 390 := 
by
  sorry

theorem convert_radian_to_degree_part3 : -(5 / 12) * π * (180 / π) = -75 := 
by
  sorry

theorem convert_degree_to_radian_part1 : 36 * (π / 180) = (π / 5) := 
by
  sorry

theorem convert_degree_to_radian_part2 : -105 * (π / 180) = -(7 * π / 12) := 
by
  sorry

end convert_radian_to_degree_part1_convert_radian_to_degree_part2_convert_radian_to_degree_part3_convert_degree_to_radian_part1_convert_degree_to_radian_part2_l287_287553


namespace equal_costs_at_60_guests_l287_287466

def caesars_cost (x : ℕ) : ℕ := 800 + 30 * x
def venus_cost (x : ℕ) : ℕ := 500 + 35 * x

theorem equal_costs_at_60_guests : 
  ∃ x : ℕ, caesars_cost x = venus_cost x ∧ x = 60 := 
by
  existsi 60
  unfold caesars_cost venus_cost
  split
  . sorry
  . refl

end equal_costs_at_60_guests_l287_287466


namespace rachel_homework_difference_l287_287162

def pages_of_math_homework : Nat := 5
def pages_of_reading_homework : Nat := 2

theorem rachel_homework_difference : 
  pages_of_math_homework - pages_of_reading_homework = 3 :=
sorry

end rachel_homework_difference_l287_287162


namespace average_cookies_l287_287543

theorem average_cookies (cookie_counts : List ℕ) (h : cookie_counts = [8, 10, 12, 15, 16, 17, 20]) :
  (cookie_counts.sum : ℚ) / cookie_counts.length = 14 := by
    -- Proof goes here
  sorry

end average_cookies_l287_287543


namespace hypotenuse_of_right_triangle_l287_287820

theorem hypotenuse_of_right_triangle (a b : ℕ) (h : ℕ)
  (h1 : a = 15) (h2 : b = 36) (right_triangle : a^2 + b^2 = h^2) : h = 39 :=
by
  sorry

end hypotenuse_of_right_triangle_l287_287820


namespace integer_solutions_count_l287_287487

theorem integer_solutions_count :
  (finset.filter (λ (x : ℤ), 5 < real.sqrt (x : ℝ) ∧ real.sqrt (x : ℝ) < 6) 
  (finset.Icc 26 35)).card = 10 :=
by
  sorry

end integer_solutions_count_l287_287487


namespace minimum_value_condition_l287_287713

theorem minimum_value_condition (m n : ℝ) (hm : m > 0) (hn : n > 0) 
                                (h_line : ∀ x y : ℝ, m * x + n * y + 2 = 0 → (x + 3)^2 + (y + 1)^2 = 1) 
                                (h_chord : ∀ x1 y1 x2 y2 : ℝ, m * x1 + n * y1 + 2 = 0 ∧ (x1 + 3)^2 + (y1 + 1)^2 = 1 ∧
                                           m * x2 + n * y2 + 2 = 0 ∧ (x2 + 3)^2 + (y2 + 1)^2 = 1 ∧
                                           (x1 - x2)^2 + (y1 - y2)^2 = 4) 
                                (h_relation : 3 * m + n = 2) : 
    ∃ (C : ℝ), C = 6 ∧ (C = (1 / m + 3 / n)) := 
by
  sorry

end minimum_value_condition_l287_287713


namespace athenas_min_wins_l287_287169

theorem athenas_min_wins (total_games : ℕ) (games_played : ℕ) (wins_so_far : ℕ) (losses_so_far : ℕ) 
                          (win_percentage_threshold : ℝ) (remaining_games : ℕ) (additional_wins_needed : ℕ) :
  total_games = 44 ∧ games_played = wins_so_far + losses_so_far ∧ wins_so_far = 20 ∧ losses_so_far = 15 ∧ 
  win_percentage_threshold = 0.6 ∧ remaining_games = total_games - games_played ∧ additional_wins_needed = 27 - wins_so_far → 
  additional_wins_needed = 7 :=
by
  sorry

end athenas_min_wins_l287_287169


namespace inequality_solution_sets_l287_287263

theorem inequality_solution_sets (a : ℝ) (h : a > 1) :
  ∀ x : ℝ, ((a = 2 → (x ≠ 1 → (a-1)*x*x - a*x + 1 > 0)) ∧
            (1 < a ∧ a < 2 → (x < 1 ∨ x > 1/(a-1) → (a-1)*x*x - a*x + 1 > 0)) ∧
            (a > 2 → (x < 1/(a-1) ∨ x > 1 → (a-1)*x*x - a*x + 1 > 0))) :=
by
  sorry

end inequality_solution_sets_l287_287263


namespace range_of_b_min_value_a_add_b_min_value_ab_l287_287567

theorem range_of_b (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) : b > 1 := sorry

theorem min_value_a_add_b (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) : a + b ≥ 8 := sorry

theorem min_value_ab (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) : a * b ≥ 16 := sorry

end range_of_b_min_value_a_add_b_min_value_ab_l287_287567


namespace new_determinant_l287_287552

-- Given the condition that the determinant of the original matrix is 12
def original_determinant (x y z w : ℝ) : Prop :=
  x * w - y * z = 12

-- Proof that the determinant of the new matrix equals the expected result
theorem new_determinant (x y z w : ℝ) (h : original_determinant x y z w) :
  (2 * x + z) * w - (2 * y - w) * z = 24 + z * w + w * z := by
  sorry

end new_determinant_l287_287552


namespace casey_saves_money_l287_287239

def first_employee_hourly_wage : ℕ := 20
def second_employee_hourly_wage : ℕ := 22
def subsidy_per_hour : ℕ := 6
def weekly_work_hours : ℕ := 40

theorem casey_saves_money :
  let first_employee_weekly_cost := first_employee_hourly_wage * weekly_work_hours
  let second_employee_effective_hourly_wage := second_employee_hourly_wage - subsidy_per_hour
  let second_employee_weekly_cost := second_employee_effective_hourly_wage * weekly_work_hours
  let savings := first_employee_weekly_cost - second_employee_weekly_cost
  savings = 160 :=
by
  sorry

end casey_saves_money_l287_287239


namespace integer_solutions_count_l287_287488

theorem integer_solutions_count :
  (finset.filter (λ (x : ℤ), 5 < real.sqrt (x : ℝ) ∧ real.sqrt (x : ℝ) < 6) 
  (finset.Icc 26 35)).card = 10 :=
by
  sorry

end integer_solutions_count_l287_287488


namespace geometric_sequence_vertex_property_l287_287265

theorem geometric_sequence_vertex_property (a b c d : ℝ) 
  (h_geom : ∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r)
  (h_vertex : b = 1 ∧ c = 2) : a * d = b * c :=
by sorry

end geometric_sequence_vertex_property_l287_287265


namespace area_ratio_gt_two_ninths_l287_287882

variables {A B C P Q R : Type*}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P] [Inhabited Q] [Inhabited R]

def divides_perimeter_eq (A B C : Type*) (P Q R : Type*) : Prop :=
-- Definition that P, Q, and R divide the perimeter into three equal parts
sorry

def is_on_side_AB (A B C P Q : Type*) : Prop :=
-- Definition that points P and Q are on side AB
sorry

theorem area_ratio_gt_two_ninths (A B C P Q R : Type*)
  (H1 : divides_perimeter_eq A B C P Q R)
  (H2 : is_on_side_AB A B C P Q) :
  -- Statement to prove that the area ratio is greater than 2/9
  (S_ΔPQR / S_ΔABC) > (2 / 9) :=
sorry

end area_ratio_gt_two_ninths_l287_287882


namespace casey_saving_l287_287235

-- Define the conditions
def cost_per_hour_first_employee : ℝ := 20
def cost_per_hour_second_employee : ℝ := 22
def subsidy_per_hour : ℝ := 6
def hours_per_week : ℝ := 40

-- Define the weekly cost calculations
def weekly_cost_first_employee := cost_per_hour_first_employee * hours_per_week
def effective_cost_per_hour_second_employee := cost_per_hour_second_employee - subsidy_per_hour
def weekly_cost_second_employee := effective_cost_per_hour_second_employee * hours_per_week

-- State the theorem
theorem casey_saving :
    weekly_cost_first_employee - weekly_cost_second_employee = 160 := 
by
  sorry

end casey_saving_l287_287235


namespace percent_decrease_call_cost_l287_287010

theorem percent_decrease_call_cost (c1990 c2010 : ℝ) (h1990 : c1990 = 50) (h2010 : c2010 = 10) :
  ((c1990 - c2010) / c1990) * 100 = 80 :=
by
  sorry

end percent_decrease_call_cost_l287_287010


namespace elle_practices_hours_l287_287261

variable (practice_time_weekday : ℕ) (days_weekday : ℕ) (multiplier_saturday : ℕ) (minutes_in_an_hour : ℕ) 
          (total_minutes_weekdays : ℕ) (total_minutes_saturday : ℕ) (total_minutes_week : ℕ) (total_hours : ℕ)

theorem elle_practices_hours :
  practice_time_weekday = 30 ∧
  days_weekday = 5 ∧
  multiplier_saturday = 3 ∧
  minutes_in_an_hour = 60 →
  total_minutes_weekdays = practice_time_weekday * days_weekday →
  total_minutes_saturday = practice_time_weekday * multiplier_saturday →
  total_minutes_week = total_minutes_weekdays + total_minutes_saturday →
  total_hours = total_minutes_week / minutes_in_an_hour →
  total_hours = 4 :=
by
  intros
  sorry

end elle_practices_hours_l287_287261


namespace tangent_parallel_and_point_P_l287_287625

noncomputable def f (x : ℝ) : ℝ := x^3 - x + 3

theorem tangent_parallel_and_point_P (P : ℝ × ℝ) (hP1 : P = (1, f 1)) (hP2 : P = (-1, f (-1))) :
  (f 1 = 3 ∧ f (-1) = 3) ∧ (deriv f 1 = 2 ∧ deriv f (-1) = 2) :=
by
  sorry

end tangent_parallel_and_point_P_l287_287625


namespace find_m_for_root_l287_287408

-- Define the fractional equation to find m
def fractional_equation (x m : ℝ) : Prop :=
  (x + 2) / (x - 1) = m / (1 - x)

-- State the theorem that we need to prove
theorem find_m_for_root : ∃ m : ℝ, (∃ x : ℝ, fractional_equation x m) ∧ m = -3 :=
by
  sorry

end find_m_for_root_l287_287408


namespace thirty_ml_of_one_liter_is_decimal_fraction_l287_287507

-- We define the known conversion rule between liters and milliliters.
def liter_to_ml := 1000

-- We define the volume in milliliters that we are considering.
def volume_ml := 30

-- We state the main theorem which asserts that 30 ml of a liter is equal to the decimal fraction 0.03.
theorem thirty_ml_of_one_liter_is_decimal_fraction : (volume_ml / (liter_to_ml : ℝ)) = 0.03 := by
  -- insert proof here
  sorry

end thirty_ml_of_one_liter_is_decimal_fraction_l287_287507


namespace third_rectangle_area_l287_287191

-- Definitions for dimensions of the first two rectangles
def rect1_length := 3
def rect1_width := 8

def rect2_length := 2
def rect2_width := 5

-- Total area of the first two rectangles
def total_area := (rect1_length * rect1_width) + (rect2_length * rect2_width)

-- Declaration of the theorem to be proven
theorem third_rectangle_area :
  ∃ a b : ℝ, a * b = 4 ∧ total_area + a * b = total_area + 4 :=
by
  sorry

end third_rectangle_area_l287_287191


namespace striped_nails_painted_l287_287900

theorem striped_nails_painted (total_nails purple_nails blue_nails : ℕ) (h_total : total_nails = 20)
    (h_purple : purple_nails = 6) (h_blue : blue_nails = 8)
    (h_diff_percent : |(blue_nails:ℚ) / total_nails * 100 - 
    ((total_nails - purple_nails - blue_nails):ℚ) / total_nails * 100| = 10) :
    (total_nails - purple_nails - blue_nails) = 6 := 
by 
  sorry

end striped_nails_painted_l287_287900


namespace jayden_current_age_l287_287591

def current_age_of_Jayden (e : ℕ) (j_in_3_years : ℕ) : ℕ :=
  j_in_3_years - 3

theorem jayden_current_age (e : ℕ) (h1 : e = 11) (h2 : ∃ j : ℕ, j = ((e + 3) / 2) ∧ j_in_3_years = j) : 
  current_age_of_Jayden e j_in_3_years = 4 :=
by
  sorry

end jayden_current_age_l287_287591


namespace council_counts_l287_287886

theorem council_counts 
    (total_classes : ℕ := 20)
    (students_per_class : ℕ := 5)
    (total_students : ℕ := 100)
    (petya_class_council : ℕ × ℕ := (1, 4))  -- (boys, girls)
    (equal_boys_girls : 2 * 50 = total_students)  -- Equal number of boys and girls
    (more_girls_classes : ℕ := 15)
    (min_girls_each : ℕ := 3)
    (remaining_classes : ℕ := 4)
    (remaining_students : ℕ := 20)
    : (19, 1) = (19, 1) :=
by
    -- actual proof goes here
    sorry

end council_counts_l287_287886


namespace greatest_divisor_of_arithmetic_sequence_sum_l287_287342

theorem greatest_divisor_of_arithmetic_sequence_sum :
  ∀ (x c : ℕ), ∃ k : ℕ, k = 15 ∧ 15 ∣ (15 * x + 105 * c) :=
by
  intro x c
  exists 15
  split
  . rfl
  . sorry

end greatest_divisor_of_arithmetic_sequence_sum_l287_287342


namespace course_selection_count_l287_287956

-- Definitions for the conditions
def num_PE_courses : ℕ := 4
def num_art_courses : ℕ := 4
def total_courses : ℕ := num_PE_courses + num_art_courses

-- The number of different course selection schemes
theorem course_selection_count : 
  (choose num_PE_courses 1) * (choose num_art_courses 1) + 
  (choose num_PE_courses 2) * (choose num_art_courses 1) + 
  (choose num_PE_courses 1) * (choose num_art_courses 2) = 64 := 
by sorry

end course_selection_count_l287_287956


namespace first_markup_percentage_l287_287392

-- Definitions for conditions
variables (C : ℝ)  -- cost price
variables (M : ℝ)  -- initial markup percentage
variables (R : ℝ := C * (1 + M / 100))  -- initial retail price
variables (N : ℝ := R * 1.25)  -- New Year's retail price after 25% markup
variables (F : ℝ := N * 0.80)  -- Final price in February after 20% discount

-- The target condition: 20% profit
variables (P : ℝ := 1.20 * C)  -- 120% of the cost price

-- The theorem to prove
theorem first_markup_percentage : F = P → M = 20 := 
by sorry

end first_markup_percentage_l287_287392


namespace find_x_l287_287113

theorem find_x (x : ℝ) (h : (1 / Real.log x / Real.log 5 + 1 / Real.log x / Real.log 7 + 1 / Real.log x / Real.log 11) = 1) : x = 385 := 
sorry

end find_x_l287_287113


namespace is_linear_equation_with_one_var_l287_287374

-- Definitions
def eqA := ∀ (x : ℝ), x^2 + 1 = 5
def eqB := ∀ (x y : ℝ), x + 2 = y - 3
def eqC := ∀ (x : ℝ), 1 / (2 * x) = 10
def eqD := ∀ (x : ℝ), x = 4

-- Theorem stating which equation represents a linear equation in one variable
theorem is_linear_equation_with_one_var : eqD :=
by
  -- Proof skipped
  sorry

end is_linear_equation_with_one_var_l287_287374


namespace problem1_problem2_l287_287268

section ProofProblems

variables {a b : ℝ}

-- Given that a and b are distinct positive numbers
axiom a_pos : a > 0
axiom b_pos : b > 0
axiom a_neq_b : a ≠ b

-- Problem (i): Prove that a^4 + b^4 > a^3 * b + a * b^3
theorem problem1 : a^4 + b^4 > a^3 * b + a * b^3 :=
by {
  sorry
}

-- Problem (ii): Prove that a^5 + b^5 > a^3 * b^2 + a^2 * b^3
theorem problem2 : a^5 + b^5 > a^3 * b^2 + a^2 * b^3 :=
by {
  sorry
}

end ProofProblems

end problem1_problem2_l287_287268


namespace gcf_2550_7140_l287_287935

def gcf (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcf_2550_7140 : gcf 2550 7140 = 510 := 
  by 
    sorry

end gcf_2550_7140_l287_287935


namespace pinecones_left_l287_287501

theorem pinecones_left (initial_pinecones : ℕ)
    (percent_eaten_by_reindeer : ℝ)
    (percent_collected_for_fires : ℝ)
    (twice_eaten_by_squirrels : ℕ → ℕ)
    (eaten_by_reindeer : ℕ → ℝ → ℕ)
    (collected_for_fires : ℕ → ℝ → ℕ)
    (h_initial : initial_pinecones = 2000)
    (h_percent_reindeer : percent_eaten_by_reindeer = 0.20)
    (h_twice_squirrels : ∀ n, twice_eaten_by_squirrels n = 2 * n)
    (h_percent_fires : percent_collected_for_fires = 0.25)
    (h_eaten_reindeer : ∀ n p, eaten_by_reindeer n p = n * p)
    (h_collected_fires : ∀ n p, collected_for_fires n p = n * p) :
  let reindeer_eat := eaten_by_reindeer initial_pinecones percent_eaten_by_reindeer
  let squirrel_eat := twice_eaten_by_squirrels reindeer_eat
  let after_eaten := initial_pinecones - reindeer_eat - squirrel_eat
  let fire_collect := collected_for_fires after_eaten percent_collected_for_fires
  let final_pinecones := after_eaten - fire_collect
  final_pinecones = 600 :=
by sorry

end pinecones_left_l287_287501


namespace fraction_value_l287_287496

theorem fraction_value : (10 + 20 + 30 + 40) / 10 = 10 := by
  sorry

end fraction_value_l287_287496


namespace greatest_divisor_of_sum_of_arithmetic_sequence_l287_287361

theorem greatest_divisor_of_sum_of_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ k : ℕ, (sum (λ n, x + n * c) (range 15)) = 15 * k :=
by sorry

end greatest_divisor_of_sum_of_arithmetic_sequence_l287_287361


namespace sum_of_possible_two_digit_values_l287_287452

theorem sum_of_possible_two_digit_values (d : ℕ) (h1 : 0 < d) (h2 : d < 100) (h3 : 137 % d = 6) : d = 131 :=
by
  sorry

end sum_of_possible_two_digit_values_l287_287452


namespace greatest_divisor_sum_of_first_fifteen_terms_l287_287360

theorem greatest_divisor_sum_of_first_fifteen_terms 
  (x c : ℕ) (hx : x > 0) (hc : c > 0):
  ∃ d, d = 15 ∧ d ∣ (15*x + 105*c) :=
by
  existsi 15
  split
  . refl
  . apply Nat.dvd.intro
    existsi (x + 7*c)
    refl
  sorry

end greatest_divisor_sum_of_first_fifteen_terms_l287_287360


namespace tom_used_10_plates_l287_287782

theorem tom_used_10_plates
  (weight_per_plate : ℕ := 30)
  (felt_weight : ℕ := 360)
  (heavier_factor : ℚ := 1.20) :
  (felt_weight / heavier_factor / weight_per_plate : ℚ) = 10 := by
  sorry

end tom_used_10_plates_l287_287782


namespace arcsin_sqrt_three_over_two_l287_287546

theorem arcsin_sqrt_three_over_two : Real.arcsin (Real.sqrt 3 / 2) = Real.pi / 3 :=
by
  -- The proof is omitted
  sorry

end arcsin_sqrt_three_over_two_l287_287546


namespace add_pure_acid_to_obtain_final_concentration_l287_287583

   variable (x : ℝ)

   def initial_solution_volume : ℝ := 60
   def initial_acid_concentration : ℝ := 0.10
   def final_acid_concentration : ℝ := 0.15

   axiom calculate_pure_acid (x : ℝ) :
     initial_acid_concentration * initial_solution_volume + x = final_acid_concentration * (initial_solution_volume + x)

   noncomputable def pure_acid_solution : ℝ := 3/0.85

   theorem add_pure_acid_to_obtain_final_concentration :
     x = pure_acid_solution := by
     sorry
   
end add_pure_acid_to_obtain_final_concentration_l287_287583


namespace total_selection_schemes_l287_287962

-- Define the given conditions
def num_phys_ed_courses : ℕ := 4
def num_art_courses : ℕ := 4
def total_courses : ℕ := num_phys_ed_courses + num_art_courses
def valid_course_combos : finset ℕ := {2, 3}

-- Define the number of selection schemes with the constraints
def selection_schemes : nat :=
  (num_phys_ed_courses.choose 1 * num_art_courses.choose 1) + 
  (num_phys_ed_courses.choose 2 * num_art_courses.choose 1 + 
   num_phys_ed_courses.choose 1 * num_art_courses.choose 2)

-- State the theorem to be proved
theorem total_selection_schemes : selection_schemes = 64 := by
  sorry

end total_selection_schemes_l287_287962


namespace solve_for_b_l287_287852

/-- 
Given the ellipse \( x^2 + \frac{y^2}{b^2 + 1} = 1 \) where \( b > 0 \),
and the eccentricity of the ellipse is \( \frac{\sqrt{10}}{10} \),
prove that \( b = \frac{1}{3} \).
-/
theorem solve_for_b (b : ℝ) (hb : b > 0) (heccentricity : b / (Real.sqrt (b^2 + 1)) = Real.sqrt 10 / 10) : 
  b = 1 / 3 :=
sorry

end solve_for_b_l287_287852


namespace inverse_of_f_l287_287003

def f (x : ℝ) : ℝ := 7 - 3 * x

noncomputable def f_inv (x : ℝ) : ℝ := (7 - x) / 3

theorem inverse_of_f : ∀ x : ℝ, f (f_inv x) = x ∧ f_inv (f x) = x :=
by
  intros
  sorry

end inverse_of_f_l287_287003


namespace sum_of_three_squares_l287_287535

theorem sum_of_three_squares (a b c : ℤ) (h1 : 2 * a + 2 * b + c = 27) (h2 : a + 3 * b + c = 25) : 3 * c = 33 :=
  sorry

end sum_of_three_squares_l287_287535


namespace problem_l287_287877

def f (u : ℝ) : ℝ := u^2 - 2

theorem problem : f 3 = 7 := 
by sorry

end problem_l287_287877


namespace rice_weight_employees_l287_287533

noncomputable def rice_less_than_9_9 (n : ℕ) (ξ : ℝ → ℝ) (σ : ℝ) : ℝ :=
let p_9_9 : ℝ := (1 - 0.96) / 2 in
n * p_9_9

theorem rice_weight_employees : rice_less_than_9_9 2000 (λ x, pdf (NormalDist.mk 10 σ ^ 2) x) = 40 :=
by
  sorry

end rice_weight_employees_l287_287533


namespace induced_charge_density_l287_287969

noncomputable def sigma_0 (x : ℝ) (h : ℝ) (tilde_f0 tilde_fh : ℝ → ℝ) : ℝ :=
  (1 / (4 * Real.pi ^ 2)) * ∫ λ in -∞..∞, ((tilde_f0 λ + tilde_fh λ * Real.exp (-|λ| * h)) / (1 - Real.exp (-2 * |λ| * h))) * Real.exp (-Complex.I * λ * x)

noncomputable def sigma_h (x : ℝ) (h : ℝ) (tilde_f0 tilde_fh : ℝ → ℝ) : ℝ :=
  (1 / (4 * Real.pi ^ 2)) * ∫ λ in -∞..∞, ((tilde_fh λ + tilde_f0 λ * Real.exp (-|λ| * h)) / (1 - Real.exp (-2 * |λ| * h))) * Real.exp (-Complex.I * λ * x)

theorem induced_charge_density (h : ℝ) (tilde_f0 tilde_fh : ℝ → ℝ) :
  ∃ σ0 σh, (σ0 = sigma_0 h tilde_f0 tilde_fh) ∧ (σh = sigma_h h tilde_f0 tilde_fh) :=
by
  sorry

end induced_charge_density_l287_287969


namespace library_charge_l287_287660

-- Definitions according to given conditions
def daily_charge : ℝ := 0.5
def days_in_may : ℕ := 31
def days_borrowed1 : ℕ := 20
def days_borrowed2 : ℕ := 31

-- Calculation of total charge
theorem library_charge :
  let total_charge := (daily_charge * days_borrowed1) + (2 * daily_charge * days_borrowed2)
  total_charge = 41 :=
by
  sorry

end library_charge_l287_287660


namespace product_of_primes_is_even_l287_287606

-- Define the conditions for P and Q to cover P, Q, P-Q, and P+Q being prime and positive
def is_prime (n : ℕ) : Prop := ¬ (n = 0 ∨ n = 1) ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem product_of_primes_is_even {P Q : ℕ} (hP : is_prime P) (hQ : is_prime Q) 
  (hPQ_diff : is_prime (P - Q)) (hPQ_sum : is_prime (P + Q)) 
  (hPosP : P > 0) (hPosQ : Q > 0) 
  (hPosPQ_diff : P - Q > 0) (hPosPQ_sum : P + Q > 0) : 
  ∃ k : ℕ, P * Q * (P - Q) * (P + Q) = 2 * k := 
sorry

end product_of_primes_is_even_l287_287606


namespace total_spent_l287_287145

def original_cost_vacuum_cleaner : ℝ := 250
def discount_vacuum_cleaner : ℝ := 0.20
def cost_dishwasher : ℝ := 450
def special_offer_discount : ℝ := 75

theorem total_spent :
  let discounted_vacuum_cleaner := original_cost_vacuum_cleaner * (1 - discount_vacuum_cleaner)
  let total_before_special := discounted_vacuum_cleaner + cost_dishwasher
  total_before_special - special_offer_discount = 575 := by
  sorry

end total_spent_l287_287145


namespace evaluate_expression_l287_287242

-- Definition of the function f
def f (x : ℤ) : ℤ := 3 * x^2 - 5 * x + 8

-- Theorems and lemmas
theorem evaluate_expression : 3 * f 4 + 2 * f (-4) = 260 := by
  sorry

end evaluate_expression_l287_287242


namespace total_vacations_and_classes_l287_287111

def kelvin_classes := 90
def grant_vacations := 4 * kelvin_classes
def total := grant_vacations + kelvin_classes

theorem total_vacations_and_classes :
  total = 450 :=
by
  sorry

end total_vacations_and_classes_l287_287111


namespace probability_sum_divisible_by_3_l287_287929

theorem probability_sum_divisible_by_3 :
  let balls := {1, 3, 5, 7, 9}
  let all_combinations := Finset.powersetLen 3 (Finset.of_array balls)
  let favorable_combinations := all_combinations.filter (λ s, s.sum % 3 = 0)
  (favorable_combinations.card / all_combinations.card : ℚ) = 2 / 5 :=
by {
  sorry
}

end probability_sum_divisible_by_3_l287_287929


namespace selling_price_l287_287970

theorem selling_price (profit_percent : ℝ) (cost_price : ℝ) (h_profit : profit_percent = 5) (h_cp : cost_price = 2400) :
  let profit := (profit_percent / 100) * cost_price 
  let selling_price := cost_price + profit
  selling_price = 2520 :=
by
  sorry

end selling_price_l287_287970


namespace range_of_b_min_value_a_add_b_min_value_ab_l287_287568

theorem range_of_b (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) : b > 1 := sorry

theorem min_value_a_add_b (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) : a + b ≥ 8 := sorry

theorem min_value_ab (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) : a * b ≥ 16 := sorry

end range_of_b_min_value_a_add_b_min_value_ab_l287_287568


namespace find_aroon_pin_l287_287541

theorem find_aroon_pin (a b : ℕ) (PIN : ℕ) 
  (h0 : 0 ≤ a ∧ a ≤ 9)
  (h1 : 0 ≤ b ∧ b < 1000)
  (h2 : PIN = 1000 * a + b)
  (h3 : 10 * b + a = 3 * PIN - 6) : 
  PIN = 2856 := 
sorry

end find_aroon_pin_l287_287541


namespace greatest_divisor_arithmetic_sum_l287_287336

theorem greatest_divisor_arithmetic_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ d, d = 15 ∧ ∀ S : ℕ, S = 15 * x + 105 * c → d ∣ S :=
by 
  sorry

end greatest_divisor_arithmetic_sum_l287_287336


namespace tan_30_degrees_correct_l287_287495

noncomputable def tan_30_degrees : ℝ := Real.tan (Real.pi / 6)

theorem tan_30_degrees_correct : tan_30_degrees = Real.sqrt 3 / 3 :=
by
  sorry

end tan_30_degrees_correct_l287_287495


namespace inequality_abc_ad_bc_bd_cd_l287_287092

theorem inequality_abc_ad_bc_bd_cd (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (1 / (a * b) + 1 / (a * c) + 1 / (a * d) + 1 / (b * c) + 1 / (b * d) + 1 / (c * d)) 
  ≤ (3 / 8) * (1 / a + 1 / b + 1 / c + 1 / d) ^ 2 := sorry

end inequality_abc_ad_bc_bd_cd_l287_287092


namespace vertical_shift_d_l287_287829

variable (a b c d : ℝ)

theorem vertical_shift_d (h1: d + a = 5) (h2: d - a = 1) : d = 3 := 
by
  sorry

end vertical_shift_d_l287_287829


namespace weight_of_fourth_dog_l287_287182

theorem weight_of_fourth_dog (y x : ℝ) : 
  (25 + 31 + 35 + x) / 4 = (25 + 31 + 35 + x + y) / 5 → 
  x = -91 - 5 * y :=
by
  sorry

end weight_of_fourth_dog_l287_287182


namespace limit_n_a_n_to_zero_l287_287449

open Filter Real

theorem limit_n_a_n_to_zero 
  (a : ℕ → ℝ)
  (h_bound : ∀ n k, n ≤ k ∧ k ≤ 2 * n → 0 ≤ a k ∧ a k ≤ 100 * a n)
  (h_series : Summable a) : 
  Tendsto (λ n, n * a n) atTop (𝓝 0) :=
sorry

end limit_n_a_n_to_zero_l287_287449


namespace seq_eleven_l287_287415

noncomputable def seq (n : ℕ) : ℤ := sorry

axiom seq_add (p q : ℕ) (hp : 0 < p) (hq : 0 < q) : seq (p + q) = seq p + seq q
axiom seq_two : seq 2 = -6

theorem seq_eleven : seq 11 = -33 := by
  sorry

end seq_eleven_l287_287415


namespace faster_runner_l287_287039

-- Define the speeds of A and B
variables (v_A v_B : ℝ)
-- A's speed as a multiple of B's speed
variables (k : ℝ)

-- A's and B's distances in the race
variables (d_A d_B : ℝ)
-- Distance of the race
variables (distance : ℝ)
-- Head start given to B
variables (head_start : ℝ)

-- The theorem to prove that the factor k is 4 given the conditions
theorem faster_runner (k : ℝ) (v_A v_B : ℝ) (d_A d_B distance head_start : ℝ) :
  v_A = k * v_B ∧ d_B = distance - head_start ∧ d_A = distance ∧ (d_A / v_A) = (d_B / v_B) → k = 4 :=
by
  sorry

end faster_runner_l287_287039


namespace probability_of_sum_being_odd_l287_287506

noncomputable def probability_sum_odd : ℚ :=
  let p_heads := (1/2 : ℚ) in
  let p_tails := 1 - p_heads in
  let p_one_head := (fintype.card {i : Fin 3 // i < 1} : ℕ) * p_heads * p_tails^2 in
  let p_two_heads := (fintype.card {i : Fin 3 // i < 2} * (p_heads^2) * p_tails) in
  let p_three_heads := p_heads^3 in
  let p_sum_odd_one_die := 1 / 2 in
  let p_one_die_sum_odd := p_one_head * p_sum_odd_one_die in
  let p_two_dice_sum_odd := p_two_heads * (2 * ((1/2) * (1/2))) in
  let p_three_dice_sum_odd := p_three_heads * (fintype.card {i : Fin 3 // i = 1} * (1/2)^3) in
  p_one_die_sum_odd + p_two_dice_sum_odd + p_three_dice_sum_odd

theorem probability_of_sum_being_odd : probability_sum_odd = (7 / 16 : ℚ) := 
sorry

end probability_of_sum_being_odd_l287_287506


namespace no_integer_satisfies_inequality_l287_287554

theorem no_integer_satisfies_inequality :
  ∀ x : ℤ, (30 < x ∧ x < 90) → log 10 (x - 30) + log 10 (90 - x) < 1 → false :=
by 
  intro x h1 h2
  sorry

end no_integer_satisfies_inequality_l287_287554


namespace find_slope_of_chord_l287_287716

noncomputable def slope_of_chord (x1 x2 y1 y2 : ℝ) : ℝ :=
  (y1 - y2) / (x1 - x2)

theorem find_slope_of_chord :
  (∀ (x y : ℝ), x^2 / 36 + y^2 / 9 = 1 → ∃ (x1 x2 y1 y2 : ℝ),
    x1 + x2 = 8 ∧ y1 + y2 = 4 ∧ x = (x1 + x2) / 2 ∧ y = (y1 + y2) / 2 ∧ slope_of_chord x1 x2 y1 y2 = -1 / 2) := sorry

end find_slope_of_chord_l287_287716


namespace ants_meet_again_at_P_l287_287784

-- Definitions for given radii
def radius_large : ℝ := 7
def radius_small : ℝ := 3

-- Definitions for given speeds
def speed_large : ℝ := 5 * Real.pi
def speed_small : ℝ := 4 * Real.pi

-- Circumferences (These are intermediate results derived from the initial conditions)
def circumference_large : ℝ := 2 * radius_large * Real.pi
def circumference_small : ℝ := 2 * radius_small * Real.pi

-- Time for each ant to complete one lap
def time_large : ℚ := (circumference_large / speed_large : ℝ)
def time_small : ℚ := (circumference_small / speed_small : ℝ)

-- LCM Calculation (to find when they both meet at point P again)
-- LCM of two rational numbers is their LCM divided by their gcd, promoted to rationals
def lcm_rational (a b: ℚ) : ℚ := (a.num * b.num) / (Nat.gcd a.den b.den * (Nat.gcd a.num b.num))

-- Expected result (42 Minutes in rational form for ease of comparison)
def meet_again_time : ℚ := 42

theorem ants_meet_again_at_P : lcm_rational time_large time_small = meet_again_time := 
by sorry

end ants_meet_again_at_P_l287_287784


namespace oldest_child_age_l287_287008

theorem oldest_child_age (x : ℕ) (h : (6 + 8 + x) / 3 = 9) : x = 13 := 
by {
  sorry
}

end oldest_child_age_l287_287008


namespace solution_is_correct_l287_287619

-- Define the conditions of the problem.
variable (x y z : ℝ)

-- The system of equations given in the problem
def system_of_equations (x y z : ℝ) :=
  (1/x + 1/(y+z) = 6/5) ∧
  (1/y + 1/(x+z) = 3/4) ∧
  (1/z + 1/(x+y) = 2/3)

-- The desired solution
def solution (x y z : ℝ) := x = 2 ∧ y = 3 ∧ z = 1

-- The theorem to prove
theorem solution_is_correct (h : system_of_equations x y z) : solution x y z :=
sorry

end solution_is_correct_l287_287619


namespace boys_in_other_communities_l287_287727

theorem boys_in_other_communities (total_boys : ℕ) (muslim_percent hindu_percent sikh_percent : ℕ)
  (H_total : total_boys = 400)
  (H_muslim : muslim_percent = 44)
  (H_hindu : hindu_percent = 28)
  (H_sikh : sikh_percent = 10) :
  total_boys * (1 - (muslim_percent + hindu_percent + sikh_percent) / 100) = 72 :=
by
  sorry

end boys_in_other_communities_l287_287727


namespace find_y_values_l287_287905

theorem find_y_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 90) :
  ∃ y, (y = 0 ∨ y = 144 ∨ y = -24) ∧ y = (x - 3)^2 * (x + 4) / (2 * x - 5) :=
by sorry

end find_y_values_l287_287905


namespace minimum_value_of_a_plus_5b_l287_287101

theorem minimum_value_of_a_plus_5b :
  ∀ (a b : ℝ), a > 0 → b > 0 → (1 / a + 5 / b = 1) → a + 5 * b ≥ 36 :=
by
  sorry

end minimum_value_of_a_plus_5b_l287_287101


namespace number_of_scenarios_l287_287220

theorem number_of_scenarios (n k : ℕ) (h₁ : n = 6) (h₂ : k = 6) :
    ∃ m: ℕ, m = Nat.choose 6 2 * 5^4
:= by
  use Nat.choose 6 2 * 5^4
  sorry

end number_of_scenarios_l287_287220


namespace final_number_is_6218_l287_287463

theorem final_number_is_6218 :
  ∃ n : ℕ, (n = 6218) ∧ (∀ k, 3 ≤ k ∧ k ≤ 223 ∧ k % 4 = 3) ∧
  (∀ S : List ℕ, (∀ x ∈ S, 3 ≤ x ∧ x ≤ 223 ∧ x % 4 = 3) → list.sum S - 2 * (S.length - 1) = 6218) :=
begin
  sorry
end

end final_number_is_6218_l287_287463


namespace dark_lord_squads_l287_287921

def total_weight : ℕ := 1200
def orcs_per_squad : ℕ := 8
def capacity_per_orc : ℕ := 15
def squads_needed (w n c : ℕ) : ℕ := w / (n * c)

theorem dark_lord_squads :
  squads_needed total_weight orcs_per_squad capacity_per_orc = 10 :=
by sorry

end dark_lord_squads_l287_287921


namespace fish_population_estimate_l287_287176

theorem fish_population_estimate 
  (caught_first : ℕ) 
  (caught_first_marked : ℕ) 
  (caught_second : ℕ) 
  (caught_second_marked : ℕ) 
  (proportion_eq : (caught_second_marked : ℚ) / caught_second = (caught_first_marked : ℚ) / caught_first) 
  : caught_first * caught_second / caught_second_marked = 750 := 
by 
  sorry

-- Conditions used as definitions in Lean 4
def pond_fish_total (caught_first : ℕ) (caught_second : ℕ) (caught_second_marked : ℕ) : ℚ :=
  (caught_first : ℚ) * (caught_second : ℚ) / (caught_second_marked : ℚ)

-- Example usage of conditions
example : pond_fish_total 30 50 2 = 750 := 
by
  sorry

end fish_population_estimate_l287_287176


namespace editors_min_count_l287_287828

theorem editors_min_count
  (writers : ℕ)
  (P : ℕ)
  (S : ℕ)
  (W : ℕ)
  (H1 : writers = 45)
  (H2 : P = 90)
  (H3 : ∀ x : ℕ, x ≤ 6 → (90 = (writers + W - x) + 2 * x) → W ≥ P - 51)
  : W = 39 := by
  sorry

end editors_min_count_l287_287828


namespace non_congruent_triangles_with_perimeter_11_l287_287865

theorem non_congruent_triangles_with_perimeter_11 : 
  ∀ (a b c : ℕ), a + b + c = 11 → a < b + c → b < a + c → c < a + b → 
  ∃! (a b c : ℕ), (a, b, c) = (2, 4, 5) ∨ (a, b, c) = (3, 4, 4) := 
by sorry

end non_congruent_triangles_with_perimeter_11_l287_287865


namespace batsman_total_score_eq_120_l287_287808

/-- A batsman's runs calculation including boundaries, sixes, and running between wickets. -/
def batsman_runs_calculation (T : ℝ) : Prop :=
  let runs_from_boundaries := 5 * 4
  let runs_from_sixes := 5 * 6
  let runs_from_total := runs_from_boundaries + runs_from_sixes
  let runs_from_running := 0.5833333333333334 * T
  T = runs_from_total + runs_from_running

theorem batsman_total_score_eq_120 :
  ∃ T : ℝ, batsman_runs_calculation T ∧ T = 120 :=
sorry

end batsman_total_score_eq_120_l287_287808


namespace find_n_l287_287105

theorem find_n (n : ℕ) (h₁ : 3 * n + 4 = 13) : n = 3 :=
by 
  sorry

end find_n_l287_287105


namespace intersection_of_sets_l287_287107

theorem intersection_of_sets :
  let A := {-2, -1, 0, 1, 2}
  let B := {x | -2 < x ∧ x ≤ 2}
  A ∩ B = {-1, 0, 1, 2} :=
by
  sorry

end intersection_of_sets_l287_287107


namespace a8_eq_64_l287_287270

variable (S : ℕ → ℕ)
variable (a : ℕ → ℕ)

axiom a1_eq_2 : a 1 = 2
axiom S_recurrence : ∀ (n : ℕ), S (n + 1) = 2 * S n - 1

theorem a8_eq_64 : a 8 = 64 := 
by
sorry

end a8_eq_64_l287_287270


namespace fixed_line_of_midpoint_l287_287853

theorem fixed_line_of_midpoint
  (A B : ℝ × ℝ)
  (H : ∀ (P : ℝ × ℝ), (P = A ∨ P = B) → (P.1^2 / 3 - P.2^2 / 6 = 1))
  (slope_l : (B.2 - A.2) / (B.1 - A.1) = 2)
  (midpoint_lies : (A.1 + B.1) / 2 = (A.2 + B.2) / 2) :
  ∀ (M : ℝ × ℝ), (M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2) → M.1 - M.2 = 0 :=
by
  sorry

end fixed_line_of_midpoint_l287_287853


namespace sum_of_interior_angles_n_plus_3_l287_287013

-- Define the condition that the sum of the interior angles of a convex polygon with n sides is 1260 degrees
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Prove that given the above condition for n, the sum of the interior angles of a convex polygon with n + 3 sides is 1800 degrees
theorem sum_of_interior_angles_n_plus_3 (n : ℕ) (h : sum_of_interior_angles n = 1260) : 
  sum_of_interior_angles (n + 3) = 1800 :=
by
  sorry

end sum_of_interior_angles_n_plus_3_l287_287013


namespace B_pow_5_eq_rB_plus_sI_l287_287739

def B : Matrix (Fin 2) (Fin 2) ℤ := !![1, 1; 4, 5]

def I : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 0, 1]

theorem B_pow_5_eq_rB_plus_sI : 
  ∃ (r s : ℤ), r = 1169 ∧ s = -204 ∧ B^5 = r • B + s • I := 
by
  use 1169
  use -204
  sorry

end B_pow_5_eq_rB_plus_sI_l287_287739


namespace non_congruent_triangles_with_perimeter_11_l287_287864

theorem non_congruent_triangles_with_perimeter_11 :
  { t : ℕ × ℕ × ℕ // let (a, b, c) := t in a + b + c = 11 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a ≤ b ∧ b ≤ c }.card = 4 :=
sorry

end non_congruent_triangles_with_perimeter_11_l287_287864


namespace cylindrical_to_rectangular_l287_287243

theorem cylindrical_to_rectangular :
  ∀ (r θ z : ℝ), r = 5 → θ = (3 * Real.pi) / 4 → z = 2 →
    (r * Real.cos θ, r * Real.sin θ, z) = (-5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2, 2) :=
by
  intros r θ z hr hθ hz
  rw [hr, hθ, hz]
  -- Proof steps would go here, but are omitted as they are not required.
  sorry

end cylindrical_to_rectangular_l287_287243


namespace set_intersection_eq_l287_287906

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | -2 < x ∧ x < 2}

-- The proof statement
theorem set_intersection_eq :
  A ∩ B = A :=
sorry

end set_intersection_eq_l287_287906


namespace int_values_satisfy_condition_l287_287480

theorem int_values_satisfy_condition :
  ∃ (count : ℕ), count = 10 ∧ ∀ (x : ℤ), 6 > Real.sqrt x ∧ Real.sqrt x > 5 ↔ (x ≥ 26 ∧ x ≤ 35) := by
  sorry

end int_values_satisfy_condition_l287_287480


namespace total_squares_after_erasing_lines_l287_287689

theorem total_squares_after_erasing_lines :
  ∀ (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ), a = 16 → b = 4 → c = 9 → d = 2 → 
  a - b + c - d + (a / 16) = 22 := 
by
  intro a b c d h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end total_squares_after_erasing_lines_l287_287689


namespace inverse_proposition_is_false_l287_287874

theorem inverse_proposition_is_false (a : ℤ) (h : a = 6) : ¬ (|a| = 6 → a = 6) :=
sorry

end inverse_proposition_is_false_l287_287874


namespace speed_of_stream_l287_287815

variables (V_d V_u V_m V_s : ℝ)
variables (h1 : V_d = V_m + V_s) (h2 : V_u = V_m - V_s) (h3 : V_d = 18) (h4 : V_u = 6) (h5 : V_m = 12)

theorem speed_of_stream : V_s = 6 :=
by
  sorry

end speed_of_stream_l287_287815


namespace sufficient_but_not_necessary_condition_l287_287264

theorem sufficient_but_not_necessary_condition (a : ℝ) (h : a ≠ 0) :
  (a > 2 ↔ |a - 1| > 1) ↔ (a > 2 → |a - 1| > 1) ∧ (a < 0 → |a - 1| > 1) ∧ (∃ x : ℝ, (|x - 1| > 1) ∧ x < 0 ∧ x ≠ a) :=
by
  sorry

end sufficient_but_not_necessary_condition_l287_287264


namespace sufficient_condition_for_P_l287_287585

noncomputable def increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

theorem sufficient_condition_for_P (f : ℝ → ℝ) (t : ℝ) 
  (h_inc : increasing f) (h_val1 : f (-1) = -4) (h_val2 : f 2 = 2) :
  (∀ x, (x ∈ {x | -1 - t < x ∧ x < 2 - t}) → x < -1) → t ≥ 3 :=
by
  sorry

end sufficient_condition_for_P_l287_287585


namespace greatest_divisor_of_sum_of_arithmetic_sequence_l287_287363

theorem greatest_divisor_of_sum_of_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ k : ℕ, (sum (λ n, x + n * c) (range 15)) = 15 * k :=
by sorry

end greatest_divisor_of_sum_of_arithmetic_sequence_l287_287363


namespace greatest_m_value_l287_287696

theorem greatest_m_value (x y m : ℝ) 
  (h₁: x^2 + y^2 = 1)
  (h₂ : |x^3 - y^3| + |x - y| = m^3) : 
  m ≤ 2^(1/3) :=
sorry

end greatest_m_value_l287_287696


namespace find_xyz_sum_l287_287947

theorem find_xyz_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^2 + x * y + y^2 = 12)
  (h2 : y^2 + y * z + z^2 = 25)
  (h3 : z^2 + z * x + x^2 = 37) :
  x * y + y * z + z * x = 20 :=
sorry

end find_xyz_sum_l287_287947


namespace condition_an_necessary_but_not_sufficient_l287_287071

-- Definitions for the sequence and properties
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, n ≥ 1 → a (n + 1) = r * (a n)

def condition_an (a : ℕ → ℝ) : Prop :=
  ∀ n, n ≥ 2 → a n = 2 * a (n - 1)

-- The theorem statement
theorem condition_an_necessary_but_not_sufficient (a : ℕ → ℝ) :
  (∀ n, n ≥ 1 → a (n + 1) = 2 * (a n)) → (condition_an a) ∧ ¬(is_geometric_sequence a 2) :=
by
  sorry

end condition_an_necessary_but_not_sufficient_l287_287071


namespace rate_percent_l287_287937

theorem rate_percent (SI P T: ℝ) (h₁: SI = 250) (h₂: P = 1500) (h₃: T = 5) : 
  ∃ R : ℝ, R = (SI * 100) / (P * T) := 
by
  use (250 * 100) / (1500 * 5)
  sorry

end rate_percent_l287_287937


namespace least_value_expression_l287_287509

theorem least_value_expression (x y : ℝ) : 
  (x^2 * y + x * y^2 - 1)^2 + (x + y)^2 ≥ 1 :=
sorry

end least_value_expression_l287_287509


namespace greatest_divisor_arithmetic_sequence_sum_l287_287350

theorem greatest_divisor_arithmetic_sequence_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ k, (15 * (x + 7 * c)) = 15 * k :=
sorry

end greatest_divisor_arithmetic_sequence_sum_l287_287350


namespace solution_set_inequality_one_range_m_nonempty_set_l287_287421

def f (x : ℝ) := abs (x + 1) - abs (x - 2)

theorem solution_set_inequality_one : 
  {x | f x ≥ 1} = {x | x ≥ 1} :=
by
  sorry

theorem range_m_nonempty_set (m : ℝ) : 
  (∃ x, f x ≥ x^2 - x + m) ↔ m ≤ 5 / 4 :=
by
  sorry

end solution_set_inequality_one_range_m_nonempty_set_l287_287421


namespace find_number_l287_287974

theorem find_number (x : ℤ) (h : (((55 + x) / 7 + 40) * 5 = 555)) : x = 442 :=
sorry

end find_number_l287_287974


namespace part1_part2_l287_287106

def P (x : ℝ) : Prop := |x - 1| > 2
def S (x : ℝ) (a : ℝ) : Prop := x^2 - (a + 1) * x + a > 0

theorem part1 (a : ℝ) (h : a = 2) : ∀ x, S x a ↔ x < 1 ∨ x > 2 :=
by
  sorry

theorem part2 (a : ℝ) (h : a ≠ 1) : ∀ x, (P x → S x a) → (-1 ≤ a ∧ a < 1) ∨ (1 < a ∧ a ≤ 3) :=
by
  sorry

end part1_part2_l287_287106


namespace intersection_of_sets_l287_287130

theorem intersection_of_sets :
  let M := { x : ℝ | 0 ≤ x ∧ x < 16 }
  let N := { x : ℝ | x ≥ 1/3 }
  M ∩ N = { x : ℝ | 1/3 ≤ x ∧ x < 16 } :=
by
  sorry

end intersection_of_sets_l287_287130


namespace A_and_B_worked_together_for_5_days_before_A_left_the_job_l287_287385

noncomputable def workRate_A (W : ℝ) : ℝ := W / 20
noncomputable def workRate_B (W : ℝ) : ℝ := W / 12

noncomputable def combinedWorkRate (W : ℝ) : ℝ := workRate_A W + workRate_B W

noncomputable def workDoneTogether (x : ℝ) (W : ℝ) : ℝ := x * combinedWorkRate W
noncomputable def workDoneBy_B_Alone (W : ℝ) : ℝ := 3 * workRate_B W

theorem A_and_B_worked_together_for_5_days_before_A_left_the_job (W : ℝ) :
  ∃ x : ℝ, workDoneTogether x W + workDoneBy_B_Alone W = W ∧ x = 5 :=
by
  sorry

end A_and_B_worked_together_for_5_days_before_A_left_the_job_l287_287385


namespace problem_1_problem_2_ln_ineq_problem_3_l287_287550

noncomputable def a : ℕ → ℝ
| 1       := 1
| (n + 1) := (1 + 1 / (n^2 + n)) * a n + 1 / (2^n)

noncomputable def b : ℕ → ℝ
| n := if h : n > 0 then (a (n + 1) - a n) / a n else 0

theorem problem_1 : ∀ (n : ℕ), n ≥ 2 → a n ≥ 2 := by sorry

theorem problem_2 : ∀ (n : ℕ), ∑ i in finset.range n, b (i + 1) < 7 / 4 := by sorry

theorem ln_ineq (x : ℝ) (hx : x > 0) : real.log (1 + x) < x :=
begin
  -- Here you'd include the mathematical proof for the given inequality,
  -- but it is asserted as a true theorem in the problem setup.
  sorry
end

theorem problem_3 : ∀ (n : ℕ), a n < 2 * real.exp (3 / 4) := by sorry

end problem_1_problem_2_ln_ineq_problem_3_l287_287550


namespace smallest_prime_after_six_consecutive_nonprimes_l287_287791

-- Define what it means to be prime
def is_prime (n : ℕ) : Prop :=
  nat.prime n

-- Define the sequence of six consecutive nonprime numbers
def consecutive_nonprime_sequence (a : ℕ) : Prop :=
  ¬ is_prime a ∧ ¬ is_prime (a + 1) ∧ ¬ is_prime (a + 2) ∧
  ¬ is_prime (a + 3) ∧ ¬ is_prime (a + 4) ∧ ¬ is_prime (a + 5)

-- Define the condition that there is a smallest prime number greater than a sequence of six consecutive nonprime numbers
theorem smallest_prime_after_six_consecutive_nonprimes :
  ∃ p, is_prime p ∧ p > 96 ∧ consecutive_nonprime_sequence 90 :=
sorry

end smallest_prime_after_six_consecutive_nonprimes_l287_287791


namespace calc_dz_calc_d2z_calc_d3z_l287_287837

variables (x y dx dy : ℝ)

def z : ℝ := x^5 * y^3

-- Define the first differential dz
def dz : ℝ := 5 * x^4 * y^3 * dx + 3 * x^5 * y^2 * dy

-- Define the second differential d2z
def d2z : ℝ := 20 * x^3 * y^3 * dx^2 + 30 * x^4 * y^2 * dx * dy + 6 * x^5 * y * dy^2

-- Define the third differential d3z
def d3z : ℝ := 60 * x^2 * y^3 * dx^3 + 180 * x^3 * y^2 * dx^2 * dy + 90 * x^4 * y * dx * dy^2 + 6 * x^5 * dy^3

theorem calc_dz : (dz x y dx dy) = (5 * x^4 * y^3 * dx + 3 * x^5 * y^2 * dy) := 
by sorry

theorem calc_d2z : (d2z x y dx dy) = (20 * x^3 * y^3 * dx^2 + 30 * x^4 * y^2 * dx * dy + 6 * x^5 * y * dy^2) :=
by sorry

theorem calc_d3z : (d3z x y dx dy) = (60 * x^2 * y^3 * dx^3 + 180 * x^3 * y^2 * dx^2 * dy + 90 * x^4 * y * dx * dy^2 + 6 * x^5 * dy^3) :=
by sorry

end calc_dz_calc_d2z_calc_d3z_l287_287837


namespace sum_of_digits_div_by_11_in_consecutive_39_l287_287754

-- Define the sum of digits function for natural numbers.
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- The main theorem statement.
theorem sum_of_digits_div_by_11_in_consecutive_39 :
  ∀ (N : ℕ), ∃ k : ℕ, k < 39 ∧ (sum_of_digits (N + k)) % 11 = 0 :=
by sorry

end sum_of_digits_div_by_11_in_consecutive_39_l287_287754


namespace rich_total_distance_l287_287761

-- Define the given conditions 
def distance_house_to_sidewalk := 20
def distance_down_road := 200
def total_distance_so_far := distance_house_to_sidewalk + distance_down_road
def distance_left_turn := 2 * total_distance_so_far
def distance_to_intersection := total_distance_so_far + distance_left_turn
def distance_half := distance_to_intersection / 2
def total_distance_one_way := distance_to_intersection + distance_half

-- Define the theorem to be proven 
theorem rich_total_distance : total_distance_one_way * 2 = 1980 :=
by 
  -- This line is to complete the 'prove' demand of the theorem
  sorry

end rich_total_distance_l287_287761


namespace rooms_with_two_windows_l287_287078

theorem rooms_with_two_windows
  (total_windows : ℕ)
  (rooms_with_four_windows : ℕ)
  (windows_per_four_windows : ℕ)
  (rooms_with_three_windows : ℕ)
  (windows_per_three_windows : ℕ)
  (rooms_with_two_windows : ℕ -> ℕ -> ℕ -> ℕ -> ℕ)
  (total_rooms : 5)
  (windows_four : 4)
  (rooms_three : 8)
  (windows_three : 3)
  (result : 39) :
  rooms_with_two_windows total_windows (rooms_with_four_windows * windows_per_four_windows)
  (rooms_with_three_windows * windows_per_three_windows) 2 = result :=
by
  sorry

end rooms_with_two_windows_l287_287078


namespace greatest_divisor_arithmetic_sequence_sum_l287_287347

theorem greatest_divisor_arithmetic_sequence_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ k, (15 * (x + 7 * c)) = 15 * k :=
sorry

end greatest_divisor_arithmetic_sequence_sum_l287_287347


namespace locust_population_doubling_time_l287_287477

theorem locust_population_doubling_time 
  (h: ℕ)
  (initial_population : ℕ := 1000)
  (time_past : ℕ := 4)
  (future_time: ℕ := 10)
  (population_limit: ℕ := 128000) :
  1000 * 2 ^ ((10 + 4) / h) > 128000 → h = 2 :=
by
  sorry

end locust_population_doubling_time_l287_287477


namespace rectangle_diagonal_l287_287771

theorem rectangle_diagonal (k : ℕ) (h1 : 2 * (5 * k + 4 * k) = 72) : 
  (Real.sqrt ((5 * k) ^ 2 + (4 * k) ^ 2)) = Real.sqrt 656 :=
by
  sorry

end rectangle_diagonal_l287_287771


namespace find_prime_p_l287_287991

noncomputable def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem find_prime_p (p : ℕ) (hp : p.Prime) (hsquare : isPerfectSquare (5^p + 12^p)) : p = 2 := 
sorry

end find_prime_p_l287_287991


namespace inequality_for_positive_numbers_l287_287757

theorem inequality_for_positive_numbers (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : 
  (a + b) * (a^4 + b^4) ≥ (a^2 + b^2) * (a^3 + b^3) :=
sorry

end inequality_for_positive_numbers_l287_287757


namespace ratio_of_edges_l287_287881

noncomputable def cube_volume (edge : ℝ) : ℝ := edge^3

theorem ratio_of_edges 
  {a b : ℝ} 
  (h : cube_volume a / cube_volume b = 27) : 
  a / b = 3 :=
by
  sorry

end ratio_of_edges_l287_287881


namespace plates_added_before_topple_l287_287222

theorem plates_added_before_topple (init_plates add_first add_total : ℕ) (h : init_plates = 27) (h1 : add_first = 37) (h2 : add_total = 83) : 
  add_total - (init_plates + add_first) = 19 :=
by
  -- proof goes here
  sorry

end plates_added_before_topple_l287_287222


namespace greatest_divisor_sum_of_first_fifteen_terms_l287_287357

theorem greatest_divisor_sum_of_first_fifteen_terms 
  (x c : ℕ) (hx : x > 0) (hc : c > 0):
  ∃ d, d = 15 ∧ d ∣ (15*x + 105*c) :=
by
  existsi 15
  split
  . refl
  . apply Nat.dvd.intro
    existsi (x + 7*c)
    refl
  sorry

end greatest_divisor_sum_of_first_fifteen_terms_l287_287357


namespace contractor_initial_people_l287_287965

theorem contractor_initial_people (P : ℕ) (days_total days_done : ℕ) 
  (percent_done : ℚ) (additional_people : ℕ) (T : ℕ) :
  days_total = 50 →
  days_done = 25 →
  percent_done = 0.4 →
  additional_people = 90 →
  T = P + additional_people →
  (P : ℚ) * 62.5 = (T : ℚ) * 50 →
  P = 360 :=
by
  intros h_days_total h_days_done h_percent_done h_additional_people h_T h_eq
  sorry

end contractor_initial_people_l287_287965


namespace tank_capacity_l287_287520

-- Definitions from conditions
def initial_fraction := (1 : ℚ) / 4  -- The tank is 1/4 full initially
def added_amount := 5  -- Adding 5 liters

-- The proof problem to show that the tank's total capacity c equals 60 liters
theorem tank_capacity
  (c : ℚ)  -- The total capacity of the tank in liters
  (h1 : c / 4 + added_amount = c / 3)  -- Adding 5 liters makes the tank 1/3 full
  : c = 60 := 
sorry

end tank_capacity_l287_287520


namespace identical_lines_unique_pair_l287_287992

theorem identical_lines_unique_pair :
  ∃! (a b : ℚ), 2 * (0 : ℚ) + a * (0 : ℚ) + 10 = 0 ∧ b * (0 : ℚ) - 3 * (0 : ℚ) - 15 = 0 ∧ 
  (-2 / a = b / 3) ∧ (-10 / a = 5) :=
by {
  -- Given equations in slope-intercept form:
  -- y = -2 / a * x - 10 / a
  -- y = b / 3 * x + 5
  -- Slope and intercept comparison leads to equations:
  -- -2 / a = b / 3
  -- -10 / a = 5
  sorry
}

end identical_lines_unique_pair_l287_287992


namespace odd_nat_composite_iff_exists_a_l287_287308

theorem odd_nat_composite_iff_exists_a (c : ℕ) (h_odd : c % 2 = 1) :
  (∃ a : ℕ, a ≤ c / 3 - 1 ∧ ∃ k : ℕ, (2*a - 1)^2 + 8*c = k^2) ↔
  ∃ d : ℕ, ∃ e : ℕ, d > 1 ∧ e > 1 ∧ d * e = c := 
sorry

end odd_nat_composite_iff_exists_a_l287_287308


namespace count_triangles_l287_287072

-- Define the conditions for the problem
def P (x1 x2 : ℕ) : Prop := 37 * x1 ≤ 2022 ∧ 37 * x2 ≤ 2022

def valid_points (x y : ℕ) : Prop := 37 * x + y = 2022

def area_multiple_of_3 (x1 x2 : ℕ): Prop :=
  (∃ k : ℤ, 3 * k = x1 - x2) ∧ x1 ≠ x2 ∧ P x1 x2

-- The final theorem to prove the number of such distinct triangles
theorem count_triangles : 
  (∃ (n : ℕ), n = 459 ∧ 
    ∃ x1 x2 : ℕ, area_multiple_of_3 x1 x2 ∧ x1 ≠ x2) :=
by
  sorry

end count_triangles_l287_287072


namespace proposition_2_proposition_4_l287_287262

-- Definitions from conditions.
def circle_M (x y q : ℝ) : Prop := (x + Real.cos q)^2 + (y - Real.sin q)^2 = 1
def line_l (y k x : ℝ) : Prop := y = k * x

-- Prove that the line l and circle M always intersect for any real k and q.
theorem proposition_2 : ∀ (k q : ℝ), ∃ (x y : ℝ), circle_M x y q ∧ line_l y k x := sorry

-- Prove that for any real k, there exists a real q such that the line l is tangent to the circle M.
theorem proposition_4 : ∀ (k : ℝ), ∃ (q x y : ℝ), circle_M x y q ∧ line_l y k x ∧
  (abs (Real.sin q + k * Real.cos q) = 1 / Real.sqrt (1 + k^2)) := sorry

end proposition_2_proposition_4_l287_287262


namespace domain_of_function_l287_287470

theorem domain_of_function :
  (∀ x : ℝ, (2 * Real.sin x - 1 > 0) ∧ (1 - 2 * Real.cos x ≥ 0) ↔
    ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 3 ≤ x ∧ x < 2 * k * Real.pi + 5 * Real.pi / 6) :=
sorry

end domain_of_function_l287_287470


namespace correct_proposition_l287_287200

-- Definitions based on conditions
def p : Prop := ∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0
def not_p : Prop := ∀ x : ℝ, x^2 + 2 * x + 2016 > 0

-- Proof statement
theorem correct_proposition : p → not_p :=
by sorry

end correct_proposition_l287_287200


namespace find_a_l287_287098

theorem find_a (a : ℝ) (h_pos : a > 0)
  (h_eq : ∀ (f g : ℝ → ℝ), (f = λ x => x^2 + 10) → (g = λ x => x^2 - 6) → f (g a) = 14) :
  a = 2 * Real.sqrt 2 ∨ a = 2 :=
by 
  sorry

end find_a_l287_287098


namespace ball_bounce_height_l287_287208

theorem ball_bounce_height :
  ∃ k : ℕ, k = 4 ∧ 45 * (1 / 3 : ℝ) ^ k < 2 :=
by 
  use 4
  sorry

end ball_bounce_height_l287_287208


namespace non_congruent_triangles_with_perimeter_11_l287_287868

theorem non_congruent_triangles_with_perimeter_11 :
  ∃ (triangle_count : ℕ), 
    triangle_count = 3 ∧ 
    ∀ (a b c : ℕ), 
      a + b + c = 11 → 
      a + b > c ∧ b + c > a ∧ a + c > b → 
      ∃ (t₁ t₂ t₃ : (ℕ × ℕ × ℕ)),
        (t₁ = (2, 4, 5) ∨ t₁ = (3, 4, 4) ∨ t₁ = (3, 3, 5)) ∧ 
        (t₂ = (2, 4, 5) ∨ t₂ = (3, 4, 4) ∨ t₂ = (3, 3, 5)) ∧ 
        (t₃ = (2, 4, 5) ∨ t₃ = (3, 4, 4) ∨ t₃ = (3, 3, 5)) ∧
        t₁ ≠ t₂ ∧ t₂ ≠ t₃ ∧ t₁ ≠ t₃

end non_congruent_triangles_with_perimeter_11_l287_287868


namespace decomposition_of_cube_l287_287257

theorem decomposition_of_cube (m : ℕ) (h : m^2 - m + 1 = 73) : m = 9 :=
sorry

end decomposition_of_cube_l287_287257


namespace estimated_households_exceeding_320_l287_287592

noncomputable def community_electricity_consumption (n : ℕ) (mean stddev : ℝ) : ℕ :=
  let households := n
  let mu := mean
  let sigma := stddev
  let upper_bound := 320
  let probability_exceeding := (1 - 0.954) / 2
  let expected_households := households * probability_exceeding
  (expected_households).toNat

theorem estimated_households_exceeding_320 :
  community_electricity_consumption 1000 300 10 = 23 :=
sorry

end estimated_households_exceeding_320_l287_287592


namespace painting_time_l287_287680

noncomputable def bob_rate : ℕ := 120 / 8
noncomputable def alice_rate : ℕ := 150 / 10
noncomputable def combined_rate : ℕ := bob_rate + alice_rate
noncomputable def total_area : ℕ := 120 + 150
noncomputable def working_time : ℕ := total_area / combined_rate
noncomputable def lunch_break : ℕ := 1
noncomputable def total_time : ℕ := working_time + lunch_break

theorem painting_time : total_time = 10 := by
  -- Proof skipped
  sorry

end painting_time_l287_287680


namespace eric_bike_speed_l287_287699

def swim_distance : ℝ := 0.5
def swim_speed : ℝ := 1
def run_distance : ℝ := 2
def run_speed : ℝ := 8
def bike_distance : ℝ := 12
def total_time_limit : ℝ := 2

theorem eric_bike_speed :
  (swim_distance / swim_speed) + (run_distance / run_speed) + (bike_distance / (48/5)) < total_time_limit :=
by
  sorry

end eric_bike_speed_l287_287699


namespace find_pink_highlighters_l287_287884

def yellow_highlighters : ℕ := 7
def blue_highlighters : ℕ := 5
def total_highlighters : ℕ := 15

theorem find_pink_highlighters : (total_highlighters - (yellow_highlighters + blue_highlighters)) = 3 :=
by
  sorry

end find_pink_highlighters_l287_287884


namespace inequality_solution_set_l287_287019

theorem inequality_solution_set : 
  {x : ℝ | (x - 2) * (x + 1) ≤ 0} = {x : ℝ | -1 < x ∧ x ≤ 2} :=
by
  sorry

end inequality_solution_set_l287_287019


namespace patients_per_doctor_l287_287664

theorem patients_per_doctor (total_patients : ℕ) (total_doctors : ℕ) (h_patients : total_patients = 400) (h_doctors : total_doctors = 16) : 
  (total_patients / total_doctors) = 25 :=
by
  sorry

end patients_per_doctor_l287_287664


namespace complement_M_in_U_l287_287722

def M (x : ℝ) : Prop := 0 < x ∧ x < 2

def complement_M (x : ℝ) : Prop := x ≤ 0 ∨ x ≥ 2

theorem complement_M_in_U (x : ℝ) : ¬ M x ↔ complement_M x :=
by sorry

end complement_M_in_U_l287_287722


namespace find_h2_l287_287982

noncomputable def h (x : ℝ) : ℝ := 
  ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) - 1) / (x^15 - 1)

theorem find_h2 : h 2 = 2 :=
by 
  sorry

end find_h2_l287_287982


namespace ferry_tourists_total_l287_287390

def series_sum (a d n : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem ferry_tourists_total :
  let t_0 := 90
  let d := -2
  let n := 9
  series_sum t_0 d n = 738 :=
by
  sorry

end ferry_tourists_total_l287_287390


namespace ratio_h_r_bounds_l287_287650

theorem ratio_h_r_bounds
  {a b c h r : ℝ}
  (h_right_angle : a^2 + b^2 = c^2)
  (h_area1 : 1/2 * a * b = 1/2 * c * h)
  (h_area2 : 1/2 * (a + b + c) * r = 1/2 * a * b) :
  2 < h / r ∧ h / r ≤ 2.41 :=
by
  sorry

end ratio_h_r_bounds_l287_287650


namespace final_height_of_tree_in_4_months_l287_287910

-- Definitions based on the conditions
def growth_rate_cm_per_two_weeks : ℕ := 50
def current_height_meters : ℕ := 2
def weeks_per_month : ℕ := 4
def months : ℕ := 4
def cm_per_meter : ℕ := 100

-- The final height of the tree after 4 months in centimeters
theorem final_height_of_tree_in_4_months : 
  (current_height_meters * cm_per_meter) + 
  (((months * weeks_per_month) / 2) * growth_rate_cm_per_two_weeks) = 600 := 
by
  sorry

end final_height_of_tree_in_4_months_l287_287910


namespace MrJones_pants_count_l287_287612

theorem MrJones_pants_count (P : ℕ) (h1 : 6 * P + P = 280) : P = 40 := by
  sorry

end MrJones_pants_count_l287_287612


namespace bake_sale_money_made_l287_287067

theorem bake_sale_money_made :
  let clementine_cookies := 72
  let jake_cookies := 2 * clementine_cookies
  let combined_cookies := jake_cookies + clementine_cookies
  let tory_cookies := combined_cookies / 2
  let total_cookies := clementine_cookies + jake_cookies + tory_cookies
  let price_per_cookie := 2
  let total_money := total_cookies * price_per_cookie
  total_money = 648 :=
by
  let clementine_cookies := 72
  let jake_cookies := 2 * clementine_cookies
  let combined_cookies := jake_cookies + clementine_cookies
  let tory_cookies := combined_cookies / 2
  let total_cookies := clementine_cookies + jake_cookies + tory_cookies
  let price_per_cookie := 2
  let total_money := total_cookies * price_per_cookie
  sorry

end bake_sale_money_made_l287_287067


namespace total_instruments_correct_l287_287065

def numberOfFlutesCharlie : ℕ := 1
def numberOfHornsCharlie : ℕ := 2
def numberOfHarpsCharlie : ℕ := 1
def numberOfDrumsCharlie : ℕ := 5

def numberOfFlutesCarli : ℕ := 3 * numberOfFlutesCharlie
def numberOfHornsCarli : ℕ := numberOfHornsCharlie / 2
def numberOfDrumsCarli : ℕ := 2 * numberOfDrumsCharlie
def numberOfHarpsCarli : ℕ := 0

def numberOfFlutesNick : ℕ := 2 * numberOfFlutesCarli - 1
def numberOfHornsNick : ℕ := numberOfHornsCharlie + numberOfHornsCarli
def numberOfDrumsNick : ℕ := 4 * numberOfDrumsCarli - 2
def numberOfHarpsNick : ℕ := 0

def numberOfFlutesDaisy : ℕ := numberOfFlutesNick * numberOfFlutesNick
def numberOfHornsDaisy : ℕ := (numberOfHornsNick - numberOfHornsCarli) / 2
def numberOfDrumsDaisy : ℕ := (numberOfDrumsCharlie + numberOfDrumsCarli + numberOfDrumsNick) / 3
def numberOfHarpsDaisy : ℕ := numberOfHarpsCharlie

def numberOfInstrumentsCharlie : ℕ := numberOfFlutesCharlie + numberOfHornsCharlie + numberOfHarpsCharlie + numberOfDrumsCharlie
def numberOfInstrumentsCarli : ℕ := numberOfFlutesCarli + numberOfHornsCarli + numberOfDrumsCarli
def numberOfInstrumentsNick : ℕ := numberOfFlutesNick + numberOfHornsNick + numberOfDrumsNick
def numberOfInstrumentsDaisy : ℕ := numberOfFlutesDaisy + numberOfHornsDaisy + numberOfHarpsDaisy + numberOfDrumsDaisy

def totalInstruments : ℕ := numberOfInstrumentsCharlie + numberOfInstrumentsCarli + numberOfInstrumentsNick + numberOfInstrumentsDaisy

theorem total_instruments_correct : totalInstruments = 113 := by
  sorry

end total_instruments_correct_l287_287065


namespace value_of_expression_l287_287805

theorem value_of_expression :
  (10^2 - 10) / 9 = 10 :=
by
  sorry

end value_of_expression_l287_287805


namespace quadratic_has_distinct_real_roots_find_k_l287_287855

-- Part 1: Prove the quadratic equation always has two distinct real roots
theorem quadratic_has_distinct_real_roots (k : ℝ) : 
  let a := 1
  let b := 2 * k - 1
  let c := -k - 2
  let Δ := b^2 - 4 * a * c
  (Δ > 0) :=
by
  sorry

-- Part 2: Given the roots condition, find k
theorem find_k (x1 x2 k : ℝ)
  (h1 : x1 + x2 = -(2 * k - 1))
  (h2 : x1 * x2 = -k - 2)
  (h3 : x1 + x2 - 4 * x1 * x2 = 1) : 
  k = -4 :=
by
  sorry

end quadratic_has_distinct_real_roots_find_k_l287_287855


namespace new_volume_eq_7352_l287_287218

variable (l w h : ℝ)

-- Given conditions
def volume_eq : Prop := l * w * h = 5184
def surface_area_eq : Prop := l * w + w * h + h * l = 972
def edge_sum_eq : Prop := l + w + h = 54

-- Question: New volume when dimensions are increased by two inches
def new_volume : ℝ := (l + 2) * (w + 2) * (h + 2)

-- Correct Answer: Prove that the new volume equals 7352
theorem new_volume_eq_7352 (h_vol : volume_eq l w h) (h_surf : surface_area_eq l w h) (h_edge : edge_sum_eq l w h) 
    : new_volume l w h = 7352 :=
by
  -- Proof omitted
  sorry

#check new_volume_eq_7352

end new_volume_eq_7352_l287_287218


namespace shifted_parabola_passes_through_neg1_1_l287_287766

def original_parabola (x : ℝ) : ℝ := -(x + 1)^2 + 4

def shifted_parabola (x : ℝ) : ℝ := -(x - 1)^2 + 2

theorem shifted_parabola_passes_through_neg1_1 :
  shifted_parabola (-1) = 1 :=
by 
  -- Proof goes here
  sorry

end shifted_parabola_passes_through_neg1_1_l287_287766


namespace monotonically_increasing_interval_l287_287274

noncomputable def f (x : ℝ) : ℝ := Real.log (-3 * x^2 + 4 * x + 4)

theorem monotonically_increasing_interval :
  ∀ x, x ∈ Set.Ioc (-2/3 : ℝ) (2/3 : ℝ) → MonotoneOn f (Set.Ioc (-2/3) (2/3)) :=
sorry

end monotonically_increasing_interval_l287_287274


namespace zeros_of_f_l287_287326

def f (x : ℝ) : ℝ := (x^2 - 3 * x) * (x + 4)

theorem zeros_of_f : ∀ x, f x = 0 ↔ x = 0 ∨ x = 3 ∨ x = -4 := by
  sorry

end zeros_of_f_l287_287326


namespace at_least_three_bushes_with_same_number_of_flowers_l287_287138

-- Defining the problem using conditions as definitions.
theorem at_least_three_bushes_with_same_number_of_flowers (n : ℕ) (f : Fin n → ℕ) (h1 : n = 201)
  (h2 : ∀ (i : Fin n), 1 ≤ f i ∧ f i ≤ 100) : 
  ∃ (x : ℕ), (∃ (i1 i2 i3 : Fin n), i1 ≠ i2 ∧ i1 ≠ i3 ∧ i2 ≠ i3 ∧ f i1 = x ∧ f i2 = x ∧ f i3 = x) := 
by
  sorry

end at_least_three_bushes_with_same_number_of_flowers_l287_287138


namespace quadratic_general_form_l287_287984

theorem quadratic_general_form (x : ℝ) :
  x * (x + 2) = 5 * (x - 2) → x^2 - 3 * x - 10 = 0 := by
  sorry

end quadratic_general_form_l287_287984


namespace prism_diagonal_length_l287_287516

theorem prism_diagonal_length (x y z : ℝ) (h1 : 4 * x + 4 * y + 4 * z = 24) (h2 : 2 * x * y + 2 * x * z + 2 * y * z = 11) : Real.sqrt (x^2 + y^2 + z^2) = 5 :=
  by
  sorry

end prism_diagonal_length_l287_287516


namespace same_terminal_side_angle_l287_287004

theorem same_terminal_side_angle (k : ℤ) : 
  (∃ k : ℤ, - (π / 6) = 2 * k * π + a) → a = 11 * π / 6 :=
sorry

end same_terminal_side_angle_l287_287004


namespace total_wheels_combined_l287_287779

-- Define the counts of vehicles and wheels per vehicle in each storage area
def bicycles_A : ℕ := 16
def tricycles_A : ℕ := 7
def unicycles_A : ℕ := 10
def four_wheelers_A : ℕ := 5

def bicycles_B : ℕ := 12
def tricycles_B : ℕ := 5
def unicycles_B : ℕ := 8
def four_wheelers_B : ℕ := 3

def wheels_bicycle : ℕ := 2
def wheels_tricycle : ℕ := 3
def wheels_unicycle : ℕ := 1
def wheels_four_wheeler : ℕ := 4

-- Calculate total wheels in Storage Area A
def total_wheels_A : ℕ :=
  bicycles_A * wheels_bicycle + tricycles_A * wheels_tricycle + unicycles_A * wheels_unicycle + four_wheelers_A * wheels_four_wheeler
  
-- Calculate total wheels in Storage Area B
def total_wheels_B : ℕ :=
  bicycles_B * wheels_bicycle + tricycles_B * wheels_tricycle + unicycles_B * wheels_unicycle + four_wheelers_B * wheels_four_wheeler

-- Theorem stating that the combined total number of wheels in both storage areas is 142
theorem total_wheels_combined : total_wheels_A + total_wheels_B = 142 := by
  sorry

end total_wheels_combined_l287_287779


namespace intersection_of_M_and_N_l287_287124

def M : Set ℝ := {x | x ≥ 0 ∧ x < 16}
def N : Set ℝ := {x | x ≥ 1/3}

theorem intersection_of_M_and_N :
  M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by
  sorry

end intersection_of_M_and_N_l287_287124


namespace anna_score_correct_l287_287288

-- Given conditions
def correct_answers : ℕ := 17
def incorrect_answers : ℕ := 6
def unanswered_questions : ℕ := 7
def point_per_correct : ℕ := 1
def point_per_incorrect : ℕ := 0
def deduction_per_unanswered : ℤ := -1 / 2

-- Proving the score
theorem anna_score_correct : 
  correct_answers * point_per_correct + incorrect_answers * point_per_incorrect + unanswered_questions * deduction_per_unanswered = 27 / 2 :=
by
  sorry

end anna_score_correct_l287_287288


namespace average_of_numbers_eq_x_l287_287928

theorem average_of_numbers_eq_x (x : ℝ) (h : (2 + x + 10) / 3 = x) : x = 6 := 
by sorry

end average_of_numbers_eq_x_l287_287928


namespace larger_number_l287_287787

theorem larger_number (x y : ℝ) (h₁ : x + y = 45) (h₂ : x - y = 7) : x = 26 :=
by
  sorry

end larger_number_l287_287787


namespace min_value_of_sum_squares_l287_287418

theorem min_value_of_sum_squares (a b : ℝ) (h : (9 / a^2) + (4 / b^2) = 1) : a^2 + b^2 ≥ 25 :=
sorry

end min_value_of_sum_squares_l287_287418


namespace students_not_like_any_l287_287135

variables (F B P T F_cap_B F_cap_P F_cap_T B_cap_P B_cap_T P_cap_T F_cap_B_cap_P_cap_T : ℕ)

def total_students := 30

def students_like_F := 18
def students_like_B := 12
def students_like_P := 14
def students_like_T := 10

def students_like_F_and_B := 8
def students_like_F_and_P := 6
def students_like_F_and_T := 4
def students_like_B_and_P := 5
def students_like_B_and_T := 3
def students_like_P_and_T := 7

def students_like_all_four := 2

theorem students_not_like_any :
  total_students - ((students_like_F + students_like_B + students_like_P + students_like_T)
                    - (students_like_F_and_B + students_like_F_and_P + students_like_F_and_T
                      + students_like_B_and_P + students_like_B_and_T + students_like_P_and_T)
                    + students_like_all_four) = 11 :=
by sorry

end students_not_like_any_l287_287135


namespace heating_time_correct_l287_287601

def initial_temp : ℤ := 20

def desired_temp : ℤ := 100

def heating_rate : ℤ := 5

def time_to_heat (initial desired rate : ℤ) : ℤ :=
  (desired - initial) / rate

theorem heating_time_correct :
  time_to_heat initial_temp desired_temp heating_rate = 16 :=
by
  sorry

end heating_time_correct_l287_287601


namespace total_short_trees_after_planting_l287_287931

def initial_short_trees : ℕ := 31
def planted_short_trees : ℕ := 64

theorem total_short_trees_after_planting : initial_short_trees + planted_short_trees = 95 := by
  sorry

end total_short_trees_after_planting_l287_287931


namespace smallest_positive_angle_equivalent_neg_1990_l287_287478

theorem smallest_positive_angle_equivalent_neg_1990:
  ∃ k : ℤ, 0 ≤ (θ : ℤ) ∧ θ < 360 ∧ -1990 + 360 * k = θ := by
  use 6
  sorry

end smallest_positive_angle_equivalent_neg_1990_l287_287478


namespace calculate_expression_l287_287062

theorem calculate_expression : 
  (2^10 + (3^6 / 3^2)) = 1105 := 
by 
  -- Steps involve intermediate calculations
  -- for producing (2^10 = 1024), (3^6 = 729), (3^2 = 9)
  -- and then finding (729 / 9 = 81), (1024 + 81 = 1105)
  sorry

end calculate_expression_l287_287062


namespace algebraic_identity_l287_287088

theorem algebraic_identity (a : ℚ) (h : a + a⁻¹ = 3) : a^2 + a⁻¹^2 = 7 := 
  sorry

end algebraic_identity_l287_287088


namespace cubed_identity_l287_287282

theorem cubed_identity (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := 
sorry

end cubed_identity_l287_287282


namespace determine_x_value_l287_287247

theorem determine_x_value (a b c x : ℕ) (h1 : x = a + 7) (h2 : a = b + 12) (h3 : b = c + 25) (h4 : c = 95) : x = 139 := by
  sorry

end determine_x_value_l287_287247


namespace correct_quadratic_opens_upwards_l287_287395

-- Define the quadratic functions
def A (x : ℝ) : ℝ := 1 - x - 6 * x^2
def B (x : ℝ) : ℝ := -8 * x + x^2 + 1
def C (x : ℝ) : ℝ := (1 - x) * (x + 5)
def D (x : ℝ) : ℝ := 2 - (5 - x)^2

-- The theorem stating that function B is the one that opens upwards
theorem correct_quadratic_opens_upwards :
  ∃ (f : ℝ → ℝ) (h : f = B), ∀ (a b c : ℝ), f x = a * x^2 + b * x + c → a > 0 :=
sorry

end correct_quadratic_opens_upwards_l287_287395


namespace youngest_brother_age_difference_l287_287614

def Rick_age : ℕ := 15
def Oldest_brother_age : ℕ := 2 * Rick_age
def Middle_brother_age : ℕ := Oldest_brother_age / 3
def Smallest_brother_age : ℕ := Middle_brother_age / 2
def Youngest_brother_age : ℕ := 3

theorem youngest_brother_age_difference :
  Smallest_brother_age - Youngest_brother_age = 2 :=
by
  -- sorry to skip the proof
  sorry

end youngest_brother_age_difference_l287_287614


namespace mono_sum_eq_five_l287_287433

-- Conditions
def term1 (x y : ℝ) (m : ℕ) : ℝ := x^2 * y^m
def term2 (x y : ℝ) (n : ℕ) : ℝ := x^n * y^3

def is_monomial_sum (x y : ℝ) (m n : ℕ) : Prop :=
  term1 x y m + term2 x y n = x^(2:ℕ) * y^(3:ℕ)

-- Theorem stating the result
theorem mono_sum_eq_five (x y : ℝ) (m n : ℕ) (h : is_monomial_sum x y m n) : m + n = 5 :=
by
  sorry

end mono_sum_eq_five_l287_287433


namespace triangle_count_with_perimeter_11_l287_287861

theorem triangle_count_with_perimeter_11 :
  ∃ (s : Finset (ℕ × ℕ × ℕ)), s.card = 5 ∧ ∀ (a b c : ℕ), (a, b, c) ∈ s ->
    a ≤ b ∧ b ≤ c ∧ a + b + c = 11 ∧ a + b > c :=
sorry

end triangle_count_with_perimeter_11_l287_287861


namespace correct_evaluation_l287_287030

noncomputable def evaluate_expression : ℚ :=
  - (2 : ℚ) ^ 3 + (6 / 5) * (2 / 5)

theorem correct_evaluation : evaluate_expression = -7 - 13 / 25 :=
by
  unfold evaluate_expression
  sorry

end correct_evaluation_l287_287030


namespace calc_result_neg2xy2_pow3_l287_287979

theorem calc_result_neg2xy2_pow3 (x y : ℝ) : 
  (-2 * x * y^2)^3 = -8 * x^3 * y^6 := 
by 
  sorry

end calc_result_neg2xy2_pow3_l287_287979


namespace two_integer_solutions_iff_l287_287946

theorem two_integer_solutions_iff (a : ℝ) :
  (∃ (n m : ℤ), n ≠ m ∧ |n - 1| < a * n ∧ |m - 1| < a * m ∧
    ∀ (k : ℤ), |k - 1| < a * k → k = n ∨ k = m) ↔
  (1/2 : ℝ) < a ∧ a ≤ (2/3 : ℝ) :=
by
  sorry

end two_integer_solutions_iff_l287_287946


namespace ratio_of_width_to_length_l287_287499

theorem ratio_of_width_to_length (w l : ℕ) (h1 : w * l = 800) (h2 : l - w = 20) : w / l = 1 / 2 :=
by sorry

end ratio_of_width_to_length_l287_287499


namespace intersection_of_sets_l287_287129

theorem intersection_of_sets :
  let M := { x : ℝ | 0 ≤ x ∧ x < 16 }
  let N := { x : ℝ | x ≥ 1/3 }
  M ∩ N = { x : ℝ | 1/3 ≤ x ∧ x < 16 } :=
by
  sorry

end intersection_of_sets_l287_287129


namespace cubed_identity_l287_287279

variable (x : ℝ)

theorem cubed_identity (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := 
by
  sorry

end cubed_identity_l287_287279


namespace evaluate_star_l287_287090

-- Define the operation c star d
def star (c d : ℤ) : ℤ := c^2 - 2 * c * d + d^2

-- State the theorem to prove the given problem
theorem evaluate_star : (star 3 5) = 4 := by
  sorry

end evaluate_star_l287_287090


namespace greatest_integer_l287_287310

theorem greatest_integer (m : ℕ) (h1 : 0 < m) (h2 : m < 150)
  (h3 : ∃ a : ℤ, m = 9 * a - 2) (h4 : ∃ b : ℤ, m = 5 * b + 4) :
  m = 124 := 
sorry

end greatest_integer_l287_287310


namespace Rebecca_tips_calculation_l287_287759

def price_haircut : ℤ := 30
def price_perm : ℤ := 40
def price_dye_job : ℤ := 60
def cost_hair_dye_box : ℤ := 10
def num_haircuts : ℕ := 4
def num_perms : ℕ := 1
def num_dye_jobs : ℕ := 2
def total_end_day : ℤ := 310

noncomputable def total_service_earnings : ℤ := 
  num_haircuts * price_haircut + num_perms * price_perm + num_dye_jobs * price_dye_job

noncomputable def total_hair_dye_cost : ℤ := 
  num_dye_jobs * cost_hair_dye_box

noncomputable def earnings_after_cost : ℤ := 
  total_service_earnings - total_hair_dye_cost

noncomputable def tips : ℤ := 
  total_end_day - earnings_after_cost

theorem Rebecca_tips_calculation : tips = 50 := by
  sorry

end Rebecca_tips_calculation_l287_287759


namespace number_of_5card_hands_with_4_of_a_kind_l287_287814

-- Definitions based on the given conditions
def deck_size : Nat := 52
def num_values : Nat := 13
def suits_per_value : Nat := 4

-- The function to count the number of 5-card hands with exactly four cards of the same value
def count_hands_with_four_of_a_kind : Nat :=
  num_values * (deck_size - suits_per_value)

-- Proof statement
theorem number_of_5card_hands_with_4_of_a_kind : count_hands_with_four_of_a_kind = 624 :=
by
  -- Steps to show the computation results may be added here
  -- We use the formula: 13 * (52 - 4)
  sorry

end number_of_5card_hands_with_4_of_a_kind_l287_287814


namespace scientific_notation_example_l287_287599

theorem scientific_notation_example :
  110000 = 1.1 * 10^5 :=
by {
  sorry
}

end scientific_notation_example_l287_287599


namespace polygon_interior_exterior_relation_l287_287324

theorem polygon_interior_exterior_relation :
  ∃ n : ℕ, (n ≥ 3) ∧ ((n - 2) * 180 = 2 * 360) → n = 6 :=
begin
  sorry
end

end polygon_interior_exterior_relation_l287_287324


namespace percentage_decrease_last_year_l287_287631

-- Define the percentage decrease last year
variable (x : ℝ)

-- Define the condition that expresses the stock price this year
def final_price_change (x : ℝ) : Prop :=
  (1 - x / 100) * 1.10 = 1 + 4.499999999999993 / 100

-- Theorem stating the percentage decrease
theorem percentage_decrease_last_year : final_price_change 5 := by
  sorry

end percentage_decrease_last_year_l287_287631


namespace rectangle_diagonal_length_l287_287770

theorem rectangle_diagonal_length (P L W : ℝ) (hP : P = 72) (hRatio : 5 * W = 4 * L) :
  ∃ d, d = 4 * real.sqrt 41 ∧ (d = real.sqrt (L^2 + W^2)) :=
by
  -- Define the values of k
  let k := P / (2 * (5 + 4))  -- Using the perimeter relation
  let L := 5 * k  -- Length
  let W := 4 * k  -- Width
  -- Define the diagonal
  let d := real.sqrt (L^2 + W^2)
  -- Expected answer
  existsi 4 * real.sqrt 41
  -- Demonstrate they are equal
  have : d = 4 * real.sqrt 41 := sorry
  exact ⟨rfl, this⟩

end rectangle_diagonal_length_l287_287770


namespace greatest_divisor_of_arithmetic_sequence_sum_l287_287344

theorem greatest_divisor_of_arithmetic_sequence_sum :
  ∀ (x c : ℕ), ∃ k : ℕ, k = 15 ∧ 15 ∣ (15 * x + 105 * c) :=
by
  intro x c
  exists 15
  split
  . rfl
  . sorry

end greatest_divisor_of_arithmetic_sequence_sum_l287_287344


namespace function_increasing_on_interval_l287_287555

theorem function_increasing_on_interval :
  ∀ x : ℝ, (1 / 2 < x) → (x > 0) → (8 * x - 1 / (x^2)) > 0 :=
sorry

end function_increasing_on_interval_l287_287555


namespace minimum_paper_toys_is_eight_l287_287613

noncomputable def minimum_paper_toys (s_boats: ℕ) (s_planes: ℕ) : ℕ :=
  s_boats * 8 + s_planes * 6

theorem minimum_paper_toys_is_eight :
  ∀ (s_boats s_planes : ℕ), s_boats >= 1 → minimum_paper_toys s_boats s_planes = 8 → s_planes = 0 :=
by
  intros s_boats s_planes h_boats h_eq
  have h1: s_boats * 8 + s_planes * 6 = 8 := h_eq
  sorry

end minimum_paper_toys_is_eight_l287_287613


namespace sin_double_angle_subst_l287_287998

open Real

theorem sin_double_angle_subst 
  (α : ℝ)
  (h : sin (α + π / 6) = -1 / 3) :
  sin (2 * α - π / 6) = -7 / 9 := 
by
  sorry

end sin_double_angle_subst_l287_287998


namespace temperature_drop_l287_287493

-- Define the initial temperature and the drop in temperature
def initial_temperature : ℤ := -6
def drop : ℤ := 5

-- Define the resulting temperature after the drop
def resulting_temperature : ℤ := initial_temperature - drop

-- The theorem to be proved
theorem temperature_drop : resulting_temperature = -11 :=
by
  sorry

end temperature_drop_l287_287493


namespace point_in_second_quadrant_l287_287118

variable (m : ℝ)

/-- 
If point P(m-1, 3) is in the second quadrant, 
then a possible value of m is -1
--/
theorem point_in_second_quadrant (h1 : (m - 1 < 0)) : m = -1 :=
by sorry

end point_in_second_quadrant_l287_287118


namespace money_per_postcard_l287_287644

def postcards_per_day : ℕ := 30
def days : ℕ := 6
def total_earning : ℕ := 900
def total_postcards := postcards_per_day * days
def price_per_postcard := total_earning / total_postcards

theorem money_per_postcard :
  price_per_postcard = 5 := 
sorry

end money_per_postcard_l287_287644


namespace original_selling_price_l287_287801

variable (P : ℝ)
variable (S : ℝ) 

-- Conditions
axiom profit_10_percent : S = 1.10 * P
axiom profit_diff : 1.17 * P - S = 42

-- Goal
theorem original_selling_price : S = 660 := by
  sorry

end original_selling_price_l287_287801


namespace problem1_problem2_l287_287299

-- Define the required conditions
variables {a b : ℤ}
-- Conditions
axiom h1 : a ≥ 1
axiom h2 : b ≥ 1

-- Proof statement for question 1
theorem problem1 : ¬ (a ∣ b^2 ↔ a ∣ b) := by
  sorry

-- Proof statement for question 2
theorem problem2 : (a^2 ∣ b^2 ↔ a ∣ b) := by
  sorry

end problem1_problem2_l287_287299


namespace gravel_amount_l287_287964

theorem gravel_amount (total_material sand gravel : ℝ) 
  (h1 : total_material = 14.02) 
  (h2 : sand = 8.11) 
  (h3 : gravel = total_material - sand) : 
  gravel = 5.91 :=
  sorry

end gravel_amount_l287_287964


namespace pelican_count_in_shark_bite_cove_l287_287407

theorem pelican_count_in_shark_bite_cove
  (num_sharks_pelican_bay : ℕ)
  (num_pelicans_shark_bite_cove : ℕ)
  (num_pelicans_moved : ℕ) :
  num_sharks_pelican_bay = 60 →
  num_sharks_pelican_bay = 2 * num_pelicans_shark_bite_cove →
  num_pelicans_moved = num_pelicans_shark_bite_cove / 3 →
  num_pelicans_shark_bite_cove - num_pelicans_moved = 20 :=
by
  sorry

end pelican_count_in_shark_bite_cove_l287_287407


namespace length_of_first_train_is_140_l287_287645

theorem length_of_first_train_is_140 
  (speed1 : ℝ) (speed2 : ℝ) (time_to_cross : ℝ) (length2 : ℝ) 
  (h1 : speed1 = 60) 
  (h2 : speed2 = 40) 
  (h3 : time_to_cross = 12.239020878329734) 
  (h4 : length2 = 200) : 
  ∃ (length1 : ℝ), length1 = 140 := 
by
  sorry

end length_of_first_train_is_140_l287_287645


namespace smallest_k_DIVISIBLE_by_3_67_l287_287704

theorem smallest_k_DIVISIBLE_by_3_67 :
  ∃ k : ℕ, (∀ n : ℕ, (2016^k % 3^67 = 0 ∧ (2016^n % 3^67 = 0 → k ≤ n)) ∧ k = 34) := by
  sorry

end smallest_k_DIVISIBLE_by_3_67_l287_287704


namespace centroid_of_triangle_l287_287012

theorem centroid_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) :
  let x_centroid := (x1 + x2 + x3) / 3
  let y_centroid := (y1 + y2 + y3) / 3
  (x_centroid, y_centroid) = (1/3 * (x1 + x2 + x3), 1/3 * (y1 + y2 + y3)) :=
by
  sorry

end centroid_of_triangle_l287_287012


namespace larger_number_is_26_l287_287786

theorem larger_number_is_26 {x y : ℤ} 
  (h1 : x + y = 45) 
  (h2 : x - y = 7) : 
  max x y = 26 :=
by
  sorry

end larger_number_is_26_l287_287786


namespace geometric_sequence_nec_not_suff_l287_287571

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n ≠ 0 → (a (n + 1) / a n) = (a (n + 2) / a (n + 1))

noncomputable def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 3) = a (n + 1) * a (n + 2)

theorem geometric_sequence_nec_not_suff (a : ℕ → ℝ) (hn : ∀ n : ℕ, a n ≠ 0) : 
  (is_geometric_sequence a → satisfies_condition a) ∧ ¬(satisfies_condition a → is_geometric_sequence a) :=
by
  sorry

end geometric_sequence_nec_not_suff_l287_287571


namespace prob_club_then_diamond_then_heart_l287_287641

noncomputable def prob_first_card_club := 13 / 52
noncomputable def prob_second_card_diamond_given_first_club := 13 / 51
noncomputable def prob_third_card_heart_given_first_club_second_diamond := 13 / 50

noncomputable def overall_probability := 
  prob_first_card_club * 
  prob_second_card_diamond_given_first_club * 
  prob_third_card_heart_given_first_club_second_diamond

theorem prob_club_then_diamond_then_heart :
  overall_probability = 2197 / 132600 :=
by
  sorry

end prob_club_then_diamond_then_heart_l287_287641


namespace triangle_C_squared_eq_b_a_plus_b_l287_287417

variables {A B C a b : ℝ}

theorem triangle_C_squared_eq_b_a_plus_b
  (h1 : C = 2 * B)
  (h2 : A ≠ B) :
  C^2 = b * (a + b) :=
sorry

end triangle_C_squared_eq_b_a_plus_b_l287_287417


namespace value_of_b_plus_c_l287_287411

theorem value_of_b_plus_c 
  (b c : ℝ) 
  (f : ℝ → ℝ)
  (h_def : ∀ x, f x = x^2 + 2 * b * x + c)
  (h_solution_set : ∀ x, f x ≤ 0 ↔ -1 ≤ x ∧ x ≤ 1) :
  b + c = -1 :=
sorry

end value_of_b_plus_c_l287_287411


namespace sum_of_areas_of_sixteen_disks_l287_287167

theorem sum_of_areas_of_sixteen_disks :
  let r := 1 - (2:ℝ).sqrt
  let area_one_disk := r^2 * Real.pi
  let total_area := 16 * area_one_disk
  total_area = Real.pi * (48 - 32 * (2:ℝ).sqrt) :=
by
  sorry

end sum_of_areas_of_sixteen_disks_l287_287167


namespace person_B_spheres_needed_l287_287751

-- Translate conditions to Lean definitions
def sum_squares (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6
def sum_triangulars (m : ℕ) : ℕ := (m * (m + 1) * (m + 2)) / 6

-- Define the main theorem
theorem person_B_spheres_needed (n m : ℕ) (hA : sum_squares n = 2109)
    (hB : m ≥ 25) : sum_triangulars m = 2925 :=
    sorry

end person_B_spheres_needed_l287_287751


namespace alice_additional_plates_l287_287221

theorem alice_additional_plates (initial_stack : ℕ) (first_addition : ℕ) (total_when_crashed : ℕ) 
  (h1 : initial_stack = 27) (h2 : first_addition = 37) (h3 : total_when_crashed = 83) : 
  total_when_crashed - (initial_stack + first_addition) = 19 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end alice_additional_plates_l287_287221


namespace molecular_weight_calculation_l287_287231

def molecular_weight (n_Ar n_Si n_H n_O : ℕ) (w_Ar w_Si w_H w_O : ℝ) : ℝ :=
  n_Ar * w_Ar + n_Si * w_Si + n_H * w_H + n_O * w_O

theorem molecular_weight_calculation :
  molecular_weight 2 3 12 8 39.948 28.085 1.008 15.999 = 304.239 :=
by
  sorry

end molecular_weight_calculation_l287_287231


namespace cars_needed_to_double_march_earnings_l287_287031

-- Definition of given conditions
def base_salary : Nat := 1000
def commission_per_car : Nat := 200
def march_earnings : Nat := 2000

-- Question to prove
theorem cars_needed_to_double_march_earnings : 
  (2 * march_earnings - base_salary) / commission_per_car = 15 := 
by sorry

end cars_needed_to_double_march_earnings_l287_287031


namespace num_perfect_squares_diff_consecutive_under_20000_l287_287695

theorem num_perfect_squares_diff_consecutive_under_20000 : 
  ∃ n, n = 71 ∧ ∀ a, a ^ 2 < 20000 → ∃ b, a ^ 2 = (b + 1) ^ 2 - b ^ 2 ↔ a ^ 2 % 2 = 1 :=
by
  sorry

end num_perfect_squares_diff_consecutive_under_20000_l287_287695


namespace product_of_two_numbers_l287_287325

-- Define the conditions
def two_numbers (x y : ℝ) : Prop :=
  x + y = 27 ∧ x - y = 7

-- Define the product function
def product_two_numbers (x y : ℝ) : ℝ := x * y

-- State the theorem
theorem product_of_two_numbers : ∃ x y : ℝ, two_numbers x y ∧ product_two_numbers x y = 170 := by
  sorry

end product_of_two_numbers_l287_287325


namespace find_a_l287_287879

variable {x y a : ℤ}

theorem find_a (h1 : 3 * x + y = 1 + 3 * a) (h2 : x + 3 * y = 1 - a) (h3 : x + y = 0) : a = -1 := 
sorry

end find_a_l287_287879


namespace cucumbers_after_purchase_l287_287199

theorem cucumbers_after_purchase (C U : ℕ) (h1 : C + U = 10) (h2 : C = 4) : U + 2 = 8 := by
  sorry

end cucumbers_after_purchase_l287_287199


namespace larger_number_is_26_l287_287785

theorem larger_number_is_26 {x y : ℤ} 
  (h1 : x + y = 45) 
  (h2 : x - y = 7) : 
  max x y = 26 :=
by
  sorry

end larger_number_is_26_l287_287785


namespace total_number_of_flags_is_12_l287_287244

def number_of_flags : Nat :=
  3 * 2 * 2

theorem total_number_of_flags_is_12 : number_of_flags = 12 := by
  sorry

end total_number_of_flags_is_12_l287_287244


namespace cos_triple_angle_l287_287456

theorem cos_triple_angle (x θ : ℝ) (h : x = Real.cos θ) : Real.cos (3 * θ) = 4 * x^3 - 3 * x :=
by
  sorry

end cos_triple_angle_l287_287456


namespace aqua_park_earnings_l287_287538

def admission_cost : ℕ := 12
def tour_cost : ℕ := 6
def group1_size : ℕ := 10
def group2_size : ℕ := 5

theorem aqua_park_earnings :
  (group1_size * admission_cost + group1_size * tour_cost) + (group2_size * admission_cost) = 240 :=
by
  sorry

end aqua_park_earnings_l287_287538


namespace smallest_collected_l287_287798

noncomputable def Yoongi_collections : ℕ := 4
noncomputable def Jungkook_collections : ℕ := 6 / 3
noncomputable def Yuna_collections : ℕ := 5

theorem smallest_collected : min (min Yoongi_collections Jungkook_collections) Yuna_collections = 2 :=
by
  sorry

end smallest_collected_l287_287798


namespace decimal_to_binary_25_l287_287475

theorem decimal_to_binary_25: (1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 0 * 2^1 + 1 * 2^0) = 25 :=
by 
  sorry

end decimal_to_binary_25_l287_287475


namespace arithmetic_sequence_sum_l287_287444

variable (a : ℕ → ℝ)
variable (d : ℝ)

noncomputable def arithmetic_sequence := ∀ n : ℕ, a n = a 0 + n * d

theorem arithmetic_sequence_sum (h₁ : a 1 + a 2 = 3) (h₂ : a 3 + a 4 = 5) :
  a 7 + a 8 = 9 :=
by
  sorry

end arithmetic_sequence_sum_l287_287444


namespace systematic_sampling_example_l287_287997

theorem systematic_sampling_example :
  ∃ (selected : Finset ℕ), 
    selected = {10, 30, 50, 70, 90} ∧
    ∀ n ∈ selected, 1 ≤ n ∧ n ≤ 100 ∧ 
    (∃ k, k > 0 ∧ k * 20 - 10∈ selected ∧ k * 20 - 10 ∈ Finset.range 101) := 
by
  sorry

end systematic_sampling_example_l287_287997


namespace unique_real_solution_k_l287_287373

theorem unique_real_solution_k (k : ℝ) :
  ∃! x : ℝ, (3 * x + 8) * (x - 6) = -62 + k * x ↔ k = -10 + 12 * Real.sqrt 1.5 ∨ k = -10 - 12 * Real.sqrt 1.5 := by
  sorry

end unique_real_solution_k_l287_287373


namespace budget_spent_on_salaries_l287_287518

theorem budget_spent_on_salaries :
  ∀ (B R U E S T : ℕ),
  R = 9 ∧
  U = 5 ∧
  E = 4 ∧
  S = 2 ∧
  T = (72 * 100) / 360 → 
  B = 100 →
  (B - (R + U + E + S + T)) = 60 :=
by sorry

end budget_spent_on_salaries_l287_287518


namespace cubed_identity_l287_287280

variable (x : ℝ)

theorem cubed_identity (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := 
by
  sorry

end cubed_identity_l287_287280


namespace two_circles_with_tangents_l287_287172

theorem two_circles_with_tangents
  (a b : ℝ)                -- radii of the circles
  (length_PQ length_AB : ℝ) -- lengths of the tangents PQ and AB
  (h1 : length_PQ = 14)     -- condition: length of PQ is 14
  (h2 : length_AB = 16)     -- condition: length of AB is 16
  (h3 : length_AB^2 + (a - b)^2 = length_PQ^2 + (a + b)^2) -- from the Pythagorean theorem
  : a * b = 15 := 
sorry

end two_circles_with_tangents_l287_287172


namespace digit_sum_is_14_l287_287445

theorem digit_sum_is_14 (P Q R S T : ℕ) 
  (h1 : P = 1)
  (h2 : Q = 0)
  (h3 : R = 2)
  (h4 : S = 5)
  (h5 : T = 6) :
  P + Q + R + S + T = 14 :=
by 
  sorry

end digit_sum_is_14_l287_287445


namespace greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l287_287368

theorem greatest_divisor_of_sum_first_15_terms_arithmetic_sequence
  (x c : ℕ) -- where x and c are positive integers
  (h_pos_x : 0 < x) -- x is positive
  (h_pos_c : 0 < c) -- c is positive
  : ∃ (d : ℕ), d = 15 ∧ ∀ (S : ℕ), S = 15 * x + 105 * c → d ∣ S :=
by
  sorry

end greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l287_287368


namespace find_smallest_number_l287_287707

theorem find_smallest_number
  (a1 a2 a3 a4 : ℕ)
  (h1 : (a1 + a2 + a3 + a4) / 4 = 30)
  (h2 : a2 = 28)
  (h3 : a2 = 35 - 7) :
  a1 = 27 :=
sorry

end find_smallest_number_l287_287707


namespace range_of_independent_variable_l287_287773

theorem range_of_independent_variable (x : ℝ) (h : 2 - x ≥ 0) : x ≤ 2 :=
sorry

end range_of_independent_variable_l287_287773


namespace line_eq_45_deg_y_intercept_2_circle_eq_center_neg2_3_tangent_yaxis_l287_287627

theorem line_eq_45_deg_y_intercept_2 :
  (∃ l : ℝ → ℝ, (l 0 = 2) ∧ (∀ x, l x = x + 2)) := sorry

theorem circle_eq_center_neg2_3_tangent_yaxis :
  (∃ c : ℝ × ℝ → ℝ, (c (-2, 3) = 0) ∧ (∀ x y, c (x, y) = (x + 2)^2 + (y - 3)^2 - 4)) := sorry

end line_eq_45_deg_y_intercept_2_circle_eq_center_neg2_3_tangent_yaxis_l287_287627


namespace second_crane_height_l287_287690

noncomputable def height_of_second_crane : ℝ :=
  let crane1 := 228
  let building1 := 200
  let building2 := 100
  let crane3 := 147
  let building3 := 140
  let avg_building_height := (building1 + building2 + building3) / 3
  let avg_crane_height := avg_building_height * 1.13
  let h := (avg_crane_height * 3) - (crane1 - building1 + crane3 - building3) + building2
  h

theorem second_crane_height : height_of_second_crane = 122 := 
  sorry

end second_crane_height_l287_287690


namespace negation_of_existence_l287_287016

theorem negation_of_existence :
  (¬ ∃ x : ℝ, x^2 + 2 * x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2 * x + 5 ≠ 0) :=
sorry

end negation_of_existence_l287_287016


namespace min_value_of_quadratic_l287_287193

theorem min_value_of_quadratic :
  ∃ x : ℝ, ∀ y : ℝ, y = 4 * x^2 + 8 * x + 16 → y ≥ 12 :=
by
  let f : ℝ → ℝ := λ x, 4 * x^2 + 8 * x + 16
  use -1
  intros y hy
  have h : f(-1) = 12 := by
    calc
      f(-1) = 4 * (-1)^2 + 8 * (-1) + 16 : by rfl
         ... = 4 * 1 - 8 + 16 : by norm_num
         ... = 4 - 8 + 16 : by rfl
         ... = -4 + 16 : by norm_num
         ... = 12 : by norm_num
  rw [← hy, h]
  have non_neg : 4 * (x + 1)^2 ≥ 0 := by
    apply mul_nonneg
    norm_num
    apply pow_two_nonneg
  exact add_le_add non_neg (by norm_num)

end min_value_of_quadratic_l287_287193


namespace factor_1024_into_three_factors_l287_287891

theorem factor_1024_into_three_factors :
  ∃ (factors : Finset (Finset ℕ)), factors.card = 14 ∧
  ∀ f ∈ factors, ∃ a b c : ℕ, a + b + c = 10 ∧ a ≥ b ∧ b ≥ c ∧ (2 ^ a) * (2 ^ b) * (2 ^ c) = 1024 :=
sorry

end factor_1024_into_three_factors_l287_287891


namespace library_charge_l287_287661

-- Definitions according to given conditions
def daily_charge : ℝ := 0.5
def days_in_may : ℕ := 31
def days_borrowed1 : ℕ := 20
def days_borrowed2 : ℕ := 31

-- Calculation of total charge
theorem library_charge :
  let total_charge := (daily_charge * days_borrowed1) + (2 * daily_charge * days_borrowed2)
  total_charge = 41 :=
by
  sorry

end library_charge_l287_287661


namespace cost_of_largest_pot_equals_229_l287_287803

-- Define the conditions
variables (total_cost : ℝ) (num_pots : ℕ) (cost_diff : ℝ)

-- Assume given conditions
axiom h1 : num_pots = 6
axiom h2 : total_cost = 8.25
axiom h3 : cost_diff = 0.3

-- Define the function for the cost of the smallest pot and largest pot
noncomputable def smallest_pot_cost : ℝ :=
  (total_cost - (num_pots - 1) * cost_diff) / num_pots

noncomputable def largest_pot_cost : ℝ :=
  smallest_pot_cost total_cost num_pots cost_diff + (num_pots - 1) * cost_diff

-- Prove the cost of the largest pot equals 2.29
theorem cost_of_largest_pot_equals_229 (h1 : num_pots = 6) (h2 : total_cost = 8.25) (h3 : cost_diff = 0.3) :
  largest_pot_cost total_cost num_pots cost_diff = 2.29 :=
  by sorry

end cost_of_largest_pot_equals_229_l287_287803


namespace greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l287_287366

theorem greatest_divisor_of_sum_first_15_terms_arithmetic_sequence
  (x c : ℕ) -- where x and c are positive integers
  (h_pos_x : 0 < x) -- x is positive
  (h_pos_c : 0 < c) -- c is positive
  : ∃ (d : ℕ), d = 15 ∧ ∀ (S : ℕ), S = 15 * x + 105 * c → d ∣ S :=
by
  sorry

end greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l287_287366


namespace math_problem_l287_287889

noncomputable def canA_red_balls := 3
noncomputable def canA_black_balls := 4
noncomputable def canB_red_balls := 2
noncomputable def canB_black_balls := 3

noncomputable def prob_event_A := canA_red_balls / (canA_red_balls + canA_black_balls) -- P(A)
noncomputable def prob_event_B := 
  (canA_red_balls / (canA_red_balls + canA_black_balls)) * (canB_red_balls + 1) / (6) +
  (canA_black_balls / (canA_red_balls + canA_black_balls)) * (canB_red_balls) / (6) -- P(B)

theorem math_problem : 
  (prob_event_A = 3 / 7) ∧ 
  (prob_event_B = 17 / 42) ∧
  (¬ (prob_event_A * prob_event_B = (3 / 7) * (17 / 42))) ∧
  ((prob_event_A * (canB_red_balls + 1) / 6) / prob_event_A = 1 / 2) := by
  repeat { sorry }

end math_problem_l287_287889


namespace pants_cost_correct_l287_287747

def shirt_cost : ℕ := 43
def tie_cost : ℕ := 15
def total_paid : ℕ := 200
def change_received : ℕ := 2

def total_spent : ℕ := total_paid - change_received
def combined_cost : ℕ := shirt_cost + tie_cost
def pants_cost : ℕ := total_spent - combined_cost

theorem pants_cost_correct : pants_cost = 140 :=
by
  -- We'll leave the proof as an exercise.
  sorry

end pants_cost_correct_l287_287747


namespace sale_in_fifth_month_l287_287215

def sales_month_1 := 6635
def sales_month_2 := 6927
def sales_month_3 := 6855
def sales_month_4 := 7230
def sales_month_6 := 4791
def target_average := 6500
def number_of_months := 6

def total_sales := sales_month_1 + sales_month_2 + sales_month_3 + sales_month_4 + sales_month_6

theorem sale_in_fifth_month :
  (target_average * number_of_months) - total_sales = 6562 :=
by
  sorry

end sale_in_fifth_month_l287_287215


namespace pascal_triangle_row_20_sum_l287_287686

theorem pascal_triangle_row_20_sum :
  (Nat.choose 20 2) + (Nat.choose 20 3) + (Nat.choose 20 4) = 6175 :=
by
  sorry

end pascal_triangle_row_20_sum_l287_287686


namespace find_c_for_radius_6_l287_287697

-- Define the circle equation and the radius condition.
theorem find_c_for_radius_6 (c : ℝ) :
  (∃ (x y : ℝ), x^2 + 8 * x + y^2 + 2 * y + c = 0) ∧ 6 = 6 -> c = -19 := 
by
  sorry

end find_c_for_radius_6_l287_287697


namespace problem1_problem2_l287_287063

theorem problem1 (x y : ℝ) : (x + y) * (x - y) + y * (y - 2) = x^2 - 2 * y :=
by 
  sorry

theorem problem2 (m : ℝ) (h : m ≠ 2) : (1 - m / (m + 2)) / ((m^2 - 4 * m + 4) / (m^2 - 4)) = 2 / (m - 2) :=
by 
  sorry

end problem1_problem2_l287_287063


namespace probability_white_balls_le_1_l287_287841

-- Definitions and conditions
def total_balls : ℕ := 6
def red_balls : ℕ := 4
def white_balls : ℕ := 2
def selected_balls : ℕ := 3

-- Combinatorial computations
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Calculations based on the conditions
def total_combinations : ℕ := C total_balls selected_balls
def red_combinations : ℕ := C red_balls selected_balls
def white_combinations : ℕ := C white_balls 1 * C red_balls 2

-- Probability calculations
def P_xi_le_1 : ℚ :=
  (red_combinations / total_combinations : ℚ) +
  (white_combinations / total_combinations : ℚ)

-- Problem statement: Prove that the calculated probability is 4/5
theorem probability_white_balls_le_1 : P_xi_le_1 = 4 / 5 := 
  sorry

end probability_white_balls_le_1_l287_287841


namespace breadth_of_boat_l287_287953

theorem breadth_of_boat :
  ∀ (L : ℝ) (h : ℝ) (m : ℝ) (g : ℝ) (rho : ℝ),
    L = 8 → h = 0.01 → m = 160 → g = 9.81 → rho = 1000 →
    (L * 2 * h = (m * g) / (rho * g)) :=
by
  intros L h m g rho hL hh hm hg hrho
  sorry

end breadth_of_boat_l287_287953


namespace ratio_a_to_b_l287_287317

variable (a x c d b : ℝ)
variable (h1 : d = 3 * x + c)
variable (h2 : b = 4 * x)

theorem ratio_a_to_b : a / b = -1 / 4 := by 
  sorry

end ratio_a_to_b_l287_287317


namespace find_prime_pairs_l287_287563

theorem find_prime_pairs (p q n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hn : 0 < n) :
  p * (p + 1) + q * (q + 1) = n * (n + 1) ↔ (p = 3 ∧ q = 5 ∧ n = 6) ∨ (p = 5 ∧ q = 3 ∧ n = 6) ∨ (p = 2 ∧ q = 2 ∧ n = 3) :=
by
  sorry

end find_prime_pairs_l287_287563


namespace greatest_common_divisor_sum_arithmetic_sequence_l287_287335

theorem greatest_common_divisor_sum_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, d = 15 ∧ ∀ (n : ℕ), n = 15 → ∀ k : ℕ, k = 15 ∧ 15 ∣ (15 * (x + 7 * c)) :=
by
  sorry

end greatest_common_divisor_sum_arithmetic_sequence_l287_287335


namespace exists_arith_prog_5_primes_exists_arith_prog_6_primes_l287_287248

-- Define the condition of being an arithmetic progression
def is_arith_prog (seq : List ℕ) : Prop :=
  ∀ (i : ℕ), i < seq.length - 1 → seq.get! (i + 1) - seq.get! i = seq.get! 1 - seq.get! 0

-- Define the condition of being prime
def all_prime (seq : List ℕ) : Prop :=
  ∀ (n : ℕ), n ∈ seq → Nat.Prime n

-- The main statements
theorem exists_arith_prog_5_primes :
  ∃ (seq : List ℕ), seq.length = 5 ∧ is_arith_prog seq ∧ all_prime seq := 
sorry

theorem exists_arith_prog_6_primes :
  ∃ (seq : List ℕ), seq.length = 6 ∧ is_arith_prog seq ∧ all_prime seq := 
sorry

end exists_arith_prog_5_primes_exists_arith_prog_6_primes_l287_287248


namespace eleven_hash_five_l287_287702

def my_op (r s : ℝ) : ℝ := sorry

axiom op_cond1 : ∀ r : ℝ, my_op r 0 = r
axiom op_cond2 : ∀ r s : ℝ, my_op r s = my_op s r
axiom op_cond3 : ∀ r s : ℝ, my_op (r + 1) s = (my_op r s) + s + 1

theorem eleven_hash_five : my_op 11 5 = 71 :=
by {
    sorry
}

end eleven_hash_five_l287_287702


namespace find_certain_number_l287_287474

theorem find_certain_number (x certain_number : ℤ) 
  (h1 : (28 + x + 42 + 78 + 104) / 5 = 62) 
  (h2 : (certain_number + 62 + 98 + 124 + x) / 5 = 78) : 
  certain_number = 106 := 
by 
  sorry

end find_certain_number_l287_287474


namespace solve_for_x_l287_287465

theorem solve_for_x : ∃ x : ℚ, 6 * (2 * x + 3) - 4 = -3 * (2 - 5 * x) + 3 * x ∧ x = 10 / 3 := by
  sorry

end solve_for_x_l287_287465


namespace find_a6_l287_287637

-- Defining the conditions of the problem
def a1 := 2
def S3 := 12

-- Defining the necessary arithmetic sequence properties
def Sn (a1 d : ℕ) (n : ℕ) : ℕ := n * (2 * a1 + (n - 1) * d) / 2
def an (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

-- Proof statement in Lean
theorem find_a6 (d : ℕ) (a1_val S3_val : ℕ) (h1 : a1_val = 2) (h2 : S3_val = 12) 
    (h3 : 3 * (2 * a1_val + (3 - 1) * d) / 2 = S3_val) : an a1_val d 6 = 12 :=
by 
  -- omitted proof
  sorry

end find_a6_l287_287637


namespace sequence_term_4_l287_287143

noncomputable def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 2 * sequence n

theorem sequence_term_4 : sequence 3 = 8 := 
by
  sorry

end sequence_term_4_l287_287143


namespace discount_percentage_l287_287544

variable (P : ℝ) -- Original price of the dress
variable (D : ℝ) -- Discount percentage

theorem discount_percentage
  (h1 : P * (1 - D / 100) = 68)
  (h2 : 68 * 1.25 = 85)
  (h3 : 85 - P = 5) :
  D = 15 :=
by
  sorry

end discount_percentage_l287_287544


namespace number_of_good_carrots_l287_287303

def total_carrots (nancy_picked : ℕ) (mom_picked : ℕ) : ℕ :=
  nancy_picked + mom_picked

def bad_carrots := 14

def good_carrots (total : ℕ) (bad : ℕ) : ℕ :=
  total - bad

theorem number_of_good_carrots :
  good_carrots (total_carrots 38 47) bad_carrots = 71 := by
  sorry

end number_of_good_carrots_l287_287303


namespace num_of_integers_satisfying_sqrt_condition_l287_287485

theorem num_of_integers_satisfying_sqrt_condition : 
  let S := { x : ℤ | 5 < Real.sqrt x ∧ x < 36 }
  in (S.card = 10) :=
begin
  let S := { x : ℤ | 25 < x ∧ x < 36 },
  sorry
end

end num_of_integers_satisfying_sqrt_condition_l287_287485


namespace intersection_of_M_and_N_l287_287123

def M : Set ℝ := {x | x ≥ 0 ∧ x < 16}
def N : Set ℝ := {x | x ≥ 1/3}

theorem intersection_of_M_and_N :
  M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by
  sorry

end intersection_of_M_and_N_l287_287123


namespace pier_influence_duration_l287_287494

noncomputable def distance_affected_by_typhoon (AB AC: ℝ) : ℝ :=
  let AD := 350
  let DC := (AD ^ 2 - AC ^ 2).sqrt
  2 * DC

noncomputable def duration_under_influence (distance speed: ℝ) : ℝ :=
  distance / speed

theorem pier_influence_duration :
  let AB := 400
  let AC := AB * (1 / 2)
  let speed := 40
  duration_under_influence (distance_affected_by_typhoon AB AC) speed = 2.5 :=
by
  -- Proof would go here, but since it's omitted
  sorry

end pier_influence_duration_l287_287494


namespace solve_for_buttons_l287_287171

def number_of_buttons_on_second_shirt (x : ℕ) : Prop :=
  200 * 3 + 200 * x = 1600

theorem solve_for_buttons : ∃ x : ℕ, number_of_buttons_on_second_shirt x ∧ x = 5 := by
  sorry

end solve_for_buttons_l287_287171


namespace speeds_of_bodies_l287_287014

theorem speeds_of_bodies 
  (v1 v2 : ℝ)
  (h1 : 21 * v1 + 10 * v2 = 270)
  (h2 : 51 * v1 + 40 * v2 = 540)
  (h3 : 5 * v2 = 3 * v1): 
  v1 = 10 ∧ v2 = 6 :=
by
  sorry

end speeds_of_bodies_l287_287014


namespace cost_price_per_meter_l287_287802

def total_length : ℝ := 9.25
def total_cost : ℝ := 397.75

theorem cost_price_per_meter : total_cost / total_length = 43 := sorry

end cost_price_per_meter_l287_287802


namespace old_record_was_300_points_l287_287899

theorem old_record_was_300_points :
  let touchdowns_per_game := 4
  let points_per_touchdown := 6
  let games_in_season := 15
  let conversions := 6
  let points_per_conversion := 2
  let points_beat := 72
  let total_points := touchdowns_per_game * points_per_touchdown * games_in_season + conversions * points_per_conversion
  total_points - points_beat = 300 := 
by
  sorry

end old_record_was_300_points_l287_287899


namespace triangle_count_with_perimeter_11_l287_287862

theorem triangle_count_with_perimeter_11 :
  ∃ (s : Finset (ℕ × ℕ × ℕ)), s.card = 5 ∧ ∀ (a b c : ℕ), (a, b, c) ∈ s ->
    a ≤ b ∧ b ≤ c ∧ a + b + c = 11 ∧ a + b > c :=
sorry

end triangle_count_with_perimeter_11_l287_287862


namespace integer_values_count_l287_287481

theorem integer_values_count (x : ℕ) (h1 : 5 < Real.sqrt x) (h2 : Real.sqrt x < 6) : 
  ∃ count : ℕ, count = 10 := 
by 
  sorry

end integer_values_count_l287_287481


namespace discount_is_5_percent_l287_287971

-- Defining the conditions
def cost_per_iphone : ℕ := 600
def total_cost_3_iphones : ℕ := 3 * cost_per_iphone
def savings : ℕ := 90

-- Calculating the discount percentage
def discount_percentage : ℕ := (savings * 100) / total_cost_3_iphones

-- Stating the theorem
theorem discount_is_5_percent : discount_percentage = 5 :=
  sorry

end discount_is_5_percent_l287_287971


namespace equation_holds_true_l287_287833

theorem equation_holds_true (a b : ℝ) (h₁ : a ≠ 0) (h₂ : 2 * b - a ≠ 0) :
  ((a + 2 * b) / a = b / (2 * b - a)) ↔ 
  (a = -b * (1 + Real.sqrt 17) / 2 ∨ a = -b * (1 - Real.sqrt 17) / 2) := 
sorry

end equation_holds_true_l287_287833


namespace vertex_of_parabola_l287_287497

-- Definition of the parabola
def parabola (x : ℝ) : ℝ := -2 * (x - 3)^2 - 2

-- The theorem stating the vertex of the parabola
theorem vertex_of_parabola : ∃ h k : ℝ, (h, k) = (2, -5) :=
by
  sorry

end vertex_of_parabola_l287_287497


namespace find_pairs_satisfying_conditions_l287_287245

theorem find_pairs_satisfying_conditions (x y : ℝ) :
    abs (x + y) = 3 ∧ x * y = -10 →
    (x = 5 ∧ y = -2) ∨ (x = -2 ∧ y = 5) ∨ (x = 2 ∧ y = -5) ∨ (x = -5 ∧ y = 2) :=
by
  sorry

end find_pairs_satisfying_conditions_l287_287245


namespace general_rule_equation_l287_287913

theorem general_rule_equation (n : ℕ) (hn : n > 0) : (n + 1) / n + (n + 1) = (n + 2) + 1 / n :=
by
  sorry

end general_rule_equation_l287_287913


namespace average_age_of_women_l287_287033

noncomputable def avg_age_two_women (M : ℕ) (new_avg : ℕ) (W : ℕ) :=
  let loss := 20 + 10;
  let gain := 2 * 8;
  W = loss + gain

theorem average_age_of_women (M : ℕ) (new_avg : ℕ) (W : ℕ) (avg_age : ℕ) :
  avg_age_two_women M new_avg W →
  avg_age = 23 :=
sorry

#check average_age_of_women

end average_age_of_women_l287_287033


namespace competition_score_l287_287885

theorem competition_score (x : ℕ) (h : x ≥ 15) : 10 * x - 5 * (20 - x) > 120 := by
  sorry

end competition_score_l287_287885


namespace problem_l287_287572

noncomputable def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) - a n = a 1 - a 0

noncomputable def quadratic_roots (a₃ a₁₀ : ℝ) : Prop :=
a₃^2 - 3 * a₃ - 5 = 0 ∧ a₁₀^2 - 3 * a₁₀ - 5 = 0

theorem problem (a : ℕ → ℝ) (h1 : is_arithmetic_seq a)
  (h2 : quadratic_roots (a 3) (a 10)) :
  a 5 + a 8 = 3 :=
sorry

end problem_l287_287572


namespace fraction_books_sold_l287_287212

theorem fraction_books_sold (B : ℕ) (F : ℚ) (h1 : 36 = B - F * B) (h2 : 252 = 3.50 * F * B) : F = 2 / 3 := by
  -- Proof omitted
  sorry

end fraction_books_sold_l287_287212


namespace equal_frac_implies_x_zero_l287_287705

theorem equal_frac_implies_x_zero (x : ℝ) (h : (4 + x) / (6 + x) = (2 + x) / (3 + x)) : x = 0 :=
sorry

end equal_frac_implies_x_zero_l287_287705


namespace smallest_number_l287_287026

-- Definitions based on the conditions given in the problem
def satisfies_conditions (b : ℕ) : Prop :=
  b % 5 = 2 ∧ b % 4 = 3 ∧ b % 7 = 1

-- Lean proof statement
theorem smallest_number (b : ℕ) : satisfies_conditions b → b = 87 :=
sorry

end smallest_number_l287_287026


namespace Andrews_age_l287_287056

theorem Andrews_age (a g : ℝ) (h1 : g = 15 * a) (h2 : g - a = 55) : a = 55 / 14 :=
by
  /- proof will go here -/
  sorry

end Andrews_age_l287_287056


namespace evaluate_expression_l287_287077

theorem evaluate_expression (x y : ℕ) (hx : x = 4) (hy : y = 9) : 
  2 * x^(y / 2 : ℕ) + 5 * y^(x / 2 : ℕ) = 1429 := by
  sorry

end evaluate_expression_l287_287077


namespace bowling_ball_weight_l287_287701

theorem bowling_ball_weight (b c : ℝ) (h1 : c = 36) (h2 : 5 * b = 4 * c) : b = 28.8 := by
  sorry

end bowling_ball_weight_l287_287701


namespace color_ball_ratios_l287_287629

theorem color_ball_ratios (white_balls red_balls blue_balls : ℕ)
  (h_white : white_balls = 12)
  (h_red_ratio : 4 * red_balls = 3 * white_balls)
  (h_blue_ratio : 4 * blue_balls = 2 * white_balls) :
  red_balls = 9 ∧ blue_balls = 6 :=
by
  sorry

end color_ball_ratios_l287_287629


namespace exists_sum_of_digits_div_11_l287_287756

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_sum_of_digits_div_11 (H : Finset ℕ) (h₁ : H.card = 39) :
  ∃ (a : ℕ) (h : a ∈ H), sum_of_digits a % 11 = 0 :=
by
  sorry

end exists_sum_of_digits_div_11_l287_287756


namespace number_of_integers_between_25_and_36_l287_287490

theorem number_of_integers_between_25_and_36 :
  {n : ℕ | 25 < n ∧ n < 36}.card = 10 :=
by
  sorry

end number_of_integers_between_25_and_36_l287_287490


namespace additional_emails_per_day_l287_287447

theorem additional_emails_per_day
  (emails_per_day_before : ℕ)
  (half_days : ℕ)
  (total_days : ℕ)
  (total_emails : ℕ)
  (emails_received_first_half : ℕ := emails_per_day_before * half_days)
  (emails_received_second_half : ℕ := total_emails - emails_received_first_half)
  (emails_per_day_after : ℕ := emails_received_second_half / half_days) :
  emails_per_day_before = 20 → half_days = 15 → total_days = 30 → total_emails = 675 → (emails_per_day_after - emails_per_day_before = 5) :=
by
  intros
  sorry

end additional_emails_per_day_l287_287447


namespace not_divisible_by_q_plus_one_l287_287085

theorem not_divisible_by_q_plus_one (q : ℕ) (hq_odd : q % 2 = 1) (hq_gt_two : q > 2) :
  ¬ (q + 1) ∣ ((q + 1) ^ ((q - 1) / 2) + 2) :=
by
  sorry

end not_divisible_by_q_plus_one_l287_287085


namespace difference_between_max_and_min_coins_l287_287159

theorem difference_between_max_and_min_coins (n : ℕ) : 
  (∃ x y : ℕ, x * 10 + y * 25 = 45 ∧ x + y = n) →
  (∃ p q : ℕ, p * 10 + q * 25 = 45 ∧ p + q = n) →
  (n = 2) :=
by
  sorry

end difference_between_max_and_min_coins_l287_287159


namespace solve_for_x_l287_287429

theorem solve_for_x (x : ℝ) (h: (6 / (x + 1) = 3 / 2)) : x = 3 :=
sorry

end solve_for_x_l287_287429


namespace loss_percentage_l287_287514

theorem loss_percentage (C : ℝ) (h : 40 * C = 100 * C) : 
  ∃ L : ℝ, L = 60 := 
sorry

end loss_percentage_l287_287514


namespace mono_sum_eq_five_l287_287434

-- Conditions
def term1 (x y : ℝ) (m : ℕ) : ℝ := x^2 * y^m
def term2 (x y : ℝ) (n : ℕ) : ℝ := x^n * y^3

def is_monomial_sum (x y : ℝ) (m n : ℕ) : Prop :=
  term1 x y m + term2 x y n = x^(2:ℕ) * y^(3:ℕ)

-- Theorem stating the result
theorem mono_sum_eq_five (x y : ℝ) (m n : ℕ) (h : is_monomial_sum x y m n) : m + n = 5 :=
by
  sorry

end mono_sum_eq_five_l287_287434


namespace determine_coordinates_of_M_l287_287141

def point_in_fourth_quadrant (M : ℝ × ℝ) : Prop :=
  M.1 > 0 ∧ M.2 < 0

def distance_to_x_axis (M : ℝ × ℝ) (d : ℝ) : Prop :=
  |M.2| = d

def distance_to_y_axis (M : ℝ × ℝ) (d : ℝ) : Prop :=
  |M.1| = d

theorem determine_coordinates_of_M :
  ∃ M : ℝ × ℝ, point_in_fourth_quadrant M ∧ distance_to_x_axis M 3 ∧ distance_to_y_axis M 4 ∧ M = (4, -3) :=
by
  sorry

end determine_coordinates_of_M_l287_287141


namespace petya_wins_optimal_play_l287_287184

theorem petya_wins_optimal_play : 
  ∃ n : Nat, n = 2021 ∧
  (∀ move : Nat → Nat, 
    (move = (λ n, n - 1) ∨ move = (λ n, n - 2)) → 
    (n % 3 ≡ 0) ↔ 
      (∃ optimal_move : Nat → Nat, 
        optimal_move = (λ n, n - 2 + 3) 
          → (optimal_move n % 3 = 0))) -> 
  Petya wins := 
  sorry

end petya_wins_optimal_play_l287_287184


namespace team_air_conditioner_installation_l287_287291

theorem team_air_conditioner_installation (x : ℕ) (y : ℕ) 
  (h1 : 66 % x = 0) 
  (h2 : 60 % y = 0) 
  (h3 : x = y + 2) 
  (h4 : 66 / x = 60 / y) 
  : x = 22 ∧ y = 20 :=
by
  have h5 : x = 22 := sorry
  have h6 : y = 20 := sorry
  exact ⟨h5, h6⟩

end team_air_conditioner_installation_l287_287291


namespace total_items_l287_287503

theorem total_items (B M C : ℕ) 
  (h1 : B = 58) 
  (h2 : B = M + 18) 
  (h3 : B = C - 27) : 
  B + M + C = 183 :=
by 
  sorry

end total_items_l287_287503


namespace fraction_to_decimal_l287_287251

theorem fraction_to_decimal : (7 / 12 : ℝ) = 0.5833 + (3 / 10000) * (1 / (1 - (1 / 10))) := 
by sorry

end fraction_to_decimal_l287_287251


namespace greatest_common_divisor_sum_arithmetic_sequence_l287_287334

theorem greatest_common_divisor_sum_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, d = 15 ∧ ∀ (n : ℕ), n = 15 → ∀ k : ℕ, k = 15 ∧ 15 ∣ (15 * (x + 7 * c)) :=
by
  sorry

end greatest_common_divisor_sum_arithmetic_sequence_l287_287334


namespace intersection_M_N_l287_287127

section

def M (x : ℝ) : Prop := sqrt x < 4
def N (x : ℝ) : Prop := 3 * x >= 1

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | (1 / 3) ≤ x ∧ x < 16} :=
by
  sorry

end

end intersection_M_N_l287_287127


namespace arc_length_calculation_l287_287999

theorem arc_length_calculation (C θ : ℝ) (hC : C = 72) (hθ : θ = 45) :
  (θ / 360) * C = 9 :=
by
  sorry

end arc_length_calculation_l287_287999


namespace total_money_raised_l287_287066

variable (clements_cookies jakes_cookies torys_cookies : ℕ)

def baked_clementine : ℕ := 72
def baked_jake : ℕ := baked_clementine * 2
def baked_tory : ℕ := (baked_jake + baked_clementine) / 2
def price_per_cookie : ℕ := 2

theorem total_money_raised :
  let total_cookies := baked_clementine + baked_jake + baked_tory,
      total_money := total_cookies * price_per_cookie
  in
  total_money = 648 := by
  sorry

end total_money_raised_l287_287066


namespace casey_savings_l287_287232

-- Define the constants given in the problem conditions
def wage_employee_1 : ℝ := 20
def wage_employee_2 : ℝ := 22
def subsidy : ℝ := 6
def hours_per_week : ℝ := 40

-- Define the weekly cost of each employee
def weekly_cost_employee_1 := wage_employee_1 * hours_per_week
def weekly_cost_employee_2 := (wage_employee_2 - subsidy) * hours_per_week

-- Define the savings by hiring the cheaper employee
def savings := weekly_cost_employee_1 - weekly_cost_employee_2

-- Theorem stating the expected savings
theorem casey_savings : savings = 160 := by
  -- Proof is not included
  sorry

end casey_savings_l287_287232


namespace num_O_atoms_correct_l287_287813

-- Conditions
def atomic_weight_H : ℕ := 1
def atomic_weight_Cr : ℕ := 52
def atomic_weight_O : ℕ := 16
def num_H_atoms : ℕ := 2
def num_Cr_atoms : ℕ := 1
def molecular_weight : ℕ := 118

-- Calculations
def weight_H : ℕ := num_H_atoms * atomic_weight_H
def weight_Cr : ℕ := num_Cr_atoms * atomic_weight_Cr
def total_weight_H_Cr : ℕ := weight_H + weight_Cr
def weight_O : ℕ := molecular_weight - total_weight_H_Cr
def num_O_atoms : ℕ := weight_O / atomic_weight_O

-- Theorem to prove the number of Oxygen atoms is 4
theorem num_O_atoms_correct : num_O_atoms = 4 :=
by {
  sorry -- Proof not provided.
}

end num_O_atoms_correct_l287_287813


namespace half_lake_covered_day_l287_287523

theorem half_lake_covered_day
  (N : ℕ) -- the total number of flowers needed to cover the entire lake
  (flowers_on_day : ℕ → ℕ) -- a function that gives the number of flowers on a specific day
  (h1 : flowers_on_day 20 = N) -- on the 20th day, the number of flowers is N
  (h2 : ∀ d, flowers_on_day (d + 1) = 2 * flowers_on_day d) -- the number of flowers doubles each day
  : flowers_on_day 19 = N / 2 :=
by
  sorry

end half_lake_covered_day_l287_287523


namespace total_sacks_needed_l287_287677

def first_bakery_needs : ℕ := 2
def second_bakery_needs : ℕ := 4
def third_bakery_needs : ℕ := 12
def weeks : ℕ := 4

theorem total_sacks_needed :
  first_bakery_needs * weeks + second_bakery_needs * weeks + third_bakery_needs * weeks = 72 :=
by
  sorry

end total_sacks_needed_l287_287677


namespace range_of_a_and_t_minimum_of_y_l287_287205

noncomputable def minimum_value_y (a b : ℝ) (h : a + b = 1) : ℝ :=
(a + 1/a) * (b + 1/b)

theorem range_of_a_and_t (a b : ℝ) (h : a + b = 1) :
  0 < a ∧ a < 1 ∧ 0 < a * b ∧ a * b <= 1/4 :=
sorry

theorem minimum_of_y (a b : ℝ) (h : a + b = 1) :
  minimum_value_y a b h = 25/4 :=
sorry

end range_of_a_and_t_minimum_of_y_l287_287205


namespace ratio_twice_width_to_length_l287_287473

-- Given conditions:
def length_of_field : ℚ := 24
def width_of_field : ℚ := 13.5

-- The problem is to prove the ratio of twice the width to the length of the field is 9/8
theorem ratio_twice_width_to_length : 2 * width_of_field / length_of_field = 9 / 8 :=
by sorry

end ratio_twice_width_to_length_l287_287473


namespace X_Y_independent_normal_l287_287804

open real measure_theory probability_theory

noncomputable def rayleigh_pdf (σ : ℝ) (r : ℝ) : ℝ :=
  if r > 0 then (r / σ^2) * exp (-r^2 / (2 * σ^2)) else 0

def uniform_pdf (α : ℝ) (k : ℕ) (θ : ℝ) : ℝ :=
  if α ≤ θ ∧ θ < α + 2 * π * k then 1 / (2 * π * k) else 0

def cos_density (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 1 then 1 / (π * sqrt (1 - x^2)) else 0

def sin_density (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 1 then 1 / (π * sqrt (1 - x^2)) else 0

theorem X_Y_independent_normal (σ : ℝ) (α : ℝ) (k : ℕ) :
  (∀ r > 0, pdf (rayleigh_pdf σ) r) →
  (∀ θ, pdf (uniform_pdf α k) θ) →
  independent X Y →
  ∀ x y, has_pdf (λ X = R * (cos θ)) :=
    sorry

end X_Y_independent_normal_l287_287804


namespace houses_distance_l287_287647

theorem houses_distance (num_houses : ℕ) (total_length : ℝ) (at_both_ends : Bool) 
  (h1: num_houses = 6) (h2: total_length = 11.5) (h3: at_both_ends = true) : 
  total_length / (num_houses - 1) = 2.3 := 
by
  sorry

end houses_distance_l287_287647


namespace servings_in_one_week_l287_287080

theorem servings_in_one_week (daily_servings : ℕ) (days_in_week : ℕ) (total_servings : ℕ)
  (h1 : daily_servings = 3)
  (h2 : days_in_week = 7)
  (h3 : total_servings = daily_servings * days_in_week) :
  total_servings = 21 := by
  sorry

end servings_in_one_week_l287_287080


namespace max_possible_median_l287_287620

theorem max_possible_median (total_cups : ℕ) (total_customers : ℕ) (min_cups_per_customer : ℕ)
  (h1 : total_cups = 310) (h2 : total_customers = 120) (h3 : min_cups_per_customer = 1) :
  ∃ median : ℕ, median = 4 :=
by {
  sorry
}

end max_possible_median_l287_287620


namespace polygon_sides_eq_six_l287_287323

theorem polygon_sides_eq_six (n : ℕ) (h1 : (n - 2) * 180 = 2 * 360) : n = 6 :=
by
  sorry

end polygon_sides_eq_six_l287_287323


namespace large_pizza_slices_l287_287672

variable (L : ℕ)

theorem large_pizza_slices :
  (2 * L + 2 * 8 = 48) → (L = 16) :=
by 
  sorry

end large_pizza_slices_l287_287672


namespace find_sum_of_a_b_l287_287985

def star (a b : ℕ) : ℕ := a^b - a * b

theorem find_sum_of_a_b (a b : ℕ) (h1 : 2 ≤ a) (h2 : 2 ≤ b) (h3 : star a b = 2) : a + b = 5 := 
by
  sorry

end find_sum_of_a_b_l287_287985


namespace range_independent_variable_l287_287290

noncomputable def range_of_independent_variable (x : ℝ) : Prop :=
  x ≠ 3

theorem range_independent_variable (x : ℝ) :
  (∃ y : ℝ, y = 1 / (x - 3)) → x ≠ 3 :=
by
  intro h
  sorry

end range_independent_variable_l287_287290


namespace sum_remainder_of_consecutive_odds_l287_287938

theorem sum_remainder_of_consecutive_odds :
  (11075 + 11077 + 11079 + 11081 + 11083 + 11085 + 11087) % 14 = 7 :=
by
  -- Adding the proof here
  sorry

end sum_remainder_of_consecutive_odds_l287_287938


namespace find_a_l287_287103

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 0 then 1 - x else a * x

theorem find_a (a : ℝ) : f (-1) a = f 1 a → a = 2 := by
  intro h
  sorry

end find_a_l287_287103


namespace find_a_plus_b_l287_287104

theorem find_a_plus_b (a b x : ℝ) (h1 : x + 2 * a > 4) (h2 : 2 * x < b)
  (h3 : 0 < x) (h4 : x < 2) : a + b = 6 :=
by
  sorry

end find_a_plus_b_l287_287104


namespace problem_l287_287719

noncomputable def f (ω φ : ℝ) (x : ℝ) := 4 * Real.sin (ω * x + φ)

theorem problem (ω : ℝ) (φ : ℝ) (x1 x2 α : ℝ) (hω : 0 < ω) (hφ : |φ| < Real.pi / 2)
  (h0 : f ω φ 0 = 2 * Real.sqrt 3)
  (hx1 : f ω φ x1 = 0) (hx2 : f ω φ x2 = 0) (hx1x2 : |x1 - x2| = Real.pi / 2)
  (hα : α ∈ Set.Ioo (Real.pi / 12) (Real.pi / 2)) :
  f 2 (Real.pi / 3) α = 12 / 5 ∧ Real.sin (2 * α) = (3 + 4 * Real.sqrt 3) / 10 :=
sorry

end problem_l287_287719


namespace acute_triangle_condition_l287_287451

theorem acute_triangle_condition (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  (|a^2 - b^2| < c^2 ∧ c^2 < a^2 + b^2) ↔ (a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2) :=
sorry

end acute_triangle_condition_l287_287451


namespace expected_value_of_monicas_winnings_l287_287745

def die_outcome (n : ℕ) : ℤ :=
  if n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 then n else if n = 1 ∨ n = 4 ∨ n = 6 ∨ n = 8 then 0 else -5

noncomputable def expected_winnings : ℚ :=
  (1/2 : ℚ) * 0 + (1/8 : ℚ) * 2 + (1/8 : ℚ) * 3 + (1/8 : ℚ) * 5 + (1/8 : ℚ) * 7 + (1/8 : ℚ) * (-5)

theorem expected_value_of_monicas_winnings : expected_winnings = 3/2 := by
  sorry

end expected_value_of_monicas_winnings_l287_287745


namespace students_not_reading_novels_l287_287379

theorem students_not_reading_novels
  (total_students : ℕ)
  (students_three_or_more_novels : ℕ)
  (students_two_novels : ℕ)
  (students_one_novel : ℕ)
  (h_total_students : total_students = 240)
  (h_students_three_or_more_novels : students_three_or_more_novels = 1 / 6 * 240)
  (h_students_two_novels : students_two_novels = 35 / 100 * 240)
  (h_students_one_novel : students_one_novel = 5 / 12 * 240)
  :
  total_students - (students_three_or_more_novels + students_two_novels + students_one_novel) = 16 :=
by
  sorry

end students_not_reading_novels_l287_287379


namespace minimum_value_l287_287715

theorem minimum_value (a : ℝ) (h₀ : 0 < a) (h₁ : a < 3) :
  ∃ a : ℝ, (0 < a ∧ a < 3) ∧ (1 / a + 4 / (8 - a) = 9 / 8) := by
sorry

end minimum_value_l287_287715


namespace divides_three_and_eleven_l287_287405

theorem divides_three_and_eleven (n : ℕ) (h : n ≥ 1) : (n ∣ 3^n + 1 ∧ n ∣ 11^n + 1) ↔ (n = 1 ∨ n = 2) := by
  sorry

end divides_three_and_eleven_l287_287405


namespace uncle_bruce_dough_weight_l287_287190

-- Definitions based on the conditions
variable {TotalChocolate : ℕ} (h1 : TotalChocolate = 13)
variable {ChocolateLeftOver : ℕ} (h2 : ChocolateLeftOver = 4)
variable {ChocolatePercentage : ℝ} (h3 : ChocolatePercentage = 0.2) 
variable {WeightOfDough : ℝ}

-- Target statement expressing the final question and answer
theorem uncle_bruce_dough_weight 
  (h1 : TotalChocolate = 13) 
  (h2 : ChocolateLeftOver = 4) 
  (h3 : ChocolatePercentage = 0.2) : 
  WeightOfDough = 36 := by
  sorry

end uncle_bruce_dough_weight_l287_287190


namespace inequality_solution_l287_287083

theorem inequality_solution (x : ℝ) : 
  (x ∈ Set.Iio (-3/4) ∪ Set.Ioc 4 5 ∪ Set.Ioi 5) ↔ 
  (x+2) ≠ 0 ∧ (x-2) ≠ 0 ∧ (4 * (x^2 - 1) * (x-2) - (x+2) * (7 * x - 6)) / (4 * (x+2) * (x-2)) ≥ 0 := 
by
  sorry

end inequality_solution_l287_287083


namespace ninth_term_is_83_l287_287920

-- Definitions based on conditions
def a : ℕ := 3
def d : ℕ := 10
def arith_sequence (n : ℕ) : ℕ := a + n * d

-- Theorem to prove the 9th term is 83
theorem ninth_term_is_83 : arith_sequence 8 = 83 :=
by
  sorry

end ninth_term_is_83_l287_287920


namespace radius_of_cookie_l287_287170

theorem radius_of_cookie : 
  ∀ x y : ℝ, (x^2 + y^2 - 6.5 = x + 3 * y) → 
  ∃ (c : ℝ × ℝ) (r : ℝ), r = 3 ∧ (x - c.1)^2 + (y - c.2)^2 = r^2 :=
by {
  sorry
}

end radius_of_cookie_l287_287170


namespace time_for_nth_mile_l287_287817

noncomputable def speed (k : ℝ) (d : ℝ) : ℝ := k / (d * d)

noncomputable def time_for_mile (n : ℕ) : ℝ :=
  if n = 1 then 1
  else if n = 2 then 2
  else 2 * (n - 1) * (n - 1)

theorem time_for_nth_mile (n : ℕ) (h₁ : ∀ d : ℝ, d ≥ 1 → speed (1/2) d = 1 / (2 * d * d))
  (h₂ : time_for_mile 1 = 1)
  (h₃ : time_for_mile 2 = 2) :
  time_for_mile n = 2 * (n - 1) * (n - 1) := sorry

end time_for_nth_mile_l287_287817


namespace isosceles_triangle_base_angle_l287_287055

theorem isosceles_triangle_base_angle (a b c : ℝ) (h_triangle : a + b + c = 180)
  (h_iso : a = b ∨ b = c ∨ a = c) (h_interior : a = 50 ∨ b = 50 ∨ c = 50) :
  c = 50 ∨ c = 65 :=
by sorry

end isosceles_triangle_base_angle_l287_287055


namespace sin_alpha_eq_sin_beta_l287_287880

theorem sin_alpha_eq_sin_beta (α β : Real) (k : Int) 
  (h_symmetry : α + β = 2 * k * Real.pi + Real.pi) : 
  Real.sin α = Real.sin β := 
by 
  sorry

end sin_alpha_eq_sin_beta_l287_287880


namespace sacks_required_in_4_weeks_l287_287676

-- Definitions for the weekly requirements of each bakery
def weekly_sacks_bakery1 : Nat := 2
def weekly_sacks_bakery2 : Nat := 4
def weekly_sacks_bakery3 : Nat := 12

-- Total weeks considered
def weeks : Nat := 4

-- Calculating the total sacks needed for all bakeries over the given weeks
def total_sacks_needed : Nat :=
  (weekly_sacks_bakery1 * weeks) +
  (weekly_sacks_bakery2 * weeks) +
  (weekly_sacks_bakery3 * weeks)

-- The theorem to be proven
theorem sacks_required_in_4_weeks :
  total_sacks_needed = 72 :=
by
  sorry

end sacks_required_in_4_weeks_l287_287676
