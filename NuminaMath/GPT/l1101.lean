import Mathlib

namespace NUMINAMATH_GPT_factor_1_factor_2_factor_3_l1101_110165

-- Consider the variables a, b, x, y
variable (a b x y : ℝ)

-- Statement 1: Factorize 3a^3 - 6a^2 + 3a
theorem factor_1 : 3 * a^3 - 6 * a^2 + 3 * a = 3 * a * (a - 1)^2 :=
by
  sorry
  
-- Statement 2: Factorize a^2(x - y) + b^2(y - x)
theorem factor_2 : a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a^2 - b^2) :=
by
  sorry

-- Statement 3: Factorize 16(a + b)^2 - 9(a - b)^2
theorem factor_3 : 16 * (a + b)^2 - 9 * (a - b)^2 = (a + 7 * b) * (7 * a + b) :=
by
  sorry

end NUMINAMATH_GPT_factor_1_factor_2_factor_3_l1101_110165


namespace NUMINAMATH_GPT_f_monotonicity_g_min_l1101_110173

-- Definitions
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * a ^ x - 2 * a ^ (-x)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a ^ (2 * x) + a ^ (-2 * x) - 2 * f x a

-- Conditions
variable {a : ℝ} 
variable (a_pos : 0 < a) (a_ne_one : a ≠ 1) (f_one : f 1 a = 3) (x : ℝ) (h : 0 ≤ x ∧ x ≤ 3)

-- Monotonicity of f(x)
theorem f_monotonicity : 
  (∀ x y, x < y → f x a < f y a) ∨ (∀ x y, x < y → f y a < f x a) :=
sorry

-- Minimum value of g(x)
theorem g_min : ∃ x' : ℝ, 0 ≤ x' ∧ x' ≤ 3 ∧ g x' a = -2 :=
sorry

end NUMINAMATH_GPT_f_monotonicity_g_min_l1101_110173


namespace NUMINAMATH_GPT_trajectory_sum_of_distances_to_axes_l1101_110112

theorem trajectory_sum_of_distances_to_axes (x y : ℝ) (h : |x| + |y| = 6) :
  |x| + |y| = 6 := 
by 
  sorry

end NUMINAMATH_GPT_trajectory_sum_of_distances_to_axes_l1101_110112


namespace NUMINAMATH_GPT_race_length_l1101_110159

theorem race_length (covered_meters remaining_meters race_length : ℕ)
  (h_covered : covered_meters = 721)
  (h_remaining : remaining_meters = 279)
  (h_race_length : race_length = covered_meters + remaining_meters) :
  race_length = 1000 :=
by
  rw [h_covered, h_remaining] at h_race_length
  exact h_race_length

end NUMINAMATH_GPT_race_length_l1101_110159


namespace NUMINAMATH_GPT_Robie_gave_away_boxes_l1101_110192

theorem Robie_gave_away_boxes :
  ∀ (total_cards cards_per_box boxes_with_him remaining_cards : ℕ)
  (h_total_cards : total_cards = 75)
  (h_cards_per_box : cards_per_box = 10)
  (h_boxes_with_him : boxes_with_him = 5)
  (h_remaining_cards : remaining_cards = 5),
  (total_cards / cards_per_box) - boxes_with_him = 2 :=
by
  intros total_cards cards_per_box boxes_with_him remaining_cards
  intros h_total_cards h_cards_per_box h_boxes_with_him h_remaining_cards
  sorry

end NUMINAMATH_GPT_Robie_gave_away_boxes_l1101_110192


namespace NUMINAMATH_GPT_third_quadrant_to_first_third_fourth_l1101_110147

theorem third_quadrant_to_first_third_fourth (k : ℤ) (α : ℝ) 
  (h : 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2) : 
  ∃ n : ℤ, (2 * k / 3 % 2) * Real.pi + Real.pi / 3 < α / 3 ∧ α / 3 < (2 * k / 3 % 2) * Real.pi + Real.pi / 2 ∨
            (2 * (3 * n + 1) % 2) * Real.pi + Real.pi < α / 3 ∧ α / 3 < (2 * (3 * n + 1) % 2) * Real.pi + 7 * Real.pi / 6 ∨
            (2 * (3 * n + 2) % 2) * Real.pi + 5 * Real.pi / 3 < α / 3 ∧ α / 3 < (2 * (3 * n + 2) % 2) * Real.pi + 11 * Real.pi / 6 :=
sorry

end NUMINAMATH_GPT_third_quadrant_to_first_third_fourth_l1101_110147


namespace NUMINAMATH_GPT_xy_sum_one_l1101_110149

theorem xy_sum_one (x y : ℝ) (h : x > 0) (k : y > 0) (hx : x^5 + 5*x^3*y + 5*x^2*y^2 + 5*x*y^3 + y^5 = 1) : x + y = 1 :=
sorry

end NUMINAMATH_GPT_xy_sum_one_l1101_110149


namespace NUMINAMATH_GPT_smallest_perfect_square_greater_than_x_l1101_110162

theorem smallest_perfect_square_greater_than_x (x : ℤ)
  (h₁ : ∃ k : ℤ, k^2 ≠ x)
  (h₂ : x ≥ 0) :
  ∃ n : ℤ, n^2 > x ∧ ∀ m : ℤ, m^2 > x → n^2 ≤ m^2 :=
sorry

end NUMINAMATH_GPT_smallest_perfect_square_greater_than_x_l1101_110162


namespace NUMINAMATH_GPT_monica_total_savings_l1101_110146

theorem monica_total_savings :
  ∀ (weekly_saving : ℤ) (weeks_per_cycle : ℤ) (cycles : ℤ),
    weekly_saving = 15 →
    weeks_per_cycle = 60 →
    cycles = 5 →
    weekly_saving * weeks_per_cycle * cycles = 4500 :=
by
  intros weekly_saving weeks_per_cycle cycles
  sorry

end NUMINAMATH_GPT_monica_total_savings_l1101_110146


namespace NUMINAMATH_GPT_initial_customers_l1101_110187

theorem initial_customers (x : ℕ) (h1 : x - 31 + 26 = 28) : x = 33 := 
by 
  sorry

end NUMINAMATH_GPT_initial_customers_l1101_110187


namespace NUMINAMATH_GPT_probability_of_shaded_triangle_l1101_110197

def total_triangles : ℕ := 9
def shaded_triangles : ℕ := 3

theorem probability_of_shaded_triangle :
  total_triangles > 5 →
  (shaded_triangles : ℚ) / total_triangles = 1 / 3 :=
by
  intros h
  -- proof here
  sorry

end NUMINAMATH_GPT_probability_of_shaded_triangle_l1101_110197


namespace NUMINAMATH_GPT_place_value_ratio_l1101_110171

theorem place_value_ratio :
  let val_6 := 1000
  let val_2 := 0.1
  val_6 / val_2 = 10000 :=
by
  -- the proof would go here
  sorry

end NUMINAMATH_GPT_place_value_ratio_l1101_110171


namespace NUMINAMATH_GPT_parabola_above_line_l1101_110120

variable {a b c : ℝ}

theorem parabola_above_line
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (H : (b - c) ^ 2 - 4 * a * c < 0) :
  (b + c) ^ 2 - 4 * c * (a + b) < 0 := 
sorry

end NUMINAMATH_GPT_parabola_above_line_l1101_110120


namespace NUMINAMATH_GPT_percentage_x_eq_six_percent_y_l1101_110175

variable {x y : ℝ}

theorem percentage_x_eq_six_percent_y (h1 : ∃ P : ℝ, (P / 100) * x = (6 / 100) * y)
  (h2 : (18 / 100) * x = (9 / 100) * y) : 
  ∃ P : ℝ, P = 12 := 
sorry

end NUMINAMATH_GPT_percentage_x_eq_six_percent_y_l1101_110175


namespace NUMINAMATH_GPT_part_a_part_b_l1101_110199

/- Part (a) -/
theorem part_a (a b c d : ℝ) (h1 : (a + b ≠ c + d)) (h2 : (a + c ≠ b + d)) (h3 : (a + d ≠ b + c)) :
  ∃ (spheres : ℕ), spheres = 8 := sorry

/- Part (b) -/
theorem part_b (a b c d : ℝ) (h : (a + b = c + d) ∨ (a + c = b + d) ∨ (a + d = b + c)) :
  ∃ (spheres : ℕ), ∀ (n : ℕ), n > 0 → spheres = n := sorry

end NUMINAMATH_GPT_part_a_part_b_l1101_110199


namespace NUMINAMATH_GPT_max_lessons_l1101_110123

theorem max_lessons (x y z : ℕ) (h1 : y * z = 6) (h2 : x * z = 21) (h3 : x * y = 14) : 3 * x * y * z = 126 :=
sorry

end NUMINAMATH_GPT_max_lessons_l1101_110123


namespace NUMINAMATH_GPT_reciprocal_geometric_sum_l1101_110195

variable (n : ℕ) (r s : ℝ)
variable (h_r_nonzero : r ≠ 0)
variable (h_sum_original : (1 - r^(2 * n)) / (1 - r^2) = s^3)

theorem reciprocal_geometric_sum (n : ℕ) (r s : ℝ) (h_r_nonzero : r ≠ 0)
  (h_sum_original : (1 - r^(2 * n)) / (1 - r^2) = s^3) :
  ((1 - (1 / r^2)^n) / (1 - 1 / r^2)) = s^3 / r^2 :=
sorry

end NUMINAMATH_GPT_reciprocal_geometric_sum_l1101_110195


namespace NUMINAMATH_GPT_positive_integers_not_in_E_are_perfect_squares_l1101_110183

open Set

def E : Set ℕ := {m | ∃ n : ℕ, m = Int.floor (n + Real.sqrt n + 0.5)}

theorem positive_integers_not_in_E_are_perfect_squares (m : ℕ) (h_pos : 0 < m) :
  m ∉ E ↔ ∃ t : ℕ, m = t^2 := 
by
    sorry

end NUMINAMATH_GPT_positive_integers_not_in_E_are_perfect_squares_l1101_110183


namespace NUMINAMATH_GPT_gcd_poly_correct_l1101_110148

-- Define the conditions
def is_even_multiple_of (x k : ℕ) : Prop :=
  ∃ (n : ℕ), x = k * 2 * n

variable (b : ℕ)

-- Given condition
axiom even_multiple_7768 : is_even_multiple_of b 7768

-- Define the polynomials
def poly1 (b : ℕ) := 4 * b * b + 37 * b + 72
def poly2 (b : ℕ) := 3 * b + 8

-- Proof statement
theorem gcd_poly_correct : gcd (poly1 b) (poly2 b) = 8 :=
  sorry

end NUMINAMATH_GPT_gcd_poly_correct_l1101_110148


namespace NUMINAMATH_GPT_inverse_proposition_l1101_110100

theorem inverse_proposition (a b : ℝ) (h1 : a < 1) (h2 : b < 1) : a + b ≠ 2 :=
by sorry

end NUMINAMATH_GPT_inverse_proposition_l1101_110100


namespace NUMINAMATH_GPT_find_percentage_l1101_110110

noncomputable def percentage (X : ℝ) : ℝ := (377.8020134228188 * 100 * 5.96) / 1265

theorem find_percentage : percentage 178 = 178 := by
  -- Conditions
  let P : ℝ := 178
  let A : ℝ := 1265
  let divisor : ℝ := 5.96
  let result : ℝ := 377.8020134228188

  -- Define the percentage calculation
  let X := (result * 100 * divisor) / A

  -- Verify the calculation matches
  have h : X = P := by sorry

  trivial

end NUMINAMATH_GPT_find_percentage_l1101_110110


namespace NUMINAMATH_GPT_sum_of_three_numbers_l1101_110194

theorem sum_of_three_numbers :
  ∃ (a b c : ℕ), 
    (a ≤ b ∧ b ≤ c) ∧ 
    (b = 8) ∧ 
    ((a + b + c) / 3 = a + 8) ∧ 
    ((a + b + c) / 3 = c - 20) ∧ 
    (a + b + c = 60) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l1101_110194


namespace NUMINAMATH_GPT_shortest_chord_l1101_110107

noncomputable def line_eq (m : ℝ) (x y : ℝ) : Prop := 2 * m * x - y - 8 * m - 3 = 0
noncomputable def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y + 6)^2 = 25

theorem shortest_chord (m : ℝ) :
  (∃ x y, line_eq m x y ∧ circle_eq x y) →
  m = 1 / 6 :=
by sorry

end NUMINAMATH_GPT_shortest_chord_l1101_110107


namespace NUMINAMATH_GPT_graph_single_point_l1101_110101

theorem graph_single_point (c : ℝ) : 
  (∃ x y : ℝ, ∀ (x' y' : ℝ), 4 * x'^2 + y'^2 + 16 * x' - 6 * y' + c = 0 → (x' = x ∧ y' = y)) → c = 7 := 
by
  sorry

end NUMINAMATH_GPT_graph_single_point_l1101_110101


namespace NUMINAMATH_GPT_complex_division_l1101_110151

theorem complex_division (i : ℂ) (h_i : i * i = -1) : (3 - 4 * i) / i = 4 - 3 * i :=
by
  sorry

end NUMINAMATH_GPT_complex_division_l1101_110151


namespace NUMINAMATH_GPT_geometric_progression_general_term_l1101_110142

noncomputable def a_n (n : ℕ) : ℝ := 2^(n-1)

theorem geometric_progression_general_term :
  (∀ n : ℕ, n ≥ 1 → a_n n > 0) ∧
  a_n 1 = 1 ∧
  a_n 2 + a_n 3 = 6 →
  ∀ n, a_n n = 2^(n-1) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_geometric_progression_general_term_l1101_110142


namespace NUMINAMATH_GPT_square_difference_l1101_110139

theorem square_difference (x : ℤ) (h : x^2 = 1444) : (x + 1) * (x - 1) = 1443 := 
by 
  sorry

end NUMINAMATH_GPT_square_difference_l1101_110139


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1101_110182

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 4 ∨ a = 7) (h2 : b = 4 ∨ b = 7) (h3 : a ≠ b) :
  (a + a + b = 15 ∨ a + a + b = 18) ∨ (a + b + b = 15 ∨ a + b + b = 18) :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1101_110182


namespace NUMINAMATH_GPT_compute_one_plus_i_power_four_l1101_110188

theorem compute_one_plus_i_power_four (i : ℂ) (h : i^2 = -1) : (1 + i)^4 = -4 :=
by
  sorry

end NUMINAMATH_GPT_compute_one_plus_i_power_four_l1101_110188


namespace NUMINAMATH_GPT_triangle_altitude_l1101_110138

variable (Area : ℝ) (base : ℝ) (altitude : ℝ)

theorem triangle_altitude (hArea : Area = 1250) (hbase : base = 50) :
  2 * Area / base = altitude :=
by
  sorry

end NUMINAMATH_GPT_triangle_altitude_l1101_110138


namespace NUMINAMATH_GPT_abc_divisibility_l1101_110143

theorem abc_divisibility (a b c : ℕ) (h1 : a^2 * b ∣ a^3 + b^3 + c^3) (h2 : b^2 * c ∣ a^3 + b^3 + c^3) (h3 : c^2 * a ∣ a^3 + b^3 + c^3) :
  a = 1 ∧ b = 1 ∧ c = 1 :=
sorry

end NUMINAMATH_GPT_abc_divisibility_l1101_110143


namespace NUMINAMATH_GPT_neil_initial_games_l1101_110141

theorem neil_initial_games (N : ℕ) 
  (H₀ : ℕ) (H₀_eq : H₀ = 58)
  (H₁ : ℕ) (H₁_eq : H₁ = H₀ - 6)
  (H₁_condition : H₁ = 4 * (N + 6)) : N = 7 :=
by {
  -- Substituting the given values and simplifying to show the final equation
  sorry
}

end NUMINAMATH_GPT_neil_initial_games_l1101_110141


namespace NUMINAMATH_GPT_parabola_intersects_y_axis_l1101_110190

theorem parabola_intersects_y_axis (m n : ℝ) :
  (∃ (x y : ℝ), y = x^2 + m * x + n ∧ 
  ((x = -1 ∧ y = -6) ∨ (x = 1 ∧ y = 0))) →
  (0, (-4)) = (0, n) :=
by
  sorry

end NUMINAMATH_GPT_parabola_intersects_y_axis_l1101_110190


namespace NUMINAMATH_GPT_abs_diff_of_roots_eq_one_l1101_110140

theorem abs_diff_of_roots_eq_one {p q : ℝ} (h₁ : p + q = 7) (h₂ : p * q = 12) : |p - q| = 1 := 
by 
  sorry

end NUMINAMATH_GPT_abs_diff_of_roots_eq_one_l1101_110140


namespace NUMINAMATH_GPT_find_xyz_l1101_110126

theorem find_xyz (x y z : ℝ) (h1 : (x + y + z) * (x * y + x * z + y * z) = 45) (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 15) (h3 : x + y + z = 5) : x * y * z = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_xyz_l1101_110126


namespace NUMINAMATH_GPT_cube_root_rational_l1101_110115

theorem cube_root_rational (a b : ℚ) (r : ℚ) (h1 : ∃ x : ℚ, x^3 = a) (h2 : ∃ y : ℚ, y^3 = b) (h3 : ∃ x y : ℚ, x + y = r ∧ x^3 = a ∧ y^3 = b) :
  (∃ x : ℚ, x^3 = a) ∧ (∃ y : ℚ, y^3 = b) :=
sorry

end NUMINAMATH_GPT_cube_root_rational_l1101_110115


namespace NUMINAMATH_GPT_colin_speed_l1101_110185

noncomputable def B : Real := 1
noncomputable def T : Real := 2 * B
noncomputable def Br : Real := (1/3) * T
noncomputable def C : Real := 6 * Br

theorem colin_speed : C = 4 := by
  sorry

end NUMINAMATH_GPT_colin_speed_l1101_110185


namespace NUMINAMATH_GPT_beef_cubes_per_slab_l1101_110116

-- Define the conditions as variables
variables (kabob_sticks : ℕ) (cubes_per_stick : ℕ) (cost_per_slab : ℕ) (total_cost : ℕ) (total_kabob_sticks : ℕ)

-- Assume the conditions from step a)
theorem beef_cubes_per_slab 
  (h1 : cubes_per_stick = 4) 
  (h2 : cost_per_slab = 25) 
  (h3 : total_cost = 50) 
  (h4 : total_kabob_sticks = 40)
  : total_cost / cost_per_slab * (total_kabob_sticks * cubes_per_stick) / (total_cost / cost_per_slab) = 80 := 
by {
  -- the proof goes here
  sorry
}

end NUMINAMATH_GPT_beef_cubes_per_slab_l1101_110116


namespace NUMINAMATH_GPT_slower_speed_is_35_l1101_110109

-- Define the given conditions
def distance : ℝ := 70 -- distance is 70 km
def speed_on_time : ℝ := 40 -- on-time average speed is 40 km/hr
def delay : ℝ := 0.25 -- delay is 15 minutes or 0.25 hours

-- This is the statement we need to prove
theorem slower_speed_is_35 :
  ∃ slower_speed : ℝ, 
    slower_speed = distance / (distance / speed_on_time + delay) ∧ slower_speed = 35 :=
by
  sorry

end NUMINAMATH_GPT_slower_speed_is_35_l1101_110109


namespace NUMINAMATH_GPT_hannah_payment_l1101_110106

def costWashingMachine : ℝ := 100
def costDryer : ℝ := costWashingMachine - 30
def totalCostBeforeDiscount : ℝ := costWashingMachine + costDryer
def discount : ℝ := totalCostBeforeDiscount * 0.1
def finalCost : ℝ := totalCostBeforeDiscount - discount

theorem hannah_payment : finalCost = 153 := by
  simp [costWashingMachine, costDryer, totalCostBeforeDiscount, discount, finalCost]
  sorry

end NUMINAMATH_GPT_hannah_payment_l1101_110106


namespace NUMINAMATH_GPT_partial_fraction_decomposition_product_l1101_110184

theorem partial_fraction_decomposition_product :
  ∃ A B C : ℚ,
    (A + 2) * (A - 3) *
    (B - 2) * (B - 3) *
    (C - 2) * (C + 2) = x^2 - 12 ∧
    (A = -2) ∧
    (B = 2/5) ∧
    (C = 3/5) ∧
    (A * B * C = -12/25) :=
  sorry

end NUMINAMATH_GPT_partial_fraction_decomposition_product_l1101_110184


namespace NUMINAMATH_GPT_sandy_spent_home_currency_l1101_110177

variable (A B C D : ℝ)

def total_spent_home_currency (A B C D : ℝ) : ℝ :=
  let total_foreign := A + B + C
  total_foreign * D

theorem sandy_spent_home_currency (D : ℝ) : 
  total_spent_home_currency 13.99 12.14 7.43 D = 33.56 * D := 
by
  sorry

end NUMINAMATH_GPT_sandy_spent_home_currency_l1101_110177


namespace NUMINAMATH_GPT_sum_of_solutions_l1101_110191

theorem sum_of_solutions : 
  ∃ x1 x2 x3 : ℝ, (x1 = 10 ∧ x2 = 50/7 ∧ x3 = 50 ∧ (x1 + x2 + x3 = 470 / 7) ∧ 
  (∀ x : ℝ, x = abs (3 * x - abs (50 - 3 * x)) → (x = x1 ∨ x = x2 ∨ x = x3))) := 
sorry

end NUMINAMATH_GPT_sum_of_solutions_l1101_110191


namespace NUMINAMATH_GPT_exchanges_divisible_by_26_l1101_110198

variables (p a d : ℕ) -- Define the variables for the number of exchanges

theorem exchanges_divisible_by_26 (t : ℕ) (h1 : p = 4 * a + d) (h2 : p = a + 5 * d) :
  ∃ k : ℕ, a + p + d = 26 * k :=
by {
  -- Replace these sorry placeholders with the actual proof where needed
  sorry
}

end NUMINAMATH_GPT_exchanges_divisible_by_26_l1101_110198


namespace NUMINAMATH_GPT_factor_expression_l1101_110135

theorem factor_expression (x : ℝ) : 
  5 * x * (x - 2) + 9 * (x - 2) - 4 * (x - 2) = 5 * (x - 2) * (x + 1) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_factor_expression_l1101_110135


namespace NUMINAMATH_GPT_infinitely_many_arithmetic_progression_triples_l1101_110144

theorem infinitely_many_arithmetic_progression_triples :
  ∃ (u v: ℤ) (a b c: ℤ), 
  (∀ n: ℤ, (a = 2 * u) ∧ 
    (b = 2 * u + v) ∧
    (c = 2 * u + 2 * v) ∧ 
    (u > 0) ∧
    (v > 0) ∧
    ∃ k m n: ℤ, 
    (a * b + 1 = k * k) ∧ 
    (b * c + 1 = m * m) ∧ 
    (c * a + 1 = n * n)) :=
sorry

end NUMINAMATH_GPT_infinitely_many_arithmetic_progression_triples_l1101_110144


namespace NUMINAMATH_GPT_evaluate_expression_l1101_110172

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 7) :
  (x^5 + 3 * y^3) / 9 = 141 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1101_110172


namespace NUMINAMATH_GPT_digit_is_two_l1101_110124

theorem digit_is_two (d : ℕ) (h : d < 10) : (∃ k : ℤ, d - 2 = 11 * k) ↔ d = 2 := 
by sorry

end NUMINAMATH_GPT_digit_is_two_l1101_110124


namespace NUMINAMATH_GPT_simplify_expression_l1101_110157

noncomputable def p (a b c x k : ℝ) := 
  k * (((x + a) ^ 2 / ((a - b) * (a - c))) +
       ((x + b) ^ 2 / ((b - a) * (b - c))) +
       ((x + c) ^ 2 / ((c - a) * (c - b))))

theorem simplify_expression (a b c k : ℝ) (h₀ : a ≠ b) (h₁ : a ≠ c) (h₂ : b ≠ c) (h₃ : k ≠ 0) :
  p a b c x k = k :=
sorry

end NUMINAMATH_GPT_simplify_expression_l1101_110157


namespace NUMINAMATH_GPT_other_root_is_seven_thirds_l1101_110105

theorem other_root_is_seven_thirds {m : ℝ} (h : ∃ r : ℝ, 3 * r * r + m * r - 7 = 0 ∧ r = -1) : 
  ∃ r' : ℝ, r' ≠ -1 ∧ 3 * r' * r' + m * r' - 7 = 0 ∧ r' = 7 / 3 :=
by
  sorry

end NUMINAMATH_GPT_other_root_is_seven_thirds_l1101_110105


namespace NUMINAMATH_GPT_find_polynomial_l1101_110111

def polynomial (a b c : ℚ) : ℚ → ℚ := λ x => a * x^2 + b * x + c

theorem find_polynomial
  (a b c : ℚ)
  (h1 : polynomial a b c (-3) = 0)
  (h2 : polynomial a b c 6 = 0)
  (h3 : polynomial a b c 2 = -24) :
  a = 6/5 ∧ b = -18/5 ∧ c = -108/5 :=
by 
  sorry

end NUMINAMATH_GPT_find_polynomial_l1101_110111


namespace NUMINAMATH_GPT_fran_speed_l1101_110103

theorem fran_speed :
  ∀ (Joann_speed Fran_time : ℝ), Joann_speed = 15 → Joann_time = 4 → Fran_time = 3.5 →
    (Joann_speed * Joann_time = Fran_time * (120 / 7)) :=
by
  intro Joann_speed Joann_time Fran_time
  sorry

end NUMINAMATH_GPT_fran_speed_l1101_110103


namespace NUMINAMATH_GPT_smallest_base_to_express_100_with_three_digits_l1101_110113

theorem smallest_base_to_express_100_with_three_digits : 
  ∃ b : ℕ, (b^2 ≤ 100 ∧ 100 < b^3) ∧ ∀ b' : ℕ, (b'^2 ≤ 100 ∧ 100 < b'^3) → b ≤ b' ∧ b = 5 :=
by
  sorry

end NUMINAMATH_GPT_smallest_base_to_express_100_with_three_digits_l1101_110113


namespace NUMINAMATH_GPT_go_stones_problem_l1101_110122

theorem go_stones_problem
  (x : ℕ) 
  (h1 : x / 7 + 40 = 555 / 5) 
  (black_stones : ℕ) 
  (h2 : black_stones = 55) :
  (x - black_stones = 442) :=
sorry

end NUMINAMATH_GPT_go_stones_problem_l1101_110122


namespace NUMINAMATH_GPT_prairie_total_area_l1101_110170

theorem prairie_total_area (dust : ℕ) (untouched : ℕ) (total : ℕ) 
  (h1 : dust = 64535) (h2 : untouched = 522) : total = dust + untouched :=
by
  sorry

end NUMINAMATH_GPT_prairie_total_area_l1101_110170


namespace NUMINAMATH_GPT_digits_are_different_probability_l1101_110119

noncomputable def prob_diff_digits : ℚ :=
  let total := 999 - 100 + 1
  let same_digits := 9
  1 - (same_digits / total)

theorem digits_are_different_probability :
  prob_diff_digits = 99 / 100 :=
by
  sorry

end NUMINAMATH_GPT_digits_are_different_probability_l1101_110119


namespace NUMINAMATH_GPT_books_needed_to_buy_clarinet_l1101_110125

def cost_of_clarinet : ℕ := 90
def initial_savings : ℕ := 10
def price_per_book : ℕ := 5
def halfway_loss : ℕ := (cost_of_clarinet - initial_savings) / 2

theorem books_needed_to_buy_clarinet 
    (cost_of_clarinet initial_savings price_per_book halfway_loss : ℕ)
    (initial_savings_lost : halfway_loss = (cost_of_clarinet - initial_savings) / 2) : 
    ((cost_of_clarinet - initial_savings + halfway_loss) / price_per_book) = 24 := 
sorry

end NUMINAMATH_GPT_books_needed_to_buy_clarinet_l1101_110125


namespace NUMINAMATH_GPT_hats_needed_to_pay_51_l1101_110134

def shirt_cost : ℕ := 5
def hat_cost : ℕ := 4
def jeans_cost : ℕ := 10
def total_amount : ℕ := 51
def num_shirts : ℕ := 3
def num_jeans : ℕ := 2

theorem hats_needed_to_pay_51 :
  ∃ (n : ℕ), total_amount = num_shirts * shirt_cost + num_jeans * jeans_cost + n * hat_cost ∧ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_hats_needed_to_pay_51_l1101_110134


namespace NUMINAMATH_GPT_total_snowballs_l1101_110145

theorem total_snowballs (Lc : ℕ) (Ch : ℕ) (Pt : ℕ)
  (h1 : Ch = Lc + 31)
  (h2 : Lc = 19)
  (h3 : Pt = 47) : 
  Ch + Lc + Pt = 116 := by
  sorry

end NUMINAMATH_GPT_total_snowballs_l1101_110145


namespace NUMINAMATH_GPT_janet_more_siblings_than_carlos_l1101_110169

theorem janet_more_siblings_than_carlos :
  ∀ (masud_siblings : ℕ),
  masud_siblings = 60 →
  (janets_siblings : ℕ) →
  janets_siblings = 4 * masud_siblings - 60 →
  (carlos_siblings : ℕ) →
  carlos_siblings = 3 * masud_siblings / 4 →
  janets_siblings - carlos_siblings = 45 :=
by
  intros masud_siblings hms janets_siblings hjs carlos_siblings hcs
  sorry

end NUMINAMATH_GPT_janet_more_siblings_than_carlos_l1101_110169


namespace NUMINAMATH_GPT_sale_price_lower_by_2_5_percent_l1101_110193

open Real

theorem sale_price_lower_by_2_5_percent (x : ℝ) : 
  let increased_price := 1.30 * x
  let sale_price := 0.75 * increased_price 
  sale_price = 0.975 * x :=
by
  let increased_price := 1.30 * x
  let sale_price := 0.75 * increased_price 
  show sale_price = 0.975 * x
  sorry

end NUMINAMATH_GPT_sale_price_lower_by_2_5_percent_l1101_110193


namespace NUMINAMATH_GPT_olympic_medals_l1101_110152

theorem olympic_medals (total_sprinters british_sprinters non_british_sprinters ways_case1 ways_case2 ways_case3 : ℕ)
  (h_total : total_sprinters = 10)
  (h_british : british_sprinters = 4)
  (h_non_british : non_british_sprinters = 6)
  (h_case1 : ways_case1 = 6 * 5 * 4)
  (h_case2 : ways_case2 = 4 * 3 * (6 * 5))
  (h_case3 : ways_case3 = (4 * 3) * (3 * 2) * 6) :
  ways_case1 + ways_case2 + ways_case3 = 912 := by
  sorry

end NUMINAMATH_GPT_olympic_medals_l1101_110152


namespace NUMINAMATH_GPT_total_lives_l1101_110114

-- Defining the number of lives for each animal according to the given conditions:
def cat_lives : ℕ := 9
def dog_lives : ℕ := cat_lives - 3
def mouse_lives : ℕ := dog_lives + 7
def elephant_lives : ℕ := 2 * cat_lives - 5
def fish_lives : ℕ := if (dog_lives + mouse_lives) < (elephant_lives / 2) then (dog_lives + mouse_lives) else elephant_lives / 2

-- The main statement we need to prove:
theorem total_lives :
  cat_lives + dog_lives + mouse_lives + elephant_lives + fish_lives = 47 :=
by
  sorry

end NUMINAMATH_GPT_total_lives_l1101_110114


namespace NUMINAMATH_GPT_douglas_votes_percentage_l1101_110189

theorem douglas_votes_percentage 
  (V : ℝ)
  (hx : 0.62 * 2 * V + 0.38 * V = 1.62 * V)
  (hy : 3 * V > 0) : 
  ((1.62 * V) / (3 * V)) * 100 = 54 := 
by
  sorry

end NUMINAMATH_GPT_douglas_votes_percentage_l1101_110189


namespace NUMINAMATH_GPT_inequality_proof_l1101_110131

theorem inequality_proof (a1 a2 a3 b1 b2 b3 : ℝ)
  (h1 : a1 ≥ a2) (h2 : a2 ≥ a3) (h3 : a3 > 0) 
  (h4 : b1 ≥ b2) (h5 : b2 ≥ b3) (h6 : b3 > 0)
  (h7 : a1 * a2 * a3 = b1 * b2 * b3)
  (h8 : a1 - a3 ≤ b1 - b3) : 
  a1 + a2 + a3 ≤ 2 * (b1 + b2 + b3) := sorry

end NUMINAMATH_GPT_inequality_proof_l1101_110131


namespace NUMINAMATH_GPT_arithmetic_mean_eqn_l1101_110127

theorem arithmetic_mean_eqn : 
  (3/5 + 6/7) / 2 = 51/70 :=
  by sorry

end NUMINAMATH_GPT_arithmetic_mean_eqn_l1101_110127


namespace NUMINAMATH_GPT_total_boxes_correct_l1101_110156

def boxes_chocolate : ℕ := 2
def boxes_sugar : ℕ := 5
def boxes_gum : ℕ := 2
def total_boxes : ℕ := boxes_chocolate + boxes_sugar + boxes_gum

theorem total_boxes_correct : total_boxes = 9 := by
  sorry

end NUMINAMATH_GPT_total_boxes_correct_l1101_110156


namespace NUMINAMATH_GPT_island_of_misfortune_l1101_110176

def statement (n : ℕ) (knight : ℕ → Prop) (liar : ℕ → Prop) : Prop :=
  ∀ k : ℕ, k < n → (
    if k = 0 then ∀ m : ℕ, (m % 2 = 1) ↔ liar m
    else if k = 1 then ∀ m : ℕ, (m % 3 = 1) ↔ liar m
    else ∀ m : ℕ, (m % (k + 1) = 1) ↔ liar m
  )

theorem island_of_misfortune :
  ∃ n : ℕ, n >= 2 ∧ statement n knight liar
:= sorry

end NUMINAMATH_GPT_island_of_misfortune_l1101_110176


namespace NUMINAMATH_GPT_ratio_of_fallen_cakes_is_one_half_l1101_110118

noncomputable def ratio_fallen_to_total (total_cakes fallen_cakes pick_up destroyed_cakes : ℕ) :=
  fallen_cakes / total_cakes

theorem ratio_of_fallen_cakes_is_one_half :
  ∀ (total_cakes fallen_cakes pick_up destroyed_cakes : ℕ),
    total_cakes = 12 →
    pick_up = fallen_cakes / 2 →
    pick_up = destroyed_cakes →
    destroyed_cakes = 3 →
    ratio_fallen_to_total total_cakes fallen_cakes pick_up destroyed_cakes = 1 / 2 :=
by
  intros total_cakes fallen_cakes pick_up destroyed_cakes h1 h2 h3 h4
  rw [h1, h4, ratio_fallen_to_total]
  -- proof goes here
  sorry

end NUMINAMATH_GPT_ratio_of_fallen_cakes_is_one_half_l1101_110118


namespace NUMINAMATH_GPT_sum_eighth_row_interior_numbers_l1101_110153

-- Define the sum of the interior numbers in the nth row of Pascal's Triangle.
def sum_interior_numbers (n : ℕ) : ℕ := 2^(n-1) - 2

-- Problem statement: Prove the sum of the interior numbers of Pascal's Triangle in the eighth row is 126,
-- given the sums for the fifth and sixth rows.
theorem sum_eighth_row_interior_numbers :
  sum_interior_numbers 5 = 14 →
  sum_interior_numbers 6 = 30 →
  sum_interior_numbers 8 = 126 :=
by
  sorry

end NUMINAMATH_GPT_sum_eighth_row_interior_numbers_l1101_110153


namespace NUMINAMATH_GPT_find_m_l1101_110178

theorem find_m 
  (h : ∀ x, (0 < x ∧ x < 2) ↔ ( - (1 / 2) * x^2 + 2 * x > m * x )) :
  m = 1 :=
sorry

end NUMINAMATH_GPT_find_m_l1101_110178


namespace NUMINAMATH_GPT_drug_price_reduction_eq_l1101_110128

variable (x : ℝ)
variable (initial_price : ℝ := 144)
variable (final_price : ℝ := 81)

theorem drug_price_reduction_eq :
  initial_price * (1 - x)^2 = final_price :=
by
  sorry

end NUMINAMATH_GPT_drug_price_reduction_eq_l1101_110128


namespace NUMINAMATH_GPT_gamma_received_eight_donuts_l1101_110136

noncomputable def total_donuts : ℕ := 40
noncomputable def delta_donuts : ℕ := 8
noncomputable def remaining_donuts : ℕ := total_donuts - delta_donuts
noncomputable def gamma_donuts : ℕ := 8
noncomputable def beta_donuts : ℕ := 3 * gamma_donuts

theorem gamma_received_eight_donuts 
  (h1 : total_donuts = 40)
  (h2 : delta_donuts = 8)
  (h3 : beta_donuts = 3 * gamma_donuts)
  (h4 : remaining_donuts = total_donuts - delta_donuts)
  (h5 : remaining_donuts = gamma_donuts + beta_donuts) :
  gamma_donuts = 8 := 
sorry

end NUMINAMATH_GPT_gamma_received_eight_donuts_l1101_110136


namespace NUMINAMATH_GPT_boy_speed_in_kmph_l1101_110186

-- Define the conditions
def side_length : ℕ := 35
def time_seconds : ℕ := 56

-- Perimeter of the square field
def perimeter : ℕ := 4 * side_length

-- Speed in meters per second
def speed_mps : ℚ := perimeter / time_seconds

-- Speed in kilometers per hour
def speed_kmph : ℚ := speed_mps * (3600 / 1000)

-- Theorem stating the boy's speed is 9 km/hr
theorem boy_speed_in_kmph : speed_kmph = 9 :=
by
  sorry

end NUMINAMATH_GPT_boy_speed_in_kmph_l1101_110186


namespace NUMINAMATH_GPT_dig_site_date_l1101_110129

theorem dig_site_date (S F T Fourth : ℤ) 
  (h₁ : F = S - 352)
  (h₂ : T = F + 3700)
  (h₃ : Fourth = 2 * T)
  (h₄ : Fourth = 8400) : S = 852 := 
by 
  sorry

end NUMINAMATH_GPT_dig_site_date_l1101_110129


namespace NUMINAMATH_GPT_opposite_of_fraction_reciprocal_of_fraction_absolute_value_of_fraction_l1101_110181

def improper_fraction : ℚ := -4/3

theorem opposite_of_fraction : -improper_fraction = 4/3 :=
by sorry

theorem reciprocal_of_fraction : (improper_fraction⁻¹) = -3/4 :=
by sorry

theorem absolute_value_of_fraction : |improper_fraction| = 4/3 :=
by sorry

end NUMINAMATH_GPT_opposite_of_fraction_reciprocal_of_fraction_absolute_value_of_fraction_l1101_110181


namespace NUMINAMATH_GPT_power_mod_l1101_110167

theorem power_mod (h : 5 ^ 200 ≡ 1 [MOD 1000]) : 5 ^ 6000 ≡ 1 [MOD 1000] :=
by
  sorry

end NUMINAMATH_GPT_power_mod_l1101_110167


namespace NUMINAMATH_GPT_unique_7tuple_exists_l1101_110196

theorem unique_7tuple_exists 
  (x : Fin 7 → ℝ) 
  (h : (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + x 6^2 = 1 / 7) 
  : ∃! (x : Fin 7 → ℝ), (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + x 6^2 = 1 / 7 :=
sorry

end NUMINAMATH_GPT_unique_7tuple_exists_l1101_110196


namespace NUMINAMATH_GPT_bus_interval_l1101_110137

theorem bus_interval (num_departures : ℕ) (total_duration : ℕ) (interval : ℕ)
  (h1 : num_departures = 11)
  (h2 : total_duration = 60)
  (h3 : interval = total_duration / (num_departures - 1)) :
  interval = 6 :=
by
  sorry

end NUMINAMATH_GPT_bus_interval_l1101_110137


namespace NUMINAMATH_GPT_unique_fraction_difference_l1101_110102

theorem unique_fraction_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) :
  (1 / x) - (1 / y) = (y - x) / (x * y) :=
by sorry

end NUMINAMATH_GPT_unique_fraction_difference_l1101_110102


namespace NUMINAMATH_GPT_simplify_expression1_simplify_expression2_l1101_110133

theorem simplify_expression1 (x : ℝ) : 
  5*x^2 + x + 3 + 4*x - 8*x^2 - 2 = -3*x^2 + 5*x + 1 :=
by
  sorry

theorem simplify_expression2 (a : ℝ) : 
  (5*a^2 + 2*a - 1) - 4*(3 - 8*a + 2*a^2) = -3*a^2 + 34*a - 13 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression1_simplify_expression2_l1101_110133


namespace NUMINAMATH_GPT_geometric_sequence_first_term_l1101_110155

theorem geometric_sequence_first_term (a b c : ℕ) 
    (h1 : 16 = a * (2^3)) 
    (h2 : 32 = a * (2^4)) : 
    a = 2 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_first_term_l1101_110155


namespace NUMINAMATH_GPT_find_k_from_roots_ratio_l1101_110174

theorem find_k_from_roots_ratio (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1 + x2 = -10)
  (h2 : x1 * x2 = k)
  (h3 : x1/x2 = 3) : 
  k = 18.75 := 
sorry

end NUMINAMATH_GPT_find_k_from_roots_ratio_l1101_110174


namespace NUMINAMATH_GPT_triangle_area_l1101_110166

theorem triangle_area (a b c : ℝ) (C : ℝ) 
  (h1 : c^2 = (a - b)^2 + 6)
  (h2 : C = Real.pi / 3) : 
  (1/2) * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_area_l1101_110166


namespace NUMINAMATH_GPT_units_digit_base7_product_l1101_110108

theorem units_digit_base7_product (a b : ℕ) (ha : a = 354) (hb : b = 78) : (a * b) % 7 = 4 := by
  sorry

end NUMINAMATH_GPT_units_digit_base7_product_l1101_110108


namespace NUMINAMATH_GPT_train_speed_l1101_110117

theorem train_speed (distance_AB : ℕ) (start_time_A : ℕ) (start_time_B : ℕ) (meet_time : ℕ) (speed_B : ℕ) (time_travel_A : ℕ) (time_travel_B : ℕ)
  (total_distance : ℕ) (distance_B_covered : ℕ) (speed_A : ℕ)
  (h1 : distance_AB = 330)
  (h2 : start_time_A = 8)
  (h3 : start_time_B = 9)
  (h4 : meet_time = 11)
  (h5 : speed_B = 75)
  (h6 : time_travel_A = meet_time - start_time_A)
  (h7 : time_travel_B = meet_time - start_time_B)
  (h8 : distance_B_covered = time_travel_B * speed_B)
  (h9 : total_distance = distance_AB)
  (h10 : total_distance = time_travel_A * speed_A + distance_B_covered):
  speed_A = 60 := 
by
  sorry

end NUMINAMATH_GPT_train_speed_l1101_110117


namespace NUMINAMATH_GPT_expand_product_l1101_110104

theorem expand_product :
  (3 * x + 4) * (x - 2) * (x + 6) = 3 * x^3 + 16 * x^2 - 20 * x - 48 :=
by
  sorry

end NUMINAMATH_GPT_expand_product_l1101_110104


namespace NUMINAMATH_GPT_hike_on_saturday_l1101_110168

-- Define the conditions
variables (x : Real) -- distance hiked on Saturday
variables (y : Real) -- distance hiked on Sunday
variables (z : Real) -- total distance hiked

-- Define given values
def hiked_on_sunday : Real := 1.6
def total_hiked : Real := 9.8

-- The hypothesis: y + x = z
axiom hike_total : y + x = z

theorem hike_on_saturday : x = 8.2 :=
by
  sorry

end NUMINAMATH_GPT_hike_on_saturday_l1101_110168


namespace NUMINAMATH_GPT_days_in_month_l1101_110179

-- The number of days in the month
variable (D : ℕ)

-- The conditions provided in the problem
def mean_daily_profit (D : ℕ) := 350
def mean_first_fifteen_days := 225
def mean_last_fifteen_days := 475
def total_profit := mean_first_fifteen_days * 15 + mean_last_fifteen_days * 15

-- The Lean statement to prove the number of days in the month
theorem days_in_month : D = 30 :=
by
  -- mean_daily_profit(D) * D should be equal to total_profit
  have h : mean_daily_profit D * D = total_profit := sorry
  -- solve for D
  sorry

end NUMINAMATH_GPT_days_in_month_l1101_110179


namespace NUMINAMATH_GPT_part_1_part_2_part_3_l1101_110161

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x + m / x

theorem part_1 (h : f 1 m = 5) : m = 4 :=
sorry

theorem part_2 (m : ℝ) (h : m = 4) : ∀ x : ℝ, f (-x) m = -f x m :=
sorry

theorem part_3 (m : ℝ) (h : m = 4) : ∀ x1 x2 : ℝ, 2 < x1 → x1 < x2 → f x1 m < f x2 m :=
sorry

end NUMINAMATH_GPT_part_1_part_2_part_3_l1101_110161


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l1101_110158

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
∀ n, a n = a1 + (n - 1) * d

-- Given condition
def given_condition (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
3 * a 9 - a 15 - a 3 = 20

-- Question to prove
def question (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
2 * a 8 - a 7 = 20

-- Main theorem
theorem arithmetic_sequence_problem (a: ℕ → ℝ) (a1 d: ℝ):
  arithmetic_sequence a a1 d →
  given_condition a a1 d →
  question a a1 d :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l1101_110158


namespace NUMINAMATH_GPT_area_of_triangle_l1101_110160

noncomputable def circumradius (a b c : ℝ) (α : ℝ) : ℝ := a / (2 * Real.sin α)

theorem area_of_triangle (A B C a b c R : ℝ) (h₁ : b * Real.cos C + c * Real.cos B = Real.sqrt 3 * R)
  (h₂ : a = 2) (h₃ : b + c = 4) : 
  1 / 2 * b * (c * Real.sin A) = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_l1101_110160


namespace NUMINAMATH_GPT_min_value_frac_sqrt_l1101_110154

theorem min_value_frac_sqrt (x : ℝ) (h : x > 1) : 
  (x + 10) / Real.sqrt (x - 1) ≥ 2 * Real.sqrt 11 :=
sorry

end NUMINAMATH_GPT_min_value_frac_sqrt_l1101_110154


namespace NUMINAMATH_GPT_rectangle_to_square_l1101_110150

theorem rectangle_to_square (length width : ℕ) (h1 : 2 * (length + width) = 40) (h2 : length - 8 = width + 2) :
  width + 2 = 7 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_rectangle_to_square_l1101_110150


namespace NUMINAMATH_GPT_cost_of_traveling_roads_is_2600_l1101_110180

-- Define the lawn, roads, and the cost parameters
def width_lawn : ℝ := 80
def length_lawn : ℝ := 60
def road_width : ℝ := 10
def cost_per_sq_meter : ℝ := 2

-- Area calculations
def area_road_1 : ℝ := road_width * length_lawn
def area_road_2 : ℝ := road_width * width_lawn
def area_intersection : ℝ := road_width * road_width

def total_area_roads : ℝ := area_road_1 + area_road_2 - area_intersection

def total_cost : ℝ := total_area_roads * cost_per_sq_meter

theorem cost_of_traveling_roads_is_2600 :
  total_cost = 2600 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_traveling_roads_is_2600_l1101_110180


namespace NUMINAMATH_GPT_emberly_total_miles_l1101_110132

noncomputable def totalMilesWalkedInMarch : ℕ :=
  let daysInMarch := 31
  let daysNotWalked := 4
  let milesPerDay := 4
  (daysInMarch - daysNotWalked) * milesPerDay

theorem emberly_total_miles : totalMilesWalkedInMarch = 108 :=
by
  sorry

end NUMINAMATH_GPT_emberly_total_miles_l1101_110132


namespace NUMINAMATH_GPT_trips_and_weights_l1101_110163

theorem trips_and_weights (x : ℕ) (w : ℕ) (trips_Bill Jean_total limit_total: ℕ)
  (h1 : x + (x + 6) = 40)
  (h2 : trips_Bill = x)
  (h3 : Jean_total = x + 6)
  (h4 : w = 7850)
  (h5 : limit_total = 8000)
  : 
  trips_Bill = 17 ∧ 
  Jean_total = 23 ∧ 
  (w : ℝ) / 40 = 196.25 := 
by 
  sorry

end NUMINAMATH_GPT_trips_and_weights_l1101_110163


namespace NUMINAMATH_GPT_a3_value_l1101_110121

-- Define the geometric sequence
def geom_seq (r : ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 * r ^ n

-- Given conditions
variables (a : ℕ → ℝ) (r : ℝ)
axiom h_geom : geom_seq r a
axiom h_a1 : a 1 = 1
axiom h_a5 : a 5 = 4

-- Goal to prove
theorem a3_value : a 3 = 2 ∨ a 3 = -2 := by
  sorry

end NUMINAMATH_GPT_a3_value_l1101_110121


namespace NUMINAMATH_GPT_equalize_money_l1101_110164

theorem equalize_money (ann_money : ℕ) (bill_money : ℕ) : 
  ann_money = 777 → 
  bill_money = 1111 → 
  ∃ x, bill_money - x = ann_money + x :=
by
  sorry

end NUMINAMATH_GPT_equalize_money_l1101_110164


namespace NUMINAMATH_GPT_monthly_incomes_l1101_110130

theorem monthly_incomes (a b c d e : ℕ) : 
  a + b = 8100 ∧ 
  b + c = 10500 ∧ 
  a + c = 8400 ∧
  (a + b + d) / 3 = 4800 ∧
  (c + d + e) / 3 = 6000 ∧
  (b + a + e) / 3 = 4500 → 
  (a = 3000 ∧ b = 5100 ∧ c = 5400 ∧ d = 6300 ∧ e = 5400) :=
by sorry

end NUMINAMATH_GPT_monthly_incomes_l1101_110130
