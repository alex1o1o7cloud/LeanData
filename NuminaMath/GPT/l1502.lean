import Mathlib

namespace NUMINAMATH_GPT_quadratic_roots_l1502_150279

-- Definitions based on problem conditions
def sum_of_roots (p q : ℝ) : Prop := p + q = 12
def abs_diff_of_roots (p q : ℝ) : Prop := |p - q| = 4

-- The theorem we want to prove
theorem quadratic_roots : ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ p q, sum_of_roots p q ∧ abs_diff_of_roots p q → a * (x - p) * (x - q) = x^2 - 12 * x + 32) := sorry

end NUMINAMATH_GPT_quadratic_roots_l1502_150279


namespace NUMINAMATH_GPT_pencil_pen_costs_l1502_150288

noncomputable def cost_of_items (p q : ℝ) : ℝ := 4 * p + 4 * q

theorem pencil_pen_costs (p q : ℝ) (h1 : 6 * p + 3 * q = 5.40) (h2 : 3 * p + 5 * q = 4.80) : cost_of_items p q = 4.80 :=
by
  sorry

end NUMINAMATH_GPT_pencil_pen_costs_l1502_150288


namespace NUMINAMATH_GPT_prob_same_color_is_correct_l1502_150297

-- Define the sides of one die
def blue_sides := 6
def yellow_sides := 8
def green_sides := 10
def purple_sides := 6
def total_sides := 30

-- Define the probability each die shows a specific color
def prob_blue := blue_sides / total_sides
def prob_yellow := yellow_sides / total_sides
def prob_green := green_sides / total_sides
def prob_purple := purple_sides / total_sides

-- The probability that both dice show the same color
def prob_same_color :=
  (prob_blue * prob_blue) + 
  (prob_yellow * prob_yellow) + 
  (prob_green * prob_green) + 
  (prob_purple * prob_purple)

-- We should prove that the computed probability is equal to the given answer
theorem prob_same_color_is_correct :
  prob_same_color = 59 / 225 := 
sorry

end NUMINAMATH_GPT_prob_same_color_is_correct_l1502_150297


namespace NUMINAMATH_GPT_find_original_sales_tax_percentage_l1502_150215

noncomputable def original_sales_tax_percentage (x : ℝ) : Prop :=
∃ (x : ℝ),
  let reduced_tax := 10 / 3 / 100;
  let market_price := 9000;
  let difference := 14.999999999999986;
  (x / 100 * market_price - reduced_tax * market_price = difference) ∧ x = 0.5

theorem find_original_sales_tax_percentage : original_sales_tax_percentage 0.5 :=
sorry

end NUMINAMATH_GPT_find_original_sales_tax_percentage_l1502_150215


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1502_150275

theorem isosceles_triangle_perimeter (perimeter_eq_tri : ℕ) (side_eq_tri : ℕ) (base_iso_tri : ℕ) (perimeter_iso_tri : ℕ) 
  (h1 : perimeter_eq_tri = 60) 
  (h2 : side_eq_tri = perimeter_eq_tri / 3) 
  (h3 : base_iso_tri = 5)
  (h4 : perimeter_iso_tri = 2 * side_eq_tri + base_iso_tri) : 
  perimeter_iso_tri = 45 := by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1502_150275


namespace NUMINAMATH_GPT_tire_price_l1502_150272

theorem tire_price {p : ℤ} (h : 4 * p + 1 = 421) : p = 105 :=
sorry

end NUMINAMATH_GPT_tire_price_l1502_150272


namespace NUMINAMATH_GPT_percentage_transactions_anthony_handled_more_l1502_150286

theorem percentage_transactions_anthony_handled_more (M A C J : ℕ) (P : ℚ)
  (hM : M = 90)
  (hJ : J = 83)
  (hCJ : J = C + 17)
  (hCA : C = (2 * A) / 3)
  (hP : P = ((A - M): ℚ) / M * 100) :
  P = 10 := by
  sorry

end NUMINAMATH_GPT_percentage_transactions_anthony_handled_more_l1502_150286


namespace NUMINAMATH_GPT_integer_values_satisfying_sqrt_condition_l1502_150294

theorem integer_values_satisfying_sqrt_condition : ∃! n : Nat, 2.5 < Real.sqrt n ∧ Real.sqrt n < 3.5 :=
by {
  sorry -- Proof to be filled in
}

end NUMINAMATH_GPT_integer_values_satisfying_sqrt_condition_l1502_150294


namespace NUMINAMATH_GPT_g_at_10_is_neg48_l1502_150269

variable (g : ℝ → ℝ)

-- Given condition
axiom functional_eqn : ∀ x y : ℝ, g x + g (3 * x + y) + 7 * x * y = g (4 * x - y) + 3 * x^2 + 2

-- Mathematical proof statement
theorem g_at_10_is_neg48 : g 10 = -48 :=
  sorry

end NUMINAMATH_GPT_g_at_10_is_neg48_l1502_150269


namespace NUMINAMATH_GPT_sum_of_series_equals_one_half_l1502_150240

theorem sum_of_series_equals_one_half : 
  (∑' k : ℕ, (1 / ((2 * k + 1) * (2 * k + 3)))) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_sum_of_series_equals_one_half_l1502_150240


namespace NUMINAMATH_GPT_diane_money_l1502_150241

-- Define the conditions
def total_cost : ℤ := 65
def additional_needed : ℤ := 38
def initial_amount : ℤ := total_cost - additional_needed

-- Theorem statement
theorem diane_money : initial_amount = 27 := by
  sorry

end NUMINAMATH_GPT_diane_money_l1502_150241


namespace NUMINAMATH_GPT_find_b_perpendicular_l1502_150218

theorem find_b_perpendicular (a b : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 1 = 0 ∧ 3 * x + b * y + 5 = 0 → 
  - (a / 2) * - (3 / b) = -1) → b = -3 := 
sorry

end NUMINAMATH_GPT_find_b_perpendicular_l1502_150218


namespace NUMINAMATH_GPT_continuity_at_2_l1502_150223

theorem continuity_at_2 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |(-3 * x^2 - 5) + 17| < ε :=
by
  sorry

end NUMINAMATH_GPT_continuity_at_2_l1502_150223


namespace NUMINAMATH_GPT_cell_value_l1502_150263

variable (P Q R S : ℕ)

-- Condition definitions
def topLeftCell (P : ℕ) : ℕ := P
def topMiddleCell (P Q : ℕ) : ℕ := P + Q
def centerCell (P Q R S : ℕ) : ℕ := P + Q + R + S
def bottomLeftCell (S : ℕ) : ℕ := S

-- Given Conditions
axiom bottomLeftCell_value : bottomLeftCell S = 13
axiom topMiddleCell_value : topMiddleCell P Q = 18
axiom centerCell_value : centerCell P Q R S = 47

-- To prove: R = 16
theorem cell_value : R = 16 :=
by
  sorry

end NUMINAMATH_GPT_cell_value_l1502_150263


namespace NUMINAMATH_GPT_rational_sqrt_of_rational_xy_l1502_150256

theorem rational_sqrt_of_rational_xy (x y : ℚ) (h : x^5 + y^5 = 2 * x^2 * y^2) :
  ∃ k : ℚ, k^2 = 1 - x * y := 
sorry

end NUMINAMATH_GPT_rational_sqrt_of_rational_xy_l1502_150256


namespace NUMINAMATH_GPT_percentage_increase_from_1200_to_1680_is_40_l1502_150295

theorem percentage_increase_from_1200_to_1680_is_40 :
  let initial_value := 1200
  let final_value := 1680
  let percentage_increase := ((final_value - initial_value) / initial_value) * 100
  percentage_increase = 40 := by
  let initial_value := 1200
  let final_value := 1680
  let percentage_increase := ((final_value - initial_value) / initial_value) * 100
  sorry

end NUMINAMATH_GPT_percentage_increase_from_1200_to_1680_is_40_l1502_150295


namespace NUMINAMATH_GPT_initial_pens_l1502_150260

theorem initial_pens (P : ℤ) (INIT : 2 * (P + 22) - 19 = 39) : P = 7 :=
by
  sorry

end NUMINAMATH_GPT_initial_pens_l1502_150260


namespace NUMINAMATH_GPT_part1_part2_l1502_150204

open Real

def f (x : ℝ) (a : ℝ) : ℝ := |x - 2| + |3 * x + a|

theorem part1 (a : ℝ) (h : a = 1) :
  {x : ℝ | f x a ≥ 5} = {x : ℝ | x ≤ -1 ∨ x ≥ 1} := by
  sorry

theorem part2 (h : ∃ x_0 : ℝ, f x_0 (a := a) + 2 * |x_0 - 2| < 3) : -9 < a ∧ a < -3 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1502_150204


namespace NUMINAMATH_GPT_number_of_rooms_l1502_150248

theorem number_of_rooms (x : ℕ) (h1 : ∀ n, 6 * (n - 1) = 5 * n + 4) : x = 10 :=
sorry

end NUMINAMATH_GPT_number_of_rooms_l1502_150248


namespace NUMINAMATH_GPT_adjacent_angles_l1502_150210

theorem adjacent_angles (α β : ℝ) (h1 : α = β + 30) (h2 : α + β = 180) : α = 105 ∧ β = 75 := by
  sorry

end NUMINAMATH_GPT_adjacent_angles_l1502_150210


namespace NUMINAMATH_GPT_no_appearance_1234_or_3269_no_reappearance_1975_from_2nd_time_l1502_150237

def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 9
  | 2 => 7
  | 3 => 5
  | n + 4 => (seq n + seq (n + 1) + seq (n + 2) + seq (n + 3)) % 10

theorem no_appearance_1234_or_3269 : 
  ¬∃ n, seq n = 1 ∧ seq (n + 1) = 2 ∧ seq (n + 2) = 3 ∧ seq (n + 3) = 4 ∨
  seq n = 3 ∧ seq (n + 1) = 2 ∧ seq (n + 2) = 6 ∧ seq (n + 3) = 9 := 
sorry

theorem no_reappearance_1975_from_2nd_time : 
  ¬∃ n > 0, seq n = 1 ∧ seq (n + 1) = 9 ∧ seq (n + 2) = 7 ∧ seq (n + 3) = 5 :=
sorry

end NUMINAMATH_GPT_no_appearance_1234_or_3269_no_reappearance_1975_from_2nd_time_l1502_150237


namespace NUMINAMATH_GPT_equal_abc_l1502_150238

theorem equal_abc {a b c : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a^2 * (b + c - a) = b^2 * (c + a - b) ∧ 
       b^2 * (c + a - b) = c^2 * (a + b - c)) : a = b ∧ b = c :=
by
  sorry

end NUMINAMATH_GPT_equal_abc_l1502_150238


namespace NUMINAMATH_GPT_medal_awarding_ways_l1502_150213

def num_sprinters := 10
def num_americans := 4
def num_kenyans := 2
def medal_positions := 3 -- gold, silver, bronze

-- The main statement to be proven
theorem medal_awarding_ways :
  let ways_case1 := 2 * 3 * 5 * 4
  let ways_case2 := 4 * 3 * 2 * 2 * 5
  ways_case1 + ways_case2 = 360 :=
by
  sorry

end NUMINAMATH_GPT_medal_awarding_ways_l1502_150213


namespace NUMINAMATH_GPT_chord_bisect_angle_l1502_150202

theorem chord_bisect_angle (AB AC : ℝ) (angle_CAB : ℝ) (h1 : AB = 2) (h2 : AC = 1) (h3 : angle_CAB = 120) : 
  ∃ x : ℝ, x = 3 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_chord_bisect_angle_l1502_150202


namespace NUMINAMATH_GPT_find_unknown_rate_l1502_150203

def cost_with_discount_and_tax (original_price : ℝ) (count : ℕ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let discounted_price := (original_price * count) * (1 - discount)
  discounted_price * (1 + tax)

theorem find_unknown_rate :
  let total_blankets := 10
  let average_price := 160
  let total_cost := total_blankets * average_price
  let cost_100_blankets := cost_with_discount_and_tax 100 3 0.05 0.12
  let cost_150_blankets := cost_with_discount_and_tax 150 5 0.10 0.15
  let cost_unknown_blankets := 2 * x
  total_cost = cost_100_blankets + cost_150_blankets + cost_unknown_blankets →
  x = 252.275 :=
by
  sorry

end NUMINAMATH_GPT_find_unknown_rate_l1502_150203


namespace NUMINAMATH_GPT_max_rows_l1502_150201

theorem max_rows (m : ℕ) : (∀ T : Matrix (Fin m) (Fin 8) (Fin 4), 
  ∀ i j : Fin m, ∀ k l : Fin 8, i ≠ j ∧ T i k = T j k ∧ T i l = T j l → k ≠ l) → m ≤ 28 :=
sorry

end NUMINAMATH_GPT_max_rows_l1502_150201


namespace NUMINAMATH_GPT_no_real_solutions_l1502_150266

theorem no_real_solutions : ∀ x : ℝ, ¬(3 * x - 2 * x + 8) ^ 2 = -|x| - 4 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_no_real_solutions_l1502_150266


namespace NUMINAMATH_GPT_power_product_is_100_l1502_150234

theorem power_product_is_100 :
  (10^0.6) * (10^0.4) * (10^0.3) * (10^0.2) * (10^0.5) = 100 :=
by
  sorry

end NUMINAMATH_GPT_power_product_is_100_l1502_150234


namespace NUMINAMATH_GPT_find_positive_n_unique_solution_l1502_150289

theorem find_positive_n_unique_solution (n : ℝ) :
  (∃ x : ℝ, 4 * x^2 + n * x + 4 = 0) → n = 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_positive_n_unique_solution_l1502_150289


namespace NUMINAMATH_GPT_solve_quadratic_l1502_150290

theorem solve_quadratic (x : ℝ) : 2 * x^2 - x = 2 ↔ x = (1 + Real.sqrt 17) / 4 ∨ x = (1 - Real.sqrt 17) / 4 := by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l1502_150290


namespace NUMINAMATH_GPT_quadratic_equation_unique_solution_l1502_150209

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  16 - 4 * a * c = 0 ∧ a + c = 5 ∧ a < c → (a, c) = (1, 4) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_unique_solution_l1502_150209


namespace NUMINAMATH_GPT_total_distance_of_drive_l1502_150212

theorem total_distance_of_drive :
  let christina_speed := 30
  let christina_time_minutes := 180
  let christina_time_hours := christina_time_minutes / 60
  let friend_speed := 40
  let friend_time := 3
  let distance_christina := christina_speed * christina_time_hours
  let distance_friend := friend_speed * friend_time
  let total_distance := distance_christina + distance_friend
  total_distance = 210 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_of_drive_l1502_150212


namespace NUMINAMATH_GPT_Nishita_preferred_shares_l1502_150299

variable (P : ℕ)

def preferred_share_dividend : ℕ := 5 * P
def common_share_dividend : ℕ := 3500 * 3  -- 3.5 * 1000

theorem Nishita_preferred_shares :
  preferred_share_dividend P + common_share_dividend = 16500 → P = 1200 :=
by
  unfold preferred_share_dividend common_share_dividend
  intro h
  sorry

end NUMINAMATH_GPT_Nishita_preferred_shares_l1502_150299


namespace NUMINAMATH_GPT_equivalent_expression_l1502_150273

variable (x y : ℝ)

def is_positive_real (r : ℝ) : Prop := r > 0

theorem equivalent_expression 
  (hx : is_positive_real x) 
  (hy : is_positive_real y) : 
  (Real.sqrt (Real.sqrt (x ^ 2 * Real.sqrt (y ^ 3)))) = x ^ (1 / 2) * y ^ (1 / 12) :=
by
  sorry

end NUMINAMATH_GPT_equivalent_expression_l1502_150273


namespace NUMINAMATH_GPT_optionB_is_difference_of_squares_l1502_150247

-- Definitions from conditions
def A_expr (x : ℝ) : ℝ := (x - 2) * (x + 1)
def B_expr (x y : ℝ) : ℝ := (x + 2 * y) * (x - 2 * y)
def C_expr (x y : ℝ) : ℝ := (x + y) * (-x - y)
def D_expr (x : ℝ) : ℝ := (-x + 1) * (x - 1)

theorem optionB_is_difference_of_squares (x y : ℝ) : B_expr x y = x^2 - 4 * y^2 :=
by
  -- Proof is intentionally left out as per instructions
  sorry

end NUMINAMATH_GPT_optionB_is_difference_of_squares_l1502_150247


namespace NUMINAMATH_GPT_probability_of_purple_marble_l1502_150245

theorem probability_of_purple_marble 
  (P_blue : ℝ) 
  (P_green : ℝ) 
  (P_purple : ℝ) 
  (h1 : P_blue = 0.25) 
  (h2 : P_green = 0.55) 
  (h3 : P_blue + P_green + P_purple = 1) 
  : P_purple = 0.20 := 
by 
  sorry

end NUMINAMATH_GPT_probability_of_purple_marble_l1502_150245


namespace NUMINAMATH_GPT_cubics_inequality_l1502_150292

theorem cubics_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hneq : a ≠ b) : a^3 + b^3 > a^2 * b + a * b^2 :=
sorry

end NUMINAMATH_GPT_cubics_inequality_l1502_150292


namespace NUMINAMATH_GPT_solve_system_of_equations_l1502_150249

theorem solve_system_of_equations :
  ∃ x : ℕ → ℝ,
  (∀ i : ℕ, i < 100 → x i > 0) ∧
  (x 0 + 1 / x 1 = 4) ∧
  (x 1 + 1 / x 2 = 1) ∧
  (x 2 + 1 / x 0 = 4) ∧
  (∀ i : ℕ, 1 ≤ i ∧ i < 99 → x (2 * i + 1) + 1 / x (2 * i + 2) = 1) ∧
  (∀ i : ℕ, 1 ≤ i ∧ i < 99 → x (2 * i + 2) + 1 / x (2 * i + 3) = 4) ∧
  (x 99 + 1 / x 0 = 1) ∧
  (∀ i : ℕ, i < 50 → x (2 * i) = 2) ∧
  (∀ i : ℕ, i < 50 → x (2 * i + 1) = 1 / 2) :=
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1502_150249


namespace NUMINAMATH_GPT_range_of_m_l1502_150239

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ x^2 + (m - 1) * x + 1 = 0) → m ≤ -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1502_150239


namespace NUMINAMATH_GPT_travis_flight_cost_l1502_150291

theorem travis_flight_cost 
  (cost_leg1 : ℕ := 1500) 
  (cost_leg2 : ℕ := 1000) 
  (discount_leg1 : ℕ := 25) 
  (discount_leg2 : ℕ := 35) : 
  cost_leg1 - (discount_leg1 * cost_leg1 / 100) + cost_leg2 - (discount_leg2 * cost_leg2 / 100) = 1775 :=
by
  sorry

end NUMINAMATH_GPT_travis_flight_cost_l1502_150291


namespace NUMINAMATH_GPT_slower_pipe_filling_time_l1502_150265

-- Definitions based on conditions
def faster_pipe_rate (S : ℝ) : ℝ := 3 * S
def combined_rate (S : ℝ) : ℝ := (faster_pipe_rate S) + S

-- Statement of what needs to be proved 
theorem slower_pipe_filling_time :
  (∀ S : ℝ, combined_rate S * 40 = 1) →
  ∃ t : ℝ, t = 160 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_slower_pipe_filling_time_l1502_150265


namespace NUMINAMATH_GPT_birch_tree_taller_than_pine_tree_l1502_150217

theorem birch_tree_taller_than_pine_tree :
  let pine_tree_height := (49 : ℚ) / 4
  let birch_tree_height := (37 : ℚ) / 2
  birch_tree_height - pine_tree_height = 25 / 4 :=
by
  sorry

end NUMINAMATH_GPT_birch_tree_taller_than_pine_tree_l1502_150217


namespace NUMINAMATH_GPT_moles_of_ammonia_formed_l1502_150282

def reaction (n_koh n_nh4i n_nh3 : ℕ) := 
  n_koh + n_nh4i + n_nh3 

theorem moles_of_ammonia_formed (n_koh : ℕ) :
  reaction n_koh 3 3 = n_koh + 3 + 3 := 
sorry

end NUMINAMATH_GPT_moles_of_ammonia_formed_l1502_150282


namespace NUMINAMATH_GPT_sum_of_squares_not_perfect_square_l1502_150227

theorem sum_of_squares_not_perfect_square (n : ℕ) (h : n > 4) :
  ¬ (∃ k : ℕ, 10 * n^2 + 10 * n + 85 = k^2) :=
sorry

end NUMINAMATH_GPT_sum_of_squares_not_perfect_square_l1502_150227


namespace NUMINAMATH_GPT_mean_of_remaining_l1502_150208

variable (a b c : ℝ)
variable (mean_of_four : ℝ := 90)
variable (largest : ℝ := 105)

theorem mean_of_remaining (h1 : (a + b + c + largest) / 4 = mean_of_four) : (a + b + c) / 3 = 85 := by
  sorry

end NUMINAMATH_GPT_mean_of_remaining_l1502_150208


namespace NUMINAMATH_GPT_least_positive_integer_property_l1502_150287

theorem least_positive_integer_property : 
  ∃ (n d : ℕ) (p : ℕ) (h₁ : 1 ≤ d) (h₂ : d ≤ 9) (h₃ : p ≥ 2), 
  (10^p * d = 24 * n) ∧ (∃ k : ℕ, (n = 100 * 10^(p-2) / 3) ∧ (900 = 8 * 10^p + 100 / 3 * 10^(p-2))) := sorry

end NUMINAMATH_GPT_least_positive_integer_property_l1502_150287


namespace NUMINAMATH_GPT_geometric_sequence_a_l1502_150225

open Real

theorem geometric_sequence_a (a : ℝ) (r : ℝ) (h1 : 20 * r = a) (h2 : a * r = 5/4) (h3 : 0 < a) : a = 5 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_geometric_sequence_a_l1502_150225


namespace NUMINAMATH_GPT_mike_gave_pens_l1502_150254

theorem mike_gave_pens (M : ℕ) 
  (initial_pens : ℕ := 5) 
  (pens_after_mike : ℕ := initial_pens + M)
  (pens_after_cindy : ℕ := 2 * pens_after_mike)
  (pens_after_sharon : ℕ := pens_after_cindy - 10)
  (final_pens : ℕ := 40) : 
  pens_after_sharon = final_pens → M = 20 := 
by 
  sorry

end NUMINAMATH_GPT_mike_gave_pens_l1502_150254


namespace NUMINAMATH_GPT_find_value_of_expression_l1502_150228

theorem find_value_of_expression (x y : ℝ)
  (h1 : 5 * x + y = 19)
  (h2 : x + 3 * y = 1) :
  3 * x + 2 * y = 10 :=
sorry

end NUMINAMATH_GPT_find_value_of_expression_l1502_150228


namespace NUMINAMATH_GPT_crossing_time_correct_l1502_150235

def length_of_train : ℝ := 150 -- Length of the train in meters
def speed_of_train_km_per_hr : ℝ := 72 -- Speed of the train in km/hr
def length_of_bridge : ℝ := 132 -- Length of the bridge in meters

noncomputable def speed_of_train_m_per_s : ℝ := (speed_of_train_km_per_hr * 1000) / 3600 -- Speed of the train in m/s

noncomputable def time_to_cross_bridge : ℝ := (length_of_train + length_of_bridge) / speed_of_train_m_per_s -- Time in seconds

theorem crossing_time_correct : time_to_cross_bridge = 14.1 := by
  sorry

end NUMINAMATH_GPT_crossing_time_correct_l1502_150235


namespace NUMINAMATH_GPT_cleaning_time_ratio_l1502_150220

/-- 
Given that Lilly and Fiona together take a total of 480 minutes to clean a room and Fiona
was cleaning for 360 minutes, prove that the ratio of the time Lilly spent cleaning 
to the total time spent cleaning the room is 1:4.
-/
theorem cleaning_time_ratio (total_time minutes Fiona_time : ℕ) 
  (h1 : total_time = 480)
  (h2 : Fiona_time = 360) : 
  (total_time - Fiona_time) / total_time = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_cleaning_time_ratio_l1502_150220


namespace NUMINAMATH_GPT_planes_parallel_l1502_150298

theorem planes_parallel (n1 n2 : ℝ × ℝ × ℝ)
  (h1 : n1 = (2, -1, 0)) 
  (h2 : n2 = (-4, 2, 0)) :
  ∃ k : ℝ, n2 = k • n1 := by
  -- Proof is beyond the scope of this exercise.
  sorry

end NUMINAMATH_GPT_planes_parallel_l1502_150298


namespace NUMINAMATH_GPT_red_ball_probability_l1502_150264

-- Define the conditions
def total_balls : ℕ := 10
def yellow_balls : ℕ := 1
def green_balls : ℕ := 3
def red_balls : ℕ := total_balls - yellow_balls - green_balls

-- Define the probability function
def probability_of_red_ball (total red : ℕ) : ℚ := red / total

-- The main theorem statement to prove
theorem red_ball_probability :
  probability_of_red_ball total_balls red_balls = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_red_ball_probability_l1502_150264


namespace NUMINAMATH_GPT_distance_between_A_and_B_l1502_150253

theorem distance_between_A_and_B 
    (Time_E : ℝ) (Time_F : ℝ) (D_AC : ℝ) (V_ratio : ℝ)
    (E_time : Time_E = 3) (F_time : Time_F = 4) 
    (AC_distance : D_AC = 300) (speed_ratio : V_ratio = 4) : 
    ∃ D_AB : ℝ, D_AB = 900 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_A_and_B_l1502_150253


namespace NUMINAMATH_GPT_max_mixed_gender_groups_l1502_150268

theorem max_mixed_gender_groups (b g : ℕ) (h_b : b = 31) (h_g : g = 32) : 
  ∃ max_groups, max_groups = min (b / 2) (g / 3) :=
by
  use 10
  sorry

end NUMINAMATH_GPT_max_mixed_gender_groups_l1502_150268


namespace NUMINAMATH_GPT_binom_1450_2_eq_1050205_l1502_150283

def binom_coefficient (n k : ℕ) : ℕ :=
  n.choose k

theorem binom_1450_2_eq_1050205 : binom_coefficient 1450 2 = 1050205 :=
by {
  sorry
}

end NUMINAMATH_GPT_binom_1450_2_eq_1050205_l1502_150283


namespace NUMINAMATH_GPT_miles_driven_l1502_150296

-- Definitions based on the conditions
def years : ℕ := 9
def months_in_a_year : ℕ := 12
def months_in_a_period : ℕ := 4
def miles_per_period : ℕ := 37000

-- The proof statement
theorem miles_driven : years * months_in_a_year / months_in_a_period * miles_per_period = 999000 := 
sorry

end NUMINAMATH_GPT_miles_driven_l1502_150296


namespace NUMINAMATH_GPT_manager_monthly_salary_l1502_150214

theorem manager_monthly_salary (average_salary_20 : ℝ) (new_average_salary_21 : ℝ) (m : ℝ) 
  (h1 : average_salary_20 = 1300) 
  (h2 : new_average_salary_21 = 1400) 
  (h3 : 20 * average_salary_20 + m = 21 * new_average_salary_21) : 
  m = 3400 := 
by 
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_manager_monthly_salary_l1502_150214


namespace NUMINAMATH_GPT_bob_wins_game_l1502_150284

theorem bob_wins_game : 
  ∀ n : ℕ, 0 < n → 
  (∃ k ≥ 1, ∀ m : ℕ, 0 < m → (∃ a : ℕ, a ≥ 1 ∧ m - a*a = 0) ∨ 
    (∃ k : ℕ, k ≥ 1 ∧ (m = m^k → ¬ (∃ a : ℕ, a ≥ 1 ∧ m - a*a = 0)))
  ) :=
sorry

end NUMINAMATH_GPT_bob_wins_game_l1502_150284


namespace NUMINAMATH_GPT_correct_operation_l1502_150243

theorem correct_operation (a b : ℝ) : (-a * b^2)^2 = a^2 * b^4 := 
by 
  sorry

end NUMINAMATH_GPT_correct_operation_l1502_150243


namespace NUMINAMATH_GPT_carlos_marbles_l1502_150229

theorem carlos_marbles :
  ∃ N : ℕ, N > 2 ∧
  (N % 6 = 2) ∧
  (N % 7 = 2) ∧
  (N % 8 = 2) ∧
  (N % 11 = 2) ∧
  N = 3698 :=
by
  sorry

end NUMINAMATH_GPT_carlos_marbles_l1502_150229


namespace NUMINAMATH_GPT_initial_catfish_count_l1502_150278

theorem initial_catfish_count (goldfish : ℕ) (remaining_fish : ℕ) (disappeared_fish : ℕ) (catfish : ℕ) :
  goldfish = 7 → 
  remaining_fish = 15 → 
  disappeared_fish = 4 → 
  catfish + goldfish = 19 →
  catfish = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_initial_catfish_count_l1502_150278


namespace NUMINAMATH_GPT_women_in_department_l1502_150231

theorem women_in_department : 
  ∀ (total_students men women : ℕ) (men_percentage women_percentage : ℝ),
  men_percentage = 0.70 →
  women_percentage = 0.30 →
  men = 420 →
  total_students = men / men_percentage →
  women = total_students * women_percentage →
  women = 180 :=
by
  intros total_students men women men_percentage women_percentage
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_women_in_department_l1502_150231


namespace NUMINAMATH_GPT_blue_bird_high_school_team_arrangement_l1502_150232

theorem blue_bird_high_school_team_arrangement : 
  let girls := 2
  let boys := 3
  let girls_permutations := Nat.factorial girls
  let boys_permutations := Nat.factorial boys
  girls_permutations * boys_permutations = 12 := by
  sorry

end NUMINAMATH_GPT_blue_bird_high_school_team_arrangement_l1502_150232


namespace NUMINAMATH_GPT_soja_finished_fraction_l1502_150276

def pages_finished (x pages_left total_pages : ℕ) : Prop :=
  x - pages_left = 100 ∧ x + pages_left = total_pages

noncomputable def fraction_finished (x total_pages : ℕ) : ℚ :=
  x / total_pages

theorem soja_finished_fraction (x : ℕ) (h1 : pages_finished x (x - 100) 300) :
  fraction_finished x 300 = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_soja_finished_fraction_l1502_150276


namespace NUMINAMATH_GPT_smallest_part_of_80_divided_by_proportion_l1502_150293

theorem smallest_part_of_80_divided_by_proportion (x : ℕ) (h1 : 1 * x + 3 * x + 5 * x + 7 * x = 80) : x = 5 :=
sorry

end NUMINAMATH_GPT_smallest_part_of_80_divided_by_proportion_l1502_150293


namespace NUMINAMATH_GPT_max_path_length_is_32_l1502_150255
-- Import the entire Mathlib library to use its definitions and lemmas

-- Definition of the problem setup
def number_of_edges_4x4_grid : Nat := 
  let total_squares := 4 * 4
  let total_edges_per_square := 4
  total_squares * total_edges_per_square

-- Definitions of internal edges shared by adjacent squares
def distinct_edges_4x4_grid : Nat := 
  let horizontal_lines := 5 * 4
  let vertical_lines := 5 * 4
  horizontal_lines + vertical_lines

-- Calculate the maximum length of the path
def max_length_of_path_4x4_grid : Nat := 
  let degree_3_nodes := 8
  distinct_edges_4x4_grid - degree_3_nodes

-- Main statement: Prove that the maximum length of the path is 32
theorem max_path_length_is_32 : max_length_of_path_4x4_grid = 32 := by
  -- Definitions for clarity and correctness
  have h1 : number_of_edges_4x4_grid = 64 := rfl
  have h2 : distinct_edges_4x4_grid = 40 := rfl
  have h3 : max_length_of_path_4x4_grid = 32 := rfl
  exact h3

end NUMINAMATH_GPT_max_path_length_is_32_l1502_150255


namespace NUMINAMATH_GPT_reduced_less_than_scaled_l1502_150246

-- Define the conditions
def original_flow_rate : ℝ := 5.0
def reduced_flow_rate : ℝ := 2.0
def scaled_flow_rate : ℝ := 0.6 * original_flow_rate

-- State the theorem we need to prove
theorem reduced_less_than_scaled : scaled_flow_rate - reduced_flow_rate = 1.0 := 
by
  -- insert the detailed proof steps here
  sorry

end NUMINAMATH_GPT_reduced_less_than_scaled_l1502_150246


namespace NUMINAMATH_GPT_campers_in_two_classes_l1502_150224

-- Definitions of the sets and conditions
variable (S A R : Finset ℕ)
variable (n : ℕ)
variable (x : ℕ)

-- Given conditions
axiom hyp1 : S.card = 20
axiom hyp2 : A.card = 20
axiom hyp3 : R.card = 20
axiom hyp4 : (S ∩ A ∩ R).card = 4
axiom hyp5 : (S \ (A ∪ R)).card + (A \ (S ∪ R)).card + (R \ (S ∪ A)).card = 24

-- The hypothesis that n = |S ∪ A ∪ R|
axiom hyp6 : n = (S ∪ A ∪ R).card

-- Statement to be proven in Lean
theorem campers_in_two_classes : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_campers_in_two_classes_l1502_150224


namespace NUMINAMATH_GPT_minimum_possible_area_l1502_150200

theorem minimum_possible_area (l w l_min w_min : ℝ) (hl : l = 5) (hw : w = 7) 
  (hl_min : l_min = l - 0.5) (hw_min : w_min = w - 0.5) : 
  l_min * w_min = 29.25 :=
by
  sorry

end NUMINAMATH_GPT_minimum_possible_area_l1502_150200


namespace NUMINAMATH_GPT_alpha_in_fourth_quadrant_l1502_150270

def point_in_third_quadrant (α : ℝ) : Prop :=
  (Real.tan α < 0) ∧ (Real.sin α < 0)

theorem alpha_in_fourth_quadrant (α : ℝ) (h : point_in_third_quadrant α) : 
  α ∈ Set.Ioc (3 * Real.pi / 2) (2 * Real.pi) :=
by sorry

end NUMINAMATH_GPT_alpha_in_fourth_quadrant_l1502_150270


namespace NUMINAMATH_GPT_simplify_expression_correct_l1502_150205

def simplify_expression (y : ℝ) : ℝ :=
  (3 * y - 2) * (5 * y ^ 12 + 3 * y ^ 11 + 5 * y ^ 9 + y ^ 8)

theorem simplify_expression_correct (y : ℝ) :
  simplify_expression y = 15 * y ^ 13 - y ^ 12 + 6 * y ^ 11 + 5 * y ^ 10 - 7 * y ^ 9 - 2 * y ^ 8 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_correct_l1502_150205


namespace NUMINAMATH_GPT_exponentiation_of_squares_l1502_150280

theorem exponentiation_of_squares :
  ((Real.sqrt 2 + 1)^2000 * (Real.sqrt 2 - 1)^2000 = 1) :=
by
  sorry

end NUMINAMATH_GPT_exponentiation_of_squares_l1502_150280


namespace NUMINAMATH_GPT_clock_angle_at_3_30_l1502_150261

theorem clock_angle_at_3_30 
    (deg_per_hour: Real := 30)
    (full_circle_deg: Real := 360)
    (hours_on_clock: Real := 12)
    (hour_hand_extra_deg: Real := 30 / 2)
    (hour_hand_deg: Real := 3 * deg_per_hour + hour_hand_extra_deg)
    (minute_hand_deg: Real := 6 * deg_per_hour) : 
    hour_hand_deg = 105 ∧ minute_hand_deg = 180 ∧ (minute_hand_deg - hour_hand_deg) = 75 := 
sorry

-- The problem specifies to write the theorem statement only, without the proof steps.

end NUMINAMATH_GPT_clock_angle_at_3_30_l1502_150261


namespace NUMINAMATH_GPT_girls_more_than_boys_l1502_150252

theorem girls_more_than_boys (total_students boys : ℕ) (h : total_students = 466) (b : boys = 127) (gt : total_students - boys > boys) :
  total_students - 2 * boys = 212 := by
  sorry

end NUMINAMATH_GPT_girls_more_than_boys_l1502_150252


namespace NUMINAMATH_GPT_minimum_value_of_3a_plus_b_l1502_150230

theorem minimum_value_of_3a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 / a + 1 / b = 2) : 
  3 * a + b ≥ (7 + 2 * Real.sqrt 6) / 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_3a_plus_b_l1502_150230


namespace NUMINAMATH_GPT_range_of_e_l1502_150277

theorem range_of_e (a b c d e : ℝ)
  (h1 : a + b + c + d + e = 8)
  (h2 : a^2 + b^2 + c^2 + d^2 + e^2 = 16) :
  0 ≤ e ∧ e ≤ 16 / 5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_e_l1502_150277


namespace NUMINAMATH_GPT_telephone_number_fraction_calculation_l1502_150251

theorem telephone_number_fraction_calculation :
  let valid_phone_numbers := 7 * 10^6
  let special_phone_numbers := 10^5
  (special_phone_numbers / valid_phone_numbers : ℚ) = 1 / 70 :=
by
  sorry

end NUMINAMATH_GPT_telephone_number_fraction_calculation_l1502_150251


namespace NUMINAMATH_GPT_time_to_cross_signal_post_l1502_150250

def train_length := 600 -- in meters
def bridge_length := 5400 -- in meters (5.4 kilometers)
def crossing_time_bridge := 6 * 60 -- in seconds (6 minutes)
def speed := bridge_length / crossing_time_bridge -- in meters per second

theorem time_to_cross_signal_post : 
  (600 / speed) = 40 :=
by
  sorry

end NUMINAMATH_GPT_time_to_cross_signal_post_l1502_150250


namespace NUMINAMATH_GPT_sixteen_is_sixtyfour_percent_l1502_150281

theorem sixteen_is_sixtyfour_percent (x : ℝ) (h : 16 / x = 64 / 100) : x = 25 :=
by sorry

end NUMINAMATH_GPT_sixteen_is_sixtyfour_percent_l1502_150281


namespace NUMINAMATH_GPT_solve_for_A_l1502_150211

def diamond (A B : ℝ) := 4 * A + 3 * B + 7

theorem solve_for_A : diamond A 5 = 71 → A = 12.25 := by
  intro h
  unfold diamond at h
  sorry

end NUMINAMATH_GPT_solve_for_A_l1502_150211


namespace NUMINAMATH_GPT_ax0_eq_b_condition_l1502_150259

theorem ax0_eq_b_condition (a b x0 : ℝ) (h : a < 0) : (ax0 = b) ↔ (∀ x : ℝ, (1/2 * a * x^2 - b * x) ≤ (1/2 * a * x0^2 - b * x0)) :=
sorry

end NUMINAMATH_GPT_ax0_eq_b_condition_l1502_150259


namespace NUMINAMATH_GPT_shifts_needed_l1502_150244

-- Given definitions
def total_workers : ℕ := 12
def workers_per_shift : ℕ := 2
def total_ways_to_assign : ℕ := 23760

-- Prove the number of shifts needed
theorem shifts_needed : total_workers / workers_per_shift = 6 := by
  sorry

end NUMINAMATH_GPT_shifts_needed_l1502_150244


namespace NUMINAMATH_GPT_cosine_of_difference_l1502_150226

theorem cosine_of_difference (α : ℝ) (h : Real.sin (α + π / 6) = 1 / 3) :
  Real.cos (α - π / 3) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cosine_of_difference_l1502_150226


namespace NUMINAMATH_GPT_prob_A_wins_4_consecutive_prob_fifth_game_needed_prob_C_ultimate_winner_l1502_150267

-- Conditions for the game settings
def initial_conditions (a b c : ℕ) : Prop :=
  a = 0 ∧ b = 0 ∧ c = 0

-- Probability of a player winning any game
def win_probability : ℚ := 1 / 2 

-- Probability calculation for A winning four consecutive games
theorem prob_A_wins_4_consecutive :
  win_probability ^ 4 = 1 / 16 :=
by
  sorry

-- Probability calculation for needing a fifth game to be played
theorem prob_fifth_game_needed :
  1 - 4 * (win_probability ^ 4) = 3 / 4 :=
by
  sorry

-- Probability calculation for C being the ultimate winner
theorem prob_C_ultimate_winner :
  1 - 2 * (9 / 32) = 7 / 16 :=
by
  sorry

end NUMINAMATH_GPT_prob_A_wins_4_consecutive_prob_fifth_game_needed_prob_C_ultimate_winner_l1502_150267


namespace NUMINAMATH_GPT_verify_cube_modifications_l1502_150262

-- Definitions and conditions from the problem
def side_length : ℝ := 9
def initial_volume : ℝ := side_length^3
def initial_surface_area : ℝ := 6 * side_length^2

def volume_remaining : ℝ := 639
def surface_area_remaining : ℝ := 510

-- The theorem proving the volume and surface area of the remaining part after carving the cross-shaped groove
theorem verify_cube_modifications :
  initial_volume - (initial_volume - volume_remaining) = 639 ∧
  510 = surface_area_remaining :=
by
  sorry

end NUMINAMATH_GPT_verify_cube_modifications_l1502_150262


namespace NUMINAMATH_GPT_max_two_alphas_l1502_150274

theorem max_two_alphas (k : ℕ) (α : ℕ → ℝ) (hα : ∀ n, ∃! i p : ℕ, n = ⌊p * α i⌋ + 1) : k ≤ 2 := 
sorry

end NUMINAMATH_GPT_max_two_alphas_l1502_150274


namespace NUMINAMATH_GPT_moses_percentage_l1502_150285

theorem moses_percentage (P : ℝ) (T : ℝ) (E : ℝ) (total_amount : ℝ) (moses_more : ℝ)
  (h1 : total_amount = 50)
  (h2 : moses_more = 5)
  (h3 : T = E)
  (h4 : P / 100 * total_amount = E + moses_more)
  (h5 : 2 * E = (1 - P / 100) * total_amount) :
  P = 40 :=
by
  sorry

end NUMINAMATH_GPT_moses_percentage_l1502_150285


namespace NUMINAMATH_GPT_prove_inequality_l1502_150222

variables {a b c A B C k : ℝ}

-- Define the conditions
def conditions (a b c A B C k : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ k > 0 ∧
  a + A = k ∧ b + B = k ∧ c + C = k

-- Define the theorem to be proven
theorem prove_inequality (a b c A B C k : ℝ) (h : conditions a b c A B C k) :
  a * B + b * C + c * A ≤ k^2 :=
sorry

end NUMINAMATH_GPT_prove_inequality_l1502_150222


namespace NUMINAMATH_GPT_picture_area_l1502_150258

-- Given dimensions of the paper
def paper_width : ℝ := 8.5
def paper_length : ℝ := 10

-- Given margins
def margin : ℝ := 1.5

-- Calculated dimensions of the picture
def picture_width := paper_width - 2 * margin
def picture_length := paper_length - 2 * margin

-- Statement to prove
theorem picture_area : picture_width * picture_length = 38.5 := by
  -- skipped the proof
  sorry

end NUMINAMATH_GPT_picture_area_l1502_150258


namespace NUMINAMATH_GPT_fiona_pairs_l1502_150257

-- Define the combinatorial calculation using the combination formula
def combination (n k : ℕ) := n.choose k

-- The main theorem stating that the number of pairs from 6 people is 15
theorem fiona_pairs : combination 6 2 = 15 :=
by
  sorry

end NUMINAMATH_GPT_fiona_pairs_l1502_150257


namespace NUMINAMATH_GPT_total_money_is_2800_l1502_150236

-- Define variables for money
def Cecil_money : ℕ := 600
def Catherine_money : ℕ := 2 * Cecil_money - 250
def Carmela_money : ℕ := 2 * Cecil_money + 50

-- Assertion to prove the total money 
theorem total_money_is_2800 : Cecil_money + Catherine_money + Carmela_money = 2800 :=
by
  -- placeholder proof
  sorry

end NUMINAMATH_GPT_total_money_is_2800_l1502_150236


namespace NUMINAMATH_GPT_true_propositions_in_reverse_neg_neg_reverse_l1502_150242

theorem true_propositions_in_reverse_neg_neg_reverse (a b : ℕ) : 
  (¬ (a ≠ 0 → a * b ≠ 0) ∧ ∃ (a : ℕ), (a = 0 ∧ a * b ≠ 0) ∨ (a ≠ 0 ∧ a * b = 0) ∧ ¬ (¬ ∃ (a : ℕ), a ≠ 0 ∧ a * b ≠ 0 ∧ ¬ ∃ (a : ℕ), a = 0 ∧ a * b = 0)) ∧ (0 = 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_true_propositions_in_reverse_neg_neg_reverse_l1502_150242


namespace NUMINAMATH_GPT_initial_production_rate_l1502_150221

variable (x : ℕ) (t : ℝ)

-- Conditions
def produces_initial (x : ℕ) (t : ℝ) : Prop := x * t = 60
def produces_subsequent : Prop := 60 * 1 = 60
def overall_average (t : ℝ) : Prop := 72 = 120 / (t + 1)

-- Goal: Prove the initial production rate
theorem initial_production_rate : 
  (∃ t : ℝ, produces_initial x t ∧ produces_subsequent ∧ overall_average t) → x = 90 := 
  by
    sorry

end NUMINAMATH_GPT_initial_production_rate_l1502_150221


namespace NUMINAMATH_GPT_prob_draw_correct_l1502_150219

-- Given conditions
def prob_A_wins : ℝ := 0.40
def prob_A_not_lose : ℝ := 0.90

-- Definition to be proved
def prob_draw : ℝ := prob_A_not_lose - prob_A_wins

theorem prob_draw_correct : prob_draw = 0.50 := by
  sorry

end NUMINAMATH_GPT_prob_draw_correct_l1502_150219


namespace NUMINAMATH_GPT_falsity_of_proposition_implies_a_range_l1502_150216

theorem falsity_of_proposition_implies_a_range (a : ℝ) : 
  (¬ ∃ x₀ : ℝ, a * Real.sin x₀ + Real.cos x₀ ≥ 2) →
  a ∈ Set.Ioo (-Real.sqrt 3) (Real.sqrt 3) :=
by 
  sorry

end NUMINAMATH_GPT_falsity_of_proposition_implies_a_range_l1502_150216


namespace NUMINAMATH_GPT_find_x_l1502_150206

theorem find_x (x : ℝ) (h : 0.75 * x = (1 / 3) * x + 110) : x = 264 :=
sorry

end NUMINAMATH_GPT_find_x_l1502_150206


namespace NUMINAMATH_GPT_car_trip_time_l1502_150271

theorem car_trip_time (walking_mixed: 1.5 = 1.25 + x) 
                      (walking_both: 2.5 = 2 * 1.25) : 
  2 * x * 60 = 30 :=
by sorry

end NUMINAMATH_GPT_car_trip_time_l1502_150271


namespace NUMINAMATH_GPT_crayons_received_l1502_150207

theorem crayons_received (crayons_left : ℕ) (crayons_lost_given_away : ℕ) (lost_twice_given : ∃ (G L : ℕ), L = 2 * G ∧ L + G = crayons_lost_given_away) :
  crayons_left = 2560 →
  crayons_lost_given_away = 9750 →
  ∃ (total_crayons_received : ℕ), total_crayons_received = 12310 :=
by
  intros h1 h2
  obtain ⟨G, L, hL, h_sum⟩ := lost_twice_given
  sorry -- Proof goes here

end NUMINAMATH_GPT_crayons_received_l1502_150207


namespace NUMINAMATH_GPT_sequence_add_l1502_150233

theorem sequence_add (x y : ℝ) (h1 : x = 81 * (1 / 3)) (h2 : y = x * (1 / 3)) : x + y = 36 :=
sorry

end NUMINAMATH_GPT_sequence_add_l1502_150233
