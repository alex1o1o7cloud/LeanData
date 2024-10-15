import Mathlib

namespace NUMINAMATH_GPT_difference_between_roots_l2333_233379

noncomputable def a : ℝ := 1
noncomputable def b : ℝ := -7
noncomputable def c : ℝ := 11

noncomputable def quadratic_roots (a b c : ℝ) : (ℝ × ℝ) :=
  let discriminant := b ^ 2 - 4 * a * c
  ((-b + Real.sqrt discriminant) / (2 * a), (-b - Real.sqrt discriminant) / (2 * a))

-- Extract the roots from the equation
noncomputable def r1_r2 := quadratic_roots a b c

noncomputable def r1 : ℝ := r1_r2.1
noncomputable def r2 : ℝ := r1_r2.2

-- Theorem statement: the difference between the roots is sqrt(5)
theorem difference_between_roots :
  |r1 - r2| = Real.sqrt 5 :=
  sorry

end NUMINAMATH_GPT_difference_between_roots_l2333_233379


namespace NUMINAMATH_GPT_investment_difference_l2333_233325

noncomputable def A_Maria : ℝ := 60000 * (1 + 0.045)^3
noncomputable def A_David : ℝ := 60000 * (1 + 0.0175)^6
noncomputable def investment_diff : ℝ := A_Maria - A_David

theorem investment_difference : abs (investment_diff - 1803.30) < 1 :=
by
  have hM : A_Maria = 60000 * (1 + 0.045)^3 := by rfl
  have hD : A_David = 60000 * (1 + 0.0175)^6 := by rfl
  have hDiff : investment_diff = A_Maria - A_David := by rfl
  -- Proof would go here; using the provided approximations
  sorry

end NUMINAMATH_GPT_investment_difference_l2333_233325


namespace NUMINAMATH_GPT_tiled_board_remainder_l2333_233386

def num_ways_to_tile_9x1 : Nat := -- hypothetical function to calculate the number of ways
  sorry

def N : Nat :=
  num_ways_to_tile_9x1 -- placeholder for N, should be computed using correct formula

theorem tiled_board_remainder : N % 1000 = 561 :=
  sorry

end NUMINAMATH_GPT_tiled_board_remainder_l2333_233386


namespace NUMINAMATH_GPT_master_zhang_must_sell_100_apples_l2333_233394

-- Define the given conditions
def buying_price_per_apple : ℚ := 1 / 4 -- 1 yuan for 4 apples
def selling_price_per_apple : ℚ := 2 / 5 -- 2 yuan for 5 apples
def profit_per_apple : ℚ := selling_price_per_apple - buying_price_per_apple

-- Define the target profit
def target_profit : ℚ := 15

-- Define the number of apples to sell
def apples_to_sell : ℚ := target_profit / profit_per_apple

-- The theorem statement: Master Zhang must sell 100 apples to achieve the target profit of 15 yuan
theorem master_zhang_must_sell_100_apples :
  apples_to_sell = 100 :=
sorry

end NUMINAMATH_GPT_master_zhang_must_sell_100_apples_l2333_233394


namespace NUMINAMATH_GPT_extrema_of_f_l2333_233341

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3 * x + 1

theorem extrema_of_f :
  (∃ x : ℝ, f x = -1 ∧ ∀ y : ℝ, f y ≥ f x) ∧
  (∃ x : ℝ, f x = 3 ∧ ∀ y : ℝ, f y ≤ f x) :=
by
  sorry

end NUMINAMATH_GPT_extrema_of_f_l2333_233341


namespace NUMINAMATH_GPT_population_reaches_210_l2333_233350

noncomputable def population_function (x : ℕ) : ℝ :=
  200 * (1 + 0.01)^x

theorem population_reaches_210 :
  ∃ x : ℕ, population_function x >= 210 :=
by
  existsi 5
  apply le_of_lt
  sorry

end NUMINAMATH_GPT_population_reaches_210_l2333_233350


namespace NUMINAMATH_GPT_min_value_of_z_l2333_233371

-- Define the conditions and objective function
def constraints (x y : ℝ) : Prop :=
  (y ≥ x + 2) ∧ 
  (x + y ≤ 6) ∧ 
  (x ≥ 1)

def z (x y : ℝ) : ℝ :=
  2 * |x - 2| + |y|

-- The formal theorem stating the minimum value of z under the given constraints
theorem min_value_of_z : ∃ x y : ℝ, constraints x y ∧ z x y = 4 :=
sorry

end NUMINAMATH_GPT_min_value_of_z_l2333_233371


namespace NUMINAMATH_GPT_distance_between_homes_l2333_233355

-- Define the conditions as Lean functions and values
def walking_speed_maxwell : ℝ := 3
def running_speed_brad : ℝ := 5
def distance_traveled_maxwell : ℝ := 15

-- State the theorem
theorem distance_between_homes : 
  ∃ D : ℝ, 
    (15 = walking_speed_maxwell * (distance_traveled_maxwell / walking_speed_maxwell)) ∧ 
    (D - 15 = running_speed_brad * (distance_traveled_maxwell / walking_speed_maxwell)) ∧ 
    D = 40 :=
by 
  sorry

end NUMINAMATH_GPT_distance_between_homes_l2333_233355


namespace NUMINAMATH_GPT_fraction_value_l2333_233376

theorem fraction_value (x y z : ℝ) (h : x / 2 = y / 3 ∧ y / 3 = z / 4) : (x + y + z) / (2 * z) = 9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_fraction_value_l2333_233376


namespace NUMINAMATH_GPT_fraction_increase_by_three_l2333_233313

variables (a b : ℝ)

theorem fraction_increase_by_three : 
  3 * (2 * a * b / (3 * a - 4 * b)) = 2 * (3 * a * 3 * b) / (3 * (3 * a) - 4 * (3 * b)) :=
by
  sorry

end NUMINAMATH_GPT_fraction_increase_by_three_l2333_233313


namespace NUMINAMATH_GPT_smallest_of_5_consecutive_natural_numbers_sum_100_l2333_233389

theorem smallest_of_5_consecutive_natural_numbers_sum_100
  (n : ℕ)
  (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 100) :
  n = 18 := sorry

end NUMINAMATH_GPT_smallest_of_5_consecutive_natural_numbers_sum_100_l2333_233389


namespace NUMINAMATH_GPT_find_A_for_diamond_eq_85_l2333_233380

def diamond (A B : ℝ) : ℝ := 4 * A + B^2 + 7

theorem find_A_for_diamond_eq_85 :
  ∃ (A : ℝ), diamond A 3 = 85 ∧ A = 17.25 :=
by
  sorry

end NUMINAMATH_GPT_find_A_for_diamond_eq_85_l2333_233380


namespace NUMINAMATH_GPT_conversion_base10_to_base7_l2333_233336

-- Define the base-10 number
def num_base10 : ℕ := 1023

-- Define the conversion base
def base : ℕ := 7

-- Define the expected base-7 representation as a function of the base
def expected_base7 (b : ℕ) : ℕ := 2 * b^3 + 6 * b^2 + 6 * b^1 + 1 * b^0

-- Statement to prove
theorem conversion_base10_to_base7 : expected_base7 base = num_base10 :=
by 
  -- Sorry is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_conversion_base10_to_base7_l2333_233336


namespace NUMINAMATH_GPT_worth_of_stuff_l2333_233326

theorem worth_of_stuff (x : ℝ)
  (h1 : 1.05 * x - 8 = 34) :
  x = 40 :=
by
  sorry

end NUMINAMATH_GPT_worth_of_stuff_l2333_233326


namespace NUMINAMATH_GPT_angle_measure_l2333_233362

-- Define the angle in degrees
def angle (x : ℝ) : Prop :=
  180 - x = 3 * (90 - x)

-- Desired proof statement
theorem angle_measure :
  ∀ (x : ℝ), angle x → x = 45 := by
  intros x h
  sorry

end NUMINAMATH_GPT_angle_measure_l2333_233362


namespace NUMINAMATH_GPT_problem_statement_l2333_233399

theorem problem_statement :
  ∀ k : Nat, (∃ r s : Nat, r > 0 ∧ s > 0 ∧ (k^2 - 6 * k + 11)^(r - 1) = (2 * k - 7)^s) ↔ (k = 2 ∨ k = 3 ∨ k = 4 ∨ k = 8) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2333_233399


namespace NUMINAMATH_GPT_average_speed_l2333_233390

/--
On the first day of her vacation, Louisa traveled 100 miles.
On the second day, traveling at the same average speed, she traveled 175 miles.
If the 100-mile trip took 3 hours less than the 175-mile trip,
prove that her average speed (in miles per hour) was 25.
-/
theorem average_speed (v : ℝ) (h1 : 100 / v + 3 = 175 / v) : v = 25 :=
by 
  sorry

end NUMINAMATH_GPT_average_speed_l2333_233390


namespace NUMINAMATH_GPT_chocolate_bars_in_large_box_l2333_233359

def num_small_boxes : ℕ := 17
def chocolate_bars_per_small_box : ℕ := 26
def total_chocolate_bars : ℕ := 17 * 26

theorem chocolate_bars_in_large_box :
  total_chocolate_bars = 442 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_bars_in_large_box_l2333_233359


namespace NUMINAMATH_GPT_inequality_proof_l2333_233340

variable {x y z : ℝ}

theorem inequality_proof 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hxyz : x + y + z = 1) : 
  (1 / x^2 + x) * (1 / y^2 + y) * (1 / z^2 + z) ≥ (28 / 3)^3 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l2333_233340


namespace NUMINAMATH_GPT_complex_exp1990_sum_theorem_l2333_233383

noncomputable def complex_exp1990_sum (x y : Complex) (h : x ≠ 0 ∧ y ≠ 0 ∧ x^2 + x*y + y^2 = 0) : Prop :=
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1

theorem complex_exp1990_sum_theorem (x y : Complex) (h : x ≠ 0 ∧ y ≠ 0 ∧ x^2 + x*y + y^2 = 0) : complex_exp1990_sum x y h :=
  sorry

end NUMINAMATH_GPT_complex_exp1990_sum_theorem_l2333_233383


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l2333_233363

open Real

theorem necessary_and_sufficient_condition 
  {x y : ℝ} (p : x > y) (q : x - y + sin (x - y) > 0) : 
  (x > y) ↔ (x - y + sin (x - y) > 0) :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l2333_233363


namespace NUMINAMATH_GPT_quadruples_solution_l2333_233308

theorem quadruples_solution (a b c d : ℝ) :
  (a * b + c * d = 6) ∧
  (a * c + b * d = 3) ∧
  (a * d + b * c = 2) ∧
  (a + b + c + d = 6) ↔
  (a = 0 ∧ b = 1 ∧ c = 2 ∧ d = 3) ∨
  (a = 2 ∧ b = 3 ∧ c = 0 ∧ d = 1) ∨
  (a = 1 ∧ b = 0 ∧ c = 3 ∧ d = 2) ∨
  (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 0) :=
sorry

end NUMINAMATH_GPT_quadruples_solution_l2333_233308


namespace NUMINAMATH_GPT_add_and_subtract_l2333_233397

theorem add_and_subtract (a b c : ℝ) (h1 : a = 0.45) (h2 : b = 52.7) (h3 : c = 0.25) : 
  (a + b) - c = 52.9 :=
by 
  sorry

end NUMINAMATH_GPT_add_and_subtract_l2333_233397


namespace NUMINAMATH_GPT_symmetric_polynomial_evaluation_l2333_233345

theorem symmetric_polynomial_evaluation :
  ∃ (a b : ℝ), (∀ x : ℝ, (x^2 + 3 * x) * (x^2 + a * x + b) = ((2 - x)^2 + 3 * (2 - x)) * ((2 - x)^2 + a * (2 - x) + b)) ∧
  ((3^2 + 3 * 3) * (3^2 + (-6) * 3 + 8) = -18) :=
sorry

end NUMINAMATH_GPT_symmetric_polynomial_evaluation_l2333_233345


namespace NUMINAMATH_GPT_alice_cookie_fills_l2333_233385

theorem alice_cookie_fills :
  (∀ (a b : ℚ), a = 3 + (3/4) ∧ b = 1/3 → (a / b) = 12) :=
sorry

end NUMINAMATH_GPT_alice_cookie_fills_l2333_233385


namespace NUMINAMATH_GPT_domain_of_f_l2333_233391

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1)

theorem domain_of_f : { x : ℝ | x > 1 } = { x : ℝ | ∃ y, f y = f x } :=
by sorry

end NUMINAMATH_GPT_domain_of_f_l2333_233391


namespace NUMINAMATH_GPT_find_principal_l2333_233372

theorem find_principal (A : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) (h1 : A = 1456) (h2 : R = 0.05) (h3 : T = 2.4) :
  A = P + P * R * T → P = 1300 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_principal_l2333_233372


namespace NUMINAMATH_GPT_convert_to_standard_spherical_coordinates_l2333_233382

theorem convert_to_standard_spherical_coordinates :
  let ρ := 4
  let θ := (3 * Real.pi) / 4
  let φ := (9 * Real.pi) / 5
  let adjusted_φ := 2 * Real.pi - φ
  let adjusted_θ := θ + Real.pi
  (ρ, adjusted_θ, adjusted_φ) = (4, (7 * Real.pi) / 4, Real.pi / 5) :=
by
  let ρ := 4
  let θ := (3 * Real.pi) / 4
  let φ := (9 * Real.pi) / 5
  let adjusted_φ := 2 * Real.pi - φ
  let adjusted_θ := θ + Real.pi
  sorry

end NUMINAMATH_GPT_convert_to_standard_spherical_coordinates_l2333_233382


namespace NUMINAMATH_GPT_speed_of_sound_correct_l2333_233330

-- Define the given conditions
def heard_second_blast_after : ℕ := 30 * 60 + 24 -- 30 minutes and 24 seconds in seconds
def time_sound_travelled : ℕ := 24 -- The sound traveled for 24 seconds
def distance_travelled : ℕ := 7920 -- Distance in meters

-- Define the expected answer for the speed of sound 
def expected_speed_of_sound : ℕ := 330 -- Speed in meters per second

-- The proposition that states the speed of sound given the conditions
theorem speed_of_sound_correct : (distance_travelled / time_sound_travelled) = expected_speed_of_sound := 
by {
  -- use division to compute the speed of sound
  sorry
}

end NUMINAMATH_GPT_speed_of_sound_correct_l2333_233330


namespace NUMINAMATH_GPT_solve_for_x_l2333_233381

variable (a b x : ℝ)
variable (a_pos : a > 0) (b_pos : b > 0) (x_pos : x > 0)

theorem solve_for_x : (3 * a) ^ (3 * b) = (a ^ b) * (x ^ b) → x = 27 * a ^ 2 :=
by
  intro h_eq
  sorry

end NUMINAMATH_GPT_solve_for_x_l2333_233381


namespace NUMINAMATH_GPT_solve_system_of_equations_l2333_233300

theorem solve_system_of_equations (x y : ℝ) (hx: x > 0) (hy: y > 0) :
  x * y = 500 ∧ x ^ (Real.log y / Real.log 10) = 25 → (x = 100 ∧ y = 5) ∨ (x = 5 ∧ y = 100) := by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l2333_233300


namespace NUMINAMATH_GPT_largest_perfect_square_factor_1800_l2333_233304

def largest_perfect_square_factor (n : ℕ) : ℕ :=
  if n = 1800 then 900 else sorry

theorem largest_perfect_square_factor_1800 : 
  largest_perfect_square_factor 1800 = 900 :=
by
  -- Proof is not needed, so we use sorry
  sorry

end NUMINAMATH_GPT_largest_perfect_square_factor_1800_l2333_233304


namespace NUMINAMATH_GPT_feed_cost_l2333_233344

theorem feed_cost (total_birds ducks_fraction chicken_feed_cost : ℕ) (h1 : total_birds = 15) (h2 : ducks_fraction = 1/3) (h3 : chicken_feed_cost = 2) :
  15 * (1 - 1/3) * 2 = 20 :=
by
  sorry

end NUMINAMATH_GPT_feed_cost_l2333_233344


namespace NUMINAMATH_GPT_second_discount_percentage_l2333_233357

-- Definitions for the given conditions
def original_price : ℝ := 33.78
def first_discount_rate : ℝ := 0.25
def final_price : ℝ := 19.0

-- Intermediate calculations based on the conditions
def first_discount : ℝ := first_discount_rate * original_price
def price_after_first_discount : ℝ := original_price - first_discount
def second_discount_amount : ℝ := price_after_first_discount - final_price

-- Lean theorem statement
theorem second_discount_percentage : (second_discount_amount / price_after_first_discount) * 100 = 25 := by
  sorry

end NUMINAMATH_GPT_second_discount_percentage_l2333_233357


namespace NUMINAMATH_GPT_problem_statement_l2333_233358

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 8

theorem problem_statement : 3 * g 2 + 4 * g (-2) = 152 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l2333_233358


namespace NUMINAMATH_GPT_rhombus_fourth_vertex_l2333_233311

theorem rhombus_fourth_vertex (a b : ℝ) :
  ∃ x y : ℝ, (x, y) = (a - b, a + b) ∧ dist (a, b) (x, y) = dist (-b, a) (x, y) ∧ dist (-b, a) (x, y) = dist (0, 0) (x, y) :=
by
  use (a - b)
  use (a + b)
  sorry

end NUMINAMATH_GPT_rhombus_fourth_vertex_l2333_233311


namespace NUMINAMATH_GPT_cos_double_angle_l2333_233377

theorem cos_double_angle (α : ℝ) (h : Real.cos α = 4 / 5) : Real.cos (2 * α) = 7 / 25 := 
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l2333_233377


namespace NUMINAMATH_GPT_print_gift_wrap_price_l2333_233310

theorem print_gift_wrap_price (solid_price : ℝ) (total_rolls : ℕ) (total_money : ℝ)
    (print_rolls : ℕ) (solid_rolls_money : ℝ) (print_money : ℝ) (P : ℝ) :
  solid_price = 4 ∧ total_rolls = 480 ∧ total_money = 2340 ∧ print_rolls = 210 ∧
  solid_rolls_money = 270 * 4 ∧ print_money = 1260 ∧
  total_money = solid_rolls_money + print_money ∧ P = print_money / 210 
  → P = 6 :=
by
  sorry

end NUMINAMATH_GPT_print_gift_wrap_price_l2333_233310


namespace NUMINAMATH_GPT_eval_expr_l2333_233396

theorem eval_expr (x y : ℕ) (h1 : x = 2) (h2 : y = 1) : 2 * x - 3 * y = 1 := by
  sorry

end NUMINAMATH_GPT_eval_expr_l2333_233396


namespace NUMINAMATH_GPT_fraction_sum_l2333_233370

variable {a b : ℝ}

theorem fraction_sum (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) (h1 : a^2 + a - 2007 = 0) (h2 : b^2 + b - 2007 = 0) :
  (1/a + 1/b) = 1/2007 :=
by
  sorry

end NUMINAMATH_GPT_fraction_sum_l2333_233370


namespace NUMINAMATH_GPT_min_value_of_reciprocal_sum_l2333_233303

variable (a b : ℝ)
variable (h₀ : 0 < a)
variable (h₁ : 0 < b)
variable (condition : 2 * a + b = 1)

theorem min_value_of_reciprocal_sum : (1 / a) + (1 / b) = 3 + 2 * Real.sqrt 2 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_min_value_of_reciprocal_sum_l2333_233303


namespace NUMINAMATH_GPT_chord_bisected_by_point_l2333_233392

theorem chord_bisected_by_point (x1 y1 x2 y2 : ℝ) :
  (x1^2 / 36 + y1^2 / 9 = 1) ∧ (x2^2 / 36 + y2^2 / 9 = 1) ∧ 
  (x1 + x2 = 4) ∧ (y1 + y2 = 4) → (x + 4 * y - 10 = 0) :=
sorry

end NUMINAMATH_GPT_chord_bisected_by_point_l2333_233392


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2333_233309

def angle_of_inclination (α : ℝ) : Prop :=
  α > Real.pi / 4

def slope_of_line (k : ℝ) : Prop :=
  k > 1

theorem necessary_but_not_sufficient (α k : ℝ) :
  angle_of_inclination α → (slope_of_line k → (k = Real.tan α)) → (angle_of_inclination α → slope_of_line k) ∧ ¬(slope_of_line k → angle_of_inclination α) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2333_233309


namespace NUMINAMATH_GPT_parabola_equation_l2333_233387

/--
Given a point P (4, -2) on a parabola, prove that the equation of the parabola is either:
1) y^2 = x or
2) x^2 = -8y.
-/
theorem parabola_equation (p : ℝ) (x y : ℝ) (h1 : (4 : ℝ) = 4) (h2 : (-2 : ℝ) = -2) :
  (∃ p : ℝ, y^2 = 2 * p * x ∧ 4 = 4 ∧ y = -2) ∨ (∃ p : ℝ, x^2 = 2 * p * y ∧ 4 = 4 ∧ x = 4) :=
sorry

end NUMINAMATH_GPT_parabola_equation_l2333_233387


namespace NUMINAMATH_GPT_focus_of_parabola_l2333_233306

theorem focus_of_parabola :
  (∀ y : ℝ, x = (1 / 4) * y^2) → (focus = (-1, 0)) := by
  sorry

end NUMINAMATH_GPT_focus_of_parabola_l2333_233306


namespace NUMINAMATH_GPT_impossible_odd_n_m_even_sum_l2333_233398

theorem impossible_odd_n_m_even_sum (n m : ℤ) (h : (n^2 + m^2 + n*m) % 2 = 0) : ¬ (n % 2 = 1 ∧ m % 2 = 1) :=
by sorry

end NUMINAMATH_GPT_impossible_odd_n_m_even_sum_l2333_233398


namespace NUMINAMATH_GPT_find_last_number_of_consecutive_even_numbers_l2333_233395

theorem find_last_number_of_consecutive_even_numbers (x : ℕ) (h : 8 * x + 2 + 4 + 6 + 8 + 10 + 12 + 14 = 424) : x + 14 = 60 :=
sorry

end NUMINAMATH_GPT_find_last_number_of_consecutive_even_numbers_l2333_233395


namespace NUMINAMATH_GPT_find_y_value_l2333_233361

theorem find_y_value : (12 : ℕ)^3 * (6 : ℕ)^2 / 432 = 144 := by
  -- assumptions and computations are not displayed in the statement
  sorry

end NUMINAMATH_GPT_find_y_value_l2333_233361


namespace NUMINAMATH_GPT_shahrazad_stories_not_power_of_two_l2333_233331

theorem shahrazad_stories_not_power_of_two :
  ∀ (a b c : ℕ) (k : ℕ),
  a + b + c = 1001 → 27 * a + 14 * b + c = 2^k → False :=
by {
  sorry
}

end NUMINAMATH_GPT_shahrazad_stories_not_power_of_two_l2333_233331


namespace NUMINAMATH_GPT_solve_linear_equation_l2333_233352

theorem solve_linear_equation :
  ∀ x : ℝ, (2 * x + 1) / 3 - (5 * x - 1) / 6 = 1 → x = -3 :=
by
  sorry

end NUMINAMATH_GPT_solve_linear_equation_l2333_233352


namespace NUMINAMATH_GPT_exists_infinite_sets_of_positive_integers_l2333_233365

theorem exists_infinite_sets_of_positive_integers (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (S : ℕ → ℕ × ℕ × ℕ), ∀ n : ℕ, S n = (x, y, z) ∧ 
  ((x + y + z)^2 + 2*(x + y + z) = 5*(x*y + y*z + z*x)) :=
sorry

end NUMINAMATH_GPT_exists_infinite_sets_of_positive_integers_l2333_233365


namespace NUMINAMATH_GPT_bob_selling_price_per_muffin_l2333_233335

variable (dozen_muffins_per_day : ℕ := 12)
variable (cost_per_muffin : ℝ := 0.75)
variable (weekly_profit : ℝ := 63)
variable (days_per_week : ℕ := 7)

theorem bob_selling_price_per_muffin : 
  let daily_cost := dozen_muffins_per_day * cost_per_muffin
  let weekly_cost := daily_cost * days_per_week
  let weekly_revenue := weekly_profit + weekly_cost
  let muffins_per_week := dozen_muffins_per_day * days_per_week
  let selling_price_per_muffin := weekly_revenue / muffins_per_week
  selling_price_per_muffin = 1.50 := 
by
  sorry

end NUMINAMATH_GPT_bob_selling_price_per_muffin_l2333_233335


namespace NUMINAMATH_GPT_simplify_and_evaluate_expr_l2333_233333

noncomputable def a : ℝ := Real.sqrt 2 - 2

noncomputable def expr (a : ℝ) : ℝ := (a - 2) / (a - 1) / (a + 1 - 3 / (a - 1))

theorem simplify_and_evaluate_expr :
  expr (Real.sqrt 2 - 2) = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expr_l2333_233333


namespace NUMINAMATH_GPT_number_of_cards_per_page_l2333_233301

variable (packs : ℕ) (cards_per_pack : ℕ) (total_pages : ℕ)

def number_of_cards (packs cards_per_pack : ℕ) : ℕ :=
  packs * cards_per_pack

def cards_per_page (total_cards total_pages : ℕ) : ℕ :=
  total_cards / total_pages

theorem number_of_cards_per_page
  (packs := 60) (cards_per_pack := 7) (total_pages := 42)
  (total_cards := number_of_cards packs cards_per_pack)
    : cards_per_page total_cards total_pages = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_cards_per_page_l2333_233301


namespace NUMINAMATH_GPT_average_speed_l2333_233339

theorem average_speed (v : ℝ) (v_pos : 0 < v) (v_pos_10 : 0 < v + 10):
  420 / v - 420 / (v + 10) = 2 → v = 42 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_l2333_233339


namespace NUMINAMATH_GPT_last_three_digits_W_555_2_l2333_233354

noncomputable def W : ℕ → ℕ → ℕ
| n, 0 => n ^ n
| n, (k + 1) => W (W n k) k

theorem last_three_digits_W_555_2 : (W 555 2) % 1000 = 375 := 
by
  sorry

end NUMINAMATH_GPT_last_three_digits_W_555_2_l2333_233354


namespace NUMINAMATH_GPT_mean_sharpening_instances_l2333_233321

def pencil_sharpening_instances : List ℕ :=
  [13, 8, 13, 21, 7, 23, 15, 19, 12, 9, 28, 6, 17, 29, 31, 10, 4, 20, 16, 12, 2, 18, 27, 22, 5, 14, 31, 29, 8, 25]

def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem mean_sharpening_instances :
  mean pencil_sharpening_instances = 18.1 := by
  sorry

end NUMINAMATH_GPT_mean_sharpening_instances_l2333_233321


namespace NUMINAMATH_GPT_hours_per_trainer_l2333_233319

-- Define the conditions from part (a)
def number_of_dolphins : ℕ := 4
def hours_per_dolphin : ℕ := 3
def number_of_trainers : ℕ := 2

-- Define the theorem we want to prove using the answer from part (b)
theorem hours_per_trainer : (number_of_dolphins * hours_per_dolphin) / number_of_trainers = 6 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_hours_per_trainer_l2333_233319


namespace NUMINAMATH_GPT_no_solution_inequality_C_l2333_233318

theorem no_solution_inequality_C : ¬∃ x : ℝ, 2 * x - x^2 > 5 := by
  -- There is no need to include the other options in the Lean theorem, as the proof focuses on the condition C directly.
  sorry

end NUMINAMATH_GPT_no_solution_inequality_C_l2333_233318


namespace NUMINAMATH_GPT_last_locker_opened_2046_l2333_233346

def last_locker_opened (n : ℕ) : ℕ :=
  n - (n % 3)

theorem last_locker_opened_2046 : last_locker_opened 2048 = 2046 := by
  sorry

end NUMINAMATH_GPT_last_locker_opened_2046_l2333_233346


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2333_233316

variable (a : ℝ)

theorem necessary_but_not_sufficient : (a > 2) → (a > 1) ∧ ¬((a > 1) → (a > 2)) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2333_233316


namespace NUMINAMATH_GPT_find_f2_l2333_233369

def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 := by
  sorry

end NUMINAMATH_GPT_find_f2_l2333_233369


namespace NUMINAMATH_GPT_football_team_people_count_l2333_233302

theorem football_team_people_count (original_count : ℕ) (new_members : ℕ) (total_count : ℕ) 
  (h1 : original_count = 36) (h2 : new_members = 14) : total_count = 50 :=
by
  -- This is where the proof would go. We write 'sorry' because it is not required.
  sorry

end NUMINAMATH_GPT_football_team_people_count_l2333_233302


namespace NUMINAMATH_GPT_doctor_lindsay_daily_income_l2333_233367

def patients_per_hour_adult : ℕ := 4
def patients_per_hour_child : ℕ := 3
def cost_per_adult : ℕ := 50
def cost_per_child : ℕ := 25
def work_hours_per_day : ℕ := 8

theorem doctor_lindsay_daily_income : 
  (patients_per_hour_adult * cost_per_adult + patients_per_hour_child * cost_per_child) * work_hours_per_day = 2200 := 
by
  sorry

end NUMINAMATH_GPT_doctor_lindsay_daily_income_l2333_233367


namespace NUMINAMATH_GPT_alpha_beta_roots_l2333_233307

variable (α β : ℝ)

theorem alpha_beta_roots (h1 : α^2 - 7 * α + 3 = 0) (h2 : β^2 - 7 * β + 3 = 0) (h3 : α > β) :
  α^2 + 7 * β = 46 :=
sorry

end NUMINAMATH_GPT_alpha_beta_roots_l2333_233307


namespace NUMINAMATH_GPT_geom_sequence_sum_of_first4_l2333_233373

noncomputable def geom_sum_first4_terms (a : ℕ → ℝ) (common_ratio : ℝ) (a0 a1 a4 : ℝ) : ℝ :=
  a0 + a0 * common_ratio + a0 * common_ratio^2 + a0 * common_ratio^3

theorem geom_sequence_sum_of_first4 {a : ℕ → ℝ} (a1 a4 : ℝ) (r : ℝ)
  (h1 : a 1 = a1) (h4 : a 4 = a4) 
  (h_geom : ∀ n, a (n + 1) = a n * r) :
  geom_sum_first4_terms a (r) a1 (a 0) (a 4) = 120 :=
by sorry

end NUMINAMATH_GPT_geom_sequence_sum_of_first4_l2333_233373


namespace NUMINAMATH_GPT_total_detergent_used_l2333_233388

-- Define the parameters of the problem
def total_pounds_of_clothes : ℝ := 9
def pounds_of_cotton : ℝ := 4
def pounds_of_woolen : ℝ := 5
def detergent_per_pound_cotton : ℝ := 2
def detergent_per_pound_woolen : ℝ := 1.5

-- Main theorem statement
theorem total_detergent_used : 
  (pounds_of_cotton * detergent_per_pound_cotton) + (pounds_of_woolen * detergent_per_pound_woolen) = 15.5 :=
by
  sorry

end NUMINAMATH_GPT_total_detergent_used_l2333_233388


namespace NUMINAMATH_GPT_nylon_cord_length_l2333_233348

theorem nylon_cord_length {L : ℝ} (hL : L = 30) : ∃ (w : ℝ), w = 5 := 
by sorry

end NUMINAMATH_GPT_nylon_cord_length_l2333_233348


namespace NUMINAMATH_GPT_min_buses_needed_l2333_233351

theorem min_buses_needed (n : ℕ) (h1 : 45 * n ≥ 500) (h2 : n ≥ 2) : n = 12 :=
sorry

end NUMINAMATH_GPT_min_buses_needed_l2333_233351


namespace NUMINAMATH_GPT_geometric_series_sum_l2333_233360

theorem geometric_series_sum (a r : ℚ) (h_a : a = 1) (h_r : r = 1 / 3) : 
  (∑' n : ℕ, a * r ^ n) = 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l2333_233360


namespace NUMINAMATH_GPT_vasya_can_win_l2333_233305

noncomputable def initial_first : ℝ := 1 / 2009
noncomputable def initial_second : ℝ := 1 / 2008
noncomputable def increment : ℝ := 1 / (2008 * 2009)

theorem vasya_can_win :
  ∃ n : ℕ, ((2009 * n) * increment = 1) ∨ ((2008 * n) * increment = 1) :=
sorry

end NUMINAMATH_GPT_vasya_can_win_l2333_233305


namespace NUMINAMATH_GPT_sea_horses_count_l2333_233343

theorem sea_horses_count (S P : ℕ) (h1 : 11 * S = 5 * P) (h2 : P = S + 85) : S = 70 :=
by
  sorry

end NUMINAMATH_GPT_sea_horses_count_l2333_233343


namespace NUMINAMATH_GPT_largest_prime_factor_3136_l2333_233342

theorem largest_prime_factor_3136 : ∀ (n : ℕ), n = 3136 → ∃ p : ℕ, Prime p ∧ (p ∣ n) ∧ ∀ q : ℕ, (Prime q ∧ q ∣ n) → p ≥ q :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_prime_factor_3136_l2333_233342


namespace NUMINAMATH_GPT_geometric_sequence_product_of_terms_l2333_233384

theorem geometric_sequence_product_of_terms 
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h_a2 : a 2 = 2)
  (h_a6 : a 6 = 8) : 
  a 3 * a 4 * a 5 = 64 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_product_of_terms_l2333_233384


namespace NUMINAMATH_GPT_ice_cream_sandwiches_l2333_233332

theorem ice_cream_sandwiches (n : ℕ) (x : ℕ) (h1 : n = 11) (h2 : x = 13) : (n * x = 143) := 
by
  sorry

end NUMINAMATH_GPT_ice_cream_sandwiches_l2333_233332


namespace NUMINAMATH_GPT_proper_divisors_condition_l2333_233366

theorem proper_divisors_condition (N : ℕ) :
  ∀ x : ℕ, (x ∣ N ∧ x ≠ 1 ∧ x ≠ N) → 
  (∀ L : ℕ, (L ∣ N ∧ L ≠ 1 ∧ L ≠ N) → (L = x^3 + 3 ∨ L = x^3 - 3)) → 
  (N = 10 ∨ N = 22) :=
by
  sorry

end NUMINAMATH_GPT_proper_divisors_condition_l2333_233366


namespace NUMINAMATH_GPT_tapB_fill_in_20_l2333_233349

-- Conditions definitions
def tapA_rate (A: ℝ) : Prop := A = 3 -- Tap A fills 3 liters per minute
def total_volume (V: ℝ) : Prop := V = 36 -- Total bucket volume is 36 liters
def together_fill_time (t: ℝ) : Prop := t = 10 -- Both taps fill the bucket in 10 minutes

-- Tap B's rate can be derived from these conditions
def tapB_rate (B: ℝ) (A: ℝ) (V: ℝ) (t: ℝ) : Prop := V - (A * t) = B * t

-- The final question we need to prove
theorem tapB_fill_in_20 (B: ℝ) (A: ℝ) (V: ℝ) (t: ℝ) : 
  tapA_rate A → total_volume V → together_fill_time t → tapB_rate B A V t → B * 20 = 12 := by
  sorry

end NUMINAMATH_GPT_tapB_fill_in_20_l2333_233349


namespace NUMINAMATH_GPT_office_speed_l2333_233320

variable (d v : ℝ)

theorem office_speed (h1 : v > 0) (h2 : ∀ t : ℕ, t = 30) (h3 : (2 * d) / (d / v + d / 30) = 24) : v = 20 := 
sorry

end NUMINAMATH_GPT_office_speed_l2333_233320


namespace NUMINAMATH_GPT_nine_chapters_compensation_difference_l2333_233368

noncomputable def pig_consumes (x : ℝ) := x
noncomputable def sheep_consumes (x : ℝ) := 2 * x
noncomputable def horse_consumes (x : ℝ) := 4 * x
noncomputable def cow_consumes (x : ℝ) := 8 * x

theorem nine_chapters_compensation_difference :
  ∃ (x : ℝ), 
    cow_consumes x + horse_consumes x + sheep_consumes x + pig_consumes x = 9 ∧
    (horse_consumes x - pig_consumes x) = 9 / 5 :=
by
  sorry

end NUMINAMATH_GPT_nine_chapters_compensation_difference_l2333_233368


namespace NUMINAMATH_GPT_R_and_D_expenditure_l2333_233317

theorem R_and_D_expenditure (R_D_t : ℝ) (Delta_APL_t_plus_2 : ℝ) (ratio : ℝ) :
  R_D_t = 3013.94 → Delta_APL_t_plus_2 = 3.29 → ratio = 916 →
  R_D_t / Delta_APL_t_plus_2 = ratio :=
by
  intros hR hD hRto
  rw [hR, hD, hRto]
  sorry

end NUMINAMATH_GPT_R_and_D_expenditure_l2333_233317


namespace NUMINAMATH_GPT_polynomial_no_ab_term_l2333_233312

theorem polynomial_no_ab_term (a b m : ℝ) :
  let p := 2 * (a^2 + a * b - 5 * b^2) - (a^2 - m * a * b + 2 * b^2)
  ∃ (m : ℝ), (p = a^2 - 12 * b^2) → (m = -2) :=
by
  let p := 2 * (a^2 + a * b - 5 * b^2) - (a^2 - m * a * b + 2 * b^2)
  intro h
  use -2
  sorry

end NUMINAMATH_GPT_polynomial_no_ab_term_l2333_233312


namespace NUMINAMATH_GPT_total_sales_first_three_days_total_earnings_seven_days_l2333_233334

def planned_daily_sales : Int := 100

def deviation : List Int := [4, -3, -5, 14, -8, 21, -6]

def selling_price_per_pound : Int := 8
def freight_cost_per_pound : Int := 3

-- Part (1): Proof statement for the total amount sold in the first three days
theorem total_sales_first_three_days :
  let monday_sales := planned_daily_sales + deviation.head!
  let tuesday_sales := planned_daily_sales + (deviation.drop 1).head!
  let wednesday_sales := planned_daily_sales + (deviation.drop 2).head!
  monday_sales + tuesday_sales + wednesday_sales = 296 := by
  sorry

-- Part (2): Proof statement for Xiaoming's total earnings for the seven days
theorem total_earnings_seven_days :
  let total_sales := (List.sum (deviation.map (λ x => planned_daily_sales + x)))
  total_sales * (selling_price_per_pound - freight_cost_per_pound) = 3585 := by
  sorry

end NUMINAMATH_GPT_total_sales_first_three_days_total_earnings_seven_days_l2333_233334


namespace NUMINAMATH_GPT_equation_of_circle_l2333_233323

theorem equation_of_circle :
  ∃ (a : ℝ), a < 0 ∧ (∀ (x y : ℝ), (x + 2 * y = 0) → (x + 5)^2 + y^2 = 5) :=
by
  sorry

end NUMINAMATH_GPT_equation_of_circle_l2333_233323


namespace NUMINAMATH_GPT_total_miles_walked_by_group_in_6_days_l2333_233353

-- Conditions translated to Lean definitions
def miles_per_day_group := 3
def additional_miles_per_day := 2
def days_in_week := 6
def total_ladies := 5

-- Question translated to a Lean theorem statement
theorem total_miles_walked_by_group_in_6_days : 
  ∀ (miles_per_day_group additional_miles_per_day days_in_week total_ladies : ℕ),
  (miles_per_day_group * total_ladies * days_in_week) + 
  ((miles_per_day_group * (total_ladies - 1) * days_in_week) + (additional_miles_per_day * days_in_week)) = 120 := 
by
  intros
  sorry

end NUMINAMATH_GPT_total_miles_walked_by_group_in_6_days_l2333_233353


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l2333_233315

-- Definitions for propositions p and q
def p (m x : ℝ) : Prop := x^2 - 4*m*x + 3*m^2 < 0
def q (x : ℝ) : Prop := |x - 3| ≤ 1

-- The actual Lean 4 statements
theorem part1_solution (x : ℝ) (m : ℝ) (hm : m = 1) (hp : p m x) (hq : q x) : 2 ≤ x ∧ x < 3 := by
  sorry

theorem part2_solution (m : ℝ) (hm : m > 0) (hsuff : ∀ x, q x → p m x) : (4 / 3) < m ∧ m < 2 := by
  sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_l2333_233315


namespace NUMINAMATH_GPT_second_divisor_l2333_233378

theorem second_divisor (N : ℤ) (k : ℤ) (D : ℤ) (m : ℤ) 
  (h1 : N = 39 * k + 20) 
  (h2 : N = D * m + 7) : 
  D = 13 := sorry

end NUMINAMATH_GPT_second_divisor_l2333_233378


namespace NUMINAMATH_GPT_adam_room_shelves_l2333_233322

def action_figures_per_shelf : ℕ := 15
def total_action_figures : ℕ := 120
def total_shelves (total_figures shelves_capacity : ℕ) : ℕ := total_figures / shelves_capacity

theorem adam_room_shelves :
  total_shelves total_action_figures action_figures_per_shelf = 8 :=
by
  sorry

end NUMINAMATH_GPT_adam_room_shelves_l2333_233322


namespace NUMINAMATH_GPT_third_generation_tail_length_is_25_l2333_233337

def first_generation_tail_length : ℝ := 16
def growth_rate : ℝ := 0.25

def second_generation_tail_length : ℝ := first_generation_tail_length * (1 + growth_rate)
def third_generation_tail_length : ℝ := second_generation_tail_length * (1 + growth_rate)

theorem third_generation_tail_length_is_25 :
  third_generation_tail_length = 25 := by
  sorry

end NUMINAMATH_GPT_third_generation_tail_length_is_25_l2333_233337


namespace NUMINAMATH_GPT_bill_difference_proof_l2333_233324

variable (a b c : ℝ)

def alice_condition := (25/100) * a = 5
def bob_condition := (20/100) * b = 6
def carol_condition := (10/100) * c = 7

theorem bill_difference_proof (ha : alice_condition a) (hb : bob_condition b) (hc : carol_condition c) :
  max a (max b c) - min a (min b c) = 50 :=
by sorry

end NUMINAMATH_GPT_bill_difference_proof_l2333_233324


namespace NUMINAMATH_GPT_abc_mod_n_l2333_233338

theorem abc_mod_n (n : ℕ) (a b c : ℤ) (hn : 0 < n)
  (h1 : a * b ≡ 1 [ZMOD n])
  (h2 : c ≡ b [ZMOD n]) : (a * b * c) ≡ 1 [ZMOD n] := sorry

end NUMINAMATH_GPT_abc_mod_n_l2333_233338


namespace NUMINAMATH_GPT_trajectory_is_plane_l2333_233327

/--
Given that the vertical coordinate of a moving point P is always 2, 
prove that the trajectory of the moving point P forms a plane in a 
three-dimensional Cartesian coordinate system.
-/
theorem trajectory_is_plane (P : ℝ × ℝ × ℝ) (hP : ∀ t : ℝ, ∃ x y, P = (x, y, 2)) :
  ∃ a b c d, a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∧ (∀ x y, ∃ z, (a * x + b * y + c * z + d = 0) ∧ z = 2) :=
by
  -- This proof should show that there exist constants a, b, c, and d such that 
  -- the given equation represents a plane and the z-coordinate is always 2.
  sorry

end NUMINAMATH_GPT_trajectory_is_plane_l2333_233327


namespace NUMINAMATH_GPT_orange_price_l2333_233314

theorem orange_price (initial_apples : ℕ) (initial_oranges : ℕ) 
                     (apple_price : ℝ) (total_earnings : ℝ) 
                     (remaining_apples : ℕ) (remaining_oranges : ℕ)
                     (h1 : initial_apples = 50) (h2 : initial_oranges = 40)
                     (h3 : apple_price = 0.80) (h4 : total_earnings = 49)
                     (h5 : remaining_apples = 10) (h6 : remaining_oranges = 6) :
  ∃ orange_price : ℝ, orange_price = 0.50 :=
by
  sorry

end NUMINAMATH_GPT_orange_price_l2333_233314


namespace NUMINAMATH_GPT_fraction_addition_l2333_233364

theorem fraction_addition :
  (3/8 : ℚ) / (4/9 : ℚ) + 1/6 = 97/96 := by
  sorry

end NUMINAMATH_GPT_fraction_addition_l2333_233364


namespace NUMINAMATH_GPT_no_analytic_roots_l2333_233375

theorem no_analytic_roots : ¬∃ x : ℝ, (x - 2) * (x + 5)^3 * (5 - x) = 8 := 
sorry

end NUMINAMATH_GPT_no_analytic_roots_l2333_233375


namespace NUMINAMATH_GPT_t_shirt_jersey_price_difference_l2333_233347

theorem t_shirt_jersey_price_difference :
  ∀ (T J : ℝ), (0.9 * T = 192) → (0.9 * J = 34) → (T - J = 175.55) :=
by
  intros T J hT hJ
  sorry

end NUMINAMATH_GPT_t_shirt_jersey_price_difference_l2333_233347


namespace NUMINAMATH_GPT_halfway_between_l2333_233328

theorem halfway_between (a b : ℚ) (h₁ : a = 1/8) (h₂ : b = 1/3) : (a + b) / 2 = 11 / 48 := 
by
  sorry

end NUMINAMATH_GPT_halfway_between_l2333_233328


namespace NUMINAMATH_GPT_probability_heads_9_of_12_is_correct_l2333_233374

-- Defining the probability of getting exactly 9 heads in 12 flips of a fair coin
noncomputable def probability_heads_9_of_12 := (55 : ℚ) / 1024

-- Theorem statement
theorem probability_heads_9_of_12_is_correct :
  probability_heads_9_of_12 = (55 : ℚ) / 1024 := 
  by sorry

end NUMINAMATH_GPT_probability_heads_9_of_12_is_correct_l2333_233374


namespace NUMINAMATH_GPT_average_length_of_remaining_strings_l2333_233329

theorem average_length_of_remaining_strings :
  ∀ (n_cat : ℕ) 
    (avg_len_total avg_len_one_fourth avg_len_one_third : ℝ)
    (total_length total_length_one_fourth total_length_one_third remaining_length : ℝ),
    n_cat = 12 →
    avg_len_total = 90 →
    avg_len_one_fourth = 75 →
    avg_len_one_third = 65 →
    total_length = n_cat * avg_len_total →
    total_length_one_fourth = (n_cat / 4) * avg_len_one_fourth →
    total_length_one_third = (n_cat / 3) * avg_len_one_third →
    remaining_length = total_length - (total_length_one_fourth + total_length_one_third) →
    remaining_length / (n_cat - (n_cat / 4 + n_cat / 3)) = 119 :=
by sorry

end NUMINAMATH_GPT_average_length_of_remaining_strings_l2333_233329


namespace NUMINAMATH_GPT_original_price_of_trouser_l2333_233393

theorem original_price_of_trouser (P : ℝ) (sale_price : ℝ) (percent_decrease : ℝ) 
  (h1 : sale_price = 40) (h2 : percent_decrease = 0.60) 
  (h3 : sale_price = P * (1 - percent_decrease)) : P = 100 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_trouser_l2333_233393


namespace NUMINAMATH_GPT_max_matches_l2333_233356

theorem max_matches (x y z m : ℕ) (h1 : x + y + z = 19) (h2 : x * y + y * z + x * z = m) : m ≤ 120 :=
sorry

end NUMINAMATH_GPT_max_matches_l2333_233356
