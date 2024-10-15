import Mathlib

namespace NUMINAMATH_GPT_min_sine_range_l1602_160209

noncomputable def min_sine_ratio (α β γ : ℝ) := min (Real.sin β / Real.sin α) (Real.sin γ / Real.sin β)

theorem min_sine_range (α β γ : ℝ) (h1 : 0 < α) (h2 : α ≤ β) (h3 : β ≤ γ) (h4 : α + β + γ = Real.pi) :
  1 ≤ min_sine_ratio α β γ ∧ min_sine_ratio α β γ < (1 + Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_sine_range_l1602_160209


namespace NUMINAMATH_GPT_person_age_is_30_l1602_160205

-- Definitions based on the conditions
def age (x : ℕ) := x
def age_5_years_hence (x : ℕ) := x + 5
def age_5_years_ago (x : ℕ) := x - 5

-- The main theorem to prove
theorem person_age_is_30 (x : ℕ) (h : 3 * age_5_years_hence x - 3 * age_5_years_ago x = age x) : x = 30 :=
by
  sorry

end NUMINAMATH_GPT_person_age_is_30_l1602_160205


namespace NUMINAMATH_GPT_quadratic_equation_same_solutions_l1602_160285

theorem quadratic_equation_same_solutions :
  ∃ b c : ℝ, (b, c) = (1, -7) ∧ (∀ x : ℝ, (x - 3 = 4 ∨ 3 - x = 4) ↔ (x^2 + b * x + c = 0)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_same_solutions_l1602_160285


namespace NUMINAMATH_GPT_no_nonzero_integer_solution_l1602_160283

theorem no_nonzero_integer_solution 
(a b c n : ℤ) (h : 6 * (6 * a ^ 2 + 3 * b ^ 2 + c ^ 2) = 5 * n ^ 2) : 
a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := 
sorry

end NUMINAMATH_GPT_no_nonzero_integer_solution_l1602_160283


namespace NUMINAMATH_GPT_circle_radius_l1602_160264

theorem circle_radius (k r : ℝ) (h : k > 8) 
  (h1 : r = |k - 8|)
  (h2 : r = k / Real.sqrt 5) : 
  r = 8 * Real.sqrt 5 + 8 := 
sorry

end NUMINAMATH_GPT_circle_radius_l1602_160264


namespace NUMINAMATH_GPT_sin_value_given_cos_condition_l1602_160284

theorem sin_value_given_cos_condition (theta : ℝ) (h : Real.cos (5 * Real.pi / 12 - theta) = 1 / 3) :
  Real.sin (Real.pi / 12 + theta) = 1 / 3 :=
sorry

end NUMINAMATH_GPT_sin_value_given_cos_condition_l1602_160284


namespace NUMINAMATH_GPT_part1_part2_l1602_160229

def custom_op (a b : ℤ) : ℤ := a^2 - b + a * b

theorem part1  : custom_op (-3) (-2) = 17 := by
  sorry

theorem part2 : custom_op (-2) (custom_op (-3) (-2)) = -47 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1602_160229


namespace NUMINAMATH_GPT_range_of_m_l1602_160206

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (Real.exp x) * (2 * x - 1) - m * x + m

def exists_unique_int_n (m : ℝ) : Prop :=
∃! n : ℤ, f n m < 0

theorem range_of_m {m : ℝ} (h : m < 1) (h2 : exists_unique_int_n m) : 
  (Real.exp 1) * (1 / 2) ≤ m ∧ m < 1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1602_160206


namespace NUMINAMATH_GPT_find_a_from_derivative_l1602_160245

-- Define the function f(x) = ax^3 + 3x^2 - 6
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - 6

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 6 * x

-- State the theorem to prove that a = 10/3 given f'(-1) = 4
theorem find_a_from_derivative (a : ℝ) (h : f' a (-1) = 4) : a = 10 / 3 := 
  sorry

end NUMINAMATH_GPT_find_a_from_derivative_l1602_160245


namespace NUMINAMATH_GPT_ivanov_voted_against_kuznetsov_l1602_160251

theorem ivanov_voted_against_kuznetsov
    (members : List String)
    (vote : String → String)
    (majority_dismissed : (String × Nat))
    (petrov_statement : String)
    (ivanov_concluded : Bool) :
  members = ["Ivanov", "Petrov", "Sidorov", "Kuznetsov"] →
  (∀ x ∈ members, vote x ∈ members ∧ vote x ≠ x) →
  majority_dismissed = ("Ivanov", 3) →
  petrov_statement = "Petrov voted against Kuznetsov" →
  ivanov_concluded = True →
  vote "Ivanov" = "Kuznetsov" :=
by
  intros members_cond vote_cond majority_cond petrov_cond ivanov_cond
  sorry

end NUMINAMATH_GPT_ivanov_voted_against_kuznetsov_l1602_160251


namespace NUMINAMATH_GPT_tap_b_fill_time_l1602_160256

theorem tap_b_fill_time (t : ℝ) (h1 : t > 0) : 
  (∀ (A_fill B_fill together_fill : ℝ), 
    A_fill = 1/45 ∧ 
    B_fill = 1/t ∧ 
    together_fill = A_fill + B_fill ∧ 
    (9 * A_fill) + (23 * B_fill) = 1) → 
    t = 115 / 4 :=
by
  sorry

end NUMINAMATH_GPT_tap_b_fill_time_l1602_160256


namespace NUMINAMATH_GPT_speed_ratio_A_to_B_l1602_160237

variables {u v : ℝ}

axiom perp_lines_intersect_at_o : true
axiom points_move_along_lines_at_constant_speed : true
axiom point_A_at_O_B_500_yards_away_at_t_0 : true
axiom after_2_minutes_A_and_B_equidistant : 2 * u = 500 - 2 * v
axiom after_10_minutes_A_and_B_equidistant : 10 * u = 10 * v - 500

theorem speed_ratio_A_to_B : u / v = 2 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_speed_ratio_A_to_B_l1602_160237


namespace NUMINAMATH_GPT_inequality_solutions_l1602_160262

theorem inequality_solutions (a : ℝ) (h_pos : 0 < a) 
  (h_ineq_1 : ∃! x : ℕ, 10 < a ^ x ∧ a ^ x < 100) : ∃! x : ℕ, 100 < a ^ x ∧ a ^ x < 1000 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solutions_l1602_160262


namespace NUMINAMATH_GPT_abs_neg_2023_l1602_160200

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end NUMINAMATH_GPT_abs_neg_2023_l1602_160200


namespace NUMINAMATH_GPT_simplify_exponent_multiplication_l1602_160289

theorem simplify_exponent_multiplication :
  (10 ^ 0.25) * (10 ^ 0.15) * (10 ^ 0.35) * (10 ^ 0.05) * (10 ^ 0.85) * (10 ^ 0.35) = 10 ^ 2 := by
  sorry

end NUMINAMATH_GPT_simplify_exponent_multiplication_l1602_160289


namespace NUMINAMATH_GPT_optimal_strategies_and_value_l1602_160202

-- Define the payoff matrix for the two-player zero-sum game
def payoff_matrix : Matrix (Fin 2) (Fin 2) ℕ := ![![12, 22], ![32, 2]]

-- Define the optimal mixed strategies for both players
def optimal_strategy_row_player : Fin 2 → ℚ
| 0 => 3 / 4
| 1 => 1 / 4

def optimal_strategy_column_player : Fin 2 → ℚ
| 0 => 1 / 2
| 1 => 1 / 2

-- Define the value of the game
def value_of_game := (17 : ℚ)

theorem optimal_strategies_and_value :
  (∀ i j, (optimal_strategy_row_player 0 * payoff_matrix 0 j + optimal_strategy_row_player 1 * payoff_matrix 1 j = value_of_game) ∧
           (optimal_strategy_column_player 0 * payoff_matrix i 0 + optimal_strategy_column_player 1 * payoff_matrix i 1 = value_of_game)) :=
by 
  -- sorry is used as a placeholder for the proof
  sorry

end NUMINAMATH_GPT_optimal_strategies_and_value_l1602_160202


namespace NUMINAMATH_GPT_angle_in_triangle_PQR_l1602_160243

theorem angle_in_triangle_PQR
  (Q P R : ℝ)
  (h1 : P = 2 * Q)
  (h2 : R = 5 * Q)
  (h3 : Q + P + R = 180) : 
  P = 45 := 
by sorry

end NUMINAMATH_GPT_angle_in_triangle_PQR_l1602_160243


namespace NUMINAMATH_GPT_number_of_sandwiches_l1602_160207

-- Define the constants and assumptions

def soda_cost : ℤ := 1
def number_of_sodas : ℤ := 3
def cost_of_sodas : ℤ := number_of_sodas * soda_cost

def number_of_soups : ℤ := 2
def soup_cost : ℤ := cost_of_sodas
def cost_of_soups : ℤ := number_of_soups * soup_cost

def sandwich_cost : ℤ := 3 * soup_cost
def total_cost : ℤ := 18

-- The mathematical statement we want to prove
theorem number_of_sandwiches :
  ∃ n : ℤ, (n * sandwich_cost + cost_of_sodas + cost_of_soups = total_cost) ∧ n = 1 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sandwiches_l1602_160207


namespace NUMINAMATH_GPT_book_loss_percentage_l1602_160267

theorem book_loss_percentage (CP SP_profit SP_loss : ℝ) (L : ℝ) 
  (h1 : CP = 50) 
  (h2 : SP_profit = CP + 0.09 * CP) 
  (h3 : SP_loss = CP - L / 100 * CP) 
  (h4 : SP_profit - SP_loss = 9) : 
  L = 9 :=
by
  sorry

end NUMINAMATH_GPT_book_loss_percentage_l1602_160267


namespace NUMINAMATH_GPT_sales_of_stationery_accessories_l1602_160280

def percentage_of_sales_notebooks : ℝ := 25
def percentage_of_sales_markers : ℝ := 40
def total_sales_percentage : ℝ := 100

theorem sales_of_stationery_accessories : 
  percentage_of_sales_notebooks + percentage_of_sales_markers = 65 → 
  total_sales_percentage - (percentage_of_sales_notebooks + percentage_of_sales_markers) = 35 :=
by
  sorry

end NUMINAMATH_GPT_sales_of_stationery_accessories_l1602_160280


namespace NUMINAMATH_GPT_tan_45_eq_one_l1602_160255

theorem tan_45_eq_one 
  (θ : ℝ)
  (hθ : θ = 45)
  (sin_45_eq : Real.sin (θ * Real.pi / 180) = Real.sqrt 2 / 2)
  (cos_45_eq : Real.cos (θ * Real.pi /180) = Real.sqrt 2 / 2)
  (tan_def : ∀ x : ℝ, 0 < Real.cos x → Real.tan x = Real.sin x / Real.cos x) : 
  Real.tan (θ * Real.pi / 180) = 1 := sorry

end NUMINAMATH_GPT_tan_45_eq_one_l1602_160255


namespace NUMINAMATH_GPT_right_triangle_sides_l1602_160263

theorem right_triangle_sides (a d : ℝ) (k : ℕ) (h_pos_a : 0 < a) (h_pos_d : 0 < d) (h_pos_k : 0 < k) :
  (a = 3) ∧ (d = 1) ∧ (k = 2) ↔ (a^2 + (a + d)^2 = (a + k * d)^2) :=
by 
  sorry

end NUMINAMATH_GPT_right_triangle_sides_l1602_160263


namespace NUMINAMATH_GPT_three_pow_12_mul_three_pow_8_equals_243_pow_4_l1602_160225

theorem three_pow_12_mul_three_pow_8_equals_243_pow_4 : 3^12 * 3^8 = 243^4 := 
by sorry

end NUMINAMATH_GPT_three_pow_12_mul_three_pow_8_equals_243_pow_4_l1602_160225


namespace NUMINAMATH_GPT_other_root_l1602_160230

theorem other_root (m n : ℝ) (h : (3 : ℂ) + (1 : ℂ) * Complex.I ∈ {x : ℂ | x^2 + ↑m * x + ↑n = 0}) : 
    (3 : ℂ) - (1 : ℂ) * Complex.I ∈ {x : ℂ | x^2 + ↑m * x + ↑n = 0} :=
sorry

end NUMINAMATH_GPT_other_root_l1602_160230


namespace NUMINAMATH_GPT_mod_11_residue_l1602_160266

theorem mod_11_residue :
  (312 ≡ 4 [MOD 11]) ∧
  (47 ≡ 3 [MOD 11]) ∧
  (154 ≡ 0 [MOD 11]) ∧
  (22 ≡ 0 [MOD 11]) →
  (312 + 6 * 47 + 8 * 154 + 5 * 22 ≡ 0 [MOD 11]) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_mod_11_residue_l1602_160266


namespace NUMINAMATH_GPT_unique_intersection_point_l1602_160293

theorem unique_intersection_point (m : ℝ) :
  (∀ x : ℝ, ((m + 1) * x^2 - 2 * (m + 1) * x - 1 = 0) → x = -1) ↔ m = -2 :=
by
  sorry

end NUMINAMATH_GPT_unique_intersection_point_l1602_160293


namespace NUMINAMATH_GPT_amount_exceeds_l1602_160247

theorem amount_exceeds (N : ℕ) (A : ℕ) (h1 : N = 1925) (h2 : N / 7 - N / 11 = A) :
  A = 100 :=
sorry

end NUMINAMATH_GPT_amount_exceeds_l1602_160247


namespace NUMINAMATH_GPT_songs_in_first_two_albums_l1602_160232

/-
Beyonce releases 5 different singles on iTunes.
She releases 2 albums that each has some songs.
She releases 1 album that has 20 songs.
Beyonce has released 55 songs in total.
Prove that the total number of songs in the first two albums is 30.
-/

theorem songs_in_first_two_albums {A B : ℕ} 
  (h1 : 5 + A + B + 20 = 55) : 
  A + B = 30 :=
by
  sorry

end NUMINAMATH_GPT_songs_in_first_two_albums_l1602_160232


namespace NUMINAMATH_GPT_sum_abs_diff_is_18_l1602_160288

noncomputable def sum_of_possible_abs_diff (a b c d : ℝ) : ℝ :=
  let possible_values := [
      abs ((a + 2) - (d - 7)),
      abs ((a + 2) - (d + 1)),
      abs ((a + 2) - (d - 1)),
      abs ((a + 2) - (d + 7)),
      abs ((a - 2) - (d - 7)),
      abs ((a - 2) - (d + 1)),
      abs ((a - 2) - (d - 1)),
      abs ((a - 2) - (d + 7))
  ]
  possible_values.foldl (· + ·) 0

theorem sum_abs_diff_is_18 (a b c d : ℝ) (h1 : abs (a - b) = 2) (h2 : abs (b - c) = 3) (h3 : abs (c - d) = 4) :
  sum_of_possible_abs_diff a b c d = 18 := by
  sorry

end NUMINAMATH_GPT_sum_abs_diff_is_18_l1602_160288


namespace NUMINAMATH_GPT_average_of_remaining_two_numbers_l1602_160261

theorem average_of_remaining_two_numbers 
  (avg_6 : ℝ) (avg1_2 : ℝ) (avg2_2 : ℝ)
  (n1 n2 n3 : ℕ)
  (h_avg6 : n1 = 6 ∧ avg_6 = 4.60)
  (h_avg1_2 : n2 = 2 ∧ avg1_2 = 3.4)
  (h_avg2_2 : n3 = 2 ∧ avg2_2 = 3.8) :
  ∃ avg_rem2 : ℝ, avg_rem2 = 6.6 :=
by {
  sorry
}

end NUMINAMATH_GPT_average_of_remaining_two_numbers_l1602_160261


namespace NUMINAMATH_GPT_product_of_divisors_of_30_l1602_160257

open Nat

def divisors_of_30 : List ℕ := [1, 2, 3, 5, 6, 10, 15, 30]

theorem product_of_divisors_of_30 :
  (divisors_of_30.foldr (· * ·) 1) = 810000 := by
  sorry

end NUMINAMATH_GPT_product_of_divisors_of_30_l1602_160257


namespace NUMINAMATH_GPT_product_fraction_simplification_l1602_160213

theorem product_fraction_simplification :
  (1 - (1 / 3)) * (1 - (1 / 4)) * (1 - (1 / 5)) = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_product_fraction_simplification_l1602_160213


namespace NUMINAMATH_GPT_foci_ellipsoid_hyperboloid_l1602_160240

theorem foci_ellipsoid_hyperboloid (a b : ℝ) 
(h1 : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1 → dist (0,y) (0, 5) = 5)
(h2 : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1 → dist (x,0) (7, 0) = 7) :
  |a * b| = Real.sqrt 444 := sorry

end NUMINAMATH_GPT_foci_ellipsoid_hyperboloid_l1602_160240


namespace NUMINAMATH_GPT_sum_of_seven_consecutive_integers_l1602_160218

theorem sum_of_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_seven_consecutive_integers_l1602_160218


namespace NUMINAMATH_GPT_problem1_problem2_l1602_160281

-- Problem 1
theorem problem1 : 2 * Real.cos (30 * Real.pi / 180) - Real.tan (60 * Real.pi / 180) + Real.sin (45 * Real.pi / 180) * Real.cos (45 * Real.pi / 180) = 1 / 2 :=
by sorry

-- Problem 2
theorem problem2 : (-1) ^ 2023 + 2 * Real.sin (45 * Real.pi / 180) - Real.cos (30 * Real.pi / 180) + Real.sin (60 * Real.pi / 180) + (Real.tan (60 * Real.pi / 180)) ^ 2 = 2 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1602_160281


namespace NUMINAMATH_GPT__l1602_160275

lemma power_of_a_point_theorem (AP BP CP DP : ℝ) (hAP : AP = 5) (hCP : CP = 2) (h_theorem : AP * BP = CP * DP) :
  BP / DP = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT__l1602_160275


namespace NUMINAMATH_GPT_unattainable_y_l1602_160270

theorem unattainable_y (x : ℝ) (h1 : x ≠ -3/2) : y = (1 - x) / (2 * x + 3) -> ¬(y = -1 / 2) :=
by sorry

end NUMINAMATH_GPT_unattainable_y_l1602_160270


namespace NUMINAMATH_GPT_initial_bottles_of_water_l1602_160216

theorem initial_bottles_of_water {B : ℕ} (h1 : 100 - (6 * B + 5) = 71) : B = 4 :=
by
  sorry

end NUMINAMATH_GPT_initial_bottles_of_water_l1602_160216


namespace NUMINAMATH_GPT_solution_set_l1602_160279

theorem solution_set (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l1602_160279


namespace NUMINAMATH_GPT_no_real_sqrt_neg_six_pow_three_l1602_160249

theorem no_real_sqrt_neg_six_pow_three : 
  ∀ x : ℝ, 
    (¬ ∃ y : ℝ, y * y = -6 ^ 3) :=
by
  sorry

end NUMINAMATH_GPT_no_real_sqrt_neg_six_pow_three_l1602_160249


namespace NUMINAMATH_GPT_ironman_age_greater_than_16_l1602_160294

variable (Ironman_age : ℕ)
variable (Thor_age : ℕ := 1456)
variable (CaptainAmerica_age : ℕ := Thor_age / 13)
variable (PeterParker_age : ℕ := CaptainAmerica_age / 7)

theorem ironman_age_greater_than_16
  (Thor_13_times_CaptainAmerica : Thor_age = 13 * CaptainAmerica_age)
  (CaptainAmerica_7_times_PeterParker : CaptainAmerica_age = 7 * PeterParker_age)
  (Thor_age_given : Thor_age = 1456) :
  Ironman_age > 16 :=
by
  sorry

end NUMINAMATH_GPT_ironman_age_greater_than_16_l1602_160294


namespace NUMINAMATH_GPT_necessary_condition_for_x_gt_5_l1602_160250

theorem necessary_condition_for_x_gt_5 (x : ℝ) : x > 5 → x > 3 :=
by
  intros h
  exact lt_trans (show 3 < 5 from by linarith) h

end NUMINAMATH_GPT_necessary_condition_for_x_gt_5_l1602_160250


namespace NUMINAMATH_GPT_ellipse_equation_x_axis_ellipse_equation_y_axis_parabola_equation_x_axis_parabola_equation_y_axis_l1602_160217

-- Define the conditions for the ellipse problem
def major_axis_length : ℝ := 10
def focal_length : ℝ := 4

-- Define the conditions for the parabola problem
def point_P : ℝ × ℝ := (-2, -4)

-- The equations to be proven
theorem ellipse_equation_x_axis :
  2 * (5 : ℝ) = major_axis_length ∧ 2 * (2 : ℝ) = focal_length →
  (5 : ℝ)^2 - (2 : ℝ)^2 = 21 →
  (∀ x y : ℝ, x^2 / 25 + y^2 / 21 = 1) := sorry

theorem ellipse_equation_y_axis :
  2 * (5 : ℝ) = major_axis_length ∧ 2 * (2 : ℝ) = focal_length →
  (5 : ℝ)^2 - (2 : ℝ)^2 = 21 →
  (∀ x y : ℝ, y^2 / 25 + x^2 / 21 = 1) := sorry

theorem parabola_equation_x_axis :
  point_P = (-2, -4) →
  (∀ x y : ℝ, y^2 = -8 * x) := sorry

theorem parabola_equation_y_axis :
  point_P = (-2, -4) →
  (∀ x y : ℝ, x^2 = -y) := sorry

end NUMINAMATH_GPT_ellipse_equation_x_axis_ellipse_equation_y_axis_parabola_equation_x_axis_parabola_equation_y_axis_l1602_160217


namespace NUMINAMATH_GPT_annual_interest_earned_l1602_160246
noncomputable section

-- Define the total money
def total_money : ℝ := 3200

-- Define the first part of the investment
def P1 : ℝ := 800

-- Define the second part of the investment as total money minus the first part
def P2 : ℝ := total_money - P1

-- Define the interest rates for both parts
def rate1 : ℝ := 0.03
def rate2 : ℝ := 0.05

-- Define the time period (in years)
def time_period : ℝ := 1

-- Define the interest earned from each part
def interest1 : ℝ := P1 * rate1 * time_period
def interest2 : ℝ := P2 * rate2 * time_period

-- The total interest earned from both investments
def total_interest : ℝ := interest1 + interest2

-- The proof statement
theorem annual_interest_earned : total_interest = 144 := by
  sorry

end NUMINAMATH_GPT_annual_interest_earned_l1602_160246


namespace NUMINAMATH_GPT_total_floors_combined_l1602_160269

-- Let C be the number of floors in the Chrysler Building
-- Let L be the number of floors in the Leeward Center
-- Given that C = 23 and C = L + 11
-- Prove that the total floors in both buildings combined equals 35

theorem total_floors_combined (C L : ℕ) (h1 : C = 23) (h2 : C = L + 11) : C + L = 35 :=
by
  sorry

end NUMINAMATH_GPT_total_floors_combined_l1602_160269


namespace NUMINAMATH_GPT_line_equation_isosceles_triangle_l1602_160259

theorem line_equation_isosceles_triangle 
  (x y : ℝ)
  (l : ℝ → ℝ → Prop)
  (h1 : l 3 2)
  (h2 : ∀ x y, l x y → (x = y ∨ x + y = 2 * intercept))
  (intercept : ℝ) :
  l x y ↔ (x - y = 1 ∨ x + y = 5) :=
by
  sorry

end NUMINAMATH_GPT_line_equation_isosceles_triangle_l1602_160259


namespace NUMINAMATH_GPT_solve_for_y_l1602_160231

theorem solve_for_y (y : ℕ) : 9^y = 3^12 → y = 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1602_160231


namespace NUMINAMATH_GPT_notebook_costs_2_20_l1602_160277

theorem notebook_costs_2_20 (n c : ℝ) (h1 : n + c = 2.40) (h2 : n = 2 + c) : n = 2.20 :=
by
  sorry

end NUMINAMATH_GPT_notebook_costs_2_20_l1602_160277


namespace NUMINAMATH_GPT_rolls_combinations_l1602_160219

theorem rolls_combinations {n k : ℕ} (h_n : n = 4) (h_k : k = 5) :
  (Nat.choose (n + k - 1) k) = 56 :=
by
  rw [h_n, h_k]
  norm_num
  sorry

end NUMINAMATH_GPT_rolls_combinations_l1602_160219


namespace NUMINAMATH_GPT_shadow_length_when_eight_meters_away_l1602_160203

noncomputable def lamp_post_height : ℝ := 8
noncomputable def sam_initial_distance : ℝ := 12
noncomputable def shadow_initial_length : ℝ := 4
noncomputable def sam_initial_height : ℝ := 2 -- derived from the problem's steps

theorem shadow_length_when_eight_meters_away :
  ∀ (L : ℝ), (L * lamp_post_height) / (lamp_post_height + sam_initial_distance - shadow_initial_length) = 2 → L = 8 / 3 :=
by
  intro L
  sorry

end NUMINAMATH_GPT_shadow_length_when_eight_meters_away_l1602_160203


namespace NUMINAMATH_GPT_length_of_bridge_l1602_160297

theorem length_of_bridge (speed_kmh : ℝ) (time_min : ℝ) (length_m : ℝ) :
  speed_kmh = 5 → time_min = 15 → length_m = 1250 :=
by
  sorry

end NUMINAMATH_GPT_length_of_bridge_l1602_160297


namespace NUMINAMATH_GPT_suresh_completion_time_l1602_160295

theorem suresh_completion_time (S : ℕ) 
  (ashu_time : ℕ := 30) 
  (suresh_work_time : ℕ := 9) 
  (ashu_remaining_time : ℕ := 12) 
  (ashu_fraction : ℚ := ashu_remaining_time / ashu_time) :
  (suresh_work_time / S + ashu_fraction = 1) → S = 15 :=
by
  intro h
  -- Proof here
  sorry

end NUMINAMATH_GPT_suresh_completion_time_l1602_160295


namespace NUMINAMATH_GPT_carnival_rent_l1602_160260

-- Define the daily popcorn earnings
def daily_popcorn : ℝ := 50
-- Define the multiplier for cotton candy earnings
def multiplier : ℝ := 3
-- Define the number of operational days
def days : ℕ := 5
-- Define the cost of ingredients
def ingredients_cost : ℝ := 75
-- Define the net earnings after expenses
def net_earnings : ℝ := 895
-- Define the total earnings from selling popcorn for all days
def total_popcorn_earnings : ℝ := daily_popcorn * days
-- Define the total earnings from selling cotton candy for all days
def total_cottoncandy_earnings : ℝ := (daily_popcorn * multiplier) * days
-- Define the total earnings before expenses
def total_earnings : ℝ := total_popcorn_earnings + total_cottoncandy_earnings
-- Define the amount remaining after paying the rent (which includes net earnings and ingredient cost)
def remaining_after_rent : ℝ := net_earnings + ingredients_cost
-- Define the rent
def rent : ℝ := total_earnings - remaining_after_rent

theorem carnival_rent : rent = 30 := by
  sorry

end NUMINAMATH_GPT_carnival_rent_l1602_160260


namespace NUMINAMATH_GPT_smallest_number_l1602_160244

theorem smallest_number (a b c d : ℤ) (h_a : a = 0) (h_b : b = -1) (h_c : c = -4) (h_d : d = 5) : 
  c < b ∧ c < a ∧ c < d :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_number_l1602_160244


namespace NUMINAMATH_GPT_triangle_inequality_satisfied_for_n_six_l1602_160222

theorem triangle_inequality_satisfied_for_n_six :
  ∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
  (a + b > c ∧ a + c > b ∧ b + c > a) := sorry

end NUMINAMATH_GPT_triangle_inequality_satisfied_for_n_six_l1602_160222


namespace NUMINAMATH_GPT_prime_solution_l1602_160298

theorem prime_solution (p : ℕ) (x y : ℕ) (hp : Prime p) (hx : 0 < x) (hy : 0 < y) :
  p^x = y^3 + 1 → p = 2 ∨ p = 3 :=
by
  sorry

end NUMINAMATH_GPT_prime_solution_l1602_160298


namespace NUMINAMATH_GPT_find_m_l1602_160271

open Real

-- Definitions based on problem conditions
def vector_a (m : ℝ) : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (3, -2)

-- The dot product
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
v₁.1 * v₂.1 + v₁.2 * v₂.2

-- Prove the final statement using given conditions
theorem find_m (m : ℝ) (h1 : dot_product (vector_a m) vector_b + dot_product vector_b vector_b = 0) :
  m = 8 :=
sorry

end NUMINAMATH_GPT_find_m_l1602_160271


namespace NUMINAMATH_GPT_max_distance_from_curve_to_line_l1602_160299

theorem max_distance_from_curve_to_line
  (θ : ℝ) (t : ℝ)
  (C_polar_eqn : ∀ θ, ∃ (ρ : ℝ), ρ = 2 * Real.cos θ)
  (line_eqn : ∀ t, ∃ (x y : ℝ), x = -1 + t ∧ y = 2 * t) :
  ∃ (max_dist : ℝ), max_dist = (4 * Real.sqrt 5 + 5) / 5 := sorry

end NUMINAMATH_GPT_max_distance_from_curve_to_line_l1602_160299


namespace NUMINAMATH_GPT_mark_cans_l1602_160214

theorem mark_cans (R J M : ℕ) (h1 : J = 2 * R + 5) (h2 : M = 4 * J) (h3 : R + J + M = 135) : M = 100 :=
by
  sorry

end NUMINAMATH_GPT_mark_cans_l1602_160214


namespace NUMINAMATH_GPT_arithmetic_sequence_max_sum_l1602_160286

noncomputable def max_S_n (n : ℕ) (a : ℕ → ℝ) (d : ℝ) : ℝ :=
  n * a 1 + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_max_sum :
  ∃ d, ∃ a : ℕ → ℝ, 
  (a 1 = 1) ∧ (3 * (a 1 + 7 * d) = 5 * (a 1 + 12 * d)) ∧ 
  (∀ n, max_S_n n a d ≤ max_S_n 20 a d) := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_max_sum_l1602_160286


namespace NUMINAMATH_GPT_theorem_perimeter_shaded_region_theorem_area_shaded_region_l1602_160212

noncomputable section

-- Definitions based on the conditions
def r : ℝ := Real.sqrt (1 / Real.pi)  -- radius of the unit circle

-- Define the perimeter and area functions for the shaded region
def perimeter_shaded_region (r : ℝ) : ℝ :=
  2 * Real.sqrt Real.pi

def area_shaded_region (r : ℝ) : ℝ :=
  1 / 5

-- Main theorem statements to prove
theorem theorem_perimeter_shaded_region
  (h : Real.pi * r^2 = 1) : perimeter_shaded_region r = 2 * Real.sqrt Real.pi :=
by
  sorry

theorem theorem_area_shaded_region
  (h : Real.pi * r^2 = 1) : area_shaded_region r = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_theorem_perimeter_shaded_region_theorem_area_shaded_region_l1602_160212


namespace NUMINAMATH_GPT_range_of_values_for_m_l1602_160227

theorem range_of_values_for_m (m : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - 4| < m) → m > 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_values_for_m_l1602_160227


namespace NUMINAMATH_GPT_smallest_square_perimeter_of_isosceles_triangle_with_composite_sides_l1602_160291

def is_composite (n : ℕ) : Prop := (∃ m k : ℕ, 1 < m ∧ 1 < k ∧ n = m * k)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_square_perimeter_of_isosceles_triangle_with_composite_sides :
  ∃ a b : ℕ,
    is_composite a ∧
    is_composite b ∧
    (2 * a + b) ^ 2 = 256 :=
sorry

end NUMINAMATH_GPT_smallest_square_perimeter_of_isosceles_triangle_with_composite_sides_l1602_160291


namespace NUMINAMATH_GPT_expressing_population_in_scientific_notation_l1602_160220

def population_in_scientific_notation (population : ℝ) : Prop :=
  population = 1.412 * 10^9

theorem expressing_population_in_scientific_notation : 
  population_in_scientific_notation (1.412 * 10^9) :=
by
  sorry

end NUMINAMATH_GPT_expressing_population_in_scientific_notation_l1602_160220


namespace NUMINAMATH_GPT_count_distinct_ways_l1602_160274

theorem count_distinct_ways (p : ℕ × ℕ → ℕ) (h_condition : ∃ j : ℕ × ℕ, j ∈ [(0, 0), (0, 1)] ∧ p j = 4)
  (h_grid_size : ∀ i : ℕ × ℕ, i ∈ [(0, 0), (0, 1), (1, 0), (1, 1)] → 1 ≤ p i ∧ p i ≤ 4)
  (h_distinct : ∀ i j : ℕ × ℕ, i ∈ [(0, 0), (0, 1), (1, 0), (1, 1)] → j ∈ [(0, 0), (0, 1), (1, 0), (1, 1)] → i ≠ j → p i ≠ p j) :
  ∃! l : Finset (ℕ × ℕ → ℕ), l.card = 12 :=
by
  sorry

end NUMINAMATH_GPT_count_distinct_ways_l1602_160274


namespace NUMINAMATH_GPT_real_seq_proof_l1602_160282

noncomputable def real_seq_ineq (a : ℕ → ℝ) : Prop :=
  ∀ k m : ℕ, k > 0 → m > 0 → |a (k + m) - a k - a m| ≤ 1

theorem real_seq_proof (a : ℕ → ℝ) (h : real_seq_ineq a) :
  ∀ k m : ℕ, k > 0 → m > 0 → |a k / k - a m / m| < 1 / k + 1 / m :=
by
  sorry

end NUMINAMATH_GPT_real_seq_proof_l1602_160282


namespace NUMINAMATH_GPT_original_price_dish_l1602_160224

-- Conditions
variables (P : ℝ) -- Original price of the dish
-- Discount and tips
def john_discounted_and_tip := 0.9 * P + 0.15 * P
def jane_discounted_and_tip := 0.9 * P + 0.135 * P

-- Condition of payment difference
def payment_difference := john_discounted_and_tip P = jane_discounted_and_tip P + 0.36

-- The theorem to prove
theorem original_price_dish : payment_difference P → P = 24 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_original_price_dish_l1602_160224


namespace NUMINAMATH_GPT_trapezoid_base_lengths_l1602_160228

noncomputable def trapezoid_bases (d h : Real) : Real × Real :=
  let b := h - 2 * d
  let B := h + 2 * d
  (b, B)

theorem trapezoid_base_lengths :
  ∀ (d : Real), d = Real.sqrt 3 →
  ∀ (h : Real), h = Real.sqrt 48 →
  ∃ (b B : Real), trapezoid_bases d h = (b, B) ∧ b = Real.sqrt 48 - 2 * Real.sqrt 3 ∧ B = Real.sqrt 48 + 2 * Real.sqrt 3 := by 
  sorry

end NUMINAMATH_GPT_trapezoid_base_lengths_l1602_160228


namespace NUMINAMATH_GPT_roots_of_equation_l1602_160226

theorem roots_of_equation (x : ℝ) : x * (x - 1) = 0 ↔ x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_GPT_roots_of_equation_l1602_160226


namespace NUMINAMATH_GPT_exists_five_numbers_l1602_160223

theorem exists_five_numbers :
  ∃ a1 a2 a3 a4 a5 : ℤ,
  a1 + a2 < 0 ∧
  a2 + a3 < 0 ∧
  a3 + a4 < 0 ∧
  a4 + a5 < 0 ∧
  a5 + a1 < 0 ∧
  a1 + a2 + a3 + a4 + a5 > 0 :=
by
  sorry

end NUMINAMATH_GPT_exists_five_numbers_l1602_160223


namespace NUMINAMATH_GPT_smallest_total_hot_dogs_l1602_160236

def packs_hot_dogs := 12
def packs_buns := 9
def packs_mustard := 18
def packs_ketchup := 24

theorem smallest_total_hot_dogs : Nat.lcm (Nat.lcm (Nat.lcm packs_hot_dogs packs_buns) packs_mustard) packs_ketchup = 72 := by
  sorry

end NUMINAMATH_GPT_smallest_total_hot_dogs_l1602_160236


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1602_160287

def P (m : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → (1/x + 4 * x + 6 * m) ≥ 0

def Q (m : ℝ) : Prop :=
  m ≥ -5

theorem sufficient_but_not_necessary_condition (m : ℝ) : 
  (P m → Q m) ∧ ¬(Q m → P m) := sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1602_160287


namespace NUMINAMATH_GPT_sequence_an_solution_l1602_160201

noncomputable def a_n (n : ℕ) : ℝ := (
  (1 / 2) * (2 + Real.sqrt 3)^n + 
  (1 / 2) * (2 - Real.sqrt 3)^n
)^2

theorem sequence_an_solution (n : ℕ) : 
  ∀ (a b : ℕ → ℝ),
  a 0 = 1 → 
  b 0 = 0 → 
  (∀ n, a (n + 1) = 7 * a n + 6 * b n - 3) → 
  (∀ n, b (n + 1) = 8 * a n + 7 * b n - 4) → 
  a n = a_n n := sorry

end NUMINAMATH_GPT_sequence_an_solution_l1602_160201


namespace NUMINAMATH_GPT_amaya_movie_watching_time_l1602_160234

theorem amaya_movie_watching_time :
  let uninterrupted_time_1 := 35
  let uninterrupted_time_2 := 45
  let uninterrupted_time_3 := 20
  let rewind_time_1 := 5
  let rewind_time_2 := 15
  let total_uninterrupted := uninterrupted_time_1 + uninterrupted_time_2 + uninterrupted_time_3
  let total_rewind := rewind_time_1 + rewind_time_2
  let total_time := total_uninterrupted + total_rewind
  total_time = 120 := by
  sorry

end NUMINAMATH_GPT_amaya_movie_watching_time_l1602_160234


namespace NUMINAMATH_GPT_initial_outlay_l1602_160211

-- Definition of given conditions
def manufacturing_cost (I : ℝ) (sets : ℕ) (cost_per_set : ℝ) : ℝ := I + sets * cost_per_set
def revenue (sets : ℕ) (price_per_set : ℝ) : ℝ := sets * price_per_set
def profit (revenue manufacturing_cost : ℝ) : ℝ := revenue - manufacturing_cost

-- Given data
def sets : ℕ := 500
def cost_per_set : ℝ := 20
def price_per_set : ℝ := 50
def given_profit : ℝ := 5000

-- The statement to prove
theorem initial_outlay (I : ℝ) : 
  profit (revenue sets price_per_set) (manufacturing_cost I sets cost_per_set) = given_profit → 
  I = 10000 := by
  sorry

end NUMINAMATH_GPT_initial_outlay_l1602_160211


namespace NUMINAMATH_GPT_savings_calculation_l1602_160208

theorem savings_calculation (income expenditure : ℝ) (h_ratio : income = 5 / 4 * expenditure) (h_income : income = 19000) :
  income - expenditure = 3800 := 
by
  -- The solution will be filled in here,
  -- showing the calculus automatically.
  sorry

end NUMINAMATH_GPT_savings_calculation_l1602_160208


namespace NUMINAMATH_GPT_cost_of_2000_pieces_of_gum_l1602_160210

theorem cost_of_2000_pieces_of_gum
  (cost_per_piece_in_cents : Nat)
  (pieces_of_gum : Nat)
  (conversion_rate_cents_to_dollars : Nat)
  (h1 : cost_per_piece_in_cents = 5)
  (h2 : pieces_of_gum = 2000)
  (h3 : conversion_rate_cents_to_dollars = 100) :
  (cost_per_piece_in_cents * pieces_of_gum) / conversion_rate_cents_to_dollars = 100 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_2000_pieces_of_gum_l1602_160210


namespace NUMINAMATH_GPT_log_equation_solution_l1602_160248

theorem log_equation_solution (x : ℝ) (hpos : x > 0) (hneq : x ≠ 1) : (Real.log 8 / Real.log x) * (2 * Real.log x / Real.log 2) = 6 * Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_log_equation_solution_l1602_160248


namespace NUMINAMATH_GPT_average_speed_interval_l1602_160233

theorem average_speed_interval {s t : ℝ → ℝ} (h_eq : ∀ t, s t = t^2 + 1) : 
  (s 2 - s 1) / (2 - 1) = 3 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_interval_l1602_160233


namespace NUMINAMATH_GPT_system_of_equations_solution_l1602_160276

theorem system_of_equations_solution :
  ∃ (x1 x2 x3 x4 x5 : ℝ),
  (x1 + 2 * x2 + 2 * x3 + 2 * x4 + 2 * x5 = 1) ∧
  (x1 + 3 * x2 + 4 * x3 + 4 * x4 + 4 * x5 = 2) ∧
  (x1 + 3 * x2 + 5 * x3 + 6 * x4 + 6 * x5 = 3) ∧
  (x1 + 3 * x2 + 5 * x3 + 7 * x4 + 8 * x5 = 4) ∧
  (x1 + 3 * x2 + 5 * x3 + 7 * x4 + 9 * x5 = 5) ∧
  (x1 = 1) ∧ (x2 = -1) ∧ (x3 = 1) ∧ (x4 = -1) ∧ (x5 = 1) := by
sorry

end NUMINAMATH_GPT_system_of_equations_solution_l1602_160276


namespace NUMINAMATH_GPT_number_of_spiders_l1602_160265

theorem number_of_spiders (total_legs : ℕ) (legs_per_spider : ℕ) 
  (h1 : total_legs = 40) (h2 : legs_per_spider = 8) : 
  (total_legs / legs_per_spider = 5) :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_number_of_spiders_l1602_160265


namespace NUMINAMATH_GPT_pet_store_has_70_birds_l1602_160253

-- Define the given conditions
def num_cages : ℕ := 7
def parrots_per_cage : ℕ := 4
def parakeets_per_cage : ℕ := 3
def cockatiels_per_cage : ℕ := 2
def canaries_per_cage : ℕ := 1

-- Total number of birds in one cage
def birds_per_cage : ℕ := parrots_per_cage + parakeets_per_cage + cockatiels_per_cage + canaries_per_cage

-- Total number of birds in all cages
def total_birds := birds_per_cage * num_cages

-- Prove that the total number of birds is 70
theorem pet_store_has_70_birds : total_birds = 70 :=
sorry

end NUMINAMATH_GPT_pet_store_has_70_birds_l1602_160253


namespace NUMINAMATH_GPT_unique_positive_integer_triples_l1602_160241

theorem unique_positive_integer_triples (a b c : ℕ) (h1 : ab + 3 * b * c = 63) (h2 : ac + 3 * b * c = 39) : 
∃! (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ ab + 3 * b * c = 63 ∧ ac + 3 * b * c = 39 :=
by sorry

end NUMINAMATH_GPT_unique_positive_integer_triples_l1602_160241


namespace NUMINAMATH_GPT_find_c_l1602_160258

noncomputable def f (x c : ℝ) := x * (x - c) ^ 2
noncomputable def f' (x c : ℝ) := 3 * x ^ 2 - 4 * c * x + c ^ 2
noncomputable def f'' (x c : ℝ) := 6 * x - 4 * c

theorem find_c (c : ℝ) : f' 2 c = 0 ∧ f'' 2 c < 0 → c = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_c_l1602_160258


namespace NUMINAMATH_GPT_value_of_C_l1602_160278

theorem value_of_C (C : ℝ) (h : 4 * C + 3 = 25) : C = 5.5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_C_l1602_160278


namespace NUMINAMATH_GPT_find_x_l1602_160239

noncomputable def angle_sum_triangle (A B C: ℝ) : Prop :=
  A + B + C = 180

noncomputable def vertical_angles_equal (A B: ℝ) : Prop :=
  A = B

noncomputable def right_angle_sum (D E: ℝ) : Prop :=
  D + E = 90

theorem find_x 
  (angle_ABC angle_BAC angle_DCE : ℝ) 
  (h1 : angle_ABC = 70)
  (h2 : angle_BAC = 50)
  (h3 : angle_sum_triangle angle_ABC angle_BAC angle_DCE)
  (h4 : vertical_angles_equal angle_DCE angle_DCE)
  (h5 : right_angle_sum angle_DCE 30) :
  angle_DCE = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1602_160239


namespace NUMINAMATH_GPT_expression_expansion_l1602_160221

noncomputable def expand_expression : Polynomial ℤ :=
 -2 * (5 * Polynomial.X^3 - 7 * Polynomial.X^2 + Polynomial.X - 4)

theorem expression_expansion :
  expand_expression = -10 * Polynomial.X^3 + 14 * Polynomial.X^2 - 2 * Polynomial.X + 8 :=
by
  sorry

end NUMINAMATH_GPT_expression_expansion_l1602_160221


namespace NUMINAMATH_GPT_anne_equals_bob_l1602_160242

-- Define the conditions as constants and functions
def original_price : ℝ := 120.00
def tax_rate : ℝ := 0.06
def discount_rate : ℝ := 0.25

-- Calculation models for Anne and Bob
def anne_total (price : ℝ) (tax : ℝ) (discount : ℝ) : ℝ :=
  (price * (1 + tax)) * (1 - discount)

def bob_total (price : ℝ) (tax : ℝ) (discount : ℝ) : ℝ :=
  (price * (1 - discount)) * (1 + tax)

-- The theorem that states what we need to prove
theorem anne_equals_bob : anne_total original_price tax_rate discount_rate = bob_total original_price tax_rate discount_rate :=
by
  sorry

end NUMINAMATH_GPT_anne_equals_bob_l1602_160242


namespace NUMINAMATH_GPT_point_on_y_axis_coordinates_l1602_160254

theorem point_on_y_axis_coordinates (m : ℤ) (P : ℤ × ℤ) (hP : P = (m - 1, m + 3)) (hY : P.1 = 0) : P = (0, 4) :=
sorry

end NUMINAMATH_GPT_point_on_y_axis_coordinates_l1602_160254


namespace NUMINAMATH_GPT_andrew_bought_6_kg_of_grapes_l1602_160272

def rate_grapes := 74
def rate_mangoes := 59
def kg_mangoes := 9
def total_paid := 975

noncomputable def number_of_kg_grapes := 6

theorem andrew_bought_6_kg_of_grapes :
  ∃ G : ℕ, (rate_grapes * G + rate_mangoes * kg_mangoes = total_paid) ∧ G = number_of_kg_grapes := 
by
  sorry

end NUMINAMATH_GPT_andrew_bought_6_kg_of_grapes_l1602_160272


namespace NUMINAMATH_GPT_find_original_number_l1602_160235

def is_valid_digit (d : ℕ) : Prop := d < 10

def original_number (a b c : ℕ) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧
  222 * (a + b + c) - 5 * (100 * a + 10 * b + c) = 3194

theorem find_original_number (a b c : ℕ) (h_valid: is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c)
  (h_sum : 222 * (a + b + c) - 5 * (100 * a + 10 * b + c) = 3194) : 
  100 * a + 10 * b + c = 358 := 
sorry

end NUMINAMATH_GPT_find_original_number_l1602_160235


namespace NUMINAMATH_GPT_minimum_number_of_guests_l1602_160252

def total_food : ℤ := 327
def max_food_per_guest : ℤ := 2

theorem minimum_number_of_guests :
  ∀ (n : ℤ), total_food ≤ n * max_food_per_guest → n = 164 :=
by
  sorry

end NUMINAMATH_GPT_minimum_number_of_guests_l1602_160252


namespace NUMINAMATH_GPT_largest_base_5_five_digit_number_in_decimal_l1602_160296

theorem largest_base_5_five_digit_number_in_decimal :
  (4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0) = 3124 :=
  sorry

end NUMINAMATH_GPT_largest_base_5_five_digit_number_in_decimal_l1602_160296


namespace NUMINAMATH_GPT_manuscript_total_cost_l1602_160273

theorem manuscript_total_cost
  (P R1 R2 R3 : ℕ)
  (RateFirst RateRevision : ℕ)
  (hP : P = 300)
  (hR1 : R1 = 55)
  (hR2 : R2 = 35)
  (hR3 : R3 = 25)
  (hRateFirst : RateFirst = 8)
  (hRateRevision : RateRevision = 6) :
  let RemainingPages := P - (R1 + R2 + R3)
  let CostNoRevisions := RemainingPages * RateFirst
  let CostOneRevision := R1 * (RateFirst + RateRevision)
  let CostTwoRevisions := R2 * (RateFirst + 2 * RateRevision)
  let CostThreeRevisions := R3 * (RateFirst + 3 * RateRevision)
  let TotalCost := CostNoRevisions + CostOneRevision + CostTwoRevisions + CostThreeRevisions
  TotalCost = 3600 :=
by
  sorry

end NUMINAMATH_GPT_manuscript_total_cost_l1602_160273


namespace NUMINAMATH_GPT_perpendicular_lines_slope_l1602_160292

theorem perpendicular_lines_slope (m : ℝ) : 
  ((m ≠ -3) ∧ (m ≠ -5) ∧ 
  (- (m + 3) / 4 * - (2 / (m + 5)) = -1)) ↔ m = -13 / 3 := 
sorry

end NUMINAMATH_GPT_perpendicular_lines_slope_l1602_160292


namespace NUMINAMATH_GPT_rate_of_interest_per_annum_l1602_160268

theorem rate_of_interest_per_annum (P R : ℝ) (T : ℝ) 
  (h1 : T = 8)
  (h2 : (P / 5) = (P * R * T) / 100) : 
  R = 2.5 := 
by
  sorry

end NUMINAMATH_GPT_rate_of_interest_per_annum_l1602_160268


namespace NUMINAMATH_GPT_smallest_odd_number_with_five_different_prime_factors_l1602_160238

noncomputable def smallest_odd_with_five_prime_factors : Nat :=
  3 * 5 * 7 * 11 * 13

theorem smallest_odd_number_with_five_different_prime_factors :
  smallest_odd_with_five_prime_factors = 15015 :=
by
  sorry -- Proof to be filled in later

end NUMINAMATH_GPT_smallest_odd_number_with_five_different_prime_factors_l1602_160238


namespace NUMINAMATH_GPT_amount_after_two_years_l1602_160290

def present_value : ℝ := 62000
def rate_of_increase : ℝ := 0.125
def time_period : ℕ := 2

theorem amount_after_two_years:
  let amount_after_n_years (pv : ℝ) (r : ℝ) (n : ℕ) := pv * (1 + r)^n
  amount_after_n_years present_value rate_of_increase time_period = 78468.75 := 
  by 
    -- This is where your proof would go
    sorry

end NUMINAMATH_GPT_amount_after_two_years_l1602_160290


namespace NUMINAMATH_GPT_benzoic_acid_molecular_weight_l1602_160215

-- Definitions for atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Molecular formula for Benzoic acid: C7H6O2
def benzoic_acid_formula : ℕ × ℕ × ℕ := (7, 6, 2)

-- Definition for the molecular weight calculation
def molecular_weight := λ (c h o : ℝ) (nC nH nO : ℕ) => 
  (nC * c) + (nH * h) + (nO * o)

-- Proof statement
theorem benzoic_acid_molecular_weight :
  molecular_weight atomic_weight_C atomic_weight_H atomic_weight_O 7 6 2 = 122.118 := by
  sorry

end NUMINAMATH_GPT_benzoic_acid_molecular_weight_l1602_160215


namespace NUMINAMATH_GPT_articles_correct_l1602_160204

-- Define the problem conditions
def refersToSpecific (word : String) : Prop :=
  word = "keyboard"

def refersToGeneral (word : String) : Prop :=
  word = "computer"

-- Define the articles
def the_article : String := "the"
def a_article : String := "a"

-- State the theorem for the corresponding solution
theorem articles_correct :
  refersToSpecific "keyboard" → refersToGeneral "computer" →  
  (the_article, a_article) = ("the", "a") :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_articles_correct_l1602_160204
