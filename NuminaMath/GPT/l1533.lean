import Mathlib

namespace steve_oranges_count_l1533_153328

variable (Brian_oranges Marcie_oranges Shawn_oranges Steve_oranges : ℝ)

def oranges_conditions : Prop :=
  (Marcie_oranges = 12) ∧
  (Brian_oranges = Marcie_oranges) ∧
  (Shawn_oranges = 1.075 * (Brian_oranges + Marcie_oranges)) ∧
  (Steve_oranges = 3 * (Marcie_oranges + Brian_oranges + Shawn_oranges))

theorem steve_oranges_count (h : oranges_conditions Brian_oranges Marcie_oranges Shawn_oranges Steve_oranges) :
  Steve_oranges = 149.4 :=
sorry

end steve_oranges_count_l1533_153328


namespace geometric_sequence_product_l1533_153309

theorem geometric_sequence_product (a b : ℝ) (h : 2 * b = a * 16) : a * b = 32 :=
sorry

end geometric_sequence_product_l1533_153309


namespace solve_for_x_l1533_153323

theorem solve_for_x (x : ℝ) (h : x ≠ 0) (h_eq : (8 * x) ^ 16 = (32 * x) ^ 8) : x = 1 / 2 :=
by
  sorry

end solve_for_x_l1533_153323


namespace combin_sum_l1533_153350

def combin (n m : ℕ) : ℕ := Nat.factorial n / (Nat.factorial m * Nat.factorial (n - m))

theorem combin_sum (n : ℕ) (h₁ : n = 99) : combin n 2 + combin n 3 = 161700 := by
  sorry

end combin_sum_l1533_153350


namespace find_t_from_integral_l1533_153393

theorem find_t_from_integral :
  (∫ x in (1 : ℝ)..t, (-1 / x + 2 * x)) = (3 - Real.log 2) → t = 2 :=
by
  sorry

end find_t_from_integral_l1533_153393


namespace ezekiel_shoes_l1533_153315

theorem ezekiel_shoes (pairs : ℕ) (shoes_per_pair : ℕ) (bought_pairs : pairs = 3) (pair_contains : shoes_per_pair = 2) : pairs * shoes_per_pair = 6 := by
  sorry

end ezekiel_shoes_l1533_153315


namespace find_number_l1533_153381

theorem find_number (x : ℝ) (h : ((x / 8) + 8 - 30) * 6 = 12) : x = 192 :=
sorry

end find_number_l1533_153381


namespace eval_inverse_l1533_153342

variable (g : ℕ → ℕ)
variable (g_inv : ℕ → ℕ)
variable (h₁ : g 4 = 6)
variable (h₂ : g 7 = 2)
variable (h₃ : g 3 = 7)
variable (h_inv₁ : g_inv 6 = 4)
variable (h_inv₂ : g_inv 7 = 3)

theorem eval_inverse (g : ℕ → ℕ)
(g_inv : ℕ → ℕ)
(h₁ : g 4 = 6)
(h₂ : g 7 = 2)
(h₃ : g 3 = 7)
(h_inv₁ : g_inv 6 = 4)
(h_inv₂ : g_inv 7 = 3) :
g_inv (g_inv 7 + g_inv 6) = 3 := by
  sorry

end eval_inverse_l1533_153342


namespace percent_of_y_l1533_153303

theorem percent_of_y (y : ℝ) : 0.30 * (0.80 * y) = 0.24 * y :=
by sorry

end percent_of_y_l1533_153303


namespace all_statements_correct_l1533_153314

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem all_statements_correct (b : ℝ) (h1 : b > 0) (h2 : b ≠ 1) :
  (f b b = 1) ∧
  (f b 1 = 0) ∧
  (¬(0 ∈ Set.range (f b))) ∧
  (∀ x, 0 < x ∧ x < b → f b x < 1) ∧
  (∀ x, x > b → f b x > 1) := by
  unfold f
  sorry

end all_statements_correct_l1533_153314


namespace matrix_diagonal_neg5_l1533_153320

variable (M : Matrix (Fin 3) (Fin 3) ℝ)

theorem matrix_diagonal_neg5 
    (h : ∀ v : Fin 3 → ℝ, (M.mulVec v) = -5 • v) : 
    M = !![-5, 0, 0; 0, -5, 0; 0, 0, -5] :=
by
  sorry

end matrix_diagonal_neg5_l1533_153320


namespace students_on_playground_l1533_153329

theorem students_on_playground (rows_left : ℕ) (rows_right : ℕ) (rows_front : ℕ) (rows_back : ℕ) (h1 : rows_left = 12) (h2 : rows_right = 11) (h3 : rows_front = 18) (h4 : rows_back = 8) :
    (rows_left + rows_right - 1) * (rows_front + rows_back - 1) = 550 := 
by
  sorry

end students_on_playground_l1533_153329


namespace largest_divisor_of_seven_consecutive_odd_numbers_l1533_153313

theorem largest_divisor_of_seven_consecutive_odd_numbers (n : ℕ) (h : Even n) (h_pos : n > 0) :
  ∃ d, d = 45 ∧ ∀ k, k ∣ ((n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13)) → k ≤ 45 :=
sorry

end largest_divisor_of_seven_consecutive_odd_numbers_l1533_153313


namespace brianna_initial_marbles_l1533_153331

-- Defining the variables and constants
def initial_marbles : Nat := 24
def marbles_lost : Nat := 4
def marbles_given : Nat := 2 * marbles_lost
def marbles_ate : Nat := marbles_lost / 2
def marbles_remaining : Nat := 10

-- The main statement to prove
theorem brianna_initial_marbles :
  marbles_remaining + marbles_ate + marbles_given + marbles_lost = initial_marbles :=
by
  sorry

end brianna_initial_marbles_l1533_153331


namespace correct_statement_b_l1533_153399

open Set 

variables {Point Line Plane : Type}
variable (m n : Line)
variable (α : Plane)
variable (perpendicular_to_plane : Line → Plane → Prop) 
variable (parallel_to_plane : Line → Plane → Prop)
variable (is_subline_of_plane : Line → Plane → Prop)
variable (perpendicular_to_line : Line → Line → Prop)

theorem correct_statement_b (hm : perpendicular_to_plane m α) (hn : is_subline_of_plane n α) : perpendicular_to_line m n :=
sorry

end correct_statement_b_l1533_153399


namespace modular_expression_problem_l1533_153385

theorem modular_expression_problem
  (m : ℕ)
  (hm : 0 ≤ m ∧ m < 29)
  (hmod : 4 * m % 29 = 1) :
  (5^m % 29)^4 - 3 % 29 = 13 % 29 :=
by
  sorry

end modular_expression_problem_l1533_153385


namespace ernie_income_ratio_l1533_153367

-- Define constants and properties based on the conditions
def previous_income := 6000
def jack_income := 2 * previous_income
def combined_income := 16800

-- Lean proof statement that the ratio of Ernie's current income to his previous income is 2/3
theorem ernie_income_ratio (current_income : ℕ) (h1 : current_income + jack_income = combined_income) :
    current_income / previous_income = 2 / 3 :=
sorry

end ernie_income_ratio_l1533_153367


namespace x_intercept_of_line_l1533_153362

theorem x_intercept_of_line : ∃ x : ℝ, ∃ y : ℝ, 4 * x + 7 * y = 28 ∧ y = 0 ∧ x = 7 :=
by
  sorry

end x_intercept_of_line_l1533_153362


namespace marble_catch_up_time_l1533_153337

theorem marble_catch_up_time 
    (a b c : ℝ) 
    (L : ℝ)
    (h1 : a - b = L / 50)
    (h2 : a - c = L / 40) 
    : (110 * (c - b)) / (c - b) = 110 := 
by 
    sorry

end marble_catch_up_time_l1533_153337


namespace intersection_with_xz_plane_l1533_153345

-- Initial points on the line
structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def point1 : Point3D := ⟨2, -1, 3⟩
def point2 : Point3D := ⟨6, -4, 7⟩

-- Definition of the line parametrization
def param_line (t : ℝ) : Point3D :=
  ⟨ point1.x + t * (point2.x - point1.x)
  , point1.y + t * (point2.y - point1.y)
  , point1.z + t * (point2.z - point1.z) ⟩

-- Prove that the line intersects the xz-plane at the expected point
theorem intersection_with_xz_plane :
  ∃ t : ℝ, param_line t = ⟨ 2/3, 0, 5/3 ⟩ :=
sorry

end intersection_with_xz_plane_l1533_153345


namespace perfect_square_of_expression_l1533_153322

theorem perfect_square_of_expression (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) : 
  ∃ m : ℕ, (2 * l - n - k) * (2 * l - n + k) / 2 = m^2 := 
by 
  sorry

end perfect_square_of_expression_l1533_153322


namespace intersection_point_l1533_153390

def satisfies_first_line (p : ℝ × ℝ) : Prop :=
  8 * p.1 - 5 * p.2 = 40

def satisfies_second_line (p : ℝ × ℝ) : Prop :=
  6 * p.1 + 2 * p.2 = 14

theorem intersection_point :
  satisfies_first_line (75 / 23, -64 / 23) ∧ satisfies_second_line (75 / 23, -64 / 23) :=
by 
  sorry

end intersection_point_l1533_153390


namespace point_in_fourth_quadrant_l1533_153378

/-- A point in a Cartesian coordinate system -/
structure Point (α : Type) :=
(x : α)
(y : α)

/-- Given a point (4, -3) in the Cartesian plane, prove it lies in the fourth quadrant -/
theorem point_in_fourth_quadrant (P : Point Int) (hx : P.x = 4) (hy : P.y = -3) : 
  P.x > 0 ∧ P.y < 0 :=
by
  sorry

end point_in_fourth_quadrant_l1533_153378


namespace calculate_diff_of_squares_l1533_153391

noncomputable def diff_of_squares (a b : ℕ) : ℕ :=
  a^2 - b^2

theorem calculate_diff_of_squares :
  diff_of_squares 601 597 = 4792 :=
by
  sorry

end calculate_diff_of_squares_l1533_153391


namespace total_cost_correct_l1533_153370

def sandwich_cost : ℝ := 2.44
def soda_cost : ℝ := 0.87
def num_sandwiches : ℕ := 2
def num_sodas : ℕ := 4

theorem total_cost_correct :
  (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost) = 8.36 := by
  sorry

end total_cost_correct_l1533_153370


namespace sum_of_roots_l1533_153335

theorem sum_of_roots (g : ℝ → ℝ) 
  (h_symmetry : ∀ x : ℝ, g (3 + x) = g (3 - x))
  (h_roots : ∃ s1 s2 s3 s4 : ℝ, 
               g s1 = 0 ∧ 
               g s2 = 0 ∧ 
               g s3 = 0 ∧ 
               g s4 = 0 ∧ 
               s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ 
               s2 ≠ s3 ∧ s2 ≠ s4 ∧ s3 ≠ s4) :
  s1 + s2 + s3 + s4 = 12 :=
by 
  sorry

end sum_of_roots_l1533_153335


namespace max_of_three_diff_pos_int_with_mean_7_l1533_153334

theorem max_of_three_diff_pos_int_with_mean_7 (a b c : ℕ) (h_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_mean : (a + b + c) / 3 = 7) :
  max a (max b c) = 18 := 
sorry

end max_of_three_diff_pos_int_with_mean_7_l1533_153334


namespace school_year_hours_per_week_l1533_153366

-- Definitions based on the conditions of the problem
def summer_weeks : ℕ := 8
def summer_hours_per_week : ℕ := 40
def summer_earnings : ℕ := 3200

def school_year_weeks : ℕ := 24
def needed_school_year_earnings : ℕ := 6400

-- Question translated to a Lean statement
theorem school_year_hours_per_week :
  let hourly_rate := summer_earnings / (summer_hours_per_week * summer_weeks)
  let total_school_year_hours := needed_school_year_earnings / hourly_rate
  total_school_year_hours / school_year_weeks = (80 / 3) :=
by {
  -- The implementation of the proof goes here
  sorry
}

end school_year_hours_per_week_l1533_153366


namespace simplify_expr_l1533_153394

noncomputable def expr : ℝ := (18 * 10^10) / (6 * 10^4) * 2

theorem simplify_expr : expr = 6 * 10^6 := sorry

end simplify_expr_l1533_153394


namespace repeated_number_divisibility_l1533_153306

theorem repeated_number_divisibility (x : ℕ) (h : 1000 ≤ x ∧ x < 10000) :
  73 ∣ (10001 * x) ∧ 137 ∣ (10001 * x) :=
sorry

end repeated_number_divisibility_l1533_153306


namespace max_value_inequality_l1533_153333

theorem max_value_inequality : 
  ∀ (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (n : ℕ) (m : ℝ),
  (∀ n, S_n n = (n * a_n 1 + (1 / 2) * n * (n - 1) * d) ∧
  (∀ n, a_n n ^ 2 + (S_n n ^ 2 / n ^ 2) >= m * (a_n 1) ^ 2)) → 
  m ≤ 1 / 5 := 
sorry

end max_value_inequality_l1533_153333


namespace calculate_expression_l1533_153305

theorem calculate_expression : ((-1 + 2) * 3 + 2^2 / (-4)) = 2 :=
by
  sorry

end calculate_expression_l1533_153305


namespace gold_copper_alloy_ratio_l1533_153380

theorem gold_copper_alloy_ratio 
  (G C : ℝ) 
  (h_gold : G / weight_of_water = 19) 
  (h_copper : C / weight_of_water = 9)
  (weight_of_alloy : (G + C) / weight_of_water = 17) :
  G / C = 4 :=
sorry

end gold_copper_alloy_ratio_l1533_153380


namespace cost_of_traveling_roads_l1533_153374

def lawn_length : ℕ := 80
def lawn_breadth : ℕ := 40
def road_width : ℕ := 10
def cost_per_sqm : ℕ := 3

def area_road_parallel_length : ℕ := road_width * lawn_length
def area_road_parallel_breadth : ℕ := road_width * lawn_breadth
def area_intersection : ℕ := road_width * road_width

def total_area_roads : ℕ := area_road_parallel_length + area_road_parallel_breadth - area_intersection
def total_cost : ℕ := total_area_roads * cost_per_sqm

theorem cost_of_traveling_roads : total_cost = 3300 :=
by
  sorry

end cost_of_traveling_roads_l1533_153374


namespace eval_power_expression_l1533_153364

theorem eval_power_expression : (3^3)^2 / 3^2 = 81 := by
  sorry -- Proof omitted as instructed

end eval_power_expression_l1533_153364


namespace triangle_shape_statements_l1533_153375

theorem triangle_shape_statements (a b c : ℝ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (h : a^2 + b^2 + c^2 = ab + bc + ca) :
  (a = b ∧ b = c ∧ a = c) :=
by
  sorry 

end triangle_shape_statements_l1533_153375


namespace p_plus_q_l1533_153377

-- Define the problem conditions
def p (x : ℝ) : ℝ := 4 * (x - 2)
def q (x : ℝ) : ℝ := (x + 2) * (x - 2)

-- Main theorem to prove the answer
theorem p_plus_q (x : ℝ) : p x + q x = x^2 + 4 * x - 12 := 
by
  sorry

end p_plus_q_l1533_153377


namespace coffee_shop_brewed_cups_in_week_l1533_153339

theorem coffee_shop_brewed_cups_in_week 
    (weekday_rate : ℕ) (weekend_rate : ℕ)
    (weekday_hours : ℕ) (saturday_hours : ℕ) (sunday_hours : ℕ)
    (num_weekdays : ℕ) (num_saturdays : ℕ) (num_sundays : ℕ)
    (h1 : weekday_rate = 10)
    (h2 : weekend_rate = 15)
    (h3 : weekday_hours = 5)
    (h4 : saturday_hours = 6)
    (h5 : sunday_hours = 4)
    (h6 : num_weekdays = 5)
    (h7 : num_saturdays = 1)
    (h8 : num_sundays = 1) :
    (weekday_rate * weekday_hours * num_weekdays) + 
    (weekend_rate * saturday_hours * num_saturdays) + 
    (weekend_rate * sunday_hours * num_sundays) = 400 := 
by
  sorry

end coffee_shop_brewed_cups_in_week_l1533_153339


namespace janice_overtime_shifts_l1533_153347

theorem janice_overtime_shifts (x : ℕ) (h1 : 5 * 30 + 15 * x = 195) : x = 3 :=
by
  -- leaving the proof unfinished, as asked
  sorry

end janice_overtime_shifts_l1533_153347


namespace train_lengths_l1533_153373

noncomputable def train_problem : Prop :=
  let speed_T1_mps := 54 * (5/18)
  let speed_T2_mps := 72 * (5/18)
  let L_T1 := speed_T1_mps * 20
  let L_p := (speed_T1_mps * 44) - L_T1
  let L_T2 := speed_T2_mps * 16
  (L_p = 360) ∧ (L_T1 = 300) ∧ (L_T2 = 320)

theorem train_lengths : train_problem := sorry

end train_lengths_l1533_153373


namespace fraction_eaten_on_third_day_l1533_153312

theorem fraction_eaten_on_third_day
  (total_pieces : ℕ)
  (first_day_fraction : ℚ)
  (second_day_fraction : ℚ)
  (remaining_after_third_day : ℕ)
  (initial_pieces : total_pieces = 200)
  (first_day_eaten : first_day_fraction = 1/4)
  (second_day_eaten : second_day_fraction = 2/5)
  (remaining_bread_after_third_day : remaining_after_third_day = 45) :
  (1 : ℚ) / 2 = 1/2 := sorry

end fraction_eaten_on_third_day_l1533_153312


namespace time_to_send_data_in_minutes_l1533_153317

def blocks := 100
def chunks_per_block := 256
def transmission_rate := 100 -- chunks per second
def seconds_per_minute := 60

theorem time_to_send_data_in_minutes :
    (blocks * chunks_per_block) / transmission_rate / seconds_per_minute = 4 := by
  sorry

end time_to_send_data_in_minutes_l1533_153317


namespace handshake_problem_l1533_153357

theorem handshake_problem :
  let team_size := 6
  let teams := 2
  let referees := 3
  let handshakes_between_teams := team_size * team_size
  let handshakes_within_teams := teams * (team_size * (team_size - 1)) / 2
  let handshakes_with_referees := (teams * team_size) * referees
  handshakes_between_teams + handshakes_within_teams + handshakes_with_referees = 102 := by
  sorry

end handshake_problem_l1533_153357


namespace find_product_l1533_153387

def a : ℕ := 4
def g : ℕ := 8
def d : ℕ := 10

theorem find_product (A B C D E F : ℕ) (hA : A % 2 = 0) (hB : B % 3 = 0) (hC : C % 4 = 0) 
  (hD : D % 5 = 0) (hE : E % 6 = 0) (hF : F % 7 = 0) :
  a * g * d = 320 :=
by
  sorry

end find_product_l1533_153387


namespace hotel_charge_l1533_153304

variable (R G P : ℝ)

theorem hotel_charge (h1 : P = 0.60 * R) (h2 : P = 0.90 * G) : (R - G) / G = 0.50 :=
by
  sorry

end hotel_charge_l1533_153304


namespace greatest_two_digit_multiple_of_17_l1533_153327

theorem greatest_two_digit_multiple_of_17 : ∃ m : ℕ, (10 ≤ m) ∧ (m ≤ 99) ∧ (17 ∣ m) ∧ (∀ n : ℕ, (10 ≤ n) ∧ (n ≤ 99) ∧ (17 ∣ n) → n ≤ m) ∧ m = 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l1533_153327


namespace range_of_m_l1533_153302

-- Defining the quadratic function with the given condition
def quadratic (m : ℝ) (x : ℝ) : ℝ := (m-1)*x^2 + (m-1)*x + 2

-- Stating the problem
theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, quadratic m x > 0) ↔ 1 ≤ m ∧ m < 9 :=
by
  sorry

end range_of_m_l1533_153302


namespace inequality_proof_l1533_153340

theorem inequality_proof (x a : ℝ) (h1 : x > a) (h2 : a > 0) : x^2 > ax ∧ ax > a^2 :=
by
  sorry

end inequality_proof_l1533_153340


namespace find_values_of_a_and_b_l1533_153365

-- Define the problem
theorem find_values_of_a_and_b (a b : ℚ) (h1 : a + (a / 4) = 3) (h2 : b - 2 * a = 1) :
  a = 12 / 5 ∧ b = 29 / 5 := by
  sorry

end find_values_of_a_and_b_l1533_153365


namespace ab2_plus_bc2_plus_ca2_le_27_div_8_l1533_153352

theorem ab2_plus_bc2_plus_ca2_le_27_div_8 (a b c : ℝ) 
  (h1 : a ≥ b) 
  (h2 : b ≥ c) 
  (h3 : c > 0) 
  (h4 : a + b + c = 3) : 
  ab^2 + bc^2 + ca^2 ≤ 27 / 8 :=
sorry

end ab2_plus_bc2_plus_ca2_le_27_div_8_l1533_153352


namespace zero_function_l1533_153397

variable (f : ℝ × ℝ × ℝ → ℝ)

theorem zero_function (h : ∀ x y z : ℝ, f (x, y, z) = 2 * f (z, x, y)) : ∀ x y z : ℝ, f (x, y, z) = 0 :=
by
  intros
  sorry

end zero_function_l1533_153397


namespace kyler_wins_one_game_l1533_153353

theorem kyler_wins_one_game :
  ∃ (Kyler_wins : ℕ),
    (Kyler_wins + 3 + 2 + 2 = 6 ∧
    Kyler_wins + 3 = 6 ∧
    Kyler_wins = 1) := by
  sorry

end kyler_wins_one_game_l1533_153353


namespace base_7_multiplication_addition_l1533_153398

theorem base_7_multiplication_addition :
  (25 * 3 + 144) % 7^3 = 303 :=
by sorry

end base_7_multiplication_addition_l1533_153398


namespace sqrt_2_plus_x_nonnegative_l1533_153330

theorem sqrt_2_plus_x_nonnegative (x : ℝ) : (2 + x ≥ 0) → (x ≥ -2) :=
by
  sorry

end sqrt_2_plus_x_nonnegative_l1533_153330


namespace intersection_M_N_l1533_153383

def M : Set ℝ := { x : ℝ | (2 - x) / (x + 1) ≥ 0 }
def N : Set ℝ := Set.univ

theorem intersection_M_N : M ∩ N = { x : ℝ | -1 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_M_N_l1533_153383


namespace total_students_left_l1533_153368

def initial_boys : Nat := 14
def initial_girls : Nat := 10
def boys_dropout : Nat := 4
def girls_dropout : Nat := 3

def boys_left : Nat := initial_boys - boys_dropout
def girls_left : Nat := initial_girls - girls_dropout

theorem total_students_left : boys_left + girls_left = 17 :=
by 
  sorry

end total_students_left_l1533_153368


namespace find_a_l1533_153396

-- Definition of * in terms of 2a - b^2
def custom_mul (a b : ℤ) := 2 * a - b^2

-- The proof statement
theorem find_a (a : ℤ) : custom_mul a 3 = 3 → a = 6 :=
by
  sorry

end find_a_l1533_153396


namespace find_sum_l1533_153343

theorem find_sum (I r1 r2 r3 r4 r5: ℝ) (t1 t2 t3 t4 t5 : ℝ) (P: ℝ) 
  (hI: I = 6016.75)
  (hr1: r1 = 0.06) (hr2: r2 = 0.075) (hr3: r3 = 0.08) (hr4: r4 = 0.085) (hr5: r5 = 0.09)
  (ht: ∀ i, (i = t1 ∨ i = t2 ∨ i = t3 ∨ i = t4 ∨ i = t5) → i = 1): 
  I = P * (r1 * t1 + r2 * t2 + r3 * t3 + r4 * t4 + r5 * t5) → P = 15430 :=
by
  sorry

end find_sum_l1533_153343


namespace track_length_l1533_153388

theorem track_length (x : ℝ) (hb hs : ℝ) (h_opposite : hs = x / 2 - 120) (h_first_meet : hb = 120) (h_second_meet : hs + 180 = x / 2 + 60) : x = 600 := 
by
  sorry

end track_length_l1533_153388


namespace original_number_is_seven_l1533_153325

theorem original_number_is_seven (x : ℕ) (h : 3 * x - 5 = 16) : x = 7 := by
sorry

end original_number_is_seven_l1533_153325


namespace total_players_is_139_l1533_153354

def num_kabadi := 60
def num_kho_kho := 90
def num_soccer := 40
def num_basketball := 70
def num_volleyball := 50
def num_badminton := 30

def num_k_kh := 25
def num_k_s := 15
def num_k_b := 13
def num_k_v := 20
def num_k_ba := 10
def num_kh_s := 35
def num_kh_b := 16
def num_kh_v := 30
def num_kh_ba := 12
def num_s_b := 20
def num_s_v := 18
def num_s_ba := 7
def num_b_v := 15
def num_b_ba := 8
def num_v_ba := 10

def num_k_kh_s := 5
def num_k_b_v := 4
def num_s_b_ba := 3
def num_v_ba_kh := 2

def num_all_sports := 1

noncomputable def total_players : Nat :=
  (num_kabadi + num_kho_kho + num_soccer + num_basketball + num_volleyball + num_badminton) 
  - (num_k_kh + num_k_s + num_k_b + num_k_v + num_k_ba + num_kh_s + num_kh_b + num_kh_v + num_kh_ba + num_s_b + num_s_v + num_s_ba + num_b_v + num_b_ba + num_v_ba)
  + (num_k_kh_s + num_k_b_v + num_s_b_ba + num_v_ba_kh)
  - num_all_sports

theorem total_players_is_139 : total_players = 139 := 
  by 
    sorry

end total_players_is_139_l1533_153354


namespace event_day_is_Sunday_l1533_153389

def days_in_week := 7

def event_day := 1500

def start_day := "Friday"

def day_of_week_according_to_mod : Nat → String 
| 0 => "Friday"
| 1 => "Saturday"
| 2 => "Sunday"
| 3 => "Monday"
| 4 => "Tuesday"
| 5 => "Wednesday"
| 6 => "Thursday"
| _ => "Invalid"

theorem event_day_is_Sunday : day_of_week_according_to_mod (event_day % days_in_week) = "Sunday" :=
sorry

end event_day_is_Sunday_l1533_153389


namespace number_of_girls_more_than_boys_l1533_153300

theorem number_of_girls_more_than_boys
    (total_students : ℕ)
    (number_of_boys : ℕ)
    (h1 : total_students = 485)
    (h2 : number_of_boys = 208) :
    total_students - number_of_boys - number_of_boys = 69 :=
by
    sorry

end number_of_girls_more_than_boys_l1533_153300


namespace batsman_average_after_17th_inning_l1533_153341

theorem batsman_average_after_17th_inning
  (A : ℝ) -- average before 17th inning
  (h1 : (16 * A + 50) / 17 = A + 2) : 
  (A + 2) = 18 :=
by
  -- Proof goes here
  sorry

end batsman_average_after_17th_inning_l1533_153341


namespace concert_parking_fee_l1533_153336

theorem concert_parking_fee :
  let ticket_cost := 50 
  let processing_fee_percentage := 0.15 
  let entrance_fee_per_person := 5 
  let total_cost_concert := 135
  let num_people := 2 

  let total_ticket_cost := ticket_cost * num_people
  let processing_fee := total_ticket_cost * processing_fee_percentage
  let total_ticktet_cost_with_fee := total_ticket_cost + processing_fee
  let total_entrance_fee := entrance_fee_per_person * num_people
  let total_cost_without_parking := total_ticktet_cost_with_fee + total_entrance_fee
  total_cost_concert - total_cost_without_parking = 10 := by 
  sorry

end concert_parking_fee_l1533_153336


namespace bianca_ate_candy_l1533_153358

theorem bianca_ate_candy (original_candies : ℕ) (pieces_per_pile : ℕ) 
                         (number_of_piles : ℕ) 
                         (remaining_candies : ℕ) 
                         (h_original : original_candies = 78) 
                         (h_pieces_per_pile : pieces_per_pile = 8) 
                         (h_number_of_piles : number_of_piles = 6) 
                         (h_remaining : remaining_candies = pieces_per_pile * number_of_piles) :
  original_candies - remaining_candies = 30 := by
  subst_vars
  sorry

end bianca_ate_candy_l1533_153358


namespace clyde_picked_bushels_l1533_153363

theorem clyde_picked_bushels (weight_per_bushel : ℕ) (weight_per_cob : ℕ) (cobs_picked : ℕ) :
  weight_per_bushel = 56 →
  weight_per_cob = 1 / 2 →
  cobs_picked = 224 →
  cobs_picked * weight_per_cob / weight_per_bushel = 2 :=
by
  intros
  sorry

end clyde_picked_bushels_l1533_153363


namespace circle_center_and_sum_l1533_153319

/-- Given the equation of a circle x^2 + y^2 - 6x + 14y = -28,
    prove that the coordinates (h, k) of the center of the circle are (3, -7)
    and compute h + k. -/
theorem circle_center_and_sum (x y : ℝ) :
  (∃ h k, (x^2 + y^2 - 6*x + 14*y = -28) ∧ (h = 3) ∧ (k = -7) ∧ (h + k = -4)) :=
by {
  sorry
}

end circle_center_and_sum_l1533_153319


namespace natasha_average_speed_l1533_153372

theorem natasha_average_speed
  (time_up time_down : ℝ)
  (speed_up distance_up total_distance total_time average_speed : ℝ)
  (h1 : time_up = 4)
  (h2 : time_down = 2)
  (h3 : speed_up = 3)
  (h4 : distance_up = speed_up * time_up)
  (h5 : total_distance = distance_up + distance_up)
  (h6 : total_time = time_up + time_down)
  (h7 : average_speed = total_distance / total_time) :
  average_speed = 4 := by
  sorry

end natasha_average_speed_l1533_153372


namespace cos_of_pi_over_3_minus_alpha_l1533_153360

theorem cos_of_pi_over_3_minus_alpha (α : Real) (h : Real.sin (Real.pi / 6 + α) = 2 / 3) :
  Real.cos (Real.pi / 3 - α) = 2 / 3 :=
by
  sorry

end cos_of_pi_over_3_minus_alpha_l1533_153360


namespace inverse_proportion_relationship_l1533_153346

theorem inverse_proportion_relationship (k : ℝ) (y1 y2 y3 : ℝ) :
  y1 = (k^2 + 1) / -1 →
  y2 = (k^2 + 1) / 1 →
  y3 = (k^2 + 1) / 2 →
  y1 < y3 ∧ y3 < y2 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end inverse_proportion_relationship_l1533_153346


namespace find_k_l1533_153395

noncomputable def line1 (t : ℝ) (k : ℝ) : ℝ × ℝ := (1 - 2 * t, 2 + k * t)
noncomputable def line2 (s : ℝ) : ℝ × ℝ := (s, 1 - 2 * s)

def correct_k (k : ℝ) : Prop :=
  let slope1 := -k / 2
  let slope2 := -2
  slope1 * slope2 = -1

theorem find_k (k : ℝ) (h_perpendicular : correct_k k) : k = -1 :=
sorry

end find_k_l1533_153395


namespace Gretchen_weekend_profit_l1533_153308

theorem Gretchen_weekend_profit :
  let saturday_revenue := 24 * 25
  let sunday_revenue := 16 * 15
  let total_revenue := saturday_revenue + sunday_revenue
  let park_fee := 5 * 6 * 2
  let art_supplies_cost := 8 * 2
  let total_expenses := park_fee + art_supplies_cost
  let profit := total_revenue - total_expenses
  profit = 764 :=
by
  sorry

end Gretchen_weekend_profit_l1533_153308


namespace find_angle_beta_l1533_153384

open Real

theorem find_angle_beta
  (α β : ℝ)
  (h1 : sin α = (sqrt 5) / 5)
  (h2 : sin (α - β) = - (sqrt 10) / 10)
  (hα_range : 0 < α ∧ α < π / 2)
  (hβ_range : 0 < β ∧ β < π / 2) :
  β = π / 4 :=
sorry

end find_angle_beta_l1533_153384


namespace find_unknown_number_l1533_153326

theorem find_unknown_number (x : ℝ) : 
  (1000 * 7) / (x * 17) = 10000 → x = 24.285714285714286 := by
  sorry

end find_unknown_number_l1533_153326


namespace smallest_k_for_ten_ruble_heads_up_l1533_153307

-- Conditions
def num_total_coins : ℕ := 30
def num_ten_ruble_coins : ℕ := 23
def num_five_ruble_coins : ℕ := 7
def num_heads_up : ℕ := 20
def num_tails_up : ℕ := 10

-- Prove the smallest k such that any k coins chosen include at least one ten-ruble coin heads-up.
theorem smallest_k_for_ten_ruble_heads_up (k : ℕ) :
  (∀ (coins : Finset ℕ), coins.card = k → (∃ (coin : ℕ) (h : coin ∈ coins), coin < num_ten_ruble_coins ∧ coin < num_heads_up)) →
  k = 18 :=
sorry

end smallest_k_for_ten_ruble_heads_up_l1533_153307


namespace proof_problem_l1533_153321

def x := 3
def y := 4

theorem proof_problem : 3 * x - 2 * y = 1 := by
  -- We will rely on these definitions and properties of arithmetic to show the result.
  -- The necessary proof steps would follow here, but are skipped for now.
  sorry

end proof_problem_l1533_153321


namespace length_of_marquita_garden_l1533_153310

variable (length_marquita_garden : ℕ)

def total_area_mancino_gardens : ℕ := 3 * (16 * 5)
def total_gardens_area : ℕ := 304
def total_area_marquita_gardens : ℕ := total_gardens_area - total_area_mancino_gardens
def area_one_marquita_garden : ℕ := total_area_marquita_gardens / 2

theorem length_of_marquita_garden :
  (4 * length_marquita_garden = area_one_marquita_garden) →
  length_marquita_garden = 8 := by
  sorry

end length_of_marquita_garden_l1533_153310


namespace overall_average_speed_l1533_153379

-- Define the conditions for Mark's travel
def time_cycling : ℝ := 1
def speed_cycling : ℝ := 20
def time_walking : ℝ := 2
def speed_walking : ℝ := 3

-- Define the total distance and total time
def total_distance : ℝ :=
  (time_cycling * speed_cycling) + (time_walking * speed_walking)

def total_time : ℝ :=
  time_cycling + time_walking

-- Define the proved statement for the average speed
theorem overall_average_speed : total_distance / total_time = 8.67 :=
by
  sorry

end overall_average_speed_l1533_153379


namespace original_flour_quantity_l1533_153324

-- Definitions based on conditions
def flour_called (x : ℝ) : Prop := 
  -- total flour Mary uses is x + extra 2 cups, which equals to 9 cups.
  x + 2 = 9

-- The proof statement we need to show
theorem original_flour_quantity : ∃ x : ℝ, flour_called x ∧ x = 7 := 
  sorry

end original_flour_quantity_l1533_153324


namespace alice_total_distance_correct_l1533_153316

noncomputable def alice_daily_morning_distance : ℕ := 10

noncomputable def alice_daily_afternoon_distance : ℕ := 12

noncomputable def alice_daily_distance : ℕ :=
  alice_daily_morning_distance + alice_daily_afternoon_distance

noncomputable def alice_weekly_distance : ℕ :=
  5 * alice_daily_distance

theorem alice_total_distance_correct :
  alice_weekly_distance = 110 :=
by
  unfold alice_weekly_distance alice_daily_distance alice_daily_morning_distance alice_daily_afternoon_distance
  norm_num

end alice_total_distance_correct_l1533_153316


namespace circle_equation_condition_l1533_153351

theorem circle_equation_condition (m : ℝ) : 
  (∃ h k r : ℝ, (r > 0) ∧ ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r^2 → x^2 + y^2 - 2*x - 4*y + m = 0) ↔ m < 5 :=
sorry

end circle_equation_condition_l1533_153351


namespace segment_length_after_reflection_l1533_153371

structure Point :=
(x : ℝ)
(y : ℝ)

def reflect_over_x_axis (p : Point) : Point :=
{ x := p.x, y := -p.y }

def distance (p1 p2 : Point) : ℝ :=
abs (p1.y - p2.y)

theorem segment_length_after_reflection :
  let C : Point := {x := -3, y := 2}
  let C' : Point := reflect_over_x_axis C
  distance C C' = 4 :=
by
  sorry

end segment_length_after_reflection_l1533_153371


namespace monotone_decreasing_interval_3_l1533_153332

variable {f : ℝ → ℝ}

theorem monotone_decreasing_interval_3 
  (h1 : ∀ x, f (x + 3) = f (x - 3))
  (h2 : ∀ x, f (x + 3) = f (-x + 3))
  (h3 : ∀ ⦃x y⦄, 0 < x → x < 3 → 0 < y → y < 3 → x < y → f y < f x) :
  f 3.5 < f (-4.5) ∧ f (-4.5) < f 12.5 :=
sorry

end monotone_decreasing_interval_3_l1533_153332


namespace sum_of_three_numbers_l1533_153311

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 241)
  (h2 : a * b + b * c + c * a = 100) : 
  a + b + c = 21 := 
by
  sorry

end sum_of_three_numbers_l1533_153311


namespace ellen_painted_roses_l1533_153344

theorem ellen_painted_roses :
  ∀ (r : ℕ),
    (5 * 17 + 7 * r + 3 * 6 + 2 * 20 = 213) → (r = 10) :=
by
  intros r h
  sorry

end ellen_painted_roses_l1533_153344


namespace expand_product_l1533_153349

theorem expand_product (x : ℝ) : (3 * x + 4) * (2 * x + 6) = 6 * x^2 + 26 * x + 24 := 
by 
  sorry

end expand_product_l1533_153349


namespace subset_if_a_neg_third_set_of_real_numbers_for_A_union_B_eq_A_l1533_153338

def A : Set ℝ := {x | x ^ 2 - 8 * x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

theorem subset_if_a_neg_third (a : ℝ) (h : a = -1/3) : B a ⊆ A := by
  sorry

theorem set_of_real_numbers_for_A_union_B_eq_A : {a : ℝ | A ∪ B a = A} = {0, -1/3, -1/5} := by
  sorry

end subset_if_a_neg_third_set_of_real_numbers_for_A_union_B_eq_A_l1533_153338


namespace face_opposite_to_turquoise_is_pink_l1533_153359

-- Declare the inductive type for the color of the face
inductive Color
| P -- Pink
| V -- Violet
| T -- Turquoise
| O -- Orange

open Color

-- Define the setup conditions of the problem
def cube_faces : List Color :=
  [P, P, P, V, V, T, O]

-- Define the positions of the faces for the particular folded cube configuration
-- Assuming the function cube_configuration gives the face opposite to a given face.
axiom cube_configuration : Color → Color

-- State the main theorem regarding the opposite face
theorem face_opposite_to_turquoise_is_pink : cube_configuration T = P :=
sorry

end face_opposite_to_turquoise_is_pink_l1533_153359


namespace sum_of_numbers_l1533_153386

theorem sum_of_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : a * b + b * c + c * a = 131) : 
  a + b + c = 22 := 
by 
  sorry

end sum_of_numbers_l1533_153386


namespace geometric_body_view_circle_l1533_153355

theorem geometric_body_view_circle (P : Type) (is_circle : P → Prop) (is_sphere : P → Prop)
  (is_cylinder : P → Prop) (is_cone : P → Prop) (is_rectangular_prism : P → Prop) :
  (∀ x, is_sphere x → is_circle x) →
  (∃ x, is_cylinder x ∧ is_circle x) →
  (∃ x, is_cone x ∧ is_circle x) →
  ¬ (∃ x, is_rectangular_prism x ∧ is_circle x) :=
by
  intros h_sphere h_cylinder h_cone h_rectangular_prism
  sorry

end geometric_body_view_circle_l1533_153355


namespace fred_more_than_daniel_l1533_153301

-- Definitions and conditions from the given problem.
def total_stickers : ℕ := 750
def andrew_kept : ℕ := 130
def daniel_received : ℕ := 250
def fred_received : ℕ := total_stickers - andrew_kept - daniel_received

-- The proof problem statement.
theorem fred_more_than_daniel : fred_received - daniel_received = 120 := by 
  sorry

end fred_more_than_daniel_l1533_153301


namespace max_car_passing_400_l1533_153348

noncomputable def max_cars_passing (speed : ℕ) (car_length : ℤ) (hour : ℕ) : ℕ :=
  20000 * speed / (5 * (speed + 1))

theorem max_car_passing_400 :
  max_cars_passing 20 5 1 / 10 = 400 := by
  sorry

end max_car_passing_400_l1533_153348


namespace mixed_number_sum_l1533_153392

theorem mixed_number_sum :
  481 + 1/6  + 265 + 1/12 + 904 + 1/20 -
  (184 + 29/30) - (160 + 41/42) - (703 + 55/56) =
  603 + 3/8 :=
by
  sorry

end mixed_number_sum_l1533_153392


namespace root_of_equation_l1533_153318

theorem root_of_equation :
  ∀ x : ℝ, (x - 3)^2 = x - 3 ↔ x = 3 ∨ x = 4 :=
by
  sorry

end root_of_equation_l1533_153318


namespace eval_expression_l1533_153356

theorem eval_expression : (538 * 538) - (537 * 539) = 1 :=
by
  sorry

end eval_expression_l1533_153356


namespace total_revenue_correct_l1533_153376

-- Define the costs of different types of returns
def cost_federal : ℕ := 50
def cost_state : ℕ := 30
def cost_quarterly : ℕ := 80

-- Define the quantities sold for different types of returns
def qty_federal : ℕ := 60
def qty_state : ℕ := 20
def qty_quarterly : ℕ := 10

-- Calculate the total revenue for the day
def total_revenue : ℕ := (cost_federal * qty_federal) + (cost_state * qty_state) + (cost_quarterly * qty_quarterly)

-- The theorem stating the total revenue calculation
theorem total_revenue_correct : total_revenue = 4400 := by
  sorry

end total_revenue_correct_l1533_153376


namespace min_value_expression_l1533_153369

theorem min_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (a + 2) ^ 2 + (b + 2) ^ 2 = 25 / 2 :=
sorry

end min_value_expression_l1533_153369


namespace deck_card_count_l1533_153382

theorem deck_card_count (r n : ℕ) (h1 : n = 2 * r) (h2 : n + 4 = 3 * r) : r + n = 12 :=
by
  sorry

end deck_card_count_l1533_153382


namespace veronica_pits_cherries_in_2_hours_l1533_153361

theorem veronica_pits_cherries_in_2_hours :
  ∀ (pounds_cherries : ℕ) (cherries_per_pound : ℕ)
    (time_first_pound : ℕ) (cherries_first_pound : ℕ)
    (time_second_pound : ℕ) (cherries_second_pound : ℕ)
    (time_third_pound : ℕ) (cherries_third_pound : ℕ)
    (minutes_per_hour : ℕ),
  pounds_cherries = 3 →
  cherries_per_pound = 80 →
  time_first_pound = 10 →
  cherries_first_pound = 20 →
  time_second_pound = 8 →
  cherries_second_pound = 20 →
  time_third_pound = 12 →
  cherries_third_pound = 20 →
  minutes_per_hour = 60 →
  ((time_first_pound / cherries_first_pound * cherries_per_pound) + 
   (time_second_pound / cherries_second_pound * cherries_per_pound) + 
   (time_third_pound / cherries_third_pound * cherries_per_pound)) / minutes_per_hour = 2 :=
by
  intros pounds_cherries cherries_per_pound
         time_first_pound cherries_first_pound
         time_second_pound cherries_second_pound
         time_third_pound cherries_third_pound
         minutes_per_hour
         pounds_eq cherries_eq
         time1_eq cherries1_eq
         time2_eq cherries2_eq
         time3_eq cherries3_eq
         mins_eq

  -- You would insert the proof here
  sorry

end veronica_pits_cherries_in_2_hours_l1533_153361
