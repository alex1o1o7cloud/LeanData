import Mathlib

namespace NUMINAMATH_GPT_fifth_iteration_perimeter_l1660_166006

theorem fifth_iteration_perimeter :
  let A1_side_length := 1
  let P1 := 3 * A1_side_length
  let P2 := 3 * (A1_side_length * 4 / 3)
  ∀ n : ℕ, P_n = 3 * (4 / 3) ^ (n - 1) →
  P_5 = 3 * (4 / 3) ^ 4 :=
  by sorry

end NUMINAMATH_GPT_fifth_iteration_perimeter_l1660_166006


namespace NUMINAMATH_GPT_inequality_proof_l1660_166008

variable (a b : ℝ)

theorem inequality_proof (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 4) : 
  (1 / (a^2 + b^2) ≤ 1 / 8) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1660_166008


namespace NUMINAMATH_GPT_shaded_region_area_l1660_166063

-- Given conditions
def diagonal_PQ : ℝ := 10
def number_of_squares : ℕ := 20

-- Definition of the side length of the squares
noncomputable def side_length := diagonal_PQ / (4 * Real.sqrt 2)

-- Area of one smaller square
noncomputable def one_square_area := side_length * side_length

-- Total area of the shaded region
noncomputable def total_area_of_shaded_region := number_of_squares * one_square_area

-- The theorem to be proven
theorem shaded_region_area : total_area_of_shaded_region = 62.5 := by
  sorry

end NUMINAMATH_GPT_shaded_region_area_l1660_166063


namespace NUMINAMATH_GPT_find_number_l1660_166039

theorem find_number (x : ℤ) (h : 7 * x + 37 = 100) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1660_166039


namespace NUMINAMATH_GPT_fibonacci_expression_equality_l1660_166041

-- Definition of the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

-- Statement to be proven
theorem fibonacci_expression_equality :
  (fibonacci 0 * fibonacci 2 + fibonacci 1 * fibonacci 3 + fibonacci 2 * fibonacci 4 +
  fibonacci 3 * fibonacci 5 + fibonacci 4 * fibonacci 6 + fibonacci 5 * fibonacci 7)
  - (fibonacci 1 ^ 2 + fibonacci 2 ^ 2 + fibonacci 3 ^ 2 + fibonacci 4 ^ 2 + fibonacci 5 ^ 2 + fibonacci 6 ^ 2)
  = 0 :=
by
  sorry

end NUMINAMATH_GPT_fibonacci_expression_equality_l1660_166041


namespace NUMINAMATH_GPT_geometric_series_first_term_l1660_166002

theorem geometric_series_first_term (r : ℚ) (S : ℚ) (a : ℚ) (h_ratio : r = 1/4) (h_sum : S = 80) (h_series : S = a / (1 - r)) :
  a = 60 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_first_term_l1660_166002


namespace NUMINAMATH_GPT_remainder_3n_plus_2_l1660_166028

-- Define the condition
def n_condition (n : ℤ) : Prop := n % 7 = 5

-- Define the theorem to be proved
theorem remainder_3n_plus_2 (n : ℤ) (h : n_condition n) : (3 * n + 2) % 7 = 3 := 
by sorry

end NUMINAMATH_GPT_remainder_3n_plus_2_l1660_166028


namespace NUMINAMATH_GPT_least_possible_value_l1660_166023

theorem least_possible_value (x y : ℝ) : (x + y - 1)^2 + (x * y)^2 ≥ 0 :=
by 
  sorry

end NUMINAMATH_GPT_least_possible_value_l1660_166023


namespace NUMINAMATH_GPT_businessmen_drink_neither_l1660_166003

theorem businessmen_drink_neither (n c t b : ℕ) 
  (h_n : n = 30) 
  (h_c : c = 15) 
  (h_t : t = 13) 
  (h_b : b = 7) : 
  n - (c + t - b) = 9 := 
  by
  sorry

end NUMINAMATH_GPT_businessmen_drink_neither_l1660_166003


namespace NUMINAMATH_GPT_polygons_after_cuts_l1660_166034

theorem polygons_after_cuts (initial_polygons : ℕ) (cuts : ℕ) 
  (initial_vertices : ℕ) (max_vertices_added_per_cut : ℕ) :
  (initial_polygons = 10) →
  (cuts = 51) →
  (initial_vertices = 100) →
  (max_vertices_added_per_cut = 4) →
  ∃ p, (p < 5 ∧ p ≥ 3) :=
by
  intros h_initial_polygons h_cuts h_initial_vertices h_max_vertices_added_per_cut
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_polygons_after_cuts_l1660_166034


namespace NUMINAMATH_GPT_minimum_value_expr_pos_reals_l1660_166092

noncomputable def expr (a b : ℝ) := a^2 + b^2 + 2 * a * b + 1 / (a + b)^2

theorem minimum_value_expr_pos_reals (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) : 
  (expr a b) ≥ 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_expr_pos_reals_l1660_166092


namespace NUMINAMATH_GPT_probability_f4_positive_l1660_166005

theorem probability_f4_positive {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x)
  (h_fn : ∀ x < 0, f x = a + x + Real.logb 2 (-x)) (h_a : a > -4 ∧ a < 5) :
  (1/3 : ℝ) < (2/3 : ℝ) :=
sorry

end NUMINAMATH_GPT_probability_f4_positive_l1660_166005


namespace NUMINAMATH_GPT_find_theta_in_interval_l1660_166070

variable (θ : ℝ)

def angle_condition (θ : ℝ) (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 1 ∧ (x^3 * Real.cos θ - x * (1 - x) + (1 - x)^3 * Real.tan θ > 0)

theorem find_theta_in_interval :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → angle_condition θ x) →
  0 < θ ∧ θ < Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_theta_in_interval_l1660_166070


namespace NUMINAMATH_GPT_ferry_P_travel_time_l1660_166076

-- Definitions of conditions
def speed_P : ℝ := 6 -- speed of ferry P in km/h
def speed_diff_PQ : ℝ := 3 -- speed difference between ferry Q and ferry P in km/h
def travel_longer_Q : ℝ := 2 -- ferry Q travels a route twice as long as ferry P
def time_diff_PQ : ℝ := 1 -- time difference between ferry Q and ferry P in hours

-- Distance traveled by ferry P
def distance_P (t_P : ℝ) : ℝ := speed_P * t_P

-- Distance traveled by ferry Q
def distance_Q (t_P : ℝ) : ℝ := travel_longer_Q * (speed_P * t_P)

-- Speed of ferry Q
def speed_Q : ℝ := speed_P + speed_diff_PQ

-- Time taken by ferry Q
def time_Q (t_P : ℝ) : ℝ := t_P + time_diff_PQ

-- Main theorem statement
theorem ferry_P_travel_time (t_P : ℝ) : t_P = 3 :=
by
  have eq_Q : speed_Q * (time_Q t_P) = distance_Q t_P := sorry
  have eq_P : speed_P * t_P = distance_P t_P := sorry
  sorry

end NUMINAMATH_GPT_ferry_P_travel_time_l1660_166076


namespace NUMINAMATH_GPT_area_to_paint_l1660_166050

def wall_height : ℕ := 10
def wall_length : ℕ := 15
def door_height : ℕ := 3
def door_length : ℕ := 5
def window_height : ℕ := 2
def window_length : ℕ := 3

theorem area_to_paint : (wall_height * wall_length) - (door_height * door_length + window_height * window_length) = 129 := by
  sorry

end NUMINAMATH_GPT_area_to_paint_l1660_166050


namespace NUMINAMATH_GPT_total_donuts_three_days_l1660_166036

def donuts_on_Monday := 14

def donuts_on_Tuesday := donuts_on_Monday / 2

def donuts_on_Wednesday := 4 * donuts_on_Monday

def total_donuts := donuts_on_Monday + donuts_on_Tuesday + donuts_on_Wednesday

theorem total_donuts_three_days : total_donuts = 77 :=
  by
    sorry

end NUMINAMATH_GPT_total_donuts_three_days_l1660_166036


namespace NUMINAMATH_GPT_remainder_140_div_k_l1660_166085

theorem remainder_140_div_k (k : ℕ) (hk : k > 0) :
  (80 % k^2 = 8) → (140 % k = 2) :=
by
  sorry

end NUMINAMATH_GPT_remainder_140_div_k_l1660_166085


namespace NUMINAMATH_GPT_carpet_needed_l1660_166015

/-- A rectangular room with dimensions 15 feet by 9 feet has a non-carpeted area occupied by 
a table with dimensions 3 feet by 2 feet. We want to prove that the number of square yards 
of carpet needed to cover the rest of the floor is 15. -/
theorem carpet_needed
  (room_length : ℝ) (room_width : ℝ) (table_length : ℝ) (table_width : ℝ)
  (h_room : room_length = 15) (h_room_width : room_width = 9)
  (h_table : table_length = 3) (h_table_width : table_width = 2) : 
  (⌈(((room_length * room_width) - (table_length * table_width)) / 9 : ℝ)⌉ = 15) := 
by
  sorry

end NUMINAMATH_GPT_carpet_needed_l1660_166015


namespace NUMINAMATH_GPT_sum_of_solutions_eq_l1660_166037

theorem sum_of_solutions_eq :
  let A := 100
  let B := 3
  (∃ x₁ x₂ x₃ : ℝ, 
    (x₁ = abs (B*x₁ - abs (A - B*x₁)) ∧ 
    x₂ = abs (B*x₂ - abs (A - B*x₂)) ∧ 
    x₃ = abs (B*x₃ - abs (A - B*x₃))) ∧ 
    (x₁ + x₂ + x₃ = (1900 : ℝ) / 7)) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_eq_l1660_166037


namespace NUMINAMATH_GPT_johnny_yellow_picks_l1660_166093

variable (total_picks red_picks blue_picks yellow_picks : ℕ)

theorem johnny_yellow_picks
    (h_total_picks : total_picks = 3 * blue_picks)
    (h_half_red_picks : red_picks = total_picks / 2)
    (h_blue_picks : blue_picks = 12)
    (h_pick_sum : total_picks = red_picks + blue_picks + yellow_picks) :
    yellow_picks = 6 := by
  sorry

end NUMINAMATH_GPT_johnny_yellow_picks_l1660_166093


namespace NUMINAMATH_GPT_difference_of_numbers_is_21938_l1660_166040

theorem difference_of_numbers_is_21938 
  (x y : ℕ) 
  (h1 : x + y = 26832) 
  (h2 : x % 10 = 0) 
  (h3 : y = x / 10 + 4) 
  : x - y = 21938 :=
sorry

end NUMINAMATH_GPT_difference_of_numbers_is_21938_l1660_166040


namespace NUMINAMATH_GPT_minimum_value_l1660_166009

theorem minimum_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 1) :
  (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) ≥ 9 / 4 :=
by sorry

end NUMINAMATH_GPT_minimum_value_l1660_166009


namespace NUMINAMATH_GPT_number_of_sides_of_regular_polygon_l1660_166012

theorem number_of_sides_of_regular_polygon (n : ℕ) (h : 0 < n) (h_angle : ∀ i, i < n → (2 * n - 4) * 90 / n = 150) : n = 12 :=
sorry

end NUMINAMATH_GPT_number_of_sides_of_regular_polygon_l1660_166012


namespace NUMINAMATH_GPT_triangle_sets_l1660_166071

def forms_triangle (a b c : ℕ) : Prop := 
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_sets :
  ¬ forms_triangle 1 2 3 ∧ forms_triangle 20 20 30 ∧ forms_triangle 30 10 15 ∧ forms_triangle 4 15 7 :=
by
  sorry

end NUMINAMATH_GPT_triangle_sets_l1660_166071


namespace NUMINAMATH_GPT_find_a_l1660_166045

def A (x : ℝ) : Set ℝ := {1, 2, x^2 - 5 * x + 9}
def B (x a : ℝ) : Set ℝ := {3, x^2 + a * x + a}

theorem find_a (a x : ℝ) (hxA : A x = {1, 2, 3}) (h2B : 2 ∈ B x a) :
  a = -2/3 ∨ a = -7/4 :=
by sorry

end NUMINAMATH_GPT_find_a_l1660_166045


namespace NUMINAMATH_GPT_tank_capacity_ratio_l1660_166014

-- Definitions from the problem conditions
def tank1_filled : ℝ := 300
def tank2_filled : ℝ := 450
def tank2_percentage_filled : ℝ := 0.45
def additional_needed : ℝ := 1250

-- Theorem statement
theorem tank_capacity_ratio (C1 C2 : ℝ) 
  (h1 : tank1_filled + tank2_filled + additional_needed = C1 + C2)
  (h2 : tank2_filled = tank2_percentage_filled * C2) : 
  C1 / C2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_tank_capacity_ratio_l1660_166014


namespace NUMINAMATH_GPT_find_m_from_root_l1660_166033

theorem find_m_from_root (m : ℝ) : (x : ℝ) = 1 → x^2 + m * x + 2 = 0 → m = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_from_root_l1660_166033


namespace NUMINAMATH_GPT_Sam_age_proof_l1660_166001

-- Define the conditions (Phoebe's current age, Raven's age relation, Sam's age definition)
def Phoebe_current_age : ℕ := 10
def Raven_in_5_years (R : ℕ) : Prop := R + 5 = 4 * (Phoebe_current_age + 5)
def Sam_age (R : ℕ) : ℕ := 2 * ((R + 3) - (Phoebe_current_age + 3))

-- The proof statement for Sam's current age
theorem Sam_age_proof (R : ℕ) (h : Raven_in_5_years R) : Sam_age R = 90 := by
  sorry

end NUMINAMATH_GPT_Sam_age_proof_l1660_166001


namespace NUMINAMATH_GPT_problem_ineq_l1660_166095

variable {a b c : ℝ}

theorem problem_ineq 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * a^2 * b^2 * c^2 := 
sorry

end NUMINAMATH_GPT_problem_ineq_l1660_166095


namespace NUMINAMATH_GPT_largest_fraction_l1660_166020

theorem largest_fraction (a b c d e : ℝ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < e) :
  (b + d + e) / (a + c) > max ((a + b + e) / (c + d))
                        (max ((a + d) / (b + e))
                            (max ((b + c) / (a + e)) ((c + e) / (a + b + d)))) := 
sorry

end NUMINAMATH_GPT_largest_fraction_l1660_166020


namespace NUMINAMATH_GPT_choose_3_out_of_13_l1660_166049

theorem choose_3_out_of_13: (Nat.choose 13 3) = 286 :=
by
  sorry

end NUMINAMATH_GPT_choose_3_out_of_13_l1660_166049


namespace NUMINAMATH_GPT_total_people_in_group_l1660_166066

-- Given conditions as definitions
def numChinese : Nat := 22
def numAmericans : Nat := 16
def numAustralians : Nat := 11

-- Statement of the theorem to prove
theorem total_people_in_group : (numChinese + numAmericans + numAustralians) = 49 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_people_in_group_l1660_166066


namespace NUMINAMATH_GPT_repeating_decimal_356_fraction_l1660_166094

noncomputable def repeating_decimal_356 := 3.0 + 56 / 99

theorem repeating_decimal_356_fraction : repeating_decimal_356 = 353 / 99 := by
  sorry

end NUMINAMATH_GPT_repeating_decimal_356_fraction_l1660_166094


namespace NUMINAMATH_GPT_unit_cost_decreases_l1660_166043

def regression_equation (x : ℝ) : ℝ := 356 - 1.5 * x

theorem unit_cost_decreases (x : ℝ) :
  regression_equation (x + 1) - regression_equation x = -1.5 := 
by sorry


end NUMINAMATH_GPT_unit_cost_decreases_l1660_166043


namespace NUMINAMATH_GPT_find_2g_x_l1660_166047

theorem find_2g_x (g : ℝ → ℝ) (h : ∀ x > 0, g (3 * x) = 3 / (3 + x)) (x : ℝ) (hx : x > 0) :
  2 * g x = 18 / (9 + x) :=
sorry

end NUMINAMATH_GPT_find_2g_x_l1660_166047


namespace NUMINAMATH_GPT_squared_difference_l1660_166080

theorem squared_difference (x y : ℝ) (h₁ : (x + y)^2 = 49) (h₂ : x * y = 8) : (x - y)^2 = 17 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_squared_difference_l1660_166080


namespace NUMINAMATH_GPT_charity_years_l1660_166052

theorem charity_years :
  ∃! pairs : List (ℕ × ℕ), 
    (∀ (w m : ℕ), (w, m) ∈ pairs → 18 * w + 30 * m = 55 * 12) ∧
    pairs.length = 6 :=
by
  sorry

end NUMINAMATH_GPT_charity_years_l1660_166052


namespace NUMINAMATH_GPT_range_of_a_l1660_166074

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2 * x + 1 + a * Real.log x

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = f a x2) ↔ (0 < a ∧ a < 1/2) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1660_166074


namespace NUMINAMATH_GPT_contrapositive_eq_l1660_166042

variables (P Q : Prop)

theorem contrapositive_eq : (¬P → Q) ↔ (¬Q → P) := 
by {
    sorry
}

end NUMINAMATH_GPT_contrapositive_eq_l1660_166042


namespace NUMINAMATH_GPT_smallest_next_divisor_l1660_166084

def isOddFourDigitNumber (n : ℕ) : Prop :=
  n % 2 = 1 ∧ 1000 ≤ n ∧ n < 10000

noncomputable def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ d => d > 0 ∧ n % d = 0)

theorem smallest_next_divisor (m : ℕ) (h₁ : isOddFourDigitNumber m) (h₂ : 437 ∈ divisors m) :
  ∃ k, k > 437 ∧ k ∈ divisors m ∧ k % 2 = 1 ∧ ∀ n, n > 437 ∧ n < k → n ∉ divisors m := by
  sorry

end NUMINAMATH_GPT_smallest_next_divisor_l1660_166084


namespace NUMINAMATH_GPT_find_point_P_l1660_166026

theorem find_point_P :
  ∃ (P : ℝ × ℝ), P.1 = 1 ∧ P.2 = 0 ∧ 
  (P.2 = P.1^4 - P.1) ∧
  (∃ m, m = 4 * P.1^3 - 1 ∧ m = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_point_P_l1660_166026


namespace NUMINAMATH_GPT_train_length_l1660_166011

noncomputable def length_of_train (speed_kmph : ℝ) (time_sec : ℝ) (length_platform_m : ℝ) : ℝ :=
  let speed_ms := (speed_kmph * 1000) / 3600
  let distance_covered := speed_ms * time_sec
  distance_covered - length_platform_m

theorem train_length :
  length_of_train 72 25 340.04 = 159.96 := by
  sorry

end NUMINAMATH_GPT_train_length_l1660_166011


namespace NUMINAMATH_GPT_find_breadth_of_rectangle_l1660_166054

noncomputable def breadth_of_rectangle (s : ℝ) (π_approx : ℝ := 3.14) : ℝ :=
2 * s - 22

theorem find_breadth_of_rectangle (b s : ℝ) (π_approx : ℝ := 3.14) :
  4 * s = 2 * (22 + b) →
  π_approx * s / 2 + s = 29.85 →
  b = 1.22 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_find_breadth_of_rectangle_l1660_166054


namespace NUMINAMATH_GPT_parabola_focus_distance_l1660_166022

noncomputable def distance_to_focus (p : ℝ) (M : ℝ × ℝ) : ℝ :=
  let focus := (p, 0)
  let (x1, y1) := M
  let (x2, y2) := focus
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) + p

theorem parabola_focus_distance
  (M : ℝ × ℝ) (p : ℝ)
  (hM : M = (2, 2))
  (hp : p = 1) :
  distance_to_focus p M = Real.sqrt 5 + 1 :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_distance_l1660_166022


namespace NUMINAMATH_GPT_zookeeper_configurations_l1660_166065

theorem zookeeper_configurations :
  ∃ (configs : ℕ), configs = 3 ∧ 
  (∀ (r p : ℕ), 
    30 * r + 35 * p = 1400 ∧ p ≥ r → 
    ((r, p) = (7, 34) ∨ (r, p) = (14, 28) ∨ (r, p) = (21, 22))) :=
sorry

end NUMINAMATH_GPT_zookeeper_configurations_l1660_166065


namespace NUMINAMATH_GPT_rate_of_pipe_B_l1660_166079

-- Definitions based on conditions
def tank_capacity : ℕ := 850
def pipe_A_rate : ℕ := 40
def pipe_C_rate : ℕ := 20
def cycle_time : ℕ := 3
def full_time : ℕ := 51

-- Prove that the rate of pipe B is 30 liters per minute
theorem rate_of_pipe_B (B : ℕ) : 
  (17 * (B + 20) = 850) → B = 30 := 
by 
  introv h1
  sorry

end NUMINAMATH_GPT_rate_of_pipe_B_l1660_166079


namespace NUMINAMATH_GPT_f_odd_and_inequality_l1660_166072

noncomputable def f (x : ℝ) : ℝ := (-2^x + 1) / (2^(x+1) + 2)

theorem f_odd_and_inequality (x c : ℝ) : ∀ x c, 
  f x < c^2 - 3 * c + 3 := by 
  sorry

end NUMINAMATH_GPT_f_odd_and_inequality_l1660_166072


namespace NUMINAMATH_GPT_complete_pairs_of_socks_l1660_166096

def initial_pairs_blue : ℕ := 20
def initial_pairs_green : ℕ := 15
def initial_pairs_red : ℕ := 15

def lost_socks_blue : ℕ := 3
def lost_socks_green : ℕ := 2
def lost_socks_red : ℕ := 2

def donated_socks_blue : ℕ := 10
def donated_socks_green : ℕ := 15
def donated_socks_red : ℕ := 10

def purchased_pairs_blue : ℕ := 5
def purchased_pairs_green : ℕ := 3
def purchased_pairs_red : ℕ := 2

def gifted_pairs_blue : ℕ := 2
def gifted_pairs_green : ℕ := 1

theorem complete_pairs_of_socks : 
  (initial_pairs_blue - 1 - (donated_socks_blue / 2) + purchased_pairs_blue + gifted_pairs_blue) +
  (initial_pairs_green - 1 - (donated_socks_green / 2) + purchased_pairs_green + gifted_pairs_green) +
  (initial_pairs_red - 1 - (donated_socks_red / 2) + purchased_pairs_red) = 43 := by
  sorry

end NUMINAMATH_GPT_complete_pairs_of_socks_l1660_166096


namespace NUMINAMATH_GPT_math_proof_problem_l1660_166098

noncomputable def sum_of_distinct_squares (a b c : ℕ) : ℕ :=
3 * ((a^2 + b^2 + c^2 : ℕ))

theorem math_proof_problem (a b c : ℕ)
  (h1 : a + b + c = 27)
  (h2 : Nat.gcd a b + Nat.gcd b c + Nat.gcd c a = 11) :
  sum_of_distinct_squares a b c = 2274 :=
sorry

end NUMINAMATH_GPT_math_proof_problem_l1660_166098


namespace NUMINAMATH_GPT_supermarket_A_is_more_cost_effective_l1660_166046

def price_A (kg : ℕ) : ℕ :=
  if kg <= 4 then kg * 10
  else 4 * 10 + (kg - 4) * 6

def price_B (kg : ℕ) : ℕ :=
  kg * 10 * 8 / 10

theorem supermarket_A_is_more_cost_effective :
  price_A 3 = 30 ∧ 
  price_A 5 = 46 ∧ 
  ∀ (x : ℕ), (x > 4) → price_A x = 6 * x + 16 ∧ 
  price_A 10 < price_B 10 :=
by 
  sorry

end NUMINAMATH_GPT_supermarket_A_is_more_cost_effective_l1660_166046


namespace NUMINAMATH_GPT_back_seat_people_l1660_166087

/-- Define the number of seats on the left side of the bus --/
def left_side_seats : ℕ := 15

/-- Define the number of seats on the right side of the bus (3 fewer because of the rear exit door) --/
def right_side_seats : ℕ := left_side_seats - 3

/-- Define the number of people each seat can hold --/
def people_per_seat : ℕ := 3

/-- Define the total capacity of the bus --/
def total_capacity : ℕ := 90

/-- Define the total number of people that can sit on the regular seats (left and right sides) --/
def regular_seats_people := (left_side_seats + right_side_seats) * people_per_seat

/-- Theorem stating the number of people that can sit at the back seat --/
theorem back_seat_people : (total_capacity - regular_seats_people) = 9 := by
  sorry

end NUMINAMATH_GPT_back_seat_people_l1660_166087


namespace NUMINAMATH_GPT_sum_of_edges_corners_faces_of_rectangular_prism_l1660_166060

-- Definitions based on conditions
def rectangular_prism_edges := 12
def rectangular_prism_corners := 8
def rectangular_prism_faces := 6
def resulting_sum := rectangular_prism_edges + rectangular_prism_corners + rectangular_prism_faces

-- Statement we want to prove
theorem sum_of_edges_corners_faces_of_rectangular_prism :
  resulting_sum = 26 := 
by 
  sorry -- Placeholder for the proof

end NUMINAMATH_GPT_sum_of_edges_corners_faces_of_rectangular_prism_l1660_166060


namespace NUMINAMATH_GPT_no_valid_partition_exists_l1660_166058

namespace MathProof

-- Define the set of positive integers
def N := {n : ℕ // n > 0}

-- Define non-empty sets A, B, C which are disjoint and partition N
def valid_partition (A B C : N → Prop) : Prop :=
  (∃ a, A a) ∧ (∃ b, B b) ∧ (∃ c, C c) ∧
  (∀ n, A n → ¬ B n ∧ ¬ C n) ∧
  (∀ n, B n → ¬ A n ∧ ¬ C n) ∧
  (∀ n, C n → ¬ A n ∧ ¬ B n) ∧
  (∀ n, A n ∨ B n ∨ C n)

-- Define the conditions in the problem
def condition_1 (A B C : N → Prop) : Prop :=
  ∀ a b, A a → B b → C ⟨a.val + b.val + 1, by linarith [a.prop, b.prop]⟩

def condition_2 (A B C : N → Prop) : Prop :=
  ∀ b c, B b → C c → A ⟨b.val + c.val + 1, by linarith [b.prop, c.prop]⟩

def condition_3 (A B C : N → Prop) : Prop :=
  ∀ c a, C c → A a → B ⟨c.val + a.val + 1, by linarith [c.prop, a.prop]⟩

-- State the problem that no valid partition exists
theorem no_valid_partition_exists :
  ¬ ∃ (A B C : N → Prop), valid_partition A B C ∧
    condition_1 A B C ∧
    condition_2 A B C ∧
    condition_3 A B C :=
by
  sorry

end MathProof

end NUMINAMATH_GPT_no_valid_partition_exists_l1660_166058


namespace NUMINAMATH_GPT_airplane_distance_difference_l1660_166057

theorem airplane_distance_difference (a : ℕ) : 
  let against_wind_distance := (a - 20) * 3
  let with_wind_distance := (a + 20) * 4
  with_wind_distance - against_wind_distance = a + 140 :=
by
  sorry

end NUMINAMATH_GPT_airplane_distance_difference_l1660_166057


namespace NUMINAMATH_GPT_find_analytical_expression_function_increasing_inequality_solution_l1660_166067

noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

-- Conditions
variables {a b x : ℝ}
axiom odd_function : ∀ x : ℝ, f a b (-x) = -f a b x
axiom half_value : f a b (1/2) = 2/5

-- Questions/Statements

-- 1. Analytical expression
theorem find_analytical_expression :
  ∃ a b, f a b x = x / (1 + x^2) := 
sorry

-- 2. Increasing function
theorem function_increasing :
  ∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < x2 ∧ x2 < 1 → f 1 0 x1 < f 1 0 x2 := 
sorry

-- 3. Inequality solution
theorem inequality_solution :
  ∀ x : ℝ, (x ∈ Set.Ioo (-1) 0 ∪ Set.Ioo 0 ((-1 + Real.sqrt 5) / 2)) → f 1 0 (x^2 - 1) + f 1 0 x < 0 := 
sorry

end NUMINAMATH_GPT_find_analytical_expression_function_increasing_inequality_solution_l1660_166067


namespace NUMINAMATH_GPT_binomial_coefficient_10_3_l1660_166086

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 :=
by
  sorry

end NUMINAMATH_GPT_binomial_coefficient_10_3_l1660_166086


namespace NUMINAMATH_GPT_find_distance_l1660_166059

variable (y : ℚ) -- The circumference of the bicycle wheel
variable (x : ℚ) -- The distance between the village and the field

-- Condition 1: The circumference of the truck's wheel is 4/3 of the bicycle's wheel
def circum_truck_eq : Prop := (4 / 3 : ℚ) * y = y

-- Condition 2: The circumference of the truck's wheel is 2 meters shorter than the tractor's track
def circum_truck_less : Prop := (4 / 3 : ℚ) * y + 2 = y + 2

-- Condition 3: Truck's wheel makes 100 fewer revolutions than the bicycle's wheel
def truck_100_fewer : Prop := x / ((4 / 3 : ℚ) * y) = (x / y) - 100

-- Condition 4: Truck's wheel makes 150 more revolutions than the tractor track
def truck_150_more : Prop := x / ((4 / 3 : ℚ) * y) = (x / ((4 / 3 : ℚ) * y + 2)) + 150

theorem find_distance (y : ℚ) (x : ℚ) :
  circum_truck_eq y →
  circum_truck_less y →
  truck_100_fewer x y →
  truck_150_more x y →
  x = 600 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_distance_l1660_166059


namespace NUMINAMATH_GPT_triangle_area_l1660_166010

theorem triangle_area (a c : ℝ) (B : ℝ) (h_a : a = 7) (h_c : c = 5) (h_B : B = 120 * Real.pi / 180) : 
  (1 / 2 * a * c * Real.sin B) = 35 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_GPT_triangle_area_l1660_166010


namespace NUMINAMATH_GPT_bisection_method_termination_condition_l1660_166030

theorem bisection_method_termination_condition (x1 x2 : ℝ) (ε : ℝ) : Prop :=
  |x1 - x2| < ε

end NUMINAMATH_GPT_bisection_method_termination_condition_l1660_166030


namespace NUMINAMATH_GPT_avg_speed_between_B_and_C_l1660_166029

noncomputable def avg_speed_from_B_to_C : ℕ := 20

theorem avg_speed_between_B_and_C
    (A_to_B_dist : ℕ := 120)
    (A_to_B_time : ℕ := 4)
    (B_to_C_dist : ℕ := 120) -- three-thirds of A_to_B_dist
    (C_to_D_dist : ℕ := 60) -- half of B_to_C_dist
    (C_to_D_time : ℕ := 2)
    (total_avg_speed : ℕ := 25)
    : avg_speed_from_B_to_C = 20 := 
  sorry

end NUMINAMATH_GPT_avg_speed_between_B_and_C_l1660_166029


namespace NUMINAMATH_GPT_quadratic_function_opens_downwards_l1660_166035

theorem quadratic_function_opens_downwards (m : ℤ) (h1 : |m| = 2) (h2 : m + 1 < 0) : m = -2 := by
  sorry

end NUMINAMATH_GPT_quadratic_function_opens_downwards_l1660_166035


namespace NUMINAMATH_GPT_circumradius_inradius_inequality_l1660_166069

theorem circumradius_inradius_inequality (a b c R r : ℝ) (hR : R > 0) (hr : r > 0) :
  R / (2 * r) ≥ ((64 * a^2 * b^2 * c^2) / 
  ((4 * a^2 - (b - c)^2) * (4 * b^2 - (c - a)^2) * (4 * c^2 - (a - b)^2)))^2 :=
sorry

end NUMINAMATH_GPT_circumradius_inradius_inequality_l1660_166069


namespace NUMINAMATH_GPT_square_side_length_l1660_166088

theorem square_side_length 
  (AF DH BG AE : ℝ) 
  (AF_eq : AF = 7) 
  (DH_eq : DH = 4) 
  (BG_eq : BG = 5) 
  (AE_eq : AE = 1) 
  (area_EFGH : ℝ) 
  (area_EFGH_eq : area_EFGH = 78) : 
  (∃ s : ℝ, s^2 = 144) :=
by
  use 12
  sorry

end NUMINAMATH_GPT_square_side_length_l1660_166088


namespace NUMINAMATH_GPT_find_third_number_l1660_166027

-- Definitions based on given conditions
def A : ℕ := 200
def C : ℕ := 100
def B : ℕ := 2 * C

-- The condition that the sum of A, B, and C is 500
def sum_condition : Prop := A + B + C = 500

-- The proof statement
theorem find_third_number : sum_condition → C = 100 := 
by
  have h1 : A = 200 := rfl
  have h2 : B = 2 * C := rfl
  have h3 : A + B + C = 500 := sorry
  sorry

end NUMINAMATH_GPT_find_third_number_l1660_166027


namespace NUMINAMATH_GPT_profit_percentage_l1660_166097

theorem profit_percentage (SP CP : ℤ) (h_SP : SP = 1170) (h_CP : CP = 975) :
  ((SP - CP : ℤ) * 100) / CP = 20 :=
by 
  sorry

end NUMINAMATH_GPT_profit_percentage_l1660_166097


namespace NUMINAMATH_GPT_volume_of_regular_tetrahedron_l1660_166073

noncomputable def volume_of_tetrahedron (a H : ℝ) : ℝ :=
  (a^2 * H) / (6 * Real.sqrt 2)

theorem volume_of_regular_tetrahedron
  (d_face : ℝ)
  (d_edge : ℝ)
  (h : Real.sqrt 14 = d_edge)
  (h1 : 2 = d_face)
  (volume_approx : ℝ) :
  ∃ a H, (d_face = Real.sqrt ((H / 2)^2 + (a * Real.sqrt 3 / 6)^2) ∧ 
          d_edge = Real.sqrt ((H / 2)^2 + (a / (2 * Real.sqrt 3))^2) ∧ 
          Real.sqrt (volume_of_tetrahedron a H) = 533.38) :=
  sorry

end NUMINAMATH_GPT_volume_of_regular_tetrahedron_l1660_166073


namespace NUMINAMATH_GPT_eval_floor_abs_value_l1660_166078

theorem eval_floor_abs_value : ⌊|(-45.8 : ℝ)|⌋ = 45 := by
  sorry -- Proof is to be filled in

end NUMINAMATH_GPT_eval_floor_abs_value_l1660_166078


namespace NUMINAMATH_GPT_find_number_l1660_166021

theorem find_number (x number : ℝ) (h₁ : 5 - (5 / x) = number + (4 / x)) (h₂ : x = 9) : number = 4 :=
by
  subst h₂
  -- proof steps
  sorry

end NUMINAMATH_GPT_find_number_l1660_166021


namespace NUMINAMATH_GPT_expression_value_zero_l1660_166024

variable (x : ℝ)

theorem expression_value_zero (h : x^2 + 3 * x - 3 = 0) : x^3 + 2 * x^2 - 6 * x + 3 = 0 := 
by
  sorry

end NUMINAMATH_GPT_expression_value_zero_l1660_166024


namespace NUMINAMATH_GPT_solve_inequality_l1660_166099

theorem solve_inequality (x : ℝ) : (x^2 + 7 * x < 8) ↔ x ∈ (Set.Ioo (-8 : ℝ) 1) := by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1660_166099


namespace NUMINAMATH_GPT_pieces_of_wood_for_table_l1660_166077

theorem pieces_of_wood_for_table :
  ∀ (T : ℕ), (24 * T + 48 * 8 = 672) → T = 12 :=
by
  intro T
  intro h
  sorry

end NUMINAMATH_GPT_pieces_of_wood_for_table_l1660_166077


namespace NUMINAMATH_GPT_largest_three_digit_int_l1660_166025

theorem largest_three_digit_int (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 999) (h2 : 75 * n ≡ 225 [MOD 300]) : n = 999 :=
sorry

end NUMINAMATH_GPT_largest_three_digit_int_l1660_166025


namespace NUMINAMATH_GPT_condition_implies_at_least_one_gt_one_l1660_166075

theorem condition_implies_at_least_one_gt_one (a b : ℝ) :
  (a + b > 2 → (a > 1 ∨ b > 1)) ∧ ¬(a^2 + b^2 > 2 → (a > 1 ∨ b > 1)) :=
by
  sorry

end NUMINAMATH_GPT_condition_implies_at_least_one_gt_one_l1660_166075


namespace NUMINAMATH_GPT_sum_of_numbers_l1660_166090

theorem sum_of_numbers (a b : ℕ) (h_lcm : Nat.lcm a b = 54) (h_ratio : a / b = 2 / 3) : a + b = 45 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l1660_166090


namespace NUMINAMATH_GPT_hyperbola_satisfies_conditions_l1660_166004

-- Define the equations of the hyperbolas as predicates
def hyperbola_A (x y : ℝ) : Prop := x^2 - (y^2 / 4) = 1
def hyperbola_B (x y : ℝ) : Prop := (x^2 / 4) - y^2 = 1
def hyperbola_C (x y : ℝ) : Prop := (y^2 / 4) - x^2 = 1
def hyperbola_D (x y : ℝ) : Prop := y^2 - (x^2 / 4) = 1

-- Define the conditions on foci and asymptotes
def foci_on_y_axis (h : (ℝ → ℝ → Prop)) : Prop := 
  h = hyperbola_C ∨ h = hyperbola_D

def has_asymptotes (h : (ℝ → ℝ → Prop)) : Prop :=
  ∀ x y, h x y → (y = (1/2) * x ∨ y = -(1/2) * x)

-- The proof statement
theorem hyperbola_satisfies_conditions :
  foci_on_y_axis hyperbola_D ∧ has_asymptotes hyperbola_D ∧ 
    (¬ (foci_on_y_axis hyperbola_A ∧ has_asymptotes hyperbola_A)) ∧ 
    (¬ (foci_on_y_axis hyperbola_B ∧ has_asymptotes hyperbola_B)) ∧ 
    (¬ (foci_on_y_axis hyperbola_C ∧ has_asymptotes hyperbola_C)) := 
by
  sorry

end NUMINAMATH_GPT_hyperbola_satisfies_conditions_l1660_166004


namespace NUMINAMATH_GPT_tan_theta_minus_pi_over_4_l1660_166068

theorem tan_theta_minus_pi_over_4 
  (θ : Real) (h1 : π / 2 < θ ∧ θ < 2 * π)
  (h2 : Real.sin (θ + π / 4) = -3 / 5) :
  Real.tan (θ - π / 4) = 4 / 3 := 
  sorry

end NUMINAMATH_GPT_tan_theta_minus_pi_over_4_l1660_166068


namespace NUMINAMATH_GPT_fraction_evaluation_l1660_166064

theorem fraction_evaluation : (3 / 8 : ℚ) + 7 / 12 - 2 / 9 = 53 / 72 := by
  sorry

end NUMINAMATH_GPT_fraction_evaluation_l1660_166064


namespace NUMINAMATH_GPT_positive_difference_l1660_166055

def a := 8^2
def b := a + a
def c := a * a
theorem positive_difference : ((b / 8) - (c / 8)) = 496 := by
  sorry

end NUMINAMATH_GPT_positive_difference_l1660_166055


namespace NUMINAMATH_GPT_ned_short_sleeve_shirts_l1660_166062

theorem ned_short_sleeve_shirts (washed_shirts not_washed_shirts long_sleeve_shirts total_shirts : ℕ)
  (h1 : washed_shirts = 29) (h2 : not_washed_shirts = 1) (h3 : long_sleeve_shirts = 21)
  (h4 : total_shirts = washed_shirts + not_washed_shirts) :
  total_shirts - long_sleeve_shirts = 9 :=
by
  sorry

end NUMINAMATH_GPT_ned_short_sleeve_shirts_l1660_166062


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1660_166044

theorem arithmetic_sequence_sum :
  let a := -3
  let d := 7
  let n := 10
  let s := n * (2 * a + (n - 1) * d) / 2
  s = 285 :=
by
  -- Details of the proof are omitted as per instructions
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1660_166044


namespace NUMINAMATH_GPT_hexagon_angles_sum_l1660_166053

theorem hexagon_angles_sum (α β γ δ ε ζ : ℝ)
  (h1 : α + γ + ε = 180)
  (h2 : β + δ + ζ = 180) : 
  α + β + γ + δ + ε + ζ = 360 :=
by 
  sorry

end NUMINAMATH_GPT_hexagon_angles_sum_l1660_166053


namespace NUMINAMATH_GPT_books_sold_correct_l1660_166038

-- Define the number of books sold by Matias, Olivia, and Luke on each day
def matias_monday := 7
def olivia_monday := 5
def luke_monday := 12

def matias_tuesday := 2 * matias_monday
def olivia_tuesday := 3 * olivia_monday
def luke_tuesday := luke_monday / 2

def matias_wednesday := 3 * matias_tuesday
def olivia_wednesday := 4 * olivia_tuesday
def luke_wednesday := luke_tuesday

-- Calculate the total books sold by each person over three days
def matias_total := matias_monday + matias_tuesday + matias_wednesday
def olivia_total := olivia_monday + olivia_tuesday + olivia_wednesday
def luke_total := luke_monday + luke_tuesday + luke_wednesday

-- Calculate the combined total of books sold by Matias, Olivia, and Luke
def combined_total := matias_total + olivia_total + luke_total

-- Prove the combined total equals 167
theorem books_sold_correct : combined_total = 167 := by
  sorry

end NUMINAMATH_GPT_books_sold_correct_l1660_166038


namespace NUMINAMATH_GPT_hyperbola_parabola_focus_l1660_166081

theorem hyperbola_parabola_focus (m : ℝ) :
  (m + (m - 2) = 4) → m = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_hyperbola_parabola_focus_l1660_166081


namespace NUMINAMATH_GPT_problem_proof_l1660_166091

theorem problem_proof (p : ℕ) (hodd : p % 2 = 1) (hgt : p > 3):
  ((p - 3) ^ (1 / 2 * (p - 1)) - 1 ∣ p - 4) ∧
  ¬((p - 3) ^ (1 / 2 * (p - 1)) + 1 ∣ p) ∧
  ¬((p - 3) ^ (1 / 2 * (p - 1)) ∣ p) ∧
  ¬((p - 3) ^ (1 / 2 * (p - 1)) + 1 ∣ p + 1) ∧
  ¬((p - 3) ^ (1 / 2 * (p - 1)) - 1 ∣ p - 3) :=
by
  sorry

end NUMINAMATH_GPT_problem_proof_l1660_166091


namespace NUMINAMATH_GPT_stones_in_10th_pattern_l1660_166013

def stones_in_nth_pattern (n : ℕ) : ℕ :=
n * (3 * n - 1) / 2 + 1

theorem stones_in_10th_pattern : stones_in_nth_pattern 10 = 145 :=
by
  sorry

end NUMINAMATH_GPT_stones_in_10th_pattern_l1660_166013


namespace NUMINAMATH_GPT_number_of_workers_l1660_166048

theorem number_of_workers 
  (W : ℕ) 
  (h1 : 750 * W = (5 * 900) + 700 * (W - 5)) : 
  W = 20 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_workers_l1660_166048


namespace NUMINAMATH_GPT_min_value_of_sum_of_squares_l1660_166000

theorem min_value_of_sum_of_squares (a b c : ℕ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
  (h : a^2 + b^2 - c = 2022) : 
  a^2 + b^2 + c^2 = 2034 ∧ a = 27 ∧ b = 36 ∧ c = 3 := 
sorry

end NUMINAMATH_GPT_min_value_of_sum_of_squares_l1660_166000


namespace NUMINAMATH_GPT_test_two_categorical_features_l1660_166017

-- Definitions based on the problem conditions
def is_testing_method (method : String) : Prop :=
  method = "Three-dimensional bar chart" ∨
  method = "Two-dimensional bar chart" ∨
  method = "Contour bar chart" ∨
  method = "Independence test"

noncomputable def correct_method : String :=
  "Independence test"

-- Theorem statement based on the problem and solution
theorem test_two_categorical_features :
  ∀ m : String, is_testing_method m → m = correct_method :=
by
  sorry

end NUMINAMATH_GPT_test_two_categorical_features_l1660_166017


namespace NUMINAMATH_GPT_binom_30_3_eq_4060_l1660_166083

theorem binom_30_3_eq_4060 : Nat.choose 30 3 = 4060 := 
  by sorry

end NUMINAMATH_GPT_binom_30_3_eq_4060_l1660_166083


namespace NUMINAMATH_GPT_cos_sin_gt_sin_cos_l1660_166031

theorem cos_sin_gt_sin_cos (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi) : Real.cos (Real.sin x) > Real.sin (Real.cos x) :=
by
  sorry

end NUMINAMATH_GPT_cos_sin_gt_sin_cos_l1660_166031


namespace NUMINAMATH_GPT_triangle_angle_sum_cannot_exist_l1660_166032

theorem triangle_angle_sum (A : Real) (B : Real) (C : Real) :
    A + B + C = 180 :=
sorry

theorem cannot_exist (right_two_60 : ¬∃ A B C : Real, A = 90 ∧ B = 60 ∧ C = 60 ∧ A + B + C = 180) 
    (scalene_100 : ∃ A B C : Real, A = 100 ∧ A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ A + B + C = 180)
    (isosceles_two_70 : ∃ A B C : Real, A = B ∧ A = 70 ∧ C = 180 - 2 * A ∧ A + B + C = 180)
    (equilateral_60 : ∃ A B C : Real, A = 60 ∧ B = 60 ∧ C = 60 ∧ A + B + C = 180)
    (one_90_two_50 : ¬∃ A B C : Real, A = 90 ∧ B = 50 ∧ C = 50 ∧ A + B + C = 180) :
  (¬∃ A B C : Real, A = 90 ∧ B = 60 ∧ C = 60 ∧ A + B + C = 180) ∧
  (¬∃ A B C : Real, A = 90 ∧ B = 50 ∧ C = 50 ∧ A + B + C = 180) :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_sum_cannot_exist_l1660_166032


namespace NUMINAMATH_GPT_find_m_l1660_166016

open Nat

theorem find_m (m : ℕ) (h_pos : 0 < m) 
  (a : ℕ := Nat.choose (2 * m) m) 
  (b : ℕ := Nat.choose (2 * m + 1) m)
  (h_eq : 13 * a = 7 * b) : 
  m = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1660_166016


namespace NUMINAMATH_GPT_prove_min_period_and_max_value_l1660_166018

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 - (Real.cos x)^2

theorem prove_min_period_and_max_value :
  (∀ x : ℝ, f (x + π) = f x) ∧ (∀ y : ℝ, y ≤ f y) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_prove_min_period_and_max_value_l1660_166018


namespace NUMINAMATH_GPT_distance_A_to_B_l1660_166082

theorem distance_A_to_B : 
  ∀ (D : ℕ),
    let boat_speed_with_wind := 21
    let boat_speed_against_wind := 17
    let time_for_round_trip := 7
    let stream_speed_ab := 3
    let stream_speed_ba := 2
    let effective_speed_ab := boat_speed_with_wind + stream_speed_ab
    let effective_speed_ba := boat_speed_against_wind - stream_speed_ba
    D / effective_speed_ab + D / effective_speed_ba = time_for_round_trip →
    D = 65 :=
by
  sorry

end NUMINAMATH_GPT_distance_A_to_B_l1660_166082


namespace NUMINAMATH_GPT_max_pizzas_l1660_166051

theorem max_pizzas (dough_available cheese_available sauce_available pepperoni_available mushroom_available olive_available sausage_available: ℝ)
  (dough_per_pizza cheese_per_pizza sauce_per_pizza toppings_per_pizza: ℝ)
  (total_toppings: ℝ)
  (toppings_per_pizza_sum: total_toppings = pepperoni_available + mushroom_available + olive_available + sausage_available)
  (dough_cond: dough_available = 200)
  (cheese_cond: cheese_available = 20)
  (sauce_cond: sauce_available = 20)
  (pepperoni_cond: pepperoni_available = 15)
  (mushroom_cond: mushroom_available = 5)
  (olive_cond: olive_available = 5)
  (sausage_cond: sausage_available = 10)
  (dough_per_pizza_cond: dough_per_pizza = 1)
  (cheese_per_pizza_cond: cheese_per_pizza = 1/4)
  (sauce_per_pizza_cond: sauce_per_pizza = 1/6)
  (toppings_per_pizza_cond: toppings_per_pizza = 1/3)
  : (min (dough_available / dough_per_pizza) (min (cheese_available / cheese_per_pizza) (min (sauce_available / sauce_per_pizza) (total_toppings / toppings_per_pizza))) = 80) :=
by
  sorry

end NUMINAMATH_GPT_max_pizzas_l1660_166051


namespace NUMINAMATH_GPT_distance_after_four_steps_l1660_166056

theorem distance_after_four_steps (total_distance : ℝ) (steps : ℕ) (steps_taken : ℕ) :
   total_distance = 25 → steps = 7 → steps_taken = 4 → (steps_taken * (total_distance / steps) = 100 / 7) :=
by
    intro h1 h2 h3
    rw [h1, h2, h3]
    simp
    sorry

end NUMINAMATH_GPT_distance_after_four_steps_l1660_166056


namespace NUMINAMATH_GPT_determine_m_type_l1660_166007

theorem determine_m_type (m : ℝ) :
  ((m^2 + 2*m - 8 = 0) ↔ (m = -4)) ∧
  ((m^2 - 2*m = 0) ↔ (m = 0 ∨ m = 2)) ∧
  ((m^2 - 2*m ≠ 0) ↔ (m ≠ 0 ∧ m ≠ 2)) :=
by sorry

end NUMINAMATH_GPT_determine_m_type_l1660_166007


namespace NUMINAMATH_GPT_dislikes_TV_and_books_l1660_166089

-- The problem conditions
def total_people : ℕ := 800
def percent_dislikes_TV : ℚ := 25 / 100
def percent_dislikes_both : ℚ := 15 / 100

-- The expected answer
def expected_dislikes_TV_and_books : ℕ := 30

-- The proof problem statement
theorem dislikes_TV_and_books : 
  (total_people * percent_dislikes_TV) * percent_dislikes_both = expected_dislikes_TV_and_books := by 
  sorry

end NUMINAMATH_GPT_dislikes_TV_and_books_l1660_166089


namespace NUMINAMATH_GPT_group_allocation_minimizes_time_total_duration_after_transfer_l1660_166019

theorem group_allocation_minimizes_time :
  ∃ x y : ℕ,
  x + y = 52 ∧
  (x = 20 ∧ y = 32) ∧
  (min (60 / x) (100 / y) = 25 / 8) := sorry

theorem total_duration_after_transfer (x y x' y' : ℕ) (H : x = 20) (H1 : y = 32) (H2 : x' = x - 6) (H3 : y' = y + 6) :
  min ((100 * (2/5)) / x') ((152 * (2/3)) / y') = 27 / 7 := sorry

end NUMINAMATH_GPT_group_allocation_minimizes_time_total_duration_after_transfer_l1660_166019


namespace NUMINAMATH_GPT_fraction_increase_by_two_l1660_166061

theorem fraction_increase_by_two (x y : ℝ) : 
  (3 * (2 * x) * (2 * y)) / (2 * x + 2 * y) = 2 * (3 * x * y) / (x + y) :=
by
  sorry

end NUMINAMATH_GPT_fraction_increase_by_two_l1660_166061
