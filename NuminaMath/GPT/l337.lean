import Mathlib

namespace NUMINAMATH_GPT_calculate_total_area_of_figure_l337_33768

-- Defining the lengths of the segments according to the problem conditions.
def length_1 : ℕ := 8
def length_2 : ℕ := 6
def length_3 : ℕ := 3
def length_4 : ℕ := 5
def length_5 : ℕ := 2
def length_6 : ℕ := 4

-- Using the given lengths to compute the areas of the smaller rectangles
def area_A : ℕ := length_1 * length_2
def area_B : ℕ := length_4 * (10 - 6)
def area_C : ℕ := (6 - 3) * (15 - 10)

-- The total area of the figure is the sum of the areas of the smaller rectangles
def total_area : ℕ := area_A + area_B + area_C

-- The statement to prove
theorem calculate_total_area_of_figure : total_area = 83 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_calculate_total_area_of_figure_l337_33768


namespace NUMINAMATH_GPT_find_fourth_number_l337_33716

theorem find_fourth_number (x : ℝ) (h : 3 + 33 + 333 + x = 369.63) : x = 0.63 :=
sorry

end NUMINAMATH_GPT_find_fourth_number_l337_33716


namespace NUMINAMATH_GPT_perpendicular_lines_slope_l337_33759

theorem perpendicular_lines_slope (a : ℝ) (h1 :  a * (a + 2) = -1) : a = -1 :=
by 
-- Perpendicularity condition given
sorry

end NUMINAMATH_GPT_perpendicular_lines_slope_l337_33759


namespace NUMINAMATH_GPT_least_number_of_roots_l337_33770

variable {g : ℝ → ℝ}

-- Conditions
axiom g_defined (x : ℝ) : g x = g x
axiom g_symmetry_1 (x : ℝ) : g (3 + x) = g (3 - x)
axiom g_symmetry_2 (x : ℝ) : g (5 + x) = g (5 - x)
axiom g_at_1 : g 1 = 0

-- Root count in the interval
theorem least_number_of_roots : ∃ (n : ℕ), n >= 250 ∧ (∀ m, -1000 ≤ (1 + 8 * m:ℝ) ∧ (1 + 8 * m:ℝ) ≤ 1000 → g (1 + 8 * m) = 0) :=
sorry

end NUMINAMATH_GPT_least_number_of_roots_l337_33770


namespace NUMINAMATH_GPT_x_plus_p_eq_2p_plus_3_l337_33720

theorem x_plus_p_eq_2p_plus_3 (x p : ℝ) (h1 : |x - 3| = p) (h2 : x > 3) : x + p = 2 * p + 3 := by
  sorry

end NUMINAMATH_GPT_x_plus_p_eq_2p_plus_3_l337_33720


namespace NUMINAMATH_GPT_blue_balls_count_l337_33711

def num_purple : Nat := 7
def num_yellow : Nat := 11
def min_tries : Nat := 19

theorem blue_balls_count (num_blue: Nat): num_blue = 1 :=
by
  have worst_case_picks := num_purple + num_yellow
  have h := min_tries
  sorry

end NUMINAMATH_GPT_blue_balls_count_l337_33711


namespace NUMINAMATH_GPT_find_original_number_l337_33751

theorem find_original_number (x : ℝ) :
  (((x / 2.5) - 10.5) * 0.3 = 5.85) -> x = 75 :=
by
  sorry

end NUMINAMATH_GPT_find_original_number_l337_33751


namespace NUMINAMATH_GPT_sum_of_cubes_condition_l337_33729

theorem sum_of_cubes_condition (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_condition : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) :
  a^3 + b^3 + c^3 = -36 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_condition_l337_33729


namespace NUMINAMATH_GPT_factor_tree_X_value_l337_33778

def H : ℕ := 2 * 5
def J : ℕ := 3 * 7
def F : ℕ := 7 * H
def G : ℕ := 11 * J
def X : ℕ := F * G

theorem factor_tree_X_value : X = 16170 := by
  sorry

end NUMINAMATH_GPT_factor_tree_X_value_l337_33778


namespace NUMINAMATH_GPT_pablo_books_read_l337_33773

noncomputable def pages_per_book : ℕ := 150
noncomputable def cents_per_page : ℕ := 1
noncomputable def cost_of_candy : ℕ := 1500    -- $15 in cents
noncomputable def leftover_money : ℕ := 300    -- $3 in cents
noncomputable def total_money := cost_of_candy + leftover_money
noncomputable def earnings_per_book := pages_per_book * cents_per_page

theorem pablo_books_read : total_money / earnings_per_book = 12 := by
  sorry

end NUMINAMATH_GPT_pablo_books_read_l337_33773


namespace NUMINAMATH_GPT_partition_cities_l337_33787

-- Define the type for cities and airlines.
variable (City : Type) (Airline : Type)

-- Define the number of cities and airlines
variable (n k : ℕ)

-- Define a relation to represent bidirectional direct flights
variable (flight : Airline → City → City → Prop)

-- Define the condition: Some pairs of cities are connected by exactly one direct flight operated by one of the airline companies
-- or there are no such flights between them.
axiom unique_flight : ∀ (a : Airline) (c1 c2 : City), flight a c1 c2 → ¬ (∃ (a' : Airline), flight a' c1 c2 ∧ a' ≠ a)

-- Define the condition: Any two direct flights operated by the same company share a common endpoint
axiom shared_endpoint :
  ∀ (a : Airline) (c1 c2 c3 c4 : City), flight a c1 c2 → flight a c3 c4 → (c1 = c3 ∨ c1 = c4 ∨ c2 = c3 ∨ c2 = c4)

-- The main theorem to prove
theorem partition_cities :
  ∃ (partition : City → Fin (k + 2)), ∀ (c1 c2 : City) (a : Airline), flight a c1 c2 → partition c1 ≠ partition c2 :=
sorry

end NUMINAMATH_GPT_partition_cities_l337_33787


namespace NUMINAMATH_GPT_problem_solution_l337_33704

-- Define the problem conditions and state the theorem
variable (a b : ℝ)
variable (h1 : a^2 - 4 * a + 3 = 0)
variable (h2 : b^2 - 4 * b + 3 = 0)
variable (h3 : a ≠ b)

theorem problem_solution : (a+1)*(b+1) = 8 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l337_33704


namespace NUMINAMATH_GPT_pencil_eraser_cost_l337_33709

theorem pencil_eraser_cost (p e : ℕ) (h1 : 15 * p + 5 * e = 200) (h2 : p > e) (h_p_pos : p > 0) (h_e_pos : e > 0) :
  p + e = 18 :=
  sorry

end NUMINAMATH_GPT_pencil_eraser_cost_l337_33709


namespace NUMINAMATH_GPT_part1_solution_count_part2_solution_count_l337_33727

theorem part1_solution_count :
  ∃ (solutions : Finset (ℕ × ℕ × ℕ)), solutions.card = 7 ∧
    ∀ (m n r : ℕ), (m, n, r) ∈ solutions ↔ mn + nr + mr = 2 * (m + n + r) := sorry

theorem part2_solution_count (k : ℕ) (h : 1 < k) :
  ∃ (solutions : Finset (ℕ × ℕ × ℕ)), solutions.card ≥ 3 * k + 1 ∧
    ∀ (m n r : ℕ), (m, n, r) ∈ solutions ↔ mn + nr + mr = k * (m + n + r) := sorry

end NUMINAMATH_GPT_part1_solution_count_part2_solution_count_l337_33727


namespace NUMINAMATH_GPT_next_bell_ringing_time_l337_33713

theorem next_bell_ringing_time (post_office_interval train_station_interval town_hall_interval start_time : ℕ)
  (h1 : post_office_interval = 18)
  (h2 : train_station_interval = 24)
  (h3 : town_hall_interval = 30)
  (h4 : start_time = 9) :
  let lcm := Nat.lcm post_office_interval (Nat.lcm train_station_interval town_hall_interval)
  lcm + start_time = 15 := by
  sorry

end NUMINAMATH_GPT_next_bell_ringing_time_l337_33713


namespace NUMINAMATH_GPT_x_gt_1_implies_inv_x_lt_1_inv_x_lt_1_not_necessitates_x_gt_1_l337_33776

theorem x_gt_1_implies_inv_x_lt_1 (x : ℝ) (h : x > 1) : 1 / x < 1 :=
by
  sorry

theorem inv_x_lt_1_not_necessitates_x_gt_1 (x : ℝ) (h : 1 / x < 1) : ¬(x > 1) ∨ (x ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_x_gt_1_implies_inv_x_lt_1_inv_x_lt_1_not_necessitates_x_gt_1_l337_33776


namespace NUMINAMATH_GPT_ratio_A_B_l337_33761

-- Define constants for non-zero numbers A and B
variables {A B : ℕ} (h1 : A ≠ 0) (h2 : B ≠ 0)

-- Define the given condition
theorem ratio_A_B (h : (2 * A) * 7 = (3 * B) * 3) : A / B = 9 / 14 := by
  sorry

end NUMINAMATH_GPT_ratio_A_B_l337_33761


namespace NUMINAMATH_GPT_solve_for_x_l337_33765

theorem solve_for_x (y : ℝ) : 
  ∃ x : ℝ, 19 * (x + y) + 17 = 19 * (-x + y) - 21 ∧ 
           x = -21 / 38 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l337_33765


namespace NUMINAMATH_GPT_real_part_of_z_l337_33769

open Complex

theorem real_part_of_z (z : ℂ) (h : I * z = 1 + 2 * I) : z.re = 2 :=
sorry

end NUMINAMATH_GPT_real_part_of_z_l337_33769


namespace NUMINAMATH_GPT_find_x_l337_33717

theorem find_x {x : ℝ} (hx : x^2 - 5 * x = -4) : x = 1 ∨ x = 4 :=
sorry

end NUMINAMATH_GPT_find_x_l337_33717


namespace NUMINAMATH_GPT_fg_3_eq_123_l337_33750

def f (x : ℤ) : ℤ := x^2 + 2
def g (x : ℤ) : ℤ := 3 * x + 2

theorem fg_3_eq_123 : f (g 3) = 123 := by
  sorry

end NUMINAMATH_GPT_fg_3_eq_123_l337_33750


namespace NUMINAMATH_GPT_Haleigh_can_make_3_candles_l337_33703

variable (n20 n5 n1 : ℕ) (w20 w5 w1 oz10 : ℝ)

def wax_leftover (n20 n5 n1 : ℕ) (w20 w5 w1 oz10 : ℝ) : ℝ := 
  n20 * w20 + n5 * w5 + n1 * w1 

theorem Haleigh_can_make_3_candles :
  ∀ (n20 n5 n1 : ℕ) (w20 w5 w1 oz10 : ℝ), 
  n20 = 5 →
  w20 = 2 →
  n5 = 5 →
  w5 = 0.5 →
  n1 = 25 →
  w1 = 0.1 →
  oz10 = 10 →
  (wax_leftover n20 n5 n1 w20 w5 w1 oz10) / 5 = 3 := 
by
  intros n20 n5 n1 w20 w5 w1 oz10 hn20 hw20 hn5 hw5 hn1 hw1 hoz10
  rw [hn20, hw20, hn5, hw5, hn1, hw1, hoz10]
  sorry

end NUMINAMATH_GPT_Haleigh_can_make_3_candles_l337_33703


namespace NUMINAMATH_GPT_alex_walking_distance_l337_33743

theorem alex_walking_distance
  (distance : ℝ)
  (time_45 : ℝ)
  (walking_rate : distance = 1.5 ∧ time_45 = 45):
  ∃ distance_90, distance_90 = 3 :=
by 
  sorry

end NUMINAMATH_GPT_alex_walking_distance_l337_33743


namespace NUMINAMATH_GPT_percentage_passed_exam_l337_33723

theorem percentage_passed_exam (total_students failed_students : ℕ) (h_total : total_students = 540) (h_failed : failed_students = 351) :
  (total_students - failed_students) * 100 / total_students = 35 :=
by
  sorry

end NUMINAMATH_GPT_percentage_passed_exam_l337_33723


namespace NUMINAMATH_GPT_original_price_of_radio_l337_33706

theorem original_price_of_radio (P : ℝ) (h : 0.95 * P = 465.5) : P = 490 :=
sorry

end NUMINAMATH_GPT_original_price_of_radio_l337_33706


namespace NUMINAMATH_GPT_total_strings_correct_l337_33735

-- Definitions based on conditions
def num_ukuleles : ℕ := 2
def num_guitars : ℕ := 4
def num_violins : ℕ := 2
def strings_per_ukulele : ℕ := 4
def strings_per_guitar : ℕ := 6
def strings_per_violin : ℕ := 4

-- Total number of strings
def total_strings : ℕ := num_ukuleles * strings_per_ukulele +
                         num_guitars * strings_per_guitar +
                         num_violins * strings_per_violin

-- The proof statement
theorem total_strings_correct : total_strings = 40 :=
by
  -- Proof omitted.
  sorry

end NUMINAMATH_GPT_total_strings_correct_l337_33735


namespace NUMINAMATH_GPT_bicycle_cost_price_l337_33780

theorem bicycle_cost_price (CP_A : ℝ) (CP_B : ℝ) (SP_C : ℝ)
    (h1 : CP_B = 1.60 * CP_A)
    (h2 : SP_C = 1.25 * CP_B)
    (h3 : SP_C = 225) :
    CP_A = 225 / 2.00 :=
by
  sorry -- the proof steps will follow here

end NUMINAMATH_GPT_bicycle_cost_price_l337_33780


namespace NUMINAMATH_GPT_min_value_f_at_0_l337_33744

noncomputable def f (a x : ℝ) : ℝ :=
if x ≤ 0 then (x - a)^2 else x + 1/x + a

theorem min_value_f_at_0 (a : ℝ) : (∀ x : ℝ, f a 0 ≤ f a x) ↔ 0 ≤ a ∧ a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_f_at_0_l337_33744


namespace NUMINAMATH_GPT_probability_neither_red_nor_purple_l337_33726

def total_balls : ℕ := 60
def white_balls : ℕ := 22
def green_balls : ℕ := 18
def yellow_balls : ℕ := 8
def red_balls : ℕ := 5
def purple_balls : ℕ := 7

theorem probability_neither_red_nor_purple : 
  (total_balls - (red_balls + purple_balls)) / total_balls = 4 / 5 :=
by sorry

end NUMINAMATH_GPT_probability_neither_red_nor_purple_l337_33726


namespace NUMINAMATH_GPT_factorize_diff_of_squares_l337_33708

theorem factorize_diff_of_squares (m : ℝ) : m^2 - 4 = (m + 2) * (m - 2) :=
  sorry

end NUMINAMATH_GPT_factorize_diff_of_squares_l337_33708


namespace NUMINAMATH_GPT_correct_statements_l337_33791

theorem correct_statements (f : ℝ → ℝ) (t : ℝ)
  (h1 : ∀ x1 x2 : ℝ, x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) < 0)
  (h2 : (∀ x : ℝ, f x = f (-x)) ∧ (∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 < x2 → f x1 > f x2) ∧ f (-2) = 0)
  (h3 : ∀ x : ℝ, f (-x) = -f x)
  (h4 : ∀ x : ℝ, f (x - t) = f (x + t)) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 > f x2 ↔ x1 < x2) ∧
  (∀ x : ℝ, f x - f (|x|) = - (f (-x) - f (|x|))) :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_l337_33791


namespace NUMINAMATH_GPT_fabric_difference_fabric_total_l337_33731

noncomputable def fabric_used_coat : ℝ := 1.55
noncomputable def fabric_used_pants : ℝ := 1.05

theorem fabric_difference : fabric_used_coat - fabric_used_pants = 0.5 :=
by
  sorry

theorem fabric_total : fabric_used_coat + fabric_used_ppants = 2.6 :=
by
  sorry

end NUMINAMATH_GPT_fabric_difference_fabric_total_l337_33731


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_l337_33705

-- Proof for Problem 1
theorem problem1_solution (x y : ℝ) 
(h1 : x - y - 1 = 4)
(h2 : 4 * (x - y) - y = 5) : 
x = 20 ∧ y = 15 := sorry

-- Proof for Problem 2
theorem problem2_solution (x : ℝ) 
(h1 : 4 * x - 1 ≥ x + 1)
(h2 : (1 - x) / 2 < x) : 
x ≥ 2 / 3 := sorry

end NUMINAMATH_GPT_problem1_solution_problem2_solution_l337_33705


namespace NUMINAMATH_GPT_second_card_is_three_l337_33796

theorem second_card_is_three (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
                             (h_sum : a + b + c + d = 30)
                             (h_increasing : a < b ∧ b < c ∧ c < d)
                             (h_dennis : ∀ x y z, x = a → (y ≠ b ∨ z ≠ c ∨ d ≠ 30 - a - y - z))
                             (h_mandy : ∀ x y z, x = b → (y ≠ a ∨ z ≠ c ∨ d ≠ 30 - x - y - z))
                             (h_sandy : ∀ x y z, x = c → (y ≠ a ∨ z ≠ b ∨ d ≠ 30 - x - y - z))
                             (h_randy : ∀ x y z, x = d → (y ≠ a ∨ z ≠ b ∨ c ≠ 30 - x - y - z)) :
  b = 3 := 
sorry

end NUMINAMATH_GPT_second_card_is_three_l337_33796


namespace NUMINAMATH_GPT_compute_expression_l337_33781

-- Definition of the imaginary unit i
class ImaginaryUnit (i : ℂ) where
  I_square : i * i = -1

-- Definition of non-zero real number a
variable (a : ℝ) (h_a : a ≠ 0)

-- Theorem to prove the equivalence
theorem compute_expression (i : ℂ) [ImaginaryUnit i] :
  (a * i - i⁻¹)⁻¹ = -i / (a + 1) :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l337_33781


namespace NUMINAMATH_GPT_workers_contribution_l337_33737

theorem workers_contribution (W C : ℕ) (h1 : W * C = 300000) (h2 : W * (C + 50) = 360000) : W = 1200 :=
by
  sorry

end NUMINAMATH_GPT_workers_contribution_l337_33737


namespace NUMINAMATH_GPT_all_statements_correct_l337_33758

theorem all_statements_correct :
  (∀ (b h : ℝ), (3 * b * h = 3 * (b * h))) ∧
  (∀ (b h : ℝ), (1/2 * b * (1/2 * h) = 1/2 * (1/2 * b * h))) ∧
  (∀ (r : ℝ), (π * (2 * r) ^ 2 = 4 * (π * r ^ 2))) ∧
  (∀ (r : ℝ), (π * (3 * r) ^ 2 = 9 * (π * r ^ 2))) ∧
  (∀ (s : ℝ), ((2 * s) ^ 2 = 4 * (s ^ 2)))
  → False := 
by 
  intros h
  sorry

end NUMINAMATH_GPT_all_statements_correct_l337_33758


namespace NUMINAMATH_GPT_min_value_512_l337_33771

noncomputable def min_value (a b c d e f g h : ℝ) : ℝ :=
  (2 * a * e)^2 + (2 * b * f)^2 + (2 * c * g)^2 + (2 * d * h)^2

theorem min_value_512 
  (a b c d e f g h : ℝ)
  (H1 : a * b * c * d = 8)
  (H2 : e * f * g * h = 16) : 
  ∃ (min_val : ℝ), min_val = 512 ∧ min_value a b c d e f g h = min_val :=
sorry

end NUMINAMATH_GPT_min_value_512_l337_33771


namespace NUMINAMATH_GPT_a3_plus_a4_value_l337_33718

theorem a3_plus_a4_value
  (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h : (1 - 2*x)^5 = a_0 + a_1*(1 + x) + a_2*(1 + x)^2 + a_3*(1 + x)^3 + a_4*(1 + x)^4 + a_5*(1 + x)^5) :
  a_3 + a_4 = -480 := 
sorry

end NUMINAMATH_GPT_a3_plus_a4_value_l337_33718


namespace NUMINAMATH_GPT_geometric_progression_complex_l337_33740

theorem geometric_progression_complex (a b c m : ℂ) (r : ℂ) (hr : r ≠ 0) 
    (h1 : a = r) (h2 : b = r^2) (h3 : c = r^3) 
    (h4 : a / (1 - b) = m) (h5 : b / (1 - c) = m) (h6 : c / (1 - a) = m) : 
    ∃ m : ℂ, ∀ a b c : ℂ, ∃ r : ℂ, a = r ∧ b = r^2 ∧ c = r^3 
    ∧ r ≠ 0 
    ∧ (a / (1 - b) = m) 
    ∧ (b / (1 - c) = m) 
    ∧ (c / (1 - a) = m) := 
sorry

end NUMINAMATH_GPT_geometric_progression_complex_l337_33740


namespace NUMINAMATH_GPT_not_perfect_square_l337_33733

theorem not_perfect_square (a b : ℤ) (ha : 0 < a) (hb : 0 < b) (h : ¬ (a^2 - b^2) % 4 = 0) : 
  ¬ ∃ k : ℤ, (a + 3*b) * (5*a + 7*b) = k^2 :=
sorry

end NUMINAMATH_GPT_not_perfect_square_l337_33733


namespace NUMINAMATH_GPT_arithmetic_seq_sum_l337_33757

/-- Given an arithmetic sequence {a_n} such that a_5 + a_6 + a_7 = 15,
prove that the sum of the first 11 terms of the sequence S_11 is 55. -/
theorem arithmetic_seq_sum (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h₁ : a 5 + a 6 + a 7 = 15)
  (h₂ : ∀ n, S n = n * (a 1 + a n) / 2) :
  S 11 = 55 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l337_33757


namespace NUMINAMATH_GPT_nonnegative_for_interval_l337_33715

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 * (x - 2)^2) / ((1 - x) * (1 + x + x^2))

theorem nonnegative_for_interval (x : ℝ) : (f x >= 0) ↔ (0 <= x) :=
by
  sorry

end NUMINAMATH_GPT_nonnegative_for_interval_l337_33715


namespace NUMINAMATH_GPT_amount_per_friend_l337_33777

-- Definitions based on conditions
def cost_of_erasers : ℝ := 5 * 200
def cost_of_pencils : ℝ := 7 * 800
def total_cost : ℝ := cost_of_erasers + cost_of_pencils
def number_of_friends : ℝ := 4

-- The proof statement
theorem amount_per_friend : (total_cost / number_of_friends) = 1650 := by
  sorry

end NUMINAMATH_GPT_amount_per_friend_l337_33777


namespace NUMINAMATH_GPT_hoseok_needs_17_more_jumps_l337_33762

/-- Define the number of jumps by Hoseok and Minyoung -/
def hoseok_jumps : ℕ := 34
def minyoung_jumps : ℕ := 51

/-- Define the number of additional jumps Hoseok needs -/
def additional_jumps_hoseok : ℕ := minyoung_jumps - hoseok_jumps

/-- Prove that the additional jumps Hoseok needs is equal to 17 -/
theorem hoseok_needs_17_more_jumps (h_jumps : ℕ := hoseok_jumps) (m_jumps : ℕ := minyoung_jumps) :
  additional_jumps_hoseok = 17 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_hoseok_needs_17_more_jumps_l337_33762


namespace NUMINAMATH_GPT_max_value_f_l337_33772

noncomputable def f (x : ℝ) : ℝ := Real.sin (2*x) - 2 * Real.sqrt 3 * (Real.sin x)^2

theorem max_value_f : ∃ x : ℝ, f x = 2 - Real.sqrt 3 :=
  sorry

end NUMINAMATH_GPT_max_value_f_l337_33772


namespace NUMINAMATH_GPT_Dima_floor_l337_33788

theorem Dima_floor (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 9)
  (h2 : 60 = (n - 1))
  (h3 : 70 = (n - 1) / (n - 1) * 60 + (n - n / 2) * 2 * 60)
  (h4 : ∀ m : ℕ, 1 ≤ m ∧ m ≤ 9 → (5 * n = 6 * m + 1) → (n = 7 ∧ m = 6)) :
  n = 7 :=
by
  sorry

end NUMINAMATH_GPT_Dima_floor_l337_33788


namespace NUMINAMATH_GPT_fountain_area_l337_33755

theorem fountain_area (A B D C : ℝ) (h₁ : B - A = 20) (h₂ : D = (A + B) / 2) (h₃ : C - D = 12) :
  ∃ R : ℝ, R^2 = 244 ∧ π * R^2 = 244 * π :=
by
  sorry

end NUMINAMATH_GPT_fountain_area_l337_33755


namespace NUMINAMATH_GPT_simplify_expression_l337_33760

theorem simplify_expression (x y : ℝ) :
  (2 * x + 25) + (150 * x + 40) + (5 * y + 10) = 152 * x + 5 * y + 75 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l337_33760


namespace NUMINAMATH_GPT_sweet_treats_per_student_l337_33766

theorem sweet_treats_per_student :
  let cookies := 20
  let cupcakes := 25
  let brownies := 35
  let students := 20
  (cookies + cupcakes + brownies) / students = 4 :=
by 
  sorry

end NUMINAMATH_GPT_sweet_treats_per_student_l337_33766


namespace NUMINAMATH_GPT_count_valid_pairs_l337_33725

open Nat

-- Define the conditions
def room_conditions (p q : ℕ) : Prop :=
  q > p ∧
  (∃ (p' q' : ℕ), p = p' + 6 ∧ q = q' + 6 ∧ p' * q' = 48)

-- State the theorem to prove the number of valid pairs (p, q)
theorem count_valid_pairs : 
  (∃ l : List (ℕ × ℕ), 
    (∀ pq ∈ l, room_conditions pq.fst pq.snd) ∧ 
    l.length = 5) := 
sorry

end NUMINAMATH_GPT_count_valid_pairs_l337_33725


namespace NUMINAMATH_GPT_find_N_l337_33741

theorem find_N : ∀ N : ℕ, (991 + 993 + 995 + 997 + 999 = 5000 - N) → N = 25 :=
by
  intro N h
  sorry

end NUMINAMATH_GPT_find_N_l337_33741


namespace NUMINAMATH_GPT_grasshopper_max_reach_points_l337_33721

theorem grasshopper_max_reach_points
  (α : ℝ) (α_eq : α = 36 * Real.pi / 180)
  (L : ℕ)
  (jump_constant : ∀ (n : ℕ), L = L) :
  ∃ (N : ℕ), N ≤ 10 :=
by 
  sorry

end NUMINAMATH_GPT_grasshopper_max_reach_points_l337_33721


namespace NUMINAMATH_GPT_pos_int_solns_to_eq_l337_33738

open Int

theorem pos_int_solns_to_eq (x y z : ℤ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  x^2 + y^2 - z^2 = 9 - 2 * x * y ↔ 
    (x, y, z) = (5, 0, 4) ∨ (x, y, z) = (4, 1, 4) ∨ (x, y, z) = (3, 2, 4) ∨ 
    (x, y, z) = (2, 3, 4) ∨ (x, y, z) = (1, 4, 4) ∨ (x, y, z) = (0, 5, 4) ∨ 
    (x, y, z) = (3, 0, 0) ∨ (x, y, z) = (2, 1, 0) ∨ (x, y, z) = (1, 2, 0) ∨ 
    (x, y, z) = (0, 3, 0) :=
by sorry

end NUMINAMATH_GPT_pos_int_solns_to_eq_l337_33738


namespace NUMINAMATH_GPT_cars_towards_each_other_cars_same_direction_A_to_B_cars_same_direction_B_to_A_l337_33700

-- Definitions based on conditions
def distanceAB := 18  -- km
def speedCarA := 54   -- km/h
def speedCarB := 36   -- km/h
def targetDistance := 45  -- km

-- Proof problem statements
theorem cars_towards_each_other {y : ℝ} : 54 * y + 36 * y = 18 + 45 ↔ y = 0.7 :=
by sorry

theorem cars_same_direction_A_to_B {x : ℝ} : 54 * x - (36 * x + 18) = 45 ↔ x = 3.5 :=
by sorry

theorem cars_same_direction_B_to_A {x : ℝ} : 54 * x + 18 - 36 * x = 45 ↔ x = 1.5 :=
by sorry

end NUMINAMATH_GPT_cars_towards_each_other_cars_same_direction_A_to_B_cars_same_direction_B_to_A_l337_33700


namespace NUMINAMATH_GPT_smallest_non_factor_product_of_factors_of_60_l337_33794

theorem smallest_non_factor_product_of_factors_of_60 :
  ∃ x y : ℕ, x ≠ y ∧ x ∣ 60 ∧ y ∣ 60 ∧ ¬ (x * y ∣ 60) ∧ ∀ x' y' : ℕ, x' ≠ y' → x' ∣ 60 → y' ∣ 60 → ¬(x' * y' ∣ 60) → x * y ≤ x' * y' := 
sorry

end NUMINAMATH_GPT_smallest_non_factor_product_of_factors_of_60_l337_33794


namespace NUMINAMATH_GPT_total_bins_used_l337_33752

def bins_of_soup : ℝ := 0.12
def bins_of_vegetables : ℝ := 0.12
def bins_of_pasta : ℝ := 0.5

theorem total_bins_used : bins_of_soup + bins_of_vegetables + bins_of_pasta = 0.74 :=
by
  sorry

end NUMINAMATH_GPT_total_bins_used_l337_33752


namespace NUMINAMATH_GPT_conditional_probabilities_l337_33712

def PA : ℝ := 0.20
def PB : ℝ := 0.18
def PAB : ℝ := 0.12

theorem conditional_probabilities :
  PAB / PB = 2 / 3 ∧ PAB / PA = 3 / 5 := by
  sorry

end NUMINAMATH_GPT_conditional_probabilities_l337_33712


namespace NUMINAMATH_GPT_max_values_of_f_smallest_positive_period_of_f_intervals_where_f_is_monotonically_increasing_l337_33748

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (x / 2) + Real.sqrt 3 * Real.cos (x / 2)

theorem max_values_of_f (k : ℤ) : 
  ∃ x, f x = 2 ∧ x = 4 * (k : ℝ) * Real.pi - (2 * Real.pi / 3) := 
sorry

theorem smallest_positive_period_of_f : 
  ∃ T, T = 4 * Real.pi := 
sorry

theorem intervals_where_f_is_monotonically_increasing (k : ℤ) : 
  ∀ x, (-(5 * Real.pi / 3) + 4 * (k : ℝ) * Real.pi ≤ x) ∧ (x ≤ Real.pi / 3 + 4 * (k : ℝ) * Real.pi) → 
  ∀ y, (-(5 * Real.pi / 3) + 4 * (k : ℝ) * Real.pi ≤ y) ∧ (y ≤ Real.pi / 3 + 4 * (k : ℝ) * Real.pi) → 
  (x ≤ y ↔ f x ≤ f y) :=
sorry

end NUMINAMATH_GPT_max_values_of_f_smallest_positive_period_of_f_intervals_where_f_is_monotonically_increasing_l337_33748


namespace NUMINAMATH_GPT_find_a_plus_b_l337_33734

theorem find_a_plus_b (a b : ℤ) (h1 : a^2 = 16) (h2 : b^3 = -27) (h3 : |a - b| = a - b) : a + b = 1 := by
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l337_33734


namespace NUMINAMATH_GPT_perfect_square_trinomial_t_l337_33745

theorem perfect_square_trinomial_t (a b t : ℝ) :
  (∃ (x y : ℝ), x = a ∧ y = 2 * b ∧ a^2 + (2 * t - 1) * a * b + 4 * b^2 = (x + y)^2) →
  (t = 5 / 2 ∨ t = -3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_t_l337_33745


namespace NUMINAMATH_GPT_third_chest_coin_difference_l337_33722

variable (g1 g2 g3 s1 s2 s3 : ℕ)

-- Conditions
axiom h1 : g1 + g2 + g3 = 40
axiom h2 : s1 + s2 + s3 = 40
axiom h3 : g1 = s1 + 7
axiom h4 : g2 = s2 + 15

-- Goal
theorem third_chest_coin_difference : s3 = g3 + 22 :=
sorry

end NUMINAMATH_GPT_third_chest_coin_difference_l337_33722


namespace NUMINAMATH_GPT_intersection_area_l337_33785

-- Define the square vertices
def vertex1 : (ℝ × ℝ) := (2, 8)
def vertex2 : (ℝ × ℝ) := (13, 8)
def vertex3 : (ℝ × ℝ) := (13, -3)
def vertex4 : (ℝ × ℝ) := (2, -3)  -- Derived from the conditions

-- Define the circle with center and radius
def circle_center : (ℝ × ℝ) := (2, -3)
def circle_radius : ℝ := 4

-- Define the square side length
def square_side_length : ℝ := 11  -- From vertex (2, 8) to vertex (2, -3)

-- Prove the intersection area
theorem intersection_area :
  let area := (1 / 4) * Real.pi * (circle_radius^2)
  area = 4 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_intersection_area_l337_33785


namespace NUMINAMATH_GPT_minimum_apples_collected_l337_33782

-- Anya, Vanya, Dania, Sanya, and Tanya each collected an integer percentage of the total number of apples,
-- with all these percentages distinct and greater than zero.
-- Prove that the minimum total number of apples is 20.

theorem minimum_apples_collected :
  ∃ (n : ℕ), (∀ (a v d s t : ℕ), 
    1 ≤ a ∧ 1 ≤ v ∧ 1 ≤ d ∧ 1 ≤ s ∧ 1 ≤ t ∧
    a ≠ v ∧ a ≠ d ∧ a ≠ s ∧ a ≠ t ∧ 
    v ≠ d ∧ v ≠ s ∧ v ≠ t ∧ 
    d ≠ s ∧ d ≠ t ∧ 
    s ≠ t ∧
    a + v + d + s + t = 100) →
  n ≥ 20 :=
by 
  sorry

end NUMINAMATH_GPT_minimum_apples_collected_l337_33782


namespace NUMINAMATH_GPT_rooks_non_attacking_kings_non_attacking_bishops_non_attacking_knights_non_attacking_queens_non_attacking_l337_33728

-- Define the problem conditions: number of ways to place two same-color rooks that do not attack each other.
def num_ways_rooks : ℕ := 1568
theorem rooks_non_attacking : ∃ (n : ℕ), n = num_ways_rooks := by
  sorry

-- Define the problem conditions: number of ways to place two same-color kings that do not attack each other.
def num_ways_kings : ℕ := 1806
theorem kings_non_attacking : ∃ (n : ℕ), n = num_ways_kings := by
  sorry

-- Define the problem conditions: number of ways to place two same-color bishops that do not attack each other.
def num_ways_bishops : ℕ := 1736
theorem bishops_non_attacking : ∃ (n : ℕ), n = num_ways_bishops := by
  sorry

-- Define the problem conditions: number of ways to place two same-color knights that do not attack each other.
def num_ways_knights : ℕ := 1848
theorem knights_non_attacking : ∃ (n : ℕ), n = num_ways_knights := by
  sorry

-- Define the problem conditions: number of ways to place two same-color queens that do not attack each other.
def num_ways_queens : ℕ := 1288
theorem queens_non_attacking : ∃ (n : ℕ), n = num_ways_queens := by
  sorry

end NUMINAMATH_GPT_rooks_non_attacking_kings_non_attacking_bishops_non_attacking_knights_non_attacking_queens_non_attacking_l337_33728


namespace NUMINAMATH_GPT_total_penalty_kicks_l337_33746

theorem total_penalty_kicks (total_players : ℕ) (goalies : ℕ) (hoop_challenges : ℕ)
  (h_total : total_players = 25) (h_goalies : goalies = 5) (h_hoop_challenges : hoop_challenges = 10) :
  (goalies * (total_players - 1)) = 120 :=
by
  sorry

end NUMINAMATH_GPT_total_penalty_kicks_l337_33746


namespace NUMINAMATH_GPT_similar_triangles_iff_l337_33784

variables {a b c a' b' c' : ℂ}

theorem similar_triangles_iff :
  (∃ (z w : ℂ), a' = a * z + w ∧ b' = b * z + w ∧ c' = c * z + w) ↔
  a' * (b - c) + b' * (c - a) + c' * (a - b) = 0 :=
sorry

end NUMINAMATH_GPT_similar_triangles_iff_l337_33784


namespace NUMINAMATH_GPT_water_left_after_experiment_l337_33724

theorem water_left_after_experiment (initial_water : ℝ) (used_water : ℝ) (result_water : ℝ) 
  (h1 : initial_water = 3) 
  (h2 : used_water = 9 / 4) 
  (h3 : result_water = 3 / 4) : 
  initial_water - used_water = result_water := by
  sorry

end NUMINAMATH_GPT_water_left_after_experiment_l337_33724


namespace NUMINAMATH_GPT_smallest_intersection_value_l337_33797

theorem smallest_intersection_value (a b : ℝ) (f g : ℝ → ℝ)
    (Hf : ∀ x, f x = x^4 - 6 * x^3 + 11 * x^2 - 6 * x + a)
    (Hg : ∀ x, g x = x + b)
    (Hinter : ∀ x, f x = g x → true):
  ∃ x₀, x₀ = 0 :=
by
  intros
  -- Further steps would involve proving roots and conditions stated but omitted here.
  sorry

end NUMINAMATH_GPT_smallest_intersection_value_l337_33797


namespace NUMINAMATH_GPT_number_of_students_in_third_batch_l337_33707

theorem number_of_students_in_third_batch
  (avg1 avg2 avg3 : ℕ)
  (total_avg : ℚ)
  (students1 students2 : ℕ)
  (h_avg1 : avg1 = 45)
  (h_avg2 : avg2 = 55)
  (h_avg3 : avg3 = 65)
  (h_total_avg : total_avg = 56.333333333333336)
  (h_students1 : students1 = 40)
  (h_students2 : students2 = 50) :
  ∃ x : ℕ, (students1 * avg1 + students2 * avg2 + x * avg3 = total_avg * (students1 + students2 + x) ∧ x = 60) :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_in_third_batch_l337_33707


namespace NUMINAMATH_GPT_redPoints_l337_33701

open Nat

def isRedPoint (x y : ℕ) : Prop :=
  (y = (x - 36) * (x - 144) - 1991) ∧ (∃ m : ℕ, y = m * m)

theorem redPoints :
  {p : ℕ × ℕ | isRedPoint p.1 p.2} = { (2544, 6017209), (444, 120409) } :=
by
  sorry

end NUMINAMATH_GPT_redPoints_l337_33701


namespace NUMINAMATH_GPT_length_of_train_l337_33742

variable (L : ℝ) (S : ℝ)

-- Condition 1: The train crosses a 120 meters platform in 15 seconds
axiom condition1 : S = (L + 120) / 15

-- Condition 2: The train crosses a 250 meters platform in 20 seconds
axiom condition2 : S = (L + 250) / 20

-- The theorem to be proved
theorem length_of_train : L = 270 :=
by
  sorry

end NUMINAMATH_GPT_length_of_train_l337_33742


namespace NUMINAMATH_GPT_find_time_for_compound_interest_l337_33789

noncomputable def compound_interest_time 
  (A P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  (Real.log (A / P)) / (n * Real.log (1 + r / n))

theorem find_time_for_compound_interest :
  compound_interest_time 500 453.51473922902494 0.05 1 = 2 :=
sorry

end NUMINAMATH_GPT_find_time_for_compound_interest_l337_33789


namespace NUMINAMATH_GPT_triangle_side_relation_l337_33798

variable {α β γ : ℝ} -- angles in the triangle
variable {a b c : ℝ} -- sides opposite to the angles

theorem triangle_side_relation
  (h1 : α = 3 * β)
  (h2 : α = 6 * γ)
  (h_sum : α + β + γ = 180)
  : b * c^2 = (a + b) * (a - b)^2 := 
by
  sorry

end NUMINAMATH_GPT_triangle_side_relation_l337_33798


namespace NUMINAMATH_GPT_hours_l337_33799

def mechanic_hours_charged (h : ℕ) : Prop :=
  45 * h + 225 = 450

theorem hours (h : ℕ) : mechanic_hours_charged h → h = 5 :=
by
  intro h_eq
  have : 45 * h + 225 = 450 := h_eq
  sorry

end NUMINAMATH_GPT_hours_l337_33799


namespace NUMINAMATH_GPT_simplify_expression_l337_33763

theorem simplify_expression (k : ℤ) : 
  let a := 1
  let b := 3
  (6 * k + 18) / 6 = k + 3 ∧ a / b = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l337_33763


namespace NUMINAMATH_GPT_determinant_of_sine_matrix_is_zero_l337_33730

theorem determinant_of_sine_matrix_is_zero : 
  let M : Matrix (Fin 3) (Fin 3) ℝ :=
    ![![Real.sin 2, Real.sin 3, Real.sin 4],
      ![Real.sin 5, Real.sin 6, Real.sin 7],
      ![Real.sin 8, Real.sin 9, Real.sin 10]]
  Matrix.det M = 0 := 
by sorry

end NUMINAMATH_GPT_determinant_of_sine_matrix_is_zero_l337_33730


namespace NUMINAMATH_GPT_number_of_integers_between_sqrt10_and_sqrt100_l337_33714

theorem number_of_integers_between_sqrt10_and_sqrt100 : 
  let a := Real.sqrt 10
  let b := Real.sqrt 100
  ∃ (n : ℕ), n = 6 ∧ (∀ x : ℕ, (x > a ∧ x < b) → (4 ≤ x ∧ x ≤ 9)) :=
by
  sorry

end NUMINAMATH_GPT_number_of_integers_between_sqrt10_and_sqrt100_l337_33714


namespace NUMINAMATH_GPT_five_digit_number_is_40637_l337_33779

theorem five_digit_number_is_40637 
  (A B C D E F G : ℕ) 
  (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ 
        B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ 
        C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ 
        D ≠ E ∧ D ≠ F ∧ D ≠ G ∧
        E ≠ F ∧ E ≠ G ∧ 
        F ≠ G)
  (h2 : 0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < D ∧ 0 < E ∧ 0 < F ∧ 0 < G)
  (h3 : A + 11 * A = 2 * (10 * B + A))
  (h4 : A + 10 * C + D = 2 * (10 * A + B))
  (h5 : 10 * C + D = 20 * A)
  (h6 : 20 + 62 = 2 * (10 * C + A)) -- for sequences formed by AB, CA, EF
  (h7 : 21 + 63 = 2 * (10 * G + A)) -- for sequences formed by BA, CA, GA
  : ∃ (C D E F G : ℕ), C * 10000 + D * 1000 + E * 100 + F * 10 + G = 40637 := 
sorry

end NUMINAMATH_GPT_five_digit_number_is_40637_l337_33779


namespace NUMINAMATH_GPT_s_plough_time_l337_33702

theorem s_plough_time (r_s_combined_time : ℝ) (r_time : ℝ) (t_time : ℝ) (s_time : ℝ) :
  r_s_combined_time = 10 → r_time = 15 → t_time = 20 → s_time = 30 :=
by
  sorry

end NUMINAMATH_GPT_s_plough_time_l337_33702


namespace NUMINAMATH_GPT_jack_keeps_10800_pounds_l337_33719

def number_of_months_in_a_quarter := 12 / 4
def monthly_hunting_trips := 6
def total_hunting_trips := monthly_hunting_trips * number_of_months_in_a_quarter
def deers_per_trip := 2
def total_deers := total_hunting_trips * deers_per_trip
def weight_per_deer := 600
def total_weight := total_deers * weight_per_deer
def kept_weight_fraction := 1 / 2
def kept_weight := total_weight * kept_weight_fraction

theorem jack_keeps_10800_pounds :
  kept_weight = 10800 :=
by
  -- This is a stub for the automated proof
  sorry

end NUMINAMATH_GPT_jack_keeps_10800_pounds_l337_33719


namespace NUMINAMATH_GPT_loan_amount_l337_33783

theorem loan_amount
  (P : ℝ)
  (SI : ℝ := 704)
  (R : ℝ := 8)
  (T : ℝ := 8)
  (h : SI = (P * R * T) / 100) : P = 1100 :=
by
  sorry

end NUMINAMATH_GPT_loan_amount_l337_33783


namespace NUMINAMATH_GPT_log_sum_l337_33747

variable (m a b : ℝ)
variable (m_pos : 0 < m)
variable (m_ne_one : m ≠ 1)
variable (h1 : m^2 = a)
variable (h2 : m^3 = b)

theorem log_sum (m_pos : 0 < m) (m_ne_one : m ≠ 1) (h1 : m^2 = a) (h2 : m^3 = b) :
  2 * Real.log (a) / Real.log (m) + Real.log (b) / Real.log (m) = 7 := 
sorry

end NUMINAMATH_GPT_log_sum_l337_33747


namespace NUMINAMATH_GPT_exists_x_for_ax2_plus_2x_plus_a_lt_0_l337_33754

theorem exists_x_for_ax2_plus_2x_plus_a_lt_0 (a : ℝ) : (∃ x : ℝ, a * x^2 + 2 * x + a < 0) ↔ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_exists_x_for_ax2_plus_2x_plus_a_lt_0_l337_33754


namespace NUMINAMATH_GPT_cookies_remaining_in_jar_l337_33786

-- Definition of the conditions
variable (initial_cookies : Nat)

def cookies_taken_by_Lou_Senior := 3 + 1
def cookies_taken_by_Louie_Junior := 7
def total_cookies_taken := cookies_taken_by_Lou_Senior + cookies_taken_by_Louie_Junior

-- Debra's assumption and the proof goal
theorem cookies_remaining_in_jar (half_cookies_removed : total_cookies_taken = initial_cookies / 2) : 
  initial_cookies - total_cookies_taken = 11 := by
  sorry

end NUMINAMATH_GPT_cookies_remaining_in_jar_l337_33786


namespace NUMINAMATH_GPT_cumulative_distribution_X_maximized_expected_score_l337_33732

noncomputable def distribution_X (p_A : ℝ) (p_B : ℝ) : (ℝ × ℝ × ℝ) :=
(1 - p_A, p_A * (1 - p_B), p_A * p_B)

def expected_score (p_A : ℝ) (p_B : ℝ) (s_A : ℝ) (s_B : ℝ) : ℝ :=
0 * (1 - p_A) + s_A * (p_A * (1 - p_B)) + (s_A + s_B) * (p_A * p_B)

theorem cumulative_distribution_X :
  distribution_X 0.8 0.6 = (0.2, 0.32, 0.48) :=
sorry

theorem maximized_expected_score :
  expected_score 0.8 0.6 20 80 < expected_score 0.6 0.8 80 20 :=
sorry

end NUMINAMATH_GPT_cumulative_distribution_X_maximized_expected_score_l337_33732


namespace NUMINAMATH_GPT_unique_m_value_l337_33749

theorem unique_m_value : ∀ m : ℝ,
  (m ^ 2 - 5 * m + 6 = 0 ∧ m ^ 2 - 3 * m + 2 = 0) →
  (m ^ 2 - 3 * m + 2 = 2 * (m ^ 2 - 5 * m + 6)) →
  ((m ^ 2 - 5 * m + 6) * (m ^ 2 - 3 * m + 2) > 0) →
  m = 2 :=
by
  sorry

end NUMINAMATH_GPT_unique_m_value_l337_33749


namespace NUMINAMATH_GPT_calculate_savings_l337_33739

noncomputable def monthly_salary : ℕ := 10000
noncomputable def spent_on_food (S : ℕ) : ℕ := (40 * S) / 100
noncomputable def spent_on_rent (S : ℕ) : ℕ := (20 * S) / 100
noncomputable def spent_on_entertainment (S : ℕ) : ℕ := (10 * S) / 100
noncomputable def spent_on_conveyance (S : ℕ) : ℕ := (10 * S) / 100
noncomputable def total_spent (S : ℕ) : ℕ := spent_on_food S + spent_on_rent S + spent_on_entertainment S + spent_on_conveyance S
noncomputable def amount_saved (S : ℕ) : ℕ := S - total_spent S

theorem calculate_savings : amount_saved monthly_salary = 2000 :=
by
  sorry

end NUMINAMATH_GPT_calculate_savings_l337_33739


namespace NUMINAMATH_GPT_domain_of_log_base_5_range_of_3_pow_neg_l337_33736

theorem domain_of_log_base_5 (x : ℝ) : (1 - x > 0) -> (x < 1) :=
sorry

theorem range_of_3_pow_neg (y : ℝ) : (∃ x : ℝ, y = 3 ^ (-x)) -> (y > 0) :=
sorry

end NUMINAMATH_GPT_domain_of_log_base_5_range_of_3_pow_neg_l337_33736


namespace NUMINAMATH_GPT_JimSiblings_l337_33793

-- Define the students and their characteristics.
structure Student :=
  (name : String)
  (eyeColor : String)
  (hairColor : String)
  (wearsGlasses : Bool)

def Benjamin : Student := ⟨"Benjamin", "Blue", "Blond", true⟩
def Jim : Student := ⟨"Jim", "Brown", "Blond", false⟩
def Nadeen : Student := ⟨"Nadeen", "Brown", "Black", true⟩
def Austin : Student := ⟨"Austin", "Blue", "Black", false⟩
def Tevyn : Student := ⟨"Tevyn", "Blue", "Blond", true⟩
def Sue : Student := ⟨"Sue", "Brown", "Blond", false⟩

-- Define the condition that students from the same family share at least one characteristic.
def shareCharacteristic (s1 s2 : Student) : Bool :=
  (s1.eyeColor = s2.eyeColor) ∨
  (s1.hairColor = s2.hairColor) ∨
  (s1.wearsGlasses = s2.wearsGlasses)

-- Define what it means to be siblings of a student.
def areSiblings (s1 s2 s3 : Student) : Bool :=
  shareCharacteristic s1 s2 ∧
  shareCharacteristic s1 s3 ∧
  shareCharacteristic s2 s3

-- The theorem we are trying to prove.
theorem JimSiblings : areSiblings Jim Sue Benjamin = true := 
  by sorry

end NUMINAMATH_GPT_JimSiblings_l337_33793


namespace NUMINAMATH_GPT_proof_problem_l337_33774

theorem proof_problem
  (n : ℕ)
  (h : n = 16^3018) :
  n / 8 = 2^9032 := by
  sorry

end NUMINAMATH_GPT_proof_problem_l337_33774


namespace NUMINAMATH_GPT_midpoint_product_l337_33764

-- Defining the endpoints of the line segment
def x1 : ℤ := 4
def y1 : ℤ := 7
def x2 : ℤ := -8
def y2 : ℤ := 9

-- Proof goal: show that the product of the coordinates of the midpoint is -16
theorem midpoint_product : ((x1 + x2) / 2) * ((y1 + y2) / 2) = -16 := 
by sorry

end NUMINAMATH_GPT_midpoint_product_l337_33764


namespace NUMINAMATH_GPT_find_point_N_l337_33756

theorem find_point_N 
  (M N : ℝ × ℝ) 
  (MN_length : Real.sqrt (((N.1 - M.1) ^ 2) + ((N.2 - M.2) ^ 2)) = 4)
  (MN_parallel_y_axis : N.1 = M.1)
  (M_coord : M = (-1, 2)) 
  : (N = (-1, 6)) ∨ (N = (-1, -2)) :=
sorry

end NUMINAMATH_GPT_find_point_N_l337_33756


namespace NUMINAMATH_GPT_blocks_added_l337_33792

theorem blocks_added (original_blocks new_blocks added_blocks : ℕ) 
  (h1 : original_blocks = 35) 
  (h2 : new_blocks = 65) 
  (h3 : new_blocks = original_blocks + added_blocks) : 
  added_blocks = 30 :=
by
  -- We use the given conditions to prove the statement
  sorry

end NUMINAMATH_GPT_blocks_added_l337_33792


namespace NUMINAMATH_GPT_value_of_m_squared_plus_2m_minus_3_l337_33795

theorem value_of_m_squared_plus_2m_minus_3 (m : ℤ) : 
  (∀ x : ℤ, 4 * (x - 1) - m * x + 6 = 8 → x = 3) →
  m^2 + 2 * m - 3 = 5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_squared_plus_2m_minus_3_l337_33795


namespace NUMINAMATH_GPT_polar_to_cartesian_l337_33753

theorem polar_to_cartesian (ρ θ : ℝ) : (ρ * Real.cos θ = 0) → ρ = 0 ∨ θ = π/2 :=
by 
  sorry

end NUMINAMATH_GPT_polar_to_cartesian_l337_33753


namespace NUMINAMATH_GPT_union_prob_inconsistency_l337_33767

noncomputable def p_a : ℚ := 2/15
noncomputable def p_b : ℚ := 4/15
noncomputable def p_b_given_a : ℚ := 3

theorem union_prob_inconsistency : p_a + p_b - p_b_given_a * p_a = 0 → false := by
  sorry

end NUMINAMATH_GPT_union_prob_inconsistency_l337_33767


namespace NUMINAMATH_GPT_harry_friday_speed_l337_33775

theorem harry_friday_speed :
  let monday_speed := 10
  let tuesday_thursday_speed := monday_speed + monday_speed * (50 / 100)
  let friday_speed := tuesday_thursday_speed + tuesday_thursday_speed * (60 / 100)
  friday_speed = 24 :=
by
  sorry

end NUMINAMATH_GPT_harry_friday_speed_l337_33775


namespace NUMINAMATH_GPT_composite_expression_l337_33710

theorem composite_expression (n : ℕ) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n^3 + 6 * n^2 + 12 * n + 7 = a * b :=
by
  sorry

end NUMINAMATH_GPT_composite_expression_l337_33710


namespace NUMINAMATH_GPT_total_spectators_after_halftime_l337_33790

theorem total_spectators_after_halftime
  (initial_boys : ℕ := 300)
  (initial_girls : ℕ := 400)
  (initial_adults : ℕ := 300)
  (total_people : ℕ := 1000)
  (quarter_boys_leave_fraction : ℚ := 1 / 4)
  (quarter_girls_leave_fraction : ℚ := 1 / 8)
  (quarter_adults_leave_fraction : ℚ := 1 / 5)
  (halftime_new_boys : ℕ := 50)
  (halftime_new_girls : ℕ := 90)
  (halftime_adults_leave_fraction : ℚ := 3 / 100) :
  let boys_after_first_quarter := initial_boys - initial_boys * quarter_boys_leave_fraction
  let girls_after_first_quarter := initial_girls - initial_girls * quarter_girls_leave_fraction
  let adults_after_first_quarter := initial_adults - initial_adults * quarter_adults_leave_fraction
  let boys_after_halftime := boys_after_first_quarter + halftime_new_boys
  let girls_after_halftime := girls_after_first_quarter + halftime_new_girls
  let adults_after_halftime := adults_after_first_quarter * (1 - halftime_adults_leave_fraction)
  boys_after_halftime + girls_after_halftime + adults_after_halftime = 948 :=
by sorry

end NUMINAMATH_GPT_total_spectators_after_halftime_l337_33790
