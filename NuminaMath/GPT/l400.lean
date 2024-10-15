import Mathlib

namespace NUMINAMATH_GPT_sum_A_B_equals_1_l400_40049

-- Definitions for the digits and the properties defined in conditions
variables (A B C D : ℕ)
variable (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
variable (h_digit_bounds : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10)
noncomputable def ABCD := 1000 * A + 100 * B + 10 * C + D
axiom h_mult : ABCD * 2 = ABCD * 10

theorem sum_A_B_equals_1 : A + B = 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_A_B_equals_1_l400_40049


namespace NUMINAMATH_GPT_ratio_of_distances_l400_40009

theorem ratio_of_distances (d_5 d_4 : ℝ) (h1 : d_5 + d_4 ≤ 26.67) (h2 : d_5 / 5 + d_4 / 4 = 6) : 
  d_5 / (d_5 + d_4) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_ratio_of_distances_l400_40009


namespace NUMINAMATH_GPT_consistent_values_l400_40007

theorem consistent_values (a x: ℝ) :
    (12 * x^2 + 48 * x - a + 36 = 0) ∧ ((a + 60) * x - 3 * (a - 20) = 0) ↔
    ((a = -12 ∧ x = -2) ∨ (a = 0 ∧ x = -1) ∨ (a = 180 ∧ x = 2)) := 
by
  -- proof steps should be filled here
  sorry

end NUMINAMATH_GPT_consistent_values_l400_40007


namespace NUMINAMATH_GPT_total_distance_of_relay_race_l400_40016

theorem total_distance_of_relay_race 
    (fraction_siwon : ℝ := 3/10) 
    (fraction_dawon : ℝ := 4/10) 
    (distance_together : ℝ := 140) :
    (fraction_siwon + fraction_dawon) * 200 = distance_together :=
by
    sorry

end NUMINAMATH_GPT_total_distance_of_relay_race_l400_40016


namespace NUMINAMATH_GPT_trajectory_of_M_l400_40079

theorem trajectory_of_M {x y x₀ y₀ : ℝ} (P_on_parabola : x₀^2 = 2 * y₀)
(line_PQ_perpendicular : ∀ Q : ℝ, true)
(vector_PM_PQ_relation : x₀ = x ∧ y₀ = 2 * y) :
  x^2 = 4 * y := by
  sorry

end NUMINAMATH_GPT_trajectory_of_M_l400_40079


namespace NUMINAMATH_GPT_sum_of_three_numbers_l400_40012

theorem sum_of_three_numbers (x y z : ℕ) (h1 : x ≤ y) (h2 : y ≤ z) (h3 : y = 7) 
    (h4 : (x + y + z) / 3 = x + 12) (h5 : (x + y + z) / 3 = z - 18) : 
    x + y + z = 39 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l400_40012


namespace NUMINAMATH_GPT_investment_time_p_l400_40040

theorem investment_time_p (p_investment q_investment p_profit q_profit : ℝ) (p_invest_time : ℝ) (investment_ratio_pq : p_investment / q_investment = 7 / 5.00001) (profit_ratio_pq : p_profit / q_profit = 7.00001 / 10) (q_invest_time : q_invest_time = 9.999965714374696) : p_invest_time = 50 :=
sorry

end NUMINAMATH_GPT_investment_time_p_l400_40040


namespace NUMINAMATH_GPT_integral_sin_pi_half_to_three_pi_half_l400_40051

theorem integral_sin_pi_half_to_three_pi_half :
  ∫ x in (Set.Icc (Real.pi / 2) (3 * Real.pi / 2)), Real.sin x = 0 :=
by
  sorry

end NUMINAMATH_GPT_integral_sin_pi_half_to_three_pi_half_l400_40051


namespace NUMINAMATH_GPT_mail_distribution_l400_40038

-- Define the number of houses
def num_houses : ℕ := 10

-- Define the pieces of junk mail per house
def mail_per_house : ℕ := 35

-- Define total pieces of junk mail delivered
def total_pieces_of_junk_mail : ℕ := num_houses * mail_per_house

-- Main theorem statement
theorem mail_distribution : total_pieces_of_junk_mail = 350 := by
  sorry

end NUMINAMATH_GPT_mail_distribution_l400_40038


namespace NUMINAMATH_GPT_Mobius_speed_without_load_l400_40094

theorem Mobius_speed_without_load
  (v : ℝ)
  (distance : ℝ := 143)
  (load_speed : ℝ := 11)
  (rest_time : ℝ := 2)
  (total_time : ℝ := 26) :
  (total_time - rest_time = (distance / load_speed + distance / v)) → v = 13 :=
by
  intros h
  exact sorry

end NUMINAMATH_GPT_Mobius_speed_without_load_l400_40094


namespace NUMINAMATH_GPT_slower_time_l400_40071

-- Definitions for the problem conditions
def num_stories : ℕ := 50
def lola_time_per_story : ℕ := 12
def tara_time_per_story : ℕ := 10
def tara_stop_time : ℕ := 4
def tara_num_stops : ℕ := num_stories - 2 -- Stops on each floor except the first and last

-- Calculations based on the conditions
def lola_total_time : ℕ := num_stories * lola_time_per_story
def tara_total_time : ℕ := num_stories * tara_time_per_story + tara_num_stops * tara_stop_time

-- Target statement to be proven
theorem slower_time : tara_total_time = 692 := by
  sorry  -- Proof goes here (excluded as per instructions)

end NUMINAMATH_GPT_slower_time_l400_40071


namespace NUMINAMATH_GPT_ticket_cost_l400_40085

theorem ticket_cost (a : ℝ) (h1 : (6 * a + 5 * (2 / 3 * a) = 47.25)) :
  10 * a + 8 * (2 / 3 * a) = 77.625 :=
by
  sorry

end NUMINAMATH_GPT_ticket_cost_l400_40085


namespace NUMINAMATH_GPT_degrees_to_radians_neg_210_l400_40060

theorem degrees_to_radians_neg_210 :
  -210 * (Real.pi / 180) = - (7 / 6) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_degrees_to_radians_neg_210_l400_40060


namespace NUMINAMATH_GPT_total_cups_of_liquid_drunk_l400_40074

-- Definitions for the problem conditions
def elijah_pints : ℝ := 8.5
def emilio_pints : ℝ := 9.5
def cups_per_pint : ℝ := 2
def elijah_cups : ℝ := elijah_pints * cups_per_pint
def emilio_cups : ℝ := emilio_pints * cups_per_pint
def total_cups : ℝ := elijah_cups + emilio_cups

-- Theorem to prove the required equality
theorem total_cups_of_liquid_drunk : total_cups = 36 :=
by
  sorry

end NUMINAMATH_GPT_total_cups_of_liquid_drunk_l400_40074


namespace NUMINAMATH_GPT_find_pairs_l400_40063

theorem find_pairs (m n: ℕ) (h: m > 0 ∧ n > 0 ∧ m + n - (3 * m * n) / (m + n) = 2011 / 3) : (m = 1144 ∧ n = 377) ∨ (m = 377 ∧ n = 1144) :=
by sorry

end NUMINAMATH_GPT_find_pairs_l400_40063


namespace NUMINAMATH_GPT_find_values_of_symbols_l400_40036

theorem find_values_of_symbols (a b : ℕ) (h1 : a + b + b = 55) (h2 : a + b = 40) : b = 15 ∧ a = 25 :=
  by
    sorry

end NUMINAMATH_GPT_find_values_of_symbols_l400_40036


namespace NUMINAMATH_GPT_opposite_of_neg5_l400_40015

-- Define the concept of the opposite of a number
def opposite (x : Int) : Int :=
  -x

-- The proof problem: Prove that the opposite of -5 is 5
theorem opposite_of_neg5 : opposite (-5) = 5 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg5_l400_40015


namespace NUMINAMATH_GPT_area_of_inner_square_l400_40054

theorem area_of_inner_square (s₁ s₂ : ℝ) (side_length_WXYZ : ℝ) (WI : ℝ) (area_IJKL : ℝ) 
  (h1 : s₁ = 10) 
  (h2 : s₂ = 10 - 2 * Real.sqrt 2)
  (h3 : side_length_WXYZ = 10)
  (h4 : WI = 2)
  (h5 : area_IJKL = (s₂)^2): 
  area_IJKL = 102 - 20 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_inner_square_l400_40054


namespace NUMINAMATH_GPT_alice_twice_bob_in_some_years_l400_40047

def alice_age (B : ℕ) : ℕ := B + 10
def future_age_condition (A : ℕ) : Prop := A + 5 = 19
def twice_as_old_condition (A B x : ℕ) : Prop := A + x = 2 * (B + x)

theorem alice_twice_bob_in_some_years :
  ∃ x, ∀ A B,
  alice_age B = A →
  future_age_condition A →
  twice_as_old_condition A B x := by
  sorry

end NUMINAMATH_GPT_alice_twice_bob_in_some_years_l400_40047


namespace NUMINAMATH_GPT_meals_neither_vegan_kosher_nor_gluten_free_l400_40024

def total_clients : ℕ := 50
def n_vegan : ℕ := 10
def n_kosher : ℕ := 12
def n_gluten_free : ℕ := 6
def n_both_vegan_kosher : ℕ := 3
def n_both_vegan_gluten_free : ℕ := 4
def n_both_kosher_gluten_free : ℕ := 2
def n_all_three : ℕ := 1

/-- The number of clients who need a meal that is neither vegan, kosher, nor gluten-free. --/
theorem meals_neither_vegan_kosher_nor_gluten_free :
  total_clients - (n_vegan + n_kosher + n_gluten_free - n_both_vegan_kosher - n_both_vegan_gluten_free - n_both_kosher_gluten_free + n_all_three) = 30 :=
by
  sorry

end NUMINAMATH_GPT_meals_neither_vegan_kosher_nor_gluten_free_l400_40024


namespace NUMINAMATH_GPT_simplify_composite_product_fraction_l400_40062

def first_four_composite_product : ℤ := 4 * 6 * 8 * 9
def next_four_composite_product : ℤ := 10 * 12 * 14 * 15
def expected_fraction_num : ℤ := 12
def expected_fraction_den : ℤ := 175

theorem simplify_composite_product_fraction :
  (first_four_composite_product / next_four_composite_product : ℚ) = (expected_fraction_num / expected_fraction_den) :=
by
  rw [first_four_composite_product, next_four_composite_product]
  norm_num
  sorry

end NUMINAMATH_GPT_simplify_composite_product_fraction_l400_40062


namespace NUMINAMATH_GPT_rhombus_diagonal_length_l400_40055

theorem rhombus_diagonal_length (d1 : ℝ) : 
  (d1 * 12) / 2 = 60 → d1 = 10 := 
by 
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_length_l400_40055


namespace NUMINAMATH_GPT_find_point_W_coordinates_l400_40001

theorem find_point_W_coordinates 
(O U S V : ℝ × ℝ)
(hO : O = (0, 0))
(hU : U = (3, 3))
(hS : S = (3, 0))
(hV : V = (0, 3))
(hSquare : (O.1 - U.1)^2 + (O.2 - U.2)^2 = 18)
(hArea_Square : 3 * 3 = 9) :
  ∃ W : ℝ × ℝ, W = (3, 9) ∧ 1 / 2 * (abs (S.1 - V.1) * abs (W.2 - S.2)) = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_point_W_coordinates_l400_40001


namespace NUMINAMATH_GPT_sum_5n_is_630_l400_40091

variable (n : ℕ)

def sum_first_k (k : ℕ) : ℕ :=
  k * (k + 1) / 2

theorem sum_5n_is_630 (h : sum_first_k (3 * n) = sum_first_k n + 210) : sum_first_k (5 * n) = 630 := sorry

end NUMINAMATH_GPT_sum_5n_is_630_l400_40091


namespace NUMINAMATH_GPT_triangle_perimeter_l400_40033

-- Define the conditions of the problem
def a := 4
def b := 8
def quadratic_eq (x : ℝ) : Prop := x^2 - 14 * x + 40 = 0

-- Define the perimeter calculation, ensuring triangle inequality and correct side length
def valid_triangle (x : ℝ) : Prop :=
  x ≠ a ∧ x ≠ b ∧ quadratic_eq x ∧ (a + b > x) ∧ (a + x > b) ∧ (b + x > a)

-- Define the problem statement as a theorem
theorem triangle_perimeter : ∃ x : ℝ, valid_triangle x ∧ (a + b + x = 22) :=
by {
  -- Placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_triangle_perimeter_l400_40033


namespace NUMINAMATH_GPT_maximum_alpha_l400_40067

noncomputable def is_in_F (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, f (3 * x) ≥ f (f (2 * x)) + x

theorem maximum_alpha :
  (∀ f : ℝ → ℝ, is_in_F f → ∀ x > 0, f x ≥ (1 / 2) * x) := 
by
  sorry

end NUMINAMATH_GPT_maximum_alpha_l400_40067


namespace NUMINAMATH_GPT_merchants_tea_cups_l400_40043

theorem merchants_tea_cups (a b c : ℕ) 
  (h1 : a + b = 11)
  (h2 : b + c = 15)
  (h3 : a + c = 14) : 
  a + b + c = 20 :=
by
  sorry

end NUMINAMATH_GPT_merchants_tea_cups_l400_40043


namespace NUMINAMATH_GPT_solve_inequality_l400_40013

def within_interval (x : ℝ) : Prop :=
  x < 2 ∧ x > -5

theorem solve_inequality (x : ℝ) : (x^2 + 3 * x < 10) ↔ within_interval x :=
sorry

end NUMINAMATH_GPT_solve_inequality_l400_40013


namespace NUMINAMATH_GPT_initial_time_is_11_55_l400_40065

-- Definitions for the conditions
variable (X : ℕ) (Y : ℕ)

def initial_time_shown_by_clock (X Y : ℕ) : Prop :=
  (5 * (18 - X) = 35) ∧ (Y = 60 - 5)

theorem initial_time_is_11_55 (h : initial_time_shown_by_clock X Y) : (X = 11) ∧ (Y = 55) :=
sorry

end NUMINAMATH_GPT_initial_time_is_11_55_l400_40065


namespace NUMINAMATH_GPT_books_bought_at_bookstore_l400_40000

-- Define the initial count of books
def initial_books : ℕ := 72

-- Define the number of books received each month from the book club
def books_from_club (months : ℕ) : ℕ := months

-- Number of books received as gifts
def books_from_daughter : ℕ := 1
def books_from_mother : ℕ := 4

-- Number of books bought
def books_from_yard_sales : ℕ := 2

-- Number of books donated and sold
def books_donated : ℕ := 12
def books_sold : ℕ := 3

-- Final total count of books
def final_books : ℕ := 81

-- Calculate the number of books acquired and then removed, and prove 
-- the number of books bought at the bookstore halfway through the year
theorem books_bought_at_bookstore (months : ℕ) (b : ℕ) :
  initial_books + books_from_club months + books_from_daughter + books_from_mother + books_from_yard_sales + b - books_donated - books_sold = final_books → b = 5 :=
by sorry

end NUMINAMATH_GPT_books_bought_at_bookstore_l400_40000


namespace NUMINAMATH_GPT_expression_value_l400_40010

theorem expression_value : (28 * 2 + (48 / 6) ^ 2 - 5) * (69 / 3) + 24 * (3 ^ 2 - 2) = 2813 := by
  sorry

end NUMINAMATH_GPT_expression_value_l400_40010


namespace NUMINAMATH_GPT_pre_bought_ticket_price_l400_40034

variable (P : ℕ)

theorem pre_bought_ticket_price :
  (20 * P = 6000 - 2900) → P = 155 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_pre_bought_ticket_price_l400_40034


namespace NUMINAMATH_GPT_students_taking_art_l400_40004

theorem students_taking_art :
  ∀ (total_students music_students both_music_art neither_music_art : ℕ),
  total_students = 500 →
  music_students = 30 →
  both_music_art = 10 →
  neither_music_art = 470 →
  (total_students - neither_music_art) - (music_students - both_music_art) - both_music_art = 10 :=
by
  intros total_students music_students both_music_art neither_music_art h_total h_music h_both h_neither
  sorry

end NUMINAMATH_GPT_students_taking_art_l400_40004


namespace NUMINAMATH_GPT_greatest_sum_consecutive_lt_400_l400_40018

noncomputable def greatest_sum_of_consecutive_integers (n : ℤ) : ℤ :=
if n * (n + 1) < 400 then n + (n + 1) else 0

theorem greatest_sum_consecutive_lt_400 : ∃ n : ℤ, n * (n + 1) < 400 ∧ greatest_sum_of_consecutive_integers n = 39 :=
by
  sorry

end NUMINAMATH_GPT_greatest_sum_consecutive_lt_400_l400_40018


namespace NUMINAMATH_GPT_cos_pi_minus_2alpha_l400_40093

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.sin α = 2 / 3) : Real.cos (Real.pi - 2 * α) = -1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cos_pi_minus_2alpha_l400_40093


namespace NUMINAMATH_GPT_eval_frac_equal_two_l400_40039

noncomputable def eval_frac (a b : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 - a*b + b^2 = 0) : ℂ :=
  (a^8 + b^8) / (a^2 + b^2)^4

theorem eval_frac_equal_two (a b : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 - a*b + b^2 = 0) : eval_frac a b h1 h2 h3 = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_eval_frac_equal_two_l400_40039


namespace NUMINAMATH_GPT_students_taking_german_l400_40028

theorem students_taking_german
  (total_students : ℕ)
  (french_students : ℕ)
  (both_courses_students : ℕ)
  (no_course_students : ℕ)
  (h1 : total_students = 87)
  (h2 : french_students = 41)
  (h3 : both_courses_students = 9)
  (h4 : no_course_students = 33)
  : ∃ german_students : ℕ, german_students = 22 := 
by
  -- proof can be filled in here
  sorry

end NUMINAMATH_GPT_students_taking_german_l400_40028


namespace NUMINAMATH_GPT_even_function_f_D_l400_40003

noncomputable def f_A (x : ℝ) : ℝ := 2 * |x| - 1
def D_f_A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

def f_B (x : ℕ) : ℕ := x^2 + x

def f_C (x : ℝ) : ℝ := x ^ 3

noncomputable def f_D (x : ℝ) : ℝ := x^2
def D_f_D := {x : ℝ | (-1 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 1)}

theorem even_function_f_D : 
  ∀ x ∈ D_f_D, f_D (-x) = f_D (x) :=
sorry

end NUMINAMATH_GPT_even_function_f_D_l400_40003


namespace NUMINAMATH_GPT_total_sleep_correct_l400_40019

namespace SleepProblem

def recommended_sleep_per_day : ℝ := 8
def sleep_days_part1 : ℕ := 2
def sleep_hours_part1 : ℝ := 3
def days_in_week : ℕ := 7
def remaining_days := days_in_week - sleep_days_part1
def percentage_sleep : ℝ := 0.6
def sleep_per_remaining_day := recommended_sleep_per_day * percentage_sleep

theorem total_sleep_correct (h1 : 2 * sleep_hours_part1 = 6)
                            (h2 : remaining_days = 5)
                            (h3 : sleep_per_remaining_day = 4.8)
                            (h4 : remaining_days * sleep_per_remaining_day = 24) :
  2 * sleep_hours_part1 + remaining_days * sleep_per_remaining_day = 30 := by
  sorry

end SleepProblem

end NUMINAMATH_GPT_total_sleep_correct_l400_40019


namespace NUMINAMATH_GPT_cody_initial_money_l400_40050

variable (x : ℤ)

theorem cody_initial_money :
  (x + 9 - 19 = 35) → (x = 45) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cody_initial_money_l400_40050


namespace NUMINAMATH_GPT_greatest_value_of_x_l400_40035

theorem greatest_value_of_x (x : ℕ) (h : Nat.lcm (Nat.lcm x 15) 21 = 105) : x = 105 :=
sorry

end NUMINAMATH_GPT_greatest_value_of_x_l400_40035


namespace NUMINAMATH_GPT_trailing_zeros_a6_l400_40002

theorem trailing_zeros_a6:
  (∃ a : ℕ+ → ℚ, 
    a 1 = 3 / 2 ∧ 
    (∀ n : ℕ+, a (n + 1) = (1 / 2) * (a n + (1 / a n))) ∧
    (∃ k, 10^k ≤ a 6 ∧ a 6 < 10^(k + 1))) →
  (∃ m, m = 22) :=
sorry

end NUMINAMATH_GPT_trailing_zeros_a6_l400_40002


namespace NUMINAMATH_GPT_tan_45_degree_is_one_l400_40026

theorem tan_45_degree_is_one : Real.tan (Real.pi / 4) = 1 := sorry

end NUMINAMATH_GPT_tan_45_degree_is_one_l400_40026


namespace NUMINAMATH_GPT_range_of_a_l400_40008

def quadratic_inequality (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ + 1 ≤ 0

theorem range_of_a :
  ¬ quadratic_inequality a ↔ -1 < a ∧ a < 3 :=
  by
  sorry

end NUMINAMATH_GPT_range_of_a_l400_40008


namespace NUMINAMATH_GPT_no_common_points_l400_40097

theorem no_common_points (x0 y0 : ℝ) (h : x0^2 < 4 * y0) :
  ∀ (x y : ℝ), (x^2 = 4 * y) → (x0 * x = 2 * (y + y0)) →
  false := 
by
  sorry

end NUMINAMATH_GPT_no_common_points_l400_40097


namespace NUMINAMATH_GPT_hyperbola_properties_l400_40011

theorem hyperbola_properties (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (c := Real.sqrt (a^2 + b^2))
  (F2 := (c, 0)) (P : ℝ × ℝ)
  (h_perpendicular : ∃ (x y : ℝ), P = (x, y) ∧ y = -a/b * (x - c))
  (h_distance : Real.sqrt ((P.1 - c)^2 + P.2^2) = 2)
  (h_slope : P.2 / (P.1 - c) = -1/2) :
  
  b = 2 ∧
  (∀ x y, x^2 - y^2 / 4 = 1 ↔ x^2 - y^2 / b^2 = 1) ∧
  P = (Real.sqrt (5) / 5, 2 * Real.sqrt (5) / 5) :=
sorry

end NUMINAMATH_GPT_hyperbola_properties_l400_40011


namespace NUMINAMATH_GPT_rectangular_plot_breadth_l400_40052

theorem rectangular_plot_breadth (b : ℝ) 
    (h1 : ∃ l : ℝ, l = 3 * b)
    (h2 : 432 = 3 * b * b) : b = 12 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_plot_breadth_l400_40052


namespace NUMINAMATH_GPT_drainage_capacity_per_day_l400_40068

theorem drainage_capacity_per_day
  (capacity : ℝ)
  (rain_1 : ℝ)
  (rain_2 : ℝ)
  (rain_3 : ℝ)
  (rain_4_min : ℝ)
  (total_days : ℕ) 
  (days_to_drain : ℕ)
  (feet_to_inches : ℝ := 12)
  (required_rain_capacity : ℝ) 
  (drain_capacity_per_day : ℝ)

  (h1: capacity = 6 * feet_to_inches)
  (h2: rain_1 = 10)
  (h3: rain_2 = 2 * rain_1)
  (h4: rain_3 = 1.5 * rain_2)
  (h5: rain_4_min = 21)
  (h6: total_days = 4)
  (h7: days_to_drain = 3)
  (h8: required_rain_capacity = capacity - (rain_1 + rain_2 + rain_3))

  : drain_capacity_per_day = (rain_1 + rain_2 + rain_3 - required_rain_capacity + rain_4_min) / days_to_drain :=
sorry

end NUMINAMATH_GPT_drainage_capacity_per_day_l400_40068


namespace NUMINAMATH_GPT_small_ball_rubber_bands_l400_40082

theorem small_ball_rubber_bands (S : ℕ) 
    (large_ball : ℕ := 300) 
    (initial_rubber_bands : ℕ := 5000) 
    (small_balls : ℕ := 22) 
    (large_balls : ℕ := 13) :
  (small_balls * S + large_balls * large_ball = initial_rubber_bands) → S = 50 := by
    sorry

end NUMINAMATH_GPT_small_ball_rubber_bands_l400_40082


namespace NUMINAMATH_GPT_thirty_percent_less_eq_one_fourth_more_l400_40080

theorem thirty_percent_less_eq_one_fourth_more (x : ℝ) (hx1 : 0.7 * 90 = 63) (hx2 : (5 / 4) * x = 63) : x = 50 :=
sorry

end NUMINAMATH_GPT_thirty_percent_less_eq_one_fourth_more_l400_40080


namespace NUMINAMATH_GPT_sum_of_perimeters_l400_40045

theorem sum_of_perimeters (x y z : ℝ) 
    (h_large_triangle_perimeter : 3 * 20 = 60)
    (h_hexagon_perimeter : 60 - (x + y + z) = 40) :
    3 * (x + y + z) = 60 := by
  sorry

end NUMINAMATH_GPT_sum_of_perimeters_l400_40045


namespace NUMINAMATH_GPT_sequence_third_term_l400_40058

theorem sequence_third_term (n : ℕ) (a : ℕ → ℤ) (h : ∀ n, a n = 3 * n - 5) : a 3 = 4 := by
  sorry

end NUMINAMATH_GPT_sequence_third_term_l400_40058


namespace NUMINAMATH_GPT_x_coordinate_of_first_point_l400_40070

theorem x_coordinate_of_first_point (m n : ℝ) :
  (m = 2 * n + 3) ↔ (∃ (p1 p2 : ℝ × ℝ), p1 = (m, n) ∧ p2 = (m + 2, n + 1) ∧ 
    (p1.1 = 2 * p1.2 + 3) ∧ (p2.1 = 2 * p2.2 + 3)) :=
by
  sorry

end NUMINAMATH_GPT_x_coordinate_of_first_point_l400_40070


namespace NUMINAMATH_GPT_smallest_five_digit_neg_int_congruent_to_one_mod_17_l400_40069

theorem smallest_five_digit_neg_int_congruent_to_one_mod_17 :
  ∃ (x : ℤ), x < -9999 ∧ x % 17 = 1 ∧ x = -10011 := by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_smallest_five_digit_neg_int_congruent_to_one_mod_17_l400_40069


namespace NUMINAMATH_GPT_largest_n_for_factorization_l400_40077

theorem largest_n_for_factorization :
  ∃ (n : ℤ), (∀ (A B : ℤ), AB = 96 → n = 4 * B + A) ∧ (n = 385) := by
  sorry

end NUMINAMATH_GPT_largest_n_for_factorization_l400_40077


namespace NUMINAMATH_GPT_exists_fraction_expression_l400_40053

theorem exists_fraction_expression (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_three : p > 3) :
  ∃ (m : ℕ) (h₀ : 3 ≤ m) (h₁ : m ≤ p - 2) (x y : ℕ), (m : ℚ) / (p^2 : ℚ) = 1 / (x : ℚ) + 1 / (y : ℚ) :=
sorry

end NUMINAMATH_GPT_exists_fraction_expression_l400_40053


namespace NUMINAMATH_GPT_cucumber_weight_l400_40099

theorem cucumber_weight (W : ℝ)
  (h1 : W * 0.99 + W * 0.01 = W)
  (h2 : (W * 0.01) / 20 = 1 / 95) :
  W = 100 :=
by
  sorry

end NUMINAMATH_GPT_cucumber_weight_l400_40099


namespace NUMINAMATH_GPT_integer_a_for_factoring_l400_40029

theorem integer_a_for_factoring (a : ℤ) :
  (∃ c d : ℤ, (x - a) * (x - 10) + 1 = (x + c) * (x + d)) → (a = 8 ∨ a = 12) :=
by
  sorry

end NUMINAMATH_GPT_integer_a_for_factoring_l400_40029


namespace NUMINAMATH_GPT_magnet_cost_times_sticker_l400_40073

theorem magnet_cost_times_sticker
  (M S A : ℝ)
  (hM : M = 3)
  (hA : A = 6)
  (hMagnetCost : M = (1/4) * 2 * A) :
  M = 4 * S :=
by
  -- Placeholder, the actual proof would go here
  sorry

end NUMINAMATH_GPT_magnet_cost_times_sticker_l400_40073


namespace NUMINAMATH_GPT_no_solution_exists_l400_40075

   theorem no_solution_exists (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
     ¬ (3 / a + 4 / b = 12 / (a + b)) := 
   sorry
   
end NUMINAMATH_GPT_no_solution_exists_l400_40075


namespace NUMINAMATH_GPT_intersection_A_B_union_A_B_subset_C_B_l400_40056

open Set

noncomputable def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
noncomputable def B : Set ℝ := {x | 2 < x ∧ x < 9}
noncomputable def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem intersection_A_B : A ∩ B = {x | 3 ≤ x ∧ x < 6} :=
by
  sorry

theorem union_A_B : A ∪ B = {x | 2 < x ∧ x < 9} :=
by
  sorry

theorem subset_C_B (a : ℝ) : C a ⊆ B → 2 ≤ a ∧ a ≤ 8 :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_union_A_B_subset_C_B_l400_40056


namespace NUMINAMATH_GPT_perimeter_not_55_l400_40090

def is_valid_perimeter (a b p : ℕ) : Prop :=
  ∃ x : ℕ, a + b > x ∧ a + x > b ∧ b + x > a ∧ p = a + b + x

theorem perimeter_not_55 (a b : ℕ) (h1 : a = 18) (h2 : b = 10) : ¬ is_valid_perimeter a b 55 :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_perimeter_not_55_l400_40090


namespace NUMINAMATH_GPT_distance_between_cityA_and_cityB_l400_40037

noncomputable def distanceBetweenCities (time_to_cityB time_from_cityB saved_time round_trip_speed: ℝ) : ℝ :=
  let total_distance := 90 * (time_to_cityB + saved_time + time_from_cityB + saved_time) / 2
  total_distance / 2

theorem distance_between_cityA_and_cityB 
  (time_to_cityB : ℝ)
  (time_from_cityB : ℝ)
  (saved_time : ℝ)
  (round_trip_speed : ℝ)
  (distance : ℝ)
  (h1 : time_to_cityB = 6)
  (h2 : time_from_cityB = 4.5)
  (h3 : saved_time = 0.5)
  (h4 : round_trip_speed = 90)
  (h5 : distanceBetweenCities time_to_cityB time_from_cityB saved_time round_trip_speed = distance)
: distance = 427.5 := by
  sorry

end NUMINAMATH_GPT_distance_between_cityA_and_cityB_l400_40037


namespace NUMINAMATH_GPT_angle_D_measure_l400_40076

theorem angle_D_measure (A B C D : ℝ) (hA : A = 50) (hB : B = 35) (hC : C = 35) :
  D = 120 :=
  sorry

end NUMINAMATH_GPT_angle_D_measure_l400_40076


namespace NUMINAMATH_GPT_regular_15gon_symmetry_l400_40023

theorem regular_15gon_symmetry :
  ∀ (L R : ℕ),
  (L = 15) →
  (R = 24) →
  L + R = 39 :=
by
  intros L R hL hR
  exact sorry

end NUMINAMATH_GPT_regular_15gon_symmetry_l400_40023


namespace NUMINAMATH_GPT_equation_is_point_l400_40005

-- Definition of the condition in the problem
def equation (x y : ℝ) := x^2 + 36*y^2 - 12*x - 72*y + 36 = 0

-- The theorem stating the equivalence to the point (6, 1)
theorem equation_is_point :
  ∀ (x y : ℝ), equation x y → (x = 6 ∧ y = 1) :=
by
  intros x y h
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_equation_is_point_l400_40005


namespace NUMINAMATH_GPT_probability_major_A_less_than_25_l400_40021

def total_students : ℕ := 100 -- assuming a total of 100 students for simplicity

def male_percent : ℝ := 0.40
def major_A_percent : ℝ := 0.50
def major_B_percent : ℝ := 0.30
def major_C_percent : ℝ := 0.20
def major_A_25_or_older_percent : ℝ := 0.60
def major_A_less_than_25_percent : ℝ := 1 - major_A_25_or_older_percent

theorem probability_major_A_less_than_25 :
  (major_A_percent * major_A_less_than_25_percent) = 0.20 :=
by
  sorry

end NUMINAMATH_GPT_probability_major_A_less_than_25_l400_40021


namespace NUMINAMATH_GPT_sum_of_first_15_terms_of_geometric_sequence_l400_40046

theorem sum_of_first_15_terms_of_geometric_sequence (a r : ℝ) 
  (h₁ : (a * (1 - r^5)) / (1 - r) = 10) 
  (h₂ : (a * (1 - r^10)) / (1 - r) = 50) : 
  (a * (1 - r^15)) / (1 - r) = 210 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_first_15_terms_of_geometric_sequence_l400_40046


namespace NUMINAMATH_GPT_average_speed_l400_40086

theorem average_speed (x : ℝ) (h₀ : x > 0) : 
  let time1 := x / 90
  let time2 := 2 * x / 20
  let total_distance := 3 * x
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 27 := 
by
  sorry

end NUMINAMATH_GPT_average_speed_l400_40086


namespace NUMINAMATH_GPT_math_problem_l400_40031

theorem math_problem 
  (num := 1 * 2 * 3 * 4 * 5 * 6 * 7)
  (den := 1 + 2 + 3 + 4 + 5 + 6 + 7) :
  (num / den) = 180 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l400_40031


namespace NUMINAMATH_GPT_gain_percent_l400_40022

theorem gain_percent (CP SP : ℕ) (h1 : CP = 20) (h2 : SP = 25) : 
  (SP - CP) * 100 / CP = 25 := by
  sorry

end NUMINAMATH_GPT_gain_percent_l400_40022


namespace NUMINAMATH_GPT_solve_for_C_l400_40041

theorem solve_for_C : 
  ∃ C : ℝ, 80 - (5 - (6 + 2 * (7 - C - 5))) = 89 ∧ C = -2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_C_l400_40041


namespace NUMINAMATH_GPT_sqrt_of_sixteen_l400_40096

theorem sqrt_of_sixteen : Real.sqrt 16 = 4 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_of_sixteen_l400_40096


namespace NUMINAMATH_GPT_number_of_boys_in_school_l400_40089

theorem number_of_boys_in_school (B : ℕ) (girls : ℕ) (difference : ℕ) 
    (h1 : girls = 697) (h2 : girls = B + 228) : B = 469 := 
by
  sorry

end NUMINAMATH_GPT_number_of_boys_in_school_l400_40089


namespace NUMINAMATH_GPT_rationalize_denominator_l400_40078

theorem rationalize_denominator :
  let a := (5 : ℝ)^(1/3)
  let b := (4 : ℝ)^(1/3)
  let A := 25
  let B := 20
  let C := 16
  let D := 1
  (1 / (a - b)) = ((A : ℝ)^(1/3) + (B : ℝ)^(1/3) + (C : ℝ)^(1/3)) / D ∧ (A + B + C + D = 62) := by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l400_40078


namespace NUMINAMATH_GPT_part1_is_geometric_part2_is_arithmetic_general_formula_for_a_sum_of_first_n_terms_l400_40092

open Nat

variable {α : Type*}
variables (a : ℕ → ℕ) (S : ℕ → ℕ)

axiom a1 : a 1 = 1
axiom S_def : ∀ (n : ℕ), S (n + 1) = 4 * a n + 2 

def b (n : ℕ) : ℕ := a (n + 1) - 2 * a n

def c (n : ℕ) : ℚ := a n / 2^n

theorem part1_is_geometric :
  ∃ r, ∀ n, b n = r * b (n - 1) := sorry

theorem part2_is_arithmetic :
  ∃ d, ∀ n, c n - c (n - 1) = d := sorry

theorem general_formula_for_a :
  ∀ n, a n = (1 / 4) * (3 * n - 1) * 2 ^ n := sorry

theorem sum_of_first_n_terms :
  ∀ n, S n = (1 / 4) * (8 + (3 * n - 4) * 2 ^ (n + 1)) := sorry

end NUMINAMATH_GPT_part1_is_geometric_part2_is_arithmetic_general_formula_for_a_sum_of_first_n_terms_l400_40092


namespace NUMINAMATH_GPT_coeff_x3y2z5_in_expansion_l400_40066

def binomialCoeff (n k : ℕ) : ℕ := Nat.choose n k

theorem coeff_x3y2z5_in_expansion :
  let x := 1
  let y := 1
  let z := 1
  let x_term := 2 * x
  let y_term := y
  let z_term := z
  let target_term := x_term ^ 3 * y_term ^ 2 * z_term ^ 5
  let coeff := 2^3 * binomialCoeff 10 3 * binomialCoeff 7 2 * binomialCoeff 5 5
  coeff = 20160 :=
by
  sorry

end NUMINAMATH_GPT_coeff_x3y2z5_in_expansion_l400_40066


namespace NUMINAMATH_GPT_store_profit_l400_40027

variable (C : ℝ)  -- Cost price of a turtleneck sweater

noncomputable def initial_marked_price : ℝ := 1.20 * C
noncomputable def new_year_marked_price : ℝ := 1.25 * initial_marked_price C
noncomputable def discount_amount : ℝ := 0.08 * new_year_marked_price C
noncomputable def final_selling_price : ℝ := new_year_marked_price C - discount_amount C
noncomputable def profit : ℝ := final_selling_price C - C

theorem store_profit (C : ℝ) : profit C = 0.38 * C :=
by
  -- The detailed steps are omitted, as required by the instructions.
  sorry

end NUMINAMATH_GPT_store_profit_l400_40027


namespace NUMINAMATH_GPT_magician_assistant_strategy_l400_40081

-- Define the possible states of a coin
inductive CoinState | heads | tails

-- Define the circle of coins
def coinCircle := Fin 11 → CoinState

-- The strategy ensures there are adjacent coins with the same state
theorem magician_assistant_strategy (c : coinCircle) :
  ∃ i : Fin 11, c i = c ((i + 1) % 11) := by sorry

end NUMINAMATH_GPT_magician_assistant_strategy_l400_40081


namespace NUMINAMATH_GPT_largest_divisible_number_l400_40098

theorem largest_divisible_number : ∃ n, n = 9950 ∧ n ≤ 9999 ∧ (∀ m, m ≤ 9999 ∧ m % 50 = 0 → m ≤ n) :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_divisible_number_l400_40098


namespace NUMINAMATH_GPT_point_Q_and_d_l400_40095

theorem point_Q_and_d :
  ∃ (a b c d : ℝ),
    (∀ x y z : ℝ, (x - 2)^2 + (y - 3)^2 + (z + 4)^2 = (x - a)^2 + (y - b)^2 + (z - c)^2) ∧
    (8 * a - 6 * b + 32 * c = d) ∧ a = 6 ∧ b = 0 ∧ c = 12 ∧ d = 151 :=
by
  existsi 6, 0, 12, 151
  sorry

end NUMINAMATH_GPT_point_Q_and_d_l400_40095


namespace NUMINAMATH_GPT_simplify_power_l400_40087

theorem simplify_power (x : ℝ) : (3 * x^4)^4 = 81 * x^16 :=
by sorry

end NUMINAMATH_GPT_simplify_power_l400_40087


namespace NUMINAMATH_GPT_rationalize_denominator_l400_40014

theorem rationalize_denominator (a b c : Real) (h : b*c*c = a) :
  2 / (b + c) = (c*c) / (3 * 2) :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l400_40014


namespace NUMINAMATH_GPT_isabel_spending_ratio_l400_40020

theorem isabel_spending_ratio :
  ∀ (initial_amount toy_cost remaining_amount : ℝ),
    initial_amount = 204 ∧
    toy_cost = initial_amount / 2 ∧
    remaining_amount = 51 →
    ((initial_amount - toy_cost - remaining_amount) / remaining_amount) = 1 / 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_isabel_spending_ratio_l400_40020


namespace NUMINAMATH_GPT_g_is_even_l400_40084

noncomputable def g (x : ℝ) : ℝ := 4 / (3 * x^8 - 7)

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  sorry

end NUMINAMATH_GPT_g_is_even_l400_40084


namespace NUMINAMATH_GPT_solve_for_y_l400_40042

theorem solve_for_y (x y : ℝ) (h1 : x = 8) (h2 : x^(3 * y) = 8) : y = 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l400_40042


namespace NUMINAMATH_GPT_quadratic_function_negative_values_l400_40088

theorem quadratic_function_negative_values (a : ℝ) : 
  (∃ x : ℝ, (x^2 - a*x + 1) < 0) ↔ (a > 2 ∨ a < -2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_negative_values_l400_40088


namespace NUMINAMATH_GPT_average_weight_of_three_l400_40057

theorem average_weight_of_three (Ishmael Ponce Jalen : ℕ) 
  (h1 : Jalen = 160) 
  (h2 : Ponce = Jalen - 10) 
  (h3 : Ishmael = Ponce + 20) : 
  (Ishmael + Ponce + Jalen) / 3 = 160 := 
sorry

end NUMINAMATH_GPT_average_weight_of_three_l400_40057


namespace NUMINAMATH_GPT_john_needs_packs_l400_40030

-- Definitions based on conditions
def utensils_per_pack : Nat := 30
def utensils_types : Nat := 3
def spoons_per_pack : Nat := utensils_per_pack / utensils_types
def spoons_needed : Nat := 50

-- Statement to prove
theorem john_needs_packs : (50 / spoons_per_pack) = 5 :=
by
  -- To complete the proof
  sorry

end NUMINAMATH_GPT_john_needs_packs_l400_40030


namespace NUMINAMATH_GPT_sqrt_nested_l400_40048

theorem sqrt_nested : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_GPT_sqrt_nested_l400_40048


namespace NUMINAMATH_GPT_stickers_per_friend_l400_40006

variable (d: ℕ) (h_d : d > 0)

theorem stickers_per_friend (h : 72 % d = 0) : 72 / d = 72 / d := by
  sorry

end NUMINAMATH_GPT_stickers_per_friend_l400_40006


namespace NUMINAMATH_GPT_superdomino_probability_l400_40017

-- Definitions based on conditions
def is_superdomino (a b : ℕ) : Prop := 0 ≤ a ∧ a ≤ 12 ∧ 0 ≤ b ∧ b ≤ 12
def is_superdouble (a b : ℕ) : Prop := a = b
def total_superdomino_count : ℕ := 13 * 13
def superdouble_count : ℕ := 13

-- Proof statement
theorem superdomino_probability : (superdouble_count : ℚ) / total_superdomino_count = 13 / 169 :=
by
  sorry

end NUMINAMATH_GPT_superdomino_probability_l400_40017


namespace NUMINAMATH_GPT_ratio_of_length_to_width_of_field_is_two_to_one_l400_40072

-- Definitions based on conditions
def lengthOfField : ℕ := 80
def widthOfField (field_area pond_area : ℕ) : ℕ := field_area / lengthOfField
def pondSideLength : ℕ := 8
def pondArea : ℕ := pondSideLength * pondSideLength
def fieldArea : ℕ := pondArea * 50
def lengthMultipleOfWidth (length width : ℕ) := ∃ k : ℕ, length = k * width

-- Main statement to prove the ratio of length to width is 2:1
theorem ratio_of_length_to_width_of_field_is_two_to_one :
  lengthMultipleOfWidth lengthOfField (widthOfField fieldArea pondArea) →
  lengthOfField = 2 * (widthOfField fieldArea pondArea) :=
by
  -- Conditions
  have h1 : pondSideLength = 8 := rfl
  have h2 : pondArea = pondSideLength * pondSideLength := rfl
  have h3 : fieldArea = pondArea * 50 := rfl
  have h4 : lengthOfField = 80 := rfl
  sorry

end NUMINAMATH_GPT_ratio_of_length_to_width_of_field_is_two_to_one_l400_40072


namespace NUMINAMATH_GPT_percentage_of_profit_without_discount_l400_40061

-- Definitions for the conditions
def cost_price : ℝ := 100
def discount_rate : ℝ := 0.04
def profit_rate : ℝ := 0.32

-- The statement to prove
theorem percentage_of_profit_without_discount :
  let selling_price := cost_price + (profit_rate * cost_price)
  (selling_price - cost_price) / cost_price * 100 = 32 := by
  let selling_price := cost_price + (profit_rate * cost_price)
  sorry

end NUMINAMATH_GPT_percentage_of_profit_without_discount_l400_40061


namespace NUMINAMATH_GPT_convert_binary_to_decimal_l400_40025

theorem convert_binary_to_decimal : (1 * 2^2 + 1 * 2^1 + 1 * 2^0) = 7 := by
  sorry

end NUMINAMATH_GPT_convert_binary_to_decimal_l400_40025


namespace NUMINAMATH_GPT_abs_less_than_zero_impossible_l400_40059

theorem abs_less_than_zero_impossible (x : ℝ) : |x| < 0 → false :=
by
  sorry

end NUMINAMATH_GPT_abs_less_than_zero_impossible_l400_40059


namespace NUMINAMATH_GPT_range_of_x_l400_40032

noncomputable def function_y (x : ℝ) : ℝ := 2 / (Real.sqrt (x + 4))

theorem range_of_x : ∀ x : ℝ, (∃ y : ℝ, y = function_y x) → x > -4 :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_range_of_x_l400_40032


namespace NUMINAMATH_GPT_irrational_power_to_nat_l400_40083

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.log 3 / Real.log (Real.sqrt 2) 

theorem irrational_power_to_nat 
  (ha_irr : ¬ ∃ (q : ℚ), a = q)
  (hb_irr : ¬ ∃ (q : ℚ), b = q) : (a ^ b) = 3 := by
  -- \[a = \sqrt{2}, b = \log_{\sqrt{2}}(3)\]
  sorry

end NUMINAMATH_GPT_irrational_power_to_nat_l400_40083


namespace NUMINAMATH_GPT_no_real_sol_l400_40044

open Complex

theorem no_real_sol (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (↑(x.re) ≠ x ∨ ↑(y.re) ≠ y) → (x + y) / y ≠ x / (y + x) := by
  sorry

end NUMINAMATH_GPT_no_real_sol_l400_40044


namespace NUMINAMATH_GPT_triangle_angle_contradiction_l400_40064

theorem triangle_angle_contradiction (α β γ : ℝ)
  (h_sum : α + β + γ = 180)
  (h1 : α > 60) (h2 : β > 60) (h3 : γ > 60) :
  false :=
by
  have h : α + β + γ > 180 := by
  { linarith }
  linarith

end NUMINAMATH_GPT_triangle_angle_contradiction_l400_40064
