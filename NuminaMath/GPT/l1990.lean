import Mathlib

namespace NUMINAMATH_GPT_find_x_for_g_statement_l1990_199087

noncomputable def g (x : ℝ) : ℝ := (x + 4) ^ (1/3) / 5 ^ (1/3)

theorem find_x_for_g_statement (x : ℝ) : g (3 * x) = 3 * g x ↔ x = -13 / 3 := by
  sorry

end NUMINAMATH_GPT_find_x_for_g_statement_l1990_199087


namespace NUMINAMATH_GPT_total_cards_beginning_l1990_199044

-- Define the initial conditions
def num_boxes_orig : ℕ := 2 + 5  -- Robie originally had 2 + 5 boxes
def cards_per_box : ℕ := 10      -- Each box contains 10 cards
def extra_cards : ℕ := 5         -- 5 cards were not placed in a box

-- Prove the total number of cards Robie had in the beginning
theorem total_cards_beginning : (num_boxes_orig * cards_per_box) + extra_cards = 75 :=
by sorry

end NUMINAMATH_GPT_total_cards_beginning_l1990_199044


namespace NUMINAMATH_GPT_mary_mac_download_time_l1990_199055

theorem mary_mac_download_time (x : ℕ) (windows_download : ℕ) (total_glitch : ℕ) (time_without_glitches : ℕ) (total_time : ℕ) :
  windows_download = 3 * x ∧
  total_glitch = 14 ∧
  time_without_glitches = 2 * total_glitch ∧
  total_time = 82 ∧
  x + windows_download + total_glitch + time_without_glitches = total_time →
  x = 10 :=
by 
  sorry

end NUMINAMATH_GPT_mary_mac_download_time_l1990_199055


namespace NUMINAMATH_GPT_volume_of_prism_l1990_199049

noncomputable def prismVolume {x y z : ℝ} 
  (h1 : x * y = 20) 
  (h2 : y * z = 12) 
  (h3 : x * z = 8) : ℝ :=
  x * y * z

theorem volume_of_prism (x y z : ℝ)
  (h1 : x * y = 20)
  (h2 : y * z = 12)
  (h3 : x * z = 8) : prismVolume h1 h2 h3 = 8 * Real.sqrt 15 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_prism_l1990_199049


namespace NUMINAMATH_GPT_maximum_M_k_l1990_199016

-- Define the problem
def J (k : ℕ) : ℕ := 10^(k + 2) + 128

-- Define M(k) as the number of factors of 2 in the prime factorization of J(k)
def M (k : ℕ) : ℕ :=
  -- implementation details omitted
  sorry

-- The core theorem to prove
theorem maximum_M_k : ∃ k > 0, M k = 8 :=
by sorry

end NUMINAMATH_GPT_maximum_M_k_l1990_199016


namespace NUMINAMATH_GPT_total_handshakes_l1990_199025

-- Definitions based on conditions
def num_wizards : ℕ := 25
def num_elves : ℕ := 18

-- Each wizard shakes hands with every other wizard
def wizard_handshakes : ℕ := num_wizards * (num_wizards - 1) / 2

-- Each elf shakes hands with every wizard
def elf_wizard_handshakes : ℕ := num_elves * num_wizards

-- Total handshakes is the sum of the above two
theorem total_handshakes : wizard_handshakes + elf_wizard_handshakes = 750 := by
  sorry

end NUMINAMATH_GPT_total_handshakes_l1990_199025


namespace NUMINAMATH_GPT_area_enclosed_by_curve_l1990_199038

theorem area_enclosed_by_curve :
  let arc_length := (3 * Real.pi) / 4
  let side_length := 3
  let radius := arc_length / ((3 * Real.pi) / 4)
  let sector_area := (radius ^ 2 * Real.pi * (3 * Real.pi) / (4 * 2 * Real.pi))
  let total_sector_area := 8 * sector_area
  let octagon_area := 2 * (1 + Real.sqrt 2) * (side_length ^ 2)
  total_sector_area + octagon_area = 54 + 54 * Real.sqrt 2 + 3 * Real.pi
:= sorry

end NUMINAMATH_GPT_area_enclosed_by_curve_l1990_199038


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1990_199009

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 - 6 * x + 5 > 0) ↔ (x < 1 ∨ x > 5) :=
by sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1990_199009


namespace NUMINAMATH_GPT_trigonometric_identity_l1990_199000

theorem trigonometric_identity :
  let cos_30 := (Real.sqrt 3) / 2
  let sin_60 := (Real.sqrt 3) / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 1 := by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1990_199000


namespace NUMINAMATH_GPT_complement_union_eq_l1990_199097

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x ^ 2 - 4 * x + 3 = 0}

theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 0} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_eq_l1990_199097


namespace NUMINAMATH_GPT_parabola_transformation_l1990_199076

-- Defining the original parabola
def original_parabola (x : ℝ) : ℝ :=
  3 * x^2

-- Condition: Transformation 1 -> Translation 4 units to the right
def translated_right_parabola (x : ℝ) : ℝ :=
  original_parabola (x - 4)

-- Condition: Transformation 2 -> Translation 1 unit upwards
def translated_up_parabola (x : ℝ) : ℝ :=
  translated_right_parabola x + 1

-- Statement that needs to be proved
theorem parabola_transformation :
  ∀ x : ℝ, translated_up_parabola x = 3 * (x - 4)^2 + 1 :=
by
  intros x
  sorry

end NUMINAMATH_GPT_parabola_transformation_l1990_199076


namespace NUMINAMATH_GPT_proof_parabola_statements_l1990_199023

theorem proof_parabola_statements (b c : ℝ)
  (h1 : 1/2 - b + c < 0)
  (h2 : 2 - 2 * b + c < 0) :
  (b^2 > 2 * c) ∧
  (c > 1 → b > 3/2) ∧
  (∀ (m1 m2 : ℝ), m1 < m2 ∧ m2 < b → ∀ (y : ℝ), y = (1/2)*m1^2 - b*m1 + c → ∀ (y2 : ℝ), y2 = (1/2)*m2^2 - b*m2 + c → y > y2) ∧
  (¬(∃ x1 x2 : ℝ, (1/2) * x1^2 - b * x1 + c = 0 ∧ (1/2) * x2^2 - b * x2 + c = 0 ∧ x1 + x2 > 3)) :=
by sorry

end NUMINAMATH_GPT_proof_parabola_statements_l1990_199023


namespace NUMINAMATH_GPT_classroom_students_l1990_199048

theorem classroom_students (n : ℕ) (h1 : 20 < n ∧ n < 30) 
  (h2 : ∃ n_y : ℕ, n = 3 * n_y + 1) 
  (h3 : ∃ n_y' : ℕ, n = (4 * (n - 1)) / 3 + 1) :
  n = 25 := 
by sorry

end NUMINAMATH_GPT_classroom_students_l1990_199048


namespace NUMINAMATH_GPT_walter_fraction_fewer_bananas_l1990_199093

theorem walter_fraction_fewer_bananas (f : ℚ) (h1 : 56 + (56 - 56 * f) = 98) : f = 1 / 4 :=
sorry

end NUMINAMATH_GPT_walter_fraction_fewer_bananas_l1990_199093


namespace NUMINAMATH_GPT_units_digit_sum_squares_of_odd_integers_l1990_199090

theorem units_digit_sum_squares_of_odd_integers :
  let first_2005_odd_units := [802, 802, 401] -- counts for units 1, 9, 5 respectively
  let extra_squares_last_6 := [9, 1, 3, 9, 5, 9] -- units digits of the squares of the last 6 numbers
  let total_sum :=
        (first_2005_odd_units[0] * 1 + 
         first_2005_odd_units[1] * 9 + 
         first_2005_odd_units[2] * 5) +
        (extra_squares_last_6.sum)
  (total_sum % 10) = 1 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_sum_squares_of_odd_integers_l1990_199090


namespace NUMINAMATH_GPT_sum_of_digits_of_N_l1990_199098

open Nat

theorem sum_of_digits_of_N (T : ℕ) (hT : T = 3003) :
  ∃ N : ℕ, (N * (N + 1)) / 2 = T ∧ (digits 10 N).sum = 14 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_N_l1990_199098


namespace NUMINAMATH_GPT_price_comparison_2010_l1990_199083

def X_initial : ℝ := 4.20
def Y_initial : ℝ := 6.30
def r_X : ℝ := 0.45
def r_Y : ℝ := 0.20
def n : ℕ := 9

theorem price_comparison_2010: 
  X_initial + r_X * n > Y_initial + r_Y * n := by
  sorry

end NUMINAMATH_GPT_price_comparison_2010_l1990_199083


namespace NUMINAMATH_GPT_viewing_spot_coordinate_correct_l1990_199053

-- Define the coordinates of the landmarks
def first_landmark := 150
def second_landmark := 450

-- The expected coordinate of the viewing spot
def expected_viewing_spot := 350

-- The theorem that formalizes the problem
theorem viewing_spot_coordinate_correct :
  let distance := second_landmark - first_landmark
  let fractional_distance := (2 / 3) * distance
  let viewing_spot := first_landmark + fractional_distance
  viewing_spot = expected_viewing_spot := 
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_viewing_spot_coordinate_correct_l1990_199053


namespace NUMINAMATH_GPT_tommys_books_l1990_199007

-- Define the cost of each book
def book_cost : ℕ := 5

-- Define the amount Tommy already has
def tommy_money : ℕ := 13

-- Define the amount Tommy needs to save up
def tommy_goal : ℕ := 27

-- Prove the number of books Tommy wants to buy
theorem tommys_books : tommy_goal + tommy_money = 40 ∧ (tommy_goal + tommy_money) / book_cost = 8 :=
by
  sorry

end NUMINAMATH_GPT_tommys_books_l1990_199007


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l1990_199002

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 > Real.log x)) ↔ (∃ x : ℝ, x^2 ≤ Real.log x) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l1990_199002


namespace NUMINAMATH_GPT_lisa_total_miles_flown_l1990_199026

variable (distance_per_trip : ℝ := 256.0)
variable (number_of_trips : ℝ := 32.0)

theorem lisa_total_miles_flown : distance_per_trip * number_of_trips = 8192.0 := by
  sorry

end NUMINAMATH_GPT_lisa_total_miles_flown_l1990_199026


namespace NUMINAMATH_GPT_twice_plus_eight_lt_five_times_x_l1990_199069

theorem twice_plus_eight_lt_five_times_x (x : ℝ) : 2 * x + 8 < 5 * x := 
sorry

end NUMINAMATH_GPT_twice_plus_eight_lt_five_times_x_l1990_199069


namespace NUMINAMATH_GPT_most_likely_outcome_is_draw_l1990_199051

noncomputable def prob_A_win : ℝ := 0.3
noncomputable def prob_A_not_lose : ℝ := 0.7
noncomputable def prob_draw : ℝ := prob_A_not_lose - prob_A_win

theorem most_likely_outcome_is_draw :
  prob_draw = 0.4 ∧ prob_draw > prob_A_win ∧ prob_draw > (1 - prob_A_not_lose) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_most_likely_outcome_is_draw_l1990_199051


namespace NUMINAMATH_GPT_cubic_expression_l1990_199042

theorem cubic_expression (a b c : ℝ) (h1 : a + b + c = 15) (h2 : ab + ac + bc = 50) : 
  a^3 + b^3 + c^3 - 3 * a * b * c = 1125 :=
sorry

end NUMINAMATH_GPT_cubic_expression_l1990_199042


namespace NUMINAMATH_GPT_point_not_on_line_l1990_199077

theorem point_not_on_line
  (p q : ℝ)
  (h : p * q > 0) :
  ¬ (∃ (x y : ℝ), x = 2023 ∧ y = 0 ∧ y = p * x + q) :=
by
  sorry

end NUMINAMATH_GPT_point_not_on_line_l1990_199077


namespace NUMINAMATH_GPT_min_value_one_over_a_plus_two_over_b_l1990_199099

/-- Given a > 0, b > 0, 2a + b = 1, prove that the minimum value of (1/a) + (2/b) is 8 --/
theorem min_value_one_over_a_plus_two_over_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  (1 / a) + (2 / b) ≥ 8 :=
sorry

end NUMINAMATH_GPT_min_value_one_over_a_plus_two_over_b_l1990_199099


namespace NUMINAMATH_GPT_find_t_value_l1990_199096

theorem find_t_value (k t : ℤ) (h1 : 0 < k) (h2 : k < 10) (h3 : 0 < t) (h4 : t < 10) : t = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_t_value_l1990_199096


namespace NUMINAMATH_GPT_merchant_profit_l1990_199005

theorem merchant_profit 
  (CP MP SP profit : ℝ)
  (markup_percentage discount_percentage : ℝ)
  (h1 : CP = 100)
  (h2 : markup_percentage = 0.40)
  (h3 : discount_percentage = 0.10)
  (h4 : MP = CP + (markup_percentage * CP))
  (h5 : SP = MP - (discount_percentage * MP))
  (h6 : profit = SP - CP) :
  profit / CP * 100 = 26 :=
by sorry

end NUMINAMATH_GPT_merchant_profit_l1990_199005


namespace NUMINAMATH_GPT_num_even_3digit_nums_lt_700_l1990_199058

theorem num_even_3digit_nums_lt_700 
  (digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}) 
  (even_digits : Finset ℕ := {2, 4, 6}) 
  (h1 : ∀ n ∈ digits, n < 10)
  (h2 : 0 ∉ digits) : 
  ∃ n, n = 126 ∧ ∀ d, d ∈ digits → 
  (d < 10) ∧ ∀ u, u ∈ even_digits → 
  (u < 10) 
:=
  sorry

end NUMINAMATH_GPT_num_even_3digit_nums_lt_700_l1990_199058


namespace NUMINAMATH_GPT_triangle_similar_l1990_199015

variables {a b c m_a m_b m_c t : ℝ}

-- Define the triangle ABC and its properties
def triangle_ABC (a b c m_a m_b m_c t : ℝ) : Prop :=
  t = (1 / 2) * a * m_a ∧
  t = (1 / 2) * b * m_b ∧
  t = (1 / 2) * c * m_c ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  m_a > 0 ∧ m_b > 0 ∧ m_c > 0 ∧
  t > 0

-- Define the similarity condition for the triangles
def similitude_from_reciprocals (a b c m_a m_b m_c t : ℝ) : Prop :=
  (1 / m_a) / (1 / m_b) = a / b ∧
  (1 / m_b) / (1 / m_c) = b / c ∧
  (1 / m_a) / (1 / m_c) = a / c

theorem triangle_similar (a b c m_a m_b m_c t : ℝ) :
  triangle_ABC a b c m_a m_b m_c t →
  similitude_from_reciprocals a b c m_a m_b m_c t :=
by
  intro h
  sorry

end NUMINAMATH_GPT_triangle_similar_l1990_199015


namespace NUMINAMATH_GPT_original_number_of_coins_in_first_pile_l1990_199094

noncomputable def originalCoinsInFirstPile (x y z : ℕ) : ℕ :=
  if h : (2 * (x - y) = 16) ∧ (2 * y - z = 16) ∧ (2 * z - (x + y) = 16) then x else 0

theorem original_number_of_coins_in_first_pile (x y z : ℕ) (h1 : 2 * (x - y) = 16) 
                                              (h2 : 2 * y - z = 16) 
                                              (h3 : 2 * z - (x + y) = 16) : x = 22 :=
by sorry

end NUMINAMATH_GPT_original_number_of_coins_in_first_pile_l1990_199094


namespace NUMINAMATH_GPT_prod_ge_27_eq_iff_equality_l1990_199037

variable (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
          (h4 : a + b + c + 2 = a * b * c)

theorem prod_ge_27 (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a + b + c + 2 = a * b * c) : (a + 1) * (b + 1) * (c + 1) ≥ 27 :=
by sorry

theorem eq_iff_equality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a + b + c + 2 = a * b * c) : 
  ((a + 1) * (b + 1) * (c + 1) = 27) ↔ (a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end NUMINAMATH_GPT_prod_ge_27_eq_iff_equality_l1990_199037


namespace NUMINAMATH_GPT_pets_remaining_is_correct_l1990_199033

-- Definitions for the initial conditions and actions taken
def initial_puppies : Nat := 7
def initial_kittens : Nat := 6
def puppies_sold : Nat := 2
def kittens_sold : Nat := 3

-- Definition that calculates the remaining number of pets
def remaining_pets : Nat := initial_puppies + initial_kittens - (puppies_sold + kittens_sold)

-- The theorem to prove
theorem pets_remaining_is_correct : remaining_pets = 8 := by sorry

end NUMINAMATH_GPT_pets_remaining_is_correct_l1990_199033


namespace NUMINAMATH_GPT_sufficient_condition_hyperbola_l1990_199001

theorem sufficient_condition_hyperbola (m : ℝ) (h : 5 < m) : 
  ∃ a b : ℝ, (a > 0) ∧ (b < 0) ∧ (∀ x y : ℝ, (x^2)/(a) + (y^2)/(b) = 1) := 
sorry

end NUMINAMATH_GPT_sufficient_condition_hyperbola_l1990_199001


namespace NUMINAMATH_GPT_tan_symmetric_about_k_pi_over_2_min_value_cos2x_plus_sinx_l1990_199084

theorem tan_symmetric_about_k_pi_over_2 (k : ℤ) : 
  (∀ x : ℝ, Real.tan (x + k * Real.pi / 2) = Real.tan x) := 
sorry

theorem min_value_cos2x_plus_sinx : 
  (∀ x : ℝ, Real.cos x ^ 2 + Real.sin x ≥ -1) ∧ (∃ x : ℝ, Real.cos x ^ 2 + Real.sin x = -1) :=
sorry

end NUMINAMATH_GPT_tan_symmetric_about_k_pi_over_2_min_value_cos2x_plus_sinx_l1990_199084


namespace NUMINAMATH_GPT_age_difference_l1990_199054

variables (F S M B : ℕ)

theorem age_difference:
  (F - S = 38) → (M - B = 36) → (F - M = 6) → (S - B = 4) :=
by
  intros h1 h2 h3
  -- Use the conditions to derive that S - B = 4
  sorry

end NUMINAMATH_GPT_age_difference_l1990_199054


namespace NUMINAMATH_GPT_pascal_triangle_row10_sum_l1990_199061

def pascal_triangle_row_sum (n : ℕ) : ℕ :=
  2 ^ n

theorem pascal_triangle_row10_sum : pascal_triangle_row_sum 10 = 1024 :=
by
  -- Proof will demonstrate that 2^10 = 1024
  sorry

end NUMINAMATH_GPT_pascal_triangle_row10_sum_l1990_199061


namespace NUMINAMATH_GPT_rhombus_side_length_l1990_199039

noncomputable def side_length_rhombus (AB BC AC : ℝ) (condition1 : AB = 12) (condition2 : BC = 12) (condition3 : AC = 6) : ℝ :=
  4

theorem rhombus_side_length (AB BC AC : ℝ) (condition1 : AB = 12) (condition2 : BC = 12) (condition3 : AC = 6) (x : ℝ) :
  side_length_rhombus AB BC AC condition1 condition2 condition3 = x ↔ x = 4 := by
  sorry

end NUMINAMATH_GPT_rhombus_side_length_l1990_199039


namespace NUMINAMATH_GPT_area_of_circle_l1990_199070

theorem area_of_circle (r θ : ℝ) (h : r = 4 * Real.cos θ - 3 * Real.sin θ) :
  ∃ π : ℝ, π * (5/2)^2 = 25 * π / 4 :=
by 
  sorry

end NUMINAMATH_GPT_area_of_circle_l1990_199070


namespace NUMINAMATH_GPT_find_d_l1990_199010

theorem find_d (d : ℝ) (h₁ : ∃ x, x = ⌊d⌋ ∧ 3 * x^2 + 19 * x - 84 = 0)
                (h₂ : ∃ y, y = d - ⌊d⌋ ∧ 5 * y^2 - 28 * y + 12 = 0 ∧ 0 ≤ y ∧ y < 1) :
  d = 3.2 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l1990_199010


namespace NUMINAMATH_GPT_correct_answer_of_john_l1990_199017

theorem correct_answer_of_john (x : ℝ) (h : 5 * x + 4 = 104) : (x + 5) / 4 = 6.25 :=
by
  sorry

end NUMINAMATH_GPT_correct_answer_of_john_l1990_199017


namespace NUMINAMATH_GPT_polar_equation_of_circle_slope_of_line_l1990_199003

-- Part 1: Polar equation of circle C
theorem polar_equation_of_circle (x y : ℝ) :
  (x - 2) ^ 2 + y ^ 2 = 9 -> ∃ (ρ θ : ℝ), ρ^2 - 4*ρ*Real.cos θ - 5 = 0 := 
sorry

-- Part 2: Slope of line L intersecting C at points A and B
theorem slope_of_line (α : ℝ) (L : ℝ → ℝ × ℝ) (A B : ℝ × ℝ) :
  (∀ t, L t = (t * Real.cos α, t * Real.sin α)) ∧ dist A B = 2 * Real.sqrt 7 ∧ 
  (∃ x y, (x - 2) ^ 2 + y ^ 2 = 9 ∧ L (Real.sqrt ((x - 2) ^ 2 + y ^ 2)) = (x, y))
  -> Real.tan α = 1 ∨ Real.tan α = -1 :=
sorry

end NUMINAMATH_GPT_polar_equation_of_circle_slope_of_line_l1990_199003


namespace NUMINAMATH_GPT_sum_of_three_digit_even_naturals_correct_l1990_199082

noncomputable def sum_of_three_digit_even_naturals : ℕ := 
  let a := 100
  let l := 998
  let d := 2
  let n := (l - a) / d + 1
  n / 2 * (a + l)

theorem sum_of_three_digit_even_naturals_correct : 
  sum_of_three_digit_even_naturals = 247050 := by 
  sorry

end NUMINAMATH_GPT_sum_of_three_digit_even_naturals_correct_l1990_199082


namespace NUMINAMATH_GPT_randy_piggy_bank_final_amount_l1990_199071

def initial_amount : ℕ := 200
def spending_per_trip : ℕ := 2
def trips_per_month : ℕ := 4
def months_per_year : ℕ := 12

theorem randy_piggy_bank_final_amount :
  initial_amount - (spending_per_trip * trips_per_month * months_per_year) = 104 :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_randy_piggy_bank_final_amount_l1990_199071


namespace NUMINAMATH_GPT_eight_point_shots_count_is_nine_l1990_199064

def num_8_point_shots (x y z : ℕ) := 8 * x + 9 * y + 10 * z = 100 ∧
                                      x + y + z > 11 ∧ 
                                      x + y + z ≤ 12 ∧ 
                                      x > 0 ∧ 
                                      y > 0 ∧ 
                                      z > 0

theorem eight_point_shots_count_is_nine : 
  ∃ x y z : ℕ, num_8_point_shots x y z ∧ x = 9 :=
by
  sorry

end NUMINAMATH_GPT_eight_point_shots_count_is_nine_l1990_199064


namespace NUMINAMATH_GPT_geometric_sequences_l1990_199067

theorem geometric_sequences :
  ∃ (a q : ℝ) (a1 a2 a3 : ℕ → ℝ), 
    (∀ n, a1 n = a * (q - 2) ^ n) ∧ 
    (∀ n, a2 n = 2 * a * (q - 1) ^ n) ∧ 
    (∀ n, a3 n = 4 * a * q ^ n) ∧
    a = 1 ∧ q = 4 ∨ a = 192 / 31 ∧ q = 9 / 8 ∧
    (a + 2 * a + 4 * a = 84) ∧
    (a * (q - 2) + 2 * a * (q - 1) + 4 * a * q = 24) :=
sorry

end NUMINAMATH_GPT_geometric_sequences_l1990_199067


namespace NUMINAMATH_GPT_lesser_of_two_numbers_l1990_199089

theorem lesser_of_two_numbers (x y : ℝ) (h1 : x + y = 50) (h2 : x - y = 6) : y = 22 :=
by
  sorry

end NUMINAMATH_GPT_lesser_of_two_numbers_l1990_199089


namespace NUMINAMATH_GPT_total_litter_pieces_l1990_199088

-- Define the number of glass bottles and aluminum cans as constants.
def glass_bottles : ℕ := 10
def aluminum_cans : ℕ := 8

-- State the theorem that the sum of glass bottles and aluminum cans is 18.
theorem total_litter_pieces : glass_bottles + aluminum_cans = 18 := by
  sorry

end NUMINAMATH_GPT_total_litter_pieces_l1990_199088


namespace NUMINAMATH_GPT_nm_odd_if_squares_sum_odd_l1990_199091

theorem nm_odd_if_squares_sum_odd
  (n m : ℤ)
  (h : (n^2 + m^2) % 2 = 1) :
  (n * m) % 2 = 1 :=
sorry

end NUMINAMATH_GPT_nm_odd_if_squares_sum_odd_l1990_199091


namespace NUMINAMATH_GPT_stock_value_sale_l1990_199066

theorem stock_value_sale
  (X : ℝ)
  (h1 : 0.20 * X * 0.10 - 0.80 * X * 0.05 = -350) :
  X = 17500 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_stock_value_sale_l1990_199066


namespace NUMINAMATH_GPT_a14_eq_33_l1990_199018

variable {a : ℕ → ℝ}
variables (d : ℝ) (a1 : ℝ)

-- Defining the arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℝ := a1 + n * d

-- Given conditions
axiom a5_eq_6 : arithmetic_sequence 4 = 6
axiom a8_eq_15 : arithmetic_sequence 7 = 15

-- Theorem statement
theorem a14_eq_33 : arithmetic_sequence 13 = 33 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_a14_eq_33_l1990_199018


namespace NUMINAMATH_GPT_range_of_a_for_positive_f_l1990_199041

-- Let the function \(f(x) = ax^2 - 2x + 2\)
def f (a x : ℝ) := a * x^2 - 2 * x + 2

-- Theorem: The range of the real number \( a \) such that \( f(x) > 0 \) for all \( x \) in \( 1 < x < 4 \) is \((\dfrac{1}{2}, +\infty)\)
theorem range_of_a_for_positive_f :
  { a : ℝ | ∀ x : ℝ, 1 < x ∧ x < 4 → f a x > 0 } = { a : ℝ | a > 1/2 } :=
sorry

end NUMINAMATH_GPT_range_of_a_for_positive_f_l1990_199041


namespace NUMINAMATH_GPT_ball_attendance_l1990_199035

variable (n m : ℕ)

def ball_conditions (n m : ℕ) := 
  n + m < 50 ∧ 
  4 ∣ 3 * n ∧ 
  7 ∣ 5 * m

theorem ball_attendance (n m : ℕ) (h : ball_conditions n m) : 
  n + m = 41 :=
sorry

end NUMINAMATH_GPT_ball_attendance_l1990_199035


namespace NUMINAMATH_GPT_cost_price_proof_l1990_199086

noncomputable def cost_price_per_bowl : ℚ := 1400 / 103

theorem cost_price_proof
  (total_bowls: ℕ) (sold_bowls: ℕ) (selling_price_per_bowl: ℚ)
  (percentage_gain: ℚ) 
  (total_bowls_eq: total_bowls = 110)
  (sold_bowls_eq: sold_bowls = 100)
  (selling_price_per_bowl_eq: selling_price_per_bowl = 14)
  (percentage_gain_eq: percentage_gain = 300 / 11) :
  (selling_price_per_bowl * sold_bowls - (sold_bowls + 3) * (selling_price_per_bowl / (3 * percentage_gain / 100))) = cost_price_per_bowl :=
by
  sorry

end NUMINAMATH_GPT_cost_price_proof_l1990_199086


namespace NUMINAMATH_GPT_sanya_towels_count_l1990_199056

-- Defining the conditions based on the problem
def towels_per_hour := 7
def hours_per_day := 2
def days_needed := 7

-- The main statement to prove
theorem sanya_towels_count : 
  (towels_per_hour * hours_per_day * days_needed = 98) :=
by
  sorry

end NUMINAMATH_GPT_sanya_towels_count_l1990_199056


namespace NUMINAMATH_GPT_sin_2alpha_pos_of_tan_alpha_pos_l1990_199030

theorem sin_2alpha_pos_of_tan_alpha_pos (α : Real) (h : Real.tan α > 0) : Real.sin (2 * α) > 0 :=
sorry

end NUMINAMATH_GPT_sin_2alpha_pos_of_tan_alpha_pos_l1990_199030


namespace NUMINAMATH_GPT_solve_for_x_l1990_199004

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1990_199004


namespace NUMINAMATH_GPT_eight_hash_six_l1990_199057

def op (r s : ℝ) : ℝ := sorry

axiom op_r_zero (r : ℝ): op r 0 = r + 1
axiom op_comm (r s : ℝ) : op r s = op s r
axiom op_r_add_one_s (r s : ℝ): op (r + 1) s = (op r s) + s + 2

theorem eight_hash_six : op 8 6 = 69 := 
by sorry

end NUMINAMATH_GPT_eight_hash_six_l1990_199057


namespace NUMINAMATH_GPT_new_ratio_first_term_less_than_implied_l1990_199036

-- Define the original and new ratios
def original_ratio := (6, 7)
def subtracted_value := 3
def new_ratio := (original_ratio.1 - subtracted_value, original_ratio.2 - subtracted_value)

-- Prove the required property
theorem new_ratio_first_term_less_than_implied {r1 r2 : ℕ} (h : new_ratio = (3, 4))
  (h_less : r1 > 3) :
  new_ratio.1 < r1 := 
sorry

end NUMINAMATH_GPT_new_ratio_first_term_less_than_implied_l1990_199036


namespace NUMINAMATH_GPT_fraction_multiplication_simplifies_l1990_199014

theorem fraction_multiplication_simplifies :
  (3 : ℚ) / 4 * (4 / 5) * (2 / 3) = 2 / 5 := 
by 
  -- Prove the equality step-by-step
  sorry

end NUMINAMATH_GPT_fraction_multiplication_simplifies_l1990_199014


namespace NUMINAMATH_GPT_sum_mod_15_l1990_199024

theorem sum_mod_15 
  (d e f : ℕ) 
  (hd : d % 15 = 11)
  (he : e % 15 = 12)
  (hf : f % 15 = 13) : 
  (d + e + f) % 15 = 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_mod_15_l1990_199024


namespace NUMINAMATH_GPT_stratified_sampling_l1990_199012

theorem stratified_sampling (total_students : ℕ) (ratio_grade1 ratio_grade2 ratio_grade3 : ℕ) (sample_size : ℕ) (h_ratio : ratio_grade1 = 3 ∧ ratio_grade2 = 3 ∧ ratio_grade3 = 4) (h_sample_size : sample_size = 50) : 
  (ratio_grade2 / (ratio_grade1 + ratio_grade2 + ratio_grade3) : ℚ) * sample_size = 15 := 
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_l1990_199012


namespace NUMINAMATH_GPT_approximation_hundred_thousandth_place_l1990_199020

theorem approximation_hundred_thousandth_place (n : ℕ) (h : n = 537400000) : 
  ∃ p : ℕ, p = 100000 := 
sorry

end NUMINAMATH_GPT_approximation_hundred_thousandth_place_l1990_199020


namespace NUMINAMATH_GPT_distance_between_consecutive_trees_l1990_199019

-- Definitions from the problem statement
def yard_length : ℕ := 414
def number_of_trees : ℕ := 24
def number_of_intervals : ℕ := number_of_trees - 1
def distance_between_trees : ℕ := yard_length / number_of_intervals

-- Main theorem we want to prove
theorem distance_between_consecutive_trees :
  distance_between_trees = 18 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_distance_between_consecutive_trees_l1990_199019


namespace NUMINAMATH_GPT_division_problem_l1990_199059

theorem division_problem (D d q r : ℕ) 
  (h1 : D + d + q + r = 205)
  (h2 : q = d) :
  D = 174 ∧ d = 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_division_problem_l1990_199059


namespace NUMINAMATH_GPT_find_a1_l1990_199008

theorem find_a1 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_rec : ∀ n ≥ 2, a n + 2 * S n * S (n - 1) = 0)
  (h_S5 : S 5 = 1 / 11) : 
  a 1 = 1 / 3 := 
sorry

end NUMINAMATH_GPT_find_a1_l1990_199008


namespace NUMINAMATH_GPT_max_fridays_in_year_l1990_199032

theorem max_fridays_in_year (days_in_common_year days_in_leap_year : ℕ) 
  (h_common_year : days_in_common_year = 365)
  (h_leap_year : days_in_leap_year = 366) : 
  ∃ (max_fridays : ℕ), max_fridays = 53 := 
by
  existsi 53
  sorry

end NUMINAMATH_GPT_max_fridays_in_year_l1990_199032


namespace NUMINAMATH_GPT_find_number_l1990_199011

theorem find_number (x : ℤ) (h : 3 * x + 3 * 12 + 3 * 13 + 11 = 134) : x = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1990_199011


namespace NUMINAMATH_GPT_regular_triangular_pyramid_volume_l1990_199040

noncomputable def pyramid_volume (a h γ : ℝ) : ℝ :=
  (Real.sqrt 3 * a^2 * h) / 12

theorem regular_triangular_pyramid_volume
  (a h γ : ℝ) (h_nonneg : 0 ≤ h) (γ_nonneg : 0 ≤ γ) :
  pyramid_volume a h γ = (Real.sqrt 3 * a^2 * h) / 12 :=
by
  sorry

end NUMINAMATH_GPT_regular_triangular_pyramid_volume_l1990_199040


namespace NUMINAMATH_GPT_hall_reunion_attendance_l1990_199063

/-- At the Taj Hotel, two family reunions are happening: the Oates reunion and the Hall reunion.
All 150 guests at the hotel attend at least one of the reunions.
70 people attend the Oates reunion.
28 people attend both reunions.
Prove that 108 people attend the Hall reunion. -/
theorem hall_reunion_attendance (total oates both : ℕ) (h_total : total = 150) (h_oates : oates = 70) (h_both : both = 28) :
  ∃ hall : ℕ, total = oates + hall - both ∧ hall = 108 :=
by
  -- Proof will be skipped and not considered for this task
  sorry

end NUMINAMATH_GPT_hall_reunion_attendance_l1990_199063


namespace NUMINAMATH_GPT_men_left_the_job_l1990_199078

theorem men_left_the_job
    (work_rate_20men : 20 * 4 = 30)
    (work_rate_remaining : 6 * 6 = 36) :
    4 = 20 - (20 * 4) / (6 * 6)  :=
by
  sorry

end NUMINAMATH_GPT_men_left_the_job_l1990_199078


namespace NUMINAMATH_GPT_no_real_roots_of_quadratic_l1990_199075

def quadratic (a b c : ℝ) : ℝ × ℝ × ℝ := (a^2, b^2 + a^2 - c^2, b^2)

def discriminant (A B C : ℝ) : ℝ := B^2 - 4 * A * C

theorem no_real_roots_of_quadratic (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : c > 0)
  (h3 : |a - b| < c)
  : (discriminant (a^2) (b^2 + a^2 - c^2) (b^2)) < 0 :=
sorry

end NUMINAMATH_GPT_no_real_roots_of_quadratic_l1990_199075


namespace NUMINAMATH_GPT_minimum_a_condition_l1990_199052

theorem minimum_a_condition (a : ℝ) (h₀ : 0 < a) 
  (h₁ : ∀ x : ℝ, 1 < x → x + a / (x - 1) ≥ 5) :
  4 ≤ a :=
sorry

end NUMINAMATH_GPT_minimum_a_condition_l1990_199052


namespace NUMINAMATH_GPT_S7_value_l1990_199072

def arithmetic_seq_sum (n : ℕ) (a_1 d : ℚ) : ℚ :=
  n * a_1 + (n * (n - 1) / 2) * d

def a_n (n : ℕ) (a_1 d : ℚ) : ℚ :=
  a_1 + (n - 1) * d

theorem S7_value (a_1 d : ℚ) (S_n : ℕ → ℚ)
  (hSn_def : ∀ n, S_n n = arithmetic_seq_sum n a_1 d)
  (h_sum_condition : S_n 7 + S_n 5 = 10)
  (h_a3_condition : a_n 3 a_1 d = 5) :
  S_n 7 = -15 :=
by
  sorry

end NUMINAMATH_GPT_S7_value_l1990_199072


namespace NUMINAMATH_GPT_range_of_a_l1990_199045

noncomputable def f (x : ℝ) : ℝ := 4 * x + 3 * Real.sin x

theorem range_of_a (a : ℝ) (h : f (1 - a) + f (1 - a^2) < 0) : 1 < a ∧ a < Real.sqrt 2 := sorry

end NUMINAMATH_GPT_range_of_a_l1990_199045


namespace NUMINAMATH_GPT_average_age_of_cricket_team_l1990_199060

theorem average_age_of_cricket_team 
  (num_members : ℕ)
  (avg_age : ℕ)
  (wicket_keeper_age : ℕ)
  (remaining_avg : ℕ)
  (cond1 : num_members = 11)
  (cond2 : avg_age = 29)
  (cond3 : wicket_keeper_age = avg_age + 3)
  (cond4 : remaining_avg = avg_age - 1) : 
  avg_age = 29 := 
by 
  have h1 : num_members = 11 := cond1
  have h2 : avg_age = 29 := cond2
  have h3 : wicket_keeper_age = avg_age + 3 := cond3
  have h4 : remaining_avg = avg_age - 1 := cond4
  -- proof steps will go here
  sorry

end NUMINAMATH_GPT_average_age_of_cricket_team_l1990_199060


namespace NUMINAMATH_GPT_smallest_integer_n_l1990_199006

theorem smallest_integer_n (n : ℕ) (h₁ : 50 ∣ n^2) (h₂ : 294 ∣ n^3) : n = 210 :=
sorry

end NUMINAMATH_GPT_smallest_integer_n_l1990_199006


namespace NUMINAMATH_GPT_quadratic_discriminant_l1990_199043

noncomputable def discriminant (a b c : ℚ) : ℚ :=
  b^2 - 4 * a * c

theorem quadratic_discriminant :
  discriminant 6 (6 + 1/6) (1/6) = 1225 / 36 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_discriminant_l1990_199043


namespace NUMINAMATH_GPT_find_smaller_number_l1990_199081

theorem find_smaller_number (a b : ℕ) 
  (h1 : a + b = 15) 
  (h2 : 3 * (a - b) = 21) : b = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_smaller_number_l1990_199081


namespace NUMINAMATH_GPT_carnival_total_cost_l1990_199073

def morning_costs (under18_cost over18_cost : ℕ) : ℕ :=
  under18_cost + over18_cost

def afternoon_costs (under18_cost over18_cost : ℕ) : ℕ :=
  under18_cost + 1 + over18_cost + 1

noncomputable def mara_cost : ℕ :=
  let bumper_car_cost := morning_costs 2 0 + afternoon_costs 2 0
  let ferris_wheel_cost := morning_costs 5 5 + 5
  bumper_car_cost + ferris_wheel_cost

noncomputable def riley_cost : ℕ :=
  let space_shuttle_cost := morning_costs 0 5 + afternoon_costs 0 5
  let ferris_wheel_cost := morning_costs 0 6 + (6 + 1)
  space_shuttle_cost + ferris_wheel_cost

theorem carnival_total_cost :
  mara_cost + riley_cost = 61 := by
  sorry

end NUMINAMATH_GPT_carnival_total_cost_l1990_199073


namespace NUMINAMATH_GPT_solve_system_of_equations_l1990_199050

theorem solve_system_of_equations (x y : ℝ) (h1 : x - y = -5) (h2 : 3 * x + 2 * y = 10) : x = 0 ∧ y = 5 := by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1990_199050


namespace NUMINAMATH_GPT_sum_of_digits_10pow97_minus_97_l1990_199085

-- Define a function that computes the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the main statement we want to prove
theorem sum_of_digits_10pow97_minus_97 :
  sum_of_digits (10^97 - 97) = 858 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_10pow97_minus_97_l1990_199085


namespace NUMINAMATH_GPT_max_a_value_l1990_199022

theorem max_a_value (a : ℝ)
  (H : ∀ x : ℝ, (x - 1) * x - (a - 2) * (a + 1) ≥ 1) :
  a ≤ 3 / 2 := by
  sorry

end NUMINAMATH_GPT_max_a_value_l1990_199022


namespace NUMINAMATH_GPT_factorize_expression_l1990_199013

theorem factorize_expression (x y : ℝ) : 
  6 * x^3 * y^2 + 3 * x^2 * y^2 = 3 * x^2 * y^2 * (2 * x + 1) := 
by 
  sorry

end NUMINAMATH_GPT_factorize_expression_l1990_199013


namespace NUMINAMATH_GPT_ion_electronic_structure_l1990_199062

theorem ion_electronic_structure (R M Z n m X : ℤ) (h1 : R + X = M - n) (h2 : M - n = Z - m) (h3 : n > m) : M > Z ∧ Z > R := 
by 
  sorry

end NUMINAMATH_GPT_ion_electronic_structure_l1990_199062


namespace NUMINAMATH_GPT_femaleRainbowTroutCount_l1990_199029

noncomputable def numFemaleRainbowTrout : ℕ := 
  let numSpeckledTrout := 645
  let numFemaleSpeckled := 200
  let numMaleSpeckled := 445
  let numMaleRainbow := 150
  let totalTrout := 1000
  let numRainbowTrout := totalTrout - numSpeckledTrout
  numRainbowTrout - numMaleRainbow

theorem femaleRainbowTroutCount : numFemaleRainbowTrout = 205 := by
  -- Conditions
  let numSpeckledTrout : ℕ := 645
  let numMaleSpeckled := 2 * 200 + 45
  let totalTrout := 645 + 355
  let numRainbowTrout := totalTrout - numSpeckledTrout
  let numFemaleRainbow := numRainbowTrout - 150
  
  -- The proof would proceed here
  sorry

end NUMINAMATH_GPT_femaleRainbowTroutCount_l1990_199029


namespace NUMINAMATH_GPT_largest_c_value_l1990_199079

theorem largest_c_value (c : ℝ) (h : -2 * c^2 + 8 * c - 6 ≥ 0) : c ≤ 3 := 
sorry

end NUMINAMATH_GPT_largest_c_value_l1990_199079


namespace NUMINAMATH_GPT_power_sum_ge_three_l1990_199031

theorem power_sum_ge_three {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 3) :
  a ^ a + b ^ b + c ^ c ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_power_sum_ge_three_l1990_199031


namespace NUMINAMATH_GPT_solve_for_x_l1990_199028

theorem solve_for_x (x : ℝ) (h₀ : x > 0) (h₁ : 1 / 2 * x * (3 * x) = 96) : x = 8 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1990_199028


namespace NUMINAMATH_GPT_islander_real_name_l1990_199021

-- Definition of types of people on the island
inductive IslanderType
| Knight   -- Always tells the truth
| Liar     -- Always lies
| Normal   -- Can lie or tell the truth

-- The possible names of the islander
inductive Name
| Edwin
| Edward

-- Condition: You met the islander who can be Edwin or Edward
def possible_names : List Name := [Name.Edwin, Name.Edward]

-- Condition: The islander said their name is Edward
def islander_statement : Name := Name.Edward

-- Condition: The islander is a Liar (as per the solution interpretation)
def islander_type : IslanderType := IslanderType.Liar

-- The proof problem: Prove the islander's real name is Edwin
theorem islander_real_name : islander_type = IslanderType.Liar ∧ islander_statement = Name.Edward → ∃ n : Name, n = Name.Edwin :=
by
  sorry

end NUMINAMATH_GPT_islander_real_name_l1990_199021


namespace NUMINAMATH_GPT_shaded_region_area_l1990_199080

theorem shaded_region_area (a b : ℕ) (H : a = 2) (K : b = 4) :
  let s := a + b
  let area_square_EFGH := s * s
  let area_smaller_square_FG := a * a
  let area_smaller_square_EF := b * b
  let shaded_area := area_square_EFGH - (area_smaller_square_FG + area_smaller_square_EF)
  shaded_area = 16 := 
by
  sorry

end NUMINAMATH_GPT_shaded_region_area_l1990_199080


namespace NUMINAMATH_GPT_fraction_power_evaluation_l1990_199047

theorem fraction_power_evaluation (x y : ℚ) (h1 : x = 2 / 3) (h2 : y = 3 / 2) : 
  (3 / 4) * x^8 * y^9 = 9 / 8 := 
by
  sorry

end NUMINAMATH_GPT_fraction_power_evaluation_l1990_199047


namespace NUMINAMATH_GPT_imaginary_unit_power_l1990_199027

def i := Complex.I

theorem imaginary_unit_power :
  ∀ a : ℝ, (2 - i + a * i ^ 2011).im = 0 → i ^ 2011 = i :=
by
  intro a
  intro h
  sorry

end NUMINAMATH_GPT_imaginary_unit_power_l1990_199027


namespace NUMINAMATH_GPT_decagon_side_length_in_rectangle_l1990_199095

theorem decagon_side_length_in_rectangle
  (AB CD : ℝ)
  (AE FB : ℝ)
  (s : ℝ)
  (cond1 : AB = 10)
  (cond2 : CD = 15)
  (cond3 : AE = 5)
  (cond4 : FB = 5)
  (regular_decagon : ℝ → Prop)
  (h : regular_decagon s) : 
  s = 5 * (Real.sqrt 2 - 1) :=
by 
  sorry

end NUMINAMATH_GPT_decagon_side_length_in_rectangle_l1990_199095


namespace NUMINAMATH_GPT_total_dolls_48_l1990_199074

def dolls_sister : ℕ := 8

def dolls_hannah : ℕ := 5 * dolls_sister

def total_dolls : ℕ := dolls_hannah + dolls_sister

theorem total_dolls_48 : total_dolls = 48 := 
by
  unfold total_dolls dolls_hannah dolls_sister
  rfl

end NUMINAMATH_GPT_total_dolls_48_l1990_199074


namespace NUMINAMATH_GPT_arithmetic_mean_correct_l1990_199092

noncomputable def arithmetic_mean (n : ℕ) (h : n > 1) : ℝ :=
  let one_minus_one_div_n := 1 - (1 / n : ℝ)
  let rest_ones := (n - 1 : ℕ) • 1
  let total_sum : ℝ := rest_ones + one_minus_one_div_n
  total_sum / n

theorem arithmetic_mean_correct (n : ℕ) (h : n > 1) :
  arithmetic_mean n h = 1 - (1 / (n * n : ℝ)) := sorry

end NUMINAMATH_GPT_arithmetic_mean_correct_l1990_199092


namespace NUMINAMATH_GPT_common_factor_l1990_199034

-- Definition of the polynomial
def polynomial (x y m n : ℝ) : ℝ := 4 * x * (m - n) + 2 * y * (m - n) ^ 2

-- The theorem statement
theorem common_factor (x y m n : ℝ) : ∃ k : ℝ, k * (m - n) = polynomial x y m n :=
sorry

end NUMINAMATH_GPT_common_factor_l1990_199034


namespace NUMINAMATH_GPT_max_stamps_l1990_199046

theorem max_stamps (n friends extra total: ℕ) (h1: friends = 15) (h2: extra = 5) (h3: total < 150) : total ≤ 140 :=
by
  sorry

end NUMINAMATH_GPT_max_stamps_l1990_199046


namespace NUMINAMATH_GPT_chocolates_bought_l1990_199065

theorem chocolates_bought (C S N : ℕ) (h1 : 4 * C = 7 * (S - C)) (h2 : N * C = 77 * S) :
  N = 121 :=
by
  sorry

end NUMINAMATH_GPT_chocolates_bought_l1990_199065


namespace NUMINAMATH_GPT_friends_total_sales_l1990_199068

theorem friends_total_sales :
  (Ryan Jason Zachary : ℕ) →
  (H1 : Ryan = Jason + 50) →
  (H2 : Jason = Zachary + (3 * Zachary / 10)) →
  (H3 : Zachary = 40 * 5) →
  Ryan + Jason + Zachary = 770 :=
by
  sorry

end NUMINAMATH_GPT_friends_total_sales_l1990_199068
